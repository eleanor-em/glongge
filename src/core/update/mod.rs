pub mod collision;
pub mod builtin;

use std::{
    cell::{
        Ref,
        RefMut
    },
    collections::{BTreeMap, BTreeSet},
    sync::{
        Arc,
        mpsc,
        Mutex,
        mpsc::{Receiver, Sender}
    },
    time::{Duration, Instant},
    ops::RangeInclusive
};
use tracing::{warn};
use serde::{
    Serialize,
    de::DeserializeOwned
};
use num_traits::{FromPrimitive, Zero};
use collision::{Collision, CollisionHandler, CollisionNotification, CollisionResponse};
use crate::{core::{
    prelude::*,
    AnySceneObject,
    ObjectId,
    ObjectTypeEnum,
    SceneObjectWithId,
    config::{FIXED_UPDATE_INTERVAL_US, MAX_FIXED_UPDATES},
    coroutine::{Coroutine, CoroutineId, CoroutineResponse, CoroutineState},
    input::InputHandler,
    render::{RenderInfoFull, RenderDataChannel, RenderItem, VertexMap},
    scene::{SceneHandlerInstruction, SceneInstruction, SceneName, SceneDestination},
    vk::AdjustedViewport,
    util::{
        collision::{
            GenericCollider,
            Collider,
            GgInternalCollisionShape
        },
        gg_time::TimeIt,
        linalg::{AxisAlignedExtent, Vec2},
        colour::Colour,
        NonemptyVec,
        linalg::Transform,
    }
}, resource::ResourceHandler};
use crate::core::render::StoredRenderItem;
use crate::core::scene::GuiClosure;
use crate::gui::debug_gui::DebugGui;
use crate::core::util::collision::BoxCollider;
use crate::core::util::{gg_err, gg_iter};
use crate::core::vk::RenderPerfStats;
use crate::resource::sprite::GgInternalSprite;
use crate::shader::{BasicShader, get_shader, Shader};

pub(crate) struct ObjectHandler<ObjectType: ObjectTypeEnum> {
    objects: BTreeMap<ObjectId, AnySceneObject<ObjectType>>,
    parents: BTreeMap<ObjectId, ObjectId>,
    pub(crate) absolute_transforms: BTreeMap<ObjectId, Transform>,
    children: BTreeMap<ObjectId, Vec<SceneObjectWithId<ObjectType>>>,

    collision_handler: CollisionHandler,
}

impl<ObjectType: ObjectTypeEnum> ObjectHandler<ObjectType> {
    fn new() -> Self {
        Self {
            objects: BTreeMap::new(),
            parents: BTreeMap::new(),
            absolute_transforms: BTreeMap::new(),
            children: BTreeMap::new(),
            collision_handler: CollisionHandler::new(),
        }
    }
    fn get_object_type_string(&self, id: ObjectId) -> Result<String> {
        let object_type = self.objects.get(&id)
            .with_context(|| format!("missing object_id from objects: {id:?}"))?
            .borrow().get_type();
        Ok(format!("{object_type:?}"))
    }
    pub(crate) fn get_first_object_id(&self) -> Option<ObjectId> {
        self.objects.first_key_value().map(|o| o.0).copied()
    }
    pub(crate) fn get_next_object_id(&self, source: ObjectId) -> Option<ObjectId> {
        let object_ids = self.objects.keys().copied().collect_vec();
        let i = gg_iter::index_of(&object_ids, &source)?;
        Some(object_ids[(i + 1) % object_ids.len()])
    }
    // fn get_first_object(&self) -> Option<&AnySceneObject<ObjectType>> {
    //     self.objects.first_key_value().map(|o| o.1)
    // }
    pub(crate) fn get_object(&self, id: ObjectId) -> Result<Option<&AnySceneObject<ObjectType>>> {
        if id.is_root() {
            Ok(None)
        } else if let Some(object) = self.objects.get(&id) {
            Ok(Some(object))
        } else {
            bail!("missing object_id from objects: {:?} [{:?}]",
                id, self.get_object_type_string(id)?)
        }
    }

    fn get_parent_id(&self, id: ObjectId) -> Result<ObjectId> {
        if let Some(parent) = self.parents.get(&id) {
            Ok(*parent)
        } else {
            let object_type = self.objects.get(&id)
                .with_context(|| format!("missing object_id from objects: {id:?}"))?
                .borrow().get_type();
            bail!("missing object_id from parents: {:?} [{:?}]", id, object_type)
        }
    }
    pub(crate) fn get_parent_chain(&self, mut id: ObjectId) -> Result<Vec<ObjectId>> {
        let mut parents = Vec::new();
        while !id.is_root() {
            parents.push(id);
            id = self.get_parent_id(id)?;
        }
        Ok(parents)
    }
    pub(crate) fn get_children(&self, id: ObjectId) -> Result<&Vec<SceneObjectWithId<ObjectType>>> {
        if let Some(child) = self.children.get(&id) {
            Ok(child)
        } else {
            let object_type = self.objects.get(&id)
                .with_context(|| format!("missing object_id from objects: {id:?}"))?
                .borrow().get_type();
            bail!("missing object_id from children: {:?} [{:?}]", id, object_type)
        }
    }
    fn get_children_mut(&mut self, id: ObjectId) -> Result<&mut Vec<SceneObjectWithId<ObjectType>>> {
        if let Some(child) = self.children.get_mut(&id) {
            Ok(child)
        } else {
            let object_type = self.objects.get(&id)
                .with_context(|| format!("missing object_id from objects: {id:?}"))?
                .borrow().get_type();
            bail!("missing object_id from children: {:?} [{:?}]", id, object_type)
        }
    }
    pub(crate) fn get_sprite(&self, id: ObjectId) -> Result<Option<Ref<GgInternalSprite>>> {
        Ok(self.get_object(id)?
            .and_then(AnySceneObject::downcast::<GgInternalSprite>)
            .or(self.get_children(id)?.iter()
                .find_map(SceneObjectWithId::downcast::<GgInternalSprite>)))
    }
    pub(crate) fn get_collision_shape(&self, id: ObjectId) -> Result<Option<Ref<CollisionShape>>> {
        Ok(self.get_object(id)?
            .and_then(AnySceneObject::downcast::<CollisionShape>)
               .or(self.get_children(id)?.iter()
                   .find_map(SceneObjectWithId::downcast::<CollisionShape>)))
    }
    pub(crate) fn get_collision_shape_mut(&self, id: ObjectId) -> Result<Option<RefMut<CollisionShape>>> {
        Ok(self.get_object(id)?
            .and_then(AnySceneObject::downcast_mut::<CollisionShape>)
            .or(self.get_children(id)?.iter()
                .find_map(SceneObjectWithId::downcast_mut::<CollisionShape>)))
    }

    fn remove_object(&mut self, remove_id: ObjectId) {
        self.objects.remove(&remove_id);
        if let Err(e) = self.get_parent_id(remove_id)
            .and_then(|parent_id| {
                self.get_children_mut(parent_id)?
                    .retain(|obj| obj.object_id != remove_id);
                self.parents.remove(&remove_id);
                Ok(())
            }) {
            error!("{e:?}");
        }
    }
    fn add_object(&mut self,
                  new_id: ObjectId,
                  new_obj: PendingAddObject<ObjectType>
    ) -> Result<SceneObjectWithId<ObjectType>> {
        if self.children.is_empty() {
            self.children.insert(ObjectId(0), Vec::new());
            self.absolute_transforms.insert(ObjectId(0), Transform::default());
        }
        self.objects.insert(new_id, new_obj.inner.clone());
        self.parents.insert(new_id, new_obj.parent_id);
        self.children.insert(new_id, Vec::new());
        let new_object = SceneObjectWithId::new(new_id, new_obj.inner);
        let children = self.get_children_mut(new_obj.parent_id)?;
        children.push(new_object.clone());

        Ok(new_object)
    }

    fn get_collisions(&mut self) -> Vec<CollisionNotification<ObjectType>> {
        self.update_all_transforms();
        self.collision_handler.get_collisions(&self.absolute_transforms, &self.parents, &self.objects)
    }

    pub(crate) fn get_parent(&self, this_id: ObjectId) -> Result<Option<SceneObjectWithId<ObjectType>>> {
        if this_id.0 == 0 {
            return Ok(None);
        }

        let parent_id = self.get_parent_id(this_id)?;
        if parent_id.0 == 0 {
            Ok(None)
        } else {
            Ok(self.get_object(parent_id)?
                .map(|parent| SceneObjectWithId::new(parent_id, parent.clone())))
        }
    }
    fn get_children_owned(&self, this_id: ObjectId) -> Result<Vec<SceneObjectWithId<ObjectType>>> {
        Ok(self.get_children(this_id)?
            .iter()
            .map(SceneObjectWithId::clone)
            .collect())
    }

    fn update_all_transforms(&mut self) {
        let mut child_stack = Vec::with_capacity(self.objects.len());
        child_stack.push((ObjectId(0), Transform::default()));
        while let Some((parent_id, parent_transform)) = child_stack.pop() {
            self.absolute_transforms.insert(parent_id, parent_transform);
            match self.get_children(parent_id) {
                Ok(children) => {
                    for child in children {
                        child_stack.push((child.object_id, child.transform() * parent_transform));
                    }
                },
                Err(e) => error!("{e:?}"),
            }
        }
    }

    #[cold]
    fn maybe_replace_invalid_shader_id(render_info: &mut RenderInfo) {
        if !render_info.shader_id.is_valid() {
            render_info.shader_id = get_shader(BasicShader::name());
        }
    }

    fn create_render_infos(&mut self, vertex_map: &mut VertexMap, viewport: &AdjustedViewport) -> Vec<RenderInfoFull> {
        self.update_all_transforms();
        for (this_id, mut this) in self.objects.iter()
            .filter_map(|(this_id, this)| {
                RefMut::filter_map(this.borrow_mut(), SceneObject::as_renderable_object)
                    .ok()
                    .map(|this| (this_id, this))
            }) {
            let mut render_ctx = RenderContext::new(
                *this_id,
                &*this as &dyn SceneObject<ObjectType>,
                vertex_map
            );
            this.on_render(&mut render_ctx);
        }
        let mut render_infos = Vec::with_capacity(vertex_map.len());
        let mut start = 0;
        for item in vertex_map.render_items() {
            let mut render_info = if let Some(o) = gg_err::ok_and_log(self.get_object(item.object_id))
                .and_then(|o| {
                    RefMut::filter_map(o.borrow_mut(), |o| {
                        o.as_renderable_object()
                    }).ok()
                }) {
                o.render_info()
            } else {
                error!("object in vertex_map not renderable: {:?} [{:?}]",
                        item.object_id, gg_err::log_unwrap_or("unknown", self.get_object_type_string(item.object_id)));
                continue;
            };
            Self::maybe_replace_invalid_shader_id(&mut render_info);
            let transform = if let Some(t) = self.absolute_transforms.get(&item.object_id) {
                t.translated(-viewport.translation)
            } else {
                error!("missing object_id in transforms: {:?} [{:?}]",
                         item.object_id,
                         gg_err::log_unwrap_or("unknown", self.get_object_type_string(item.object_id)));
                continue;
            };

            let end = start + item.len() as u32;
            render_infos.push(RenderInfoFull {
                vertex_indices: start..end,
                inner: render_info,
                transform: transform.as_f32_lossy(),
                depth: item.render_item.depth,
            });
            start = end;
        }
        render_infos
    }
}

pub(crate) struct UpdateHandler<ObjectType: ObjectTypeEnum> {
    input_handler: Arc<Mutex<InputHandler>>,
    object_handler: ObjectHandler<ObjectType>,

    vertex_map: VertexMap,
    viewport: AdjustedViewport,
    resource_handler: ResourceHandler,
    render_data_channel: Arc<Mutex<RenderDataChannel>>,
    clear_col: Colour,

    coroutines: BTreeMap<ObjectId, BTreeMap<CoroutineId, Coroutine<ObjectType>>>,
    scene_instruction_tx: Sender<SceneInstruction>,
    scene_instruction_rx: Receiver<SceneInstruction>,
    scene_name: SceneName,
    scene_data: Arc<Mutex<Vec<u8>>>,

    debug_gui: DebugGui,
    gui_cmd: Option<Box<GuiClosure>>,

    perf_stats: UpdatePerfStats,
    last_render_perf_stats: Option<RenderPerfStats>,
}

impl<ObjectType: ObjectTypeEnum> UpdateHandler<ObjectType> {
    pub(crate) fn new(
        objects: Vec<AnySceneObject<ObjectType>>,
        input_handler: Arc<Mutex<InputHandler>>,
        resource_handler: ResourceHandler,
        render_data_channel: Arc<Mutex<RenderDataChannel>>,
        scene_name: SceneName,
        scene_data: Arc<Mutex<Vec<u8>>>
    ) -> Result<Self> {
        let (scene_instruction_tx, scene_instruction_rx) = mpsc::channel();
        let mut rv = Self {
            input_handler,
            object_handler: ObjectHandler::new(),
            vertex_map: VertexMap::new(),
            viewport: render_data_channel.clone().lock().unwrap().current_viewport(),
            resource_handler,
            render_data_channel,
            clear_col: Colour::black(),
            coroutines: BTreeMap::new(),
            scene_instruction_tx,
            scene_instruction_rx,
            scene_name,
            scene_data,
            debug_gui: DebugGui::new()?,
            gui_cmd: None,
            perf_stats: UpdatePerfStats::new(),
            last_render_perf_stats: None,
        };

        let input_handler = rv.input_handler.lock().unwrap().clone();
        rv.perf_stats.add_objects.start();
        rv.update_with_added_objects(
            &input_handler,
            objects.into_iter()
                .map(|obj| {
                    PendingAddObject {
                        inner: obj,
                        parent_id: ObjectId(0)
                    }
                })
                .collect()
        )?;
        rv.perf_stats.add_objects.stop();
        rv.update_and_send_render_infos();
        Ok(rv)
    }

    pub(crate) fn consume(mut self) -> Result<SceneHandlerInstruction> {
        let mut delta = Duration::from_secs(0);
        let mut is_running = true;
        let mut fixed_update_us = 0;

        loop {
            if is_running {
                let now = Instant::now();
                self.perf_stats.total_stats.start();
                fixed_update_us += delta.as_micros();
                let fixed_updates = fixed_update_us / FIXED_UPDATE_INTERVAL_US;
                if fixed_updates > 0 {
                    fixed_update_us -= FIXED_UPDATE_INTERVAL_US;
                    if fixed_update_us >= FIXED_UPDATE_INTERVAL_US {
                        warn!("fixed update behind by {:.1} ms",
                            f64::from_u128(fixed_update_us - FIXED_UPDATE_INTERVAL_US)
                                .unwrap_or(f64::INFINITY) / 1000.);
                    }
                    if fixed_update_us >= FIXED_UPDATE_TIMEOUT {
                        warn!("fixed update behind by {:.1} ms, giving up",
                            f64::from_u128(fixed_update_us - FIXED_UPDATE_INTERVAL_US)
                                .unwrap_or(f64::INFINITY) / 1000.);
                        fixed_update_us = 0;
                    }
                }

                let input_handler = self.input_handler.lock().unwrap().clone();
                let (pending_add_objects, pending_remove_objects) =
                    self.call_on_update(&input_handler, fixed_updates)
                        .into_pending();

                self.perf_stats.remove_objects.start();
                self.update_with_removed_objects(pending_remove_objects);
                self.perf_stats.remove_objects.stop();
                self.perf_stats.add_objects.start();
                self.update_with_added_objects(&input_handler, pending_add_objects)?;
                self.perf_stats.add_objects.stop();
                self.update_and_send_render_infos();
                self.input_handler.lock().unwrap().update_step();

                self.perf_stats.total_stats.stop();
                self.debug_gui.console_log.update_perf_stats(self.perf_stats.get());
                self.debug_gui.console_log.render_perf_stats(self.last_render_perf_stats.clone());
                delta = now.elapsed();
            }

            match self.scene_instruction_rx.try_iter().next() {
                Some(SceneInstruction::Stop) => {
                    return Ok(SceneHandlerInstruction::Exit);
                },
                Some(SceneInstruction::Goto(instruction)) => {
                    return Ok(SceneHandlerInstruction::Goto(instruction));
                }
                Some(SceneInstruction::Pause) => {
                    is_running = false;
                },
                Some(SceneInstruction::Resume) => {
                    is_running = true;
                },
                None => {},
            }
        }
    }

    fn update_with_added_objects(&mut self,
                                 input_handler: &InputHandler,
                                 mut pending_add_objects: Vec<PendingAddObject<ObjectType>>
    ) -> Result<()> {
        while !pending_add_objects.is_empty() {
            pending_add_objects.retain(|obj| {
                let rv = obj.parent_id.0 == 0 ||
                    self.object_handler.objects.contains_key(&obj.parent_id);
                if !rv {
                    info!("removed orphaned object: {:?} (parent {:?})",
                          obj.inner.borrow().get_type(), obj.parent_id);
                }
                rv
            });
            if pending_add_objects.is_empty() { break; }
            let pending_add = pending_add_objects.drain(..)
                .map(|obj| (ObjectId::next(), obj))
                .collect_vec();
            let first_new_id = pending_add[0].0.0;
            let last_new_id = pending_add.last().unwrap().0.0;

            let mut object_tracker = ObjectTracker::new(&self.object_handler);
            self.object_handler.collision_handler.add_objects(pending_add.iter());
            self.load_new_objects(&mut object_tracker, pending_add)?;
            self.call_on_ready(&mut object_tracker, input_handler, first_new_id..=last_new_id);

            let (pending_add, pending_remove) = object_tracker.into_pending();
            pending_add_objects = pending_add;
            self.update_with_removed_objects(pending_remove);
        }
        Ok(())
    }

    fn call_on_ready(
        &mut self,
        object_tracker: &mut ObjectTracker<ObjectType>,
        input_handler: &InputHandler,
        new_ids: RangeInclusive<usize>
    ) {
        for this_id in new_ids.into_iter().map(ObjectId) {
            gg_err::log_err(self.object_handler.objects.get_mut(&this_id)
                .with_context(|| format!("tried to call on_ready() for nonexistent added object: {this_id:?}"))
                .cloned()
                .and_then(|this| {
                    let mut ctx = UpdateContext::new(self, input_handler, this_id, object_tracker)?;
                    this.borrow_mut().on_ready(&mut ctx);
                    Ok(())
                }));
        }
    }

    fn load_new_objects<I>(
        &mut self,
        object_tracker: &mut ObjectTracker<ObjectType>,
        pending_add: I
    ) -> Result<()>
    where I: IntoIterator<Item=(ObjectId, PendingAddObject<ObjectType>)>
    {
        for (new_id, new_obj) in pending_add {
            let parent = match self.object_handler.get_parent(new_obj.parent_id) {
                Ok(p) => p,
                Err(e) => {
                    error!("{e:?}");
                    continue;
                }
            };
            let new_obj = match self.object_handler.add_object(new_id, new_obj.clone()) {
                Ok(o) => o,
                Err(e) => {
                    error!("{e:?}");
                    continue;
                }
            };
            let mut object_ctx = ObjectContext {
                collision_handler: &self.object_handler.collision_handler,
                this_id: new_id,
                parent,
                children: Vec::new(),
                object_tracker,
                all_absolute_transforms: &self.object_handler.absolute_transforms,
                all_parents: &self.object_handler.parents,
                all_children: &self.object_handler.children,
            };
            if let Some(new_vertices) = new_obj.inner.borrow_mut()
                .on_load(&mut object_ctx, &mut self.resource_handler)? {
                self.vertex_map.insert(new_id, new_vertices);
            };
            self.debug_gui.object_tree.on_add_object(&self.object_handler, &new_obj);
        }
        self.debug_gui.object_tree.refresh_labels(&self.object_handler);
        Ok(())
    }

    fn update_with_removed_objects(&mut self, pending_remove_objects: BTreeSet<ObjectId>) {
        self.object_handler.collision_handler.remove_objects(&pending_remove_objects);
        for remove_id in &pending_remove_objects {
            self.debug_gui.object_tree.on_remove_object(&self.object_handler, *remove_id);
        }
        for remove_id in pending_remove_objects {
            self.object_handler.remove_object(remove_id);
            self.vertex_map.remove(remove_id);
            self.coroutines.remove(&remove_id);
        }
    }

    fn call_on_update(&mut self,
                      input_handler: &InputHandler,
                      mut fixed_updates: u128
    ) -> ObjectTracker<ObjectType> {
        let mut object_tracker = ObjectTracker {
            last: self.object_handler.objects.clone(),
            pending_add: Vec::new(),
            pending_remove: BTreeSet::new()
        };

        self.perf_stats.on_gui.start();
        self.update_gui(input_handler, &mut object_tracker);
        self.perf_stats.on_gui.stop();

        self.perf_stats.on_update_begin.start();
        self.iter_with_other_map(input_handler, &mut object_tracker,
                                 |mut obj, ctx| {
                                     obj.on_update_begin(ctx);
                                 });
        self.perf_stats.on_update_begin.stop();
        self.perf_stats.coroutines.start();
        self.update_coroutines(input_handler, &mut object_tracker);
        self.perf_stats.coroutines.stop();
        self.perf_stats.on_update.start();
        self.iter_with_other_map(input_handler, &mut object_tracker,
                                 |mut obj, ctx| {
                                     obj.on_update(ctx);
                                 });
        self.perf_stats.on_update.stop();

        self.perf_stats.fixed_update.start();
        for _ in 0..fixed_updates.min(MAX_FIXED_UPDATES) {
            self.iter_with_other_map(input_handler, &mut object_tracker,
                                     |mut obj, ctx| {
                                         obj.on_fixed_update(ctx);
                                     });
            fixed_updates -= 1;
            // Detect collisions after each fixed update: important to prevent glitching through walls etc.
            self.perf_stats.fixed_update.pause();
            self.handle_collisions(input_handler, &mut object_tracker);
            self.perf_stats.fixed_update.unpause();
        }
        self.perf_stats.fixed_update.stop();

        self.perf_stats.on_update_end.start();
        self.iter_with_other_map(input_handler, &mut object_tracker,
                                 |mut obj, ctx| {
                                     obj.on_update_end(ctx);
                                 });
        self.perf_stats.on_update_end.stop();
        object_tracker
    }

    fn update_gui(&mut self, input_handler: &InputHandler, object_tracker: &mut ObjectTracker<ObjectType>) {
        if input_handler.pressed(KeyCode::Backquote) {
            self.debug_gui.toggle();
        }

        if self.debug_gui.enabled {
            let all_tags = self.object_handler.collision_handler.all_tags();
            self.debug_gui.clear_mouseovers(&self.object_handler);
            let mouse_pos = self.viewport.top_left() + input_handler.screen_mouse_pos();
            let maybe_collisions = match UpdateContext::new(
                self, input_handler, ObjectId::root(), object_tracker
            ) {
                Ok(ctx) => ctx.object.test_collision_point(mouse_pos, all_tags),
                Err(e) => {
                    error!("{e:?}");
                    None
                },
            };
            if let Some(collisions) = maybe_collisions {
                self.debug_gui.on_mouseovers(&self.object_handler, collisions);
            }
            let gui_objects = self.object_handler.objects.iter()
                .filter(|(_, obj)| obj.borrow().as_gui_object().is_some())
                .map(|(id, obj)| (*id, obj.clone()))
                .collect_vec();
            let gui_cmds = gui_objects.into_iter()
                .filter_map(|(id, obj)| {
                    match UpdateContext::new(
                        self, input_handler, id, object_tracker
                    ) {
                        Ok(ctx) => Some((id, obj.borrow().as_gui_object().unwrap().on_gui(&ctx))),
                        Err(e) => {
                            error!("{e:?}");
                            None
                        }
                    }
                })
                .collect();
            self.gui_cmd = Some(self.debug_gui.build(input_handler, &self.object_handler, gui_cmds));
        }
    }

    fn handle_collisions(&mut self, input_handler: &InputHandler, object_tracker: &mut ObjectTracker<ObjectType>) {
        self.perf_stats.detect_collision.start();
        let collisions = self.object_handler.get_collisions();
        self.perf_stats.detect_collision.stop();
        self.perf_stats.on_collision.start();
        let mut done_with_collisions = BTreeSet::new();
        for CollisionNotification { this, other, mtv } in collisions {
            if !done_with_collisions.contains(&this.object_id) {
                match UpdateContext::new(
                    self,
                    input_handler,
                    this.object_id,
                    object_tracker
                ) {
                    Ok(mut ctx) => match this.inner.borrow_mut().on_collision(&mut ctx, other, mtv) {
                        CollisionResponse::Continue => {},
                        CollisionResponse::Done => { done_with_collisions.insert(this.object_id); },
                    },
                    Err(e) => error!("{e:?}"),
                }
            }
        }
        self.perf_stats.on_collision.stop();
    }

    fn update_coroutines(&mut self,
                         input_handler: &InputHandler,
                         object_tracker: &mut ObjectTracker<ObjectType>
    ) {
        for (this_id, this) in self.object_handler.objects.clone() {
            let this = SceneObjectWithId::new(this_id, this.clone());
            let last_coroutines = self.coroutines.remove(&this_id).unwrap_or_default();
            for (id, coroutine) in last_coroutines {
                match UpdateContext::new(
                    self,
                    input_handler,
                    this_id,
                    object_tracker
                ) {
                    Ok(mut ctx) => {
                        if let Some(coroutine) = coroutine.resume(this.clone(), &mut ctx) {
                            ctx.scene.coroutines.insert(id, coroutine);
                        }
                    },
                    Err(e) => error!("{e:?}"),
                }
            }
        }
    }
    fn iter_with_other_map<F>(&mut self,
                              input_handler: &InputHandler,
                              object_tracker: &mut ObjectTracker<ObjectType>,
                              call_obj_event: F)
    where F: Fn(RefMut<dyn SceneObject<ObjectType>>, &mut UpdateContext<ObjectType>) {
        for (this_id, this) in self.object_handler.objects.clone() {
            let this = SceneObjectWithId::new(this_id, this.clone());
            match UpdateContext::new(
                self,
                input_handler,
                this_id,
                object_tracker
            ) {
                Ok(mut ctx) => call_obj_event(this.inner.borrow_mut(), &mut ctx),
                Err(e) => error!("{e:?}"),
            }
        }
    }
    fn update_and_send_render_infos(&mut self) {
        self.perf_stats.render_infos.start();
        let render_infos = self.object_handler.create_render_infos(
            &mut self.vertex_map, &self.viewport);
        self.send_render_infos(render_infos);
        self.perf_stats.render_infos.stop();
    }

    fn send_render_infos(&mut self, render_infos: Vec<RenderInfoFull>) {
        let maybe_vertices = if self.vertex_map.consume_vertices_changed() {
            Some(self.vertex_map.render_items()
                .flat_map(StoredRenderItem::vertices)
                .copied()
                .collect_vec())
        } else {
            None
        };
        let mut render_data_channel = self.render_data_channel.lock().unwrap();
        self.last_render_perf_stats = render_data_channel.last_render_stats.clone();
        render_data_channel.gui_commands = self.gui_cmd.take().into_iter().collect_vec();
        if let Some(vertices) = maybe_vertices {
            render_data_channel.vertices = vertices;
        }
        render_data_channel.render_infos = render_infos;
        render_data_channel.set_clear_col(self.clear_col);
        self.viewport = render_data_channel.current_viewport()
            .translated(self.viewport.translation);
    }

}

#[derive(Clone)]
pub(crate) struct UpdatePerfStats {
    total_stats: TimeIt,
    on_gui: TimeIt,
    on_update_begin: TimeIt,
    coroutines: TimeIt,
    on_update: TimeIt,
    on_update_end: TimeIt,
    fixed_update: TimeIt,
    detect_collision: TimeIt,
    on_collision: TimeIt,
    remove_objects: TimeIt,
    add_objects: TimeIt,
    render_infos: TimeIt,
    last_perf_stats: Option<Box<UpdatePerfStats>>,
    last_report: Instant,
}

impl UpdatePerfStats {
    fn new() -> Self {
        Self {
            total_stats: TimeIt::new("total"),
            on_gui: TimeIt::new("on_gui"),
            on_update_begin: TimeIt::new("on_update_begin"),
            coroutines: TimeIt::new("coroutines"),
            on_update: TimeIt::new("on_update"),
            on_update_end: TimeIt::new("on_update_end"),
            fixed_update: TimeIt::new("fixed_update"),
            detect_collision: TimeIt::new("detect collisions"),
            on_collision: TimeIt::new("on_collision"),
            remove_objects: TimeIt::new("remove objects"),
            add_objects: TimeIt::new("add objects"),
            render_infos: TimeIt::new("render_infos"),
            last_perf_stats: None,
            last_report: Instant::now(),
        }
    }

    fn get(&mut self) -> Option<Self> {
        if self.last_report.elapsed().as_secs() >= 2 {
            self.last_perf_stats = Some(Box::new(Self {
                total_stats: self.total_stats.report_take(),
                on_gui: self.on_gui.report_take(),
                on_update_begin: self.on_update_begin.report_take(),
                coroutines: self.coroutines.report_take(),
                on_update: self.on_update.report_take(),
                on_update_end: self.on_update_end.report_take(),
                fixed_update: self.fixed_update.report_take(),
                detect_collision: self.detect_collision.report_take(),
                on_collision: self.on_collision.report_take(),
                remove_objects: self.remove_objects.report_take(),
                add_objects: self.add_objects.report_take(),
                render_infos: self.render_infos.report_take(),
                last_perf_stats: None,
                last_report: Instant::now(),
            }));
            self.last_report = Instant::now();
        }
        self.last_perf_stats.clone().map(|s| *s)
    }

    pub(crate) fn as_tuples_ms(&self) -> Vec<(String, f64, f64)> {
        vec![
            self.total_stats.as_tuple_ms(),
            self.on_gui.as_tuple_ms(),
            self.on_update_begin.as_tuple_ms(),
            self.coroutines.as_tuple_ms(),
            self.on_update.as_tuple_ms(),
            self.on_update_end.as_tuple_ms(),
            self.fixed_update.as_tuple_ms(),
            self.detect_collision.as_tuple_ms(),
            self.remove_objects.as_tuple_ms(),
            self.add_objects.as_tuple_ms(),
            self.render_infos.as_tuple_ms(),
        ]
    }
}

pub struct UpdateContext<'a, ObjectType: ObjectTypeEnum> {
    input: &'a InputHandler,
    scene: SceneContext<'a, ObjectType>,
    object: ObjectContext<'a, ObjectType>,
    viewport: ViewportContext<'a>,
}

impl<'a, ObjectType: ObjectTypeEnum> UpdateContext<'a, ObjectType> {
    fn new(
        caller: &'a mut UpdateHandler<ObjectType>,
        input_handler: &'a InputHandler,
        this_id: ObjectId,
        object_tracker: &'a mut ObjectTracker<ObjectType>
    ) -> Result<Self> {
        let parent = caller.object_handler.get_parent(this_id)?;
        let children = caller.object_handler.get_children_owned(this_id)?;
        Ok(Self {
            input: input_handler,
            scene: SceneContext {
                scene_instruction_tx: caller.scene_instruction_tx.clone(),
                scene_name: caller.scene_name,
                scene_data: caller.scene_data.clone(),
                coroutines: caller.coroutines.entry(this_id).or_default(),
                pending_removed_coroutines: BTreeSet::new(),
                debug_enabled: &mut caller.debug_gui.enabled,
            },
            object: ObjectContext {
                collision_handler: &caller.object_handler.collision_handler,
                this_id,
                parent,
                children,
                object_tracker,
                all_absolute_transforms: &caller.object_handler.absolute_transforms,
                all_parents: &caller.object_handler.parents,
                all_children: &caller.object_handler.children,
            },
            viewport: ViewportContext {
                viewport: &mut caller.viewport,
                clear_col: &mut caller.clear_col,
            },
        })
    }

    pub fn object_mut(&mut self) -> &mut ObjectContext<'a, ObjectType> { &mut self.object }
    pub fn object(&self) -> &ObjectContext<'a, ObjectType> { &self.object }
    pub fn scene_mut(&mut self) -> &mut SceneContext<'a, ObjectType> { &mut self.scene }
    pub fn scene(&self) -> &SceneContext<'a, ObjectType> { &self.scene }
    pub fn viewport_mut(&mut self) -> &mut ViewportContext<'a> { &mut self.viewport }
    pub fn viewport(&self) -> &ViewportContext<'a> { &self.viewport }
    pub fn input(&self) -> &InputHandler { self.input }
}

impl<ObjectType: ObjectTypeEnum> Drop for UpdateContext<'_, ObjectType> {
    fn drop(&mut self) {
        for id in &self.scene.pending_removed_coroutines {
            self.scene.coroutines.remove(id);
        }
        self.scene.pending_removed_coroutines.clear();
        check!(self.scene.pending_removed_coroutines.is_empty());
    }
}


pub struct SceneData<T>
where
    T: Default + Serialize + DeserializeOwned
{
    raw: Arc<Mutex<Vec<u8>>>,
    deserialized: T,
    modified: bool,
}

impl<T> SceneData<T>
where
    T: Default + Serialize + DeserializeOwned
{
    fn new(raw: Arc<Mutex<Vec<u8>>>) -> Result<Option<Self>> {
        let deserialized = {
            let raw = raw.try_lock().expect("scene_data locked?");
            if raw.is_empty() {
                return Ok(None);
            }
            bincode::deserialize::<T>(&raw)?
        };
        Ok(Some(Self {
            raw,
            deserialized,
            modified: false,
        }))
    }

    pub fn reset(&mut self) {
        *self.write() = T::default();
    }

    pub fn read(&self) -> &T { &self.deserialized }
    pub fn write(&mut self) -> &mut T {
        self.modified = true;
        &mut self.deserialized
    }
}

impl<T> Drop for SceneData<T>
where
    T: Default + Serialize + DeserializeOwned
{
    fn drop(&mut self) {
        if self.modified {
            *self.raw.try_lock().expect("scene_data locked?") =
                bincode::serialize(&self.deserialized)
                    .expect("failed to serialize scene data");
        }
    }
}

pub struct SceneContext<'a, ObjectType: ObjectTypeEnum> {
    scene_instruction_tx: Sender<SceneInstruction>,
    scene_name: SceneName,
    scene_data: Arc<Mutex<Vec<u8>>>,
    coroutines: &'a mut BTreeMap<CoroutineId, Coroutine<ObjectType>>,
    pending_removed_coroutines: BTreeSet<CoroutineId>,
    debug_enabled: &'a mut bool,
}

impl<'a, ObjectType: ObjectTypeEnum> SceneContext<'a, ObjectType> {
    pub fn stop(&self) {
        self.scene_instruction_tx.send(SceneInstruction::Stop).unwrap();
    }
    pub fn goto(&self, instruction: SceneDestination) {
        self.scene_instruction_tx.send(SceneInstruction::Goto(instruction)).unwrap();
    }
    pub fn name(&self) -> SceneName { self.scene_name }
    pub fn data<T>(&mut self) -> Option<SceneData<T>>
    where
        T: Default + Serialize + DeserializeOwned
    {
        SceneData::new(self.scene_data.clone())
            .expect("failed to ser/de scene_data, do the types match?")
    }

    pub fn start_coroutine<F>(&mut self, func: F) -> CoroutineId
    where
        F: FnMut(SceneObjectWithId<ObjectType>, &mut UpdateContext<ObjectType>, CoroutineState) -> CoroutineResponse + 'static
    {
        let id = CoroutineId::next();
        self.coroutines.insert(id, Coroutine::new(func));
        id
    }
    pub fn start_coroutine_after<F>(&mut self, mut func: F, duration: Duration) -> CoroutineId
    where
        F: FnMut(SceneObjectWithId<ObjectType>, &mut UpdateContext<ObjectType>, CoroutineState) -> CoroutineResponse + 'static
    {
        self.start_coroutine(move |this, ctx, action| {
            match action {
                CoroutineState::Starting => CoroutineResponse::Wait(duration),
                _ => func(this, ctx, action)
            }
        })
    }
    pub fn maybe_cancel_coroutine(&mut self, id: &mut Option<CoroutineId>) {
        if let Some(id) = id.take() {
            self.cancel_coroutine(id);
        }
    }
    pub fn cancel_coroutine(&mut self, id: CoroutineId) {
        self.pending_removed_coroutines.insert(id);
    }

    pub fn is_debug_enabled(&self) -> bool { *self.debug_enabled }
    pub fn set_debug_enabled(&mut self, value: bool) { *self.debug_enabled = value; }
}

#[derive(Clone)]
pub(crate) struct PendingAddObject<ObjectType: ObjectTypeEnum> {
    inner: AnySceneObject<ObjectType>,
    parent_id: ObjectId,
}

struct ObjectTracker<ObjectType: ObjectTypeEnum> {
    last: BTreeMap<ObjectId, AnySceneObject<ObjectType>>,
    pending_add: Vec<PendingAddObject<ObjectType>>,
    pending_remove: BTreeSet<ObjectId>,
}

impl<ObjectType: ObjectTypeEnum> ObjectTracker<ObjectType> {
    fn new(object_handler: &ObjectHandler<ObjectType>) -> Self {
        Self {
            last: object_handler.objects.clone(),
            pending_add: Vec::new(),
            pending_remove: BTreeSet::new(),
        }
    }

    fn get(&self, object_id: ObjectId) -> Option<&AnySceneObject<ObjectType>> {
        self.last.get(&object_id)
    }
}

impl<ObjectType: ObjectTypeEnum> ObjectTracker<ObjectType> {
    fn into_pending(self) -> (Vec<PendingAddObject<ObjectType>>, BTreeSet<ObjectId>) {
        (self.pending_add, self.pending_remove)
    }
}

pub struct ObjectContext<'a, ObjectType: ObjectTypeEnum> {
    collision_handler: &'a CollisionHandler,
    this_id: ObjectId,
    parent: Option<SceneObjectWithId<ObjectType>>,
    children: Vec<SceneObjectWithId<ObjectType>>,
    object_tracker: &'a mut ObjectTracker<ObjectType>,
    all_absolute_transforms: &'a BTreeMap<ObjectId, Transform>,
    all_parents: &'a BTreeMap<ObjectId, ObjectId>,
    all_children: &'a BTreeMap<ObjectId, Vec<SceneObjectWithId<ObjectType>>>,
}

impl<'a, ObjectType: ObjectTypeEnum> ObjectContext<'a, ObjectType> {
    pub fn parent(&self) -> Option<&SceneObjectWithId<ObjectType>> { self.parent.as_ref() }
    pub fn children(&self) -> Vec<SceneObjectWithId<ObjectType>> {
        self.children.iter()
            .map(SceneObjectWithId::clone)
            .collect()
    }
    fn others_inner(&self) -> impl Iterator<Item=(ObjectId, &AnySceneObject<ObjectType>)> {
        self.object_tracker.last.iter()
            .filter(|(object_id, _)| !self.object_tracker.pending_remove.contains(object_id))
            .filter(|(object_id, _)| self.this_id != **object_id)
            .map(|(object_id, obj)| (*object_id, obj))
    }
    pub fn others(&self) -> Vec<SceneObjectWithId<ObjectType>> {
        self.others_inner()
            .map(|(object_id, obj)| SceneObjectWithId::new(object_id, obj.clone()))
            .collect()
    }
    pub fn others_as_ref<T: SceneObject<ObjectType>>(&self) -> Vec<Ref<T>> {
        self.others_inner()
            .filter_map(|(_, obj)| obj.downcast())
            .collect()
    }
    pub fn others_as_mut<T: SceneObject<ObjectType>>(&self) -> Vec<RefMut<T>> {
        self.others_inner()
            .filter_map(|(_, obj)| obj.downcast_mut())
            .collect()
    }
    pub fn first_other_as_mut<T: SceneObject<ObjectType>>(&self) -> Option<RefMut<T>> {
        self.others_inner()
            .find_map(|(_, obj)| obj.downcast_mut())
    }

    pub fn absolute_transform(&self) -> Transform {
        self.all_absolute_transforms.get(&self.this_id)
            .copied()
            .unwrap_or_else(|| {
                error!("missing object_id in absolute_transforms: this={:?}", self.this_id);
                Transform::default()
            })
    }
    pub fn absolute_transform_of(&self, other: &SceneObjectWithId<ObjectType>) -> Transform {
        // Should not be possible to get an invalid object_id here if called from public.
        self.all_absolute_transforms.get(&other.object_id)
            .copied()
            .unwrap_or_else(|| {
                error!("missing object_id in absolute_transforms: {:?}", other.object_id);
                Transform::default()
            })
    }
    pub fn rect(&self) -> Rect {
        self.collider()
            .unwrap_or_default()
            .transformed(&self.absolute_transform())
            .as_rect()
    }
    pub fn rect_of(&self, other: &SceneObjectWithId<ObjectType>) -> Rect {
        self.collider_of(other)
            .unwrap_or_default()
            .transformed(&self.absolute_transform_of(other))
            .as_rect()
    }
    pub fn extent(&self) -> Vec2 {
        self.collider()
            .unwrap_or_default()
            .aa_extent()
    }
    pub fn extent_of(&self, other: &SceneObjectWithId<ObjectType>) -> Vec2 {
        // Should not be possible to get an invalid object_id here if called from public.
        self.collider_of(other)
            .unwrap_or_default()
            .aa_extent()
    }
    pub fn collider(&self) -> Option<GenericCollider> {
        gg_err::ok_and_log(self.collider_of_inner(self.this_id))
    }
    pub fn collider_of(&self, other: &SceneObjectWithId<ObjectType>) -> Option<GenericCollider> {
        gg_err::ok_and_log(self.collider_of_inner(other.object_id))
    }
    fn collider_of_inner(&self, object_id: ObjectId) -> Result<Option<GenericCollider>> {
        let children = if object_id == self.this_id {
            &self.children
        } else {
            self.all_children.get(&object_id)
                .with_context(|| format!("missing object_id in children: {object_id:?}"))?
        };
        if let Some((collision_shape_id, collision_shape)) = children.iter()
            .find_map(|obj| {
                obj.downcast::<GgInternalCollisionShape>()
                    .map(|inner| (obj.object_id, inner))
            }) {
            if let Some(t) = self.all_absolute_transforms.get(&collision_shape_id) {
                Ok(Some(collision_shape.collider().transformed(t)))
            } else {
                bail!("missing object_id in absolute_transforms: {collision_shape_id:?}");
            }
        } else {
            Ok(None)
        }
    }

    pub fn add_vec(&mut self, objects: Vec<AnySceneObject<ObjectType>>) {
        let pending_add = &mut self.object_tracker.pending_add;
        pending_add.extend(objects.into_iter().map(|inner| {
            PendingAddObject {
                inner: inner.clone(),
                parent_id: self.this_id,
            }
        }));
    }
    pub fn add_sibling(&mut self, object: AnySceneObject<ObjectType>) {
        self.object_tracker.pending_add.push(PendingAddObject {
            inner: object,
            parent_id: self.parent().map_or(ObjectId(0), |obj| {
                obj.object_id
            })
        });
    }
    pub fn add_child(&mut self, object: AnySceneObject<ObjectType>) {
        self.object_tracker.pending_add.push(PendingAddObject {
            inner: object,
            parent_id: self.this_id,
        });
    }
    pub fn remove(&mut self, obj: &SceneObjectWithId<ObjectType>) {
        self.object_tracker.pending_remove.insert(obj.object_id);
        for child in &self.children {
            self.object_tracker.pending_remove.insert(child.object_id);
        }
    }
    pub fn remove_this(&mut self) {
        self.object_tracker.pending_remove.insert(self.this_id);
        self.remove_children();
    }
    pub fn remove_children(&mut self) {
        for child in &self.children {
            self.object_tracker.pending_remove.insert(child.object_id);
        }
    }
    pub fn test_collision_point(&self,
                                point: Vec2,
                                listening_tags: Vec<&'static str>
    ) -> Option<NonemptyVec<Collision<ObjectType>>> {
        self.test_collision_with(
            &BoxCollider::from_top_left(point - Vec2::one() / 2, Vec2::one()).into_generic(),
            listening_tags
        )
    }
    pub fn test_collision(&self,
                          listening_tags: Vec<&'static str>
    ) -> Option<NonemptyVec<Collision<ObjectType>>> {
        self.collider()
            .and_then(|collider| {
                self.test_collision_with(&collider, listening_tags)
            })
    }
    pub fn test_collision_along(&self,
                                axis: Vec2,
                                distance: f64,
                                listening_tags: Vec<&'static str>,
    ) -> Option<NonemptyVec<Collision<ObjectType>>> {
        self.collider()
            .and_then(|collider| {
                self.test_collision_with(&collider.translated(distance * axis), listening_tags)
            })
            .and_then(|vec| {
                NonemptyVec::try_from_iter(vec
                    .into_iter()
                    .filter(|coll| !coll.mtv.dot(axis).is_zero()))
            })
    }

    pub fn test_collision_with(
        &self,
        collider: &GenericCollider,
        listening_tags: Vec<&'static str>
    ) -> Option<NonemptyVec<Collision<ObjectType>>> {
        let mut rv = Vec::new();
        for tag in listening_tags {
            match self.collision_handler.get_object_ids_by_emitting_tag(tag) {
                Ok(colliding_ids) => {
                    rv.extend(colliding_ids.iter()
                        .filter_map(|other_id| {
                            gg_err::ok_and_log(self.test_collision_inner(collider, *other_id))
                        }));
                },
                Err(e) => error!("{e:?}"),
            }
        }
        NonemptyVec::try_from_vec(rv)
    }

    fn test_collision_inner(&self,
                            collider: &GenericCollider,
                            other_id: ObjectId
    ) -> Result<Option<Collision<ObjectType>>> {
        let other = self.object_tracker.get(other_id)
            .with_context(|| format!("missing object_id in objects: {other_id:?}"))?;
        let other_collider = other.checked_downcast::<GgInternalCollisionShape>()
            .collider()
            .transformed(
                self.all_absolute_transforms.get(&other_id)
                    .with_context(|| format!("missing object_id in absolute_transforms: {other_id:?}"))?
            );
        if let Some(mtv) = collider.collides_with(&other_collider) {
            let other = self.lookup_parent(other_id)
                .with_context(|| format!("orphaned GgInternalCollisionShape: {other_id:?}"))?;
            Ok(Some(Collision { other, mtv }))
        } else {
            Ok(None)
        }
    }
    fn lookup_parent(&self, object_id: ObjectId) -> Option<SceneObjectWithId<ObjectType>> {
        let parent_id = self.all_parents.get(&object_id)?;
        if parent_id.0 == 0 {
            None
        } else {
            self.object_tracker.get(*parent_id)
                .map(|parent| SceneObjectWithId::new(*parent_id, parent.clone()))
                .or_else(|| {
                    error!("missing object_id in parents: {parent_id:?}");
                    None
                })
        }
    }
}

pub struct ViewportContext<'a> {
    viewport: &'a mut AdjustedViewport,
    clear_col: &'a mut Colour,
}

impl<'a> ViewportContext<'a> {
    pub fn clamp_to_left(&mut self, min: Option<f64>, max: Option<f64>) {
        if let Some(min) = min {
            if self.viewport.left() < min {
                self.translate((min - self.viewport.left()) * Vec2::right());
            }
        }
        if let Some(max) = max {
            if self.viewport.left() > max {
                self.translate((self.viewport.left() - max) * Vec2::left());
            }
        }
    }
    pub fn clamp_to_right(&mut self, min: Option<f64>, max: Option<f64>) {
        if let Some(min) = min {
            if self.viewport.right() < min {
                self.translate((min - self.viewport.right()) * Vec2::right());
            }
        }
        if let Some(max) = max {
            if self.viewport.right() > max {
                self.translate((self.viewport.right() - max) * Vec2::left());
            }
        }
    }
    pub fn centre_at(&mut self, centre: Vec2) -> &mut Self {
        self.translate(centre - self.viewport.centre());
        self
    }
    pub fn translate(&mut self, delta: Vec2) -> &mut Self {
        self.viewport.translation += delta;
        self
    }
    pub fn clear_col(&mut self) -> &mut Colour { self.clear_col }
    pub fn inner(&self) -> AdjustedViewport { self.viewport.clone() }
}

impl AxisAlignedExtent for ViewportContext<'_> {
    fn aa_extent(&self) -> Vec2 {
        self.viewport.aa_extent()
    }

    fn centre(&self) -> Vec2 {
        self.viewport.centre()
    }
}

pub struct RenderContext<'a> {
    pub(crate) this_id: ObjectId,
    this_type: String,
    vertex_map: &'a mut VertexMap,
}

impl<'a> RenderContext<'a> {
    pub(crate) fn new<ObjectType: ObjectTypeEnum>(
        this_id: ObjectId,
        obj: &dyn SceneObject<ObjectType>,
        vertex_map: &'a mut VertexMap
    ) -> Self {
        Self {
            this_id,
            this_type: format!("{:?}", obj.get_type()),
            vertex_map,
        }
    }

    pub fn update_render_item(&mut self, new_render_item: &RenderItem) {
        self.remove_render_item();
        self.vertex_map.insert(self.this_id, new_render_item.clone());
    }
    pub fn insert_render_item(&mut self, new_render_item: &RenderItem) {
        if let Some(existing) = self.vertex_map.remove(self.this_id) {
            self.vertex_map.insert(self.this_id, existing.concat(new_render_item.clone()));
        } else {
            self.vertex_map.insert(self.this_id, new_render_item.clone());
        }
    }
    pub fn remove_render_item(&mut self) {
        check_is_some!(self.vertex_map.remove(self.this_id),
                       format!("removed nonexistent vertices: {:?} [{}]", self.this_id, self.this_type));
    }
}
