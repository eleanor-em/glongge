pub mod collision;

use std::{
    rc::Rc,
    collections::{BTreeMap, BTreeSet},
    cell::{RefCell, RefMut},
    sync::{
        Arc,
        mpsc,
        Mutex,
        mpsc::{Receiver, Sender}
    },
    time::{Duration, Instant}
};
use std::ops::RangeInclusive;
use tracing::{warn};
use serde::{
    Serialize,
    de::DeserializeOwned
};
use num_traits::{FromPrimitive, Zero};
use collision::{Collision, CollisionHandler, CollisionNotification, CollisionResponse};
use crate::{
    core::{
        prelude::*,
        AnySceneObject,
        ObjectId,
        ObjectTypeEnum,
        SceneObjectWithId,
        config::{FIXED_UPDATE_INTERVAL_US, MAX_FIXED_UPDATES},
        coroutine::{Coroutine, CoroutineId, CoroutineResponse, CoroutineState},
        input::InputHandler,
        render::{RenderInfoFull, RenderInfoReceiver, RenderItem, VertexMap},
        scene::{SceneHandlerInstruction, SceneInstruction, SceneName, SceneDestination},
        vk::AdjustedViewport,
        util::{
            collision::GenericCollider,
            gg_time::TimeIt,
            linalg::{AxisAlignedExtent, Vec2},
            colour::Colour,
            collision::Collider,
            NonemptyVec,
            linalg::Transform
        },
        BorrowedSceneObjectWithId
    },
    resource::ResourceHandler,
};

struct ObjectHandler<ObjectType: ObjectTypeEnum> {
    objects: BTreeMap<ObjectId, Rc<RefCell<AnySceneObject<ObjectType>>>>,
    parents: BTreeMap<ObjectId, ObjectId>,
    absolute_transforms: BTreeMap<ObjectId, Transform>,
    children: BTreeMap<ObjectId, Vec<SceneObjectWithId<ObjectType>>>,

    collision_handler: CollisionHandler,

    render_infos: BTreeMap<ObjectId, RenderInfoFull>,
}

impl<ObjectType: ObjectTypeEnum> ObjectHandler<ObjectType> {
    fn new() -> Self {
        Self {
            objects: BTreeMap::new(),
            parents: BTreeMap::new(),
            absolute_transforms: BTreeMap::new(),
            children: BTreeMap::new(),
            collision_handler: CollisionHandler::new(),
            render_infos: BTreeMap::new()
        }
    }

    fn remove_object(&mut self, remove_id: ObjectId) {
        self.render_infos.remove(&remove_id);
        self.objects.remove(&remove_id);
        let parent = self.parents.get(&remove_id)
            .unwrap_or_else(|| panic!("missing object_id in parents: {remove_id:?}"));
        self.children.get_mut(parent)
            .unwrap_or_else(|| panic!("missing object_id in children: {remove_id:?}"))
            .retain(|obj| obj.object_id != remove_id);
        self.parents.remove(&remove_id);
    }
    fn add_object(&mut self, new_id: ObjectId, new_obj: PendingAddObject<ObjectType>) {
        if self.children.is_empty() {
            self.children.insert(ObjectId(0), Vec::new());
            self.absolute_transforms.insert(ObjectId(0), Transform::default());
        }
        self.objects.insert(new_id, new_obj.inner.clone());
        self.parents.insert(new_id, new_obj.parent_id);
        self.children.insert(new_id, Vec::new());
        self.children.get_mut(&new_obj.parent_id)
            .unwrap_or_else(|| panic!("missing object_id in children: {:?}", new_obj.parent_id))
            .push(SceneObjectWithId::new(new_id, new_obj.inner));
    }

    fn get_collisions(&mut self) -> Vec<CollisionNotification<ObjectType>> {
        self.update_all_transforms();
        self.collision_handler.get_collisions(&self.absolute_transforms, &self.parents, &self.objects)
    }

    fn get_parent(&self, this_id: ObjectId) -> Option<BorrowedSceneObjectWithId<ObjectType>> {
        if this_id.0 == 0 {
            return None;
        }

        let parent_id = self.parents.get(&this_id)
            .unwrap_or_else(|| panic!("missing object_id in parents: {this_id:?}"));
        if parent_id.0 == 0 {
            None
        } else {
            let parent = self.objects.get(parent_id)
                .unwrap_or_else(|| panic!("the only missing parent should be the root node: {parent_id:?}"));
            Some(BorrowedSceneObjectWithId::new(*parent_id, parent))
        }
    }
    fn get_children(&self, this_id: ObjectId) -> Vec<SceneObjectWithId<ObjectType>> {
        self.children
            .get(&this_id)
            .unwrap_or_else(|| panic!("missing object_id in children: {this_id:?}"))
            .iter()
            .map(SceneObjectWithId::clone)
            .collect()
    }

    fn update_all_transforms(&mut self) {
        self.update_transforms(ObjectId(0), Transform::default());
    }
    fn update_transforms(
        &mut self,
        parent_id: ObjectId,
        parent_transform: Transform,
    ) {
        let mut child_stack = Vec::with_capacity(self.objects.len());
        child_stack.push((parent_id, parent_transform));
        while let Some((parent_id, parent_transform)) = child_stack.pop() {
            self.absolute_transforms.insert(parent_id, parent_transform);
            for child in self.children.get(&parent_id)
                .unwrap_or_else(|| panic!("missing object_id in children: {parent_id:?}")) {
                let child_id = child.object_id;
                let child_transform = child.transform() * parent_transform;
                child_stack.push((child_id, child_transform));
            }
        }
    }

    fn update_render_infos(&mut self, vertex_map: &VertexMap, viewport: &AdjustedViewport) {
        self.update_all_transforms();
        for (object_id, obj) in &self.objects {
            if let Some(obj) = obj.borrow().as_renderable_object() {
                let (indices, _) = vertex_map.get(*object_id)
                    .unwrap_or_else(|| panic!("missing object_id in vertex_map: {object_id:?}"));
                let transform = self.absolute_transforms.get(object_id)
                    .unwrap_or_else(|| panic!("missing object_id in transforms: {object_id:?}"))
                    .translated(-viewport.translation);
                self.render_infos.insert(*object_id, RenderInfoFull {
                    vertex_indices: indices.clone(),
                    inner: obj.render_info(),
                    transform,
                });
            }
        }
    }
}

pub(crate) struct UpdateHandler<ObjectType: ObjectTypeEnum, RenderReceiver: RenderInfoReceiver> {
    input_handler: Arc<Mutex<InputHandler>>,
    object_handler: ObjectHandler<ObjectType>,

    vertex_map: VertexMap,
    viewport: AdjustedViewport,
    resource_handler: ResourceHandler,
    render_info_receiver: Arc<Mutex<RenderReceiver>>,
    clear_col: Colour,

    coroutines: BTreeMap<ObjectId, BTreeMap<CoroutineId, Coroutine<ObjectType>>>,
    scene_instruction_tx: Sender<SceneInstruction>,
    scene_instruction_rx: Receiver<SceneInstruction>,
    scene_name: SceneName,
    scene_data: Arc<Mutex<Vec<u8>>>,

    perf_stats: UpdatePerfStats,
}

impl<ObjectType: ObjectTypeEnum, RenderReceiver: RenderInfoReceiver> UpdateHandler<ObjectType, RenderReceiver> {
    pub(crate) fn new(
        objects: Vec<AnySceneObject<ObjectType>>,
        input_handler: Arc<Mutex<InputHandler>>,
        resource_handler: ResourceHandler,
        render_info_receiver: Arc<Mutex<RenderReceiver>>,
        scene_name: SceneName,
        scene_data: Arc<Mutex<Vec<u8>>>
    ) -> Result<Self> {
        let (scene_instruction_tx, scene_instruction_rx) = mpsc::channel();
        let mut rv = Self {
            input_handler,
            object_handler: ObjectHandler::new(),
            vertex_map: VertexMap::new(),
            viewport: render_info_receiver.clone().lock().unwrap().current_viewport(),
            resource_handler,
            render_info_receiver,
            clear_col: Colour::black(),
            coroutines: BTreeMap::new(),
            scene_instruction_tx,
            scene_instruction_rx,
            scene_name,
            scene_data,
            perf_stats: UpdatePerfStats::new(),
        };

        let input_handler = rv.input_handler.lock().unwrap().clone();
        rv.perf_stats.add_objects.start();
        rv.update_with_added_objects(
            &input_handler,
            objects.into_iter()
                .map(|obj| {
                    PendingAddObject {
                        inner: Rc::new(RefCell::new(obj)),
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
                        warn!("fixed update behind by {:.2} ms",
                            f64::from_u128(fixed_update_us - FIXED_UPDATE_INTERVAL_US)
                                .unwrap_or(f64::INFINITY) / 1000.);
                    }
                }

                let input_handler = self.input_handler.lock().unwrap().clone();
                let (pending_add_objects, pending_remove_objects) =
                    self.call_on_update(delta, &input_handler, fixed_updates)
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
                self.perf_stats.report();
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

    fn update_with_added_objects(&mut self, input_handler: &InputHandler, mut pending_add_objects: Vec<PendingAddObject<ObjectType>>) -> Result<()> {
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
            let pending_add = pending_add_objects.drain(..)
                .map(|obj| (ObjectId::next(), obj))
                .collect::<BTreeMap<_, _>>();
            let first_new_id = *pending_add.first_key_value().expect("inexplicable").0;
            let last_new_id = *pending_add.last_key_value().expect("inexplicable").0;
            let new_ids = first_new_id.0..=last_new_id.0;
            self.object_handler.collision_handler.add_objects(&pending_add);

            let mut object_tracker = ObjectTracker::new(&self.object_handler);
            self.load_new_objects(&mut object_tracker, pending_add)?;
            self.call_on_ready(&mut object_tracker, input_handler, new_ids);
            let (pending_add, pending_remove) = object_tracker.into_pending();
            self.update_with_removed_objects(pending_remove);
            pending_add_objects = pending_add;
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
            let this = self.object_handler.objects.get_mut(&this_id)
                .unwrap_or_else(|| panic!("tried to call on_ready() for nonexistent added object: {this_id:?}"))
                .clone();
            let mut ctx = UpdateContext::new(self, input_handler, this_id, object_tracker);
            this.borrow_mut().on_ready(&mut ctx);
        }
    }

    fn load_new_objects(&mut self, object_tracker: &mut ObjectTracker<ObjectType>, pending_add: BTreeMap<ObjectId, PendingAddObject<ObjectType>>) -> Result<()> {
        for (new_id, new_obj) in pending_add {
            self.object_handler.add_object(new_id, new_obj.clone());
            let new_vertices = {
                let parent = self.object_handler.get_parent(new_obj.parent_id);
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
                new_obj.inner.borrow_mut().on_load(&mut object_ctx, &mut self.resource_handler)?
            };
            self.vertex_map.insert(new_id, new_vertices);
        }
        Ok(())
    }

    fn update_with_removed_objects(&mut self, pending_remove_objects: BTreeSet<ObjectId>) {
        self.object_handler.collision_handler.remove_objects(&pending_remove_objects);
        for remove_id in pending_remove_objects.into_iter().rev() {
            self.object_handler.remove_object(remove_id);
            self.vertex_map.remove(remove_id);
            self.coroutines.remove(&remove_id);
        }
    }

    fn call_on_update(&mut self,
                      delta: Duration,
                      input_handler: &InputHandler,
                      mut fixed_updates: u128
    ) -> ObjectTracker<ObjectType> {
        self.perf_stats.on_update_begin.start();
        let mut object_tracker = ObjectTracker {
            last: self.object_handler.objects.clone(),
            pending_add: Vec::new(),
            pending_remove: BTreeSet::new()
        };

        self.iter_with_other_map(delta, input_handler, &mut object_tracker,
                                 |mut obj, delta, ctx| {
                                     obj.on_update_begin(delta, ctx);
                                 });
        self.perf_stats.on_update_begin.stop();
        self.perf_stats.coroutines.start();
        self.update_coroutines(input_handler, &mut object_tracker);
        self.perf_stats.coroutines.stop();
        self.perf_stats.on_update.start();
        self.iter_with_other_map(delta, input_handler, &mut object_tracker,
                                 |mut obj, delta, ctx| {
                                     obj.on_update(delta, ctx);
                                 });
        self.perf_stats.on_update.stop();

        self.perf_stats.fixed_update.start();
        for _ in 0..fixed_updates.min(MAX_FIXED_UPDATES) {
            self.iter_with_other_map(delta, input_handler, &mut object_tracker,
                                     |mut obj, _delta, ctx| {
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
        self.iter_with_other_map(delta, input_handler, &mut object_tracker,
                                 |mut obj, delta, ctx| {
                                     obj.on_update_end(delta, ctx);
                                 });
        self.perf_stats.on_update_end.stop();
        object_tracker
    }

    fn handle_collisions(&mut self, input_handler: &InputHandler, object_tracker: &mut ObjectTracker<ObjectType>) {
        self.perf_stats.detect_collision.start();
        let collisions = self.object_handler.get_collisions();
        self.perf_stats.detect_collision.stop();
        self.perf_stats.on_collision.start();
        let mut done_with_collisions = BTreeSet::new();
        for CollisionNotification { this, other, mtv } in collisions {
            if !done_with_collisions.contains(&this.object_id) {
                let mut ctx = UpdateContext::new(
                    self,
                    input_handler,
                    this.object_id,
                    object_tracker
                );
                match this.inner.borrow_mut().on_collision(&mut ctx, other, mtv) {
                    CollisionResponse::Continue => {},
                    CollisionResponse::Done => { done_with_collisions.insert(this.object_id); },
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
                let mut ctx = UpdateContext::new(
                    self,
                    input_handler,
                    this_id,
                    object_tracker
                );
                if let Some(coroutine) = coroutine.resume(this.clone(), &mut ctx) {
                    ctx.scene.coroutines.insert(id, coroutine);
                }
            }
        }
    }
    fn iter_with_other_map<F>(&mut self,
                              delta: Duration,
                              input_handler: &InputHandler,
                              object_tracker: &mut ObjectTracker<ObjectType>,
                              call_obj_event: F)
    where F: Fn(RefMut<AnySceneObject<ObjectType>>, Duration, &mut UpdateContext<ObjectType>) {
        for (this_id, this) in self.object_handler.objects.clone() {
            let this = SceneObjectWithId::new(this_id, this.clone());
            let mut ctx = UpdateContext::new(
                self,
                input_handler,
                this_id,
                object_tracker
            );
            call_obj_event(this.inner.borrow_mut(), delta, &mut ctx);
        }
    }
    fn update_and_send_render_infos(&mut self) {
        self.perf_stats.render_infos.start();
        self.object_handler.update_render_infos(&self.vertex_map, &self.viewport);
        self.send_render_infos();
        self.perf_stats.render_infos.stop();
    }

    fn send_render_infos(&mut self) {
        let vertex_map = &mut self.vertex_map;
        let maybe_vertices = if vertex_map.consume_vertices_changed() {
            let mut vertices = Vec::with_capacity(vertex_map.vertex_count());
            for r in vertex_map.render_items() {
                vertices.extend(r.vertices);
            }
            check_eq!(vertices.len(), vertices.capacity());
            Some(vertices)
        } else {
            None
        };
        let mut render_info_receiver = self.render_info_receiver.lock().unwrap();
        if let Some(vertices) = maybe_vertices {
            render_info_receiver.update_vertices(vertices);
        }
        render_info_receiver.update_render_info(self.object_handler.render_infos.values().cloned().collect());
        render_info_receiver.set_clear_col(self.clear_col);
        self.viewport = render_info_receiver.current_viewport()
            .translated(self.viewport.translation);
    }

}

struct UpdatePerfStats {
    total_stats: TimeIt,
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
    last_report: Instant,
}

impl UpdatePerfStats {
    fn new() -> Self {
        Self {
            total_stats: TimeIt::new("total (update)"),
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
            last_report: Instant::now(),
        }
    }

    fn report(&mut self) {
        if self.last_report.elapsed().as_secs() >= 5 {
            self.on_update_begin.report_ms_if_at_least(2.);
            self.coroutines.report_ms_if_at_least(2.);
            self.on_update.report_ms_if_at_least(2.);
            self.on_update_end.report_ms_if_at_least(2.);
            self.fixed_update.report_ms_if_at_least(2.);
            self.detect_collision.report_ms_if_at_least(2.);
            self.on_collision.report_ms_if_at_least(2.);
            self.remove_objects.report_ms_if_at_least(2.);
            self.add_objects.report_ms_if_at_least(2.);
            self.render_infos.report_ms_if_at_least(2.);
            self.total_stats.report_ms_if_at_least(2.);
            self.last_report = Instant::now();
        }
    }
}

pub struct UpdateContext<'a, ObjectType: ObjectTypeEnum> {
    input: &'a InputHandler,
    scene: SceneContext<'a, ObjectType>,
    object: ObjectContext<'a, ObjectType>,
    viewport: ViewportContext<'a>,
    render: RenderContext<'a>,
}

impl<'a, ObjectType: ObjectTypeEnum> UpdateContext<'a, ObjectType> {
    fn new<R: RenderInfoReceiver>(
        caller: &'a mut UpdateHandler<ObjectType, R>,
        input_handler: &'a InputHandler,
        this_id: ObjectId,
        object_tracker: &'a mut ObjectTracker<ObjectType>
    ) -> Self {
        let parent = caller.object_handler.get_parent(this_id);
        let children = caller.object_handler.get_children(this_id);
        Self {
            input: input_handler,
            scene: SceneContext {
                scene_instruction_tx: caller.scene_instruction_tx.clone(),
                scene_name: caller.scene_name,
                scene_data: caller.scene_data.clone(),
                coroutines: caller.coroutines.entry(this_id).or_default(),
                pending_removed_coroutines: BTreeSet::new(),
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
            render: RenderContext {
                this_id,
                vertex_map: &mut caller.vertex_map,
            }
        }
    }

    pub fn object(&mut self) -> &mut ObjectContext<'a, ObjectType> { &mut self.object }
    pub fn scene(&mut self) -> &mut SceneContext<'a, ObjectType> { &mut self.scene }
    pub fn viewport(&mut self) -> &mut ViewportContext<'a> { &mut self.viewport }
    pub fn input(&self) -> &InputHandler { self.input }
    pub fn render(&mut self) -> &mut RenderContext<'a> { &mut self.render }
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
}

#[derive(Clone)]
pub(crate) struct PendingAddObject<ObjectType: ObjectTypeEnum> {
    inner: Rc<RefCell<AnySceneObject<ObjectType>>>,
    parent_id: ObjectId,
}

struct ObjectTracker<ObjectType: ObjectTypeEnum> {
    last: BTreeMap<ObjectId, Rc<RefCell<AnySceneObject<ObjectType>>>>,
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

    fn get(&self, object_id: ObjectId) -> Option<&Rc<RefCell<AnySceneObject<ObjectType>>>> {
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
    parent: Option<BorrowedSceneObjectWithId<'a, ObjectType>>,
    children: Vec<SceneObjectWithId<ObjectType>>,
    object_tracker: &'a mut ObjectTracker<ObjectType>,
    all_absolute_transforms: &'a BTreeMap<ObjectId, Transform>,
    all_parents: &'a BTreeMap<ObjectId, ObjectId>,
    all_children: &'a BTreeMap<ObjectId, Vec<SceneObjectWithId<ObjectType>>>,
}

impl<'a, ObjectType: ObjectTypeEnum> ObjectContext<'a, ObjectType> {
    pub fn parent(&self) -> Option<&BorrowedSceneObjectWithId<'a, ObjectType>> { self.parent.as_ref() }
    pub fn children(&self) -> &[SceneObjectWithId<ObjectType>] { &self.children }
    pub fn others(&self) -> Vec<SceneObjectWithId<ObjectType>> {
        self.object_tracker.last.iter()
            .filter(|(object_id, _)| !self.object_tracker.pending_remove.contains(object_id))
            .filter(|(object_id, _)| self.this_id != **object_id)
            .map(|(object_id, obj)| SceneObjectWithId::new(*object_id, obj.clone()))
            .collect()
    }
    pub fn absolute_transform(&self) -> Transform {
        *self.all_absolute_transforms.get(&self.this_id)
            .unwrap_or_else(|| panic!("missing object_id in absolute_transforms: this={:?}", self.this_id))
    }
    pub fn absolute_transform_of(&self, object_id: ObjectId) -> Transform {
        // Should not be possible to get an invalid object_id here if called from public.
        *self.all_absolute_transforms.get(&object_id)
            .unwrap_or_else(|| panic!("missing object_id in absolute_transforms: {object_id:?}"))
    }
    pub fn extents(&self) -> Rect {
        self.collider_of(self.this_id)
            .translated(self.absolute_transform().centre)
            .as_rect()
    }
    pub fn extents_of(&self, other: &SceneObjectWithId<ObjectType>) -> Rect {
        // Should not be possible to get an invalid object_id here if called from public.
        self.collider_of(other.object_id)
            .translated(self.absolute_transform_of(other.object_id).centre)
            .as_rect()
    }
    pub fn this_id(&self) -> ObjectId { self.this_id }
    fn collider_of(&self, object_id: ObjectId) -> GenericCollider {
        let children = if object_id == self.this_id {
            &self.children
        } else {
            // Should not be possible to get an invalid object_id here if called from public.
            self.all_children.get(&object_id)
                .unwrap_or_else(|| panic!("missing object_id in children: {object_id:?}"))
        };
        children.iter().find(|obj| obj.inner.borrow().get_type() == ObjectTypeEnum::gg_collider())
            .map(SceneObjectWithId::collider)
            .unwrap_or_default()
    }

    pub fn add_vec(&mut self, objects: Vec<AnySceneObject<ObjectType>>) -> Vec<Rc<RefCell<AnySceneObject<ObjectType>>>> {
        let pending_add = &mut self.object_tracker.pending_add;
        let begin = pending_add.len();
        pending_add.extend(objects.into_iter().map(|inner| {
            PendingAddObject {
                inner: Rc::new(RefCell::new(inner)),
                parent_id: self.this_id,
            }
        }));
        let end = pending_add.len();
        pending_add[begin..end].iter().map(|obj| obj.inner.clone()).collect()
    }
    pub fn add_child(&mut self, object: AnySceneObject<ObjectType>) -> Rc<RefCell<AnySceneObject<ObjectType>>> {
        let object = PendingAddObject {
            inner: Rc::new(RefCell::new(object)),
            parent_id: self.this_id,
        };
        self.object_tracker.pending_add.push(object.clone());
        object.inner
    }
    pub fn remove(&mut self, obj: &SceneObjectWithId<ObjectType>) {
        self.object_tracker.pending_remove.insert(obj.object_id);
        for child in &self.children {
            self.object_tracker.pending_remove.insert(child.object_id);
        }
    }
    pub fn remove_this(&mut self) {
        self.object_tracker.pending_remove.insert(self.this_id());
        for child in &self.children {
            self.object_tracker.pending_remove.insert(child.object_id);
        }
    }
    pub fn test_collision(&self,
                          listening_tags: Vec<&'static str>
    ) -> Option<NonemptyVec<Collision<ObjectType>>> {
        if let Some(collider) = self.children.iter()
            .find(|obj| obj.inner.borrow().get_type() == ObjectTypeEnum::gg_collider()) {
            let collider = collider.collider()
                .translated(self.all_absolute_transforms
                    .get(&collider.object_id)
                    .unwrap_or_else(|| panic!("missing object_id in absolute_transforms: {:?}",collider.object_id))
                    .centre
                );
            self.test_collision_inner(&collider, listening_tags)
        } else {
            None
        }
    }

    fn lookup_parent(&self, object_id: ObjectId) -> Option<SceneObjectWithId<ObjectType>> {
        let parent_id = self.all_parents.get(&object_id)?;
        if parent_id.0 == 0 {
            None
        } else {
            let parent = self.object_tracker.get(*parent_id)
                .unwrap_or_else(|| panic!("missing object_id in parents: {parent_id:?}"));
            Some(SceneObjectWithId::new(*parent_id, parent.clone()))
        }
    }
    fn test_collision_inner(
        &self,
        collider: &GenericCollider,
        listening_tags: Vec<&'static str>
    ) -> Option<NonemptyVec<Collision<ObjectType>>> {
        let mut rv = Vec::new();
        for tag in listening_tags {
            for other_id in self.collision_handler.get_object_ids_by_emitting_tag(tag) {
                if let Some(other) = self.object_tracker.get(*other_id) {
                    let other_collider = other.borrow().collider().translated(
                        self.all_absolute_transforms.get(other_id)
                            .unwrap_or_else(|| panic!("missing object_id in absolute_transforms: {other_id:?}"))
                            .centre
                    );
                    if let Some(mtv) = collider.collides_with(&other_collider) {
                        let other = self.lookup_parent(*other_id)
                            .unwrap_or_else(|| panic!("orphaned GgInternalCollisionShape: {other_id:?}"));
                        rv.push(Collision {
                            other,
                            mtv,
                        });
                    }
                }
            }
        }
        NonemptyVec::try_from_vec(rv)
    }
    pub fn test_collision_along(&self,
                                axis: Vec2,
                                distance: f64,
                                tags: Vec<&'static str>,
    ) -> Option<NonemptyVec<Collision<ObjectType>>> {
        if let Some(collider) = self.children.iter()
            .find(|obj| obj.inner.borrow().get_type() == ObjectTypeEnum::gg_collider()) {
            let collider = collider.collider()
                .translated(self.all_absolute_transforms
                    .get(&collider.object_id)
                    .unwrap_or_else(|| panic!("missing object_id in absolute_transforms: {:?}", collider.object_id))
                    .centre + distance * axis
                );
            self.test_collision_inner(&collider, tags)
                .and_then(|vec| {
                    NonemptyVec::try_from_iter(vec
                        .into_iter()
                        .filter(|coll| !coll.mtv.dot(axis).is_zero()))
                })
        } else {
            None
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
}

impl AxisAlignedExtent for ViewportContext<'_> {
    fn aa_extent(&self) -> Vec2 {
        self.viewport.aa_extent()
    }

    fn centre(&self) -> Vec2 {
        self.viewport.centre()
    }
}

impl<'a> RenderContext<'a> {
    pub fn update_vertices(&mut self, new_vertices: RenderItem) {
        self.vertex_map.remove(self.this_id);
        self.vertex_map.insert(self.this_id, new_vertices);
    }
}

pub struct RenderContext<'a> {
    this_id: ObjectId,
    vertex_map: &'a mut VertexMap,
}
