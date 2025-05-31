pub mod collision;

use crate::core::TreeObjectOfType;
use crate::shader::SpriteShader;
use crate::util::{InspectMut, gg_float};
use crate::{
    core::render::StoredRenderItem,
    core::scene::GuiClosure,
    core::vk::RenderPerfStats,
    core::{
        ObjectId, SceneObjectWrapper, TreeSceneObject,
        config::{FIXED_UPDATE_INTERVAL_US, MAX_FIXED_UPDATES},
        coroutine::{Coroutine, CoroutineId, CoroutineResponse, CoroutineState},
        input::InputHandler,
        prelude::*,
        render::{RenderDataChannel, RenderItem, ShaderExecWithVertexData, VertexMap},
        scene::{SceneDestination, SceneHandlerInstruction, SceneInstruction, SceneName},
        vk::AdjustedViewport,
    },
    gui::debug_gui::DebugGui,
    resource::ResourceHandler,
    resource::sprite::GgInternalSprite,
    shader::{Shader, get_shader},
    util::collision::BoxCollider,
    util::gg_err,
    util::{
        NonemptyVec,
        collision::{Collider, GenericCollider, GgInternalCollisionShape},
        colour::Colour,
        gg_time::TimeIt,
        linalg::Transform,
        linalg::{AxisAlignedExtent, Vec2},
    },
    warn_every_seconds,
};
use collision::{Collision, CollisionHandler, CollisionNotification, CollisionResponse};
use serde::{Serialize, de::DeserializeOwned};
use std::cell::RefCell;
use std::rc::Rc;
use std::{
    cell::{Ref, RefMut},
    collections::{BTreeMap, BTreeSet},
    sync::{
        Arc, Mutex, mpsc,
        mpsc::{Receiver, Sender},
    },
    time::{Duration, Instant},
};
use tracing::{span, warn};

/// A container responsible for managing scene objects and their relationships in a hierarchical
/// structure.
///
/// `ObjectHandler` maintains a collection of scene objects organized in a tree structure, tracks
/// their parent-child relationships, computes absolute transforms, and handles collision detection
/// between objects.
pub(crate) struct ObjectHandler {
    objects: BTreeMap<ObjectId, TreeSceneObject>,
    parents: BTreeMap<ObjectId, ObjectId>,
    pub(crate) absolute_transforms: BTreeMap<ObjectId, Transform>,
    children: BTreeMap<ObjectId, Vec<TreeSceneObject>>,

    object_ref_tracker: BTreeMap<ObjectId, TreeSceneObject>,
    dangling_names: BTreeMap<ObjectId, String>,

    collision_handler: CollisionHandler,
}

impl ObjectHandler {
    fn new() -> Self {
        Self {
            objects: BTreeMap::new(),
            parents: BTreeMap::new(),
            absolute_transforms: BTreeMap::new(),
            children: BTreeMap::new(),
            object_ref_tracker: BTreeMap::new(),
            dangling_names: BTreeMap::new(),
            collision_handler: CollisionHandler::new(),
        }
    }

    /// Returns `None` if and only if `id.is_root()`.
    pub(crate) fn get_object_by_id(&self, id: ObjectId) -> Result<Option<&TreeSceneObject>> {
        if id.is_root() {
            Ok(None)
        } else {
            let Some(scene_object) = self.objects.get(&id) else {
                bail!(
                    "ObjectHandler::get_object_by_id(): missing ObjectId in `objects`: {}",
                    self.format_object_id_for_logging(id)
                )
            };
            Ok(Some(scene_object))
        }
    }
    /// Returns `None` if and only if `id.is_root()`.
    pub(crate) fn get_parent_by_id(&self, id: ObjectId) -> Result<Option<&TreeSceneObject>> {
        let Some(parent_id) = self
            .lookup_parent_id(id)
            .context("ObjectHandler::get_parent_by_id()")?
        else {
            return Ok(None);
        };
        self.get_object_by_id(parent_id)
            .context("ObjectHandler::get_parent_by_id()")
    }
    fn lookup_parent_id(&self, id: ObjectId) -> Result<Option<ObjectId>> {
        if id.is_root() {
            Ok(None)
        } else {
            self.parents.get(&id).copied().map(Some).ok_or_else(|| {
                anyhow!(
                    "ObjectHandler::lookup_parent_id(): missing ObjectId in `parents`: {}",
                    self.format_object_id_for_logging(id)
                )
            })
        }
    }

    /// Returns the chain of parent IDs from a given [`ObjectId`] to the root.
    ///
    /// Traverses up the object hierarchy starting from the given ID, collecting
    /// all parent IDs until reaching the root object.
    ///
    /// # Returns
    /// * `Ok(Vec<ObjectId>)` - Vector of parent IDs in order. For example, if the hierarchy is
    ///   `root -> A -> B`, returns `vec![B, A]`.
    /// * `Err` - If a parent ID is missing from the internal parent map.
    pub(crate) fn get_parent_chain(&self, mut id: ObjectId) -> Result<Vec<ObjectId>> {
        let mut rv = Vec::new();
        let orig_id = id;
        while !id.is_root() {
            rv.push(id);
            id = self
                .lookup_parent_id(id)
                .with_context(|| {
                    format!(
                        "ObjectHandler::get_parent_chain(): {:?}",
                        self.format_object_id_for_logging(orig_id)
                    )
                })?
                .unwrap();
        }
        Ok(rv)
    }

    /// Look up the children of the given [`ObjectId`]. The returned [`Vec`] is in no particular
    /// order.
    pub(crate) fn get_children(&self, id: ObjectId) -> Result<&Vec<TreeSceneObject>> {
        self.children.get(&id).ok_or_else(|| {
            anyhow!(
                "ObjectHandler::get_children(): missing ObjectId in `children`: {}",
                self.format_object_id_for_logging(id)
            )
        })
    }
    fn get_children_mut(&mut self, id: ObjectId) -> Result<&mut Vec<TreeSceneObject>> {
        if !self.children.contains_key(&id) {
            return Err(anyhow!(
                "ObjectHandler::get_children_mut(): missing ObjectId in `children`: {}",
                self.format_object_id_for_logging(id)
            ));
        }
        // SAFETY: checked immediately above.
        unsafe { Ok(self.children.get_mut(&id).unwrap_unchecked()) }
    }

    /// Returns all collision shapes in a scene tree, starting from a given node.
    ///
    /// This method traverses the scene tree recursively starting from the given `id`, collecting
    /// all collision shape components found in the object itself and its children.
    ///
    /// # Returns
    /// * A vector of tuples containing the object ID and a reference to its collision shape
    /// * Returns `Err` if any object lookup fails
    pub(crate) fn get_collision_shapes(
        &self,
        id: ObjectId,
    ) -> Result<Vec<TreeObjectOfType<CollisionShape>>> {
        let mut rv = Vec::new();
        if let Some(c) = self
            .get_object_by_id(id)
            .context("ObjectHandler::get_collision_shapes()")?
            .and_then(TreeObjectOfType::of)
        {
            rv.push(c);
        }
        for child in self.get_children(id)? {
            rv.extend(self.get_collision_shapes(child.object_id)?);
        }
        Ok(rv)
    }

    /// Caution: does not automatically remove children.
    fn remove_object(&mut self, remove_id: ObjectId) {
        let name = self.get_object_by_id(remove_id).ok().flatten().map_or(
            "<unknown>".to_string(),
            TreeSceneObject::nickname_or_type_name,
        );
        // Remove this object from its parent's list of children.
        let parent_id = gg_err::log_err_then(
            self.get_parent_by_id(remove_id)
                .context("ObjectHandler::remove_object()"),
        )
        .map_or(ObjectId::root(), |p| p.object_id);
        if let Ok(children) = self.get_children_mut(parent_id) {
            // If this object's parent has already been removed, `children` may not exist.
            // This is not an error.
            children.retain(|obj| obj.object_id != remove_id);
        }
        self.parents.remove(&remove_id);

        self.objects.remove(&remove_id);
        self.absolute_transforms.remove(&remove_id);
        self.children.remove(&remove_id);

        let o = self.object_ref_tracker.get(&remove_id).unwrap();
        let count = Rc::strong_count(&o.scene_object.wrapped);
        if count > 1 {
            info!("remaining references to `{name} ({remove_id:?})`: {count}");
            self.dangling_names.insert(remove_id, name);
        } else {
            self.object_ref_tracker.remove(&remove_id);
        }
    }

    /// Adds a new object to the scene hierarchy and establishes its parent-child relationships.
    /// If this is the first object being added, will first initialise the root node (ID 0), then
    /// the new object to be added.
    ///
    /// # Errors
    /// Returns error if the new object's parent ID is not found in the hierarchy
    fn add_object(&mut self, new_obj: &TreeSceneObject) -> Result<()> {
        if self.children.is_empty() {
            self.children.insert(ObjectId(0), Vec::new());
            self.absolute_transforms
                .insert(ObjectId(0), Transform::default());
        }
        self.objects.insert(new_obj.object_id, new_obj.clone());
        self.object_ref_tracker
            .insert(new_obj.object_id, new_obj.clone());
        self.parents.insert(new_obj.object_id, new_obj.parent_id);
        self.children.insert(new_obj.object_id, Vec::new());
        let children = self.get_children_mut(new_obj.parent_id)?;
        children.push(new_obj.clone());
        Ok(())
    }

    fn reparent_object(
        &mut self,
        target_id: ObjectId,
        new_parent_id: ObjectId,
    ) -> Result<TreeSceneObject> {
        let last_parent_id = self
            .lookup_parent_id(target_id)
            .with_context(|| {
                format!("ObjectHandler::reparent_object({target_id:?}, {new_parent_id:?})")
            })?
            .unwrap_or(ObjectId::root());
        self.parents.insert(target_id, new_parent_id);
        self.get_children_mut(last_parent_id)
            .with_context(|| format!("ObjectHandler::reparent_object({target_id:?}, {new_parent_id:?}): remove from last_parent_id children"))?
            .retain(|o| o.object_id != target_id);
        let o = self
            .get_object_by_id(target_id)
            .with_context(|| format!("ObjectHandler::reparent_object({target_id:?}, {new_parent_id:?})"))?
            .cloned()
            .with_context(|| format!("ObjectHandler::reparent_object({target_id:?}, {new_parent_id:?}): target_id == root?"))?;
        self.get_children_mut(new_parent_id)
            .with_context(|| format!("ObjectHandler::reparent_object({target_id:?}, {new_parent_id:?}): add to new_parent_id children"))?
            .push(o.clone());
        Ok(o)
    }

    fn get_collisions(&mut self) -> Vec<CollisionNotification> {
        self.collision_handler.get_collisions(self)
    }

    fn update_all_transforms(&mut self) {
        let mut child_stack = Vec::with_capacity(self.objects.len());
        child_stack.push((ObjectId::root(), Transform::default()));
        while let Some((parent_id, parent_transform)) = child_stack.pop() {
            self.absolute_transforms.insert(parent_id, parent_transform);
            if let Some(children) = gg_err::log_and_ok(
                self.get_children(parent_id)
                    .context("ObjectHandler::update_all_transforms()"),
            ) {
                for child in children {
                    let absolute_transform = child.transform() * parent_transform;
                    child
                        .downcast_mut::<GgInternalCollisionShape>()
                        .inspect_mut(|shape| shape.update_transform(absolute_transform));
                    child_stack.push((child.object_id, absolute_transform));
                }
            }
        }
    }

    fn create_shader_execs(&mut self, vertex_map: &mut VertexMap) -> Vec<ShaderExecWithVertexData> {
        self.update_all_transforms();
        for (id, object) in &self.objects {
            if let Some(renderable) = object.inner_mut().as_renderable_object() {
                renderable.on_render(&mut RenderContext::new(*id, vertex_map));
            }
        }
        let mut shader_execs = Vec::with_capacity(vertex_map.len());
        let mut start = 0;
        for item in vertex_map.render_items() {
            let Some(transform) = self.absolute_transforms.get(&item.object_id) else {
                error!(
                    "ObjectHandler: missing ObjectId in `absolute_transforms`: {}",
                    self.format_object_id_for_logging(item.object_id)
                );
                continue;
            };
            check_false!(item.object_id.is_root());
            let Some(object) = gg_err::log_err_then(
                self.get_object_by_id(item.object_id)
                    .context("create_shader_execs()"),
            ) else {
                continue;
            };
            let mut object = object.scene_object.wrapped.borrow_mut();
            let Some(renderable) = object.as_renderable_object() else {
                error!(
                    "ObjectHandler: object in vertex_map not renderable: {}",
                    self.format_object_id_for_logging(item.object_id)
                );
                continue;
            };
            let mut shader_exec_inner = renderable.shader_execs();
            for shader_exec in &mut shader_exec_inner {
                if !shader_exec.shader_id.is_valid() {
                    shader_exec.shader_id = get_shader(SpriteShader::name());
                }
            }

            let end = start + item.len() as u32;
            shader_execs.push(ShaderExecWithVertexData {
                vertex_indices: start..end,
                inner: shader_exec_inner,
                transform: *transform,
                depth: item.render_item.depth,
            });
            start = end;
        }
        shader_execs
    }

    pub(crate) fn cleanup_references(&mut self) {
        let mut leaked_bytes = 0;
        for id in self.object_ref_tracker.keys().copied().collect_vec() {
            let o = self.object_ref_tracker.get(&id).unwrap();
            let count = Rc::strong_count(&o.scene_object.wrapped);
            if !self.objects.contains_key(&id) {
                leaked_bytes += count * size_of::<TreeSceneObject>();
                let name = self
                    .dangling_names
                    .get(&id)
                    .cloned()
                    .unwrap_or("<unknown>".to_string());
                if count > 1 {
                    warn_every_seconds!(1, "dangling references to {name} ({id:?}): {count}");
                } else {
                    self.object_ref_tracker.remove(&id);
                    self.dangling_names.remove(&id);
                }
            }
        }
        if leaked_bytes > 0 {
            warn_every_seconds!(1, "leaked memory: {:.2} KiB", (leaked_bytes as f64) / 1024.);
        }
    }

    pub(crate) fn get_first_object_id_for_gui(&self) -> Option<ObjectId> {
        self.objects.first_key_value().map(|o| o.0).copied()
    }
    pub(crate) fn has_sprite_for_gui(&self, id: ObjectId) -> Result<bool> {
        let Some(object) = self
            .get_object_by_id(id)
            .context("ObjectHandler::has_sprite_for_gui()")?
        else {
            check!(id.is_root());
            return Ok(false);
        };
        if object.gg_is::<GgInternalSprite>() {
            Ok(true)
        } else {
            for child in self.get_children(id)? {
                if self.has_sprite_for_gui(child.object_id)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
    }
    pub(crate) fn format_object_id_for_logging(&self, id: ObjectId) -> String {
        let label = self.objects.get(&id).map_or(
            "<unknown>".to_string(),
            TreeSceneObject::nickname_or_type_name,
        );
        format!("{label} [{id:?}]")
    }
}

pub(crate) struct UpdateHandler {
    input_handler: Arc<Mutex<InputHandler>>,
    object_handler: ObjectHandler,

    vertex_map: VertexMap,
    viewport: AdjustedViewport,
    resource_handler: ResourceHandler,
    render_data_channel: Arc<Mutex<RenderDataChannel>>,
    clear_col: Colour,

    coroutines: BTreeMap<ObjectId, BTreeMap<CoroutineId, Coroutine>>,
    scene_instruction_tx: Sender<SceneInstruction>,
    scene_instruction_rx: Receiver<SceneInstruction>,
    scene_name: SceneName,
    scene_data: Arc<Mutex<Vec<u8>>>,

    debug_gui: DebugGui,
    gui_cmd: Option<Box<GuiClosure>>,

    perf_stats: UpdatePerfStats,
    last_render_perf_stats: Option<RenderPerfStats>,
    delta: Duration,
    frame_counter: usize,
    fixed_frame_counter: usize,
}

impl UpdateHandler {
    pub(crate) fn new(
        objects: Vec<SceneObjectWrapper>,
        input_handler: Arc<Mutex<InputHandler>>,
        resource_handler: ResourceHandler,
        render_data_channel: Arc<Mutex<RenderDataChannel>>,
        scene_name: SceneName,
        scene_data: Arc<Mutex<Vec<u8>>>,
    ) -> Result<Self> {
        let (scene_instruction_tx, scene_instruction_rx) = mpsc::channel();
        let clear_col = render_data_channel.lock().unwrap().get_clear_col();
        let mut rv = Self {
            input_handler,
            object_handler: ObjectHandler::new(),
            vertex_map: VertexMap::new(),
            viewport: render_data_channel
                .clone()
                .lock()
                .unwrap()
                .current_viewport(),
            resource_handler,
            render_data_channel,
            clear_col,
            coroutines: BTreeMap::new(),
            scene_instruction_tx,
            scene_instruction_rx,
            scene_name,
            scene_data,
            debug_gui: DebugGui::new()?,
            gui_cmd: None,
            perf_stats: UpdatePerfStats::new(),
            last_render_perf_stats: None,
            delta: Duration::from_secs(0),
            frame_counter: 0,
            fixed_frame_counter: 0,
        };

        let input_handler = rv.input_handler.lock().unwrap().clone();
        rv.perf_stats.add_objects.start();
        let mut pending_move_objects = BTreeMap::new();
        rv.update_with_added_objects(
            &input_handler,
            objects
                .into_iter()
                .map(|obj| TreeSceneObject {
                    scene_object: obj,
                    object_id: ObjectId::next(),
                    parent_id: ObjectId::root(),
                })
                .collect(),
            &mut pending_move_objects,
        );
        rv.perf_stats.add_objects.stop();
        rv.update_with_moved_objects(pending_move_objects);
        rv.update_and_send_render_infos();
        Ok(rv)
    }

    pub(crate) fn consume(mut self) -> Result<SceneHandlerInstruction> {
        let mut is_running = true;
        let mut fixed_update_us = 0;

        loop {
            if is_running {
                let now = Instant::now();
                self.perf_stats.total_stats.start();

                let update_span = span!(
                    tracing::Level::INFO,
                    "update",
                    fc = self.frame_counter,
                    // TODO: update `ffc`.
                    ffc = self.fixed_frame_counter
                );
                let _enter = update_span.enter();

                // Handle fixed update.
                fixed_update_us += self.delta.as_micros();
                let fixed_updates = fixed_update_us / FIXED_UPDATE_INTERVAL_US;
                if fixed_updates > 0 {
                    fixed_update_us -= FIXED_UPDATE_INTERVAL_US;
                    if fixed_update_us >= FIXED_UPDATE_INTERVAL_US {
                        warn!(
                            "fixed update behind by {:.1} ms",
                            gg_float::from_u128_or_inf(fixed_update_us - FIXED_UPDATE_INTERVAL_US)
                                / 1000.
                        );
                    }
                    if fixed_update_us >= FIXED_UPDATE_TIMEOUT {
                        warn!(
                            "fixed update behind by {:.1} ms, giving up",
                            gg_float::from_u128_or_inf(fixed_update_us - FIXED_UPDATE_INTERVAL_US)
                                / 1000.
                        );
                        fixed_update_us = 0;
                    }
                }

                // Handle regular update.
                let input_handler = self.input_handler.lock().unwrap().clone();
                let (pending_add_objects, pending_remove_objects, mut pending_move_objects) = self
                    .call_on_update(&input_handler, fixed_updates)
                    .into_pending();

                self.perf_stats.remove_objects.start();
                self.update_with_removed_objects(pending_remove_objects);
                self.perf_stats.remove_objects.stop();
                self.perf_stats.add_objects.start();
                self.update_with_added_objects(
                    &input_handler,
                    pending_add_objects,
                    &mut pending_move_objects,
                );
                self.perf_stats.add_objects.stop();
                // TODO: add perf stats.
                self.update_with_moved_objects(pending_move_objects);
                self.debug_gui
                    .on_end_step(&input_handler, &mut self.viewport);

                // Handle render.
                self.update_and_send_render_infos();
                self.input_handler.lock().unwrap().update_step();

                // Update performance statistics.
                self.perf_stats.total_stats.stop();
                self.debug_gui
                    .on_perf_stats(self.perf_stats.get(), self.last_render_perf_stats.clone());
                if !self.debug_gui.scene_control.is_paused() {
                    self.frame_counter += 1;
                }

                if self.perf_stats.totals_s.len() == self.perf_stats.totals_s.capacity() {
                    self.perf_stats.totals_s.remove(0);
                }
                self.perf_stats.totals_s.push(now.elapsed().as_secs_f32());
                self.delta = now.elapsed();
            }

            match self.scene_instruction_rx.try_iter().next() {
                Some(SceneInstruction::Stop) => {
                    return Ok(SceneHandlerInstruction::Exit);
                }
                Some(SceneInstruction::Goto(instruction)) => {
                    return Ok(SceneHandlerInstruction::Goto(instruction));
                }
                Some(SceneInstruction::Pause) => {
                    is_running = false;
                }
                Some(SceneInstruction::Resume) => {
                    is_running = true;
                }
                None => {}
            }
        }
    }

    fn update_with_added_objects(
        &mut self,
        input_handler: &InputHandler,
        mut pending_add_objects: Vec<TreeSceneObject>,
        pending_move_objects: &mut BTreeMap<ObjectId, ObjectId>,
    ) {
        // Multiple iterations, because on_load() may add more objects.
        // See e.g. GgInternalContainer.
        let mut new_ids = BTreeSet::new();
        while !pending_add_objects.is_empty() {
            pending_add_objects.retain(|obj| {
                let rv = obj.parent_id.is_root()
                    || self.object_handler.objects.contains_key(&obj.parent_id);
                if !rv {
                    info!(
                        "removed orphaned object: {:?} (parent {:?})",
                        obj.nickname_or_type_name(),
                        obj.parent_id
                    );
                }
                rv
            });
            if pending_add_objects.is_empty() {
                break;
            }
            let pending_add = pending_add_objects.drain(..).collect_vec();
            for id in pending_add[0].object_id.0..=pending_add.last().unwrap().object_id.0 {
                new_ids.insert(ObjectId(id));
            }

            let mut object_tracker = ObjectTracker::new(&self.object_handler);
            self.object_handler
                .collision_handler
                .add_objects(pending_add.iter());
            self.load_new_objects(&mut object_tracker, pending_add);
            self.object_handler.update_all_transforms();

            let (pending_add, pending_remove, new_pending_move_objects) =
                object_tracker.into_pending();
            for (object_id, new_parent_id) in new_pending_move_objects {
                check!(!pending_move_objects.keys().contains(&object_id));
                pending_move_objects.insert(object_id, new_parent_id);
            }
            pending_add_objects = pending_add;
            self.update_with_removed_objects(pending_remove);
        }

        let mut object_tracker = ObjectTracker::new(&self.object_handler);
        self.call_on_ready(&mut object_tracker, input_handler, new_ids);
        let (pending_add, pending_remove, new_pending_move_objects) = object_tracker.into_pending();
        for (object_id, new_parent_id) in new_pending_move_objects {
            check!(!pending_move_objects.keys().contains(&object_id));
            pending_move_objects.insert(object_id, new_parent_id);
        }
        self.update_with_removed_objects(pending_remove);
        self.object_handler.update_all_transforms();

        if !pending_add.is_empty() {
            warn!(
                "fc={}: recursive call to update_with_added_objects(); should not add objects in on_ready()",
                self.frame_counter
            );
            self.update_with_added_objects(input_handler, pending_add, pending_move_objects);
        }
    }
    fn update_with_moved_objects(&mut self, pending_move_objects: BTreeMap<ObjectId, ObjectId>) {
        for (target_id, new_parent_id) in pending_move_objects {
            check_ne!(target_id, new_parent_id);
            if let Some(last_parent_id) = gg_err::log_and_ok(self.object_handler
                .lookup_parent_id(target_id)
                .with_context(|| {
                    format!("UpdateHandler::update_with_moved_objects({target_id:?}, {new_parent_id:?})")
                })) {
                let last_parent_id = last_parent_id.unwrap_or(ObjectId::root());
                gg_err::log_err_and_ignore(
                    self.debug_gui
                        .on_move_object(&self.object_handler, target_id, last_parent_id, new_parent_id)
                        .context("UpdateHandler::update_with_moved_objects()"),
                );
            }
            gg_err::log_err_and_ignore(
                self.object_handler
                    .reparent_object(target_id, new_parent_id),
            );
        }
    }

    fn call_on_ready(
        &mut self,
        object_tracker: &mut ObjectTracker,
        input_handler: &InputHandler,
        new_ids: impl IntoIterator<Item = ObjectId>,
    ) {
        for this_id in new_ids {
            gg_err::log_and_ok(
                self.object_handler
                    .objects
                    .get_mut(&this_id)
                    .with_context(|| {
                        format!(
                            "tried to call on_ready() for nonexistent added object: {this_id:?}"
                        )
                    })
                    .cloned()
                    .and_then(|this| {
                        let mut ctx =
                            UpdateContext::new(self, input_handler, this_id, object_tracker)
                                .context("UpdateHandler::call_on_ready()")?;
                        this.inner_mut().on_ready(&mut ctx);
                        Ok(())
                    }),
            );
        }
    }

    fn load_new_objects<I>(&mut self, object_tracker: &mut ObjectTracker, pending_add: I)
    where
        I: IntoIterator<Item = TreeSceneObject>,
    {
        for new_obj in pending_add {
            let Some(parent) = gg_err::log_and_ok(
                self.object_handler
                    .get_parent_by_id(new_obj.parent_id)
                    .context("UpdateHandler::load_new_objects()"),
            ) else {
                continue;
            };
            let parent = parent.cloned();
            if gg_err::log_and_ok(
                self.object_handler
                    .add_object(&new_obj)
                    .context("UpdateHandler::load_new_objects()"),
            )
            .is_none()
            {
                continue;
            }
            object_tracker
                .objects
                .insert(new_obj.object_id(), new_obj.clone());
            gg_err::log_err_and_ignore(
                self.debug_gui
                    .on_add_object(
                        &self.object_handler,
                        &new_obj,
                        // parent.as_ref().map_or(ObjectId::root(), |p| p.object_id),
                    )
                    .context("UpdateHandler::load_new_objects()"),
            );
            let mut object_ctx = ObjectContext {
                collision_handler: &self.object_handler.collision_handler,
                this_id: new_obj.object_id,
                this_parent: parent,
                this_children: Vec::new(),
                object_tracker,
                all_absolute_transforms: &self.object_handler.absolute_transforms,
                all_parents: &self.object_handler.parents,
                all_children: &self.object_handler.children,
                dummy_transform: Rc::new(RefCell::new(Transform::default())),
            };
            if let Some(new_vertices) = gg_err::log_and_ok(
                new_obj
                    .inner_mut()
                    .on_load(&mut object_ctx, &mut self.resource_handler)
                    .context("UpdateHandler::load_new_objects()"),
            )
            .flatten()
            {
                self.vertex_map.insert(new_obj.object_id, new_vertices);
            }
        }
        self.debug_gui.on_done_adding_objects(&self.object_handler);
    }

    fn update_with_removed_objects(&mut self, pending_remove_objects: BTreeSet<ObjectId>) {
        self.object_handler
            .collision_handler
            .remove_objects(&pending_remove_objects);
        let pending_remove_objects = pending_remove_objects
            .into_iter()
            .sorted()
            .rev()
            .collect_vec();
        for remove_id in &pending_remove_objects {
            gg_err::log_err_and_ignore(
                self.debug_gui
                    .on_remove_object(&self.object_handler, *remove_id)
                    .context("UpdateHandler::update_with_removed_objects()"),
            );
        }
        // Iterate in reverse order so that children are removed before parents.
        for remove_id in pending_remove_objects {
            self.object_handler.remove_object(remove_id);
            self.vertex_map.remove(remove_id);
            self.coroutines.remove(&remove_id);
        }
    }

    fn call_on_update(
        &mut self,
        input_handler: &InputHandler,
        mut fixed_updates: u128,
    ) -> ObjectTracker {
        let mut object_tracker = ObjectTracker {
            objects: self.object_handler.objects.clone(),
            pending_add: Vec::new(),
            pending_remove: BTreeSet::new(),
            pending_move: BTreeMap::new(),
        };

        self.perf_stats.on_gui.start();
        self.update_gui(input_handler, &mut object_tracker);
        self.perf_stats.on_gui.stop();
        self.object_handler.update_all_transforms();
        if self.debug_gui.scene_control.is_paused() {
            // TODO: cache pressed buttons until step.
            if self.debug_gui.scene_control.should_step() {
                fixed_updates = 1;
            } else {
                return object_tracker;
            }
        }

        self.perf_stats.on_update_begin.start();
        self.iter_with_other_map(
            input_handler,
            &mut object_tracker,
            |mut obj, ctx| {
                obj.on_update_begin(ctx);
            },
            "on_update_begin",
        );
        self.object_handler.update_all_transforms();
        self.perf_stats.on_update_begin.stop();
        self.perf_stats.coroutines.start();
        self.update_coroutines(input_handler, &mut object_tracker);
        self.object_handler.update_all_transforms();
        self.perf_stats.coroutines.stop();
        self.perf_stats.on_update.start();
        self.iter_with_other_map(
            input_handler,
            &mut object_tracker,
            |mut obj, ctx| {
                obj.on_update(ctx);
            },
            "on_update",
        );
        self.object_handler.update_all_transforms();
        self.perf_stats.on_update.stop();

        self.perf_stats.fixed_update.start();
        for _ in 0..fixed_updates.min(MAX_FIXED_UPDATES) {
            for this_id in self.object_handler.objects.keys().copied().collect_vec() {
                let Some(this) = gg_err::log_err_then(
                    self.object_handler
                        .get_object_by_id(this_id)
                        .context("UpdateHandler::call_on_update(): on_fixed_update: parent"),
                ) else {
                    error!(
                        "UpdateHandler::call_on_update(): tried to call on_fixed_update() on root object"
                    );
                    continue;
                };
                let this = this.clone(); // borrowck issues
                if let Some(mut ctx) = gg_err::log_and_ok(
                    FixedUpdateContext::new(self, this_id, &mut object_tracker).context(
                        "UpdateHandler::call_on_update(): on_fixed_update: FixedUpdateContext",
                    ),
                ) {
                    this.inner_mut().on_fixed_update(&mut ctx);
                }
            }
            self.object_handler.update_all_transforms();
            fixed_updates -= 1;
            self.fixed_frame_counter += 1;
        }
        self.perf_stats.fixed_update.stop();

        self.handle_collisions(input_handler, &mut object_tracker);

        self.perf_stats.on_update_end.start();
        self.iter_with_other_map(
            input_handler,
            &mut object_tracker,
            |mut obj, ctx| {
                obj.on_update_end(ctx);
            },
            "on_update_end",
        );
        self.object_handler.update_all_transforms();
        self.object_handler.cleanup_references();
        self.perf_stats.on_update_end.stop();
        object_tracker
    }

    fn update_gui(&mut self, input_handler: &InputHandler, object_tracker: &mut ObjectTracker) {
        if !USE_DEBUG_GUI {
            return;
        }
        if input_handler.pressed(KeyCode::Backquote) {
            self.debug_gui.toggle();
        }

        if self.debug_gui.enabled() {
            // Handle mouseovers.
            let all_tags = self.object_handler.collision_handler.all_tags();
            self.debug_gui.clear_mouseovers(&self.object_handler);
            if let Some(screen_mouse_pos) = input_handler.screen_mouse_pos() {
                let mouse_pos = self.viewport.top_left() + screen_mouse_pos;
                if let Some(collisions) = gg_err::log_and_ok(
                    UpdateContext::new(self, input_handler, ObjectId::root(), object_tracker)
                        .context("UpdateHandler::update_gui(): on_mouseovers"),
                )
                .and_then(|ctx| ctx.object.test_collision_point(mouse_pos, all_tags))
                {
                    self.debug_gui
                        .on_mouseovers(&self.object_handler, collisions);
                }
            }
        }
        // We have to do this even if !self.debug_gui.enabled() so that the in/out animations work.
        let selected_object = self.debug_gui.selected_object();
        let gui_cmds = self
            .object_handler
            .objects
            .clone()
            .into_iter()
            .filter_map(|(id, obj)| {
                gg_err::log_and_ok(
                    UpdateContext::new(self, input_handler, id, object_tracker)
                        .context("UpdateHandler::update_gui(): on_gui"),
                )
                .and_then(|ctx| {
                    obj.inner_mut()
                        .as_gui_object()
                        .map(|gui_obj| (id, gui_obj.on_gui(&ctx, selected_object == Some(id))))
                })
            })
            .collect();
        self.gui_cmd = Some(self.debug_gui.build(
            input_handler,
            &mut self.object_handler,
            gui_cmds,
        ));
    }

    fn handle_collisions(
        &mut self,
        input_handler: &InputHandler,
        object_tracker: &mut ObjectTracker,
    ) {
        self.perf_stats.detect_collision.start();
        let collisions = self.object_handler.get_collisions();
        self.perf_stats.detect_collision.stop();
        self.perf_stats.on_collision.start();
        let mut done_with_collisions = BTreeSet::new();
        for CollisionNotification { this, other, mtv } in collisions {
            if done_with_collisions.contains(&this.object_id) {
                continue;
            }
            if let Some(CollisionResponse::Done) = gg_err::log_and_ok(
                UpdateContext::new(self, input_handler, this.object_id, object_tracker)
                    .with_context(|| {
                        format!(
                            "UpdateHandler::handle_collisions(): {:?} - {:?}",
                            this.object_id, other.object_id
                        )
                    }),
            )
            .map(|mut ctx| {
                this.scene_object
                    .wrapped
                    .borrow_mut()
                    .on_collision(&mut ctx, &other, mtv)
            }) {
                done_with_collisions.insert(this.object_id);
            }
        }
        self.object_handler.update_all_transforms();
        self.perf_stats.on_collision.stop();
    }

    fn update_coroutines(
        &mut self,
        input_handler: &InputHandler,
        object_tracker: &mut ObjectTracker,
    ) {
        for this_id in self.object_handler.objects.keys().copied().collect_vec() {
            let Some(this) = gg_err::log_err_then(
                self.object_handler
                    .get_object_by_id(this_id)
                    .context("UpdateHandler::call_on_update(): update_coroutines"),
            ) else {
                error!(
                    "UpdateHandler::call_on_update(): tried to call update_coroutines on root object"
                );
                continue;
            };
            let this = this.clone(); // borrowck issues
            for (coroutine_id, coroutine) in self.coroutines.remove(&this_id).unwrap_or_default() {
                let Some(mut ctx) = gg_err::log_and_ok(
                    UpdateContext::new(self, input_handler, this_id, object_tracker)
                        .context("UpdateHandler::update_coroutines()"),
                ) else {
                    continue;
                };
                if let Some(coroutine) = coroutine.resume(&this, &mut ctx) {
                    ctx.scene.coroutines.insert(coroutine_id, coroutine);
                }
            }
        }
    }
    fn iter_with_other_map<F>(
        &mut self,
        input_handler: &InputHandler,
        object_tracker: &mut ObjectTracker,
        call_obj_event: F,
        description: &str,
    ) where
        F: Fn(RefMut<dyn SceneObject>, &mut UpdateContext),
    {
        for (this_id, this) in self.object_handler.objects.clone() {
            gg_err::log_and_ok(
                UpdateContext::new(self, input_handler, this_id, object_tracker).with_context(
                    || format!("UpdateHandler::iter_with_other_map(): {description}"),
                ),
            )
            .inspect_mut(|ctx| call_obj_event(this.inner_mut(), ctx));
        }
    }
    fn update_and_send_render_infos(&mut self) {
        self.perf_stats.render_infos.start();

        let shader_execs = self
            .object_handler
            .create_shader_execs(&mut self.vertex_map);
        let maybe_vertices = if self.vertex_map.consume_vertices_changed() {
            Some(
                self.vertex_map
                    .render_items()
                    .flat_map(StoredRenderItem::vertices)
                    .copied()
                    .collect_vec(),
            )
        } else {
            None
        };
        let mut render_data_channel = self.render_data_channel.lock().unwrap();
        self.last_render_perf_stats = render_data_channel.last_render_stats.clone();
        render_data_channel.gui_commands = self.gui_cmd.take().into_iter().collect_vec();
        render_data_channel.gui_enabled = self.debug_gui.enabled();
        if let Some(vertices) = maybe_vertices {
            render_data_channel.vertices = vertices;
        }
        render_data_channel.shader_execs = shader_execs;
        render_data_channel.set_global_scale_factor(self.viewport.global_scale_factor());
        render_data_channel.set_clear_col(self.clear_col);
        render_data_channel.set_translation(self.viewport.translation);
        self.viewport = render_data_channel.current_viewport();

        self.perf_stats.render_infos.stop();
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
    extra_debug: TimeIt,
    last_perf_stats: Option<Box<UpdatePerfStats>>,
    last_report: Instant,
    totals_s: Vec<f32>,
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
            extra_debug: TimeIt::new("extra_debug"),
            last_perf_stats: None,
            last_report: Instant::now(),
            totals_s: Vec::with_capacity(10),
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
                extra_debug: self.extra_debug.report_take(),
                last_perf_stats: None,
                last_report: Instant::now(),
                totals_s: self.totals_s.clone(),
            }));
            self.last_report = Instant::now();
        }
        self.last_perf_stats.clone().map(|s| *s)
    }

    pub(crate) fn as_tuples_ms(&self) -> Vec<(String, f32, f32)> {
        let mut default = vec![
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
        ];
        if self.extra_debug.last_ms() != 0. {
            default.push(self.extra_debug.as_tuple_ms());
        }
        default
    }

    pub fn fps(&self) -> f32 {
        self.totals_s.iter().map(|t| 1. / t).sum::<f32>() / self.totals_s.len() as f32
    }
}

/// A context object that provides access to various subsystems during scene object update events.
///
/// `UpdateContext` is passed to scene objects during the events `on_update()`,
/// `on_collision()`, etc. It provides:
///
/// - Access to input state via `input()`
/// - Scene management (coroutines, scene data) via `scene_mut()`
/// - Object manipulation (transforms, collisions, hierarchy) via `object_mut()`
/// - Viewport control via `viewport_mut()`
/// - Frame timing information like delta time and FPS
///
/// # Example
///
/// ```ignore
/// use glongge::core::prelude::*;
///
/// fn on_update(&mut self, ctx: &mut UpdateContext) {
///     // Access input
///     if ctx.input().pressed(KeyCode::Space) {
///         // Modify object transform
///         ctx.object_mut().transform_mut().centre += Vec2::up();
///     }
///
///     // Start a coroutine
///     ctx.scene_mut().start_coroutine(|this, ctx, action| {
///         // Coroutine logic
///         CoroutineResponse::Complete
///     });
///
///     // Update viewport
///     ctx.viewport_mut().centre_at(ctx.object_mut().transform().centre);
/// }
/// ```
pub struct UpdateContext<'a> {
    input: &'a InputHandler,
    scene: SceneContext<'a>,
    object: ObjectContext<'a>,
    viewport: ViewportContext<'a>,
    delta: Duration,
    fps: f32,
    frame_counter: usize,
    fixed_frame_counter: usize,
}

impl<'a> UpdateContext<'a> {
    fn new(
        caller: &'a mut UpdateHandler,
        input_handler: &'a InputHandler,
        this_id: ObjectId,
        object_tracker: &'a mut ObjectTracker,
    ) -> Result<Self> {
        let fps = caller.perf_stats.fps();
        let parent = caller
            .object_handler
            .get_parent_by_id(this_id)
            .context("UpdateContext::new()")?;
        let children = caller
            .object_handler
            .get_children(this_id)
            .context("UpdateContext::new()")?
            .clone();
        Ok(Self {
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
                this_parent: parent.cloned(),
                this_children: children,
                object_tracker,
                all_absolute_transforms: &caller.object_handler.absolute_transforms,
                all_parents: &caller.object_handler.parents,
                all_children: &caller.object_handler.children,
                dummy_transform: Rc::new(RefCell::new(Transform::default())),
            },
            viewport: ViewportContext {
                viewport: &mut caller.viewport,
                clear_col: &mut caller.clear_col,
            },
            delta: caller.delta,
            fps,
            frame_counter: caller.frame_counter,
            fixed_frame_counter: caller.fixed_frame_counter,
        })
    }

    /// Returns an immutable reference to the [`ObjectContext`] for the current object.
    ///
    /// The [`ObjectContext`] provides methods to:
    /// - Access and modify the object's transform/position
    /// - Add/remove child objects
    /// - Check collisions with other objects
    /// - Access parent/child relationships
    /// - Query other objects in the scene
    ///
    /// Similar to [`object_mut()`](UpdateContext::object_mut), but provides read-only access.
    /// Should be used in the majority of cases; if you need to add or remove objects, see
    /// [`object_mut()`](UpdateContext::object_mut).
    ///
    /// # Examples
    /// ```ignore
    /// use glongge::core::prelude::*;
    ///
    /// // Get parent object
    /// if let Some(parent) = ctx.object().parent() {
    ///     // Access parent properties...
    /// }
    ///
    /// // Check collisions
    /// if let Some(collisions) = ctx.object().test_collision(tags) {
    ///     // Handle collisions...
    /// }
    /// ```
    pub fn object(&self) -> &ObjectContext<'a> {
        &self.object
    }
    /// Returns a mutable reference to the [`ObjectContext`] for the current object.
    ///
    /// Should usually only be used when you need to add or remove objects. For general use cases,
    /// see [`object()`](UpdateContext::object).
    ///
    /// # Examples
    /// ```ignore
    /// use glongge::core::prelude::*;
    ///
    /// // Add a child object
    /// ctx.object_mut().add_child(MyObject::new());
    ///
    /// // Remove a specific object and its children
    /// ctx.object_mut().remove(other_object);
    ///
    /// // Remove this object and its children
    /// ctx.object_mut().remove_this();
    /// ```
    pub fn object_mut(&mut self) -> &mut ObjectContext<'a> {
        &mut self.object
    }
    /// Returns an immutable reference to the [`SceneContext`] for managing scene-related
    /// functionality.
    ///
    /// The [`SceneContext`] provides methods to:
    /// - Access and modify scene-specific persistent data
    /// - Start, stop and manage coroutines (background tasks)
    /// - Control scene flow (stop, pause, resume, transition between scenes)
    /// - Access scene metadata like the current scene name
    ///
    /// Provides read access to scene data, coroutines, and scene control. For write access, use
    /// [`scene_mut()`](UpdateContext::scene_mut).
    ///
    /// # Examples
    /// ```ignore
    /// use glongge::core::prelude::*;
    ///
    /// // Access scene data
    /// if let Some(mut data) = ctx.scene().data::<GameState>() {
    ///     // Read game state...
    /// }
    ///
    /// // Start a coroutine
    /// ctx.scene_mut().start_coroutine(|this, ctx, action| {
    ///     // Coroutine logic...
    ///     CoroutineResponse::Complete
    /// });
    ///
    /// // Scene flow control
    /// ctx.scene().goto(SceneDestination::MainMenu);
    /// ```
    pub fn scene(&self) -> &SceneContext<'a> {
        &self.scene
    }

    /// Returns a mutable reference to the [`SceneContext`] for managing scene-related functionality.
    ///
    /// Provides write access to scene data, coroutines, and scene control.
    /// See [`scene()`](UpdateContext::scene) for more information.
    pub fn scene_mut(&mut self) -> &mut SceneContext<'a> {
        &mut self.scene
    }

    /// Returns an immutable reference to the [`ViewportContext`] for viewport operations.
    ///
    /// The [`ViewportContext`] provides methods to:
    /// - Control the viewport's position and scale
    /// - Set viewport boundaries and perform clamping
    /// - Change the clear color for rendering
    /// - Access viewport properties like dimensions and center
    ///
    /// Provides read access to viewport properties. For write access, use
    /// [`viewport_mut()`](UpdateContext::viewport_mut).
    ///
    /// # Examples
    /// ```ignore
    /// // Check if point is visible in viewport
    /// if ctx.viewport().contains_point(pos) {
    ///     // Point is visible...
    /// }
    ///
    /// // Get viewport dimensions
    /// let viewport_size = ctx.viewport().aa_extent();
    ///
    /// // Get viewport center position
    /// let center = ctx.viewport().centre();
    /// ```
    pub fn viewport(&self) -> &ViewportContext<'a> {
        &self.viewport
    }

    /// Returns a mutable reference to the [`ViewportContext`] for viewport operations.
    ///
    /// See [`viewport()`](UpdateContext::viewport) for more information.
    ///
    /// # Examples
    /// ```ignore
    /// use glongge::core::prelude::*;
    ///
    /// // Center viewport on object
    /// ctx.viewport_mut().centre_at(ctx.transform().centre);
    ///
    /// // Translate viewport
    /// ctx.viewport_mut().translate(Vec2::right() * 10.0);
    ///
    /// // Clamp viewport position
    /// ctx.viewport_mut().clamp_to_left(Some(-100.0), Some(100.0));
    /// ctx.viewport_mut().clamp_to_right(Some(-100.0), Some(100.0));
    ///
    /// // Change clear color
    /// ctx.viewport_mut().clear_col().set_rgba(1.0, 0.0, 0.0, 1.0);
    ///
    /// // Change viewport scale
    /// ctx.viewport_mut().set_global_scale_factor(2.0);
    /// ```
    pub fn viewport_mut(&mut self) -> &mut ViewportContext<'a> {
        &mut self.viewport
    }

    /// Returns a reference to the [`InputHandler`] for querying input states.
    ///
    /// Provides access to keyboard, mouse, and touch input states.
    pub fn input(&self) -> &InputHandler {
        self.input
    }

    /// Returns the absolute transform of this object in world space.
    ///
    /// The absolute transform accounts for all parent transforms in the hierarchy chain.
    pub fn absolute_transform(&self) -> Transform {
        self.object.absolute_transform()
    }

    /// Returns the local transform of this object relative to its parent.
    pub fn transform(&self) -> Transform {
        self.object.transform()
    }

    /// Returns a mutable reference to the local transform of this object.
    ///
    /// Modifying this transform affects the object's position, rotation, and scale relative to its
    /// parent.
    ///
    /// # Examples
    /// ```ignore
    /// use glongge::core::prelude::*;
    ///
    /// // Change position
    /// ctx.transform_mut().centre += Vec2::right() * 10.0;  // Move right
    /// ctx.transform_mut().centre = Vec2::new(100.0, 200.0); // Set absolute position
    ///
    /// // Rotate object
    /// ctx.transform_mut().rotation += 0.1;  // Rotate clockwise
    /// ctx.transform_mut().rotation = std::f32::consts::PI; // Set to 180 degrees
    ///
    /// // Scale object
    /// ctx.transform_mut().scale *= 1.5; // Increase size by 50%
    /// ctx.transform_mut().scale = Vec2::splat(2.0); // Double size on both axes
    ///
    /// // Chain multiple transform changes
    /// let mut transform = ctx.transform_mut();
    /// transform.centre += Vec2::up() * 5.0;
    /// transform.rotation += 0.5;
    /// transform.scale *= 1.1;
    /// ```
    pub fn transform_mut(&self) -> RefMut<Transform> {
        self.object.transform_mut()
    }

    /// Returns the time elapsed since the last frame as a [`Duration`].
    /// See [`delta_60fps()`](UpdateContext::delta_60fps()) for more information.
    /// In general, prefer using [`delta_60fps()`](UpdateContext::delta_60fps()).
    pub fn delta(&self) -> Duration {
        self.delta
    }

    /// Returns the time elapsed since the last frame scaled to 60 FPS.
    /// This is useful for frame-rate independent movement/animations.
    ///
    /// # Example
    /// ```ignore
    /// // Move object at consistent speed regardless of frame rate
    /// let speed = 5.0;
    /// ctx.transform_mut().centre += Vec2::right() * speed * ctx.delta_60fps();
    /// ```
    pub fn delta_60fps(&self) -> f32 {
        self.delta.as_secs_f32() * 60.0
    }

    /// Returns the current frames per second (FPS).
    ///
    /// # Example
    /// ```ignore
    /// // Check if running at target frame rate
    /// if ctx.fps() < 60.0 {
    ///     // Optimize/reduce effects...
    /// }
    /// ```
    pub fn fps(&self) -> f32 {
        self.fps
    }

    /// Returns the total number of frames rendered since scene start.
    ///
    /// # Example
    /// ```ignore
    /// // Do something every 60 frames
    /// if ctx.frame_counter() % 60 == 0 {
    ///     // Periodic effect...
    /// }
    /// ```
    pub fn frame_counter(&self) -> usize {
        self.frame_counter
    }

    /// Returns the total number of fixed update frames processed.
    /// Fixed updates run at a constant time interval for time-dependent logic. See
    /// [`on_fixed_update()`](SceneObject::on_fixed_update).
    pub fn fixed_frame_counter(&self) -> usize {
        self.fixed_frame_counter
    }
}

impl Drop for UpdateContext<'_> {
    fn drop(&mut self) {
        for id in &self.scene.pending_removed_coroutines {
            self.scene.coroutines.remove(id);
        }
        self.scene.pending_removed_coroutines.clear();
        check!(self.scene.pending_removed_coroutines.is_empty());
    }
}

/// Stripped-down version of [`UpdateContext`](UpdateContext) for use during
/// [`on_fixed_update()`](SceneObject::on_fixed_update).
pub struct FixedUpdateContext<'a> {
    scene: SceneContext<'a>,
    object: ObjectContext<'a>,
    viewport: ViewportContext<'a>,
    frame_counter: usize,
    fixed_frame_counter: usize,
}

impl<'a> FixedUpdateContext<'a> {
    fn new(
        caller: &'a mut UpdateHandler,
        this_id: ObjectId,
        object_tracker: &'a mut ObjectTracker,
    ) -> Result<Self> {
        let parent = caller
            .object_handler
            .get_parent_by_id(this_id)
            .context("FixedUpdateContext::new()")?;
        let children = caller.object_handler.get_children(this_id)?.clone();
        Ok(Self {
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
                this_parent: parent.cloned(),
                this_children: children,
                object_tracker,
                all_absolute_transforms: &caller.object_handler.absolute_transforms,
                all_parents: &caller.object_handler.parents,
                all_children: &caller.object_handler.children,
                dummy_transform: Rc::new(RefCell::new(Transform::default())),
            },
            viewport: ViewportContext {
                viewport: &mut caller.viewport,
                clear_col: &mut caller.clear_col,
            },
            frame_counter: caller.frame_counter,
            fixed_frame_counter: caller.fixed_frame_counter,
        })
    }

    pub fn object_mut(&mut self) -> &mut ObjectContext<'a> {
        &mut self.object
    }
    pub fn object(&self) -> &ObjectContext<'a> {
        &self.object
    }
    pub fn scene_mut(&mut self) -> &mut SceneContext<'a> {
        &mut self.scene
    }
    pub fn scene(&self) -> &SceneContext<'a> {
        &self.scene
    }
    pub fn viewport_mut(&mut self) -> &mut ViewportContext<'a> {
        &mut self.viewport
    }
    pub fn viewport(&self) -> &ViewportContext<'a> {
        &self.viewport
    }

    pub fn absolute_transform(&self) -> Transform {
        self.object.absolute_transform()
    }
    pub fn transform(&self) -> Transform {
        self.object.transform()
    }
    pub fn transform_mut(&self) -> RefMut<Transform> {
        self.object.transform_mut()
    }

    pub fn frame_counter(&self) -> usize {
        self.frame_counter
    }
    pub fn fixed_frame_counter(&self) -> usize {
        self.fixed_frame_counter
    }
}

/// Represents persistent scene data that can be serialised/deserialised between scene loads.
///
/// This type provides a way to store and access persistent data across scene loads. The data is
/// automatically serialised when the `SceneData` is dropped.
///
/// # Examples
///
/// ```ignore
/// use serde::{Serialize, Deserialize};
///
/// // Define your scene data structure
/// #[derive(Default, Serialize, Deserialize)]
/// struct AliveEnemyMap {
///     alive_enemies: HashSet<Vec2i>,
/// }
///
/// // Access the data in scene objects
/// fn on_ready(&mut self, ctx: &mut UpdateContext) {
///     // Get mutable access to scene data
///     let mut data = ctx.scene_mut().data::<AliveEnemyMap>().unwrap();
///     
///     // Read data
///     if !data.read().alive_enemies.contains(&pos) {
///         ctx.object_mut().remove_this();
///     }
///
///     // Modify data
///     data.write().alive_enemies.insert(pos);
/// }
/// ```
pub struct SceneData<T>
where
    T: Default + Serialize + DeserializeOwned,
{
    raw: Arc<Mutex<Vec<u8>>>,
    deserialized: T,
    modified: bool,
}

impl<T> SceneData<T>
where
    T: Default + Serialize + DeserializeOwned,
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

    pub fn read(&self) -> &T {
        &self.deserialized
    }
    pub fn write(&mut self) -> &mut T {
        self.modified = true;
        &mut self.deserialized
    }
}

impl<T> Drop for SceneData<T>
where
    T: Default + Serialize + DeserializeOwned,
{
    fn drop(&mut self) {
        if self.modified {
            *self.raw.try_lock().expect("scene_data locked?") =
                bincode::serialize(&self.deserialized).expect("failed to serialize scene data");
        }
    }
}

/// See [`scene()`](UpdateContext::scene) for usage information.
pub struct SceneContext<'a> {
    scene_instruction_tx: Sender<SceneInstruction>,
    scene_name: SceneName,
    scene_data: Arc<Mutex<Vec<u8>>>,
    coroutines: &'a mut BTreeMap<CoroutineId, Coroutine>,
    pending_removed_coroutines: BTreeSet<CoroutineId>,
}

impl SceneContext<'_> {
    /// Stop the game and exit gracefully.
    pub fn stop(&self) {
        self.scene_instruction_tx
            .send(SceneInstruction::Stop)
            .unwrap();
    }
    /// Go to a new scene.
    pub fn goto(&self, instruction: SceneDestination) {
        self.scene_instruction_tx
            .send(SceneInstruction::Goto(instruction))
            .unwrap();
    }
    pub fn name(&self) -> SceneName {
        self.scene_name
    }
    pub fn data<T>(&mut self) -> Option<SceneData<T>>
    where
        T: Default + Serialize + DeserializeOwned,
    {
        SceneData::new(self.scene_data.clone())
            .expect("failed to ser/de scene_data, do the types match?")
    }

    /// Starts a new coroutine (background task) for this object.
    ///
    /// Coroutines allow performing actions over multiple frames, like animations or delayed effects.
    /// The coroutine function is called each frame until it returns `CoroutineResponse::Complete`.
    ///
    /// # Parameters  
    /// * `func` - The coroutine function to execute. Takes:
    ///   - A reference to this object as a [`TreeSceneObject`]
    ///   - A mutable reference to the current [`UpdateContext`]
    ///   - The most recent [`CoroutineState`], indicating if the coroutine has just started,
    ///     whether it yielded, or whether it was waiting for a duration.
    ///
    /// # Returns
    /// * A unique [`CoroutineId`] that can be used to cancel the coroutine later.
    ///
    /// # Examples
    /// ```ignore
    /// // Start a coroutine that moves an object over time
    /// ctx.scene_mut().start_coroutine(|this, ctx, last_state| {
    ///     match last_state {
    ///         CoroutineState::Starting => {
    ///             // Initialize state
    ///             CoroutineResponse::Wait(Duration::from_secs(1))
    ///         }
    ///         CoroutineState::Resuming => {
    ///             // Move object
    ///             ctx.object_mut().transform_mut().centre += Vec2::up();
    ///             CoroutineResponse::Complete
    ///         }
    ///     }
    /// });
    /// ```
    pub fn start_coroutine<F>(&mut self, func: F) -> CoroutineId
    where
        F: FnMut(&TreeSceneObject, &mut UpdateContext, CoroutineState) -> CoroutineResponse
            + 'static,
    {
        let id = CoroutineId::next();
        self.coroutines.insert(id, Coroutine::new(func));
        id
    }
    /// Helper function to start a new coroutine after a specified delay duration.
    ///
    /// Similar to [`start_coroutine()`](SceneContext::start_coroutine), but waits for the given
    /// duration before executing the coroutine function for the first time.
    ///
    /// # Parameters
    /// * `func` - The coroutine function to execute after the delay. Takes:
    ///   - A reference to this object as a [`TreeSceneObject`]
    ///   - A mutable reference to the current [`UpdateContext`]
    ///   - The most recent [`CoroutineState`]
    /// * `duration` - How long to wait before starting the coroutine
    ///
    /// # Returns
    /// * A unique [`CoroutineId`] that can be used to cancel the coroutine.
    ///
    /// # Examples
    /// ```ignore
    /// // Start a coroutine after 2 seconds
    /// ctx.scene_mut().start_coroutine_after(|this, ctx, state| {
    ///     // Coroutine logic...
    ///     CoroutineResponse::Complete
    /// }, Duration::from_secs(2));
    /// ```
    pub fn start_coroutine_after<F>(&mut self, mut func: F, duration: Duration) -> CoroutineId
    where
        F: FnMut(&TreeSceneObject, &mut UpdateContext, CoroutineState) -> CoroutineResponse
            + 'static,
    {
        self.start_coroutine(move |this, ctx, action| match action {
            CoroutineState::Starting => CoroutineResponse::Wait(duration),
            _ => func(this, ctx, action),
        })
    }
    /// Cancels a running coroutine with the given ID.
    ///
    /// This will stop the coroutine from executing in future frames. The coroutine will be
    /// removed at the end of the current frame.
    ///
    /// # Parameters
    /// * `id` - The ID of the coroutine to cancel, obtained when starting the coroutine (see
    ///   [`start_coroutine()`](SceneContext::start_coroutine)).
    ///
    /// # Examples
    /// ```ignore
    /// // Start a coroutine and store its ID
    /// let coroutine_id = ctx.scene_mut().start_coroutine(|this, ctx, state| {
    ///     // Coroutine logic...
    ///     CoroutineResponse::Complete
    /// });
    ///
    /// // Later, cancel the coroutine
    /// ctx.scene_mut().cancel_coroutine(coroutine_id);
    /// ```
    pub fn cancel_coroutine(&mut self, id: CoroutineId) {
        self.pending_removed_coroutines.insert(id);
    }
}

struct ObjectTracker {
    objects: BTreeMap<ObjectId, TreeSceneObject>,
    pending_add: Vec<TreeSceneObject>,
    pending_remove: BTreeSet<ObjectId>,
    pending_move: BTreeMap<ObjectId, ObjectId>,
}

impl ObjectTracker {
    fn new(object_handler: &ObjectHandler) -> Self {
        Self {
            objects: object_handler.objects.clone(),
            pending_add: Vec::new(),
            pending_remove: BTreeSet::new(),
            pending_move: BTreeMap::new(),
        }
    }

    fn get(&self, object_id: ObjectId) -> Option<&TreeSceneObject> {
        self.objects.get(&object_id)
    }
    // fn get_mut(&mut self, object_id: ObjectId) -> Option<&mut SceneObjectWrapper> {
    //     self.last.get_mut(&object_id)
    // }
}

impl ObjectTracker {
    fn into_pending(
        self,
    ) -> (
        Vec<TreeSceneObject>,
        BTreeSet<ObjectId>,
        BTreeMap<ObjectId, ObjectId>,
    ) {
        (self.pending_add, self.pending_remove, self.pending_move)
    }
}

/// Provides access to scene object operations and hierarchy during event callbacks.
///
/// `ObjectContext` allows manipulation of a scene object and its relationships during event
/// callbacks like `on_update()`. It provides methods to:
///
/// - Access and modify transforms (position, rotation, scale)
/// - Add/remove child objects and manage hierarchies
/// - Check collisions with other objects
/// - Query scene object relationships (parent/child/siblings)  
/// - Access and modify object state
///
/// Note: The `this_id` field is never set to the root object (ObjectId(0)) in this context.
///
/// # Example
/// ```ignore
/// fn on_update(&mut self, ctx: &mut UpdateContext) {
///     // Access object transform
///     let pos = ctx.object().transform().centre;
///     
///     // Add a child object
///     ctx.object_mut().add_child(MyChild::new());
///     
///     // Check collisions
///     if let Some(collisions) = ctx.object().test_collision(tags) {
///         // Handle collisions...
///     }
/// }
/// ```
pub struct ObjectContext<'a> {
    collision_handler: &'a CollisionHandler,
    this_id: ObjectId,
    this_parent: Option<TreeSceneObject>,
    this_children: Vec<TreeSceneObject>,
    object_tracker: &'a mut ObjectTracker,
    all_absolute_transforms: &'a BTreeMap<ObjectId, Transform>,
    all_parents: &'a BTreeMap<ObjectId, ObjectId>,
    all_children: &'a BTreeMap<ObjectId, Vec<TreeSceneObject>>,
    // Used if a mutable lookup fails.
    dummy_transform: Rc<RefCell<Transform>>,
}

impl ObjectContext<'_> {
    /// Returns a reference to the parent scene object if one exists.
    ///
    /// Returns [`None`] if this is a root object or if the parent object has been removed.
    ///
    /// # Examples
    /// ```ignore
    /// if let Some(parent) = ctx.object().parent() {
    ///     // Access parent properties...
    ///     let parent_transform = parent.transform();
    /// }
    /// ```
    pub fn parent(&self) -> Option<&TreeSceneObject> {
        self.this_parent.as_ref()
    }
    /// Returns the chain of parent objects from this object to the root.
    ///
    /// Traverses up the scene hierarchy starting from this object, collecting
    /// all parent objects until reaching the root.
    ///
    /// # Returns
    /// * A vector containing parent objects in order, starting from the immediate parent.
    ///   For example, if the hierarchy is `root -> A -> B -> this`, returns `[B, A]`.
    pub fn parent_chain(&self) -> Vec<&TreeSceneObject> {
        let mut object_id = self.this_id;
        let mut parent_chain = Vec::new();
        while let Some(p) = self.lookup_parent(object_id) {
            object_id = p.object_id;
            parent_chain.push(p);
        }
        parent_chain
    }
    /// Returns a vector containing all child objects of this object.
    ///
    /// # Examples
    /// ```ignore
    /// // Get all children
    /// for child in ctx.object().children() {
    ///     // Process each child...
    /// }
    ///
    /// // Find specific child objects
    /// let sprites = ctx.object().children().into_iter()
    ///     .filter_map(|c| c.downcast::<Enemy>())
    ///     .collect::<Vec<_>>();
    /// ```
    pub fn children(&self) -> &Vec<TreeSceneObject> {
        &self.this_children
    }

    /// Returns a reference to the vector of child objects for the given scene object.
    ///
    /// Similar to [`children()`](ObjectContext::children), but for accessing another object's
    /// children instead of this object's children.
    ///
    /// # Arguments
    /// * `obj` - The scene object whose children to return
    ///
    /// # Returns
    /// * `Some(&Vec<TreeSceneObject>)` - Reference to the vector of child objects if found
    /// * `None` - If the object has no children or if the object ID is invalid
    pub fn children_of(&self, obj: &TreeSceneObject) -> Option<&Vec<TreeSceneObject>> {
        gg_err::log_and_ok(
            self.children_of_inner(obj.object_id)
                .context("ObjectContext::children_of()"),
        )
    }
    pub fn children_of_inner(&self, object_id: ObjectId) -> Result<&Vec<TreeSceneObject>> {
        if object_id == self.this_id {
            Ok(&self.this_children)
        } else {
            self.all_children
                .get(&object_id)
                .with_context(|| format!("ObjectContext::children_of_inner(): missing object_id in children: {object_id:?}"))
        }
    }

    /// Returns the [`ObjectId`] of this object's parent in the scene hierarchy, or [`None`] if
    /// this is a root object.
    ///
    /// Note: You should rarely need to access object IDs directly. Consider using higher-level
    /// methods like [`parent()`](ObjectContext::parent) instead.
    pub fn parent_id(&self) -> Option<ObjectId> {
        self.this_parent.as_ref().map(TreeSceneObject::object_id)
    }
    /// Returns the unique [`ObjectId`] for this object in the scene hierarchy.
    ///
    /// Note: You should rarely need to access object IDs directly. Consider using higher-level
    /// methods like using the context's object references instead.
    pub fn this_id(&self) -> ObjectId {
        self.this_id
    }

    /// Returns the fully qualified name of this object as a path string.
    ///
    /// The name consists of parent object names in hierarchical order from root, separated by '/'.
    /// For example, if the hierarchy is `root -> A -> B -> this`, returns `/A/B/this_name`.
    /// Each object's name is determined by either its nickname if set, or its type name.
    ///
    /// Returns "MISSING" if the object ID is not found in the object tracker.
    pub fn fully_qualified_name(&self) -> String {
        self.object_tracker.get(self.this_id).map_or_else(
            || {
                error!(
                    "ObjectContext: missing ObjectId in `object_tracker`: this={:?}",
                    self.this_id
                );
                "MISSING".to_string()
            },
            |o| {
                let mut chain = self
                    .parent_chain()
                    .into_iter()
                    .map(TreeSceneObject::nickname_or_type_name)
                    .rev()
                    .collect_vec();
                chain.insert(0, String::new()); // root
                chain.push(o.nickname_or_type_name());
                chain.join("/")
            },
        )
    }

    /// Returns the first child object of type `T` in this object's children list.
    /// Note: if you intend to downcast to `T`, consider
    /// [`first_child_as_ref()`](ObjectContext::first_child_as_ref) or
    /// [`first_child_as_mut()`](ObjectContext::first_child_as_mut) instead.
    ///
    /// # Type Parameters
    /// * `T` - The type of scene object to search for
    ///
    /// # Returns
    /// * `Some(TreeSceneObject)` - The first child matching type T, if found
    /// * `None` - If no child of type T exists
    pub fn first_child<T: SceneObject>(&self) -> Option<&TreeSceneObject> {
        self.this_children.iter().find(|&obj| obj.gg_is::<T>())
    }

    /// Returns a reference to the first child object of type `T` in this object's children list.
    ///
    /// # Type Parameters
    /// * `T` - The type of scene object to search for
    ///
    /// # Returns
    /// * `Some(Ref<T>)` - A reference to the first child matching type T, if found
    /// * `None` - If no child of type T exists
    pub fn first_child_as_ref<T: SceneObject>(&self) -> Option<Ref<T>> {
        self.this_children
            .iter()
            .find_map(DowncastRef::downcast::<T>)
    }

    /// Returns a mutable reference to the first child object of type `T` in this object's children
    /// list.
    ///
    /// # Type Parameters
    /// * `T` - The type of scene object to search for
    ///
    /// # Returns
    /// * `Some(RefMut<T>)` - A mutable reference to the first child matching type T, if found
    /// * `None` - If no child of type T exists
    pub fn first_child_as_mut<T: SceneObject>(&self) -> Option<RefMut<T>> {
        self.this_children
            .iter()
            .find_map(DowncastRef::downcast_mut::<T>)
    }

    pub fn first_child_into<T: SceneObject>(&self) -> Option<TreeObjectOfType<T>> {
        let o = self.first_child::<T>()?;
        Some(
            o.try_into()
                .expect("should be guaranteed by TreeObjectOfType::of()"),
        )
    }

    /// Returns the first child object of type `T` for the given object ID.
    /// Note: if you intend to downcast to `T`, consider
    /// [`first_child_of_as_ref()`](ObjectContext::first_child_of_as_ref) or
    /// [`first_child_of_as_mut()`](ObjectContext::first_child_of_as_mut) instead.
    ///
    /// # Type Parameters
    /// * `T` - The type of scene object to search for
    ///
    /// # Parameters
    /// * `id` - The ID of the object whose children to search
    ///
    /// # Returns
    /// * `Some(TreeSceneObject)` - The first child matching type T, if found
    /// * `None` - If no child of type T exists or if the object ID is invalid
    pub fn first_child_of<T: SceneObject>(&self, id: ObjectId) -> Option<&TreeSceneObject> {
        self.all_children
            .get(&id)?
            .iter()
            .find(|&obj| obj.gg_is::<T>())
    }

    /// Returns a reference to the first child object of type `T` for the given object ID.
    ///
    /// # Type Parameters
    /// * `T` - The type of scene object to search for
    ///
    /// # Parameters
    /// * `id` - The ID of the object whose children to search
    ///
    /// # Returns
    /// * `Some(Ref<T>)` - A reference to the first child matching type T, if found
    /// * `None` - If no child of type T exists or if the object ID is invalid
    pub fn first_child_of_as_ref<T: SceneObject>(&self, id: ObjectId) -> Option<Ref<T>> {
        self.all_children
            .get(&id)?
            .iter()
            .find_map(DowncastRef::downcast::<T>)
    }

    /// Returns a mutable reference to the first child object of type `T` for the given object ID.
    ///
    /// # Type Parameters
    /// * `T` - The type of scene object to search for
    ///
    /// # Parameters
    /// * `id` - The ID of the object whose children to search
    ///
    /// # Returns
    /// * `Some(RefMut<T>)` - A mutable reference to the first child matching type T, if found
    /// * `None` - If no child of type T exists or if the object ID is invalid
    pub fn first_child_of_as_mut<T: SceneObject>(&self, id: ObjectId) -> Option<RefMut<T>> {
        self.all_children
            .get(&id)?
            .iter()
            .find_map(DowncastRef::downcast_mut::<T>)
    }
    pub fn first_child_of_into<T: SceneObject>(&self, id: ObjectId) -> Option<TreeObjectOfType<T>> {
        let o = self.first_child_of::<T>(id)?;
        Some(
            o.try_into()
                .expect("should be guaranteed by TreeObjectOfType::of()"),
        )
    }
    pub fn first_sibling<T: SceneObject>(&self) -> Option<&TreeSceneObject> {
        self.first_child_of::<T>(self.parent_id().unwrap_or(ObjectId::root()))
    }
    pub fn first_sibling_as_ref<T: SceneObject>(&self) -> Option<Ref<T>> {
        self.first_child_of_as_ref::<T>(self.parent_id().unwrap_or(ObjectId::root()))
    }
    pub fn first_sibling_as_mut<T: SceneObject>(&self) -> Option<RefMut<T>> {
        self.first_child_of_as_mut::<T>(self.parent_id().unwrap_or(ObjectId::root()))
    }
    pub fn first_sibling_into<T: SceneObject>(&self) -> Option<TreeObjectOfType<T>> {
        self.first_child_of_into::<T>(self.parent_id().unwrap_or(ObjectId::root()))
    }

    fn others_inner(&self) -> impl Iterator<Item = &TreeSceneObject> {
        self.object_tracker
            .objects
            .iter()
            .filter(|(object_id, _)| !self.object_tracker.pending_remove.contains(object_id))
            .filter(|(object_id, _)| self.this_id != **object_id)
            .map(|(_, obj)| obj)
    }
    /// Returns a vector of all other objects in the scene, excluding this object.
    ///
    /// This includes all objects that:
    /// - Are not this object
    /// - Are not pending addition/removal
    /// - Exist in the object tracker
    ///
    /// Objects are returned as [`TreeSceneObject`]s which include their ID, parent ID,
    /// and the wrapped scene object.
    ///
    /// # Examples
    /// ```ignore
    /// // Process all other objects
    /// for other in ctx.object().others() {
    ///     // Access other object properties
    ///     println!("Found object: {:?}", other.scene_object.gg_type_name());
    /// }
    /// ```
    pub fn others(&self) -> Vec<&TreeSceneObject> {
        self.others_inner().collect()
    }
    /// Returns immutable references to all other objects in the scene that match type `T`.
    pub fn others_as_ref<T: SceneObject>(&self) -> Vec<Ref<T>> {
        self.others_inner()
            .filter_map(TreeSceneObject::downcast)
            .collect()
    }

    /// Returns mutable references to all other objects in the scene that match type `T`.
    pub fn others_as_mut<T: SceneObject>(&self) -> Vec<RefMut<T>> {
        self.others_inner()
            .filter_map(TreeSceneObject::downcast_mut)
            .collect()
    }

    /// Returns the first object in the scene of type `T`, excluding this object.
    ///
    /// Note: if you intend to downcast to `T`, consider
    /// [`first_other_as_ref()`](ObjectContext::first_other_as_ref) or
    /// [`first_other_as_mut()`](ObjectContext::first_other_as_mut) instead.
    pub fn first_other<T: SceneObject>(&self) -> Option<&TreeSceneObject> {
        self.others_inner().find(|object| object.gg_is::<T>())
    }

    /// Returns a reference to the first object in the scene of type `T`, excluding this object.
    pub fn first_other_as_ref<T: SceneObject>(&self) -> Option<Ref<T>> {
        self.others_inner().find_map(TreeSceneObject::downcast)
    }

    /// Returns a mutable reference to the first object in the scene of type `T`, excluding this
    /// object.
    pub fn first_other_as_mut<T: SceneObject>(&self) -> Option<RefMut<T>> {
        self.others_inner().find_map(TreeSceneObject::downcast_mut)
    }

    pub fn first_other_into<T: SceneObject>(&self) -> Option<TreeObjectOfType<T>> {
        let o = self.first_other::<T>()?;
        Some(
            o.try_into()
                .expect("should be guaranteed by TreeObjectOfType::of()"),
        )
    }

    /// Returns the local transform of this object relative to its parent.
    ///
    /// This transform represents the object's local position, rotation and scale,
    /// before any parent transforms are applied.
    ///
    /// Returns [`Transform::default()`] and logs an error if the object ID is not found.
    pub fn transform(&self) -> Transform {
        self.object_tracker.get(self.this_id).map_or_else(
            || {
                error!("missing object_id in objects: this={:?}", self.this_id);
                Transform::default()
            },
            TreeSceneObject::transform,
        )
    }

    /// Returns the local transform of another object relative to its parent.
    ///
    /// Similar to [`transform()`](Self::transform), but for accessing another object's transform.
    ///
    /// Returns [`Transform::default()`] and logs an error if the object ID is not found.
    pub fn transform_of(&self, other: &TreeSceneObject) -> Transform {
        self.object_tracker.get(other.object_id).map_or_else(
            || {
                error!("missing object_id in objects: {:?}", other.object_id);
                Transform::default()
            },
            TreeSceneObject::transform,
        )
    }

    /// Returns a mutable reference to this object's local transform.
    ///
    /// This allows modifying the object's local position, rotation and scale
    /// relative to its parent.
    ///
    /// Returns a dummy transform and logs an error if the object ID is not found.
    pub fn transform_mut(&self) -> RefMut<Transform> {
        self.object_tracker.get(self.this_id).map_or_else(
            || {
                error!("missing object_id in objects: this={:?}", self.this_id);
                self.dummy_transform.borrow_mut()
            },
            TreeSceneObject::transform_mut,
        )
    }

    /// Returns the absolute transform of this object in world space.
    ///
    /// This transform represents the object's final position, rotation and scale
    /// after all parent transforms in the hierarchy have been applied.
    ///
    /// Returns [`Transform::default()`] and logs an error if the object ID is not found.
    pub fn absolute_transform(&self) -> Transform {
        self.all_absolute_transforms
            .get(&self.this_id)
            .copied()
            .unwrap_or_else(|| {
                error!(
                    "missing object_id in absolute_transforms: this={:?}",
                    self.this_id
                );
                Transform::default()
            })
    }

    /// Returns the absolute transform of another object in world space.
    ///
    /// Similar to [`absolute_transform()`](Self::absolute_transform), but for accessing
    /// another object's absolute transform.
    ///
    /// Returns [`Transform::default()`] and logs an error if the object ID is not found.
    /// This should not occur when called through public APIs.
    pub fn absolute_transform_of(&self, other: &TreeSceneObject) -> Transform {
        // Should not be possible to get an invalid object_id here if called from public.
        self.all_absolute_transforms
            .get(&other.object_id)
            .copied()
            .unwrap_or_else(|| {
                error!(
                    "missing object_id in absolute_transforms: {:?}",
                    other.object_id
                );
                Transform::default()
            })
    }
    /// Returns the rectangular bounds of this object's collider.
    ///
    /// If the object has multiple child colliders, returns the bounds of the first collider found.
    /// Returns a default [`Rect`] if no collider is found.
    pub fn rect(&self) -> Rect {
        self.collider().unwrap_or_default().as_rect()
    }

    /// Returns the rectangular bounds of another object's collider.
    ///
    /// If the other object has multiple child colliders, returns the bounds of the first collider found.
    /// Returns a default [`Rect`] if no collider is found.
    pub fn rect_of(&self, other: &TreeSceneObject) -> Rect {
        self.collider_of(other).unwrap_or_default().as_rect()
    }

    /// Returns the width and height of this object's collider.
    ///
    /// If the object has multiple child colliders, returns the extent of the first collider found.
    /// Returns a default [`Vec2`] if no collider is found.
    pub fn extent(&self) -> Vec2 {
        self.collider().unwrap_or_default().aa_extent()
    }

    /// Returns the width and height of another object's collider.
    ///
    /// If the other object has multiple child colliders, returns the extent of the first collider found.
    /// Returns a default [`Vec2`] if no collider is found.
    pub fn extent_of(&self, other: &TreeSceneObject) -> Vec2 {
        self.collider_of(other).unwrap_or_default().aa_extent()
    }

    /// Returns this object's collider.
    ///
    /// If the object has multiple child colliders, returns only the first collider found.
    /// Returns [`None`] if no collider is found.
    pub fn collider(&self) -> Option<GenericCollider> {
        gg_err::log_err_then(
            self.collider_of_inner(self.this_id)
                .context("ObjectContext::collider()"),
        )
    }

    /// Returns another object's collider.
    ///
    /// If the other object has multiple child colliders, returns only the first collider found.
    /// Returns [`None`] if no collider is found.
    pub fn collider_of(&self, other: &TreeSceneObject) -> Option<GenericCollider> {
        gg_err::log_err_then(
            self.collider_of_inner(other.object_id)
                .context("ObjectContext::collider_of()"),
        )
    }
    fn collider_of_inner(&self, object_id: ObjectId) -> Result<Option<GenericCollider>> {
        // TODO: some more sensible way to handle objects with multiple child colliders.
        let children = self.children_of_inner(object_id)?;
        Ok(children.iter()
            .find_map(|scene_object| {
                // Find GgInternalCollisionShape objects, and update their transforms.
                gg_err::log_err_then_invert(scene_object.downcast_mut::<GgInternalCollisionShape>()
                    .map(|mut o| {
                        let other_absolute_transform = self.all_absolute_transforms.get(&object_id)
                            .with_context(|| format!("ObjectContext::collider_of_inner(): missing ObjectId in `all_absolute_transforms`: {object_id:?}"))?;
                        o.update_transform(*other_absolute_transform);
                        Ok(o)
                    }))
            })
            .map(|o| o.collider().clone())
            .or(children.iter().find_map(|o| {
                gg_err::log_and_ok(self.collider_of_inner(o.object_id)
                    .context("ObjectHandler::collider_of_inner(): recursive case")).flatten()
            })))
    }

    /// Adds multiple scene objects as children of this object.
    ///
    /// Takes a vector of pre-wrapped objects and adds them all as children of this object,
    /// automatically assigning new object IDs.
    ///
    /// # Parameters
    /// * `objects` - A vector of pre-wrapped scene objects to add as children
    pub fn add_vec(&mut self, objects: Vec<SceneObjectWrapper>) {
        self.object_tracker
            .pending_add
            .extend(objects.into_iter().map(|inner| TreeSceneObject {
                object_id: ObjectId::next(),
                parent_id: self.this_id,
                scene_object: inner.clone(),
            }));
    }

    /// Adds a new scene object as a sibling of this object.
    ///
    /// Creates a new scene object and adds it as a sibling with the same parent as this object.
    ///
    /// # Parameters
    /// * `object` - The scene object to add as a sibling
    pub fn add_sibling(&mut self, object: impl SceneObject) -> TreeSceneObject {
        let rv = TreeSceneObject {
            object_id: ObjectId::next(),
            parent_id: self.parent().map_or(ObjectId::root(), |obj| obj.object_id),
            scene_object: object.into_wrapper(),
        };
        self.object_tracker.pending_add.push(rv.clone());
        rv
    }

    /// Adds a new scene object as a child of this object.
    ///
    /// Creates a new scene object and adds it as a child of this object.
    /// Returns the newly created `TreeSceneObject` which contains the object's ID and wrapper.
    ///
    /// # Parameters
    /// * `object` - The scene object to add as a child
    ///
    /// # Returns
    /// The newly created `TreeSceneObject` containing the child's ID and wrapper
    pub fn add_child(&mut self, object: impl SceneObject) -> TreeSceneObject {
        let rv = TreeSceneObject {
            object_id: ObjectId::next(),
            parent_id: self.this_id,
            scene_object: object.into_wrapper(),
        };
        self.object_tracker.pending_add.push(rv.clone());
        rv
    }

    /// Removes the given scene object and all its child objects from the scene.
    ///
    /// # Parameters
    /// * `obj` - The scene object to remove
    pub fn remove(&mut self, obj: &TreeSceneObject) {
        self.object_tracker.pending_remove.insert(obj.object_id);
        if let Some(children) = self.children_of(obj).cloned() {
            for child in children {
                self.object_tracker.pending_remove.insert(child.object_id);
            }
        }
    }

    /// Removes this object and all its child objects from the scene.
    pub fn remove_this(&mut self) {
        self.object_tracker.pending_remove.insert(self.this_id);
        self.remove_children();
    }

    /// Removes all child objects of this object from the scene.
    ///
    /// This object remains in the scene.
    pub fn remove_children(&mut self) {
        for child in &self.this_children {
            self.object_tracker.pending_remove.insert(child.object_id);
        }
    }

    pub fn reparent(&mut self, target: &TreeSceneObject, new_parent: &TreeSceneObject) {
        self.reparent_inner(target.object_id, new_parent.object_id);
    }
    pub fn reparent_as_child(&mut self, target: &TreeSceneObject) {
        self.reparent_inner(target.object_id, self.this_id);
    }
    pub fn reparent_as_sibling(&mut self, target: &TreeSceneObject) {
        self.reparent_inner(
            target.object_id,
            self.parent_id().unwrap_or(ObjectId::root()),
        );
    }
    pub fn reparent_to_root(&mut self, target: &TreeSceneObject) {
        self.reparent_inner(target.object_id, ObjectId::root());
    }
    pub fn reparent_this(&mut self, new_parent: &TreeSceneObject) {
        self.reparent_inner(self.this_id, new_parent.object_id);
    }
    pub fn reparent_this_to_root(&mut self) {
        self.reparent_inner(self.this_id, ObjectId::root());
    }
    fn reparent_inner(&mut self, target_id: ObjectId, new_parent_id: ObjectId) {
        if let Some(existing_new_parent_id) = self
            .object_tracker
            .pending_move
            .insert(target_id, new_parent_id)
        {
            // TODO: verbose!
            info!(
                "ObjectContext::reparent_inner(): inserted ({:?}, {:?}) but already had {:?}",
                target_id, new_parent_id, existing_new_parent_id
            );
        }
    }

    fn test_collision_inner(
        &self,
        collider: &GenericCollider,
        other_id: ObjectId,
    ) -> Result<Option<Collision>> {
        let mut other = self
            .object_tracker
            .get(other_id)
            .with_context(|| format!("ObjectContext::test_collision_inner(): missing ObjectId in `objects`: {other_id:?}"))?
            .downcast_mut::<GgInternalCollisionShape>()
            .unwrap();
        let other_absolute_transform =
            self.all_absolute_transforms
                .get(&other_id)
                .with_context(|| {
                    format!("ObjectContext::test_collision_inner(): missing ObjectId in `all_absolute_transforms`: {other_id:?}")
                })?;
        other.update_transform(*other_absolute_transform);
        if let Some(mtv) = collider.collides_with(other.collider()) {
            let other = self
                .lookup_parent(other_id)
                .with_context(|| format!("ObjectContext::test_collision_inner(): orphaned GgInternalCollisionShape: {other_id:?}"))?
                .clone();
            Ok(Some(Collision { other, mtv }))
        } else {
            Ok(None)
        }
    }
    fn test_collision_using(
        &self,
        collider: &GenericCollider,
        listening_tags: Vec<&'static str>,
    ) -> Option<NonemptyVec<Collision>> {
        let mut rv = Vec::new();
        for tag in listening_tags {
            match self.collision_handler.get_object_ids_by_emitting_tag(tag) {
                Ok(colliding_ids) => {
                    rv.extend(colliding_ids.iter().filter_map(|other_id| {
                        gg_err::log_err_then(
                            self.test_collision_inner(collider, *other_id)
                                .context("ObjectContext::test_collision_using()"),
                        )
                    }));
                }
                Err(e) => error!("{}", e.root_cause()),
            }
        }
        NonemptyVec::try_from_vec(rv)
    }

    /// Tests for collisions between this object's collider and other objects with the given tags.
    ///
    /// # Parameters
    /// * `listening_tags` - List of tags to check collisions against
    ///
    /// # Returns
    /// * `Some(NonemptyVec<Collision>)` - Vector of collisions if any found
    /// * `None` - If no collisions found or if this object has no collider
    pub fn test_collision(
        &self,
        listening_tags: Vec<&'static str>,
    ) -> Option<NonemptyVec<Collision>> {
        self.collider()
            .and_then(|collider| self.test_collision_using(&collider, listening_tags))
    }

    /// Tests for collisions at a specific point against other objects with the given tags.
    ///
    /// Creates a 1x1 box collider at the given point and checks for collisions.
    ///
    /// # Parameters
    /// * `point` - The point to test collisions at
    /// * `listening_tags` - List of tags to check collisions against
    ///
    /// # Returns
    /// * `Some(NonemptyVec<Collision>)` - Vector of collisions if any found
    /// * `None` - If no collisions found
    pub fn test_collision_point(
        &self,
        point: Vec2,
        listening_tags: Vec<&'static str>,
    ) -> Option<NonemptyVec<Collision>> {
        self.test_collision_using(
            &BoxCollider::from_top_left(point - Vec2::one() / 2, Vec2::one()).into_generic(),
            listening_tags,
        )
    }

    /// Tests for collisions after applying an offset to this object's collider.
    ///
    /// # Parameters
    /// * `offset` - Vector offset to apply to this object's collider
    /// * `listening_tags` - List of tags to check collisions against
    ///  
    /// # Returns
    /// * `Some(NonemptyVec<Collision>)` - Vector of collisions if any found
    /// * `None` - If no collisions found or if this object has no collider
    pub fn test_collision_offset(
        &self,
        offset: Vec2,
        listening_tags: Vec<&'static str>,
    ) -> Option<NonemptyVec<Collision>> {
        self.collider().and_then(|collider| {
            self.test_collision_using(&collider.translated(offset), listening_tags)
        })
    }

    /// Tests for collisions along an axis at a given distance.
    ///
    /// Translates this object's collider along the given axis and distance, then checks for
    /// collisions. Only returns collisions where the collision normal has a non-zero dot product
    /// with the axis.
    ///
    /// # Parameters
    /// * `axis` - Unit vector defining the direction to check
    /// * `distance` - How far along the axis to project the collider
    /// * `listening_tags` - List of tags to check collisions against
    ///
    /// # Returns
    /// * `Some(NonemptyVec<Collision>)` - Vector of collisions if any found along the axis
    /// * `None` - If no collisions found or if this object has no collider
    pub fn test_collision_along(
        &self,
        axis: Vec2,
        distance: f32,
        listening_tags: Vec<&'static str>,
    ) -> Option<NonemptyVec<Collision>> {
        self.collider()
            .and_then(|collider| {
                self.test_collision_using(&collider.translated(distance * axis), listening_tags)
            })
            .and_then(|vec| {
                NonemptyVec::try_from_iter(
                    vec.into_iter()
                        .filter(|coll| coll.mtv.dot(axis).abs() > EPSILON),
                )
            })
    }
    fn lookup_parent(&self, object_id: ObjectId) -> Option<&TreeSceneObject> {
        let parent_id = self.all_parents.get(&object_id)?;
        if parent_id.is_root() {
            None
        } else {
            self.object_tracker.get(*parent_id).or_else(|| {
                error!("missing object_id in parents: {parent_id:?}");
                None
            })
        }
    }
}

/// A context object for manipulating the game viewport and rendering settings.
///
/// `ViewportContext` provides methods to:
/// - Control the viewport's position and scale
/// - Set viewport boundaries and perform clamping
/// - Change the clear color for rendering  
/// - Access viewport properties like dimensions and center
///
/// # Examples
/// ```ignore
/// // Center viewport on object
/// ctx.viewport_mut().centre_at(player_pos);
///
/// // Clamp viewport position
/// ctx.viewport_mut().clamp_to_left(Some(0.0), Some(400.0));
///
/// // Change clear color
/// ctx.viewport_mut().clear_col().set_rgba(1.0, 0.0, 0.0, 1.0);
/// ```
pub struct ViewportContext<'a> {
    viewport: &'a mut AdjustedViewport,
    clear_col: &'a mut Colour,
}

impl ViewportContext<'_> {
    /// Clamps the left edge of the viewport between optional minimum and maximum bounds.
    ///
    /// # Arguments
    /// * `min` - Optional minimum x-coordinate for the left edge
    /// * `max` - Optional maximum x-coordinate for the left edge
    ///
    /// # Examples
    /// ```ignore
    /// // Keep viewport left edge between 0 and 400 units
    /// ctx.viewport_mut().clamp_to_left(Some(0.0), Some(400.0));
    ///
    /// // Only clamp maximum position, allow unlimited leftward scrolling
    /// ctx.viewport_mut().clamp_to_left(None, Some(500.0));
    /// ```
    pub fn clamp_to_left(&mut self, min: Option<f32>, max: Option<f32>) {
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

    /// Clamps the right edge of the viewport between optional minimum and maximum bounds.
    ///
    /// # Arguments
    /// * `min` - Optional minimum x-coordinate for the right edge
    /// * `max` - Optional maximum x-coordinate for the right edge
    ///
    /// # Examples
    /// ```ignore
    /// // Keep viewport right edge between 100 and 500 units
    /// ctx.viewport_mut().clamp_to_right(Some(100.0), Some(500.0));
    ///
    /// // Only clamp minimum position, allow unlimited rightward scrolling  
    /// ctx.viewport_mut().clamp_to_right(Some(0.0), None);
    /// ```
    pub fn clamp_to_right(&mut self, min: Option<f32>, max: Option<f32>) {
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

    /// Centers the viewport on the given point.
    ///
    /// # Arguments
    /// * `centre` - The point to center the viewport on
    ///
    /// # Returns
    /// * `&mut Self` - Allows method chaining
    pub fn centre_at(&mut self, centre: Vec2) -> &mut Self {
        self.translate(centre - self.viewport.centre());
        self
    }

    /// Moves the viewport by the given delta vector.
    ///
    /// # Arguments
    /// * `delta` - Vector to translate the viewport by
    ///
    /// # Returns
    /// * `&mut Self` - Allows method chaining
    pub fn translate(&mut self, delta: Vec2) -> &mut Self {
        self.viewport.translation += delta;
        self
    }

    /// Returns a mutable reference to the clear color used for rendering.
    pub fn clear_col(&mut self) -> &mut Colour {
        self.clear_col
    }

    /// Sets the global scale factor for the viewport.
    ///
    /// # Arguments
    /// * `global_scale_factor` - New scale factor to apply to the viewport
    pub fn set_global_scale_factor(&mut self, global_scale_factor: f32) {
        self.viewport.set_global_scale_factor(global_scale_factor);
    }
}

impl AxisAlignedExtent for ViewportContext<'_> {
    fn aa_extent(&self) -> Vec2 {
        self.viewport.aa_extent()
    }

    fn centre(&self) -> Vec2 {
        self.viewport.centre()
    }
}

/// A context object provided during render operations that allows scene objects to manage their
/// render items.
///
/// A render item represents the visual appearance of an object, containing vertex data and shader
/// information needed to draw the object on screen. [`RenderContext`] provides methods to:
///
/// - Add new render items for drawing vertices and textures
/// - Update existing render items to change an object's appearance
/// - Remove render items to make objects invisible
///
/// # Examples
///
/// ```ignore
/// // Update sprite render item with new texture coordinates
/// fn on_animation_frame(&mut self, render_ctx: &mut RenderContext) {
///     let new_render_item = RenderItem {
///         vertices: self.sprite.get_vertices(),
///         texture_coords: self.next_frame_coords(),
///         depth: 0.5,
///     };
///     render_ctx.update_render_item(&new_render_item);
/// }
///
/// // Add multiple render items for composite objects
/// fn on_render(&mut self, render_ctx: &mut RenderContext) {
///     // Add base shape
///     render_ctx.insert_render_item(&self.body_render_item);
///     
///     // Add details on top
///     render_ctx.insert_render_item(&self.details_render_item);
/// }
///
/// // Remove render item to hide object
/// fn on_hide(&mut self, render_ctx: &mut RenderContext) {
///     render_ctx.remove_render_item();
/// }
/// ```
pub struct RenderContext<'a> {
    pub(crate) this_id: ObjectId,
    vertex_map: &'a mut VertexMap,
}

impl<'a> RenderContext<'a> {
    pub(crate) fn new(this_id: ObjectId, vertex_map: &'a mut VertexMap) -> Self {
        Self {
            this_id,
            vertex_map,
        }
    }

    /// Updates this object's render item by removing any existing render item and inserting the
    /// new one.
    ///
    /// # Arguments
    /// * `new_render_item` - The new render item to use for this object
    pub fn update_render_item(&mut self, new_render_item: &RenderItem) {
        self.remove_render_item();
        self.vertex_map
            .insert(self.this_id, new_render_item.clone());
    }

    /// Inserts a render item for this object, concatenating with any existing render item.
    ///
    /// If a render item already exists for this object, the new render item will be concatenated
    /// with the existing one. Otherwise, the new render item will be inserted directly.
    ///
    /// # Arguments
    /// * `new_render_item` - The render item to insert or concatenate
    pub fn insert_render_item(&mut self, new_render_item: &RenderItem) {
        if let Some(existing) = self.vertex_map.remove(self.this_id) {
            self.vertex_map
                .insert(self.this_id, existing.concat(new_render_item.clone()));
        } else {
            self.vertex_map
                .insert(self.this_id, new_render_item.clone());
        }
    }

    /// Removes this object's render item from the vertex map.
    ///
    /// Logs an error if no render item exists to remove.
    pub fn remove_render_item(&mut self) {
        if self.vertex_map.remove(self.this_id).is_none() {
            error!("removed nonexistent vertices: {:?}", self.this_id);
        }
    }
}
