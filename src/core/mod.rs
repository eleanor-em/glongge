use std::{
    any::{Any, TypeId},
    cell::{Ref, RefCell, RefMut},
    collections::{BTreeMap, BTreeSet},
    fmt::{Debug, Formatter},
    ops::Range,
    rc::Rc,
    sync::{
        Arc,
        mpsc,
        Mutex,
        atomic::{AtomicUsize, Ordering},
        mpsc::{Receiver, Sender}
    },
    time::{Duration, Instant}
};
use num_traits::{FromPrimitive, Zero};
use crate::{
    core::{
        collision::{Collider, NullCollider},
        colour::Colour,
        coroutine::{Coroutine, CoroutineId, CoroutineResponse, CoroutineState},
        input::InputHandler,
        linalg::{AxisAlignedExtent, Rect, Vec2, Vec2Int},
        prelude::*,
        scene::{SceneHandlerInstruction, SceneName, SceneStartInstruction},
        util::{NonemptyVec, TimeIt, UnorderedPair},
        vk_core::AdjustedViewport,
    },
    resource::{
        ResourceHandler,
        texture::{Texture, TextureId}
    }
};

pub mod collision;
pub mod colour;
pub mod input;
pub mod linalg;
pub mod util;
pub mod vk_core;
pub mod prelude;
pub mod config;
pub mod assert;
pub mod coroutine;
pub mod render;
pub mod scene;

pub trait ObjectTypeEnum: Clone + Copy + Debug + Eq + PartialEq + Sized + 'static {
    fn as_default(self) -> Box<dyn SceneObject<Self>>;
    fn as_typeid(self) -> TypeId;
    fn all_values() -> Vec<Self>;
    fn preload_all(resource_handler: &mut ResourceHandler) -> Result<()>;
    fn checked_downcast<T: SceneObject<Self> + 'static>(obj: &dyn SceneObject<Self>) -> &T;
    fn checked_downcast_mut<T: SceneObject<Self> + 'static>(obj: &mut dyn SceneObject<Self>) -> &mut T;
}

#[derive(Copy, Clone)]
pub struct Transform {
    pub centre: Vec2,
    pub rotation: f64,
    pub scale: Vec2,
}

impl Default for Transform {
    fn default() -> Self {
        Self { centre: Vec2::zero(), rotation: 0., scale: Vec2::one() }
    }
}

static NEXT_OBJECT_ID: AtomicUsize = AtomicUsize::new(0);
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct ObjectId(usize);

impl ObjectId {
    fn next() -> Self { ObjectId(NEXT_OBJECT_ID.fetch_add(1, Ordering::Relaxed)) }
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum CollisionResponse {
    Continue,
    Done,
}

pub trait SceneObject<ObjectType: ObjectTypeEnum>: Send {
    fn get_type(&self) -> ObjectType;
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;

    fn new() -> Box<Self> where Self: Sized;

    #[allow(unused_variables)]
    fn on_load(&mut self, resource_handler: &mut ResourceHandler) -> Result<Vec<VertexWithUV>> { Ok(Vec::new()) }
    #[allow(unused_variables)]
    fn on_ready(&mut self, ctx: &mut UpdateContext<ObjectType>) {}
    #[allow(unused_variables)]
    fn on_update_begin(&mut self, delta: Duration, ctx: &mut UpdateContext<ObjectType>) {}
    #[allow(unused_variables)]
    fn on_update(&mut self, delta: Duration, ctx: &mut UpdateContext<ObjectType>) {}
    // TODO: probably should somehow restrict UpdateContext for on_update_begin/end().
    #[allow(unused_variables)]
    fn on_update_end(&mut self, delta: Duration, ctx: &mut UpdateContext<ObjectType>) {}
    #[allow(unused_variables)]
    fn on_fixed_update(&mut self, ctx: &mut UpdateContext<ObjectType>) {}
    #[allow(unused_variables)]
    fn on_collision(&mut self, ctx: &mut UpdateContext<ObjectType>, other: SceneObjectWithId<ObjectType>, mtv: Vec2) -> CollisionResponse {
        CollisionResponse::Done
    }

    fn transform(&self) -> Transform;
    fn as_renderable_object(&self) -> Option<&dyn RenderableObject<ObjectType>> {
        None
    }
    fn collider(&self) -> Box<dyn Collider> { Box::new(NullCollider) }
    fn emitting_tags(&self) -> Vec<&'static str> { [].into() }
    fn listening_tags(&self) -> Vec<&'static str> { [].into() }
}

#[derive(Copy, Clone, Debug)]
pub struct VertexWithUV {
    pub vertex: Vec2,
    pub uv: Vec2,
}

impl VertexWithUV {
    pub fn from_vertex(vertex: Vec2) -> Self {
        Self { vertex, uv: Vec2::zero() }
    }

    pub fn from_vec2s<I: IntoIterator<Item=Vec2>>(vertices: I) -> Vec<Self> {
        vertices.into_iter().map(Self::from_vertex).collect()
    }
    pub fn zip_from_vec2s<I: IntoIterator<Item=Vec2>, J: IntoIterator<Item=Vec2>>(vertices: I, uvs: J) -> Vec<Self> {
        vertices.into_iter().zip(uvs)
            .map(|(vertex, uv)| Self { vertex, uv })
            .collect()
    }
}

pub trait RenderableObject<ObjectType: ObjectTypeEnum>: SceneObject<ObjectType> {
    fn render_info(&self) -> RenderInfo;
}

impl<ObjectType, T> From<Box<T>> for Box<dyn SceneObject<ObjectType>>
where
    ObjectType: ObjectTypeEnum,
    T: SceneObject<ObjectType> + 'static
{
    fn from(value: Box<T>) -> Self { value }
}

#[derive(Clone)]
pub struct SceneObjectWithId<ObjectType> {
    object_id: ObjectId,
    inner: Rc<RefCell<AnySceneObject<ObjectType>>>,
}

impl<ObjectType: ObjectTypeEnum> SceneObjectWithId<ObjectType> {
    fn new(object_id: ObjectId, obj: Rc<RefCell<AnySceneObject<ObjectType>>>) -> Self {
        Self { object_id, inner: obj }
    }

    pub fn get_type(&self) -> ObjectType { self.inner.borrow().get_type() }
    pub fn downcast<T: SceneObject<ObjectType> + 'static>(&self) -> Option<Ref<T>> {
        Ref::filter_map(self.inner.borrow(), |obj| {
            obj.as_any().downcast_ref::<T>()
        }).ok()
    }
    pub fn downcast_mut<T: SceneObject<ObjectType> + 'static>(&mut self) -> Option<RefMut<T>> {
        RefMut::filter_map(self.inner.borrow_mut(), |obj| {
            obj.as_any_mut().downcast_mut::<T>()
        }).ok()
    }
    // TODO: the below may turn out to still be useful.
    #[allow(dead_code)]
    fn checked_downcast<T: SceneObject<ObjectType> + 'static>(&self) -> Ref<T> {
        Ref::map(self.inner.borrow(), |obj| ObjectType::checked_downcast::<T>(obj.as_ref()))
    }
    #[allow(dead_code)]
    fn checked_downcast_mut<T: SceneObject<ObjectType> + 'static>(&self) -> RefMut<T> {
        RefMut::map(self.inner.borrow_mut(), |obj| ObjectType::checked_downcast_mut::<T>(obj.as_mut()))
    }

    pub fn transform(&self) -> Transform { self.inner.borrow().transform() }
    pub fn collider(&self) -> Box<dyn Collider> { self.inner.borrow().collider() }
    pub fn emitting_tags(&self) -> Vec<&'static str> { self.inner.borrow().emitting_tags() }
    pub fn listening_tags(&self) -> Vec<&'static str> { self.inner.borrow().listening_tags() }
}

pub struct SceneContext<'a, ObjectType: ObjectTypeEnum> {
    scene_instruction_tx: Sender<SceneInstruction>,
    scene_name: SceneName,
    scene_data: &'a mut Vec<u8>,
    coroutines: &'a mut BTreeMap<CoroutineId, Coroutine<ObjectType>>,
    pending_removed_coroutines: BTreeSet<CoroutineId>,
}

impl<'a, ObjectType: ObjectTypeEnum> SceneContext<'a, ObjectType> {
    pub fn stop(&self) {
        self.scene_instruction_tx.send(SceneInstruction::Stop(self.scene_data.clone())).unwrap();
    }
    pub fn goto(&self, instruction: SceneStartInstruction) {
        self.scene_instruction_tx.send(SceneInstruction::Goto(instruction, self.scene_data.clone())).unwrap();
    }
    pub fn name(&self) -> SceneName { self.scene_name }
    pub fn data(&mut self) -> &mut Vec<u8> { self.scene_data }

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
        self.start_coroutine(move |this, update_ctx, action| {
            match action {
                CoroutineState::Starting => CoroutineResponse::Wait(duration),
                _ => func(this, update_ctx, action)
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

struct ObjectTracker<ObjectType: ObjectTypeEnum> {
    last: BTreeMap<ObjectId, Rc<RefCell<AnySceneObject<ObjectType>>>>,
    pending_add: Vec<AnySceneObject<ObjectType>>,
    pending_remove: BTreeSet<ObjectId>,
}

impl<ObjectType: ObjectTypeEnum> ObjectTracker<ObjectType> {
    fn into_pending(self) -> (Vec<AnySceneObject<ObjectType>>, BTreeSet<ObjectId>) {
        (self.pending_add, self.pending_remove)
    }
}

pub struct ObjectContext<'a, ObjectType: ObjectTypeEnum> {
    collision_handler: &'a CollisionHandler,
    this: SceneObjectWithId<ObjectType>,
    object_tracker: Option<&'a mut ObjectTracker<ObjectType>>,
}

impl<'a, ObjectType: ObjectTypeEnum> ObjectContext<'a, ObjectType> {
    fn unwrap_tracker(&self) -> &ObjectTracker<ObjectType> {
        self.object_tracker.as_ref()
            .expect("can only use within on_update(), on_update_begin(), on_update_end(), \
                    on_fixed_update(), or coroutines")
    }
    fn unwrap_tracker_mut(&mut self) -> &mut ObjectTracker<ObjectType> {
        self.object_tracker.as_mut()
            .expect("can only use within on_update(), on_update_begin(), on_update_end(), \
                    on_fixed_update(), or coroutines")
    }
    pub fn others(&self) -> Vec<SceneObjectWithId<ObjectType>> {
        let tracker = self.unwrap_tracker();
        tracker.last.iter()
            .filter(|(object_id, _)| !tracker.pending_remove.contains(object_id))
            .filter(|(object_id, _)| self.this.object_id != **object_id)
            .map(|(object_id, obj)| SceneObjectWithId::new(*object_id, obj.clone()))
            .collect()
    }

    pub fn add_vec(&mut self, objects: Vec<AnySceneObject<ObjectType>>) {
        let tracker = self.unwrap_tracker_mut();
        tracker.pending_add.extend(objects);
    }
    pub fn add(&mut self, object: AnySceneObject<ObjectType>) {
        let tracker = self.unwrap_tracker_mut();
        tracker.pending_add.push(object);
    }
    pub fn remove(&mut self, obj: &SceneObjectWithId<ObjectType>) {
        let tracker = self.unwrap_tracker_mut();
        tracker.pending_remove.insert(obj.object_id);
    }
    pub fn remove_this(&mut self) {
        let this_id = self.this.object_id;
        let tracker = self.unwrap_tracker_mut();
        tracker.pending_remove.insert(this_id);
    }
    pub fn test_collision(&self,
                          collider: &dyn Collider,
                          listening_tags: Vec<&'static str>
    ) -> Option<NonemptyVec<Collision<ObjectType>>> {
        let mut rv = Vec::new();
        for tag in listening_tags {
            for other_id in self.collision_handler.object_ids_by_emitting_tag.get(tag).unwrap() {
                if let Some(other) = self.unwrap_tracker().last
                    .get(other_id) {
                    if let Some(mtv) = collider.collides_with(other.borrow().collider().as_ref()) {
                        rv.push(Collision {
                            other: SceneObjectWithId::new(*other_id, other.clone()),
                            mtv,
                        });
                    }
                }
            }
        }
        NonemptyVec::try_from_vec(rv)
    }
    pub fn test_collision_along(&self,
                                mut collider: Box<dyn Collider>,
                                tags: Vec<&'static str>,
                                axis: Vec2,
                                distance: f64,
    ) -> Option<NonemptyVec<Collision<ObjectType>>> {
        self.test_collision(collider.translate(distance * axis), tags)
            .and_then(|vec| {
                NonemptyVec::try_from_iter(vec
                    .into_iter()
                    .filter(|coll| !coll.mtv.dot(axis).is_zero()))
            })
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

pub struct UpdateContext<'a, ObjectType: ObjectTypeEnum> {
    input: &'a InputHandler,
    scene: SceneContext<'a, ObjectType>,
    object: ObjectContext<'a, ObjectType>,
    viewport: ViewportContext<'a>,
}

impl<'a, ObjectType: ObjectTypeEnum> UpdateContext<'a, ObjectType> {
    fn new<R: RenderInfoReceiver>(
        caller: &'a mut UpdateHandler<ObjectType, R>,
        input_handler: &'a InputHandler,
        this: SceneObjectWithId<ObjectType>
    ) -> Self {
        Self {
            input: input_handler,
            scene: SceneContext {
                scene_instruction_tx: caller.scene_instruction_tx.clone(),
                scene_name: caller.scene_name,
                scene_data: &mut caller.scene_data,
                coroutines: caller.coroutines.entry(this.object_id).or_default(),
                pending_removed_coroutines: BTreeSet::new(),
            },
            object: ObjectContext {
                collision_handler: &caller.collision_handler,
                this,
                object_tracker: None,
            },
            viewport: ViewportContext {
                viewport: &mut caller.viewport,
                clear_col: &mut caller.clear_col,
            },
        }
    }
    fn new_with_objects<R: RenderInfoReceiver>(
        caller: &'a mut UpdateHandler<ObjectType, R>,
        input_handler: &'a InputHandler,
        this: SceneObjectWithId<ObjectType>,
        object_tracker: &'a mut ObjectTracker<ObjectType>
    ) -> Self {
        Self {
            input: input_handler,
            scene: SceneContext {
                scene_instruction_tx: caller.scene_instruction_tx.clone(),
                scene_name: caller.scene_name,
                scene_data: &mut caller.scene_data,
                coroutines: caller.coroutines.entry(this.object_id).or_default(),
                pending_removed_coroutines: BTreeSet::new(),
            },
            object: ObjectContext {
                collision_handler: &caller.collision_handler,
                this,
                object_tracker: Some(object_tracker),
            },
            viewport: ViewportContext {
                viewport: &mut caller.viewport,
                clear_col: &mut caller.clear_col,
            },
        }
    }

    pub fn object(&mut self) -> &mut ObjectContext<'a, ObjectType> { &mut self.object }
    pub fn scene(&mut self) -> &mut SceneContext<'a, ObjectType> { &mut self.scene }
    pub fn viewport(&mut self) -> &mut ViewportContext<'a> { &mut self.viewport }
    pub fn input(&self) -> &InputHandler { self.input }

    fn apply_coroutines(mut self) {
        for id in &self.scene.pending_removed_coroutines {
            self.scene.coroutines.remove(id);
        }
        self.scene.pending_removed_coroutines.clear();
    }
}

impl<ObjectType: ObjectTypeEnum> Drop for UpdateContext<'_, ObjectType> {
    fn drop(&mut self) {
        // pending_removed_coroutines must be used. Call apply_coroutines() before dropping.
        check!(self.scene.pending_removed_coroutines.is_empty());
    }
}

#[allow(dead_code)]
enum SceneInstruction {
    Pause,
    Resume,
    Stop(Vec<u8>),
    Goto(SceneStartInstruction, Vec<u8>),
}

pub type AnySceneObject<ObjectType> = Box<dyn SceneObject<ObjectType>>;

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
            total_stats: TimeIt::new("total"),
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
            info!("update stats:");
            self.on_update_begin.report_ms_if_at_least(1.);
            self.coroutines.report_ms_if_at_least(1.);
            self.on_update.report_ms_if_at_least(1.);
            self.on_update_end.report_ms_if_at_least(1.);
            self.fixed_update.report_ms_if_at_least(1.);
            self.detect_collision.report_ms_if_at_least(1.);
            self.on_collision.report_ms_if_at_least(1.);
            self.remove_objects.report_ms_if_at_least(1.);
            self.add_objects.report_ms_if_at_least(1.);
            self.render_infos.report_ms_if_at_least(1.);
            self.total_stats.report_ms();
            self.last_report = Instant::now();
        }
    }
}

pub(crate) struct UpdateHandler<ObjectType: ObjectTypeEnum, RenderReceiver: RenderInfoReceiver> {
    objects: BTreeMap<ObjectId, Rc<RefCell<AnySceneObject<ObjectType>>>>,
    vertices: BTreeMap<ObjectId, (Range<usize>, Vec<VertexWithUV>)>,
    vertex_count: usize, // saves allocation time in update_and_send_render_infos()
    render_infos: BTreeMap<ObjectId, RenderInfoFull>,
    coroutines: BTreeMap<ObjectId, BTreeMap<CoroutineId, Coroutine<ObjectType>>>,
    viewport: AdjustedViewport,
    input_handler: Arc<Mutex<InputHandler>>,
    resource_handler: ResourceHandler,
    render_info_receiver: Arc<Mutex<RenderReceiver>>,
    clear_col: Colour,
    collision_handler: CollisionHandler,
    scene_instruction_tx: Sender<SceneInstruction>,
    scene_instruction_rx: Receiver<SceneInstruction>,
    scene_name: SceneName,
    scene_data: Vec<u8>,
    perf_stats: UpdatePerfStats,
}

impl<ObjectType: ObjectTypeEnum, RenderReceiver: RenderInfoReceiver> UpdateHandler<ObjectType, RenderReceiver> {
    pub(crate) fn new(
        objects: Vec<AnySceneObject<ObjectType>>,
        input_handler: Arc<Mutex<InputHandler>>,
        mut resource_handler: ResourceHandler,
        render_info_receiver: Arc<Mutex<RenderReceiver>>,
        scene_name: SceneName,
        scene_data: Vec<u8>
    ) -> Result<Self> {
        let mut objects = objects.into_iter()
            .map(|obj| (ObjectId::next(), obj))
            .collect::<BTreeMap<_, _>>();
        let collision_handler = CollisionHandler::new(&objects);

        // Call on_load().
        let mut vertices = BTreeMap::new();
        let mut render_infos = BTreeMap::new();
        let mut vertex_index = 0;
        for (&i, obj) in &mut objects {
            let new_vertices = obj.on_load(&mut resource_handler)?;
            if let Some(obj) = obj.as_renderable_object() {
                let vertex_index_range = vertex_index..vertex_index + new_vertices.len();
                vertex_index += new_vertices.len();
                vertices.insert(i, (vertex_index_range.clone(), new_vertices));
                render_infos.insert(i, RenderInfoFull {
                    inner: obj.render_info(),
                    transform: obj.transform(),
                    vertex_indices: vertex_index_range,
                });
            }
        }

        let viewport = render_info_receiver.lock().unwrap().current_viewport().clone();
        let objects = objects.into_iter()
            .map(|(id, obj)| (id, Rc::new(RefCell::new(obj))))
            .collect();
        let (scene_instruction_tx, scene_instruction_rx) = mpsc::channel();
        let mut rv = Self {
            objects,
            vertices,
            vertex_count: vertex_index,
            render_infos,
            viewport,
            input_handler,
            resource_handler,
            render_info_receiver,
            clear_col: Colour::black(),
            collision_handler,
            coroutines: BTreeMap::new(),
            scene_instruction_tx,
            scene_instruction_rx,
            scene_name,
            scene_data,
            perf_stats: UpdatePerfStats::new(),
        };

        // Call on_ready().
        let input_handler = rv.input_handler.lock().unwrap().clone();
        for (this_id, this) in rv.objects.clone() {
            let mut ctx = UpdateContext::new(
                &mut rv,
                &input_handler,
                SceneObjectWithId::new(this_id, this.clone()),
            );
            this.borrow_mut().on_ready(&mut ctx);
            ctx.apply_coroutines();
        }
        rv.update_and_send_render_infos(true);
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
                    self.call_on_update(delta, &input_handler, fixed_updates);
                let did_update_vertices = !pending_add_objects.is_empty() || !pending_remove_objects.is_empty();

                self.update_with_removed_objects(pending_remove_objects);
                self.update_with_added_objects(&input_handler, pending_add_objects)?;
                self.update_and_send_render_infos(did_update_vertices);
                self.input_handler.lock().unwrap().update_step();

                self.perf_stats.total_stats.stop();
                self.perf_stats.report();
                delta = now.elapsed();
            }

            match self.scene_instruction_rx.try_iter().next() {
                Some(SceneInstruction::Stop(scene_data)) => {
                    return Ok(SceneHandlerInstruction::SaveAndExit(scene_data));
                },
                Some(SceneInstruction::Goto(instruction, scene_data)) => {
                    return Ok(SceneHandlerInstruction::SaveAndGoto(instruction, scene_data));
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

    fn update_with_removed_objects(&mut self, pending_remove_objects: BTreeSet<ObjectId>) {
        self.perf_stats.remove_objects.start();
        self.collision_handler.update_with_removed_objects(&pending_remove_objects);
        for remove_index in pending_remove_objects.into_iter().rev() {
            if self.render_infos.remove(&remove_index).is_some() {
                // Update vertex indices.
                // This is technically quadratic, but is fast enough.
                check!(self.vertices.contains_key(&remove_index));
                let vertices_removed = self.vertices[&remove_index].1.len();
                self.vertices.remove(&remove_index);
                let indices = self.vertices.keys()
                    .copied()
                    .filter(|&i| i > remove_index)
                    .collect_vec();
                for i in indices {
                    let count = &mut self.vertices.get_mut(&i)
                        .unwrap_or_else(|| panic!("missing ObjectId in self.vertices: {i:?}"))
                        .0;
                    *count = (count.start - vertices_removed)..(count.end - vertices_removed);
                    let count = &mut self.render_infos.get_mut(&i)
                        .unwrap_or_else(|| panic!("missing ObjectId in self.render_infos: {i:?}"))
                        .vertex_indices;
                    *count = (count.start - vertices_removed)..(count.end - vertices_removed);
                }
                self.vertex_count -= vertices_removed;
            } else {
                check_false!(self.vertices.contains_key(&remove_index));
            }
            self.objects.remove(&remove_index);
            self.coroutines.remove(&remove_index);
        }
        self.perf_stats.remove_objects.stop();
    }
    fn update_with_added_objects(&mut self, input_handler: &InputHandler, pending_add_objects: Vec<AnySceneObject<ObjectType>>) -> Result<()> {
        self.perf_stats.add_objects.start();
        if !pending_add_objects.is_empty() {
            let pending_add_objects = pending_add_objects.into_iter()
                .map(|obj| (ObjectId::next(), obj))
                .collect::<BTreeMap<ObjectId, _>>();
            let first_new_id = *pending_add_objects.first_key_value()
                .expect("pending_add_objects empty?")
                .0;
            let last_new_id = *pending_add_objects.last_key_value()
                .expect("pending_add_objects empty?")
                .0;
            self.collision_handler.update_with_added_objects(&pending_add_objects);

            // Call on_load().
            let mut next_vertex_index = self.vertices.last_key_value()
                .map_or(0, |(_, (indices, _))| indices.end);
            for (new_id, mut new_obj) in pending_add_objects {
                let new_vertices = new_obj.on_load(&mut self.resource_handler)?;
                if let Some(obj) = new_obj.as_renderable_object() {
                    let vertex_indices = next_vertex_index..next_vertex_index + new_vertices.len();
                    self.vertex_count += new_vertices.len();
                    next_vertex_index += new_vertices.len();
                    self.vertices.insert(new_id, (vertex_indices.clone(), new_vertices));
                    self.render_infos.insert(new_id, RenderInfoFull {
                        inner: obj.render_info(),
                        transform: obj.transform(),
                        vertex_indices,
                    });
                }
                self.objects.insert(new_id, Rc::new(RefCell::new(new_obj)));
            }

            // Call on_ready().
            for i in first_new_id.0..=last_new_id.0 {
                let this_id = ObjectId(i);
                let this = self.objects.get_mut(&this_id)
                    .unwrap_or_else(|| panic!("tried to call on_ready() for nonexistent added object: {this_id:?}"))
                    .clone();
                let mut ctx = UpdateContext::new(
                    self,
                    input_handler,
                    SceneObjectWithId::new(this_id, this.clone())
                );
                this.borrow_mut().on_ready(&mut ctx);
                ctx.apply_coroutines();
            }
        }
        self.perf_stats.add_objects.stop();
        Ok(())
    }

    fn call_on_update(&mut self,
                      delta: Duration,
                      input_handler: &InputHandler,
                      mut fixed_updates: u128
    ) -> (Vec<AnySceneObject<ObjectType>>, BTreeSet<ObjectId>) {
        self.perf_stats.on_update_begin.start();
        let mut object_tracker = ObjectTracker {
            last: self.objects.clone(),
            pending_add: Vec::new(),
            pending_remove: BTreeSet::new()
        };

        self.iter_with_other_map(delta, input_handler, &mut object_tracker,
                                 |mut obj, delta, update_ctx| {
                                     obj.on_update_begin(delta, update_ctx);
                                 });
        self.perf_stats.on_update_begin.stop();
        self.perf_stats.coroutines.start();
        self.update_coroutines(input_handler, &mut object_tracker);
        self.perf_stats.coroutines.stop();
        self.perf_stats.on_update.start();
        self.iter_with_other_map(delta, input_handler, &mut object_tracker,
                                 |mut obj, delta, update_ctx| {
                                     obj.on_update(delta, update_ctx);
                                 });
        self.perf_stats.on_update.stop();

        for _ in 0..fixed_updates.min(MAX_FIXED_UPDATES) {
            self.iter_with_other_map(delta, input_handler, &mut object_tracker,
                                     |mut obj, _delta, update_ctx| {
                                         obj.on_fixed_update(update_ctx);
                                     });
            fixed_updates -= 1;
            // Detect collisions after each fixed update: important to prevent glitching through walls etc.
            self.handle_collisions(input_handler);
        }

        self.perf_stats.on_update_end.start();
        self.iter_with_other_map(delta, input_handler, &mut object_tracker,
                                 |mut obj, delta, update_ctx| {
                                     obj.on_update_end(delta, update_ctx);
                                 });
        self.perf_stats.on_update_end.stop();
        object_tracker.into_pending()
    }

    fn handle_collisions(&mut self, input_handler: &InputHandler) {
        self.perf_stats.detect_collision.start();
        let collisions = self.collision_handler.get_collisions(&self.objects);
        self.perf_stats.detect_collision.stop();
        self.perf_stats.on_collision.start();
        let mut done_with_collisions = BTreeSet::new();
        for CollisionNotification { this, other, mtv } in collisions {
            if !done_with_collisions.contains(&this.object_id) {
                let mut ctx = UpdateContext::new(
                    self,
                    input_handler,
                    this.clone()
                );
                match this.inner.borrow_mut().on_collision(&mut ctx, other, mtv) {
                    CollisionResponse::Continue => {},
                    CollisionResponse::Done => { done_with_collisions.insert(this.object_id); },
                }
                ctx.apply_coroutines();
            }
        }
        self.perf_stats.on_collision.stop();
    }

    fn update_coroutines(&mut self,
                         input_handler: &InputHandler,
                         object_tracker: &mut ObjectTracker<ObjectType>
    ) {
        for (this_id, this) in self.objects.clone() {
            let this = SceneObjectWithId::new(this_id, this.clone());
            let last_coroutines = self.coroutines.remove(&this_id).unwrap_or_default();
            for (id, coroutine) in last_coroutines {
                let mut ctx = UpdateContext::new_with_objects(
                    self,
                    input_handler,
                    this.clone(),
                    object_tracker
                );
                if let Some(coroutine) = coroutine.resume(this.clone(), &mut ctx) {
                    ctx.scene.coroutines.insert(id, coroutine);
                }
                ctx.apply_coroutines();
            }
        }
    }
    fn iter_with_other_map<F>(&mut self,
                              delta: Duration,
                              input_handler: &InputHandler,
                              object_tracker: &mut ObjectTracker<ObjectType>,
                              call_obj_event: F)
    where F: Fn(RefMut<AnySceneObject<ObjectType>>, Duration, &mut UpdateContext<ObjectType>) {
        for (this_id, this) in self.objects.clone() {
            let this = SceneObjectWithId::new(this_id, this.clone());
            let mut ctx = UpdateContext::new_with_objects(
                self,
                input_handler,
                this.clone(),
                object_tracker
            );
            call_obj_event(this.inner.borrow_mut(), delta, &mut ctx);
            ctx.apply_coroutines();
        }
    }
    fn update_and_send_render_infos(&mut self, did_update_vertices: bool) {
        self.perf_stats.render_infos.start();
        self.update_render_infos();
        self.send_render_infos(did_update_vertices);
        self.perf_stats.render_infos.stop();
    }
    fn update_render_infos(&mut self) {
        for (object_id, obj) in &self.objects {
            if let Some(obj) = obj.borrow().as_renderable_object() {
                let render_info = self.render_infos.get_mut(object_id)
                    .unwrap_or_else(|| panic!("missing object_id in render_info: {object_id:?}"));
                render_info.inner = obj.render_info();
                render_info.transform = obj.transform();
                render_info.transform.centre -= self.viewport.translation;
            }
        }
    }
    fn send_render_infos(&mut self, did_update_vertices: bool) {
        let mut render_info_receiver = self.render_info_receiver.lock().unwrap();
        if did_update_vertices {
            let mut vertices = Vec::with_capacity(self.vertex_count);
            for (_, values) in self.vertices.values() {
                vertices.extend(values);
            }
            check_eq!(vertices.len(), vertices.capacity());
            check_eq!(vertices.capacity(), self.vertex_count);
            render_info_receiver.update_vertices(vertices);
        }
        render_info_receiver.update_render_info(self.render_infos.values().cloned().collect());
        render_info_receiver.set_clear_col(self.clear_col);
        self.viewport = render_info_receiver.current_viewport()
            .translated(self.viewport.translation);
    }

}

#[derive(Clone)]
pub struct Collision<ObjectType: ObjectTypeEnum> {
    pub other: SceneObjectWithId<ObjectType>,
    pub mtv: Vec2,
}

impl<ObjectType: ObjectTypeEnum> Debug for Collision<ObjectType> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "({:?} at {}, mtv={})", self.other.object_id, self.other.transform().centre, self.mtv)
    }
}

struct CollisionNotification<ObjectType: ObjectTypeEnum> {
    this: SceneObjectWithId<ObjectType>,
    other: SceneObjectWithId<ObjectType>,
    mtv: Vec2,
}

struct CollisionHandler {
    object_ids_by_emitting_tag: BTreeMap<&'static str, BTreeSet<ObjectId>>,
    object_ids_by_listening_tag: BTreeMap<&'static str, BTreeSet<ObjectId>>,
    possible_collisions: BTreeSet<UnorderedPair<ObjectId>>,
}

impl CollisionHandler {
    fn new<ObjectType: ObjectTypeEnum>(
        objects: &BTreeMap<ObjectId, AnySceneObject<ObjectType>>)
        -> Self {
        let mut rv = Self {
            object_ids_by_emitting_tag: BTreeMap::new(),
            object_ids_by_listening_tag: BTreeMap::new(),
            possible_collisions: BTreeSet::new()
        };
        rv.update_with_added_objects(objects);
        rv
    }
    fn update_with_added_objects<ObjectType: ObjectTypeEnum>(&mut self, added_objects: &BTreeMap<ObjectId, AnySceneObject<ObjectType>>) {
        let mut new_object_ids_by_emitting_tag = BTreeMap::<&'static str, Vec<ObjectId>>::new();
        let mut new_object_ids_by_listening_tag = BTreeMap::<&'static str, Vec<ObjectId>>::new();

        for (id, obj) in added_objects {
            for tag in obj.emitting_tags() {
                new_object_ids_by_emitting_tag.entry(tag).or_default().push(*id);
                new_object_ids_by_listening_tag.entry(tag).or_default();
            }
            for tag in obj.listening_tags() {
                new_object_ids_by_emitting_tag.entry(tag).or_default();
                new_object_ids_by_listening_tag.entry(tag).or_default().push(*id);
            }
        }
        for tag in new_object_ids_by_emitting_tag.keys() {
            self.object_ids_by_emitting_tag.entry(tag).or_default().extend(
                new_object_ids_by_emitting_tag.get(tag).unwrap());
        }
        for tag in new_object_ids_by_listening_tag.keys() {
            self.object_ids_by_listening_tag.entry(tag).or_default().extend(
                new_object_ids_by_listening_tag.get(tag).unwrap());
        }

        for (tag, emitters) in new_object_ids_by_emitting_tag {
            let listeners = self.object_ids_by_listening_tag.get(tag)
                .unwrap_or_else(|| panic!("object_ids_by_listening tag missing tag: {tag}"));
            let new_possible_collisions = emitters.into_iter().cartesian_product(listeners.iter())
                .filter_map(|(emitter, listener)| UnorderedPair::new_distinct(emitter, *listener));
            self.possible_collisions.extend(new_possible_collisions);
        }
        for (tag, listeners) in new_object_ids_by_listening_tag {
            let emitters = self.object_ids_by_emitting_tag.get(tag)
                .unwrap_or_else(|| panic!("object_ids_by_tag tag missing tag: {tag}"));
            let new_possible_collisions = emitters.iter().cartesian_product(listeners.into_iter())
                .filter_map(|(emitter, listener)| UnorderedPair::new_distinct(*emitter, listener));
            self.possible_collisions.extend(new_possible_collisions);
        }
    }
    fn update_with_removed_objects(&mut self, removed_ids: &BTreeSet<ObjectId>) {
        for ids in self.object_ids_by_emitting_tag.values_mut() {
            ids.retain(|id| !removed_ids.contains(id));
        }
        for ids in self.object_ids_by_listening_tag.values_mut() {
            ids.retain(|id| !removed_ids.contains(id));
        }
        self.possible_collisions.retain(|pair| !removed_ids.contains(&pair.fst()) && !removed_ids.contains(&pair.snd()));
    }
    fn get_collisions<ObjectType: ObjectTypeEnum>(
        &self,
        objects: &BTreeMap<ObjectId, Rc<RefCell<AnySceneObject<ObjectType>>>>
    ) -> Vec<CollisionNotification<ObjectType>> {
        let collisions = self.get_collisions_inner(objects);
        let mut rv = Vec::with_capacity(collisions.len() * 2);
        for (ids, mtv) in collisions {
            let this = SceneObjectWithId::new(ids.fst(), objects[&ids.fst()].clone());
            let (this_listening, this_emitting) = {
                let this = this.inner.borrow();
                (this.listening_tags(), this.emitting_tags())
            };
            let other = SceneObjectWithId::new(ids.snd(), objects[&ids.snd()].clone());
            let (other_listening, other_emitting) = {
                let other = other.inner.borrow();
                (other.listening_tags(), other.emitting_tags())
            };
            if !this_listening.into_iter().chain(other_emitting).all_unique() {
                rv.push(CollisionNotification {
                    this: this.clone(),
                    other: other.clone(),
                    mtv,
                });
            };
            if !other_listening.into_iter().chain(this_emitting).all_unique() {
                rv.push(CollisionNotification {
                    this: other,
                    other: this,
                    mtv: -mtv,
                });
            }
        }
        rv
    }

    fn get_collisions_inner<ObjectType: ObjectTypeEnum>(
        &self,
        objects: &BTreeMap<ObjectId, Rc<RefCell<AnySceneObject<ObjectType>>>>
    ) -> Vec<(UnorderedPair<ObjectId>, Vec2)> {
        self.possible_collisions.iter()
            .filter_map(|ids| {
                let this = objects[&ids.fst()].borrow().collider();
                let other = objects[&ids.snd()].borrow().collider();
                this.collides_with(other.as_ref()).map(|mtv| (*ids, mtv))
            })
            .collect()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TextureSubArea {
    rect: Rect,
}

impl TextureSubArea {
    pub fn new(centre: Vec2Int, half_widths: Vec2Int) -> Self {
        Self::from_rect(Rect::new(centre.into(), half_widths.into()))
    }
    pub fn from_rect(rect: Rect) -> Self {
        Self { rect }
    }

    pub(crate) fn uv(&self, texture: &Texture, raw_uv: Vec2) -> Vec2 {
        if self.rect == Rect::default() {
            raw_uv
        } else {
            let extent = texture.extent();
            let u0 = self.rect.top_left().x / extent.x;
            let v0 = self.rect.top_left().y / extent.y;
            let u1 = self.rect.bottom_right().x / extent.x;
            let v1 = self.rect.bottom_right().y / extent.y;
            Vec2 { x: linalg::lerp(u0, u1, raw_uv.x), y: linalg::lerp(v0, v1, raw_uv.y) }
        }
    }
}

impl AxisAlignedExtent for TextureSubArea {
    fn aa_extent(&self) -> Vec2 {
        self.rect.aa_extent()
    }

    fn centre(&self) -> Vec2 {
        self.rect.centre()
    }
}

#[derive(Clone)]
pub struct RenderInfo {
    pub col: Colour,
    pub texture_id: Option<TextureId>,
    pub texture_sub_area: TextureSubArea,
}

impl Default for RenderInfo {
    fn default() -> Self {
        Self { col: Colour::white(), texture_id: None, texture_sub_area: TextureSubArea::default() }
    }
}

#[derive(Clone)]
pub struct RenderInfoFull {
    pub(crate) inner: RenderInfo,
    pub(crate) transform: Transform,
    pub(crate) vertex_indices: Range<usize>,
}

pub trait RenderInfoReceiver: Clone + Send {
    fn update_vertices(&mut self, vertices: Vec<VertexWithUV>);
    fn update_render_info(&mut self, render_info: Vec<RenderInfoFull>);
    fn current_viewport(&self) -> AdjustedViewport;
    fn is_ready(&self) -> bool;
    fn get_clear_col(&self) -> Colour;
    fn set_clear_col(&mut self, col: Colour);
}
