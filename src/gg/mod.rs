pub mod render;
pub mod scene;

use std::{
    any::{Any, TypeId},
    cell::{
        Ref,
        RefCell,
        RefMut
    },
    collections::{
        BTreeMap,
        HashSet,
        HashMap
    },
    default::Default,
    fmt::Debug,
    ops::Range,
    rc::Rc,
    sync::{
        mpsc::{Receiver, Sender},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};
use num_traits::Zero;

use tracing::info;

use crate::{
    assert::{
        check,
        check_false,
        check_eq,
        check_ne
    },
    core::{
        collision::Collider,
        colour::Colour,
        input::InputHandler,
        linalg::Vec2,
        util::{
            TimeIt,
            UnorderedPair
        },
        vk_core::AdjustedViewport,
    }
};

pub trait ObjectTypeEnum: Clone + Copy + Debug + Eq + PartialEq + Sized + 'static {
    fn as_typeid(self) -> TypeId;
    fn all_values() -> Vec<Self>;
    fn checked_downcast<T: SceneObject<Self> + 'static>(obj: &dyn SceneObject<Self>) -> &T;
    fn checked_downcast_mut<T: SceneObject<Self> + 'static>(obj: &mut dyn SceneObject<Self>) -> &mut T;
}

#[derive(Copy, Clone)]
pub struct Transform {
    pub position: Vec2,
    pub rotation: f64,
}

impl Default for Transform {
    fn default() -> Self {
        Self { position: Vec2::zero(), rotation: 0.0 }
    }
}

pub trait SceneObject<ObjectType>: Send {
    fn get_type(&self) -> ObjectType;
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;

    fn on_ready(&mut self);
    #[allow(unused_variables)]
    fn on_update_begin(&mut self, delta: Duration, update_ctx: UpdateContext<ObjectType>) {}
    fn on_update(&mut self, delta: Duration, update_ctx: UpdateContext<ObjectType>);
    // TODO: probably should somehow restrict UpdateContext for on_update_begin/end().
    #[allow(unused_variables)]
    fn on_update_end(&mut self, delta: Duration, update_ctx: UpdateContext<ObjectType>) {}
    #[allow(unused_variables)]
    fn on_collision(&mut self, other: SceneObjectWithId<ObjectType>, mtv: Vec2) {}

    fn transform(&self) -> Transform;
    fn as_renderable_object(&self) -> Option<&dyn RenderableObject<ObjectType>> {
        None
    }
    fn collider(&self) -> Option<Box<dyn Collider>> { None }
    fn collision_tags(&self) -> HashSet<&'static str> { [].into() }
    fn listening_tags(&self) -> HashSet<&'static str> { [].into() }
}

pub trait RenderableObject<ObjectType>: SceneObject<ObjectType> {
    fn create_vertices(&self) -> Vec<Vec2>;
    fn render_data(&self) -> RenderData;
}

impl<ObjectType, T: SceneObject<ObjectType> + 'static> From<Box<T>> for Box<dyn SceneObject<ObjectType>> {
    fn from(value: Box<T>) -> Self { value }
}

#[derive(Clone)]
pub struct SceneObjectWithId<ObjectType> {
    object_id: usize,
    inner: Rc<RefCell<Box<dyn SceneObject<ObjectType>>>>,
}

impl<ObjectType: ObjectTypeEnum> SceneObjectWithId<ObjectType> {
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
    pub fn collider(&self) -> Option<Box<dyn Collider>> { self.inner.borrow().collider() }
}

pub struct UpdateContext<'a, ObjectType> {
    input_handler: &'a InputHandler,
    scene_instruction_tx: Sender<SceneInstruction>,
    object_id: usize,
    other_map: &'a HashMap<usize, SceneObjectWithId<ObjectType>>,
    pending_add_objects: &'a mut Vec<Box<dyn SceneObject<ObjectType>>>,
    pending_remove_objects: &'a mut Vec<usize>,
    viewport: AdjustedViewport,
}

impl<'a, ObjectType: Clone> UpdateContext<'a, ObjectType> {
    pub fn input(&self) -> &InputHandler { self.input_handler }

    pub fn others(&self) -> Vec<SceneObjectWithId<ObjectType>> {
        self.other_map
            .values()
            .filter(|obj| !self.pending_remove_objects.contains(&obj.object_id))
            .cloned()
            .collect()
    }

    pub fn add_object_vec(&mut self, mut objects: Vec<Box<dyn SceneObject<ObjectType>>>) {
        self.pending_add_objects.append(&mut objects);
    }
    pub fn add_object(&mut self, object: AnySceneObject<ObjectType>) {
        self.pending_add_objects.push(object);
    }
    pub fn remove_other_object(&mut self, obj: &SceneObjectWithId<ObjectType>) {
        self.pending_remove_objects.push(obj.object_id);
    }
    pub fn remove_this_object(&mut self) {
        self.pending_remove_objects.push(self.object_id);
    }

    pub fn scene_stop(&self) {
        self.scene_instruction_tx
            .send(SceneInstruction::Stop)
            .unwrap();
    }

    pub fn viewport(&self) -> AdjustedViewport {
        self.viewport.clone()
    }
}

#[allow(dead_code)]
pub enum SceneInstruction {
    Pause,
    Resume,
    Stop,
}

pub type AnySceneObject<ObjectType> = Box<dyn SceneObject<ObjectType>>;

struct UpdatePerfStats {
    total_stats: TimeIt,
    on_update_begin: TimeIt,
    on_update: TimeIt,
    on_update_end: TimeIt,
    detect_collision: TimeIt,
    on_collision: TimeIt,
    remove_objects: TimeIt,
    add_objects: TimeIt,
    render_data: TimeIt,
    last_report: Instant,
}

impl UpdatePerfStats {
    fn new() -> Self {
        Self {
            total_stats: TimeIt::new("total"),
            on_update_begin: TimeIt::new("on_update_begin"),
            on_update: TimeIt::new("on_update"),
            on_update_end: TimeIt::new("on_update_end"),
            detect_collision: TimeIt::new("detect collisions"),
            on_collision: TimeIt::new("on_collision"),
            remove_objects: TimeIt::new("remove objects"),
            add_objects: TimeIt::new("add objects"),
            render_data: TimeIt::new("render_data"),
            last_report: Instant::now(),
        }
    }

    fn report(&mut self) {
        if self.last_report.elapsed().as_secs() >= 5 {
            info!("update stats:");
            self.on_update_begin.report_ms_if_at_least(1.0);
            self.on_update.report_ms_if_at_least(1.0);
            self.on_update_end.report_ms_if_at_least(1.0);
            self.detect_collision.report_ms_if_at_least(1.0);
            self.on_collision.report_ms_if_at_least(1.0);
            self.remove_objects.report_ms_if_at_least(1.0);
            self.add_objects.report_ms_if_at_least(1.0);
            self.render_data.report_ms_if_at_least(1.0);
            self.total_stats.report_ms();
            self.last_report = Instant::now();
        }
    }
}

pub struct UpdateHandler<ObjectType: ObjectTypeEnum, RenderReceiver: RenderDataReceiver> {
    objects: BTreeMap<usize, Rc<RefCell<AnySceneObject<ObjectType>>>>,
    vertices: BTreeMap<usize, (Range<usize>, Vec<Vec2>)>,
    render_data: BTreeMap<usize, RenderDataFull>,
    viewport: AdjustedViewport,
    render_data_receiver: Arc<Mutex<RenderReceiver>>,
    input_handler: Arc<Mutex<InputHandler>>,
    collision_handler: CollisionHandler,
    scene_instruction_tx: Sender<SceneInstruction>,
    scene_instruction_rx: Receiver<SceneInstruction>,
    perf_stats: UpdatePerfStats,
}

impl<ObjectType: ObjectTypeEnum, RenderReceiver: RenderDataReceiver> UpdateHandler<ObjectType, RenderReceiver> {
    pub(crate) fn new(
        objects: Vec<AnySceneObject<ObjectType>>,
        render_data_receiver: Arc<Mutex<RenderReceiver>>,
        input_handler: Arc<Mutex<InputHandler>>,
        scene_instruction_tx: Sender<SceneInstruction>,
        scene_instruction_rx: Receiver<SceneInstruction>,
    ) -> Self {
        let objects: BTreeMap<usize, _> = objects.into_iter()
            .map(RefCell::new)
            .map(Rc::new)
            .enumerate().collect();
        let collision_handler = CollisionHandler::new(&objects);
        let mut vertices = BTreeMap::new();
        let mut render_data = BTreeMap::new();
        let mut vertex_index = 0;
        for (&i, obj) in objects.iter() {
            if let Some(obj) = obj.borrow().as_renderable_object() {
                let new_vertices = obj.create_vertices();
                let vertex_index_range = vertex_index..vertex_index + new_vertices.len();
                vertex_index += new_vertices.len();
                vertices.insert(i, (vertex_index_range.clone(), new_vertices));
                render_data.insert(i, RenderDataFull {
                    inner: obj.render_data(),
                    transform: obj.transform(),
                    vertex_indices: vertex_index_range,
                });
            }
        }

        let viewport = render_data_receiver.lock().unwrap().current_viewport().clone();
        let mut rv = Self {
            objects,
            vertices,
            render_data,
            viewport,
            render_data_receiver,
            input_handler,
            collision_handler,
            scene_instruction_tx,
            scene_instruction_rx,
            perf_stats: UpdatePerfStats::new(),
        };
        rv.update_render_data(true);
        rv
    }

    pub fn consume(mut self) {
        let mut delta = Duration::from_secs(0);
        let mut is_running = true;

        loop {
            if is_running {
                let now = Instant::now();
                self.perf_stats.total_stats.start();

                let input_handler = self.input_handler.lock().unwrap().clone();
                let (pending_add_objects, pending_remove_objects) = self.call_on_update(delta, input_handler);
                let did_update_vertices = !pending_add_objects.is_empty() || !pending_remove_objects.is_empty();

                self.update_with_removed_objects(pending_remove_objects);
                self.update_with_added_objects(pending_add_objects);
                self.update_render_data(did_update_vertices);
                self.input_handler.lock().unwrap().update_step();

                self.perf_stats.total_stats.stop();
                self.perf_stats.report();
                delta = now.elapsed();
            }

            match self.scene_instruction_rx.try_iter().next() {
                Some(SceneInstruction::Stop) => return,
                Some(SceneInstruction::Pause) => is_running = false,
                Some(SceneInstruction::Resume) => is_running = true,
                None => {}
            }
        }
    }

    fn update_with_removed_objects(&mut self, pending_remove_objects: Vec<usize>) {
        self.perf_stats.remove_objects.start();
        for remove_index in pending_remove_objects.into_iter().rev() {
            match self.render_data.remove(&remove_index) {
                Some(_) => {
                    check!(self.vertices.contains_key(&remove_index));
                    let vertices_removed = self.vertices[&remove_index].1.len();
                    self.vertices.iter_mut()
                        .filter(|(&i, _)| i >= remove_index)
                        .for_each(|(_, (count, _))| {
                            *count = (count.start - vertices_removed)..(count.end - vertices_removed);
                        });
                    self.vertices.remove(&remove_index);
                }
                None => {
                    check_false!(self.vertices.contains_key(&remove_index));
                }
            }
            self.objects.remove(&remove_index);
            self.collision_handler.update_with_remove_object(remove_index);
        }
        self.perf_stats.remove_objects.stop();
    }
    fn update_with_added_objects(&mut self, pending_add_objects: Vec<AnySceneObject<ObjectType>>) {
        self.perf_stats.add_objects.start();
        // TODO: leak new_id as 'static?
        let mut new_id = *self.objects.last_key_value()
            .map(|(id, _)| id)
            .unwrap_or(&0);
        let mut next_vertex_index = self.vertices.last_key_value()
            .map(|(_, (indices, _))| indices.end)
            .unwrap_or(0);
        let first_new_id = new_id + 1;

        for new_obj in pending_add_objects {
            new_id += 1;
            if let Some(obj) = new_obj.as_renderable_object() {
                let new_vertices = obj.create_vertices();
                let vertex_indices = next_vertex_index..next_vertex_index + new_vertices.len();
                next_vertex_index += new_vertices.len();
                self.vertices.insert(new_id, (vertex_indices.clone(), new_vertices));
                self.render_data.insert(new_id, RenderDataFull {
                    inner: obj.render_data(),
                    transform: obj.transform(),
                    vertex_indices,
                });
            }
            self.collision_handler.update_with_add_object(new_id, &new_obj);
            self.objects.insert(new_id, Rc::new(RefCell::new(new_obj)));
        }


        // Ensure all objects actually exist before calling on_ready().
        for i in first_new_id..=new_id {
            self.objects[&i].borrow_mut().on_ready();
        }
        self.perf_stats.add_objects.stop();
    }

    fn call_on_update(&mut self, delta: Duration, input_handler: InputHandler) -> (Vec<AnySceneObject<ObjectType>>, Vec<usize>) {
        self.perf_stats.on_update_begin.start();
        let mut pending_add_objects = Vec::new();
        let mut pending_remove_objects = Vec::new();
        let mut other_map: HashMap<usize, _> = self
            .objects
            .iter()
            .map(|(&i, obj)| (i, SceneObjectWithId {
                object_id: i,
                inner: obj.clone(),
            }))
            .collect();

        self.iter_with_other_map(delta, &input_handler, &mut pending_add_objects, &mut pending_remove_objects, &mut other_map,
                                 |mut obj, delta, update_ctx| {
                                     obj.on_update_begin(delta, update_ctx)
                                 });
        self.perf_stats.on_update_begin.stop();
        self.perf_stats.on_update.start();
        self.iter_with_other_map(delta, &input_handler, &mut pending_add_objects, &mut pending_remove_objects, &mut other_map,
                                 |mut obj, delta, update_ctx| {
                                     obj.on_update(delta, update_ctx)
                                 });
        self.perf_stats.on_update.stop();

        self.perf_stats.detect_collision.start();
        let collisions = self.collision_handler.get_collisions(&self.objects);
        self.perf_stats.detect_collision.stop();
        self.perf_stats.on_collision.start();
        for Collision { this, other, mtv } in collisions {
            this.borrow_mut().on_collision(other, mtv)
        }
        self.perf_stats.on_collision.stop();

        self.perf_stats.on_update_end.start();
        self.iter_with_other_map(delta, &input_handler, &mut pending_add_objects, &mut pending_remove_objects, &mut other_map,
                               |mut obj, delta, update_ctx| {
                                   obj.on_update_end(delta, update_ctx)
                               });
        self.perf_stats.on_update_end.stop();
        (pending_add_objects, pending_remove_objects)
    }
    fn iter_with_other_map<F>(&self,
                              delta: Duration,
                              input_handler: &InputHandler,
                              pending_add_objects: &mut Vec<AnySceneObject<ObjectType>>,
                              pending_remove_objects: &mut Vec<usize>,
                              other_map: &mut HashMap<usize, SceneObjectWithId<ObjectType>>,
                              call_obj_event: F)
    where F: Fn(RefMut<AnySceneObject<ObjectType>>, Duration, UpdateContext<ObjectType>) {
        for &object_id in self.objects.keys() {
            other_map.remove(&object_id);
            let update_ctx = UpdateContext {
                input_handler,
                scene_instruction_tx: self.scene_instruction_tx.clone(),
                object_id, other_map, pending_add_objects, pending_remove_objects,
                viewport: self.viewport.clone(),
            };
            let obj = &self.objects[&object_id];
            call_obj_event(obj.borrow_mut(), delta, update_ctx);
            other_map.insert(
                object_id,
                SceneObjectWithId {
                    object_id,
                    inner: obj.clone(),
                },
            );
        }
    }
    fn update_render_data(&mut self, did_update_vertices: bool) {
        self.perf_stats.render_data.start();
        for object_id in self.objects.keys() {
            if let Some(obj) = self.objects[object_id].borrow().as_renderable_object() {
                let render_data = self.render_data.get_mut(object_id).unwrap();
                render_data.inner = obj.render_data();
                render_data.transform = obj.transform();
            }
        }
        let mut render_data_receiver = self.render_data_receiver.lock().unwrap();
        if did_update_vertices {
            render_data_receiver.update_vertices(self.vertices.values()
                .cloned()
                .flat_map(|(_, values)| values)
                .collect());
        }
        render_data_receiver.update_render_data(self.render_data.values().cloned().collect());
        self.viewport = render_data_receiver.current_viewport();
        self.perf_stats.render_data.stop();
    }
}

struct Collision<ObjectType: ObjectTypeEnum> {
    this: Rc<RefCell<AnySceneObject<ObjectType>>>,
    other: SceneObjectWithId<ObjectType>,
    mtv: Vec2,
}

struct CollisionHandler {
    object_ids_by_tag: HashMap<&'static str, HashSet<usize>>,
    object_ids_by_listening_tag: HashMap<&'static str, HashSet<usize>>,
    possible_collisions: HashSet<UnorderedPair<usize>>,
}

impl CollisionHandler {
    pub fn new<ObjectType: ObjectTypeEnum>(
        objects: &BTreeMap<usize, Rc<RefCell<AnySceneObject<ObjectType>>>>)
    -> Self {
        let mut rv = Self {
            object_ids_by_tag: HashMap::new(),
            object_ids_by_listening_tag: HashMap::new(),
            possible_collisions: HashSet::new()
        };
        for (id, obj) in objects.iter() {
            rv.update_with_add_object(*id, &obj.borrow());
        }
        rv
    }
    pub fn update_with_add_object<ObjectType: ObjectTypeEnum>(&mut self, added_id: usize, obj: &AnySceneObject<ObjectType>) {
        let collision_tags = obj.collision_tags();
        let listening_tags = obj.listening_tags();
        let all_tags = collision_tags.union(&listening_tags).copied().collect::<HashSet<_>>();
        // Ensure all tags exist in the maps.
        check_eq!(self.object_ids_by_tag.keys().collect::<Vec<_>>(),
                  self.object_ids_by_listening_tag.keys().collect::<Vec<_>>());
        let keys = self.object_ids_by_tag.keys().copied().collect::<HashSet<_>>();
        for tag in keys.symmetric_difference(&all_tags) {
            self.object_ids_by_tag.insert(tag, HashSet::new());
            self.object_ids_by_listening_tag.insert(tag, HashSet::new());
        }
        // Add to possible_collisions.
        for tag in all_tags {
            for id in self.object_ids_by_tag.get(tag).unwrap()
                    .union(self.object_ids_by_listening_tag.get(tag).unwrap()) {
                check_ne!(*id, added_id);
                self.possible_collisions.insert(UnorderedPair::new(*id, added_id));
            }
        }
        // Update the maps.
        for (_, ids) in self.object_ids_by_tag.iter_mut()
                .filter(|(tag, _)| collision_tags.contains(*tag)) {
            ids.insert(added_id);
        }
        for (_, ids) in self.object_ids_by_listening_tag.iter_mut()
                .filter(|(tag, _)| listening_tags.contains(*tag)) {
            ids.insert(added_id);
        }
    }
    pub fn update_with_remove_object(&mut self, removed_id: usize) {
        for (_, ids) in self.object_ids_by_tag.iter_mut() {
            ids.remove(&removed_id);
        }
        for (_, ids) in self.object_ids_by_listening_tag.iter_mut() {
            ids.remove(&removed_id);
        }
        self.possible_collisions.retain(|pair| !pair.contains(&removed_id));
    }
    pub fn get_collisions<ObjectType: ObjectTypeEnum>(
        &self,
        objects: &BTreeMap<usize, Rc<RefCell<AnySceneObject<ObjectType>>>>)
    -> Vec<Collision<ObjectType>> {
        self.possible_collisions.iter().copied()
            .filter_map(|ids| {
                let this = objects[ids.fst()].borrow().collider().unwrap();
                let other = objects[ids.snd()].borrow().collider().unwrap();
                this.collides_with(other.as_ref()).map(|mtv| (ids, mtv))
            })
            .flat_map(|(ids, mtv)| {
                let this = objects[ids.fst()].clone();
                let other = objects[ids.snd()].clone();
                vec![Collision {
                    this: this.clone(),
                    other: SceneObjectWithId {
                        object_id: *ids.snd(),
                        inner: other.clone(),
                    },
                    mtv,
                }, Collision {
                    this: other,
                    other: SceneObjectWithId {
                        object_id: *ids.fst(),
                        inner: this,
                    },
                    mtv: -mtv,
                }]
            })
            .collect()
    }
}

#[derive(Clone)]
pub struct RenderData {
    pub col: Colour,
}

#[derive(Clone)]
pub struct RenderDataFull {
    inner: RenderData,
    transform: Transform,
    vertex_indices: Range<usize>,
}

pub trait RenderDataReceiver: Send {
    fn update_vertices(&mut self, vertices: Vec<Vec2>);
    fn update_render_data(&mut self, render_data: Vec<RenderDataFull>);
    fn current_viewport(&self) -> AdjustedViewport;
}
