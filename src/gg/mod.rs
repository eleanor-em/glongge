pub mod sample;
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
    assert::{check, check_false},
    core::{
        collision::Collider,
        input::InputHandler,
        linalg::Vec2,
        util::TimeIt,
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
    fn emit_collisions_for(&self) -> HashSet<&'static str> { [].into() }
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
    pub fn checked_downcast<T: SceneObject<ObjectType> + 'static>(&self) -> Ref<T> {
        Ref::map(self.inner.borrow(), |obj| ObjectType::checked_downcast::<T>(obj.as_ref()))
    }
    pub fn checked_downcast_mut<T: SceneObject<ObjectType> + 'static>(&self) -> RefMut<T> {
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

pub struct UpdateHandler<ObjectType: ObjectTypeEnum, RenderReceiver: RenderDataReceiver> {
    objects: BTreeMap<usize, Rc<RefCell<AnySceneObject<ObjectType>>>>,
    object_ids_by_tag: HashMap<&'static str, HashSet<usize>>,
    collision_map: HashMap<usize, HashSet<usize>>,
    vertices: BTreeMap<usize, (Range<usize>, Vec<Vec2>)>,
    render_data: BTreeMap<usize, RenderDataFull>,
    viewport: AdjustedViewport,
    render_data_receiver: Arc<Mutex<RenderReceiver>>,
    input_handler: Arc<Mutex<InputHandler>>,
    scene_instruction_tx: Sender<SceneInstruction>,
    scene_instruction_rx: Receiver<SceneInstruction>,
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
        let mut object_ids_by_tag: HashMap<&'static str, HashSet<usize>> = HashMap::new();
        for (&id, tags) in objects.iter()
                .map(|(id, obj)| (id, obj.borrow().collision_tags())) {
            for tag in tags {
                let _ = object_ids_by_tag.try_insert(tag, HashSet::new());
                object_ids_by_tag.get_mut(tag).unwrap().insert(id);
            }
        }
        let collision_map = objects.iter().filter_map(|(&id, obj)| {
            let obj = obj.borrow();
            let other_ids: HashSet<usize> = objects.iter()
                .filter(|(&other_id, _)| other_id != id)
                .filter(|(_, other_obj)| {
                    !other_obj.borrow().collision_tags()
                        .intersection(&obj.emit_collisions_for())
                        .collect::<Vec<_>>().is_empty()
                })
                .map(|(&other_id, _)| other_id)
                .collect();
            if other_ids.is_empty() {
                None
            } else {
                Some((id, other_ids))
            }
        }).collect();

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
            object_ids_by_tag,
            collision_map,
            vertices,
            render_data,
            viewport,
            render_data_receiver,
            input_handler,
            scene_instruction_tx,
            scene_instruction_rx,
        };
        rv.update_render_data(true);
        rv
    }

    pub fn consume(mut self) {
        let mut delta = Duration::from_secs(0);
        let mut total_stats = TimeIt::new("total");
        let mut update_objects_stats = TimeIt::new("on_update");
        let mut add_objects_stats = TimeIt::new("add objects");
        let mut remove_objects_stats = TimeIt::new("remove objects");
        let mut render_data_stats = TimeIt::new("render_data");
        let mut last_report = Instant::now();

        let mut is_running = true;

        loop {
            if is_running {
                let now = Instant::now();
                total_stats.start();

                let input_handler = self.input_handler.lock().unwrap().clone();

                update_objects_stats.start();
                let (pending_add_objects, pending_remove_objects) = self.call_on_update(delta, &input_handler);
                let did_update_vertices =
                    !pending_add_objects.is_empty() || !pending_remove_objects.is_empty();
                update_objects_stats.stop();

                remove_objects_stats.start();
                for remove_index in pending_remove_objects.into_iter().rev() {
                    if self.render_data.contains_key(&remove_index) {
                        check!(self.vertices.contains_key(&remove_index));
                        self.render_data.remove(&remove_index);
                        let vertices_removed = self.vertices[&remove_index].1.len();
                        for (&i, (count, _)) in self.vertices.iter_mut() {
                            if i >= remove_index {
                                *count = (count.start - vertices_removed)..(count.end - vertices_removed);
                            }
                        }
                        self.vertices.remove(&remove_index);
                    } else {
                        check_false!(self.vertices.contains_key(&remove_index));
                    }
                    self.objects.remove(&remove_index);
                    for object_ids in self.object_ids_by_tag.values_mut() {
                        object_ids.remove(&remove_index);
                    }
                    for object_ids in self.collision_map.values_mut() {
                        object_ids.remove(&remove_index);
                    }
                    self.collision_map.remove(&remove_index);
                }
                remove_objects_stats.stop();

                add_objects_stats.start();
                let mut next_id = *self.objects.keys().max().unwrap_or(&0);
                let mut next_vertex_index = self.vertices.values()
                    .map(|(count, _)| count.end)
                    .max()
                    .unwrap_or(0);
                let first_new_id = next_id + 1;
                for new_obj in pending_add_objects {
                    next_id += 1;
                    if let Some(obj) = new_obj.as_renderable_object() {
                        let new_vertices = obj.create_vertices();
                        let vertex_indices = next_vertex_index..next_vertex_index + new_vertices.len();
                        next_vertex_index += new_vertices.len();
                        self.vertices.insert(next_id, (vertex_indices.clone(), new_vertices));
                        self.render_data.insert(next_id, RenderDataFull {
                            inner: obj.render_data(),
                            transform: obj.transform(),
                            vertex_indices,
                        });
                    }
                    let mut other_ids: HashSet<usize> = HashSet::new();
                    for tag in new_obj.collision_tags().clone() {
                        for &other_id in self.object_ids_by_tag[tag].iter() {
                            other_ids.insert(other_id);
                        }
                        self.object_ids_by_tag.get_mut(tag).unwrap()
                            .insert(next_id);
                    }
                    if !other_ids.is_empty() {
                        self.collision_map.insert(next_id, other_ids);
                    }
                    self.objects.insert(next_id, Rc::new(RefCell::new(new_obj)));
                }
                // Ensure all objects actually exist before calling on_ready().
                let last_new_id = next_id;
                for i in first_new_id..=last_new_id {
                    self.objects[&i].borrow_mut().on_ready();
                }
                add_objects_stats.stop();

                render_data_stats.start();
                self.update_render_data(did_update_vertices);
                render_data_stats.stop();

                self.input_handler.lock().unwrap().update_step();
                total_stats.stop();
                if last_report.elapsed().as_secs() >= 5 {
                    info!("update stats:");
                    update_objects_stats.report_ms_if_at_least(1.0);
                    remove_objects_stats.report_ms_if_at_least(1.0);
                    add_objects_stats.report_ms_if_at_least(1.0);
                    render_data_stats.report_ms_if_at_least(1.0);
                    total_stats.report_ms();
                    last_report = Instant::now();
                }
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

    fn call_on_update(&mut self, delta: Duration, input_handler: &InputHandler) -> (Vec<Box<dyn SceneObject<ObjectType>>>, Vec<usize>) {
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

        self.iter_with_other_map(delta, input_handler, &mut pending_add_objects, &mut pending_remove_objects, &mut other_map,
                                 |mut obj, delta, update_ctx| obj.on_update_begin(delta, update_ctx));
        self.iter_with_other_map(delta, input_handler, &mut pending_add_objects, &mut pending_remove_objects, &mut other_map,
                                 |mut obj, delta, update_ctx| obj.on_update(delta, update_ctx));

        self.collision_map.iter()
            .flat_map(|(obj_id, others)| {
                others.iter().filter_map(|other_id| {
                    let collider = self.objects[obj_id].borrow().collider().unwrap();
                    let other_collider = self.objects[other_id].borrow().collider().unwrap();
                    collider.collides_with(other_collider.as_ref()).map(|mtv| {
                        let other_obj = SceneObjectWithId {
                            object_id: *other_id,
                            inner: self.objects[other_id].clone(),
                        };
                        (self.objects[obj_id].clone(), other_obj, mtv)
                    })
                })
            })
            .for_each(|(obj, other_obj, mtv)| {
                obj.borrow_mut().on_collision(other_obj, mtv)
            });

        self.iter_with_other_map(delta, input_handler, &mut pending_add_objects, &mut pending_remove_objects, &mut other_map,
                               |mut obj, delta, update_ctx| obj.on_update_end(delta, update_ctx));

        for object_id in self.objects.keys() {
            if let Some(obj) = self.objects[object_id].borrow().as_renderable_object() {
                let render_data = self.render_data.get_mut(object_id).unwrap();
                render_data.inner = obj.render_data();
                render_data.transform = obj.transform();
            }
        }
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
        let mut render_data_receiver = self.render_data_receiver.lock().unwrap();
        if did_update_vertices {
            render_data_receiver.update_vertices(self.vertices.values()
                .cloned()
                .flat_map(|(_, values)| values)
                .collect());
        }
        render_data_receiver.update_render_data(self.render_data.values().cloned().collect());
        self.viewport = render_data_receiver.current_viewport();
    }
}

#[derive(Clone)]
pub struct RenderData {
    pub colour: [f32; 4],
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
