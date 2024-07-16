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
        BTreeSet,
    },
    default::Default,
    fmt::Debug,
    ops::{
        Add,
        Range,
    },
    rc::Rc,
    sync::{
        mpsc::{Receiver, Sender},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};
use itertools::Itertools;
use num_traits::Zero;

use anyhow::{anyhow, Result};
use tracing::info;

use crate::{
    assert::*,
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
    },
    resource::{
        ResourceHandler,
        texture::TextureId
    }
};
use crate::core::linalg;
use crate::core::linalg::{Rect, Vec2Int};
use crate::resource::texture::Texture;

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

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct ObjectId(usize);

impl ObjectId {
    fn next(&self) -> Self { ObjectId(self.0 + 1) }
}

impl Add<usize> for ObjectId {
    type Output = Self;

    fn add(self, rhs: usize) -> Self::Output {
        Self(self.0 + rhs)
    }
}

pub trait SceneObject<ObjectType>: Send {
    fn get_type(&self) -> ObjectType;
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;

    fn new() -> Self where Self: Sized;

    #[allow(unused_variables)]
    fn on_load(&mut self, resource_handler: &mut ResourceHandler) -> Result<()> { Ok(()) }
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
    fn collision_tags(&self) -> Vec<&'static str> { [].into() }
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

    pub fn from_iter<I: IntoIterator<Item=Vec2>>(vertices: I) -> Vec<Self> {
        vertices.into_iter().map(Self::from_vertex).collect()
    }
    pub fn zip_from_iter<I: IntoIterator<Item=Vec2>, J: IntoIterator<Item=Vec2>>(vertices: I, uvs: J) -> Vec<Self> {
        vertices.into_iter().zip(uvs)
            .map(|(vertex, uv)| Self { vertex, uv })
            .collect()
    }
}

pub trait RenderableObject<ObjectType>: SceneObject<ObjectType> {
    fn create_vertices(&self) -> Vec<VertexWithUV>;
    fn render_info(&self) -> RenderInfo;
}

impl<ObjectType, T: SceneObject<ObjectType> + 'static> From<Box<T>> for Box<dyn SceneObject<ObjectType>> {
    fn from(value: Box<T>) -> Self { value }
}

#[derive(Clone)]
pub struct SceneObjectWithId<ObjectType> {
    object_id: ObjectId,
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
    object_id: ObjectId,
    other_map: &'a BTreeMap<ObjectId, SceneObjectWithId<ObjectType>>,
    pending_add_objects: &'a mut Vec<Box<dyn SceneObject<ObjectType>>>,
    pending_remove_objects: &'a mut BTreeSet<ObjectId>,
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
        self.pending_remove_objects.insert(obj.object_id);
    }
    pub fn remove_this_object(&mut self) {
        self.pending_remove_objects.insert(self.object_id);
    }

    pub fn scene_stop(&self) {
        self.scene_instruction_tx.send(SceneInstruction::Stop).unwrap();
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
    render_info: TimeIt,
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
            render_info: TimeIt::new("render_info"),
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
            self.render_info.report_ms_if_at_least(1.0);
            self.total_stats.report_ms();
            self.last_report = Instant::now();
        }
    }
}

pub struct UpdateHandler<ObjectType: ObjectTypeEnum, RenderReceiver: RenderInfoReceiver> {
    objects: BTreeMap<ObjectId, Rc<RefCell<AnySceneObject<ObjectType>>>>,
    vertices: BTreeMap<ObjectId, (Range<usize>, Vec<VertexWithUV>)>,
    render_info: BTreeMap<ObjectId, RenderInfoFull>,
    viewport: AdjustedViewport,
    input_handler: Arc<Mutex<InputHandler>>,
    resource_handler: ResourceHandler,
    render_info_receiver: Arc<Mutex<RenderReceiver>>,
    collision_handler: CollisionHandler,
    scene_instruction_tx: Sender<SceneInstruction>,
    scene_instruction_rx: Receiver<SceneInstruction>,
    perf_stats: UpdatePerfStats,
}

impl<ObjectType: ObjectTypeEnum, RenderReceiver: RenderInfoReceiver> UpdateHandler<ObjectType, RenderReceiver> {
    pub(crate) fn new(
        objects: Vec<AnySceneObject<ObjectType>>,
        input_handler: Arc<Mutex<InputHandler>>,
        mut resource_handler: ResourceHandler,
        render_info_receiver: Arc<Mutex<RenderReceiver>>,
        scene_instruction_tx: Sender<SceneInstruction>,
        scene_instruction_rx: Receiver<SceneInstruction>,
    ) -> Result<Self> {
        let mut objects = objects.into_iter()
            .enumerate()
            .map(|(i, obj)| (ObjectId(i), obj))
            .collect::<BTreeMap<_, _>>();
        let collision_handler = CollisionHandler::new(&objects);
        let mut vertices = BTreeMap::new();
        let mut render_info = BTreeMap::new();
        let mut vertex_index = 0;
        for (&i, obj) in objects.iter_mut() {
            obj.on_load(&mut resource_handler)?;
            if let Some(obj) = obj.as_renderable_object() {
                let new_vertices = obj.create_vertices();
                let vertex_index_range = vertex_index..vertex_index + new_vertices.len();
                vertex_index += new_vertices.len();
                vertices.insert(i, (vertex_index_range.clone(), new_vertices));
                render_info.insert(i, RenderInfoFull {
                    inner: obj.render_info(),
                    transform: obj.transform(),
                    vertex_indices: vertex_index_range,
                });
            }
        }

        let viewport = render_info_receiver.lock().unwrap().current_viewport().clone();

        for object in objects.values_mut() {
            object.on_ready();
        }

        let objects = objects.into_iter()
            .map(|(id, obj)| (id, Rc::new(RefCell::new(obj))))
            .collect();

        let mut rv = Self {
            objects,
            vertices,
            render_info,
            viewport,
            input_handler,
            resource_handler,
            render_info_receiver,
            collision_handler,
            scene_instruction_tx,
            scene_instruction_rx,
            perf_stats: UpdatePerfStats::new(),
        };
        rv.update_render_info(true)?;
        Ok(rv)
    }

    pub fn consume(mut self) -> Result<()> {
        let mut delta = Duration::from_secs(0);
        let mut is_running = true;

        loop {
            if is_running {
                let now = Instant::now();
                self.perf_stats.total_stats.start();

                let input_handler = self.input_handler.lock().unwrap().clone();
                let (pending_add_objects, pending_remove_objects) = self.call_on_update(delta, input_handler);
                let did_update_vertices = !pending_add_objects.is_empty() || !pending_remove_objects.is_empty();

                self.update_with_remove_objects(pending_remove_objects);
                self.update_with_added_objects(pending_add_objects)?;
                self.update_render_info(did_update_vertices)?;
                self.input_handler.lock().unwrap().update_step();

                self.perf_stats.total_stats.stop();
                self.perf_stats.report();
                delta = now.elapsed();
            }

            match self.scene_instruction_rx.try_iter().next() {
                Some(SceneInstruction::Stop) => {},
                Some(SceneInstruction::Pause) => is_running = false,
                Some(SceneInstruction::Resume) => is_running = true,
                None => {}
            }
        }
    }

    fn update_with_remove_objects(&mut self, pending_remove_objects: BTreeSet<ObjectId>) {
        self.perf_stats.remove_objects.start();
        self.collision_handler.update_with_remove_objects(&pending_remove_objects);
        for remove_index in pending_remove_objects.into_iter().rev() {
            match self.render_info.remove(&remove_index) {
                Some(_) => {
                    // This is technically quadratic, but is fast enough.
                    check!(self.vertices.contains_key(&remove_index));
                    let vertices_removed = self.vertices[&remove_index].1.len();
                    for (_, (count, _)) in self.vertices.iter_mut().filter(|(&i, _)| i >= remove_index) {
                        *count = (count.start - vertices_removed)..(count.end - vertices_removed);
                    }
                    self.vertices.remove(&remove_index);
                }
                None => {
                    check_false!(self.vertices.contains_key(&remove_index));
                }
            }
            self.objects.remove(&remove_index);
        }
        self.perf_stats.remove_objects.stop();
    }
    fn update_with_added_objects(&mut self, pending_add_objects: Vec<AnySceneObject<ObjectType>>) -> Result<()> {
        self.perf_stats.add_objects.start();
        if !pending_add_objects.is_empty() {
            // TODO: leak new_id as 'static?
            let first_new_id = self.objects.last_key_value()
                .map(|(id, _)| id.next())
                .unwrap_or(ObjectId(0));
            let pending_add_objects = pending_add_objects.into_iter()
                .enumerate()
                .map(|(i, obj)| (first_new_id + i, obj))
                .collect::<BTreeMap<ObjectId, _>>();
            let last_new_id = *pending_add_objects.last_key_value()
                .expect("pending_add_objects empty?")
                .0;
            self.collision_handler.update_with_add_objects(&pending_add_objects);

            let mut next_vertex_index = self.vertices.last_key_value()
                .map(|(_, (indices, _))| indices.end)
                .unwrap_or(0);
            for (new_id, mut new_obj) in pending_add_objects {
                new_obj.on_load(&mut self.resource_handler)?;
                if let Some(obj) = new_obj.as_renderable_object() {
                    let new_vertices = obj.create_vertices();
                    let vertex_indices = next_vertex_index..next_vertex_index + new_vertices.len();
                    next_vertex_index += new_vertices.len();
                    self.vertices.insert(new_id, (vertex_indices.clone(), new_vertices));
                    self.render_info.insert(new_id, RenderInfoFull {
                        inner: obj.render_info(),
                        transform: obj.transform(),
                        vertex_indices,
                    });
                }
                self.objects.insert(new_id, Rc::new(RefCell::new(new_obj)));
            }

            for i in first_new_id.0..=last_new_id.0 {
                self.objects[&ObjectId(i)].borrow_mut().on_ready();
            }
        }
        self.perf_stats.add_objects.stop();
        Ok(())
    }

    fn call_on_update(&mut self, delta: Duration, input_handler: InputHandler) -> (Vec<AnySceneObject<ObjectType>>, BTreeSet<ObjectId>) {
        self.perf_stats.on_update_begin.start();
        let mut pending_add_objects = Vec::new();
        let mut pending_remove_objects = BTreeSet::new();
        let mut other_map = self.objects
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
                              pending_remove_objects: &mut BTreeSet<ObjectId>,
                              other_map: &mut BTreeMap<ObjectId, SceneObjectWithId<ObjectType>>,
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
    fn update_render_info(&mut self, did_update_vertices: bool) -> Result<()> {
        self.perf_stats.render_info.start();
        for object_id in self.objects.keys() {
            if let Some(obj) = self.objects[object_id].borrow().as_renderable_object() {
                let render_info = self.render_info.get_mut(object_id)
                    .ok_or(anyhow!("missing object_id in render_info: {:?}", object_id))?;
                render_info.inner = obj.render_info();
                render_info.transform = obj.transform();
            }
        }
        let mut render_info_receiver = self.render_info_receiver.lock().unwrap();
        if did_update_vertices {
            render_info_receiver.update_vertices(self.vertices.values()
                .cloned()
                .flat_map(|(_, values)| values)
                .collect());
        }
        render_info_receiver.update_render_info(self.render_info.values().cloned().collect());
        self.viewport = render_info_receiver.current_viewport();
        self.perf_stats.render_info.stop();
        Ok(())
    }
}

struct Collision<ObjectType: ObjectTypeEnum> {
    this: Rc<RefCell<AnySceneObject<ObjectType>>>,
    other: SceneObjectWithId<ObjectType>,
    mtv: Vec2,
}

struct CollisionHandler {
    object_ids_by_tag: BTreeMap<&'static str, BTreeSet<ObjectId>>,
    object_ids_by_listening_tag: BTreeMap<&'static str, BTreeSet<ObjectId>>,
    possible_collisions: BTreeSet<UnorderedPair<ObjectId>>,
}

impl CollisionHandler {
    pub fn new<ObjectType: ObjectTypeEnum>(
        objects: &BTreeMap<ObjectId, AnySceneObject<ObjectType>>)
    -> Self {
        let mut rv = Self {
            object_ids_by_tag: BTreeMap::new(),
            object_ids_by_listening_tag: BTreeMap::new(),
            possible_collisions: BTreeSet::new()
        };
        rv.update_with_add_objects(objects);
        rv
    }
    pub fn update_with_add_objects<ObjectType: ObjectTypeEnum>(&mut self, objects: &BTreeMap<ObjectId, AnySceneObject<ObjectType>>) {
        let mut new_object_ids_by_tag = BTreeMap::<&'static str, Vec<ObjectId>>::new();
        let mut new_object_ids_by_listening_tag = BTreeMap::<&'static str, Vec<ObjectId>>::new();
        for (id, obj) in objects.iter() {
            for tag in obj.collision_tags() {
                new_object_ids_by_tag.entry(tag).or_default().push(*id);
            }
            for tag in obj.listening_tags() {
                new_object_ids_by_listening_tag.entry(tag).or_default().push(*id);
            }
        }
        let all_tags = new_object_ids_by_tag.keys().chain(new_object_ids_by_listening_tag.keys());
        for tag in all_tags {
            self.object_ids_by_tag.entry(tag).or_default().extend(
                new_object_ids_by_tag.get(tag).unwrap_or(&Vec::new()));
            self.object_ids_by_listening_tag.entry(tag).or_default().extend(
                new_object_ids_by_listening_tag.get(tag).unwrap_or(&Vec::new()));
        }
        for (tag, emitters) in new_object_ids_by_tag {
            let listeners = self.object_ids_by_listening_tag.get(tag)
                .unwrap_or_else(|| panic!("object_ids_by_listening tag missing tag: {}", tag));
            let new_possible_collisions = emitters.into_iter().cartesian_product(listeners.iter())
                .filter_map(|(emitter, listener)| UnorderedPair::new_distinct(emitter, *listener));
            self.possible_collisions.extend(new_possible_collisions);
        }
        for (tag, listeners) in new_object_ids_by_listening_tag {
            let emitters = self.object_ids_by_tag.get(tag)
                .unwrap_or_else(|| panic!("object_ids_by_tag tag missing tag: {}", tag));
            let new_possible_collisions = emitters.iter().cartesian_product(listeners.into_iter())
                .filter_map(|(emitter, listener)| UnorderedPair::new_distinct(*emitter, listener));
            self.possible_collisions.extend(new_possible_collisions);
        }
    }
    pub fn update_with_remove_objects(&mut self, removed_ids: &BTreeSet<ObjectId>) {
        for (_, ids) in self.object_ids_by_tag.iter_mut() {
            ids.retain(|id| !removed_ids.contains(id));
        }
        for (_, ids) in self.object_ids_by_listening_tag.iter_mut() {
            ids.retain(|id| !removed_ids.contains(id));
        }
        self.possible_collisions.retain(|pair| !removed_ids.contains(&pair.fst()) && !removed_ids.contains(&pair.snd()));
    }
    pub fn get_collisions<ObjectType: ObjectTypeEnum>(
        &self,
        objects: &BTreeMap<ObjectId, Rc<RefCell<AnySceneObject<ObjectType>>>>)
    -> Vec<Collision<ObjectType>> {
        self.possible_collisions.iter().copied()
            .filter_map(|ids| {
                let this = objects[&ids.fst()].borrow().collider()
                    .unwrap_or_else(|| panic!("object id {:?} missing collider, but in possible_collisions", ids.fst()));
                let other = objects[&ids.snd()].borrow().collider()
                    .unwrap_or_else(|| panic!("object id {:?} missing collider, but in possible_collisions", ids.snd()));
                this.collides_with(other.as_ref()).map(|mtv| (ids, mtv))
            })
            .flat_map(|(ids, mtv)| {
                let this = objects[&ids.fst()].clone();
                let other = objects[&ids.snd()].clone();
                vec![Collision {
                    this: this.clone(),
                    other: SceneObjectWithId {
                        object_id: ids.snd(),
                        inner: other.clone(),
                    },
                    mtv,
                }, Collision {
                    this: other,
                    other: SceneObjectWithId {
                        object_id: ids.fst(),
                        inner: this,
                    },
                    mtv: -mtv,
                }]
            })
            .collect()
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TextureSubArea {
    rect: Option<Rect>,
}

impl TextureSubArea {
    pub fn new(centre: Vec2Int, half_widths: Vec2Int) -> Self {
        Self::from_rect(Rect::new(centre.into(), half_widths.into()))
    }
    pub fn from_rect(rect: Rect) -> Self {
        Self { rect: Some(rect) }
    }

    pub fn uv(&self, texture: &Texture, raw_uv: Vec2) -> Vec2 {
        match self.rect {
            None => raw_uv,
            Some(rect) => {
                let extent = texture.extent();
                let u0 = rect.top_left().x / extent.x;
                let v0 = rect.top_left().y / extent.y;
                let u1 = rect.bottom_right().x / extent.x;
                let v1 = rect.bottom_right().y / extent.y;
                Vec2 { x: linalg::lerp(u0, u1, raw_uv.x), y: linalg::lerp(v0, v1, raw_uv.y) }
            }
        }
    }

    pub fn half_widths(&self) -> Vec2 {
        match self.rect {
            None => Vec2::zero(),
            Some(rect) => rect.half_widths(),
        }
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
    inner: RenderInfo,
    transform: Transform,
    vertex_indices: Range<usize>,
}

pub trait RenderInfoReceiver: Send {
    fn update_vertices(&mut self, vertices: Vec<VertexWithUV>);
    fn update_render_info(&mut self, render_info: Vec<RenderInfoFull>);
    fn current_viewport(&self) -> AdjustedViewport;
    fn is_ready(&self) -> bool;
}
