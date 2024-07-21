pub mod collision;

use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;
use std::cell::{RefCell, RefMut};
use std::sync::{Arc, mpsc, Mutex};
use std::sync::mpsc::{Receiver, Sender};
use std::time::{Duration, Instant};
use tracing::{info, warn};
use serde::Serialize;
use serde::de::DeserializeOwned;
use num_traits::{FromPrimitive, Zero};
use collision::{Collision, CollisionHandler, CollisionNotification, CollisionResponse};
use crate::{
    core::{
        util::{
            gg_time::TimeIt,
            linalg::{AxisAlignedExtent, Vec2},
            colour::Colour,
            collision::Collider,
            NonemptyVec
        },
        prelude::*,
        AnySceneObject,
        ObjectId,
        ObjectTypeEnum,
        PendingObjectVec,
        SceneObjectWithId,
        config::{FIXED_UPDATE_INTERVAL_US, MAX_FIXED_UPDATES},
        coroutine::{Coroutine, CoroutineId, CoroutineResponse, CoroutineState},
        input::InputHandler,
        render::{RenderInfoFull, RenderInfoReceiver, RenderItem, VertexMap},
        scene::{SceneHandlerInstruction, SceneInstruction, SceneName, SceneDestination},
        vk::AdjustedViewport
    },
    resource::ResourceHandler
};

pub(crate) struct UpdateHandler<ObjectType: ObjectTypeEnum, RenderReceiver: RenderInfoReceiver> {
    objects: BTreeMap<ObjectId, Rc<RefCell<AnySceneObject<ObjectType>>>>,
    vertex_map: VertexMap,
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
    ) -> anyhow::Result<Self> {
        let (scene_instruction_tx, scene_instruction_rx) = mpsc::channel();
        let mut rv = Self {
            objects: BTreeMap::new(),
            vertex_map: VertexMap::new(),
            render_infos: BTreeMap::new(),
            viewport: render_info_receiver.clone().lock().unwrap().current_viewport(),
            input_handler,
            resource_handler,
            render_info_receiver,
            clear_col: Colour::black(),
            collision_handler: CollisionHandler::new(),
            coroutines: BTreeMap::new(),
            scene_instruction_tx,
            scene_instruction_rx,
            scene_name,
            scene_data,
            perf_stats: UpdatePerfStats::new(),
        };

        let input_handler = rv.input_handler.lock().unwrap().clone();
        rv.update_with_added_objects(
            &input_handler,
            objects.into_iter()
                .map(|obj| Rc::new(RefCell::new(obj)))
                .collect()
        )?;
        rv.update_and_send_render_infos();
        Ok(rv)
    }

    pub(crate) fn consume(mut self) -> anyhow::Result<SceneHandlerInstruction> {
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

                self.update_with_removed_objects(pending_remove_objects);
                self.update_with_added_objects(&input_handler, pending_add_objects)?;
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

    fn update_with_removed_objects(&mut self, pending_remove_objects: BTreeSet<ObjectId>) {
        self.perf_stats.remove_objects.start();
        self.collision_handler.update_with_removed_objects(&pending_remove_objects);
        for remove_index in pending_remove_objects.into_iter().rev() {
            self.vertex_map.remove(remove_index);
            self.render_infos.remove(&remove_index);
            self.objects.remove(&remove_index);
            self.coroutines.remove(&remove_index);
        }
        self.perf_stats.remove_objects.stop();
    }
    fn update_with_added_objects(&mut self, input_handler: &InputHandler, mut pending_add_objects: Vec<Rc<RefCell<AnySceneObject<ObjectType>>>>) -> anyhow::Result<()> {
        self.perf_stats.add_objects.start();
        loop {
            if pending_add_objects.is_empty() {
                break;
            }
            let pending_add = pending_add_objects.drain(..)
                .map(|obj| (ObjectId::next(), obj))
                .collect::<BTreeMap<ObjectId, _>>();
            let first_new_id = *pending_add.first_key_value()
                .expect("pending_add_objects empty?")
                .0;
            let last_new_id = *pending_add.last_key_value()
                .expect("pending_add_objects empty?")
                .0;
            self.collision_handler.update_with_added_objects(&pending_add);

            // Call on_load().
            for (new_id, new_obj) in pending_add {
                let new_vertices = new_obj.borrow_mut().on_load(&mut self.resource_handler)?;
                self.vertex_map.insert(new_id, new_vertices);
                self.objects.insert(new_id, new_obj);
            }

            // Call on_ready().
            let mut object_tracker = ObjectTracker {
                last: self.objects.clone(),
                pending_add: Vec::new(),
                pending_remove: BTreeSet::new()
            };
            for i in first_new_id.0..=last_new_id.0 {
                let this_id = ObjectId(i);
                let this = self.objects.get_mut(&this_id)
                    .unwrap_or_else(|| panic!("tried to call on_ready() for nonexistent added object: {this_id:?}"))
                    .clone();
                let mut ctx = UpdateContext::new(
                    self,
                    input_handler,
                    SceneObjectWithId::new(this_id, this.clone()),
                    &mut object_tracker,
                );
                this.borrow_mut().on_ready(&mut ctx);
            }
            let (pending_add, pending_remove) = object_tracker.into_pending();
            self.update_with_removed_objects(pending_remove);
            pending_add_objects = pending_add;
        }
        self.perf_stats.add_objects.stop();
        Ok(())
    }

    fn call_on_update(&mut self,
                      delta: Duration,
                      input_handler: &InputHandler,
                      mut fixed_updates: u128
    ) -> ObjectTracker<ObjectType> {
        self.perf_stats.on_update_begin.start();
        let mut object_tracker = ObjectTracker {
            last: self.objects.clone(),
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

        for _ in 0..fixed_updates.min(MAX_FIXED_UPDATES) {
            self.iter_with_other_map(delta, input_handler, &mut object_tracker,
                                     |mut obj, _delta, ctx| {
                                         obj.on_fixed_update(ctx);
                                     });
            fixed_updates -= 1;
            // Detect collisions after each fixed update: important to prevent glitching through walls etc.
            self.handle_collisions(input_handler, &mut object_tracker);
        }

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
        let collisions = self.collision_handler.get_collisions(&self.objects);
        self.perf_stats.detect_collision.stop();
        self.perf_stats.on_collision.start();
        let mut done_with_collisions = BTreeSet::new();
        for CollisionNotification { this, other, mtv } in collisions {
            if !done_with_collisions.contains(&this.object_id) {
                let mut ctx = UpdateContext::new(
                    self,
                    input_handler,
                    this.clone(),
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
        for (this_id, this) in self.objects.clone() {
            let this = SceneObjectWithId::new(this_id, this.clone());
            let last_coroutines = self.coroutines.remove(&this_id).unwrap_or_default();
            for (id, coroutine) in last_coroutines {
                let mut ctx = UpdateContext::new(
                    self,
                    input_handler,
                    this.clone(),
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
        for (this_id, this) in self.objects.clone() {
            let this = SceneObjectWithId::new(this_id, this.clone());
            let mut ctx = UpdateContext::new(
                self,
                input_handler,
                this.clone(),
                object_tracker
            );
            call_obj_event(this.inner.borrow_mut(), delta, &mut ctx);
        }
    }
    fn update_and_send_render_infos(&mut self) {
        self.perf_stats.render_infos.start();
        self.update_render_infos();
        self.send_render_infos();
        self.perf_stats.render_infos.stop();
    }
    fn update_render_infos(&mut self) {
        for (object_id, obj) in &self.objects {
            if let Some(obj) = obj.borrow().as_renderable_object() {
                let (indices, _) = self.vertex_map.get(*object_id)
                    .unwrap_or_else(|| panic!("missing object_id in vertex_map: {object_id:?}"));
                self.render_infos.insert(*object_id, RenderInfoFull {
                    vertex_indices: indices.clone(),
                    inner: obj.render_info(),
                    transform: obj.transform().translated(-self.viewport.translation),
                });
            }
        }
    }
    fn send_render_infos(&mut self) {
        let mut render_info_receiver = self.render_info_receiver.lock().unwrap();
        if self.vertex_map.consume_vertices_changed() {
            let mut vertices = Vec::with_capacity(self.vertex_map.vertex_count());
            for r in self.vertex_map.render_items() {
                vertices.extend(r.vertices);
            }
            check_eq!(vertices.len(), vertices.capacity());
            render_info_receiver.update_vertices(vertices);
        }
        render_info_receiver.update_render_info(self.render_infos.values().cloned().collect());
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
        this: SceneObjectWithId<ObjectType>,
        object_tracker: &'a mut ObjectTracker<ObjectType>
    ) -> Self {
        let this_id = this.object_id;
        Self {
            input: input_handler,
            scene: SceneContext {
                scene_instruction_tx: caller.scene_instruction_tx.clone(),
                scene_name: caller.scene_name,
                scene_data: caller.scene_data.clone(),
                coroutines: caller.coroutines.entry(this.object_id).or_default(),
                pending_removed_coroutines: BTreeSet::new(),
            },
            object: ObjectContext {
                collision_handler: &caller.collision_handler,
                this,
                object_tracker,
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
    fn new(raw: Arc<Mutex<Vec<u8>>>) -> anyhow::Result<Option<Self>> {
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

struct ObjectTracker<ObjectType: ObjectTypeEnum> {
    last: BTreeMap<ObjectId, Rc<RefCell<AnySceneObject<ObjectType>>>>,
    pending_add: Vec<AnySceneObject<ObjectType>>,
    pending_remove: BTreeSet<ObjectId>,
}

impl<ObjectType: ObjectTypeEnum> ObjectTracker<ObjectType> {
    fn into_pending(self) -> (PendingObjectVec<ObjectType>, BTreeSet<ObjectId>) {
        (self.pending_add.into_iter().map(|obj| Rc::new(RefCell::new(obj))).collect(),
         self.pending_remove)
    }
}

pub struct ObjectContext<'a, ObjectType: ObjectTypeEnum> {
    collision_handler: &'a CollisionHandler,
    this: SceneObjectWithId<ObjectType>,
    object_tracker: &'a mut ObjectTracker<ObjectType>,
}

impl<'a, ObjectType: ObjectTypeEnum> ObjectContext<'a, ObjectType> {
    pub fn others(&self) -> Vec<SceneObjectWithId<ObjectType>> {
        self.object_tracker.last.iter()
            .filter(|(object_id, _)| !self.object_tracker.pending_remove.contains(object_id))
            .filter(|(object_id, _)| self.this.object_id != **object_id)
            .map(|(object_id, obj)| SceneObjectWithId::new(*object_id, obj.clone()))
            .collect()
    }

    pub fn add_vec(&mut self, objects: Vec<AnySceneObject<ObjectType>>) -> &mut [AnySceneObject<ObjectType>] {
        let pending_add = &mut self.object_tracker.pending_add;
        let begin = pending_add.len();
        pending_add.extend(objects);
        let end = pending_add.len();
        &mut pending_add[begin..end]
    }
    pub fn add(&mut self, object: AnySceneObject<ObjectType>) -> &mut AnySceneObject<ObjectType>{
        self.object_tracker.pending_add.push(object);
        self.object_tracker.pending_add.last_mut().unwrap()
    }
    pub fn remove(&mut self, obj: &SceneObjectWithId<ObjectType>) {
        self.object_tracker.pending_remove.insert(obj.object_id);
    }
    pub fn remove_this(&mut self) {
        let this_id = self.this.object_id;
        self.object_tracker.pending_remove.insert(this_id);
    }
    pub fn test_collision(&self,
                          collider: &dyn Collider,
                          listening_tags: Vec<&'static str>
    ) -> Option<NonemptyVec<Collision<ObjectType>>> {
        let mut rv = Vec::new();
        for tag in listening_tags {
            for other_id in self.collision_handler.get_object_ids_by_emitting_tag(tag).unwrap() {
                if let Some(other) = self.object_tracker.last
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
