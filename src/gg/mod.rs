pub mod sample;

use std::{
    cell::{RefCell, RefMut},
    collections::HashMap,
    sync::{
        mpsc::{Receiver, Sender},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};

use tracing::info;

use crate::core::{
    linalg::Vec2,
    util::TimeIt,
    vk_core::AdjustedViewport
};
use crate::core::input::InputHandler;

pub struct SceneObjectWithId<'a> {
    object_id: usize,
    inner: RefMut<'a, Box<dyn SceneObject>>,
}

pub trait SceneObject: Send {
    fn on_ready(&mut self);
    fn on_update(&mut self, delta: Duration, update_ctx: UpdateContext);
    // TODO: probably should somehow restrict UpdateContext for on_update_begin/end().
    #[allow(unused_variables)]
    fn on_update_begin(&mut self, delta: Duration, update_ctx: UpdateContext) {}
    #[allow(unused_variables)]
    fn on_update_end(&mut self, delta: Duration, update_ctx: UpdateContext) {}

    fn as_world_object(&self) -> Option<&dyn WorldObject> {
        None
    }
    fn as_renderable_object(&self) -> Option<&dyn RenderableObject> {
        None
    }
}

pub trait WorldObject: SceneObject {
    fn world_pos(&self) -> Vec2;
}

pub trait RenderableObject: SceneObject {
    fn create_vertices(&self) -> Vec<Vec2>;
    fn render_data(&self) -> RenderData;
}

impl<'a> SceneObjectWithId<'a> {
    pub fn on_ready(&mut self) {
        self.inner.on_ready()
    }
    pub fn on_update(&mut self, delta: Duration, update_ctx: UpdateContext) {
        self.inner.on_update(delta, update_ctx)
    }
    pub fn on_update_end(&mut self, delta: Duration, update_ctx: UpdateContext) {
        self.inner.on_update_end(delta, update_ctx)
    }

    pub fn as_world_object(&self) -> Option<&dyn WorldObject> {
        self.inner.as_world_object()
    }

    pub fn as_renderable_object(&self) -> Option<&dyn RenderableObject> {
        self.inner.as_renderable_object()
    }
}

pub struct UpdateContext<'a> {
    input_handler: &'a InputHandler,
    scene_instruction_tx: Sender<SceneInstruction>,
    object_id: usize,
    other_map: &'a HashMap<usize, SceneObjectWithId<'a>>,
    pending_add_objects: &'a mut Vec<Box<dyn SceneObject>>,
    pending_remove_objects: &'a mut Vec<usize>,
    viewport: AdjustedViewport,
}

impl<'a> UpdateContext<'a> {
    pub fn input(&self) -> &InputHandler { self.input_handler }

    pub fn others(&self) -> Vec<&SceneObjectWithId> {
        self.other_map
            .values()
            .filter(|obj| !self.pending_remove_objects.contains(&obj.object_id))
            .collect()
    }

    pub fn add_object_vec(&mut self, mut objects: Vec<Box<dyn SceneObject>>) {
        self.pending_add_objects.append(&mut objects);
    }
    pub fn add_object(&mut self, object: Box<dyn SceneObject>) {
        self.pending_add_objects.push(object);
    }
    pub fn remove_other_object(&mut self, obj: &SceneObjectWithId) {
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

pub struct UpdateHandler<RenderReceiver: RenderDataReceiver> {
    objects: HashMap<usize, RefCell<Box<dyn SceneObject>>>,
    vertices: HashMap<usize, Vec<Vec2>>,
    render_data: HashMap<usize, RenderData>,
    viewport: AdjustedViewport,
    render_data_receiver: Arc<Mutex<RenderReceiver>>,
    input_handler: Arc<Mutex<InputHandler>>,
    scene_instruction_tx: Sender<SceneInstruction>,
    scene_instruction_rx: Receiver<SceneInstruction>,
}

impl<RenderReceiver: RenderDataReceiver> UpdateHandler<RenderReceiver> {
    pub(crate) fn new(
        objects: Vec<RefCell<Box<dyn SceneObject>>>,
        render_data_receiver: Arc<Mutex<RenderReceiver>>,
        input_handler: Arc<Mutex<InputHandler>>,
        scene_instruction_tx: Sender<SceneInstruction>,
        scene_instruction_rx: Receiver<SceneInstruction>,
    ) -> Self {
        let objects: HashMap<usize, _> = objects.into_iter().enumerate().collect();
        let vertices = objects
            .iter()
            .filter_map(|(&i, obj)| {
                obj.borrow()
                    .as_renderable_object()
                    .map(|obj| (i, obj.create_vertices()))
            })
            .collect();
        let render_data = objects
            .iter()
            .filter_map(|(&i, obj)| {
                obj.borrow()
                    .as_renderable_object()
                    .map(|obj| (i, obj.render_data()))
            })
            .collect();
        let viewport = render_data_receiver.lock().unwrap().current_viewport().clone();
        let mut rv = Self {
            objects,
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
}

impl<RenderReceiver: RenderDataReceiver> UpdateHandler<RenderReceiver> {
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
                    self.render_data.remove(&remove_index);
                    self.vertices.remove(&remove_index);
                    self.objects.remove(&remove_index);
                }
                remove_objects_stats.stop();

                add_objects_stats.start();
                let mut next_id = *self.objects.keys().max().unwrap_or(&0);
                let first_new_id = next_id + 1;
                for new_obj in pending_add_objects {
                    next_id += 1;
                    if let Some(obj) = new_obj.as_renderable_object() {
                        self.vertices.insert(next_id, obj.create_vertices());
                        self.render_data.insert(next_id, obj.render_data());
                    }
                    self.objects.insert(next_id, RefCell::new(new_obj));
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

    fn call_on_update(&mut self, delta: Duration, input_handler: &InputHandler) -> (Vec<Box<dyn SceneObject>>, Vec<usize>) {
        let mut pending_add_objects = Vec::new();
        let mut pending_remove_objects = Vec::new();

        {
            let mut other_map: HashMap<usize, _> = self
                .objects
                .iter()
                .map(|(&i, obj)| (i, SceneObjectWithId {
                    object_id: i,
                    inner: obj.borrow_mut(),
                }))
                .collect();
            self.iter_with_other_map(delta, input_handler, &mut pending_add_objects, &mut pending_remove_objects, &mut other_map,
                                     |mut obj, delta, update_ctx| obj.on_update_begin(delta, update_ctx));
            self.iter_with_other_map(delta, input_handler, &mut pending_add_objects, &mut pending_remove_objects, &mut other_map,
                                     |mut obj, delta, update_ctx| obj.on_update(delta, update_ctx));
            self.iter_with_other_map(delta, input_handler, &mut pending_add_objects, &mut pending_remove_objects, &mut other_map,
                                     |mut obj, delta, update_ctx| obj.on_update_end(delta, update_ctx));
        }

        // render_data()
        for &object_id in self.objects.keys() {
            if let Some(obj) = self.objects[&object_id].borrow().as_renderable_object() {
                self.render_data.insert(object_id, obj.render_data());
            }
        }
        (pending_add_objects, pending_remove_objects)
    }
    fn iter_with_other_map<'a, F>(&'a self,
                                  delta: Duration,
                                  input_handler: &InputHandler,
                                  pending_add_objects: &mut Vec<Box<dyn SceneObject>>,
                                  pending_remove_objects: &mut Vec<usize>,
                                  other_map: &mut HashMap<usize, SceneObjectWithId<'a>>,
                                  call_obj_event: F)
    where F: Fn(RefMut<Box<dyn SceneObject>>, Duration, UpdateContext) {
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
                    inner: obj.borrow_mut(),
                },
            );
        }
    }
    fn update_render_data(&mut self, did_update_vertices: bool) {
        let mut render_data_receiver = self.render_data_receiver.lock().unwrap();
        if did_update_vertices {
            render_data_receiver.update_vertices(self.vertices.values().flatten().cloned().collect());
        }
        render_data_receiver.update_render_data(self.render_data.values().cloned().collect());
        self.viewport = render_data_receiver.current_viewport();
    }
}

#[derive(Clone)]
pub struct RenderData {
    pub position: Vec2,
    pub rotation: f64,
}

pub trait RenderDataReceiver: Send {
    fn update_vertices(&mut self, vertices: Vec<Vec2>);
    fn update_render_data(&mut self, render_data: Vec<RenderData>);
    fn current_viewport(&self) -> AdjustedViewport;
}
