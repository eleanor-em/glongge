use std::cell::{Ref, RefCell};
use std::collections::hash_map::Values;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use num_traits::Zero;
use tracing::info;

use crate::{
    core::{
        linalg::Vec2,
        util::TimeIt
    },
};

pub trait SceneObject: Send {
    fn create_vertices(&self) -> Vec<Vec2> { Vec::new() }
    fn on_ready(&mut self);
    fn on_update(&mut self, delta: Duration, update_handler: UpdateContext) -> RenderData;
    fn world_pos(&self) -> Vec2 { Vec2::zero() }
}

pub struct UpdateContext<'a> {
    object_id: usize,
    others: &'a HashMap<usize, Ref<'a, Box<dyn SceneObject>>>,
    pending_add_objects: &'a mut Vec<Box<dyn SceneObject>>,
    pending_remove_objects: &'a mut Vec<usize>,
}

impl<'a> UpdateContext<'a> {
    pub fn others(&self) -> Values<usize, Ref<'a, Box<dyn SceneObject>>> { self.others.values() }

    pub fn add_object(&mut self, object: Box<dyn SceneObject>) {
        self.pending_add_objects.push(object);
    }
    pub fn remove_this_object(&mut self) {
        self.pending_remove_objects.push(self.object_id);
    }
}

pub struct UpdateHandler<Receiver: RenderDataReceiver> {
    objects: HashMap<usize, RefCell<Box<dyn SceneObject>>>,
    vertices: HashMap<usize, Vec<Vec2>>,
    render_data: HashMap<usize, RenderData>,
    render_data_receiver: Arc<Mutex<Receiver>>,
}

impl<Receiver: RenderDataReceiver> UpdateHandler<Receiver> {
    pub(crate) fn new(objects: Vec<RefCell<Box<dyn SceneObject>>>, render_data_receiver: Arc<Mutex<Receiver>>) -> Self {
        let objects: HashMap<usize, _> = objects.into_iter().enumerate().collect();
        let vertices = objects.iter()
            .map(|(&i, obj)| (i, obj.borrow().create_vertices()))
            .collect();
        let render_data = objects.keys()
            .map(|&i| (i, RenderData::empty()))
            .collect();
        let rv = Self {
            objects,
            vertices,
            render_data,
            render_data_receiver,
        };
        rv.update_render_data(true);
        rv
    }
}

impl<Receiver: RenderDataReceiver> UpdateHandler<Receiver> {
    fn call_on_update(&mut self, delta: Duration) -> (Vec<Box<dyn SceneObject>>, Vec<usize>) {
        let mut pending_add_objects = Vec::new();
        let mut pending_remove_objects = Vec::new();

        let mut other_map: HashMap<usize, _> = self.objects.iter()
            .map(|(&i, obj)| (i, obj.borrow()))
            .collect();
        for &object_id in self.objects.keys() {
            other_map.remove(&object_id);
            let update_ctx = UpdateContext {
                object_id,
                others: &other_map,
                pending_add_objects: &mut pending_add_objects,
                pending_remove_objects: &mut pending_remove_objects,
            };
            let next_render_data = self.objects[&object_id]
                .borrow_mut()
                .on_update(delta, update_ctx);
            self.render_data.insert(object_id, next_render_data);
            other_map.insert(object_id, self.objects[&object_id].borrow());
        }

        (pending_add_objects, pending_remove_objects)
    }
    fn update_render_data(&self, did_update_vertices: bool) {
        let mut render_data_receiver = self.render_data_receiver.lock().unwrap();
        if did_update_vertices {
            render_data_receiver.update_vertices(self.vertices.values().flatten().cloned().collect());
        }
        render_data_receiver.update_render_data(self.render_data.values().cloned().collect());
    }
    pub fn consume(mut self) {
        let mut delta = Duration::from_secs(0);
        let mut total_stats = TimeIt::new("total");
        let mut update_objects_stats = TimeIt::new("on_update");
        let mut add_objects_stats = TimeIt::new("add objects");
        let mut remove_objects_stats = TimeIt::new("remove objects");
        let mut render_data_stats = TimeIt::new("render_data");
        let mut last_report = Instant::now();
        loop {
            let now = Instant::now();
            total_stats.start();

            update_objects_stats.start();
            let (pending_add_objects, pending_remove_objects) =
                self.call_on_update(delta);
            let did_update_vertices = !pending_add_objects.is_empty() || !pending_remove_objects.is_empty();
            update_objects_stats.stop();

            remove_objects_stats.start();
            for remove_index in pending_remove_objects.into_iter().rev() {
                self.render_data.remove(&remove_index);
                self.vertices.remove(&remove_index);
                self.objects.remove(&remove_index);
            }
            remove_objects_stats.stop();

            add_objects_stats.start();
            for new_obj in pending_add_objects {
                let next_id = match self.objects.keys().max() {
                    Some(&last_id) => last_id + 1,
                    None => 0,
                };
                self.render_data.insert(next_id, RenderData::empty());
                self.vertices.insert(next_id, new_obj.create_vertices());
                self.objects.insert(next_id, RefCell::new(new_obj));
                self.objects[&next_id].borrow_mut().on_ready();
            }
            add_objects_stats.stop();
            render_data_stats.start();
            self.update_render_data(did_update_vertices);
            render_data_stats.stop();

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
    }
}

#[derive(Clone)]
pub struct RenderData {
    pub position: Vec2,
    pub rotation: f64,
}

impl RenderData {
    pub(crate) fn empty() -> Self { Self { position: Vec2::zero(), rotation: 0.0 } }
}

pub trait RenderDataReceiver {
    fn update_vertices(&mut self, vertices: Vec<Vec2>);
    fn update_render_data(&mut self, render_data: Vec<RenderData>);
}
