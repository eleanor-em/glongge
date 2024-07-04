use std::cell::{Ref, RefCell};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use num_traits::Zero;

use crate::{
    core::{
        linalg::Vec2,
        util::TimeIt
    },
};

pub trait SceneObject: Send {
    fn create_vertices(&self) -> Vec<Vec2> { Vec::new() }
    fn on_ready(&mut self);
    fn on_update(&mut self, delta: f64, update_handler: UpdateContext) -> RenderData;
    fn world_pos(&self) -> Vec2 { Vec2::zero() }
}

pub struct UpdateContext<'a> {
    others: SafeObjectList<'a>,
    pending_objects: &'a mut Vec<Box<dyn SceneObject>>,
}

impl<'a> UpdateContext<'a> {
    pub fn others(&self) -> SafeObjectList<'a> { self.others.clone() }

    pub fn add_object(&mut self, object: Box<dyn SceneObject>) {
        self.pending_objects.push(object);
    }
}

pub struct UpdateHandler<Receiver: RenderDataReceiver> {
    pub(crate) objects: Vec<RefCell<Box<dyn SceneObject>>>,
    pub(crate) render_data_receiver: Arc<Mutex<Receiver>>,
}

impl<Receiver: RenderDataReceiver> UpdateHandler<Receiver> {
    pub fn consume(mut self) {
        let mut delta = 0.0;
        let mut timer = TimeIt::new("update");
        let mut render_data = self.render_data_receiver.lock().unwrap().cloned();
        loop {
            let now = Instant::now();
            timer.start();
            let mut pending_objects = Vec::new();
            for i in 0..self.objects.len() {
                let update_ctx = UpdateContext {
                    others: SafeObjectList::new(i, &self.objects),
                    pending_objects: &mut pending_objects,
                };
                let next_render_data = self.objects[i]
                    .borrow_mut()
                    .on_update(delta, update_ctx);
                render_data[i] = next_render_data;
            }
            let vertices: Vec<Vec2> = pending_objects.iter()
                .map(|obj| obj.create_vertices())
                .flatten()
                .collect();
            for mut pending_obj in pending_objects {
                render_data.push(RenderData::empty());
                pending_obj.on_ready();
                self.objects.push(RefCell::new(pending_obj));
            }
            {
                let mut render_data_receiver = self.render_data_receiver.lock().unwrap();
                if !vertices.is_empty() {
                    render_data_receiver.add_vertices(vertices);
                }
                render_data_receiver.update_from(render_data.clone());
            }

            timer.stop();
            timer.report_ms_every(5);
            delta = now.elapsed().as_secs_f64();
        }
    }
}

#[derive(Clone)]
pub struct SafeObjectList<'a> {
    owner_index: usize,
    objects: &'a [RefCell<Box<dyn SceneObject>>],
    curr: usize,
}

impl<'a> SafeObjectList<'a> {
    pub fn new(owner_index: usize, objects: &'a [RefCell<Box<dyn SceneObject>>]) -> Self {
        Self { owner_index, objects, curr: 0, }
    }

    pub fn len(&self) -> usize { self.objects.len() }
}

impl<'a> Iterator for SafeObjectList<'a> {
    type Item = Ref<'a, Box<dyn SceneObject>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.curr += 1;
        if self.curr == self.owner_index {
            self.next()
        } else if self.curr >= self.objects.len() {
            None
        } else {
            Some(self.objects[self.curr].borrow())
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
    fn add_vertices(&mut self, vertices: Vec<Vec2>);
    fn update_from(&mut self, render_data: Vec<RenderData>);
    fn cloned(&self) -> Vec<RenderData>;
}
