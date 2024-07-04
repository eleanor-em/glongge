use std::cell::{Ref, RefCell};
use std::collections::HashMap;
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
    pending_add_objects: &'a mut Vec<Box<dyn SceneObject>>,
    pending_remove_objects: &'a mut Vec<usize>,
}

impl<'a> UpdateContext<'a> {
    pub fn others(&self) -> SafeObjectList<'a> { self.others.clone() }

    pub fn add_object(&mut self, object: Box<dyn SceneObject>) {
        self.pending_add_objects.push(object);
    }
    pub fn remove_this_object(&mut self) {
        self.pending_remove_objects.push(self.others.owner_index);
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
        let mut vertices: Vec<Vec<Vec2>> = self.objects.iter()
            .map(|obj| obj.borrow().create_vertices())
            .collect();
        let mut render_data = {
            let mut render_data_receiver = self.render_data_receiver.lock().unwrap();
            render_data_receiver.replace_vertices(vertices.clone().into_iter().flatten().collect());
            render_data_receiver.clone_data().1
        };
        loop {
            let now = Instant::now();
            timer.start();
            let mut pending_add_objects = Vec::new();
            let mut pending_remove_objects = Vec::new();
            for i in 0..self.objects.len() {
                let update_ctx = UpdateContext {
                    others: SafeObjectList::new(i, &self.objects),
                    pending_add_objects: &mut pending_add_objects,
                    pending_remove_objects: &mut pending_remove_objects,
                };
                let next_render_data = self.objects[i]
                    .borrow_mut()
                    .on_update(delta, update_ctx);
                render_data[i] = next_render_data;
            }

            let did_update_vertices = !pending_add_objects.is_empty() || !pending_remove_objects.is_empty();

            for remove_index in pending_remove_objects.into_iter().rev() {
                vertices.remove(remove_index);
                render_data.remove(remove_index);
                self.objects.remove(remove_index);
            }

            for mut new_obj in pending_add_objects {
                new_obj.on_ready();
                render_data.push(RenderData::empty());
                vertices.push(new_obj.create_vertices());
                self.objects.push(RefCell::new(new_obj));
            }
            {
                let mut render_data_receiver = self.render_data_receiver.lock().unwrap();
                if did_update_vertices {
                    render_data_receiver.replace_vertices(vertices.clone().into_iter().flatten().collect());
                }
                render_data_receiver.update_render_data(render_data.clone());
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
    fn replace_vertices(&mut self, vertices: Vec<Vec2>);
    fn update_render_data(&mut self, render_data: Vec<RenderData>);
    fn clone_data(&self) -> (Vec<Vec2>, Vec<RenderData>);
}
