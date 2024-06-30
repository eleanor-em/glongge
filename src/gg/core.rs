use std::cell::{Ref, RefCell};

use crate::core::linalg::Vec2;

pub trait SceneObject: Send {
    fn create_vertices(&self) -> Vec<Vec2>;
    fn on_update(&mut self, delta: f64, others: SafeObjectList) -> RenderData;
    fn world_pos(&self) -> Vec2;
}

pub struct SafeObjectList<'a> {
    owner_index: usize,
    objects: &'a [RefCell<Box<dyn SceneObject>>],
    curr: usize,
}

impl<'a> SafeObjectList<'a> {
    pub fn new(owner_index: usize, objects: &'a [RefCell<Box<dyn SceneObject>>]) -> Self {
        Self { owner_index, objects, curr: 0, }
    }
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
