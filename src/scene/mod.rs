pub mod sample;

use std::cell::RefCell;
use std::sync::{Arc, Mutex};
use crate::core::vk_core::RenderEventHandler;
use crate::gg::core::{SceneObject, UpdateHandler};

pub struct Scene<RenderHandler: RenderEventHandler> {
    initial_objects: Option<Vec<RefCell<Box<dyn SceneObject>>>>,
    render_data_receiver: Option<Arc<Mutex<RenderHandler::Receiver>>>,
}

impl<RenderHandler: RenderEventHandler> Scene<RenderHandler> {
    pub fn new(initial_objects: Vec<RefCell<Box<dyn SceneObject>>>, render_handler: &RenderHandler) -> Self {
        Self {
            initial_objects: Some(initial_objects),
            render_data_receiver: Some(render_handler.get_receiver()),
        }
    }

    pub fn run(&mut self) {
        let initial_objects = self.initial_objects.take()
            .expect("run() already called for this scene!");
        let render_update_receiver = self.render_data_receiver.take()
            .expect("run() already called for this scene!");
        let update_handler = UpdateHandler::new(initial_objects, render_update_receiver);
        std::thread::spawn(move || update_handler.consume());
    }
}
