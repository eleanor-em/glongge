pub mod sample;

use std::cell::RefCell;
use std::sync::{Arc, mpsc, Mutex};
use std::sync::mpsc::{Receiver, Sender};
use crate::assert::*;
use crate::core::vk_core::RenderEventHandler;
use crate::gg::core::{SceneInstruction, SceneObject, UpdateHandler};

pub struct Scene<RenderHandler: RenderEventHandler> {
    initial_objects: Option<Vec<RefCell<Box<dyn SceneObject>>>>,
    render_data_receiver: Option<Arc<Mutex<RenderHandler::DataReceiver>>>,
    scene_instruction_rx: Option<Receiver<SceneInstruction>>,
    scene_instruction_tx: Sender<SceneInstruction>,
}

impl<RenderHandler: RenderEventHandler> Scene<RenderHandler> {
    pub fn new(initial_objects: Vec<RefCell<Box<dyn SceneObject>>>, render_handler: &RenderHandler) -> Self {
        let (scene_instruction_tx, scene_instruction_rx) = mpsc::channel();
        Self {
            initial_objects: Some(initial_objects),
            render_data_receiver: Some(render_handler.get_receiver()),
            scene_instruction_tx,
            scene_instruction_rx: Some(scene_instruction_rx),
        }
    }

    pub fn run(&mut self) {
        let initial_objects = self.initial_objects.take()
            .expect("run() already called for this scene!");
        let render_update_receiver = self.render_data_receiver.take()
            .expect("run() already called for this scene!");
        let scene_instruction_rx = self.scene_instruction_rx.take()
            .expect("run() already called for this scene!");
        let update_handler = UpdateHandler::new(
            initial_objects,
            render_update_receiver,
            self.scene_instruction_tx.clone(),
            scene_instruction_rx);
        std::thread::spawn(move || update_handler.consume());
    }

    pub fn stop(&self) {
        check!(self.initial_objects.is_none());
        self.scene_instruction_tx.send(SceneInstruction::Stop).unwrap();
    }
}
