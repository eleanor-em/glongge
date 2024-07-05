pub mod sample;

use crate::{
    assert::*,
    core::vk_core::RenderEventHandler,
    gg::{SceneInstruction, SceneObject, UpdateHandler},
};
use std::{
    cell::RefCell,
    sync::{
        mpsc,
        mpsc::{Receiver, Sender},
        Arc, Mutex,
    },
};
use crate::core::input::InputHandler;

pub struct Scene<RenderHandler: RenderEventHandler> {
    initial_objects: Option<Vec<RefCell<Box<dyn SceneObject>>>>,
    render_data_receiver: Option<Arc<Mutex<RenderHandler::DataReceiver>>>,
    input_handler: Option<Arc<Mutex<InputHandler>>>,
    scene_instruction_rx: Option<Receiver<SceneInstruction>>,
    scene_instruction_tx: Sender<SceneInstruction>,
}

impl<RenderHandler: RenderEventHandler> Scene<RenderHandler> {
    pub fn new(initial_objects: Vec<Box<dyn SceneObject>>, render_handler: &RenderHandler, input_handler: Arc<Mutex<InputHandler>>) -> Self {
        let (scene_instruction_tx, scene_instruction_rx) = mpsc::channel();
        Self {
            initial_objects: Some(initial_objects.into_iter().map(RefCell::new).collect()),
            render_data_receiver: Some(render_handler.get_receiver()),
            input_handler: Some(input_handler),
            scene_instruction_tx,
            scene_instruction_rx: Some(scene_instruction_rx),
        }
    }

    pub fn run(&mut self) {
        let initial_objects = self
            .initial_objects
            .take()
            .expect("run() already called for this scene!");
        let render_update_receiver = self
            .render_data_receiver
            .take()
            .expect("run() already called for this scene!");
        let input_handler = self
            .input_handler
            .take()
            .expect("run() already called for this scene!");
        let scene_instruction_rx = self
            .scene_instruction_rx
            .take()
            .expect("run() already called for this scene!");
        let update_handler = UpdateHandler::new(
            initial_objects,
            render_update_receiver,
            input_handler,
            self.scene_instruction_tx.clone(),
            scene_instruction_rx,
        );
        std::thread::spawn(move || update_handler.consume());
    }

    pub fn stop(&self) {
        check!(self.initial_objects.is_none());
        self.scene_instruction_tx
            .send(SceneInstruction::Stop)
            .unwrap();
    }
}
