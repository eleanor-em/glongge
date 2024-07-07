pub mod sample;

use std::sync::{
    mpsc,
    mpsc::{Receiver, Sender},
    Arc, Mutex,
};
use crate::{
    assert::*,
    core::{
        input::InputHandler,
        vk_core::RenderEventHandler,
    },
    gg::{
        AnySceneObject,
        ObjectTypeEnum,
        SceneInstruction,
        SceneObject,
        UpdateHandler,
    }
};

struct SceneInnerData<ObjectType: ObjectTypeEnum, RenderHandler: RenderEventHandler> {
    initial_objects: Vec<Box<dyn SceneObject<ObjectType>>>,
    render_data_receiver: Arc<Mutex<RenderHandler::DataReceiver>>,
    input_handler: Arc<Mutex<InputHandler>>,
    scene_instruction_rx: Receiver<SceneInstruction>,
}

pub struct Scene<ObjectType: ObjectTypeEnum, RenderHandler: RenderEventHandler> {
    scene_instruction_tx: Sender<SceneInstruction>,
    inner: Option<SceneInnerData<ObjectType, RenderHandler>>,
}

impl<ObjectType: ObjectTypeEnum, RenderHandler: RenderEventHandler> Scene<ObjectType, RenderHandler> {
    pub fn new(initial_objects: Vec<AnySceneObject<ObjectType>>,
               render_handler: &RenderHandler,
               input_handler: Arc<Mutex<InputHandler>>) -> Self {
        let (scene_instruction_tx, scene_instruction_rx) = mpsc::channel();
        Self {
            scene_instruction_tx,
            inner: Some(SceneInnerData {
                initial_objects,
                render_data_receiver: render_handler.get_receiver(),
                input_handler,
                scene_instruction_rx,
            }),
        }
    }

    pub fn run(&mut self) {
        let inner = self.inner.take().expect("run() already called for this scene!");
        let scene_instruction_tx = self.scene_instruction_tx.clone();
        std::thread::spawn(move || {
            let update_handler = UpdateHandler::new(
                inner.initial_objects,
                inner.render_data_receiver,
                inner.input_handler,
                scene_instruction_tx,
                inner.scene_instruction_rx,
            );
            update_handler.consume();
        });
    }

    pub fn stop(&self) {
        check!(self.inner.is_none());
        self.scene_instruction_tx
            .send(SceneInstruction::Stop)
            .unwrap();
    }
}
