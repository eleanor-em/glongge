#[allow(unused_imports)]
use crate::core::prelude::*;

use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        mpsc,
        Mutex,
        mpsc::{Receiver, Sender}
    }
};
use crate::{
    core::{
        input::InputHandler,
        vk_core::RenderEventHandler,
        AnySceneObject,
        ObjectTypeEnum,
        UpdateHandler
    },
    resource::ResourceHandler
};
use crate::core::RenderInfoReceiver;

#[derive(Clone)]
struct InternalScene<ObjectType: ObjectTypeEnum, InfoReceiver: RenderInfoReceiver + 'static> {
    scene: Arc<Mutex<dyn Scene<ObjectType>>>,
    name: SceneName,
    input_handler: Arc<Mutex<InputHandler>>,
    resource_handler: ResourceHandler,
    render_info_receiver: Arc<Mutex<InfoReceiver>>,
    tx: Sender<SceneHandlerInstruction>,
}

impl<ObjectType: ObjectTypeEnum, InfoReceiver: RenderInfoReceiver + 'static> InternalScene<ObjectType, InfoReceiver> {
    fn new(scene: Arc<Mutex<dyn Scene<ObjectType>>>,
           input_handler: Arc<Mutex<InputHandler>>,
           resource_handler: ResourceHandler,
           render_info_receiver: Arc<Mutex<InfoReceiver>>,
           tx: Sender<SceneHandlerInstruction>) -> Self {
        let name = scene.try_lock()
            .expect("scene locked in InternalScene::new(), could not get scene name")
            .name();
        Self {
            scene,
            name,
            input_handler,
            resource_handler,
            render_info_receiver,
            tx,
        }
    }

    fn run(&self, data: Vec<u8>, entrance_id: usize, current_scene_name: Arc<Mutex<Option<SceneName>>>) {
        let existing_name = current_scene_name.try_lock()
            .expect("scene locked in InternalScene::run()")
            .replace(self.name);
        check_eq!(existing_name, None::<SceneName>);

        let this = self.clone();
        let this_name = self.name;
        let initial_objects = {
            let mut scene = this.scene.try_lock()
                .unwrap_or_else(|_| panic!("scene locked in InternalScene::run(): {this_name:?}"));
            scene.load(&data)
                .unwrap_or_else(|_| panic!("could not load data for {this_name:?}"));
            scene.create_objects(entrance_id)
        };
        std::thread::spawn(move || {
            let update_handler = UpdateHandler::new(
                initial_objects,
                this.input_handler,
                this.resource_handler,
                this.render_info_receiver,
                this_name,
                data
            );
            let instruction = update_handler
                .unwrap_or_else(|_| panic!("failed to create scene: {this_name:?}"))
                .consume()
                .unwrap_or_else(|_| panic!("scene exited with error: {this_name:?}"));
            current_scene_name.lock().unwrap().take();
            this.tx.send(instruction).expect("failed to send scene instruction");
        });
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub struct SceneName(&'static str);

impl From<&'static str> for SceneName {
    fn from(value: &'static str) -> Self {
        Self(value)
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub struct SceneStartInstruction {
    name: SceneName,
    entrance_id: usize,
}

impl SceneStartInstruction {
    pub fn new(name: SceneName, entrance_id: usize) -> Self {
        Self { name, entrance_id }
    }
}

pub trait Scene<ObjectType: ObjectTypeEnum> {
    fn name(&self) -> SceneName;
    fn create_objects(&self, entrance_id: usize) -> Vec<AnySceneObject<ObjectType>>;

    #[allow(unused_variables)]
    fn load(&mut self, data: &[u8]) -> Result<()> { Ok(()) }
    
    fn at_entrance(&self, entrance_id: usize) -> SceneStartInstruction {
        SceneStartInstruction::new(self.name(), entrance_id)
    }
}

#[allow(dead_code)]
pub(crate) enum SceneHandlerInstruction {
    Exit,
    Goto(SceneStartInstruction),
    SaveAndExit(Vec<u8>),
    SaveAndGoto(SceneStartInstruction, Vec<u8>),
}

pub struct SceneHandler<ObjectType: ObjectTypeEnum, RenderHandler: RenderEventHandler> {
    input_handler: Arc<Mutex<InputHandler>>,
    resource_handler: ResourceHandler,
    render_handler: RenderHandler,
    scenes: BTreeMap<SceneName, InternalScene<ObjectType, RenderHandler::InfoReceiver>>,
    scene_data: BTreeMap<SceneName, Vec<u8>>,
    current_scene_name: Arc<Mutex<Option<SceneName>>>,
    tx: Sender<SceneHandlerInstruction>,
    rx: Receiver<SceneHandlerInstruction>,
}

impl<ObjectType: ObjectTypeEnum, RenderHandler: RenderEventHandler> SceneHandler<ObjectType, RenderHandler> {
    pub fn new(input_handler: Arc<Mutex<InputHandler>>,
               resource_handler: ResourceHandler,
               render_handler: RenderHandler) -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            input_handler,
            resource_handler,
            render_handler,
            scenes: BTreeMap::new(),
            current_scene_name: Arc::new(Mutex::new(None)),
            scene_data: BTreeMap::new(),
            tx, rx,
        }
    }
    pub fn create_scene<S: Scene<ObjectType> + 'static>(&mut self, scene: S) {
        check_false!(self.scenes.contains_key(&scene.name()));
        self.scenes.insert(scene.name(), InternalScene::new(
            Arc::new(Mutex::new(scene)),
            self.input_handler.clone(),
            self.resource_handler.clone(),
            self.render_handler.get_receiver(),
            self.tx.clone()
        ));
    }
    pub fn consume_with_scene(mut self, mut name: SceneName, mut entrance_id: usize) {
        loop {
            self.run_scene(name, entrance_id);
            match self.rx.recv().expect("failed to receive scene instruction") {
                SceneHandlerInstruction::Exit => std::process::exit(0),
                SceneHandlerInstruction::Goto(SceneStartInstruction {
                                                  name: next_name,
                                                  entrance_id: next_entrance_id
                                              }) => {
                    name = next_name;
                    entrance_id = next_entrance_id;
                }
                SceneHandlerInstruction::SaveAndExit(data) => {
                    *self.scene_data.entry(name).or_default() = data;
                }
                SceneHandlerInstruction::SaveAndGoto(SceneStartInstruction {
                                                         name: next_name,
                                                         entrance_id: next_entrance_id
                                                     }, data) => {
                    *self.scene_data.entry(name).or_default() = data;
                    name = next_name;
                    entrance_id = next_entrance_id;
                }
            }
        }
    }
    fn run_scene(&self, name: SceneName, entrance_id: usize) {
        if let Some(scene) = self.scenes.get(&name) {
            info!("starting scene: {:?} [entrance {}]", name, entrance_id);
            scene.run(self.scene_data.get(&name).unwrap_or(&Vec::new()).clone(),
                      entrance_id,
                      self.current_scene_name.clone());
        } else {
            error!("could not start scene {:?}: scene missing?", name);
        }
    }
}
