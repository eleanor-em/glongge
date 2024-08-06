use std::{
    any::Any,
    collections::BTreeMap,
    sync::{
        Arc,
        mpsc,
        mpsc::{Receiver, Sender},
        Mutex
    },
};
use crate::{
    resource::ResourceHandler,
    core::{
        update::{
            collision::CollisionResponse,
            ObjectContext,
            UpdateContext,
            UpdateHandler
        },
        AnySceneObject,
        input::InputHandler,
        ObjectTypeEnum,
        prelude::*,
        SceneObjectWithId,
        render::{RenderInfo, RenderDataChannel, RenderItem},
    }
};
use crate::core::render::RenderHandler;
use crate::core::update::RenderContext;
use crate::gui::{GuiContext, GuiUi};
use crate::shader::ensure_shaders_locked;

#[derive(Clone)]
struct InternalScene<ObjectType: ObjectTypeEnum> {
    scene: Arc<Mutex<dyn Scene<ObjectType> + Send>>,
    name: SceneName,
    input_handler: Arc<Mutex<InputHandler>>,
    resource_handler: ResourceHandler,
    render_data_channel: Arc<Mutex<RenderDataChannel>>,
    tx: Sender<SceneHandlerInstruction>,
}

impl<ObjectType: ObjectTypeEnum> InternalScene<ObjectType> {
    fn new(scene: Arc<Mutex<dyn Scene<ObjectType> + Send>>,
           input_handler: Arc<Mutex<InputHandler>>,
           resource_handler: ResourceHandler,
           render_data_channel: Arc<Mutex<RenderDataChannel>>,
           tx: Sender<SceneHandlerInstruction>) -> Self {
        let name = scene.try_lock()
            .expect("scene locked in InternalScene::new(), could not get scene name")
            .name();
        Self {
            scene,
            name,
            input_handler,
            resource_handler,
            render_data_channel,
            tx,
        }
    }

    fn run(&self, data: Arc<Mutex<Vec<u8>>>, entrance_id: usize, current_scene_name: Arc<Mutex<Option<SceneName>>>) {
        let existing_name = current_scene_name.try_lock()
            .expect("scene locked in InternalScene::run()")
            .replace(self.name);
        check_eq!(existing_name, None::<SceneName>);

        let this = self.clone();
        let this_name = self.name;
        std::thread::spawn(move || {
            let initial_objects = {
                let mut scene = this.scene.try_lock()
                    .unwrap_or_else(|_| panic!("scene locked in InternalScene::run(): {this_name:?}"));
                scene.load(&data.try_lock()
                    .expect("scene_data still locked?"))
                    .unwrap_or_else(|_| panic!("could not load data for {this_name:?}"));
                scene.create_objects(entrance_id)
            };
            check_false!(initial_objects.is_empty(), "must create at least one object");
            let update_handler = UpdateHandler::new(
                initial_objects,
                this.input_handler,
                this.resource_handler,
                this.render_data_channel,
                this_name,
                data
            );
            let instruction = update_handler
                .context("failed to create scene: {this_name:?}").unwrap()
                .consume()
                .context("scene exited with error: {this_name:?}").unwrap();
            current_scene_name.lock().unwrap().take();
            this.tx.send(instruction).expect("failed to send scene instruction");
        });
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub struct SceneName(&'static str);

impl SceneName {
    pub fn new(text: &'static str) -> Self {
        Self(text)
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub struct SceneDestination {
    name: SceneName,
    entrance_id: usize,
}

impl SceneDestination {
    pub fn new(name: SceneName, entrance_id: usize) -> Self {
        Self { name, entrance_id }
    }
}

pub trait Scene<ObjectType: ObjectTypeEnum>: Send {
    fn name(&self) -> SceneName;
    fn create_objects(&self, entrance_id: usize) -> Vec<AnySceneObject<ObjectType>>;

    #[allow(unused_variables)]
    fn load(&mut self, data: &[u8]) -> Result<()> { Ok(()) }
    #[allow(unused_variables)]
    fn initial_data(&self) -> Vec<u8> { Vec::new() }
    
    fn at_entrance(&self, entrance_id: usize) -> SceneDestination {
        SceneDestination::new(self.name(), entrance_id)
    }
}

#[allow(dead_code)]
pub(crate) enum SceneHandlerInstruction {
    Exit,
    Goto(SceneDestination),
}

#[allow(private_bounds)]
pub struct SceneHandler<ObjectType: ObjectTypeEnum> {
    input_handler: Arc<Mutex<InputHandler>>,
    resource_handler: ResourceHandler,
    render_handler: RenderHandler,
    scenes: BTreeMap<SceneName, InternalScene<ObjectType>>,
    scene_data: BTreeMap<SceneName, Arc<Mutex<Vec<u8>>>>,
    current_scene_name: Arc<Mutex<Option<SceneName>>>,
    tx: Sender<SceneHandlerInstruction>,
    rx: Receiver<SceneHandlerInstruction>,
}

#[allow(private_bounds)]
impl<ObjectType: ObjectTypeEnum> SceneHandler<ObjectType> {
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
        check_false!(self.scene_data.contains_key(&scene.name()));
        self.scene_data.insert(scene.name(), Arc::new(Mutex::new(
            scene.initial_data()
        )));
        self.scenes.insert(scene.name(), InternalScene::new(
            Arc::new(Mutex::new(scene)),
            self.input_handler.clone(),
            self.resource_handler.clone(),
            self.render_handler.get_receiver(),
            self.tx.clone()
        ));
    }
    pub fn consume_with_scene(mut self, mut name: SceneName, mut entrance_id: usize) {
        ensure_shaders_locked();
        loop {
            self.run_scene(name, entrance_id);
            match self.rx.recv().expect("failed to receive scene instruction") {
                SceneHandlerInstruction::Exit => std::process::exit(0),
                SceneHandlerInstruction::Goto(SceneDestination {
                                                  name: next_name,
                                                  entrance_id: next_entrance_id
                                              }) => {
                    name = next_name;
                    entrance_id = next_entrance_id;
                }
            }
        }
    }
    fn run_scene(&mut self, name: SceneName, entrance_id: usize) {
        if let (Some(scene), Some(scene_data)) = (self.scenes.get(&name), self.scene_data.get(&name)) {
            info!("starting scene: {:?} [entrance {}]", name, entrance_id);
            scene.run(scene_data.clone(),
                      entrance_id,
                      self.current_scene_name.clone());
        } else {
            error!("could not start scene {:?}: scene missing?", name);
        }
    }
}

pub trait SceneObject<ObjectType: ObjectTypeEnum>: 'static {
    fn get_type(&self) -> ObjectType;
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;

    fn name(&self) -> String { format!("{:?}", self.get_type()) }

    #[allow(clippy::new_ret_no_self)]
    fn create() -> AnySceneObject<ObjectType> where Self: Default { AnySceneObject::new(Self::default()) }

    #[allow(unused_variables)]
    fn on_preload(&mut self, resource_handler: &mut ResourceHandler) -> Result<()> { Ok(()) }
    #[allow(unused_variables)]
    fn on_load(&mut self, object_ctx: &mut ObjectContext<ObjectType>, resource_handler: &mut ResourceHandler) -> Result<Option<RenderItem>> {
        Ok(None)
    }
    #[allow(unused_variables)]
    fn on_ready(&mut self, ctx: &mut UpdateContext<ObjectType>) {}

    #[allow(unused_variables)]
    fn on_update_begin(&mut self, ctx: &mut UpdateContext<ObjectType>) {}
    #[allow(unused_variables)]
    fn on_update(&mut self, ctx: &mut UpdateContext<ObjectType>) {}
    #[allow(unused_variables)]
    fn on_update_end(&mut self, ctx: &mut UpdateContext<ObjectType>) {}
    #[allow(unused_variables)]
    fn on_fixed_update(&mut self, ctx: &mut UpdateContext<ObjectType>) {}
    #[allow(unused_variables)]
    fn on_collision(&mut self, ctx: &mut UpdateContext<ObjectType>, other: SceneObjectWithId<ObjectType>, mtv: Vec2) -> CollisionResponse {
        CollisionResponse::Done
    }

    fn as_renderable_object(&mut self) -> Option<&mut dyn RenderableObject<ObjectType>> {
        None
    }
    fn as_gui_object(&self) -> Option<&dyn GuiObject<ObjectType>> { None }
    fn emitting_tags(&self) -> Vec<&'static str> { [].into() }
    fn listening_tags(&self) -> Vec<&'static str> { [].into() }
}

pub trait RenderableObject<ObjectType: ObjectTypeEnum>: SceneObject<ObjectType> {
    #[allow(unused_variables)]
    fn on_render(&mut self, render_ctx: &mut RenderContext) {}
    fn render_info(&self) -> RenderInfo;
}

pub type GuiClosure = dyn FnOnce(&GuiContext) + Send;
pub type GuiInsideClosure = dyn FnOnce(&mut GuiUi) + Send;
pub trait GuiObject<ObjectType: ObjectTypeEnum>: SceneObject<ObjectType> {
    fn on_gui(&self, ctx: &UpdateContext<ObjectType>) -> Box<GuiInsideClosure>;
}

impl<ObjectType, T> From<Box<T>> for Box<dyn SceneObject<ObjectType>>
where
    ObjectType: ObjectTypeEnum,
    T: SceneObject<ObjectType>
{
    fn from(value: Box<T>) -> Self { value }
}

#[allow(dead_code)]
pub enum SceneInstruction {
    Pause,
    Resume,
    Stop,
    Goto(SceneDestination),
}
