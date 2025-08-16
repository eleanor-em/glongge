use crate::core::render::RenderHandler;
use crate::core::update::RenderContext;
use crate::gui::GuiUi;
use crate::shader::ensure_shaders_locked;
use crate::util::UniqueShared;
use crate::{
    core::{
        SceneObjectWrapper, TreeSceneObject,
        input::InputHandler,
        prelude::*,
        render::{RenderItem, ShaderExec},
        update::{ObjectContext, UpdateContext, UpdateHandler, collision::CollisionResponse},
    },
    resource::ResourceHandler,
};
use egui::text::LayoutJob;
use egui::{Color32, TextFormat};
use std::cell::RefCell;
use std::rc::Rc;
use std::{
    any::Any,
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

struct InternalScene {
    scene: Arc<Mutex<dyn Scene + Send>>,
    name: SceneName,
    input_handler: Arc<Mutex<InputHandler>>,
    render_handler: RenderHandler,
    update_handler: Option<UpdateHandler>,
}

impl InternalScene {
    fn new(
        scene: Arc<Mutex<dyn Scene + Send>>,
        input_handler: Arc<Mutex<InputHandler>>,
        render_handler: RenderHandler,
    ) -> Self {
        let name = scene
            .try_lock()
            .expect("scene locked in InternalScene::new(), could not get scene name")
            .name();
        Self {
            scene,
            name,
            input_handler,
            render_handler,
            update_handler: None,
        }
    }

    fn init(&mut self, data: UniqueShared<Vec<u8>>, entrance_id: usize) {
        let initial_objects = {
            let mut scene = self.scene.try_lock().unwrap_or_else(|_| {
                panic!("scene locked in InternalScene::run(): {:?}", self.name)
            });
            scene
                .load(&data.lock())
                .unwrap_or_else(|_| panic!("could not load data for {:?}", self.name));
            scene.create_objects(entrance_id)
        };
        check_false!(
            initial_objects.is_empty(),
            "must create at least one object"
        );
        self.update_handler = Some(
            UpdateHandler::new(
                initial_objects,
                self.input_handler.clone(),
                &self.render_handler,
                self.name,
                data,
            )
            .context("failed to create scene: {this_name:?}")
            .unwrap(),
        );
    }

    fn run_update(&mut self) -> Option<SceneHandlerInstruction> {
        let instruction = self.update_handler.as_mut().unwrap().run_update()?;
        Some(
            instruction
                .with_context(|| format!("scene exited with error: {:?}", self.name))
                .unwrap(),
        )
    }
}

/// A key to uniquely identify a scene in scene management collections and for scene navigation.
/// Simple wrapper around `&'static str`.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub struct SceneName(&'static str);

impl SceneName {
    pub fn new(text: &'static str) -> Self {
        Self(text)
    }
}

/// Specifies a scene destination with an entrance point.
/// `entrance_id` allows objects to change their behaviour based on how the scene was entered.
/// For example, in a Mario-style game, different `entrance_id`s could represent different pipes
/// or doors that lead into the scene, each positioning the player at a different location.
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

/// Defines a game scene that can create and manage game objects.
///
/// A Scene represents a distinct section of the game, like a level, menu, or other self-contained
/// game state.
/// Each scene is responsible for:
/// - Creating and initialising its objects when entered
/// - Managing scene-specific data and state
/// - Handling scene transitions through entrances/exits
///
/// # Implementation Notes
/// Scenes must be `Send` to allow transfer between threads during scene transitions.
///
/// # Examples
///
/// ```ignore
/// use glongge::core::prelude::*;
///
/// struct Level1 {
///     player_start: Vec2,
/// }
///
/// impl Scene for Level1 {
///     fn name(&self) -> SceneName {
///         SceneName::new("level_1")
///     }
///
///     fn create_objects(&self, entrance_id: usize) -> Vec<SceneObjectWrapper> {
///         // Create initial objects for the level
///         scene_object_vec![
///             Box::new(Player::new(self.player_start)),
///             Box::new(Platform::new()),
///         ]
///     }
/// }
/// ```
pub trait Scene: Send {
    fn name(&self) -> SceneName;

    /// Loads scene-specific data from a byte array.
    ///
    /// This method is called before `create_objects()` to load any scene-specific data that was
    /// previously saved or provided. The default implementation does nothing.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use glongge::core::prelude::*;
    ///
    /// #[derive(Default, Serialize, Deserialize)]
    /// struct MySceneData {
    ///     player_start: Vec2,
    /// }
    /// fn load(&mut self, data: &[u8]) -> Result<()> {
    ///     // Deserialize scene data from bytes
    ///     let scene_data: MySceneData = bincode::deserialize(data)?;
    ///     self.player_start = scene_data.player_start;
    ///     Ok(())
    /// }
    /// fn create_objects(&self, entrance_id: usize) -> Vec<SceneObjectWrapper> {
    ///     // Create initial objects for the level
    ///     scene_object_vec![
    ///         Box::new(Player::new(self.player_start)),
    ///         Box::new(Platform::new()),
    ///     ]
    /// }
    /// ```
    #[allow(unused_variables)]
    fn load(&mut self, data: &[u8]) -> Result<()> {
        Ok(())
    }
    /// Create the initial objects for the scene.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// fn create_objects(&self, entrance_id: usize) -> Vec<SceneObjectWrapper> {
    ///     // Create initial objects for the level
    ///     scene_object_vec![
    ///         Box::new(Player::new(self.player_start)),
    ///         Box::new(Platform::new()),
    ///     ]
    /// }
    /// ```
    fn create_objects(&self, entrance_id: usize) -> Vec<SceneObjectWrapper>;

    #[allow(unused_variables)]
    fn initial_data(&self) -> Vec<u8> {
        Vec::new()
    }

    /// Returns a `SceneDestination` for a specific entrance point in this scene.
    ///
    /// This helper method simplifies creating scene destinations when transitioning between scenes.
    /// The `entrance_id` parameter specifies which entry point in the scene to use, allowing for
    /// multiple entry points like doors, pipes, or portals.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // In MarioUndergroundScene, define entrance IDs as constants
    /// const PIPE_ENTRANCE: usize = 0;
    /// const DOOR_ENTRANCE: usize = 1;
    ///
    /// // Use at_entrance() to create a destination when transitioning scenes
    /// fn on_update(&mut self, ctx: &mut UpdateContext) {
    ///     if self.player_entered_pipe {
    ///         // Go to underground scene through pipe entrance
    ///         ctx.goto(underground_scene.at_entrance(PIPE_ENTRANCE));
    ///     }
    /// }
    /// ```
    fn at_entrance(&self, entrance_id: usize) -> SceneDestination {
        SceneDestination::new(self.name(), entrance_id)
    }
}

pub(crate) enum SceneHandlerInstruction {
    Exit,
    Goto(SceneDestination),
}

/// Manages the lifecycle and transitions between game scenes.
///
/// The `[SceneHandler`] is responsible for:
/// - Creating and storing game scenes
/// - Managing scene transitions
/// - Managing scene-specific data persistence
///
/// # Examples
///
/// ```ignore
/// use glongge::core::prelude::*;
///
/// GgContextBuilder::::new([1280, 800])?
///     .with_global_scale_factor(2.0)
///     .build_and_run_window(|scene_handler| {
///         std::thread::spawn(move || {
///             // Create scene handler
///             let mut scene_handler = scene_handler.build(input_handler, resource_handler, render_handler);
///             
///             // Add scenes
///             scene_handler.create_scene(MainMenuScene);
///             scene_handler.create_scene(Level1Scene);
///
///             // Start game with initial scene
///             scene_handler.consume_with_scene(MainMenuScene.name(), 0);
///         });
///     })
/// ```
pub struct SceneHandler {
    input_handler: Arc<Mutex<InputHandler>>,
    render_handler: RenderHandler,
    scenes: BTreeMap<SceneName, Rc<RefCell<InternalScene>>>,
    scene_data: BTreeMap<SceneName, UniqueShared<Vec<u8>>>,
    current_scene: Option<Rc<RefCell<InternalScene>>>,
}

// #[allow(private_bounds)]
impl SceneHandler {
    pub(crate) fn new(
        input_handler: Arc<Mutex<InputHandler>>,
        render_handler: RenderHandler,
    ) -> Self {
        Self {
            input_handler,
            render_handler,
            scenes: BTreeMap::new(),
            current_scene: None,
            scene_data: BTreeMap::new(),
        }
    }
    /// Creates and registers a new scene. See [`SceneHandler`] example.
    pub fn create_scene<S: Scene + 'static>(&mut self, scene: S) {
        check_false!(self.scenes.contains_key(&scene.name()));
        check_false!(self.scene_data.contains_key(&scene.name()));
        self.scene_data
            .insert(scene.name(), UniqueShared::new(scene.initial_data()));
        self.scenes.insert(
            scene.name(),
            Rc::new(RefCell::new(InternalScene::new(
                Arc::new(Mutex::new(scene)),
                self.input_handler.clone(),
                self.render_handler.clone(),
            ))),
        );
    }

    pub fn set_initial_scene(&mut self, name: SceneName, entrance_id: usize) {
        ensure_shaders_locked();
        check_is_none!(self.current_scene);
        self.current_scene = Some(self.init_scene(name, entrance_id));
    }

    pub fn run_update(&mut self) {
        let instruction = self
            .current_scene
            .as_ref()
            .unwrap()
            .borrow_mut()
            .run_update();
        match instruction {
            Some(SceneHandlerInstruction::Exit) => std::process::exit(0),
            Some(SceneHandlerInstruction::Goto(SceneDestination {
                name: next_name,
                entrance_id: next_entrance_id,
            })) => {
                self.current_scene = Some(self.init_scene(next_name, next_entrance_id));
            }
            None => {}
        }
    }
    fn init_scene(&mut self, name: SceneName, entrance_id: usize) -> Rc<RefCell<InternalScene>> {
        let (Some(scene), Some(scene_data)) =
            (self.scenes.get_mut(&name), self.scene_data.get(&name))
        else {
            panic!("could not start scene {name:?}: scene missing?");
        };
        info!("starting scene: {name:?} [entrance {entrance_id}]");
        scene.borrow_mut().init(scene_data.clone(), entrance_id);
        scene.clone()
    }
}

/// A trait representing an object that can exist in a game scene. Scene objects are the core
/// building blocks for game scenes, representing entities like players, enemies, platforms, UI
/// elements, etc.
///
/// # Lifecycle Methods
/// Scene objects have several lifecycle methods that are called in sequence:
/// - [`on_load()`](SceneObject::on_load) - Initial setup when added to scene
/// - [`on_ready()`](SceneObject::on_ready) - Called after all objects for this update are loaded
/// - [`on_update_begin()`](SceneObject::on_update_begin) - Start of each update frame
/// - [`on_update()`](SceneObject::on_update) - Main update logic
/// - [`on_fixed_update()`](SceneObject::on_fixed_update) - Physics/time-dependent updates
/// - [`on_collision()`](SceneObject::on_collision) - Handle object collisions
/// - [`on_update_end()`](SceneObject::on_update_end) - End of update frame
///
/// # Optional Rendering and GUI
/// Objects can optionally implement rendering and GUI functionality through the following traits:
/// - [`RenderableObject`] - For objects that need to be drawn to the screen
/// - [`GuiObject`] - For objects that display GUI elements
///
/// # Examples
///
/// Basic scene object with movement:
/// ```ignore
/// use glongge::core::prelude::*;
///
/// #[derive(Default)]
/// struct Player {
///     velocity: Vec2,
/// }
///
/// impl SceneObject for Player {
///     fn on_update(&mut self, ctx: &mut UpdateContext) {
///         // Handle input
///         if ctx.input().pressed(KeyCode::Space) {
///             self.velocity.y = -200.0; // Jump
///         }
///         
///         // Update position
///         let pos = &mut ctx.object_mut().transform_mut().centre;
///         *pos += self.velocity * ctx.delta().as_secs_f32();
///         
///         // Apply gravity
///         self.velocity.y += 400.0 * ctx.delta().as_secs_f32();
///     }
/// }
/// ```
///
/// Scene object with collision handling:
/// ```ignore
/// use glongge::core::prelude::*;
///
/// struct Platform;
///
/// impl SceneObject for Platform {
///     fn on_load(
///         &mut self,
///         object_ctx: &mut ObjectContext,
///         _resource_handler: &mut ResourceHandler,
///     ) -> Result<Option<RenderItem>> {
///         // Create collision shape
///         object_ctx.add_child(CollisionShape::from_rect(
///             Rect::from_centre_and_size(Vec2::zero(), Vec2::new(100.0, 20.0)),
///             &self.emitting_tags(),
///             &[],
///         ));
///         Ok(None)
///     }
///
///     fn emitting_tags(&self) -> Vec<&'static str> {
///         vec!["SOLID"] // Other objects can detect collisions with this platform
///     }
/// }
/// ```
pub trait SceneObject: 'static {
    /// Called when the object is first added to the object handler to set up sprites, rendering,
    /// and any initial setup not dependent on other objects existing.
    /// Generally, adding children should be done here (so that they are available in
    /// [`on_ready()`](SceneObject::on_ready)).
    /// Other objects added in the same update may not yet be available.
    ///
    /// # Example
    /// ```ignore
    /// fn on_load(
    ///     &mut self,
    ///     object_ctx: &mut ObjectContext,
    ///     resource_handler: &mut ResourceHandler,
    /// ) -> Result<Option<RenderItem>> {
    ///     // Setup initial vertices for rendering a triangle
    ///     let vertices = vec![
    ///         Vec2::new(-10.0, -10.0),
    ///         Vec2::new(10.0, -10.0),
    ///         Vec2::new(0.0, 10.0),
    ///     ];
    ///     
    ///     // Return render item with the triangle vertices
    ///     Ok(Some(RenderItem::from_raw_vertices(vertices)))
    /// }
    /// ```
    ///
    /// # Returns
    /// - `Ok(Some(item))` - Initial vertex data for rendering this object.
    /// - `Ok(None)` - This object does not require rendering.
    /// - `Err(_)` - Loading failed; the object should not be added.
    #[allow(unused_variables)]
    fn on_load(
        &mut self,
        object_ctx: &mut ObjectContext,
        resource_handler: &mut ResourceHandler,
    ) -> Result<Option<RenderItem>> {
        Ok(None)
    }
    /// Called after all objects for this update are added to the object handler.
    /// Use for initial setup that does depend on other objects.
    ///
    /// # Example
    ///
    /// ```ignore
    /// fn on_ready(&mut self, ctx: &mut UpdateContext) {
    ///     // Initialize position relative to other spawned objects
    ///     let others = ctx.object().others();
    ///     if !others.is_empty() {
    ///         // Position this object to the right of the last spawned object
    ///         let last_pos = ctx.object().absolute_transform_of(&others.last().unwrap()).centre;
    ///         ctx.object_mut().transform_mut().centre = last_pos + Vec2::right() * 50.0;
    ///     }
    /// }
    /// ```
    #[allow(unused_variables)]
    fn on_ready(&mut self, ctx: &mut UpdateContext) {}

    /// Called at the beginning of each update frame, before any other update methods.
    /// Use this for initial state changes that other objects may depend on.
    ///
    /// # Example
    ///
    /// ```ignore
    /// fn on_update_begin(&mut self, ctx: &mut UpdateContext) {
    ///     // Update position that other objects may depend on
    ///     ctx.object_mut().transform_mut().centre += self.velocity * ctx.delta().as_secs_f32();
    /// }
    /// ```
    #[allow(unused_variables)]
    fn on_update_begin(&mut self, ctx: &mut UpdateContext) {}
    /// Called during each update frame.
    /// This is the main update method where most object behavior and logic should be implemented.
    ///
    /// # Example
    ///
    /// ```ignore
    /// fn on_update(&mut self, ctx: &mut UpdateContext) {
    ///     // Check for input
    ///     if ctx.input().pressed(KeyCode::Space) {
    ///         self.jump();
    ///     }
    ///     
    ///     // Update position
    ///     ctx.object_mut().transform_mut().centre += self.velocity * ctx.delta_60fps();
    ///     
    ///     // Check for object removal
    ///     if self.health <= 0.0 {
    ///         ctx.object_mut().remove_this();
    ///     }
    /// }
    /// ```
    #[allow(unused_variables)]
    fn on_update(&mut self, ctx: &mut UpdateContext) {}
    /// Use mostly for things that have a higher-order dependency on time, e.g. acceleration.
    /// Called after `on_update()`, but before `on_collision()` and `on_update_end()`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// fn on_fixed_update(&mut self, ctx: &mut FixedUpdateContext) {
    ///     // Apply gravity acceleration
    ///     self.velocity.y += GRAVITY_ACCEL * ctx.delta_60fps();
    /// }
    /// ```
    #[allow(unused_variables)]
    fn on_fixed_update(&mut self, ctx: &mut FixedUpdateContext) {}
    /// Called after `on_fixed_update()`, but before `on_update_end()`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use glongge::core::prelude::*;
    ///
    /// fn on_collision(
    ///     &mut self,
    ///     ctx: &mut UpdateContext,
    ///     other: TreeSceneObject,
    ///     mtv: Vec2,
    /// ) -> CollisionResponse {
    ///     // Adjust velocity based on collision
    ///     if !mtv.dot(Vec2::right()).is_zero() {
    ///         self.vel.x = -self.vel.x;  // Reverse horizontal velocity
    ///     }
    ///     
    ///     // Move object out of collision
    ///     if other.emitting_tags().contains(&"BLOCK") {
    ///         ctx.transform_mut().centre += mtv;
    ///     }
    ///     
    ///     // Return Done to stop processing additional collisions
    ///     CollisionResponse::Done
    /// }
    /// ```
    #[allow(unused_variables)]
    fn on_collision(
        &mut self,
        ctx: &mut UpdateContext,
        other: &TreeSceneObject,
        mtv: Vec2,
    ) -> CollisionResponse {
        CollisionResponse::Done
    }
    #[allow(unused_variables)]
    /// Called at the end of each update frame, after object collisions and fixed updates.
    /// Use this for finalising state changes.
    ///
    /// # Example
    ///
    /// ```ignore
    /// fn on_update_end(&mut self, ctx: &mut UpdateContext) {
    ///     // Update animation state
    ///     self.animation_time += ctx.delta().as_secs_f32();
    ///     
    ///     // Check for state transitions
    ///     if self.is_grounded && self.velocity.y < 0.0 {
    ///         self.state = State::Idle;
    ///     }
    /// }
    /// ```
    fn on_update_end(&mut self, ctx: &mut UpdateContext) {}

    /// Returns a mutable reference to this object as a [`RenderableObject`], if it implements that
    /// trait.
    ///
    /// The default implementation returns `None`. Override this method to return `Some(self)` if
    /// your object implements `RenderableObject`.
    ///
    /// # Returns
    /// - `Some(&mut dyn RenderableObject)` if this object implements rendering
    /// - `None` if this object does not render
    fn as_renderable_object(&mut self) -> Option<&mut dyn RenderableObject> {
        None
    }

    /// Returns a mutable reference to this object as a [`GuiObject`], if it implements that trait.
    ///
    /// The default implementation returns `None`. Override this method to return `Some(self)` if
    /// your object implements `GuiObject`.
    ///
    /// # Returns
    /// - `Some(&mut dyn GuiObject)` if this object has GUI elements
    /// - `None` if this object has no GUI
    fn as_gui_object(&mut self) -> Option<&mut dyn GuiObject> {
        None
    }

    /// Returns a list of collision tags emitted by this object that other objects can listen for.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// fn emitting_tags(&self) -> Vec<&'static str> {
    ///     vec!["PLAYER", "CAMERA_FOCUS"]
    /// }
    /// ```
    fn emitting_tags(&self) -> Vec<&'static str> {
        [].into()
    }

    /// Returns a list of collision tags this object listens for from other objects. Only objects
    /// with these tags trigger [`on_collision()`](SceneObject::on_collision) events.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// fn listening_tags(&self) -> Vec<&'static str> {
    ///     vec!["WALL", "PICKUP"]  
    /// }
    /// ```
    fn listening_tags(&self) -> Vec<&'static str> {
        [].into()
    }

    /// Implemented by
    /// [`partially_derive_scene_object!`](glongge_derive::partially_derive_scene_object). Do not
    /// implement manually.
    fn as_any(&self) -> &dyn Any;
    /// Implemented by
    /// [`partially_derive_scene_object!`](glongge_derive::partially_derive_scene_object). Do not
    /// implement manually.
    fn as_any_mut(&mut self) -> &mut dyn Any;

    /// Meant to give a descriptive name of the object type. Can be overridden if the default
    /// (generated by
    /// [`partially_derive_scene_object!`](glongge_derive::partially_derive_scene_object)) would be
    /// confusing.
    fn gg_type_name(&self) -> String;
}

pub trait RenderableObject: SceneObject {
    /// Called during each render pass to update the render item for this object.
    ///
    /// This is useful for conditionally rendering different textures.
    ///
    /// # Example
    ///
    /// ```ignore
    /// fn on_render(&mut self, render_ctx: &mut RenderContext) {
    ///     if self.show_wireframe {
    ///         if !self.last_show_wireframe {
    ///             render_ctx.insert_render_item(&self.wireframe);
    ///         } else {
    ///             render_ctx.update_render_item(&self.wireframe);
    ///         }
    ///     }
    ///     if !self.show_wireframe && self.last_show_wireframe {
    ///         render_ctx.remove_render_item();
    ///     }
    ///     self.last_show_wireframe = self.show_wireframe;
    /// }
    /// ```
    #[allow(unused_variables)]
    fn on_render(&mut self, render_ctx: &mut RenderContext) {}
    /// Returns a list of shader execution configurations used to render this object.
    ///
    /// Each [`ShaderExec`] defines properties like color blending, texture mapping, and other
    /// rendering parameters that will be applied when drawing the object.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use glongge::core::prelude::*;
    /// use glongge::core::render::ShaderExec;
    /// fn shader_execs(&self) -> Vec<ShaderExec> {
    ///     // Basic colored triangle
    ///     vec![ShaderExec {
    ///         blend_col: Colour::red(),
    ///         ..Default::default()
    ///     }]
    /// }
    /// ```
    ///
    /// ```ignore
    /// use glongge::core::prelude::*;
    /// use glongge::core::render::ShaderExec;
    /// fn shader_execs(&self) -> Vec<ShaderExec> {
    ///     // Multiple shader passes for effects
    ///     vec![
    ///         // Base texture
    ///         ShaderExec {
    ///             material_id: self.texture_id,
    ///             ..Default::default()
    ///         },
    ///         // Overlay color
    ///         ShaderExec {
    ///             blend_col: Colour::rgba(1.0, 0.0, 0.0, 0.5),
    ///             ..Default::default()
    ///         }
    ///     ]
    /// }
    /// ```
    ///
    /// # Returns
    /// A vector of `ShaderExec` configurations defining how to render the object.
    fn shader_execs(&self) -> Vec<ShaderExec>;
}

pub(crate) type GuiClosure = dyn FnOnce(&egui::Context) + Send;
pub struct GuiCommand {
    func: Box<dyn FnOnce(&mut GuiUi) + Send>,
    title: Option<String>,
    add_separator: bool,
}
impl GuiCommand {
    pub fn new<F: FnOnce(&mut GuiUi) + Send + 'static>(f: F) -> Self {
        Self {
            func: Box::new(f),
            title: None,
            add_separator: true,
        }
    }

    #[must_use]
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    #[must_use]
    pub fn with_no_add_separator(mut self) -> Self {
        self.add_separator = false;
        self
    }

    pub fn call(self, ui: &mut GuiUi) {
        if let Some(title) = self.title {
            let mut layout_job = LayoutJob::default();
            layout_job.append(
                title.as_str(),
                0.0,
                TextFormat {
                    color: Color32::from_white_alpha(255),
                    ..Default::default()
                },
            );
            ui.add(egui::Label::new(layout_job).selectable(false));
        }
        (self.func)(ui);
    }

    #[must_use]
    pub fn join(self, other: GuiCommand) -> GuiCommand {
        let add_separator = self.add_separator;
        GuiCommand::new(move |ui| {
            self.call(ui);
            if add_separator {
                ui.separator();
            }
            other.call(ui);
        })
    }
}
pub trait GuiObject: SceneObject {
    /// Called during each frame to build and return GUI elements for this object.
    ///
    /// The `selected` parameter indicates if this object is currently selected in the scene.
    /// Returns a boxed closure that will be executed to render the GUI elements.
    /// # Examples
    ///
    /// Show some simple debugging information about the object:
    /// ```ignore
    /// fn on_gui(&mut self, _ctx: &UpdateContext, _selected: bool) -> GuiCommand {
    ///     let height = self.height;
    ///     let ground_height = self.last_ground_height;
    ///     let v_speed = self.v_speed;
    ///     Box::new(move |ui| {
    ///         ui.label(format!("height: {height:.1}"));
    ///         ui.label(format!("ground height: {ground_height:.1}"));
    ///         ui.label(format!("v_speed: {v_speed:.1}"));
    ///     })
    /// }
    /// ```
    /// Show some debugging information about the object, and allow editing it with
    /// [`EditCell`](crate::gui::EditCell)/[`EditCellSender`](crate::gui::EditCellSender):
    /// ```ignore
    /// fn on_gui(&mut self, ctx: &UpdateContext, _selected: bool) -> GuiCommand {
    ///     let extent = self.collider.extent();
    ///     let (next_x, next_y) = (
    ///         self.extent_cell_receiver_x.try_recv(),
    ///         self.extent_cell_receiver_y.try_recv(),
    ///     );
    ///     if next_x.is_some() || next_y.is_some() {
    ///         self.collider = self.collider.with_extent(Vec2 {
    ///             x: next_x.unwrap_or(extent.x).max(0.1),
    ///             y: next_y.unwrap_or(extent.y).max(0.1),
    ///         });
    ///         self.regenerate_wireframe(&ctx.absolute_transform());
    ///     }
    ///     self.extent_cell_receiver_x.update_live(extent.x);
    ///     self.extent_cell_receiver_y.update_live(extent.y);
    //
    ///     let extent_cell_sender_x = self.extent_cell_receiver_x.sender();
    ///     let extent_cell_sender_y = self.extent_cell_receiver_y.sender();
    ///
    ///     Box::new(move |ui| {
    ///         ui.add(egui::Label::new("Extent").selectable(false));
    ///         collider
    ///             .extent()
    ///             .build_gui(ui, 0.1, extent_cell_sender_x, extent_cell_sender_y);
    ///     })
    /// }
    /// ```
    /// Show some debugging information about the object, and allow copying it to the clipboard:
    /// ```ignore
    /// fn on_gui(&mut self, _ctx: &UpdateContext, selected: bool) -> GuiCommand {
    ///     self.gui_selected = selected || self.force_visible;
    ///     let string_desc = self
    ///         .control_points
    ///         .iter()
    ///         .map(|v| format!("\t{v:?},\n"))
    ///         .reduce(|acc: String, x: String| acc + &x)
    ///         .unwrap_or_default();
    ///     Box::new(move |ui| {
    ///         ui.with_layout(Layout::top_down(Align::Center), |ui| {
    ///             if ui.button("Copy as Vec").clicked() {
    ///                 ui.output_mut(|o| {
    ///                     o.copied_text = format!("vec![\n{string_desc}]\n");
    ///                 });
    ///             }
    ///         });
    ///     })
    /// }
    /// ```
    fn on_gui(&mut self, ctx: &UpdateContext, selected: bool) -> GuiCommand;
}

impl<T> From<Box<T>> for Box<dyn SceneObject>
where
    T: SceneObject,
{
    fn from(value: Box<T>) -> Self {
        value
    }
}

pub enum SceneInstruction {
    Pause,
    Resume,
    Stop,
    Goto(SceneDestination),
}
