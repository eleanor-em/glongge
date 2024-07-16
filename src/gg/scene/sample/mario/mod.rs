use std::sync::{Arc, Mutex};
use glongge_derive::register_object_type;
use crate::{
    core::input::InputHandler,
    gg,
    gg::{
        render::BasicRenderHandler,
        scene::Scene
    },
    resource::ResourceHandler
};

mod player;
use player::*;
use crate::gg::SceneObject;

pub fn create_scene(
    resource_handler: ResourceHandler,
    render_handler: BasicRenderHandler,
    input_handler: Arc<Mutex<InputHandler>>
) -> Scene<ObjectType, BasicRenderHandler> {
    Scene::new(
        vec![
            Box::new(Player::new()),
        ],
        input_handler,
        resource_handler,
        render_handler
    )
}

#[register_object_type]
pub enum ObjectType {
    Player,
}