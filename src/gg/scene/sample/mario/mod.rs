use std::sync::{Arc, Mutex};
use glongge_derive::register_object_type;
use crate::{
    core::input::InputHandler,
    gg::{
        render::BasicRenderHandler,
        scene::Scene,
        SceneObject,
        self,
    },
    resource::ResourceHandler,
};

mod player;
mod brick;

use player::*;
use brick::*;
use crate::core::linalg::{Vec2, Vec2Int};

const BRICK_COLLISION_TAG: &str = "BRICK";
const PLAYER_COLLISION_TAG: &str = "PLAYER";

pub fn create_scene(
    resource_handler: ResourceHandler,
    render_handler: BasicRenderHandler,
    input_handler: Arc<Mutex<InputHandler>>
) -> Scene<ObjectType, BasicRenderHandler> {
    let mut initial_objects: Vec<Box<dyn SceneObject<ObjectType>>> = Vec::new();
    for (tile_x, tile_y) in Vec2Int::range_from_zero([64, 2].into()) {
        initial_objects.push(Box::new(Brick::new(Vec2 {
            x: (tile_x as f64 + 0.5) * 16.,
            y: 768. - (tile_y as f64 + 0.5) * 16.
        })));
    }
    initial_objects.push(Box::new(Player::new()));
    initial_objects.push(Box::new(Brick::new(Vec2 { x: 120., y: 696. })));
    initial_objects.push(Box::new(Brick::new(Vec2 { x: 136., y: 712. })));

    Scene::new(
        initial_objects,
        input_handler,
        resource_handler,
        render_handler
    )
}

#[register_object_type]
pub enum ObjectType {
    Player,
    Brick,
}
