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
use crate::core::linalg::Vec2Int;

const BRICK_COLLISION_TAG: &str = "BRICK";
const PLAYER_COLLISION_TAG: &str = "PLAYER";

pub fn create_scene(
    resource_handler: ResourceHandler,
    render_handler: BasicRenderHandler,
    input_handler: Arc<Mutex<InputHandler>>
) -> Scene<ObjectType, BasicRenderHandler> {
    let mut initial_objects: Vec<Box<dyn SceneObject<ObjectType>>> = Vec::new();
    for (tile_x, tile_y) in Vec2Int::range_from_zero([64, 2].into()) {
        initial_objects.push(Brick::new(Vec2Int {
            x: tile_x * 16,
            y: 384 - (tile_y + 1) * 16
        }));
    }
    initial_objects.push(Player::new());
    initial_objects.push(Brick::new(Vec2Int { x: 3 * 16, y: 384 - 5 * 16 }));
    initial_objects.push(Brick::new(Vec2Int { x: 4 * 16, y: 384 - 5 * 16 }));

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
