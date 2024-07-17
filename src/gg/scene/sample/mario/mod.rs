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
mod floor;
mod question_block;
mod brick;
mod goomba;

use player::*;
use floor::*;
use question_block::*;
use brick::*;
use goomba::*;
use crate::core::linalg::Vec2Int;

const fn from_nes(pixels: u8, subpixels: u8, subsubpixels: u8, subsubsubpixels: u8) -> f64 {
    // fixed update at 100 fps
    (pixels as f64
        + subpixels as f64 / 16.
        + subsubpixels as f64 / 256.
        + subsubsubpixels as f64 / 4096.) * 60. / 100.
}
const fn from_nes_accel(pixels: u8, subpixels: u8, subsubpixels: u8, subsubsubpixels: u8) -> f64 {
    // fixed update at 100 fps
    from_nes(pixels, subpixels, subsubpixels, subsubsubpixels) * (60. / 100.)
}
const BASE_GRAVITY: f64 = from_nes_accel(0, 7, 0, 0);
const BRICK_COLLISION_TAG: &str = "BRICK";
const PLAYER_COLLISION_TAG: &str = "PLAYER";
const ENEMY_COLLISION_TAG: &str = "ENEMY";

pub fn create_scene(
    resource_handler: ResourceHandler,
    render_handler: BasicRenderHandler,
    input_handler: Arc<Mutex<InputHandler>>
) -> Scene<ObjectType, BasicRenderHandler> {
    let mut initial_objects: Vec<Box<dyn SceneObject<ObjectType>>> = Vec::new();
    // left wall
    for (tile_x, tile_y) in Vec2Int::range_from_zero([1, 24].into()) {
        initial_objects.push(Floor::new(Vec2Int {
            x: tile_x * 16,
            y: tile_y * 16,
        }));
    }
    // floor
    for (tile_x, tile_y) in Vec2Int::range_from_zero([31, 2].into()) {
        initial_objects.push(Floor::new(Vec2Int {
            x: (tile_x + 1) * 16,
            y: 384 - (tile_y + 1) * 16
        }));
    }

    initial_objects.push(QuestionBlock::new(Vec2Int { x: 20 * 16, y: 384 - 6 * 16 }));
    initial_objects.push(Brick::new(Vec2Int { x: 24 * 16, y: 384 - 6 * 16 }));
    initial_objects.push(QuestionBlock::new(Vec2Int { x: 25 * 16, y: 384 - 6 * 16 }));
    initial_objects.push(Brick::new(Vec2Int { x: 26 * 16, y: 384 - 6 * 16 }));
    initial_objects.push(QuestionBlock::new(Vec2Int { x: 27 * 16, y: 384 - 6 * 16 }));
    initial_objects.push(Brick::new(Vec2Int { x: 28 * 16, y: 384 - 6 * 16 }));

    initial_objects.push(QuestionBlock::new(Vec2Int { x: 26 * 16, y: 384 - 10 * 16 }));
    initial_objects.push(Goomba::new(Vec2Int { x: 26 * 16, y: 384 - 3 * 16 }));

    initial_objects.push(Player::new());

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
    Floor,
    QuestionBlock,
    Brick,
    Goomba,
}
