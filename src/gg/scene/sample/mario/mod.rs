#[allow(unused_imports)]
use crate::core::prelude::*;

use glongge_derive::register_object_type;
use crate::{
    core::{
        linalg::Vec2Int,
    },
    gg::{
        SceneObject,
        self,
    },
};

mod player;
mod floor;
mod enemy;
mod background;
mod block;

use player::*;
use floor::*;
use block::question_block::*;
use block::brick::*;
use enemy::goomba::*;
use background::hill1::*;
use background::hill2::*;
use background::hill3::*;
use background::hill4::*;
use block::pipe::*;
use crate::gg::AnySceneObject;
use crate::gg::scene::{Scene, SceneName};

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

#[derive(Copy, Clone)]
pub struct MarioScene {}
impl Scene<ObjectType> for MarioScene {
    fn name(&self) -> SceneName { "mario".into() }

    fn create_objects(&self, _entrance_id: usize) -> Vec<AnySceneObject<ObjectType>> {
        let mut initial_objects: Vec<Box<dyn SceneObject<ObjectType>>> = vec![
            Hill1::new(Vec2Int {
                x: 16,
                y: 384 - 2 * 16 - 48,
            }),
            Hill2::new(Vec2Int {
                x: 12 * 16,
                y: 384 - 2 * 16 - 16,
            }),
            Hill3::new(Vec2Int {
                x: 17 * 16,
                y: 384 - 2 * 16 - 32,
            }),
            Hill4::new(Vec2Int {
                x: 24 * 16,
                y: 384 - 2 * 16 - 16,
            }),
            Hill1::new(Vec2Int {
                x: 49 * 16,
                y: 384 - 2 * 16 - 48,
            }),
        ];
        // left wall
        for (tile_x, tile_y) in Vec2Int::range_from_zero([1, 24].into()) {
            initial_objects.push(Floor::new(Vec2Int {
                x: tile_x * 16,
                y: tile_y * 16,
            }));
        }
        initial_objects.push(Pipe::new(Vec2Int {
            x: 29 * 16,
            y: 384 - 4 * 16,
        }));
        initial_objects.push(Pipe::new(Vec2Int {
            x: 39 * 16,
            y: 384 - 5 * 16,
        }));
        initial_objects.push(Goomba::new(Vec2Int { x: 41 * 16, y: 384 - 3 * 16 }));
        initial_objects.push(Pipe::new(Vec2Int {
            x: 47 * 16,
            y: 384 - 6 * 16,
        }));
        initial_objects.push(Goomba::new(Vec2Int { x: 51 * 16 + 8, y: 384 - 3 * 16 }));
        initial_objects.push(Goomba::new(Vec2Int { x: 53 * 16 + 8, y: 384 - 3 * 16 }));
        initial_objects.push(Pipe::new(Vec2Int {
            x: 58 * 16,
            y: 384 - 6 * 16,
        }));
        // floor
        for (tile_x, tile_y) in Vec2Int::range_from_zero([69, 2].into()) {
            initial_objects.push(Floor::new(Vec2Int {
                x: (tile_x + 1) * 16,
                y: 384 - (tile_y + 1) * 16
            }));
        }

        initial_objects.push(QuestionBlock::new(Vec2Int { x: 17 * 16, y: 384 - 6 * 16 }));
        initial_objects.push(Brick::new(Vec2Int { x: 21 * 16, y: 384 - 6 * 16 }));
        initial_objects.push(QuestionBlock::new(Vec2Int { x: 22 * 16, y: 384 - 6 * 16 }));
        initial_objects.push(Brick::new(Vec2Int { x: 23 * 16, y: 384 - 6 * 16 }));
        initial_objects.push(QuestionBlock::new(Vec2Int { x: 24 * 16, y: 384 - 6 * 16 }));
        initial_objects.push(Brick::new(Vec2Int { x: 25 * 16, y: 384 - 6 * 16 }));

        initial_objects.push(QuestionBlock::new(Vec2Int { x: 23 * 16, y: 384 - 10 * 16 }));
        initial_objects.push(Goomba::new(Vec2Int { x: 23 * 16, y: 384 - 3 * 16 }));

        initial_objects.push(Player::new());
        initial_objects
    }
}

#[register_object_type]
pub enum ObjectType {
    Player,
    Floor,
    QuestionBlock,
    Brick,
    Goomba,
    Hill1,
    Hill2,
    Hill3,
    Hill4,
    Pipe,
}
