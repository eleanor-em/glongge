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
mod underground_floor;
mod enemy;
mod background;
mod block;

use player::*;
use floor::*;
use underground_floor::*;
use block::question_block::*;
use block::brick::*;
use block::underground_brick::*;
use enemy::goomba::*;
use background::hill1::*;
use background::hill2::*;
use background::hill3::*;
use background::hill4::*;
use block::pipe::*;
use block::decorative_pipe::*;
use crate::core::linalg::Vec2;
use crate::gg::AnySceneObject;
use crate::gg::scene::{Scene, SceneName, SceneStartInstruction};

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
const BLOCK_COLLISION_TAG: &str = "BLOCK";
const PIPE_COLLISION_TAG: &str = "PIPE";
const PLAYER_COLLISION_TAG: &str = "PLAYER";
const ENEMY_COLLISION_TAG: &str = "ENEMY";

#[derive(Copy, Clone)]
pub struct MarioScene;
impl Scene<ObjectType> for MarioScene {
    fn name(&self) -> SceneName { "mario".into() }

    fn create_objects(&self, entrance_id: usize) -> Vec<AnySceneObject<ObjectType>> {
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
            Hill2::new(Vec2Int {
                x: 60 * 16,
                y: 384 - 2 * 16 - 16,
            }),
            Hill1::new(Vec2Int {
                x: 64 * 16,
                y: 384 - 2 * 16 - 32,
            }),
        ];

        initial_objects.push(QuestionBlock::new(Vec2Int { x: 17 * 16, y: 384 - 6 * 16 }));
        initial_objects.push(Brick::new(Vec2Int { x: 21 * 16, y: 384 - 6 * 16 }));
        initial_objects.push(QuestionBlock::new(Vec2Int { x: 22 * 16, y: 384 - 6 * 16 }));
        initial_objects.push(Brick::new(Vec2Int { x: 23 * 16, y: 384 - 6 * 16 }));
        initial_objects.push(QuestionBlock::new(Vec2Int { x: 24 * 16, y: 384 - 6 * 16 }));
        initial_objects.push(Brick::new(Vec2Int { x: 25 * 16, y: 384 - 6 * 16 }));

        initial_objects.push(QuestionBlock::new(Vec2Int { x: 23 * 16, y: 384 - 10 * 16 }));
        initial_objects.push(Goomba::new(Vec2Int { x: 23 * 16, y: 384 - 3 * 16 }));

        initial_objects.push(match entrance_id {
            1 => Player::new(Vec2Int {
                x: 59 * 16,
                y: 384 - 4 * 16 - 8,
            }, true),
            _ => Player::new(Vec2Int {
                x: 16 * 3 + 8,
                y: 384 - 3 * 16 + 8
            }, false)
        });
        initial_objects.push(Pipe::new(Vec2Int {
            x: 29 * 16,
            y: 384 - 4 * 16,
        }, Vec2::up(), None));
        initial_objects.push(Pipe::new(Vec2Int {
            x: 39 * 16,
            y: 384 - 5 * 16,
        }, Vec2::up(), None));
        initial_objects.push(Pipe::new(Vec2Int {
            x: 47 * 16,
            y: 384 - 6 * 16,
        }, Vec2::up(), None));
        initial_objects.push(Pipe::new(Vec2Int {
            x: 58 * 16,
            y: 384 - 6 * 16,
        }, Vec2::up(), Some(SceneStartInstruction::new(MarioUndergroundScene{}.name(), 0))));
        // left wall
        for (tile_x, tile_y) in Vec2Int::range_from_zero([1, 24].into()) {
            initial_objects.push(Floor::new(Vec2Int {
                x: tile_x * 16,
                y: tile_y * 16,
            }));
        }
        initial_objects.push(Goomba::new(Vec2Int { x: 41 * 16, y: 384 - 3 * 16 }));
        initial_objects.push(Goomba::new(Vec2Int { x: 51 * 16 + 8, y: 384 - 3 * 16 }));
        initial_objects.push(Goomba::new(Vec2Int { x: 53 * 16 + 8, y: 384 - 3 * 16 }));
        // floor
        for (tile_x, tile_y) in Vec2Int::range_from_zero([69, 2].into()) {
            initial_objects.push(Floor::new(Vec2Int {
                x: (tile_x + 1) * 16,
                y: 384 - (tile_y + 1) * 16
            }));
        }
        for (tile_x, tile_y) in Vec2Int::range_from_zero([15, 2].into()) {
            initial_objects.push(Floor::new(Vec2Int {
                x: (tile_x + 72) * 16,
                y: 384 - (tile_y + 1) * 16
            }));
        }
        initial_objects.push(Brick::new(Vec2Int { x: 78 * 16, y: 384 - 6 * 16 }));
        initial_objects.push(QuestionBlock::new(Vec2Int { x: 79 * 16, y: 384 - 6 * 16 }));
        initial_objects.push(Brick::new(Vec2Int { x: 80 * 16, y: 384 - 6 * 16 }));
        for (tile_x, _tile_y) in Vec2Int::range_from_zero([8, 1].into()) {
            initial_objects.push(Brick::new(Vec2Int {
                x: (tile_x + 81) * 16,
                y: 384 - 10 * 16
            }));
        }
        initial_objects.push(Goomba::new(Vec2Int { x: 81 * 16, y: 384 - 11 * 16 }));
        initial_objects.push(Goomba::new(Vec2Int { x: 83 * 16, y: 384 - 11 * 16 }));
        for (tile_x, tile_y) in Vec2Int::range_from_zero([50, 2].into()) {
            initial_objects.push(Floor::new(Vec2Int {
                x: (tile_x + 90) * 16,
                y: 384 - (tile_y + 1) * 16
            }));
        }
        initial_objects
    }
}

#[derive(Copy, Clone)]
pub struct MarioUndergroundScene;
impl Scene<ObjectType> for MarioUndergroundScene {
    fn name(&self) -> SceneName { "mario-underground".into() }

    fn create_objects(&self, _entrance_id: usize) -> Vec<AnySceneObject<ObjectType>> {
        let mut initial_objects: Vec<Box<dyn SceneObject<ObjectType>>> = vec![
            Player::new(Vec2Int {
                x: 2*16 + 8,
                y: 8,
            }, false)
        ];
        // left wall
        for (tile_x, tile_y) in Vec2Int::range_from_zero([1, 24].into()) {
            initial_objects.push(UndergroundFloor::new(Vec2Int {
                x: tile_x * 16,
                y: tile_y * 16,
            }));
        }
        // floor
        for (tile_x, tile_y) in Vec2Int::range_from_zero([17, 2].into()) {
            initial_objects.push(UndergroundFloor::new(Vec2Int {
                x: (tile_x + 1) * 16,
                y: 384 - (tile_y + 1) * 16
            }));
        }
        for (tile_x, tile_y) in Vec2Int::range_from_zero([7, 3].into()) {
            initial_objects.push(UndergroundBrick::new(Vec2Int {
                x: (tile_x + 4) * 16,
                y: 384 - (tile_y + 3) * 16
            }));
        }
        for (tile_x, _tile_y) in Vec2Int::range_from_zero([7, 1].into()) {
            initial_objects.push(UndergroundBrick::new(Vec2Int {
                x: (tile_x + 4) * 16,
                y: 0,
            }));
        }
        initial_objects.push(Pipe::new(Vec2Int {
            x: 14 * 16,
            y: 384 - 4*16,
        }, Vec2::left(), Some(SceneStartInstruction::new(MarioScene.name(), 1))));
        initial_objects.push(DecorativePipe::new(Vec2Int {
            x: 16 * 16,
            y: 0,
        }));
        initial_objects
    }

}

#[register_object_type]
pub enum ObjectType {
    Player,
    Floor,
    UndergroundFloor,
    QuestionBlock,
    Brick,
    UndergroundBrick,
    Goomba,
    Hill1,
    Hill2,
    Hill3,
    Hill4,
    Pipe,
    DecorativePipe,
}
