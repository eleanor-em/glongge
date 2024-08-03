use std::collections::BTreeMap;
use serde::{Deserialize, Serialize};
use glongge::core::{
    prelude::*,
    scene::{Scene, SceneName},
};

pub mod player;
pub mod floor;
pub mod underground_floor;
pub mod enemy;
pub mod background;
pub mod block;
pub mod text_display;

use player::*;
use floor::*;
use underground_floor::*;
use text_display::*;
use block::question_block::*;
use block::brick::*;
use block::plain_block::*;
use block::underground_brick::*;
use block::pipe::*;
use block::flagpole::*;
use block::decorative_pipe::*;
use enemy::goomba::*;
use background::hill1::*;
use background::hill2::*;
use background::hill3::*;
use background::hill4::*;
use background::castle::*;
use glongge::core::update::builtin::Container;
use glongge::core::util::canvas::Canvas;
use crate::object_type::ObjectType;

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
const FLAG_COLLISION_TAG: &str = "FLAG";
const PIPE_COLLISION_TAG: &str = "PIPE";
const PLAYER_COLLISION_TAG: &str = "PLAYER";
const ENEMY_COLLISION_TAG: &str = "ENEMY";

#[derive(Default, Serialize, Deserialize)]
pub struct AliveEnemyMap {
    inner: BTreeMap<Vec2Int, bool>,
}

impl AliveEnemyMap {
    fn register(&mut self, initial_coord: Vec2Int) {
        self.inner.entry(initial_coord).or_insert(true);
    }

    fn is_alive(&self, initial_coord: Vec2Int) -> bool {
        self.inner.get(&initial_coord)
            .copied()
            .unwrap_or(true)
    }
    fn set_dead(&mut self, initial_coord: Vec2Int) {
        *self.inner.entry(initial_coord).or_default() = false;
    }
}

#[derive(Copy, Clone)]
pub struct MarioOverworldScene;

impl Scene<ObjectType> for MarioOverworldScene {
    fn name(&self) -> SceneName { SceneName::new("mario-overworld") }

    fn initial_data(&self) -> Vec<u8> {
        bincode::serialize(&AliveEnemyMap::default()).unwrap()
    }

    fn create_objects(&self, entrance_id: usize) -> Vec<AnySceneObject<ObjectType>> {
        vec![
            Canvas::create(),
            match entrance_id {
                1 => Player::create(Vec2Int {
                    x: 164 * 16,
                    y: 384 - 2 * 16 - 8,
                }, true),
                _ => Player::create(Vec2Int {
                    x: 8*16 + 8,
                    y: 384 - 3 * 16 + 8
                }, false),
            },
            Container::create("background", vec![
                Hill1::create(Vec2Int {
                    x: 16,
                    y: 384 - 2 * 16 - 48,
                }),
                Hill2::create(Vec2Int {
                    x: 12 * 16,
                    y: 384 - 2 * 16 - 16,
                }),
                Hill3::create(Vec2Int {
                    x: 17 * 16,
                    y: 384 - 2 * 16 - 32,
                }),
                Hill4::create(Vec2Int {
                    x: 24 * 16,
                    y: 384 - 2 * 16 - 16,
                }),
                Hill1::create(Vec2Int {
                    x: 49 * 16,
                    y: 384 - 2 * 16 - 48,
                }),
                Hill2::create(Vec2Int {
                    x: 60 * 16,
                    y: 384 - 2 * 16 - 16,
                }),
                Hill1::create(Vec2Int {
                    x: 64 * 16,
                    y: 384 - 2 * 16 - 32,
                }),
                Castle::create(Vec2Int {
                    x: 202*16,
                    y: 384 - 7*16,
                }),
            ]),
            Container::create("level", vec![
                QuestionBlock::create(Vec2Int { x: 17 * 16, y: 384 - 6 * 16 }),
                Brick::create(Vec2Int { x: 21 * 16, y: 384 - 6 * 16 }),
                QuestionBlock::create(Vec2Int { x: 22 * 16, y: 384 - 6 * 16 }),
                Brick::create(Vec2Int { x: 23 * 16, y: 384 - 6 * 16 }),
                QuestionBlock::create(Vec2Int { x: 24 * 16, y: 384 - 6 * 16 }),
                Brick::create(Vec2Int { x: 25 * 16, y: 384 - 6 * 16 }),
                QuestionBlock::create(Vec2Int { x: 23 * 16, y: 384 - 10 * 16 }),
                Pipe::create(Vec2Int {
                    x: 29 * 16,
                    y: 384 - 4 * 16,
                }, Vec2::up(), None),
                Pipe::create(Vec2Int {
                    x: 39 * 16,
                    y: 384 - 5 * 16,
                }, Vec2::up(), None),
                Pipe::create(Vec2Int {
                    x: 47 * 16,
                    y: 384 - 6 * 16,
                }, Vec2::up(), None),
                Pipe::create(Vec2Int {
                    x: 58 * 16,
                    y: 384 - 6 * 16,
                }, Vec2::up(), Some(MarioUndergroundScene.at_entrance(0))),
                Block::create(Vec2Int { x: 181 * 16, y: 384 - 3 * 16 }),
                Block::create(Vec2Int { x: 182 * 16, y: 384 - 3 * 16 }),
                Block::create(Vec2Int { x: 183 * 16, y: 384 - 3 * 16 }),
                Block::create(Vec2Int { x: 184 * 16, y: 384 - 3 * 16 }),
                Block::create(Vec2Int { x: 185 * 16, y: 384 - 3 * 16 }),
                Block::create(Vec2Int { x: 186 * 16, y: 384 - 3 * 16 }),
                Block::create(Vec2Int { x: 187 * 16, y: 384 - 3 * 16 }),
                Block::create(Vec2Int { x: 188 * 16, y: 384 - 3 * 16 }),
                Block::create(Vec2Int { x: 189 * 16, y: 384 - 3 * 16 }),

                Block::create(Vec2Int { x: 182 * 16, y: 384 - 4 * 16 }),
                Block::create(Vec2Int { x: 183 * 16, y: 384 - 4 * 16 }),
                Block::create(Vec2Int { x: 184 * 16, y: 384 - 4 * 16 }),
                Block::create(Vec2Int { x: 185 * 16, y: 384 - 4 * 16 }),
                Block::create(Vec2Int { x: 186 * 16, y: 384 - 4 * 16 }),
                Block::create(Vec2Int { x: 187 * 16, y: 384 - 4 * 16 }),
                Block::create(Vec2Int { x: 188 * 16, y: 384 - 4 * 16 }),
                Block::create(Vec2Int { x: 189 * 16, y: 384 - 4 * 16 }),

                Block::create(Vec2Int { x: 183 * 16, y: 384 - 5 * 16 }),
                Block::create(Vec2Int { x: 184 * 16, y: 384 - 5 * 16 }),
                Block::create(Vec2Int { x: 185 * 16, y: 384 - 5 * 16 }),
                Block::create(Vec2Int { x: 186 * 16, y: 384 - 5 * 16 }),
                Block::create(Vec2Int { x: 187 * 16, y: 384 - 5 * 16 }),
                Block::create(Vec2Int { x: 188 * 16, y: 384 - 5 * 16 }),
                Block::create(Vec2Int { x: 189 * 16, y: 384 - 5 * 16 }),

                Block::create(Vec2Int { x: 184 * 16, y: 384 - 6 * 16 }),
                Block::create(Vec2Int { x: 185 * 16, y: 384 - 6 * 16 }),
                Block::create(Vec2Int { x: 186 * 16, y: 384 - 6 * 16 }),
                Block::create(Vec2Int { x: 187 * 16, y: 384 - 6 * 16 }),
                Block::create(Vec2Int { x: 188 * 16, y: 384 - 6 * 16 }),
                Block::create(Vec2Int { x: 189 * 16, y: 384 - 6 * 16 }),

                Block::create(Vec2Int { x: 185 * 16, y: 384 - 7 * 16 }),
                Block::create(Vec2Int { x: 186 * 16, y: 384 - 7 * 16 }),
                Block::create(Vec2Int { x: 187 * 16, y: 384 - 7 * 16 }),
                Block::create(Vec2Int { x: 188 * 16, y: 384 - 7 * 16 }),
                Block::create(Vec2Int { x: 189 * 16, y: 384 - 7 * 16 }),

                Block::create(Vec2Int { x: 186 * 16, y: 384 - 8 * 16 }),
                Block::create(Vec2Int { x: 187 * 16, y: 384 - 8 * 16 }),
                Block::create(Vec2Int { x: 188 * 16, y: 384 - 8 * 16 }),
                Block::create(Vec2Int { x: 189 * 16, y: 384 - 8 * 16 }),

                Block::create(Vec2Int { x: 187 * 16, y: 384 - 9 * 16 }),
                Block::create(Vec2Int { x: 188 * 16, y: 384 - 9 * 16 }),
                Block::create(Vec2Int { x: 189 * 16, y: 384 - 9 * 16 }),

                Block::create(Vec2Int { x: 188 * 16, y: 384 - 10 * 16 }),
                Block::create(Vec2Int { x: 189 * 16, y: 384 - 10 * 16 }),

                Block::create(Vec2Int { x: 198 * 16, y: 384 - 3 * 16 }),
                Flagpole::create(Vec2Int { x: 198 * 16, y: 384 - 13 * 16 }),
                Brick::create(Vec2Int { x: 92 * 16, y: 384 - 10 * 16 }),
                Brick::create(Vec2Int { x: 93 * 16, y: 384 - 10 * 16 }),
                Brick::create(Vec2Int { x: 94 * 16, y: 384 - 10 * 16 }),
                QuestionBlock::create(Vec2Int { x: 95 * 16, y: 384 - 10 * 16 }),
                Brick::create(Vec2Int { x: 95 * 16, y: 384 - 6 * 16 }),
                Brick::create(Vec2Int { x: 101 * 16, y: 384 - 6 * 16 }),
                Brick::create(Vec2Int { x: 102 * 16, y: 384 - 6 * 16 }),
                QuestionBlock::create(Vec2Int { x: 107 * 16, y: 384 - 6 * 16 }),
                QuestionBlock::create(Vec2Int { x: 110 * 16, y: 384 - 6 * 16 }),
                QuestionBlock::create(Vec2Int { x: 110 * 16, y: 384 - 10 * 16 }),
                QuestionBlock::create(Vec2Int { x: 113 * 16, y: 384 - 6 * 16 }),
                Brick::create(Vec2Int { x: 119 * 16, y: 384 - 6 * 16 }),
                Brick::create(Vec2Int { x: 122 * 16, y: 384 - 10 * 16 }),
                Brick::create(Vec2Int { x: 123 * 16, y: 384 - 10 * 16 }),
                Brick::create(Vec2Int { x: 124 * 16, y: 384 - 10 * 16 }),
                Brick::create(Vec2Int { x: 128 * 16, y: 384 - 10 * 16 }),
                QuestionBlock::create(Vec2Int { x: 129 * 16, y: 384 - 10 * 16 }),
                QuestionBlock::create(Vec2Int { x: 130 * 16, y: 384 - 10 * 16 }),
                Brick::create(Vec2Int { x: 131 * 16, y: 384 - 10 * 16 }),
                Brick::create(Vec2Int { x: 129 * 16, y: 384 - 6 * 16 }),
                Brick::create(Vec2Int { x: 130 * 16, y: 384 - 6 * 16 }),

                Block::create(Vec2Int { x: 134 * 16, y: 384 - 3 * 16 }),
                Block::create(Vec2Int { x: 135 * 16, y: 384 - 3 * 16 }),
                Block::create(Vec2Int { x: 136 * 16, y: 384 - 3 * 16 }),
                Block::create(Vec2Int { x: 137 * 16, y: 384 - 3 * 16 }),
                Block::create(Vec2Int { x: 140 * 16, y: 384 - 3 * 16 }),
                Block::create(Vec2Int { x: 141 * 16, y: 384 - 3 * 16 }),
                Block::create(Vec2Int { x: 142 * 16, y: 384 - 3 * 16 }),
                Block::create(Vec2Int { x: 143 * 16, y: 384 - 3 * 16 }),

                Block::create(Vec2Int { x: 135 * 16, y: 384 - 4 * 16 }),
                Block::create(Vec2Int { x: 136 * 16, y: 384 - 4 * 16 }),
                Block::create(Vec2Int { x: 137 * 16, y: 384 - 4 * 16 }),
                Block::create(Vec2Int { x: 140 * 16, y: 384 - 4 * 16 }),
                Block::create(Vec2Int { x: 141 * 16, y: 384 - 4 * 16 }),
                Block::create(Vec2Int { x: 142 * 16, y: 384 - 4 * 16 }),

                Block::create(Vec2Int { x: 136 * 16, y: 384 - 5 * 16 }),
                Block::create(Vec2Int { x: 137 * 16, y: 384 - 5 * 16 }),
                Block::create(Vec2Int { x: 140 * 16, y: 384 - 5 * 16 }),
                Block::create(Vec2Int { x: 141 * 16, y: 384 - 5 * 16 }),

                Block::create(Vec2Int { x: 137 * 16, y: 384 - 6 * 16 }),
                Block::create(Vec2Int { x: 140 * 16, y: 384 - 6 * 16 }),

                Block::create(Vec2Int { x: 148 * 16, y: 384 - 3 * 16 }),
                Block::create(Vec2Int { x: 149 * 16, y: 384 - 3 * 16 }),
                Block::create(Vec2Int { x: 150 * 16, y: 384 - 3 * 16 }),
                Block::create(Vec2Int { x: 151 * 16, y: 384 - 3 * 16 }),
                Block::create(Vec2Int { x: 152 * 16, y: 384 - 3 * 16 }),

                Block::create(Vec2Int { x: 149 * 16, y: 384 - 4 * 16 }),
                Block::create(Vec2Int { x: 150 * 16, y: 384 - 4 * 16 }),
                Block::create(Vec2Int { x: 151 * 16, y: 384 - 4 * 16 }),
                Block::create(Vec2Int { x: 152 * 16, y: 384 - 4 * 16 }),

                Block::create(Vec2Int { x: 150 * 16, y: 384 - 5 * 16 }),
                Block::create(Vec2Int { x: 151 * 16, y: 384 - 5 * 16 }),
                Block::create(Vec2Int { x: 152 * 16, y: 384 - 5 * 16 }),

                Block::create(Vec2Int { x: 151 * 16, y: 384 - 6 * 16 }),
                Block::create(Vec2Int { x: 152 * 16, y: 384 - 6 * 16 }),

                Block::create(Vec2Int { x: 155 * 16, y: 384 - 3 * 16 }),
                Block::create(Vec2Int { x: 156 * 16, y: 384 - 3 * 16 }),
                Block::create(Vec2Int { x: 157 * 16, y: 384 - 3 * 16 }),
                Block::create(Vec2Int { x: 158 * 16, y: 384 - 3 * 16 }),

                Block::create(Vec2Int { x: 155 * 16, y: 384 - 4 * 16 }),
                Block::create(Vec2Int { x: 156 * 16, y: 384 - 4 * 16 }),
                Block::create(Vec2Int { x: 157 * 16, y: 384 - 4 * 16 }),

                Block::create(Vec2Int { x: 155 * 16, y: 384 - 5 * 16 }),
                Block::create(Vec2Int { x: 156 * 16, y: 384 - 5 * 16 }),

                Block::create(Vec2Int { x: 155 * 16, y: 384 - 6 * 16 }),
                Brick::create(Vec2Int { x: 168 * 16, y: 384 - 6 * 16 }),
                Brick::create(Vec2Int { x: 169 * 16, y: 384 - 6 * 16 }),
                QuestionBlock::create(Vec2Int { x: 170 * 16, y: 384 - 6 * 16 }),
                Brick::create(Vec2Int { x: 171 * 16, y: 384 - 6 * 16 }),
                Brick::create(Vec2Int { x: 78 * 16, y: 384 - 6 * 16 }),
                QuestionBlock::create(Vec2Int { x: 79 * 16, y: 384 - 6 * 16 }),
                Brick::create(Vec2Int { x: 80 * 16, y: 384 - 6 * 16 }),
                Pipe::create(Vec2Int {
                    x: 179 * 16,
                    y: 384 - 4 * 16,
                }, Vec2::up(), None),
                Pipe::create(Vec2Int {
                    x: 163 * 16,
                    y: 384 - 4 * 16,
                }, Vec2::up(), None),
            ].into_iter()
                .chain(Vec2Int::range_from_zero([1, 24])
                    .map(|(tile_x, tile_y)| {
                        Floor::create(Vec2Int {
                            x: tile_x * 16,
                            y: tile_y * 16,
                        })
                    }))
                .chain(Vec2Int::range_from_zero([69, 2])
                    .map(|(tile_x, tile_y)| {
                        Floor::create(Vec2Int {
                            x: (tile_x + 1) * 16,
                            y: 384 - (tile_y + 1) * 16
                        })
                    }))
                .chain(Vec2Int::range_from_zero([15, 2])
                    .map(|(tile_x, tile_y)| {
                        Floor::create(Vec2Int {
                            x: (tile_x + 72) * 16,
                            y: 384 - (tile_y + 1) * 16
                        })
                    }))
                .chain(Vec2Int::range_from_zero([80, 2])
                    .map(|(tile_x, tile_y)| {
                        Floor::create(Vec2Int {
                            x: (tile_x + 155) * 16,
                            y: 384 - (tile_y + 1) * 16
                        })
                    }))
                .chain(Vec2Int::range_from_zero([8, 1])
                    .map(|(tile_x, _tile_y)| {
                        Brick::create(Vec2Int {
                            x: (tile_x + 81) * 16,
                            y: 384 - 10 * 16
                        })
                    }))
                .chain(Vec2Int::range_from_zero([63, 2])
                    .map(|(tile_x, tile_y)| {
                        Floor::create(Vec2Int {
                            x: (tile_x + 90) * 16,
                            y: 384 - (tile_y + 1) * 16
                        })
                    }))
                .collect_vec()),
            Container::create("enemy", vec![
                Goomba::create(Vec2Int { x: 23 * 16, y: 384 - 3 * 16 }),
                Goomba::create(Vec2Int { x: 53 * 16 + 8, y: 384 - 3 * 16 }),
                Goomba::create(Vec2Int { x: 97 * 16, y: 384 - 3 * 16 }),
                Goomba::create(Vec2Int { x: 99 * 16 + 8, y: 384 - 3 * 16 }),
                Goomba::create(Vec2Int { x: 81 * 16, y: 384 - 11 * 16 }),
                Goomba::create(Vec2Int { x: 83 * 16, y: 384 - 11 * 16 }),
                Goomba::create(Vec2Int { x: 174 * 16, y: 384 - 3 * 16 }),
                Goomba::create(Vec2Int { x: 176 * 16, y: 384 - 3 * 16 }),
            ]),
        ]
    }
}

#[derive(Copy, Clone)]
pub struct MarioUndergroundScene;
impl Scene<ObjectType> for MarioUndergroundScene {
    fn name(&self) -> SceneName { SceneName::new("mario-underground") }

    fn create_objects(&self, _entrance_id: usize) -> Vec<AnySceneObject<ObjectType>> {
        vec![
            Canvas::create(),
            Player::create(Vec2Int {
                x: 2*16 + 8,
                y: 8,
            }, false),
            Container::create("level", vec![
                Pipe::create(Vec2Int {
                    x: 14 * 16,
                    y: 384 - 4*16,
                }, Vec2::left(), Some(MarioOverworldScene.at_entrance(1))),
                DecorativePipe::create(Vec2Int {
                    x: 16 * 16,
                    y: 0,
                }),
            ].into_iter()
            .chain(Vec2Int::range_from_zero([1, 24])
                .map(|(tile_x, tile_y)| {
                UndergroundFloor::create(Vec2Int {
                    x: tile_x * 16,
                    y: tile_y * 16,
                })
            }))
            .chain(Vec2Int::range_from_zero([17, 2])
                .map(|(tile_x, tile_y)| {
                    UndergroundFloor::create(Vec2Int {
                    x: (tile_x + 1) * 16,
                    y: 384 - (tile_y + 1) * 16
                    })
                }))
            .chain(Vec2Int::range_from_zero([7, 3])
                .map(|(tile_x, tile_y)| {
                    UndergroundBrick::create(Vec2Int {
                    x: (tile_x + 4) * 16,
                    y: 384 - (tile_y + 3) * 16
                    })
                }))
            .chain(Vec2Int::range_from_zero([7, 1])
                .map(|(tile_x, _tile_y)| {
                    UndergroundBrick::create(Vec2Int {
                    x: (tile_x + 4) * 16,
                    y: 0,
                    })
                }))
            .collect_vec()),
        ]
    }
}
