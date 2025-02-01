use glongge::core::{
    prelude::*,
    scene::{Scene, SceneName},
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub mod block;
pub mod enemy;
pub mod player;
pub mod text_display;
pub mod underground_floor;

use crate::object_type::ObjectType;
use block::brick::*;
use block::decorative_pipe::*;
use block::flagpole::*;
use block::pipe::*;
use block::plain_block::*;
use block::question_block::*;
use block::underground_brick::*;
use enemy::goomba::*;
use glongge::core::builtin::{Container, StaticSprite};
use glongge::core::render::VertexDepth;
use glongge::util::canvas::Canvas;
use glongge::util::tileset::TilesetBuilder;
use player::*;
use text_display::*;
use underground_floor::*;

const fn from_nes(pixels: u8, subpixels: u8, subsubpixels: u8, subsubsubpixels: u8) -> f32 {
    // fixed update at 100 fps
    (pixels as f32
        + subpixels as f32 / 16.
        + subsubpixels as f32 / 256.
        + subsubsubpixels as f32 / 4096.)
        * 60.
        / 100.
}
const fn from_nes_accel(pixels: u8, subpixels: u8, subsubpixels: u8, subsubsubpixels: u8) -> f32 {
    // fixed update at 100 fps
    from_nes(pixels, subpixels, subsubpixels, subsubsubpixels) * (60. / 100.)
}
const BASE_GRAVITY: f32 = from_nes_accel(0, 7, 0, 0);
const BLOCK_COLLISION_TAG: &str = "BLOCK";
const FLAG_COLLISION_TAG: &str = "FLAG";
const PIPE_COLLISION_TAG: &str = "PIPE";
const PLAYER_COLLISION_TAG: &str = "PLAYER";
const ENEMY_COLLISION_TAG: &str = "ENEMY";

#[derive(Default, Serialize, Deserialize)]
pub struct AliveEnemyMap {
    inner: BTreeMap<Vec2i, bool>,
}

impl AliveEnemyMap {
    fn register(&mut self, initial_coord: Vec2i) {
        self.inner.entry(initial_coord).or_insert(true);
    }

    fn is_alive(&self, initial_coord: Vec2i) -> bool {
        self.inner.get(&initial_coord).copied().unwrap_or(true)
    }
    fn set_dead(&mut self, initial_coord: Vec2i) {
        *self.inner.entry(initial_coord).or_default() = false;
    }
}

fn create_hill1(top_left: impl Into<Vec2>) -> AnySceneObject<ObjectType> {
    StaticSprite::new("res/world_sheet.png")
        .at_top_left(top_left)
        .with_single_coords(Vec2i { x: 112, y: 716 }, Vec2i { x: 192, y: 764 })
        .with_depth(VertexDepth::Back(0))
        .named("Hill1")
        .build()
}
fn create_hill2(top_left: impl Into<Vec2>) -> AnySceneObject<ObjectType> {
    StaticSprite::new("res/world_sheet.png")
        .at_top_left(top_left)
        .with_single_coords(Vec2i { x: 112, y: 692 }, Vec2i { x: 192, y: 708 })
        .with_depth(VertexDepth::Back(0))
        .named("Hill2")
        .build()
}
fn create_hill3(top_left: impl Into<Vec2>) -> AnySceneObject<ObjectType> {
    StaticSprite::new("res/world_sheet.png")
        .at_top_left(top_left)
        .with_single_coords(Vec2i { x: 200, y: 732 }, Vec2i { x: 248, y: 764 })
        .with_depth(VertexDepth::Back(0))
        .named("Hill3")
        .build()
}
fn create_hill4(top_left: impl Into<Vec2>) -> AnySceneObject<ObjectType> {
    StaticSprite::new("res/world_sheet.png")
        .at_top_left(top_left)
        .with_single_coords(Vec2i { x: 200, y: 692 }, Vec2i { x: 248, y: 708 })
        .with_depth(VertexDepth::Back(0))
        .named("Hill4")
        .build()
}
fn create_castle(top_left: impl Into<Vec2>) -> AnySceneObject<ObjectType> {
    StaticSprite::new("res/world_sheet.png")
        .at_top_left(top_left)
        .with_single_coords(Vec2i { x: 24, y: 684 }, Vec2i { x: 104, y: 764 })
        .with_depth(VertexDepth::Back(0))
        .named("Castle")
        .build()
}

#[derive(Copy, Clone)]
pub struct MarioOverworldScene;

impl Scene<ObjectType> for MarioOverworldScene {
    fn name(&self) -> SceneName {
        SceneName::new("mario-overworld")
    }

    fn initial_data(&self) -> Vec<u8> {
        bincode::serialize(&AliveEnemyMap::default()).unwrap()
    }

    fn create_objects(&self, entrance_id: usize) -> Vec<AnySceneObject<ObjectType>> {
        let mut ts = TilesetBuilder::new("res/world_sheet.png", 16).named("Doughnut");
        let block = ts.create_tile_collision([0, 33], &vec![BLOCK_COLLISION_TAG]);
        let crumble = ts.create_tile_collision([0, 50], &vec![BLOCK_COLLISION_TAG]);
        ts.insert(&block, [9, 19]);
        ts.insert(&block, [9, 18]);
        ts.insert(&block, [9, 17]);
        ts.insert(&block, [10, 19]);
        ts.insert(&crumble, [11, 19]);
        ts.insert(&crumble, [11, 18]);
        ts.insert(&crumble, [11, 17]);
        ts.insert(&crumble, [10, 17]);
        let ts = ts.build();

        let mut floor_ts = TilesetBuilder::new("res/world_sheet.png", 16).named("Floor");
        let floor = floor_ts.create_tile_collision([0, 16], &vec![BLOCK_COLLISION_TAG]);
        floor.set_depth(VertexDepth::Front(2000));
        Vec2i::range_from_zero([1, 25]).for_each(|(tile_x, tile_y)| {
            floor_ts.insert(&floor, [tile_x, tile_y]);
        });
        Vec2i::range_from_zero([69, 3]).for_each(|(tile_x, tile_y)| {
            floor_ts.insert(&floor, [tile_x + 1, 25 - (tile_y + 1)]);
        });
        Vec2i::range_from_zero([15, 3]).for_each(|(tile_x, tile_y)| {
            floor_ts.insert(&floor, [tile_x + 72, 25 - (tile_y + 1)]);
        });
        Vec2i::range_from_zero([63, 3]).for_each(|(tile_x, tile_y)| {
            floor_ts.insert(&floor, [tile_x + 90, 25 - (tile_y + 1)]);
        });
        Vec2i::range_from_zero([80, 3]).for_each(|(tile_x, tile_y)| {
            floor_ts.insert(&floor, [tile_x + 155, 25 - (tile_y + 1)]);
        });
        let floor_ts = floor_ts.build();
        vec![
            Canvas::create(),
            ts,
            floor_ts,
            match entrance_id {
                1 => Player::create(
                    Vec2i {
                        x: 164 * 16,
                        y: 384 - 2 * 16 - 8,
                    },
                    true,
                ),
                _ => Player::create(
                    Vec2i {
                        x: 8 * 16 + 8,
                        y: 384 - 3 * 16 + 8,
                    },
                    false,
                ),
            },
            Container::create(
                "background",
                vec![
                    create_hill1(Vec2i {
                        x: 16,
                        y: 384 - 2 * 16 - 48,
                    }),
                    create_hill2(Vec2i {
                        x: 12 * 16,
                        y: 384 - 2 * 16 - 16,
                    }),
                    create_hill3(Vec2i {
                        x: 17 * 16,
                        y: 384 - 2 * 16 - 32,
                    }),
                    create_hill4(Vec2i {
                        x: 24 * 16,
                        y: 384 - 2 * 16 - 16,
                    }),
                    create_hill1(Vec2i {
                        x: 49 * 16,
                        y: 384 - 2 * 16 - 48,
                    }),
                    create_hill2(Vec2i {
                        x: 60 * 16,
                        y: 384 - 2 * 16 - 16,
                    }),
                    create_hill1(Vec2i {
                        x: 64 * 16,
                        y: 384 - 2 * 16 - 32,
                    }),
                    create_castle(Vec2i {
                        x: 202 * 16,
                        y: 384 - 7 * 16,
                    }),
                ],
            ),
            Container::create(
                "level",
                vec![
                    QuestionBlock::create(Vec2i {
                        x: 17 * 16,
                        y: 384 - 6 * 16,
                    }),
                    Brick::create(Vec2i {
                        x: 21 * 16,
                        y: 384 - 6 * 16,
                    }),
                    QuestionBlock::create(Vec2i {
                        x: 22 * 16,
                        y: 384 - 6 * 16,
                    }),
                    Brick::create(Vec2i {
                        x: 23 * 16,
                        y: 384 - 6 * 16,
                    }),
                    QuestionBlock::create(Vec2i {
                        x: 24 * 16,
                        y: 384 - 6 * 16,
                    }),
                    Brick::create(Vec2i {
                        x: 25 * 16,
                        y: 384 - 6 * 16,
                    }),
                    QuestionBlock::create(Vec2i {
                        x: 23 * 16,
                        y: 384 - 10 * 16,
                    }),
                    Pipe::create(
                        Vec2i {
                            x: 29 * 16,
                            y: 384 - 4 * 16,
                        },
                        Vec2::up(),
                        None,
                    ),
                    Pipe::create(
                        Vec2i {
                            x: 39 * 16,
                            y: 384 - 5 * 16,
                        },
                        Vec2::up(),
                        None,
                    ),
                    Pipe::create(
                        Vec2i {
                            x: 47 * 16,
                            y: 384 - 6 * 16,
                        },
                        Vec2::up(),
                        None,
                    ),
                    Pipe::create(
                        Vec2i {
                            x: 58 * 16,
                            y: 384 - 6 * 16,
                        },
                        Vec2::up(),
                        Some(MarioUndergroundScene.at_entrance(0)),
                    ),
                    Block::create(Vec2i {
                        x: 181 * 16,
                        y: 384 - 3 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 182 * 16,
                        y: 384 - 3 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 183 * 16,
                        y: 384 - 3 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 184 * 16,
                        y: 384 - 3 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 185 * 16,
                        y: 384 - 3 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 186 * 16,
                        y: 384 - 3 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 187 * 16,
                        y: 384 - 3 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 188 * 16,
                        y: 384 - 3 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 189 * 16,
                        y: 384 - 3 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 182 * 16,
                        y: 384 - 4 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 183 * 16,
                        y: 384 - 4 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 184 * 16,
                        y: 384 - 4 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 185 * 16,
                        y: 384 - 4 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 186 * 16,
                        y: 384 - 4 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 187 * 16,
                        y: 384 - 4 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 188 * 16,
                        y: 384 - 4 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 189 * 16,
                        y: 384 - 4 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 183 * 16,
                        y: 384 - 5 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 184 * 16,
                        y: 384 - 5 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 185 * 16,
                        y: 384 - 5 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 186 * 16,
                        y: 384 - 5 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 187 * 16,
                        y: 384 - 5 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 188 * 16,
                        y: 384 - 5 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 189 * 16,
                        y: 384 - 5 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 184 * 16,
                        y: 384 - 6 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 185 * 16,
                        y: 384 - 6 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 186 * 16,
                        y: 384 - 6 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 187 * 16,
                        y: 384 - 6 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 188 * 16,
                        y: 384 - 6 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 189 * 16,
                        y: 384 - 6 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 185 * 16,
                        y: 384 - 7 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 186 * 16,
                        y: 384 - 7 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 187 * 16,
                        y: 384 - 7 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 188 * 16,
                        y: 384 - 7 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 189 * 16,
                        y: 384 - 7 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 186 * 16,
                        y: 384 - 8 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 187 * 16,
                        y: 384 - 8 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 188 * 16,
                        y: 384 - 8 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 189 * 16,
                        y: 384 - 8 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 187 * 16,
                        y: 384 - 9 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 188 * 16,
                        y: 384 - 9 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 189 * 16,
                        y: 384 - 9 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 188 * 16,
                        y: 384 - 10 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 189 * 16,
                        y: 384 - 10 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 198 * 16,
                        y: 384 - 3 * 16,
                    }),
                    Flagpole::create(Vec2i {
                        x: 198 * 16,
                        y: 384 - 13 * 16,
                    }),
                    Brick::create(Vec2i {
                        x: 92 * 16,
                        y: 384 - 10 * 16,
                    }),
                    Brick::create(Vec2i {
                        x: 93 * 16,
                        y: 384 - 10 * 16,
                    }),
                    Brick::create(Vec2i {
                        x: 94 * 16,
                        y: 384 - 10 * 16,
                    }),
                    QuestionBlock::create(Vec2i {
                        x: 95 * 16,
                        y: 384 - 10 * 16,
                    }),
                    Brick::create(Vec2i {
                        x: 95 * 16,
                        y: 384 - 6 * 16,
                    }),
                    Brick::create(Vec2i {
                        x: 101 * 16,
                        y: 384 - 6 * 16,
                    }),
                    Brick::create(Vec2i {
                        x: 102 * 16,
                        y: 384 - 6 * 16,
                    }),
                    QuestionBlock::create(Vec2i {
                        x: 107 * 16,
                        y: 384 - 6 * 16,
                    }),
                    QuestionBlock::create(Vec2i {
                        x: 110 * 16,
                        y: 384 - 6 * 16,
                    }),
                    QuestionBlock::create(Vec2i {
                        x: 110 * 16,
                        y: 384 - 10 * 16,
                    }),
                    QuestionBlock::create(Vec2i {
                        x: 113 * 16,
                        y: 384 - 6 * 16,
                    }),
                    Brick::create(Vec2i {
                        x: 119 * 16,
                        y: 384 - 6 * 16,
                    }),
                    Brick::create(Vec2i {
                        x: 122 * 16,
                        y: 384 - 10 * 16,
                    }),
                    Brick::create(Vec2i {
                        x: 123 * 16,
                        y: 384 - 10 * 16,
                    }),
                    Brick::create(Vec2i {
                        x: 124 * 16,
                        y: 384 - 10 * 16,
                    }),
                    Brick::create(Vec2i {
                        x: 128 * 16,
                        y: 384 - 10 * 16,
                    }),
                    QuestionBlock::create(Vec2i {
                        x: 129 * 16,
                        y: 384 - 10 * 16,
                    }),
                    QuestionBlock::create(Vec2i {
                        x: 130 * 16,
                        y: 384 - 10 * 16,
                    }),
                    Brick::create(Vec2i {
                        x: 131 * 16,
                        y: 384 - 10 * 16,
                    }),
                    Brick::create(Vec2i {
                        x: 129 * 16,
                        y: 384 - 6 * 16,
                    }),
                    Brick::create(Vec2i {
                        x: 130 * 16,
                        y: 384 - 6 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 134 * 16,
                        y: 384 - 3 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 135 * 16,
                        y: 384 - 3 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 136 * 16,
                        y: 384 - 3 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 137 * 16,
                        y: 384 - 3 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 140 * 16,
                        y: 384 - 3 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 141 * 16,
                        y: 384 - 3 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 142 * 16,
                        y: 384 - 3 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 143 * 16,
                        y: 384 - 3 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 135 * 16,
                        y: 384 - 4 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 136 * 16,
                        y: 384 - 4 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 137 * 16,
                        y: 384 - 4 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 140 * 16,
                        y: 384 - 4 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 141 * 16,
                        y: 384 - 4 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 142 * 16,
                        y: 384 - 4 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 136 * 16,
                        y: 384 - 5 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 137 * 16,
                        y: 384 - 5 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 140 * 16,
                        y: 384 - 5 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 141 * 16,
                        y: 384 - 5 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 137 * 16,
                        y: 384 - 6 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 140 * 16,
                        y: 384 - 6 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 148 * 16,
                        y: 384 - 3 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 149 * 16,
                        y: 384 - 3 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 150 * 16,
                        y: 384 - 3 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 151 * 16,
                        y: 384 - 3 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 152 * 16,
                        y: 384 - 3 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 149 * 16,
                        y: 384 - 4 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 150 * 16,
                        y: 384 - 4 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 151 * 16,
                        y: 384 - 4 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 152 * 16,
                        y: 384 - 4 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 150 * 16,
                        y: 384 - 5 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 151 * 16,
                        y: 384 - 5 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 152 * 16,
                        y: 384 - 5 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 151 * 16,
                        y: 384 - 6 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 152 * 16,
                        y: 384 - 6 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 155 * 16,
                        y: 384 - 3 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 156 * 16,
                        y: 384 - 3 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 157 * 16,
                        y: 384 - 3 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 158 * 16,
                        y: 384 - 3 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 155 * 16,
                        y: 384 - 4 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 156 * 16,
                        y: 384 - 4 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 157 * 16,
                        y: 384 - 4 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 155 * 16,
                        y: 384 - 5 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 156 * 16,
                        y: 384 - 5 * 16,
                    }),
                    Block::create(Vec2i {
                        x: 155 * 16,
                        y: 384 - 6 * 16,
                    }),
                    Brick::create(Vec2i {
                        x: 168 * 16,
                        y: 384 - 6 * 16,
                    }),
                    Brick::create(Vec2i {
                        x: 169 * 16,
                        y: 384 - 6 * 16,
                    }),
                    QuestionBlock::create(Vec2i {
                        x: 170 * 16,
                        y: 384 - 6 * 16,
                    }),
                    Brick::create(Vec2i {
                        x: 171 * 16,
                        y: 384 - 6 * 16,
                    }),
                    Brick::create(Vec2i {
                        x: 78 * 16,
                        y: 384 - 6 * 16,
                    }),
                    QuestionBlock::create(Vec2i {
                        x: 79 * 16,
                        y: 384 - 6 * 16,
                    }),
                    Brick::create(Vec2i {
                        x: 80 * 16,
                        y: 384 - 6 * 16,
                    }),
                    Pipe::create(
                        Vec2i {
                            x: 179 * 16,
                            y: 384 - 4 * 16,
                        },
                        Vec2::up(),
                        None,
                    ),
                    Pipe::create(
                        Vec2i {
                            x: 163 * 16,
                            y: 384 - 4 * 16,
                        },
                        Vec2::up(),
                        None,
                    ),
                ]
                .into_iter()
                .chain(Vec2i::range_from_zero([8, 1]).map(|(tile_x, _tile_y)| {
                    Brick::create(Vec2i {
                        x: (tile_x + 81) * 16,
                        y: 384 - 10 * 16,
                    })
                }))
                .collect_vec(),
            ),
            Container::create(
                "enemy",
                vec![
                    Goomba::create(Vec2i {
                        x: 23 * 16,
                        y: 384 - 3 * 16,
                    }),
                    Goomba::create(Vec2i {
                        x: 53 * 16 + 8,
                        y: 384 - 3 * 16,
                    }),
                    Goomba::create(Vec2i {
                        x: 97 * 16,
                        y: 384 - 3 * 16,
                    }),
                    Goomba::create(Vec2i {
                        x: 99 * 16 + 8,
                        y: 384 - 3 * 16,
                    }),
                    Goomba::create(Vec2i {
                        x: 81 * 16,
                        y: 384 - 11 * 16,
                    }),
                    Goomba::create(Vec2i {
                        x: 83 * 16,
                        y: 384 - 11 * 16,
                    }),
                    Goomba::create(Vec2i {
                        x: 174 * 16,
                        y: 384 - 3 * 16,
                    }),
                    Goomba::create(Vec2i {
                        x: 176 * 16,
                        y: 384 - 3 * 16,
                    }),
                ],
            ),
        ]
    }
}

#[derive(Copy, Clone)]
pub struct MarioUndergroundScene;
impl Scene<ObjectType> for MarioUndergroundScene {
    fn name(&self) -> SceneName {
        SceneName::new("mario-underground")
    }

    fn create_objects(&self, _entrance_id: usize) -> Vec<AnySceneObject<ObjectType>> {
        vec![
            Canvas::create(),
            Player::create(
                Vec2i {
                    x: 2 * 16 + 8,
                    y: 8,
                },
                false,
            ),
            Container::create(
                "level",
                vec![
                    Pipe::create(
                        Vec2i {
                            x: 14 * 16,
                            y: 384 - 4 * 16,
                        },
                        Vec2::left(),
                        Some(MarioOverworldScene.at_entrance(1)),
                    ),
                    DecorativePipe::create(Vec2i { x: 16 * 16, y: 0 }),
                ]
                .into_iter()
                .chain(Vec2i::range_from_zero([1, 25]).map(|(tile_x, tile_y)| {
                    UndergroundFloor::create(Vec2i {
                        x: tile_x * 16,
                        y: tile_y * 16,
                    })
                }))
                .chain(Vec2i::range_from_zero([17, 3]).map(|(tile_x, tile_y)| {
                    UndergroundFloor::create(Vec2i {
                        x: (tile_x + 1) * 16,
                        y: 400 - (tile_y + 1) * 16,
                    })
                }))
                .chain(Vec2i::range_from_zero([7, 3]).map(|(tile_x, tile_y)| {
                    UndergroundBrick::create(Vec2i {
                        x: (tile_x + 4) * 16,
                        y: 384 - (tile_y + 3) * 16,
                    })
                }))
                .chain(Vec2i::range_from_zero([7, 1]).map(|(tile_x, _tile_y)| {
                    UndergroundBrick::create(Vec2i {
                        x: (tile_x + 4) * 16,
                        y: 0,
                    })
                }))
                .collect_vec(),
            ),
        ]
    }
}
