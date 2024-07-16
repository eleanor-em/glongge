use std::{
    any::Any,
    time::Duration
};
use std::time::Instant;
use num_traits::Zero;
use glongge_derive::{partially_derive_scene_object, register_scene_object};
use crate::{
    resource::ResourceHandler,
    gg::{
        RenderableObject,
        RenderInfo,
        SceneObject,
        Transform,
        UpdateContext,
        VertexWithUV,
        scene::sample::mario::ObjectType
    },
    core::{
        linalg::{Vec2, Vec2Int},
        input::KeyCode,
        collision::{BoxCollider, Collider}
    },
    resource::sprite::Sprite,
};
use crate::core::util::gg_time;

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
enum PlayerState {
    Idle,
    Walking,
    Running,
    Skidding,
}

impl Default for PlayerState {
    fn default() -> Self { Self::Idle }
}

#[register_scene_object]
pub struct Player {
    pos: Vec2,
    dir: Vec2,
    speed: f64,
    accel: f64,
    run_sprite: Sprite,
    idle_sprite: Sprite,
    skid_sprite: Sprite,
    state: PlayerState,

    last_should_run: Instant,
    last_nonzero_dir: Vec2,
}

const fn from_nes(pixels: u8, subpixels: u8, subsubpixels: u8, subsubsubpixels: u8) -> f64 {
    // fixed update at 100 fps
    (pixels as f64
        + subpixels as f64 / 16.0
        + subsubpixels as f64 / 256.0
        + subsubsubpixels as f64 / 4096.0) * 100.0 / 60.0
}

impl Player {
    const MIN_WALK_SPEED: f64 = from_nes(0, 1, 3, 0);
    const MAX_WALK_SPEED: f64 = from_nes(1, 9, 0, 0);
    const MAX_RUN_SPEED: f64 = from_nes(2, 9, 0, 0);
    const WALK_ACCEL: f64 = from_nes(0, 0, 9, 8);
    const RUN_ACCEL: f64 = from_nes(0, 0, 15, 4);
    const RELEASE_DECEL: f64 = -from_nes(0, 0, 14, 0);
    const SKID_DECEL: f64 = -from_nes(0, 1, 10, 0);
    const SKID_TURNAROUND: f64 = from_nes(0, 9, 0, 0);

    fn current_sprite(&self) -> &Sprite {
        match self.state {
            PlayerState::Idle => &self.idle_sprite,
            PlayerState::Walking | PlayerState::Running => &self.run_sprite,
            PlayerState::Skidding => &self.skid_sprite,
        }
    }

    fn update_as_idle(&mut self, new_dir: Vec2, should_run: bool) {
        if !new_dir.is_zero() {
            self.dir = new_dir;
            self.speed = Self::MIN_WALK_SPEED;
            if should_run {
                self.state = PlayerState::Running;
                self.update_as_running(new_dir);
            } else {
                self.state = PlayerState::Walking;
                self.update_as_walking(new_dir, false);
            };
        }
    }

    fn update_as_walking(&mut self, new_dir: Vec2, should_run: bool) {
        self.speed = self.speed.min(Self::MAX_WALK_SPEED);
        if should_run {
            self.state = PlayerState::Running;
            self.update_as_running(new_dir);
        } else if self.dir == new_dir {
            self.accel = Self::WALK_ACCEL;
        } else {
            self.accel = Self::RELEASE_DECEL;
        }
    }

    fn update_as_running(&mut self, new_dir: Vec2) {
        self.speed = self.speed.min(Self::MAX_RUN_SPEED);
        if gg_time::as_frames(self.last_should_run.elapsed()) >= 10 {
            self.state = PlayerState::Walking;
            self.update_as_walking(new_dir, false);
        } else if self.dir == new_dir {
            self.accel = Self::RUN_ACCEL;
        } else if new_dir.is_zero() {
            self.accel = Self::RELEASE_DECEL;
        } else {
            self.state = PlayerState::Skidding;
            self.update_as_skidding(new_dir, true);
        }
    }

    fn update_as_skidding(&mut self, new_dir: Vec2, should_run: bool) {
        self.accel = Self::SKID_DECEL;
        if self.speed < Self::SKID_TURNAROUND && !new_dir.is_zero() {
            self.state = if should_run { PlayerState::Running } else { PlayerState::Walking };
            self.dir = new_dir;
        }
    }
}

#[partially_derive_scene_object]
impl SceneObject<ObjectType> for Player {
    fn on_load(&mut self, resource_handler: &mut ResourceHandler) -> anyhow::Result<()> {
        let texture_id = resource_handler.texture.wait_load_file("res/mario_sheet.png".to_string())?;
        self.idle_sprite = Sprite::from_single(
            texture_id,
            Vec2Int { x: 16, y: 16 },
            Vec2Int { x: 0, y: 8 },
        );
        self.run_sprite = Sprite::from_tileset(
            texture_id,
            Vec2Int { x: 3, y: 1 },
            Vec2Int { x: 16, y: 16 },
            Vec2Int { x: 20, y: 8 },
            Vec2Int { x: 2, y: 0 },
            100
        );
        self.skid_sprite = Sprite::from_single(
            texture_id,
            Vec2Int { x: 16, y: 16 },
            Vec2Int { x: 76, y: 8 },
        );
        Ok(())
    }
    fn on_ready(&mut self) {
        self.pos = Vec2 { x: 512.0, y: 384.0 };
        self.last_nonzero_dir = Vec2::right();
    }
    fn on_update(&mut self, _delta: Duration, update_ctx: UpdateContext<ObjectType>) {
        let new_dir = if update_ctx.input().down(KeyCode::Left) && !update_ctx.input().down(KeyCode::Right) {
            Vec2::left()
        } else if !update_ctx.input().down(KeyCode::Left) && update_ctx.input().down(KeyCode::Right) {
            Vec2::right()
        } else {
            Vec2::zero()
        };
        let should_run = update_ctx.input().down(KeyCode::X);
        if should_run { self.last_should_run = Instant::now(); }

        self.accel = 0.0;
        match self.state {
            PlayerState::Idle => self.update_as_idle(new_dir, should_run),
            PlayerState::Walking => self.update_as_walking(new_dir, should_run),
            PlayerState::Running => self.update_as_running(new_dir),
            PlayerState::Skidding => self.update_as_skidding(new_dir, should_run),
        }

        if !self.dir.is_zero() {
            self.last_nonzero_dir = self.dir;
        }
    }
    fn on_fixed_update(&mut self, _update_ctx: UpdateContext<ObjectType>) {
        self.speed += self.accel;
        self.pos += self.speed * self.dir;
        if self.speed < 0.0 {
            self.speed = 0.0;
            self.dir = Vec2::zero();
            self.state = PlayerState::Idle;
        }
    }

    fn transform(&self) -> Transform {
        Transform {
            position: self.pos,
            rotation: 0.0,
            scale: Vec2 {
                x: 2.0 * self.last_nonzero_dir.x,
                y: 2.0
            }
        }
    }
    fn as_renderable_object(&self) -> Option<&dyn RenderableObject<ObjectType>> {
        Some(self)
    }
    fn collider(&self) -> Option<Box<dyn Collider>> {
        Some(Box::new(BoxCollider::new(self.transform(), self.current_sprite().half_widths())))
    }
    fn collision_tags(&self) -> Vec<&'static str> {
        [].into()
    }
}

impl RenderableObject<ObjectType> for Player {
    fn create_vertices(&self) -> Vec<VertexWithUV> {
        self.current_sprite().create_vertices()
    }

    fn render_info(&self) -> RenderInfo {
        self.current_sprite().render_info_default()
    }
}
