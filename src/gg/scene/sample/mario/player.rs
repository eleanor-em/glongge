use std::{
    any::Any,
    time::{
        Duration,
    }
};
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
        collision::{BoxCollider, Collider},
        linalg::{Vec2, Vec2Int},
        input::KeyCode,
    },
    resource::sprite::Sprite,
};
use crate::core::linalg::SquareExtent;
use crate::gg::coroutine::{CoroutineId, CoroutineResponse};
use crate::gg::scene::sample::mario::{BRICK_COLLISION_TAG, PLAYER_COLLISION_TAG};

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
enum PlayerState {
    Idle,
    Walking,
    Running,
    Skidding,
    Falling,
}

impl Default for PlayerState {
    fn default() -> Self { Self::Idle }
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
enum SpeedRegime {
    Slow,
    Medium,
    Fast,
}

impl Default for SpeedRegime {
    fn default() -> Self { Self::Slow }
}

#[register_scene_object]
pub struct Player {
    centre: Vec2,
    dir: Vec2,
    speed: f64,
    accel: f64,
    v_speed: f64,
    v_accel: f64,

    speed_regime: SpeedRegime,
    state: PlayerState,
    last_ground_state: PlayerState,
    last_nonzero_dir: Vec2,
    cancel_run_crt: Option<CoroutineId>,
    coyote_crt: Option<CoroutineId>,

    run_sprite: Sprite,
    idle_sprite: Sprite,
    skid_sprite: Sprite,
    fall_sprite: Sprite,
}

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

impl Player {
    const MIN_WALK_SPEED: f64 = from_nes(0, 1, 3, 0);
    const MAX_WALK_SPEED: f64 = from_nes(1, 9, 0, 0);
    const MAX_RUN_SPEED: f64 = from_nes(2, 9, 0, 0);
    const WALK_ACCEL: f64 = from_nes_accel(0, 0, 9, 8);
    const RUN_ACCEL: f64 = from_nes_accel(0, 0, 15, 4);
    const RELEASE_DECEL: f64 = -from_nes_accel(0, 0, 14, 0);
    const SKID_DECEL: f64 = -from_nes_accel(0, 1, 10, 0);
    const SKID_TURNAROUND: f64 = from_nes(0, 9, 0, 0);
    const MAX_VSPEED: f64 = from_nes(4, 8, 0, 0);

    fn initial_vspeed(&self) -> f64 {
        match self.speed_regime {
            SpeedRegime::Slow => -from_nes(4, 0, 0, 0),
            SpeedRegime::Medium => -from_nes(4, 0, 0, 0),
            SpeedRegime::Fast => -from_nes(5, 0, 0, 0),
        }
    }
    fn gravity(&self) -> f64 {
        match self.speed_regime {
            SpeedRegime::Slow => from_nes_accel(0, 7, 0, 0),
            SpeedRegime::Medium => from_nes_accel(0, 6, 0, 0),
            SpeedRegime::Fast => from_nes_accel(0, 9, 0, 0)
        }
    }
    fn hold_gravity(&self) -> f64 {
        match self.speed_regime {
            SpeedRegime::Slow => from_nes_accel(0, 2, 0, 0),
            SpeedRegime::Medium => from_nes_accel(0, 1, 14, 0),
            SpeedRegime::Fast => from_nes_accel(0, 2, 8, 0)
        }
    }

    fn current_sprite(&self) -> &Sprite {
        match self.state {
            PlayerState::Idle => &self.idle_sprite,
            PlayerState::Walking | PlayerState::Running => &self.run_sprite,
            PlayerState::Skidding => &self.skid_sprite,
            PlayerState::Falling => &self.fall_sprite,
        }
    }

    fn update_as_idle(&mut self, update_ctx: &mut UpdateContext<ObjectType>, new_dir: Vec2, hold_run: bool) {
        if !new_dir.is_zero() {
            self.dir = new_dir;
            self.speed = Self::MIN_WALK_SPEED;
            if hold_run {
                self.state = PlayerState::Running;
                self.update_as_running(update_ctx, new_dir, hold_run);
            } else {
                self.state = PlayerState::Walking;
                self.update_as_walking(update_ctx, new_dir, hold_run);
            };
        }
    }

    fn update_as_walking(&mut self, update_ctx: &mut UpdateContext<ObjectType>, new_dir: Vec2, hold_run: bool) {
        self.speed = self.speed.min(Self::MAX_WALK_SPEED);
        if hold_run {
            self.state = PlayerState::Running;
            self.update_as_running(update_ctx, new_dir, hold_run);
        } else if self.dir == new_dir {
            self.accel = Self::WALK_ACCEL;
        } else {
            self.accel = Self::RELEASE_DECEL;
        }
    }

    fn update_as_running(&mut self, update_ctx: &mut UpdateContext<ObjectType>, new_dir: Vec2, hold_run: bool) {
        if hold_run {
            self.cancel_run_crt.take()
                .map(|id| update_ctx.cancel_coroutine(id));
        } else {
            self.cancel_run_crt.get_or_insert_with(|| {
                update_ctx.add_coroutine_after(|mut this, _action| {
                    let mut this = this.downcast_mut::<Self>().unwrap();
                    if this.state == PlayerState::Running {
                        this.state = PlayerState::Walking;
                    }
                    CoroutineResponse::Complete
                }, Duration::from_secs_f64(1./6.))
            });
        }
        self.speed = self.speed.min(Self::MAX_RUN_SPEED);
        if self.dir == new_dir {
            self.accel = Self::RUN_ACCEL;
        } else if new_dir.is_zero() {
            self.accel = Self::RELEASE_DECEL;
        } else {
            self.state = PlayerState::Skidding;
            self.update_as_skidding(new_dir, true);
        }
    }

    fn update_as_skidding(&mut self, new_dir: Vec2, hold_run: bool) {
        self.accel = Self::SKID_DECEL;
        if self.speed < Self::SKID_TURNAROUND && !new_dir.is_zero() {
            self.state = if hold_run { PlayerState::Running } else { PlayerState::Walking };
            self.dir = new_dir;
        }
    }

    fn update_as_falling(&mut self, new_dir: Vec2, hold_jump: bool) {
        if self.dir.is_zero() {
            self.dir = new_dir;
        }
        self.speed = self.speed.min(match self.last_ground_state {
            PlayerState::Running => Self::MAX_RUN_SPEED,
            _ => Self::MAX_WALK_SPEED,
        });
        if new_dir == self.dir {
            self.speed = Self::MIN_WALK_SPEED.max(self.speed);
            if self.speed < from_nes(1, 9, 0, 0) {
                self.accel = from_nes(0, 0, 9, 8);
            } else {
                self.accel = from_nes(0, 0, 14, 4);
            }
        } else if self.speed < from_nes(1, 9, 0, 0) {
            self.accel = -from_nes(0, 0, 14, 4);
        } else {
            self.accel = -from_nes(0, 0, 13, 0);
        }
        if hold_jump && self.v_speed < 0. {
            self.v_accel = self.hold_gravity();
        } else {
            self.v_accel = self.gravity();
        }
    }

    fn maybe_start_falling(&mut self, update_ctx: &mut UpdateContext<ObjectType>) {
        if self.state != PlayerState::Falling {
            self.speed_regime = match self.speed {
                x if (0.0
                    ..from_nes(1, 0, 0, 0)).contains(&x) => SpeedRegime::Slow,
                x if (from_nes(1, 0, 0, 0)
                    ..from_nes(2, 5, 0, 0)).contains(&x) => SpeedRegime::Medium,
                _ => SpeedRegime::Fast,
            };
            if update_ctx.input().pressed(KeyCode::Z) {
                self.state = PlayerState::Falling;
                self.v_speed = self.initial_vspeed();
            }
        }

        let ray = self.collider().translated(Vec2::down());
        if update_ctx.test_collision(ray.as_ref(), vec![BRICK_COLLISION_TAG]).is_none() {
            self.coyote_crt.get_or_insert_with(|| {
                update_ctx.add_coroutine_after(|mut this, _action| {
                    let mut this = this.downcast_mut::<Self>().unwrap();
                    this.state = PlayerState::Falling;
                    CoroutineResponse::Complete
                }, Duration::from_secs_f64(0.1))
            });
        } else {
            self.coyote_crt.take().map(|id| update_ctx.cancel_coroutine(id));
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
        self.fall_sprite = Sprite::from_single(
            texture_id,
            Vec2Int { x: 16, y: 16 },
            Vec2Int { x: 96, y: 8 },
        );
        Ok(())
    }
    fn on_ready(&mut self) {
        self.centre = Vec2 { x: 200., y: 300. };
        self.last_nonzero_dir = Vec2::right();
    }

    fn on_update_begin(&mut self, _delta: Duration, _update_ctx: UpdateContext<ObjectType>) {
        self.accel = 0.;
        self.v_accel = 0.;
    }
    fn on_update(&mut self, _delta: Duration, mut update_ctx: UpdateContext<ObjectType>) {
        let new_dir = if update_ctx.input().down(KeyCode::Left) && !update_ctx.input().down(KeyCode::Right) {
            Vec2::left()
        } else if !update_ctx.input().down(KeyCode::Left) && update_ctx.input().down(KeyCode::Right) {
            Vec2::right()
        } else {
            Vec2::zero()
        };
        let hold_run = update_ctx.input().down(KeyCode::X);
        let hold_jump = update_ctx.input().down(KeyCode::Z);

        if self.state != PlayerState::Falling {
            self.last_ground_state = self.state;
        }
        self.maybe_start_falling(&mut update_ctx);
        match self.state {
            PlayerState::Idle => self.update_as_idle(&mut update_ctx, new_dir, hold_run),
            PlayerState::Walking => self.update_as_walking(&mut update_ctx, new_dir, hold_run),
            PlayerState::Running => self.update_as_running(&mut update_ctx, new_dir, hold_run),
            PlayerState::Skidding => self.update_as_skidding(new_dir, hold_run),
            PlayerState::Falling => self.update_as_falling(new_dir, hold_jump),
        }

        if !self.dir.is_zero() {
            self.last_nonzero_dir = self.dir;
        }
    }
    fn on_fixed_update(&mut self, update_ctx: UpdateContext<ObjectType>) {
        self.speed += self.accel;
        self.v_speed += self.v_accel;
        self.v_speed = Self::MAX_VSPEED.min(self.v_speed);

        let v_ray = self.collider().translated(self.v_speed * Vec2::down());
        match update_ctx.test_collision_along(v_ray.as_ref(), vec![BRICK_COLLISION_TAG], Vec2::down()) {
            Some(collisions) => {
                self.centre += self.v_speed * Vec2::down() + collisions.first().mtv;
                self.v_speed = 0.;
                self.state = self.last_ground_state;
            }
            None => self.centre += self.v_speed * Vec2::down(),
        }

        let h_ray = self.collider().translated(self.speed * self.dir);
        match update_ctx.test_collision_along(h_ray.as_ref(), vec![BRICK_COLLISION_TAG], Vec2::right()) {
            Some(collisions) => {
                self.centre += self.speed * self.dir + collisions.first().mtv;
                self.speed *= 0.9;
            }
            None => self.centre += self.speed * self.dir,
        }

        if self.speed < 0. {
            if self.state == PlayerState::Falling {
                self.speed = -self.speed;
                self.dir = -self.dir;
            } else {
                self.speed = 0.;
                self.dir = Vec2::zero();
                self.state = PlayerState::Idle;
            }
        }
    }

    fn transform(&self) -> Transform {
        Transform {
            centre: self.centre,
            rotation: 0.,
            scale: Vec2 {
                x: self.last_nonzero_dir.x,
                y: 1.,
            }
        }
    }
    fn as_renderable_object(&self) -> Option<&dyn RenderableObject<ObjectType>> {
        Some(self)
    }
    fn collider(&self) -> Box<dyn Collider> {
        Box::new(BoxCollider::from_centre(self.centre, self.current_sprite().half_widths()))
    }
    fn collision_tags(&self) -> Vec<&'static str> {
        [PLAYER_COLLISION_TAG].into()
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
