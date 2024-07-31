use std::time::Duration;
use num_traits::Zero;
use glongge_derive::{partially_derive_scene_object, register_scene_object};
use glongge::{
    core::{
        coroutine::prelude::*,
        prelude::*,
        render::VertexDepth,
        scene::{Scene, SceneDestination},
    },
    resource::{
        sound::Sound,
        sprite::Sprite
    },
};
use crate::examples::mario::{
    AliveEnemyMap,
    BASE_GRAVITY,
    block::downcast_bumpable_mut,
    block::pipe::Pipe,
    BLOCK_COLLISION_TAG,
    enemy::downcast_stompable_mut,
    ENEMY_COLLISION_TAG,
    FLAG_COLLISION_TAG,
    from_nes,
    from_nes_accel,
    MarioOverworldScene,
    PIPE_COLLISION_TAG,
    PLAYER_COLLISION_TAG,
    WinTextDisplay
};
use crate::object_type::ObjectType;

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
enum PlayerState {
    Idle,
    Walking,
    Running,
    Skidding,
    Falling,
    Dying,
    EnteringPipe,
    ExitingPipe,

    RidingFlagpole,
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

    hold_jump: bool,
    show_wireframes: bool,

    speed_regime: SpeedRegime,
    state: PlayerState,
    last_state: PlayerState,
    last_ground_state: PlayerState,
    last_nonzero_dir: Vec2,
    cancel_run_crt: Option<CoroutineId>,
    coyote_crt: Option<CoroutineId>,
    exit_pipe_crt: Option<CoroutineId>,

    walk_sprite: Sprite,
    run_sprite: Sprite,
    idle_sprite: Sprite,
    skid_sprite: Sprite,
    fall_sprite: Sprite,
    die_sprite: Sprite,
    flagpole_sprite: Sprite,

    jump_sound: Sound,
    stomp_sound: Sound,
    die_sound: Sound,
    pipe_sound: Sound,
    bump_sound: Sound,
    flagpole_sound: Sound,
    clear_sound: Sound,

    overworld_music: Sound,
    underground_music: Sound,
    music: Sound,
}

// For a guide to Super Mario Bros. (NES) physics, see:
// https://web.archive.org/web/20130807122227/http://i276.photobucket.com/albums/kk21/jdaster64/smb_playerphysics.png
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

    pub fn new(centre: Vec2Int, exiting_pipe: bool) -> AnySceneObject<ObjectType> {
        AnySceneObject::new(Self {
            centre: centre.into(),
            // Prevents player getting "stuck" on ground when level starts in air.
            last_ground_state: PlayerState::Walking,
            state: if exiting_pipe { PlayerState::ExitingPipe } else { PlayerState::Idle },
            show_wireframes: false,
            ..Default::default()
        })
    }

    fn initial_vspeed(&self) -> f64 {
        match self.speed_regime {
            SpeedRegime::Slow => -from_nes(4, 1, 0, 0),
            SpeedRegime::Medium => -from_nes(4, 1, 0, 0),
            SpeedRegime::Fast => -from_nes(5, 0, 0, 0),
        }
    }
    fn gravity(&self) -> f64 {
        match self.speed_regime {
            SpeedRegime::Slow => BASE_GRAVITY,
            SpeedRegime::Medium => from_nes_accel(0, 6, 0, 0),
            SpeedRegime::Fast => from_nes_accel(0, 9, 0, 0)
        }
    }
    fn current_sprite(&self) -> &Sprite {
        match self.state {
            PlayerState::Idle => &self.idle_sprite,
            PlayerState::Walking => &self.walk_sprite,
            PlayerState::Running => &self.run_sprite,
            PlayerState::Skidding => &self.skid_sprite,
            PlayerState::Falling => &self.fall_sprite,
            PlayerState::Dying => &self.die_sprite,
            PlayerState::EnteringPipe => &self.idle_sprite,
            PlayerState::ExitingPipe => &self.idle_sprite,
            PlayerState::RidingFlagpole => &self.flagpole_sprite,
        }
    }
    fn current_sprite_mut(&mut self) -> &mut Sprite {
        match self.state {
            PlayerState::Idle => &mut self.idle_sprite,
            PlayerState::Walking => &mut self.walk_sprite,
            PlayerState::Running => &mut self.run_sprite,
            PlayerState::Skidding => &mut self.skid_sprite,
            PlayerState::Falling => &mut self.fall_sprite,
            PlayerState::Dying => &mut self.die_sprite,
            PlayerState::EnteringPipe => &mut self.idle_sprite,
            PlayerState::ExitingPipe => &mut self.idle_sprite,
            PlayerState::RidingFlagpole => &mut self.flagpole_sprite,
        }
    }

    fn hold_gravity(&self) -> f64 {
        match self.speed_regime {
            SpeedRegime::Slow => from_nes_accel(0, 2, 0, 0),
            SpeedRegime::Medium => from_nes_accel(0, 1, 14, 0),
            SpeedRegime::Fast => from_nes_accel(0, 2, 8, 0)
        }
    }

    fn has_control(&self) -> bool {
        self.state != PlayerState::Dying &&
            self.state != PlayerState::EnteringPipe &&
            self.state != PlayerState::ExitingPipe &&
            self.state != PlayerState::RidingFlagpole
    }

    fn update_as_idle(&mut self, ctx: &mut UpdateContext<ObjectType>, new_dir: Vec2, hold_run: bool) {
        if !new_dir.is_zero() {
            self.dir = new_dir;
            self.speed = Self::MIN_WALK_SPEED;
            if hold_run {
                self.state = PlayerState::Running;
                self.update_as_running(ctx, new_dir, hold_run);
            } else {
                self.state = PlayerState::Walking;
                self.update_as_walking(ctx, new_dir, hold_run);
            };
        }
    }

    fn update_as_walking(&mut self, ctx: &mut UpdateContext<ObjectType>, new_dir: Vec2, hold_run: bool) {
        self.speed = self.speed.min(Self::MAX_WALK_SPEED);
        if hold_run {
            self.state = PlayerState::Running;
            self.update_as_running(ctx, new_dir, hold_run);
        } else if self.dir == new_dir {
            self.accel = Self::WALK_ACCEL;
        } else {
            self.accel = Self::RELEASE_DECEL;
        }
    }

    fn update_as_running(&mut self, ctx: &mut UpdateContext<ObjectType>, new_dir: Vec2, hold_run: bool) {
        if hold_run {
            ctx.scene().maybe_cancel_coroutine(&mut self.cancel_run_crt);
        } else {
            self.cancel_run_crt.get_or_insert_with(|| {
                ctx.scene().start_coroutine_after(|this, _ctx, _last_state| {
                    let mut this = this.downcast_mut::<Self>().unwrap();
                    if !this.has_control() {
                        return CoroutineResponse::Complete;
                    }
                    if this.state == PlayerState::Running {
                        this.state = PlayerState::Walking;
                    }
                    this.cancel_run_crt = None;
                    CoroutineResponse::Complete
                }, Duration::from_millis(80))
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

    fn update_as_falling(&mut self, new_dir: Vec2) {
        self.speed = self.speed.min(match self.last_ground_state {
            PlayerState::Running => Self::MAX_RUN_SPEED,
            _ => Self::MAX_WALK_SPEED,
        });
        if !new_dir.is_zero() {
            if self.dir.is_zero() {
                self.dir = new_dir;
            }
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
        }
        if self.hold_jump && self.v_speed < 0. {
            self.v_accel = self.hold_gravity();
        } else {
            self.v_accel = self.gravity();
        }
    }

    fn maybe_start_jump(&mut self, ctx: &mut UpdateContext<ObjectType>) {
        if self.state != PlayerState::Falling {
            self.speed_regime = match self.speed {
                x if (0.0
                    ..from_nes(1, 0, 0, 0)).contains(&x) => SpeedRegime::Slow,
                x if (from_nes(1, 0, 0, 0)
                    ..from_nes(2, 5, 0, 0)).contains(&x) => SpeedRegime::Medium,
                _ => SpeedRegime::Fast,
            };
            if ctx.input().pressed(KeyCode::Z) {
                self.start_jump();
                return;
            }
        }

        if ctx.object().test_collision_along(Vec2::down(), 1., vec![BLOCK_COLLISION_TAG]).is_none() {
            self.coyote_crt.get_or_insert_with(|| {
                ctx.scene().start_coroutine_after(|this, _ctx, _last_state| {
                    let mut this = this.downcast_mut::<Self>().unwrap();
                    if !this.has_control() {
                        return CoroutineResponse::Complete;
                    }
                    this.state = PlayerState::Falling;
                    this.coyote_crt = None;
                    CoroutineResponse::Complete
                }, Duration::from_millis(60))
            });
        } else {
            ctx.scene().maybe_cancel_coroutine(&mut self.coyote_crt);
        }

        if self.coyote_crt.is_none() && self.state != PlayerState::Falling {
            self.last_ground_state = self.state;
        }
    }

    fn maybe_start_exit_pipe(&mut self, ctx: &mut UpdateContext<ObjectType>) {
        if self.state == PlayerState::ExitingPipe && self.exit_pipe_crt.is_none() {
            self.exit_pipe_crt.replace(ctx.scene().start_coroutine(|this, ctx, last_state| {
                let mut this = this.downcast_mut::<Self>().unwrap();
                match last_state {
                    CoroutineState::Starting => return CoroutineResponse::Wait(Duration::from_millis(500)),
                    CoroutineState::Waiting => { this.pipe_sound.play(); }
                    _ => {}
                }
                // TODO: replace this with some sort of cached collision feature (from end of last update).
                if ctx.object().test_collision(vec![PIPE_COLLISION_TAG]).is_none() {
                    this.state = PlayerState::Idle;
                    this.v_speed = 0.;
                    // Snap to top of pipe.
                    this.centre.y = (this.centre.y / 8.).round() * 8.;
                    CoroutineResponse::Complete
                } else {
                    this.v_speed = -Self::MAX_VSPEED / 3.;
                    CoroutineResponse::Yield
                }
            }));
        }
    }
    fn maybe_start_pipe(&mut self, ctx: &mut UpdateContext<ObjectType>) {
        if ctx.input().down(KeyCode::Down) {
            if let Some(collisions) = ctx.object()
                .test_collision_along(Vec2::down(), 1., vec![PIPE_COLLISION_TAG]) {
                let pipe = collisions.first().other.checked_downcast::<Pipe>();
                if !pipe.orientation().dot(Vec2::down()).is_zero() {
                    if let Some(instruction) = pipe.destination() {
                        self.start_pipe(ctx, Vec2::down(), pipe.transform().centre, instruction);
                    }
                }
            }
        } else if ctx.input().down(KeyCode::Right) {
            if let Some(collisions) = ctx.object()
                .test_collision_along(Vec2::right(), 1., vec![PIPE_COLLISION_TAG]) {
                let pipe = collisions.first().other.checked_downcast::<Pipe>();
                if let Some(instruction) = pipe.destination() {
                    if !pipe.orientation().dot(Vec2::right()).is_zero() &&
                            !collisions.first().mtv.dot(Vec2::right()).is_zero() {
                        self.start_pipe(ctx, Vec2::right(), pipe.transform().centre, instruction);
                    }
                }
            }
        }
    }
    fn start_pipe(&mut self,
                  ctx: &mut UpdateContext<ObjectType>,
                  direction: Vec2,
                  pipe_centre: Vec2,
                  pipe_instruction: SceneDestination) {
        self.music.stop();
        self.pipe_sound.play();
        self.state = PlayerState::EnteringPipe;
        ctx.scene().start_coroutine(move |this, ctx, last_state| {
            let mut this = this.downcast_mut::<Self>().unwrap();
            if direction.x.is_zero() {
                // Vertical travel through pipe.
                if (pipe_centre.x - this.centre.x).abs() > Self::MAX_WALK_SPEED * 1.1 {
                    this.speed = (pipe_centre.x - this.centre.x).signum() * Self::MAX_WALK_SPEED;
                    return CoroutineResponse::Yield
                }

                this.centre.x = pipe_centre.x;
                this.speed = 0.;
                this.v_speed = if pipe_centre.y > this.centre.y {
                    Self::MAX_VSPEED / 2.
                } else {
                    -Self::MAX_VSPEED / 2.
                };
            } else {
                this.speed = Self::MAX_WALK_SPEED;
                this.v_speed = 0.;
            }

            match last_state {
                CoroutineState::Starting | CoroutineState::Yielding => {
                    CoroutineResponse::Wait(Duration::from_millis(600))
                }
                CoroutineState::Waiting => {
                    ctx.scene().goto(pipe_instruction);
                    CoroutineResponse::Complete
                }
            }
        });
    }

    fn start_jump(&mut self) {
        self.jump_sound.play_shifted(0.03);
        self.state = PlayerState::Falling;
        self.v_speed = self.initial_vspeed();
    }

    fn start_die(&mut self, ctx: &mut UpdateContext<ObjectType>) {
        self.music.stop();
        self.die_sprite.set_depth(VertexDepth::Front(10000));
        ctx.scene().start_coroutine(|this, ctx, last_state| {
            let mut this = this.downcast_mut::<Self>().unwrap();
            match last_state {
                CoroutineState::Starting => {
                    this.die_sound.play();
                    CoroutineResponse::Yield
                },
                CoroutineState::Yielding => {
                    if this.die_sound.is_playing() {
                        CoroutineResponse::Yield
                    } else {
                        CoroutineResponse::Wait(Duration::from_secs(1))
                    }
                }
                CoroutineState::Waiting => {
                    if ctx.scene().name() == MarioOverworldScene.name() {
                        ctx.scene().data::<AliveEnemyMap>().unwrap().reset();
                    }
                    ctx.scene().goto(MarioOverworldScene.at_entrance(0));
                    CoroutineResponse::Complete
                }
            }
        });
        self.v_speed = -from_nes(13, 0, 0, 0);
        self.state = PlayerState::Dying;
    }
}

#[partially_derive_scene_object]
impl SceneObject<ObjectType> for Player {
    fn on_preload(&mut self, resource_handler: &mut ResourceHandler) -> Result<()> {
        resource_handler.sound.spawn_load_file("res/jump-small.wav".to_string());
        resource_handler.sound.spawn_load_file("res/stomp.wav".to_string());
        resource_handler.sound.spawn_load_file("res/death.wav".to_string());
        resource_handler.sound.spawn_load_file("res/pipe.wav".to_string());
        resource_handler.sound.spawn_load_file("res/bump.wav".to_string());
        resource_handler.sound.spawn_load_file("res/flagpole.wav".to_string());
        resource_handler.sound.spawn_load_file("res/stage-clear.wav".to_string());
        resource_handler.sound.spawn_load_file("res/overworld.ogg".to_string());
        resource_handler.sound.spawn_load_file("res/underground.ogg".to_string());
        resource_handler.texture.wait_load_file("res/mario_sheet.png".to_string())?;
        resource_handler.texture.wait_load_file("res/world_sheet.png".to_string())?;
        resource_handler.texture.wait_load_file("res/enemies_sheet.png".to_string())?;
        Ok(())
    }
    fn on_load(&mut self, object_ctx: &mut ObjectContext<ObjectType>, resource_handler: &mut ResourceHandler) -> Result<Option<RenderItem>> {
        let texture = resource_handler.texture.wait_load_file("res/mario_sheet.png".to_string())?;
        self.idle_sprite = Sprite::from_single_extent(
            object_ctx,
            texture.clone(),
            Vec2Int { x: 16, y: 16 },
            Vec2Int { x: 0, y: 8 },
        );
        self.walk_sprite = Sprite::from_tileset(
            object_ctx,
            texture.clone(),
            Vec2Int { x: 3, y: 1 },
            Vec2Int { x: 16, y: 16 },
            Vec2Int { x: 20, y: 8 },
            Vec2Int { x: 2, y: 0 }
        )
            .with_fixed_ms_per_frame(110)
            .with_hidden();
        self.run_sprite = Sprite::from_tileset(
            object_ctx,
            texture.clone(),
            Vec2Int { x: 3, y: 1 },
            Vec2Int { x: 16, y: 16 },
            Vec2Int { x: 20, y: 8 },
            Vec2Int { x: 2, y: 0 }
        )
            .with_fixed_ms_per_frame(60)
            .with_hidden();
        self.skid_sprite = Sprite::from_single_extent(
            object_ctx,
            texture.clone(),
            Vec2Int { x: 16, y: 16 },
            Vec2Int { x: 76, y: 8 },
        )
            .with_hidden();
        self.fall_sprite = Sprite::from_single_extent(
            object_ctx,
            texture.clone(),
            Vec2Int { x: 16, y: 16 },
            Vec2Int { x: 96, y: 8 },
        )
            .with_hidden();
        self.die_sprite = Sprite::from_single_extent(
            object_ctx,
            texture.clone(),
            Vec2Int { x: 16, y: 16 },
            Vec2Int { x: 116, y: 8 },
        )
            .with_hidden();
        self.flagpole_sprite = Sprite::from_single_extent(
            object_ctx,
            texture.clone(),
            Vec2Int { x: 16, y: 16 },
            Vec2Int { x: 136, y: 8 },
        )
            .with_hidden();

        self.jump_sound = resource_handler.sound.wait_load_file("res/jump-small.wav".to_string())?;
        self.stomp_sound = resource_handler.sound.wait_load_file("res/stomp.wav".to_string())?;
        self.die_sound = resource_handler.sound.wait_load_file("res/death.wav".to_string())?;
        self.pipe_sound = resource_handler.sound.wait_load_file("res/pipe.wav".to_string())?;
        self.bump_sound = resource_handler.sound.wait_load_file("res/bump.wav".to_string())?;
        self.flagpole_sound = resource_handler.sound.wait_load_file("res/flagpole.wav".to_string())?;
        self.clear_sound = resource_handler.sound.wait_load_file("res/stage-clear.wav".to_string())?;

        self.overworld_music = resource_handler.sound.wait_load_file("res/overworld.ogg".to_string())?;
        self.underground_music = resource_handler.sound.wait_load_file("res/underground.ogg".to_string())?;
        Ok(None)
    }
    fn on_ready(&mut self, ctx: &mut UpdateContext<ObjectType>) {
        if ctx.scene().name() == MarioOverworldScene.name() {
            *ctx.viewport().clear_col() = Colour::from_bytes(92, 148, 252, 255);
            self.music = self.overworld_music.clone();
        } else {
            *ctx.viewport().clear_col() = Colour::black();
            self.music = self.underground_music.clone();
        }
        self.music.play_loop();
        self.last_nonzero_dir = Vec2::right();
        ctx.object().add_child(
            CollisionShape::from_object_sprite(self, self.current_sprite())
        );
    }

    fn on_update_begin(&mut self, _delta: Duration, ctx: &mut UpdateContext<ObjectType>) {
        self.accel = 0.;
        self.v_accel = 0.;
        self.maybe_start_exit_pipe(ctx);
    }
    fn on_update(&mut self, _delta: Duration, ctx: &mut UpdateContext<ObjectType>) {
        if ctx.input().pressed(KeyCode::W) {
            if self.show_wireframes {
                ctx.object().others_as_mut::<CollisionShape>()
                    .into_iter()
                    .for_each(|mut shape| shape.hide_wireframe());
                self.show_wireframes = false;
            } else {
                ctx.object().others_as_mut::<CollisionShape>()
                    .into_iter()
                    .for_each(|mut shape| shape.show_wireframe());
                self.show_wireframes = true;
            }
        }

        if self.state == PlayerState::Dying {
            self.speed = 0.;
            self.v_accel = BASE_GRAVITY;
            return;
        }
        if self.state == PlayerState::EnteringPipe {
            return;
        }
        let new_dir = if ctx.input().down(KeyCode::Left) && !ctx.input().down(KeyCode::Right) {
            Vec2::left()
        } else if !ctx.input().down(KeyCode::Left) && ctx.input().down(KeyCode::Right) {
            Vec2::right()
        } else {
            Vec2::zero()
        };
        let hold_run = ctx.input().down(KeyCode::X);
        self.hold_jump = ctx.input().down(KeyCode::Z);

        self.maybe_start_jump(ctx);
        match self.state {
            PlayerState::Idle => self.update_as_idle(ctx, new_dir, hold_run),
            PlayerState::Walking => self.update_as_walking(ctx, new_dir, hold_run),
            PlayerState::Running => self.update_as_running(ctx, new_dir, hold_run),
            PlayerState::Skidding => self.update_as_skidding(new_dir, hold_run),
            PlayerState::Falling => self.update_as_falling(new_dir),
            _ => return,
        }

        if !self.dir.is_zero() {
            self.last_nonzero_dir = self.dir;
        }
    }
    fn on_fixed_update(&mut self, ctx: &mut UpdateContext<ObjectType>) {
        self.speed += self.accel;
        self.v_speed += self.v_accel;
        self.v_speed = Self::MAX_VSPEED.min(self.v_speed);

        if !self.has_control() {
            self.centre += self.speed * Vec2::right() + self.v_speed * Vec2::down();
            return;
        }

        self.maybe_start_pipe(ctx);
        match ctx.object().test_collision_along(self.dir, self.speed, vec![BLOCK_COLLISION_TAG]) {
            Some(collisions) => {
                self.centre += self.speed * self.dir + collisions.first().mtv.project(Vec2::right());
                self.speed *= 0.9;
            }
            None => self.centre += self.speed * self.dir,
        }

        match ctx.object().test_collision_along(Vec2::down(), self.v_speed, vec![BLOCK_COLLISION_TAG]) {
            Some(collisions) => {
                let mut coll = collisions.into_iter()
                    .min_by(|a, b| {
                        let da = self.centre.x - a.other.transform().centre.x;
                        let db = self.centre.x - b.other.transform().centre.x;
                        da.abs().partial_cmp(&db.abs()).unwrap()
                    })
                    .unwrap();
                let mtv = coll.mtv.project(Vec2::down());
                self.centre += self.v_speed * Vec2::down() + mtv;
                self.v_speed = 0.;
                if mtv.y < 0. {
                    // Collision with the ground.
                    self.state = self.last_ground_state;
                    if self.speed.is_zero() {
                        self.state = PlayerState::Idle;
                    }
                } else if let Some(mut other) = downcast_bumpable_mut(&mut coll.other) {
                    // Collision with a block from below.
                    self.bump_sound.play();
                    other.bump(self);
                }
            }
            None => self.centre += self.v_speed * Vec2::down(),
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

    fn on_collision(&mut self, ctx: &mut UpdateContext<ObjectType>, mut other: SceneObjectWithId<ObjectType>, mtv: Vec2) -> CollisionResponse {
        if !self.has_control() {
            return CollisionResponse::Done;
        }
        {
            let bottom = ctx.object().rect_of(&other).bottom();
            if let Some(mut stompable) = downcast_stompable_mut(&mut other) {
                if !stompable.dead() {
                    if ctx.object().rect().bottom() < bottom {
                        stompable.stomp();
                        self.stomp_sound.play();
                        self.v_speed = self.initial_vspeed();
                        self.state = PlayerState::Falling;
                    } else {
                        self.start_die(ctx);
                    }
                }
            }
        }
        if other.get_type() == ObjectType::Flagpole {
            self.state = PlayerState::RidingFlagpole;
            self.music.stop();
            self.flagpole_sound.play();
            let dest_x = other.transform().centre.x - self.current_sprite().half_widths().x;
            ctx.scene().start_coroutine(move |this, ctx, _last_state| {
                let mut this = this.downcast_mut::<Self>().unwrap();
                this.centre.x = linalg::lerp(this.centre.x, dest_x, 0.2);
                this.speed = 0.;
                this.v_speed = Self::MAX_VSPEED / 3.;
                if let Some(collisions) = ctx.object()
                    .test_collision(vec![BLOCK_COLLISION_TAG]) {
                    this.v_speed = 0.;
                    this.centre += collisions.first().mtv;
                    ctx.object().add_child(WinTextDisplay::new(Vec2 { x: 8., y: -200. }));
                    this.clear_sound.play();
                    CoroutineResponse::Complete
                } else {
                    CoroutineResponse::Yield
                }
            });
        } else {
            self.centre += mtv;
        }
        CollisionResponse::Done
    }
    fn on_update_end(&mut self, _delta: Duration, ctx: &mut UpdateContext<ObjectType>) {
        ctx.viewport().clamp_to_left(None, Some(self.centre.x - 200.));
        ctx.viewport().clamp_to_right(Some(self.centre.x + 200.), None);
        ctx.viewport().clamp_to_left(Some(0.), None);

        let death_y = ctx.viewport().bottom() + self.current_sprite_mut().aa_extent().y;
        if self.has_control() && self.centre.y > death_y {
            self.start_die(ctx);
        }

        if self.state != self.last_state {
            self.walk_sprite.hide();
            self.run_sprite.hide();
            self.idle_sprite.hide();
            self.skid_sprite.hide();
            self.fall_sprite.hide();
            self.die_sprite.hide();
            self.flagpole_sprite.hide();
            self.current_sprite_mut().show();
            self.last_state = self.state;
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
    fn emitting_tags(&self) -> Vec<&'static str> {
        [PLAYER_COLLISION_TAG].into()
    }
    fn listening_tags(&self) -> Vec<&'static str> {
        [BLOCK_COLLISION_TAG, ENEMY_COLLISION_TAG, FLAG_COLLISION_TAG].into()
    }
}
