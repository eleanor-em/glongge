use std::time::{Duration, Instant};
use num_traits::{FloatConst, Zero};
use rand::{distributions::{Distribution, Uniform}, Rng};
use glongge_derive::*;
use glongge::{
    core::{
        prelude::*,
        scene::{Scene, SceneName},
    },
    resource::sprite::Sprite
};
use crate::object_type::ObjectType;

#[allow(dead_code)]
#[derive(Copy, Clone)]
pub struct RectangleScene;
impl Scene<ObjectType> for RectangleScene {
    fn name(&self) -> SceneName { "rectangle".into() }

    fn create_objects(&self, _entrance_id: usize) -> Vec<AnySceneObject<ObjectType>> {
        vec![
            RectangleSpawner::new(),
            RectanglePlayer::new(),
        ]
    }
}

// #[register_object_type]
// pub enum ObjectType {
//     Spawner,
//     Player,
//     SpinningRectangle,
// }

const RECTANGLE_COLL_TAG: &str = "RECTANGLE_COLL_TAG";

#[register_scene_object]
pub struct RectangleSpawner {}

#[partially_derive_scene_object]
impl SceneObject<ObjectType> for RectangleSpawner {
    fn on_update(&mut self, _delta: Duration, ctx: &mut UpdateContext<ObjectType>) {
        const N: usize = 1;
        let objects = Uniform::new(0., ctx.viewport().right())
            .sample_iter(rand::thread_rng())
            .zip(Uniform::new(0., ctx.viewport().bottom())
                .sample_iter(rand::thread_rng()))
            .zip(Uniform::new(-1., 1.)
                .sample_iter(rand::thread_rng()))
            .zip(Uniform::new(-1., 1.)
                .sample_iter(rand::thread_rng()))
            .take(N)
            .map(|(((x, y), vx), vy)|  {
                let pos = Vec2 { x, y };
                let vel = Vec2 { x: vx, y: vy };
                SpinningRectangle::new(pos, vel.normed())
            })
            .collect();
        ctx.object().add_vec(objects);
        ctx.object().remove_this();
    }

    fn transform(&self) -> Transform {
        Transform::default()
    }
}

#[register_scene_object]
pub struct RectanglePlayer {
    pos: Vec2,
    vel: Vec2,
    sprite: Sprite,
}

impl RectanglePlayer {
    // const SIZE: f64 = 100.;
    const SPEED: f64 = 300.;
}

#[partially_derive_scene_object]
impl SceneObject<ObjectType> for RectanglePlayer {
    fn on_load(&mut self, object_ctx: &mut ObjectContext<ObjectType>, resource_handler: &mut ResourceHandler) -> Result<Option<RenderItem>> {
        let texture = resource_handler.texture.wait_load_file("res/mario.png".to_string())?;
        self.sprite = Sprite::from_tileset(
            object_ctx,
            texture,
            Vec2Int { x: 3, y: 1 },
            Vec2Int { x: 16, y: 16 },
            Vec2Int { x: 0, y: 0 },
            Vec2Int { x: 2, y: 0 }
        ).with_fixed_ms_per_frame(100);
        Ok(None)
    }
    fn on_ready(&mut self, _ctx: &mut UpdateContext<ObjectType>) {
        self.pos = Vec2 { x: 512., y: 384. };
    }
    fn on_update(&mut self, delta: Duration, ctx: &mut UpdateContext<ObjectType>) {
        let mut direction = Vec2::zero();
        if ctx.input().down(KeyCode::Left) { direction += Vec2::left(); }
        if ctx.input().down(KeyCode::Right) { direction += Vec2::right(); }
        if ctx.input().down(KeyCode::Up) { direction += Vec2::up(); }
        if ctx.input().down(KeyCode::Down) { direction += Vec2::down(); }
        self.vel = Self::SPEED * direction.normed();
        self.pos += self.vel * delta.as_secs_f64();
    }

    fn transform(&self) -> Transform {
        Transform {
            centre: self.pos,
            ..Default::default()
        }
    }
    fn emitting_tags(&self) -> Vec<&'static str> {
        [RECTANGLE_COLL_TAG].into()
    }
}

#[register_scene_object]
pub struct SpinningRectangle {
    pos: Vec2,
    velocity: Vec2,
    t: f64,
    col: Colour,
    sprite: Sprite,
    alive_since: Instant,
}

impl SpinningRectangle {
    // const SIZE: f64 = 8.;
    const VELOCITY: f64 = 220.;
    const ANGULAR_VELOCITY: f64 = 2.;

    pub fn new(pos: Vec2, vel_normed: Vec2) -> AnySceneObject<ObjectType> {
        let mut rng = rand::thread_rng();
        let col = match rng.gen_range(0..6) {
            0 => Colour::red(),
            1 => Colour::blue(),
            2 => Colour::green(),
            3 => Colour::cyan(),
            4 => Colour::magenta(),
            5 => Colour::yellow(),
            _ => unreachable!(),
        };
        AnySceneObject::new(Self {
            pos,
            velocity: vel_normed * Self::VELOCITY,
            col,
            ..Default::default()
        })
    }

    fn rotation(&self) -> f64 { Self::ANGULAR_VELOCITY * f64::PI() * self.t }
}
#[partially_derive_scene_object]
impl SceneObject<ObjectType> for SpinningRectangle {
    fn on_load(&mut self, object_ctx: &mut ObjectContext<ObjectType>, resource_handler: &mut ResourceHandler) -> Result<Option<RenderItem>> {
        let texture = resource_handler.texture.wait_load_file("res/goomba.png".to_string())?;
        self.sprite = Sprite::from_tileset(
            object_ctx,
            texture,
            Vec2Int{ x: 2, y: 1 },
            Vec2Int { x: 16, y: 16 },
            Vec2Int { x: 0, y: 0 },
            Vec2Int { x: 2, y: 0 }
        ).with_fixed_ms_per_frame(500);
        Ok(None)
    }
    fn on_update_begin(&mut self, delta: Duration, ctx: &mut UpdateContext<ObjectType>) {
        let next_pos = self.pos + self.velocity * delta.as_secs_f64();
        if !(0.0..ctx.viewport().right()).contains(&next_pos.x) {
            self.velocity.x = -self.velocity.x;
        }
        if !(0.0..ctx.viewport().bottom()).contains(&next_pos.y) {
            self.velocity.y = -self.velocity.y;
        }
    }
    fn on_update(&mut self, delta: Duration, ctx: &mut UpdateContext<ObjectType>) {
        self.t += delta.as_secs_f64();
        self.pos += self.velocity * delta.as_secs_f64();

        if ctx.input().pressed(KeyCode::Space) {
            let mut rng = rand::thread_rng();
            let angle = rng.gen_range(0.0..(2. * f64::PI()));
            ctx.object().add_child(SpinningRectangle::new(
                self.pos,
                Vec2::one().rotated(angle)
            ));
            self.velocity = -Self::VELOCITY * Vec2::one().rotated(angle);
        }
    }
    fn on_collision(&mut self, _ctx: &mut UpdateContext<ObjectType>, other: SceneObjectWithId<ObjectType>, mtv: Vec2) -> CollisionResponse {
        self.pos += mtv;

        if let Some(mut rect) = other.downcast_mut::<SpinningRectangle>() {
            if self.alive_since.elapsed().as_secs_f64() > 0.5 && rect.alive_since.elapsed().as_secs_f64() > 0.5 {
                self.velocity = self.velocity.reflect(mtv.normed());
                rect.col = self.col;
            }
        } else {
            self.velocity = self.velocity.reflect(mtv.normed());
        }
        CollisionResponse::Continue
    }

    fn transform(&self) -> Transform {
        Transform {
            centre: self.pos,
            rotation: self.rotation(),
            ..Default::default()
        }
    }
    fn emitting_tags(&self) -> Vec<&'static str> {
        [RECTANGLE_COLL_TAG].into()
    }
    fn listening_tags(&self) -> Vec<&'static str> {
        [RECTANGLE_COLL_TAG].into()
    }
}
