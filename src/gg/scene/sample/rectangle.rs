#[allow(unused_imports)]
use crate::core::prelude::*;

use std::{
    any::Any,
    time::{
        Duration,
        Instant
    }
};
use num_traits::{FloatConst, Zero};
use rand::{distributions::{Distribution, Uniform}, Rng};

use glongge_derive::*;
use crate::{
    core::linalg::Vec2Int,
    core::{
        collision::{BoxCollider, Collider},
        colour::Colour,
        input::KeyCode,
        linalg::Vec2,
    },
    gg::{
        self,
        UpdateContext,
        RenderableObject,
        RenderInfo,
        SceneObject,
        SceneObjectWithId,
        Transform,
        VertexWithUV
    },
    resource::{
        ResourceHandler,
        sprite::Sprite
    },
    shader,
};
use crate::gg::{AnySceneObject, CollisionResponse};
use crate::gg::scene::{Scene, SceneName};

#[allow(dead_code)]
#[derive(Copy, Clone)]
struct RectangleScene {}
impl Scene<ObjectType> for RectangleScene {
    fn name(&self) -> SceneName { "rectangle".into() }

    fn create_objects(&self, _entrance_id: usize) -> Vec<AnySceneObject<ObjectType>> {
        vec![
            Spawner::new(),
            Player::new(),
        ]
    }
}

#[register_object_type]
pub enum ObjectType {
    Spawner,
    Player,
    SpinningRectangle,
}

const RECTANGLE_COLL_TAG: &str = "RECTANGLE_COLL_TAG";

#[register_scene_object]
struct Spawner {}

#[partially_derive_scene_object]
impl SceneObject<ObjectType> for Spawner {
    fn on_ready(&mut self) {}
    fn on_update(&mut self, _delta: Duration, mut update_ctx: UpdateContext<ObjectType>) {
        const N: usize = 1;
        let objects = Uniform::new(0., update_ctx.viewport.logical_width() as f64)
            .sample_iter(rand::thread_rng())
            .zip(Uniform::new(0., update_ctx.viewport.logical_height() as f64)
                .sample_iter(rand::thread_rng()))
            .zip(Uniform::new(-1., 1.)
                .sample_iter(rand::thread_rng()))
            .zip(Uniform::new(-1., 1.)
                .sample_iter(rand::thread_rng()))
            .take(N)
            .map(|(((x, y), vx), vy)|  {
                let pos = Vec2 { x, y };
                let vel = Vec2 { x: vx, y: vy };
                Box::new(SpinningRectangle::new(pos, vel.normed())).into()
            })
            .collect();
        update_ctx.add_object_vec(objects);
        update_ctx.remove_this_object();
    }

    fn transform(&self) -> Transform {
        Transform::default()
    }
}

#[register_scene_object]
struct Player {
    pos: Vec2,
    vel: Vec2,
    sprite: Sprite,
}

impl Player {
    const SIZE: f64 = 100.;
    const SPEED: f64 = 300.;
}

#[partially_derive_scene_object]
impl SceneObject<ObjectType> for Player {
    fn on_load(&mut self, resource_handler: &mut ResourceHandler) -> Result<()> {
        let texture_id = resource_handler.texture.wait_load_file("res/mario.png".to_string())?;
        self.sprite = Sprite::from_tileset(
            texture_id,
            Vec2Int { x: 3, y: 1 },
            Vec2Int { x: 16, y: 16 },
            Vec2Int { x: 0, y: 0 },
            Vec2Int { x: 2, y: 0 }
        ).with_fixed_ms_per_frame(100);
        Ok(())
    }
    fn on_ready(&mut self) {
        self.pos = Vec2 { x: 512., y: 384. };
    }
    fn on_update(&mut self, delta: Duration, update_ctx: UpdateContext<ObjectType>) {
        let mut direction = Vec2::zero();
        if update_ctx.input().down(KeyCode::Left) { direction += Vec2::left(); }
        if update_ctx.input().down(KeyCode::Right) { direction += Vec2::right(); }
        if update_ctx.input().down(KeyCode::Up) { direction += Vec2::up(); }
        if update_ctx.input().down(KeyCode::Down) { direction += Vec2::down(); }
        self.vel = Self::SPEED * direction.normed();
        self.pos += self.vel * delta.as_secs_f64();
    }

    fn transform(&self) -> Transform {
        Transform {
            centre: self.pos,
            ..Default::default()
        }
    }
    fn as_renderable_object(&self) -> Option<&dyn RenderableObject<ObjectType>> {
        Some(self)
    }
    fn collider(&self) -> Box<dyn Collider> {
        Box::new(BoxCollider::square(self.transform(), Self::SIZE))
    }
    fn emitting_tags(&self) -> Vec<&'static str> {
        [RECTANGLE_COLL_TAG].into()
    }
}

impl RenderableObject<ObjectType> for Player {
    fn create_vertices(&self) -> Vec<VertexWithUV> {
        shader::vertex::rectangle_with_uv(Vec2::zero(), Self::SIZE * Vec2::one())
    }

    fn render_info(&self) -> RenderInfo {
        self.sprite.render_info_default()
    }
}

#[register_scene_object]
struct SpinningRectangle {
    pos: Vec2,
    velocity: Vec2,
    t: f64,
    col: Colour,
    sprite: Sprite,
    alive_since: Instant,
}

impl SpinningRectangle {
    const SIZE: f64 = 8.;
    const VELOCITY: f64 = 220.;
    const ANGULAR_VELOCITY: f64 = 2.;

    pub fn new(pos: Vec2, vel_normed: Vec2) -> Self {
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
        Self {
            pos,
            velocity: vel_normed * Self::VELOCITY,
            t: 0.,
            col,
            sprite: Default::default(),
            alive_since: Instant::now(),
        }
    }

    fn rotation(&self) -> f64 { Self::ANGULAR_VELOCITY * f64::PI() * self.t }
}
#[partially_derive_scene_object]
impl SceneObject<ObjectType> for SpinningRectangle {
    fn on_load(&mut self, resource_handler: &mut ResourceHandler) -> Result<()> {
        let texture_id = resource_handler.texture.wait_load_file("res/goomba.png".to_string())?;
        self.sprite = Sprite::from_tileset(texture_id,
            Vec2Int{ x: 2, y: 1 },
            Vec2Int { x: 16, y: 16 },
            Vec2Int { x: 0, y: 0 },
            Vec2Int { x: 2, y: 0 }
        ).with_fixed_ms_per_frame(500);
        Ok(())
    }
    fn on_ready(&mut self) {}
    fn on_update_begin(&mut self, delta: Duration, update_ctx: UpdateContext<ObjectType>) {
        let next_pos = self.pos + self.velocity * delta.as_secs_f64();
        if !(0.0..update_ctx.viewport().logical_width() as f64).contains(&next_pos.x) {
            self.velocity.x = -self.velocity.x;
        }
        if !(0.0..update_ctx.viewport().logical_height() as f64).contains(&next_pos.y) {
            self.velocity.y = -self.velocity.y;
        }
    }
    fn on_update(&mut self, delta: Duration, mut update_ctx: UpdateContext<ObjectType>) {
        self.t += delta.as_secs_f64();
        self.pos += self.velocity * delta.as_secs_f64();

        if update_ctx.input().pressed(KeyCode::Space) {
            let mut rng = rand::thread_rng();
            let angle = rng.gen_range(0.0..(2. * f64::PI()));
            update_ctx.add_object(Box::new(SpinningRectangle::new(
                self.pos,
                Vec2::one().rotated(angle)
            )));
            self.velocity = -Self::VELOCITY * Vec2::one().rotated(angle);
        }
    }
    fn on_collision(&mut self, _update_ctx: UpdateContext<ObjectType>, mut other: SceneObjectWithId<ObjectType>, mtv: Vec2) -> CollisionResponse {
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
    fn as_renderable_object(&self) -> Option<&dyn RenderableObject<ObjectType>> {
        Some(self)
    }
    fn collider(&self) -> Box<dyn Collider> {
        Box::new(BoxCollider::square(self.transform(), Self::SIZE))
    }
    fn emitting_tags(&self) -> Vec<&'static str> {
        [RECTANGLE_COLL_TAG].into()
    }
    fn listening_tags(&self) -> Vec<&'static str> {
        [RECTANGLE_COLL_TAG].into()
    }
}

impl RenderableObject<ObjectType> for SpinningRectangle {
    fn create_vertices(&self) -> Vec<VertexWithUV> {
        shader::vertex::rectangle_with_uv(Vec2::zero(), Self::SIZE * Vec2::one())
    }

    fn render_info(&self) -> RenderInfo {
        self.sprite.render_info_from(RenderInfo {
            col: self.col,
            ..Default::default()
        })
    }
}
