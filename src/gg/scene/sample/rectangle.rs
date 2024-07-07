use std::{
    any::Any,
    sync::{Arc, Mutex},
    time::{
        Duration,
    }
};
use num_traits::{FloatConst, Zero};
use rand::{
    distributions::{Distribution, Uniform},
    Rng
};

use crate::{
    core::{
        collision::{BoxCollider, Collider},
        input::{InputHandler, KeyCode},
        linalg::Vec2,
    },
    gg::{
        self,
        sample::BasicRenderHandler,
        scene::Scene,
        UpdateContext,
        RenderableObject,
        SceneObjectWithId,
        Transform
    },
    shader,
};
use glongge_derive::register_object_type;

pub fn create_scene(
    render_handler: &BasicRenderHandler,
    input_handler: Arc<Mutex<InputHandler>>
) -> Scene<ObjectType, BasicRenderHandler> {
    Scene::new(vec![Box::new(Spawner {}),
                    Box::new(Player {
                        pos: Vec2 { x: 500.0, y: 500.0 },
                        vel: Vec2::zero(),
                    })],
               render_handler, input_handler)
}

#[register_object_type]
pub enum ObjectType {
    Player,
    Spawner,
    SpinningRectangle,
}

const RECTANGLE_COLL_TAG: &str = "RECTANGLE_COLL_TAG";

struct Player {
    pos: Vec2,
    vel: Vec2,
}

impl Player {
    const SIZE: f64 = 200.0;
    const SPEED: f64 = 300.0;
}

impl gg::SceneObject<ObjectType> for Player {
    fn get_type(&self) -> ObjectType { ObjectType::Player }
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }

    fn on_ready(&mut self) {}

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
            position: self.pos,
            rotation: f64::FRAC_PI_4(),
        }
    }

    fn as_renderable_object(&self) -> Option<&dyn RenderableObject<ObjectType>> {
        Some(self)
    }
    fn collider(&self) -> Option<Box<dyn Collider>> {
        Some(Box::new(BoxCollider::square(self.transform(), Self::SIZE)))
    }
    fn collision_tags(&self) -> Vec<&'static str> {
        vec![RECTANGLE_COLL_TAG]
    }
}

impl gg::RenderableObject<ObjectType> for Player {
    fn create_vertices(&self) -> Vec<Vec2> {
        shader::vertex::rectangle(Vec2::zero(), Self::SIZE * Vec2::one())
    }

    fn render_data(&self) -> gg::RenderData {
        gg::RenderData { colour: [0.0, 1.0, 0.0, 1.0], }
    }
}

#[derive(Default)]
struct Spawner {}

impl gg::SceneObject<ObjectType> for Spawner {
    fn get_type(&self) -> ObjectType { ObjectType::Spawner }
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }

    fn on_ready(&mut self) {}

    fn on_update(&mut self, _delta: Duration, mut update_ctx: gg::UpdateContext<ObjectType>) {
        const N: usize = 10;
        let objects = Uniform::new(0.0, update_ctx.viewport.logical_width() as f64)
            .sample_iter(rand::thread_rng())
            .zip(Uniform::new(0.0, update_ctx.viewport.logical_height() as f64)
                .sample_iter(rand::thread_rng()))
            .zip(Uniform::new(-1.0, 1.0)
                .sample_iter(rand::thread_rng()))
            .zip(Uniform::new(-1.0, 1.0)
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

#[derive(Default)]
struct SpinningRectangle {
    pos: Vec2,
    velocity: Vec2,
    t: f64,
}

impl SpinningRectangle {
    const SIZE: f64 = 12.0;
    const VELOCITY: f64 = 220.0;
    const ANGULAR_VELOCITY: f64 = 2.0;

    pub fn new(pos: Vec2, vel_normed: Vec2) -> Self {
        Self {
            pos,
            velocity: vel_normed * Self::VELOCITY,
            t: 0.0,
        }
    }

    fn rotation(&self) -> f64 { Self::ANGULAR_VELOCITY * f64::PI() * self.t }
}
impl gg::SceneObject<ObjectType> for SpinningRectangle {
    fn get_type(&self) -> ObjectType { ObjectType::SpinningRectangle }
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }

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
    fn on_update(&mut self, delta: Duration, mut update_ctx: gg::UpdateContext<ObjectType>) {
        self.t += delta.as_secs_f64();
        self.pos += self.velocity * delta.as_secs_f64();

        if update_ctx.input().pressed(KeyCode::Space) &&
                update_ctx.others().len() < 2500 &&
                update_ctx.viewport().contains(self.pos) {
            let mut rng = rand::thread_rng();
            if rng.gen_bool(0.3) {
                let vel = Vec2 {
                    x: rng.gen_range(-1.0..1.0),
                    y: rng.gen_range(-1.0..1.0),
                };
                update_ctx.add_object(Box::new(SpinningRectangle::new(
                    self.pos,
                    (self.velocity - vel).normed(),
                )));
                update_ctx.add_object(Box::new(SpinningRectangle::new(
                    self.pos,
                    (self.velocity + vel).normed(),
                )));
                update_ctx.remove_this_object();
            }
        }
    }
    fn on_collision(&mut self, _other: SceneObjectWithId<ObjectType>, mtv: Vec2) {
        self.velocity = self.velocity.reflect(mtv.normed());
        self.pos += mtv;
    }

    fn as_renderable_object(&self) -> Option<&dyn gg::RenderableObject<ObjectType>> {
        Some(self)
    }

    fn transform(&self) -> Transform {
        Transform {
            position: self.pos,
            rotation: self.rotation(),
        }
    }

    fn collider(&self) -> Option<Box<dyn Collider>> {
        Some(Box::new(BoxCollider::square(self.transform(), Self::SIZE)))
    }
    fn collision_tags(&self) -> Vec<&'static str> {
        vec![RECTANGLE_COLL_TAG]
    }
    fn emit_collisions_for(&self) -> Vec<&'static str> {
        vec![RECTANGLE_COLL_TAG]
    }
}

impl gg::RenderableObject<ObjectType> for SpinningRectangle {
    fn create_vertices(&self) -> Vec<Vec2> {
        shader::vertex::rectangle(Vec2::zero(), Self::SIZE * Vec2::one())
    }

    fn render_data(&self) -> gg::RenderData {
        gg::RenderData {
            colour: [1.0, 0.0, 0.0, 1.0],
        }
    }
}
