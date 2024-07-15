use std::{
    any::Any,
    sync::{Arc, Mutex},
    time::{
        Duration,
        Instant
    }
};
use num_traits::{FloatConst, Zero};
use rand::{distributions::{Distribution, Uniform}, Rng};

use anyhow::Result;

use crate::{
    core::{
        collision::{BoxCollider, Collider},
        colour::Colour,
        input::{InputHandler, KeyCode},
        linalg::Vec2,
    },
    gg::{
        self,
        render::BasicRenderHandler,
        scene::Scene,
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
        texture::TextureId
    },
    shader,
};
use glongge_derive::register_object_type;

pub fn create_scene(
    resource_handler: ResourceHandler,
    render_handler: BasicRenderHandler,
    input_handler: Arc<Mutex<InputHandler>>
) -> Scene<ObjectType, BasicRenderHandler> {
    Scene::new(vec![Box::new(Spawner {}),
                Box::new(Player {
                    pos: Vec2 { x: 500.0, y: 500.0 },
                    vel: Vec2::zero(),
                    texture_id: TextureId::default(),
                })],
               input_handler, resource_handler, render_handler)
}

#[register_object_type]
pub enum ObjectType {
    Spawner,
    Player,
    SpinningRectangle,
}

const RECTANGLE_COLL_TAG: &str = "RECTANGLE_COLL_TAG";

struct Spawner {}

impl SceneObject<ObjectType> for Spawner {
    fn get_type(&self) -> ObjectType { ObjectType::Spawner }
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }

    fn on_ready(&mut self) {}
    fn on_update(&mut self, _delta: Duration, mut update_ctx: UpdateContext<ObjectType>) {
        const N: usize = 1;
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

struct Player {
    pos: Vec2,
    vel: Vec2,
    texture_id: TextureId,
}

impl Player {
    const SIZE: f64 = 200.0;
    const SPEED: f64 = 300.0;
}

impl SceneObject<ObjectType> for Player {
    fn get_type(&self) -> ObjectType { ObjectType::Player }
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }

    fn on_load(&mut self, resource_handler: &mut ResourceHandler) -> Result<()> {
        self.texture_id = resource_handler.texture.wait_load_file("res/mario.png".to_string())?;
        Ok(())
    }
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
        [RECTANGLE_COLL_TAG].into()
    }
}

impl RenderableObject<ObjectType> for Player {
    fn create_vertices(&self) -> Vec<VertexWithUV> {
        shader::vertex::rectangle_with_uv(Vec2::zero(), Self::SIZE * Vec2::one())
    }

    fn render_data(&self) -> RenderInfo {
        RenderInfo { col: Colour::green(), texture_id: Some(self.texture_id) }
    }
}

struct SpinningRectangle {
    pos: Vec2,
    velocity: Vec2,
    t: f64,
    col: Colour,
    texture_id: TextureId,
    alive_since: Instant,
}

impl SpinningRectangle {
    const SIZE: f64 = 16.0;
    const VELOCITY: f64 = 220.0;
    const ANGULAR_VELOCITY: f64 = 2.0;

    pub fn new(pos: Vec2, vel_normed: Vec2) -> Self {
        let mut rng = rand::thread_rng();
        let col = match rng.gen_range(0..6) {
            0 => Colour::red(),
            1 => Colour::blue(),
            2 => Colour::green(),
            3 => Colour::cyan(),
            4 => Colour::magenta(),
            5 => Colour::yellow(),
            _ => panic!(),
        };
        Self {
            pos,
            velocity: vel_normed * Self::VELOCITY,
            t: 0.0,
            col,
            texture_id: TextureId::default(),
            alive_since: Instant::now(),
        }
    }

    fn rotation(&self) -> f64 { Self::ANGULAR_VELOCITY * f64::PI() * self.t }
}
impl SceneObject<ObjectType> for SpinningRectangle {
    fn get_type(&self) -> ObjectType { ObjectType::SpinningRectangle }
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }

    fn on_load(&mut self, resource_handler: &mut ResourceHandler) -> Result<()> {
        self.texture_id = resource_handler.texture.wait_load_file("res/goomba.png".to_string())?;
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
            let angle = rng.gen_range(0.0..(2.0 * f64::PI()));
            update_ctx.add_object(Box::new(SpinningRectangle::new(
                self.pos,
                Vec2::one().rotated(angle)
            )));
            self.velocity = -Self::VELOCITY * Vec2::one().rotated(angle);
        }
    }
    fn on_collision(&mut self, mut other: SceneObjectWithId<ObjectType>, mtv: Vec2) {
        self.pos += mtv;

        if let Some(mut rect) = other.downcast_mut::<SpinningRectangle>() {
            if self.alive_since.elapsed().as_secs_f64() > 0.5 && rect.alive_since.elapsed().as_secs_f64() > 0.5 {
                self.velocity = self.velocity.reflect(mtv.normed());
                rect.col = self.col;
            }
        } else {
            self.velocity = self.velocity.reflect(mtv.normed());
        }
    }

    fn transform(&self) -> Transform {
        Transform {
            position: self.pos,
            rotation: self.rotation(),
        }
    }
    fn as_renderable_object(&self) -> Option<&dyn RenderableObject<ObjectType>> {
        Some(self)
    }
    fn collider(&self) -> Option<Box<dyn Collider>> {
        Some(Box::new(BoxCollider::square(self.transform(), Self::SIZE)))
    }
    fn collision_tags(&self) -> Vec<&'static str> {
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

    fn render_data(&self) -> RenderInfo {
        RenderInfo {
            col: self.col,
            texture_id: Some(self.texture_id),
        }
    }
}
