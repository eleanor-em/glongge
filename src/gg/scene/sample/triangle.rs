use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant}
};
use num_traits::{Float, FloatConst, Zero};
use rand::{
    distributions::{Distribution, Uniform},
    Rng
};
use glongge_derive::register_object_type;

use crate::{
    core::{
        linalg::Vec2,
        input::{InputHandler, KeyCode}
    },
    gg::{
        self,
        sample::BasicRenderHandler,
        scene::Scene,
        SceneObject,
        UpdateContext
    },
};
use crate::gg::Transform;

pub fn create_scene(
    render_handler: &BasicRenderHandler,
    input_handler: Arc<Mutex<InputHandler>>
) -> Scene<ObjectType, BasicRenderHandler> {
    Scene::new(vec![Box::new(Spawner {})], render_handler, input_handler)
}

#[register_object_type]
pub enum ObjectType {
    Spawner,
    SpinningTriangle,
}

#[derive(Default)]
struct Spawner {}

impl gg::SceneObject<ObjectType> for Spawner {
    fn on_ready(&mut self) {}
    fn get_type(&self) -> ObjectType { ObjectType::Spawner }

    fn on_update(&mut self, _delta: Duration, mut update_ctx: gg::UpdateContext<ObjectType>) {
        const N: usize = 10;
        let mut rng = rand::thread_rng();
        let xs: Vec<f64> = Uniform::new(0.0, 1024.0)
            .sample_iter(&mut rng)
            .take(N)
            .collect();
        let ys: Vec<f64> = Uniform::new(0.0, 768.0)
            .sample_iter(&mut rng)
            .take(N)
            .collect();
        let vxs: Vec<f64> = Uniform::new(-1.0, 1.0)
            .sample_iter(&mut rng)
            .take(N)
            .collect();
        let vys: Vec<f64> = Uniform::new(-1.0, 1.0)
            .sample_iter(&mut rng)
            .take(N)
            .collect();
        let objects = (0..N)
            .map(|i| {
                let pos = Vec2 { x: xs[i], y: ys[i] };
                let vel = Vec2 {
                    x: vxs[i],
                    y: vys[i],
                };
                Box::new(SpinningTriangle { pos, velocity: vel.normed(), t: 0.0, alive_since: Instant::now() }) as Box<dyn gg::SceneObject<ObjectType>>
            })
            .collect();
        update_ctx.add_object_vec(objects);
        update_ctx.remove_this_object();
    }

    fn transform(&self) -> Transform {
        Transform::default()
    }
}

struct SpinningTriangle {
    pos: Vec2,
    velocity: Vec2,
    t: f64,
    alive_since: Instant,
}

impl Default for SpinningTriangle {
    fn default() -> Self {
        Self { pos: Vec2::zero(), velocity: Vec2::zero(), t: 0.0, alive_since: Instant::now() }
    }
}

impl SpinningTriangle {
    const TRI_WIDTH: f64 = 5.0;
    const VELOCITY: f64 = 200.0;
    const ANGULAR_VELOCITY: f64 = 1.0;

    pub fn new(pos: Vec2, vel_normed: Vec2) -> Self {
        Self {
            pos,
            velocity: vel_normed * Self::VELOCITY,
            t: 0.0,
            alive_since: Instant::now(),
        }
    }

    fn rotation(&self) -> f64 { Self::ANGULAR_VELOCITY * f64::PI() * self.t }
}
impl gg::SceneObject<ObjectType> for SpinningTriangle {
    fn on_ready(&mut self) {}
    fn get_type(&self) -> ObjectType { ObjectType::SpinningTriangle }

    fn on_update(&mut self, delta: Duration, mut update_ctx: gg::UpdateContext<ObjectType>) {
        let delta_s = delta.as_secs_f64();
        self.t += delta_s;
        let next_pos = self.pos + self.velocity * delta_s;
        if !(0.0..update_ctx.viewport().logical_width() as f64).contains(&next_pos.x) {
            self.velocity.x = -self.velocity.x;
        }
        if !(0.0..update_ctx.viewport().logical_height() as f64).contains(&next_pos.y) {
            self.velocity.y = -self.velocity.y;
        }
        self.pos += self.velocity * delta_s;

        if update_ctx.input().pressed(KeyCode::Space) &&
                update_ctx.others().len() < 2500 &&
                update_ctx.viewport().contains(self.pos) {
            let mut rng = rand::thread_rng();
            if rng.gen_bool(0.2) {
                let vel = Vec2 {
                    x: rng.gen_range(-1.0..1.0),
                    y: rng.gen_range(-1.0..1.0),
                };
                update_ctx.add_object(Box::new(SpinningTriangle::new(
                    self.pos,
                    (self.velocity - vel).normed(),
                )));
                update_ctx.add_object(Box::new(SpinningTriangle::new(
                    self.pos,
                    (self.velocity + vel).normed(),
                )));
                update_ctx.remove_this_object();
            }
        }
    }

    fn on_update_end(&mut self, _delta: Duration, update_ctx: UpdateContext<ObjectType>) {
        if self.alive_since.elapsed().as_secs_f64() > 0.1 &&
                update_ctx.viewport().contains(self.pos) {
            for other in update_ctx.others() {
                if (other.transform().position -  self.pos).len() < Self::TRI_WIDTH {
                    self.velocity = (self.pos - other.transform().position).normed() * Self::VELOCITY;
                }
            }
        }
    }

    fn transform(&self) -> Transform {
        Transform {
            position: self.pos,
            rotation: self.rotation(),
        }
    }

    fn as_renderable_object(&self) -> Option<&dyn gg::RenderableObject<ObjectType>> {
        Some(self)
    }
}

impl gg::RenderableObject<ObjectType> for SpinningTriangle {
    fn create_vertices(&self) -> Vec<Vec2> {
        let tri_height = SpinningTriangle::TRI_WIDTH * 3.0.sqrt();
        let centre_correction = -tri_height / 6.0;
        let vertex1 = Vec2 {
            x: -Self::TRI_WIDTH,
            y: -tri_height / 2.0 - centre_correction,
        };
        let vertex2 = Vec2 {
            x: Self::TRI_WIDTH,
            y: -tri_height / 2.0 - centre_correction,
        };
        let vertex3 = Vec2 {
            x: 0.0,
            y: tri_height / 2.0 - centre_correction,
        };
        vec![vertex1, vertex2, vertex3]
    }

    fn render_data(&self) -> gg::RenderData {
        gg::RenderData {
            colour: [1.0, 0.0, 0.0, 1.0],
        }
    }
}
