use num_traits::{Float, FloatConst};
use rand::distributions::{Distribution, Uniform};
use rand::Rng;
use std::{
    cell::RefCell,
    time::{Duration, Instant},
};

use crate::{
    core::linalg::Vec2,
    gg::{
        sample::BasicRenderHandler, RenderData, RenderableObject, SceneObject, UpdateContext,
        WorldObject,
    },
    scene::Scene,
};

pub fn create_spinning_triangle_scene(
    render_handler: &BasicRenderHandler,
) -> Scene<BasicRenderHandler> {
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
    let objects: Vec<_> = (0..N)
        .map(|i| {
            let pos = Vec2 { x: xs[i], y: ys[i] };
            let vel = Vec2 {
                x: vxs[i],
                y: vys[i],
            };
            RefCell::new(Box::new(SpinningTriangle::new(pos, vel.normed())) as Box<dyn SceneObject>)
        })
        .collect();
    Scene::new(objects, render_handler)
}

#[derive(Clone)]
pub struct SpinningTriangle {
    pos: Vec2,
    velocity: Vec2,
    t: f64,
    last_spawn: Instant,
    alive_since: Instant,
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
            last_spawn: Instant::now(),
            alive_since: Instant::now(),
        }
    }
}
impl SceneObject for SpinningTriangle {
    fn on_ready(&mut self) {}

    fn on_update(&mut self, delta: Duration, mut update_ctx: UpdateContext) {
        let delta_s = delta.as_secs_f64();
        self.t += delta_s;
        let next_pos = self.pos + self.velocity * delta_s;
        if !(0.0..1024.0).contains(&next_pos.x) {
            self.velocity.x = -self.velocity.x;
        }
        if !(0.0..768.0).contains(&next_pos.y) {
            self.velocity.y = -self.velocity.y;
        }
        if self.alive_since.elapsed().as_secs_f64() > 0.1 {
            for other in update_ctx.others() {
                if let Some(other) = other.as_world_object() {
                    if (other.world_pos() - self.pos).mag() < Self::TRI_WIDTH {
                        self.velocity = (self.pos - other.world_pos()).normed() * Self::VELOCITY;
                    }
                }
            }
        }
        self.pos += self.velocity * delta_s;

        if self.last_spawn.elapsed().as_secs() >= 1 && update_ctx.others().len() < 2500 {
            self.last_spawn = Instant::now();
            let mut rng = rand::thread_rng();
            if rng.gen_bool(0.1) {
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

    fn as_world_object(&self) -> Option<&dyn WorldObject> {
        Some(self)
    }
    fn as_renderable_object(&self) -> Option<&dyn RenderableObject> {
        Some(self)
    }
}

impl WorldObject for SpinningTriangle {
    fn world_pos(&self) -> Vec2 {
        self.pos
    }
}

impl RenderableObject for SpinningTriangle {
    fn create_vertices(&self) -> Vec<Vec2> {
        let tri_height = SpinningTriangle::TRI_WIDTH * 3.0.sqrt();
        let centre_correction = -tri_height / 6.0;
        let vertex1 = Vec2 {
            x: -SpinningTriangle::TRI_WIDTH,
            y: -tri_height / 2.0 - centre_correction,
        };
        let vertex2 = Vec2 {
            x: SpinningTriangle::TRI_WIDTH,
            y: -tri_height / 2.0 - centre_correction,
        };
        let vertex3 = Vec2 {
            x: 0.0,
            y: tri_height / 2.0 - centre_correction,
        };
        vec![vertex1, vertex2, vertex3]
    }

    fn render_data(&self) -> RenderData {
        RenderData {
            position: self.pos,
            rotation: SpinningTriangle::ANGULAR_VELOCITY * f64::PI() * self.t,
        }
    }
}
