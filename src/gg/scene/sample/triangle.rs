#[allow(unused_imports)]
use crate::core::prelude::*;

use std::{
    any::Any,
    time::{Duration, Instant},
};
use num_traits::{Float, FloatConst, Zero};
use rand::{
    distributions::{Distribution, Uniform},
    Rng
};
use glongge_derive::{partially_derive_scene_object, register_object_type};

use crate::{
    core::{
        colour::Colour,
        linalg::Vec2,
        input::KeyCode
    },
    gg::{
        self,
        Transform,
        UpdateContext,
        VertexWithUV
    },
};
use crate::core::linalg::AxisAlignedExtent;
use crate::gg::AnySceneObject;
use crate::gg::scene::{Scene, SceneName};

#[allow(dead_code)]
#[derive(Copy, Clone)]
struct TriangleScene {}
impl Scene<ObjectType> for TriangleScene {
    fn name(&self) -> SceneName { "triangle".into() }

    fn create_objects(&self, _entrance_id: usize) -> Vec<AnySceneObject<ObjectType>> {
        vec![
            Box::new(Spawner{}),
        ]
    }
}

#[register_object_type]
pub enum ObjectType {
    Spawner,
    SpinningTriangle,
}

#[derive(Default)]
struct Spawner {}

#[partially_derive_scene_object]
impl gg::SceneObject<ObjectType> for Spawner {
    fn on_ready(&mut self) {}

    fn on_update(&mut self, _delta: Duration, mut update_ctx: gg::UpdateContext<ObjectType>) {
        const N: usize = 10;
        let mut rng = rand::thread_rng();
        let xs: Vec<f64> = Uniform::new(0., 1024.)
            .sample_iter(&mut rng)
            .take(N)
            .collect();
        let ys: Vec<f64> = Uniform::new(0., 768.)
            .sample_iter(&mut rng)
            .take(N)
            .collect();
        let vxs: Vec<f64> = Uniform::new(-1., 1.)
            .sample_iter(&mut rng)
            .take(N)
            .collect();
        let vys: Vec<f64> = Uniform::new(-1., 1.)
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
                Box::new(SpinningTriangle { pos, velocity: vel.normed(), t: 0., alive_since: Instant::now() }) as Box<dyn gg::SceneObject<ObjectType>>
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
        Self { pos: Vec2::zero(), velocity: Vec2::zero(), t: 0., alive_since: Instant::now() }
    }
}

impl SpinningTriangle {
    const TRI_WIDTH: f64 = 5.;
    const VELOCITY: f64 = 200.;
    const ANGULAR_VELOCITY: f64 = 1.;

    pub fn new(pos: Vec2, vel_normed: Vec2) -> Self {
        Self {
            pos,
            velocity: vel_normed * Self::VELOCITY,
            t: 0.,
            alive_since: Instant::now(),
        }
    }

    fn rotation(&self) -> f64 { Self::ANGULAR_VELOCITY * f64::PI() * self.t }
}
#[partially_derive_scene_object]
impl gg::SceneObject<ObjectType> for SpinningTriangle {
    fn on_ready(&mut self) {}
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
                update_ctx.viewport().contains_point(self.pos) {
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
                update_ctx.viewport().contains_point(self.pos) {
            for other in update_ctx.others() {
                if (other.transform().centre -  self.pos).len() < Self::TRI_WIDTH {
                    self.velocity = (self.pos - other.transform().centre).normed() * Self::VELOCITY;
                }
            }
        }
    }

    fn transform(&self) -> Transform {
        Transform {
            centre: self.pos,
            rotation: self.rotation(),
            ..Default::default()
        }
    }

    fn as_renderable_object(&self) -> Option<&dyn gg::RenderableObject<ObjectType>> {
        Some(self)
    }
}

impl gg::RenderableObject<ObjectType> for SpinningTriangle {
    fn create_vertices(&self) -> Vec<VertexWithUV> {
        let tri_height = SpinningTriangle::TRI_WIDTH * 3.0.sqrt();
        let centre_correction = -tri_height / 6.;
        let vertex1 = Vec2 {
            x: -Self::TRI_WIDTH,
            y: -tri_height / 2. - centre_correction,
        };
        let vertex2 = Vec2 {
            x: Self::TRI_WIDTH,
            y: -tri_height / 2. - centre_correction,
        };
        let vertex3 = Vec2 {
            x: 0.,
            y: tri_height / 2. - centre_correction,
        };
        VertexWithUV::from_iter(vec![vertex1, vertex2, vertex3])
    }

    fn render_info(&self) -> gg::RenderInfo {
        gg::RenderInfo {
            col: Colour::red(),
            ..Default::default()
        }
    }
}
