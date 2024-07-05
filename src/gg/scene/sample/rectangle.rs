use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant}
};
use num_traits::{FloatConst, Zero};
use rand::{
    distributions::{Distribution, Uniform},
    Rng
};

use crate::{
    core::{
        linalg::Vec2,
        input::{InputHandler, KeyCode}
    },
    gg::{
        self,
        sample::BasicRenderHandler,
        scene::Scene,
        UpdateContext
    },
    shader,
};
use crate::gg::{RenderableObject, WorldObject};

pub fn create_scene(
    render_handler: &BasicRenderHandler,
    input_handler: Arc<Mutex<InputHandler>>
) -> Scene<BasicRenderHandler> {
    Scene::new(vec![Box::new(Spawner {}),
                    Box::new(Player { pos: Vec2 { x: 512.0, y: 384.0 } })],
               render_handler, input_handler)
}

struct Player {
    pos: Vec2,
}

impl Player {
    const SIZE: f64 = 190.0;
    const SPEED: f64 = 170.0;
}

impl gg::SceneObject for Player {
    fn on_ready(&mut self) {}
    fn get_name(&self) -> &'static str { "Player" }

    fn on_update(&mut self, delta: Duration, update_ctx: UpdateContext) {
        let mut direction = Vec2::zero();
        if update_ctx.input().down(KeyCode::Left) { direction += Vec2::left(); }
        if update_ctx.input().down(KeyCode::Right) { direction += Vec2::right(); }
        if update_ctx.input().down(KeyCode::Up) { direction += Vec2::up(); }
        if update_ctx.input().down(KeyCode::Down) { direction += Vec2::down(); }
        self.pos += Self::SPEED * direction.normed() * delta.as_secs_f64();
    }

    fn as_world_object(&self) -> Option<&dyn WorldObject> {
        Some(self)
    }
    fn as_renderable_object(&self) -> Option<&dyn RenderableObject> {
        Some(self)
    }
}

impl gg::WorldObject for Player {
    fn world_pos(&self) -> Vec2 {
        self.pos
    }
}

impl gg::RenderableObject for Player {
    fn create_vertices(&self) -> Vec<Vec2> {
        shader::vertex::rectangle(-Self::SIZE/2.0 * Vec2::one(),
                                  Self::SIZE * Vec2::one())
    }

    fn render_data(&self) -> gg::RenderData {
        gg::RenderData {
            position: self.pos,
            rotation: 0.0,
            colour: [0.0, 1.0, 0.0, 1.0],
        }
    }
}

struct Spawner {}

impl gg::SceneObject for Spawner {
    fn on_ready(&mut self) {}
    fn get_name(&self) -> &'static str { "Spawner" }

    fn on_update(&mut self, _delta: Duration, mut update_ctx: gg::UpdateContext) {
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
                Box::new(SpinningRectangle::new(pos, vel.normed())) as Box<dyn gg::SceneObject>
            })
            .collect();
        update_ctx.add_object_vec(objects);
        update_ctx.remove_this_object();
    }
}

struct SpinningRectangle {
    pos: Vec2,
    velocity: Vec2,
    t: f64,
    alive_since: Instant,
}

impl SpinningRectangle {
    const SIZE: f64 = 24.0;
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
}
impl gg::SceneObject for SpinningRectangle {
    fn on_ready(&mut self) {}
    fn get_name(&self) -> &'static str { "SpinningRectangle" }

    fn on_update(&mut self, delta: Duration, mut update_ctx: gg::UpdateContext) {
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

    fn on_update_end(&mut self, _delta: Duration, update_ctx: UpdateContext) {
        if self.alive_since.elapsed().as_secs_f64() > 0.1 &&
                update_ctx.viewport().contains(self.pos) {
            for other in update_ctx.others() {
                if let Some(other) = other.as_world_object() {
                    let other_size = if other.get_name() == "Player" { Player::SIZE } else { Self::SIZE };
                    if (other.world_pos() - self.pos).mag() < (Self::SIZE + other_size) / 2.0 {
                        self.velocity = (self.pos - other.world_pos()).normed() * Self::VELOCITY;
                    }
                }
            }
        }
    }

    fn as_world_object(&self) -> Option<&dyn gg::WorldObject> {
        Some(self)
    }
    fn as_renderable_object(&self) -> Option<&dyn gg::RenderableObject> {
        Some(self)
    }
}

impl gg::WorldObject for SpinningRectangle {
    fn world_pos(&self) -> Vec2 {
        self.pos
    }
}

impl gg::RenderableObject for SpinningRectangle {
    fn create_vertices(&self) -> Vec<Vec2> {
        shader::vertex::rectangle(-Self::SIZE/2.0 * Vec2::one(),
                                  Self::SIZE * Vec2::one())
    }

    fn render_data(&self) -> gg::RenderData {
        gg::RenderData {
            position: self.pos,
            rotation: Self::ANGULAR_VELOCITY * f64::PI() * self.t,
            colour: [1.0, 0.0, 0.0, 1.0],
        }
    }
}
