use num_traits::{Float, FloatConst};

use crate::{
    core::linalg::Vec2,
    gg::core::{RenderData, SafeObjectList, SceneObject}
};

#[derive(Clone)]
pub struct SpinningTriangle {
    pos: Vec2,
    velocity: Vec2,
    t: f64,
}

impl SpinningTriangle {
    const TRI_WIDTH: f64 = 5.0;
    const VELOCITY: f64 = 200.0;
    const ANGULAR_VELOCITY: f64 = 1.0;

    pub fn new(pos: Vec2, vel_normed: Vec2) -> Self {
        Self { pos, velocity: vel_normed * Self::VELOCITY, t: 0.0 }
    }
}
impl SceneObject for SpinningTriangle {
    fn create_vertices(&self) -> Vec<Vec2> {
        let tri_height = Self::TRI_WIDTH * 3.0.sqrt();
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

    fn on_update(&mut self, delta: f64, others: SafeObjectList) -> RenderData {
        self.t += delta;
        let next_pos = self.pos + self.velocity * delta;
        if !(0.0..1024.0).contains(&next_pos.x) {
            self.velocity.x = -self.velocity.x;
        }
        if !(0.0..768.0).contains(&next_pos.y) {
            self.velocity.y = -self.velocity.y;
        }
        for other in others {
            if (other.world_pos() - self.pos).mag() < Self::TRI_WIDTH {
                self.velocity = (self.pos - other.world_pos()).normed() * Self::VELOCITY;
            }
        }
        self.pos += self.velocity * delta;
        RenderData {
            position: self.pos,
            rotation: Self::ANGULAR_VELOCITY * f64::PI() * self.t,
        }
    }

    fn world_pos(&self) -> Vec2 { self.pos }
}
