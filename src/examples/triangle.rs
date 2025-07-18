use glongge::core::{
    prelude::*,
    scene::{Scene, SceneName},
};
use glongge_derive::partially_derive_scene_object;
use num_traits::FloatConst;
use rand::Rng;
use rand::distr::Distribution;
use rand::distr::Uniform;
use std::time::Instant;

#[allow(dead_code)]
#[derive(Copy, Clone)]
pub struct TriangleScene;
impl Scene for TriangleScene {
    fn name(&self) -> SceneName {
        SceneName::new("triangle")
    }

    fn create_objects(&self, _entrance_id: usize) -> Vec<SceneObjectWrapper> {
        const N: usize = 1;
        let mut rng = rand::rng();
        let xs: Vec<f32> = Uniform::new(0.0, 200.0)
            .unwrap()
            .sample_iter(&mut rng)
            .take(N)
            .collect();
        let ys: Vec<f32> = Uniform::new(0.0, 200.0)
            .unwrap()
            .sample_iter(&mut rng)
            .take(N)
            .collect();
        let vxs: Vec<f32> = Uniform::new(-1.0, 1.0)
            .unwrap()
            .sample_iter(&mut rng)
            .take(N)
            .collect();
        let vys: Vec<f32> = Uniform::new(-1.0, 1.0)
            .unwrap()
            .sample_iter(&mut rng)
            .take(N)
            .collect();
        (0..N)
            .map(|i| {
                let pos = Vec2 { x: xs[i], y: ys[i] };
                let vel = Vec2 {
                    x: vxs[i],
                    y: vys[i],
                };
                SpinningTriangle {
                    pos,
                    velocity: vel.normed(),
                    t: 0.0,
                    alive_since: Instant::now(),
                }
                .into_wrapper()
            })
            .collect()
    }
}

pub struct TriangleSpawner {}

pub struct SpinningTriangle {
    pos: Vec2,
    velocity: Vec2,
    t: f32,
    alive_since: Instant,
}

impl Default for SpinningTriangle {
    fn default() -> Self {
        Self {
            pos: Vec2::zero(),
            velocity: Vec2::zero(),
            t: 0.0,
            alive_since: Instant::now(),
        }
    }
}

impl SpinningTriangle {
    const TRI_WIDTH: f32 = 20.0;
    const VELOCITY: f32 = 20.0;
    const ANGULAR_VELOCITY: f32 = 0.1;

    pub fn new(pos: Vec2, vel_normed: Vec2) -> Self {
        Self {
            pos,
            velocity: vel_normed * Self::VELOCITY,
            t: 0.0,
            alive_since: Instant::now(),
        }
    }

    fn rotation(&self) -> f32 {
        Self::ANGULAR_VELOCITY * f32::PI() * self.t
    }
}
#[partially_derive_scene_object]
impl SceneObject for SpinningTriangle {
    fn on_load(
        &mut self,
        _object_ctx: &mut ObjectContext,
        _resource_handler: &mut ResourceHandler,
    ) -> Result<Option<RenderItem>> {
        let tri_height = SpinningTriangle::TRI_WIDTH * 3.0_f32.sqrt();
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
        Ok(Some(RenderItem::from_raw_vertices(vec![
            vertex1, vertex2, vertex3,
        ])))
    }
    fn on_update(&mut self, ctx: &mut UpdateContext) {
        if ctx.input().pressed(KeyCode::Space)
            && ctx.object().others().len() < 2500
            && ctx.viewport().contains_point(self.pos)
        {
            let mut rng = rand::rng();
            if rng.random_bool(0.2) {
                let vel = Vec2 {
                    x: rng.random_range(-1.0..1.0),
                    y: rng.random_range(-1.0..1.0),
                };
                ctx.object_mut().add_sibling(SpinningTriangle::new(
                    self.pos,
                    (self.velocity - vel).normed(),
                ));
                ctx.object_mut().add_sibling(SpinningTriangle::new(
                    self.pos,
                    (self.velocity + vel).normed(),
                ));
                ctx.object_mut().remove_this();
            }
        }

        self.t += Self::ANGULAR_VELOCITY * ctx.delta_60fps();
        let next_pos = self.pos + self.velocity * ctx.delta_60fps();
        if !ctx.viewport().contains_point(Vec2 {
            x: next_pos.x,
            y: self.pos.y,
        }) {
            self.velocity.x = -self.velocity.x;
        }
        if !ctx.viewport().contains_point(Vec2 {
            x: self.pos.x,
            y: next_pos.y,
        }) {
            self.velocity.y = -self.velocity.y;
        }
        self.pos += self.velocity * ctx.delta_60fps();
    }

    fn on_update_end(&mut self, ctx: &mut UpdateContext) {
        if self.alive_since.elapsed().as_secs_f32() > 0.1 && ctx.viewport().contains_point(self.pos)
        {
            for other in ctx.object().others() {
                let dist = ctx.object().absolute_transform_of(other).centre
                    - ctx.object().absolute_transform().centre;
                if dist.len() < Self::TRI_WIDTH {
                    self.velocity = -dist.normed() * Self::VELOCITY;
                }
            }
        }
        let mut transform = ctx.object().transform_mut();
        transform.centre = self.pos;
        transform.rotation = self.rotation();
    }

    fn as_renderable_object(&mut self) -> Option<&mut dyn RenderableObject> {
        Some(self)
    }
}

impl RenderableObject for SpinningTriangle {
    fn shader_execs(&self) -> Vec<ShaderExec> {
        vec![ShaderExec {
            blend_col: Colour::red(),
            ..Default::default()
        }]
    }
}
