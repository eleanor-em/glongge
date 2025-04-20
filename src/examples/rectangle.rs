use crate::object_type::ObjectType;
use glongge::util::canvas::Canvas;
use glongge::{
    core::{
        prelude::*,
        scene::{Scene, SceneName},
    },
    resource::sprite::Sprite,
};
use glongge_derive::*;
use num_traits::{FloatConst, Zero};
use rand::{
    Rng,
    distributions::{Distribution, Uniform},
};
use std::time::Instant;

#[allow(dead_code)]
#[derive(Copy, Clone)]
pub struct RectangleScene;
impl Scene<ObjectType> for RectangleScene {
    fn name(&self) -> SceneName {
        SceneName::new("rectangle")
    }

    fn create_objects(&self, _entrance_id: usize) -> Vec<SceneObjectWrapper<ObjectType>> {
        const N: usize = 10;
        let mut objects = Uniform::new(50., 350.)
            .sample_iter(rand::thread_rng())
            .zip(Uniform::new(50., 350.).sample_iter(rand::thread_rng()))
            .zip(Uniform::new(-1., 1.).sample_iter(rand::thread_rng()))
            .zip(Uniform::new(-1., 1.).sample_iter(rand::thread_rng()))
            .take(N)
            .map(|(((x, y), vx), vy)| {
                let pos = Vec2 { x, y };
                let vel = Vec2 { x: vx, y: vy };
                SpinningRectangle::new(pos, vel.normed()).into_wrapper()
            })
            .collect_vec();
        objects.push(RectanglePlayer::default().into_wrapper());
        objects.push(Canvas::default().into_wrapper());
        objects
    }
}

const RECTANGLE_COLL_TAG: &str = "RECTANGLE_COLL_TAG";

#[register_scene_object]
pub struct RectanglePlayer {
    pos: Vec2,
    vel: Vec2,
    sprite: Sprite,
}

impl RectanglePlayer {
    const SPEED: f32 = 2.;
}

#[partially_derive_scene_object]
impl SceneObject<ObjectType> for RectanglePlayer {
    fn on_load(
        &mut self,
        object_ctx: &mut ObjectContext<ObjectType>,
        resource_handler: &mut ResourceHandler,
    ) -> Result<Option<RenderItem>> {
        let texture = resource_handler.texture.wait_load_file("res/mario.png")?;
        self.sprite = Sprite::add_from_tileset(
            object_ctx,
            resource_handler,
            texture,
            Vec2i { x: 1, y: 1 },
            Vec2i { x: 16, y: 16 },
            Vec2i { x: 0, y: 0 },
            Vec2i { x: 2, y: 0 },
        )
        .with_fixed_ms_per_frame(100);
        Ok(None)
    }
    fn on_ready(&mut self, _ctx: &mut UpdateContext<ObjectType>) {
        self.pos = Vec2 { x: 512., y: 384. };
    }

    fn on_update(&mut self, ctx: &mut UpdateContext<ObjectType>) {
        let mut direction = Vec2::zero();
        if ctx.input().down(KeyCode::ArrowLeft) {
            direction += Vec2::left();
        }
        if ctx.input().down(KeyCode::ArrowRight) {
            direction += Vec2::right();
        }
        if ctx.input().down(KeyCode::ArrowUp) {
            direction += Vec2::up();
        }
        if ctx.input().down(KeyCode::ArrowDown) {
            direction += Vec2::down();
        }
        self.vel = Self::SPEED * direction.normed();
        self.pos += self.vel * ctx.delta_60fps();
    }

    fn on_update_end(&mut self, ctx: &mut UpdateContext<ObjectType>) {
        ctx.object().transform_mut().centre = self.pos;
    }
    fn emitting_tags(&self) -> Vec<&'static str> {
        [RECTANGLE_COLL_TAG].into()
    }
}

#[register_scene_object]
pub struct SpinningRectangle {
    pos: Vec2,
    velocity: Vec2,
    t: f32,
    col: Colour,
    sprite: Sprite,
    alive_since: Instant,
}

impl SpinningRectangle {
    const VELOCITY: f32 = 2.;
    // const ANGULAR_VELOCITY: f32 = 2.;

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
            col,
            alive_since: Instant::now(),
            ..Default::default()
        }
    }

    fn rotation(&self) -> f32 {
        // let mut rv = Self::ANGULAR_VELOCITY * f32::PI() * self.t;
        // while rv > 2. * f32::PI() {
        //     rv -= 2. * f32::PI();
        // }
        // rv
        0.
    }
}
#[partially_derive_scene_object]
impl SceneObject<ObjectType> for SpinningRectangle {
    fn on_load(
        &mut self,
        object_ctx: &mut ObjectContext<ObjectType>,
        resource_handler: &mut ResourceHandler,
    ) -> Result<Option<RenderItem>> {
        let texture = resource_handler.texture.wait_load_file("res/goomba.png")?;
        self.sprite = Sprite::add_from_tileset(
            object_ctx,
            resource_handler,
            texture,
            Vec2i { x: 1, y: 1 },
            Vec2i { x: 16, y: 16 },
            Vec2i { x: 0, y: 0 },
            Vec2i { x: 2, y: 0 },
        )
        .with_fixed_ms_per_frame(500)
        .with_blend_col(self.col);
        object_ctx.transform_mut().centre = self.pos;
        Ok(None)
    }
    fn on_ready(&mut self, ctx: &mut UpdateContext<ObjectType>) {
        ctx.object_mut()
            .add_child(CollisionShape::from_object_sprite(self, &self.sprite));
    }
    fn on_update(&mut self, ctx: &mut UpdateContext<ObjectType>) {
        if ctx.input().pressed(KeyCode::Space) {
            let mut rng = rand::thread_rng();
            let angle = rng.gen_range(0.0..(2. * f32::PI()));
            ctx.object_mut()
                .add_sibling(SpinningRectangle::new(self.pos, Vec2::one().rotated(angle)));
            self.velocity = -Self::VELOCITY * Vec2::one().rotated(angle);
        }

        let next_pos = ctx.object().transform().centre + self.velocity;
        if !(0.0..ctx.viewport().right()).contains(&next_pos.x) {
            self.velocity.x = -self.velocity.x;
        }
        if !(0.0..ctx.viewport().bottom()).contains(&next_pos.y) {
            self.velocity.y = -self.velocity.y;
        }
        self.t += 0.01 * ctx.delta_60fps();
        let mut transform = ctx.object().transform_mut();
        transform.centre += self.velocity * ctx.delta_60fps();
        transform.rotation = self.rotation();
    }

    fn on_collision(
        &mut self,
        ctx: &mut UpdateContext<ObjectType>,
        other: TreeSceneObject<ObjectType>,
        mtv: Vec2,
    ) -> CollisionResponse {
        if let Some(rect) = other.downcast_mut::<SpinningRectangle>() {
            if self.alive_since.elapsed().as_secs_f32() > 0.5
                && rect.alive_since.elapsed().as_secs_f32() > 0.5
            {
                ctx.transform_mut().centre += mtv;
                self.velocity = self.velocity.reflect(mtv.normed());
                rect.sprite.set_blend_col(self.col);
            }
        }
        CollisionResponse::Done
    }

    fn emitting_tags(&self) -> Vec<&'static str> {
        [RECTANGLE_COLL_TAG].into()
    }
    fn listening_tags(&self) -> Vec<&'static str> {
        [RECTANGLE_COLL_TAG].into()
    }
}
