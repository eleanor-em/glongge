use glongge::util::canvas::Canvas;
use glongge::{
    core::{
        prelude::*,
        scene::{Scene, SceneName},
    },
    resource::sprite::Sprite,
};
use glongge_derive::partially_derive_scene_object;
use num_traits::FloatConst;
use rand::Rng;
use rand::distr::{Distribution, Uniform};
use std::time::Instant;

#[allow(dead_code)]
#[derive(Default, Copy, Clone)]
pub struct RectangleScene;
impl Scene for RectangleScene {
    fn name(&self) -> SceneName {
        SceneName::new("rectangle")
    }

    fn create_objects(&self, _entrance_id: usize) -> Vec<SceneObjectWrapper> {
        const N: usize = 10;
        let mut objects = Uniform::new(50.0, 350.0)
            .unwrap()
            .sample_iter(rand::rng())
            .zip(Uniform::new(50.0, 350.0).unwrap().sample_iter(rand::rng()))
            .zip(Uniform::new(-1.0, 1.0).unwrap().sample_iter(rand::rng()))
            .zip(Uniform::new(-1.0, 1.0).unwrap().sample_iter(rand::rng()))
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

#[derive(Default)]
pub struct RectanglePlayer {
    pos: Vec2,
    vel: Vec2,
    sprite: Sprite,
}

impl RectanglePlayer {
    const SPEED: f32 = 2.0;
}

#[partially_derive_scene_object]
impl SceneObject for RectanglePlayer {
    fn on_load(
        &mut self,
        object_ctx: &mut ObjectContext,
        resource_handler: &mut ResourceHandler,
    ) -> Result<Option<RenderItem>> {
        let texture = resource_handler.texture.wait_load_file("res/mario.png")?;
        self.sprite = Sprite::add_from_tileset(
            object_ctx,
            texture,
            Vec2i { x: 1, y: 1 },
            Vec2i { x: 16, y: 16 },
            Vec2i { x: 0, y: 0 },
            Vec2i { x: 2, y: 0 },
        )
        .with_fixed_ms_per_frame(100);
        Ok(None)
    }
    fn on_ready(&mut self, _ctx: &mut UpdateContext) {
        self.pos = Vec2 { x: 512.0, y: 384.0 };
    }

    fn on_update(&mut self, ctx: &mut UpdateContext) {
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

    fn on_update_end(&mut self, ctx: &mut UpdateContext) {
        ctx.object().transform_mut().centre = self.pos;
    }
    fn emitting_tags(&self) -> Vec<&'static str> {
        [RECTANGLE_COLL_TAG].into()
    }
}

#[derive(Default)]
pub struct SpinningRectangle {
    pos: Vec2,
    velocity: Vec2,
    t: f32,
    col: Colour,
    sprite: Sprite,
    alive_since: Option<Instant>,
}

impl SpinningRectangle {
    const VELOCITY: f32 = 2.0;
    // const ANGULAR_VELOCITY: f32 = 2.0;

    pub fn new(pos: Vec2, vel_normed: Vec2) -> Self {
        let mut rng = rand::rng();
        let col = match rng.random_range(0..6) {
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
            alive_since: Some(Instant::now()),
            ..Default::default()
        }
    }

    #[allow(clippy::unused_self)]
    fn rotation(&self) -> f32 {
        // let mut rv = Self::ANGULAR_VELOCITY * f32::PI() * self.t;
        // while rv > 2.0 * f32::PI() {
        //     rv -= 2.0 * f32::PI();
        // }
        // rv
        0.0
    }
}
#[partially_derive_scene_object]
impl SceneObject for SpinningRectangle {
    fn on_load(
        &mut self,
        object_ctx: &mut ObjectContext,
        resource_handler: &mut ResourceHandler,
    ) -> Result<Option<RenderItem>> {
        let texture = resource_handler.texture.wait_load_file("res/goomba.png")?;
        self.sprite = Sprite::add_from_tileset(
            object_ctx,
            texture,
            Vec2i { x: 1, y: 1 },
            Vec2i { x: 16, y: 16 },
            Vec2i { x: 0, y: 0 },
            Vec2i { x: 2, y: 0 },
        )
        .with_fixed_ms_per_frame(500)
        .with_blend_col(self.col);
        object_ctx.transform_mut().centre = self.pos;
        object_ctx.add_child(CollisionShape::from_object_sprite(self, &self.sprite));
        self.alive_since = Some(Instant::now());
        Ok(None)
    }
    fn on_update(&mut self, ctx: &mut UpdateContext) {
        if ctx.input().pressed(KeyCode::Space) {
            let mut rng = rand::rng();
            let angle = rng.random_range(0.0..(2.0 * f32::PI()));
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
        ctx: &mut UpdateContext,
        other: &TreeSceneObject,
        mtv: Vec2,
    ) -> CollisionResponse {
        if let Some(rect) = other.downcast_mut::<SpinningRectangle>()
            && self.alive_since.unwrap().elapsed().as_secs_f32() > 0.5
            && rect.alive_since.unwrap().elapsed().as_secs_f32() > 0.5
        {
            ctx.transform_mut().centre += mtv;
            self.velocity = self.velocity.reflect(mtv.normed());
            rect.sprite.set_blend_col(self.col);
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
