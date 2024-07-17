use std::any::Any;
#[allow(unused_imports)]
use crate::core::prelude::*;

use std::time::Duration;
use num_traits::Zero;
use glongge_derive::{partially_derive_scene_object, register_scene_object};
use crate::core::collision::{BoxCollider, Collider};
use crate::core::linalg::{SquareExtent, Vec2, Vec2Int};
use crate::gg::{CollisionResponse, RenderableObject, RenderInfo, SceneObject, SceneObjectWithId, Transform, UpdateContext, VertexWithUV};
use crate::gg::coroutine::CoroutineResponse;
use crate::gg::scene::sample::mario::{BASE_GRAVITY, BRICK_COLLISION_TAG, ENEMY_COLLISION_TAG, ObjectType};
use crate::resource::ResourceHandler;
use crate::resource::sprite::Sprite;

#[register_scene_object]
pub struct Goomba {
    dead: bool,
    started_death: bool,
    top_left: Vec2,
    vel: Vec2,
    v_accel: f64,
    sprite: Sprite,
    die_sprite: Sprite,
}

impl Goomba {
    pub fn new(top_left: Vec2Int) -> Box<Self> {
        Box::new(Self {
            dead: false,
            started_death: false,
            top_left: top_left.into(),
            sprite: Sprite::default(),
            die_sprite: Sprite::default(),
            vel: Vec2 { x: -0.5, y: 0. },
            v_accel: 0.,
        })
    }

    pub fn die(&mut self) { self.dead = true; }

    pub fn dead(&self) -> bool { self.dead }
}

#[partially_derive_scene_object]
impl SceneObject<ObjectType> for Goomba {
    fn on_load(&mut self, resource_handler: &mut ResourceHandler) -> Result<()> {
        let texture_id = resource_handler.texture.wait_load_file("res/enemies_sheet.png".to_string())?;
        self.sprite = Sprite::from_tileset(
            texture_id,
            Vec2Int { x: 2, y: 1},
            Vec2Int { x: 16, y: 16 },
            Vec2Int { x: 0, y: 16 },
            Vec2Int { x: 2, y: 0 }
        ).with_fixed_ms_per_frame(200);
        self.die_sprite = Sprite::from_single(
            texture_id,
            Vec2Int { x: 16, y: 16 },
            Vec2Int { x: 36, y: 16 }
        );
        Ok(())
    }
    fn on_ready(&mut self) {}
    fn on_update(&mut self, _delta: Duration, update_ctx: UpdateContext<ObjectType>) {
        self.v_accel = 0.;
        let ray = self.collider().translated(2 * Vec2::down());
        if update_ctx.test_collision(ray.as_ref(), vec![BRICK_COLLISION_TAG]).is_none() {
            self.v_accel = BASE_GRAVITY;
        }
    }
    fn on_fixed_update(&mut self, _update_ctx: UpdateContext<ObjectType>) {
        if !self.dead {
            self.vel.y += self.v_accel;
            self.top_left += self.vel;
        }
    }
    fn on_collision(&mut self, _other: SceneObjectWithId<ObjectType>, mtv: Vec2) -> CollisionResponse {
        if !mtv.dot(Vec2::right()).is_zero() {
            self.vel.x = -self.vel.x;
            self.top_left += mtv;
        }
        if !mtv.dot(Vec2::up()).is_zero() {
            self.vel.y = 0.;
            self.top_left += mtv;
        }
        CollisionResponse::Done
    }
    fn on_update_end(&mut self, _delta: Duration, mut update_ctx: UpdateContext<ObjectType>) {
        if self.dead && !self.started_death {
            update_ctx.add_coroutine_after(|_this, update_ctx, _action| {
                update_ctx.remove_this_object();
                CoroutineResponse::Complete
            }, Duration::from_millis(300));
            self.started_death = true;
        }
    }

    fn transform(&self) -> Transform {
        Transform {
            centre: self.top_left + self.sprite.half_widths(),
            ..Default::default()
        }
    }
    fn as_renderable_object(&self) -> Option<&dyn RenderableObject<ObjectType>> {
        Some(self)
    }
    fn collider(&self) -> Box<dyn Collider> {
        Box::new(BoxCollider::from_transform(self.transform(), self.sprite.half_widths()))
    }
    fn emitting_tags(&self) -> Vec<&'static str> {
        [ENEMY_COLLISION_TAG].into()
    }
    fn listening_tags(&self) -> Vec<&'static str> {
        [BRICK_COLLISION_TAG].into()
    }
}

impl RenderableObject<ObjectType> for Goomba {
    fn create_vertices(&self) -> Vec<VertexWithUV> {
        self.sprite.create_vertices()
    }

    fn render_info(&self) -> RenderInfo {
        if self.dead {
            self.die_sprite.render_info_default()
        } else {
            self.sprite.render_info_default()
        }
    }
}
