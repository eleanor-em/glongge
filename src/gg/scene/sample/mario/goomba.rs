use std::any::Any;
use std::time::Duration;
use anyhow::Result;
use num_traits::Zero;
use glongge_derive::{partially_derive_scene_object, register_scene_object};
use crate::core::collision::{BoxCollider, Collider};
use crate::core::linalg::{SquareExtent, Vec2, Vec2Int};
use crate::gg::{CollisionResponse, RenderableObject, RenderInfo, SceneObject, SceneObjectWithId, Transform, UpdateContext, VertexWithUV};
use crate::gg::scene::sample::mario::{BASE_GRAVITY, BRICK_COLLISION_TAG, ENEMY_COLLISION_TAG, ObjectType};
use crate::resource::ResourceHandler;
use crate::resource::sprite::Sprite;

#[register_scene_object]
pub struct Goomba {
    top_left: Vec2,
    vel: Vec2,
    v_accel: f64,
    sprite: Sprite,
}

impl Goomba {
    pub fn new(top_left: Vec2Int) -> Box<Self> {
        Box::new(Self {
            top_left: top_left.into(),
            sprite: Sprite::default(),
            vel: Vec2 { x: -0.5, y: 0. },
            v_accel: 0.,
        })
    }
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
        self.vel.y += self.v_accel;
        self.top_left += self.vel;
    }
    fn on_collision(&mut self, _other: SceneObjectWithId<ObjectType>, mtv: Vec2) -> CollisionResponse {
        self.top_left += mtv;
        if !mtv.dot(Vec2::right()).is_zero() {
            self.vel.x = -self.vel.x;
        }
        if !mtv.dot(Vec2::down()).is_zero() {
            self.vel.y = 0.;
        }
        CollisionResponse::Done
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
    fn collision_tags(&self) -> Vec<&'static str> {
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
        self.sprite.render_info_default()
    }
}
