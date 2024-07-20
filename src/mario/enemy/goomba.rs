use std::time::Duration;
use num_traits::Zero;
use glongge_derive::{partially_derive_scene_object, register_scene_object};
use glongge::{
    core::{
        linalg::{AxisAlignedExtent, Vec2, Vec2Int},
        CollisionResponse,
        RenderableObject,
        RenderInfo,
        SceneObject,
        SceneObjectWithId,
        Transform,
        UpdateContext,
        VertexWithUV,
        collision::Collider,
        prelude::*,
        coroutine::CoroutineResponse,
    },
    resource::{
        ResourceHandler,
        sprite::Sprite
    }
};
use crate::mario::{
    BASE_GRAVITY,
    BLOCK_COLLISION_TAG,
    ENEMY_COLLISION_TAG,
    ObjectType,
    enemy::Stompable
};

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

}

impl Stompable for Goomba {
    fn stomp(&mut self) { self.dead = true; }
    fn dead(&self) -> bool { self.dead }
}

#[partially_derive_scene_object]
impl SceneObject<ObjectType> for Goomba {
    fn on_load(&mut self, resource_handler: &mut ResourceHandler) -> Result<Vec<VertexWithUV>> {
        let texture_id = resource_handler.texture.wait_load_file("res/enemies_sheet.png".to_string())?;
        self.sprite = Sprite::from_tileset(
            texture_id,
            Vec2Int { x: 2, y: 1},
            Vec2Int { x: 16, y: 16 },
            Vec2Int { x: 0, y: 16 },
            Vec2Int { x: 2, y: 0 }
        ).with_fixed_ms_per_frame(200);
        self.die_sprite = Sprite::from_single_extent(
            texture_id,
            Vec2Int { x: 16, y: 16 },
            Vec2Int { x: 36, y: 16 }
        );
        Ok(self.sprite.create_vertices())
    }
    fn on_update(&mut self, _delta: Duration, ctx: &mut UpdateContext<ObjectType>) {
        self.v_accel = 0.;
        if ctx.object().test_collision_along(self.collider(), vec![BLOCK_COLLISION_TAG], Vec2::down(), 1.).is_none() {
            self.v_accel = BASE_GRAVITY;
        }
    }
    fn on_fixed_update(&mut self, ctx: &mut UpdateContext<ObjectType>) {
        self.sprite.fixed_update();
        let in_view = ctx.viewport().contains_point(self.top_left) ||
            ctx.viewport().contains_point(self.top_left + self.sprite.aa_extent());
        if !self.dead && in_view {
            self.vel.y += self.v_accel;
            self.top_left += self.vel;
        }
    }
    fn on_collision(&mut self, _ctx: &mut UpdateContext<ObjectType>, other: SceneObjectWithId<ObjectType>, mtv: Vec2) -> CollisionResponse {
        if !mtv.dot(Vec2::right()).is_zero() {
            self.vel.x = -self.vel.x;
        }
        if !mtv.dot(Vec2::up()).is_zero() {
            if self.vel.y.is_zero() && mtv.y < 0. && other.emitting_tags().contains(&BLOCK_COLLISION_TAG) {
                self.stomp();
            }
            self.vel.y = 0.;
        }
        if other.emitting_tags().contains(&BLOCK_COLLISION_TAG) {
            self.top_left += mtv;
        }
        CollisionResponse::Done
    }
    fn on_update_end(&mut self, _delta: Duration, ctx: &mut UpdateContext<ObjectType>) {
        if self.dead && !self.started_death {
            ctx.scene().start_coroutine_after(|_this, update_ctx, _action| {
                update_ctx.object().remove_this();
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
        self.sprite.as_box_collider(self.transform())
    }
    fn emitting_tags(&self) -> Vec<&'static str> {
        [ENEMY_COLLISION_TAG].into()
    }
    fn listening_tags(&self) -> Vec<&'static str> {
        [ENEMY_COLLISION_TAG, BLOCK_COLLISION_TAG].into()
    }
}

impl RenderableObject<ObjectType> for Goomba {
    fn render_info(&self) -> RenderInfo {
        if self.dead {
            self.die_sprite.render_info_default()
        } else {
            self.sprite.render_info_default()
        }
    }
}
