use crate::examples::mario::{
    AliveEnemyMap, BASE_GRAVITY, BLOCK_COLLISION_TAG, ENEMY_COLLISION_TAG, enemy::Stompable,
};

use glongge::{core::prelude::*, resource::sprite::Sprite};
use glongge_derive::partially_derive_scene_object;
use num_traits::Zero;
use std::time::Duration;

#[derive(Default)]
pub struct Goomba {
    initial_coord: Vec2i,
    dead: bool,
    started_death: bool,
    top_left: Vec2,
    vel: Vec2,
    v_accel: f32,
    sprite: Sprite,
    die_sprite: Sprite,
}

impl Goomba {
    pub fn create(top_left: Vec2i) -> SceneObjectWrapper {
        Self {
            initial_coord: top_left,
            dead: false,
            started_death: false,
            top_left: top_left.into(),
            vel: Vec2 { x: -0.8, y: 0.0 },
            ..Default::default()
        }
        .into_wrapper()
    }
}

impl Stompable for Goomba {
    fn stomp(&mut self) {
        self.dead = true;
    }
    fn dead(&self) -> bool {
        self.dead
    }
}

#[partially_derive_scene_object]
impl SceneObject for Goomba {
    fn on_load(&mut self, ctx: &mut LoadContext) -> Result<Option<RenderItem>> {
        let texture = ctx
            .resource()
            .texture
            .wait_load_file("res/enemies_sheet.png")?;
        self.sprite = Sprite::add_from_tileset(
            ctx,
            texture.clone(),
            Vec2i { x: 2, y: 1 },
            Vec2i { x: 16, y: 16 },
            Vec2i { x: 0, y: 16 },
            Vec2i { x: 2, y: 0 },
        )
        .with_fixed_ms_per_frame(200);
        self.die_sprite = Sprite::add_from_single_extent(
            ctx,
            texture.clone(),
            Vec2i { x: 36, y: 16 },
            Vec2i { x: 16, y: 16 },
        )
        .with_hidden();
        ctx.object().transform_mut().centre = self.top_left + self.sprite.half_widths();
        ctx.object_mut().add_child(CollisionShape::from_collider(
            self.sprite.as_box_collider(),
            &self.emitting_tags(),
            &self.listening_tags(),
        ));
        Ok(None)
    }
    fn on_ready(&mut self, ctx: &mut UpdateContext) {
        let mut data = ctx.scene_mut().data::<AliveEnemyMap>();
        data.write().register(self.initial_coord);
        if !data.write().is_alive(self.initial_coord) {
            ctx.object_mut().remove_this();
        }
    }

    fn on_update(&mut self, ctx: &mut UpdateContext) {
        let in_view = ctx.viewport().contains_point(self.top_left)
            || ctx
                .viewport()
                .contains_point(ctx.object().transform().centre + self.sprite.half_widths());
        if !self.dead && in_view {
            if ctx
                .object()
                .test_collision_offset(Vec2::down(), vec![BLOCK_COLLISION_TAG])
                .is_none()
            {
                self.v_accel = BASE_GRAVITY;
            } else {
                self.v_accel = 0.0;
            }
            ctx.object().transform_mut().centre += self.vel * ctx.delta_60fps();
        }
    }

    fn on_fixed_update(&mut self, _ctx: &mut FixedUpdateContext) {
        self.vel.y += self.v_accel;
    }

    fn on_collision(
        &mut self,
        ctx: &mut UpdateContext,
        other: &TreeSceneObject,
        mtv: Vec2,
    ) -> CollisionResponse {
        if !mtv.dot(Vec2::right()).is_zero() {
            self.vel.x = -self.vel.x;
        }
        if !mtv.dot(Vec2::up()).is_zero() {
            if self.vel.y.is_zero()
                && mtv.y < 0.0
                && other.emitting_tags().contains(&BLOCK_COLLISION_TAG)
            {
                // Player pushed a block into the Goomba from below.
                self.stomp();
            } else {
                self.vel.y = 0.0;
            }
        }
        if other.emitting_tags().contains(&BLOCK_COLLISION_TAG) {
            ctx.transform_mut().centre += mtv;
        }
        CollisionResponse::Done
    }
    fn on_update_end(&mut self, ctx: &mut UpdateContext) {
        if self.dead && !self.started_death {
            ctx.scene_mut().start_coroutine_after(
                |_this, ctx, _action| {
                    ctx.object_mut().remove_this();
                    CoroutineResponse::Complete
                },
                Duration::from_millis(300),
            );
            self.started_death = true;
            ctx.scene_mut()
                .data::<AliveEnemyMap>()
                .write()
                .set_dead(self.initial_coord);
            self.sprite.hide();
            self.die_sprite.show();
        }
    }

    fn emitting_tags(&self) -> Vec<&'static str> {
        [ENEMY_COLLISION_TAG].into()
    }
    fn listening_tags(&self) -> Vec<&'static str> {
        [ENEMY_COLLISION_TAG, BLOCK_COLLISION_TAG].into()
    }
}
