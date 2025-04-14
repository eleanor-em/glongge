use crate::examples::mario::{
    AliveEnemyMap, BASE_GRAVITY, BLOCK_COLLISION_TAG, ENEMY_COLLISION_TAG, enemy::Stompable,
};
use crate::object_type::ObjectType;
use glongge::{core::prelude::*, resource::sprite::Sprite};
use glongge_derive::{partially_derive_scene_object, register_scene_object};
use num_traits::Zero;
use std::time::Duration;

#[register_scene_object]
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
    pub fn create(top_left: Vec2i) -> AnySceneObject<ObjectType> {
        AnySceneObject::new(Self {
            initial_coord: top_left,
            dead: false,
            started_death: false,
            top_left: top_left.into(),
            vel: Vec2 { x: -1., y: 0. },
            ..Default::default()
        })
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
impl SceneObject<ObjectType> for Goomba {
    fn on_load(
        &mut self,
        object_ctx: &mut ObjectContext<ObjectType>,
        resource_handler: &mut ResourceHandler,
    ) -> Result<Option<RenderItem>> {
        let texture = resource_handler
            .texture
            .wait_load_file("res/enemies_sheet.png")?;
        self.sprite = Sprite::from_tileset(
            object_ctx,
            resource_handler,
            texture.clone(),
            Vec2i { x: 2, y: 1 },
            Vec2i { x: 16, y: 16 },
            Vec2i { x: 0, y: 16 },
            Vec2i { x: 2, y: 0 },
        )
        .with_fixed_ms_per_frame(200);
        self.die_sprite = Sprite::from_single_extent(
            object_ctx,
            resource_handler,
            texture.clone(),
            Vec2i { x: 36, y: 16 },
            Vec2i { x: 16, y: 16 },
        )
        .with_hidden();
        object_ctx.transform_mut().centre = self.top_left + self.sprite.half_widths();
        Ok(None)
    }
    fn on_ready(&mut self, ctx: &mut UpdateContext<ObjectType>) {
        let mut data = ctx.scene_mut().data::<AliveEnemyMap>().unwrap();
        data.write().register(self.initial_coord);
        if !data.write().is_alive(self.initial_coord) {
            ctx.object_mut().remove_this();
        } else {
            ctx.object_mut().add_child(CollisionShape::from_collider(
                self.sprite.as_box_collider(),
                &self.emitting_tags(),
                &self.listening_tags(),
            ));
        }
    }
    fn on_fixed_update(&mut self, ctx: &mut FixedUpdateContext<ObjectType>) {
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
                self.v_accel = 0.;
            }
            self.vel.y += self.v_accel;
            ctx.object().transform_mut().centre += self.vel;
        }
    }
    fn on_collision(
        &mut self,
        ctx: &mut UpdateContext<ObjectType>,
        other: SceneObjectWithId<ObjectType>,
        mtv: Vec2,
    ) -> CollisionResponse {
        if !mtv.dot(Vec2::right()).is_zero() {
            self.vel.x = -self.vel.x;
        }
        if !mtv.dot(Vec2::up()).is_zero() {
            if self.vel.y.is_zero()
                && mtv.y < 0.
                && other.emitting_tags().contains(&BLOCK_COLLISION_TAG)
            {
                // Player pushed a block into the Goomba from below.
                self.stomp();
            } else {
                self.vel.y = 0.;
            }
        }
        if other.emitting_tags().contains(&BLOCK_COLLISION_TAG) {
            ctx.transform_mut().centre += mtv;
        }
        CollisionResponse::Done
    }
    fn on_update_end(&mut self, ctx: &mut UpdateContext<ObjectType>) {
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
                .unwrap()
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
