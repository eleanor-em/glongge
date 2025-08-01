use crate::examples::mario::BLOCK_COLLISION_TAG;

use glongge::{core::prelude::*, resource::sprite::Sprite};
use glongge_derive::partially_derive_scene_object;

#[derive(Default)]
pub struct Block {
    top_left: Vec2,
    sprite: Sprite,

    initial_y: f32,
    v_speed: f32,
    v_accel: f32,
}

impl Block {
    pub fn new(top_left: Vec2i) -> Self {
        Self {
            top_left: top_left.into(),
            initial_y: top_left.y as f32,
            ..Default::default()
        }
    }
}

#[partially_derive_scene_object]
impl SceneObject for Block {
    fn on_load(
        &mut self,
        object_ctx: &mut ObjectContext,
        resource_handler: &mut ResourceHandler,
    ) -> Result<Option<RenderItem>> {
        let texture = resource_handler
            .texture
            .wait_load_file("res/world_sheet.png")?;
        self.sprite = Sprite::add_from_single_extent(
            object_ctx,
            texture,
            Vec2i { x: 48, y: 476 },
            Vec2i { x: 16, y: 16 },
        );
        object_ctx.add_child(CollisionShape::from_collider(
            self.sprite.as_box_collider(),
            &self.emitting_tags(),
            &self.listening_tags(),
        ));
        Ok(None)
    }

    fn on_update_begin(&mut self, ctx: &mut UpdateContext) {
        self.top_left.y += self.v_speed * ctx.delta_60fps();
        if self.top_left.y > self.initial_y {
            self.top_left.y = self.initial_y;
            self.v_speed = 0.0;
            self.v_accel = 0.0;
        }
    }

    fn on_fixed_update(&mut self, _ctx: &mut FixedUpdateContext) {
        self.v_speed += self.v_accel;
    }
    fn on_update_end(&mut self, ctx: &mut UpdateContext) {
        ctx.object().transform_mut().centre = self.top_left + self.sprite.half_widths();
    }
    fn emitting_tags(&self) -> Vec<&'static str> {
        [BLOCK_COLLISION_TAG].into()
    }
}
