use crate::examples::mario::BLOCK_COLLISION_TAG;

use glongge::{core::prelude::*, resource::sprite::Sprite};
use glongge_derive::partially_derive_scene_object;

#[derive(Default)]
pub struct DecorativePipe {
    top_left: Vec2,
    sprite: Sprite,
}

impl DecorativePipe {
    pub fn new(top_left: Vec2i) -> Self {
        Self {
            top_left: top_left.into(),
            ..Default::default()
        }
    }
}

#[partially_derive_scene_object]
impl SceneObject for DecorativePipe {
    fn on_load(
        &mut self,
        object_ctx: &mut ObjectContext,
        resource_handler: &mut ResourceHandler,
    ) -> Result<Option<RenderItem>> {
        let texture = resource_handler
            .texture
            .wait_load_file("res/world_sheet.png")?;
        self.sprite = Sprite::add_from_single_coords(
            object_ctx,
            resource_handler,
            texture,
            Vec2i { x: 224, y: 324 },
            Vec2i { x: 256, y: 676 },
        );
        object_ctx.transform_mut().centre = self.top_left + self.sprite.half_widths();
        object_ctx.add_child(CollisionShape::from_collider(
            self.sprite.as_box_collider(),
            &self.emitting_tags(),
            &self.listening_tags(),
        ));
        Ok(None)
    }

    fn emitting_tags(&self) -> Vec<&'static str> {
        [BLOCK_COLLISION_TAG].into()
    }
}
