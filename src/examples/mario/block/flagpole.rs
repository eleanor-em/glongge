use crate::examples::mario::FLAG_COLLISION_TAG;

use glongge::{core::prelude::*, resource::sprite::Sprite};
use glongge_derive::partially_derive_scene_object;

#[derive(Default)]
pub struct Flagpole {
    top_left: Vec2,
    sprite: Sprite,
}

impl Flagpole {
    pub fn new(top_left: Vec2i) -> Self {
        Self {
            top_left: top_left.into(),
            ..Default::default()
        }
    }
}

#[partially_derive_scene_object]
impl SceneObject for Flagpole {
    fn on_load(&mut self, ctx: &mut LoadContext) -> Result<Option<RenderItem>> {
        let texture = ctx
            .resource()
            .texture
            .wait_load_file("res/world_sheet.png")?;
        self.sprite = Sprite::add_from_single_coords(
            ctx,
            texture,
            Vec2i { x: 0, y: 588 },
            Vec2i { x: 16, y: 748 },
        );
        ctx.object().transform_mut().centre = self.top_left + self.sprite.half_widths();
        ctx.object_mut().add_child(CollisionShape::from_collider(
            self.sprite.as_box_collider(),
            &self.emitting_tags(),
            &self.listening_tags(),
        ));
        Ok(None)
    }

    fn emitting_tags(&self) -> Vec<&'static str> {
        [FLAG_COLLISION_TAG].into()
    }
}
