use glongge::{
    core::prelude::*,
    include_bytes_root,
    resource::{
        font::{Font, TextWrapMode},
        sprite::Sprite,
    },
};
use glongge_derive::partially_derive_scene_object;

#[derive(Default)]
pub struct WinTextDisplay {
    centre: Vec2,
    font: Option<Font>,
    sprite: Sprite,
}

impl WinTextDisplay {
    pub fn new(centre: Vec2) -> Self {
        Self {
            centre,
            ..Default::default()
        }
    }
}

#[partially_derive_scene_object]
impl SceneObject for WinTextDisplay {
    fn on_load(
        &mut self,
        object_ctx: &mut ObjectContext,
        _resource_handler: &mut ResourceHandler,
    ) -> Result<Option<RenderItem>> {
        let font = Font::from_slice(include_bytes_root!("res/DejaVuSansMono.ttf"), 20.0)?;
        self.sprite =
            font.render_to_sprite(object_ctx, "You win!", 200.0, TextWrapMode::WrapAnywhere)?;
        self.font = Some(font);
        let mut transform = object_ctx.transform_mut();
        transform.centre = self.centre;
        transform.scale = Vec2::one() / self.font.as_ref().map_or_else(|| 1.0, Font::sample_ratio);
        Ok(None)
    }
}
