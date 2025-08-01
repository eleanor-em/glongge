use glongge::resource::font::FontRenderSettings;
use glongge::{
    core::prelude::*,
    font_from_file, include_bytes_root,
    resource::{font::Font, sprite::Sprite},
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
        let font = font_from_file!("res/DejaVuSansMono.ttf", 20.0)?;
        self.sprite = font
            .layout(
                "You win!",
                FontRenderSettings {
                    max_width: 200.0,
                    ..FontRenderSettings::default()
                },
            )
            .render_to_sprite(object_ctx)
            .unwrap();
        self.font = Some(font);
        let mut transform = object_ctx.transform_mut();
        transform.centre = self.centre;
        transform.scale = Vec2::one() / self.font.as_ref().map_or_else(|| 1.0, Font::sample_ratio);
        Ok(None)
    }
}
