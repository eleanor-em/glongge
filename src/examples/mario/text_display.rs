use glongge::resource::font::FontRenderSettings;
use glongge::{core::prelude::*, font_from_file, include_bytes_root, resource::sprite::Sprite};
use glongge_derive::partially_derive_scene_object;

#[derive(Default)]
pub struct WinTextDisplay {
    centre: Vec2,
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
    fn on_load(&mut self, ctx: &mut LoadContext) -> Result<Option<RenderItem>> {
        let font = font_from_file!("res/DejaVuSansMono.ttf", 20.0)?;
        self.sprite = font
            .layout(
                "You win!",
                FontRenderSettings {
                    max_width: 200.0,
                    ..FontRenderSettings::default()
                },
            )
            .render_to_sprite(ctx.object_mut())
            .unwrap();
        let mut transform = ctx.object().transform_mut();
        transform.centre = self.centre;
        transform.scale = Vec2::one() / font.sample_ratio();
        Ok(None)
    }
}
