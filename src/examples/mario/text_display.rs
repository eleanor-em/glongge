use crate::object_type::ObjectType;
use glongge::{
    core::prelude::*,
    resource::{
        font::{Font, TextWrapMode},
        sprite::Sprite,
    },
};
use glongge_derive::{partially_derive_scene_object, register_scene_object};

#[register_scene_object]
pub struct WinTextDisplay {
    centre: Vec2,
    font: Option<Font>,
    sprite: Sprite,
}

impl WinTextDisplay {
    pub fn create(centre: Vec2) -> ConcreteSceneObject<ObjectType> {
        ConcreteSceneObject::new(Self {
            centre,
            ..Default::default()
        })
    }
}

#[partially_derive_scene_object]
impl SceneObject<ObjectType> for WinTextDisplay {
    fn on_load(
        &mut self,
        object_ctx: &mut ObjectContext<ObjectType>,
        resource_handler: &mut ResourceHandler,
    ) -> Result<Option<RenderItem>> {
        let font = Font::from_slice(include_bytes!("../../../res/DejaVuSansMono.ttf"), 20.)?;
        self.sprite = font.render_to_sprite(
            object_ctx,
            resource_handler,
            "You win!",
            200.0,
            TextWrapMode::WrapAnywhere,
        )?;
        self.font = Some(font);
        let mut transform = object_ctx.transform_mut();
        transform.centre = self.centre;
        transform.scale = Vec2::one() / self.font.as_ref().map_or_else(|| 1., |f| f.sample_ratio());
        Ok(None)
    }
}
