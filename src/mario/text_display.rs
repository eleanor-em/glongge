use glongge_derive::{partially_derive_scene_object, register_scene_object};
use glongge::{
    core::{
        prelude::*,
        util::linalg::Vec2,
        util::linalg::Transform,
    },
    resource::{
        ResourceHandler,
        sprite::Sprite
    }
};
use glongge::core::render::{RenderInfo, RenderItem, VertexDepth};
use glongge::core::scene::{RenderableObject, SceneObject};
use glongge::resource::font::{Font, TextWrapMode};
use crate::mario::ObjectType;

#[register_scene_object]
pub struct WinTextDisplay {
    centre: Vec2,
    font: Option<Font>,
    sprite: Sprite,
}

impl WinTextDisplay {
    pub fn new(centre: Vec2) -> Box<Self> {
        Box::new(Self { centre: centre, font: None, sprite: Sprite::default() })
    }
}

#[partially_derive_scene_object]
impl SceneObject<ObjectType> for WinTextDisplay {
    fn on_load(&mut self, resource_handler: &mut ResourceHandler) -> Result<RenderItem> {
        let font = Font::from_slice(include_bytes!("../../res/DejaVuSansMono.ttf"), 20.)?;
        self.sprite = Sprite::from_texture(font.render_texture(
            resource_handler,
            "You win!",
            200.0,
            TextWrapMode::WrapAnywhere
        )?);
        self.font = Some(font);
        Ok(self.sprite.create_vertices().with_depth(VertexDepth::Back(0)))
    }
    fn transform(&self) -> Transform {
        Transform {
            centre: self.centre,
            scale: Vec2::one() / self.font.as_ref().map_or_else(|| 1., |f| f.sample_ratio()),
            ..Default::default()
        }
    }
    fn as_renderable_object(&self) -> Option<&dyn RenderableObject<ObjectType>> {
        Some(self)
    }
}

impl RenderableObject<ObjectType> for WinTextDisplay {
    fn render_info(&self) -> RenderInfo {
        self.sprite.render_info_default()
    }
}
