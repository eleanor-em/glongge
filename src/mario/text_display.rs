use glongge_derive::{partially_derive_scene_object, register_scene_object};
use glongge::{
    core::{
        prelude::*,
        util::linalg::{Vec2, Vec2Int},
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
pub struct TextDisplay {
    centre: Vec2,
    font: Option<Font>,
    sprite: Sprite,
}

impl TextDisplay {
    pub fn new(centre: Vec2Int) -> Box<Self> {
        Box::new(Self { centre: centre.into(), font: None, sprite: Sprite::default() })
    }
}

const SENTENCE: &str =
    "a set of words that is complete in itself, typically containing a subject and predicate, \
     conveying a statement, question, exclamation, or command, and consisting of a main \
     clause and sometimes one or more subordinate clauses.";

#[partially_derive_scene_object]
impl SceneObject<ObjectType> for TextDisplay {
    fn on_load(&mut self, resource_handler: &mut ResourceHandler) -> Result<RenderItem> {
        let font = Font::from_slice(include_bytes!("../../res/DejaVuSansMono.ttf"), 12.)?;
        self.sprite = Sprite::from_texture(font.render_texture(
            resource_handler,
            SENTENCE,
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

impl RenderableObject<ObjectType> for TextDisplay {
    fn render_info(&self) -> RenderInfo {
        self.sprite.render_info_default()
    }
}
