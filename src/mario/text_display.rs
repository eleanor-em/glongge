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
use glongge::resource::font;
use glongge::resource::font::TextWrapMode;
use crate::mario::ObjectType;

#[register_scene_object]
pub struct TextDisplay {
    centre: Vec2,
    sprite: Sprite,
}

impl TextDisplay {
    pub fn new(centre: Vec2Int) -> Box<Self> {
        Box::new(Self { centre: centre.into(), sprite: Sprite::default() })
    }
}

const SENTENCE: &str =
    "a set of words that is complete in itself, typically containing a subject and predicate, \
     conveying a statement, question, exclamation, or command, and consisting of a main \
     clause and sometimes one or more subordinate clauses.";

const SAMPLE_RATIO: u32 = 8;

#[partially_derive_scene_object]
impl SceneObject<ObjectType> for TextDisplay {
    fn on_load(&mut self, resource_handler: &mut ResourceHandler) -> Result<RenderItem> {
        self.sprite = Sprite::from_texture(font::create_bitmap(
            resource_handler,
            SENTENCE,
            12.0,
            200.0,
            TextWrapMode::WrapAnywhere,
            SAMPLE_RATIO
        )?);
        Ok(self.sprite.create_vertices().with_depth(VertexDepth::Back(0)))
    }
    fn transform(&self) -> Transform {
        Transform {
            centre: self.centre,
            scale: Vec2::one() / SAMPLE_RATIO,
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
