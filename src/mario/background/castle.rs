use glongge_derive::{partially_derive_scene_object, register_scene_object};
use glongge::{
    core::{
        RenderableObject,
        RenderInfo,
        SceneObject,
        Transform,
        prelude::*,
        linalg::{AxisAlignedExtent, Vec2, Vec2Int},
    },
    resource::{
        ResourceHandler,
        sprite::Sprite
    }
};
use glongge::core::{RenderItem, VertexDepth};
use crate::mario::ObjectType;

#[register_scene_object]
pub struct Castle {
    top_left: Vec2,
    sprite: Sprite,
}

impl Castle {
    pub fn new(top_left: Vec2Int) -> Box<Self> {
        Box::new(Self { top_left: top_left.into(), sprite: Sprite::default() })
    }
}

#[partially_derive_scene_object]
impl SceneObject<ObjectType> for Castle {
    fn on_load(&mut self, resource_handler: &mut ResourceHandler) -> Result<RenderItem> {
        let texture_id = resource_handler.texture.wait_load_file("res/world_sheet.png".to_string())?;
        self.sprite = Sprite::from_single_coords(
            texture_id,
            Vec2Int { x: 24, y: 684 },
            Vec2Int { x: 104, y: 764 }
        );
        Ok(self.sprite.create_vertices().with_depth(VertexDepth::Back(0)))
    }
    fn transform(&self) -> Transform {
        Transform {
            centre: self.top_left + self.sprite.half_widths(),
            ..Default::default()
        }
    }
    fn as_renderable_object(&self) -> Option<&dyn RenderableObject<ObjectType>> {
        Some(self)
    }
}

impl RenderableObject<ObjectType> for Castle {
    fn render_info(&self) -> RenderInfo {
        self.sprite.render_info_default()
    }
}
