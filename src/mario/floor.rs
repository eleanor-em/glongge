use glongge_derive::{partially_derive_scene_object, register_scene_object};
use glongge::{
    core::{
        prelude::*,
        util::collision::Collider,
        util::linalg::{AxisAlignedExtent, Vec2, Vec2Int},
        util::linalg::Transform,
    },
    resource::{
        ResourceHandler,
        sprite::Sprite
    },
};
use glongge::core::render::{RenderInfo, RenderItem, VertexDepth};
use glongge::core::scene::{RenderableObject, SceneObject};
use crate::mario::{BLOCK_COLLISION_TAG};
use crate::object_type::ObjectType;

#[register_scene_object]
pub struct Floor {
    top_left: Vec2,
    sprite: Sprite,
}

impl Floor {
    pub fn new(top_left: Vec2Int) -> Box<Self> {
        Box::new(Self { top_left: top_left.into(), sprite: Sprite::default() })
    }
}

#[partially_derive_scene_object]
impl SceneObject<ObjectType> for Floor {
    fn on_load(&mut self, resource_handler: &mut ResourceHandler) -> Result<RenderItem> {
        let texture = resource_handler.texture.wait_load_file("res/world_sheet.png".to_string())?;
        self.sprite = Sprite::from_single_extent(
            texture.clone(),
            Vec2Int { x: 16, y: 16 },
            Vec2Int { x: 0, y: 16 }
        );
        Ok(self.sprite.create_vertices().with_depth(VertexDepth::Front(2000)))
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
    fn collider(&self) -> Box<dyn Collider> {
        self.sprite.as_box_collider(self.transform())
    }
    fn emitting_tags(&self) -> Vec<&'static str> {
        [BLOCK_COLLISION_TAG].into()
    }
}

impl RenderableObject<ObjectType> for Floor {
    fn render_info(&self) -> RenderInfo {
        self.sprite.render_info_default()
    }
}
