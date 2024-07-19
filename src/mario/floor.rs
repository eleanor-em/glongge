use glongge_derive::{partially_derive_scene_object, register_scene_object};
use glongge::{
    core::{
        RenderableObject,
        RenderInfo,
        SceneObject,
        Transform,
        VertexWithUV,
        linalg::{AxisAlignedExtent, Vec2, Vec2Int},
        collision::{BoxCollider, Collider},
        prelude::*,
    },
    resource::{
        ResourceHandler,
        sprite::Sprite
    },
};
use crate::mario::{BLOCK_COLLISION_TAG, ObjectType};

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
    fn on_load(&mut self, resource_handler: &mut ResourceHandler) -> Result<()> {
        let texture_id = resource_handler.texture.wait_load_file("res/world_sheet.png".to_string())?;
        self.sprite = Sprite::from_single_extent(
            texture_id,
            Vec2Int { x: 16, y: 16 },
            Vec2Int { x: 0, y: 16 }
        );
        Ok(())
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
        Box::new(BoxCollider::from_transform(self.transform(), self.sprite.half_widths()))
    }
    fn emitting_tags(&self) -> Vec<&'static str> {
        [BLOCK_COLLISION_TAG].into()
    }
}

impl RenderableObject<ObjectType> for Floor {
    fn create_vertices(&self) -> Vec<VertexWithUV> {
        self.sprite.create_vertices()
    }

    fn render_info(&self) -> RenderInfo {
        self.sprite.render_info_default()
    }
}
