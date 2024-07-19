#[allow(unused_imports)]
use crate::core::prelude::*;

use std::any::Any;
use std::time::Duration;
use glongge_derive::{partially_derive_scene_object, register_scene_object};
use crate::core::collision::{BoxCollider, Collider};
use crate::core::linalg::{AxisAlignedExtent, Vec2, Vec2Int};
use crate::gg::scene::sample::mario::{BRICK_COLLISION_TAG, ObjectType};
use crate::gg::{RenderableObject, RenderInfo, SceneObject, Transform, UpdateContext, VertexWithUV};
use crate::resource::ResourceHandler;
use crate::resource::sprite::Sprite;

#[register_scene_object]
pub struct Pipe {
    top_left: Vec2,
    sprite: Sprite,
}

impl Pipe {
    pub fn new(top_left: Vec2Int) -> Box<Self> {
        Box::new(Self {
            top_left: top_left.into(),
            sprite: Sprite::default(),
        })
    }
}

#[partially_derive_scene_object]
impl SceneObject<ObjectType> for Pipe {
    fn on_load(&mut self, resource_handler: &mut ResourceHandler) -> Result<()> {
        let texture_id = resource_handler.texture.wait_load_file("res/world_sheet.png".to_string())?;
        self.sprite = Sprite::from_single_coords(
            texture_id,
            Vec2Int { x: 112, y: 612 },
            Vec2Int { x: 144, y: 676}
        );
        Ok(())
    }
    fn on_ready(&mut self) {}
    fn on_update(&mut self, _delta: Duration, _ctx: &mut UpdateContext<ObjectType>) {}
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
        [BRICK_COLLISION_TAG].into()
    }
}

impl RenderableObject<ObjectType> for Pipe {
    fn create_vertices(&self) -> Vec<VertexWithUV> {
        self.sprite.create_vertices()
    }

    fn render_info(&self) -> RenderInfo {
        self.sprite.render_info_default()
    }
}
