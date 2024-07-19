use std::time::Duration;
use glongge_derive::{partially_derive_scene_object, register_scene_object};
use glongge::{
    core::{
        linalg::{AxisAlignedExtent, Vec2, Vec2Int},
        collision::{BoxCollider, Collider},
        prelude::*,
        RenderableObject,
        RenderInfo,
        SceneObject,
        Transform,
        UpdateContext,
        VertexWithUV,
    },
    resource::{
        ResourceHandler,
        sprite::Sprite
    }
};
use crate::mario::{BLOCK_COLLISION_TAG, ObjectType};

#[register_scene_object]
pub struct Block {
    top_left: Vec2,
    sprite: Sprite,

    initial_y: f64,
    v_speed: f64,
    v_accel: f64,
}

impl Block {
    pub fn new(top_left: Vec2Int) -> Box<Self> {
        Box::new(Self {
            top_left: top_left.into(),
            sprite: Sprite::default(),
            initial_y: top_left.y as f64,
            v_speed: 0.,
            v_accel: 0.,
        })
    }
}

#[partially_derive_scene_object]
impl SceneObject<ObjectType> for Block {
    fn on_load(&mut self, resource_handler: &mut ResourceHandler) -> Result<()> {
        let texture_id = resource_handler.texture.wait_load_file("res/world_sheet.png".to_string())?;
        self.sprite = Sprite::from_single_extent(
            texture_id,
            Vec2Int { x: 16, y: 16},
            Vec2Int { x: 48, y: 476 });
        Ok(())
    }
    fn on_ready(&mut self, _ctx: &mut UpdateContext<ObjectType>) {}
    fn on_update(&mut self, _delta: Duration, _ctx: &mut UpdateContext<ObjectType>) {}
    fn on_fixed_update(&mut self, _ctx: &mut UpdateContext<ObjectType>) {
        self.v_speed += self.v_accel;
        self.top_left.y += self.v_speed;
        if self.top_left.y > self.initial_y {
            self.top_left.y = self.initial_y;
            self.v_speed = 0.;
            self.v_accel = 0.;
        }
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

impl RenderableObject<ObjectType> for Block {
    fn create_vertices(&self) -> Vec<VertexWithUV> {
        self.sprite.create_vertices()
    }

    fn render_info(&self) -> RenderInfo {
        self.sprite.render_info_default()
    }
}
