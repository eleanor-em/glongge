use num_traits::Zero;
use glongge_derive::{partially_derive_scene_object, register_scene_object};
use glongge::{
    core::{
        prelude::*,
        scene::SceneDestination,
        util::collision::Collider,
        util::linalg::{AxisAlignedExtent, Vec2, Vec2Int},
        util::linalg::Transform
    },
    resource::{
        ResourceHandler,
        sprite::Sprite
    }
};
use glongge::core::render::{RenderInfo, RenderItem, VertexDepth};
use glongge::core::scene::{RenderableObject, SceneObject};
use crate::mario::{BLOCK_COLLISION_TAG, ObjectType, PIPE_COLLISION_TAG};

#[register_scene_object]
pub struct Pipe {
    top_left: Vec2,
    sprite: Sprite,
    orientation: Vec2,
    destination: Option<SceneDestination>,
}

impl Pipe {
    pub fn new(top_left: Vec2Int, orientation: Vec2, destination: Option<SceneDestination>) -> Box<Self> {
        Box::new(Self {
            top_left: top_left.into(),
            sprite: Sprite::default(),
            orientation,
            destination
        })
    }

    pub fn orientation(&self) -> Vec2 { self.orientation }
    pub fn destination(&self) -> Option<SceneDestination> { self.destination }
    pub fn top(&self) -> f64 { self.transform().centre.y - self.sprite.half_widths().y }
}

#[partially_derive_scene_object]
impl SceneObject<ObjectType> for Pipe {
    fn on_load(&mut self, resource_handler: &mut ResourceHandler) -> Result<RenderItem> {
        let texture = resource_handler.texture.wait_load_file("res/world_sheet.png".to_string())?;
        self.sprite = if self.orientation.x.is_zero() {
            Sprite::from_single_coords(
                texture,
                Vec2Int { x: 112, y: 612 },
                Vec2Int { x: 144, y: 676}
            )
        } else {
            Sprite::from_single_coords(
                texture,
                Vec2Int { x: 192, y: 644 },
                Vec2Int { x: 256, y: 676}
            )
        };
        Ok(self.sprite.create_vertices().with_depth(VertexDepth::Front(1000)))
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
        [PIPE_COLLISION_TAG, BLOCK_COLLISION_TAG].into()
    }
}

impl RenderableObject<ObjectType> for Pipe {
    fn render_info(&self) -> RenderInfo {
        self.sprite.render_info_default()
    }
}
