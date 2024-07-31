use glongge_derive::{partially_derive_scene_object, register_scene_object};
use glongge::{
    core::{
        prelude::*,
        util::collision::Collider,
        util::linalg::{AxisAlignedExtent, Vec2, Vec2Int},
        util::linalg::Transform
    },
    resource::ResourceHandler,
};
use glongge::core::render::RenderInfo;
use glongge::core::render::RenderItem;
use glongge::core::scene::{RenderableObject, SceneObject};
use glongge::core::update::ObjectContext;
use glongge::resource::sprite::Sprite;
use crate::mario::{BLOCK_COLLISION_TAG};
use crate::object_type::ObjectType;

#[register_scene_object]
pub struct DecorativePipe {
    top_left: Vec2,
    sprite: Sprite<ObjectType>,
}

impl DecorativePipe {
    pub fn new(top_left: Vec2Int) -> Box<Self> {
        Box::new(Self {
            top_left: top_left.into(),
            ..Default::default()
        })
    }
}

#[partially_derive_scene_object]
impl SceneObject<ObjectType> for DecorativePipe {
    fn on_load(&mut self, object_ctx: &mut ObjectContext<ObjectType>, resource_handler: &mut ResourceHandler) -> Result<RenderItem> {
        let texture = resource_handler.texture.wait_load_file("res/world_sheet.png".to_string())?;
        self.sprite = Sprite::from_single_coords(
            object_ctx,
            texture,
            Vec2Int { x: 224, y: 324 },
            Vec2Int { x: 256, y: 676}
        );
        Ok(self.sprite.create_vertices())
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

impl RenderableObject<ObjectType> for DecorativePipe {
    fn render_info(&self) -> RenderInfo {
        self.sprite.render_info_default()
    }
}
