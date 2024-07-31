use glongge_derive::{partially_derive_scene_object, register_scene_object};
use glongge::{
    core::prelude::*,
    resource::sprite::Sprite
};
use crate::mario::{FLAG_COLLISION_TAG};
use crate::object_type::ObjectType;

#[register_scene_object]
pub struct Flagpole {
    top_left: Vec2,
    sprite: Sprite<ObjectType>,
}

impl Flagpole {
    pub fn new(top_left: Vec2Int) -> Box<Self> {
        Box::new(Self {
            top_left: top_left.into(),
            ..Default::default()
        })
    }
}

#[partially_derive_scene_object]
impl SceneObject<ObjectType> for Flagpole {
    fn on_load(&mut self, object_ctx: &mut ObjectContext<ObjectType>, resource_handler: &mut ResourceHandler) -> Result<RenderItem> {
        let texture = resource_handler.texture.wait_load_file("res/world_sheet.png".to_string())?;
        self.sprite = Sprite::from_single_coords(
            object_ctx,
            texture,
            Vec2Int { x: 0, y: 588},
            Vec2Int { x: 16, y: 748 });
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
    fn collider(&self) -> GenericCollider {
        self.sprite.as_box_collider(self.transform()).as_generic()
    }
    fn emitting_tags(&self) -> Vec<&'static str> {
        [FLAG_COLLISION_TAG].into()
    }
}

impl RenderableObject<ObjectType> for Flagpole {
    fn render_info(&self) -> RenderInfo {
        self.sprite.render_info_default()
    }
}
