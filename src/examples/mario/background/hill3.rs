use glongge_derive::{partially_derive_scene_object, register_scene_object};
use glongge::{
    core::{
        prelude::*,
        render::VertexDepth,
    },
    resource::sprite::Sprite,
};
use crate::object_type::ObjectType;

#[register_scene_object]
pub struct Hill3 {
    top_left: Vec2,
    sprite: Sprite,
}

impl Hill3 {
    pub fn new(top_left: Vec2Int) -> AnySceneObject<ObjectType> {
        AnySceneObject::new(Self { top_left: top_left.into(), ..Default::default() })
    }
}

#[partially_derive_scene_object]
impl SceneObject<ObjectType> for Hill3 {
    fn on_load(&mut self, object_ctx: &mut ObjectContext<ObjectType>, resource_handler: &mut ResourceHandler) -> Result<Option<RenderItem>> {
        let texture = resource_handler.texture.wait_load_file("res/world_sheet.png".to_string())?;
        self.sprite = Sprite::from_single_coords(
            object_ctx,
            texture,
            Vec2Int { x: 200, y: 732 },
            Vec2Int { x: 248, y: 764 }
        ).with_depth(VertexDepth::Back(0));
        Ok(None)
    }

    fn transform(&self) -> Transform {
        Transform {
            centre: self.top_left + self.sprite.half_widths(),
            ..Default::default()
        }
    }
}
