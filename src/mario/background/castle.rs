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
pub struct Castle {
    top_left: Vec2,
    sprite: Sprite<ObjectType>,
}

impl Castle {
    pub fn new(top_left: Vec2Int) -> Box<Self> {
        Box::new(Self { top_left: top_left.into(), ..Default::default() })
    }
}

#[partially_derive_scene_object]
impl SceneObject<ObjectType> for Castle {
    fn on_load(&mut self, object_ctx: &mut ObjectContext<ObjectType>, resource_handler: &mut ResourceHandler) -> Result<RenderItem> {
        let texture = resource_handler.texture.wait_load_file("res/world_sheet.png".to_string())?;
        self.sprite = Sprite::from_single_coords(
            object_ctx,
            texture,
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
