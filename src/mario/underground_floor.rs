use glongge_derive::{partially_derive_scene_object, register_scene_object};
use glongge::{
    core::prelude::*,
    resource::sprite::Sprite,
};
use crate::mario::{BLOCK_COLLISION_TAG};
use crate::object_type::ObjectType;

#[register_scene_object]
pub struct UndergroundFloor {
    top_left: Vec2,
    sprite: Sprite<ObjectType>,
}

impl UndergroundFloor {
    pub fn new(top_left: Vec2Int) -> Box<Self> {
        Box::new(Self { top_left: top_left.into(), ..Default::default() })
    }
}

#[partially_derive_scene_object]
impl SceneObject<ObjectType> for UndergroundFloor {
    fn on_load(&mut self, object_ctx: &mut ObjectContext<ObjectType>, resource_handler: &mut ResourceHandler) -> Result<RenderItem> {
        let texture = resource_handler.texture.wait_load_file("res/world_sheet.png".to_string())?;
        self.sprite = Sprite::from_single_extent(
            object_ctx,
            texture.clone(),
            Vec2Int { x: 16, y: 16 },
            Vec2Int { x: 147, y: 16 }
        );
        Ok(self.sprite.create_vertices())
    }

    fn on_ready(&mut self, ctx: &mut UpdateContext<ObjectType>) {
        ctx.object().add_child(
            CollisionShape::new(self.sprite.as_box_collider(),
            &self.emitting_tags(),
            &self.listening_tags()
        ));
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
    fn emitting_tags(&self) -> Vec<&'static str> {
        [BLOCK_COLLISION_TAG].into()
    }
}

impl RenderableObject<ObjectType> for UndergroundFloor {
    fn render_info(&self) -> RenderInfo {
        self.sprite.render_info_default()
    }
}
