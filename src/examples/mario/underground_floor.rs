use crate::examples::mario::BLOCK_COLLISION_TAG;
use crate::object_type::ObjectType;
use glongge::{core::prelude::*, resource::sprite::Sprite};
use glongge_derive::{partially_derive_scene_object, register_scene_object};

#[register_scene_object]
pub struct UndergroundFloor {
    top_left: Vec2,
    sprite: Sprite,
}

impl UndergroundFloor {
    pub fn new(top_left: Vec2i) -> Self {
        Self {
            top_left: top_left.into(),
            ..Default::default()
        }
    }
}

#[partially_derive_scene_object]
impl SceneObject<ObjectType> for UndergroundFloor {
    fn on_load(
        &mut self,
        object_ctx: &mut ObjectContext<ObjectType>,
        resource_handler: &mut ResourceHandler,
    ) -> Result<Option<RenderItem>> {
        let texture = resource_handler
            .texture
            .wait_load_file("res/world_sheet.png")?;
        self.sprite = Sprite::add_from_single_extent(
            object_ctx,
            resource_handler,
            texture.clone(),
            Vec2i { x: 147, y: 16 },
            Vec2i { x: 16, y: 16 },
        );
        object_ctx.transform_mut().centre = self.top_left + self.sprite.half_widths();
        Ok(None)
    }

    fn on_ready(&mut self, ctx: &mut UpdateContext<ObjectType>) {
        ctx.object_mut().add_child(CollisionShape::from_collider(
            self.sprite.as_box_collider(),
            &self.emitting_tags(),
            &self.listening_tags(),
        ));
    }

    fn emitting_tags(&self) -> Vec<&'static str> {
        [BLOCK_COLLISION_TAG].into()
    }
}
