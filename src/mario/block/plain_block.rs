use glongge_derive::{partially_derive_scene_object, register_scene_object};
use glongge::{
    core::prelude::*,
    resource::sprite::Sprite
};
use crate::mario::{BLOCK_COLLISION_TAG};
use crate::object_type::ObjectType;

#[register_scene_object]
pub struct Block {
    top_left: Vec2,
    sprite: Sprite<ObjectType>,

    initial_y: f64,
    v_speed: f64,
    v_accel: f64,
}

impl Block {
    pub fn new(top_left: Vec2Int) -> Box<Self> {
        Box::new(Self {
            top_left: top_left.into(),
            initial_y: top_left.y as f64,
            ..Default::default()
        })
    }
}

#[partially_derive_scene_object]
impl SceneObject<ObjectType> for Block {
    fn on_load(&mut self, object_ctx: &mut ObjectContext<ObjectType>, resource_handler: &mut ResourceHandler) -> Result<RenderItem> {
        let texture = resource_handler.texture.wait_load_file("res/world_sheet.png".to_string())?;
        self.sprite = Sprite::from_single_extent(
            object_ctx,
            texture,
            Vec2Int { x: 16, y: 16},
            Vec2Int { x: 48, y: 476 });
        Ok(self.sprite.create_vertices())
    }
    fn on_ready(&mut self, ctx: &mut UpdateContext<ObjectType>) {
        ctx.object().add_child(CollisionShape::new(
            self.sprite.as_box_collider(),
            &self.emitting_tags(),
            &self.listening_tags()
        ));
    }

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
    fn emitting_tags(&self) -> Vec<&'static str> {
        [BLOCK_COLLISION_TAG].into()
    }
}

impl RenderableObject<ObjectType> for Block {
    fn render_info(&self) -> RenderInfo {
        self.sprite.render_info_default()
    }
}
