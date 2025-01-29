use glongge_derive::{partially_derive_scene_object, register_scene_object};
use glongge::{
    core::prelude::*,
    resource::sprite::Sprite
};
use crate::examples::mario::{
    block::Bumpable,
    BLOCK_COLLISION_TAG,
    from_nes,
    from_nes_accel,
    player::Player
};
use crate::object_type::ObjectType;

#[register_scene_object]
pub struct UndergroundBrick {
    top_left: Vec2,
    sprite: Sprite,

    initial_y: f32,
    v_speed: f32,
    v_accel: f32,
}

impl UndergroundBrick {
    pub fn create(top_left: Vec2i) -> AnySceneObject<ObjectType> {
        AnySceneObject::new(Self {
            top_left: top_left.into(),
            initial_y: top_left.y as f32,
            ..Default::default()
        })
    }
}

#[partially_derive_scene_object]
impl SceneObject<ObjectType> for UndergroundBrick {
    fn on_load(&mut self, object_ctx: &mut ObjectContext<ObjectType>, resource_handler: &mut ResourceHandler) -> Result<Option<RenderItem>> {
        let texture = resource_handler.texture.wait_load_file("res/world_sheet.png")?;
        self.sprite = Sprite::from_single_extent(
            object_ctx,
            resource_handler,
            texture,
            Vec2i { x: 164, y: 16 },
            Vec2i { x: 16, y: 16});
        object_ctx.transform_mut().centre = self.top_left + self.sprite.half_widths();
        self.initial_y += self.sprite.half_widths().y;
        Ok(None)
    }
    fn on_ready(&mut self, ctx: &mut UpdateContext<ObjectType>) {
        ctx.object_mut().add_child(CollisionShape::from_collider(
            self.sprite.as_box_collider(),
            &self.emitting_tags(),
            &self.listening_tags()
        ));
    }
    fn on_fixed_update(&mut self, ctx: &mut FixedUpdateContext<ObjectType>) {
        let mut transform = ctx.object().transform_mut();
        self.v_speed += self.v_accel;
        transform.centre.y += self.v_speed;
        if transform.centre.y > self.initial_y {
            transform.centre.y = self.initial_y;
            self.v_speed = 0.;
            self.v_accel = 0.;
        }
    }
    fn emitting_tags(&self) -> Vec<&'static str> {
        [BLOCK_COLLISION_TAG].into()
    }
}

impl Bumpable for UndergroundBrick {
    fn bump(&mut self, _player: &mut Player) {
        self.v_speed = -from_nes(3, 0, 0, 0);
        self.v_accel = from_nes_accel(0, 9, 15, 0);
    }
}
