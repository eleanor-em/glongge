use glongge_derive::{partially_derive_scene_object, register_scene_object};
use glongge::{
    core::{
        prelude::*,
        util::collision::Collider,
        util::linalg::{AxisAlignedExtent, Vec2, Vec2Int},
        util::linalg::Transform,
    },
    resource::{
        ResourceHandler,
        sprite::Sprite
    }
};
use glongge::core::render::RenderInfo;
use glongge::core::render::RenderItem;
use glongge::core::scene::{RenderableObject, SceneObject};
use glongge::core::update::UpdateContext;
use crate::mario::{
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

    initial_y: f64,
    v_speed: f64,
    v_accel: f64,
}

impl UndergroundBrick {
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
impl SceneObject<ObjectType> for UndergroundBrick {
    fn on_load(&mut self, resource_handler: &mut ResourceHandler) -> Result<RenderItem> {
        let texture = resource_handler.texture.wait_load_file("res/world_sheet.png".to_string())?;
        self.sprite = Sprite::from_single_extent(
            texture,
            Vec2Int { x: 16, y: 16},
            Vec2Int { x: 164, y: 16 });
        Ok(self.sprite.create_vertices())
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
    fn collider(&self) -> Box<dyn Collider> {
        self.sprite.as_box_collider(self.transform())
    }
    fn emitting_tags(&self) -> Vec<&'static str> {
        [BLOCK_COLLISION_TAG].into()
    }
}

impl RenderableObject<ObjectType> for UndergroundBrick {
    fn render_info(&self) -> RenderInfo {
        self.sprite.render_info_default()
    }
}

impl Bumpable for UndergroundBrick {
    fn bump(&mut self, _player: &mut Player) {
        self.v_speed = -from_nes(3, 0, 0, 0);
        self.v_accel = from_nes_accel(0, 9, 15, 0);
    }
}
