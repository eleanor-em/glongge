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
    ObjectType,
    player::Player
};


#[register_scene_object]
pub struct QuestionBlock {
    top_left: Vec2,
    sprite: Sprite,
    empty_sprite: Sprite,
    is_empty: bool,

    initial_y: f64,
    v_speed: f64,
    v_accel: f64,
}

impl QuestionBlock {
    pub fn new(top_left: Vec2Int) -> Box<Self> {
        Box::new(Self {
            top_left: top_left.into(),
            sprite: Sprite::default(),
            empty_sprite: Sprite::default(),
            is_empty: false,
            initial_y: top_left.y as f64,
            v_speed: 0.,
            v_accel: 0.,
        })
    }

    fn current_sprite(&self) -> &Sprite {
        if self.is_empty {
            &self.empty_sprite
        } else {
            &self.sprite
        }
    }
}

#[partially_derive_scene_object]
impl SceneObject<ObjectType> for QuestionBlock {
    fn on_load(&mut self, resource_handler: &mut ResourceHandler) -> Result<RenderItem> {
        let texture_id = resource_handler.texture.wait_load_file("res/world_sheet.png".to_string())?;
        self.sprite = Sprite::from_tileset(
            texture_id,
            Vec2Int { x: 3, y: 1},
            Vec2Int { x: 16, y: 16 },
            Vec2Int { x: 298, y: 78 },
            Vec2Int { x: 1, y: 0 })
            .with_frame_orders(vec![0, 1, 2, 1])
            .with_frame_time_ms(vec![600, 100, 100, 100]);
        self.empty_sprite = Sprite::from_single_extent(
            texture_id,
            Vec2Int { x: 16, y: 16 },
            Vec2Int { x: 349, y: 78 });
        Ok(self.sprite.create_vertices())
    }

    fn on_fixed_update(&mut self, _ctx: &mut UpdateContext<ObjectType>) {
        self.sprite.fixed_update();
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
            centre: self.top_left + self.current_sprite().half_widths(),
            ..Default::default()
        }
    }
    fn as_renderable_object(&self) -> Option<&dyn RenderableObject<ObjectType>> {
        Some(self)
    }
    fn collider(&self) -> Box<dyn Collider> {
        self.current_sprite().as_box_collider(self.transform())
    }
    fn emitting_tags(&self) -> Vec<&'static str> {
        [BLOCK_COLLISION_TAG].into()
    }
}

impl RenderableObject<ObjectType> for QuestionBlock {
    fn render_info(&self) -> RenderInfo {
        self.current_sprite().render_info_default()
    }
}

impl Bumpable for QuestionBlock {
    fn bump(&mut self, _player: &mut Player) {
        self.v_speed = -from_nes(3, 0, 0, 0);
        self.v_accel = from_nes_accel(0, 9, 15, 0);
        self.is_empty = true;
        // TODO: drop item
    }
}
