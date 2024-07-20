use glongge_derive::{partially_derive_scene_object, register_scene_object};
use glongge::{
    core::{
        linalg::{AxisAlignedExtent, Vec2, Vec2Int},
        prelude::*,
        collision::Collider,
        RenderableObject,
        RenderInfo,
        SceneObject,
        Transform,
        UpdateContext,
        VertexWithUV,
    },
    resource::{
        ResourceHandler,
        sprite::Sprite
    }
};
use crate::mario::{FLAG_COLLISION_TAG, ObjectType};

#[register_scene_object]
pub struct Flagpole {
    top_left: Vec2,
    sprite: Sprite,

    initial_y: f64,
    v_speed: f64,
    v_accel: f64,
}

impl Flagpole {
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
impl SceneObject<ObjectType> for Flagpole {
    fn on_load(&mut self, resource_handler: &mut ResourceHandler) -> Result<Vec<VertexWithUV>> {
        let texture_id = resource_handler.texture.wait_load_file("res/world_sheet.png".to_string())?;
        self.sprite = Sprite::from_single_coords(
            texture_id,
            Vec2Int { x: 0, y: 588},
            Vec2Int { x: 16, y: 748 });
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
        [FLAG_COLLISION_TAG].into()
    }
}

impl RenderableObject<ObjectType> for Flagpole {
    fn render_info(&self) -> RenderInfo {
        self.sprite.render_info_default()
    }
}
