use std::{
    any::Any,
    time::Duration
};
use num_traits::Zero;
use glongge_derive::{partially_derive_scene_object, register_scene_object};
use crate::{
    resource::ResourceHandler,
    gg::{
        RenderableObject,
        RenderInfo,
        SceneObject,
        Transform,
        UpdateContext,
        VertexWithUV,
        scene::sample::mario::ObjectType
    },
    core::{
        linalg::{Vec2, Vec2Int},
        input::KeyCode,
        collision::{BoxCollider, Collider}
    },
    resource::sprite::Sprite,
};

#[register_scene_object]
pub struct Player {
    pos: Vec2,
    vel: Vec2,
    sprite: Sprite,
}

impl Player {
    const SPEED: f64 = 300.0;
}

#[partially_derive_scene_object]
impl SceneObject<ObjectType> for Player {
    fn on_load(&mut self, resource_handler: &mut ResourceHandler) -> anyhow::Result<()> {
        let texture_id = resource_handler.texture.wait_load_file("res/mario.png".to_string())?;
        self.sprite = Sprite::from_tileset(
            texture_id,
            Vec2Int { x: 3, y: 1 },
            Vec2Int { x: 16, y: 16 },
            Vec2Int { x: 0, y: 0 },
            Vec2Int { x: 2, y: 0 },
            100
        );
        Ok(())
    }
    fn on_ready(&mut self) {
        self.pos = Vec2 { x: 512.0, y: 384.0 };
    }
    fn on_update(&mut self, delta: Duration, update_ctx: UpdateContext<ObjectType>) {
        let mut direction = Vec2::zero();
        if update_ctx.input().down(KeyCode::Left) { direction += Vec2::left(); }
        if update_ctx.input().down(KeyCode::Right) { direction += Vec2::right(); }
        if update_ctx.input().down(KeyCode::Up) { direction += Vec2::up(); }
        if update_ctx.input().down(KeyCode::Down) { direction += Vec2::down(); }
        self.vel = Self::SPEED * direction.normed();
        self.pos += self.vel * delta.as_secs_f64();
    }

    fn transform(&self) -> Transform {
        Transform {
            position: self.pos,
            rotation: 0.0,
        }
    }
    fn as_renderable_object(&self) -> Option<&dyn RenderableObject<ObjectType>> {
        Some(self)
    }
    fn collider(&self) -> Option<Box<dyn Collider>> {
        Some(Box::new(BoxCollider::new(self.transform(), self.sprite.half_widths())))
    }
    fn collision_tags(&self) -> Vec<&'static str> {
        [].into()
    }
}

impl RenderableObject<ObjectType> for Player {
    fn create_vertices(&self) -> Vec<VertexWithUV> {
        self.sprite.create_vertices()
    }

    fn render_info(&self) -> RenderInfo {
        self.sprite.render_info_default()
    }
}
