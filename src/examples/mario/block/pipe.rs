use crate::examples::mario::{BLOCK_COLLISION_TAG, PIPE_COLLISION_TAG};

use glongge::{
    core::{prelude::*, render::VertexDepth, scene::SceneDestination},
    resource::sprite::Sprite,
};
use glongge_derive::partially_derive_scene_object;
use num_traits::Zero;

#[derive(Default)]
pub struct Pipe {
    top_left: Vec2,
    sprite: Sprite,
    orientation: Vec2,
    destination: Option<SceneDestination>,
}

impl Pipe {
    pub fn new(top_left: Vec2i, orientation: Vec2, destination: Option<SceneDestination>) -> Self {
        Self {
            top_left: top_left.into(),
            orientation,
            destination,
            ..Default::default()
        }
    }

    pub fn orientation(&self) -> Vec2 {
        self.orientation
    }
    pub fn destination(&self) -> Option<SceneDestination> {
        self.destination
    }
}

#[partially_derive_scene_object]
impl SceneObject for Pipe {
    fn on_load(
        &mut self,
        object_ctx: &mut ObjectContext,
        resource_handler: &mut ResourceHandler,
    ) -> Result<Option<RenderItem>> {
        let texture = resource_handler
            .texture
            .wait_load_file("res/world_sheet.png")?;
        self.sprite = if self.orientation.x.is_zero() {
            Sprite::add_from_single_coords(
                object_ctx,
                resource_handler,
                texture,
                Vec2i { x: 112, y: 612 },
                Vec2i { x: 144, y: 676 },
            )
        } else {
            Sprite::add_from_single_coords(
                object_ctx,
                resource_handler,
                texture,
                Vec2i { x: 192, y: 644 },
                Vec2i { x: 256, y: 676 },
            )
        }
        .with_depth(VertexDepth::Front(1000));
        object_ctx.transform_mut().centre = self.top_left + self.sprite.half_widths();
        Ok(None)
    }

    fn on_ready(&mut self, ctx: &mut UpdateContext) {
        ctx.object_mut()
            .add_child(CollisionShape::from_object_sprite(self, &self.sprite));
    }

    fn emitting_tags(&self) -> Vec<&'static str> {
        [PIPE_COLLISION_TAG, BLOCK_COLLISION_TAG].into()
    }
}
