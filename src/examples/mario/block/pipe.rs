use num_traits::Zero;
use glongge_derive::{partially_derive_scene_object, register_scene_object};
use glongge::{
    core::{
        prelude::*,
        scene::SceneDestination,
        render::VertexDepth
    },
    resource::sprite::Sprite,
};
use crate::examples::mario::{BLOCK_COLLISION_TAG, PIPE_COLLISION_TAG};
use crate::object_type::ObjectType;

#[register_scene_object]
pub struct Pipe {
    top_left: Vec2,
    sprite: Sprite,
    orientation: Vec2,
    destination: Option<SceneDestination>,
}

impl Pipe {
    pub fn create(top_left: Vec2Int, orientation: Vec2, destination: Option<SceneDestination>) -> AnySceneObject<ObjectType> {
        AnySceneObject::new(Self {
            top_left: top_left.into(),
            orientation,
            destination,
            ..Default::default()
        })
    }

    pub fn orientation(&self) -> Vec2 { self.orientation }
    pub fn destination(&self) -> Option<SceneDestination> { self.destination }
}

#[partially_derive_scene_object]
impl SceneObject<ObjectType> for Pipe {
    fn on_load(&mut self, object_ctx: &mut ObjectContext<ObjectType>, resource_handler: &mut ResourceHandler) -> Result<Option<RenderItem>> {
        let texture = resource_handler.texture.wait_load_file("res/world_sheet.png".to_string())?;
        self.sprite = if self.orientation.x.is_zero() {
            Sprite::from_single_coords(
                object_ctx,
                texture,
                Vec2Int { x: 112, y: 612 },
                Vec2Int { x: 144, y: 676}
            )
        } else {
            Sprite::from_single_coords(
                object_ctx,
                texture,
                Vec2Int { x: 192, y: 644 },
                Vec2Int { x: 256, y: 676}
            )
        }.with_depth(VertexDepth::Front(1000));
        Ok(None)
    }

    fn on_ready(&mut self, ctx: &mut UpdateContext<ObjectType>) {
        ctx.object_mut().add_child(CollisionShape::from_object_sprite(self, &self.sprite));
    }

    fn transform(&self) -> Transform {
        Transform {
            centre: self.top_left + self.sprite.half_widths(),
            ..Default::default()
        }
    }
    fn emitting_tags(&self) -> Vec<&'static str> {
        [PIPE_COLLISION_TAG, BLOCK_COLLISION_TAG].into()
    }
}
