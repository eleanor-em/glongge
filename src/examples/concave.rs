use glongge_derive::*;
use glongge::core::{
    prelude::*,
    scene::{Scene, SceneName},
};
use glongge::core::util::collision::ConvexCollider;
use crate::object_type::ObjectType;

#[allow(dead_code)]
#[derive(Copy, Clone)]
pub struct ConcaveScene;
impl Scene<ObjectType> for ConcaveScene {
    fn name(&self) -> SceneName { SceneName::new("concave") }

    fn create_objects(&self, _entrance_id: usize) -> Vec<AnySceneObject<ObjectType>> {
        vec![
            ConvexHull::new(),
        ]
    }
}

#[register_scene_object]
pub struct ConvexHull {}

#[partially_derive_scene_object]
impl SceneObject<ObjectType> for ConvexHull {
    fn on_ready(&mut self, ctx: &mut UpdateContext<ObjectType>) {
        let collider = CollisionShape::from_collider(ConvexCollider::convex_hull_of(vec![
            Vec2 { x: -16., y: -16. } * 2,
            Vec2 { x: 16., y: -18. } * 2,
            Vec2 { x: 10., y: 12. } * 2,
            Vec2 { x: 1., y: 4. } * 2,
            Vec2 { x: -8., y: 14. } * 2,
        ]), &vec![], &vec![]);
        collider.checked_downcast_mut::<CollisionShape>().show_wireframe();
        ctx.object().add_child(collider);
    }

    fn transform(&self) -> Transform {
        Transform {
            centre: [100., 100.].into(),
            ..Default::default()
        }
    }
}
