use glongge_derive::*;
use glongge::core::{
    prelude::*,
    scene::{Scene, SceneName},
};
use glongge::core::util::canvas::Canvas;
use glongge::core::util::collision::{Polygonal, BoxCollider, CompoundCollider, ConvexCollider};
use crate::object_type::ObjectType;

#[allow(dead_code)]
#[derive(Copy, Clone)]
pub struct ConcaveScene;
impl Scene<ObjectType> for ConcaveScene {
    fn name(&self) -> SceneName { SceneName::new("concave") }

    fn create_objects(&self, _entrance_id: usize) -> Vec<AnySceneObject<ObjectType>> {
        vec![
            Canvas::new(),
            ConvexHull::new(),
            Compound::new(),
            // TrivialDecomposed::new(),
            // Decomposed::new(),
            DecomposedCorner::new(),
            // DecomposedTee::new(),
            // DecomposedU::new(),
            // DecomposedCompound::new(),
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

#[register_scene_object]
pub struct Compound {}

#[partially_derive_scene_object]
impl SceneObject<ObjectType> for Compound {
    fn on_ready(&mut self, ctx: &mut UpdateContext<ObjectType>) {
        let collider = CollisionShape::from_collider(CompoundCollider::new(vec![
            BoxCollider::from_top_left(Vec2 { x: 0., y: 8. }, 32 * Vec2::one())
                .as_convex(),
            ConvexCollider::convex_hull_of(vec![
                Vec2 { x: 0., y: 8. },
                Vec2 { x: 4., y: 0. },
                Vec2 { x: 8., y: 8. },
            ]),
            ConvexCollider::convex_hull_of(vec![
                Vec2 { x: 24., y: 8. },
                Vec2 { x: 28., y: 0. },
                Vec2 { x: 32., y: 8. },
            ]),
        ]), &vec![], &vec![]);
        collider.checked_downcast_mut::<CollisionShape>().show_wireframe();
        ctx.object().add_child(collider);
    }

    fn transform(&self) -> Transform {
        Transform {
            centre: [200., 100.].into(),
            ..Default::default()
        }
    }
}

#[register_scene_object]
pub struct TrivialDecomposed {}

#[partially_derive_scene_object]
impl SceneObject<ObjectType> for TrivialDecomposed {
    fn on_ready(&mut self, ctx: &mut UpdateContext<ObjectType>) {
        let compound = CompoundCollider::decompose(ConvexCollider::convex_hull_of(vec![
            Vec2 { x: -16., y: -16. } * 2,
            Vec2 { x: 16., y: -18. } * 2,
            Vec2 { x: 10., y: 12. } * 2,
            Vec2 { x: 1., y: 4. } * 2,
            Vec2 { x: -8., y: 14. } * 2,
        ]).vertices());
        let collider = CollisionShape::from_collider(compound, &vec![], &vec![]);
        collider.checked_downcast_mut::<CollisionShape>().show_wireframe();
        ctx.object().add_child(collider);
    }

    fn transform(&self) -> Transform {
        Transform {
            centre: [100., 200.].into(),
            ..Default::default()
        }
    }
}

#[register_scene_object]
pub struct Decomposed {}

#[partially_derive_scene_object]
impl SceneObject<ObjectType> for Decomposed {
    fn on_ready(&mut self, ctx: &mut UpdateContext<ObjectType>) {
        let compound = CompoundCollider::decompose(
            vec![
                Vec2 { x: -16., y: -16. } * 2,
                Vec2 { x: 16., y: -18. } * 2,
                Vec2 { x: 10., y: 12. } * 2,
                Vec2 { x: 1., y: 4. } * 2,
                Vec2 { x: -8., y: 14. } * 2,
            ]
        );
        println!("pieces: {}", compound.len());
        let collider = CollisionShape::from_collider(compound, &vec![], &vec![]);
        collider.checked_downcast_mut::<CollisionShape>().show_wireframe();
        ctx.object().add_child(collider);
    }

    fn transform(&self) -> Transform {
        Transform {
            centre: [300., 100.].into(),
            ..Default::default()
        }
    }
}

#[register_scene_object]
pub struct DecomposedCorner {}
#[partially_derive_scene_object]
impl SceneObject<ObjectType> for DecomposedCorner {
    fn on_ready(&mut self, ctx: &mut UpdateContext<ObjectType>) {
        let compound = CompoundCollider::decompose(vec![
            Vec2 { x: 0., y: 1. },
            Vec2 { x: 0., y: 0. },
            Vec2 { x: 1., y: 0. },
            Vec2 { x: 1., y: -1. },
            Vec2 { x: 2., y: -1. },
            Vec2 { x: 2., y: 1. },
        ]);
        println!("pieces: {}", compound.len());
        let collider = CollisionShape::from_collider(compound, &vec![], &vec![]);
        collider.checked_downcast_mut::<CollisionShape>().show_wireframe();
        ctx.object().add_child(collider);
    }

    fn transform(&self) -> Transform {
        Transform {
            centre: [200., 200.].into(),
            ..Default::default()
        }
    }
}

#[register_scene_object]
pub struct DecomposedTee {}
#[partially_derive_scene_object]
impl SceneObject<ObjectType> for DecomposedTee {
    fn on_ready(&mut self, ctx: &mut UpdateContext<ObjectType>) {
        let compound = CompoundCollider::decompose(vec![
            Vec2 { x: 0., y: 16. },
            Vec2 { x: 0., y: 0. },
            Vec2 { x: 16., y: 0. },
            Vec2 { x: 16., y: -16. },
            Vec2 { x: 32., y: -16. },
            // Vec2 { x: 32., y: 16. },
            Vec2 { x: 32., y: 0. },
            Vec2 { x: 48., y: 0. },
            Vec2 { x: 48., y: 16. },
        ]);
        println!("pieces: {}", compound.len());
        let collider = CollisionShape::from_collider(compound, &vec![], &vec![]);
        collider.checked_downcast_mut::<CollisionShape>().show_wireframe();
        ctx.object().add_child(collider);
    }

    fn transform(&self) -> Transform {
        Transform {
            centre: [200., 200.].into(),
            ..Default::default()
        }
    }
}

#[register_scene_object]
pub struct DecomposedU {}
#[partially_derive_scene_object]
impl SceneObject<ObjectType> for DecomposedU {
    fn on_ready(&mut self, ctx: &mut UpdateContext<ObjectType>) {
        let compound = CompoundCollider::decompose(vec![
            Vec2 { x: 0., y: 0. },
            Vec2 { x: 16., y: 0. },
            Vec2 { x: 16., y: 16. },
            Vec2 { x: 32., y: 16. },
            Vec2 { x: 32., y: 0. },
            Vec2 { x: 48., y: 0. },
            Vec2 { x: 48., y: 32. },
            Vec2 { x: 0., y: 32. },
        ]);
        println!("pieces: {}", compound.len());
        let collider = CollisionShape::from_collider(compound, &vec![], &vec![]);
        collider.checked_downcast_mut::<CollisionShape>().show_wireframe();
        ctx.object().add_child(collider);
    }

    fn transform(&self) -> Transform {
        Transform {
            centre: [300., 200.].into(),
            ..Default::default()
        }
    }
}

#[register_scene_object]
pub struct DecomposedCompound {}
#[partially_derive_scene_object]
impl SceneObject<ObjectType> for DecomposedCompound {
    fn on_ready(&mut self, ctx: &mut UpdateContext<ObjectType>) {
        let compound = CompoundCollider::decompose(vec![
            Vec2 { x: 0., y: 8. },
            Vec2 { x: 32., y: 8. },
            Vec2 { x: 32., y: 40. },
            Vec2 { x: 8., y: 40. },
            Vec2 { x: 4., y: 0. },
            Vec2 { x: 24., y: 8. },
            Vec2 { x: 28., y: 0. },
        ]);
        println!("pieces: {}", compound.len());
        let collider = CollisionShape::from_collider(compound, &vec![], &vec![]);
        collider.checked_downcast_mut::<CollisionShape>().show_wireframe();
        ctx.object().add_child(collider);
    }

    fn transform(&self) -> Transform {
        Transform {
            centre: [200., 200.].into(),
            ..Default::default()
        }
    }
}
