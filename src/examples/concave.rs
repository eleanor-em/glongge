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
            Canvas::create(),
            ConvexHull::create(),
            Compound::create(),
            TrivialDecomposed::create(),
            Decomposed::create(),
            DecomposedCorner::create(),
            DecomposedTee::create(),
            DecomposedU::create(),
            DecomposedBigU::create(),
            DecomposedCompound::create(),
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
        ]).unwrap(), &[], &[]);
        ctx.object().transform_mut().centre = [100., 100.].into();
        collider.checked_downcast_mut::<CollisionShape>().show_wireframe();
        ctx.object_mut().add_child(collider);
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
            ]).unwrap(),
            ConvexCollider::convex_hull_of(vec![
                Vec2 { x: 24., y: 8. },
                Vec2 { x: 28., y: 0. },
                Vec2 { x: 32., y: 8. },
            ]).unwrap(),
        ]), &[], &[]);
        collider.checked_downcast_mut::<CollisionShape>().show_wireframe();
        ctx.object_mut().add_child(collider);
        ctx.object().transform_mut().centre = [200., 100.].into();
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
        ]).unwrap().vertices());
        let collider = CollisionShape::from_collider(compound, &[], &[]);
        collider.checked_downcast_mut::<CollisionShape>().show_wireframe();
        ctx.object_mut().add_child(collider);
        ctx.object().transform_mut().centre = [100., 200.].into();
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
        let collider = CollisionShape::from_collider(compound, &[], &[]);
        collider.checked_downcast_mut::<CollisionShape>().show_wireframe();
        ctx.object_mut().add_child(collider);
        ctx.object().transform_mut().centre = [300., 100.].into();
    }
}

#[register_scene_object]
pub struct DecomposedCorner {}
#[partially_derive_scene_object]
impl SceneObject<ObjectType> for DecomposedCorner {
    fn on_ready(&mut self, ctx: &mut UpdateContext<ObjectType>) {
        let size = 16;
        let compound = CompoundCollider::decompose(vec![
            Vec2 { x: 0., y: 2. } * size,
            Vec2 { x: 0., y: 1. } * size,
            Vec2 { x: 1., y: 1. } * size,
            Vec2 { x: 1., y: 0. } * size,
            Vec2 { x: 2., y: 0. } * size,
            Vec2 { x: 2., y: 2. } * size,
        ]);
        println!("pieces: {}", compound.len());
        let collider = CollisionShape::from_collider(compound, &[], &[]);
        collider.checked_downcast_mut::<CollisionShape>().show_wireframe();
        ctx.object_mut().add_child(collider);
        ctx.object().transform_mut().centre = [200., 300.].into();
    }
}

#[register_scene_object]
pub struct DecomposedTee {}
#[partially_derive_scene_object]
impl SceneObject<ObjectType> for DecomposedTee {
    fn on_ready(&mut self, ctx: &mut UpdateContext<ObjectType>) {
        let size = 16.;
        let compound = CompoundCollider::decompose(vec![
            Vec2 { x: 0., y: 1. } * size,
            Vec2 { x: 0., y: 0. } * size,
            Vec2 { x: 1., y: 0. } * size,
            Vec2 { x: 1., y: -1. } * size,
            Vec2 { x: 2., y: -1. } * size,
            Vec2 { x: 2., y: 0. } * size,
            Vec2 { x: 3., y: 0. } * size,
            Vec2 { x: 3., y: 1. } * size,
        ]);
        println!("pieces: {}", compound.len());
        let collider = CollisionShape::from_collider(compound, &[], &[]);
        collider.checked_downcast_mut::<CollisionShape>().show_wireframe();
        ctx.object_mut().add_child(collider);
        ctx.object().transform_mut().centre = [200., 200.].into();
    }
}

#[register_scene_object]
pub struct DecomposedU {}
#[partially_derive_scene_object]
impl SceneObject<ObjectType> for DecomposedU {
    fn on_ready(&mut self, ctx: &mut UpdateContext<ObjectType>) {
        let size = 16;
        let compound = CompoundCollider::decompose(vec![
            Vec2 { x: 0., y: 0. } * size,
            Vec2 { x: 1., y: 0. } * size,
            Vec2 { x: 1., y: 1. } * size,
            Vec2 { x: 2., y: 1. } * size,
            Vec2 { x: 2., y: 0. } * size,
            Vec2 { x: 3., y: 0. } * size,
            Vec2 { x: 3., y: 2. } * size,
            Vec2 { x: 0., y: 2. } * size,
        ]);
        println!("pieces: {}", compound.len());
        let collider = CollisionShape::from_collider(compound, &[], &[]);
        collider.checked_downcast_mut::<CollisionShape>().show_wireframe();
        ctx.object_mut().add_child(collider);
        ctx.object().transform_mut().centre = [300., 200.].into();
    }
}

#[register_scene_object]
pub struct DecomposedCompound {}
#[partially_derive_scene_object]
impl SceneObject<ObjectType> for DecomposedCompound {
    fn on_ready(&mut self, ctx: &mut UpdateContext<ObjectType>) {
        let compound = CompoundCollider::decompose(vec![
            Vec2 { x: 0., y: 8. },
            Vec2 { x: 4., y: 0. },
            Vec2 { x: 8., y: 8. },
            Vec2 { x: 24., y: 8. },
            Vec2 { x: 28., y: 0. },
            Vec2 { x: 32., y: 8. },
            Vec2 { x: 32., y: 40. },
            Vec2 { x: 8., y: 40. },
            Vec2 { x: 0., y: 40. },
        ]);
        println!("pieces: {}", compound.len());
        let collider = CollisionShape::from_collider(compound, &[], &[]);
        collider.checked_downcast_mut::<CollisionShape>().show_wireframe();
        ctx.object_mut().add_child(collider);
        ctx.object().transform_mut().centre = [200., 250.].into();
    }
}

#[register_scene_object]
pub struct DecomposedBigU {}
#[partially_derive_scene_object]
impl SceneObject<ObjectType> for DecomposedBigU {
    fn on_ready(&mut self, ctx: &mut UpdateContext<ObjectType>) {
        let size = 16;
        let compound = CompoundCollider::decompose(vec![
            Vec2 { x: 0., y: -2. } * size,
            Vec2 { x: 1., y: -2. } * size,
            Vec2 { x: 1., y: 1. } * size,
            Vec2 { x: 2., y: 1. } * size,
            Vec2 { x: 2., y: -2. } * size,
            Vec2 { x: 3., y: -2. } * size,
            Vec2 { x: 3., y: 2. } * size,
            Vec2 { x: 0., y: 2. } * size,
        ]);
        println!("pieces: {}", compound.len());
        let collider = CollisionShape::from_collider(compound, &[], &[]);
        collider.checked_downcast_mut::<CollisionShape>().show_wireframe();
        ctx.object_mut().add_child(collider);
        ctx.object().transform_mut().centre = [300., 300.].into();
    }
}
