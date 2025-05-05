use glongge::core::{
    prelude::*,
    scene::{Scene, SceneName},
};
use glongge::util::canvas::Canvas;
use glongge::util::collision::{BoxCollider, CompoundCollider, ConvexCollider, Polygonal};
use glongge_derive::partially_derive_scene_object;

#[allow(dead_code)]
#[derive(Copy, Clone)]
pub struct ConcaveScene;
impl Scene for ConcaveScene {
    fn name(&self) -> SceneName {
        SceneName::new("concave")
    }

    fn create_objects(&self, _entrance_id: usize) -> Vec<SceneObjectWrapper> {
        vec![
            Canvas::new().into_wrapper(),
            ConvexHull::default().into_wrapper(),
            Compound::default().into_wrapper(),
            TrivialDecomposed::default().into_wrapper(),
            Decomposed::default().into_wrapper(),
            DecomposedCorner::default().into_wrapper(),
            DecomposedTee::default().into_wrapper(),
            DecomposedU::default().into_wrapper(),
            DecomposedBigU::default().into_wrapper(),
            DecomposedCompound::default().into_wrapper(),
            PixelPerfect::default().into_wrapper(),
            PixelPerfectConvex::default().into_wrapper(),
        ]
    }
}

#[derive(Default)]
pub struct ConvexHull {}

#[partially_derive_scene_object]
impl SceneObject for ConvexHull {
    fn on_ready(&mut self, ctx: &mut UpdateContext) {
        let mut collider = CollisionShape::from_collider(
            ConvexCollider::convex_hull_of(vec![
                Vec2 { x: -16., y: -16. } * 2,
                Vec2 { x: 16., y: -18. } * 2,
                Vec2 { x: 10., y: 12. } * 2,
                Vec2 { x: 1., y: 4. } * 2,
                Vec2 { x: -8., y: 14. } * 2,
            ])
            .unwrap(),
            &[],
            &[],
        );
        ctx.object().transform_mut().centre = [100., 100.].into();
        collider.show_wireframe();
        ctx.object_mut().add_child(collider);
    }
}

#[derive(Default)]
pub struct Compound {}

#[partially_derive_scene_object]
impl SceneObject for Compound {
    fn on_ready(&mut self, ctx: &mut UpdateContext) {
        let mut collider = CollisionShape::from_collider(
            CompoundCollider::new(vec![
                BoxCollider::from_top_left(Vec2 { x: 0., y: 8. }, 32 * Vec2::one()).as_convex(),
                ConvexCollider::convex_hull_of(vec![
                    Vec2 { x: 0., y: 8. },
                    Vec2 { x: 4., y: 0. },
                    Vec2 { x: 8., y: 8. },
                ])
                .unwrap(),
                ConvexCollider::convex_hull_of(vec![
                    Vec2 { x: 24., y: 8. },
                    Vec2 { x: 28., y: 0. },
                    Vec2 { x: 32., y: 8. },
                ])
                .unwrap(),
            ]),
            &[],
            &[],
        );
        collider.show_wireframe();
        ctx.object_mut().add_child(collider);
        ctx.object().transform_mut().centre = [200., 100.].into();
    }
}

#[derive(Default)]
pub struct PixelPerfect {}

#[partially_derive_scene_object]
impl SceneObject for PixelPerfect {
    fn on_load(
        &mut self,
        object_ctx: &mut ObjectContext,
        resource_handler: &mut ResourceHandler,
    ) -> Result<Option<RenderItem>> {
        let tex = resource_handler.texture.wait_load_file("res/mario.png")?;
        let Some(tex_raw) = resource_handler.texture.wait_get_raw(tex.id())? else {
            panic!("missing sprite")
        };
        object_ctx.add_child(CollisionShape::from_object(
            self,
            CompoundCollider::pixel_perfect(&tex_raw)?,
        ));
        object_ctx.transform_mut().centre = [400., 200.].into();
        object_ctx.transform_mut().scale = [8., 8.].into();
        Ok(None)
    }
}

#[derive(Default)]
pub struct PixelPerfectConvex {}

#[partially_derive_scene_object]
impl SceneObject for PixelPerfectConvex {
    fn on_load(
        &mut self,
        object_ctx: &mut ObjectContext,
        resource_handler: &mut ResourceHandler,
    ) -> Result<Option<RenderItem>> {
        let tex = resource_handler.texture.wait_load_file("res/mario.png")?;
        let Some(tex_raw) = resource_handler.texture.wait_get_raw(tex.id())? else {
            panic!("missing sprite")
        };
        object_ctx.add_child(CollisionShape::from_object(
            self,
            CompoundCollider::pixel_perfect_convex(&tex_raw)?,
        ));
        object_ctx.transform_mut().centre = [400., 300.].into();
        object_ctx.transform_mut().scale = [8., 8.].into();
        Ok(None)
    }
}
#[derive(Default)]
pub struct TrivialDecomposed {}

#[partially_derive_scene_object]
impl SceneObject for TrivialDecomposed {
    fn on_ready(&mut self, ctx: &mut UpdateContext) {
        let compound = CompoundCollider::decompose(
            ConvexCollider::convex_hull_of(vec![
                Vec2 { x: -16., y: -16. } * 2,
                Vec2 { x: 16., y: -18. } * 2,
                Vec2 { x: 10., y: 12. } * 2,
                Vec2 { x: 1., y: 4. } * 2,
                Vec2 { x: -8., y: 14. } * 2,
            ])
            .unwrap()
            .vertices(),
        );
        let mut collider = CollisionShape::from_collider(compound, &[], &[]);
        collider.show_wireframe();
        ctx.object_mut().add_child(collider);
        ctx.object().transform_mut().centre = [100., 200.].into();
    }
}

#[derive(Default)]
pub struct Decomposed {}

#[partially_derive_scene_object]
impl SceneObject for Decomposed {
    fn on_ready(&mut self, ctx: &mut UpdateContext) {
        let compound = CompoundCollider::decompose(vec![
            Vec2 { x: -16., y: -16. } * 2,
            Vec2 { x: 16., y: -18. } * 2,
            Vec2 { x: 10., y: 12. } * 2,
            Vec2 { x: 1., y: 4. } * 2,
            Vec2 { x: -8., y: 14. } * 2,
        ]);
        println!("pieces: {}", compound.len());
        let mut collider = CollisionShape::from_collider(compound, &[], &[]);
        collider.show_wireframe();
        ctx.object_mut().add_child(collider);
        ctx.object().transform_mut().centre = [300., 100.].into();
    }
}

#[derive(Default)]
pub struct DecomposedCorner {}
#[partially_derive_scene_object]
impl SceneObject for DecomposedCorner {
    fn on_ready(&mut self, ctx: &mut UpdateContext) {
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
        let mut collider = CollisionShape::from_collider(compound, &[], &[]);
        collider.show_wireframe();
        ctx.object_mut().add_child(collider);
        ctx.object().transform_mut().centre = [200., 300.].into();
    }
}

#[derive(Default)]
pub struct DecomposedTee {}
#[partially_derive_scene_object]
impl SceneObject for DecomposedTee {
    fn on_ready(&mut self, ctx: &mut UpdateContext) {
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
        let mut collider = CollisionShape::from_collider(compound, &[], &[]);
        collider.show_wireframe();
        ctx.object_mut().add_child(collider);
        ctx.object().transform_mut().centre = [200., 200.].into();
    }
}

#[derive(Default)]
pub struct DecomposedU {}
#[partially_derive_scene_object]
impl SceneObject for DecomposedU {
    fn on_ready(&mut self, ctx: &mut UpdateContext) {
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
        let mut collider = CollisionShape::from_collider(compound, &[], &[]);
        collider.show_wireframe();
        ctx.object_mut().add_child(collider);
        ctx.object().transform_mut().centre = [300., 200.].into();
    }
}

#[derive(Default)]
pub struct DecomposedCompound {}
#[partially_derive_scene_object]
impl SceneObject for DecomposedCompound {
    fn on_ready(&mut self, ctx: &mut UpdateContext) {
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
        let mut collider = CollisionShape::from_collider(compound, &[], &[]);
        collider.show_wireframe();
        ctx.object_mut().add_child(collider);
        ctx.object().transform_mut().centre = [200., 250.].into();
    }
}

#[derive(Default)]
pub struct DecomposedBigU {}
#[partially_derive_scene_object]
impl SceneObject for DecomposedBigU {
    fn on_ready(&mut self, ctx: &mut UpdateContext) {
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
        let mut collider = CollisionShape::from_collider(compound, &[], &[]);
        collider.show_wireframe();
        ctx.object_mut().add_child(collider);
        ctx.object().transform_mut().centre = [300., 300.].into();
    }
}
