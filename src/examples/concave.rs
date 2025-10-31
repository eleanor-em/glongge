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
                Vec2 { x: -16.0, y: -16.0 } * 2,
                Vec2 { x: 16.0, y: -18.0 } * 2,
                Vec2 { x: 10.0, y: 12.0 } * 2,
                Vec2 { x: 1.0, y: 4.0 } * 2,
                Vec2 { x: -8.0, y: 14.0 } * 2,
            ])
            .unwrap(),
            &[],
            &[],
        );
        ctx.object().transform_mut().centre = [100.0, 100.0].into();
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
                BoxCollider::from_top_left(Vec2 { x: 0.0, y: 8.0 }, 32 * Vec2::one()).as_convex(),
                ConvexCollider::convex_hull_of(vec![
                    Vec2 { x: 0.0, y: 8.0 },
                    Vec2 { x: 4.0, y: 0.0 },
                    Vec2 { x: 8.0, y: 8.0 },
                ])
                .unwrap(),
                ConvexCollider::convex_hull_of(vec![
                    Vec2 { x: 24.0, y: 8.0 },
                    Vec2 { x: 28.0, y: 0.0 },
                    Vec2 { x: 32.0, y: 8.0 },
                ])
                .unwrap(),
            ]),
            &[],
            &[],
        );
        collider.show_wireframe();
        ctx.object_mut().add_child(collider);
        ctx.object().transform_mut().centre = [200.0, 100.0].into();
    }
}

#[derive(Default)]
pub struct PixelPerfect {}

#[partially_derive_scene_object]
impl SceneObject for PixelPerfect {
    fn on_load(&mut self, ctx: &mut LoadContext) -> Result<Option<RenderItem>> {
        let tex = ctx.resource().texture.wait_load_file("res/mario.png")?;
        let Some(tex_raw) = ctx.resource().texture.wait_get_raw(tex.id())? else {
            panic!("missing sprite")
        };
        ctx.object_mut().add_child(CollisionShape::from_object(
            self,
            CompoundCollider::pixel_perfect(&tex_raw)?,
        ));
        ctx.object().transform_mut().centre = [400.0, 200.0].into();
        ctx.object().transform_mut().scale = [8.0, 8.0].into();
        Ok(None)
    }
}

#[derive(Default)]
pub struct PixelPerfectConvex {}

#[partially_derive_scene_object]
impl SceneObject for PixelPerfectConvex {
    fn on_load(&mut self, ctx: &mut LoadContext) -> Result<Option<RenderItem>> {
        let tex = ctx.resource().texture.wait_load_file("res/mario.png")?;
        let Some(tex_raw) = ctx.resource().texture.wait_get_raw(tex.id())? else {
            panic!("missing sprite")
        };
        ctx.object_mut().add_child(CollisionShape::from_object(
            self,
            CompoundCollider::pixel_perfect_convex(&tex_raw)?,
        ));
        ctx.object().transform_mut().centre = [400.0, 300.0].into();
        ctx.object().transform_mut().scale = [8.0, 8.0].into();
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
                Vec2 { x: -16.0, y: -16.0 } * 2,
                Vec2 { x: 16.0, y: -18.0 } * 2,
                Vec2 { x: 10.0, y: 12.0 } * 2,
                Vec2 { x: 1.0, y: 4.0 } * 2,
                Vec2 { x: -8.0, y: 14.0 } * 2,
            ])
            .unwrap()
            .vertices(),
        );
        let mut collider = CollisionShape::from_collider(compound, &[], &[]);
        collider.show_wireframe();
        ctx.object_mut().add_child(collider);
        ctx.object().transform_mut().centre = [100.0, 200.0].into();
    }
}

#[derive(Default)]
pub struct Decomposed {}

#[partially_derive_scene_object]
impl SceneObject for Decomposed {
    fn on_ready(&mut self, ctx: &mut UpdateContext) {
        let compound = CompoundCollider::decompose(vec![
            Vec2 { x: -16.0, y: -16.0 } * 2,
            Vec2 { x: 16.0, y: -18.0 } * 2,
            Vec2 { x: 10.0, y: 12.0 } * 2,
            Vec2 { x: 1.0, y: 4.0 } * 2,
            Vec2 { x: -8.0, y: 14.0 } * 2,
        ]);
        println!("pieces: {}", compound.len());
        let mut collider = CollisionShape::from_collider(compound, &[], &[]);
        collider.show_wireframe();
        ctx.object_mut().add_child(collider);
        ctx.object().transform_mut().centre = [300.0, 100.0].into();
    }
}

#[derive(Default)]
pub struct DecomposedCorner {}
#[partially_derive_scene_object]
impl SceneObject for DecomposedCorner {
    fn on_ready(&mut self, ctx: &mut UpdateContext) {
        let size = 16;
        let compound = CompoundCollider::decompose(vec![
            Vec2 { x: 0.0, y: 2.0 } * size,
            Vec2 { x: 0.0, y: 1.0 } * size,
            Vec2 { x: 1.0, y: 1.0 } * size,
            Vec2 { x: 1.0, y: 0.0 } * size,
            Vec2 { x: 2.0, y: 0.0 } * size,
            Vec2 { x: 2.0, y: 2.0 } * size,
        ]);
        println!("pieces: {}", compound.len());
        let mut collider = CollisionShape::from_collider(compound, &[], &[]);
        collider.show_wireframe();
        ctx.object_mut().add_child(collider);
        ctx.object().transform_mut().centre = [200.0, 300.0].into();
    }
}

#[derive(Default)]
pub struct DecomposedTee {}
#[partially_derive_scene_object]
impl SceneObject for DecomposedTee {
    fn on_ready(&mut self, ctx: &mut UpdateContext) {
        let size = 16.0;
        let compound = CompoundCollider::decompose(vec![
            Vec2 { x: 0.0, y: 1.0 } * size,
            Vec2 { x: 0.0, y: 0.0 } * size,
            Vec2 { x: 1.0, y: 0.0 } * size,
            Vec2 { x: 1.0, y: -1.0 } * size,
            Vec2 { x: 2.0, y: -1.0 } * size,
            Vec2 { x: 2.0, y: 0.0 } * size,
            Vec2 { x: 3.0, y: 0.0 } * size,
            Vec2 { x: 3.0, y: 1.0 } * size,
        ]);
        println!("pieces: {}", compound.len());
        let mut collider = CollisionShape::from_collider(compound, &[], &[]);
        collider.show_wireframe();
        ctx.object_mut().add_child(collider);
        ctx.object().transform_mut().centre = [200.0, 200.0].into();
    }
}

#[derive(Default)]
pub struct DecomposedU {}
#[partially_derive_scene_object]
impl SceneObject for DecomposedU {
    fn on_ready(&mut self, ctx: &mut UpdateContext) {
        let size = 16;
        let compound = CompoundCollider::decompose(vec![
            Vec2 { x: 0.0, y: 0.0 } * size,
            Vec2 { x: 1.0, y: 0.0 } * size,
            Vec2 { x: 1.0, y: 1.0 } * size,
            Vec2 { x: 2.0, y: 1.0 } * size,
            Vec2 { x: 2.0, y: 0.0 } * size,
            Vec2 { x: 3.0, y: 0.0 } * size,
            Vec2 { x: 3.0, y: 2.0 } * size,
            Vec2 { x: 0.0, y: 2.0 } * size,
        ]);
        println!("pieces: {}", compound.len());
        let mut collider = CollisionShape::from_collider(compound, &[], &[]);
        collider.show_wireframe();
        ctx.object_mut().add_child(collider);
        ctx.object().transform_mut().centre = [300.0, 200.0].into();
    }
}

#[derive(Default)]
pub struct DecomposedCompound {}
#[partially_derive_scene_object]
impl SceneObject for DecomposedCompound {
    fn on_ready(&mut self, ctx: &mut UpdateContext) {
        let compound = CompoundCollider::decompose(vec![
            Vec2 { x: 0.0, y: 8.0 },
            Vec2 { x: 4.0, y: 0.0 },
            Vec2 { x: 8.0, y: 8.0 },
            Vec2 { x: 24.0, y: 8.0 },
            Vec2 { x: 28.0, y: 0.0 },
            Vec2 { x: 32.0, y: 8.0 },
            Vec2 { x: 32.0, y: 40.0 },
            Vec2 { x: 8.0, y: 40.0 },
            Vec2 { x: 0.0, y: 40.0 },
        ]);
        println!("pieces: {}", compound.len());
        let mut collider = CollisionShape::from_collider(compound, &[], &[]);
        collider.show_wireframe();
        ctx.object_mut().add_child(collider);
        ctx.object().transform_mut().centre = [200.0, 250.0].into();
    }
}

#[derive(Default)]
pub struct DecomposedBigU {}
#[partially_derive_scene_object]
impl SceneObject for DecomposedBigU {
    fn on_ready(&mut self, ctx: &mut UpdateContext) {
        let size = 16;
        let compound = CompoundCollider::decompose(vec![
            Vec2 { x: 0.0, y: -2.0 } * size,
            Vec2 { x: 1.0, y: -2.0 } * size,
            Vec2 { x: 1.0, y: 1.0 } * size,
            Vec2 { x: 2.0, y: 1.0 } * size,
            Vec2 { x: 2.0, y: -2.0 } * size,
            Vec2 { x: 3.0, y: -2.0 } * size,
            Vec2 { x: 3.0, y: 2.0 } * size,
            Vec2 { x: 0.0, y: 2.0 } * size,
        ]);
        println!("pieces: {}", compound.len());
        let mut collider = CollisionShape::from_collider(compound, &[], &[]);
        collider.show_wireframe();
        ctx.object_mut().add_child(collider);
        ctx.object().transform_mut().centre = [300.0, 300.0].into();
    }
}
