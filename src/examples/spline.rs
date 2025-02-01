use std::iter;
use rand::Rng;
use crate::object_type::ObjectType;
use glongge::core::{
    prelude::*,
    scene::{Scene, SceneName},
};
use glongge::util::canvas::Canvas;
use glongge::util::spline::InteractiveSpline;

#[allow(dead_code)]
#[derive(Copy, Clone)]
pub struct SplineScene;
impl Scene<ObjectType> for SplineScene {
    fn name(&self) -> SceneName {
        SceneName::new("spline")
    }

    fn create_objects(&self, _entrance_id: usize) -> Vec<AnySceneObject<ObjectType>> {
        let spline = InteractiveSpline::create();
        {
            let mut spline_inner = spline.downcast_mut::<InteractiveSpline>().unwrap();
            let mut rng = rand::thread_rng();
            for point in iter::from_fn(|| {
                Some(
                    Vec2::from([rng.gen_range(0.0..200.0), rng.gen_range(0.0..200.0)])
                        + 200. * Vec2::one(),
                )
            })
                .take(3) {
                spline_inner.spline_mut().push(point);
            }
            spline_inner.force_visible();
            spline_inner.recalculate();
        }

        vec![Canvas::create(), spline]
    }
}
