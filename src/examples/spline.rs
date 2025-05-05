use glongge::core::{
    prelude::*,
    scene::{Scene, SceneName},
};
use glongge::scene_object_vec;
use glongge::util::canvas::Canvas;
use glongge::util::spline::InteractiveSpline;
use rand::Rng;
use std::iter;

#[allow(dead_code)]
#[derive(Copy, Clone)]
pub struct SplineScene;
impl Scene for SplineScene {
    fn name(&self) -> SceneName {
        SceneName::new("spline")
    }

    fn create_objects(&self, _entrance_id: usize) -> Vec<SceneObjectWrapper> {
        let mut spline = InteractiveSpline::default();
        {
            let mut rng = rand::thread_rng();
            for point in iter::from_fn(|| {
                Some(
                    Vec2::from([rng.gen_range(0.0..200.0), rng.gen_range(0.0..200.0)])
                        + 200. * Vec2::one(),
                )
            })
            .take(3)
            {
                spline.spline_mut().push(point);
            }
            spline.force_visible();
            spline.recalculate();
        }

        scene_object_vec![Canvas::default(), spline]
    }
}
