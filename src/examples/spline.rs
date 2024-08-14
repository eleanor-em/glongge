use glongge::core::{prelude::*, scene::{Scene, SceneName}};
use glongge::util::canvas::Canvas;
use glongge::util::spline::InteractiveSpline;
use crate::object_type::ObjectType;

#[allow(dead_code)]
#[derive(Copy, Clone)]
pub struct SplineScene;
impl Scene<ObjectType> for SplineScene {
    fn name(&self) -> SceneName { SceneName::new("spline") }

    fn create_objects(&self, _entrance_id: usize) -> Vec<AnySceneObject<ObjectType>> {
        vec![
            Canvas::create(),
            InteractiveSpline::create()
        ]
    }
}
