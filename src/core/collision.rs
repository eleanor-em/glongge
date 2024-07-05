use num_traits::Float;
use crate::core::linalg::{Mat3x3, Vec2};

#[derive(Clone)]
pub struct BoxCollider {
    pub centre: Vec2,
    pub extents: Vec2,
    pub rotation: f64,
}
impl BoxCollider {
    pub fn top_left(&self) -> Vec2 {
        self.centre + Mat3x3::rotation(self.rotation) * -self.extents / 2.0
    }
    pub fn top_right(&self) -> Vec2 {
        self.centre + Mat3x3::rotation(self.rotation) *
            Vec2 { x: self.extents.x / 2.0, y: -self.extents.y / 2.0 }
    }
    pub fn bottom_left(&self) -> Vec2 {
        self.centre + Mat3x3::rotation(self.rotation) *
            Vec2 { x: -self.extents.x / 2.0, y: self.extents.y / 2.0 }
    }
    pub fn bottom_right(&self) -> Vec2 {
        self.centre + Mat3x3::rotation(self.rotation) *
            Vec2 { x: self.extents.x / 2.0, y: self.extents.y / 2.0 }
    }

    pub fn collides_with(&self, other: &BoxCollider) -> bool {
        for axis in [self.normals(), other.normals()].into_iter().flatten() {
            let mut self_min_along = f64::max_value();
            let mut self_max_along = f64::min_value();
            for pt in self.corners() {
                let proj = pt.dot(axis);
                self_min_along = f64::min(self_min_along, proj);
                self_max_along = f64::max(self_max_along, proj);
            }
            let mut other_min_along = f64::max_value();
            let mut other_max_along = f64::min_value();
            for pt in other.corners() {
                let proj = pt.dot(axis);
                other_min_along = f64::min(other_min_along, proj);
                other_max_along = f64::max(other_max_along, proj);
            }
            if !(self_min_along..self_max_along).contains(&other_min_along) &&
                    !(other_min_along..other_max_along).contains(&self_min_along) {
                return false;
            }
        }
        true
    }

    fn corners(&self) -> Vec<Vec2> { vec![self.top_left(), self.top_right(), self.bottom_right(), self.bottom_left()] }
    fn normals(&self) -> Vec<Vec2> {
        vec![Mat3x3::rotation(self.rotation) * Vec2::up(),
             Mat3x3::rotation(self.rotation) * Vec2::down(),
             Mat3x3::rotation(self.rotation) * Vec2::left(),
             Mat3x3::rotation(self.rotation) * Vec2::right()]
    }
}
