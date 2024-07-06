use std::ops::Range;
use num_traits::Float;
use crate::{
    core::linalg::Vec2,
    core::util::range
};
use crate::gg::Transform;

#[derive(Clone)]
pub struct BoxCollider {
    pub centre: Vec2,
    pub rotation: f64,
    pub extents: Vec2,
}
impl BoxCollider {
    pub fn new(transform: Transform, extents: Vec2) -> Self {
        Self {
            centre: transform.position,
            rotation: transform.rotation,
            extents,
        }
    }
    pub fn square(transform: Transform, size: f64) -> Self {
        Self::new(transform, size * Vec2::one())
    }

    pub fn half_widths(&self) -> Vec2 { self.extents / 2.0 }

    pub fn top_left(&self) -> Vec2 {
        self.centre + (-self.half_widths()).rotated(self.rotation)
    }
    pub fn top_right(&self) -> Vec2 {
        self.centre + Vec2 { x: self.half_widths().x, y: -self.half_widths().y }.rotated(self.rotation)
    }
    pub fn bottom_left(&self) -> Vec2 {
        self.centre + Vec2 { x: -self.half_widths().x, y: self.half_widths().y }.rotated(self.rotation)
    }
    pub fn bottom_right(&self) -> Vec2 {
        self.centre + self.half_widths().rotated(self.rotation)
    }

    pub fn collides_with(&self, other: &BoxCollider) -> Option<Vec2> {
        let mut mtv = f64::max_value() * Vec2::right();

        for axis in [self.normals(), other.normals()].into_iter().flatten() {
            let self_proj = self.project(axis);
            let other_proj = other.project(axis);
            match range::overlap_len_f64(&self_proj, &other_proj) {
                Some(0.0) => return None,
                Some(mut dist) => {
                    if dist == 0.0 { return None; }
                    if range::contains_f64(&self_proj, &other_proj) ||
                        range::contains_f64(&other_proj, &self_proj) {
                        let starts = (self_proj.start - other_proj.start).abs();
                        let ends = (self_proj.end - other_proj.end).abs();
                        dist += f64::min(starts, ends);
                    }
                    if dist < mtv.len() {
                        mtv = dist * axis;
                    }
                },
                _ => return None,
            }
        }

        if self.centre.dot(mtv) < other.centre.dot(mtv) {
            Some(-mtv)
        } else {
            Some(mtv)
        }
    }

    fn vertices(&self) -> Vec<Vec2> { vec![self.bottom_right(), self.top_right(), self.top_left(), self.bottom_left()] }
    fn normals(&self) -> Vec<Vec2> {
        vec![Vec2::right().rotated(self.rotation), Vec2::down().rotated(self.rotation)]
    }
    fn project(&self, axis: Vec2) -> Range<f64> {
        let mut min = f64::max_value();
        let mut max = f64::min_value();
        for vertex in self.vertices() {
            let proj = axis.dot(vertex);
            min = f64::min(min, proj);
            max = f64::max(max, proj);
        }
        min..max
    }
}
