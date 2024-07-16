use std::ops::{Neg, Range};
use num_traits::{Float, Zero};
use crate::{
    core::{
        linalg::Vec2,
        util::gg_range
    },
    gg::Transform
};

pub trait Collider {
    fn collides_with_box(&self, other: &BoxCollider) -> Option<Vec2>;
    fn collides_with(&self, other: &dyn Collider) -> Option<Vec2>;
}

#[derive(Clone)]
pub struct BoxCollider {
    pub centre: Vec2,
    pub rotation: f64,
    pub half_widths: Vec2,
}
impl BoxCollider {
    pub fn new(transform: Transform, half_widths: Vec2) -> Self {
        Self {
            centre: transform.position,
            rotation: transform.rotation,
            half_widths,
        }
    }
    pub fn square(transform: Transform, half_width: f64) -> Self {
        Self::new(transform, half_width * Vec2::one())
    }

    pub fn top_left(&self) -> Vec2 {
        self.centre + (-self.half_widths).rotated(self.rotation)
    }
    pub fn top_right(&self) -> Vec2 {
        self.centre + Vec2 { x: self.half_widths.x, y: -self.half_widths.y }.rotated(self.rotation)
    }
    pub fn bottom_left(&self) -> Vec2 {
        self.centre + Vec2 { x: -self.half_widths.x, y: self.half_widths.y }.rotated(self.rotation)
    }
    pub fn bottom_right(&self) -> Vec2 {
        self.centre + self.half_widths.rotated(self.rotation)
    }

    fn vertices(&self) -> [Vec2; 4] {
        [self.bottom_right(), self.top_right(), self.top_left(), self.bottom_left()]
    }
    fn normals(&self) -> [Vec2; 2] {
        [Vec2::right().rotated(self.rotation), Vec2::down().rotated(self.rotation)]
    }
    fn project(&self, axis: Vec2) -> Range<f64> {
        let mut start = f64::max_value();
        let mut end = f64::min_value();
        for projection in self.vertices().map(|vertex| axis.dot(vertex)) {
            start = start.min(projection);
            end = end.max(projection);
        }
        start..end
    }
}

impl Collider for BoxCollider {
    fn collides_with_box(&self, other: &BoxCollider) -> Option<Vec2> {
        let mut min_axis = Vec2::zero();
        let mut min_dist = f64::max_value();

        for &axis in [self.normals(), other.normals()].as_flattened() {
            let self_proj = self.project(axis);
            let other_proj = other.project(axis);
            match gg_range::overlap_len_f64(&self_proj, &other_proj) {
                Some(0.0) => return None,
                Some(mut dist) => {
                    if gg_range::contains_f64(&self_proj, &other_proj) ||
                            gg_range::contains_f64(&other_proj, &self_proj) {
                        let starts = (self_proj.start - other_proj.start).abs();
                        let ends = (self_proj.end - other_proj.end).abs();
                        dist += f64::min(starts, ends);
                    }
                    if dist < min_dist {
                        min_dist = dist;
                        min_axis = axis;
                    }
                },
                _ => return None,
            }
        }

        let mtv = min_dist * min_axis;
        if self.centre.dot(min_axis) < other.centre.dot(min_axis) {
            Some(-mtv)
        } else {
            Some(mtv)
        }
    }

    fn collides_with(&self, other: &dyn Collider) -> Option<Vec2> {
        other.collides_with_box(self).map(Vec2::neg)
    }
}
