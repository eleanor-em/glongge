#[allow(unused_imports)]
use crate::core::prelude::*;

use std::{
    any::Any,
    fmt::Debug,
    ops::{Neg, Range}
};
use num_traits::{Float, Zero};
use crate::core::{
    linalg::Vec2,
    util::gg_range,
    Transform
};

pub trait Collider: Debug {
    fn as_any(&self) -> &dyn Any;

    fn collides_with_box(&self, other: &BoxCollider) -> Option<Vec2>;
    fn collides_with(&self, other: &dyn Collider) -> Option<Vec2>;

    fn translated(&self, by: Vec2) -> Box<dyn Collider>;
}

impl Collider for Box<(dyn Collider + 'static)> {
    fn as_any(&self) -> &dyn Any {
        self.as_ref().as_any()
    }

    fn collides_with_box(&self, other: &BoxCollider) -> Option<Vec2> {
        self.as_ref().collides_with_box(other)
    }

    fn collides_with(&self, other: &dyn Collider) -> Option<Vec2> {
        self.as_ref().collides_with(other)
    }

    fn translated(&self, by: Vec2) -> Box<dyn Collider> {
        self.as_ref().translated(by)
    }
}

#[derive(Debug, Clone)]
pub struct NullCollider;
impl Collider for NullCollider {
    fn as_any(&self) -> &dyn Any { self }

    fn collides_with_box(&self, _other: &BoxCollider) -> Option<Vec2> { None }
    fn collides_with(&self, _other: &dyn Collider) -> Option<Vec2> { None }

    fn translated(&self, _by: Vec2) -> Box<dyn Collider> { Box::new(Self) }
}

#[derive(Debug, Clone)]
pub struct BoxCollider {
    pub centre: Vec2,
    pub rotation: f64,
    pub half_widths: Vec2,
}
impl BoxCollider {
    pub fn from_centre(centre: Vec2, half_widths: Vec2) -> Self {
        Self {
            centre,
            rotation: 0.,
            half_widths,
        }
    }
    pub fn from_top_left(top_left: Vec2, extent: Vec2) -> Self {
        Self {
            centre: top_left + extent.abs() / 2,
            rotation: 0.,
            half_widths: extent.abs() / 2,
        }
    }
    pub fn from_transform(transform: Transform, half_widths: Vec2) -> Self {
        Self {
            centre: transform.centre,
            rotation: transform.rotation,
            half_widths: transform.scale.component_wise(half_widths).abs(),
        }
    }
    pub fn square(transform: Transform, width: f64) -> Self {
        Self::from_transform(transform, width.abs() * Vec2::one())
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
    fn as_any(&self) -> &dyn Any { self }

    fn collides_with_box(&self, other: &BoxCollider) -> Option<Vec2> {
        let mut min_axis = Vec2::zero();
        let mut min_dist = f64::max_value();

        for &axis in [self.normals(), other.normals()].as_flattened() {
            let self_proj = self.project(axis);
            let other_proj = other.project(axis);
            match gg_range::overlap_len_f64(&self_proj, &other_proj) {
                Some(0.) => return None,
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
        if let Some(other) = other.as_any().downcast_ref::<BoxCollider>() {
            other.collides_with_box(self).map(Vec2::neg)
        } else if other.as_any().downcast_ref::<NullCollider>().is_some() {
            None
        } else {
            unreachable!();
        }
    }

    fn translated(&self, by: Vec2) -> Box<dyn Collider> {
        let mut rv = self.clone();
        rv.centre += by.rotated(self.rotation);
        Box::new(rv)
    }
}
