#[allow(unused_imports)]
use crate::core::prelude::*;

use std::{
    any::Any,
    fmt::Debug,
    ops::Range
};
use num_traits::{Float, Zero};
use crate::core::{
    linalg::Vec2,
    util::gg_range,
    Transform
};

pub enum ColliderType {
    Null,
    Box,
    Convex,
}

pub trait Collider: Debug {
    fn as_any(&self) -> &dyn Any;
    fn get_type(&self) -> ColliderType;

    fn collides_with_box(&self, other: &BoxCollider) -> Option<Vec2>;
    fn collides_with_convex(&self, other: &ConvexCollider) -> Option<Vec2>;

    fn collides_with(&self, other: &dyn Collider) -> Option<Vec2> {
        match other.get_type() {
            ColliderType::Null => None,
            ColliderType::Box => self.collides_with_box(other.as_any().downcast_ref().unwrap()),
            ColliderType::Convex => self.collides_with_convex(other.as_any().downcast_ref().unwrap())
        }
    }

    fn translate(&mut self, by: Vec2) -> &mut dyn Collider;
}

// impl Collider for Box<(dyn Collider + 'static)> {
//     fn as_any(&self) -> &dyn Any {
//         self.as_ref().as_any()
//     }
//
//     fn collides_with_box(&self, other: &BoxCollider) -> Option<Vec2> {
//         self.as_ref().collides_with_box(other)
//     }
//
//     fn collides_with(&self, other: &dyn Collider) -> Option<Vec2> {
//         self.as_ref().collides_with(other)
//     }
//
//     fn translate(&mut self, by: Vec2) -> &mut dyn Collider {
//         self.as_mut().translate(by)
//     }
//
//     fn translate_boxed(self: Box<dyn Collider>, by: Vec2) -> Box<dyn Collider> {
//         self.
//     }
// }

#[derive(Debug, Clone)]
pub struct NullCollider;
impl Collider for NullCollider {
    fn as_any(&self) -> &dyn Any { self }
    fn get_type(&self) -> ColliderType { ColliderType::Null }

    fn collides_with_convex(&self, _other: &ConvexCollider) -> Option<Vec2> { None }
    fn collides_with_box(&self, _other: &BoxCollider) -> Option<Vec2> { None }

    fn translate(&mut self, _by: Vec2) -> &mut dyn Collider { self }
}

fn project(vertices: &[Vec2], axis: Vec2) -> Range<f64> {
    let mut start = f64::max_value();
    let mut end = f64::min_value();
    for &vertex in vertices {
        let projection = axis.dot(vertex);
        start = start.min(projection);
        end = end.max(projection);
    }
    start..end
}

fn polygon_collision(
    this_vertices: &[Vec2],
    this_normals: &[Vec2],
    this_centre: Vec2,
    other_vertices: &[Vec2],
    other_normals: &[Vec2],
    other_centre: Vec2
) -> Option<Vec2> {
    let mut min_axis = Vec2::zero();
    let mut min_dist = f64::max_value();

    for &axis in this_normals.into_iter().chain(other_normals) {
        let self_proj = project(&this_vertices, axis);
        let other_proj = project(&other_vertices, axis);
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
    if this_centre.dot(min_axis) < other_centre.dot(min_axis) {
        Some(-mtv)
    } else {
        Some(mtv)
    }
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

    fn vertices(&self) -> Vec<Vec2> {
        vec![self.bottom_right(), self.top_right(), self.top_left(), self.bottom_left()]
    }
    fn normals(&self) -> Vec<Vec2> {
        vec![Vec2::right().rotated(self.rotation), Vec2::down().rotated(self.rotation)]
    }
}

impl Collider for BoxCollider {
    fn as_any(&self) -> &dyn Any { self }
    fn get_type(&self) -> ColliderType { ColliderType::Box }

    fn collides_with_box(&self, other: &BoxCollider) -> Option<Vec2> {
        polygon_collision(&self.vertices(), &self.normals(), self.centre,
                          &other.vertices(), &other.normals(), other.centre)
    }

    fn collides_with_convex(&self, other: &ConvexCollider) -> Option<Vec2> {
        polygon_collision(&self.vertices(), &self.normals(), self.centre,
                          other.vertices(), &other.normals(), other.centre())
    }

    fn translate(&mut self, by: Vec2) -> &mut dyn Collider {
        self.centre += by.rotated(self.rotation);
        self
    }
}

#[derive(Debug, Clone)]
pub struct ConvexCollider {
    vertices: Vec<Vec2>,
}

impl ConvexCollider {
    pub fn convex_hull_of(vertices: Vec<Vec2>) -> Self {
        todo!();
    }

    pub fn vertices(&self) -> &[Vec2] {
        &self.vertices
    }
    pub fn normals(&self) -> Vec<Vec2> {
        todo!();
    }
    pub fn centre(&self) -> Vec2 {
        todo!();
    }
}
