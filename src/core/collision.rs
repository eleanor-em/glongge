use std::{
    any::Any,
    fmt::Debug,
    ops::Range
};
use num_traits::{Float, Zero};
use crate::{
    core::{
        prelude::*,
        linalg::{
            Vec2,
            AxisAlignedExtent
        },
        util::{
            gg_range,
            gg_iter
        },
        Transform,
    }
};

pub enum ColliderType {
    Null,
    Box,
    Convex,
}

pub trait Collider: AxisAlignedExtent + Debug {
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
impl AxisAlignedExtent for NullCollider {
    fn extent(&self) -> Vec2 { Vec2::zero() }

    fn centre(&self) -> Vec2 { Vec2::zero() }
}
impl Collider for NullCollider {
    fn as_any(&self) -> &dyn Any { self }
    fn get_type(&self) -> ColliderType { ColliderType::Null }

    fn collides_with_box(&self, _other: &BoxCollider) -> Option<Vec2> { None }
    fn collides_with_convex(&self, _other: &ConvexCollider) -> Option<Vec2> { None }

    fn translate(&mut self, _by: Vec2) -> &mut dyn Collider { self }
}

trait Polygonal {
    fn vertices(&self) -> Vec<Vec2>;
    fn normals(&self) -> Vec<Vec2>;
    fn polygon_centre(&self) -> Vec2;

    fn project(&self, axis: Vec2) -> Range<f64> {
        let mut start = f64::max_value();
        let mut end = f64::min_value();
        for vertex in self.vertices() {
            let projection = axis.dot(vertex);
            start = start.min(projection);
            end = end.max(projection);
        }
        start..end
    }
    fn collision<P: Polygonal>(&self, other: P) -> Option<Vec2> {
        let mut min_axis = Vec2::zero();
        let mut min_dist = f64::max_value();

        for axis in self.normals().into_iter().chain(other.normals()) {
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
        if self.polygon_centre().dot(min_axis) < other.polygon_centre().dot(min_axis) {
            Some(-mtv)
        } else {
            Some(mtv)
        }
    }

    fn normals_of(mut vertices: Vec<Vec2>) -> Vec<Vec2> {
        vertices.push(*vertices.first().unwrap());
        vertices.windows(2)
            .map(|vs| {
                let dx = vs[1].x - vs[0].x;
                let dy = vs[1].y - vs[0].y;
                Vec2 { x: dy, y: -dx }
            })
            .collect()
    }
    fn centre_of(mut vertices: Vec<Vec2>) -> Vec2 {
        if let Some(vertex) = vertices.first() {
            vertices.push(*vertex);
            let (area, x, y) = vertices.windows(2)
                .map(|vs| {
                    let u = vs[0];
                    let v = vs[1];
                    let area = u.cross(v);
                    (area, (u.x + v.x) * area, (u.y + v.y) * area)
                })
                .reduce(gg_iter::sum_tuple3)
                .expect("should be unreachable");
            Vec2 {
                x: x / (6. * (area / 2.)),
                y: y / (6. * (area / 2.)),
            }
        } else {
            Vec2::zero()
        }
    }
}

impl<T: Polygonal> Polygonal for &T {
    fn vertices(&self) -> Vec<Vec2> {
        (*self).vertices()
    }

    fn normals(&self) -> Vec<Vec2> {
        (*self).vertices()
    }

    fn polygon_centre(&self) -> Vec2 {
        (*self).polygon_centre()
    }
}

impl<T: Polygonal> AxisAlignedExtent for T {
    fn extent(&self) -> Vec2 {
        let mut min_x = f64::MAX;
        let mut min_y = f64::MAX;
        let mut max_x = f64::MIN;
        let mut max_y = f64::MIN;
        for vertex in self.vertices() {
            min_x = vertex.x.min(min_x);
            min_y = vertex.y.min(min_y);
            max_x = vertex.x.max(max_x);
            max_y = vertex.y.max(max_y);
        }
        Vec2 { x: max_x - min_x, y: max_y - min_y }
    }

    fn centre(&self) -> Vec2 {
        self.polygon_centre()
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

    fn top_left_rotated(&self) -> Vec2 {
        self.centre + (-self.half_widths).rotated(self.rotation)
    }
    fn top_right_rotated(&self) -> Vec2 {
        self.centre + Vec2 { x: self.half_widths.x, y: -self.half_widths.y }.rotated(self.rotation)
    }
    fn bottom_left_rotated(&self) -> Vec2 {
        self.centre + Vec2 { x: -self.half_widths.x, y: self.half_widths.y }.rotated(self.rotation)
    }
    fn bottom_right_rotated(&self) -> Vec2 {
        self.centre + self.half_widths.rotated(self.rotation)
    }

}

impl Polygonal for BoxCollider {
    fn vertices(&self) -> Vec<Vec2> {
        vec![
            self.bottom_right_rotated(), self.top_right_rotated(), self.top_left_rotated(), self.bottom_left_rotated()
        ]
    }
    fn normals(&self) -> Vec<Vec2> {
        vec![Vec2::right().rotated(self.rotation), Vec2::down().rotated(self.rotation)]
    }

    fn polygon_centre(&self) -> Vec2 {
        self.centre
    }
}

impl Collider for BoxCollider {
    fn as_any(&self) -> &dyn Any { self }
    fn get_type(&self) -> ColliderType { ColliderType::Box }

    fn collides_with_box(&self, other: &BoxCollider) -> Option<Vec2> {
        self.collision(other)
    }

    fn collides_with_convex(&self, other: &ConvexCollider) -> Option<Vec2> {
        self.collision(other)
    }

    fn translate(&mut self, by: Vec2) -> &mut dyn Collider {
        self.centre += by.rotated(self.rotation);
        self
    }
}

#[derive(Debug, Clone)]
pub struct ConvexCollider {
    vertices: Vec<Vec2>,
    normals: Vec<Vec2>,
    centre: Vec2,
}

impl ConvexCollider {
    fn hull<I: Iterator<Item=Vec2>>(vertices: I) -> Vec<Vec2> {
        let mut hull: Vec<Vec2> = Vec::new();
        for vertex in vertices {
            while hull.len() >= 2 {
                let last = hull[hull.len() - 1];
                let snd_last = hull[hull.len() - 2];
                if (last - snd_last).cross(vertex - snd_last) > 0. {
                    break;
                }
                hull.pop();
            }
            hull.push(vertex);
        }
        hull
    }

    fn from_vertices_unchecked(vertices: Vec<Vec2>) -> Self {
        // Does not check that the vertices are convex.
        let normals = Self::normals_of(vertices.clone());
        let centre = Self::centre_of(vertices.clone());
        Self { vertices, normals, centre }
    }

    pub fn convex_hull_of(mut vertices: Vec<Vec2>) -> Self {
        vertices.sort_unstable_by(|u, v| u.partial_cmp(v).unwrap());
        if vertices.len() <= 1 {
            return Self::from_vertices_unchecked(vertices);
        }

        let mut lower = Self::hull(vertices.iter().copied());
        let mut upper = Self::hull(vertices.into_iter().rev());
        check_eq!(lower.last().unwrap(), upper.first().unwrap());
        check_eq!(lower.first().unwrap(), upper.last().unwrap());
        lower.pop();
        upper.pop();

        let vertices = lower.into_iter().chain(upper).collect_vec();
        Self::from_vertices_unchecked(vertices)
    }
}

impl Polygonal for ConvexCollider {
    fn vertices(&self) -> Vec<Vec2> {
        self.vertices.clone()
    }

    fn normals(&self) -> Vec<Vec2> {
        self.normals.clone()
    }

    fn polygon_centre(&self) -> Vec2 {
        self.centre
    }
}

impl Collider for ConvexCollider {
    fn as_any(&self) -> &dyn Any { self }

    fn get_type(&self) -> ColliderType { ColliderType::Convex }

    fn collides_with_box(&self, other: &BoxCollider) -> Option<Vec2> {
        self.collision(other)
    }

    fn collides_with_convex(&self, other: &ConvexCollider) -> Option<Vec2> {
        self.collision(other)
    }

    fn translate(&mut self, by: Vec2) -> &mut dyn Collider {
        for vertex in &mut self.vertices {
            *vertex += by;
        }
        self
    }
}
