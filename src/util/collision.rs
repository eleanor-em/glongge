use crate::core::scene::{GuiInsideClosure, GuiObject};
use crate::util::{UnorderedPair, gg_iter};
use crate::{
    core::{ObjectTypeEnum, prelude::*, scene::SceneObject},
    resource::sprite::Sprite,
    util::{
        gg_range,
        linalg::{AxisAlignedExtent, Transform, Vec2},
    },
};
use glongge_derive::{partially_derive_scene_object, register_scene_object};
use num_traits::{Float, Zero};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Display, Formatter};
use std::{any::Any, fmt::Debug, ops::Range};

#[derive(Debug)]
pub enum ColliderType {
    Null,
    Box,
    OrientedBox,
    Convex,
    Compound,
}

pub trait Collider: AxisAlignedExtent + Debug + Send + Sync + 'static {
    fn as_any(&self) -> &dyn Any;
    fn get_type(&self) -> ColliderType;

    fn collides_with_box(&self, other: &BoxCollider) -> Option<Vec2>;
    fn collides_with_oriented_box(&self, other: &OrientedBoxCollider) -> Option<Vec2>;
    fn collides_with_convex(&self, other: &ConvexCollider) -> Option<Vec2>;

    fn collides_with(&self, other: &GenericCollider) -> Option<Vec2> {
        match other.get_type() {
            ColliderType::Null => None,
            ColliderType::Box => self.collides_with_box(other.as_any().downcast_ref()?),
            ColliderType::OrientedBox => {
                self.collides_with_oriented_box(other.as_any().downcast_ref()?)
            }
            ColliderType::Convex => self.collides_with_convex(other.as_any().downcast_ref()?),
            ColliderType::Compound => match self.get_type() {
                ColliderType::Null => None,
                ColliderType::Box => other.collides_with_box(self.as_any().downcast_ref()?),
                ColliderType::OrientedBox => {
                    other.collides_with_oriented_box(self.as_any().downcast_ref()?)
                }
                ColliderType::Convex => other.collides_with_convex(self.as_any().downcast_ref()?),
                ColliderType::Compound => {
                    let this = self.as_any().downcast_ref::<CompoundCollider>()?;
                    this.inner_colliders()
                        .into_iter()
                        .filter_map(|c| other.collides_with_convex(&c))
                        .filter(|&mtv| !this.is_internal_mtv(other, mtv))
                        .min_by(Vec2::cmp_by_length)
                }
            }
            .map(|v| -v),
        }
    }

    fn as_generic(&self) -> GenericCollider
    where
        Self: Clone,
    {
        self.clone().into_generic()
    }
    fn into_generic(self) -> GenericCollider
    where
        Self: Sized + 'static;

    #[must_use]
    fn translated(&self, by: Vec2) -> Self;
    #[must_use]
    fn scaled(&self, by: Vec2) -> Self;
    #[must_use]
    fn rotated(&self, by: f32) -> Self;
    #[must_use]
    fn transformed(&self, by: &Transform) -> Self
    where
        Self: Sized,
    {
        self.translated(by.centre)
            .scaled(by.scale)
            .rotated(by.rotation)
    }
    #[must_use]
    fn with_half_widths(&self, half_widths: Vec2) -> Self
    where
        Self: Sized,
    {
        self.scaled(half_widths.component_wise_div(self.half_widths()))
    }
    #[must_use]
    fn with_extent(&self, extent: Vec2) -> Self
    where
        Self: Sized,
    {
        self.scaled(extent.component_wise_div(self.aa_extent()))
    }

    #[must_use]
    fn with_centre(&self, centre: Vec2) -> Self
    where
        Self: Sized,
    {
        self.translated(centre - self.centre())
    }

    fn as_polygon(&self) -> Vec<Vec2>;
    fn as_triangles(&self) -> Vec<[Vec2; 3]>;
}

#[derive(Debug, Clone, Copy)]
pub struct NullCollider;
impl AxisAlignedExtent for NullCollider {
    fn aa_extent(&self) -> Vec2 {
        Vec2::zero()
    }

    fn centre(&self) -> Vec2 {
        Vec2::zero()
    }
}
impl Collider for NullCollider {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn get_type(&self) -> ColliderType {
        ColliderType::Null
    }

    fn collides_with_box(&self, _other: &BoxCollider) -> Option<Vec2> {
        None
    }
    fn collides_with_oriented_box(&self, _other: &OrientedBoxCollider) -> Option<Vec2> {
        None
    }
    fn collides_with_convex(&self, _other: &ConvexCollider) -> Option<Vec2> {
        None
    }

    fn into_generic(self) -> GenericCollider
    where
        Self: Sized + Send + Sync + 'static,
    {
        GenericCollider::Null
    }

    fn translated(&self, _by: Vec2) -> Self {
        NullCollider
    }
    fn scaled(&self, _by: Vec2) -> Self {
        NullCollider
    }
    fn rotated(&self, _by: f32) -> Self {
        NullCollider
    }

    // By convention, clockwise edges starting from the top-leftmost vertex.
    fn as_polygon(&self) -> Vec<Vec2> {
        Vec::new()
    }
    fn as_triangles(&self) -> Vec<[Vec2; 3]> {
        Vec::new()
    }
}

mod polygon {
    use crate::core::prelude::*;
    use crate::util::{gg_iter, gg_range};
    use itertools::Itertools;
    use num_traits::Zero;
    use std::ops::Range;
    use tracing::warn;

    pub fn hull<I: Iterator<Item = Vec2>>(vertices: I) -> Vec<Vec2> {
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
    pub fn adjust_for_containment(self_proj: &Range<f32>, other_proj: &Range<f32>) -> f32 {
        if gg_range::contains_f32(self_proj, other_proj)
            || gg_range::contains_f32(other_proj, self_proj)
        {
            let starts = (self_proj.start - other_proj.start).abs();
            let ends = (self_proj.end - other_proj.end).abs();
            f32::min(starts, ends)
        } else {
            0.
        }
    }
    pub fn normals_of(mut vertices: Vec<Vec2>) -> Vec<Vec2> {
        if let Some(first) = vertices.first() {
            vertices.push(*first);
            vertices
                .iter()
                .tuple_windows()
                .map(|(u, v)| (*v - *u).orthog().normed())
                .collect()
        } else {
            warn!("asked for normals of empty vertex set");
            Vec::new()
        }
    }
    pub fn centre_of(mut vertices: Vec<Vec2>) -> Vec2 {
        if let Some(vertex) = vertices.first() {
            vertices.push(*vertex);
            let (area, x, y) = vertices
                .iter()
                .tuple_windows()
                .map(|(&u, &v)| {
                    let area = u.cross(v);
                    (area, (u.x + v.x) * area, (u.y + v.y) * area)
                })
                .reduce(gg_iter::sum_tuple3)
                .expect("inexplicable");
            Vec2 {
                x: x / (6. * (area / 2.)),
                y: y / (6. * (area / 2.)),
            }
        } else {
            Vec2::zero()
        }
    }
    pub fn extent_of(vertices: Vec<Vec2>) -> Vec2 {
        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        let mut max_x = f32::MIN;
        let mut max_y = f32::MIN;
        for vertex in vertices {
            min_x = vertex.x.min(min_x);
            min_y = vertex.y.min(min_y);
            max_x = vertex.x.max(max_x);
            max_y = vertex.y.max(max_y);
        }
        Vec2 {
            x: max_x - min_x,
            y: max_y - min_y,
        }
    }
    pub fn is_convex(vertices: &[Vec2]) -> bool {
        vertices
            .iter()
            .circular_tuple_windows()
            .map(|(&u, &v, &w)| {
                let d1 = v - u;
                let d2 = w - v;
                d1.cross(d2).signum()
            })
            .all_equal()
    }
}

pub trait Polygonal {
    fn vertices(&self) -> Vec<Vec2>;
    fn normals(&self) -> Vec<Vec2>;
    fn polygon_centre(&self) -> Vec2;

    fn project(&self, axis: Vec2) -> Range<f32> {
        let mut start = f32::max_value();
        let mut end = f32::min_value();
        for vertex in self.vertices() {
            let projection = axis.dot(vertex);
            start = start.min(projection);
            end = end.max(projection);
        }
        start..end
    }

    fn polygon_collision<P: Polygonal>(&self, other: P) -> Option<Vec2> {
        let mut min_axis = Vec2::zero();
        let mut min_dist = f32::max_value();

        let mut all_normals = BTreeSet::new();
        all_normals.extend(self.normals());
        all_normals.extend(other.normals());
        for axis in all_normals.iter().copied() {
            let self_proj = self.project(axis);
            let other_proj = other.project(axis);
            match gg_range::overlap_len_f32(&self_proj, &other_proj) {
                None => return None,
                Some(mut dist) => {
                    if dist.abs() < EPSILON {
                        return None;
                    }
                    dist += polygon::adjust_for_containment(&self_proj, &other_proj);
                    if dist < min_dist {
                        min_dist = dist;
                        min_axis = axis;
                    }
                }
            }
        }

        let mtv = min_dist * min_axis;
        check!(
            all_normals.contains(&min_axis),
            format!("{all_normals:?}, {min_axis}")
        );
        if self.polygon_centre().dot(min_axis) < other.polygon_centre().dot(min_axis) {
            Some(-mtv)
        } else {
            Some(mtv)
        }
    }

    fn draw_polygonal(&self, canvas: &mut Canvas, col: Colour) {
        let normals = self.normals();
        let vertices = self.vertices();
        for (start, end) in normals
            .into_iter()
            .zip(vertices.iter().circular_tuple_windows())
            .map(|(normal, (u, v))| {
                let start = (*u + *v) / 2;
                let end = start + normal.normed() * 8;
                (start, end)
            })
        {
            canvas.line(start, end, 1., col);
            canvas.rect(
                self.polygon_centre() - Vec2::one(),
                self.polygon_centre() + Vec2::one(),
                col,
            );
        }
        for (a, b) in vertices.into_iter().circular_tuple_windows() {
            canvas.line(a, b, 1., col);
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

#[derive(Debug, Clone)]
pub struct OrientedBoxCollider {
    centre: Vec2,
    rotation: f32,
    axis_aligned_half_widths: Vec2,
    extent: Vec2,
}
impl OrientedBoxCollider {
    pub fn from_centre(centre: Vec2, half_widths: Vec2) -> Self {
        let mut rv = Self {
            centre,
            rotation: 0.,
            axis_aligned_half_widths: half_widths,
            extent: Vec2::zero(),
        };
        rv.extent = polygon::extent_of(rv.vertices());
        rv
    }
    pub fn from_top_left(top_left: Vec2, extent: Vec2) -> Self {
        let mut rv = Self {
            centre: top_left + extent.abs() / 2,
            rotation: 0.,
            axis_aligned_half_widths: extent.abs() / 2,
            extent: Vec2::zero(),
        };
        rv.extent = polygon::extent_of(rv.vertices());
        rv
    }
    pub fn from_transform(transform: Transform, half_widths: Vec2) -> Self {
        let mut rv = Self {
            centre: transform.centre,
            rotation: transform.rotation,
            axis_aligned_half_widths: transform.scale.component_wise(half_widths).abs(),
            extent: Vec2::zero(),
        };
        rv.extent = polygon::extent_of(rv.vertices());
        rv
    }
    pub fn square(transform: Transform, width: f32) -> Self {
        Self::from_transform(transform, width.abs() * Vec2::one())
    }

    pub fn top_left_rotated(&self) -> Vec2 {
        self.centre + (-self.axis_aligned_half_widths).rotated(self.rotation)
    }
    pub fn top_right_rotated(&self) -> Vec2 {
        self.centre
            + Vec2 {
                x: self.axis_aligned_half_widths.x,
                y: -self.axis_aligned_half_widths.y,
            }
            .rotated(self.rotation)
    }
    pub fn bottom_left_rotated(&self) -> Vec2 {
        self.centre
            + Vec2 {
                x: -self.axis_aligned_half_widths.x,
                y: self.axis_aligned_half_widths.y,
            }
            .rotated(self.rotation)
    }
    pub fn bottom_right_rotated(&self) -> Vec2 {
        self.centre + self.axis_aligned_half_widths.rotated(self.rotation)
    }
}
impl Polygonal for OrientedBoxCollider {
    fn vertices(&self) -> Vec<Vec2> {
        vec![
            self.top_left_rotated(),
            self.top_right_rotated(),
            self.bottom_right_rotated(),
            self.bottom_left_rotated(),
        ]
    }
    fn normals(&self) -> Vec<Vec2> {
        vec![
            Vec2::up().rotated(self.rotation),
            Vec2::right().rotated(self.rotation),
        ]
    }

    fn polygon_centre(&self) -> Vec2 {
        self.centre
    }
}

impl AxisAlignedExtent for OrientedBoxCollider {
    fn aa_extent(&self) -> Vec2 {
        self.extent
    }

    fn centre(&self) -> Vec2 {
        self.centre
    }
}

impl Collider for OrientedBoxCollider {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn get_type(&self) -> ColliderType {
        ColliderType::OrientedBox
    }

    fn collides_with_box(&self, other: &BoxCollider) -> Option<Vec2> {
        self.polygon_collision(other)
    }

    fn collides_with_oriented_box(&self, other: &OrientedBoxCollider) -> Option<Vec2> {
        self.polygon_collision(other)
    }

    fn collides_with_convex(&self, other: &ConvexCollider) -> Option<Vec2> {
        self.polygon_collision(other)
    }

    fn into_generic(self) -> GenericCollider
    where
        Self: Sized + 'static,
    {
        GenericCollider::OrientedBox(self)
    }

    fn translated(&self, by: Vec2) -> Self {
        let mut rv = self.clone();
        rv.centre += by.rotated(self.rotation);
        rv
    }

    fn scaled(&self, by: Vec2) -> Self {
        let mut rv = self.clone();
        rv.axis_aligned_half_widths = self.axis_aligned_half_widths.component_wise(by).abs();
        rv
    }

    fn rotated(&self, by: f32) -> Self {
        let mut rv = self.clone();
        rv.rotation += by;
        rv
    }

    fn as_polygon(&self) -> Vec<Vec2> {
        self.vertices()
    }

    fn as_triangles(&self) -> Vec<[Vec2; 3]> {
        vec![
            [
                self.top_left_rotated(),
                self.top_right_rotated(),
                self.bottom_left_rotated(),
            ],
            [
                self.top_right_rotated(),
                self.bottom_right_rotated(),
                self.bottom_left_rotated(),
            ],
        ]
    }
}
#[derive(Debug, Default, Clone)]
pub struct BoxCollider {
    centre: Vec2,
    extent: Vec2,
}
impl BoxCollider {
    #[must_use]
    pub fn from_centre(centre: Vec2, half_widths: Vec2) -> Self {
        Self {
            centre,
            extent: half_widths.abs() * 2,
        }
    }
    #[must_use]
    pub fn from_top_left(top_left: Vec2, extent: Vec2) -> Self {
        Self {
            centre: top_left + extent.abs() / 2,
            extent: extent.abs(),
        }
    }
    #[must_use]
    pub fn from_transform(transform: Transform, extent: Vec2) -> Self {
        check_eq!(transform.rotation, 0.);
        Self {
            centre: transform.centre,
            extent: transform.scale.component_wise(extent).abs(),
        }
    }

    #[must_use]
    pub fn square(transform: Transform, width: f32) -> Self {
        Self::from_transform(transform, width.abs() * Vec2::one())
    }

    pub fn as_convex(&self) -> ConvexCollider {
        ConvexCollider::from_vertices_unchecked(self.vertices())
    }
}

impl Polygonal for BoxCollider {
    fn vertices(&self) -> Vec<Vec2> {
        vec![
            self.top_left(),
            self.top_right(),
            self.bottom_right(),
            self.bottom_left(),
        ]
    }
    fn normals(&self) -> Vec<Vec2> {
        vec![Vec2::up(), Vec2::right()]
    }

    fn polygon_centre(&self) -> Vec2 {
        self.centre
    }
}

impl AxisAlignedExtent for BoxCollider {
    fn aa_extent(&self) -> Vec2 {
        self.extent
    }

    fn centre(&self) -> Vec2 {
        self.centre
    }
}

impl Collider for BoxCollider {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn get_type(&self) -> ColliderType {
        ColliderType::Box
    }

    fn collides_with_box(&self, other: &BoxCollider) -> Option<Vec2> {
        let self_proj = self.left()..self.right();
        let other_proj = other.left()..other.right();
        let right_dist = match gg_range::overlap_len_f32(&self_proj, &other_proj) {
            None | Some(0.) => return None,
            Some(dist) => dist + polygon::adjust_for_containment(&self_proj, &other_proj),
        };

        let self_proj = self.top()..self.bottom();
        let other_proj = other.top()..other.bottom();
        match gg_range::overlap_len_f32(&self_proj, &other_proj) {
            None | Some(0.) => None,
            Some(mut dist) => {
                dist += polygon::adjust_for_containment(&self_proj, &other_proj);
                if dist < right_dist {
                    // Collision along vertical axis.
                    let mtv = dist * Vec2::down();
                    if self.centre.y < other.centre.y {
                        Some(-mtv)
                    } else {
                        Some(mtv)
                    }
                } else {
                    // Collision along horizontal axis.
                    let mtv = right_dist * Vec2::right();
                    if self.centre.x < other.centre.x {
                        Some(-mtv)
                    } else {
                        Some(mtv)
                    }
                }
            }
        }
    }

    fn collides_with_oriented_box(&self, other: &OrientedBoxCollider) -> Option<Vec2> {
        self.polygon_collision(other)
    }

    fn collides_with_convex(&self, other: &ConvexCollider) -> Option<Vec2> {
        self.polygon_collision(other)
    }

    fn into_generic(self) -> GenericCollider
    where
        Self: Sized + 'static,
    {
        GenericCollider::Box(self)
    }

    fn translated(&self, by: Vec2) -> Self {
        let mut rv = self.clone();
        rv.centre += by;
        rv
    }

    fn scaled(&self, by: Vec2) -> Self {
        let mut rv = self.clone();
        rv.extent = self.extent.component_wise(by).abs();
        rv
    }

    fn rotated(&self, by: f32) -> Self {
        check_eq!(by, 0., "cannot rotate BoxCollider");
        self.clone()
    }

    fn as_polygon(&self) -> Vec<Vec2> {
        self.vertices()
    }

    fn as_triangles(&self) -> Vec<[Vec2; 3]> {
        vec![
            [self.top_left(), self.top_right(), self.bottom_left()],
            [self.top_right(), self.bottom_right(), self.bottom_left()],
        ]
    }
}

#[derive(Debug, Clone)]
pub struct ConvexCollider {
    vertices: Vec<Vec2>,
    normals_cached: Vec<Vec2>,
    centre_cached: Vec2,
    extent_cached: Vec2,
}

impl ConvexCollider {
    fn from_vertices_unchecked(vertices: Vec<Vec2>) -> Self {
        // Does not check that the vertices are convex.
        let normals = polygon::normals_of(vertices.clone());
        let centre = polygon::centre_of(vertices.clone());
        let extent = polygon::extent_of(vertices.clone());
        Self {
            vertices,
            normals_cached: normals,
            centre_cached: centre,
            extent_cached: extent,
        }
    }

    pub fn convex_hull_of(mut vertices: Vec<Vec2>) -> Result<Self> {
        check_false!(vertices.is_empty());
        let centre = polygon::centre_of(vertices.clone());
        for (a, b) in vertices.iter().sorted().tuple_windows() {
            if a == b {
                bail!("duplicate vertices: {a}");
            }
        }
        // Sort by clockwise winding order.
        vertices.sort_unstable_by(|&a, &b| {
            let det = (a - centre).cross(b - centre);
            if det > 0. {
                Ordering::Less
            } else if det < 0. {
                Ordering::Greater
            } else {
                a.len_squared().total_cmp(&b.len_squared())
            }
        });
        if vertices.len() <= 1 {
            return Ok(Self::from_vertices_unchecked(vertices));
        }

        let mut lower = polygon::hull(vertices.iter().copied());
        let mut upper = polygon::hull(vertices.into_iter().rev());
        check_eq!(lower.last(), upper.first());
        check_eq!(lower.first(), upper.last());
        check_is_some!(lower.pop());
        check_is_some!(upper.pop());

        let vertices = lower.into_iter().chain(upper).collect_vec();
        let to_remove = vertices
            .iter()
            .circular_tuple_windows()
            .filter_map(|(&u, &v, &w)| {
                if (v - u).cross(w - v).is_zero() {
                    gg_iter::index_of(&vertices, &v)
                } else {
                    None
                }
            })
            .collect::<BTreeSet<_>>();
        let vertices = vertices
            .into_iter()
            .enumerate()
            .filter(|(i, _)| !to_remove.contains(i))
            .map(|(_, v)| v)
            .collect_vec();
        Ok(Self::from_vertices_unchecked(vertices))
    }
}

impl Polygonal for ConvexCollider {
    fn vertices(&self) -> Vec<Vec2> {
        self.vertices.clone()
    }

    fn normals(&self) -> Vec<Vec2> {
        self.normals_cached.clone()
    }

    fn polygon_centre(&self) -> Vec2 {
        self.centre_cached
    }
}

impl AxisAlignedExtent for ConvexCollider {
    fn aa_extent(&self) -> Vec2 {
        self.extent_cached
    }

    fn centre(&self) -> Vec2 {
        self.centre_cached
    }
}

impl Collider for ConvexCollider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_type(&self) -> ColliderType {
        ColliderType::Convex
    }

    fn collides_with_box(&self, other: &BoxCollider) -> Option<Vec2> {
        self.polygon_collision(other)
    }

    fn collides_with_oriented_box(&self, other: &OrientedBoxCollider) -> Option<Vec2> {
        self.polygon_collision(other)
    }

    fn collides_with_convex(&self, other: &ConvexCollider) -> Option<Vec2> {
        self.polygon_collision(other)
    }

    fn into_generic(self) -> GenericCollider
    where
        Self: Sized + 'static,
    {
        GenericCollider::Convex(self)
    }

    fn translated(&self, by: Vec2) -> Self {
        let mut rv = self.clone();
        for vertex in &mut rv.vertices {
            *vertex += by;
        }
        rv.centre_cached = polygon::centre_of(rv.vertices.clone());
        rv
    }
    fn scaled(&self, by: Vec2) -> Self {
        let mut rv = self.clone();
        for vertex in &mut rv.vertices {
            *vertex = vertex.component_wise(by).abs();
        }
        rv.extent_cached = polygon::extent_of(rv.vertices.clone());
        rv
    }
    fn rotated(&self, by: f32) -> Self {
        let mut rv = self.clone();
        for vertex in &mut rv.vertices {
            *vertex = vertex.rotated(by);
        }
        rv.normals_cached = polygon::normals_of(rv.vertices.clone());
        rv
    }

    fn as_polygon(&self) -> Vec<Vec2> {
        self.vertices.clone()
    }

    fn as_triangles(&self) -> Vec<[Vec2; 3]> {
        let origin = self.vertices[0];
        self.vertices[1..]
            .iter()
            .copied()
            .tuple_windows()
            .map(|(u, v)| [origin, u, v])
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct CompoundCollider {
    inner: Vec<ConvexCollider>,
    override_normals: Vec<Vec2>,
}

impl CompoundCollider {
    pub fn new(inner: Vec<ConvexCollider>) -> Self {
        Self {
            inner,
            override_normals: Vec::new(),
        }
    }

    fn get_new_vertex(
        edges: &[(Vec2, Vec2)],
        prev: Vec2,
        origin: Vec2,
        next: Vec2,
    ) -> Option<Vec2> {
        let filtered_edges = edges
            .iter()
            .filter(|(a, b)| *a != origin && *b != origin)
            .collect_vec();

        let intersections_1 = filtered_edges
            .iter()
            .filter_map(|(a, b)| {
                Vec2::intersect(origin, (origin - prev) * ONE_OVER_EPSILON, *a, *b - *a)
            })
            .min_by(Vec2::cmp_by_length);

        let intersections_2 = filtered_edges
            .iter()
            .filter_map(|(a, b)| {
                Vec2::intersect(origin, (origin - next) * ONE_OVER_EPSILON, *a, *b - *a)
            })
            .min_by(Vec2::cmp_by_length);

        if let (Some(start), Some(end)) = (intersections_1, intersections_2) {
            let centre: Vec2 = (start + end) / 2;
            filtered_edges
                .iter()
                .filter_map(|(a, b)| {
                    Vec2::intersect(origin, (centre - origin) * ONE_OVER_EPSILON, *a, *b - *a)
                })
                .min_by(|a, b| a.cmp_by_dist(b, origin))
        } else {
            None
        }
    }
    #[must_use]
    pub fn decompose(mut vertices: Vec<Vec2>) -> Self {
        check_ge!(vertices.len(), 3);
        for _i in 0..vertices.len() {
            if let Some(rv) = Self::decompose_inner(&vertices) {
                return rv;
            }
            // Try a different starting point, maybe the algorithm finds a solution now.
            vertices.rotate_right(1);
        }
        panic!("failed to find a convex decomposition");
    }

    fn decompose_inner(vertices: &[Vec2]) -> Option<CompoundCollider> {
        if polygon::is_convex(vertices) {
            return Some(Self::new(vec![
                ConvexCollider::convex_hull_of(vertices.to_vec()).ok()?,
            ]));
        }

        let cycled_vertices = vertices
            .iter()
            .cycle()
            .take(vertices.len() + 2)
            .copied()
            .collect_vec();
        let edges = cycled_vertices
            .iter()
            .copied()
            .tuple_windows()
            .collect_vec();
        let angles = cycled_vertices
            .iter()
            .copied()
            .tuple_windows()
            .collect_vec();
        let (prev, origin, next) = angles
            .into_iter()
            .find(|(prev, origin, next)| (*origin - *prev).cross(*origin - *next) > 0.)?;
        let new_vertex = Self::get_new_vertex(&edges, prev, origin, next)?;

        let mut left_vertices = vec![origin, new_vertex];
        let mut changed = true;
        while changed {
            changed = false;
            for (start, end) in &edges {
                if left_vertices.contains(start) && !left_vertices.contains(end) {
                    if (*end - origin).cross(new_vertex - origin) < 0. {
                        left_vertices
                            .insert(gg_iter::index_of(&left_vertices, start).unwrap() + 1, *end);
                        changed = true;
                    }
                } else if left_vertices.contains(end)
                    && !left_vertices.contains(start)
                    && (*start - origin).cross(new_vertex - origin) < 0.
                {
                    left_vertices.insert(gg_iter::index_of(&left_vertices, end).unwrap(), *start);
                    changed = true;
                }
            }
        }

        let mut right_vertices = vec![origin, new_vertex];
        let mut changed = true;
        while changed {
            changed = false;
            for (start, end) in &edges {
                if right_vertices.contains(start)
                    && !right_vertices.contains(end)
                    && !left_vertices.contains(end)
                {
                    right_vertices
                        .insert(gg_iter::index_of(&right_vertices, start).unwrap() + 1, *end);
                    changed = true;
                } else if right_vertices.contains(end)
                    && !right_vertices.contains(start)
                    && !left_vertices.contains(start)
                {
                    right_vertices.insert(gg_iter::index_of(&right_vertices, end).unwrap(), *start);
                    changed = true;
                }
            }
        }

        let mut rv = Self::decompose_inner(&left_vertices)?;
        rv.extend(Self::decompose_inner(&right_vertices)?);
        Some(rv)
    }

    pub fn pixel_perfect_convex(data: &[Vec<Colour>]) -> Result<ConvexCollider> {
        let (_, _, vertices) = Self::pixel_perfect_vertices(data)?;
        ConvexCollider::convex_hull_of(vertices)
    }

    pub fn pixel_perfect(data: &[Vec<Colour>]) -> Result<CompoundCollider> {
        let (w, h, vertices) = Self::pixel_perfect_vertices(data)?;
        let mut collider = Self::decompose(vertices);
        collider.override_normals = vec![Vec2::up(), Vec2::right(), Vec2::down(), Vec2::left()];
        Ok(collider.translated(-Vec2::from([w as f32 / 2. + 0.75, h as f32 / 2. + 0.75])))
    }

    fn pixel_perfect_vertices(data: &[Vec<Colour>]) -> Result<(i32, i32, Vec<Vec2>)> {
        check_false!(data.is_empty());
        check_false!(data[0].is_empty());
        let w = i32::try_from(data[0].len())?;
        let h = i32::try_from(data.len())?;

        let (mut vertex_set, edge_set) = Self::extract_pixel_outline(data, w, h);
        let centre = vertex_set.iter().map(Vec2i::as_vec2).sum::<Vec2>() / vertex_set.len() as u32;
        let mut vertices = Vec::new();
        let mut current = vertex_set.pop_first().unwrap();
        loop {
            vertices.push(current.into());
            let candidates = vec![
                current + Vec2i::right(),
                current + Vec2i::up(),
                current + Vec2i::left(),
                current + Vec2i::down(),
            ]
            .into_iter()
            .filter(|next| vertex_set.contains(next))
            .filter(|next| edge_set.contains(&UnorderedPair::new(current, *next)))
            .collect_vec();
            let next = candidates
                .iter()
                .max_by(|a, b| {
                    (current.as_vec2() - centre)
                        .cross(a.as_vec2() - centre)
                        .partial_cmp(&(current.as_vec2() - centre).cross(b.as_vec2() - centre))
                        .unwrap()
                })
                .copied();
            if let Some(next) = next {
                vertex_set.remove(&next);
                current = next;
            } else if vertex_set.is_empty() {
                break;
            } else {
                panic!(
                    "discontinuity:{}",
                    vertex_set
                        .into_iter()
                        .map(|v| format!(
                            "\n\t{v}\t{}",
                            (current.as_vec2() - centre).cross(v.as_vec2() - centre)
                        ))
                        .fold(String::new(), |acc, e| acc + e.as_str())
                );
            }
        }
        Ok((w, h, vertices))
    }

    // TODO
    #[allow(clippy::too_many_lines)]
    fn extract_pixel_outline(
        data: &[Vec<Colour>],
        w: i32,
        h: i32,
    ) -> (BTreeSet<Vec2i>, BTreeSet<UnorderedPair<Vec2i>>) {
        let mut vertex_set = BTreeSet::new();
        let mut edge_set = BTreeSet::new();
        // Outermost pixels.
        let mut to_explore = (0..w)
            .map(|x| Vec2i::from([x, 0]))
            .chain((0..h).map(|y| Vec2i::from([0, y])))
            .chain((0..w).map(|x| Vec2i::from([x, h - 1])))
            .chain((0..h).map(|y| Vec2i::from([w - 1, y])))
            .collect_vec();
        // Outside frame of vertices in clockwise order.
        let mut outside_edge_set = (0..=w)
            .map(|x| Vec2i::from([x, 0]))
            .chain((0..=h).map(|y| Vec2i::from([w, y])))
            .chain((0..=w).rev().map(|x| Vec2i::from([x, h])))
            .chain((0..=h).rev().map(|y| Vec2i::from([0, y])))
            .collect_vec()
            .into_iter()
            .circular_tuple_windows()
            .map(|(a, b)| UnorderedPair::new(a, b))
            .collect::<BTreeSet<UnorderedPair<Vec2i>>>();
        // Search for pixels with nonzero alpha.
        let mut visited = BTreeSet::new();
        while let Some(next) = to_explore.pop() {
            if !visited.insert(next) || next.y >= h || next.x >= w {
                continue;
            }
            let Ok(x) = usize::try_from(next.x) else {
                continue;
            };
            let Ok(y) = usize::try_from(next.y) else {
                continue;
            };
            // Guaranteed to be safe by above check.
            if data[y][x].a.is_zero() {
                outside_edge_set.insert(UnorderedPair::new(next, next + Vec2i::right()));
                outside_edge_set.insert(UnorderedPair::new(
                    next + Vec2i::right(),
                    next + Vec2i::one(),
                ));
                outside_edge_set.insert(UnorderedPair::new(
                    next + Vec2i::one(),
                    next + Vec2i::down(),
                ));
                outside_edge_set.insert(UnorderedPair::new(next + Vec2i::down(), next));
            } else {
                vertex_set.insert(next);
                edge_set.insert(UnorderedPair::new(next, next + Vec2i::right()));
                edge_set.insert(UnorderedPair::new(
                    next + Vec2i::right(),
                    next + Vec2i::one(),
                ));
                edge_set.insert(UnorderedPair::new(
                    next + Vec2i::one(),
                    next + Vec2i::down(),
                ));
                edge_set.insert(UnorderedPair::new(next + Vec2i::down(), next));
            }
            to_explore.extend([
                next + Vec2i::left(),
                next + Vec2i::right(),
                next + Vec2i::up(),
                next + Vec2i::down(),
            ]);
        }
        // Turn pixels into squares of vertices.
        let to_add = vertex_set
            .iter()
            .copied()
            .flat_map(|vertex| {
                vec![
                    vertex + Vec2i::right(),
                    vertex + Vec2i::one(),
                    vertex + Vec2i::down(),
                ]
            })
            .collect::<BTreeSet<_>>();
        vertex_set.extend(to_add);
        // Remove vertices that are totally internal.
        let to_remove = vertex_set
            .iter()
            .copied()
            .filter(|&vertex| {
                edge_set.contains(&UnorderedPair::new(
                    vertex + Vec2i::one(),
                    vertex + Vec2i::down(),
                )) && edge_set.contains(&UnorderedPair::new(
                    vertex + Vec2i::down(),
                    vertex + Vec2i::down() + Vec2i::left(),
                )) && edge_set.contains(&UnorderedPair::new(
                    vertex + Vec2i::down() + Vec2i::left(),
                    vertex + Vec2i::left(),
                )) && edge_set.contains(&UnorderedPair::new(
                    vertex + Vec2i::left(),
                    vertex + Vec2i::left() + Vec2i::up(),
                )) && edge_set.contains(&UnorderedPair::new(
                    vertex + Vec2i::left() + Vec2i::up(),
                    vertex + Vec2i::up(),
                )) && edge_set.contains(&UnorderedPair::new(
                    vertex + Vec2i::up(),
                    vertex + Vec2i::up() + Vec2i::right(),
                )) && edge_set.contains(&UnorderedPair::new(
                    vertex + Vec2i::up() + Vec2i::right(),
                    vertex + Vec2i::right(),
                )) && edge_set.contains(&UnorderedPair::new(
                    vertex + Vec2i::right(),
                    vertex + Vec2i::one(),
                ))
            })
            .collect_vec();
        for vertex in to_remove {
            vertex_set.remove(&vertex);
        }
        // Keep only external edges.
        let edge_set = edge_set
            .intersection(&outside_edge_set)
            .copied()
            .collect::<BTreeSet<_>>();
        (vertex_set, edge_set)
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    #[must_use]
    pub fn combined(mut self, other: CompoundCollider) -> Self {
        self.extend(other);
        self
    }
    pub fn extend(&mut self, mut other: CompoundCollider) {
        self.inner.append(&mut other.inner);
    }

    fn get_unique_normals(&self) -> Vec<Vec2> {
        if !self.override_normals.is_empty() {
            return self.override_normals.clone();
        }

        let mut normals = BTreeMap::<Vec2, BTreeMap<UnorderedPair<Vec2>, i32>>::new();
        for (normal, edge) in self.inner.iter().flat_map(|c| {
            let edges = c
                .vertices()
                .into_iter()
                .circular_tuple_windows()
                .map(|(u, v)| UnorderedPair::new(u, v));
            c.normals().into_iter().zip(edges)
        }) {
            if let Some(edge_count) = normals
                .get_mut(&-normal)
                .and_then(|edges| edges.get_mut(&edge))
            {
                // duplicate normal
                *edge_count -= 1;
                if let Some(other_edge) = normals.entry(normal).or_default().get_mut(&edge) {
                    *other_edge -= 1;
                }
            } else {
                *normals.entry(normal).or_default().entry(edge).or_default() += 1;
            }
        }
        normals
            .iter()
            .filter(|(_, edges)| edges.values().all(|i| *i > 0))
            .map(|(normal, _)| *normal)
            .collect()
    }

    fn inner_colliders(&self) -> Vec<ConvexCollider> {
        let normals = self.get_unique_normals();
        self.inner
            .iter()
            .map(|c| {
                let mut c = c.clone();
                c.normals_cached.clone_from(&normals);
                c
            })
            .collect()
    }

    fn is_internal_mtv<C: Collider>(&self, other: &C, mtv: Vec2) -> bool {
        // Note: translated() returns GenericCollider, so we have to do collides_with()
        // with the opposite arguments, hence we use -mtv not mtv.
        let translated = other.translated(-mtv);
        self.inner_colliders()
            .into_iter()
            .any(|c| translated.collides_with_convex(&c).is_some())
    }
}

impl Polygonal for CompoundCollider {
    fn vertices(&self) -> Vec<Vec2> {
        self.inner
            .iter()
            .flat_map(ConvexCollider::vertices)
            .collect()
    }

    fn normals(&self) -> Vec<Vec2> {
        self.get_unique_normals()
    }

    fn polygon_centre(&self) -> Vec2 {
        polygon::centre_of(self.vertices())
    }
}

impl AxisAlignedExtent for CompoundCollider {
    fn aa_extent(&self) -> Vec2 {
        let mut max = Vec2::zero();
        for extent in self.inner.iter().map(ConvexCollider::aa_extent) {
            max.x = max.x.max(extent.x);
            max.y = max.y.max(extent.y);
        }
        max
    }

    fn centre(&self) -> Vec2 {
        self.polygon_centre()
    }
}

impl CompoundCollider {
    fn filter_candidate_collisions<C: Collider>(
        &self,
        other: &C,
        candidates: Vec<Vec2>,
    ) -> Option<Vec2> {
        candidates
            .iter()
            .filter(|&&mtv| !self.is_internal_mtv(other, mtv))
            .copied()
            .min_by(Vec2::cmp_by_length)
            .or(candidates.into_iter().min_by(Vec2::cmp_by_length))
    }
}

impl Collider for CompoundCollider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_type(&self) -> ColliderType {
        ColliderType::Compound
    }

    fn collides_with_box(&self, other: &BoxCollider) -> Option<Vec2> {
        self.filter_candidate_collisions(
            other,
            self.inner_colliders()
                .into_iter()
                .filter_map(|c| c.collides_with_box(other))
                .collect(),
        )
    }

    fn collides_with_oriented_box(&self, other: &OrientedBoxCollider) -> Option<Vec2> {
        self.filter_candidate_collisions(
            other,
            self.inner_colliders()
                .into_iter()
                .filter_map(|c| c.collides_with_oriented_box(other))
                .collect(),
        )
    }

    fn collides_with_convex(&self, other: &ConvexCollider) -> Option<Vec2> {
        self.filter_candidate_collisions(
            other,
            self.inner_colliders()
                .into_iter()
                .filter_map(|c| c.collides_with_convex(other))
                .collect(),
        )
    }

    fn into_generic(self) -> GenericCollider
    where
        Self: Sized + 'static,
    {
        GenericCollider::Compound(self)
    }

    fn translated(&self, by: Vec2) -> Self {
        let new_inner = self
            .inner
            .clone()
            .into_iter()
            .map(|c| c.translated(by))
            .collect_vec();
        Self {
            inner: new_inner,
            override_normals: self.override_normals.clone(),
        }
    }

    fn scaled(&self, by: Vec2) -> Self {
        let centre = self.centre();
        let new_inner = self
            .inner
            .clone()
            .into_iter()
            .map(|mut c| {
                for v in &mut c.vertices {
                    *v -= centre;
                    *v = v.component_wise(by);
                    *v += centre;
                }
                c
            })
            .collect_vec();
        Self {
            inner: new_inner,
            override_normals: self.override_normals.clone(),
        }
    }

    fn rotated(&self, _by: f32) -> Self {
        // TODO: implement
        warn!("CompoundCollider::rotated(): not implemented");
        self.clone()
    }

    fn as_polygon(&self) -> Vec<Vec2> {
        self.inner
            .iter()
            .flat_map(ConvexCollider::as_polygon)
            .collect()
    }

    fn as_triangles(&self) -> Vec<[Vec2; 3]> {
        self.inner
            .iter()
            .flat_map(ConvexCollider::as_triangles)
            .collect()
    }
}

#[derive(Clone, Debug)]
pub enum GenericCollider {
    Null,
    Box(BoxCollider),
    OrientedBox(OrientedBoxCollider),
    Convex(ConvexCollider),
    Compound(CompoundCollider),
}

impl Default for GenericCollider {
    fn default() -> Self {
        Self::Null
    }
}

impl AxisAlignedExtent for GenericCollider {
    fn aa_extent(&self) -> Vec2 {
        match self {
            GenericCollider::Null => NullCollider.aa_extent(),
            GenericCollider::Box(c) => c.aa_extent(),
            GenericCollider::OrientedBox(c) => c.aa_extent(),
            GenericCollider::Convex(c) => c.aa_extent(),
            GenericCollider::Compound(c) => c.aa_extent(),
        }
    }

    fn centre(&self) -> Vec2 {
        match self {
            GenericCollider::Null => NullCollider.centre(),
            GenericCollider::Box(c) => c.centre(),
            GenericCollider::OrientedBox(c) => c.centre(),
            GenericCollider::Convex(c) => c.centre(),
            GenericCollider::Compound(c) => c.centre(),
        }
    }
}

impl Collider for GenericCollider {
    fn as_any(&self) -> &dyn Any {
        match self {
            GenericCollider::Null => NullCollider.as_any(),
            GenericCollider::Box(c) => c.as_any(),
            GenericCollider::OrientedBox(c) => c.as_any(),
            GenericCollider::Convex(c) => c.as_any(),
            GenericCollider::Compound(c) => c.as_any(),
        }
    }

    fn get_type(&self) -> ColliderType {
        match self {
            GenericCollider::Null => NullCollider.get_type(),
            GenericCollider::Box(c) => c.get_type(),
            GenericCollider::OrientedBox(c) => c.get_type(),
            GenericCollider::Convex(c) => c.get_type(),
            GenericCollider::Compound(c) => c.get_type(),
        }
    }

    fn collides_with_box(&self, other: &BoxCollider) -> Option<Vec2> {
        match self {
            GenericCollider::Null => NullCollider.collides_with_box(other),
            GenericCollider::Box(c) => c.collides_with_box(other),
            GenericCollider::OrientedBox(c) => c.collides_with_box(other),
            GenericCollider::Convex(c) => c.collides_with_box(other),
            GenericCollider::Compound(c) => c.collides_with_box(other),
        }
    }

    fn collides_with_oriented_box(&self, other: &OrientedBoxCollider) -> Option<Vec2> {
        match self {
            GenericCollider::Null => NullCollider.collides_with_oriented_box(other),
            GenericCollider::Box(c) => c.collides_with_oriented_box(other),
            GenericCollider::OrientedBox(c) => c.collides_with_oriented_box(other),
            GenericCollider::Convex(c) => c.collides_with_oriented_box(other),
            GenericCollider::Compound(c) => c.collides_with_oriented_box(other),
        }
    }

    fn collides_with_convex(&self, other: &ConvexCollider) -> Option<Vec2> {
        match self {
            GenericCollider::Null => NullCollider.collides_with_convex(other),
            GenericCollider::Box(c) => c.collides_with_convex(other),
            GenericCollider::OrientedBox(c) => c.collides_with_convex(other),
            GenericCollider::Convex(c) => c.collides_with_convex(other),
            GenericCollider::Compound(c) => c.collides_with_convex(other),
        }
    }

    fn into_generic(self) -> GenericCollider
    where
        Self: Sized + 'static,
    {
        self
    }

    fn translated(&self, by: Vec2) -> GenericCollider {
        match self {
            GenericCollider::Null => NullCollider.translated(by).into_generic(),
            GenericCollider::Box(c) => c.translated(by).into_generic(),
            GenericCollider::OrientedBox(c) => c.translated(by).into_generic(),
            GenericCollider::Convex(c) => c.translated(by).into_generic(),
            GenericCollider::Compound(c) => c.translated(by).into_generic(),
        }
    }

    fn scaled(&self, by: Vec2) -> GenericCollider {
        match self {
            GenericCollider::Null => NullCollider.scaled(by).into_generic(),
            GenericCollider::Box(c) => c.scaled(by).into_generic(),
            GenericCollider::OrientedBox(c) => c.scaled(by).into_generic(),
            GenericCollider::Convex(c) => c.scaled(by).into_generic(),
            GenericCollider::Compound(c) => c.scaled(by).into_generic(),
        }
    }

    fn rotated(&self, by: f32) -> GenericCollider {
        match self {
            GenericCollider::Null => NullCollider.rotated(by).into_generic(),
            GenericCollider::Box(c) => c.rotated(by).into_generic(),
            GenericCollider::OrientedBox(c) => c.rotated(by).into_generic(),
            GenericCollider::Convex(c) => c.rotated(by).into_generic(),
            GenericCollider::Compound(c) => c.rotated(by).into_generic(),
        }
    }

    fn as_polygon(&self) -> Vec<Vec2> {
        match self {
            GenericCollider::Null => NullCollider.as_polygon(),
            GenericCollider::Box(c) => c.as_polygon(),
            GenericCollider::OrientedBox(c) => c.as_polygon(),
            GenericCollider::Convex(c) => c.as_polygon(),
            GenericCollider::Compound(c) => c.as_polygon(),
        }
    }

    fn as_triangles(&self) -> Vec<[Vec2; 3]> {
        match self {
            GenericCollider::Null => NullCollider.as_triangles(),
            GenericCollider::Box(c) => c.as_triangles(),
            GenericCollider::OrientedBox(c) => c.as_triangles(),
            GenericCollider::Convex(c) => c.as_triangles(),
            GenericCollider::Compound(c) => c.as_triangles(),
        }
    }
}

impl Display for GenericCollider {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match &self {
            GenericCollider::Null => {
                write!(f, "<null>")
            }
            GenericCollider::Box(_) => {
                write!(f, "Box")
            }
            GenericCollider::OrientedBox(inner) => {
                write!(f, "OrientedBox: {} deg.", inner.rotation.to_degrees())
            }
            GenericCollider::Convex(inner) => {
                write!(f, "Convex: {} edges", inner.normals_cached.len())
            }
            GenericCollider::Compound(inner) => {
                write!(
                    f,
                    "Compound: {} pieces, {:?} edges",
                    inner.inner.len(),
                    inner
                        .inner
                        .iter()
                        .map(|c| c.normals_cached.len())
                        .collect_vec()
                )
            }
        }
    }
}

#[register_scene_object]
pub struct GgInternalCollisionShape {
    last_transform: Transform,
    collider: GenericCollider,
    emitting_tags: Vec<&'static str>,
    listening_tags: Vec<&'static str>,

    // For GUI:
    // <RenderItem, should_be_updated>
    wireframe: RenderItem,
    show_wireframe: bool,
    last_show_wireframe: bool,
    extent_cell_receiver_x: EditCellReceiver<f32>,
    extent_cell_receiver_y: EditCellReceiver<f32>,
    centre_cell_receiver_x: EditCellReceiver<f32>,
    centre_cell_receiver_y: EditCellReceiver<f32>,
}

impl GgInternalCollisionShape {
    pub fn from_collider<C: Collider, O: ObjectTypeEnum>(
        collider: C,
        emitting_tags: &[&'static str],
        listening_tags: &[&'static str],
    ) -> AnySceneObject<O> {
        let mut rv = Self {
            last_transform: Transform::default(),
            collider: collider.into_generic(),
            emitting_tags: emitting_tags.to_vec(),
            listening_tags: listening_tags.to_vec(),
            wireframe: RenderItem::default(),
            show_wireframe: false,
            last_show_wireframe: false,
            extent_cell_receiver_x: EditCellReceiver::new(),
            extent_cell_receiver_y: EditCellReceiver::new(),
            centre_cell_receiver_x: EditCellReceiver::new(),
            centre_cell_receiver_y: EditCellReceiver::new(),
        };
        rv.regenerate_wireframe(&Transform::default());
        AnySceneObject::new(rv)
    }

    pub fn from_object<ObjectType: ObjectTypeEnum, O: SceneObject<ObjectType>, C: Collider>(
        object: &O,
        collider: C,
    ) -> AnySceneObject<ObjectType> {
        Self::from_collider(collider, &object.emitting_tags(), &object.listening_tags())
    }
    pub fn from_object_sprite<ObjectType: ObjectTypeEnum, O: SceneObject<ObjectType>>(
        object: &O,
        sprite: &Sprite,
    ) -> AnySceneObject<ObjectType> {
        Self::from_collider(
            sprite.as_box_collider(),
            &object.emitting_tags(),
            &object.listening_tags(),
        )
    }

    pub fn collider(&self) -> &GenericCollider {
        &self.collider
    }

    fn regenerate_wireframe(&mut self, absolute_transform: &Transform) {
        self.wireframe = RenderItem::from_raw_vertices(
            self.collider
                .as_triangles()
                .into_flattened()
                .into_iter()
                .map(|v| v - absolute_transform.centre)
                .collect(),
        )
        .with_depth(VertexDepth::max_value());
    }

    pub fn show_wireframe(&mut self) {
        self.show_wireframe = true;
    }
    pub fn hide_wireframe(&mut self) {
        self.show_wireframe = false;
    }
}

#[partially_derive_scene_object]
impl<ObjectType: ObjectTypeEnum> SceneObject<ObjectType> for GgInternalCollisionShape {
    fn name(&self) -> String {
        format!("CollisionShape [{:?}]", self.collider.get_type()).to_string()
    }

    fn on_load(
        &mut self,
        _object_ctx: &mut ObjectContext<ObjectType>,
        _resource_handler: &mut ResourceHandler,
    ) -> Result<Option<RenderItem>> {
        Ok(None)
    }
    fn on_ready(&mut self, ctx: &mut UpdateContext<ObjectType>) {
        check_is_some!(ctx.object().parent(), "CollisionShapes must have a parent");
    }
    fn on_update_begin(&mut self, ctx: &mut UpdateContext<ObjectType>) {
        self.update_transform(ctx.absolute_transform());
    }
    fn on_fixed_update(&mut self, ctx: &mut FixedUpdateContext<ObjectType>) {
        self.update_transform(ctx.absolute_transform());
    }

    fn on_update(&mut self, ctx: &mut UpdateContext<ObjectType>) {
        self.update_transform(ctx.absolute_transform());
        if self.show_wireframe {
            let mut canvas = ctx
                .object_mut()
                .first_other_as_mut::<Canvas>()
                .expect("No Canvas object in scene!");
            match &self.collider {
                GenericCollider::Compound(compound) => {
                    let mut colours = [
                        Colour::green(),
                        Colour::red(),
                        Colour::blue(),
                        Colour::magenta(),
                        Colour::yellow(),
                    ];
                    colours.reverse();
                    for inner in &compound.inner {
                        let col = *colours.last().unwrap();
                        colours.rotate_right(1);
                        inner.draw_polygonal(&mut canvas, col);
                    }
                }
                GenericCollider::OrientedBox(c) => c.draw_polygonal(&mut canvas, Colour::green()),
                GenericCollider::Box(c) => c.draw_polygonal(&mut canvas, Colour::green()),
                GenericCollider::Convex(c) => c.draw_polygonal(&mut canvas, Colour::green()),
                GenericCollider::Null => {}
            }
        }
    }

    fn on_update_end(&mut self, ctx: &mut UpdateContext<ObjectType>) {
        self.update_transform(ctx.absolute_transform());
    }

    fn get_type(&self) -> ObjectType {
        ObjectType::gg_collider()
    }

    fn emitting_tags(&self) -> Vec<&'static str> {
        self.emitting_tags.clone()
    }
    fn listening_tags(&self) -> Vec<&'static str> {
        self.listening_tags.clone()
    }

    fn as_renderable_object(&mut self) -> Option<&mut dyn RenderableObject<ObjectType>> {
        Some(self)
    }
    fn as_gui_object(&mut self) -> Option<&mut dyn GuiObject<ObjectType>> {
        if self.show_wireframe {
            Some(self)
        } else {
            None
        }
    }
}

impl GgInternalCollisionShape {
    pub(crate) fn update_transform(&mut self, next_transform: Transform) {
        if self.last_transform != next_transform {
            self.collider = self.collider.transformed(&self.last_transform.inverse());
            self.collider = self.collider.transformed(&next_transform);
            self.last_transform = next_transform;
        }
    }
}

impl<ObjectType: ObjectTypeEnum> RenderableObject<ObjectType> for GgInternalCollisionShape {
    #[allow(clippy::if_not_else)] // clearer as written
    fn on_render(&mut self, render_ctx: &mut RenderContext) {
        if self.show_wireframe {
            if !self.last_show_wireframe {
                render_ctx.insert_render_item(&self.wireframe);
            } else {
                render_ctx.update_render_item(&self.wireframe);
            }
        }
        if !self.show_wireframe && self.last_show_wireframe {
            render_ctx.remove_render_item();
        }
        self.last_show_wireframe = self.show_wireframe;
    }
    fn shader_execs(&self) -> Vec<ShaderExec> {
        check!(self.show_wireframe);
        vec![
            ShaderExec {
                blend_col: Colour::cyan().with_alpha(0.2),
                shader_id: get_shader(SpriteShader::name()),
                ..Default::default()
            },
            ShaderExec {
                blend_col: Colour::green(),
                shader_id: get_shader(WireframeShader::name()),
                ..Default::default()
            },
        ]
    }
}

impl<ObjectType: ObjectTypeEnum> GuiObject<ObjectType> for GgInternalCollisionShape {
    fn on_gui(
        &mut self,
        ctx: &UpdateContext<ObjectType>,
        _selected: bool,
    ) -> Box<GuiInsideClosure> {
        let extent = self.collider.aa_extent();
        let (next_x, next_y) = (
            self.extent_cell_receiver_x.try_recv(),
            self.extent_cell_receiver_y.try_recv(),
        );
        if next_x.is_some() || next_y.is_some() {
            self.collider = self.collider.with_extent(Vec2 {
                x: next_x.unwrap_or(extent.x).max(0.1),
                y: next_y.unwrap_or(extent.y).max(0.1),
            });
            self.regenerate_wireframe(&ctx.absolute_transform());
        }
        self.extent_cell_receiver_x.update_live(extent.x);
        self.extent_cell_receiver_y.update_live(extent.y);

        let extent_cell_sender_x = self.extent_cell_receiver_x.sender();
        let extent_cell_sender_y = self.extent_cell_receiver_y.sender();

        let centre = self.collider.centre();
        let (next_x, next_y) = (
            self.centre_cell_receiver_x.try_recv(),
            self.centre_cell_receiver_y.try_recv(),
        );
        if next_x.is_some() || next_y.is_some() {
            self.collider = self.collider.with_centre(Vec2 {
                x: next_x.unwrap_or(centre.x),
                y: next_y.unwrap_or(centre.y),
            });
            self.regenerate_wireframe(&ctx.absolute_transform());
        }
        self.centre_cell_receiver_x.update_live(centre.x);
        self.centre_cell_receiver_y.update_live(centre.y);

        let centre_cell_sender_x = self.centre_cell_receiver_x.sender();
        let centre_cell_sender_y = self.centre_cell_receiver_y.sender();

        let collider = self.collider.clone();
        Box::new(move |ui| {
            ui.label(collider.to_string());
            ui.add(egui::Label::new("Extent").selectable(false));
            collider
                .aa_extent()
                .build_gui(ui, 0.1, extent_cell_sender_x, extent_cell_sender_y);
            ui.end_row();
            ui.add(egui::Label::new("Centre").selectable(false));
            collider
                .centre()
                .build_gui(ui, 0.1, centre_cell_sender_x, centre_cell_sender_y);
        })
    }
}

use crate::core::render::VertexDepth;
use crate::core::update::RenderContext;
use crate::gui::EditCellReceiver;
use crate::shader::{Shader, SpriteShader, WireframeShader, get_shader};
use crate::util::canvas::Canvas;
pub use GgInternalCollisionShape as CollisionShape;
