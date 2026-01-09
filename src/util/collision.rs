use crate::core::render::VertexDepth;
use crate::core::scene::{GuiCommand, GuiObject};
use crate::core::update::RenderContext;
use crate::gui::EditCell;
use crate::resource::sprite::Sprite;
use crate::util::canvas::Canvas;
use crate::util::gg_sync::GgMutex;
use crate::util::{UnorderedPair, gg_iter};
use crate::{
    check, check_is_some,
    core::prelude::*,
    util::{
        gg_range,
        linalg::{AxisAlignedExtent, Transform, Vec2},
    },
};
use glongge_derive::partially_derive_scene_object;
use itertools::Itertools;
use num_traits::{Float, Zero};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Display, Formatter};
use std::{any::Any, fmt::Debug, ops::Range};

#[derive(Debug, Default, Clone, bincode::Encode, bincode::Decode)]
pub struct BoxCollider {
    pub(crate) centre: Vec2,
    pub(crate) extent: Vec2,
}

#[derive(Debug, Default, Clone, bincode::Encode, bincode::Decode)]
pub struct BoxCollider3d {
    pub(crate) centre: Vec2,
    pub(crate) extent: Vec2,
    pub(crate) front: f32,
    pub(crate) back: f32,
}

#[derive(Debug, Default, Clone, bincode::Encode, bincode::Decode)]
pub struct ConvexCollider {
    pub(crate) vertices: Vec<Vec2>,
    pub(crate) normals_cached: Vec<Vec2>,
    pub(crate) centre_cached: Vec2,
    pub(crate) extent_cached: Vec2,
}

#[derive(Debug, Default, Clone, bincode::Encode, bincode::Decode)]
pub struct OrientedBoxCollider {
    pub(crate) centre: Vec2,
    pub(crate) rotation: f32,
    pub(crate) unrotated_half_widths: Vec2,
    pub(crate) extent: Vec2,
}

#[derive(Debug, PartialEq, Eq)]
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
            ColliderType::Box => {
                let other = other
                    .as_any()
                    .downcast_ref()
                    .expect("unreachable: get_type() already matched");
                self.collides_with_box(other)
            }
            ColliderType::OrientedBox => {
                let other = other
                    .as_any()
                    .downcast_ref()
                    .expect("unreachable: get_type() already matched");
                self.collides_with_oriented_box(other)
            }
            ColliderType::Convex => {
                let other = other
                    .as_any()
                    .downcast_ref()
                    .expect("unreachable: get_type() already matched");
                self.collides_with_convex(other)
            }
            ColliderType::Compound => match self.get_type() {
                ColliderType::Null => None,
                ColliderType::Box => {
                    let this = self
                        .as_any()
                        .downcast_ref()
                        .expect("unreachable: get_type() already matched");
                    other.collides_with_box(this)
                }
                ColliderType::OrientedBox => {
                    let this = self
                        .as_any()
                        .downcast_ref()
                        .expect("unreachable: get_type() already matched");
                    other.collides_with_oriented_box(this)
                }
                ColliderType::Convex => {
                    let this = self
                        .as_any()
                        .downcast_ref()
                        .expect("unreachable: get_type() already matched");
                    other.collides_with_convex(this)
                }
                ColliderType::Compound => {
                    let this: &CompoundCollider = self
                        .as_any()
                        .downcast_ref()
                        .expect("unreachable: get_type() already matched");
                    let other: &CompoundCollider = other
                        .as_any()
                        .downcast_ref()
                        .expect("unreachable: get_type() already matched");
                    this.inner_colliders()
                        .into_iter()
                        .filter_map(|c| other.collides_with_convex(&c))
                        .filter(|&mtv| !other.is_internal_mtv(this, mtv))
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

    type Rotated: Collider;

    #[must_use]
    fn translated(&self, by: Vec2) -> Self;
    #[must_use]
    fn scaled(&self, by: Vec2) -> Self;
    #[must_use]
    fn rotated(&self, by: f32) -> Self::Rotated;
    #[must_use]
    fn transformed(&self, by: &Transform) -> Self::Rotated
    where
        Self: Sized,
    {
        self.scaled(by.scale)
            .rotated(by.rotation)
            .translated(by.centre)
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
        self.scaled(extent.component_wise_div(self.extent()))
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
    fn extent(&self) -> Vec2 {
        Vec2::zero()
    }

    fn centre(&self) -> Vec2 {
        Vec2::zero()
    }
}
impl Collider for NullCollider {
    type Rotated = Self;

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
    use std::ops::Range;
    use tracing::warn;

    pub fn hull<I: Iterator<Item = Vec2>>(vertices: I) -> Vec<Vec2> {
        let mut hull: Vec<Vec2> = Vec::new();
        for vertex in vertices {
            while hull.len() >= 2 {
                let last = hull[hull.len() - 1];
                let snd_last = hull[hull.len() - 2];
                if (last - snd_last).cross(vertex - snd_last) > 0.0 {
                    break;
                }
                hull.pop();
            }
            hull.push(vertex);
        }
        hull
    }
    pub fn adjust_for_containment(self_proj: &Range<f32>, other_proj: &Range<f32>) -> f32 {
        if gg_range::overlaps_f32(self_proj, other_proj) {
            let starts = (self_proj.start - other_proj.start).abs();
            let ends = (self_proj.end - other_proj.end).abs();
            f32::min(starts, ends)
        } else {
            0.0
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
        match vertices.len() {
            0 => Vec2::zero(),
            1 => vertices[0],
            2 => (vertices[0] + vertices[1]) / 2.0,
            _ => {
                // Translate vertices to be near origin to avoid precision loss
                // in the shoelace formula when coordinates are large
                let offset = vertices[0];
                for v in &mut vertices {
                    *v -= offset;
                }
                vertices.push(vertices[0]);
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
                    x: x / (6. * (area / 2.0)),
                    y: y / (6. * (area / 2.0)),
                } + offset
            }
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
                    dist += polygon::adjust_for_containment(&self_proj, &other_proj);
                    // EPSILON check should be unnecessary (see overlap_len_f32()).
                    // Add it just in case.
                    if dist < min_dist && dist.abs() > EPSILON {
                        min_dist = dist;
                        min_axis = axis;
                    }
                }
            }
        }

        let mtv = min_dist * min_axis;
        check!(all_normals.contains(&min_axis));
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
            canvas.line(start, end, 1.0, col);
            canvas.rect(
                self.polygon_centre() - Vec2::one(),
                self.polygon_centre() + Vec2::one(),
                col,
            );
        }
        for (a, b) in vertices.into_iter().circular_tuple_windows() {
            canvas.line(a, b, 1.0, col);
        }
    }
}

impl<T: Polygonal> Polygonal for &T {
    fn vertices(&self) -> Vec<Vec2> {
        (*self).vertices()
    }

    fn normals(&self) -> Vec<Vec2> {
        (*self).normals()
    }

    fn polygon_centre(&self) -> Vec2 {
        (*self).polygon_centre()
    }
}

impl OrientedBoxCollider {
    pub fn from_centre(centre: Vec2, half_widths: Vec2) -> Self {
        let mut rv = Self {
            centre,
            rotation: 0.0,
            unrotated_half_widths: half_widths,
            extent: Vec2::zero(),
        };
        rv.extent = polygon::extent_of(rv.vertices());
        rv
    }
    #[must_use]
    pub fn from_aa_extent(extent: &impl AxisAlignedExtent) -> Self {
        Self::from_centre(extent.centre(), extent.half_widths())
    }
    pub fn from_top_left(top_left: Vec2, extent: Vec2) -> Self {
        let mut rv = Self {
            centre: top_left + extent.abs() / 2,
            rotation: 0.0,
            unrotated_half_widths: extent.abs() / 2,
            extent: Vec2::zero(),
        };
        rv.extent = polygon::extent_of(rv.vertices());
        rv
    }
    pub fn from_transform(transform: Transform, half_widths: Vec2) -> Self {
        let mut rv = Self {
            centre: transform.centre,
            rotation: transform.rotation,
            unrotated_half_widths: transform.scale.component_wise(half_widths).abs(),
            extent: Vec2::zero(),
        };
        rv.extent = polygon::extent_of(rv.vertices());
        rv
    }
    pub fn square(transform: Transform, half_width: f32) -> Self {
        Self::from_transform(transform, half_width.abs() * Vec2::one())
    }

    pub fn top_left_rotated(&self) -> Vec2 {
        self.centre + (-self.unrotated_half_widths).rotated(self.rotation)
    }
    pub fn top_right_rotated(&self) -> Vec2 {
        self.centre
            + Vec2 {
                x: self.unrotated_half_widths.x,
                y: -self.unrotated_half_widths.y,
            }
            .rotated(self.rotation)
    }
    pub fn bottom_left_rotated(&self) -> Vec2 {
        self.centre
            + Vec2 {
                x: -self.unrotated_half_widths.x,
                y: self.unrotated_half_widths.y,
            }
            .rotated(self.rotation)
    }
    pub fn bottom_right_rotated(&self) -> Vec2 {
        self.centre + self.unrotated_half_widths.rotated(self.rotation)
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
    fn extent(&self) -> Vec2 {
        self.extent
    }

    fn centre(&self) -> Vec2 {
        self.centre
    }
}

impl Collider for OrientedBoxCollider {
    type Rotated = Self;

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
        rv.centre += by;
        rv
    }

    fn scaled(&self, by: Vec2) -> Self {
        let mut rv = self.clone();
        rv.unrotated_half_widths = self.unrotated_half_widths.component_wise(by).abs();
        rv.extent = polygon::extent_of(rv.vertices());
        rv
    }

    fn rotated(&self, by: f32) -> Self {
        let mut rv = self.clone();
        rv.rotation += by;
        rv.extent = polygon::extent_of(rv.vertices());
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
    pub fn from_aa_extent(extent: &impl AxisAlignedExtent) -> Self {
        Self::from_centre(extent.centre(), extent.half_widths())
    }
    #[must_use]
    pub fn from_transform(transform: Transform, half_widths: Vec2) -> Self {
        check_eq!(transform.rotation, 0.0);
        Self::from_centre(
            transform.centre,
            transform.scale.component_wise(half_widths).abs(),
        )
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
    fn extent(&self) -> Vec2 {
        self.extent
    }

    fn centre(&self) -> Vec2 {
        self.centre
    }
}

impl Collider for BoxCollider {
    type Rotated = OrientedBoxCollider;

    fn as_any(&self) -> &dyn Any {
        self
    }
    fn get_type(&self) -> ColliderType {
        ColliderType::Box
    }

    fn collides_with_box(&self, other: &BoxCollider) -> Option<Vec2> {
        let self_proj = self.left()..self.right();
        let other_proj = other.left()..other.right();
        let horizontal_dist = match gg_range::overlap_len_f32(&self_proj, &other_proj) {
            None | Some(0.0) => return None,
            Some(dist) => dist + polygon::adjust_for_containment(&self_proj, &other_proj),
        };
        let self_proj = self.top()..self.bottom();
        let other_proj = other.top()..other.bottom();
        let vertical_dist = match gg_range::overlap_len_f32(&self_proj, &other_proj) {
            None | Some(0.0) => return None,
            Some(dist) => dist + polygon::adjust_for_containment(&self_proj, &other_proj),
        };
        if vertical_dist < horizontal_dist {
            // Collision along vertical axis.
            let mtv = vertical_dist * Vec2::down();
            if self.centre.y < other.centre.y {
                Some(-mtv)
            } else {
                Some(mtv)
            }
        } else {
            // Collision along horizontal axis.
            let mtv = horizontal_dist * Vec2::right();
            if self.centre.x < other.centre.x {
                Some(-mtv)
            } else {
                Some(mtv)
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

    fn rotated(&self, by: f32) -> OrientedBoxCollider {
        OrientedBoxCollider::from_aa_extent(self).rotated(by)
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

impl AxisAlignedExtent for BoxCollider3d {
    fn extent(&self) -> Vec2 {
        self.extent
    }

    fn centre(&self) -> Vec2 {
        self.centre
    }
}

impl BoxCollider3d {
    pub fn from_2d(collider: &BoxCollider, back: f32, front: f32) -> Self {
        check_le!(back, front);
        Self {
            centre: collider.centre,
            extent: collider.extent,
            front,
            back,
        }
    }

    /// Returns the minimum translation vector (MTV) as a pair (Vec2{ x, y}, z).
    pub fn collides_with(&self, other: &BoxCollider3d) -> Option<(Vec2, f32)> {
        let self_proj = self.left()..self.right();
        let other_proj = other.left()..other.right();
        let horizontal_dist = match gg_range::overlap_len_f32(&self_proj, &other_proj) {
            None | Some(0.0) => return None,
            Some(dist) => dist + polygon::adjust_for_containment(&self_proj, &other_proj),
        };
        let self_proj = self.top()..self.bottom();
        let other_proj = other.top()..other.bottom();
        let vertical_dist = match gg_range::overlap_len_f32(&self_proj, &other_proj) {
            None | Some(0.0) => return None,
            Some(dist) => dist + polygon::adjust_for_containment(&self_proj, &other_proj),
        };
        let self_proj = self.back..self.front;
        let other_proj = other.back..other.front;
        let depth_dist = match gg_range::overlap_len_f32(&self_proj, &other_proj) {
            None | Some(0.0) => return None,
            Some(dist) => dist + polygon::adjust_for_containment(&self_proj, &other_proj),
        };

        // Find the minimum distance to determine collision axis
        if vertical_dist <= horizontal_dist && vertical_dist <= depth_dist {
            // Collision along vertical axis.
            let mtv = vertical_dist * Vec2::down();
            if self.centre.y < other.centre.y {
                Some((-mtv, 0.0))
            } else {
                Some((mtv, 0.0))
            }
        } else if horizontal_dist <= vertical_dist && horizontal_dist <= depth_dist {
            // Collision along horizontal axis.
            let mtv = horizontal_dist * Vec2::right();
            if self.centre.x < other.centre.x {
                Some((-mtv, 0.0))
            } else {
                Some((mtv, 0.0))
            }
        } else {
            // Collision along third axis.
            let dh = depth_dist;
            if (self.front - self.back) / 2.0 < (other.front - other.back) / 2.0 {
                Some((Vec2::zero(), -dh))
            } else {
                Some((Vec2::zero(), dh))
            }
        }
    }
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
            if det > 0.0 {
                Ordering::Less
            } else if det < 0.0 {
                Ordering::Greater
            } else {
                a.len_squared().total_cmp(&b.len_squared())
            }
        });
        if vertices.len() <= 2 {
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
    fn extent(&self) -> Vec2 {
        self.extent_cached
    }

    fn centre(&self) -> Vec2 {
        self.centre_cached
    }
}

impl Collider for ConvexCollider {
    type Rotated = Self;

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
        let centre = self.centre();
        let mut rv = self.clone();
        for vertex in &mut rv.vertices {
            *vertex -= centre;
            *vertex = vertex.component_wise(by);
            *vertex += centre;
        }
        rv.extent_cached = polygon::extent_of(rv.vertices.clone());
        rv
    }
    fn rotated(&self, by: f32) -> Self {
        let centre = self.centre();
        let mut rv = self.clone();
        for vertex in &mut rv.vertices {
            *vertex -= centre;
            *vertex = vertex.rotated(by);
            *vertex += centre;
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

pub struct CompoundCollider {
    pub(crate) inner: Vec<ConvexCollider>,
    override_normals: Vec<Vec2>,
    unique_normals_cached: GgMutex<Vec<Vec2>>,
}

impl CompoundCollider {
    pub fn new(inner: Vec<ConvexCollider>) -> Self {
        Self {
            inner,
            override_normals: Vec::new(),
            unique_normals_cached: GgMutex::new(Vec::new()),
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
            .filter_map(|(a, b)| Vec2::intersect(origin, (origin - prev) / EPSILON, *a, *b - *a))
            .min_by(Vec2::cmp_by_length);

        let intersections_2 = filtered_edges
            .iter()
            .filter_map(|(a, b)| Vec2::intersect(origin, (origin - next) / EPSILON, *a, *b - *a))
            .min_by(Vec2::cmp_by_length);

        if let (Some(start), Some(end)) = (intersections_1, intersections_2) {
            let centre: Vec2 = (start + end) / 2;
            filtered_edges
                .iter()
                .filter_map(|(a, b)| {
                    Vec2::intersect(origin, (centre - origin) / EPSILON, *a, *b - *a)
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
            .find(|(prev, origin, next)| (*origin - *prev).cross(*origin - *next) > 0.0)?;
        let new_vertex = Self::get_new_vertex(&edges, prev, origin, next)?;

        let mut left_vertices = vec![origin, new_vertex];
        let mut changed = true;
        while changed {
            changed = false;
            for (start, end) in &edges {
                if left_vertices.contains(start) && !left_vertices.contains(end) {
                    if (*end - origin).cross(new_vertex - origin) < 0.0 {
                        left_vertices
                            .insert(gg_iter::index_of(&left_vertices, start).unwrap() + 1, *end);
                        changed = true;
                    }
                } else if left_vertices.contains(end)
                    && !left_vertices.contains(start)
                    && (*start - origin).cross(new_vertex - origin) < 0.0
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
        let (_, _, vertices) = Self::pixel_perfect_vertices(data);
        ConvexCollider::convex_hull_of(vertices)
    }

    pub fn pixel_perfect(data: &[Vec<Colour>]) -> CompoundCollider {
        let (w, h, vertices) = Self::pixel_perfect_vertices(data);
        let mut collider = Self::decompose(vertices);
        collider.override_normals = vec![Vec2::up(), Vec2::right(), Vec2::down(), Vec2::left()];
        collider.translated(-Vec2::from([w as f32 / 2.0 + 0.75, h as f32 / 2.0 + 0.75]))
    }

    fn pixel_perfect_vertices(data: &[Vec<Colour>]) -> (i32, i32, Vec<Vec2>) {
        check_false!(data.is_empty());
        check_false!(data[0].is_empty());
        check_lt!(data[0].len(), i32::MAX as usize);
        check_lt!(data.len(), i32::MAX as usize);
        let w = i32::try_from(data[0].len()).expect("unreachable: checked above");
        let h = i32::try_from(data.len()).expect("unreachable: checked above");

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
        (w, h, vertices)
    }

    // TODO: clean up?
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
        self.unique_normals_cached
            .try_lock("CompoundCollider::extend()")
            .expect("CompoundCollider::extend()")
            .expect("should never be contested")
            .clear();
    }

    fn get_unique_normals(&self) -> Vec<Vec2> {
        if !self.override_normals.is_empty() {
            return self.override_normals.clone();
        }
        let mut unique_normals_cached = self
            .unique_normals_cached
            .try_lock("CompoundCollider::get_unique_normals()")
            .expect("CompoundCollider::get_unique_normals()")
            .expect("should never be contested");
        if !unique_normals_cached.is_empty() {
            return unique_normals_cached.clone();
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
                let other_normal = normals.entry(normal).or_default();
                check_false!(other_normal.contains_key(&edge));
            } else {
                *normals.entry(normal).or_default().entry(edge).or_default() += 1;
            }
        }
        *unique_normals_cached = normals
            .iter()
            .filter(|(_, edges)| edges.values().all(|i| *i > 0))
            .map(|(normal, _)| *normal)
            .collect();
        unique_normals_cached.clone()
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
    fn extent(&self) -> Vec2 {
        let mut min = Vec2::splat(f32::MAX);
        let mut max = Vec2::splat(f32::MIN);
        for collider in &self.inner {
            for vertex in collider.vertices() {
                min.x = min.x.min(vertex.x);
                min.y = min.y.min(vertex.y);
                max.x = max.x.max(vertex.x);
                max.y = max.y.max(vertex.y);
            }
        }
        #[allow(clippy::float_cmp)]
        if min.x == f32::MAX {
            Vec2::zero()
        } else {
            max - min
        }
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

impl Clone for CompoundCollider {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            override_normals: self.override_normals.clone(),
            unique_normals_cached: GgMutex::new(
                self.unique_normals_cached
                    .try_lock("CompoundCollider::clone()")
                    .expect("CompoundCollider::clone()")
                    .expect("should never be contested")
                    .clone(),
            ),
        }
    }
}
impl Debug for CompoundCollider {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "CompoundCollider({:?})", self.inner)
    }
}

impl Collider for CompoundCollider {
    type Rotated = Self;

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
            unique_normals_cached: GgMutex::new(Vec::new()),
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
            unique_normals_cached: GgMutex::new(Vec::new()),
        }
    }

    fn rotated(&self, by: f32) -> Self {
        let centre = self.centre();
        let new_inner = self
            .inner
            .clone()
            .into_iter()
            .map(|mut c| {
                for v in &mut c.vertices {
                    *v -= centre;
                    *v = v.rotated(by);
                    *v += centre;
                }
                c.normals_cached = polygon::normals_of(c.vertices.clone());
                c.centre_cached = polygon::centre_of(c.vertices.clone());
                c.extent_cached = polygon::extent_of(c.vertices.clone());
                c
            })
            .collect_vec();
        let new_override_normals = self
            .override_normals
            .iter()
            .map(|n| n.rotated(by))
            .collect_vec();
        Self {
            inner: new_inner,
            override_normals: new_override_normals,
            unique_normals_cached: GgMutex::new(Vec::new()),
        }
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

#[derive(Clone, Debug, Default)]
pub enum GenericCollider {
    #[default]
    Null,
    Box(BoxCollider),
    OrientedBox(OrientedBoxCollider),
    Convex(ConvexCollider),
    Compound(CompoundCollider),
}

impl AxisAlignedExtent for GenericCollider {
    fn extent(&self) -> Vec2 {
        match self {
            GenericCollider::Null => NullCollider.extent(),
            GenericCollider::Box(c) => c.extent(),
            GenericCollider::OrientedBox(c) => c.extent(),
            GenericCollider::Convex(c) => c.extent(),
            GenericCollider::Compound(c) => c.extent(),
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
    type Rotated = Self;

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

    fn transformed(&self, by: &Transform) -> Self::Rotated
    where
        Self: Sized,
    {
        let original_centre = self.centre();
        if self.get_type() == ColliderType::Box && by.rotation == 0.0 {
            self.translated(-original_centre)
                .scaled(by.scale)
                .translated(original_centre + by.centre)
        } else if let GenericCollider::OrientedBox(c) = self
            && c.rotation + by.rotation == 0.0
        {
            BoxCollider::from_aa_extent(c)
                .translated(-original_centre)
                .scaled(by.scale)
                .translated(original_centre + by.centre)
                .as_generic()
        } else {
            self.translated(-original_centre)
                .scaled(by.scale)
                .rotated(by.rotation)
                .translated(original_centre + by.centre)
        }
    }
}

pub struct GgInternalCollisionShape {
    base_collider: GenericCollider,
    collider: GenericCollider,
    emitting_tags: Vec<&'static str>,
    listening_tags: Vec<&'static str>,

    // For GUI:
    // <RenderItem, should_be_updated>
    wireframe: RenderItem,
    show_wireframe: bool,
    last_show_wireframe: bool,
    extent_cell_receiver_x: EditCell<f32>,
    extent_cell_receiver_y: EditCell<f32>,
    centre_cell_receiver_x: EditCell<f32>,
    centre_cell_receiver_y: EditCell<f32>,
}

#[cfg_attr(coverage_nightly, coverage(off))]
impl GgInternalCollisionShape {
    pub fn from_collider<C: Collider>(
        collider: C,
        emitting_tags: &[&'static str],
        listening_tags: &[&'static str],
    ) -> Self {
        let base_collider = collider.into_generic();
        let mut rv = Self {
            base_collider: base_collider.clone(),
            collider: base_collider,
            emitting_tags: emitting_tags.to_vec(),
            listening_tags: listening_tags.to_vec(),
            wireframe: RenderItem::default(),
            show_wireframe: false,
            last_show_wireframe: false,
            extent_cell_receiver_x: EditCell::new(),
            extent_cell_receiver_y: EditCell::new(),
            centre_cell_receiver_x: EditCell::new(),
            centre_cell_receiver_y: EditCell::new(),
        };
        rv.regenerate_wireframe();
        rv
    }

    pub fn from_object<O: SceneObject, C: Collider>(object: &O, collider: C) -> Self {
        Self::from_collider(collider, &object.emitting_tags(), &object.listening_tags())
    }
    pub fn from_object_sprite<O: SceneObject>(object: &O, sprite: &Sprite) -> Self {
        Self::from_collider(
            sprite.as_box_collider(),
            &object.emitting_tags(),
            &object.listening_tags(),
        )
    }

    pub fn collider(&self) -> &GenericCollider {
        &self.collider
    }

    fn regenerate_wireframe(&mut self) {
        self.wireframe =
            RenderItem::from_raw_vertices(self.base_collider.as_triangles().into_flattened())
                .with_depth(VertexDepth::max_value());
    }

    pub fn show_wireframe(&mut self) {
        self.show_wireframe = true;
    }
    pub fn hide_wireframe(&mut self) {
        self.show_wireframe = false;
    }
}

#[cfg_attr(coverage_nightly, coverage(off))]
#[partially_derive_scene_object]
impl SceneObject for GgInternalCollisionShape {
    fn gg_type_name(&self) -> String {
        format!("CollisionShape [{:?}]", self.collider.get_type()).to_string()
    }

    fn on_ready(&mut self, ctx: &mut UpdateContext) {
        check_is_some!(ctx.object().parent(), "CollisionShapes must have a parent");
    }
    fn on_update_begin(&mut self, ctx: &mut UpdateContext) {
        self.update_transform(ctx.absolute_transform());
    }
    fn on_fixed_update(&mut self, ctx: &mut FixedUpdateContext) {
        self.update_transform(ctx.absolute_transform());
    }

    fn on_update(&mut self, ctx: &mut UpdateContext) {
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

    fn on_update_end(&mut self, ctx: &mut UpdateContext) {
        self.update_transform(ctx.absolute_transform());
    }

    fn emitting_tags(&self) -> Vec<&'static str> {
        self.emitting_tags.clone()
    }
    fn listening_tags(&self) -> Vec<&'static str> {
        self.listening_tags.clone()
    }

    fn as_renderable_object(&mut self) -> Option<&mut dyn RenderableObject> {
        Some(self)
    }
    fn as_gui_object(&mut self) -> Option<&mut dyn GuiObject> {
        if self.show_wireframe {
            Some(self)
        } else {
            None
        }
    }
}

#[cfg_attr(coverage_nightly, coverage(off))]
impl GgInternalCollisionShape {
    pub(crate) fn update_transform(&mut self, next_transform: Transform) {
        self.collider = self.base_collider.transformed(&next_transform);
    }
}

#[cfg_attr(coverage_nightly, coverage(off))]
impl RenderableObject for GgInternalCollisionShape {
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
        vec![ShaderExec {
            blend_col: Colour::cyan().with_alpha(0.2),
            ..Default::default()
        }]
    }
}

#[cfg_attr(coverage_nightly, coverage(off))]
impl GuiObject for GgInternalCollisionShape {
    fn on_gui(&mut self, _ctx: &UpdateContext, selected: bool) -> GuiCommand {
        if !selected {
            self.extent_cell_receiver_x.clear_state();
            self.extent_cell_receiver_y.clear_state();
            self.centre_cell_receiver_x.clear_state();
            self.centre_cell_receiver_y.clear_state();
        }
        let extent = self.collider.extent();
        let (next_x, next_y) = (
            self.extent_cell_receiver_x.try_recv(),
            self.extent_cell_receiver_y.try_recv(),
        );
        if next_x.is_some() || next_y.is_some() {
            self.collider = self.collider.with_extent(Vec2 {
                x: next_x.unwrap_or(extent.x).max(0.1),
                y: next_y.unwrap_or(extent.y).max(0.1),
            });
            self.regenerate_wireframe();
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
            self.regenerate_wireframe();
        }
        self.centre_cell_receiver_x.update_live(centre.x);
        self.centre_cell_receiver_y.update_live(centre.y);

        let centre_cell_sender_x = self.centre_cell_receiver_x.sender();
        let centre_cell_sender_y = self.centre_cell_receiver_y.sender();

        let emitting_tags = self.emitting_tags.join(", ");
        let listening_tags = self.listening_tags.join(", ");

        let collider = self.collider.clone();
        GuiCommand::new(move |ui| {
            ui.label(collider.to_string());
            ui.add(egui::Label::new("Extent").selectable(false));
            collider
                .extent()
                .build_gui(ui, 0.1, extent_cell_sender_x, extent_cell_sender_y);
            ui.end_row();
            ui.add(egui::Label::new("Centre").selectable(false));
            collider
                .centre()
                .build_gui(ui, 0.1, centre_cell_sender_x, centre_cell_sender_y);
            ui.end_row();
            ui.add(egui::Label::new(format!("Emitting: {emitting_tags}")).selectable(false));
            ui.end_row();
            ui.add(egui::Label::new(format!("Listening: {listening_tags}")).selectable(false));
            ui.end_row();
        })
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

pub use GgInternalCollisionShape as CollisionShape;

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::core::prelude::*;
    use std::collections::HashSet;
    use std::f32::consts::SQRT_2;

    // ========== BoxCollider Tests ==========

    #[test]
    fn box_collider_no_collision() {
        let box1 = BoxCollider::from_centre(Vec2::zero(), Vec2::one());
        // Test x-axis separation
        let box2 = BoxCollider::from_centre(Vec2 { x: 10.0, y: 0.0 }, Vec2::one());
        assert!(box1.collides_with_box(&box2).is_none());
        // Test y-axis separation
        let box3 = BoxCollider::from_centre(Vec2 { x: 0.0, y: 10.0 }, Vec2::one());
        assert!(box1.collides_with_box(&box3).is_none());
    }

    #[test]
    fn box_collider_horizontal_overlap() {
        let box1 = BoxCollider::from_centre(Vec2::zero(), Vec2::one());
        let box2 = BoxCollider::from_centre(Vec2 { x: 1.0, y: 0.0 }, Vec2::one());
        let mtv = box1.collides_with_box(&box2);
        assert!(mtv.is_some());
        let mtv = mtv.unwrap();
        // Should push box1 to the left (negative x direction) by 1 unit (the overlap amount)
        assert_eq!(mtv.x, -1.0);
        assert_eq!(mtv.y, 0.0);
    }

    #[test]
    fn box_collider_fully_contained() {
        let box1 = BoxCollider::from_centre(Vec2::zero(), Vec2 { x: 5.0, y: 5.0 });
        let box2 = BoxCollider::from_centre(Vec2::zero(), Vec2::one());
        let mtv = box1.collides_with_box(&box2);
        assert!(mtv.is_some());
        let mtv = mtv.unwrap();
        // MTV should be 6 units along x-axis (sum of half-extents: 5 + 1 = 6)
        assert_eq!(mtv.x, 6.0);
        assert_eq!(mtv.y, 0.0);
        // Applying the MTV should separate the boxes
        let separated = box2.translated(mtv);
        assert!(box1.collides_with_box(&separated).is_none());
    }

    #[test]
    fn box_collider_touching_edges() {
        let box1 = BoxCollider::from_centre(Vec2::zero(), Vec2::one());
        let box2 = BoxCollider::from_centre(Vec2 { x: 2.0, y: 0.0 }, Vec2::one());
        // Boxes are exactly touching, should not collide
        let mtv = box1.collides_with_box(&box2);
        assert!(mtv.is_none());
    }

    #[test]
    fn box_collider_vertical_overlap() {
        let box1 = BoxCollider::from_centre(Vec2::zero(), Vec2::one());
        let box2 = BoxCollider::from_centre(Vec2 { x: 0.0, y: 1.0 }, Vec2::one());
        let mtv = box1.collides_with_box(&box2);
        assert!(mtv.is_some());
        let mtv = mtv.unwrap();
        // Should push box1 up (negative y direction) by 1 unit (the overlap amount)
        assert_eq!(mtv.x, 0.0);
        assert_eq!(mtv.y, -1.0);
    }

    #[test]
    fn box_collider_from_top_left() {
        let box1 = BoxCollider::from_top_left(Vec2::zero(), Vec2 { x: 4.0, y: 4.0 });
        assert_eq!(box1.extent(), Vec2 { x: 4.0, y: 4.0 });
        assert_eq!(box1.centre(), Vec2 { x: 2.0, y: 2.0 });
    }

    #[test]
    fn box_collider_translation() {
        let box1 = BoxCollider::from_centre(Vec2::zero(), Vec2::one());
        let box2 = box1.translated(Vec2 { x: 5.0, y: 3.0 });
        assert_eq!(box2.centre(), Vec2 { x: 5.0, y: 3.0 });
        assert_eq!(box2.extent(), box1.extent());
    }

    #[test]
    fn box_collider_scaling() {
        let box1 = BoxCollider::from_centre(Vec2::zero(), Vec2::one());
        assert_eq!(box1.extent(), Vec2 { x: 2.0, y: 2.0 });
        let box2 = box1.scaled(Vec2 { x: 2.0, y: 3.0 });
        assert_eq!(box2.extent(), Vec2 { x: 4.0, y: 6.0 });
        assert_eq!(box2.centre(), box1.centre());

        // Negative scaling still produces positive extent
        let box3 = box1.scaled(Vec2 { x: -2.0, y: -3.0 });
        assert_eq!(box3.extent(), Vec2 { x: 4.0, y: 6.0 });
        assert_eq!(box3.centre(), box1.centre());
    }

    // ========== OrientedBoxCollider Tests ==========

    #[test]
    fn oriented_box_no_rotation_no_collision() {
        let box1 = OrientedBoxCollider::from_centre(Vec2::zero(), Vec2::one());
        let box2 = OrientedBoxCollider::from_centre(Vec2 { x: 10.0, y: 0.0 }, Vec2::one());
        assert!(box1.collides_with_oriented_box(&box2).is_none());
    }

    #[test]
    fn oriented_box_no_rotation_collision() {
        let box1 = OrientedBoxCollider::from_centre(Vec2::zero(), Vec2::one());
        let box2 = OrientedBoxCollider::from_centre(Vec2 { x: 1.0, y: 0.0 }, Vec2::one());
        let mtv = box1.collides_with_oriented_box(&box2);
        assert!(mtv.is_some());
        let mtv = mtv.unwrap();
        assert_eq!(mtv.x, -1.0);
        assert_eq!(mtv.y, 0.0);
    }

    #[test]
    fn oriented_box_45_degree_rotation() {
        let box1 = OrientedBoxCollider::from_centre(Vec2::zero(), Vec2::one());
        let box2 = box1.rotated(45_f32.to_radians());
        // Rotated box should have same center
        assert_eq!(box2.centre(), box1.centre());
        // The rotation value should be updated
        assert!((box2.rotation - 45_f32.to_radians()).abs() < EPSILON);
    }

    #[test]
    fn oriented_box_translation() {
        let box1 = OrientedBoxCollider::from_centre(Vec2::zero(), Vec2::one())
            .rotated(45_f32.to_radians());
        let translation = Vec2 { x: 5.0, y: 3.0 };
        let box2 = box1.translated(translation);
        // Translation moves the centre directly without regard to rotation
        assert_eq!(box2.centre(), translation);
    }

    #[test]
    fn oriented_box_collides_with_regular_box() {
        let box1 = BoxCollider::from_centre(Vec2::zero(), Vec2::one());
        let oriented = OrientedBoxCollider::from_centre(Vec2 { x: 1.0, y: 0.0 }, Vec2::one())
            .rotated(45_f32.to_radians());
        let mtv = box1.collides_with_oriented_box(&oriented);
        assert!(mtv.is_some());
        let mtv = mtv.unwrap();
        // The rotated box's corner is at x = 1 - sqrt(2), and box1's right edge is at x = 1.
        assert!((mtv.x - -2_f32.sqrt()).abs() < EPSILON);
        assert_eq!(mtv.y, 0.0);
    }

    #[test]
    fn oriented_box_rotated_collision() {
        let box1 = OrientedBoxCollider::from_centre(Vec2::zero(), Vec2 { x: 2.0, y: 1.0 });
        let box2_unrotated =
            OrientedBoxCollider::from_centre(Vec2 { x: 3.5, y: 0.0 }, Vec2 { x: 1.0, y: 2.0 });
        // Without rotation, they don't collide (box1 right edge at x=2, box2 left edge at x=2.5)
        assert!(box1.collides_with_oriented_box(&box2_unrotated).is_none());
        // With 45-degree rotation, the box extends further left and they collide.
        // The rotated box's leftmost corner is at x = 3.5 - 3*sqrt(2)/2 ~= 1.38, overlapping
        // box1's right edge at x=2. Penetration depth = 2 - (3.5 - 3*sqrt(2)/2) = 1.5*(sqrt(2) - 1).
        let box2 = box2_unrotated.rotated(45_f32.to_radians());
        let mtv = box1.collides_with_oriented_box(&box2);
        assert!(mtv.is_some());
        let mtv = mtv.unwrap();
        assert!((mtv.x - 1.5 * (1.0 - 2_f32.sqrt())).abs() < EPSILON);
        assert_eq!(mtv.y, 0.0);
    }

    // ========== ConvexCollider Tests ==========

    #[test]
    fn convex_triangle_no_collision() {
        let verts1 = vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 2.0, y: 0.0 },
            Vec2 { x: 1.0, y: 2.0 },
        ];
        let triangle1 = ConvexCollider::convex_hull_of(verts1.clone()).unwrap();
        assert_eq!(triangle1.vertices(), verts1);
        let verts2 = vec![
            Vec2 { x: 10.0, y: 0.0 },
            Vec2 { x: 12.0, y: 0.0 },
            Vec2 { x: 11.0, y: 2.0 },
        ];
        let triangle2 = ConvexCollider::convex_hull_of(verts2.clone()).unwrap();
        assert_eq!(triangle2.vertices(), verts2);
        assert!(triangle1.collides_with_convex(&triangle2).is_none());
    }

    #[test]
    fn convex_triangle_collision() {
        let triangle1 = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 2.0, y: 0.0 },
            Vec2 { x: 1.0, y: 2.0 },
        ])
        .unwrap();
        let triangle2 = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 1.0, y: 0.0 },
            Vec2 { x: 3.0, y: 0.0 },
            Vec2 { x: 2.0, y: 2.0 },
        ])
        .unwrap();
        let mtv = triangle1.collides_with_convex(&triangle2);
        assert!(mtv.is_some());
        let mtv = mtv.unwrap();
        // MTV is along the normal to triangle1's left edge (0,0)->(1,2), direction (-2,1) normalized.
        // Penetration depth is 2/sqrt(5), so MTV = (-4/5, 2/5).
        assert!((mtv.x - (-0.8)).abs() < EPSILON);
        assert!((mtv.y - 0.4).abs() < EPSILON);
    }

    #[test]
    fn convex_hull_from_square_vertices() {
        let verts = vec![
            Vec2 { x: -1.0, y: -1.0 },
            Vec2 { x: 1.0, y: -1.0 },
            Vec2 { x: 1.0, y: 1.0 },
            Vec2 { x: -1.0, y: 1.0 },
        ];
        let square = ConvexCollider::convex_hull_of(verts.clone()).unwrap();
        assert_eq!(square.vertices(), verts);
    }

    #[test]
    fn convex_hull_removes_collinear_points() {
        // Add extra point on the edge
        let square = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: -1.0, y: -1.0 },
            Vec2 { x: 0.0, y: -1.0 }, // Collinear point
            Vec2 { x: 1.0, y: -1.0 },
            Vec2 { x: 1.0, y: 1.0 },
            Vec2 { x: -1.0, y: 1.0 },
        ])
        .unwrap();
        // Collinear point should be removed, leaving just the 4 corners
        let expected = vec![
            Vec2 { x: -1.0, y: -1.0 },
            Vec2 { x: 1.0, y: -1.0 },
            Vec2 { x: 1.0, y: 1.0 },
            Vec2 { x: -1.0, y: 1.0 },
        ];
        assert_eq!(square.vertices(), expected);

        // Test with collinear point on a sloped edge (slope = 2/3)
        let triangle = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 1.5, y: 1.0 }, // Collinear point on edge from (0,0) to (3,2)
            Vec2 { x: 3.0, y: 2.0 },
            Vec2 { x: 0.0, y: 2.0 },
        ])
        .unwrap();
        let expected = vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 3.0, y: 2.0 },
            Vec2 { x: 0.0, y: 2.0 },
        ];
        assert_eq!(triangle.vertices(), expected);
    }

    #[test]
    fn convex_collider_translation() {
        let verts = vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 2.0, y: 0.0 },
            Vec2 { x: 1.0, y: 2.0 },
        ];
        let triangle = ConvexCollider::convex_hull_of(verts.clone()).unwrap();
        let offset = Vec2 { x: 5.0, y: 3.0 };
        let translated = triangle.translated(offset);
        assert_eq!(translated.centre(), triangle.centre() + offset);
        let expected_verts: Vec<Vec2> = verts.iter().map(|v| *v + offset).collect();
        assert_eq!(translated.vertices(), expected_verts);
    }

    #[test]
    fn convex_collider_rotation() {
        // Create a triangle - rotation happens around the centre
        let verts = vec![
            Vec2 { x: 1.0, y: 0.0 },
            Vec2 { x: -1.0, y: 0.0 },
            Vec2 { x: 0.0, y: 1.0 },
        ];
        let triangle = ConvexCollider::convex_hull_of(verts.clone()).unwrap();
        let original_centre = triangle.centre();
        let angle = 90_f32.to_radians();
        let rotated = triangle.rotated(angle);
        // Rotation around centre preserves the centre
        assert_eq!(rotated.centre(), original_centre);
        // Vertices are rotated around the centre
        let expected_verts: Vec<Vec2> = verts
            .iter()
            .map(|v| (*v - original_centre).rotated(angle) + original_centre)
            .collect();
        let rotated_verts = rotated.vertices();
        assert_eq!(rotated_verts.len(), expected_verts.len());
        let expected_set: HashSet<_> = expected_verts.into_iter().collect();
        let actual_set: HashSet<_> = rotated_verts.iter().copied().collect();
        assert_eq!(actual_set, expected_set);

        // Test with an irregular pentagon not centered at origin
        let pentagon_verts = vec![
            Vec2 { x: 5.0, y: 3.0 },
            Vec2 { x: 4.0, y: 4.5 },
            Vec2 { x: 2.0, y: 4.0 },
            Vec2 { x: 1.5, y: 2.5 },
            Vec2 { x: 3.5, y: 2.0 },
        ];
        let pentagon = ConvexCollider::convex_hull_of(pentagon_verts.clone()).unwrap();
        let pentagon_centre = pentagon.centre();
        assert_ne!(pentagon_centre, Vec2::zero());
        let angle = 37_f32.to_radians();
        let rotated = pentagon.rotated(angle);
        // Rotation around centre preserves the centre
        assert_eq!(rotated.centre(), pentagon_centre);
        // Vertices are rotated around the centre
        let expected_verts: Vec<Vec2> = pentagon_verts
            .iter()
            .map(|v| (*v - pentagon_centre).rotated(angle) + pentagon_centre)
            .collect();
        let rotated_verts = rotated.vertices();
        assert_eq!(rotated_verts.len(), expected_verts.len());
        let expected_set: HashSet<_> = expected_verts.into_iter().collect();
        let actual_set: HashSet<_> = rotated_verts.iter().copied().collect();
        assert_eq!(actual_set, expected_set);
    }

    #[test]
    fn convex_collider_with_box() {
        let triangle = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 2.0, y: 0.0 },
            Vec2 { x: 1.0, y: 2.0 },
        ])
        .unwrap();
        let box_collider = BoxCollider::from_centre(Vec2 { x: 1.0, y: 0.5 }, Vec2::one());
        let mtv = triangle.collides_with_box(&box_collider);
        assert!(mtv.is_some());
        let mtv = mtv.unwrap();
        assert_eq!(mtv.x, 0.0);
        assert_eq!(mtv.y, 1.5);
    }

    // ========== GenericCollider Tests ==========

    #[test]
    fn generic_collider_box_to_box() {
        let box1 = BoxCollider::from_centre(Vec2::zero(), Vec2::one());
        let box2 = BoxCollider::from_centre(Vec2 { x: 1.0, y: 0.0 }, Vec2::one());
        let generic1 = box1.as_generic();
        let generic2 = box2.as_generic();
        assert!(generic1.collides_with(&generic2).is_some());
    }

    #[test]
    fn generic_collider_oriented_to_box() {
        let box1 = BoxCollider::from_centre(Vec2::zero(), Vec2::one());
        let oriented = OrientedBoxCollider::from_centre(Vec2 { x: 1.0, y: 0.0 }, Vec2::one());
        let generic1 = box1.as_generic();
        let generic2 = oriented.as_generic();
        assert!(generic1.collides_with(&generic2).is_some());
    }

    #[test]
    fn generic_collider_null_never_collides() {
        let null = NullCollider;
        let box_collider = BoxCollider::from_centre(Vec2::zero(), Vec2::one());
        let generic_null = null.as_generic();
        let generic_box = box_collider.as_generic();
        assert!(generic_null.collides_with(&generic_box).is_none());
        assert!(generic_box.collides_with(&generic_null).is_none());
    }

    // ========== BoxCollider3d Tests ==========

    #[test]
    fn box_3d_no_collision_x_axis() {
        let box2d = BoxCollider::from_centre(Vec2::zero(), Vec2::one());
        let box1 = BoxCollider3d::from_2d(&box2d, -1.0, 1.0);
        let box2d2 = BoxCollider::from_centre(Vec2 { x: 10.0, y: 0.0 }, Vec2::one());
        let box2 = BoxCollider3d::from_2d(&box2d2, -1.0, 1.0);
        assert!(box1.collides_with(&box2).is_none());
    }

    #[test]
    fn box_3d_no_collision_z_axis() {
        let box2d = BoxCollider::from_centre(Vec2::zero(), Vec2::one());
        let box1 = BoxCollider3d::from_2d(&box2d, -1.0, 1.0);
        let box2 = BoxCollider3d::from_2d(&box2d, 5.0, 7.0);
        assert!(box1.collides_with(&box2).is_none());
    }

    #[test]
    fn box_3d_collision() {
        let box2d = BoxCollider::from_centre(Vec2::zero(), Vec2::one());
        let box1 = BoxCollider3d::from_2d(&box2d, -1.0, 1.0);
        let box2d2 = BoxCollider::from_centre(Vec2 { x: 1.0, y: 0.0 }, Vec2::one());
        let box2 = BoxCollider3d::from_2d(&box2d2, -0.5, 0.5);
        let mtv = box1.collides_with(&box2);
        assert!(mtv.is_some());
        let mtv = mtv.unwrap();
        // Minimum overlap is 1 unit in x-direction
        assert_eq!(mtv.0.x, -1.0);
        assert_eq!(mtv.0.y, 0.0);
        assert_eq!(mtv.1, 0.0);
    }

    // ========== Edge Cases and Special Scenarios ==========

    #[test]
    fn identical_boxes_collide() {
        let box1 = BoxCollider::from_centre(Vec2::zero(), Vec2::one());
        let box2 = BoxCollider::from_centre(Vec2::zero(), Vec2::one());
        let mtv = box1.collides_with_box(&box2);
        assert!(mtv.is_some());
        let mtv = mtv.unwrap();
        // Full overlap of 2 units in both axes, algorithm picks x
        assert_eq!(mtv.x, 2.0);
        assert_eq!(mtv.y, 0.0);
    }

    #[test]
    fn zero_size_box() {
        let box1 = BoxCollider::from_centre(Vec2::zero(), Vec2::zero());
        let box2 = BoxCollider::from_centre(Vec2::zero(), Vec2::one());
        // Zero-size box should not collide
        assert!(box1.collides_with_box(&box2).is_none());
        assert!(box2.collides_with_box(&box1).is_none());
    }

    #[test]
    fn negative_extent_converted_to_positive() {
        let box1 = BoxCollider::from_centre(Vec2::zero(), Vec2 { x: -2.0, y: -3.0 });
        // Negative extents should be converted to positive
        assert_eq!(box1.extent(), Vec2 { x: 4.0, y: 6.0 });
    }

    #[test]
    fn polygon_convexity_check() {
        // Square is convex
        let square_vertices = vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 1.0, y: 0.0 },
            Vec2 { x: 1.0, y: 1.0 },
            Vec2 { x: 0.0, y: 1.0 },
        ];
        assert!(polygon::is_convex(&square_vertices));

        // Triangle is convex
        let triangle_vertices = vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 1.0, y: 0.0 },
            Vec2 { x: 0.5, y: 1.0 },
        ];
        assert!(polygon::is_convex(&triangle_vertices));

        // Arrow shape is concave (has inward-pointing vertex)
        let arrow_vertices = vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 2.0, y: 0.0 },
            Vec2 { x: 2.0, y: 1.0 },
            Vec2 { x: 1.0, y: 0.5 }, // Inward-pointing vertex
            Vec2 { x: 0.0, y: 1.0 },
        ];
        assert!(!polygon::is_convex(&arrow_vertices));
    }

    // ========== Additional NullCollider Tests ==========

    #[test]
    fn null_collider_extent_and_centre() {
        let null = NullCollider;
        assert_eq!(null.extent(), Vec2::zero());
        assert_eq!(null.centre(), Vec2::zero());
    }

    #[test]
    fn null_collider_transformations() {
        let null = NullCollider;
        let translated = null.translated(Vec2 { x: 5.0, y: 3.0 });
        let scaled = null.scaled(Vec2 { x: 2.0, y: 2.0 });
        let rotated = null.rotated(45_f32.to_radians());
        // All should remain NullCollider
        assert_eq!(translated.centre(), Vec2::zero());
        assert_eq!(scaled.extent(), Vec2::zero());
        assert_eq!(rotated.centre(), Vec2::zero());
    }

    #[test]
    fn null_collider_collides_with_oriented_box() {
        let null = NullCollider;
        let oriented = OrientedBoxCollider::from_centre(Vec2::zero(), Vec2::one());
        assert!(null.collides_with_oriented_box(&oriented).is_none());
        assert!(
            oriented
                .as_generic()
                .collides_with(&null.as_generic())
                .is_none()
        );
    }

    #[test]
    fn null_collider_collides_with_convex() {
        let null = NullCollider;
        let triangle = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 2.0, y: 0.0 },
            Vec2 { x: 1.0, y: 2.0 },
        ])
        .unwrap();
        assert!(null.collides_with_convex(&triangle).is_none());
        assert!(
            triangle
                .as_generic()
                .collides_with(&null.as_generic())
                .is_none()
        );
    }

    #[test]
    fn null_collider_as_polygon_and_triangles() {
        let null = NullCollider;
        assert_eq!(null.as_polygon().len(), 0);
        assert_eq!(null.as_triangles().len(), 0);
    }

    // ========== Additional OrientedBoxCollider Tests ==========

    #[test]
    fn oriented_box_from_top_left() {
        let box1 = OrientedBoxCollider::from_top_left(Vec2::zero(), Vec2 { x: 4.0, y: 4.0 });
        assert_eq!(box1.centre(), Vec2 { x: 2.0, y: 2.0 });
    }

    #[test]
    fn oriented_box_from_transform() {
        let transform = Transform {
            centre: Vec2 { x: 5.0, y: 5.0 },
            rotation: 45_f32.to_radians(),
            scale: Vec2 { x: 2.0, y: 2.0 },
        };
        let box1 = OrientedBoxCollider::from_transform(transform, Vec2::one());
        assert_eq!(box1.centre(), Vec2 { x: 5.0, y: 5.0 });
        assert!((box1.rotation - 45_f32.to_radians()).abs() < EPSILON);
        // extent() returns AABB extent, which is 4*sqrt(2) when rotated 45 degrees
        assert_eq!(box1.extent(), Vec2::splat(4.0 * 2_f32.sqrt()));
    }

    #[test]
    fn oriented_box_square() {
        let transform = Transform {
            centre: Vec2::zero(),
            rotation: 0.0,
            scale: Vec2 { x: 2.0, y: 2.0 },
        };
        let box1 = OrientedBoxCollider::square(transform, 5.0);
        assert_eq!(box1.centre(), Vec2::zero());
        assert_eq!(box1.extent(), Vec2::splat(20.0));
    }

    #[test]
    fn oriented_box_scaled() {
        let box1 = OrientedBoxCollider::from_centre(Vec2::zero(), Vec2::one());
        assert_eq!(box1.extent(), Vec2 { x: 2.0, y: 2.0 });
        let box2 = box1.scaled(Vec2 { x: 2.0, y: 3.0 });
        assert_eq!(box2.centre(), box1.centre());
        assert_eq!(box2.extent(), Vec2 { x: 4.0, y: 6.0 });
    }

    #[test]
    fn oriented_box_collides_with_box() {
        let centre = Vec2 { x: 3.0, y: 2.0 };
        let oriented = OrientedBoxCollider::from_centre(centre, Vec2::one());
        let box1 = BoxCollider::from_centre(centre, Vec2::one());
        let box2 = BoxCollider::from_centre(Vec2 { x: 4.0, y: 2.0 }, Vec2::one());
        let mtv_oriented = oriented.collides_with_box(&box2);
        let mtv_box = box1.collides_with_box(&box2);
        assert!(mtv_oriented.is_some());
        assert_eq!(mtv_oriented, mtv_box);
    }

    #[test]
    fn oriented_box_collides_with_convex() {
        let centre = Vec2 { x: 3.0, y: 2.0 };
        let oriented = OrientedBoxCollider::from_centre(centre, Vec2::one());
        let box1 = BoxCollider::from_centre(centre, Vec2::one());
        let triangle = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 2.5, y: 1.5 },
            Vec2 { x: 5.0, y: 2.0 },
            Vec2 { x: 4.0, y: 4.0 },
        ])
        .unwrap();
        let mtv_oriented = oriented.collides_with_convex(&triangle);
        let mtv_box = box1.collides_with_convex(&triangle);
        assert!(mtv_oriented.is_some());
        assert_eq!(mtv_oriented, mtv_box);
    }

    #[test]
    fn oriented_box_as_polygon_and_triangles() {
        let oriented = OrientedBoxCollider::from_centre(Vec2::zero(), Vec2::one());
        let polygon = oriented.as_polygon();
        assert_eq!(polygon.len(), 4);
        assert_eq!(polygon[0], Vec2 { x: -1.0, y: -1.0 });
        assert_eq!(polygon[1], Vec2 { x: 1.0, y: -1.0 });
        assert_eq!(polygon[2], Vec2 { x: 1.0, y: 1.0 });
        assert_eq!(polygon[3], Vec2 { x: -1.0, y: 1.0 });
        let triangles = oriented.as_triangles();
        assert_eq!(triangles.len(), 2);
        assert_eq!(triangles[0], [polygon[0], polygon[1], polygon[3]]);
        assert_eq!(triangles[1], [polygon[1], polygon[2], polygon[3]]);
    }

    #[test]
    fn oriented_box_extent() {
        let oriented = OrientedBoxCollider::from_centre(Vec2::zero(), Vec2::one());
        assert_eq!(oriented.extent(), Vec2::splat(2.0));
    }

    // ========== Additional BoxCollider Tests ==========

    #[test]
    fn box_collider_from_aa_extent() {
        let box1 = BoxCollider::from_centre(Vec2 { x: 5.0, y: 5.0 }, Vec2 { x: 2.0, y: 3.0 });
        let box2 = BoxCollider::from_aa_extent(&box1);
        assert_eq!(box2.centre(), box1.centre());
        assert_eq!(box2.extent(), box1.extent());
    }

    #[test]
    fn box_collider_from_transform() {
        let transform = Transform {
            centre: Vec2 { x: 5.0, y: 5.0 },
            rotation: 0.0,
            scale: Vec2 { x: 2.0, y: 2.0 },
        };
        let box1 = BoxCollider::from_transform(transform, Vec2 { x: 2.0, y: 2.0 });
        assert_eq!(box1.centre(), Vec2 { x: 5.0, y: 5.0 });
        assert_eq!(box1.extent(), Vec2::splat(8.0));
    }

    #[test]
    fn box_collider_square() {
        let transform = Transform {
            centre: Vec2::zero(),
            rotation: 0.0,
            scale: Vec2::one(),
        };
        let box1 = BoxCollider::square(transform, 5.0);
        assert_eq!(box1.extent(), Vec2::splat(10.0));
    }

    #[test]
    fn box_collider_as_convex() {
        let box1 = BoxCollider::from_centre(Vec2::zero(), Vec2::one());
        let convex = box1.as_convex();
        let verts = convex.vertices();
        assert_eq!(verts.len(), 4);
        assert_eq!(verts[0], Vec2 { x: -1.0, y: -1.0 });
        assert_eq!(verts[1], Vec2 { x: 1.0, y: -1.0 });
        assert_eq!(verts[2], Vec2 { x: 1.0, y: 1.0 });
        assert_eq!(verts[3], Vec2 { x: -1.0, y: 1.0 });
    }

    #[test]
    fn box_collider_rotated_zero() {
        let box1 = BoxCollider::from_centre(Vec2::zero(), Vec2::one());
        let box2 = box1.rotated(0.0);
        assert_eq!(box2.centre(), box1.centre());
        assert_eq!(box2.extent(), box1.extent());
    }

    #[test]
    fn box_collider_rotated_returns_oriented() {
        let box1 = BoxCollider::from_centre(Vec2 { x: 5.0, y: 10.0 }, Vec2::one());
        let rotated: OrientedBoxCollider = box1.rotated(45_f32.to_radians());
        assert_eq!(rotated.centre, box1.centre);
        assert!((rotated.rotation - 45_f32.to_radians()).abs() < 1e-6);
        assert_eq!(rotated.unrotated_half_widths, box1.half_widths());
        // 45-degree rotated square has extent sqrt(2) * side_length in both dimensions
        let expected_extent = 2.0 * 2.0_f32.sqrt();
        assert!((rotated.extent.x - expected_extent).abs() < 1e-5);
        assert!((rotated.extent.y - expected_extent).abs() < 1e-5);
    }

    #[test]
    fn box_collider_as_polygon_and_triangles() {
        let box1 = BoxCollider::from_centre(Vec2::zero(), Vec2::one());
        let polygon = box1.as_polygon();
        assert_eq!(polygon.len(), 4);
        assert_eq!(polygon[0], Vec2 { x: -1.0, y: -1.0 });
        assert_eq!(polygon[1], Vec2 { x: 1.0, y: -1.0 });
        assert_eq!(polygon[2], Vec2 { x: 1.0, y: 1.0 });
        assert_eq!(polygon[3], Vec2 { x: -1.0, y: 1.0 });
        let triangles = box1.as_triangles();
        assert_eq!(triangles.len(), 2);
        assert_eq!(triangles[0], [polygon[0], polygon[1], polygon[3]]);
        assert_eq!(triangles[1], [polygon[1], polygon[2], polygon[3]]);
    }

    #[test]
    fn box_collider_vertical_collision_direction() {
        let box1 = BoxCollider::from_centre(Vec2::zero(), Vec2::one());
        let box2 = BoxCollider::from_centre(Vec2 { x: 0.0, y: 1.5 }, Vec2::one());
        let mtv = box1.collides_with_box(&box2).unwrap();
        // box1 spans y=[-1,1], box2 spans y=[0.5,2.5], overlap is 0.5
        assert_eq!(mtv.x, 0.0);
        assert_eq!(mtv.y, -0.5);
    }

    #[test]
    fn box_collider_horizontal_collision_opposite_direction() {
        let box1 = BoxCollider::from_centre(Vec2 { x: 1.5, y: 0.0 }, Vec2::one());
        let box2 = BoxCollider::from_centre(Vec2::zero(), Vec2::one());
        let mtv = box1.collides_with_box(&box2).unwrap();
        // box1 spans x=[0.5,2.5], box2 spans x=[-1,1], overlap is 0.5
        assert_eq!(mtv.x, 0.5);
        assert_eq!(mtv.y, 0.0);
    }

    // ========== Additional ConvexCollider Tests ==========

    #[test]
    fn convex_collider_extent() {
        let triangle = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 2.0, y: 0.0 },
            Vec2 { x: 1.0, y: 2.0 },
        ])
        .unwrap();
        // Triangle spans x=[0,2], y=[0,2]
        assert_eq!(triangle.extent(), Vec2::splat(2.0));

        // Irregular convex heptagon not centered at origin
        let heptagon = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 5.0, y: 3.0 },
            Vec2 { x: 7.0, y: 3.5 },
            Vec2 { x: 8.0, y: 5.0 },
            Vec2 { x: 7.5, y: 7.0 },
            Vec2 { x: 6.0, y: 8.0 },
            Vec2 { x: 4.0, y: 7.0 },
            Vec2 { x: 3.5, y: 5.0 },
        ])
        .unwrap();
        // Heptagon spans x=[3.5,8], y=[3,8]
        assert_eq!(heptagon.extent(), Vec2 { x: 4.5, y: 5.0 });

        // Same heptagon but with one vertex moved inward to make it concave
        let concave_heptagon = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 5.0, y: 3.0 },
            Vec2 { x: 7.0, y: 3.5 },
            Vec2 { x: 8.0, y: 5.0 },
            Vec2 { x: 7.5, y: 7.0 },
            Vec2 { x: 6.0, y: 5.5 }, // Moved inward from (6,8)
            Vec2 { x: 4.0, y: 7.0 },
            Vec2 { x: 3.5, y: 5.0 },
        ])
        .unwrap();
        // Convex hull excludes the inward vertex, spans x=[3.5,8], y=[3,7]
        assert_eq!(concave_heptagon.extent(), Vec2 { x: 4.5, y: 4.0 });
    }

    #[test]
    fn convex_collider_get_type() {
        let triangle = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 2.0, y: 0.0 },
            Vec2 { x: 1.0, y: 2.0 },
        ])
        .unwrap();
        assert_eq!(triangle.get_type(), ColliderType::Convex);
    }

    #[test]
    fn convex_collider_into_generic() {
        let triangle = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 2.0, y: 0.0 },
            Vec2 { x: 1.0, y: 2.0 },
        ])
        .unwrap();
        let generic = triangle.into_generic();
        assert_eq!(generic.get_type(), ColliderType::Convex);
    }

    #[test]
    fn convex_collider_scaled() {
        let triangle = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 2.0, y: 0.0 },
            Vec2 { x: 1.0, y: 2.0 },
        ])
        .unwrap();
        let original_centre = triangle.centre();
        let scaled = triangle.scaled(Vec2::splat(2.0));
        // Scaling preserves centre
        assert_eq!(scaled.centre(), original_centre);
        // Extent doubles
        assert_eq!(scaled.extent(), triangle.extent() * 2.0);
        // Vertices scaled around centre (1, 2/3):
        // (0,0) -> (-1, -2/3), (2,0) -> (3, -2/3), (1,2) -> (1, 10/3)
        let expected: HashSet<_> = [
            Vec2 {
                x: -1.0,
                y: -2.0 / 3.0,
            },
            Vec2 {
                x: 3.0,
                y: -2.0 / 3.0,
            },
            Vec2 {
                x: 1.0,
                y: 10.0 / 3.0,
            },
        ]
        .into_iter()
        .collect();
        let actual: HashSet<_> = scaled.vertices().iter().copied().collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn convex_collider_collides_with_oriented_box() {
        let triangle = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 2.0, y: 0.0 },
            Vec2 { x: 1.0, y: 2.0 },
        ])
        .unwrap();
        let oriented = OrientedBoxCollider::from_centre(Vec2 { x: 1.0, y: 0.5 }, Vec2::one());
        let mtv = triangle.collides_with_oriented_box(&oriented);
        assert!(mtv.is_some());
        // Converse check
        let mtv_converse = oriented.collides_with_convex(&triangle);
        assert!(mtv_converse.is_some());
        // MTVs should be negatives of each other
        assert_eq!(mtv.unwrap(), -mtv_converse.unwrap());
    }

    #[test]
    fn convex_collider_as_polygon_and_triangles() {
        let triangle = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 2.0, y: 0.0 },
            Vec2 { x: 1.0, y: 2.0 },
        ])
        .unwrap();
        assert_eq!(triangle.as_polygon().len(), 3);
        // Check polygon vertices match input
        let expected_verts: HashSet<_> = [
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 2.0, y: 0.0 },
            Vec2 { x: 1.0, y: 2.0 },
        ]
        .into_iter()
        .collect();
        let actual_verts: HashSet<_> = triangle.as_polygon().into_iter().collect();
        assert_eq!(actual_verts, expected_verts);
        // Check triangulation
        assert_eq!(triangle.as_triangles().len(), 1);
        let tri_verts: HashSet<_> = triangle.as_triangles()[0].into_iter().collect();
        assert_eq!(tri_verts, expected_verts);
    }

    // ========== Additional BoxCollider3d Tests ==========

    #[test]
    fn box_3d_vertical_collision() {
        let box2d1 = BoxCollider::from_centre(Vec2 { x: 0.0, y: 0.0 }, Vec2::one());
        let box1 = BoxCollider3d::from_2d(&box2d1, -1.0, 1.0);
        let box2d2 = BoxCollider::from_centre(Vec2 { x: 0.0, y: 1.5 }, Vec2::one());
        let box2 = BoxCollider3d::from_2d(&box2d2, -1.0, 1.0);
        let result = box1.collides_with(&box2);
        assert!(result.is_some());
        // box1 spans y=[-1,1], box2 spans y=[0.5,2.5], overlap is 0.5
        let (mtv, depth_overlap) = result.unwrap();
        assert_eq!(mtv, Vec2 { x: 0.0, y: -0.5 });
        assert_eq!(depth_overlap, 0.0); // Both have same depth [-1,1]
        // Converse check
        let result_converse = box2.collides_with(&box1);
        assert!(result_converse.is_some());
        let (mtv_converse, _) = result_converse.unwrap();
        assert_eq!(mtv_converse, -mtv);
    }

    #[test]
    fn box_3d_depth_collision() {
        let box2d = BoxCollider::from_centre(Vec2::zero(), Vec2::one());
        let box1 = BoxCollider3d::from_2d(&box2d, 0.0, 2.0);
        let box2 = BoxCollider3d::from_2d(&box2d, 1.5, 3.5);
        let result = box1.collides_with(&box2);
        assert!(result.is_some());
        // box1 depth=[0,2], box2 depth=[1.5,3.5], depth overlap is 0.5
        // 2D overlap is 2.0, so depth is the shortest separation axis
        let (mtv, depth_overlap) = result.unwrap();
        assert_eq!(mtv, Vec2::zero());
        assert_eq!(depth_overlap, 0.5);
        // Converse check
        let result_converse = box2.collides_with(&box1);
        assert!(result_converse.is_some());
        let (mtv_converse, depth_converse) = result_converse.unwrap();
        assert_eq!(mtv_converse, Vec2::zero());
        assert_eq!(depth_converse, 0.5);
    }

    // ========== Collider Trait Helper Methods Tests ==========

    #[test]
    fn collider_transformed() {
        let box1 = BoxCollider::from_centre(Vec2::zero(), Vec2::one());
        let transform = Transform {
            centre: Vec2 { x: 5.0, y: 3.0 },
            rotation: 0.0,
            scale: Vec2 { x: 2.0, y: 2.0 },
        };
        let transformed = box1.transformed(&transform);
        assert_eq!(transformed.centre(), Vec2 { x: 5.0, y: 3.0 });
        assert_eq!(transformed.extent(), Vec2 { x: 4.0, y: 4.0 });
    }

    #[test]
    fn collider_with_half_widths() {
        let box1 = BoxCollider::from_centre(Vec2::zero(), Vec2::one());
        let box2 = box1.with_half_widths(Vec2 { x: 2.0, y: 3.0 });
        assert_eq!(box2.centre(), Vec2::zero());
        assert_eq!(box2.extent(), Vec2 { x: 4.0, y: 6.0 });

        // Non-zero centre should be preserved
        let box3 = BoxCollider::from_centre(Vec2 { x: 5.0, y: 7.0 }, Vec2::one());
        let box4 = box3.with_half_widths(Vec2 { x: 1.5, y: 2.5 });
        assert_eq!(box4.centre(), Vec2 { x: 5.0, y: 7.0 });
        assert_eq!(box4.extent(), Vec2 { x: 3.0, y: 5.0 });
    }

    #[test]
    fn collider_with_extent() {
        let box1 = BoxCollider::from_centre(Vec2::zero(), Vec2::one());
        let box2 = box1.with_extent(Vec2 { x: 4.0, y: 6.0 });
        assert_eq!(box2.centre(), Vec2::zero());
        assert_eq!(box2.extent(), Vec2 { x: 4.0, y: 6.0 });

        // Non-zero centre should be preserved
        let box3 = BoxCollider::from_centre(Vec2 { x: 5.0, y: 7.0 }, Vec2::one());
        let box4 = box3.with_extent(Vec2 { x: 3.0, y: 5.0 });
        assert_eq!(box4.centre(), Vec2 { x: 5.0, y: 7.0 });
        assert_eq!(box4.extent(), Vec2 { x: 3.0, y: 5.0 });
    }

    #[test]
    fn collider_with_centre() {
        let box1 = BoxCollider::from_centre(Vec2::zero(), Vec2::one());
        let box2 = box1.with_centre(Vec2 { x: 5.0, y: 3.0 });
        assert_eq!(box2.centre(), Vec2 { x: 5.0, y: 3.0 });
        assert_eq!(box2.extent(), box1.extent());
    }

    // ========== GenericCollider Convex Variant Tests ==========

    #[test]
    fn generic_collider_convex_variant() {
        let triangle = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 2.0, y: 0.0 },
            Vec2 { x: 1.0, y: 2.0 },
        ])
        .unwrap();
        let box_collider = BoxCollider::from_centre(Vec2 { x: 1.0, y: 0.5 }, Vec2::one());
        let generic_convex = triangle.as_generic();
        let generic_box = box_collider.as_generic();
        let mtv = generic_convex.collides_with(&generic_box);
        assert!(mtv.is_some());
        // Converse check
        let mtv_converse = generic_box.collides_with(&generic_convex);
        assert!(mtv_converse.is_some());
        assert_eq!(mtv.unwrap(), -mtv_converse.unwrap());
    }

    // ========== Additional Edge Case Tests ==========

    #[test]
    fn null_collider_as_any() {
        let null = NullCollider;
        let any_ref = null.as_any();
        assert!(any_ref.downcast_ref::<NullCollider>().is_some());
    }

    #[test]
    fn box_collider_collides_with_convex() {
        let box1 = BoxCollider::from_centre(Vec2::zero(), Vec2::one());
        let triangle = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: -0.5, y: -0.5 },
            Vec2 { x: 2.0, y: 0.0 },
            Vec2 { x: 1.0, y: 2.0 },
        ])
        .unwrap();
        let mtv = box1.collides_with_convex(&triangle);
        assert!(mtv.is_some());
        // Converse check
        let mtv_converse = triangle.collides_with_box(&box1);
        assert!(mtv_converse.is_some());
        assert_eq!(mtv.unwrap(), -mtv_converse.unwrap());
    }

    #[test]
    fn box_3d_collision_vertical_direction() {
        // box1: spans x=[-2,2], y=[-1,1], depth=[-1,1]
        let box2d1 = BoxCollider::from_centre(Vec2::zero(), Vec2 { x: 2.0, y: 1.0 });
        let box1 = BoxCollider3d::from_2d(&box2d1, -1.0, 1.0);
        // box2: spans x=[-2,2], y=[0.5,2.5], depth=[-1,1]
        let box2d2 = BoxCollider::from_centre(Vec2 { x: 0.0, y: 1.5 }, Vec2 { x: 2.0, y: 1.0 });
        let box2 = BoxCollider3d::from_2d(&box2d2, -1.0, 1.0);
        // y overlap is 0.5, x overlap is 4.0, depth overlap is 2.0
        // y is smallest, so MTV should be along y-axis
        let result = box1.collides_with(&box2);
        assert!(result.is_some());
        let (mtv, depth_overlap) = result.unwrap();
        assert_eq!(mtv, Vec2 { x: 0.0, y: -0.5 });
        assert_eq!(depth_overlap, 0.0);
        // Converse check
        let result_converse = box2.collides_with(&box1);
        assert!(result_converse.is_some());
        let (mtv_converse, _) = result_converse.unwrap();
        assert_eq!(mtv_converse, -mtv);
    }

    #[test]
    fn box_3d_collision_depth_direction() {
        // Both boxes share same 2D collider (full 2D overlap of 10x10)
        let box2d = BoxCollider::from_centre(Vec2::zero(), Vec2 { x: 5.0, y: 5.0 });
        // box1: depth [0, 2], box2: depth [1.8, 3.8]
        // Depth overlap is [1.8, 2] = 0.2, which is smallest
        let box1 = BoxCollider3d::from_2d(&box2d, 0.0, 2.0);
        let box2 = BoxCollider3d::from_2d(&box2d, 1.8, 3.8);
        let result = box1.collides_with(&box2);
        assert!(result.is_some());
        let (mtv, depth_overlap) = result.unwrap();
        assert_eq!(mtv, Vec2::zero());
        assert!((depth_overlap - 0.2).abs() < EPSILON);
        // Converse check
        let result_converse = box2.collides_with(&box1);
        assert!(result_converse.is_some());
        let (mtv_converse, depth_converse) = result_converse.unwrap();
        assert_eq!(mtv_converse, Vec2::zero());
        assert!((depth_converse - 0.2).abs() < EPSILON);
    }

    #[test]
    fn box_3d_collision_depth_direction_opposite() {
        // Both boxes share same 2D collider (full 2D overlap of 10x10)
        let box2d = BoxCollider::from_centre(Vec2::zero(), Vec2 { x: 5.0, y: 5.0 });
        // box1: depth [0, 1], box2: depth [0.2, 5]
        // Depth overlap is [0.2, 1] = 0.8, which is smallest
        let box1 = BoxCollider3d::from_2d(&box2d, 0.0, 1.0);
        let box2 = BoxCollider3d::from_2d(&box2d, 0.2, 5.0);
        let result = box1.collides_with(&box2);
        assert!(result.is_some());
        let (mtv, depth_overlap) = result.unwrap();
        assert_eq!(mtv, Vec2::zero());
        assert!((depth_overlap - (-0.8)).abs() < EPSILON);
        // Converse check
        let result_converse = box2.collides_with(&box1);
        assert!(result_converse.is_some());
        let (mtv_converse, depth_converse) = result_converse.unwrap();
        assert_eq!(mtv_converse, Vec2::zero());
        assert!((depth_converse - 0.8).abs() < EPSILON);
    }

    #[test]
    fn convex_collider_as_any() {
        let triangle = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 2.0, y: 0.0 },
            Vec2 { x: 1.0, y: 2.0 },
        ])
        .unwrap();
        let any_ref = triangle.as_any();
        assert!(any_ref.downcast_ref::<ConvexCollider>().is_some());
    }

    #[test]
    fn convex_hull_single_vertex() {
        let result = ConvexCollider::convex_hull_of(vec![Vec2 { x: 1.0, y: 1.0 }]);
        assert!(result.is_ok());
        let collider = result.unwrap();
        assert_eq!(collider.vertices(), &[Vec2 { x: 1.0, y: 1.0 }]);
        assert_eq!(collider.centre(), Vec2 { x: 1.0, y: 1.0 });
        assert_eq!(collider.extent(), Vec2::zero());
    }

    #[test]
    fn convex_hull_with_collinear_points_removed() {
        let square = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 0.5, y: 0.0 }, // Collinear
            Vec2 { x: 1.0, y: 0.0 },
            Vec2 { x: 1.0, y: 0.5 }, // Collinear
            Vec2 { x: 1.0, y: 1.0 },
            Vec2 { x: 0.5, y: 1.0 }, // Collinear
            Vec2 { x: 0.0, y: 1.0 },
            Vec2 { x: 0.0, y: 0.5 }, // Collinear
        ])
        .unwrap();
        // All collinear points should be removed, leaving 4 corners
        assert_eq!(square.vertices().len(), 4);
        let expected_verts: HashSet<_> = [
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 1.0, y: 0.0 },
            Vec2 { x: 1.0, y: 1.0 },
            Vec2 { x: 0.0, y: 1.0 },
        ]
        .into_iter()
        .collect();
        let actual_verts: HashSet<_> = square.vertices().iter().copied().collect();
        assert_eq!(actual_verts, expected_verts);
        assert_eq!(square.centre(), Vec2::splat(0.5));
        assert_eq!(square.extent(), Vec2::splat(1.0));
    }

    // ========== CompoundCollider Tests ==========

    #[test]
    fn compound_collider_new() {
        let triangle = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 2.0, y: 0.0 },
            Vec2 { x: 1.0, y: 2.0 },
        ])
        .unwrap();
        let compound = CompoundCollider::new(vec![triangle]);
        assert_eq!(compound.len(), 1);
        assert!(!compound.is_empty());
    }

    #[test]
    fn compound_collider_decompose_convex() {
        // Simple convex polygon should return single collider
        let vertices = vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 2.0, y: 0.0 },
            Vec2 { x: 2.0, y: 2.0 },
            Vec2 { x: 0.0, y: 2.0 },
        ];
        let compound = CompoundCollider::decompose(vertices);
        assert_eq!(compound.len(), 1);
        let collider = &compound.inner_colliders()[0];
        let expected_verts: HashSet<_> = [
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 2.0, y: 0.0 },
            Vec2 { x: 2.0, y: 2.0 },
            Vec2 { x: 0.0, y: 2.0 },
        ]
        .into_iter()
        .collect();
        let actual_verts: HashSet<_> = collider.vertices().iter().copied().collect();
        assert_eq!(actual_verts, expected_verts);
        assert_eq!(collider.centre(), Vec2::splat(1.0));
        assert_eq!(collider.extent(), Vec2::splat(2.0));
    }

    #[test]
    fn compound_collider_decompose_concave() {
        // L-shaped concave polygon
        let vertices = vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 2.0, y: 0.0 },
            Vec2 { x: 2.0, y: 1.0 },
            Vec2 { x: 1.0, y: 1.0 },
            Vec2 { x: 1.0, y: 2.0 },
            Vec2 { x: 0.0, y: 2.0 },
        ];
        let compound = CompoundCollider::decompose(vertices);
        // Concave polygon should be decomposed into 2 convex pieces
        assert_eq!(compound.len(), 2);

        // Piece 0: bottom part of L
        let piece0_expected: HashSet<_> = [
            Vec2 { x: 1.0, y: 1.0 },
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 2.0, y: 0.0 },
            Vec2 { x: 2.0, y: 1.0 },
        ]
        .into_iter()
        .collect();
        let piece0_actual: HashSet<_> = compound.inner_colliders()[0]
            .vertices()
            .iter()
            .copied()
            .collect();
        assert_eq!(piece0_actual, piece0_expected);

        // Piece 1: left part of L
        let piece1_expected: HashSet<_> = [
            Vec2 { x: 1.0, y: 1.0 },
            Vec2 { x: 1.0, y: 2.0 },
            Vec2 { x: 0.0, y: 2.0 },
            Vec2 { x: 0.0, y: 0.0 },
        ]
        .into_iter()
        .collect();
        let piece1_actual: HashSet<_> = compound.inner_colliders()[1]
            .vertices()
            .iter()
            .copied()
            .collect();
        assert_eq!(piece1_actual, piece1_expected);
    }

    #[test]
    fn compound_collider_combined() {
        let triangle1_verts = vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 1.0, y: 0.0 },
            Vec2 { x: 0.5, y: 1.0 },
        ];
        let triangle2_verts = vec![
            Vec2 { x: 2.0, y: 0.0 },
            Vec2 { x: 3.0, y: 0.0 },
            Vec2 { x: 2.5, y: 1.0 },
        ];
        let triangle1 = ConvexCollider::convex_hull_of(triangle1_verts.clone()).unwrap();
        let triangle2 = ConvexCollider::convex_hull_of(triangle2_verts.clone()).unwrap();
        // Verify individual triangle extents
        assert_eq!(triangle1.extent(), Vec2::splat(1.0));
        assert_eq!(triangle2.extent(), Vec2::splat(1.0));
        let compound1 = CompoundCollider::new(vec![triangle1]);
        let compound2 = CompoundCollider::new(vec![triangle2]);
        let combined = compound1.combined(compound2);
        assert_eq!(combined.len(), 2);

        // Verify both triangles are present
        let tri1_expected: HashSet<_> = triangle1_verts.into_iter().collect();
        let tri2_expected: HashSet<_> = triangle2_verts.into_iter().collect();
        let tri1_actual: HashSet<_> = combined.inner_colliders()[0]
            .vertices()
            .iter()
            .copied()
            .collect();
        let tri2_actual: HashSet<_> = combined.inner_colliders()[1]
            .vertices()
            .iter()
            .copied()
            .collect();
        assert_eq!(tri1_actual, tri1_expected);
        assert_eq!(tri2_actual, tri2_expected);

        // Combined extent spans x=[0,3], y=[0,1]
        assert_eq!(combined.extent(), Vec2 { x: 3.0, y: 1.0 });
    }

    #[test]
    fn compound_collider_extend() {
        let triangle1_verts = vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 1.0, y: 0.0 },
            Vec2 { x: 0.5, y: 1.0 },
        ];
        let triangle2_verts = vec![
            Vec2 { x: 2.0, y: 0.0 },
            Vec2 { x: 3.0, y: 0.0 },
            Vec2 { x: 2.5, y: 1.0 },
        ];
        let triangle1 = ConvexCollider::convex_hull_of(triangle1_verts.clone()).unwrap();
        let triangle2 = ConvexCollider::convex_hull_of(triangle2_verts.clone()).unwrap();
        let mut compound1 = CompoundCollider::new(vec![triangle1]);
        let compound2 = CompoundCollider::new(vec![triangle2]);
        compound1.extend(compound2);
        assert_eq!(compound1.len(), 2);

        // Verify both triangles are present
        let tri1_expected: HashSet<_> = triangle1_verts.into_iter().collect();
        let tri2_expected: HashSet<_> = triangle2_verts.into_iter().collect();
        let tri1_actual: HashSet<_> = compound1.inner_colliders()[0]
            .vertices()
            .iter()
            .copied()
            .collect();
        let tri2_actual: HashSet<_> = compound1.inner_colliders()[1]
            .vertices()
            .iter()
            .copied()
            .collect();
        assert_eq!(tri1_actual, tri1_expected);
        assert_eq!(tri2_actual, tri2_expected);

        // Extended extent spans x=[0,3], y=[0,1]
        assert_eq!(compound1.extent(), Vec2 { x: 3.0, y: 1.0 });
    }

    #[test]
    fn compound_collider_pixel_perfect_convex() {
        // Create a 3x3 grid with a solid square in the middle
        let data = vec![
            vec![Colour::empty(), Colour::empty(), Colour::empty()],
            vec![Colour::empty(), Colour::white(), Colour::empty()],
            vec![Colour::empty(), Colour::empty(), Colour::empty()],
        ];
        let result = CompoundCollider::pixel_perfect_convex(&data);
        assert!(result.is_ok());
        let collider = result.unwrap();
        // Single pixel at (1,1) should produce a 1x1 collider
        assert_eq!(collider.extent(), Vec2::splat(1.0));
        assert_eq!(collider.vertices().len(), 4);
    }

    #[test]
    fn compound_collider_pixel_perfect() {
        // Create a 3x3 grid with a solid square
        let data = vec![
            vec![Colour::white(), Colour::white(), Colour::white()],
            vec![Colour::white(), Colour::white(), Colour::white()],
            vec![Colour::white(), Colour::white(), Colour::white()],
        ];
        let collider = CompoundCollider::pixel_perfect(&data);
        // 3x3 solid square should produce a 3x3 collider
        assert_eq!(collider.extent(), Vec2::splat(3.0));
        assert_eq!(collider.vertices().len(), 4);
    }

    // ========== Compound collider collision detection ==========

    #[test]
    fn compound_collider_collides_with_box() {
        let triangle = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 2.0, y: 0.0 },
            Vec2 { x: 1.0, y: 2.0 },
        ])
        .unwrap();
        let compound = CompoundCollider::new(vec![triangle]);
        let box_collider = BoxCollider::from_centre(Vec2 { x: 1.0, y: 0.5 }, Vec2::one());
        let generic_compound = compound.as_generic();
        let generic_box = box_collider.as_generic();
        let mtv = generic_box.collides_with(&generic_compound).unwrap();
        assert_eq!(mtv, Vec2 { x: 0.0, y: -1.5 });
        // Converse check
        let mtv_converse = generic_compound.collides_with(&generic_box).unwrap();
        assert_eq!(mtv_converse, -mtv);
    }

    #[test]
    fn compound_collider_collides_with_oriented_box() {
        let triangle = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 2.0, y: 0.0 },
            Vec2 { x: 1.0, y: 2.0 },
        ])
        .unwrap();
        let compound = CompoundCollider::new(vec![triangle]);
        let oriented = OrientedBoxCollider::from_centre(Vec2 { x: 1.0, y: 0.5 }, Vec2::one());
        let generic_compound = compound.as_generic();
        let generic_oriented = oriented.as_generic();
        let mtv = generic_oriented.collides_with(&generic_compound).unwrap();
        assert_eq!(mtv, Vec2 { x: 0.0, y: -1.5 });
        // Converse check
        let mtv_converse = generic_compound.collides_with(&generic_oriented).unwrap();
        assert_eq!(mtv_converse, -mtv);
    }

    #[test]
    fn compound_collider_collides_with_convex() {
        let triangle1 = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 2.0, y: 0.0 },
            Vec2 { x: 1.0, y: 2.0 },
        ])
        .unwrap();
        let triangle2 = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.5, y: 0.5 },
            Vec2 { x: 2.5, y: 0.5 },
            Vec2 { x: 1.5, y: 2.5 },
        ])
        .unwrap();
        let compound = CompoundCollider::new(vec![triangle1]);
        let generic_compound = compound.as_generic();
        let generic_convex = triangle2.as_generic();
        let mtv = generic_convex.collides_with(&generic_compound).unwrap();
        assert_eq!(mtv, Vec2 { x: 1.0, y: 0.5 });
        // Converse check
        let mtv_converse = generic_compound.collides_with(&generic_convex).unwrap();
        assert_eq!(mtv_converse, -mtv);
    }

    #[test]
    fn verify_triangles_collide_directly() {
        // First verify the triangles actually collide when tested directly
        let triangle1 = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 2.0, y: 0.0 },
            Vec2 { x: 1.0, y: 2.0 },
        ])
        .unwrap();
        let triangle2 = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.5, y: 0.5 },
            Vec2 { x: 1.5, y: 0.5 },
            Vec2 { x: 1.0, y: 1.5 },
        ])
        .unwrap();
        // Test direct collision
        let mtv = triangle1.collides_with_convex(&triangle2).unwrap();
        assert_eq!(mtv, Vec2 { x: 1.0, y: -0.5 });
        // Converse check
        let mtv_converse = triangle2.collides_with_convex(&triangle1).unwrap();
        assert_eq!(mtv_converse, -mtv);
    }

    #[test]
    // This test caught bugs in collides_with() - tests all collision dispatch paths work correctly.
    fn collision_all_dispatch_paths() {
        let triangle1 = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 2.0, y: 0.0 },
            Vec2 { x: 1.0, y: 2.0 },
        ])
        .unwrap();
        let triangle2 = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.5, y: 0.5 },
            Vec2 { x: 1.5, y: 0.5 },
            Vec2 { x: 1.0, y: 1.5 },
        ])
        .unwrap();

        // Step 1: Direct Convex-to-Convex collision works
        assert!(triangle1.collides_with_convex(&triangle2).is_some());
        assert!(triangle2.collides_with_convex(&triangle1).is_some());

        // Step 2: Compound collides with raw Convex (doesn't use GenericCollider)
        let compound1 = CompoundCollider::new(vec![triangle1.clone()]);
        let compound2 = CompoundCollider::new(vec![triangle2.clone()]);
        assert!(compound1.collides_with_convex(&triangle2).is_some());
        assert!(compound2.collides_with_convex(&triangle1).is_some());

        // Step 3: GenericCollider(Convex) vs GenericCollider(Compound)
        let generic1 = compound1.as_generic();
        let generic2 = compound2.as_generic();
        let generic_convex1 = triangle1.as_generic();
        assert!(generic_convex1.collides_with(&generic2).is_some());
        assert!(generic2.collides_with(&generic_convex1).is_some());
        assert!(generic1.collides_with(&generic2).is_some());
        assert!(generic2.collides_with(&generic1).is_some());
    }

    #[test]
    fn compound_collider_collides_with_compound() {
        let triangle1 = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 2.0, y: 0.0 },
            Vec2 { x: 1.0, y: 2.0 },
        ])
        .unwrap();
        let triangle2 = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.5, y: 0.5 }, // Clearly overlapping with triangle1
            Vec2 { x: 1.5, y: 0.5 },
            Vec2 { x: 1.0, y: 1.5 },
        ])
        .unwrap();
        let compound1 = CompoundCollider::new(vec![triangle1]);
        let compound2 = CompoundCollider::new(vec![triangle2]);
        let generic1 = compound1.as_generic();
        let generic2 = compound2.as_generic();
        let mtv = generic1.collides_with(&generic2).unwrap();
        assert_eq!(mtv, Vec2 { x: 1.0, y: -0.5 });
        // Converse check
        let mtv_converse = generic2.collides_with(&generic1).unwrap();
        assert_eq!(mtv_converse, -mtv);
    }

    #[test]
    fn null_collider_collides_with_compound() {
        let triangle = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 2.0, y: 0.0 },
            Vec2 { x: 1.0, y: 2.0 },
        ])
        .unwrap();
        let compound = CompoundCollider::new(vec![triangle]);
        let null = NullCollider;
        let generic_compound = compound.as_generic();
        let generic_null = null.as_generic();
        assert!(generic_null.collides_with(&generic_compound).is_none());
        // Converse check
        assert!(generic_compound.collides_with(&generic_null).is_none());
    }

    // ========== Drawing/rendering tests (for coverage) ==========

    #[test]
    fn polygonal_draw() {
        // Test draw_polygonal for coverage - we don't validate output, just call it
        use crate::util::canvas::Canvas;
        let mut canvas = Canvas::new();
        let box_collider =
            BoxCollider::from_centre(Vec2 { x: 50.0, y: 50.0 }, Vec2 { x: 10.0, y: 10.0 });
        box_collider.draw_polygonal(&mut canvas, Colour::white());
        // Just checking it doesn't crash
    }

    // ========== Edge Case Tests for 100% Coverage ==========

    #[test]
    fn polygon_normals_empty_vertices() {
        // Initialize tracing to cover the warn! path
        let _ = crate::util::setup_log();
        let normals = polygon::normals_of(vec![]);
        assert!(normals.is_empty());
    }

    #[test]
    fn polygon_centre_empty_vertices() {
        // centre_of() with empty vertex set
        let centre = polygon::centre_of(vec![]);
        assert_eq!(centre, Vec2::zero());
    }

    #[test]
    fn convex_hull_duplicate_vertices() {
        // ConvexCollider::convex_hull_of() with duplicate vertices
        let result = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 1.0, y: 0.0 },
            Vec2 { x: 1.0, y: 0.0 }, // Duplicate
            Vec2 { x: 0.5, y: 1.0 },
        ]);
        assert!(result.is_err());
    }

    #[test]
    fn convex_hull_collinear_with_same_distance() {
        // Length comparison when det == 0 (collinear points)
        let result = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 1.0, y: 1.0 },
            Vec2 { x: 2.0, y: 2.0 }, // Collinear with first two
            Vec2 { x: 0.0, y: 2.0 },
        ]);
        assert!(result.is_ok());
        let hull = result.unwrap();
        // Collinear middle point (1,1) should be filtered out, leaving 3 vertices
        assert_eq!(hull.vertices().len(), 3);
        let vertices: HashSet<_> = hull.vertices().into_iter().collect();
        let expected: HashSet<_> = vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 2.0, y: 2.0 },
            Vec2 { x: 0.0, y: 2.0 },
        ]
        .into_iter()
        .collect();
        assert_eq!(vertices, expected);
    }

    #[test]
    fn compound_collider_vertices_method() {
        // CompoundCollider::vertices()
        let triangle1 = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 1.0, y: 0.0 },
            Vec2 { x: 0.5, y: 1.0 },
        ])
        .unwrap();
        let triangle2 = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 2.0, y: 0.0 },
            Vec2 { x: 3.0, y: 0.0 },
            Vec2 { x: 2.5, y: 1.0 },
        ])
        .unwrap();
        let compound = CompoundCollider::new(vec![triangle1, triangle2]);
        let vertices = compound.vertices();
        assert_eq!(vertices.len(), 6);
        let vertices_set: HashSet<_> = vertices.into_iter().collect();
        let expected: HashSet<_> = vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 1.0, y: 0.0 },
            Vec2 { x: 0.5, y: 1.0 },
            Vec2 { x: 2.0, y: 0.0 },
            Vec2 { x: 3.0, y: 0.0 },
            Vec2 { x: 2.5, y: 1.0 },
        ]
        .into_iter()
        .collect();
        assert_eq!(vertices_set, expected);
    }

    #[test]
    fn compound_collider_normals_method() {
        // CompoundCollider::normals()
        let triangle = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 1.0, y: 0.0 },
            Vec2 { x: 0.5, y: 1.0 },
        ])
        .unwrap();
        let compound = CompoundCollider::new(vec![triangle]);
        let normals = compound.normals();
        assert_eq!(normals.len(), 3);
        let normals_set: HashSet<_> = normals.into_iter().collect();
        let expected: HashSet<_> = vec![
            Vec2 {
                x: -0.894_427_2,
                y: 0.447_213_6,
            },
            Vec2 { x: 0.0, y: -1.0 },
            Vec2 {
                x: 0.894_427_2,
                y: 0.447_213_6,
            },
        ]
        .into_iter()
        .collect();
        assert_eq!(normals_set, expected);
    }

    #[test]
    fn compound_collider_with_override_normals() {
        // override_normals return path
        let triangle = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 1.0, y: 0.0 },
            Vec2 { x: 0.5, y: 1.0 },
        ])
        .unwrap();
        let override_normal = Vec2 { x: 1.0, y: 0.0 }.normed();
        let compound = CompoundCollider {
            inner: vec![triangle],
            override_normals: vec![override_normal],
            unique_normals_cached: GgMutex::default(),
        };
        let normals = compound.normals();
        assert_eq!(normals.len(), 1);
        assert_eq!(normals[0], override_normal);
    }

    #[test]
    fn compound_collider_with_shared_edges() {
        // Edge count decrement for duplicate normals
        // Create two triangles sharing an edge along x-axis from (0,0) to (1,0)
        let triangle1 = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 1.0, y: 0.0 },
            Vec2 { x: 0.5, y: 1.0 },
        ])
        .unwrap();
        let triangle2 = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 1.0, y: 0.0 },
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 0.5, y: -1.0 },
        ])
        .unwrap();
        let compound = CompoundCollider::new(vec![triangle1, triangle2]);
        // Shared edge has opposite normals: (0,-1) from triangle1, (0,1) from triangle2
        // The (0,-1) gets filtered out as a duplicate, leaving 5 unique normals
        let normals = compound.get_unique_normals();
        assert_eq!(normals.len(), 5);
        let normals_set: HashSet<_> = normals.into_iter().collect();
        let expected: HashSet<_> = vec![
            Vec2 {
                x: -0.894_427_2,
                y: -0.447_213_6,
            },
            Vec2 {
                x: -0.894_427_2,
                y: 0.447_213_6,
            },
            Vec2 { x: 0.0, y: 1.0 },
            Vec2 {
                x: 0.894_427_2,
                y: -0.447_213_6,
            },
            Vec2 {
                x: 0.894_427_2,
                y: 0.447_213_6,
            },
        ]
        .into_iter()
        .collect();
        assert_eq!(normals_set, expected);
    }

    #[test]
    fn polygon_collision_epsilon_case() {
        // Test collision detection when boxes are separated by EPSILON/2
        // box1 at (0,0) with half-extent 1 spans x: -1 to 1
        // box2 at (2 + EPSILON/2, 0) with half-extent 1 spans x: 1 + EPSILON/2 to 3 + EPSILON/2
        // Gap between them is EPSILON/2
        let box1 = BoxCollider::from_centre(Vec2::zero(), Vec2::one());
        let box2 = BoxCollider::from_centre(
            Vec2 {
                x: 2.0 + EPSILON / 2.0,
                y: 0.0,
            },
            Vec2::one(),
        );
        let result = box1.collides_with_box(&box2);
        assert!(result.is_none());
    }

    #[test]
    fn convex_collider_single_vertex_hull() {
        // Test edge case with single vertex
        let result = ConvexCollider::convex_hull_of(vec![Vec2 { x: 0.0, y: 0.0 }]);
        assert!(result.is_ok());
        let hull = result.unwrap();
        assert_eq!(hull.vertices().len(), 1);
        assert_eq!(hull.centre(), Vec2::zero());
        assert_eq!(hull.extent(), Vec2::zero());
    }

    #[test]
    fn convex_collider_two_vertex_hull() {
        let result =
            ConvexCollider::convex_hull_of(vec![Vec2 { x: 0.0, y: 0.0 }, Vec2 { x: 1.0, y: 0.0 }]);
        assert!(result.is_ok());
        let collider = result.unwrap();
        assert_eq!(collider.vertices().len(), 2);
        assert_eq!(collider.centre(), Vec2 { x: 0.5, y: 0.0 });
        assert_eq!(collider.extent(), Vec2 { x: 1.0, y: 0.0 });
    }

    #[test]
    fn compound_collider_polygon_centre() {
        // Test CompoundCollider::polygon_centre()
        // Triangle centroid = ((0+2+1)/3, (0+0+2)/3) = (1, 2/3)
        let triangle = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 2.0, y: 0.0 },
            Vec2 { x: 1.0, y: 2.0 },
        ])
        .unwrap();
        let compound = CompoundCollider::new(vec![triangle]);
        let centre = compound.polygon_centre();
        assert_eq!(
            centre,
            Vec2 {
                x: 1.0,
                y: 2.0 / 3.0
            }
        );
    }

    #[test]
    fn compound_collider_extent() {
        // Test CompoundCollider extent calculation
        // triangle1: x: 0->1, y: 0->1
        // triangle2: x: 2->3, y: 0->2
        // Combined bounding box: x: 0->3, y: 0->2, extent = (3, 2)
        let triangle1 = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 1.0, y: 0.0 },
            Vec2 { x: 0.5, y: 1.0 },
        ])
        .unwrap();
        let triangle2 = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 2.0, y: 0.0 },
            Vec2 { x: 3.0, y: 0.0 },
            Vec2 { x: 2.5, y: 2.0 },
        ])
        .unwrap();
        let compound = CompoundCollider::new(vec![triangle1, triangle2]);
        let extent = compound.extent();
        assert_eq!(extent, Vec2 { x: 3.0, y: 2.0 });
    }

    #[test]
    fn polygon_extent_of() {
        // Test polygon::extent_of function
        let vertices = vec![
            Vec2 { x: -1.0, y: -1.0 },
            Vec2 { x: 2.0, y: -1.0 },
            Vec2 { x: 2.0, y: 3.0 },
            Vec2 { x: -1.0, y: 3.0 },
        ];
        let extent = polygon::extent_of(vertices);
        assert_eq!(extent.x, 3.0); // max_x - min_x = 2.0 - (-1.0)
        assert_eq!(extent.y, 4.0); // max_y - min_y = 3.0 - (-1.0)
    }

    #[test]
    fn polygon_is_convex_test() {
        // Test polygon::is_convex function
        let convex_vertices = vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 1.0, y: 0.0 },
            Vec2 { x: 1.0, y: 1.0 },
            Vec2 { x: 0.0, y: 1.0 },
        ];
        assert!(polygon::is_convex(&convex_vertices));

        // L-shaped polygon (concave)
        let concave_vertices = vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 2.0, y: 0.0 },
            Vec2 { x: 2.0, y: 1.0 },
            Vec2 { x: 1.0, y: 1.0 },
            Vec2 { x: 1.0, y: 2.0 },
            Vec2 { x: 0.0, y: 2.0 },
        ];
        assert!(!polygon::is_convex(&concave_vertices));
    }

    #[test]
    fn compound_collider_decompose_already_convex() {
        // Test CompoundCollider::decompose with already convex polygon
        let vertices = vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 1.0, y: 0.0 },
            Vec2 { x: 0.5, y: 1.0 },
        ];
        let compound = CompoundCollider::decompose(vertices);
        // Already convex, should have just one collider with same vertices
        assert_eq!(compound.inner.len(), 1);
        let result_vertices: HashSet<_> = compound.inner[0].vertices().into_iter().collect();
        let expected: HashSet<_> = vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 1.0, y: 0.0 },
            Vec2 { x: 0.5, y: 1.0 },
        ]
        .into_iter()
        .collect();
        assert_eq!(result_vertices, expected);
    }

    #[test]
    fn compound_collider_extend_merges_inner() {
        // Test CompoundCollider::extend method merges inner colliders
        let triangle1 = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 1.0, y: 0.0 },
            Vec2 { x: 0.5, y: 1.0 },
        ])
        .unwrap();
        let triangle2 = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 2.0, y: 0.0 },
            Vec2 { x: 3.0, y: 0.0 },
            Vec2 { x: 2.5, y: 1.0 },
        ])
        .unwrap();
        let mut compound1 = CompoundCollider::new(vec![triangle1]);
        let compound2 = CompoundCollider::new(vec![triangle2]);
        compound1.extend(compound2);
        assert_eq!(compound1.inner.len(), 2);
        // Verify all vertices from both triangles are present
        let all_vertices: HashSet<_> = compound1.vertices().into_iter().collect();
        let expected: HashSet<_> = vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 1.0, y: 0.0 },
            Vec2 { x: 0.5, y: 1.0 },
            Vec2 { x: 2.0, y: 0.0 },
            Vec2 { x: 3.0, y: 0.0 },
            Vec2 { x: 2.5, y: 1.0 },
        ]
        .into_iter()
        .collect();
        assert_eq!(all_vertices, expected);
    }

    #[test]
    fn oriented_box_collider_rotation_parameter() {
        // Test OrientedBoxCollider with non-zero rotation
        // Box with half-extent (1,1) rotated 45 deg has vertices at (0,sqrt(2)), (sqrt(2),0), (0,-sqrt(2)), (-sqrt(2),0)
        let oriented = OrientedBoxCollider::from_centre(Vec2::zero(), Vec2::one())
            .rotated(std::f32::consts::PI / 4.0);
        let vertices = oriented.vertices();
        assert_eq!(vertices.len(), 4);
        let vertices_set: HashSet<_> = vertices.into_iter().collect();
        let expected: HashSet<_> = vec![
            Vec2 { x: 0.0, y: SQRT_2 },
            Vec2 { x: SQRT_2, y: 0.0 },
            Vec2 { x: 0.0, y: -SQRT_2 },
            Vec2 { x: -SQRT_2, y: 0.0 },
        ]
        .into_iter()
        .collect();
        assert_eq!(vertices_set, expected);
    }

    #[test]
    fn box_collider_3d_no_overlap_vertical() {
        // Test BoxCollider3d with no vertical overlap
        let box2d = BoxCollider::from_centre(Vec2::zero(), Vec2::one());
        let box1 = BoxCollider3d::from_2d(&box2d, 0.0, 1.0);
        let box2 = BoxCollider3d::from_2d(&box2d, 2.0, 3.0);
        assert!(box1.collides_with(&box2).is_none());
    }

    #[test]
    fn generic_collider_type_checking() {
        // Test GenericCollider::get_type for all types
        let null = NullCollider.as_generic();
        assert_eq!(null.get_type(), ColliderType::Null);

        let box_col = BoxCollider::from_centre(Vec2::zero(), Vec2::one()).as_generic();
        assert_eq!(box_col.get_type(), ColliderType::Box);

        let oriented = OrientedBoxCollider::from_centre(Vec2::zero(), Vec2::one()).as_generic();
        assert_eq!(oriented.get_type(), ColliderType::OrientedBox);

        let triangle = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 1.0, y: 0.0 },
            Vec2 { x: 0.5, y: 1.0 },
        ])
        .unwrap()
        .as_generic();
        assert_eq!(triangle.get_type(), ColliderType::Convex);

        let compound = CompoundCollider::new(vec![]).as_generic();
        assert_eq!(compound.get_type(), ColliderType::Compound);
    }

    #[test]
    fn convex_collider_with_centre() {
        // Test ConvexCollider::with_centre transformation
        let triangle = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 1.0, y: 0.0 },
            Vec2 { x: 0.5, y: 1.0 },
        ])
        .unwrap();
        let original_extent = triangle.extent();
        let new_centre = Vec2 { x: 10.0, y: 10.0 };
        let moved = triangle.with_centre(new_centre);
        assert_eq!(moved.centre(), new_centre);
        assert_eq!(moved.extent(), original_extent);
    }

    #[test]
    fn convex_collider_with_extent() {
        // Test ConvexCollider::with_extent transformation
        let triangle = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 1.0, y: 0.0 },
            Vec2 { x: 0.5, y: 1.0 },
        ])
        .unwrap();
        let original_centre = triangle.centre();
        let new_extent = Vec2 { x: 2.0, y: 2.0 };
        let scaled = triangle.with_extent(new_extent);
        assert_eq!(scaled.extent(), new_extent);
        assert_eq!(scaled.centre(), original_centre);
    }

    // ========== CompoundCollider Additional Coverage Tests ==========

    #[test]
    fn compound_collider_debug() {
        // Test Debug trait for CompoundCollider
        let triangle = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 1.0, y: 0.0 },
            Vec2 { x: 0.5, y: 1.0 },
        ])
        .unwrap();
        let compound = CompoundCollider::new(vec![triangle]);
        let debug_str = format!("{compound:?}");
        assert!(debug_str.contains("CompoundCollider"));
    }

    #[test]
    fn compound_collider_as_any() {
        // Test CompoundCollider::as_any()
        let triangle = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 1.0, y: 0.0 },
            Vec2 { x: 0.5, y: 1.0 },
        ])
        .unwrap();
        let compound = CompoundCollider::new(vec![triangle]);
        let any_ref = compound.as_any();
        assert!(any_ref.downcast_ref::<CompoundCollider>().is_some());
    }

    #[test]
    fn compound_collider_scaled() {
        // Test CompoundCollider::scaled()
        let triangle = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 1.0, y: 0.0 },
            Vec2 { x: 0.5, y: 1.0 },
        ])
        .unwrap();
        let compound = CompoundCollider::new(vec![triangle]);
        let original_centre = compound.centre();
        let original_extent = compound.extent();
        let scaled = compound.scaled(Vec2 { x: 2.0, y: 2.0 });
        assert_eq!(scaled.centre(), original_centre);
        assert_eq!(scaled.extent(), original_extent * 2.0);
    }

    #[test]
    fn compound_collider_rotated() {
        // Test CompoundCollider::rotated()
        let triangle = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 2.0, y: 0.0 },
            Vec2 { x: 1.0, y: 2.0 },
        ])
        .unwrap();
        let compound = CompoundCollider::new(vec![triangle]);
        let original_centre = compound.centre();
        let rotated = compound.rotated(std::f32::consts::PI / 2.0);

        // Centre should be preserved after rotation
        assert_eq!(rotated.centre(), original_centre);
        // Inner collider count unchanged
        assert_eq!(rotated.inner.len(), compound.inner.len());
        // Normals should be rotated
        assert!(!rotated.inner[0].normals_cached.is_empty());
    }

    #[test]
    fn compound_collider_rotated_inner_centre_updated() {
        // Create two triangles: one on the left, one on the right
        let left_triangle = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: -2.0, y: -1.0 },
            Vec2 { x: 0.0, y: -1.0 },
            Vec2 { x: -1.0, y: 1.0 },
        ])
        .unwrap();
        let right_triangle = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: -1.0 },
            Vec2 { x: 2.0, y: -1.0 },
            Vec2 { x: 1.0, y: 1.0 },
        ])
        .unwrap();
        let compound = CompoundCollider::new(vec![left_triangle, right_triangle]);

        // Inner colliders have different centres
        let left_centre_before = compound.inner[0].centre();
        let right_centre_before = compound.inner[1].centre();
        assert!(left_centre_before.x < 0.0);
        assert!(right_centre_before.x > 0.0);

        // Rotate 90 degrees
        let rotated = compound.rotated(std::f32::consts::PI / 2.0);

        // Each inner collider's cached centre must match its actual vertices
        let left_centre_after = rotated.inner[0].centre();
        let left_expected = polygon::centre_of(rotated.inner[0].vertices.clone());
        assert_eq!(left_centre_after, left_expected);

        let right_centre_after = rotated.inner[1].centre();
        let right_expected = polygon::centre_of(rotated.inner[1].vertices.clone());
        assert_eq!(right_centre_after, right_expected);
    }

    #[test]
    fn compound_collider_rotated_inner_extent_updated() {
        // Create a wide, short triangle - extent will change when rotated 90 degrees
        let wide_triangle = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 4.0, y: 0.0 },
            Vec2 { x: 2.0, y: 1.0 },
        ])
        .unwrap();
        let original_extent = wide_triangle.extent();
        // Should be wider than tall
        assert!(original_extent.x > original_extent.y);

        let compound = CompoundCollider::new(vec![wide_triangle]);

        // Rotate 90 degrees
        let rotated = compound.rotated(std::f32::consts::PI / 2.0);

        // Inner collider's cached extent must match its actual vertices
        let inner_extent_after = rotated.inner[0].extent();
        let expected_extent = polygon::extent_of(rotated.inner[0].vertices.clone());
        assert_eq!(inner_extent_after, expected_extent);
    }

    #[test]
    fn compound_collider_transform_inverse_roundtrip() {
        // Test that applying a transform then its inverse returns to original state
        let triangle = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 2.0, y: 0.0 },
            Vec2 { x: 1.0, y: 2.0 },
        ])
        .unwrap();
        let original = CompoundCollider::new(vec![triangle]);
        let original_centre = original.centre();
        let original_vertices = original.inner[0].vertices.clone();

        // Apply a transform with both rotation and translation
        let transform = Transform {
            centre: Vec2 { x: 5.0, y: 3.0 },
            rotation: std::f32::consts::PI / 4.0,
            scale: Vec2::one(),
        };

        let transformed = original.transformed(&transform);
        let restored = transformed.transformed(&transform.inverse());

        // Should return to original state
        assert_eq!(restored.centre(), original_centre);
        for (restored_v, original_v) in restored.inner[0]
            .vertices
            .iter()
            .zip(original_vertices.iter())
        {
            assert_eq!(*restored_v, *original_v);
        }
    }

    #[test]
    fn compound_collider_update_transform_simulation() {
        // Simulate what update_transform does: apply inverse of last, then apply next
        // This tests the scenario: rotate, then translate
        let triangle = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 2.0, y: 0.0 },
            Vec2 { x: 1.0, y: 2.0 },
        ])
        .unwrap();
        let base_collider = CompoundCollider::new(vec![triangle]);

        // First transform: rotation only
        let transform1 = Transform {
            centre: Vec2::zero(),
            rotation: std::f32::consts::PI / 4.0,
            scale: Vec2::one(),
        };

        // Apply first transform (simulating initial state)
        let mut collider = base_collider.transformed(&transform1);
        let after_rotation_centre = collider.centre();

        // Second transform: same rotation + translation (user translates in GUI)
        let transform2 = Transform {
            centre: Vec2 { x: 5.0, y: 0.0 },
            rotation: std::f32::consts::PI / 4.0,
            scale: Vec2::one(),
        };

        // Simulate update_transform: undo last, apply new
        collider = collider.transformed(&transform1.inverse());
        collider = collider.transformed(&transform2);

        // The collider should now be translated by (5, 0) from its rotated position
        // The rotation should be unchanged (still PI/4)
        let expected_centre = after_rotation_centre + Vec2 { x: 5.0, y: 0.0 };
        assert_eq!(collider.centre(), expected_centre);
    }

    #[test]
    fn compound_collider_update_transform_repeated() {
        // Simulate repeated update_transform calls with small translations
        let triangle = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 2.0, y: 0.0 },
            Vec2 { x: 1.0, y: 2.0 },
        ])
        .unwrap();
        let base_collider = CompoundCollider::new(vec![triangle]);
        let original_vertices = base_collider.inner[0].vertices.clone();

        // Start with a rotation
        let mut last_transform = Transform {
            centre: Vec2::zero(),
            rotation: std::f32::consts::PI / 4.0,
            scale: Vec2::one(),
        };
        let mut collider = base_collider.transformed(&last_transform);

        // Simulate many small translations while keeping the same rotation.
        // Fails at iteration 240.
        for i in 0..239 {
            let next_transform = Transform {
                centre: Vec2 {
                    x: i as f32 / 100.0,
                    y: 0.0,
                },
                rotation: std::f32::consts::PI / 4.0,
                scale: Vec2::one(),
            };

            collider = collider.transformed(&last_transform.inverse());
            collider = collider.transformed(&next_transform);
            last_transform = next_transform;
        }

        // After all transforms, undo completely
        collider = collider.transformed(&last_transform.inverse());

        // Should be back to base collider vertices
        for (restored_v, original_v) in collider.inner[0]
            .vertices
            .iter()
            .zip(original_vertices.iter())
        {
            assert_eq!(restored_v, original_v);
        }
    }

    #[test]
    fn polygon_centre_of_precision_loss() {
        // polygon::centre_of previously returned infinity for valid triangles far from origin
        // because the shoelace formula loses precision with large coordinates.
        // The area computed from relative vectors is ~2.0, but the shoelace formula
        // previously computed area from absolute coordinates, which causes precision loss.
        let vertices = vec![
            Vec2 {
                x: 79380.15,
                y: -45760.39,
            },
            Vec2 {
                x: 7939.567,
                y: -4574.9766,
            },
            Vec2 {
                x: 7937.4653,
                y: -4574.263,
            },
        ];

        // The actual area computed using relative coordinates
        let ab = vertices[1] - vertices[0];
        let ac = vertices[2] - vertices[0];
        let actual_area = 0.5 * ab.cross(ac).abs();
        assert!(actual_area > 1.0);

        let centre = polygon::centre_of(vertices.clone());

        assert!(centre.x.is_finite() && centre.y.is_finite(),);
    }

    #[test]
    fn compound_collider_repeated_transform_centre_stays_finite() {
        // Test that repeated transform/inverse cycles don't cause precision loss
        // (Previously failed at iteration 70743 before centre_of was fixed)
        let triangle = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 2.0, y: 0.0 },
            Vec2 { x: 1.0, y: 2.0 },
        ])
        .unwrap();
        let base_collider = CompoundCollider::new(vec![triangle]);

        let mut last_transform = Transform {
            centre: Vec2::zero(),
            rotation: std::f32::consts::PI / 4.0,
            scale: Vec2::one(),
        };
        let mut collider = base_collider.transformed(&last_transform);

        for i in 0..100_000 {
            let next_transform = Transform {
                centre: Vec2 {
                    x: i as f32 * 0.1,
                    y: 0.0,
                },
                rotation: std::f32::consts::PI / 4.0,
                scale: Vec2::one(),
            };

            collider = collider.transformed(&last_transform.inverse());
            collider = collider.transformed(&next_transform);
            let centre = collider.centre();
            last_transform = next_transform;

            assert!(centre.x.is_finite() && centre.y.is_finite());
        }
    }

    #[test]
    fn compound_collider_as_polygon() {
        // Test CompoundCollider::as_polygon()
        let triangle = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 1.0, y: 0.0 },
            Vec2 { x: 0.5, y: 1.0 },
        ])
        .unwrap();
        let compound = CompoundCollider::new(vec![triangle.clone()]);
        let polygon = compound.as_polygon();
        assert_eq!(polygon.len(), 3);
        let expected: HashSet<_> = vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 1.0, y: 0.0 },
            Vec2 { x: 0.5, y: 1.0 },
        ]
        .into_iter()
        .collect();
        let actual: HashSet<_> = polygon.into_iter().collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn compound_collider_as_triangles() {
        // Test CompoundCollider::as_triangles()
        let triangle = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 1.0, y: 0.0 },
            Vec2 { x: 0.5, y: 1.0 },
        ])
        .unwrap();
        let compound = CompoundCollider::new(vec![triangle.clone()]);
        let triangles = compound.as_triangles();
        assert_eq!(triangles.len(), 1);
        let expected: HashSet<_> = vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 1.0, y: 0.0 },
            Vec2 { x: 0.5, y: 1.0 },
        ]
        .into_iter()
        .collect();
        let actual: HashSet<_> = triangles[0].into_iter().collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn compound_collider_centre_via_extent_trait() {
        // Test CompoundCollider::centre() via AxisAlignedExtent
        let triangle = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 2.0, y: 0.0 },
            Vec2 { x: 1.0, y: 2.0 },
        ])
        .unwrap();
        let compound = CompoundCollider::new(vec![triangle]);
        // Call centre() via the trait - triangle centroid is ((0+2+1)/3, (0+0+2)/3) = (1, 2/3)
        let centre: Vec2 = <CompoundCollider as AxisAlignedExtent>::centre(&compound);
        assert_eq!(
            centre,
            Vec2 {
                x: 1.0,
                y: 2.0 / 3.0
            }
        );
    }

    #[test]
    fn generic_collider_extent() {
        // Test GenericCollider::extent()
        let null = NullCollider.as_generic();
        assert_eq!(null.extent(), Vec2::zero());

        // Box with half-extent (1,1) has full extent (2,2)
        let box_col = BoxCollider::from_centre(Vec2::zero(), Vec2::one()).as_generic();
        assert_eq!(box_col.extent(), Vec2::one() * 2.0);

        // OrientedBox with half-extent (1,1) at zero rotation has extent (2,2)
        let oriented = OrientedBoxCollider::from_centre(Vec2::zero(), Vec2::one()).as_generic();
        assert_eq!(oriented.extent(), Vec2::one() * 2.0);

        // Triangle with vertices (0,0), (1,0), (0.5,1) has bounding box (0,0) to (1,1)
        let triangle = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 1.0, y: 0.0 },
            Vec2 { x: 0.5, y: 1.0 },
        ])
        .unwrap()
        .as_generic();
        assert_eq!(triangle.extent(), Vec2::one());

        // Empty compound has zero extent
        let compound = CompoundCollider::new(vec![]).as_generic();
        assert_eq!(compound.extent(), Vec2::zero());
    }

    #[test]
    fn generic_collider_centre() {
        // Test GenericCollider::centre()
        let null = NullCollider.as_generic();
        assert_eq!(null.centre(), Vec2::zero());

        let box_col = BoxCollider::from_centre(Vec2 { x: 5.0, y: 5.0 }, Vec2::one()).as_generic();
        assert_eq!(box_col.centre(), Vec2 { x: 5.0, y: 5.0 });

        let oriented =
            OrientedBoxCollider::from_centre(Vec2 { x: 3.0, y: 3.0 }, Vec2::one()).as_generic();
        assert_eq!(oriented.centre(), Vec2 { x: 3.0, y: 3.0 });

        // Triangle centroid is ((0+1+0.5)/3, (0+0+1)/3) = (0.5, 1/3)
        let triangle = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 1.0, y: 0.0 },
            Vec2 { x: 0.5, y: 1.0 },
        ])
        .unwrap()
        .as_generic();
        assert_eq!(
            triangle.centre(),
            Vec2 {
                x: 0.5,
                y: 1.0 / 3.0
            }
        );

        // Empty compound has zero centre
        let compound = CompoundCollider::new(vec![]).as_generic();
        assert_eq!(compound.centre(), Vec2::zero());
    }

    #[test]
    fn generic_collider_transformations() {
        // Test GenericCollider transformation methods
        let box_col = BoxCollider::from_centre(Vec2::zero(), Vec2::one()).as_generic();

        let translated = box_col.translated(Vec2::splat(5.0));
        assert_eq!(translated.centre(), Vec2::splat(5.0));
        assert_eq!(translated.extent(), Vec2::splat(2.0));

        // Box has half-extent (1,1) -> full extent (2,2) -> scaled by 2 -> extent (4,4)
        let scaled = box_col.scaled(Vec2::splat(2.0));
        assert_eq!(scaled.extent(), Vec2::splat(4.0));
        assert_eq!(scaled.centre(), Vec2::zero());

        // BoxCollider can only rotate by 0
        let rotated = box_col.rotated(0.0);
        assert_eq!(rotated.centre(), Vec2::zero());
        assert_eq!(rotated.extent(), Vec2::splat(2.0));
    }

    #[test]
    fn generic_collider_as_polygon() {
        // Test GenericCollider::as_polygon()
        let box_col = BoxCollider::from_centre(Vec2::zero(), Vec2::one()).as_generic();
        let polygon = box_col.as_polygon();
        assert_eq!(polygon.len(), 4);
        let expected: HashSet<_> = vec![
            Vec2 { x: -1.0, y: -1.0 },
            Vec2 { x: 1.0, y: -1.0 },
            Vec2 { x: 1.0, y: 1.0 },
            Vec2 { x: -1.0, y: 1.0 },
        ]
        .into_iter()
        .collect();
        let actual: HashSet<_> = polygon.into_iter().collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn generic_collider_as_triangles() {
        // Test GenericCollider::as_triangles()
        let box_col = BoxCollider::from_centre(Vec2::zero(), Vec2::one()).as_generic();
        let triangles = box_col.as_triangles();
        assert_eq!(triangles.len(), 2);
        // All 4 box vertices should appear across the 2 triangles
        let all_vertices: HashSet<_> = triangles.into_iter().flatten().collect();
        let expected: HashSet<_> = vec![
            Vec2 { x: -1.0, y: -1.0 },
            Vec2 { x: 1.0, y: -1.0 },
            Vec2 { x: 1.0, y: 1.0 },
            Vec2 { x: -1.0, y: 1.0 },
        ]
        .into_iter()
        .collect();
        assert_eq!(all_vertices, expected);
    }

    #[test]
    fn convex_collider_transformed() {
        // Test Collider::transformed() method
        let triangle = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 1.0, y: 0.0 },
            Vec2 { x: 0.5, y: 1.0 },
        ])
        .unwrap();
        let transform = Transform {
            centre: Vec2::splat(10.0),
            scale: Vec2::splat(2.0),
            rotation: 0.0,
        };
        let transformed = triangle.transformed(&transform);
        // Original centroid (0.5, 1/3), translated by (10,10) -> (10.5, 10+1/3)
        // Scaling around centre preserves the centre
        assert_eq!(
            transformed.centre(),
            Vec2 {
                x: 10.5,
                y: 10.0 + 1.0 / 3.0
            }
        );
        // Original extent (1, 1) scaled by 2 -> (2, 2)
        assert_eq!(transformed.extent(), Vec2::splat(2.0));
    }

    #[test]
    fn box_collider_with_half_widths() {
        // Test Collider::with_half_widths() method
        let box_col = BoxCollider::from_centre(Vec2::zero(), Vec2::one());
        let new_box = box_col.with_half_widths(Vec2::splat(2.0));
        // Half-widths of 2 means full extent of 4
        assert_eq!(new_box.extent(), Vec2::splat(4.0));
        // Centre is preserved
        assert_eq!(new_box.centre(), Vec2::zero());
    }

    #[test]
    fn convex_collider_decomposition_complex() {
        // Test CompoundCollider::decompose with concave L-shaped polygon
        let vertices = vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 2.0, y: 0.0 },
            Vec2 { x: 2.0, y: 1.0 },
            Vec2 { x: 1.0, y: 1.0 },
            Vec2 { x: 1.0, y: 2.0 },
            Vec2 { x: 0.0, y: 2.0 },
        ];
        let compound = CompoundCollider::decompose(vertices);
        // L-shape decomposes into 2 convex parts
        assert_eq!(compound.inner.len(), 2);
        // All parts must be convex
        for part in &compound.inner {
            assert!(polygon::is_convex(&part.vertices()));
        }
    }

    #[test]
    fn null_collider_methods_coverage() {
        // Test NullCollider methods for additional coverage
        let null = NullCollider;
        assert_eq!(null.as_polygon().len(), 0);
        assert_eq!(null.as_triangles().len(), 0);
    }

    // ========== GenericCollider Match Arms Coverage ==========

    #[test]
    fn generic_collider_as_any_all_types() {
        // Test GenericCollider::as_any() for all types - verify downcasting works
        let null = NullCollider.as_generic();
        assert!(null.as_any().downcast_ref::<NullCollider>().is_some());

        let box_col = BoxCollider::from_centre(Vec2::zero(), Vec2::one()).as_generic();
        assert!(box_col.as_any().downcast_ref::<BoxCollider>().is_some());

        let oriented = OrientedBoxCollider::from_centre(Vec2::zero(), Vec2::one()).as_generic();
        assert!(
            oriented
                .as_any()
                .downcast_ref::<OrientedBoxCollider>()
                .is_some()
        );

        let triangle = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 1.0, y: 0.0 },
            Vec2 { x: 0.5, y: 1.0 },
        ])
        .unwrap();
        let convex = triangle.clone().as_generic();
        assert!(convex.as_any().downcast_ref::<ConvexCollider>().is_some());

        let compound = CompoundCollider::new(vec![triangle]).as_generic();
        assert!(
            compound
                .as_any()
                .downcast_ref::<CompoundCollider>()
                .is_some()
        );
    }

    #[test]
    fn generic_collider_into_generic() {
        // Test GenericCollider::into_generic() returns self unchanged
        let generic = BoxCollider::from_centre(Vec2::splat(5.0), Vec2::one()).as_generic();
        let same = generic.clone().into_generic();
        assert_eq!(same.centre(), generic.centre());
        assert_eq!(same.extent(), generic.extent());
        assert_eq!(same.get_type(), generic.get_type());
    }

    #[test]
    fn generic_collider_translated_all_types() {
        // Test GenericCollider::translated() for all types
        let translation = Vec2::splat(5.0);

        let null = NullCollider.as_generic();
        assert_eq!(null.translated(translation).centre(), Vec2::zero());

        let oriented = OrientedBoxCollider::from_centre(Vec2::zero(), Vec2::one()).as_generic();
        assert_eq!(oriented.translated(translation).centre(), translation);

        let triangle = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 1.0, y: 0.0 },
            Vec2 { x: 0.5, y: 1.0 },
        ])
        .unwrap();
        let original_centre = triangle.centre();
        let translated = triangle.as_generic().translated(translation);
        assert_eq!(translated.centre(), original_centre + translation);

        let compound = CompoundCollider::new(vec![]).as_generic();
        assert_eq!(compound.translated(translation).centre(), Vec2::zero());
    }

    #[test]
    fn generic_collider_scaled_all_types() {
        // Test GenericCollider::scaled() for all types
        let scale = Vec2::splat(2.0);

        let null = NullCollider.as_generic();
        assert_eq!(null.scaled(scale).extent(), Vec2::zero());

        let oriented = OrientedBoxCollider::from_centre(Vec2::zero(), Vec2::one()).as_generic();
        let original_extent = oriented.extent();
        let scaled = oriented.scaled(scale);
        assert_eq!(scaled.centre(), Vec2::zero());
        assert_eq!(scaled.extent(), original_extent * 2.0);

        let triangle = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 1.0, y: 0.0 },
            Vec2 { x: 0.5, y: 1.0 },
        ])
        .unwrap();
        let original_centre = triangle.centre();
        let original_extent = triangle.extent();
        let scaled = triangle.as_generic().scaled(scale);
        assert_eq!(scaled.centre(), original_centre);
        assert_eq!(scaled.extent(), original_extent * 2.0);

        let compound = CompoundCollider::new(vec![]).as_generic();
        assert_eq!(compound.scaled(scale).extent(), Vec2::zero());
    }

    #[test]
    fn generic_collider_rotated_all_types() {
        // Test GenericCollider::rotated() for all types
        let rotation = std::f32::consts::PI / 4.0;

        let null = NullCollider.as_generic();
        assert_eq!(null.rotated(rotation).centre(), Vec2::zero());

        let oriented = OrientedBoxCollider::from_centre(Vec2::splat(3.0), Vec2::one()).as_generic();
        let rotated = oriented.rotated(rotation);
        assert_eq!(rotated.centre(), Vec2::splat(3.0));

        let triangle = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 1.0, y: 0.0 },
            Vec2 { x: 0.5, y: 1.0 },
        ])
        .unwrap();
        let original_centre = triangle.centre();
        let rotated = triangle.as_generic().rotated(rotation);
        assert_eq!(rotated.centre(), original_centre);

        // CompoundCollider::rotated() is not implemented, just clones
        let compound = CompoundCollider::new(vec![]).as_generic();
        assert_eq!(compound.rotated(rotation).centre(), Vec2::zero());
    }

    #[test]
    fn generic_collider_as_polygon_all_types() {
        // Test GenericCollider::as_polygon() for all types
        let null = NullCollider.as_generic();
        assert_eq!(null.as_polygon().len(), 0);

        let oriented = OrientedBoxCollider::from_centre(Vec2::zero(), Vec2::one()).as_generic();
        assert_eq!(oriented.as_polygon().len(), 4);

        let triangle = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 1.0, y: 0.0 },
            Vec2 { x: 0.5, y: 1.0 },
        ])
        .unwrap()
        .as_generic();
        assert_eq!(triangle.as_polygon().len(), 3);

        let compound = CompoundCollider::new(vec![]).as_generic();
        assert_eq!(compound.as_polygon().len(), 0);
    }

    #[test]
    fn generic_collider_as_triangles_all_types() {
        // Test GenericCollider::as_triangles() for all types
        let null = NullCollider.as_generic();
        assert_eq!(null.as_triangles().len(), 0);

        let oriented = OrientedBoxCollider::from_centre(Vec2::zero(), Vec2::one()).as_generic();
        assert_eq!(oriented.as_triangles().len(), 2);

        let convex_tri = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 1.0, y: 0.0 },
            Vec2 { x: 0.5, y: 1.0 },
        ])
        .unwrap();
        assert_eq!(convex_tri.clone().as_generic().as_triangles().len(), 1);

        // Empty compound has 0 triangles
        let empty_compound = CompoundCollider::new(vec![]).as_generic();
        assert_eq!(empty_compound.as_triangles().len(), 0);

        // Compound with 2 triangles has 2 triangles
        let tri2 = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 2.0, y: 0.0 },
            Vec2 { x: 3.0, y: 0.0 },
            Vec2 { x: 2.5, y: 1.0 },
        ])
        .unwrap();
        let compound = CompoundCollider::new(vec![convex_tri, tri2]).as_generic();
        assert_eq!(compound.as_triangles().len(), 2);
    }

    #[test]
    fn generic_collider_display_fmt() {
        // Test GenericCollider Display impl
        let null = NullCollider.as_generic();
        let display_str = format!("{null}");
        assert!(display_str.contains("null"));

        let box_col = BoxCollider::from_centre(Vec2::zero(), Vec2::one()).as_generic();
        let display_str = format!("{box_col}");
        assert!(display_str.contains("Box"));

        let oriented = OrientedBoxCollider::from_centre(Vec2::zero(), Vec2::one()).as_generic();
        let display_str = format!("{oriented}");
        assert!(display_str.contains("OrientedBox"));

        let triangle = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 1.0, y: 0.0 },
            Vec2 { x: 0.5, y: 1.0 },
        ])
        .unwrap()
        .as_generic();
        let display_str = format!("{triangle}");
        assert!(display_str.contains("Convex"));

        let compound = CompoundCollider::new(vec![]).as_generic();
        let display_str = format!("{compound}");
        assert!(display_str.contains("Compound"));

        // Test Compound with actual inner colliders to exercise the closure
        let tri1 = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 1.0, y: 0.0 },
            Vec2 { x: 0.5, y: 1.0 },
        ])
        .unwrap();
        let tri2 = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 2.0, y: 0.0 },
            Vec2 { x: 3.0, y: 0.0 },
            Vec2 { x: 2.5, y: 1.0 },
        ])
        .unwrap();
        let compound_with_pieces = CompoundCollider::new(vec![tri1, tri2]).as_generic();
        let display_str = format!("{compound_with_pieces}");
        assert!(display_str.contains("Compound"));
        assert!(display_str.contains("2 pieces"));
    }

    // ========== Edge Case and Error Path Tests ==========

    #[test]
    fn polygon_collision_exactly_epsilon_overlap() {
        // Overlap distance exactly at EPSILON threshold
        // Create two boxes that overlap by exactly EPSILON
        use crate::core::config::EPSILON;

        // Two boxes side by side with tiny overlap
        let box1 = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 1.0, y: 0.0 },
            Vec2 { x: 1.0, y: 1.0 },
            Vec2 { x: 0.0, y: 1.0 },
        ])
        .unwrap();

        // Second box overlaps by less than EPSILON
        let box2 = ConvexCollider::convex_hull_of(vec![
            Vec2 {
                x: 1.0 - EPSILON / 2.0,
                y: 0.0,
            },
            Vec2 { x: 2.0, y: 0.0 },
            Vec2 { x: 2.0, y: 1.0 },
            Vec2 {
                x: 1.0 - EPSILON / 2.0,
                y: 1.0,
            },
        ])
        .unwrap();

        // Should return None because overlap < EPSILON
        let result = box1.polygon_collision(&box2);
        assert!(result.is_none());
    }

    #[test]
    fn compound_collider_complex_concave_decomposition() {
        // Test decomposition of a complex concave U-shape
        let vertices = vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 3.0, y: 0.0 },
            Vec2 { x: 3.0, y: 1.0 },
            Vec2 { x: 2.0, y: 1.0 },
            Vec2 { x: 2.0, y: 2.0 },
            Vec2 { x: 1.0, y: 2.0 },
            Vec2 { x: 1.0, y: 1.0 },
            Vec2 { x: 0.0, y: 1.0 },
        ];

        let compound = CompoundCollider::decompose(vertices);
        assert_eq!(compound.inner.len(), 3);
        // All parts must be convex
        for part in &compound.inner {
            assert!(polygon::is_convex(&part.vertices()));
        }
    }

    #[test]
    fn compound_collider_spiral_concave() {
        // Test decomposition of a spiral-shaped concave polygon
        let vertices = vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 4.0, y: 0.0 },
            Vec2 { x: 4.0, y: 3.0 },
            Vec2 { x: 1.0, y: 3.0 },
            Vec2 { x: 1.0, y: 1.0 },
            Vec2 { x: 3.0, y: 1.0 },
            Vec2 { x: 3.0, y: 2.0 },
            Vec2 { x: 2.0, y: 2.0 },
            Vec2 { x: 2.0, y: 0.5 },
            Vec2 { x: 0.0, y: 0.5 },
        ];

        let compound = CompoundCollider::decompose(vertices);
        assert_eq!(compound.inner.len(), 2);
        for part in &compound.inner {
            assert!(polygon::is_convex(&part.vertices()));
        }
    }

    #[test]
    fn compound_collider_star_shape() {
        // Test star-shaped concave polygon
        use std::f32::consts::PI;
        let mut vertices = Vec::new();
        for i in 0..10 {
            let angle = (i as f32) * 2.0 * PI / 10.0;
            let radius = if i % 2 == 0 { 2.0 } else { 1.0 };
            vertices.push(Vec2 {
                x: radius * angle.cos(),
                y: radius * angle.sin(),
            });
        }

        let compound = CompoundCollider::decompose(vertices);
        assert_eq!(compound.inner.len(), 6);
        for part in &compound.inner {
            assert!(polygon::is_convex(&part.vertices()));
        }
    }

    #[test]
    fn convex_collider_collides_with_generic_convex() {
        let triangle1 = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 2.0, y: 0.0 },
            Vec2 { x: 1.0, y: 2.0 },
        ])
        .unwrap();

        let triangle2 = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.5, y: 0.5 },
            Vec2 { x: 1.5, y: 0.5 },
            Vec2 { x: 1.0, y: 1.5 },
        ])
        .unwrap();

        let generic1 = triangle1.as_generic();
        let generic2 = triangle2.as_generic();

        // Tests ColliderType::Convex branch in GenericCollider::collides_with
        let result = generic1.collides_with(&generic2);
        assert!(result.is_some());
        assert!(result.unwrap().len() > 0.0);

        // Converse check
        let result2 = generic2.collides_with(&generic1);
        assert!(result2.is_some());
        assert!(result2.unwrap().len() > 0.0);
    }

    #[test]
    fn pixel_perfect_collider_simple() {
        use crate::util::colour::Colour;

        // Create a simple 10x10 square of white pixels
        let mut data: Vec<Vec<Colour>> = Vec::new();
        for _y in 0..10 {
            let mut row = Vec::new();
            for _x in 0..10 {
                row.push(Colour::white());
            }
            data.push(row);
        }

        let compound = CompoundCollider::pixel_perfect(&data);

        // A solid rectangle is convex, so should be a single part
        assert_eq!(compound.inner.len(), 1);
        assert!(polygon::is_convex(&compound.inner[0].vertices()));

        // Extent should match pixel dimensions (10x10)
        assert_eq!(compound.extent(), Vec2::splat(10.0));
    }

    #[test]
    fn pixel_perfect_collider_l_shape() {
        use crate::util::colour::Colour;

        let mut data: Vec<Vec<Colour>> = Vec::new();

        // Create an L shape:
        // XXX
        // X
        // X
        for y in 0..5 {
            let mut row = Vec::new();
            for x in 0..5 {
                if (y == 0) || (y > 0 && x == 0) {
                    row.push(Colour::white());
                } else {
                    row.push(Colour::empty());
                }
            }
            data.push(row);
        }

        let compound = CompoundCollider::pixel_perfect(&data);
        assert_eq!(compound.inner.len(), 2);
        for part in &compound.inner {
            assert!(polygon::is_convex(&part.vertices()));
        }
    }

    #[test]
    fn compound_collider_unique_normals_with_shared_edges() {
        // Edge count decrement for shared edges
        // Create two triangles that share an edge perfectly
        let tri1 = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 1.0, y: 0.0 },
            Vec2 { x: 0.5, y: 1.0 },
        ])
        .unwrap();

        let tri2 = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 1.0, y: 0.0 },
            Vec2 { x: 0.5, y: -1.0 },
        ])
        .unwrap();

        let compound = CompoundCollider::new(vec![tri1, tri2]);

        // Get unique normals - should filter out shared edge
        let normals = compound.get_unique_normals();

        // Each triangle has 3 normals (6 total), but the shared horizontal edge
        // contributes one duplicate, leaving 5 unique normals
        assert_eq!(normals.len(), 5);
    }

    #[test]
    #[allow(clippy::similar_names)]
    fn box_collider_3d_all_axes() {
        let box2d = BoxCollider::from_centre(Vec2::zero(), Vec2::one());
        let box1 = BoxCollider3d::from_2d(&box2d, 0.0, 1.0);

        // X-axis collision: all adjusted distances equal, z-axis wins as fallback
        let box2d_x = BoxCollider::from_centre(Vec2 { x: 0.5, y: 0.0 }, Vec2::one());
        let box_x = BoxCollider3d::from_2d(&box2d_x, 0.0, 1.0);
        assert_eq!(box1.collides_with(&box_x), Some((Vec2::zero(), 1.0)));

        // X-axis separation
        let box2d_x_sep = BoxCollider::from_centre(Vec2 { x: 2.0, y: 0.0 }, Vec2::one());
        let box_x_sep = BoxCollider3d::from_2d(&box2d_x_sep, 0.0, 1.0);
        assert!(box1.collides_with(&box_x_sep).is_none());

        // Y-axis collision: all adjusted distances equal, z-axis wins as fallback
        let box2d_y = BoxCollider::from_centre(Vec2 { x: 0.0, y: 0.5 }, Vec2::one());
        let box_y = BoxCollider3d::from_2d(&box2d_y, 0.0, 1.0);
        assert_eq!(box1.collides_with(&box_y), Some((Vec2::zero(), 1.0)));

        // Y-axis separation
        let box2d_y_sep = BoxCollider::from_centre(Vec2 { x: 0.0, y: 2.0 }, Vec2::one());
        let box_y_sep = BoxCollider3d::from_2d(&box2d_y_sep, 0.0, 1.0);
        assert!(box1.collides_with(&box_y_sep).is_none());

        // Z-axis collision: z overlap (0.5) is minimum
        let box_z = BoxCollider3d::from_2d(&box2d, 0.5, 1.5);
        assert_eq!(box1.collides_with(&box_z), Some((Vec2::zero(), 0.5)));

        // Z-axis separation
        let box_z_sep = BoxCollider3d::from_2d(&box2d, 2.0, 3.0);
        assert!(box1.collides_with(&box_z_sep).is_none());
    }

    #[test]
    fn compound_collider_with_exact_opposite_edge_normals() {
        // Exact shared edge with opposite normals
        // Create two triangles sharing an edge with EXACT same vertices
        let shared_v1 = Vec2 { x: 0.0, y: 0.0 };
        let shared_v2 = Vec2 { x: 1.0, y: 0.0 };

        // Triangle 1: above shared edge
        let tri1 = ConvexCollider {
            vertices: vec![shared_v1, shared_v2, Vec2 { x: 0.5, y: 1.0 }],
            normals_cached: polygon::normals_of(vec![
                shared_v1,
                shared_v2,
                Vec2 { x: 0.5, y: 1.0 },
            ]),
            centre_cached: polygon::centre_of(vec![shared_v1, shared_v2, Vec2 { x: 0.5, y: 1.0 }]),
            extent_cached: polygon::extent_of(vec![shared_v1, shared_v2, Vec2 { x: 0.5, y: 1.0 }]),
        };

        // Triangle 2: below shared edge (reverse winding for opposite normal)
        let tri2 = ConvexCollider {
            vertices: vec![shared_v2, shared_v1, Vec2 { x: 0.5, y: -1.0 }],
            normals_cached: polygon::normals_of(vec![
                shared_v2,
                shared_v1,
                Vec2 { x: 0.5, y: -1.0 },
            ]),
            centre_cached: polygon::centre_of(vec![shared_v2, shared_v1, Vec2 { x: 0.5, y: -1.0 }]),
            extent_cached: polygon::extent_of(vec![shared_v2, shared_v1, Vec2 { x: 0.5, y: -1.0 }]),
        };

        let compound = CompoundCollider::new(vec![tri1, tri2]);
        let normals = compound.get_unique_normals();

        // Each triangle has 3 normals (6 total), shared edge filtered leaves 5
        assert_eq!(normals.len(), 5);
    }

    #[test]
    #[should_panic(expected = "discontinuity")]
    fn pixel_perfect_with_discontinuous_outline() {
        use crate::util::colour::Colour;

        // Create data that will cause a discontinuity
        // This needs to have isolated pixels that can't form a continuous outline
        let mut data: Vec<Vec<Colour>> = Vec::new();

        // Create a pattern with two separate islands that can't connect
        for y in 0..20 {
            let mut row = Vec::new();
            for x in 0..20 {
                // Two separate islands
                if ((2..=4).contains(&x) && (2..=4).contains(&y))
                    || ((10..=12).contains(&x) && (10..=12).contains(&y))
                {
                    row.push(Colour::white());
                } else {
                    row.push(Colour::empty());
                }
            }
            data.push(row);
        }

        // This should panic with discontinuity
        let _ = CompoundCollider::pixel_perfect(&data);
    }

    #[test]
    fn pixel_perfect_complex_shape() {
        use crate::resource::texture::TextureHandler;
        use crate::util::colour::Colour;

        // Load the actual mario.png sprite and verify decomposition succeeds.
        let raw = TextureHandler::load_file_inner_png("res/mario.png").unwrap();
        let w = raw.extent.width as usize;
        let h = raw.extent.height as usize;
        let mut data = vec![vec![Colour::empty(); w]; h];
        let mut x = 0;
        let mut y = 0;
        for bytes in raw.buf.chunks(4) {
            data[y][x] = Colour::from_bytes(bytes[0], bytes[1], bytes[2], bytes[3]);
            x += 1;
            if x == w {
                x = 0;
                y += 1;
            }
        }

        let collider = CompoundCollider::pixel_perfect(&data);
        assert!(!collider.inner.is_empty());
        for part in &collider.inner {
            assert!(polygon::is_convex(&part.vertices()));
        }
    }

    #[test]
    fn box_collider_vertical_collision_self_below() {
        // Tests the self.centre.y >= other.centre.y branch in vertical collision
        // half_widths (1,1) -> extent (2,2)
        let box1 = BoxCollider::from_centre(Vec2 { x: 0.0, y: 1.0 }, Vec2::one());
        let box2 = BoxCollider::from_centre(Vec2 { x: 0.0, y: 0.5 }, Vec2::one());
        assert_eq!(box1.collides_with_box(&box2), Some(Vec2 { x: 0.0, y: 1.5 }));
    }

    #[test]
    fn box_collider_3d_horizontal_self_right() {
        // Tests self.centre.x >= other.centre.x branch in 3D horizontal collision
        // Use small half_widths and large z to make horizontal the minimum axis
        let box2d1 = BoxCollider::from_centre(Vec2 { x: 0.4, y: 0.0 }, Vec2::splat(0.5));
        let box2d2 = BoxCollider::from_centre(Vec2::zero(), Vec2::splat(0.5));
        let box1 = BoxCollider3d::from_2d(&box2d1, 0.0, 10.0);
        let box2 = BoxCollider3d::from_2d(&box2d2, 0.0, 10.0);
        assert_eq!(
            box1.collides_with(&box2),
            Some((Vec2 { x: 0.6, y: 0.0 }, 0.0))
        );
    }

    #[test]
    fn generic_collider_oriented_box_dispatch_paths() {
        let oriented = OrientedBoxCollider::from_centre(Vec2::zero(), Vec2::one()).rotated(0.1);
        let generic_oriented = oriented.as_generic();

        // OrientedBox.collides_with_box
        let box_collider = BoxCollider::from_centre(Vec2::zero(), Vec2::one());
        let result = generic_oriented.collides_with_box(&box_collider);
        assert!(result.is_some());

        // OrientedBox.collides_with_oriented_box
        let other_oriented =
            OrientedBoxCollider::from_centre(Vec2::zero(), Vec2::one()).rotated(0.2);
        let result = generic_oriented.collides_with_oriented_box(&other_oriented);
        assert!(result.is_some());

        // OrientedBox.collides_with_convex
        let convex = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: -1.0, y: -1.0 },
            Vec2 { x: 1.0, y: -1.0 },
            Vec2 { x: 0.0, y: 1.0 },
        ])
        .unwrap();
        let result = generic_oriented.collides_with_convex(&convex);
        assert!(result.is_some());

        // Null.collides_with_oriented_box
        let generic_null = NullCollider.as_generic();
        assert!(generic_null.collides_with_oriented_box(&oriented).is_none());

        // Null.collides_with_convex
        assert!(generic_null.collides_with_convex(&convex).is_none());

        // Convex.collides_with_oriented_box
        let generic_convex = convex.as_generic();
        let result = generic_convex.collides_with_oriented_box(&oriented);
        assert!(result.is_some());
    }

    #[test]
    #[should_panic(expected = "failed to find a convex decomposition")]
    fn compound_collider_decompose_ray_miss() {
        // A self-intersecting (bowtie) polygon where rays from the reflex vertex
        // may not find valid edge intersections due to the invalid geometry.
        let vertices = vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 2.0, y: 2.0 }, // crosses to opposite corner
            Vec2 { x: 0.0, y: 2.0 },
            Vec2 { x: 2.0, y: 0.0 }, // crosses back
        ];
        let _compound = CompoundCollider::decompose(vertices);
    }

    #[test]
    #[should_panic(expected = "failed to find a convex decomposition")]
    fn compound_collider_decompose_fails_on_spiral() {
        // A spiral-like shape that the decomposition algorithm cannot handle.
        // The shape winds inward creating multiple reflex vertices that prevent
        // valid cuts from being found regardless of starting vertex.
        let vertices = vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 4.0, y: 0.0 },
            Vec2 { x: 4.0, y: 4.0 },
            Vec2 { x: 1.0, y: 4.0 },
            Vec2 { x: 1.0, y: 1.0 },
            Vec2 { x: 3.0, y: 1.0 },
            Vec2 { x: 3.0, y: 3.0 },
            Vec2 { x: 2.0, y: 3.0 },
            Vec2 { x: 2.0, y: 2.0 },
            Vec2 { x: 2.5, y: 2.0 },
            Vec2 { x: 2.5, y: 2.5 },
            Vec2 { x: 1.5, y: 2.5 },
            Vec2 { x: 1.5, y: 0.5 },
            Vec2 { x: 3.5, y: 0.5 },
            Vec2 { x: 3.5, y: 3.5 },
            Vec2 { x: 0.5, y: 3.5 },
            Vec2 { x: 0.5, y: 0.0 },
        ];
        let _compound = CompoundCollider::decompose(vertices);
    }

    #[test]
    fn compound_collider_rotated_with_override_normals() {
        // Test that override_normals are rotated correctly
        let triangle = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 2.0, y: 0.0 },
            Vec2 { x: 1.0, y: 2.0 },
        ])
        .unwrap();
        let compound = CompoundCollider {
            inner: vec![triangle],
            override_normals: vec![Vec2::up(), Vec2::right()],
            unique_normals_cached: GgMutex::default(),
        };
        let rotated = compound.rotated(std::f32::consts::PI / 2.0);

        // Centre should be preserved
        assert_eq!(rotated.centre(), compound.centre());
        // Override normals should be rotated 90 degrees counterclockwise
        // up (0, -1) -> right (1, 0), right (1, 0) -> down (0, 1)
        assert_eq!(rotated.override_normals.len(), 2);
        assert_eq!(rotated.override_normals[0], Vec2::right());
        assert_eq!(rotated.override_normals[1], Vec2::down());
    }

    #[test]
    #[should_panic(expected = "failed to find a convex decomposition")]
    fn compound_collider_decompose_convex_with_duplicates() {
        // A convex polygon with duplicate vertices: is_convex() returns true,
        // but convex_hull_of() fails due to duplicates, causing decompose to fail.
        let vertices = vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 1.0, y: 0.0 },
            Vec2 { x: 1.0, y: 0.0 }, // duplicate
            Vec2 { x: 1.0, y: 1.0 },
            Vec2 { x: 0.0, y: 1.0 },
        ];
        let _compound = CompoundCollider::decompose(vertices);
    }

    #[test]
    fn bincode_box_collider() {
        let config = bincode::config::standard();
        let collider = BoxCollider::from_centre(Vec2 { x: 1.0, y: 2.0 }, Vec2 { x: 3.0, y: 4.0 });
        let encoded = bincode::encode_to_vec(&collider, config).unwrap();
        let (decoded, _): (BoxCollider, _) = bincode::decode_from_slice(&encoded, config).unwrap();
        assert_eq!(decoded.centre(), collider.centre());
        assert_eq!(decoded.extent(), collider.extent());

        // Test error paths
        crate::util::test_util::test_bincode_error_paths::<BoxCollider>();
    }

    #[test]
    fn bincode_box_collider_3d() {
        let config = bincode::config::standard();
        let box_2d = BoxCollider::from_centre(Vec2 { x: 1.0, y: 2.0 }, Vec2 { x: 3.0, y: 4.0 });
        let collider = BoxCollider3d::from_2d(&box_2d, 0.5, 1.5);
        let encoded = bincode::encode_to_vec(&collider, config).unwrap();
        let (decoded, _): (BoxCollider3d, _) =
            bincode::decode_from_slice(&encoded, config).unwrap();
        assert_eq!(decoded.centre(), collider.centre());
        assert_eq!(decoded.extent(), collider.extent());

        // Test error paths
        crate::util::test_util::test_bincode_error_paths::<BoxCollider3d>();
    }

    #[test]
    fn bincode_convex_collider() {
        let config = bincode::config::standard();
        let collider = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 2.0, y: 0.0 },
            Vec2 { x: 1.0, y: 2.0 },
        ])
        .unwrap();
        let encoded = bincode::encode_to_vec(&collider, config).unwrap();
        let (decoded, _): (ConvexCollider, _) =
            bincode::decode_from_slice(&encoded, config).unwrap();
        assert_eq!(decoded.centre(), collider.centre());
        assert_eq!(decoded.extent(), collider.extent());
        assert_eq!(decoded.vertices, collider.vertices);

        // Test error paths using the non-default collider
        let cfg = bincode::config::legacy();
        let encoded_legacy = bincode::encode_to_vec(&collider, cfg).unwrap();

        // Test encode error with too-small buffer
        let mut small_buf = vec![0u8; encoded_legacy.len() - 1];
        assert!(bincode::encode_into_slice(&collider, &mut small_buf, cfg).is_err());

        // Test decode errors with truncated data
        let truncated = &encoded_legacy[..encoded_legacy.len() - 1];
        assert!(bincode::decode_from_slice::<ConvexCollider, _>(truncated, cfg).is_err());
        assert!(bincode::borrow_decode_from_slice::<ConvexCollider, _>(truncated, cfg).is_err());
    }

    #[test]
    fn bincode_oriented_box_collider() {
        let config = bincode::config::standard();
        let collider =
            OrientedBoxCollider::from_centre(Vec2 { x: 1.0, y: 2.0 }, Vec2 { x: 3.0, y: 4.0 })
                .rotated(0.5);
        let encoded = bincode::encode_to_vec(&collider, config).unwrap();
        let (decoded, _): (OrientedBoxCollider, _) =
            bincode::decode_from_slice(&encoded, config).unwrap();
        assert_eq!(decoded.centre(), collider.centre());
        assert_eq!(decoded.rotation, collider.rotation);

        // Test error paths
        crate::util::test_util::test_bincode_error_paths::<OrientedBoxCollider>();
    }

    #[test]
    fn generic_collider_transformed_box_no_rotation() {
        // Test transformed() on BoxCollider with no rotation (optimized path)
        let box_col = BoxCollider::from_centre(Vec2 { x: 1.0, y: 2.0 }, Vec2::one()).as_generic();
        let transform = Transform {
            centre: Vec2 { x: 5.0, y: 10.0 },
            scale: Vec2 { x: 2.0, y: 3.0 },
            rotation: 0.0,
        };
        let result = box_col.transformed(&transform);
        assert_eq!(result.get_type(), ColliderType::Box);
        // Centre (1,2) translated by (5,10) = (6,12). Scale does not affect centre.
        assert_eq!(result.centre(), Vec2 { x: 6.0, y: 12.0 });
        // Extent (2,2) scaled by (2,3) = (4,6)
        assert_eq!(result.extent(), Vec2 { x: 4.0, y: 6.0 });
    }

    #[test]
    fn generic_collider_transformed_box_with_rotation() {
        // Test transformed() on BoxCollider with rotation (general path)
        let box_col = BoxCollider::from_centre(Vec2::zero(), Vec2::one()).as_generic();
        let transform = Transform {
            centre: Vec2::zero(),
            scale: Vec2::one(),
            rotation: std::f32::consts::PI / 4.0,
        };
        let result = box_col.transformed(&transform);
        assert_eq!(result.get_type(), ColliderType::OrientedBox);
        // After rotation around origin, centre stays at origin
        assert_eq!(result.centre(), Vec2::zero());

        // Also test with scale and translation
        let box_col2 = BoxCollider::from_centre(Vec2 { x: 1.0, y: 2.0 }, Vec2::one()).as_generic();
        let transform2 = Transform {
            centre: Vec2 { x: 5.0, y: 10.0 },
            scale: Vec2 { x: 2.0, y: 3.0 },
            rotation: std::f32::consts::PI / 2.0,
        };
        let result2 = box_col2.transformed(&transform2);
        assert_eq!(result2.get_type(), ColliderType::OrientedBox);

        // Same test but starting with OrientedBoxCollider to check consistency
        let oriented_col =
            OrientedBoxCollider::from_centre(Vec2 { x: 1.0, y: 2.0 }, Vec2::one()).as_generic();
        let result3 = oriented_col.transformed(&transform2);
        assert_eq!(result3.get_type(), ColliderType::OrientedBox);

        // Both paths should produce the same result
        assert_eq!(result2.centre(), result3.centre());
        assert_eq!(result2.extent(), result3.extent());
        // Centre (1,2) + translation (5,10) = (6,12)
        assert_eq!(result2.centre(), Vec2 { x: 6.0, y: 12.0 });
        // Extent (2,2) scaled by (2,3) = (4,6), then rotated 90° = (6,4)
        assert_eq!(result2.extent(), Vec2 { x: 6.0, y: 4.0 });
    }

    #[test]
    fn generic_collider_transformed_non_box() {
        // Test transformed() on non-BoxCollider (always uses general path)
        let triangle = ConvexCollider::convex_hull_of(vec![
            Vec2 { x: -0.5, y: -0.5 },
            Vec2 { x: 0.5, y: -0.5 },
            Vec2 { x: 0.0, y: 0.5 },
        ])
        .unwrap()
        .as_generic();
        let transform = Transform {
            centre: Vec2 { x: 5.0, y: 10.0 },
            scale: Vec2 { x: 2.0, y: 3.0 },
            rotation: std::f32::consts::PI / 2.0,
        };
        let result = triangle.transformed(&transform);
        assert_eq!(result.get_type(), ColliderType::Convex);
        // Triangle centre is near origin, so scale/rotate have minimal effect
        assert_eq!(
            result.centre(),
            Vec2 {
                x: 5.0,
                y: 9.833_333
            }
        );
        assert_eq!(result.extent(), Vec2 { x: 2.0, y: 3.0 });
    }

    #[test]
    fn generic_collider_transformed_oriented_box_canceling_rotation() {
        // Test transformed() on OrientedBox where rotation cancels out
        let rotation = std::f32::consts::PI / 4.0;
        let oriented = OrientedBoxCollider::from_centre(Vec2 { x: 1.0, y: 2.0 }, Vec2::one())
            .rotated(rotation)
            .as_generic();
        let transform = Transform {
            centre: Vec2 { x: 5.0, y: 10.0 },
            scale: Vec2 { x: 2.0, y: 3.0 },
            rotation: -rotation, // Cancels out the OrientedBox rotation
        };
        let result = oriented.transformed(&transform);
        // Should become a regular BoxCollider (no rotation)
        assert_eq!(result.get_type(), ColliderType::Box);
        // Centre (1,2) translated by (5,10) = (6,12)
        assert_eq!(result.centre(), Vec2 { x: 6.0, y: 12.0 });
        // Original extent is 2*sqrt(2) (rotated unit box), scaled by (2,3)
        assert_eq!(
            result.extent(),
            Vec2 {
                x: 2.0 * 2.0 * SQRT_2,
                y: 3.0 * 2.0 * SQRT_2
            }
        );
    }
}
