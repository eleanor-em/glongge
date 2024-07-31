use std::{
    marker::PhantomData,
    any::Any,
    fmt::Debug,
    ops::Range,
    sync::Arc
};
use num_traits::{Float, Zero};
use glongge_derive::{partially_derive_scene_object, register_scene_object};
use crate::{
    core::{
        prelude::*,
        util::{
            gg_range,
            gg_iter,
            linalg::{
                Vec2,
                AxisAlignedExtent,
                Transform
            }
        },
        ObjectTypeEnum,
        scene::SceneObject
    },
    resource::sprite::Sprite
};

pub enum ColliderType {
    Null,
    Box,
    OrientedBox,
    Convex,
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
            ColliderType::Box => self.collides_with_box(other.as_any().downcast_ref().unwrap()),
            ColliderType::OrientedBox => self.collides_with_oriented_box(other.as_any().downcast_ref().unwrap()),
            ColliderType::Convex => self.collides_with_convex(other.as_any().downcast_ref().unwrap())
        }
    }

    fn as_generic(&self) -> GenericCollider where Self: Clone + Send + Sync + 'static {
        GenericCollider::new(self.clone())
    }
    fn into_generic(self) -> GenericCollider where Self: Sized + Send + Sync + 'static {
        GenericCollider::new(self)
    }

    fn translated(&self, by: Vec2) -> GenericCollider;
    // TODO: scaled/rotated/transformed

    fn as_polygon(&self) -> Vec<Vec2>;
}

#[derive(Debug, Clone, Copy)]
pub struct NullCollider;
impl AxisAlignedExtent for NullCollider {
    fn aa_extent(&self) -> Vec2 { Vec2::zero() }

    fn centre(&self) -> Vec2 { Vec2::zero() }
}
impl Collider for NullCollider {
    fn as_any(&self) -> &dyn Any { self }
    fn get_type(&self) -> ColliderType { ColliderType::Null }

    fn collides_with_box(&self, _other: &BoxCollider) -> Option<Vec2> { None }
    fn collides_with_oriented_box(&self, _other: &OrientedBoxCollider) -> Option<Vec2> { None }
    fn collides_with_convex(&self, _other: &ConvexCollider) -> Option<Vec2> { None }

    fn translated(&self, _by: Vec2) -> GenericCollider { Self.as_generic() }

    // By convention, clockwise edges starting from the top-leftmost vertex.
    fn as_polygon(&self) -> Vec<Vec2> {
        Vec::new()
    }
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

    fn adjust_for_containment(self_proj: &Range<f64>, other_proj: &Range<f64>) -> f64 {
        if gg_range::contains_f64(self_proj, other_proj) ||
                gg_range::contains_f64(other_proj, self_proj) {
            let starts = (self_proj.start - other_proj.start).abs();
            let ends = (self_proj.end - other_proj.end).abs();
            f64::min(starts, ends)
        } else {
            0.
        }
    }
    fn polygon_collision<P: Polygonal>(&self, other: P) -> Option<Vec2> {
        let mut min_axis = Vec2::zero();
        let mut min_dist = f64::max_value();

        for axis in self.normals().into_iter().chain(other.normals()) {
            let self_proj = self.project(axis);
            let other_proj = other.project(axis);
            match gg_range::overlap_len_f64(&self_proj, &other_proj) {
                None | Some(0.) => return None,
                Some(mut dist) => {
                    dist += Self::adjust_for_containment(&self_proj, &other_proj);
                    if dist < min_dist {
                        min_dist = dist;
                        min_axis = axis;
                    }
                },
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
    fn extent_of(vertices: Vec<Vec2>) -> Vec2 {
        let mut min_x = f64::MAX;
        let mut min_y = f64::MAX;
        let mut max_x = f64::MIN;
        let mut max_y = f64::MIN;
        for vertex in vertices {
            min_x = vertex.x.min(min_x);
            min_y = vertex.y.min(min_y);
            max_x = vertex.x.max(max_x);
            max_y = vertex.y.max(max_y);
        }
        Vec2 { x: max_x - min_x, y: max_y - min_y }
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
    rotation: f64,
    axis_aligned_half_widths: Vec2,
    extent: Vec2,
}
impl OrientedBoxCollider {
    pub fn from_centre(centre: Vec2, half_widths: Vec2) -> Self {
        let mut rv = Self {
            centre,
            rotation: 0.,
            axis_aligned_half_widths: half_widths,
            extent: Vec2::zero()
        };
        rv.extent = Self::extent_of(rv.vertices());
        rv
    }
    pub fn from_top_left(top_left: Vec2, extent: Vec2) -> Self {
        let mut rv = Self {
            centre: top_left + extent.abs() / 2,
            rotation: 0.,
            axis_aligned_half_widths: extent.abs() / 2,
            extent: Vec2::zero()
        };
        rv.extent = Self::extent_of(rv.vertices());
        rv
    }
    pub fn from_transform(transform: Transform, half_widths: Vec2) -> Self {
        let mut rv = Self {
            centre: transform.centre,
            rotation: transform.rotation,
            axis_aligned_half_widths: transform.scale.component_wise(half_widths).abs(),
            extent: Vec2::zero()
        };
        rv.extent = Self::extent_of(rv.vertices());
        rv
    }
    pub fn square(transform: Transform, width: f64) -> Self {
        Self::from_transform(transform, width.abs() * Vec2::one())
    }

    pub fn top_left_rotated(&self) -> Vec2 {
        self.centre + (-self.axis_aligned_half_widths).rotated(self.rotation)
    }
    pub fn top_right_rotated(&self) -> Vec2 {
        self.centre + Vec2 { x: self.axis_aligned_half_widths.x, y: -self.axis_aligned_half_widths.y }.rotated(self.rotation)
    }
    pub fn bottom_left_rotated(&self) -> Vec2 {
        self.centre + Vec2 { x: -self.axis_aligned_half_widths.x, y: self.axis_aligned_half_widths.y }.rotated(self.rotation)
    }
    pub fn bottom_right_rotated(&self) -> Vec2 {
        self.centre + self.axis_aligned_half_widths.rotated(self.rotation)
    }
}
impl Polygonal for OrientedBoxCollider {
    fn vertices(&self) -> Vec<Vec2> {
        vec![
            self.top_left_rotated(), self.top_right_rotated(), self.bottom_right_rotated(), self.bottom_left_rotated()
        ]
    }
    fn normals(&self) -> Vec<Vec2> {
        vec![Vec2::right().rotated(self.rotation), Vec2::down().rotated(self.rotation)]
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
    fn as_any(&self) -> &dyn Any { self }
    fn get_type(&self) -> ColliderType { ColliderType::OrientedBox }

    fn collides_with_box(&self, other: &BoxCollider) -> Option<Vec2> {
        self.polygon_collision(other)
    }

    fn collides_with_oriented_box(&self, other: &OrientedBoxCollider) -> Option<Vec2> {
        self.polygon_collision(other)
    }

    fn collides_with_convex(&self, other: &ConvexCollider) -> Option<Vec2> {
        self.polygon_collision(other)
    }

    fn translated(&self, by: Vec2) -> GenericCollider {
        let mut rv = self.clone();
        rv.centre += by.rotated(self.rotation);
        rv.as_generic()
    }

    fn as_polygon(&self) -> Vec<Vec2> {
        self.vertices()
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
    pub fn square(transform: Transform, width: f64) -> Self {
        Self::from_transform(transform, width.abs() * Vec2::one())
    }
    #[must_use]
    pub fn transformed(&self, by: Transform) -> Self {
        Self {
            centre: self.centre + by.centre - self.half_widths(),
            extent: self.extent.component_wise(by.scale.abs()),
        }
    }
}

impl Polygonal for BoxCollider {
    fn vertices(&self) -> Vec<Vec2> {
        vec![
            self.top_left(), self.top_right(), self.bottom_right(), self.bottom_left()
        ]
    }
    fn normals(&self) -> Vec<Vec2> {
        vec![Vec2::right(), Vec2::down()]
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
    fn as_any(&self) -> &dyn Any { self }
    fn get_type(&self) -> ColliderType { ColliderType::Box }

    fn collides_with_box(&self, other: &BoxCollider) -> Option<Vec2> {
        let self_proj = self.left()..self.right();
        let other_proj = other.left()..other.right();
        let right_dist = match gg_range::overlap_len_f64(&self_proj, &other_proj) {
            None | Some(0.) => return None,
            Some(dist) => {
                dist + Self::adjust_for_containment(&self_proj, &other_proj)
            },
        };

        let self_proj = self.top()..self.bottom();
        let other_proj = other.top()..other.bottom();
        match gg_range::overlap_len_f64(&self_proj, &other_proj) {
            None | Some(0.) => None,
            Some(mut dist) => {
                dist += Self::adjust_for_containment(&self_proj, &other_proj);
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
            },
        }
    }

    fn collides_with_oriented_box(&self, other: &OrientedBoxCollider) -> Option<Vec2> {
        self.polygon_collision(other)
    }

    fn collides_with_convex(&self, other: &ConvexCollider) -> Option<Vec2> {
        self.polygon_collision(other)
    }

    fn translated(&self, by: Vec2) -> GenericCollider {
        let mut rv = self.clone();
        rv.centre += by;
        rv.into_generic()
    }

    fn as_polygon(&self) -> Vec<Vec2> {
        self.vertices()
    }
}

#[derive(Debug, Clone)]
pub struct ConvexCollider {
    vertices: Vec<Vec2>,
    normals: Vec<Vec2>,
    centre: Vec2,
    extent: Vec2,
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
        let extent = Self::extent_of(vertices.clone());
        Self { vertices, normals, centre, extent }
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

impl AxisAlignedExtent for ConvexCollider {
    fn aa_extent(&self) -> Vec2 { self.extent }

    fn centre(&self) -> Vec2 { self.centre }
}

impl Collider for ConvexCollider {
    fn as_any(&self) -> &dyn Any { self }

    fn get_type(&self) -> ColliderType { ColliderType::Convex }

    fn collides_with_box(&self, other: &BoxCollider) -> Option<Vec2> {
        self.polygon_collision(other)
    }

    fn collides_with_oriented_box(&self, other: &OrientedBoxCollider) -> Option<Vec2> {
        self.polygon_collision(other)
    }

    fn collides_with_convex(&self, other: &ConvexCollider) -> Option<Vec2> {
        self.polygon_collision(other)
    }

    fn translated(&self, by: Vec2) -> GenericCollider {
        let mut rv = self.clone();
        for vertex in &mut rv.vertices {
            *vertex += by;
        }
        rv.into_generic()
    }

    fn as_polygon(&self) -> Vec<Vec2> {
        // TODO: check that this conforms to the spec
        self.vertices.clone()
    }
}

#[derive(Clone, Debug)]
pub struct GenericCollider {
    inner: Arc<dyn Collider>,
}

impl GenericCollider {
    pub fn new<C: Collider>(inner: C) -> Self {
        let extent = inner.aa_extent();
        check_eq!(extent, extent.abs());
        Self { inner: Arc::new(inner) as Arc<dyn Collider> }
    }
}

impl Default for GenericCollider {
    fn default() -> Self {
        Self { inner: Arc::new(NullCollider) }
    }
}

impl AxisAlignedExtent for GenericCollider {
    fn aa_extent(&self) -> Vec2 {
        self.inner.aa_extent()
    }

    fn centre(&self) -> Vec2 {
        self.inner.centre()
    }
}

impl Collider for GenericCollider {
    fn as_any(&self) -> &dyn Any {
        self.inner.as_any()
    }

    fn get_type(&self) -> ColliderType {
        self.inner.get_type()
    }

    fn collides_with_box(&self, other: &BoxCollider) -> Option<Vec2> {
        self.inner.collides_with_box(other)
    }

    fn collides_with_oriented_box(&self, other: &OrientedBoxCollider) -> Option<Vec2> {
        self.inner.collides_with_oriented_box(other)
    }

    fn collides_with_convex(&self, other: &ConvexCollider) -> Option<Vec2> {
        self.inner.collides_with_convex(other)
    }

    fn translated(&self, by: Vec2) -> GenericCollider {
        self.inner.translated(by).into_generic()
    }

    fn as_generic(&self) -> GenericCollider
    where
        Self: Clone + Send + Sync + 'static
    {
        self.clone()
    }
    fn into_generic(self) -> GenericCollider
    where
        Self: Sized + Send + Sync + 'static,
    {
        self
    }

    fn as_polygon(&self) -> Vec<Vec2> {
        self.inner.as_polygon()
    }
}

#[register_scene_object]
pub struct GgInternalCollisionShape<ObjectType> {
    collider: GenericCollider,
    emitting_tags: Vec<&'static str>,
    listening_tags: Vec<&'static str>,
    object_type: PhantomData<ObjectType>,
}

impl<ObjectType: ObjectTypeEnum> GgInternalCollisionShape<ObjectType> {
    pub fn new<C: Collider>(
        collider: C,
        emitting_tags: &[&'static str],
        listening_tags: &[&'static str]
    ) -> Box<Self> {
        Box::new(Self {
            collider: collider.into_generic(),
            emitting_tags: emitting_tags.to_vec(),
            listening_tags: listening_tags.to_vec(),
            object_type: PhantomData,
        })
    }

    pub fn from_object<O: SceneObject<ObjectType>, C: Collider>(
        object: &O,
        collider: C,
    ) -> Box<Self> { Self::new(collider, &object.emitting_tags(), &object.listening_tags()) }
    pub fn from_object_sprite<O: SceneObject<ObjectType>>(
        object: &O,
        sprite: &Sprite<ObjectType>
    ) -> Box<Self> { Self::new(sprite.as_box_collider(), &object.emitting_tags(), &object.listening_tags()) }
}

#[partially_derive_scene_object]
impl<ObjectType: ObjectTypeEnum> SceneObject<ObjectType> for GgInternalCollisionShape<ObjectType> {
    fn on_ready(&mut self, ctx: &mut UpdateContext<ObjectType>) {
        check!(ctx.object().parent().is_some(), "CollisionShapes must have a parent");
    }
    fn get_type(&self) -> ObjectType { ObjectType::gg_collider() }

    fn collider(&self) -> GenericCollider { self.collider.clone() }
    fn emitting_tags(&self) -> Vec<&'static str> {
        self.emitting_tags.clone()
    }
    fn listening_tags(&self) -> Vec<&'static str> {
        self.listening_tags.clone()
    }
}

pub use GgInternalCollisionShape as CollisionShape;
