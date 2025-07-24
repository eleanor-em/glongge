#[allow(unused_imports)]
use crate::core::prelude::*;

use crate::util::gg_float;
use crate::util::gg_float::GgFloat;
use itertools::Product;
use num_traits::{One, Zero};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::iter::Sum;
use std::{
    fmt,
    fmt::Formatter,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Range, Sub, SubAssign},
};

/// A 2D vector representation using 32-bit floating point coordinates.
///
/// [`Vec2`] provides a comprehensive set of operations for 2D vector mathematics,
/// including common vector operations like addition, subtraction, scaling,
/// normalisation, dot and cross products, and various geometric utilities.
///
/// # Examples
///
/// ```
/// use glongge::util::linalg::Vec2;
///
/// // Create vectors
/// let v1 = Vec2 { x: 3.0, y: 4.0 };
/// let v2 = Vec2 { x: 1.0, y: 2.0 };
///
/// // Vector operations
/// let sum = v1 + v2;
/// ```
///
/// # Equality and ordering
/// [`Vec2`] provides [`Eq`] and [`Ord`] implementations that enable total ordering of 2D vectors.
///
/// ## Equality
/// Two vectors are considered equal if their components differ by less than
/// [`EPSILON`](crate::core::config::EPSILON). This handles floating point imprecision while
/// still ensuring reflexivity and transitivity.
///
/// ## Ordering
/// Since floating point values don't have a natural total ordering due to `NaN` values,
/// this implementation creates a deterministic ordering by:
///
/// 1. First comparing the vectors for equality using [`PartialEq`]
/// 2. If different, comparing the `x` coordinates if they differ by more than
///    [`EPSILON`](crate::core::config::EPSILON)
/// 3. Otherwise, comparing the `y` coordinates
///
/// When comparing floating point components, it first attempts to use
/// [`partial_cmp`](f32::partial_cmp), and falls back to [`total_cmp`](f32::total_cmp) if needed
/// (handles `NaN` values).
///
/// This implementation ensures:
/// - Consistent ordering even with edge cases like `NaN` or infinite values
/// - Stable sorting in collections like [`BTreeMap`](std::collections::BTreeMap)
///   /[`BTreeSet`](std::collections::BTreeSet)
/// - Compatibility with comparison-based algorithms
///
/// Note: This ordering doesn't have a particular geometric meaning, it just provides a stable,
/// deterministic ordering of vectors.
#[derive(Default, Debug, Copy, Clone)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl PartialEq for Vec2 {
    fn eq(&self, other: &Self) -> bool {
        if self.is_finite() || other.is_finite() {
            (self.x - other.x).abs() < EPSILON && (self.y - other.y).abs() < EPSILON
        } else {
            self.x == other.x && self.y == other.y
        }
    }
}
impl Eq for Vec2 {}

impl PartialOrd<Self> for Vec2 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Vec2 {
    fn cmp(&self, other: &Self) -> Ordering {
        if self == other {
            return Ordering::Equal;
        }
        if (self.x - other.x).abs() < EPSILON {
            return self.y.partial_cmp(&other.y).unwrap_or_else(|| {
                warn!("Vec2: partial_cmp() failed for y: {} vs. {}", self, other);
                self.y.total_cmp(&other.y)
            });
        }
        if let Some(o) = self.x.partial_cmp(&other.x) {
            o
        } else {
            warn!("Vec2: partial_cmp() failed for x: {} vs. {}", self, other);
            match self.x.total_cmp(&other.x) {
                Ordering::Equal => {
                    if let Some(o) = self.y.partial_cmp(&other.y) {
                        o
                    } else {
                        warn!("Vec2: partial_cmp() failed for x: {} vs. {}", self, other);
                        self.y.total_cmp(&other.y)
                    }
                }
                o => o,
            }
        }
    }
}

impl Hash for Vec2 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.x.to_bits().hash(state);
        self.y.to_bits().hash(state);
    }
}

// TODO: make const functions
impl Vec2 {
    /// Returns a unit vector pointing to the right (positive x-axis).
    #[must_use]
    pub fn right() -> Vec2 {
        Vec2 { x: 1.0, y: 0.0 }
    }
    /// Returns a unit vector pointing upward (negative y-axis).
    ///
    /// Note: This follows a coordinate system where y increases downward,
    /// which is common in 2D graphics applications.
    #[must_use]
    pub fn up() -> Vec2 {
        Vec2 { x: 0.0, y: -1.0 }
    }
    /// Returns a unit vector pointing to the left (negative x-axis).
    #[must_use]
    pub fn left() -> Vec2 {
        Vec2 { x: -1.0, y: 0.0 }
    }
    /// Returns a unit vector pointing downward (positive y-axis).
    ///
    /// Note: This follows a coordinate system where y increases downward,
    /// which is common in 2D graphics applications.
    #[must_use]
    pub fn down() -> Vec2 {
        Vec2 { x: 0.0, y: 1.0 }
    }
    /// Returns a vector with both components set to 1.0.
    ///
    /// This can be useful for scaling operations or as a basis
    /// for component-wise operations.
    #[must_use]
    pub fn one() -> Vec2 {
        Vec2 { x: 1.0, y: 1.0 }
    }
    /// Returns a vector with both components set to 0.0.
    ///
    /// This can be useful for initialisation or as a neutral element
    /// for addition operations.
    #[must_use]
    pub fn zero() -> Vec2 {
        Vec2 { x: 0.0, y: 0.0 }
    }

    /// Creates a new vector with both components set to the given value.
    ///
    /// This is useful for operations where you want uniform scaling or offsets
    /// in both dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// use glongge::core::prelude::*;
    /// let vec = Vec2::splat(3.0);
    /// assert_eq!(vec.x, 3.0);
    /// assert_eq!(vec.y, 3.0);
    /// ```
    #[must_use]
    pub fn splat(v: f32) -> Vec2 {
        Vec2 { x: v, y: v }
    }

    /// Returns the squared length of the vector.
    ///
    /// Use this instead of [`len`] when comparing lengths to avoid the computationally expensive
    /// square root operation.
    #[must_use]
    pub fn len_squared(&self) -> f32 {
        self.dot(*self)
    }

    /// Returns the length of the vector.
    ///
    /// If you only need to compare vector lengths, consider using [`len_squared`] to avoid the
    /// computationally expensive square root operation.
    #[must_use]
    pub fn len(&self) -> f32 {
        self.len_squared().sqrt()
    }

    /// Returns a normalised (unit) vector in the same direction as this vector.
    ///
    /// If the original vector's length is zero, returns a zero vector to avoid
    /// division by zero. Also handles conversion of negative zero (-0.0) to
    /// positive zero (0.0) for both x and y components.
    ///
    /// This function is often used when only the direction of a vector matters,
    /// such as when calculating angles between vectors or projecting vectors.
    #[must_use]
    pub fn normed(&self) -> Vec2 {
        let mut rv = match self.len() {
            0.0 => Vec2::zero(),
            len => *self / len,
        };
        rv.x = gg_float::force_positive_zero(rv.x);
        rv.y = gg_float::force_positive_zero(rv.y);
        rv
    }

    /// Returns the magnitude of the vector's largest component.
    ///
    /// # Examples
    ///
    /// ```
    /// use glongge::core::prelude::*;
    /// let vec = Vec2 { x: -3.0, y: 2.0 };
    /// assert_eq!(vec.longest_component(), 3.0);
    /// ```
    #[must_use]
    pub fn longest_component(&self) -> f32 {
        self.x.abs().max(self.y.abs())
    }

    /// Returns a new vector with the absolute values of each component. That is, it always lies
    /// in the first quadrant.
    ///
    /// # Examples
    ///
    /// ```
    /// use glongge::core::prelude::*;
    /// let vec = Vec2 { x: -3.0, y: -2.0 };
    /// let abs_vec = vec.abs();
    /// assert_eq!(abs_vec.x, 3.0);
    /// assert_eq!(abs_vec.y, 2.0);
    /// ```
    #[must_use]
    pub fn abs(&self) -> Vec2 {
        Vec2 {
            x: self.x.abs(),
            y: self.y.abs(),
        }
    }

    /// Returns a new vector rotated clockwise by the given angle in radians.
    ///
    /// # Examples
    ///
    /// ```
    /// use glongge::core::prelude::*;
    /// let vec = Vec2::right();
    /// let rotated = vec.rotated(std::f32::consts::PI / 2.0); // 90 degrees
    /// assert!(rotated.almost_eq(Vec2::down()));
    /// ```
    #[must_use]
    pub fn rotated(&self, radians: f32) -> Vec2 {
        Mat3x3::rotation(radians) * *self
    }

    /// Reflects the vector about a normal vector.
    ///
    /// # Parameters
    ///
    /// * `normal` - The normal vector to reflect about. Must be already normalised.
    ///
    /// # Examples
    ///
    /// ```
    /// use glongge::core::prelude::*;
    /// let vec = Vec2 { x: 1.0, y: 1.0 };
    /// let normal = Vec2::up(); // Reflect about the y-axis
    /// let reflected = vec.reflect(normal);
    /// assert_eq!(reflected.x, 1.0);
    /// assert_eq!(reflected.y, -1.0);
    /// ```
    #[must_use]
    pub fn reflect(&self, normal: Vec2) -> Vec2 {
        *self - 2.0 * self.dot(normal) * normal
    }

    /// Returns a new vector where each component is the reciprocal (1/x) of the corresponding component.
    ///
    ///
    /// # Examples
    ///
    /// ```
    /// use glongge::core::prelude::*;
    /// let vec = Vec2 { x: 2.0, y: 4.0 };
    /// let reciprocal = vec.reciprocal();
    /// assert_eq!(reciprocal.x, 0.5);
    /// assert_eq!(reciprocal.y, 0.25);
    /// ```
    #[must_use]
    pub fn reciprocal(&self) -> Vec2 {
        if self.is_zero() {
            let mut rv = *self;
            rv.x = gg_float::force_positive_zero(rv.x);
            rv.y = gg_float::force_positive_zero(rv.y);
            rv
        } else {
            Vec2 {
                x: 1.0 / self.x,
                y: 1.0 / self.y,
            }
        }
    }

    /// Returns an orthogonal vector, which is perpendicular to this vector.
    ///
    /// The orthogonal vector is obtained by swapping the components and negating the x component.
    /// That is, the result is rotated 90 degrees clockwise from the original vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use glongge::core::prelude::*;
    /// let vec = Vec2 { x: 3.0, y: 2.0 };
    /// let perpendicular = vec.orthog();
    /// assert_eq!(perpendicular.x, 2.0);
    /// assert_eq!(perpendicular.y, -3.0);
    /// assert_eq!(vec.dot(perpendicular), 0.0); // Vectors are perpendicular
    /// ```
    #[must_use]
    pub fn orthog(&self) -> Vec2 {
        Vec2 {
            x: self.y,
            y: -self.x,
        }
    }
    /// Performs a component-wise multiplication of two vectors.
    ///
    /// # Examples
    ///
    /// ```
    /// use glongge::core::prelude::*;
    /// let v1 = Vec2 { x: 2.0, y: 3.0 };
    /// let v2 = Vec2 { x: 4.0, y: 5.0 };
    /// let result = v1.component_wise(v2);
    /// assert_eq!(result, Vec2 { x: 8.0, y: 15.0 });
    /// ```
    #[must_use]
    pub fn component_wise(&self, other: Vec2) -> Vec2 {
        Vec2 {
            x: self.x * other.x,
            y: self.y * other.y,
        }
    }

    /// Performs a component-wise division of two vectors.
    ///
    /// # Examples
    ///
    /// ```
    /// use glongge::core::prelude::*;
    /// let v1 = Vec2 { x: 8.0, y: 15.0 };
    /// let v2 = Vec2 { x: 4.0, y: 5.0 };
    /// let result = v1.component_wise_div(v2);
    /// assert_eq!(result, Vec2 { x: 2.0, y: 3.0 });
    /// ```
    #[must_use]
    pub fn component_wise_div(&self, other: Vec2) -> Vec2 {
        Vec2 {
            x: self.x / other.x,
            y: self.y / other.y,
        }
    }

    /// Computes the dot product of two vectors.
    ///
    /// # Examples
    ///
    /// ```
    /// use glongge::core::prelude::*;
    /// let v1 = Vec2 { x: 2.0, y: 3.0 };
    /// let v2 = Vec2 { x: 4.0, y: 5.0 };
    /// let dot_product = v1.dot(v2);
    /// assert_eq!(dot_product, 23.0); // 2*4 + 3*5
    /// ```
    #[must_use]
    pub fn dot(&self, other: Vec2) -> f32 {
        self.x * other.x + self.y * other.y
    }

    /// Computes the 2D cross product of two vectors.
    ///
    /// In 2D, the cross product is a scalar representing the signed area of the
    /// parallelogram formed by the two vectors. It is positive if the second vector
    /// is counter-clockwise from the first vector, and negative otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use glongge::core::prelude::*;
    /// let v1 = Vec2 { x: 2.0, y: 0.0 };
    /// let v2 = Vec2 { x: 0.0, y: 3.0 };
    /// let cross_product = v1.cross(v2);
    /// assert_eq!(cross_product, 6.0); // 2*3 - 0*0
    /// ```
    /// ```
    /// use glongge::core::prelude::*;
    /// let v1 = Vec2 { x: 2.0, y: 0.0 };
    /// let v2 = Vec2 { x: 0.0, y: -3.0 };
    /// let cross_product = v1.cross(v2);
    /// assert_eq!(cross_product, -6.0); // 2*-3 - 0*0
    /// ```
    #[must_use]
    pub fn cross(&self, other: Vec2) -> f32 {
        self.x * other.y - self.y * other.x
    }

    /// Calculates the angle in radians between two vectors.
    ///
    /// Returns the smallest angle between the two vectors.
    /// The result is always in the range [0, π).
    ///
    /// # Examples
    ///
    /// ```
    /// use glongge::core::prelude::*;
    /// let v1 = Vec2::right(); // (1, 0)
    /// let v2 = Vec2::up();    // (0, 1)
    /// let angle = v1.angle_radians(v2);
    /// assert_eq!(angle, std::f32::consts::FRAC_PI_2); // 90 degrees
    /// ```
    /// ```
    /// use glongge::core::prelude::*;
    /// let v1 = Vec2::right(); // (1, 0)
    /// let v2 = Vec2::left();  // (-1, 0)
    /// let angle = v1.angle_radians(v2);
    /// assert!((angle - std::f32::consts::PI).abs() < EPSILON); // 180 degrees
    /// ```
    #[must_use]
    pub fn angle_radians(&self, other: Vec2) -> f32 {
        self.normed().dot(other.normed()).acos()
    }

    /// Projects this vector onto the given axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use glongge::core::prelude::*;
    /// let v = Vec2 { x: 3.0, y: 4.0 };
    /// let axis = Vec2::right(); // (1, 0)
    /// let projected = v.project(axis);
    /// assert_eq!(projected, Vec2 { x: 3.0, y: 0.0 });
    /// ```
    #[must_use]
    pub fn project(&self, axis: Vec2) -> Vec2 {
        self.dot(axis.normed()) * axis.normed()
    }
    /// Projects the vector onto the x-axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use glongge::core::prelude::*;
    /// let v = Vec2 { x: 3.0, y: 4.0 };
    /// assert_eq!(v.project_x(), Vec2 { x: 3.0, y: 0.0 });
    /// ```
    #[must_use]
    pub fn project_x(&self) -> Vec2 {
        self.x * Vec2::right()
    }

    /// Projects the vector onto the y-axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use glongge::core::prelude::*;
    /// let v = Vec2 { x: 3.0, y: 4.0 };
    /// assert_eq!(v.project_y(), Vec2 { x: 0.0, y: 4.0 });
    /// ```
    #[must_use]
    pub fn project_y(&self) -> Vec2 {
        self.y * Vec2::down()
    }

    /// Computes the Euclidean distance between two points.
    ///
    /// # Examples
    ///
    /// ```
    /// use glongge::core::prelude::*;
    /// let p1 = Vec2 { x: 0.0, y: 0.0 };
    /// let p2 = Vec2 { x: 3.0, y: 4.0 };
    /// let distance = p1.dist(p2);
    /// assert_eq!(distance, 5.0);
    /// ```
    #[must_use]
    pub fn dist(&self, other: Vec2) -> f32 {
        (other - *self).len()
    }

    /// Computes the squared Euclidean distance between two points.
    ///
    /// More efficient than `dist` when only comparing distances, as it avoids the square root
    /// operation.
    ///
    /// # Examples
    ///
    /// ```
    /// use glongge::core::prelude::*;
    /// let p1 = Vec2 { x: 0.0, y: 0.0 };
    /// let p2 = Vec2 { x: 3.0, y: 4.0 };
    /// let distance_squared = p1.dist_squared(p2);
    /// assert_eq!(distance_squared, 25.0);
    /// ```
    #[must_use]
    pub fn dist_squared(&self, other: Vec2) -> f32 {
        (other - *self).len_squared()
    }

    /// Calculates the shortest distance from this point to a line segment.
    ///
    /// # Examples
    ///
    /// ```
    /// use glongge::core::prelude::*;
    /// let point = Vec2 { x: 0.0, y: 1.0 };
    /// let start = Vec2 { x: -1.0, y: 0.0 };
    /// let end = Vec2 { x: 1.0, y: 0.0 };
    /// let distance = point.dist_to_line(start, end);
    /// assert_eq!(distance, 1.0);
    /// ```
    ///
    /// # Edge Cases
    ///
    /// - If `start` and `end` are the same point, returns the distance to that point.
    /// - The projection of the point onto the line is clamped to the segment.
    #[must_use]
    pub fn dist_to_line(&self, start: Vec2, end: Vec2) -> f32 {
        if start == end {
            return self.dist(start);
        }
        let dx = end - start;
        let l2 = dx.len_squared();
        let t = ((*self - start).dot(dx) / l2).clamp(0.0, 1.0);
        self.dist(start + t * dx)
    }

    /// Calculates the intersection point of two line segments.
    ///
    /// The first line segment is from `p1` extending along `ax1`,
    /// and the second is from `p2` extending along `ax2`.
    ///
    /// # Parameters
    ///
    /// - `p1`: Starting point of the first line segment
    /// - `ax1`: Direction and length of the first line segment
    /// - `p2`: Starting point of the second line segment
    /// - `ax2`: Direction and length of the second line segment
    ///
    /// # Returns
    ///
    /// - `Some(Vec2)`: The intersection point if the segments intersect
    /// - `None`: If the segments are parallel or do not intersect
    ///
    /// # Examples
    ///
    /// ```
    /// use glongge::core::prelude::*;
    /// let p1 = Vec2 { x: 0.0, y: 0.0 };
    /// let ax1 = Vec2 { x: 1.0, y: 0.0 }; // Horizontal segment from (0,0) to (1,0)
    /// let p2 = Vec2 { x: 0.5, y: -1.0 };
    /// let ax2 = Vec2 { x: 0.0, y: 2.0 }; // Vertical segment from (0.5,-1) to (0.5,1)
    /// let intersection = Vec2::intersect(p1, ax1, p2, ax2);
    /// assert_eq!(intersection, Some(Vec2 { x: 0.5, y: 0.0 }));
    /// ```
    #[must_use]
    pub fn intersect(p1: Vec2, ax1: Vec2, p2: Vec2, ax2: Vec2) -> Option<Vec2> {
        let denom = ax1.cross(ax2);
        if denom.is_zero() {
            None
        } else {
            let t = (p2 - p1).cross(ax2) / denom;
            let u = (p2 - p1).cross(ax1) / denom;
            if (0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u) {
                Some(p1 + t * ax1)
            } else {
                None
            }
        }
    }

    /// Linearly interpolates between this vector and another vector.
    ///
    /// The interpolation parameter `t` should be in the range [0, 1], where:
    /// - `t = 0.0` returns this vector
    /// - `t = 1.0` returns the `to` vector
    /// - Values in between return a proportional mix of the two vectors.
    /// - `t` is clamped if it is outside the range [0, 1].
    ///
    /// # Examples
    ///
    /// ```
    /// use glongge::core::prelude::*;
    /// let v1 = Vec2 { x: 0.0, y: 0.0 };
    /// let v2 = Vec2 { x: 10.0, y: 20.0 };
    /// let halfway = v1.lerp(v2, 0.5);
    /// assert_eq!(halfway, Vec2 { x: 5.0, y: 10.0 });
    /// ```
    #[must_use]
    pub fn lerp(&self, to: Vec2, t: f32) -> Vec2 {
        let t = t.clamp(0.0, 1.0);
        Vec2 {
            x: lerp(self.x, to.x, t),
            y: lerp(self.y, to.y, t),
        }
    }

    /// Checks if the vector is approximately equal to another vector.
    ///
    /// Two vectors are considered approximately equal if the length of their difference
    /// is less than [`EPSILON`](crate::core::config::EPSILON).
    pub fn almost_eq(&self, rhs: Vec2) -> bool {
        (*self - rhs).len() < EPSILON
    }

    /// Converts the vector to a [`Vec2i`] by rounding each component to the nearest integer.
    ///
    /// This is a lossy conversion as it truncates the decimal portion after rounding.
    #[must_use]
    pub fn as_vec2int_lossy(&self) -> Vec2i {
        Vec2i {
            x: self.x.round() as i32,
            y: self.y.round() as i32,
        }
    }

    /// Compares two vectors based on their squared length.
    ///
    /// This function first attempts to compare using [`partial_cmp()`](f32::partial_cmp), which may
    /// fail with NaN values. If partial comparison fails, it falls back to
    /// [`total_cmp()`](f32::total_cmp) and logs a warning.
    ///
    /// # Edge Cases
    /// - If either vector contains NaN components, [`partial_cmp()`](f32::partial_cmp) will return
    ///   `None`, and the function will fall back to [`total_cmp()`](f32::total_cmp) which handles
    ///   NaN values deterministically.
    #[must_use]
    pub fn cmp_by_length(&self, other: &Vec2) -> Ordering {
        let self_len = self.len_squared();
        let other_len = other.len_squared();
        self_len.partial_cmp(&other_len).unwrap_or_else(|| {
            warn!(
                "cmp_by_length(): partial_cmp() failed: {} vs. {}",
                self, other
            );
            self_len.total_cmp(&other_len)
        })
    }

    /// Compares two vectors based on their distance from a given origin point.
    ///
    /// This function first attempts to compare using [`partial_cmp()`](f32::partial_cmp), which may
    /// fail with NaN values. If partial comparison fails, it falls back to
    /// [`total_cmp()`](f32::total_cmp) and logs a warning that includes the origin point.
    ///
    /// # Edge Cases
    /// - If the arguments contain NaN components, [`partial_cmp()`](f32::partial_cmp)
    ///   will return `None`, and the function will fall back to [`total_cmp()`](f32::total_cmp)
    ///   which handles NaN values deterministically.
    #[must_use]
    pub fn cmp_by_dist(&self, other: &Vec2, origin: Vec2) -> Ordering {
        let self_len = (*self - origin).len_squared();
        let other_len = (*other - origin).len_squared();
        self_len.partial_cmp(&other_len).unwrap_or_else(|| {
            warn!(
                "cmp_by_dist() to {}: partial_cmp() failed: {} vs. {}",
                origin, self, other
            );
            self_len.total_cmp(&other_len)
        })
    }
}

impl Zero for Vec2 {
    fn zero() -> Self {
        Vec2::zero()
    }

    fn is_zero(&self) -> bool {
        self.almost_eq(Self::zero())
    }
}

impl From<[f32; 2]> for Vec2 {
    fn from(value: [f32; 2]) -> Self {
        Vec2 {
            x: value[0],
            y: value[1],
        }
    }
}
impl From<[i32; 2]> for Vec2 {
    fn from(value: [i32; 2]) -> Self {
        Vec2 {
            x: value[0] as f32,
            y: value[1] as f32,
        }
    }
}

impl From<Vec2> for [f32; 2] {
    fn from(value: Vec2) -> Self {
        [value.x, value.y]
    }
}

impl fmt::Display for Vec2 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let precision = f.precision();

        write!(f, "vec(")?;
        if let Some(p) = precision {
            write!(f, "{0:.1$}", self.x, p)?;
            write!(f, ", {0:.1$}", self.y, p)?;
        } else {
            write!(f, "{}, {}", self.x, self.y)?;
        }
        write!(f, ")")
    }
}

impl Add<Vec2> for Vec2 {
    type Output = Vec2;

    fn add(self, rhs: Vec2) -> Self::Output {
        Vec2 {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}
impl AddAssign<Vec2> for Vec2 {
    fn add_assign(&mut self, rhs: Vec2) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl Sub<Vec2> for Vec2 {
    type Output = Vec2;

    fn sub(self, rhs: Vec2) -> Self::Output {
        Vec2 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}
impl SubAssign<Vec2> for Vec2 {
    fn sub_assign(&mut self, rhs: Vec2) {
        self.x -= rhs.x;
        self.y -= rhs.y;
    }
}

impl Sum<Vec2> for Vec2 {
    fn sum<I: Iterator<Item = Vec2>>(iter: I) -> Self {
        iter.fold(Vec2::zero(), Vec2::add)
    }
}

impl Mul<f32> for Vec2 {
    type Output = Vec2;

    fn mul(self, rhs: f32) -> Self::Output {
        rhs * self
    }
}
impl Mul<Vec2> for f32 {
    type Output = Vec2;

    fn mul(self, rhs: Vec2) -> Self::Output {
        Vec2 {
            x: self * rhs.x,
            y: self * rhs.y,
        }
    }
}
impl Mul<&Vec2> for f32 {
    type Output = Vec2;

    fn mul(self, rhs: &Vec2) -> Self::Output {
        Vec2 {
            x: self * rhs.x,
            y: self * rhs.y,
        }
    }
}
impl MulAssign<f32> for Vec2 {
    fn mul_assign(&mut self, rhs: f32) {
        self.x *= rhs;
        self.y *= rhs;
    }
}
impl Mul<i32> for Vec2 {
    type Output = Vec2;

    fn mul(self, rhs: i32) -> Self::Output {
        (rhs as f32) * self
    }
}
impl Mul<Vec2> for i32 {
    type Output = Vec2;

    fn mul(self, rhs: Vec2) -> Self::Output {
        Vec2 {
            x: self as f32 * rhs.x,
            y: self as f32 * rhs.y,
        }
    }
}
impl Mul<&Vec2> for i32 {
    type Output = Vec2;

    fn mul(self, rhs: &Vec2) -> Self::Output {
        Vec2 {
            x: self as f32 * rhs.x,
            y: self as f32 * rhs.y,
        }
    }
}
impl MulAssign<i32> for Vec2 {
    fn mul_assign(&mut self, rhs: i32) {
        self.x *= rhs as f32;
        self.y *= rhs as f32;
    }
}
impl Mul<u32> for Vec2 {
    type Output = Vec2;

    fn mul(self, rhs: u32) -> Self::Output {
        rhs as f32 * self
    }
}
impl Mul<Vec2> for u32 {
    type Output = Vec2;

    fn mul(self, rhs: Vec2) -> Self::Output {
        Vec2 {
            x: self as f32 * rhs.x,
            y: self as f32 * rhs.y,
        }
    }
}
impl Mul<&Vec2> for u32 {
    type Output = Vec2;

    fn mul(self, rhs: &Vec2) -> Self::Output {
        Vec2 {
            x: self as f32 * rhs.x,
            y: self as f32 * rhs.y,
        }
    }
}
impl MulAssign<u32> for Vec2 {
    fn mul_assign(&mut self, rhs: u32) {
        self.x *= rhs as f32;
        self.y *= rhs as f32;
    }
}

impl Div<f32> for Vec2 {
    type Output = Vec2;

    fn div(self, rhs: f32) -> Self::Output {
        Vec2 {
            x: self.x / rhs,
            y: self.y / rhs,
        }
    }
}
impl DivAssign<f32> for Vec2 {
    fn div_assign(&mut self, rhs: f32) {
        self.x /= rhs;
        self.y /= rhs;
    }
}
impl Div<i32> for Vec2 {
    type Output = Vec2;

    fn div(self, rhs: i32) -> Self::Output {
        Vec2 {
            x: self.x / rhs as f32,
            y: self.y / rhs as f32,
        }
    }
}
impl DivAssign<i32> for Vec2 {
    fn div_assign(&mut self, rhs: i32) {
        self.x /= rhs as f32;
        self.y /= rhs as f32;
    }
}
impl Div<u32> for Vec2 {
    type Output = Vec2;

    fn div(self, rhs: u32) -> Self::Output {
        Vec2 {
            x: self.x / rhs as f32,
            y: self.y / rhs as f32,
        }
    }
}
impl DivAssign<u32> for Vec2 {
    fn div_assign(&mut self, rhs: u32) {
        self.x /= rhs as f32;
        self.y /= rhs as f32;
    }
}

impl Neg for Vec2 {
    type Output = Vec2;

    fn neg(self) -> Self::Output {
        Vec2 {
            x: -self.x,
            y: -self.y,
        }
    }
}
impl Neg for &Vec2 {
    type Output = Vec2;

    fn neg(self) -> Self::Output {
        Vec2 {
            x: -self.x,
            y: -self.y,
        }
    }
}

#[derive(
    Default, Debug, Eq, PartialEq, Ord, PartialOrd, Copy, Clone, Hash, Serialize, Deserialize,
)]
pub struct Vec2i {
    pub x: i32,
    pub y: i32,
}

impl Vec2i {
    /// Returns a unit vector pointing to the right (positive x-axis).
    #[must_use]
    pub fn right() -> Vec2i {
        Vec2i { x: 1, y: 0 }
    }
    /// Returns a unit vector pointing upward (negative y-axis).
    ///
    /// Note: This follows a coordinate system where y increases downward,
    /// which is common in 2D graphics applications.
    #[must_use]
    pub fn up() -> Vec2i {
        Vec2i { x: 0, y: -1 }
    }
    /// Returns a unit vector pointing to the left (negative x-axis).
    #[must_use]
    pub fn left() -> Vec2i {
        Vec2i { x: -1, y: 0 }
    }
    /// Returns a unit vector pointing downward (positive y-axis).
    ///
    /// Note: This follows a coordinate system where y increases downward,
    /// which is common in 2D graphics applications.
    #[must_use]
    pub fn down() -> Vec2i {
        Vec2i { x: 0, y: 1 }
    }
    /// Returns a vector with both components set to 1.0.
    ///
    /// This can be useful for scaling operations or as a basis
    /// for component-wise operations.
    #[must_use]
    pub fn one() -> Vec2i {
        Vec2i { x: 1, y: 1 }
    }
    /// Returns a vector with both components set to 0.0.
    ///
    /// This can be useful for initialisation or as a neutral element
    /// for addition operations.
    #[must_use]
    pub fn zero() -> Vec2i {
        Vec2i { x: 0, y: 0 }
    }

    /// Converts a [`Vec2i`] to [`Vec2`].
    ///
    /// This is a convenience method that simply calls `Into::<Vec2>::into(*self)`.
    pub fn as_vec2(&self) -> Vec2 {
        Into::<Vec2>::into(*self)
    }

    /// Creates a Cartesian product of two ranges, from `start` to `end` (exclusive).
    ///
    /// This method produces a two-dimensional range that iterates through all
    /// integer coordinates in the rectangle defined by `start` (inclusive) and
    /// `end` (exclusive).
    pub fn range(start: Vec2i, end: Vec2i) -> Product<Range<i32>, Range<i32>> {
        (start.x..end.x).cartesian_product(start.y..end.y)
    }

    /// Creates a Cartesian product of two ranges, from `(0, 0)` to the given `end` (exclusive).
    ///
    /// This is a convenience wrapper around `range()` that starts at the origin.
    /// Commonly used for iterating through grid-based data like tilesets or pixel regions.
    pub fn range_from_zero(end: impl Into<Vec2i>) -> Product<Range<i32>, Range<i32>> {
        Self::range(Vec2i::zero(), end.into())
    }

    /// Calculates a linear index into a 2D array with the given dimensions.
    ///
    /// Converts a 2D coordinate to a 1D index using row-major order.
    /// Includes bounds checking to ensure the coordinates are within the valid range.
    #[allow(clippy::cast_sign_loss)]
    pub fn as_index(&self, width: u32, height: u32) -> usize {
        check_ge!(self.x, 0);
        check_ge!(self.y, 0);
        check_lt!(self.x as u32, width);
        check_lt!(self.y as u32, height);
        (self.y as u32 * width + self.x as u32) as usize
    }
}

impl From<Vec2i> for Vec2 {
    fn from(value: Vec2i) -> Self {
        Self {
            x: value.x as f32,
            y: value.y as f32,
        }
    }
}

impl Zero for Vec2i {
    fn zero() -> Self {
        Self::zero()
    }

    fn is_zero(&self) -> bool {
        *self == Self::zero()
    }
}

impl From<[i32; 2]> for Vec2i {
    fn from(value: [i32; 2]) -> Self {
        Vec2i {
            x: value[0],
            y: value[1],
        }
    }
}

impl From<Vec2i> for [i32; 2] {
    fn from(value: Vec2i) -> Self {
        [value.x, value.y]
    }
}

impl From<Vec2i> for [u32; 2] {
    fn from(value: Vec2i) -> Self {
        [
            value.x.abs().try_into().unwrap(),
            value.y.abs().try_into().unwrap(),
        ]
    }
}

impl fmt::Display for Vec2i {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "vec({}, {})", self.x, self.y)
    }
}

/// Represents a 2D edge between two integer points.
///
/// An [`Edge2i`] consists of two [`Vec2i`] points that define the start and end points of the edge.
/// The edge is directed from the first point (`0`) to the second point (`1`).
///
/// # Examples
///
/// ```
/// use glongge::util::linalg::*;
///
/// let start = Vec2i { x: 0, y: 0 };
/// let end = Vec2i { x: 5, y: 3 };
/// let edge = Edge2i(start, end);
///
/// // Create a reversed edge
/// let reversed = edge.reverse();
/// assert_eq!(reversed, Edge2i(end, start));
/// ```
#[derive(
    Default, Debug, Eq, PartialEq, Ord, PartialOrd, Copy, Clone, Hash, Serialize, Deserialize,
)]
pub struct Edge2i(pub Vec2i, pub Vec2i);

impl Edge2i {
    /// Creates a new edge with the start and end points reversed.
    #[must_use]
    pub fn reverse(self) -> Self {
        Self(self.1, self.0)
    }
}

impl fmt::Display for Edge2i {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Edge[{}, {}]", self.0, self.1)
    }
}

impl Add<Vec2i> for Vec2i {
    type Output = Vec2i;

    fn add(self, rhs: Vec2i) -> Self::Output {
        Vec2i {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}
impl AddAssign<Vec2i> for Vec2i {
    fn add_assign(&mut self, rhs: Vec2i) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl Sub<Vec2i> for Vec2i {
    type Output = Vec2i;

    fn sub(self, rhs: Vec2i) -> Self::Output {
        Vec2i {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}
impl SubAssign<Vec2i> for Vec2i {
    fn sub_assign(&mut self, rhs: Vec2i) {
        self.x -= rhs.x;
        self.y -= rhs.y;
    }
}

impl Mul<i32> for Vec2i {
    type Output = Vec2i;

    fn mul(self, rhs: i32) -> Self::Output {
        rhs * self
    }
}
impl Mul<Vec2i> for i32 {
    type Output = Vec2i;

    fn mul(self, rhs: Vec2i) -> Self::Output {
        Vec2i {
            x: self * rhs.x,
            y: self * rhs.y,
        }
    }
}
impl Mul<&Vec2i> for i32 {
    type Output = Vec2i;

    fn mul(self, rhs: &Vec2i) -> Self::Output {
        Vec2i {
            x: self * rhs.x,
            y: self * rhs.y,
        }
    }
}
impl MulAssign<i32> for Vec2i {
    fn mul_assign(&mut self, rhs: i32) {
        self.x *= rhs;
        self.y *= rhs;
    }
}

impl Div<i32> for Vec2i {
    type Output = Vec2i;

    fn div(self, rhs: i32) -> Self::Output {
        Vec2i {
            x: self.x / rhs,
            y: self.y / rhs,
        }
    }
}
impl DivAssign<i32> for Vec2i {
    fn div_assign(&mut self, rhs: i32) {
        self.x /= rhs;
        self.y /= rhs;
    }
}

impl Neg for Vec2i {
    type Output = Vec2i;

    fn neg(self) -> Self::Output {
        Vec2i {
            x: -self.x,
            y: -self.y,
        }
    }
}
impl Neg for &Vec2i {
    type Output = Vec2i;

    fn neg(self) -> Self::Output {
        Vec2i {
            x: -self.x,
            y: -self.y,
        }
    }
}

/// A 3x3 matrix representation for 2D transformations.
///
/// This matrix uses homogeneous coordinates to represent 2D transformations.
/// The elements are arranged as follows:
/// ```text
/// | xx xy xw |
/// | yx yy yw |
/// | wx wy ww |
/// ```
/// where the first two columns represent linear transformation components,
/// and the third column represents translation components.
#[derive(Copy, Clone, PartialEq)]
#[must_use]
pub struct Mat3x3 {
    pub xx: f32,
    pub xy: f32,
    pub xw: f32,
    pub yx: f32,
    pub yy: f32,
    pub yw: f32,
    pub wx: f32,
    pub wy: f32,
    pub ww: f32,
}

impl Mat3x3 {
    /// Creates an identity matrix.
    ///
    /// Returns a matrix representing no transformation (identity matrix):
    /// ```text
    /// | 1 0 0 |
    /// | 0 1 0 |
    /// | 0 0 1 |
    /// ```
    pub fn one() -> Mat3x3 {
        Mat3x3 {
            xx: 1.0,
            xy: 0.0,
            xw: 0.0,
            yx: 0.0,
            yy: 1.0,
            yw: 0.0,
            wx: 0.0,
            wy: 0.0,
            ww: 1.0,
        }
    }

    /// Creates a zero matrix.
    ///
    /// Returns a matrix with all elements set to 0:
    /// ```text
    /// | 0 0 0 |
    /// | 0 0 0 |
    /// | 0 0 0 |
    /// ```
    pub fn zero() -> Mat3x3 {
        Mat3x3 {
            xx: 0.0,
            xy: 0.0,
            xw: 0.0,
            yx: 0.0,
            yy: 0.0,
            yw: 0.0,
            wx: 0.0,
            wy: 0.0,
            ww: 0.0,
        }
    }

    /// Creates a translation matrix.
    ///
    /// Returns a matrix that translates points by (dx, dy):
    /// ```text
    /// | 1 0 dx |
    /// | 0 1 dy |
    /// | 0 0 1  |
    /// ```
    pub fn translation(dx: f32, dy: f32) -> Mat3x3 {
        Mat3x3 {
            xx: 1.0,
            xy: 0.0,
            xw: dx,
            yx: 0.0,
            yy: 1.0,
            yw: dy,
            wx: 0.0,
            wy: 0.0,
            ww: 1.0,
        }
    }

    /// Creates a translation matrix from a Vec2.
    pub fn translation_vec2(vec2: Vec2) -> Mat3x3 {
        Self::translation(vec2.x, vec2.y)
    }

    /// Creates a rotation matrix.
    ///
    /// Returns a matrix that rotates points counterclockwise by the specified angle:
    /// ```text
    /// | cos(θ)  -sin(θ)  0 |
    /// | sin(θ)   cos(θ)  0 |
    /// | 0        0       1 |
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// use glongge::core::prelude::*;
    ///
    /// let rot = Mat3x3::rotation(std::f32::consts::FRAC_PI_2); // 90 degrees
    /// let v = Vec2 { x: 1.0, y: 0.0 };
    /// let rotated = rot * v;
    ///
    /// // Vector should be rotated 90 degrees counterclockwise
    /// assert!((rotated.x - 0.0).abs() < EPSILON);
    /// assert!((rotated.y - 1.0).abs() < EPSILON);
    /// ```
    pub fn rotation(radians: f32) -> Mat3x3 {
        Mat3x3 {
            xx: f32::cos(radians),
            xy: -f32::sin(radians),
            xw: 0.0,
            yx: f32::sin(radians),
            yy: f32::cos(radians),
            yw: 0.0,
            wx: 0.0,
            wy: 0.0,
            ww: 1.0,
        }
    }

    /// Calculates the determinant of the matrix.
    ///
    /// # Examples
    /// ```
    /// use glongge::util::linalg::Mat3x3;
    ///
    /// let m = Mat3x3::rotation(0.0); // Identity matrix
    /// assert_eq!(m.det(), 1.0);
    /// ```
    pub fn det(&self) -> f32 {
        self.xx * (self.yy * self.ww - self.yw * self.wy)
            + self.xy * (self.yx * self.ww - self.yw * self.wx)
            + self.xw * (self.yx * self.wy - self.yy * self.wx)
    }

    /// Creates a new matrix that is the transpose of this matrix.
    ///
    /// The transpose swaps rows and columns:
    /// ```text
    /// | xx yx wx |
    /// | xy yy wy |
    /// | xw yw ww |
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// use glongge::core::prelude::*;
    ///
    /// let m = Mat3x3::translation(2.0, 3.0);
    /// let m_t = m.transposed();
    ///
    /// // Translation components move from third column to third row
    /// assert_eq!(m_t.wx, m.xw);
    /// assert_eq!(m_t.wy, m.yw);
    /// ```
    pub fn transposed(&self) -> Mat3x3 {
        Mat3x3 {
            xx: self.xx,
            xy: self.yx,
            xw: self.wx,
            yx: self.xy,
            yy: self.yy,
            yw: self.wy,
            wx: self.xw,
            wy: self.yw,
            ww: self.ww,
        }
    }

    /// Compares two matrices for approximate equality.
    ///
    /// # Examples
    ///
    /// ```
    /// use glongge::core::prelude::*;
    ///
    /// let m1 = Mat3x3::rotation(0.1);
    /// let m2 = Mat3x3::rotation(0.1 + EPSILON); // Slightly different angle
    /// assert!(m1.almost_eq(m2)); // Should be approximately equal
    ///
    /// let m3 = Mat3x3::rotation(0.2); // Different angle
    /// assert!(!m1.almost_eq(m3)); // Should not be equal
    /// ```
    pub fn almost_eq(&self, rhs: Mat3x3) -> bool {
        f32::abs(self.xx - rhs.xx) < EPSILON
            && f32::abs(self.xy - rhs.xy) < EPSILON
            && f32::abs(self.xw - rhs.xw) < EPSILON
            && f32::abs(self.yx - rhs.yx) < EPSILON
            && f32::abs(self.yy - rhs.yy) < EPSILON
            && f32::abs(self.yw - rhs.yw) < EPSILON
            && f32::abs(self.wx - rhs.wx) < EPSILON
            && f32::abs(self.wy - rhs.wy) < EPSILON
            && f32::abs(self.ww - rhs.ww) < EPSILON
    }
}

impl One for Mat3x3 {
    fn one() -> Self {
        Self::one()
    }
}

impl Zero for Mat3x3 {
    fn zero() -> Self {
        Self::zero()
    }

    fn is_zero(&self) -> bool {
        self.almost_eq(Self::zero())
    }
}

impl Add<Mat3x3> for Mat3x3 {
    type Output = Mat3x3;

    fn add(self, rhs: Mat3x3) -> Self::Output {
        Mat3x3 {
            xx: rhs.xx * self.xx,
            xy: self.xy * self.xy,
            xw: self.xw * self.xw,
            yx: rhs.yx * self.yx,
            yy: self.yy * self.yy,
            yw: self.yw * self.yw,
            wx: rhs.wx * self.wx,
            wy: self.wy * self.wy,
            ww: self.ww * self.ww,
        }
    }
}

impl Mul<f32> for Mat3x3 {
    type Output = Mat3x3;

    fn mul(self, rhs: f32) -> Self::Output {
        Mat3x3 {
            xx: rhs * self.xx,
            xy: rhs * self.xy,
            xw: rhs * self.xw,
            yx: rhs * self.yx,
            yy: rhs * self.yy,
            yw: rhs * self.yw,
            wx: rhs * self.wx,
            wy: rhs * self.wy,
            ww: rhs * self.ww,
        }
    }
}
impl Mul<Mat3x3> for f32 {
    type Output = Mat3x3;

    fn mul(self, rhs: Mat3x3) -> Self::Output {
        Mat3x3 {
            xx: self * rhs.xx,
            xy: self * rhs.xy,
            xw: self * rhs.xw,
            yx: self * rhs.yx,
            yy: self * rhs.yy,
            yw: self * rhs.yw,
            wx: self * rhs.wx,
            wy: self * rhs.wy,
            ww: self * rhs.ww,
        }
    }
}
impl MulAssign<f32> for Mat3x3 {
    fn mul_assign(&mut self, rhs: f32) {
        self.xx *= rhs;
        self.xy *= rhs;
        self.xw *= rhs;
        self.yx *= rhs;
        self.yy *= rhs;
        self.yw *= rhs;
        self.wx *= rhs;
        self.wy *= rhs;
        self.ww *= rhs;
    }
}

impl Div<f32> for Mat3x3 {
    type Output = Mat3x3;

    fn div(self, rhs: f32) -> Self::Output {
        Mat3x3 {
            xx: self.xx / rhs,
            xy: self.xy / rhs,
            xw: self.xw / rhs,
            yx: self.yx / rhs,
            yy: self.yy / rhs,
            yw: self.yw / rhs,
            wx: self.wx / rhs,
            wy: self.wy / rhs,
            ww: self.ww / rhs,
        }
    }
}
impl DivAssign<f32> for Mat3x3 {
    fn div_assign(&mut self, rhs: f32) {
        self.xx /= rhs;
        self.xy /= rhs;
        self.xw /= rhs;
        self.yx /= rhs;
        self.yy /= rhs;
        self.yw /= rhs;
        self.wx /= rhs;
        self.wy /= rhs;
        self.ww /= rhs;
    }
}

impl Mul<Vec2> for Mat3x3 {
    type Output = Vec2;

    fn mul(self, rhs: Vec2) -> Self::Output {
        Vec2 {
            x: self.xx * rhs.x + self.xy * rhs.y + self.xw * 1.0,
            y: self.yx * rhs.x + self.yy * rhs.y + self.yw * 1.0,
        }
    }
}
impl MulAssign<Mat3x3> for Vec2 {
    fn mul_assign(&mut self, rhs: Mat3x3) {
        (self.x, self.y) = (
            rhs.xx * self.x + rhs.xy * self.y + rhs.xw * 1.0,
            rhs.yx * self.x + rhs.yy * self.y + rhs.yw * 1.0,
        );
    }
}

impl Mul<Mat3x3> for Mat3x3 {
    type Output = Mat3x3;

    fn mul(self, rhs: Mat3x3) -> Self::Output {
        Mat3x3 {
            xx: self.xx * rhs.xx + self.xy * rhs.yx + self.xw * rhs.wx,
            xy: self.xx * rhs.xy + self.xy * rhs.yy + self.xw * rhs.wy,
            xw: self.xx * rhs.xw + self.xy * rhs.yw + self.xw * rhs.ww,
            yx: self.yx * rhs.xx + self.yy * rhs.yx + self.yw * rhs.wx,
            yy: self.yx * rhs.xy + self.yy * rhs.yy + self.yw * rhs.wy,
            yw: self.yx * rhs.xw + self.yy * rhs.yw + self.yw * rhs.ww,
            wx: self.wx * rhs.xx + self.wy * rhs.yx + self.ww * rhs.wx,
            wy: self.wx * rhs.xy + self.wy * rhs.yy + self.ww * rhs.wy,
            ww: self.wx * rhs.xw + self.wy * rhs.yw + self.ww * rhs.ww,
        }
    }
}

impl From<Mat3x3> for [[f32; 4]; 4] {
    fn from(value: Mat3x3) -> Self {
        [
            [value.xx, value.xy, 0.0, value.xw],
            [value.yx, value.yy, 0.0, value.yw],
            [0.0, 0.0, 1.0, value.ww],
            [value.wx, value.wy, 0.0, 1.0],
        ]
    }
}

/// Trait for types that have an axis-aligned bounding box representation.
///
/// This trait provides methods to work with rectangular boundaries aligned to coordinate axes.
/// It allows querying dimensions, corners, edges, and containment tests for the implementing type.
///
/// The trait requires implementing two core methods:
/// - [`extent()`](AxisAlignedExtent::extent) - Returns the width and height as a [`Vec2`]
/// - [`centre()`](AxisAlignedExtent::centre) - Returns the center point as a [`Vec2`]
///
/// All other methods have default implementations based on these two.
///
/// # Examples
///
/// Basic usage with a [`Rect`](crate::util::linalg::Rect):
///
/// ```
/// use glongge::core::prelude::*;
///
/// let rect = Rect::new(
///     Vec2 { x: 1.0, y: 2.0 },   // center
///     Vec2 { x: 2.0, y: 3.0 }    // half-widths
/// );
///
/// assert_eq!(rect.centre(), Vec2 { x: 1.0, y: 2.0 });
/// assert_eq!(rect.extent(), Vec2 { x: 4.0, y: 6.0 });
/// assert_eq!(rect.top_left(), Vec2 { x: -1.0, y: -1.0 });
/// assert_eq!(rect.bottom_right(), Vec2 { x: 3.0, y: 5.0 });
///
/// // Test point containment
/// assert!(rect.contains_point(Vec2 { x: 0.0, y: 0.0 }));
/// assert!(!rect.contains_point(Vec2 { x: 4.0, y: 4.0 }));
/// ```
///
/// The trait is commonly used in collision detection and UI layout systems to work
/// with bounding boxes in a consistent way:
///
/// ```
/// use glongge::util::linalg::{Vec2, AxisAlignedExtent};
/// use std::sync::Arc;
///
/// // Function that works with any type implementing AxisAlignedExtent
/// fn is_visible<T: AxisAlignedExtent>(object: &T, viewport: &T) -> bool {
///     object.right() >= viewport.left() &&
///     object.left() <= viewport.right() &&
///     object.bottom() >= viewport.top() &&
///     object.top() <= viewport.bottom()
/// }
/// ```
pub trait AxisAlignedExtent {
    fn extent(&self) -> Vec2;
    fn centre(&self) -> Vec2;

    fn half_widths(&self) -> Vec2 {
        self.extent() / 2
    }
    fn top_left(&self) -> Vec2 {
        self.centre() - self.half_widths()
    }
    fn top_right(&self) -> Vec2 {
        self.top_left() + self.extent().project_x()
    }
    fn bottom_left(&self) -> Vec2 {
        self.top_left() + self.extent().project_y()
    }
    fn bottom_right(&self) -> Vec2 {
        self.top_left() + self.extent()
    }

    fn left(&self) -> f32 {
        self.top_left().x
    }
    fn right(&self) -> f32 {
        self.top_right().x
    }
    fn top(&self) -> f32 {
        self.top_left().y
    }
    fn bottom(&self) -> f32 {
        self.bottom_left().y
    }

    fn as_rect(&self) -> Rect {
        Rect::new(self.centre(), self.half_widths())
    }
    fn contains_point(&self, pos: Vec2) -> bool {
        (self.left()..self.right()).contains(&pos.x) && (self.top()..self.bottom()).contains(&pos.y)
    }
    fn contains_rect(&self, rect: &Rect) -> bool {
        self.left() <= rect.left()
            && self.right() >= rect.right()
            && self.top() <= rect.top()
            && self.bottom() >= rect.bottom()
    }

    fn union(&self, rhs: impl AxisAlignedExtent) -> Rect {
        self.as_rect().union(&rhs.as_rect())
    }
}

/// A rectangular shape defined by a center point and half-widths.
///
/// [`Rect`] represents a 2D axis-aligned rectangle by its center point (`centre`) and
/// half-widths (`half_widths`). It implements [`AxisAlignedExtent`] for convenient
/// queries about its bounds and dimensions.
///
/// # Examples
///
/// ```
/// use glongge::core::prelude::*;
///
/// let rect = Rect::new(
///     Vec2 { x: 0.0, y: 0.0 }, // Center point
///     Vec2 { x: 2.0, y: 1.5 }  // Half-widths
/// );
///
/// // Total width is twice the half-width
/// assert_eq!(rect.extent(), Vec2 { x: 4.0, y: 3.0 });
///
/// // Test if point is inside rectangle
/// assert!(rect.contains_point(Vec2 { x: 1.0, y: 0.5 }));
/// assert!(!rect.contains_point(Vec2 { x: 3.0, y: 2.0 }));
///
/// // Create from corner points
/// let rect = Rect::from_coords(
///     Vec2 { x: -1.0, y: -2.0 },  // Top-left
///     Vec2 { x: 3.0, y: 4.0 }     // Bottom-right
/// );
/// assert_eq!(rect.centre(), Vec2 { x: 1.0, y: 1.0 });
/// ```
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Rect {
    centre: Vec2,
    half_widths: Vec2,
}

impl Rect {
    /// Creates a new rectangle with the given center point and half-widths.
    pub fn new(centre: Vec2, half_widths: Vec2) -> Self {
        Self {
            centre,
            half_widths,
        }
    }
    /// Creates a new rectangle from two diagonal corner points.
    pub fn from_coords(top_left: Vec2, bottom_right: Vec2) -> Self {
        let half_widths = (bottom_right - top_left) / 2;
        let centre = top_left + half_widths;
        Self {
            centre,
            half_widths,
        }
    }
    /// Creates an empty rectangle with zero size at the origin.
    pub fn empty() -> Self {
        Self {
            centre: Vec2::zero(),
            half_widths: Vec2::zero(),
        }
    }
    pub fn unbounded() -> Self {
        Self {
            centre: Vec2::zero(),
            half_widths: Vec2::splat(f32::INFINITY),
        }
    }

    #[must_use]
    pub fn union(&self, rhs: &Rect) -> Rect {
        let top_left = self.top_left().min(rhs.top_left());
        let bottom_right = self.bottom_right().max(rhs.bottom_right());
        Self::from_coords(top_left, bottom_right)
    }

    #[must_use]
    pub fn with_centre(mut self, centre: Vec2) -> Rect {
        self.centre = centre;
        self
    }
}

impl Mul<f32> for Rect {
    type Output = Rect;

    fn mul(self, rhs: f32) -> Self::Output {
        Self::Output {
            centre: self.centre * rhs,
            half_widths: self.half_widths * rhs,
        }
    }
}
impl Div<f32> for Rect {
    type Output = Rect;

    fn div(self, rhs: f32) -> Self::Output {
        Self::Output {
            centre: self.centre / rhs,
            half_widths: self.half_widths / rhs,
        }
    }
}

impl AxisAlignedExtent for Rect {
    fn extent(&self) -> Vec2 {
        self.half_widths * 2.0
    }
    fn centre(&self) -> Vec2 {
        self.centre
    }
}

/// A 2D transformation consisting of translation, rotation, and scale.
///
/// The [`Transform`] represents a basic 2D transformation that can be applied to points
/// and vectors. It stores the position of an object's center, its rotation angle in radians,
/// and its scale factors along x and y axes.
///
/// This is commonly used in the game engine's scene graph system to represent object positions
/// and transformations hierarchically.
///
/// # Examples
///
/// ```
/// use glongge::core::prelude::*;
///
/// // Create a transform at a specific position
/// let transform = Transform::with_centre(Vec2 { x: 10.0, y: 20.0 });
/// assert_eq!(transform.centre, Vec2 { x: 10.0, y: 20.0 });
///
/// // Compose transformations
/// let translated = transform.translated(Vec2::right()); // Move right by 1 unit
/// assert_eq!(translated.centre, Vec2 { x: 11.0, y: 20.0 });
///
/// // Get local axis directions after rotation
/// let rotated = Transform::with_rotation(std::f32::consts::FRAC_PI_2); // 90 degrees
/// assert!(rotated.right().almost_eq(Vec2::down())); // Right becomes down
/// assert!(rotated.up().almost_eq(Vec2::right())); // Up becomes right
/// ```
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Transform {
    pub centre: Vec2,
    pub rotation: f32,
    pub scale: Vec2,
}

impl Transform {
    /// Creates a new transform at the specified center position.
    ///
    /// # Examples
    ///
    /// ```
    /// use glongge::core::prelude::*;
    ///
    /// let pos = Vec2 { x: 5.0, y: 10.0 };
    /// let transform = Transform::with_centre(pos);
    /// assert_eq!(transform.centre, pos);
    /// assert_eq!(transform.rotation, 0.0);
    /// assert_eq!(transform.scale, Vec2::one());
    /// ```
    #[must_use]
    pub fn with_centre(centre: Vec2) -> Self {
        Self {
            centre,
            ..Default::default()
        }
    }

    /// Creates a new transform with the specified rotation in radians.
    ///
    /// # Examples
    ///
    /// ```
    /// use glongge::core::prelude::*;
    ///
    /// let transform = Transform::with_rotation(std::f32::consts::PI); // 180 degrees
    /// assert_eq!(transform.centre, Vec2::zero());
    /// assert_eq!(transform.rotation, std::f32::consts::PI);
    /// assert!(transform.right().almost_eq(Vec2::left())); // Right becomes left after 180° rotation
    /// ```
    #[must_use]
    pub fn with_rotation(rotation: f32) -> Self {
        Self {
            rotation,
            ..Default::default()
        }
    }

    /// Creates a new transform with the specified scale factors.
    ///
    /// # Examples
    ///
    /// ```
    /// use glongge::core::prelude::*;
    ///
    /// let scale = Vec2 { x: 2.0, y: 0.5 };
    /// let transform = Transform::with_scale(scale);
    /// assert_eq!(transform.scale, scale);
    /// assert_eq!(transform.centre, Vec2::zero());
    /// assert_eq!(transform.rotation, 0.0);
    /// ```
    #[must_use]
    pub fn with_scale(scale: Vec2) -> Self {
        Self {
            scale,
            ..Default::default()
        }
    }

    /// Returns a new transform translated by the given offset vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use glongge::core::prelude::*;
    ///
    /// let transform = Transform::with_centre(Vec2::zero());
    /// let offset = Vec2 { x: 3.0, y: 4.0 };
    /// let moved = transform.translated(offset);
    /// assert_eq!(moved.centre, offset);
    /// assert_eq!(moved.rotation, transform.rotation);
    /// assert_eq!(moved.scale, transform.scale);
    /// ```
    #[must_use]
    pub fn translated(&self, by: Vec2) -> Self {
        Self {
            centre: self.centre + by,
            rotation: self.rotation,
            scale: self.scale,
        }
    }

    /// Returns a new transform that is the inverse of this transform.
    ///
    /// # Examples
    ///
    /// ```
    /// use glongge::core::prelude::*;
    ///
    /// let transform = Transform {
    ///     centre: Vec2 { x: 2.0, y: 3.0 },
    ///     rotation: std::f32::consts::FRAC_PI_2,
    ///     scale: Vec2 { x: 2.0, y: 2.0 },
    /// };
    /// let inverse = transform.inverse();
    /// assert_eq!(inverse.centre, -transform.centre);
    /// assert_eq!(inverse.rotation, -transform.rotation);
    /// assert_eq!(inverse.scale, transform.scale.reciprocal());
    /// ```
    #[must_use]
    pub fn inverse(&self) -> Self {
        Self {
            centre: -self.centre,
            rotation: -self.rotation,
            scale: self.scale.reciprocal(),
        }
    }

    /// Returns the left direction vector after applying the transform's rotation.
    ///
    /// This is equivalent to rotating the vector (-1, 0) by the transform's rotation angle.
    pub fn left(&self) -> Vec2 {
        Vec2::left().rotated(self.rotation)
    }

    /// Returns the up direction vector after applying the transform's rotation.
    ///
    /// This is equivalent to rotating the vector (0, -1) by the transform's rotation angle.
    pub fn up(&self) -> Vec2 {
        Vec2::up().rotated(self.rotation)
    }

    /// Returns the right direction vector after applying the transform's rotation.
    ///
    /// This is equivalent to rotating the vector (1, 0) by the transform's rotation angle.
    pub fn right(&self) -> Vec2 {
        Vec2::right().rotated(self.rotation)
    }

    /// Returns the down direction vector after applying the transform's rotation.
    ///
    /// This is equivalent to rotating the vector (0, 1) by the transform's rotation angle.
    pub fn down(&self) -> Vec2 {
        Vec2::down().rotated(self.rotation)
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            centre: Vec2::zero(),
            rotation: 0.0,
            scale: Vec2::one(),
        }
    }
}

impl Mul<Transform> for Transform {
    type Output = Transform;

    /// Multiplies two transforms together, analogous to matrix multiplication.
    ///
    /// This operation combines two transforms into a single transform that represents
    /// applying both transformations in sequence (right to left), with the following behavior:
    ///
    /// - Centre positions are added together
    /// - Rotation angles are added together
    /// - Scale factors are multiplied component-wise
    ///
    /// This matches how transformation matrices multiply together, but uses a more compact
    /// representation optimized for 2D transforms.
    fn mul(self, rhs: Transform) -> Self::Output {
        Self {
            centre: self.centre + rhs.centre,
            rotation: self.rotation + rhs.rotation,
            scale: self.scale.component_wise(rhs.scale),
        }
    }
}
impl MulAssign<Transform> for Transform {
    fn mul_assign(&mut self, rhs: Transform) {
        *self = *self * rhs;
    }
}
/// A linear interpolation between two values.
///
/// # Examples
/// ```
/// use glongge::core::prelude::*;
/// let start = 0.0;
/// let end = 10.0;
/// assert_eq!(linalg::lerp(start, end, 0.0), start);
/// assert_eq!(linalg::lerp(start, end, 1.0), end);
/// assert_eq!(linalg::lerp(start, end, 0.5), 5.0);
/// ```
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

/// An exponential interpolation between two values, useful for transitions that follow a
/// multiplicative (percentage-based) scale rather than a linear (additive) scale.
///
/// Useful for things like zooming or scaling objects.
///
/// # Examples
/// ```
/// use glongge::core::prelude::*;
/// // Maintains multiplicative linearity between interpolation points; every step of
/// // constant size multiplies the value by a constant amount.
/// assert_eq!(linalg::eerp(1.0, 16.0, 0.25), 2.0);
/// assert_eq!(linalg::eerp(1.0, 16.0, 0.5), 4.0);
/// assert_eq!(linalg::eerp(1.0, 16.0, 0.75), 8.0);
/// assert_eq!(linalg::eerp(1.0, 16.0, 0.1), 1.319508);
/// assert_eq!(linalg::eerp(1.0, 16.0, 0.2), 1.7411011); // = 1.319508 * 1.319508
/// assert_eq!(linalg::eerp(1.0, 16.0, 0.3), 2.297397); // = 1.319508 * 1.7411011
/// // The interpolation reaches the same endpoints as lerp.
/// let start = 1.0;
/// let end = 16.0;
/// assert_eq!(linalg::eerp(start, end, 0.0), start);
/// assert_eq!(linalg::eerp(start, end, 1.0), end);
/// ```
pub fn eerp(a: f32, b: f32, t: f32) -> f32 {
    a * (t * (b / a).ln()).exp()
}

/// A smoothstep function that creates a smooth transition between 0 and 1.
/// Uses a 5th order polynomial for a smooth acceleration and deceleration curve.
///
/// # Examples
/// ```
/// use glongge::core::prelude::*;
/// assert_eq!(linalg::smooth(0.0), 0.0); // Start of transition
/// assert_eq!(linalg::smooth(1.0), 1.0); // End of transition
/// assert_eq!(linalg::smooth(0.5), 0.5); // Midpoint
/// // Output is always clamped between 0 and 1
/// assert_eq!(linalg::smooth(-1.0), 0.0);
/// assert_eq!(linalg::smooth(2.0), 1.0);
/// ```
pub fn smooth(t: f32) -> f32 {
    (6.0 * t * t * t * t * t - 15.0 * t * t * t * t + 10.0 * t * t * t).clamp(0.0, 1.0)
}

/// A sigmoid function that creates an S-shaped curve, useful for smooth transitions.
/// The 'k' parameter controls the steepness of the curve at the midpoint.
///
/// # Examples
/// ```
/// use glongge::core::prelude::*;
/// let k = 0.1; // Steepness factor
/// // Approaches 0 and 1 asymptotically
/// assert!(linalg::sigmoid(0.0, k) < 0.1);
/// assert!(linalg::sigmoid(1.0, k) > 0.9);
/// // Centered at 0.5
/// assert!((linalg::sigmoid(0.5, k) - 0.5).abs() < EPSILON);
/// ```
pub fn sigmoid(t: f32, k: f32) -> f32 {
    1.0 / (1.0 + (-(t - 0.5) / k).exp())
}
