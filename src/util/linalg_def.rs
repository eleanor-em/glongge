//! Separate module to prevent bincode derive macros from screwing up test coverage.

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
#[derive(Default, Debug, Copy, Clone, bincode::Encode, bincode::Decode)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

#[derive(
    Default,
    Debug,
    Eq,
    PartialEq,
    Ord,
    PartialOrd,
    Copy,
    Clone,
    Hash,
    bincode::Encode,
    bincode::Decode,
)]
pub struct Vec2i {
    pub x: i32,
    pub y: i32,
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
    Default,
    Debug,
    Eq,
    PartialEq,
    Ord,
    PartialOrd,
    Copy,
    Clone,
    Hash,
    bincode::Encode,
    bincode::Decode,
)]
pub struct Edge2i(pub Vec2i, pub Vec2i);

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
#[derive(Debug, Copy, Clone, PartialEq, bincode::Encode, bincode::Decode)]
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
#[derive(
    Copy,
    Clone,
    Debug,
    Default,
    Eq,
    PartialEq,
    Ord,
    PartialOrd,
    Hash,
    bincode::Encode,
    bincode::Decode,
)]
pub struct Rect {
    pub(crate) centre: Vec2,
    pub(crate) half_widths: Vec2,
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
#[derive(Copy, Clone, Debug, PartialEq, bincode::Encode, bincode::Decode)]
pub struct Transform {
    pub centre: Vec2,
    pub rotation: f32,
    pub scale: Vec2,
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use crate::core::prelude::{Mat3x3, Rect, Transform, Vec2, Vec2i};
    use crate::util::linalg::Edge2i;
    use std::f32::consts::{FRAC_PI_4, FRAC_PI_6};

    #[test]
    #[allow(clippy::too_many_lines)]
    fn bincode_serialization() {
        // Test bincode serialization/deserialization for types with bincode derives
        let config = bincode::config::standard();

        let v = Vec2 { x: 1.0, y: 2.0 };
        let encoded = bincode::encode_to_vec(v, config).unwrap();
        let (decoded, _): (Vec2, _) = bincode::decode_from_slice(&encoded, config).unwrap();
        assert_eq!(v, decoded);

        let v_inf = Vec2 {
            x: f32::INFINITY,
            y: f32::NEG_INFINITY,
        };
        let encoded = bincode::encode_to_vec(v_inf, config).unwrap();
        let (decoded, _): (Vec2, _) = bincode::decode_from_slice(&encoded, config).unwrap();
        assert_eq!(v_inf, decoded);

        let v_nan = Vec2 {
            x: f32::NAN,
            y: f32::NAN,
        };
        let encoded = bincode::encode_to_vec(v_nan, config).unwrap();
        let (decoded, _): (Vec2, _) = bincode::decode_from_slice(&encoded, config).unwrap();
        assert!(decoded.x.is_nan() && decoded.y.is_nan());

        let v_inf_nan = Vec2 {
            x: f32::INFINITY,
            y: f32::NAN,
        };
        let encoded = bincode::encode_to_vec(v_inf_nan, config).unwrap();
        let (decoded, _): (Vec2, _) = bincode::decode_from_slice(&encoded, config).unwrap();
        assert_eq!(decoded.x, f32::INFINITY);
        assert!(decoded.y.is_nan());

        let v_neg_zero_inf = Vec2 {
            x: -0.0,
            y: f32::INFINITY,
        };
        let encoded = bincode::encode_to_vec(v_neg_zero_inf, config).unwrap();
        let (decoded, _): (Vec2, _) = bincode::decode_from_slice(&encoded, config).unwrap();
        assert!(decoded.x.is_sign_negative() && decoded.x == 0.0);
        assert_eq!(decoded.y, f32::INFINITY);

        let subnormal = f32::MIN_POSITIVE / 2.0;
        assert!(subnormal.is_subnormal());
        let v_subnormal = Vec2 {
            x: subnormal,
            y: -subnormal,
        };
        let encoded = bincode::encode_to_vec(v_subnormal, config).unwrap();
        let (decoded, _): (Vec2, _) = bincode::decode_from_slice(&encoded, config).unwrap();
        assert_eq!(v_subnormal, decoded);

        // Smallest and largest subnormals
        let smallest_subnormal = f32::from_bits(1);
        let largest_subnormal = f32::from_bits(0x007F_FFFF);
        assert!(smallest_subnormal.is_subnormal());
        assert!(largest_subnormal.is_subnormal());
        let v_subnormal_range = Vec2 {
            x: smallest_subnormal,
            y: largest_subnormal,
        };
        let encoded = bincode::encode_to_vec(v_subnormal_range, config).unwrap();
        let (decoded, _): (Vec2, _) = bincode::decode_from_slice(&encoded, config).unwrap();
        assert_eq!(v_subnormal_range, decoded);

        // f32::MAX and f32::MIN
        let v_extremes = Vec2 {
            x: f32::MAX,
            y: f32::MIN,
        };
        let encoded = bincode::encode_to_vec(v_extremes, config).unwrap();
        let (decoded, _): (Vec2, _) = bincode::decode_from_slice(&encoded, config).unwrap();
        assert_eq!(v_extremes, decoded);

        // Negative NaN (NaN with sign bit set) - verify bit pattern preserved
        let neg_nan = f32::from_bits(0xFF80_0001);
        assert!(neg_nan.is_nan());
        let v_neg_nan = Vec2 { x: neg_nan, y: 1.0 };
        let encoded = bincode::encode_to_vec(v_neg_nan, config).unwrap();
        let (decoded, _): (Vec2, _) = bincode::decode_from_slice(&encoded, config).unwrap();
        assert_eq!(decoded.x.to_bits(), neg_nan.to_bits());
        assert_eq!(decoded.y, 1.0);

        // Specific NaN payload preservation
        let nan_with_payload = f32::from_bits(0x7F80_ABCD);
        assert!(nan_with_payload.is_nan());
        let v_nan_payload = Vec2 {
            x: nan_with_payload,
            y: 0.0,
        };
        let encoded = bincode::encode_to_vec(v_nan_payload, config).unwrap();
        let (decoded, _): (Vec2, _) = bincode::decode_from_slice(&encoded, config).unwrap();
        assert_eq!(decoded.x.to_bits(), nan_with_payload.to_bits());

        let vi = Vec2i { x: 3, y: 4 };
        let encoded = bincode::encode_to_vec(vi, config).unwrap();
        let (decoded, _): (Vec2i, _) = bincode::decode_from_slice(&encoded, config).unwrap();
        assert_eq!(vi, decoded);

        // Vec2i with extreme integer values
        let vi_extreme = Vec2i {
            x: i32::MAX,
            y: i32::MIN,
        };
        let encoded = bincode::encode_to_vec(vi_extreme, config).unwrap();
        let (decoded, _): (Vec2i, _) = bincode::decode_from_slice(&encoded, config).unwrap();
        assert_eq!(vi_extreme, decoded);

        let edge = Edge2i(Vec2i { x: 0, y: 0 }, Vec2i { x: 1, y: 1 });
        let encoded = bincode::encode_to_vec(edge, config).unwrap();
        let (decoded, _): (Edge2i, _) = bincode::decode_from_slice(&encoded, config).unwrap();
        assert_eq!(edge, decoded);

        // Edge2i with extreme values
        let edge_extreme = Edge2i(
            Vec2i {
                x: i32::MIN,
                y: i32::MAX,
            },
            Vec2i {
                x: i32::MAX,
                y: i32::MIN,
            },
        );
        let encoded = bincode::encode_to_vec(edge_extreme, config).unwrap();
        let (decoded, _): (Edge2i, _) = bincode::decode_from_slice(&encoded, config).unwrap();
        assert_eq!(edge_extreme, decoded);

        let m = Mat3x3::rotation(FRAC_PI_6);
        let encoded = bincode::encode_to_vec(m, config).unwrap();
        let (decoded, _): (Mat3x3, _) = bincode::decode_from_slice(&encoded, config).unwrap();
        assert_eq!(m, decoded);

        let r = Rect::new(Vec2 { x: 1.0, y: 2.0 }, Vec2 { x: 3.0, y: 4.0 });
        let encoded = bincode::encode_to_vec(r, config).unwrap();
        let (decoded, _): (Rect, _) = bincode::decode_from_slice(&encoded, config).unwrap();
        assert_eq!(r, decoded);

        // Rect with special float values
        let r_special = Rect::new(
            Vec2 {
                x: f32::INFINITY,
                y: f32::NEG_INFINITY,
            },
            Vec2 {
                x: f32::MAX,
                y: f32::MIN_POSITIVE,
            },
        );
        let encoded = bincode::encode_to_vec(r_special, config).unwrap();
        let (decoded, _): (Rect, _) = bincode::decode_from_slice(&encoded, config).unwrap();
        assert_eq!(r_special, decoded);

        let t = Transform {
            centre: Vec2 { x: 5.0, y: 6.0 },
            rotation: FRAC_PI_4,
            scale: Vec2 { x: 2.0, y: 3.0 },
        };
        let encoded = bincode::encode_to_vec(t, config).unwrap();
        let (decoded, _): (Transform, _) = bincode::decode_from_slice(&encoded, config).unwrap();
        assert_eq!(t.centre, decoded.centre);
        assert_eq!(t.rotation, decoded.rotation);
        assert_eq!(t.scale, decoded.scale);

        // Test borrow_decode path for all types (exercises BorrowDecode derive)
        let v = Vec2 { x: 1.0, y: 2.0 };
        let encoded = bincode::encode_to_vec(v, config).unwrap();
        let (decoded, _): (Vec2, _) = bincode::borrow_decode_from_slice(&encoded, config).unwrap();
        assert_eq!(v, decoded);

        let vi = Vec2i { x: 1, y: 2 };
        let encoded = bincode::encode_to_vec(vi, config).unwrap();
        let (decoded, _): (Vec2i, _) = bincode::borrow_decode_from_slice(&encoded, config).unwrap();
        assert_eq!(vi, decoded);

        let edge = Edge2i(Vec2i { x: 1, y: 2 }, Vec2i { x: 3, y: 4 });
        let encoded = bincode::encode_to_vec(edge, config).unwrap();
        let (decoded, _): (Edge2i, _) =
            bincode::borrow_decode_from_slice(&encoded, config).unwrap();
        assert_eq!(edge, decoded);

        let m = Mat3x3::rotation(FRAC_PI_6);
        let encoded = bincode::encode_to_vec(m, config).unwrap();
        let (decoded, _): (Mat3x3, _) =
            bincode::borrow_decode_from_slice(&encoded, config).unwrap();
        assert_eq!(m, decoded);

        let r = Rect::new(Vec2 { x: 1.0, y: 2.0 }, Vec2 { x: 3.0, y: 4.0 });
        let encoded = bincode::encode_to_vec(r, config).unwrap();
        let (decoded, _): (Rect, _) = bincode::borrow_decode_from_slice(&encoded, config).unwrap();
        assert_eq!(r, decoded);

        let t = Transform::default();
        let encoded = bincode::encode_to_vec(t, config).unwrap();
        let (decoded, _): (Transform, _) =
            bincode::borrow_decode_from_slice(&encoded, config).unwrap();
        assert_eq!(t, decoded);
    }

    #[test]
    fn bincode_encode_into_slice() {
        // Test encode_into_slice path with standard config
        let config = bincode::config::standard();
        let mut buf = [0u8; 128];

        let v = Vec2 { x: 1.0, y: 2.0 };
        let len = bincode::encode_into_slice(v, &mut buf, config).unwrap();
        let (decoded, _): (Vec2, _) = bincode::decode_from_slice(&buf[..len], config).unwrap();
        assert_eq!(v, decoded);

        let vi = Vec2i { x: 1, y: 2 };
        let len = bincode::encode_into_slice(vi, &mut buf, config).unwrap();
        let (decoded, _): (Vec2i, _) = bincode::decode_from_slice(&buf[..len], config).unwrap();
        assert_eq!(vi, decoded);

        let edge = Edge2i(Vec2i { x: 1, y: 2 }, Vec2i { x: 3, y: 4 });
        let len = bincode::encode_into_slice(edge, &mut buf, config).unwrap();
        let (decoded, _): (Edge2i, _) = bincode::decode_from_slice(&buf[..len], config).unwrap();
        assert_eq!(edge, decoded);

        let m = Mat3x3::rotation(FRAC_PI_6);
        let len = bincode::encode_into_slice(m, &mut buf, config).unwrap();
        let (decoded, _): (Mat3x3, _) = bincode::decode_from_slice(&buf[..len], config).unwrap();
        assert_eq!(m, decoded);

        let r = Rect::new(Vec2 { x: 1.0, y: 2.0 }, Vec2 { x: 3.0, y: 4.0 });
        let len = bincode::encode_into_slice(r, &mut buf, config).unwrap();
        let (decoded, _): (Rect, _) = bincode::decode_from_slice(&buf[..len], config).unwrap();
        assert_eq!(r, decoded);

        let t = Transform::default();
        let len = bincode::encode_into_slice(t, &mut buf, config).unwrap();
        let (decoded, _): (Transform, _) = bincode::decode_from_slice(&buf[..len], config).unwrap();
        assert_eq!(t, decoded);
    }

    #[test]
    fn bincode_legacy_config() {
        // Test with legacy config (different endianness and int encoding)
        let config = bincode::config::legacy();
        let mut buf = [0u8; 128];

        let v = Vec2 { x: 1.0, y: 2.0 };
        let encoded = bincode::encode_to_vec(v, config).unwrap();
        let (decoded, _): (Vec2, _) = bincode::decode_from_slice(&encoded, config).unwrap();
        assert_eq!(v, decoded);
        let (decoded, _): (Vec2, _) = bincode::borrow_decode_from_slice(&encoded, config).unwrap();
        assert_eq!(v, decoded);
        let len = bincode::encode_into_slice(v, &mut buf, config).unwrap();
        let (decoded, _): (Vec2, _) = bincode::decode_from_slice(&buf[..len], config).unwrap();
        assert_eq!(v, decoded);

        let vi = Vec2i { x: 1, y: 2 };
        let encoded = bincode::encode_to_vec(vi, config).unwrap();
        let (decoded, _): (Vec2i, _) = bincode::decode_from_slice(&encoded, config).unwrap();
        assert_eq!(vi, decoded);

        let edge = Edge2i(Vec2i { x: 1, y: 2 }, Vec2i { x: 3, y: 4 });
        let encoded = bincode::encode_to_vec(edge, config).unwrap();
        let (decoded, _): (Edge2i, _) = bincode::decode_from_slice(&encoded, config).unwrap();
        assert_eq!(edge, decoded);

        let m = Mat3x3::rotation(FRAC_PI_6);
        let encoded = bincode::encode_to_vec(m, config).unwrap();
        let (decoded, _): (Mat3x3, _) = bincode::decode_from_slice(&encoded, config).unwrap();
        assert_eq!(m, decoded);

        let r = Rect::new(Vec2 { x: 1.0, y: 2.0 }, Vec2 { x: 3.0, y: 4.0 });
        let encoded = bincode::encode_to_vec(r, config).unwrap();
        let (decoded, _): (Rect, _) = bincode::decode_from_slice(&encoded, config).unwrap();
        assert_eq!(r, decoded);

        let t = Transform::default();
        let encoded = bincode::encode_to_vec(t, config).unwrap();
        let (decoded, _): (Transform, _) = bincode::decode_from_slice(&encoded, config).unwrap();
        assert_eq!(t, decoded);
    }
}
