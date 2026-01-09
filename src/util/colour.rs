use anyhow::Result;
use num_traits::{FromPrimitive, PrimInt, ToPrimitive};
use std::ops::Mul;

#[derive(Copy, Clone, Debug, PartialEq, bincode::Encode, bincode::Decode)]
pub struct Colour {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Colour {
    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }
    pub fn from_bytes(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self {
            r: f32::from(r) / 255.0,
            g: f32::from(g) / 255.0,
            b: f32::from(b) / 255.0,
            a: f32::from(a) / 255.0,
        }
    }
    pub fn from_bytes_clamp<I: PrimInt + FromPrimitive + ToPrimitive>(
        r: I,
        g: I,
        b: I,
        a: I,
    ) -> Self {
        let min = I::from_u8(u8::MIN).expect("weird conversion failure from u8");
        let max = I::from_u8(u8::MAX).expect("weird conversion failure from u8");
        Self::from_bytes(
            r.clamp(min, max)
                .to_u8()
                .expect("weird conversion failure to u8"),
            g.clamp(min, max)
                .to_u8()
                .expect("weird conversion failure to u8"),
            b.clamp(min, max)
                .to_u8()
                .expect("weird conversion failure to u8"),
            a.clamp(min, max)
                .to_u8()
                .expect("weird conversion failure to u8"),
        )
    }

    pub fn from_hex_rgb(hex: &str) -> Result<Self> {
        let r = u8::from_str_radix(&hex[..2], 16)?;
        let g = u8::from_str_radix(&hex[2..4], 16)?;
        let b = u8::from_str_radix(&hex[4..], 16)?;
        Ok(Self::from_bytes(r, g, b, 255))
    }

    pub fn red() -> Self {
        Self {
            r: 1.0,
            ..Default::default()
        }
    }
    pub fn green() -> Self {
        Self {
            g: 1.0,
            ..Default::default()
        }
    }
    pub fn blue() -> Self {
        Self {
            b: 1.0,
            ..Default::default()
        }
    }
    pub fn yellow() -> Self {
        Self {
            r: 1.0,
            g: 1.0,
            ..Default::default()
        }
    }
    pub fn magenta() -> Self {
        Self {
            r: 1.0,
            b: 1.0,
            ..Default::default()
        }
    }
    pub fn cyan() -> Self {
        Self {
            g: 1.0,
            b: 1.0,
            ..Default::default()
        }
    }
    pub fn black() -> Self {
        Self::default()
    }
    pub fn white() -> Self {
        Self {
            r: 1.0,
            g: 1.0,
            b: 1.0,
            a: 1.0,
        }
    }
    pub fn empty() -> Self {
        Self {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 0.0,
        }
    }

    #[must_use]
    pub fn scaled(mut self, ratio: f32) -> Self {
        self.r *= ratio;
        self.g *= ratio;
        self.b *= ratio;
        self.r = self.r.clamp(0.0, 1.0);
        self.g = self.g.clamp(0.0, 1.0);
        self.b = self.b.clamp(0.0, 1.0);
        self
    }
    #[must_use]
    pub fn with_alpha(mut self, a: f32) -> Self {
        self.a = a;
        self
    }

    pub fn as_bytes(&self) -> [u8; 4] {
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        [
            (self.r * 255.0) as u8,
            (self.g * 255.0) as u8,
            (self.b * 255.0) as u8,
            (self.a * 255.0) as u8,
        ]
    }
    pub fn as_f32(&self) -> [f32; 4] {
        self.into()
    }
}

impl Default for Colour {
    fn default() -> Self {
        Self {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 1.0,
        }
    }
}

impl From<Colour> for [f32; 4] {
    fn from(value: Colour) -> Self {
        [value.r, value.g, value.b, value.a]
    }
}

impl From<&Colour> for [f32; 4] {
    fn from(value: &Colour) -> Self {
        (*value).into()
    }
}

impl Mul for Colour {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            r: self.r * rhs.r,
            g: self.g * rhs.g,
            b: self.b * rhs.b,
            a: self.a * rhs.a,
        }
    }
}

pub mod gg_col {
    use crate::core::prelude::*;
    use std::collections::BTreeSet;

    #[allow(clippy::cast_possible_wrap)]
    pub fn flood_fill(pixels: &[Vec<Colour>], start: Vec2i, tolerance: i32) -> BTreeSet<Vec2i> {
        let height = pixels.len() as i32;
        let width = pixels.first().map_or(0, |row| row.len() as i32);

        let mut rv = BTreeSet::new();
        rv.insert(start);
        let mut stack = vec![start];
        while let Some(next) = stack.pop() {
            for i in 0..=tolerance {
                let i = i + 1;
                let mut candidates = vec![
                    next + i * Vec2i::right(),
                    next + i * Vec2i::up(),
                    next + i * Vec2i::left(),
                    next + i * Vec2i::down(),
                ];
                if tolerance > 0 {
                    candidates.push(next + Vec2i { x: i, y: i });
                    candidates.push(next + Vec2i { x: i, y: -i });
                    candidates.push(next + Vec2i { x: -i, y: i });
                    candidates.push(next + Vec2i { x: -i, y: -i });
                }
                candidates.retain(|p| p.x >= 0 && p.x < width && p.y >= 0 && p.y < height);
                for candidate in candidates {
                    if pixels[candidate.y as usize][candidate.x as usize].a > 0.0
                        && !rv.contains(&candidate)
                    {
                        rv.insert(candidate);
                        stack.push(candidate);
                    }
                }
            }
        }
        rv
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let c = Colour::new(0.1, 0.2, 0.3, 0.4);
        assert_eq!(c.r, 0.1);
        assert_eq!(c.g, 0.2);
        assert_eq!(c.b, 0.3);
        assert_eq!(c.a, 0.4);
    }

    #[test]
    fn test_from_bytes() {
        let c = Colour::from_bytes(0, 127, 255, 255);
        assert_eq!(c.r, 0.0);
        assert!((c.g - 127.0 / 255.0).abs() < f32::EPSILON);
        assert_eq!(c.b, 1.0);
        assert_eq!(c.a, 1.0);
    }

    #[test]
    fn test_from_bytes_clamp() {
        // Values within range
        let c = Colour::from_bytes_clamp(100i32, 150i32, 200i32, 255i32);
        assert!((c.r - 100.0 / 255.0).abs() < f32::EPSILON);
        assert!((c.g - 150.0 / 255.0).abs() < f32::EPSILON);
        assert!((c.b - 200.0 / 255.0).abs() < f32::EPSILON);

        // Values out of range (should clamp)
        let c = Colour::from_bytes_clamp(-50i32, 300i32, 128i32, 1000i32);
        assert_eq!(c.r, 0.0); // clamped from -50
        assert_eq!(c.b, 128.0 / 255.0);
        assert_eq!(c.g, 1.0); // clamped from 300
        assert_eq!(c.a, 1.0); // clamped from 1000
    }

    #[test]
    fn test_from_hex_rgb() {
        let c = Colour::from_hex_rgb("FF0000").unwrap();
        assert_eq!(c.r, 1.0);
        assert_eq!(c.g, 0.0);
        assert_eq!(c.b, 0.0);
        assert_eq!(c.a, 1.0);

        let c = Colour::from_hex_rgb("00FF00").unwrap();
        assert_eq!(c.r, 0.0);
        assert_eq!(c.g, 1.0);
        assert_eq!(c.b, 0.0);

        let c = Colour::from_hex_rgb("0000FF").unwrap();
        assert_eq!(c.r, 0.0);
        assert_eq!(c.g, 0.0);
        assert_eq!(c.b, 1.0);

        let c = Colour::from_hex_rgb("808080").unwrap();
        assert!((c.r - 128.0 / 255.0).abs() < f32::EPSILON);
        assert!((c.g - 128.0 / 255.0).abs() < f32::EPSILON);
        assert!((c.b - 128.0 / 255.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_from_hex_rgb_invalid() {
        // Invalid in first two characters
        assert!(Colour::from_hex_rgb("GGGGGG").is_err());
        assert!(Colour::from_hex_rgb("ZZ0000").is_err());
        // Invalid in middle two characters (valid first, invalid middle)
        assert!(Colour::from_hex_rgb("00GG00").is_err());
        // Invalid in last two characters (valid first four, invalid last)
        assert!(Colour::from_hex_rgb("0000GG").is_err());
    }

    #[test]
    fn test_copy() {
        let a = Colour::red();
        let b = a; // Copy
        let c = a; // Can still use a because it's Copy
        assert_eq!(b, c);
    }

    #[test]
    fn test_partial_eq_ne() {
        let a = Colour::red();
        let b = Colour::blue();
        assert!(a != b);
        assert!(!(a != a));
    }

    #[test]
    fn test_named_colors() {
        assert_eq!(Colour::red(), Colour::new(1.0, 0.0, 0.0, 1.0));
        assert_eq!(Colour::green(), Colour::new(0.0, 1.0, 0.0, 1.0));
        assert_eq!(Colour::blue(), Colour::new(0.0, 0.0, 1.0, 1.0));
        assert_eq!(Colour::yellow(), Colour::new(1.0, 1.0, 0.0, 1.0));
        assert_eq!(Colour::magenta(), Colour::new(1.0, 0.0, 1.0, 1.0));
        assert_eq!(Colour::cyan(), Colour::new(0.0, 1.0, 1.0, 1.0));
        assert_eq!(Colour::black(), Colour::new(0.0, 0.0, 0.0, 1.0));
        assert_eq!(Colour::white(), Colour::new(1.0, 1.0, 1.0, 1.0));
        assert_eq!(Colour::empty(), Colour::new(0.0, 0.0, 0.0, 0.0));
    }

    #[test]
    fn test_default() {
        let c = Colour::default();
        assert_eq!(c.r, 0.0);
        assert_eq!(c.g, 0.0);
        assert_eq!(c.b, 0.0);
        assert_eq!(c.a, 1.0);
    }

    #[test]
    fn test_scaled() {
        let c = Colour::white().scaled(0.5);
        assert_eq!(c.r, 0.5);
        assert_eq!(c.g, 0.5);
        assert_eq!(c.b, 0.5);
        assert_eq!(c.a, 1.0); // alpha unchanged

        // Scaling above 1.0 should clamp
        let c = Colour::new(0.5, 0.5, 0.5, 1.0).scaled(3.0);
        assert_eq!(c.r, 1.0);
        assert_eq!(c.g, 1.0);
        assert_eq!(c.b, 1.0);

        // Negative scaling should clamp to 0
        let c = Colour::white().scaled(-1.0);
        assert_eq!(c.r, 0.0);
        assert_eq!(c.g, 0.0);
        assert_eq!(c.b, 0.0);
    }

    #[test]
    fn test_with_alpha() {
        let c = Colour::red().with_alpha(0.5);
        assert_eq!(c.r, 1.0);
        assert_eq!(c.a, 0.5);
    }

    #[test]
    fn test_as_bytes() {
        let c = Colour::new(1.0, 0.5, 0.0, 1.0);
        let bytes = c.as_bytes();
        assert_eq!(bytes[0], 255);
        assert_eq!(bytes[1], 127);
        assert_eq!(bytes[2], 0);
        assert_eq!(bytes[3], 255);
    }

    #[test]
    fn test_as_f32() {
        let c = Colour::new(0.1, 0.2, 0.3, 0.4);
        let arr = c.as_f32();
        assert_eq!(arr, [0.1, 0.2, 0.3, 0.4]);
    }

    #[test]
    fn test_into_f32_array() {
        let c = Colour::new(0.1, 0.2, 0.3, 0.4);
        let arr: [f32; 4] = c.into();
        assert_eq!(arr, [0.1, 0.2, 0.3, 0.4]);

        let c = Colour::new(0.5, 0.6, 0.7, 0.8);
        let arr: [f32; 4] = (&c).into();
        assert_eq!(arr, [0.5, 0.6, 0.7, 0.8]);
    }

    #[test]
    fn test_mul() {
        let a = Colour::new(1.0, 0.5, 0.25, 1.0);
        let b = Colour::new(0.5, 0.5, 0.5, 0.5);
        let c = a * b;
        assert_eq!(c.r, 0.5);
        assert_eq!(c.g, 0.25);
        assert_eq!(c.b, 0.125);
        assert_eq!(c.a, 0.5);
    }

    #[test]
    fn test_roundtrip_bytes() {
        let original = Colour::from_bytes(100, 150, 200, 255);
        let bytes = original.as_bytes();
        let restored = Colour::from_bytes(bytes[0], bytes[1], bytes[2], bytes[3]);
        assert!((original.r - restored.r).abs() < f32::EPSILON);
        assert!((original.g - restored.g).abs() < f32::EPSILON);
        assert!((original.b - restored.b).abs() < f32::EPSILON);
        assert!((original.a - restored.a).abs() < f32::EPSILON);
    }

    #[test]
    #[allow(clippy::clone_on_copy)]
    fn test_clone() {
        let original = Colour::new(0.1, 0.2, 0.3, 0.4);
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    #[test]
    fn test_debug() {
        let c = Colour::new(1.0, 0.0, 0.0, 1.0);
        let debug_str = format!("{c:?}");
        assert!(debug_str.contains("Colour"));
        assert!(debug_str.contains("r:"));
    }

    #[test]
    fn test_bincode() {
        let c = Colour::new(0.25, 0.5, 0.75, 1.0);
        let encoded = bincode::encode_to_vec(c, bincode::config::standard()).unwrap();
        let (decoded, _): (Colour, _) =
            bincode::decode_from_slice(&encoded, bincode::config::standard()).unwrap();
        let (borrowed, _): (Colour, _) =
            bincode::borrow_decode_from_slice(&encoded, bincode::config::standard()).unwrap();
        assert_eq!(c, decoded);
        assert_eq!(c, borrowed);

        crate::util::test_util::test_bincode_error_paths::<Colour>();
    }
}

#[cfg(test)]
mod flood_fill_tests {
    use super::Colour;
    use super::gg_col::flood_fill;
    use crate::util::linalg::Vec2i;

    fn make_grid(width: usize, height: usize, filled: &[(usize, usize)]) -> Vec<Vec<Colour>> {
        let mut grid = vec![vec![Colour::empty(); width]; height];
        for &(x, y) in filled {
            grid[y][x] = Colour::white();
        }
        grid
    }

    #[test]
    fn test_flood_fill_single_pixel() {
        // 3x3 grid with only center pixel filled
        let grid = make_grid(3, 3, &[(1, 1)]);
        let result = flood_fill(&grid, Vec2i { x: 1, y: 1 }, 0);
        assert_eq!(result.len(), 1);
        assert!(result.contains(&Vec2i { x: 1, y: 1 }));
    }

    #[test]
    fn test_flood_fill_horizontal_line() {
        // 5x3 grid with horizontal line in middle row
        let grid = make_grid(5, 3, &[(1, 1), (2, 1), (3, 1)]);
        let result = flood_fill(&grid, Vec2i { x: 2, y: 1 }, 0);
        assert_eq!(result.len(), 3);
        assert!(result.contains(&Vec2i { x: 1, y: 1 }));
        assert!(result.contains(&Vec2i { x: 2, y: 1 }));
        assert!(result.contains(&Vec2i { x: 3, y: 1 }));
    }

    #[test]
    fn test_flood_fill_vertical_line() {
        // 3x5 grid with vertical line in middle column
        let grid = make_grid(3, 5, &[(1, 1), (1, 2), (1, 3)]);
        let result = flood_fill(&grid, Vec2i { x: 1, y: 2 }, 0);
        assert_eq!(result.len(), 3);
        assert!(result.contains(&Vec2i { x: 1, y: 1 }));
        assert!(result.contains(&Vec2i { x: 1, y: 2 }));
        assert!(result.contains(&Vec2i { x: 1, y: 3 }));
    }

    #[test]
    fn test_flood_fill_square() {
        // 5x5 grid with 3x3 filled square in center
        let grid = make_grid(
            5,
            5,
            &[
                (1, 1),
                (2, 1),
                (3, 1),
                (1, 2),
                (2, 2),
                (3, 2),
                (1, 3),
                (2, 3),
                (3, 3),
            ],
        );
        let result = flood_fill(&grid, Vec2i { x: 2, y: 2 }, 0);
        assert_eq!(result.len(), 9);
    }

    #[test]
    fn test_flood_fill_with_tolerance() {
        // 5x5 grid with diagonal pixels - tolerance allows diagonal connections
        let grid = make_grid(5, 5, &[(1, 1), (2, 2), (3, 3)]);
        let result = flood_fill(&grid, Vec2i { x: 2, y: 2 }, 1);
        assert_eq!(result.len(), 3);
        assert!(result.contains(&Vec2i { x: 1, y: 1 }));
        assert!(result.contains(&Vec2i { x: 2, y: 2 }));
        assert!(result.contains(&Vec2i { x: 3, y: 3 }));
    }

    #[test]
    fn test_flood_fill_disconnected_regions() {
        // Two separate filled regions - should only fill one
        let grid = make_grid(7, 3, &[(1, 1), (5, 1)]);
        let result = flood_fill(&grid, Vec2i { x: 1, y: 1 }, 0);
        assert_eq!(result.len(), 1);
        assert!(result.contains(&Vec2i { x: 1, y: 1 }));
        assert!(!result.contains(&Vec2i { x: 5, y: 1 }));
    }
}
