use num_traits::{FromPrimitive, PrimInt, ToPrimitive};
use std::ops::Mul;

#[derive(Copy, Clone, Debug, PartialEq)]
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
