#[allow(unused_imports)]
use crate::core::prelude::*;

use num_traits::{FromPrimitive, PrimInt, ToPrimitive};

#[derive(Copy, Clone)]
pub struct Colour {
    pub r: f64,
    pub g: f64,
    pub b: f64,
    pub a: f64,
}

impl Colour {
    pub fn new(r: f64, g: f64, b: f64, a: f64) -> Self {
        Self { r, g, b, a }
    }
    pub fn from_bytes(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r: r as f64 / 255., g: g as f64 / 255., b: b as f64 / 255., a: a as f64 / 255. }
    }
    pub fn from_bytes_clamp<I: PrimInt + FromPrimitive + ToPrimitive>(r: I, g: I, b: I, a: I) -> Self {
        let min = I::from_u8(u8::MIN).expect("weird conversion failure from u8");
        let max = I::from_u8(u8::MAX).expect("weird conversion failure from u8");
        Self::from_bytes(
            r.clamp(min, max).to_u8().expect("weird conversion failure to u8"),
            g.clamp(min, max).to_u8().expect("weird conversion failure to u8"),
            b.clamp(min, max).to_u8().expect("weird conversion failure to u8"),
            a.clamp(min, max).to_u8().expect("weird conversion failure to u8"))
    }

    pub fn red() -> Self { Self { r: 1., ..Default::default() } }
    pub fn green() -> Self { Self { g: 1., ..Default::default() } }
    pub fn blue() -> Self { Self { b: 1., ..Default::default() } }
    pub fn yellow() -> Self { Self { r: 1., g: 1., ..Default::default() } }
    pub fn magenta() -> Self { Self { r: 1., b: 1., ..Default::default() } }
    pub fn cyan() -> Self { Self { g: 1., b: 1., ..Default::default() } }
    pub fn black() -> Self { Self::default() }
    pub fn white() -> Self { Self { r: 1., g: 1., b: 1., a: 1. } }

    pub fn as_bytes(&self) -> [u8; 4] {
        [(self.r * 255.) as u8, (self.g * 255.) as u8, (self.b * 255.) as u8, (self.a * 255.) as u8]
    }
    pub fn as_f32(&self) -> [f32; 4] { self.into() }
}

impl Default for Colour {
    fn default() -> Self {
        Self { r: 0., g: 0., b: 0., a: 1. }
    }
}

impl From<Colour> for [f32; 4] {
    fn from(value: Colour) -> Self {
        [value.r as f32, value.g as f32, value.b as f32, value.a as f32]
    }
}

impl From<&Colour> for [f32; 4] {
    fn from(value: &Colour) -> Self {
        (*value).into()
    }
}
