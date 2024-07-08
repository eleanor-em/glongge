
#[derive(Copy, Clone)]
pub struct Colour {
    pub r: f64,
    pub g: f64,
    pub b: f64,
    pub a: f64,
}

impl Colour {
    pub fn red() -> Self { Self { r: 1., ..Default::default() } }
    pub fn green() -> Self { Self { g: 1., ..Default::default() } }
    pub fn blue() -> Self { Self { b: 1., ..Default::default() } }
    pub fn yellow() -> Self { Self { r: 1., g: 1., ..Default::default() } }
    pub fn magenta() -> Self { Self { r: 1., b: 1., ..Default::default() } }
    pub fn cyan() -> Self { Self { g: 1., b: 1., ..Default::default() } }
    pub fn black() -> Self { Self::default() }
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
