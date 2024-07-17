use std::{
    fmt,
    fmt::Formatter,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign}
};
use std::ops::Range;
use itertools::{Itertools, Product};
use num_traits::{float::Float, One, Zero};

#[derive(Default, Debug, Eq, PartialEq, Copy, Clone, Hash)]
pub struct Vec2Int {
    pub x: i32,
    pub y: i32,
}

impl Vec2Int {
    pub fn right() -> Vec2Int { Vec2Int { x: 1, y: 0 } }
    pub fn up() -> Vec2Int { Vec2Int { x: 0, y: -1 } }
    pub fn left() -> Vec2Int { Vec2Int { x: -1, y: 0 } }
    pub fn down() -> Vec2Int { Vec2Int { x: 0, y: 1 } }
    pub fn one() -> Vec2Int { Vec2Int { x: 1, y: 1 } }

    pub fn len(&self) -> f64 { (self.dot(*self) as f64).sqrt() }
    pub fn dot(&self, other: Vec2Int) -> i32 { self.x * other.x + self.y * other.y }

    pub fn as_vec2(&self) -> Vec2 { Into::<Vec2>::into(*self) }

    pub fn range(start: Vec2Int, end: Vec2Int) -> Product<Range<i32>, Range<i32>> {
        (start.x..end.x).cartesian_product(start.y..end.y)
    }
    pub fn range_from_zero(end: Vec2Int) -> Product<Range<i32>, Range<i32>> {
        Self::range(Vec2Int::zero(), end)
    }
}

impl From<Vec2Int> for Vec2 {
    fn from(value: Vec2Int) -> Self {
        Self { x: value.x as f64, y: value.y as f64 }
    }
}

impl Zero for Vec2Int {
    fn zero() -> Self {
        Vec2Int { x: 0, y: 0 }
    }

    fn is_zero(&self) -> bool {
        *self == Self::zero()
    }
}

impl From<[i32; 2]> for Vec2Int {
    fn from(value: [i32; 2]) -> Self {
        Vec2Int {
            x: value[0],
            y: value[1],
        }
    }
}

impl From<Vec2Int> for [i32; 2] {
    fn from(value: Vec2Int) -> Self {
        [value.x, value.y]
    }
}

impl fmt::Display for Vec2Int {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "vec({}, {})", self.x, self.y)
    }
}

impl Add<Vec2Int> for Vec2Int {
    type Output = Vec2Int;

    fn add(self, rhs: Vec2Int) -> Self::Output {
        Vec2Int {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}
impl AddAssign<Vec2Int> for Vec2Int {
    fn add_assign(&mut self, rhs: Vec2Int) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl Sub<Vec2Int> for Vec2Int {
    type Output = Vec2Int;

    fn sub(self, rhs: Vec2Int) -> Self::Output {
        Vec2Int {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}
impl SubAssign<Vec2Int> for Vec2Int {
    fn sub_assign(&mut self, rhs: Vec2Int) {
        self.x -= rhs.x;
        self.y -= rhs.y;
    }
}

impl Mul<i32> for Vec2Int {
    type Output = Vec2Int;

    fn mul(self, rhs: i32) -> Self::Output {
        rhs * self
    }
}
impl Mul<Vec2Int> for i32 {
    type Output = Vec2Int;

    fn mul(self, rhs: Vec2Int) -> Self::Output {
        Vec2Int {
            x: self * rhs.x,
            y: self * rhs.y,
        }
    }
}
impl Mul<&Vec2Int> for i32 {
    type Output = Vec2Int;

    fn mul(self, rhs: &Vec2Int) -> Self::Output {
        Vec2Int {
            x: self * rhs.x,
            y: self * rhs.y,
        }
    }
}
impl MulAssign<i32> for Vec2Int {
    fn mul_assign(&mut self, rhs: i32) {
        self.x *= rhs;
        self.y *= rhs;
    }
}

impl Div<i32> for Vec2Int {
    type Output = Vec2Int;

    fn div(self, rhs: i32) -> Self::Output {
        Vec2Int {
            x: self.x / rhs,
            y: self.y / rhs,
        }
    }
}
impl DivAssign<i32> for Vec2Int {
    fn div_assign(&mut self, rhs: i32) {
        self.x /= rhs;
        self.y /= rhs;
    }
}

impl Neg for Vec2Int {
    type Output = Vec2Int;

    fn neg(self) -> Self::Output {
        Vec2Int {
            x: -self.x,
            y: -self.y,
        }
    }
}
impl Neg for &Vec2Int {
    type Output = Vec2Int;

    fn neg(self) -> Self::Output {
        Vec2Int {
            x: -self.x,
            y: -self.y,
        }
    }
}

#[derive(Default, Debug, Copy, Clone, PartialEq)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    pub fn right() -> Vec2 { Vec2 { x: 1., y: 0. } }
    pub fn up() -> Vec2 { Vec2 { x: 0., y: -1. } }
    pub fn left() -> Vec2 { Vec2 { x: -1., y: 0. } }
    pub fn down() -> Vec2 { Vec2 { x: 0., y: 1. } }
    pub fn one() -> Vec2 { Vec2 { x: 1., y: 1. } }

    pub fn len_squared(&self) -> f64 { self.dot(*self) }
    pub fn len(&self) -> f64 { self.len_squared().sqrt() }
    pub fn normed(&self) -> Vec2 {
        match self.len() {
            0. => Vec2::zero(),
            len => *self / len
        }
    }

    pub fn component_wise(&self, other: Vec2) -> Vec2 {
        Vec2 {
            x: self.x * other.x,
            y: self.y * other.y,
        }
    }

    pub fn dot(&self, other: Vec2) -> f64 { self.x * other.x + self.y * other.y }
    pub fn angle_radians(&self, other: Vec2) -> f64 { self.normed().dot(other.normed()).acos() }

    pub fn abs(&self) -> Vec2 { Vec2 { x: self.x.abs(), y: self.y.abs() }}
    pub fn rotated(&self, radians: f64) -> Vec2 {
        Mat3x3::rotation(radians) * *self
    }
    pub fn reflect(&self, normal: Vec2) -> Vec2 {
        *self - 2. * self.dot(normal) * normal
    }

    pub fn almost_eq(&self, rhs: Vec2) -> bool {
        (*self - rhs).len() < f32::epsilon() as f64
    }
}

impl Zero for Vec2 {
    fn zero() -> Self {
        Vec2 { x: 0., y: 0. }
    }

    fn is_zero(&self) -> bool {
        self.almost_eq(Self::zero())
    }
}

impl From<[f64; 2]> for Vec2 {
    fn from(value: [f64; 2]) -> Self {
        Vec2 {
            x: value[0],
            y: value[1],
        }
    }
}
impl From<[f32; 2]> for Vec2 {
    fn from(value: [f32; 2]) -> Self {
        Vec2 {
            x: value[0] as f64,
            y: value[1] as f64,
        }
    }
}

impl From<Vec2> for [f64; 2] {
    fn from(value: Vec2) -> Self {
        [value.x, value.y]
    }
}
impl From<Vec2> for [f32; 2] {
    fn from(value: Vec2) -> Self {
        [value.x as f32, value.y as f32]
    }
}

impl fmt::Display for Vec2 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "vec({}, {})", self.x, self.y)
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

impl Mul<f64> for Vec2 {
    type Output = Vec2;

    fn mul(self, rhs: f64) -> Self::Output {
        rhs * self
    }
}
impl Mul<Vec2> for f64 {
    type Output = Vec2;

    fn mul(self, rhs: Vec2) -> Self::Output {
        Vec2 {
            x: self * rhs.x,
            y: self * rhs.y,
        }
    }
}
impl Mul<&Vec2> for f64 {
    type Output = Vec2;

    fn mul(self, rhs: &Vec2) -> Self::Output {
        Vec2 {
            x: self * rhs.x,
            y: self * rhs.y,
        }
    }
}
impl MulAssign<f64> for Vec2 {
    fn mul_assign(&mut self, rhs: f64) {
        self.x *= rhs;
        self.y *= rhs;
    }
}
impl Mul<i32> for Vec2 {
    type Output = Vec2;

    fn mul(self, rhs: i32) -> Self::Output {
        rhs as f64 * self
    }
}
impl Mul<Vec2> for i32 {
    type Output = Vec2;

    fn mul(self, rhs: Vec2) -> Self::Output {
        Vec2 {
            x: self as f64 * rhs.x,
            y: self as f64 * rhs.y,
        }
    }
}
impl Mul<&Vec2> for i32 {
    type Output = Vec2;

    fn mul(self, rhs: &Vec2) -> Self::Output {
        Vec2 {
            x: self as f64 * rhs.x,
            y: self as f64 * rhs.y,
        }
    }
}
impl MulAssign<i32> for Vec2 {
    fn mul_assign(&mut self, rhs: i32) {
        self.x *= rhs as f64;
        self.y *= rhs as f64;
    }
}

impl Div<f64> for Vec2 {
    type Output = Vec2;

    fn div(self, rhs: f64) -> Self::Output {
        Vec2 {
            x: self.x / rhs,
            y: self.y / rhs,
        }
    }
}
impl DivAssign<f64> for Vec2 {
    fn div_assign(&mut self, rhs: f64) {
        self.x /= rhs;
        self.y /= rhs;
    }
}
impl Div<i32> for Vec2 {
    type Output = Vec2;

    fn div(self, rhs: i32) -> Self::Output {
        Vec2 {
            x: self.x / rhs as f64,
            y: self.y / rhs as f64,
        }
    }
}
impl DivAssign<i32> for Vec2 {
    fn div_assign(&mut self, rhs: i32) {
        self.x /= rhs as f64;
        self.y /= rhs as f64;
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

#[derive(Copy, Clone, PartialEq)]
pub struct Mat3x3 {
    pub xx: f64,
    pub xy: f64,
    pub xw: f64,
    pub yx: f64,
    pub yy: f64,
    pub yw: f64,
    pub wx: f64,
    pub wy: f64,
    pub ww: f64,
}

impl Mat3x3 {
    pub fn translation(dx: f64, dy: f64) -> Mat3x3 {
        Mat3x3 {
            xx: 1.,
            xy: 0.,
            xw: dx,
            yx: 0.,
            yy: 1.,
            yw: dy,
            wx: 0.,
            wy: 0.,
            ww: 1.,
        }
    }
    pub fn translation_vec2(vec2: Vec2) -> Mat3x3 {
        Self::translation(vec2.x, vec2.y)
    }
    pub fn rotation(radians: f64) -> Mat3x3 {
        Mat3x3 {
            xx: f64::cos(radians),
            xy: -f64::sin(radians),
            xw: 0.,
            yx: f64::sin(radians),
            yy: f64::cos(radians),
            yw: 0.,
            wx: 0.,
            wy: 0.,
            ww: 1.,
        }
    }
    pub fn det(&self) -> f64 {
        self.xx * (self.yy * self.ww - self.yw * self.wy)
            + self.xy * (self.yx * self.ww - self.yw * self.wx)
            + self.xw * (self.yx * self.wy - self.yy * self.wx)
    }
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

    pub fn almost_eq(&self, rhs: Mat3x3) -> bool {
        f64::abs(self.xx - rhs.xx) < f64::epsilon()
            && f64::abs(self.xy - rhs.xy) < f64::epsilon()
            && f64::abs(self.xw - rhs.xw) < f64::epsilon()
            && f64::abs(self.yx - rhs.yx) < f64::epsilon()
            && f64::abs(self.yy - rhs.yy) < f64::epsilon()
            && f64::abs(self.yw - rhs.yw) < f64::epsilon()
            && f64::abs(self.wx - rhs.wx) < f64::epsilon()
            && f64::abs(self.wy - rhs.wy) < f64::epsilon()
            && f64::abs(self.ww - rhs.ww) < f64::epsilon()
    }
}

impl One for Mat3x3 {
    fn one() -> Self {
        Mat3x3 {
            xx: 1.,
            xy: 0.,
            xw: 0.,
            yx: 0.,
            yy: 1.,
            yw: 0.,
            wx: 0.,
            wy: 0.,
            ww: 1.,
        }
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

impl Mul<f64> for Mat3x3 {
    type Output = Mat3x3;

    fn mul(self, rhs: f64) -> Self::Output {
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
impl Mul<Mat3x3> for f64 {
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
impl MulAssign<f64> for Mat3x3 {
    fn mul_assign(&mut self, rhs: f64) {
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

impl Div<f64> for Mat3x3 {
    type Output = Mat3x3;

    fn div(self, rhs: f64) -> Self::Output {
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
impl DivAssign<f64> for Mat3x3 {
    fn div_assign(&mut self, rhs: f64) {
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
            x: self.xx * rhs.x + self.xy * rhs.y + self.xw * 1.,
            y: self.yx * rhs.x + self.yy * rhs.y + self.yw * 1.,
        }
    }
}
impl MulAssign<Mat3x3> for Vec2 {
    fn mul_assign(&mut self, rhs: Mat3x3) {
        (self.x, self.y) = (
            rhs.xx * self.x + rhs.xy * self.y + rhs.xw * 1.,
            rhs.yx * self.x + rhs.yy * self.y + rhs.yw * 1.,
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
            [value.xx as f32, value.xy as f32, 0., value.xw as f32],
            [value.yx as f32, value.yy as f32, 0., value.yw as f32],
            [0., 0., 1., value.ww as f32],
            [value.wx as f32, value.wy as f32, 0., 1.],
        ]
    }
}

#[allow(dead_code)]
pub trait SquareExtent {
    fn extent(&self) -> Vec2;
    fn centre(&self) -> Vec2;

    fn half_widths(&self) -> Vec2 { self.extent() / 2 }
    fn top_left(&self) -> Vec2 { self.centre() - self.half_widths() }
    fn top_right(&self) -> Vec2 { self.centre() + self.half_widths().x * Vec2::right() / 2 }
    fn bottom_left(&self) -> Vec2 { self.centre() + self.half_widths().y * Vec2::down() / 2 }
    fn bottom_right(&self) -> Vec2 { self.centre() + self.half_widths() }
}

#[derive(Copy, Clone, Debug)]
pub struct Rect {
    centre: Vec2,
    half_widths: Vec2,
}

impl Rect {
    pub fn new(centre: Vec2, half_widths: Vec2) -> Self {
        Self { centre, half_widths }
    }
}

impl SquareExtent for Rect {
    fn extent(&self) -> Vec2 { self.half_widths * 2. }
    fn centre(&self) -> Vec2 { self.centre }
}

pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}
