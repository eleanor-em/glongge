#[allow(unused_imports)]
use crate::core::prelude::*;

use std::{
    fmt,
    fmt::Formatter,
    ops::{
        Add,
        AddAssign,
        Div,
        DivAssign,
        Mul,
        MulAssign,
        Neg,
        Sub,
        SubAssign,
        Range
    }
};
use std::cmp::Ordering;
use std::iter::Sum;
use std::sync::Arc;
use itertools::Product;
use num_traits::{float::Float, One, Zero};
use serde::{Deserialize, Serialize};
use crate::util::gg_float;

#[derive(Default, Debug, Eq, PartialEq, Ord, PartialOrd, Copy, Clone, Hash, Serialize, Deserialize)]
pub struct Vec2i {
    pub x: i32,
    pub y: i32,
}

impl Vec2i {
    pub fn right() -> Vec2i { Vec2i { x: 1, y: 0 } }
    pub fn up() -> Vec2i { Vec2i { x: 0, y: -1 } }
    pub fn left() -> Vec2i { Vec2i { x: -1, y: 0 } }
    pub fn down() -> Vec2i { Vec2i { x: 0, y: 1 } }
    pub fn one() -> Vec2i { Vec2i { x: 1, y: 1 } }

    pub fn len(&self) -> f64 { f64::from(self.dot(*self)).sqrt() }
    pub fn dot(&self, other: Vec2i) -> i32 { self.x * other.x + self.y * other.y }

    pub fn as_vec2(&self) -> Vec2 { Into::<Vec2>::into(*self) }

    pub fn range(start: Vec2i, end: Vec2i) -> Product<Range<i32>, Range<i32>> {
        (start.x..end.x).cartesian_product(start.y..end.y)
    }
    pub fn range_from_zero(end: impl Into<Vec2i>) -> Product<Range<i32>, Range<i32>> {
        Self::range(Vec2i::zero(), end.into())
    }

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
        Self { x: f64::from(value.x), y: f64::from(value.y) }
    }
}

impl Zero for Vec2i {
    fn zero() -> Self {
        Vec2i { x: 0, y: 0 }
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
        [value.x.abs().try_into().unwrap(), value.y.abs().try_into().unwrap()]
    }
}

impl fmt::Display for Vec2i {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "vec({}, {})", self.x, self.y)
    }
}

#[derive(Default, Debug, Eq, PartialEq, Ord, PartialOrd, Copy, Clone, Hash, Serialize, Deserialize)]
pub struct Edge2i(pub Vec2i, pub Vec2i);

impl Edge2i {
    pub fn as_tuple(&self) -> (Vec2i, Vec2i) { (self.0, self.1) }
    #[must_use]
    pub fn reverse(self) -> Self { Self(self.1, self.0) }
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

#[derive(Default, Debug, Copy, Clone)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl PartialEq for Vec2 {
    fn eq(&self, other: &Self) -> bool {
        (self.x - other.x).abs() < EPSILON &&
            (self.y - other.y).abs() < EPSILON
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
        if self == other { return Ordering::Equal; }
        if (self.x - other.x).abs() < EPSILON {
            return self.y.partial_cmp(&other.y)
                .unwrap_or_else(|| {
                    warn!("Vec2: partial_cmp() failed for y: {} vs. {}", self, other);
                    self.y.total_cmp(&other.y)
                });
        }
        if let Some(o) = self.x.partial_cmp(&other.x) { o } else {
            warn!("Vec2: partial_cmp() failed for x: {} vs. {}", self, other);
            match self.x.total_cmp(&other.x) {
                Ordering::Equal => if let Some(o) = self.y.partial_cmp(&other.y) {
                    o
                } else {
                    warn!("Vec2: partial_cmp() failed for x: {} vs. {}", self, other);
                    self.y.total_cmp(&other.y)
                }
                o => o
            }
        }
    }
}

// TODO: make const functions
#[allow(clippy::return_self_not_must_use)]
impl Vec2 {
    pub fn right() -> Vec2 { Vec2 { x: 1., y: 0. } }
    pub fn up() -> Vec2 { Vec2 { x: 0., y: -1. } }
    pub fn left() -> Vec2 { Vec2 { x: -1., y: 0. } }
    pub fn down() -> Vec2 { Vec2 { x: 0., y: 1. } }
    pub fn one() -> Vec2 { Vec2 { x: 1., y: 1. } }

    pub fn len_squared(&self) -> f64 { self.dot(*self) }
    pub fn len(&self) -> f64 { self.len_squared().sqrt() }
    pub fn normed(&self) -> Vec2 {
        let mut rv = match self.len() {
            0. => Vec2::zero(),
            len => *self / len
        };
        if rv.x == -0.{ rv.x = 0.; }
        if rv.y == -0. { rv.y = 0.; }
        rv
    }

    pub fn component_wise(&self, other: Vec2) -> Vec2 {
        Vec2 {
            x: self.x * other.x,
            y: self.y * other.y,
        }
    }

    pub fn dot(&self, other: Vec2) -> f64 { self.x * other.x + self.y * other.y }
    pub fn cross(&self, other: Vec2) -> f64 {
        self.x * other.y - self.y * other.x
    }
    pub fn angle_radians(&self, other: Vec2) -> f64 { self.normed().dot(other.normed()).acos() }
    pub fn longest_component(&self) -> f64 { self.x.abs().max(self.y.abs()) }

    pub fn abs(&self) -> Vec2 { Vec2 { x: self.x.abs(), y: self.y.abs() }}
    pub fn rotated(&self, radians: f64) -> Vec2 {
        Mat3x3::rotation(radians) * *self
    }
    pub fn reflect(&self, normal: Vec2) -> Vec2 {
        *self - 2. * self.dot(normal) * normal
    }
    pub fn reciprocal(&self) -> Vec2 { Vec2 { x: 1. / self.x, y: 1. / self.y } }
    pub fn project(&self, axis: Vec2) -> Vec2 { self.dot(axis.normed()) * axis.normed() }
    pub fn dist(&self, other: Vec2) -> f64 { (other - *self).len() }
    pub fn dist_squared(&self, other: Vec2) -> f64 { (other - *self).len_squared() }
    pub fn dist_to_line(&self, start: Vec2, end: Vec2) -> f64 {
        let dx = end - start;
        let l2 = dx.len_squared();
        if l2.is_zero() { return self.dist(start); }
        let t = ((*self - start).dot(dx) / l2).clamp(0., 1.);
        self.dist(start + t * dx)
    }
    pub fn intersect(p1: Vec2, ax1: Vec2, p2: Vec2, ax2: Vec2) -> Option<Vec2> {
        let denom = ax1.cross(ax2);
        if denom.is_zero() {
            None
        } else {
            let t = (p2 - p1).cross(ax2) / denom;
            let u = (p2 - p1).cross(ax1) / denom;
            if (0. ..=1.).contains(&t) &&
                (0. ..=1.).contains(&u) {
                Some(p1 + t * ax1)
            } else {
                None
            }
        }
    }
    pub fn orthog(&self) -> Vec2 { Vec2 { x: self.y, y: -self.x } }
    pub fn lerp(&self, to: Vec2, t: f64) -> Vec2 {
        Vec2 {
            x: linalg::lerp(self.x, to.x, t),
            y: linalg::lerp(self.y, to.y, t),
        }
    }

    pub fn almost_eq(&self, rhs: Vec2) -> bool {
        (*self - rhs).len() < f64::from(f32::epsilon())
    }

    #[allow(clippy::cast_possible_truncation)]
    pub fn as_vec2int_lossy(&self) -> Vec2i {
        Vec2i { x: self.x.round() as i32, y: self.y.round() as i32 }
    }
    #[allow(clippy::cast_possible_truncation)]
    pub fn as_f32_lossy(&self) -> [f32; 2] {
        (*self).into()
    }

    pub fn is_normal_or_zero(&self) -> bool {
        gg_float::is_normal_or_zero(self.x) || gg_float::is_normal_or_zero(self.y)
    }

    pub fn cmp_by_length(&self, other: &Vec2) -> Ordering {
        let self_len = self.len_squared();
        let other_len = other.len_squared();
        self_len.partial_cmp(&other_len)
            .unwrap_or_else(|| {
                warn!("cmp_by_length(): partial_cmp() failed: {} vs. {}", self, other);
                self_len.total_cmp(&other_len)
            })
    }
    pub fn cmp_by_dist(&self, other: &Vec2, origin: Vec2) -> Ordering {
        let self_len = (*self - origin).len_squared();
        let other_len = (*other - origin).len_squared();
        self_len.partial_cmp(&other_len)
            .unwrap_or_else(|| {
                warn!("cmp_by_dist() to {}: partial_cmp() failed: {} vs. {}", origin, self, other);
                self_len.total_cmp(&other_len)
            })
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
            x: f64::from(value[0]),
            y: f64::from(value[1]),
        }
    }
}
impl From<[i32; 2]> for Vec2 {
    fn from(value: [i32; 2]) -> Self {
        Vec2 {
            x: f64::from(value[0]),
            y: f64::from(value[1]),
        }
    }
}

impl From<Vec2> for [f64; 2] {
    fn from(value: Vec2) -> Self {
        [value.x, value.y]
    }
}
#[allow(clippy::cast_possible_truncation)]
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

impl Sum<Vec2> for Vec2 {
    fn sum<I: Iterator<Item=Vec2>>(iter: I) -> Self {
        iter.fold(Vec2::zero(), Vec2::add)
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
        f64::from(rhs) * self
    }
}
impl Mul<Vec2> for i32 {
    type Output = Vec2;

    fn mul(self, rhs: Vec2) -> Self::Output {
        Vec2 {
            x: f64::from(self) * rhs.x,
            y: f64::from(self) * rhs.y,
        }
    }
}
impl Mul<&Vec2> for i32 {
    type Output = Vec2;

    fn mul(self, rhs: &Vec2) -> Self::Output {
        Vec2 {
            x: f64::from(self) * rhs.x,
            y: f64::from(self) * rhs.y,
        }
    }
}
impl MulAssign<i32> for Vec2 {
    fn mul_assign(&mut self, rhs: i32) {
        self.x *= f64::from(rhs);
        self.y *= f64::from(rhs);
    }
}
impl Mul<u32> for Vec2 {
    type Output = Vec2;

    fn mul(self, rhs: u32) -> Self::Output {
        f64::from(rhs) * self
    }
}
impl Mul<Vec2> for u32 {
    type Output = Vec2;

    fn mul(self, rhs: Vec2) -> Self::Output {
        Vec2 {
            x: f64::from(self) * rhs.x,
            y: f64::from(self) * rhs.y,
        }
    }
}
impl Mul<&Vec2> for u32 {
    type Output = Vec2;

    fn mul(self, rhs: &Vec2) -> Self::Output {
        Vec2 {
            x: f64::from(self) * rhs.x,
            y: f64::from(self) * rhs.y,
        }
    }
}
impl MulAssign<u32> for Vec2 {
    fn mul_assign(&mut self, rhs: u32) {
        self.x *= f64::from(rhs);
        self.y *= f64::from(rhs);
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
            x: self.x / f64::from(rhs),
            y: self.y / f64::from(rhs),
        }
    }
}
impl DivAssign<i32> for Vec2 {
    fn div_assign(&mut self, rhs: i32) {
        self.x /= f64::from(rhs);
        self.y /= f64::from(rhs);
    }
}
impl Div<u32> for Vec2 {
    type Output = Vec2;

    fn div(self, rhs: u32) -> Self::Output {
        Vec2 {
            x: self.x / f64::from(rhs),
            y: self.y / f64::from(rhs),
        }
    }
}
impl DivAssign<u32> for Vec2 {
    fn div_assign(&mut self, rhs: u32) {
        self.x /= f64::from(rhs);
        self.y /= f64::from(rhs);
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

#[derive(Default, Debug, Copy, Clone, PartialOrd, PartialEq)]
pub struct Vec2Small {
    pub x: f32,
    pub y: f32,
}

#[derive(Copy, Clone, PartialEq)]
#[must_use]
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
    #[allow(clippy::cast_possible_truncation)]
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
pub trait AxisAlignedExtent {
    fn aa_extent(&self) -> Vec2;
    fn centre(&self) -> Vec2;

    fn half_widths(&self) -> Vec2 { self.aa_extent() / 2 }
    fn top_left(&self) -> Vec2 { self.centre() - self.half_widths() }
    fn top_right(&self) -> Vec2 { self.top_left() + self.aa_extent().x * Vec2::right() }
    fn bottom_left(&self) -> Vec2 { self.top_left() + self.aa_extent().y * Vec2::down() }
    fn bottom_right(&self) -> Vec2 { self.top_left() + self.aa_extent() }

    fn left(&self) -> f64 { self.top_left().x }
    fn right(&self) -> f64 { self.top_right().x }
    fn top(&self) -> f64 { self.top_left().y }
    fn bottom(&self) -> f64 { self.bottom_left().y }

    fn as_rect(&self) -> Rect {
        Rect::new(self.centre(), self.half_widths())
    }
    fn contains_point(&self, pos: Vec2) -> bool {
        (self.left()..self.right()).contains(&pos.x) &&
            (self.top()..self.bottom()).contains(&pos.y)
    }
}

impl<T: AxisAlignedExtent> AxisAlignedExtent for Arc<T> {
    fn aa_extent(&self) -> Vec2 {
        self.as_ref().aa_extent()
    }

    fn centre(&self) -> Vec2 {
        self.as_ref().centre()
    }
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd)]
pub struct Rect {
    centre: Vec2,
    half_widths: Vec2,
}

impl Rect {
    pub fn new(centre: Vec2, half_widths: Vec2) -> Self {
        Self { centre, half_widths }
    }
    pub fn from_coords(top_left: Vec2, bottom_right: Vec2) -> Self {
        let half_widths = (bottom_right - top_left) / 2;
        let centre = top_left + half_widths;
        Self { centre, half_widths }
    }
    pub fn empty() -> Self { Self { centre: Vec2::zero(), half_widths: Vec2::zero() } }
}

impl AxisAlignedExtent for Rect {
    fn aa_extent(&self) -> Vec2 { self.half_widths * 2. }
    fn centre(&self) -> Vec2 { self.centre }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Transform {
    pub centre: Vec2,
    pub rotation: f64,
    pub scale: Vec2,
}

impl Transform {
    #[must_use]
    pub fn with_centre(centre: Vec2) -> Self {
        Self { centre, ..Default::default() }
    }
    #[must_use]
    pub fn with_rotation(rotation: f64) -> Self {
        Self { rotation, ..Default::default() }
    }
    #[must_use]
    pub fn with_scale(scale: Vec2) -> Self {
        Self { scale, ..Default::default() }
    }

    #[must_use]
    pub fn translated(&self, by: Vec2) -> Self {
        Self {
            centre: self.centre + by,
            rotation: self.rotation,
            scale: self.scale,
        }
    }

    pub fn as_f32_lossy(&self) -> TransformF32 {
        TransformF32 {
            centre: self.centre.into(),
            rotation: self.rotation as f32,
            scale: self.scale.into(),
        }
    }

    #[must_use]
    pub fn inverse(&self) -> Self {
        Self {
            centre: -self.centre,
            rotation: -self.rotation,
            scale: self.scale.reciprocal(),
        }
    }

    pub fn left(&self) -> Vec2 {
        Vec2::left().rotated(self.rotation)
    }
    pub fn up(&self) -> Vec2 {
        Vec2::up().rotated(self.rotation)
    }
    pub fn right(&self) -> Vec2 {
        Vec2::right().rotated(self.rotation)
    }
    pub fn down(&self) -> Vec2 {
        Vec2::down().rotated(self.rotation)
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self { centre: Vec2::zero(), rotation: 0., scale: Vec2::one() }
    }
}

impl Mul<Transform> for Transform {
    type Output = Transform;

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

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct TransformF32 {
    pub centre: [f32; 2],
    pub rotation: f32,
    pub scale: [f32; 2],
}

pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}
pub fn eerp(a: f64, b: f64, t: f64) -> f64 {
    a * (t * (b / a).ln()).exp()
}
pub fn smooth(t: f64) -> f64 { (6. * t*t*t*t*t - 15. * t*t*t*t + 10. * t*t*t).clamp(0., 1.) }
pub fn sigmoid(t: f64, k: f64) -> f64 {
    1. / (1. + (-(t - 0.5) / k).exp())
}
