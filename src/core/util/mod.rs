#[allow(unused_imports)]
use crate::core::prelude::*;

use std::hash::Hash;
use std::ops::Deref;
use std::vec::IntoIter;

pub mod linalg;
pub mod colour;
pub mod assert;
pub mod collision;

#[allow(dead_code)]
pub mod gg_time {
    use std::cmp;
    use std::time::{Duration, Instant};
    use tracing::info;

    pub fn frames(n: u64) -> Duration {
        Duration::from_millis(10 * n)
    }

    pub fn as_frames(duration: Duration) -> u128 {
        duration.as_micros() / crate::core::config::FIXED_UPDATE_INTERVAL_US
    }

    pub struct TimeIt {
        tag: String,
        n: u128,
        total_ns: u128,
        max_ns: u128,
        last_ns: u128,
        last_wall: Instant,
        last_report: Instant,
    }

    impl TimeIt {
        pub fn new(tag: &str) -> Self {
            Self {
                tag: tag.to_string(),
                n: 0,
                total_ns: 0,
                max_ns: 0,
                last_ns: 0,
                last_wall: Instant::now(),
                last_report: Instant::now(),
            }
        }

        pub fn start(&mut self) {
            self.last_wall = Instant::now();
            self.last_ns = 0;
        }
        pub fn stop(&mut self) {
            let delta = self.last_wall.elapsed().as_nanos();
            self.n += 1;
            self.total_ns += delta;
            self.last_ns += delta;
            self.max_ns = cmp::max(self.max_ns, self.last_ns);
        }

        pub fn pause(&mut self) {
            let delta = self.last_wall.elapsed().as_nanos();
            self.total_ns += delta;
            self.last_ns += delta;
        }
        pub fn unpause(&mut self) {
            self.last_wall = Instant::now();
        }

        fn reset(&mut self) {
            *self = Self::new(&self.tag);
            self.last_report = Instant::now();
        }
        fn max_ms(&self) -> f64 {
            #[allow(clippy::cast_precision_loss)]
            let max_ns = self.max_ns as f64;
            max_ns / 1_000_000.
        }
        pub fn report_ms(&mut self) {
            info!(
                "TimeIt [{:>18}]: {} events, mean={:.2} ms, max={:.2} ms",
                self.tag,
                self.n,
                self.mean_ms(),
                self.max_ms()
            );
            self.reset();
        }
        pub fn report_ms_if_at_least(&mut self, milliseconds: f64) {
            if self.mean_ms() > milliseconds || self.max_ms() > milliseconds {
                self.report_ms();
            } else {
                self.reset();
            }
        }
        pub fn report_ms_every(&mut self, seconds: u64) {
            if self.last_report.elapsed().as_secs() > seconds {
                self.report_ms();
            }
        }
        fn mean_us(&self) -> f64 {
            if self.n == 0 {
                return 0.;
            }
            #[allow(clippy::cast_precision_loss)]
            let mean = (self.total_ns / self.n) as f64;
            mean / 1_000.
        }
        fn mean_ms(&self) -> f64 {
            self.mean_us() / 1_000.
        }
        pub fn last_ms(&self) -> f64 {
            #[allow(clippy::cast_precision_loss)]
            let last_ns = self.last_ns as f64;
            last_ns / 1_000_000.
        }
    }
}

#[allow(dead_code)]
pub mod gg_iter {
    use std::ops::Add;

    pub fn sum_tuple3<T: Add<Output=T>>(acc: (T, T, T), x: (T, T, T)) -> (T, T, T) {
        (acc.0 + x.0, acc.1 + x.1, acc.2 + x.2)
    }

    #[must_use]
    pub struct CumSum<I>
    where
        I: Iterator,
        I::Item: Clone + Add,
        <I::Item as Add>::Output: Into<I::Item>
    {
        it: I,
        cum_sum: Option<I::Item>,
    }

    impl<I> Iterator for CumSum<I>
    where
        I: Iterator,
        I::Item: Clone + Add,
        <I::Item as Add>::Output: Into<I::Item>
    {
        type Item = I::Item;

        fn next(&mut self) -> Option<Self::Item> {
            if let Some(next) = self.it.next() {
                if let Some(cum_sum) = self.cum_sum.take() {
                    let next: Self::Item = (cum_sum + next).into();
                    self.cum_sum = Some(next);
                } else {
                    self.cum_sum = Some(next);
                }
                self.cum_sum.clone()
            } else {
                None
            }
        }
    }

    pub trait GgIter: Iterator {
        fn cumsum(self) -> CumSum<Self>
        where
            Self::Item: Clone + Add,
            <Self::Item as Add>::Output: Into<Self::Item>,
            Self: Sized
        {
            CumSum {
                it: self,
                cum_sum: None,
            }
        }
    }

    impl<T> GgIter for T where T: Iterator + ?Sized {}
}

pub mod gg_range {
    use std::ops::Range;

    pub fn contains_f64(r1: &Range<f64>, r2: &Range<f64>) -> bool {
        r1.start <= r2.start && r1.end >= r2.end
    }

    pub fn overlap_f64(r1: &Range<f64>, r2: &Range<f64>) -> Option<Range<f64>> {
        if r1.start > r2.start {
            return overlap_f64(r2, r1);
        }
        if r1.end < r2.start {
            return None;
        }

        let start = r2.start;
        let end = f64::min(r1.end, r2.end);
        #[allow(clippy::float_cmp)]
        if start == end {
            None
        } else {
            Some(start..end)
        }
    }

    pub fn overlap_len_f64(r1: &Range<f64>, r2: &Range<f64>) -> Option<f64> {
        overlap_f64(r1, r2).map(|r| r.end - r.start)
    }
}

#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct UnorderedPair<T: Copy + Clone + Ord + PartialOrd + Eq + PartialEq + Hash>(T, T);
impl<T: Copy + Clone + Ord + PartialOrd + Eq + PartialEq + Hash> UnorderedPair<T> {
    pub fn new(a: T, b: T) -> Self {
        if a < b { Self(a, b) } else { Self(b, a) }
    }
    pub fn new_distinct(a: T, b: T) -> Option<Self> {
        if a == b {
            None
        } else {
            Some(Self::new(a, b))
        }
    }

    pub fn fst(&self) -> T { self.0 }
    pub fn snd(&self) -> T { self.1 }
    pub fn contains(&self, value: T) -> bool { self.fst() == value || self.snd() == value }
}

impl<T: Copy + Clone + Ord + PartialOrd + Eq + PartialEq + Hash> From<(T, T)> for UnorderedPair<T> {
    fn from(value: (T, T)) -> Self {
        Self::new(value.0, value.1)
    }
}

impl<T: Copy + Clone + Ord + PartialOrd + Eq + PartialEq + Hash> From<(&T, &T)> for UnorderedPair<T> {
    fn from(value: (&T, &T)) -> Self {
        Self::new(*value.0, *value.1)
    }
}

pub trait Nonempty: Sized {
    type Inner: IntoIterator;

    fn iter(&self) -> impl Iterator<Item=&<Self::Inner as IntoIterator>::Item>;
    fn min(&self) -> &<Self::Inner as IntoIterator>::Item
    where
        <Self::Inner as IntoIterator>::Item: Ord
    {
        unsafe { self.iter().min().unwrap_unchecked() }
    }
    fn max(&self) -> &<Self::Inner as IntoIterator>::Item
    where
        <Self::Inner as IntoIterator>::Item: Ord
    {
        unsafe { self.iter().max().unwrap_unchecked() }
    }
}

// TODO: implement max()
pub struct NonemptyVec<T> {
    inner: Vec<T>,
}

impl<T> NonemptyVec<T> {
    pub fn is_empty(&self) -> bool { false }
    pub fn len(&self) -> usize { self.inner.len() }

    pub fn first(&self) -> &T {
        unsafe { self.inner.first().unwrap_unchecked() }
    }

    pub fn try_from_vec(vec: Vec<T>) -> Option<Self> {
        if vec.is_empty() {
            None
        } else {
            Some(Self { inner: vec })
        }
    }
    pub fn try_from_iter<I>(iter: I) -> Option<Self>
    where
        I: Iterator<Item=T>
    {
        Self::try_from_vec(iter.collect_vec())
    }
}

impl<T> IntoIterator for NonemptyVec<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.inner.into_iter()
    }
}

impl<T> Nonempty for NonemptyVec<T> {
    type Inner = Vec<T>;

    fn iter(&self) -> impl Iterator<Item=&<Self::Inner as IntoIterator>::Item> {
        self.inner.iter()
    }
}

pub struct NonemptyVecRefMut<'a, T> {
    inner: &'a mut Vec<T>,
}

impl<'a, T> NonemptyVecRefMut<'a, T> {
    pub fn try_from_vec(vec: &'a mut Vec<T>) -> Option<Self> {
        if vec.is_empty() {
            None
        } else {
            Some(Self { inner: vec })
        }
    }
}

impl<'a, T> Nonempty for NonemptyVecRefMut<'a, T> {
    type Inner = Vec<T>;

    fn iter(&self) -> impl Iterator<Item=&<Self::Inner as IntoIterator>::Item> {
        self.inner.iter()
    }
}

impl<'a, T> Deref for NonemptyVecRefMut<'a, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target { self.inner }
}