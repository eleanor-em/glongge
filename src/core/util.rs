#[allow(unused_imports)]
use crate::core::prelude::*;

use std::{
    cmp,
    hash::Hash,
    time::Instant,
};

pub struct TimeIt {
    tag: String,
    n: u64,
    total_ns: u64,
    max_ns: u64,
    last_ns: u64,
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
        let delta = self.last_wall.elapsed().as_nanos() as u64;
        self.n += 1;
        self.total_ns += delta;
        self.last_ns += delta;
        self.max_ns = cmp::max(self.max_ns, self.last_ns);
    }

    pub fn pause(&mut self) {
        let delta = self.last_wall.elapsed().as_nanos() as u64;
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
    pub fn report_ms(&mut self) {
        info!(
            "TimeIt [{:>18}]: {} events, mean={:.2} ms, max={:.2} ms",
            self.tag,
            self.n,
            self.mean_ms(),
            self.max_ns as f64 / 1_000_000.
        );
        self.reset();
    }
    pub fn report_ms_if_at_least(&mut self, milliseconds: f64) {
        if self.mean_ms() > milliseconds || self.max_ns as f64 / 1_000_000. > milliseconds {
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
        (self.total_ns / self.n) as f64 / 1_000.
    }
    fn mean_ms(&self) -> f64 {
        self.mean_us() / 1_000.
    }
    pub fn last_ms(&self) -> f64 {
        self.last_ns as f64 / 1_000_000.
    }
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

#[allow(dead_code)]
pub mod gg_time {
    use std::time::Duration;

    pub fn frames(n: u64) -> Duration {
        Duration::from_millis(10 * n)
    }

    pub fn as_frames(duration: Duration) -> u128 {
        duration.as_micros() / crate::core::config::FIXED_UPDATE_INTERVAL_US
    }
}

#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct UnorderedPair<T: Copy + Clone + Ord + PartialOrd + Eq + PartialEq + Hash>(T, T);
impl<T: Copy + Clone + Ord + PartialOrd + Eq + PartialEq + Hash> UnorderedPair<T> {
    pub fn new(a: T, b: T) -> Self {
        if a < b { Self(a, b) } else { Self(b, a) }
    }
    pub fn new_distinct(a: T, b: T) -> Option<Self> {
        if a != b {
            Some(Self::new(a, b))
        } else {
            None
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

// TODO: implement max()
#[derive(Clone)]
pub struct NonemptyVec<T: Clone> {
    inner: Vec<T>,
}

impl<T: Clone> NonemptyVec<T> {
    pub fn is_empty(&self) -> bool { false }
    pub fn len(&self) -> usize { self.inner.len() }
    pub fn first(&self) -> &T { unsafe { self.inner.first().unwrap_unchecked() } }
    pub fn into_inner(self) -> Vec<T> { self.inner }

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

impl<T: Clone> IntoIterator for NonemptyVec<T> {
    type Item = T;
    type IntoIter = <Vec<T> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.into_inner().into_iter()
    }
}
