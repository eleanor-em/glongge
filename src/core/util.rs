use std::{
    cmp,
    hash::Hash,
    time::Instant,
};

use tracing::info;

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
            self.max_ns as f64 / 1_000_000.0
        );
        self.reset();
    }
    pub fn report_ms_if_at_least(&mut self, milliseconds: f64) {
        if self.mean_ms() > milliseconds || self.max_ns as f64 / 1_000_000.0 > milliseconds {
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
            return 0.0;
        }
        (self.total_ns / self.n) as f64 / 1_000.0
    }
    fn mean_ms(&self) -> f64 {
        self.mean_us() / 1_000.0
    }
    pub fn last_ms(&self) -> f64 {
        self.last_ns as f64 / 1_000_000.0
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

pub mod gg_time {
    use std::time::Duration;

    pub fn as_frames(duration: Duration) -> u128 {
        // 100 frames per second
        duration.as_micros() / 10000
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
