use std::cmp;
use std::time::Instant;
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
    fn reset(&mut self) {
        *self = Self::new(&self.tag);
        self.last_report = Instant::now();
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
    pub fn report_ms(&mut self) {
        info!(
            "TimeIt [{:>16}]: {} events, mean={:.2} ms, max={:.2} ms",
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
