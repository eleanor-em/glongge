use crate::core::prelude::*;

use crate::core::input::InputHandler;
use crate::core::render::RenderHandler;
use crate::core::scene::SceneHandler;
use crate::core::vk::WindowEventHandler;
use crate::gui::{GuiContext, GuiUi};
use egui::{Button, WidgetText};
use std::fmt::{Debug, Display, Formatter};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, mpsc};
use std::{hash::Hash, ops::Deref, vec::IntoIter};
use tracing_subscriber::fmt::time::OffsetTime;

pub mod assert;
pub mod canvas;
pub mod collision;
pub mod colour;
pub mod linalg;
pub mod log;
pub mod spline;
pub mod tileset;

/// A macro that takes a path (e.g.0, "/res/DejaVuSansMono.ttf") and returns
/// `include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), path))`.
///
/// # Examples
///
/// ```
/// use glongge::include_bytes_root;
/// let font_bytes = include_bytes_root!("res/DejaVuSansMono.ttf");
/// ```
#[macro_export]
macro_rules! include_bytes_root {
    ($path:expr) => {
        include_bytes!(concat!(concat!(env!("CARGO_MANIFEST_DIR"), "/"), $path))
    };
}

#[derive(Default)]
pub(crate) struct GlobalStats {
    pub(crate) total_leaked_memory_bytes: usize,
}

pub(crate) static GLOBAL_STATS: OnceLock<UniqueShared<GlobalStats>> = OnceLock::new();

#[allow(dead_code)]
pub mod gg_time {
    use crate::util::gg_float;
    use std::time::{Duration, Instant};
    use tracing::info;

    pub fn frames(n: u64) -> Duration {
        Duration::from_millis(10 * n)
    }

    pub fn as_frames(duration: Duration) -> u128 {
        duration.as_micros() / crate::core::config::FIXED_UPDATE_INTERVAL_US
    }

    #[derive(Clone)]
    pub struct TimeIt {
        tag: String,
        n: u128,
        total_ns: u128,
        min_ns: u128,
        max_ns: u128,
        last_ns: u128,
        last_wall: Instant,
        last_report: Instant,
        running: bool,
    }

    impl TimeIt {
        pub fn new(tag: &str) -> Self {
            Self {
                tag: tag.to_string(),
                n: 0,
                total_ns: 0,
                min_ns: u128::MAX,
                max_ns: 0,
                last_ns: 0,
                last_wall: Instant::now(),
                last_report: Instant::now(),
                running: false,
            }
        }

        pub fn start(&mut self) {
            self.last_wall = Instant::now();
            self.last_ns = 0;
            self.running = true;
        }
        pub fn stop(&mut self) {
            if self.running {
                let delta = self.last_wall.elapsed().as_nanos();
                self.n += 1;
                self.total_ns += delta;
                self.last_ns += delta;
                self.min_ns = self.min_ns.min(self.last_ns);
                self.max_ns = self.max_ns.max(self.last_ns);
                self.running = false;
            }
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
            self.running = false;
            *self = Self::new(&self.tag);
            self.last_report = Instant::now();
        }
        fn min_ms(&self) -> f32 {
            let min_ns = gg_float::from_u128_or_inf(self.min_ns);
            min_ns / 1_000_000.0
        }
        fn max_ms(&self) -> f32 {
            let max_ns = gg_float::from_u128_or_inf(self.max_ns);
            max_ns / 1_000_000.0
        }
        pub fn as_tuple_ms(&self) -> (String, f32, f32, f32) {
            (
                self.tag.clone(),
                self.min_ms(),
                self.mean_ms(),
                self.max_ms(),
            )
        }
        #[must_use]
        pub fn report_take(&mut self) -> TimeIt {
            let rv = self.clone();
            self.reset();
            rv
        }
        pub fn report_ms(&mut self) {
            info!(
                "TimeIt [{:>18}]: {} events, min={:.2} ms, mean={:.2} ms, max={:.2} ms",
                self.tag,
                self.n,
                self.min_ms(),
                self.mean_ms(),
                self.max_ms()
            );
            self.reset();
        }
        pub fn report_ms_if_at_least(&mut self, milliseconds: f32) {
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
        fn mean_us(&self) -> f32 {
            if self.n == 0 {
                return 0.0;
            }
            let mean = gg_float::from_u128_or_inf(self.total_ns / self.n);
            mean / 1_000.0
        }
        fn mean_ms(&self) -> f32 {
            self.mean_us() / 1_000.0
        }
        pub fn last_ms(&self) -> f32 {
            gg_float::from_u128_or_inf(self.last_ns) / 1_000_000.0
        }
    }
}

#[allow(dead_code)]
pub mod gg_iter {
    use std::cmp;
    use std::ops::Add;

    pub fn sum_tuple3<T: Add<Output = T>>(acc: (T, T, T), x: (T, T, T)) -> (T, T, T) {
        (acc.0 + x.0, acc.1 + x.1, acc.2 + x.2)
    }

    pub fn index_of<T: Eq>(slice: &[T], value: &T) -> Option<usize> {
        for (i, x) in slice.iter().enumerate() {
            if x == value {
                return Some(i);
            }
        }
        None
    }

    pub fn partition_point_by<T, F>(slice: &[T], mut comparator: F) -> usize
    where
        F: FnMut(&T) -> cmp::Ordering,
    {
        slice.partition_point(|x| (comparator)(x) != cmp::Ordering::Greater)
    }

    #[must_use]
    pub struct CumSum<I>
    where
        I: Iterator,
        I::Item: Clone + Add,
        <I::Item as Add>::Output: Into<I::Item>,
    {
        it: I,
        cum_sum: Option<I::Item>,
    }

    impl<I> Iterator for CumSum<I>
    where
        I: Iterator,
        I::Item: Clone + Add,
        <I::Item as Add>::Output: Into<I::Item>,
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
            Self: Sized,
        {
            CumSum {
                it: self,
                cum_sum: None,
            }
        }
    }

    impl<T> GgIter for T where T: Iterator + ?Sized {}

    pub trait GgFloatIter: Iterator<Item = f32> {
        /// Returns the "obvious" max, with the following caveats:
        /// - if any input is NaN, returns the first NaN encountered;
        /// - +0.0 vs. -0.0 is handled nondeterministically, see `f32::max()`.
        fn max_f32(self) -> Option<f32>
        where
            Self: Sized,
        {
            self.fold(None, |max, x| {
                if x.is_nan() {
                    return Some(x);
                }
                match max {
                    None => Some(x),
                    Some(m) => Some(m.max(x)),
                }
            })
        }
        fn min_f32(self) -> Option<f32>
        where
            Self: Sized,
        {
            self.fold(None, |min, x| {
                if x.is_nan() {
                    return Some(x);
                }
                match min {
                    None => Some(x),
                    Some(m) => Some(m.min(x)),
                }
            })
        }
    }

    impl<T: Iterator<Item = f32>> GgFloatIter for T {}
}

#[allow(dead_code)]
pub mod gg_err {
    use anyhow::Result;
    use tracing::error;
    use vulkano::command_buffer::CommandBufferExecError;
    use vulkano::{Validated, ValidationError, VulkanError};
    use vulkano_taskgraph::InvalidSlotError;
    use vulkano_taskgraph::graph::ExecuteError;

    fn log_error(e: &anyhow::Error) {
        error!("{}", e);
        e.chain()
            .skip(1)
            .for_each(|cause| error!("caused by: {}", cause));
    }

    pub fn is_some_and_log<T>(result: Result<Option<T>>) -> bool {
        match result {
            Ok(Some(_)) => true,
            Ok(None) => false,
            Err(e) => {
                log_error(&e);
                false
            }
        }
    }

    pub fn log_err_and_ignore<T>(result: Result<T>) {
        if let Err(e) = result {
            log_error(&e);
        }
    }

    pub fn log_err_then<T>(result: Result<Option<T>>) -> Option<T> {
        match result {
            Ok(o) => o,
            Err(e) => {
                log_error(&e);
                None
            }
        }
    }
    pub fn log_err_then_invert<T>(val: Option<Result<T>>) -> Option<T> {
        val.and_then(|result| match result {
            Ok(o) => Some(o),
            Err(e) => {
                log_error(&e);
                None
            }
        })
    }

    pub fn log_unwrap_or<T, U: Into<T>>(default: U, result: Result<T>) -> T {
        match result {
            Ok(v) => v,
            Err(e) => {
                log_error(&e);
                default.into()
            }
        }
    }

    pub fn log_and_ok<T>(result: Result<T>) -> Option<T> {
        match result {
            Ok(v) => Some(v),
            Err(e) => {
                log_error(&e);
                None
            }
        }
    }

    #[derive(Debug)]
    pub(crate) enum CatchOutOfDate {
        Anyhow(anyhow::Error),
        VulkanOutOfDateError,
    }

    impl From<VulkanError> for CatchOutOfDate {
        fn from(value: VulkanError) -> Self {
            match value {
                VulkanError::OutOfDate => Self::VulkanOutOfDateError,
                e => Self::Anyhow(e.into()),
            }
        }
    }
    impl From<Validated<VulkanError>> for CatchOutOfDate {
        fn from(value: Validated<VulkanError>) -> Self {
            match value {
                Validated::Error(e) => e.into(),
                Validated::ValidationError(_) => Self::Anyhow(value.into()),
            }
        }
    }
    impl From<Box<ValidationError>> for CatchOutOfDate {
        fn from(value: Box<ValidationError>) -> Self {
            Self::Anyhow(value.into())
        }
    }
    impl From<CommandBufferExecError> for CatchOutOfDate {
        fn from(value: CommandBufferExecError) -> Self {
            Self::Anyhow(value.into())
        }
    }
    impl From<InvalidSlotError> for CatchOutOfDate {
        fn from(value: InvalidSlotError) -> Self {
            Self::Anyhow(value.into())
        }
    }
    impl From<ExecuteError> for CatchOutOfDate {
        fn from(value: ExecuteError) -> Self {
            Self::Anyhow(value.into())
        }
    }
    impl From<anyhow::Error> for CatchOutOfDate {
        fn from(value: anyhow::Error) -> Self {
            Self::Anyhow(value)
        }
    }
    impl From<CatchOutOfDate> for anyhow::Error {
        fn from(value: CatchOutOfDate) -> Self {
            match value {
                CatchOutOfDate::Anyhow(e) => e,
                CatchOutOfDate::VulkanOutOfDateError => VulkanError::OutOfDate.into(),
            }
        }
    }
}

pub mod gg_float {
    use crate::util::linalg::{Transform, Vec2};
    use anyhow::{Result, bail};
    use num_traits::{FromPrimitive, Zero};
    use std::num::FpCategory;
    use std::time::Duration;

    pub trait GgFloat {
        fn is_finite(&self) -> bool;
    }

    impl GgFloat for f32 {
        fn is_finite(&self) -> bool {
            self.is_normal() || self.is_zero()
        }
    }

    impl GgFloat for Vec2 {
        fn is_finite(&self) -> bool {
            self.x.is_finite() && self.y.is_finite()
        }
    }

    impl GgFloat for Transform {
        fn is_finite(&self) -> bool {
            self.centre.is_finite() && self.rotation.is_finite() && self.scale.is_finite()
        }
    }
    pub fn is_finite(x: f32) -> bool {
        matches!(x.classify(), FpCategory::Zero | FpCategory::Normal)
    }

    pub fn micros(duration: Duration) -> f32 {
        from_u128_or_inf(duration.as_micros()) / 1_000_000.0
    }

    pub fn f32_to_u32(x: f32) -> Result<u32> {
        if x > u32::MAX as f32 || x < 0.0 {
            bail!("{x} does not fit in range of u32");
        }
        #[allow(clippy::cast_sign_loss)]
        Ok(x as u32)
    }

    pub fn from_u128_or_inf(x: u128) -> f32 {
        f32::from_u128(x).unwrap_or(f32::INFINITY)
    }

    pub fn force_positive_zero(x: f32) -> f32 {
        if x.is_zero() { 0.0 } else { x }
    }

    pub fn sign_zero(x: f32) -> f32 {
        if x.is_zero() { 0.0 } else { x.signum() }
    }
}

pub mod gg_range {
    use crate::core::config::EPSILON;
    use std::ops::Range;

    pub fn contains_f32(r1: &Range<f32>, r2: &Range<f32>) -> bool {
        r1.start <= r2.start && r1.end >= r2.end
    }

    pub fn overlap_f32(r1: &Range<f32>, r2: &Range<f32>) -> Option<Range<f32>> {
        if r1.start > r2.start {
            return overlap_f32(r2, r1);
        }
        if r1.end < r2.start {
            return None;
        }

        let start = r2.start;
        let end = f32::min(r1.end, r2.end);
        if (start - end).abs() < EPSILON {
            None
        } else {
            Some(start..end)
        }
    }

    pub fn overlap_len_f32(r1: &Range<f32>, r2: &Range<f32>) -> Option<f32> {
        overlap_f32(r1, r2).map(|r| r.end - r.start)
    }
}

#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
pub struct UnorderedPair<T: Copy + Clone + Ord + PartialOrd + Eq + PartialEq>(T, T);
impl<T: Copy + Clone + Ord + PartialOrd + Eq + PartialEq> UnorderedPair<T> {
    pub fn new(a: T, b: T) -> Self {
        if a < b { Self(a, b) } else { Self(b, a) }
    }
    pub fn new_distinct(a: T, b: T) -> Option<Self> {
        if a == b { None } else { Some(Self::new(a, b)) }
    }

    pub fn fst(&self) -> T {
        self.0
    }
    pub fn snd(&self) -> T {
        self.1
    }
    pub fn contains(&self, value: T) -> bool {
        self.fst() == value || self.snd() == value
    }
}

impl<T: Copy + Clone + Ord + PartialOrd + Eq + PartialEq + Hash> From<(T, T)> for UnorderedPair<T> {
    fn from(value: (T, T)) -> Self {
        Self::new(value.0, value.1)
    }
}

impl<T: Copy + Clone + Ord + PartialOrd + Eq + PartialEq + Hash> From<(&T, &T)>
    for UnorderedPair<T>
{
    fn from(value: (&T, &T)) -> Self {
        Self::new(*value.0, *value.1)
    }
}

pub trait Nonempty: Sized {
    type Inner: IntoIterator;

    fn iter(&self) -> impl Iterator<Item = &<Self::Inner as IntoIterator>::Item>;
    fn min(&self) -> &<Self::Inner as IntoIterator>::Item
    where
        <Self::Inner as IntoIterator>::Item: Ord,
    {
        unsafe { self.iter().min().unwrap_unchecked() }
    }
    fn max(&self) -> &<Self::Inner as IntoIterator>::Item
    where
        <Self::Inner as IntoIterator>::Item: Ord,
    {
        unsafe { self.iter().max().unwrap_unchecked() }
    }
}

pub struct NonemptyVec<T> {
    inner: Vec<T>,
}

impl<T> NonemptyVec<T> {
    pub fn is_empty(&self) -> bool {
        false
    }
    pub fn len(&self) -> usize {
        self.inner.len()
    }

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
        I: Iterator<Item = T>,
    {
        Self::try_from_vec(iter.collect_vec())
    }
}

impl<T> From<NonemptyVec<T>> for Vec<T> {
    fn from(value: NonemptyVec<T>) -> Self {
        value.inner
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

    fn iter(&self) -> impl Iterator<Item = &<Self::Inner as IntoIterator>::Item> {
        self.inner.iter()
    }
}

pub trait IntoVec<T> {
    fn into_vec(self) -> Vec<T>;
}

impl<T> IntoVec<T> for Option<NonemptyVec<T>> {
    fn into_vec(self) -> Vec<T> {
        match self {
            None => Vec::new(),
            Some(v) => v.inner,
        }
    }
}

pub trait IntoFlatIter<T> {
    fn into_flat_iter(self) -> IntoIter<T>;
}

impl<T> IntoFlatIter<T> for Option<NonemptyVec<T>> {
    fn into_flat_iter(self) -> IntoIter<T> {
        match self {
            None => Vec::new(),
            Some(v) => v.inner,
        }
        .into_iter()
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

impl<T> Nonempty for NonemptyVecRefMut<'_, T> {
    type Inner = Vec<T>;

    fn iter(&self) -> impl Iterator<Item = &<Self::Inner as IntoIterator>::Item> {
        self.inner.iter()
    }
}

impl<T> Deref for NonemptyVecRefMut<'_, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.inner
    }
}

pub struct UniqueShared<T: ?Sized> {
    inner: Arc<Mutex<T>>,
}

unsafe impl<T: ?Sized + Send> Send for UniqueShared<T> {}
unsafe impl<T: ?Sized + Send> Sync for UniqueShared<T> {}

// #[derive(Clone)] does not respect ?Sized.
impl<T: ?Sized> Clone for UniqueShared<T> {
    fn clone(&self) -> Self {
        UniqueShared {
            inner: self.inner.clone(),
        }
    }
}

impl<T> UniqueShared<T> {
    pub fn new(value: T) -> Self {
        Self {
            inner: Arc::new(Mutex::new(value)),
        }
    }
}
impl<T: Clone> UniqueShared<T> {
    pub fn new_from_ref(value: &T) -> Self {
        Self {
            inner: Arc::new(Mutex::new(value.clone())),
        }
    }
}

impl<T: ?Sized> UniqueShared<T> {
    pub fn get(&self) -> MutexGuard<'_, T> {
        self.inner
            .try_lock()
            .expect("attempted to acquire UniqueShared but it was already in use")
    }
}

impl<T: Clone> UniqueShared<T> {
    pub fn clone_inner(&self) -> T {
        self.get().clone()
    }
}

impl<T> UniqueShared<Option<T>> {
    pub fn take_inner(&self) -> Option<T> {
        self.get().take()
    }
}

impl<T: Debug> Debug for UniqueShared<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "UniqueShared[{:?}]", self.get())
    }
}

impl<T: Display> Display for UniqueShared<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "UniqueShared[{}]", self.get())
    }
}

impl<T: Default> Default for UniqueShared<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

fn setup_log() -> Result<()> {
    let logfile = std::fs::OpenOptions::new()
        .write(true)
        .truncate(true)
        .create(true)
        .open("run.log")?;
    let timer = OffsetTime::new(
        time::UtcOffset::UTC,
        time::macros::format_description!("[hour]:[minute]:[second].[subsecond digits:6]"),
    );
    tracing_subscriber::fmt()
        .event_format(
            tracing_subscriber::fmt::format()
                .with_target(false)
                .with_source_location(true)
                .with_timer(timer),
        )
        .with_writer(logfile)
        .init();
    Ok(())
}

pub struct GgContextBuilder {
    window_size: Vec2i,
    gui_ctx: GuiContext,
    global_scale_factor: f32,
    clear_col: Colour,
}

impl GgContextBuilder {
    pub fn new(window_size: impl Into<Vec2i>) -> Result<Self> {
        setup_log()?;
        let gui_ctx = GuiContext::new();
        Ok(Self {
            window_size: window_size.into(),
            gui_ctx,
            global_scale_factor: 1.0,
            clear_col: Colour::black(),
        })
    }

    // #[must_use]
    // pub fn with_extra_shaders(
    //     mut self,
    //     create_shaders: impl FnOnce(&GgContextBuilder) -> Vec<UniqueShared<Box<dyn Shader>>>
    // ) -> Self {
    //     self.shaders.extend(create_shaders(&self));
    //     self
    // }
    #[must_use]
    pub fn with_global_scale_factor(mut self, global_scale_factor: f32) -> Self {
        self.global_scale_factor = global_scale_factor;
        self
    }
    #[must_use]
    pub fn with_clear_col(mut self, clear_col: Colour) -> Self {
        self.clear_col = clear_col;
        self
    }

    pub fn build_and_run_window<F>(self, create_and_start_scene_handler: F) -> Result<()>
    where
        F: FnOnce(SceneHandlerBuilder) -> SceneHandler + Send + 'static,
    {
        WindowEventHandler::create_and_run(
            self.window_size,
            self.global_scale_factor,
            self.clear_col,
            self.gui_ctx,
            create_and_start_scene_handler,
        )
    }
}

pub struct SceneHandlerBuilder {
    input_handler: Arc<Mutex<InputHandler>>,
    render_handler: RenderHandler,
}

impl SceneHandlerBuilder {
    pub(crate) fn new(
        input_handler: Arc<Mutex<InputHandler>>,
        render_handler: RenderHandler,
    ) -> Self {
        Self {
            input_handler,
            render_handler,
        }
    }

    pub fn resource_handler(&self) -> &ResourceHandler {
        &self.render_handler.resource_handler
    }

    pub fn build(self) -> SceneHandler {
        SceneHandler::new(self.input_handler, self.render_handler)
    }
}

// Just to avoid having to write .into_wrapper() at the end of each line.
#[macro_export]
macro_rules! scene_object_vec {
    ($($x:expr),* $(,)?) => {
        vec![$(
           $x.into_wrapper()
        ),*]
    };
}

pub(crate) struct ValueChannel<T: Clone> {
    value: T,
    tx: Sender<T>,
    rx: Receiver<T>,
}

impl<T: Clone> ValueChannel<T> {
    pub fn with_value(value: T) -> Self {
        let (tx, rx) = mpsc::channel();
        Self { value, tx, rx }
    }

    pub fn overwrite(&mut self, value: T) {
        self.value = value;
    }

    pub fn try_recv_and_update_cloned(&mut self) -> Option<T> {
        self.value = self.rx.try_iter().last()?;
        Some(self.value.clone())
    }

    pub fn sender(&self) -> ValueChannelSender<T> {
        ValueChannelSender {
            value: self.value.clone(),
            tx: self.tx.clone(),
        }
    }

    pub fn send(&self, value: T) {
        self.tx.send(value).unwrap();
    }
}
impl<T: Copy> ValueChannel<T> {
    pub fn get(&self) -> T {
        self.value
    }

    pub fn try_recv_and_update(&mut self) -> Option<T> {
        self.value = self.rx.try_iter().last()?;
        Some(self.value)
    }
}

impl<T: Clone + Default> Default for ValueChannel<T> {
    fn default() -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            value: Default::default(),
            tx,
            rx,
        }
    }
}

#[derive(Clone)]
pub(crate) struct ValueChannelSender<T: Clone> {
    value: T,
    tx: Sender<T>,
}

impl<T: Clone> ValueChannelSender<T> {
    pub fn get_cloned(&self) -> T {
        self.value.clone()
    }
    pub fn get_ref(&self) -> &T {
        &self.value
    }

    pub fn send_cloned(&self, value: &T) {
        self.tx.send(value.clone()).unwrap();
    }
}

impl<T: Copy> ValueChannelSender<T> {
    pub fn get(&self) -> T {
        self.value
    }

    pub fn send(&self, value: T) {
        self.tx.send(value).unwrap();
    }
}

impl ValueChannelSender<bool> {
    pub fn toggle(&self) {
        self.send(!self.get());
    }

    pub fn add_as_button(&self, ui: &mut GuiUi, text: impl Into<WidgetText>) {
        if ui.add(Button::new(text).selected(self.get())).clicked() {
            self.toggle();
        }
    }
}

pub trait InspectMut<T> {
    #[allow(clippy::return_self_not_must_use)]
    fn inspect_mut<F>(self, f: F) -> Self
    where
        F: FnOnce(&mut T);
}

impl<T> InspectMut<T> for Option<T> {
    fn inspect_mut<F>(mut self, f: F) -> Self
    where
        F: FnOnce(&mut T),
    {
        if let Some(ref mut value) = self {
            f(value);
        }
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OrElse {
    Done,
    Continue,
}
impl OrElse {
    // Intentionally allowed to end a chain of OrElse.
    #[allow(clippy::return_self_not_must_use)]
    pub fn or_else(self, mut f: impl FnMut() -> OrElse) -> OrElse {
        match self {
            OrElse::Done => OrElse::Done,
            OrElse::Continue => f(),
        }
    }
}
