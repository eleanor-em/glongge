use crate::core::prelude::*;

use std::{
    hash::Hash,
    ops::Deref,
    vec::IntoIter
};
use std::fmt::{Debug, Display, Formatter};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex, MutexGuard};
use tracing_subscriber::fmt::time::OffsetTime;
use crate::core::input::InputHandler;
use crate::core::ObjectTypeEnum;
use crate::core::render::RenderHandler;
use crate::core::scene::SceneHandler;
use crate::core::vk::{AdjustedViewport, VulkanoContext, WindowContext, WindowEventHandler};
use crate::gui::GuiContext;
use crate::shader::{BasicShader, Shader, SpriteShader, WireframeShader};

pub mod linalg;
pub mod colour;
pub mod assert;
pub mod collision;
pub mod canvas;
pub mod tileset;

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

    #[derive(Clone)]
    pub struct TimeIt {
        tag: String,
        n: u128,
        total_ns: u128,
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
                self.max_ns = cmp::max(self.max_ns, self.last_ns);
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
        fn max_ms(&self) -> f64 {
            #[allow(clippy::cast_precision_loss)]
            let max_ns = self.max_ns as f64;
            max_ns / 1_000_000.
        }
        pub fn as_tuple_ms(&self) -> (String, f64, f64) {
            (self.tag.clone(), self.mean_ms(), self.max_ms())
        }
        #[must_use]
        pub fn report_take(&mut self) -> TimeIt {
            let rv = self.clone();
            self.reset();
            rv
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
    use std::cmp;
    use std::ops::Add;

    pub fn sum_tuple3<T: Add<Output=T>>(acc: (T, T, T), x: (T, T, T)) -> (T, T, T) {
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
    where F: FnMut(&T) -> cmp::Ordering
    {
        slice.partition_point(|x| (comparator)(x) != cmp::Ordering::Greater)
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

#[allow(dead_code)]
pub mod gg_err {
    use anyhow::Result;
    use tracing::{error, warn};

    pub fn is_some_and_warn<T>(result: Result<Option<T>>) -> bool {
        match result {
            Ok(Some(_)) => true,
            Ok(None) => false,
            Err(e) => {
                warn!("{}", e.root_cause());
                false
            }
        }
    }

    pub fn warn_err_then<T>(result: Result<Option<T>>) -> Option<T> {
        match result {
            Ok(o) => o,
            Err(e) => {
                warn!("{}", e.root_cause());
                None
            }
        }
    }

    pub fn warn_unwrap_or<T, U: Into<T>>(default: U, result: Result<T>) -> T {
        match result {
            Ok(v) => v,
            Err(e) => {
                warn!("{}", e.root_cause());
                default.into()
            }
        }
    }

    pub fn warn_and_ok<T>(result: Result<T>) -> Option<T> {
        match result {
            Ok(v) => Some(v),
            Err(e) => {
                warn!("{}", e.root_cause());
                None
            }
        }
    }
    pub fn warn_err(result: Result<()>) {
        if let Err(e) = result {
            warn!("{}", e.root_cause());
        }
    }
    pub fn is_some_and_log<T>(result: Result<Option<T>>) -> bool {
        match result {
            Ok(Some(_)) => true,
            Ok(None) => false,
            Err(e) => {
                error!("{}", e.root_cause());
                false
            }
        }
    }

    pub fn log_err_then<T>(result: Result<Option<T>>) -> Option<T> {
        match result {
            Ok(o) => o,
            Err(e) => {
                error!("{}", e.root_cause());
                None
            }
        }
    }

    pub fn log_unwrap_or<T, U: Into<T>>(default: U, result: Result<T>) -> T {
        match result {
            Ok(v) => v,
            Err(e) => {
                error!("{}", e.root_cause());
                default.into()
            }
        }
    }

    pub fn log_and_ok<T>(result: Result<T>) -> Option<T> {
        match result {
            Ok(v) => Some(v),
            Err(e) => {
                error!("{}", e.root_cause());
                None
            }
        }
    }
    pub fn log_err(result: Result<()>) {
        if let Err(e) = result {
            error!("{}", e.root_cause());
        }
    }
}

pub mod gg_float {
    use std::num::FpCategory;
    use num_traits::Zero;
    use crate::util::linalg::{Transform, Vec2};

    pub trait GgFloat{
        fn is_normal_or_zero(&self) -> bool;
    }

    impl GgFloat for f64 {
        fn is_normal_or_zero(&self) -> bool {
            self.is_normal() || self.is_zero()
        }
    }

    impl GgFloat for Vec2 {
        fn is_normal_or_zero(&self) -> bool {
            self.x.is_normal_or_zero() && self.y.is_normal_or_zero()
        }
    }

    impl GgFloat for Transform {
        fn is_normal_or_zero(&self) -> bool {
            self.centre.is_normal_or_zero() &&
                self.rotation.is_normal_or_zero() &&
                self.scale.is_normal_or_zero()
        }
    }
    pub fn is_normal_or_zero(x: f64) -> bool {
        match x.classify() {
            FpCategory::Zero | FpCategory::Normal => true,
            _ => false
        }
    }
}

pub mod gg_range {
    use std::ops::Range;
    use crate::core::config::EPSILON;

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
        if (start - end).abs() < EPSILON {
            None
        } else {
            Some(start..end)
        }
    }

    pub fn overlap_len_f64(r1: &Range<f64>, r2: &Range<f64>) -> Option<f64> {
        overlap_f64(r1, r2).map(|r| r.end - r.start)
    }
}

#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
pub struct UnorderedPair<T: Copy + Clone + Ord + PartialOrd + Eq + PartialEq>(T, T);
impl<T: Copy + Clone + Ord + PartialOrd + Eq + PartialEq> UnorderedPair<T> {
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

pub struct UniqueShared<T: ?Sized> {
    inner: Arc<Mutex<T>>,
}


unsafe impl<T: ?Sized + Send> Send for UniqueShared<T> {}
unsafe impl<T: ?Sized + Send> Sync for UniqueShared<T> {}

// #[derive(Clone)] does not respect ?Sized.
impl<T: ?Sized> Clone for UniqueShared<T> {
    fn clone(&self) -> Self {
        UniqueShared { inner: self.inner.clone() }
    }
}

impl<T> UniqueShared<T> {
    pub fn new(value: T) -> Self {
        Self { inner: Arc::new(Mutex::new(value)) }
    }
}
impl<T: Clone> UniqueShared<T> {
    pub fn new_from_ref(value: &T) -> Self {
        Self { inner: Arc::new(Mutex::new(value.clone())) }
    }
}

impl<T: ?Sized> UniqueShared<T> {
    pub fn get(&self) -> MutexGuard<T> {
        self.inner.try_lock()
            .expect("attempted to acquire UniqueShared but it was already in use")
    }
}

impl<T: Clone> UniqueShared<T> {
    pub fn clone_inner(&self) -> T { self.get().clone() }
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

fn setup_log() -> Result<()> {
    let logfile = std::fs::OpenOptions::new()
        .write(true)
        .truncate(true)
        .create(true)
        .open("run.log")?;
    let timer = OffsetTime::new(
        time::UtcOffset::UTC,
        time::macros::format_description!("[hour]:[minute]:[second].[subsecond digits:6]")
    );
    tracing_subscriber::fmt()
        .event_format(
            tracing_subscriber::fmt::format()
                .with_target(false)
                .with_file(true)
                .with_line_number(true)
                .with_timer(timer)
        )
        .with_writer(logfile)
        .init();
    Ok(())
}

pub struct GgContextBuilder<ObjectType: ObjectTypeEnum> {
    window_ctx: WindowContext,
    vk_ctx: VulkanoContext,
    gui_ctx: GuiContext,
    resource_handler: ResourceHandler,
    viewport: UniqueShared<AdjustedViewport>,
    shaders: Vec<UniqueShared<Box<dyn Shader>>>,
    global_scale_factor: f64,
    object_type: PhantomData<ObjectType>
}

impl<ObjectType: ObjectTypeEnum> GgContextBuilder<ObjectType> {
    pub fn new(
        window_size: impl Into<Vec2i>,
    ) -> Result<Self> {
        setup_log()?;
        let window_ctx = WindowContext::new(window_size.into())?;
        let gui_ctx = GuiContext::default();
        let vk_ctx = VulkanoContext::new(&window_ctx)?;
        let mut resource_handler = ResourceHandler::new(&vk_ctx)?;
        ObjectType::preload_all(&mut resource_handler)?;
        let viewport = window_ctx.create_default_viewport();

        let shaders: Vec<UniqueShared<Box<dyn Shader>>> = vec![
            SpriteShader::new(vk_ctx.clone(), viewport.clone(), resource_handler.clone())?,
            WireframeShader::new(vk_ctx.clone(), viewport.clone())?,
            BasicShader::new(vk_ctx.clone(), viewport.clone())?,
        ];
        Ok(Self {
            window_ctx,
            vk_ctx,
            gui_ctx,
            resource_handler,
            viewport,
            shaders,
            global_scale_factor: 1.,
            object_type: PhantomData
        })
    }

    #[must_use]
    pub fn with_extra_shaders(
        mut self,
        create_shaders: impl FnOnce(&GgContextBuilder<ObjectType>) -> Vec<UniqueShared<Box<dyn Shader>>>
    ) -> Self {
        self.shaders.extend(create_shaders(&self));
        self
    }
    #[must_use]
    pub fn with_global_scale_factor(mut self, global_scale_factor: f64) -> Self {
        self.global_scale_factor = global_scale_factor;
        self
    }

    pub fn build(self) -> Result<GgContext<ObjectType>> {
        let input_handler = InputHandler::new();
        let render_handler = RenderHandler::new(
            &self.vk_ctx,
            self.gui_ctx.clone(),
            self.viewport.clone(),
            self.shaders,
        )?
            .with_global_scale_factor(self.global_scale_factor);
        Ok(GgContext {
            window_ctx: self.window_ctx,
            vk_ctx: self.vk_ctx,
            gui_ctx: self.gui_ctx,
            resource_handler: self.resource_handler,
            viewport: self.viewport,
            input_handler,
            render_handler,
            object_type: self.object_type
        })
    }
}

#[allow(dead_code)]
pub struct GgContext<ObjectType: ObjectTypeEnum> {
    window_ctx: WindowContext,
    vk_ctx: VulkanoContext,
    gui_ctx: GuiContext,
    resource_handler: ResourceHandler,
    viewport: UniqueShared<AdjustedViewport>,
    input_handler: Arc<Mutex<InputHandler>>,
    render_handler: RenderHandler,
    object_type: PhantomData<ObjectType>
}

impl<ObjectType: ObjectTypeEnum> GgContext<ObjectType> {
    pub fn scene_handler(&self) -> SceneHandler<ObjectType> {
        SceneHandler::new(
            self.input_handler.clone(),
            self.resource_handler.clone(),
            self.render_handler.clone()
        )
    }

    pub fn consume_run_window(self) {
        let (event_loop, window) = self.window_ctx.consume();
        WindowEventHandler::new(window, self.vk_ctx, self.gui_ctx, self.render_handler, self.input_handler, self.resource_handler)
            .consume(event_loop);
    }
}
