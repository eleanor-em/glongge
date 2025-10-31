use std::borrow::Cow;
// tulivuori
use crate::core::prelude::*;
use crate::core::render::RenderHandler;
use crate::core::scene::SceneHandler;
use crate::gui::GuiContext;
use crate::resource::ResourceHandler;
use crate::util::{SceneHandlerBuilder, UniqueShared, gg_float};
use crate::{info_every_seconds, util::gg_time::TimeIt, warn_every_seconds};
use anyhow::{Context, Result};
use ash::ext::debug_utils;
use ash::khr::surface;
use ash::{Device, Entry, Instance, vk};
use egui::ViewportId;
use egui_winit::winit::application::ApplicationHandler;
use egui_winit::winit::dpi::PhysicalSize;
use egui_winit::winit::event_loop::ActiveEventLoop;
use egui_winit::winit::keyboard::PhysicalKey;
use egui_winit::winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use egui_winit::winit::window::{Window, WindowAttributes, WindowId};
use egui_winit::winit::{dpi::LogicalSize, event::WindowEvent, event_loop::EventLoop};
use std::ffi::CString;
use std::str::FromStr;
use std::sync::mpsc;
use std::sync::mpsc::{Receiver, Sender};
use std::time::SystemTime;
use std::{
    ffi,
    sync::{Arc, Mutex},
    time::Instant,
};
use tracing::{error, info_span, warn};

#[derive(Clone)]
pub(crate) struct GgWindow {
    pub(crate) inner: Arc<Window>,
    refresh_time: f32,
}

impl GgWindow {
    pub(crate) fn new(event_loop: &ActiveEventLoop, size: impl Into<Vec2i>) -> Result<Self> {
        let size = size.into();
        let mut window_attrs = WindowAttributes::default();
        // TODO: allow setting window title.
        window_attrs.title = "glongge".to_string();
        window_attrs.resizable = true;
        window_attrs.inner_size = Some(egui_winit::winit::dpi::Size::Logical(LogicalSize::new(
            f64::from(size.x),
            f64::from(size.y),
        )));
        let window = Arc::new(event_loop.create_window(window_attrs)?);
        let refresh_time = window
            .current_monitor()
            .and_then(|m| m.refresh_rate_millihertz())
            .map_or_else(
                || {
                    warn!("failed to determine refresh rate, assuming 60 Hz");
                    1_000.0 / 60.0
                },
                |r| {
                    let refresh_time = 1_000_000.0 / r as f32;
                    info!("refresh every: {refresh_time:.2} ms");
                    refresh_time
                },
            );
        check_gt!(refresh_time, 1.0);
        check_lt!(refresh_time, 1_000.0 / 30.0);
        Ok(Self {
            inner: window,
            refresh_time,
        })
    }

    pub(crate) fn inner_size(&self) -> PhysicalSize<u32> {
        self.inner.inner_size()
    }
    pub(crate) fn winit_scale_factor(&self) -> f32 {
        self.inner.scale_factor() as f32
    }
}

#[derive(Clone)]
pub(crate) struct GgViewport {
    window: GgWindow,
    inner: vk::Viewport,
    winit_scale_factor: f32,
    extra_scale_factor: f32,
}

impl GgViewport {
    pub(crate) fn new(window: &GgWindow) -> Self {
        Self {
            window: window.clone(),
            inner: vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: window.inner_size().width as f32,
                height: window.inner_size().height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            },
            winit_scale_factor: window.winit_scale_factor(),
            extra_scale_factor: 1.0,
        }
    }

    pub(crate) fn physical_width(&self) -> f32 {
        self.inner.width
    }
    pub(crate) fn physical_height(&self) -> f32 {
        self.inner.height
    }
    pub(crate) fn physical_left(&self) -> f32 {
        self.inner.x
    }
    pub(crate) fn physical_top(&self) -> f32 {
        self.inner.y
    }
    pub(crate) fn physical_top_left(&self) -> Vec2 {
        Vec2 {
            x: self.physical_left(),
            y: self.physical_top(),
        }
    }
    pub(crate) fn logical_left(&self) -> f32 {
        self.inner.x / self.winit_scale_factor
    }
    pub(crate) fn logical_top(&self) -> f32 {
        self.inner.y / self.winit_scale_factor
    }
    pub(crate) fn world_left(&self) -> f32 {
        self.inner.x / self.combined_scale_factor()
    }
    pub(crate) fn world_top(&self) -> f32 {
        self.inner.y / self.combined_scale_factor()
    }
    pub(crate) fn world_top_left(&self) -> Vec2 {
        Vec2 {
            x: self.world_left(),
            y: self.world_top(),
        }
    }

    pub(crate) fn combined_scale_factor(&self) -> f32 {
        self.winit_scale_factor * self.extra_scale_factor
    }
    pub(crate) fn set_extra_scale_factor(&mut self, extra_scale_factor: f32) {
        self.extra_scale_factor = extra_scale_factor;
    }
    pub(crate) fn extra_scale_factor(&self) -> f32 {
        self.extra_scale_factor
    }
    pub(crate) fn refresh_time(&self) -> f32 {
        self.window.refresh_time
    }

    pub(crate) fn set_physical_top_left(&mut self, top_left: Vec2) {
        self.inner.x = top_left.x;
        self.inner.y = top_left.y;
    }
}

#[allow(unused)]
struct WindowEventHandlerInner {
    window: GgWindow,
    platform: egui_winit::State,
    winit_scale_factor_for_logging: f32,
    input_handler: Arc<Mutex<InputHandler>>,
    render_stats: RenderPerfStats,
    last_render_stats: Option<RenderPerfStats>,
    is_first_window_event: bool,

    window_event_rx: Receiver<WindowEvent>,
    scale_factor_rx: Receiver<f32>,
    recreate_swapchain_rx: Receiver<Instant>,
    render_handler: RenderHandler,
    render_count: usize,
}
impl WindowEventHandlerInner {
    fn handle_window_events(&mut self) {
        while let Ok(event) = self.window_event_rx.try_recv() {
            let _response = self.platform.on_window_event(&self.window.inner, &event);
        }
        if let Some(new_scale_factor) = self.scale_factor_rx.try_iter().last() {
            // Since scale_factor is given by winit, we expect an exact comparison to work.
            #[allow(clippy::float_cmp)]
            if self.winit_scale_factor_for_logging != new_scale_factor {
                // TODO: verbose_every_seconds!
                info_every_seconds!(
                    1,
                    "WindowEvent::ScaleFactorChanged: {} -> {}: recreating swapchain",
                    self.winit_scale_factor_for_logging,
                    new_scale_factor
                );
                self.winit_scale_factor_for_logging = new_scale_factor;
                // self.recreate_swapchain().unwrap();
            }
        }
        if let Some(request_time) = self.recreate_swapchain_rx.try_iter().last() {
            info!(
                "recreating swapchain: {:.2} ms old",
                request_time.elapsed().as_micros() as f32 / 1000.0
            );
            // self.recreate_swapchain().unwrap();
        }
    }

    fn render_update(&mut self) {
        let n = self.render_count;
        let span = info_span!("render_update", n);
        let _enter = span.enter();
        self.render_handler.wait_update_done();
        self.handle_window_events();

        self.render_handler.render_update().unwrap();
        self.render_count += 1;
    }
}

struct WindowEventHandlerCreateInfo<F>
where
    F: FnOnce(SceneHandlerBuilder) -> SceneHandler + Send + 'static,
{
    window_size: Vec2i,
    scene_handler_builder_callback: Option<F>,
    extra_scale_factor: f32,
    clear_col: Colour,
}

pub(crate) struct WindowEventHandler<F>
where
    F: FnOnce(SceneHandlerBuilder) -> SceneHandler + Send + 'static,
{
    create_info: WindowEventHandlerCreateInfo<F>,
    input_handler: Arc<Mutex<InputHandler>>,

    gui_ctx: Option<GuiContext>,

    window_event_tx: Sender<WindowEvent>,
    window_event_rx: Option<Receiver<WindowEvent>>,
    scale_factor_tx: Sender<f32>,
    scale_factor_rx: Option<Receiver<f32>>,
    recreate_swapchain_tx: Sender<Instant>,
    recreate_swapchain_rx: Option<Receiver<Instant>>,
}

impl<SceneHandlerBuilderCallback> WindowEventHandler<SceneHandlerBuilderCallback>
where
    SceneHandlerBuilderCallback: FnOnce(SceneHandlerBuilder) -> SceneHandler + Send + 'static,
{
    pub(crate) fn create_and_run(
        window_size: Vec2i,
        extra_scale_factor: f32,
        clear_col: Colour,
        gui_ctx: GuiContext,
        scene_handler_builder_callback: SceneHandlerBuilderCallback,
    ) -> Result<()> {
        let (window_event_tx, window_event_rx) = mpsc::channel();
        let (scale_factor_tx, scale_factor_rx) = mpsc::channel();
        let (recreate_swapchain_tx, recreate_swapchain_rx) = mpsc::channel();
        let mut this = Self {
            create_info: WindowEventHandlerCreateInfo {
                window_size,
                extra_scale_factor,
                clear_col,
                scene_handler_builder_callback: Some(scene_handler_builder_callback),
            },
            input_handler: InputHandler::new(),
            gui_ctx: Some(gui_ctx),
            window_event_tx,
            window_event_rx: Some(window_event_rx),
            scale_factor_tx,
            scale_factor_rx: Some(scale_factor_rx),
            recreate_swapchain_tx,
            recreate_swapchain_rx: Some(recreate_swapchain_rx),
        };

        let event_loop = EventLoop::new()?;
        Ok(event_loop.run_app(&mut this)?)
    }

    fn create_inner2(
        &mut self,
        event_loop: &ActiveEventLoop,
        scene_handler_builder_callback: SceneHandlerBuilderCallback,
    ) -> Result<()> {
        let window = GgWindow::new(event_loop, self.create_info.window_size)?;
        let ctx = TvWindowContextBuilder::new()
            .with_app_name(CString::from_str("ash-noodling").unwrap())
            .with_flag_debug_tools()
            .with_flag_validation_layers()
            .with_flag_verbose_logging()
            .build(&window.inner)?;
        let platform = egui_winit::State::new(
            self.gui_ctx.clone().unwrap().inner.clone(),
            ViewportId::ROOT,
            &event_loop,
            Some(window.winit_scale_factor()),
            None,
            None,
        );

        let input_handler = self.input_handler.clone();
        let window_event_rx = self.window_event_rx.take().unwrap();
        let scale_factor_rx = self.scale_factor_rx.take().unwrap();
        let recreate_swapchain_rx = self.recreate_swapchain_rx.take().unwrap();
        let resource_handler = ResourceHandler::new(&ctx)?;
        let viewport = UniqueShared::new(GgViewport::new(&window));
        let render_handler = RenderHandler::new(
            ctx.clone(),
            self.gui_ctx.clone().unwrap(),
            window.clone(),
            viewport,
            resource_handler,
        )?
        .with_extra_scale_factor(self.create_info.extra_scale_factor)
        .with_clear_col(self.create_info.clear_col);
        let scene_handler_builder =
            SceneHandlerBuilder::new(input_handler.clone(), render_handler.clone());
        std::thread::spawn(move || {
            let mut scene_handler = scene_handler_builder_callback(scene_handler_builder);
            loop {
                scene_handler.run_update();
            }
        });
        let winit_scale_factor_for_logging = window.winit_scale_factor();
        std::thread::spawn(move || {
            let render_stats = RenderPerfStats::new(&window);
            let mut inner = WindowEventHandlerInner {
                window,
                platform,
                winit_scale_factor_for_logging,
                input_handler,
                render_stats,
                last_render_stats: None,
                is_first_window_event: true,
                window_event_rx,
                scale_factor_rx,
                recreate_swapchain_rx,
                render_handler,
                render_count: 0,
            };
            loop {
                inner.render_update();
            }
        });
        Ok(())
    }
}

impl<F> ApplicationHandler for WindowEventHandler<F>
where
    F: FnOnce(SceneHandlerBuilder) -> SceneHandler + Send + 'static,
{
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(scene_handler_builder_callback) =
            self.create_info.scene_handler_builder_callback.take()
        {
            // First event. Note winit documentation:
            // "This is a common indicator that you can create a window."
            self.create_inner2(event_loop, scene_handler_builder_callback)
                .expect("error initialising");
        }
    }

    fn window_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        self.window_event_tx.send(event.clone()).unwrap();
        match event {
            WindowEvent::CloseRequested => {
                info!("received WindowEvent::CloseRequested, calling exit(0)");
                std::process::exit(0);
            }
            WindowEvent::KeyboardInput { event, .. } => match event.physical_key {
                PhysicalKey::Code(keycode) => {
                    self.input_handler
                        .lock()
                        .unwrap()
                        .queue_key_event(keycode, event.state);
                }
                PhysicalKey::Unidentified(keycode) => {
                    info!("PhysicalKey::Unidentified({keycode:?}), ignoring");
                }
            },
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                self.scale_factor_tx.send(scale_factor as f32).unwrap();
            }
            WindowEvent::Resized(physical_size) => {
                // TODO: verbose_every_seconds!
                info_every_seconds!(
                    1,
                    "WindowEvent::Resized: {:?}: recreating swapchain",
                    physical_size
                );
                self.recreate_swapchain_tx.send(Instant::now()).unwrap();
            }
            WindowEvent::RedrawRequested => {
                // self.expect_inner().run_update();
                // self.expect_inner().window.inner.request_redraw();
            }
            _other_event => {}
        }
    }
}

#[allow(unused)]
#[derive(Clone)]
pub(crate) struct RenderPerfStats {
    update_gui: TimeIt,
    execute: TimeIt,
    between_renders: TimeIt,
    extra_debug: TimeIt,

    total: TimeIt,
    on_time: u64,
    count: u64,
    penultimate_step: Instant,
    last_step: Instant,
    last_report: Instant,
    totals_ms: Vec<f32>,

    refresh_time: f32,

    last_perf_stats: Option<Box<RenderPerfStats>>,
}

impl RenderPerfStats {
    fn new(window: &GgWindow) -> Self {
        Self {
            update_gui: TimeIt::new("update_gui"),
            execute: TimeIt::new("execute"),
            between_renders: TimeIt::new("between renders"),
            extra_debug: TimeIt::new("extra_debug"),
            total: TimeIt::new("total"),
            on_time: 0,
            count: 0,
            penultimate_step: Instant::now(),
            last_step: Instant::now(),
            last_report: Instant::now(),
            totals_ms: Vec::with_capacity(10),
            refresh_time: window.refresh_time,
            last_perf_stats: None,
        }
    }

    #[allow(unused)]
    fn start(&mut self) {
        self.between_renders.stop();
    }

    #[allow(unused)]
    fn end(&mut self) -> Option<Self> {
        // Allow a bit of slack.
        let deadline_ms = self.refresh_time * 1.2;

        self.between_renders.start();

        if self.totals_ms.len() == self.totals_ms.capacity() {
            self.totals_ms.remove(0);
        }
        let render_time = gg_float::micros(self.last_step.elapsed()) * 1000.0;
        self.totals_ms.push(render_time);

        let late_in_row = self
            .totals_ms
            .iter()
            .rev()
            .take_while(|&&t| t > deadline_ms)
            .collect_vec();
        if late_in_row.len() > 1 {
            let mut msg = format!("{} frames late in a row: ", late_in_row.len());
            for time in &late_in_row[..late_in_row.len() - 1] {
                msg += format!("{time:.1}, ").as_str();
            }
            msg += format!("{:.1}", late_in_row.last().unwrap()).as_str();
            warn_every_seconds!(1, "{msg}");
        }
        if render_time <= deadline_ms {
            self.on_time += 1;
        }
        self.count += 1;

        self.total.stop();
        self.total.start();
        self.penultimate_step = self.last_step;
        self.last_step = Instant::now();

        if self.last_report.elapsed().as_secs() >= 2 {
            #[allow(clippy::cast_precision_loss)]
            let on_time_rate = self.on_time as f32 / self.count as f32 * 100.0;
            if on_time_rate.round() < 100.0 {
                // TODO: this calculation is dubious, improve.
                warn!("frames on time: {on_time_rate:.1}%");
            }
            self.last_perf_stats = Some(Box::new(Self {
                update_gui: self.update_gui.report_take(),
                execute: self.execute.report_take(),
                between_renders: self.between_renders.report_take(),
                extra_debug: self.extra_debug.report_take(),
                total: self.total.report_take(),
                on_time: 0,
                count: 0,
                last_perf_stats: None,
                last_report: Instant::now(),
                penultimate_step: self.penultimate_step,
                last_step: self.last_step,
                totals_ms: vec![],
                refresh_time: self.refresh_time,
            }));
            self.last_report = Instant::now();
            self.last_report = Instant::now();
            self.on_time = 0;
            self.count = 0;
        }

        self.last_perf_stats.clone().map(|s| *s)
    }

    pub(crate) fn as_tuples_ms(&self) -> Vec<(String, f32, f32, f32)> {
        let mut default = vec![
            self.total.as_tuple_ms(),
            self.update_gui.as_tuple_ms(),
            self.execute.as_tuple_ms(),
            self.between_renders.as_tuple_ms(),
        ];
        if self.extra_debug.last_ms() != 0.0 {
            default.push(self.extra_debug.as_tuple_ms());
        }
        default
    }
}

pub mod buffer;
pub mod pipeline;
pub mod shader;
pub mod swapchain;
pub mod texture;
#[derive(Clone)]
pub struct VulkanPerfStats {
    active: bool,
    stats: Arc<Mutex<Vec<(String, SystemTime)>>>,
}

impl Default for VulkanPerfStats {
    fn default() -> Self {
        Self::new()
    }
}

impl VulkanPerfStats {
    pub fn new() -> Self {
        Self {
            active: false,
            stats: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn lap(&self, name: impl AsRef<str>) {
        if self.active {
            if let Some((_, last)) = self.stats.try_lock().unwrap().last().cloned() {
                check_ge!(SystemTime::now(), last);
            }
            self.stats
                .try_lock()
                .unwrap()
                .push((name.as_ref().to_string(), SystemTime::now()));
        }
    }

    pub fn report(&self, threshold_ms: u128) {
        if self.active {
            let span = info_span!("VulkanPerfStats");
            let _enter = span.enter();
            let stats_ms = self
                .stats
                .try_lock()
                .unwrap()
                .drain(..)
                .tuple_windows()
                .map(|((_, i1), (name, i2))| {
                    (
                        name,
                        i2.duration_since(i1).unwrap().as_micros() as f32 / 1000.0,
                    )
                })
                .collect_vec();
            let total_ms = stats_ms
                .iter()
                .map(|(_, elapsed_ms)| elapsed_ms)
                .sum::<f32>();
            if total_ms >= threshold_ms as f32 {
                for (name, elapsed_ms) in stats_ms {
                    info!("{name}: {:.2} ms", elapsed_ms);
                }
                info!("total: {:.2} ms", total_ms);
            } else {
                info_every_millis!(1000, "{total_ms:.2} ms");
            }
        }
    }
}

pub(crate) struct DebugHandler {
    debug_utils_loader: debug_utils::Instance,
    debug_callback: vk::DebugUtilsMessengerEXT,
}

impl DebugHandler {
    pub fn new(entry: &Entry, instance: &Instance) -> Result<Self> {
        let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(vulkan_debug_callback));
        let debug_utils_loader = debug_utils::Instance::new(entry, instance);
        let debug_callback =
            unsafe { debug_utils_loader.create_debug_utils_messenger(&debug_info, None)? };
        Ok(Self {
            debug_utils_loader,
            debug_callback,
        })
    }
}

impl Drop for DebugHandler {
    fn drop(&mut self) {
        unsafe {
            self.debug_utils_loader
                .destroy_debug_utils_messenger(self.debug_callback, None);
        }
    }
}

/// # Safety: lol
unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let vk_span = info_span!("vulkan");
    let _enter = vk_span.enter();
    let callback_data = unsafe { *p_callback_data };
    let message_id_number = callback_data.message_id_number;

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        Cow::from("")
    } else {
        unsafe { ffi::CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy() }
    };

    let message = if callback_data.p_message.is_null() {
        Cow::from("")
    } else {
        unsafe { ffi::CStr::from_ptr(callback_data.p_message).to_string_lossy() }
    };

    if message_severity == vk::DebugUtilsMessageSeverityFlagsEXT::INFO {
        info!("{message_type:?} [{message_id_name} ({message_id_number})] : {message}",);
    } else if message_severity == vk::DebugUtilsMessageSeverityFlagsEXT::WARNING {
        warn!("{message_type:?} [{message_id_name} ({message_id_number})] : {message}",);
    } else if message_severity == vk::DebugUtilsMessageSeverityFlagsEXT::ERROR {
        error!("{message_type:?} [{message_id_name} ({message_id_number})] : {message}",);
    }

    vk::FALSE
}

pub struct TvWindowContextBuilder {
    app_name: CString,
    instance_extension_names: Vec<*const ffi::c_char>,
    device_extension_names: Vec<*const ffi::c_char>,
    features: vk::PhysicalDeviceFeatures,

    flag_add_validation_layers: bool,
    flag_use_debug_tools: bool,
    flag_verbose_logging: bool,
}

impl Default for TvWindowContextBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl TvWindowContextBuilder {
    pub fn new() -> TvWindowContextBuilder {
        TvWindowContextBuilder {
            app_name: CString::from_str("tulivuori").unwrap(),
            instance_extension_names: Vec::new(),
            device_extension_names: Vec::new(),
            features: vk::PhysicalDeviceFeatures::default(),
            flag_add_validation_layers: false,
            flag_use_debug_tools: false,
            flag_verbose_logging: false,
        }
    }

    #[must_use]
    pub fn with_app_name(mut self, app_name: CString) -> Self {
        self.app_name = app_name;
        self
    }
    #[allow(unused)]
    #[must_use]
    pub fn with_instance_extension(mut self, extension: &ffi::CStr) -> Self {
        self.instance_extension_names.push(extension.as_ptr());
        self
    }
    #[allow(unused)]
    #[must_use]
    pub fn with_device_extension(mut self, device_extension: &ffi::CStr) -> Self {
        self.device_extension_names.push(device_extension.as_ptr());
        self
    }
    #[allow(unused)]
    #[must_use]
    pub fn with_features(mut self, features: vk::PhysicalDeviceFeatures) -> Self {
        self.features = features;
        self
    }
    #[must_use]
    pub fn with_flag_debug_tools(mut self) -> Self {
        self.flag_use_debug_tools = true;
        self
    }
    #[must_use]
    pub fn with_flag_validation_layers(mut self) -> Self {
        self.flag_add_validation_layers = true;
        self
    }
    #[must_use]
    pub fn with_flag_verbose_logging(mut self) -> Self {
        self.flag_verbose_logging = true;
        self
    }

    pub fn build(self, window: &Arc<Window>) -> Result<Arc<TvWindowContext>> {
        let span = info_span!("TvWindowContext");
        let _enter = span.enter();

        let entry = Entry::linked();

        let mut layers_names_raw = Vec::new();
        if self.flag_add_validation_layers {
            layers_names_raw.push(c"VK_LAYER_KHRONOS_validation".as_ptr());
        }

        let mut instance_extension_names = Self::create_min_instance_extension_names(window)?;
        instance_extension_names.extend(self.instance_extension_names);
        if self.flag_verbose_logging {
            info!("create instance");
        }
        let instance = unsafe {
            Self::create_instance(
                &self.app_name,
                &entry,
                &layers_names_raw,
                &instance_extension_names,
            )?
        };

        let debug_handler = if self.flag_use_debug_tools {
            Some(DebugHandler::new(&entry, &instance)?)
        } else {
            None
        };

        if self.flag_verbose_logging {
            info!("create surface");
        }
        let surface = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                window.display_handle()?.as_raw(),
                window.window_handle()?.as_raw(),
                None,
            )?
        };
        if self.flag_verbose_logging {
            info!("create physical device");
        }
        let (surface_loader, physical_device, queue_family_index) =
            unsafe { Self::create_physical_device(&entry, &instance, surface)? };

        if self.flag_verbose_logging {
            info!("create logical device (and present queue)");
        }
        let mut device_extension_names_raw = vec![
            ash::khr::swapchain::NAME.as_ptr(),
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            ash::khr::portability_subset::NAME.as_ptr(),
            // Dynamic rendering:
            ash::khr::dynamic_rendering::NAME.as_ptr(),
        ];
        device_extension_names_raw.extend(self.device_extension_names);
        // TODO: find a way to make these into struct members.
        let v12_features = vk::PhysicalDeviceVulkan12Features {
            descriptor_indexing: 1,
            descriptor_binding_sampled_image_update_after_bind: 1,
            descriptor_binding_storage_buffer_update_after_bind: 1,
            shader_sampled_image_array_non_uniform_indexing: 1,
            ..vk::PhysicalDeviceVulkan12Features::default()
        };
        let v13_features = vk::PhysicalDeviceVulkan13Features {
            dynamic_rendering: 1,
            ..vk::PhysicalDeviceVulkan13Features::default()
        };
        let device = Self::create_device(
            &instance,
            physical_device,
            queue_family_index,
            &device_extension_names_raw,
            &self.features,
            v12_features,
            v13_features,
        )?;
        let present_queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        Ok(Arc::new(TvWindowContext {
            _entry: entry,
            instance,
            debug_handler,
            window: window.clone(),
            surface,
            surface_loader,
            physical_device,
            queue_family_index,
            device,
            present_queue,
        }))
    }

    fn create_min_instance_extension_names(
        window: &Arc<Window>,
    ) -> Result<Vec<*const ffi::c_char>> {
        let mut min_extension_names =
            ash_window::enumerate_required_extensions(window.display_handle()?.as_raw())?.to_vec();
        min_extension_names.push(debug_utils::NAME.as_ptr());

        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            min_extension_names.push(ash::khr::portability_enumeration::NAME.as_ptr());
            // Enabling this extension is a requirement when using `VK_KHR_portability_subset`
            min_extension_names.push(ash::khr::get_physical_device_properties2::NAME.as_ptr());
        }

        Ok(min_extension_names)
    }

    unsafe fn create_instance(
        app_name: &ffi::CStr,
        entry: &Entry,
        layers_names_raw: &[*const ffi::c_char],
        min_extension_names: &[*const ffi::c_char],
    ) -> Result<Instance> {
        if std::env::consts::OS == "macos" {
            let var = match std::env::var("MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS") {
                Ok(var) => var,
                Err(e) => {
                    panic!(
                        "on macOS, environment variable `MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS` must be set; \
                        do you have .cargo/config.toml set up correctly? got: {e:?}"
                    );
                }
            };
            check_eq!(var, "1");
        }
        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name)
            .application_version(0)
            .engine_name(app_name)
            .engine_version(0)
            .api_version(vk::make_api_version(0, 1, 3, 0));
        let create_flags = if cfg!(any(target_os = "macos", target_os = "ios")) {
            vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
        } else {
            vk::InstanceCreateFlags::default()
        };
        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(layers_names_raw)
            .enabled_extension_names(min_extension_names)
            .flags(create_flags);
        let instance = unsafe { entry.create_instance(&create_info, None)? };
        Ok(instance)
    }

    unsafe fn create_physical_device(
        entry: &Entry,
        instance: &Instance,
        surface: vk::SurfaceKHR,
    ) -> Result<(surface::Instance, vk::PhysicalDevice, u32)> {
        let physical_devices = unsafe { instance.enumerate_physical_devices()? };
        let surface_loader = surface::Instance::new(entry, instance);
        let (physical_device, queue_family_index) = physical_devices
            .iter()
            .find_map(|&candidate| unsafe {
                // TODO: check supported features and extensions.
                let queue_family_index = instance
                    .get_physical_device_queue_family_properties(candidate)
                    .iter()
                    .enumerate()
                    .find_map(|(index, info)| {
                        if !info.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                            return None;
                        }
                        match surface_loader.get_physical_device_surface_support(
                            candidate,
                            index as u32,
                            surface,
                        ) {
                            Ok(true) => {}
                            Ok(false) => {
                                return None;
                            }
                            Err(e) => {
                                error!(
                                    "get_physical_device_surface_support(): index={index}: {e:?}"
                                );
                                return None;
                            }
                        }
                        Some(index)
                    })?;
                Some((candidate, queue_family_index))
            })
            .context("Couldn't find suitable device.")?;
        let queue_family_index = queue_family_index as u32;
        Ok((surface_loader, physical_device, queue_family_index))
    }

    fn create_device(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        queue_family_index: u32,
        device_extension_names_raw: &[*const ffi::c_char],
        features: &vk::PhysicalDeviceFeatures,
        mut features_v12: vk::PhysicalDeviceVulkan12Features,
        mut features_v13: vk::PhysicalDeviceVulkan13Features,
    ) -> Result<Device> {
        let priorities = [1.0];

        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&priorities);

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .enabled_extension_names(device_extension_names_raw)
            .enabled_features(features)
            .push_next(&mut features_v12)
            .push_next(&mut features_v13);
        let device = unsafe { instance.create_device(physical_device, &device_create_info, None)? };
        Ok(device)
    }
}

pub struct TvWindowContext {
    _entry: Entry,
    instance: Instance,
    debug_handler: Option<DebugHandler>,

    // Just seems prudent to also have this.
    #[allow(unused)]
    window: Arc<Window>,

    surface: vk::SurfaceKHR,
    surface_loader: surface::Instance,
    physical_device: vk::PhysicalDevice,
    queue_family_index: u32,
    device: Device,
    present_queue: vk::Queue,
}

impl TvWindowContext {
    pub fn surface(&self) -> vk::SurfaceKHR {
        self.surface
    }
    pub fn queue_family_index(&self) -> u32 {
        self.queue_family_index
    }
    pub fn device(&self) -> &Device {
        &self.device
    }
    pub fn present_queue(&self) -> vk::Queue {
        self.present_queue
    }

    pub fn create_swapchain_device(&self) -> ash::khr::swapchain::Device {
        ash::khr::swapchain::Device::new(&self.instance, &self.device)
    }

    pub fn get_physical_device_memory_properties(&self) -> vk::PhysicalDeviceMemoryProperties {
        unsafe {
            self.instance
                .get_physical_device_memory_properties(self.physical_device)
        }
    }
    pub fn get_physical_device_surface_capabilities(&self) -> Result<vk::SurfaceCapabilitiesKHR> {
        unsafe {
            Ok(self
                .surface_loader
                .get_physical_device_surface_capabilities(self.physical_device, self.surface)?)
        }
    }
    pub fn get_physical_device_surface_formats(&self) -> Result<Vec<vk::SurfaceFormatKHR>> {
        unsafe {
            Ok(self
                .surface_loader
                .get_physical_device_surface_formats(self.physical_device, self.surface)?)
        }
    }
    pub fn get_physical_device_surface_present_modes(&self) -> Result<Vec<vk::PresentModeKHR>> {
        unsafe {
            Ok(self
                .surface_loader
                .get_physical_device_surface_present_modes(self.physical_device, self.surface)?)
        }
    }
}

impl Drop for TvWindowContext {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.debug_handler.take();
            self.instance.destroy_instance(None);
        }
    }
}

pub fn find_memorytype_index(
    memory_req: &vk::MemoryRequirements,
    memory_prop: &vk::PhysicalDeviceMemoryProperties,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    memory_prop.memory_types[..memory_prop.memory_type_count as _]
        .iter()
        .enumerate()
        .find(|(index, memory_type)| {
            (1 << index) & memory_req.memory_type_bits != 0
                && memory_type.property_flags & flags == flags
        })
        .map(|(index, _memory_type)| index as _)
}
