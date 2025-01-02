use std::{cell::RefCell, rc::Rc, sync::{Arc, Mutex, MutexGuard}, time::Instant};
use std::marker::PhantomData;
use egui::{FullOutput, ViewportId, ViewportInfo};
use num_traits::Zero;

use vulkano::{command_buffer::{
    CommandBufferExecFuture,
    PrimaryAutoCommandBuffer,
}, pipeline::graphics::viewport::Viewport, swapchain::{
    PresentFuture,
    SwapchainAcquireFuture,
}, sync::{
    future::{FenceSignalFuture, JoinFuture},
    GpuFuture,
}, Validated};
use egui_winit::winit::{dpi::LogicalSize, event::WindowEvent, event_loop::EventLoop};
use egui_winit::winit::application::ApplicationHandler;
use egui_winit::winit::dpi::PhysicalSize;
use egui_winit::winit::event_loop::ActiveEventLoop;
use egui_winit::winit::keyboard::PhysicalKey;
use egui_winit::winit::window::{Window, WindowAttributes, WindowId};
use std::time::Duration;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::pipeline::graphics::viewport::{Scissor, ViewportState};
use crate::{core::{
    input::InputHandler,
    prelude::*,
}, info_every_seconds, resource::ResourceHandler, util::{
    gg_time::TimeIt
}, warn_every_seconds};
use crate::core::ObjectTypeEnum;
use crate::core::render::RenderHandler;
use crate::core::vk::vk_ctx::{DataPerImage, VulkanoContext};
use crate::gui::GuiContext;
use crate::shader::{ensure_shaders_locked, BasicShader, Shader, SpriteShader, TriangleFanShader, WireframeShader};
use crate::util::{gg_err, gg_float, SceneHandlerBuilder, UniqueShared};

pub mod vk_ctx;

#[derive(Clone)]
pub struct GgWindow {
    inner: Arc<Window>,
}

impl GgWindow {
    pub fn new(event_loop: &ActiveEventLoop, size: impl Into<Vec2i>) -> Result<Self> {
        let size = size.into();
        let mut window_attrs = WindowAttributes::default();
        window_attrs.title = "glongge".to_string();
        window_attrs.resizable = true;
        window_attrs.inner_size = Some(egui_winit::winit::dpi::Size::Logical(
            LogicalSize::new(f64::from(size.x), f64::from(size.y)))
        );
        let window = Arc::new(event_loop.create_window( window_attrs)?);
        Ok(Self { inner: window })
    }

    pub fn create_default_viewport(&self) -> AdjustedViewport {
        AdjustedViewport {
            inner: Viewport {
                offset: [0., 0.],
                extent: self.inner_size().into(),
                depth_range: 0.0..=1.,
            },
            scale_factor: self.scale_factor(),
            global_scale_factor: 1.,
            translation: Vec2::zero(),
        }
    }

    pub fn inner_size(&self) -> PhysicalSize<u32> { self.inner.inner_size() }
    pub fn scale_factor(&self) -> f64 { self.inner.scale_factor() }
}

#[derive(Clone, Default)]
pub struct AdjustedViewport {
    inner: Viewport,
    scale_factor: f64,
    global_scale_factor: f64,
    pub(crate) translation: Vec2,
}

impl AdjustedViewport {
    pub fn set_global_scale_factor(&mut self, global_scale_factor: f64) {
        self.global_scale_factor = global_scale_factor;
    }
    pub(crate) fn get_global_scale_factor(&self) -> f64 {
        self.global_scale_factor
    }
    pub fn update_from_window(&mut self, window: &GgWindow) {
        self.inner.extent = window.inner_size().into();
        self.scale_factor = window.scale_factor() * self.global_scale_factor;
        info_every_seconds!(1, "update_from_window(): extent={:?}, scale_factor={}",
            self.inner.extent, self.scale_factor);
    }

    pub fn physical_width(&self) -> f64 { f64::from(self.inner.extent[0]) }
    pub fn physical_height(&self) -> f64 { f64::from(self.inner.extent[1]) }
    pub fn logical_width(&self) -> f64 { f64::from(self.inner.extent[0]) / self.scale_factor() }
    pub fn logical_height(&self) -> f64 { f64::from(self.inner.extent[1]) / self.scale_factor() }
    pub fn gui_scale_factor(&self) -> f64 { self.scale_factor / self.global_scale_factor }
    pub fn scale_factor(&self) -> f64 { self.scale_factor }

    pub fn as_viewport_state(&self) -> ViewportState {
        ViewportState {
            viewports: [self.inner.clone()].into_iter().collect(),
            scissors: [Scissor {
                offset: [
                    gg_float::f32_to_u32(self.inner.offset[0].floor()).expect("very weird viewport extent"),
                    gg_float::f32_to_u32(self.inner.offset[1].floor()).expect("very weird viewport extent"),
                ],
                extent: [
                    gg_float::f32_to_u32(self.inner.extent[0].floor()).expect("very weird viewport extent"),
                    gg_float::f32_to_u32(self.inner.extent[1].floor()).expect("very weird viewport extent"),
                ],
            }].into_iter().collect(),
            ..ViewportState::default()
        }
    }

    pub fn inner(&self) -> Viewport { self.inner.clone() }

    #[must_use]
    pub fn translated(&self, translation: Vec2) -> AdjustedViewport {
        let mut rv = self.clone();
        rv.translation = translation;
        rv
    }
}

impl AxisAlignedExtent for AdjustedViewport {
    fn aa_extent(&self) -> Vec2 {
        Vec2 { x: self.logical_width(), y: self.logical_height() }
    }

    fn centre(&self) -> Vec2 {
        self.translation + self.half_widths()
    }
}

#[derive(Clone)]
pub struct PerImageContext {
    last: usize,
    current: Option<usize>,
}

impl PerImageContext {
    fn new() -> UniqueShared<Self> {
        UniqueShared::new(Self {
            last: 0,
            current: None,
        })
    }
}

type SwapchainJoinFuture = JoinFuture<Box<dyn GpuFuture>, SwapchainAcquireFuture>;
type FenceFuture = FenceSignalFuture<PresentFuture<CommandBufferExecFuture<SwapchainJoinFuture>>>;

struct WindowEventHandlerInner {
    window: GgWindow,
    scale_factor: f64,
    vk_ctx: VulkanoContext,
    render_handler: RenderHandler,
    input_handler: Arc<Mutex<InputHandler>>,
    resource_handler: ResourceHandler,
    platform: egui_winit::State,
    fences: DataPerImage<Rc<RefCell<Option<FenceFuture>>>>,
}

struct WindowEventHandlerCreateInfo<F, ObjectType>
where
    F: FnOnce(SceneHandlerBuilder<ObjectType>),
    ObjectType: ObjectTypeEnum,
{
    window_size: Vec2i,
    create_and_start_scene_handler: Option<F>,
    global_scale_factor: f64,
    clear_col: Colour,
    phantom_data: PhantomData<ObjectType>,
}

pub struct WindowEventHandler<F, ObjectType>
where
    F: FnOnce(SceneHandlerBuilder<ObjectType>),
    ObjectType: ObjectTypeEnum,
{
    create_info: WindowEventHandlerCreateInfo<F, ObjectType>,
    inner: Option<WindowEventHandlerInner>,

    gui_ctx: GuiContext,
    render_stats: RenderPerfStats,
    last_render_stats: Option<RenderPerfStats>,

    is_first_window_event: bool,
}

#[allow(private_bounds)]
impl<F, ObjectType> WindowEventHandler<F, ObjectType>
where
    F: FnOnce(SceneHandlerBuilder<ObjectType>),
    ObjectType: ObjectTypeEnum,
{
    pub fn create_and_run(
        window_size: Vec2i,
        global_scale_factor: f64,
        clear_col: Colour,
        gui_ctx: GuiContext,
        create_and_start_scene_handler: F
    ) -> Result<()> {
        let mut this = Self {
            create_info: WindowEventHandlerCreateInfo {
                window_size,
                global_scale_factor,
                clear_col,
                create_and_start_scene_handler: Some(create_and_start_scene_handler),
                phantom_data: PhantomData,
            },
            inner: None,

            gui_ctx,
            render_stats: RenderPerfStats::new(),
            last_render_stats: None,

            is_first_window_event: false,
        };

        let event_loop = EventLoop::new()?;
        Ok(event_loop.run_app(&mut this)?)
    }

    fn expect_inner(&mut self) -> &mut WindowEventHandlerInner {
        self.inner.as_mut().expect("missing WindowEventHandlerInner")
    }

    fn recreate_swapchain(&mut self) -> Result<(), gg_err::CatchOutOfDate> {
        self.expect_inner().fences = DataPerImage::new_with_generator(&self.expect_inner().vk_ctx, || Rc::new(RefCell::new(None)));
        self.expect_inner().vk_ctx.per_image_ctx.get().current.take();
        let window = self.expect_inner().window.clone();
        self.expect_inner().vk_ctx.recreate_swapchain(&window)
            .context("could not recreate swapchain")?;
        self.expect_inner().render_handler.on_recreate_swapchain(window);
        Ok(())
    }

    fn do_full_render(
        &mut self,
        image_idx: usize,
        acquire_future: SwapchainAcquireFuture,
        full_output: FullOutput
    ) -> Result<(), gg_err::CatchOutOfDate> {
        let per_image_ctx = self.expect_inner().vk_ctx.per_image_ctx.clone();
        let mut per_image_ctx = per_image_ctx.get();
        if let Some(last_image_idx) = per_image_ctx.current.replace(image_idx) {
            if last_image_idx == image_idx {
                warn!("last_image_idx == image_idx == {}", image_idx);
            }
        }

        self.render_stats.synchronise.start();
        let ready_future = self.synchronise(&mut per_image_ctx, acquire_future);
        self.render_stats.synchronise.stop();

        self.render_stats.do_render.start();
        let vk_ctx = self.expect_inner().vk_ctx.clone();
        let mut builder = AutoCommandBufferBuilder::primary(
            vk_ctx.command_buffer_allocator(),
            vk_ctx.queue().queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        ).map_err(gg_err::CatchOutOfDate::from)?;
        self.expect_inner().resource_handler.texture
            .wait_maybe_upload_textures(&vk_ctx, &mut builder)?;
        let command_buffer = self.expect_inner().render_handler.do_render(
            builder,
            &vk_ctx.current_framebuffer(&per_image_ctx),
            full_output
        ).map_err(gg_err::CatchOutOfDate::from)?;
        self.render_stats.do_render.stop();

        self.expect_inner().window.inner.pre_present_notify();

        self.render_stats.submit_command_buffers.start();
        self.submit_command_buffer(&mut per_image_ctx, command_buffer, ready_future)?;
        self.render_stats.submit_command_buffers.stop();

        if (per_image_ctx.last + 1) % self.expect_inner().vk_ctx.image_count() != image_idx {
            warn_every_seconds!(1, "per_image_ctx: last={}, next={}, count={}",
                                per_image_ctx.last,
                                image_idx,
                                self.expect_inner().vk_ctx.image_count());
        }
        per_image_ctx.last = image_idx;
        Ok(())
    }

    fn synchronise(
        &mut self,
        per_image_ctx: &mut MutexGuard<PerImageContext>,
        acquire_future: SwapchainAcquireFuture,
    ) -> SwapchainJoinFuture {
        if let Some(last_fence) = self.expect_inner().fences.last_value(per_image_ctx).borrow_mut().as_mut() {
            if let Err(e) = last_fence.wait(None).map_err(Validated::unwrap) {
                // try to continue -- it might be an outdated future
                // XXX: macOS often just segfaults instead of giving an error here
                error!("{}", e);
            }
            last_fence.cleanup_finished();
        }
        let next_fence = if let Some(mut fence) = self.expect_inner().fences.current_value(per_image_ctx).take() {
            // Future should already be completed by the time we acquire the image.
            fence.wait(Some(Duration::from_nanos(1))).unwrap();
            fence.cleanup_finished();
            fence.boxed()
        } else {
            let mut now = vulkano::sync::now(self.expect_inner().vk_ctx.device());
            now.cleanup_finished();
            now.boxed()
        };
        next_fence.join(acquire_future)
    }

    fn submit_command_buffer(
        &mut self,
        per_image_ctx: &mut MutexGuard<PerImageContext>,
        command_buffer: Arc<PrimaryAutoCommandBuffer>,
        ready_future: SwapchainJoinFuture,
    ) -> Result<()> {
        let vk_ctx = self.expect_inner().vk_ctx.clone();
        let image_idx = per_image_ctx.current.expect("no current image?");
        let mut current_fence = self.expect_inner().fences
            .current_value_mut(per_image_ctx)
            .borrow_mut();
        check_is_none!(current_fence);
        current_fence.replace(
            ready_future
                .then_execute(vk_ctx.queue(), command_buffer)?
                .then_swapchain_present(
                    vk_ctx.queue(),
                    vk_ctx.swapchain_present_info(image_idx)?
                )
                .then_signal_fence_and_flush()?
            );
        Ok(())
    }
    
    fn create_inner(&mut self, event_loop: &ActiveEventLoop) -> Result<()> {
        check_is_none!(self.inner);

        let window = GgWindow::new(event_loop, self.create_info.window_size)?;
        let scale_factor = window.scale_factor();
        let viewport = UniqueShared::new(window.create_default_viewport());

        let vk_ctx = VulkanoContext::new(event_loop, &window)?;
        let input_handler = InputHandler::new();
        let mut resource_handler = ResourceHandler::new(&vk_ctx)?;
        ObjectType::preload_all(&mut resource_handler)?;

        let shaders: Vec<UniqueShared<Box<dyn Shader>>> = vec![
            SpriteShader::create(vk_ctx.clone(), viewport.clone(), resource_handler.clone())?,
            WireframeShader::create(vk_ctx.clone(), viewport.clone())?,
            BasicShader::create(vk_ctx.clone(), viewport.clone())?,
            TriangleFanShader::create(vk_ctx.clone(), viewport.clone())?,
        ];

        let render_handler = RenderHandler::new(
            &vk_ctx,
            self.gui_ctx.clone(),
            window.clone(),
            viewport.clone(),
            shaders,
        )?
            .with_global_scale_factor(self.create_info.global_scale_factor)
            .with_clear_col(self.create_info.clear_col);
        ensure_shaders_locked();

        let platform = egui_winit::State::new(
            self.gui_ctx.clone(),
            ViewportId::ROOT,
            &event_loop,
            Some(window.scale_factor() as f32),
            None, None
        );

        let fences = DataPerImage::new_with_generator(&vk_ctx, || Rc::new(RefCell::new(None)));

        self.inner = Some(WindowEventHandlerInner {
            window,
            scale_factor,
            vk_ctx,
            render_handler,
            input_handler,
            resource_handler,
            platform,
            fences,
        });
        Ok(())
    }
}

impl<F, ObjectType> ApplicationHandler for WindowEventHandler<F, ObjectType>
where
    F: FnOnce(SceneHandlerBuilder<ObjectType>),
    ObjectType: ObjectTypeEnum,
{
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(callback) = self.create_info.create_and_start_scene_handler.take() {
            // First event. Note winit documentation:
            // "This is a common indicator that you can create a window."
            self.create_inner(event_loop).expect("error initialising");
            callback(SceneHandlerBuilder::new(
                self.expect_inner().input_handler.clone(),
                self.expect_inner().resource_handler.clone(),
                self.expect_inner().render_handler.clone()
            ));
            self.is_first_window_event = true;
        }
        check_is_some!(self.inner);
    }

    fn window_event(&mut self, _event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent) {
        let window = self.expect_inner().window.clone();
        let _response = self.expect_inner().platform.on_window_event(&window.inner, &event);
        match event {
            WindowEvent::CloseRequested => {
                info!("received WindowEvent::CloseRequested, calling exit(0)");
                std::process::exit(0);
            }
            WindowEvent::KeyboardInput {
                event, ..
            } => {
                match event.physical_key {
                    PhysicalKey::Code(keycode) => {
                        self.expect_inner().input_handler.lock().unwrap().queue_key_event(keycode, event.state);
                    }
                    PhysicalKey::Unidentified(_) => {}
                }
            }
            WindowEvent::ScaleFactorChanged {
                scale_factor, ..
            } => {
                // Since scale_factor is given by winit, we expect exact comparison to work.
                #[allow(clippy::float_cmp)]
                if self.expect_inner().scale_factor != scale_factor {
                    info_every_seconds!(1, "WindowEvent::ScaleFactorChanged: {} -> {}: recreating swapchain",
                        self.expect_inner().scale_factor, scale_factor);
                    self.expect_inner().scale_factor = scale_factor;
                    self.recreate_swapchain().unwrap();
                }
            }
            WindowEvent::Resized(physical_size) => {
                info_every_seconds!(1, "WindowEvent::Resized: {:?}: recreating swapchain",
                    physical_size);
                self.recreate_swapchain().unwrap();
            }
            WindowEvent::RedrawRequested => {
                match self.acquire_and_handle_image() {
                    Err(gg_err::CatchOutOfDate::VulkanOutOfDateError) => {
                        info_every_seconds!(1, "VulkanError::OutOfDate, recreating swapchain");
                        self.recreate_swapchain().unwrap();
                    }
                    rv => rv.unwrap(),
                }
            }
            _other_event => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        self.expect_inner().window.inner.request_redraw();
    }
}

impl<F, ObjectType> WindowEventHandler<F, ObjectType>
where
    F: FnOnce(SceneHandlerBuilder<ObjectType>),
    ObjectType: ObjectTypeEnum,
{
    fn acquire_and_handle_image(&mut self, ) -> Result<(), gg_err::CatchOutOfDate> {
        let acquired = self.expect_inner().vk_ctx.acquire_next_image()?;

        self.render_stats.start();
        let rv = if let (image_idx, /* suboptimal= */ false, acquire_future) = acquired {
            let full_output = self.handle_egui();
            self.do_full_render(image_idx as usize, acquire_future, full_output)
        } else {
            info_every_seconds!(1, "suboptimal: recreating swapchain");
            self.recreate_swapchain()
        };
        self.last_render_stats = self.render_stats.end();
        rv
    }

    fn handle_egui(&mut self) -> FullOutput {
        let window = self.expect_inner().window.clone();
        egui_winit::update_viewport_info(
            &mut ViewportInfo::default(),
            &self.gui_ctx,
            &window.inner,
            self.is_first_window_event
        );
        self.is_first_window_event = false;
        let raw_input = self.expect_inner().platform.take_egui_input(&window.inner);
        let stats = self.last_render_stats.clone();
        let mut render_handler = self.expect_inner().render_handler.clone();
        let input = self.expect_inner().input_handler.clone();
        let full_output = self.gui_ctx.run(raw_input, |ctx| {
            let mut input = input.lock().unwrap();
            input.set_viewport(render_handler.viewport());
            input.update_mouse(ctx);
            render_handler.do_gui(ctx, stats.clone());
        });
        self.expect_inner().platform.handle_platform_output(
            &window.inner,
            full_output.platform_output.clone()
        );
        full_output
    }
}

#[derive(Clone)]
pub(crate) struct RenderPerfStats {
    handle_swapchain: TimeIt,
    synchronise: TimeIt,
    do_render: TimeIt,
    submit_command_buffers: TimeIt,
    between_renders: TimeIt,
    extra_debug: TimeIt,

    total: TimeIt,
    on_time: u64,
    count: u64,
    penultimate_step: Instant,
    last_step: Instant,
    last_report: Instant,
    totals_ms: Vec<f64>,

    last_perf_stats: Option<Box<RenderPerfStats>>,
}

impl RenderPerfStats {
    fn new() -> Self {
        Self {
            handle_swapchain: TimeIt::new("handle swapchain"),
            synchronise: TimeIt::new("synchronise"),
            do_render: TimeIt::new("do_render"),
            submit_command_buffers: TimeIt::new("submit cmdbufs"),
            between_renders: TimeIt::new("between renders"),
            extra_debug: TimeIt::new("extra_debug"),
            total: TimeIt::new("total"),
            on_time: 0,
            count: 0,
            penultimate_step: Instant::now(),
            last_step: Instant::now(),
            last_report: Instant::now(),
            totals_ms: Vec::with_capacity(10),
            last_perf_stats: None,
        }
    }

    fn start(&mut self) {
        self.between_renders.stop();
    }

    fn end(&mut self) -> Option<Self> {
        const DEADLINE_MS: f64 = 16.8;

        self.between_renders.start();

        if self.totals_ms.len() == self.totals_ms.capacity() {
            self.totals_ms.remove(0);
        }
        let render_time = gg_float::micros(self.last_step.elapsed()) * 1000.;
        self.totals_ms.push(render_time);

        let late_in_row = self.totals_ms.iter()
            .rev()
            .take_while(|&&t| t > DEADLINE_MS)
            .collect_vec();
        if late_in_row.len() > 1 {
            let mut msg = format!("{} frames late in a row: ", late_in_row.len());
            for time in &late_in_row[..late_in_row.len() - 1] {
                msg += format!("{time:.1}, ").as_str();
            }
            msg += format!("{:.1}", late_in_row.last().unwrap()).as_str();
            warn!("{msg}");
        }
        if render_time <= DEADLINE_MS {
            self.on_time += 1;
        }
        self.count += 1;

        self.total.stop();
        self.total.start();
        self.penultimate_step = self.last_step;
        self.last_step = Instant::now();

        if self.last_report.elapsed().as_secs() >= 2 {
            #[allow(clippy::cast_precision_loss)]
            let on_time_rate = self.on_time as f64 / self.count as f64 * 100.;
            if on_time_rate.round() < 100. {
                warn!("frames on time: {on_time_rate:.1}%");
            }
            self.last_perf_stats = Some(Box::new(Self {
                handle_swapchain: self.handle_swapchain.report_take(),
                synchronise: self.synchronise.report_take(),
                do_render: self.do_render.report_take(),
                submit_command_buffers: self.submit_command_buffers.report_take(),
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
            }));
            self.last_report = Instant::now();
            self.last_report = Instant::now();
            self.on_time = 0;
            self.count = 0;
        }

        self.last_perf_stats.clone().map(|s| *s)
    }

    pub(crate) fn as_tuples_ms(&self) -> Vec<(String, f64, f64)> {
        let mut default = vec![
            self.total.as_tuple_ms(),
            self.handle_swapchain.as_tuple_ms(),
            self.synchronise.as_tuple_ms(),
            self.do_render.as_tuple_ms(),
            self.submit_command_buffers.as_tuple_ms(),
            self.between_renders.as_tuple_ms(),
        ];
        if self.extra_debug.last_ms() != 0. {
            default.push(self.extra_debug.as_tuple_ms());
        }
        default
    }
}
