use egui::{FullOutput, ViewportId, ViewportInfo};
use std::{
    sync::{Arc, Mutex},
    time::Instant,
};

use crate::core::render::RenderHandler;
use crate::core::vk::vk_ctx::VulkanoContext;
use crate::gui::GuiContext;
use crate::shader::{Shader, SpriteShader, WireframeShader, ensure_shaders_locked};
use crate::util::{SceneHandlerBuilder, UniqueShared, gg_err, gg_float};
use crate::{
    core::{input::InputHandler, prelude::*},
    info_every_seconds,
    resource::ResourceHandler,
    util::gg_time::TimeIt,
    warn_every_seconds,
};
use egui_winit::winit::application::ApplicationHandler;
use egui_winit::winit::dpi::PhysicalSize;
use egui_winit::winit::event_loop::ActiveEventLoop;
use egui_winit::winit::keyboard::PhysicalKey;
use egui_winit::winit::window::{Window, WindowAttributes, WindowId};
use egui_winit::winit::{dpi::LogicalSize, event::WindowEvent, event_loop::EventLoop};
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::swapchain::{Swapchain, SwapchainCreateInfo};
use vulkano_taskgraph::graph::{CompileInfo, ExecutableTaskGraph, TaskGraph};
use vulkano_taskgraph::{Id, resource_map};

// Public to allow access to VulkanoContext.
pub mod vk_ctx;

#[derive(Clone)]
pub(crate) struct GgWindow {
    inner: Arc<Window>,
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
        Ok(Self { inner: window })
    }

    pub(crate) fn create_default_viewport(&self) -> AdjustedViewport {
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

    pub(crate) fn inner_size(&self) -> PhysicalSize<u32> {
        self.inner.inner_size()
    }
    pub(crate) fn scale_factor(&self) -> f32 {
        self.inner.scale_factor() as f32
    }
}

#[derive(Clone, Default)]
pub(crate) struct AdjustedViewport {
    inner: Viewport,
    scale_factor: f32,
    global_scale_factor: f32,
    pub(crate) translation: Vec2,
}

impl AdjustedViewport {
    pub(crate) fn update_from_window(&mut self, window: &GgWindow) {
        self.inner.extent = window.inner_size().into();
        self.scale_factor = window.scale_factor() * self.global_scale_factor;
        // TODO: verbose_every_seconds!
        info_every_seconds!(
            1,
            "update_from_window(): extent={:?}, scale_factor={}",
            self.inner.extent,
            self.scale_factor
        );
    }

    pub(crate) fn physical_width(&self) -> f32 {
        self.inner.extent[0]
    }
    pub(crate) fn physical_height(&self) -> f32 {
        self.inner.extent[1]
    }
    pub(crate) fn logical_width(&self) -> f32 {
        self.inner.extent[0] / self.scale_factor()
    }
    pub(crate) fn logical_height(&self) -> f32 {
        self.inner.extent[1] / self.scale_factor()
    }
    pub(crate) fn scale_factor(&self) -> f32 {
        self.scale_factor
    }
    pub(crate) fn set_global_scale_factor(&mut self, global_scale_factor: f32) {
        self.global_scale_factor = global_scale_factor;
    }
    pub(crate) fn global_scale_factor(&self) -> f32 {
        self.global_scale_factor
    }
    pub(crate) fn gui_scale_factor(&self) -> f32 {
        self.scale_factor / self.global_scale_factor
    }

    pub(crate) fn inner(&self) -> Viewport {
        self.inner.clone()
    }
}

impl AxisAlignedExtent for AdjustedViewport {
    fn aa_extent(&self) -> Vec2 {
        Vec2 {
            x: self.logical_width(),
            y: self.logical_height(),
        }
    }

    fn centre(&self) -> Vec2 {
        self.translation + self.half_widths()
    }
}

struct WindowEventHandlerInner {
    window: GgWindow,
    scale_factor: f32,
    vk_ctx: VulkanoContext,
    render_handler: RenderHandler,
    input_handler: Arc<Mutex<InputHandler>>,
    resource_handler: ResourceHandler,
    platform: egui_winit::State,
    task_graph: ExecutableTaskGraph<VulkanoContext>,
    virtual_swapchain_id: Id<Swapchain>,
}

struct WindowEventHandlerCreateInfo<F>
where
    F: FnOnce(SceneHandlerBuilder),
{
    window_size: Vec2i,
    create_and_start_scene_handler: Option<F>,
    global_scale_factor: f32,
    clear_col: Colour,
}

pub(crate) struct WindowEventHandler<F>
where
    F: FnOnce(SceneHandlerBuilder),
{
    create_info: WindowEventHandlerCreateInfo<F>,
    inner: Option<WindowEventHandlerInner>,

    gui_ctx: GuiContext,
    render_stats: RenderPerfStats,
    last_render_stats: Option<RenderPerfStats>,

    is_first_window_event: bool,
}

impl<F> WindowEventHandler<F>
where
    F: FnOnce(SceneHandlerBuilder),
{
    pub(crate) fn create_and_run(
        window_size: Vec2i,
        global_scale_factor: f32,
        clear_col: Colour,
        gui_ctx: GuiContext,
        create_and_start_scene_handler: F,
    ) -> Result<()> {
        let mut this = Self {
            create_info: WindowEventHandlerCreateInfo {
                window_size,
                global_scale_factor,
                clear_col,
                create_and_start_scene_handler: Some(create_and_start_scene_handler),
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
        self.inner
            .as_mut()
            .expect("missing WindowEventHandlerInner")
    }

    fn recreate_swapchain(&mut self) -> Result<(), gg_err::CatchOutOfDate> {
        let window = self.expect_inner().window.clone();
        self.expect_inner()
            .vk_ctx
            .recreate_swapchain(&window)
            .context("could not recreate swapchain")?;
        self.expect_inner()
            .render_handler
            .on_recreate_swapchain(window);
        Ok(())
    }

    fn create_inner(&mut self, event_loop: &ActiveEventLoop) -> Result<()> {
        check_is_none!(self.inner);

        let window = GgWindow::new(event_loop, self.create_info.window_size)?;
        let scale_factor = window.scale_factor();
        let viewport = UniqueShared::new(window.create_default_viewport());

        let vk_ctx = VulkanoContext::new(event_loop, &window)?;
        let input_handler = InputHandler::new();
        let resource_handler = ResourceHandler::new(&vk_ctx)?;
        // TODO: have preloading somehow.

        // TODO: these need a barrier between executions because they access the current image.
        //       That means the order of this vector matters. Need some way to let the user decide
        //       for default shaders too.
        let shaders: Vec<UniqueShared<Box<dyn Shader>>> = vec![
            SpriteShader::create(vk_ctx.clone(), viewport.clone(), resource_handler.clone())?,
            WireframeShader::create(vk_ctx.clone(), viewport.clone())?,
        ];

        let render_handler = RenderHandler::new(
            &vk_ctx,
            self.gui_ctx.clone(),
            resource_handler.clone(),
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
            Some(window.scale_factor()),
            None,
            None,
        );

        let (task_graph, virtual_swapchain_id) =
            Self::build_task_graph(&vk_ctx, &render_handler, &resource_handler)?;

        self.inner = Some(WindowEventHandlerInner {
            window,
            scale_factor,
            vk_ctx,
            render_handler,
            input_handler,
            resource_handler,
            platform,
            task_graph,
            virtual_swapchain_id,
        });
        Ok(())
    }

    fn build_task_graph(
        vk_ctx: &VulkanoContext,
        render_handler: &RenderHandler,
        resource_handler: &ResourceHandler,
    ) -> Result<(ExecutableTaskGraph<VulkanoContext>, Id<Swapchain>)> {
        // TODO: verbose!
        info!("building task graph");
        let mut task_graph = TaskGraph::new(&vk_ctx.resources(), 100, 100);
        let virtual_swapchain_id = task_graph.add_swapchain(&SwapchainCreateInfo::default());
        let (texture_node, images) = resource_handler.texture.build_task_graph(&mut task_graph);
        render_handler.build_shader_task_graphs(
            &mut task_graph,
            texture_node,
            virtual_swapchain_id,
            &images,
        )?;
        let task_graph = unsafe {
            task_graph.compile(&CompileInfo {
                queues: &[&vk_ctx.queue()],
                present_queue: Some(&vk_ctx.queue()),
                flight_id: vk_ctx.flight_id(),
                ..Default::default()
            })?
        };
        Ok((task_graph, virtual_swapchain_id))
    }
}

impl<F> ApplicationHandler for WindowEventHandler<F>
where
    F: FnOnce(SceneHandlerBuilder),
{
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(callback) = self.create_info.create_and_start_scene_handler.take() {
            // First event. Note winit documentation:
            // "This is a common indicator that you can create a window."
            self.create_inner(event_loop).expect("error initialising");
            callback(SceneHandlerBuilder::new(
                self.expect_inner().input_handler.clone(),
                self.expect_inner().resource_handler.clone(),
                self.expect_inner().render_handler.clone(),
            ));
            self.is_first_window_event = true;
        }
        check_is_some!(self.inner);
    }

    fn window_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let window = self.expect_inner().window.clone();
        let _response = self
            .expect_inner()
            .platform
            .on_window_event(&window.inner, &event);
        match event {
            WindowEvent::CloseRequested => {
                info!("received WindowEvent::CloseRequested, calling exit(0)");
                std::process::exit(0);
            }
            WindowEvent::KeyboardInput { event, .. } => match event.physical_key {
                PhysicalKey::Code(keycode) => {
                    self.expect_inner()
                        .input_handler
                        .lock()
                        .unwrap()
                        .queue_key_event(keycode, event.state);
                }
                PhysicalKey::Unidentified(keycode) => {
                    info!("PhysicalKey::Unidentified({keycode:?}), ignoring");
                }
            },
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                let scale_factor = scale_factor as f32;
                // Since scale_factor is given by winit, we expect an exact comparison to work.
                #[allow(clippy::float_cmp)]
                if self.expect_inner().scale_factor != scale_factor {
                    // TODO: verbose_every_seconds!
                    info_every_seconds!(
                        1,
                        "WindowEvent::ScaleFactorChanged: {} -> {}: recreating swapchain",
                        self.expect_inner().scale_factor,
                        scale_factor
                    );
                    self.expect_inner().scale_factor = scale_factor;
                    self.recreate_swapchain().unwrap();
                }
            }
            WindowEvent::Resized(physical_size) => {
                // TODO: verbose_every_seconds!
                info_every_seconds!(
                    1,
                    "WindowEvent::Resized: {:?}: recreating swapchain",
                    physical_size
                );
                self.recreate_swapchain().unwrap();
            }
            WindowEvent::RedrawRequested => {
                if self
                    .expect_inner()
                    .resource_handler
                    .texture
                    .wait_textures_dirty()
                    || self.expect_inner().render_handler.is_dirty()
                {
                    let vk_ctx = self.expect_inner().vk_ctx.clone();
                    let render_handler = self.expect_inner().render_handler.clone();
                    let resource_handler = self.expect_inner().resource_handler.clone();
                    let (task_graph, virtual_swapchain_id) =
                        Self::build_task_graph(&vk_ctx, &render_handler, &resource_handler)
                            .unwrap();
                    self.expect_inner().task_graph = task_graph;
                    self.expect_inner().virtual_swapchain_id = virtual_swapchain_id;
                }
                match self.acquire_and_handle_image() {
                    Err(gg_err::CatchOutOfDate::VulkanOutOfDateError) => {
                        // TODO: verbose_every_seconds!
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

impl<F> WindowEventHandler<F>
where
    F: FnOnce(SceneHandlerBuilder),
{
    fn acquire_and_handle_image(&mut self) -> Result<(), gg_err::CatchOutOfDate> {
        self.render_stats.start();

        self.render_stats.synchronise.start();
        let vk_ctx = self.expect_inner().vk_ctx.clone();
        vk_ctx
            .resources()
            .flight(vk_ctx.flight_id())
            .map_err(gg_err::CatchOutOfDate::from)?
            .wait(None)
            .map_err(gg_err::CatchOutOfDate::from)?;
        self.render_stats.synchronise.stop();

        self.render_stats.update_gui.start();
        let _full_output = self.update_gui();
        self.render_stats.update_gui.stop();

        self.render_stats.execute.start();
        let inner = self.expect_inner();
        // TODO: add more stuff to resource_map instead of recreating it all the time. From Marc:
        //  "Or even if you use the same resources all the time, it's useful to use the resource map
        //   for frame-local resources such as uniform buffers. All a ResourceMap is for is to
        //   allow you to specify resources at task graph runtime rather than compile time. If you
        //   use physical resources, you have to recompile the task graph each time."
        let resource_map = resource_map!(
            &inner.task_graph,
            inner.virtual_swapchain_id => vk_ctx.swapchain_id(),
        )
        .map_err(gg_err::CatchOutOfDate::from)?;

        unsafe {
            inner
                .task_graph
                .execute(resource_map, &vk_ctx, || {
                    inner.window.inner.pre_present_notify();
                })
                .map_err(gg_err::CatchOutOfDate::from)?;
        }
        self.render_stats.execute.stop();

        self.last_render_stats = self.render_stats.end();
        Ok(())
    }

    fn update_gui(&mut self) -> FullOutput {
        let window = self.expect_inner().window.clone();
        egui_winit::update_viewport_info(
            &mut ViewportInfo::default(),
            &self.gui_ctx,
            &window.inner,
            self.is_first_window_event,
        );
        self.is_first_window_event = false;
        let raw_input = self.expect_inner().platform.take_egui_input(&window.inner);
        let stats = self.last_render_stats.clone();
        let render_handler = self.expect_inner().render_handler.clone();
        let input = self.expect_inner().input_handler.clone();
        let full_output = self.gui_ctx.run(raw_input, |ctx| {
            {
                let mut input = input.lock().unwrap();
                input.set_viewport(render_handler.viewport());
                input.update_mouse(ctx);
            }
            render_handler.do_gui(ctx, stats.clone());
        });
        // TODO: messy.
        render_handler.update_full_output(full_output.clone());
        self.expect_inner()
            .platform
            .handle_platform_output(&window.inner, full_output.platform_output.clone());
        full_output
    }
}

#[derive(Clone)]
pub(crate) struct RenderPerfStats {
    synchronise: TimeIt,
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

    last_perf_stats: Option<Box<RenderPerfStats>>,
}

impl RenderPerfStats {
    fn new() -> Self {
        Self {
            synchronise: TimeIt::new("synchronise"),
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
            last_perf_stats: None,
        }
    }

    fn start(&mut self) {
        self.between_renders.stop();
    }

    fn end(&mut self) -> Option<Self> {
        const DEADLINE_MS: f32 = 16.8;

        self.between_renders.start();

        if self.totals_ms.len() == self.totals_ms.capacity() {
            self.totals_ms.remove(0);
        }
        let render_time = gg_float::micros(self.last_step.elapsed()) * 1000.;
        self.totals_ms.push(render_time);

        let late_in_row = self
            .totals_ms
            .iter()
            .rev()
            .take_while(|&&t| t > DEADLINE_MS)
            .collect_vec();
        if late_in_row.len() > 1 {
            let mut msg = format!("{} frames late in a row: ", late_in_row.len());
            for time in &late_in_row[..late_in_row.len() - 1] {
                msg += format!("{time:.1}, ").as_str();
            }
            msg += format!("{:.1}", late_in_row.last().unwrap()).as_str();
            warn_every_seconds!(1, "{msg}");
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
            let on_time_rate = self.on_time as f32 / self.count as f32 * 100.;
            if on_time_rate.round() < 100. {
                warn!("frames on time: {on_time_rate:.1}%");
            }
            self.last_perf_stats = Some(Box::new(Self {
                synchronise: self.synchronise.report_take(),
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
            }));
            self.last_report = Instant::now();
            self.last_report = Instant::now();
            self.on_time = 0;
            self.count = 0;
        }

        self.last_perf_stats.clone().map(|s| *s)
    }

    pub(crate) fn as_tuples_ms(&self) -> Vec<(String, f32, f32)> {
        let mut default = vec![
            self.total.as_tuple_ms(),
            self.synchronise.as_tuple_ms(),
            self.update_gui.as_tuple_ms(),
            self.execute.as_tuple_ms(),
            self.between_renders.as_tuple_ms(),
        ];
        if self.extra_debug.last_ms() != 0. {
            default.push(self.extra_debug.as_tuple_ms());
        }
        default
    }
}
