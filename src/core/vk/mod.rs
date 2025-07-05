use crate::core::render::RenderHandler;
use crate::core::scene::SceneHandler;
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
use egui::{FullOutput, ViewportId, ViewportInfo};
use egui_winit::winit::application::ApplicationHandler;
use egui_winit::winit::dpi::PhysicalSize;
use egui_winit::winit::event_loop::ActiveEventLoop;
use egui_winit::winit::keyboard::PhysicalKey;
use egui_winit::winit::window::{Window, WindowAttributes, WindowId};
use egui_winit::winit::{dpi::LogicalSize, event::WindowEvent, event_loop::EventLoop};
use std::sync::mpsc;
use std::sync::mpsc::{Receiver, Sender};
use std::time::Duration;
use std::{
    sync::{Arc, Mutex},
    time::Instant,
};
use vulkano::VulkanError;
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
                offset: [0.0, 0.0],
                extent: self.inner_size().into(),
                depth_range: 0.0..=1.0,
            },
            scale_factor: self.scale_factor(),
            global_scale_factor: 1.0,
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
    gui_ctx: GuiContext,
    platform: egui_winit::State,
    task_graph: ExecutableTaskGraph<VulkanoContext>,
    virtual_swapchain_id: Id<Swapchain>,
    render_stats: RenderPerfStats,
    last_render_stats: Option<RenderPerfStats>,
    is_first_window_event: bool,

    window_event_rx: Receiver<WindowEvent>,
    scale_factor_rx: Receiver<f32>,
    recreate_swapchain_rx: Receiver<Instant>,
}

impl WindowEventHandlerInner {
    fn run_update(&mut self) {
        self.vk_ctx.perf_stats().lap("start");
        self.handle_window_events();
        self.vk_ctx.perf_stats().lap("handle_window_event()");
        if self.resource_handler.texture.wait_textures_dirty() || self.render_handler.is_dirty() {
            let vk_ctx = self.vk_ctx.clone();
            let render_handler = self.render_handler.clone();
            let resource_handler = self.resource_handler.clone();
            let (task_graph, virtual_swapchain_id) =
                build_task_graph(&vk_ctx, &render_handler, &resource_handler).unwrap();
            self.task_graph = task_graph;
            self.virtual_swapchain_id = virtual_swapchain_id;
            self.vk_ctx.perf_stats().lap("build_task_graph()");
        }
        match self.acquire_and_handle_image() {
            Err(gg_err::CatchOutOfDate::VulkanOutOfDateError) => {
                // TODO: verbose_every_seconds!
                info_every_seconds!(1, "VulkanError::OutOfDate, recreating swapchain");
                self.recreate_swapchain().unwrap();
            }
            rv => rv.unwrap(),
        }
        self.vk_ctx.perf_stats().lap("acquire_and_handle_image()");

        self.vk_ctx.perf_stats().report(20);
    }

    fn handle_window_events(&mut self) {
        while let Ok(event) = self.window_event_rx.try_recv() {
            let _response = self.platform.on_window_event(&self.window.inner, &event);
        }
        if let Some(new_scale_factor) = self.scale_factor_rx.try_iter().last() {
            // Since scale_factor is given by winit, we expect an exact comparison to work.
            #[allow(clippy::float_cmp)]
            if self.scale_factor != new_scale_factor {
                // TODO: verbose_every_seconds!
                info_every_seconds!(
                    1,
                    "WindowEvent::ScaleFactorChanged: {} -> {}: recreating swapchain",
                    self.scale_factor,
                    new_scale_factor
                );
                self.scale_factor = new_scale_factor;
                self.recreate_swapchain().unwrap();
            }
        }
        if let Some(request_time) = self.recreate_swapchain_rx.try_iter().last() {
            info!(
                "recreating swapchain: {:.2} ms old",
                request_time.elapsed().as_micros() as f32 / 1000.0
            );
            self.recreate_swapchain().unwrap();
        }
    }

    fn update_gui(&mut self) -> FullOutput {
        egui_winit::update_viewport_info(
            &mut ViewportInfo::default(),
            &self.gui_ctx.clone(),
            &self.window.inner,
            self.is_first_window_event,
        );
        self.is_first_window_event = false;
        let raw_input = self.platform.take_egui_input(&self.window.inner);
        let stats = self.last_render_stats.clone();
        let render_handler = self.render_handler.clone();
        let input = self.input_handler.clone();
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
        self.platform
            .handle_platform_output(&self.window.inner, full_output.platform_output.clone());
        full_output
    }

    fn acquire_and_handle_image(&mut self) -> Result<(), gg_err::CatchOutOfDate> {
        match self
            .vk_ctx
            .resources()
            .flight(self.vk_ctx.flight_id())
            .map_err(gg_err::CatchOutOfDate::from)?
            .wait(Some(Duration::ZERO))
        {
            Ok(()) => {}
            Err(VulkanError::Timeout) => return Ok(()),
            Err(e) => return Err(e.into()),
        }

        self.render_stats.start();
        self.render_stats.update_gui.start();
        let _full_output = self.update_gui();
        self.render_stats.update_gui.stop();
        self.vk_ctx.perf_stats().lap("update_gui()");

        self.render_stats.execute.start();
        // TODO: add more stuff to resource_map instead of recreating it all the time. From Marc:
        //  "Or even if you use the same resources all the time, it's useful to use the resource map
        //   for frame-local resources such as uniform buffers. All a ResourceMap is for is to
        //   allow you to specify resources at task graph runtime rather than compile time. If you
        //   use physical resources, you have to recompile the task graph each time."
        let resource_map = resource_map!(
            &self.task_graph,
            self.virtual_swapchain_id => self.vk_ctx.swapchain_id(),
        )
        .map_err(gg_err::CatchOutOfDate::from)?;
        self.vk_ctx.perf_stats().lap("resource_map!()");

        unsafe {
            self.task_graph
                .execute(resource_map, &self.vk_ctx, || {
                    self.window.inner.pre_present_notify();
                })
                .map_err(gg_err::CatchOutOfDate::from)?;
        }
        self.render_stats.execute.stop();
        self.vk_ctx.perf_stats().lap("render_stats.execute()");

        self.last_render_stats = self.render_stats.end();
        Ok(())
    }

    fn recreate_swapchain(&mut self) -> Result<(), gg_err::CatchOutOfDate> {
        _ = self.recreate_swapchain_rx.try_iter().last();
        self.vk_ctx
            .recreate_swapchain(&self.window)
            .context("could not recreate swapchain")?;
        self.render_handler
            .on_recreate_swapchain(self.window.clone());
        Ok(())
    }
}

fn build_task_graph(
    vk_ctx: &VulkanoContext,
    render_handler: &RenderHandler,
    resource_handler: &ResourceHandler,
) -> Result<(ExecutableTaskGraph<VulkanoContext>, Id<Swapchain>)> {
    // TODO: verbose!
    info_every_seconds!(1, "building task graph");
    let mut task_graph = TaskGraph::new(&vk_ctx.resources(), 100, 10000);
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

struct WindowEventHandlerCreateInfo<F>
where
    F: FnOnce(SceneHandlerBuilder) -> SceneHandler + Send + 'static,
{
    window_size: Vec2i,
    scene_handler_builder_callback: Option<F>,
    global_scale_factor: f32,
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
        global_scale_factor: f32,
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
                global_scale_factor,
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

    fn create_inner(
        &mut self,
        event_loop: &ActiveEventLoop,
        scene_handler_builder_callback: SceneHandlerBuilderCallback,
    ) -> Result<()> {
        info!("call create_inner()");
        let window = GgWindow::new(event_loop, self.create_info.window_size)?;
        let scale_factor = window.scale_factor();
        let viewport = UniqueShared::new(window.create_default_viewport());

        let vk_ctx = VulkanoContext::new(event_loop, &window)?;
        let resource_handler = ResourceHandler::new(&vk_ctx)?;
        // TODO: have preloading somehow.

        // TODO: these need a barrier between executions because they access the current image.
        //       That means the order of this vector matters. Need some way to let the user decide
        //       for default shaders too.
        let shaders: Vec<UniqueShared<Box<dyn Shader>>> = vec![
            SpriteShader::create(vk_ctx.clone(), viewport.clone(), resource_handler.clone())?,
            WireframeShader::create(vk_ctx.clone(), viewport.clone())?,
        ];

        let gui_ctx = self.gui_ctx.take().unwrap();
        let render_handler = RenderHandler::new(
            &vk_ctx,
            gui_ctx.clone(),
            resource_handler.clone(),
            window.clone(),
            viewport.clone(),
            shaders,
        )?
        .with_global_scale_factor(self.create_info.global_scale_factor)
        .with_clear_col(self.create_info.clear_col);
        ensure_shaders_locked();

        let platform = egui_winit::State::new(
            gui_ctx.clone(),
            ViewportId::ROOT,
            &event_loop,
            Some(window.scale_factor()),
            None,
            None,
        );

        let (task_graph, virtual_swapchain_id) =
            build_task_graph(&vk_ctx, &render_handler, &resource_handler)?;

        let input_handler = self.input_handler.clone();
        let window_event_rx = self.window_event_rx.take().unwrap();
        let scale_factor_rx = self.scale_factor_rx.take().unwrap();
        let recreate_swapchain_rx = self.recreate_swapchain_rx.take().unwrap();

        let scene_handler_builder = SceneHandlerBuilder::new(
            input_handler.clone(),
            resource_handler.clone(),
            render_handler.clone(),
        );
        std::thread::spawn(move || {
            let mut scene_handler = scene_handler_builder_callback(scene_handler_builder);
            loop {
                scene_handler.run_update();
            }
        });

        std::thread::spawn(move || {
            let mut inner = WindowEventHandlerInner {
                window,
                scale_factor,
                vk_ctx,
                render_handler,
                input_handler,
                resource_handler,
                gui_ctx,
                platform,
                task_graph,
                virtual_swapchain_id,
                render_stats: RenderPerfStats::new(),
                last_render_stats: None,
                is_first_window_event: true,
                window_event_rx,
                scale_factor_rx,
                recreate_swapchain_rx,
            };
            loop {
                inner.run_update();
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
            self.create_inner(event_loop, scene_handler_builder_callback)
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

    last_perf_stats: Option<Box<RenderPerfStats>>,
}

impl RenderPerfStats {
    fn new() -> Self {
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
        let render_time = gg_float::micros(self.last_step.elapsed()) * 1000.0;
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
            let on_time_rate = self.on_time as f32 / self.count as f32 * 100.0;
            if on_time_rate.round() < 100.0 {
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
