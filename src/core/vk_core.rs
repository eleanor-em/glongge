use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use anyhow::{Context, Result};
use tracing::{info, warn};

use crate::assert::*;

use vulkano::{
    command_buffer::{
        allocator::{
            StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
        },
        CommandBufferExecFuture,
        PrimaryCommandBufferAbstract
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::{
        Device,
        DeviceCreateInfo,
        DeviceExtensions,
        Queue,
        QueueCreateInfo,
        QueueFlags,
        physical::{PhysicalDevice, PhysicalDeviceType}
    },
    image::{
        Image,
        ImageUsage,
        view::ImageView
    },
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::StandardMemoryAllocator,
    pipeline::graphics::viewport::Viewport,
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass},
    swapchain::{
        self,
        PresentFuture,
        Surface,
        Swapchain,
        SwapchainAcquireFuture,
        SwapchainCreateInfo,
        SwapchainPresentInfo
    },
    sync::{
        GpuFuture,
        future::{FenceSignalFuture, JoinFuture}
    },
    Validated,
    VulkanError,
    VulkanLibrary
};
use vulkano::command_buffer::PrimaryAutoCommandBuffer;
use winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder}
};

use crate::core::util::TimeIt;
use crate::gg::core::RenderDataReceiver;

pub struct WindowContext {
    event_loop: EventLoop<()>,
    window: Arc<Window>,
}

impl WindowContext {
    pub fn new() -> Result<Self> {
        let event_loop = EventLoop::new();
        let window = Arc::new(
            WindowBuilder::new()
                .with_inner_size(LogicalSize::new(1024, 768))
                .build(&event_loop)?,
        );
        Ok(Self { event_loop, window })
    }

    fn event_loop(&self) -> &EventLoop<()> {
        &self.event_loop
    }
    fn window(&self) -> Arc<Window> {
        self.window.clone()
    }

    pub fn consume(self) -> (EventLoop<()>, Arc<Window>) {
        (self.event_loop, self.window)
    }

    pub fn create_default_viewport(&self) -> Viewport {
        Viewport {
            offset: [0.0, 0.0],
            extent: self.window.inner_size().into(),
            depth_range: 0.0..=1.0,
        }
    }
}

#[derive(Clone)]
pub struct DataPerImage<T: Clone> {
    data: Vec<T>,
}

impl<T: Clone> DataPerImage<T> {
    pub fn new_with_data(ctx: &VulkanoContext, data: Vec<T>) -> Self {
        check_eq!(data.len(), ctx.images.len());
        Self { data }
    }
    pub fn new_with_value(ctx: &VulkanoContext, initial_value: T) -> Self {
        let data = vec![initial_value; ctx.images.len()];
        Self { data }
    }
    pub fn try_new_with_generator<F: Fn() -> Result<T>>(ctx: &VulkanoContext, generator: F) -> Result<Self> {
        let mut data = Vec::new();
        for _ in 0..ctx.images.len() {
            data.push(generator()?);
        }
        Ok(Self { data })
    }
    pub fn new_with_generator<F: Fn() -> T>(ctx: &VulkanoContext, generator: F) -> Self {
        let mut data = Vec::new();
        for _ in 0..ctx.images.len() {
            data.push(generator());
        }
        Self { data }
    }

    pub fn clone_from_value(&mut self, new_value: T) {
        self.data = vec![new_value; self.data.len()];
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn last_value(&self, per_image_ctx: &MutexGuard<PerImageContext>) -> &T {
        &self.data[per_image_ctx.last]
    }
    pub fn current_value(&self, per_image_ctx: &MutexGuard<PerImageContext>) -> &T {
        &self.data[per_image_ctx.current.unwrap()]
    }
    pub fn last_value_mut(&mut self, per_image_ctx: &mut MutexGuard<PerImageContext>) -> &mut T {
        &mut self.data[per_image_ctx.last]
    }
    pub fn current_value_mut(&mut self, per_image_ctx: &mut MutexGuard<PerImageContext>) -> &mut T {
        &mut self.data[per_image_ctx.current.unwrap()]
    }

    pub fn map<U: Clone, F>(&self, func: F) -> DataPerImage<U>
    where
        F: FnMut(&T) -> U,
    {
        DataPerImage::<U> {
            data: self.as_slice().iter().map(func).collect(),
        }
    }
    pub fn try_map<U: Clone, F>(&self, func: F) -> Result<DataPerImage<U>>
    where
        F: FnMut(&T) -> Result<U>,
    {
        Ok(DataPerImage::<U> {
            data: self.as_slice().iter().map(func).try_collect()?,
        })
    }
    pub fn try_map_with<U: Clone, V: Clone, F>(
        &self,
        other: &DataPerImage<U>,
        func: F,
    ) -> Result<DataPerImage<V>>
    where
        F: FnMut((&T, &U)) -> Result<V>,
    {
        Ok(DataPerImage::<V> {
            data: self
                .as_slice()
                .iter()
                .zip(other.as_slice())
                .map(func)
                .try_collect()?,
        })
    }
    pub fn try_map_with_3<U: Clone, V: Clone, W: Clone, F>(
        &self,
        other1: &DataPerImage<U>,
        other2: &DataPerImage<V>,
        func: F,
    ) -> Result<DataPerImage<W>>
    where
        F: FnMut(((&T, &U), &V)) -> Result<W>,
    {
        Ok(DataPerImage::<W> {
            data: self
                .as_slice()
                .iter()
                .zip(other1.as_slice())
                .zip(other2.as_slice())
                .map(func)
                .try_collect()?,
        })
    }
    pub fn count<P>(&self, predicate: P) -> usize
    where
        P: Fn(&T) -> bool,
    {
        let mut rv = 0;
        for value in self.as_slice() {
            rv += predicate(value) as usize;
        }
        rv
    }
    pub fn try_count<P>(&self, predicate: P) -> Result<usize>
    where
        P: Fn(&T) -> Result<bool>,
    {
        let mut rv = 0;
        for value in self.as_slice() {
            rv += predicate(value)? as usize;
        }
        Ok(rv)
    }
    fn as_slice(&self) -> &[T] {
        self.data.as_slice()
    }
}

pub struct VulkanoContext {
    device: Arc<Device>,
    queue: Arc<Queue>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    swapchain: Arc<Swapchain>,
    per_image_ctx: Arc<Mutex<PerImageContext>>,
    images: DataPerImage<Arc<Image>>,
    render_pass: Arc<RenderPass>,
    framebuffers: DataPerImage<Arc<Framebuffer>>,
}

fn device_extensions() -> DeviceExtensions {
    DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::empty()
    }
}

impl VulkanoContext {
    pub fn new(window_ctx: &WindowContext) -> Result<Self> {
        let start = Instant::now();
        let library = VulkanLibrary::new().context("vulkano: no local Vulkan library/DLL")?;
        let instance = macos_instance(window_ctx.event_loop(), library)?;
        let surface = Surface::from_window(instance.clone(), window_ctx.window())?;
        let physical_device = any_physical_device(instance.clone(), surface.clone())?;
        let (device, queue) = any_graphical_queue_family(physical_device.clone())?;
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        ));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));
        let (swapchain, images) = create_swapchain(
            window_ctx.window(),
            surface.clone(),
            physical_device.clone(),
            device.clone(),
        )?;
        let images = DataPerImage { data: images };
        let render_pass = create_render_pass(device.clone(), swapchain.clone())?;
        let framebuffers = create_framebuffers(images.as_slice(), render_pass.clone())?;

        check_eq!(swapchain.image_count() as usize, images.len());

        info!(
            "created vulkano context in: {:.2} ms",
            start.elapsed().as_micros() as f64 / 1_000.0
        );
        Ok(Self {
            // Appears to not be necessary:
            // surface,
            device,
            queue,
            memory_allocator,
            command_buffer_allocator,
            descriptor_set_allocator,
            swapchain,
            images,
            render_pass,
            framebuffers,
            per_image_ctx: PerImageContext::new(),
        })
    }

    pub fn device(&self) -> Arc<Device> {
        self.device.clone()
    }
    pub fn queue(&self) -> Arc<Queue> {
        self.queue.clone()
    }
    pub fn memory_allocator(&self) -> Arc<StandardMemoryAllocator> {
        self.memory_allocator.clone()
    }
    pub fn command_buffer_allocator(&self) -> &StandardCommandBufferAllocator {
        &self.command_buffer_allocator
    }
    pub fn descriptor_set_allocator(&self) -> Arc<StandardDescriptorSetAllocator> {
        self.descriptor_set_allocator.clone()
    }
    fn swapchain(&self) -> Arc<Swapchain> {
        self.swapchain.clone()
    }
    pub fn render_pass(&self) -> Arc<RenderPass> {
        self.render_pass.clone()
    }
    pub fn framebuffers(&self) -> DataPerImage<Arc<Framebuffer>> {
        self.framebuffers.clone()
    }

    fn recreate_swapchain(&mut self, window: Arc<Window>) -> Result<()> {
        let (new_swapchain, new_images) = self.swapchain.recreate(SwapchainCreateInfo {
            image_extent: window.inner_size().into(),
            ..self.swapchain.create_info()
        })?;
        self.swapchain = new_swapchain;
        self.framebuffers = create_framebuffers(&new_images, self.render_pass.clone())?;
        Ok(())
    }
}

fn macos_instance<T>(
    event_loop: &EventLoop<T>,
    library: Arc<VulkanLibrary>,
) -> Result<Arc<Instance>> {
    let required_extensions = Surface::required_extensions(event_loop);
    let instance_create_info = InstanceCreateInfo {
        flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
        enabled_extensions: required_extensions,
        ..Default::default()
    };
    Instance::new(library, instance_create_info).context("vulkano: failed to create instance")
}
fn any_physical_device(
    instance: Arc<Instance>,
    surface: Arc<Surface>,
) -> Result<Arc<PhysicalDevice>> {
    Ok(instance
        .enumerate_physical_devices()?
        .filter(|p| p.supported_extensions().contains(&device_extensions()))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    q.queue_flags.contains(QueueFlags::GRAPHICS)
                        && p.surface_support(i as u32, &surface).unwrap_or(false)
                })
                .map(|q| (p, q as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            _ => 4,
        })
        .context("vulkano: no appropriate physical device available")?
        .0)
}
fn any_graphical_queue_family(
    physical_device: Arc<PhysicalDevice>,
) -> Result<(Arc<Device>, Arc<Queue>)> {
    let queue_family_index = physical_device
        .queue_family_properties()
        .iter()
        .enumerate()
        .position(|(_queue_family_index, queue_family_properties)| {
            queue_family_properties
                .queue_flags
                .contains(QueueFlags::GRAPHICS)
        })
        .context("vulkano: couldn't find a graphical queue family")?
        as u32;
    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            enabled_extensions: device_extensions(),
            ..Default::default()
        },
    )?;
    info!("found {} queue(s), using first", queues.len());
    Ok((
        device,
        queues.next().context("vulkano: UNEXPECTED: zero queues?")?,
    ))
}

fn create_swapchain(
    window: Arc<Window>,
    surface: Arc<Surface>,
    physical_device: Arc<PhysicalDevice>,
    device: Arc<Device>,
) -> Result<(Arc<Swapchain>, Vec<Arc<Image>>)> {
    let caps = physical_device.surface_capabilities(&surface, Default::default())?;
    let dimensions = window.inner_size();
    let composite_alpha = caps
        .supported_composite_alpha
        .into_iter()
        .next()
        .context("vulkano: no composite alpha modes supported")?;
    let image_format = physical_device
        .surface_formats(&surface, Default::default())?
        .first()
        .context("vulkano: no surface formats found")?
        .0;
    Ok(Swapchain::new(
        device.clone(),
        surface.clone(),
        SwapchainCreateInfo {
            min_image_count: caps.min_image_count + 1,
            image_format,
            image_extent: dimensions.into(),
            image_usage: ImageUsage::COLOR_ATTACHMENT,
            composite_alpha,
            ..Default::default()
        },
    )?)
}

fn create_render_pass(device: Arc<Device>, swapchain: Arc<Swapchain>) -> Result<Arc<RenderPass>> {
    Ok(vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
                // Set the format the same as the swapchain.
                format: swapchain.image_format(),
                samples: 1,
                load_op: Clear,
                store_op: Store,
            },
        },
        pass: {
            color: [color],
            depth_stencil: {},
        },
    )?)
}

fn create_framebuffers(
    images: &[Arc<Image>],
    render_pass: Arc<RenderPass>,
) -> Result<DataPerImage<Arc<Framebuffer>>> {
    Ok(DataPerImage {
        data: images
            .iter()
            .map(|image| {
                let view = ImageView::new_default(image.clone())?;
                Framebuffer::new(
                    render_pass.clone(),
                    FramebufferCreateInfo {
                        attachments: vec![view],
                        ..Default::default()
                    },
                )
            })
            .try_collect()?,
    })
}

#[derive(Clone)]
pub struct PerImageContext {
    last: usize,
    current: Option<usize>,
}

impl PerImageContext {
    fn new() -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self {
            last: 0,
            current: None,
        }))
    }
}

pub trait RenderEventHandler<CommandBuffer: PrimaryCommandBufferAbstract = PrimaryAutoCommandBuffer> {
    type DataReceiver: RenderDataReceiver + 'static;

    fn on_resize(
        &mut self,
        ctx: &VulkanoContext,
        per_image_ctx: &mut MutexGuard<PerImageContext>,
        window: Arc<Window>,
    ) -> Result<()>;
    fn on_render(
        &mut self,
        ctx: &VulkanoContext,
        per_image_ctx: &mut MutexGuard<PerImageContext>,
    ) -> Result<DataPerImage<Arc<CommandBuffer>>>;

    fn get_receiver(&self) -> Arc<Mutex<Self::DataReceiver>>;
}

type SwapchainJoinFuture = JoinFuture<Box<dyn GpuFuture>, SwapchainAcquireFuture>;
type FenceFuture = FenceSignalFuture<PresentFuture<CommandBufferExecFuture<SwapchainJoinFuture>>>;

pub struct WindowEventHandler<
    CommandBuffer: PrimaryCommandBufferAbstract,
    RenderHandler: RenderEventHandler<CommandBuffer> + 'static,
> {
    window: Arc<Window>,
    ctx: VulkanoContext,
    render_handler: RenderHandler,

    window_was_resized: bool,
    should_recreate_swapchain: bool,
    fences: DataPerImage<Rc<RefCell<Option<FenceFuture>>>>,
    render_stats: RenderPerfStats,
    command_buffer_type: PhantomData<CommandBuffer>,
}

impl<
        CommandBuffer: PrimaryCommandBufferAbstract + 'static,
        RenderHandler: RenderEventHandler<CommandBuffer> + 'static,
    > WindowEventHandler<CommandBuffer, RenderHandler>
{
    pub fn new(window: Arc<Window>, ctx: VulkanoContext, handler: RenderHandler) -> Self {
        let fences = DataPerImage::new_with_value(&ctx, Rc::new(RefCell::new(None)));
        Self {
            window,
            ctx,
            render_handler: handler,
            window_was_resized: false,
            should_recreate_swapchain: false,
            fences,
            render_stats: RenderPerfStats::new(),
            command_buffer_type: PhantomData,
        }
    }

    pub fn run(mut self, event_loop: EventLoop<()>) {
        event_loop.run(move |event, _, control_flow| self.run_inner(event, control_flow).unwrap());
    }

    fn maybe_recreate_swapchain(
        &mut self,
        per_image_ctx: &mut MutexGuard<PerImageContext>,
    ) -> Result<()> {
        if self.window_was_resized || self.should_recreate_swapchain {
            self.should_recreate_swapchain = false;
            self.ctx
                .recreate_swapchain(self.window.clone())
                .context("could not recreate swapchain")?;
        }
        if self.window_was_resized {
            self.window_was_resized = false;
            self.render_handler
                .on_resize(&self.ctx, per_image_ctx, self.window.clone())?;
        }
        Ok(())
    }

    fn idle(
        &mut self,
        per_image_ctx: &mut MutexGuard<PerImageContext>,
        acquire_future: SwapchainAcquireFuture,
    ) -> Result<()> {
        self.render_stats.begin_acquire_and_sync();
        let ready_future = self.acquire_and_synchronise(per_image_ctx, acquire_future)?;
        self.render_stats.begin_on_render();
        let command_buffers = self.render_handler.on_render(&self.ctx, per_image_ctx)?;
        self.render_stats.begin_submit_command_buffers();
        self.submit_command_buffers(per_image_ctx, command_buffers, ready_future)?;
        self.render_stats.end_render();
        self.update_last_image_idx(per_image_ctx)?;
        Ok(())
    }

    fn acquire_and_synchronise(
        &mut self,
        per_image_ctx: &mut MutexGuard<PerImageContext>,
        acquire_future: SwapchainAcquireFuture,
    ) -> Result<SwapchainJoinFuture> {
        self.render_stats.pause_render_active();
        if let Some(fence) = self.fences.current_value(per_image_ctx).borrow().as_ref() {
            fence.wait(None)?;
        }
        self.render_stats.unpause_render_active();
        let last_fence = match self.fences.last_value(per_image_ctx).take() {
            Some(fence) => fence.boxed(),
            _ => {
                // synchronise only if there is no previous future (swapchain was just created)
                info!("synchronised via vulkano::sync::now()");
                let mut now = vulkano::sync::now(self.ctx.device());
                now.cleanup_finished();
                now.boxed()
            }
        };
        Ok(last_fence.join(acquire_future))
    }

    fn submit_command_buffers(
        &mut self,
        per_image_ctx: &mut MutexGuard<PerImageContext>,
        command_buffers: DataPerImage<Arc<CommandBuffer>>,
        ready_future: SwapchainJoinFuture,
    ) -> Result<()> {
        // TODO: raise custom error type instead of unwrap()
        let image_idx = per_image_ctx.current.unwrap();
        self.fences
            .current_value_mut(per_image_ctx)
            .borrow_mut()
            .replace(
                ready_future
                    .then_execute(
                        self.ctx.queue(),
                        command_buffers.current_value(per_image_ctx).clone(),
                    )?
                    .then_swapchain_present(
                        self.ctx.queue(),
                        SwapchainPresentInfo::swapchain_image_index(
                            self.ctx.swapchain(),
                            image_idx as u32,
                        ),
                    )
                    .then_signal_fence(),
            );
        Ok(())
    }

    fn update_last_image_idx(
        &mut self,
        per_image_ctx: &mut MutexGuard<PerImageContext>,
    ) -> Result<()> {
        let expected_image_idx = (per_image_ctx.last + 1) % self.ctx.images.len();
        let image_idx = per_image_ctx.current.unwrap();
        if image_idx != expected_image_idx && per_image_ctx.last != image_idx {
            info!(
                "out-of-order framebuffer: {} -> {}",
                per_image_ctx.last, image_idx
            );
        }
        per_image_ctx.last = image_idx;
        Ok(())
    }

    fn run_inner(&mut self, event: Event<()>, control_flow: &mut ControlFlow) -> Result<()> {
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
                Ok(())
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                self.window_was_resized = true;
                Ok(())
            }
            Event::MainEventsCleared => {
                let per_image_ctx = self.ctx.per_image_ctx.clone();
                let mut per_image_ctx = per_image_ctx.lock().unwrap();
                self.render_stats.begin_handle_swapchain();
                self.maybe_recreate_swapchain(&mut per_image_ctx)?;
                // XXX: "acquire_next_image" is somewhat misleading, since it does not block
                let rv = match swapchain::acquire_next_image(self.ctx.swapchain(), None)
                    .map_err(Validated::unwrap)
                {
                    Ok((image_idx, suboptimal, acquire_future)) => {
                        if suboptimal {
                            self.should_recreate_swapchain = true;
                        }
                        per_image_ctx.current.replace(image_idx as usize);
                        self.idle(&mut per_image_ctx, acquire_future)
                    }
                    Err(VulkanError::OutOfDate) => {
                        self.should_recreate_swapchain = true;
                        Ok(())
                    }
                    Err(e) => Err(e.into()),
                };
                self.render_stats.report_and_end_step();
                rv
            }
            _ => Ok(()),
        }
    }
}

#[derive(Eq, PartialEq, Debug)]
enum RenderState {
    HandleSwapchain,
    AcquireAndSync,
    OnRender,
    SubmitCommandBuffers,
    EndRender,
    BetweenRenders,
}

struct RenderPerfStats {
    state: RenderState,
    handle_swapchain: TimeIt,
    acquire_and_sync: TimeIt,
    on_render: TimeIt,
    submit_command_buffers: TimeIt,
    end_step: TimeIt,
    render_wait: TimeIt,
    render_active: TimeIt,
    between_renders: TimeIt,
    total: TimeIt,
    on_time: u64,
    count: u64,
    last_report: Instant,
}

impl RenderPerfStats {
    fn new() -> Self {
        Self {
            state: RenderState::BetweenRenders,
            handle_swapchain: TimeIt::new("handle swapchain"),
            acquire_and_sync: TimeIt::new("acquire-and-sync"),
            on_render: TimeIt::new("on render"),
            submit_command_buffers: TimeIt::new("submit cmdbufs"),
            end_step: TimeIt::new("end step"),
            render_wait: TimeIt::new("wait for render"),
            render_active: TimeIt::new("render"),
            between_renders: TimeIt::new("between renders"),
            total: TimeIt::new("total"),
            on_time: 0,
            count: 0,
            last_report: Instant::now(),
        }
    }

    fn begin_handle_swapchain(&mut self) {
        check_eq!(self.state, RenderState::BetweenRenders);
        self.state = RenderState::HandleSwapchain;
        self.total.stop();
        self.total.start();
        self.between_renders.stop();
        self.render_active.start();
        self.handle_swapchain.start();
    }

    fn begin_acquire_and_sync(&mut self) {
        check_eq!(self.state, RenderState::HandleSwapchain);
        self.state = RenderState::AcquireAndSync;
        self.acquire_and_sync.start();
    }
    fn begin_on_render(&mut self) {
        check_eq!(self.state, RenderState::AcquireAndSync);
        self.state = RenderState::OnRender;
        self.acquire_and_sync.stop();
        self.on_render.start();
    }

    fn begin_submit_command_buffers(&mut self) {
        check_eq!(self.state, RenderState::OnRender);
        self.state = RenderState::SubmitCommandBuffers;
        self.on_render.stop();
        self.submit_command_buffers.start();
    }

    fn end_render(&mut self) {
        check_eq!(self.state, RenderState::SubmitCommandBuffers);
        self.state = RenderState::EndRender;
        self.submit_command_buffers.stop();
        self.end_step.start();
    }

    fn pause_render_active(&mut self) {
        self.render_active.pause();
        self.render_wait.start();
    }
    fn unpause_render_active(&mut self) {
        self.render_wait.stop();
        self.render_active.unpause();
    }

    fn report_and_end_step(&mut self) {
        // in some error conditions, we are in a different state at the end of a step:
        // crate::check_eq!(self.state, RenderState::EndRender);
        self.state = RenderState::BetweenRenders;
        self.end_step.stop();
        self.render_active.stop();

        // track how many frames are late
        let active_ms = self.render_active.last_ms() + self.between_renders.last_ms();
        if active_ms < 1000.0 / 60.0 {
            self.on_time += 1;
        } else {
            warn!("late frame: {active_ms:.2} ms");
        }
        self.count += 1;

        // arbitrary; report every 5 seconds
        if self.last_report.elapsed().as_secs() >= 5 {
            info!(
                "frames on time: {:.1}%",
                self.on_time as f64 / self.count as f64 * 100.0
            );
            let min_report_ms = 0.1;
            self.render_wait.report_ms_if_at_least(min_report_ms);
            self.render_active.report_ms_if_at_least(min_report_ms);
            self.between_renders.report_ms_if_at_least(min_report_ms);
            self.handle_swapchain.report_ms_if_at_least(min_report_ms);
            self.acquire_and_sync.report_ms_if_at_least(min_report_ms);
            self.on_render.report_ms_if_at_least(min_report_ms);
            self.submit_command_buffers.report_ms_if_at_least(min_report_ms);
            self.end_step.report_ms_if_at_least(min_report_ms);
            self.total.report_ms_if_at_least(min_report_ms);
            self.last_report = Instant::now();
            self.on_time = 0;
            self.count = 0;
        }
        self.between_renders.start();
    }
}
