use std::{cell::RefCell, env, rc::Rc, sync::{Arc, Mutex, MutexGuard}, time::Instant};
use std::marker::PhantomData;
use egui::{FullOutput, ViewportId, ViewportInfo};
use num_traits::Zero;

use vulkano::{
    format::Format,
    command_buffer::{
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
        CommandBufferExecFuture,
        PrimaryAutoCommandBuffer,
    },
    descriptor_set::allocator::{
        StandardDescriptorSetAllocator,
        StandardDescriptorSetAllocatorCreateInfo
    },
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device,
        DeviceCreateInfo,
        DeviceExtensions,
        Queue,
        QueueCreateInfo,
        QueueFlags,
        DeviceFeatures
    },
    image::{view::ImageView, Image, ImageUsage},
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
        SwapchainPresentInfo,
        ColorSpace,
        SurfaceInfo
    },
    sync::{
        future::{FenceSignalFuture, JoinFuture},
        GpuFuture,
    },
    Validated,
    VulkanError,
    VulkanLibrary,
};
use egui_winit::winit::{dpi::LogicalSize, event::WindowEvent, event_loop::EventLoop};
use egui_winit::winit::application::ApplicationHandler;
use egui_winit::winit::dpi::PhysicalSize;
use egui_winit::winit::event_loop::ActiveEventLoop;
use egui_winit::winit::keyboard::PhysicalKey;
use egui_winit::winit::window::{Window, WindowAttributes, WindowId};
use std::time::Duration;
use vulkano::pipeline::GraphicsPipeline;
use crate::{core::{
    input::InputHandler,
    prelude::*,
}, info_every_seconds, resource::ResourceHandler, util::{
    gg_time::TimeIt
}, warn_every_seconds};
use crate::core::ObjectTypeEnum;
use crate::core::render::RenderHandler;
use crate::gui::GuiContext;
use crate::shader::{ensure_shaders_locked, BasicShader, Shader, SpriteShader, TriangleFanShader, WireframeShader};
use crate::util::{gg_float, SceneHandlerBuilder, UniqueShared};

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
pub struct DataPerImage<T: Clone> {
    data: Vec<T>,
}

impl <T: Clone + Copy> DataPerImage<T> {
    pub fn new_with_value(ctx: &VulkanoContext, initial_value: T) -> Self {
        let data = vec![initial_value; ctx.images.get().len()];
        Self { data }
    }

    pub fn clone_from_value(&mut self, new_value: T) {
        self.data = vec![new_value; self.data.len()];
    }
}

impl<T: Clone> DataPerImage<T> {
    pub fn new_with_data(ctx: &VulkanoContext, data: Vec<T>) -> Self {
        check_eq!(data.len(), ctx.images.get().len());
        Self { data }
    }
    pub fn try_new_with_generator<F: Fn() -> Result<T>>(
        ctx: &VulkanoContext,
        generator: F,
    ) -> Result<Self> {
        let mut data = Vec::new();
        for _ in 0..ctx.images.get().len() {
            data.push(generator()?);
        }
        Ok(Self { data })
    }
    pub fn new_with_generator<F: Fn() -> T>(ctx: &VulkanoContext, generator: F) -> Self {
        let mut data = Vec::new();
        for _ in 0..ctx.images.get().len() {
            data.push(generator());
        }
        Self { data }
    }

    pub fn is_empty(&self) -> bool { self.data.is_empty() }
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn last_value(&self, per_image_ctx: &MutexGuard<PerImageContext>) -> &T {
        &self.data[per_image_ctx.last]
    }
    pub fn current_value(&self, per_image_ctx: &MutexGuard<PerImageContext>) -> &T {
        &self.data[per_image_ctx.current.expect("no current value?")]
    }
    pub fn last_value_mut(&mut self, per_image_ctx: &mut MutexGuard<PerImageContext>) -> &mut T {
        &mut self.data[per_image_ctx.last]
    }
    pub fn current_value_mut(&mut self, per_image_ctx: &mut MutexGuard<PerImageContext>) -> &mut T {
        &mut self.data[per_image_ctx.current.expect("no current value?")]
    }

    pub fn map<U: Clone, F>(&self, func: F) -> DataPerImage<U>
    where
        F: FnMut(&T) -> U,
    {
        DataPerImage::<U> {
            data: self.as_slice().iter().map(func).collect(),
        }
    }
    pub fn count<P>(&self, predicate: P) -> usize
    where
        P: Fn(&T) -> bool,
    {
        let mut rv = 0;
        for value in self.as_slice() {
            rv += usize::from(predicate(value));
        }
        rv
    }
    pub fn try_count<P>(&self, predicate: P) -> Result<usize>
    where
        P: Fn(&T) -> Result<bool>,
    {
        let mut rv = 0;
        for value in self.as_slice() {
            rv += usize::from(predicate(value)?);
        }
        Ok(rv)
    }
    fn as_slice(&self) -> &[T] {
        self.data.as_slice()
    }
}

#[derive(Clone)]
pub struct VulkanoContext {
    // Should only ever be created once:
    device: Arc<Device>,
    queue: Arc<Queue>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    per_image_ctx: UniqueShared<PerImageContext>,

    // May be recreated, e.g. due to window resizing:
    swapchain: UniqueShared<Arc<Swapchain>>,
    images: UniqueShared<DataPerImage<Arc<Image>>>,
    render_pass: UniqueShared<Arc<RenderPass>>,
    framebuffers: UniqueShared<DataPerImage<Arc<Framebuffer>>>,
    image_count: UniqueShared<usize>,
    pipelines: UniqueShared<Vec<UniqueShared<Option<Arc<GraphicsPipeline>>>>>,
}

fn device_extensions() -> DeviceExtensions {
    DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::empty()
    }
}

impl VulkanoContext {
    pub fn new(event_loop: &ActiveEventLoop, window: &GgWindow) -> Result<Self> {
        let start = Instant::now();
        let library = VulkanLibrary::new().context("vulkano: no local Vulkan library/DLL")?;
        let instance = macos_instance(event_loop, library)?;
        let surface = Surface::from_window(instance.clone(), window.inner.clone())?;
        let physical_device = any_physical_device(&instance, &surface)?;
        info!("physical device: {physical_device:?}");
        let (device, queue) = any_graphical_queue_family(physical_device.clone())?;
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        ));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            StandardDescriptorSetAllocatorCreateInfo {
                update_after_bind: true,
                ..Default::default()
            }
        ));
        let (swapchain, images) = create_swapchain(
            window,
            surface.clone(),
            &physical_device,
            device.clone(),
        )?;
        let swapchain = UniqueShared::new(swapchain);
        let images = UniqueShared::new(DataPerImage { data: images });
        let render_pass = UniqueShared::new(create_render_pass(device.clone(), &swapchain)?);
        let framebuffers = UniqueShared::new(create_framebuffers(images.get().as_slice(), &render_pass.get())?);
        let image_count = UniqueShared::new(framebuffers.get().len());

        check_eq!(swapchain.get().image_count() as usize, images.get().len());

        info!(
            "created vulkano context in: {:.2} ms",
            gg_float::micros(start.elapsed()) * 1000.
        );
        Ok(Self {
            device,
            queue,
            memory_allocator,
            command_buffer_allocator,
            descriptor_set_allocator,

            swapchain,
            per_image_ctx: PerImageContext::new(),
            images,
            render_pass,
            framebuffers,
            image_count,
            pipelines: UniqueShared::new(Vec::new()),
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
    pub fn command_buffer_allocator(&self) -> Arc<StandardCommandBufferAllocator> {
        self.command_buffer_allocator.clone()
    }
    pub fn descriptor_set_allocator(&self) -> Arc<StandardDescriptorSetAllocator> {
        self.descriptor_set_allocator.clone()
    }

    // Warning: this object may become invalid when recreate_swapchain() is called.
    fn swapchain_cloned(&self) -> Arc<Swapchain> {
        self.swapchain.get().clone()
    }

    // When the created pipeline is invalidated, it will be destroyed.
    pub fn create_pipeline<F>(&mut self, f: F) -> Result<UniqueShared<Option<Arc<GraphicsPipeline>>>>
    where F: FnOnce(Arc<RenderPass>) -> Result<Arc<GraphicsPipeline>>
    {
        let pipeline = UniqueShared::new(Some(f(self.render_pass.get().clone())?));
        self.pipelines.get().push(pipeline.clone());
        Ok(pipeline)
    }
    pub fn image_count(&self) -> usize { *self.image_count.get() }

    fn recreate_swapchain(&mut self, window: &GgWindow) -> Result<()> {
        let swapchain_create_info = SwapchainCreateInfo {
            image_extent: window.inner_size().into(),
            ..self.swapchain.get().create_info()
        };
        let (new_swapchain, new_images) = self.swapchain.get()
            .recreate(swapchain_create_info).map_err(Validated::unwrap)?;
        *self.swapchain.get() = new_swapchain;
        *self.images.get() = DataPerImage { data: new_images };
        *self.render_pass.get() = create_render_pass(self.device.clone(), &self.swapchain)?;
        *self.framebuffers.get() = create_framebuffers(self.images.get().as_slice(), &self.render_pass.get())?;
        *self.image_count.get() = self.framebuffers.get().len();
        for pipeline in self.pipelines.get().drain(..) {
            *pipeline.get() = None;
        }
        Ok(())
    }
}

fn macos_instance(
    event_loop: &ActiveEventLoop,
    library: Arc<VulkanLibrary>,
) -> Result<Arc<Instance>> {
    if env::consts::OS == "macos" {
        let var = match env::var("MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS") {
            Ok(var) => var,
            Err(e) => {
                panic!("on macOS, environment variable `MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS` must be set; \
                        do you have .cargo/config.toml set up correctly? got: {e:?}");
            }
        };
        check_eq!(var, "1");
    }
    let required_extensions = Surface::required_extensions(&event_loop)?;
    let instance_create_info = InstanceCreateInfo {
        flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
        enabled_extensions: required_extensions,
        ..Default::default()
    };
    Instance::new(library, instance_create_info).context("vulkano: failed to create instance")
}
// TODO: more flexible approach here.
fn features() -> DeviceFeatures {
    DeviceFeatures {
        // Required for extra texture samplers on macOS:
        descriptor_indexing: true,
        fill_mode_non_solid: true,
        ..Default::default()
    }
}
fn any_physical_device(
    instance: &Arc<Instance>,
    surface: &Arc<Surface>,
) -> Result<Arc<PhysicalDevice>> {
    Ok(instance
        .enumerate_physical_devices()?
        .filter(|p| p.supported_extensions().contains(&device_extensions()))
        .filter(|p| p.supported_features().contains(&features()))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    #[allow(clippy::cast_possible_truncation)]
                    let i = i as u32;
                    q.queue_flags.contains(QueueFlags::GRAPHICS)
                        && p.surface_support(i, surface).unwrap_or(false)
                })
                .map(|q| {
                    #[allow(clippy::cast_possible_truncation)]
                    (p, q as u32)
                })
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
    #[allow(clippy::cast_possible_truncation)]
    let queue_family_index = physical_device
        .queue_family_properties()
        .iter()
        .position(|queue_family_properties| {
            queue_family_properties
                .queue_flags
                .contains(QueueFlags::GRAPHICS)
        })
        .context("vulkano: couldn't find a graphical queue family")?;
    info!("queue family properties: {:?}",
        physical_device.queue_family_properties()[queue_family_index]);
    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index: queue_family_index as u32,
                ..Default::default()
            }],
            enabled_extensions: device_extensions(),
            enabled_features: features(),
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
    window: &GgWindow,
    surface: Arc<Surface>,
    physical_device: &Arc<PhysicalDevice>,
    device: Arc<Device>,
) -> Result<(Arc<Swapchain>, Vec<Arc<Image>>)> {
    let caps = physical_device.surface_capabilities(&surface, SurfaceInfo::default())?;
    let dimensions = window.inner_size();
    let composite_alpha = caps
        .supported_composite_alpha
        .into_iter()
        .next()
        .context("vulkano: no composite alpha modes supported")?;
    let supported_formats = physical_device
        .surface_formats(&surface, SurfaceInfo::default())?;
    if !supported_formats.contains(&(Format::B8G8R8A8_SRGB, ColorSpace::SrgbNonLinear)) {
        error!("supported formats missing (Format::B8G8R8A8_SRGB, ColorSpace::SrgbNonLinear):\n{:?}", supported_formats);
    }
    info!("surface capabilities: {caps:?}");
    Ok(Swapchain::new(
        device,
        surface,
        SwapchainCreateInfo {
            min_image_count: caps.min_image_count + 1,
            image_format: Format::B8G8R8A8_SRGB,
            image_color_space: ColorSpace::SrgbNonLinear,
            image_extent: dimensions.into(),
            image_usage: ImageUsage::COLOR_ATTACHMENT,
            composite_alpha,
            ..Default::default()
        },
    )?)
}

fn create_render_pass(
    device: Arc<Device>,
    swapchain: &UniqueShared<Arc<Swapchain>>
) -> Result<Arc<RenderPass>> {
    Ok(vulkano::single_pass_renderpass!(
        device,
        attachments: {
            color: {
                // Set the format the same as the swapchain.
                format: swapchain.get().image_format(),
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
    render_pass: &Arc<RenderPass>,
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
            .collect::<Result<Vec<_>, _>>().map_err(Validated::unwrap)?,
    })
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
    is_ready: bool,
    last_ready_poll: Instant,
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
            is_ready: false,
            last_ready_poll: Instant::now(),

        };

        let event_loop = EventLoop::new()?;
        Ok(event_loop.run_app(&mut this)?)
    }

    fn expect_inner(&mut self) -> &mut WindowEventHandlerInner {
        self.inner.as_mut().expect("missing WindowEventHandlerInner")
    }

    fn recreate_swapchain(&mut self) -> Result<()> {
        self.expect_inner().fences = DataPerImage::new_with_generator(&self.expect_inner().vk_ctx, || Rc::new(RefCell::new(None)));
        let window = self.expect_inner().window.clone();
        self.expect_inner().vk_ctx.recreate_swapchain(&window)
            .context("could not recreate swapchain")?;
        self.expect_inner().render_handler.on_recreate_swapchain(window);
        Ok(())
    }

    fn idle(
        &mut self,
        image_idx: usize,
        acquire_future: SwapchainAcquireFuture,
        full_output: FullOutput
    ) -> Result<()> {
        let per_image_ctx = self.expect_inner().vk_ctx.per_image_ctx.clone();
        let mut per_image_ctx = per_image_ctx.get();
        per_image_ctx.current.replace(image_idx);

        self.render_stats.synchronise.start();
        let ready_future = self.synchronise(&mut per_image_ctx, acquire_future)?;
        self.render_stats.synchronise.stop();
        self.render_stats.do_render.start();
        let vk_ctx = self.expect_inner().vk_ctx.clone();
        let command_buffer = self.expect_inner().render_handler.do_render(
            &vk_ctx,
            vk_ctx.framebuffers.get().current_value(&per_image_ctx),
            full_output
        )?;
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
    ) -> Result<SwapchainJoinFuture> {
        let vk_ctx = self.expect_inner().vk_ctx.clone();
        if let Some(uploads) = self.expect_inner().resource_handler.texture
                .wait_build_command_buffer(&vk_ctx)? {
            uploads.flush()?;
            info!("loaded textures");
        }
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
        Ok(next_fence.join(acquire_future))
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
                    SwapchainPresentInfo::swapchain_image_index(
                        vk_ctx.swapchain_cloned(),
                        u32::try_from(image_idx)
                            .context("image_idx overflowed: {image_idx}")?
                    ),
                )
                .then_signal_fence_and_flush()?
            );
        Ok(())
    }

    fn poll_ready(&mut self) -> bool {
        if !self.is_ready && self.last_ready_poll.elapsed().as_millis() >= 10 {
            self.is_ready = self.expect_inner().render_handler.get_receiver().lock().unwrap().is_ready();
            self.last_ready_poll = Instant::now();
        }
        self.is_ready
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
                while !self.poll_ready() {}
                // XXX: macOS seems to behave weirdly badly with then_signal_fence_and_flush()
                //      if you send commands too fast.
                // TODO: test effects of this on Windows/Linux.
                if self.render_stats.penultimate_step.elapsed().as_millis() >= 15 {
                    let acquired = {
                        let swapchain = self.expect_inner().vk_ctx.swapchain_cloned();
                        // XXX: "acquire_next_image" is somewhat misleading, since it does not block
                        swapchain::acquire_next_image(swapchain, None)
                            .map_err(Validated::unwrap)
                    };
                    self.handle_acquired_image(acquired).unwrap();
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
    fn handle_acquired_image(&mut self,
                             acquired: Result<(u32, bool, SwapchainAcquireFuture), VulkanError>
    ) -> Result<()> {
        self.render_stats.start();

        let rv = match acquired {
            Ok((image_idx, /* suboptimal= */ false, acquire_future)) => {
                let full_output = self.handle_egui();
                self.idle(image_idx as usize, acquire_future, full_output)?;
                Ok(())
            },
            Ok((_, /* suboptimal= */ true, _)) => {
                info_every_seconds!(1, "suboptimal: recreating swapchain");
                self.recreate_swapchain()?;
                Ok(())
            }
            Err(VulkanError::OutOfDate) => {
                info_every_seconds!(1, "VulkanError::OutOfDate: recreating swapchain");
                self.recreate_swapchain()?;
                Ok(())
            },
            Err(e) => Err(e.into()),
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
