use std::{cell::RefCell, env, rc::Rc, sync::{Arc, Mutex, MutexGuard}, time::Instant};
use egui::{FullOutput, ViewportId, ViewportInfo};
use egui_winit::EventResponse;
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
        Features
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
use egui_winit::winit::{dpi::LogicalSize, event::{Event, WindowEvent}, event_loop::EventLoop,  window::{Window, WindowBuilder}};
use egui_winit::winit::keyboard::PhysicalKey;

use crate::{
    core::{
        input::InputHandler,
        prelude::*,
    },
    resource::ResourceHandler,
    util::{
        gg_time::TimeIt
    },
};
use crate::core::render::RenderHandler;
use crate::gui::GuiContext;
use crate::shader::ensure_shaders_locked;
use crate::util::{gg_float, UniqueShared};

pub struct WindowContext {
    event_loop: EventLoop<()>,
    window: Arc<Window>,
}

impl WindowContext {
    pub fn new(size: impl Into<Vec2i>) -> Result<Self> {
        let size = size.into();
        let event_loop = EventLoop::new()?;
        let window = Arc::new(
            WindowBuilder::new()
                .with_inner_size(LogicalSize::new(size.x, size.y))
                .build(&event_loop)?,
        );
        Ok(Self { event_loop, window })
    }

    pub(crate) fn window(&self) -> Arc<Window> {
        self.window.clone()
    }

    pub fn consume(self) -> (EventLoop<()>, Arc<Window>) {
        (self.event_loop, self.window)
    }

    pub fn create_default_viewport(&self) -> UniqueShared<AdjustedViewport> {
        UniqueShared::new(AdjustedViewport {
            inner: Viewport {
                offset: [0., 0.],
                extent: self.window.inner_size().into(),
                depth_range: 0.0..=1.,
            },
            scale_factor: self.window.scale_factor(),
            global_scale_factor: 1.,
            translation: Vec2::zero(),
        })
    }
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
    pub fn update_from_window(&mut self, window: &Arc<Window>) {
        self.inner.extent = window.inner_size().into();
        self.scale_factor = window.scale_factor() * self.global_scale_factor;
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
        let data = vec![initial_value; ctx.images.len()];
        Self { data }
    }

    pub fn clone_from_value(&mut self, new_value: T) {
        self.data = vec![new_value; self.data.len()];
    }
}

impl<T: Clone> DataPerImage<T> {
    pub fn new_with_data(ctx: &VulkanoContext, data: Vec<T>) -> Self {
        check_eq!(data.len(), ctx.images.len());
        Self { data }
    }
    pub fn try_new_with_generator<F: Fn() -> Result<T>>(
        ctx: &VulkanoContext,
        generator: F,
    ) -> Result<Self> {
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
        let instance = macos_instance(&window_ctx.event_loop, library)?;
        let surface = compat::surface_from_window(instance.clone(), &window_ctx.window)?;
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
            &window_ctx.window(),
            surface.clone(),
            &physical_device,
            device.clone(),
        )?;
        let images = DataPerImage { data: images };
        let render_pass = create_render_pass(device.clone(), &swapchain)?;
        let framebuffers = create_framebuffers(images.as_slice(), &render_pass)?;

        check_eq!(swapchain.image_count() as usize, images.len());

        info!(
            "created vulkano context in: {:.2} ms",
            gg_float::micros(start.elapsed()) * 1000.
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

    fn recreate_swapchain(&mut self, window: &Arc<Window>) -> Result<()> {
        info!("recreate swapchain");
        let (new_swapchain, new_images) = self.swapchain.recreate(SwapchainCreateInfo {
            image_extent: window.inner_size().into(),
            ..self.swapchain.create_info()
        }).map_err(Validated::unwrap)?;
        self.swapchain = new_swapchain;
        self.images = DataPerImage { data: new_images };
        self.framebuffers = create_framebuffers(self.images.as_slice(), &self.render_pass)?;
        Ok(())
    }
}

mod compat {
    use anyhow::Result;
    use std::any::Any;
    use std::ffi::c_void;
    use std::sync::Arc;
    #[allow(deprecated)]
    use egui_winit::winit::raw_window_handle::{AppKitWindowHandle, HasDisplayHandle, HasRawDisplayHandle, HasRawWindowHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};
    use raw_window_handle::AppKitDisplayHandle;
    use vulkano::instance::{Instance, InstanceExtensions};
    use vulkano::swapchain::Surface;

    struct MacosWrapper {
        ns_view: *mut c_void,
    }

    impl From<AppKitWindowHandle> for MacosWrapper {
        fn from(value: AppKitWindowHandle) -> Self {
            Self {
                ns_view: value.ns_view.as_ptr()
            }
        }
    }

    unsafe impl raw_window_handle::HasRawWindowHandle for MacosWrapper {
        fn raw_window_handle(&self) -> raw_window_handle::RawWindowHandle {
            let mut handle = raw_window_handle::AppKitWindowHandle::empty();
            handle.ns_view = self.ns_view;
            raw_window_handle::RawWindowHandle::AppKit(handle)
        }
    }

    unsafe impl raw_window_handle::HasRawDisplayHandle for MacosWrapper {
        fn raw_display_handle(&self) -> raw_window_handle::RawDisplayHandle {
            // Unused
            raw_window_handle::RawDisplayHandle::AppKit(AppKitDisplayHandle::empty())
        }
    }

    pub fn surface_from_window(
        instance: Arc<Instance>,
        window: &Arc<impl HasWindowHandle + HasDisplayHandle + Any + Send + Sync>
    ) -> Result<Arc<Surface>> {
        unsafe { from_window_copied(instance, window) }
    }

    #[allow(deprecated)]
    unsafe fn from_window_copied(
        instance: Arc<Instance>,
        window: &Arc<impl HasWindowHandle + HasDisplayHandle + Any + Send + Sync>
    ) -> Result<Arc<Surface>> {
        match (window.window_handle()?.raw_window_handle()?, window.display_handle()?.raw_display_handle()?) {
            #[cfg(target_os = "macos")]
            (RawWindowHandle::AppKit(window), _) => {
                Ok(Surface::from_window_ref(instance, &MacosWrapper::from(window))?)
            }
            (RawWindowHandle::Wayland(window), RawDisplayHandle::Wayland(display)) => {
                Ok(Surface::from_wayland(instance, display.display.as_ptr(), window.surface.as_ptr(), None)?)
            }
            (RawWindowHandle::Win32(window), _) => {
                Ok(Surface::from_win32(instance,
                                       // These casts make no goddamn sense, but they work.
                                       window.hinstance.unwrap().get() as *mut isize,
                                       window.hwnd.get() as *const isize,
                                       None)?)
            }
            (RawWindowHandle::Xcb(window), RawDisplayHandle::Xcb(display)) => {
                Ok(Surface::from_xcb(instance, display.connection.unwrap().as_ptr(), window.window.into(), None)?)
            }
            (RawWindowHandle::Xlib(window), RawDisplayHandle::Xlib(display)) => {
                Ok(Surface::from_xlib(instance, display.display.unwrap().as_ptr(), window.window, None)?)
            }
            _ => unimplemented!(
                "the window was created with a windowing API that is not supported \
                by Vulkan/Vulkano"
            ),
        }
    }

    #[allow(deprecated)]
    pub fn required_extensions_copied(event_loop: &impl HasDisplayHandle) -> Result<InstanceExtensions> {
        let mut extensions = InstanceExtensions {
            khr_surface: true,
            ..InstanceExtensions::empty()
        };
        match event_loop.display_handle()?.raw_display_handle()? {
            RawDisplayHandle::Android(_) => extensions.khr_android_surface = true,
            // FIXME: `mvk_macos_surface` and `mvk_ios_surface` are deprecated.
            RawDisplayHandle::AppKit(_) => extensions.mvk_macos_surface = true,
            RawDisplayHandle::UiKit(_) => extensions.mvk_ios_surface = true,
            RawDisplayHandle::Windows(_) => extensions.khr_win32_surface = true,
            RawDisplayHandle::Wayland(_) => extensions.khr_wayland_surface = true,
            RawDisplayHandle::Xcb(_) => extensions.khr_xcb_surface = true,
            RawDisplayHandle::Xlib(_) => extensions.khr_xlib_surface = true,
            _ => unimplemented!(),
        }

        Ok(extensions)
    }
}
fn macos_instance<T>(
    event_loop: &EventLoop<T>,
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
    let required_extensions = compat::required_extensions_copied(&event_loop)?;
    let instance_create_info = InstanceCreateInfo {
        flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
        enabled_extensions: required_extensions,
        ..Default::default()
    };
    Instance::new(library, instance_create_info).context("vulkano: failed to create instance")
}
// TODO: more flexible approach here.
fn features() -> Features {
    Features {
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
    window: &Arc<Window>,
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

fn create_render_pass(device: Arc<Device>, swapchain: &Arc<Swapchain>) -> Result<Arc<RenderPass>> {
    Ok(vulkano::single_pass_renderpass!(
        device,
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
    fn new() -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self {
            last: 0,
            current: None,
        }))
    }
}

type SwapchainJoinFuture = JoinFuture<Box<dyn GpuFuture>, SwapchainAcquireFuture>;
type FenceFuture = FenceSignalFuture<PresentFuture<CommandBufferExecFuture<SwapchainJoinFuture>>>;

pub struct WindowEventHandler {
    window: Arc<Window>,
    scale_factor: f64,
    vk_ctx: VulkanoContext,
    gui_ctx: GuiContext,
    render_handler: RenderHandler,
    input_handler: Arc<Mutex<InputHandler>>,
    resource_handler: ResourceHandler,

    fences: DataPerImage<Rc<RefCell<Option<FenceFuture>>>>,
    render_stats: RenderPerfStats,
    last_render_stats: Option<RenderPerfStats>,

    is_ready: bool,
    last_ready_poll: Instant,
}

#[allow(private_bounds)]
impl WindowEventHandler {
    pub fn new(
        window: Arc<Window>,
        vk_ctx: VulkanoContext,
        gui_ctx: GuiContext,
        render_handler: RenderHandler,
        input_handler: Arc<Mutex<InputHandler>>,
        resource_handler: ResourceHandler,
    ) -> Self {
        let fences = DataPerImage::new_with_generator(&vk_ctx, || Rc::new(RefCell::new(None)));
        let scale_factor = window.scale_factor();
        Self {
            window,
            scale_factor,
            vk_ctx,
            gui_ctx,
            render_handler,
            input_handler,
            resource_handler,
            fences,
            render_stats: RenderPerfStats::new(),
            last_render_stats: None,
            is_ready: false,
            last_ready_poll: Instant::now(),
        }
    }

    pub fn consume(mut self, event_loop: EventLoop<()>) {
        let mut egui_window_state = egui_winit::State::new(
            self.gui_ctx.clone(),
            ViewportId::ROOT,
            &event_loop,
            Some(self.window.scale_factor() as f32),
            None
        );

        ensure_shaders_locked();
        event_loop.run(move |event, _| {
            self.run_inner(&mut egui_window_state, &event)
                .expect("error running event loop");
        })
            .expect("error running event loop");
    }

    fn recreate_swapchain(
        &mut self
    ) -> Result<()> {
        self.fences = DataPerImage::new_with_generator(&self.vk_ctx, || Rc::new(RefCell::new(None)));
        self.vk_ctx.recreate_swapchain(&self.window)
            .context("could not recreate swapchain")?;
        self.render_handler.on_resize(self.window.clone());
        Ok(())
    }

    fn idle(
        &mut self,
        per_image_ctx: &mut MutexGuard<PerImageContext>,
        acquire_future: SwapchainAcquireFuture,
        full_output: FullOutput
    ) -> Result<()> {
        self.render_stats.synchronise.start();
        let ready_future = self.synchronise(per_image_ctx, acquire_future)?;
        self.render_stats.synchronise.stop();
        self.render_stats.do_render.start();
        let command_buffer = self.render_handler.do_render(
            &self.vk_ctx,
            self.vk_ctx.framebuffers.current_value(per_image_ctx),
            full_output
        )?;
        self.render_stats.do_render.stop();
        self.render_stats.submit_command_buffers.start();
        self.submit_command_buffer(per_image_ctx, command_buffer, ready_future)?;
        self.render_stats.submit_command_buffers.stop();
        self.render_stats.end_step.start();
        Ok(())
    }

    fn synchronise(
        &mut self,
        per_image_ctx: &mut MutexGuard<PerImageContext>,
        acquire_future: SwapchainAcquireFuture,
    ) -> Result<SwapchainJoinFuture> {
        if let Some(uploads) = self.resource_handler.texture.wait_build_command_buffer(&self.vk_ctx)? {
            uploads.flush()?;
            info!("loaded textures");
        }
        if let Some(fence) = self.fences.last_value(per_image_ctx).borrow().as_ref() {
            if let Err(e) = fence.wait(None).map_err(Validated::unwrap) {
                // try to continue -- it might be an outdated future
                // XXX: macOS often just segfaults instead of giving an error here
                error!("{}", e);
            }
        }
        let last_fence = if let Some(fence) = self.fences.current_value(per_image_ctx).take() {
            fence.boxed()
        } else {
            let mut now = vulkano::sync::now(self.vk_ctx.device());
            now.cleanup_finished();
            now.boxed()
        };
        Ok(last_fence.join(acquire_future))
    }

    fn submit_command_buffer(
        &mut self,
        per_image_ctx: &mut MutexGuard<PerImageContext>,
        command_buffer: Arc<PrimaryAutoCommandBuffer>,
        ready_future: SwapchainJoinFuture,
    ) -> Result<()> {
        let image_idx = per_image_ctx.current.expect("no current image?");
        self.fences
            .current_value_mut(per_image_ctx)
            .borrow_mut()
            .replace(
                ready_future
                    .then_execute(
                        self.vk_ctx.queue(),
                        command_buffer
                    )?
                    .then_swapchain_present(
                        self.vk_ctx.queue(),
                        SwapchainPresentInfo::swapchain_image_index(
                            self.vk_ctx.swapchain(),
                            u32::try_from(image_idx)
                                .context("image_idx overflowed: {image_idx}")?
                        ),
                    )
                    .then_signal_fence(),
            );
        Ok(())
    }

    fn poll_ready(&mut self) -> bool {
        if !self.is_ready && self.last_ready_poll.elapsed().as_millis() >= 10 {
            self.is_ready = self.render_handler.get_receiver().lock().unwrap().is_ready();
            self.last_ready_poll = Instant::now();
        }
        self.is_ready
    }

    fn run_inner(&mut self, platform: &mut egui_winit::State, event: &Event<()>) -> Result<()> {
        let _response = match event {
            Event::WindowEvent { event, .. } => platform.on_window_event(&self.window, event),
            _ => EventResponse { consumed: false, repaint: false }
        };

        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                std::process::exit(0);
            }
            Event::WindowEvent { event: WindowEvent::KeyboardInput {
                event, ..
            }, .. } => {
                match event.physical_key {
                    PhysicalKey::Code(keycode) => {
                        self.input_handler.lock().unwrap().queue_key_event(keycode, event.state);
                    }
                    PhysicalKey::Unidentified(_) => {}
                }
                Ok(())
            }
            Event::WindowEvent { event: WindowEvent::ScaleFactorChanged {
                scale_factor, ..
            }, .. } => {
                self.scale_factor = *scale_factor;
                self.recreate_swapchain()
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                self.recreate_swapchain()
            }
            Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
                while !self.poll_ready() {}
                let per_image_ctx = self.vk_ctx.per_image_ctx.clone();
                let mut per_image_ctx = per_image_ctx.lock().unwrap();
                // XXX: "acquire_next_image" is somewhat misleading, since it does not block
                let acquired = swapchain::acquire_next_image(self.vk_ctx.swapchain(), None)
                    .map_err(Validated::unwrap);
                let rv = match acquired {
                    Ok((image_idx, /* suboptimal= */ false, acquire_future)) => {
                        egui_winit::update_viewport_info(
                            &mut ViewportInfo::default(),
                            &self.gui_ctx,
                            &self.window,
                            false
                        );
                        let raw_input = platform.take_egui_input(&self.window);
                        let full_output = self.gui_ctx.run(raw_input, |ctx| {
                            let mut input = self.input_handler.lock().unwrap();
                            input.set_viewport(self.render_handler.viewport());
                            input.update_mouse(ctx);
                            self.render_handler.do_gui(ctx, self.last_render_stats.clone());
                        });
                        self.render_stats.between_renders.stop();
                        platform.handle_platform_output(&self.window, full_output.platform_output.clone());
                        per_image_ctx.current.replace(image_idx as usize);
                        self.idle(&mut per_image_ctx, acquire_future, full_output)?;
                        let image_idx = per_image_ctx.current.expect("no current image?");
                        per_image_ctx.last = image_idx;
                        Ok(())
                    },
                    Ok((_, /* suboptimal= */ true, _)) | Err(VulkanError::OutOfDate) => {
                        self.recreate_swapchain()?;
                        self.render_stats.between_renders.stop();
                        Ok(())
                    },
                    Err(e) => Err(e.into()),
                };
                self.last_render_stats = self.render_stats.end();
                self.window.request_redraw();
                rv
            }
            _ => Ok(()),
        }
    }
}

#[derive(Clone)]
pub(crate) struct RenderPerfStats {
    handle_swapchain: TimeIt,
    synchronise: TimeIt,
    do_render: TimeIt,
    submit_command_buffers: TimeIt,
    end_step: TimeIt,
    between_renders: TimeIt,

    total: TimeIt,
    on_time: u64,
    count: u64,
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
            end_step: TimeIt::new("end step"),
            between_renders: TimeIt::new("between renders"),
            total: TimeIt::new("total"),
            on_time: 0,
            count: 0,
            last_step: Instant::now(),
            last_report: Instant::now(),
            totals_ms: Vec::with_capacity(10),
            last_perf_stats: None,
        }
    }

    fn end(&mut self) -> Option<Self> {
        const DEADLINE_MS: f64 = 16.8;

        self.end_step.stop();
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
                end_step: self.end_step.report_take(),
                between_renders: self.between_renders.report_take(),
                total: self.total.report_take(),
                on_time: 0,
                count: 0,
                last_perf_stats: None,
                last_report: Instant::now(),
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
        vec![
            self.total.as_tuple_ms(),
            self.handle_swapchain.as_tuple_ms(),
            self.synchronise.as_tuple_ms(),
            self.do_render.as_tuple_ms(),
            self.submit_command_buffers.as_tuple_ms(),
            self.end_step.as_tuple_ms(),
            self.between_renders.as_tuple_ms(),
        ]
    }
}
