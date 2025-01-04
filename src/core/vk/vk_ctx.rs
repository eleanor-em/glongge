// VulkanoContext is in a separate module because the details of handling Swapchain, and objects
// derived from it, are rather complicated. Don't make stuff here public unless really necessary.
use std::env;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::Instant;
use anyhow::{Context, Result};
use egui_winit::winit::event_loop::ActiveEventLoop;
use tracing::info;
use vulkano::command_buffer::allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo};
use vulkano::descriptor_set::allocator::{StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo};
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo, QueueFlags};
use vulkano::image::{Image, ImageUsage};
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::swapchain::{ColorSpace, PresentMode, Surface, SurfaceInfo, Swapchain, SwapchainAcquireFuture, SwapchainCreateInfo, SwapchainPresentInfo};
use vulkano::{swapchain, Validated, Version, VulkanLibrary};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo, InstanceExtensions};
use crate::check_eq;
use crate::core::vk::GgWindow;
use crate::core::prelude::*;
use crate::util::{gg_err, gg_float, UniqueShared};

static VULKANO_CONTEXT_CREATED: LazyLock<Mutex<bool>> = LazyLock::new(|| Mutex::new(false));

#[derive(Clone)]
pub struct VulkanoContext {
    // Should only ever be created once:
    device: Arc<Device>,
    queue: Arc<Queue>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,

    // May be recreated, e.g. due to window resizing:
    swapchain: UniqueShared<Arc<Swapchain>>,
    images: UniqueShared<Vec<Arc<Image>>>,
    image_views: UniqueShared<Vec<Arc<ImageView>>>,
    image_count: UniqueShared<usize>,
    pipelines: UniqueShared<Vec<UniqueShared<Option<Arc<GraphicsPipeline>>>>>,
}

impl VulkanoContext {
    pub(crate) fn new(event_loop: &ActiveEventLoop, window: &GgWindow) -> Result<Self> {
        {
            let mut vulkano_context_created = VULKANO_CONTEXT_CREATED.lock().unwrap();
            check_false!(*vulkano_context_created);
            *vulkano_context_created = true;
        }
        info!("operating system: {}", std::env::consts::OS);
        let start = Instant::now();
        let library = VulkanLibrary::new().context("vulkano: no local Vulkan library/DLL")?;
        let instance = create_instance(event_loop, library)?;
        let surface = Surface::from_window(instance.clone(), window.inner.clone())?;
        let physical_device = create_any_physical_device(&instance, &surface)?;
        let (device, queue) = create_any_graphical_queue_family(physical_device.clone())?;
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
        let images = UniqueShared::new(images);
        let image_views = UniqueShared::new(create_image_views(images.get().as_slice())?);
        let image_count = UniqueShared::new(image_views.get().len());

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
            images,
            image_views,
            image_count,
            pipelines: UniqueShared::new(Vec::new()),
        })
    }

    pub(crate) fn recreate_swapchain(&mut self, window: &GgWindow) -> Result<()> {
        let swapchain_create_info = SwapchainCreateInfo {
            image_extent: window.inner_size().into(),
            ..self.swapchain.get().create_info()
        };
        let (new_swapchain, new_images) = self.swapchain.get()
            .recreate(swapchain_create_info).map_err(Validated::unwrap)?;
        *self.swapchain.get() = new_swapchain;
        *self.images.get() = new_images;
        *self.image_views.get() = create_image_views(self.images.get().as_slice())?;
        *self.image_count.get() = self.image_views.get().len();
        for pipeline in self.pipelines.get().drain(..) {
            *pipeline.get() = None;
        }
        Ok(())
    }

    // The below should never be re-created, so it's safe to store them.
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

    // The return values of the below pub(crate) functions should not be stored between frames.
    pub(crate) fn acquire_next_image(&self) -> Result<AcquiredSwapchainFuture, gg_err::CatchOutOfDate> {
        // XXX: "acquire_next_image" is somewhat misleading, since it does not block
        swapchain::acquire_next_image(self.swapchain.get().clone(), None)
            .map_err(Validated::unwrap)
            .map_err(gg_err::CatchOutOfDate::from)
    }
    pub(crate) fn swapchain_present_info(&self, image_idx: usize) -> Result<SwapchainPresentInfo> {
        Ok(SwapchainPresentInfo::swapchain_image_index(
            self.swapchain.get().clone(),
            u32::try_from(image_idx)
                .context("image_idx overflowed: {image_idx}")?
        ))
    }
    pub(crate) fn current_image_view(&self, image_idx: usize) -> Arc<ImageView> {
        self.image_views.get()[image_idx].clone()
    }

    // When the created pipeline is invalidated, it will be destroyed => safe to store this.
    pub fn create_pipeline<F>(&mut self, f: F) -> Result<UniqueShared<Option<Arc<GraphicsPipeline>>>>
    where F: FnOnce(Arc<Swapchain>) -> Result<Arc<GraphicsPipeline>>
    {
        let pipeline = UniqueShared::new(Some(f(self.swapchain.get().clone())?));
        self.pipelines.get().push(pipeline.clone());
        Ok(pipeline)
    }
    // May change between frames, e.g. due to recreate_swapchain().
    pub fn image_count(&self) -> usize { *self.image_count.get() }
}

// TODO: more flexible approach here.
fn instance_extensions(event_loop: &ActiveEventLoop) -> Result<InstanceExtensions> {
    let mut extensions = Surface::required_extensions(&event_loop)?;
    extensions.ext_surface_maintenance1 = true;
    Ok(extensions)
}
fn device_extensions() -> DeviceExtensions {
    DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::empty()
    }
}
fn features() -> DeviceFeatures {
    DeviceFeatures {
        // Required for extra texture samplers on macOS:
        descriptor_indexing: true,
        fill_mode_non_solid: true,
        ..Default::default()
    }
}

pub(crate) type AcquiredSwapchainFuture = (u32, bool, SwapchainAcquireFuture);

fn create_instance(
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
    let enabled_extensions = instance_extensions(event_loop)?;
    let instance_create_info = InstanceCreateInfo {
        flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
        enabled_extensions,
        ..Default::default()
    };
    Instance::new(library, instance_create_info).context("vulkano: failed to create instance")
}
fn create_any_physical_device(
    instance: &Arc<Instance>,
    surface: &Arc<Surface>,
) -> Result<Arc<PhysicalDevice>> {
    let device = instance
        .enumerate_physical_devices()?
        .filter(|p| p.supported_extensions().contains(&device_extensions()))
        .filter(|p| p.supported_features().contains(&features()))
        // Dynamic rendering support:
        .filter(|p| p.api_version() >= Version::V1_3 || p.supported_extensions().khr_dynamic_rendering)
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
        .0;
    info!("device: {} [{:?}]", device.properties().device_name, device.properties().device_type);
    info!("vulkan version: {}", device.api_version());
    Ok(device)
}
fn create_any_graphical_queue_family(
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

    let mut enabled_extensions = device_extensions();
    let mut enabled_features = features();
    // Also enable dynamic rendering, guaranteed to be available by filtering:
    enabled_extensions.khr_dynamic_rendering = true;
    enabled_features.dynamic_rendering = true;
    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index: queue_family_index as u32,
                ..Default::default()
            }],
            enabled_extensions,
            enabled_features,
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
    let has_mailbox = physical_device.surface_capabilities(&surface, SurfaceInfo {
        present_mode: Some(PresentMode::Mailbox),
        ..SurfaceInfo::default()
    }).is_ok();
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
    let (min_image_count, present_mode) = if has_mailbox {
        (caps.max_image_count.unwrap_or(3.max(caps.min_image_count + 1)),
         PresentMode::Mailbox)
    } else {
        (3.max(caps.min_image_count),
         PresentMode::Fifo)
    };
    if let Some(max_image_count) = caps.max_image_count {
        check_le!(min_image_count, max_image_count);
    }
    info!("swapchain properties: min_image_count={min_image_count}, present_mode={present_mode:?}");
    Ok(Swapchain::new(
        device,
        surface,
        SwapchainCreateInfo {
            min_image_count,
            image_format: Format::B8G8R8A8_SRGB,
            image_color_space: ColorSpace::SrgbNonLinear,
            image_extent: dimensions.into(),
            image_usage: ImageUsage::COLOR_ATTACHMENT,
            composite_alpha,
            present_mode,
            ..Default::default()
        },
    )?)
}

fn create_image_views(images: &[Arc<Image>]) -> Result<Vec<Arc<ImageView>>> {
    Ok(images.iter()
            .map(|image| ImageView::new_default(image.clone()))
            .collect::<Result<Vec<_>, _>>().map_err(Validated::unwrap)?)
}
