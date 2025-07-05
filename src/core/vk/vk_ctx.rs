// VulkanoContext is in a separate module because the details of handling Swapchain, and objects
// derived from it, are rather complicated. Don't make stuff here public unless really necessary.
use crate::core::prelude::*;
use crate::core::vk::GgWindow;
use crate::util::{UniqueShared, gg_float};
use crate::{check_eq, info_every_millis};
use anyhow::{Context, Result};
use egui_winit::winit::event_loop::ActiveEventLoop;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Instant, SystemTime};
use tracing::{info, info_span};
use vulkano::descriptor_set::allocator::{
    StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo,
};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{
    Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo, QueueFlags,
};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageUsage};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo, InstanceExtensions};
use vulkano::swapchain::{
    ColorSpace, PresentMode, Surface, SurfaceInfo, Swapchain, SwapchainCreateInfo,
};
use vulkano::{Validated, Version, VulkanLibrary};
use vulkano_taskgraph::Id;
use vulkano_taskgraph::resource::{Flight, Resources, ResourcesCreateInfo};

static VULKANO_CONTEXT_CREATED: LazyLock<Mutex<bool>> = LazyLock::new(|| Mutex::new(false));

#[derive(Clone)]
pub(crate) struct VulkanoPerfStats {
    active: bool,
    stats: Arc<Mutex<Vec<(String, SystemTime)>>>,
}

impl VulkanoPerfStats {
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
            let span = info_span!("VulkanoPerfStats");
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

/// Provides access to the Vulkan context and resources.
/// This struct is public to allow custom shader implementations, though this functionality
/// is currently work-in-progress and incomplete.
#[derive(Clone)]
pub struct VulkanoContext {
    // Should only ever be created once:
    device: Arc<Device>,
    queue: Arc<Queue>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    resources: Arc<Resources>,
    flight_id: Id<Flight>,

    // May be recreated, e.g. due to window resizing:
    swapchain: UniqueShared<Id<Swapchain>>,
    images: UniqueShared<Vec<Arc<Image>>>,
    image_views: UniqueShared<Vec<Arc<ImageView>>>,
    image_count: UniqueShared<usize>,

    stats: VulkanoPerfStats,
}

impl VulkanoContext {
    // There's just a ton of stuff to do here, so disable the lint.
    #[allow(clippy::too_many_lines)]
    pub(crate) fn new(event_loop: &ActiveEventLoop, window: &GgWindow) -> Result<Self> {
        {
            let mut vulkano_context_created = VULKANO_CONTEXT_CREATED.lock().unwrap();
            check_false!(*vulkano_context_created);
            *vulkano_context_created = true;
        }
        check_eq!(size_of::<u64>(), size_of::<usize>()); // assumption made in many places
        info!("operating system: {}", std::env::consts::OS);
        let start = Instant::now();
        let library = VulkanLibrary::new().context("vulkano: no local Vulkan library/DLL")?;
        let instance = create_instance(event_loop, library)?;

        let surface = Surface::from_window(instance.clone(), window.inner.clone())?;
        let physical_device = create_any_physical_device(&instance, &surface)?;
        let (device, queue) = create_any_graphical_queue_family(physical_device.clone())?;
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            StandardDescriptorSetAllocatorCreateInfo {
                update_after_bind: true,
                ..Default::default()
            },
        ));
        let resources = Resources::new(&device, &ResourcesCreateInfo::default());

        let has_mailbox = physical_device
            .surface_capabilities(
                &surface,
                SurfaceInfo {
                    present_mode: Some(PresentMode::Mailbox),
                    ..SurfaceInfo::default()
                },
            )
            .is_ok();
        let caps = physical_device.surface_capabilities(&surface, SurfaceInfo::default())?;
        let supported_formats =
            physical_device.surface_formats(&surface, SurfaceInfo::default())?;
        if !supported_formats.contains(&(Format::B8G8R8A8_SRGB, ColorSpace::SrgbNonLinear)) {
            error!(
                "supported formats missing (Format::B8G8R8A8_SRGB, ColorSpace::SrgbNonLinear):\n{:?}",
                supported_formats
            );
        }
        info!("surface capabilities: {caps:?}");
        let (min_image_count, present_mode) = if has_mailbox {
            (
                caps.max_image_count
                    .unwrap_or(3.max(caps.min_image_count + 1)),
                if USE_VSYNC {
                    PresentMode::Mailbox
                } else {
                    PresentMode::Immediate
                },
            )
        } else {
            (
                3.max(caps.min_image_count),
                if USE_VSYNC {
                    PresentMode::Fifo
                } else {
                    PresentMode::Immediate
                },
            )
        };
        if let Some(max_image_count) = caps.max_image_count {
            check_le!(min_image_count, max_image_count);
        }
        info!(
            "swapchain properties: min_image_count={min_image_count}, present_mode={present_mode:?}"
        );

        // Using more than 2 frames in flight just adds latency for no gain.
        let flight_id = resources.create_flight(2)?;

        let dimensions = window.inner_size();
        let composite_alpha = caps
            .supported_composite_alpha
            .into_iter()
            .next()
            .context("vulkano: no composite alpha modes supported")?;
        let swapchain = resources.create_swapchain(
            flight_id,
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
        )?;
        let images = resources
            .swapchain(swapchain)?
            .images()
            .iter()
            .cloned()
            .collect_vec();
        let image_views = UniqueShared::new(
            images
                .iter()
                .map(|image| ImageView::new_default(image.clone()))
                .collect::<Result<Vec<_>, _>>()
                .map_err(Validated::unwrap)?,
        );
        let image_count = UniqueShared::new(images.len());
        let swapchain = UniqueShared::new(swapchain);
        let images = UniqueShared::new(images);

        info!(
            "created vulkano context in: {:.2} ms",
            gg_float::micros(start.elapsed()) * 1000.0
        );
        Ok(Self {
            device,
            queue,
            descriptor_set_allocator,
            resources,
            flight_id,

            swapchain,
            images,
            image_views,
            image_count,
            stats: VulkanoPerfStats::new(),
        })
    }

    pub(crate) fn recreate_swapchain(&mut self, window: &GgWindow) -> Result<()> {
        let new_swapchain =
            self.resources
                .recreate_swapchain(*self.swapchain.get(), |create_info| SwapchainCreateInfo {
                    image_extent: window.inner_size().into(),
                    ..create_info
                })?;
        *self.swapchain.get() = new_swapchain;
        *self.images.get() = self
            .resources
            .swapchain(new_swapchain)?
            .images()
            .iter()
            .cloned()
            .collect_vec();
        *self.image_views.get() = self
            .images
            .get()
            .iter()
            .map(|image| ImageView::new_default(image.clone()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(Validated::unwrap)?;
        Ok(())
    }

    // The below should never be re-created, so it's safe to store them.
    pub fn device(&self) -> Arc<Device> {
        self.device.clone()
    }
    pub fn queue(&self) -> Arc<Queue> {
        self.queue.clone()
    }
    pub fn descriptor_set_allocator(&self) -> Arc<StandardDescriptorSetAllocator> {
        self.descriptor_set_allocator.clone()
    }
    pub fn resources(&self) -> Arc<Resources> {
        self.resources.clone()
    }
    pub fn flight_id(&self) -> Id<Flight> {
        self.flight_id
    }

    pub(crate) fn swapchain(&self) -> Result<Arc<Swapchain>> {
        Ok(self
            .resources
            .swapchain(*self.swapchain.get())?
            .swapchain()
            .clone())
    }
    pub(crate) fn swapchain_id(&self) -> Id<Swapchain> {
        *self.swapchain.get()
    }

    pub(crate) fn current_image_view(&self, image_idx: usize) -> Arc<ImageView> {
        self.image_views.get()[image_idx].clone()
    }
    // May change between frames, e.g. due to recreate_swapchain().
    pub fn image_count(&self) -> usize {
        *self.image_count.get()
    }

    pub(crate) fn perf_stats(&self) -> &VulkanoPerfStats {
        &self.stats
    }
}

// TODO: more flexible approach here.
fn instance_extensions(event_loop: &ActiveEventLoop) -> Result<InstanceExtensions> {
    let mut extensions = Surface::required_extensions(&event_loop)?;
    extensions.ext_debug_utils = true;
    Ok(extensions)
}
fn device_extensions() -> DeviceExtensions {
    DeviceExtensions {
        khr_swapchain: true,
        khr_dynamic_rendering: true,
        ..DeviceExtensions::empty()
    }
}
fn device_features() -> DeviceFeatures {
    DeviceFeatures {
        // See comment in `create_instance()`.
        // descriptor_indexing: true,
        fill_mode_non_solid: true,
        // Required to fix sampler cross-talk with AMD drivers:
        shader_sampled_image_array_non_uniform_indexing: true,
        dynamic_rendering: true,
        ..Default::default()
    }
}

fn create_instance(
    event_loop: &ActiveEventLoop,
    library: Arc<VulkanLibrary>,
) -> Result<Arc<Instance>> {
    // TODO: this was required at one point. Should not be required due to a MoltenVK update,
    //  however it may be needed at some point for older versions of macOS.
    // if std::env::consts::OS == "macos" {
    //     let var = match std::env::var("MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS") {
    //         Ok(var) => var,
    //         Err(e) => {
    //             panic!(
    //                 "on macOS, environment variable `MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS` must be set; \
    //                     do you have .cargo/config.toml set up correctly? got: {e:?}"
    //             );
    //         }
    //     };
    //     check_eq!(var, "1");
    // }
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
        .filter(|p| p.supported_features().contains(&device_features()))
        .filter(|p| {
            p.api_version() >= Version::V1_3 || p.supported_extensions().khr_dynamic_rendering
        })
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    let i = i as u32;
                    q.queue_flags.contains(QueueFlags::GRAPHICS)
                        && p.surface_support(i, surface).unwrap_or(false)
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
        .0;
    info!(
        "device: {} [{:?}]",
        device.properties().device_name,
        device.properties().device_type
    );
    info!("vulkan version: {}", device.api_version());
    Ok(device)
}
fn create_any_graphical_queue_family(
    physical_device: Arc<PhysicalDevice>,
) -> Result<(Arc<Device>, Arc<Queue>)> {
    let queue_family_index = physical_device
        .queue_family_properties()
        .iter()
        .position(|queue_family_properties| {
            queue_family_properties
                .queue_flags
                .contains(QueueFlags::GRAPHICS)
        })
        .context("vulkano: couldn't find a graphical queue family")?;
    info!(
        "queue family properties: {:?}",
        physical_device.queue_family_properties()[queue_family_index]
    );

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index: queue_family_index as u32,
                ..Default::default()
            }],
            enabled_extensions: device_extensions(),
            enabled_features: device_features(),
            ..Default::default()
        },
    )?;
    info!("found {} queue(s), using first", queues.len());
    Ok((
        device,
        queues.next().context("vulkano: UNEXPECTED: zero queues?")?,
    ))
}
