// VulkanoContext is in a separate module because the details of handling Swapchain, and objects
// derived from it, are rather complicated. Don't make stuff here public unless really necessary.
use std::env;
use std::sync::{Arc, MutexGuard};
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
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass};
use vulkano::swapchain::{ColorSpace, Surface, SurfaceInfo, Swapchain, SwapchainAcquireFuture, SwapchainCreateInfo, SwapchainPresentInfo};
use vulkano::{swapchain, Validated, VulkanError, VulkanLibrary};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use crate::check_eq;
use crate::core::vk::{GgWindow, PerImageContext};
use crate::core::prelude::*;
use crate::util::{gg_float, UniqueShared};

#[derive(Clone)]
pub struct VulkanoContext {
    // Should only ever be created once:
    device: Arc<Device>,
    queue: Arc<Queue>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    pub(crate) per_image_ctx: UniqueShared<PerImageContext>,

    // May be recreated, e.g. due to window resizing:
    swapchain: UniqueShared<Arc<Swapchain>>,
    images: UniqueShared<DataPerImage<Arc<Image>>>,
    render_pass: UniqueShared<Arc<RenderPass>>,
    framebuffers: UniqueShared<DataPerImage<Arc<Framebuffer>>>,
    image_count: UniqueShared<usize>,
    pipelines: UniqueShared<Vec<UniqueShared<Option<Arc<GraphicsPipeline>>>>>,
}

impl VulkanoContext {
    pub(crate) fn new(event_loop: &ActiveEventLoop, window: &GgWindow) -> Result<Self> {
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

    pub(crate) fn recreate_swapchain(&mut self, window: &GgWindow) -> Result<()> {
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
    pub(crate) fn acquire_next_image(&self) -> Result<AcquiredSwapchainFuture, VulkanError> {
        // XXX: "acquire_next_image" is somewhat misleading, since it does not block
        swapchain::acquire_next_image(self.swapchain.get().clone(), None)
            .map_err(Validated::unwrap)
    }
    pub(crate) fn swapchain_present_info(&self, image_idx: usize) -> Result<SwapchainPresentInfo> {
        Ok(SwapchainPresentInfo::swapchain_image_index(
            self.swapchain.get().clone(),
            u32::try_from(image_idx)
                .context("image_idx overflowed: {image_idx}")?
        ))
    }
    pub(crate) fn current_framebuffer(&self, per_image_ctx: &MutexGuard<PerImageContext>) -> Arc<Framebuffer> {
        self.framebuffers.get().current_value(per_image_ctx).clone()
    }

    // When the created pipeline is invalidated, it will be destroyed => safe to store this.
    pub fn create_pipeline<F>(&mut self, f: F) -> Result<UniqueShared<Option<Arc<GraphicsPipeline>>>>
    where F: FnOnce(Arc<RenderPass>) -> Result<Arc<GraphicsPipeline>>
    {
        let pipeline = UniqueShared::new(Some(f(self.render_pass.get().clone())?));
        self.pipelines.get().push(pipeline.clone());
        Ok(pipeline)
    }
    // May change between frames, e.g. due to recreate_swapchain().
    pub fn image_count(&self) -> usize { *self.image_count.get() }
}

// TODO: more flexible approach here.
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
pub(crate) struct DataPerImage<T: Clone> {
    data: Vec<T>,
}

impl <T: Clone + Copy> DataPerImage<T> {
    // Not thoroughly tested:
    // pub fn new_with_value(ctx: &VulkanoContext, initial_value: T) -> Self {
    //     let data = vec![initial_value; ctx.images.get().len()];
    //     Self { data }
    // }
    //
    // pub fn clone_from_value(&mut self, new_value: T) {
    //     self.data = vec![new_value; self.data.len()];
    // }
}

impl<T: Clone> DataPerImage<T> {
    // Not thoroughly tested:
    // pub fn new_with_data(ctx: &VulkanoContext, data: Vec<T>) -> Self {
    //     check_eq!(data.len(), ctx.images.get().len());
    //     Self { data }
    // }
    // pub fn try_new_with_generator<F: Fn() -> Result<T>>(
    //     ctx: &VulkanoContext,
    //     generator: F,
    // ) -> Result<Self> {
    //     let mut data = Vec::new();
    //     for _ in 0..ctx.images.get().len() {
    //         data.push(generator()?);
    //     }
    //     Ok(Self { data })
    // }
    pub fn new_with_generator<F: Fn() -> T>(ctx: &VulkanoContext, generator: F) -> Self {
        let mut data = Vec::new();
        for _ in 0..ctx.images.get().len() {
            data.push(generator());
        }
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn last_value(&self, per_image_ctx: &MutexGuard<PerImageContext>) -> &T {
        &self.data[per_image_ctx.last]
    }
    pub fn current_value(&self, per_image_ctx: &MutexGuard<PerImageContext>) -> &T {
        &self.data[per_image_ctx.current.expect("no current value?")]
    }
    // pub fn last_value_mut(&mut self, per_image_ctx: &mut MutexGuard<PerImageContext>) -> &mut T {
    //     &mut self.data[per_image_ctx.last]
    // }
    pub fn current_value_mut(&mut self, per_image_ctx: &mut MutexGuard<PerImageContext>) -> &mut T {
        &mut self.data[per_image_ctx.current.expect("no current value?")]
    }

    // Not thoroughly tested:
    // pub fn map<U: Clone, F>(&self, func: F) -> DataPerImage<U>
    // where
    //     F: FnMut(&T) -> U,
    // {
    //     DataPerImage::<U> {
    //         data: self.as_slice().iter().map(func).collect(),
    //     }
    // }
    // pub fn count<P>(&self, predicate: P) -> usize
    // where
    //     P: Fn(&T) -> bool,
    // {
    //     let mut rv = 0;
    //     for value in self.as_slice() {
    //         rv += usize::from(predicate(value));
    //     }
    //     rv
    // }
    // pub fn try_count<P>(&self, predicate: P) -> Result<usize>
    // where
    //     P: Fn(&T) -> Result<bool>,
    // {
    //     let mut rv = 0;
    //     for value in self.as_slice() {
    //         rv += usize::from(predicate(value)?);
    //     }
    //     Ok(rv)
    // }
    fn as_slice(&self) -> &[T] {
        self.data.as_slice()
    }
}
