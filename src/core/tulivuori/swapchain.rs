use crate::core::tulivuori::TvWindowContext;
use crate::util::colour::Colour;
use anyhow::{Context, Result, bail};
use ash::khr::swapchain;
use ash::vk;
use egui_winit::winit::window::Window;
use itertools::Itertools;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

pub struct SwapchainBuilder<'a> {
    ctx: Arc<TvWindowContext>,
    window: &'a Arc<Window>,

    // Settings:
    filter_surface_format: Option<fn(&'_ vk::SurfaceFormatKHR) -> bool>,
    filter_present_mode: Option<fn(&'_ vk::PresentModeKHR) -> bool>,
    get_desired_image_count: Option<fn(vk::SurfaceCapabilitiesKHR) -> u32>,
    get_desired_frames_in_flight: Option<fn(vk::SurfaceCapabilitiesKHR) -> usize>,
    swapchain_create_info: Option<vk::SwapchainCreateInfoKHR<'a>>,
}

impl<'a> SwapchainBuilder<'a> {
    pub fn new(ctx: &Arc<TvWindowContext>, window: &'a Arc<Window>) -> SwapchainBuilder<'a> {
        Self {
            ctx: ctx.clone(),
            window,
            filter_surface_format: None,
            filter_present_mode: None,
            get_desired_image_count: None,
            get_desired_frames_in_flight: None,
            swapchain_create_info: None,
        }
    }

    fn desired_image_count(&self, surface_capabilities: vk::SurfaceCapabilitiesKHR) -> Result<u32> {
        let rv = if let Some(get_desired_image_count) = self.get_desired_image_count {
            get_desired_image_count(surface_capabilities)
        } else {
            // Default case
            surface_capabilities.min_image_count + 1
        };
        if rv > surface_capabilities.max_image_count {
            bail!(
                "calculated desired_image_count > max_image_count: {rv} vs. {}",
                surface_capabilities.max_image_count
            );
        }
        Ok(rv)
    }
    fn desired_frames_in_flight(
        &self,
        surface_capabilities: vk::SurfaceCapabilitiesKHR,
    ) -> Result<usize> {
        let rv = if let Some(get_desired_frames_in_flight) = self.get_desired_frames_in_flight {
            get_desired_frames_in_flight(surface_capabilities)
        } else {
            // Default case
            2
        };
        let desired_image_count = self.desired_image_count(surface_capabilities)?;
        if rv > desired_image_count as usize {
            bail!(
                "more frames in flight than images, this is almost certainly unintended: {rv} vs. {desired_image_count}",
            );
        }
        Ok(rv)
    }

    #[allow(clippy::too_many_lines)]
    pub fn build(mut self) -> Result<Arc<Swapchain>> {
        let surface_format = {
            let all_surface_formats = self.ctx.get_physical_device_surface_formats()?;
            if let Some(filter) = self.filter_surface_format {
                all_surface_formats.into_iter().find_or_first(filter)
            } else {
                all_surface_formats.first().copied()
            }
            .context("get_physical_device_surface_formats() returned empty")?
        };

        let surface_capabilities = self.ctx.get_physical_device_surface_capabilities()?;
        let desired_image_count = self.desired_image_count(surface_capabilities)?;
        let desired_frames_in_flight = self.desired_frames_in_flight(surface_capabilities)?;
        let surface_resolution = match surface_capabilities.current_extent.width {
            u32::MAX => vk::Extent2D {
                width: self.window.inner_size().width,
                height: self.window.inner_size().height,
            },
            _ => surface_capabilities.current_extent,
        };

        let pre_transform = if surface_capabilities
            .supported_transforms
            .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
        {
            vk::SurfaceTransformFlagsKHR::IDENTITY
        } else {
            surface_capabilities.current_transform
        };
        let present_mode = {
            let all_present_modes = self.ctx.get_physical_device_surface_present_modes()?;
            if let Some(filter) = self.filter_present_mode.take() {
                all_present_modes.into_iter().find_or_first(filter)
            } else {
                all_present_modes.first().copied()
            }
            .context("UNEXPECTED: get_physical_device_surface_present_modes() returned empty")?
        };

        let swapchain_loader = self.ctx.create_swapchain_device();
        let swapchain_khr = unsafe {
            swapchain_loader.create_swapchain(
                &self.swapchain_create_info.unwrap_or(
                    vk::SwapchainCreateInfoKHR::default()
                        .surface(self.ctx.surface())
                        .min_image_count(desired_image_count)
                        .image_color_space(surface_format.color_space)
                        .image_format(surface_format.format)
                        .image_extent(surface_resolution)
                        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .pre_transform(pre_transform)
                        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                        .present_mode(present_mode)
                        .clipped(true)
                        .image_array_layers(1),
                ),
                None,
            )?
        };

        let images =
            SwapchainImages::new(&self.ctx, swapchain_khr, &swapchain_loader, surface_format)?;

        let mut present_semaphores = Vec::new();
        let mut submit_semaphores = Vec::new();
        let mut acquire_fences = Vec::new();
        let mut present_fences = Vec::new();
        unsafe {
            for _ in 0..desired_image_count {
                present_semaphores.push(
                    self.ctx
                        .device()
                        .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?,
                );
                submit_semaphores.push(
                    self.ctx
                        .device()
                        .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?,
                );
            }
            for i in 0..desired_frames_in_flight {
                acquire_fences.insert(
                    0,
                    self.ctx
                        .device()
                        .create_fence(&vk::FenceCreateInfo::default(), None)?,
                );
                let present_fence_create_info = if i == 0 {
                    vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED)
                } else {
                    vk::FenceCreateInfo::default()
                };
                present_fences.insert(
                    0,
                    self.ctx
                        .device()
                        .create_fence(&present_fence_create_info, None)?,
                );
            }
        }

        let mut present_command_pools = Vec::new();
        let mut present_command_buffers = Vec::new();
        unsafe {
            for _ in 0..desired_frames_in_flight {
                let command_pool = self.ctx.device().create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                        .queue_family_index(self.ctx.queue_family_index()),
                    None,
                )?;
                present_command_pools.push(command_pool);
                present_command_buffers.extend(
                    self.ctx.device().allocate_command_buffers(
                        &vk::CommandBufferAllocateInfo::default()
                            .command_buffer_count(1)
                            .command_pool(command_pool)
                            .level(vk::CommandBufferLevel::PRIMARY),
                    )?,
                );
            }
        }

        let frame_index = FrameIndex::new(desired_frames_in_flight);
        let image_index = PresentIndex::new();
        let last_frame_index = AtomicUsize::new(frame_index.lock().unwrap().current);

        Ok(Arc::new(Swapchain {
            ctx: self.ctx,
            khr: swapchain_khr,
            loader: swapchain_loader,
            surface_format,
            surface_resolution,
            images,
            frames_in_flight: desired_frames_in_flight,
            present_semaphores,
            submit_semaphores,
            acquire_fences,
            present_fences,
            present_command_pools,
            present_command_buffers,
            frame_index,
            image_index,
            current_frame_index: last_frame_index,
        }))
    }
}

struct FrameIndex {
    last: usize,
    current: usize,
}

impl FrameIndex {
    fn new(frames_in_flight: usize) -> Mutex<Self> {
        Mutex::new(Self {
            last: 0,
            current: frames_in_flight - 1,
        })
    }
}
struct PresentIndex {
    last: usize,
    current: usize,
}

impl PresentIndex {
    fn new() -> Mutex<Self> {
        Mutex::new(Self {
            last: 0,
            current: 0,
        })
    }
}

struct SwapchainImages {
    ctx: Arc<TvWindowContext>,
    present_images: Vec<vk::Image>,
    present_image_views: Vec<vk::ImageView>,
}

impl SwapchainImages {
    fn new(
        ctx: &Arc<TvWindowContext>,
        swapchain_khr: vk::SwapchainKHR,
        swapchain_loader: &swapchain::Device,
        surface_format: vk::SurfaceFormatKHR,
    ) -> Result<Self> {
        unsafe {
            let present_images = swapchain_loader.get_swapchain_images(swapchain_khr)?;
            let present_image_views = present_images
                .iter()
                .map(|&image| {
                    let create_view_info = vk::ImageViewCreateInfo::default()
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(surface_format.format)
                        .components(vk::ComponentMapping {
                            r: vk::ComponentSwizzle::R,
                            g: vk::ComponentSwizzle::G,
                            b: vk::ComponentSwizzle::B,
                            a: vk::ComponentSwizzle::A,
                        })
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .image(image);
                    let image_view = ctx.device().create_image_view(&create_view_info, None)?;
                    Ok(image_view)
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(Self {
                ctx: ctx.clone(),
                present_images,
                present_image_views,
            })
        }
    }
}

impl Drop for SwapchainImages {
    fn drop(&mut self) {
        unsafe {
            self.ctx.device().device_wait_idle().unwrap();
            for image_view in self.present_image_views.drain(..) {
                self.ctx.device().destroy_image_view(image_view, None);
            }
        }
    }
}

pub struct SwapchainAcquireInfo {
    acquired_image_index: usize,
    acquired_frame_index: usize,
    frames_in_flight: usize,
    is_suboptimal: bool,
}

impl SwapchainAcquireInfo {
    pub fn acquired_image_index(&self) -> usize {
        self.acquired_image_index
    }
    pub fn acquired_frame_index(&self) -> usize {
        self.acquired_frame_index
    }
    pub fn frames_in_flight(&self) -> usize {
        self.frames_in_flight
    }
    pub fn is_suboptimal(&self) -> bool {
        self.is_suboptimal
    }
}

pub struct Swapchain {
    ctx: Arc<TvWindowContext>,
    khr: vk::SwapchainKHR,
    loader: swapchain::Device,

    surface_format: vk::SurfaceFormatKHR,
    surface_resolution: vk::Extent2D,
    images: SwapchainImages,

    frames_in_flight: usize,
    present_semaphores: Vec<vk::Semaphore>,
    submit_semaphores: Vec<vk::Semaphore>,
    acquire_fences: Vec<vk::Fence>,
    present_fences: Vec<vk::Fence>,

    present_command_pools: Vec<vk::CommandPool>,
    present_command_buffers: Vec<vk::CommandBuffer>,

    frame_index: Mutex<FrameIndex>,
    image_index: Mutex<PresentIndex>,
    current_frame_index: AtomicUsize,
}

impl Swapchain {
    pub fn surface_resolution(&self) -> vk::Extent2D {
        self.surface_resolution
    }
    pub fn frames_in_flight(&self) -> usize {
        self.frames_in_flight
    }
    pub fn current_frame_index(&self) -> usize {
        self.current_frame_index.load(Ordering::Relaxed)
    }
    pub fn surface_format(&self) -> vk::SurfaceFormatKHR {
        self.surface_format
    }

    pub fn acquire_next_image(&self, extra_fences: &[vk::Fence]) -> Result<SwapchainAcquireInfo> {
        self.acquire_next_image_timeout(extra_fences, u64::MAX, u64::MAX)?
            .context("timeout even though we waited forever")
    }
    pub fn acquire_next_image_timeout(
        &self,
        extra_fences: &[vk::Fence],
        wait_timeout: u64,
        acquire_timeout: u64,
    ) -> Result<Option<SwapchainAcquireInfo>> {
        {
            // Update {frame,present}_index.
            let mut frame_index = self.frame_index.lock().unwrap();
            frame_index.last = frame_index.current;
            frame_index.current = (frame_index.last + 1) % self.frames_in_flight;
            self.current_frame_index
                .store(frame_index.current, Ordering::Relaxed);

            let mut present_index = self.image_index.lock().unwrap();
            present_index.last = present_index.current;
        }

        match unsafe {
            let mut fences = vec![self.last_present_fence()];
            fences.extend_from_slice(extra_fences);
            self.ctx
                .device()
                .wait_for_fences(&fences, true, wait_timeout)?;
            self.ctx.device().reset_fences(&fences)?;
            let rv = self.loader.acquire_next_image(
                self.khr,
                acquire_timeout,
                self.present_semaphore(),
                self.last_acquire_fence(),
            )?;
            self.ctx
                .device()
                .wait_for_fences(&[self.last_acquire_fence()], true, wait_timeout)?;
            self.ctx
                .device()
                .reset_fences(&[self.last_acquire_fence()])?;
            Ok(rv)
        } {
            Ok((next_image_index, is_suboptimal)) => {
                let next_image_index = next_image_index as usize;
                self.image_index.lock().unwrap().current = next_image_index;
                Ok(Some(SwapchainAcquireInfo {
                    acquired_image_index: next_image_index,
                    acquired_frame_index: self.current_frame_index.load(Ordering::Relaxed),
                    frames_in_flight: self.frames_in_flight,
                    is_suboptimal,
                }))
            }
            Err(vk::Result::TIMEOUT) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    pub fn acquire_present_command_buffer(&self) -> Result<vk::CommandBuffer> {
        let current_frame_index = self.current_frame_index.load(Ordering::Relaxed);
        unsafe {
            self.ctx.device().reset_command_pool(
                self.present_command_pools[current_frame_index],
                vk::CommandPoolResetFlags::RELEASE_RESOURCES,
            )?;
        }
        Ok(self.present_command_buffers[current_frame_index])
    }

    pub fn cmd_begin_rendering(&self, command_buffer: vk::CommandBuffer, clear_col: Colour) {
        unsafe {
            self.ctx.device().cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .image(self.current_present_image())
                    .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1)
                            .level_count(1),
                    )],
            );
            self.ctx.device().cmd_begin_rendering(
                command_buffer,
                &vk::RenderingInfo::default()
                    .color_attachments(&[vk::RenderingAttachmentInfo::default()
                        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .clear_value(vk::ClearValue {
                            color: vk::ClearColorValue {
                                float32: clear_col.into(),
                            },
                        })
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .image_view(self.current_present_image_view())])
                    .render_area(self.surface_resolution().into())
                    .layer_count(1),
            );
        }
    }
    pub fn cmd_end_rendering(&self, command_buffer: vk::CommandBuffer) {
        unsafe {
            self.ctx.device().cmd_end_rendering(command_buffer);
            self.ctx.device().cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .image(self.current_present_image())
                    .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1)
                            .level_count(1),
                    )],
            );
        }
    }

    pub fn submit_and_present_queue(&self, command_buffers: &[vk::CommandBuffer]) -> Result<()> {
        unsafe {
            self.ctx.device().queue_submit(
                self.ctx.present_queue(),
                &[vk::SubmitInfo::default()
                    .wait_semaphores(&[self.present_semaphore()])
                    .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
                    .command_buffers(command_buffers)
                    .signal_semaphores(&[self.submit_semaphore()])],
                self.current_present_fence(),
            )?;
            self.loader.queue_present(
                self.ctx.present_queue(),
                &vk::PresentInfoKHR::default()
                    .wait_semaphores(&[self.submit_semaphore()])
                    .swapchains(&[self.khr])
                    .image_indices(&[self.image_index.lock().unwrap().current as u32]),
            )?;
        }
        Ok(())
    }

    fn present_semaphore(&self) -> vk::Semaphore {
        // The present semaphore must be chosen before we know the next image index,
        // so we have to use `.last`.
        self.present_semaphores[self.image_index.lock().unwrap().last]
    }
    fn submit_semaphore(&self) -> vk::Semaphore {
        self.submit_semaphores[self.image_index.lock().unwrap().current]
    }
    fn last_acquire_fence(&self) -> vk::Fence {
        self.acquire_fences[self.frame_index.lock().unwrap().last]
    }
    fn last_present_fence(&self) -> vk::Fence {
        self.present_fences[self.frame_index.lock().unwrap().last]
    }
    fn current_present_fence(&self) -> vk::Fence {
        self.present_fences[self.frame_index.lock().unwrap().current]
    }
    fn current_present_image(&self) -> vk::Image {
        self.images.present_images[self.image_index.lock().unwrap().current]
    }
    fn current_present_image_view(&self) -> vk::ImageView {
        self.images.present_image_views[self.image_index.lock().unwrap().current]
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe {
            self.ctx.device().device_wait_idle().unwrap();
            for command_pool in self.present_command_pools.drain(..) {
                self.ctx.device().destroy_command_pool(command_pool, None);
            }
            for semaphore in self.present_semaphores.drain(..) {
                self.ctx.device().destroy_semaphore(semaphore, None);
            }
            for semaphore in self.submit_semaphores.drain(..) {
                self.ctx.device().destroy_semaphore(semaphore, None);
            }
            for fence in self.acquire_fences.drain(..) {
                self.ctx.device().destroy_fence(fence, None);
            }
            for fence in self.present_fences.drain(..) {
                self.ctx.device().destroy_fence(fence, None);
            }
            self.loader.destroy_swapchain(self.khr, None);
        }
    }
}
