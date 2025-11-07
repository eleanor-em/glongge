use crate::{
    check_false, check_le,
    core::tulivuori::{TvWindowContext, tv},
    util::colour::Colour,
};
use anyhow::{Context, Result, bail};
use ash::{khr::swapchain, vk};
use egui_winit::winit::window::Window;
use itertools::Itertools;
use std::{
    sync::Arc,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};
use tracing::{error, info};

pub struct SwapchainBuilder<'a> {
    ctx: Arc<TvWindowContext>,
    window: Arc<Window>,

    // Settings:
    filter_surface_format: fn(&'_ vk::SurfaceFormatKHR) -> bool,
    get_desired_image_count: Option<fn(vk::SurfaceCapabilitiesKHR) -> u32>,
    get_desired_frames_in_flight: Option<fn(vk::SurfaceCapabilitiesKHR) -> usize>,
    swapchain_create_info: Option<vk::SwapchainCreateInfoKHR<'a>>,
}

impl<'a> SwapchainBuilder<'a> {
    pub fn new(ctx: &Arc<TvWindowContext>, window: Arc<Window>) -> SwapchainBuilder<'a> {
        Self {
            ctx: ctx.clone(),
            window,
            filter_surface_format: |format| {
                format.format == vk::Format::B8G8R8A8_SRGB
                    && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            },
            get_desired_image_count: None,
            get_desired_frames_in_flight: None,
            swapchain_create_info: None,
        }
    }

    fn desired_image_count(&self, surface_capabilities: vk::SurfaceCapabilitiesKHR) -> u32 {
        let rv = if let Some(get_desired_image_count) = self.get_desired_image_count {
            get_desired_image_count(surface_capabilities)
        } else {
            // Default case
            surface_capabilities.min_image_count + 1
        };
        // Some platforms (especially Intel) use 0 to mean "no limit".
        if surface_capabilities.max_image_count > 0 {
            check_le!(rv, surface_capabilities.max_image_count);
        }
        rv
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
        let desired_image_count = self.desired_image_count(surface_capabilities);
        if rv > desired_image_count as usize {
            bail!(
                "more frames in flight than images, this is almost certainly unintended: {rv} vs. {desired_image_count}",
            );
        }
        Ok(rv)
    }

    #[allow(clippy::too_many_lines)]
    pub fn build(self) -> Result<Swapchain> {
        let surface_format =
            self.ctx.get_physical_device_surface_formats()?
                .into_iter()
                .find(self.filter_surface_format)
                .context("get_physical_device_surface_formats(): could not find surface format matching filter")?;
        let surface_capabilities = self.ctx.get_physical_device_surface_capabilities()?;
        let desired_image_count = self.desired_image_count(surface_capabilities);
        let desired_frames_in_flight = self.desired_frames_in_flight(surface_capabilities)?;
        let surface_resolution = match surface_capabilities.current_extent.width {
            u32::MAX => vk::Extent2D::default()
                .width(self.window.inner_size().width)
                .height(self.window.inner_size().height),
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
            self.ctx
                .get_physical_device_surface_present_modes()?
                .into_iter()
                .min_by_key(tv::present_mode_key)
                .context("get_physical_device_surface_present_modes() returned empty?")?
        };
        info!("using present mode: {present_mode:?}");

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
                        .flags(vk::CommandPoolCreateFlags::TRANSIENT)
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
        let last_frame_index = AtomicUsize::new(frame_index.current);

        Ok(Swapchain {
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
            current_frame_index: Arc::new(last_frame_index),
            did_vk_free: AtomicBool::new(false),
        })
    }
}

struct FrameIndex {
    last: usize,
    current: usize,
}

impl FrameIndex {
    fn new(frames_in_flight: usize) -> Self {
        Self {
            last: 0,
            current: frames_in_flight - 1,
        }
    }
}
struct PresentIndex {
    last: usize,
    current: usize,
}

impl PresentIndex {
    fn new() -> Self {
        Self {
            last: 0,
            current: 0,
        }
    }
}

struct SwapchainImages {
    ctx: Arc<TvWindowContext>,
    present_images: Vec<vk::Image>,
    present_image_views: Vec<vk::ImageView>,
    did_vk_free: AtomicBool,
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
                    let image_view = ctx.device().create_image_view(
                        &vk::ImageViewCreateInfo::default()
                            .view_type(vk::ImageViewType::TYPE_2D)
                            .format(surface_format.format)
                            .components(tv::default_component_mapping())
                            .subresource_range(tv::default_image_subresource_range())
                            .image(image),
                        None,
                    )?;
                    Ok(image_view)
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(Self {
                ctx: ctx.clone(),
                present_images,
                present_image_views,
                did_vk_free: AtomicBool::new(false),
            })
        }
    }

    pub fn vk_free(&self) {
        check_false!(self.did_vk_free.swap(true, Ordering::Relaxed));
        unsafe {
            self.ctx.device().device_wait_idle().unwrap();
            for &image_view in &self.present_image_views {
                self.ctx.device().destroy_image_view(image_view, None);
            }
        }
    }
}

impl Drop for SwapchainImages {
    fn drop(&mut self) {
        if !self.did_vk_free.load(Ordering::Relaxed) {
            error!("leaked resource: SwapchainImages");
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

    frame_index: FrameIndex,
    image_index: PresentIndex,
    pub(crate) current_frame_index: Arc<AtomicUsize>,

    did_vk_free: AtomicBool,
}

impl Swapchain {
    pub fn surface_resolution(&self) -> vk::Extent2D {
        check_false!(self.did_vk_free.load(Ordering::Relaxed));
        self.surface_resolution
    }
    pub fn frames_in_flight(&self) -> usize {
        check_false!(self.did_vk_free.load(Ordering::Relaxed));
        self.frames_in_flight
    }
    pub fn current_frame_index(&self) -> usize {
        check_false!(self.did_vk_free.load(Ordering::Relaxed));
        self.current_frame_index.load(Ordering::Relaxed)
    }
    pub fn surface_format(&self) -> vk::SurfaceFormatKHR {
        check_false!(self.did_vk_free.load(Ordering::Relaxed));
        self.surface_format
    }

    pub fn acquire_next_image(
        &mut self,
        extra_fences: &[vk::Fence],
    ) -> Result<SwapchainAcquireInfo> {
        check_false!(self.did_vk_free.load(Ordering::Relaxed));
        self.acquire_next_image_timeout(extra_fences, u64::MAX, u64::MAX)?
            .context("timeout even though we waited forever")
    }
    pub fn acquire_next_image_timeout(
        &mut self,
        extra_fences: &[vk::Fence],
        wait_timeout: u64,
        acquire_timeout: u64,
    ) -> Result<Option<SwapchainAcquireInfo>> {
        check_false!(self.did_vk_free.load(Ordering::Relaxed));
        {
            // Update {frame,present}_index.
            self.frame_index.last = self.frame_index.current;
            self.frame_index.current = (self.frame_index.last + 1) % self.frames_in_flight;
            self.current_frame_index
                .store(self.frame_index.current, Ordering::Relaxed);

            self.image_index.last = self.image_index.current;
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
                self.image_index.current = next_image_index;
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
        check_false!(self.did_vk_free.load(Ordering::Relaxed));
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
        check_false!(self.did_vk_free.load(Ordering::Relaxed));
        unsafe {
            self.ctx.device().cmd_pipeline_barrier2(
                command_buffer,
                &vk::DependencyInfo::default().image_memory_barriers(&[
                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                        .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .src_access_mask(vk::AccessFlags2::NONE)
                        .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                        .image(self.current_present_image())
                        .subresource_range(tv::default_image_subresource_range()),
                ]),
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
            self.ctx.device().cmd_pipeline_barrier2(
                command_buffer,
                &vk::DependencyInfo::default().image_memory_barriers(&[
                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                        .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                        .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                        .dst_access_mask(vk::AccessFlags2::NONE)
                        .image(self.current_present_image())
                        .subresource_range(tv::default_image_subresource_range()),
                ]),
            );
        }
    }

    pub fn submit_and_present_queue(&self, command_buffers: &[vk::CommandBuffer]) -> Result<()> {
        check_false!(self.did_vk_free.load(Ordering::Relaxed));
        unsafe {
            let queue = self.ctx.present_queue()?;
            self.ctx.device().queue_submit2(
                *queue,
                &[vk::SubmitInfo2::default()
                    .wait_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                        .semaphore(self.present_semaphore())
                        .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)])
                    .signal_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                        .semaphore(self.submit_semaphore())
                        .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)])
                    .command_buffer_infos(
                        &command_buffers
                            .iter()
                            .map(|&cb| vk::CommandBufferSubmitInfo::default().command_buffer(cb))
                            .collect_vec(),
                    )],
                self.current_present_fence(),
            )?;
            self.loader.queue_present(
                *queue,
                &vk::PresentInfoKHR::default()
                    .wait_semaphores(&[self.submit_semaphore()])
                    .swapchains(&[self.khr])
                    .image_indices(&[self.image_index.current as u32]),
            )?;
        }
        Ok(())
    }

    fn present_semaphore(&self) -> vk::Semaphore {
        // The present semaphore must be chosen before we know the next image index,
        // so we have to use `.last`.
        self.present_semaphores[self.image_index.last]
    }
    fn submit_semaphore(&self) -> vk::Semaphore {
        self.submit_semaphores[self.image_index.current]
    }
    fn last_acquire_fence(&self) -> vk::Fence {
        self.acquire_fences[self.frame_index.last]
    }
    fn last_present_fence(&self) -> vk::Fence {
        self.present_fences[self.frame_index.last]
    }
    fn current_present_fence(&self) -> vk::Fence {
        self.present_fences[self.frame_index.current]
    }
    fn current_present_image(&self) -> vk::Image {
        self.images.present_images[self.image_index.current]
    }
    fn current_present_image_view(&self) -> vk::ImageView {
        self.images.present_image_views[self.image_index.current]
    }

    pub fn vk_free(&self) {
        check_false!(self.did_vk_free.swap(true, Ordering::Relaxed));
        unsafe {
            self.ctx.device().device_wait_idle().unwrap();
            for &command_pool in &self.present_command_pools {
                self.ctx.device().destroy_command_pool(command_pool, None);
            }
            for &semaphore in &self.present_semaphores {
                self.ctx.device().destroy_semaphore(semaphore, None);
            }
            for &semaphore in &self.submit_semaphores {
                self.ctx.device().destroy_semaphore(semaphore, None);
            }
            for &fence in &self.acquire_fences {
                self.ctx.device().destroy_fence(fence, None);
            }
            for &fence in &self.present_fences {
                self.ctx.device().destroy_fence(fence, None);
            }
            self.images.vk_free();
            self.loader.destroy_swapchain(self.khr, None);
        }
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        if !self.did_vk_free.load(Ordering::Relaxed) {
            error!("leaked resource: Swapchain");
        }
    }
}
