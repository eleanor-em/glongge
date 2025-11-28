use crate::util::gg_sync::GgMutex;
use crate::{
    check_false, check_le,
    core::tulivuori::{TvWindowContext, tv},
    util::colour::Colour,
};
use anyhow::{Context, Result, bail};
use ash::{khr::swapchain, vk};
use egui_winit::winit::window::Window;
use itertools::Itertools;
use parking_lot::MutexGuard;
use std::{
    sync::Arc,
    sync::atomic::{AtomicBool, Ordering},
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
        unsafe {
            let surface_format = self
                .ctx
                .surface_loader
                .get_physical_device_surface_formats(self.ctx.physical_device, self.ctx.surface)
                .context("SwapchainBuilder::build(): vkGetPhysicalDeviceSurfaceFormats() failed")?
                .into_iter()
                .find(self.filter_surface_format)
                .context(
                    "SwapchainBuilder::build(): could not find surface format matching filter",
                )?;
            let surface_capabilities = self
                .ctx
                .surface_loader
                .get_physical_device_surface_capabilities(
                    self.ctx.physical_device,
                    self.ctx.surface,
                )
                .context(
                    "SwapchainBuilder::build(): vkGetPhysicalDeviceSurfaceCapabilities() failed",
                )?;
            let desired_image_count = self.desired_image_count(surface_capabilities);
            let desired_frames_in_flight = self
                .desired_frames_in_flight(surface_capabilities)
                .context("SwapchainBuilder::build()")?;
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
            let present_mode = self
                .ctx
                .surface_loader
                .get_physical_device_surface_present_modes(
                    self.ctx.physical_device,
                    self.ctx.surface,
                ).context("SwapchainBuilder::build(): vkGetPhysicalDeviceSurfacePresentModes() failed")?
                .into_iter()
                .min_by_key(tv::present_mode_key)
                .context("SwapchainBuilder::build(): vkGetPhysicalDeviceSurfacePresentModes() returned empty?")?;
            info!("using present mode: {present_mode:?}");

            let swapchain_loader = self.ctx.create_swapchain_device();
            let swapchain_khr = swapchain_loader
                .create_swapchain(
                    &self.swapchain_create_info.unwrap_or(
                        // TODO: study settings here.
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
                )
                .context("SwapchainBuilder::build(): vkCreateSwapchain() failed")?;

            let images =
                SwapchainImages::new(&self.ctx, swapchain_khr, &swapchain_loader, surface_format)
                    .context("SwapchainBuilder::build()")?;

            let mut present_semaphores = Vec::new();
            let mut submit_semaphores = Vec::new();
            let mut present_fences = Vec::new();
            for _ in 0..desired_image_count {
                present_semaphores.push(
                    self.ctx
                        .device()
                        .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                        .context("SwapchainBuilder::build(): vkCreateSemaphore() failed")?,
                );
                submit_semaphores.push(
                    self.ctx
                        .device()
                        .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                        .context("SwapchainBuilder::build(): vkCreateSemaphore() failed")?,
                );
            }
            let first_present_semaphore = self
                .ctx
                .device()
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                .context("SwapchainBuilder::build(): vkCreateSemaphore() failed")?;
            for _ in 0..desired_frames_in_flight {
                present_fences.insert(
                    0,
                    self.ctx
                        .device()
                        .create_fence(
                            &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                            None,
                        )
                        .context("SwapchainBuilder::build(): vkCreateFence() failed")?,
                );
            }

            let mut present_command_pools = Vec::new();
            let mut present_command_buffers = Vec::new();
            for _ in 0..desired_frames_in_flight {
                let command_pool = self
                    .ctx
                    .device()
                    .create_command_pool(
                        &vk::CommandPoolCreateInfo::default()
                            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
                            .queue_family_index(self.ctx.queue_family_index()),
                        None,
                    )
                    .context("SwapchainBuilder::build(): vkCreateCommandPool() failed")?;
                present_command_pools.push(command_pool);
                present_command_buffers.extend(
                    self.ctx
                        .device()
                        .allocate_command_buffers(
                            &vk::CommandBufferAllocateInfo::default()
                                .command_buffer_count(1)
                                .command_pool(command_pool)
                                .level(vk::CommandBufferLevel::PRIMARY),
                        )
                        .context("SwapchainBuilder::build(): vkAllocateCommandBuffers() failed")?,
                );
            }

            let frame_index = GgMutex::new(FrameIndex::new());
            let image_index = PresentImageIndex::new();

            Ok(Swapchain {
                ctx: self.ctx,
                khr: swapchain_khr,
                loader: swapchain_loader,
                surface_format,
                surface_resolution,
                images,
                frames_in_flight: desired_frames_in_flight,
                present_semaphores,
                first_present_semaphore,
                submit_semaphores,
                present_fences,
                present_command_pools,
                present_command_buffers,
                frame_index,
                image_index,
                did_vk_free: AtomicBool::new(false),
            })
        }
    }
}

struct FrameIndex {
    last: Option<usize>,
    current: Option<usize>,
}

impl FrameIndex {
    fn new() -> Self {
        Self {
            last: None,
            current: None,
        }
    }
}
struct PresentImageIndex {
    last: Option<usize>,
    current: Option<usize>,
}

impl PresentImageIndex {
    fn new() -> Self {
        Self {
            last: None,
            current: None,
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
            let present_images = swapchain_loader
                .get_swapchain_images(swapchain_khr)
                .context("SwapchainImages::new(): vkGetSwapchainImages() failed")?;
            let present_image_views = present_images
                .iter()
                .map(|&image| {
                    ctx.device()
                        .create_image_view(
                            &vk::ImageViewCreateInfo::default()
                                .view_type(vk::ImageViewType::TYPE_2D)
                                .format(surface_format.format)
                                .components(tv::default_component_mapping())
                                .subresource_range(tv::default_image_subresource_range())
                                .image(image),
                            None,
                        )
                        .context("SwapchainImages::new(): vkCreateImageView() failed")
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
    acquired_frame_index: usize,
    frames_in_flight: usize,
    is_suboptimal: bool,
}

impl SwapchainAcquireInfo {
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
    first_present_semaphore: vk::Semaphore,
    submit_semaphores: Vec<vk::Semaphore>,
    present_fences: Vec<vk::Fence>,

    present_command_pools: Vec<vk::CommandPool>,
    present_command_buffers: Vec<vk::CommandBuffer>,

    frame_index: GgMutex<FrameIndex>,
    image_index: PresentImageIndex,

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
    fn frame_index(&self) -> Result<MutexGuard<'_, FrameIndex>> {
        self.frame_index
            .try_lock("Swapchain::frame_index()")?
            .context("should only be accessed by render thread")
    }
    pub fn current_frame_index(&self) -> Result<usize> {
        check_false!(self.did_vk_free.load(Ordering::Relaxed));
        self.frame_index()
            .context("Swapchain::current_frame_index()")?
            .current
            .context(
                "Swapchain::current_frame_index(): no image acquired; there is no current frame",
            )
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
        self.acquire_next_image_timeout(extra_fences, u64::MAX, u64::MAX)
            .context("Swapchain::acquire_next_image()")?
            .context("Swapchain::acquire_next_image(): VK_TIMEOUT even though we waited forever (spec violation?)")
    }
    pub fn acquire_next_image_timeout(
        &mut self,
        extra_fences: &[vk::Fence],
        wait_timeout: u64,
        acquire_timeout: u64,
    ) -> Result<Option<SwapchainAcquireInfo>> {
        check_false!(self.did_vk_free.load(Ordering::Relaxed));

        // Update {frame,present}_index.
        let acquired_frame_index = {
            let mut frame_index = self
                .frame_index()
                .context("Swapchain::acquire_next_image_timeout()")?;
            frame_index.last = frame_index.current;
            let current_frame_index = if let Some(current_frame_index) = frame_index.current {
                (current_frame_index + 1) % self.frames_in_flight
            } else {
                0
            };
            frame_index.current = Some(current_frame_index);
            current_frame_index
        };
        self.image_index.last = self.image_index.current;

        match unsafe {
            let mut fences = vec![
                self.current_present_fence()
                    .context("Swapchain::acquire_next_image_timeout()")?,
            ];
            fences.extend_from_slice(extra_fences);
            self.ctx
                .device()
                .wait_for_fences(&fences, true, wait_timeout)
                .context("Swapchain::acquire_next_image_timeout(): vkWaitForFences() failed")?;
            self.ctx
                .device()
                .reset_fences(&fences)
                .context("Swapchain::acquire_next_image_timeout(): vkResetFences() failed")?;
            let rv = self
                .loader
                .acquire_next_image(
                    self.khr,
                    acquire_timeout,
                    self.present_semaphore(),
                    vk::Fence::null(),
                )
                .context("Swapchain::acquire_next_image_timeout(): vkAcquireNextImage() failed")?;
            Ok(rv)
        } {
            Ok((acquired_image_index, is_suboptimal)) => {
                self.image_index.current = Some(acquired_image_index as usize);
                Ok(Some(SwapchainAcquireInfo {
                    acquired_frame_index,
                    frames_in_flight: self.frames_in_flight,
                    is_suboptimal,
                }))
            }
            Err(vk::Result::TIMEOUT) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    pub fn submit_and_present_queue(&self, command_buffers: &[vk::CommandBuffer]) -> Result<()> {
        check_false!(self.did_vk_free.load(Ordering::Relaxed));
        unsafe {
            let queue = self.ctx.present_queue("submit_and_present_queue")?;
            self.ctx
                .device()
                .queue_submit2(
                    *queue,
                    &[vk::SubmitInfo2::default()
                        .wait_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                            .semaphore(self.present_semaphore())
                            .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)])
                        .signal_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                            .semaphore(self.submit_semaphore().context(
                                "Swapchain::submit_and_present_queue(): vkQueueSubmit2()",
                            )?)
                            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)])
                        .command_buffer_infos(
                            &command_buffers
                                .iter()
                                .map(|&cb| {
                                    vk::CommandBufferSubmitInfo::default().command_buffer(cb)
                                })
                                .collect_vec(),
                        )],
                    self.current_present_fence()
                        .context("Swapchain::submit_and_present_queue(): vkQueueSubmit2()")?,
                )
                .context("Swapchain::submit_and_present_queue(): vkQueueSubmit2() failed")?;
            self.loader
                .queue_present(
                    *queue,
                    &vk::PresentInfoKHR::default()
                        .wait_semaphores(&[self
                            .submit_semaphore()
                            .context("Swapchain::submit_and_present_queue(): vkQueuePresent()")?])
                        .swapchains(&[self.khr])
                        .image_indices(&[self
                            .current_image_index()
                            .context("Swapchain::submit_and_present_queue(): vkQueuePresent()")?
                            as u32]),
                )
                .context("Swapchain::submit_and_present_queue(): vkQueuePresent() failed")?;
        }
        Ok(())
    }

    pub fn acquire_present_command_buffer(&self) -> Result<vk::CommandBuffer> {
        check_false!(self.did_vk_free.load(Ordering::Relaxed));
        let current_frame_index = self
            .current_frame_index()
            .context("Swapchain::acquire_present_command_buffer")?;
        unsafe {
            self.ctx.device().reset_command_pool(
                self.present_command_pools[current_frame_index],
                vk::CommandPoolResetFlags::RELEASE_RESOURCES,
            )?;
        }
        Ok(self.present_command_buffers[current_frame_index])
    }

    pub fn cmd_begin_rendering(
        &self,
        command_buffer: vk::CommandBuffer,
        clear_col: Option<Colour>,
    ) -> Result<()> {
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
                        .dst_access_mask(
                            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
                                | vk::AccessFlags2::COLOR_ATTACHMENT_READ,
                        )
                        .image(
                            self.current_present_image().context(
                                "Swapchain::cmd_begin_rendering(): vkCmdPipelineBarrier2()",
                            )?,
                        )
                        .subresource_range(tv::default_image_subresource_range()),
                ]),
            );
            self.ctx.device().cmd_begin_rendering(
                command_buffer,
                &vk::RenderingInfo::default()
                    .color_attachments(&[vk::RenderingAttachmentInfo::default()
                        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .load_op(if clear_col.is_some() {
                            vk::AttachmentLoadOp::CLEAR
                        } else {
                            vk::AttachmentLoadOp::LOAD
                        })
                        .clear_value(vk::ClearValue {
                            color: vk::ClearColorValue {
                                float32: clear_col.unwrap_or(Colour::empty()).into(),
                            },
                        })
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .image_view(
                            self.current_present_image_view().context(
                                "Swapchain::cmd_begin_rendering(): vkCmdBeginRendering()",
                            )?,
                        )])
                    .render_area(self.surface_resolution().into())
                    .layer_count(1),
            );
            Ok(())
        }
    }
    pub fn cmd_end_rendering(&self, command_buffer: vk::CommandBuffer) -> Result<()> {
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
                        .image(
                            self.current_present_image().context(
                                "Swapchain::cmd_end_rendering(): vkCmdPipelineBarrier2()",
                            )?,
                        )
                        .subresource_range(tv::default_image_subresource_range()),
                ]),
            );
            Ok(())
        }
    }

    fn present_semaphore(&self) -> vk::Semaphore {
        // The present semaphore must be chosen before we know the next image index,
        // so we have to use `.last`.
        if let Some(last_image_index) = self.image_index.last {
            self.present_semaphores[last_image_index]
        } else {
            self.first_present_semaphore
        }
    }
    fn current_image_index(&self) -> Result<usize> {
        self.image_index
            .current
            .context("no current image index, call acquire_next_image{,_timeout}() first")
    }
    fn submit_semaphore(&self) -> Result<vk::Semaphore> {
        Ok(self.submit_semaphores[self
            .current_image_index()
            .context("Swapchain::submit_semaphore()")?])
    }
    fn current_present_fence(&self) -> Result<vk::Fence> {
        Ok(self.present_fences[self
            .current_frame_index()
            .context("Swapchain::current_present_fence()")?])
    }
    fn current_present_image(&self) -> Result<vk::Image> {
        Ok(self.images.present_images[self
            .current_image_index()
            .context("Swapchain::current_present_image()")?])
    }
    fn current_present_image_view(&self) -> Result<vk::ImageView> {
        Ok(self.images.present_image_views[self
            .current_image_index()
            .context("Swapchain::current_present_image_view()")?])
    }

    pub fn vk_free(&self) {
        check_false!(self.did_vk_free.swap(true, Ordering::Relaxed));
        unsafe {
            for &command_pool in &self.present_command_pools {
                self.ctx.device().destroy_command_pool(command_pool, None);
            }
            for &semaphore in &self.present_semaphores {
                self.ctx.device().destroy_semaphore(semaphore, None);
            }
            for &semaphore in &self.submit_semaphores {
                self.ctx.device().destroy_semaphore(semaphore, None);
            }
            self.ctx
                .device()
                .destroy_semaphore(self.first_present_semaphore, None);
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
