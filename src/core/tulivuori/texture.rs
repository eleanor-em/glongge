use crate::core::config::TEXTURE_STATS_INTERVAL_S;
use crate::core::prelude::*;
use crate::core::tulivuori::swapchain::SwapchainAcquireInfo;
use crate::{
    check, check_eq, check_false,
    core::config::{FONT_SAMPLE_RATIO, FRAMES_IN_FLIGHT, MAX_MATERIAL_COUNT, MAX_TEXTURE_COUNT},
    core::linalg::AxisAlignedExtent,
    core::tulivuori::buffer::GenericBuffer,
    core::tulivuori::buffer::GenericDeviceBuffer,
    core::tulivuori::{TvWindowContext, tv},
    resource::texture::{Material, MaterialId},
};
use anyhow::{Context, Result};
use ash::{util::Align, vk};
use std::collections::BTreeSet;
use std::time::{Duration, Instant};
use std::{
    collections::BTreeMap,
    sync::Arc,
    sync::atomic::{AtomicBool, Ordering},
};
use tracing::{error, info};
use vk_mem::Alloc;

/// Buffer index used for out-of-band uploads (init, async texture loading outside frame loop).
const OUT_OF_BAND_FRAME_INDEX: usize = FRAMES_IN_FLIGHT;

#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Default, Hash)]
pub struct TextureId(u32);

impl TextureId {
    pub(crate) fn as_usize(self) -> usize {
        self.0 as usize
    }
}

pub struct TvInternalTexture {
    ctx: Arc<TvWindowContext>,
    id: TextureId,
    image_buffer: vk::Buffer,
    image_buffer_alloc: vk_mem::Allocation,
    image_extent: vk::Extent2D,
    tex_image_view: vk::ImageView,
    tex_image: vk::Image,
    tex_alloc: vk_mem::Allocation,
    data: Vec<u8>,
    ready_flag: Arc<AtomicBool>,
    do_bind: bool,
    did_vk_free: AtomicBool,
}

impl TvInternalTexture {
    fn new(
        ctx: Arc<TvWindowContext>,
        id: TextureId,
        image_extent: vk::Extent2D,
        format: vk::Format,
        image_data: &[u8],
        do_bind: bool,
        ready_flag: Arc<AtomicBool>,
    ) -> Result<Arc<Self>> {
        unsafe {
            let allocator = ctx.allocator("TvInternalTexture::new")?;
            let (image_buffer, mut image_buffer_alloc) = allocator
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(size_of_val(image_data) as u64)
                        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    &vk_mem::AllocationCreateInfo {
                        flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
                        usage: vk_mem::MemoryUsage::Auto,
                        ..Default::default()
                    },
                )
                .context("TvInternalTexture::new(): vmaCreateBuffer() failed")?;
            let image_alloc_size = allocator.get_allocation_info(&image_buffer_alloc).size;
            let image_ptr = allocator
                .map_memory(&mut image_buffer_alloc)
                .context("TvInternalTexture::new(): vmaMapMemory() failed")?;
            Align::new(image_ptr.cast(), align_of::<u8>() as u64, image_alloc_size)
                .copy_from_slice(image_data);
            allocator.unmap_memory(&mut image_buffer_alloc);

            let (tex_image, tex_alloc) = allocator
                .create_image(
                    // TODO: study settings here.
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(format)
                        .extent(image_extent.into())
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    &vk_mem::AllocationCreateInfo {
                        usage: vk_mem::MemoryUsage::AutoPreferDevice,
                        ..Default::default()
                    },
                )
                .context("TvInternalTexture::new(): vmaCreateImage() failed")?;
            let tex_image_view = ctx
                .device()
                .create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(format)
                        .components(tv::default_component_mapping())
                        .subresource_range(tv::default_image_subresource_range())
                        .image(tex_image),
                    None,
                )
                .context("TvInternalTexture::new(): vkCreateImageView() failed")?;

            Ok(Arc::new(Self {
                ctx,
                id,
                image_buffer,
                image_buffer_alloc,
                image_extent,
                tex_image_view,
                tex_image,
                tex_alloc,
                data: image_data.to_vec(),
                ready_flag,
                do_bind,
                did_vk_free: AtomicBool::new(false),
            }))
        }
    }

    pub fn id(&self) -> TextureId {
        self.id
    }
    pub fn extent(&self) -> vk::Extent2D {
        self.image_extent
    }
    pub fn data(&self) -> &[u8] {
        &self.data
    }
    pub fn is_ready(&self) -> bool {
        self.ready_flag.load(Ordering::Relaxed)
    }
    pub fn tex_image_view(&self) -> vk::ImageView {
        self.tex_image_view
    }

    pub fn vk_free(&self) {
        check_false!(self.did_vk_free.swap(true, Ordering::Relaxed));
        unsafe {
            let allocator = self.ctx.allocator("TvInternalTexture::vk_free").unwrap();
            allocator.destroy_buffer(self.image_buffer, &mut self.image_buffer_alloc.clone());
            self.ctx
                .device()
                .destroy_image_view(self.tex_image_view, None);
            allocator.destroy_image(self.tex_image, &mut self.tex_alloc.clone());
        }
    }
}

impl Drop for TvInternalTexture {
    fn drop(&mut self) {
        if !self.did_vk_free.load(Ordering::Relaxed) {
            error!("leaked resource: TvInternalTexture");
        }
    }
}

#[derive(Copy, Clone, Default)]
#[repr(C)]
struct RawMaterial {
    uv_top_left: [f32; 2],
    uv_bottom_right: [f32; 2],
    texture_id: TextureId,
    #[allow(unused)]
    dummy1: u32,
    #[allow(unused)]
    dummy2: u32,
    #[allow(unused)]
    dummy3: u32,
}

pub struct TextureManager {
    ctx: Arc<TvWindowContext>,
    sampler: vk::Sampler,
    descriptor_pool: vk::DescriptorPool,
    desc_set_layout: vk::DescriptorSetLayout,
    descriptor_set: vk::DescriptorSet,
    descriptor_image_infos: Vec<vk::DescriptorImageInfo>,
    pipeline_layout: vk::PipelineLayout,

    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    upload_fence: vk::Fence,
    is_upload_fence_in_use: bool,

    next_texture_id: u32,
    blank_texture: Option<Arc<TvInternalTexture>>,
    pending_textures: BTreeMap<TextureId, Arc<TvInternalTexture>>,
    uploading_textures: BTreeMap<TextureId, Arc<TvInternalTexture>>,
    textures: BTreeMap<TextureId, Arc<TvInternalTexture>>,
    unused_texture_ids: BTreeSet<TextureId>,

    materials_upload_countdown: usize,
    last_staged_buffer_index: Option<usize>,
    material_device_buffer: GenericDeviceBuffer<RawMaterial>,
    material_staging_buffer: GenericBuffer<RawMaterial>,

    last_reported_stage: Option<Instant>,
    staged_since_last_report: usize,
    last_reported_upload: Option<Instant>,
    uploaded_since_last_report: usize,

    did_vk_free: AtomicBool,
}

impl TextureManager {
    // TODO: fix this properly
    #[allow(
        clippy::too_many_lines,
        clippy::large_stack_arrays,
        clippy::unreadable_literal
    )]
    pub(crate) fn new(ctx: Arc<TvWindowContext>) -> Result<TextureManager> {
        check_eq!(
            size_of_val(&[RawMaterial::default(); MAX_MATERIAL_COUNT]),
            524288
        );
        unsafe {
            let sampler = ctx
                .device()
                .create_sampler(
                    // TODO: really should use a separate sampler for fonts.
                    &vk::SamplerCreateInfo::default()
                        .mag_filter(vk::Filter::NEAREST)
                        .min_filter(vk::Filter::LINEAR) // for fonts
                        .min_lod(0.0)
                        .max_lod(FONT_SAMPLE_RATIO.log2() + 1.0)
                        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                        .max_anisotropy(
                            ctx.physical_device_properties()
                                .limits
                                .max_sampler_anisotropy,
                        )
                        .border_color(vk::BorderColor::FLOAT_TRANSPARENT_BLACK),
                    None,
                )
                .context("TextureManager::new(): vkCreateSampler() failed")?;

            let descriptor_pool = ctx
                .device()
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
                        .pool_sizes(&[
                            vk::DescriptorPoolSize {
                                ty: vk::DescriptorType::SAMPLER,
                                descriptor_count: 1,
                            },
                            vk::DescriptorPoolSize {
                                ty: vk::DescriptorType::SAMPLED_IMAGE,
                                descriptor_count: MAX_TEXTURE_COUNT as u32,
                            },
                        ])
                        .max_sets(1),
                    None,
                )
                .context("TextureManager::new(): vkCreateDescriptorPool() failed")?;
            let desc_set_layout = ctx
                .device()
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default()
                        .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                        .bindings(&[
                            vk::DescriptorSetLayoutBinding::default()
                                .binding(0)
                                .descriptor_type(vk::DescriptorType::SAMPLER)
                                .immutable_samplers(&[sampler])
                                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
                            vk::DescriptorSetLayoutBinding::default()
                                .binding(1)
                                .descriptor_count(MAX_TEXTURE_COUNT as u32)
                                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
                        ])
                        .push_next(
                            &mut vk::DescriptorSetLayoutBindingFlagsCreateInfo::default()
                                .binding_flags(&[
                                    vk::DescriptorBindingFlags::empty(),
                                    vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
                                ]),
                        ),
                    None,
                )
                .context("TextureManager::new(): vkCreateDescriptorSetLayout() failed")?;
            let descriptor_set = ctx
                .device()
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(descriptor_pool)
                        .set_layouts(&[desc_set_layout]),
                )
                .context("TextureManager::new(): vkAllocateDescriptorSets() failed")?[0];

            let pipeline_layout = ctx
                .device()
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&[desc_set_layout])
                        .push_constant_ranges(&[vk::PushConstantRange::default()
                            .stage_flags(vk::ShaderStageFlags::VERTEX)
                            .size(16)]),
                    None,
                )
                .context("TextureManager::new(): vkCreatePipelineLayout() failed")?;

            let command_pool = ctx
                .device()
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .flags(vk::CommandPoolCreateFlags::TRANSIENT)
                        .queue_family_index(ctx.queue_family_index()),
                    None,
                )
                .context("TextureManager::new(): vkCreateCommandPool() failed")?;
            let command_buffer = ctx
                .device()
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_buffer_count(1)
                        .command_pool(command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY),
                )
                .context("TextureManager::new(): vkAllocateCommandBuffers() failed")?
                .first()
                .copied()
                .context("TextureManager::new(): vkAllocateCommandBuffers() returned empty?")?;
            let upload_fence = ctx
                .device()
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .context("TextureManager::new(): vkCreateFence() failed")?;

            // XXX: see comment above FRAMES_IN_FLIGHT in config.rs.
            let material_device_buffer = GenericDeviceBuffer::new(
                ctx.clone(),
                FRAMES_IN_FLIGHT,
                MAX_MATERIAL_COUNT,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            )
            .context("TextureManager::new()")?;
            // +1 for out-of-band uploads (see OUT_OF_BAND_FRAME_INDEX).
            let material_staging_buffer = GenericBuffer::new(
                ctx.clone(),
                FRAMES_IN_FLIGHT + 1,
                MAX_MATERIAL_COUNT,
                vk::BufferUsageFlags::TRANSFER_SRC,
            )
            .context("TextureManager::new()")?;

            let mut rv = Self {
                ctx,
                sampler,
                descriptor_pool,
                desc_set_layout,
                descriptor_set,
                descriptor_image_infos: Vec::new(),
                pipeline_layout,
                command_pool,
                command_buffer,
                upload_fence,
                is_upload_fence_in_use: false,
                next_texture_id: 0,
                blank_texture: None,
                pending_textures: BTreeMap::new(),
                uploading_textures: BTreeMap::new(),
                textures: BTreeMap::new(),
                unused_texture_ids: BTreeSet::new(),
                materials_upload_countdown: 0,
                last_staged_buffer_index: None,
                material_device_buffer,
                material_staging_buffer,
                last_reported_stage: None,
                staged_since_last_report: 0,
                last_reported_upload: None,
                uploaded_since_last_report: 0,
                did_vk_free: AtomicBool::new(false),
            };

            // Set up blank initial textures.
            rv.blank_texture = Some(
                rv.create_texture(
                    vk::Extent2D {
                        width: 1,
                        height: 1,
                    },
                    vk::Format::R8G8B8A8_UNORM,
                    &[255; 4],
                    Arc::new(AtomicBool::new(true)),
                )?
                .context("TextureManager::new(): failed to create blank texture")?,
            );
            rv.descriptor_image_infos = vec![
                vk::DescriptorImageInfo {
                    image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    image_view: rv.blank_texture.as_ref().unwrap().tex_image_view,
                    sampler,
                };
                MAX_TEXTURE_COUNT
            ];
            rv.stage_materials(BTreeMap::new(), None)?;
            rv.upload_all_pending_out_of_band("TextureManager::new()")?;
            check!(rv.wait_complete_upload()?);

            Ok(rv)
        }
    }

    pub fn bind(&self, command_buffer: vk::CommandBuffer) {
        check_false!(self.did_vk_free.load(Ordering::Relaxed));
        unsafe {
            self.ctx.device().cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.descriptor_set],
                &[],
            );
        }
    }

    pub fn materials_buffer_address(&self, frame_index: usize) -> vk::DeviceAddress {
        self.material_device_buffer.device_address(frame_index)
    }

    fn get_next_texture_id(&mut self) -> Option<TextureId> {
        let id = self.next_texture_id;
        self.next_texture_id += 1;
        if id < MAX_TEXTURE_COUNT as u32 {
            Some(TextureId(id))
        } else {
            self.unused_texture_ids.pop_first()
        }
    }

    pub fn create_texture_unbound(
        &mut self,
        extent: vk::Extent2D,
        format: vk::Format,
        image_data: &[u8],
        ready_flag: Arc<AtomicBool>,
    ) -> Result<Option<Arc<TvInternalTexture>>> {
        check_false!(self.did_vk_free.load(Ordering::Relaxed));
        let Some(id) = self.get_next_texture_id() else {
            return Ok(None);
        };
        let tex = TvInternalTexture::new(
            self.ctx.clone(),
            id,
            extent,
            format,
            image_data,
            false,
            ready_flag,
        )
        .context("TextureManager::create_texture_unbound()")?;
        self.pending_textures.insert(id, tex.clone());
        Ok(Some(tex))
    }
    pub fn create_texture(
        &mut self,
        extent: vk::Extent2D,
        format: vk::Format,
        image_data: &[u8],
        ready_flag: Arc<AtomicBool>,
    ) -> Result<Option<Arc<TvInternalTexture>>> {
        check_false!(self.did_vk_free.load(Ordering::Relaxed));
        let Some(id) = self.get_next_texture_id() else {
            return Ok(None);
        };
        let tex = TvInternalTexture::new(
            self.ctx.clone(),
            id,
            extent,
            format,
            image_data,
            true,
            ready_flag,
        )
        .context("TextureManager::create_texture()")?;
        self.pending_textures.insert(id, tex.clone());
        Ok(Some(tex))
    }
    pub fn get_blank_texture(&self) -> &Arc<TvInternalTexture> {
        self.blank_texture
            .as_ref()
            .expect("TextureManager::get_blank_texture(): blank texture missing")
    }

    pub fn wait_complete_upload(&mut self) -> Result<bool> {
        check_false!(self.did_vk_free.load(Ordering::Relaxed));
        if self.is_upload_fence_in_use {
            unsafe {
                self.ctx
                    .device()
                    .wait_for_fences(&[self.upload_fence], true, 1_000_000_000)
                    .context("TextureManager::wait_complete_upload(): vkWaitForFences() failed")?;
                self.ctx
                    .device()
                    .reset_fences(&[self.upload_fence])
                    .context("TextureManager::wait_complete_upload(): vkResetFences() failed")?;
                self.is_upload_fence_in_use = false;
            }
        }

        if self.uploading_textures.is_empty() {
            Ok(false)
        } else {
            for (&id, texture) in &self.uploading_textures {
                texture.ready_flag.store(true, Ordering::Relaxed);
                self.textures.insert(id, texture.clone());
                if texture.do_bind {
                    self.descriptor_image_infos[id.as_usize()] = vk::DescriptorImageInfo {
                        sampler: self.sampler,
                        image_view: texture.tex_image_view,
                        image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    };
                }
            }
            self.uploading_textures.clear();

            unsafe {
                self.ctx.device().update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default()
                        .dst_set(self.descriptor_set)
                        .dst_binding(1)
                        .descriptor_count(MAX_TEXTURE_COUNT as u32)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .image_info(&self.descriptor_image_infos)],
                    &[],
                );
            }
            Ok(true)
        }
    }

    fn is_anything_pending(&self) -> bool {
        !self.pending_textures.is_empty() || self.materials_upload_countdown > 0
    }
    /// Synchronously uploads pending textures to the GPU.
    ///
    /// Note: materials are NOT copied to the device buffer here. Since this is an out-of-band
    /// upload (no swapchain frame context), we don't know which device buffer to target.
    /// Instead, `materials_upload_countdown` is set by `stage_materials()`, and subsequent
    /// frame renders will copy from the staging buffer to each frame's device buffer.
    pub fn upload_all_pending_out_of_band(&mut self, by: &'static str) -> Result<()> {
        check_false!(self.did_vk_free.load(Ordering::Relaxed));
        if !self.is_anything_pending() {
            return Ok(());
        }

        self.wait_complete_upload()
            .context("TextureManager::upload_all_pending_out_of_band()")?;
        unsafe {
            self.ctx
                .device()
                .reset_command_pool(
                    self.command_pool,
                    vk::CommandPoolResetFlags::RELEASE_RESOURCES,
                )
                .context(
                    "TextureManager::upload_all_pending_out_of_band(): vkResetCommandPool() failed",
                )?;
            self.ctx
                .device()
                .begin_command_buffer(
                    self.command_buffer,
                    &tv::default_command_buffer_begin_info(),
                )
                .context("TextureManager::upload_all_pending_out_of_band(): vkBeginCommandBuffer() failed")?;
            self.bind(self.command_buffer);
            self.upload_materials(self.command_buffer, None);
            self.upload_textures(self.command_buffer);
            self.ctx
                .device()
                .end_command_buffer(self.command_buffer)
                .context(
                    "TextureManager::upload_all_pending_out_of_band(): vkEndCommandBuffer() failed",
                )?;
            let queue = self.ctx.present_queue(by)?;
            self.ctx
                .device()
                .queue_submit2(
                    *queue,
                    &[vk::SubmitInfo2::default()
                        .command_buffer_infos(&[vk::CommandBufferSubmitInfo::default()
                            .command_buffer(self.command_buffer)])],
                    self.upload_fence,
                )
                .context(
                    "TextureManager::upload_all_pending_out_of_band(): vkQueueSubmit2() failed",
                )?;
            self.is_upload_fence_in_use = true;

            Ok(())
        }
    }
    pub fn upload_all_pending_with(
        &mut self,
        command_buffer: vk::CommandBuffer,
        swapchain_acquire_info: &SwapchainAcquireInfo,
    ) {
        check_false!(self.did_vk_free.load(Ordering::Relaxed));
        if self.is_anything_pending() {
            self.upload_materials(command_buffer, Some(swapchain_acquire_info));
            self.upload_textures(command_buffer);
        }
    }

    pub(crate) fn stage_materials(
        &mut self,
        materials: BTreeMap<MaterialId, Material>,
        swapchain_acquire_info: Option<&SwapchainAcquireInfo>,
    ) -> Result<()> {
        check_false!(self.did_vk_free.load(Ordering::Relaxed));
        let mut data = vec![RawMaterial::default(); MAX_MATERIAL_COUNT];
        self.staged_since_last_report = self.staged_since_last_report.max(materials.len());
        if self.staged_since_last_report > 0
            && self
                .last_reported_stage
                .is_none_or(|i| i.elapsed() >= Duration::from_secs_f32(TEXTURE_STATS_INTERVAL_S))
        {
            info!("staged {} material(s)", self.staged_since_last_report);
            self.staged_since_last_report = 0;
            self.last_reported_stage = Some(Instant::now());
        }
        for (id, mat) in materials {
            let entry = &mut data[id as usize];
            // Inset by a tiny amount to avoid precision issues at texel boundaries
            // that can cause sampling adjacent texels.
            let epsilon = Vec2::splat(1.0 / 256.0);
            entry.uv_top_left = (mat.area.top_left() + epsilon)
                .component_wise_div(mat.texture_extent)
                .into();
            entry.uv_bottom_right = (mat.area.bottom_right() - epsilon)
                .component_wise_div(mat.texture_extent)
                .into();
            entry.texture_id = mat.texture_id;
        }
        // TODO: write only part of the buffer?
        let staging_index = swapchain_acquire_info.map_or(
            OUT_OF_BAND_FRAME_INDEX,
            SwapchainAcquireInfo::acquired_frame_index,
        );
        self.material_staging_buffer
            .write(&data, staging_index)
            .context("TextureManager::stage_materials()")?;
        self.last_staged_buffer_index = Some(staging_index);
        self.materials_upload_countdown = FRAMES_IN_FLIGHT;

        Ok(())
    }
    fn upload_materials(
        &mut self,
        command_buffer: vk::CommandBuffer,
        swapchain_acquire_info: Option<&SwapchainAcquireInfo>,
    ) {
        // For out-of-band uploads, don't copy -- let frame renders handle it via the countdown.
        let Some(acquire_info) = swapchain_acquire_info else {
            return;
        };
        unsafe {
            if self.materials_upload_countdown > 0 {
                let Some(last_staged_buffer_index) = self.last_staged_buffer_index else {
                    error!(
                        "materials_upload_countdown = {} but no last_staged_buffer_index",
                        self.materials_upload_countdown
                    );
                    return;
                };
                let staging_buffer = self
                    .material_staging_buffer
                    .buffer(last_staged_buffer_index);
                let device_buffer = self
                    .material_device_buffer
                    .buffer(acquire_info.acquired_frame_index());
                check_eq!(staging_buffer.alloc.size(), device_buffer.alloc.size());
                self.ctx.device().cmd_copy_buffer2(
                    command_buffer,
                    &vk::CopyBufferInfo2::default()
                        .src_buffer(staging_buffer.buffer)
                        .dst_buffer(device_buffer.buffer)
                        .regions(&[vk::BufferCopy2::default()
                            .src_offset(0)
                            .dst_offset(0)
                            .size(device_buffer.alloc.size())]),
                );
                self.ctx.device().cmd_pipeline_barrier2(
                    command_buffer,
                    &vk::DependencyInfo::default().buffer_memory_barriers(&[
                        vk::BufferMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::COPY)
                            .dst_stage_mask(
                                vk::PipelineStageFlags2::COPY
                                    | vk::PipelineStageFlags2::VERTEX_SHADER,
                            )
                            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                            .dst_access_mask(
                                vk::AccessFlags2::TRANSFER_WRITE | vk::AccessFlags2::SHADER_READ,
                            )
                            .buffer(device_buffer.buffer)
                            .size(device_buffer.alloc.size()),
                    ]),
                );
                self.materials_upload_countdown -= 1;
            }
        }
    }

    fn upload_textures(&mut self, command_buffer: vk::CommandBuffer) {
        unsafe {
            check!(self.uploading_textures.is_empty());
            for (&id, texture) in &self.pending_textures {
                self.ctx.device().cmd_pipeline_barrier2(
                    command_buffer,
                    &vk::DependencyInfo::default().image_memory_barriers(&[
                        vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                            .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                            .old_layout(vk::ImageLayout::UNDEFINED)
                            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                            .src_access_mask(vk::AccessFlags2::NONE)
                            .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                            .image(texture.tex_image)
                            .subresource_range(tv::default_image_subresource_range()),
                    ]),
                );
                let buffer_copy_regions = vk::BufferImageCopy::default()
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1),
                    )
                    .image_extent(texture.image_extent.into());

                self.ctx.device().cmd_copy_buffer_to_image(
                    command_buffer,
                    texture.image_buffer,
                    texture.tex_image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[buffer_copy_regions],
                );
                self.ctx.device().cmd_pipeline_barrier2(
                    command_buffer,
                    &vk::DependencyInfo::default().image_memory_barriers(&[
                        vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                            .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                            .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                            .image(texture.tex_image)
                            .subresource_range(tv::default_image_subresource_range()),
                    ]),
                );
                self.uploading_textures.insert(id, texture.clone());
            }
            self.uploaded_since_last_report += self.uploading_textures.len();
            if self.uploaded_since_last_report > 0
                && self.last_reported_upload.is_none_or(|i| {
                    i.elapsed() >= Duration::from_secs_f32(TEXTURE_STATS_INTERVAL_S)
                })
            {
                info!("uploaded {} texture(s)", self.uploaded_since_last_report);
                self.uploaded_since_last_report = 0;
                self.last_reported_upload = Some(Instant::now());
            }
            self.pending_textures.clear();
        }
    }

    pub(crate) fn pipeline_layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }

    pub fn is_texture_ready(&self, texture: TextureId) -> bool {
        self.textures.contains_key(&texture)
    }
    pub(crate) fn is_texture_id_unused(&self, texture: TextureId) -> bool {
        self.unused_texture_ids.contains(&texture)
    }
    pub(crate) fn get_internal_texture(
        &self,
        texture: TextureId,
    ) -> Option<Arc<TvInternalTexture>> {
        check_false!(self.did_vk_free.load(Ordering::Relaxed));
        self.textures
            .get(&texture)
            .or(self.uploading_textures.get(&texture))
            .or(self.pending_textures.get(&texture))
            .cloned()
    }
    pub(crate) fn free_internal_texture(
        &mut self,
        texture_id: TextureId,
    ) -> Option<Arc<TvInternalTexture>> {
        check_false!(self.did_vk_free.load(Ordering::Relaxed));
        let Some(rv) = self.textures.remove(&texture_id) else {
            error!("free_internal_texture(): removed nonexistent internal texture: {texture_id:?}");
            return None;
        };
        // Overwrite with blank texture.
        self.descriptor_image_infos[texture_id.as_usize()] = self.descriptor_image_infos[0];
        self.unused_texture_ids.insert(texture_id);
        rv.vk_free();
        Some(rv)
    }

    pub(crate) fn vk_free(&self) -> Result<()> {
        check_false!(self.did_vk_free.swap(true, Ordering::Relaxed));
        unsafe {
            self.material_device_buffer
                .vk_free()
                .context("TextureManager::vk_free()")?;
            self.material_staging_buffer
                .vk_free()
                .context("TextureManager::vk_free()")?;
            for texture in self.textures.values() {
                texture.vk_free();
            }
            self.ctx
                .device()
                .destroy_command_pool(self.command_pool, None);
            self.ctx.device().destroy_fence(self.upload_fence, None);
            self.ctx
                .device()
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.ctx
                .device()
                .destroy_descriptor_set_layout(self.desc_set_layout, None);
            self.ctx
                .device()
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.ctx.device().destroy_sampler(self.sampler, None);
        }
        Ok(())
    }
}

impl Drop for TextureManager {
    fn drop(&mut self) {
        if !self.did_vk_free.load(Ordering::Relaxed) {
            error!("leaked resource: TextureManager");
        }
    }
}
