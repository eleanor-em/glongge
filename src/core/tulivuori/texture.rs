use crate::{
    check_false,
    core::config::{FONT_SAMPLE_RATIO, MAX_MATERIAL_COUNT, MAX_TEXTURE_COUNT},
    core::prelude::{AxisAlignedExtent, Vec2},
    core::tulivuori::buffer::GenericBuffer,
    core::tulivuori::{TvWindowContext, tv},
    util::UniqueShared,
};
use anyhow::{Context, Result};
use ash::{util::Align, vk};
use std::{
    collections::BTreeMap,
    sync::atomic::{AtomicBool, AtomicU32, Ordering},
    sync::{Arc, Mutex},
};
use tracing::error;

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
    image_buffer_memory: vk::DeviceMemory,
    image_extent: vk::Extent2D,
    tex_memory: vk::DeviceMemory,
    tex_image_view: vk::ImageView,
    tex_image: vk::Image,
    data: Vec<u8>,
    ready_flag: Arc<AtomicBool>,
    did_vk_free: AtomicBool,
}

impl TvInternalTexture {
    fn new(
        ctx: Arc<TvWindowContext>,
        id: TextureId,
        image_extent: vk::Extent2D,
        format: vk::Format,
        image_data: &[u8],
        ready_flag: Arc<AtomicBool>,
    ) -> Result<Arc<Self>> {
        unsafe {
            let device_memory_properties = ctx.get_physical_device_memory_properties();
            let image_buffer = ctx.device().create_buffer(
                &vk::BufferCreateInfo {
                    size: size_of_val(image_data) as u64,
                    usage: vk::BufferUsageFlags::TRANSFER_SRC,
                    sharing_mode: vk::SharingMode::EXCLUSIVE,
                    ..Default::default()
                },
                None,
            )?;
            let image_buffer_memory_req = ctx.device().get_buffer_memory_requirements(image_buffer);

            let image_buffer_memory = ctx.device().allocate_memory(
                &vk::MemoryAllocateInfo {
                    allocation_size: image_buffer_memory_req.size,
                    memory_type_index: tv::find_memorytype_index(
                        &image_buffer_memory_req,
                        &device_memory_properties,
                        vk::MemoryPropertyFlags::HOST_VISIBLE
                            | vk::MemoryPropertyFlags::HOST_COHERENT,
                    )
                    .context("Unable to find suitable memorytype for the image buffer.")?,
                    ..Default::default()
                },
                None,
            )?;
            let image_ptr = ctx.device().map_memory(
                image_buffer_memory,
                0,
                image_buffer_memory_req.size,
                vk::MemoryMapFlags::empty(),
            )?;
            Align::new(
                image_ptr,
                align_of::<u8>() as u64,
                image_buffer_memory_req.size,
            )
            .copy_from_slice(image_data);
            ctx.device().unmap_memory(image_buffer_memory);
            ctx.device()
                .bind_buffer_memory(image_buffer, image_buffer_memory, 0)?;

            let tex_image = ctx.device().create_image(
                &vk::ImageCreateInfo {
                    image_type: vk::ImageType::TYPE_2D,
                    format,
                    extent: image_extent.into(),
                    mip_levels: 1,
                    array_layers: 1,
                    samples: vk::SampleCountFlags::TYPE_1,
                    tiling: vk::ImageTiling::OPTIMAL,
                    usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
                    sharing_mode: vk::SharingMode::EXCLUSIVE,
                    ..Default::default()
                },
                None,
            )?;
            let tex_memory_req = ctx.device().get_image_memory_requirements(tex_image);

            let texture_allocate_info = vk::MemoryAllocateInfo {
                allocation_size: tex_memory_req.size,
                memory_type_index: tv::find_memorytype_index(
                    &tex_memory_req,
                    &device_memory_properties,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                )
                .context("Unable to find suitable memory index for depth image.")?,
                ..Default::default()
            };
            let tex_memory = ctx.device().allocate_memory(&texture_allocate_info, None)?;
            ctx.device().bind_image_memory(tex_image, tex_memory, 0)?;
            let tex_image_view = ctx.device().create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .components(tv::default_component_mapping())
                    .subresource_range(tv::default_image_subresource_range())
                    .image(tex_image),
                None,
            )?;

            Ok(Arc::new(Self {
                ctx,
                id,
                image_buffer,
                image_buffer_memory,
                image_extent,
                tex_memory,
                tex_image_view,
                tex_image,
                data: image_data.to_vec(),
                ready_flag,
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

    pub fn vk_free(&self) {
        check_false!(self.did_vk_free.load(Ordering::Relaxed));
        unsafe {
            self.ctx
                .device()
                .free_memory(self.image_buffer_memory, None);
            self.ctx.device().destroy_buffer(self.image_buffer, None);
            self.ctx.device().free_memory(self.tex_memory, None);
            self.ctx
                .device()
                .destroy_image_view(self.tex_image_view, None);
            self.ctx.device().destroy_image(self.tex_image, None);
        }
        self.did_vk_free.store(true, Ordering::Relaxed);
    }
}

impl Drop for TvInternalTexture {
    fn drop(&mut self) {
        if !self.did_vk_free.load(Ordering::Relaxed) {
            error!("leaked resource: TvInternalTexture");
        }
    }
}

#[derive(Copy, Clone)]
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

pub(crate) struct TextureManager {
    ctx: Arc<TvWindowContext>,
    sampler: vk::Sampler,
    descriptor_pool: vk::DescriptorPool,
    desc_set_layout: vk::DescriptorSetLayout,
    descriptor_set: vk::DescriptorSet,
    descriptor_image_infos: Mutex<Vec<vk::DescriptorImageInfo>>,
    pipeline_layout: vk::PipelineLayout,

    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    upload_fence: vk::Fence,
    is_upload_in_progress: AtomicBool,

    next_texture_id: AtomicU32,
    pending_textures: Mutex<BTreeMap<TextureId, Arc<TvInternalTexture>>>,
    uploading_textures: Mutex<BTreeMap<TextureId, Arc<TvInternalTexture>>>,
    textures: Mutex<BTreeMap<TextureId, Arc<TvInternalTexture>>>,
    unused_texture_ids: Mutex<Vec<TextureId>>,

    next_material_buffer_index: usize,
    material_buffer: GenericBuffer<RawMaterial>,

    did_vk_free: AtomicBool,
}

impl TextureManager {
    #[allow(clippy::too_many_lines)]
    pub fn new(ctx: Arc<TvWindowContext>) -> Result<TextureManager> {
        unsafe {
            let sampler = ctx.device().create_sampler(
                &vk::SamplerCreateInfo {
                    mag_filter: vk::Filter::NEAREST,
                    min_filter: vk::Filter::LINEAR,
                    min_lod: 0.0,
                    max_lod: FONT_SAMPLE_RATIO.log2() + 1.0,
                    mipmap_mode: vk::SamplerMipmapMode::LINEAR,
                    address_mode_u: vk::SamplerAddressMode::MIRRORED_REPEAT,
                    address_mode_v: vk::SamplerAddressMode::MIRRORED_REPEAT,
                    address_mode_w: vk::SamplerAddressMode::MIRRORED_REPEAT,
                    max_anisotropy: 1.0,
                    border_color: vk::BorderColor::FLOAT_OPAQUE_WHITE,
                    compare_op: vk::CompareOp::NEVER,
                    ..Default::default()
                },
                None,
            )?;

            let descriptor_pool = ctx.device().create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
                    .pool_sizes(&[
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: MAX_MATERIAL_COUNT as u32,
                        },
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
            )?;
            let desc_set_layout = ctx.device().create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default()
                    .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                    .bindings(&[
                        vk::DescriptorSetLayoutBinding {
                            binding: 0,
                            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::VERTEX,
                            ..Default::default()
                        },
                        vk::DescriptorSetLayoutBinding {
                            binding: 1,
                            descriptor_type: vk::DescriptorType::SAMPLER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::FRAGMENT,
                            p_immutable_samplers: &raw const sampler,
                            ..Default::default()
                        },
                        vk::DescriptorSetLayoutBinding {
                            binding: 2,
                            descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                            descriptor_count: MAX_TEXTURE_COUNT as u32,
                            stage_flags: vk::ShaderStageFlags::FRAGMENT,
                            ..Default::default()
                        },
                    ])
                    .push_next(
                        &mut vk::DescriptorSetLayoutBindingFlagsCreateInfo::default()
                            .binding_flags(&[
                                vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
                                vk::DescriptorBindingFlags::empty(),
                                vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
                            ]),
                    ),
                None,
            )?;
            let descriptor_set = ctx.device().allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&[desc_set_layout]),
            )?[0];

            let pipeline_layout = ctx.device().create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&[desc_set_layout])
                    .push_constant_ranges(&[vk::PushConstantRange::default()
                        .stage_flags(vk::ShaderStageFlags::VERTEX)
                        .offset(0)
                        .size(8)]),
                None,
            )?;

            let command_pool = ctx.device().create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                    .queue_family_index(ctx.queue_family_index()),
                None,
            )?;
            let command_buffer = ctx.device().allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_buffer_count(1)
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY),
            )?[0];
            let upload_fence = ctx
                .device()
                .create_fence(&vk::FenceCreateInfo::default(), None)?;

            let material_buffer = GenericBuffer::new(
                ctx.clone(),
                2,
                MAX_MATERIAL_COUNT,
                vk::BufferUsageFlags::STORAGE_BUFFER,
            )?;

            let rv = Self {
                ctx,
                sampler,
                descriptor_pool,
                desc_set_layout,
                descriptor_set,
                descriptor_image_infos: Mutex::new(Vec::new()),
                pipeline_layout,
                command_pool,
                command_buffer,
                upload_fence,
                is_upload_in_progress: AtomicBool::new(false),
                next_texture_id: AtomicU32::new(0),
                pending_textures: Mutex::new(BTreeMap::new()),
                uploading_textures: Mutex::new(BTreeMap::new()),
                textures: Mutex::new(BTreeMap::new()),
                unused_texture_ids: Mutex::new(Vec::new()),
                next_material_buffer_index: 0,
                material_buffer,
                did_vk_free: AtomicBool::new(false),
            };

            // Set up blank initial textures.
            let blank = rv
                .create_texture(
                    vk::Extent2D {
                        width: 1,
                        height: 1,
                    },
                    vk::Format::R8G8B8A8_UNORM,
                    &[255; 4],
                    Arc::new(AtomicBool::new(true)),
                )?
                .unwrap();
            rv.upload_pending()?;

            *rv.descriptor_image_infos.lock().unwrap() = vec![
                vk::DescriptorImageInfo {
                    image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    image_view: blank.tex_image_view,
                    sampler,
                };
                MAX_TEXTURE_COUNT
            ];
            rv.ctx.device().update_descriptor_sets(
                &[vk::WriteDescriptorSet {
                    dst_set: rv.descriptor_set,
                    dst_binding: 2,
                    descriptor_count: MAX_TEXTURE_COUNT as u32,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    p_image_info: rv.descriptor_image_infos.lock().unwrap().as_ptr(),
                    ..Default::default()
                }],
                &[],
            );

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

    fn get_next_texture_id(&self) -> Option<TextureId> {
        let id = self.next_texture_id.fetch_add(1, Ordering::Relaxed);
        if id < MAX_TEXTURE_COUNT as u32 {
            Some(TextureId(id))
        } else {
            self.unused_texture_ids.lock().unwrap().pop()
        }
    }

    pub fn create_texture(
        &self,
        extent: vk::Extent2D,
        format: vk::Format,
        image_data: &[u8],
        ready_signal: Arc<AtomicBool>,
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
            ready_signal,
        )?;
        self.pending_textures
            .lock()
            .unwrap()
            .insert(id, tex.clone());
        Ok(Some(tex))
    }

    pub fn wait_for_upload(&self) -> Result<bool> {
        check_false!(self.did_vk_free.load(Ordering::Relaxed));
        if self.is_upload_in_progress.swap(false, Ordering::Relaxed) {
            unsafe {
                self.ctx
                    .device()
                    .wait_for_fences(&[self.upload_fence], true, u64::MAX)?;
                self.ctx.device().reset_fences(&[self.upload_fence])?;

                let mut uploading_textures = self.uploading_textures.lock().unwrap();
                let mut textures = self.textures.lock().unwrap();
                let mut descriptor_image_infos = self.descriptor_image_infos.lock().unwrap();
                for (&id, texture) in uploading_textures.iter() {
                    texture.ready_flag.store(true, Ordering::Relaxed);
                    textures.insert(id, texture.clone());
                    descriptor_image_infos[id.as_usize()] = vk::DescriptorImageInfo {
                        image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        image_view: texture.tex_image_view,
                        sampler: self.sampler,
                    };
                }
                uploading_textures.clear();

                self.ctx.device().update_descriptor_sets(
                    &[vk::WriteDescriptorSet {
                        dst_set: self.descriptor_set,
                        dst_binding: 2,
                        descriptor_count: MAX_TEXTURE_COUNT as u32,
                        descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                        p_image_info: descriptor_image_infos.as_ptr(),
                        ..Default::default()
                    }],
                    &[],
                );
            }
            Ok(true)
        } else {
            Ok(false)
        }
    }
    pub fn upload_pending(&self) -> Result<()> {
        check_false!(self.did_vk_free.load(Ordering::Relaxed));
        let pending_textures = self.pending_textures.lock().unwrap().clone();
        if pending_textures.is_empty() {
            return Ok(());
        }

        unsafe {
            self.ctx.device().reset_command_pool(
                self.command_pool,
                vk::CommandPoolResetFlags::RELEASE_RESOURCES,
            )?;
            self.ctx.device().begin_command_buffer(
                self.command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;

            let mut uploading_textures = BTreeMap::new();
            for (&id, texture) in &pending_textures {
                self.ctx.device().cmd_pipeline_barrier2(
                    self.command_buffer,
                    &vk::DependencyInfo::default().image_memory_barriers(&[
                        vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
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
                    self.command_buffer,
                    texture.image_buffer,
                    texture.tex_image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[buffer_copy_regions],
                );
                self.ctx.device().cmd_pipeline_barrier2(
                    self.command_buffer,
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
                uploading_textures.insert(id, texture.clone());
            }
            *self.uploading_textures.lock().unwrap() = uploading_textures;
            self.pending_textures.lock().unwrap().clear();

            self.ctx.device().end_command_buffer(self.command_buffer)?;

            self.ctx.device().queue_submit2(
                self.ctx.present_queue(),
                &[vk::SubmitInfo2::default().command_buffer_infos(&[
                    vk::CommandBufferSubmitInfo::default().command_buffer(self.command_buffer),
                ])],
                self.upload_fence,
            )?;
            self.is_upload_in_progress.store(true, Ordering::Relaxed);

            Ok(())
        }
    }

    pub fn pipeline_layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }

    pub(crate) fn initialise_materials(
        &mut self,
        material_handler: &UniqueShared<crate::resource::texture::MaterialHandler>,
    ) -> Result<()> {
        loop {
            self.upload_materials(material_handler)?;
            self.set_next_material_buffer_index();
            if self.next_material_buffer_index == 0 {
                break;
            }
        }
        Ok(())
    }
    pub fn upload_materials(
        &mut self,
        material_handler: &UniqueShared<crate::resource::texture::MaterialHandler>,
    ) -> Result<()> {
        check_false!(self.did_vk_free.load(Ordering::Relaxed));
        let mut data = vec![
            RawMaterial {
                texture_id: TextureId::default(),
                uv_top_left: Vec2::zero().into(),
                uv_bottom_right: Vec2::one().into(),
                dummy1: 0,
                dummy2: 0,
                dummy3: 0,
            };
            MAX_MATERIAL_COUNT
        ];
        let material_handler = material_handler.lock();

        for (&id, mat) in material_handler.materials() {
            let entry = &mut data[id as usize];
            entry.uv_top_left = mat
                .area
                .top_left()
                .component_wise_div(mat.texture_extent)
                .into();
            entry.uv_bottom_right = mat
                .area
                .bottom_right()
                .component_wise_div(mat.texture_extent)
                .into();
            entry.texture_id = mat.texture_id;
        }
        self.material_buffer
            .write(&data, self.next_material_buffer_index)?;
        unsafe {
            self.ctx.device().update_descriptor_sets(
                &[vk::WriteDescriptorSet {
                    dst_set: self.descriptor_set,
                    dst_binding: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    p_buffer_info: [vk::DescriptorBufferInfo::default()
                        .buffer(self.material_buffer.buffer(self.next_material_buffer_index))
                        .offset(0)
                        .range(self.material_buffer.size())]
                    .as_ptr(),
                    ..Default::default()
                }],
                &[],
            );
        }
        self.set_next_material_buffer_index();

        Ok(())
    }
    fn set_next_material_buffer_index(&mut self) {
        self.next_material_buffer_index = 1 - self.next_material_buffer_index;
    }

    pub fn is_texture_ready(&self, texture: TextureId) -> bool {
        self.textures.lock().unwrap().contains_key(&texture)
    }
    pub(crate) fn get_internal_texture(
        &self,
        texture: TextureId,
    ) -> Option<Arc<TvInternalTexture>> {
        check_false!(self.did_vk_free.load(Ordering::Relaxed));
        self.textures
            .lock()
            .unwrap()
            .get(&texture)
            .or(self.uploading_textures.lock().unwrap().get(&texture))
            .or(self.pending_textures.lock().unwrap().get(&texture))
            .cloned()
    }
    pub(crate) fn free_internal_texture(
        &mut self,
        texture_id: TextureId,
    ) -> Option<Arc<TvInternalTexture>> {
        check_false!(self.did_vk_free.load(Ordering::Relaxed));
        let rv = self.textures.lock().unwrap().remove(&texture_id)?;
        let mut descriptor_image_infos = self.descriptor_image_infos.lock().unwrap();
        // Overwrite with blank texture.
        descriptor_image_infos[texture_id.as_usize()] = descriptor_image_infos[0];
        self.unused_texture_ids.lock().unwrap().push(texture_id);
        rv.vk_free();
        Some(rv)
    }

    pub fn vk_free(&self) {
        check_false!(self.did_vk_free.load(Ordering::Relaxed));
        unsafe {
            self.ctx.device().device_wait_idle().unwrap();
            self.material_buffer.vk_free();
            for texture in self.textures.lock().unwrap().values() {
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
        self.did_vk_free.store(true, Ordering::Relaxed);
    }
}

impl Drop for TextureManager {
    fn drop(&mut self) {
        if !self.did_vk_free.load(Ordering::Relaxed) {
            error!("leaked resource: TextureManager");
        }
    }
}
