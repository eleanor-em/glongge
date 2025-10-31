use crate::core::config::{FONT_SAMPLE_RATIO, MAX_MATERIAL_COUNT, MAX_TEXTURE_COUNT};
use crate::core::prelude::{AxisAlignedExtent, Vec2};
use crate::core::tulivuori;
use crate::core::tulivuori::TvWindowContext;
use crate::core::tulivuori::buffer::GenericBuffer;
use crate::util::UniqueShared;
use anyhow::{Context, Result};
use ash::util::Align;
use ash::vk;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

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
}

impl TvInternalTexture {
    fn new(
        ctx: Arc<TvWindowContext>,
        id: TextureId,
        image_extent: vk::Extent2D,
        image_data: &[u8],
        ready_flag: Arc<AtomicBool>,
    ) -> Result<Arc<Self>> {
        unsafe {
            let device_memory_properties = ctx.get_physical_device_memory_properties();
            let image_buffer = ctx.device().create_buffer(
                &vk::BufferCreateInfo {
                    size: std::mem::size_of_val(image_data) as u64,
                    usage: vk::BufferUsageFlags::TRANSFER_SRC,
                    sharing_mode: vk::SharingMode::EXCLUSIVE,
                    ..Default::default()
                },
                None,
            )?;
            let image_buffer_memory_req = ctx.device().get_buffer_memory_requirements(image_buffer);
            let image_buffer_memory_index = tulivuori::find_memorytype_index(
                &image_buffer_memory_req,
                &device_memory_properties,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .context("Unable to find suitable memorytype for the image buffer.")?;

            let image_buffer_memory = ctx.device().allocate_memory(
                &vk::MemoryAllocateInfo {
                    allocation_size: image_buffer_memory_req.size,
                    memory_type_index: image_buffer_memory_index,
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
            let mut image_slice = Align::new(
                image_ptr,
                align_of::<u8>() as u64,
                image_buffer_memory_req.size,
            );
            image_slice.copy_from_slice(image_data);
            ctx.device().unmap_memory(image_buffer_memory);
            ctx.device()
                .bind_buffer_memory(image_buffer, image_buffer_memory, 0)?;

            let texture_create_info = vk::ImageCreateInfo {
                image_type: vk::ImageType::TYPE_2D,
                format: vk::Format::R8G8B8A8_UNORM,
                extent: image_extent.into(),
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                tiling: vk::ImageTiling::OPTIMAL,
                usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                ..Default::default()
            };
            let tex_image = ctx.device().create_image(&texture_create_info, None)?;
            let tex_memory_req = ctx.device().get_image_memory_requirements(tex_image);
            let tex_memory_index = tulivuori::find_memorytype_index(
                &tex_memory_req,
                &device_memory_properties,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .context("Unable to find suitable memory index for depth image.")?;

            let texture_allocate_info = vk::MemoryAllocateInfo {
                allocation_size: tex_memory_req.size,
                memory_type_index: tex_memory_index,
                ..Default::default()
            };
            let tex_memory = ctx.device().allocate_memory(&texture_allocate_info, None)?;
            ctx.device().bind_image_memory(tex_image, tex_memory, 0)?;
            let tex_image_view_info = vk::ImageViewCreateInfo {
                view_type: vk::ImageViewType::TYPE_2D,
                format: texture_create_info.format,
                components: vk::ComponentMapping {
                    r: vk::ComponentSwizzle::R,
                    g: vk::ComponentSwizzle::G,
                    b: vk::ComponentSwizzle::B,
                    a: vk::ComponentSwizzle::A,
                },
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    level_count: 1,
                    layer_count: 1,
                    ..Default::default()
                },
                image: tex_image,
                ..Default::default()
            };
            let tex_image_view = ctx.device().create_image_view(&tex_image_view_info, None)?;

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
}

impl Drop for TvInternalTexture {
    fn drop(&mut self) {
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
                            descriptor_count: MAX_MATERIAL_COUNT as u32,
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
            };

            // Set up blank initial textures.
            let blank = rv
                .create_texture(
                    vk::Extent2D {
                        width: 1,
                        height: 1,
                    },
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
        image_data: &[u8],
        ready_signal: Arc<AtomicBool>,
    ) -> Result<Option<Arc<TvInternalTexture>>> {
        let Some(id) = self.get_next_texture_id() else {
            return Ok(None);
        };
        let tex = TvInternalTexture::new(self.ctx.clone(), id, extent, image_data, ready_signal)?;
        self.pending_textures
            .lock()
            .unwrap()
            .insert(id, tex.clone());
        Ok(Some(tex))
    }

    pub fn wait_for_upload(&self) -> Result<bool> {
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
                let texture_barrier = vk::ImageMemoryBarrier {
                    dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                    new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    image: texture.tex_image,
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        level_count: 1,
                        layer_count: 1,
                        ..Default::default()
                    },
                    ..Default::default()
                };
                self.ctx.device().cmd_pipeline_barrier(
                    self.command_buffer,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[texture_barrier],
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
                let texture_barrier_end = vk::ImageMemoryBarrier {
                    src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                    dst_access_mask: vk::AccessFlags::SHADER_READ,
                    old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    image: texture.tex_image,
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        level_count: 1,
                        layer_count: 1,
                        ..Default::default()
                    },
                    ..Default::default()
                };
                self.ctx.device().cmd_pipeline_barrier(
                    self.command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[texture_barrier_end],
                );
                uploading_textures.insert(id, texture.clone());
            }
            *self.uploading_textures.lock().unwrap() = uploading_textures;
            self.pending_textures.lock().unwrap().clear();

            self.ctx.device().end_command_buffer(self.command_buffer)?;

            self.ctx.device().queue_submit(
                self.ctx.present_queue(),
                &[vk::SubmitInfo::default().command_buffers(&[self.command_buffer])],
                self.upload_fence,
            )?;
            self.is_upload_in_progress.store(true, Ordering::Relaxed);

            Ok(())
        }
    }

    pub fn pipeline_layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }

    pub fn upload_materials(
        &mut self,
        material_handler: &UniqueShared<crate::resource::texture::MaterialHandler>,
    ) -> Result<()> {
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
            let mut buffer_infos = Vec::new();
            for i in 0..MAX_MATERIAL_COUNT {
                buffer_infos.push(
                    vk::DescriptorBufferInfo::default()
                        .buffer(self.material_buffer.buffer(self.next_material_buffer_index))
                        .offset((i * size_of::<RawMaterial>()) as vk::DeviceSize)
                        .range(size_of::<RawMaterial>() as vk::DeviceSize),
                );
            }

            self.ctx.device().update_descriptor_sets(
                &[vk::WriteDescriptorSet {
                    dst_set: self.descriptor_set,
                    dst_binding: 0,
                    descriptor_count: MAX_MATERIAL_COUNT as u32,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    p_buffer_info: buffer_infos.as_ptr(),
                    ..Default::default()
                }],
                &[],
            );
        }

        self.next_material_buffer_index = 1 - self.next_material_buffer_index;

        Ok(())
    }

    pub fn is_texture_ready(&self, texture: TextureId) -> bool {
        self.textures.lock().unwrap().contains_key(&texture)
    }
    pub(crate) fn get_internal_texture(
        &self,
        texture: TextureId,
    ) -> Option<Arc<TvInternalTexture>> {
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
        let rv = self.textures.lock().unwrap().remove(&texture_id)?;
        let mut descriptor_image_infos = self.descriptor_image_infos.lock().unwrap();
        // Overwrite with blank texture.
        descriptor_image_infos[texture_id.as_usize()] = descriptor_image_infos[0];
        self.unused_texture_ids.lock().unwrap().push(texture_id);
        Some(rv)
    }
}

impl Drop for TextureManager {
    fn drop(&mut self) {
        unsafe {
            self.ctx.device().device_wait_idle().unwrap();
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
    }
}
