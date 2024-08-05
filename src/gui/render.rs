// Based on: https://github.com/hakolao/egui_winit_vulkano

use std::collections::BTreeMap;
use std::sync::Arc;
use anyhow::{Context, Result};
use egui::{ClippedPrimitive, Mesh};
use egui::epaint::Primitive;
use itertools::Itertools;
use vulkano::{
    buffer::Buffer,
    buffer::BufferContents,
    buffer::BufferCreateInfo,
    buffer::BufferUsage,
    buffer::Subbuffer,
    command_buffer::AutoCommandBufferBuilder,
    command_buffer::PrimaryAutoCommandBuffer,
    descriptor_set::layout::DescriptorSetLayoutCreateFlags,
    descriptor_set::PersistentDescriptorSet,
    descriptor_set::WriteDescriptorSet,
    image::sampler::SamplerCreateInfo,
    memory::allocator::AllocationCreateInfo,
    memory::allocator::MemoryTypeFilter,
    pipeline::graphics::viewport::ViewportState,
    pipeline::graphics::vertex_input::VertexDefinition,
    pipeline::graphics::rasterization::RasterizationState,
    pipeline::graphics::multisample::MultisampleState,
    pipeline::graphics::input_assembly::InputAssemblyState,
    pipeline::graphics::GraphicsPipelineCreateInfo,
    pipeline::graphics::color_blend::AttachmentBlend,
    pipeline::graphics::color_blend::ColorBlendAttachmentState,
    pipeline::graphics::color_blend::ColorBlendState,
    pipeline::GraphicsPipeline,
    pipeline::Pipeline,
    pipeline::PipelineBindPoint,
    pipeline::PipelineLayout,
    pipeline::PipelineShaderStageCreateInfo,
    pipeline::layout::PipelineDescriptorSetLayoutCreateInfo,
    render_pass::Subpass,
    shader::ShaderModule,
    Validated,
    image::sampler::Sampler,
    image::view::ImageView
};
use vulkano::command_buffer::{BufferImageCopy, CommandBufferUsage, CopyBufferToImageInfo, PrimaryCommandBufferAbstract};
use vulkano::format::Format;
use vulkano::image::{Image, ImageAspects, ImageCreateInfo, ImageLayout, ImageSubresourceLayers, ImageType, ImageUsage};
use vulkano::image::sampler::{Filter, SamplerAddressMode, SamplerMipmapMode};
use vulkano::image::view::ImageViewCreateInfo;
use vulkano::memory::allocator::DeviceLayout;
use vulkano::memory::DeviceAlignment;
use vulkano::sync::GpuFuture;
use crate::{
    core::{
        prelude::*,
        util::UniqueShared,
        vk::{AdjustedViewport, VulkanoContext}
    },
    shader::VkVertex,
};

pub mod vs {
    vulkano_shaders::shader!{
        ty: "vertex",
        src: r"
#version 460

layout(push_constant) uniform VertPC {
    vec2 u_screen_size;
};
layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec2 a_tc;
layout(location = 2) in vec4 a_srgba; // 0-255 sRGB
layout(location = 0) out vec4 v_rgba_in_gamma;
layout(location = 1) out vec2 v_tc;

void main() {
    gl_Position = vec4(
                      2.0 * a_pos.x / u_screen_size.x - 1.0,
                      2.0 * a_pos.y / u_screen_size.y - 1.0,
                      0.0,
                      1.0);
    v_rgba_in_gamma = a_srgba;
    v_tc = a_tc;
}
        ",
    }
}

pub mod fs {
    vulkano_shaders::shader!{
        ty: "fragment",
        src: r"
#version 460

layout(binding = 0, set = 0) uniform sampler2D font_texture;

layout(location = 0) in vec4 v_rgba_in_gamma;
layout(location = 1) in vec2 v_tc;
layout(location = 0) out vec4 f_color;

// 0-1 sRGB gamma  from  0-1 linear
vec3 srgb_gamma_from_linear(vec3 rgb) {
    bvec3 cutoff = lessThan(rgb, vec3(0.0031308));
    vec3 lower = rgb * vec3(12.92);
    vec3 higher = vec3(1.055) * pow(rgb, vec3(1.0 / 2.4)) - vec3(0.055);
    return mix(higher, lower, vec3(cutoff));
}

// 0-1 sRGBA gamma  from  0-1 linear
vec4 srgba_gamma_from_linear(vec4 rgba) {
    return vec4(srgb_gamma_from_linear(rgba.rgb), rgba.a);
}

void main() {
    vec4 texture_in_gamma = srgba_gamma_from_linear(texture(font_texture, v_tc));

    // We multiply the colors in gamma space, because that's the only way to get text to look right.
    vec4 frag_color_gamma = v_rgba_in_gamma * texture_in_gamma;

    f_color = frag_color_gamma;
}
        ",
    }
}
type EguiVertexBuffer = Subbuffer<[egui::epaint::Vertex]>;
type EguiIndexBuffer = Subbuffer<[u32]>;
#[repr(C)]
#[derive(BufferContents, VkVertex)]
pub struct EguiVertex {
    #[format(R32G32_SFLOAT)]
    pub a_pos: [f32; 2],
    #[format(R32G32_SFLOAT)]
    pub a_tc: [f32; 2],
    #[format(R8G8B8A8_UNORM)]
    pub a_srgba: [u8; 4],
}

fn image_size_bytes(delta: &egui::epaint::ImageDelta) -> usize {
    match &delta.image {
        egui::ImageData::Color(c) => {
            // Always four bytes per pixel for sRGBA
            c.width() * c.height() * 4
        }
        egui::ImageData::Font(f) => {
            f.width() * f.height() * 4
        }
    }
}

pub struct GuiRenderer {
    vk_ctx: VulkanoContext,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    viewport: UniqueShared<AdjustedViewport>,
    pipeline: Option<Arc<GraphicsPipeline>>,

    font_sampler: Arc<Sampler>,
    texture_desc_sets: BTreeMap<egui::TextureId, Arc<PersistentDescriptorSet>>,
    texture_images: BTreeMap<egui::TextureId, Arc<ImageView>>,
    next_native_tex_id: u64,
}

impl GuiRenderer {
    pub fn new(
        vk_ctx: VulkanoContext,
        viewport: UniqueShared<AdjustedViewport>,
    ) -> Result<UniqueShared<Self>> {
        let device = vk_ctx.device();
        let font_sampler = Sampler::new(vk_ctx.device(), SamplerCreateInfo {
            mag_filter: Filter::Linear,
            min_filter: Filter::Linear,
            address_mode: [SamplerAddressMode::ClampToEdge; 3],
            mipmap_mode: SamplerMipmapMode::Linear,
            ..Default::default()
        })?;
        Ok(UniqueShared::new(Self {
            vk_ctx,
            vs: vs::load(device.clone()).context("failed to create shader module")?,
            fs: fs::load(device).context("failed to create shader module")?,
            viewport,
            pipeline: None,
            font_sampler,
            texture_desc_sets: BTreeMap::new(),
            texture_images: BTreeMap::new(),
            next_native_tex_id: 0,
        }))
    }
    fn get_or_create_pipeline(&mut self) -> Result<Arc<GraphicsPipeline>> {
        match self.pipeline.clone() {
            None => {
                let vs = self.vs.entry_point("main")
                    .context("vertex shader: entry point missing")?;
                let fs = self.fs.entry_point("main")
                    .context("fragment shader: entry point missing")?;
                let vertex_input_state =
                    EguiVertex::per_vertex().definition(&vs.info().input_interface)?;
                let stages = [
                    PipelineShaderStageCreateInfo::new(vs),
                    PipelineShaderStageCreateInfo::new(fs),
                ];
                let mut create_info = PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages);
                for layout in &mut create_info.set_layouts {
                    layout.flags |= DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL;
                }
                let layout = PipelineLayout::new(
                    self.vk_ctx.device(),
                    create_info.into_pipeline_layout_create_info(self.vk_ctx.device())?,
                ).map_err(Validated::unwrap)?;
                let subpass = Subpass::from(self.vk_ctx.render_pass(), 0).context("failed to create subpass")?;

                let pipeline = GraphicsPipeline::new(
                    self.vk_ctx.device(),
                    /* cache= */ None,
                    GraphicsPipelineCreateInfo {
                        stages: stages.into_iter().collect(),
                        vertex_input_state: Some(vertex_input_state),
                        input_assembly_state: Some(InputAssemblyState::default()),
                        viewport_state: Some(ViewportState {
                            viewports: [self.viewport.get().inner()].into_iter().collect(),
                            ..Default::default()
                        }),
                        rasterization_state: Some(RasterizationState::default()),
                        multisample_state: Some(MultisampleState::default()),
                        color_blend_state: Some(ColorBlendState::with_attachment_states(
                            subpass.num_color_attachments(),
                            ColorBlendAttachmentState {
                                blend: Some(AttachmentBlend::alpha()),
                                ..Default::default()
                            },
                        )),
                        subpass: Some(subpass.into()),
                        ..GraphicsPipelineCreateInfo::layout(layout)
                    })?;
                self.pipeline = Some(pipeline.clone());
                Ok(pipeline)
            },
            Some(pipeline) => Ok(pipeline)
        }
    }

    pub fn register_image(
        &mut self,
        image: Arc<ImageView>,
        sampler_create_info: SamplerCreateInfo,
    ) -> Result<egui::TextureId> {
        let pipeline = self.get_or_create_pipeline()?;
        let layout = pipeline.layout().set_layouts().first()
            .context("pipeline layout missing descriptor set layout")?;
        let sampler = Sampler::new(self.vk_ctx.device(), sampler_create_info)?;
        let desc_set = PersistentDescriptorSet::new(
            &self.vk_ctx.descriptor_set_allocator(),
            layout.clone(),
            [WriteDescriptorSet::image_view_sampler(0, image.clone(), sampler)],
            [],
        )?;
        let id = egui::TextureId::User(self.next_native_tex_id);
        self.next_native_tex_id += 1;
        self.texture_desc_sets.insert(id, desc_set);
        self.texture_images.insert(id, image);
        Ok(id)
    }
    pub fn unregister_image(&mut self, texture_id: egui::TextureId) {
        self.texture_desc_sets.remove(&texture_id);
        self.texture_images.remove(&texture_id);
    }
    /// Write a single texture delta using the provided staging region and commandbuffer
    fn update_texture_within(
        &mut self,
        id: egui::TextureId,
        delta: &egui::epaint::ImageDelta,
        stage: Subbuffer<[u8]>,
        mapped_stage: &mut [u8],
        cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) -> Result<()> {
        // egui has a small oversight: Color32::to_array() takes &self not self, so we can't
        // eliminate the "redundant" closure.
        #[allow(clippy::redundant_closure_for_method_calls)]
        match &delta.image {
            egui::ImageData::Color(image) => {
                assert_eq!(
                    image.width() * image.height(),
                    image.pixels.len(),
                    "Mismatch between texture size and texel count"
                );
                let bytes = image.pixels.iter().flat_map(|color| color.to_array());
                mapped_stage.iter_mut().zip(bytes).for_each(|(into, from)| *into = from);
            }
            egui::ImageData::Font(image) => {
                let bytes = image.srgba_pixels(None).flat_map(|color| color.to_array());
                mapped_stage.iter_mut().zip(bytes).for_each(|(into, from)| *into = from);
            }
        };

        // Copy texture data to existing image if delta pos exists (e.g. font changed)
        if let Some(pos) = delta.pos {
            let existing_image = self.texture_images.get(&id)
                .context("attempt to write into non-existing image")?;
            check_eq!(existing_image.format(), Format::R8G8B8A8_SRGB);

            // Defer upload of data
            cbb.copy_buffer_to_image(CopyBufferToImageInfo {
                regions: [BufferImageCopy {
                    // Buffer offsets are derived
                    image_offset: [pos[0] as u32, pos[1] as u32, 0],
                    image_extent: [delta.image.width() as u32, delta.image.height() as u32, 1],
                    // Always use the whole image (no arrays or mips are performed)
                    image_subresource: ImageSubresourceLayers {
                        aspects: ImageAspects::COLOR,
                        mip_level: 0,
                        array_layers: 0..1,
                    },
                    ..Default::default()
                }]
                    .into(),
                ..CopyBufferToImageInfo::buffer_image(stage, existing_image.image().clone())
            })?;
        } else {
            // Otherwise save the newly created image
            let img = {
                let extent = [delta.image.width() as u32, delta.image.height() as u32, 1];
                Image::new(
                    self.vk_ctx.memory_allocator(),
                    ImageCreateInfo {
                        image_type: ImageType::Dim2d,
                        format: Format::R8G8B8A8_SRGB,
                        extent,
                        usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                        initial_layout: ImageLayout::Undefined,
                        ..Default::default()
                    },
                    AllocationCreateInfo::default(),
                )
                    .unwrap()
            };
            cbb.copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(stage, img.clone()))?;
            let view = ImageView::new(img.clone(), ImageViewCreateInfo::from_image(&img))?;
            let pipeline = self.get_or_create_pipeline()?;
            let layout = pipeline.layout().set_layouts().first()
                .context("no descriptor sets in pipeline layout")?;
            let desc_set = PersistentDescriptorSet::new(
                &self.vk_ctx.descriptor_set_allocator(),
                layout.clone(),
                [WriteDescriptorSet::image_view_sampler(0, view.clone(), self.font_sampler.clone())],
                [],
            )?;
            self.texture_desc_sets.insert(id, desc_set);
            self.texture_images.insert(id, view);
        };
        Ok(())
    }
    /// Write the entire texture delta for this frame.
    pub fn update_textures(&mut self, sets: &[(egui::TextureId, egui::epaint::ImageDelta)]) -> Result<()> {
        let total_size_bytes =
            sets.iter().map(|(_, set)| image_size_bytes(set)).sum::<usize>() * 4;
        let total_size_bytes = u64::try_from(total_size_bytes)?;
        let Ok(total_size_bytes) = vulkano::NonZeroDeviceSize::try_from(total_size_bytes) else {
            // Nothing to upload!
            return Ok(());
        };
        let buffer = Buffer::new(
            self.vk_ctx.memory_allocator(),
            BufferCreateInfo { usage: BufferUsage::TRANSFER_SRC, ..Default::default() },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            DeviceLayout::new(total_size_bytes, DeviceAlignment::MIN)
                .context("overflowed total_size_bytes: {total_size_bytes}")?,
        )?;
        let buffer = Subbuffer::new(buffer);

        // Shared command buffer for every upload in this batch.
        let mut cbb = AutoCommandBufferBuilder::primary(
            self.vk_ctx.command_buffer_allocator(),
            self.vk_ctx.queue().queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;

        {
            let mut writer = buffer.write()?;
            let mut past_buffer_end = 0usize;

            for (id, delta) in sets {
                let image_size_bytes = image_size_bytes(delta);
                let range = past_buffer_end..(image_size_bytes + past_buffer_end);
                past_buffer_end += image_size_bytes;

                let stage = buffer.clone().slice(range.start as u64..range.end as u64);
                let mapped_stage = &mut writer[range];

                self.update_texture_within(*id, delta, stage, mapped_stage, &mut cbb)?;
            }
        }

        let command_buffer = cbb.build()?;
        command_buffer
            .execute(self.vk_ctx.queue())?
            .then_signal_fence_and_flush()?
            .wait(None)?;
        Ok(())
    }

    fn create_vertex_index_buffers(&mut self, meshes: &[Mesh]) -> Result<(EguiVertexBuffer, EguiIndexBuffer)> {
        let vertex_buffer = Buffer::from_iter(
            self.vk_ctx.memory_allocator(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            meshes.iter().flat_map(|m| m.vertices.clone()).collect_vec()
        ).map_err(Validated::unwrap)?;
        let index_buffer = Buffer::from_iter(
            self.vk_ctx.memory_allocator(),
            BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            meshes.iter().flat_map(|m| m.indices.clone()).collect_vec()
        ).map_err(Validated::unwrap)?;
        Ok((vertex_buffer, index_buffer))
    }
    pub fn build_render_pass(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        primitives: &[ClippedPrimitive]
    ) -> Result<()> {
        let meshes = primitives.iter().filter_map(|mesh| match &mesh.primitive {
            Primitive::Mesh(m) => Some(m),
            Primitive::Callback(_) => None,
        }).cloned().collect_vec();
        if meshes.is_empty() {
            return Ok(());
        }

        let push_constants = {
            let viewport = self.viewport.get();
            let viewport_extent = viewport.aa_extent();
            let scale_factor = viewport.scale_factor() / viewport.gui_scale_factor();
            vs::VertPC {
                u_screen_size: (viewport_extent * scale_factor).as_f32_lossy()
            }
        };
        let (vertex_buffer, index_buffer) = self.create_vertex_index_buffers(&meshes)?;
        let mut vertex_cursor = 0;
        let mut index_cursor = 0;
        for mesh in meshes {
            if let Err(e) = self.prepare_mesh_draw(
                builder,
                push_constants,
                vertex_buffer.clone(),
                index_buffer.clone(),
                &mesh
            ) {
                error!("{e:?}");
            } else {
                builder.draw_indexed(
                    mesh.indices.len() as u32,
                    1,
                    index_cursor,
                    vertex_cursor,
                    0
                )?;
            }
            index_cursor += mesh.indices.len() as u32;
            vertex_cursor += i32::try_from(mesh.vertices.len())
                .with_context(|| format!("overflowed vertex_cursor: {}", mesh.vertices.len()))?;
        }
        Ok(())
    }

    fn prepare_mesh_draw(&mut self,
                         builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
                         push_constants: vs::VertPC,
                         vertex_buffer: EguiVertexBuffer,
                         index_buffer: EguiIndexBuffer,
                         mesh: &Mesh
    ) -> Result<()> {
        let pipeline = self.get_or_create_pipeline()?;
        let desc_set = if let Some(desc) = self.texture_desc_sets.get(&mesh.texture_id) {
            desc.clone()
        } else {
            bail!("texture no longer exists: {:?}", mesh.texture_id);
        };
        let layout = pipeline.layout().clone();
        builder
            .bind_pipeline_graphics(pipeline)?
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                layout.clone(),
                0,
                desc_set,
            )?
            .bind_vertex_buffers(0, vertex_buffer)?
            .bind_index_buffer(index_buffer)?
            .push_constants(layout, 0, push_constants)?;
        Ok(())
    }
}
