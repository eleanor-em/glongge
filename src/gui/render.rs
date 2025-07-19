// Based on: https://github.com/hakolao/egui_winit_vulkano
// TODO: refactor. Maybe take from upstream above?

use crate::core::vk::vk_ctx::VulkanoContext;
use crate::{
    core::{prelude::*, vk::AdjustedViewport},
    shader::VkVertex,
    util::UniqueShared,
};
use anyhow::{Context, Result};
use egui::epaint::Primitive;
use egui::{ClippedPrimitive, Color32, Mesh, TextureId};
use itertools::Itertools;
use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::Arc;
use vulkano::buffer::IndexType;
use vulkano::command_buffer::{RenderingAttachmentInfo, RenderingInfo};
use vulkano::descriptor_set::sys::RawDescriptorSet;
use vulkano::format::Format;
use vulkano::image::sampler::{Filter, SamplerAddressMode, SamplerMipmapMode};
use vulkano::image::view::ImageViewCreateInfo;
use vulkano::image::{
    Image, ImageAspects, ImageCreateInfo, ImageSubresourceLayers, ImageType, ImageUsage,
};
use vulkano::memory::DeviceAlignment;
use vulkano::memory::allocator::DeviceLayout;
use vulkano::pipeline::graphics::subpass::PipelineRenderingCreateInfo;
use vulkano::render_pass::AttachmentLoadOp::Load;
use vulkano::render_pass::AttachmentStoreOp::Store;
use vulkano::swapchain::Swapchain;
use vulkano::{
    DeviceSize, NonZeroDeviceSize, Validated, buffer::Buffer, buffer::BufferContents,
    buffer::BufferCreateInfo, buffer::BufferUsage, descriptor_set::WriteDescriptorSet,
    descriptor_set::layout::DescriptorSetLayoutCreateFlags, image::sampler::Sampler,
    image::sampler::SamplerCreateInfo, image::view::ImageView,
    memory::allocator::AllocationCreateInfo, memory::allocator::MemoryTypeFilter,
    pipeline::GraphicsPipeline, pipeline::Pipeline, pipeline::PipelineBindPoint,
    pipeline::PipelineLayout, pipeline::PipelineShaderStageCreateInfo,
    pipeline::graphics::GraphicsPipelineCreateInfo,
    pipeline::graphics::color_blend::AttachmentBlend,
    pipeline::graphics::color_blend::ColorBlendAttachmentState,
    pipeline::graphics::color_blend::ColorBlendState,
    pipeline::graphics::input_assembly::InputAssemblyState,
    pipeline::graphics::multisample::MultisampleState,
    pipeline::graphics::rasterization::RasterizationState,
    pipeline::graphics::vertex_input::VertexDefinition,
    pipeline::graphics::viewport::ViewportState,
    pipeline::layout::PipelineDescriptorSetLayoutCreateInfo, shader::ShaderModule,
};
use vulkano_taskgraph::command_buffer::{
    BufferImageCopy, CopyBufferToImageInfo, RecordingCommandBuffer,
};
use vulkano_taskgraph::graph::{NodeId, TaskGraph};
use vulkano_taskgraph::resource::{AccessTypes, ImageLayoutType};
use vulkano_taskgraph::{Id, QueueFamilyType, Task, TaskContext, TaskResult};

pub mod vs {
    vulkano_shaders::shader! {
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
    vulkano_shaders::shader! {
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
#[repr(C)]
#[derive(BufferContents, VkVertex, Clone, Copy)]
pub struct EguiVertex {
    #[format(R32G32_SFLOAT)]
    pub a_pos: [f32; 2],
    #[format(R32G32_SFLOAT)]
    pub a_tc: [f32; 2],
    #[format(R8G8B8A8_UNORM)]
    pub a_srgba: [u8; 4],
}

#[derive(Clone)]
struct GuiVertexIndexBuffers {
    ctx: VulkanoContext,
    vertices: Id<Buffer>,
    indices: Id<Buffer>,
    last_image_idx: usize,
    last_vertex_count: usize,
    last_index_count: usize,
    num_sets: usize,
    elements_per_set: usize,
    is_dirty: bool,
}

fn image_size_bytes(delta: &egui::epaint::ImageDelta) -> u64 {
    match &delta.image {
        egui::ImageData::Color(c) => {
            // Always four bytes per pixel for sRGBA
            (c.width() * c.height() * 4) as u64
        }
        egui::ImageData::Font(f) => (f.width() * f.height() * 4) as u64,
    }
}

impl GuiVertexIndexBuffers {
    fn new(ctx: VulkanoContext, size: usize) -> Result<Self> {
        let num_sets = ctx.image_count();
        let vertices = Self::create_vertex_buffer(
            &ctx,
            (size * size_of::<egui::epaint::Vertex>() * num_sets) as DeviceSize,
        )?;
        let indices =
            Self::create_index_buffer(&ctx, (size * size_of::<u32>() * num_sets) as DeviceSize)?;
        let rv = Self {
            ctx,
            vertices,
            indices,
            last_image_idx: 0,
            last_vertex_count: 0,
            last_index_count: 0,
            num_sets,
            elements_per_set: size,
            is_dirty: true,
        };
        info!(
            "created GUI vertex/index buffer: {} KiB",
            (rv.vertex_size_in_bytes() + rv.index_size_in_bytes()) / 1024
        );
        Ok(rv)
    }

    fn vertex_size_in_bytes(&self) -> usize {
        self.ctx
            .resources()
            .buffer(self.vertices)
            .unwrap()
            .buffer()
            .size() as usize
    }
    fn index_size_in_bytes(&self) -> usize {
        self.ctx
            .resources()
            .buffer(self.indices)
            .unwrap()
            .buffer()
            .size() as usize
    }

    fn realloc(&mut self) -> Result<()> {
        let size = self.vertex_size_in_bytes();
        if size / 1024 / 1024 == 0 {
            warn!(
                "reallocating vertex buffer: {} KiB -> {} KiB",
                size / 1024,
                size * 2 / 1024
            );
        } else {
            warn!(
                "reallocating vertex buffer: {} MiB -> {} MiB",
                size / 1024 / 1024,
                size * 2 / 1024 / 1024
            );
        }
        // Just double the size.
        self.vertices =
            Self::create_vertex_buffer(&self.ctx, (self.vertex_size_in_bytes() * 2) as DeviceSize)?;
        self.indices =
            Self::create_index_buffer(&self.ctx, (self.index_size_in_bytes() * 2) as DeviceSize)?;
        self.elements_per_set *= 2;
        self.is_dirty = true;
        Ok(())
    }

    fn create_vertex_buffer(ctx: &VulkanoContext, size: DeviceSize) -> Result<Id<Buffer>> {
        Ok(ctx
            .resources()
            .create_buffer(
                BufferCreateInfo {
                    usage: BufferUsage::VERTEX_BUFFER | BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                DeviceLayout::new(
                    NonZeroDeviceSize::new(size).unwrap(),
                    DeviceAlignment::of::<egui::epaint::Vertex>(),
                )
                .context("failed to create vertex buffer of size {size}")?,
            )
            .map_err(Validated::unwrap)?)
    }
    fn create_index_buffer(ctx: &VulkanoContext, size: DeviceSize) -> Result<Id<Buffer>> {
        Ok(ctx
            .resources()
            .create_buffer(
                BufferCreateInfo {
                    usage: BufferUsage::INDEX_BUFFER | BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                DeviceLayout::new(
                    NonZeroDeviceSize::new(size).unwrap(),
                    DeviceAlignment::of::<u32>(),
                )
                .context("failed to create index buffer of size {size}")?,
            )
            .map_err(Validated::unwrap)?)
    }

    fn write(
        &mut self,
        image_idx: usize,
        vertices: &[egui::epaint::Vertex],
        indices: &[u32],
        tcx: &mut TaskContext,
    ) -> Result<()> {
        self.last_image_idx = image_idx;
        self.last_vertex_count = vertices.len();
        self.last_index_count = indices.len();
        if vertices.is_empty() {
            return Ok(());
        }

        // Reallocate if needed:
        while vertices.len() * self.num_sets * size_of::<egui::epaint::Vertex>()
            > self.vertex_size_in_bytes()
        {
            self.realloc()?;
        }
        while indices.len() * self.num_sets * size_of::<u32>() > self.index_size_in_bytes() {
            self.realloc()?;
        }
        if self.is_dirty {
            return Ok(());
        }

        // Write buffers:
        let start = (self.last_image_idx
            * self.elements_per_set
            * size_of::<egui::epaint::Vertex>()) as DeviceSize;
        let end = start + size_of_val(vertices) as DeviceSize;
        tcx.write_buffer::<[egui::epaint::Vertex]>(self.vertices, start..end)?
            .copy_from_slice(vertices);
        let start = (self.last_image_idx * self.elements_per_set * size_of::<u32>()) as DeviceSize;
        let end = start + size_of_val(indices) as DeviceSize;
        tcx.write_buffer::<[u32]>(self.indices, start..end)?
            .copy_from_slice(indices);
        Ok(())
    }

    fn bind(&self, cbf: &mut RecordingCommandBuffer) -> Result<()> {
        let start = (self.last_image_idx
            * self.elements_per_set
            * size_of::<egui::epaint::Vertex>()) as DeviceSize;
        let end =
            start + (self.last_vertex_count * size_of::<egui::epaint::Vertex>()) as DeviceSize;
        unsafe {
            cbf.bind_vertex_buffers(
                0,
                &[self.vertices],
                &[start],
                &[DeviceSize::from(end - start)],
                &[],
            )?;

            let start =
                (self.last_image_idx * self.elements_per_set * size_of::<u32>()) as DeviceSize;
            let end = start + (self.last_index_count * size_of::<u32>()) as DeviceSize;
            cbf.bind_index_buffer(
                self.indices,
                start,
                DeviceSize::from(end - start),
                IndexType::U32,
            )?;
        }
        Ok(())
    }
}

#[derive(Clone)]
pub(crate) struct GuiRenderer {
    vk_ctx: VulkanoContext,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    viewport: UniqueShared<AdjustedViewport>,
    pipeline: UniqueShared<Option<Arc<GraphicsPipeline>>>,

    texture_desc_sets:
        UniqueShared<BTreeMap<egui::TextureId, (RawDescriptorSet, Vec<WriteDescriptorSet>)>>,
    texture_images: UniqueShared<BTreeMap<egui::TextureId, Id<Image>>>,
    texture_images_dirty: UniqueShared<bool>,

    next_frame: UniqueShared<Option<Vec<Mesh>>>,
    last_nonempty_frame: UniqueShared<Vec<Mesh>>,
    pub(crate) primitives: UniqueShared<Option<Vec<ClippedPrimitive>>>,
    pub(crate) gui_enabled: UniqueShared<bool>,

    staging_buffer: Id<Buffer>,
    draw_buffer: UniqueShared<GuiVertexIndexBuffers>,
}

impl GuiRenderer {
    const MAX_STAGING_SIZE_BYTES: DeviceSize = 10 * 1024 * 1024;

    pub(crate) fn new(
        vk_ctx: VulkanoContext,
        viewport: UniqueShared<AdjustedViewport>,
    ) -> Result<Self> {
        let device = vk_ctx.device();
        let staging_buffer = vk_ctx.resources().create_buffer(
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            DeviceLayout::new(
                NonZeroDeviceSize::new(Self::MAX_STAGING_SIZE_BYTES).unwrap(),
                DeviceAlignment::MIN,
            )
            .context("too large size for min alignment: {Self::MAX_STAGING_SIZE_BYTES}")?,
        )?;
        let draw_buffer = UniqueShared::new(GuiVertexIndexBuffers::new(vk_ctx.clone(), 20_000)?);
        Ok(Self {
            vk_ctx,
            vs: vs::load(device.clone()).context("failed to create shader module")?,
            fs: fs::load(device).context("failed to create shader module")?,
            viewport,
            pipeline: UniqueShared::default(),
            texture_desc_sets: UniqueShared::new(BTreeMap::new()),
            texture_images: UniqueShared::new(BTreeMap::new()),
            texture_images_dirty: UniqueShared::new(false),
            next_frame: UniqueShared::new(None),
            last_nonempty_frame: UniqueShared::new(Vec::new()),
            primitives: UniqueShared::new(None),
            gui_enabled: UniqueShared::new(false),
            staging_buffer,
            draw_buffer,
        })
    }

    fn get_or_create_pipeline(&self) -> Result<Arc<GraphicsPipeline>> {
        if self.pipeline.get().is_none() {
            let vs = self
                .vs
                .entry_point("main")
                .context("vertex shader: entry point missing")?;
            let fs = self
                .fs
                .entry_point("main")
                .context("fragment shader: entry point missing")?;
            let vertex_input_state = EguiVertex::per_vertex().definition(&vs)?;
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
            )
            .map_err(Validated::unwrap)?;
            let device = self.vk_ctx.device();
            let swapchain = self.vk_ctx.swapchain()?;
            let subpass = PipelineRenderingCreateInfo {
                color_attachment_formats: vec![Some(swapchain.image_format())],
                ..Default::default()
            };
            *self.pipeline.get() = Some(GraphicsPipeline::new(
                device,
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
                        subpass.color_attachment_formats.len() as u32,
                        ColorBlendAttachmentState {
                            blend: Some(AttachmentBlend::alpha()),
                            ..Default::default()
                        },
                    )),
                    subpass: Some(subpass.into()),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                },
            )?);
        }
        Ok(self.pipeline.get().clone().unwrap())
    }

    fn create_image_view(
        &self,
        id: TextureId,
        image: Id<Image>,
        cbf: &mut RecordingCommandBuffer,
        tcx: &mut TaskContext,
    ) -> Result<()> {
        unsafe {
            cbf.copy_buffer_to_image(&CopyBufferToImageInfo {
                src_buffer: self.staging_buffer,
                dst_image: image,
                ..Default::default()
            })?;
        }
        let image_inner = tcx.image(image)?.image();
        let view = ImageView::new(
            image_inner.clone(),
            ImageViewCreateInfo::from_image(image_inner),
        )?;
        let pipeline = self.get_or_create_pipeline()?;
        let layout = pipeline
            .layout()
            .set_layouts()
            .first()
            .context("no descriptor sets in pipeline layout")?;
        let desc_set = RawDescriptorSet::new(self.vk_ctx.descriptor_set_allocator(), layout, 0)?;
        let desc_writes = vec![WriteDescriptorSet::image_view_sampler(
            0,
            view.clone(),
            Sampler::new(
                self.vk_ctx.device(),
                SamplerCreateInfo {
                    mag_filter: Filter::Linear,
                    min_filter: Filter::Linear,
                    address_mode: [SamplerAddressMode::ClampToEdge; 3],
                    mipmap_mode: SamplerMipmapMode::Linear,
                    ..Default::default()
                },
            )?,
        )];
        unsafe {
            desc_set.update(&desc_writes, &[])?;
        }
        self.texture_desc_sets
            .get()
            .insert(id, (desc_set, desc_writes));
        Ok(())
    }

    /// Write a single texture delta using the provided staging region and commandbuffer
    fn update_texture_within(
        &self,
        id: egui::TextureId,
        delta: &egui::epaint::ImageDelta,
        range: Range<u64>,
        cbf: &mut RecordingCommandBuffer,
        tcx: &mut TaskContext,
    ) -> Result<()> {
        let writer = tcx.write_buffer::<[u8]>(self.staging_buffer, range)?;
        match &delta.image {
            egui::ImageData::Color(image) => {
                assert_eq!(
                    image.width() * image.height(),
                    image.pixels.len(),
                    "Mismatch between texture size and texel count"
                );
                let bytes = image.pixels.iter().flat_map(Color32::to_array);
                writer
                    .iter_mut()
                    .zip(bytes)
                    .for_each(|(into, from)| *into = from);
            }
            egui::ImageData::Font(image) => {
                let bytes = image.srgba_pixels(None).flat_map(|color| color.to_array());
                writer
                    .iter_mut()
                    .zip(bytes)
                    .for_each(|(into, from)| *into = from);
            }
        }

        // Copy texture data to existing image if delta pos exists (e.g. font changed)
        if let Some(pos) = delta.pos {
            let texture_images = self.texture_images.get();
            let Some(existing_image) = texture_images.get(&id) else {
                error!("attempted to write to nonexistent image: {id:?}");
                return Ok(());
            };

            unsafe {
                cbf.copy_buffer_to_image(&CopyBufferToImageInfo {
                    src_buffer: self.staging_buffer,
                    dst_image: *existing_image,
                    regions: &[BufferImageCopy {
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
                    }],
                    ..Default::default()
                })?;
            }
        } else {
            let extent = [delta.image.width() as u32, delta.image.height() as u32, 1];
            let img = self
                .vk_ctx
                .resources()
                .create_image(
                    ImageCreateInfo {
                        image_type: ImageType::Dim2d,
                        format: Format::R8G8B8A8_SRGB,
                        extent,
                        usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                        ..Default::default()
                    },
                    AllocationCreateInfo::default(),
                )
                .unwrap();
            self.texture_images.get().insert(id, img);
            *self.texture_images_dirty.get() = true;
        }
        Ok(())
    }
    pub(crate) fn pre_render_update(
        &self,
        cbf: &mut RecordingCommandBuffer,
        tcx: &mut TaskContext,
        world: &VulkanoContext,
        sets: &[(egui::TextureId, egui::epaint::ImageDelta)],
    ) -> Result<()> {
        let primitives = self.primitives.get().take().expect("no GUI primitives");
        let mut meshes = primitives
            .iter()
            .filter_map(|mesh| match &mesh.primitive {
                Primitive::Mesh(m) => Some(m),
                Primitive::Callback(_) => unimplemented!(),
            })
            .cloned()
            .collect_vec();
        if *self.gui_enabled.get() && meshes.is_empty() {
            // There is some jank whereby the meshes can be empty while updating the textures.
            // This is a hack to prevent the GUI from flickering in this case.
            meshes = self.last_nonempty_frame.clone_inner();
        } else {
            self.last_nonempty_frame.get().clone_from(&meshes);
        }
        let image_idx = tcx
            .swapchain(world.swapchain_id())
            .unwrap()
            .current_image_index()
            .unwrap() as usize;
        self.draw_buffer.get().write(
            image_idx,
            &meshes.iter().flat_map(|m| m.vertices.clone()).collect_vec(),
            &meshes.iter().flat_map(|m| m.indices.clone()).collect_vec(),
            tcx,
        )?;
        let mut next_frame = self.next_frame.get();
        *next_frame = Some(meshes);

        for (id, image) in self.texture_images.get().iter() {
            if !self.texture_desc_sets.get().contains_key(id) {
                self.create_image_view(*id, *image, cbf, tcx)?;
            }
        }

        let total_size_bytes = sets
            .iter()
            .map(|(_, set)| image_size_bytes(set))
            .sum::<u64>()
            * 4;
        if total_size_bytes != 0 {
            check_lt!(total_size_bytes, Self::MAX_STAGING_SIZE_BYTES);
            let mut past_buffer_end = 0;

            for (id, delta) in sets {
                let image_size_bytes = image_size_bytes(delta);
                let range = past_buffer_end..(image_size_bytes + past_buffer_end);
                past_buffer_end += image_size_bytes;
                self.update_texture_within(*id, delta, range, cbf, tcx)?;
            }
        }
        Ok(())
    }

    pub(crate) fn image_writes(&self) -> Vec<Id<Image>> {
        self.texture_images.get().values().copied().collect()
    }
    pub(crate) fn buffer_writes(&self) -> Vec<Id<Buffer>> {
        let vertices_id = self.draw_buffer.get().vertices;
        let indices_id = self.draw_buffer.get().indices;
        vec![self.staging_buffer, vertices_id, indices_id]
    }
    pub(crate) fn build_task_graph(
        &self,
        task_graph: &mut TaskGraph<VulkanoContext>,
        virtual_swapchain_id: Id<Swapchain>,
    ) -> NodeId {
        let mut node =
            task_graph.create_task_node("gui_handler", QueueFamilyType::Graphics, self.clone());
        node.image_access(
            virtual_swapchain_id.current_image_id(),
            AccessTypes::COLOR_ATTACHMENT_WRITE,
            ImageLayoutType::Optimal,
        );
        node.buffer_access(self.staging_buffer, AccessTypes::COPY_TRANSFER_READ);
        node.buffer_access(
            self.draw_buffer.get().vertices,
            AccessTypes::VERTEX_ATTRIBUTE_READ,
        );
        node.buffer_access(self.draw_buffer.get().indices, AccessTypes::INDEX_READ);
        for &image in self.texture_images.get().values() {
            node.image_access(
                image,
                AccessTypes::FRAGMENT_SHADER_SAMPLED_READ,
                ImageLayoutType::Optimal,
            );
        }
        *self.texture_images_dirty.get() = false;
        self.draw_buffer.get().is_dirty = false;
        node.build()
    }
    pub(crate) fn is_dirty(&self) -> bool {
        *self.texture_images_dirty.get() || self.draw_buffer.get().is_dirty
    }
}

// Execution methods
impl GuiRenderer {
    // I think Clippy is wrong on this one.
    #[allow(clippy::redundant_else)]
    fn prepare_mesh_draw(
        &self,
        cbf: &mut RecordingCommandBuffer,
        layout: &PipelineLayout,
        mesh: &Mesh,
    ) -> Result<()> {
        let texture_desc_sets = self.texture_desc_sets.get();
        let Some((desc_set, _)) = texture_desc_sets.get(&mesh.texture_id) else {
            if self.is_dirty() {
                // Expected case; textures not yet uploaded.
                return Ok(());
            } else {
                bail!("GUI texture missing: {:?}", mesh.texture_id);
            }
        };
        unsafe {
            cbf.as_raw().bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                layout,
                0,
                &[desc_set],
                &[],
            )?;
        }
        Ok(())
    }
}

impl Task for GuiRenderer {
    type World = VulkanoContext;

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer,
        tcx: &mut TaskContext,
        world: &Self::World,
    ) -> TaskResult {
        let Some(frame) = self.next_frame.get().take() else {
            return Ok(());
        };

        let viewport = self.viewport.get().clone();
        let push_constants = {
            let viewport_extent = viewport.aa_extent();
            // TODO: fix this
            let scale_factor = viewport.scale_factor() / viewport.gui_scale_factor();
            vs::VertPC {
                u_screen_size: (viewport_extent * scale_factor).into(),
            }
        };

        let image_idx = tcx
            .swapchain(world.swapchain_id())
            .unwrap()
            .current_image_index()
            .unwrap() as usize;
        let image_view = world.current_image_view(image_idx);
        let pipeline = self.get_or_create_pipeline().unwrap();
        unsafe {
            cbf.as_raw()
                .begin_rendering(&RenderingInfo {
                    color_attachments: vec![Some(RenderingAttachmentInfo {
                        load_op: Load,
                        store_op: Store,
                        ..RenderingAttachmentInfo::image_view(image_view)
                    })],
                    render_area_extent: [
                        viewport.inner().extent[0] as u32,
                        viewport.inner().extent[1] as u32,
                    ],
                    layer_count: 1,
                    ..Default::default()
                })
                .unwrap();
            cbf.bind_pipeline_graphics(&pipeline)?.push_constants(
                pipeline.layout(),
                0,
                &push_constants,
            )?;
        }
        self.draw_buffer.get().bind(cbf).unwrap();
        world.perf_stats().lap("GuiRenderer: begin_rendering()");

        let mut vertex_cursor = 0;
        let mut index_cursor = 0;
        for mesh in &frame {
            if let Err(e) = self.prepare_mesh_draw(cbf, pipeline.layout(), mesh) {
                error!("{}", e.root_cause());
            } else {
                unsafe {
                    cbf.draw_indexed(mesh.indices.len() as u32, 1, index_cursor, vertex_cursor, 0)?;
                }
            }
            index_cursor += mesh.indices.len() as u32;
            vertex_cursor += i32::try_from(mesh.vertices.len())
                .with_context(|| format!("overflowed vertex_cursor: {}", mesh.vertices.len()))
                .unwrap();
        }
        world.perf_stats().lap("GuiRenderer: draw_indexed()");
        unsafe {
            cbf.as_raw().end_rendering()?;
        }
        world.perf_stats().lap("GuiRenderer: end_rendering()");
        Ok(())
    }
}
