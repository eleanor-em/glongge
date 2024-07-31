use std::sync::{Arc, Mutex};
use anyhow::{Context, Result};
use itertools::Itertools;
use tracing::error;
use vulkano::{pipeline::{
    graphics::{
        multisample::MultisampleState,
        input_assembly::InputAssemblyState,
        GraphicsPipelineCreateInfo,
        rasterization::RasterizationState,
        vertex_input::{Vertex, VertexDefinition},
        viewport::ViewportState,
        color_blend::{AttachmentBlend, ColorBlendAttachmentState, ColorBlendState}
    },
    GraphicsPipeline,
    Pipeline,
    PipelineBindPoint,
    PipelineLayout,
    PipelineShaderStageCreateInfo,
    layout::PipelineDescriptorSetLayoutCreateInfo
}, memory::allocator::{AllocationCreateInfo, MemoryTypeFilter}, image::sampler::{BorderColor, Filter, Sampler, SamplerAddressMode, SamplerCreateInfo}, descriptor_set::{
    PersistentDescriptorSet,
    WriteDescriptorSet,
    layout::DescriptorSetLayoutCreateFlags
}, command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer}, buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer}, shader::ShaderModule, render_pass::Subpass, Validated, DeviceSize};
use crate::{
    core::{
        render::RenderFrame,
        prelude::*,
        vk::{AdjustedViewport, VulkanoContext},
        util::UniqueShared
    },
    shader::glsl::*,
};
pub use vulkano::pipeline::graphics::vertex_input::Vertex as VkVertex;

pub mod vertex;
pub mod glsl;

pub type ShaderName = &'static str;

pub trait Shader: Send {
    fn name() -> ShaderName where Self: Sized;
    fn on_render(
        &mut self,
        render_frame: &RenderFrame
    ) -> Result<()>;
    fn build_render_pass(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>
    ) -> Result<()>;
}

#[derive(Clone)]
struct CachedVertexBuffer<T: VkVertex> {
    ctx: VulkanoContext,
    inner: Subbuffer<[T]>,
    vertex_count: usize,
    begin: usize,
}

impl<T: VkVertex> CachedVertexBuffer<T> {
    fn new(ctx: VulkanoContext, size: usize) -> Result<Self> {
        let inner = Self::create_vertex_buffer(
            &ctx,
            (size * ctx.framebuffers().len()) as DeviceSize
        )?;
        Ok(Self { ctx, inner, vertex_count: 0, begin: 0 })
    }

    fn len(&self) -> usize { self.inner.len() as usize / self.ctx.framebuffers().len() }

    fn write(&mut self, mut data: Vec<T>) -> Result<()> {
        let buf_len = usize::try_from(self.inner.len())
            .unwrap_or_else(|_| panic!("inexplicable: self.inner.len() = {}", self.inner.len()));
        check_le!(self.begin, buf_len);
        self.begin += self.len();
        if self.begin == buf_len {
            self.begin = 0;
        }
        let begin = self.begin as u64;
        let end = (self.begin + self.len()) as u64;
        let slice = self.inner.clone().slice(begin..end);
        slice.write()?[..data.len()].swap_with_slice(&mut data);

        self.vertex_count = data.len();
        Ok(())
    }

    fn create_vertex_buffer(ctx: &VulkanoContext, size: DeviceSize) -> Result<Subbuffer<[T]>> {
        Ok(Buffer::new_unsized(
            ctx.memory_allocator(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            size
        ).map_err(Validated::unwrap)?)
    }

    fn draw(&self, builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>) -> Result<()> {
        let vertex_count = u32::try_from(self.vertex_count)
            .unwrap_or_else(|_| panic!("tried to draw too many vertices: {}", self.vertex_count));
        let begin = u32::try_from(self.begin)
            .unwrap_or_else(|_| panic!("inexplicable: self.begin = {}", self.begin));
        check_le!(begin + vertex_count, self.inner.len() as u32);
        builder.bind_vertex_buffers(0, self.inner.clone())?
            .draw(vertex_count, 1, begin, 0)?;
        Ok(())
    }
}

#[derive(Clone)]
pub struct SpriteShader {
    ctx: VulkanoContext,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    viewport: UniqueShared<AdjustedViewport>,
    pipeline: Option<Arc<GraphicsPipeline>>,
    vertex_buffer: CachedVertexBuffer<sprite::Vertex>,
    resource_handler: ResourceHandler,
}

impl SpriteShader {
    pub fn new(
        ctx: VulkanoContext,
        viewport: UniqueShared<AdjustedViewport>,
        resource_handler: ResourceHandler
    ) -> Result<Arc<Mutex<dyn Shader>>> {
        let device = ctx.device();
        let vertex_buffer = CachedVertexBuffer::new(ctx.clone(), 1_000_000)?;
        Ok(Arc::new(Mutex::new(Self {
            ctx,
            vs: sprite::vertex_shader::load(device.clone()).context("failed to create shader module")?,
            fs: sprite::fragment_shader::load(device).context("failed to create shader module")?,
            viewport,
            pipeline: None,
            vertex_buffer,
            resource_handler,
        })) as Arc<Mutex<dyn Shader>>)
    }

    fn get_or_create_pipeline(&mut self) -> Result<Arc<GraphicsPipeline>> {
        match self.pipeline.clone() {
            None => {
                let vs = self.vs.entry_point("main")
                    .context("vertex shader: entry point missing")?;
                let fs = self.fs.entry_point("main")
                    .context("fragment shader: entry point missing")?;
                let vertex_input_state =
                    sprite::Vertex::per_vertex().definition(&vs.info().input_interface)?;
                let stages = [
                    PipelineShaderStageCreateInfo::new(vs),
                    PipelineShaderStageCreateInfo::new(fs),
                ];
                let mut create_info = PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages);
                for layout in &mut create_info.set_layouts {
                    layout.flags |= DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL;
                }
                let layout = PipelineLayout::new(
                    self.ctx.device(),
                    create_info.into_pipeline_layout_create_info(self.ctx.device())?,
                ).map_err(Validated::unwrap)?;
                let subpass = Subpass::from(self.ctx.render_pass(), 0).context("failed to create subpass")?;

                let pipeline = GraphicsPipeline::new(
                    self.ctx.device(),
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
    fn create_uniform_buffer(&mut self) -> Result<Subbuffer<sprite::UniformData>> {
        let uniform_buffer= Buffer::new_sized(
            self.ctx.memory_allocator(),
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        ).map_err(Validated::unwrap)?;

        let uniform_data = {
            let viewport = self.viewport.get();
            sprite::UniformData {
                #[allow(clippy::cast_possible_truncation)]
                window_width: viewport.physical_width() as f32,
                #[allow(clippy::cast_possible_truncation)]
                window_height: viewport.physical_height() as f32,
                #[allow(clippy::cast_possible_truncation)]
                scale_factor: viewport.scale_factor() as f32,
            }
        };
        *uniform_buffer.write()? = uniform_data;
        Ok(uniform_buffer)
    }

    fn create_uniform_buffer_set(&mut self) -> Result<Arc<PersistentDescriptorSet>> {
        let pipeline = self.get_or_create_pipeline()?;
        let sampler = Sampler::new(
            self.ctx.device(),
            SamplerCreateInfo {
                mag_filter: Filter::Nearest,
                min_filter: Filter::Nearest,
                address_mode: [SamplerAddressMode::ClampToBorder; 3],
                border_color: BorderColor::FloatTransparentBlack,
                ..Default::default()
            }).map_err(Validated::unwrap)?;
        let mut textures = self.resource_handler.texture.ready_values()
            .into_iter()
            .filter_map(|tex| tex.image_view())
            .collect_vec();
        check_le!(textures.len(), MAX_TEXTURE_COUNT);
        let blank = textures.first()
            .expect("textures.first() should always contain a blank texture")
            .clone();
        textures.extend(vec![blank; MAX_TEXTURE_COUNT - textures.len()]);
        Ok(PersistentDescriptorSet::new(
            &self.ctx.descriptor_set_allocator(),
            pipeline.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, self.create_uniform_buffer()?),
                WriteDescriptorSet::image_view_sampler_array(
                    1,
                    0,
                    textures.iter().cloned().zip(vec![sampler.clone(); textures.len()])
                ),
            ],
            [],
        ).map_err(Validated::unwrap)?)
    }
}

impl Shader for SpriteShader {
    fn name() -> ShaderName
    where
        Self: Sized
    {
        "basic"
    }

    fn on_render(
        &mut self,
        render_frame: &RenderFrame
    ) -> Result<()> {
        let render_infos = render_frame.render_infos
            .iter()
            .sorted_unstable_by_key(|item| item.depth);
        let mut vertices = Vec::with_capacity(self.vertex_buffer.len());
        for render_info in render_infos {
            for vertex_index in render_info.vertex_indices.clone() {
                // Calculate transformed UVs.
                let texture = render_info.inner.texture.clone().unwrap_or_default();
                // TODO: better way to switch shaders.
                let final_vertex = if texture.id() != 0 {
                    let vertex = render_frame.vertices[vertex_index];
                    let mut blend_col = render_info.inner.col;
                    let mut uv = vertex.uv;
                    if let Some(tex) = self.resource_handler.texture.get(&texture) {
                        if let Some(tex) = tex.ready() {
                            uv = render_info.inner.texture_sub_area.uv(&tex, uv);
                        } else {
                            warn!("texture not ready: {}", texture);
                            blend_col = Colour::empty();
                        }
                    } else {
                        error!("missing texture: {}", texture);
                        blend_col = Colour::magenta();
                    }
                    sprite::Vertex {
                        position: vertex.xy.into(),
                        uv: uv.into(),
                        texture_id: texture.into(),
                        translation: render_info.transform.centre.into(),
                        #[allow(clippy::cast_possible_truncation)]
                        rotation: render_info.transform.rotation as f32,
                        scale: render_info.transform.scale.into(),
                        blend_col: blend_col.into(),
                    }
                } else {
                    sprite::Vertex {
                        blend_col: Colour::empty().into(),
                        ..Default::default()
                    }
                };
                vertices.push(final_vertex);
            }
        }
        self.vertex_buffer.write(vertices)?;
        Ok(())
    }
    fn build_render_pass(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>
    ) -> Result<()> {
        let pipeline = self.get_or_create_pipeline()?;
        let uniform_buffer_set = self.create_uniform_buffer_set()?;
        let layout = pipeline.layout().clone();
        builder
            .bind_pipeline_graphics(pipeline.clone())?
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                layout,
                0,
                uniform_buffer_set.clone(),
            )?;
        self.vertex_buffer.draw(builder)?;
        Ok(())
    }
}

#[derive(Clone)]
pub struct WireframeShader {
    ctx: VulkanoContext,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    viewport: UniqueShared<AdjustedViewport>,
    pipeline: Option<Arc<GraphicsPipeline>>,
    vertex_buffer: CachedVertexBuffer<wireframe::Vertex>,
}

impl WireframeShader {
    pub fn new(
        ctx: VulkanoContext,
        viewport: UniqueShared<AdjustedViewport>
    ) -> Result<Arc<Mutex<dyn Shader>>> {
        let device = ctx.device();
        let vertex_buffer = CachedVertexBuffer::new(ctx.clone(), 1_000_000)?;
        Ok(Arc::new(Mutex::new(Self {
            ctx,
            vs: wireframe::vertex_shader::load(device.clone()).context("failed to create shader module")?,
            fs: wireframe::fragment_shader::load(device).context("failed to create shader module")?,
            viewport,
            pipeline: None,
            vertex_buffer,
        })) as Arc<Mutex<dyn Shader>>)
    }

    fn get_or_create_pipeline(&mut self) -> Result<Arc<GraphicsPipeline>> {
        match self.pipeline.clone() {
            None => {
                let vs = self.vs.entry_point("main")
                    .context("vertex shader: entry point missing")?;
                let fs = self.fs.entry_point("main")
                    .context("fragment shader: entry point missing")?;
                let vertex_input_state =
                    wireframe::Vertex::per_vertex().definition(&vs.info().input_interface)?;
                let stages = [
                    PipelineShaderStageCreateInfo::new(vs),
                    PipelineShaderStageCreateInfo::new(fs),
                ];
                let mut create_info = PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages);
                for layout in &mut create_info.set_layouts {
                    layout.flags |= DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL;
                }
                let layout = PipelineLayout::new(
                    self.ctx.device(),
                    create_info.into_pipeline_layout_create_info(self.ctx.device())?,
                ).map_err(Validated::unwrap)?;
                let subpass = Subpass::from(self.ctx.render_pass(), 0).context("failed to create subpass")?;

                let pipeline = GraphicsPipeline::new(
                    self.ctx.device(),
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
    fn create_uniform_buffer(&mut self) -> Result<Subbuffer<sprite::UniformData>> {
        let uniform_buffer= Buffer::new_sized(
            self.ctx.memory_allocator(),
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        ).map_err(Validated::unwrap)?;

        let uniform_data = {
            let viewport = self.viewport.get();
            sprite::UniformData {
                #[allow(clippy::cast_possible_truncation)]
                window_width: viewport.physical_width() as f32,
                #[allow(clippy::cast_possible_truncation)]
                window_height: viewport.physical_height() as f32,
                #[allow(clippy::cast_possible_truncation)]
                scale_factor: viewport.scale_factor() as f32,
            }
        };
        *uniform_buffer.write()? = uniform_data;
        Ok(uniform_buffer)
    }

    fn create_uniform_buffer_set(&mut self) -> Result<Arc<PersistentDescriptorSet>> {
        let pipeline = self.get_or_create_pipeline()?;
        Ok(PersistentDescriptorSet::new(
            &self.ctx.descriptor_set_allocator(),
            pipeline.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, self.create_uniform_buffer()?),
            ],
            [],
        ).map_err(Validated::unwrap)?)
    }
}

impl Shader for WireframeShader {
    fn name() -> ShaderName
    where
        Self: Sized
    {
        "basic"
    }

    fn on_render(
        &mut self,
        render_frame: &RenderFrame,
    ) -> Result<()> {
        let mut vertices = Vec::with_capacity(self.vertex_buffer.len());
        for render_info in &render_frame.render_infos {
            for vertex_index in render_info.vertex_indices.clone() {
                // Calculate transformed UVs.
                let texture = render_info.inner.texture.clone().unwrap_or_default();
                // TODO: better way to switch shaders.
                let vertex = if texture.id() == 0 {
                    wireframe::Vertex {
                        position: render_frame.vertices[vertex_index].xy.into(),
                        translation: render_info.transform.centre.into(),
                        #[allow(clippy::cast_possible_truncation)]
                        rotation: render_info.transform.rotation as f32,
                        scale: render_info.transform.scale.into(),
                        blend_col: render_info.inner.col.into(),
                    }
                } else {
                    wireframe::Vertex {
                        blend_col: Colour::empty().into(),
                        ..Default::default()
                    }
                };
                vertices.push(vertex);
            }
        }
        self.vertex_buffer.write(vertices)?;
        Ok(())
    }
    fn build_render_pass(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>
    ) -> Result<()> {
        let pipeline = self.get_or_create_pipeline()?;
        let uniform_buffer_set = self.create_uniform_buffer_set()?;
        let layout = pipeline.layout().clone();
        builder
            .bind_pipeline_graphics(pipeline.clone())?
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                layout,
                0,
                uniform_buffer_set.clone(),
            )?;
        self.vertex_buffer.draw(builder)?;
        Ok(())
    }
}
