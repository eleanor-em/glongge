use std::sync::{Arc, Mutex, MutexGuard};
use anyhow::{Context, Result};
use itertools::Itertools;
use tracing::error;
use vulkano::{
    pipeline::{
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
    },
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    image::sampler::{BorderColor, Filter, Sampler, SamplerAddressMode, SamplerCreateInfo},
    descriptor_set::{
        PersistentDescriptorSet,
        WriteDescriptorSet,
        layout::DescriptorSetLayoutCreateFlags
    },
    command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer},
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    shader::ShaderModule,
    render_pass::Subpass,
    Validated
};
use crate::{
    core::{
        prelude::*,
        render::RenderInfoReceiver,
        vk::{AdjustedViewport, VulkanoContext}
    },
    shader::glsl::*,
};

pub mod vertex;
mod glsl;


pub type ShaderName = &'static str;

pub trait Shader: Send {
    fn name() -> ShaderName where Self: Sized;
    fn on_render(
        &mut self,
        receiver: &mut MutexGuard<RenderInfoReceiver>
    ) -> Result<()>;
    fn build_render_pass(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>
    ) -> Result<()>;
}

#[derive(Clone)]
pub struct SpriteShader {
    ctx: VulkanoContext,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    viewport: Arc<Mutex<AdjustedViewport>>,
    pipeline: Option<Arc<GraphicsPipeline>>,
    vertex_buffer: Option<Subbuffer<[sprite::Vertex]>>,
    clear_col: Colour,
    resource_handler: ResourceHandler,
}

impl SpriteShader {
    pub fn new(
        ctx: VulkanoContext,
        viewport: Arc<Mutex<AdjustedViewport>>,
        resource_handler: ResourceHandler
    ) -> Result<Arc<Mutex<dyn Shader>>> {
        let device = ctx.device();
        Ok(Arc::new(Mutex::new(Self {
            ctx,
            vs: sprite::vertex_shader::load(device.clone()).context("failed to create shader module")?,
            fs: sprite::fragment_shader::load(device).context("failed to create shader module")?,
            viewport,
            pipeline: None,
            vertex_buffer: None,
            clear_col: Colour::default(),
            resource_handler,
        })) as Arc<Mutex<dyn Shader>>)
    }

    fn write_vertices(
        &self,
        receiver: &mut MutexGuard<RenderInfoReceiver>,
        buf: &mut [sprite::Vertex]
    ) {
        check_le!(receiver.vertices.len(), buf.len());
        for render_info in &receiver.render_info {
            for vertex_index in render_info.vertex_indices.clone() {
                // Calculate transformed UVs.
                let texture = render_info.inner.texture.clone().unwrap_or_default();
                // TODO: better way to switch shaders.
                let vertex = if texture.id() != 0 {
                    let mut blend_col = render_info.inner.col;
                    let mut uv = receiver.vertices[vertex_index].uv;
                    if let Some(tex) = self.resource_handler.texture.get(&texture) {
                        if let Some(tex) = tex.ready() {
                            uv = render_info.inner.texture_sub_area.uv(&tex, uv);
                        } else {
                            blend_col = Colour::new(0., 0., 0., 0.);
                        }
                    } else {
                        error!("missing texture: {}", texture);
                        blend_col = Colour::magenta();
                    }
                    sprite::Vertex {
                        position: receiver.vertices[vertex_index].vertex.into(),
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
                buf[vertex_index] = vertex;
            }
        }
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
                            viewports: [self.viewport.try_lock().unwrap().inner()].into_iter().collect(),
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
    fn get_or_create_vertex_buffer(
        &mut self,
        receiver: &mut MutexGuard<RenderInfoReceiver>,
    ) -> Result<Subbuffer<[sprite::Vertex]>> {
        if let Some(vertex_buffer) = &self.vertex_buffer {
            if receiver.vertices.len() == vertex_buffer.len() as usize {
                return Ok(vertex_buffer.clone());
            }
        }

        let mut vertices = vec![sprite::Vertex {
            blend_col: Colour::magenta().into(),
            ..Default::default()
        }; receiver.vertices.len()];
        self.write_vertices(receiver, &mut vertices);
        let vertex_buffer = Buffer::from_iter(
            self.ctx.memory_allocator(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertices
        ).map_err(Validated::unwrap)?;
        self.vertex_buffer = Some(vertex_buffer.clone());
        Ok(vertex_buffer)
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
            let viewport = self.viewport.try_lock().unwrap();
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
        receiver: &mut MutexGuard<RenderInfoReceiver>,
    ) -> Result<()> {
        self.clear_col = receiver.get_clear_col();
        let vertex_buffer = self.get_or_create_vertex_buffer(receiver)?;
        self.write_vertices(receiver, &mut vertex_buffer.write()?);
        Ok(())
    }
    fn build_render_pass(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>
    ) -> Result<()> {
        let pipeline = self.get_or_create_pipeline()?;
        let uniform_buffer_set = self.create_uniform_buffer_set()?;
        let vertex_buffer = self.vertex_buffer.as_ref().unwrap();
        let layout = pipeline.layout().clone();
        builder
            .bind_pipeline_graphics(pipeline.clone())?
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                layout,
                0,
                uniform_buffer_set.clone(),
            )?
            .bind_vertex_buffers(0, vertex_buffer.clone())?
            .draw(u32::try_from(vertex_buffer.len())
                      .unwrap_or_else(|_| panic!("tried to draw too many vertices: {}", vertex_buffer.len())),
                  1, 0, 0)?;
        Ok(())
    }
}

#[derive(Clone)]
pub struct WireframeShader {
    ctx: VulkanoContext,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    viewport: Arc<Mutex<AdjustedViewport>>,
    pipeline: Option<Arc<GraphicsPipeline>>,
    vertex_buffer: Option<Subbuffer<[wireframe::Vertex]>>,
    clear_col: Colour,
}

impl WireframeShader {
    pub fn new(
        ctx: VulkanoContext,
        viewport: Arc<Mutex<AdjustedViewport>>
    ) -> Result<Arc<Mutex<dyn Shader>>> {
        let device = ctx.device();
        Ok(Arc::new(Mutex::new(Self {
            ctx,
            vs: wireframe::vertex_shader::load(device.clone()).context("failed to create shader module")?,
            fs: wireframe::fragment_shader::load(device).context("failed to create shader module")?,
            viewport,
            pipeline: None,
            vertex_buffer: None,
            clear_col: Colour::default(),
        })) as Arc<Mutex<dyn Shader>>)
    }

    fn write_vertices(
        &self,
        receiver: &mut MutexGuard<RenderInfoReceiver>,
        buf: &mut [wireframe::Vertex]
    ) {
        check_le!(receiver.vertices.len(), buf.len());
        for render_info in &receiver.render_info {
            for vertex_index in render_info.vertex_indices.clone() {
                // Calculate transformed UVs.
                let texture = render_info.inner.texture.clone().unwrap_or_default();
                // TODO: better way to switch shaders.
                let vertex = if texture.id() == 0 {
                    wireframe::Vertex {
                        position: receiver.vertices[vertex_index].vertex.into(),
                        translation: render_info.transform.centre.into(),
                        #[allow(clippy::cast_possible_truncation)]
                        rotation: render_info.transform.rotation as f32,
                        scale: render_info.transform.scale.into(),
                        blend_col: render_info.inner.col.into(),
                    }
                } else {
                    wireframe::Vertex {
                        position: receiver.vertices[vertex_index].vertex.into(),
                        blend_col: Colour::empty().into(),
                        ..Default::default()
                    }
                };
                buf[vertex_index] = vertex;
            }
        }
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
                            viewports: [self.viewport.try_lock().unwrap().inner()].into_iter().collect(),
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
    fn get_or_create_vertex_buffer(
        &mut self,
        receiver: &mut MutexGuard<RenderInfoReceiver>,
    ) -> Result<Subbuffer<[wireframe::Vertex]>> {
        if let Some(vertex_buffer) = &self.vertex_buffer {
            if receiver.vertices.len() == vertex_buffer.len() as usize {
                return Ok(vertex_buffer.clone());
            }
        }

        let mut vertices = vec![wireframe::Vertex::default(); receiver.vertices.len()];
        self.write_vertices(receiver, &mut vertices);
        let vertex_buffer = Buffer::from_iter(
            self.ctx.memory_allocator(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertices
        ).map_err(Validated::unwrap)?;
        self.vertex_buffer = Some(vertex_buffer.clone());
        Ok(vertex_buffer)
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
            let viewport = self.viewport.try_lock().unwrap();
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
        receiver: &mut MutexGuard<RenderInfoReceiver>,
    ) -> Result<()> {
        self.clear_col = receiver.get_clear_col();
        let vertex_buffer = self.get_or_create_vertex_buffer(receiver)?;
        self.write_vertices(receiver, &mut vertex_buffer.write()?);
        Ok(())
    }
    fn build_render_pass(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>
    ) -> Result<()> {
        let pipeline = self.get_or_create_pipeline()?;
        let uniform_buffer_set = self.create_uniform_buffer_set()?;
        let vertex_buffer = self.vertex_buffer.as_ref().unwrap();
        let layout = pipeline.layout().clone();
        builder
            .bind_pipeline_graphics(pipeline.clone())?
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                layout,
                0,
                uniform_buffer_set.clone(),
            )?
            .bind_vertex_buffers(0, vertex_buffer.clone())?
            .draw(u32::try_from(vertex_buffer.len())
                      .unwrap_or_else(|_| panic!("tried to draw too many vertices: {}", vertex_buffer.len())),
                  1, 0, 0)?;
        Ok(())
    }
}
