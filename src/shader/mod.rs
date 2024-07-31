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
    device::Device,
    descriptor_set::{
        PersistentDescriptorSet,
        WriteDescriptorSet,
        layout::DescriptorSetLayoutCreateFlags
    },
    command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassBeginInfo, SubpassContents, SubpassEndInfo},
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    shader::{EntryPoint, ShaderModule},
    render_pass::{Framebuffer, Subpass},
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
#[derive(Clone)]
pub struct ShaderPair {
    name: ShaderName,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
}

impl ShaderPair {
    pub(crate) fn new_basic(device: Arc<Device>) -> Result<Self> {
        Ok(Self {
            name: "basic",
            vs: basic::vertex_shader::load(device.clone()).context("failed to create shader module")?,
            fs: basic::fragment_shader::load(device).context("failed to create shader module")?,
        })
    }
    pub(crate) fn entry_vs(&self) -> Result<EntryPoint> {
        self.vs.entry_point("main")
            .context("vertex shader: entry point missing")
    }
    pub(crate) fn entry_fs(&self) -> Result<EntryPoint> {
        self.fs.entry_point("main")
            .context("fragment shader: entry point missing")
    }

    pub fn name(&self) -> ShaderName { self.name }
}

#[derive(Clone)]
pub(crate) struct ShaderWithPipeline {
    shaders: ShaderPair,
    viewport: Arc<Mutex<AdjustedViewport>>,
    pipeline: Option<Arc<GraphicsPipeline>>,
    vertex_buffer: Option<Subbuffer<[basic::Vertex]>>,
    uniform_buffer: Option<Subbuffer<basic::UniformData>>,
    uniform_buffer_set: Option<Arc<PersistentDescriptorSet>>,
    resource_handler: ResourceHandler,
}

impl ShaderWithPipeline {
    pub(crate) fn new(shaders: ShaderPair, viewport: Arc<Mutex<AdjustedViewport>>, resource_handler: ResourceHandler) -> Self {
        Self {
            shaders,
            viewport,
            pipeline: None,
            vertex_buffer: None,
            uniform_buffer: None,
            uniform_buffer_set: None,
            resource_handler,
        }
    }
    pub(crate) fn on_render(
        &mut self,
        ctx: &VulkanoContext,
        receiver: &mut MutexGuard<RenderInfoReceiver>,
    ) -> Result<()> {
        let vertex_buffer = self.get_or_create_vertex_buffer(ctx, receiver)?;
        let mut writer = vertex_buffer.write()?;
        for render_info in &receiver.render_info {
            for vertex_index in render_info.vertex_indices.clone() {
                // Calculate transformed UVs.
                let texture = render_info.inner.texture.clone().unwrap_or_default();
                let mut blend_col = render_info.inner.col;
                let mut uv = receiver.vertices[vertex_index].uv;
                if let Some(tex) = self.resource_handler.texture.get(&texture) {
                    if let Some(tex) = tex.ready() {
                        uv = render_info.inner.texture_sub_area.uv(&tex, uv);
                    } else {
                        // blend_col = Colour::new(0., 0., 0., 0.);
                        blend_col = Colour::magenta();
                    }
                } else {
                    error!("missing texture: {}", texture);
                    blend_col = Colour::magenta();
                }
                writer[vertex_index] = basic::Vertex {
                    position: receiver.vertices[vertex_index].vertex.into(),
                    uv: uv.into(),
                    texture_id: texture.into(),
                    translation: render_info.transform.centre.into(),
                    #[allow(clippy::cast_possible_truncation)]
                    rotation: render_info.transform.rotation as f32,
                    scale: render_info.transform.scale.into(),
                    blend_col: blend_col.into(),
                };
            }
        }
        Ok(())
    }
    pub(crate) fn build_render_pass(
        &mut self,
        ctx: &VulkanoContext,
        framebuffer: Arc<Framebuffer>,
        receiver: &mut MutexGuard<RenderInfoReceiver>,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>
    ) -> Result<()> {
        self.on_render(ctx, receiver)?;

        let pipeline = self.get_or_create_pipeline(ctx)?;
        let uniform_buffer_set = self.get_or_create_uniform_buffer_set(ctx)?;
        let vertex_buffer = self.get_or_create_vertex_buffer(ctx, receiver)?;
        let layout = pipeline.layout().clone();
        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some(receiver.get_clear_col().as_f32().into())],
                    ..RenderPassBeginInfo::framebuffer(framebuffer)
                },
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )?
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
                  1, 0, 0)?
            .end_render_pass(SubpassEndInfo::default())?;
        Ok(())
    }
    pub(crate) fn reset_uniform_buffer(&mut self) {
        self.uniform_buffer = None;
    }
    pub(crate) fn reset_vertex_buffer(&mut self) {
        self.vertex_buffer = None;
    }

    fn get_or_create_pipeline(&mut self, ctx: &VulkanoContext) -> Result<Arc<GraphicsPipeline>> {
        match self.pipeline.clone() {
            None => {
                let vs = self.shaders.entry_vs()?;
                let fs = self.shaders.entry_fs()?;
                let vertex_input_state =
                    basic::Vertex::per_vertex().definition(&vs.info().input_interface)?;
                let stages = [
                    PipelineShaderStageCreateInfo::new(vs),
                    PipelineShaderStageCreateInfo::new(fs),
                ];
                let mut create_info = PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages);
                for layout in &mut create_info.set_layouts {
                    layout.flags |= DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL;
                }
                let layout = PipelineLayout::new(
                    ctx.device(),
                    create_info.into_pipeline_layout_create_info(ctx.device())?,
                ).map_err(Validated::unwrap)?;
                let subpass = Subpass::from(ctx.render_pass(), 0).context("failed to create subpass")?;

                let pipeline = GraphicsPipeline::new(
                    ctx.device(),
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
        ctx: &VulkanoContext,
        receiver: &mut MutexGuard<RenderInfoReceiver>,
    ) -> Result<Subbuffer<[basic::Vertex]>> {
        match self.vertex_buffer.clone() {
            None => {
                let vertices = receiver.vertices.iter().map(|_| basic::Vertex {
                    blend_col: Colour::magenta().into(),
                    ..Default::default()
                });
                let vertex_buffer = Buffer::from_iter(
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
                    vertices,
                ).map_err(Validated::unwrap)?;
                self.vertex_buffer = Some(vertex_buffer.clone());
                Ok(vertex_buffer)
            },
            Some(vertex_buffer) => Ok(vertex_buffer)
        }
    }
    fn get_or_create_uniform_buffer(&mut self, ctx: &VulkanoContext) -> Result<Subbuffer<basic::UniformData>> {
        Ok(match self.uniform_buffer.clone() {
            None => {
                let uniform_buffer= Buffer::new_sized(
                    ctx.memory_allocator(),
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
                    basic::UniformData {
                        #[allow(clippy::cast_possible_truncation)]
                        window_width: viewport.physical_width() as f32,
                        #[allow(clippy::cast_possible_truncation)]
                        window_height: viewport.physical_height() as f32,
                        #[allow(clippy::cast_possible_truncation)]
                        scale_factor: viewport.scale_factor() as f32,
                    }
                };
                *uniform_buffer.write()? = uniform_data;
                self.uniform_buffer = Some(uniform_buffer.clone());
                uniform_buffer
            },
            Some(uniform_buffer) => uniform_buffer,
        })
    }

    fn get_or_create_uniform_buffer_set(&mut self,
                                        ctx: &VulkanoContext,
    ) -> Result<Arc<PersistentDescriptorSet>> {
        let pipeline = self.get_or_create_pipeline(ctx)?;
        Ok(match self.uniform_buffer_set.clone() {
            None => {
                let sampler = Sampler::new(
                    ctx.device(),
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
                let uniform_buffer_set = PersistentDescriptorSet::new(
                    &ctx.descriptor_set_allocator(),
                    pipeline.layout().set_layouts()[0].clone(),
                    [
                        WriteDescriptorSet::buffer(0, self.get_or_create_uniform_buffer(ctx)?),
                        WriteDescriptorSet::image_view_sampler_array(
                            1,
                            0,
                            textures.iter().cloned().zip(vec![sampler.clone(); textures.len()])
                        ),
                    ],
                    [],
                ).map_err(Validated::unwrap)?;
                self.uniform_buffer_set = Some(uniform_buffer_set.clone());
                uniform_buffer_set
            },
            Some(uniform_buffer_set) => uniform_buffer_set,
        })
    }
}
