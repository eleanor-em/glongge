use std::{
    default::Default,
    sync::{Arc, Mutex, MutexGuard}
};
use num_traits::Zero;

use anyhow::{Context, Result};
use tracing::info;

use vulkano::{
    Validated,
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
        RenderPassBeginInfo, SubpassBeginInfo, SubpassContents, SubpassEndInfo,
    },
    descriptor_set::{
        PersistentDescriptorSet,
        WriteDescriptorSet,
        layout::DescriptorSetLayoutCreateFlags
    },
    image::sampler::{BorderColor, Filter, Sampler, SamplerAddressMode, SamplerCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    pipeline::{
        graphics::{
            color_blend::{
                ColorBlendAttachmentState,
                ColorBlendState,
                AttachmentBlend
            },
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::{Vertex, VertexDefinition},
            viewport::ViewportState,
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        GraphicsPipeline,
        Pipeline,
        PipelineBindPoint,
        PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    render_pass::{
        Framebuffer,
        Subpass
    },
    shader::ShaderModule,
};
use winit::window::Window;

use crate::{
    assert::check_le,
    core::{
        linalg::Vec2,
        vk_core::{
            AdjustedViewport,
            DataPerImage,
            PerImageContext,
            RenderEventHandler,
            VulkanoContext,
            WindowContext,
        }
    },
    gg::{
        RenderInfoFull,
        RenderDataReceiver,
        VertexWithUV,
    },
    resource::{
        ResourceHandler,
        texture::{
            MAX_TEXTURE_COUNT,
            TextureId,
        }
    },
    shader::sample::{basic_fragment_shader, basic_vertex_shader},
};

#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
struct UniformData {
    window_width: f32,
    window_height: f32,
    scale_factor: f32,
}
#[derive(BufferContents, Vertex, Debug, Clone, Copy)]
#[repr(C)]
struct BasicVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
    #[format(R32G32_SFLOAT)]
    translation: [f32; 2],
    #[format(R32_SFLOAT)]
    rotation: f32,
    #[format(R32G32_SFLOAT)]
    uv: [f32; 2],
    #[format(R32_UINT)]
    texture_id: u32,
    #[format(R32G32B32A32_SFLOAT)]
    blend_col: [f32; 4],
}

pub struct BasicRenderDataReceiver {
    vertices: Vec<VertexWithUV>,
    vertices_up_to_date: DataPerImage<bool>,
    render_data: Vec<RenderInfoFull>,
    viewport: AdjustedViewport,
}
impl BasicRenderDataReceiver {
    fn new(ctx: &VulkanoContext, viewport: AdjustedViewport) -> Self {
        Self {
            vertices: Vec::new(),
            vertices_up_to_date: DataPerImage::new_with_value(ctx, true),
            render_data: Vec::new(),
            viewport,
        }
    }
}
impl RenderDataReceiver for BasicRenderDataReceiver {
    fn update_vertices(&mut self, vertices: Vec<VertexWithUV>) {
        self.vertices_up_to_date.clone_from_value(false);
        self.vertices = vertices;
    }

    fn update_render_data(&mut self, render_data: Vec<RenderInfoFull>) {
        self.render_data = render_data;
    }

    fn current_viewport(&self) -> AdjustedViewport {
        self.viewport.clone()
    }
}

#[derive(Clone)]
pub struct BasicRenderHandler {
    resource_handler: ResourceHandler,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    viewport: AdjustedViewport,
    pipeline: Option<Arc<GraphicsPipeline>>,
    vertex_buffers: Option<DataPerImage<Subbuffer<[BasicVertex]>>>,
    uniform_buffers: Option<DataPerImage<Subbuffer<UniformData>>>,
    uniform_buffer_sets: Option<DataPerImage<Arc<PersistentDescriptorSet>>>,
    command_buffers: Option<DataPerImage<Arc<PrimaryAutoCommandBuffer>>>,
    render_data_receiver: Arc<Mutex<BasicRenderDataReceiver>>,
}

impl BasicRenderHandler {
    pub fn new(window_ctx: &WindowContext, ctx: &VulkanoContext, resource_handler: ResourceHandler) -> Result<Self> {
        let vs = basic_vertex_shader::load(ctx.device()).context("failed to create shader module")?;
        let fs = basic_fragment_shader::load(ctx.device()).context("failed to create shader module")?;
        let viewport = window_ctx.create_default_viewport();
        let render_data_receiver = Arc::new(Mutex::new(BasicRenderDataReceiver::new(
            ctx, viewport.clone()
        )));
        Ok(Self {
            resource_handler,
            vs,
            fs,
            viewport,
            pipeline: None,
            vertex_buffers: None,
            uniform_buffers: None,
            uniform_buffer_sets: None,
            command_buffers: None,
            render_data_receiver,
        })
    }

    fn get_or_create_vertex_buffers(&mut self,
                                    ctx: &VulkanoContext,
                                    receiver: &mut MutexGuard<BasicRenderDataReceiver>
    ) -> Result<&DataPerImage<Subbuffer<[BasicVertex]>>> {
        if self.vertex_buffers.is_none() {
            self.vertex_buffers = Some(DataPerImage::try_new_with_generator(ctx, || {
                Self::create_single_vertex_buffer(ctx, &receiver.vertices)
            })?);
            self.command_buffers = None;
            receiver
                .vertices_up_to_date
                .clone_from_value(true);
        }
        Ok(self.vertex_buffers.as_ref().unwrap())
    }
    fn create_single_vertex_buffer(
        ctx: &VulkanoContext,
        vertices: &[VertexWithUV],
    ) -> Result<Subbuffer<[BasicVertex]>> {
        let vertices = vertices.iter().map(|&v| BasicVertex {
            position: v.vertex.into(),
            translation: Vec2::zero().into(),
            rotation: 0.0,
            uv: v.uv.into(),
            texture_id: TextureId::default().into(),
            blend_col: [0.0, 0.0, 0.0, 0.0],
        });
        Ok(Buffer::from_iter(
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
        ).map_err(Validated::unwrap)?)
    }

    fn maybe_update_with_new_vertices(&mut self,
                                      ctx: &VulkanoContext,
                                      receiver: &mut MutexGuard<BasicRenderDataReceiver>,
                                      per_image_ctx: &mut MutexGuard<PerImageContext>
    ) -> Result<()> {
        if !per_image_ctx.get_current_value(&receiver.vertices_up_to_date) {
            per_image_ctx.replace_current_value(
                &mut self.vertex_buffers,
                Self::create_single_vertex_buffer(ctx, &receiver.vertices)?);
            let command_buffer = self.create_single_command_buffer(
                ctx,
                per_image_ctx.current_value_cloned(&ctx.framebuffers()),
                per_image_ctx.current_value_as_ref(&self.uniform_buffer_sets).clone(),
                per_image_ctx.current_value_as_ref(&self.vertex_buffers),
            )?;
            per_image_ctx.replace_current_value(&mut self.command_buffers, command_buffer);
            per_image_ctx.set_current_value(&mut receiver.vertices_up_to_date, true);
        }
        Ok(())
    }

    fn write_vertex_buffer(&mut self,
                           receiver: &mut MutexGuard<BasicRenderDataReceiver>,
                           per_image_ctx: &mut MutexGuard<PerImageContext>) -> Result<()> {
        let vertex_buffer = per_image_ctx.current_value_as_mut(&mut self.vertex_buffers);
        let mut out_vertices = Vec::new();
        for render_data in receiver.render_data.iter() {
            for vertex_index in render_data.vertex_indices.clone() {
                out_vertices.push(BasicVertex {
                    position: receiver.vertices[vertex_index].vertex.into(),
                    uv: receiver.vertices[vertex_index].uv.into(),
                    texture_id: render_data.inner.texture_id.unwrap_or_default().into(),
                    translation: render_data.transform.position.into(),
                    rotation: render_data.transform.rotation as f32,
                    blend_col: render_data.inner.col.into(),
                });
            }
        }
        vertex_buffer.write()?.swap_with_slice(&mut out_vertices);
        Ok(())
    }

    fn create_pipeline(&self, ctx: &VulkanoContext) -> Result<Arc<GraphicsPipeline>> {
        let vs = self
            .vs
            .entry_point("main")
            .context("vertex shader: entry point missing")?;
        let fs = self
            .fs
            .entry_point("main")
            .context("fragment shader: entry point missing")?;
        let vertex_input_state =
            BasicVertex::per_vertex().definition(&vs.info().input_interface)?;
        let stages = [
            PipelineShaderStageCreateInfo::new(vs),
            PipelineShaderStageCreateInfo::new(fs),
        ];
        let mut create_info = PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages);
        for layout in create_info.set_layouts.iter_mut() {
            layout.flags |= DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL;
        }
        let layout = PipelineLayout::new(
            ctx.device(),
            create_info.into_pipeline_layout_create_info(ctx.device())?,
        ).map_err(Validated::unwrap)?;
        let subpass = Subpass::from(ctx.render_pass(), 0).context("failed to create subpass")?;

        Ok(GraphicsPipeline::new(
            ctx.device(),
            /* cache= */ None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(vertex_input_state),
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState {
                    viewports: [self.viewport.inner()].into_iter().collect(),
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
            },
        )?)
    }

    fn create_single_command_buffer(
        &self,
        ctx: &VulkanoContext,
        framebuffer: Arc<Framebuffer>,
        uniform_buffer_set: Arc<PersistentDescriptorSet>,
        vertex_buffer: &Subbuffer<[BasicVertex]>,
    ) -> Result<Arc<PrimaryAutoCommandBuffer>> {
        let mut builder = AutoCommandBufferBuilder::primary(
            ctx.command_buffer_allocator(),
            ctx.queue().queue_family_index(),
            CommandBufferUsage::MultipleSubmit,
        )?;

        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some([0.1, 0.1, 0.1, 1.0].into())],
                    ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                },
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )?
            .bind_pipeline_graphics(self.pipeline.clone().unwrap())?
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.clone().unwrap().layout().clone(),
                0,
                uniform_buffer_set.clone(),
            )?
            .bind_vertex_buffers(0, vertex_buffer.clone())?
            .draw(vertex_buffer.len() as u32, 1, 0, 0)?
            .end_render_pass(SubpassEndInfo::default())?;
        Ok(builder.build().map_err(Validated::unwrap)?)
    }


    fn get_or_create_command_buffers(&mut self,
                                     ctx: &VulkanoContext
    ) -> Result<()> {
        if self.pipeline.is_none() {
            self.pipeline = Some(self.create_pipeline(ctx)?);
        }
        if self.uniform_buffers.is_none() {
            self.uniform_buffers = Some(DataPerImage::try_new_with_generator(
                ctx,
                || {
                    let buf = Buffer::new_sized(
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
                    *buf.write()? = UniformData {
                        window_width: self.viewport.physical_width(),
                        window_height: self.viewport.physical_height(),
                        scale_factor: self.viewport.scale_factor(),
                    };
                    Ok(buf)
                })?
            );
        }
        if self.uniform_buffer_sets.is_none() {
            let sampler = Sampler::new(
                ctx.device(),
                SamplerCreateInfo {
                    mag_filter: Filter::Nearest,
                    min_filter: Filter::Nearest,
                    address_mode: [SamplerAddressMode::ClampToBorder; 3],
                    border_color: BorderColor::FloatTransparentBlack,
                    ..Default::default()
                }).map_err(Validated::unwrap)?;
            let mut textures = self.resource_handler.texture.wait_values()
                .into_iter()
                .filter_map(|tex| tex.image_view())
                .collect::<Vec<_>>();
            check_le!(textures.len(), MAX_TEXTURE_COUNT);
            textures.extend(vec![textures.first().unwrap().clone(); MAX_TEXTURE_COUNT - textures.len()]);
            self.uniform_buffer_sets = Some(self.uniform_buffers.as_mut().unwrap().try_map(|buffer| {
                Ok(PersistentDescriptorSet::new(
                    &ctx.descriptor_set_allocator(),
                    self.pipeline.clone().unwrap().layout().set_layouts()[0].clone(),
                    [
                        WriteDescriptorSet::buffer(0, buffer.clone()),
                        WriteDescriptorSet::image_view_sampler_array(
                            1,
                            0,
                            textures.iter().cloned().zip(vec![sampler.clone(); textures.len()])
                        ),
                    ],
                    [],
                ).map_err(Validated::unwrap)?)
            })?);
        }
        self.command_buffers = Some(ctx.framebuffers().try_map_with_3(
            self.uniform_buffer_sets.as_ref().unwrap(),
            self.vertex_buffers.as_ref().unwrap(),
            |((framebuffer, uniform_buffer_set), vertex_buffer)| {
                self.create_single_command_buffer(
                    ctx,
                    framebuffer.clone(),
                    uniform_buffer_set.clone(),
                    vertex_buffer,
                )
            },
        )?);
        Ok(())
    }
}

impl RenderEventHandler<PrimaryAutoCommandBuffer> for BasicRenderHandler {
    type DataReceiver = BasicRenderDataReceiver;

    fn on_resize(
        &mut self,
        _ctx: &VulkanoContext,
        window: Arc<Window>,
    ) -> Result<()> {
        info!("on_resize()");
        self.viewport.update_from_window(window.clone());
        self.vertex_buffers = None;
        self.command_buffers = None;
        self.pipeline = None;
        self.uniform_buffers = None;
        self.uniform_buffer_sets = None;
        self.render_data_receiver.lock().unwrap().viewport = self.viewport.clone();
        Ok(())
    }

    fn on_render(
        &mut self,
        ctx: &VulkanoContext,
        per_image_ctx: &mut MutexGuard<PerImageContext>,
    ) -> Result<DataPerImage<Arc<PrimaryAutoCommandBuffer>>> {
        if let Ok(mut receiver) = self.render_data_receiver.clone().lock() {
            self.get_or_create_vertex_buffers(ctx, &mut receiver)?;
            self.get_or_create_command_buffers(ctx)?;
            self.maybe_update_with_new_vertices(ctx, &mut receiver, per_image_ctx)?;
            self.write_vertex_buffer(&mut receiver, per_image_ctx)?;
        }
        Ok(self.command_buffers.clone().unwrap())
    }

    fn get_receiver(&self) -> Arc<Mutex<BasicRenderDataReceiver>> {
        self.render_data_receiver.clone()
    }
}
