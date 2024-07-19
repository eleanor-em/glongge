use std::{
    default::Default,
    sync::{Arc, Mutex, MutexGuard}
};

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
    resource::{
        ResourceHandler,
    },
    shader::sample::{basic_fragment_shader, basic_vertex_shader},
    core::{
        colour::Colour,
        prelude::*,
        vk_core::{
            AdjustedViewport,
            DataPerImage,
            PerImageContext,
            RenderEventHandler,
            VulkanoContext,
            WindowContext,
        },
        RenderInfoFull,
        RenderInfoReceiver,
        VertexWithUV,
    }
};

#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
struct UniformData {
    window_width: f32,
    window_height: f32,
    scale_factor: f32,
}
#[derive(BufferContents, Vertex, Debug, Default, Clone, Copy)]
#[repr(C)]
struct BasicVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
    #[format(R32G32_SFLOAT)]
    translation: [f32; 2],
    #[format(R32_SFLOAT)]
    rotation: f32,
    #[format(R32G32_SFLOAT)]
    scale: [f32; 2],
    #[format(R32G32_SFLOAT)]
    uv: [f32; 2],
    #[format(R32_UINT)]
    texture_id: u32,
    #[format(R32G32B32A32_SFLOAT)]
    blend_col: [f32; 4],
}

#[derive(Clone)]
pub struct BasicRenderInfoReceiver {
    vertices: Vec<VertexWithUV>,
    vertices_up_to_date: DataPerImage<bool>,
    pub(crate) render_info: Vec<RenderInfoFull>,
    viewport: AdjustedViewport,
    clear_col: Colour,
}
impl BasicRenderInfoReceiver {
    fn new(ctx: &VulkanoContext, viewport: AdjustedViewport) -> Self {
        Self {
            vertices: Vec::new(),
            vertices_up_to_date: DataPerImage::new_with_value(ctx, true),
            render_info: Vec::new(),
            viewport,
            clear_col: Colour::black(),
        }
    }
}
impl RenderInfoReceiver for BasicRenderInfoReceiver {
    fn update_vertices(&mut self, vertices: Vec<VertexWithUV>) {
        self.vertices_up_to_date.clone_from_value(false);
        self.vertices = vertices;
    }

    fn update_render_info(&mut self, render_info: Vec<RenderInfoFull>) {
        self.render_info = render_info;
    }

    fn current_viewport(&self) -> AdjustedViewport {
        self.viewport.clone()
    }

    fn is_ready(&self) -> bool {
        !self.vertices.is_empty() && !self.render_info.is_empty()
    }

    fn get_clear_col(&self) -> Colour { self.clear_col }
    fn set_clear_col(&mut self, col: Colour) { self.clear_col = col; }
}

#[derive(Clone)]
pub struct BasicRenderHandler {
    pub(crate) resource_handler: ResourceHandler,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    viewport: AdjustedViewport,
    pipeline: Option<Arc<GraphicsPipeline>>,
    vertex_buffers: Option<DataPerImage<Subbuffer<[BasicVertex]>>>,
    uniform_buffers: Option<DataPerImage<Subbuffer<UniformData>>>,
    uniform_buffer_sets: Option<DataPerImage<Arc<PersistentDescriptorSet>>>,
    command_buffers: Option<DataPerImage<Arc<PrimaryAutoCommandBuffer>>>,
    render_info_receiver: Arc<Mutex<BasicRenderInfoReceiver>>,
}

impl BasicRenderHandler {
    pub fn new(window_ctx: &WindowContext, ctx: &VulkanoContext, resource_handler: ResourceHandler) -> Result<Self> {
        let vs = basic_vertex_shader::load(ctx.device()).context("failed to create shader module")?;
        let fs = basic_fragment_shader::load(ctx.device()).context("failed to create shader module")?;
        let viewport = window_ctx.create_default_viewport();
        let render_info_receiver = Arc::new(Mutex::new(BasicRenderInfoReceiver::new(
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
            render_info_receiver,
        })
    }

    pub fn with_global_scale_factor(mut self, global_scale_factor: f64) -> Self {
        self.viewport.set_global_scale_factor(global_scale_factor);
        self
    }

    fn maybe_create_vertex_buffers(&mut self,
                                   ctx: &VulkanoContext,
                                   receiver: &mut MutexGuard<BasicRenderInfoReceiver>
    ) -> Result<()> {
        if self.vertex_buffers.is_none() {
            self.vertex_buffers = Some(DataPerImage::try_new_with_generator(ctx, || {
                Self::create_single_vertex_buffer(ctx, &receiver.vertices)
            })?);
            self.command_buffers = None;
            receiver.vertices_up_to_date.clone_from_value(true);
        }
        Ok(())
    }
    fn create_single_vertex_buffer(
        ctx: &VulkanoContext,
        vertices: &[VertexWithUV],
    ) -> Result<Subbuffer<[BasicVertex]>> {
        let vertices = vertices.iter().map(|_| BasicVertex {
            blend_col: Colour::magenta().into(),
            ..Default::default()
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
                                      receiver: &mut MutexGuard<BasicRenderInfoReceiver>,
                                      per_image_ctx: &mut MutexGuard<PerImageContext>
    ) -> Result<()> {
        if !per_image_ctx.get_current_value(&receiver.vertices_up_to_date) {
            per_image_ctx.replace_current_value(
                &mut self.vertex_buffers,
                Self::create_single_vertex_buffer(ctx, &receiver.vertices)?);
            let pipeline = self.get_or_create_pipeline(ctx)?;
            let command_buffer = self.create_single_command_buffer(
                ctx,
                receiver,
                pipeline,
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
                           receiver: &mut MutexGuard<BasicRenderInfoReceiver>,
                           per_image_ctx: &mut MutexGuard<PerImageContext>) -> Result<()> {
        let vertex_buffer = per_image_ctx.current_value_as_mut(&mut self.vertex_buffers);
        let mut out_vertices = Vec::new();
        for render_info in receiver.render_info.iter() {
            for vertex_index in render_info.vertex_indices.clone() {
                // Calculate transformed UVs.
                let texture_id = render_info.inner.texture_id.unwrap_or_default();
                let mut blend_col = render_info.inner.col;
                let mut uv = receiver.vertices[vertex_index].uv;
                if let Some(tex) = self.resource_handler.texture.get(texture_id) {
                    if let Some(tex) = tex.ready() {
                        uv = render_info.inner.texture_sub_area.uv(&tex, uv);
                    } else {
                        blend_col = Colour::new(0., 0., 0., 0.);
                    }
                } else {
                    error!("missing texture id: {:?}", texture_id);
                }
                out_vertices.push(BasicVertex {
                    position: receiver.vertices[vertex_index].vertex.into(),
                    uv: uv.into(),
                    texture_id: texture_id.into(),
                    translation: render_info.transform.centre.into(),
                    rotation: render_info.transform.rotation as f32,
                    scale: render_info.transform.scale.into(),
                    blend_col: blend_col.into(),
                });
            }
        }
        vertex_buffer.write()?.swap_with_slice(&mut out_vertices);
        Ok(())
    }

    fn get_or_create_pipeline(&mut self, ctx: &VulkanoContext) -> Result<Arc<GraphicsPipeline>> {
        match self.pipeline.clone() {
            None => {
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

                let pipeline = GraphicsPipeline::new(
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
                )?;
                self.pipeline = Some(pipeline.clone());
                Ok(pipeline)
            },
            Some(pipeline) => Ok(pipeline)
        }
    }

    fn create_single_command_buffer(
        &self,
        ctx: &VulkanoContext,
        receiver: &MutexGuard<BasicRenderInfoReceiver>,
        pipeline: Arc<GraphicsPipeline>,
        framebuffer: Arc<Framebuffer>,
        uniform_buffer_set: Arc<PersistentDescriptorSet>,
        vertex_buffer: &Subbuffer<[BasicVertex]>,
    ) -> Result<Arc<PrimaryAutoCommandBuffer>> {
        let mut builder = AutoCommandBufferBuilder::primary(
            ctx.command_buffer_allocator(),
            ctx.queue().queue_family_index(),
            CommandBufferUsage::MultipleSubmit,
        )?;

        let layout = pipeline.layout().clone();
        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some(receiver.get_clear_col().as_f32().into())],
                    ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                },
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )?
            .bind_pipeline_graphics(pipeline)?
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                layout,
                0,
                uniform_buffer_set.clone(),
            )?
            .bind_vertex_buffers(0, vertex_buffer.clone())?
            .draw(vertex_buffer.len() as u32, 1, 0, 0)?
            .end_render_pass(SubpassEndInfo::default())?;
        Ok(builder.build().map_err(Validated::unwrap)?)
    }

    fn get_or_create_command_buffers(
        &mut self,
        ctx: &VulkanoContext,
        receiver: &MutexGuard<BasicRenderInfoReceiver>)
    -> Result<DataPerImage<Arc<PrimaryAutoCommandBuffer>>> {
        if self.pipeline.is_none() ||
                self.uniform_buffers.is_none() ||
                self.uniform_buffer_sets.is_none() ||
                self.vertex_buffers.is_none() {
            self.command_buffers = None;
        }
        match self.command_buffers.clone() {
            None => {
                let pipeline = self.get_or_create_pipeline(ctx)?;
                let uniform_buffers = self.get_or_create_uniform_buffers(ctx)?;
                let uniform_buffer_sets = self.get_or_create_uniform_buffer_sets(
                    ctx,
                    pipeline.clone(),
                    uniform_buffers)?;
                let command_buffers = ctx.framebuffers().try_map_with_3(
                    &uniform_buffer_sets,
                    self.vertex_buffers.as_ref().unwrap(),
                    |((framebuffer, uniform_buffer_set), vertex_buffer)| {
                        self.create_single_command_buffer(
                            ctx,
                            receiver,
                            pipeline.clone(),
                            framebuffer.clone(),
                            uniform_buffer_set.clone(),
                            vertex_buffer,
                        )
                    },
                )?;
                self.command_buffers = Some(command_buffers.clone());
                Ok(command_buffers)
            },
            Some(command_buffers) => Ok(command_buffers),
        }
    }

    fn get_or_create_uniform_buffer_sets(&mut self,
                                         ctx: &VulkanoContext,
                                         pipeline: Arc<GraphicsPipeline>,
                                         uniform_buffers: DataPerImage<Subbuffer<UniformData>>
    ) -> Result<DataPerImage<Arc<PersistentDescriptorSet>>> {
        Ok(match self.uniform_buffer_sets.clone() {
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
                let uniform_buffer_sets = uniform_buffers.try_map(|buffer| {
                    Ok(PersistentDescriptorSet::new(
                        &ctx.descriptor_set_allocator(),
                        pipeline.layout().set_layouts()[0].clone(),
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
                })?;
                self.uniform_buffer_sets = Some(uniform_buffer_sets.clone());
                uniform_buffer_sets
            },
            Some(uniform_buffer_sets) => uniform_buffer_sets,
        })
    }

    fn get_or_create_uniform_buffers(&mut self, ctx: &VulkanoContext) -> Result<DataPerImage<Subbuffer<UniformData>>> {
        Ok(match self.uniform_buffers.clone() {
            None => {
                let uniform_buffers = DataPerImage::try_new_with_generator(
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
                    })?;
                self.uniform_buffers = Some(uniform_buffers.clone());
                uniform_buffers
            },
            Some(uniform_buffers) => uniform_buffers,
        })
    }
}

impl RenderEventHandler<PrimaryAutoCommandBuffer> for BasicRenderHandler {
    type InfoReceiver = BasicRenderInfoReceiver;

    fn on_resize(
        &mut self,
        _ctx: &VulkanoContext,
        window: Arc<Window>,
    ) -> Result<()> {
        self.viewport.update_from_window(window.clone());
        self.vertex_buffers = None;
        self.command_buffers = None;
        self.pipeline = None;
        self.uniform_buffers = None;
        self.uniform_buffer_sets = None;
        self.render_info_receiver.lock().unwrap().viewport = self.viewport.clone();
        Ok(())
    }

    fn on_render(
        &mut self,
        ctx: &VulkanoContext,
        per_image_ctx: &mut MutexGuard<PerImageContext>,
    ) -> Result<DataPerImage<Arc<PrimaryAutoCommandBuffer>>> {
        let render_info_receiver = self.render_info_receiver.clone();
        let mut receiver = render_info_receiver.lock().unwrap();
        self.maybe_create_vertex_buffers(ctx, &mut receiver)?;
        let command_buffers = self.get_or_create_command_buffers(ctx, &receiver)?;
        self.maybe_update_with_new_vertices(ctx, &mut receiver, per_image_ctx)?;
        self.write_vertex_buffer(&mut receiver, per_image_ctx)?;
        Ok(command_buffers)
    }

    fn on_reload_textures(&mut self, _ctx: &VulkanoContext) -> Result<()> {
        self.uniform_buffer_sets = None;
        Ok(())
    }

    fn get_receiver(&self) -> Arc<Mutex<BasicRenderInfoReceiver>> {
        self.render_info_receiver.clone()
    }
}
