use std::cell::RefCell;
use std::sync::{Arc, Mutex, MutexGuard};
use num_traits::{One, Zero};

use anyhow::{Context, Result};

use vulkano::{
    pipeline::{
        graphics::{
            vertex_input::{Vertex, VertexDefinition},
            rasterization::RasterizationState,
            multisample::MultisampleState,
            input_assembly::InputAssemblyState,
            GraphicsPipelineCreateInfo,
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            viewport::{Viewport, ViewportState}
        },
        GraphicsPipeline,
        Pipeline,
        PipelineBindPoint,
        PipelineLayout,
        PipelineShaderStageCreateInfo,
        layout::PipelineDescriptorSetLayoutCreateInfo
    },
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassBeginInfo, SubpassContents, SubpassEndInfo},
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    render_pass::Subpass,
    shader::ShaderModule
};
use winit::window::Window;

use crate::{
    core::{
        linalg::{Mat3x3, Vec2},
        vk_core::{DataPerImage, PerImageContext, RenderEventHandler, VulkanoContext, WindowContext},
    },
    gg::core::{RenderData, SceneObject},
    shader::sample::{basic_fragment_shader, basic_vertex_shader},
};
use crate::gg::core::{RenderDataReceiver, UpdateHandler};

#[derive(BufferContents, Vertex, Clone, Copy)]
#[repr(C)]
struct BasicVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}

struct BasicRenderDataReceiver(Vec<RenderData>);
impl RenderDataReceiver for BasicRenderDataReceiver {
    fn update_from(&mut self, render_data: &[RenderData]) -> Result<()> {
        let max_size = self.0.len();
        self.0.clone_from_slice(&render_data[0..max_size]);
        Ok(())
    }

    fn cloned(&self) -> Vec<RenderData> {
        self.0.clone()
    }
}

pub struct BasicRenderHandler {
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    vertices: Vec<Vec2>,
    vertex_buffers: DataPerImage<Subbuffer<[BasicVertex]>>,
    uniform_buffers: DataPerImage<Subbuffer<[[f32; 4]; 4]>>,
    viewport: Viewport,
    command_buffers: Option<DataPerImage<Arc<PrimaryAutoCommandBuffer>>>,
    update_handler: Option<UpdateHandler<BasicRenderDataReceiver>>,
    render_data_receiver: Arc<Mutex<BasicRenderDataReceiver>>,
}

impl RenderEventHandler<PrimaryAutoCommandBuffer> for BasicRenderHandler {
    fn on_resize(
        &mut self,
        ctx: &VulkanoContext,
        _per_image_ctx: &mut MutexGuard<PerImageContext>,
        window: Arc<Window>,
    ) -> Result<()> {
        self.viewport.extent = window.inner_size().into();
        self.create_command_buffers(ctx)?;
        Ok(())
    }

    fn on_render(
        &mut self,
        ctx: &VulkanoContext,
        per_image_ctx: &mut MutexGuard<PerImageContext>,
    ) -> Result<DataPerImage<Arc<PrimaryAutoCommandBuffer>>> {
        if let Ok(render_data) = self.render_data_receiver.lock() {
            let vertex_buffer = self.vertex_buffers.current_value_mut(per_image_ctx);
            for (i, vertex) in vertex_buffer.write()?.iter_mut().enumerate() {
                *vertex = BasicVertex {
                    position: (Mat3x3::translation_vec2(render_data.0[i/3].position)
                        * Mat3x3::rotation(render_data.0[i/3].rotation)
                        * self.vertices[i]).into()
                };
            }
        }
        *self.uniform_buffers.current_value(per_image_ctx).write()? = Mat3x3::one().into();
        Ok(self.command_buffers.clone().unwrap())
    }
}

impl BasicRenderHandler {
    pub fn new(objects: Vec<RefCell<Box<dyn SceneObject>>>, window_ctx: &WindowContext, ctx: &VulkanoContext) -> Result<Self> {
        // 3 vertices per object is a safe lower bound.
        let mut vertices = Vec::with_capacity(3 * objects.len());
        for object in objects.iter() {
            vertices.append(&mut object.borrow().create_vertices());
        }
        let render_data = vec![RenderData { position: Vec2::zero(), rotation: 0.0 }; objects.len()];
        let render_data_receiver = Arc::new(Mutex::new(BasicRenderDataReceiver(render_data)));

        let vs = basic_vertex_shader::load(ctx.device())
            .context("failed to create shader module")?;
        let fs = basic_fragment_shader::load(ctx.device())
            .context("failed to create shader module")?;
        let vertex_buffers = Self::create_vertex_buffers(ctx, &vertices)?;
        let uniform_buffers = Self::create_uniform_buffers(ctx)?;
        let viewport = window_ctx.create_default_viewport();
        Ok(Self {
            vs,
            fs,
            vertices,
            vertex_buffers,
            uniform_buffers,
            viewport,
            command_buffers: None,
            update_handler: Some(UpdateHandler { objects, render_data_receiver: render_data_receiver.clone() }),
            render_data_receiver,
        })
    }

    pub fn start_update_thread(&mut self) {
        let update_handler = self.update_handler.take().unwrap();
        std::thread::spawn(move || update_handler.consume());
    }

    fn create_vertex_buffers(ctx: &VulkanoContext, vertices: &[Vec2]) -> Result<DataPerImage<Subbuffer<[BasicVertex]>>> {
        let vertices = vertices
            .iter()
            .map(|&v| BasicVertex { position: v.into() });
        Ok(DataPerImage::new_with_value(
            ctx,
            Buffer::from_iter(
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
            )?))
    }

    fn create_uniform_buffers(ctx: &VulkanoContext) -> Result<DataPerImage<Subbuffer<[[f32; 4]; 4]>>> {
        Ok(DataPerImage::new_with_value(
            ctx,
            Buffer::new_sized::<[[f32; 4]; 4]>(
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
            )?,
        ))
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
        let vertex_input_state = BasicVertex::per_vertex().definition(&vs.info().input_interface)?;
        let stages = [
            PipelineShaderStageCreateInfo::new(vs),
            PipelineShaderStageCreateInfo::new(fs),
        ];
        let layout = PipelineLayout::new(
            ctx.device(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(ctx.device())?,
        )?;
        let subpass = Subpass::from(ctx.render_pass(), 0).context("failed to create subpass")?;

        Ok(GraphicsPipeline::new(
            ctx.device(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(vertex_input_state),
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState {
                    viewports: [self.viewport.clone()].into_iter().collect(),
                    ..Default::default()
                }),
                rasterization_state: Some(RasterizationState::default()),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    subpass.num_color_attachments(),
                    ColorBlendAttachmentState::default(),
                )),
                subpass: Some(subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )?)
    }

    fn create_command_buffers(&mut self, ctx: &VulkanoContext) -> Result<()> {
        let pipeline = self.create_pipeline(ctx)?;
        let uniform_buffer_sets = self.uniform_buffers.try_map(|buffer| {
            Ok(PersistentDescriptorSet::new(
                &ctx.descriptor_set_allocator(),
                pipeline.layout().set_layouts()[0].clone(),
                [WriteDescriptorSet::buffer(0, buffer.clone())],
                [],
            )?)
        })?;
        let command_buffers = ctx.framebuffers().try_map_with_3(
            &uniform_buffer_sets,
            &self.vertex_buffers,
            |((framebuffer, uniform_buffer_set), vertex_buffer)| {
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
                    .bind_pipeline_graphics(pipeline.clone())?
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        pipeline.layout().clone(),
                        0,
                        uniform_buffer_set.clone(),
                    )?
                    .bind_vertex_buffers(0, vertex_buffer.clone())?
                    .draw(vertex_buffer.len() as u32, 1, 0, 0)?
                    .end_render_pass(SubpassEndInfo::default())?;
                Ok(builder.build()?)
            },
        )?;
        self.command_buffers = Some(command_buffers);
        Ok(())
    }
}
