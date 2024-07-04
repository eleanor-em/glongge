use std::cell::RefCell;
use std::sync::{Arc, Mutex, MutexGuard};
use num_traits::{One, Zero};

use anyhow::{Context, Result};
use tracing::info;

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
use vulkano::render_pass::Framebuffer;
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
    #[format(R32G32_SFLOAT)]
    translation: [f32; 2],
    #[format(R32_SFLOAT)]
    rotation: f32,
}

struct BasicRenderDataReceiver {
    vertices: Vec<Vec2>,
    vertices_up_to_date: DataPerImage<bool>,
    render_data: Vec<RenderData>,
}
impl BasicRenderDataReceiver {
    fn new(ctx: &VulkanoContext, vertices: Vec<Vec2>, render_data: Vec<RenderData>) -> Self {
        Self {
            vertices,
            vertices_up_to_date: DataPerImage::new_with_value(ctx, true),
            render_data,
        }
    }
}
impl RenderDataReceiver for BasicRenderDataReceiver {
    fn replace_vertices(&mut self, vertices: Vec<Vec2>) {
        self.vertices_up_to_date.clone_from_value(false);
        self.vertices = vertices;
    }

    fn update_render_data(&mut self, render_data: Vec<RenderData>) {
        self.render_data = render_data;
    }

    fn clone_data(&self) -> (Vec<Vec2>, Vec<RenderData>) {
        (self.vertices.clone(), self.render_data.clone())
    }
}

pub struct BasicRenderHandler {
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    vertex_buffers: DataPerImage<Subbuffer<[BasicVertex]>>,
    uniform_buffers: DataPerImage<Subbuffer<[[f32; 4]; 4]>>,
    uniform_buffer_sets: Option<DataPerImage<Arc<PersistentDescriptorSet>>>,
    viewport: Viewport,
    pipeline: Option<Arc<GraphicsPipeline>>,
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
        if let Ok(mut receiver) = self.render_data_receiver.lock() {
            if !receiver.vertices_up_to_date.current_value(per_image_ctx) {
                *self.vertex_buffers.current_value_mut(per_image_ctx) = Self::create_single_vertex_buffer(ctx, &receiver.vertices)?;
                *self.command_buffers.as_mut().unwrap().current_value_mut(per_image_ctx) =
                    self.create_single_command_buffer(
                        ctx,
                        ctx.framebuffers().current_value(per_image_ctx).clone(),
                        self.uniform_buffer_sets.clone().unwrap().current_value(per_image_ctx).clone(),
                        self.vertex_buffers.current_value(per_image_ctx)
                    )?;
                *receiver.vertices_up_to_date.current_value_mut(per_image_ctx) = true;
            }

            let vertex_buffer = self.vertex_buffers.current_value_mut(per_image_ctx);
            for (i, vertex) in vertex_buffer.write()?.iter_mut().enumerate() {
                *vertex = BasicVertex {
                    position: receiver.vertices[i].into(),
                    translation: receiver.render_data[i/3].position.into(),
                    rotation: receiver.render_data[i/3].rotation as f32,
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
        let render_data_receiver = Arc::new(Mutex::new(
            BasicRenderDataReceiver::new(ctx, vertices.clone(), render_data)));

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
            vertex_buffers,
            uniform_buffers,
            uniform_buffer_sets: None,
            viewport,
            pipeline: None,
            command_buffers: None,
            update_handler: Some(UpdateHandler { objects, render_data_receiver: render_data_receiver.clone() }),
            render_data_receiver,
        })
    }

    pub fn start_update_thread(&mut self) {
        let update_handler = self.update_handler.take().unwrap();
        std::thread::spawn(move || update_handler.consume());
    }

    fn create_single_vertex_buffer(ctx: &VulkanoContext, vertices: &[Vec2]) -> Result<Subbuffer<[BasicVertex]>> {
        let vertices = vertices
            .iter()
            .map(|&v| BasicVertex { position: v.into(), translation: Vec2::zero().into(), rotation: 0.0 });
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
        )?)
    }
    fn create_vertex_buffers(ctx: &VulkanoContext, vertices: &[Vec2]) -> Result<DataPerImage<Subbuffer<[BasicVertex]>>> {
        Ok(DataPerImage::new_with_value(ctx, Self::create_single_vertex_buffer(ctx, vertices)?))
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

    fn create_single_command_buffer(&self,
                                    ctx: &VulkanoContext,
                                    framebuffer: Arc<Framebuffer>,
                                    uniform_buffer_set: Arc<PersistentDescriptorSet>,
                                    vertex_buffer: &Subbuffer<[BasicVertex]>) -> Result<Arc<PrimaryAutoCommandBuffer>> {
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
        Ok(builder.build()?)
    }

    fn create_command_buffers(&mut self, ctx: &VulkanoContext) -> Result<()> {
        if self.pipeline.is_none() {
            self.pipeline = Some(self.create_pipeline(ctx)?);
        }
        let uniform_buffer_sets = self.uniform_buffers.try_map(|buffer| {
            Ok(PersistentDescriptorSet::new(
                &ctx.descriptor_set_allocator(),
                self.pipeline.clone().unwrap().layout().set_layouts()[0].clone(),
                [WriteDescriptorSet::buffer(0, buffer.clone())],
                [],
            )?)
        })?;
        let command_buffers = ctx.framebuffers().try_map_with_3(
            &uniform_buffer_sets,
            &self.vertex_buffers,
            |((framebuffer, uniform_buffer_set), vertex_buffer)| {
                self.create_single_command_buffer(ctx, framebuffer.clone(), uniform_buffer_set.clone(), vertex_buffer)
            },
        )?;
        self.uniform_buffer_sets = Some(uniform_buffer_sets);
        self.command_buffers = Some(command_buffers);
        Ok(())
    }
}
