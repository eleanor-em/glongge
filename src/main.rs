#![feature(iterator_try_collect)]

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, MutexGuard};
use std::thread;
use std::time::{Duration, Instant};

use rand::Rng;

use anyhow::{Context, Result};
use num_traits::{Float, FloatConst, One};

use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, RenderPassBeginInfo,
    SubpassBeginInfo, SubpassContents, SubpassEndInfo,
};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
};
use vulkano::render_pass::Subpass;
use vulkano::shader::ShaderModule;
use winit::window::Window;

mod assert;
mod linalg;
mod util;
mod vk_core;

use crate::vk_core::{DataPerImage, PerImageContext};
use linalg::{Mat3x3, Vec2};
use vk_core::VulkanoContext;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .event_format(
            tracing_subscriber::fmt::format()
                .with_target(false)
                .with_file(true)
                .with_line_number(true),
        )
        .init();

    let window_ctx = vk_core::WindowContext::new()?;
    let ctx = vk_core::VulkanoContext::new(&window_ctx)?;
    main_test(window_ctx, ctx)
}

// this is the "main" test (i.e. used for active dev)
#[derive(BufferContents, Vertex, Clone, Copy)]
#[repr(C)]
struct TestVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}
mod main_test_vertex_shader {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
            #version 460

            layout(location = 0) in vec2 position;
            layout(set = 0, binding = 0) uniform Data {
                mat4 transform;
            };

            void main() {
                gl_Position = transform * vec4(position, 0.0, 1.0);
            }
        ",
    }
}
mod main_test_fragment_shader {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
            #version 460

            layout(location = 0) out vec4 f_color;

            void main() {
                f_color = vec4(1.0, 0.0, 0.0, 1.0);
            }
        ",
    }
}
fn main_test(window_ctx: vk_core::WindowContext, ctx: vk_core::VulkanoContext) -> Result<()> {
    let handler = TestRenderHandler::new(&window_ctx, &ctx)?;

    // TODO: proper test cases...
    let a = Vec2 { x: 1.0, y: 1.0 };
    crate::check!((a * 2.0).almost_eq(Vec2 { x: 2.0, y: 2.0 }));
    crate::check!((2.0 * a).almost_eq(Vec2 { x: 2.0, y: 2.0 }));
    crate::check_lt!(f64::abs((a * 2.0 - a).x - 1.0), f64::epsilon());
    crate::check_lt!(f64::abs((a * 2.0 - a).y - 1.0), f64::epsilon());
    crate::check!(
        (Mat3x3::rotation(-1.0) * Mat3x3::rotation(0.5) * Mat3x3::rotation(0.5))
            .almost_eq(Mat3x3::one())
    );

    let (event_loop, window) = window_ctx.consume();
    vk_core::WindowEventHandler::new(window, ctx, handler).run(event_loop);
    Ok(())
}

struct TestRenderHandler {
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    vertex_buffer: Subbuffer<[TestVertex]>,
    uniform_buffers: DataPerImage<Subbuffer<[[f32; 4]; 4]>>,
    viewport: Viewport,
    command_buffers: Option<DataPerImage<Arc<PrimaryAutoCommandBuffer>>>,
    t: Arc<AtomicU64>,
}

impl vk_core::RenderEventHandler<PrimaryAutoCommandBuffer> for TestRenderHandler {
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
        _ctx: &vk_core::VulkanoContext,
        per_image_ctx: &mut MutexGuard<PerImageContext>,
    ) -> Result<DataPerImage<Arc<PrimaryAutoCommandBuffer>>> {
        let t_secs = self.t.load(Ordering::Acquire) as f64 / 1_000_000.0;
        let radians = 2.0 * f64::PI() * t_secs;
        let transform = Mat3x3::rotation(radians);
        *self.uniform_buffers.current_value(per_image_ctx).write()? = transform.into();
        Ok(self.command_buffers.clone().unwrap())
    }
}

impl TestRenderHandler {
    fn new(window_ctx: &vk_core::WindowContext, ctx: &VulkanoContext) -> Result<Self> {
        let vs = main_test_vertex_shader::load(ctx.device())
            .context("failed to create shader module")?;
        let fs = main_test_fragment_shader::load(ctx.device())
            .context("failed to create shader module")?;

        let vertex1 = Vec2 { x: -0.5, y: -0.5 };
        let vertex2 = Vec2 { x: 0.0, y: 0.5 };
        let vertex3 = Vec2 { x: 0.5, y: -0.25 };
        let vertices = vec![vertex1, vertex2, vertex3]
            .into_iter()
            .map(|v| TestVertex { position: v.into() });
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
        )?;

        let uniform_buffers = DataPerImage::new_with_value(
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
        );

        let viewport = window_ctx.create_default_viewport();

        let t = Arc::new(AtomicU64::default());
        let t_update = t.clone();
        std::thread::spawn(move || {
            let mut delta = 0u64;
            let mut rng = rand::thread_rng();
            loop {
                let now = Instant::now();
                t_update.fetch_add(delta, Ordering::Release);
                thread::sleep(Duration::from_micros(rng.gen_range(500..5_000)));
                delta = now.elapsed().as_micros() as u64;
            }
        });
        Ok(Self {
            vs,
            fs,
            vertex_buffer,
            uniform_buffers,
            viewport,
            command_buffers: None,
            t,
        })
    }

    fn create_pipeline(&self, ctx: &vk_core::VulkanoContext) -> Result<Arc<GraphicsPipeline>> {
        let vs = self
            .vs
            .entry_point("main")
            .context("vertex shader: entry point missing")?;
        let fs = self
            .fs
            .entry_point("main")
            .context("fragment shader: entry point missing")?;
        let vertex_input_state = TestVertex::per_vertex().definition(&vs.info().input_interface)?;
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

    fn create_command_buffers(&mut self, ctx: &vk_core::VulkanoContext) -> Result<()> {
        let pipeline = self.create_pipeline(ctx)?;
        let uniform_buffer_sets = self.uniform_buffers.try_map(|buffer| {
            Ok(PersistentDescriptorSet::new(
                &ctx.descriptor_set_allocator(),
                pipeline.layout().set_layouts()[0].clone(),
                [WriteDescriptorSet::buffer(0, buffer.clone())],
                [],
            )?)
        })?;
        let command_buffers = ctx.framebuffers().try_map_with(
            &uniform_buffer_sets,
            |(framebuffer, uniform_buffer_set)| {
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
                    .bind_vertex_buffers(0, self.vertex_buffer.clone())?
                    .draw(self.vertex_buffer.len() as u32, 1, 0, 0)?
                    .end_render_pass(SubpassEndInfo::default())?;
                Ok(builder.build()?)
            },
        )?;
        self.command_buffers = Some(command_buffers);
        Ok(())
    }
}
