#![feature(iterator_try_collect)]

use std::sync::{Arc, Mutex, MutexGuard};
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
                // Transforms (width=2, height=2) -> (width=1, height=1)
                mat4 window_absolute_scale = mat4(
                    vec4(2, 0, 0, 0),
                    vec4(0, 2, 0, 0),
                    vec4(0, 0, 1, 0),
                    vec4(0, 0, 0, 1));
                // Transforms (width=1, height=1) -> (width=1024, height=768)
                mat4 window_pixel_scale = mat4(
                    vec4(1/1024.0, 0, 0, 0),
                    vec4(0, 1/768.0, 0, 0),
                    vec4(0, 0, 1, 0),
                    vec4(0, 0, 0, 1));
                // Transforms (-1024/2, -768/2) top-left -> (0, 0) top-left
                mat4 window_translation = mat4(
                    vec4(1, 0, 0, 0),
                    vec4(0, 1, 0, 0),
                    vec4(0, 0, 1, 0),
                    vec4(-1, -1, 0, 1));
                mat4 window_scale = window_pixel_scale * window_absolute_scale;
                vec4 world_position = vec4(transform * vec4(position, 0.0, 1.0));
                gl_Position = vec4(window_translation * window_scale * world_position);
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
    run_test_cases();

    let handler = TestRenderHandler::new(&window_ctx, &ctx)?;
    handler.start_update_thread();

    let (event_loop, window) = window_ctx.consume();
    vk_core::WindowEventHandler::new(window, ctx, handler).run(event_loop);
    Ok(())
}

struct RenderData {
    transform: Mat3x3,
}

impl RenderData {
    fn new() -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self { transform: Mat3x3::one() }))
    }
}

struct TestRenderHandler {
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    vertex_buffer: Subbuffer<[TestVertex]>,
    uniform_buffers: DataPerImage<Subbuffer<[[f32; 4]; 4]>>,
    viewport: Viewport,
    command_buffers: Option<DataPerImage<Arc<PrimaryAutoCommandBuffer>>>,
    render_data: Arc<Mutex<RenderData>>,
    pos: Vec2,
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
        let transform = self.render_data.lock().unwrap().transform;
        *self.uniform_buffers.current_value(per_image_ctx).write()? = transform.transposed().into();
        Ok(self.command_buffers.clone().unwrap())
    }
}

impl TestRenderHandler {
    fn create_vertices() -> (Vec2, Vec<Vec2>) {
        let tri_width = 100.0;
        let tri_height = tri_width * 3.0.sqrt();
        let centre_correction = -tri_height / 6.0;
        let vertex1 = Vec2 {
            x: -tri_width,
            y: -tri_height / 2.0 - centre_correction,
        };
        let vertex2 = Vec2 {
            x: tri_width,
            y: -tri_height / 2.0 - centre_correction,
        };
        let vertex3 = Vec2 {
            x: 0.0,
            y: tri_height / 2.0 - centre_correction,
        };
        let pos = Vec2 {
            x: (vertex1.x + vertex2.x + vertex3.x) / 3.0,
            y: (vertex1.y + vertex2.y + vertex3.y) / 3.0,
        };
        (pos, vec![vertex1, vertex2, vertex3])
    }

    fn start_update_thread(&self) {
        let render_data = self.render_data.clone();
        let mut pos = self.pos;

        std::thread::spawn(move || {
            let mut rng = rand::thread_rng();
            let mut t = 0.0;
            let world_pos_offset = Vec2 { x: 512.0, y: 384.0 };
            let mut velocity = Vec2 {
                x: rng.gen_range(-1.0..1.0),
                y: rng.gen_range(-1.0..1.0),
            }
                .normed() * 200.0;
            let mut delta = 0.0;
            loop {
                let now = Instant::now();
                t += delta;
                let next_world_pos = pos + world_pos_offset + velocity * delta;
                if !(0.0..1024.0).contains(&next_world_pos.x) {
                    velocity.x = -velocity.x;
                }
                if !(0.0..768.0).contains(&next_world_pos.y) {
                    velocity.y = -velocity.y;
                }
                pos += velocity * delta;
                let radians = 1.0 * f64::PI() * t;
                let rotation = Mat3x3::translation_vec2(pos)
                    * Mat3x3::rotation(radians)
                    * Mat3x3::translation_vec2(-pos);
                let translation = Mat3x3::translation_vec2(pos);
                let world = Mat3x3::translation_vec2(world_pos_offset);
                render_data.lock().unwrap().transform = world * rotation * translation;
                thread::sleep(Duration::from_micros(rng.gen_range(500..5_000)));
                delta = now.elapsed().as_secs_f64();
            }
        });
    }

    fn new(window_ctx: &vk_core::WindowContext, ctx: &VulkanoContext) -> Result<Self> {
        let vs = main_test_vertex_shader::load(ctx.device())
            .context("failed to create shader module")?;
        let fs = main_test_fragment_shader::load(ctx.device())
            .context("failed to create shader module")?;
        let (pos, _) = Self::create_vertices();
        let vertex_buffer = Self::create_vertex_buffer(ctx)?;
        let uniform_buffers = Self::create_uniform_buffers(ctx)?;
        let viewport = window_ctx.create_default_viewport();
        let render_data = RenderData::new();
        Ok(Self {
            vs,
            fs,
            vertex_buffer,
            uniform_buffers,
            viewport,
            command_buffers: None,
            render_data,
            pos,
        })
    }

    fn create_vertex_buffer(ctx: &VulkanoContext) -> Result<Subbuffer<[TestVertex]>> {
        let (_, vertices) = Self::create_vertices();
        let vertices = vertices
            .into_iter()
            .map(|v| TestVertex { position: v.into() });
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

fn run_test_cases() {
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
}
