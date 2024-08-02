// Based on: https://github.com/Tenebryo/imgui-vulkano-renderer

use std::sync::Arc;
use anyhow::{Context, Result};
use imgui::{DrawCmd, DrawCmdParams, DrawVert, FontConfig, FontSource, Textures, internal::RawWrapper, DrawList, DrawIdx};
use num_traits::Zero;
use tracing::warn;
use vulkano::{
    descriptor_set::layout::DescriptorSetLayoutCreateFlags,
    command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer},
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    format::Format,
    image::sampler::{BorderColor, Filter, SamplerAddressMode, SamplerCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    pipeline::{
        graphics::{
            viewport::ViewportState,
            vertex_input::VertexDefinition,
            rasterization::RasterizationState,
            multisample::MultisampleState,
            input_assembly::InputAssemblyState,
            GraphicsPipelineCreateInfo,
            color_blend::{AttachmentBlend, ColorBlendAttachmentState, ColorBlendState}
        },
        GraphicsPipeline,
        Pipeline,
        PipelineBindPoint,
        PipelineLayout,
        PipelineShaderStageCreateInfo,
        layout::PipelineDescriptorSetLayoutCreateInfo
    },
    render_pass::Subpass,
    shader::ShaderModule,
    Validated
};
use crate::{
    core::{
        util::UniqueShared,
        vk::{AdjustedViewport, VulkanoContext}
    },
    shader::VkVertex,
    gui::ImGuiContext,
    resource::{
        ResourceHandler,
        texture::Texture
    }
};

pub mod vs {
    vulkano_shaders::shader!{
        ty: "vertex",
        src: r"
        #version 460

        layout(set = 0, binding = 0) uniform VertPC {
            mat4 matrix;
        };

        layout(location = 0) in vec2 pos;
        layout(location = 1) in vec2 uv;
        layout(location = 2) in uint col;

        layout(location = 0) out vec2 f_uv;
        layout(location = 1) out vec4 f_color;

        // Built-in:
        // vec4 gl_Position

        void main() {
            f_uv = uv;
            f_color = unpackUnorm4x8(col);
            gl_Position = matrix * vec4(pos.xy, 0, 1);
        }
        ",
    }
}

pub mod fs {
    vulkano_shaders::shader!{
        ty: "fragment",
        src: r"
        #version 460

        layout(set = 0, binding = 1) uniform sampler2D tex;

        layout(location = 0) in vec2 f_uv;
        layout(location = 1) in vec4 f_color;

        layout(location = 0) out vec4 Target0;

        void main() {
            Target0 = f_color * texture(tex, f_uv.st);
        }
        ",
    }
}

#[derive(BufferContents, VkVertex, Debug, Default, Clone, Copy)]
#[repr(C)]
pub struct ImGuiVertex {
    #[format(R32G32_SFLOAT)]
    pub pos: [f32; 2],
    #[format(R32G32_SFLOAT)]
    pub uv : [f32; 2],
    #[format(R32_UINT)]
    pub col: u32,
}

impl From<DrawVert> for ImGuiVertex {
    fn from(v : DrawVert) -> ImGuiVertex {
        unsafe{std::mem::transmute(v)}
    }
}

type ImGuiVertexBuffer = Subbuffer<[ImGuiVertex]>;
type ImGuiIndexBuffer = Subbuffer<[DrawIdx]>;

pub struct ImGuiRenderer {
    vk_ctx: VulkanoContext,
    _imgui: UniqueShared<ImGuiContext>,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    viewport: UniqueShared<AdjustedViewport>,
    pipeline: Option<Arc<GraphicsPipeline>>,
    resource_handler: ResourceHandler,
    font_texture: Texture,
    textures: Textures<Texture>,
}

impl ImGuiRenderer {
    pub fn new(
        vk_ctx: VulkanoContext,
        imgui: UniqueShared<ImGuiContext>,
        viewport: UniqueShared<AdjustedViewport>,
        resource_handler: ResourceHandler
    ) -> Result<UniqueShared<Self>> {
        let device = vk_ctx.device();
        let textures = Textures::new();

        let imgui_clone = imgui.clone();
        let mut imgui_guard = imgui_clone.get();
        let font_size = 13.0 * viewport.get().scale_factor() as f32;
        imgui_guard.fonts().add_font(&[
            FontSource::DefaultFontData {
                config: Some(FontConfig {
                    size_pixels: font_size,
                    ..FontConfig::default()
                }),
            },
            FontSource::TtfData {
                data: include_bytes!("../../res/mplus-1p-regular.ttf"),
                size_pixels: font_size,
                config: Some(FontConfig {
                    rasterizer_multiply: 1.75,
                    ..FontConfig::default()
                }),
            },
        ]);
        imgui_guard.io_mut().font_global_scale = 1.0 / viewport.get().scale_factor() as f32;
        imgui_guard.set_renderer_name(Some(format!("glongge {}", env!("CARGO_PKG_VERSION"))));

        let texture = imgui_guard.fonts().build_rgba32_texture();
        let font_texture = resource_handler.texture.wait_load_reader_rgba(
            &mut texture.data.to_vec().as_slice(),
            texture.width,
            texture.height,
            Format::R8G8B8A8_SRGB
        )?;
        imgui_guard.fonts().tex_id = imgui::TextureId::from(usize::MAX);
        Ok(UniqueShared::new(Self {
            vk_ctx,
            _imgui: imgui,
            vs: vs::load(device.clone()).context("failed to create shader module")?,
            fs: fs::load(device).context("failed to create shader module")?,
            viewport,
            pipeline: None,
            resource_handler,
            font_texture,
            textures,
        }))
    }
    fn get_or_create_pipeline(&mut self) -> Result<Arc<GraphicsPipeline>> {
        match self.pipeline.clone() {
            None => {
                let vs = self.vs.entry_point("main")
                    .context("vertex shader: entry point missing")?;
                let fs = self.fs.entry_point("main")
                    .context("fragment shader: entry point missing")?;
                let vertex_input_state =
                    ImGuiVertex::per_vertex().definition(&vs.info().input_interface)?;
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
                ).map_err(Validated::unwrap)?;
                let subpass = Subpass::from(self.vk_ctx.render_pass(), 0).context("failed to create subpass")?;

                let pipeline = GraphicsPipeline::new(
                    self.vk_ctx.device(),
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
    fn lookup_texture(&self, texture_id: imgui::TextureId) -> &Texture {
        if texture_id.id() == usize::MAX {
            &self.font_texture
        } else if let Some(texture) = self.textures.get(texture_id) {
            texture
        } else {
            panic!("bad texture_id: {texture_id:?}");
        }
    }
    fn create_vertex_index_buffers(&mut self, draw_list: &DrawList) -> Result<(ImGuiVertexBuffer, ImGuiIndexBuffer)> {
        let vertex_buffer = Buffer::from_iter(
            self.vk_ctx.memory_allocator(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            draw_list.vtx_buffer().iter().map(|&v| ImGuiVertex::from(v))
        ).map_err(Validated::unwrap)?;
        let index_buffer = Buffer::from_iter(
            self.vk_ctx.memory_allocator(),
            BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            draw_list.idx_buffer().iter().copied()
        ).map_err(Validated::unwrap)?;
        Ok((vertex_buffer, index_buffer))
    }
    fn create_uniform_buffer(&mut self, pc: vs::VertPC) -> Result<Subbuffer<vs::VertPC>> {
        let uniform_buffer= Buffer::new_sized(
            self.vk_ctx.memory_allocator(),
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
        *uniform_buffer.write()? = pc;

        Ok(uniform_buffer)
    }
    fn create_uniform_buffer_set(&mut self, pc: vs::VertPC, texture_id: imgui::TextureId) -> Result<Arc<PersistentDescriptorSet>> {
        let pipeline = self.get_or_create_pipeline()?;
        let sampler = vulkano::image::sampler::Sampler::new(
            self.vk_ctx.device(),
            SamplerCreateInfo {
                mag_filter: Filter::Nearest,
                min_filter: Filter::Nearest,
                address_mode: [SamplerAddressMode::ClampToBorder; 3],
                border_color: BorderColor::FloatTransparentBlack,
                ..Default::default()
            }).map_err(Validated::unwrap)?;
        let texture_id = self.lookup_texture(texture_id).id();
        let texture = self.resource_handler.texture
            .get(texture_id).unwrap()
            .ready().unwrap();
        Ok(PersistentDescriptorSet::new(
            &self.vk_ctx.descriptor_set_allocator(),
            pipeline.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, self.create_uniform_buffer(pc)?),
                WriteDescriptorSet::image_view_sampler(
                    1,
                    texture.image_view().unwrap(),
                    sampler
                ),
            ],
            [],
        ).map_err(Validated::unwrap)?)
    }
    pub fn build_render_pass(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        draw_data : &imgui::DrawData
    ) -> Result<()> {
        if draw_data.draw_lists_count().is_zero() {
            warn!("attempted to draw gui, but nothing to draw");
            return Ok(());
        }
        let fb_width = draw_data.display_size[0] * draw_data.framebuffer_scale[0];
        let fb_height = draw_data.display_size[1] * draw_data.framebuffer_scale[1];
        if !(fb_width > 0.0 && fb_height > 0.0) {
            return Ok(());
        }
        let left = draw_data.display_pos[0];
        let right = draw_data.display_pos[0] + draw_data.display_size[0];
        let top = draw_data.display_pos[1];
        let bottom = draw_data.display_pos[1] + draw_data.display_size[1];
        let pc = vs::VertPC {
            matrix: [
                [(2.0 / (right - left)), 0.0, 0.0, 0.0],
                [0.0, (2.0 / (bottom - top)), 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [
                    (right + left) / (left - right),
                    (top + bottom) / (top - bottom),
                    0.0,
                    1.0,
                ],
            ]
        };
        let clip_off = draw_data.display_pos;
        let clip_scale = draw_data.framebuffer_scale;
        for draw_list in draw_data.draw_lists() {
            let (vertex_buffer, index_buffer) = self.create_vertex_index_buffers(draw_list)?;

            for cmd in draw_list.commands() {
                match cmd {
                    DrawCmd::Elements {
                        count: _,
                        cmd_params:
                        DrawCmdParams {
                            clip_rect,
                            texture_id,
                            vtx_offset,
                            idx_offset,
                            ..
                        },
                    } => {
                        let clip_rect = [
                            (clip_rect[0] - clip_off[0]) * clip_scale[0],
                            (clip_rect[1] - clip_off[1]) * clip_scale[1],
                            (clip_rect[2] - clip_off[0]) * clip_scale[0],
                            (clip_rect[3] - clip_off[1]) * clip_scale[1],
                        ];

                        if clip_rect[0] < fb_width
                            && clip_rect[1] < fb_height
                            && clip_rect[2] >= 0.0
                            && clip_rect[3] >= 0.0 {

                            let pipeline = self.get_or_create_pipeline()?;
                            let uniform_buffer_set = self.create_uniform_buffer_set(pc, texture_id)?;
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
                                .bind_index_buffer(index_buffer.clone())?
                                .draw_indexed(
                                    index_buffer.len() as u32 - idx_offset as u32,
                                    1,
                                    idx_offset as u32,
                                    vtx_offset as i32,
                                    0)?;
                        }
                    }
                    DrawCmd::ResetRenderState => (), // TODO
                    DrawCmd::RawCallback { callback, raw_cmd } => unsafe {
                        callback(draw_list.raw(), raw_cmd);
                    },
                }
            }
        }
        Ok(())
    }
}
