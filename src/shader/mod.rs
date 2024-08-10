use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use anyhow::{Context, Result};
use itertools::Itertools;
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
use vulkano::pipeline::graphics::rasterization::PolygonMode;
use crate::{
    core::{
        prelude::*,
        vk::{AdjustedViewport, VulkanoContext},
    },
    shader::glsl::{basic, sprite},
    util::UniqueShared
};
pub use vulkano::pipeline::graphics::vertex_input::Vertex as VkVertex;
use crate::core::render::ShaderRenderFrame;

pub mod vertex;
pub mod glsl;

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub struct ShaderName(&'static str);

impl ShaderName {
    pub fn new(text: &'static str) -> Self {
        Self(text)
    }
}

// Shader ID 0 is reserved for an error code.
static NEXT_SHADER_ID: AtomicU8 = AtomicU8::new(1);
#[derive(Copy, Clone, Debug, Default, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct ShaderId(u8);
impl ShaderId {
    fn next() -> Self {
        ShaderId(NEXT_SHADER_ID.fetch_add(1, Ordering::Relaxed))
    }

    pub(crate) fn is_valid(self) -> bool { self.0 != 0 }
}
static SHADER_IDS_INIT: LazyLock<Arc<Mutex<HashMap<ShaderName, ShaderId>>>> = LazyLock::new(|| {
    Arc::new(Mutex::new(HashMap::new()))
});
static SHADERS_LOCKED: AtomicBool = AtomicBool::new(false);

static SHADER_IDS_FINAL: LazyLock<HashMap<ShaderName, ShaderId>> = LazyLock::new(|| {
    check!(SHADERS_LOCKED.load(Ordering::Acquire),
           "attempted to load shader IDs too early");
    let shader_ids = SHADER_IDS_INIT.lock().unwrap();
    shader_ids.clone()
});

pub fn register_shader<S: Shader + Sized>() -> ShaderId {
    check_false!(SHADERS_LOCKED.load(Ordering::Acquire),
                 format!("attempted to register shader too late: {:?}", S::name()));
    let mut shader_ids = SHADER_IDS_INIT.lock().unwrap();
    *shader_ids.entry(S::name())
        .or_insert_with(ShaderId::next)
}
pub fn get_shader(name: ShaderName) -> ShaderId {
    SHADER_IDS_FINAL.get(&name).copied()
        .unwrap_or_else(|| {
            error!("unknown shader: {name:?}");
            ShaderId::default()
        })
}
pub(crate) fn ensure_shaders_locked() {
    SHADERS_LOCKED.swap(true, Ordering::Release);
}

pub trait Shader: Send {
    fn name() -> ShaderName where Self: Sized;
    fn name_concrete(&self) -> ShaderName;
    fn id(&self) -> ShaderId { get_shader(self.name_concrete()) }
    fn on_render(
        &mut self,
        render_frame: ShaderRenderFrame
    ) -> Result<()>;
    fn build_render_pass(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>
    ) -> Result<()>;
}

#[derive(Clone)]
struct CachedVertexBuffer<T: VkVertex + Copy> {
    ctx: VulkanoContext,
    inner: Subbuffer<[T]>,
    vertex_count: usize,
    begin: usize,
}

impl<T: VkVertex + Copy> CachedVertexBuffer<T> {
    fn new(ctx: VulkanoContext, size: usize) -> Result<Self> {
        let inner = Self::create_vertex_buffer(
            &ctx,
            (size * ctx.framebuffers().len()) as DeviceSize
        )?;
        Ok(Self { ctx, inner, vertex_count: 0, begin: 0 })
    }

    fn len(&self) -> usize { self.inner.len() as usize / self.ctx.framebuffers().len() }

    fn realloc(&mut self) -> Result<()> {
        let size = self.inner.len() as usize * std::mem::size_of::<T>();
        if size / 1024 / 1024 == 0 {
            warn!("reallocating vertex buffer: {} KiB -> {} KiB",
                size / 1024 , size * 2 / 1024);
        } else {
            warn!("reallocating vertex buffer: {} MiB -> {} MiB",
                size / 1024 / 1024, size * 2 / 1024 / 1024);
        }
        self.inner = Self::create_vertex_buffer(
            &self.ctx,
            (self.inner.len() * 2) as DeviceSize
        )?;
        Ok(())
    }

    fn write(&mut self, data: &[T]) -> Result<()> {
        let mut buf_len = usize::try_from(self.inner.len())
            .with_context(|| format!("self.inner.len() overflowed: {}", self.inner.len()))?;
        self.begin += self.len();
        if self.begin > buf_len {
            bail!("self.begin accounting wrong? {} > {buf_len}", self.begin)
        }
        if self.begin == buf_len {
            self.begin = 0;
        }
        while self.begin + data.len() > buf_len {
            buf_len = usize::try_from(self.inner.len())
                .with_context(|| format!("self.inner.len() overflowed: {}", self.inner.len()))?;
            self.realloc()?;
            self.begin = 0;
        }
        self.inner.write()?[self.begin..self.begin + data.len()].copy_from_slice(data);

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
            .with_context(|| format!("tried to draw too many vertices: {}", self.vertex_count))?;
        let begin = u32::try_from(self.begin)
            .with_context(|| format!("self.begin overflowed: {}", self.begin))?;
        let buf_len = u32::try_from(self.inner.len())
            .with_context(|| format!("self.inner.len() overflowed: {}", self.inner.len()))?;
        if begin + vertex_count >= buf_len {
            bail!("too many vertices: {begin} + {vertex_count} = {} >= {buf_len}",
                       begin + vertex_count);
        }
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
    ) -> Result<UniqueShared<Box<dyn Shader>>> {
        register_shader::<Self>();
        let device = ctx.device();
        let vertex_buffer = CachedVertexBuffer::new(ctx.clone(), 10_000)?;
        Ok(UniqueShared::new(Box::new(Self {
            ctx,
            vs: sprite::vertex_shader::load(device.clone()).context("failed to create shader module")?,
            fs: sprite::fragment_shader::load(device).context("failed to create shader module")?,
            viewport,
            pipeline: None,
            vertex_buffer,
            resource_handler,
        }) as Box<dyn Shader>))
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
    fn create_uniform_desc_set(&mut self) -> Result<Arc<PersistentDescriptorSet>> {
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
                WriteDescriptorSet::image_view_sampler_array(
                    0,
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
        ShaderName::new("sprite")
    }
    fn name_concrete(&self) -> ShaderName { Self::name() }

    fn on_render(
        &mut self,
        render_frame: ShaderRenderFrame
    ) -> Result<()> {
        let render_infos = render_frame.render_infos
            .iter()
            .sorted_unstable_by_key(|item| item.depth);
        let mut vertices = Vec::with_capacity(self.vertex_buffer.len());
        for render_info in render_infos {
            for vertex_index in render_info.vertex_indices.clone() {
                // Calculate transformed UVs.
                let vertex = render_frame.vertices[vertex_index as usize];
                for ri in &render_info.inner {
                    let mut blend_col = ri.col;
                    let mut uv = vertex.uv;
                    if let Some(tex) = self.resource_handler.texture.get_nonblank(ri.texture_id) {
                        if let Some(tex) = tex.ready() {
                            uv = ri.texture_sub_area.uv(&tex, uv.into()).into();
                        } else {
                            warn!("texture not ready: {}", ri.texture_id);
                            blend_col = Colour::empty().into();
                        }
                    } else {
                        error!("missing texture: {}", ri.texture_id);
                        blend_col = Colour::magenta().into();
                    }

                    vertices.push(sprite::Vertex {
                        position: vertex.xy,
                        uv,
                        texture_id: ri.texture_id,
                        translation: render_info.transform.centre,
                        rotation: render_info.transform.rotation,
                        scale: render_info.transform.scale,
                        blend_col,
                    });
                }
            }
        }
        self.vertex_buffer.write(&vertices)?;
        Ok(())
    }
    fn build_render_pass(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>
    ) -> Result<()> {
        if self.vertex_buffer.vertex_count == 0 { return Ok(()); }
        let pipeline = self.get_or_create_pipeline()?;
        let uniform_desc_set = self.create_uniform_desc_set()?;
        let layout = pipeline.layout().clone();
        let viewport = self.viewport.get();
        let pc = sprite::vertex_shader::WindowData {
            #[allow(clippy::cast_possible_truncation)]
            window_width: viewport.physical_width() as f32,
            #[allow(clippy::cast_possible_truncation)]
            window_height: viewport.physical_height() as f32,
            #[allow(clippy::cast_possible_truncation)]
            scale_factor: viewport.scale_factor() as f32,
        };
        builder
            .bind_pipeline_graphics(pipeline.clone())?
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                layout.clone(),
                0,
                uniform_desc_set.clone(),
            )?
            .push_constants(layout, 0, pc)?;
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
    vertex_buffer: CachedVertexBuffer<basic::Vertex>,
}

impl WireframeShader {
    pub fn new(
        ctx: VulkanoContext,
        viewport: UniqueShared<AdjustedViewport>
    ) -> Result<UniqueShared<Box<dyn Shader>>> {
        register_shader::<Self>();
        let device = ctx.device();
        let vertex_buffer = CachedVertexBuffer::new(ctx.clone(), 10_000)?;
        Ok(UniqueShared::new(Box::new(Self {
            ctx,
            vs: basic::vertex_shader::load(device.clone()).context("failed to create shader module")?,
            fs: basic::fragment_shader::load(device).context("failed to create shader module")?,
            viewport,
            pipeline: None,
            vertex_buffer,
        }) as Box<dyn Shader>))
    }

    fn get_or_create_pipeline(&mut self) -> Result<Arc<GraphicsPipeline>> {
        match self.pipeline.clone() {
            None => {
                let vs = self.vs.entry_point("main")
                    .context("vertex shader: entry point missing")?;
                let fs = self.fs.entry_point("main")
                    .context("fragment shader: entry point missing")?;
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
                        rasterization_state: Some(RasterizationState {
                            polygon_mode: PolygonMode::Line,
                            ..RasterizationState::default()
                        }),
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
}

impl Shader for WireframeShader {
    fn name() -> ShaderName
    where
        Self: Sized
    {
        ShaderName::new("wireframe")
    }
    fn name_concrete(&self) -> ShaderName { Self::name() }

    fn on_render(
        &mut self,
        render_frame: ShaderRenderFrame,
    ) -> Result<()> {
        let mut vertices = Vec::with_capacity(self.vertex_buffer.len());
        for render_info in &render_frame.render_infos {
            for vertex_index in render_info.vertex_indices.clone() {
                for ri in &render_info.inner {
                    vertices.push(basic::Vertex {
                        position: render_frame.vertices[vertex_index as usize].xy,
                        translation: render_info.transform.centre,
                        rotation: render_info.transform.rotation,
                        scale: render_info.transform.scale,
                        blend_col: ri.col,
                    });
                }
            }
        }
        self.vertex_buffer.write(&vertices)?;
        Ok(())
    }
    fn build_render_pass(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>
    ) -> Result<()> {
        if self.vertex_buffer.vertex_count == 0 { return Ok(()); }
        let pipeline = self.get_or_create_pipeline()?;
        let layout = pipeline.layout().clone();
        let viewport = self.viewport.get();
        let pc = basic::vertex_shader::WindowData {
            #[allow(clippy::cast_possible_truncation)]
            window_width: viewport.physical_width() as f32,
            #[allow(clippy::cast_possible_truncation)]
            window_height: viewport.physical_height() as f32,
            #[allow(clippy::cast_possible_truncation)]
            scale_factor: viewport.scale_factor() as f32,
        };
        builder
            .bind_pipeline_graphics(pipeline.clone())?
            .push_constants(layout, 0, pc)?;
        if self.vertex_buffer.draw(builder).is_err() {
            self.vertex_buffer.realloc()?;
        }
        Ok(())
    }
}
#[derive(Clone)]
pub struct BasicShader {
    ctx: VulkanoContext,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    viewport: UniqueShared<AdjustedViewport>,
    pipeline: Option<Arc<GraphicsPipeline>>,
    vertex_buffer: CachedVertexBuffer<basic::Vertex>,
}

impl BasicShader {
    pub fn new(
        ctx: VulkanoContext,
        viewport: UniqueShared<AdjustedViewport>
    ) -> Result<UniqueShared<Box<dyn Shader>>> {
        register_shader::<Self>();
        let device = ctx.device();
        let vertex_buffer = CachedVertexBuffer::new(ctx.clone(), 10_000)?;
        Ok(UniqueShared::new(Box::new(Self {
            ctx,
            vs: basic::vertex_shader::load(device.clone()).context("failed to create shader module")?,
            fs: basic::fragment_shader::load(device).context("failed to create shader module")?,
            viewport,
            pipeline: None,
            vertex_buffer,
        }) as Box<dyn Shader>))
    }

    fn get_or_create_pipeline(&mut self) -> Result<Arc<GraphicsPipeline>> {
        match self.pipeline.clone() {
            None => {
                let vs = self.vs.entry_point("main")
                    .context("vertex shader: entry point missing")?;
                let fs = self.fs.entry_point("main")
                    .context("fragment shader: entry point missing")?;
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
}

impl Shader for BasicShader {
    fn name() -> ShaderName
    where
        Self: Sized
    {
        ShaderName::new("basic")
    }
    fn name_concrete(&self) -> ShaderName { Self::name() }

    fn on_render(
        &mut self,
        render_frame: ShaderRenderFrame,
    ) -> Result<()> {
        let mut vertices = Vec::with_capacity(self.vertex_buffer.len());
        for render_info in &render_frame.render_infos {
            for vertex_index in render_info.vertex_indices.clone() {
                for ri in &render_info.inner {
                    vertices.push(basic::Vertex {
                        position: render_frame.vertices[vertex_index as usize].xy,
                        translation: render_info.transform.centre,
                        rotation: render_info.transform.rotation,
                        scale: render_info.transform.scale,
                        blend_col: ri.col,
                    });
                }
            }
        }
        self.vertex_buffer.write(&vertices)?;
        Ok(())
    }
    fn build_render_pass(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>
    ) -> Result<()> {
        if self.vertex_buffer.vertex_count == 0 { return Ok(()); }
        let pipeline = self.get_or_create_pipeline()?;
        let layout = pipeline.layout().clone();
        let viewport = self.viewport.get();
        let pc = basic::vertex_shader::WindowData {
            #[allow(clippy::cast_possible_truncation)]
            window_width: viewport.physical_width() as f32,
            #[allow(clippy::cast_possible_truncation)]
            window_height: viewport.physical_height() as f32,
            #[allow(clippy::cast_possible_truncation)]
            scale_factor: viewport.scale_factor() as f32,
        };
        builder
            .bind_pipeline_graphics(pipeline.clone())?
            .push_constants(layout, 0, pc)?;
        self.vertex_buffer.draw(builder)?;
        Ok(())
    }
}
