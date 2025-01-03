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
    DescriptorSet,
    WriteDescriptorSet,
    layout::DescriptorSetLayoutCreateFlags
}, command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer}, buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer}, shader::ShaderModule, render_pass::Subpass, Validated, DeviceSize};
use vulkano::pipeline::graphics::input_assembly::PrimitiveTopology;
use vulkano::pipeline::graphics::rasterization::PolygonMode;
use crate::{core::{
    prelude::*,
    vk::AdjustedViewport,
}, shader::glsl::{basic, sprite}, util::UniqueShared};
pub use vulkano::pipeline::graphics::vertex_input::Vertex as VkVertex;
use crate::core::render::ShaderRenderFrame;
use crate::core::vk::vk_ctx::VulkanoContext;

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
    fn do_render_shader(
        &mut self,
        render_frame: ShaderRenderFrame
    ) -> Result<()>;
    fn build_render_pass(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>
    ) -> Result<()>;
    fn on_recreate_swapchain(&mut self);
}

#[derive(Clone)]
struct CachedVertexBuffer<T: VkVertex + Copy> {
    ctx: VulkanoContext,
    inner: Subbuffer<[T]>,
    next_vertex_idx: usize,
    vertex_count: Option<usize>,
    next_free_idx: usize,
    num_vertex_sets: usize,
}

impl<T: VkVertex + Copy> CachedVertexBuffer<T> {
    fn new(ctx: VulkanoContext, size: usize) -> Result<Self> {
        // Allow some headroom:
        let num_vertex_sets = ctx.image_count() + 1;
        let inner = Self::create_vertex_buffer(
            &ctx,
            (size * num_vertex_sets) as DeviceSize
        )?;
        Ok(Self {
            ctx,
            inner,
            next_vertex_idx: 0,
            vertex_count: None,
            next_free_idx: 0,
            num_vertex_sets
        })
    }

    fn len(&self) -> usize { self.inner.len() as usize / self.num_vertex_sets }

    fn realloc(&mut self) -> Result<()> {
        let size = self.inner.len() as usize * std::mem::size_of::<T>();
        if size / 1024 / 1024 == 0 {
            warn!("reallocating vertex buffer: {} KiB -> {} KiB",
                size / 1024 , size * 2 / 1024);
        } else {
            warn!("reallocating vertex buffer: {} MiB -> {} MiB",
                size / 1024 / 1024, size * 2 / 1024 / 1024);
        }
        // Just double the size.
        self.inner = Self::create_vertex_buffer(
            &self.ctx,
            (self.inner.len() * 2) as DeviceSize
        )?;
        Ok(())
    }

    fn write(&mut self, data: &[T]) -> Result<()> {
        if data.is_empty() { return Ok(()); }
        // Reallocate if needed:
        let mut buf_len = usize::try_from(self.inner.len())
            .with_context(|| format!("self.inner.len() overflowed: {}", self.inner.len()))?;
        while self.next_free_idx + data.len() > buf_len {
            buf_len = usize::try_from(self.inner.len())
                .with_context(|| format!("self.inner.len() overflowed: {}", self.inner.len()))?;
            self.realloc()?;
            self.next_free_idx = 0;
        }

        self.next_vertex_idx = self.next_free_idx;
        check_is_none!(self.vertex_count);
        self.vertex_count = Some(data.len());

        // Perform write:
        self.inner.write()?[self.next_free_idx..self.next_free_idx + data.len()].copy_from_slice(data);

        // Wrap around the ring buffer if needed:
        self.next_free_idx += self.len();
        if self.next_free_idx > buf_len {
            bail!("self.next_free_idx accounting wrong? {} > {buf_len}", self.next_free_idx)
        }
        if self.next_free_idx == buf_len {
            self.next_free_idx = 0;
        }

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

    fn draw(&mut self, builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>) -> Result<()> {
        let vertex_count = self.vertex_count.take()
            .with_context(|| format!("tried to draw vertices but none prepared: {:?}", self.vertex_count))?;
        let vertex_count = u32::try_from(vertex_count)
            .with_context(|| format!("tried to draw too many vertices: {vertex_count}"))?;
        let first_vertex_idx = u32::try_from(self.next_vertex_idx)
            .with_context(|| format!("self.begin overflowed: {}", self.next_vertex_idx))?;
        let buf_len = u32::try_from(self.inner.len())
            .with_context(|| format!("self.inner.len() overflowed: {}", self.inner.len()))?;
        if first_vertex_idx + vertex_count >= buf_len {
            bail!("too many vertices: {first_vertex_idx} + {vertex_count} = {} >= {buf_len}",
                       first_vertex_idx + vertex_count);
        }
        check_ne!(self.next_free_idx, self.next_vertex_idx);
        unsafe {
            builder.bind_vertex_buffers(0, self.inner.clone())?
                .draw(vertex_count, 1, first_vertex_idx, 0)?;
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct SpriteShader {
    ctx: VulkanoContext,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    viewport: UniqueShared<AdjustedViewport>,
    pipeline: UniqueShared<Option<Arc<GraphicsPipeline>>>,
    vertex_buffer: CachedVertexBuffer<sprite::Vertex>,
    resource_handler: ResourceHandler,
}

impl SpriteShader {
    pub fn create(
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
            pipeline: UniqueShared::default(),
            vertex_buffer,
            resource_handler,
        }) as Box<dyn Shader>))
    }

    fn get_or_create_pipeline(&mut self) -> Result<Arc<GraphicsPipeline>> {
        if self.pipeline.get().is_none() {
            let vs = self.vs.entry_point("main")
                .context("vertex shader: entry point missing")?;
            let fs = self.fs.entry_point("main")
                .context("fragment shader: entry point missing")?;
            let vertex_input_state =
                sprite::Vertex::per_vertex().definition(&vs)?;
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

            let device = self.ctx.device();
            self.pipeline = self.ctx.create_pipeline(|render_pass| {
                let subpass = Subpass::from(render_pass, 0)
                    .context("failed to create subpass")?;
                Ok(GraphicsPipeline::new(device,
                     /* cache= */ None,
                     GraphicsPipelineCreateInfo {
                         stages: stages.into_iter().collect(),
                         vertex_input_state: Some(vertex_input_state),
                         input_assembly_state: Some(InputAssemblyState::default()),
                         viewport_state: Some(self.viewport.get().as_viewport_state()),
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
                     })?)
            })?;
        }
        Ok(self.pipeline.get().clone().unwrap())
    }
    fn create_uniform_desc_set(&mut self) -> Result<Arc<DescriptorSet>> {
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
        check_eq!(textures.len(), MAX_TEXTURE_COUNT);
        Ok(DescriptorSet::new(
            self.ctx.descriptor_set_allocator(),
            pipeline.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::image_view_sampler_array(
                    0,
                    0,
                    textures.into_iter().zip(vec![sampler.clone(); MAX_TEXTURE_COUNT])
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

    fn do_render_shader(
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
        if self.vertex_buffer.vertex_count.is_none() { return Ok(()); }
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

    fn on_recreate_swapchain(&mut self) {}
}

#[derive(Clone)]
pub struct WireframeShader {
    ctx: VulkanoContext,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    viewport: UniqueShared<AdjustedViewport>,
    pipeline: UniqueShared<Option<Arc<GraphicsPipeline>>>,
    vertex_buffer: CachedVertexBuffer<basic::Vertex>,
}

impl WireframeShader {
    pub fn create(
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
            pipeline: UniqueShared::default(),
            vertex_buffer,
        }) as Box<dyn Shader>))
    }

    fn get_or_create_pipeline(&mut self) -> Result<Arc<GraphicsPipeline>> {
        if self.pipeline.get().is_none() {
            let vs = self.vs.entry_point("main")
                .context("vertex shader: entry point missing")?;
            let fs = self.fs.entry_point("main")
                .context("fragment shader: entry point missing")?;
            let vertex_input_state =
                basic::Vertex::per_vertex().definition(&vs)?;
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
            let device = self.ctx.device();
            self.pipeline = self.ctx.create_pipeline(|render_pass| {
                let subpass = Subpass::from(render_pass, 0)
                    .context("failed to create subpass")?;
                Ok(GraphicsPipeline::new(
                    device,
                    /* cache= */ None,
                    GraphicsPipelineCreateInfo {
                        stages: stages.into_iter().collect(),
                        vertex_input_state: Some(vertex_input_state),
                        input_assembly_state: Some(InputAssemblyState::default()),
                        viewport_state: Some(self.viewport.get().as_viewport_state()),
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
                    })?)
            })?;
        }
        Ok(self.pipeline.get().clone().unwrap())
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

    fn do_render_shader(
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
        if self.vertex_buffer.vertex_count.is_none() { return Ok(()); }
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

    fn on_recreate_swapchain(&mut self) {}
}
#[derive(Clone)]
pub struct TriangleFanShader {
    ctx: VulkanoContext,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    viewport: UniqueShared<AdjustedViewport>,
    pipeline: UniqueShared<Option<Arc<GraphicsPipeline>>>,
    vertex_buffer: CachedVertexBuffer<basic::Vertex>,
}

impl TriangleFanShader {
    pub fn create(
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
            pipeline: UniqueShared::default(),
            vertex_buffer,
        }) as Box<dyn Shader>))
    }

    fn get_or_create_pipeline(&mut self) -> Result<Arc<GraphicsPipeline>> {
        if self.pipeline.get().is_none() {
            let vs = self.vs.entry_point("main")
                .context("vertex shader: entry point missing")?;
            let fs = self.fs.entry_point("main")
                .context("fragment shader: entry point missing")?;
            let vertex_input_state =
                basic::Vertex::per_vertex().definition(&vs)?;
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
            let device = self.ctx.device();
            self.pipeline = self.ctx.create_pipeline(|render_pass| {
                let subpass = Subpass::from(render_pass, 0)
                    .context("failed to create subpass")?;
                Ok(GraphicsPipeline::new(
                    device,
                    /* cache= */ None,
                    GraphicsPipelineCreateInfo {
                        stages: stages.into_iter().collect(),
                        vertex_input_state: Some(vertex_input_state),
                        input_assembly_state: Some(InputAssemblyState::default()),
                        viewport_state: Some(self.viewport.get().as_viewport_state()),
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
                    })?)
            })?;
        }
        Ok(self.pipeline.get().clone().unwrap())
    }
}

impl Shader for TriangleFanShader {
    fn name() -> ShaderName
    where
        Self: Sized
    {
        ShaderName::new("triangle_fan")
    }
    fn name_concrete(&self) -> ShaderName { Self::name() }

    fn do_render_shader(
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
        if self.vertex_buffer.vertex_count.is_none() { return Ok(()); }
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
            .set_primitive_topology(PrimitiveTopology::TriangleFan)?
            .bind_pipeline_graphics(pipeline.clone())?
            .push_constants(layout, 0, pc)?;
        self.vertex_buffer.draw(builder)?;
        Ok(())
    }

    fn on_recreate_swapchain(&mut self) {}
}

#[derive(Clone)]
pub struct BasicShader {
    ctx: VulkanoContext,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    viewport: UniqueShared<AdjustedViewport>,
    pipeline: UniqueShared<Option<Arc<GraphicsPipeline>>>,
    vertex_buffer: CachedVertexBuffer<basic::Vertex>,
}

impl BasicShader {
    pub fn create(
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
            pipeline: UniqueShared::default(),
            vertex_buffer,
        }) as Box<dyn Shader>))
    }

    fn get_or_create_pipeline(&mut self) -> Result<Arc<GraphicsPipeline>> {
        if self.pipeline.get().is_none() {
            let vs = self.vs.entry_point("main")
                .context("vertex shader: entry point missing")?;
            let fs = self.fs.entry_point("main")
                .context("fragment shader: entry point missing")?;
            let vertex_input_state =
                basic::Vertex::per_vertex().definition(&vs)?;
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
            let device = self.ctx.device();
            self.pipeline = self.ctx.create_pipeline(|render_pass| {
                let subpass = Subpass::from(render_pass, 0)
                    .context("failed to create subpass")?;
                Ok(GraphicsPipeline::new(
                    device,
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
                    })?)
            })?;
        }
        Ok(self.pipeline.get().clone().unwrap())
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

    fn do_render_shader(
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
        if self.vertex_buffer.vertex_count.is_none() { return Ok(()); }
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

    fn on_recreate_swapchain(&mut self) {}
}
