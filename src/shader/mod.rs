use crate::core::render::ShaderRenderFrame;
use crate::core::vk::vk_ctx::VulkanoContext;
use crate::resource::texture::{Material, MaterialId};
use crate::shader::glsl::basic;
use crate::{
    core::{prelude::*, vk::AdjustedViewport},
    info_every_seconds,
    shader::glsl::sprite,
    util::UniqueShared,
};
use anyhow::{Context, Result};
use itertools::Itertools;
use num_traits::Zero;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use vulkano::command_buffer::{RenderingAttachmentInfo, RenderingInfo};
use vulkano::descriptor_set::sys::RawDescriptorSet;
use vulkano::image::Image;
use vulkano::image::sampler::{Filter, SamplerMipmapMode};
use vulkano::memory::DeviceAlignment;
use vulkano::memory::allocator::DeviceLayout;
use vulkano::pipeline::DynamicState;
use vulkano::pipeline::graphics::rasterization::PolygonMode;
use vulkano::pipeline::graphics::subpass::PipelineRenderingCreateInfo;
pub use vulkano::pipeline::graphics::vertex_input::Vertex as VkVertex;
use vulkano::render_pass::AttachmentLoadOp::Load;
use vulkano::render_pass::AttachmentStoreOp::Store;
use vulkano::swapchain::Swapchain;
use vulkano::{
    DeviceSize, NonZeroDeviceSize, Validated,
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    descriptor_set::{WriteDescriptorSet, layout::DescriptorSetLayoutCreateFlags},
    image::sampler::{Sampler, SamplerCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    pipeline::{
        GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
        graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::{AttachmentBlend, ColorBlendAttachmentState, ColorBlendState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::VertexDefinition,
            viewport::ViewportState,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
};
use vulkano_taskgraph::command_buffer::RecordingCommandBuffer;
use vulkano_taskgraph::graph::{NodeId, TaskGraph};
use vulkano_taskgraph::resource::{AccessTypes, ImageLayoutType};
use vulkano_taskgraph::{Id, QueueFamilyType, Task, TaskContext, TaskResult};

pub mod glsl;
pub mod vertex;

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
        ShaderId(NEXT_SHADER_ID.fetch_add(1, Ordering::SeqCst))
    }

    pub(crate) fn is_valid(self) -> bool {
        self.0 != 0
    }
}
static SHADER_IDS_INIT: LazyLock<Arc<Mutex<HashMap<ShaderName, ShaderId>>>> =
    LazyLock::new(|| Arc::new(Mutex::new(HashMap::new())));
static SHADERS_LOCKED: AtomicBool = AtomicBool::new(false);

static SHADER_IDS_FINAL: LazyLock<HashMap<ShaderName, ShaderId>> = LazyLock::new(|| {
    check!(
        SHADERS_LOCKED.load(Ordering::Acquire),
        "attempted to load shader IDs too early"
    );
    let shader_ids = SHADER_IDS_INIT.lock().unwrap();
    shader_ids.clone()
});

pub fn register_shader<S: Shader + Sized>() -> ShaderId {
    check_false!(
        SHADERS_LOCKED.load(Ordering::Acquire),
        format!("attempted to register shader too late: {:?}", S::name())
    );
    let mut shader_ids = SHADER_IDS_INIT.lock().unwrap();
    *shader_ids.entry(S::name()).or_insert_with(ShaderId::next)
}
pub fn get_shader(name: ShaderName) -> ShaderId {
    SHADER_IDS_FINAL.get(&name).copied().unwrap_or_else(|| {
        error!("unknown shader: {name:?}");
        ShaderId::default()
    })
}
pub(crate) fn ensure_shaders_locked() {
    SHADERS_LOCKED.swap(true, Ordering::Release);
}

pub fn basic_shader_pipeline_info(
    subpass: PipelineRenderingCreateInfo,
    layout: Arc<PipelineLayout>,
) -> GraphicsPipelineCreateInfo {
    GraphicsPipelineCreateInfo {
        input_assembly_state: Some(InputAssemblyState::default()),
        viewport_state: Some(ViewportState::default()),
        rasterization_state: Some(RasterizationState::default()),
        multisample_state: Some(MultisampleState::default()),
        dynamic_state: [DynamicState::Viewport].into_iter().collect(),
        color_blend_state: Some(ColorBlendState::with_attachment_states(
            subpass.color_attachment_formats.len() as u32,
            ColorBlendAttachmentState {
                blend: Some(AttachmentBlend::alpha()),
                ..Default::default()
            },
        )),
        subpass: Some(subpass.into()),
        ..GraphicsPipelineCreateInfo::layout(layout)
    }
}

pub trait Shader: Send {
    fn name() -> ShaderName
    where
        Self: Sized;
    fn name_concrete(&self) -> ShaderName;
    fn id(&self) -> ShaderId {
        get_shader(self.name_concrete())
    }
    fn pre_render_update(
        &mut self,
        image_idx: usize,
        render_frame: ShaderRenderFrame,
        tcx: &mut TaskContext,
    ) -> Result<()>;

    fn buffer_writes(&self) -> Vec<Id<Buffer>>;
    fn build_task_node(
        &mut self,
        task_graph: &mut TaskGraph<VulkanoContext>,
        virtual_swapchain_id: Id<Swapchain>,
        textures: &[Id<Image>],
    ) -> NodeId;
}

#[derive(Clone)]
struct CachedVertexBuffer<T: Default + VkVertex + Copy> {
    ctx: VulkanoContext,
    inner: Id<Buffer>,
    next_vertex_idx: usize,
    vertex_count: usize,
    num_vertex_sets: usize,
    phantom_data: PhantomData<T>,
}

impl<T: Default + VkVertex + Copy> CachedVertexBuffer<T> {
    fn new(ctx: VulkanoContext, size: usize) -> Result<Self> {
        let num_vertex_sets = ctx.image_count();
        let inner = Self::create_vertex_buffer(
            &ctx,
            (size * size_of::<T>() * num_vertex_sets) as DeviceSize,
        )?;
        let rv = Self {
            ctx,
            inner,
            next_vertex_idx: 0,
            vertex_count: 0,
            num_vertex_sets,
            phantom_data: PhantomData,
        };
        info!("created vertex buffer: {} KiB", rv.size_in_bytes() / 1024);
        Ok(rv)
    }

    fn size_in_bytes(&self) -> usize {
        self.ctx
            .resources()
            .buffer(self.inner)
            .unwrap()
            .buffer()
            .size() as usize
    }
    fn len(&self) -> usize {
        self.size_in_bytes() / size_of::<T>()
    }
    fn single_len(&self) -> usize {
        self.len() / self.num_vertex_sets
    }

    fn realloc(&mut self) -> Result<()> {
        let size = self.size_in_bytes();
        if size / 1024 / 1024 == 0 {
            info!(
                "reallocating vertex buffer: {} KiB -> {} KiB",
                size / 1024,
                size * 2 / 1024
            );
        } else {
            info!(
                "reallocating vertex buffer: {} MiB -> {} MiB",
                size / 1024 / 1024,
                size * 2 / 1024 / 1024
            );
        }
        // Just double the size.
        self.inner = Self::create_vertex_buffer(&self.ctx, (self.len() * 2) as DeviceSize)?;
        Ok(())
    }

    fn write(&mut self, image_idx: usize, vertices: &[T], tcx: &mut TaskContext) -> Result<()> {
        self.next_vertex_idx = image_idx * self.single_len();
        self.vertex_count = vertices.len();

        if !vertices.is_empty() {
            // Reallocate if needed:
            while self.next_vertex_idx + vertices.len() > self.len() {
                self.realloc()?;
            }
            let start = (self.next_vertex_idx * size_of::<T>()) as DeviceSize;
            let end = ((self.next_vertex_idx + vertices.len()) * size_of::<T>()) as DeviceSize;
            tcx.write_buffer::<[T]>(self.inner, start..end)?
                .copy_from_slice(vertices);
        }
        Ok(())
    }

    fn create_vertex_buffer(ctx: &VulkanoContext, size: DeviceSize) -> Result<Id<Buffer>> {
        Ok(ctx
            .resources()
            .create_buffer(
                BufferCreateInfo {
                    usage: BufferUsage::VERTEX_BUFFER | BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                DeviceLayout::new(
                    NonZeroDeviceSize::new(size).unwrap(),
                    DeviceAlignment::of::<T>(),
                )
                .context("failed to create vertex buffer of size {size}")?,
            )
            .map_err(Validated::unwrap)?)
    }

    fn draw(&self, builder: &mut RecordingCommandBuffer) -> Result<()> {
        if self.next_vertex_idx + self.vertex_count >= self.len() {
            bail!(
                "too many vertices: {} + {} = {} >= {}",
                self.next_vertex_idx,
                self.vertex_count,
                self.len(),
                self.next_vertex_idx + self.vertex_count
            );
        }
        let start = (self.next_vertex_idx * size_of::<T>()) as DeviceSize;
        let end = ((self.next_vertex_idx + self.vertex_count) * size_of::<T>()) as DeviceSize;
        let vertex_count = u32::try_from(self.vertex_count)
            .with_context(|| format!("tried to draw too many vertices: {}", self.vertex_count))?;
        unsafe {
            builder.bind_vertex_buffers(
                0,
                &[self.inner],
                &[start],
                &[DeviceSize::from(end - start)],
                &[],
            )?;
            builder.draw(vertex_count, 1, 0, 0)?;
        }
        Ok(())
    }
}

#[derive(Clone)]
struct SpriteShaderDescriptorSet {
    desc: Arc<RawDescriptorSet>,
    // Ensure the below are not prematurely dropped:
    _samplers: Vec<Arc<Sampler>>,
    _writes: Vec<WriteDescriptorSet>,
}

// Can also be used as a "basic shader" (no texture) by using material ID 0.
#[derive(Clone)]
pub struct SpriteShader {
    ctx: VulkanoContext,
    viewport: UniqueShared<AdjustedViewport>,
    pipeline: Arc<GraphicsPipeline>,
    vertex_buffer: UniqueShared<CachedVertexBuffer<sprite::Vertex>>,
    resource_handler: ResourceHandler,
    materials: Id<Buffer>,
    virtual_swapchain_id: Option<Id<Swapchain>>,
    descriptor_set: UniqueShared<Option<SpriteShaderDescriptorSet>>,
    descriptor_set_backup: UniqueShared<Vec<SpriteShaderDescriptorSet>>,
}

impl SpriteShader {
    pub(crate) fn create(
        ctx: VulkanoContext,
        viewport: UniqueShared<AdjustedViewport>,
        resource_handler: ResourceHandler,
    ) -> Result<UniqueShared<Box<dyn Shader>>> {
        register_shader::<Self>();
        let vertex_buffer = UniqueShared::new(CachedVertexBuffer::new(
            ctx.clone(),
            INITIAL_VERTEX_BUFFER_SIZE,
        )?);
        let materials = ctx.resources().create_buffer(
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            DeviceLayout::new_unsized::<sprite::vertex_shader::MaterialData>(
                MAX_MATERIAL_COUNT as DeviceSize,
            )
            .unwrap(),
        )?;
        // Create pipeline:
        let pipeline = Self::create_pipeline(&ctx)?;
        Ok(UniqueShared::new(Box::new(Self {
            ctx,
            viewport,
            pipeline,
            vertex_buffer,
            resource_handler,
            materials,
            virtual_swapchain_id: None,
            descriptor_set: UniqueShared::new(None),
            descriptor_set_backup: UniqueShared::new(Vec::new()),
        }) as Box<dyn Shader>))
    }

    fn create_pipeline(ctx: &VulkanoContext) -> Result<Arc<GraphicsPipeline>> {
        let vs =
            sprite::vertex_shader::load(ctx.device()).context("failed to create shader module")?;
        let fs = sprite::fragment_shader::load(ctx.device())
            .context("failed to create shader module")?;
        let vs_entry = vs
            .entry_point("main")
            .context("vertex shader: entry point missing")?;
        let fs_entry = fs
            .entry_point("main")
            .context("fragment shader: entry point missing")?;
        let vertex_input_state = sprite::Vertex::per_vertex().definition(&vs_entry)?;
        let stages = [
            PipelineShaderStageCreateInfo::new(vs_entry),
            PipelineShaderStageCreateInfo::new(fs_entry),
        ];
        let mut create_info = PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages);
        for layout in &mut create_info.set_layouts {
            layout.flags |= DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL;
        }
        let layout = PipelineLayout::new(
            ctx.device(),
            create_info.into_pipeline_layout_create_info(ctx.device())?,
        )
        .map_err(Validated::unwrap)?;
        let subpass = PipelineRenderingCreateInfo {
            color_attachment_formats: vec![Some(ctx.swapchain()?.image_format())],
            ..Default::default()
        };
        let pipeline = GraphicsPipeline::new(
            ctx.device(),
            /* cache= */ None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(vertex_input_state),
                ..basic_shader_pipeline_info(subpass, layout)
            },
        )?;
        Ok(pipeline)
    }

    fn maybe_update_desc_sets(&mut self, tcx: &mut TaskContext) -> Result<()> {
        let maybe_materials = self.resource_handler.texture.get_updated_materials();
        if maybe_materials.is_none() && self.descriptor_set.lock().is_some() {
            return Ok(());
        }

        let mut textures = vec![None; MAX_TEXTURE_COUNT];
        let mut nonempty_texture_ids = Vec::new();
        for texture in self.resource_handler.texture.ready_values() {
            let texture_id = texture.id() as usize;
            check_le!(texture_id, MAX_TEXTURE_COUNT);
            check_is_none!(textures[texture_id]);
            textures[texture_id] = texture.image_view();
            nonempty_texture_ids.push(texture_id);
        }

        let blank = textures
            .first()
            .expect("textures.first() should always contain a blank texture")
            .clone()
            .expect("textures.first() should always contain a blank texture that is loaded");
        let textures = textures
            .into_iter()
            .map(|t| t.unwrap_or(blank.clone()))
            .collect_vec();

        let sampler_create_info = SamplerCreateInfo {
            min_filter: Filter::Linear,
            mipmap_mode: SamplerMipmapMode::Linear,
            lod: 0.0..=(FONT_SAMPLE_RATIO.log2() + 1.0),
            ..SamplerCreateInfo::default()
        };
        let sampler = Sampler::new(self.ctx.device(), sampler_create_info.clone())
            .map_err(Validated::unwrap)?;
        let samplers = vec![sampler.clone(); MAX_TEXTURE_COUNT];

        if let Some(materials) = maybe_materials {
            info_every_seconds!(
                1,
                "updating materials: materials.len() = {}, texture count: {}",
                materials.len(),
                nonempty_texture_ids.len()
            );
            self.update_materials(tcx, materials)?;
        }
        let desc_set = RawDescriptorSet::new(
            self.ctx.descriptor_set_allocator(),
            &self.pipeline.layout().set_layouts()[0],
            0,
        )?;
        let desc_writes = vec![
            WriteDescriptorSet::buffer(
                0,
                self.ctx
                    .resources()
                    .buffer(self.materials)?
                    .buffer()
                    .clone()
                    .into(),
            ),
            WriteDescriptorSet::image_view_sampler_array(1, 0, textures.into_iter().zip(samplers)),
        ];
        unsafe {
            desc_set.update(&desc_writes, &[])?;
        }
        if let Some(desc_set) = self.descriptor_set.lock().take() {
            self.descriptor_set_backup.lock().push(desc_set);
            if self.descriptor_set_backup.lock().len() > 1 {
                self.descriptor_set_backup.lock().remove(0);
            }
        }
        *self.descriptor_set.lock() = Some(SpriteShaderDescriptorSet {
            desc: Arc::new(desc_set),
            _samplers: vec![sampler],
            _writes: desc_writes,
        });
        Ok(())
    }

    fn update_materials(
        &mut self,
        tcx: &mut TaskContext,
        materials: Vec<(MaterialId, Material)>,
    ) -> Result<()> {
        let mut data = vec![
            sprite::vertex_shader::Material {
                texture_id: 0,
                uv_top_left: Vec2::zero().into(),
                uv_bottom_right: Vec2::one().into(),
                dummy1: 0,
                dummy2: 0,
                dummy3: 0,
            };
            MAX_MATERIAL_COUNT
        ];
        for (id, mat) in materials {
            let entry = &mut data[id as usize];
            entry.texture_id = mat.texture_id;
            entry.uv_top_left = mat
                .area
                .top_left()
                .component_wise_div(mat.texture_extent)
                .into();
            entry.uv_bottom_right = mat
                .area
                .bottom_right()
                .component_wise_div(mat.texture_extent)
                .into();
        }
        tcx.write_buffer::<sprite::vertex_shader::MaterialData>(self.materials, ..)?
            .data
            .copy_from_slice(&data);
        Ok(())
    }
}

impl Shader for SpriteShader {
    fn name() -> ShaderName
    where
        Self: Sized,
    {
        ShaderName::new("sprite")
    }
    fn name_concrete(&self) -> ShaderName {
        Self::name()
    }

    fn pre_render_update(
        &mut self,
        image_idx: usize,
        render_frame: ShaderRenderFrame,
        tcx: &mut TaskContext,
    ) -> Result<()> {
        self.maybe_update_desc_sets(tcx)?;
        let render_infos = render_frame
            .render_infos
            .iter()
            .sorted_unstable_by_key(|item| item.depth);
        let mut vertices = Vec::with_capacity(self.vertex_buffer.lock().single_len());
        for render_info in render_infos {
            for vertex_index in render_info.vertex_indices.clone() {
                let vertex = render_frame.vertices[vertex_index as usize];
                let clip = render_info.clip * self.viewport.lock().total_scale_factor();
                for ri in &render_info.inner {
                    vertices.push(sprite::Vertex {
                        position: vertex.inner.into(),
                        material_id: ri.material_id,
                        translation: render_info.transform.centre.into(),
                        rotation: render_info.transform.rotation,
                        scale: render_info.transform.scale.into(),
                        blend_col: (vertex.blend_col * ri.blend_col).into(),
                        clip_min: clip.top_left().into(),
                        clip_max: clip.bottom_right().into(),
                    });
                }
            }
        }
        self.vertex_buffer.lock().write(image_idx, &vertices, tcx)?;
        Ok(())
    }

    fn buffer_writes(&self) -> Vec<Id<Buffer>> {
        vec![self.vertex_buffer.lock().inner, self.materials]
    }

    fn build_task_node(
        &mut self,
        task_graph: &mut TaskGraph<VulkanoContext>,
        virtual_swapchain_id: Id<Swapchain>,
        textures: &[Id<Image>],
    ) -> NodeId {
        self.virtual_swapchain_id = Some(virtual_swapchain_id);
        let mut node = task_graph.create_task_node(
            self.name_concrete().0,
            QueueFamilyType::Graphics,
            self.clone(),
        );
        node.image_access(
            virtual_swapchain_id.current_image_id(),
            AccessTypes::COLOR_ATTACHMENT_WRITE,
            ImageLayoutType::Optimal,
        );
        for tex in textures {
            node.image_access(
                *tex,
                AccessTypes::FRAGMENT_SHADER_SAMPLED_READ,
                ImageLayoutType::Optimal,
            );
        }
        node.buffer_access(
            self.vertex_buffer.lock().inner,
            AccessTypes::VERTEX_ATTRIBUTE_READ,
        );
        node.buffer_access(self.materials, AccessTypes::VERTEX_ATTRIBUTE_READ);
        node.build()
    }
}

impl Task for SpriteShader {
    type World = VulkanoContext;

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        tcx: &mut TaskContext<'_>,
        world: &Self::World,
    ) -> TaskResult {
        if self.vertex_buffer.lock().vertex_count.is_zero() {
            return Ok(());
        }
        let layout = self.pipeline.layout().clone();
        let viewport = self.viewport.lock();
        let pc = sprite::vertex_shader::WindowData {
            window_width: viewport.physical_width(),
            window_height: viewport.physical_height(),
            scale_factor: viewport.scale_factor(),
            view_translate_x: viewport.translation.x,
            view_translate_y: viewport.translation.y,
        };

        let viewport = viewport.inner();
        unsafe {
            cbf.set_viewport(0, std::slice::from_ref(&viewport))
                .unwrap();
        }
        let image_idx = tcx
            .swapchain(world.swapchain_id())
            .unwrap()
            .current_image_index()
            .unwrap() as usize;
        let image_view = world.current_image_view(image_idx);
        unsafe {
            cbf.as_raw()
                .begin_rendering(&RenderingInfo {
                    color_attachments: vec![Some(RenderingAttachmentInfo {
                        load_op: Load,
                        store_op: Store,
                        ..RenderingAttachmentInfo::image_view(image_view)
                    })],
                    render_area_extent: [viewport.extent[0] as u32, viewport.extent[1] as u32],
                    layer_count: 1,
                    ..Default::default()
                })
                .unwrap();
            world.perf_stats().lap("SpriteShader: begin_rendering()");

            cbf.bind_pipeline_graphics(&self.pipeline)?
                .as_raw()
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    &layout.clone(),
                    0,
                    &[&self.descriptor_set.lock().clone().unwrap().desc],
                    &[],
                )?
                .push_constants(&layout, 0, &pc)?;
            world
                .perf_stats()
                .lap("SpriteShader: bind_pipeline_graphics()");
        }

        self.vertex_buffer.lock().draw(cbf).unwrap();
        world.perf_stats().lap("SpriteShader: draw()");

        unsafe {
            cbf.as_raw().end_rendering().unwrap();
        }
        world.perf_stats().lap("SpriteShader: end_rendering()");

        Ok(())
    }
}

#[derive(Clone)]
pub struct WireframeShader {
    viewport: UniqueShared<AdjustedViewport>,
    pipeline: Arc<GraphicsPipeline>,
    vertex_buffer: UniqueShared<CachedVertexBuffer<basic::Vertex>>,
    virtual_swapchain_id: Option<Id<Swapchain>>,
}

impl WireframeShader {
    pub(crate) fn create(
        ctx: VulkanoContext,
        viewport: UniqueShared<AdjustedViewport>,
    ) -> Result<UniqueShared<Box<dyn Shader>>> {
        register_shader::<Self>();
        let pipeline = Self::create_pipeline(&ctx)?;
        let vertex_buffer =
            UniqueShared::new(CachedVertexBuffer::new(ctx, INITIAL_VERTEX_BUFFER_SIZE)?);
        // Create pipeline:
        Ok(UniqueShared::new(Box::new(Self {
            viewport,
            pipeline,
            vertex_buffer,
            virtual_swapchain_id: None,
        }) as Box<dyn Shader>))
    }

    fn create_pipeline(ctx: &VulkanoContext) -> Result<Arc<GraphicsPipeline>> {
        let vs =
            basic::vertex_shader::load(ctx.device()).context("failed to create shader module")?;
        let fs =
            basic::fragment_shader::load(ctx.device()).context("failed to create shader module")?;
        let vs_entry = vs
            .entry_point("main")
            .context("vertex shader: entry point missing")?;
        let fs_entry = fs
            .entry_point("main")
            .context("fragment shader: entry point missing")?;
        let vertex_input_state = basic::Vertex::per_vertex().definition(&vs_entry)?;
        let stages = [
            PipelineShaderStageCreateInfo::new(vs_entry),
            PipelineShaderStageCreateInfo::new(fs_entry),
        ];
        let mut create_info = PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages);
        for layout in &mut create_info.set_layouts {
            layout.flags |= DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL;
        }
        let layout = PipelineLayout::new(
            ctx.device(),
            create_info.into_pipeline_layout_create_info(ctx.device())?,
        )
        .map_err(Validated::unwrap)?;
        let subpass = PipelineRenderingCreateInfo {
            color_attachment_formats: vec![Some(ctx.swapchain()?.image_format())],
            ..Default::default()
        };
        let pipeline = GraphicsPipeline::new(
            ctx.device(),
            /* cache= */ None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(vertex_input_state),
                rasterization_state: Some(RasterizationState {
                    polygon_mode: PolygonMode::Line,
                    ..RasterizationState::default()
                }),
                ..basic_shader_pipeline_info(subpass, layout)
            },
        )?;
        Ok(pipeline)
    }
}

impl Shader for WireframeShader {
    fn name() -> ShaderName
    where
        Self: Sized,
    {
        ShaderName::new("wireframe")
    }
    fn name_concrete(&self) -> ShaderName {
        Self::name()
    }

    fn pre_render_update(
        &mut self,
        image_idx: usize,
        render_frame: ShaderRenderFrame,
        tcx: &mut TaskContext,
    ) -> Result<()> {
        let render_infos = render_frame
            .render_infos
            .iter()
            .sorted_unstable_by_key(|item| item.depth);
        let mut vertices = Vec::with_capacity(self.vertex_buffer.lock().single_len());
        for render_info in render_infos {
            for vertex_index in render_info.vertex_indices.clone() {
                let vertex = render_frame.vertices[vertex_index as usize];
                for ri in &render_info.inner {
                    vertices.push(basic::Vertex {
                        position: vertex.inner.into(),
                        translation: render_info.transform.centre.into(),
                        rotation: render_info.transform.rotation,
                        scale: render_info.transform.scale.into(),
                        blend_col: (vertex.blend_col * ri.blend_col).into(),
                    });
                }
            }
        }
        self.vertex_buffer.lock().write(image_idx, &vertices, tcx)?;
        Ok(())
    }

    fn buffer_writes(&self) -> Vec<Id<Buffer>> {
        vec![self.vertex_buffer.lock().inner]
    }

    fn build_task_node(
        &mut self,
        task_graph: &mut TaskGraph<VulkanoContext>,
        virtual_swapchain_id: Id<Swapchain>,
        _textures: &[Id<Image>],
    ) -> NodeId {
        self.virtual_swapchain_id = Some(virtual_swapchain_id);
        let mut node = task_graph.create_task_node(
            self.name_concrete().0,
            QueueFamilyType::Graphics,
            self.clone(),
        );
        node.image_access(
            virtual_swapchain_id.current_image_id(),
            AccessTypes::COLOR_ATTACHMENT_WRITE,
            ImageLayoutType::Optimal,
        );
        node.buffer_access(
            self.vertex_buffer.lock().inner,
            AccessTypes::VERTEX_ATTRIBUTE_READ,
        );
        node.build()
    }
}

impl Task for WireframeShader {
    type World = VulkanoContext;

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        tcx: &mut TaskContext<'_>,
        world: &Self::World,
    ) -> TaskResult {
        if self.vertex_buffer.lock().vertex_count.is_zero() {
            return Ok(());
        }
        let layout = self.pipeline.layout().clone();
        let viewport = self.viewport.lock();
        let pc = basic::vertex_shader::WindowData {
            window_width: viewport.physical_width(),
            window_height: viewport.physical_height(),
            scale_factor: viewport.scale_factor(),
            view_translate_x: viewport.translation.x,
            view_translate_y: viewport.translation.y,
        };

        let viewport = viewport.inner();
        unsafe {
            cbf.set_viewport(0, std::slice::from_ref(&viewport))
                .unwrap();
        }
        let image_idx = tcx
            .swapchain(world.swapchain_id())
            .unwrap()
            .current_image_index()
            .unwrap() as usize;
        let image_view = world.current_image_view(image_idx);
        unsafe {
            cbf.as_raw()
                .begin_rendering(&RenderingInfo {
                    color_attachments: vec![Some(RenderingAttachmentInfo {
                        load_op: Load,
                        store_op: Store,
                        ..RenderingAttachmentInfo::image_view(image_view)
                    })],
                    render_area_extent: [viewport.extent[0] as u32, viewport.extent[1] as u32],
                    layer_count: 1,
                    ..Default::default()
                })
                .unwrap();

            cbf.bind_pipeline_graphics(&self.pipeline)?
                .push_constants(&layout, 0, &pc)?;
        }

        self.vertex_buffer.lock().draw(cbf).unwrap();

        unsafe {
            cbf.as_raw().end_rendering().unwrap();
        }

        Ok(())
    }
}
