use std::{
    cmp,
    default::Default,
    sync::{Arc, Mutex},
    ops::Range,
    collections::BTreeMap
};

use vulkano::{
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
    },
    render_pass::Framebuffer,
    Validated,
};
use winit::window::Window;
use num_traits::Zero;
use vulkano::command_buffer::{RenderPassBeginInfo, SubpassBeginInfo, SubpassEndInfo};

use crate::{
    core::{
        prelude::*,
        vk::{
            AdjustedViewport,
            VulkanoContext,
        },
        ObjectId,
    },
    resource::texture::TextureSubArea,
};
use crate::core::prelude::linalg::TransformF32;
use crate::core::util::UniqueShared;
use crate::shader::Shader;

#[derive(Clone, Debug)]
pub struct RenderInfo {
    pub col: [f32; 4],
    pub texture_id: u16,
    pub texture_sub_area: TextureSubArea,
}

impl Default for RenderInfo {
    fn default() -> Self {
        Self { col: Colour::white().into(), texture_id: 0, texture_sub_area: TextureSubArea::default() }
    }
}

#[derive(Clone, Debug)]
pub struct RenderInfoFull {
    pub inner: RenderInfo,
    pub transform: TransformF32,
    pub vertex_indices: Range<u32>,
    pub depth: VertexDepth,
}

#[derive(Clone)]
pub struct RenderDataChannel {
    pub(crate) vertices: Vec<VertexWithUV>,
    pub(crate) render_infos: Vec<RenderInfoFull>,
    viewport: AdjustedViewport,
    clear_col: Colour,
}
impl RenderDataChannel {
    fn new(viewport: AdjustedViewport) -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self {
            vertices: Vec::new(),
            render_infos: Vec::new(),
            viewport,
            clear_col: Colour::black(),
        }))
    }

    pub(crate) fn next_frame(&self) -> RenderFrame {
        RenderFrame {
            vertices: self.vertices.clone(),
            render_infos: self.render_infos.clone(),
            clear_col: self.clear_col
        }
    }

    pub(crate) fn update_vertices(&mut self, vertices: Vec<VertexWithUV>) {
        self.vertices = vertices;
    }

    pub(crate) fn update_render_info(&mut self, render_info: Vec<RenderInfoFull>) {
        self.render_infos = render_info;
    }

    pub(crate) fn current_viewport(&self) -> AdjustedViewport {
        self.viewport.clone()
    }

    pub(crate) fn is_ready(&self) -> bool {
        !self.vertices.is_empty() && !self.render_infos.is_empty()
    }

    pub(crate) fn set_clear_col(&mut self, col: Colour) { self.clear_col = col; }
}

#[derive(Clone)]
pub struct RenderFrame {
    pub vertices: Vec<VertexWithUV>,
    pub render_infos: Vec<RenderInfoFull>,
    pub clear_col: Colour,
}

#[derive(Clone)]
pub struct RenderHandler {
    render_data_channel: Arc<Mutex<RenderDataChannel>>,
    viewport: UniqueShared<AdjustedViewport>,
    shaders: Vec<Arc<Mutex<dyn Shader>>>,
    command_buffer: Option<Arc<PrimaryAutoCommandBuffer>>,
}

impl RenderHandler {
    pub fn new(
        viewport: UniqueShared<AdjustedViewport>,
        shaders: Vec<Arc<Mutex<dyn Shader>>>,
    ) -> Self {
        let render_info_receiver = RenderDataChannel::new(viewport.clone_inner());
        Self {
            shaders,
            viewport,
            command_buffer: None,
            render_data_channel: render_info_receiver,
        }
    }

    #[must_use]
    pub fn with_global_scale_factor(self, global_scale_factor: f64) -> Self {
        self.viewport.get().set_global_scale_factor(global_scale_factor);
        self
    }
}

impl RenderHandler {
    pub(crate) fn on_resize(
        &mut self,
        _ctx: &VulkanoContext,
        window: &Arc<Window>,
    ) {
        self.viewport.get().update_from_window(window);
        self.command_buffer = None;
        self.render_data_channel.lock().unwrap().viewport = self.viewport.get().clone();
    }

    pub(crate) fn on_render(
        &mut self,
        ctx: &VulkanoContext,
        framebuffer: &Arc<Framebuffer>,
    ) -> Result<Arc<PrimaryAutoCommandBuffer>> {
        let render_frame = self.render_data_channel.lock().unwrap().next_frame();
        for shader in &mut self.shaders {
            shader.try_lock().unwrap().on_render(&render_frame)?;
        }
        let mut builder = AutoCommandBufferBuilder::primary(
            ctx.command_buffer_allocator(),
            ctx.queue().queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;
        // TODO: will be useful for adding the gui console.
        // let top_left = [framebuffer.extent()[0] / 8, framebuffer.extent()[1] / 8];
        // let extent = [6 * framebuffer.extent()[0] / 8, 6 * framebuffer.extent()[1] / 8];
        let top_left = [0, 0];
        let extent = framebuffer.extent();
        builder.begin_render_pass(
            RenderPassBeginInfo {
                render_area_offset: top_left,
                render_area_extent: extent,
                clear_values: vec![Some(render_frame.clear_col.as_f32().into())],
                ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
            },
            SubpassBeginInfo::default(),
        )?;
        for shader in &mut self.shaders {
            shader.try_lock().unwrap().build_render_pass(&mut builder)?;
        }
        builder.end_render_pass(SubpassEndInfo::default())?;
        Ok(builder.build().map_err(Validated::unwrap)?)
    }

    pub(crate) fn get_receiver(&self) -> Arc<Mutex<RenderDataChannel>> {
        self.render_data_channel.clone()
    }
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub enum VertexDepth {
    Back(u16),
    #[default]
    Middle,
    Front(u16),
}

impl VertexDepth {
    pub fn min_value() -> Self { Self::Back(0) }
    pub fn max_value() -> Self { Self::Front(u16::MAX) }
}

impl PartialOrd for VertexDepth {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for VertexDepth {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        match self {
            VertexDepth::Back(depth) => {
                match other {
                    VertexDepth::Back(other_depth) => depth.cmp(other_depth),
                    _ => cmp::Ordering::Less,
                }
            },
            VertexDepth::Middle => {
                match other {
                    VertexDepth::Back(_) => cmp::Ordering::Greater,
                    VertexDepth::Middle => cmp::Ordering::Equal,
                    VertexDepth::Front(_) => cmp::Ordering::Less,
                }
            },
            VertexDepth::Front(depth) => {
                match other {
                    VertexDepth::Front(other_depth) => depth.cmp(other_depth),
                    _ => cmp::Ordering::Greater,
                }
            },
        }
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct VertexWithUV {
    pub xy: [f32; 2],
    pub uv: [f32; 2],
}

impl VertexWithUV {
    pub fn from_vertex(vertex: Vec2) -> Self {
        Self { xy: vertex.into(), uv: Vec2::zero().into() }
    }

    pub fn from_vec2s<I: IntoIterator<Item=Vec2>>(vertices: I) -> Vec<Self> {
        vertices.into_iter().map(Self::from_vertex).collect()
    }
    pub fn zip_from_vec2s<I: IntoIterator<Item=Vec2>, J: IntoIterator<Item=Vec2>>(vertices: I, uvs: J) -> Vec<Self> {
        vertices.into_iter().zip(uvs)
            .map(|(vertex, uv)| Self { xy: vertex.into(), uv: uv.into() })
            .collect()
    }
}

#[derive(Clone, Debug, Default)]
pub struct RenderItem {
    pub depth: VertexDepth,
    pub vertices: Vec<VertexWithUV>,
}

impl RenderItem {
    pub fn new(vertices: Vec<VertexWithUV>) -> Self {
        Self {
            depth: VertexDepth::Middle,
            vertices,
        }
    }
    #[must_use]
    pub fn with_depth(mut self, depth: VertexDepth) -> Self {
        self.depth = depth;
        self
    }

    #[must_use]
    pub fn concat(mut self, other: RenderItem) -> Self {
        self.vertices.extend(other.vertices);
        Self {
            depth: self.depth.max(other.depth),
            vertices: self.vertices,
        }
    }

    pub fn is_empty(&self) -> bool { self.vertices.is_empty() }
    pub fn len(&self) -> usize { self.vertices.len() }
}

pub(crate) struct VertexMap {
    render_items: BTreeMap<ObjectId, StoredRenderItem>,
    vertices_changed: bool,
}

impl VertexMap {
    pub(crate) fn new() -> Self {
        Self {
            render_items: BTreeMap::new(),
            vertices_changed: false,
        }
    }

    pub(crate) fn insert(&mut self, object_id: ObjectId, render_item: RenderItem) {
        check_eq!(render_item.len() % 3, 0);
        self.render_items.insert(object_id, StoredRenderItem { object_id, render_item });
        self.vertices_changed = true;
    }
    pub(crate) fn remove(&mut self, object_id: ObjectId) -> Option<RenderItem> {
        if let Some(removed) = self.render_items.remove(&object_id) {
            check_eq!(removed.object_id, object_id);
            self.vertices_changed = true;
            Some(removed.render_item)
        } else {
            None
        }
    }
    pub(crate) fn render_items(&self) -> impl Iterator<Item=&StoredRenderItem> {
        self.render_items.values()
    }
    pub(crate) fn len(&self) -> usize { self.render_items.len() }

    pub(crate) fn consume_vertices_changed(&mut self) -> bool {
        let rv = self.vertices_changed;
        self.vertices_changed = false;
        rv
    }
}

#[derive(Clone, Debug)]
pub(crate) struct StoredRenderItem {
    pub(crate) object_id: ObjectId,
    pub(crate) render_item: RenderItem,
}

impl StoredRenderItem {
    pub(crate) fn vertices(&self) -> &[VertexWithUV] { &self.render_item.vertices }
    pub(crate) fn len(&self) -> usize { self.render_item.len()}
}
