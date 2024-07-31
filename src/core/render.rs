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

use crate::{
    core::{
        prelude::*,
        util::{
            colour::Colour,
            linalg::{Transform, Vec2}
        },
        vk::{
            AdjustedViewport,
            VulkanoContext,
            WindowContext,
        },
        ObjectId,
    },
    resource::{
        ResourceHandler,
        texture::{Texture, TextureSubArea}
    },
};
use crate::shader::{ShaderName, ShaderPair, ShaderWithPipeline};

#[derive(Clone, Debug)]
pub struct RenderInfo {
    pub col: Colour,
    pub texture: Option<Texture>,
    pub texture_sub_area: TextureSubArea,
}

impl Default for RenderInfo {
    fn default() -> Self {
        Self { col: Colour::white(), texture: None, texture_sub_area: TextureSubArea::default() }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct RenderInfoFull {
    pub(crate) inner: RenderInfo,
    pub(crate) transform: Transform,
    pub(crate) vertex_indices: Range<usize>,
}

#[derive(Clone)]
pub struct RenderInfoReceiver {
    pub(crate) vertices: Vec<VertexWithUV>,
    vertices_up_to_date: bool,
    pub(crate) render_info: Vec<RenderInfoFull>,
    viewport: AdjustedViewport,
    clear_col: Colour,
}
impl RenderInfoReceiver {
    fn new(viewport: AdjustedViewport) -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self {
            vertices: Vec::new(),
            vertices_up_to_date: false,
            render_info: Vec::new(),
            viewport,
            clear_col: Colour::black(),
        }))
    }
}
impl RenderInfoReceiver {
    pub(crate) fn update_vertices(&mut self, vertices: Vec<VertexWithUV>) {
        self.vertices_up_to_date = false;
        self.vertices = vertices;
    }

    pub(crate) fn update_render_info(&mut self, render_info: Vec<RenderInfoFull>) {
        self.render_info = render_info;
    }

    pub(crate) fn current_viewport(&self) -> AdjustedViewport {
        self.viewport.clone()
    }

    pub(crate) fn is_ready(&self) -> bool {
        !self.vertices.is_empty() && !self.render_info.is_empty()
    }

    pub(crate) fn get_clear_col(&self) -> Colour { self.clear_col }
    pub(crate) fn set_clear_col(&mut self, col: Colour) { self.clear_col = col; }
}

#[derive(Clone)]
pub struct RenderHandler {
    render_info_receiver: Arc<Mutex<RenderInfoReceiver>>,
    viewport: Arc<Mutex<AdjustedViewport>>,
    shaders: BTreeMap<ShaderName, ShaderWithPipeline>,
    command_buffer: Option<Arc<PrimaryAutoCommandBuffer>>,
}

impl RenderHandler {
    pub fn new(window_ctx: &WindowContext, ctx: &VulkanoContext, resource_handler: ResourceHandler) -> Result<Self> {
        let viewport = Arc::new(Mutex::new(window_ctx.create_default_viewport()));
        let mut shaders = BTreeMap::new();
        let basic_shader = ShaderPair::new_basic(ctx.device().clone())?;
        shaders.insert(basic_shader.name(), ShaderWithPipeline::new(basic_shader, viewport.clone(), resource_handler.clone()));
        let render_info_receiver = RenderInfoReceiver::new(viewport.try_lock().unwrap().clone());
        Ok(Self {
            shaders,
            viewport,
            command_buffer: None,
            render_info_receiver,
        })
    }

    #[must_use]
    pub fn with_global_scale_factor(self, global_scale_factor: f64) -> Self {
        self.viewport.try_lock().unwrap().set_global_scale_factor(global_scale_factor);
        self
    }
}

impl RenderHandler {
    pub(crate) fn on_resize(
        &mut self,
        _ctx: &VulkanoContext,
        window: &Arc<Window>,
    ) -> Result<()> {
        self.viewport.try_lock().unwrap().update_from_window(window);
        self.command_buffer = None;
        self.render_info_receiver.lock().unwrap().viewport = self.viewport.try_lock().unwrap().clone();
        Ok(())
    }

    pub(crate) fn on_render(
        &mut self,
        ctx: &VulkanoContext,
        framebuffer: &Arc<Framebuffer>,
    ) -> Result<Arc<PrimaryAutoCommandBuffer>> {
        {
            let mut receiver = self.render_info_receiver.lock().unwrap();
            for shader in self.shaders.values_mut() {
                if !receiver.vertices_up_to_date {
                    shader.reset_vertex_buffer();
                }
                shader.on_render(ctx, &mut receiver)?;
            }
            receiver.vertices_up_to_date = true;
        }
        let mut builder = AutoCommandBufferBuilder::primary(
            ctx.command_buffer_allocator(),
            ctx.queue().queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;
        for shader in self.shaders.values_mut() {
            shader.build_render_pass(ctx, framebuffer.clone(), &mut builder)?;
        }
        Ok(builder.build().map_err(Validated::unwrap)?)
    }

    pub(crate) fn get_receiver(&self) -> Arc<Mutex<RenderInfoReceiver>> {
        self.render_info_receiver.clone()
    }
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub enum VertexDepth {
    Back(u64),
    #[default]
    Middle,
    Front(u64),
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
    pub vertex: Vec2,
    pub uv: Vec2,
}

impl VertexWithUV {
    pub fn from_vertex(vertex: Vec2) -> Self {
        Self { vertex, uv: Vec2::zero() }
    }

    pub fn from_vec2s<I: IntoIterator<Item=Vec2>>(vertices: I) -> Vec<Self> {
        vertices.into_iter().map(Self::from_vertex).collect()
    }
    pub fn zip_from_vec2s<I: IntoIterator<Item=Vec2>, J: IntoIterator<Item=Vec2>>(vertices: I, uvs: J) -> Vec<Self> {
        vertices.into_iter().zip(uvs)
            .map(|(vertex, uv)| Self { vertex, uv })
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
    vertex_count: usize,
    vertices_changed: bool,
}

impl VertexMap {
    pub(crate) fn new() -> Self {
        Self {
            render_items: BTreeMap::new(),
            vertex_count: 0,
            vertices_changed: false,
        }
    }

    pub(crate) fn insert(&mut self, object_id: ObjectId, render_item: RenderItem) {
        self.vertex_count += render_item.len();
        check_eq!(render_item.len() % 3, 0);
        self.render_items.insert(object_id, StoredRenderItem { object_id, render_item });
        self.vertices_changed = true;
    }
    pub(crate) fn remove(&mut self, object_id: ObjectId) -> Option<RenderItem> {
        if let Some(removed) = self.render_items.remove(&object_id) {
            check_eq!(removed.object_id, object_id);
            self.vertex_count -= removed.len();
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
    pub(crate) fn vertex_count(&self) -> usize { self.vertex_count }

    pub(crate) fn consume_vertices_changed(&mut self) -> bool {
        let rv = self.vertices_changed;
        self.vertices_changed = false;
        rv
    }
}

#[derive(Clone, Debug)]
pub(crate) struct StoredRenderItem {
    pub(crate) object_id: ObjectId,
    render_item: RenderItem,
}

impl StoredRenderItem {
    pub(crate) fn vertices(&self) -> &[VertexWithUV] { &self.render_item.vertices }
    pub(crate) fn len(&self) -> usize { self.render_item.len()}
}
