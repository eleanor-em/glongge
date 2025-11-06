use crate::core::scene::GuiClosure;
use crate::core::tulivuori::buffer::VertexBuffer;
use crate::core::tulivuori::pipeline::Pipeline;
use crate::core::tulivuori::shader::VertFragShader;
use crate::core::tulivuori::swapchain::{Swapchain, SwapchainBuilder};
use crate::core::tulivuori::{GgWindow, RenderPerfStats};
use crate::core::tulivuori::{TvWindowContext, tv};
use crate::core::{ObjectId, prelude::*, tulivuori::GgViewport};
use crate::gui::GuiContext;
use crate::resource::ResourceHandler;
use crate::resource::texture::MaterialId;
use crate::shader::{ShaderId, SpriteVertex, vertex};
use crate::util::UniqueShared;
use ash::vk;
use std::io::Cursor;
use std::mem::offset_of;
use std::sync::atomic::{AtomicBool, Ordering};
use std::{
    cmp,
    collections::BTreeMap,
    default::Default,
    ops::Range,
    sync::{Arc, Mutex},
};

#[derive(Clone, Debug)]
pub struct ShaderExec {
    pub blend_col: Colour,
    pub material_id: MaterialId,
    pub shader_id: ShaderId,
}

impl Default for ShaderExec {
    fn default() -> Self {
        Self {
            blend_col: Colour::white(),
            material_id: 0,
            shader_id: ShaderId::default(),
        }
    }
}

/// Container for shader execution data and geometry information.
/// Public for the work-in-progress custom shader system.
#[derive(Clone, Debug)]
pub struct ShaderExecWithVertexData {
    pub inner: Vec<ShaderExec>,
    pub transform: Transform,
    pub vertex_indices: Range<u32>,
    pub depth: VertexDepth,
    pub clip: Rect,
}

pub(crate) struct RenderDataChannel {
    pub(crate) vertices: Vec<VertexWithCol>,
    pub(crate) shader_execs: Vec<ShaderExecWithVertexData>,
    pub(crate) gui_commands: Vec<Box<GuiClosure>>,
    pub(crate) gui_enabled: bool,
    viewport: GgViewport,
    should_resize: bool,
    clear_col: Colour,

    pub(crate) last_render_stats: Option<RenderPerfStats>,
}
impl RenderDataChannel {
    fn new(viewport: GgViewport) -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self {
            vertices: Vec::new(),
            shader_execs: Vec::new(),
            gui_commands: Vec::new(),
            gui_enabled: false,
            viewport,
            should_resize: false,
            clear_col: Colour::black(),
            last_render_stats: None,
        }))
    }

    pub(crate) fn next_frame(&self) -> RenderFrame<'_> {
        RenderFrame {
            vertices: &self.vertices,
            shader_execs: &self.shader_execs,
            clear_col: self.clear_col,
        }
    }

    pub(crate) fn current_viewport(&self) -> GgViewport {
        self.viewport.clone()
    }

    #[allow(clippy::float_cmp)]
    pub(crate) fn set_extra_scale_factor(&mut self, extra_scale_factor: f32) {
        if self.viewport.extra_scale_factor() != extra_scale_factor {
            self.viewport.set_extra_scale_factor(extra_scale_factor);
            self.should_resize = true;
        }
    }
    pub(crate) fn set_clear_col(&mut self, col: Colour) {
        self.clear_col = col;
    }
    pub(crate) fn get_clear_col(&mut self) -> Colour {
        self.clear_col
    }

    pub(crate) fn should_resize_with_scale_factor(&mut self) -> Option<f32> {
        let rv = self.should_resize;
        self.should_resize = false;
        if rv {
            Some(self.viewport.extra_scale_factor())
        } else {
            None
        }
    }

    pub(crate) fn set_viewport_physical_top_left(&mut self, top_left: Vec2) {
        self.viewport.set_physical_top_left(top_left);
    }
}

/// Public for the work-in-progress custom shader system.
#[derive(Clone)]
pub struct RenderFrame<'a> {
    pub vertices: &'a Vec<VertexWithCol>,
    pub shader_execs: &'a Vec<ShaderExecWithVertexData>,
    pub clear_col: Colour,
}

impl RenderFrame<'_> {
    fn for_shader(&self, _id: ShaderId) -> ShaderRenderFrame<'_> {
        // TODO: we could probably borrow shader_execs somehow.
        let shader_execs = self
            .shader_execs
            .clone()
            .into_iter()
            .map(move |mut ri| {
                ri.inner = ri
                    .inner
                    .into_iter()
                    // .filter(|ri| ri.shader_id == id)
                    .collect_vec();
                ri
            })
            .collect_vec();
        ShaderRenderFrame {
            vertices: self.vertices,
            render_infos: shader_execs,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct VertexWithCol {
    pub inner: Vec2,
    pub blend_col: Colour,
}

impl VertexWithCol {
    pub fn white(inner: Vec2) -> Self {
        Self {
            inner,
            blend_col: Colour::white(),
        }
    }
}

/// Public for the work-in-progress custom shader system.
pub struct ShaderRenderFrame<'a> {
    pub vertices: &'a [VertexWithCol],
    pub render_infos: Vec<ShaderExecWithVertexData>,
}

#[derive(Clone)]
pub(crate) struct UpdateSync {
    update_done: Arc<AtomicBool>,
    render_done: Arc<AtomicBool>,
}

impl UpdateSync {
    pub fn new() -> Self {
        Self {
            update_done: Arc::new(AtomicBool::new(false)),
            render_done: Arc::new(AtomicBool::new(false)),
        }
    }

    pub(crate) fn mark_update_done(&self) {
        let _ = self
            .update_done
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst);
    }
    pub(crate) fn mark_render_done(&self) {
        let _ = self
            .render_done
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst);
    }

    pub(crate) fn try_render_done(&self) -> bool {
        !SYNC_UPDATE_TO_RENDER
            || self
                .render_done
                .compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
    }

    pub(crate) fn wait_update_done(&self) {
        if SYNC_UPDATE_TO_RENDER {
            while self
                .update_done
                .compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst)
                .is_err()
            {
                // spin
            }
        }
    }
}

#[derive(Clone)]
pub(crate) struct RenderHandlerLite {
    pub(crate) render_data_channel: Arc<Mutex<RenderDataChannel>>,
    pub(crate) update_sync: UpdateSync,
    pub(crate) resource_handler: ResourceHandler,
    pub(crate) gui_ctx: GuiContext,
    pub(crate) window: GgWindow,
}

pub(crate) struct RenderHandler {
    ctx: Arc<TvWindowContext>,
    pub(crate) window: GgWindow,
    viewport: UniqueShared<GgViewport>,
    pub(crate) resource_handler: ResourceHandler,
    render_data_channel: Arc<Mutex<RenderDataChannel>>,
    update_sync: UpdateSync,
    pub(crate) gui_ctx: GuiContext,

    swapchain: Swapchain,
    vertex_buffer: VertexBuffer<SpriteVertex>,
    shader: Arc<VertFragShader>,
    pipeline: Pipeline,
}

impl RenderHandler {
    pub(crate) fn new(
        ctx: Arc<TvWindowContext>,
        gui_ctx: GuiContext,
        window: GgWindow,
        viewport: UniqueShared<GgViewport>,
        resource_handler: ResourceHandler,
    ) -> Result<Self> {
        let render_data_channel = RenderDataChannel::new(viewport.clone_inner());

        let swapchain = SwapchainBuilder::new(&ctx, window.inner.clone()).build()?;
        let vertex_buffer = VertexBuffer::new(ctx.clone(), &swapchain, 100 * 1024)?;
        let shader = VertFragShader::new(
            ctx.clone(),
            &mut Cursor::new(&include_bytes!("../shader/glsl/vert.spv")[..]),
            &mut Cursor::new(&include_bytes!("../shader/glsl/frag.spv")[..]),
            vec![vk::VertexInputBindingDescription {
                binding: 0,
                stride: size_of::<SpriteVertex>() as u32,
                input_rate: vk::VertexInputRate::VERTEX,
            }],
            vec![
                vk::VertexInputAttributeDescription {
                    location: 0,
                    binding: 0,
                    format: vk::Format::R32G32_SFLOAT,
                    offset: offset_of!(SpriteVertex, position) as u32,
                },
                vk::VertexInputAttributeDescription {
                    location: 1,
                    binding: 0,
                    format: vk::Format::R32G32_SFLOAT,
                    offset: offset_of!(SpriteVertex, translation) as u32,
                },
                vk::VertexInputAttributeDescription {
                    location: 2,
                    binding: 0,
                    format: vk::Format::R32_SFLOAT,
                    offset: offset_of!(SpriteVertex, rotation) as u32,
                },
                vk::VertexInputAttributeDescription {
                    location: 3,
                    binding: 0,
                    format: vk::Format::R32G32_SFLOAT,
                    offset: offset_of!(SpriteVertex, scale) as u32,
                },
                vk::VertexInputAttributeDescription {
                    location: 4,
                    binding: 0,
                    format: vk::Format::R32_UINT,
                    offset: offset_of!(SpriteVertex, material_id) as u32,
                },
                vk::VertexInputAttributeDescription {
                    location: 5,
                    binding: 0,
                    format: vk::Format::R32G32B32A32_SFLOAT,
                    offset: offset_of!(SpriteVertex, blend_col) as u32,
                },
                vk::VertexInputAttributeDescription {
                    location: 6,
                    binding: 0,
                    format: vk::Format::R32G32_SFLOAT,
                    offset: offset_of!(SpriteVertex, clip_min) as u32,
                },
                vk::VertexInputAttributeDescription {
                    location: 7,
                    binding: 0,
                    format: vk::Format::R32G32_SFLOAT,
                    offset: offset_of!(SpriteVertex, clip_max) as u32,
                },
            ],
        )?;
        let pipeline = Pipeline::new(
            ctx.clone(),
            &swapchain,
            &shader,
            resource_handler.texture.pipeline_layout(),
            &viewport.lock(),
        )?;

        Ok(Self {
            ctx,
            gui_ctx,
            swapchain,
            vertex_buffer,
            shader,
            window,
            viewport,
            resource_handler,
            render_data_channel,
            update_sync: UpdateSync::new(),
            pipeline,
        })
    }
    #[must_use]
    pub(crate) fn with_extra_scale_factor(self, extra_scale_factor: f32) -> Self {
        self.viewport
            .lock()
            .set_extra_scale_factor(extra_scale_factor);
        {
            let mut rc = self.render_data_channel.lock().unwrap();
            rc.set_extra_scale_factor(extra_scale_factor);
            let _ = rc.should_resize_with_scale_factor();
        }
        self
    }

    #[must_use]
    pub(crate) fn with_clear_col(self, clear_col: Colour) -> Self {
        self.render_data_channel
            .lock()
            .unwrap()
            .set_clear_col(clear_col);
        self
    }

    pub(crate) fn wait_update_done(&self) {
        self.update_sync.wait_update_done();
    }

    pub(crate) fn render_update(&mut self) -> Result<()> {
        unsafe {
            let (viewport, vertex_count, clear_col) = {
                let mut rx = self.render_data_channel.lock().unwrap();
                let mut viewport = self.viewport.lock();
                if let Some(extra_scale_factor) = rx.should_resize_with_scale_factor() {
                    viewport.set_extra_scale_factor(extra_scale_factor);
                }
                viewport.set_physical_top_left(rx.viewport.physical_top_left());
                let vertex_count = self.update_vertex_buffer(&rx.next_frame(), &viewport)?;
                (viewport.clone(), vertex_count, rx.clear_col)
            };
            // TODO: in theory this should not be necessary, there is some kind of bug where
            // removing it causes a deadlock.
            self.resource_handler.texture.wait_for_upload()?;

            let acquire = self.swapchain.acquire_next_image(&[])?;
            let draw_command_buffer = self.swapchain.acquire_present_command_buffer()?;
            self.ctx.device().begin_command_buffer(
                draw_command_buffer,
                &tv::default_command_buffer_begin_info(),
            )?;
            self.swapchain
                .cmd_begin_rendering(draw_command_buffer, clear_col);
            self.resource_handler.texture.bind(draw_command_buffer);
            self.pipeline.bind(draw_command_buffer, &viewport);
            self.vertex_buffer.bind(draw_command_buffer);
            self.ctx
                .device()
                .cmd_draw(draw_command_buffer, vertex_count, 1, 0, 0);
            self.swapchain.cmd_end_rendering(draw_command_buffer);
            self.ctx.device().end_command_buffer(draw_command_buffer)?;

            self.swapchain
                .submit_and_present_queue(&[draw_command_buffer])?;
            self.resource_handler.texture.on_render_done(&acquire)?;
            self.update_sync.mark_render_done();
        }
        Ok(())
    }

    fn update_vertex_buffer(
        &self,
        render_frame: &RenderFrame,
        viewport: &GgViewport,
    ) -> Result<u32> {
        let shader_render_frame = render_frame.for_shader(ShaderId::default());
        let render_infos = shader_render_frame
            .render_infos
            .iter()
            .sorted_unstable_by_key(|item| item.depth);
        let mut vertices = Vec::new();
        for render_info in render_infos {
            for vertex_index in render_info.vertex_indices.clone() {
                let vertex = render_frame.vertices[vertex_index as usize];
                for ri in &render_info.inner {
                    if self
                        .resource_handler
                        .texture
                        .is_material_ready(ri.material_id)
                    {
                        vertices.push(SpriteVertex {
                            position: vertex.inner.into(),
                            material_id: ri.material_id,
                            translation: (render_info.transform.centre - viewport.world_top_left())
                                .into(),
                            rotation: render_info.transform.rotation,
                            scale: render_info.transform.scale.into(),
                            blend_col: (vertex.blend_col * ri.blend_col).into(),
                            clip_min: (render_info.clip.top_left()
                                * viewport.combined_scale_factor())
                            .into(),
                            clip_max: (render_info.clip.bottom_right()
                                * viewport.combined_scale_factor())
                            .into(),
                        });
                    } else {
                        error!(
                            "material not ready: {:?} {:?}",
                            ri.material_id,
                            self.resource_handler
                                .texture
                                .material_to_texture(ri.material_id)
                        );
                    }
                }
            }
        }
        self.vertex_buffer.write(&vertices)?;
        Ok(vertices.len() as u32)
    }

    pub(crate) fn as_lite(&self) -> RenderHandlerLite {
        RenderHandlerLite {
            render_data_channel: self.render_data_channel.clone(),
            update_sync: self.update_sync.clone(),
            resource_handler: self.resource_handler.clone(),
            gui_ctx: self.gui_ctx.clone(),
            window: self.window.clone(),
        }
    }

    pub fn vk_free(&self) {
        self.pipeline.vk_free();
        self.shader.vk_free();
        self.vertex_buffer.vk_free();
        self.resource_handler.texture.vk_free();
        self.swapchain.vk_free();
        self.ctx.vk_free();
    }
}

impl Drop for RenderHandler {
    fn drop(&mut self) {
        if !self.ctx.did_vk_free() {
            error!("leaked resource: RenderHandler");
        }
        info!("RenderHandler dropped, all Vulkan objects should have been freed");
    }
}

/// Represents the depth ordering of vertices.
///
/// The depth determines the rendering order of vertices, with three main layers:
/// - `Back`: Renders behind the middle layer; 0 is the backmost value.
/// - `Middle`: The default middle layer between back and front
/// - `Front`: Renders in front of the middle layer; [`u16::MAX`] is the frontmost value.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub enum VertexDepth {
    /// Back layer with u16 depth value (0 = furthest back)
    Back(u16),
    /// Middle layer between back and front
    #[default]
    Middle,
    /// Front layer with u16 depth value ([`u16::MAX`] = furthest front)
    Front(u16),
}

impl VertexDepth {
    pub fn min_value() -> Self {
        Self::Back(0)
    }
    pub fn max_value() -> Self {
        Self::Front(u16::MAX)
    }

    #[must_use]
    pub fn next_smaller(self) -> Self {
        match self {
            VertexDepth::Back(depth) => VertexDepth::Back(depth.saturating_sub(1)),
            VertexDepth::Middle => VertexDepth::Back(u16::MAX),
            VertexDepth::Front(0) => VertexDepth::Middle,
            VertexDepth::Front(depth) => VertexDepth::Front(depth.saturating_sub(1)),
        }
    }
    #[must_use]
    pub fn next_larger(self) -> Self {
        match self {
            VertexDepth::Back(u16::MAX) => VertexDepth::Middle,
            VertexDepth::Back(depth) => VertexDepth::Back(depth.saturating_add(1)),
            VertexDepth::Middle => VertexDepth::Front(0),
            VertexDepth::Front(depth) => VertexDepth::Front(depth.saturating_add(1)),
        }
    }
}

impl PartialOrd for VertexDepth {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for VertexDepth {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        match self {
            VertexDepth::Back(depth) => match other {
                VertexDepth::Back(other_depth) => depth.cmp(other_depth),
                _ => cmp::Ordering::Less,
            },
            VertexDepth::Middle => match other {
                VertexDepth::Back(_) => cmp::Ordering::Greater,
                VertexDepth::Middle => cmp::Ordering::Equal,
                VertexDepth::Front(_) => cmp::Ordering::Less,
            },
            VertexDepth::Front(depth) => match other {
                VertexDepth::Front(other_depth) => depth.cmp(other_depth),
                _ => cmp::Ordering::Greater,
            },
        }
    }
}

/// A list of coloured vertices to be rendered by a shader at a fixed depth.
#[derive(Clone, Debug)]
pub struct RenderItem {
    pub vertices: Vec<VertexWithCol>,
    pub depth: VertexDepth,
    pub clip: Rect,
}

impl RenderItem {
    pub fn new(vertices: Vec<VertexWithCol>) -> Self {
        Self {
            vertices,
            ..Self::default()
        }
    }
    pub fn from_raw_vertices(vertices: Vec<Vec2>) -> Self {
        Self::new(vertex::map_raw_vertices(vertices))
    }
    #[must_use]
    pub fn with_depth(mut self, depth: VertexDepth) -> Self {
        self.depth = depth;
        self
    }
    #[must_use]
    pub fn with_blend_col(mut self, col: Colour) -> Self {
        for vertex in &mut self.vertices {
            vertex.blend_col = col;
        }
        self
    }
    #[must_use]
    pub fn with_clip(mut self, clip: Rect) -> Self {
        self.clip = clip;
        self
    }

    /// Concatenates this render item with another one.
    /// Takes the maximum depth between the two items.
    #[must_use]
    pub fn concat(mut self, other: RenderItem) -> Self {
        self.vertices.extend(other.vertices);
        check_eq!(self.clip, other.clip);
        Self {
            vertices: self.vertices,
            depth: self.depth.max(other.depth),
            clip: self.clip,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty()
    }
    pub fn len(&self) -> usize {
        self.vertices.len()
    }
}

impl Default for RenderItem {
    fn default() -> Self {
        Self {
            vertices: vec![],
            depth: VertexDepth::default(),
            clip: Rect::unbounded(),
        }
    }
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
        self.render_items.insert(
            object_id,
            StoredRenderItem {
                object_id,
                render_item,
            },
        );
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
    pub(crate) fn render_items(&self) -> impl Iterator<Item = &StoredRenderItem> {
        self.render_items.values()
    }
    pub(crate) fn len(&self) -> usize {
        self.render_items.len()
    }

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
    pub(crate) fn vertices(&self) -> &[VertexWithCol] {
        &self.render_item.vertices
    }
    pub(crate) fn len(&self) -> usize {
        self.render_item.len()
    }
}
