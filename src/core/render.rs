use crate::core::input::InputHandler;
use crate::core::scene::GuiClosure;
use crate::core::vk::vk_ctx::VulkanoContext;
use crate::core::vk::{GgWindow, RenderPerfStats};
use crate::core::{ObjectId, prelude::*, vk::AdjustedViewport};
use crate::gui::GuiContext;
use crate::gui::render::GuiRenderer;
use crate::resource::texture::MaterialId;
use crate::shader::{Shader, ShaderId, vertex};
use crate::util::{UniqueShared, gg_err};
use egui::FullOutput;
use std::sync::atomic::{AtomicBool, Ordering};
use std::{
    cmp,
    collections::BTreeMap,
    default::Default,
    ops::Range,
    sync::{Arc, Mutex},
};
use vulkano::command_buffer::{RenderingAttachmentInfo, RenderingInfo};
use vulkano::image::Image;
use vulkano::render_pass::AttachmentLoadOp::Clear;
use vulkano::render_pass::AttachmentStoreOp::Store;
use vulkano::swapchain::Swapchain;
use vulkano_taskgraph::command_buffer::RecordingCommandBuffer;
use vulkano_taskgraph::graph::{NodeId, TaskGraph};
use vulkano_taskgraph::resource::{AccessTypes, HostAccessType, ImageLayoutType};
use vulkano_taskgraph::{Id, QueueFamilyType, Task, TaskContext, TaskResult};

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
    viewport: AdjustedViewport,
    should_resize: bool,
    clear_col: Colour,

    pub(crate) last_render_stats: Option<RenderPerfStats>,
}
impl RenderDataChannel {
    fn new(viewport: AdjustedViewport) -> Arc<Mutex<Self>> {
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

    pub(crate) fn current_viewport(&self) -> AdjustedViewport {
        self.viewport.clone()
    }

    #[allow(clippy::float_cmp)]
    pub(crate) fn set_global_scale_factor(&mut self, global_scale_factor: f32) {
        if self.viewport.global_scale_factor() != global_scale_factor {
            self.viewport.set_global_scale_factor(global_scale_factor);
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
            Some(self.viewport.global_scale_factor())
        } else {
            None
        }
    }

    pub(crate) fn set_translation(&mut self, translation: Vec2) {
        self.viewport.translation = translation;
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
    fn for_shader(&self, id: ShaderId) -> ShaderRenderFrame<'_> {
        // TODO: we could probably borrow shader_execs somehow.
        let shader_execs = self
            .shader_execs
            .clone()
            .into_iter()
            .map(move |mut ri| {
                ri.inner = ri
                    .inner
                    .into_iter()
                    .filter(|ri| ri.shader_id == id)
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
pub(crate) struct RenderHandler {
    pub(crate) gui_ctx: GuiContext,
    render_data_channel: Arc<Mutex<RenderDataChannel>>,
    update_sync: UpdateSync,
    pub(crate) resource_handler: ResourceHandler,
    pub(crate) window: GgWindow,
    viewport: UniqueShared<AdjustedViewport>,
    shaders: Vec<UniqueShared<Box<dyn Shader>>>,
    gui_shader: GuiRenderer,
    last_full_output: UniqueShared<Option<FullOutput>>,
}

impl RenderHandler {
    pub(crate) fn new(
        vk_ctx: &VulkanoContext,
        gui_ctx: GuiContext,
        resource_handler: ResourceHandler,
        window: GgWindow,
        viewport: UniqueShared<AdjustedViewport>,
        shaders: Vec<UniqueShared<Box<dyn Shader>>>,
    ) -> Result<Self> {
        let render_data_channel = RenderDataChannel::new(viewport.clone_inner());
        for (a, b) in shaders.iter().tuple_combinations() {
            check_ne!(
                a.lock().name_concrete(),
                b.lock().name_concrete(),
                "duplicate shader name"
            );
        }
        let gui_shader = GuiRenderer::new(vk_ctx.clone(), viewport.clone())?;
        Ok(Self {
            gui_ctx,
            render_data_channel,
            update_sync: UpdateSync::new(),
            resource_handler,
            window,
            viewport,
            shaders,
            last_full_output: UniqueShared::new(None),
            gui_shader,
        })
    }

    #[must_use]
    pub(crate) fn with_global_scale_factor(self, global_scale_factor: f32) -> Self {
        self.viewport
            .lock()
            .set_global_scale_factor(global_scale_factor);
        {
            let mut rc = self.render_data_channel.lock().unwrap();
            rc.set_global_scale_factor(global_scale_factor);
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

    pub(crate) fn get_viewport(&self) -> AdjustedViewport {
        self.viewport.lock().clone()
    }

    pub(crate) fn get_receiver(&self) -> (Arc<Mutex<RenderDataChannel>>, UpdateSync) {
        (self.render_data_channel.clone(), self.update_sync.clone())
    }

    pub(crate) fn wait_update_done(&self) {
        self.update_sync.wait_update_done();
    }

    pub(crate) fn on_recreate_swapchain(&self) {
        self.viewport.lock().update_from_window(&self.window);
        self.render_data_channel.lock().unwrap().viewport = self.viewport.lock().clone();
    }

    pub(crate) fn do_gui(
        &self,
        ctx: &GuiContext,
        platform: &mut egui_winit::State,
        input: &Arc<Mutex<InputHandler>>,
        last_render_stats: Option<&RenderPerfStats>,
    ) {
        let egui_input = platform.take_egui_input(&self.window.inner);
        {
            let mut input = input.lock().unwrap();
            input.set_viewport(self.get_viewport());
            input.update_mouse(&ctx.inner);
        }
        if !ctx.is_ever_enabled() {
            return;
        }
        let full_output = ctx.inner.run(egui_input, |ctx| {
            let gui_commands = {
                let mut channel = self.render_data_channel.lock().unwrap();
                channel.last_render_stats = last_render_stats.cloned();
                channel.gui_commands.drain(..).collect_vec()
            };
            for cmd in gui_commands {
                cmd(ctx);
            }
        });
        platform.handle_platform_output(&self.window.inner, full_output.platform_output.clone());
        self.last_full_output.lock().replace(full_output);
    }

    pub(crate) fn build_shader_task_graphs(
        &self,
        task_graph: &mut TaskGraph<VulkanoContext>,
        texture_node: NodeId,
        virtual_swapchain_id: Id<Swapchain>,
        textures: &[Id<Image>],
    ) -> Result<()> {
        // Host buffer accesses
        for buffer in self.shaders.iter().flat_map(|s| s.lock().buffer_writes()) {
            task_graph.add_host_buffer_access(buffer, HostAccessType::Write);
        }
        for buffer in self.gui_shader.buffer_writes() {
            task_graph.add_host_buffer_access(buffer, HostAccessType::Write);
        }

        // Preparation for the render
        let mut pre_render_node = task_graph.create_task_node(
            "pre_render_handler",
            QueueFamilyType::Graphics,
            PreRenderTask {
                handler: self.clone(),
                resource_handler: self.resource_handler.clone(),
            },
        );
        for image in self.gui_shader.image_writes() {
            pre_render_node.image_access(
                image,
                AccessTypes::COPY_TRANSFER_WRITE,
                ImageLayoutType::Optimal,
            );
        }
        let pre_render_node = pre_render_node.build();
        task_graph.add_edge(texture_node, pre_render_node)?;
        let clear_node = task_graph
            .create_task_node(
                "clear_handler",
                QueueFamilyType::Graphics,
                ClearTask {
                    handler: self.clone(),
                },
            )
            .image_access(
                virtual_swapchain_id.current_image_id(),
                AccessTypes::COLOR_ATTACHMENT_WRITE,
                ImageLayoutType::Optimal,
            )
            .build();
        task_graph.add_edge(pre_render_node, clear_node)?;

        // The actual render
        let shader_nodes = self
            .shaders
            .iter()
            .map(|s| {
                s.lock()
                    .build_task_node(task_graph, virtual_swapchain_id, textures)
            })
            .collect_vec();
        if let Some(&first_shader) = shader_nodes.first() {
            task_graph.add_edge(clear_node, first_shader)?;
        }
        let last_non_gui_node = *shader_nodes.last().unwrap_or(&clear_node);
        for (a, b) in shader_nodes.into_iter().tuple_windows() {
            task_graph.add_edge(a, b)?;
        }

        // GUI
        let gui_node = self
            .gui_shader
            .build_task_graph(task_graph, virtual_swapchain_id);
        task_graph.add_edge(last_non_gui_node, gui_node)?;

        let mut post_render_node = task_graph.create_task_node(
            "post_render_handler",
            QueueFamilyType::Graphics,
            PostRenderTask {
                handler: self.clone(),
                resource_handler: self.resource_handler.clone(),
            },
        );
        let post_render_node = post_render_node.build();
        task_graph.add_edge(gui_node, post_render_node)?;

        Ok(())
    }

    pub(crate) fn should_build_task_graph(&self) -> bool {
        self.gui_shader.should_build_task_graph() ||
            self.shaders.iter().any(|s| s.lock().should_build_task_graph())
    }
}

/// Task responsible for preparing render data before actual rendering.
/// Handles viewport updates, shader pre-render updates, and GUI primitive tessellation.
/// Updates render frame data from the render data channel and propagates it to shaders.
struct PreRenderTask {
    handler: RenderHandler,
    resource_handler: ResourceHandler,
}

impl Task for PreRenderTask {
    type World = VulkanoContext;

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer,
        tcx: &mut TaskContext,
        world: &Self::World,
    ) -> TaskResult {
        if self.resource_handler.texture.is_not_yet_initialised() {
            warn!("resource handler not yet initialised, skip PreRenderTask::execute()");
            return Ok(());
        }
        let image_idx = tcx
            .swapchain(world.swapchain_id())
            .unwrap()
            .current_image_index()
            .unwrap() as usize;
        let mut rx = self.handler.render_data_channel.lock().unwrap();
        let (global_scale_factor, render_frame) = {
            self.handler.viewport.lock().translation = rx.viewport.translation;
            let global_scale_factor = rx.should_resize_with_scale_factor();
            *self.handler.gui_shader.gui_enabled.lock() = rx.gui_enabled;
            (global_scale_factor, rx.next_frame())
        };
        world.perf_stats().lap("PreRenderTask: next_frame()");
        if let Some(global_scale_factor) = global_scale_factor {
            self.handler
                .viewport
                .lock()
                .set_global_scale_factor(global_scale_factor);
            self.handler.on_recreate_swapchain();
        }
        for mut shader in self.handler.shaders.iter().map(|s| s.lock()) {
            let shader_id = shader.id();
            shader
                .pre_render_update(image_idx, render_frame.for_shader(shader_id), tcx)
                .map_err(gg_err::CatchOutOfDate::from)
                .unwrap();
        }
        world
            .perf_stats()
            .lap("PreRenderTask: shader.pre_render_update()");
        if self.handler.gui_ctx.is_ever_enabled() {
            let full_output = self
                .handler
                .last_full_output
                .take_inner()
                .expect("GUI output missing");
            let primitives = self
                .handler
                .gui_ctx
                .inner
                .tessellate(full_output.shapes, full_output.pixels_per_point);
            world
                .perf_stats()
                .lap("PreRenderTask: gui_ctx.tessellate()");
            *self.handler.gui_shader.primitives.lock() = Some(primitives);
            self.handler
                .gui_shader
                .pre_render_update(cbf, tcx, world, &full_output.textures_delta.set)
                .unwrap();
            world
                .perf_stats()
                .lap("PreRenderTask: gui_shader.pre_render_update()");
        }
        Ok(())
    }
}

/// Task responsible for clearing the swapchain image with the set clear colour before rendering.
struct ClearTask {
    handler: RenderHandler,
}

impl Task for ClearTask {
    type World = VulkanoContext;

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer,
        tcx: &mut TaskContext,
        world: &Self::World,
    ) -> TaskResult {
        let image_idx = tcx
            .swapchain(world.swapchain_id())
            .unwrap()
            .current_image_index()
            .unwrap() as usize;
        let image_view = world.current_image_view(image_idx);
        let viewport_extent = self.handler.viewport.lock().inner().extent;
        unsafe {
            cbf.as_raw()
                .begin_rendering(&RenderingInfo {
                    color_attachments: vec![Some(RenderingAttachmentInfo {
                        clear_value: Some(
                            self.handler
                                .render_data_channel
                                .lock()
                                .unwrap()
                                .clear_col
                                .as_f32()
                                .into(),
                        ),
                        load_op: Clear,
                        store_op: Store,
                        ..RenderingAttachmentInfo::image_view(image_view)
                    })],
                    render_area_extent: [viewport_extent[0] as u32, viewport_extent[1] as u32],
                    layer_count: 1,
                    ..Default::default()
                })
                .unwrap();
            cbf.as_raw().end_rendering().unwrap();
        }
        world.perf_stats().lap("ClearTask: done");

        Ok(())
    }
}

struct PostRenderTask {
    handler: RenderHandler,
    resource_handler: ResourceHandler,
}

impl Task for PostRenderTask {
    type World = VulkanoContext;

    unsafe fn execute(
        &self,
        _cbf: &mut RecordingCommandBuffer,
        _tcx: &mut TaskContext,
        world: &Self::World,
    ) -> TaskResult {
        if self.resource_handler.texture.is_not_yet_initialised() {
            warn!("resource handler not yet initialised, skip PostRenderTask::execute()");
            return Ok(());
        }
        self.resource_handler.texture.wait_free_unused_textures();
        self.handler.update_sync.mark_render_done();

        world
            .perf_stats()
            .lap("PostRenderTask: wait_free_unused_textures()");
        Ok(())
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
