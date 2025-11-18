use crate::core::scene::GuiClosure;
use crate::core::tulivuori::buffer::{IndexBuffer32, VertexBuffer};
use crate::core::tulivuori::pipeline::Pipeline;
use crate::core::tulivuori::shader::{GuiVertFragShader, ShaderInfo, VertFragShader};
use crate::core::tulivuori::swapchain::{Swapchain, SwapchainBuilder};
use crate::core::tulivuori::texture::TvInternalTexture;
use crate::core::tulivuori::{GgWindow, RenderPerfStats};
use crate::core::tulivuori::{TvWindowContext, tv};
use crate::core::{ObjectId, prelude::*, tulivuori::GgViewport};
use crate::gui::GuiContext;
use crate::resource::ResourceHandler;
use crate::resource::texture::{MaterialId, TextureHandler};
use crate::shader::{ShaderId, SpriteVertex, vertex};
use crate::util::{UniqueShared, gg_err};
use ash::vk;
use egui::epaint;
use num_traits::ToPrimitive;
use std::collections::VecDeque;
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
    pub(crate) is_gui_enabled: bool,
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
            is_gui_enabled: false,
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

    swapchain: Swapchain,
    vertex_buffer: VertexBuffer<SpriteVertex>,
    shader: Arc<VertFragShader>,
    pipeline: Pipeline,

    gui: GuiRenderHandler,
    perf_stats: RenderPerfStats,
    last_perf_stats: Option<RenderPerfStats>,
}

impl RenderHandler {
    pub(crate) fn new(
        ctx: Arc<TvWindowContext>,
        gui_ctx: GuiContext,
        window: GgWindow,
        viewport: UniqueShared<GgViewport>,
        resource_handler: ResourceHandler,
        input_handler: Arc<Mutex<InputHandler>>,
    ) -> Result<Self> {
        let render_data_channel = RenderDataChannel::new(viewport.clone_inner());

        let swapchain = SwapchainBuilder::new(&ctx, window.inner.clone()).build()?;
        let vertex_buffer = VertexBuffer::new(ctx.clone(), &swapchain, 100 * 1024)?;
        let shader = Arc::new(VertFragShader::new(
            ctx.clone(),
            &mut Cursor::new(&include_bytes!("../shader/glsl/sprite-vert.spv")[..]),
            &mut Cursor::new(&include_bytes!("../shader/glsl/sprite-frag.spv")[..]),
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
        )?);
        let pipeline = Pipeline::new(
            ctx.clone(),
            &swapchain,
            &(shader.clone() as Arc<dyn ShaderInfo>),
            resource_handler.texture.pipeline_layout(),
            &viewport.lock(),
        )?;

        let gui = GuiRenderHandler::new(
            ctx.clone(),
            &swapchain,
            viewport.clone(),
            window.clone(),
            gui_ctx,
            input_handler,
            resource_handler.texture.clone(),
        )?;
        let perf_stats = RenderPerfStats::new(&window);

        Ok(Self {
            ctx,
            swapchain,
            vertex_buffer,
            shader,
            window,
            viewport,
            resource_handler,
            render_data_channel,
            update_sync: UpdateSync::new(),
            pipeline,
            gui,
            perf_stats,
            last_perf_stats: None,
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

    pub(crate) fn render_update(&mut self, egui_state: &mut egui_winit::State) -> Result<()> {
        unsafe {
            self.perf_stats.start();
            self.perf_stats.update_vertices.start();
            let (viewport, vertex_count, gui_commands, clear_col) = {
                let mut rx = self.render_data_channel.lock().unwrap();
                rx.last_render_stats.clone_from(&self.last_perf_stats);
                let mut viewport = self.viewport.lock();
                if let Some(extra_scale_factor) = rx.should_resize_with_scale_factor() {
                    viewport.set_extra_scale_factor(extra_scale_factor);
                }
                viewport.set_physical_top_left(rx.viewport.physical_top_left());
                let vertex_count = self.update_vertex_buffer(&rx.next_frame(), &viewport)?;
                let gui_commands = rx.gui_commands.drain(..).collect_vec();
                self.gui.is_gui_enabled = rx.is_gui_enabled;
                (viewport.clone(), vertex_count, gui_commands, rx.clear_col)
            };
            self.perf_stats.update_vertices.stop();

            self.perf_stats.update_gui.start();
            let do_render_gui = self.gui.pre_render_update(egui_state, gui_commands)?;
            self.perf_stats.update_gui.stop();

            self.perf_stats.acquire.start();
            let acquire = self.swapchain.acquire_next_image(&[])?;
            let draw_command_buffer = self.swapchain.acquire_present_command_buffer()?;
            self.perf_stats.acquire.stop();

            self.perf_stats.record_command_buffer.start();
            self.ctx.device().begin_command_buffer(
                draw_command_buffer,
                &tv::default_command_buffer_begin_info(),
            )?;
            self.resource_handler
                .texture
                .upload_all_pending_with(draw_command_buffer)?;
            self.swapchain
                .cmd_begin_rendering(draw_command_buffer, Some(clear_col));
            self.resource_handler.texture.bind(draw_command_buffer)?;
            let mut bytes = (viewport.physical_width() / viewport.combined_scale_factor())
                .to_le_bytes()
                .to_vec();
            bytes.extend(
                (viewport.physical_height() / viewport.combined_scale_factor())
                    .to_le_bytes()
                    .to_vec(),
            );
            self.pipeline
                .bind(draw_command_buffer, &viewport, &bytes, &[]);
            self.vertex_buffer.bind(draw_command_buffer);
            self.ctx
                .device()
                .cmd_draw(draw_command_buffer, vertex_count, 1, 0, 0);
            self.swapchain.cmd_end_rendering(draw_command_buffer);

            if do_render_gui {
                self.gui.do_render(draw_command_buffer, &self.swapchain)?;
            }

            self.ctx.device().end_command_buffer(draw_command_buffer)?;
            self.perf_stats.record_command_buffer.stop();
            self.perf_stats.submit.start();
            self.swapchain
                .submit_and_present_queue(&[draw_command_buffer])?;
            self.perf_stats.submit.stop();
            self.perf_stats.end_render.start();
            self.resource_handler.texture.on_render_done(&acquire)?;
            self.update_sync.mark_render_done();
            self.perf_stats.end_render.stop();
            self.last_perf_stats = self.perf_stats.end();
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
                        .is_material_ready(ri.material_id)?
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
            gui_ctx: self.gui.gui_ctx.clone(),
            window: self.window.clone(),
        }
    }

    pub fn vk_free(&self) {
        self.gui.vk_free();
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

struct GuiRenderHandler {
    ctx: Arc<TvWindowContext>,
    texture_handler: Arc<TextureHandler>,
    is_gui_enabled: bool,
    viewport: UniqueShared<GgViewport>,
    window: GgWindow,

    gui_ctx: GuiContext,
    input_handler: Arc<Mutex<InputHandler>>,
    gui_vertex_buffer: VertexBuffer<epaint::Vertex>,
    gui_index_buffer: IndexBuffer32,
    gui_shader: Arc<GuiVertFragShader>,
    gui_pipeline: Pipeline,

    last_meshes: Vec<egui::Mesh>,
    next_meshes: Vec<egui::Mesh>,
    font_texture: Option<Arc<TvInternalTexture>>,
    next_font_textures: VecDeque<Arc<TvInternalTexture>>,
}

impl GuiRenderHandler {
    fn new(
        ctx: Arc<TvWindowContext>,
        swapchain: &Swapchain,
        viewport: UniqueShared<GgViewport>,
        window: GgWindow,
        gui_ctx: GuiContext,
        input_handler: Arc<Mutex<InputHandler>>,
        texture_handler: Arc<TextureHandler>,
    ) -> Result<Self> {
        let gui_vertex_buffer = VertexBuffer::new(ctx.clone(), swapchain, 100 * 1024)?;
        let gui_index_buffer = IndexBuffer32::new(ctx.clone(), swapchain, 100 * 1024)?;
        let gui_shader = Arc::new(GuiVertFragShader::new(ctx.clone(), &texture_handler)?);
        let gui_pipeline = Pipeline::new(
            ctx.clone(),
            swapchain,
            &(gui_shader.clone() as Arc<dyn ShaderInfo>),
            gui_shader.pipeline_layout(),
            &viewport.lock(),
        )?;
        Ok(Self {
            ctx,
            texture_handler,
            is_gui_enabled: false,
            viewport,
            window,
            gui_ctx,
            input_handler,
            gui_vertex_buffer,
            gui_index_buffer,
            gui_shader,
            gui_pipeline,
            last_meshes: Vec::new(),
            next_meshes: Vec::new(),
            font_texture: None,
            next_font_textures: VecDeque::new(),
        })
    }

    fn pre_render_update(
        &mut self,
        egui_state: &mut egui_winit::State,
        mut gui_commands: Vec<Box<GuiClosure>>,
    ) -> Result<bool> {
        let egui_input = egui_state.take_egui_input(&self.window.inner);
        {
            let mut input = self.input_handler.lock().unwrap();
            input.set_viewport(self.viewport.clone_inner());
            input.update_mouse(&self.gui_ctx.inner);
        }
        if !self.gui_ctx.is_ever_enabled() {
            return Ok(false);
        }
        let full_output = self.gui_ctx.inner.run(egui_input, move |ctx| {
            for cmd in gui_commands.drain(..) {
                cmd(ctx);
            }
        });
        let image_deltas = full_output.textures_delta.set;
        egui_state.handle_platform_output(&self.window.inner, full_output.platform_output.clone());

        for (id, delta) in image_deltas {
            check_eq!(id, egui::TextureId::Managed(0));
            let bytes_per_pixel = delta
                .image
                .bytes_per_pixel()
                .to_i32()
                .context("bytes_per_pixel wrapped around")?;
            check_eq!(
                bytes_per_pixel,
                size_of::<egui::Color32>()
                    .to_i32()
                    .context("size_of::<egui::Color32>() wrapped around")?
            );
            #[allow(irrefutable_let_patterns)]
            let egui::ImageData::Color(color_image) = delta.image else {
                unreachable!()
            };
            self.update_font_texture(delta.pos, &color_image, bytes_per_pixel)?;
        }
        while self
            .next_font_textures
            .front()
            .is_some_and(|t| t.is_ready())
        {
            if let Some(last_font_texture) = self.font_texture.take() {
                self.texture_handler
                    .free_internal_texture(&last_font_texture)?;
            }
            self.font_texture = self.next_font_textures.pop_front();
        }
        let next_meshes = self
            .gui_ctx
            .inner
            .tessellate(full_output.shapes, full_output.pixels_per_point)
            .into_iter()
            .filter_map(|mesh| match mesh.primitive {
                epaint::Primitive::Mesh(m) => Some(m),
                epaint::Primitive::Callback(cb) => {
                    error!("epaint::Primitive::Callback() not implemented: {cb:?}");
                    None
                }
            })
            .collect_vec();
        if next_meshes.is_empty() && self.is_gui_enabled {
            // No updates to meshes since the last call to pre_render_update().
            self.next_meshes = self.last_meshes.clone();
        } else {
            self.last_meshes = self.next_meshes.drain(..).collect_vec();
            self.next_meshes = next_meshes;
        }
        self.gui_vertex_buffer.write(
            &self
                .next_meshes
                .iter()
                .flat_map(|m| m.vertices.clone())
                .collect_vec(),
        )?;
        self.gui_index_buffer.write(
            &self
                .next_meshes
                .iter()
                .flat_map(|m| m.indices.clone())
                .collect_vec(),
        )?;
        Ok(true)
    }

    fn update_font_texture(
        &mut self,
        image_pos: Option<[usize; 2]>,
        color_image: &Arc<egui::ColorImage>,
        bytes_per_pixel: i32,
    ) -> Result<()> {
        let color_image_data = color_image
            .pixels
            .iter()
            .flat_map(egui::Color32::to_array)
            .collect_vec();
        let color_image_extent = vk::Extent2D {
            width: color_image.width() as u32,
            height: color_image.height() as u32,
        };
        let color_image_width_in_bytes = bytes_per_pixel
            * color_image
                .width()
                .to_i32()
                .context("color_image.width() wrapped around")?;
        if let Some(old_font_texture) = self
            .next_font_textures
            .back()
            .or(self.font_texture.as_ref())
            .cloned()
        {
            let (new_data, new_extent) = if let Some(image_pos) = image_pos {
                check_le!(color_image_extent.width, old_font_texture.extent().width);
                check_le!(color_image_extent.height, old_font_texture.extent().height);
                check_le!(
                    image_pos[0] + color_image_extent.width as usize,
                    old_font_texture.extent().width as usize
                );
                check_le!(
                    image_pos[1] + color_image_extent.height as usize,
                    old_font_texture.extent().height as usize
                );
                check_eq!(
                    color_image_data
                        .len()
                        .to_u32()
                        .context("color_image_data.len() wrapped around")?
                        / (bytes_per_pixel
                            .to_u32()
                            .context("bytes_per_pixel is negative")?
                            * color_image_extent.width),
                    color_image_extent.height
                );
                let mut new_data = old_font_texture.data().to_vec();
                let origin = Vec2i {
                    x: image_pos[0]
                        .to_i32()
                        .context("image_pos[0] wrapped around")?
                        * bytes_per_pixel,
                    y: image_pos[1]
                        .to_i32()
                        .context("image_pos[0] wrapped around")?,
                };
                for (i, &byte) in color_image_data.iter().enumerate() {
                    let i = i
                        .to_i32()
                        .context("color_image_data count wrapped around")?;
                    let x = i % color_image_width_in_bytes;
                    let y = i / color_image_width_in_bytes;
                    let write_pos = origin + Vec2i { x, y };
                    new_data[write_pos.as_index(
                        old_font_texture.extent().width * bytes_per_pixel as u32,
                        old_font_texture.extent().height,
                    )] = byte;
                }
                (new_data, old_font_texture.extent())
            } else {
                (color_image_data, color_image_extent)
            };

            self.next_font_textures.push_back(
                self.texture_handler
                    .create_internal_texture(
                        new_extent,
                        vk::Format::R8G8B8A8_SRGB,
                        &new_data,
                        false,
                        Arc::new(AtomicBool::new(false)),
                    )?
                    .context("GuiRenderHandler::update_font_texture(): failed to create texture")?,
            );
        } else {
            check_is_none!(image_pos);
            self.next_font_textures.push_back(
                self.texture_handler
                    .create_internal_texture(
                        color_image_extent,
                        vk::Format::R8G8B8A8_SRGB,
                        &color_image_data,
                        false,
                        Arc::new(AtomicBool::new(false)),
                    )?
                    .unwrap(),
            );
        }
        Ok(())
    }

    fn do_render(
        &mut self,
        command_buffer: vk::CommandBuffer,
        swapchain: &Swapchain,
    ) -> Result<()> {
        unsafe {
            if let Some(font_texture) = self.font_texture.as_ref() {
                self.gui_shader.update_font_texture(font_texture, swapchain);
            }
            swapchain.cmd_begin_rendering(command_buffer, None);
            let viewport = self.viewport.lock();
            let mut vert_bytes = (viewport.physical_width() / viewport.combined_scale_factor()
                * 2.0)
                .to_le_bytes()
                .to_vec();
            vert_bytes.extend(
                (viewport.physical_height() / viewport.combined_scale_factor() * 2.0)
                    .to_le_bytes()
                    .to_vec(),
            );
            let frag_bytes = (swapchain.current_frame_index() as u32)
                .to_le_bytes()
                .to_vec();
            self.gui_pipeline
                .bind(command_buffer, &viewport, &vert_bytes, &frag_bytes);
            self.gui_shader.bind(command_buffer);
            self.gui_index_buffer.bind(command_buffer);
            self.gui_vertex_buffer.bind(command_buffer);
            let mut index = 0;
            let mut vertex = 0;
            for mesh in &self.next_meshes {
                self.ctx.device().cmd_draw_indexed(
                    command_buffer,
                    mesh.indices.len() as u32,
                    1,
                    index,
                    vertex,
                    0,
                );
                index += mesh
                    .indices
                    .len()
                    .to_u32()
                    .context("index count wrapped around")?;
                vertex += mesh
                    .vertices
                    .len()
                    .to_i32()
                    .context("vertex count wrapped around")?;
            }
            swapchain.cmd_end_rendering(command_buffer);
            Ok(())
        }
    }

    fn vk_free(&self) {
        self.gui_pipeline.vk_free();
        self.gui_shader.vk_free();
        self.gui_index_buffer.vk_free();
        self.gui_vertex_buffer.vk_free();
        if let Some(font_texture) = self.font_texture.as_ref() {
            gg_err::log_err_and_ignore(self.texture_handler.free_internal_texture(font_texture));
        }
        for font_texture in &self.next_font_textures {
            gg_err::log_err_and_ignore(self.texture_handler.free_internal_texture(font_texture));
        }
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
