use crate::core::tulivuori::texture::TvInternalTexture;
use crate::util::gg_sync::GgMutex;
use crate::{
    core::prelude::*,
    core::tulivuori::TvWindowContext,
    core::tulivuori::swapchain::SwapchainAcquireInfo,
    core::tulivuori::texture::{TextureId, TextureManager},
};
use asefile::AsepriteFile;
use ash::vk;
use image::{ImageReader, metadata::Cicp};
use parking_lot::MutexGuard;
use std::time::Instant;
use std::{
    collections::BTreeSet,
    collections::{BTreeMap, HashMap},
    fmt::{Display, Formatter},
    io::Read,
    path::Path,
    sync::Arc,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
    time::Duration,
};

// Used by util::collision::tests::pixel_perfect_complex_shape
#[derive(Clone)]
pub(crate) struct RawTexture {
    pub(crate) buf: Vec<u8>,
    pub(crate) extent: vk::Extent2D,
    format: vk::Format,
    duration: Option<Duration>,
}

#[derive(Debug)]
pub struct Texture {
    id: TextureId,
    duration: Option<Duration>,
    extent: Vec2,
    ref_count: Arc<AtomicUsize>,
    ready: Arc<AtomicBool>,
}

impl Texture {
    pub fn id(&self) -> TextureId {
        self.id
    }
    pub fn duration(&self) -> Option<Duration> {
        self.duration
    }
    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::SeqCst)
    }
}

impl AxisAlignedExtent for Texture {
    fn extent(&self) -> Vec2 {
        self.extent
    }

    fn centre(&self) -> Vec2 {
        self.extent / 2
    }
}

impl Clone for Texture {
    fn clone(&self) -> Self {
        self.ref_count.fetch_add(1, Ordering::Relaxed);
        Self {
            id: self.id,
            duration: self.duration,
            extent: self.extent,
            ref_count: self.ref_count.clone(),
            ready: self.ready.clone(),
        }
    }
}

impl Display for Texture {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Texture({:?}, {}x{})",
            self.id, self.extent.x, self.extent.y
        )
    }
}

impl Drop for Texture {
    fn drop(&mut self) {
        self.ref_count.fetch_sub(1, Ordering::Relaxed);
    }
}

struct ZombieTexture {
    _texture: Texture,
    waited_frames_in_flight: BTreeMap<usize, usize>,
}

impl ZombieTexture {
    pub fn waited_all(&self) -> bool {
        // 2 frames because the first frame is added right away.
        self.waited_frames_in_flight.values().all(|&x| x >= 2)
    }
}

struct TextureHandlerInner {
    loaded_files: BTreeMap<String, Vec<TextureId>>,
    loaded_files_inverse: BTreeMap<TextureId, String>,
    textures: BTreeMap<TextureId, Texture>,
    zombie_textures: BTreeMap<TextureId, ZombieTexture>,
    material_handler: MaterialHandler,
    texture_manager: TextureManager,
    frames_in_flight: Option<usize>,

    cleaned_up_since_last_report: usize,
    last_reported_cleanup: Option<Instant>,
}

impl TextureHandlerInner {
    fn new(ctx: Arc<TvWindowContext>) -> Result<Self> {
        Ok(Self {
            loaded_files: BTreeMap::new(),
            loaded_files_inverse: BTreeMap::new(),
            textures: BTreeMap::new(),
            zombie_textures: BTreeMap::new(),
            material_handler: MaterialHandler::new(),
            texture_manager: TextureManager::new(ctx)?,
            frames_in_flight: None,
            cleaned_up_since_last_report: 0,
            last_reported_cleanup: None,
        })
    }

    fn update_with_acquire_info(&mut self, acquire: &SwapchainAcquireInfo) -> Result<()> {
        self.frames_in_flight = Some(acquire.frames_in_flight());
        let ids_to_remove = self
            .zombie_textures
            .iter()
            .filter(|(_, t)| t.waited_all())
            .map(|(&id, _)| id)
            .collect_vec();
        for id in ids_to_remove {
            self.texture_manager.free_internal_texture(id);
            self.zombie_textures.remove(&id);
        }
        let mut missing_ids = Vec::new();
        for (id, zombie) in &mut self.zombie_textures {
            match zombie
                .waited_frames_in_flight
                .get_mut(&acquire.acquired_frame_index())
            {
                Some(r) => *r += 1,
                None => missing_ids.push(id),
            }
        }
        if missing_ids.is_empty() {
            Ok(())
        } else {
            bail!(
                "missing frame index for id(s) {missing_ids:?}: frame {}",
                acquire.acquired_frame_index()
            );
        }
    }

    fn clean_up_textures(&mut self) {
        let Some(frames_in_flight) = self.frames_in_flight else {
            return;
        };
        let to_remove = self
            .textures
            .iter()
            .filter(|(_, texture)| texture.ref_count.load(Ordering::Relaxed) == 1)
            .filter(|(id, _)| {
                // Don't free textures for loaded files.
                // TODO: some sort of more controlled cleanup for loaded files.
                !self.loaded_files_inverse.contains_key(id)
            })
            .map(|(&id, _)| id)
            .collect::<BTreeSet<_>>();
        for &id in &to_remove {
            self.zombie_textures.insert(
                id,
                ZombieTexture {
                    _texture: self
                        .textures
                        .remove(&id)
                        .with_context(|| format!("TextureHandlerInner::free_unused_textures(): missing texture: {id:?}"))
                        .unwrap(),
                    waited_frames_in_flight: (0..frames_in_flight).map(|i| (i, 0)).collect(),
                },
            );
        }
        self.cleaned_up_since_last_report += to_remove.len();
        if self.cleaned_up_since_last_report > 0
            && self
                .last_reported_cleanup
                .is_none_or(|i| i.elapsed() >= Duration::from_secs_f32(TEXTURE_STATS_INTERVAL_S))
        {
            info!(
                "cleaned up {} unused texture(s)",
                self.cleaned_up_since_last_report
            );
            self.cleaned_up_since_last_report = 0;
            self.last_reported_cleanup = Some(Instant::now());
        }
    }

    fn vk_free(&mut self) -> Result<()> {
        self.texture_manager.vk_free()
    }
}

// Note: material ID 0 represents a plain white texture.
pub type MaterialId = u32;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Material {
    pub texture_id: TextureId,
    pub area: Rect,
    pub texture_extent: Vec2,
}

pub(crate) struct MaterialHandler {
    materials: BTreeMap<MaterialId, Material>,
    materials_inverse: HashMap<Material, MaterialId>,
    has_pending_materials: bool,

    cleaned_up_since_last_report: usize,
    last_reported_cleanup: Option<Instant>,
}

impl MaterialHandler {
    fn new() -> Self {
        let blank_material = Material {
            texture_id: TextureId::default(),
            area: Rect::from_coords(Vec2::zero(), Vec2::one()),
            texture_extent: Vec2::one(),
        };
        let mut materials = BTreeMap::new();
        let mut materials_inverse = HashMap::new();
        materials.insert(0, blank_material.clone());
        materials_inverse.insert(blank_material, 0);
        Self {
            materials,
            materials_inverse,
            has_pending_materials: false,
            cleaned_up_since_last_report: 0,
            last_reported_cleanup: None,
        }
    }
    fn create_material_from_texture(&mut self, texture: &Texture, area: &Rect) -> MaterialId {
        let material = Material {
            texture_id: texture.id,
            area: *area,
            texture_extent: texture.extent,
        };
        if let Some(id) = self.materials_inverse.get(&material) {
            *id
        } else {
            let mut id = self
                .materials
                .last_key_value()
                .expect("materials empty? should have blank material")
                .0
                + 1;
            if id as usize == MAX_MATERIAL_COUNT {
                id = self
                    .materials
                    .keys()
                    .copied()
                    .tuple_windows()
                    .find(|(a, b)| *a + 1 != *b)
                    .map(|(a, _)| a + 1)
                    .or_else(|| self.materials.last_key_value().map(|(id, _)| id + 1))
                    .unwrap_or_default();
            }
            self.materials.insert(id, material.clone());
            self.materials_inverse.insert(material, id);
            self.has_pending_materials = true;
            id
        }
    }

    // Required to avoid concurrent borrows.
    fn get_unused_material_ids(&self, texture_manager: &TextureManager) -> Vec<MaterialId> {
        self.materials
            .iter()
            .filter(|(_, material)| texture_manager.is_texture_id_unused(material.texture_id))
            .map(|(id, _)| id)
            .copied()
            .collect()
    }
    fn clean_up_materials(&mut self, unused_ids: Vec<MaterialId>) -> Result<()> {
        self.cleaned_up_since_last_report += unused_ids.len();
        if self.cleaned_up_since_last_report > 0
            && self
                .last_reported_cleanup
                .is_none_or(|i| i.elapsed() >= Duration::from_secs_f32(TEXTURE_STATS_INTERVAL_S))
        {
            info!(
                "cleaned up {} unused material(s)",
                self.cleaned_up_since_last_report
            );
            self.cleaned_up_since_last_report = 0;
            self.last_reported_cleanup = Some(Instant::now());
        }
        for id in unused_ids {
            self.materials_inverse
                .remove(self.materials.get(&id).with_context(|| {
                    format!("MaterialHandler::clean_up_materials(): missing material: {id:?}")
                })?);
            self.materials.remove(&id);
        }
        Ok(())
    }

    pub fn materials(&self) -> &BTreeMap<MaterialId, Material> {
        &self.materials
    }
}

pub struct TextureHandler {
    // TODO: better lock design (avoid constant locking and unlocking).
    inner: GgMutex<TextureHandlerInner>,
    pipeline_layout: vk::PipelineLayout,
}

impl TextureHandler {
    pub(crate) fn new(ctx: Arc<TvWindowContext>) -> Result<Arc<Self>> {
        let inner = TextureHandlerInner::new(ctx)?;
        let pipeline_layout = inner.texture_manager.pipeline_layout();
        Ok(Arc::new(Self {
            inner: GgMutex::new(inner),
            pipeline_layout,
        }))
    }

    pub(crate) fn create_internal_texture(
        &self,
        extent: vk::Extent2D,
        format: vk::Format,
        image_data: &[u8],
        do_bind: bool,
        ready_signal: Arc<AtomicBool>,
    ) -> Result<Option<Arc<TvInternalTexture>>> {
        if do_bind {
            self.lock_inner("create_internal_texture")?
                .texture_manager
                .create_texture(extent, format, image_data, ready_signal)
        } else {
            self.lock_inner("create_internal_texture")?
                .texture_manager
                .create_texture_unbound(extent, format, image_data, ready_signal)
        }
    }
    pub(crate) fn free_internal_texture(&self, texture: &Arc<TvInternalTexture>) -> Result<()> {
        self.lock_inner("free_internal_texture")?
            .texture_manager
            .free_internal_texture(texture.id());
        Ok(())
    }

    fn lock_inner(&self, by: &'static str) -> Result<MutexGuard<'_, TextureHandlerInner>> {
        self.inner.try_lock_short(by)
    }

    pub fn get_blank_texture(&self) -> Result<Arc<TvInternalTexture>> {
        Ok(self
            .lock_inner("get_blank_texture")?
            .texture_manager
            .get_blank_texture()
            .clone())
    }

    pub fn materials_buffer_address(&self, frame_index: usize) -> Result<vk::DeviceAddress> {
        Ok(self
            .lock_inner("TextureHandler::materials_buffer_address()")?
            .texture_manager
            .materials_buffer_address(frame_index))
    }

    fn load_file_inner(filename: &str) -> Result<RawTexture> {
        let path = Path::new(filename);
        let ext = path
            .extension()
            .with_context(|| format!("no file extension: {filename}"))?
            .to_str()
            .with_context(|| format!("failed conversion from OsStr: {filename}"))?;
        match ext {
            "png" => Self::load_file_inner_png(filename),
            "aseprite" => Ok(Self::load_file_inner_animated_aseprite(filename)?
                .into_iter()
                .next()
                .context("loading aseprite file succeeded but no frames?")?),
            ext => bail!("unknown file extension: {ext} (while loading {filename})"),
        }
    }

    // Used by util::collision::tests::pixel_perfect_complex_shape
    pub(crate) fn load_file_inner_png(filename: &str) -> Result<RawTexture> {
        let mut image = ImageReader::open(filename)?.decode()?.into_rgba8();
        image.set_color_space(Cicp::SRGB)?;
        Self::load_reader_rgba_inner(
            &mut image.to_vec().as_slice(),
            image.width(),
            image.height(),
            vk::Format::R8G8B8A8_SRGB,
        )
    }

    fn load_reader_rgba_inner<R: Read>(
        reader: &mut R,
        width: u32,
        height: u32,
        format: vk::Format,
    ) -> Result<RawTexture> {
        if format != vk::Format::R8G8B8A8_SRGB {
            check_eq!(format, vk::Format::R8G8B8A8_UNORM);
        }
        let mut buf = vec![0; width as usize * height as usize * 4];
        reader.read_exact(&mut buf)?;
        Ok(RawTexture {
            buf,
            extent: vk::Extent2D { width, height },
            format,
            duration: None,
        })
    }

    pub fn wait_load_file(&self, filename: impl AsRef<str>) -> Result<Texture> {
        let filename = filename.as_ref().to_string();
        let mut inner = self.lock_inner("wait_load_file")?;
        if let Some(&texture_id) = inner.loaded_files.get(&filename).and_then(|v| {
            check_eq!(
                v.len(),
                1,
                "tried to load animated texture as a single frame"
            );
            v.first()
        }) {
            return Ok(inner
                .textures
                .get(&texture_id)
                .with_context(|| format!("TextureHandler::wait_load_file(\"{filename}\"): missing texture: {texture_id:?}"))?
                .clone());
        }
        let loaded = Self::load_file_inner(&filename).with_context(|| {
            format!("TextureHandler::wait_load_file(): loading file: {filename}")
        })?;
        let ready_flag = Arc::new(AtomicBool::new(false));
        let texture = inner
            .texture_manager
            .create_texture(
                loaded.extent,
                loaded.format,
                &loaded.buf,
                ready_flag.clone(),
            )
            .with_context(|| {
                format!("TextureHandler::wait_load_file(): creating texture: {filename}")
            })?
            .unwrap();
        let texture = Texture {
            id: texture.id(),
            duration: loaded.duration,
            extent: Vec2 {
                x: loaded.extent.width as f32,
                y: loaded.extent.height as f32,
            },
            ref_count: Arc::new(AtomicUsize::new(1)),
            ready: ready_flag,
        };
        info!("loaded texture: {filename} = {:?}", texture.id());
        inner
            .loaded_files
            .insert(filename.clone(), vec![texture.id()]);
        inner.loaded_files_inverse.insert(texture.id(), filename);
        inner.textures.insert(texture.id(), texture.clone());
        Ok(texture)
    }
    pub fn wait_load_file_animated(&self, filename: impl AsRef<str>) -> Result<Vec<Texture>> {
        let filename = filename.as_ref().to_string();
        let mut inner = self.lock_inner("wait_load_file_animated")?;
        if let Some(textures) = inner.loaded_files.get(&filename) {
            return textures
                .iter()
                .map(|id| {
                    inner
                        .textures
                        .get(id)
                        .with_context(|| {
                            format!(
                                "TextureHandler::wait_load_file_animated(): missing texture: {id:?}"
                            )
                        })
                        .cloned()
                })
                .collect();
        }
        let results = Self::load_file_inner_animated(&filename)?;
        let textures = results
            .into_iter()
            .map(|loaded| {
                let ready_flag = Arc::new(AtomicBool::new(false));
                let texture = inner
                    .texture_manager
                    .create_texture(
                        loaded.extent,
                        loaded.format,
                        &loaded.buf,
                        ready_flag.clone(),
                    )?
                    .unwrap();
                Ok(Texture {
                    id: texture.id(),
                    duration: loaded.duration,
                    extent: Vec2 {
                        x: loaded.extent.width as f32,
                        y: loaded.extent.height as f32,
                    },
                    ref_count: Arc::new(AtomicUsize::new(1)),
                    ready: ready_flag,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let texture_ids = textures.iter().map(Texture::id).collect();
        info!("loaded texture: {filename} = {texture_ids:?}");
        for texture in &textures {
            inner.textures.insert(texture.id(), texture.clone());
            inner
                .loaded_files_inverse
                .insert(texture.id(), filename.clone());
        }
        inner.loaded_files.insert(filename, texture_ids);
        Ok(textures)
    }
    fn load_file_inner_animated(filename: &str) -> Result<Vec<RawTexture>> {
        let path = Path::new(filename);
        let ext = path
            .extension()
            .with_context(|| format!("no file extension: {filename}"))?
            .to_str()
            .with_context(|| format!("failed conversion from OsStr: {filename}"))?;
        match ext {
            "png" => todo!("png files with multiple frames not supported"),
            "aseprite" => Self::load_file_inner_animated_aseprite(filename),
            ext => bail!("unknown file extension: {ext} (while loading {filename})"),
        }
    }
    fn load_file_inner_animated_aseprite(filename: &str) -> Result<Vec<RawTexture>> {
        let ase = AsepriteFile::read_file(filename.as_ref())?;
        (0..ase.num_frames())
            .map(|i| ase.frame(i))
            .map(|frame| {
                let image = frame.image();
                let mut loaded = Self::load_reader_rgba_inner(
                    &mut image.to_vec().as_slice(),
                    image.width(),
                    image.height(),
                    vk::Format::R8G8B8A8_SRGB,
                )?;
                loaded.duration = Some(Duration::from_millis(u64::from(frame.duration())));
                Ok(loaded)
            })
            .collect::<Result<Vec<_>>>()
    }

    pub(crate) fn wait_load_reader_rgba<R: Read>(
        &self,
        reader: &mut R,
        width: u32,
        height: u32,
        format: vk::Format,
    ) -> Result<Texture> {
        let loaded = Self::load_reader_rgba_inner(reader, width, height, format)?;
        let ready_flag = Arc::new(AtomicBool::new(false));
        let mut inner = self.lock_inner("wait_load_reader_rgba")?;
        let texture = inner
            .texture_manager
            .create_texture(
                loaded.extent,
                loaded.format,
                &loaded.buf,
                ready_flag.clone(),
            )?
            .unwrap();
        let texture = Texture {
            id: texture.id(),
            duration: loaded.duration,
            extent: Vec2 {
                x: loaded.extent.width as f32,
                y: loaded.extent.height as f32,
            },
            ref_count: Arc::new(AtomicUsize::new(1)),
            ready: ready_flag,
        };
        inner.textures.insert(texture.id(), texture.clone());
        Ok(texture)
    }
    pub fn create_material_from_texture(
        &self,
        texture: &Texture,
        area: &Rect,
    ) -> Result<MaterialId> {
        Ok(self
            .lock_inner("create_material_from_texture")?
            .material_handler
            .create_material_from_texture(texture, area))
    }

    pub fn wait_get_raw(&self, texture_id: TextureId) -> Result<Option<Vec<Vec<Colour>>>> {
        let Some(tex) = self
            .lock_inner("wait_get_raw")?
            .texture_manager
            .get_internal_texture(texture_id)
        else {
            return Ok(None);
        };
        let w = tex.extent().width as usize;
        let h = tex.extent().height as usize;
        let mut rv = vec![vec![Colour::empty(); w]; h];
        let mut x = 0;
        let mut y = 0;
        for bytes in tex.data().chunks(4) {
            let col = Colour::from_bytes(bytes[0], bytes[1], bytes[2], bytes[3]);
            rv[y][x] = col;
            x += 1;
            if x == w {
                x = 0;
                y += 1;
            }
        }
        Ok(Some(rv))
    }

    pub(crate) fn upload_all_pending_out_of_band(&self, by: &'static str) -> Result<()> {
        let mut inner = self.lock_inner("upload_all_pending_out_of_band")?;
        if inner.material_handler.has_pending_materials {
            let materials = inner.material_handler.materials().clone();
            inner.texture_manager.stage_materials(materials, None)?;
            inner.material_handler.has_pending_materials = false;
        }
        inner.texture_manager.wait_complete_upload()?;
        inner.texture_manager.upload_all_pending_out_of_band(by)?;
        Ok(())
    }
    pub fn upload_all_pending_with(
        &self,
        command_buffer: vk::CommandBuffer,
        swapchain_acquire_info: &SwapchainAcquireInfo,
    ) -> Result<()> {
        let mut inner = self.lock_inner("upload_all_pending_with")?;
        if inner.material_handler.has_pending_materials {
            let materials = inner.material_handler.materials().clone();
            inner
                .texture_manager
                .stage_materials(materials, Some(swapchain_acquire_info))?;
            inner.material_handler.has_pending_materials = false;
        }
        inner.texture_manager.wait_complete_upload()?;
        inner
            .texture_manager
            .upload_all_pending_with(command_buffer, swapchain_acquire_info);
        Ok(())
    }

    pub(crate) fn on_render_done(&self, acquire: &SwapchainAcquireInfo) -> Result<()> {
        let mut inner = self.lock_inner("on_render_done")?;
        inner.update_with_acquire_info(acquire)?;
        inner.clean_up_textures();
        let unused_material_ids = inner
            .material_handler
            .get_unused_material_ids(&inner.texture_manager);
        inner
            .material_handler
            .clean_up_materials(unused_material_ids)?;
        Ok(())
    }

    pub fn bind(&self, command_buffer: vk::CommandBuffer) -> Result<()> {
        self.lock_inner("bind")?
            .texture_manager
            .bind(command_buffer);
        Ok(())
    }

    pub fn pipeline_layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }

    pub fn material_to_texture(&self, id: MaterialId) -> Result<Option<TextureId>> {
        Ok(self
            .lock_inner("material_to_texture")?
            .material_handler
            .materials
            .get(&id)
            .map(|material| material.texture_id))
    }
    pub fn get_ready_materials(&self) -> Result<BTreeSet<MaterialId>> {
        let inner = self.lock_inner("TextureHandler::get_ready_materials()")?;
        let mut rv = BTreeSet::new();
        for (&id, material) in &inner.material_handler.materials {
            if inner.texture_manager.is_texture_ready(material.texture_id) {
                rv.insert(id);
            }
        }
        Ok(rv)
    }

    pub fn vk_free(&self) -> Result<()> {
        self.inner
            .try_lock("TextureHandler::vk_free()")?
            .expect("contention should be impossible (no other references should exist)")
            .vk_free()
    }
}
