use crate::core::prelude::*;
use crate::core::tulivuori::TvWindowContext;
use crate::core::tulivuori::swapchain::SwapchainAcquireInfo;
use crate::core::tulivuori::texture::{TextureId, TextureManager};
use crate::util::UniqueShared;
use asefile::AsepriteFile;
use ash::vk;
use png::ColorType;
use std::collections::BTreeSet;
use std::sync::atomic::AtomicBool;
use std::time::Duration;
use std::{
    collections::{BTreeMap, HashMap},
    fmt::{Display, Formatter},
    fs,
    io::{Cursor, Read},
    path::Path,
    sync::{Arc, atomic::Ordering},
};

#[derive(Clone)]
struct RawTexture {
    buf: Vec<u8>,
    extent: vk::Extent2D,
    _format: vk::Format,
    duration: Option<Duration>,
}

#[derive(Debug)]
pub struct Texture {
    id: TextureId,
    duration: Option<Duration>,
    extent: Vec2,
    ref_count: UniqueShared<usize>,
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
        *self.ref_count.lock() += 1;
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
        *self.ref_count.lock() -= 1;
    }
}

struct WrappedPngReader<R: Read>(png::Reader<R>);

impl<R: Read> Read for WrappedPngReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.0
            .next_frame(buf)
            .map_err(|_| std::io::Error::from(std::io::ErrorKind::InvalidInput))?;
        Ok(self.0.output_buffer_size())
    }
}

struct ZombieTexture {
    _texture: Texture,
    waited_frames_in_flight: BTreeMap<usize, bool>,
}

impl ZombieTexture {
    pub fn waited_all(&self) -> bool {
        self.waited_frames_in_flight.values().all(|&x| x)
    }
}

struct TextureHandlerInner {
    loaded_files: BTreeMap<String, Vec<TextureId>>,
    textures: BTreeMap<TextureId, Texture>,
    zombie_textures: BTreeMap<TextureId, ZombieTexture>,
    material_handler: UniqueShared<MaterialHandler>,
    texture_manager: UniqueShared<TextureManager>,
    frames_in_flight: Option<usize>,
}

impl TextureHandlerInner {
    fn new(
        ctx: Arc<TvWindowContext>,
        material_handler: UniqueShared<MaterialHandler>,
    ) -> Result<Self> {
        Ok(Self {
            loaded_files: BTreeMap::new(),
            textures: BTreeMap::new(),
            zombie_textures: BTreeMap::new(),
            material_handler,
            texture_manager: UniqueShared::new(TextureManager::new(ctx)?),
            frames_in_flight: None,
        })
    }

    fn update_with_acquire_info(&mut self, acquire: &SwapchainAcquireInfo) {
        self.frames_in_flight = Some(acquire.frames_in_flight());
        self.zombie_textures.retain(|_, t| !t.waited_all());
        for zombie in self.zombie_textures.values_mut() {
            zombie
                .waited_frames_in_flight
                .insert(acquire.acquired_frame_index(), true);
        }
    }

    fn maybe_upload_materials(&self) -> Result<()> {
        if self.material_handler.lock().dirty {
            self.texture_manager
                .lock()
                .upload_materials(&self.material_handler)?;
            self.material_handler.lock().dirty = false;
        }
        Ok(())
    }

    fn free_unused_textures(&mut self) {
        let Some(frames_in_flight) = self.frames_in_flight else {
            return;
        };
        let to_remove = self
            .textures
            .iter()
            .filter(|(_, texture)| *texture.ref_count.lock() == 1)
            .map(|(id, _)| id)
            .copied()
            .collect::<BTreeSet<_>>();
        for &id in &to_remove {
            self.texture_manager.lock().free_internal_texture(id);

            self.zombie_textures.insert(
                id,
                ZombieTexture {
                    _texture: self
                        .textures
                        .remove(&id)
                        .context("missing texture id {id:?}")
                        .unwrap(),
                    waited_frames_in_flight: (0..frames_in_flight).map(|i| (i, false)).collect(),
                },
            );
        }
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
    dirty: bool,
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
            dirty: false,
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
            self.dirty = true;
            id
        }
    }
    #[allow(unused)]
    fn on_remove_texture(&mut self, texture_id: TextureId) {
        self.materials_inverse.retain(|material, material_id| {
            if material.texture_id == texture_id {
                self.materials.remove(material_id);
                self.dirty = true;
                false
            } else {
                true
            }
        });
    }

    pub fn materials(&self) -> &BTreeMap<MaterialId, Material> {
        &self.materials
    }
}

#[derive(Clone)]
pub struct TextureHandler {
    inner: UniqueShared<TextureHandlerInner>,
}

impl TextureHandler {
    pub(crate) fn new(ctx: Arc<TvWindowContext>) -> Result<Self> {
        let material_handler = UniqueShared::new(MaterialHandler::new());
        let inner = UniqueShared::new(TextureHandlerInner::new(ctx, material_handler.clone())?);
        let rv = Self { inner };
        rv.inner
            .lock()
            .texture_manager
            .lock()
            .initialise_materials(&material_handler)?;
        Ok(rv)
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
    fn load_file_inner_png(filename: &str) -> Result<RawTexture> {
        let png_bytes = fs::read(filename)?;
        let cursor = Cursor::new(png_bytes);
        let decoder = png::Decoder::new(cursor);
        let reader = decoder.read_info()?;
        let info = reader.info();

        if info.srgb.is_none() {
            error!(
                "loading {}: SRGB not enabled, may display incorrectly",
                filename
            );
        }
        if info.color_type != ColorType::Rgba {
            error!(
                "loading {}: no alpha channel, *will* display incorrectly (re-encode as RGBA)",
                filename
            );
        }
        if let Some(animation_control) = info.animation_control
            && animation_control.num_frames != 1
        {
            error!(
                "loading {}: unexpected num_frames {} (animated PNG not supported)",
                filename, animation_control.num_frames
            );
        }
        let format = match info.srgb {
            Some(_) => vk::Format::R8G8B8A8_SRGB,
            None => vk::Format::R8G8B8A8_UNORM,
        };

        let mut image_data = Vec::new();
        let depth: u32 = match info.bit_depth {
            png::BitDepth::One => 1,
            png::BitDepth::Two => 2,
            png::BitDepth::Four => 4,
            png::BitDepth::Eight => 8,
            png::BitDepth::Sixteen => 16,
        };
        let width = info.width;
        let height = info.height;
        image_data.resize((width * height * depth) as usize, 0);
        Self::load_reader_rgba_inner(&mut WrappedPngReader(reader), width, height, format)
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
            _format: format,
            duration: None,
        })
    }
    pub fn wait_load_file(&self, filename: impl AsRef<str>) -> Result<Texture> {
        let filename = filename.as_ref().to_string();
        let mut inner = self.inner.lock();
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
                .context("missing texture {texture_id:?}")?
                .clone());
        }
        let loaded = Self::load_file_inner(&filename).with_context(|| {
            format!("TextureHandler::wait_load_file(): loading file: {filename}")
        })?;
        let ready_flag = Arc::new(AtomicBool::new(false));
        let texture = inner
            .texture_manager
            .lock()
            .create_texture(loaded.extent, &loaded.buf, ready_flag.clone())
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
            ref_count: UniqueShared::new(1),
            ready: ready_flag,
        };
        info!("loaded texture: {filename} = {:?}", texture.id());
        inner.loaded_files.insert(filename, vec![texture.id()]);
        inner.textures.insert(texture.id(), texture.clone());
        Ok(texture)
    }
    pub fn wait_load_file_animated(&self, filename: impl AsRef<str>) -> Result<Vec<Texture>> {
        let filename = filename.as_ref().to_string();
        let mut inner = self.inner.lock();
        if let Some(textures) = inner.loaded_files.get(&filename) {
            return textures
                .iter()
                .map(|id| {
                    inner
                        .textures
                        .get(id)
                        .context("missing texture id {id:?}")
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
                    .lock()
                    .create_texture(loaded.extent, &loaded.buf, ready_flag.clone())?
                    .unwrap();
                Ok(Texture {
                    id: texture.id(),
                    duration: loaded.duration,
                    extent: Vec2 {
                        x: loaded.extent.width as f32,
                        y: loaded.extent.height as f32,
                    },
                    ref_count: UniqueShared::new(1),
                    ready: ready_flag,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let texture_ids = textures.iter().map(Texture::id).collect();
        info!("loaded texture: {filename} = {texture_ids:?}");
        inner.loaded_files.insert(filename, texture_ids);
        for texture in &textures {
            inner.textures.insert(texture.id(), texture.clone());
        }
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
        let texture = self
            .inner
            .lock()
            .texture_manager
            .lock()
            .create_texture(loaded.extent, &loaded.buf, ready_flag.clone())?
            .unwrap();
        let texture = Texture {
            id: texture.id(),
            duration: loaded.duration,
            extent: Vec2 {
                x: loaded.extent.width as f32,
                y: loaded.extent.height as f32,
            },
            ref_count: UniqueShared::new(1),
            ready: ready_flag,
        };
        self.inner
            .lock()
            .textures
            .insert(texture.id(), texture.clone());
        Ok(texture)
    }
    pub fn create_material_from_texture(&self, texture: &Texture, area: &Rect) -> MaterialId {
        self.inner
            .lock()
            .material_handler
            .lock()
            .create_material_from_texture(texture, area)
    }

    pub fn wait_get_raw(&self, texture_id: TextureId) -> Result<Option<Vec<Vec<Colour>>>> {
        let Some(tex) = self
            .inner
            .lock()
            .texture_manager
            .lock()
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

    pub fn wait_for_upload(&self) -> Result<bool> {
        self.inner.lock().texture_manager.lock().wait_for_upload()
    }
    pub fn maybe_upload_pending(&self) -> Result<()> {
        self.inner.lock().texture_manager.lock().upload_pending()?;
        self.inner.lock().maybe_upload_materials()
    }

    pub(crate) fn on_render_done(&self, acquire: &SwapchainAcquireInfo) -> Result<()> {
        self.maybe_upload_pending()?;
        self.inner.lock().update_with_acquire_info(acquire);
        self.free_unused_textures();
        Ok(())
    }
    pub fn free_unused_textures(&self) {
        self.inner.lock().free_unused_textures();
    }

    pub fn bind(&self, command_buffer: vk::CommandBuffer) {
        self.inner
            .lock()
            .texture_manager
            .lock()
            .bind(command_buffer);
    }

    pub fn pipeline_layout(&self) -> vk::PipelineLayout {
        self.inner.lock().texture_manager.lock().pipeline_layout()
    }

    pub fn is_texture_ready(&self, texture: TextureId) -> bool {
        self.inner
            .lock()
            .texture_manager
            .lock()
            .is_texture_ready(texture)
    }
    pub fn material_to_texture(&self, material: MaterialId) -> Option<TextureId> {
        self.inner
            .lock()
            .material_handler
            .lock()
            .materials
            .get(&material)
            .map(|material| material.texture_id)
    }
    pub fn is_material_ready(&self, material: MaterialId) -> bool {
        let Some(texture_id) = self.material_to_texture(material) else {
            return false;
        };
        self.is_texture_ready(texture_id)
    }
}
