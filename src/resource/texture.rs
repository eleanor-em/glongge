use asefile::AsepriteFile;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};
use std::{
    collections::{BTreeMap, HashMap},
    default::Default,
    fmt::{Display, Formatter},
    fs,
    io::{Cursor, Read},
    path::Path,
    sync::{Arc, Mutex, atomic::Ordering},
};
use vulkano::{
    NonZeroDeviceSize, Validated,
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    format::Format,
    image::{ImageCreateInfo, ImageType, ImageUsage, view::ImageView},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
};

use crate::core::prelude::*;
use crate::core::vk::vk_ctx::VulkanoContext;
use crate::util::UniqueShared;
use png::ColorType;
use vulkano::image::Image;
use vulkano::memory::DeviceAlignment;
use vulkano::memory::allocator::{DeviceLayout, MemoryAllocatePreference};
use vulkano_taskgraph::command_buffer::{CopyBufferToImageInfo, RecordingCommandBuffer};
use vulkano_taskgraph::graph::{NodeId, TaskGraph};
use vulkano_taskgraph::resource::{AccessTypes, HostAccessType, ImageLayoutType};
use vulkano_taskgraph::{Id, QueueFamilyType, Task, TaskContext, TaskResult};

#[derive(Clone)]
struct RawTexture {
    buf: Vec<u8>,
    info: ImageCreateInfo,
    duration: Option<Duration>,
}

// Note: texture ID 0 represents a plain white texture.
pub type TextureId = u32;

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
        self.ready.load(Ordering::Relaxed)
    }
}

impl AxisAlignedExtent for Texture {
    fn aa_extent(&self) -> Vec2 {
        self.extent
    }

    fn centre(&self) -> Vec2 {
        self.extent / 2
    }
}

impl Clone for Texture {
    fn clone(&self) -> Self {
        *self.ref_count.get() += 1;
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
            "Texture(id={}, {}x{})",
            self.id, self.extent.x, self.extent.y
        )
    }
}

impl Drop for Texture {
    fn drop(&mut self) {
        *self.ref_count.get() -= 1;
    }
}

#[derive(Clone)]
pub(crate) struct InternalTexture {
    filename: String,
    id: TextureId,
    raw: Arc<RawTexture>,
    buf: Id<Buffer>,
    image: Id<Image>,
    uploaded_image_view: Option<Arc<ImageView>>,
    ref_count: UniqueShared<usize>,
    has_write_access: bool,
    ready: Arc<AtomicBool>,
}

impl InternalTexture {
    pub fn image_view(&self) -> Option<Arc<ImageView>> {
        self.uploaded_image_view.clone()
    }
    pub fn id(&self) -> TextureId {
        self.id
    }
    fn is_ready(&self) -> bool {
        check_eq!(
            self.ready.load(Ordering::Relaxed),
            self.uploaded_image_view.is_some()
        );
        self.uploaded_image_view.is_some()
    }

    fn create_image_view(
        &mut self,
        ctx: &VulkanoContext,
        cbf: &mut RecordingCommandBuffer,
        _tcx: &mut TaskContext,
    ) -> TaskResult {
        if !self.is_ready() {
            unsafe {
                cbf.copy_buffer_to_image(&CopyBufferToImageInfo {
                    src_buffer: self.buf,
                    dst_image: self.image,
                    ..Default::default()
                })?;
            }
            let image_view =
                ImageView::new_default(ctx.resources().image(self.image)?.image().clone()).unwrap();
            self.uploaded_image_view = Some(image_view);
            self.ready.store(true, Ordering::Relaxed);
        }
        Ok(())
    }

    fn can_upload(&self) -> bool {
        self.has_write_access && !self.is_ready()
    }
}

impl AxisAlignedExtent for InternalTexture {
    fn aa_extent(&self) -> Vec2 {
        Vec2 {
            x: self.raw.info.extent[0] as f32,
            y: self.raw.info.extent[1] as f32,
        }
    }

    fn centre(&self) -> Vec2 {
        Vec2::zero()
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

struct TextureHandlerInner {
    loaded_files: BTreeMap<String, Vec<Texture>>,
    textures: BTreeMap<TextureId, InternalTexture>,
    material_handler: Arc<Mutex<MaterialHandler>>,
    last_freed_textures: Instant,
}

impl TextureHandlerInner {
    fn new(ctx: &VulkanoContext, material_handler: Arc<Mutex<MaterialHandler>>) -> Result<Self> {
        let mut textures = BTreeMap::new();

        // Create blank texture
        let info = ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::R8G8B8A8_SRGB,
            extent: [1, 1, 1],
            usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
            ..Default::default()
        };
        let buf = ctx
            .resources()
            .create_buffer(
                BufferCreateInfo {
                    usage: BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_HOST
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                DeviceLayout::for_value(&[255u8; 4])
                    .context("failed to create layout for blank texture")?,
            )
            .map_err(Validated::unwrap)?;

        let image = ctx
            .resources()
            .create_image(
                info.clone(),
                AllocationCreateInfo {
                    allocate_preference: MemoryAllocatePreference::AlwaysAllocate,
                    ..AllocationCreateInfo::default()
                },
            )
            .map_err(Validated::unwrap)?;
        let internal_texture = InternalTexture {
            filename: "[blank]".to_string(),
            id: 0,
            raw: Arc::new(RawTexture {
                buf: Colour::white().as_bytes().to_vec(),
                info,
                duration: None,
            }),
            buf,
            image,
            uploaded_image_view: None,
            ref_count: UniqueShared::new(1),
            has_write_access: false,
            ready: Arc::new(AtomicBool::new(false)),
        };
        textures.insert(0, internal_texture);

        Ok(Self {
            loaded_files: BTreeMap::new(),
            textures,
            material_handler,
            last_freed_textures: Instant::now(),
        })
    }

    fn create_texture(
        &mut self,
        ctx: &VulkanoContext,
        filename: String,
        loaded: RawTexture,
    ) -> Result<Texture> {
        let id = self
            .textures
            .keys()
            .copied()
            .tuple_windows()
            .find(|(a, b)| *a + 1 != *b)
            .map(|(a, _)| a + 1)
            .or_else(|| self.textures.last_key_value().map(|(id, _)| id + 1))
            .expect("empty textures? (blank texture missing)");
        let buf = ctx
            .resources()
            .create_buffer(
                BufferCreateInfo {
                    usage: BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_HOST
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                DeviceLayout::new(
                    NonZeroDeviceSize::new(loaded.buf.len() as u64).unwrap(),
                    DeviceAlignment::of::<u8>(),
                )
                .context("failed to create layout for texture")?,
            )
            .map_err(Validated::unwrap)?;
        let duration = loaded.duration;
        let image = ctx
            .resources()
            .create_image(
                loaded.info.clone(),
                AllocationCreateInfo {
                    allocate_preference: MemoryAllocatePreference::AlwaysAllocate,
                    ..AllocationCreateInfo::default()
                },
            )
            .map_err(Validated::unwrap)?;
        let ref_count = UniqueShared::new(1);
        let ready = Arc::new(AtomicBool::new(false));
        let internal_texture = InternalTexture {
            filename,
            id,
            raw: Arc::new(loaded),
            buf,
            image,
            uploaded_image_view: None,
            ref_count: ref_count.clone(),
            has_write_access: false,
            ready: ready.clone(),
        };
        let extent = internal_texture.aa_extent();
        if let Some(existing) = self.textures.insert(id, internal_texture) {
            panic!(
                "tried to use texture id {id}, but ref_count={}",
                existing.ref_count.get()
            );
        }
        info!("created texture id {id}");
        Ok(Texture {
            id,
            duration,
            extent,
            ref_count,
            ready,
        })
    }

    fn free_unused_textures(&mut self) {
        if self.last_freed_textures.elapsed().as_millis() < 1000
            && self.textures.len() < MAX_TEXTURE_COUNT
        {
            return;
        }
        self.last_freed_textures = Instant::now();
        let unused_ids = self
            .textures
            .iter()
            .filter(|(_, tex)| tex.is_ready() && *tex.ref_count.get() == 0)
            .map(|(id, _)| *id)
            .collect_vec();
        if unused_ids.is_empty() {
            return;
        }
        let mut material_handler = self.material_handler.lock().unwrap();
        info!("freeing texture ids: {unused_ids:?}");
        for unused_id in unused_ids {
            self.textures.remove(&unused_id);
            material_handler.on_remove_texture(unused_id);
        }
    }

    fn free_unused_files(&mut self) {
        for filename in self
            .loaded_files
            .iter()
            .filter(|(_, textures)| textures.iter().all(|tex| *tex.ref_count.get() <= 1))
            .map(|(filename, _)| filename.clone())
            .collect_vec()
        {
            info!("freeing texture {filename}");
            self.loaded_files.remove(&filename);
        }
        self.free_unused_textures();
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

struct MaterialHandler {
    materials: BTreeMap<MaterialId, Material>,
    materials_inverse: HashMap<Material, MaterialId>,
    dirty: bool,
}

impl MaterialHandler {
    fn new() -> Self {
        let blank_material = Material {
            texture_id: 0,
            area: Rect::empty(),
            texture_extent: Vec2::zero(),
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
    fn material_from_texture(&mut self, texture: &Texture, area: &Rect) -> MaterialId {
        let material = Material {
            texture_id: texture.id,
            area: *area,
            texture_extent: texture.extent,
        };
        if let Some(id) = self.materials_inverse.get(&material) {
            *id
        } else {
            let id = self
                .materials
                .keys()
                .copied()
                .tuple_windows()
                .find(|(a, b)| *a + 1 != *b)
                .map(|(a, _)| a + 1)
                .or_else(|| self.materials.last_key_value().map(|(id, _)| id + 1))
                .unwrap_or_default();
            self.materials.insert(id, material.clone());
            self.materials_inverse.insert(material, id);
            self.dirty = true;
            id
        }
    }
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
}

#[derive(Clone)]
pub struct TextureHandler {
    ctx: VulkanoContext,
    inner: Arc<Mutex<TextureHandlerInner>>,
    material_handler: Arc<Mutex<MaterialHandler>>,
}

impl TextureHandler {
    pub(crate) fn new(ctx: VulkanoContext) -> Result<Self> {
        let material_handler = Arc::new(Mutex::new(MaterialHandler::new()));
        let inner = Arc::new(Mutex::new(TextureHandlerInner::new(
            &ctx,
            material_handler.clone(),
        )?));
        Ok(Self {
            ctx,
            inner,
            material_handler,
        })
    }

    pub fn material_from_texture(&self, texture: &Texture, area: &Rect) -> MaterialId {
        self.material_handler
            .lock()
            .unwrap()
            .material_from_texture(texture, area)
    }

    // TODO: implement spawn_load_file().
    pub fn wait_load_file(&self, filename: impl AsRef<str>) -> Result<Texture> {
        let filename = filename.as_ref().to_string();
        // Beware: do not lock `inner` longer than necessary.
        if let Some(texture) = self
            .inner
            .lock()
            .unwrap()
            .loaded_files
            .get(&filename)
            .and_then(|v| v.first())
        {
            return Ok(texture.clone());
        }
        let loaded = Self::load_file_inner(&filename)?;
        let mut inner = self.inner.lock().unwrap();
        let texture = inner.create_texture(&self.ctx, filename.to_string(), loaded)?;
        info!("loaded texture: {} = {:?}", filename, texture.id());
        inner.loaded_files.insert(filename, vec![texture.clone()]);
        Ok(texture)
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
            Some(_) => Format::R8G8B8A8_SRGB,
            None => Format::R8G8B8A8_UNORM,
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

    pub fn wait_load_file_animated(&self, filename: impl AsRef<str>) -> Result<Vec<Texture>> {
        let filename = filename.as_ref().to_string();
        // Beware: do not lock `inner` longer than necessary.
        if let Some(texture) = self.inner.lock().unwrap().loaded_files.get(&filename) {
            return Ok(texture.clone());
        }
        let results = Self::load_file_inner_animated(&filename)?;
        let mut inner = self.inner.lock().unwrap();
        let textures = results
            .into_iter()
            .map(|loaded| inner.create_texture(&self.ctx, filename.to_string(), loaded))
            .collect::<Result<Vec<_>>>()?;
        inner.loaded_files.insert(filename, textures.clone());
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
                    Format::R8G8B8A8_SRGB,
                )?;
                loaded.duration = Some(Duration::from_millis(u64::from(frame.duration())));
                Ok(loaded)
            })
            .collect::<Result<Vec<_>>>()
    }

    pub(crate) fn wait_load_reader_rgba<R: Read>(
        &self,
        filename: String,
        reader: &mut R,
        width: u32,
        height: u32,
        format: Format,
    ) -> Result<Texture> {
        let loaded = Self::load_reader_rgba_inner(reader, width, height, format)?;
        let mut inner = self.inner.lock().unwrap();
        inner.create_texture(&self.ctx, filename, loaded)
    }
    fn load_reader_rgba_inner<R: Read>(
        reader: &mut R,
        width: u32,
        height: u32,
        format: Format,
    ) -> Result<RawTexture> {
        if format != Format::R8G8B8A8_SRGB {
            check_eq!(format, Format::R8G8B8A8_UNORM);
        }
        let image_create_info = ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format,
            extent: [width, height, 1],
            usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
            ..Default::default()
        };
        let mut buf = vec![0; width as usize * height as usize * 4];
        reader.read_exact(&mut buf)?;
        Ok(RawTexture {
            buf,
            info: image_create_info,
            duration: None,
        })
    }

    pub fn wait_free_unused_files(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.free_unused_files();
    }

    pub(crate) fn ready_values(&self) -> Vec<InternalTexture> {
        self.inner
            .lock()
            .unwrap()
            .textures
            .values()
            .filter(|t| t.is_ready())
            .cloned()
            .collect()
    }

    pub(crate) fn is_not_yet_initialised(&self) -> bool {
        self.ready_values().is_empty()
    }

    pub(crate) fn get_updated_materials(&self) -> Option<Vec<(MaterialId, Material)>> {
        let mut material_handler = self.material_handler.lock().unwrap();
        if material_handler.dirty {
            let rv = material_handler
                .materials
                .iter()
                .map(|(id, mat)| (*id, mat.clone()))
                .collect();
            material_handler.dirty = false;
            Some(rv)
        } else {
            None
        }
    }

    pub fn wait_get_raw(&self, texture_id: TextureId) -> Result<Option<Vec<Vec<Colour>>>> {
        let Some(tex) = self
            .inner
            .lock()
            .unwrap()
            .textures
            .get(&texture_id)
            .cloned()
        else {
            return Ok(None);
        };
        let w = tex.raw.info.extent[0] as usize;
        let h = tex.raw.info.extent[1] as usize;
        let mut rv = vec![vec![Colour::empty(); w]; h];
        let mut x = 0;
        let mut y = 0;
        for bytes in tex.raw.buf.chunks(4) {
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

    pub(crate) fn wait_textures_dirty(&self) -> bool {
        self.inner
            .lock()
            .unwrap()
            .textures
            .values()
            .any(|t| !t.has_write_access)
    }
    pub(crate) fn build_task_graph(
        &self,
        task_graph: &mut TaskGraph<VulkanoContext>,
    ) -> (NodeId, Vec<Id<Image>>) {
        let mut inner = self.inner.lock().unwrap();
        for tex in inner.textures.values() {
            task_graph.add_host_buffer_access(tex.buf, HostAccessType::Write);
        }
        let mut upload_node = task_graph.create_task_node(
            "texture_handler_upload",
            QueueFamilyType::Transfer,
            UploadTexturesTask {
                inner: Arc::new(self.clone()),
            },
        );
        let mut images = Vec::new();
        for tex in inner.textures.values_mut() {
            upload_node.buffer_access(tex.buf, AccessTypes::COPY_TRANSFER_READ);
            upload_node.image_access(
                tex.image,
                AccessTypes::COPY_TRANSFER_WRITE,
                ImageLayoutType::Optimal,
            );
            images.push(tex.image);
            tex.has_write_access = true;
        }
        let upload_node = upload_node.build();
        (upload_node, images)
    }
}

pub(crate) struct UploadTexturesTask {
    pub(crate) inner: Arc<TextureHandler>,
}

impl Task for UploadTexturesTask {
    type World = VulkanoContext;

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer,
        tcx: &mut TaskContext,
        world: &Self::World,
    ) -> TaskResult {
        let mut inner = self.inner.inner.lock().unwrap();
        for (id, tex) in inner
            .textures
            .iter_mut()
            .filter(|(_, tex)| tex.can_upload())
        {
            tcx.write_buffer::<[u8]>(tex.buf, ..)?
                .clone_from_slice(&tex.raw.buf);
            tex.create_image_view(world, cbf, tcx)?;
            info!(
                "created image view for: {} (id {id:?}, {:.1} KiB)",
                tex.filename,
                (tex.raw.buf.len() as f32) / 1024.0
            );
        }
        Ok(())
    }
}
