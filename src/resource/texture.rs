use std::{
    collections::{BTreeMap, HashMap},
    default::Default,
    fmt::{Display, Formatter},
    fs,
    io::{
        Cursor,
        Read
    },
    path::Path,
    sync::{
        Arc,
        Mutex,
        atomic::{AtomicUsize, Ordering}
    },
};
use std::time::Duration;
use asefile::AsepriteFile;
use num_traits::Zero;

use vulkano::{buffer::{Buffer, BufferCreateInfo, BufferUsage}, format::Format, image::{
    ImageCreateInfo,
    ImageType,
    ImageUsage,
    view::ImageView
}, memory::allocator::{AllocationCreateInfo, MemoryTypeFilter}, NonZeroDeviceSize, Validated};

use png::ColorType;
use vulkano::image::Image;
use vulkano::memory::allocator::{DeviceLayout, MemoryAllocatePreference};
use vulkano::memory::DeviceAlignment;
use vulkano_taskgraph::{Id, QueueFamilyType, Task, TaskContext, TaskResult};
use vulkano_taskgraph::command_buffer::{CopyBufferToImageInfo, RecordingCommandBuffer};
use vulkano_taskgraph::graph::{NodeId, TaskGraph};
use vulkano_taskgraph::resource::{AccessType, HostAccessType, ImageLayoutType};
use crate::core::prelude::*;
use crate::core::vk::vk_ctx::VulkanoContext;

#[derive(Clone)]
struct RawLoadedTexture {
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
    ref_count: Arc<AtomicUsize>,
}

impl Texture {
    pub fn id(&self) -> TextureId { self.id }
    pub fn duration(&self) -> Option<Duration> { self.duration }
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
        self.ref_count.fetch_add(1, Ordering::Relaxed);
        Self {
            id: self.id,
            duration: self.duration,
            extent: self.extent,
            ref_count: self.ref_count.clone(),
        }
    }
}

impl Default for Texture {
    fn default() -> Self {
        Self {
            id: 0,
            duration: None,
            extent: Vec2::one(),
            ref_count: Arc::new(AtomicUsize::new(0)),
        }
    }
}

impl Display for Texture {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Texture(id={}, {}x{})", self.id, self.extent.x, self.extent.y)
    }
}

impl Drop for Texture {
    fn drop(&mut self) {
        self.ref_count.fetch_sub(1, Ordering::Relaxed);
    }
}

#[derive(Clone)]
pub(crate) struct InternalTexture {
    raw: Arc<RawLoadedTexture>,
    buf: Id<Buffer>,
    image: Id<Image>,
    uploaded_image_view: Option<Arc<ImageView>>,
    ref_count: Arc<AtomicUsize>,
}

impl InternalTexture {
    pub fn image_view(&self) -> Option<Arc<ImageView>> { self.uploaded_image_view.clone() }

    fn create_image_view(&mut self,
                       ctx: &VulkanoContext,
                       cbf: &mut RecordingCommandBuffer,
                       _tcx: &mut TaskContext)
    -> TaskResult {
        if self.uploaded_image_view.is_none() {
            unsafe {
                cbf.copy_buffer_to_image(&CopyBufferToImageInfo {
                    src_buffer: self.buf,
                    dst_image: self.image,
                    ..Default::default()
                })?;
            }
            let image_view = ImageView::new_default(ctx.resources().image(self.image)?.image().clone())
                .unwrap();
            self.uploaded_image_view = Some(image_view);
        }
        Ok(())
    }
}

impl AxisAlignedExtent for InternalTexture {
    fn aa_extent(&self) -> Vec2 {
        Vec2 { x: self.raw.info.extent[0] as f32, y: self.raw.info.extent[1] as f32 }
    }

    fn centre(&self) -> Vec2 {
        Vec2::zero()
    }
}

struct WrappedPngReader<R: Read>(png::Reader<R>);

impl<R: Read> Read for WrappedPngReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.0.next_frame(buf).map_err(|_| std::io::Error::from(std::io::ErrorKind::InvalidInput))?;
        Ok(self.0.output_buffer_size())
    }
}

struct TextureHandlerInner {
    loaded_files: BTreeMap<String, Vec<Texture>>,
    textures: BTreeMap<TextureId, InternalTexture>,
    textures_dirty: bool,
}

impl TextureHandlerInner {
    fn new(ctx: &VulkanoContext) -> Result<Self> {
        let mut textures = BTreeMap::new();

        // Create blank texture
        let info = ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::R8G8B8A8_SRGB,
            extent: [1, 1, 1],
            usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
            ..Default::default()
        };
        let buf = ctx.resources().create_buffer(
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST |
                    MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            DeviceLayout::for_value(&[0u8; 4]).context("failed to create layout for [0, 0, 0, 0]")?
        ).map_err(Validated::unwrap)?;

        let image = ctx.resources().create_image(
            info.clone(),
            AllocationCreateInfo {
                allocate_preference: MemoryAllocatePreference::AlwaysAllocate,
                ..AllocationCreateInfo::default()
            }
        ).map_err(Validated::unwrap)?;
        let internal_texture = InternalTexture {
            raw: Arc::new(RawLoadedTexture {
                buf: Colour::white().as_bytes().to_vec(),
                info,
                duration: None,
            }),
            buf,
            image,
            uploaded_image_view: None,
            ref_count: Arc::new(AtomicUsize::new(1)),
        };
        textures.insert(0, internal_texture);

        Ok(Self {
            loaded_files: BTreeMap::new(),
            textures,
            textures_dirty: false,
        })
    }

    fn create_texture(&mut self, ctx: &VulkanoContext, loaded: RawLoadedTexture) -> Result<Texture> {
        // Free up unused textures first.
        self.free_all_unused_textures();

        let id = self.textures.keys()
            .copied()
            .tuple_windows()
            .find(|(a, b)| *a + 1 != *b)
            .map(|(a, _)| a + 1)
            .or_else(|| self.textures.last_key_value().map(|(id, _)| id + 1))
            .expect("empty textures? (blank texture missing)");
        let ref_count = Arc::new(AtomicUsize::new(1));
        let buf = ctx.resources().create_buffer(
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST |
                    MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            DeviceLayout::new(NonZeroDeviceSize::new(loaded.buf.len() as u64).unwrap(), DeviceAlignment::of::<u8>())
                .context("failed to create layout for texture")?
        ).map_err(Validated::unwrap)?;
        let duration = loaded.duration;
        let image = ctx.resources().create_image(
            loaded.info.clone(),
            AllocationCreateInfo {
                allocate_preference: MemoryAllocatePreference::AlwaysAllocate,
                ..AllocationCreateInfo::default()
            }
        ).map_err(Validated::unwrap)?;
        let internal_texture = InternalTexture {
            raw: Arc::new(loaded),
            buf,
            image,
            uploaded_image_view: None,
            ref_count: ref_count.clone(),
        };
        let extent = internal_texture.aa_extent();
        if let Some(existing) = self.textures.insert(id, internal_texture) {
            panic!("tried to use texture id {id}, but ref_count={}",
                   existing.ref_count.load(Ordering::Relaxed));
        }
        self.textures_dirty = true;
        Ok(Texture { id, duration, extent, ref_count })
    }

    fn free_all_unused_textures(&mut self) {
        for unused_id in self.textures.iter()
            .filter(|(_, tex)| tex.ref_count.load(Ordering::Relaxed) == 0)
            .map(|(id, _)| *id)
            .collect_vec() {
            info!("freeing texture id {unused_id}");
            self.textures.remove(&unused_id);
        }
    }

    fn free_unused_files(&mut self) {
        for filename in self.loaded_files.iter()
            .filter(|(_, textures)| textures.iter().all(|tex| {
                tex.ref_count.load(Ordering::Relaxed) <= 1
            }))
            .map(|(filename, _)| filename.clone())
            .collect_vec() {
            info!("freeing texture {filename}");
            self.loaded_files.remove(&filename);
        }
        self.free_all_unused_textures();
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
        Self {
            materials: BTreeMap::new(),
            materials_inverse: HashMap::new(),
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
            let id = self.materials
                .last_key_value().map_or(0, |(&k, _)| k + 1);
            self.materials.insert(id, material.clone());
            self.materials_inverse.insert(material, id);
            self.dirty = true;
            id
        }
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
        let inner = Arc::new(Mutex::new(TextureHandlerInner::new(&ctx)?));
        let material_handler = Arc::new(Mutex::new(MaterialHandler::new()));
        Ok(Self { ctx, inner, material_handler })
    }

    pub fn material_from_texture(&self, texture: &Texture, area: &Rect) -> MaterialId {
        self.material_handler.lock().unwrap().material_from_texture(texture, area)
    }

    // TODO: implement spawn_load_file().
    pub fn wait_load_file(&self, filename: impl AsRef<str>) -> Result<Texture> {
        let filename = filename.as_ref().to_string();
        // Beware: do not lock `inner` longer than necessary.
        if let Some(texture) = self.inner.lock().unwrap()
                .loaded_files.get(&filename)
                .and_then(|v| v.first()) {
            return Ok(texture.clone());
        }
        let loaded = Self::load_file_inner(&filename)?;
        let mut inner = self.inner.lock().unwrap();
        let texture = inner.create_texture(&self.ctx, loaded)?;
        info!("loaded texture: {} = {:?}", filename, texture.id());
        inner.loaded_files.insert(filename, vec![texture.clone()]);
        Ok(texture)
    }
    fn load_file_inner(filename: &str) -> Result<RawLoadedTexture> {
        let path = Path::new(filename);
        let ext = path.extension()
            .with_context(|| format!("no file extension: {filename}"))?
            .to_str()
            .with_context(|| format!("failed conversion from OsStr: {filename}"))?;
        match ext {
            "png" => Self::load_file_inner_png(filename),
            "aseprite" => Ok(Self::load_file_inner_animated_aseprite(filename)?
                .into_iter().next().context("loading aseprite file succeeded but no frames?")?),
            ext => bail!("unknown file extension: {ext} (while loading {filename})"),
        }
    }
    fn load_file_inner_png(filename: &str) -> Result<RawLoadedTexture> {
        let png_bytes = fs::read(filename)?;
        let cursor = Cursor::new(png_bytes);
        let decoder = png::Decoder::new(cursor);
        let reader = decoder.read_info()?;
        let info = reader.info();

        if info.srgb.is_none() {
            error!("loading {}: SRGB not enabled, may display incorrectly", filename);
        }
        if info.color_type != ColorType::Rgba {
            error!("loading {}: no alpha channel, *will* display incorrectly (re-encode as RGBA)", filename);
        }
        if let Some(animation_control) = info.animation_control {
            if animation_control.num_frames != 1 {
                error!("loading {}: unexpected num_frames {} (animated PNG not supported)",
                    filename, animation_control.num_frames);
            }
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
        let textures = results.into_iter().map(|loaded| {
            inner.create_texture(&self.ctx, loaded)
        }).collect::<Result<Vec<_>>>()?;
        inner.loaded_files.insert(filename, textures.clone());
        Ok(textures)
    }
    fn load_file_inner_animated(filename: &str) -> Result<Vec<RawLoadedTexture>> {
        let path = Path::new(filename);
        let ext = path.extension()
            .with_context(|| format!("no file extension: {filename}"))?
            .to_str()
            .with_context(|| format!("failed conversion from OsStr: {filename}"))?;
        match ext {
            "png" => todo!("png files with multiple frames not supported"),
            "aseprite" => Self::load_file_inner_animated_aseprite(filename),
            ext => bail!("unknown file extension: {ext} (while loading {filename})"),
        }
    }
    fn load_file_inner_animated_aseprite(filename: &str) -> Result<Vec<RawLoadedTexture>> {
        let ase = AsepriteFile::read_file(filename.as_ref())?;
        (0..ase.num_frames())
            .map(|i| ase.frame(i))
            .map(|frame| {
                let image = frame.image();
                let mut loaded = Self::load_reader_rgba_inner(
                     &mut  image.to_vec().as_slice(),
                     image.width(),
                     image.height(),
                     Format::R8G8B8A8_SRGB
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
        format: Format
    ) -> Result<Texture> {
        let loaded = Self::load_reader_rgba_inner(reader, width, height, format)?;
        let mut inner = self.inner.lock().unwrap();
        inner.create_texture(&self.ctx, loaded)
    }
    fn load_reader_rgba_inner<R: Read>(
        reader: &mut R,
        width: u32,
        height: u32,
        format: Format
    ) -> Result<RawLoadedTexture> {
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
        Ok(RawLoadedTexture {
            buf,
            info: image_create_info,
            duration: None
        })
    }

    pub fn wait_free_unused_files(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.free_unused_files();
    }

    pub(crate) fn ready_values(&self) -> Vec<InternalTexture> {
        self.inner.lock().unwrap().textures.values()
            .filter(|t| t.uploaded_image_view.is_some())
            .cloned().collect()
    }

    pub(crate) fn get_updated_materials(&self) -> Option<Vec<Material>> {
        let mut material_handler = self.material_handler.lock().unwrap();
        if material_handler.dirty {
            let rv = material_handler.materials.values().cloned().collect();
            material_handler.dirty = false;
            Some(rv)
        } else {
            None
        }
    }

    pub fn wait_get_raw(&self, texture_id: TextureId) -> Result<Option<Vec<Vec<Colour>>>> {
        let Some(tex) = self.inner.lock().unwrap().textures.get(&texture_id).cloned() else {
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
        self.inner.lock().unwrap().textures_dirty
    }
    pub(crate) fn build_task_graph(&self, task_graph: &mut TaskGraph<VulkanoContext>)
            -> (NodeId, Vec<Id<Image>>) {
        let mut inner = self.inner.lock().unwrap();
        for tex in inner.textures.values() {
            task_graph.add_host_buffer_access(tex.buf, HostAccessType::Write);
        }
        let mut upload_node = task_graph.create_task_node(
            "texture_handler_upload",
            QueueFamilyType::Transfer,
            UploadTexturesTask { inner: Arc::new(self.clone()) }
        );
        let mut images = Vec::new();
        for tex in inner.textures.values() {
            upload_node.buffer_access(tex.buf, AccessType::CopyTransferRead);
            upload_node.image_access(tex.image, AccessType::CopyTransferWrite, ImageLayoutType::Optimal);
            images.push(tex.image);
        }
        let upload_node = upload_node.build();
        inner.textures_dirty = false;
        (upload_node, images)
    }
}

pub(crate) struct UploadTexturesTask {
    pub(crate) inner: Arc<TextureHandler>
}

impl Task for UploadTexturesTask {
    type World = VulkanoContext;

    unsafe fn execute(&self, cbf: &mut RecordingCommandBuffer, tcx: &mut TaskContext, world: &Self::World) -> TaskResult {
        let mut inner = self.inner.inner.lock().unwrap();
        if inner.textures_dirty {
            info!("textures_dirty, skip upload");
            return Ok(());
        }
        let textures_to_upload = inner.textures.iter_mut()
            .filter(|(_, tex)| tex.uploaded_image_view.is_none())
            .collect_vec();
        if textures_to_upload.is_empty() {
            return Ok(());
        }

        for (id, tex) in textures_to_upload {
            tcx.write_buffer::<[u8]>(tex.buf, ..)?.clone_from_slice(&tex.raw.buf);
            tex.create_image_view(world, cbf, tcx)?;
            info!("created image view for: {:?}", id);
        }
        Ok(())
    }
}
