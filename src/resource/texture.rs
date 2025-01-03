use std::{
    collections::BTreeMap,
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
        RwLock,
        atomic::{AtomicUsize, Ordering}
    },
};
use std::time::Duration;
use asefile::AsepriteFile;
use num_traits::Zero;

use vulkano::{buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer}, command_buffer::{AutoCommandBufferBuilder, CopyBufferToImageInfo, PrimaryAutoCommandBuffer}, format::Format, image::{
    Image,
    ImageCreateInfo,
    ImageType,
    ImageUsage,
    view::ImageView
}, memory::allocator::{AllocationCreateInfo, MemoryTypeFilter}, DeviceSize, Validated};

use png::ColorType;
use vulkano::memory::allocator::MemoryAllocatePreference;
use crate::core::prelude::*;
use crate::core::vk::vk_ctx::VulkanoContext;
use crate::util::gg_err;

struct RawLoadedTexture {
    buf: Subbuffer<[u8]>,
    info: ImageCreateInfo,
    duration: Option<Duration>,
}

pub type TextureId = u16;

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

impl From<Texture> for u32 {
    fn from(value: Texture) -> Self {
        u32::from(value.id)
    }
}

#[derive(Clone)]
pub(crate) struct InternalTexture {
    buf: Subbuffer<[u8]>,
    info: ImageCreateInfo,
    cached_image_view: Option<Arc<ImageView>>,
    ref_count: Arc<AtomicUsize>,
}

impl InternalTexture {
    pub fn image_view(&self) -> Option<Arc<ImageView>> { self.cached_image_view.clone() }

    fn create_image_view(&mut self,
                       ctx: &VulkanoContext,
                       builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>)
    -> Result<Arc<ImageView>> {
        match self.cached_image_view.clone() {
            None => {
                let image = Image::new(
                    ctx.memory_allocator().clone(),
                    self.info.clone(),
                    AllocationCreateInfo {
                        allocate_preference: MemoryAllocatePreference::AlwaysAllocate,
                        ..AllocationCreateInfo::default()
                    }
                ).map_err(Validated::unwrap)?;
                builder.copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
                    self.buf.clone(),
                    image.clone()
                ))?;
                let image_view = ImageView::new_default(image)?;
                self.cached_image_view = Some(image_view.clone());
                Ok(image_view)
            },
            Some(image_view) => Ok(image_view),
        }
    }
}

impl AxisAlignedExtent for InternalTexture {
    fn aa_extent(&self) -> Vec2 {
        Vec2 { x: f64::from(self.info.extent[0]), y: f64::from(self.info.extent[1]) }
    }

    fn centre(&self) -> Vec2 {
        Vec2::zero()
    }
}

#[derive(Clone)]
pub(crate) enum CachedTexture {
    Loading,
    Ready(Arc<InternalTexture>),
}

impl CachedTexture {
    pub fn ready(self) -> Option<Arc<InternalTexture>> {
        match self {
            Self::Loading => None,
            Self::Ready(tex) => Some(tex)
        }
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
        let buf: Subbuffer<[u8]> = Buffer::new_slice(
            ctx.memory_allocator(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC, ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST |
                    MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            DeviceSize::from(4 as char)
        ).map_err(Validated::unwrap)?;
        buf.write()?.swap_with_slice(&mut Colour::white().as_bytes());

        let internal_texture = InternalTexture {
            buf,
            info,
            cached_image_view: None,
            ref_count: Arc::new(AtomicUsize::new(1)),
        };
        textures.insert(0, internal_texture);

        Ok(Self {
            loaded_files: BTreeMap::new(),
            textures,
        })
    }

    fn create_texture(&mut self, loaded: RawLoadedTexture) -> Texture {
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
        let internal_texture = InternalTexture {
            buf: loaded.buf,
            info: loaded.info,
            cached_image_view: None,
            ref_count: ref_count.clone(),
        };
        let extent = internal_texture.aa_extent();
        if let Some(existing) = self.textures.insert(id, internal_texture) {
            panic!("tried to use texture id {id}, but ref_count={}",
                   existing.ref_count.load(Ordering::Relaxed));
        }
        Texture { id, duration: loaded.duration, extent, ref_count }
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

#[derive(Clone)]
pub struct TextureHandler {
    ctx: VulkanoContext,
    inner: Arc<Mutex<TextureHandlerInner>>,
    cached_textures: Arc<RwLock<BTreeMap<TextureId, CachedTexture>>>,
}

impl TextureHandler {
    pub(crate) fn new(ctx: VulkanoContext) -> Result<Self> {
        let inner = Arc::new(Mutex::new(TextureHandlerInner::new(&ctx)?));
        let cached_textures = Arc::new(RwLock::new(BTreeMap::new()));
        Ok(Self { ctx, inner, cached_textures })
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
        let loaded = self.load_file_inner(&filename)?;
        let mut inner = self.inner.lock().unwrap();
        let texture = inner.create_texture(loaded);
        info!("loaded texture: {} = {:?}", filename, texture.id());
        inner.loaded_files.insert(filename, vec![texture.clone()]);
        self.cached_textures.write().unwrap().insert(texture.id(), CachedTexture::Loading);
        Ok(texture)
    }
    fn load_file_inner(&self, filename: &str) -> Result<RawLoadedTexture> {
        let path = Path::new(filename);
        let ext = path.extension()
            .with_context(|| format!("no file extension: {filename}"))?
            .to_str()
            .with_context(|| format!("failed conversion from OsStr: {filename}"))?;
        match ext {
            "png" => self.load_file_inner_png(filename),
            "aseprite" => Ok(self.load_file_inner_animated_aseprite(filename)?
                .into_iter().next().context("loading aseprite file succeeded but no frames?")?),
            ext => bail!("unknown file extension: {ext} (while loading {filename})"),
        }
    }
    fn load_file_inner_png(&self, filename: &str) -> Result<RawLoadedTexture> {
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
        self.load_reader_rgba_inner(&mut WrappedPngReader(reader), width, height, format)
    }

    pub fn wait_load_file_animated(&self, filename: impl AsRef<str>) -> Result<Vec<Texture>> {
        let filename = filename.as_ref().to_string();
        // Beware: do not lock `inner` longer than necessary.
        if let Some(texture) = self.inner.lock().unwrap().loaded_files.get(&filename) {
            return Ok(texture.clone());
        }
        let results = self.load_file_inner_animated(&filename)?;
        let mut inner = self.inner.lock().unwrap();
        let textures = results.into_iter().map(|loaded| {
            let texture = inner.create_texture(loaded);
            self.cached_textures.write().unwrap().insert(texture.id(), CachedTexture::Loading);
            texture
        }).collect_vec();
        inner.loaded_files.insert(filename, textures.clone());
        Ok(textures)
    }
    fn load_file_inner_animated(&self, filename: &str) -> Result<Vec<RawLoadedTexture>> {
        let path = Path::new(filename);
        let ext = path.extension()
            .with_context(|| format!("no file extension: {filename}"))?
            .to_str()
            .with_context(|| format!("failed conversion from OsStr: {filename}"))?;
        match ext {
            "png" => todo!("png files with multiple frames not supported"),
            "aseprite" => self.load_file_inner_animated_aseprite(filename),
            ext => bail!("unknown file extension: {ext} (while loading {filename})"),
        }
    }
    fn load_file_inner_animated_aseprite(&self, filename: &str) -> Result<Vec<RawLoadedTexture>> {
        let ase = AsepriteFile::read_file(filename.as_ref())?;
        (0..ase.num_frames())
            .map(|i| ase.frame(i))
            .map(|frame| {
                let image = frame.image();
                let mut loaded = self.load_reader_rgba_inner(
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
        let loaded = self.load_reader_rgba_inner(reader, width, height, format)?;

        let mut inner = self.inner.lock().unwrap();
        let texture = inner.create_texture(loaded);
        self.cached_textures.write().unwrap().insert(texture.id(), CachedTexture::Loading);
        Ok(texture)
    }
    fn load_reader_rgba_inner<R: Read>(
        &self,
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
        let buf = Buffer::new_slice(
            self.ctx.memory_allocator(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC, ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST |
                    MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            DeviceSize::from(width * height * 4)
        ).map_err(Validated::unwrap)?;

        reader.read_exact(&mut buf.write()?)?;

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

    pub(crate) fn wait_maybe_upload_textures(&self, ctx: &VulkanoContext, builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>) -> Result<(), gg_err::CatchOutOfDate> {
        let mut inner = self.inner.lock().unwrap();
        let textures_to_upload = inner.textures.iter_mut()
            .filter(|(_, tex)| tex.cached_image_view.is_none())
            .collect_vec();
        if textures_to_upload.is_empty() {
            return Ok(());
        }

        for (id, tex) in textures_to_upload {
            info!("created image view for: {:?}", id);
            tex.create_image_view(ctx, builder).map_err(gg_err::CatchOutOfDate::from)?;
            self.cached_textures.write().unwrap().insert(*id, CachedTexture::Ready(Arc::new(tex.clone())));
        }
        Ok(())
    }

    // Uses RwLock. Blocks only if another thread is loading a texture, see wait_load_file().
    pub(crate) fn ready_values(&self) -> Vec<Arc<InternalTexture>> {
        self.cached_textures.read().unwrap()
            .values()
            .filter_map(|tex| match tex {
                CachedTexture::Loading => None,
                CachedTexture::Ready(tex) => Some(tex)
            })
            .cloned().collect()
    }

    // Uses RwLock. Blocks only if another thread is loading a texture, see wait_load_file().
    pub(crate) fn get(&self, texture_id: TextureId) -> Option<CachedTexture> {
        self.cached_textures.read().unwrap().get(&texture_id).cloned()
    }
    pub(crate) fn get_nonblank(&self, texture_id: TextureId) -> Option<CachedTexture> {
        if texture_id.is_zero() {
            None
        } else {
            self.get(texture_id)
        }
    }
    pub fn wait_get_raw(&self, texture_id: TextureId) -> Result<Option<Vec<Vec<Colour>>>> {
        let Some(tex) = self.inner.lock().unwrap().textures.get(&texture_id).cloned() else {
            return Ok(None);
        };
        let w = tex.info.extent[0] as usize;
        let h = tex.info.extent[1] as usize;
        let mut rv = vec![vec![Colour::empty(); w]; h];
        let mut x = 0;
        let mut y = 0;
        for bytes in tex.buf.read()?.chunks(4) {
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
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TextureSubArea {
    rect: Rect,
}

impl TextureSubArea {
    pub fn new(centre: Vec2i, half_widths: Vec2i) -> Self {
        Self::from_rect(Rect::new(centre.into(), half_widths.into()))
    }
    pub fn from_rect(rect: Rect) -> Self {
        Self { rect }
    }

    pub(crate) fn uv<T: AxisAlignedExtent>(&self, texture: &T, raw_uv: Vec2) -> Vec2 {
        if self.rect == Rect::default() {
            raw_uv
        } else {
            let extent = texture.aa_extent();
            let u0 = self.rect.top_left().x / extent.x;
            let v0 = self.rect.top_left().y / extent.y;
            let u1 = self.rect.bottom_right().x / extent.x;
            let v1 = self.rect.bottom_right().y / extent.y;
            Vec2 { x: linalg::lerp(u0, u1, raw_uv.x), y: linalg::lerp(v0, v1, raw_uv.y) }
        }
    }
}

impl AxisAlignedExtent for TextureSubArea {
    fn aa_extent(&self) -> Vec2 {
        self.rect.aa_extent()
    }

    fn centre(&self) -> Vec2 {
        self.rect.centre()
    }
}
