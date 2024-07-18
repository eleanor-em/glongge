#[allow(unused_imports)]
use crate::core::prelude::*;

use std::{
    collections::BTreeMap,
    default::Default,
    fs,
    io::Cursor,
    path::Path,
    sync::{
        Arc,
        Mutex,
        MappedRwLockReadGuard,
        RwLock,
        RwLockReadGuard,
        atomic::{AtomicUsize, Ordering}
    }
};

use png::ColorType;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferToImageInfo, PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract},
    format::Format,
    image::{
        Image,
        ImageCreateInfo,
        ImageType,
        ImageUsage,
        view::ImageView
    },
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    sync::GpuFuture,
    DeviceSize,
    Validated,
};
use crate::{
    core::{
        linalg::Vec2,
        vk_core::VulkanoContext,
    }
};

pub const MAX_TEXTURE_COUNT: usize = 1023;

static NEXT_TEXTURE_ID: AtomicUsize = AtomicUsize::new(0);
#[derive(Copy, Clone, Debug, Default, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct TextureId(usize);

impl TextureId {
    fn next() -> Self {
        Self(NEXT_TEXTURE_ID.fetch_add(1, Ordering::Relaxed))
    }
}

impl From<TextureId> for u32 {
    fn from(value: TextureId) -> Self {
        // TODO: better bounds checking
        check_lt!(value.0, MAX_TEXTURE_COUNT);
        value.0 as u32
    }
}

#[derive(Clone)]
pub struct Texture {
    buf: Subbuffer<[u8]>,
    info: ImageCreateInfo,
    cached_image_view: Option<Arc<ImageView>>,
}

impl Texture {
    pub fn image_view(&self) -> Option<Arc<ImageView>> { self.cached_image_view.clone() }
    pub fn extent(&self) -> Vec2 { Vec2 { x: self.info.extent[0] as f64, y: self.info.extent[1] as f64 } }

    fn create_image_view(&mut self,
                       ctx: &VulkanoContext,
                       builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>)
    -> Result<Arc<ImageView>> {
        match self.cached_image_view.clone() {
            None => {
                let image = Image::new(
                    ctx.memory_allocator().clone(),
                    self.info.clone(),
                    AllocationCreateInfo::default()
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

struct TextureHandlerInner {
    loaded_files: BTreeMap<String, TextureId>,
    textures: BTreeMap<TextureId, Texture>,
}

#[derive(Clone)]
pub struct TextureHandler {
    ctx: VulkanoContext,
    inner: Arc<Mutex<TextureHandlerInner>>,
    // TODO: consider using Option<Arc<Texture>> to indicate "texture file loaded but image not ready"
    cached_textures: Arc<RwLock<BTreeMap<TextureId, Arc<Texture>>>>,
}

impl TextureHandler {
    pub(crate) fn new(ctx: VulkanoContext) -> Self {
        let mut textures = BTreeMap::new();
        textures.insert(TextureId(0), Self::blank_texture(&ctx)
            .expect("could not create blank texture"));

        let inner = Arc::new(Mutex::new(TextureHandlerInner {
            loaded_files: BTreeMap::new(),
            textures: textures.clone(),
        }));
        let cached_textures = Arc::new(RwLock::new(BTreeMap::new()));
        Self { ctx, inner, cached_textures }
    }

    fn blank_texture(ctx: &VulkanoContext) -> Result<Texture> {
        let image_create_info = ImageCreateInfo {
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
            4 as DeviceSize
        ).map_err(Validated::unwrap)?;
        buf.write()?.swap_with_slice(&mut [255, 255, 255, 255]);
        Ok(Texture {
            buf, info: image_create_info,
            cached_image_view: None,
        })
    }

    pub fn wait_load_file(&mut self, filename: String) -> Result<TextureId> {
        // Beware: do not lock `inner` longer than necessary.
        if let Some(id) = self.inner.lock().unwrap().loaded_files.get(&filename) {
            return Ok(*id);
        }

        let texture = self.load_file_inner(&filename)?;

        let mut inner = self.inner.lock().unwrap();
        let texture_id = TextureId::next();
        inner.loaded_files.insert(filename, texture_id);
        inner.textures.insert(texture_id, texture);
        Ok(texture_id)
    }

    fn load_file_inner(&self, filename: &str) -> Result<Texture> {
        let path = Path::new(filename);
        let ext = path.extension()
            .ok_or(anyhow!("no file extension: {}", filename))?
            .to_str()
            .ok_or(anyhow!("failed conversion from OsStr: {}", filename))?;
        match ext {
            "png" => self.load_file_inner_png(filename),
            "aseprite" => unimplemented!("TODO: use asefile crate"),
            ext => bail!("unknown file extension: {} (while loading {})", ext, filename),
        }
    }
    fn load_file_inner_png(&self, filename: &str) -> Result<Texture> {
        let png_bytes = fs::read(filename)?;
        let cursor = Cursor::new(png_bytes);
        let decoder = png::Decoder::new(cursor);
        let mut reader = decoder.read_info()?;
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

        let image_create_info = ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format,
            extent: [info.width, info.height, 1],
            usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
            ..Default::default()
        };
        let mut image_data = Vec::new();
        let depth: u32 = match info.bit_depth {
            png::BitDepth::One => 1,
            png::BitDepth::Two => 2,
            png::BitDepth::Four => 4,
            png::BitDepth::Eight => 8,
            png::BitDepth::Sixteen => 16,
        };
        image_data.resize((info.width * info.height * depth) as usize, 0);
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
            (info.width * info.height * 4) as DeviceSize
        ).map_err(Validated::unwrap)?;
        reader.next_frame(&mut buf.write()?)?;

        Ok(Texture {
            buf,
            info: image_create_info,
            cached_image_view: None,
        })
    }

    pub fn wait_build_command_buffer(&mut self, ctx: &VulkanoContext) -> Result<Option<Box<dyn GpuFuture>>> {
        let mut uploads = AutoCommandBufferBuilder::primary(
            ctx.command_buffer_allocator(),
            ctx.queue().queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        ).map_err(Validated::unwrap)?;

        // Beware: do not lock `inner` longer than necessary.
        {
            let mut inner = self.inner.lock().unwrap();
            let textures_to_upload = inner.textures.iter_mut()
                .filter(|(_, tex)| tex.cached_image_view.is_none())
                .collect_vec();
            if textures_to_upload.is_empty() {
                return Ok(None);
            }

            for (id, tex) in textures_to_upload {
                tex.create_image_view(ctx, &mut uploads)?;
                self.cached_textures.write().unwrap().insert(*id, Arc::new(tex.clone()));
            }
        }

        Ok(Some(uploads
            .build().map_err(Validated::unwrap)?
            .execute(ctx.queue())?
            .boxed()))
    }

    // Uses RwLock. Blocks only if another thread is loading a texture, see wait_load_file().
    pub fn values(&self) -> Vec<Arc<Texture>> {
        self.cached_textures.read().unwrap().values().cloned().collect()
    }

    // Uses RwLock. Blocks only if another thread is loading a texture, see wait_load_file().
    pub fn get(&self, texture_id: TextureId) -> Option<MappedRwLockReadGuard<Arc<Texture>>> {
        RwLockReadGuard::try_map(self.cached_textures.read().unwrap(), |textures| {
            textures.get(&texture_id)
        }).ok()
    }
}
