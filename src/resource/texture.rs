use std::{
    collections::BTreeMap,
    default::Default,
    fs,
    io::Cursor,
    path::Path,
    sync::{
        Arc,
        Mutex
    },
};

use anyhow::{anyhow, bail, Result};
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
    assert::check_lt,
    core::vk_core::VulkanoContext
};

pub const MAX_TEXTURE_COUNT: usize = 1023;

#[derive(Copy, Clone, Debug, Default, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct TextureId(usize);

impl TextureId {
    fn next(&self) -> Self {
        Self(self.0 + 1)
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

    fn create_image_view(&mut self,
                       ctx: &VulkanoContext,
                       builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>)
    -> Result<Arc<ImageView>> {
        if self.cached_image_view.is_none() {
            let image = Image::new(
                ctx.memory_allocator().clone(),
                self.info.clone(),
                AllocationCreateInfo::default()
            ).map_err(Validated::unwrap)?;
            builder.copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
                self.buf.clone(),
                image.clone()
            ))?;
            self.cached_image_view = Some(ImageView::new_default(image)?);
        }
        Ok(self.cached_image_view.clone().unwrap())
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
}

impl TextureHandler {
    pub(crate) fn new(ctx: VulkanoContext) -> Self {
        let mut textures = BTreeMap::new();
        textures.insert(TextureId(0), Self::blank_texture(&ctx)
            .expect("could not create blank texture"));
        Self {
            ctx,
            inner: Arc::new(Mutex::new(TextureHandlerInner {
                loaded_files: BTreeMap::new(),
                textures,
            })),
        }
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
        if let Some(id) = self.inner.lock().unwrap().loaded_files.get(&filename) {
            return Ok(*id);
        }
        let texture = self.load_file_inner(&filename)?;
        Ok({
            let mut inner = self.inner.lock().unwrap();
            let texture_id = match inner.textures.last_key_value() {
                Some((id, _)) => id.next(),
                None => TextureId(0),
            };
            inner.loaded_files.insert(filename, texture_id);
            inner.textures.insert(texture_id, texture);
            texture_id
        })
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
        let image_create_info = ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::R8G8B8A8_SRGB,
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

        {
            let mut inner = self.inner.lock().unwrap();
            let textures_to_upload = inner.textures.values_mut()
                .filter(|tex| tex.cached_image_view.is_none())
                .collect::<Vec<_>>();
            if textures_to_upload.is_empty() {
                return Ok(None);
            }

            for tex in textures_to_upload {
                tex.create_image_view(ctx, &mut uploads)?;
            }
        }

        Ok(Some(uploads
            .build().map_err(Validated::unwrap)?
            .execute(ctx.queue())?
            .boxed()))
    }

    pub fn wait_values(&self) -> Vec<Texture> {
        self.inner.lock().unwrap().textures.values().cloned().collect()
    }
}
