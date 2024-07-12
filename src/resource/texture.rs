use std::{
    collections::{BTreeMap, HashMap},
    default::Default,
    fs,
    io::Cursor,
    sync::Arc
};

use anyhow::Result;
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

use crate::core::vk_core::VulkanoContext;

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
// TODO: remove pub
pub struct TextureId(pub usize);

impl TextureId {
    pub fn no_texture() -> Self { Self(u32::MAX as usize) }
    fn next(&self) -> Self {
        Self(self.0 + 1)
    }
}

impl From<TextureId> for u32 {
    fn from(value: TextureId) -> Self {
        value.0 as u32
    }
}

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

pub struct TextureHandler {
    loaded_files: HashMap<String, TextureId>,
    textures: BTreeMap<TextureId, Texture>,
}

impl TextureHandler {
    pub(crate) fn new() -> Self {
        Self { loaded_files: HashMap::new(), textures: BTreeMap::new() }
    }
    pub fn load_file(&mut self, ctx: &VulkanoContext, filename: String) -> Result<TextureId> {
        if let Some(id) = self.loaded_files.get(&filename) {
            return Ok(*id);
        }

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
            ctx.memory_allocator(),
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

        let texture_id = match self.textures.last_key_value() {
            Some((id, _)) => id.next(),
            None => TextureId(0),
        };
        let texture = Texture {
            buf,
            info: image_create_info,
            cached_image_view: None,
        };
        self.textures.insert(texture_id, texture);
        Ok(texture_id)
    }

    pub fn build_command_buffer(&mut self, ctx: &VulkanoContext) -> Result<Option<Box<dyn GpuFuture>>> {
        let textures_to_upload = self.textures.values_mut()
            .filter(|tex| tex.cached_image_view.is_none())
            .collect::<Vec<_>>();
        if textures_to_upload.is_empty() {
            return Ok(None);
        }

        let mut uploads = AutoCommandBufferBuilder::primary(
            ctx.command_buffer_allocator(),
            ctx.queue().queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        ).map_err(Validated::unwrap)?;

        for tex in textures_to_upload {
            tex.create_image_view(ctx, &mut uploads)?;
        }

        Ok(Some(uploads
            .build().map_err(Validated::unwrap)?
            .execute(ctx.queue())?
            .boxed()))
    }

    pub fn get(&self, texture_id: TextureId) -> Option<&Texture> {
        self.textures.get(&texture_id)
    }
    pub fn get_mut(&mut self, texture_id: TextureId) -> Option<&mut Texture> {
        self.textures.get_mut(&texture_id)
    }

    pub fn values(&self) -> Vec<&Texture> {
        self.textures.values().collect()
    }
}
