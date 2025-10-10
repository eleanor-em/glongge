use crate::core::vk::AdjustedViewport;
use crate::core::vk::vk_ctx::VulkanoContext;
use crate::util::UniqueShared;
use crate::{
    core::prelude::*,
    resource::{sound::SoundHandler, texture::TextureHandler},
};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

pub mod font;
pub mod rich_text;
pub mod sound;
pub mod sprite;
pub mod texture;

static CREATED_RESOURCE_HANDLER: AtomicBool = AtomicBool::new(false);
#[derive(Clone)]
pub struct ResourceHandler {
    pub texture: Arc<TextureHandler>,
    pub sound: Arc<SoundHandler>,
    viewport: UniqueShared<AdjustedViewport>,
}

impl ResourceHandler {
    pub(crate) fn new(
        ctx: &VulkanoContext,
        viewport: UniqueShared<AdjustedViewport>,
    ) -> Result<Self> {
        let resource_handler_already_exists = CREATED_RESOURCE_HANDLER.swap(true, Ordering::SeqCst);
        check_false!(resource_handler_already_exists);
        Ok(Self {
            texture: Arc::new(TextureHandler::new(ctx.clone())?),
            sound: Arc::new(SoundHandler::new()?),
            viewport,
        })
    }

    pub fn wait_all(&self) -> Result<()> {
        self.sound.wait()?;
        Ok(())
    }

    pub fn total_scale_factor(&self) -> f32 {
        self.viewport.lock().total_scale_factor()
    }
    pub fn viewport_extent(&self) -> Vec2 {
        self.viewport.lock().extent()
    }
}

pub trait Loader<T> {
    fn spawn_load_file(&self, filename: impl AsRef<str>);
    fn wait_load_file(&self, filename: impl AsRef<str>) -> Result<T>;

    fn wait(&self) -> Result<()>;
}
