use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use crate::core::prelude::*;

use crate::{
    core::vk::VulkanoContext,
    resource::texture::TextureHandler
};
use crate::resource::sound::SoundHandler;

pub mod sprite;
pub mod texture;
pub mod sound;
pub mod font;

static CREATED_RESOURCE_HANDLER: AtomicBool = AtomicBool::new(false);
#[derive(Clone)]
pub struct ResourceHandler {
    pub texture: Arc<TextureHandler>,
    pub sound: Arc<SoundHandler>,
}

impl ResourceHandler {
    pub fn new(ctx: &VulkanoContext) -> Result<Self> {
        let resource_handler_already_exists = CREATED_RESOURCE_HANDLER.swap(true, Ordering::Relaxed);
        check_false!(resource_handler_already_exists);
        Ok(Self {
            texture: Arc::new(TextureHandler::new(ctx.clone())?),
            sound: Arc::new(SoundHandler::new()?),
        })
    }
}

pub trait Loader<T> {
    fn spawn_load_file(&self, filename: String);
    fn wait_load_file(&self, filename: String) -> Result<T>;
}
