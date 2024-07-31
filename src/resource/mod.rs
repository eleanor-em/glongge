use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering}
};
use crate::{
    core::{
        prelude::*,
        vk::VulkanoContext
    },
    resource::{
        sound::SoundHandler,
        texture::TextureHandler,
    }
};

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

    pub fn wait_all(&self) -> Result<()>{
        self.sound.wait()?;
        Ok(())
    }
}

pub trait Loader<T> {
    fn spawn_load_file(&self, filename: String);
    fn wait_load_file(&self, filename: String) -> Result<T>;

    fn wait(&self) -> Result<()>;
}
