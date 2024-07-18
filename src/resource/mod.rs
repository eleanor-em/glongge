#[allow(unused_imports)]
use crate::core::prelude::*;

use crate::{
    core::vk_core::VulkanoContext,
    resource::texture::TextureHandler
};
use crate::resource::sound::SoundHandler;

pub mod sprite;
pub mod texture;
pub mod sound;

#[derive(Clone)]
pub struct ResourceHandler {
    pub texture: TextureHandler,
    pub sound: SoundHandler,
}

impl ResourceHandler {
    pub fn new(ctx: VulkanoContext) -> Result<Self> {
        Ok(Self {
            texture: TextureHandler::new(ctx.clone()),
            sound: SoundHandler::new()?,
        })
    }
}
