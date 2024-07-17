#[allow(unused_imports)]
use crate::core::prelude::*;

use crate::{
    core::vk_core::VulkanoContext,
    resource::texture::TextureHandler
};

pub mod sprite;
pub mod texture;

#[derive(Clone)]
pub struct ResourceHandler {
    pub texture: TextureHandler,
}

impl ResourceHandler {
    pub fn new(ctx: VulkanoContext) -> Self {
        Self {
            texture: TextureHandler::new(ctx.clone()),
        }
    }
}
