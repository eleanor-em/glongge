use std::sync::{Arc, Mutex};
use crate::resource::texture::TextureHandler;

pub mod texture;

pub struct ResourceHandler {
    pub texture: TextureHandler,
}

impl ResourceHandler {
    // TODO: can probably do smarter synchronisation than just Mutex here.
    pub fn new() -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self {
            texture: TextureHandler::new()
        }))
    }
}
