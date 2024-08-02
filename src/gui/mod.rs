use std::ops::{Deref, DerefMut};

pub mod render;
pub mod window;
pub mod command;

// imgui::Context does not implement Send, for some reason.
pub struct ImGuiContext(imgui::Context);

unsafe impl Send for ImGuiContext {}

impl ImGuiContext {
    pub fn new() -> Self { Self(imgui::Context::create()) }
}

impl Default for ImGuiContext {
    fn default() -> Self {
        Self::new()
    }
}

impl Deref for ImGuiContext {
    type Target = imgui::Context;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for ImGuiContext {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
