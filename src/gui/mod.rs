use std::ops::{Deref, DerefMut};

pub mod render;
pub mod window;

// imgui::Context does not implement Send, for some reason.
pub struct GuiContext(imgui::Context);

unsafe impl Send for GuiContext {}

impl GuiContext {
    pub fn new() -> Self { Self(imgui::Context::create()) }
}

impl Default for GuiContext {
    fn default() -> Self {
        Self::new()
    }
}

impl Deref for GuiContext {
    type Target = imgui::Context;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for GuiContext {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

pub type GuiUi = imgui::Ui;
