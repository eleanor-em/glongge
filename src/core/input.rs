use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};
use egui_winit::winit::event::ElementState;

pub use egui_winit::winit::keyboard::KeyCode as KeyCode;
use num_traits::Zero;
use crate::core::prelude::*;
use crate::core::vk::AdjustedViewport;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InputState {
    Pressed,
    Held,
    Released,
}

#[derive(Clone)]
pub struct InputHandler {
    data: BTreeMap<KeyCode, InputState>,
    queued_events: Vec<(KeyCode, ElementState)>,
    mouse_pos: Vec2,
    viewport: AdjustedViewport,
    mod_shift: bool,
    mod_alt: bool,
    mod_ctrl: bool,
    mod_super: bool,
}

impl InputHandler {
    pub fn new() -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(InputHandler {
            data: BTreeMap::new(),
            queued_events: Vec::new(),
            mouse_pos: Vec2::zero(),
            viewport: AdjustedViewport::default(),
            mod_shift: false,
            mod_alt: false,
            mod_ctrl: false,
            mod_super: false,
        }))
    }

    pub fn pressed(&self, key: KeyCode) -> bool {
        self.data.get(&key) == Some(&InputState::Pressed)
    }
    pub fn released(&self, key: KeyCode) -> bool {
        self.data.get(&key) == Some(&InputState::Released)
    }
    pub fn held(&self, key: KeyCode) -> bool {
        self.data.get(&key) == Some(&InputState::Held)
    }
    pub fn stayed_up(&self, key: KeyCode) -> bool {
        !self.data.contains_key(&key)
    }
    pub fn down(&self, key: KeyCode) -> bool {
        self.pressed(key) || self.held(key)
    }
    pub fn up(&self, key: KeyCode) -> bool {
        self.released(key) || self.stayed_up(key)
    }

    pub fn mod_shift(&self) -> bool { self.mod_shift }
    pub fn mod_alt(&self) -> bool { self.mod_alt }
    pub fn mod_ctrl(&self) -> bool { self.mod_ctrl }
    pub fn mod_super(&self) -> bool { self.mod_super }

    pub(crate) fn set_mouse_pos(&mut self, pos: Vec2) { self.mouse_pos = pos; }
    pub(crate) fn set_viewport(&mut self, viewport: AdjustedViewport) { self.viewport = viewport; }
    pub fn screen_mouse_pos(&self) -> Vec2 { self.mouse_pos / self.viewport.gui_scale_factor() }

    pub(crate) fn queue_event(&mut self, key: KeyCode, state: ElementState) {
        if key == KeyCode::ShiftLeft || key == KeyCode::ShiftRight {
            self.mod_shift = state == ElementState::Pressed;
        }
        if key == KeyCode::AltLeft || key == KeyCode::AltRight {
            self.mod_alt = state == ElementState::Pressed;
        }
        if key == KeyCode::ControlLeft || key == KeyCode::ControlRight {
            self.mod_ctrl = state == ElementState::Pressed;
        }
        if key == KeyCode::SuperLeft || key == KeyCode::SuperRight {
            self.mod_super = state == ElementState::Pressed;
        }
        self.queued_events.push((key, state));
    }

    pub(crate) fn update_step(&mut self) {
        self.data = self.data.iter()
            .filter_map(|(key, state)| match state {
                InputState::Pressed | InputState::Held => Some((*key, InputState::Held)),
                InputState::Released => None,
            })
            .collect();
        for (key, state) in self.queued_events.drain(..) {
            match self.data.get(&key) {
                None => {
                    // I don't really understand this, but some OS stuff can cause a Released state
                    // here.
                    self.data.insert(key, match state {
                        ElementState::Pressed => InputState::Pressed,
                        ElementState::Released => InputState::Released,
                    });
                }
                Some(InputState::Pressed | InputState::Held) => {
                    match state {
                        ElementState::Pressed => {},
                        ElementState::Released => {
                            self.data.insert(key, InputState::Released);
                        }
                    }
                }
                Some(InputState::Released) => {
                    match state {
                        ElementState::Pressed => {
                            self.data.insert(key, InputState::Pressed);
                        },
                        ElementState::Released => {}
                    }
                }
            }
        }
    }
}
