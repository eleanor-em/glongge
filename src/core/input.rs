use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};
use egui_winit::winit::event::ElementState;

pub use egui_winit::winit::keyboard::KeyCode as KeyCode;
use num_traits::Zero;
use crate::core::prelude::Vec2;
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
}

impl InputHandler {
    pub fn new() -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(InputHandler {
            data: BTreeMap::new(),
            queued_events: Vec::new(),
            mouse_pos: Vec2::zero(),
            viewport: AdjustedViewport::default(),
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

    pub(crate) fn set_mouse_pos(&mut self, pos: Vec2) { self.mouse_pos = pos; }
    pub(crate) fn set_viewport(&mut self, viewport: AdjustedViewport) { self.viewport = viewport; }
    pub fn mouse_pos(&self) -> Vec2 { self.mouse_pos / self.viewport.gui_scale_factor() }

    pub(crate) fn queue_event(&mut self, key: KeyCode, state: ElementState) {
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
