use egui::PointerButton;
use egui_winit::winit::event::ElementState;
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

use crate::core::prelude::*;
use crate::core::tulivuori::GgViewport;
pub use egui_winit::winit::keyboard::KeyCode;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MouseButton {
    Primary,
    Secondary,
    Middle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InputState {
    Pressed,
    Held,
    Released,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InputEvent {
    Key(KeyCode, ElementState),
    Mouse(MouseButton, ElementState),
    MouseDoubleClick(MouseButton),
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct MouseButtonState {
    inner: Option<InputState>,
    double_clicked: bool,
}

#[allow(clippy::struct_excessive_bools)]
#[derive(Clone)]
pub struct InputHandler {
    data: BTreeMap<KeyCode, InputState>,
    primary_mouse: MouseButtonState,
    secondary_mouse: MouseButtonState,
    middle_mouse: MouseButtonState,
    queued_events: Vec<InputEvent>,
    mouse_pos: Option<Vec2>,
    viewport: Option<GgViewport>,
    mod_shift: bool,
    mod_alt: bool,
    mod_ctrl: bool,
    mod_super: bool,
}

impl InputHandler {
    pub fn new() -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(InputHandler {
            data: BTreeMap::new(),
            primary_mouse: MouseButtonState::default(),
            secondary_mouse: MouseButtonState::default(),
            middle_mouse: MouseButtonState::default(),
            queued_events: Vec::new(),
            mouse_pos: None,
            viewport: None,
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

    pub fn arrows_as_joystick(&self) -> Vec2 {
        let mut dir = Vec2::zero();
        if self.down(KeyCode::ArrowRight) {
            dir += Vec2::right();
        }
        if self.down(KeyCode::ArrowLeft) {
            dir += Vec2::left();
        }
        if self.down(KeyCode::ArrowUp) {
            dir += Vec2::up();
        }
        if self.down(KeyCode::ArrowDown) {
            dir += Vec2::down();
        }
        dir.normed()
    }

    pub fn mouse_pressed(&self, button: MouseButton) -> bool {
        match button {
            MouseButton::Primary => self.primary_mouse.inner == Some(InputState::Pressed),
            MouseButton::Secondary => self.secondary_mouse.inner == Some(InputState::Pressed),
            MouseButton::Middle => self.middle_mouse.inner == Some(InputState::Pressed),
        }
    }
    pub fn mouse_released(&self, button: MouseButton) -> bool {
        match button {
            MouseButton::Primary => self.primary_mouse.inner == Some(InputState::Released),
            MouseButton::Secondary => self.secondary_mouse.inner == Some(InputState::Released),
            MouseButton::Middle => self.middle_mouse.inner == Some(InputState::Released),
        }
    }
    pub fn mouse_down(&self, button: MouseButton) -> bool {
        match button {
            MouseButton::Primary => self.primary_mouse,
            MouseButton::Secondary => self.secondary_mouse,
            MouseButton::Middle => self.middle_mouse,
        }
        .inner
        .is_some()
    }
    pub fn mouse_double_clicked(&self, button: MouseButton) -> bool {
        match button {
            MouseButton::Primary => self.primary_mouse.double_clicked,
            MouseButton::Secondary => self.secondary_mouse.double_clicked,
            MouseButton::Middle => self.middle_mouse.double_clicked,
        }
    }

    pub fn mod_shift(&self) -> bool {
        self.mod_shift
    }
    pub fn mod_alt(&self) -> bool {
        self.mod_alt
    }
    pub fn mod_ctrl(&self) -> bool {
        self.mod_ctrl
    }
    pub fn mod_super(&self) -> bool {
        self.mod_super
    }

    #[allow(unused)]
    pub(crate) fn update_mouse(&mut self, ctx: &egui::Context) {
        self.mouse_pos = ctx.pointer_latest_pos().map(|p| Vec2 { x: p.x, y: p.y });
        ctx.input(|input| {
            if input.pointer.primary_pressed() {
                self.queued_events.push(InputEvent::Mouse(
                    MouseButton::Primary,
                    ElementState::Pressed,
                ));
            } else if input.pointer.primary_released() {
                self.queued_events.push(InputEvent::Mouse(
                    MouseButton::Primary,
                    ElementState::Released,
                ));
            }
            if input.pointer.button_double_clicked(PointerButton::Primary) {
                self.queued_events
                    .push(InputEvent::MouseDoubleClick(MouseButton::Primary));
            }

            if input.pointer.secondary_pressed() {
                self.queued_events.push(InputEvent::Mouse(
                    MouseButton::Secondary,
                    ElementState::Pressed,
                ));
            } else if input.pointer.secondary_released() {
                self.queued_events.push(InputEvent::Mouse(
                    MouseButton::Secondary,
                    ElementState::Released,
                ));
            }
            if input
                .pointer
                .button_double_clicked(PointerButton::Secondary)
            {
                self.queued_events
                    .push(InputEvent::MouseDoubleClick(MouseButton::Secondary));
            }

            if input.pointer.button_pressed(PointerButton::Middle) {
                self.queued_events.push(InputEvent::Mouse(
                    MouseButton::Middle,
                    ElementState::Pressed,
                ));
            } else if input.pointer.button_released(PointerButton::Middle) {
                self.queued_events.push(InputEvent::Mouse(
                    MouseButton::Middle,
                    ElementState::Released,
                ));
            }
            if input.pointer.button_double_clicked(PointerButton::Middle) {
                self.queued_events
                    .push(InputEvent::MouseDoubleClick(MouseButton::Middle));
            }
        });
    }
    #[allow(unused)]
    pub(crate) fn set_viewport(&mut self, viewport: GgViewport) {
        self.viewport = Some(viewport);
    }
    pub fn screen_mouse_pos(&self) -> Option<Vec2> {
        self.mouse_pos
            .and_then(|p| self.viewport.as_ref().map(|v| p / v.extra_scale_factor()))
    }

    pub(crate) fn queue_key_event(&mut self, key: KeyCode, state: ElementState) {
        self.queued_events.push(InputEvent::Key(key, state));
    }

    pub(crate) fn complete_update(&mut self) {
        self.data = self
            .data
            .iter()
            .filter_map(|(key, state)| match state {
                InputState::Pressed | InputState::Held => Some((*key, InputState::Held)),
                InputState::Released => None,
            })
            .collect();
        for state in [
            &mut self.primary_mouse,
            &mut self.secondary_mouse,
            &mut self.middle_mouse,
        ] {
            state.inner = match state.inner {
                Some(InputState::Pressed | InputState::Held) => Some(InputState::Held),
                Some(InputState::Released) | None => None,
            };
            state.double_clicked = false;
        }
        // TODO: drain queued_events more intelligently -- a press and release within a single
        //  frame, e.g. when update() is very far behind, should still register as a press and
        //  (on the next frame) a release.
        let events = self.queued_events.drain(..).collect::<Vec<_>>();
        for event in events {
            match event {
                InputEvent::Key(key, state) => {
                    self.update_modifiers(key, state);
                    match self.data.get(&key) {
                        None => {
                            self.data.insert(
                                key,
                                match state {
                                    ElementState::Pressed => InputState::Pressed,
                                    // I don't really understand this, but some OS stuff can cause
                                    // a Released state here.
                                    ElementState::Released => InputState::Released,
                                },
                            );
                        }
                        Some(InputState::Pressed | InputState::Held) => match state {
                            ElementState::Pressed => {}
                            ElementState::Released => {
                                self.data.insert(key, InputState::Released);
                            }
                        },
                        Some(InputState::Released) => match state {
                            ElementState::Pressed => {
                                self.data.insert(key, InputState::Pressed);
                            }
                            ElementState::Released => {}
                        },
                    }
                }
                InputEvent::Mouse(button, state) => {
                    let stored_button = match button {
                        MouseButton::Primary => &mut self.primary_mouse,
                        MouseButton::Secondary => &mut self.secondary_mouse,
                        MouseButton::Middle => &mut self.middle_mouse,
                    };
                    stored_button.inner = match state {
                        ElementState::Pressed => Some(InputState::Pressed),
                        ElementState::Released => Some(InputState::Released),
                    };
                }
                InputEvent::MouseDoubleClick(button) => {
                    let stored_button = match button {
                        MouseButton::Primary => &mut self.primary_mouse,
                        MouseButton::Secondary => &mut self.secondary_mouse,
                        MouseButton::Middle => &mut self.middle_mouse,
                    };
                    stored_button.double_clicked = true;
                }
            }
        }
    }

    fn update_modifiers(&mut self, key: KeyCode, state: ElementState) {
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
    }
}
