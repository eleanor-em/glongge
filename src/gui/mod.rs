use std::fmt::Display;
use std::str::FromStr;
use std::sync::{Arc, Mutex};
use crate::core::prelude::*;

pub mod render;
pub mod debug_gui;

pub type GuiContext = egui::Context;

pub type GuiUi = egui::Ui;

#[derive(Clone)]
struct EditCell<T> {
    live: Arc<Mutex<T>>,
    edited: Arc<Mutex<Option<String>>>,
    done: Arc<Mutex<bool>>,
    text: String,
}

impl<T: Clone + Default + Display + FromStr> EditCell<T> {
    fn new() -> Self { Self {
        live: Arc::new(Mutex::new(T::default())),
        edited: Arc::new(Mutex::new(None)),
        done: Arc::new(Mutex::new(false)),
        text: String::new(),
    } }

    fn update_live(&mut self, live_value: T) {
        let mut live = self.live.lock().unwrap();
        *live = live_value.clone();
        self.text = if let Some(edited) = self.edited.lock().unwrap().as_ref() {
            edited.clone()
        } else {
            live.to_string()
        };
    }
    fn update_edit(&mut self) {
        *self.edited.lock().unwrap() = Some(self.text.clone());
    }
    fn update_done(&mut self) { *self.done.lock().unwrap() = true; }
    fn take(&mut self) -> Option<T> {
        let mut done = self.done.lock().unwrap();
        if *done {
            *done = false;
            self.edited.lock().unwrap().take()
                .and_then(|s| s.parse().ok())
        } else {
            None
        }
    }

    fn is_valid(&self) -> bool {
        let maybe_edited = self.edited.lock().unwrap().clone();
        if let Some(edited) = maybe_edited {
            edited.parse::<T>().is_ok()
        } else {
            true
        }
    }
}

impl Vec2 {
    pub fn build_gui(&self, ui: &mut GuiUi, x: EditCell<f64>, y: EditCell<f64>) {
        egui::Grid::new(ui.next_auto_id())
            .num_columns(2)
            .show(ui, |ui| {
                ui.add(egui::Label::new("x: ").selectable(false));
                egui::TextEdit::singleline(&mut format!("{:.2}", self.x))
                    .show(ui);
                ui.end_row();
                ui.add(egui::Label::new("y: ").selectable(false));
                egui::TextEdit::singleline(&mut format!("{:.2}", self.y))
                    .show(ui);
            })
            .inner
    }
}

impl Transform {
    pub fn build_gui(&self, ui: &mut GuiUi) {
        egui::Grid::new(ui.next_auto_id())
            .num_columns(2)
            .show(ui, |ui| {
                ui.add(egui::Label::new("Centre").selectable(false));
                // self.centre.build_gui(ui);
                ui.end_row();
                ui.add(egui::Label::new("Rotation").selectable(false));
                egui::TextEdit::singleline(&mut format!("{:.2}", self.rotation.to_degrees()))
                    .show(ui);
                ui.end_row();
                ui.add(egui::Label::new("Scale").selectable(false));
                // self.scale.build_gui(ui);
                ui.end_row();
            }).inner
    }
}
