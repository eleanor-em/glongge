use std::fmt::Display;
use std::str::FromStr;
use std::sync::{Arc, Mutex};
use egui::{Color32, WidgetText};
use crate::core::prelude::*;

pub mod render;
pub mod debug_gui;

pub type GuiContext = egui::Context;

pub type GuiUi = egui::Ui;

#[derive(Clone)]
pub struct EditCell<T> {
    live: Arc<Mutex<T>>,
    edited: Arc<Mutex<Option<String>>>,
    done: Arc<Mutex<bool>>,
    text: String,
}

impl<T: Clone + Default + Display + FromStr> EditCell<T> {
    pub fn new() -> Self { Self {
        live: Arc::new(Mutex::new(T::default())),
        edited: Arc::new(Mutex::new(None)),
        done: Arc::new(Mutex::new(false)),
        text: String::new(),
    } }

    pub fn update_live(&mut self, live_value: T) {
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
    pub fn take(&mut self) -> Option<T> {
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

    fn singleline(&mut self, ui: &mut GuiUi, text: impl Into<WidgetText>) {
        ui.add(egui::Label::new(text).selectable(false));
        let col = if self.is_valid() {
            Color32::from_gray(240)
        } else {
            Color32::from_rgb(240, 0, 0)
        };
        let response = egui::TextEdit::singleline(&mut self.text)
            .text_color(col)
            .show(ui)
            .response;
        if response.gained_focus() || response.changed() {
            self.update_edit();
        }
        if response.lost_focus() {
            self.update_done();
        }
    }
}

impl Vec2 {
    pub fn build_gui(&self, ui: &mut GuiUi, mut x: EditCell<f64>, mut y: EditCell<f64>) {
        x.update_live(self.x);
        y.update_live(self.y);
        egui::Grid::new(ui.next_auto_id())
            .num_columns(2)
            .show(ui, |ui| {
                x.singleline(ui, "x: ");
                ui.end_row();
                y.singleline(ui, "y: ");
            })
            .inner
    }
}

#[derive(Clone)]
pub struct TransformCell {
    centre_x: EditCell<f64>,
    centre_y: EditCell<f64>,
    rotation: EditCell<f64>,
    scale_x: EditCell<f64>,
    scale_y: EditCell<f64>,
}

impl TransformCell {
    pub fn new() -> Self {
        Self {
            centre_x: EditCell::new(),
            centre_y: EditCell::new(),
            rotation: EditCell::new(),
            scale_x: EditCell::new(),
            scale_y: EditCell::new(),
        }
    }
    
    pub fn maybe_take(&mut self, target: &mut Transform) {
        if let Some(next) = self.centre_x.take() {
            target.centre.x = next;
        }
        if let Some(next) = self.centre_y.take() {
            target.centre.y = next;
        }
        if let Some(next) = self.rotation.take() {
            target.rotation = next.to_radians();
        }
        if let Some(next) = self.scale_x.take() {
            target.scale.x = next;
        }
        if let Some(next) = self.scale_y.take() {
            target.scale.y = next;
        }
    }
}

impl Transform {
    pub fn build_gui(&self,
                     ui: &mut GuiUi,
                     mut cell: TransformCell,
    ) {
        cell.rotation.update_live(self.rotation.to_degrees());
        egui::Grid::new(ui.next_auto_id())
            .num_columns(2)
            .show(ui, |ui| {
                ui.add(egui::Label::new("Centre").selectable(false));
                self.centre.build_gui(ui, cell.centre_x, cell.centre_y);
                ui.end_row();
                cell.rotation.singleline(ui, "Rotation: ");
                ui.end_row();
                ui.add(egui::Label::new("Scale").selectable(false));
                self.scale.build_gui(ui, cell.scale_x, cell.scale_y);
                ui.end_row();
            }).inner
    }
}
