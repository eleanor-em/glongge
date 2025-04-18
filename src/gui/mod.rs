use crate::core::prelude::*;
use egui::{Color32, Response, WidgetText};
use num_traits::Zero;
use std::fmt::Display;
use std::marker::PhantomData;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex, mpsc};

pub mod debug_gui;
pub mod render;

pub type GuiContext = egui::Context;

pub type GuiUi = egui::Ui;

pub struct EditCellReceiver<T> {
    text: Arc<Mutex<String>>,
    editing: Arc<AtomicBool>,
    dragging: Arc<AtomicBool>,
    last_value: T,
    tx: Sender<String>,
    rx: Receiver<String>,
}

#[derive(Clone)]
pub struct EditCellSender<T> {
    text: Arc<Mutex<String>>,
    editing: Arc<AtomicBool>,
    dragging: Arc<AtomicBool>,
    tx: Sender<String>,
    is_valid: bool,
    ty: PhantomData<T>,
}

impl<T: Clone + Default + Display + FromStr> EditCellReceiver<T> {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            text: Arc::new(Mutex::new(String::new())),
            editing: Arc::new(AtomicBool::new(false)),
            dragging: Arc::new(AtomicBool::new(false)),
            last_value: T::default(),
            tx,
            rx,
        }
    }

    pub fn update_live(&mut self, live_value: T) {
        if !self.editing.load(Ordering::Relaxed) {
            *self.text.lock().unwrap() = live_value.to_string();
            self.last_value = live_value;
        }
    }

    pub fn reset(&mut self) {
        self.editing.store(false, Ordering::Relaxed);
    }
    pub fn try_recv(&mut self) -> Option<T> {
        self.rx.try_iter().last().and_then(|s| s.parse().ok())
    }

    pub fn recv(&mut self) -> T {
        if let Some(next) = self.try_recv() {
            self.last_value = next;
        }
        self.last_value.clone()
    }

    pub fn sender(&self) -> EditCellSender<T> {
        EditCellSender {
            text: self.text.clone(),
            editing: self.editing.clone(),
            dragging: self.dragging.clone(),
            tx: self.tx.clone(),
            ty: PhantomData,
            is_valid: true,
        }
    }
}
impl<T: Clone + Default + Display + FromStr> Default for EditCellReceiver<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone + Default + Display + FromStr> EditCellSender<T> {
    pub fn is_valid(&self) -> bool {
        self.is_valid
    }

    pub fn singleline(&mut self, ui: &mut GuiUi, label: impl Into<WidgetText>) -> Response {
        ui.with_layout(*ui.layout(), |ui| {
            ui.add(egui::Label::new(label).selectable(false));
            let col = if self.is_valid() {
                Color32::from_gray(240)
            } else {
                Color32::from_rgb(240, 0, 0)
            };
            let response = egui::TextEdit::singleline(&mut *self.text.lock().unwrap())
                .text_color(col)
                .interactive(!self.dragging.load(Ordering::Relaxed))
                .desired_width(f32::INFINITY)
                .show(ui)
                .response;
            if response.gained_focus() {
                self.editing.store(true, Ordering::Relaxed);
            }
            if response.lost_focus() {
                self.editing.store(false, Ordering::Relaxed);
                let text = self.text.lock().unwrap();
                self.tx.send(text.clone()).unwrap();
                self.is_valid = text.parse::<T>().is_ok();
            }
            response
        })
        .inner
    }
}

impl EditCellSender<f32> {
    fn get_zoom(response: &Response, ui: &mut GuiUi) -> Option<f32> {
        let mut delta = response.drag_delta().y;
        if delta.is_zero() && response.hovered() {
            ui.input(|i| {
                delta = i.raw_scroll_delta.y;
            });
        }
        if delta.is_zero() { None } else { Some(delta) }
    }

    pub fn singleline_with_drag(
        &mut self,
        ui: &mut GuiUi,
        drag_speed: f32,
        label: impl Into<WidgetText>,
    ) {
        let response = self.singleline(ui, label);
        self.dragging.store(response.dragged(), Ordering::Relaxed);
        if let Some(dy) = Self::get_zoom(&response, ui) {
            let mut text = self.text.lock().unwrap();
            if let Ok(mut value) = text.parse::<f32>() {
                value += dy * drag_speed;
                *text = format!("{value:.1}");
                if self.tx.send(text.clone()).is_err() {
                    warn!("EditCellSender: send failed (scene ended?)");
                }
            }
        }
    }
}

impl Vec2 {
    pub fn build_gui(
        &self,
        ui: &mut GuiUi,
        drag_speed: f32,
        mut x: EditCellSender<f32>,
        mut y: EditCellSender<f32>,
    ) {
        egui::Grid::new(ui.next_auto_id())
            .num_columns(2)
            .show(ui, |ui| {
                x.singleline_with_drag(ui, drag_speed, "x: ");
                ui.end_row();
                y.singleline_with_drag(ui, drag_speed, "y: ");
            });
    }
}

pub struct TransformCell {
    centre_x: EditCellReceiver<f32>,
    centre_y: EditCellReceiver<f32>,
    rotation: EditCellReceiver<f32>,
    scale_x: EditCellReceiver<f32>,
    scale_y: EditCellReceiver<f32>,
}

impl TransformCell {
    pub fn new() -> Self {
        Self {
            centre_x: EditCellReceiver::new(),
            centre_y: EditCellReceiver::new(),
            rotation: EditCellReceiver::new(),
            scale_x: EditCellReceiver::new(),
            scale_y: EditCellReceiver::new(),
        }
    }
    pub fn recv(&mut self) -> Transform {
        Transform {
            centre: Vec2 {
                x: self.centre_x.recv(),
                y: self.centre_y.recv(),
            },
            rotation: self.rotation.recv().to_radians(),
            scale: Vec2 {
                x: self.scale_x.recv(),
                y: self.scale_y.recv(),
            },
        }
    }
    pub fn update_live(&mut self, transform: Transform) {
        self.centre_x.update_live(transform.centre.x);
        self.centre_y.update_live(transform.centre.y);
        self.rotation.update_live(transform.rotation.to_degrees());
        self.scale_x.update_live(transform.scale.x);
        self.scale_y.update_live(transform.scale.y);
    }

    pub fn reset(&mut self) {
        self.centre_x.reset();
        self.centre_y.reset();
        self.rotation.reset();
        self.scale_x.reset();
        self.scale_y.reset();
    }

    pub fn sender(&self) -> TransformCellSender {
        TransformCellSender {
            centre_x: self.centre_x.sender(),
            centre_y: self.centre_y.sender(),
            rotation: self.rotation.sender(),
            scale_x: self.scale_x.sender(),
            scale_y: self.scale_y.sender(),
        }
    }
}

impl Default for TransformCell {
    fn default() -> Self {
        Self::new()
    }
}

pub struct TransformCellSender {
    centre_x: EditCellSender<f32>,
    centre_y: EditCellSender<f32>,
    rotation: EditCellSender<f32>,
    scale_x: EditCellSender<f32>,
    scale_y: EditCellSender<f32>,
}

impl Transform {
    pub fn build_gui(&self, ui: &mut GuiUi, mut cell: TransformCellSender) {
        egui::Grid::new(ui.next_auto_id())
            .num_columns(2)
            .show(ui, |ui| {
                ui.add(egui::Label::new("Centre").selectable(false));
                self.centre.build_gui(ui, 1.0, cell.centre_x, cell.centre_y);
                ui.end_row();
                cell.rotation.singleline_with_drag(ui, -1.0, "Rotation: ");
                ui.end_row();
                ui.add(egui::Label::new("Scale").selectable(false));
                self.scale.build_gui(ui, 0.1, cell.scale_x, cell.scale_y);
            });
    }
}
