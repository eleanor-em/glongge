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
#[derive(Clone)]
pub struct GuiContext {
    pub(crate) inner: egui::Context,
    ever_enabled: Arc<AtomicBool>,
}

impl GuiContext {
    pub(crate) fn new() -> Self {
        Self {
            inner: egui::Context::default(),
            ever_enabled: Arc::new(AtomicBool::new(false)),
        }
    }

    pub(crate) fn mark_enabled(&self) {
        if self
            .ever_enabled
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            info!("enable debug GUI");
        }
    }
    pub(crate) fn is_ever_enabled(&self) -> bool {
        self.ever_enabled.load(Ordering::SeqCst)
    }
}

pub type GuiUi = egui::Ui;

/// A struct for managing editable values in the GUI.
/// Provides thread-safe communication and state management for editable fields.
/// - Handles text editing state and validation
/// - Manages drag operations for numeric inputs
///
/// See examples in [`on_gui()`](crate::core::scene::GuiObject::on_gui).
pub struct EditCell<T> {
    text: Arc<Mutex<String>>,
    editing: Arc<AtomicBool>,
    // Only used to set interactivity of the text box to avoid jank when dragging (= scrolling with
    // the mouse wheel).
    dragging: Arc<AtomicBool>,
    is_valid: Arc<AtomicBool>,
    last_value: T,
    tx: Sender<String>,
    rx: Receiver<String>,
}

/// Sender component for [`EditCell`], to be used within the
/// [`GuiCommand`](crate::core::scene::GuiCommand) closure.
/// See examples in [`on_gui()`](crate::core::scene::GuiObject::on_gui).
#[derive(Clone)]
pub struct EditCellSender<T> {
    text: Arc<Mutex<String>>,
    editing: Arc<AtomicBool>,
    dragging: Arc<AtomicBool>,
    is_valid: Arc<AtomicBool>,
    tx: Sender<String>,
    ty: PhantomData<T>,
}

impl<T: Clone + Default + Display + FromStr> EditCell<T> {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            text: Arc::new(Mutex::new(String::new())),
            editing: Arc::new(AtomicBool::new(false)),
            dragging: Arc::new(AtomicBool::new(false)),
            is_valid: Arc::new(AtomicBool::new(true)),
            last_value: T::default(),
            tx,
            rx,
        }
    }

    /// Updates the cell's text and last value only if we're not currently editing.
    /// "Live" refers to the actual value being used by the game, e.g. the actual bounding box in a
    /// collider.
    pub fn update_live(&mut self, live_value: T) {
        if !self.editing.load(Ordering::SeqCst) && self.is_valid.load(Ordering::SeqCst) {
            *self.text.lock().unwrap() = live_value.to_string();
            self.last_value = live_value;
        }
    }

    pub fn try_recv(&mut self) -> Option<T> {
        // Note: no need to update `self.text` explicitly, this is done by egui.
        self.rx.try_iter().last().and_then(|s| s.parse().ok())
    }

    /// Used to forcibly clear the state, e.g. because the selected object changed.
    pub fn clear_state(&mut self) {
        self.editing.store(false, Ordering::SeqCst);
        self.dragging.store(false, Ordering::SeqCst);
        self.is_valid.store(true, Ordering::SeqCst);
    }

    pub fn last_value(&self) -> T {
        self.last_value.clone()
    }

    pub fn sender(&self) -> EditCellSender<T> {
        EditCellSender {
            text: self.text.clone(),
            editing: self.editing.clone(),
            dragging: self.dragging.clone(),
            is_valid: self.is_valid.clone(),
            tx: self.tx.clone(),
            ty: PhantomData,
        }
    }
}
impl<T: Clone + Default + Display + FromStr> Default for EditCell<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone + Default + Display + FromStr> EditCellSender<T> {
    /// Returns whether the last text input was valid for type T.
    /// Returns true by default, and false after losing focus if the text couldn't be parsed.
    pub fn is_valid(&self) -> bool {
        self.is_valid.load(Ordering::SeqCst)
    }

    pub fn singleline(&mut self, ui: &mut GuiUi, label: impl Into<WidgetText>) -> Response {
        ui.with_layout(*ui.layout(), |ui| {
            ui.add(egui::Label::new(label).selectable(false));
            let col = if self.is_valid() {
                Color32::from_gray(240)
            } else {
                Color32::from_rgb(245, 0, 0)
            };
            let response = egui::TextEdit::singleline(&mut *self.text.lock().unwrap())
                .text_color(col)
                .interactive(!self.dragging.load(Ordering::SeqCst))
                .desired_width(f32::INFINITY)
                .show(ui)
                .response;
            if response.gained_focus() {
                self.editing.store(true, Ordering::SeqCst);
            }
            if response.lost_focus() {
                self.editing.store(false, Ordering::SeqCst);
                let text = self.text.lock().unwrap();
                self.tx.send(text.clone()).unwrap();
                self.is_valid
                    .store(text.parse::<T>().is_ok(), Ordering::SeqCst);
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
        self.dragging.store(response.dragged(), Ordering::SeqCst);
        if let Some(dy) = Self::get_zoom(&response, ui) {
            let mut text = self.text.lock().unwrap();
            if let Ok(mut value) = text.parse::<f32>() {
                value += dy * drag_speed;
                *text = format!("{value:.1}");
                if self.tx.send(text.clone()).is_err() {
                    // I think this occurred at some point while testing, hence why this is not
                    // .unwrap().
                    error!("EditCellSender: send failed (scene ended?)");
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
    centre_x: EditCell<f32>,
    centre_y: EditCell<f32>,
    rotation: EditCell<f32>,
    scale_x: EditCell<f32>,
    scale_y: EditCell<f32>,
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

    pub fn update_live(&mut self, transform: Transform) {
        self.centre_x.update_live(transform.centre.x);
        self.centre_y.update_live(transform.centre.y);
        // Have to store self.rotation in degrees, because it's what the user sees.
        self.rotation.update_live(transform.rotation.to_degrees());
        self.scale_x.update_live(transform.scale.x);
        self.scale_y.update_live(transform.scale.y);
    }

    pub fn try_recv(&mut self) -> Option<Transform> {
        let mut rv = self.last_value();
        let mut changed = false;
        if let Some(centre_x) = self.centre_x.try_recv() {
            rv.centre.x = centre_x;
            changed = true;
        }
        if let Some(centre_y) = self.centre_y.try_recv() {
            rv.centre.y = centre_y;
            changed = true;
        }
        if let Some(rotation) = self.rotation.try_recv() {
            rv.rotation = rotation.to_radians();
            changed = true;
        }
        if let Some(scale_x) = self.scale_x.try_recv() {
            rv.scale.x = scale_x;
            changed = true;
        }
        if let Some(scale_y) = self.scale_y.try_recv() {
            rv.scale.y = scale_y;
            changed = true;
        }
        if changed { Some(rv) } else { None }
    }

    pub fn clear_state(&mut self) {
        self.centre_x.clear_state();
        self.centre_y.clear_state();
        self.rotation.clear_state();
        self.scale_x.clear_state();
        self.scale_y.clear_state();
    }

    pub fn last_value(&self) -> Transform {
        Transform {
            centre: Vec2 {
                x: self.centre_x.last_value,
                y: self.centre_y.last_value,
            },
            // Have to store self.rotation in degrees, because it's what the user sees.
            rotation: self.rotation.last_value.to_radians(),
            scale: Vec2 {
                x: self.scale_x.last_value,
                y: self.scale_y.last_value,
            },
        }
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
        // TODO: document these, change the cursor maybe?
        let scroll_factor = ui.input(|i| {
            if i.modifiers.shift {
                3.0
            } else if i.modifiers.alt {
                0.2
            } else {
                1.0
            }
        });
        egui::Grid::new(ui.next_auto_id())
            .num_columns(2)
            .show(ui, |ui| {
                ui.add(egui::Label::new("Centre").selectable(false));
                self.centre
                    .build_gui(ui, scroll_factor, cell.centre_x, cell.centre_y);
                ui.end_row();
                cell.rotation
                    .singleline_with_drag(ui, -scroll_factor, "Rotation: ");
                ui.end_row();
                ui.add(egui::Label::new("Scale").selectable(false));
                self.scale
                    .build_gui(ui, 0.05 * scroll_factor, cell.scale_x, cell.scale_y);
            });
    }
}
