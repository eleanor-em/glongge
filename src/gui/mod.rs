use egui::text_edit::TextEditOutput;
use crate::core::prelude::{Transform, Vec2};

pub mod render;
pub mod debug_gui;

pub type GuiContext = egui::Context;

pub type GuiUi = egui::Ui;

pub struct Vec2Output {
    x: TextEditOutput,
    y: TextEditOutput,
}

impl Vec2 {
    pub fn build_gui(&self, ui: &mut GuiUi) -> Vec2Output {
        egui::Grid::new(ui.next_auto_id())
            .num_columns(2)
            .show(ui, |ui| {
                ui.add(egui::Label::new("x: ").selectable(false));
                let x = egui::TextEdit::singleline(&mut format!("{:.2}", self.x))
                    .show(ui);
                ui.end_row();
                ui.add(egui::Label::new("y: ").selectable(false));
                let y = egui::TextEdit::singleline(&mut format!("{:.2}", self.y))
                    .show(ui);
                Vec2Output { x, y }
            })
            .inner
    }
}

pub struct TransformOutput {
    centre: Vec2Output,
    rotation: TextEditOutput,
    scale: Vec2Output,
}

impl Transform {
    pub fn build_gui(&self, ui: &mut GuiUi) -> TransformOutput {
        egui::Grid::new(ui.next_auto_id())
            .num_columns(2)
            .show(ui, |ui| {
                ui.add(egui::Label::new("Centre").selectable(false));
                let centre = self.centre.build_gui(ui);
                ui.end_row();
                ui.add(egui::Label::new("Rotation").selectable(false));
                let rotation = egui::TextEdit::singleline(&mut format!("{:.2}", self.rotation.to_degrees()))
                    .show(ui);
                ui.end_row();
                ui.add(egui::Label::new("Scale").selectable(false));
                let scale = self.scale.build_gui(ui);
                ui.end_row();
                TransformOutput { centre, rotation, scale }
            }).inner
    }
}
