use crate::core::prelude::{Transform, Vec2};

pub mod render;

pub type GuiContext = egui::Context;

pub type GuiUi = egui::Ui;

impl Vec2 {
    pub fn build_gui(&self, ui: &mut GuiUi) {
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
            });
    }
}

impl Transform {
    pub fn build_gui(&self, ui: &mut GuiUi) {
        egui::Grid::new(ui.next_auto_id())
            .num_columns(2)
            .show(ui, |ui| {
                ui.add(egui::Label::new("Centre").selectable(false));
                self.centre.build_gui(ui);
                ui.end_row();
                ui.add(egui::Label::new("Rotation").selectable(false));
                egui::TextEdit::singleline(&mut format!("{:.2}", self.rotation.to_degrees()))
                    .show(ui);
                ui.end_row();
                ui.add(egui::Label::new("Scale").selectable(false));
                self.scale.build_gui(ui);
                ui.end_row();
            });
    }
}
