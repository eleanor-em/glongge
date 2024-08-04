use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::rc::Rc;
use std::sync::mpsc;
use std::sync::mpsc::{Receiver, Sender};
use egui::{Align, Color32, FontSelection, Frame, Id, Layout, Style, TextFormat};
use egui::scroll_area::ScrollBarVisibility;
use egui::style::ScrollStyle;
use egui::text::LayoutJob;
use itertools::Itertools;
use tracing::warn;
use crate::core::{ObjectId, ObjectTypeEnum, SceneObjectWithId};
use crate::core::prelude::*;
use crate::core::scene::{GuiClosure, GuiInsideClosure};
use crate::core::update::debug_gui::ObjectLabel::Disambiguated;
use crate::core::update::ObjectHandler;
use crate::gui::GuiUi;

#[derive(Clone, Eq, PartialEq)]
enum ObjectLabel {
    Root,
    Unique(String, String),
    Disambiguated(String, String, usize),
}

impl ObjectLabel {
    fn name(&self) -> &str {
        match self {
            ObjectLabel::Root => "<root>",
            ObjectLabel::Unique(name, _) | Disambiguated(name, _, _) => name.as_str(),
        }
    }

    fn id_source(&self) -> String {
        match self {
            ObjectLabel::Root => "<root>".to_string(),
            ObjectLabel::Unique(name, _) => name.clone(),
            Disambiguated(name, _, count) => format!("{name} {count}"),
        }
    }

    fn set_tags(&mut self, new_tags: impl AsRef<str>) {
        match self {
            ObjectLabel::Root => {},
            ObjectLabel::Unique(_, tags) | Disambiguated(_, tags, _) => {
                *tags = new_tags.as_ref().to_string()
            },
        }
    }

    fn get_tags(&self) -> &str {
        match self {
            ObjectLabel::Root => "",
            ObjectLabel::Unique(_, tags) | Disambiguated(_, tags, _) =>  tags.as_str()
        }
    }
}

struct GuiObjectView {
    object_id: ObjectId,
}

impl GuiObjectView {
    fn new() -> Self {
        Self { object_id: ObjectId::root() }
    }

    fn update_selection<O: ObjectTypeEnum>(
        &mut self,
        object_handler: &ObjectHandler<O>,
        selected_id: ObjectId
    ) {
        if !self.object_id.is_root() {
            object_handler.get_collision_shape_mut(selected_id)
                .map(|mut c| c.hide_wireframe());
        }
        self.object_id = selected_id;
        if !selected_id.is_root() {
            object_handler.get_collision_shape_mut(selected_id)
                .map(|mut c| c.show_wireframe());
        }
    }

    fn build_closure<O: ObjectTypeEnum>(
        &mut self,
        object_handler: &ObjectHandler<O>,
        mut gui_cmds: BTreeMap<ObjectId, Box<GuiInsideClosure>>,
        frame: Frame,
        enabled: bool
    ) -> Box<GuiClosure> {
        let object_id = self.object_id;
        let mut name = None;
        let mut absolute_transform = None;
        let mut relative_transform = None;
        let mut gui_cmd = None;
        if !object_id.is_root() {
            let object = object_handler.get_object_or_panic(object_id).borrow();
            name = Some(object.name());
            absolute_transform = object_handler.absolute_transforms.get(&object_id).cloned();
            relative_transform = Some(object.transform());
            gui_cmd = gui_cmds.remove(&object_id);
        }
        Box::new(move |ctx| {
            egui::SidePanel::right(Id::new("object-view"))
                .frame(frame)
                .show_animated(ctx, enabled && !object_id.is_root(), |ui| {
                    egui::ScrollArea::vertical()
                        .show(ui, |ui| {
                            ui.vertical_centered(|ui| {
                                let mut layout_job = LayoutJob {
                                    halign: Align::Center,
                                    justify: true,
                                    ..Default::default()
                                };
                                layout_job.append(name.unwrap().as_str(), 0., TextFormat {
                                    color: Color32::from_gray(255),
                                    ..Default::default()
                                });
                                ui.add(egui::Label::new(layout_job)
                                    .selectable(false));
                                ui.separator();
                            });
                            egui::CollapsingHeader::new("Absolute transform")
                                .default_open(true)
                                .show(ui, |ui| {
                                    absolute_transform.unwrap().build_gui(ui);
                                });
                            egui::CollapsingHeader::new("Relative transform")
                                .default_open(true)
                                .show(ui, |ui| {
                                    relative_transform.unwrap().build_gui(ui);
                                });
                            if let Some(gui_cmd) = gui_cmd {
                                ui.separator();
                                gui_cmd(ui);
                            }
                        });
                });
        })
    }
}

struct GuiObjectTreeNode {
    label: ObjectLabel,
    object_id: ObjectId,
    displayed: BTreeMap<ObjectId, GuiObjectTreeNode>,
    depth: usize,
    disambiguation: Rc<RefCell<BTreeMap<String, usize>>>,
    open: bool,
    open_tx: Sender<bool>,
    open_rx: Receiver<bool>,
}

impl GuiObjectTreeNode {
    fn new() -> Self {
        let (open_tx, open_rx) = std::sync::mpsc::channel();
        Self {
            label: ObjectLabel::Root,
            object_id: ObjectId::root(),
            displayed: BTreeMap::new(),
            depth: 0,
            disambiguation: Rc::new(RefCell::new(BTreeMap::new())),
            open: false,
            open_tx, open_rx
        }
    }

    fn refresh_label<O: ObjectTypeEnum>(&mut self, object_handler: &ObjectHandler<O>) {
        if !self.object_id.is_root() {
            let mut tags = String::new();
            if object_handler.get_collision_shape(self.object_id).is_some() {
                tags += "▣ ";
            }
            if object_handler.get_sprite(self.object_id).is_some() {
                tags += "👾 ";
            }
            self.label.set_tags(tags.trim());
        }
        for child in self.displayed.values_mut() {
            child.refresh_label(object_handler);
        }
    }

    fn child<O: ObjectTypeEnum>(&self, object: &SceneObjectWithId<O>) -> Self {
        let name = object.inner.borrow().name();
        let count = *self.disambiguation.borrow_mut().entry(name.clone())
            .and_modify(|count| { *count += 1 })
            .or_default();
        let (open_tx, open_rx) = mpsc::channel();
        Self {
            label: if count > 0 {
                ObjectLabel::Disambiguated(name, String::new(), count)
            } else {
                ObjectLabel::Unique(name, String::new())
            },
            object_id: object.object_id,
            displayed: BTreeMap::new(),
            depth: self.depth + 1,
            disambiguation: self.disambiguation.clone(),
            open: false,
            open_tx, open_rx
        }
    }

    fn as_builder(&mut self, selected_id: ObjectId, selected_tx: Sender<ObjectId>) -> GuiObjectTreeBuilder {
        if let Some(next) = self.open_rx.try_iter().last() {
            self.open = next;
        }
        let selected = selected_id == self.object_id;
        GuiObjectTreeBuilder {
            label: self.label.clone(),
            object_id: self.object_id,
            displayed: self.displayed.iter_mut()
                .map(|(id, tree)| (*id, tree.as_builder(selected_id, selected_tx.clone())))
                .collect(),
            open: self.open,
            open_tx: self.open_tx.clone(),
            selected, selected_tx
        }
    }
}

pub(crate) struct GuiObjectTree {
    root: GuiObjectTreeNode,
    selected_id: ObjectId,
    selected_tx: Sender<ObjectId>,
    selected_rx: Receiver<ObjectId>,
}

impl GuiObjectTree {
    fn new() -> Self {
        let selected_id = ObjectId::root();
        let (selected_tx, selected_rx) = mpsc::channel();
        Self {
            root: GuiObjectTreeNode::new(),
            selected_id, selected_tx, selected_rx,
        }
    }

    fn build_closure(&mut self, frame: Frame, enabled: bool) -> Box<GuiClosure> {
        if let Some(next) = self.selected_rx.try_iter().last() {
            self.selected_id = next;
        }
        let mut root = self.root.as_builder(
            self.selected_id,
            self.selected_tx.clone()
        );
        Box::new(move |ctx| {
            egui::SidePanel::left(Id::new("object-tree"))
                .frame(frame)
                .show_animated(ctx, enabled, |ui| {
                    ui.spacing_mut().scroll = ScrollStyle {
                        floating: false,
                        bar_width: 5.,
                        ..Default::default()
                    };
                    egui::ScrollArea::vertical()
                        .show(ui, |ui| {
                            ui.add(egui::Label::new("Object Tree 🌳"));
                            ui.separator();
                            root.build(ui);
                        });
                });
        })
    }


    pub fn on_add_object<O: ObjectTypeEnum>(&mut self, object_handler: &ObjectHandler<O>, object: &SceneObjectWithId<O>) {
        let mut tree = &mut self.root;
        for id in object_handler.get_parent_chain_or_panic(object.object_id).into_iter().rev() {
            if tree.displayed.contains_key(&id) {
                tree = tree.displayed.get_mut(&id).unwrap();
            } else {
                let child = tree.child(object);
                tree.displayed.insert(object.object_id, child);
                return;
            }
        };
    }

    pub fn on_remove_object<O: ObjectTypeEnum>(&mut self, object_handler: &ObjectHandler<O>, removed_id: ObjectId) {
        let mut chain = object_handler.get_parent_chain_or_panic(removed_id);
        let mut tree = &mut self.root;
        let mut id = chain.pop().unwrap();
        while id != removed_id {
            if let Some(next) = tree.displayed.get_mut(&id) {
                tree = next;
            } else {
                // Orphaned object, nothing to remove
                return;
            }
            id = chain.pop().unwrap();
        }
        tree.displayed.remove(&id);
    }

    pub fn refresh_labels<O: ObjectTypeEnum>(&mut self, object_handler: &ObjectHandler<O>) {
        self.root.refresh_label(object_handler);
    }
}
struct GuiObjectTreeBuilder {
    label: ObjectLabel,
    object_id: ObjectId,
    displayed: BTreeMap<ObjectId, GuiObjectTreeBuilder>,
    open: bool,
    open_tx: Sender<bool>,
    selected: bool,
    selected_tx: Sender<ObjectId>,
}
impl GuiObjectTreeBuilder {
    fn build(&mut self, ui: &mut GuiUi) {
        if self.label == ObjectLabel::Root {
            self.displayed.values_mut().for_each(|tree| tree.build(ui));
        } else {
            ui.with_layout(Layout::right_to_left(Align::TOP), |ui| {
                let parent_max_w = ui.max_rect().width();
                let offset = ui.min_rect().left() - parent_max_w;

                let mut layout_job = LayoutJob {
                    halign: Align::Center,
                    ..Default::default()
                };
                layout_job.append(self.label.get_tags(), 0., TextFormat::default());
                ui.add(egui::Label::new(layout_job).selectable(false));

                let response = egui::CollapsingHeader::new(self.label.name())
                    .id_source(self.label.id_source())
                    .show_background(self.selected)
                    .open(Some(self.open))
                    .show(ui, |ui| {
                        ui.set_max_width(parent_max_w - ui.min_rect().left() + offset);
                        let by_name = self.displayed.values_mut()
                            .chunk_by(|tree| tree.label.name().to_string());
                        for (_, child_group) in by_name.into_iter() {
                            let mut child_group = child_group.collect_vec();
                            let max_displayed = 10;
                            child_group.iter_mut().take(max_displayed)
                                .for_each(|tree| tree.build(ui));
                            if child_group.len() > max_displayed {
                                ui.label(format!("[..{}]", child_group.len()));
                                ui.end_row();
                            }
                        }
                    });
                if response.header_response.double_clicked() || response.header_response.secondary_clicked() {
                    self.open_tx.send(!self.open).unwrap();
                }
                if response.header_response.clicked() {
                    self.selected_tx.send(self.object_id).unwrap();
                }
            });
        }
    }
}

struct GuiConsoleLog {
    log_output: Vec<String>,
    log_file: BufReader<File>,
}
impl GuiConsoleLog {
    fn new() -> Result<Self> {
        let log_file = BufReader::new(std::fs::OpenOptions::new()
            .read(true)
            .open("run.log")?);
        Ok(Self {
            log_output: Vec::new(),
            log_file,
        })
    }

    fn build_closure(&mut self, frame: Frame, enabled: bool) -> Box<GuiClosure> {
        let mut line = String::new();
        if self.log_file.read_line(&mut line).unwrap() > 0 {
            self.log_output.push(line);
        }
        let log_output = self.log_output.clone();

        Box::new(move |ctx| {
            egui::TopBottomPanel::top(Id::new("log"))
                .frame(frame)
                .default_height(150.)
                .resizable(true)
                .show_animated(ctx, enabled, |ui| {
                    egui::ScrollArea::vertical()
                        .scroll_bar_visibility(ScrollBarVisibility::AlwaysVisible)
                        .stick_to_bottom(true)
                        .show(ui, |ui| {
                            ui.separator();
                            let mut layout_job = LayoutJob::default();
                            let last_log = log_output.len() - 1;
                            for (i, line) in log_output.into_iter().enumerate() {
                                let segments = if i == last_log {
                                    line.trim()
                                } else {
                                    line.as_str()
                                }.split("\x1b[");
                                let style = Style::default();
                                for segment in segments.filter(|s| !s.is_empty()) {
                                    let sep = segment.find('m').unwrap();
                                    let colour_code = &segment[..sep];
                                    let col = colour_code.parse::<i32>().unwrap();
                                    let text = &segment[sep + 1..];
                                    egui::RichText::new(text)
                                        .color(match col {
                                            2 => Color32::from_gray(120),
                                            32 => Color32::from_rgb(0, 166, 0),
                                            33 => Color32::from_rgb(153, 153, 0),
                                            0 => Color32::from_gray(240),
                                            _ => {
                                                warn!("unrecognised colour code: {col}");
                                                Color32::from_gray(240)
                                            },
                                        })
                                        .monospace()
                                        .append_to(&mut layout_job,
                                                   &style,
                                                   FontSelection::Default,
                                                   Align::Center)
                                }
                            }
                            ui.add(egui::Label::new(layout_job)
                                .selectable(true));
                        })
                });
        })
    }
}

pub(crate) struct DebugGui {
    pub(crate) enabled: bool,
    pub(crate) object_tree: GuiObjectTree,
    object_view: GuiObjectView,
    console_log: GuiConsoleLog,
    frame: Frame,
}

impl DebugGui {
    pub fn new() -> Result<Self> {
        Ok(Self {
            enabled: false,
            object_tree: GuiObjectTree::new(),
            object_view: GuiObjectView::new(),
            console_log: GuiConsoleLog::new()?,
            frame: egui::Frame::default()
                .fill(Color32::from_rgba_unmultiplied(12, 12, 12, 245))
                .inner_margin(egui::Margin::same(6.))
        })
    }

    pub fn build<O: ObjectTypeEnum>(&mut self,
                                    object_handler: &ObjectHandler<O>,
                                    gui_cmds: BTreeMap<ObjectId, Box<GuiInsideClosure>>
    ) -> Box<GuiClosure> {
        if !self.object_tree.selected_id.is_root() {
            self.object_view.update_selection(object_handler, self.object_tree.selected_id);
        }

        let build_object_tree = self.object_tree.build_closure(self.frame.clone(), self.enabled);
        let build_object_view = self.object_view.build_closure(
            object_handler,
            gui_cmds,
            self.frame.clone(),
            self.enabled);
        let build_console_log = self.console_log.build_closure(self.frame.clone(), self.enabled);
        Box::new(move |ctx| {
            build_console_log(ctx);
            build_object_tree(ctx);
            build_object_view(ctx);
        })
    }

    pub fn toggle(&mut self) { self.enabled = !self.enabled; }
}
