use crate::core::input::InputHandler;
use crate::core::prelude::*;
use crate::core::scene::{GuiClosure, GuiInsideClosure};
use crate::core::update::collision::Collision;
use crate::core::update::{ObjectHandler, UpdatePerfStats};
use crate::core::vk::{AdjustedViewport, RenderPerfStats};
use crate::core::{ObjectId, ObjectTypeEnum, SceneObjectWithId};
use crate::gui::{GuiUi, TransformCell};
use crate::util::{gg_err, gg_float, gg_iter, NonemptyVec};
use egui::style::ScrollStyle;
use egui::text::LayoutJob;
use egui::{
    Align, Button, Color32, FontSelection, Frame, Id, Layout, Style, TextFormat, TextStyle, Ui,
};
use itertools::Itertools;
use num_traits::Zero;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::rc::Rc;
use std::sync::mpsc;
use std::sync::mpsc::{Receiver, Sender};
use std::time::Instant;
use tracing::warn;

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
            ObjectLabel::Unique(name, _) | ObjectLabel::Disambiguated(name, _, _) => name.as_str(),
        }
    }

    fn id_salt(&self) -> String {
        match self {
            ObjectLabel::Root => "<root>".to_string(),
            ObjectLabel::Unique(name, _) => name.clone(),
            ObjectLabel::Disambiguated(name, _, count) => format!("{name} {count}"),
        }
    }

    fn set_tags(&mut self, new_tags: impl AsRef<str>) {
        match self {
            ObjectLabel::Root => {}
            ObjectLabel::Unique(_, tags) | ObjectLabel::Disambiguated(_, tags, _) => {
                *tags = new_tags.as_ref().to_string();
            }
        }
    }

    fn get_tags(&self) -> &str {
        match self {
            ObjectLabel::Root => "",
            ObjectLabel::Unique(_, tags) | ObjectLabel::Disambiguated(_, tags, _) => tags.as_str(),
        }
    }
}

struct GuiObjectView {
    object_id: ObjectId,
    absolute_cell: TransformCell,
    relative_cell: TransformCell,
}

impl GuiObjectView {
    fn new() -> Self {
        Self {
            object_id: ObjectId::root(),
            absolute_cell: TransformCell::new(),
            relative_cell: TransformCell::new(),
        }
    }

    fn clear_selection(&mut self) {
        self.absolute_cell.reset();
        self.relative_cell.reset();
        self.object_id = ObjectId::root();
    }

    fn update_selection<O: ObjectTypeEnum>(
        &mut self,
        object_handler: &ObjectHandler<O>,
        selected_id: ObjectId,
    ) -> Result<()> {
        if self.object_id != selected_id {
            if let Some(mut c) = object_handler.get_collision_shape_mut(self.object_id)? {
                c.hide_wireframe();
            }
            self.absolute_cell.reset();
            self.relative_cell.reset();
            self.object_id = selected_id;
            if let Some(mut c) = object_handler.get_collision_shape_mut(selected_id)? {
                c.show_wireframe();
            }
        }
        Ok(())
    }

    fn build_closure<O: ObjectTypeEnum>(
        &mut self,
        object_handler: &mut ObjectHandler<O>,
        mut gui_cmds: BTreeMap<ObjectId, Box<GuiInsideClosure>>,
        frame: Frame,
        enabled: bool,
    ) -> Result<Box<GuiClosure>> {
        let object_id = self.object_id;
        if object_id.is_root() {
            return Ok(Box::new(|_| {}));
        }

        let mut absolute_transform = object_handler
            .absolute_transforms
            .get(&object_id)
            .copied()
            .with_context(|| format!("missing object_id in absolute_transforms: {object_id:?}"))?;
        let object = gg_err::log_err_then(object_handler.get_object_mut(object_id))
            .with_context(|| {
                self.clear_selection();
                format!("!object_id.is_root() but object_handler.get_object(object_id) returned None: {object_id:?}")
            })?;
        let name = object.name();
        let gui_cmd = gui_cmds.remove(&object_id);

        self.absolute_cell.update_live(absolute_transform);
        {
            let next = self.absolute_cell.recv();
            let dx = next.centre - absolute_transform.centre;
            let dt = next.rotation - absolute_transform.rotation;
            let ds = next.scale - absolute_transform.scale;
            let mut transform = object.transform_mut();
            transform.centre += dx;
            transform.rotation += dt;
            transform.scale += ds;
            absolute_transform = next;
        }

        let relative_transform = object.transform();
        self.relative_cell.update_live(relative_transform);
        let next = self.relative_cell.recv();
        *object.transform_mut() = next;

        let absolute_sender = self.absolute_cell.sender();
        let relative_sender = self.relative_cell.sender();

        Ok(Box::new(move |ctx| {
            egui::SidePanel::right(Id::new("object-view"))
                .frame(frame)
                .show_animated(ctx, enabled && !object_id.is_root(), |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        ui.vertical_centered(|ui| {
                            let mut layout_job = LayoutJob::default();
                            layout_job.append(
                                format!("{name} [{}]", object_id.0).as_str(),
                                0.,
                                TextFormat {
                                    color: Color32::from_white_alpha(255),
                                    ..Default::default()
                                },
                            );
                            ui.add(egui::Label::new(layout_job).selectable(false));
                            ui.separator();
                        });

                        let mut layout_job = LayoutJob::default();
                        layout_job.append(
                            "Absolute transform:",
                            0.,
                            TextFormat {
                                color: Color32::from_white_alpha(255),
                                ..Default::default()
                            },
                        );
                        ui.add(egui::Label::new(layout_job).selectable(false));
                        absolute_transform.build_gui(ui, absolute_sender);
                        ui.separator();
                        let mut layout_job = LayoutJob::default();
                        layout_job.append(
                            "Relative transform:",
                            0.,
                            TextFormat {
                                color: Color32::from_white_alpha(255),
                                ..Default::default()
                            },
                        );
                        ui.add(egui::Label::new(layout_job).selectable(false));
                        relative_transform.build_gui(ui, relative_sender);

                        if let Some(gui_cmd) = gui_cmd {
                            ui.separator();
                            gui_cmd(ui);
                        }
                    });
                });
        }))
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
            open_tx,
            open_rx,
        }
    }

    fn refresh_label<O: ObjectTypeEnum>(&mut self, object_handler: &ObjectHandler<O>) {
        if !self.object_id.is_root() {
            let mut tags = String::new();
            if gg_err::is_some_and_log(object_handler.get_collision_shape(self.object_id)) {
                tags += "▣ ";
            }
            if gg_err::is_some_and_log(object_handler.get_sprite(self.object_id)) {
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
        let count = *self
            .disambiguation
            .borrow_mut()
            .entry(name.clone())
            .and_modify(|count| *count += 1)
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
            open_tx,
            open_rx,
        }
    }

    fn update_open_with_selected(&mut self, selected_id: ObjectId) -> bool {
        let mut rv = false;
        if self.displayed.contains_key(&selected_id)
            || self
                .displayed
                .values_mut()
                .any(|c| c.update_open_with_selected(selected_id))
        {
            self.open = true;
            rv = true;
        }
        rv
    }

    fn as_builder(
        &mut self,
        selected_changed: bool,
        selected_id: ObjectId,
        selected_tx: Sender<ObjectId>,
    ) -> GuiObjectTreeBuilder {
        if let Some(next) = self.open_rx.try_iter().last() {
            self.open = next;
        }
        let selected = selected_id == self.object_id;
        GuiObjectTreeBuilder {
            label: self.label.clone(),
            object_id: self.object_id,
            displayed: self
                .displayed
                .iter_mut()
                .map(|(id, tree)| {
                    (
                        *id,
                        tree.as_builder(selected_changed, selected_id, selected_tx.clone()),
                    )
                })
                .collect(),
            open: self.open,
            open_tx: self.open_tx.clone(),
            selected_changed,
            selected,
            selected_tx,
        }
    }
}

pub(crate) struct GuiObjectTree {
    root: GuiObjectTreeNode,
    show: bool,
    show_tx: Sender<bool>,
    show_rx: Receiver<bool>,

    selected_id: ObjectId,
    selected_tx: Sender<ObjectId>,
    selected_rx: Receiver<ObjectId>,
}

impl GuiObjectTree {
    fn new() -> Self {
        let (show_tx, show_rx) = mpsc::channel();
        let selected_id = ObjectId::root();
        let (selected_tx, selected_rx) = mpsc::channel();
        Self {
            root: GuiObjectTreeNode::new(),
            show: true,
            show_tx,
            show_rx,
            selected_id,
            selected_tx,
            selected_rx,
        }
    }

    fn build_closure(&mut self, frame: Frame, enabled: bool) -> Box<GuiClosure> {
        if let Some(next) = self.show_rx.try_iter().last() {
            self.show = next;
        }
        let show = self.show;
        let show_tx = self.show_tx.clone();
        let mut selected_changed = false;
        if let Some(next) = self.selected_rx.try_iter().last() {
            self.selected_id = next;
            self.root.update_open_with_selected(self.selected_id);
            selected_changed = true;
        }
        let mut root =
            self.root
                .as_builder(selected_changed, self.selected_id, self.selected_tx.clone());
        Box::new(move |ctx| {
            egui::SidePanel::left(Id::new("object-tree"))
                .resizable(true)
                .min_width(0.)
                .default_width(250.)
                .frame(frame)
                .show_animated(ctx, enabled, |ui| {
                    ui.spacing_mut().scroll = ScrollStyle {
                        floating: false,
                        bar_width: 5.,
                        ..Default::default()
                    };
                    if show {
                        ui.vertical_centered(|ui| {
                            if ui.add(Button::new("🌳").selected(show)).clicked() {
                                show_tx.send(!show).unwrap();
                            }
                            egui::ScrollArea::vertical().show(ui, |ui| {
                                let mut layout_job = LayoutJob::default();
                                layout_job.append(
                                    "Object Tree",
                                    0.,
                                    TextFormat {
                                        color: Color32::from_white_alpha(255),
                                        ..Default::default()
                                    },
                                );
                                ui.add(egui::Label::new(layout_job).selectable(false));
                                ui.separator();
                                root.build(ui);
                            });
                        });
                    } else if ui.add(Button::new("🌳").selected(show)).clicked() {
                        show_tx.send(!show).unwrap();
                    }
                });
        })
    }

    fn on_input<O: ObjectTypeEnum>(
        &mut self,
        input_handler: &InputHandler,
        object_handler: &ObjectHandler<O>,
        wireframe_mouseovers: &[ObjectId],
    ) {
        if input_handler.pressed(KeyCode::KeyF) {
            if let Some(i) = gg_iter::index_of(wireframe_mouseovers, &self.selected_id) {
                let i = (i + 1) % wireframe_mouseovers.len();
                self.selected_tx.send(wireframe_mouseovers[i]).unwrap();
            } else if !wireframe_mouseovers.is_empty() {
                self.selected_tx.send(wireframe_mouseovers[0]).unwrap();
            }
        }
        if input_handler.pressed(KeyCode::KeyC) {
            if self.selected_id.is_root() {
                if let Some(object_id) = object_handler.get_first_object_id() {
                    self.selected_tx.send(object_id).unwrap();
                }
            } else if let Some(child) = gg_err::log_err_then(
                object_handler
                    .get_children(self.selected_id)
                    .map(|v| v.first()),
            ) {
                self.selected_tx.send(child.object_id).unwrap();
            } else {
                self.select_next_sibling(object_handler);
            }
        }
        if !input_handler.mod_super() && input_handler.pressed(KeyCode::KeyS) {
            self.select_next_sibling(object_handler);
        }
        if !input_handler.mod_super() && input_handler.pressed(KeyCode::KeyP) {
            if let Some(parent) = gg_err::log_err_then(object_handler.get_parent(self.selected_id))
            {
                self.selected_tx.send(parent.object_id).unwrap();
            }
        }
        if input_handler.pressed(KeyCode::KeyO) {
            self.show = !self.show;
        }
    }

    fn select_next_sibling<O: ObjectTypeEnum>(&mut self, object_handler: &ObjectHandler<O>) {
        if let Some(sibling_id) = gg_err::log_err_then(object_handler.get_parent(self.selected_id))
            .and_then(|parent| {
                let siblings = gg_err::log_unwrap_or(
                    &Vec::new(),
                    object_handler.get_children(parent.object_id),
                )
                .iter()
                .map(|o| o.object_id)
                .collect_vec();
                gg_iter::index_of(&siblings, &self.selected_id)
                    .map(|i| siblings[(i + 1) % siblings.len()])
            })
        {
            self.selected_tx.send(sibling_id).unwrap();
        } else if let Some(sibling_id) = object_handler.get_next_object_id(self.selected_id) {
            self.selected_tx.send(sibling_id).unwrap();
        }
    }

    pub fn on_add_object<O: ObjectTypeEnum>(
        &mut self,
        object_handler: &ObjectHandler<O>,
        object: &SceneObjectWithId<O>,
    ) {
        let mut tree = &mut self.root;
        match object_handler.get_parent_chain(object.object_id) {
            Ok(chain) => {
                for id in chain.into_iter().rev() {
                    if tree.displayed.contains_key(&id) {
                        tree = tree.displayed.get_mut(&id).unwrap();
                    } else {
                        let child = tree.child(object);
                        tree.displayed.insert(object.object_id, child);
                        return;
                    }
                }
            }
            Err(e) => error!("{}", e.root_cause()),
        }
    }

    pub fn on_remove_object<O: ObjectTypeEnum>(
        &mut self,
        object_handler: &ObjectHandler<O>,
        removed_id: ObjectId,
    ) {
        match object_handler.get_parent_chain(removed_id) {
            Ok(mut chain) => {
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
            Err(e) => error!("{}", e.root_cause()),
        }
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
    selected_changed: bool,
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
                    .id_salt(self.label.id_salt())
                    .show_background(self.selected)
                    .open(Some(self.open))
                    .show(ui, |ui| {
                        ui.set_max_width(parent_max_w - ui.min_rect().left() + offset);
                        for (_, child_group) in &self
                            .displayed
                            .values_mut()
                            .chunk_by(|tree| tree.label.name().to_string())
                        {
                            let mut child_group = child_group.collect_vec();
                            let max_displayed = 10;
                            child_group
                                .iter_mut()
                                .take(max_displayed)
                                .for_each(|tree| tree.build(ui));
                            if child_group.len() > max_displayed {
                                ui.label(format!("[..{}]", child_group.len()));
                                ui.end_row();
                            }
                        }
                    });
                if self.selected_changed && self.selected {
                    response.header_response.scroll_to_me(Some(Align::Center));
                }
                if response.header_response.double_clicked()
                    || response.header_response.secondary_clicked()
                {
                    self.open_tx.send(!self.open).unwrap();
                }
                if response.header_response.clicked() {
                    self.selected_tx.send(self.object_id).unwrap();
                }
            });
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
enum ViewPerfMode {
    Update,
    Render,
    None,
}

pub(crate) struct GuiConsoleLog {
    log_output: Vec<String>,
    log_file: BufReader<File>,
    view_perf: ViewPerfMode,
    view_perf_tx: Sender<ViewPerfMode>,
    view_perf_rx: Receiver<ViewPerfMode>,
    update_perf_stats: Option<UpdatePerfStats>,
    render_perf_stats: Option<RenderPerfStats>,
}
impl GuiConsoleLog {
    const MAX_LOG_LINES: usize = 40;

    fn new() -> Result<Self> {
        let log_file = BufReader::new(std::fs::OpenOptions::new().read(true).open("run.log")?);
        let (view_perf_tx, view_perf_rx) = mpsc::channel();
        Ok(Self {
            log_output: Vec::new(),
            log_file,
            view_perf: ViewPerfMode::None,
            view_perf_tx,
            view_perf_rx,
            update_perf_stats: None,
            render_perf_stats: None,
        })
    }

    pub(crate) fn update_perf_stats(&mut self, stats: Option<UpdatePerfStats>) {
        self.update_perf_stats = stats;
    }
    pub(crate) fn render_perf_stats(&mut self, stats: Option<RenderPerfStats>) {
        self.render_perf_stats = stats;
    }

    fn transform_log_line(line: &str) -> String {
        let re = Regex::new("((?:INFO|WARN|ERROR)\x1b\\[0m \x1b\\[2m).*?(glongge(?:\\/|\\\\)src(?:\\/|\\\\))").unwrap();
        re.replace_all(line, "$1$2").take()
    }

    fn build_closure(&mut self, frame: Frame, enabled: bool) -> Box<GuiClosure> {
        let mut line = String::new();
        if self.log_file.read_line(&mut line).unwrap() > 0 {
            self.log_output.push(Self::transform_log_line(line));
            if self.log_output.len() > Self::MAX_LOG_LINES {
                self.log_output.remove(0);
            }
        }
        let log_output = self.log_output.clone();
        if let Some(next) = self.view_perf_rx.try_iter().last() {
            self.view_perf = next;
        }
        let view_perf = self.view_perf;
        let view_perf_tx = self.view_perf_tx.clone();
        let update_perf_stats = self.update_perf_stats.clone();
        let render_perf_stats = self.render_perf_stats.clone();

        Box::new(move |ctx| {
            egui::TopBottomPanel::top(Id::new("log"))
                .frame(frame)
                .default_height(180.)
                .resizable(true)
                .show_animated(ctx, enabled, |ui| {
                    ui.with_layout(egui::Layout::right_to_left(Align::TOP), |ui| {
                        if ui
                            .add(egui::Button::new("🖥").selected(view_perf == ViewPerfMode::Update))
                            .clicked()
                        {
                            let next = if view_perf == ViewPerfMode::Update {
                                ViewPerfMode::None
                            } else {
                                ViewPerfMode::Update
                            };
                            view_perf_tx.send(next).unwrap();
                        }
                        if ui
                            .add(
                                egui::Button::new("🎨").selected(view_perf == ViewPerfMode::Render),
                            )
                            .clicked()
                        {
                            let next = if view_perf == ViewPerfMode::Render {
                                ViewPerfMode::None
                            } else {
                                ViewPerfMode::Render
                            };
                            view_perf_tx.send(next).unwrap();
                        }
                        Self::build_update_perf(ui, view_perf, frame, update_perf_stats);
                        Self::build_render_perf(ui, view_perf, frame, render_perf_stats);
                        Self::build_log_scroll_area(ui, log_output);
                    });
                });
        })
    }

    fn build_update_perf(
        ui: &mut Ui,
        view_perf: ViewPerfMode,
        frame: Frame,
        update_perf_stats: Option<UpdatePerfStats>,
    ) {
        egui::SidePanel::right("perf-update")
            .default_width(160.)
            .frame(frame)
            .show_animated_inside(ui, view_perf == ViewPerfMode::Update, |ui| {
                ui.vertical_centered(|ui| {
                    let mut layout_job = LayoutJob::default();
                    layout_job.append(
                        "Update Performance",
                        0.,
                        TextFormat {
                            color: Color32::from_white_alpha(255),
                            ..Default::default()
                        },
                    );
                    ui.add(egui::Label::new(layout_job).extend().selectable(false));
                    ui.separator();
                });
                if let Some(perf_stats) = update_perf_stats {
                    egui::Grid::new("perf-update-data")
                        .num_columns(3)
                        .show(ui, |ui| {
                            for (tag, mean, max) in perf_stats.as_tuples_ms() {
                                let is_total = tag == "total";
                                ui.add(egui::Label::new(tag).selectable(false));
                                ui.add(egui::Label::new(format!("{mean:.1}")).selectable(false));
                                ui.add(egui::Label::new(format!("{max:.1}")).selectable(false));
                                ui.end_row();
                                if is_total {
                                    ui.separator();
                                    ui.end_row();
                                }
                            }
                        });
                }
            });
    }
    fn build_render_perf(
        ui: &mut Ui,
        view_perf: ViewPerfMode,
        frame: Frame,
        render_perf_stats: Option<RenderPerfStats>,
    ) {
        egui::SidePanel::right("perf-render")
            .frame(frame)
            .show_animated_inside(ui, view_perf == ViewPerfMode::Render, |ui| {
                ui.vertical_centered(|ui| {
                    let mut layout_job = LayoutJob::default();
                    layout_job.append(
                        "Render Performance",
                        0.,
                        TextFormat {
                            color: Color32::from_white_alpha(255),
                            ..Default::default()
                        },
                    );
                    ui.add(egui::Label::new(layout_job).extend().selectable(false));
                    ui.separator();
                });
                if let Some(perf_stats) = render_perf_stats {
                    egui::Grid::new("perf-render-data")
                        .num_columns(3)
                        .show(ui, |ui| {
                            for (tag, mean, max) in perf_stats.as_tuples_ms() {
                                let is_total = tag == "total";
                                ui.add(egui::Label::new(tag).selectable(false));
                                ui.add(egui::Label::new(format!("{mean:.1}")).selectable(false));
                                ui.add(egui::Label::new(format!("{max:.1}")).selectable(false));
                                ui.end_row();
                                if is_total {
                                    ui.separator();
                                    ui.end_row();
                                }
                            }
                        });
                }
            });
    }

    fn build_log_scroll_area(ui: &mut Ui, log_output: Vec<String>) {
        ui.with_layout(egui::Layout::top_down(Align::LEFT), |ui| {
            egui::ScrollArea::both()
                .min_scrolled_height(180.)
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    // XXX: separator needed to make it fill available space.
                    ui.separator();
                    let mut layout_job = LayoutJob::default();
                    if !log_output.is_empty() {
                        let last_log = log_output.len() - 1;
                        for (i, line) in log_output.into_iter().enumerate() {
                            let segments = if i == last_log {
                                line.trim()
                            } else {
                                line.as_str()
                            }
                            .split("\x1b[");
                            let style = Style::default();
                            for segment in segments.filter(|s| !s.is_empty()) {
                                let Some(sep) = segment.find('m') else {
                                    println!("no 'm' in {segment:?}");
                                    continue;
                                };
                                let colour_code = &segment[..sep];
                                let col = colour_code.parse::<i32>().unwrap_or(
                                    // Probably a log line with a newline?
                                    2,
                                );
                                let text = &segment[sep + 1..];
                                egui::RichText::new(text)
                                    .color(match col {
                                        2 => Color32::from_gray(120),
                                        31 => Color32::from_rgb(197, 15, 31),
                                        32 => Color32::from_rgb(19, 161, 14),
                                        33 => Color32::from_rgb(193, 156, 0),
                                        34 => Color32::from_rgb(0, 55, 218),
                                        0 => Color32::from_gray(240),
                                        _ => {
                                            warn!("unrecognised colour code: {col}");
                                            Color32::from_gray(240)
                                        }
                                    })
                                    .monospace()
                                    .append_to(
                                        &mut layout_job,
                                        &style,
                                        FontSelection::Default,
                                        Align::Center,
                                    );
                            }
                        }
                    }
                    ui.add(egui::Label::new(layout_job).selectable(true));
                });
        });
    }
}

enum SceneControlCommand {
    TogglePause,
    Step,
    BigStep,
}
pub(crate) struct GuiSceneControl {
    paused: bool,
    should_step: usize,
    cmd_tx: Sender<SceneControlCommand>,
    cmd_rx: Receiver<SceneControlCommand>,
}

impl GuiSceneControl {
    fn new() -> Self {
        let (cmd_tx, cmd_rx) = mpsc::channel();
        Self {
            paused: false,
            should_step: 0,
            cmd_tx,
            cmd_rx,
        }
    }

    fn on_input(&mut self, input_handler: &InputHandler) {
        if input_handler.mod_super() && input_handler.pressed(KeyCode::KeyP) {
            self.cmd_tx.send(SceneControlCommand::TogglePause).unwrap();
        }
        if input_handler.mod_super() && input_handler.pressed(KeyCode::KeyS) {
            if input_handler.mod_shift() {
                self.cmd_tx.send(SceneControlCommand::BigStep).unwrap();
            } else {
                self.cmd_tx.send(SceneControlCommand::Step).unwrap();
            }
        }
    }

    fn build_closure(&mut self, frame: Frame, enabled: bool) -> Box<GuiClosure> {
        if let Some(next) = self.cmd_rx.try_iter().last() {
            match next {
                SceneControlCommand::TogglePause => self.paused = !self.paused,
                SceneControlCommand::Step => self.should_step += 1,
                SceneControlCommand::BigStep => self.should_step += 5,
            }
        }
        let paused = self.paused;
        let cmd_tx = self.cmd_tx.clone();

        Box::new(move |ctx| {
            egui::TopBottomPanel::bottom("scene-control")
                .frame(
                    frame
                        .fill(Color32::from_black_alpha(0))
                        .outer_margin(12.)
                        .inner_margin(0.),
                )
                .show_separator_line(false)
                .show_animated(ctx, enabled, |ui| {
                    ui.style_mut().override_text_style = Some(TextStyle::Monospace);
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        ui.spacing_mut().item_spacing = [2., 0.].into();
                        let size = 24.;
                        if ui
                            .add(
                                Button::new("⏸")
                                    .selected(paused)
                                    .min_size([size, size].into()),
                            )
                            .clicked()
                        {
                            cmd_tx.send(SceneControlCommand::TogglePause).unwrap();
                        };
                        if ui
                            .add(Button::new("⟳").min_size([size, size].into()))
                            .clicked()
                        {
                            if ui.input(|i| i.modifiers.shift) {
                                cmd_tx.send(SceneControlCommand::BigStep).unwrap();
                            } else {
                                cmd_tx.send(SceneControlCommand::Step).unwrap();
                            }
                        };
                    });
                });
        })
    }
    pub fn is_paused(&self) -> bool {
        self.paused
    }
    pub fn should_step(&mut self) -> bool {
        let rv = self.paused && self.should_step > 0;
        if rv {
            self.should_step -= 1;
        } else if self.paused {
            self.should_step = 0;
        }
        rv
    }
}

pub(crate) struct DebugGui {
    enabled: bool,
    pub(crate) object_tree: GuiObjectTree,
    object_view: GuiObjectView,
    pub(crate) console_log: GuiConsoleLog,
    pub(crate) scene_control: GuiSceneControl,
    last_viewport: AdjustedViewport,
    frame: Frame,

    wireframe_mouseovers: Vec<ObjectId>,
    last_update: Instant,
}

impl DebugGui {
    pub fn new() -> Result<Self> {
        Ok(Self {
            enabled: false,
            object_tree: GuiObjectTree::new(),
            object_view: GuiObjectView::new(),
            console_log: GuiConsoleLog::new()?,
            scene_control: GuiSceneControl::new(),
            last_viewport: AdjustedViewport::default(),
            frame: egui::Frame::default()
                .fill(Color32::from_rgba_unmultiplied(12, 12, 12, 245))
                .inner_margin(egui::Margin::same(6.)),
            wireframe_mouseovers: Vec::new(),
            last_update: Instant::now(),
        })
    }

    pub fn clear_mouseovers<O: ObjectTypeEnum>(&mut self, object_handler: &ObjectHandler<O>) {
        self.wireframe_mouseovers
            .drain(..)
            .filter(|o| self.object_tree.selected_id != *o)
            .filter_map(|o| gg_err::log_err_then(object_handler.get_collision_shape_mut(o)))
            .for_each(|mut c| c.hide_wireframe());
    }

    pub fn on_mouseovers<O: ObjectTypeEnum>(
        &mut self,
        object_handler: &ObjectHandler<O>,
        collisions: NonemptyVec<Collision<O>>,
    ) {
        self.wireframe_mouseovers = collisions
            .into_iter()
            .map(|c| c.other.object_id)
            .inspect(|&o| {
                if let Some(mut c) = gg_err::log_err_then(object_handler.get_collision_shape_mut(o))
                {
                    c.show_wireframe();
                }
            })
            .collect_vec();
    }

    pub fn build<O: ObjectTypeEnum>(
        &mut self,
        input_handler: &InputHandler,
        object_handler: &mut ObjectHandler<O>,
        gui_cmds: BTreeMap<ObjectId, Box<GuiInsideClosure>>,
    ) -> Box<GuiClosure> {
        if self.enabled {
            if input_handler.pressed(KeyCode::Escape) {
                self.object_tree.selected_id = ObjectId::root();
                gg_err::log_err(
                    self.object_view
                        .update_selection(object_handler, ObjectId::root()),
                );
            }
            self.object_tree
                .on_input(input_handler, object_handler, &self.wireframe_mouseovers);
            self.scene_control.on_input(input_handler);
        }
        if !self.object_tree.selected_id.is_root() {
            if gg_err::log_err_then(object_handler.get_object(self.object_tree.selected_id))
                .is_some()
            {
                gg_err::log_err(
                    self.object_view
                        .update_selection(object_handler, self.object_tree.selected_id),
                );
            } else {
                self.object_tree.selected_id = ObjectId::root();
                self.object_view.clear_selection();
            }
        }

        let build_object_tree = self.object_tree.build_closure(self.frame, self.enabled);
        let build_object_view = gg_err::log_and_ok(self.object_view.build_closure(
            object_handler,
            gui_cmds,
            self.frame,
            self.enabled,
        ));
        let build_console_log = self.console_log.build_closure(self.frame, self.enabled);
        let build_scene_control = self.scene_control.build_closure(self.frame, self.enabled);
        self.last_update = Instant::now();
        Box::new(move |ctx| {
            build_console_log(ctx);
            build_object_tree(ctx);
            if let Some(build_object_view) = build_object_view {
                build_object_view(ctx);
            }
            build_scene_control(ctx);
        })
    }

    pub fn on_end_step(&mut self, input_handler: &InputHandler, viewport: &mut AdjustedViewport) {
        let mut viewport_moved = false;
        if self.enabled && input_handler.mod_super() {
            let mut direction = Vec2::zero();
            if input_handler.down(KeyCode::ArrowLeft) {
                direction += Vec2::left();
            }
            if input_handler.down(KeyCode::ArrowRight) {
                direction += Vec2::right();
            }
            if input_handler.down(KeyCode::ArrowUp) {
                direction += Vec2::up();
            }
            if input_handler.down(KeyCode::ArrowDown) {
                direction += Vec2::down();
            }
            direction *= gg_float::micros(self.last_update.elapsed()) * 128.;
            let dx = if input_handler.mod_shift() {
                direction * 5.
            } else {
                direction
            };
            self.last_viewport.translation += dx;
            viewport_moved = true;
        }
        if viewport_moved {
            *viewport = self.last_viewport.clone();
        } else {
            self.last_viewport = viewport.clone();
        }
    }

    pub fn toggle(&mut self) {
        self.enabled = !self.enabled;
    }
    pub fn enabled(&self) -> bool {
        self.enabled
    }
    pub fn selected_object(&self) -> Option<ObjectId> {
        if self.object_tree.selected_id.is_root() {
            None
        } else {
            Some(self.object_tree.selected_id)
        }
    }
}
