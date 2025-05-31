use crate::core::input::InputHandler;
use crate::core::prelude::*;
use crate::core::scene::{GuiClosure, GuiCommand};
use crate::core::update::collision::Collision;
use crate::core::update::{ObjectHandler, UpdatePerfStats};
use crate::core::vk::{AdjustedViewport, RenderPerfStats};
use crate::core::{ObjectId, TreeSceneObject};
use crate::gui::{GuiUi, TransformCell};
use crate::util::{NonemptyVec, ValueChannel, ValueChannelSender, gg_err, gg_float, gg_iter};
use egui::style::ScrollStyle;
use egui::text::LayoutJob;
use egui::{
    Align, Button, Color32, FontSelection, Frame, Id, Layout, Sense, Style, TextBuffer, TextFormat,
    TextStyle, Ui,
};
use itertools::Itertools;
use regex::Regex;
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

impl std::fmt::Debug for ObjectLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ObjectLabel::Root => write!(f, "<root>"),
            ObjectLabel::Unique(name, tags) => write!(f, "{name} {tags}"),
            ObjectLabel::Disambiguated(name, tags, idx) => write!(f, "{name} {tags} #{idx}"),
        }
    }
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
            ObjectLabel::Disambiguated(name, _, count) => format!("{name} #{count}"),
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
        self.absolute_cell.clear_state();
        self.relative_cell.clear_state();
        self.object_id = ObjectId::root();
    }

    fn update_selection(
        &mut self,
        object_handler: &ObjectHandler,
        selected_id: ObjectId,
    ) -> Result<()> {
        if self.object_id != selected_id {
            for c in object_handler
                .get_collision_shapes(self.object_id)
                .context("GuiObjectView::update_selection()")?
            {
                c.borrow_mut().hide_wireframe();
            }
            self.absolute_cell.clear_state();
            self.relative_cell.clear_state();
            self.object_id = selected_id;
            if !selected_id.is_root() {
                for c in object_handler
                    .get_collision_shapes(selected_id)
                    .context("GuiObjectView::update_selection()")?
                {
                    c.borrow_mut().show_wireframe();
                }
            }
        }
        Ok(())
    }

    fn get_object<'a>(&'a self, object_handler: &'a ObjectHandler) -> Result<&'a TreeSceneObject> {
        check_false!(self.object_id.is_root());
        Ok(object_handler
            .get_object_by_id(self.object_id)
            .context("GuiObjectView::get_object()")?
            // infallible
            .unwrap())
    }

    fn build_closure(
        &mut self,
        object_handler: &mut ObjectHandler,
        mut gui_cmds: BTreeMap<ObjectId, GuiCommand>,
        frame: Frame,
        enabled: bool,
    ) -> Result<Box<GuiClosure>> {
        let is_root = self.object_id.is_root();
        if is_root {
            return Ok(Box::new(|_| {}));
        }

        let name_cmd = self.create_name_cmd(object_handler)?;
        let transform_cmd = self.create_transform_cmd(object_handler)?;
        let gui_cmd = gui_cmds.remove(&self.object_id);

        Ok(Box::new(move |ctx| {
            egui::SidePanel::right(Id::new("object-view"))
                .frame(frame)
                .show_animated(ctx, enabled && !is_root, |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        name_cmd.call(ui);
                        transform_cmd.call(ui);
                        if let Some(gui_cmd) = gui_cmd {
                            ui.separator();
                            gui_cmd.call(ui);
                        }
                    });
                });
        }))
    }

    fn create_name_cmd(&self, object_handler: &mut ObjectHandler) -> Result<GuiCommand> {
        let name = format!(
            "{} [{}]",
            self.get_object(object_handler)
                .context("GuiObjectView::create_name_cmd()")?
                .nickname_or_type_name(),
            self.object_id.value_for_gui()
        );
        Ok(GuiCommand::new(move |ui: &mut Ui| {
            ui.vertical_centered(|ui| {
                let mut layout_job = LayoutJob::default();
                layout_job.append(
                    &name,
                    0.,
                    TextFormat {
                        color: Color32::from_white_alpha(255),
                        ..Default::default()
                    },
                );
                ui.add(egui::Label::new(layout_job).selectable(false));
                ui.separator();
            });
        }))
    }

    fn create_transform_cmd(&mut self, object_handler: &ObjectHandler) -> Result<GuiCommand> {
        let object_id = self.object_id;
        let object = self
            .get_object(object_handler)
            .context("GuiObjectView::create_transform_cmd()")?
            .clone(); // borrowck issues

        let mut absolute_transform = object_handler
            .absolute_transforms
            .get(&object_id)
            .copied()
            .with_context(|| format!("GuiObjectView::create_transform_cmd(): missing ObjectId in `absolute_transforms`: {object_id:?}"))?;
        let mut relative_transform = object.transform();
        self.absolute_cell.update_live(absolute_transform);
        self.relative_cell.update_live(relative_transform);

        match (self.absolute_cell.try_recv(), self.relative_cell.try_recv()) {
            (None, None) => {}
            (Some(_), Some(_)) => {
                check!(
                    false,
                    "absolute_cell and relative_cell should not both be received"
                );
            }
            (Some(next), None) => {
                let mut transform = object.transform_mut();
                transform.centre += next.centre - absolute_transform.centre;
                transform.rotation += next.rotation - absolute_transform.rotation;
                transform.scale += next.scale - absolute_transform.scale;
                absolute_transform = next;
            }
            (None, Some(next)) => {
                let mut transform = object.transform_mut();
                transform.centre += next.centre - relative_transform.centre;
                transform.rotation += next.rotation - relative_transform.rotation;
                transform.scale += next.scale - relative_transform.scale;
                relative_transform = next;
            }
        }

        let absolute_sender = self.absolute_cell.sender();
        let relative_sender = self.relative_cell.sender();
        let transform_cmd = move |ui: &mut Ui| {
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
        };
        Ok(GuiCommand::new(transform_cmd))
    }
}

struct GuiObjectTreeNode {
    label: ObjectLabel,
    object_id: ObjectId,
    displayed: BTreeMap<ObjectId, GuiObjectTreeNode>,
    disambiguation: Rc<RefCell<BTreeMap<String, usize>>>, // for id_salt()
    open: ValueChannel<bool>,
    expand_all_children: ValueChannel<bool>,
}

impl GuiObjectTreeNode {
    fn root() -> Self {
        Self {
            label: ObjectLabel::Root,
            object_id: ObjectId::root(),
            displayed: BTreeMap::new(),
            disambiguation: Rc::new(RefCell::new(BTreeMap::new())),
            open: ValueChannel::default(),
            expand_all_children: ValueChannel::default(),
        }
    }
    fn node(&self, object: &TreeSceneObject) -> Self {
        let name = object.nickname_or_type_name();
        let count = *self
            .disambiguation
            .borrow_mut()
            .entry(name.clone())
            .and_modify(|count| *count += 1)
            .or_default();
        Self {
            label: if count > 0 {
                ObjectLabel::Disambiguated(name, String::new(), count)
            } else {
                ObjectLabel::Unique(name, String::new())
            },
            object_id: object.object_id,
            displayed: BTreeMap::new(),
            disambiguation: self.disambiguation.clone(),
            open: ValueChannel::default(),
            expand_all_children: ValueChannel::default(),
        }
    }

    fn refresh_label(&mut self, object_handler: &ObjectHandler) {
        if !self.object_id.is_root() {
            let mut tags = String::new();
            if !gg_err::log_unwrap_or(
                Vec::new(),
                object_handler.get_collision_shapes(self.object_id),
            )
            .is_empty()
            {
                tags += "▣ ";
            }
            if gg_err::log_unwrap_or(false, object_handler.has_sprite_for_gui(self.object_id)) {
                tags += "👾 ";
            }
            self.label.set_tags(tags.trim());
        }
        for child in self.displayed.values_mut() {
            child.refresh_label(object_handler);
        }
    }

    fn update_open_with_selected(&mut self, selected_id: ObjectId) -> bool {
        if self.object_id == selected_id
            || self
                .displayed
                .values_mut()
                .any(|c| c.update_open_with_selected(selected_id))
        {
            self.open.overwrite(true);
            true
        } else {
            false
        }
    }

    fn as_builder(
        &mut self,
        selected_changed: bool,
        selected: &ValueChannel<ObjectId>,
    ) -> GuiObjectTreeBuilder {
        self.expand_all_children.try_recv_and_update();
        if let Some(false) = self.open.try_recv_and_update() {
            self.expand_all_children.overwrite(false);
        }
        GuiObjectTreeBuilder {
            label: self.label.clone(),
            object_id: self.object_id,
            displayed: self
                .displayed
                .iter_mut()
                .map(|(id, tree)| (*id, tree.as_builder(selected_changed, selected)))
                .collect(),
            open_tx: self.open.sender(),
            is_selected: selected.get() == self.object_id,
            selected_changed,
            selected_tx: selected.sender(),
            expand_all_children_tx: self.expand_all_children.sender(),
        }
    }
}

struct GuiObjectTree {
    root: GuiObjectTreeNode,
    show: ValueChannel<bool>,
    selected_id: ValueChannel<ObjectId>,
}

impl GuiObjectTree {
    fn new() -> Self {
        let selected_id = ObjectId::root();
        Self {
            root: GuiObjectTreeNode::root(),
            show: ValueChannel::with_value(true),
            selected_id: ValueChannel::with_value(selected_id),
        }
    }

    fn build_closure(&mut self, frame: Frame, enabled: bool) -> Box<GuiClosure> {
        self.show.try_recv_and_update();
        let show_cmd = self.create_show_cmd();
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
                    show_cmd.call(ui);
                });
        })
    }

    fn create_show_cmd(&mut self) -> GuiCommand {
        let selected_changed = self
            .selected_id
            .try_recv_and_update()
            .is_some_and(|next| self.root.update_open_with_selected(next));
        let mut root = self.root.as_builder(selected_changed, &self.selected_id);
        let show_tx = self.show.sender();
        GuiCommand::new(move |ui| {
            if show_tx.get() {
                ui.vertical_centered(|ui| {
                    show_tx.add_as_button(ui, "🌳");
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
            } else {
                show_tx.add_as_button(ui, "🌳");
            }
        })
    }

    fn get_parent_or_object(
        object_handler: &ObjectHandler,
        mouseover_id: ObjectId,
    ) -> Option<&TreeSceneObject> {
        gg_err::log_err_then(
            object_handler
                .get_parent_by_id(mouseover_id)
                .context("GuiObjectTree::get_parent_or_object(): parent"),
        )
        .or_else(|| {
            gg_err::log_err_then(
                object_handler
                    .get_object_by_id(mouseover_id)
                    .context("GuiObjectTree::get_parent_or_object(): object"),
            )
        })
    }

    fn on_input(
        &mut self,
        input_handler: &InputHandler,
        object_handler: &ObjectHandler,
        wireframe_mouseovers: &[ObjectId],
    ) {
        if input_handler.pressed(KeyCode::KeyF) {
            if let Some(object) = wireframe_mouseovers
                .get(
                    gg_iter::index_of(wireframe_mouseovers, &self.selected_id.get())
                        .map_or(0, |i| (i + 1) % wireframe_mouseovers.len()),
                )
                .and_then(|mouseover_id| Self::get_parent_or_object(object_handler, *mouseover_id))
            {
                self.selected_id.send(object.object_id);
            }
        }
        if input_handler.pressed(KeyCode::KeyC) {
            if self.selected_id.get().is_root() {
                if let Some(object_id) = object_handler.get_first_object_id_for_gui() {
                    self.selected_id.send(object_id);
                }
            } else if let Some(child) = gg_err::log_err_then(
                object_handler
                    .get_children(self.selected_id.get())
                    .map(|v| v.first()),
            ) {
                self.selected_id.send(child.object_id);
            } else {
                self.select_next_sibling(object_handler);
            }
        }
        if !input_handler.mod_super() && input_handler.pressed(KeyCode::KeyS) {
            self.select_next_sibling(object_handler);
        }
        if !input_handler.mod_super()
            && input_handler.mod_shift()
            && input_handler.pressed(KeyCode::KeyS)
        {
            self.select_prev_sibling(object_handler);
        }
        if !input_handler.mod_super() && input_handler.pressed(KeyCode::KeyP) {
            if let Some(parent) = gg_err::log_err_then(
                object_handler
                    .get_parent_by_id(self.selected_id.get())
                    .context("GuiObjectTree::on_input(): <P>: parent not found"),
            ) {
                self.selected_id.send(parent.object_id);
            }
        }
        if input_handler.pressed(KeyCode::KeyO) {
            self.show.overwrite(!self.show.get());
        }
    }
    fn select_next_sibling(&mut self, object_handler: &ObjectHandler) {
        self.select_nth_sibling(object_handler, 1);
    }
    fn select_prev_sibling(&mut self, object_handler: &ObjectHandler) {
        self.select_nth_sibling(object_handler, -1);
    }

    fn select_nth_sibling(&mut self, object_handler: &ObjectHandler, n: isize) {
        let parent_id = gg_err::log_err_then(
            object_handler
                .get_parent_by_id(self.selected_id.get())
                .context("GuiObjectTree::on_input(): parent not found"),
        )
        .map_or(ObjectId::root(), |parent| parent.object_id);
        let siblings = gg_err::log_unwrap_or(&Vec::new(), object_handler.get_children(parent_id))
            .iter()
            .map(|o| o.object_id)
            .collect_vec();
        if let Some(sibling_id) = gg_iter::index_of(&siblings, &self.selected_id.get())
            .map(|i| {
                let ix = isize::try_from(i).unwrap() + n;
                let ix = ix.rem_euclid(isize::try_from(siblings.len()).unwrap());
                siblings[ix as usize]
            })
            .or_else(|| {
                check_eq!(self.selected_id.get(), ObjectId::root());
                siblings.first().copied()
            })
        {
            self.selected_id.send(sibling_id);
        } else {
            error!(
                "select_next_sibling(): no sibling found for {:?} (parent={:?})",
                self.selected_id.get(),
                object_handler
                    .get_parent_by_id(self.selected_id.get())
                    .context("GuiObjectTree::on_input(): parent not found")
            );
        }
    }

    pub fn on_add_object(
        &mut self,
        object_handler: &ObjectHandler,
        object: &TreeSceneObject,
    ) -> Result<()> {
        let mut tree = &mut self.root;
        let chain = object_handler
            .get_parent_chain(object.object_id)
            .context("GuiObjectTree::on_add_object()")?;
        for id in chain.into_iter().rev() {
            if tree.displayed.contains_key(&id) {
                tree = tree.displayed.get_mut(&id).unwrap();
            } else {
                let child = tree.node(object);
                tree.displayed.insert(object.object_id, child);
                break;
            }
        }
        Ok(())
    }

    fn get_node_by_object_id(
        &mut self,
        object_handler: &ObjectHandler,
        object_id: ObjectId,
    ) -> Result<Option<&mut GuiObjectTreeNode>> {
        let chain = object_handler
            .get_parent_chain(object_id)
            .context("GuiObjectTree::get_node_by_object_id()")?;
        let mut tree = &mut self.root;
        for id in chain.into_iter().rev() {
            if id == object_id {
                return Ok(Some(tree));
            }
            if let Some(next) = tree.displayed.get_mut(&id) {
                tree = next;
            } else {
                // Orphaned object.
                bail!(
                    "GuiObjectTree::get_node_by_object_id(): orphaned object: {}",
                    object_handler.format_object_id_for_logging(object_id)
                );
            }
        }
        Ok(None)
    }

    pub fn on_remove_object(
        &mut self,
        object_handler: &ObjectHandler,
        removed_id: ObjectId,
    ) -> Result<()> {
        info!("GuiObjectTree::on_remove_object({removed_id:?})");
        if let Some(tree) = self
            .get_node_by_object_id(object_handler, removed_id)
            .context("GuiObjectTree::on_remove_object()")?
        {
            tree.displayed.remove(&removed_id);
        }
        if self.selected_id.get() == removed_id {
            self.selected_id.overwrite(ObjectId::root());
        }
        Ok(())
    }

    pub fn refresh_labels(&mut self, object_handler: &ObjectHandler) {
        self.root.refresh_label(object_handler);
    }
}

struct GuiObjectTreeBuilder {
    label: ObjectLabel,
    object_id: ObjectId,
    displayed: BTreeMap<ObjectId, GuiObjectTreeBuilder>,
    open_tx: ValueChannelSender<bool>,
    is_selected: bool,
    selected_changed: bool,
    selected_tx: ValueChannelSender<ObjectId>,
    expand_all_children_tx: ValueChannelSender<bool>,
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

                let mut header = egui::CollapsingHeader::new(self.label.name())
                    .id_salt(self.label.id_salt())
                    .show_background(self.is_selected)
                    .open(Some(self.open_tx.get() && !self.displayed.is_empty()));
                // Don't show it as "openable" if there are no children.
                if self.displayed.is_empty() {
                    header = header.icon(|ui, _openness, response| {
                        // Copied from egui documentation.
                        let stroke = ui.style().interact(response).fg_stroke;
                        let radius = 2.0;
                        ui.painter()
                            .circle_filled(response.rect.center(), radius, stroke.color);
                    });
                }
                let response = header.show(ui, |ui| {
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
                        if !self.expand_all_children_tx.get() && child_group.len() > max_displayed {
                            let expander_label =
                                egui::Label::new(format!("[..{}]", child_group.len()))
                                    .extend()
                                    .sense(Sense::click())
                                    .selectable(false);
                            if ui.add(expander_label).clicked() {
                                self.expand_all_children_tx.send(true);
                            }
                            ui.end_row();
                        }
                    }
                });
                if self.selected_changed && self.is_selected {
                    response.header_response.scroll_to_me(Some(Align::Center));
                }
                if response.header_response.double_clicked()
                    || response.header_response.secondary_clicked()
                {
                    self.open_tx.toggle();
                }
                if response.header_response.clicked() {
                    self.selected_tx.send(self.object_id);
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
    const MAX_LOG_LINES: usize = 1000;

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
        let re = Regex::new(
            "((?:INFO|WARN|ERROR)\x1b\\[0m \
                 (?:\x1b\\[1mupdate.*\x1b\\[1m\\}\x1b\\[0m\x1b\\[2m:\x1b\\[0m )?\
                 \x1b\\[2m).*?\
                 (glongge(?:\\/|\\\\)src(?:\\/|\\\\))",
        )
        .unwrap();
        re.replace_all(line, "$1$2").take()
    }

    fn build_closure(&mut self, frame: Frame, enabled: bool) -> Box<GuiClosure> {
        let mut line = String::new();
        if self.log_file.read_line(&mut line).unwrap() > 0 {
            self.log_output.push(Self::transform_log_line(&line));
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
                    ui.with_layout(Layout::right_to_left(Align::TOP), |ui| {
                        if ui
                            .add(Button::new("🖥").selected(view_perf == ViewPerfMode::Update))
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
                            .add(Button::new("🎨").selected(view_perf == ViewPerfMode::Render))
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
                            ui.add(egui::Label::new("FPS").selectable(false));
                            ui.add(
                                egui::Label::new(format!("{:.1}", perf_stats.fps()))
                                    .selectable(false),
                            );
                            ui.end_row();
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
                                let mut rich_text = egui::RichText::new(text)
                                    .color(match col {
                                        2 => Color32::from_gray(120),
                                        31 => Color32::from_rgb(197, 15, 31),
                                        32 => Color32::from_rgb(19, 161, 14),
                                        33 => Color32::from_rgb(193, 156, 0),
                                        34 => Color32::from_rgb(0, 55, 218),
                                        0 | 1 | 3 => Color32::from_gray(240),
                                        _ => {
                                            warn!("unrecognised colour code: {col}");
                                            Color32::from_gray(240)
                                        }
                                    })
                                    .monospace();
                                if col == 1 {
                                    // XXX: doesn't seem to do anything.
                                    rich_text = rich_text.strong();
                                }
                                if col == 3 {
                                    rich_text = rich_text.italics();
                                }
                                rich_text.append_to(
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
                        }
                        if ui
                            .add(Button::new("⟳").min_size([size, size].into()))
                            .clicked()
                        {
                            if ui.input(|i| i.modifiers.shift) {
                                cmd_tx.send(SceneControlCommand::BigStep).unwrap();
                            } else {
                                cmd_tx.send(SceneControlCommand::Step).unwrap();
                            }
                        }
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
    object_tree: GuiObjectTree,
    object_view: GuiObjectView,
    console_log: GuiConsoleLog,
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

    pub fn toggle(&mut self) {
        self.enabled = !self.enabled;
    }
    pub fn enabled(&self) -> bool {
        self.enabled
    }
    pub fn selected_object(&self) -> Option<ObjectId> {
        if !self.enabled || self.object_tree.selected_id.get().is_root() {
            None
        } else {
            Some(self.object_tree.selected_id.get())
        }
    }

    pub fn build(
        &mut self,
        input_handler: &InputHandler,
        object_handler: &mut ObjectHandler,
        gui_cmds: BTreeMap<ObjectId, GuiCommand>,
    ) -> Box<GuiClosure> {
        // Update state.
        self.build_update_with_enabled(input_handler, object_handler);
        gg_err::log_and_ok(
            self.object_view
                .update_selection(object_handler, self.object_tree.selected_id.get()),
        );
        self.last_update = Instant::now();

        // Build closures.
        let build_object_tree = self.object_tree.build_closure(self.frame, self.enabled);
        let build_object_view = gg_err::log_and_ok(self.object_view.build_closure(
            object_handler,
            gui_cmds,
            self.frame,
            self.enabled,
        ));
        let build_console_log = self.console_log.build_closure(self.frame, self.enabled);
        let build_scene_control = self.scene_control.build_closure(self.frame, self.enabled);
        Box::new(move |ctx| {
            build_console_log(ctx);
            build_object_tree(ctx);
            if let Some(build_object_view) = build_object_view {
                build_object_view(ctx);
            }
            build_scene_control(ctx);
        })
    }

    fn build_update_with_enabled(
        &mut self,
        input_handler: &InputHandler,
        object_handler: &mut ObjectHandler,
    ) {
        if self.enabled {
            if input_handler.pressed(KeyCode::Escape) {
                self.object_tree.selected_id.overwrite(ObjectId::root());
                gg_err::log_and_ok(
                    self.object_view
                        .update_selection(object_handler, ObjectId::root()),
                );
            }
            self.object_tree
                .on_input(input_handler, object_handler, &self.wireframe_mouseovers);
            self.scene_control.on_input(input_handler);
        } else {
            self.object_view.clear_selection();
        }
    }

    // Events:
    pub fn clear_mouseovers(&mut self, object_handler: &ObjectHandler) {
        self.wireframe_mouseovers
            .drain(..)
            .filter(|o| {
                gg_err::log_and_ok(object_handler.get_parent_chain(*o))
                    .is_some_and(|chain| !chain.contains(&self.object_tree.selected_id.get()))
            })
            .flat_map(|o| gg_err::log_unwrap_or(Vec::new(), object_handler.get_collision_shapes(o)))
            .for_each(|c| c.borrow_mut().hide_wireframe());
    }
    pub fn on_mouseovers(
        &mut self,
        object_handler: &ObjectHandler,
        collisions: NonemptyVec<Collision>,
    ) {
        self.wireframe_mouseovers = collisions
            .into_iter()
            .flat_map(|c| {
                let result = object_handler.get_collision_shapes(c.other.object_id);
                gg_err::log_unwrap_or(Vec::new(), result)
                    .into_iter()
                    .map(|c| {
                        c.borrow_mut().show_wireframe();
                        c.object_id()
                    })
            })
            .unique()
            .collect_vec();
    }
    pub fn on_add_object(
        &mut self,
        object_handler: &ObjectHandler,
        object: &TreeSceneObject,
    ) -> Result<()> {
        self.object_tree.on_add_object(object_handler, object)
    }
    pub fn on_done_adding_objects(&mut self, object_handler: &ObjectHandler) {
        self.object_tree.refresh_labels(object_handler);
    }
    pub fn on_remove_object(
        &mut self,
        object_handler: &ObjectHandler,
        remove_id: ObjectId,
    ) -> Result<()> {
        if self.object_tree.selected_id.get() == remove_id {
            self.object_view.clear_selection();
        }
        self.object_tree.on_remove_object(object_handler, remove_id)
    }
    /// Handles viewport moving with the arrow keys.
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
    pub fn on_perf_stats(
        &mut self,
        update_perf_stats: Option<UpdatePerfStats>,
        render_perf_stats: Option<RenderPerfStats>,
    ) {
        self.console_log.update_perf_stats(update_perf_stats);
        self.console_log.render_perf_stats(render_perf_stats);
    }
}
