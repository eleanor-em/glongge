use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;
use egui::{Color32, Id};
use itertools::Itertools;
use crate::core::{ObjectId, ObjectTypeEnum, SceneObjectWithId};
use crate::core::scene::GuiClosure;
use crate::core::update::debug_gui::ObjectLabel::Disambiguated;
use crate::core::update::ObjectHandler;
use crate::gui::GuiUi;

#[derive(Clone, Eq, PartialEq)]
enum ObjectLabel {
    Root,
    Unique(String),
    Disambiguated(String, usize),
}

impl ObjectLabel {
    fn name(&self) -> String {
        match self {
            ObjectLabel::Root => "<root>".to_string(),
            ObjectLabel::Unique(name) => name.clone(),
            Disambiguated(name, _) => name.clone(),
        }
    }

    fn to_string(&self) -> String {
        match self {
            ObjectLabel::Root => "<root>".to_string(),
            ObjectLabel::Unique(name) => name.clone(),
            Disambiguated(name, count) => format!("{name} {count}"),
        }
    }
}

pub(crate) struct GuiObjectTree {
    label: ObjectLabel,
    displayed: BTreeMap<ObjectId, GuiObjectTree>,
    depth: usize,
    disambiguation: Rc<RefCell<BTreeMap<String, usize>>>,
}

impl GuiObjectTree {
    fn new() -> Self {
        Self {
            label: ObjectLabel::Root,
            displayed: BTreeMap::new(),
            depth: 0,
            disambiguation: Rc::new(RefCell::new(BTreeMap::new()))
        }
    }

    fn child(&self, label: String) -> Self {
        let count = *self.disambiguation.borrow_mut().entry(label.clone())
            .and_modify(|count| { *count += 1 })
            .or_default();
        Self {
            label: if count > 0 {
                ObjectLabel::Disambiguated(label, count)
            } else {
                ObjectLabel::Unique(label)
            },
            displayed: BTreeMap::new(),
            depth: 0,
            disambiguation: self.disambiguation.clone(),
        }
    }

    pub fn on_add_object<O: ObjectTypeEnum>(&mut self, object_handler: &ObjectHandler<O>, object: &SceneObjectWithId<O>) {
        let mut child = self.child(object.inner.borrow().name());
        let mut tree = self;
        for id in object_handler.get_parent_chain_or_panic(object.object_id).into_iter().rev() {
            if tree.displayed.contains_key(&id) {
                child.depth += 1;
                tree = tree.displayed.get_mut(&id).unwrap();
            } else {
                tree.displayed.insert(object.object_id, child);
                return;
            }
        };
    }

    pub fn on_remove_object<O: ObjectTypeEnum>(&mut self, object_handler: &ObjectHandler<O>, removed_id: ObjectId) {
        let mut chain = object_handler.get_parent_chain_or_panic(removed_id);
        let mut tree = self;
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

    fn as_builder(&self) -> GuiObjectTreeBuilder {
        GuiObjectTreeBuilder {
            label: self.label.clone(),
            displayed: self.displayed.iter()
                .map(|(id, tree)| (*id, tree.as_builder()))
                .collect(),
            depth: self.depth,
        }
    }
}

struct GuiObjectTreeBuilder {
    label: ObjectLabel,
    displayed: BTreeMap<ObjectId, GuiObjectTreeBuilder>,
    depth: usize,
}
impl GuiObjectTreeBuilder {
    fn build(&self, ui: &mut GuiUi) {
        if self.label == ObjectLabel::Root {
            self.displayed.values().for_each(|tree| tree.build(ui));
        } else {
            ui.collapsing(self.label.to_string(), |ui| {
                let by_name = self.displayed.values()
                    .chunk_by(|tree| tree.label.name());
                for (_, child_group) in by_name.into_iter() {
                    let child_group = child_group.collect_vec();
                    let max_displayed = 5;
                    ui.indent(0, |ui| {
                        child_group.iter().take(max_displayed)
                            .for_each(|tree| tree.build(ui));
                        if child_group.len() > max_displayed {
                            ui.label(format!("[..{}]", child_group.len()));
                        }
                    });
                }
            });
        }

        if self.depth == 0 {
            ui.spacing();
        }
    }
}

pub(crate) struct DebugGui {
    pub(crate) object_tree: GuiObjectTree,
    pub(crate) enabled: bool,
}

impl DebugGui {
    pub fn new() -> Self {
        Self {
            object_tree: GuiObjectTree::new(),
            enabled: false,
        }
    }

    pub fn build(&self) -> Box<GuiClosure> {
        let enabled = self.enabled;
        let object_tree_builder = self.object_tree.as_builder();
        Box::new(move |ctx| {
            egui::SidePanel::left(Id::new("Object Tree"))
                .frame(egui::Frame::default()
                    .fill(Color32::from_rgba_unmultiplied(12, 12, 12, 245))
                    .inner_margin(egui::Margin::same(6.)))
                .show_animated(&ctx, enabled, |ui| {
                    egui::ScrollArea::new([false, true])
                        .show(ui, |ui| {
                            ui.add(egui::Label::new("Object Tree 🌳")
                               .extend());
                            ui.separator();
                            object_tree_builder.build(ui);
                        });
                });
        })
    }

    pub fn toggle(&mut self) { self.enabled = !self.enabled; }
}
