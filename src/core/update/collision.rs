use crate::core::update::ObjectHandler;
use crate::core::{ObjectId, TreeSceneObject, prelude::*};
use crate::util::{
    UnorderedPair,
    collision::{Collider, GgInternalCollisionShape},
    gg_err,
};
use itertools::Itertools;
use std::any::TypeId;
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::{Debug, Formatter},
};

/// Specifies how collision processing should proceed after handling a collision.
///
/// * `Continue` - Continue processing additional collisions after this one
/// * `Done` - Stop processing additional collisions after this one
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum CollisionResponse {
    Continue,
    Done,
}

#[derive(Debug)]
pub(crate) struct CollisionNotification {
    pub(crate) this: TreeSceneObject,
    pub(crate) other: TreeSceneObject,
    pub(crate) mtv: Vec2,
}

/// Represents a collision between two objects in the scene.
///
/// # Fields
/// * `other` - The scene object that this object collided with
/// * `mtv` - Minimum Translation Vector (MTV): the vector needed to separate the colliding
///   objects. Direction points from the other object towards this one.
pub struct Collision {
    pub other: TreeSceneObject,
    pub mtv: Vec2,
}

impl Debug for Collision {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({:?} at {}, mtv={})",
            self.other.object_id,
            self.other.transform().centre,
            self.mtv
        )
    }
}

pub(crate) struct CollisionHandler {
    object_ids_by_emitting_tag: BTreeMap<&'static str, BTreeSet<ObjectId>>,
    object_ids_by_listening_tag: BTreeMap<&'static str, BTreeSet<ObjectId>>,
    possible_collisions: BTreeSet<UnorderedPair<ObjectId>>,
}

impl CollisionHandler {
    pub(crate) fn new() -> Self {
        Self {
            object_ids_by_emitting_tag: BTreeMap::new(),
            object_ids_by_listening_tag: BTreeMap::new(),
            possible_collisions: BTreeSet::new(),
        }
    }
    pub(crate) fn add_objects<'a, I>(&'a mut self, added_objects: I)
    where
        I: Iterator<Item = &'a TreeSceneObject>,
    {
        let mut new_object_ids_by_emitting_tag = BTreeMap::<&'static str, Vec<ObjectId>>::new();
        let mut new_object_ids_by_listening_tag = BTreeMap::<&'static str, Vec<ObjectId>>::new();
        for obj in added_objects
            .filter(|obj| obj.scene_object.type_id == TypeId::of::<GgInternalCollisionShape>())
        {
            let id = obj.object_id();
            let obj = obj.scene_object.wrapped.borrow();
            for tag in obj.emitting_tags() {
                new_object_ids_by_emitting_tag
                    .entry(tag)
                    .or_default()
                    .push(id);
            }
            for tag in obj.listening_tags() {
                new_object_ids_by_listening_tag
                    .entry(tag)
                    .or_default()
                    .push(id);
            }
        }

        for (tag, new_object_ids) in &new_object_ids_by_emitting_tag {
            self.object_ids_by_emitting_tag
                .entry(tag)
                .or_default()
                .extend(new_object_ids);
            self.object_ids_by_listening_tag.entry(tag).or_default();
        }
        for (tag, new_object_ids) in &new_object_ids_by_listening_tag {
            self.object_ids_by_emitting_tag.entry(tag).or_default();
            self.object_ids_by_listening_tag
                .entry(tag)
                .or_default()
                .extend(new_object_ids);
        }

        for (tag, emitters) in new_object_ids_by_emitting_tag {
            if let Some(listeners) = self.object_ids_by_listening_tag.get(tag) {
                let new_possible_collisions = emitters
                    .into_iter()
                    .cartesian_product(listeners.iter())
                    .filter_map(|(emitter, listener)| {
                        UnorderedPair::new_distinct(emitter, *listener)
                    });
                self.possible_collisions.extend(new_possible_collisions);
            } else {
                error!("CollisionHandler: `object_ids_by_listening` tag missing tag: {tag}");
            }
        }
        for (tag, listeners) in new_object_ids_by_listening_tag {
            if let Some(emitters) = self.object_ids_by_emitting_tag.get(tag) {
                let new_possible_collisions = emitters
                    .iter()
                    .cartesian_product(listeners.into_iter())
                    .filter_map(|(emitter, listener)| {
                        UnorderedPair::new_distinct(*emitter, listener)
                    });
                self.possible_collisions.extend(new_possible_collisions);
            } else {
                error!("CollisionHandler: `object_ids_by_tag` tag missing tag: {tag}");
            }
        }
    }
    pub(crate) fn remove_objects(&mut self, removed_ids: &BTreeSet<ObjectId>) {
        for ids in self
            .object_ids_by_emitting_tag
            .values_mut()
            .chain(self.object_ids_by_listening_tag.values_mut())
        {
            ids.retain(|id| !removed_ids.contains(id));
        }
        self.possible_collisions.retain(|pair| {
            !removed_ids.contains(&pair.fst()) && !removed_ids.contains(&pair.snd())
        });
    }
    pub(crate) fn get_collisions(
        &self,
        object_handler: &ObjectHandler,
    ) -> Vec<CollisionNotification> {
        let collisions = self.get_collisions_inner(object_handler);
        let mut rv = Vec::with_capacity(collisions.len() * 2);
        for (ids, mtv) in collisions {
            gg_err::log_err_and_ignore(Self::process_collision_inner(
                object_handler,
                &mut rv,
                &ids,
                mtv,
            ));
        }
        rv
    }

    fn get_collisions_inner(
        &self,
        object_handler: &ObjectHandler,
    ) -> Vec<(UnorderedPair<ObjectId>, Vec2)> {
        self.possible_collisions
            .iter()
            .filter_map(|ids| {
                let o1 = object_handler.objects[&ids.fst()]
                    .downcast::<GgInternalCollisionShape>()
                    .unwrap();
                let o2 = object_handler.objects[&ids.snd()]
                    .downcast::<GgInternalCollisionShape>()
                    .unwrap();
                o1.collider()
                    .collides_with(o2.collider())
                    .map(|mtv| (*ids, mtv))
            })
            .collect()
    }

    fn process_collision_inner(
        object_handler: &ObjectHandler,
        rv: &mut Vec<CollisionNotification>,
        ids: &UnorderedPair<ObjectId>,
        mtv: Vec2,
    ) -> Result<()> {
        let this = object_handler
            .get_parent_by_id(ids.fst())
            .context("CollisionHandler::process_collision_inner()")?
            .context("CollisionHandler: root in `ids`")?;
        let other = object_handler
            .get_parent_by_id(ids.snd())
            .context("CollisionHandler::process_collision_inner()")?
            .context("CollisionHandler: root in `ids`")?;

        let (this_listening, this_emitting) = {
            let this = this.scene_object.wrapped.borrow();
            (this.listening_tags(), this.emitting_tags())
        };
        let (other_listening, other_emitting) = {
            let other = other.scene_object.wrapped.borrow();
            (other.listening_tags(), other.emitting_tags())
        };
        if !this_listening
            .into_iter()
            .chain(other_emitting)
            .all_unique()
        {
            rv.push(CollisionNotification {
                this: this.clone(),
                other: other.clone(),
                mtv,
            });
        }
        if !other_listening
            .into_iter()
            .chain(this_emitting)
            .all_unique()
        {
            rv.push(CollisionNotification {
                this: other,
                other: this,
                mtv: -mtv,
            });
        }
        Ok(())
    }

    pub(crate) fn all_tags(&self) -> Vec<&'static str> {
        self.object_ids_by_emitting_tag
            .keys()
            .copied()
            .chain(self.object_ids_by_listening_tag.keys().copied())
            .collect()
    }

    pub(crate) fn get_object_ids_by_emitting_tag(
        &self,
        tag: &'static str,
    ) -> Result<&BTreeSet<ObjectId>> {
        self.object_ids_by_emitting_tag.get(tag).with_context(|| {
            format!("CollisionHandler: `object_ids_by_emitting_tag` missing tag: {tag}")
        })
    }
}
