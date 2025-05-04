use crate::core::{ObjectId, ObjectTypeEnum, SceneObjectWrapper, TreeSceneObject, prelude::*};
use crate::util::{
    UnorderedPair,
    collision::{Collider, GgInternalCollisionShape},
    gg_err,
};
use itertools::Itertools;
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
pub(crate) struct CollisionNotification<ObjectType: ObjectTypeEnum> {
    pub(crate) this: TreeSceneObject<ObjectType>,
    pub(crate) other: TreeSceneObject<ObjectType>,
    pub(crate) mtv: Vec2,
}

/// Represents a collision between two objects in the scene.
///
/// # Fields
/// * `other` - The scene object that this object collided with
/// * `mtv` - Minimum Translation Vector (MTV): the vector needed to separate the colliding
///   objects. Direction points from the other object towards this one.
pub struct Collision<ObjectType: ObjectTypeEnum> {
    pub other: TreeSceneObject<ObjectType>,
    pub mtv: Vec2,
}

impl<ObjectType: ObjectTypeEnum> Debug for Collision<ObjectType> {
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
    pub(crate) fn add_objects<'a, ObjectType: ObjectTypeEnum, I>(&'a mut self, added_objects: I)
    where
        I: Iterator<Item = &'a TreeSceneObject<ObjectType>>,
    {
        let mut new_object_ids_by_emitting_tag = BTreeMap::<&'static str, Vec<ObjectId>>::new();
        let mut new_object_ids_by_listening_tag = BTreeMap::<&'static str, Vec<ObjectId>>::new();
        for obj in added_objects.filter(|obj| {
            obj.scene_object.wrapped.borrow().gg_type_enum() == ObjectTypeEnum::gg_collider()
        }) {
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
    pub(crate) fn get_collisions<ObjectType: ObjectTypeEnum>(
        &self,
        parents: &BTreeMap<ObjectId, ObjectId>,
        objects: &BTreeMap<ObjectId, SceneObjectWrapper<ObjectType>>,
    ) -> Vec<CollisionNotification<ObjectType>> {
        let collisions = self.get_collisions_inner(objects);
        let mut rv = Vec::with_capacity(collisions.len() * 2);
        for (ids, mtv) in collisions {
            gg_err::log_err_and_ignore(Self::process_collision_inner(
                parents, objects, &mut rv, &ids, mtv,
            ));
        }
        rv
    }

    fn get_collisions_inner<ObjectType: ObjectTypeEnum>(
        &self,
        objects: &BTreeMap<ObjectId, SceneObjectWrapper<ObjectType>>,
    ) -> Vec<(UnorderedPair<ObjectId>, Vec2)> {
        self.possible_collisions
            .iter()
            .filter_map(|ids| {
                let o1 = objects[&ids.fst()].checked_downcast::<GgInternalCollisionShape>();
                let o2 = objects[&ids.snd()].checked_downcast::<GgInternalCollisionShape>();
                o1.collider()
                    .collides_with(o2.collider())
                    .map(|mtv| (*ids, mtv))
            })
            .collect()
    }

    fn process_collision_inner<ObjectType: ObjectTypeEnum>(
        parents: &BTreeMap<ObjectId, ObjectId>,
        objects: &BTreeMap<ObjectId, SceneObjectWrapper<ObjectType>>,
        rv: &mut Vec<CollisionNotification<ObjectType>>,
        ids: &UnorderedPair<ObjectId>,
        mtv: Vec2,
    ) -> Result<()> {
        let this_id = parents.get(&ids.fst()).with_context(|| {
            format!(
                "CollisionHandler: missing ObjectId in `parents`: {:?}",
                ids.fst()
            )
        })?;
        let other_id = parents.get(&ids.snd()).with_context(|| {
            format!(
                "CollisionHandler: missing ObjectId in `parents`: {:?}",
                ids.snd()
            )
        })?;

        let this = TreeSceneObject {
            object_id: *this_id,
            parent_id: *parents.get(this_id).unwrap_or(&ObjectId::root()),
            scene_object: objects[this_id].clone(),
        };
        let (this_listening, this_emitting) = {
            let this = this.scene_object.wrapped.borrow();
            (this.listening_tags(), this.emitting_tags())
        };
        let other = TreeSceneObject {
            object_id: *other_id,
            parent_id: *parents.get(other_id).unwrap_or(&ObjectId::root()),
            scene_object: objects[other_id].clone(),
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
