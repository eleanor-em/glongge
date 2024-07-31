use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::{Debug, Formatter},
};
use itertools::Itertools;
use crate::core::{
    prelude::*,
    update::PendingAddObject,
    AnySceneObject,
    ObjectId,
    ObjectTypeEnum,
    SceneObjectWithId,
    util::{
        linalg::Vec2,
        UnorderedPair,
        collision::Collider
    }
};
use crate::core::util::collision::GgInternalCollisionShape;

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum CollisionResponse {
    Continue,
    Done,
}

#[derive(Debug)]
pub(crate) struct CollisionNotification<ObjectType: ObjectTypeEnum> {
    pub(crate) this: SceneObjectWithId<ObjectType>,
    pub(crate) other: SceneObjectWithId<ObjectType>,
    pub(crate) mtv: Vec2,
}

pub struct Collision<ObjectType: ObjectTypeEnum> {
    pub other: SceneObjectWithId<ObjectType>,
    pub mtv: Vec2,
}

impl<ObjectType: ObjectTypeEnum> Debug for Collision<ObjectType> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "({:?} at {}, mtv={})", self.other.object_id, self.other.transform().centre, self.mtv)
    }
}

pub struct CollisionHandler {
    object_ids_by_emitting_tag: BTreeMap<&'static str, BTreeSet<ObjectId>>,
    object_ids_by_listening_tag: BTreeMap<&'static str, BTreeSet<ObjectId>>,
    possible_collisions: BTreeSet<UnorderedPair<ObjectId>>,
}

impl CollisionHandler {
    pub(crate) fn new() -> Self {
        Self {
            object_ids_by_emitting_tag: BTreeMap::new(),
            object_ids_by_listening_tag: BTreeMap::new(),
            possible_collisions: BTreeSet::new()
        }
    }
    pub(crate) fn add_objects<'a, ObjectType: ObjectTypeEnum, I>(
        &'a mut self,
        added_objects: I
    )
    where I: Iterator<Item=&'a (ObjectId, PendingAddObject<ObjectType>)>
    {
        let mut new_object_ids_by_emitting_tag = BTreeMap::<&'static str, Vec<ObjectId>>::new();
        let mut new_object_ids_by_listening_tag = BTreeMap::<&'static str, Vec<ObjectId>>::new();
        for (id, obj) in added_objects
            .filter(|(_, obj)| obj.inner.borrow().get_type() == ObjectTypeEnum::gg_collider()) {
            let obj = obj.inner.borrow();
            for tag in obj.emitting_tags() {
                new_object_ids_by_emitting_tag.entry(tag).or_default().push(*id);
            }
            for tag in obj.listening_tags() {
                new_object_ids_by_listening_tag.entry(tag).or_default().push(*id);
            }
        }

        for tag in new_object_ids_by_emitting_tag.keys() {
            self.object_ids_by_emitting_tag.entry(tag).or_default().extend(
                new_object_ids_by_emitting_tag.get(tag).unwrap());
            self.object_ids_by_listening_tag.entry(tag).or_default();
        }
        for tag in new_object_ids_by_listening_tag.keys() {
            self.object_ids_by_emitting_tag.entry(tag).or_default();
            self.object_ids_by_listening_tag.entry(tag).or_default().extend(
                new_object_ids_by_listening_tag.get(tag).unwrap());
        }

        for (tag, emitters) in new_object_ids_by_emitting_tag {
            let listeners = self.object_ids_by_listening_tag.get(tag)
                .unwrap_or_else(|| panic!("object_ids_by_listening tag missing tag: {tag}"));
            let new_possible_collisions = emitters.into_iter().cartesian_product(listeners.iter())
                .filter_map(|(emitter, listener)| UnorderedPair::new_distinct(emitter, *listener));
            self.possible_collisions.extend(new_possible_collisions);
        }
        for (tag, listeners) in new_object_ids_by_listening_tag {
            let emitters = self.object_ids_by_emitting_tag.get(tag)
                .unwrap_or_else(|| panic!("object_ids_by_tag tag missing tag: {tag}"));
            let new_possible_collisions = emitters.iter().cartesian_product(listeners.into_iter())
                .filter_map(|(emitter, listener)| UnorderedPair::new_distinct(*emitter, listener));
            self.possible_collisions.extend(new_possible_collisions);
        }
    }
    pub(crate) fn remove_objects(&mut self, removed_ids: &BTreeSet<ObjectId>) {
        for ids in self.object_ids_by_emitting_tag.values_mut() {
            ids.retain(|id| !removed_ids.contains(id));
        }
        for ids in self.object_ids_by_listening_tag.values_mut() {
            ids.retain(|id| !removed_ids.contains(id));
        }
        self.possible_collisions.retain(|pair| !removed_ids.contains(&pair.fst()) && !removed_ids.contains(&pair.snd()));
    }
    pub(crate) fn get_collisions<ObjectType: ObjectTypeEnum>(
        &self,
        absolute_transforms: &BTreeMap<ObjectId, Transform>,
        parents: &BTreeMap<ObjectId, ObjectId>,
        objects: &BTreeMap<ObjectId, AnySceneObject<ObjectType>>
    ) -> Vec<CollisionNotification<ObjectType>> {
        let collisions = self.get_collisions_inner(absolute_transforms, objects);
        let mut rv = Vec::with_capacity(collisions.len() * 2);
        for (ids, mtv) in collisions {
            let this_id = parents.get(&ids.fst())
                .unwrap_or_else(|| panic!("missing object_id in parents: {:?}", ids.fst()));
            let other_id = parents.get(&ids.snd())
                .unwrap_or_else(|| panic!("missing object_id in parents: {:?}", ids.snd()));

            let this = SceneObjectWithId::new(*this_id, objects[this_id].clone());
            let (this_listening, this_emitting) = {
                let this = this.inner.borrow();
                (this.listening_tags(), this.emitting_tags())
            };
            let other = SceneObjectWithId::new(*other_id, objects[other_id].clone());
            let (other_listening, other_emitting) = {
                let other = other.inner.borrow();
                (other.listening_tags(), other.emitting_tags())
            };
            if !this_listening.into_iter().chain(other_emitting).all_unique() {
                rv.push(CollisionNotification {
                    this: this.clone(),
                    other: other.clone(),
                    mtv,
                });
            };
            if !other_listening.into_iter().chain(this_emitting).all_unique() {
                rv.push(CollisionNotification {
                    this: other,
                    other: this,
                    mtv: -mtv,
                });
            }
        }
        rv
    }

    fn get_collisions_inner<ObjectType: ObjectTypeEnum>(
        &self,
        absolute_transforms: &BTreeMap<ObjectId, Transform>,
        objects: &BTreeMap<ObjectId, AnySceneObject<ObjectType>>
    ) -> Vec<(UnorderedPair<ObjectId>, Vec2)> {
        self.possible_collisions.iter()
            .filter_map(|ids| {
                let this = objects[&ids.fst()].checked_downcast::<GgInternalCollisionShape>().collider()
                    .transformed(absolute_transforms
                        .get(&ids.fst())
                        .unwrap_or_else(|| panic!("missing object_id in absolute_transforms: {:?}", ids.fst())));
                let other = objects[&ids.snd()].checked_downcast::<GgInternalCollisionShape>().collider()
                    .transformed(absolute_transforms
                        .get(&ids.snd())
                        .unwrap_or_else(|| panic!("missing object_id in absolute_transforms: {:?}", ids.snd())));
                this.collides_with(&other).map(|mtv| (*ids, mtv))
            })
            .collect()
    }

    pub(crate) fn get_object_ids_by_emitting_tag(&self, tag: &'static str) -> &BTreeSet<ObjectId> {
        self.object_ids_by_emitting_tag.get(tag)
            .unwrap_or_else(|| panic!("missing tag in object_ids_by_emitting_tag: {tag}"))
    }
}
