use std::{
    rc::Rc,
    collections::{BTreeMap, BTreeSet},
    fmt::{Debug, Formatter},
    cell::RefCell
};
use itertools::Itertools;
use crate::{
    core::{
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
    }
};

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum CollisionResponse {
    Continue,
    Done,
}

pub(crate) struct CollisionNotification<ObjectType: ObjectTypeEnum> {
    pub(crate) this: SceneObjectWithId<ObjectType>,
    pub(crate) other: SceneObjectWithId<ObjectType>,
    pub(crate) mtv: Vec2,
}

#[derive(Clone)]
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
    pub(crate) fn update_with_added_objects<ObjectType: ObjectTypeEnum>(
        &mut self,
        added_objects: &BTreeMap<ObjectId, PendingAddObject<ObjectType>>
    ) {
        let mut new_object_ids_by_emitting_tag = BTreeMap::<&'static str, Vec<ObjectId>>::new();
        let mut new_object_ids_by_listening_tag = BTreeMap::<&'static str, Vec<ObjectId>>::new();

        for (id, obj) in added_objects {
            let obj = obj.inner.borrow();
            for tag in obj.emitting_tags() {
                new_object_ids_by_emitting_tag.entry(tag).or_default().push(*id);
                new_object_ids_by_listening_tag.entry(tag).or_default();
            }
            for tag in obj.listening_tags() {
                new_object_ids_by_emitting_tag.entry(tag).or_default();
                new_object_ids_by_listening_tag.entry(tag).or_default().push(*id);
            }
        }
        for tag in new_object_ids_by_emitting_tag.keys() {
            self.object_ids_by_emitting_tag.entry(tag).or_default().extend(
                new_object_ids_by_emitting_tag.get(tag).unwrap());
        }
        for tag in new_object_ids_by_listening_tag.keys() {
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
    pub(crate) fn update_with_removed_objects(&mut self, removed_ids: &BTreeSet<ObjectId>) {
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
        objects: &BTreeMap<ObjectId, Rc<RefCell<AnySceneObject<ObjectType>>>>
    ) -> Vec<CollisionNotification<ObjectType>> {
        let collisions = self.get_collisions_inner(objects);
        let mut rv = Vec::with_capacity(collisions.len() * 2);
        for (ids, mtv) in collisions {
            let this = SceneObjectWithId::new(ids.fst(), objects[&ids.fst()].clone());
            let (this_listening, this_emitting) = {
                let this = this.inner.borrow();
                (this.listening_tags(), this.emitting_tags())
            };
            let other = SceneObjectWithId::new(ids.snd(), objects[&ids.snd()].clone());
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
        objects: &BTreeMap<ObjectId, Rc<RefCell<AnySceneObject<ObjectType>>>>
    ) -> Vec<(UnorderedPair<ObjectId>, Vec2)> {
        self.possible_collisions.iter()
            .filter_map(|ids| {
                let this = objects[&ids.fst()].borrow();
                let other = objects[&ids.snd()].borrow();
                this.collider().collides_with(&other.collider()).map(|mtv| (*ids, mtv))
            })
            .collect()
    }

    pub(crate) fn get_object_ids_by_emitting_tag(&self, tag: &'static str) -> Option<&BTreeSet<ObjectId>> {
        self.object_ids_by_emitting_tag.get(tag)
    }
}
