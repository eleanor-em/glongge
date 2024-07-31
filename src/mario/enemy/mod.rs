use std::cell::RefMut;
use glongge::core::prelude::*;
use crate::mario::{
    enemy::goomba::Goomba,
};
use crate::object_type::ObjectType;

pub mod goomba;

pub trait Stompable: SceneObject<ObjectType> {
    fn stomp(&mut self);
    fn dead(&self) -> bool;
}

pub fn downcast_stompable_mut(obj: &mut SceneObjectWithId<ObjectType>) -> Option<RefMut<dyn Stompable>> {
    match obj.get_type() {
        ObjectType::Goomba => Some(obj.downcast_mut::<Goomba>().unwrap() as RefMut<dyn Stompable>),
        _ => None
    }
}
