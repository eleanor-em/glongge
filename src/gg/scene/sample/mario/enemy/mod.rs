use std::cell::RefMut;
use crate::gg::scene::sample::mario::enemy::goomba::Goomba;
use crate::gg::scene::sample::mario::ObjectType;
use crate::gg::{SceneObject, SceneObjectWithId};

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
