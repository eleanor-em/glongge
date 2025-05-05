use crate::examples::mario::enemy::goomba::Goomba;

use glongge::core::prelude::*;
use std::cell::RefMut;

pub mod goomba;

pub trait Stompable: SceneObject {
    fn stomp(&mut self);
    fn dead(&self) -> bool;
}

pub fn downcast_stompable_mut(obj: &mut TreeSceneObject) -> Option<RefMut<dyn Stompable>> {
    obj.downcast_mut::<Goomba>()
        .map(|o| o as RefMut<dyn Stompable>)
}
