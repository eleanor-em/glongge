use crate::examples::mario::{
    block::{brick::Brick, question_block::QuestionBlock, underground_brick::UndergroundBrick},
    player::Player,
};
use crate::object_type::ObjectType;
use glongge::core::prelude::*;
use std::cell::RefMut;

pub mod brick;
pub mod decorative_pipe;
pub mod flagpole;
pub mod pipe;
pub mod plain_block;
pub mod question_block;
pub mod underground_brick;

pub trait Bumpable: SceneObject<ObjectType> {
    fn bump(&mut self, player: &mut Player);
}

pub fn downcast_bumpable_mut(
    obj: &mut SceneObjectWithId<ObjectType>,
) -> Option<RefMut<dyn Bumpable>> {
    match obj.get_type() {
        ObjectType::QuestionBlock => {
            Some(obj.checked_downcast_mut::<QuestionBlock>() as RefMut<dyn Bumpable>)
        }
        ObjectType::Brick => Some(obj.checked_downcast_mut::<Brick>() as RefMut<dyn Bumpable>),
        ObjectType::UndergroundBrick => {
            Some(obj.checked_downcast_mut::<UndergroundBrick>() as RefMut<dyn Bumpable>)
        }
        _ => None,
    }
}
