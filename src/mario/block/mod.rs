use std::cell::RefMut;
use glongge::core::{SceneObject, SceneObjectWithId};
use crate::mario::{
    block::brick::Brick,
    ObjectType,
    block::question_block::QuestionBlock,
    player::Player
};

pub mod brick;
pub mod plain_block;
pub mod flagpole;
pub mod question_block;
pub mod pipe;
pub mod decorative_pipe;
pub mod underground_brick;

pub trait Bumpable: SceneObject<ObjectType> {
    fn bump(&mut self, player: &mut Player);
}

pub fn downcast_bumpable_mut(obj: &mut SceneObjectWithId<ObjectType>) -> Option<RefMut<dyn Bumpable>> {
    match obj.get_type() {
        ObjectType::QuestionBlock => Some(obj.downcast_mut::<QuestionBlock>().unwrap() as RefMut<dyn Bumpable>),
        ObjectType::Brick => Some(obj.downcast_mut::<Brick>().unwrap() as RefMut<dyn Bumpable>),
        _ => None
    }
}
