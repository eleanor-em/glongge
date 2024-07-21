use std::cell::RefMut;
use glongge::core::SceneObjectWithId;
use glongge::core::scene::SceneObject;
use crate::mario::{
    block::brick::Brick,
    block::question_block::QuestionBlock,
    ObjectType,
    player::Player
};
use crate::mario::block::underground_brick::UndergroundBrick;

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
        ObjectType::UndergroundBrick => Some(obj.downcast_mut::<UndergroundBrick>().unwrap() as RefMut<dyn Bumpable>),
        _ => None
    }
}
