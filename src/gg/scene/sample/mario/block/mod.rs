use std::cell::RefMut;
use crate::gg::scene::sample::mario::ObjectType;
use crate::gg::{SceneObject, SceneObjectWithId};
use crate::gg::scene::sample::mario::block::brick::Brick;
use crate::gg::scene::sample::mario::block::question_block::QuestionBlock;
use crate::gg::scene::sample::mario::player::Player;

pub mod brick;
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
