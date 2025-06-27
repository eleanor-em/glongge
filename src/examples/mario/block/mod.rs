use crate::examples::mario::{
    block::{brick::Brick, question_block::QuestionBlock, underground_brick::UndergroundBrick},
    player::Player,
};

use glongge::core::prelude::*;
use std::cell::RefMut;

pub mod brick;
pub mod decorative_pipe;
pub mod flagpole;
pub mod pipe;
pub mod plain_block;
pub mod question_block;
pub mod underground_brick;

pub trait Bumpable: SceneObject {
    fn bump(&mut self, player: &mut Player);
}

pub fn downcast_bumpable_mut(obj: &mut TreeSceneObject) -> Option<RefMut<'_, dyn Bumpable>> {
    obj.downcast_mut::<QuestionBlock>()
        .map(|o| o as RefMut<dyn Bumpable>)
        .or(obj
            .downcast_mut::<Brick>()
            .map(|o| o as RefMut<dyn Bumpable>))
        .or(obj
            .downcast_mut::<UndergroundBrick>()
            .map(|o| o as RefMut<dyn Bumpable>))
}
