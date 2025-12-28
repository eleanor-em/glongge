use glongge::core::prelude::*;
use glongge::core::scene::Scene;
use glongge::util::GgContextBuilder;

pub mod examples;

use crate::examples::concave::ConcaveScene;
use crate::examples::spline::SplineScene;
#[allow(unused_imports)]
use crate::examples::{
    mario::{MarioOverworldScene, MarioUndergroundScene},
    rectangle::RectangleScene,
    triangle::TriangleScene,
};

fn main() -> Result<()> {
    GgContextBuilder::new([1280, 800])?
        .with_extra_scale_factor(2.0)
        .build_and_run_window(|scene_handler| {
            let resource_handler = scene_handler.resource_handler();
            for sound in [
                "res/overworld.ogg",
                "res/underground.ogg",
                "res/jump-small.wav",
                "res/stomp.wav",
                "res/death.wav",
                "res/pipe.wav",
                "res/bump.wav",
                "res/flagpole.wav",
                "res/stage-clear.wav",
            ] {
                resource_handler.sound.spawn_load_file(sound);
            }
            let _stored_textures = [
                "res/mario_sheet.png",
                "res/world_sheet.png",
                "res/enemies_sheet.png",
            ]
            .into_iter()
            .map(|filename| resource_handler.texture.wait_load_file(filename).unwrap())
            .collect_vec();
            resource_handler.wait_all().unwrap();
            let mut scene_handler = scene_handler.build();
            scene_handler.create_scene(TriangleScene)?;
            scene_handler.create_scene(RectangleScene)?;
            scene_handler.create_scene(ConcaveScene)?;
            scene_handler.create_scene(SplineScene)?;
            scene_handler.create_scene(MarioOverworldScene)?;
            scene_handler.create_scene(MarioUndergroundScene)?;
            // let name = TriangleScene.name();
            // let name = RectangleScene.name();
            // let name = ConcaveScene.name();
            // let name = SplineScene.name();
            let name = MarioOverworldScene.name();
            // let name = MarioUndergroundScene.name();
            scene_handler.set_initial_scene(&name, 0)?;
            Ok(scene_handler)
        })
}
