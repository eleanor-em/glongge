use num_traits::Float;

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
    run_test_cases();
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
            scene_handler.create_scene(TriangleScene);
            scene_handler.create_scene(RectangleScene);
            scene_handler.create_scene(ConcaveScene);
            scene_handler.create_scene(SplineScene);
            scene_handler.create_scene(MarioOverworldScene);
            scene_handler.create_scene(MarioUndergroundScene);
            // let name = TriangleScene.name();
            // let name = RectangleScene.name();
            // let name = ConcaveScene.name();
            // let name = SplineScene.name();
            let name = MarioOverworldScene.name();
            // let name = MarioUndergroundScene.name();
            scene_handler.set_initial_scene(name, 0);
            scene_handler
        })
}

fn run_test_cases() {
    // TODO: proper test cases...
    let a = Vec2 { x: 1.0, y: 1. };
    check!(a * 2. == Vec2 { x: 2.0, y: 2.0 });
    check!(2. * a == Vec2 { x: 2.0, y: 2.0 });
    check_lt!(f32::abs((a * 2.0 - a).x - 1.0), f32::epsilon());
    check_lt!(f32::abs((a * 2.0 - a).y - 1.0), f32::epsilon());
    check!(
        (Mat3x3::rotation(-1.0) * Mat3x3::rotation(0.5) * Mat3x3::rotation(0.5))
            .almost_eq(Mat3x3::one())
    );

    check_almost_eq!(
        Vec2::right().rotated(45_f32.to_radians()),
        Vec2 { x: 1.0, y: 1.0 }.normed()
    );
    check_almost_eq!(Vec2::right().rotated(90_f32.to_radians()), Vec2::down());
    check_almost_eq!(
        Vec2::right().rotated(135_f32.to_radians()),
        Vec2 { x: -1.0, y: 1.0 }.normed()
    );
    check_almost_eq!(Vec2::right().rotated(180_f32.to_radians()), Vec2::left());
    check_almost_eq!(
        Vec2::right().rotated(225_f32.to_radians()),
        Vec2 { x: -1.0, y: -1.0 }.normed()
    );
    check_almost_eq!(Vec2::right().rotated(270_f32.to_radians()), Vec2::up());
    check_almost_eq!(
        Vec2::right().rotated(315_f32.to_radians()),
        Vec2 { x: 1.0, y: -1.0 }.normed()
    );
    check_almost_eq!(Vec2::right().rotated(360_f32.to_radians()), Vec2::right());

    for vec in [Vec2::right(), Vec2::up(), Vec2::left(), Vec2::down()] {
        check_almost_eq!(
            vec.rotated(45_f32.to_radians()),
            vec.rotated((-315_f32).to_radians())
        );
        check_almost_eq!(
            vec.rotated(90_f32.to_radians()),
            vec.rotated((-270_f32).to_radians())
        );
        check_almost_eq!(
            vec.rotated(135_f32.to_radians()),
            vec.rotated((-225_f32).to_radians())
        );
        check_almost_eq!(
            vec.rotated(180_f32.to_radians()),
            vec.rotated((-180_f32).to_radians())
        );
        check_almost_eq!(
            vec.rotated(225_f32.to_radians()),
            vec.rotated((-135_f32).to_radians())
        );
        check_almost_eq!(
            vec.rotated(270_f32.to_radians()),
            vec.rotated((-90_f32).to_radians())
        );
        check_almost_eq!(
            vec.rotated(315_f32.to_radians()),
            vec.rotated((-45_f32).to_radians())
        );
    }
}
