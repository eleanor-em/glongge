#![feature(iterator_try_collect)]
#![feature(arbitrary_self_types)]
#![feature(mapped_lock_guards)]
#![feature(const_fn_floating_point_arithmetic)]

#[allow(unused_imports)]
use crate::core::prelude::*;

use num_traits::{Float, One};

use crate::{
    core::{
        input::InputHandler,
        linalg::{Mat3x3, Vec2},
        vk_core::{VulkanoContext, WindowContext, WindowEventHandler},
    },
    gg::{
        render::BasicRenderHandler,
        scene::SceneHandler
    },
    resource::ResourceHandler
};
#[allow(unused_imports)]
use crate::gg::scene::sample::*;
#[allow(unused_imports)]
use crate::gg::scene::sample::mario::{MarioScene, MarioUndergroundScene};
#[allow(unused_imports)]
use crate::gg::scene::sample::rectangle::RectangleScene;
#[allow(unused_imports)]
use crate::gg::scene::sample::triangle::TriangleScene;
use crate::gg::scene::Scene;

mod assert;
mod core;
mod gg;
mod resource;
mod shader;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .event_format(
            tracing_subscriber::fmt::format()
                .with_target(false)
                .with_file(true)
                .with_line_number(true),
        )
        .init();
    run_test_cases();

    let window_ctx = WindowContext::new()?;
    let ctx = VulkanoContext::new(&window_ctx)?;
    let resource_handler = ResourceHandler::new(ctx.clone())?;
    let render_handler = BasicRenderHandler::new(&window_ctx, &ctx, resource_handler.clone())?
        .with_global_scale_factor(2.);
    let input_handler = InputHandler::new();
    {
        let input_handler = input_handler.clone();
        let resource_handler = resource_handler.clone();
        let render_handler = render_handler.clone();
        std::thread::spawn(move || {
            let mut scene_handler = SceneHandler::new(
                input_handler,
                resource_handler,
                render_handler
            );
            // scene_handler.create_scene(TriangleScene); let name = TriangleScene.name();
            // scene_handler.create_scene(RectangleScene); let name = RectangleScene.name();
            scene_handler.create_scene(MarioScene);
            scene_handler.create_scene(MarioUndergroundScene);
            let name = MarioScene.name();
            // let name = MarioUndergroundScene{}.name();
            scene_handler.consume_with_scene(name, 0);
        });
    }
    let (event_loop, window) = window_ctx.consume();
    WindowEventHandler::new(window, ctx, render_handler, input_handler, resource_handler)
        .consume(event_loop, false);
    Ok(())
}

fn run_test_cases() {
    // TODO: proper test cases...
    let a = Vec2 { x: 1., y: 1. };
    check!(a * 2. == Vec2 { x: 2., y: 2. });
    check!(2. * a == Vec2 { x: 2., y: 2. });
    check_lt!(f64::abs((a * 2. - a).x - 1.), f64::epsilon());
    check_lt!(f64::abs((a * 2. - a).y - 1.), f64::epsilon());
    check!(
        (Mat3x3::rotation(-1.) * Mat3x3::rotation(0.5) * Mat3x3::rotation(0.5))
            .almost_eq(Mat3x3::one())
    );

    check_almost_eq!(Vec2::right().rotated(45_f64.to_radians()), Vec2 { x: 1., y: 1. }.normed());
    check_almost_eq!(Vec2::right().rotated(90_f64.to_radians()), Vec2::down());
    check_almost_eq!(Vec2::right().rotated(135_f64.to_radians()), Vec2 { x: -1., y: 1. }.normed());
    check_almost_eq!(Vec2::right().rotated(180_f64.to_radians()), Vec2::left());
    check_almost_eq!(Vec2::right().rotated(225_f64.to_radians()), Vec2 { x: -1., y: -1. }.normed());
    check_almost_eq!(Vec2::right().rotated(270_f64.to_radians()), Vec2::up());
    check_almost_eq!(Vec2::right().rotated(315_f64.to_radians()), Vec2 { x: 1., y: -1. }.normed());
    check_almost_eq!(Vec2::right().rotated(360_f64.to_radians()), Vec2::right());

    for vec in [Vec2::right(), Vec2::up(), Vec2::left(), Vec2::down()] {
        check_almost_eq!(vec.rotated(45_f64.to_radians()), vec.rotated(-315_f64.to_radians()));
        check_almost_eq!(vec.rotated(90_f64.to_radians()), vec.rotated(-270_f64.to_radians()));
        check_almost_eq!(vec.rotated(135_f64.to_radians()), vec.rotated(-225_f64.to_radians()));
        check_almost_eq!(vec.rotated(180_f64.to_radians()), vec.rotated(-180_f64.to_radians()));
        check_almost_eq!(vec.rotated(225_f64.to_radians()), vec.rotated(-135_f64.to_radians()));
        check_almost_eq!(vec.rotated(270_f64.to_radians()), vec.rotated(-90_f64.to_radians()));
        check_almost_eq!(vec.rotated(315_f64.to_radians()), vec.rotated(-45_f64.to_radians()));
    }
}
