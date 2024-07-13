#![feature(iterator_try_collect)]

use num_traits::{Float, One};

use anyhow::Result;

use crate::{
    assert::*,
    core::{
        input::InputHandler,
        linalg::{Mat3x3, Vec2},
        vk_core::{VulkanoContext, WindowContext, WindowEventHandler},
    },
    gg::{
        render::BasicRenderHandler,
        scene::sample::{rectangle, triangle}
    },
    resource::ResourceHandler
};

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
    let resource_handler = ResourceHandler::new(ctx.clone());
    let render_handler = BasicRenderHandler::new(&window_ctx, &ctx, resource_handler.clone())?;
    let input_handler = InputHandler::new();
    let mut scene = rectangle::create_scene(
        resource_handler.clone(),
        render_handler.clone(),
        input_handler.clone());
    let mut _scene = triangle::create_scene(
        resource_handler.clone(),
        render_handler.clone(),
        input_handler.clone()
    );
    scene.run();
    let (event_loop, window) = window_ctx.consume();
    WindowEventHandler::new(window, ctx, render_handler, input_handler, resource_handler)
        .run(event_loop, false);
    Ok(())
}

fn run_test_cases() {
    // TODO: proper test cases...
    let a = Vec2 { x: 1.0, y: 1.0 };
    check!((a * 2.0).almost_eq(Vec2 { x: 2.0, y: 2.0 }));
    check!((2.0 * a).almost_eq(Vec2 { x: 2.0, y: 2.0 }));
    check_lt!(f64::abs((a * 2.0 - a).x - 1.0), f64::epsilon());
    check_lt!(f64::abs((a * 2.0 - a).y - 1.0), f64::epsilon());
    check!(
        (Mat3x3::rotation(-1.0) * Mat3x3::rotation(0.5) * Mat3x3::rotation(0.5))
            .almost_eq(Mat3x3::one())
    );

    check_almost_eq!(Vec2::right().rotated(45_f64.to_radians()), Vec2 { x: 1.0, y: 1.0 }.normed());
    check_almost_eq!(Vec2::right().rotated(90_f64.to_radians()), Vec2::down());
    check_almost_eq!(Vec2::right().rotated(135_f64.to_radians()), Vec2 { x: -1.0, y: 1.0 }.normed());
    check_almost_eq!(Vec2::right().rotated(180_f64.to_radians()), Vec2::left());
    check_almost_eq!(Vec2::right().rotated(225_f64.to_radians()), Vec2 { x: -1.0, y: -1.0 }.normed());
    check_almost_eq!(Vec2::right().rotated(270_f64.to_radians()), Vec2::up());
    check_almost_eq!(Vec2::right().rotated(315_f64.to_radians()), Vec2 { x: 1.0, y: -1.0 }.normed());
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
