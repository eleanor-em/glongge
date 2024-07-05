#![feature(iterator_try_collect)]

use num_traits::{Float, One};

use anyhow::Result;

use crate::{
    assert::*,
    core::{
        linalg::{Mat3x3, Vec2},
        vk_core::{VulkanoContext, WindowContext, WindowEventHandler},
    },
    gg::{
        sample::BasicRenderHandler,
        scene::sample::{rectangle, triangle}
    }
};
use crate::core::input::InputHandler;

mod assert;
mod core;
mod gg;
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
    let render_handler = BasicRenderHandler::new(&window_ctx, &ctx)?;
    let input_handler = InputHandler::new();
    let mut scene = rectangle::create_scene(&render_handler, input_handler.clone());
    let mut _scene = triangle::create_scene(&render_handler, input_handler.clone());
    scene.run();
    let (event_loop, window) = window_ctx.consume();
    WindowEventHandler::new(window, ctx, render_handler, input_handler).run(event_loop);
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
}
