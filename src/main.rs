#![feature(iterator_try_collect)]

use anyhow::Result;
use num_traits::{Float, One};

use crate::{
    assert::*,
    core::{
        linalg::{Mat3x3, Vec2},
        vk_core::{VulkanoContext, WindowContext, WindowEventHandler},
    },
    gg::sample::BasicRenderHandler,
    scene::sample::create_spinning_triangle_scene,
};

mod assert;
mod core;
mod gg;
mod scene;
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

    let window_ctx = WindowContext::new()?;
    let ctx = VulkanoContext::new(&window_ctx)?;
    main_test(window_ctx, ctx)
}

// this is the "main" test (i.e. used for active dev)
fn main_test(window_ctx: WindowContext, ctx: VulkanoContext) -> Result<()> {
    run_test_cases();

    // TODO: replace with Builder pattern
    let handler = BasicRenderHandler::new(&window_ctx, &ctx)?;
    let mut scene = create_spinning_triangle_scene(&handler);
    scene.run();

    let (event_loop, window) = window_ctx.consume();
    WindowEventHandler::new(window, ctx, handler).run(event_loop);
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
