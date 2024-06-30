#![feature(iterator_try_collect)]

use std::cell::RefCell;
use num_traits::{Float, One};
use rand::{
    distributions::Distribution,
    distributions::Uniform,
};

use anyhow::Result;

use crate::{
    core::{
        linalg::{Mat3x3, Vec2},
        vk_core::{VulkanoContext, WindowContext, WindowEventHandler},
    },
    gg::{
        core::SceneObject,
        sample::BasicRenderHandler,
    },
    scene::sample::SpinningTriangle,
};

mod assert;
mod core;
mod scene;
mod shader;
mod gg;

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

    const N: usize = 500;
    let mut rng = rand::thread_rng();
    let xs: Vec<f64> = Uniform::new(0.0, 1024.0).sample_iter(&mut rng).take(N).collect();
    let ys: Vec<f64> = Uniform::new(0.0, 768.0).sample_iter(&mut rng).take(N).collect();
    let vxs: Vec<f64> = Uniform::new(-1.0, 1.0).sample_iter(&mut rng).take(N).collect();
    let vys: Vec<f64> = Uniform::new(-1.0, 1.0).sample_iter(&mut rng).take(N).collect();
    let objects: Vec<_> = (0..N)
        .map(|i| {
            let pos = Vec2 { x: xs[i], y: ys[i] };
            let vel = Vec2 { x: vxs[i], y: vys[i] };
            RefCell::new(Box::new(SpinningTriangle::new(pos, vel.normed())) as Box<dyn SceneObject>)
        })
        .collect();

    // TODO: replace with Builder pattern
    let mut handler = BasicRenderHandler::new(objects, &window_ctx, &ctx)?;
    handler.start_update_thread();

    let (event_loop, window) = window_ctx.consume();
    WindowEventHandler::new(window, ctx, handler).run(event_loop);
    Ok(())
}

fn run_test_cases() {
    // TODO: proper test cases...
    let a = Vec2 { x: 1.0, y: 1.0 };
    crate::check!((a * 2.0).almost_eq(Vec2 { x: 2.0, y: 2.0 }));
    crate::check!((2.0 * a).almost_eq(Vec2 { x: 2.0, y: 2.0 }));
    crate::check_lt!(f64::abs((a * 2.0 - a).x - 1.0), f64::epsilon());
    crate::check_lt!(f64::abs((a * 2.0 - a).y - 1.0), f64::epsilon());
    crate::check!(
        (Mat3x3::rotation(-1.0) * Mat3x3::rotation(0.5) * Mat3x3::rotation(0.5))
            .almost_eq(Mat3x3::one())
    );
}
