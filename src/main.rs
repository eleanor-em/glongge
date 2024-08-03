#![feature(const_fn_floating_point_arithmetic)]
include!(concat!(env!("OUT_DIR"), "/object_type.rs"));

use num_traits::{Float, One};
use tracing_subscriber::fmt::time::OffsetTime;

use glongge::core::{
    input::InputHandler,
    prelude::*,
    render::RenderHandler,
    scene::SceneHandler,
    scene::Scene,
    vk::{VulkanoContext, WindowContext, WindowEventHandler},
    ObjectTypeEnum,
};
use glongge::core::util::UniqueShared;
use glongge::gui::GuiContext;
use glongge::shader::{BasicShader, Shader, SpriteShader, WireframeShader};

use crate::object_type::ObjectType;

pub mod examples;

#[allow(unused_imports)]
use crate::examples::{
    triangle::TriangleScene,
    rectangle::RectangleScene,
    mario::{MarioOverworldScene, MarioUndergroundScene},
};
use crate::examples::concave::ConcaveScene;

fn main() -> Result<()> {
    let logfile = std::fs::OpenOptions::new()
        .write(true)
        .truncate(true)
        .create(true)
        .open("run.log")
        .unwrap();
    let timer = OffsetTime::new(
        time::UtcOffset::UTC,
        time::macros::format_description!("[hour]:[minute]:[second].[subsecond digits:6]")
    );
    tracing_subscriber::fmt()
        .event_format(
            tracing_subscriber::fmt::format()
                .with_target(false)
                .with_file(true)
                .with_line_number(true)
                .with_timer(timer)
        )
        .with_writer(logfile)
        .init();
    run_test_cases();

    let window_ctx = WindowContext::new()?;
    let gui_ctx = GuiContext::default();
    let vk_ctx = VulkanoContext::new(&window_ctx)?;
    let mut resource_handler = ResourceHandler::new(&vk_ctx)?;
    ObjectType::preload_all(&mut resource_handler)?;

    let viewport = UniqueShared::new(window_ctx.create_default_viewport());
    let shaders: Vec<UniqueShared<dyn Shader>> = vec![
        SpriteShader::new(vk_ctx.clone(), viewport.clone(), resource_handler.clone())?,
        WireframeShader::new(vk_ctx.clone(), viewport.clone())?,
        BasicShader::new(vk_ctx.clone(), viewport.clone())?,
    ];
    let render_handler = RenderHandler::new(
        &vk_ctx,
        gui_ctx.clone(),
        viewport.clone(),
        shaders,
    )?
        // .with_global_scale_factor(1.);
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
            scene_handler.create_scene(TriangleScene);
            scene_handler.create_scene(RectangleScene);
            scene_handler.create_scene(ConcaveScene);
            scene_handler.create_scene(MarioOverworldScene);
            scene_handler.create_scene(MarioUndergroundScene);
            // let name = TriangleScene.name();
            // let name = RectangleScene.name();
            // let name = ConcaveScene.name();
            let name = MarioOverworldScene.name();
            // let name = MarioUndergroundScene.name();
            scene_handler.consume_with_scene(name, 0);
        });
    }

    let (event_loop, window) = window_ctx.consume();
    WindowEventHandler::new(window, vk_ctx, gui_ctx.clone(), render_handler, input_handler, resource_handler)
        .consume(event_loop);
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
        check_almost_eq!(vec.rotated(45_f64.to_radians()), vec.rotated((-315_f64).to_radians()));
        check_almost_eq!(vec.rotated(90_f64.to_radians()), vec.rotated((-270_f64).to_radians()));
        check_almost_eq!(vec.rotated(135_f64.to_radians()), vec.rotated((-225_f64).to_radians()));
        check_almost_eq!(vec.rotated(180_f64.to_radians()), vec.rotated((-180_f64).to_radians()));
        check_almost_eq!(vec.rotated(225_f64.to_radians()), vec.rotated((-135_f64).to_radians()));
        check_almost_eq!(vec.rotated(270_f64.to_radians()), vec.rotated((-90_f64).to_radians()));
        check_almost_eq!(vec.rotated(315_f64.to_radians()), vec.rotated((-45_f64).to_radians()));
    }
}
