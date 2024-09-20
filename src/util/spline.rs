use std::iter;
use egui::{Align, Layout};
use rand::Rng;
use glongge_derive::{partially_derive_scene_object, register_scene_object};
use crate::core::{ObjectTypeEnum, prelude::*};
use crate::core::input::MouseButton;
use crate::core::scene::{GuiInsideClosure, GuiObject};
use crate::util::canvas::Canvas;

#[derive(Clone, Default)]
pub struct GgInternalSpline {
    control_points: Vec<Vec2>,
}

impl GgInternalSpline {
    pub fn new(control_points: Vec<Vec2>) -> Self {
        Self { control_points }
    }

    pub fn point(&self, t: f64) -> Vec2 {
        let mut points = self.control_points.clone();
        while points.len() > 1 {
            points = points.into_iter()
                .tuple_windows()
                .map(|(u, v)| u.lerp(v, t))
                .collect_vec();
        }
        *points.first().unwrap()
    }

    pub fn keep_last_n(&mut self, n: usize) {
        for _ in 0..(self.control_points.len().saturating_sub(n)) {
            self.control_points.remove(0);
        }
    }
    pub fn push(&mut self, point: Vec2) {
        self.control_points.push(point);
    }
    pub fn push_front(&mut self, point: Vec2) {
        self.control_points.insert(0, point);
    }
}

#[register_scene_object]
pub struct GgInternalInteractiveSpline {
    spline: GgInternalSpline,
    line_points: Vec<Vec2>,
    gui_selected: bool,
    selected_control_point: Option<usize>,
}

impl GgInternalInteractiveSpline {
    const RADIUS: f64 = 3.;

    pub fn spline(&self) -> &GgInternalSpline { &self.spline }
    pub fn spline_mut(&mut self) -> &mut GgInternalSpline { &mut self.spline }
    pub fn recalculate(&mut self) {
        const N: u32 = 1000;
        self.line_points = vec![self.spline.control_points[0]];
        for i in 1..=N {
            let next = self.spline.point(f64::from(i) / f64::from(N));
            self.line_points.push(next);
        }
    }

    pub fn draw_to_canvas(&self, canvas: &mut Canvas, width: f64, colour: Colour) {
        for (&u, &v) in self.line_points.iter().tuple_windows() {
            canvas.line(u, v, width, colour);
        }
    }
}

#[partially_derive_scene_object]
impl<ObjectType: ObjectTypeEnum> SceneObject<ObjectType> for GgInternalInteractiveSpline {
    fn name(&self) -> String {
        "InteractiveSpline".to_string()
    }
    fn get_type(&self) -> ObjectType { ObjectType::gg_interactive_spline() }
    fn on_ready(&mut self, _ctx: &mut UpdateContext<ObjectType>) {
        let mut rng = rand::thread_rng();
        self.spline = GgInternalSpline::new(iter::from_fn(|| {
            Some(Vec2::from([rng.gen_range(0.0..200.0), rng.gen_range(0.0..200.0)])
                + 200. * Vec2::one())
        }).take(3).collect_vec());
        self.recalculate();
    }

    fn on_update(&mut self, ctx: &mut UpdateContext<ObjectType>) {
        if ctx.input().pressed(KeyCode::KeyG) {
            ctx.viewport_mut().set_global_scale_factor(1.);
        }

        if self.gui_selected {
            let mouse_pos = ctx.input().screen_mouse_pos();
            let mouse_pressed = ctx.input().mouse_pressed(MouseButton::Primary);
            let mouse_double_clicked = ctx.input().mouse_double_clicked(MouseButton::Primary);
            if ctx.input().mouse_released(MouseButton::Primary) {
                self.selected_control_point = None;
            }
            if let Some(i) = self.selected_control_point {
                self.spline.control_points[i] = mouse_pos;
                self.recalculate();
            }
            if let Some(i) = self.spline.control_points.iter().tuple_windows().enumerate()
                .filter(|(_, (u, v))| {
                    mouse_double_clicked && mouse_pos.dist_to_line(**u, **v) < 2.
                })
                .map(|(i, _)| i + 1)
                .next()
            {
                self.spline.control_points.insert(i, mouse_pos);
                self.recalculate();
            }

            let mut canvas = ctx.object_mut().first_other_as_mut::<Canvas>().unwrap();
            self.draw_to_canvas(&mut canvas, 1., Colour::green());
            for (&u, &v) in self.spline.control_points.iter().tuple_windows() {
                canvas.line(u, v, 0.5, if mouse_pos.dist_to_line(u, v) < 2. {
                    Colour::blue().scaled(0.8)
                } else {
                    Colour::cyan()
                });
            }
            for (i, &v) in self.spline.control_points.iter().enumerate() {
                if mouse_pos.dist_squared(v) <= Self::RADIUS * Self::RADIUS {
                    canvas.circle(v, Self::RADIUS, 20, Colour::blue());
                    if mouse_pressed {
                        self.selected_control_point = Some(i);
                    }
                } else {
                    canvas.circle(v, Self::RADIUS, 20, Colour::cyan());
                }
            }
        }
    }

    fn as_gui_object(&mut self) -> Option<&mut dyn GuiObject<ObjectType>> {
        Some(self)
    }
}

impl<ObjectType: ObjectTypeEnum> GuiObject<ObjectType> for GgInternalInteractiveSpline {
    fn on_gui(&mut self, _ctx: &UpdateContext<ObjectType>, selected: bool) -> Box<GuiInsideClosure> {
        self.gui_selected = selected;
        let string_desc = self.spline.control_points.iter()
            .map(|v| format!("\t{v:?},\n"))
            .reduce(|acc: String, x: String| acc + &x)
            .unwrap_or_default();
        Box::new(move |ui| {
            ui.with_layout(Layout::top_down(Align::Center), |ui| {
                if ui.button("Copy as Vec")
                    .clicked() {
                    ui.output_mut(|o| {
                        o.copied_text = format!("vec![\n{string_desc}]\n");
                    });
                }
            });
        })
    }
}

pub use GgInternalSpline as Spline;
pub use GgInternalInteractiveSpline as InteractiveSpline;
