use crate::core::input::MouseButton;
use crate::core::scene::{GuiInsideClosure, GuiObject};
use crate::core::{ObjectTypeEnum, prelude::*};
use crate::util::canvas::Canvas;
use egui::{Align, Layout};
use glongge_derive::{partially_derive_scene_object, register_scene_object};
use rand::Rng;
use rand::thread_rng;

#[derive(Clone, Default)]
pub struct GgInternalSpline {
    control_points: Vec<Vec2>,
}

impl GgInternalSpline {
    pub fn new(control_points: Vec<Vec2>) -> Self {
        Self { control_points }
    }
    pub fn new_joined(&self) -> Option<Self> {
        let p = self.get(self.len() - 2)?;
        let s = self.last()?;
        let q = 2 * s - p;
        Some(Self {
            control_points: vec![s, q],
        })
    }

    pub fn point(&self, t: f32) -> Option<Vec2> {
        let mut points = self.control_points.clone();
        while points.len() > 1 {
            points = points
                .into_iter()
                .tuple_windows()
                .map(|(u, v)| u.lerp(v, t))
                .collect_vec();
        }
        points.first().copied()
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

    pub fn get(&self, i: usize) -> Option<Vec2> {
        self.control_points.get(i).copied()
    }
    pub fn last(&self) -> Option<Vec2> {
        self.control_points.last().copied()
    }

    pub fn closest_to(&self, point: Vec2) -> Option<f32> {
        let mut t = 0.;
        let mut rv = 0.;
        let mut best_distance = f32::MAX;
        while t <= 1. {
            let x = self.point(t)?;
            if x.dist(point) < best_distance {
                best_distance = x.dist(point);
                rv = t;
            }
            t += 0.001;
        }
        Some(rv)
    }

    pub fn len(&self) -> usize {
        self.control_points.len()
    }
    pub fn is_empty(&self) -> bool {
        self.control_points.is_empty()
    }
}

#[register_scene_object]
pub struct GgInternalInteractiveSpline {
    spline: GgInternalSpline,
    line_points: Vec<Vec2>,
    gui_selected: bool,
    selected_control_point: Option<usize>,
    force_visible: bool,
    colour: Colour,
}

impl GgInternalInteractiveSpline {
    const RADIUS: f32 = 3.;

    pub fn spline(&self) -> &GgInternalSpline {
        &self.spline
    }
    pub fn spline_mut(&mut self) -> &mut GgInternalSpline {
        &mut self.spline
    }
    pub fn recalculate(&mut self) {
        if !self.gui_selected {
            warn!("called recalculate() but not shown; skipping to save processing");
            return;
        }
        if self.spline.is_empty() {
            self.line_points = Vec::new();
        } else {
            const N: u32 = 100;
            self.line_points = vec![self.spline.control_points[0]];
            for i in 1..=N {
                let next = self.spline.point(i as f32 / N as f32).unwrap();
                self.line_points.push(next);
            }
        }
    }

    pub fn draw_to_canvas(&self, canvas: &mut Canvas, width: f32, colour: Colour) {
        for (&u, &v) in self.line_points.iter().tuple_windows() {
            canvas.line(u, v, width, colour);
        }
    }

    pub fn force_visible(&mut self) {
        self.gui_selected = true;
        self.force_visible = true;
    }
}

#[partially_derive_scene_object]
impl<ObjectType: ObjectTypeEnum> SceneObject<ObjectType> for GgInternalInteractiveSpline {
    fn name(&self) -> String {
        "InteractiveSpline".to_string()
    }
    fn gg_type_enum(&self) -> ObjectType {
        ObjectType::gg_interactive_spline()
    }
    fn on_ready(&mut self, _ctx: &mut UpdateContext<ObjectType>) {
        self.colour = match thread_rng().gen_range(0..6) {
            0 => Colour::red(),
            1 => Colour::green(),
            2 => Colour::blue(),
            3 => Colour::cyan(),
            4 => Colour::magenta(),
            5 => Colour::yellow(),
            _ => panic!(),
        }
    }

    fn on_update(&mut self, ctx: &mut UpdateContext<ObjectType>) {
        self.gui_selected |= self.force_visible;
        if self.gui_selected {
            if let Some(mouse_pos) = ctx.input().screen_mouse_pos() {
                let mouse_pressed = ctx.input().mouse_pressed(MouseButton::Primary);
                let mouse_double_clicked = ctx.input().mouse_double_clicked(MouseButton::Primary);
                if ctx.input().mouse_released(MouseButton::Primary) {
                    self.selected_control_point = None;
                }
                if let Some(i) = self.selected_control_point {
                    self.spline.control_points[i] = mouse_pos;
                    self.recalculate();
                }
                if let Some(i) = self
                    .spline
                    .control_points
                    .iter()
                    .tuple_windows()
                    .enumerate()
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
                self.draw_to_canvas(&mut canvas, 1., self.colour);
                for (&u, &v) in self.spline.control_points.iter().tuple_windows() {
                    canvas.line(
                        u,
                        v,
                        0.5,
                        if mouse_pos.dist_to_line(u, v) < 2. {
                            Colour::blue().scaled(0.8)
                        } else {
                            Colour::cyan()
                        },
                    );
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
    }

    fn as_gui_object(&mut self) -> Option<&mut dyn GuiObject<ObjectType>> {
        Some(self)
    }
}

impl<ObjectType: ObjectTypeEnum> GuiObject<ObjectType> for GgInternalInteractiveSpline {
    fn on_gui(
        &mut self,
        _ctx: &UpdateContext<ObjectType>,
        selected: bool,
    ) -> Box<GuiInsideClosure> {
        self.gui_selected = selected || self.force_visible;
        let string_desc = self
            .spline
            .control_points
            .iter()
            .map(|v| format!("\t{v:?},\n"))
            .reduce(|acc: String, x: String| acc + &x)
            .unwrap_or_default();
        Box::new(move |ui| {
            ui.with_layout(Layout::top_down(Align::Center), |ui| {
                if ui.button("Copy as Vec").clicked() {
                    ui.output_mut(|o| {
                        o.copied_text = format!("vec![\n{string_desc}]\n");
                    });
                }
            });
        })
    }
}

pub use GgInternalInteractiveSpline as InteractiveSpline;
pub use GgInternalSpline as Spline;
