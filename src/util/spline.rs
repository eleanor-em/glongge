use crate::core::input::MouseButton;
use crate::core::prelude::*;
use crate::core::scene::{GuiCommand, GuiObject};
use crate::util::canvas::Canvas;
use egui::{Align, Layout, OutputCommand};
use glongge_derive::partially_derive_scene_object;
use rand::Rng;
use rand::rng;

#[derive(Clone, Default)]
pub struct GgInternalSpline {
    control_points: Vec<Vec2>,
}

impl GgInternalSpline {
    pub fn new(control_points: Vec<Vec2>) -> Self {
        Self { control_points }
    }
    pub fn new_joined(&self) -> Option<Self> {
        let ix = self.len().checked_sub(2)?;
        // After checked_sub succeeds, we know len >= 2, so these cannot fail
        let p = self.get(ix).expect("unreachable");
        let s = self.last().expect("unreachable");
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
    pub fn replace_last(&mut self, point: Vec2) {
        if let Some(existing) = self.control_points.last_mut() {
            *existing = point;
        }
    }

    pub fn get(&self, i: usize) -> Option<Vec2> {
        self.control_points.get(i).copied()
    }
    pub fn last(&self) -> Option<Vec2> {
        self.control_points.last().copied()
    }

    pub fn closest_to(&self, point: Vec2) -> Option<f32> {
        let mut t = 0.0;
        let mut rv = 0.0;
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
    pub fn closest_point_to(&self, point: Vec2) -> Option<Vec2> {
        self.closest_to(point).and_then(|t| self.point(t))
    }

    pub fn len(&self) -> usize {
        self.control_points.len()
    }
    pub fn is_empty(&self) -> bool {
        self.control_points.is_empty()
    }
}

#[derive(Default)]
pub struct GgInternalInteractiveSpline {
    spline: GgInternalSpline,
    line_points: Vec<Vec2>,
    gui_selected: bool,
    selected_control_point: Option<usize>,
    force_visible: bool,
    colour: Colour,
}

#[cfg_attr(coverage_nightly, coverage(off))]
impl GgInternalInteractiveSpline {
    const RADIUS: f32 = 3.0;

    pub fn spline(&self) -> &GgInternalSpline {
        &self.spline
    }
    pub fn spline_mut(&mut self) -> &mut GgInternalSpline {
        &mut self.spline
    }
    pub fn recalculate(&mut self) {
        // This whole gui_selected business is dodgy as heck.
        if !self.gui_selected {
            warn!("called recalculate() but not selected; skipping to save processing");
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
#[cfg_attr(coverage_nightly, coverage(off))]
impl SceneObject for GgInternalInteractiveSpline {
    fn gg_type_name(&self) -> String {
        "InteractiveSpline".to_string()
    }
    fn on_ready(&mut self, _ctx: &mut UpdateContext) {
        self.colour = match rng().random_range(0..6) {
            0 => Colour::red(),
            1 => Colour::green(),
            2 => Colour::blue(),
            3 => Colour::cyan(),
            4 => Colour::magenta(),
            5 => Colour::yellow(),
            _ => panic!(),
        }
    }

    fn on_update(&mut self, ctx: &mut UpdateContext) {
        self.gui_selected |= self.force_visible;
        if self.gui_selected {
            let mouse_pos = ctx.input().screen_mouse_pos().unwrap_or_default();
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
                    mouse_double_clicked && mouse_pos.dist_to_line(**u, **v) < 2.0
                })
                .map(|(i, _)| i + 1)
                .next()
            {
                self.spline.control_points.insert(i, mouse_pos);
                self.recalculate();
            }

            let mut canvas = ctx.object_mut().first_other_as_mut::<Canvas>().unwrap();
            self.draw_to_canvas(&mut canvas, 1.0, self.colour);
            for (&u, &v) in self.spline.control_points.iter().tuple_windows() {
                canvas.line(
                    u,
                    v,
                    0.5,
                    if mouse_pos.dist_to_line(u, v) < 2.0 {
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

    fn as_gui_object(&mut self) -> Option<&mut dyn GuiObject> {
        Some(self)
    }
}

#[cfg_attr(coverage_nightly, coverage(off))]
impl GuiObject for GgInternalInteractiveSpline {
    fn on_gui(&mut self, _ctx: &UpdateContext, selected: bool) -> GuiCommand {
        self.gui_selected = selected || self.force_visible;
        let string_desc = self
            .spline
            .control_points
            .iter()
            .map(|v| format!("\t{v:?},\n"))
            .reduce(|acc: String, x: String| acc + &x)
            .unwrap_or_default();
        GuiCommand::new(move |ui| {
            ui.with_layout(Layout::top_down(Align::Center), |ui| {
                if ui.button("Copy as Vec").clicked() {
                    ui.output_mut(|o| {
                        o.commands
                            .push(OutputCommand::CopyText(format!("vec![\n{string_desc}]\n")));
                    });
                }
            });
        })
    }
}

pub use GgInternalInteractiveSpline as InteractiveSpline;
pub use GgInternalSpline as Spline;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_and_default() {
        let spline = GgInternalSpline::new(vec![Vec2::zero(), Vec2::one()]);
        assert_eq!(spline.len(), 2);

        let default = GgInternalSpline::default();
        assert!(default.is_empty());
    }

    #[test]
    fn test_len_and_is_empty() {
        let empty = GgInternalSpline::new(vec![]);
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);

        let spline = GgInternalSpline::new(vec![Vec2::zero()]);
        assert!(!spline.is_empty());
        assert_eq!(spline.len(), 1);
    }

    #[test]
    fn test_get_and_last() {
        let spline = GgInternalSpline::new(vec![
            Vec2 { x: 1.0, y: 2.0 },
            Vec2 { x: 3.0, y: 4.0 },
            Vec2 { x: 5.0, y: 6.0 },
        ]);

        assert_eq!(spline.get(0), Some(Vec2 { x: 1.0, y: 2.0 }));
        assert_eq!(spline.get(1), Some(Vec2 { x: 3.0, y: 4.0 }));
        assert_eq!(spline.get(2), Some(Vec2 { x: 5.0, y: 6.0 }));
        assert_eq!(spline.get(3), None);

        assert_eq!(spline.last(), Some(Vec2 { x: 5.0, y: 6.0 }));
    }

    #[test]
    fn test_last_empty() {
        let spline = GgInternalSpline::new(vec![]);
        assert_eq!(spline.last(), None);
    }

    #[test]
    fn test_push() {
        let mut spline = GgInternalSpline::new(vec![Vec2::zero()]);
        spline.push(Vec2::one());
        assert_eq!(spline.len(), 2);
        assert_eq!(spline.last(), Some(Vec2::one()));
    }

    #[test]
    fn test_push_front() {
        let mut spline = GgInternalSpline::new(vec![Vec2::one()]);
        spline.push_front(Vec2::zero());
        assert_eq!(spline.len(), 2);
        assert_eq!(spline.get(0), Some(Vec2::zero()));
        assert_eq!(spline.get(1), Some(Vec2::one()));
    }

    #[test]
    fn test_replace_last() {
        let mut spline = GgInternalSpline::new(vec![Vec2::zero(), Vec2::one()]);
        spline.replace_last(Vec2 { x: 5.0, y: 5.0 });
        assert_eq!(spline.last(), Some(Vec2 { x: 5.0, y: 5.0 }));
    }

    #[test]
    fn test_replace_last_empty() {
        let mut spline = GgInternalSpline::new(vec![]);
        spline.replace_last(Vec2::one()); // Should not panic
        assert!(spline.is_empty());
    }

    #[test]
    fn test_keep_last_n() {
        let mut spline = GgInternalSpline::new(vec![
            Vec2 { x: 1.0, y: 0.0 },
            Vec2 { x: 2.0, y: 0.0 },
            Vec2 { x: 3.0, y: 0.0 },
            Vec2 { x: 4.0, y: 0.0 },
        ]);
        spline.keep_last_n(2);
        assert_eq!(spline.len(), 2);
        assert_eq!(spline.get(0), Some(Vec2 { x: 3.0, y: 0.0 }));
        assert_eq!(spline.get(1), Some(Vec2 { x: 4.0, y: 0.0 }));
    }

    #[test]
    fn test_keep_last_n_more_than_len() {
        let mut spline = GgInternalSpline::new(vec![Vec2::zero(), Vec2::one()]);
        spline.keep_last_n(10);
        assert_eq!(spline.len(), 2); // Unchanged
    }

    #[test]
    fn test_point_empty() {
        let spline = GgInternalSpline::new(vec![]);
        assert_eq!(spline.point(0.5), None);
    }

    #[test]
    fn test_point_single() {
        let spline = GgInternalSpline::new(vec![Vec2 { x: 5.0, y: 5.0 }]);
        assert_eq!(spline.point(0.0), Some(Vec2 { x: 5.0, y: 5.0 }));
        assert_eq!(spline.point(0.5), Some(Vec2 { x: 5.0, y: 5.0 }));
        assert_eq!(spline.point(1.0), Some(Vec2 { x: 5.0, y: 5.0 }));
    }

    #[test]
    fn test_point_linear() {
        // Two points = linear interpolation
        let spline = GgInternalSpline::new(vec![Vec2::zero(), Vec2 { x: 10.0, y: 10.0 }]);

        let p0 = spline.point(0.0).unwrap();
        assert!((p0.x - 0.0).abs() < 0.001);
        assert!((p0.y - 0.0).abs() < 0.001);

        let p_mid = spline.point(0.5).unwrap();
        assert!((p_mid.x - 5.0).abs() < 0.001);
        assert!((p_mid.y - 5.0).abs() < 0.001);

        let p1 = spline.point(1.0).unwrap();
        assert!((p1.x - 10.0).abs() < 0.001);
        assert!((p1.y - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_point_quadratic() {
        // Three points = quadratic Bezier
        let spline = GgInternalSpline::new(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 5.0, y: 10.0 }, // Control point
            Vec2 { x: 10.0, y: 0.0 },
        ]);

        // At t=0, should be at the first point
        let p0 = spline.point(0.0).unwrap();
        assert!((p0.x - 0.0).abs() < EPSILON);

        // At t=1, should be at the last point
        let p1 = spline.point(1.0).unwrap();
        assert!((p1.x - 10.0).abs() < EPSILON);

        // At t=0.5, quadratic Bezier: B(0.5) = 0.25*P0 + 0.5*P1 + 0.25*P2
        let p_mid = spline.point(0.5).unwrap();
        assert_eq!(p_mid, Vec2::splat(5.0));
    }

    #[test]
    fn test_new_joined() {
        let spline = GgInternalSpline::new(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 5.0, y: 5.0 },
            Vec2 { x: 10.0, y: 10.0 },
        ]);

        let joined = spline.new_joined().unwrap();
        assert_eq!(joined.len(), 2);

        // First point should be last point of original
        assert_eq!(joined.get(0), Some(Vec2 { x: 10.0, y: 10.0 }));

        // Second point should be reflection: 2*s - p = 2*(10,10) - (5,5) = (15,15)
        let second = joined.get(1).unwrap();
        assert_eq!(second, Vec2::splat(15.0));
    }

    #[test]
    fn test_new_joined_insufficient_points() {
        let empty = GgInternalSpline::new(vec![]);
        assert!(empty.new_joined().is_none());

        let single = GgInternalSpline::new(vec![Vec2::zero()]);
        assert!(single.new_joined().is_none());
    }

    #[test]
    fn test_closest_to() {
        let spline = GgInternalSpline::new(vec![Vec2::zero(), Vec2 { x: 10.0, y: 0.0 }]);

        // Point on the line at x=5 should have t~=0.5
        let t = spline.closest_to(Vec2 { x: 5.0, y: 0.0 }).unwrap();
        assert!((t - 0.5).abs() < 0.01);

        // Point at start should have t~=0
        let t_start = spline.closest_to(Vec2 { x: 0.0, y: 0.0 }).unwrap();
        assert!(t_start < 0.01);

        // Point at end should have t~=1
        let t_end = spline.closest_to(Vec2 { x: 10.0, y: 0.0 }).unwrap();
        assert!(t_end > 0.99);
    }

    #[test]
    fn test_closest_to_empty() {
        let spline = GgInternalSpline::new(vec![]);
        assert!(spline.closest_to(Vec2::zero()).is_none());
    }

    #[test]
    fn test_closest_point_to() {
        let spline = GgInternalSpline::new(vec![Vec2::zero(), Vec2 { x: 10.0, y: 0.0 }]);

        // Point above the line should snap to closest point on line
        let closest = spline.closest_point_to(Vec2 { x: 5.0, y: 100.0 }).unwrap();
        assert_eq!(closest, Vec2 { x: 4.97998, y: 0.0 });
    }

    #[test]
    fn test_closest_point_to_empty() {
        let spline = GgInternalSpline::new(vec![]);
        assert!(spline.closest_point_to(Vec2::zero()).is_none());
    }

    #[test]
    fn test_clone() {
        let spline = GgInternalSpline::new(vec![Vec2::zero(), Vec2::one()]);
        let cloned = spline.clone();
        assert_eq!(cloned.len(), 2);
        assert_eq!(cloned.get(0), spline.get(0));
        assert_eq!(cloned.get(1), spline.get(1));
    }

    #[test]
    fn test_default() {
        let spline = GgInternalSpline::default();
        assert!(spline.is_empty());
        assert_eq!(spline.len(), 0);
        assert_eq!(spline.last(), None);
    }

    #[test]
    fn test_point_clamped_negative_t() {
        // Test behavior when t < 0 - lerp clamps to [0,1] so returns start point
        let spline = GgInternalSpline::new(vec![Vec2::zero(), Vec2 { x: 10.0, y: 0.0 }]);
        let p = spline.point(-0.5).unwrap();
        assert_eq!(p, Vec2::zero());
    }

    #[test]
    fn test_point_clamped_beyond_one() {
        // Test behavior when t > 1 - lerp clamps to [0,1] so returns end point
        let spline = GgInternalSpline::new(vec![Vec2::zero(), Vec2 { x: 10.0, y: 0.0 }]);
        let p = spline.point(1.5).unwrap();
        assert_eq!(p, Vec2 { x: 10.0, y: 0.0 });
    }

    #[test]
    fn test_point_cubic() {
        // Four points = cubic Bezier
        let spline = GgInternalSpline::new(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 0.0, y: 10.0 },
            Vec2 { x: 10.0, y: 10.0 },
            Vec2 { x: 10.0, y: 0.0 },
        ]);

        // At t=0, should be at first point
        assert_eq!(spline.point(0.0).unwrap(), Vec2 { x: 0.0, y: 0.0 });

        // At t=1, should be at last point
        assert_eq!(spline.point(1.0).unwrap(), Vec2 { x: 10.0, y: 0.0 });

        // At t=0.5, cubic Bezier midpoint
        let p_mid = spline.point(0.5).unwrap();
        // For this symmetric curve, x should be 5.0, y should be 7.5
        assert!((p_mid.x - 5.0).abs() < EPSILON);
        assert!((p_mid.y - 7.5).abs() < EPSILON);
    }

    #[test]
    fn test_keep_last_n_zero() {
        let mut spline = GgInternalSpline::new(vec![
            Vec2 { x: 1.0, y: 0.0 },
            Vec2 { x: 2.0, y: 0.0 },
            Vec2 { x: 3.0, y: 0.0 },
        ]);
        spline.keep_last_n(0);
        assert!(spline.is_empty());
    }

    #[test]
    fn test_new_joined_exactly_two_points() {
        // Minimum valid case for new_joined
        let spline = GgInternalSpline::new(vec![Vec2 { x: 0.0, y: 0.0 }, Vec2 { x: 5.0, y: 5.0 }]);

        let joined = spline.new_joined().unwrap();
        assert_eq!(joined.len(), 2);

        // First point should be last point of original
        assert_eq!(joined.get(0), Some(Vec2 { x: 5.0, y: 5.0 }));

        // Second point: 2*s - p = 2*(5,5) - (0,0) = (10,10)
        assert_eq!(joined.get(1), Some(Vec2 { x: 10.0, y: 10.0 }));
    }

    #[test]
    fn test_replace_last_single_element() {
        let mut spline = GgInternalSpline::new(vec![Vec2::zero()]);
        spline.replace_last(Vec2 { x: 42.0, y: 42.0 });
        assert_eq!(spline.len(), 1);
        assert_eq!(spline.last(), Some(Vec2 { x: 42.0, y: 42.0 }));
        assert_eq!(spline.get(0), Some(Vec2 { x: 42.0, y: 42.0 }));
    }

    #[test]
    fn test_push_to_empty() {
        let mut spline = GgInternalSpline::new(vec![]);
        assert!(spline.is_empty());

        spline.push(Vec2 { x: 1.0, y: 2.0 });
        assert_eq!(spline.len(), 1);
        assert_eq!(spline.get(0), Some(Vec2 { x: 1.0, y: 2.0 }));
    }

    #[test]
    fn test_push_front_to_empty() {
        let mut spline = GgInternalSpline::new(vec![]);
        assert!(spline.is_empty());

        spline.push_front(Vec2 { x: 3.0, y: 4.0 });
        assert_eq!(spline.len(), 1);
        assert_eq!(spline.get(0), Some(Vec2 { x: 3.0, y: 4.0 }));
    }

    #[test]
    fn test_closest_to_curved_spline() {
        // Test closest_to on a curved spline, not just a line
        let spline = GgInternalSpline::new(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 5.0, y: 10.0 },
            Vec2 { x: 10.0, y: 0.0 },
        ]);

        // Point at the apex of the curve (5, 5) should be closest to t~=0.5
        let t = spline.closest_to(Vec2 { x: 5.0, y: 5.0 }).unwrap();
        assert!((t - 0.5).abs() < 0.01);

        // Point far below the curve
        let t_below = spline.closest_to(Vec2 { x: 5.0, y: -100.0 }).unwrap();
        // Should still find a reasonable t value
        assert!((0.0..=1.0).contains(&t_below));
    }

    #[test]
    fn test_linear_interpolation_values() {
        // Linear Bezier: B(t) = (1-t)P0 + tP1
        let spline =
            GgInternalSpline::new(vec![Vec2 { x: 0.0, y: 0.0 }, Vec2 { x: 10.0, y: 20.0 }]);

        // Endpoints
        assert_eq!(spline.point(0.0).unwrap(), Vec2 { x: 0.0, y: 0.0 });
        assert_eq!(spline.point(1.0).unwrap(), Vec2 { x: 10.0, y: 20.0 });

        // t=0.1: 0.9*(0,0) + 0.1*(10,20) = (1, 2)
        assert_eq!(spline.point(0.1).unwrap(), Vec2 { x: 1.0, y: 2.0 });

        // t=0.2: 0.8*(0,0) + 0.2*(10,20) = (2, 4)
        assert_eq!(spline.point(0.2).unwrap(), Vec2 { x: 2.0, y: 4.0 });

        // t=0.3: 0.7*(0,0) + 0.3*(10,20) = (3, 6)
        assert_eq!(spline.point(0.3).unwrap(), Vec2 { x: 3.0, y: 6.0 });

        // t=0.4: 0.6*(0,0) + 0.4*(10,20) = (4, 8)
        assert_eq!(spline.point(0.4).unwrap(), Vec2 { x: 4.0, y: 8.0 });

        // t=0.5: 0.5*(0,0) + 0.5*(10,20) = (5, 10)
        assert_eq!(spline.point(0.5).unwrap(), Vec2 { x: 5.0, y: 10.0 });

        // t=0.6: 0.4*(0,0) + 0.6*(10,20) = (6, 12)
        assert_eq!(spline.point(0.6).unwrap(), Vec2 { x: 6.0, y: 12.0 });

        // t=0.7: 0.3*(0,0) + 0.7*(10,20) = (7, 14)
        assert_eq!(spline.point(0.7).unwrap(), Vec2 { x: 7.0, y: 14.0 });

        // t=0.8: 0.2*(0,0) + 0.8*(10,20) = (8, 16)
        assert_eq!(spline.point(0.8).unwrap(), Vec2 { x: 8.0, y: 16.0 });

        // t=0.9: 0.1*(0,0) + 0.9*(10,20) = (9, 18)
        assert_eq!(spline.point(0.9).unwrap(), Vec2 { x: 9.0, y: 18.0 });
    }

    #[test]
    fn test_quadratic_bezier_values() {
        // Quadratic Bezier: B(t) = (1-t)^2*P0 + 2(1-t)t*P1 + t^2*P2
        let spline = GgInternalSpline::new(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 5.0, y: 10.0 },
            Vec2 { x: 10.0, y: 0.0 },
        ]);

        // Endpoints
        assert_eq!(spline.point(0.0).unwrap(), Vec2 { x: 0.0, y: 0.0 });
        assert_eq!(spline.point(1.0).unwrap(), Vec2 { x: 10.0, y: 0.0 });

        // t=0.1: 0.81*(0,0) + 0.18*(5,10) + 0.01*(10,0) = (1, 1.8)
        assert_eq!(spline.point(0.1).unwrap(), Vec2 { x: 1.0, y: 1.8 });

        // t=0.2: 0.64*(0,0) + 0.32*(5,10) + 0.04*(10,0) = (2, 3.2)
        assert_eq!(spline.point(0.2).unwrap(), Vec2 { x: 2.0, y: 3.2 });

        // t=0.3: 0.49*(0,0) + 0.42*(5,10) + 0.09*(10,0) = (3, 4.2)
        assert_eq!(spline.point(0.3).unwrap(), Vec2 { x: 3.0, y: 4.2 });

        // t=0.4: 0.36*(0,0) + 0.48*(5,10) + 0.16*(10,0) = (4, 4.8)
        assert_eq!(spline.point(0.4).unwrap(), Vec2 { x: 4.0, y: 4.8 });

        // t=0.5: 0.25*(0,0) + 0.5*(5,10) + 0.25*(10,0) = (5, 5)
        assert_eq!(spline.point(0.5).unwrap(), Vec2 { x: 5.0, y: 5.0 });

        // t=0.6: 0.16*(0,0) + 0.48*(5,10) + 0.36*(10,0) = (6, 4.8)
        assert_eq!(spline.point(0.6).unwrap(), Vec2 { x: 6.0, y: 4.8 });

        // t=0.7: 0.09*(0,0) + 0.42*(5,10) + 0.49*(10,0) = (7, 4.2)
        assert_eq!(spline.point(0.7).unwrap(), Vec2 { x: 7.0, y: 4.2 });

        // t=0.8: 0.04*(0,0) + 0.32*(5,10) + 0.64*(10,0) = (8, 3.2)
        assert_eq!(spline.point(0.8).unwrap(), Vec2 { x: 8.0, y: 3.2 });

        // t=0.9: 0.01*(0,0) + 0.18*(5,10) + 0.81*(10,0) = (9, 1.8)
        assert_eq!(spline.point(0.9).unwrap(), Vec2 { x: 9.0, y: 1.8 });
    }

    #[test]
    fn test_cubic_bezier_values() {
        // Cubic Bezier: B(t) = (1-t)^3*P0 + 3(1-t)^2*t*P1 + 3(1-t)*t^2*P2 + t^3*P3
        let spline = GgInternalSpline::new(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 0.0, y: 10.0 },
            Vec2 { x: 10.0, y: 10.0 },
            Vec2 { x: 10.0, y: 0.0 },
        ]);

        // Endpoints
        assert_eq!(spline.point(0.0).unwrap(), Vec2 { x: 0.0, y: 0.0 });
        assert_eq!(spline.point(1.0).unwrap(), Vec2 { x: 10.0, y: 0.0 });

        // t=0.1: 0.729*(0,0) + 0.243*(0,10) + 0.027*(10,10) + 0.001*(10,0) = (0.28, 2.7)
        assert_eq!(spline.point(0.1).unwrap(), Vec2 { x: 0.28, y: 2.7 });

        // t=0.2: 0.512*(0,0) + 0.384*(0,10) + 0.096*(10,10) + 0.008*(10,0) = (1.04, 4.8)
        assert_eq!(spline.point(0.2).unwrap(), Vec2 { x: 1.04, y: 4.8 });

        // t=0.3: 0.343*(0,0) + 0.441*(0,10) + 0.189*(10,10) + 0.027*(10,0) = (2.16, 6.3)
        assert_eq!(spline.point(0.3).unwrap(), Vec2 { x: 2.16, y: 6.3 });

        // t=0.4: 0.216*(0,0) + 0.432*(0,10) + 0.288*(10,10) + 0.064*(10,0) = (3.52, 7.2)
        assert_eq!(spline.point(0.4).unwrap(), Vec2 { x: 3.52, y: 7.2 });

        // t=0.5: 0.125*(0,0) + 0.375*(0,10) + 0.375*(10,10) + 0.125*(10,0) = (5, 7.5)
        assert_eq!(spline.point(0.5).unwrap(), Vec2 { x: 5.0, y: 7.5 });

        // t=0.6: 0.064*(0,0) + 0.288*(0,10) + 0.432*(10,10) + 0.216*(10,0) = (6.48, 7.2)
        assert_eq!(spline.point(0.6).unwrap(), Vec2 { x: 6.48, y: 7.2 });

        // t=0.7: 0.027*(0,0) + 0.189*(0,10) + 0.441*(10,10) + 0.343*(10,0) = (7.84, 6.3)
        assert_eq!(spline.point(0.7).unwrap(), Vec2 { x: 7.84, y: 6.3 });

        // t=0.8: 0.008*(0,0) + 0.096*(0,10) + 0.384*(10,10) + 0.512*(10,0) = (8.96, 4.8)
        assert_eq!(spline.point(0.8).unwrap(), Vec2 { x: 8.96, y: 4.8 });

        // t=0.9: 0.001*(0,0) + 0.027*(0,10) + 0.243*(10,10) + 0.729*(10,0) = (9.72, 2.7)
        assert_eq!(spline.point(0.9).unwrap(), Vec2 { x: 9.72, y: 2.7 });
    }

    #[test]
    fn test_quartic_bezier_values() {
        // Quartic (5 points) Bezier: B(t) = Sum C(4,i)(1-t)^(4-i)t^i * Pi
        // Using simple points for easier calculation
        let spline = GgInternalSpline::new(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 2.0, y: 4.0 },
            Vec2 { x: 4.0, y: 0.0 },
            Vec2 { x: 6.0, y: 4.0 },
            Vec2 { x: 8.0, y: 0.0 },
        ]);

        // At endpoints
        assert_eq!(spline.point(0.0).unwrap(), Vec2 { x: 0.0, y: 0.0 });
        assert_eq!(spline.point(1.0).unwrap(), Vec2 { x: 8.0, y: 0.0 });

        // t=0.5: Binomial coefficients for n=4: 1, 4, 6, 4, 1
        // B(0.5) = (0.5)^4 * [1*P0 + 4*P1 + 6*P2 + 4*P3 + 1*P4]
        // = 0.0625 * [(0,0) + 4(2,4) + 6(4,0) + 4(6,4) + (8,0)]
        // = 0.0625 * [(0,0) + (8,16) + (24,0) + (24,16) + (8,0)]
        // = 0.0625 * (64, 32) = (4, 2)
        assert_eq!(spline.point(0.5).unwrap(), Vec2 { x: 4.0, y: 2.0 });
    }

    #[test]
    fn test_bezier_symmetry() {
        // A symmetric quadratic curve should have symmetric values
        let spline = GgInternalSpline::new(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 5.0, y: 10.0 },
            Vec2 { x: 10.0, y: 0.0 },
        ]);

        let p_quarter = spline.point(0.25).unwrap();
        let p_three_quarter = spline.point(0.75).unwrap();

        // x values should be symmetric around 5
        assert!((p_quarter.x + p_three_quarter.x - 10.0).abs() < 0.0001);
        // y values should be equal
        assert!((p_quarter.y - p_three_quarter.y).abs() < 0.0001);
    }

    #[test]
    fn test_cubic_bezier_s_curve() {
        // S-curve: starts going up, ends going down
        let spline = GgInternalSpline::new(vec![
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 0.0, y: 5.0 },
            Vec2 { x: 10.0, y: 5.0 },
            Vec2 { x: 10.0, y: 10.0 },
        ]);

        // Endpoints
        assert_eq!(spline.point(0.0).unwrap(), Vec2 { x: 0.0, y: 0.0 });
        assert_eq!(spline.point(1.0).unwrap(), Vec2 { x: 10.0, y: 10.0 });

        // At t=0.5, the S-curve should pass through (5, 5)
        // B(0.5) = 0.125(0,0) + 0.375(0,5) + 0.375(10,5) + 0.125(10,10)
        // = (0,0) + (0, 1.875) + (3.75, 1.875) + (1.25, 1.25)
        // = (5, 5)
        assert_eq!(spline.point(0.5).unwrap(), Vec2 { x: 5.0, y: 5.0 });
    }
}
