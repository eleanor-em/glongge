use crate::core::linalg::Vec2;

pub fn rectangle(centre: Vec2, extents: Vec2) -> Vec<Vec2> {
    let half_widths = extents / 2.0;
    let top_left = centre - half_widths;
    let top_right = centre + Vec2 {  x: half_widths.x, y: -half_widths.y };
    let bottom_left = centre + Vec2 { x: -half_widths.x, y: half_widths.y };
    let bottom_right = centre + half_widths;
    vec![top_left, top_right, bottom_left, top_right, bottom_right, bottom_left]
}
