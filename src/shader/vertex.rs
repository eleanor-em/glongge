use crate::core::linalg::Vec2;

pub fn rectangle(top_left: Vec2, extents: Vec2) -> Vec<Vec2> {
    let top_right = top_left + Vec2::right() * extents.x;
    let bottom_left = top_left + Vec2::down() * extents.y;
    let bottom_right = top_left + extents;
    vec![top_left, top_right, bottom_left, top_right, bottom_right, bottom_left]
}
