use num_traits::Zero;

use crate::{
    core::linalg::Vec2,
    core::VertexWithUV
};

pub fn rectangle(centre: Vec2, half_widths: Vec2) -> Vec<Vec2> {
    let top_left = centre - half_widths;
    let top_right = centre + Vec2 {  x: half_widths.x, y: -half_widths.y };
    let bottom_left = centre + Vec2 { x: -half_widths.x, y: half_widths.y };
    let bottom_right = centre + half_widths;
    vec![top_left, top_right, bottom_left, top_right, bottom_right, bottom_left]
}

pub fn rectangle_with_uv(centre: Vec2, half_widths: Vec2) -> Vec<VertexWithUV> {
    let uvs = vec![Vec2::zero(), Vec2::right(), Vec2::down(), Vec2::right(), Vec2::one(), Vec2::down()];
    VertexWithUV::zip_from_iter(rectangle(centre, half_widths), uvs)
}
