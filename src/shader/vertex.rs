use num_traits::Zero;

use crate::core::{
    prelude::*,
    render::VertexWithUV
};

pub fn rectangle(centre: Vec2, half_widths: Vec2) -> Vec<Vec2> {
    let top_left = centre - half_widths;
    let top_right = centre + Vec2 {  x: half_widths.x, y: -half_widths.y };
    let bottom_left = centre + Vec2 { x: -half_widths.x, y: half_widths.y };
    let bottom_right = centre + half_widths;
    vec![top_left, top_right, bottom_left, top_right, bottom_right, bottom_left]
}

pub fn rectangle_with_uv(centre: Vec2, half_widths: Vec2) -> RenderItem {
    let uvs = vec![Vec2::zero(), Vec2::right(), Vec2::down(), Vec2::right(), Vec2::one(), Vec2::down()];
    RenderItem {
        vertices: VertexWithUV::zip_from_vec2s(rectangle(centre, half_widths), uvs),
        ..Default::default()
    }
}

pub fn line(start: Vec2, end: Vec2, width: f64) -> RenderItem {
    let axis = (end - start).normed().orthog();
    let bottom_left = start - axis * width / 2;
    let bottom_right = start + axis * width / 2;
    let top_left = end - axis * width / 2;
    let top_right = end + axis * width / 2;
    RenderItem {
        vertices: vec![
            bottom_left, bottom_right, top_left,
            bottom_right, top_left, top_right,
        ].into_iter().map(VertexWithUV::from_vertex).collect(),
        ..Default::default()
    }
}
