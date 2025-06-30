use crate::core::prelude::*;
use crate::core::render::VertexWithCol;
use num_traits::FloatConst;

pub fn map_raw_vertices(vertices: Vec<Vec2>) -> Vec<VertexWithCol> {
    vertices.into_iter().map(VertexWithCol::white).collect()
}

pub fn rectangle(centre: Vec2, half_widths: Vec2) -> RenderItem {
    let top_left = centre - half_widths;
    let top_right = centre
        + Vec2 {
            x: half_widths.x,
            y: -half_widths.y,
        };
    let bottom_left = centre
        + Vec2 {
            x: -half_widths.x,
            y: half_widths.y,
        };
    let bottom_right = centre + half_widths;
    quadrilateral(top_left, top_right, bottom_left, bottom_right)
}

pub fn quadrilateral(
    top_left: Vec2,
    top_right: Vec2,
    bottom_left: Vec2,
    bottom_right: Vec2,
) -> RenderItem {
    RenderItem::from_raw_vertices(vec![
        top_left,
        top_right,
        bottom_left,
        top_right,
        bottom_right,
        bottom_left,
    ])
}

pub fn line(start: Vec2, end: Vec2, width: f32) -> RenderItem {
    let axis = (end - start).normed().orthog();
    let bottom_left = start - axis * width / 2;
    let bottom_right = start + axis * width / 2;
    let top_left = end - axis * width / 2;
    let top_right = end + axis * width / 2;
    RenderItem::from_raw_vertices(vec![
        bottom_left,
        bottom_right,
        top_left,
        bottom_right,
        top_right,
        top_left,
    ])
}

pub fn circle(centre: Vec2, radius: f32, steps: u32) -> RenderItem {
    let dt = 2.0 * f32::PI() / steps as f32;
    RenderItem::from_raw_vertices(
        (0..steps)
            .circular_tuple_windows()
            .flat_map(|(i, j)| {
                vec![
                    centre,
                    centre
                        + Vec2 {
                            x: radius * (i as f32 * dt).cos(),
                            y: radius * (i as f32 * dt).sin(),
                        },
                    centre
                        + Vec2 {
                            x: radius * (j as f32 * dt).cos(),
                            y: radius * (j as f32 * dt).sin(),
                        },
                ]
            })
            .collect(),
    )
}
