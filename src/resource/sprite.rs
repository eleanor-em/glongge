use num_traits::Zero;
use crate::{
    core::{
        RenderInfo,
        TextureSubArea,
        linalg::Vec2Int,
        prelude::*,
        linalg::{AxisAlignedExtent, Vec2},
        VertexWithUV
    },
    resource::texture::TextureId,
    shader,
};

#[derive(Clone, Default)]
pub struct Sprite {
    texture_id: TextureId,
    areas: Vec<TextureSubArea>,
    elapsed_us: u128,
    paused: bool,
    frame_time_ms: Vec<u32>,
    frame: usize,
}

impl Sprite {
    pub fn from_single_extent(
        texture_id: TextureId,
        extent: Vec2Int,
        top_left: Vec2Int
    ) -> Self {
        Self::from_tileset(
            texture_id,
            Vec2Int::one(),
            extent,
            top_left,
            Vec2Int::zero()
        )
    }
    pub fn from_single_coords(
        texture_id: TextureId,
        top_left: Vec2Int,
        bottom_right: Vec2Int,
    ) -> Self {
        Self::from_single_extent(
            texture_id,
            bottom_right - top_left,
            top_left
        )
    }

    pub fn from_tileset(
        texture_id: TextureId,
        tile_count: Vec2Int,
        tile_size: Vec2Int,
        offset: Vec2Int,
        margin: Vec2Int
    ) -> Self {
        let areas = Vec2Int::range_from_zero(tile_count)
            .map(|(tile_x, tile_y)| {
                let top_left = offset
                    + tile_x * (tile_size + margin).x * Vec2Int::right()
                    + tile_y * (tile_size + margin).y * Vec2Int::down();
                TextureSubArea::new(top_left + tile_size / 2, tile_size / 2)
            })
            .collect_vec();
        let frame_time_ms = vec![1000; areas.len()];
        Self {
            texture_id, areas, frame_time_ms,
            paused: false,
            elapsed_us: 0,
            frame: 0,
        }
    }
    pub fn with_fixed_ms_per_frame(mut self, ms: u32) -> Self {
        self.frame_time_ms = vec![ms; self.areas.len()];
        self
    }
    pub fn with_frame_time_ms(mut self, times: Vec<u32>) -> Self {
        check_eq!(times.len(), self.areas.len());
        self.frame_time_ms = times;
        self
    }
    pub fn with_frame_orders(mut self, frames: Vec<usize>) -> Self {
        self.areas = frames.into_iter().map(|i| self.areas[i]).collect();
        self
    }

    pub fn ready(&self) -> bool { !self.areas.is_empty() }

    pub fn reset(&mut self) { self.elapsed_us = 0; }
    pub fn pause(&mut self) { self.paused = true; }
    pub fn play(&mut self) {
        self.paused = false;
    }

    pub fn fixed_update(&mut self) {
        if self.paused {
            return;
        }
        self.elapsed_us += FIXED_UPDATE_INTERVAL_US;
        check!(self.ready());
        let elapsed_ms = self.elapsed_us / 1000;
        let total_animation_time_ms = self.frame_time_ms.iter().sum::<u32>() as u128;
        let cycle_elapsed_ms = elapsed_ms % total_animation_time_ms;
        let mut cum_sum_ms = 0;
        self.frame = self.frame_time_ms.iter()
            .filter(|&&ms| {
                cum_sum_ms += ms as u128;
                cycle_elapsed_ms >= cum_sum_ms
            })
            .count();
        check_lt!(self.frame, self.areas.len());
    }
    pub fn current_frame(&self) -> TextureSubArea {
        self.areas[self.frame]
    }

    pub fn render_info_default(&self) -> RenderInfo {
        self.render_info_from(RenderInfo::default())
    }
    pub fn render_info_from(&self, mut render_info: RenderInfo) -> RenderInfo {
        if self.ready() {
            render_info.texture_id = Some(self.texture_id);
            render_info.texture_sub_area = self.current_frame();
        }
        render_info
    }

    pub fn create_vertices(&self) -> Vec<VertexWithUV> {
        shader::vertex::rectangle_with_uv(Vec2::zero(), self.half_widths())
    }
}

impl AxisAlignedExtent for Sprite {
    fn extent(&self) -> Vec2 {
        self.current_frame().extent()
    }

    fn centre(&self) -> Vec2 {
        self.current_frame().centre()
    }
}
