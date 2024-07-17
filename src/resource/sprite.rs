#[allow(unused_imports)]
use crate::core::prelude::*;

use std::time::Instant;
use num_traits::Zero;
use crate::{core::linalg::Vec2Int, gg::{RenderInfo, TextureSubArea}, resource::texture::TextureId, shader};
use crate::core::linalg::{SquareExtent, Vec2};
use crate::gg::VertexWithUV;

pub struct Sprite {
    texture_id: TextureId,
    areas: Vec<TextureSubArea>,
    started: Instant,
    paused: Option<Instant>,
    frame_time_ms: Vec<u32>,
}

impl Default for Sprite {
    fn default() -> Self {
        Self {
            texture_id: Default::default(),
            areas: vec![],
            started: Instant::now(),
            paused: None,
            frame_time_ms: vec![],
        }
    }
}

impl Sprite {
    pub fn from_single(
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
            started: Instant::now(),
            paused: None,
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

    pub fn reset(&mut self) { self.started = Instant::now(); }
    pub fn pause(&mut self) { self.paused = Some(Instant::now()); }
    pub fn play(&mut self) {
        if let Some(paused) = self.paused.take() {
            self.started = paused;
        }
    }

    pub fn current_frame(&self) -> TextureSubArea {
        check!(self.ready());
        let instant = match self.paused {
            None => self.started,
            Some(_) => Instant::now(),
        };
        let total_animation_time = self.frame_time_ms.iter().sum::<u32>() as u128;
        let cycle_elapsed = instant.elapsed().as_millis() % total_animation_time;
        let mut cum_sum = 0;
        let frame_index = self.frame_time_ms.iter()
            .filter(|&&t| {
                cum_sum += t as u128;
                cycle_elapsed >= cum_sum
            })
            .count();
        check_lt!(frame_index, self.areas.len());
        self.areas[frame_index]
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

impl SquareExtent for Sprite {
    fn extent(&self) -> Vec2 {
        self.current_frame().extent()
    }

    fn centre(&self) -> Vec2 {
        self.current_frame().centre()
    }
}
