use std::time::Instant;
use itertools::Itertools;
use crate::{
    assert::*,
    core::linalg::Vec2Int,
    gg::{RenderInfo, TextureSubArea},
    resource::texture::TextureId
};

pub struct Sprite {
    texture_id: TextureId,
    areas: Vec<TextureSubArea>,
    started: Instant,
    paused: Option<Instant>,
    ms_per_frame: u32,
}

impl Default for Sprite {
    fn default() -> Self {
        Self {
            texture_id: Default::default(),
            areas: vec![],
            started: Instant::now(),
            paused: None,
            ms_per_frame: 0,
        }
    }
}

impl Sprite {
    pub fn from_tileset(
        texture_id: TextureId,
        tile_count: Vec2Int,
        tile_size: Vec2Int,
        border: Vec2Int,
        margin: Vec2Int,
        ms_per_frame: u32
    ) -> Self {
        let areas = (0..tile_count.x).cartesian_product(0..tile_count.y)
            .map(|(tile_x, tile_y)| {
                let top_left = border
                    + tile_x * (tile_size + margin).x * Vec2Int::right()
                    + tile_y * (tile_size + margin).y * Vec2Int::down();
                TextureSubArea::new(top_left + tile_size / 2, tile_size / 2)
            })
            .collect();
        Self {
            texture_id, areas, ms_per_frame,
            started: Instant::now(),
            paused: None,
        }
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
        let instant = match self.paused {
            None => self.started,
            Some(_) => Instant::now(),
        };
        let frames_elapsed = instant.elapsed().as_millis() / self.ms_per_frame as u128;
        let frame_index = (frames_elapsed as usize) % self.areas.len();
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
}
