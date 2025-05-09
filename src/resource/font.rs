use std::io::{ErrorKind, Read};

use itertools::Itertools;
use num_traits::ToPrimitive;
use vulkano::format::Format;

use crate::{core::prelude::*, resource::sprite::Sprite};
use ab_glyph::{FontVec, Glyph, OutlinedGlyph, PxScaleFont, ScaleFont, point};

const SAMPLE_RATIO: f32 = 8.;

mod internal {
    use crate::resource::font::SAMPLE_RATIO;
    use ab_glyph::{Font, FontVec, PxScale, PxScaleFont};
    use anyhow::Result;
    use itertools::Itertools;

    pub fn font_from_slice(slice: &[u8], size: f32) -> Result<PxScaleFont<FontVec>> {
        let font = FontVec::try_from_vec(slice.iter().copied().collect_vec())?;
        let scale = PxScale::from(size * SAMPLE_RATIO);
        Ok(font.into_scaled(scale))
    }
}

pub struct Font {
    inner: PxScaleFont<FontVec>,
}

impl Font {
    pub fn from_slice(slice: &[u8], size: f32) -> Result<Self> {
        Ok(Self {
            inner: internal::font_from_slice(slice, size)?,
        })
    }

    pub fn sample_ratio(&self) -> f32 {
        SAMPLE_RATIO
    }
    pub fn height(&self) -> f32 {
        self.inner.height()
    }

    fn layout(&self, text: &str, max_width: f32, text_wrap_mode: TextWrapMode) -> Vec<Glyph> {
        match text_wrap_mode {
            TextWrapMode::WrapAnywhere => self.layout_wrap_anywhere(text, max_width),
        }
    }

    fn layout_wrap_anywhere(&self, text: &str, max_width: f32) -> Vec<Glyph> {
        let mut rv = Vec::new();
        let v_advance = self.height() + self.inner.line_gap();
        let mut caret = point(0.0, self.inner.ascent());
        let mut last_glyph: Option<Glyph> = None;
        for c in text.chars() {
            if c.is_control() {
                if c == '\n' {
                    caret.x = 0.;
                    caret.y += v_advance;
                    last_glyph = None;
                }
                continue;
            }

            let mut glyph = self.inner.scaled_glyph(c);
            if let Some(previous) = last_glyph.take() {
                caret.x += self.inner.h_advance(previous.id);
                caret.x += self.inner.kern(previous.id, glyph.id);
            }
            let next_x = caret.x + self.inner.h_advance(glyph.id);
            if !c.is_whitespace() && next_x > max_width {
                caret.x = 0.;
                caret.y += v_advance;
            }
            glyph.position = caret;
            last_glyph = Some(glyph.clone());
            rv.push(glyph);
        }
        rv
    }

    pub fn render_to_sprite(
        &self,
        object_ctx: &mut ObjectContext,
        resource_handler: &mut ResourceHandler,
        text: &str,
        max_width: f32,
        text_wrap_mode: TextWrapMode,
    ) -> Result<Sprite> {
        let glyphs = self.layout(text, max_width * SAMPLE_RATIO, text_wrap_mode);
        let mut reader = GlyphReader::new(self, glyphs, Colour::white())?;
        let width = reader.width();
        let height = reader.height();
        Ok(Sprite::add_from_texture(
            object_ctx,
            resource_handler,
            resource_handler.texture.wait_load_reader_rgba(
                "[font]".to_string(),
                &mut reader,
                width,
                height,
                Format::R8G8B8A8_UNORM,
            )?,
        ))
    }
}

#[derive(Copy, Clone, Debug, Default, Hash, Eq, PartialEq)]
pub enum TextWrapMode {
    #[default]
    WrapAnywhere,
}

struct GlyphReader {
    inner: Option<Vec<OutlinedGlyph>>,
    col: [u8; 4],
    all_px_bounds: ab_glyph::Rect,
}

impl GlyphReader {
    fn new(font: &Font, glyphs: Vec<Glyph>, col: Colour) -> Result<Self> {
        // to work out the exact size needed for the drawn glyphs we need to outline
        // them and use their `px_bounds` which hold the coords of their render bounds.
        let glyphs = glyphs
            .into_iter()
            .filter_map(|g| font.inner.outline_glyph(g))
            .collect_vec();
        let all_px_bounds = glyphs
            .iter()
            .map(ab_glyph::OutlinedGlyph::px_bounds)
            .reduce(|mut b, next| {
                b.min.x = b.min.x.min(next.min.x);
                b.max.x = b.max.x.max(next.max.x);
                b.min.y = b.min.y.min(next.min.y);
                b.max.y = b.max.y.max(next.max.y);
                b
            })
            .context("could not get outline of glyphs")?;
        check_ge!(all_px_bounds.min.x, 0.);
        check_ge!(all_px_bounds.min.y, 0.);
        check_ge!(all_px_bounds.max.x, 0.);
        check_ge!(all_px_bounds.max.y, 0.);
        Ok(Self {
            inner: Some(glyphs),
            col: col.as_bytes(),
            all_px_bounds,
        })
    }

    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    fn width(&self) -> u32 {
        self.all_px_bounds.max.x as u32
    }
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    fn height(&self) -> u32 {
        self.all_px_bounds.max.y as u32
    }
}

impl Read for GlyphReader {
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let glyphs = self
            .inner
            .take()
            .ok_or(std::io::Error::from(ErrorKind::UnexpectedEof))?;

        // Zero out the buffer first.
        buf.iter_mut().for_each(|val| *val = 0);

        for glyph in glyphs {
            let bounds = glyph.px_bounds();
            let img_left = bounds.min.x as u32 - self.all_px_bounds.min.x as u32;
            let img_top = bounds.min.y as u32 - self.all_px_bounds.min.y as u32;
            glyph.draw(|x, y, v| {
                let Some(x) = (img_left + x).to_i32() else {
                    warn!("glyph x out of range: {}", img_left + x);
                    return;
                };
                let Some(y) = (img_top + y).to_i32() else {
                    warn!("glyph y out of range: {}", img_top + y);
                    return;
                };
                let px = Vec2i { x, y }.as_index(self.width(), self.height()) * 4;

                buf[px] = self.col[0];
                buf[px + 1] = self.col[1];
                buf[px + 2] = self.col[2];

                let a = f32::from(self.col[3]) / 255.;
                buf[px + 3] = buf[px + 3].saturating_add((v * a * 255.) as u8);
            });
        }
        Ok(buf.len())
    }
}
