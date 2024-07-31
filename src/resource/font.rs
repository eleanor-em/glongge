use std::io::{ErrorKind, Read};
use ab_glyph::{point, Glyph, ScaleFont, OutlinedGlyph, FontVec, PxScaleFont};
use crate::core::util::colour::Colour;
use crate::core::util::linalg::Vec2Int;
use anyhow::{anyhow, Result};
use itertools::Itertools;
use num_traits::ToPrimitive;
use vulkano::format::Format;
use crate::check_ge;
use crate::core::ObjectTypeEnum;
use crate::core::update::ObjectContext;
use crate::resource::ResourceHandler;
use crate::resource::sprite::Sprite;

const SAMPLE_RATIO: f64 = 8.;

mod internal {
    use anyhow::Result;
    use ab_glyph::{Font, FontVec, PxScale, PxScaleFont};
    use itertools::Itertools;
    use crate::resource::font::SAMPLE_RATIO;

    #[allow(clippy::cast_possible_truncation)]
    pub fn font_from_slice(slice: &[u8], size: f64) -> Result<PxScaleFont<FontVec>> {
        let font = FontVec::try_from_vec(slice.iter().copied().collect_vec())?;
        let scale = PxScale::from((size * SAMPLE_RATIO) as f32);
        Ok(font.into_scaled(scale))
    }
}

pub struct Font {
    inner: PxScaleFont<FontVec>,
}

impl Font {
    pub fn from_slice(slice: &[u8], size: f64) -> Result<Self> {
        Ok(Self { inner: internal::font_from_slice(slice, size)? })
    }

    pub fn sample_ratio(&self) -> f64 { SAMPLE_RATIO }
    pub fn height(&self) -> f64 { f64::from(self.inner.height()) }

    fn layout(&self, text: &str, max_width: f64, text_wrap_mode: TextWrapMode) -> Vec<Glyph> {
        match text_wrap_mode {
            TextWrapMode::WrapAnywhere => self.layout_wrap_anywhere(text, max_width),
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn layout_wrap_anywhere(&self, text: &str, max_width: f64) -> Vec<Glyph> {
        let mut rv = Vec::new();
        let v_advance = self.height() + f64::from(self.inner.line_gap());
        let mut caret = point(0.0, self.inner.ascent());
        let mut last_glyph: Option<Glyph> = None;
        for c in text.chars() {
            if c.is_control() {
                if c == '\n' {
                    caret.x = 0.;
                    caret.y += v_advance as f32;
                    last_glyph = None;
                }
                continue;
            }

            let mut glyph = self.inner.scaled_glyph(c);
            if let Some(previous) = last_glyph.take() {
                caret.x += self.inner.h_advance(previous.id);
                caret.x += self.inner.kern(previous.id, glyph.id);
            }
            let next_x = f64::from(caret.x + self.inner.h_advance(glyph.id));
            if !c.is_whitespace() && next_x > max_width {
                caret.x = 0.;
                caret.y += v_advance as f32;
            }
            glyph.position = caret;
            last_glyph = Some(glyph.clone());
            rv.push(glyph);
        }
        rv
    }

    pub fn render_to_sprite<ObjectType: ObjectTypeEnum>(
        &self,
        object_ctx: &mut ObjectContext<ObjectType>,
        resource_handler: &mut ResourceHandler,
        text: &str,
        max_width: f64,
        text_wrap_mode: TextWrapMode
    ) -> Result<Sprite<ObjectType>> {
        let glyphs = self.layout(text, max_width * SAMPLE_RATIO, text_wrap_mode);
        let mut reader = GlyphReader::new(self, glyphs, Colour::white())?;
        let width = reader.width();
        let height = reader.height();
        Ok(Sprite::from_texture(object_ctx,
            resource_handler.texture.wait_load_reader_rgba(
                &mut reader,
                width,
                height,
                Format::R8G8B8A8_UNORM
            )?))
    }

}

pub enum TextWrapMode {
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
            .ok_or(anyhow!("could not get outline of glyphs"))?;
        check_ge!(all_px_bounds.min.x, 0.);
        check_ge!(all_px_bounds.min.y, 0.);
        check_ge!(all_px_bounds.max.x, 0.);
        check_ge!(all_px_bounds.max.y, 0.);
        Ok(Self { inner: Some(glyphs), col: col.as_bytes(), all_px_bounds })
    }

    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    fn width(&self) -> u32 { self.all_px_bounds.max.x as u32 }
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    fn height(&self) -> u32 { self.all_px_bounds.max.y as u32 }
}

impl Read for GlyphReader {
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let glyphs = self.inner.take()
            .ok_or(std::io::Error::from(ErrorKind::UnexpectedEof))?;

        // Zero out the buffer first.
        buf.iter_mut().for_each(|val| *val = 0);

        for glyph in glyphs {
            let bounds = glyph.px_bounds();
            let img_left = bounds.min.x as u32 - self.all_px_bounds.min.x as u32;
            let img_top = bounds.min.y as u32 - self.all_px_bounds.min.y as u32;
            glyph.draw(|x, y, v| {
                let px = Vec2Int {
                    x: (img_left + x).to_i32().unwrap(),
                    y: (img_top + y).to_i32().unwrap(),
                }.as_index(self.width(), self.height()) * 4;

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
