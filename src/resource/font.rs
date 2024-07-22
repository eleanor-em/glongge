use std::io::{ErrorKind, Read};
use ab_glyph::{point, Font, Glyph, ScaleFont, PxScale, FontRef, OutlinedGlyph};
use crate::core::util::colour::Colour;
use crate::core::util::linalg::Vec2Int;
use crate::resource::texture::Texture;
use anyhow::{anyhow, Result};
use itertools::Itertools;
use vulkano::format::Format;
use crate::resource::ResourceHandler;

// based on https://github.com/alexheretic/ab-glyph/blob/main/dev/src/layout.rs
fn layout_paragraph<F, SF>(
    font: SF,
    max_width: f32,
    text: &str,
    target: &mut Vec<Glyph>,
) where
    F: Font,
    SF: ScaleFont<F>,
{
    let v_advance = font.height() + font.line_gap();
    let mut caret = point(0.0, font.ascent());
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

        let mut glyph = font.scaled_glyph(c);
        if let Some(previous) = last_glyph.take() {
            caret.x += font.h_advance(previous.id);
            caret.x += font.kern(previous.id, glyph.id);
        }
        if !c.is_whitespace() && caret.x + font.h_advance(glyph.id) > max_width {
            caret.x = 0.;
            caret.y += v_advance;
        }
        glyph.position = caret;
        last_glyph = Some(glyph.clone());
        target.push(glyph);
    }
}

pub enum TextWrapMode {
    WrapAnywhere,
}

pub fn create_bitmap(
    resource_handler: &mut ResourceHandler,
    text: &str,
    size: f64,
    max_width: f64,
    _text_wrap_mode: TextWrapMode,
    sample_ratio: u32
) -> Result<Texture> {
    let font = FontRef::try_from_slice(include_bytes!("../../res/DejaVuSansMono.ttf"))?;

    let scale = PxScale::from((size * sample_ratio as f64) as f32);

    let scaled_font = font.as_scaled(scale);

    let mut glyphs = Vec::new();
    layout_paragraph(scaled_font, (max_width * sample_ratio as f64) as f32, text, &mut glyphs);

    let mut reader = GlyphReader::new(font, glyphs, Colour::white())?;
    let width = reader.width();
    let height = reader.height();
    resource_handler.texture.wait_load_reader_rgba(
        &mut reader,
        width,
        height,
        Format::R8G8B8A8_UNORM
    )
}

struct GlyphReader {
    inner: Option<Vec<OutlinedGlyph>>,
    col: [u8; 4],
    all_px_bounds: ab_glyph::Rect,
}

impl GlyphReader {
    fn new<F: Font>(font: F, glyphs: Vec<Glyph>, col: Colour) -> Result<Self> {
        // to work out the exact size needed for the drawn glyphs we need to outline
        // them and use their `px_bounds` which hold the coords of their render bounds.
        let glyphs = glyphs
            .into_iter()
            .filter_map(|g| font.outline_glyph(g))
            .collect_vec();
        let all_px_bounds = glyphs
            .iter()
            .map(|g| g.px_bounds())
            .reduce(|mut b, next| {
                b.min.x = b.min.x.min(next.min.x);
                b.max.x = b.max.x.max(next.max.x);
                b.min.y = b.min.y.min(next.min.y);
                b.max.y = b.max.y.max(next.max.y);
                b
            })
            .ok_or(anyhow!("could not get outline of glyphs"))?;
        Ok(Self { inner: Some(glyphs), col: col.as_bytes(), all_px_bounds })
    }

    fn width(&self) -> u32 { self.all_px_bounds.max.x as u32 }
    fn height(&self) -> u32 { self.all_px_bounds.max.y as u32 }
}

impl Read for GlyphReader {
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
                    x: (img_left + x) as i32,
                    y: (img_top + y) as i32
                }.as_index(self.width(), self.height()) * 4;

                buf[px] = self.col[0];
                buf[px + 1] = self.col[1];
                buf[px + 2] = self.col[2];

                let a = self.col[3] as f32 / 255.;
                buf[px + 3] = buf[px + 3].saturating_add((v * a * 255.) as u8);
            });
        }
        Ok(buf.len())
    }
}
