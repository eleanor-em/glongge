use itertools::Itertools;
use num_traits::ToPrimitive;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::io::{ErrorKind, Read};
use std::rc::Rc;
use vulkano::format::Format;

use crate::core::render::VertexDepth;
use crate::core::scene::{GuiCommand, GuiObject};
use crate::util::gg_float;
use crate::util::gg_iter::GgFloatIter;
use crate::{core::prelude::*, resource::sprite::Sprite};
use ab_glyph::{FontVec, Glyph, OutlinedGlyph, PxScaleFont, ScaleFont, point};
use glongge_derive::partially_derive_scene_object;

mod internal {
    use crate::core::config::FONT_SAMPLE_RATIO;
    use ab_glyph::{Font, FontVec, PxScale, PxScaleFont};
    use anyhow::Result;
    use itertools::Itertools;

    pub fn font_from_slice(slice: &[u8], size: f32) -> Result<PxScaleFont<FontVec>> {
        let font = FontVec::try_from_vec(slice.iter().copied().collect_vec())?;
        let scale = PxScale::from(size * FONT_SAMPLE_RATIO);
        Ok(font.into_scaled(scale))
    }
}

#[derive(Clone)]
pub struct Font {
    inner: Rc<RefCell<PxScaleFont<FontVec>>>,
    cached_layout: Rc<RefCell<Option<(String, FontRenderSettings, Layout)>>>,
    max_glyph_width: f32,
    max_line_height: f32,
}

impl Font {
    fn new(inner: PxScaleFont<FontVec>) -> Self {
        let mut rv = Self {
            inner: Rc::new(RefCell::new(inner)),
            cached_layout: Rc::new(RefCell::new(None)),
            max_glyph_width: 0.0,
            max_line_height: 0.0,
        };

        // Get the largest possible dimensions.
        rv.max_glyph_width = (0..0xffff)
            .filter_map(char::from_u32)
            .map(|c| {
                let glyphs = rv.layout_no_cache(c.to_string(), &FontRenderSettings::default());
                let Ok(reader) = GlyphReader::new(glyphs, usize::MAX, Colour::white()) else {
                    return 0.0;
                };
                reader.width() as f32 / rv.sample_ratio()
            })
            .max_f32()
            .unwrap_or(0.0);
        let all_chars = (0..0xffff).filter_map(char::from_u32).collect::<String>();
        let layout = rv.layout_no_cache(all_chars, &FontRenderSettings::default());
        let reader = GlyphReader::new(layout, usize::MAX, Colour::white()).unwrap();
        rv.max_line_height = reader.height() as f32 / rv.sample_ratio();

        rv
    }

    pub fn from_slice(slice: &[u8], size: f32) -> Result<Self> {
        Ok(Self::new(internal::font_from_slice(slice, size)?))
    }

    pub fn sample_ratio(&self) -> f32 {
        FONT_SAMPLE_RATIO
    }
    pub fn height(&self) -> f32 {
        self.inner.borrow().height()
    }
    pub fn max_glyph_width(&self) -> f32 {
        self.max_glyph_width
    }
    pub fn max_line_height(&self) -> f32 {
        self.max_line_height
    }

    fn layout(&self, text: impl AsRef<str>, settings: FontRenderSettings) -> Layout {
        let text = text.as_ref();
        if let Some((cached_text, cached_settings, cached_layout)) =
            self.cached_layout.borrow().as_ref()
            && cached_text == text
            && cached_settings.is_same_layout(&settings)
        {
            cached_layout.clone()
        } else {
            let layout = self.layout_no_cache(text, &settings);
            self.cached_layout
                .borrow_mut()
                .replace((text.to_string(), settings, layout.clone()));
            layout
        }
    }
    fn layout_no_cache(&self, text: impl AsRef<str>, settings: &FontRenderSettings) -> Layout {
        if settings.max_width == f32::INFINITY {
            self.layout_no_wrap(text)
        } else {
            match settings.text_wrap_mode {
                TextWrapMode::WrapAnywhere => {
                    self.layout_wrap_anywhere(text, settings.max_width * self.sample_ratio())
                }
                TextWrapMode::WrapAtWordBoundary => self
                    .layout_wrap_at_word_boundary(text, settings.max_width * self.sample_ratio()),
            }
        }
    }
    fn layout_no_wrap(&self, text: impl AsRef<str>) -> Layout {
        let mut glyphs_by_line = vec![Vec::new()];
        let mut last_glyph: Option<Glyph> = None;
        for c in text.as_ref().chars() {
            if c.is_control() {
                if c == '\n' {
                    last_glyph = None;
                    glyphs_by_line.push(Vec::new());
                }
                continue;
            }

            let glyph = self.inner.borrow().scaled_glyph(c);
            let dx = if let Some(previous) = last_glyph.take() {
                let font = self.inner.borrow();
                font.h_advance(previous.id) + font.kern(previous.id, glyph.id)
            } else {
                0.0
            };
            glyphs_by_line.last_mut().unwrap().push((c, dx));
            last_glyph = Some(glyph.clone());
        }
        self.lines_to_layout(glyphs_by_line, f32::INFINITY)
    }
    fn layout_wrap_anywhere(&self, text: impl AsRef<str>, max_width: f32) -> Layout {
        let mut glyphs_by_line = vec![Vec::new()];
        let mut last_glyph: Option<Glyph> = None;
        for c in text.as_ref().chars() {
            if c.is_control() {
                if c == '\n' {
                    last_glyph = None;
                    glyphs_by_line.push(Vec::new());
                }
                continue;
            }

            let glyph = self.inner.borrow().scaled_glyph(c);
            let dx = if let Some(previous) = last_glyph.take() {
                let font = self.inner.borrow();
                font.h_advance(previous.id) + font.kern(previous.id, glyph.id)
            } else {
                0.0
            };
            let last_line = glyphs_by_line.last_mut().unwrap();
            let next_x = last_line.iter().map(|(_, dx)| *dx).sum::<f32>()
                + dx
                + self.inner.borrow().h_advance(glyph.id);
            if !c.is_whitespace() && next_x > max_width {
                glyphs_by_line.push(vec![(c, 0.0)]);
            } else {
                last_line.push((c, dx));
            }
            last_glyph = Some(glyph.clone());
        }
        self.lines_to_layout(glyphs_by_line, max_width)
    }
    fn layout_wrap_at_word_boundary(&self, text: impl AsRef<str>, max_width: f32) -> Layout {
        let mut glyphs_by_line = vec![Vec::new()];
        let mut last_glyph: Option<Glyph> = None;
        for c in text.as_ref().chars() {
            if c.is_control() {
                if c == '\n' {
                    last_glyph = None;
                    glyphs_by_line.push(Vec::new());
                }
                continue;
            }

            let glyph = self.inner.borrow().scaled_glyph(c);
            let dx = if let Some(previous) = last_glyph.take() {
                let font = self.inner.borrow();
                font.h_advance(previous.id) + font.kern(previous.id, glyph.id)
            } else {
                0.0
            };
            let last_line = glyphs_by_line.last_mut().unwrap();
            let next_x = last_line
                .iter()
                .map(|(_, dx): &(char, f32)| dx)
                .sum::<f32>()
                + dx
                + self.inner.borrow().h_advance(glyph.id);
            if !c.is_whitespace() && next_x > max_width {
                let mut word = last_line
                    .iter()
                    .rev()
                    .take_while(|(c, _)| !c.is_whitespace())
                    .copied()
                    .collect_vec();
                last_line.truncate(last_line.len() - word.len());
                word.reverse();
                word.push((c, dx));
                word[0].1 = 0.0;
                glyphs_by_line.push(word);
            } else {
                last_line.push((c, dx));
            }
            last_glyph = Some(glyph.clone());
        }
        self.lines_to_layout(glyphs_by_line, max_width)
    }
    // max_width for justification algorithms (TODO).
    fn lines_to_layout(&self, glyphs_by_line: Vec<Vec<(char, f32)>>, _max_width: f32) -> Layout {
        let mut caret = point(0.0, self.inner.borrow().ascent());
        let mut glyphs = Vec::new();
        let mut line_breaks = Vec::new();
        for line in glyphs_by_line {
            line_breaks.push(line_breaks.last().unwrap_or(&0) + line.len());
            for (c, dx) in line {
                caret.x += dx;
                let mut glyph = self.inner.borrow().scaled_glyph(c);
                glyph.position = caret;
                glyphs.push(glyph);
            }
            caret.x = 0.0;
            caret.y += self.height() + self.inner.borrow().line_gap();
        }
        let glyphs = glyphs
            .into_iter()
            .enumerate()
            .filter_map(|(i, g)| self.inner.borrow().outline_glyph(g).map(|g| (i, g)))
            .collect();
        Layout {
            glyphs,
            line_breaks,
        }
    }

    pub fn dry_run_render(
        &self,
        text: impl AsRef<str>,
        settings: &FontRenderSettings,
    ) -> Result<bool> {
        settings.validate();
        let layout = self.layout(text, settings.clone());
        let reader = GlyphReader::new(layout, settings.max_glyphs, Colour::white())?;
        let width = reader.width();
        let height = reader.height();
        Ok(!(width as f32 > settings.max_width * self.sample_ratio()
            || height as f32 > settings.max_height * self.sample_ratio()))
    }

    pub fn render_to_sprite(
        &self,
        object_ctx: &mut ObjectContext,
        text: impl AsRef<str>,
        settings: &FontRenderSettings,
    ) -> Result<Option<Sprite>> {
        settings.validate();
        let layout = self.layout(text, settings.clone());
        let mut reader = GlyphReader::new(layout, settings.max_glyphs, Colour::white())?;
        let width = reader.width();
        let height = reader.height();
        if width as f32 > settings.max_width * self.sample_ratio()
            || height as f32 > settings.max_height * self.sample_ratio()
        {
            Ok(None)
        } else {
            Ok(Some(Sprite::add_from_texture_deferred(
                object_ctx,
                Box::new(move |resource_handler| {
                    resource_handler.texture.wait_load_reader_rgba(
                        "[font]".to_string(),
                        &mut reader,
                        width,
                        height,
                        Format::R8G8B8A8_UNORM,
                    )
                }),
            )))
        }
    }
}

#[derive(Clone)]
struct Layout {
    glyphs: BTreeMap<usize, OutlinedGlyph>,
    line_breaks: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct FontRenderSettings {
    pub max_width: f32,
    pub max_height: f32,
    pub max_glyphs: usize,
    pub text_wrap_mode: TextWrapMode,
}

impl FontRenderSettings {
    pub fn validate(&self) {
        check_gt!(self.max_width, 0.0);
        check_gt!(self.max_height, 0.0);
        check!(self.max_glyphs > 0);
    }

    #[allow(clippy::float_cmp)]
    fn is_same_layout(&self, other: &Self) -> bool {
        self.max_width == other.max_width
            && self.max_height == other.max_height
            && self.text_wrap_mode == other.text_wrap_mode
    }
}

impl Default for FontRenderSettings {
    fn default() -> Self {
        Self {
            max_width: f32::INFINITY,
            max_height: f32::INFINITY,
            max_glyphs: usize::MAX,
            text_wrap_mode: TextWrapMode::default(),
        }
    }
}

impl PartialEq for FontRenderSettings {
    fn eq(&self, other: &Self) -> bool {
        gg_float::is_finite(self.max_width) == gg_float::is_finite(other.max_width)
            && (!gg_float::is_finite(self.max_width) || self.max_width == other.max_width)
            && gg_float::is_finite(self.max_height) == gg_float::is_finite(other.max_height)
            && (!gg_float::is_finite(self.max_height) || self.max_height == other.max_height)
            && self.max_glyphs == other.max_glyphs
            && self.text_wrap_mode == other.text_wrap_mode
    }
}

impl Eq for FontRenderSettings {}

#[derive(Copy, Clone, Debug, Default, Hash, Eq, PartialEq)]
pub enum TextWrapMode {
    #[default]
    WrapAnywhere,
    // TODO:
    WrapAtWordBoundary,
}

struct GlyphReader {
    _layout: Layout,
    inner: Option<Vec<OutlinedGlyph>>,
    col: [u8; 4],
    all_px_bounds: ab_glyph::Rect,
}

impl GlyphReader {
    fn new(layout: Layout, max_glyphs: usize, col: Colour) -> Result<Self> {
        // To work out the exact size needed for the drawn glyphs, we need to outline
        // them and use their px_bounds which hold the coordinates of their render bounds.
        let next_line_break = *layout
            .line_breaks
            .iter()
            .find(|i| **i >= max_glyphs)
            .unwrap_or(&usize::MAX);
        let (_, all_px_bounds) = layout
            .glyphs
            .iter()
            .map(|(i, g)| (i, g.px_bounds()))
            .reduce(|(_i, mut bounds), (j, next)| {
                // Use all lines to find the width:
                bounds.min.x = bounds.min.x.min(next.min.x);
                bounds.max.x = bounds.max.x.max(next.max.x);
                // Use until the next line break to find the height:
                if *j < next_line_break {
                    bounds.min.y = bounds.min.y.min(next.min.y);
                    bounds.max.y = bounds.max.y.max(next.max.y);
                }
                (j, bounds)
            })
            .context("could not get outline of glyphs")?;
        // Now get the actual glyphs we will use:
        let glyphs = layout
            .glyphs
            .iter()
            .filter(|(i, _)| **i < max_glyphs)
            .map(|(_, g)| g)
            .cloned()
            .collect_vec();

        Ok(Self {
            _layout: layout,
            inner: Some(glyphs),
            col: col.as_bytes(),
            all_px_bounds,
        })
    }

    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    fn width(&self) -> u32 {
        self.all_px_bounds.max.x as u32 - self.all_px_bounds.min.x as u32
    }
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    fn height(&self) -> u32 {
        self.all_px_bounds.max.y as u32 - self.all_px_bounds.min.y as u32
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
        for val in buf.iter_mut() {
            *val = 0;
        }

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

                let a = f32::from(self.col[3]) / 255.0;
                buf[px + 3] = buf[px + 3].saturating_add((v * a * 255.0) as u8);
            });
        }
        Ok(buf.len())
    }
}

pub struct Label {
    font: Font,
    sprite: Option<Sprite>,
    next_sprite: Option<Sprite>,

    overflowed: bool,

    text_to_set: Option<String>,
    last_text: Option<String>,
    render_settings: FontRenderSettings,
    last_render_settings: Option<FontRenderSettings>,

    depth: VertexDepth,
    blend_col: Colour,
}

impl Label {
    pub fn new(font: Font, render_settings: FontRenderSettings) -> Self {
        Self {
            font,
            sprite: None,
            next_sprite: None,
            overflowed: false,
            text_to_set: None,
            last_text: None,
            render_settings,
            last_render_settings: None,
            depth: VertexDepth::default(),
            blend_col: Colour::white(),
        }
    }

    pub fn font(&self) -> &Font {
        &self.font
    }

    pub fn set_text(&mut self, text: impl AsRef<str>) {
        self.text_to_set = Some(text.as_ref().to_string());
    }

    pub fn set_depth(&mut self, depth: VertexDepth) {
        self.depth = depth;
    }
    pub fn set_blend_col(&mut self, colour: Colour) {
        self.blend_col = colour;
    }
    pub fn render_settings(&mut self) -> &mut FontRenderSettings {
        &mut self.render_settings
    }

    pub fn would_overflow(&self) -> bool {
        let Some(text) = self.last_text.as_ref() else {
            return false;
        };
        !self
            .font
            .dry_run_render(text, &self.render_settings)
            .unwrap()
    }
    pub fn overflowed(&self) -> bool {
        self.overflowed
    }

    fn changed_render_settings(&self) -> bool {
        self.last_render_settings
            .as_ref()
            .is_none_or(|s| *s != self.render_settings)
    }

    fn render_text(&mut self, ctx: &mut UpdateContext, text: String) {
        if text.is_empty() {
            self.last_text = Some(text);
            self.last_render_settings = Some(self.render_settings.clone());
            self.sprite = None;
            self.next_sprite = None;
            ctx.object_mut().remove_children();
        } else if text.as_str() == self.last_text.clone().unwrap_or_default().as_str()
            && !self.changed_render_settings()
        {
            // No update necessary.
        } else if self.next_sprite.is_none() {
            self.next_sprite = self
                .font
                .render_to_sprite(ctx.object_mut(), &text, &self.render_settings.clone())
                .unwrap()
                .map(Sprite::with_hidden);
            if self.next_sprite.is_some() {
                self.last_text = Some(text);
                self.last_render_settings = Some(self.render_settings.clone());
            }
            self.overflowed = self.next_sprite.is_none();
        } else {
            // Still loading the previous next_sprite, try again next update.
            self.text_to_set = Some(text);
        }
    }
}

#[partially_derive_scene_object]
impl SceneObject for Label {
    fn on_load(
        &mut self,
        object_ctx: &mut ObjectContext,
        _resource_handler: &mut ResourceHandler,
    ) -> Result<Option<RenderItem>> {
        object_ctx.transform_mut().scale = Vec2::one() / self.font.sample_ratio();
        Ok(None)
    }

    fn on_update_begin(&mut self, ctx: &mut UpdateContext) {
        if self
            .next_sprite
            .as_ref()
            .is_some_and(Sprite::textures_ready)
        {
            if let Some(sprite) = self.sprite.take() {
                ctx.object_mut().remove(&sprite.inner.unwrap());
            }
            self.sprite = self.next_sprite.take();
            let sprite = self.sprite.as_mut().unwrap();
            sprite.show();
            sprite.set_depth(self.depth);
            sprite.set_blend_col(self.blend_col);
        }
    }

    fn on_update_end(&mut self, ctx: &mut UpdateContext) {
        if let Some(text) = self.text_to_set.take() {
            self.render_text(ctx, text);
        } else if self.changed_render_settings()
            && let Some(text) = self.last_text.clone()
        {
            self.render_text(ctx, text);
        }
    }

    fn as_gui_object(&mut self) -> Option<&mut dyn GuiObject> {
        Some(self)
    }
}

impl GuiObject for Label {
    fn on_gui(&mut self, _ctx: &UpdateContext, _selected: bool) -> GuiCommand {
        if let Some(text) = self.last_text.as_ref() {
            let text = text.clone();
            GuiCommand::new(move |ui| {
                ui.add(egui::Label::new(text).selectable(false));
            })
        } else {
            GuiCommand::new(move |_ui| {})
        }
    }
}

impl AxisAlignedExtent for Label {
    fn aa_extent(&self) -> Vec2 {
        self.next_sprite
            .as_ref()
            .or(self.sprite.as_ref())
            .map_or(Vec2::zero(), Sprite::aa_extent)
            / self.font.sample_ratio()
    }

    fn centre(&self) -> Vec2 {
        self.next_sprite
            .as_ref()
            .or(self.sprite.as_ref())
            .map_or(Vec2::zero(), Sprite::centre)
    }
}
