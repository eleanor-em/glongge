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
use crate::util::gg_vec::GgVec;
use crate::{core::prelude::*, resource::sprite::Sprite};
use ab_glyph::{
    Font as AbGlyphFontTrait, FontVec, Glyph, OutlinedGlyph, PxScale, PxScaleFont, ScaleFont,
};
use glongge_derive::partially_derive_scene_object;

#[allow(clippy::nonminimal_bool)]
pub fn is_unsupported_codepoint(c: u32) -> bool {
    false
        || (0x01c4..=0x01cc).contains(&c) // Annoying Latin digraphs
        || (0x01f1..=0x01f3).contains(&c) // Annoying Latin digraphs
        || (0x0400..=0x04ff).contains(&c) // Cyrillic; contains some weird large variants
        || (0x0500..=0x052f).contains(&c) // Cyrillic supplement
        || (0x0530..=0x058f).contains(&c) // Armenian
        || (0x0600..=0x06ff).contains(&c) // Arabic; contains some annoying calligraphic characters
        || (0x1400..=0x167f).contains(&c) // Unified Canadian Aboriginal Syllabics
        || (0x1680..=0x169F).contains(&c) // Ogham
        || (0x1f00..=0x1fff).contains(&c) // Greek Extended
        || (0x2160..=0x2188).contains(&c) // Roman numerals
        || (0x2c60..=0x2c7f).contains(&c) // Latin Extended-C
        || (0xa640..=0xa69f).contains(&c) // Cyrillic Extended-B
        || (0xa720..=0xa7ff).contains(&c) // Latin Extended-D
        || (0xfb00..=0xfb4f).contains(&c) // Ligatures
        || (0xfb50..=0xfdff).contains(&c) // Arabic Presentation Forms-A
        || (0xfe70..=0xfeff).contains(&c) // Arabic Presentation Forms-B
        || c == 0x2152 // ⅒, annoyingly large in many fonts
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum FontRenderError {
    Empty,
    TooLarge,
}

#[derive(Clone)]
pub struct Font {
    // Use RefCell for clonability.
    inner: Rc<RefCell<PxScaleFont<FontVec>>>,
    // Use RefCell for interior mutability.
    cached_layout: Rc<RefCell<Option<(String, FontRenderSettings, Layout)>>>,
    max_glyph_width: f32,
}

impl Font {
    #[allow(clippy::nonminimal_bool)]
    fn new(inner: PxScaleFont<FontVec>) -> Self {
        let mut rv = Self {
            inner: Rc::new(RefCell::new(inner)),
            cached_layout: Rc::new(RefCell::new(None)),
            max_glyph_width: 0.0,
        };

        // Get the largest supported character width; used for font-independent margins etc.
        rv.max_glyph_width = (0..0xffff)
            .filter(|n| !is_unsupported_codepoint(*n))
            .filter(|n| {
                // CJK (should work fine, but very slow to lay out)
                !((0x3400..=0x4db5).contains(n)   // CJKUI Ext A
                || (0x4e00..=0x9fcc).contains(n)  // CJK Unified Ideographs
                || (0xac00..=0xd7af).contains(n)  // Hangul Syllables
                || (0xd7b0..=0xd7ff).contains(n)) // Hangul Jamo Extended-B
            })
            .filter_map(char::from_u32)
            .filter(|c| c.is_alphanumeric())
            .map(|c| {
                let inner = rv.inner.borrow();
                let Some(glyph) = inner.outline_glyph(inner.scaled_glyph(c)) else {
                    return 0.0;
                };
                glyph.px_bounds().width() / rv.sample_ratio()
            })
            .max_f32()
            .unwrap_or(0.0);

        rv
    }

    pub fn from_slice(slice: &[u8], size: f32) -> Result<Self> {
        let font = FontVec::try_from_vec(slice.iter().copied().collect_vec())?;
        let scale = PxScale::from(size * FONT_SAMPLE_RATIO);
        Ok(Self::new(font.into_scaled(scale)))
    }

    pub fn sample_ratio(&self) -> f32 {
        FONT_SAMPLE_RATIO
    }
    pub fn max_glyph_width(&self) -> f32 {
        self.max_glyph_width
    }
    /// NOTE: Some lines may be slightly larger than this.
    pub fn line_height(&self) -> f32 {
        self.inner.borrow().height() / self.sample_ratio()
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
                glyphs_by_line.last_mut().unwrap().push((c, 0.0));
                if c == '\n' {
                    glyphs_by_line.push(Vec::new());
                    last_glyph = None;
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
            last_line.push((c, dx));

            // Check if we need to wrap the line.
            let next_x = last_line
                .iter()
                .map(|(_, dx): &(char, f32)| dx)
                .sum::<f32>()
                + dx
                + self.inner.borrow().h_advance(glyph.id);
            if !c.is_whitespace()
                && next_x > max_width
                && let Some((sep, mut word)) = last_line.rsplit_owned(|(c, _)| c.is_whitespace())
            {
                last_line.push(sep);
                word[0].1 = 0.0;
                glyphs_by_line.push(word);
            }

            last_glyph = Some(glyph.clone());
        }
        self.lines_to_layout(glyphs_by_line, max_width)
    }
    // max_width for justification algorithms (TODO).
    fn lines_to_layout(&self, glyphs_by_line: Vec<Vec<(char, f32)>>, _max_width: f32) -> Layout {
        let line_height = self.inner.borrow().height() + self.inner.borrow().line_gap();
        let mut glyphs = Vec::new();
        let mut glyph_ix = 0;
        let mut caret = Vec2 {
            x: 0.0,
            y: self.inner.borrow().ascent(),
        };
        for line in glyphs_by_line {
            glyphs.push(LineGlyphs::new(
                line.into_iter()
                    .map(|(c, dx)| {
                        if is_unsupported_codepoint(c as u32) {
                            warn!("unsupported codepoint: {:?} (0x{:x})", c, c as u32);
                        }
                        caret.x += dx;
                        glyph_ix += 1;
                        let mut glyph = self.inner.borrow().scaled_glyph(c);
                        glyph.position = caret.into();
                        (glyph_ix, (c, glyph))
                    })
                    .filter_map(|(i, (c, g))| {
                        self.inner.borrow().outline_glyph(g).map(|g| (i, (c, g)))
                    })
                    .collect(),
            ));

            caret.x = 0.0;
            caret.y += line_height;
        }
        Layout::new(glyphs, self.sample_ratio())
    }

    pub fn render_to_sprite(
        &self,
        object_ctx: &mut ObjectContext,
        text: impl AsRef<str>,
        settings: &FontRenderSettings,
    ) -> Result<Sprite, FontRenderError> {
        settings.validate();
        let layout = self.layout(text, settings.clone());
        let mut reader = GlyphReader::new(layout, settings.max_glyphs, Colour::white())
            .ok_or(FontRenderError::Empty)?;
        let width = reader.width();
        let height = reader.height();
        if width as f32 > settings.max_width * self.sample_ratio()
            || height as f32 > settings.max_height * self.sample_ratio()
        {
            Err(FontRenderError::TooLarge)
        } else {
            check!(width != 0);
            check!(height != 0);
            Ok(Sprite::add_from_texture_deferred(
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
            ))
        }
    }

    pub fn last_layout(&self) -> Option<Layout> {
        self.cached_layout
            .borrow()
            .as_ref()
            .map(|(_, _, l)| l.clone())
    }
}

#[derive(Clone)]
struct LineGlyphs {
    glyphs: BTreeMap<usize, (char, OutlinedGlyph)>,
}

impl LineGlyphs {
    fn new(glyphs: BTreeMap<usize, (char, OutlinedGlyph)>) -> Self {
        Self { glyphs }
    }

    fn start_of_line_ix(&self) -> Option<usize> {
        self.glyphs.keys().min().copied()
    }
    fn end_of_line_ix(&self) -> Option<usize> {
        self.glyphs.keys().max().copied()
    }

    fn min_x(&self) -> Option<f32> {
        self.glyphs
            .values()
            .map(|(_, g)| g.px_bounds().min.x)
            .min_f32()
    }
    fn max_x(&self) -> Option<f32> {
        self.glyphs
            .values()
            .map(|(_, g)| g.px_bounds().max.x)
            .max_f32()
    }
    fn min_y(&self) -> Option<f32> {
        self.glyphs
            .values()
            .map(|(_, g)| g.px_bounds().min.y)
            .min_f32()
    }
    fn max_y(&self) -> Option<f32> {
        self.glyphs
            .values()
            .map(|(_, g)| g.px_bounds().max.y)
            .max_f32()
    }

    fn is_empty(&self) -> bool {
        self.glyphs.is_empty()
    }

    fn take(&self, max_glyphs: usize) -> impl Iterator<Item = &OutlinedGlyph> {
        self.glyphs
            .iter()
            .take_while(move |(i, _)| **i <= max_glyphs)
            .map(|(_, (_, g))| g)
    }
}

impl From<ab_glyph::Point> for Vec2 {
    fn from(value: ab_glyph::Point) -> Self {
        Self {
            x: value.x,
            y: value.y,
        }
    }
}
impl From<Vec2> for ab_glyph::Point {
    fn from(value: Vec2) -> Self {
        Self {
            x: value.x,
            y: value.y,
        }
    }
}

impl From<ab_glyph::Rect> for Rect {
    fn from(value: ab_glyph::Rect) -> Self {
        Self::from_coords(value.min.into(), value.max.into())
    }
}
impl From<Rect> for ab_glyph::Rect {
    fn from(value: Rect) -> Self {
        Self {
            min: value.top_left().into(),
            max: value.bottom_right().into(),
        }
    }
}

#[derive(Clone)]
pub struct Layout {
    glyphs_by_line: Vec<LineGlyphs>,
    bounds_by_line: Vec<Rect>,
    sample_ratio: f32,
}

impl Layout {
    fn new(glyphs_by_line: Vec<LineGlyphs>, sample_ratio: f32) -> Self {
        check_false!(glyphs_by_line.is_empty());
        check!(
            glyphs_by_line
                .iter()
                .all(|line_glyphs| !line_glyphs.is_empty())
        );
        check!(sample_ratio.is_finite());
        check_gt!(sample_ratio, 0.0);

        let mut bounds_by_line = Vec::new();
        // Calculate width using all lines:
        let mut bounds = glyphs_by_line
            .iter()
            .fold(Rect::default(), |bounds, line_glyphs| {
                bounds.union(&Rect::from_coords(
                    Vec2 {
                        x: line_glyphs.min_x().unwrap(),
                        y: bounds.top(),
                    },
                    Vec2 {
                        x: line_glyphs.max_x().unwrap(),
                        y: bounds.bottom(),
                    },
                ))
            });
        // Calculate height per-line:
        for line_glyphs in &glyphs_by_line {
            bounds = bounds.union(&Rect::from_coords(
                Vec2 {
                    x: bounds.left(),
                    y: line_glyphs.min_y().unwrap(),
                },
                Vec2 {
                    x: bounds.right(),
                    y: line_glyphs.max_y().unwrap(),
                },
            ));
            bounds_by_line.push(bounds);
        }
        check_eq!(glyphs_by_line.len(), bounds_by_line.len());
        Self {
            glyphs_by_line,
            bounds_by_line,
            sample_ratio,
        }
    }

    pub fn bounds_for_max_glyphs(&self, max_glyphs: usize) -> Option<Rect> {
        self.glyphs_by_line
            .iter()
            .zip(self.bounds_by_line.iter())
            .filter_map(|(line_glyphs, line_bounds)| {
                if line_glyphs.start_of_line_ix()? <= max_glyphs {
                    Some(*line_bounds / self.sample_ratio)
                } else {
                    None
                }
            })
            .next_back()
    }

    pub fn max_glyphs_for_bounds(&self, bounds: Rect) -> Option<usize> {
        let bounds = bounds * self.sample_ratio;
        let ixs = self
            .glyphs_by_line
            .iter()
            .zip(self.bounds_by_line.iter())
            .filter_map(|(line_glyphs, line_bounds)| {
                Some((
                    line_glyphs.start_of_line_ix()?,
                    line_glyphs.end_of_line_ix()?,
                    line_bounds,
                ))
            })
            .collect_vec();
        if ixs.is_empty() {
            None
        } else if let Some((first_overflowed_ix, _, _)) = ixs
            .iter()
            .find(|(_, _, line_bounds)| !bounds.contains_rect(line_bounds))
        {
            Some(first_overflowed_ix - 1)
        } else {
            ixs.last().map(|(_, end_of_line_ix, _)| *end_of_line_ix)
        }
    }
}

#[derive(Clone, Debug)]
pub struct FontRenderSettings {
    pub max_width: f32,
    pub max_height: f32,
    pub max_glyphs: usize,
    pub text_wrap_mode: TextWrapMode,
}

impl FontRenderSettings {
    fn validate(&self) {
        check_gt!(self.max_width, 0.0);
        check_gt!(self.max_height, 0.0);
    }

    #[allow(clippy::float_cmp)]
    fn is_same_layout(&self, other: &Self) -> bool {
        self.max_width == other.max_width
            && self.max_height == other.max_height
            && self.text_wrap_mode == other.text_wrap_mode
    }

    pub fn bounds(&self) -> Rect {
        Rect::from_coords(
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 {
                x: self.max_width,
                y: self.max_height,
            },
        )
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
    WrapAtWordBoundary,
}

struct GlyphReader {
    _layout: Layout,
    inner: Option<Vec<OutlinedGlyph>>,
    col: [u8; 4],
    all_px_bounds: Rect,
}

impl GlyphReader {
    fn new(layout: Layout, max_glyphs: usize, col: Colour) -> Option<Self> {
        if max_glyphs == 0 {
            return None;
        }
        let glyphs = layout
            .glyphs_by_line
            .iter()
            .flat_map(|line_glyphs| line_glyphs.take(max_glyphs))
            .cloned()
            .collect_vec();
        let Some(scaled_bounds) = layout.bounds_for_max_glyphs(max_glyphs) else {
            warn!("no bounds for max_glyphs = {max_glyphs}");
            return None;
        };
        let all_px_bounds = scaled_bounds * layout.sample_ratio;
        Some(Self {
            _layout: layout,
            inner: Some(glyphs),
            col: col.as_bytes(),
            all_px_bounds,
        })
    }

    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    fn width(&self) -> u32 {
        self.all_px_bounds.extent().x as u32
    }
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    fn height(&self) -> u32 {
        self.all_px_bounds.extent().y as u32
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
            let img_left = bounds.min.x as u32 - self.all_px_bounds.left() as u32;
            let img_top = bounds.min.y as u32 - self.all_px_bounds.top() as u32;
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
            self.last_render_settings = Some(self.render_settings.clone());
            self.sprite = None;
            self.next_sprite = None;
            self.overflowed = false;
            ctx.object_mut().remove_children();
        } else if self.next_sprite.is_none() {
            self.overflowed = false;
            match self
                .font
                .render_to_sprite(ctx.object_mut(), &text, &self.render_settings)
            {
                Ok(next_sprite) => {
                    self.next_sprite = Some(next_sprite.with_hidden());
                    self.last_render_settings = Some(self.render_settings.clone());
                }
                Err(FontRenderError::TooLarge) => {
                    self.overflowed = true;
                }
                Err(FontRenderError::Empty) => {}
            }
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
            self.last_text = Some(text.clone());
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
    fn extent(&self) -> Vec2 {
        self.next_sprite
            .as_ref()
            .or(self.sprite.as_ref())
            .map_or(Vec2::zero(), Sprite::extent)
            / self.font.sample_ratio()
    }

    fn centre(&self) -> Vec2 {
        self.next_sprite
            .as_ref()
            .or(self.sprite.as_ref())
            .map_or(Vec2::zero(), Sprite::centre)
    }
}
