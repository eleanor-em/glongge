use itertools::Itertools;
use num_traits::ToPrimitive;
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::io::{ErrorKind, Read};
use std::sync::OnceLock;

use crate::core::render::VertexDepth;
use crate::core::scene::{GuiCommand, GuiObject};
use crate::core::update::{ObjectContext, RenderContext};
use crate::resource::rich_text::{FormatInstruction, FormattedChars};
use crate::util::gg_float::{FloatKey, GgFloat};
use crate::util::gg_iter::GgFloatIter;
use crate::util::gg_sync::GgMutex;
use crate::util::gg_vec::GgVec;
use crate::util::{GLOBAL_STATS, gg_float};
use crate::{core::prelude::*, resource::sprite::Sprite};
use ab_glyph::{
    Font as AbGlyphFontTrait, FontVec, Glyph, OutlinedGlyph, PxScale, PxScaleFont, ScaleFont,
};
use ash::vk;
use glongge_derive::partially_derive_scene_object;

pub static LOADED_FONTS: OnceLock<GgMutex<BTreeMap<String, BTreeMap<FloatKey, Font>>>> =
    OnceLock::new();

/// Create [`Font`]s with these macros.
#[macro_export]
macro_rules! font_from_file {
    ($path:expr, $size:expr) => {{
        match $crate::resource::font::LOADED_FONTS
            .get_or_init($crate::util::gg_sync::GgMutex::default)
            .try_lock("font_from_file!()")
            .expect("font_from_file!()")
            .expect("LOADED_FONTS should not be used anywhere else")
            .entry($path.to_string())
            .or_default()
            .entry($crate::util::gg_float::FloatKey($size))
        {
            std::collections::btree_map::Entry::Vacant(vacant) => {
                $crate::resource::font::Font::from_slice_uncached(include_bytes_root!($path), $size)
                    .map(|f| vacant.insert(f))
                    .cloned()
            }
            std::collections::btree_map::Entry::Occupied(value) => Ok(value.get().clone()),
        }
    }};
}

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

impl Display for FontRenderError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(self, f)
    }
}

impl Error for FontRenderError {}

pub struct Font {
    inner: PxScaleFont<FontVec>,
    cached_layout: GgMutex<Option<(String, FontLayout)>>,
    max_glyph_width: f32,
}

impl Font {
    #[allow(clippy::nonminimal_bool)]
    fn new(inner: PxScaleFont<FontVec>) -> Self {
        let mut rv = Self {
            inner,
            cached_layout: GgMutex::default(),
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
                let Some(glyph) = rv.inner.outline_glyph(rv.inner.scaled_glyph(c)) else {
                    return 0.0;
                };
                glyph.px_bounds().width() / rv.sample_ratio()
            })
            .max_f32()
            .unwrap_or(0.0);

        rv
    }

    /// Probably should not be called directly! Use e.g. above macro [`font_from_file`].
    pub fn from_slice_uncached(slice: &[u8], size: f32) -> Result<Self> {
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
        self.inner.height() / self.sample_ratio()
    }

    fn format_rich_text(text: impl AsRef<str>, settings: &FontRenderSettings) -> FormattedChars {
        let text = text.as_ref();
        if settings.do_parse_rich_text
            && let Some(parsed) = FormattedChars::parse(text)
        {
            parsed
        } else {
            FormattedChars::unformatted(text)
        }
    }

    pub fn layout(&self, text: impl AsRef<str>, settings: FontRenderSettings) -> FontLayout {
        settings.validate();
        let text = text.as_ref();
        if let Some((cached_text, cached_layout)) = self
            .cached_layout
            .try_lock_short("Font::layout()")
            .expect("deadlock should be impossible")
            .as_ref()
            && cached_text == text
            && cached_layout.settings.is_same_layout(&settings)
        {
            cached_layout.clone().with_settings(settings)
        } else {
            let layout = self.layout_no_cache(text, settings);
            self.cached_layout
                .try_lock_short("Font::layout()")
                .expect("deadlock should be impossible")
                .replace((text.to_string(), layout.clone()));
            layout
        }
    }
    fn layout_no_cache(&self, text: impl AsRef<str>, settings: FontRenderSettings) -> FontLayout {
        settings.validate();
        let formatted_chars = Self::format_rich_text(text.as_ref(), &settings);
        if settings.max_width == f32::INFINITY {
            self.layout_no_wrap(formatted_chars, settings)
        } else {
            match settings.text_wrap_mode {
                TextWrapMode::WrapAnywhere => self.layout_wrap_anywhere(formatted_chars, settings),
                TextWrapMode::WrapAtWordBoundary => {
                    self.layout_wrap_at_word_boundary(formatted_chars, settings)
                }
            }
        }
    }
    fn layout_no_wrap(
        &self,
        formatted_chars: FormattedChars,
        settings: FontRenderSettings,
    ) -> FontLayout {
        check_eq!(settings.max_width, f32::INFINITY);
        let mut glyphs_by_line = vec![Vec::new()];
        let mut last_glyph: Option<Glyph> = None;
        // Below case possibly doesn't work?
        // check!(
        //     formatted_chars.format_instructions.is_empty(),
        //     "not implemented"
        // );
        for c in formatted_chars.chars {
            if c.is_control() {
                if c == '\n' {
                    last_glyph = None;
                    glyphs_by_line.push(Vec::new());
                }
                continue;
            }

            let glyph = self.inner.scaled_glyph(c);
            let dx = if let Some(previous) = last_glyph.take() {
                self.inner.h_advance(previous.id) + self.inner.kern(previous.id, glyph.id)
            } else {
                0.0
            };
            glyphs_by_line.last_mut().unwrap().push((c, dx));
            last_glyph = Some(glyph.clone());
        }
        self.lines_to_layout(
            glyphs_by_line,
            formatted_chars.format_instructions,
            settings,
        )
    }
    fn layout_wrap_anywhere(
        &self,
        formatted_chars: FormattedChars,
        settings: FontRenderSettings,
    ) -> FontLayout {
        let mut glyphs_by_line = vec![Vec::new()];
        let mut last_glyph: Option<Glyph> = None;
        // Below case possibly doesn't work?
        // check!(
        //     formatted_chars.format_instructions.is_empty(),
        //     "not implemented"
        // );
        for c in formatted_chars.chars {
            if c.is_control() {
                if c == '\n' {
                    last_glyph = None;
                    glyphs_by_line.push(Vec::new());
                }
                continue;
            }

            let glyph = self.inner.scaled_glyph(c);
            let dx = if let Some(previous) = last_glyph.take() {
                self.inner.h_advance(previous.id) + self.inner.kern(previous.id, glyph.id)
            } else {
                0.0
            };
            let last_line = glyphs_by_line.last_mut().unwrap();
            let next_x = last_line.iter().map(|(_, dx)| *dx).sum::<f32>()
                + dx
                + self.inner.h_advance(glyph.id);
            if !c.is_whitespace() && next_x > settings.max_width * self.sample_ratio() {
                glyphs_by_line.push(vec![(c, 0.0)]);
            } else {
                last_line.push((c, dx));
            }
            last_glyph = Some(glyph.clone());
        }
        self.lines_to_layout(
            glyphs_by_line,
            formatted_chars.format_instructions,
            settings,
        )
    }
    fn layout_wrap_at_word_boundary(
        &self,
        formatted_chars: FormattedChars,
        settings: FontRenderSettings,
    ) -> FontLayout {
        let mut glyphs_by_line = vec![Vec::new()];
        let mut last_glyph: Option<Glyph> = None;
        for c in formatted_chars.chars {
            if c.is_control() {
                glyphs_by_line.last_mut().unwrap().push((' ', 0.0));
                if c == '\n' {
                    glyphs_by_line.push(Vec::new());
                    last_glyph = None;
                }
                continue;
            }

            let glyph = self.inner.scaled_glyph(c);
            let dx = if let Some(previous) = last_glyph.take() {
                self.inner.h_advance(previous.id) + self.inner.kern(previous.id, glyph.id)
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
                + self.inner.h_advance(glyph.id);
            if next_x > settings.max_width * self.sample_ratio()
                && !c.is_whitespace()
                && let Some((mut sep, mut word)) =
                    last_line.rsplit_owned(|(c, _)| c.is_whitespace())
            {
                sep.1 = 0.0;
                last_line.push(sep);
                word[0].1 = 0.0;
                glyphs_by_line.push(word);
            }

            last_glyph = Some(glyph.clone());
        }
        self.lines_to_layout(
            glyphs_by_line,
            formatted_chars.format_instructions,
            settings,
        )
    }
    fn lines_to_layout(
        &self,
        glyphs_by_line: Vec<Vec<(char, f32)>>,
        mut format_instructions: BTreeMap<usize, FormatInstruction>,
        settings: FontRenderSettings,
    ) -> FontLayout {
        let line_height = self.inner.height() + self.inner.line_gap();
        let mut glyphs = Vec::new();
        let mut glyph_ix = 0;
        let mut outlined_glyph_ix = 0;
        let mut out_format_instructions = BTreeMap::new();
        let mut caret = Vec2 {
            x: 0.0,
            y: self.inner.ascent(),
        };
        for line in glyphs_by_line {
            let mut line_glyphs = BTreeMap::new();
            for (c, dx) in line {
                check_le!(outlined_glyph_ix, glyph_ix);
                if let Some(&format_ix) = format_instructions.keys().next() {
                    if glyph_ix == format_ix {
                        let (_, instr) = format_instructions.pop_first().unwrap();
                        out_format_instructions.insert(outlined_glyph_ix, instr);
                    } else {
                        check_lt!(glyph_ix, format_ix);
                    }
                }
                if is_unsupported_codepoint(c as u32)
                    && GLOBAL_STATS
                        .get_or_init(GgMutex::default)
                        .try_lock_short("Font::lines_to_layout()")
                        .unwrap()
                        .warned_unsupported_codepoints
                        .insert(c as u32)
                {
                    // This will probably still render OK, it just may be weirdly wide.
                    warn!("unsupported codepoint: {:?} (0x{:x})", c, c as u32);
                }
                caret.x += dx;
                // TODO: dubious; should really increment at the end, but this breaks stuff.
                glyph_ix += 1;
                let mut glyph = self.inner.scaled_glyph(c);
                glyph.position = caret.into();
                if let Some(outlined) = self.inner.outline_glyph(glyph) {
                    outlined_glyph_ix += 1;
                    line_glyphs.insert(
                        glyph_ix,
                        LayoutGlyph {
                            _source: c,
                            glyph: outlined,
                        },
                    );
                }
            }
            glyphs.push(LineGlyphs {
                glyphs: line_glyphs,
            });

            caret.x = 0.0;
            caret.y += line_height;
        }
        FontLayout::new(
            glyphs,
            out_format_instructions,
            settings,
            self.sample_ratio(),
        )
    }

    pub fn last_layout(&self) -> Option<FontLayout> {
        self.cached_layout
            .try_lock_short("Font::last_layout()")
            .unwrap()
            .as_ref()
            .map(|(_, l)| l.clone())
    }
}

// let's not talk about this
impl Clone for Font {
    fn clone(&self) -> Self {
        // unwrap() because this can only fail on a parse failure, which it can't if we already
        // successfully constructed `self`.
        let font =
            FontVec::try_from_vec(self.inner.font().as_slice().iter().copied().collect_vec())
                .unwrap();
        let scale = self.inner.scale();
        Self {
            inner: font.into_scaled(scale),
            cached_layout: GgMutex::default(),
            max_glyph_width: self.max_glyph_width,
        }
    }
}

#[derive(Clone)]
struct LayoutGlyph {
    _source: char,
    glyph: OutlinedGlyph,
}

impl LayoutGlyph {
    fn px_bounds(&self) -> Rect {
        self.glyph.px_bounds().into()
    }
}

#[derive(Clone)]
struct LineGlyphs {
    glyphs: BTreeMap<usize, LayoutGlyph>,
}

impl LineGlyphs {
    fn start_of_line_ix(&self) -> Option<usize> {
        self.glyphs.keys().min().copied()
    }
    fn end_of_line_ix(&self) -> Option<usize> {
        self.glyphs.keys().max().copied()
    }

    fn min_x(&self) -> Option<f32> {
        self.glyphs
            .values()
            .map(|glyph| glyph.px_bounds().left())
            .min_f32()
    }
    fn max_x(&self) -> Option<f32> {
        self.glyphs
            .values()
            .map(|glyph| glyph.px_bounds().right())
            .max_f32()
    }
    fn min_y(&self) -> Option<f32> {
        self.glyphs
            .values()
            .map(|glyph| glyph.px_bounds().top())
            .min_f32()
    }
    fn max_y(&self) -> Option<f32> {
        self.glyphs
            .values()
            .map(|glyph| glyph.px_bounds().bottom())
            .max_f32()
    }

    fn take(&self, max_glyph_ix: usize) -> impl Iterator<Item = &OutlinedGlyph> {
        self.glyphs
            .iter()
            .take_while(move |(i, _)| **i <= max_glyph_ix)
            .map(|(_, g)| &g.glyph)
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
pub struct FontLayout {
    glyphs_by_line: Vec<LineGlyphs>,
    bounds_by_line: Vec<Rect>,
    format_instructions: BTreeMap<usize, FormatInstruction>,
    settings: FontRenderSettings,
    sample_ratio: f32,
}

impl FontLayout {
    fn new(
        glyphs_by_line: Vec<LineGlyphs>,
        format_instructions: BTreeMap<usize, FormatInstruction>,
        settings: FontRenderSettings,
        sample_ratio: f32,
    ) -> Self {
        check_false!(glyphs_by_line.is_empty()); // untested case
        check!(sample_ratio.is_finite());
        check_gt!(sample_ratio, 0.0);

        let mut bounds_by_line = Vec::new();
        // Calculate width using all lines:
        let mut bounds = glyphs_by_line
            .iter()
            .fold(Rect::default(), |bounds, line_glyphs| {
                match (line_glyphs.min_x(), line_glyphs.max_x()) {
                    (Some(0.0), Some(0.0)) => bounds,
                    (Some(mut min_x), Some(max_x)) => {
                        // Kludge: it's a bit weird that min_x can be negative.
                        if min_x < 0.0 {
                            check_ge!(min_x, -sample_ratio);
                            min_x = 0.0;
                        }
                        bounds.union(&Rect::from_coords(
                            Vec2 {
                                x: min_x,
                                y: bounds.top(),
                            },
                            Vec2 {
                                x: max_x,
                                y: bounds.bottom(),
                            },
                        ))
                    }
                    _ => bounds,
                }
            });
        // Calculate height per-line:
        for line_glyphs in &glyphs_by_line {
            bounds = match (line_glyphs.min_y(), line_glyphs.max_y()) {
                (Some(0.0), Some(0.0)) => bounds,
                (Some(mut min_y), Some(max_y)) => {
                    // Kludge: it's a bit weird that min_y can be negative.
                    if min_y < 0.0 {
                        check_gt!(min_y, -sample_ratio);
                        min_y = 0.0;
                    }
                    bounds.union(&Rect::from_coords(
                        Vec2 {
                            x: bounds.left(),
                            y: min_y,
                        },
                        Vec2 {
                            x: bounds.right(),
                            y: max_y,
                        },
                    ))
                }
                _ => bounds,
            };
            bounds_by_line.push(bounds);
        }
        check_eq!(glyphs_by_line.len(), bounds_by_line.len());
        Self {
            glyphs_by_line,
            bounds_by_line,
            format_instructions,
            settings,
            sample_ratio,
        }
    }

    #[must_use]
    pub fn with_settings(mut self, settings: FontRenderSettings) -> Self {
        check!(self.settings.is_same_layout(&settings));
        self.settings = settings;
        self
    }

    pub fn render_to_sprite(
        &self,
        object_ctx: &mut ObjectContext,
    ) -> Result<Sprite, FontRenderError> {
        let mut reader =
            GlyphReader::new(self, self.settings.max_glyph_ix).ok_or(FontRenderError::Empty)?;
        let width = reader.width();
        let height = reader.height();
        if width as f32 > self.settings.max_width * self.sample_ratio
            || height as f32 > self.settings.max_height * self.sample_ratio
        {
            Err(FontRenderError::TooLarge)
        } else {
            check!(width != 0);
            check!(height != 0);
            Ok(Sprite::add_from_texture_deferred(
                object_ctx,
                Box::new(move |resource_handler| {
                    resource_handler.texture.wait_load_reader_rgba(
                        &mut reader,
                        width,
                        height,
                        vk::Format::R8G8B8A8_UNORM,
                    )
                }),
            ))
        }
    }

    pub fn bounds_for_max_glyph_ix(&self, max_glyph_ix: usize) -> Option<Rect> {
        self.glyphs_by_line
            .iter()
            .zip(self.bounds_by_line.iter())
            .filter_map(|(line_glyphs, line_bounds)| {
                if line_glyphs.start_of_line_ix()? <= max_glyph_ix {
                    Some(*line_bounds / self.sample_ratio)
                } else {
                    None
                }
            })
            .next_back()
    }

    pub fn max_glyph_ix_for_bounds(&self, bounds: Rect) -> Option<usize> {
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
    pub max_glyph_ix: usize,
    pub text_wrap_mode: TextWrapMode,
    pub do_parse_rich_text: bool,
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
            && self.do_parse_rich_text == other.do_parse_rich_text
    }
}

impl AxisAlignedExtent for FontRenderSettings {
    fn extent(&self) -> Vec2 {
        Vec2 {
            x: self.max_width,
            y: self.max_height,
        }
    }

    fn centre(&self) -> Vec2 {
        self.half_widths()
    }
}

impl Default for FontRenderSettings {
    fn default() -> Self {
        Self {
            max_width: f32::INFINITY,
            max_height: f32::INFINITY,
            max_glyph_ix: usize::MAX,
            text_wrap_mode: TextWrapMode::default(),
            do_parse_rich_text: false,
        }
    }
}

impl PartialEq for FontRenderSettings {
    fn eq(&self, other: &Self) -> bool {
        gg_float::is_finite(self.max_width) == gg_float::is_finite(other.max_width)
            && (!gg_float::is_finite(self.max_width) || self.max_width == other.max_width)
            && gg_float::is_finite(self.max_height) == gg_float::is_finite(other.max_height)
            && (!gg_float::is_finite(self.max_height) || self.max_height == other.max_height)
            && self.max_glyph_ix == other.max_glyph_ix
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
    inner: Option<Vec<OutlinedGlyph>>,
    format_instructions: Option<BTreeMap<usize, FormatInstruction>>,
    all_px_bounds: Rect,
}

impl GlyphReader {
    fn new(layout: &FontLayout, max_glyph_ix: usize) -> Option<Self> {
        if max_glyph_ix == 0 {
            return None;
        }
        let glyphs = layout
            .glyphs_by_line
            .iter()
            .flat_map(|line_glyphs| line_glyphs.take(max_glyph_ix))
            .cloned()
            .collect_vec();
        let Some(scaled_bounds) = layout.bounds_for_max_glyph_ix(max_glyph_ix) else {
            // e.g. because the layout characters start with whitespace?
            warn!("no bounds for max_glyph_ix = {max_glyph_ix}");
            return None;
        };
        let all_px_bounds = scaled_bounds * layout.sample_ratio;
        Some(Self {
            inner: Some(glyphs),
            format_instructions: Some(layout.format_instructions.clone()),
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
        let mut format_instructions = self
            .format_instructions
            .take()
            .ok_or(std::io::Error::from(ErrorKind::UnexpectedEof))?;
        let mut colour = Colour::black();

        // Zero out the buffer first.
        for val in buf.iter_mut() {
            *val = 0;
        }

        for (i, glyph) in glyphs.into_iter().enumerate() {
            // Handle formatting.
            if let Some(&format_ix) = format_instructions.keys().next() {
                check_le!(i, format_ix);
                if i == format_ix {
                    match format_instructions.pop_first() {
                        None => unreachable!(),
                        Some((_, FormatInstruction::SetColourTo(new_colour))) => {
                            colour = new_colour;
                        }
                    }
                }
            }
            // Rasterise.
            let bounds = glyph.px_bounds();
            let img_left = bounds.min.x as u32 - self.all_px_bounds.left() as u32;
            let img_top = bounds.min.y as u32 - self.all_px_bounds.top() as u32;
            glyph.draw(|x, y, v| {
                // Below calls should never really fail.
                let Some(x) = (img_left + x).to_i32() else {
                    error!("glyph x out of range: {}", img_left + x);
                    return;
                };
                let Some(y) = (img_top + y).to_i32() else {
                    error!("glyph y out of range: {}", img_top + y);
                    return;
                };
                let px = Vec2i { x, y }.as_index(self.width(), self.height()) * 4;
                buf[px..px + 3].copy_from_slice(&colour.as_bytes()[..3]);
                buf[px + 3] = buf[px + 3].saturating_add((v * colour.a * 255.0) as u8);
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

    nickname: Option<String>,
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
            nickname: None,
        }
    }

    pub fn font(&self) -> &Font {
        &self.font
    }

    pub fn set_text(&mut self, text: impl AsRef<str>) {
        self.text_to_set = Some(text.as_ref().to_string());
    }
    pub fn set_nickname(&mut self, nickname: impl AsRef<str>) {
        self.nickname = Some(nickname.as_ref().to_string());
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
                .layout(&text, self.render_settings.clone())
                .render_to_sprite(ctx.object_mut())
            {
                Ok(next_sprite) => {
                    if let Some(nickname) = self.nickname.as_ref() {
                        next_sprite.set_name(format!("{nickname}Sprite"));
                    }
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
    fn on_load(&mut self, ctx: &mut LoadContext) -> Result<Option<RenderItem>> {
        ctx.object().transform_mut().scale = Vec2::one() / self.font.sample_ratio();
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
        if let Some(sprite) = self.sprite.as_mut()
            && let Some(settings) = self.last_render_settings.as_ref()
        {
            let clip = Rect::new(ctx.absolute_transform().centre, settings.half_widths());
            if clip.top_left().is_nan() {
                warn!("NaN clipping boundary? {clip:?}");
            }
            if clip.bottom_right().is_nan() {
                warn!("NaN clipping boundary? {clip:?}");
            }
            sprite.set_clip(clip);
        }
    }

    fn as_gui_object(&mut self) -> Option<&mut dyn GuiObject> {
        Some(self)
    }
    fn as_renderable_object(&mut self) -> Option<&mut dyn RenderableObject> {
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

impl RenderableObject for Label {
    fn on_render(&mut self, render_ctx: &mut RenderContext) {
        render_ctx.wait_upload_textures();
    }

    fn shader_execs(&self) -> Vec<ShaderExec> {
        Vec::new()
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
