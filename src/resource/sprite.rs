use crate::core::render::VertexDepth;
use crate::core::scene::{GuiCommand, GuiObject};
use crate::core::update::{ObjectContext, RenderContext};
use crate::resource::ResourceHandler;
use crate::shader::vertex;
use crate::util::{collision::BoxCollider, gg_iter::GgIter};
use crate::{
    core::prelude::*,
    resource::texture::{MaterialId, Texture},
};
use glongge_derive::partially_derive_scene_object;
use num_traits::ToPrimitive;
use std::cell::RefMut;

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
enum SpriteState {
    Hide,
    #[default]
    Show,
    ShouldHide,
    ShouldShow,
    ShouldUpdate,
}

type DeferredTextureLoader = Box<dyn FnOnce(&ResourceHandler) -> Result<Texture>>;

#[derive(Default)]
pub struct GgInternalSprite {
    pub(crate) textures: Vec<Texture>,
    areas: Vec<Rect>,
    materials: Vec<MaterialId>,
    material_indices: Vec<usize>,
    render_item: RenderItem,

    elapsed_us: u128,
    frame_time_ms: Vec<u32>,

    paused: bool,
    state: SpriteState,
    last_state: SpriteState,

    name: String,

    deferred: Option<DeferredTextureLoader>,
}

impl GgInternalSprite {
    fn add_from_textures(ctx: &mut LoadContext, textures: Vec<Texture>) -> Sprite {
        check_false!(textures.is_empty());
        let areas = textures.iter().map(Texture::as_rect).collect_vec();
        let frame_time_ms = textures
            .iter()
            .map(|tex| {
                tex.duration()
                    .map_or(1000, |d| u32::try_from(d.as_millis()).unwrap_or(u32::MAX))
            })
            .collect_vec();
        let render_item = vertex::rectangle(Vec2::zero(), textures[0].half_widths());
        let material_indices = (0..areas.len()).collect_vec();
        let inner = Some(ctx.object_mut().add_child(Self {
            textures,
            areas,
            materials: Vec::new(),
            material_indices,
            frame_time_ms,
            render_item,
            paused: false,
            elapsed_us: 0,
            state: SpriteState::Show,
            last_state: SpriteState::Show,
            name: "Sprite".to_string(),
            deferred: None,
        }));
        Sprite { inner }
    }
    fn add_from_texture_deferred(
        object_ctx: &mut ObjectContext,
        get_texture: DeferredTextureLoader,
    ) -> Sprite {
        let inner = Some(object_ctx.add_child(Self {
            textures: Vec::new(),
            areas: Vec::new(),
            materials: Vec::new(),
            material_indices: Vec::new(),
            render_item: RenderItem::default(),
            frame_time_ms: Vec::new(),
            paused: false,
            elapsed_us: 0,
            state: SpriteState::Show,
            last_state: SpriteState::Show,
            name: "Sprite".to_string(),
            deferred: Some(get_texture),
        }));
        Sprite { inner }
    }

    fn add_from_tileset(
        ctx: &mut LoadContext,
        texture: Texture,
        tile_count: Vec2i,
        tile_size: Vec2i,
        offset: Vec2i,
        margin: Vec2i,
    ) -> Sprite {
        let frame_count = (tile_count.x as usize) * (tile_count.y as usize);
        check_gt!(frame_count, 0);
        let textures = vec![texture; frame_count];
        let areas = Vec2i::range_from_zero(tile_count)
            .map(|(tile_x, tile_y)| {
                let top_left = offset
                    + tile_x * (tile_size + margin).x * Vec2i::right()
                    + tile_y * (tile_size + margin).y * Vec2i::down();
                Rect::new((top_left + tile_size / 2).into(), (tile_size / 2).into())
            })
            .collect_vec();
        let frame_time_ms = vec![1000; frame_count];
        let render_item = vertex::rectangle(Vec2::zero(), tile_size.as_vec2() / 2.0);
        let material_indices = (0..areas.len()).collect_vec();
        let inner = Some(ctx.object_mut().add_child(Self {
            textures,
            areas,
            materials: Vec::new(),
            material_indices,
            render_item,
            frame_time_ms,
            paused: false,
            elapsed_us: 0,
            state: SpriteState::Show,
            last_state: SpriteState::Show,
            name: "Sprite".to_string(),
            deferred: None,
        }));
        Sprite { inner }
    }

    fn set_frame_orders(&mut self, frames: Vec<usize>) {
        self.material_indices = frames;
    }

    fn set_depth(&mut self, depth: VertexDepth) {
        self.render_item.depth = depth;
        if self.state == SpriteState::Show {
            self.state = SpriteState::ShouldUpdate;
        }
    }
    fn set_blend_col(&mut self, col: Colour) {
        self.render_item = self.render_item.clone().with_blend_col(col);
        if self.state == SpriteState::Show {
            self.state = SpriteState::ShouldUpdate;
        }
    }
    pub fn set_clip(&mut self, clip: Rect) {
        self.render_item = self.render_item.clone().with_clip(clip);
        if self.state == SpriteState::Show {
            self.state = SpriteState::ShouldUpdate;
        }
    }
    pub fn set_name(&mut self, name: impl AsRef<str>) {
        self.name = name.as_ref().to_string();
    }

    pub fn max_extent(&self) -> Vec2 {
        self.areas
            .iter()
            .copied()
            .reduce(|a, b| {
                a.with_centre(Vec2::zero())
                    .union(&b.with_centre(Vec2::zero()))
            })
            .unwrap_or_default()
            .extent()
    }

    pub fn textures_ready(&self) -> bool {
        !self.render_item.is_empty() && self.textures.iter().all(Texture::is_ready)
    }

    pub fn frame(&self) -> usize {
        let elapsed_ms = self.elapsed_us / 1000;
        let total_animation_time_ms = u128::from(self.frame_time_ms.iter().sum::<u32>());
        let cycle_elapsed_ms = elapsed_ms % total_animation_time_ms;
        self.frame_time_ms
            .iter()
            .copied()
            .cumsum()
            .filter(|&ms| cycle_elapsed_ms >= u128::from(ms))
            .count()
    }
}

#[partially_derive_scene_object]
impl SceneObject for GgInternalSprite {
    fn gg_type_name(&self) -> String {
        self.name.clone()
    }

    fn on_load(&mut self, ctx: &mut LoadContext) -> Result<Option<RenderItem>> {
        if let Some(deferred) = self.deferred.take() {
            let texture = deferred(ctx.resource())?;
            let areas = vec![texture.as_rect()];
            let frame_time_ms = vec![
                texture
                    .duration()
                    .map_or(1000, |d| u32::try_from(d.as_millis()).unwrap_or(u32::MAX)),
            ];
            let render_item = vertex::rectangle(Vec2::zero(), texture.half_widths());
            let material_indices = (0..areas.len()).collect_vec();
            self.textures = vec![texture];
            self.areas = areas;
            self.material_indices = material_indices;
            self.render_item = render_item;
            self.frame_time_ms = frame_time_ms;
        }

        self.materials = self
            .textures
            .iter()
            .zip(&self.areas)
            .map(|(tex, area)| {
                ctx.resource()
                    .texture
                    .create_material_from_texture(tex, area)
            })
            .collect_vec();

        check_false!(self.textures.is_empty());
        check_eq!(self.textures.len(), self.areas.len());
        check_eq!(self.textures.len(), self.materials.len());
        check_false!(self.render_item.is_empty());

        Ok(
            if self.state == SpriteState::Show
                || self.state == SpriteState::ShouldShow
                || self.state == SpriteState::ShouldUpdate
            {
                self.last_state = SpriteState::Show;
                Some(self.render_item.clone())
            } else {
                self.state = SpriteState::Hide;
                self.last_state = self.state;
                None
            },
        )
    }

    fn on_update(&mut self, ctx: &mut UpdateContext) {
        if self.paused {
            return;
        }
        self.elapsed_us += ctx.delta().as_micros();
    }

    fn as_renderable_object(&mut self) -> Option<&mut dyn RenderableObject> {
        Some(self)
    }
    fn as_gui_object(&mut self) -> Option<&mut dyn GuiObject> {
        Some(self)
    }
}

impl RenderableObject for GgInternalSprite {
    fn on_render(&mut self, render_ctx: &mut RenderContext) {
        match self.state {
            SpriteState::Hide | SpriteState::Show => {}
            SpriteState::ShouldHide => {
                if self.last_state == SpriteState::Show {
                    render_ctx.remove_render_item();
                }
                self.state = SpriteState::Hide;
            }
            SpriteState::ShouldShow => {
                if self.last_state == SpriteState::Hide {
                    render_ctx.insert_render_item(&self.render_item);
                }
                self.state = SpriteState::Show;
            }
            SpriteState::ShouldUpdate => {
                check_eq!(self.last_state, SpriteState::Show);
                render_ctx.update_render_item(&self.render_item);
                self.state = SpriteState::Show;
            }
        }
        self.last_state = self.state;
    }
    fn shader_execs(&self) -> Vec<ShaderExec> {
        check_eq!(self.state, SpriteState::Show);
        let frame = self.frame();
        check_lt!(frame, self.material_indices.len());
        let material_index = self.material_indices[frame];
        let material_id = self.materials[material_index];
        if self.textures_ready() {
            vec![ShaderExec {
                material_id,
                ..Default::default()
            }]
        } else {
            Vec::new()
        }
    }
}

impl GuiObject for GgInternalSprite {
    fn on_gui(&mut self, _ctx: &UpdateContext, _selected: bool) -> GuiCommand {
        let state = self.state;
        let textures_ready = self.textures_ready();
        let texture_ids = self.textures.iter().map(Texture::id).collect_vec();
        let depth = self.render_item.depth;
        GuiCommand::new(move |ui| {
            ui.add(egui::Label::new(format!("state: {state:?}")).selectable(false));
            ui.add(egui::Label::new(format!("textures_ready: {textures_ready}")).selectable(false));
            ui.add(egui::Label::new(format!("texture ids: {texture_ids:?}")).selectable(false));
            ui.add(egui::Label::new(format!("depth: {depth:?}")).selectable(false));
        })
    }
}

#[derive(Clone, Default, PartialEq, Eq)]
pub struct Sprite {
    pub(crate) inner: Option<TreeSceneObject>,
}

impl Sprite {
    pub fn add_from_tileset(
        ctx: &mut LoadContext,
        texture: Texture,
        tile_count: Vec2i,
        tile_size: Vec2i,
        offset: Vec2i,
        margin: Vec2i,
    ) -> Sprite {
        GgInternalSprite::add_from_tileset(ctx, texture, tile_count, tile_size, offset, margin)
    }
    pub fn add_from_single_extent(
        ctx: &mut LoadContext,
        texture: Texture,
        top_left: Vec2i,
        extent: Vec2i,
    ) -> Sprite {
        Self::add_from_tileset(ctx, texture, Vec2i::one(), extent, top_left, Vec2i::zero())
    }
    pub fn add_from_single_rect(ctx: &mut LoadContext, texture: Texture, rect: Rect) -> Sprite {
        Self::add_from_single_extent(
            ctx,
            texture,
            rect.top_left().as_vec2int_lossy(),
            rect.extent().as_vec2int_lossy(),
        )
    }
    pub fn add_from_single_coords(
        ctx: &mut LoadContext,
        texture: Texture,
        top_left: Vec2i,
        bottom_right: Vec2i,
    ) -> Sprite {
        Self::add_from_single_extent(ctx, texture, top_left, bottom_right - top_left)
    }
    pub fn add_from_file(ctx: &mut LoadContext, filename: impl AsRef<str>) -> Result<Sprite> {
        Ok(GgInternalSprite::add_from_textures(
            ctx,
            vec![ctx.resource().texture.wait_load_file(filename)?],
        ))
    }
    pub fn add_from_file_animated(
        ctx: &mut LoadContext,
        filename: impl AsRef<str>,
    ) -> Result<Sprite> {
        Ok(GgInternalSprite::add_from_textures(
            ctx,
            ctx.resource().texture.wait_load_file_animated(filename)?,
        ))
    }
    pub fn add_from_tileset_file(
        ctx: &mut LoadContext,
        filename: impl AsRef<str>,
        tile_count: Vec2i,
        tile_size: Vec2i,
        offset: Vec2i,
        margin: Vec2i,
    ) -> Result<Sprite> {
        Ok(GgInternalSprite::add_from_tileset(
            ctx,
            ctx.resource().texture.wait_load_file(filename)?,
            tile_count,
            tile_size,
            offset,
            margin,
        ))
    }
    pub fn add_from_single_extent_file(
        ctx: &mut LoadContext,
        filename: impl AsRef<str>,
        top_left: Vec2i,
        extent: Vec2i,
    ) -> Result<Sprite> {
        Self::add_from_tileset_file(ctx, filename, Vec2i::one(), extent, top_left, Vec2i::zero())
    }
    pub fn add_from_single_rect_file(
        ctx: &mut LoadContext,
        filename: impl AsRef<str>,
        rect: Rect,
    ) -> Result<Sprite> {
        Self::add_from_single_extent_file(
            ctx,
            filename,
            rect.top_left().as_vec2int_lossy(),
            rect.extent().as_vec2int_lossy(),
        )
    }
    pub fn add_from_single_coords_file(
        ctx: &mut LoadContext,
        filename: impl AsRef<str>,
        top_left: Vec2i,
        bottom_right: Vec2i,
    ) -> Result<Sprite> {
        Self::add_from_single_extent_file(ctx, filename, top_left, bottom_right - top_left)
    }

    pub(crate) fn add_from_texture(ctx: &mut LoadContext, texture: Texture) -> Sprite {
        let extent = texture.extent().as_vec2int_lossy();
        Self::add_from_single_extent(ctx, texture, Vec2i::zero(), extent)
    }
    pub(crate) fn add_from_texture_deferred(
        object_ctx: &mut ObjectContext,
        get_texture: DeferredTextureLoader,
    ) -> Sprite {
        GgInternalSprite::add_from_texture_deferred(object_ctx, get_texture)
    }

    #[must_use]
    pub fn with_depth(self, depth: VertexDepth) -> Self {
        self.inner_unwrap().render_item.depth = depth;
        if self.inner_unwrap().state == SpriteState::Show {
            self.inner_unwrap().state = SpriteState::ShouldUpdate;
        }
        self
    }
    #[must_use]
    pub fn with_blend_col(self, col: Colour) -> Self {
        self.inner_unwrap().set_blend_col(col);
        if self.inner_unwrap().state == SpriteState::Show {
            self.inner_unwrap().state = SpriteState::ShouldUpdate;
        }
        self
    }
    #[must_use]
    pub fn with_fixed_ms_per_frame(self, ms: u32) -> Self {
        {
            let mut inner = self.inner_unwrap();
            inner.frame_time_ms = vec![ms; inner.frame_time_ms.len()];
        }
        self
    }
    #[must_use]
    pub fn with_frame_time_ms(self, times: Vec<u32>) -> Self {
        {
            let mut inner = self.inner_unwrap();
            check_eq!(times.len(), inner.material_indices.len());
            inner.frame_time_ms = times;
        }
        self
    }
    #[must_use]
    pub fn with_frame_time_factor(self, factor: f32) -> Self {
        {
            let mut inner = self.inner_unwrap();
            inner.frame_time_ms = inner
                .frame_time_ms
                .iter()
                .map(|t| (*t as f32) * factor)
                .map(|t| t.round().to_u32().unwrap_or(u32::MAX))
                .collect_vec();
        }
        self
    }
    #[must_use]
    pub fn with_frame_orders(self, frames: Vec<usize>) -> Self {
        self.inner_unwrap().set_frame_orders(frames);
        self
    }
    #[must_use]
    pub fn with_hidden(self) -> Self {
        {
            self.inner_unwrap().state = SpriteState::Hide;
        }
        self
    }
    #[must_use]
    pub fn with_name(self, name: impl AsRef<str>) -> Self {
        {
            self.inner_unwrap().name = name.as_ref().to_string();
        }
        self
    }
    #[must_use]
    pub fn with_translated(self, by: Vec2) -> Self {
        self.translate(by);
        self
    }
    #[must_use]
    pub fn with_scaled(self, by: Vec2) -> Self {
        self.scale(by);
        self
    }
    #[must_use]
    pub fn with_paused(self) -> Self {
        self.pause();
        self
    }

    pub fn reset(&self) {
        self.inner_unwrap().elapsed_us = 0;
    }
    pub fn pause(&self) {
        self.inner_unwrap().paused = true;
    }
    pub fn play(&self) {
        self.inner_unwrap().paused = false;
    }
    pub fn hide(&self) {
        self.inner_unwrap().state = SpriteState::ShouldHide;
    }
    pub fn show(&self) {
        self.inner_unwrap().state = SpriteState::ShouldShow;
    }

    pub fn as_box_collider(&self) -> BoxCollider {
        BoxCollider::from_aa_extent(self)
    }

    pub fn set_name(&self, name: impl AsRef<str>) {
        self.inner_unwrap().set_name(name);
    }
    pub fn set_depth(&self, depth: VertexDepth) {
        self.inner_unwrap().set_depth(depth);
    }

    pub fn set_blend_col(&self, col: Colour) {
        self.inner_unwrap().set_blend_col(col);
    }
    pub fn set_clip(&self, clip: Rect) {
        self.inner_unwrap().set_clip(clip);
    }

    pub fn translate(&self, by: Vec2) {
        self.inner.as_ref().unwrap().transform_mut().centre += by;
    }
    pub fn scale(&self, by: Vec2) {
        let new_scale = self
            .inner
            .as_ref()
            .unwrap()
            .transform_mut()
            .scale
            .component_wise(by);
        self.inner.as_ref().unwrap().transform_mut().scale = new_scale;
    }
    pub fn set_centre(&self, centre: Vec2) {
        self.inner.as_ref().unwrap().transform_mut().centre = centre;
    }
    pub fn set_rotation(&self, rotation: f32) {
        self.inner.as_ref().unwrap().transform_mut().rotation = rotation;
    }

    pub(crate) fn inner_unwrap(&self) -> RefMut<'_, GgInternalSprite> {
        self.inner
            .as_ref()
            .unwrap()
            .downcast_mut::<GgInternalSprite>()
            .unwrap()
    }

    pub fn textures_ready(&self) -> bool {
        self.inner_unwrap().textures_ready()
    }

    pub fn frame(&self) -> usize {
        self.inner_unwrap().frame()
    }
}

impl AxisAlignedExtent for Sprite {
    fn extent(&self) -> Vec2 {
        let scale = self.inner.as_ref().unwrap().transform_mut().scale;
        self.inner_unwrap().max_extent().component_wise(scale)
    }

    fn centre(&self) -> Vec2 {
        self.inner.as_ref().unwrap().transform().centre
    }
}
