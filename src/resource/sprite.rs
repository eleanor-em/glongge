use crate::core::render::VertexDepth;
use crate::core::update::RenderContext;
use crate::shader::{Shader, SpriteShader, get_shader, vertex};
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

#[derive(Default)]
pub struct GgInternalSprite {
    materials: Vec<MaterialId>,
    material_indices: Vec<usize>,
    elapsed_us: u128,
    frame_time_ms: Vec<u32>,
    frame: usize,
    render_item: RenderItem,

    paused: bool,
    state: SpriteState,
    last_state: SpriteState,

    name: String,
}

impl GgInternalSprite {
    fn add_from_textures(
        object_ctx: &mut ObjectContext,
        resource_handler: &mut ResourceHandler,
        textures: Vec<Texture>,
    ) -> Sprite {
        let areas = textures.iter().map(Texture::as_rect).collect_vec();
        let frame_time_ms = textures
            .iter()
            .map(|tex| {
                tex.duration()
                    .map_or(1000, |d| u32::try_from(d.as_millis()).unwrap_or(u32::MAX))
            })
            .collect_vec();
        let render_item = vertex::rectangle(Vec2::zero(), textures[0].half_widths());
        let extent = textures[0].aa_extent();
        let materials = textures
            .into_iter()
            .zip(&areas)
            .map(|(tex, area)| resource_handler.texture.material_from_texture(&tex, area))
            .collect_vec();
        let material_indices = (0..areas.len()).collect_vec();
        let inner = Some(object_ctx.add_child(Self {
            materials,
            material_indices,
            frame_time_ms,
            render_item,
            paused: false,
            elapsed_us: 0,
            frame: 0,
            state: SpriteState::Show,
            last_state: SpriteState::Show,
            name: "Sprite".to_string(),
        }));
        let collider = BoxCollider::from_centre(Vec2::zero(), extent / 2);
        Sprite {
            inner,
            extent,
            collider,
        }
    }

    fn add_from_tileset(
        object_ctx: &mut ObjectContext,
        resource_handler: &mut ResourceHandler,
        texture: Texture,
        tile_count: Vec2i,
        tile_size: Vec2i,
        offset: Vec2i,
        margin: Vec2i,
    ) -> Sprite {
        let areas = Vec2i::range_from_zero(tile_count)
            .map(|(tile_x, tile_y)| {
                let top_left = offset
                    + tile_x * (tile_size + margin).x * Vec2i::right()
                    + tile_y * (tile_size + margin).y * Vec2i::down();
                Rect::new((top_left + tile_size / 2).into(), (tile_size / 2).into())
            })
            .collect_vec();
        let frame_time_ms = vec![1000; areas.len()];
        let render_item = vertex::rectangle(Vec2::zero(), (tile_size / 2).into());
        let textures = vec![texture; areas.len()];
        let materials = textures
            .into_iter()
            .zip(&areas)
            .map(|(tex, area)| resource_handler.texture.material_from_texture(&tex, area))
            .collect_vec();
        let material_indices = (0..areas.len()).collect_vec();
        let inner = Some(object_ctx.add_child(Self {
            materials,
            material_indices,
            frame_time_ms,
            render_item,
            paused: false,
            elapsed_us: 0,
            frame: 0,
            state: SpriteState::Show,
            last_state: SpriteState::Show,
            name: "Sprite".to_string(),
        }));
        let extent = tile_size.into();
        let collider = BoxCollider::from_centre(Vec2::zero(), extent / 2);
        Sprite {
            inner,
            extent,
            collider,
        }
    }

    fn set_frame_orders(&mut self, frames: Vec<usize>) {
        for frame in &frames {
            check_lt!(*frame, self.materials.len());
        }
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
    }
}

#[partially_derive_scene_object]
impl SceneObject for GgInternalSprite {
    fn gg_type_name(&self) -> String {
        self.name.clone()
    }

    fn on_load(
        &mut self,
        _object_ctx: &mut ObjectContext,
        _resource_handler: &mut ResourceHandler,
    ) -> Result<Option<RenderItem>> {
        Ok(if self.state == SpriteState::Show {
            self.last_state = SpriteState::Show;
            Some(self.render_item.clone())
        } else {
            self.state = SpriteState::Hide;
            self.last_state = SpriteState::Hide;
            None
        })
    }

    fn on_update(&mut self, ctx: &mut UpdateContext) {
        if self.paused {
            return;
        }
        self.elapsed_us += ctx.delta().as_micros();
        let elapsed_ms = self.elapsed_us / 1000;
        let total_animation_time_ms = u128::from(self.frame_time_ms.iter().sum::<u32>());
        let cycle_elapsed_ms = elapsed_ms % total_animation_time_ms;
        self.frame = self
            .frame_time_ms
            .iter()
            .copied()
            .cumsum()
            .filter(|&ms| cycle_elapsed_ms >= u128::from(ms))
            .count();
        check_lt!(self.frame, self.material_indices.len());
    }

    fn as_renderable_object(&mut self) -> Option<&mut dyn RenderableObject> {
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
        check_lt!(self.frame, self.material_indices.len());
        let material_index = self.material_indices[self.frame];
        let material_id = self.materials[material_index];
        vec![ShaderExec {
            shader_id: get_shader(SpriteShader::name()),
            material_id,
            ..Default::default()
        }]
    }
}

#[derive(Clone)]
pub struct Sprite {
    inner: Option<TreeSceneObject>,
    extent: Vec2,
    collider: BoxCollider,
}

impl Default for Sprite {
    fn default() -> Self {
        Self {
            inner: None,
            extent: Vec2::zero(),
            collider: BoxCollider::default(),
        }
    }
}

impl Sprite {
    pub fn add_from_file(
        object_ctx: &mut ObjectContext,
        resource_handler: &mut ResourceHandler,
        filename: impl AsRef<str>,
    ) -> Result<Sprite> {
        Ok(GgInternalSprite::add_from_textures(
            object_ctx,
            resource_handler,
            vec![resource_handler.texture.wait_load_file(filename)?],
        ))
    }
    pub fn add_from_file_animated(
        object_ctx: &mut ObjectContext,
        resource_handler: &mut ResourceHandler,
        filename: impl AsRef<str>,
    ) -> Result<Sprite> {
        Ok(GgInternalSprite::add_from_textures(
            object_ctx,
            resource_handler,
            resource_handler.texture.wait_load_file_animated(filename)?,
        ))
    }
    pub fn add_from_tileset(
        object_ctx: &mut ObjectContext,
        resource_handler: &mut ResourceHandler,
        texture: Texture,
        tile_count: Vec2i,
        tile_size: Vec2i,
        offset: Vec2i,
        margin: Vec2i,
    ) -> Sprite {
        GgInternalSprite::add_from_tileset(
            object_ctx,
            resource_handler,
            texture,
            tile_count,
            tile_size,
            offset,
            margin,
        )
    }
    pub fn add_from_single_extent(
        object_ctx: &mut ObjectContext,
        resource_handler: &mut ResourceHandler,
        texture: Texture,
        top_left: Vec2i,
        extent: Vec2i,
    ) -> Sprite {
        Self::add_from_tileset(
            object_ctx,
            resource_handler,
            texture,
            Vec2i::one(),
            extent,
            top_left,
            Vec2i::zero(),
        )
    }
    pub fn add_from_single_coords(
        object_ctx: &mut ObjectContext,
        resource_handler: &mut ResourceHandler,
        texture: Texture,
        top_left: Vec2i,
        bottom_right: Vec2i,
    ) -> Sprite {
        Self::add_from_single_extent(
            object_ctx,
            resource_handler,
            texture,
            top_left,
            bottom_right - top_left,
        )
    }

    pub(crate) fn add_from_texture(
        object_ctx: &mut ObjectContext,
        resource_handler: &mut ResourceHandler,
        texture: Texture,
    ) -> Sprite {
        let extent = texture.aa_extent().as_vec2int_lossy();
        Self::add_from_single_extent(object_ctx, resource_handler, texture, Vec2i::zero(), extent)
    }
    #[must_use]
    pub fn with_collider_half_widths(mut self, half_widths: Vec2) -> Self {
        self.collider = self.collider.with_half_widths(half_widths);
        self
    }
    #[must_use]
    pub fn with_collider_extent(mut self, extent: Vec2) -> Self {
        self.collider = self.collider.with_extent(extent);
        self
    }
    #[must_use]
    pub fn with_collider_centre(mut self, centre: Vec2) -> Self {
        self.collider = self.collider.with_centre(centre);
        self
    }
    #[must_use]
    pub fn with_collider_top_left(mut self, top_left: Vec2) -> Self {
        self.collider = self.collider.with_centre(top_left - self.centre());
        self
    }
    #[must_use]
    pub fn with_depth(self, depth: VertexDepth) -> Self {
        self.inner_unwrap().render_item.depth = depth;
        self
    }
    #[must_use]
    pub fn with_blend_col(self, col: Colour) -> Self {
        self.inner_unwrap().set_blend_col(col);
        self
    }
    #[must_use]
    pub fn with_fixed_ms_per_frame(self, ms: u32) -> Self {
        {
            let mut inner = self.inner_unwrap();
            inner.frame_time_ms = vec![ms; inner.materials.len()];
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

    pub fn reset(&mut self) {
        self.inner_unwrap().elapsed_us = 0;
    }
    pub fn pause(&mut self) {
        self.inner_unwrap().paused = true;
    }
    pub fn play(&mut self) {
        self.inner_unwrap().paused = false;
    }
    pub fn hide(&mut self) {
        self.inner_unwrap().state = SpriteState::ShouldHide;
    }
    pub fn show(&mut self) {
        self.inner_unwrap().state = SpriteState::ShouldShow;
    }

    pub fn as_box_collider(&self) -> BoxCollider {
        self.collider.clone()
    }

    pub fn set_depth(&mut self, depth: VertexDepth) {
        self.inner_unwrap().set_depth(depth);
    }

    pub fn set_blend_col(&self, col: Colour) {
        self.inner_unwrap().set_blend_col(col);
    }

    fn inner_unwrap(&self) -> RefMut<'_, GgInternalSprite> {
        self.inner
            .as_ref()
            .unwrap()
            .downcast_mut::<GgInternalSprite>()
            .unwrap()
    }
}

impl AxisAlignedExtent for Sprite {
    fn aa_extent(&self) -> Vec2 {
        self.extent
    }

    fn centre(&self) -> Vec2 {
        Vec2::zero()
    }
}
