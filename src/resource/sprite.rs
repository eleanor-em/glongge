use std::cell::RefCell;
use std::rc::Rc;
use num_traits::Zero;
use glongge_derive::{partially_derive_scene_object, register_scene_object};
use crate::{
    core::{
        AnySceneObject,
        ObjectTypeEnum,
        prelude::*,
        util::{
            collision::BoxCollider,
            gg_iter::GgIter
        }
    },
    shader,
    resource::texture::{Texture, TextureSubArea}
};
use crate::core::render::VertexDepth;
use crate::core::update::RenderContext;
use crate::shader::{get_shader, Shader, SpriteShader};

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
enum SpriteState {
    Hide,
    #[default]
    Show,
    ShouldHide,
    ShouldShow,
    ShouldUpdate
}

#[register_scene_object]
pub struct GgInternalSprite {
    texture: Texture,
    areas: Vec<TextureSubArea>,
    elapsed_us: u128,
    frame_time_ms: Vec<u32>,
    frame: usize,
    render_item: RenderItem,

    paused: bool,
    state: SpriteState,
    last_state: SpriteState,
}

pub struct Sprite {
    inner: Rc<RefCell<GgInternalSprite>>,
    extent: Vec2,
    collider: BoxCollider,
}

impl Default for Sprite {
    fn default() -> Self {
        Self {
            inner: Rc::new(RefCell::new(GgInternalSprite::default())),
            extent: Vec2::zero(),
            collider: BoxCollider::default(),
        }
    }
}

impl GgInternalSprite {
    fn from_tileset<ObjectType: ObjectTypeEnum>(
        object_ctx: &mut ObjectContext<ObjectType>,
        texture: Texture,
        tile_count: Vec2Int,
        tile_size: Vec2Int,
        offset: Vec2Int,
        margin: Vec2Int
    ) -> Sprite {
        let areas = Vec2Int::range_from_zero(tile_count)
            .map(|(tile_x, tile_y)| {
                let top_left = offset
                    + tile_x * (tile_size + margin).x * Vec2Int::right()
                    + tile_y * (tile_size + margin).y * Vec2Int::down();
                TextureSubArea::new(top_left + tile_size / 2, tile_size / 2)
            })
            .collect_vec();
        let frame_time_ms = vec![1000; areas.len()];
        let render_item = RenderItem::new(shader::vertex::rectangle_with_uv(
            Vec2::zero(), (tile_size / 2).into())
        );
        let inner = Rc::new(RefCell::new(Self {
            texture, areas, frame_time_ms, render_item,
            paused: false,
            elapsed_us: 0,
            frame: 0,
            state: SpriteState::Show,
            last_state: SpriteState::Show,
        }));
        object_ctx.add_child(AnySceneObject::from_rc(inner.clone()));
        let extent = tile_size.into();
        let collider = BoxCollider::from_centre(Vec2::zero(), extent / 2);
        Sprite { inner, extent, collider }
    }

    fn current_frame(&self) -> TextureSubArea {
        self.areas[self.frame]
    }

    fn set_depth(&mut self, depth: VertexDepth) {
        self.render_item.depth = depth;
        if self.state == SpriteState::Show {
            self.state = SpriteState::ShouldUpdate;
        }
    }
}

#[partially_derive_scene_object]
impl<ObjectType: ObjectTypeEnum> SceneObject<ObjectType> for GgInternalSprite {
    fn get_type(&self) -> ObjectType { ObjectType::gg_sprite() }

    fn on_load(
        &mut self,
        _object_ctx: &mut ObjectContext<ObjectType>,
        _resource_handler: &mut ResourceHandler
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

    fn on_fixed_update(&mut self, _ctx: &mut UpdateContext<ObjectType>) {
        if self.paused {
            return;
        }
        self.elapsed_us += FIXED_UPDATE_INTERVAL_US;
        let elapsed_ms = self.elapsed_us / 1000;
        let total_animation_time_ms = u128::from(self.frame_time_ms.iter().sum::<u32>());
        let cycle_elapsed_ms = elapsed_ms % total_animation_time_ms;
        self.frame = self.frame_time_ms.iter().copied()
            .cumsum()
            .filter(|&ms| cycle_elapsed_ms >= u128::from(ms))
            .count();
        check_lt!(self.frame, self.areas.len());
    }

    fn as_renderable_object(&mut self) -> Option<&mut dyn RenderableObject<ObjectType>> {
        Some(self)
    }
}

impl<ObjectType: ObjectTypeEnum> RenderableObject<ObjectType> for GgInternalSprite {
    fn on_render(&mut self, render_ctx: &mut RenderContext) {
        match self.state {
            SpriteState::Hide | SpriteState::Show => {},
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
    fn render_info(&self) -> RenderInfo {
        check_eq!(self.state, SpriteState::Show);
        RenderInfo {
            shader_id: get_shader(SpriteShader::name()),
            texture_id: self.texture.id(),
            texture_sub_area: self.current_frame(),
            ..Default::default()
        }
    }
}

impl Sprite {
    pub fn from_tileset<ObjectType: ObjectTypeEnum>(
        object_ctx: &mut ObjectContext<ObjectType>,
        texture: Texture,
        tile_count: Vec2Int,
        tile_size: Vec2Int,
        offset: Vec2Int,
        margin: Vec2Int
    ) -> Sprite {
        GgInternalSprite::from_tileset(object_ctx, texture, tile_count, tile_size, offset, margin)
    }
    pub fn from_single_extent<ObjectType: ObjectTypeEnum>(
        object_ctx: &mut ObjectContext<ObjectType>,
        texture: Texture,
        extent: Vec2Int,
        top_left: Vec2Int
    ) -> Sprite {
        Self::from_tileset(
            object_ctx,
            texture,
            Vec2Int::one(),
            extent,
            top_left,
            Vec2Int::zero()
        )
    }
    pub fn from_single_coords<ObjectType: ObjectTypeEnum>(
        object_ctx: &mut ObjectContext<ObjectType>,
        texture: Texture,
        top_left: Vec2Int,
        bottom_right: Vec2Int,
    ) -> Sprite {
        Self::from_single_extent(
            object_ctx,
            texture,
            bottom_right - top_left,
            top_left
        )
    }
    pub(crate) fn from_texture<ObjectType: ObjectTypeEnum>(
        object_ctx: &mut ObjectContext<ObjectType>,
        texture: Texture
    ) -> Sprite {
        let extent = texture.aa_extent().as_vec2int_lossy();
        Self::from_single_extent(object_ctx, texture, extent, Vec2Int::zero())
    }

    #[must_use]
    pub fn with_depth(self, depth: VertexDepth) -> Self {
        self.inner.borrow_mut().render_item.depth = depth;
        self
    }
    #[must_use]
    pub fn with_fixed_ms_per_frame(self, ms: u32) -> Self {
        {
            let mut inner = self.inner.borrow_mut();
            inner.frame_time_ms = vec![ms; inner.areas.len()];
        }
        self
    }
    #[must_use]
    pub fn with_frame_time_ms(self, times: Vec<u32>) -> Self {
        {
            let mut inner = self.inner.borrow_mut();
            check_eq!(times.len(), inner.areas.len());
            inner.frame_time_ms = times;
        }
        self
    }
    #[must_use]
    pub fn with_frame_orders(self, frames: Vec<usize>) -> Self {
        {
            let mut inner = self.inner.borrow_mut();
            inner.areas = frames.into_iter().map(|i| inner.areas[i]).collect();
        }
        self
    }
    #[must_use]
    pub fn with_hidden(self) -> Self {
        {
            self.inner.borrow_mut().state = SpriteState::Hide;
        }
        self
    }

    pub fn reset(&mut self) {
        self.inner.borrow_mut().elapsed_us = 0;
    }
    pub fn pause(&mut self) {
        self.inner.borrow_mut().paused = true;
    }
    pub fn play(&mut self) {
        self.inner.borrow_mut().paused = false;
    }
    pub fn hide(&mut self) {
        self.inner.borrow_mut().state = SpriteState::ShouldHide;
    }
    pub fn show(&mut self) {
        self.inner.borrow_mut().state = SpriteState::ShouldShow;
    }

    pub fn as_box_collider(&self) -> BoxCollider {
        self.collider.clone()
    }

    pub fn set_depth(&mut self, depth: VertexDepth) {
        self.inner.borrow_mut().set_depth(depth);
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
