use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;
use num_traits::Zero;
use glongge_derive::{partially_derive_scene_object, register_scene_object};
use crate::{
    core::{
        prelude::*,
        util::collision::{BoxCollider, Collider},
        util::linalg::{AxisAlignedExtent, Vec2},
        util::linalg::Vec2Int
    },
    shader,
};
use crate::core::{AnySceneObject, DowncastRef, ObjectTypeEnum};
use crate::core::util::linalg::Transform;
use crate::core::render::{RenderInfo, RenderItem};
use crate::core::scene::SceneObject;
use crate::core::update::{ObjectContext, UpdateContext};
use crate::core::util::gg_iter::GgIter;
use crate::resource::texture::{Texture, TextureSubArea};

#[register_scene_object]
pub struct GgSprite<ObjectType> {
    texture: Texture,
    areas: Vec<TextureSubArea>,
    elapsed_us: u128,
    paused: bool,
    frame_time_ms: Vec<u32>,
    frame: usize,
    object_type: PhantomData<ObjectType>,
}

pub struct BoxedGgSprite<ObjectType> {
    inner: Rc<RefCell<AnySceneObject<ObjectType>>>,
}

impl<ObjectType: ObjectTypeEnum> Default for BoxedGgSprite<ObjectType> {
    fn default() -> Self {
        Self { inner: Rc::new(RefCell::new(Box::new(GgSprite::default()))) }
    }
}

impl<ObjectType: ObjectTypeEnum> GgSprite<ObjectType> {
    pub fn from_texture(object_ctx: &mut ObjectContext<ObjectType>, texture: Texture) -> BoxedGgSprite<ObjectType> {
        let extent = texture.extent();
        Self::from_single_extent(object_ctx, texture, extent.as_vec2int_lossy(), Vec2Int::zero())
    }
    pub fn from_single_extent(
        object_ctx: &mut ObjectContext<ObjectType>,
        texture: Texture,
        extent: Vec2Int,
        top_left: Vec2Int
    ) -> BoxedGgSprite<ObjectType> {
        Self::from_tileset(
            object_ctx,
            texture,
            Vec2Int::one(),
            extent,
            top_left,
            Vec2Int::zero()
        )
    }
    pub fn from_single_coords(
        object_ctx: &mut ObjectContext<ObjectType>,
        texture: Texture,
        top_left: Vec2Int,
        bottom_right: Vec2Int,
    ) -> BoxedGgSprite<ObjectType> {
        Self::from_single_extent(
            object_ctx,
            texture,
            bottom_right - top_left,
            top_left
        )
    }

    pub fn from_tileset(
        object_ctx: &mut ObjectContext<ObjectType>,
        texture: Texture,
        tile_count: Vec2Int,
        tile_size: Vec2Int,
        offset: Vec2Int,
        margin: Vec2Int
    ) -> BoxedGgSprite<ObjectType> {
        let areas = Vec2Int::range_from_zero(tile_count)
            .map(|(tile_x, tile_y)| {
                let top_left = offset
                    + tile_x * (tile_size + margin).x * Vec2Int::right()
                    + tile_y * (tile_size + margin).y * Vec2Int::down();
                TextureSubArea::new(top_left + tile_size / 2, tile_size / 2)
            })
            .collect_vec();
        let frame_time_ms = vec![1000; areas.len()];
        let inner = Box::new(Self {
            texture, areas, frame_time_ms,
            paused: false,
            elapsed_us: 0,
            frame: 0,
            object_type: PhantomData,
        });
        let inner = object_ctx.add_child(inner);
        BoxedGgSprite { inner }
    }

    fn ready(&self) -> bool {
        !self.areas.is_empty()
    }
}

#[partially_derive_scene_object]
impl<ObjectType: ObjectTypeEnum> SceneObject<ObjectType> for GgSprite<ObjectType> {
    fn get_type(&self) -> ObjectType { ObjectType::gg_sprite() }

    fn on_fixed_update(&mut self, _ctx: &mut UpdateContext<ObjectType>) {
        if self.paused {
            return;
        }
        self.elapsed_us += FIXED_UPDATE_INTERVAL_US;
        check!(self.ready());
        let elapsed_ms = self.elapsed_us / 1000;
        let total_animation_time_ms = self.frame_time_ms.iter().sum::<u32>() as u128;
        let cycle_elapsed_ms = elapsed_ms % total_animation_time_ms;
        self.frame = self.frame_time_ms.iter().copied()
            .cumsum()
            .filter(|&ms| cycle_elapsed_ms >= u128::from(ms))
            .count();
        check_lt!(self.frame, self.areas.len());
    }
}

impl<ObjectType: ObjectTypeEnum> BoxedGgSprite<ObjectType> {
    #[must_use]
    pub fn with_fixed_ms_per_frame(self, ms: u32) -> Self {
        {
            let mut inner = self.inner.checked_downcast_mut::<GgSprite::<ObjectType>>();
            inner.frame_time_ms = vec![ms; inner.areas.len()];
        }
        self
    }
    #[must_use]
    pub fn with_frame_time_ms(self, times: Vec<u32>) -> Self {
        {
            let mut inner = self.inner.checked_downcast_mut::<GgSprite::<ObjectType>>();
            check_eq!(times.len(), inner.areas.len());
            inner.frame_time_ms = times;
        }
        self
    }
    #[must_use]
    pub fn with_frame_orders(self, frames: Vec<usize>) -> Self {
        {
            let mut inner = self.inner.checked_downcast_mut::<GgSprite::<ObjectType>>();
            inner.areas = frames.into_iter().map(|i| inner.areas[i]).collect();
        }
        self
    }

    pub fn ready(&self) -> bool {
        let inner = self.inner.checked_downcast::<GgSprite::<ObjectType>>();
        inner.ready()
    }

    pub fn reset(&mut self) {
        let mut inner = self.inner.checked_downcast_mut::<GgSprite::<ObjectType>>();
        inner.elapsed_us = 0;
    }
    pub fn pause(&mut self) {
        let mut inner = self.inner.checked_downcast_mut::<GgSprite::<ObjectType>>();
        inner.paused = true;
    }
    pub fn play(&mut self) {
        let mut inner = self.inner.checked_downcast_mut::<GgSprite::<ObjectType>>();
        inner.paused = false;
    }

    pub fn current_frame(&self) -> TextureSubArea {
        let inner = self.inner.checked_downcast::<GgSprite::<ObjectType>>();
        inner.areas[inner.frame]
    }

    pub fn as_box_collider(&self, transform: Transform) -> Box<dyn Collider> {
        Box::new(BoxCollider::from_transform(transform, self.aa_extent()))
    }

    pub fn render_info_default(&self) -> RenderInfo {
        self.render_info_from(RenderInfo::default())
    }
    pub fn render_info_from(&self, mut render_info: RenderInfo) -> RenderInfo {
        let inner = self.inner.checked_downcast::<GgSprite::<ObjectType>>();
        if inner.ready() {
            render_info.texture = Some(inner.texture.clone());
            render_info.texture_sub_area = self.current_frame();
        }
        render_info
    }

    pub fn create_vertices(&self) -> RenderItem {
        RenderItem::new(shader::vertex::rectangle_with_uv(Vec2::zero(), self.half_widths()))
    }
}

impl<ObjectType: ObjectTypeEnum> AxisAlignedExtent for BoxedGgSprite<ObjectType> {
    fn aa_extent(&self) -> Vec2 {
        self.current_frame().aa_extent()
    }

    fn centre(&self) -> Vec2 {
        self.current_frame().centre()
    }
}
