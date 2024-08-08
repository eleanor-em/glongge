use glongge_derive::{partially_derive_scene_object, register_scene_object};
use crate::core::ObjectTypeEnum;
use crate::core::prelude::*;
use crate::shader::{BasicShader, get_shader, Shader, vertex};

#[register_scene_object]
pub struct GgInternalCanvas {
    render_items: Vec<RenderItem>,
    render_infos: Vec<RenderInfo>,
    viewport: AdjustedViewport,
}

impl GgInternalCanvas {
    pub fn line(&mut self, start: Vec2, end: Vec2, width: f64, col: Colour) {
        if self.viewport.contains_point(start) || self.viewport.contains_point(end) {
            self.render_items.push(vertex::line(start, end, width));
            self.render_infos.push(RenderInfo {
                col: col.into(),
                shader_id: get_shader(BasicShader::name()),
                ..Default::default()
            });
        }
    }
    pub fn rect(&mut self, top_left: Vec2, bottom_right: Vec2, col: Colour) {
        if self.viewport.contains_point(top_left) || self.viewport.contains_point(bottom_right) {
            let half_widths = (bottom_right - top_left) / 2;
            let centre = top_left + half_widths;
            self.render_items.push(vertex::rectangle_with_uv(centre, half_widths));
            self.render_infos.push(RenderInfo {
                col: col.into(),
                shader_id: get_shader(BasicShader::name()),
                ..Default::default()
            });
        }
    }
}

#[partially_derive_scene_object]
impl<ObjectType: ObjectTypeEnum> SceneObject<ObjectType> for GgInternalCanvas {
    fn name(&self) -> String { "Canvas".to_string() }
    fn get_type(&self) -> ObjectType { ObjectType::gg_canvas() }

    fn on_update_begin(&mut self, ctx: &mut UpdateContext<ObjectType>) {
        ctx.object_mut().remove_children();
    }

    fn on_update_end(&mut self, ctx: &mut UpdateContext<ObjectType>) {
        for (render_item, render_info) in self.render_items.drain(..).zip(self.render_infos.drain(..)) {
            ctx.object_mut().add_child(GgInternalCanvasItem::create(render_item, render_info));
        }
        self.viewport = ctx.viewport().inner();
    }
}

#[register_scene_object]
pub struct GgInternalCanvasItem {
    render_item: RenderItem,
    render_info: RenderInfo,
}

impl GgInternalCanvasItem {
    pub fn create<ObjectType: ObjectTypeEnum>(render_item: RenderItem, render_info: RenderInfo) -> AnySceneObject<ObjectType> {
        AnySceneObject::new(Self { render_item, render_info })
    }
}

#[partially_derive_scene_object]
impl<ObjectType: ObjectTypeEnum> SceneObject<ObjectType> for GgInternalCanvasItem {
    fn get_type(&self) -> ObjectType { ObjectType::gg_canvas_item() }

    fn on_load(&mut self, _object_ctx: &mut ObjectContext<ObjectType>, _resource_handler: &mut ResourceHandler) -> Result<Option<RenderItem>> {
        Ok(Some(self.render_item.clone()))
    }
    fn as_renderable_object(&mut self) -> Option<&mut dyn RenderableObject<ObjectType>> {
        Some(self)
    }
}

impl<ObjectType: ObjectTypeEnum> RenderableObject<ObjectType> for GgInternalCanvasItem {
    fn render_info(&self) -> RenderInfo {
        self.render_info.clone()
    }
}

pub use GgInternalCanvas as Canvas;
use crate::core::vk::AdjustedViewport;