use imgui::Condition;
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
    fn get_type(&self) -> ObjectType { ObjectType::gg_canvas() }

    fn on_update_begin(&mut self, ctx: &mut UpdateContext<ObjectType>) {
        ctx.object_mut().remove_children();
    }

    fn on_update_end(&mut self, ctx: &mut UpdateContext<ObjectType>) {
        for (render_item, render_info) in self.render_items.drain(..).zip(self.render_infos.drain(..)) {
            ctx.object_mut().add_child(GgInternalCanvasItem::create(render_item, render_info));
        }
        self.viewport = ctx.viewport_mut().inner();
        if ctx.input().pressed(KeyCode::Grave) {
            let is_debug_enabled = ctx.scene().is_debug_enabled();
            ctx.scene_mut().set_debug_enabled(!is_debug_enabled);
        }
    }

    fn as_gui_object(&self) -> Option<&dyn GuiObject<ObjectType>> {
        Some(self)
    }
}

impl<ObjectType: ObjectTypeEnum> GuiObject<ObjectType> for GgInternalCanvas {
    fn on_gui(&self, ctx: &UpdateContext<ObjectType>) -> ImGuiCommandChain {
        if ctx.scene().is_debug_enabled() {
            ImGuiCommandChain::new()
                .window(
                    "Collision",
                    |win| win.size([300., 110.], Condition::FirstUseEver),
                    ImGuiCommandChain::new()
                        .text_wrapped(format!("Hello world: {} canvas items", ctx.object().children().len()))
                        .separator()
                        .get_mouse_pos(|mouse_pos| {
                            ImGuiCommandChain::new()
                                .text(format!(
                                    "Mouse Position: ({:.1},{:.1})",
                                    mouse_pos.x, mouse_pos.y
                                ))
                        }),
                )
        } else {
            ImGuiCommandChain::default()
        }
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
use crate::core::scene::GuiObject;
use crate::core::vk::AdjustedViewport;
use crate::gui::command::ImGuiCommandChain;
