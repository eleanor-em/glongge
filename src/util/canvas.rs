use crate::core::ObjectTypeEnum;
use crate::core::prelude::*;
use crate::core::render::VertexDepth;
use crate::shader::{Shader, SpriteShader, get_shader, vertex};
use glongge_derive::{partially_derive_scene_object, register_scene_object};

#[register_scene_object]
pub struct GgInternalCanvas {
    render_items: Vec<RenderItem>,
    depth: VertexDepth,
}

impl GgInternalCanvas {
    pub fn line(&mut self, start: Vec2, end: Vec2, width: f32, col: Colour) {
        self.render_items
            .push(vertex::line(start, end, width).with_blend_col(col));
    }
    pub fn rect(&mut self, top_left: Vec2, bottom_right: Vec2, col: Colour) {
        let half_widths = (bottom_right - top_left) / 2;
        let centre = top_left + half_widths;
        self.render_items
            .push(vertex::rectangle(centre, half_widths).with_blend_col(col));
    }
    pub fn rect_transformed(
        &mut self,
        transform: Transform,
        top_left: Vec2,
        bottom_right: Vec2,
        col: Colour,
    ) {
        let centre = (top_left + bottom_right) / 2;
        let half_widths = transform.scale.component_wise(centre - top_left);

        let top_left = transform.centre + centre + (-half_widths).rotated(transform.rotation);
        let top_right = transform.centre
            + centre
            + Vec2 {
                x: half_widths.x,
                y: -half_widths.y,
            }
            .rotated(transform.rotation);
        let bottom_right = transform.centre + centre + half_widths.rotated(transform.rotation);
        let bottom_left = transform.centre
            + centre
            + Vec2 {
                x: -half_widths.x,
                y: half_widths.y,
            }
            .rotated(transform.rotation);

        self.render_items.push(
            vertex::quadrilateral(top_left, top_right, bottom_left, bottom_right)
                .with_blend_col(col),
        );
    }
    pub fn circle(&mut self, centre: Vec2, radius: f32, steps: u32, col: Colour) {
        self.render_items
            .push(vertex::circle(centre, radius, steps).with_blend_col(col));
    }

    // TODO: hacky.
    pub fn set_depth(&mut self, depth: VertexDepth) {
        self.depth = depth;
    }
}

#[partially_derive_scene_object]
impl<ObjectType: ObjectTypeEnum> SceneObject<ObjectType> for GgInternalCanvas {
    fn type_name(&self) -> String {
        "Canvas".to_string()
    }
    fn gg_type_enum(&self) -> ObjectType {
        ObjectType::gg_canvas()
    }

    fn on_load(
        &mut self,
        _object_ctx: &mut ObjectContext<ObjectType>,
        _resource_handler: &mut ResourceHandler,
    ) -> Result<Option<RenderItem>> {
        self.depth = VertexDepth::Front(u16::MAX);
        Ok(Some(RenderItem::default()))
    }

    fn on_update_begin(&mut self, _ctx: &mut UpdateContext<ObjectType>) {
        self.render_items.clear();
    }

    fn on_update_end(&mut self, _ctx: &mut UpdateContext<ObjectType>) {
        if self.render_items.is_empty() {
            // XXX: if there is nothing but the canvas, the game won't even start, because it has to
            // render something.
            self.line(Vec2::zero(), Vec2::zero(), 0., Colour::empty());
        }
    }

    fn as_renderable_object(&mut self) -> Option<&mut dyn RenderableObject<ObjectType>> {
        Some(self)
    }
}
impl<ObjectType: ObjectTypeEnum> RenderableObject<ObjectType> for GgInternalCanvas {
    fn on_render(&mut self, render_ctx: &mut RenderContext) {
        if let Some(mut ri) = self
            .render_items
            .clone()
            .into_iter()
            .rev()
            .reduce(RenderItem::concat)
        {
            ri.depth = self.depth;
            render_ctx.update_render_item(&ri);
        }
    }
    fn shader_execs(&self) -> Vec<ShaderExec> {
        vec![ShaderExec {
            shader_id: get_shader(SpriteShader::name()),
            ..Default::default()
        }]
    }
}

use crate::core::update::RenderContext;
pub use GgInternalCanvas as Canvas;
