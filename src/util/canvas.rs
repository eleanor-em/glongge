use crate::core::prelude::*;
use crate::core::render::VertexDepth;
use crate::shader::{Shader, SpriteShader, get_shader, vertex};
use glongge_derive::partially_derive_scene_object;

#[derive(Default)]
pub struct GgInternalCanvas {
    render_items: Vec<RenderItem>,
    depth: Option<VertexDepth>,
}

impl GgInternalCanvas {
    pub fn new() -> Self {
        Self::default()
    }
    #[must_use]
    pub fn with_depth(mut self, depth: VertexDepth) -> Self {
        self.set_depth(depth);
        self
    }

    pub fn line(&mut self, start: Vec2, end: Vec2, width: f32, col: Colour) {
        self.render_items
            .push(vertex::line(start, end, width).with_blend_col(col));
    }
    pub fn rect(&mut self, top_left: Vec2, bottom_right: Vec2, col: Colour) {
        let half_widths = (bottom_right - top_left) / 2;
        let centre = top_left + half_widths;
        self.rect_centred(centre, half_widths, col);
    }
    pub fn rect_centred(&mut self, centre: Vec2, half_widths: Vec2, col: Colour) {
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

    pub fn set_depth(&mut self, depth: VertexDepth) {
        self.depth = Some(depth);
    }
}

#[partially_derive_scene_object]
impl SceneObject for GgInternalCanvas {
    fn gg_type_name(&self) -> String {
        "Canvas".to_string()
    }

    fn on_load(
        &mut self,
        _object_ctx: &mut ObjectContext,
        _resource_handler: &mut ResourceHandler,
    ) -> Result<Option<RenderItem>> {
        if self.depth.is_none() {
            self.depth = Some(VertexDepth::Front(u16::MAX));
        }
        Ok(Some(RenderItem::default()))
    }

    fn on_update_begin(&mut self, _ctx: &mut UpdateContext) {
        self.render_items.clear();
    }

    fn on_update_end(&mut self, _ctx: &mut UpdateContext) {
        if self.render_items.is_empty() {
            // XXX: if there is nothing but the canvas, the game won't even start, because it has to
            // render something.
            self.line(Vec2::zero(), Vec2::zero(), 0.0, Colour::empty());
        }
    }

    fn as_renderable_object(&mut self) -> Option<&mut dyn RenderableObject> {
        Some(self)
    }
}
impl RenderableObject for GgInternalCanvas {
    fn on_render(&mut self, render_ctx: &mut RenderContext) {
        if let Some(mut ri) = self
            .render_items
            .clone()
            .into_iter()
            .rev()
            .reduce(RenderItem::concat)
        {
            ri.depth = self.depth.unwrap();
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
