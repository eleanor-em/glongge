use crate::core::prelude::{
    AxisAlignedExtent, Collider, Colour, FixedUpdateContext, GenericCollider, RenderItem,
    RenderableObject, SceneObject, ShaderExec, Transform, UpdateContext, Vec2,
};
use crate::core::render::VertexDepth;
use crate::core::scene::{GuiCommand, GuiObject};
use crate::core::update::RenderContext;
use crate::gui::EditCell;
use crate::resource::sprite::Sprite;
use crate::util::canvas::Canvas;
use crate::util::collision::Polygonal;
use crate::{check, check_is_some};
use glongge_derive::partially_derive_scene_object;
use itertools::Itertools;
use std::fmt::{Display, Formatter};

#[derive(Debug, Default, Clone, bincode::Encode, bincode::Decode)]
pub struct BoxCollider {
    pub(crate) centre: Vec2,
    pub(crate) extent: Vec2,
}

#[derive(Debug, Default, Clone, bincode::Encode, bincode::Decode)]
pub struct BoxCollider3d {
    pub(crate) centre: Vec2,
    pub(crate) extent: Vec2,
    pub(crate) front: f32,
    pub(crate) back: f32,
}

#[derive(Debug, Clone, bincode::Encode, bincode::Decode)]
pub struct ConvexCollider {
    pub(crate) vertices: Vec<Vec2>,
    pub(crate) normals_cached: Vec<Vec2>,
    pub(crate) centre_cached: Vec2,
    pub(crate) extent_cached: Vec2,
}

#[derive(Debug, Clone, bincode::Encode, bincode::Decode)]
pub struct OrientedBoxCollider {
    pub(crate) centre: Vec2,
    pub(crate) rotation: f32,
    pub(crate) unrotated_half_widths: Vec2,
    pub(crate) extent: Vec2,
}

pub struct GgInternalCollisionShape {
    base_collider: GenericCollider,
    collider: GenericCollider,
    emitting_tags: Vec<&'static str>,
    listening_tags: Vec<&'static str>,

    // For GUI:
    // <RenderItem, should_be_updated>
    wireframe: RenderItem,
    show_wireframe: bool,
    last_show_wireframe: bool,
    extent_cell_receiver_x: EditCell<f32>,
    extent_cell_receiver_y: EditCell<f32>,
    centre_cell_receiver_x: EditCell<f32>,
    centre_cell_receiver_y: EditCell<f32>,
}

impl GgInternalCollisionShape {
    pub fn from_collider<C: Collider>(
        collider: C,
        emitting_tags: &[&'static str],
        listening_tags: &[&'static str],
    ) -> Self {
        let base_collider = collider.into_generic();
        let mut rv = Self {
            base_collider: base_collider.clone(),
            collider: base_collider,
            emitting_tags: emitting_tags.to_vec(),
            listening_tags: listening_tags.to_vec(),
            wireframe: RenderItem::default(),
            show_wireframe: false,
            last_show_wireframe: false,
            extent_cell_receiver_x: EditCell::new(),
            extent_cell_receiver_y: EditCell::new(),
            centre_cell_receiver_x: EditCell::new(),
            centre_cell_receiver_y: EditCell::new(),
        };
        rv.regenerate_wireframe();
        rv
    }

    pub fn from_object<O: SceneObject, C: Collider>(object: &O, collider: C) -> Self {
        Self::from_collider(collider, &object.emitting_tags(), &object.listening_tags())
    }
    pub fn from_object_sprite<O: SceneObject>(object: &O, sprite: &Sprite) -> Self {
        Self::from_collider(
            sprite.as_box_collider(),
            &object.emitting_tags(),
            &object.listening_tags(),
        )
    }

    pub fn collider(&self) -> &GenericCollider {
        &self.collider
    }

    fn regenerate_wireframe(&mut self) {
        self.wireframe =
            RenderItem::from_raw_vertices(self.base_collider.as_triangles().into_flattened())
                .with_depth(VertexDepth::max_value());
    }

    pub fn show_wireframe(&mut self) {
        self.show_wireframe = true;
    }
    pub fn hide_wireframe(&mut self) {
        self.show_wireframe = false;
    }
}

#[partially_derive_scene_object]
impl SceneObject for GgInternalCollisionShape {
    fn gg_type_name(&self) -> String {
        format!("CollisionShape [{:?}]", self.collider.get_type()).to_string()
    }

    fn on_ready(&mut self, ctx: &mut UpdateContext) {
        check_is_some!(ctx.object().parent(), "CollisionShapes must have a parent");
    }
    fn on_update_begin(&mut self, ctx: &mut UpdateContext) {
        self.update_transform(ctx.absolute_transform());
    }
    fn on_fixed_update(&mut self, ctx: &mut FixedUpdateContext) {
        self.update_transform(ctx.absolute_transform());
    }

    fn on_update(&mut self, ctx: &mut UpdateContext) {
        self.update_transform(ctx.absolute_transform());
        if self.show_wireframe {
            let mut canvas = ctx
                .object_mut()
                .first_other_as_mut::<Canvas>()
                .expect("No Canvas object in scene!");
            match &self.collider {
                GenericCollider::Compound(compound) => {
                    let mut colours = [
                        Colour::green(),
                        Colour::red(),
                        Colour::blue(),
                        Colour::magenta(),
                        Colour::yellow(),
                    ];
                    colours.reverse();
                    for inner in &compound.inner {
                        let col = *colours.last().unwrap();
                        colours.rotate_right(1);
                        inner.draw_polygonal(&mut canvas, col);
                    }
                }
                GenericCollider::OrientedBox(c) => c.draw_polygonal(&mut canvas, Colour::green()),
                GenericCollider::Box(c) => c.draw_polygonal(&mut canvas, Colour::green()),
                GenericCollider::Convex(c) => c.draw_polygonal(&mut canvas, Colour::green()),
                GenericCollider::Null => {}
            }
        }
    }

    fn on_update_end(&mut self, ctx: &mut UpdateContext) {
        self.update_transform(ctx.absolute_transform());
    }

    fn emitting_tags(&self) -> Vec<&'static str> {
        self.emitting_tags.clone()
    }
    fn listening_tags(&self) -> Vec<&'static str> {
        self.listening_tags.clone()
    }

    fn as_renderable_object(&mut self) -> Option<&mut dyn RenderableObject> {
        Some(self)
    }
    fn as_gui_object(&mut self) -> Option<&mut dyn GuiObject> {
        if self.show_wireframe {
            Some(self)
        } else {
            None
        }
    }
}

impl GgInternalCollisionShape {
    pub(crate) fn update_transform(&mut self, next_transform: Transform) {
        self.collider = self.base_collider.transformed(&next_transform);
    }
}

impl RenderableObject for GgInternalCollisionShape {
    #[allow(clippy::if_not_else)] // clearer as written
    fn on_render(&mut self, render_ctx: &mut RenderContext) {
        if self.show_wireframe {
            if !self.last_show_wireframe {
                render_ctx.insert_render_item(&self.wireframe);
            } else {
                render_ctx.update_render_item(&self.wireframe);
            }
        }
        if !self.show_wireframe && self.last_show_wireframe {
            render_ctx.remove_render_item();
        }
        self.last_show_wireframe = self.show_wireframe;
    }
    fn shader_execs(&self) -> Vec<ShaderExec> {
        check!(self.show_wireframe);
        vec![ShaderExec {
            blend_col: Colour::cyan().with_alpha(0.2),
            ..Default::default()
        }]
    }
}

impl GuiObject for GgInternalCollisionShape {
    fn on_gui(&mut self, _ctx: &UpdateContext, selected: bool) -> GuiCommand {
        if !selected {
            self.extent_cell_receiver_x.clear_state();
            self.extent_cell_receiver_y.clear_state();
            self.centre_cell_receiver_x.clear_state();
            self.centre_cell_receiver_y.clear_state();
        }
        let extent = self.collider.extent();
        let (next_x, next_y) = (
            self.extent_cell_receiver_x.try_recv(),
            self.extent_cell_receiver_y.try_recv(),
        );
        if next_x.is_some() || next_y.is_some() {
            self.collider = self.collider.with_extent(Vec2 {
                x: next_x.unwrap_or(extent.x).max(0.1),
                y: next_y.unwrap_or(extent.y).max(0.1),
            });
            self.regenerate_wireframe();
        }
        self.extent_cell_receiver_x.update_live(extent.x);
        self.extent_cell_receiver_y.update_live(extent.y);

        let extent_cell_sender_x = self.extent_cell_receiver_x.sender();
        let extent_cell_sender_y = self.extent_cell_receiver_y.sender();

        let centre = self.collider.centre();
        let (next_x, next_y) = (
            self.centre_cell_receiver_x.try_recv(),
            self.centre_cell_receiver_y.try_recv(),
        );
        if next_x.is_some() || next_y.is_some() {
            self.collider = self.collider.with_centre(Vec2 {
                x: next_x.unwrap_or(centre.x),
                y: next_y.unwrap_or(centre.y),
            });
            self.regenerate_wireframe();
        }
        self.centre_cell_receiver_x.update_live(centre.x);
        self.centre_cell_receiver_y.update_live(centre.y);

        let centre_cell_sender_x = self.centre_cell_receiver_x.sender();
        let centre_cell_sender_y = self.centre_cell_receiver_y.sender();

        let emitting_tags = self.emitting_tags.join(", ");
        let listening_tags = self.listening_tags.join(", ");

        let collider = self.collider.clone();
        GuiCommand::new(move |ui| {
            ui.label(collider.to_string());
            ui.add(egui::Label::new("Extent").selectable(false));
            collider
                .extent()
                .build_gui(ui, 0.1, extent_cell_sender_x, extent_cell_sender_y);
            ui.end_row();
            ui.add(egui::Label::new("Centre").selectable(false));
            collider
                .centre()
                .build_gui(ui, 0.1, centre_cell_sender_x, centre_cell_sender_y);
            ui.end_row();
            ui.add(egui::Label::new(format!("Emitting: {emitting_tags}")).selectable(false));
            ui.end_row();
            ui.add(egui::Label::new(format!("Listening: {listening_tags}")).selectable(false));
            ui.end_row();
        })
    }
}

impl Display for GenericCollider {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match &self {
            GenericCollider::Null => {
                write!(f, "<null>")
            }
            GenericCollider::Box(_) => {
                write!(f, "Box")
            }
            GenericCollider::OrientedBox(inner) => {
                write!(f, "OrientedBox: {} deg.", inner.rotation.to_degrees())
            }
            GenericCollider::Convex(inner) => {
                write!(f, "Convex: {} edges", inner.normals_cached.len())
            }
            GenericCollider::Compound(inner) => {
                write!(
                    f,
                    "Compound: {} pieces, {:?} edges",
                    inner.inner.len(),
                    inner
                        .inner
                        .iter()
                        .map(|c| c.normals_cached.len())
                        .collect_vec()
                )
            }
        }
    }
}
