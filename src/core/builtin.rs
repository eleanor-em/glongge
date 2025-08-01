use crate::core::SceneObjectWrapper;
use crate::core::prelude::*;
use crate::resource::ResourceHandler;
use crate::resource::sprite::Sprite;
use glongge_derive::partially_derive_scene_object;
use itertools::Itertools;
use std::ffi::OsStr;
use std::path::Path;

/// A scene object that serves as a container to group and organise child objects.
/// This container type only provides a name/label for the group without adding any additional
/// functionality.
pub struct GgInternalContainer {
    label: String,
    children: Vec<SceneObjectWrapper>,
}

impl GgInternalContainer {
    pub fn new(label: impl AsRef<str>, children: Vec<SceneObjectWrapper>) -> Self {
        Self {
            label: label.as_ref().to_string(),
            children,
        }
    }
}

#[partially_derive_scene_object]
impl SceneObject for GgInternalContainer {
    fn gg_type_name(&self) -> String {
        self.label.clone()
    }
    fn on_load(
        &mut self,
        object_ctx: &mut ObjectContext,
        _resource_handler: &mut ResourceHandler,
    ) -> Result<Option<RenderItem>> {
        object_ctx.add_vec(self.children.drain(..).collect_vec());
        Ok(None)
    }
}

#[derive(Copy, Clone, Default, Debug)]
enum CreateCoord {
    #[default]
    Zero,
    TopLeft(Vec2),
    Centre(Vec2),
}

/// A pre-packaged [`SceneObject`] that simply draws a sprite with no animation.
pub struct GgInternalStaticSprite {
    filename: String,
    name: Option<String>,
    create_coord: CreateCoord,
    tex_segment: Option<Rect>,
    depth: Option<VertexDepth>,
    sprite: Sprite,
}

impl GgInternalStaticSprite {
    #[must_use]
    pub fn new(filename: impl AsRef<str>) -> Self {
        Self {
            filename: filename.as_ref().to_string(),
            name: None,
            create_coord: CreateCoord::Zero,
            tex_segment: None,
            depth: None,
            sprite: Sprite::default(),
        }
    }
    #[must_use]
    pub fn with_name(mut self, name: impl AsRef<str>) -> Self {
        self.name = Some(name.as_ref().to_string());
        self
    }
    #[must_use]
    pub fn at_top_left(mut self, top_left: impl Into<Vec2>) -> Self {
        self.create_coord = CreateCoord::TopLeft(top_left.into());
        self
    }
    #[must_use]
    pub fn at_centre(mut self, centre: impl Into<Vec2>) -> Self {
        self.create_coord = CreateCoord::Centre(centre.into());
        self
    }
    #[must_use]
    pub fn with_single_coords(
        mut self,
        top_left: impl Into<Vec2>,
        bottom_right: impl Into<Vec2>,
    ) -> Self {
        self.tex_segment = Some(Rect::from_coords(top_left.into(), bottom_right.into()));
        self
    }
    #[must_use]
    pub fn with_single_extent(
        mut self,
        top_left: impl Into<Vec2>,
        extent: impl Into<Vec2>,
    ) -> Self {
        let top_left = top_left.into();
        self.tex_segment = Some(Rect::from_coords(top_left, top_left + extent.into()));
        self
    }
    #[must_use]
    pub fn with_depth(mut self, depth: VertexDepth) -> Self {
        self.depth = Some(depth);
        self
    }
    pub fn build_colliding(self, emitting_tags: Vec<&'static str>) -> GgInternalCollidingSprite {
        GgInternalCollidingSprite {
            inner: self,
            emitting_tags,
        }
    }
}

#[partially_derive_scene_object]
impl SceneObject for GgInternalStaticSprite {
    fn gg_type_name(&self) -> String {
        if let Some(name) = &self.name.as_ref() {
            format!("StaticSprite [{name}]")
        } else if let Some(filename) = Path::new(&self.filename)
            .file_stem()
            .and_then(OsStr::to_str)
        {
            format!("StaticSprite [{filename}]")
        } else {
            "StaticSprite".to_string()
        }
    }

    fn on_load(
        &mut self,
        object_ctx: &mut ObjectContext,
        resource_handler: &mut ResourceHandler,
    ) -> Result<Option<RenderItem>> {
        let sprite = if let Some(tex_segment) = self.tex_segment {
            Sprite::add_from_single_coords(
                object_ctx,
                resource_handler
                    .texture
                    .wait_load_file(self.filename.clone())?,
                tex_segment.top_left().as_vec2int_lossy(),
                tex_segment.bottom_right().as_vec2int_lossy(),
            )
        } else {
            Sprite::add_from_texture(
                object_ctx,
                resource_handler
                    .texture
                    .wait_load_file(self.filename.clone())?,
            )
        };
        if let Some(depth) = self.depth {
            self.sprite = sprite.with_depth(depth);
        } else {
            self.sprite = sprite;
        }
        let centre = match self.create_coord {
            CreateCoord::Zero => Vec2::zero(),
            CreateCoord::TopLeft(v) => v + self.sprite.half_widths(),
            CreateCoord::Centre(v) => v,
        };
        object_ctx.transform_mut().centre = centre;
        Ok(None)
    }
}

/// A pre-packaged [`SceneObject`] similar to [`GgInternalStaticSprite`] but with added collision
/// detection based on the sprite's dimensions.
pub struct GgInternalCollidingSprite {
    inner: GgInternalStaticSprite,
    emitting_tags: Vec<&'static str>,
}

#[partially_derive_scene_object]
impl SceneObject for GgInternalCollidingSprite {
    fn gg_type_name(&self) -> String {
        <GgInternalStaticSprite as SceneObject>::gg_type_name(&self.inner)
            .replace("Static", "Colliding")
    }

    fn on_load(
        &mut self,
        object_ctx: &mut ObjectContext,
        resource_handler: &mut ResourceHandler,
    ) -> Result<Option<RenderItem>> {
        self.inner.on_load(object_ctx, resource_handler)
    }

    fn on_ready(&mut self, ctx: &mut UpdateContext) {
        ctx.object_mut().add_child(CollisionShape::from_collider(
            self.inner.sprite.as_box_collider(),
            &<GgInternalCollidingSprite as SceneObject>::emitting_tags(self),
            &Vec::new(),
        ));
    }

    fn emitting_tags(&self) -> Vec<&'static str> {
        self.emitting_tags.clone()
    }
}

pub use crate::core::builtin::GgInternalContainer as Container;
pub use crate::core::builtin::GgInternalStaticSprite as StaticSprite;
pub use crate::core::builtin::GgInternalStaticSprite as CollidingSprite;
use crate::core::render::VertexDepth;
