#[register_scene_object]
pub struct GgInternalContainer<ObjectType> {
    label: String,
    children: Vec<AnySceneObject<ObjectType>>,
}

impl<ObjectType: ObjectTypeEnum> GgInternalContainer<ObjectType> {
    pub fn create(label: impl AsRef<str>, children: Vec<AnySceneObject<ObjectType>>) -> AnySceneObject<ObjectType> {
        AnySceneObject::new(Self { label: label.as_ref().to_string(), children })
    }
}

#[partially_derive_scene_object]
impl<ObjectType: ObjectTypeEnum> SceneObject<ObjectType> for GgInternalContainer<ObjectType> {
    fn name(&self) -> String { self.label.clone() }
    fn get_type(&self) -> ObjectType { ObjectType::gg_container() }

    fn on_load(&mut self, object_ctx: &mut ObjectContext<ObjectType>, _resource_handler: &mut ResourceHandler) -> Result<Option<RenderItem>> {
        object_ctx.add_vec(self.children.drain(..).collect_vec());
        Ok(None)
    }
}

use itertools::Itertools;
pub use GgInternalContainer as Container;
use glongge_derive::{partially_derive_scene_object, register_scene_object};
use crate::core::{AnySceneObject, ObjectTypeEnum};
use crate::core::prelude::*;
use crate::resource::ResourceHandler;
