use crate::{core::prelude::*, resource::ResourceHandler, util::linalg::Transform};
use scene::SceneObject;
use std::{
    any::TypeId,
    cell::{Ref, RefCell, RefMut},
    fmt::{Debug, Formatter},
    rc::Rc,
    sync::atomic::{AtomicUsize, Ordering},
};

pub mod builtin;
pub mod config;
pub mod coroutine;
pub mod input;
pub mod prelude;
pub mod render;
pub mod scene;
pub mod update;
pub mod vk;

pub trait ObjectTypeEnum: Clone + Copy + Debug + Eq + PartialEq + Sized + 'static + Send {
    fn as_default(self) -> SceneObjectWrapper<Self>;
    fn as_typeid(self) -> TypeId;
    fn all_values() -> Vec<Self>;
    fn gg_sprite() -> Self;
    fn gg_collider() -> Self;
    fn gg_canvas() -> Self;
    fn gg_container() -> Self;
    fn gg_static_sprite() -> Self;
    fn gg_colliding_sprite() -> Self;
    fn gg_tileset() -> Self;
    fn gg_interactive_spline() -> Self;

    fn preload_all(resource_handler: &mut ResourceHandler) -> Result<()> {
        for value in Self::all_values() {
            value
                .as_default()
                .wrapped
                .borrow_mut()
                .on_preload(resource_handler)?;
        }
        resource_handler.wait_all()?;
        Ok(())
    }
    fn checked_downcast<T: SceneObject<Self> + 'static>(obj: &dyn SceneObject<Self>) -> &T {
        let actual = obj.gg_type_enum().as_typeid();
        let expected = obj.as_any().type_id();
        if actual != expected {
            for value in Self::all_values() {
                check_ne!(
                    value.as_typeid(),
                    actual,
                    format!(
                        "attempt to downcast {:?} -> {:?}",
                        obj.gg_type_enum(),
                        value
                    )
                );
            }
            panic!(
                "attempt to downcast {:?}: type missing? {:?}",
                obj.gg_type_enum(),
                Self::all_values()
            );
        }
        obj.as_any().downcast_ref::<T>().unwrap()
    }
    fn checked_downcast_mut<T: SceneObject<Self> + 'static>(
        obj: &mut dyn SceneObject<Self>,
    ) -> &mut T {
        let actual = obj.gg_type_enum().as_typeid();
        let expected = obj.as_any().type_id();
        if actual != expected {
            for value in Self::all_values() {
                check_ne!(
                    value.as_typeid(),
                    actual,
                    format!(
                        "attempt to downcast {:?} -> {:?}",
                        obj.gg_type_enum(),
                        value
                    )
                );
            }
            panic!(
                "attempt to downcast {:?}: type missing? {:?}",
                obj.gg_type_enum(),
                Self::all_values()
            );
        }
        obj.as_any_mut().downcast_mut::<T>().unwrap()
    }
}

// ObjectId(0) represents the root object.
static NEXT_OBJECT_ID: AtomicUsize = AtomicUsize::new(1);
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct ObjectId(pub(crate) usize);

impl ObjectId {
    fn next() -> Self {
        ObjectId(NEXT_OBJECT_ID.fetch_add(1, Ordering::Relaxed))
    }
    pub(crate) fn is_root(self) -> bool {
        self.0 == 0
    }
    pub(crate) fn root() -> Self {
        ObjectId(0)
    }
}

#[derive(Clone)]
pub struct SceneObjectWrapper<ObjectType> {
    transform: Rc<RefCell<Transform>>,
    pub(crate) wrapped: Rc<RefCell<dyn SceneObject<ObjectType>>>,
    type_id: TypeId, // = TypeId::of::<O: SceneObject<ObjectType>>()
    nickname: Rc<RefCell<Option<String>>>,
}

impl<ObjectType: ObjectTypeEnum> SceneObjectWrapper<ObjectType> {
    pub(crate) fn gg_type_id(&self) -> TypeId {
        self.type_id
    }
    pub(crate) fn transform(&self) -> Transform {
        *self.transform.borrow()
    }
    pub(crate) fn transform_mut(&self) -> RefMut<Transform> {
        self.transform.borrow_mut()
    }
    pub(crate) fn nickname_or_type_name(&self) -> String {
        self.nickname
            .borrow()
            .clone()
            .unwrap_or_else(|| self.wrapped.borrow().type_name())
    }
    pub(crate) fn set_nickname(&mut self, name: impl Into<String>) {
        *self.nickname.borrow_mut() = Some(name.into());
    }
}

pub trait IntoSceneObjectWrapper<ObjectType: ObjectTypeEnum> {
    fn into_wrapper(self) -> SceneObjectWrapper<ObjectType>;
}

impl<ObjectType: ObjectTypeEnum, O: SceneObject<ObjectType>> IntoSceneObjectWrapper<ObjectType>
    for O
{
    fn into_wrapper(self) -> SceneObjectWrapper<ObjectType> {
        SceneObjectWrapper {
            transform: Rc::new(RefCell::new(Transform::default())),
            wrapped: Rc::new(RefCell::new(self)),
            type_id: TypeId::of::<O>(),
            nickname: Rc::new(RefCell::new(None)),
        }
    }
}

#[derive(Clone)]
pub struct TreeSceneObject<ObjectType> {
    pub(crate) scene_object: SceneObjectWrapper<ObjectType>,
    pub(crate) object_id: ObjectId,
    parent_id: ObjectId,
}

impl<ObjectType: ObjectTypeEnum> TreeSceneObject<ObjectType> {
    pub(crate) fn inner_type_id(&self) -> TypeId {
        self.scene_object.type_id
    }
    /// NOTE: Borrows!
    pub fn gg_type_enum(&self) -> ObjectType {
        self.scene_object.wrapped.borrow().gg_type_enum()
    }
    pub fn gg_type_id(&self) -> TypeId {
        self.scene_object.gg_type_id()
    }
    pub fn object_id(&self) -> ObjectId {
        self.object_id
    }
    /// NOTE: Borrows!
    pub fn transform(&self) -> Transform {
        self.scene_object.transform()
    }
    /// NOTE: Borrows!
    pub fn transform_mut(&self) -> RefMut<Transform> {
        self.scene_object.transform_mut()
    }
    /// NOTE: Borrows!
    pub fn emitting_tags(&self) -> Vec<&'static str> {
        self.scene_object.wrapped.borrow().emitting_tags()
    }
    /// NOTE: Borrows!
    pub fn listening_tags(&self) -> Vec<&'static str> {
        self.scene_object.wrapped.borrow().listening_tags()
    }

    pub fn nickname_or_type_name(&self) -> String {
        self.scene_object.nickname_or_type_name()
    }
    pub fn nickname(&self) -> Option<String> {
        self.scene_object.nickname.borrow().clone()
    }
    pub fn set_nickname(&mut self, name: impl Into<String>) {
        self.scene_object.set_nickname(name);
    }
}

impl<ObjectType: ObjectTypeEnum> Debug for TreeSceneObject<ObjectType> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?} ({:?})", self.object_id, self.gg_type_id())
    }
}

pub trait DowncastRef<ObjectType: ObjectTypeEnum> {
    fn downcast<T: SceneObject<ObjectType>>(&self) -> Option<Ref<T>>;
    fn downcast_mut<T: SceneObject<ObjectType>>(&self) -> Option<RefMut<T>>;
    fn checked_downcast<T: SceneObject<ObjectType>>(&self) -> Ref<T>;
    fn checked_downcast_mut<T: SceneObject<ObjectType>>(&self) -> RefMut<T>;
}

impl<ObjectType: ObjectTypeEnum> DowncastRef<ObjectType> for SceneObjectWrapper<ObjectType> {
    fn downcast<T: SceneObject<ObjectType>>(&self) -> Option<Ref<T>> {
        Ref::filter_map(self.wrapped.borrow(), |obj| {
            obj.as_any().downcast_ref::<T>()
        })
        .ok()
    }

    fn downcast_mut<T: SceneObject<ObjectType>>(&self) -> Option<RefMut<T>> {
        RefMut::filter_map(self.wrapped.borrow_mut(), |obj| {
            obj.as_any_mut().downcast_mut::<T>()
        })
        .ok()
    }

    fn checked_downcast<T: SceneObject<ObjectType>>(&self) -> Ref<T> {
        Ref::map(self.wrapped.borrow(), |obj| {
            ObjectType::checked_downcast::<T>(obj)
        })
    }

    fn checked_downcast_mut<T: SceneObject<ObjectType>>(&self) -> RefMut<T> {
        RefMut::map(self.wrapped.borrow_mut(), |obj| {
            ObjectType::checked_downcast_mut::<T>(obj)
        })
    }
}

impl<ObjectType: ObjectTypeEnum> DowncastRef<ObjectType> for TreeSceneObject<ObjectType> {
    fn downcast<T: SceneObject<ObjectType>>(&self) -> Option<Ref<T>> {
        self.scene_object.downcast()
    }
    fn downcast_mut<T: SceneObject<ObjectType>>(&self) -> Option<RefMut<T>> {
        self.scene_object.downcast_mut()
    }
    fn checked_downcast<T: SceneObject<ObjectType>>(&self) -> Ref<T> {
        self.scene_object.checked_downcast()
    }
    fn checked_downcast_mut<T: SceneObject<ObjectType>>(&self) -> RefMut<T> {
        self.scene_object.checked_downcast_mut()
    }
}
