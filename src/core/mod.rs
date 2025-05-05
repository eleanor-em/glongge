use crate::core::prelude::*;
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

// Start at 1 because ObjectId(0) represents the root object.
static NEXT_OBJECT_ID: AtomicUsize = AtomicUsize::new(1);

/// A unique identifier for scene objects within the engine.
///
/// [`ObjectId`] uses an atomic counter to generate unique IDs, starting at 1.
/// The ID 0 is reserved for the root object, which never actually exists in the scene -- it is only
/// used to ensure a proper tree structure.
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
/// A wrapper around scene objects that provides shared ownership and interior mutability.
///
/// The wrapper manages:
/// - The relative transform (position, rotation, scale)
/// - The actual scene object behind a dynamic `SceneObject` trait
/// - Type information for runtime type checking and safe downcasting
/// - Optional nickname for object identification
pub struct SceneObjectWrapper {
    transform: Rc<RefCell<Transform>>,
    pub(crate) wrapped: Rc<RefCell<dyn SceneObject>>,
    type_id: TypeId, // = TypeId::of::<O: SceneObject>()
    nickname: Rc<RefCell<Option<String>>>,
}

impl SceneObjectWrapper {
    pub(crate) fn gg_type_id(&self) -> TypeId {
        self.type_id
    }
    pub(crate) fn transform(&self) -> Transform {
        *self.transform.borrow()
    }
    pub(crate) fn transform_mut(&self) -> RefMut<Transform> {
        self.transform.borrow_mut()
    }
    pub fn nickname_or_type_name(&self) -> String {
        self.nickname
            .borrow()
            .clone()
            .unwrap_or_else(|| self.wrapped.borrow().gg_type_name())
    }
    pub(crate) fn set_nickname(&mut self, name: impl Into<String>) {
        *self.nickname.borrow_mut() = Some(name.into());
    }
}

/// Trait for converting objects into [`SceneObjectWrapper`].
/// **DO NOT IMPLEMENT THIS TRAIT MANUALLY**. This trait is automatically implemented for all types
/// that implement [`SceneObject`].
pub trait IntoSceneObjectWrapper {
    fn into_wrapper(self) -> SceneObjectWrapper;
}

impl<O: SceneObject> IntoSceneObjectWrapper for O {
    fn into_wrapper(self) -> SceneObjectWrapper {
        SceneObjectWrapper {
            transform: Rc::new(RefCell::new(Transform::default())),
            wrapped: Rc::new(RefCell::new(self)),
            type_id: TypeId::of::<O>(),
            nickname: Rc::new(RefCell::new(None)),
        }
    }
}

#[derive(Clone)]
/// Represents a scene object and its position within the scene tree hierarchy.
///
/// This struct combines a scene object wrapper with identifiers that determine its
/// location in the tree structure.
/// Note: `object_id` is never the root (ID = 0).
pub struct TreeSceneObject {
    pub(crate) scene_object: SceneObjectWrapper,
    pub(crate) object_id: ObjectId,
    parent_id: ObjectId,
}

impl TreeSceneObject {
    pub fn gg_type_id(&self) -> TypeId {
        self.scene_object.type_id
    }
    pub fn object_id(&self) -> ObjectId {
        self.object_id
    }
    /// NOTE: Borrows!
    pub fn transform(&self) -> Transform {
        *self.scene_object.transform.borrow()
    }
    /// NOTE: Borrows!
    pub fn transform_mut(&self) -> RefMut<Transform> {
        self.scene_object.transform.borrow_mut()
    }
    /// NOTE: Borrows!
    pub fn emitting_tags(&self) -> Vec<&'static str> {
        self.scene_object.wrapped.borrow().emitting_tags()
    }
    /// NOTE: Borrows!
    pub fn listening_tags(&self) -> Vec<&'static str> {
        self.scene_object.wrapped.borrow().listening_tags()
    }

    /// NOTE: Borrows!
    pub fn nickname_or_type_name(&self) -> String {
        self.scene_object.nickname_or_type_name()
    }
    /// NOTE: Borrows!
    pub fn nickname(&self) -> Option<String> {
        self.scene_object.nickname.borrow().clone()
    }
    /// NOTE: Borrows!
    pub fn set_nickname(&mut self, name: impl Into<String>) {
        self.scene_object.set_nickname(name);
    }
}

impl Debug for TreeSceneObject {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?} ({:?})", self.object_id, self.gg_type_id())
    }
}

pub trait DowncastRef {
    /// Attempts to downcast a reference to a specific scene object type.
    /// Returns None if the type does not match.
    ///
    /// # Examples
    /// ```ignore
    /// use glongge::core::prelude::*;
    /// let triangle = SpinningTriangle::default().into_wrapper();
    /// let triangle_ref = triangle.downcast::<SpinningTriangle>();
    /// assert!(triangle_ref.is_some());
    ///
    /// let wrong_type = triangle.downcast::<Sprite>();
    /// assert!(wrong_type.is_none());
    /// ```
    fn downcast<T: SceneObject>(&self) -> Option<Ref<T>>;

    /// Attempts to downcast a mutable reference to a specific scene object type.
    /// Returns None if the type does not match.
    ///
    /// # Examples
    /// ```ignore
    /// use glongge::core::prelude::*;
    /// let triangle = SpinningTriangle::default().into_wrapper();
    /// let mut triangle_mut = triangle.downcast_mut::<SpinningTriangle>();
    /// assert!(triangle_mut.is_some());
    ///
    /// let wrong_type = triangle.downcast_mut::<Sprite>();
    /// assert!(wrong_type.is_none());
    /// ```
    fn downcast_mut<T: SceneObject>(&self) -> Option<RefMut<T>>;
}

impl DowncastRef for SceneObjectWrapper {
    fn downcast<T: SceneObject>(&self) -> Option<Ref<T>> {
        if self.type_id != TypeId::of::<T>() {
            return None;
        }
        Ref::filter_map(self.wrapped.borrow(), |obj| {
            obj.as_any().downcast_ref::<T>()
        })
        .ok()
    }

    fn downcast_mut<T: SceneObject>(&self) -> Option<RefMut<T>> {
        if self.type_id != TypeId::of::<T>() {
            return None;
        }
        RefMut::filter_map(self.wrapped.borrow_mut(), |obj| {
            obj.as_any_mut().downcast_mut::<T>()
        })
        .ok()
    }
}

impl DowncastRef for TreeSceneObject {
    fn downcast<T: SceneObject>(&self) -> Option<Ref<T>> {
        self.scene_object.downcast()
    }
    fn downcast_mut<T: SceneObject>(&self) -> Option<RefMut<T>> {
        self.scene_object.downcast_mut()
    }
}
