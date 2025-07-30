use crate::core::prelude::*;
use scene::SceneObject;
use std::marker::PhantomData;
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
#[derive(Copy, Clone, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct ObjectId(usize);

impl ObjectId {
    fn next() -> Self {
        ObjectId(NEXT_OBJECT_ID.fetch_add(1, Ordering::SeqCst))
    }
    pub(crate) fn is_root(self) -> bool {
        self.0 == 0
    }
    pub(crate) fn root() -> Self {
        ObjectId(0)
    }

    pub(crate) fn value_for_gui(self) -> usize {
        self.0
    }
    pub fn value_eq_for_debugging(self, rhs: usize) -> bool {
        self.0 == rhs
    }
}

impl Debug for ObjectId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[ObjectId {}]", self.0)
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
    pub(crate) fn gg_is<T: SceneObject>(&self) -> bool {
        self.type_id == TypeId::of::<T>()
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
    pub fn gg_is<T: SceneObject>(&self) -> bool {
        self.scene_object.type_id == TypeId::of::<T>()
    }
    pub(crate) fn object_id(&self) -> ObjectId {
        self.object_id
    }
    /// NOTE: Borrows!
    pub fn transform(&self) -> Transform {
        *self.scene_object.transform.borrow()
    }
    /// NOTE: Borrows!
    pub fn transform_mut(&self) -> RefMut<'_, Transform> {
        self.scene_object.transform.borrow_mut()
    }
    /// NOTE: Borrows!
    pub fn emitting_tags(&self) -> Vec<&'static str> {
        self.inner().emitting_tags()
    }
    /// NOTE: Borrows!
    pub fn listening_tags(&self) -> Vec<&'static str> {
        self.inner().listening_tags()
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

    pub(crate) fn inner(&self) -> Ref<'_, dyn SceneObject> {
        self.scene_object.wrapped.borrow()
    }
    pub(crate) fn inner_mut(&self) -> RefMut<'_, dyn SceneObject> {
        self.scene_object.wrapped.borrow_mut()
    }
}

impl PartialEq for TreeSceneObject {
    fn eq(&self, other: &Self) -> bool {
        self.object_id == other.object_id
    }
}

impl Eq for TreeSceneObject {}

impl Debug for TreeSceneObject {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:?} {:?}",
            self.scene_object.nickname_or_type_name(),
            self.object_id,
        )
    }
}

/// A type-safe wrapper around [`TreeSceneObject`] that ensures the wrapped object is of a specific
/// type.
///
/// `TreeObjectOfType<T>` provides a way to safely store and access scene objects of a known type,
/// allowing ergonomic access while preventing type mismatches at runtime. It acts as a
/// strongly-typed reference to a scene object within the scene tree hierarchy.
///
/// # Type Parameters
///
/// * `T` - The specific [`SceneObject`] type that this wrapper contains.
///
/// # Examples
///
/// ```ignore
/// // Get a typed reference to a Player object
/// let player: TreeObjectOfType<Player> = ctx.object().first_other::<Player>().try_into().unwrap();
///
/// // Now you can safely call methods on the player without downcasting
/// let player_ground_position = player.inner().get_shadow().centre;
/// ```
///
/// # Notes
///
/// Most accessor methods like `transform()`, `transform_mut()`, `emitting_tags()`, etc. borrow
/// the underlying object. Be careful with these methods to avoid borrow checker issues.
pub struct TreeObjectOfType<T: SceneObject> {
    inner: Option<TreeSceneObject>,
    phantom_data: PhantomData<T>,
}

impl<T: SceneObject> TreeObjectOfType<T> {
    pub fn of(obj: &TreeSceneObject) -> Option<Self> {
        if obj.gg_is::<T>() {
            Some(Self {
                inner: Some(obj.clone()),
                phantom_data: PhantomData,
            })
        } else {
            None
        }
    }
    pub fn and_of(obj: Option<&TreeSceneObject>) -> Option<Self> {
        obj.and_then(Self::of)
    }

    pub fn borrow(&self) -> Ref<'_, T> {
        self.inner.as_ref().unwrap().downcast::<T>().unwrap()
    }
    pub fn borrow_mut(&self) -> RefMut<'_, T> {
        self.inner.as_ref().unwrap().downcast_mut::<T>().unwrap()
    }
    pub fn inner(&self) -> &TreeSceneObject {
        self.inner.as_ref().unwrap()
    }

    pub(crate) fn object_id(&self) -> ObjectId {
        self.inner.as_ref().unwrap().object_id()
    }
    /// NOTE: Borrows!
    pub fn transform(&self) -> Transform {
        self.inner.as_ref().unwrap().transform()
    }
    /// NOTE: Borrows!
    pub fn transform_mut(&self) -> RefMut<'_, Transform> {
        self.inner.as_ref().unwrap().transform_mut()
    }
    /// NOTE: Borrows!
    pub fn emitting_tags(&self) -> Vec<&'static str> {
        self.inner.as_ref().unwrap().emitting_tags()
    }
    /// NOTE: Borrows!
    pub fn listening_tags(&self) -> Vec<&'static str> {
        self.inner.as_ref().unwrap().listening_tags()
    }

    /// NOTE: Borrows!
    pub fn nickname_or_type_name(&self) -> String {
        self.inner.as_ref().unwrap().nickname_or_type_name()
    }
    /// NOTE: Borrows!
    pub fn nickname(&self) -> Option<String> {
        self.inner.as_ref().unwrap().nickname()
    }
    /// NOTE: Borrows!
    pub fn set_nickname(&mut self, name: impl Into<String>) {
        self.inner.as_mut().unwrap().set_nickname(name);
    }

    pub fn remove(&self, ctx: &mut UpdateContext) {
        ctx.object_mut().remove(self.inner.as_ref().unwrap());
    }
}

impl<T: SceneObject> Clone for TreeObjectOfType<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            phantom_data: PhantomData,
        }
    }
}

impl<T: SceneObject> Default for TreeObjectOfType<T> {
    fn default() -> Self {
        Self {
            inner: None,
            phantom_data: PhantomData,
        }
    }
}

impl<T: SceneObject> TryFrom<TreeSceneObject> for TreeObjectOfType<T> {
    type Error = anyhow::Error;

    fn try_from(value: TreeSceneObject) -> std::result::Result<Self, Self::Error> {
        Self::of(&value).ok_or_else(|| anyhow!("type mismatch"))
    }
}
impl<T: SceneObject> TryFrom<&TreeSceneObject> for TreeObjectOfType<T> {
    type Error = anyhow::Error;

    fn try_from(value: &TreeSceneObject) -> std::result::Result<Self, Self::Error> {
        Self::of(value).ok_or_else(|| anyhow!("type mismatch"))
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
    fn downcast<T: SceneObject>(&self) -> Option<Ref<'_, T>>;

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
    fn downcast_mut<T: SceneObject>(&self) -> Option<RefMut<'_, T>>;
}

impl DowncastRef for SceneObjectWrapper {
    fn downcast<T: SceneObject>(&self) -> Option<Ref<'_, T>> {
        if self.gg_is::<T>() {
            Ref::filter_map(self.wrapped.borrow(), |obj| {
                obj.as_any().downcast_ref::<T>()
            })
            .ok()
        } else {
            None
        }
    }

    fn downcast_mut<T: SceneObject>(&self) -> Option<RefMut<'_, T>> {
        if self.gg_is::<T>() {
            RefMut::filter_map(self.wrapped.borrow_mut(), |obj| {
                obj.as_any_mut().downcast_mut::<T>()
            })
            .ok()
        } else {
            None
        }
    }
}

impl DowncastRef for TreeSceneObject {
    fn downcast<T: SceneObject>(&self) -> Option<Ref<'_, T>> {
        self.scene_object.downcast()
    }
    fn downcast_mut<T: SceneObject>(&self) -> Option<RefMut<'_, T>> {
        self.scene_object.downcast_mut()
    }
}
