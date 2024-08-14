use std::{
    any::TypeId,
    cell::{Ref, RefCell, RefMut},
    fmt::{
        Debug,
        Formatter
    },
    ops::Deref,
    rc::Rc,
    sync::atomic::{AtomicUsize, Ordering},
};
use scene::SceneObject;
use crate::{
    core::prelude::*,
    resource::ResourceHandler,
    util::linalg::Transform
};

pub mod input;
pub mod vk;
pub mod prelude;
pub mod config;
pub mod coroutine;
pub mod render;
pub mod scene;
pub mod update;
pub mod builtin;

pub trait ObjectTypeEnum: Clone + Copy + Debug + Eq + PartialEq + Sized + 'static + Send {
    fn as_default(self) -> AnySceneObject<Self>;
    fn as_typeid(self) -> TypeId;
    fn all_values() -> Vec<Self>;
    fn gg_sprite() -> Self;
    fn gg_collider() -> Self;
    fn gg_canvas() -> Self;
    fn gg_canvas_item() -> Self;
    fn gg_container() -> Self;
    fn gg_static_sprite() -> Self;
    fn gg_colliding_sprite() -> Self;
    fn gg_tileset() -> Self;
    fn gg_interactive_spline() -> Self;

    fn preload_all(resource_handler: &mut ResourceHandler) -> Result<()> {
        for value in Self::all_values() {
            value.as_default().borrow_mut().on_preload(resource_handler)?;
        }
        resource_handler.wait_all()?;
        Ok(())
    }
    fn checked_downcast<T: SceneObject<Self> + 'static>(obj: &dyn SceneObject<Self>) -> &T {
        let actual = obj.get_type().as_typeid();
        let expected = obj.as_any().type_id();
        if actual != expected {
            for value in Self::all_values() {
                check_ne!(value.as_typeid(), actual,
                    format!("attempt to downcast {:?} -> {:?}", obj.get_type(), value));
            }
            panic!("attempt to downcast {:?}: type missing? {:?}", obj.get_type(), Self::all_values());
        }
        obj.as_any().downcast_ref::<T>().unwrap()
    }
    fn checked_downcast_mut<T: SceneObject<Self> + 'static>(obj: &mut dyn SceneObject<Self>) -> &mut T {
        let actual = obj.get_type().as_typeid();
        let expected = obj.as_any().type_id();
        if actual != expected {
            for value in Self::all_values() {
                check_ne!(value.as_typeid(), actual,
                    format!("attempt to downcast {:?} -> {:?}", obj.get_type(), value));
            }
            panic!("attempt to downcast {:?}: type missing? {:?}", obj.get_type(), Self::all_values());
        }
        obj.as_any_mut().downcast_mut::<T>().unwrap()
    }
}

// ObjectId(0) represents the root object.
static NEXT_OBJECT_ID: AtomicUsize = AtomicUsize::new(1);
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct ObjectId(pub(crate) usize);

impl ObjectId {
    fn next() -> Self { ObjectId(NEXT_OBJECT_ID.fetch_add(1, Ordering::Relaxed)) }
    pub(crate) fn is_root(self) -> bool { self.0 == 0 }
    pub(crate) fn root() -> Self { ObjectId(0) }
}


#[derive(Clone)]
pub struct AnySceneObject<ObjectType> {
    transform: Rc<RefCell<Transform>>,
    inner: Rc<RefCell<dyn SceneObject<ObjectType>>>,
}

impl<ObjectType: ObjectTypeEnum> AnySceneObject<ObjectType> {
    pub fn new<O: SceneObject<ObjectType>>(inner: O) -> Self {
        Self {
            transform: Rc::new(RefCell::new(Transform::default())),
            inner: Rc::new(RefCell::new(inner))
        }
    }

    pub(crate) fn from_rc<O: SceneObject<ObjectType>>(rc: Rc<RefCell<O>>) -> Self {
        Self {
            transform: Rc::new(RefCell::new(Transform::default())),
            inner: rc
        }
    }

    pub(crate) fn name(&self) -> String { self.inner.borrow().name() }
    pub(crate) fn transform(&self) -> Transform { *self.transform.borrow() }
    pub(crate) fn transform_mut(&mut self) -> RefMut<Transform> { self.transform.borrow_mut() }
}

impl<ObjectType: ObjectTypeEnum> Deref for AnySceneObject<ObjectType> {
    type Target = Rc<RefCell<dyn SceneObject<ObjectType>>>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

pub struct SceneObjectWithId<ObjectType> {
    pub(crate) object_id: ObjectId,
    pub(crate) inner: AnySceneObject<ObjectType>,
}

impl<ObjectType: ObjectTypeEnum> SceneObjectWithId<ObjectType> {
    fn new(object_id: ObjectId, obj: AnySceneObject<ObjectType>) -> Self {
        Self { object_id, inner: obj }
    }

    // Do not allow public cloning.
    fn clone(&self) -> SceneObjectWithId<ObjectType> {
        Self::new(self.object_id, self.inner.clone())
    }

    pub fn get_type(&self) -> ObjectType { self.inner.borrow().get_type() }
    pub fn name(&self) -> String { self.inner.borrow().name() }

    pub fn transform(&self) -> Transform { self.inner.transform() }
    pub fn emitting_tags(&self) -> Vec<&'static str> { self.inner.borrow().emitting_tags() }
    pub fn listening_tags(&self) -> Vec<&'static str> { self.inner.borrow().listening_tags() }
}

impl<ObjectType: ObjectTypeEnum> Debug for SceneObjectWithId<ObjectType> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?} ({:?})", self.object_id, self.get_type())
    }
}

pub trait DowncastRef<ObjectType: ObjectTypeEnum> {
    fn downcast<T: SceneObject<ObjectType>>(&self) -> Option<Ref<T>>;
    fn downcast_mut<T: SceneObject<ObjectType>>(&self) -> Option<RefMut<T>>;
    fn checked_downcast<T: SceneObject<ObjectType>>(&self) -> Ref<T>;
    fn checked_downcast_mut<T: SceneObject<ObjectType>>(&self) -> RefMut<T>;
}

impl<ObjectType: ObjectTypeEnum> DowncastRef<ObjectType> for AnySceneObject<ObjectType> {
    fn downcast<T: SceneObject<ObjectType>>(&self) -> Option<Ref<T>> {
        Ref::filter_map(self.borrow(), |obj| {
            obj.as_any().downcast_ref::<T>()
        }).ok()
    }

    fn downcast_mut<T: SceneObject<ObjectType>>(&self) -> Option<RefMut<T>> {
        RefMut::filter_map(self.borrow_mut(), |obj| {
            obj.as_any_mut().downcast_mut::<T>()
        }).ok()
    }

    fn checked_downcast<T: SceneObject<ObjectType>>(&self) -> Ref<T> {
        Ref::map(self.borrow(), |obj| ObjectType::checked_downcast::<T>(obj))
    }

    fn checked_downcast_mut<T: SceneObject<ObjectType>>(&self) -> RefMut<T> {
        RefMut::map(self.borrow_mut(), |obj| ObjectType::checked_downcast_mut::<T>(obj))
    }
}

impl<ObjectType: ObjectTypeEnum> DowncastRef<ObjectType> for SceneObjectWithId<ObjectType> {
    fn downcast<T: SceneObject<ObjectType>>(&self) -> Option<Ref<T>> {
        self.inner.downcast()
    }
    fn downcast_mut<T: SceneObject<ObjectType>>(&self) -> Option<RefMut<T>> {
        self.inner.downcast_mut()
    }
    fn checked_downcast<T: SceneObject<ObjectType>>(&self) -> Ref<T> {
        self.inner.checked_downcast()
    }
    fn checked_downcast_mut<T: SceneObject<ObjectType>>(&self) -> RefMut<T> {
        self.inner.checked_downcast_mut()
    }
}
