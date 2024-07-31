use std::{
    any::{TypeId},
    cell::{Ref, RefCell, RefMut},
    fmt::{
        Debug,
        Formatter
    },
    rc::Rc,
    sync::atomic::{AtomicUsize, Ordering},
    ops::Deref,
};
use scene::SceneObject;
use crate::{
    core::{
        prelude::*,
        util::{
            linalg::Transform
        }
    },
    resource::ResourceHandler,
};

pub mod input;
pub mod util;
pub mod vk;
pub mod prelude;
pub mod config;
pub mod coroutine;
pub mod render;
pub mod scene;
pub mod update;

pub trait ObjectTypeEnum: Clone + Copy + Debug + Eq + PartialEq + Sized + 'static + Send {

    fn as_default(self) -> AnySceneObject<Self>;
    fn as_typeid(self) -> TypeId;
    fn all_values() -> Vec<Self>;
    fn gg_sprite() -> Self;
    fn gg_collider() -> Self;

    fn preload_all(resource_handler: &mut ResourceHandler) -> Result<()> {
        for value in Self::all_values() {
            value.as_default().on_preload(resource_handler)?;
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
        }
        obj.as_any_mut().downcast_mut::<T>().unwrap()
    }
}

// ObjectId(0) represents the root object.
static NEXT_OBJECT_ID: AtomicUsize = AtomicUsize::new(1);
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct ObjectId(usize);

impl ObjectId {
    fn next() -> Self { ObjectId(NEXT_OBJECT_ID.fetch_add(1, Ordering::Relaxed)) }
}

pub struct BorrowedSceneObjectWithId<'a, ObjectType> {
    _object_id: ObjectId,
    inner: Ref<'a, AnySceneObject<ObjectType>>,
}
impl<'a, ObjectType: ObjectTypeEnum> BorrowedSceneObjectWithId<'a, ObjectType> {
    fn new(object_id: ObjectId, obj: &'a Rc<RefCell<AnySceneObject<ObjectType>>>) -> Self {
        Self { _object_id: object_id, inner: obj.borrow() }
    }
}

impl<'a, ObjectType: ObjectTypeEnum> Deref for BorrowedSceneObjectWithId<'a, ObjectType> {
    type Target = AnySceneObject<ObjectType>;
    fn deref(&self) -> &Self::Target { &self.inner }
}

pub struct SceneObjectWithId<ObjectType> {
    object_id: ObjectId,
    inner: Rc<RefCell<AnySceneObject<ObjectType>>>,
}

impl<ObjectType: ObjectTypeEnum> SceneObjectWithId<ObjectType> {
    fn new(object_id: ObjectId, obj: Rc<RefCell<AnySceneObject<ObjectType>>>) -> Self {
        Self { object_id, inner: obj }
    }

    // Do not allow public cloning.
    fn clone(&self) -> SceneObjectWithId<ObjectType> {
        Self::new(self.object_id, self.inner.clone())
    }

    pub fn get_type(&self) -> ObjectType { self.inner.borrow().get_type() }

    pub fn transform(&self) -> Transform { self.inner.borrow().transform() }
    pub fn collider(&self) -> GenericCollider { self.inner.borrow().collider() }
    pub fn emitting_tags(&self) -> Vec<&'static str> { self.inner.borrow().emitting_tags() }
    pub fn listening_tags(&self) -> Vec<&'static str> { self.inner.borrow().listening_tags() }
}

impl<ObjectType: ObjectTypeEnum> Debug for SceneObjectWithId<ObjectType> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?} ({:?})", self.object_id, self.get_type())
    }
}

pub trait Downcast<ObjectType: ObjectTypeEnum> {
    fn downcast<T: SceneObject<ObjectType>>(&self) -> Option<&T>;
    fn downcast_mut<T: SceneObject<ObjectType>>(&mut self) -> Option<&mut T>;
    fn checked_downcast<T: SceneObject<ObjectType>>(&self) -> &T;
    fn checked_downcast_mut<T: SceneObject<ObjectType>>(&mut self) -> &mut T;
}

pub trait DowncastRef<ObjectType: ObjectTypeEnum> {
    fn downcast<T: SceneObject<ObjectType>>(&self) -> Option<Ref<T>>;
    fn downcast_mut<T: SceneObject<ObjectType>>(&self) -> Option<RefMut<T>>;
    fn checked_downcast<T: SceneObject<ObjectType>>(&self) -> Ref<T>;
    fn checked_downcast_mut<T: SceneObject<ObjectType>>(&self) -> RefMut<T>;
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

impl<ObjectType: ObjectTypeEnum> Downcast<ObjectType> for dyn SceneObject<ObjectType> {
    fn downcast<T: SceneObject<ObjectType>>(&self) -> Option<&T> {
        self.as_any().downcast_ref()
    }
    fn downcast_mut<T: SceneObject<ObjectType>>(&mut self) -> Option<&mut T> {
        self.as_any_mut().downcast_mut()
    }
    fn checked_downcast<T: SceneObject<ObjectType>>(&self) -> &T {
        ObjectType::checked_downcast(self)
    }
    fn checked_downcast_mut<T: SceneObject<ObjectType>>(&mut self) -> &mut T {
        ObjectType::checked_downcast_mut(self)
    }
}

impl<ObjectType: ObjectTypeEnum> DowncastRef<ObjectType> for Rc<RefCell<AnySceneObject<ObjectType>>> {
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
        Ref::map(self.borrow(), |obj| ObjectType::checked_downcast::<T>(obj.as_ref()))
    }

    fn checked_downcast_mut<T: SceneObject<ObjectType>>(&self) -> RefMut<T> {
        RefMut::map(self.borrow_mut(), |obj| ObjectType::checked_downcast_mut::<T>(obj.as_mut()))
    }
}

pub type AnySceneObject<ObjectType> = Box<dyn SceneObject<ObjectType>>;
