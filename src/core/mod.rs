use std::{any::{TypeId}, cell::{Ref, RefCell, RefMut}, fmt::Debug, rc::Rc, sync::atomic::{AtomicUsize, Ordering}};
use scene::SceneObject;
use crate::{
    core::{
        prelude::*,
        util::{
            collision::Collider,
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

pub trait ObjectTypeEnum: Clone + Copy + Debug + Eq + PartialEq + Sized + 'static {
    fn as_default(self) -> AnySceneObject<Self>;
    fn as_typeid(self) -> TypeId;
    fn all_values() -> Vec<Self>;
    fn preload_all(resource_handler: &mut ResourceHandler) -> Result<()>;
    fn checked_downcast<T: SceneObject<Self> + 'static>(obj: &dyn SceneObject<Self>) -> &T;
    fn checked_downcast_mut<T: SceneObject<Self> + 'static>(obj: &mut dyn SceneObject<Self>) -> &mut T;
}

static NEXT_OBJECT_ID: AtomicUsize = AtomicUsize::new(0);
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct ObjectId(usize);

impl ObjectId {
    fn next() -> Self { ObjectId(NEXT_OBJECT_ID.fetch_add(1, Ordering::Relaxed)) }
}

#[derive(Clone)]
pub struct SceneObjectWithId<ObjectType> {
    object_id: ObjectId,
    inner: Rc<RefCell<AnySceneObject<ObjectType>>>,
}

impl<ObjectType: ObjectTypeEnum> SceneObjectWithId<ObjectType> {
    fn new(object_id: ObjectId, obj: Rc<RefCell<AnySceneObject<ObjectType>>>) -> Self {
        Self { object_id, inner: obj }
    }

    pub fn get_type(&self) -> ObjectType { self.inner.borrow().get_type() }
    pub fn downcast<T: SceneObject<ObjectType> + 'static>(&self) -> Option<Ref<T>> {
        Ref::filter_map(self.inner.borrow(), |obj| {
            obj.as_any().downcast_ref::<T>()
        }).ok()
    }
    pub fn downcast_mut<T: SceneObject<ObjectType> + 'static>(&mut self) -> Option<RefMut<T>> {
        RefMut::filter_map(self.inner.borrow_mut(), |obj| {
            obj.as_any_mut().downcast_mut::<T>()
        }).ok()
    }
    // TODO: the below may turn out to still be useful.
    #[allow(dead_code)]
    fn checked_downcast<T: SceneObject<ObjectType> + 'static>(&self) -> Ref<T> {
        Ref::map(self.inner.borrow(), |obj| ObjectType::checked_downcast::<T>(obj.as_ref()))
    }
    #[allow(dead_code)]
    fn checked_downcast_mut<T: SceneObject<ObjectType> + 'static>(&self) -> RefMut<T> {
        RefMut::map(self.inner.borrow_mut(), |obj| ObjectType::checked_downcast_mut::<T>(obj.as_mut()))
    }

    pub fn transform(&self) -> Transform { self.inner.borrow().transform() }
    pub fn collider(&self) -> Box<dyn Collider> { self.inner.borrow().collider() }
    pub fn emitting_tags(&self) -> Vec<&'static str> { self.inner.borrow().emitting_tags() }
    pub fn listening_tags(&self) -> Vec<&'static str> { self.inner.borrow().listening_tags() }
}

type PendingObjectVec<ObjectType> = Vec<Rc<RefCell<AnySceneObject<ObjectType>>>>;

pub type AnySceneObject<ObjectType> = Box<dyn SceneObject<ObjectType>>;
