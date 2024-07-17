#[allow(unused_imports)]
use crate::core::prelude::*;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use crate::gg::{ObjectTypeEnum, SceneObjectWithId, UpdateContext};

static NEXT_COROUTINE_ID: AtomicUsize = AtomicUsize::new(0);
#[derive(Copy, Clone, Debug, Default, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct CoroutineId(usize);

impl CoroutineId {
    pub(crate) fn next() -> Self {
        CoroutineId(NEXT_COROUTINE_ID.fetch_add(1, Ordering::Relaxed))
    }
}

pub enum CoroutineAction {
    Starting,
    Yielding,
    Waiting,
}
pub enum CoroutineResponse {
    Yield,
    Wait(Duration),
    Complete,
}

pub type CoroutineFunc<ObjectType> = dyn FnMut(
    SceneObjectWithId<ObjectType>,
    &mut UpdateContext<ObjectType>,
    CoroutineAction
) -> CoroutineResponse;

pub(crate) struct Coroutine<ObjectType: ObjectTypeEnum> {
    func: Box<CoroutineFunc<ObjectType>>,
    wait_since: Instant,
    wait_duration: Duration,
    last_action: CoroutineAction,
}

impl<ObjectType: ObjectTypeEnum> Coroutine<ObjectType> {
    pub(crate) fn new<F>(func: F) -> Self
    where
        F: FnMut(SceneObjectWithId<ObjectType>, &mut UpdateContext<ObjectType>, CoroutineAction) -> CoroutineResponse + 'static
    {
        Self {
            func: Box::new(func),
            wait_since: Instant::now(),
            wait_duration: Duration::from_secs(0),
            last_action: CoroutineAction::Starting,
        }
    }

    pub(crate) fn resume(mut self, this: SceneObjectWithId<ObjectType>, update_ctx: &mut UpdateContext<ObjectType>) -> Option<Self> {
        if self.wait_since.elapsed() < self.wait_duration {
            return Some(self);
        }
        let result = (self.func)(this, update_ctx, self.last_action);
        match result {
            CoroutineResponse::Yield => {
                self.last_action = CoroutineAction::Yielding;
                Some(self)
            }
            CoroutineResponse::Wait(time) => {
                self.wait_since = Instant::now();
                self.wait_duration = time;
                self.last_action = CoroutineAction::Waiting;
                Some(self)
            }
            CoroutineResponse::Complete => None
        }
    }
}
