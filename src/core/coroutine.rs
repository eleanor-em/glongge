use crate::core::{ObjectTypeEnum, TreeSceneObject, update::UpdateContext};
use std::{
    sync::atomic::{AtomicUsize, Ordering},
    time::{Duration, Instant},
};

static NEXT_COROUTINE_ID: AtomicUsize = AtomicUsize::new(0);
/// A unique identifier for a coroutine instance.
/// Can be stored for use with
/// [`cancel_coroutine()`](crate::core::update::SceneContext::cancel_coroutine).
#[derive(Copy, Clone, Debug, Default, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct CoroutineId(usize);

impl CoroutineId {
    pub(crate) fn next() -> Self {
        CoroutineId(NEXT_COROUTINE_ID.fetch_add(1, Ordering::Relaxed))
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
/// The current state of a coroutine's execution.
///
/// - `Starting`: Initial state when the coroutine begins execution
/// - `Yielding`: The coroutine has temporarily suspended execution
/// - `Waiting`: The coroutine is waiting for a specific duration before resuming
pub enum CoroutineState {
    /// Initial state when the coroutine begins execution
    Starting,
    /// The coroutine has temporarily suspended execution
    Yielding,
    /// The coroutine is waiting for a specific duration before resuming
    Waiting,
}
/// Represents possible responses from a coroutine during its execution.
///
/// - `Yield`: Suspend execution and return control to the engine for this update frame.
/// - `Wait(Duration)`: Pause execution for a specified duration before resuming. Note: the coroutine will not
///   resume exactly at the end of the duration, but rather at the start of the first update frame
///   that is after the end of the duration.
/// - `Complete`: The coroutine has finished its execution.
pub enum CoroutineResponse {
    /// Suspend execution and return control to the engine for this update frame.
    Yield,
    /// Pause execution for a specified duration before resuming. Note: the coroutine will not
    /// resume exactly at the end of the duration, but rather at the start of the first update frame
    /// that is after the end of the duration.
    Wait(Duration),
    /// The coroutine has finished its execution.
    Complete,
}

pub(crate) type CoroutineFunc<ObjectType> = dyn FnMut(
    &TreeSceneObject<ObjectType>,
    &mut UpdateContext<ObjectType>,
    CoroutineState,
) -> CoroutineResponse;

pub(crate) struct Coroutine<ObjectType: ObjectTypeEnum> {
    func: Box<CoroutineFunc<ObjectType>>,
    wait_since: Instant,
    wait_duration: Duration,
    last_action: CoroutineState,
}

impl<ObjectType: ObjectTypeEnum> Coroutine<ObjectType> {
    pub(crate) fn new<F>(func: F) -> Self
    where
        F: FnMut(
                &TreeSceneObject<ObjectType>,
                &mut UpdateContext<ObjectType>,
                CoroutineState,
            ) -> CoroutineResponse
            + 'static,
    {
        Self {
            func: Box::new(func),
            wait_since: Instant::now(),
            wait_duration: Duration::from_secs(0),
            last_action: CoroutineState::Starting,
        }
    }

    pub(crate) fn resume(
        mut self,
        this: &TreeSceneObject<ObjectType>,
        ctx: &mut UpdateContext<ObjectType>,
    ) -> Option<Self> {
        if self.wait_since.elapsed() < self.wait_duration {
            return Some(self);
        }
        let result = (self.func)(this, ctx, self.last_action);
        match result {
            CoroutineResponse::Yield => {
                self.last_action = CoroutineState::Yielding;
                Some(self)
            }
            CoroutineResponse::Wait(time) => {
                self.wait_since = Instant::now();
                self.wait_duration = time;
                self.last_action = CoroutineState::Waiting;
                Some(self)
            }
            CoroutineResponse::Complete => None,
        }
    }
}
