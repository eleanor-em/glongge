#[allow(unused_imports)]
pub use itertools::Itertools;
#[allow(unused_imports)]
pub use num_traits;

#[allow(unused_imports)]
pub use anyhow::{Context, Result, anyhow, bail};
pub use egui;
#[allow(unused_imports)]
pub use tracing::{error, info, warn};

pub use bincode;

#[allow(unused_imports)]
pub use crate::{
    core::{
        DowncastRef, IntoSceneObjectWrapper, SceneObjectWrapper, TreeSceneObject,
        config::*,
        coroutine::{CoroutineId, CoroutineResponse, CoroutineState},
        input::{InputHandler, KeyCode},
        render::{RenderItem, ShaderExec},
        scene::{RenderableObject, SceneObject},
        update::{FixedUpdateContext, LoadContext, UpdateContext, collision::CollisionResponse},
    },
    info_every_millis, info_every_seconds,
    resource::{Loader, ResourceHandler},
    util::{
        assert::*,
        collision::{Collider, CollisionShape, GenericCollider},
        colour::Colour,
        linalg,
        linalg::{AxisAlignedExtent, Mat3x3, Rect, Transform, Vec2, Vec2i},
    },
    warn_every_millis, warn_every_seconds,
};
