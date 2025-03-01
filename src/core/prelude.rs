#[allow(unused_imports)]
pub use itertools::Itertools;
#[allow(unused_imports)]
pub use num_traits;

#[allow(unused_imports)]
pub use anyhow::{Context, Result, anyhow, bail};
#[allow(unused_imports)]
pub use tracing::{error, info, warn};

#[allow(unused_imports)]
pub use crate::{
    core::{
        AnySceneObject, DowncastRef, SceneObjectWithId,
        config::*,
        coroutine::{CoroutineId, CoroutineResponse, CoroutineState},
        input::KeyCode,
        render::{RenderItem, ShaderExec},
        scene::{RenderableObject, SceneObject},
        update::{FixedUpdateContext, ObjectContext, UpdateContext, collision::CollisionResponse},
    },
    resource::{Loader, ResourceHandler},
    util::{
        assert::*,
        collision::{Collider, CollisionShape, GenericCollider},
        colour::Colour,
        linalg,
        linalg::{AxisAlignedExtent, Mat3x3, Rect, Transform, Vec2, Vec2i},
    },
};
