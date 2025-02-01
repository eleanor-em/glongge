#[allow(unused_imports)]
pub use itertools::Itertools;
#[allow(unused_imports)]
pub use num_traits;

#[allow(unused_imports)]
pub use anyhow::{anyhow, bail, Context, Result};
#[allow(unused_imports)]
pub use tracing::{error, info, warn};

#[allow(unused_imports)]
pub use crate::{
    core::{
        config::*,
        coroutine::{CoroutineId, CoroutineResponse, CoroutineState},
        input::KeyCode,
        render::{RenderInfo, RenderItem},
        scene::{RenderableObject, SceneObject},
        update::{collision::CollisionResponse, FixedUpdateContext, ObjectContext, UpdateContext},
        AnySceneObject, DowncastRef, SceneObjectWithId,
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
