#[allow(unused_imports)]
pub use itertools::Itertools;
#[allow(unused_imports)]
pub use num_traits;

#[allow(unused_imports)]
pub use anyhow::{anyhow, bail, Context, Result};
#[allow(unused_imports)]
pub use tracing::{info, warn, error};

#[allow(unused_imports)]
pub use crate::{
    core::{
        config::*,
        coroutine::{CoroutineId, CoroutineResponse, CoroutineState},
        render::{RenderInfo, RenderItem},
        scene::{RenderableObject, SceneObject},
        input::KeyCode,
        update::{collision::CollisionResponse, ObjectContext,
            UpdateContext, FixedUpdateContext
        },
        AnySceneObject,
        DowncastRef,
        SceneObjectWithId,
    },
    resource::{Loader, ResourceHandler},
    util::{
        assert::*,
        collision::{Collider, CollisionShape, GenericCollider,
        },
        colour::Colour,
        linalg,
        linalg::{AxisAlignedExtent, Mat3x3, Rect, Transform, Vec2, Vec2i},
    },
};
