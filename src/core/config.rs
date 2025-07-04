// TODO: some way to override these from user code.
pub const MAX_TEXTURE_COUNT: usize = 1023;
pub const MAX_MATERIAL_COUNT: usize = 16 * 1024;
pub const INITIAL_VERTEX_BUFFER_SIZE: usize = 100_000;
pub const MAX_FIXED_UPDATES: u128 = 3;
pub const FIXED_UPDATE_INTERVAL_US: u128 = 20_000;
pub const FIXED_UPDATE_TIMEOUT: u128 = 10 * FIXED_UPDATE_INTERVAL_US;
pub const USE_DEBUG_GUI: bool = true;
pub const EPSILON: f32 = 1e-5;
pub const DISABLE_SOUND: bool = true;

pub const USE_VSYNC: bool = true;
// TOOD: sync render both ways? Might improve font flickering without vsync.
pub const SYNC_UPDATE_TO_RENDER: bool = true;
