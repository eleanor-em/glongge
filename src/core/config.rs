// TODO: some way to override these from user code.
pub const MAX_TEXTURE_COUNT: usize = 1023;
pub const MAX_MATERIAL_COUNT: usize = 16 * 1024;
pub const INITIAL_VERTEX_BUFFER_SIZE: usize = 100_000;
pub const MAX_FIXED_UPDATES: u128 = 3;
pub const FIXED_UPDATE_INTERVAL_US: u128 = 20_000;
pub const FIXED_UPDATE_WARN_DELAY_US: u128 = FIXED_UPDATE_INTERVAL_US / 2;
pub const FIXED_UPDATE_TIMEOUT_US: u128 = 10 * FIXED_UPDATE_INTERVAL_US;
// Do not try to update more than this often.
pub const UPDATE_THROTTLE_NS: u128 = 1_000;
pub const USE_DEBUG_GUI: bool = true;
pub const EPSILON: f32 = 1e-5;
pub const DISABLE_SOUND: bool = false;

pub const USE_VSYNC: bool = true;
// Breaks a lot of stuff if set to false, these are arguably bugs but probably not worth fixing
// (right now).
pub const SYNC_UPDATE_TO_RENDER: bool = true;

pub const FONT_SAMPLE_RATIO: f32 = 4.0;

pub const SLOW_LOAD_DEADLINE: f32 = 4.0;
