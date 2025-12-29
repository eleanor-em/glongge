# Plan: Fix Audio Output Stream Detection

## Problem
The current audio implementation in `src/resource/sound.rs` creates a single `OutputStream` at startup and leaks it. If the user changes their default audio output device during runtime, audio continues playing on the old device.

## Root Cause
- `OutputStreamBuilder::open_default_stream()` binds to a specific device at creation time
- The stream is leaked via `Box::leak()` and never recreated
- `OnceLock` prevents creating a new stream
- Neither rodio nor cpal provide device change notifications

## Solution: Polling-Based Device Change Detection

### Step 1: Add cpal dependency for device enumeration
Rodio uses cpal internally, but we need direct access to enumerate devices.

```toml
cpal = "0.16"  # Add to Cargo.toml
```

### Step 2: Refactor SoundHandler to support stream recreation

Current structure:
```rust
static DID_CREATE_OUTPUT_STREAM: OnceLock<bool> = OnceLock::new();

pub struct SoundHandler {
    mixer: Arc<Mixer>,
    inner: Arc<Mutex<SoundHandlerInner>>,
    join_handles: Arc<Mutex<Vec<JoinHandle<()>>>>,
}
```

New structure:
```rust
pub struct SoundHandler {
    stream_state: Arc<Mutex<StreamState>>,
    inner: Arc<Mutex<SoundHandlerInner>>,
    join_handles: Arc<Mutex<Vec<JoinHandle<()>>>>,
}

struct StreamState {
    mixer: Mixer,
    current_device_name: Option<String>,
    // Store the OutputStream here instead of leaking it
    _stream: Option<OutputStream>,
}
```

### Step 3: Implement device name retrieval

```rust
fn get_default_device_name() -> Option<String> {
    use cpal::traits::{DeviceTrait, HostTrait};
    cpal::default_host()
        .default_output_device()
        .and_then(|d| d.name().ok())
}
```

### Step 4: Add check_device_change() method

```rust
impl SoundHandler {
    /// Check if the default audio device has changed and recreate stream if needed.
    /// Call this periodically from the game loop (e.g., once per second).
    pub fn check_device_change(&self) -> Result<bool> {
        let current_name = get_default_device_name();
        let mut state = self.stream_state.lock().unwrap();

        if current_name != state.current_device_name {
            info!("Audio device changed from {:?} to {:?}",
                  state.current_device_name, current_name);

            // Create new stream
            let stream = OutputStreamBuilder::open_default_stream()?;
            let mixer = stream.mixer().clone();

            // Update state
            state._stream = Some(stream);
            state.mixer = mixer;
            state.current_device_name = current_name;

            return Ok(true);
        }
        Ok(false)
    }
}
```

### Step 5: Modify Sound to handle mixer changes

Option A: Lazy sink creation (recommended)
- Don't store Sink in SoundInner
- Create a new Sink each time play() is called
- Sinks automatically use the current mixer

Option B: Sink reconnection
- Store weak reference to SoundHandler in Sound
- On play(), check if mixer has changed and recreate sink

### Step 6: Integrate with game loop

In the main update loop, call:
```rust
if frame_count % 60 == 0 {  // Check once per second at 60fps
    sound_handler.check_device_change()?;
}
```

## Implementation Order

1. [ ] Add cpal dependency to Cargo.toml
2. [ ] Create `get_default_device_name()` helper function
3. [ ] Refactor `SoundHandler` to use `StreamState` struct
4. [ ] Remove `OnceLock` restriction, store stream in struct
5. [ ] Implement `check_device_change()` method
6. [ ] Modify `Sound` to create sinks lazily (on each play call)
7. [ ] Add periodic device check call to game loop
8. [ ] Test device switching scenario

## Files to Modify

- `Cargo.toml` - add cpal dependency
- `src/resource/sound.rs` - main implementation changes
- `src/core/tulivuori/mod.rs` - add periodic check call (if needed)

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Brief audio gap during stream recreation | Acceptable for device switch scenario |
| Polling overhead | Once per second is negligible |
| macOS OutputStream not Send | Keep stream in Mutex, don't leak |
| Existing sounds stop on device change | Lazy sink creation handles this |
