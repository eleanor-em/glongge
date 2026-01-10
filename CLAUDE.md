# Glongge

Rust 2D game engine with Vulkan rendering. Includes a Super Mario-like demo game.

## Documentation
In `claude/` are various documents Claude has created when solving particular problems in development. In `claude/logs/` you will find logs of particular Claude Code sessions in case these are useful.

## Testing
You CANNOT run the game – you have no way to mock keyboard inputs etc. That means that you must ask the user to test the game when you make changes and wait until the user reports the results back to you.

Run ALL of these commands after completing a set of changes. Note you have to do the three different clippy runs separately.
```
cargo +nightly fmt   # Format code (run after editing)
cargo +nightly clippy -- -W clippy::pedantic -A clippy::must_use_candidate -A clippy::missing_errors_doc -A clippy::module_name_repetitions -A clippy::missing_panics_doc -A clippy::cast_possible_truncation -A clippy::cast_precision_loss -A clippy::cast_sign_loss -A clippy::similar_names   # Run lints
cargo +nightly clippy --lib -- -W clippy::pedantic -A clippy::must_use_candidate -A clippy::missing_errors_doc -A clippy::module_name_repetitions -A clippy::missing_panics_doc -A clippy::cast_possible_truncation -A clippy::cast_precision_loss -A clippy::cast_sign_loss -A clippy::similar_names   # Run lints
cargo +nightly clippy --tests -- -W clippy::pedantic -A clippy::must_use_candidate -A clippy::missing_errors_doc -A clippy::module_name_repetitions -A clippy::missing_panics_doc -A clippy::cast_possible_truncation -A clippy::cast_precision_loss -A clippy::cast_sign_loss -A clippy::similar_names  # Run lints
cargo test --lib   # Run unit tests
```

After modifying any of the following files, use `cargo +nightly llvm-cov` to ensure test coverage remains at 100% for regions, functions, AND lines:
- util/collision.rs
- util/colour.rs
- util/linalg.rs
- util/spline.rs
- resource/rich_text.rs

YOU MUST REACH 100% COVERAGE FOR THE ABOVE FILES. THERE ARE NO EXCEPTIONS.

Requirements: Rust stable, Vulkan SDK (on Linux: `libxkbcommon-dev`, `libwayland-dev`)

## Architecture

### Core Modules

- **`src/core/`** - Engine core: scene management, rendering, input, update loop
  - `scene.rs` - Scene trait and lifecycle management
  - `render.rs` - Vulkan rendering orchestration
  - `tulivuori/` - Vulkan abstraction layer (device, swapchain, buffers, shaders)
  - `update/` - Fixed-rate update loop (50 FPS), collision detection
  - `input.rs` - Keyboard/window event handling
  - `builtin.rs` - Built-in objects: `StaticSprite`, `Container`
  - `coroutine.rs` - Async coroutine system

- **`src/util/`** - Utilities and math
  - `linalg.rs` - Linear algebra: `Vec2`, `Mat3x3`, `Transform`, `AxisAlignedExtent`
  - `collision.rs` - Collision system: `Collider`, `BoxCollider`, `Polygon`
  - `tileset.rs` - Tile-based level building

- **`src/resource/`** - Asset management: textures, sprites, sounds, fonts

- **`src/gui/`** - Debug GUI using egui

- **`src/examples/mario/`** - Mario demo game implementation

### Key Patterns

- **Scene objects** implement the `SceneObject` trait
- Use `#[partially_derive_scene_object]` macro to auto-derive required methods
- Objects use `Rc<RefCell<T>>` for shared mutable state
- Collision uses tag-based filtering with response callbacks
- Fixed 20ms update interval (`FIXED_UPDATE_INTERVAL_US` in `config.rs`)

### Procedural Macros (`glongge-derive/`)

- `#[register_scene_object]` - Marker attribute
- `#[partially_derive_scene_object]` - Derives `as_any()`, `as_any_mut()`, `gg_type_name()`

## Configuration

Key constants in `src/core/config.rs`:
- `FIXED_UPDATE_INTERVAL_US`: 20,000 (50 FPS physics)
- `MAX_TEXTURE_COUNT`: 1,023
- `USE_DEBUG_GUI`: true (egui overlay)

## Platform Notes

Development is on macOS/MoltenVK, but the target is all modern desktop and laptop platforms: Windows, Linux, and macOS with NVIDIA, AMD, Intel, and integrated GPUs. Code should not rely on MoltenVK-specific behavior or assume coherent memory, discrete GPU, etc.

- macOS: Uses MoltenVK (Vulkan over Metal)
- Config in `.cargo/config.toml` sets `MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS`
- 
