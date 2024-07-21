# glongge
Rust game engine. Yeehaw.

## Requirements
- Vulkan (tested: 1.3.279, 1.3.283)
- Rust (nightly, see "Required nightly features" for more details)

Should just work with `cargo run`. Tested on macOS (thoroughly), Windows, and Linux (less thoroughly).

When run as a binary, it plays a demo game (see `src/mario`).

## Required nightly features
- `#![feature(iterator_try_collect)]`
- `#![feature(arbitrary_self_types)]`
- `#![feature(mapped_lock_guards)]`
- `#![feature(duration_millis_float)]`
- `#![feature(stmt_expr_attributes)]`
- `#![feature(const_fn_floating_point_arithmetic)]`
