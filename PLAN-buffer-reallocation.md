# Buffer Reallocation Plan

## Problem Statement

Currently, when buffers (vertex, material, etc.) run out of space, the application crashes with an error in `GenericBuffer::write()` at `src/core/tulivuori/buffer.rs:154-158`. We need a graceful reallocation strategy that respects frames-in-flight synchronization.

## Current Architecture Summary

- **Frames in flight**: 2 (configured in `src/core/config.rs:21`)
- **Buffer copies**: Each buffer has `FRAMES_IN_FLIGHT` copies to prevent GPU/CPU races
- **Synchronization**: Fences per frame, semaphores for present/submit
- **Key buffers**:
  - `VertexBuffer<SpriteVertex>`: 10 MB initial size (`config.rs:4`)
  - `GenericDeviceBuffer<RawMaterial>`: 16,384 materials (`config.rs:3`)
  - GUI vertex/index buffers: 100 KB each

## Synchronization Challenge

When reallocating a buffer, we cannot free the old buffer while the GPU might still be reading from it. With `FRAMES_IN_FLIGHT = 2`:
- Frame N: GPU may be reading from buffer copy 0
- Frame N+1: GPU may be reading from buffer copy 1
- We must wait for both frames to complete before freeing either copy

## Proposed Solution: Deferred Reallocation with Retire Queue

### Strategy Overview

1. When a buffer needs more space, allocate a new larger buffer immediately
2. Keep the old buffer alive in a "retire queue" for `FRAMES_IN_FLIGHT` frames
3. After `FRAMES_IN_FLIGHT` frames have completed, safely free the old buffer
4. Use exponential growth (2x) to amortize reallocation cost

### Implementation Steps

---

### Step 1: Add Retire Queue Infrastructure

**File**: `src/core/tulivuori/buffer.rs`

Create a retire queue structure to track buffers pending deletion:

```rust
struct RetiredBuffer {
    allocation: TvAllocation,
    buffer: vk::Buffer,
    frames_until_free: u32,
}

struct BufferRetireQueue {
    queue: Vec<RetiredBuffer>,
}

impl BufferRetireQueue {
    fn new() -> Self { ... }

    /// Add a buffer to the retire queue
    fn retire(&mut self, allocation: TvAllocation, buffer: vk::Buffer) {
        self.queue.push(RetiredBuffer {
            allocation,
            buffer,
            frames_until_free: FRAMES_IN_FLIGHT as u32,
        });
    }

    /// Called once per frame. Decrements counters and frees expired buffers.
    fn tick(&mut self, device: &Device, allocator: &mut vk_mem::Allocator) {
        self.queue.retain_mut(|retired| {
            if retired.frames_until_free == 0 {
                // Safe to free now
                unsafe { device.destroy_buffer(retired.buffer, None); }
                allocator.free(retired.allocation);
                false // Remove from queue
            } else {
                retired.frames_until_free -= 1;
                true // Keep in queue
            }
        });
    }
}
```

**Location**: Add to `TvWindowContext` or create new `BufferManager` struct.

---

### Step 2: Add Growable Buffer Wrapper

**File**: `src/core/tulivuori/buffer.rs`

Create a new `GrowableBuffer<T>` that wraps the existing buffer types:

```rust
pub struct GrowableBuffer<T> {
    /// Current active buffer (FRAMES_IN_FLIGHT copies)
    inner: GenericBuffer<T>,
    /// Queue of old buffers pending retirement
    retire_queue: BufferRetireQueue,
    /// Growth factor (default 2.0)
    growth_factor: f32,
}

impl<T: Copy + Default + bytemuck::Pod> GrowableBuffer<T> {
    pub fn new(/* existing params */) -> Result<Self> { ... }

    /// Write data, growing if necessary
    pub fn write_growing(
        &mut self,
        data: &[T],
        frame_index: usize,
        device: &Device,
        allocator: &GgMutex<vk_mem::Allocator>,
    ) -> Result<()> {
        let current_capacity = self.inner.alloc_count();

        if data.len() > current_capacity {
            self.grow(data.len(), device, allocator)?;
        }

        self.inner.write(data, frame_index)
    }

    /// Grow the buffer to at least `min_capacity`
    fn grow(
        &mut self,
        min_capacity: usize,
        device: &Device,
        allocator: &GgMutex<vk_mem::Allocator>,
    ) -> Result<()> {
        // Calculate new capacity with growth factor
        let new_capacity = (min_capacity as f32 * self.growth_factor) as usize;

        // Allocate new buffer
        let new_inner = GenericBuffer::new(/* ... new_capacity ... */)?;

        // Retire old buffer (all FRAMES_IN_FLIGHT copies)
        let old_inner = std::mem::replace(&mut self.inner, new_inner);
        for i in 0..FRAMES_IN_FLIGHT {
            self.retire_queue.retire(
                old_inner.allocation(i),
                old_inner.buffer(i),
            );
        }

        log::info!("Buffer grew: {} -> {} elements", current_capacity, new_capacity);
        Ok(())
    }

    /// Call once per frame to process retire queue
    pub fn tick_retire_queue(&mut self, device: &Device, allocator: &mut vk_mem::Allocator) {
        self.retire_queue.tick(device, allocator);
    }
}
```

---

### Step 3: Handle Device Buffers (GPU-only)

**File**: `src/core/tulivuori/buffer.rs`

For `GenericDeviceBuffer` (used by materials), the same pattern applies but we also need to handle:
- Buffer device addresses change on reallocation
- Descriptor sets may need updating

```rust
pub struct GrowableDeviceBuffer<T> {
    inner: GenericDeviceBuffer<T>,
    retire_queue: BufferRetireQueue,
    growth_factor: f32,
}

impl<T: Copy + Default + bytemuck::Pod> GrowableDeviceBuffer<T> {
    /// Returns true if the buffer was reallocated (caller must update references)
    pub fn grow_if_needed(&mut self, min_capacity: usize, ...) -> Result<bool> {
        if min_capacity <= self.inner.alloc_count() {
            return Ok(false);
        }

        // ... grow logic ...
        Ok(true) // Signal that buffer addresses changed
    }

    /// Get current buffer device address (may change after grow)
    pub fn device_address(&self, frame_index: usize) -> vk::DeviceAddress { ... }
}
```

---

### Step 4: Update Material Upload System

**File**: `src/core/tulivuori/texture.rs`

The materials system uses buffer device addresses. When the buffer grows:

1. Update `get_materials_address()` to return the new address
2. Since materials are uploaded via push constants/staging buffer, the address is refreshed each frame automatically

```rust
// In TvTextureManager
pub fn stage_materials_growing(&mut self, materials: &[RawMaterial], ...) -> Result<()> {
    // Check if device buffer needs to grow
    if self.device_materials_buffer.grow_if_needed(materials.len(), ...)? {
        log::info!("Materials device buffer grew, addresses updated");
        // Force re-upload to all frame copies
        self.materials_upload_countdown = FRAMES_IN_FLIGHT;
    }

    // Check if staging buffer needs to grow
    self.staging_materials_buffer.write_growing(materials, frame_index, ...)?;

    Ok(())
}
```

---

### Step 5: Update Vertex Buffer in Renderer

**File**: `src/core/render.rs`

Replace `VertexBuffer` usage with growable version:

```rust
// In VkRenderContext initialization (line ~265)
let vertex_buffer = GrowableBuffer::<SpriteVertex>::new(
    /* ... existing params ... */
)?;

// In update_vertex_buffer() (line ~587)
fn update_vertex_buffer(&mut self, sprites: &[SpriteVertex]) -> Result<()> {
    self.vertex_buffer.write_growing(
        sprites,
        self.swapchain.frame_index().current(),
        &self.device,
        &self.allocator,
    )?;
    Ok(())
}

// In render loop, after present (line ~530)
fn on_frame_complete(&mut self) {
    self.vertex_buffer.tick_retire_queue(&self.device, &mut self.allocator.lock());
    self.texture_manager.tick_retire_queues(&self.device, &mut self.allocator.lock());
}
```

---

### Step 6: Add GUI Buffer Growing (Optional)

**File**: `src/core/render.rs`

The GUI buffers (`gui_vtx_buffer`, `gui_idx_buffer`) at line ~660 can use the same pattern:

```rust
self.gui_vtx_buffer.write_growing(gui_vertices, frame_index, ...)?;
self.gui_idx_buffer.write_growing(gui_indices, frame_index, ...)?;
```

---

### Step 7: Centralized Retire Queue Management

**File**: New file `src/core/tulivuori/buffer_manager.rs` or integrate into existing

Consider a centralized `BufferManager` that:
- Owns all retire queues
- Has a single `tick()` called once per frame
- Provides statistics on buffer usage and growth events

```rust
pub struct BufferManager {
    retire_queues: Vec<BufferRetireQueue>,
    stats: BufferStats,
}

pub struct BufferStats {
    total_reallocations: u64,
    current_memory_usage: u64,
    peak_memory_usage: u64,
}
```

---

## Edge Cases and Safety

### 1. Rapid Growth
If a buffer needs to grow multiple times quickly (within FRAMES_IN_FLIGHT frames), old buffers accumulate in the retire queue. This is fine - they'll be freed in order.

### 2. Out-of-Memory on Growth
If allocation fails during growth:
- Return error, let caller handle (crash or reduce load)
- Could implement fallback: wait for retire queue to drain, then retry

### 3. Maximum Size Limits
Add configurable maximum buffer sizes to prevent runaway growth:
```rust
const MAX_VERTEX_BUFFER_SIZE_MB: usize = 256;
```

### 4. Descriptor Set Updates
For buffers referenced by descriptor sets (not currently the case for vertex/material buffers here, but future-proofing):
- Either recreate descriptor sets on grow
- Or use dynamic offsets / buffer device addresses (current approach for materials)

---

## Testing Strategy

1. **Unit test**: Buffer retire queue correctly frees after N frames
2. **Unit test**: Growth calculation is correct (2x with minimum)
3. **Integration test**: Render loop with forced buffer growth mid-frame
4. **Stress test**: Rapidly growing buffer with many small increments

---

## Migration Path

1. Implement `BufferRetireQueue` (no existing code changes)
2. Implement `GrowableBuffer<T>` wrapping `GenericBuffer<T>`
3. Replace vertex buffer with growable version
4. Replace material buffers with growable version
5. Replace GUI buffers with growable version
6. Add statistics/logging
7. Remove old crash-on-overflow code paths

---

## Summary

The key insight is that buffer reallocation is safe if we:
1. Never free a buffer that might be in use by an in-flight frame
2. Use a retire queue with `FRAMES_IN_FLIGHT` frame countdown
3. Tick the retire queue once per frame after present

This approach:
- Avoids GPU stalls (no fence waits during reallocation)
- Has minimal memory overhead (only 2 extra buffer copies during growth)
- Is simple to reason about and debug
