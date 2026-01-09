# Achieving 100% Test Coverage with Bincode Derive Macros

When a struct derives `bincode::{Encode, Decode, BorrowDecode}`, the derive macros generate trait implementations that need explicit testing to achieve full coverage.

## The Problem

The derive macro generates three trait implementations:
- `Encode` - serialization
- `Decode` - deserialization
- `BorrowDecode` - zero-copy deserialization

Each implementation has both success and error paths. Simply testing a successful roundtrip won't cover the error paths where decoding/encoding fails partway through.

## The Solution

A utility function `test_bincode_error_paths<T>()` is available in `src/util/mod.rs` under the `test_util` module:

```rust
// In src/util/mod.rs
#[cfg(test)]
pub mod test_util {
    pub fn test_bincode_error_paths<T>()
    where
        T: bincode::Encode
            + bincode::Decode<()>
            + for<'de> bincode::BorrowDecode<'de, ()>
            + Default,
    { /* ... */ }
}
```

### Usage

For types that implement `Default`:

```rust
#[test]
fn test_bincode() {
    // Success paths
    let c = MyStruct::new(/* ... */);
    let encoded = bincode::encode_to_vec(c, bincode::config::standard()).unwrap();
    let (decoded, _): (MyStruct, _) =
        bincode::decode_from_slice(&encoded, bincode::config::standard()).unwrap();
    assert_eq!(c, decoded);

    // Error paths - one line!
    crate::util::test_util::test_bincode_error_paths::<MyStruct>();
}
```

For types without `Default`, test manually using `size_of`:

```rust
use std::mem::size_of;
let cfg = bincode::config::legacy();
let mut buf = vec![0u8; size_of::<MyStruct>() - 1];
assert!(bincode::encode_into_slice(val, &mut buf, cfg).is_err());
assert!(bincode::decode_from_slice::<MyStruct, _>(&buf, cfg).is_err());
assert!(bincode::borrow_decode_from_slice::<MyStruct, _>(&buf, cfg).is_err());
```

**Note for Claude:** If a type has `Encode`/`Decode` derives but doesn't implement `Default`, ask the user whether they'd like to add `#[derive(Default)]` to the type so the helper function can be used.

## Key Points

1. **All three error paths are required** - removing any one of `encode_into_slice`, `decode_from_slice`, or `borrow_decode_from_slice` error tests will drop coverage below 100%.

2. **Use `legacy()` config for error tests** - this uses fixed-size encoding (4 bytes per f32, etc.), making buffer size calculations predictable.

3. **Use `size_of::<T>() - 1`** - this automatically calculates a buffer that's exactly 1 byte too small, triggering an error on the last field.

## Why This Works

The derive-generated code uses the `?` operator to propagate errors when encoding/decoding each field. A truncated buffer causes an error at some field, exercising that error propagation path. The specific field that fails doesn't matter for coverage - any error path through the `?` operator counts as covered.
