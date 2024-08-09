// due to be stabilised soon(tm)
// https://github.com/rust-lang/rust/issues/65991
#![feature(trait_upcasting)]

pub use bincode;

pub mod core;
pub mod resource;
pub mod shader;
pub mod gui;
pub mod util;
