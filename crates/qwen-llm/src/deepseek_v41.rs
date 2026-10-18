//! DeepSeek V4.1 GGUF loading and NativeF32 execution.
//! Geometry comes from GGUF metadata; local memory fit is a runtime decision.

pub mod engram;
pub mod forward;
pub mod gguf;
pub mod operators;

#[cfg(test)]
mod mxfp4;
