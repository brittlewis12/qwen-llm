//! qwen-llm: a from-scratch inference engine for Qwen 3.5 / 3.6 hybrid
//! Gated DeltaNet (architecture name `qwen3_5` in HF transformers,
//! `LLM_ARCH_QWEN35` in llama.cpp) on Apple Silicon.
//!
//! See [`docs/PLAN.md`](../../../../docs/PLAN.md) for the full architectural
//! decision record. This crate is structured around the v1 plan:
//!
//! * [`gguf`]      — mmap-only GGUF v3 reader. No copies. Tensor descriptor
//!                   table indexing into the mmap'd file.
//! * [`codec`]     — tooling-only CPU dequantization via `llama_cpp_sys_2`'s
//!                   `ggml_get_type_traits().to_float` seam. Universal quant
//!                   coverage; not on the GPU hot path.
//! * [`metal`]     — host-side Metal lifecycle: device, queue, library
//!                   from embedded `kernels.metallib`, pipeline state cache,
//!                   binary archive.
//! * [`model`]     — Qwen3.5/3.6 architecture description: layer pattern,
//!                   GDN/attn dims, tensor name conventions.
//! * [`tensor`]    — backend-agnostic tensor descriptor (mmap'd region +
//!                   shape + ggml type tag).
//! * [`tokenizer`] — Qwen2 byte-level BPE, vocab 248,320, embedded tokens
//!                   read from GGUF metadata.
//!
//! The numerical oracle is `~/code/llama.cpp/build/bin/llama-cli` running
//! against `~/models/Qwen3.5-0.8B.F32.gguf`. Logits-byte-equivalence on this
//! pair is the v1 correctness bar.

pub mod codec;
pub mod forward;
pub mod gguf;
pub mod loader;
pub mod metal;
pub mod model;
pub mod tensor;
pub mod tokenizer;

/// Bytes of the compiled `.metallib` produced by `build.rs` from
/// `../../kernels/*.metal`. Empty until the first kernel is added.
pub(crate) const KERNELS_METALLIB: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/kernels.metallib"));
