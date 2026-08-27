//! qwen-llm: a from-scratch inference engine for Qwen 3.5 / 3.6 hybrid
//! Gated DeltaNet (architecture name `qwen3_5` in HF transformers,
//! `LLM_ARCH_QWEN35` in llama.cpp) on Apple Silicon.
//!
//! See [`docs/PLAN.md`](../../../../docs/PLAN.md) for the full architectural
//! decision record. This crate is structured around the v1 plan:
//!
//! * [`gguf`]      — mmap-backed GGUF v3 reader. Tensor descriptor table
//!                   indexes one or more mmap'd GGUF shards.
//! * [`codec`]     — tooling-only CPU dequantization via `llama_cpp_sys_2`'s
//!                   `ggml_get_type_traits().to_float` seam. Universal quant
//!                   coverage; not on the GPU hot path.
//! * [`metal`]     — host-side Metal lifecycle: device, queue, library
//!                   from embedded `kernels.metallib`, pipeline state cache,
//!                   binary archive.
//! * [`model`]     — Qwen3.5/3.6 architecture description: layer pattern,
//!                   GDN/attn dims, tensor name conventions.
//! * [`tensor`]    — backend-agnostic tensor descriptor for GGUF-backed
//!                   tensors (shard/offset/shape/type).
//! * Metal weights use persistent `StorageModeShared` `MTLBuffer`s. Load policy
//!   may copy a tensor or retain a read-only view over its GGUF mapping.
//! * [`tokenizer`] — Qwen2 byte-level BPE, vocab 248,320, embedded tokens
//!                   read from GGUF metadata.
//!
//! The numerical oracle is `~/code/llama.cpp/build/bin/llama-cli` running
//! against `~/models/Qwen3.5-0.8B.F32.gguf`. Logits-byte-equivalence on this
//! pair is the v1 correctness bar.

pub mod cache_probe;
pub mod checkpoint_codec;
pub(crate) mod checkpoint_fs;
pub mod checkpoint_identity;
pub mod checkpoint_store;
pub mod codec;
pub mod deepseek_v4;
pub mod deepseek_v4_cache;
pub mod deepseek_v4_census;
pub mod deepseek_v4_checkpoint_store;
pub mod deepseek_v4_metal;
pub mod deepseek_v4_oracle;
pub mod dense_batch8;
pub mod env_flag;
pub mod forward;
pub mod gguf;
pub mod loader;
pub mod metal;
pub mod metal_dflash;
pub mod metal_forward;
pub mod metal_mtp;
pub mod model;
pub mod model_family;
pub mod moe_batch16;
pub mod pid_metrics;
pub mod prefetch;
pub mod prefix_cache;
pub mod prompt_lookup;
pub mod qwen4exp;
pub mod qwen4exp_forward;
pub mod qwen4exp_gdn;
pub mod qwen4exp_layer_zero;
pub mod qwen4exp_layers_zero_one;
pub mod qwen4exp_layers_zero_three;
pub mod qwen4exp_loader;
pub mod qwen4exp_metal;
pub mod qwen4exp_moe;
pub mod qwen4exp_ple;
pub mod qwen4exp_ple_metal;
pub mod qwen4exp_post_ple_block;
pub mod qwen4exp_qsa;
pub mod qwen4exp_residency;
mod qwen_queue2;
pub mod runtime;
pub mod sampling;
pub mod tensor;
pub mod tokenizer;
pub mod trellis_ldlq;
pub mod trellis_offline;

/// Bytes of the compiled `.metallib` produced by `build.rs` from
/// `../../kernels/*.metal`. Empty until the first kernel is added.
pub(crate) const KERNELS_METALLIB: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/kernels.metallib"));
