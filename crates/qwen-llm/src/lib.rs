//! qwen-llm: inference runtimes and Metal kernels for Qwen, DeepSeek, GLM,
//! Kimi, and Muse model families on Apple Silicon.
//!
//! The crate contains model loaders, runtimes, Metal kernels, and diagnostic
//! tools for these model families.
//!
//! * [`gguf`]      — mmap-backed GGUF v3 reader. Tensor descriptor table
//!                   indexes one or more mmap'd GGUF shards.
//! * [`codec`]     — tooling-only CPU dequantization via `llama_cpp_sys_2`'s
//!                   `ggml_get_type_traits().to_float` seam. Universal quant
//!                   coverage; not on the GPU hot path.
//! * [`metal`]     — host-side Metal lifecycle: device, queue, library
//!                   from embedded `kernels.metallib`, pipeline state cache,
//!                   binary archive.
//! * [`tensor`]    — backend-agnostic tensor descriptor for GGUF-backed
//!                   tensors (shard/offset/shape/type).
//! * Metal weights use persistent `StorageModeShared` `MTLBuffer`s. Load policy
//!   may copy a tensor or retain a read-only view over its GGUF mapping.
//!

pub mod cache_probe;
pub mod checkpoint_codec;
pub(crate) mod checkpoint_fs;
pub mod checkpoint_identity;
pub mod checkpoint_store;
pub mod codec;
pub mod compare;
pub mod deepseek_v4;
pub mod deepseek_v41;
pub mod deepseek_v4_cache;
pub mod deepseek_v4_census;
pub mod deepseek_v4_checkpoint_store;
pub mod deepseek_v4_metal;
pub mod deepseek_v4_oracle;
pub mod dense_batch8;
pub mod durable_store;
pub mod env_flag;
pub mod forward;
pub mod gguf;
pub mod glm5_next;
pub mod glm5_next_chat;
pub mod glm5_next_metal;
pub mod k2_horizon;
pub mod k2_horizon_chat;
pub mod k2_horizon_metal;
pub mod k2_horizon_plan;
#[cfg(test)]
pub mod k2_horizon_reference;
pub mod k2_horizon_runtime;
pub mod loader;
pub mod metal;
pub mod metal_dflash;
pub mod metal_forward;
pub mod metal_mtp;
pub mod model;
pub mod model_family;
pub mod moe_batch16;
pub mod muse_glimmer;
pub mod muse_glimmer_lens;
pub mod muse_glimmer_lens_fit;
pub mod muse_glimmer_metal;
pub mod muse_glimmer_prompt;
pub mod muse_glimmer_request;
pub mod muse_glimmer_residency;
pub mod muse_glimmer_runtime;
pub mod muse_glimmer_text_session;
pub mod pid_metrics;
pub mod prefetch;
pub mod prefix_cache;
pub mod prompt_lookup;
pub mod qwen4exp;
#[cfg(test)]
pub(crate) mod qwen4exp_child_profile;
#[cfg(test)]
pub(crate) mod qwen4exp_composition_trace;
pub mod qwen4exp_forward;
pub mod qwen4exp_gdn;
pub mod qwen4exp_layer_zero;
pub mod qwen4exp_layers_zero_one;
pub mod qwen4exp_loader;
pub mod qwen4exp_metal;
pub mod qwen4exp_moe;
pub mod qwen4exp_packed_prefill;
pub mod qwen4exp_ple;
pub mod qwen4exp_ple_metal;
pub mod qwen4exp_post_ple_block;
mod qwen4exp_profile;
pub mod qwen4exp_qsa;
pub mod qwen4exp_residency;
pub mod qwen4exp_runtime;
pub mod qwen4exp_text_session;
mod qwen_queue2;
pub mod runtime;
pub mod sampling;
pub mod snapshot_policy;
pub mod tensor;
#[doc(hidden)]
pub mod test_fixtures;
pub mod tokenizer;

pub mod tool_schema;
pub mod trellis_ldlq;
pub mod trellis_offline;
pub mod workspace_lens;

/// Deployment target used to compile the product Metal library.
pub const PRODUCT_METALLIB_TARGET: &str = env!("QWEN_PRODUCT_METALLIB_TARGET");
/// Deployment target used to compile the research Metal library.
pub const RESEARCH_METALLIB_TARGET: &str = env!("QWEN_RESEARCH_METALLIB_TARGET");
/// Compiler and macOS SDK identity used to compile both Metal libraries.
pub fn metallib_build_identity() -> (&'static str, &'static str) {
    (
        env!("QWEN_METALLIB_COMPILER_IDENTITY"),
        env!("QWEN_METALLIB_SDK_VERSION"),
    )
}

/// Cargo diagnostic features enabled in this qwen-llm build.
pub fn enabled_diagnostic_features() -> &'static [&'static str] {
    &[
        #[cfg(feature = "dflash-k0s-diagnostics")]
        "dflash-k0s-diagnostics",
        #[cfg(feature = "dsv4-diagnostics")]
        "dsv4-diagnostics",
        #[cfg(feature = "perf-ablations")]
        "perf-ablations",
    ]
}

/// Bytes of the compiled `.metallib` produced by `build.rs` from
/// `../../kernels/*.metal`.
pub(crate) const KERNELS_METALLIB: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/kernels.metallib"));
/// Bench/research-only probe kernels from `../../kernels/research/*.metal`,
/// loaded lazily on the first pipeline miss so product startup never touches
/// them.
pub(crate) const KERNELS_RESEARCH_METALLIB: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/kernels_research.metallib"));

#[cfg(test)]
mod metallib_identity_tests {
    #[test]
    fn metallib_build_identity_is_populated() {
        let (compiler, sdk) = super::metallib_build_identity();
        assert!(!compiler.is_empty());
        assert!(!sdk.is_empty());
    }
}
