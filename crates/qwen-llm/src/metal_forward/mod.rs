//! End-to-end Metal forward pass driver.
//!
//! Mirrors `crate::forward::Forward` but with all heavy ops on Metal,
//! orchestrated through one `MTLCommandBuffer` per token. The CPU oracle
//! (`forward.rs`) stays as the validation reference.
//!
//! v1 scope:
//!
//! * Single-token decode (no batched prefill).
//! * F32-only first (Qwen3.5-0.8B.F32). Quantized lift comes after the
//!   F32 path validates against the oracle — every kernel that takes
//!   `mat_vec_f32` will swap to `mat_vec_q4_K` / `mat_vec_q6_K` based on
//!   tensor `dtype`.
//! * Full-attn layers built from existing kernels (q-proj, k-proj,
//!   v-proj, q-norm, k-norm, RoPE, KV append, scoring, softmax,
//!   V-aggregate, gate, output proj). No fused attn block yet — the
//!   composition lands first; a fused version is a v2 perf project.
//! * Persistent `MetalTensor`s for all weights; one `MetalSession` owns
//!   the per-sequence state (GDN conv buffers + SSM states + KV cache).
//! * Activation arena: per-step scratch buffers held in `MetalSession`,
//!   reused across layers.
//!
//! v2 lift after the F32 forward validates:
//!
//! * Quantized weight path — change `MetalModel::load` to use
//!   `from_gguf_tensor` directly (currently dequants F32 via codec).
//! * Indirect Command Buffer (ICB) — encode the per-token sequence once,
//!   replay it. The encode-only API is already designed for this.
//! * Fused full-attn block.

use crate::gguf::GgufFile;
use crate::loader::{Block, Model, MoeFfn};
pub use crate::metal::PostBlockIntervention;
use crate::metal::{
    Buffer, GgufBackingEligibility, KernelEncoder, MetalContext, MetalError, MetalGgufBacking,
    MetalTensor, MetalTensorProvenance, MetalTimestampSampleBuffer, RetainedStorageDisposition,
    RetainedStorageFallback, RetainedStoragePlan, attn_v4_choose_nwg, attn_v4_choose_tile_c,
    encode_add_inplace_f32, encode_argmax_f32, encode_argmax_f32_greedy,
    encode_attn_decode_f16kv_f32, encode_attn_decode_v4_f32, encode_axpy_scalar_f32,
    encode_dot_sigmoid_f32, encode_ffn_swiglu_q4_K_f32, encode_fill_f32,
    encode_gdn_decay_chain_f32, encode_gdn_step_decay_f32, encode_get_rows_f32,
    encode_l2_norm_batched_f32, encode_l2_norm_pair_batched_f32, encode_mat_vec_f32,
    encode_mat_vec_f32_sigmoid, encode_mat_vec_q4_k_f32, encode_mat_vec_q5_k_f32,
    encode_mat_vec_q6_k_f32, encode_moe_down_bf16_f32, encode_moe_down_f32_f32,
    encode_moe_down_iq4_nl_f32, encode_moe_down_iq4_xs_f32, encode_moe_down_iq4_xs_f32_fast,
    encode_moe_down_q4_K_f32, encode_moe_down_q5_K_f32,
    encode_moe_down_weighted_sum_q5_K_f32_packed_slots,
    encode_moe_down_weighted_sum_q5_K_f32_packed_slots_k512_r2,
    encode_moe_down_weighted_sum_q6_K_f32, encode_moe_down_weighted_sum_q8_0_f32,
    encode_moe_grouped_finalizer_f32, encode_moe_mat_vec_bf16_f32, encode_moe_mat_vec_f32,
    encode_moe_mat_vec_iq3_s_f32, encode_moe_mat_vec_iq3_xxs_f32, encode_moe_mat_vec_q5_K_f32,
    encode_moe_shared_accum_resid_f32, encode_moe_swiglu_iq3_s_f32,
    encode_moe_swiglu_iq3_s_f32_fast, encode_moe_swiglu_iq3_xxs_f32,
    encode_moe_swiglu_iq3_xxs_f32_fast, encode_moe_swiglu_iq4_xs_f32, encode_moe_swiglu_q4_K_f32,
    encode_moe_swiglu_q6_K_f32, encode_moe_swiglu_q8_0_f32, encode_moe_weighted_sum_f32,
    encode_mul_f32, encode_post_block_intervention_f32,
    encode_qk_rms_norm_rope_f32_packed_consecutive, encode_residual_rms_norm_mul_f32,
    encode_rms_norm_batched_f32, encode_rms_norm_batched_src_strided_f32, encode_rms_norm_mul_f32,
    encode_rmsnorm_gated_f32, encode_rope_neox_f32, encode_rope_neox_pair_f32,
    encode_scatter_offset_f32_to_f16_kv, encode_scatter_offset_f32_to_q8_0_kv,
    encode_shared_swiglu_q8_0_f32, encode_sigmoid_f32, encode_sigmoid_mul_gate_strided_f32,
    encode_silu_mul_f32, encode_split_q_gate_f32, encode_ssm_conv_silu_f32,
    encode_topk_logits_softmax_dot_sigmoid_f32, encode_topk_logits_softmax_f32,
    encode_topk_logits_softmax_parallel_f32, evaluate_metal_memory_admission, host_page_size_bytes,
    plan_retained_storage,
};
use crate::model::ArchKind;
use crate::sampling::{BoundedTopKEvidence, GreedySelection, SampledToken, Sampler, SamplingError};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use sha2::{Digest, Sha256};
use std::{cell::Cell, collections::HashSet, mem::MaybeUninit, sync::OnceLock};

/// Max NWG (split-K partitions) the v4 dispatcher will ever request.
/// Sets the size of session-resident partial buffers; see
/// `attn_v4_choose_nwg` for the selection heuristic.
pub const ATTN_V4_MAX_NWG: usize = 1024;
use crate::tensor::{GgmlType, TensorDesc, ggml_type_layout_raw};

crate::env_flag!(default_off kv_q8_flag, "QWEN_KV_Q8");

crate::env_flag!(default_on concurrent_gdn_moe_decode_enabled, "QWEN_DECODE_MOE_CONCURRENT_GDN");
crate::env_flag!(
    /// Routes dense single-token decode through concurrent GDN front projections; disabling selects serial dense decode.
    default_on concurrent_gdn_dense_decode_enabled,
    "QWEN_DECODE_DENSE_CONCURRENT_GDN"
);
crate::env_flag!(default_on concurrent_shared_moe_decode_enabled, "QWEN_DECODE_MOE_CONCURRENT_SHARED");
crate::env_flag!(
    /// Fuses eligible Q8_0 shared-expert gate/up SwiGLU into one kernel; disabling restores separate mat-vec and SwiGLU dispatches.
    default_on decode_shared_swiglu_q8_enabled,
    "QWEN_DECODE_SHARED_SWIGLU_Q8"
);
crate::env_flag!(default_on decode_moe_iq3_fused_swiglu_enabled, "QWEN_DECODE_MOE_IQ3_FUSED_SWIGLU");
crate::env_flag!(default_on decode_moe_iq3_fast_swiglu_enabled, "QWEN_DECODE_MOE_IQ3_FAST_SWIGLU");
crate::env_flag!(default_on decode_moe_q5_down_fused_enabled, "QWEN_DECODE_MOE_Q5_DOWN_FUSED");
crate::env_flag!(default_on decode_moe_iq4_down_fast_enabled, "QWEN_DECODE_MOE_IQ4_DOWN_FAST");
crate::env_flag!(default_on decode_moe_q5_down_k512_r2_enabled, "QWEN_DECODE_MOE_Q5_DOWN_K512_R2");
crate::env_flag!(
    /// Fuses shared-expert accumulation, mixer update, and residual add in MoE decode; disabling restores separate accumulation and add passes.
    default_on decode_moe_fused_finalizer_enabled,
    "QWEN_DECODE_MOE_FUSED_FINALIZER"
);
crate::env_flag!(default_on decode_moe_grouped_finalizer_enabled, "QWEN_DECODE_MOE_GROUPED_FINALIZER");
crate::env_flag!(default_on decode_attn_sigmoid_mul_enabled, "QWEN_DECODE_ATTN_SIGMOID_MUL");
crate::env_flag!(default_off moe_router_f16_enabled, "QWEN_MOE_ROUTER_F16");
crate::env_flag!(perf_ablation decode_gdn_noop_front_enabled, "QWEN_DECODE_GDN_NOOP_FRONT");
crate::env_flag!(perf_ablation decode_gdn_noop_out_enabled, "QWEN_DECODE_GDN_NOOP_OUT");
crate::env_flag!(default_on decode_gdn_fused_beta_proj_enabled, "QWEN_DECODE_GDN_FUSED_BETA_PROJ");
crate::env_flag!(default_on decode_rope_pair_enabled, "QWEN_DECODE_ROPE_PAIR");
crate::env_flag!(
    default_on decode_qk_norm_rope_fused_enabled,
    "QWEN_DECODE_QK_NORM_ROPE_FUSED"
);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NativeQuantEmbeddingMode {
    Auto,
    Forced,
    Disabled,
    Invalid,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NativeQuantEmbeddingSelection {
    AutoPromoted,
    Forced,
    AutoUnpromoted,
    RollbackDisabled,
    InvalidDisabled,
    Unsupported,
}

impl NativeQuantEmbeddingSelection {
    fn uses_native(self) -> bool {
        matches!(self, Self::AutoPromoted | Self::Forced)
    }

    fn label(self) -> &'static str {
        match self {
            Self::AutoPromoted => "auto-promoted",
            Self::Forced => "forced",
            Self::AutoUnpromoted => "auto-unpromoted",
            Self::RollbackDisabled => "rollback-disabled",
            Self::InvalidDisabled => "invalid-disabled",
            Self::Unsupported => "unsupported",
        }
    }
}

const GGUF_NO_COPY_ALIGNMENT: usize = 32;
const PARALLEL_COPY_MAX_WORKERS: usize = 4;
const PARALLEL_COPY_AUTO_OVERRIDE_ENVS: [&str; 7] = [
    "QWEN_GGUF_NO_COPY",
    "QWEN_GGUF_NO_COPY_PREFAULT",
    "QWEN_NATIVE_QUANT_EMBED",
    "QWEN_MOE_ROUTER_F16",
    "QWEN_MOE_IQ3_EXPERT_NATIVE",
    "QWEN_PREFILL_MOE_GROUPED_IQ3_GATEUP",
    "QWEN_PREFILL_ATTN_FUSED_QKV_G8",
];

#[derive(Debug)]
struct ParallelCopyLoadPlan {
    request_count: usize,
    source_bytes: u64,
    page_size: usize,
    destination_bytes: u64,
    padding_bytes: u64,
    padded_resources: usize,
    fallback_bytes: u64,
    max_buffer_length: usize,
    alignment: usize,
    cuts: Vec<usize>,
    task_counts: Vec<usize>,
    worker_bytes: Vec<u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GgufNoCopyMode {
    Auto,
    Disabled,
    Forced,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GgufParallelCopyMode {
    Auto,
    Disabled,
    ForcedCopy,
    ForcedPread,
    ForcedPageRoundedCopy,
}

impl GgufParallelCopyMode {
    fn is_forced(self) -> bool {
        matches!(
            self,
            Self::ForcedCopy | Self::ForcedPread | Self::ForcedPageRoundedCopy
        )
    }

    fn forced_configuration(self) -> Option<(ParallelPopulationMethod, ParallelDestinationLength)> {
        match self {
            Self::ForcedCopy => Some((
                ParallelPopulationMethod::MmapCopy,
                ParallelDestinationLength::LogicalExact,
            )),
            Self::ForcedPread => Some((
                ParallelPopulationMethod::Pread,
                ParallelDestinationLength::LogicalExact,
            )),
            Self::ForcedPageRoundedCopy => Some((
                ParallelPopulationMethod::MmapCopy,
                ParallelDestinationLength::PageRounded,
            )),
            Self::Auto | Self::Disabled => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GgufNoCopyPrefaultMode {
    Default,
    Enabled,
    Disabled,
}

#[derive(Debug)]
pub struct GgufStorageLoadConfiguration {
    no_copy_mode: GgufNoCopyMode,
    parallel_mode: GgufParallelCopyMode,
    prefault_mode: GgufNoCopyPrefaultMode,
}

impl GgufStorageLoadConfiguration {
    pub fn storage_request(&self) -> GgufStorageRequest {
        match self.no_copy_mode {
            GgufNoCopyMode::Forced => GgufStorageRequest::Map("explicit_environment_override"),
            GgufNoCopyMode::Disabled => GgufStorageRequest::Copy("explicit_environment_override"),
            GgufNoCopyMode::Auto if self.parallel_mode.is_forced() => {
                GgufStorageRequest::Copy("explicit_parallel_copy_precedence")
            }
            GgufNoCopyMode::Auto => GgufStorageRequest::Automatic,
        }
    }
}

fn validate_forced_storage_embedding_policy(
    parallel_mode: GgufParallelCopyMode,
    embedding_mode: NativeQuantEmbeddingMode,
) -> Result<(), MfError> {
    if parallel_mode.is_forced() && embedding_mode != NativeQuantEmbeddingMode::Auto {
        return Err(MfError::LoadPolicy(
            "forced parallel storage requires production-auto native embedding selection"
                .to_string(),
        ));
    }
    Ok(())
}

fn storage_environment_value(name: &str) -> Result<Option<String>, MfError> {
    match std::env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => {
            Err(MfError::LoadPolicy(format!("{name} is not valid Unicode")))
        }
    }
}

fn resolve_gguf_storage_load_configuration_from_values(
    no_copy: Option<&str>,
    owned_arena: Option<&str>,
    parallel_copy: Option<&str>,
    prefault: Option<&str>,
    native_embedding_present: bool,
    native_embedding_mode: NativeQuantEmbeddingMode,
    router_f16: Option<&str>,
) -> Result<GgufStorageLoadConfiguration, MfError> {
    let no_copy_mode = parse_gguf_no_copy_mode(no_copy)?;
    parse_gguf_owned_arena_mode(owned_arena)?;
    let parallel_mode = parse_gguf_parallel_copy_mode(parallel_copy)?;
    let prefault_mode = parse_gguf_no_copy_prefault(prefault)?;
    validate_parallel_copy_policy(
        parallel_mode,
        no_copy_mode,
        prefault.is_some(),
        native_embedding_present,
        router_f16,
    )?;
    validate_forced_storage_embedding_policy(parallel_mode, native_embedding_mode)?;
    if matches!(
        no_copy_mode,
        GgufNoCopyMode::Disabled | GgufNoCopyMode::Auto
    ) && prefault_mode != GgufNoCopyPrefaultMode::Default
    {
        return Err(MfError::LoadPolicy(
            "QWEN_GGUF_NO_COPY_PREFAULT requires QWEN_GGUF_NO_COPY=1".to_string(),
        ));
    }
    Ok(GgufStorageLoadConfiguration {
        no_copy_mode,
        parallel_mode,
        prefault_mode,
    })
}

pub fn resolve_gguf_storage_load_configuration() -> Result<GgufStorageLoadConfiguration, MfError> {
    // Unset uses a storage-performance heuristic fitted to measurements, not a memory-admission guarantee: copy needs 15% working-set headroom plus a 256 MiB session reserve; 76% copied decoded faster, while 97% copied loaded 4–6x slower with heavy paging. 1 maps weights and 0 forces copies.
    let no_copy = storage_environment_value("QWEN_GGUF_NO_COPY")?;
    // Owned arena selects copied weight storage for ordinary Qwen.
    let owned_arena = storage_environment_value("QWEN_GGUF_OWNED_ARENA")?;
    // Forced parallel population plans realize copied weight buffers.
    let parallel_copy = storage_environment_value("QWEN_GGUF_PARALLEL_COPY")?;
    let prefault = storage_environment_value("QWEN_GGUF_NO_COPY_PREFAULT")?;
    let native_embedding_present = std::env::var_os("QWEN_NATIVE_QUANT_EMBED").is_some();
    let native_embedding_mode = match std::env::var("QWEN_NATIVE_QUANT_EMBED") {
        Ok(value) => parse_native_quant_embedding_mode(Some(&value)),
        Err(std::env::VarError::NotPresent) => NativeQuantEmbeddingMode::Auto,
        Err(std::env::VarError::NotUnicode(_)) => NativeQuantEmbeddingMode::Invalid,
    };
    let router_f16 = match storage_environment_value("QWEN_MOE_ROUTER_F16") {
        Ok(value) => value,
        Err(_)
            if parallel_copy.as_deref().is_some_and(|value| {
                parse_gguf_parallel_copy_mode(Some(value))
                    .is_ok_and(GgufParallelCopyMode::is_forced)
            }) =>
        {
            return Err(MfError::LoadPolicy(
                "QWEN_MOE_ROUTER_F16 is not valid Unicode with parallel copy".to_string(),
            ));
        }
        Err(_) => None,
    };
    resolve_gguf_storage_load_configuration_from_values(
        no_copy.as_deref(),
        owned_arena.as_deref(),
        parallel_copy.as_deref(),
        prefault.as_deref(),
        native_embedding_present,
        native_embedding_mode,
        router_f16.as_deref(),
    )
}

crate::env_flag!(perf_ablation decode_gdn_noop_qkv_flag, "QWEN_DECODE_GDN_NOOP_QKV");
crate::env_flag!(perf_ablation decode_gdn_noop_z_flag, "QWEN_DECODE_GDN_NOOP_Z");
crate::env_flag!(perf_ablation decode_gdn_noop_beta_flag, "QWEN_DECODE_GDN_NOOP_BETA");
crate::env_flag!(perf_ablation decode_gdn_noop_alpha_flag, "QWEN_DECODE_GDN_NOOP_ALPHA");
crate::env_flag!(
    /// Normalizes GDN query and key tensors with a paired L2 dispatch; disabling uses separate query and key normalization dispatches.
    default_on decode_gdn_pair_l2_enabled,
    "QWEN_DECODE_GDN_PAIR_L2"
);

crate::env_flag!(default_off phase_gdn_proj_split_enabled, "QWEN_PHASE_GDN_PROJ_SPLIT");
crate::env_flag!(default_off phase_gdn_tail_split_enabled, "QWEN_PHASE_GDN_TAIL_SPLIT");

crate::env_flag!(default_off phase_moe_cpu_route_enabled, "QWEN_PHASE_MOE_CPU_ROUTE");
crate::env_flag!(default_off phase_moe_route_replay_enabled, "QWEN_PHASE_MOE_ROUTE_REPLAY");
crate::env_flag!(default_off phase_lm_argmax_enabled, "QWEN_PHASE_LM_ARGMAX");
crate::env_flag!(default_off decode_fused_residual_rmsnorm_enabled, "QWEN_DECODE_FUSED_RESIDUAL_RMSNORM");
crate::env_flag!(perf_ablation decode_moe_noop_route_enabled, "QWEN_DECODE_MOE_NOOP_ROUTE");
crate::env_flag!(perf_ablation decode_moe_noop_routed_gateup_enabled, "QWEN_DECODE_MOE_NOOP_ROUTED_GATEUP");
crate::env_flag!(perf_ablation decode_moe_noop_routed_down_enabled, "QWEN_DECODE_MOE_NOOP_ROUTED_DOWN");

thread_local! {
    static MATMAT_BF16_BFLOAT_ACT_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
}

/// Return type for [`MetalForward::single_token_phase_pland`]:
/// `(logits, wall_with_artifact_ms, per-phase GPU ms map)`. The
/// per-phase entries are `(phase_name, gpu_ms)`.
pub type PhaseProfileOutput = (Vec<f32>, f64, Vec<(String, f64)>);

thread_local! {
    static T9_FFN_CAPTURE: std::cell::RefCell<Option<Vec<(usize, MetalTensor, MetalTensor)>>> =
        const { std::cell::RefCell::new(None) };
    static T9_FFN_CALL_IDX: Cell<usize> = const { Cell::new(0) };
    static GDN_ALPHA_CENSUS_CAPTURE: std::cell::RefCell<Option<Vec<(usize, MetalTensor)>>> =
        const { std::cell::RefCell::new(None) };
}

use objc2_metal::{
    MTLBuffer, MTLCPUCacheMode, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandQueue,
    MTLComputePipelineState, MTLDevice, MTLHazardTrackingMode, MTLResource, MTLStorageMode,
};

#[cfg(test)]
thread_local! {
    static METAL_LOAD_TEST_LINES: std::cell::RefCell<Option<Vec<String>>> =
        const { std::cell::RefCell::new(None) };
}

#[derive(Debug, thiserror::Error)]
pub enum MfError {
    #[error("metal: {0}")]
    Metal(#[from] MetalError),
    #[error("codec: {0}")]
    Codec(#[from] crate::codec::CodecError),
    #[error("token {0} out of vocab range {1}")]
    BadToken(i32, u32),
    #[error("qwen35moe Metal path not implemented yet")]
    UnsupportedMoe,
    #[error("no executable Metal kernel for tensor {name} with dtype {dtype:?}")]
    UnsupportedDtype { name: String, dtype: GgmlType },
    #[error("metal load policy: {0}")]
    LoadPolicy(String),
    #[error("snapshot validation: {0}")]
    Snapshot(#[from] SnapshotValidationError),
    #[error("Metal command buffer failed: status={status} error={error}")]
    CommandBuffer { status: String, error: String },
    #[error("Metal session is poisoned: {reason}")]
    SessionPoisoned { reason: &'static str },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ArgmaxReduction {
    SpeculativeLowest,
    GreedyTotal,
}

#[doc(hidden)]
#[derive(Clone, Copy)]
pub enum LmHeadTail<'a> {
    Resident,
    CompactQ6K {
        weight: &'a MetalTensor,
        output: &'a MetalTensor,
    },
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LmHeadTailKind {
    Resident,
    CompactQ6K,
}

impl LmHeadTailKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Resident => "resident",
            Self::CompactQ6K => "compact_q6_k",
        }
    }
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LmHeadTailEvidence {
    pub kind: LmHeadTailKind,
    pub n_in: usize,
    pub n_out: usize,
    pub weight_offset: u64,
    pub output_offset: u64,
    pub tail_dispatches: u32,
    pub full_head_dispatches: u32,
    pub command_completed: bool,
    pub command_error_none: bool,
}

/// All weight tensors, resident as `MetalTensor`s. Loaded once at session
/// start. v1: only F32 weights are supported here; quantized path comes
/// in v2 by switching `MetalModel::load` to use `MetalTensor::from_gguf_tensor`
/// directly (instead of going through the F32 codec) and the kernel
/// dispatchers to pick `_q4_k`/`_q6_k` based on dtype.
pub struct MetalModel {
    /// Reference back to the loader's bound model. Carries `arch`, the
    /// layer schedule (GDN vs Attn), tied-embedding flag.
    pub arch: crate::model::Arch,
    pub tied_embeddings: bool,

    pub token_embd: MetalTensor,
    pub output_norm: MetalTensor,
    pub lm_head: MetalTensor,

    pub blocks: Vec<MetalBlock>,
}

impl MetalModel {
    pub fn retained_gguf_weight_buffers(&self) -> Vec<&Buffer> {
        fn add<'a>(tensor: &'a MetalTensor, buffers: &mut Vec<&'a Buffer>) {
            if tensor.provenance() == MetalTensorProvenance::RetainedGgufReadOnly {
                let pointer = objc2::rc::Retained::as_ptr(&tensor.buffer);
                if !buffers
                    .iter()
                    .any(|buffer| objc2::rc::Retained::as_ptr(*buffer) == pointer)
                {
                    buffers.push(&tensor.buffer);
                }
            }
        }
        fn add_moe<'a>(moe: &'a MetalMoeFfn, buffers: &mut Vec<&'a Buffer>) {
            add(&moe.gate_inp, buffers);
            add(&moe.gate_exps, buffers);
            add(&moe.up_exps, buffers);
            add(&moe.down_exps, buffers);
            add(&moe.gate_inp_shexp, buffers);
        }
        let mut buffers = Vec::new();
        add(&self.token_embd, &mut buffers);
        add(&self.output_norm, &mut buffers);
        add(&self.lm_head, &mut buffers);
        for block in &self.blocks {
            match block {
                MetalBlock::Gdn(block) => {
                    for tensor in [
                        &block.attn_norm,
                        &block.post_attn_norm,
                        &block.ffn_gate,
                        &block.ffn_up,
                        &block.ffn_down,
                        &block.in_proj_qkv,
                        &block.in_proj_z,
                        &block.beta_proj,
                        &block.alpha_proj,
                        &block.a_log,
                        &block.dt_bias,
                        &block.conv1d,
                        &block.norm,
                        &block.out_proj,
                    ] {
                        add(tensor, &mut buffers);
                    }
                    if let Some(moe) = &block.ffn_moe {
                        add_moe(moe, &mut buffers);
                    }
                }
                MetalBlock::Attn(block) => {
                    for tensor in [
                        &block.attn_norm,
                        &block.post_attn_norm,
                        &block.ffn_gate,
                        &block.ffn_up,
                        &block.ffn_down,
                        &block.q,
                        &block.k,
                        &block.v,
                        &block.o,
                        &block.q_norm,
                        &block.k_norm,
                    ] {
                        add(tensor, &mut buffers);
                    }
                    if let Some(tensor) = &block.qkv_fused {
                        add(tensor, &mut buffers);
                    }
                    if let Some(moe) = &block.ffn_moe {
                        add_moe(moe, &mut buffers);
                    }
                }
            }
        }
        buffers
    }

    pub fn snapshot_abi(&self) -> SnapshotAbi {
        let n_attn_layers = self
            .blocks
            .iter()
            .filter(|block| matches!(block, MetalBlock::Attn(_)))
            .count() as u32;
        let n_gdn_layers = self
            .blocks
            .iter()
            .filter(|block| matches!(block, MetalBlock::Gdn(_)))
            .count() as u32;
        let arch = &self.arch;
        let kv_dim_elements = if n_attn_layers == 0 {
            0
        } else {
            arch.n_kv_heads * arch.attn_head_dim
        };
        let (kv_bytes_per_token, kv_storage_kind) = match kv_cache_dtype_for_arch(arch) {
            _ if n_attn_layers == 0 => (0, SnapshotKvStorageKind::None),
            GgmlType::F16 => (kv_dim_elements * 2, SnapshotKvStorageKind::F16),
            GgmlType::Q8_0 => {
                let (block, bytes) =
                    ggml_type_layout_raw(GgmlType::Q8_0 as u32).expect("Q8_0 layout is defined");
                assert!(u64::from(kv_dim_elements).is_multiple_of(block));
                (
                    u32::try_from(u64::from(kv_dim_elements) / block * bytes)
                        .expect("Q8_0 KV bytes per token fit u32"),
                    SnapshotKvStorageKind::Q8_0,
                )
            }
            _ => unreachable!("unsupported snapshot KV storage"),
        };
        let gdn_conv_elements_per_layer = if n_gdn_layers == 0 {
            0
        } else {
            arch.gdn_conv_kernel.saturating_sub(1)
                * (2 * arch.gdn_n_k_heads + arch.gdn_n_v_heads)
                * arch.gdn_head_dim
        };
        let gdn_state_elements_per_layer = if n_gdn_layers == 0 {
            0
        } else {
            arch.gdn_n_v_heads * arch.gdn_head_dim * arch.gdn_head_dim
        };
        SnapshotAbi {
            layout_version: SNAPSHOT_LAYOUT_VERSION,
            n_attn_layers,
            n_gdn_layers,
            kv_dim_elements,
            kv_bytes_per_token,
            kv_storage_kind,
            gdn_state_elements_per_layer,
            gdn_conv_elements_per_layer,
        }
    }

    /// Whether this model relies on a residency set attached to its load queue.
    /// Such a model cannot be submitted through unrelated command queues until
    /// those queues receive matching residency-set ownership and teardown.
    #[doc(hidden)]
    pub fn has_queue_scoped_residency_set(&self) -> bool {
        false
    }

    pub(crate) fn attach_residency_to_queue(
        &self,
        _queue: &Retained<ProtocolObject<dyn MTLCommandQueue>>,
    ) -> Option<MetalAdditionalQueueResidencySetGuard> {
        None
    }
}

pub(crate) struct MetalAdditionalQueueResidencySetGuard;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct MetalModelLoadOptions {
    pub auto_parallel_copy: bool,
    pub auto_retained_single_pass: bool,
    pub auxiliary_weight_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MetalLoadPrefetchAdvice {
    PreserveConfiguredPolicy,
    SuppressColdOnlyParallelDirectPread,
    SuppressForTightCopiedWeights,
}

#[derive(Clone, Copy, Debug)]
struct ResolvedStoragePolicy {
    no_copy_mode: GgufNoCopyMode,
    prefault_enabled: bool,
    parallel_mode: GgufParallelCopyMode,
}

#[derive(Clone, Copy, Debug)]
struct ResolvedWeightLoadChoices {
    embedding_selection: NativeQuantEmbeddingSelection,
    router_f16: bool,
    fused_qkv_g8: bool,
    dense_iq_native: bool,
}

impl ResolvedWeightLoadChoices {
    fn resolve(embedding_selection: NativeQuantEmbeddingSelection, router_f16: bool) -> Self {
        Self {
            embedding_selection,
            router_f16,
            fused_qkv_g8: prefill_attn_fused_qkv_g8_enabled(),
            dense_iq_native: dense_iq_native_storage_enabled(),
        }
    }

    fn storage_requests<'a>(
        &self,
        model: &Model<'a>,
    ) -> Result<Vec<ModelWeightStorageRequest<'a>>, MfError> {
        model_weight_storage_requests_with_policy(
            model,
            self.embedding_selection.uses_native(),
            self.router_f16,
            self.dense_iq_native,
        )
    }

    fn fused_qkv_priced_bytes(
        &self,
        ctx: &MetalContext,
        model: &Model<'_>,
    ) -> Result<u64, MfError> {
        if !self.fused_qkv_g8 {
            return Ok(0);
        }
        let logical_sizes = fused_qkv_derived_weight_sizes(model, self.fused_qkv_g8)?;
        logical_sizes
            .into_iter()
            .try_fold(0u64, |total, logical_bytes| {
                let priced = ctx
                    .shared_buffer_size_and_align(logical_bytes)
                    .map_err(MfError::Metal)?
                    .size;
                total.checked_add(priced).ok_or_else(|| {
                    MfError::LoadPolicy("fused QKV derived byte estimate overflow".into())
                })
            })
    }
}

pub fn fused_qkv_derived_weight_sizes(
    model: &Model<'_>,
    enabled: bool,
) -> Result<Vec<u64>, MfError> {
    let mut sizes = Vec::new();
    if !enabled {
        return Ok(sizes);
    }
    for block in &model.blocks {
        let Block::Attn(attn) = block else {
            continue;
        };
        if attn.q.dtype != GgmlType::Q8_0
            || attn.k.dtype != GgmlType::Q8_0
            || attn.v.dtype != GgmlType::Q8_0
            || attn.q.shape.len() != 2
            || attn.k.shape.len() != 2
            || attn.v.shape.len() != 2
            || attn.q.shape[0] != attn.k.shape[0]
            || attn.q.shape[0] != attn.v.shape[0]
        {
            continue;
        }
        let logical_bytes = attn
            .q
            .n_bytes
            .checked_add(attn.k.n_bytes)
            .and_then(|bytes| bytes.checked_add(attn.v.n_bytes))
            .ok_or_else(|| {
                MfError::LoadPolicy("fused QKV derived byte estimate overflow".into())
            })?;
        sizes.push(logical_bytes);
    }
    Ok(sizes)
}

pub fn fused_qkv_g8_enabled_for_runtime() -> bool {
    prefill_attn_fused_qkv_g8_enabled()
}

pub fn model_weight_storage_requests_for_runtime<'a>(
    _gguf: &GgufFile,
    model: &Model<'a>,
) -> Result<Vec<ModelWeightStorageRequest<'a>>, MfError> {
    let embedding_mode = native_quant_embedding_mode();
    let embedding_selection = resolve_native_quant_embedding(
        embedding_mode,
        native_quant_embedding_supported(model.token_embd.dtype, &model.token_embd.shape),
        native_quant_embedding_default_promoted(
            &model.arch,
            model.tied_embeddings,
            model.token_embd.dtype,
            &model.token_embd.shape,
        ),
    );
    ResolvedWeightLoadChoices::resolve(embedding_selection, moe_router_f16_enabled())
        .storage_requests(model)
}

pub fn runtime_retained_mapping_unsupported_reason(
    gguf: &GgufFile,
    model: &Model<'_>,
    max_buffer_length: usize,
) -> Option<&'static str> {
    let requests = model_weight_storage_requests_for_runtime(gguf, model).ok()?;
    let direct = requests
        .iter()
        .filter(|request| request.kind == ModelWeightStorageKind::Direct)
        .map(|request| request.desc)
        .collect::<Vec<_>>();
    let page_size = host_page_size_bytes().ok()?;
    retained_mapping_unsupported_reason(
        &gguf.shard_mapped_lengths(),
        &direct,
        page_size,
        max_buffer_length,
    )
}

struct PreparedParallelCopyPlan {
    plan: ParallelCopyLoadPlan,
    population: ParallelPopulationMethod,
    destination_length: ParallelDestinationLength,
    expected_identities: Vec<ModelWeightStorageIdentity>,
    sorted_request_indices: Vec<usize>,
}

enum PreparedAutoSelection {
    NotEligible,
    NoMatch,
    Selected(PreparedParallelCopyPlan),
}

struct DirectStorageConfiguration {
    no_copy_mode: GgufNoCopyMode,
    prefault_enabled: bool,
    parallel_mode: GgufParallelCopyMode,
    auto_parallel: PreparedAutoSelection,
    auto_retained: PreparedAutoRetainedSelection,
    embedding_selection: NativeQuantEmbeddingSelection,
}

enum PreparedAutoRetainedSelection {
    NotEligible,
    NoMatch(String),
    Selected(RetainedStoragePlan),
}

impl PreparedAutoSelection {
    fn prefetch_advice(&self) -> MetalLoadPrefetchAdvice {
        match self {
            Self::Selected(prepared)
                if prepared.population == ParallelPopulationMethod::Pread
                    && prepared.destination_length == ParallelDestinationLength::LogicalExact =>
            {
                MetalLoadPrefetchAdvice::SuppressColdOnlyParallelDirectPread
            }
            Self::NotEligible | Self::NoMatch | Self::Selected(_) => {
                MetalLoadPrefetchAdvice::PreserveConfiguredPolicy
            }
        }
    }
}

pub(crate) struct PreparedMetalModelLoad<'ctx, 'gguf, 'model> {
    ctx: &'ctx MetalContext,
    gguf: &'gguf GgufFile,
    model: &'model Model<'gguf>,
    storage: ResolvedStoragePolicy,
    choices: ResolvedWeightLoadChoices,
    expected: Vec<ModelWeightStorageRequest<'gguf>>,
    auto: PreparedAutoSelection,
    auto_retained: PreparedAutoRetainedSelection,
    tight_copy: bool,
}

impl PreparedMetalModelLoad<'_, '_, '_> {
    pub(crate) fn prefetch_advice(&self) -> MetalLoadPrefetchAdvice {
        if self.tight_copy {
            MetalLoadPrefetchAdvice::SuppressForTightCopiedWeights
        } else {
            self.auto.prefetch_advice()
        }
    }
}

pub enum MetalBlock {
    Gdn(MetalGdnBlock),
    Attn(MetalAttnBlock),
}

pub struct MetalGdnBlock {
    pub attn_norm: MetalTensor,
    pub post_attn_norm: MetalTensor,
    pub ffn_gate: MetalTensor,
    pub ffn_up: MetalTensor,
    pub ffn_down: MetalTensor,
    pub in_proj_qkv: MetalTensor,
    pub in_proj_z: MetalTensor,
    pub beta_proj: MetalTensor,
    pub alpha_proj: MetalTensor,
    pub a_log: MetalTensor,
    pub dt_bias: MetalTensor,
    pub conv1d: MetalTensor,
    pub norm: MetalTensor,
    pub out_proj: MetalTensor,
    pub ffn_moe: Option<MetalMoeFfn>,
}

pub struct MetalAttnBlock {
    pub attn_norm: MetalTensor,
    pub post_attn_norm: MetalTensor,
    pub ffn_gate: MetalTensor,
    pub ffn_up: MetalTensor,
    pub ffn_down: MetalTensor,
    pub q: MetalTensor, // outputs 2x q_dim — Q + gate
    pub k: MetalTensor,
    pub v: MetalTensor,
    pub qkv_fused: Option<MetalTensor>,
    pub o: MetalTensor,
    pub q_norm: MetalTensor,
    pub k_norm: MetalTensor,
    pub ffn_moe: Option<MetalMoeFfn>,
}

pub struct MetalMoeFfn {
    pub gate_inp: MetalTensor,
    pub gate_exps: MetalTensor,
    pub up_exps: MetalTensor,
    pub down_exps: MetalTensor,
    pub gate_inp_shexp: MetalTensor,
    pub gate_inp_cpu: Vec<f32>,
    pub gate_inp_shexp_cpu: Vec<f32>,
}

#[derive(Clone, Copy)]
enum MixerSlot {
    Gdn(usize),
    Attn(usize),
}

#[allow(dead_code)]
struct MoeRouteDecision {
    ranked: Vec<(usize, f32)>,
    shared_gate_scalar: f32,
}

#[derive(Clone, Debug)]
pub struct MoeRouteReplayRow {
    pub topk_idx: Vec<i32>,
    pub topk_weight: Vec<f32>,
    pub hidden: Vec<f32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SourceMaterialization {
    DirectCopy,
    DirectView,
    DirectAlias,
    TailFallback,
    ConvertedF32,
    ConvertedF16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelWeightStorageKind {
    Direct,
    ConvertedF32,
    ConvertedF16,
}

#[derive(Clone, Copy, Debug)]
pub struct ModelWeightStorageRequest<'a> {
    pub desc: &'a TensorDesc,
    pub kind: ModelWeightStorageKind,
    pub resident_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ModelWeightStorageIdentity {
    name: String,
    shard_idx: usize,
    data_offset: u64,
    source_bytes: u64,
    dtype: GgmlType,
    shape: Vec<u64>,
    kind: ModelWeightStorageKind,
    resident_bytes: u64,
}

#[derive(Default)]
struct WeightLoadLedger {
    source_descriptors: usize,
    source_bytes: u64,
    direct_copy_descriptors: usize,
    direct_copy_bytes: u64,
    direct_view_descriptors: usize,
    direct_view_bytes: u64,
    direct_alias_descriptors: usize,
    direct_alias_bytes: u64,
    tail_fallback_descriptors: usize,
    tail_fallback_bytes: u64,
    converted_descriptors: usize,
    converted_source_bytes: u64,
    converted_resident_bytes: u64,
    derived_allocations: usize,
    derived_bytes: u64,
    requests: Vec<ModelWeightStorageIdentity>,
}

impl WeightLoadLedger {
    fn record_source(
        &mut self,
        desc: &TensorDesc,
        materialization: SourceMaterialization,
        resident_bytes: u64,
    ) -> Result<(), MfError> {
        self.source_descriptors = self
            .source_descriptors
            .checked_add(1)
            .ok_or_else(|| MfError::LoadPolicy("source descriptor count overflow".to_string()))?;
        self.source_bytes = self
            .source_bytes
            .checked_add(desc.n_bytes)
            .ok_or_else(|| MfError::LoadPolicy("source byte ledger overflow".to_string()))?;
        let (count, bytes) = match materialization {
            SourceMaterialization::DirectCopy => (
                &mut self.direct_copy_descriptors,
                &mut self.direct_copy_bytes,
            ),
            SourceMaterialization::DirectView => (
                &mut self.direct_view_descriptors,
                &mut self.direct_view_bytes,
            ),
            SourceMaterialization::DirectAlias => (
                &mut self.direct_alias_descriptors,
                &mut self.direct_alias_bytes,
            ),
            SourceMaterialization::TailFallback => (
                &mut self.tail_fallback_descriptors,
                &mut self.tail_fallback_bytes,
            ),
            SourceMaterialization::ConvertedF32 | SourceMaterialization::ConvertedF16 => {
                self.converted_resident_bytes = self
                    .converted_resident_bytes
                    .checked_add(resident_bytes)
                    .ok_or_else(|| {
                        MfError::LoadPolicy("converted resident byte ledger overflow".to_string())
                    })?;
                (
                    &mut self.converted_descriptors,
                    &mut self.converted_source_bytes,
                )
            }
        };
        *count = count
            .checked_add(1)
            .ok_or_else(|| MfError::LoadPolicy("materialization count overflow".to_string()))?;
        *bytes = bytes
            .checked_add(desc.n_bytes)
            .ok_or_else(|| MfError::LoadPolicy("materialization byte overflow".to_string()))?;
        let kind = match materialization {
            SourceMaterialization::DirectCopy
            | SourceMaterialization::DirectView
            | SourceMaterialization::DirectAlias
            | SourceMaterialization::TailFallback => ModelWeightStorageKind::Direct,
            SourceMaterialization::ConvertedF32 => ModelWeightStorageKind::ConvertedF32,
            SourceMaterialization::ConvertedF16 => ModelWeightStorageKind::ConvertedF16,
        };
        self.requests.push(ModelWeightStorageIdentity {
            name: desc.name.clone(),
            shard_idx: desc.shard_idx,
            data_offset: desc.data_offset,
            source_bytes: desc.n_bytes,
            dtype: desc.dtype,
            shape: desc.shape.clone(),
            kind,
            resident_bytes,
        });
        Ok(())
    }

    fn record_derived(&mut self, tensor: &MetalTensor) -> Result<(), MfError> {
        self.derived_allocations = self
            .derived_allocations
            .checked_add(1)
            .ok_or_else(|| MfError::LoadPolicy("derived allocation count overflow".to_string()))?;
        self.derived_bytes = self
            .derived_bytes
            .checked_add(tensor.n_bytes())
            .ok_or_else(|| MfError::LoadPolicy("derived allocation byte overflow".to_string()))?;
        Ok(())
    }
}

struct PlannedRetainedStorage {
    plan: RetainedStoragePlan,
    windows: Vec<MetalGgufBacking>,
    realized: Vec<Option<MetalTensor>>,
    cursor: usize,
}

impl PlannedRetainedStorage {
    fn load_direct(
        &mut self,
        ctx: &MetalContext,
        gguf: &GgufFile,
        desc: &TensorDesc,
    ) -> Result<(MetalTensor, SourceMaterialization), MfError> {
        let entry = self.plan.entries.get(self.cursor).ok_or_else(|| {
            MfError::LoadPolicy(format!(
                "retained storage received unexpected direct tensor {:?} at index {}",
                desc.name, self.cursor
            ))
        })?;
        if entry.request_index != self.cursor
            || entry.name != desc.name
            || entry.shard_idx != desc.shard_idx
            || entry.data_offset != desc.data_offset
            || entry.n_bytes != desc.n_bytes
        {
            return Err(MfError::LoadPolicy(format!(
                "retained storage request drift at direct index {}: entry={entry:?} desc={desc:?}",
                self.cursor
            )));
        }
        let (tensor, materialization) = match entry.disposition {
            RetainedStorageDisposition::View {
                window_index,
                buffer_offset,
            } => {
                let backing = self.windows.get(window_index).ok_or_else(|| {
                    MfError::LoadPolicy(format!(
                        "retained storage window index {window_index} is missing"
                    ))
                })?;
                let (eligibility, tensor) = backing.tensor(desc)?;
                let tensor = match (eligibility, tensor) {
                    (GgufBackingEligibility::Eligible, Some(tensor)) => tensor,
                    (reason, _) => {
                        return Err(MfError::LoadPolicy(format!(
                            "planned retained tensor {:?} failed realization: {reason:?}",
                            desc.name
                        )));
                    }
                };
                if tensor.offset != buffer_offset {
                    return Err(MfError::LoadPolicy(format!(
                        "retained tensor {:?} offset drift: actual={} planned={buffer_offset}",
                        desc.name, tensor.offset
                    )));
                }
                (tensor, SourceMaterialization::DirectView)
            }
            RetainedStorageDisposition::Alias {
                source_request_index,
            } => {
                if source_request_index >= self.cursor {
                    return Err(MfError::LoadPolicy(format!(
                        "retained alias {:?} has non-prior source {source_request_index}",
                        desc.name
                    )));
                }
                let tensor = self
                    .realized
                    .get(source_request_index)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| {
                        MfError::LoadPolicy(format!(
                            "retained alias {:?} source {source_request_index} is unrealized",
                            desc.name
                        ))
                    })?
                    .clone();
                (tensor, SourceMaterialization::DirectAlias)
            }
            RetainedStorageDisposition::CopyFallback { reason } => {
                if reason != RetainedStorageFallback::FinalPartialPage {
                    return Err(MfError::LoadPolicy(format!(
                        "retained tensor {:?} has disallowed copy fallback {reason:?}",
                        desc.name
                    )));
                }
                let tensor = MetalTensor::from_gguf_tensor(ctx, desc, gguf.slice(desc))?;
                (tensor, SourceMaterialization::TailFallback)
            }
        };
        self.realized[self.cursor] = Some(tensor.clone());
        self.cursor += 1;
        Ok((tensor, materialization))
    }

    fn validate_complete(&self, ledger: &WeightLoadLedger) -> Result<(), MfError> {
        if self.cursor != self.plan.entries.len() || self.realized.iter().any(Option::is_none) {
            return Err(MfError::LoadPolicy(format!(
                "retained storage consumption mismatch: consumed={} planned={} realized={}",
                self.cursor,
                self.plan.entries.len(),
                self.realized
                    .iter()
                    .filter(|tensor| tensor.is_some())
                    .count()
            )));
        }
        let view_count = self
            .plan
            .entries
            .iter()
            .filter(|entry| matches!(entry.disposition, RetainedStorageDisposition::View { .. }))
            .count();
        let alias_count = self
            .plan
            .entries
            .iter()
            .filter(|entry| matches!(entry.disposition, RetainedStorageDisposition::Alias { .. }))
            .count();
        let fallback_count = self
            .plan
            .entries
            .iter()
            .filter(|entry| {
                matches!(
                    entry.disposition,
                    RetainedStorageDisposition::CopyFallback { .. }
                )
            })
            .count();
        if ledger.direct_copy_descriptors != 0
            || ledger.direct_view_descriptors != view_count
            || ledger.direct_view_bytes != self.plan.unique_view_bytes
            || ledger.direct_alias_descriptors != alias_count
            || ledger.direct_alias_bytes != self.plan.alias_bytes
            || ledger.tail_fallback_descriptors != fallback_count
            || ledger.tail_fallback_bytes != self.plan.unique_fallback_bytes
        {
            return Err(MfError::LoadPolicy(format!(
                concat!(
                    "retained storage ledger mismatch: copy={}/{} view={}/{} ",
                    "alias={}/{} fallback={}/{} planned_view={}/{} ",
                    "planned_alias={}/{} planned_fallback={}/{}"
                ),
                ledger.direct_copy_descriptors,
                ledger.direct_copy_bytes,
                ledger.direct_view_descriptors,
                ledger.direct_view_bytes,
                ledger.direct_alias_descriptors,
                ledger.direct_alias_bytes,
                ledger.tail_fallback_descriptors,
                ledger.tail_fallback_bytes,
                view_count,
                self.plan.unique_view_bytes,
                alias_count,
                self.plan.alias_bytes,
                fallback_count,
                self.plan.unique_fallback_bytes,
            )));
        }
        Ok(())
    }
}

struct ParallelCopyTask<'a> {
    source: &'a [u8],
    destination: &'a mut [u8],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ParallelPopulationMethod {
    MmapCopy,
    Pread,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ParallelDestinationLength {
    LogicalExact,
    PageRounded,
}

struct ParallelPreadTask<'a> {
    shard_idx: usize,
    source_offset: u64,
    destination: &'a mut [u8],
}

enum ParallelPopulationTask<'a> {
    MmapCopy(ParallelCopyTask<'a>),
    Pread(ParallelPreadTask<'a>),
}

#[derive(Clone, Copy, Debug)]
struct ParallelCopyUsage {
    minor_faults: i64,
    major_faults: i64,
    user_time_us: i64,
    system_time_us: i64,
}

#[derive(Clone, Copy, Debug)]
struct ParallelCopyProcUsage {
    instructions: u64,
    cycles: u64,
}

#[derive(Clone, Copy, Debug)]
struct ParallelCopyEndpointAccounting {
    user_cpu_us: u64,
    system_cpu_us: u64,
    total_cpu_us: u64,
    timer_minor_faults: u64,
    timer_major_faults: u64,
    instructions_delta_raw: u64,
    cycles_delta_raw: u64,
}

#[derive(Clone, Copy, Debug)]
struct ParallelCopyTiming {
    allocation_us: u64,
    source_us: u64,
    copy_us: u64,
    binding_us: u64,
    ready_us: u64,
}

struct PlannedParallelCopiedStorage {
    plan: ParallelCopyLoadPlan,
    destination_length: ParallelDestinationLength,
    expected: Vec<ModelWeightStorageIdentity>,
    sorted_request_indices: Vec<usize>,
    resources: Vec<Buffer>,
    tensors: Vec<MetalTensor>,
    cursor: usize,
}

impl PlannedParallelCopiedStorage {
    fn load_direct(
        &mut self,
        desc: &TensorDesc,
    ) -> Result<(MetalTensor, SourceMaterialization), MfError> {
        let expected = self.expected.get(self.cursor).ok_or_else(|| {
            MfError::LoadPolicy(format!(
                "parallel-copy storage received unexpected tensor {:?} at index {}",
                desc.name, self.cursor
            ))
        })?;
        let actual = ModelWeightStorageIdentity {
            name: desc.name.clone(),
            shard_idx: desc.shard_idx,
            data_offset: desc.data_offset,
            source_bytes: desc.n_bytes,
            dtype: desc.dtype,
            shape: desc.shape.clone(),
            kind: ModelWeightStorageKind::Direct,
            resident_bytes: desc.n_bytes,
        };
        if actual != *expected {
            return Err(MfError::LoadPolicy(format!(
                "parallel-copy request drift at index {}: actual={actual:?} expected={expected:?}",
                self.cursor
            )));
        }
        let tensor = self.tensors.get(self.cursor).ok_or_else(|| {
            MfError::LoadPolicy(format!("parallel-copy tensor {} is missing", self.cursor))
        })?;
        if Retained::as_ptr(&tensor.buffer) != Retained::as_ptr(&self.resources[self.cursor])
            || tensor.offset != 0
            || tensor.n_bytes() != desc.n_bytes
            || tensor.dtype != desc.dtype
            || tensor.shape != desc.shape
            || tensor.provenance() != MetalTensorProvenance::OwnedWeightReadOnly
        {
            return Err(MfError::LoadPolicy(format!(
                "parallel-copy tensor {} changed before consumption",
                self.cursor
            )));
        }
        self.cursor += 1;
        Ok((tensor.clone(), SourceMaterialization::DirectCopy))
    }

    fn validate_complete(&self, ledger: &WeightLoadLedger) -> Result<(), MfError> {
        if self.cursor != self.expected.len() {
            return Err(MfError::LoadPolicy(format!(
                "parallel-copy storage consumption mismatch: consumed={} expected={}",
                self.cursor,
                self.expected.len()
            )));
        }
        let schedule = frozen_parallel_copy_order(&self.plan, &self.expected)?;
        if schedule != self.sorted_request_indices {
            return Err(MfError::LoadPolicy(
                "parallel-copy schedule attestation drifted".to_string(),
            ));
        }
        validate_parallel_copied_topology(
            &self.plan,
            self.destination_length,
            &self.expected,
            &self.resources,
            &self.tensors,
        )?;
        if ledger.source_descriptors != self.plan.request_count
            || ledger.source_bytes != self.plan.source_bytes
            || ledger.direct_copy_descriptors != self.plan.request_count
            || ledger.direct_copy_bytes != self.plan.source_bytes
            || ledger.direct_view_descriptors != 0
            || ledger.direct_view_bytes != 0
            || ledger.direct_alias_descriptors != 0
            || ledger.direct_alias_bytes != 0
            || ledger.tail_fallback_descriptors != 0
            || ledger.tail_fallback_bytes != 0
            || ledger.converted_descriptors != 0
            || ledger.converted_source_bytes != 0
            || ledger.converted_resident_bytes != 0
            || ledger.derived_allocations != 0
            || ledger.derived_bytes != 0
        {
            return Err(MfError::LoadPolicy(format!(
                concat!(
                    "parallel-copy logical ledger drift: source={}/{} copy={}/{} ",
                    "view={}/{} alias={}/{} tail={}/{} converted={}/{}/{} derived={}/{}"
                ),
                ledger.source_descriptors,
                ledger.source_bytes,
                ledger.direct_copy_descriptors,
                ledger.direct_copy_bytes,
                ledger.direct_view_descriptors,
                ledger.direct_view_bytes,
                ledger.direct_alias_descriptors,
                ledger.direct_alias_bytes,
                ledger.tail_fallback_descriptors,
                ledger.tail_fallback_bytes,
                ledger.converted_descriptors,
                ledger.converted_source_bytes,
                ledger.converted_resident_bytes,
                ledger.derived_allocations,
                ledger.derived_bytes,
            )));
        }
        Ok(())
    }

    #[cfg(test)]
    fn validate_source_bytes(
        &self,
        gguf: &GgufFile,
        expected: &[ModelWeightStorageRequest<'_>],
    ) -> Result<(), MfError> {
        if expected.len() != self.expected.len() {
            return Err(MfError::LoadPolicy(
                "parallel-copy byte audit request count drifted".to_string(),
            ));
        }
        for (index, ((identity, buffer), request)) in self
            .expected
            .iter()
            .zip(&self.resources)
            .zip(expected)
            .enumerate()
        {
            if expected_model_weight_identity(request) != *identity {
                return Err(MfError::LoadPolicy(format!(
                    "parallel-copy byte audit identity {index} drifted"
                )));
            }
            let source = gguf.try_slice(request.desc).map_err(|error| {
                MfError::LoadPolicy(format!("parallel-copy byte audit source {index}: {error}"))
            })?;
            // SAFETY: the resource is CPU-accessible and retained by self. The
            // logical prefix has the exact source length, and no mutable CPU
            // access exists after construction. Physical padding is excluded.
            let actual = unsafe {
                std::slice::from_raw_parts(buffer.contents().as_ptr().cast::<u8>(), source.len())
            };
            if actual != source {
                return Err(MfError::LoadPolicy(format!(
                    "parallel-copy byte audit resource {index} differs from source"
                )));
            }
        }
        Ok(())
    }
}

enum DirectStorage {
    Copied,
    ForcedPlanned(PlannedRetainedStorage),
    ForcedParallelCopied(PlannedParallelCopiedStorage),
}

struct MetalWeightLoader<'a> {
    ctx: &'a MetalContext,
    gguf: &'a GgufFile,
    direct_storage: DirectStorage,
    choices: ResolvedWeightLoadChoices,
    ledger: WeightLoadLedger,
}

impl<'a> MetalWeightLoader<'a> {
    fn new(
        ctx: &'a MetalContext,
        gguf: &'a GgufFile,
        direct_storage: DirectStorage,
        choices: ResolvedWeightLoadChoices,
    ) -> Self {
        Self {
            ctx,
            gguf,
            direct_storage,
            choices,
            ledger: WeightLoadLedger::default(),
        }
    }

    fn record_source(
        &mut self,
        desc: &TensorDesc,
        materialization: SourceMaterialization,
        resident_bytes: u64,
    ) -> Result<(), MfError> {
        self.ledger
            .record_source(desc, materialization, resident_bytes)
    }

    fn load_direct(&mut self, desc: &TensorDesc) -> Result<MetalTensor, MfError> {
        let (tensor, materialization) = match &mut self.direct_storage {
            DirectStorage::Copied => {
                let tensor = MetalTensor::from_gguf_tensor(self.ctx, desc, self.gguf.slice(desc))?;
                (tensor, SourceMaterialization::DirectCopy)
            }
            DirectStorage::ForcedPlanned(storage) => {
                storage.load_direct(self.ctx, self.gguf, desc)?
            }
            DirectStorage::ForcedParallelCopied(storage) => storage.load_direct(desc)?,
        };
        self.record_source(desc, materialization, tensor.n_bytes())?;
        Ok(tensor)
    }

    fn load_f32(&mut self, desc: &TensorDesc) -> Result<MetalTensor, MfError> {
        if desc.dtype == GgmlType::F32 {
            return self.load_direct(desc);
        }
        let tensor = MetalTensor::zeros_f32(self.ctx, desc.shape.clone())?;
        let elements = usize::try_from(tensor.n_elements()).map_err(|_| {
            MfError::LoadPolicy(format!(
                "converted F32 tensor {:?} element count exceeds usize",
                desc.name
            ))
        })?;
        let bytes = elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                MfError::LoadPolicy(format!(
                    "converted F32 tensor {:?} byte count overflows usize",
                    desc.name
                ))
            })?;
        if tensor.offset != 0 || bytes > tensor.buffer.length() {
            return Err(MfError::LoadPolicy(format!(
                "converted F32 tensor {:?} destination range is invalid",
                desc.name
            )));
        }
        let start = tensor.buffer.contents().as_ptr() as *mut u8;
        if !(start as usize).is_multiple_of(std::mem::align_of::<f32>()) {
            return Err(MfError::LoadPolicy(format!(
                "converted F32 tensor {:?} destination is not f32-aligned",
                desc.name
            )));
        }
        let output = unsafe {
            std::slice::from_raw_parts_mut(start.cast::<std::mem::MaybeUninit<f32>>(), elements)
        };
        crate::codec::dequant_to_f32_into(desc, self.gguf.slice(desc), output)?;
        self.record_source(desc, SourceMaterialization::ConvertedF32, tensor.n_bytes())?;
        Ok(tensor)
    }

    fn load_weight(&mut self, role: WeightRole, desc: &TensorDesc) -> Result<MetalTensor, MfError> {
        if weight_storage_kind_with_policy(role, desc, self.choices.dense_iq_native)?
            == ModelWeightStorageKind::Direct
        {
            return self.load_direct(desc);
        }
        tracing::info!(
            target: "qwen_diag",
            "[metal-load] {} is {:?}; dequanting to F32 (no active native path)",
            desc.name, desc.dtype,
        );
        self.load_f32(desc)
    }

    fn load_embedding(
        &mut self,
        desc: &TensorDesc,
        native_quant: bool,
    ) -> Result<MetalTensor, MfError> {
        if native_quant && desc.dtype == GgmlType::IQ1_M {
            validate_native_iq_matrix_storage(desc)?;
        }
        if matches!(desc.dtype, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16) || native_quant {
            self.load_direct(desc)
        } else {
            self.load_f32(desc)
        }
    }

    fn load_router_weight(&mut self, desc: &TensorDesc) -> Result<MetalTensor, MfError> {
        if !self.choices.router_f16 {
            return self.load_f32(desc);
        }
        let f32 = crate::codec::dequant_to_f32(desc, self.gguf.slice(desc))?;
        let f16: Vec<half::f16> = f32.iter().copied().map(half::f16::from_f32).collect();
        let tensor = MetalTensor::from_bytes(
            self.ctx,
            bytemuck::cast_slice(&f16),
            desc.shape.clone(),
            GgmlType::F16,
        )?;
        self.record_source(desc, SourceMaterialization::ConvertedF16, tensor.n_bytes())?;
        Ok(tensor)
    }

    fn load_moe_expert(&mut self, desc: &TensorDesc) -> Result<MetalTensor, MfError> {
        if matches!(desc.dtype, GgmlType::IQ3_XXS | GgmlType::IQ3_S)
            && moe_iq3_expert_native_enabled(desc)
        {
            self.load_direct(desc)
        } else {
            self.load_weight(WeightRole::ExpertBank, desc)
        }
    }

    fn load_moe(&mut self, moe: &MoeFfn<'_>) -> Result<MetalMoeFfn, MfError> {
        Ok(MetalMoeFfn {
            gate_inp: self.load_router_weight(moe.gate_inp)?,
            gate_exps: self.load_moe_expert(moe.gate_exps)?,
            up_exps: self.load_moe_expert(moe.up_exps)?,
            down_exps: self.load_weight(WeightRole::ExpertBank, moe.down_exps)?,
            gate_inp_shexp: self.load_f32(moe.gate_inp_shexp)?,
            gate_inp_cpu: crate::codec::dequant_to_f32(
                moe.gate_inp,
                self.gguf.slice(moe.gate_inp),
            )?,
            gate_inp_shexp_cpu: crate::codec::dequant_to_f32(
                moe.gate_inp_shexp,
                self.gguf.slice(moe.gate_inp_shexp),
            )?,
        })
    }

    fn record_derived(&mut self, tensor: Option<&MetalTensor>) -> Result<(), MfError> {
        if let Some(tensor) = tensor {
            self.ledger.record_derived(tensor)?;
        }
        Ok(())
    }

    fn finish(self, expected: &[ModelWeightStorageRequest<'_>]) -> Result<(), MfError> {
        if let DirectStorage::ForcedPlanned(storage) = &self.direct_storage {
            storage.validate_complete(&self.ledger)?;
        }
        if let DirectStorage::ForcedParallelCopied(storage) = &self.direct_storage {
            storage.validate_complete(&self.ledger)?;
        }
        let ledger = self.ledger;
        validate_model_weight_request_sequence(&ledger.requests, expected)?;
        let accounted_source_bytes = ledger
            .direct_copy_bytes
            .checked_add(ledger.direct_view_bytes)
            .and_then(|bytes| bytes.checked_add(ledger.direct_alias_bytes))
            .and_then(|bytes| bytes.checked_add(ledger.tail_fallback_bytes))
            .and_then(|bytes| bytes.checked_add(ledger.converted_source_bytes))
            .ok_or_else(|| MfError::LoadPolicy("source accounting overflow".to_string()))?;
        if accounted_source_bytes != ledger.source_bytes {
            return Err(MfError::LoadPolicy(format!(
                "source accounting mismatch: categories={accounted_source_bytes} total={}",
                ledger.source_bytes
            )));
        }
        let expected_source_bytes = expected.iter().try_fold(0u64, |total, request| {
            total
                .checked_add(request.desc.n_bytes)
                .ok_or_else(|| MfError::LoadPolicy("expected source byte overflow".to_string()))
        })?;
        let expected_direct = expected
            .iter()
            .filter(|request| request.kind == ModelWeightStorageKind::Direct)
            .collect::<Vec<_>>();
        let expected_direct_bytes = expected_direct.iter().try_fold(0u64, |total, request| {
            total
                .checked_add(request.desc.n_bytes)
                .ok_or_else(|| MfError::LoadPolicy("expected direct byte overflow".to_string()))
        })?;
        let expected_converted = expected
            .iter()
            .filter(|request| request.kind != ModelWeightStorageKind::Direct)
            .collect::<Vec<_>>();
        let expected_converted_source_bytes =
            expected_converted.iter().try_fold(0u64, |total, request| {
                total.checked_add(request.desc.n_bytes).ok_or_else(|| {
                    MfError::LoadPolicy("expected converted source byte overflow".to_string())
                })
            })?;
        let expected_converted_resident_bytes =
            expected_converted.iter().try_fold(0u64, |total, request| {
                total.checked_add(request.resident_bytes).ok_or_else(|| {
                    MfError::LoadPolicy("expected converted resident byte overflow".to_string())
                })
            })?;
        let actual_direct_count = ledger
            .direct_copy_descriptors
            .checked_add(ledger.direct_view_descriptors)
            .and_then(|count| count.checked_add(ledger.direct_alias_descriptors))
            .and_then(|count| count.checked_add(ledger.tail_fallback_descriptors))
            .ok_or_else(|| MfError::LoadPolicy("direct descriptor overflow".to_string()))?;
        let actual_direct_bytes = ledger
            .direct_copy_bytes
            .checked_add(ledger.direct_view_bytes)
            .and_then(|bytes| bytes.checked_add(ledger.direct_alias_bytes))
            .and_then(|bytes| bytes.checked_add(ledger.tail_fallback_bytes))
            .ok_or_else(|| MfError::LoadPolicy("direct byte overflow".to_string()))?;
        if ledger.source_descriptors != expected.len()
            || ledger.source_bytes != expected_source_bytes
            || actual_direct_count != expected_direct.len()
            || actual_direct_bytes != expected_direct_bytes
            || ledger.converted_descriptors != expected_converted.len()
            || ledger.converted_source_bytes != expected_converted_source_bytes
            || ledger.converted_resident_bytes != expected_converted_resident_bytes
        {
            return Err(MfError::LoadPolicy(format!(
                concat!(
                    "model storage request drift: source={}/{} bytes={}/{} ",
                    "direct={}/{} bytes={}/{} converted={}/{} source_bytes={}/{} ",
                    "resident_bytes={}/{}"
                ),
                ledger.source_descriptors,
                expected.len(),
                ledger.source_bytes,
                expected_source_bytes,
                actual_direct_count,
                expected_direct.len(),
                actual_direct_bytes,
                expected_direct_bytes,
                ledger.converted_descriptors,
                expected_converted.len(),
                ledger.converted_source_bytes,
                expected_converted_source_bytes,
                ledger.converted_resident_bytes,
                expected_converted_resident_bytes,
            )));
        }
        emit_metal_load_line(format_args!(
            concat!(
                "[metal-load-ledger] source={}/{} direct_copy={}/{} direct_view={}/{} ",
                "direct_alias={}/{} tail_fallback={}/{} converted={}/{}/{} derived={}/{}"
            ),
            ledger.source_descriptors,
            ledger.source_bytes,
            ledger.direct_copy_descriptors,
            ledger.direct_copy_bytes,
            ledger.direct_view_descriptors,
            ledger.direct_view_bytes,
            ledger.direct_alias_descriptors,
            ledger.direct_alias_bytes,
            ledger.tail_fallback_descriptors,
            ledger.tail_fallback_bytes,
            ledger.converted_descriptors,
            ledger.converted_source_bytes,
            ledger.converted_resident_bytes,
            ledger.derived_allocations,
            ledger.derived_bytes,
        ));
        Ok(())
    }
}

impl MetalModel {
    /// Load weights from an `loader::Model` view. Native-quant path:
    /// keeps weight tensors at their on-disk dtype (Q4_K, Q6_K, F32,
    /// etc.) and the kernel dispatchers pick the right `encode_mat_vec_*`
    /// based on dtype.
    ///
    /// For weights that aren't matmul'd by a quant-supporting kernel
    /// (e.g. norms, ssm_a, dt_bias — they need F32 for the elementwise
    /// kernels), we dequant via the codec at load time. The big tensors
    /// (mat_vec inputs and lm_head) keep their native dtype. Q4_K/Q6_K/Q8_0
    /// embeddings can opt into native residency once their row kernels apply.
    pub fn load(ctx: &MetalContext, gguf: &GgufFile, model: &Model<'_>) -> Result<Self, MfError> {
        Self::load_with_options(ctx, gguf, model, MetalModelLoadOptions::default())
    }

    pub fn load_with_additional_metal_weight_bytes(
        ctx: &MetalContext,
        gguf: &GgufFile,
        model: &Model<'_>,
        auxiliary_weight_bytes: u64,
    ) -> Result<Self, MfError> {
        Self::load_with_options(
            ctx,
            gguf,
            model,
            MetalModelLoadOptions {
                auxiliary_weight_bytes,
                ..MetalModelLoadOptions::default()
            },
        )
    }

    pub(crate) fn load_with_options(
        ctx: &MetalContext,
        gguf: &GgufFile,
        model: &Model<'_>,
        options: MetalModelLoadOptions,
    ) -> Result<Self, MfError> {
        Self::load_prepared(Self::prepare_load_with_options(ctx, gguf, model, options)?)
    }

    pub(crate) fn prepare_load_with_options<'ctx, 'gguf, 'model>(
        ctx: &'ctx MetalContext,
        gguf: &'gguf GgufFile,
        model: &'model Model<'gguf>,
        options: MetalModelLoadOptions,
    ) -> Result<PreparedMetalModelLoad<'ctx, 'gguf, 'model>, MfError> {
        let storage_configuration = resolve_gguf_storage_load_configuration()?;
        let no_copy_mode = storage_configuration.no_copy_mode;
        let parallel_mode = storage_configuration.parallel_mode;
        let prefault_mode = storage_configuration.prefault_mode;
        let prefault_enabled = prefault_mode != GgufNoCopyPrefaultMode::Disabled;
        let explicit_override_present =
            auto_parallel_copy_override_present(|name| std::env::var_os(name).is_some());
        let auto_parallel_copy = auto_parallel_copy_enabled(
            options.auto_parallel_copy,
            parallel_mode,
            explicit_override_present,
        );
        let embedding_mode = native_quant_embedding_mode();
        let embedding_selection = resolve_native_quant_embedding(
            embedding_mode,
            native_quant_embedding_supported(model.token_embd.dtype, &model.token_embd.shape),
            native_quant_embedding_default_promoted(
                &model.arch,
                model.tied_embeddings,
                model.token_embd.dtype,
                &model.token_embd.shape,
            ),
        );
        emit_native_quant_embedding_policy(model, embedding_selection);
        let router_f16 = moe_router_f16_enabled();
        let choices = ResolvedWeightLoadChoices::resolve(embedding_selection, router_f16);
        let expected = choices.storage_requests(model)?;
        let direct_descriptors = expected
            .iter()
            .filter(|request| request.kind == ModelWeightStorageKind::Direct)
            .map(|request| request.desc)
            .collect::<Vec<_>>();
        let mapping_unsupported_reason = if !ctx.device.hasUnifiedMemory() {
            Some("automatic_mapping_unsupported_unified_memory_required")
        } else {
            match host_page_size_bytes() {
                Ok(page_size) => retained_mapping_unsupported_reason(
                    &gguf.shard_mapped_lengths(),
                    &direct_descriptors,
                    page_size,
                    ctx.device.maxBufferLength(),
                ),
                Err(_) => Some("automatic_mapping_unsupported_host_page_geometry"),
            }
        };
        let copied_bytes_estimate =
            expected
                .iter()
                .try_fold(0u64, |total, request| -> Result<u64, MfError> {
                    let priced = ctx
                        .shared_buffer_size_and_align(request.resident_bytes)
                        .map_err(MfError::Metal)?
                        .size;
                    total.checked_add(priced).ok_or_else(|| {
                        MfError::LoadPolicy("copied weight byte estimate overflow".into())
                    })
                })?;
        let derived_bytes = choices.fused_qkv_priced_bytes(ctx, model)?;
        let copied_bytes_estimate = copied_bytes_estimate
            .checked_add(derived_bytes)
            .ok_or_else(|| MfError::LoadPolicy("copied weight byte estimate overflow".into()))?;
        let signals = ctx.memory_signals();
        let automatic_copy_decision = decide_gguf_storage_mode_with_signals(
            copied_bytes_estimate,
            options.auxiliary_weight_bytes,
            None,
            signals,
            ctx.device.hasUnifiedMemory(),
            GgufStorageRequest::Automatic,
            mapping_unsupported_reason,
        );
        let auto_copy_capacity_admitted = automatic_parallel_copy_capacity_admitted(
            no_copy_mode,
            parallel_mode,
            automatic_copy_decision.capacity_would_map,
        );
        let mut auto_plan_host_facts_declined = false;
        let auto = if auto_parallel_copy && auto_copy_capacity_admitted {
            match select_auto_parallel_copy_plan(ctx, gguf, model, &expected, embedding_selection)?
            {
                AutoParallelCopyLoadPlanSelection::Selected(plan) => {
                    let plan_label = "descriptor-plan-v1";
                    match auto_parallel_copy_population() {
                        Some(population) => match prepare_parallel_copy_plan(
                            ctx,
                            &expected,
                            plan,
                            population,
                            ParallelDestinationLength::LogicalExact,
                        ) {
                            Ok(prepared) => {
                                emit_metal_load_line(format_args!(
                                    "[metal-gguf-parallel-policy] mode=auto plan={}",
                                    plan_label
                                ));
                                PreparedAutoSelection::Selected(prepared)
                            }
                            Err(error) => {
                                auto_plan_host_facts_declined = true;
                                emit_metal_load_line(format_args!(
                                    "[metal-gguf-parallel-policy] mode=auto plan={} declined reason=host_plan_facts details={error}",
                                    plan_label
                                ));
                                PreparedAutoSelection::NoMatch
                            }
                        },
                        None => PreparedAutoSelection::NoMatch,
                    }
                }
                AutoParallelCopyLoadPlanSelection::HostPlanUnsupported => {
                    auto_plan_host_facts_declined = true;
                    emit_metal_load_line(format_args!(
                        "[metal-gguf-parallel-policy] mode=auto plan={} declined reason=host_plan_facts max_buffer_length={} host_page_size={:?}",
                        "descriptor-plan-v1",
                        ctx.max_buffer_length(),
                        host_page_size_bytes().ok(),
                    ));
                    PreparedAutoSelection::NoMatch
                }
                AutoParallelCopyLoadPlanSelection::NoMatch => PreparedAutoSelection::NoMatch,
            }
        } else if auto_parallel_copy {
            emit_metal_load_line(format_args!(
                "[metal-gguf-parallel-policy] mode=auto plan={} declined reason={} admission={}",
                "descriptor-plan-v1",
                automatic_copy_decision.reason,
                if automatic_copy_decision.capacity_would_map {
                    "insufficient"
                } else {
                    "admitted"
                },
            ));
            PreparedAutoSelection::NoMatch
        } else {
            PreparedAutoSelection::NotEligible
        };
        let auto_retained_eligible = auto_retained_single_pass_enabled(
            options.auto_retained_single_pass,
            ctx.device.hasUnifiedMemory(),
            no_copy_mode,
            parallel_mode,
            prefault_mode,
            explicit_override_present,
        );
        let auto_retained = prepare_auto_retained_selection(
            matches!(&auto, PreparedAutoSelection::Selected(_)),
            auto_retained_eligible,
            || retained_storage_plan_for_load(ctx, gguf, &expected),
        );
        let copy_reason = if parallel_mode.is_forced() {
            Some("explicit_parallel_copy_precedence")
        } else if matches!(&auto, PreparedAutoSelection::Selected(_)) {
            Some("planned_parallel_copy_precedence")
        } else {
            None
        };
        let request = match no_copy_mode {
            GgufNoCopyMode::Forced => GgufStorageRequest::Map("explicit_environment_override"),
            GgufNoCopyMode::Disabled => GgufStorageRequest::Copy("explicit_environment_override"),
            GgufNoCopyMode::Auto => {
                if let Some(reason) = copy_reason {
                    GgufStorageRequest::Copy(reason)
                } else if auto_plan_host_facts_declined && mapping_unsupported_reason.is_none() {
                    GgufStorageRequest::Map("automatic_parallel_copy_host_plan_unsupported")
                } else if matches!(&auto_retained, PreparedAutoRetainedSelection::Selected(_)) {
                    GgufStorageRequest::Map("automatic_retained_storage_precedence")
                } else {
                    GgufStorageRequest::Automatic
                }
            }
        };
        let decision = decide_gguf_storage_mode_with_signals(
            copied_bytes_estimate,
            options.auxiliary_weight_bytes,
            None,
            signals,
            ctx.device.hasUnifiedMemory(),
            request,
            mapping_unsupported_reason,
        );
        let maps_weights = decision.map_weights;
        let resolved_no_copy_mode = if maps_weights {
            GgufNoCopyMode::Forced
        } else {
            GgufNoCopyMode::Disabled
        };
        emit_metal_load_line(format_args!(
            "{}",
            gguf_storage_log_line(
                if maps_weights { "mapped" } else { "copied" },
                decision
                    .copied_bytes
                    .saturating_add(decision.auxiliary_weight_bytes),
                signals.recommended_max_bytes,
                decision.reason,
            )
        ));
        match &auto_retained {
            PreparedAutoRetainedSelection::Selected(plan) => emit_metal_load_line(format_args!(
                concat!(
                    "[metal-gguf-retained-policy] mode=auto action=selected ",
                    "windows={} view_bytes={} fallback_bytes={} prefault=disabled"
                ),
                plan.windows.len(),
                plan.unique_view_bytes,
                plan.unique_fallback_bytes,
            )),
            PreparedAutoRetainedSelection::NoMatch(reason) => emit_metal_load_line(format_args!(
                "[metal-gguf-retained-policy] mode=auto action=fallback reason={reason}"
            )),
            PreparedAutoRetainedSelection::NotEligible => {}
        }
        Ok(PreparedMetalModelLoad {
            ctx,
            gguf,
            model,
            storage: ResolvedStoragePolicy {
                no_copy_mode: resolved_no_copy_mode,
                prefault_enabled,
                parallel_mode,
            },
            choices,
            expected,
            auto,
            auto_retained,
            tight_copy: decision.tight_copy,
        })
    }

    pub(crate) fn load_prepared(
        prepared: PreparedMetalModelLoad<'_, '_, '_>,
    ) -> Result<Self, MfError> {
        let PreparedMetalModelLoad {
            ctx,
            gguf,
            model,
            storage,
            choices,
            expected,
            auto,
            auto_retained,
            ..
        } = prepared;
        let direct_storage = direct_storage_for_load(
            ctx,
            gguf,
            model,
            &expected,
            DirectStorageConfiguration {
                no_copy_mode: storage.no_copy_mode,
                prefault_enabled: storage.prefault_enabled,
                parallel_mode: storage.parallel_mode,
                auto_parallel: auto,
                auto_retained,
                embedding_selection: choices.embedding_selection,
            },
        )?;
        Self::load_with_direct_storage(ctx, gguf, model, choices, &expected, direct_storage)
    }

    #[cfg(test)]
    fn load_with_no_copy_policy(
        ctx: &MetalContext,
        gguf: &GgufFile,
        model: &Model<'_>,
        no_copy_mode: GgufNoCopyMode,
        prefault_enabled: bool,
    ) -> Result<Self, MfError> {
        Self::load_with_storage_policy(
            ctx,
            gguf,
            model,
            no_copy_mode,
            prefault_enabled,
            GgufParallelCopyMode::Disabled,
            false,
        )
    }

    #[cfg(test)]
    fn load_with_storage_policy(
        ctx: &MetalContext,
        gguf: &GgufFile,
        model: &Model<'_>,
        no_copy_mode: GgufNoCopyMode,
        prefault_enabled: bool,
        parallel_mode: GgufParallelCopyMode,
        auto_parallel_copy: bool,
    ) -> Result<Self, MfError> {
        if auto_parallel_copy {
            return Err(MfError::LoadPolicy(
                "test storage-policy helper cannot select Auto parallel copy".to_string(),
            ));
        }
        let embedding_mode = native_quant_embedding_mode();
        validate_forced_storage_embedding_policy(parallel_mode, embedding_mode)?;
        let embedding_selection = resolve_native_quant_embedding(
            embedding_mode,
            native_quant_embedding_supported(model.token_embd.dtype, &model.token_embd.shape),
            native_quant_embedding_default_promoted(
                &model.arch,
                model.tied_embeddings,
                model.token_embd.dtype,
                &model.token_embd.shape,
            ),
        );
        emit_native_quant_embedding_policy(model, embedding_selection);
        let router_f16 = moe_router_f16_enabled();
        let choices = ResolvedWeightLoadChoices::resolve(embedding_selection, router_f16);
        let expected_storage_requests = choices.storage_requests(model)?;
        let direct_storage = direct_storage_for_load(
            ctx,
            gguf,
            model,
            &expected_storage_requests,
            DirectStorageConfiguration {
                no_copy_mode,
                prefault_enabled,
                parallel_mode,
                auto_parallel: PreparedAutoSelection::NotEligible,
                auto_retained: PreparedAutoRetainedSelection::NotEligible,
                embedding_selection,
            },
        )?;
        Self::load_with_direct_storage(
            ctx,
            gguf,
            model,
            choices,
            &expected_storage_requests,
            direct_storage,
        )
    }

    fn load_with_direct_storage(
        ctx: &MetalContext,
        gguf: &GgufFile,
        model: &Model<'_>,
        choices: ResolvedWeightLoadChoices,
        expected_storage_requests: &[ModelWeightStorageRequest<'_>],
        direct_storage: DirectStorage,
    ) -> Result<Self, MfError> {
        let mut loader = MetalWeightLoader::new(ctx, gguf, direct_storage, choices);
        let token_embd =
            loader.load_embedding(model.token_embd, choices.embedding_selection.uses_native())?;
        let output_norm = loader.load_f32(model.output_norm)?;
        let lm_head = loader.load_weight(WeightRole::OutputHead, model.lm_head)?;

        let load_attn_qkv_fused = |q: &MetalTensor,
                                   k: &MetalTensor,
                                   v: &MetalTensor|
         -> Result<Option<MetalTensor>, MfError> {
            if !choices.fused_qkv_g8 {
                return Ok(None);
            }
            if !(q.dtype == k.dtype
                && q.dtype == v.dtype
                && q.dtype == GgmlType::Q8_0
                && q.shape.len() == 2
                && k.shape.len() == 2
                && v.shape.len() == 2
                && q.shape[0] == k.shape[0]
                && q.shape[0] == v.shape[0])
            {
                return Ok(None);
            }
            let read_bytes = |t: &MetalTensor| -> Vec<u8> {
                let n = t.n_bytes() as usize;
                let mut out = vec![0u8; n];
                unsafe {
                    let src = (t.buffer.contents().as_ptr() as *const u8).add(t.offset as usize);
                    std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), n);
                }
                out
            };
            let qb = read_bytes(q);
            let kb = read_bytes(k);
            let vb = read_bytes(v);
            let mut bytes = Vec::with_capacity(qb.len() + kb.len() + vb.len());
            bytes.extend_from_slice(&qb);
            bytes.extend_from_slice(&kb);
            bytes.extend_from_slice(&vb);
            let out_dim = checked_u64_add(
                checked_u64_add(q.shape[1], k.shape[1], "attn q+k out dim overflow")?,
                v.shape[1],
                "attn qkv fused out dim overflow",
            )?;
            Ok(Some(MetalTensor::from_bytes(
                ctx,
                &bytes,
                vec![q.shape[0], out_dim],
                q.dtype,
            )?))
        };

        let projection_role = projection_weight_role(model.arch.kind);
        let mut blocks = Vec::with_capacity(model.blocks.len());
        for b in &model.blocks {
            match b {
                Block::Gdn(g) => {
                    let attn_norm = loader.load_f32(g.attn_norm)?;
                    let post_attn_norm = loader.load_f32(g.post_attention_norm)?;
                    let ffn_gate = loader.load_weight(projection_role, g.ffn_gate)?;
                    let ffn_up = loader.load_weight(projection_role, g.ffn_up)?;
                    let ffn_down = loader.load_weight(projection_role, g.ffn_down)?;
                    let in_proj_qkv = loader.load_weight(projection_role, g.in_proj_qkv)?;
                    let in_proj_z = loader.load_weight(projection_role, g.in_proj_z)?;
                    let beta_proj = loader.load_weight(projection_role, g.beta_proj)?;
                    let alpha_proj = loader.load_weight(projection_role, g.alpha_proj)?;
                    let a_log = loader.load_f32(g.a_log)?;
                    let dt_bias = loader.load_f32(g.dt_bias)?;
                    let conv1d = loader.load_f32(g.conv1d)?;
                    let norm = loader.load_f32(g.norm)?;
                    let out_proj = loader.load_weight(projection_role, g.out_proj)?;
                    let ffn_moe = match g.ffn_moe.as_ref() {
                        Some(moe) => Some(loader.load_moe(moe)?),
                        None => None,
                    };
                    blocks.push(MetalBlock::Gdn(MetalGdnBlock {
                        attn_norm,
                        post_attn_norm,
                        ffn_gate,
                        ffn_up,
                        ffn_down,
                        in_proj_qkv,
                        in_proj_z,
                        beta_proj,
                        alpha_proj,
                        a_log,
                        dt_bias,
                        conv1d,
                        norm,
                        out_proj,
                        ffn_moe,
                    }));
                }
                Block::Attn(a) => {
                    let q = loader.load_weight(projection_role, a.q)?;
                    let k = loader.load_weight(projection_role, a.k)?;
                    let v = loader.load_weight(projection_role, a.v)?;
                    let attn_norm = loader.load_f32(a.attn_norm)?;
                    let post_attn_norm = loader.load_f32(a.post_attention_norm)?;
                    let ffn_gate = loader.load_weight(projection_role, a.ffn_gate)?;
                    let ffn_up = loader.load_weight(projection_role, a.ffn_up)?;
                    let ffn_down = loader.load_weight(projection_role, a.ffn_down)?;
                    let qkv_fused = load_attn_qkv_fused(&q, &k, &v)?;
                    loader.record_derived(qkv_fused.as_ref())?;
                    let o = loader.load_weight(projection_role, a.o)?;
                    let q_norm = loader.load_f32(a.q_norm)?;
                    let k_norm = loader.load_f32(a.k_norm)?;
                    let ffn_moe = match a.ffn_moe.as_ref() {
                        Some(moe) => Some(loader.load_moe(moe)?),
                        None => None,
                    };
                    blocks.push(MetalBlock::Attn(MetalAttnBlock {
                        attn_norm,
                        post_attn_norm,
                        ffn_gate,
                        ffn_up,
                        ffn_down,
                        qkv_fused,
                        q,
                        k,
                        v,
                        o,
                        q_norm,
                        k_norm,
                        ffn_moe,
                    }));
                }
            }
        }
        loader.finish(expected_storage_requests)?;
        crate::metal_dflash::log_prefill_moe_grouped_fallbacks(&model.arch, &blocks);

        Ok(Self {
            arch: model.arch,
            tied_embeddings: model.tied_embeddings,
            token_embd,
            output_norm,
            lm_head,
            blocks,
        })
    }
}

/// Per-sequence state: GDN conv buffers + SSM states (one set per GDN
/// layer), KV cache (one set per attn layer), and a small pool of
/// scratch activation tensors that are reused across layers.
pub struct MetalSession {
    poison_reason: Option<&'static str>,
    /// (kernel-1) * conv_dim per GDN layer, F32, contiguous.
    pub gdn_conv: Vec<MetalTensor>,
    /// n_v_heads * head_dim * head_dim per GDN layer, F32.
    pub gdn_state: Vec<MetalTensor>,

    /// `[capacity_tokens, n_kv_heads, head_dim]` per attention layer in the
    /// active KV storage type (normally F16, optionally Q8_0).
    pub kv_k: Vec<MetalTensor>,
    pub kv_v: Vec<MetalTensor>,
    pub kv_n_pos: Vec<usize>,
    pub kv_capacity: usize,

    // Scratch arena — F32 buffers reused across layers within a single
    // forward step. Sized for the largest transient at each role.
    pub x: MetalTensor,            // hidden_size — residual stream
    pub h: MetalTensor,            // hidden_size — post-norm activation
    pub ffn_gate: MetalTensor,     // intermediate_size — FFN gate output
    pub ffn_up: MetalTensor,       // intermediate_size — FFN up output
    pub ffn_inner: MetalTensor,    // intermediate_size — silu(gate)*up
    pub ffn_out: MetalTensor,      // hidden_size — FFN final
    pub gdn_qkv: MetalTensor,      // conv_dim
    pub gdn_qkv_conv: MetalTensor, // conv_dim — post-conv
    pub gdn_z: MetalTensor,        // n_v * head_dim
    pub gdn_b: MetalTensor,        // n_v   (β source, pre-sigmoid)
    pub gdn_beta: MetalTensor,     // n_v   (post-sigmoid)
    pub gdn_a: MetalTensor,        // n_v   (α source, pre-softplus)
    pub gdn_alpha: MetalTensor,    // n_v   (per-head decay exp(g))
    // gdn_q/k/v removed in v0.31: q/k/v are now zero-copy views into
    // gdn_qkv_conv via MetalTensor::view_subrange; no scratch buffers needed.
    pub gdn_q_norm: MetalTensor, // n_k * head_dim — l2-normed
    pub gdn_k_norm: MetalTensor, // n_k * head_dim — l2-normed
    pub gdn_out: MetalTensor,    // n_v * head_dim — recurrence output
    pub gdn_normed: MetalTensor, // n_v * head_dim — RMSNormGated output
    pub mixer_out: MetalTensor,  // hidden_size — mixer output (GDN or attn)

    // Attention scratch.
    pub attn_q_full: MetalTensor,   // 2 * q_dim — Q + gate interleaved
    pub attn_q: MetalTensor,        // q_dim — Q only
    pub attn_gate: MetalTensor,     // q_dim — sigmoid'd gate
    pub attn_q_normed: MetalTensor, // q_dim
    pub attn_k_now: MetalTensor,    // kv_dim — current step K
    pub attn_v_now: MetalTensor,    // kv_dim — current step V
    pub attn_k_normed: MetalTensor, // kv_dim
    pub attn_o: MetalTensor,        // q_dim — attention output
    // v4 flash-attn split-K partials. Sized for ATTN_V4_MAX_NWG; the
    // dispatcher passes the chosen NWG ≤ this value.
    pub attn_v4_o_partial: MetalTensor, // n_kv * NWG_max * GROUP * head_dim
    pub attn_v4_ml_partial: MetalTensor, // n_kv * NWG_max * GROUP * 2

    pub logits: MetalTensor,           // vocab_size
    pub argmax_tok: MetalTensor,       // [1] I32
    pub moe_router_probs: MetalTensor, // [n_expert] F32 (or [1] on dense models)
    pub moe_topk_idx: MetalTensor,     // [top_k] i32 in F32 buffer (or [1] on dense models)
    pub moe_topk_weight: MetalTensor,  // [top_k] F32 (or [1] on dense models)
    pub moe_shared_gate: MetalTensor,  // [1] F32
    pub moe_inner: MetalTensor,        // [top_k, expert_ffn] F32 (or [1] on dense models)
    pub moe_expert_out: MetalTensor,   // [top_k, hidden] F32 (or [1] on dense models)
    pub ids_buf: MetalTensor,          // [1] I32 input-token scratch
}

impl MetalSession {
    pub(crate) fn ensure_usable(&self) -> Result<(), MfError> {
        match self.poison_reason {
            Some(reason) => Err(MfError::SessionPoisoned { reason }),
            None => Ok(()),
        }
    }

    pub(crate) fn poison(&mut self, reason: &'static str) {
        self.poison_reason = Some(reason);
    }

    fn any_mutable_tensor(&self, mut predicate: impl FnMut(&MetalTensor) -> bool) -> bool {
        self.gdn_conv.iter().any(&mut predicate)
            || self.gdn_state.iter().any(&mut predicate)
            || self.kv_k.iter().any(&mut predicate)
            || self.kv_v.iter().any(&mut predicate)
            || [
                &self.x,
                &self.h,
                &self.ffn_gate,
                &self.ffn_up,
                &self.ffn_inner,
                &self.ffn_out,
                &self.gdn_qkv,
                &self.gdn_qkv_conv,
                &self.gdn_z,
                &self.gdn_b,
                &self.gdn_beta,
                &self.gdn_a,
                &self.gdn_alpha,
                &self.gdn_q_norm,
                &self.gdn_k_norm,
                &self.gdn_out,
                &self.gdn_normed,
                &self.mixer_out,
                &self.attn_q_full,
                &self.attn_q,
                &self.attn_gate,
                &self.attn_q_normed,
                &self.attn_k_now,
                &self.attn_v_now,
                &self.attn_k_normed,
                &self.attn_o,
                &self.attn_v4_o_partial,
                &self.attn_v4_ml_partial,
                &self.logits,
                &self.argmax_tok,
                &self.moe_router_probs,
                &self.moe_topk_idx,
                &self.moe_topk_weight,
                &self.moe_shared_gate,
                &self.moe_inner,
                &self.moe_expert_out,
                &self.ids_buf,
            ]
            .into_iter()
            .any(predicate)
    }

    pub(crate) fn aliases_mutable_buffer(&self, candidate: &MetalTensor) -> bool {
        self.any_mutable_tensor(|tensor| {
            Retained::as_ptr(&candidate.buffer) == Retained::as_ptr(&tensor.buffer)
        })
    }

    pub(crate) fn aliases_mutable_session(&self, other: &Self) -> bool {
        other.any_mutable_tensor(|tensor| self.aliases_mutable_buffer(tensor))
    }

    pub(crate) fn has_internal_mutable_alias(&self) -> bool {
        let mut seen = std::collections::HashSet::new();
        self.any_mutable_tensor(|tensor| {
            !seen.insert(Retained::as_ptr(&tensor.buffer) as *const () as usize)
        })
    }

    pub fn fresh(
        ctx: &MetalContext,
        model: &MetalModel,
        kv_capacity: usize,
    ) -> Result<Self, MetalError> {
        let arch = &model.arch;
        let h = arch.hidden_size as u64;
        let f = if arch.kind == ArchKind::Moe {
            arch.expert_shared_feed_forward_length
                .max(arch.expert_count)
                .max(arch.expert_feed_forward_length) as u64
        } else {
            arch.intermediate_size as u64
        };
        let head_dim = arch.attn_head_dim as u64;
        let n_q = arch.n_q_heads as u64;
        let n_kv = arch.n_kv_heads as u64;
        let q_dim = checked_u64_mul(n_q, head_dim, "q_dim overflow")?;
        let kv_dim = checked_u64_mul(n_kv, head_dim, "kv_dim overflow")?;
        let q_group = checked_u64_div_exact(n_q, n_kv, "n_q_heads / n_kv_heads invalid")?;
        let moe_router_n = if arch.kind == ArchKind::Moe {
            arch.expert_count.max(1) as u64
        } else {
            1
        };
        let moe_topk_n = if arch.kind == ArchKind::Moe {
            arch.expert_used_count.max(1).min(arch.expert_count.max(1)) as u64
        } else {
            1
        };
        let moe_inner_n = if arch.kind == ArchKind::Moe {
            checked_u64_mul(
                moe_topk_n,
                arch.expert_feed_forward_length.max(1) as u64,
                "moe_inner_n overflow",
            )?
        } else {
            1
        };
        let moe_expert_out_n = if arch.kind == ArchKind::Moe {
            checked_u64_mul(moe_topk_n, h, "moe_expert_out_n overflow")?
        } else {
            1
        };
        let vh = arch.gdn_head_dim as u64;
        let n_v = arch.gdn_n_v_heads as u64;
        let n_k = arch.gdn_n_k_heads as u64;
        let conv_heads = checked_u64_add(
            checked_u64_double(n_k, "2 * gdn_n_k_heads overflow")?,
            n_v,
            "2 * gdn_n_k_heads + gdn_n_v_heads overflow",
        )?;
        let conv_dim = checked_u64_mul(conv_heads, vh, "conv_dim overflow")?;
        let conv_kernel = arch.gdn_conv_kernel as u64;
        let v_dim = checked_u64_mul(n_v, vh, "v_dim overflow")?;
        let k_dim = checked_u64_mul(n_k, vh, "k_dim overflow")?;
        let kv_shape = vec![checked_u64_mul(
            kv_capacity as u64,
            kv_dim,
            "kv cache size overflow",
        )?];
        let gdn_conv_elems = checked_u64_mul(
            conv_kernel.saturating_sub(1),
            conv_dim,
            "gdn conv scratch size overflow",
        )?;
        let gdn_state_elems = checked_u64_mul3(n_v, vh, vh, "gdn state size overflow")?;
        let attn_q_full_elems = checked_u64_double(q_dim, "2 * q_dim overflow")?;
        let attn_v4_o_partial_elems = checked_u64_mul4(
            n_kv,
            ATTN_V4_MAX_NWG as u64,
            q_group,
            head_dim,
            "attn_v4_o_partial size overflow",
        )?;
        let attn_v4_ml_partial_elems = checked_u64_mul4(
            n_kv,
            ATTN_V4_MAX_NWG as u64,
            q_group,
            2,
            "attn_v4_ml_partial size overflow",
        )?;

        let mut gdn_conv = Vec::new();
        let mut gdn_state = Vec::new();
        for b in &model.blocks {
            if matches!(b, MetalBlock::Gdn(_)) {
                gdn_conv.push(MetalTensor::zeros_f32(ctx, vec![gdn_conv_elems])?);
                gdn_state.push(MetalTensor::zeros_f32(ctx, vec![gdn_state_elems])?);
            }
        }

        // KV cache defaults to F16 storage (matches llama.cpp's default
        // --cache-type-k f16). Experimental dense long-context path can
        // switch to Q8_0 via `QWEN_KV_Q8=1`; this is currently limited to the
        // dense group6 / head_dim=256 shape.
        let kv_dtype = kv_cache_dtype_for_arch(arch);
        let mut kv_k = Vec::new();
        let mut kv_v = Vec::new();
        let mut kv_n_pos = Vec::new();
        for b in &model.blocks {
            if matches!(b, MetalBlock::Attn(_)) {
                let shape = kv_shape.clone();
                kv_k.push(match kv_dtype {
                    GgmlType::F16 => MetalTensor::zeros_f16(ctx, shape.clone())?,
                    GgmlType::Q8_0 => MetalTensor::zeros_q8_0(ctx, shape.clone())?,
                    _ => unreachable!(),
                });
                kv_v.push(match kv_dtype {
                    GgmlType::F16 => MetalTensor::zeros_f16(ctx, shape.clone())?,
                    GgmlType::Q8_0 => MetalTensor::zeros_q8_0(ctx, shape.clone())?,
                    _ => unreachable!(),
                });
                kv_n_pos.push(0);
            }
        }

        Ok(Self {
            poison_reason: None,
            gdn_conv,
            gdn_state,
            kv_k,
            kv_v,
            kv_n_pos,
            kv_capacity,
            x: MetalTensor::zeros_f32(ctx, vec![h])?,
            h: MetalTensor::zeros_f32(ctx, vec![h])?,
            ffn_gate: MetalTensor::zeros_f32(ctx, vec![f])?,
            ffn_up: MetalTensor::zeros_f32(ctx, vec![f])?,
            ffn_inner: MetalTensor::zeros_f32(ctx, vec![f])?,
            ffn_out: MetalTensor::zeros_f32(ctx, vec![h])?,
            gdn_qkv: MetalTensor::zeros_f32(ctx, vec![conv_dim])?,
            gdn_qkv_conv: MetalTensor::zeros_f32(ctx, vec![conv_dim])?,
            gdn_z: MetalTensor::zeros_f32(ctx, vec![v_dim])?,
            gdn_b: MetalTensor::zeros_f32(ctx, vec![n_v])?,
            gdn_beta: MetalTensor::zeros_f32(ctx, vec![n_v])?,
            gdn_a: MetalTensor::zeros_f32(ctx, vec![n_v])?,
            gdn_alpha: MetalTensor::zeros_f32(ctx, vec![n_v])?,
            gdn_q_norm: MetalTensor::zeros_f32(ctx, vec![k_dim])?,
            gdn_k_norm: MetalTensor::zeros_f32(ctx, vec![k_dim])?,
            gdn_out: MetalTensor::zeros_f32(ctx, vec![v_dim])?,
            gdn_normed: MetalTensor::zeros_f32(ctx, vec![v_dim])?,
            mixer_out: MetalTensor::zeros_f32(ctx, vec![h])?,
            attn_q_full: MetalTensor::zeros_f32(ctx, vec![attn_q_full_elems])?,
            attn_q: MetalTensor::zeros_f32(ctx, vec![q_dim])?,
            attn_gate: MetalTensor::zeros_f32(ctx, vec![q_dim])?,
            attn_q_normed: MetalTensor::zeros_f32(ctx, vec![q_dim])?,
            attn_k_now: MetalTensor::zeros_f32(ctx, vec![kv_dim])?,
            attn_v_now: MetalTensor::zeros_f32(ctx, vec![kv_dim])?,
            attn_k_normed: MetalTensor::zeros_f32(ctx, vec![kv_dim])?,
            attn_o: MetalTensor::zeros_f32(ctx, vec![q_dim])?,
            // v4 partials: n_kv * NWG_max * GROUP * head_dim (and *2 for ml).
            attn_v4_o_partial: MetalTensor::zeros_f32(ctx, vec![attn_v4_o_partial_elems])?,
            attn_v4_ml_partial: MetalTensor::zeros_f32(ctx, vec![attn_v4_ml_partial_elems])?,
            logits: MetalTensor::zeros_f32(ctx, vec![arch.vocab_size as u64])?,
            argmax_tok: MetalTensor::zeros_i32(ctx, vec![1])?,
            moe_router_probs: MetalTensor::zeros_f32(ctx, vec![moe_router_n])?,
            moe_topk_idx: MetalTensor::zeros_f32(ctx, vec![moe_topk_n])?,
            moe_topk_weight: MetalTensor::zeros_f32(ctx, vec![moe_topk_n])?,
            moe_shared_gate: MetalTensor::zeros_f32(ctx, vec![1])?,
            moe_inner: MetalTensor::zeros_f32(ctx, vec![moe_inner_n])?,
            moe_expert_out: MetalTensor::zeros_f32(ctx, vec![moe_expert_out_n])?,
            ids_buf: MetalTensor::zeros_i32(ctx, vec![1])?,
        })
    }
}

/// End-to-end Metal forward driver.
pub struct MetalForward<'a> {
    pub ctx: &'a MetalContext,
    pub model: &'a MetalModel,
}

#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct StructuralRowEvidence {
    pub resident_head_wait_calls: u64,
    pub validated_shared_row_calls: u64,
    pub transition_logits_copy_bytes: u64,
    pub extra_command_buffers: u64,
    pub gpu_sampling_dispatches: u64,
}

#[derive(Debug, Clone)]
pub struct DecodeStageTiming {
    pub family: String,
    pub block_kind: String,
    pub block_index: Option<usize>,
    pub local_index: Option<usize>,
    pub concurrent: bool,
    pub start_sample: usize,
    pub end_sample: usize,
    pub start_timestamp: u64,
    pub end_timestamp: u64,
    pub duration_ticks: u64,
    pub duration_ms_scaled: f64,
    pub fraction_of_gpu: f64,
}

#[derive(Debug, Clone)]
pub struct DecodeStageProfile {
    pub token: TokenProfile,
    pub stages: Vec<DecodeStageTiming>,
    pub sampled_span_ticks: u64,
    pub raw_span_ms_assuming_ns: f64,
    pub raw_coverage_assuming_ns: f64,
}

#[derive(Clone, Copy)]
struct DecodeStageMeta {
    family: &'static str,
    block_kind: &'static str,
    block_index: Option<usize>,
    local_index: Option<usize>,
}

struct DecodeStageRecord {
    meta: DecodeStageMeta,
    concurrent: bool,
    start_sample: usize,
    end_sample: usize,
}

struct DecodeStageRecorder {
    samples: MetalTimestampSampleBuffer,
    next_sample: usize,
    records: Vec<DecodeStageRecord>,
}

use crate::metal::dispatch_census_set_family as qwen_llm_dispatch_census_set_family;

mod attn;
mod dispatch;
mod gdn;
mod moe;
#[cfg(test)]
mod native_embedding_pilot;

#[cfg(test)]
mod snapshot_transfer_pilot;

mod residency;
#[cfg(test)]
mod role_tests;
mod session;
#[cfg(test)]
mod snapshot_segments_pilot;
mod support;
#[cfg(test)]
mod tests;
mod token;
#[allow(unused_imports)]
pub use attn::*;
#[allow(unused_imports)]
pub use dispatch::*;
#[allow(unused_imports)]
pub use gdn::*;
#[allow(unused_imports)]
pub use moe::*;
#[allow(unused_imports)]
pub use residency::*;
#[allow(unused_imports)]
pub use session::*;
#[allow(unused_imports)]
pub use support::*;
#[allow(unused_imports)]
pub use token::*;

pub const RMS_EPS: f32 = 1e-6;

fn automatic_parallel_copy_capacity_admitted(
    no_copy_mode: GgufNoCopyMode,
    parallel_mode: GgufParallelCopyMode,
    capacity_would_map: bool,
) -> bool {
    let automatic =
        no_copy_mode == GgufNoCopyMode::Auto && parallel_mode == GgufParallelCopyMode::Auto;
    !automatic || !capacity_would_map
}

crate::env_flag!(default_on matmat_smalln_table_enabled, "QWEN_MATMAT_SMALLN_TABLE");

crate::env_flag!(default_on matmat_q4_vec4_enabled, "QWEN_MATMAT_Q4_VEC4");
crate::env_flag!(default_on matmat_q5_k_n2_seq_enabled, "QWEN_MATMAT_Q5_K_N2_SEQ");
crate::env_flag!(default_on matmat_n1_matvec_enabled, "QWEN_MATMAT_N1_MATVEC");
crate::env_flag!(default_on matmat_iq2_s_n2_nc2_enabled, "QWEN_MATMAT_IQ2_S_N2_NC2");
crate::env_flag!(default_on matmat_iq3_s_n2_nc2_enabled, "QWEN_MATMAT_IQ3_S_N2_NC2");
crate::env_flag!(default_on matmat_q8_fewrow_enabled, "QWEN_MATMAT_Q8_FEWROW");
