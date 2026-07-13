//! `qwen-bench` — end-to-end decode throughput harness for qwen-llm.
//!
//! Replaces the prior stub (load-time only) with a real bench surface.
//! Three modes:
//!
//!   `decode`       — Run a prompt + N-token decode loop; report ms/token,
//!                    GPU vs wall split, per-token series, and (when an
//!                    oracle is available) cos vs oracle for correctness.
//!   `ctx-sweep`    — Ramp KV to each checkpoint context length, time a
//!                    short window of decodes there, report ms/token and
//!                    effective bandwidth at each point.
//!   `phase`        — Phase-resolved profile at one chosen context.
//!
//! Designed so "measure → change → measure" is `cargo run --release -p
//! qwen-cli --bin qwen-bench -- decode -m ... --tokens 64`, not "run the
//! right ignored test by name". Per Jeff & Sanjay (and the v0.32
//! re-sequencing review): the bench harness IS leverage, not hygiene.

mod attn_capture;
mod attn_stage_floor;
#[path = "../source_identity.rs"]
mod source_identity;

use anyhow::{Context, Result, anyhow};
use clap::{Parser, Subcommand, ValueEnum};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSError, NSString};
use objc2_metal::{
    MTLAllocation, MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLComputePipelineState,
    MTLDevice, MTLResidencySet, MTLResidencySetDescriptor, MTLSize,
};
use qwen_llm::{
    forward::mat_vec_pub,
    gguf::GgufFile,
    loader::{Model, open_dflash_drafter},
    metal::{
        BlitEncoder, KernelEncoder, KernelTraceCounters, MetalContext, MetalTensor,
        attn_v4_choose_group_tile, attn_v4_choose_nwg, attn_v4_choose_tile_c,
        encode_add_inplace_f32, encode_attn_decode_v4_f32, encode_attn_decode_v4_main_only_f32,
        encode_attn_decode_v4_reduce_only_f32, encode_attn_prefill_v4_g8_t2_q2_c64_f32,
        encode_attn_prefill_v4_g8_t2_q4_c64_f32, encode_attn_prefill_v4_g16_t4_q2_c64_f32,
        encode_attn_prefill_v4_g16_t4_q4_c64_f32, encode_fill_f32, encode_gdn_decay_chain_f32,
        encode_get_rows_f32, encode_moe_down_weighted_sum_q5_K_f32_packed_slots,
        encode_moe_down_weighted_sum_q5_K_f32_packed_slots_k512_r2,
        encode_moe_fused_routed_q4q5_token_f32, encode_moe_swiglu_q4_K_f32,
        encode_moe_swiglu_q4_K_f32_packed_slots, encode_mul_f32, encode_rms_norm_batched_f32,
        encode_rms_norm_mul_f32, encode_roofline_fma_f32, encode_roofline_stream_f32,
        encode_rope_neox_f32, encode_rope_neox_f32_packed_consecutive,
        encode_scatter_offset_f32_to_f16, encode_scatter_offset_f32_to_f16_kv,
        encode_scatter_offset_f32_to_q8_0_kv, encode_sigmoid_f32, encode_sigmoid_mul_f32,
        encode_split_q_gate_f32, encode_touch_bytes_f32, kernel_trace_begin, kernel_trace_snapshot,
        with_attn_v4_group_tile_override,
    },
    metal_dflash::{
        DFlashDecoder, MetalDFlashHead, MetalDFlashLayerMajorScratch, MetalDFlashSession,
        MetalDFlashVerifyScratch, prefill_tokens_prompt_only_profiled,
        prefill_tokens_with_multi_hidden, prefill_tokens_with_multi_hidden_profiled,
        with_prefill_dense_ffn_fused_swiglu_q4_override,
    },
    metal_forward::{
        MetalBlock, MetalForward, MetalModel, MetalMoeFfn, MetalSession, MoeRouteReplayRow, RMS_EPS,
    },
    metal_forward::{encode_mat_mat_dispatch, encode_mat_vec_dispatch},
    metal_mtp::{
        DecodeOutput, MetalMtpHead, MetalMtpSession, MtpBaseHiddenVariant, MtpHistoryMode,
        MtpRankRow, MtpRecursiveHiddenVariant, PackedDraftPlan, RecordedDraftStep, RecordedMtpWork,
        SpeculativeDecoder, quantize_lm_head_to_affine_q4_gs64, quantize_lm_head_to_q4_0,
        quantize_lm_head_to_q4_1,
    },
    prompt_lookup::{
        DRAFT_TOKENS, PromptLookupProposer, PromptLookupTerminalCause, ProposalSource,
        terminal_draft_window,
    },
    runtime::{LoadedModel, Runtime, SequenceConfig},
    tensor::GgmlType,
    tokenizer::{LlamaCppTokenizer, NativeTokenizer, Tokenizer},
};
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

fn env_flag_enabled(name: &str) -> bool {
    matches!(
        std::env::var(name).as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
    )
}

#[derive(Clone, Copy, Debug)]
struct MoeRouteBatchStats {
    layers: usize,
    tokens: usize,
    slots_per_layer: usize,
    avg_unique_experts: f64,
    avg_max_slots: f64,
    avg_reuse: f64,
}

fn summarize_moe_route_batch(
    routes_by_token: &[Vec<MoeRouteReplayRow>],
    eligible_indices: &[usize],
    n_expert: usize,
    topk: usize,
) -> Result<MoeRouteBatchStats> {
    let tokens = routes_by_token.len();
    if tokens == 0 || eligible_indices.is_empty() {
        return Err(anyhow!("captured route batch is empty"));
    }

    let mut unique_sum = 0usize;
    let mut max_sum = 0usize;
    let mut reuse_sum = 0.0f64;
    let slots_per_layer = tokens * topk;
    for &moe_i in eligible_indices {
        let mut counts = vec![0usize; n_expert];
        for routes in routes_by_token {
            let route = routes
                .get(moe_i)
                .ok_or_else(|| anyhow!("captured route missing layer {moe_i}"))?;
            if route.topk_idx.len() != topk {
                return Err(anyhow!(
                    "captured route has {} experts, expected topk={topk}",
                    route.topk_idx.len()
                ));
            }
            for &expert in &route.topk_idx {
                if expert < 0 || expert as usize >= n_expert {
                    return Err(anyhow!(
                        "captured expert id {expert} outside n_expert={n_expert}"
                    ));
                }
                counts[expert as usize] += 1;
            }
        }
        let unique = counts.iter().filter(|&&c| c > 0).count();
        let max_count = counts.iter().copied().max().unwrap_or(0);
        unique_sum += unique;
        max_sum += max_count;
        reuse_sum += if unique > 0 {
            slots_per_layer as f64 / unique as f64
        } else {
            0.0
        };
    }
    let layers = eligible_indices.len();
    Ok(MoeRouteBatchStats {
        layers,
        tokens,
        slots_per_layer,
        avg_unique_experts: unique_sum as f64 / layers as f64,
        avg_max_slots: max_sum as f64 / layers as f64,
        avg_reuse: reuse_sum / layers as f64,
    })
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum CaptureTokenPattern {
    Zero,
    Ramp,
}

#[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd, ValueEnum)]
enum MoeBatchSlotOrder {
    Exact,
    ExpertSortedPerfOnly,
}

impl MoeBatchSlotOrder {
    fn label(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::ExpertSortedPerfOnly => "expert-sorted-perf-only",
        }
    }
}

fn capture_replay_token(pattern: CaptureTokenPattern, tok: usize, vocab_size: u32) -> i32 {
    match pattern {
        CaptureTokenPattern::Zero => 0,
        CaptureTokenPattern::Ramp => ((1 + tok * 7919) % vocab_size as usize) as i32,
    }
}

fn captured_gateup_tensors(
    ctx: &MetalContext,
    routes_by_token: &[Vec<MoeRouteReplayRow>],
    eligible_indices: &[usize],
    h: usize,
    n_expert: usize,
    topk: usize,
    slot_order: MoeBatchSlotOrder,
) -> Result<Vec<(MetalTensor, MetalTensor)>> {
    let tokens = routes_by_token.len();
    let slots = tokens * topk;
    let mut tensors = Vec::with_capacity(eligible_indices.len());
    for &moe_i in eligible_indices {
        let hidden_t = MetalTensor::zeros_f32(ctx, vec![(tokens * h) as u64])?;
        let idx = MetalTensor::zeros_f32(ctx, vec![slots as u64])?;
        unsafe {
            let dst = hidden_t.buffer.contents().as_ptr() as *mut f32;
            let ptr = idx.buffer.contents().as_ptr() as *mut i32;
            let mut sorted_slots = Vec::new();
            if slot_order == MoeBatchSlotOrder::ExpertSortedPerfOnly {
                sorted_slots.reserve(slots);
            }
            for tok in 0..tokens {
                let route = routes_by_token[tok]
                    .get(moe_i)
                    .ok_or_else(|| anyhow!("captured route missing layer {moe_i}"))?;
                if route.hidden.len() != h {
                    return Err(anyhow!(
                        "captured hidden has {} elements, expected h={h}",
                        route.hidden.len()
                    ));
                }
                if route.topk_idx.len() != topk {
                    return Err(anyhow!(
                        "captured route has {} experts, expected topk={topk}",
                        route.topk_idx.len()
                    ));
                }
                std::ptr::copy_nonoverlapping(route.hidden.as_ptr(), dst.add(tok * h), h);
                for (slot, &expert) in route.topk_idx.iter().enumerate() {
                    if expert < 0 || expert as usize >= n_expert {
                        return Err(anyhow!(
                            "captured expert id {expert} outside n_expert={n_expert}"
                        ));
                    }
                    match slot_order {
                        MoeBatchSlotOrder::Exact => {
                            *ptr.add(tok * topk + slot) = expert;
                        }
                        MoeBatchSlotOrder::ExpertSortedPerfOnly => {
                            sorted_slots.push((expert, tok, slot));
                        }
                    }
                }
            }
            if slot_order == MoeBatchSlotOrder::ExpertSortedPerfOnly {
                sorted_slots.sort_unstable();
                for (out_slot, (expert, _, _)) in sorted_slots.iter().copied().enumerate() {
                    *ptr.add(out_slot) = expert;
                }
            }
        }
        tensors.push((hidden_t, idx));
    }
    Ok(tensors)
}

fn captured_down_tensors(
    ctx: &MetalContext,
    routes_by_token: &[Vec<MoeRouteReplayRow>],
    eligible_indices: &[usize],
    n_expert: usize,
    topk: usize,
    slot_order: MoeBatchSlotOrder,
) -> Result<Vec<(MetalTensor, MetalTensor)>> {
    let tokens = routes_by_token.len();
    let slots = tokens * topk;
    let mut tensors = Vec::with_capacity(eligible_indices.len());
    for &moe_i in eligible_indices {
        let idx = MetalTensor::zeros_f32(ctx, vec![slots as u64])?;
        let weight = MetalTensor::zeros_f32(ctx, vec![slots as u64])?;
        unsafe {
            let idx_ptr = idx.buffer.contents().as_ptr() as *mut i32;
            let w_ptr = weight.buffer.contents().as_ptr() as *mut f32;
            let mut sorted_slots = Vec::new();
            if slot_order == MoeBatchSlotOrder::ExpertSortedPerfOnly {
                sorted_slots.reserve(slots);
            }
            for tok in 0..tokens {
                let route = routes_by_token[tok]
                    .get(moe_i)
                    .ok_or_else(|| anyhow!("captured route missing layer {moe_i}"))?;
                if route.topk_idx.len() != topk || route.topk_weight.len() != topk {
                    return Err(anyhow!(
                        "captured route has idx={} weight={}, expected topk={topk}",
                        route.topk_idx.len(),
                        route.topk_weight.len()
                    ));
                }
                for slot in 0..topk {
                    let expert = route.topk_idx[slot];
                    if expert < 0 || expert as usize >= n_expert {
                        return Err(anyhow!(
                            "captured expert id {expert} outside n_expert={n_expert}"
                        ));
                    }
                    match slot_order {
                        MoeBatchSlotOrder::Exact => {
                            let out_slot = tok * topk + slot;
                            *idx_ptr.add(out_slot) = expert;
                            *w_ptr.add(out_slot) = route.topk_weight[slot];
                        }
                        MoeBatchSlotOrder::ExpertSortedPerfOnly => {
                            sorted_slots.push((expert, tok, slot, route.topk_weight[slot]));
                        }
                    }
                }
            }
            if slot_order == MoeBatchSlotOrder::ExpertSortedPerfOnly {
                sorted_slots
                    .sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
                for (out_slot, (expert, _, _, route_weight)) in
                    sorted_slots.iter().copied().enumerate()
                {
                    *idx_ptr.add(out_slot) = expert;
                    *w_ptr.add(out_slot) = route_weight;
                }
            }
        }
        tensors.push((idx, weight));
    }
    Ok(tensors)
}

fn pp_warm_moe_weight_banks(ctx: &MetalContext, mf: &MetalForward<'_>) -> Result<usize> {
    let stride_bytes = std::env::var("QWEN_PP_TOUCH_STRIDE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(16 * 1024);
    let sink = MetalTensor::zeros_f32(ctx, vec![256])?;
    let cmd = ctx.queue.commandBuffer().context("warmup command buffer")?;
    let enc = KernelEncoder::begin(&cmd);
    let mut touched = 0usize;
    for block in &mf.model.blocks {
        let moe = match block {
            MetalBlock::Gdn(g) => g.ffn_moe.as_ref(),
            MetalBlock::Attn(a) => a.ffn_moe.as_ref(),
        };
        let Some(moe) = moe else { continue };
        for tensor in [&moe.gate_exps, &moe.up_exps, &moe.down_exps] {
            encode_touch_bytes_f32(ctx, &enc, tensor, &sink, stride_bytes)?;
            touched += 1;
        }
    }
    enc.end();
    cmd.commit();
    cmd.waitUntilCompleted();
    Ok(touched)
}

fn buffer_as_allocation(
    buffer: &ProtocolObject<dyn MTLBuffer>,
) -> &ProtocolObject<dyn MTLAllocation> {
    ProtocolObject::from_ref(buffer)
}

struct PpResidencySetGuard {
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    set: Retained<ProtocolObject<dyn MTLResidencySet>>,
}

impl Drop for PpResidencySetGuard {
    fn drop(&mut self) {
        self.queue.removeResidencySet(&self.set);
        self.set.endResidency();
    }
}

fn pp_register_moe_residency_set(
    ctx: &MetalContext,
    mf: &MetalForward<'_>,
) -> Result<(PpResidencySetGuard, usize, u64)> {
    let mut seen = HashSet::new();
    let mut buffers = Vec::new();
    for block in &mf.model.blocks {
        let moe = match block {
            MetalBlock::Gdn(g) => g.ffn_moe.as_ref(),
            MetalBlock::Attn(a) => a.ffn_moe.as_ref(),
        };
        let Some(moe) = moe else { continue };
        for tensor in [&moe.gate_exps, &moe.up_exps, &moe.down_exps] {
            let ptr = Retained::as_ptr(&tensor.buffer) as *const _ as usize;
            if seen.insert(ptr) {
                buffers.push(&*tensor.buffer);
            }
        }
    }
    if buffers.is_empty() {
        return Err(anyhow!(
            "no MoE expert-bank buffers found for residency set"
        ));
    }

    let desc = MTLResidencySetDescriptor::new();
    desc.setLabel(Some(&NSString::from_str("qwen-bench-pp-moe-banks")));
    // SAFETY: initialCapacity is advisory only; we pass the exact number of
    // unique allocations we are about to register.
    unsafe { desc.setInitialCapacity(buffers.len()) };
    let set = ctx
        .device
        .newResidencySetWithDescriptor_error(&desc)
        .map_err(|e: Retained<NSError>| anyhow!(e.localizedDescription().to_string()))?;
    for buffer in &buffers {
        set.addAllocation(buffer_as_allocation(buffer));
    }
    set.commit();
    set.requestResidency();
    ctx.queue.addResidencySet(&set);
    let bytes = set.allocatedSize();
    let guard = PpResidencySetGuard {
        queue: ctx.queue.clone(),
        set,
    };
    Ok((guard, buffers.len(), bytes))
}

#[derive(Parser, Debug)]
#[command(
    name = "qwen-bench",
    version,
    about = "end-to-end throughput benchmark for qwen-llm"
)]
struct Args {
    /// Permit benchmarks from a known dirty source checkout. The dirty state
    /// remains recorded in every canonical JSON row.
    #[arg(long, global = true)]
    allow_dirty: bool,
    /// Permit a benchmark when the compiled binary cannot verify its source
    /// checkout. Commit mismatches are never overridable.
    #[arg(long, global = true)]
    allow_unverifiable_build: bool,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Capture sparse true-long prefill attention tensors; this is not a timing benchmark.
    AttnCapture(attn_capture::AttnCaptureArgs),
    /// Price the fixed 32K compressed-KV matrix staging floor.
    AttnStageFloor(attn_stage_floor::AttnStageFloorArgs),
    /// Report compiled and runtime source identity without initializing Metal
    /// or loading a model.
    BuildInfo(BuildInfoArgs),
    /// Decode N tokens after a prompt using the plain no-spec path.
    ///
    /// Packed prefill is the default no-spec path. `--sequential-prefill`
    /// keeps the legacy token-by-token prompt replay loop for A/B work.
    Decode(DecodeArgs),
    /// Prompt-only prefill benchmark aligned with llama-bench pp semantics.
    Pp(PpArgs),
    /// Generation-only benchmark aligned with `llama-bench tg<N>` semantics:
    /// empty KV per rep, random tokens, no logits readback, N decode steps.
    /// This is the apples-to-apples decode comparison. Use `decode` for real
    /// generation with a prompt.
    Tg(TgArgs),
    /// In-process synthetic pp/tg suite: load one model once, then run many
    /// shapes with fresh sessions per row.
    Suite(SuiteArgs),
    /// Sweep context length (ramp + measure window).
    CtxSweep(CtxSweepArgs),
    /// Phase-resolved profile at one context length (uses the
    /// `phase_sum` GPU time, NOT the per-phase-cmdbuf wall artifact).
    Phase(PhaseArgs),
    /// Decode attention intra-layer profile at one context length.
    AttnIntra(AttnIntraArgs),
    /// Exact-shape GDN projection primitive microbench.
    GdnProjMicro(GdnProjMicroArgs),
    /// Decode projection batching probe across GDN, attention, FFN, and lm_head.
    DecodeProjBatch(DecodeProjBatchArgs),
    /// One-layer GDN replay probe with batched qkv/z/out projections.
    DecodeGdnLayerReplay(DecodeGdnLayerReplayArgs),
    /// Chained multi-GDN replay probe with batched qkv/z/out projections.
    DecodeGdnChainReplay(DecodeGdnChainReplayArgs),
    /// MoE block-slice replay probe with normal attention/MoE around GDN replay.
    DecodeBlockSliceReplay(DecodeBlockSliceReplayArgs),
    /// Diagnostic route/topk trace for block-slice replay correctness cliffs.
    DecodeBlockSliceTrace(DecodeBlockSliceTraceArgs),
    /// Loaded-once summary sweep for block-slice replay route margins.
    DecodeBlockSliceMarginSweep(DecodeBlockSliceMarginSweepArgs),
    /// Real-prompt summary sweep for block-slice replay route margins.
    DecodeBlockSliceRealMargin(DecodeBlockSliceRealMarginArgs),
    /// Real-prompt top-k check for opt-in F16 MoE router repacks.
    DecodeMoeRouterRepackCheck(DecodeMoeRouterRepackCheckArgs),
    /// Exact-shape MoE routed-down primitive microbench.
    MoeDownMicro(MoeDownMicroArgs),
    /// Exact-shape MoE routed gate/up primitive microbench.
    MoeGateupMicro(MoeGateupMicroArgs),
    /// Loaded-once captured MoE token-batching sweep.
    MoeBatchSweep(MoeBatchSweepArgs),
    /// Calibrate simple device bandwidth and arithmetic ceilings.
    Roofline(RooflineArgs),
    /// Report Metal counter-set availability for in-process counter probes.
    MetalCounters(MetalCountersArgs),
    /// Report Metal compute-pipeline resource hints for hot kernels.
    MetalPipelines(MetalPipelinesArgs),
    /// **B0 topology probe** (Program B gate 0, cx-signed design in
    /// docs/bench/2026-07-05-b0-topology-probe/): residency census (arm R),
    /// bounded one-way signaling + cross-object reorder rate (arm S), and
    /// the causal boundary-drain ladder-vs-persistent comparison (arm D).
    /// Bench-only kernels; quiet-box rules apply.
    TopologyProbe(TopologyProbeArgs),
    /// **W-program attribution**: per-family x per-kernel dispatch WIDTH
    /// census for one decode token (grid TGs, threads/TG, simdgroups),
    /// joined with stage times. Answers "how much token time sits in
    /// dispatches too narrow to fill 40 cores" with exact shapes.
    DispatchCensus(DispatchCensusArgs),
    /// Warm to a target context, then wait for an external go signal before
    /// running a fixed decode window. Intended for attach-mode tracing so the
    /// recorder can skip the long ramp.
    DecodeWindow(DecodeWindowArgs),
    /// **H2 falsification**: compare cold prefill TTFT vs snapshot-restore
    /// TTFT for two requests sharing a token prefix.
    PrefixCache(PrefixCacheArgs),
    /// **H3 falsification**: measure rank distribution of argmax tokens
    /// over a prompt corpus to determine whether vocab pruning at lm_head
    /// is viable. Reports miss rate at K ∈ {1K, 4K, 8K, 16K, 32K, 48K,
    /// 64K, 96K}, by-category breakdown, and decoded examples of any
    /// out-of-K tokens for inspection.
    VocabAudit(VocabAuditArgs),
    /// **H4.3 measurement**: run greedy generation twice (MTP=on and
    /// MTP=off) on the same prompt+limit, compare token sequences for
    /// equivalence, report speedup + acceptance rate + per-iter MTP
    /// call counts. Requires an MTP-aware GGUF (e.g. brittlewis12/
    /// Qwen3.6-27B-MTP-GGUF or the 0.8B-MTP variant).
    Mtp(MtpArgs),
    /// Target-only prompt lookup with a frozen L8/D7 recent-match policy.
    Pld(PldArgs),
    /// **H5.2.5 lazy DFlash acceptance gate**: measure α for the DFlash
    /// drafter using H4-style single-token sequential verify (no packed
    /// kernels yet). The acceptance rate signal tells us whether the
    /// drafter is producing a useful distribution under our quants +
    /// SWA mask + hidden capture path. GO/NO-GO for H5.3 packed verify.
    ///
    /// Reports per-position acceptance, top-k rank of target's argmax
    /// in drafter logits (for the future DDTree decision), effective-N
    /// sweep, and an apples-to-apples no-spec baseline.
    DflashLazy(DflashLazyArgs),
    /// **H5.5 production DFlash decode**: end-to-end DFlash speculative
    /// decode using the H5.3 packed_verify + H5.4 restore_after_partial_accept
    /// primitives. Greedy accept-prefix per plan §1.3.
    ///
    /// Per outer step:
    ///   draft_block(carry, processed_pos+1)  -> [N] argmaxes
    ///   packed_verify(carry + drafts[0..D-1], start_pos=processed_pos+1)
    ///                                        -> [N] verify_argmax tokens
    ///   greedy match prefix → n_accepted ∈ [0, D]
    ///   emit carry + accepted drafts; bonus = verify_argmax[n_accepted] becomes next carry
    ///   restore_after_partial_accept(n_accepted+1, ...) on partial reject
    ///
    /// Reports α_chain, mean_emitted_per_step, decode-only and total t/s,
    /// speedup vs DFlash=off baseline. Greedy equivalence with DFlash=off
    /// is asserted (token sequences must be identical).
    Dflash(DflashArgs),
    /// Tokenizer microbench: compare native GGUF Qwen35 tokenizer against
    /// the current llama.cpp FFI oracle for encode/decode parity + speed.
    Tok(TokArgs),
    #[command(hide = true)]
    AttnPrefillMicro(AttnPrefillMicroArgs),
    #[command(hide = true)]
    AttnFrontMicro(AttnFrontMicroArgs),
    #[command(hide = true)]
    AttnLayerMicro(AttnLayerMicroArgs),
    #[command(hide = true)]
    PpFfnAb(PpFfnAbArgs),
    #[command(hide = true)]
    PpWait(PpWaitArgs),
}

#[derive(Parser, Debug)]
struct BuildInfoArgs {
    /// Output format. JSON emits one object rather than a benchmark-row array.
    #[arg(short = 'o', long, value_enum, default_value = "json")]
    output: OutputFormat,
}

#[derive(Parser, Debug)]
struct DecodeArgs {
    /// Path to a GGUF file.
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Prompt text. If absent, uses a fixed warmup prompt.
    #[arg(short = 'p', long)]
    prompt: Option<String>,
    /// Number of tokens to generate after the prompt.
    #[arg(long, default_value = "64")]
    tokens: usize,
    /// Optional oracle file (raw f32 logits at last position from
    /// llama.cpp/llm `--snapshot`). If provided, compares cos.
    #[arg(long)]
    oracle: Option<PathBuf>,
    /// Which logits row the oracle should validate.
    #[arg(long, value_enum, default_value = "final")]
    oracle_phase: OraclePhase,
    /// Skip the warmup pass (default is to do one warmup, then re-init
    /// the session for the timed run, exactly like the ignored tests).
    #[arg(long)]
    no_warmup: bool,
    /// Force the legacy sequential prompt replay loop. Useful for A/B timing
    /// against the dense packed prefill path.
    #[arg(long)]
    sequential_prefill: bool,
    /// Packed prefill chunk size for the layer-major path. If omitted, decode
    /// chooses a model-aware default (currently dense=256, MoE=16).
    #[arg(long)]
    prefill_chunk: Option<usize>,
    /// Override decode session KV capacity for capacity-sensitivity tests.
    /// Must be at least prompt_tokens + generation_tokens + 16.
    #[arg(long)]
    kv_capacity: Option<usize>,
    /// Force decode to read back full logits on every generated token instead
    /// of using the GPU argmax fast path.
    #[arg(long)]
    full_logits_decode: bool,
    /// Number of timed repetitions. Each rep re-tokenizes, re-prefills, and
    /// re-decodes from a fresh session. avg_ts / stddev_ts are over reps.
    #[arg(long, default_value = "1")]
    runs: usize,
    /// `text` or `json` (`llama-bench -o json` shape).
    #[arg(short = 'o', long, value_enum, default_value = "text")]
    output: OutputFormat,
}

#[derive(Parser, Debug)]
struct MetalCountersArgs {}

#[derive(Parser, Debug)]
struct MetalPipelinesArgs {
    /// Kernel names to inspect. If omitted, prints the hot decode audit set.
    #[arg(long = "kernel", value_delimiter = ',')]
    kernels: Vec<String>,
}

#[derive(Parser, Debug)]
struct DispatchCensusArgs {
    /// Path to a GGUF file (MoE arch; uses the stage-profiled decode entry).
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Context position to census at (KV warmed to this depth first).
    #[arg(long, default_value = "16384")]
    ctx: usize,
    /// Warm via the token-by-token decode ramp instead of packed prefill.
    #[arg(long)]
    decode_ramp_warm: bool,
    /// Output JSON path.
    #[arg(long, default_value = "target/profiles/dispatch-census/census.json")]
    out: PathBuf,
}

#[derive(Parser, Debug)]
struct TopologyProbeArgs {
    /// Arms to run: comma list of r,s,d or `all`. Arm D needs arm R results
    /// (or --grid-tgs) for conservative persistent-grid sizing.
    #[arg(long, default_value = "all")]
    arm: String,
    /// Timed repetitions per configuration (median reported).
    #[arg(long, default_value = "5")]
    runs: usize,
    /// Output directory for JSON artifacts.
    #[arg(long, default_value = "target/profiles/topology-probe")]
    out_dir: PathBuf,
    /// Override the persistent-grid TG count for arm D (default: arm R
    /// low-water p10 of the steady entry-alive distribution).
    #[arg(long)]
    grid_tgs: Option<usize>,
    /// Smoke mode: smaller grids/epochs/dwells for a fast end-to-end pass.
    #[arg(long)]
    quick: bool,
    /// Extra arm-R dwell points (us) for plateau confirmation, e.g.
    /// `--dwell-extend 15000`. Runs lo/no-traffic variants only.
    #[arg(long, value_delimiter = ',')]
    dwell_extend: Vec<f64>,
}

#[derive(Parser, Debug)]
struct PpArgs {
    /// Path to a GGUF file.
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Synthetic prompt token count, matching llama-bench's pp<N> shape.
    #[arg(
        short = 'p',
        long = "n-prompt",
        alias = "tokens",
        default_value = "320"
    )]
    n_prompt: usize,
    /// Optional real prompt text. If set, --n-prompt is ignored.
    #[arg(long, conflicts_with_all = ["file", "messages"])]
    prompt: Option<String>,
    /// Read prompt text from a file. If set, --n-prompt is ignored.
    #[arg(long, conflicts_with = "messages")]
    file: Option<PathBuf>,
    /// Render a JSON messages input into a Qwen chat-template prompt.
    ///
    /// Accepted shapes:
    /// - bare `[{ role, content }, ...]`
    /// - wrapped `{ messages: [...], ... }`
    #[arg(long)]
    messages: Option<PathBuf>,
    /// Use only the first N messages from `--messages` before rendering.
    #[arg(long)]
    messages_max: Option<usize>,
    /// Preserve assistant `<think>...</think>` history from `--messages`.
    #[arg(long)]
    messages_preserve_thinking: bool,
    /// Force stripping assistant `<think>...</think>` history from
    /// `--messages`, even if auto-detection would preserve it.
    #[arg(long, conflicts_with = "messages_preserve_thinking")]
    messages_strip_thinking: bool,
    /// Do not append a final `<|im_start|>assistant\n` generation marker for
    /// `--messages` prompts.
    #[arg(long)]
    messages_no_generation_prompt: bool,
    /// Number of timed repetitions after warmup.
    #[arg(long, default_value = "5")]
    runs: usize,
    /// Skip the warmup prefill pass.
    #[arg(long)]
    no_warmup: bool,
    /// Packed prefill chunk size. If omitted, uses the model-aware default.
    #[arg(long)]
    prefill_chunk: Option<usize>,
    /// Include final norm + lm_head + logits readback, like decode's prefill seed.
    #[arg(long)]
    with_tail: bool,
    /// Deterministic seed for synthetic token generation.
    #[arg(long, default_value = "1")]
    seed: u64,
    /// `text` or `json` (`llama-bench -o json` shape).
    #[arg(short = 'o', long, value_enum, default_value = "text")]
    output: OutputFormat,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum OraclePhase {
    Prefill,
    Final,
}

/// Generation-only bench, modeled after `llama-bench tg<N>`.
///
/// Each rep: fresh session (empty KV) → random first token → loop N times,
/// feeding `single_token_argmax` and discarding the returned token. The
/// argmax path still encodes the lm_head matmul (same GPU graph as
/// production decode) but skips full-vocab logits readback. lcpp does not
/// even read the argmax i32; the residual difference is one i32 readback
/// per token, dwarfed by the per-token GPU work.
#[derive(Parser, Debug)]
struct TgArgs {
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Number of tokens to generate per timed rep.
    #[arg(short = 'n', long = "n-gen", default_value = "128")]
    n_gen: usize,
    /// Number of timed reps after warmup.
    #[arg(long, default_value = "3")]
    runs: usize,
    /// Skip the warmup pass.
    #[arg(long)]
    no_warmup: bool,
    /// Bench-only CPU/GPU overlap path: encode token N+1 while token N is
    /// executing on the GPU. Commands are still committed serially.
    #[arg(long)]
    pipelined: bool,
    /// Bench-only GDN front-projection overlap path.
    #[arg(long)]
    concurrent_gdn_proj: bool,
    /// Deterministic seed for random token selection.
    #[arg(long, default_value = "1")]
    seed: u64,
    /// `text` or `json` (`llama-bench -o json` shape).
    #[arg(short = 'o', long, value_enum, default_value = "text")]
    output: OutputFormat,
}

/// Synthetic multi-shape suite that keeps one loaded model resident.
///
/// Each reported row still uses a fresh sequence/session for each measured
/// repetition. This removes repeated process/model-load overhead without
/// changing the steady-state pp/tg semantics used by the single-shape commands.
#[derive(Parser, Debug)]
struct SuiteArgs {
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Prompt-only prefill shapes. Accepts repeated flags or comma lists.
    #[arg(long = "pp", value_delimiter = ',')]
    pp: Vec<usize>,
    /// Generation-only decode shapes. Accepts repeated flags or comma lists.
    #[arg(long = "tg", value_delimiter = ',')]
    tg: Vec<usize>,
    /// Number of timed reps after each row's optional warmup.
    #[arg(long, default_value = "1")]
    runs: usize,
    /// Skip each row's warmup pass.
    #[arg(long)]
    no_warmup: bool,
    /// Packed prefill chunk size for pp rows. If omitted, uses the model-aware
    /// default for each pp shape.
    #[arg(long)]
    prefill_chunk: Option<usize>,
    /// Deterministic seed for synthetic tokens.
    #[arg(long, default_value = "1")]
    seed: u64,
    /// `text` or `json` (`llama-bench -o json` shape). Defaults to JSON because
    /// suite output usually feeds scripts.
    #[arg(short = 'o', long, value_enum, default_value = "json")]
    output: OutputFormat,
}

#[derive(Parser, Debug)]
struct CtxSweepArgs {
    /// Path to a GGUF file.
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Comma-separated context checkpoints to measure at.
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "1,64,256,1024,4096,8192,16384"
    )]
    checkpoints: Vec<usize>,
    /// How many tokens to time at each checkpoint.
    #[arg(long, default_value = "5")]
    window: usize,
    /// Use the bench-only dense path that splits GDN blocks across encoders and
    /// runs the four front projections in a concurrent compute encoder.
    #[arg(long)]
    concurrent_gdn_proj: bool,
    /// Use the bench-only dense path that splits attention blocks across
    /// encoders and runs the q/k/v front projections in a concurrent compute
    /// encoder.
    #[arg(long)]
    concurrent_attn_proj: bool,
    /// Allocate a fresh right-sized session for each checkpoint instead of one
    /// max-capacity session for the whole sweep. Slower, but avoids large unused
    /// KV capacity poisoning earlier checkpoints on memory-pressure-sensitive
    /// models.
    #[arg(long)]
    fresh_per_checkpoint: bool,
    /// Warm each checkpoint with the production packed-prefill path instead of
    /// the token-by-token decode ramp (requires --fresh-per-checkpoint).
    #[arg(long)]
    prefill_warm: bool,
}

#[derive(Parser, Debug)]
struct PhaseArgs {
    /// Path to a GGUF file.
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Context length to profile at.
    #[arg(long, default_value = "4096")]
    ctx: usize,
}

#[derive(Parser, Debug)]
struct AttnIntraArgs {
    /// Path to a GGUF file.
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Context length to ramp before profiling one attention layer.
    #[arg(long, default_value = "32768")]
    ctx: usize,
    /// Timed single-layer repetitions after the ramp.
    #[arg(long, default_value = "3")]
    runs: usize,
    /// Optional absolute block index. Defaults to the first full-attention block.
    #[arg(long)]
    block: Option<usize>,
}

#[derive(Parser, Debug)]
struct GdnProjMicroArgs {
    /// Path to a GGUF file.
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Timed repetitions after warmup.
    #[arg(long, default_value = "20")]
    iters: usize,
    /// Untimed warmup repetitions.
    #[arg(long, default_value = "5")]
    warmup: usize,
    /// Synthetic token rows for the mat-mat batch path.
    #[arg(long, default_value = "1")]
    tokens: usize,
}

#[derive(Parser, Debug)]
struct DecodeProjBatchArgs {
    /// Path to a GGUF file.
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Comma-separated token counts to replay through batched mat-mat kernels.
    #[arg(long, value_delimiter = ',', default_value = "1,2,4,8,16")]
    tokens: Vec<usize>,
    /// Timed repetitions after warmup.
    #[arg(long, default_value = "10")]
    iters: usize,
    /// Untimed warmup repetitions.
    #[arg(long, default_value = "3")]
    warmup: usize,
}

#[derive(Parser, Debug)]
struct DecodeGdnLayerReplayArgs {
    /// Path to a GGUF file.
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Comma-separated token counts to replay through one GDN layer.
    #[arg(long, value_delimiter = ',', default_value = "1,2,4,8,16")]
    tokens: Vec<usize>,
    /// Optional absolute block index. Defaults to the first GDN block.
    #[arg(long)]
    block: Option<usize>,
    /// GDN-layer indexes to measure. Accepts repeated flags or comma lists.
    #[arg(long = "gdn-index", value_delimiter = ',')]
    gdn_indexes: Vec<usize>,
    /// Measure first, middle, and last GDN layers in one model load.
    #[arg(long)]
    sample_gdn_layers: bool,
    /// Timed repetitions after warmup.
    #[arg(long, default_value = "5")]
    iters: usize,
    /// Untimed warmup repetitions.
    #[arg(long, default_value = "2")]
    warmup: usize,
    /// Skip the all-slot correctness comparison between baseline and replay.
    #[arg(long)]
    no_check: bool,
}

#[derive(Parser, Debug)]
struct DecodeGdnChainReplayArgs {
    /// Path to a GGUF file.
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Comma-separated token counts to replay through the GDN chain.
    #[arg(long, value_delimiter = ',', default_value = "8,16")]
    tokens: Vec<usize>,
    /// First GDN-layer index in the chain.
    #[arg(long, default_value = "0")]
    start_gdn: usize,
    /// Number of consecutive GDN layers to chain.
    #[arg(long = "layers", default_value = "4")]
    n_layers: usize,
    /// Timed repetitions after warmup.
    #[arg(long, default_value = "3")]
    iters: usize,
    /// Untimed warmup repetitions.
    #[arg(long, default_value = "1")]
    warmup: usize,
    /// Skip the all-slot correctness comparison between baseline and replay.
    #[arg(long)]
    no_check: bool,
}

#[derive(Parser, Debug)]
struct DecodeBlockSliceReplayArgs {
    /// Path to a GGUF file.
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Comma-separated token counts to replay through the block slice.
    #[arg(long, value_delimiter = ',', default_value = "8,16")]
    tokens: Vec<usize>,
    /// First absolute transformer block index in the slice.
    #[arg(long, default_value = "0")]
    start_block: usize,
    /// Number of consecutive absolute blocks in the slice.
    #[arg(long = "blocks", default_value = "4")]
    n_blocks: usize,
    /// Synthetic decode position for attention blocks in the slice.
    #[arg(long, default_value = "0")]
    position: u32,
    /// Timed repetitions after warmup.
    #[arg(long, default_value = "3")]
    iters: usize,
    /// Untimed warmup repetitions.
    #[arg(long, default_value = "1")]
    warmup: usize,
    /// Skip the all-slot correctness comparison between baseline and replay.
    #[arg(long)]
    no_check: bool,
}

#[derive(Parser, Debug)]
struct DecodeBlockSliceTraceArgs {
    /// Path to a GGUF file.
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Number of synthetic slots to trace.
    #[arg(long, default_value = "8")]
    tokens: usize,
    /// First absolute transformer block index in the slice.
    #[arg(long, default_value = "0")]
    start_block: usize,
    /// Number of consecutive absolute blocks in the slice.
    #[arg(long = "blocks", default_value = "4")]
    n_blocks: usize,
    /// Synthetic decode position for attention blocks in the slice.
    #[arg(long, default_value = "0")]
    position: u32,
}

#[derive(Parser, Debug)]
struct DecodeBlockSliceMarginSweepArgs {
    /// Path to a GGUF file.
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Number of synthetic slots per traced window.
    #[arg(long, default_value = "8")]
    tokens: usize,
    /// Start blocks to sweep. If omitted, uses a non-overlapping stride.
    #[arg(long = "start-block", value_delimiter = ',')]
    start_blocks: Vec<usize>,
    /// Number of consecutive absolute blocks per window.
    #[arg(long = "blocks", default_value = "4")]
    n_blocks: usize,
    /// Synthetic decode positions to sweep.
    #[arg(long = "position", value_delimiter = ',', default_value = "0,4096")]
    positions: Vec<u32>,
}

#[derive(Parser, Debug)]
struct DecodeBlockSliceRealMarginArgs {
    /// Path to a GGUF file.
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Read one or more real prompt token streams from text files.
    #[arg(long)]
    file: Vec<PathBuf>,
    /// Number of consecutive prompt positions to use as slots.
    #[arg(long, default_value = "4")]
    tokens: usize,
    /// Slot counts to measure after preparing the maximum slot prefix set.
    #[arg(long = "slot-counts", value_delimiter = ',')]
    slot_counts: Vec<usize>,
    /// Prompt context positions to sweep.
    #[arg(long = "context", value_delimiter = ',', default_value = "512")]
    contexts: Vec<usize>,
    /// Position stride between slots when one prompt file supplies multiple slots.
    #[arg(long, default_value = "1")]
    stride: usize,
    /// Start blocks to sweep. If omitted, uses a non-overlapping stride.
    #[arg(long = "start-block", value_delimiter = ',')]
    start_blocks: Vec<usize>,
    /// Number of consecutive absolute blocks per replay window.
    #[arg(long = "blocks", default_value = "4")]
    n_blocks: usize,
    /// Timed repetitions for optional real-window economics. Zero keeps this as
    /// a margin-only probe.
    #[arg(long, default_value = "0")]
    timing_iters: usize,
    /// Untimed warmup repetitions for optional real-window economics.
    #[arg(long, default_value = "1")]
    timing_warmup: usize,
    /// Replay-margin threshold used by optional economics fallback modeling.
    #[arg(long, default_value = "0.0003")]
    margin_threshold: f32,
}

#[derive(Parser, Debug)]
struct DecodeMoeRouterRepackCheckArgs {
    /// Path to a GGUF file.
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Read one or more real prompt token streams from text files.
    #[arg(long)]
    file: Vec<PathBuf>,
    /// Number of prompt files to use as independent slots.
    #[arg(long, default_value = "4")]
    tokens: usize,
    /// Prompt context positions to sweep.
    #[arg(long = "context", value_delimiter = ',', default_value = "512")]
    contexts: Vec<usize>,
}

#[derive(Parser, Debug)]
struct MoeDownMicroArgs {
    /// Path to a GGUF file.
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Timed repetitions after warmup.
    #[arg(long, default_value = "20")]
    iters: usize,
    /// Untimed warmup repetitions.
    #[arg(long, default_value = "5")]
    warmup: usize,
    /// Synthetic tokens routed through the packed-slots kernel.
    #[arg(long, default_value = "1")]
    tokens: usize,
    /// Time the existing one-token fused Q4/Q5 routed FFN monolith.
    #[arg(long)]
    fused_routed_q4q5: bool,
    /// Capture per-layer route ids/weights at this decode context.
    #[arg(long)]
    route_capture_ctx: Option<usize>,
    /// Token-id pattern used for captured replay tokens.
    #[arg(long, value_enum, default_value = "zero")]
    route_capture_token_pattern: CaptureTokenPattern,
    /// Override routed-down K dimension with a synthetic zero Q5_K bank.
    #[arg(long)]
    synthetic_f_exp: Option<usize>,
    /// Override routed-down output dimension with a synthetic zero Q5_K bank.
    #[arg(long)]
    synthetic_h: Option<usize>,
    /// Number of synthetic layer dispatches. Defaults to the real Q5 layer count.
    #[arg(long)]
    synthetic_layers: Option<usize>,
    /// Force the f_exp=512 two-row-per-simdgroup Q5 down kernel.
    #[arg(long)]
    k512_r2: bool,
    /// Use the legacy f_exp=512 Q5 down kernel instead of the production R2 path.
    #[arg(long)]
    legacy_k512: bool,
    /// Compare default vs --k512-r2 output before timing.
    #[arg(long)]
    check_k512_r2: bool,
}

#[derive(Parser, Debug)]
struct MoeGateupMicroArgs {
    /// Path to a GGUF file.
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Timed repetitions after warmup.
    #[arg(long, default_value = "20")]
    iters: usize,
    /// Untimed warmup repetitions.
    #[arg(long, default_value = "5")]
    warmup: usize,
    /// Tokens to replay through the packed-slots kernel.
    #[arg(long, default_value = "1")]
    tokens: usize,
    /// Capture per-layer hidden activations and top-k ids at this decode context.
    #[arg(long)]
    route_capture_ctx: Option<usize>,
    /// Token-id pattern used for captured replay tokens.
    #[arg(long, value_enum, default_value = "zero")]
    route_capture_token_pattern: CaptureTokenPattern,
}

#[derive(Parser, Debug)]
struct MoeBatchSweepArgs {
    /// Path to a GGUF file.
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Timed repetitions after warmup.
    #[arg(long, default_value = "10")]
    iters: usize,
    /// Untimed warmup repetitions.
    #[arg(long, default_value = "3")]
    warmup: usize,
    /// Comma-separated token counts to replay.
    #[arg(long, value_delimiter = ',', default_value = "1,2,4,8,16")]
    tokens: Vec<usize>,
    /// Capture per-layer hidden activations and top-k ids at this decode context.
    #[arg(long, default_value = "1024")]
    route_capture_ctx: usize,
    /// Position stride between captured replay tokens.
    #[arg(long, default_value = "1")]
    route_capture_stride: usize,
    /// Read one or more real prompt token streams from text files.
    #[arg(long)]
    file: Vec<PathBuf>,
    /// Token-id pattern used for captured replay tokens.
    #[arg(long, value_enum, default_value = "ramp")]
    route_capture_token_pattern: CaptureTokenPattern,
    /// Route slot order for packed replay. The expert-sorted mode is a
    /// perf-only locality upper bound, not a correctness-preserving replay.
    #[arg(
        long = "slot-order",
        value_enum,
        value_delimiter = ',',
        default_value = "exact"
    )]
    slot_orders: Vec<MoeBatchSlotOrder>,
}

#[derive(Parser, Debug)]
struct RooflineArgs {
    /// Per-buffer stream size in MiB. Stream bytes/rep are nominally 3x this.
    #[arg(long, default_value = "512")]
    stream_mib: usize,
    /// Elements for the compute-loop kernel.
    #[arg(long, default_value = "4194304")]
    fma_elements: usize,
    /// FMA iterations per element. Nominal FLOPs are 2*N*iters.
    #[arg(long, default_value = "4096")]
    fma_iters: usize,
    /// Q4_K mat-mat input width. Must be divisible by 256.
    #[arg(long, default_value = "4096")]
    mat_in: usize,
    /// Q4_K mat-mat output width. Use multiples of 64 for the tuned path.
    #[arg(long, default_value = "4096")]
    mat_out: usize,
    /// Q4_K mat-mat batch/query rows. Use multiples of 64 for the tuned path.
    #[arg(long, default_value = "1024")]
    mat_query: usize,
    /// Timed repetitions after one warmup dispatch per kernel.
    #[arg(long, default_value = "5")]
    runs: usize,
    /// `text` or compact `json`.
    #[arg(short = 'o', long, value_enum, default_value = "text")]
    output: OutputFormat,
}

#[derive(Parser, Debug)]
struct DecodeWindowArgs {
    /// Path to a GGUF file.
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Context length to ramp to before waiting.
    #[arg(long)]
    target_ctx: usize,
    /// Warm the KV/GDN state with the production packed-prefill path instead
    /// of the token-by-token decode ramp. Orders of magnitude faster to deep
    /// contexts; validate against a decode-ramp point before trusting new
    /// context regimes (v0.494 validation: ctx16384 matches within noise).
    #[arg(long)]
    prefill_warm: bool,
    /// Number of decode tokens to execute after the go signal.
    #[arg(long, default_value = "128")]
    window: usize,
    /// Independent decode streams to issue from separate command queues.
    /// Intended only for occupancy/counter discrimination; each stream owns
    /// separate KV/GDN state while sharing resident model weights.
    #[arg(long, default_value = "1")]
    streams: usize,
    /// Use Metal timestamp counter samples around existing decode-stage
    /// encoder boundaries. Bench-only attribution probe for single-stream MoE.
    #[arg(long)]
    stage_timestamps: bool,
    /// Under --stage-timestamps, split attention mixer work from MoE route prep.
    /// This is an attribution-only second-level probe and changes encoder shape.
    #[arg(long)]
    stage_split_attn_route: bool,
    /// Under --stage-timestamps, split attention blocks into pre-norm, front
    /// projections, attention body/output, residual+post-norm, and route prep.
    #[arg(long)]
    stage_split_attn_detail: bool,
    /// Under --stage-timestamps, split the serial GDN after-projection bucket
    /// into beta/alpha prep, GDN tail, output projection, post-norm, and route.
    #[arg(long)]
    stage_split_gdn_after: bool,
    /// File created when the process has reached `target_ctx` and is waiting.
    #[arg(long)]
    ready_file: PathBuf,
    /// File whose existence releases the process to run the decode window.
    #[arg(long)]
    go_file: PathBuf,
    /// Use a bench-only pipelined dense decode loop that overlaps CPU encoding of
    /// token N+1 with GPU execution of token N.
    #[arg(long)]
    pipelined: bool,
    /// Use a bench-only dense decode path that splits GDN blocks across multiple
    /// encoders and runs the four front projections in a concurrent compute
    /// encoder.
    #[arg(long)]
    concurrent_gdn_proj: bool,
    /// Use a bench-only dense decode path that splits attention blocks across
    /// encoders and runs q/k/v front projections in a concurrent compute
    /// encoder.
    #[arg(long)]
    concurrent_attn_proj: bool,
}

#[derive(Parser, Debug)]
struct PrefixCacheArgs {
    /// Path to a GGUF file.
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Shared prefix prompt (used for cold prefill of request 1, then
    /// cached). Pad with --prefix-pad-tokens to hit a target prefix len.
    #[arg(short = 'p', long, default_value = "You are a helpful assistant.")]
    prefix: String,
    /// Optionally pad the prefix to a target token count by repeating
    /// "lorem ipsum" filler. Used to hit specific prefix lengths
    /// (per codex's H2 kill criteria: 64, 256, 1024, 4096).
    #[arg(long)]
    target_prefix_len: Option<usize>,
    /// Suffix prompt for request 2 (concatenated to the cached prefix).
    #[arg(long, default_value = "\n\nUser: What time is it?\nAssistant:")]
    suffix: String,
    /// Decode tokens to generate after each request's prefill.
    #[arg(long, default_value = "8")]
    tokens: usize,
    /// Prefill implementation used by the cache probe.
    #[arg(long, value_enum, default_value_t = PrefixCachePrefillMode::Packed)]
    prefill_mode: PrefixCachePrefillMode,
    /// Prefill implementation for the post-hit suffix.
    #[arg(long, value_enum, default_value_t = PrefixCacheSuffixMode::Auto)]
    suffix_prefill_mode: PrefixCacheSuffixMode,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum PrefixCachePrefillMode {
    /// Product-shaped packed prefill path.
    Packed,
    /// Legacy per-token loop, retained as a diagnostic control.
    Single,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum PrefixCacheSuffixMode {
    /// Use the per-token path for short suffixes and packed path otherwise.
    Auto,
    /// Force product-shaped packed prefill for the suffix.
    Packed,
    /// Force the per-token loop for the suffix.
    Single,
}

fn choose_prefix_cache_suffix_mode(
    mode: PrefixCacheSuffixMode,
    suffix_len: usize,
) -> PrefixCachePrefillMode {
    match mode {
        PrefixCacheSuffixMode::Auto if suffix_len <= 64 => PrefixCachePrefillMode::Single,
        PrefixCacheSuffixMode::Auto => PrefixCachePrefillMode::Packed,
        PrefixCacheSuffixMode::Packed => PrefixCachePrefillMode::Packed,
        PrefixCacheSuffixMode::Single => PrefixCachePrefillMode::Single,
    }
}

#[derive(Parser, Debug)]
struct MtpArgs {
    /// Path to an MTP-aware GGUF file.
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Prompt text. Use a non-trivial prompt for honest acceptance rates.
    #[arg(
        short = 'p',
        long,
        default_value = "The quick brown fox jumps over the lazy dog"
    )]
    prompt: String,
    /// Render the prompt through a Qwen chat template instead of treating
    /// `--prompt` as raw text. Useful for realistic thinking-mode evals.
    #[arg(long)]
    qwen_chat: bool,
    /// Optional system prompt for `--qwen-chat` rendering.
    #[arg(long)]
    system: Option<String>,
    /// For `--qwen-chat`, render the assistant generation prompt with an
    /// empty `<think>...</think>` block instead of an open thinking block.
    #[arg(long)]
    disable_thinking: bool,
    /// Experimental speculative depth. `1` is the original H4 lazy-verify
    /// path. `2..=15` use a bench-only MTP-N prototype that chains MTP
    /// drafts recursively and verifies them with the packed base path.
    #[arg(long, default_value = "1")]
    spec_tokens: usize,
    /// Bench-only probe for pricing native-MTP draft overhead.
    #[arg(long, value_enum, default_value_t = MtpProbeMode::Normal)]
    mtp_probe: MtpProbeMode,
    /// Physical packed-verify N for packed MTP probes. When larger than
    /// `1 + --spec-tokens`, padded positions are rolled back after the logical
    /// accept window.
    #[arg(long)]
    mtp_physical_n: Option<usize>,
    /// Chain all recursive MTP draft slots into one command buffer.
    #[arg(long)]
    mtp_single_cb_draft: bool,
    /// Use token_embd.weight as a cheap Q4 draft-only LM head. Bench falsifier;
    /// target verify still uses the real output.weight.
    #[arg(long)]
    mtp_draft_token_embd_head: bool,
    /// Quantize output.weight to Q4_1 at setup and use it as draft-only lm_head.
    /// Bench probe for MTPLX-style low-bit draft heads.
    #[arg(long)]
    mtp_draft_lm_head_q4_1: bool,
    /// Quantize output.weight to Q4_0 at setup and use it as draft-only lm_head.
    /// More aggressive bench probe for draft-head bandwidth/cost sensitivity.
    #[arg(long)]
    mtp_draft_lm_head_q4_0: bool,
    /// Quantize output.weight to affine Q4 group-size-64 for the draft lm_head.
    /// MTPLX-isomorphic bench probe; target verify still uses output.weight.
    #[arg(long)]
    mtp_draft_lm_head_q4_affine64: bool,
    /// Recursive MTP hidden fed into the next draft slot.
    #[arg(long, value_enum, default_value_t = MtpRecursiveHiddenArg::PostNorm)]
    mtp_recursive_hidden: MtpRecursiveHiddenArg,
    /// Base-model hidden variant fed into MTP prompt/bridge draft slots.
    #[arg(long, value_enum, default_value_t = MtpBaseHiddenArg::PostNorm)]
    mtp_base_hidden: MtpBaseHiddenArg,
    /// MTP KV history policy for packed native-MTP decode.
    #[arg(long, value_enum, default_value_t = MtpHistoryArg::Committed)]
    mtp_history: MtpHistoryArg,
    /// Write MTP target-rank rows as JSONL. This forces full draft-logit
    /// readback and is diagnostic-only, not a timing path.
    #[arg(long)]
    mtp_rank_topk: Option<PathBuf>,
    /// Write a compact JSON summary for MTPLX/profile-parity sweeps.
    #[arg(long)]
    output: Option<PathBuf>,
    /// Include exact prompt and serial target token IDs in `--output`.
    #[arg(long, requires = "output")]
    include_token_ids: bool,
    /// Number of tokens to generate after the prompt.
    #[arg(long, default_value = "64")]
    tokens: usize,
    /// Stop tokens for generation, comma-separated (e.g.
    /// `--stop-tokens 248046,248044`). When omitted, the stop set is
    /// resolved from the GGUF's declared `tokenizer.ggml.eos_token_id`
    /// (and `eot_token_id` if present) at runtime. There is no
    /// hardcoded fallback — a GGUF that declares no stops is an error.
    #[arg(long, value_parser = parse_stop_tokens)]
    stop_tokens: Option<Vec<i32>>,
    /// Skip the warmup pass.
    #[arg(long)]
    no_warmup: bool,
}

#[derive(Parser, Debug)]
struct PldArgs {
    /// Path to a target GGUF file.
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Prompt text.
    #[arg(short = 'p', long)]
    prompt: String,
    /// Render the prompt through a Qwen chat template.
    #[arg(long)]
    qwen_chat: bool,
    /// Optional system prompt for `--qwen-chat` rendering.
    #[arg(long)]
    system: Option<String>,
    /// Render an empty thinking block for `--qwen-chat`.
    #[arg(long)]
    disable_thinking: bool,
    /// Number of tokens to generate after the prompt.
    #[arg(long, default_value = "128")]
    tokens: usize,
    /// Stop tokens for generation, comma-separated.
    #[arg(long, value_parser = parse_stop_tokens)]
    stop_tokens: Option<Vec<i32>>,
    /// Skip the warmup pass.
    #[arg(long)]
    no_warmup: bool,
    /// Write a compact charged-path JSON summary.
    #[arg(long)]
    output: Option<PathBuf>,
    /// Include semantic per-step events in `--output`.
    #[arg(long, requires = "output")]
    trace_events: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum MtpProbeMode {
    /// Run the current native MTP path.
    Normal,
    /// Record current MTP draft vectors, then replay them without draft calls.
    ReplayCurrent,
    /// Replay recorded ids while running recursive MTP bodies without lm_head.
    BodyNoLmHead,
    /// Replay recorded ids while only maintaining MTP KV bridges.
    BridgeOnly,
    /// Use the no-spec greedy stream as a perfect draft oracle.
    Oracle,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum MtpRecursiveHiddenArg {
    /// Current qwen path: MTP residual stream before shared-head norm.
    PreNorm,
    /// MTPLX contract default: MTP shared-head-normalized hidden.
    PostNorm,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum MtpBaseHiddenArg {
    /// Current qwen path: base residual stream before final output norm.
    PreNorm,
    /// MTPLX contract default: base final-output-normalized hidden.
    PostNorm,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum MtpHistoryArg {
    /// Keep canonical MTP KV history for the committed target prefix.
    Committed,
    /// Keep accepted draft-chain KV without canonical target-hidden repair.
    DraftAccepted,
    /// Reset MTP KV each speculative step; keep only within-chain draft KV.
    Cycle,
}

impl From<MtpRecursiveHiddenArg> for MtpRecursiveHiddenVariant {
    fn from(value: MtpRecursiveHiddenArg) -> Self {
        match value {
            MtpRecursiveHiddenArg::PreNorm => Self::PreNorm,
            MtpRecursiveHiddenArg::PostNorm => Self::PostNorm,
        }
    }
}

impl From<MtpBaseHiddenArg> for MtpBaseHiddenVariant {
    fn from(value: MtpBaseHiddenArg) -> Self {
        match value {
            MtpBaseHiddenArg::PreNorm => Self::PreNorm,
            MtpBaseHiddenArg::PostNorm => Self::PostNorm,
        }
    }
}

impl From<MtpHistoryArg> for MtpHistoryMode {
    fn from(value: MtpHistoryArg) -> Self {
        match value {
            MtpHistoryArg::Committed => Self::Committed,
            MtpHistoryArg::DraftAccepted => Self::DraftAccepted,
            MtpHistoryArg::Cycle => Self::Cycle,
        }
    }
}

/// Parse a comma-separated list of i32 token ids for `--stop-tokens`.
/// Rejects empty input and non-numeric components; clap surfaces the
/// error inline with the flag name.
fn parse_stop_tokens(s: &str) -> Result<Vec<i32>, String> {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return Err("must contain at least one token id".into());
    }
    trimmed
        .split(',')
        .map(|part| {
            part.trim()
                .parse::<i32>()
                .map_err(|e| format!("invalid token id {part:?}: {e}"))
        })
        .collect()
}

/// Resolve the effective stop-token set: CLI override if provided,
/// otherwise the GGUF's declared set. Errors when the GGUF declares
/// nothing AND no override is given. No heuristic fallback — silent
/// defaults are exactly the bug this is fixing.
fn resolve_stop_tokens(
    g: &qwen_llm::gguf::GgufFile,
    override_set: Option<Vec<i32>>,
) -> Result<Vec<i32>> {
    if let Some(s) = override_set {
        return Ok(s);
    }
    g.stop_token_ids()
        .map_err(|e| anyhow!("resolving stop tokens from GGUF: {e}"))
}

fn render_qwen_single_turn_prompt(
    user_prompt: &str,
    system_prompt: Option<&str>,
    enable_thinking: bool,
) -> String {
    let mut out = String::new();
    if let Some(system) = system_prompt {
        if !system.is_empty() {
            out.push_str("<|im_start|>system\n");
            out.push_str(system);
            out.push_str("<|im_end|>\n");
        }
    }
    out.push_str("<|im_start|>user\n");
    out.push_str(user_prompt);
    out.push_str("<|im_end|>\n<|im_start|>assistant\n");
    if enable_thinking {
        out.push_str("<think>\n");
    } else {
        out.push_str("<think>\n\n</think>\n\n");
    }
    out
}

/// JSON schema version for `BenchRow`. Bump when fields are renamed,
/// removed, or have their semantics changed. Adding new optional fields
/// (always-null on old emitters) does NOT require a bump.
const BENCH_SCHEMA_VERSION: u32 = 2;

/// One bench result row. Field names match `llama-bench`'s JSON schema where
/// the meaning is the same; engine-specific fields are `Option<T>` and
/// serialized as explicit `null` (NOT omitted) so downstream consumers can
/// rely on a stable field set.
#[derive(Debug, Clone, serde::Serialize)]
struct BenchRow {
    schema_version: u32,
    engine: &'static str,
    build_commit: &'static str,
    /// `1` if source changes or hidden index flags were observed at build time
    /// or runtime.
    build_dirty: u8,
    build_identity: BuildIdentity,
    test_time: String,
    model_filename: String,
    model_size: u64,
    model_n_params: u64,
    arch_kind: &'static str,
    /// `pp<N>` or `tg<N>`, matching `llama-bench`'s shape vocabulary.
    test: String,
    n_tokens: usize,
    n_repetitions: usize,
    avg_ts: f64,
    stddev_ts: f64,
    samples_ts: Vec<f64>,
    samples_ns: Vec<u64>,
    avg_ns: u64,
    avg_compute_ns: Option<u64>,
    avg_session_alloc_ns: Option<u64>,
    avg_scratch_alloc_ns: Option<u64>,
    avg_gpu_ns: Option<u64>,
    kernel_trace_command_buffers_per_token: Option<f64>,
    kernel_trace_encoders_per_token: Option<f64>,
    kernel_trace_concurrent_encoders_per_token: Option<f64>,
    kernel_trace_dispatches_per_token: Option<f64>,
    /// Effective decode bandwidth (GB/s). `None` for `pp<N>` rows and for
    /// MoE `tg<N>` rows — MoE active-param accounting is out of scope here,
    /// and the naive `model_size × t/s` overstates by ~10x for MoE.
    decode_gb_per_s: Option<f64>,
    prefill_chunk: Option<usize>,
    decode_mode: Option<&'static str>,
    prefill_mode: Option<&'static str>,
    power: Option<PowerSnapshot>,
    qwen_env: std::collections::BTreeMap<String, String>,
}

#[derive(Debug, Clone, Default, serde::Serialize)]
struct PowerSnapshot {
    source: Option<String>,
    battery_percent: Option<u8>,
    battery_state: Option<String>,
    battery_warning: Option<String>,
    powermode_battery: Option<i32>,
    powermode_ac: Option<i32>,
    thermal_warning_recorded: Option<bool>,
    performance_warning_recorded: Option<bool>,
    cpu_power_status_recorded: Option<bool>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, clap::ValueEnum, Default)]
enum OutputFormat {
    #[default]
    Text,
    /// Suppresses stderr text so `qwen-bench ... -o json | jq` works clean.
    Json,
}

/// `YYYY-MM-DDTHH:MM:SSZ`, matching lcpp's `test_time` shape. Uses
/// Hinnant's days_from_civil so we don't pull in chrono.
fn utc_iso8601_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0) as i64;
    let days = secs.div_euclid(86_400);
    let sod = secs.rem_euclid(86_400);
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = mp + if mp < 10 { 3 } else { -9 };
    let year = y + if m <= 2 { 1 } else { 0 };
    let hour = sod / 3600;
    let minute = (sod % 3600) / 60;
    let second = sod % 60;
    format!("{year:04}-{m:02}-{d:02}T{hour:02}:{minute:02}:{second:02}Z")
}

/// Sum of weight-tensor bytes (matches lcpp's `model_size` = `llama_model_size`).
/// NOT file size — GGUF metadata and alignment padding are excluded so
/// derived bandwidth numbers are apples-to-apples with lcpp.
fn model_weight_bytes(g: &qwen_llm::gguf::GgufFile) -> u64 {
    g.tensors.iter().map(|t| t.n_bytes).sum()
}

fn capture_qwen_env() -> std::collections::BTreeMap<String, String> {
    std::env::vars()
        .filter(|(k, _)| k.starts_with("QWEN_"))
        .collect()
}

fn pmset_output(args: &[&str]) -> Option<String> {
    let out = std::process::Command::new("pmset")
        .args(args)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).into_owned())
}

fn parse_pmset_source(line: &str) -> Option<String> {
    let (_, rest) = line.split_once('\'')?;
    let (source, _) = rest.split_once('\'')?;
    Some(source.to_string())
}

fn parse_battery_percent(line: &str) -> Option<u8> {
    let pct_pos = line.find('%')?;
    let digits: String = line[..pct_pos]
        .chars()
        .rev()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    digits.parse::<u8>().ok()
}

fn parse_battery_state(line: &str) -> Option<String> {
    let mut parts = line.split(';').map(str::trim);
    let _battery_and_percent = parts.next()?;
    parts.next().map(str::to_string).filter(|s| !s.is_empty())
}

fn parse_pmset_custom_powermodes(raw: &str) -> (Option<i32>, Option<i32>) {
    #[derive(Clone, Copy)]
    enum Section {
        Battery,
        Ac,
    }
    let mut section = None;
    let mut battery = None;
    let mut ac = None;
    for line in raw.lines() {
        let trimmed = line.trim();
        match trimmed {
            "Battery Power:" => {
                section = Some(Section::Battery);
                continue;
            }
            "AC Power:" => {
                section = Some(Section::Ac);
                continue;
            }
            _ => {}
        }
        let mut fields = trimmed.split_whitespace();
        if fields.next() != Some("powermode") {
            continue;
        }
        let Some(value) = fields.next().and_then(|s| s.parse::<i32>().ok()) else {
            continue;
        };
        match section {
            Some(Section::Battery) => battery = Some(value),
            Some(Section::Ac) => ac = Some(value),
            None => {}
        }
    }
    (battery, ac)
}

fn note_no_warning(raw: &str, needle: &str) -> Option<bool> {
    if raw.contains(needle) {
        Some(false)
    } else if raw.trim().is_empty() {
        None
    } else {
        Some(true)
    }
}

fn capture_power_snapshot() -> Option<PowerSnapshot> {
    let batt = pmset_output(&["-g", "batt"]);
    let therm = pmset_output(&["-g", "therm"]);
    let custom = pmset_output(&["-g", "custom"]);
    if batt.is_none() && therm.is_none() && custom.is_none() {
        return None;
    }

    let mut snap = PowerSnapshot::default();
    if let Some(raw) = &batt {
        for line in raw.lines() {
            if line.starts_with("Now drawing from") {
                snap.source = parse_pmset_source(line);
            } else if line.contains("InternalBattery") {
                snap.battery_percent = parse_battery_percent(line);
                snap.battery_state = parse_battery_state(line);
            } else if let Some((_, warning)) = line.split_once("Battery Warning:") {
                snap.battery_warning = Some(warning.trim().to_string());
            }
        }
    }
    if let Some(raw) = &custom {
        let (battery, ac) = parse_pmset_custom_powermodes(raw);
        snap.powermode_battery = battery;
        snap.powermode_ac = ac;
    }
    if let Some(raw) = &therm {
        snap.thermal_warning_recorded =
            note_no_warning(raw, "No thermal warning level has been recorded");
        snap.performance_warning_recorded =
            note_no_warning(raw, "No performance warning level has been recorded");
        snap.cpu_power_status_recorded =
            note_no_warning(raw, "No CPU power status has been recorded");
    }
    Some(snap)
}

fn power_snapshot_summary(power: Option<&PowerSnapshot>) -> String {
    let Some(p) = power else {
        return "unavailable".to_string();
    };
    let battery = match (p.battery_percent, p.battery_state.as_deref()) {
        (Some(percent), Some(state)) => format!("{percent}% {state}"),
        (Some(percent), None) => format!("{percent}%"),
        (None, Some(state)) => state.to_string(),
        (None, None) => "unknown".to_string(),
    };
    format!(
        "source={} battery={} warning={} powermode_ac={} powermode_battery={} thermal_warning={} performance_warning={}",
        p.source.as_deref().unwrap_or("unknown"),
        battery,
        p.battery_warning.as_deref().unwrap_or("none"),
        p.powermode_ac.map_or("?".to_string(), |v| v.to_string()),
        p.powermode_battery
            .map_or("?".to_string(), |v| v.to_string()),
        p.thermal_warning_recorded
            .map_or("?".to_string(), |v| v.to_string()),
        p.performance_warning_recorded
            .map_or("?".to_string(), |v| v.to_string()),
    )
}

fn default_prefill_chunk(kind: qwen_llm::model::ArchKind, prompt_len: usize) -> usize {
    match kind {
        qwen_llm::model::ArchKind::Moe => prompt_len.clamp(1, 1024),
        qwen_llm::model::ArchKind::Dense => prompt_len.clamp(1, 1024),
    }
}

fn fresh_prefill_scratch_for_prompt(
    ctx: &MetalContext,
    model: &MetalModel,
    prefill_chunk: usize,
    prompt_len: usize,
) -> Result<MetalDFlashLayerMajorScratch> {
    let block_size = u32::try_from(prefill_chunk).context("prefill chunk does not fit u32")?;
    let matrix_max_pos = prompt_len.max(prefill_chunk);
    MetalDFlashLayerMajorScratch::fresh_prefill_with_matrix_max_pos(
        ctx,
        model,
        block_size,
        matrix_max_pos,
    )
    .context("prefill scratch")
}

#[derive(Clone, Debug, serde::Serialize)]
struct BuildIdentity {
    schema_version: u32,
    build_commit: String,
    build_commit_short: String,
    build_dirty: Option<bool>,
    build_source_state: Option<String>,
    stamp_source: String,
    stamp_error: Option<String>,
    runtime_commit: Option<String>,
    runtime_dirty: Option<bool>,
    runtime_source_state: Option<String>,
    status: String,
    problems: Vec<String>,
    overrides: Vec<String>,
}

#[derive(Clone, Copy, Debug, Default)]
struct BuildIdentityPolicy {
    allow_dirty: bool,
    allow_unverifiable: bool,
}

#[derive(Clone, Debug, Default)]
struct RuntimeGitIdentity {
    commit: Option<String>,
    dirty: Option<bool>,
    source_state: Option<String>,
}

static BUILD_IDENTITY: OnceLock<BuildIdentity> = OnceLock::new();
static BUILD_IDENTITY_POLICY: OnceLock<BuildIdentityPolicy> = OnceLock::new();

fn runtime_git_identity() -> RuntimeGitIdentity {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
    let commit = source_identity::git_text(repo, &["rev-parse", "HEAD"])
        .map(|value| value.to_ascii_lowercase())
        .filter(|value| source_identity::full_object_id(value));
    RuntimeGitIdentity {
        commit,
        dirty: source_identity::git_dirty(repo),
        source_state: source_identity::tracked_source_state(repo),
    }
}

fn classify_build_identity(
    build_commit: &str,
    build_dirty: Option<bool>,
    build_source_state: Option<&str>,
    stamp_source: &str,
    stamp_error: Option<&str>,
    runtime: RuntimeGitIdentity,
) -> BuildIdentity {
    let normalized_build = build_commit.to_ascii_lowercase();
    let mut problems = Vec::new();
    if !source_identity::full_object_id(&normalized_build) {
        problems.push("build_commit_unknown".to_string());
    }
    if build_dirty.is_none() {
        problems.push("build_dirty_unknown".to_string());
    }
    let normalized_build_state = build_source_state
        .filter(|state| source_identity::valid_source_state(state))
        .map(str::to_ascii_lowercase);
    if normalized_build_state.is_none() {
        problems.push("build_source_state_unknown".to_string());
    }
    if let Some(error) = stamp_error.filter(|error| *error != "none") {
        problems.push(error.to_string());
    }
    if runtime.commit.is_none() {
        problems.push("runtime_commit_unknown".to_string());
    }
    if runtime.dirty.is_none() {
        problems.push("runtime_dirty_unknown".to_string());
    }
    if runtime.source_state.is_none() {
        problems.push("runtime_source_state_unknown".to_string());
    }
    if source_identity::full_object_id(&normalized_build)
        && let Some(runtime_commit) = runtime.commit.as_deref()
        && normalized_build != runtime_commit
    {
        problems.push("commit_mismatch".to_string());
    }
    if let (Some(build_dirty), Some(runtime_dirty)) = (build_dirty, runtime.dirty)
        && build_dirty != runtime_dirty
    {
        problems.push("dirty_state_mismatch".to_string());
    }
    if let (Some(build_state), Some(runtime_state)) = (
        normalized_build_state.as_deref(),
        runtime.source_state.as_deref(),
    ) && build_state != runtime_state
    {
        problems.push("source_state_mismatch".to_string());
    }
    if build_dirty == Some(true) || runtime.dirty == Some(true) {
        problems.push("dirty".to_string());
    }

    let status = if problems
        .iter()
        .any(|problem| problem.ends_with("_mismatch"))
    {
        "mismatch"
    } else if problems.iter().any(|p| p != "dirty") {
        "unverifiable"
    } else if problems.iter().any(|p| p == "dirty") {
        "dirty"
    } else {
        "match"
    };
    let short = if source_identity::full_object_id(&normalized_build) {
        normalized_build[..9].to_string()
    } else {
        "unknown".to_string()
    };

    BuildIdentity {
        schema_version: 2,
        build_commit: normalized_build,
        build_commit_short: short,
        build_dirty,
        build_source_state: normalized_build_state,
        stamp_source: stamp_source.to_string(),
        stamp_error: stamp_error
            .filter(|error| *error != "none")
            .map(str::to_string),
        runtime_commit: runtime.commit,
        runtime_dirty: runtime.dirty,
        runtime_source_state: runtime.source_state,
        status: status.to_string(),
        problems,
        overrides: Vec::new(),
    }
}

fn qwen_build_identity_packet() -> &'static BuildIdentity {
    BUILD_IDENTITY.get_or_init(|| {
        let build_dirty = match env!("QWEN_BUILD_DIRTY") {
            "0" => Some(false),
            "1" => Some(true),
            _ => None,
        };
        classify_build_identity(
            env!("QWEN_BUILD_COMMIT"),
            build_dirty,
            Some(env!("QWEN_BUILD_SOURCE_STATE")),
            env!("QWEN_BUILD_STAMP_SOURCE"),
            Some(env!("QWEN_BUILD_STAMP_ERROR")),
            runtime_git_identity(),
        )
    })
}

fn validate_build_identity(identity: &BuildIdentity, policy: BuildIdentityPolicy) -> Result<()> {
    if identity.status == "mismatch" {
        return Err(anyhow!(
            "benchmark binary/source identity mismatch ({:?}): binary={} source={}; rebuild qwen-bench from the current checkout",
            identity.problems,
            identity.build_commit,
            identity.runtime_commit.as_deref().unwrap_or("unknown")
        ));
    }
    let unverifiable = identity.status == "unverifiable";
    if unverifiable && !policy.allow_unverifiable {
        return Err(anyhow!(
            "benchmark build identity is unverifiable ({:?}); rebuild in a Git checkout or pass --allow-unverifiable-build for a non-canonical run",
            identity.problems
        ));
    }
    let dirty = identity.problems.iter().any(|p| p == "dirty");
    if dirty && !policy.allow_dirty {
        return Err(anyhow!(
            "benchmark source/build is dirty; commit the changes or pass --allow-dirty for a non-canonical run"
        ));
    }
    Ok(())
}

fn recorded_build_identity() -> BuildIdentity {
    let mut identity = qwen_build_identity_packet().clone();
    let policy = BUILD_IDENTITY_POLICY.get().copied().unwrap_or_default();
    if policy.allow_dirty && identity.problems.iter().any(|p| p == "dirty") {
        identity.overrides.push("allow_dirty".to_string());
    }
    if policy.allow_unverifiable && identity.status == "unverifiable" {
        identity
            .overrides
            .push("allow_unverifiable_build".to_string());
    }
    identity
}

/// Legacy aliases retained for llama-bench-compatible row consumers.
fn qwen_build_identity() -> (&'static str, u8) {
    let identity = qwen_build_identity_packet();
    let dirty =
        u8::from(identity.build_dirty == Some(true) || identity.runtime_dirty == Some(true));
    (env!("QWEN_BUILD_COMMIT_SHORT"), dirty)
}

#[cfg(test)]
mod build_identity_tests {
    use super::*;

    const A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const STATE_A: &str = concat!(
        "git-source-sha256-v2:",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    );
    const STATE_B: &str = concat!(
        "git-source-sha256-v2:",
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    );

    fn identity(
        build_commit: &str,
        build_dirty: Option<bool>,
        runtime_commit: Option<&str>,
        runtime_dirty: Option<bool>,
    ) -> BuildIdentity {
        classify_build_identity(
            build_commit,
            build_dirty,
            Some(STATE_A),
            "test",
            None,
            RuntimeGitIdentity {
                commit: runtime_commit.map(str::to_string),
                dirty: runtime_dirty,
                source_state: Some(STATE_A.to_string()),
            },
        )
    }

    #[test]
    fn clean_matching_identity_passes() {
        let id = identity(A, Some(false), Some(A), Some(false));
        assert_eq!(id.status, "match");
        assert!(validate_build_identity(&id, BuildIdentityPolicy::default()).is_ok());
    }

    #[test]
    fn commit_mismatch_is_never_overridable() {
        let id = identity(A, Some(false), Some(B), Some(false));
        assert_eq!(id.status, "mismatch");
        assert!(
            validate_build_identity(
                &id,
                BuildIdentityPolicy {
                    allow_dirty: true,
                    allow_unverifiable: true,
                },
            )
            .is_err()
        );
    }

    #[test]
    fn dirty_identity_requires_explicit_override() {
        let id = identity(A, Some(true), Some(A), Some(true));
        assert_eq!(id.status, "dirty");
        assert!(validate_build_identity(&id, BuildIdentityPolicy::default()).is_err());
        assert!(
            validate_build_identity(
                &id,
                BuildIdentityPolicy {
                    allow_dirty: true,
                    allow_unverifiable: false,
                },
            )
            .is_ok()
        );
    }

    #[test]
    fn unknown_runtime_identity_requires_unverifiable_override() {
        let id = identity(A, Some(false), None, None);
        assert_eq!(id.status, "unverifiable");
        assert!(validate_build_identity(&id, BuildIdentityPolicy::default()).is_err());
        assert!(
            validate_build_identity(
                &id,
                BuildIdentityPolicy {
                    allow_dirty: false,
                    allow_unverifiable: true,
                },
            )
            .is_ok()
        );
    }

    #[test]
    fn invalid_compile_stamp_is_unverifiable() {
        let id = classify_build_identity(
            "unknown",
            None,
            None,
            "environment-invalid",
            Some("identity_override_triple_required"),
            RuntimeGitIdentity {
                commit: Some(A.to_string()),
                dirty: Some(false),
                source_state: Some(STATE_A.to_string()),
            },
        );
        assert_eq!(id.status, "unverifiable");
        assert!(
            id.problems
                .iter()
                .any(|p| p == "identity_override_triple_required")
        );
    }

    #[test]
    fn dirty_state_disagreement_is_never_overridable() {
        let id = identity(A, Some(false), Some(A), Some(true));
        assert_eq!(id.status, "mismatch");
        assert!(id.problems.iter().any(|p| p == "dirty_state_mismatch"));
        assert!(
            validate_build_identity(
                &id,
                BuildIdentityPolicy {
                    allow_dirty: true,
                    allow_unverifiable: true,
                },
            )
            .is_err()
        );
    }

    #[test]
    fn source_state_disagreement_is_never_overridable() {
        let id = classify_build_identity(
            A,
            Some(true),
            Some(STATE_A),
            "test",
            None,
            RuntimeGitIdentity {
                commit: Some(A.to_string()),
                dirty: Some(true),
                source_state: Some(STATE_B.to_string()),
            },
        );
        assert_eq!(id.status, "mismatch");
        assert!(id.problems.iter().any(|p| p == "source_state_mismatch"));
        assert!(
            validate_build_identity(
                &id,
                BuildIdentityPolicy {
                    allow_dirty: true,
                    allow_unverifiable: true,
                },
            )
            .is_err()
        );
    }
}

fn synthetic_prompt_ids(n: usize, vocab_size: u32, seed: u64) -> Vec<i32> {
    let mut state = if seed == 0 { 1 } else { seed };
    let vocab = vocab_size.max(1) as u64;
    let mut ids = Vec::with_capacity(n);
    for _ in 0..n {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ids.push((state % vocab) as i32);
    }
    ids
}

fn sample_mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn sample_stdev(xs: &[f64]) -> f64 {
    if xs.len() <= 1 {
        return 0.0;
    }
    let mean = sample_mean(xs);
    let variance = xs
        .iter()
        .map(|x| {
            let d = x - mean;
            d * d
        })
        .sum::<f64>()
        / (xs.len() - 1) as f64;
    variance.sqrt()
}

fn print_prefill_lowering_summary(mm: &MetalModel) {
    let mat_mat_gdn = |dtype: GgmlType| {
        matches!(
            dtype,
            GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K | GgmlType::Q8_0
        )
    };
    let mat_mat_attn = |dtype: GgmlType| {
        matches!(
            dtype,
            GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K | GgmlType::Q8_0
        )
    };
    let mat_mat_dense_ffn = |dtype: GgmlType| matches!(dtype, GgmlType::Q4_K | GgmlType::Q6_K);

    let mut gdn_total = 0usize;
    let mut gdn_batched = 0usize;
    let mut attn_total = 0usize;
    let mut attn_batched = 0usize;
    let mut dense_ffn_total = 0usize;
    let mut dense_ffn_batched = 0usize;
    let mut moe_total = 0usize;
    let mut moe_gpu_supported = 0usize;
    let mut first_gdn = None;
    let mut first_attn = None;
    let mut first_moe = None;

    for block in &mm.blocks {
        match block {
            MetalBlock::Gdn(g) => {
                gdn_total += 1;
                let gdn_ok = mat_mat_gdn(g.in_proj_qkv.dtype)
                    && mat_mat_gdn(g.in_proj_z.dtype)
                    && mat_mat_gdn(g.out_proj.dtype);
                if gdn_ok {
                    gdn_batched += 1;
                }
                first_gdn.get_or_insert(format!(
                    "qkv={:?} z={:?} out={:?}",
                    g.in_proj_qkv.dtype, g.in_proj_z.dtype, g.out_proj.dtype
                ));
                if let Some(moe) = &g.ffn_moe {
                    moe_total += 1;
                    let moe_ok = matches!(moe.gate_exps.dtype, GgmlType::Q4_K | GgmlType::Q5_K)
                        && moe.gate_exps.dtype == moe.up_exps.dtype
                        && matches!(moe.down_exps.dtype, GgmlType::Q5_K | GgmlType::Q6_K);
                    if moe_ok {
                        moe_gpu_supported += 1;
                    }
                    first_moe.get_or_insert(format!(
                        "gate={:?} up={:?} down={:?}",
                        moe.gate_exps.dtype, moe.up_exps.dtype, moe.down_exps.dtype
                    ));
                } else {
                    dense_ffn_total += 1;
                    if mat_mat_dense_ffn(g.ffn_gate.dtype)
                        && mat_mat_dense_ffn(g.ffn_up.dtype)
                        && mat_mat_dense_ffn(g.ffn_down.dtype)
                    {
                        dense_ffn_batched += 1;
                    }
                }
            }
            MetalBlock::Attn(a) => {
                attn_total += 1;
                let attn_ok = mat_mat_attn(a.q.dtype)
                    && mat_mat_attn(a.k.dtype)
                    && mat_mat_attn(a.v.dtype)
                    && mat_mat_attn(a.o.dtype);
                if attn_ok {
                    attn_batched += 1;
                }
                first_attn.get_or_insert(format!(
                    "q={:?} k={:?} v={:?} o={:?}",
                    a.q.dtype, a.k.dtype, a.v.dtype, a.o.dtype
                ));
                if let Some(moe) = &a.ffn_moe {
                    moe_total += 1;
                    let moe_ok = matches!(moe.gate_exps.dtype, GgmlType::Q4_K | GgmlType::Q5_K)
                        && moe.gate_exps.dtype == moe.up_exps.dtype
                        && matches!(moe.down_exps.dtype, GgmlType::Q5_K | GgmlType::Q6_K);
                    if moe_ok {
                        moe_gpu_supported += 1;
                    }
                    first_moe.get_or_insert(format!(
                        "gate={:?} up={:?} down={:?}",
                        moe.gate_exps.dtype, moe.up_exps.dtype, moe.down_exps.dtype
                    ));
                } else {
                    dense_ffn_total += 1;
                    if mat_mat_dense_ffn(a.ffn_gate.dtype)
                        && mat_mat_dense_ffn(a.ffn_up.dtype)
                        && mat_mat_dense_ffn(a.ffn_down.dtype)
                    {
                        dense_ffn_batched += 1;
                    }
                }
            }
        }
    }

    eprintln!(
        "[pp] lowering: gdn_batched={gdn_batched}/{gdn_total} attn_batched={attn_batched}/{attn_total} dense_ffn_batched={dense_ffn_batched}/{dense_ffn_total} moe_gpu_token_loop={moe_gpu_supported}/{moe_total}"
    );
    if let Some(s) = first_gdn {
        eprintln!("[pp] dtype sample gdn: {s}");
    }
    if let Some(s) = first_attn {
        eprintln!("[pp] dtype sample attn: {s}");
    }
    if let Some(s) = first_moe {
        eprintln!("[pp] dtype sample moe: {s}");
    }
}

#[derive(Parser, Debug)]
struct DflashLazyArgs {
    /// Path to the target GGUF (e.g. Qwen3.6-27B-Q4_K_M.gguf).
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Path to the DFlash drafter GGUF (e.g.
    /// spiritbuun/Qwen3.6-27B-DFlash-GGUF / dflash-draft-3.6-q8_0.gguf).
    #[arg(long)]
    drafter: PathBuf,
    /// Prompt text. Use a meaningful prompt for honest acceptance rates.
    #[arg(
        short = 'p',
        long,
        default_value = "The quick brown fox jumps over the lazy dog"
    )]
    prompt: String,
    /// Number of tokens to generate after the prompt.
    #[arg(long, default_value = "32")]
    tokens: usize,
    /// Stop tokens for generation, comma-separated. When omitted, the
    /// stop set is resolved from the GGUF's declared
    /// `tokenizer.ggml.eos_token_id` (and `eot_token_id` if present).
    #[arg(long, value_parser = parse_stop_tokens)]
    stop_tokens: Option<Vec<i32>>,
    /// Effective-N: only consider the first M draft positions per outer
    /// step (1 ≤ M ≤ block_size - 1). Reveals where α decays in the
    /// block; if α at M=8 is close to α at M=15, larger N is just paying
    /// for verify cost without recovering tokens. M=0 means use the
    /// full block_size - 1 from the GGUF.
    #[arg(long, default_value = "0")]
    effective_n: usize,
    /// Skip the warmup pass.
    #[arg(long)]
    no_warmup: bool,
    /// **T0 (Program T)**: record, for every prefix-conditioned draft
    /// position (accepted path + first mismatch), the RANK of target's
    /// argmax in the drafter's logits at that position. Emits a JSONL
    /// artifact for the comb-tree acceptance optimizer plus a p_k(depth)
    /// table (k in 1/2/4/8/16). Uses draft_block_with_logits (slower;
    /// measurement-only).
    #[arg(long)]
    rank_topk: Option<PathBuf>,
    /// **T0b tree-sim**: simulate ONE-block tree decode exactly (static
    /// topology: chain depth D plus rank<=B sibling sets at the first R
    /// depths; DFlash's block drafter makes deeper rows path-independent,
    /// so rescued paths keep verifying against the SAME block). Emitted
    /// stream remains target-greedy by construction. Requires --rank-topk.
    #[arg(long)]
    tree_sim: bool,
    /// Tree-sim chain depth D (node budget = D + R*(B-1) must be <= 15).
    #[arg(long, default_value = "6")]
    tree_chain_d: usize,
    /// Tree-sim sibling-set count R (rescues allowed at depths 0..R).
    #[arg(long, default_value = "3")]
    tree_sibling_depths: usize,
    /// Tree-sim sibling branching B (rescue taken when rank <= B).
    #[arg(long, default_value = "4")]
    tree_b: usize,
}

#[derive(Parser, Debug)]
struct DflashArgs {
    /// Path to the target GGUF (e.g. Qwen3.6-27B-Q4_K_M.gguf).
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Path to the DFlash drafter GGUF.
    #[arg(long)]
    drafter: PathBuf,
    /// Prompt text. Use a meaningful prompt for honest acceptance rates.
    #[arg(
        short = 'p',
        long,
        default_value = "The quick brown fox jumps over the lazy dog"
    )]
    prompt: String,
    /// Number of tokens to generate after the prompt.
    #[arg(long, default_value = "64")]
    tokens: usize,
    /// Stop tokens for generation, comma-separated. When omitted, the
    /// stop set is resolved from the GGUF's declared
    /// `tokenizer.ggml.eos_token_id` (and `eot_token_id` if present).
    #[arg(long, value_parser = parse_stop_tokens)]
    stop_tokens: Option<Vec<i32>>,
    /// Skip the warmup pass.
    #[arg(long)]
    no_warmup: bool,
    /// Skip the equivalence check vs DFlash=off baseline (saves ~1×
    /// gen-time on the same prompt). Default: ON, because the bench
    /// is also a correctness gate.
    #[arg(long)]
    skip_equivalence_check: bool,
    /// **v0.72.3**: enable lightweight per-phase GPU timers in
    /// draft_block (phase1_ctx_fc_norm, phase2_embed,
    /// phase2_proj_norm_rope ×n_layer, phase3_attn_oproj_ffn_residuals
    /// ×n_layer, phase4_tail). Aggregated across all outer steps and
    /// reported at end. Used to confirm v0.72.4+ leverage map.
    #[arg(long)]
    profile: bool,
    /// **v0.76**: verify-chain length policy. One of:
    /// `adaptive` (default; ctx-keyed schedule with Off-terminal),
    /// `static-16` / `static-8` / `static-4` (fixed N, no Off ramp),
    /// `off` (no speculation; single_token decode loop). The static
    /// modes exist for the calibration sweep + as A/B comparators
    /// against `adaptive`. `static-16` matches pre-v0.76 behavior.
    /// Current `adaptive` tuning is calibrated on M4 Max + 27B Q4_K_M
    /// code-prompt sweeps; treat it as a heuristic outside that regime.
    #[arg(long, default_value = "adaptive")]
    n_policy: String,
}

#[derive(Parser, Debug)]
struct VocabAuditArgs {
    /// Path to a GGUF file.
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Optional path to a prompt corpus (one prompt per line; empty
    /// lines and lines starting with '#' ignored). If absent, uses a
    /// built-in mixed-category corpus.
    #[arg(long)]
    prompts: Option<PathBuf>,
    /// Number of greedy-decode tokens per prompt.
    #[arg(long, default_value = "32")]
    tokens: usize,
    /// Comma-separated K thresholds to evaluate (vocab-prune sizes).
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "1024,4096,8192,16384,32768,49152,65536,98304"
    )]
    ks: Vec<usize>,
    /// Show this many "out of K" decoded tokens per K (for inspection).
    #[arg(long, default_value = "5")]
    show_examples: usize,
}

#[derive(Parser, Debug)]
struct TokArgs {
    /// Path to a Qwen 3.5 / 3.6 GGUF file.
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Prompt text. If absent, uses a fixed mixed tokenizer stress prompt.
    #[arg(short = 'p', long, conflicts_with_all = ["file", "messages"])]
    prompt: Option<String>,
    /// Read prompt text from a file.
    #[arg(long, conflicts_with = "messages")]
    file: Option<PathBuf>,
    /// Render a JSON messages input into a Qwen chat-template prompt.
    ///
    /// Accepted shapes:
    /// - bare `[{ role, content }, ...]`
    /// - wrapped `{ messages: [...], ... }`
    #[arg(long)]
    messages: Option<PathBuf>,
    /// Use only the first N messages from `--messages` before rendering.
    #[arg(long)]
    messages_max: Option<usize>,
    /// Preserve assistant `<think>...</think>` history from `--messages`.
    /// By default the bench preserves thinking only for wrapped Qwen3.6
    /// rollouts and strips it otherwise.
    #[arg(long)]
    messages_preserve_thinking: bool,
    /// Force stripping assistant `<think>...</think>` history from
    /// `--messages`, even if auto-detection would preserve it.
    #[arg(long, conflicts_with = "messages_preserve_thinking")]
    messages_strip_thinking: bool,
    /// Do not append a final `<|im_start|>assistant\n` generation marker for
    /// `--messages` prompts.
    #[arg(long)]
    messages_no_generation_prompt: bool,
    /// Timed encode/decode iterations for each backend.
    #[arg(long, default_value = "1000")]
    iters: usize,
    /// Pass add_special=true to both tokenizer backends.
    #[arg(long)]
    add_special: bool,
}

#[derive(Parser, Debug)]
struct AttnPrefillMicroArgs {
    /// Path to a GGUF file.
    #[arg(
        short = 'm',
        long,
        default_value = "/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
    )]
    model: PathBuf,
    /// Absolute position of the first packed query row.
    #[arg(long, default_value = "16384")]
    base_pos: usize,
    /// Number of packed query rows.
    #[arg(long, default_value = "128")]
    rows: usize,
    /// Split-K partitions.
    #[arg(long, default_value = "64")]
    nwg: usize,
    /// Query rows processed per packed main-pass threadgroup.
    #[arg(long, default_value = "2")]
    qt: usize,
}

#[derive(Parser, Debug)]
struct AttnFrontMicroArgs {
    /// Path to a GGUF file.
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Number of packed rows.
    #[arg(long, default_value = "1024")]
    rows: usize,
}

#[derive(Parser, Debug)]
struct AttnLayerMicroArgs {
    /// Path to a GGUF file.
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Absolute position of the first packed query row.
    #[arg(long, default_value = "16384")]
    base_pos: usize,
    /// Number of packed query rows.
    #[arg(long, default_value = "4")]
    rows: usize,
    /// Forced split-K partitions for both baseline and packed body.
    #[arg(long, default_value = "64")]
    nwg: usize,
    /// Query rows processed per packed main-pass threadgroup.
    #[arg(long, default_value = "2")]
    qt: usize,
}

#[derive(Parser, Debug)]
struct PpFfnAbArgs {
    /// Path to a GGUF file.
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Synthetic prompt token count.
    #[arg(short = 'p', long, default_value = "4096")]
    n_prompt: usize,
    /// Packed prefill chunk size. If omitted, uses the model-aware default.
    #[arg(long)]
    prefill_chunk: Option<usize>,
    /// Number of base/fused pairs. Odd pairs run base->fused; even pairs reverse.
    #[arg(long, default_value = "2")]
    pairs: usize,
    /// Skip the unmeasured base and fused warmup passes.
    #[arg(long)]
    no_warmup: bool,
    /// Deterministic seed for synthetic token generation.
    #[arg(long, default_value = "1")]
    seed: u64,
}

#[derive(Parser, Debug)]
struct PpWaitArgs {
    /// Path to a GGUF file.
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Synthetic prompt token count.
    #[arg(short = 'p', long, default_value = "320")]
    n_prompt: usize,
    /// Packed prefill chunk size. If omitted, uses the model-aware default.
    #[arg(long)]
    prefill_chunk: Option<usize>,
    /// Include final norm + lm_head + logits readback.
    #[arg(long)]
    with_tail: bool,
    /// Deterministic seed for synthetic token generation.
    #[arg(long, default_value = "1")]
    seed: u64,
    /// Skip the warmup prefill pass before signaling ready.
    #[arg(long)]
    no_warmup: bool,
    /// File written once the model is loaded and warmup is complete.
    #[arg(long)]
    ready_file: PathBuf,
    /// File whose appearance triggers the timed run.
    #[arg(long)]
    go_file: PathBuf,
    /// Output format for the final timed run.
    #[arg(short = 'o', long, default_value = "json")]
    output: OutputFormat,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();
    let policy = BuildIdentityPolicy {
        allow_dirty: args.allow_dirty,
        allow_unverifiable: args.allow_unverifiable_build,
    };
    let _ = BUILD_IDENTITY_POLICY.set(policy);
    if !matches!(&args.cmd, Cmd::BuildInfo(_)) {
        validate_build_identity(qwen_build_identity_packet(), policy)?;
    }
    match args.cmd {
        Cmd::AttnCapture(a) => {
            attn_capture::run(a, serde_json::to_value(qwen_build_identity_packet())?)
        }
        Cmd::AttnStageFloor(a) => {
            attn_stage_floor::run(a, serde_json::to_value(qwen_build_identity_packet())?)
        }
        Cmd::BuildInfo(a) => run_build_info(a),
        Cmd::PrefixCache(a) => run_prefix_cache(a),
        Cmd::VocabAudit(a) => run_vocab_audit(a),
        Cmd::Decode(a) => run_decode(a),
        Cmd::Pp(a) => run_pp(a),
        Cmd::Tg(a) => run_tg(a),
        Cmd::Suite(a) => run_suite(a),
        Cmd::CtxSweep(a) => run_ctx_sweep(a),
        Cmd::Phase(a) => run_phase(a),
        Cmd::AttnIntra(a) => run_attn_intra(a),
        Cmd::GdnProjMicro(a) => run_gdn_proj_micro(a),
        Cmd::DecodeProjBatch(a) => run_decode_proj_batch(a),
        Cmd::DecodeGdnLayerReplay(a) => run_decode_gdn_layer_replay(a),
        Cmd::DecodeGdnChainReplay(a) => run_decode_gdn_chain_replay(a),
        Cmd::DecodeBlockSliceReplay(a) => run_decode_block_slice_replay(a),
        Cmd::DecodeBlockSliceTrace(a) => run_decode_block_slice_trace(a),
        Cmd::DecodeBlockSliceMarginSweep(a) => run_decode_block_slice_margin_sweep(a),
        Cmd::DecodeBlockSliceRealMargin(a) => run_decode_block_slice_real_margin(a),
        Cmd::DecodeMoeRouterRepackCheck(a) => run_decode_moe_router_repack_check(a),
        Cmd::MoeDownMicro(a) => run_moe_down_micro(a),
        Cmd::MoeGateupMicro(a) => run_moe_gateup_micro(a),
        Cmd::MoeBatchSweep(a) => run_moe_batch_sweep(a),
        Cmd::Roofline(a) => run_roofline(a),
        Cmd::MetalCounters(a) => run_metal_counters(a),
        Cmd::MetalPipelines(a) => run_metal_pipelines(a),
        Cmd::TopologyProbe(a) => run_topology_probe(a),
        Cmd::DispatchCensus(a) => run_dispatch_census(a),
        Cmd::DecodeWindow(a) => run_decode_window(a),
        Cmd::Mtp(a) => run_mtp(a),
        Cmd::Pld(a) => run_pld(a),
        Cmd::DflashLazy(a) => run_dflash_lazy(a),
        Cmd::Dflash(a) => run_dflash(a),
        Cmd::Tok(a) => run_tok(a),
        Cmd::AttnPrefillMicro(a) => run_attn_prefill_micro(a),
        Cmd::AttnFrontMicro(a) => run_attn_front_micro(a),
        Cmd::AttnLayerMicro(a) => run_attn_layer_micro(a),
        Cmd::PpFfnAb(a) => run_pp_ffn_ab(a),
        Cmd::PpWait(a) => run_pp_wait(a),
    }
}

fn run_build_info(args: BuildInfoArgs) -> Result<()> {
    let identity = qwen_build_identity_packet();
    match args.output {
        OutputFormat::Json => println!("{}", serde_json::to_string_pretty(identity)?),
        OutputFormat::Text => {
            println!("status\t{}", identity.status);
            println!("build_commit\t{}", identity.build_commit);
            println!(
                "runtime_commit\t{}",
                identity.runtime_commit.as_deref().unwrap_or("unknown")
            );
            println!(
                "build_source_state\t{}",
                identity.build_source_state.as_deref().unwrap_or("unknown")
            );
            println!(
                "runtime_source_state\t{}",
                identity
                    .runtime_source_state
                    .as_deref()
                    .unwrap_or("unknown")
            );
            println!(
                "dirty\tbuild={} runtime={}",
                identity
                    .build_dirty
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "unknown".to_string()),
                identity
                    .runtime_dirty
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "unknown".to_string())
            );
            println!("stamp_source\t{}", identity.stamp_source);
            println!("problems\t{}", identity.problems.join(","));
        }
    }
    Ok(())
}

fn run_metal_counters(_args: MetalCountersArgs) -> Result<()> {
    let ctx = MetalContext::new()?;
    let caps = ctx.counter_capabilities();
    println!("device\t{}", ctx.describe());
    println!(
        "sampling\tstage={}\tdispatch={}\tblit={}",
        caps.supports_stage_boundary, caps.supports_dispatch_boundary, caps.supports_blit_boundary
    );
    println!("counter_sets\t{}", caps.sets.len());
    for set in &caps.sets {
        println!(
            "set\t{}\tcounters={}\tsample_buffer={}",
            set.name,
            set.counters.len(),
            set.sample_buffer_status
        );
        for counter in &set.counters {
            println!("counter\t{}\t{}", set.name, counter);
        }
    }
    Ok(())
}

const HOT_DECODE_PIPELINE_AUDIT: &[&str] = &[
    "kernel_attn_decode_v4_g8_t4_c64_f32",
    "kernel_attn_decode_v4_g8_t4_c128_f32",
    "kernel_attn_decode_v4_g8_t2_c64_f32",
    "kernel_attn_decode_v4_g8_t2_c128_f32",
    "kernel_attn_decode_v4_g16_t4_c64_f32",
    "kernel_attn_decode_v4_g16_t4_c128_f32",
    "kernel_attn_decode_v4_reduce_h2_g8_f32",
    "kernel_attn_decode_v4_reduce_h2_g16_f32",
    "kernel_gdn_prep_parallel_state_f32",
    "kernel_gdn_decay_chain_f32",
    "kernel_l2_norm_pair_hd128_r4_f32",
    "kernel_gdn_step_decay_f32",
    "kernel_gdn_step_decay_packed_f32",
    "kernel_gdn_step_decay_packed_nsg4_f32",
    "kernel_rmsnorm_gated_hd128_r4_f32",
    "kernel_mat_vec_q4_K_f32",
    "kernel_mat_vec_q5_K_f32",
    "kernel_mat_vec_q6_K_f32",
    "kernel_mat_vec_q8_0_f32",
    "kernel_moe_swiglu_q4_K_f32_grouped_slots_n16",
    "kernel_moe_swiglu_q5_K_f32_grouped_slots_n16",
    "kernel_moe_down_q5_K_f32_grouped_slots",
    "kernel_moe_down_q5_K_f32_grouped_slots_tiny8_r16",
    "kernel_moe_down_q6_K_f32_grouped_slots",
    "kernel_moe_down_weighted_sum_q5_K_f32_packed_slots_k512_r2",
    "kernel_sigmoid_mul_gate_strided_f32",
    "kernel_argmax_f32",
];

fn run_metal_pipelines(args: MetalPipelinesArgs) -> Result<()> {
    let ctx = MetalContext::new()?;
    let names: Vec<String> = if args.kernels.is_empty() {
        HOT_DECODE_PIPELINE_AUDIT
            .iter()
            .map(|s| s.to_string())
            .collect()
    } else {
        args.kernels
    };
    println!("kernel\tthread_width\tmax_threads_per_tg\tstatic_tg_mem\ticb");
    for name in names {
        match ctx.pipeline_info(&name) {
            Ok(info) => println!(
                "{}\t{}\t{}\t{}\t{}",
                info.name,
                info.thread_execution_width,
                info.max_total_threads_per_threadgroup,
                info.static_threadgroup_memory_length,
                info.supports_indirect_command_buffers
            ),
            Err(e) => eprintln!("missing\t{name}\t{e}"),
        }
    }
    Ok(())
}

// ===========================================================================
// B0 topology probe (Program B gate 0). Design + pre-registered gates:
// docs/bench/2026-07-05-b0-topology-probe/README.md (cx session 019f347b-c).
// ===========================================================================

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct TpParams {
    spin_iters: u32,
    grid_tgs: u32,
    stages: u32,
    epochs: u32,
    cadence_iters: u32,
    spin_budget: u32,
    traffic_mask: u32,
    sample_every: u32,
    n_elems: u32,
}

const TP_RING_SAMPLES: usize = 64; // must match TP_LAT_SAMPLES in the kernel
const TP_LAT_UNUSED: u32 = 0xFFFF_FFFF;
const TP_GPU_CORES: f64 = 40.0; // M4 Max

struct TpCal {
    fma_ns: f64,
    poll_ns: f64,
}

type TpBuffer = Retained<ProtocolObject<dyn MTLBuffer>>;

fn tp_zero(buf: &TpBuffer, n_bytes: usize) {
    unsafe { std::ptr::write_bytes(buf.contents().as_ptr() as *mut u8, 0, n_bytes) }
}

fn tp_fill_u32(buf: &TpBuffer, n: usize, v: u32) {
    let p = buf.contents().as_ptr() as *mut u32;
    for i in 0..n {
        unsafe { p.add(i).write(v) }
    }
}

fn tp_read_u32(buf: &TpBuffer, n: usize) -> Vec<u32> {
    let p = buf.contents().as_ptr() as *const u32;
    (0..n).map(|i| unsafe { p.add(i).read() }).collect()
}

fn tp_read_f32_bits(buf: &TpBuffer, n: usize) -> Vec<u32> {
    tp_read_u32(buf, n)
}

/// One serial-encoder command buffer; returns GPU ms.
fn tp_run_timed<F>(ctx: &MetalContext, encode: F) -> Result<f64>
where
    F: FnOnce(&KernelEncoder) -> Result<()>,
{
    let cmd = ctx.queue.commandBuffer().context("tp cmd")?;
    let enc = KernelEncoder::begin(&cmd);
    encode(&enc)?;
    enc.end();
    cmd.commit();
    cmd.waitUntilCompleted();
    Ok((cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3)
}

fn tp_dispatch(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    kernel: &str,
    params: &TpParams,
    buffers: &[&TpBuffer],
    tgs: usize,
    w: usize,
) -> Result<()> {
    let pso = ctx.pipeline(kernel)?;
    anyhow::ensure!(
        pso.maxTotalThreadsPerThreadgroup() >= w,
        "{kernel}: maxTotalThreadsPerThreadgroup {} < requested width {w} \
         (itself a residency datum; record and skip)",
        pso.maxTotalThreadsPerThreadgroup()
    );
    enc.set_pipeline(&pso);
    enc.set_bytes(0, params);
    for (i, b) in buffers.iter().enumerate() {
        enc.set_buffer(i + 1, b, 0);
    }
    enc.dispatch(
        MTLSize {
            width: tgs,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: w,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

fn tp_calibrate(ctx: &MetalContext, seed: &TpBuffer) -> Result<TpCal> {
    // FMA chain rate: 1 TG x 32 threads, dependent chain, min of 3.
    let out = ctx.buffer_uninit(32 * 4)?;
    let iters = 1u32 << 22;
    let mut fma_ms = f64::INFINITY;
    for _ in 0..3 {
        let p = TpParams {
            spin_iters: iters,
            ..Default::default()
        };
        let ms = tp_run_timed(ctx, |enc| {
            tp_dispatch(ctx, enc, "tp_calibrate", &p, &[seed, &out], 1, 32)
        })?;
        fma_ms = fma_ms.min(ms);
    }
    // Poll rate: 1 TG x 32 threads polling a never-changing atomic.
    let flag = ctx.buffer_uninit(4)?;
    tp_zero(&flag, 4);
    let pout = ctx.buffer_uninit(32 * 4)?;
    let polls = 1u32 << 22;
    let mut poll_ms = f64::INFINITY;
    for _ in 0..3 {
        let p = TpParams {
            spin_budget: polls,
            ..Default::default()
        };
        let ms = tp_run_timed(ctx, |enc| {
            tp_dispatch(ctx, enc, "tp_poll_calibrate", &p, &[&flag, &pout], 1, 32)
        })?;
        poll_ms = poll_ms.min(ms);
    }
    Ok(TpCal {
        fma_ns: fma_ms * 1e6 / iters as f64,
        poll_ns: poll_ms * 1e6 / polls as f64,
    })
}

fn tp_percentile_u32(sorted: &[u32], q: f64) -> u32 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

struct TpArmROut {
    rows: Vec<serde_json::Value>,
    /// (variant, traffic, w) -> (max_alive at longest dwell, steady p10,
    /// plateaued?)
    caps: BTreeMap<(String, bool, usize), (u32, u32, bool)>,
}

fn tp_arm_r(
    ctx: &MetalContext,
    cal: &TpCal,
    seed: &TpBuffer,
    traffic: &TpBuffer,
    traffic_mask: u32,
    quick: bool,
) -> Result<TpArmROut> {
    let g_tgs: usize = if quick { 2560 } else { 10240 };
    let dwells_us_full: &[f64] = if quick {
        &[200.0, 1000.0]
    } else {
        &[200.0, 1000.0, 5000.0]
    };
    // traffic rows are qualitative (does residency change under load?) and
    // memory-slow; cap their dwell sweep to keep the budget sane
    let dwells_us_tr: &[f64] = &[200.0, 1000.0];
    let widths: &[usize] = &[32, 64, 128, 256];
    let variants: &[(&str, u32, bool)] = &[
        ("tp_residency_lo", 1, false),
        ("tp_residency_hi8", 8, false),
        ("tp_residency_hi32", 32, false),
        ("tp_residency_hi64", 64, false),
        ("tp_residency_lo_tr", 1, true),
        ("tp_residency_hi8_tr", 8, true),
        ("tp_residency_hi32_tr", 32, true),
        ("tp_residency_hi64_tr", 64, true),
    ];

    let census = ctx.buffer_uninit(8)?;
    let entry = ctx.buffer_uninit(g_tgs * 4)?;
    let out = ctx.buffer_uninit(g_tgs * 4)?;
    let dummy = ctx.buffer_uninit(4)?;
    tp_zero(&dummy, 4);

    let mut rows = Vec::new();
    let mut caps = BTreeMap::new();
    println!("\n== arm R: residency census (grid {g_tgs} TGs) ==");
    println!("variant\ttraffic\tW\tdwell_us\tmax_alive\tper_core\tsteady_p10\twaves\tgpu_ms");
    for &(name, _nacc, tr) in variants {
        // EMPIRICAL per-variant iteration cost (1 TG, uncontended): the
        // NACC accumulator chains pipeline, so scaling fma_ns by NACC
        // over-sizes dwell by up to ~8x (smoke finding). Traffic variants
        // are memory-dominated and calibrate much slower.
        let iter_ns = {
            let cal_spin: u32 = if tr { 1 << 13 } else { 1 << 16 };
            let p = TpParams {
                spin_iters: cal_spin,
                grid_tgs: 1,
                traffic_mask: if tr { traffic_mask } else { 0 },
                ..Default::default()
            };
            tp_zero(&census, 8);
            let tbuf = if tr { traffic } else { &dummy };
            let mut best = f64::INFINITY;
            for _ in 0..3 {
                let ms = tp_run_timed(ctx, |enc| {
                    tp_dispatch(
                        ctx,
                        enc,
                        name,
                        &p,
                        &[seed, &census, &entry, &out, tbuf],
                        1,
                        32,
                    )
                })?;
                best = best.min(ms);
            }
            best * 1e6 / cal_spin as f64
        };
        let dwells_us = if tr { dwells_us_tr } else { dwells_us_full };
        let _ = cal; // grid-level fma calibration retained for arms S/D
        for &w in widths {
            let mut per_dwell: Vec<(f64, u32, u32, f64)> = Vec::new();
            for &dw in dwells_us {
                let spin = ((dw * 1e3 / iter_ns).max(1.0)) as u32;
                tp_zero(&census, 8);
                tp_zero(&entry, g_tgs * 4);
                let p = TpParams {
                    spin_iters: spin,
                    grid_tgs: g_tgs as u32,
                    traffic_mask: if tr { traffic_mask } else { 0 },
                    ..Default::default()
                };
                let tbuf = if tr { traffic } else { &dummy };
                let ms = tp_run_timed(ctx, |enc| {
                    tp_dispatch(
                        ctx,
                        enc,
                        name,
                        &p,
                        &[seed, &census, &entry, &out, tbuf],
                        g_tgs,
                        w,
                    )
                })?;
                let max_alive = tp_read_u32(&census, 2)[1];
                let ea = tp_read_u32(&entry, g_tgs);
                // steady distribution: skip the first ramp wave
                let skip = (max_alive as usize).min(g_tgs.saturating_sub(1));
                let mut steady: Vec<u32> = ea[skip..].to_vec();
                steady.sort_unstable();
                let p10 = tp_percentile_u32(&steady, 0.10);
                let p50 = tp_percentile_u32(&steady, 0.50);
                let p90 = tp_percentile_u32(&steady, 0.90);
                let waves = (g_tgs as f64 / max_alive.max(1) as f64).ceil();
                println!(
                    "{name}\t{tr}\t{w}\t{dw:.0}\t{max_alive}\t{:.1}\t{p10}\t{waves:.0}\t{ms:.1}",
                    max_alive as f64 / TP_GPU_CORES
                );
                rows.push(serde_json::json!({
                    "arm": "r", "variant": name, "traffic": tr,
                    "w": w, "dwell_us_nominal": dw, "spin_iters": spin,
                    "iter_ns_1tg": iter_ns,
                    "gpu_ms": ms, "max_alive": max_alive,
                    "per_core": max_alive as f64 / TP_GPU_CORES,
                    "steady_p10": p10, "steady_p50": p50, "steady_p90": p90,
                    "waves": waves,
                }));
                per_dwell.push((dw, max_alive, p10, ms));
            }
            // plateau: last two dwells within 10%
            let n = per_dwell.len();
            let plateaued = n >= 2 && {
                let a = per_dwell[n - 2].1 as f64;
                let b = per_dwell[n - 1].1 as f64;
                (a - b).abs() / a.max(1.0) <= 0.10
            };
            let last = per_dwell[n - 1];
            caps.insert((name.to_string(), tr, w), (last.1, last.2, plateaued));
        }
    }
    Ok(TpArmROut { rows, caps })
}

fn tp_lat_stats(
    all: &[Vec<u32>],
    poll_ns: f64,
    cadence_ns: f64,
) -> (serde_json::Value, Vec<usize>) {
    // all[c][s] = poll counts per sampled epoch slot (aligned across
    // consumers). Global-stall slots: >= 80% of consumers spike >= 10x the
    // overall median sample.
    let mut flat: Vec<u32> = all
        .iter()
        .flat_map(|v| v.iter().copied())
        .filter(|&x| x != TP_LAT_UNUSED)
        .collect();
    if flat.is_empty() {
        return (serde_json::json!({"n": 0}), vec![]);
    }
    flat.sort_unstable();
    let med = flat[flat.len() / 2].max(1);
    let n_slots = all[0].len();
    let mut stall_slots = Vec::new();
    for s in 0..n_slots {
        let mut n_seen = 0usize;
        let mut n_spike = 0usize;
        for c in all {
            let v = c[s];
            if v != TP_LAT_UNUSED {
                n_seen += 1;
                if v >= med.saturating_mul(10) {
                    n_spike += 1;
                }
            }
        }
        if n_seen > 0 && (n_spike as f64) / (n_seen as f64) >= 0.8 {
            stall_slots.push(s);
        }
    }
    let filtered: Vec<u32> = {
        let mut v: Vec<u32> = Vec::new();
        for c in all {
            for (s, &x) in c.iter().enumerate() {
                if x != TP_LAT_UNUSED && !stall_slots.contains(&s) {
                    v.push(x);
                }
            }
        }
        v.sort_unstable();
        v
    };
    let pct = |v: &[u32], q: f64| tp_percentile_u32(v, q) as f64 * poll_ns / 1e3; // us
    // The sampled wait spans a full inter-epoch interval (the consumer
    // starts waiting right after observing the previous epoch), so raw
    // waits legitimately cluster near the producer cadence. The
    // propagation-relevant metric is the EXCESS over cadence; kill gates
    // read filtered_excess_p99_us (smoke finding).
    let excess = |v: &[u32]| -> Vec<u32> {
        let cad_polls = (cadence_ns / poll_ns) as i64;
        let mut e: Vec<u32> = v
            .iter()
            .map(|&x| ((x as i64 - cad_polls).max(0)) as u32)
            .collect();
        e.sort_unstable();
        e
    };
    let flat_ex = excess(&flat);
    let filt_ex = excess(&filtered);
    let j = serde_json::json!({
        "n": flat.len(),
        "raw_p50_us": pct(&flat, 0.50),
        "raw_p95_us": pct(&flat, 0.95),
        "raw_p99_us": pct(&flat, 0.99),
        "raw_max_us": *flat.last().unwrap() as f64 * poll_ns / 1e3,
        "n_stall_slots": stall_slots.len(),
        "filtered_p50_us": pct(&filtered, 0.50),
        "filtered_p95_us": pct(&filtered, 0.95),
        "filtered_p99_us": pct(&filtered, 0.99),
        "excess_p50_us": pct(&flat_ex, 0.50),
        "excess_p99_us": pct(&flat_ex, 0.99),
        "filtered_excess_p50_us": pct(&filt_ex, 0.50),
        "filtered_excess_p95_us": pct(&filt_ex, 0.95),
        "filtered_excess_p99_us": pct(&filt_ex, 0.99),
    });
    (j, stall_slots)
}

/// Launch a long-running streaming kernel on a second in-process queue to
/// generate device traffic under the signal window. Returns (queue, cmd).
fn tp_background_traffic(
    ctx: &MetalContext,
    cal: &TpCal,
    seed: &TpBuffer,
    traffic: &TpBuffer,
    traffic_mask: u32,
    target_ms: f64,
    resident_est: u32,
) -> Result<(
    Retained<ProtocolObject<dyn MTLCommandQueue>>,
    Retained<ProtocolObject<dyn MTLCommandBuffer>>,
)> {
    let q2 = ctx.device.newCommandQueue().context("bg queue")?;
    let g: usize = 10240;
    let waves = (g as f64 / resident_est.max(1) as f64).ceil().max(1.0);
    let dwell_ms = (target_ms * 1.5 / waves).max(0.5);
    let spin = ((dwell_ms * 1e6) / cal.fma_ns).max(1.0) as u32;
    let census = ctx.buffer_uninit(8)?;
    tp_zero(&census, 8);
    let entry = ctx.buffer_uninit(g * 4)?;
    let out = ctx.buffer_uninit(g * 4)?;
    let p = TpParams {
        spin_iters: spin,
        grid_tgs: g as u32,
        traffic_mask,
        ..Default::default()
    };
    let cmd = q2.commandBuffer().context("bg cmd")?;
    let enc = KernelEncoder::begin(&cmd);
    tp_dispatch(
        ctx,
        &enc,
        "tp_residency_lo_tr",
        &p,
        &[seed, &census, &entry, &out, traffic],
        g,
        32,
    )?;
    enc.end();
    cmd.commit();
    Ok((q2, cmd))
}

fn tp_arm_s(
    ctx: &MetalContext,
    cal: &TpCal,
    seed: &TpBuffer,
    traffic: &TpBuffer,
    traffic_mask: u32,
    resident_cap: u32,
    quick: bool,
) -> Result<Vec<serde_json::Value>> {
    let epochs: u32 = if quick { 1_000 } else { 10_000 };
    let mut consumer_set: Vec<usize> = vec![40, 160];
    let cap = (resident_cap.saturating_sub(1) as usize).min(1024);
    if cap > 160 {
        consumer_set.push(cap);
    }
    let cadences: &[(&str, f64)] = &[("tight", 0.0), ("25us", 25_000.0), ("100us", 100_000.0)];
    let mut rows = Vec::new();
    println!("\n== arm S: bounded one-way signaling (epochs {epochs}) ==");
    println!(
        "consumers\tcadence\ttraffic\tdelivered\ttimeouts\tcorrupt\texcess_p50/p95/p99_us\tstalls"
    );

    let ring = ctx.buffer_uninit(TP_RING_SAMPLES * 4)?; // TP_RING == 64 slots
    let sink = ctx.buffer_uninit(4)?;
    for &consumers in &consumer_set {
        let stats = ctx.buffer_uninit(consumers * 4 * 4)?;
        let lat = ctx.buffer_uninit(consumers * TP_RING_SAMPLES * 4)?;
        for &(cad_name, cad_ns) in cadences {
            for tr in [false, true] {
                let cadence_iters = (cad_ns / cal.fma_ns) as u32;
                let window_ns = epochs as f64 * cad_ns.max(1_000.0);
                let budget = (((window_ns * 2.0 + 100e6) / cal.poll_ns) as u64)
                    .min(u32::MAX as u64 - 1) as u32;
                tp_zero(&ring, TP_RING_SAMPLES * 4);
                tp_zero(&stats, consumers * 4 * 4);
                tp_fill_u32(&lat, consumers * TP_RING_SAMPLES, TP_LAT_UNUSED);
                let p = TpParams {
                    epochs,
                    cadence_iters,
                    spin_budget: budget,
                    sample_every: (epochs / TP_RING_SAMPLES as u32).max(1),
                    grid_tgs: (consumers + 1) as u32,
                    ..Default::default()
                };
                let bg = if tr {
                    Some(tp_background_traffic(
                        ctx,
                        cal,
                        seed,
                        traffic,
                        traffic_mask,
                        window_ns / 1e6 + 100.0,
                        resident_cap,
                    )?)
                } else {
                    None
                };
                let ms = tp_run_timed(ctx, |enc| {
                    tp_dispatch(
                        ctx,
                        enc,
                        "tp_signal",
                        &p,
                        &[seed, &ring, &stats, &lat, &sink],
                        consumers + 1,
                        32,
                    )
                })?;
                let mut bg_ms = 0.0;
                if let Some((_q2, cmd)) = bg {
                    cmd.waitUntilCompleted();
                    bg_ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                }
                let st = tp_read_u32(&stats, consumers * 4);
                let delivered = (0..consumers).filter(|&c| st[c * 4] == epochs).count();
                let timeouts: u32 = (0..consumers).map(|c| st[c * 4 + 2]).sum();
                let corrupt: u32 = (0..consumers).map(|c| st[c * 4 + 1]).sum();
                let lat_all: Vec<Vec<u32>> = (0..consumers)
                    .map(|c| {
                        tp_read_u32(&lat, consumers * TP_RING_SAMPLES)
                            [c * TP_RING_SAMPLES..(c + 1) * TP_RING_SAMPLES]
                            .to_vec()
                    })
                    .collect();
                let (lat_j, stalls) = tp_lat_stats(&lat_all, cal.poll_ns, cad_ns);
                println!(
                    "{consumers}\t{cad_name}\t{tr}\t{delivered}/{consumers}\t{timeouts}\t{corrupt}\t{:.1}/{:.1}/{:.1}\t{}",
                    lat_j["filtered_excess_p50_us"].as_f64().unwrap_or(0.0),
                    lat_j["filtered_excess_p95_us"].as_f64().unwrap_or(0.0),
                    lat_j["filtered_excess_p99_us"].as_f64().unwrap_or(0.0),
                    stalls.len(),
                );
                rows.push(serde_json::json!({
                    "arm": "s", "kind": "signal", "consumers": consumers,
                    "cadence": cad_name, "traffic": tr, "epochs": epochs,
                    "spin_budget": budget, "gpu_ms": ms, "bg_gpu_ms": bg_ms,
                    "delivered": delivered, "timeouts": timeouts,
                    "corrupt": corrupt, "latency": lat_j,
                    "latency_unit_note":
                        "poll counts x no-traffic poll_ns; traffic rows are \
                         indicative only (kill gate reads no-traffic rows)",
                }));
            }
        }
    }

    // Cross-object reordering probe (>= 25 us cadence only, per design).
    let re_epochs: u32 = if quick { 5_000 } else { 40_000 };
    let consumers = 160usize;
    let stats = ctx.buffer_uninit(consumers * 4 * 4)?;
    let ab = ctx.buffer_uninit(64 * 4)?;
    for tr in [false, true] {
        let cad_ns = 25_000.0;
        let budget = (((re_epochs as f64 * cad_ns * 2.0 + 100e6) / cal.poll_ns) as u64)
            .min(u32::MAX as u64 - 1) as u32;
        tp_zero(&ab, 64 * 4);
        tp_zero(&stats, consumers * 4 * 4);
        let p = TpParams {
            epochs: re_epochs,
            cadence_iters: (cad_ns / cal.fma_ns) as u32,
            spin_budget: budget,
            sample_every: 1,
            grid_tgs: (consumers + 1) as u32,
            ..Default::default()
        };
        let bg = if tr {
            Some(tp_background_traffic(
                ctx,
                cal,
                seed,
                traffic,
                traffic_mask,
                re_epochs as f64 * cad_ns / 1e6 + 100.0,
                resident_cap,
            )?)
        } else {
            None
        };
        let _ms = tp_run_timed(ctx, |enc| {
            tp_dispatch(
                ctx,
                enc,
                "tp_reorder",
                &p,
                &[seed, &ab, &stats, &sink],
                consumers + 1,
                32,
            )
        })?;
        if let Some((_q2, cmd)) = bg {
            cmd.waitUntilCompleted();
        }
        let st = tp_read_u32(&stats, consumers * 4);
        let fresh: u64 = (0..consumers).map(|c| st[c * 4] as u64).sum();
        let stale: u64 = (0..consumers).map(|c| st[c * 4 + 1] as u64).sum();
        let corrupt: u64 = (0..consumers).map(|c| st[c * 4 + 2] as u64).sum();
        let timeouts: u32 = (0..consumers).map(|c| st[c * 4 + 3]).sum();
        println!(
            "reorder\ttraffic={tr}\tfresh={fresh}\tstale={stale}\trate={:.2e}\tcorrupt={corrupt}\ttimeouts={timeouts}",
            stale as f64 / fresh.max(1) as f64
        );
        rows.push(serde_json::json!({
            "arm": "s", "kind": "reorder", "traffic": tr,
            "epochs": re_epochs, "consumers": consumers,
            "fresh": fresh, "stale": stale,
            "stale_rate": stale as f64 / fresh.max(1) as f64,
            "corrupt": corrupt, "timeouts": timeouts,
        }));
    }
    Ok(rows)
}

fn tp_median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn tp_arm_d(
    ctx: &MetalContext,
    cal: &TpCal,
    m_tgs_by_w: &BTreeMap<usize, usize>,
    runs: usize,
    quick: bool,
) -> Result<Vec<serde_json::Value>> {
    let ks: &[u32] = if quick { &[8, 110] } else { &[8, 32, 110] };
    let works_us: &[f64] = if quick {
        &[0.0, 15.0]
    } else {
        &[0.0, 5.0, 15.0, 30.0]
    };
    let widths: &[usize] = &[32, 64];
    let mut rows = Vec::new();
    println!("\n== arm D: boundary drain vs persistence ==");
    println!(
        "W\tm\tK\twork_us\tladder_ms\tlocal_mem_ms\tlocal_reg_ms\tglobal_ms\tglobal_wide_ms\tboundary_us\tbarrier_us\taborts"
    );

    for &w in widths {
        let &cap = m_tgs_by_w
            .get(&w)
            .with_context(|| format!("no arm-R low-water for W={w}; pass --grid-tgs"))?;
        // m-axis: production glue dispatches are NARROW (few TGs on a mostly
        // idle machine); the wide row keeps the design's low-water shape.
        let mut ms_list: Vec<usize> = if quick {
            vec![4, cap]
        } else {
            vec![4, 64, cap]
        };
        ms_list.sort_unstable();
        ms_list.dedup();
        let n_max = cap.max(64) * w;
        let input = ctx.buffer_uninit(n_max * 4)?;
        {
            let p = input.contents().as_ptr() as *mut f32;
            for i in 0..n_max {
                unsafe { p.add(i).write(0.5 + (i % 1024) as f32 * 1e-3) }
            }
        }
        let out = ctx.buffer_uninit(n_max * 4)?;
        let scratch = ctx.buffer_uninit(n_max * 4)?;
        let bar = ctx.buffer_uninit(8)?;
        let abort = ctx.buffer_uninit(4)?;

        // Mixed-shape ladders (timing-only): consecutive stages alternate TG
        // counts, modeling production's narrow-glue -> wide-projection
        // heterogeneity where a wide dispatch cannot start until the narrow
        // one's LAST TG drains. No persistent twin / checksum; per-boundary
        // cost is derived against the m=4 local_mem baseline (same
        // per-thread work; each shape fits one wave).
        for (mix_name, m_pat) in [
            ("mix4_64", vec![4usize, 64]),
            ("mix4_cap", vec![4usize, cap]),
        ] {
            for &k in ks {
                for &work in works_us {
                    let spin = ((work * 1e3 / cal.fma_ns).max(1.0)) as u32;
                    let mut t_mixed = Vec::new();
                    for rep in 0..=runs {
                        let ms = tp_run_timed(ctx, |enc| {
                            let mut src = &input;
                            let bufs = [&out, &scratch];
                            for s in 0..k as usize {
                                let m_s = m_pat[s % m_pat.len()];
                                let p = TpParams {
                                    spin_iters: spin,
                                    grid_tgs: m_s as u32,
                                    stages: k,
                                    n_elems: (m_s * w) as u32,
                                    ..Default::default()
                                };
                                let dst = bufs[s % 2];
                                let kn = if s % 2 == 0 {
                                    "tp_chain_stage"
                                } else {
                                    "tp_chain_stage_b"
                                };
                                tp_dispatch(ctx, enc, kn, &p, &[src, dst], m_s, w)?;
                                src = dst;
                            }
                            Ok(())
                        })?;
                        if rep > 0 {
                            t_mixed.push(ms);
                        }
                    }
                    let ml = tp_median(t_mixed);
                    println!("{w}\t{mix_name}\t{k}\t{work:.0}\t{ml:.3}\t-\t-\t-\t-\t-\t-\t-");
                    rows.push(serde_json::json!({
                        "arm": "d", "w": w, "mixed": mix_name, "cap_tgs": cap,
                        "k": k, "work_us_nominal": work, "spin_iters": spin,
                        "ladder_ms": ml,
                    }));
                }
            }
        }

        for &m in &ms_list {
            let n = m * w;
            for &k in ks {
                for &work in works_us {
                    let spin = ((work * 1e3 / cal.fma_ns).max(1.0)) as u32;
                    let p = TpParams {
                        spin_iters: spin,
                        grid_tgs: m as u32,
                        stages: k,
                        spin_budget: ((10e6 / cal.poll_ns) as u64).min(u32::MAX as u64) as u32,
                        n_elems: n as u32,
                        ..Default::default()
                    };

                    // ladder alternates two identical PSOs to match
                    // production's per-dispatch state changes
                    let ladder_encode = |enc: &KernelEncoder| -> Result<()> {
                        let mut src = &input;
                        let bufs = [&out, &scratch];
                        for s in 0..k as usize {
                            let dst = bufs[s % 2];
                            let kn = if s % 2 == 0 {
                                "tp_chain_stage"
                            } else {
                                "tp_chain_stage_b"
                            };
                            tp_dispatch(ctx, enc, kn, &p, &[src, dst], m, w)?;
                            src = dst;
                        }
                        Ok(())
                    };
                    let final_ladder = if k as usize % 2 == 1 { &out } else { &scratch };

                    let mut checksum: Option<Vec<u32>> = None;
                    let mut mismatch = false;

                    let mut t_ladder = Vec::new();
                    for rep in 0..=runs {
                        let ms = tp_run_timed(ctx, ladder_encode)?;
                        if rep == 0 {
                            checksum = Some(tp_read_f32_bits(final_ladder, n));
                            continue;
                        }
                        t_ladder.push(ms);
                    }
                    let checksum = checksum.unwrap();

                    let mut run_variant = |kernel: &str,
                                           bufs: &[&TpBuffer],
                                           launch_tgs: usize,
                                           grid_tgs_param: usize,
                                           needs_bar: bool|
                     -> Result<(Vec<f64>, u32)> {
                        let pv = TpParams {
                            grid_tgs: grid_tgs_param as u32,
                            ..p
                        };
                        let mut ts = Vec::new();
                        let mut aborts = 0u32;
                        for rep in 0..=runs {
                            if needs_bar {
                                tp_zero(&bar, 8);
                                tp_zero(&abort, 4);
                            }
                            let ms = tp_run_timed(ctx, |enc| {
                                tp_dispatch(ctx, enc, kernel, &pv, bufs, launch_tgs, w)
                            })?;
                            if needs_bar && tp_read_u32(&abort, 1)[0] != 0 {
                                aborts += 1;
                                continue; // timing not admitted
                            }
                            if rep == 0 {
                                if tp_read_f32_bits(&out, n) != checksum {
                                    mismatch = true;
                                }
                                continue;
                            }
                            ts.push(ms);
                        }
                        Ok((ts, aborts))
                    };

                    let (t_lmem, _) = run_variant(
                        "tp_chain_persistent_local_mem",
                        &[&input, &out, &scratch],
                        m,
                        m,
                        false,
                    )?;
                    let (t_lreg, _) = run_variant(
                        "tp_chain_persistent_local_reg",
                        &[&input, &out],
                        m,
                        m,
                        false,
                    )?;
                    let (t_glob, aborts) = run_variant(
                        "tp_chain_persistent_global",
                        &[&input, &out, &bar, &abort, &scratch],
                        m,
                        m,
                        true,
                    )?;
                    // persistent HOST shape: full low-water grid carries
                    // narrow stages; idle TGs still pay every barrier
                    let (t_gwide, aborts_w) = if m < cap {
                        run_variant(
                            "tp_chain_persistent_global",
                            &[&input, &out, &bar, &abort, &scratch],
                            cap,
                            cap,
                            true,
                        )?
                    } else {
                        (Vec::new(), 0)
                    };

                    let (ml, mm, mr, mg) = (
                        tp_median(t_ladder),
                        tp_median(t_lmem),
                        tp_median(t_lreg),
                        tp_median(t_glob),
                    );
                    let mgw = if t_gwide.is_empty() {
                        f64::NAN
                    } else {
                        tp_median(t_gwide)
                    };
                    let boundaries = (k - 1).max(1) as f64;
                    let boundary_us = (ml - mm) * 1e3 / boundaries;
                    let barrier_us = (mg - mm) * 1e3 / boundaries;
                    println!(
                        "{w}\t{m}\t{k}\t{work:.0}\t{ml:.3}\t{mm:.3}\t{mr:.3}\t{mg:.3}\t{mgw:.3}\t{boundary_us:.1}\t{barrier_us:.1}\t{}{}",
                        aborts + aborts_w,
                        if mismatch { "\tCHECKSUM-MISMATCH" } else { "" }
                    );
                    rows.push(serde_json::json!({
                        "arm": "d", "w": w, "m_tgs": m, "cap_tgs": cap, "k": k,
                        "work_us_nominal": work, "spin_iters": spin,
                        "ladder_ms": ml, "local_mem_ms": mm, "local_reg_ms": mr,
                        "global_ms": mg,
                        "global_wide_ms": if mgw.is_nan() { serde_json::Value::Null } else { serde_json::json!(mgw) },
                        "aborts": aborts, "aborts_wide": aborts_w,
                        "checksum_ok": !mismatch,
                        "boundary_us_per": boundary_us,
                        "barrier_us_per": barrier_us,
                        "barrier_wide_us_per": if mgw.is_nan() { serde_json::Value::Null } else { serde_json::json!((mgw - mm) * 1e3 / boundaries) },
                        "recovery_local_mem": 1.0 - mm / ml.max(1e-9),
                        "recovery_local_reg": 1.0 - mr / ml.max(1e-9),
                        "recovery_global": 1.0 - mg / ml.max(1e-9),
                        "recovery_global_wide": if mgw.is_nan() { serde_json::Value::Null } else { serde_json::json!(1.0 - mgw / ml.max(1e-9)) },
                    }));
                }
            }
        }
    }
    Ok(rows)
}

fn run_dispatch_census(args: DispatchCensusArgs) -> Result<()> {
    use qwen_llm::metal::{dispatch_census_begin, dispatch_census_take};
    let ctx = MetalContext::new()?;
    eprintln!("[census] device: {}", ctx.describe());
    let g = GgufFile::open(&args.model)?;
    let m = Model::from_gguf(&g)?;
    let mm = MetalModel::load(&ctx, &g, &m)?;
    let mf = MetalForward::new(&ctx, &mm);

    let mut s = MetalSession::fresh(&ctx, &mm, args.ctx + 32)?;
    // PSO warm on the exact profiled path (all splits on for finest labels)
    for i in 0..3 {
        let _ = mf.single_token_argmax_stage_profiled_concurrent_gdn_moe(
            0, i as u32, &mut s, true, true, true,
        )?;
    }
    let mut s = MetalSession::fresh(&ctx, &mm, args.ctx + 32)?;
    if args.ctx > 1 {
        if args.decode_ramp_warm {
            let _ = mf.single_token(0, 0, &mut s)?;
            for p in 1..(args.ctx as u32) {
                let _ = mf.single_token(0, p, &mut s)?;
            }
        } else {
            let ids = vec![0i32; args.ctx];
            let chunk = default_prefill_chunk(mm.arch.kind, args.ctx);
            let mut scratch = fresh_prefill_scratch_for_prompt(&ctx, &mm, chunk, ids.len())
                .context("census prefill scratch")?;
            let t0 = Instant::now();
            prefill_tokens_prompt_only_profiled(&mf, &ids, 0, &mut s, &mut scratch)
                .context("census prefill warm")?;
            eprintln!(
                "[census] prefill-warm to {} in {:.1}s",
                args.ctx,
                t0.elapsed().as_secs_f64()
            );
        }
    }

    dispatch_census_begin();
    let (_tok, profile) = mf.single_token_argmax_stage_profiled_concurrent_gdn_moe(
        0,
        args.ctx as u32,
        &mut s,
        true,
        true,
        true,
    )?;
    let rows = dispatch_census_take();

    // family time totals from the same token
    let mut fam_ms: BTreeMap<String, (f64, f64)> = BTreeMap::new(); // ms, pct
    for st in &profile.stages {
        let e = fam_ms.entry(st.family.clone()).or_default();
        e.0 += st.duration_ms_scaled;
        e.1 += st.fraction_of_gpu * 100.0;
    }
    // dispatch shape aggregation per (family, kernel, grid, tg)
    let mut agg: BTreeMap<(String, String, u64, u64), u64> = BTreeMap::new();
    for r in &rows {
        *agg.entry((
            r.family.to_string(),
            r.kernel.clone(),
            r.grid_tgs,
            r.tg_threads,
        ))
        .or_default() += 1;
    }

    println!(
        "[census] ctx={} dispatches={} families={} (stage-profiled token; shapes exact, times +~19% perturbed - use shares)",
        args.ctx,
        rows.len(),
        fam_ms.len()
    );
    println!("family\tms\tpct_gpu\tkernel\tcount\tgrid_tgs\ttg_threads\tsimdgroups\tcore_fill_pct");
    let mut fam_sorted: Vec<_> = fam_ms.iter().collect();
    fam_sorted.sort_by(|a, b| b.1.0.partial_cmp(&a.1.0).unwrap());
    let mut json_rows = Vec::new();
    for (fam, (ms, pct)) in &fam_sorted {
        let mut first = true;
        for ((f, kernel, grid, tg), count) in agg.iter() {
            if f != *fam {
                continue;
            }
            let sg = grid * (tg / 32).max(1);
            // one-simdgroup-per-TG fill estimate vs 40 cores
            let fill = (*grid as f64 / TP_GPU_CORES * 100.0).min(100.0);
            println!(
                "{}\t{:.3}\t{:.2}\t{}\t{}\t{}\t{}\t{}\t{:.0}",
                if first { fam.as_str() } else { "" },
                if first { *ms } else { 0.0 },
                if first { *pct } else { 0.0 },
                kernel,
                count,
                grid,
                tg,
                sg,
                fill
            );
            json_rows.push(serde_json::json!({
                "family": fam, "family_ms": ms, "family_pct": pct,
                "kernel": kernel, "count": count, "grid_tgs": grid,
                "tg_threads": tg, "simdgroups_per_dispatch": sg,
            }));
            first = false;
        }
    }
    if let Some(dir) = args.out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(
        &args.out,
        serde_json::to_string_pretty(&serde_json::json!({
            "model": args.model.display().to_string(),
            "ctx": args.ctx,
            "gpu_ms_perturbed": profile.token.gpu_kernel_ms,
            "rows": json_rows,
        }))?,
    )?;
    println!("wrote {}", args.out.display());
    Ok(())
}

fn run_topology_probe(args: TopologyProbeArgs) -> Result<()> {
    let ctx = MetalContext::new()?;
    std::fs::create_dir_all(&args.out_dir)?;
    let arms: HashSet<String> = if args.arm == "all" {
        ["r", "s", "d"].iter().map(|s| s.to_string()).collect()
    } else {
        args.arm.split(',').map(|s| s.trim().to_string()).collect()
    };

    // shared seed + traffic buffers
    let seed_host: Vec<f32> = (0..1024).map(|i| 0.5 + i as f32 * 1e-3).collect();
    let seed = ctx.buffer_from(&seed_host)?;
    let traffic_elems: usize = if args.quick { 1 << 26 } else { 1 << 29 }; // 256MB / 2GB
    let traffic = ctx.buffer_uninit(traffic_elems * 4)?;
    tp_zero(&traffic, traffic_elems * 4);
    let traffic_mask = (traffic_elems - 1) as u32;

    let cal = tp_calibrate(&ctx, &seed)?;
    println!(
        "calibration: fma {:.3} ns/iter, poll {:.3} ns/iter",
        cal.fma_ns, cal.poll_ns
    );

    let mut all_rows: Vec<serde_json::Value> = Vec::new();
    let mut r_caps: Option<TpArmROut> = None;

    if !args.dwell_extend.is_empty() {
        // focused plateau confirmation: lo variant, no traffic, W in {32,64}
        let g_tgs: usize = 10240;
        let census = ctx.buffer_uninit(8)?;
        let entry = ctx.buffer_uninit(g_tgs * 4)?;
        let out = ctx.buffer_uninit(g_tgs * 4)?;
        let dummy = ctx.buffer_uninit(4)?;
        tp_zero(&dummy, 4);
        println!("\n== arm R dwell extension (lo, no-traffic) ==");
        println!("W\tdwell_us\tmax_alive\tper_core\tsteady_p10\tgpu_ms");
        for &w in &[32usize, 64] {
            for &dw in &args.dwell_extend {
                let spin = ((dw * 1e3 / cal.fma_ns).max(1.0)) as u32;
                tp_zero(&census, 8);
                tp_zero(&entry, g_tgs * 4);
                let p = TpParams {
                    spin_iters: spin,
                    grid_tgs: g_tgs as u32,
                    ..Default::default()
                };
                let ms = tp_run_timed(&ctx, |enc| {
                    tp_dispatch(
                        &ctx,
                        enc,
                        "tp_residency_lo",
                        &p,
                        &[&seed, &census, &entry, &out, &dummy],
                        g_tgs,
                        w,
                    )
                })?;
                let max_alive = tp_read_u32(&census, 2)[1];
                let ea = tp_read_u32(&entry, g_tgs);
                let skip = (max_alive as usize).min(g_tgs.saturating_sub(1));
                let mut steady: Vec<u32> = ea[skip..].to_vec();
                steady.sort_unstable();
                let p10 = tp_percentile_u32(&steady, 0.10);
                println!(
                    "{w}\t{dw:.0}\t{max_alive}\t{:.1}\t{p10}\t{ms:.1}",
                    max_alive as f64 / TP_GPU_CORES
                );
                all_rows.push(serde_json::json!({
                    "arm": "r", "variant": "tp_residency_lo", "traffic": false,
                    "w": w, "dwell_us_nominal": dw, "spin_iters": spin,
                    "gpu_ms": ms, "max_alive": max_alive,
                    "per_core": max_alive as f64 / TP_GPU_CORES,
                    "steady_p10": p10, "dwell_extension": true,
                }));
            }
        }
    }

    if arms.contains("r") {
        let r = tp_arm_r(&ctx, &cal, &seed, &traffic, traffic_mask, args.quick)?;
        all_rows.extend(r.rows.iter().cloned());
        // pre-registered kill gate: lo / no-traffic / W=32, plateaued
        if let Some(&(max_alive, p10, plateaued)) =
            r.caps.get(&("tp_residency_lo".to_string(), false, 32))
        {
            let per_core = max_alive as f64 / TP_GPU_CORES;
            println!(
                "\narm R gate: W32/lo/no-traffic max_alive={max_alive} ({per_core:.1}/core), \
                 steady_p10={p10}, plateaued={plateaued} -> {}",
                if !plateaued {
                    "NOT PLATEAUED: extend dwell before reading the gate"
                } else if per_core < 16.0 {
                    "KILL (per-core residency < 16)"
                } else {
                    "PASS"
                }
            );
        }
        r_caps = Some(r);
    }

    let resident_cap = r_caps
        .as_ref()
        .and_then(|r| r.caps.get(&("tp_residency_lo".to_string(), false, 32)))
        .map(|&(_, p10, _)| p10)
        .or(args.grid_tgs.map(|g| g as u32))
        .unwrap_or(512);

    if arms.contains("s") {
        let rows = tp_arm_s(
            &ctx,
            &cal,
            &seed,
            &traffic,
            traffic_mask,
            resident_cap,
            args.quick,
        )?;
        all_rows.extend(rows);
    }

    if arms.contains("d") {
        let mut m_tgs_by_w = BTreeMap::new();
        for &w in &[32usize, 64] {
            let m = args.grid_tgs.or_else(|| {
                r_caps.as_ref().and_then(|r| {
                    r.caps
                        .get(&("tp_residency_lo".to_string(), false, w))
                        .map(|&(_, p10, _)| p10 as usize)
                })
            });
            if let Some(m) = m {
                m_tgs_by_w.insert(w, m.max(40));
            }
        }
        let rows = tp_arm_d(&ctx, &cal, &m_tgs_by_w, args.runs, args.quick)?;
        all_rows.extend(rows);
    }

    let summary = serde_json::json!({
        "design": "docs/bench/2026-07-05-b0-topology-probe/README.md",
        "quick": args.quick,
        "calibration": {"fma_ns": cal.fma_ns, "poll_ns": cal.poll_ns},
        "device": ctx.device.name().to_string(),
        "rows": all_rows,
    });
    let mut fname = String::from("topology-probe");
    if args.arm != "all" {
        fname.push('-');
        fname.push_str(&args.arm.replace(',', "_"));
    }
    if !args.dwell_extend.is_empty() {
        fname.push_str("-dwellext");
    }
    if args.quick {
        fname.push_str("-quick");
    }
    fname.push_str(".json");
    let path = args.out_dir.join(fname);
    std::fs::write(&path, serde_json::to_string_pretty(&summary)?)?;
    println!("\nwrote {}", path.display());
    Ok(())
}

fn time_gpu_reps<F>(
    ctx: &MetalContext,
    warmup: usize,
    iters: usize,
    mut encode: F,
) -> Result<(f64, f64)>
where
    F: FnMut(&KernelEncoder) -> Result<()>,
{
    for _ in 0..warmup {
        let cmd = ctx.queue.commandBuffer().context("warmup cmd")?;
        let enc = KernelEncoder::begin(&cmd);
        encode(&enc)?;
        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();
    }

    let mut wall_ms = 0.0f64;
    let mut gpu_ms = 0.0f64;
    for _ in 0..iters {
        let cmd = ctx.queue.commandBuffer().context("timed cmd")?;
        let enc = KernelEncoder::begin(&cmd);
        encode(&enc)?;
        enc.end();
        let t = Instant::now();
        cmd.commit();
        cmd.waitUntilCompleted();
        wall_ms += t.elapsed().as_secs_f64() * 1e3;
        gpu_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
    }
    Ok((wall_ms / iters as f64, gpu_ms / iters as f64))
}

#[derive(Clone, Copy, Debug)]
struct TimedGpuStats {
    avg_wall_ms: f64,
    avg_gpu_ms: f64,
    p50_wall_ms: f64,
    p50_gpu_ms: f64,
    p90_gpu_ms: f64,
    max_gpu_ms: f64,
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn time_gpu_reps_stats<F>(
    ctx: &MetalContext,
    warmup: usize,
    iters: usize,
    mut encode: F,
) -> Result<TimedGpuStats>
where
    F: FnMut(&KernelEncoder) -> Result<()>,
{
    for _ in 0..warmup {
        let cmd = ctx.queue.commandBuffer().context("warmup cmd")?;
        let enc = KernelEncoder::begin(&cmd);
        encode(&enc)?;
        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();
    }

    let mut wall_samples = Vec::with_capacity(iters);
    let mut gpu_samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let cmd = ctx.queue.commandBuffer().context("timed cmd")?;
        let enc = KernelEncoder::begin(&cmd);
        encode(&enc)?;
        enc.end();
        let t = Instant::now();
        cmd.commit();
        cmd.waitUntilCompleted();
        wall_samples.push(t.elapsed().as_secs_f64() * 1e3);
        gpu_samples.push((cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3);
    }

    let avg_wall_ms = wall_samples.iter().sum::<f64>() / iters as f64;
    let avg_gpu_ms = gpu_samples.iter().sum::<f64>() / iters as f64;
    wall_samples.sort_by(|a, b| a.total_cmp(b));
    gpu_samples.sort_by(|a, b| a.total_cmp(b));
    Ok(TimedGpuStats {
        avg_wall_ms,
        avg_gpu_ms,
        p50_wall_ms: percentile(&wall_samples, 0.50),
        p50_gpu_ms: percentile(&gpu_samples, 0.50),
        p90_gpu_ms: percentile(&gpu_samples, 0.90),
        max_gpu_ms: *gpu_samples.last().unwrap_or(&0.0),
    })
}

fn time_cmd_reps_stats<F>(
    ctx: &MetalContext,
    warmup: usize,
    iters: usize,
    mut encode: F,
) -> Result<TimedGpuStats>
where
    F: FnMut(&Retained<ProtocolObject<dyn MTLCommandBuffer>>) -> Result<()>,
{
    for _ in 0..warmup {
        let cmd = ctx.queue.commandBuffer().context("warmup cmd")?;
        encode(&cmd)?;
        cmd.commit();
        cmd.waitUntilCompleted();
    }

    let mut wall_samples = Vec::with_capacity(iters);
    let mut gpu_samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let cmd = ctx.queue.commandBuffer().context("timed cmd")?;
        encode(&cmd)?;
        let t = Instant::now();
        cmd.commit();
        cmd.waitUntilCompleted();
        wall_samples.push(t.elapsed().as_secs_f64() * 1e3);
        gpu_samples.push((cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3);
    }

    let avg_wall_ms = wall_samples.iter().sum::<f64>() / iters as f64;
    let avg_gpu_ms = gpu_samples.iter().sum::<f64>() / iters as f64;
    wall_samples.sort_by(|a, b| a.total_cmp(b));
    gpu_samples.sort_by(|a, b| a.total_cmp(b));
    Ok(TimedGpuStats {
        avg_wall_ms,
        avg_gpu_ms,
        p50_wall_ms: percentile(&wall_samples, 0.50),
        p50_gpu_ms: percentile(&gpu_samples, 0.50),
        p90_gpu_ms: percentile(&gpu_samples, 0.90),
        max_gpu_ms: *gpu_samples.last().unwrap_or(&0.0),
    })
}

#[derive(Clone, Copy)]
struct ProjectionWeight<'a> {
    weight: &'a MetalTensor,
    n_out: usize,
}

struct ProjectionBatchBench<'a> {
    name: &'static str,
    n_in: usize,
    max_out: usize,
    input_pack_count: usize,
    weights: Vec<ProjectionWeight<'a>>,
    x_single: MetalTensor,
    x_batch: MetalTensor,
    x_slots: Vec<MetalTensor>,
    y_single: Vec<MetalTensor>,
    y_batch: Vec<MetalTensor>,
    y_sink: MetalTensor,
    weight_bytes: u64,
}

impl<'a> ProjectionBatchBench<'a> {
    fn new(
        ctx: &MetalContext,
        name: &'static str,
        n_in: usize,
        max_tokens: usize,
        input_pack_count: usize,
        weights: Vec<ProjectionWeight<'a>>,
    ) -> Result<Self> {
        if weights.is_empty() {
            return Err(anyhow!("projection group {name} has no weights"));
        }
        let max_out = weights.iter().map(|w| w.n_out).max().unwrap_or(1);
        let weight_bytes = weights.iter().map(|w| w.weight.n_bytes()).sum();
        let mut y_single = Vec::with_capacity(weights.len());
        let mut y_batch = Vec::with_capacity(weights.len());
        for w in &weights {
            y_single.push(MetalTensor::zeros_f32(ctx, vec![w.n_out as u64])?);
            y_batch.push(MetalTensor::zeros_f32(
                ctx,
                vec![(max_tokens * w.n_out) as u64],
            )?);
        }
        let mut x_slots = Vec::with_capacity(max_tokens);
        for _ in 0..max_tokens {
            x_slots.push(MetalTensor::zeros_f32(ctx, vec![n_in as u64])?);
        }
        Ok(Self {
            name,
            n_in,
            max_out,
            input_pack_count,
            weights,
            x_single: MetalTensor::zeros_f32(ctx, vec![n_in as u64])?,
            x_batch: MetalTensor::zeros_f32(ctx, vec![(max_tokens * n_in) as u64])?,
            x_slots,
            y_single,
            y_batch,
            y_sink: MetalTensor::zeros_f32(ctx, vec![(max_tokens * max_out) as u64])?,
            weight_bytes,
        })
    }
}

fn blit_projection_group_pack(blit: &BlitEncoder, group: &ProjectionBatchBench<'_>, tokens: usize) {
    let row_bytes = (group.n_in * 4) as u64;
    for _ in 0..group.input_pack_count {
        for tok in 0..tokens {
            let dst_offset = group.x_batch.offset + tok as u64 * row_bytes;
            blit.copy_buffer(
                &group.x_slots[tok].buffer,
                group.x_slots[tok].offset,
                &group.x_batch.buffer,
                dst_offset,
                row_bytes,
            );
        }
    }
}

fn blit_projection_group_scatter(
    blit: &BlitEncoder,
    group: &ProjectionBatchBench<'_>,
    tokens: usize,
) {
    for (i, w) in group.weights.iter().enumerate() {
        let row_bytes = (w.n_out * 4) as u64;
        let sink_row_bytes = (group.max_out * 4) as u64;
        for tok in 0..tokens {
            let src_offset = group.y_batch[i].offset + tok as u64 * row_bytes;
            let dst_offset = group.y_sink.offset + tok as u64 * sink_row_bytes;
            blit.copy_buffer(
                &group.y_batch[i].buffer,
                src_offset,
                &group.y_sink.buffer,
                dst_offset,
                row_bytes,
            );
        }
    }
}

fn projection_group_pack_bytes(group: &ProjectionBatchBench<'_>) -> u64 {
    (group.input_pack_count * group.n_in * 4) as u64
}

fn projection_group_scatter_bytes(group: &ProjectionBatchBench<'_>) -> u64 {
    group.weights.iter().map(|w| (w.n_out * 4) as u64).sum()
}

fn encode_projection_group_matvec_seq(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    group: &ProjectionBatchBench<'_>,
    tokens: usize,
) -> Result<()> {
    for _ in 0..tokens {
        for (i, w) in group.weights.iter().enumerate() {
            encode_mat_vec_dispatch(
                ctx,
                enc,
                w.weight,
                &group.x_single,
                &group.y_single[i],
                group.n_in,
                w.n_out,
            )?;
        }
    }
    Ok(())
}

fn encode_projection_group_matmat_batch(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    group: &ProjectionBatchBench<'_>,
    tokens: usize,
) -> Result<()> {
    let x_batch = group
        .x_batch
        .view_subrange(0, vec![(tokens * group.n_in) as u64]);
    for (i, w) in group.weights.iter().enumerate() {
        let y_batch = group.y_batch[i].view_subrange(0, vec![(tokens * w.n_out) as u64]);
        encode_mat_mat_dispatch(
            ctx, enc, w.weight, &x_batch, &y_batch, group.n_in, w.n_out, tokens,
        )?;
    }
    Ok(())
}

fn projection_fill_inputs(ctx: &MetalContext, groups: &[ProjectionBatchBench<'_>]) -> Result<()> {
    let cmd = ctx.queue.commandBuffer().context("projection fill cmd")?;
    let enc = KernelEncoder::begin(&cmd);
    for (i, group) in groups.iter().enumerate() {
        let v = 0.03125 + (i as f32) * 0.00390625;
        encode_fill_f32(ctx, &enc, &group.x_single, v)?;
        encode_fill_f32(ctx, &enc, &group.x_batch, v)?;
        for slot in &group.x_slots {
            encode_fill_f32(ctx, &enc, slot, v)?;
        }
    }
    enc.end();
    cmd.commit();
    cmd.waitUntilCompleted();
    Ok(())
}

fn run_gdn_proj_micro(args: GdnProjMicroArgs) -> Result<()> {
    let GdnProjMicroArgs {
        model,
        iters,
        warmup,
        tokens,
    } = args;
    if iters == 0 {
        return Err(anyhow!("--iters must be >= 1"));
    }
    if tokens == 0 {
        return Err(anyhow!("--tokens must be >= 1"));
    }

    let ctx = MetalContext::new().context("init MetalContext")?;
    let g = GgufFile::open(&model).with_context(|| format!("open {}", model.display()))?;
    let m = Model::from_gguf(&g).context("parse model")?;
    let mm = MetalModel::load(&ctx, &g, &m).context("metal-load model")?;
    let s = MetalSession::fresh(&ctx, &mm, 32).context("session")?;

    let arch = &mm.arch;
    let h = arch.hidden_size as usize;
    let n_v = arch.gdn_n_v_heads as usize;
    let n_k = arch.gdn_n_k_heads as usize;
    let head_dim = arch.gdn_head_dim as usize;
    let conv_dim = (2 * n_k + n_v) * head_dim;
    let v_dim = n_v * head_dim;
    let gdn_blocks: Vec<_> = mm
        .blocks
        .iter()
        .filter_map(|b| match b {
            MetalBlock::Gdn(g) => Some(g),
            MetalBlock::Attn(_) => None,
        })
        .collect();
    if gdn_blocks.is_empty() {
        return Err(anyhow!("model has no GDN blocks"));
    }

    let h_batch = MetalTensor::zeros_f32(&ctx, vec![(tokens * h) as u64])?;
    let gdn_normed_batch = MetalTensor::zeros_f32(&ctx, vec![(tokens * v_dim) as u64])?;
    let qkv_batch = MetalTensor::zeros_f32(&ctx, vec![(tokens * conv_dim) as u64])?;
    let z_batch = MetalTensor::zeros_f32(&ctx, vec![(tokens * v_dim) as u64])?;
    let out_batch = MetalTensor::zeros_f32(&ctx, vec![(tokens * h) as u64])?;

    {
        let cmd = ctx.queue.commandBuffer().context("init fill cmd")?;
        let enc = KernelEncoder::begin(&cmd);
        encode_fill_f32(&ctx, &enc, &s.h, 0.125)?;
        encode_fill_f32(&ctx, &enc, &s.gdn_normed, 0.0625)?;
        encode_fill_f32(&ctx, &enc, &h_batch, 0.125)?;
        encode_fill_f32(&ctx, &enc, &gdn_normed_batch, 0.0625)?;
        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();
    }

    let qkv_bytes: u64 = gdn_blocks.iter().map(|gb| gb.in_proj_qkv.n_bytes()).sum();
    let z_bytes: u64 = gdn_blocks.iter().map(|gb| gb.in_proj_z.n_bytes()).sum();
    let out_bytes: u64 = gdn_blocks.iter().map(|gb| gb.out_proj.n_bytes()).sum();

    println!(
        "[gdn-proj-micro] model={} layers={} h={} conv_dim={} v_dim={} tokens={} warmup={} iters={}",
        model.display(),
        gdn_blocks.len(),
        h,
        conv_dim,
        v_dim,
        tokens,
        warmup,
        iters
    );
    println!(
        "phase\tmode\ttokens\tbytes_gb_per_token\tavg_wall_ms\tavg_gpu_ms\tavg_gpu_ms_per_tok\teff_weight_gb_s"
    );

    let report = |label: &str, mode: &str, bytes_per_token: u64, wall_ms: f64, gpu_ms: f64| {
        let bytes_gb = bytes_per_token as f64 / 1e9;
        let eff_gb = bytes_gb * tokens as f64;
        let gpu_per_tok = gpu_ms / tokens as f64;
        let gb_s = eff_gb / (gpu_ms / 1e3);
        println!(
            "{label}\t{mode}\t{tokens}\t{bytes_gb:.4}\t{wall_ms:.4}\t{gpu_ms:.4}\t{gpu_per_tok:.4}\t{gb_s:.1}"
        );
    };

    let (wall, gpu) = time_gpu_reps(&ctx, warmup, iters, |enc| {
        for _ in 0..tokens {
            for gb in &gdn_blocks {
                encode_mat_vec_dispatch(&ctx, enc, &gb.in_proj_qkv, &s.h, &s.gdn_qkv, h, conv_dim)?;
            }
        }
        Ok(())
    })?;
    report("qkv", "matvec_seq", qkv_bytes, wall, gpu);

    let (wall, gpu) = time_gpu_reps(&ctx, warmup, iters, |enc| {
        for gb in &gdn_blocks {
            encode_mat_mat_dispatch(
                &ctx,
                enc,
                &gb.in_proj_qkv,
                &h_batch,
                &qkv_batch,
                h,
                conv_dim,
                tokens,
            )?;
        }
        Ok(())
    })?;
    report("qkv", "matmat_batch", qkv_bytes, wall, gpu);

    let (wall, gpu) = time_gpu_reps(&ctx, warmup, iters, |enc| {
        for _ in 0..tokens {
            for gb in &gdn_blocks {
                encode_mat_vec_dispatch(&ctx, enc, &gb.in_proj_z, &s.h, &s.gdn_z, h, v_dim)?;
            }
        }
        Ok(())
    })?;
    report("z", "matvec_seq", z_bytes, wall, gpu);

    let (wall, gpu) = time_gpu_reps(&ctx, warmup, iters, |enc| {
        for gb in &gdn_blocks {
            encode_mat_mat_dispatch(
                &ctx,
                enc,
                &gb.in_proj_z,
                &h_batch,
                &z_batch,
                h,
                v_dim,
                tokens,
            )?;
        }
        Ok(())
    })?;
    report("z", "matmat_batch", z_bytes, wall, gpu);

    let (wall, gpu) = time_gpu_reps(&ctx, warmup, iters, |enc| {
        for _ in 0..tokens {
            for gb in &gdn_blocks {
                encode_mat_vec_dispatch(&ctx, enc, &gb.in_proj_qkv, &s.h, &s.gdn_qkv, h, conv_dim)?;
                encode_mat_vec_dispatch(&ctx, enc, &gb.in_proj_z, &s.h, &s.gdn_z, h, v_dim)?;
            }
        }
        Ok(())
    })?;
    report("qkv+z", "matvec_seq", qkv_bytes + z_bytes, wall, gpu);

    let (wall, gpu) = time_gpu_reps(&ctx, warmup, iters, |enc| {
        for gb in &gdn_blocks {
            encode_mat_mat_dispatch(
                &ctx,
                enc,
                &gb.in_proj_qkv,
                &h_batch,
                &qkv_batch,
                h,
                conv_dim,
                tokens,
            )?;
            encode_mat_mat_dispatch(
                &ctx,
                enc,
                &gb.in_proj_z,
                &h_batch,
                &z_batch,
                h,
                v_dim,
                tokens,
            )?;
        }
        Ok(())
    })?;
    report("qkv+z", "matmat_batch", qkv_bytes + z_bytes, wall, gpu);

    let (wall, gpu) = time_gpu_reps(&ctx, warmup, iters, |enc| {
        for _ in 0..tokens {
            for gb in &gdn_blocks {
                encode_mat_vec_dispatch(
                    &ctx,
                    enc,
                    &gb.out_proj,
                    &s.gdn_normed,
                    &s.mixer_out,
                    v_dim,
                    h,
                )?;
            }
        }
        Ok(())
    })?;
    report("out", "matvec_seq", out_bytes, wall, gpu);

    let (wall, gpu) = time_gpu_reps(&ctx, warmup, iters, |enc| {
        for gb in &gdn_blocks {
            encode_mat_mat_dispatch(
                &ctx,
                enc,
                &gb.out_proj,
                &gdn_normed_batch,
                &out_batch,
                v_dim,
                h,
                tokens,
            )?;
        }
        Ok(())
    })?;
    report("out", "matmat_batch", out_bytes, wall, gpu);

    Ok(())
}

fn run_decode_proj_batch(args: DecodeProjBatchArgs) -> Result<()> {
    let DecodeProjBatchArgs {
        model,
        mut tokens,
        iters,
        warmup,
    } = args;
    if iters == 0 {
        return Err(anyhow!("--iters must be >= 1"));
    }
    if tokens.is_empty() || tokens.iter().any(|&n| n == 0) {
        return Err(anyhow!("--tokens entries must be >= 1"));
    }
    tokens.sort_unstable();
    tokens.dedup();
    let max_tokens = *tokens.last().expect("non-empty tokens");

    let ctx = MetalContext::new().context("init MetalContext")?;
    let g = GgufFile::open(&model).with_context(|| format!("open {}", model.display()))?;
    let m = Model::from_gguf(&g).context("parse model")?;
    let mm = MetalModel::load(&ctx, &g, &m).context("metal-load model")?;
    let arch = &mm.arch;
    let h = arch.hidden_size as usize;
    let head_dim = arch.attn_head_dim as usize;
    let n_q = arch.n_q_heads as usize;
    let n_kv = arch.n_kv_heads as usize;
    let q_dim = n_q * head_dim;
    let kv_dim = n_kv * head_dim;
    let n_v = arch.gdn_n_v_heads as usize;
    let n_k = arch.gdn_n_k_heads as usize;
    let gdn_head_dim = arch.gdn_head_dim as usize;
    let conv_dim = (2 * n_k + n_v) * gdn_head_dim;
    let v_dim = n_v * gdn_head_dim;
    let ffn_dim = if arch.kind == qwen_llm::model::ArchKind::Moe {
        arch.expert_shared_feed_forward_length as usize
    } else {
        arch.intermediate_size as usize
    };

    let gdn_blocks: Vec<_> = mm
        .blocks
        .iter()
        .filter_map(|b| match b {
            MetalBlock::Gdn(g) => Some(g),
            MetalBlock::Attn(_) => None,
        })
        .collect();
    let attn_blocks: Vec<_> = mm
        .blocks
        .iter()
        .filter_map(|b| match b {
            MetalBlock::Gdn(_) => None,
            MetalBlock::Attn(a) => Some(a),
        })
        .collect();

    let mut groups = Vec::new();
    if !gdn_blocks.is_empty() {
        let mut weights = Vec::with_capacity(gdn_blocks.len() * 2);
        for gb in &gdn_blocks {
            weights.push(ProjectionWeight {
                weight: &gb.in_proj_qkv,
                n_out: conv_dim,
            });
            weights.push(ProjectionWeight {
                weight: &gb.in_proj_z,
                n_out: v_dim,
            });
        }
        groups.push(ProjectionBatchBench::new(
            &ctx,
            "gdn_qkv_z",
            h,
            max_tokens,
            gdn_blocks.len(),
            weights,
        )?);

        let weights = gdn_blocks
            .iter()
            .map(|gb| ProjectionWeight {
                weight: &gb.out_proj,
                n_out: h,
            })
            .collect();
        groups.push(ProjectionBatchBench::new(
            &ctx,
            "gdn_out",
            v_dim,
            max_tokens,
            gdn_blocks.len(),
            weights,
        )?);
    }

    if !attn_blocks.is_empty() {
        let mut weights = Vec::with_capacity(attn_blocks.len() * 3);
        for ab in &attn_blocks {
            weights.push(ProjectionWeight {
                weight: &ab.q,
                n_out: 2 * q_dim,
            });
            weights.push(ProjectionWeight {
                weight: &ab.k,
                n_out: kv_dim,
            });
            weights.push(ProjectionWeight {
                weight: &ab.v,
                n_out: kv_dim,
            });
        }
        groups.push(ProjectionBatchBench::new(
            &ctx,
            "attn_qkv",
            h,
            max_tokens,
            attn_blocks.len(),
            weights,
        )?);

        let weights = attn_blocks
            .iter()
            .map(|ab| ProjectionWeight {
                weight: &ab.o,
                n_out: h,
            })
            .collect();
        groups.push(ProjectionBatchBench::new(
            &ctx,
            "attn_o",
            q_dim,
            max_tokens,
            attn_blocks.len(),
            weights,
        )?);
    }

    if ffn_dim > 0 {
        let mut gate_up = Vec::with_capacity(mm.blocks.len() * 2);
        let mut down = Vec::with_capacity(mm.blocks.len());
        for block in &mm.blocks {
            match block {
                MetalBlock::Gdn(gb) => {
                    gate_up.push(ProjectionWeight {
                        weight: &gb.ffn_gate,
                        n_out: ffn_dim,
                    });
                    gate_up.push(ProjectionWeight {
                        weight: &gb.ffn_up,
                        n_out: ffn_dim,
                    });
                    down.push(ProjectionWeight {
                        weight: &gb.ffn_down,
                        n_out: h,
                    });
                }
                MetalBlock::Attn(ab) => {
                    gate_up.push(ProjectionWeight {
                        weight: &ab.ffn_gate,
                        n_out: ffn_dim,
                    });
                    gate_up.push(ProjectionWeight {
                        weight: &ab.ffn_up,
                        n_out: ffn_dim,
                    });
                    down.push(ProjectionWeight {
                        weight: &ab.ffn_down,
                        n_out: h,
                    });
                }
            }
        }
        groups.push(ProjectionBatchBench::new(
            &ctx,
            "ffn_gate_up_dense_or_shared",
            h,
            max_tokens,
            mm.blocks.len(),
            gate_up,
        )?);
        groups.push(ProjectionBatchBench::new(
            &ctx,
            "ffn_down_dense_or_shared",
            ffn_dim,
            max_tokens,
            mm.blocks.len(),
            down,
        )?);
    }

    groups.push(ProjectionBatchBench::new(
        &ctx,
        "lm_head",
        h,
        max_tokens,
        1,
        vec![ProjectionWeight {
            weight: &mm.lm_head,
            n_out: arch.vocab_size as usize,
        }],
    )?);

    projection_fill_inputs(&ctx, &groups)?;

    println!(
        "[decode-proj-batch] model={} kind={:?} layers={} gdn_layers={} attn_layers={} h={} q_dim={} kv_dim={} conv_dim={} v_dim={} ffn_dim={} vocab={} tokens={} warmup={} iters={}",
        model.display(),
        arch.kind,
        mm.blocks.len(),
        gdn_blocks.len(),
        attn_blocks.len(),
        h,
        q_dim,
        kv_dim,
        conv_dim,
        v_dim,
        ffn_dim,
        arch.vocab_size,
        tokens
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join(","),
        warmup,
        iters
    );
    println!(
        "component\tmode\ttokens\tn_in\tmax_out\tweights\tweight_gb_per_token\tdispatches_per_token\tavg_wall_ms\tavg_gpu_ms\tavg_gpu_ms_per_tok\tp50_gpu_ms_per_tok\tp90_gpu_ms_per_tok\tmax_gpu_ms_per_tok\teff_weight_gb_s\tsaving_ms_per_tok\tsaving_pct"
    );

    for &n_tokens in &tokens {
        let mut sum_seq_gpu = 0.0f64;
        let mut sum_batch_gpu = 0.0f64;
        let mut total_weight_bytes = 0u64;
        let mut total_weights = 0usize;

        for group in &groups {
            let seq = time_gpu_reps_stats(&ctx, warmup, iters, |enc| {
                encode_projection_group_matvec_seq(&ctx, enc, group, n_tokens)
            })?;
            let batch = time_gpu_reps_stats(&ctx, warmup, iters, |enc| {
                encode_projection_group_matmat_batch(&ctx, enc, group, n_tokens)
            })?;
            let seq_per_tok = seq.avg_gpu_ms / n_tokens as f64;
            let batch_per_tok = batch.avg_gpu_ms / n_tokens as f64;
            let save = seq_per_tok - batch_per_tok;
            let pct = if seq_per_tok > 0.0 {
                save / seq_per_tok * 100.0
            } else {
                0.0
            };
            let weight_gb = group.weight_bytes as f64 / 1e9;
            let seq_gb_s = weight_gb * n_tokens as f64 / (seq.avg_gpu_ms / 1e3);
            let batch_gb_s = weight_gb * n_tokens as f64 / (batch.avg_gpu_ms / 1e3);
            let seq_dispatch = group.weights.len() as f64;
            let batch_dispatch = group.weights.len() as f64 / n_tokens as f64;
            let pack_bytes = projection_group_pack_bytes(group);
            let scatter_bytes = projection_group_scatter_bytes(group);
            let pack = time_cmd_reps_stats(&ctx, warmup, iters, |cmd| {
                let blit = BlitEncoder::begin(cmd);
                blit_projection_group_pack(&blit, group, n_tokens);
                blit.end();
                Ok(())
            })?;
            let scatter = time_cmd_reps_stats(&ctx, warmup, iters, |cmd| {
                let blit = BlitEncoder::begin(cmd);
                blit_projection_group_scatter(&blit, group, n_tokens);
                blit.end();
                Ok(())
            })?;
            let with_layout = time_cmd_reps_stats(&ctx, warmup, iters, |cmd| {
                let blit = BlitEncoder::begin(cmd);
                blit_projection_group_pack(&blit, group, n_tokens);
                blit.end();
                let enc = KernelEncoder::begin(cmd);
                encode_projection_group_matmat_batch(&ctx, &enc, group, n_tokens)?;
                enc.end();
                let blit = BlitEncoder::begin(cmd);
                blit_projection_group_scatter(&blit, group, n_tokens);
                blit.end();
                Ok(())
            })?;
            let pack_gb = pack_bytes as f64 / 1e9;
            let scatter_gb = scatter_bytes as f64 / 1e9;
            let pack_gb_s = pack_gb * n_tokens as f64 / (pack.avg_gpu_ms / 1e3);
            let scatter_gb_s = scatter_gb * n_tokens as f64 / (scatter.avg_gpu_ms / 1e3);
            let with_layout_per_tok = with_layout.avg_gpu_ms / n_tokens as f64;
            let with_layout_save = seq_per_tok - with_layout_per_tok;
            let with_layout_pct = if seq_per_tok > 0.0 {
                with_layout_save / seq_per_tok * 100.0
            } else {
                0.0
            };
            println!(
                "{}\tmatvec_seq\t{}\t{}\t{}\t{}\t{:.4}\t{:.2}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.1}\t{:.4}\t{:.1}",
                group.name,
                n_tokens,
                group.n_in,
                group.max_out,
                group.weights.len(),
                weight_gb,
                seq_dispatch,
                seq.avg_wall_ms,
                seq.avg_gpu_ms,
                seq_per_tok,
                seq.p50_gpu_ms / n_tokens as f64,
                seq.p90_gpu_ms / n_tokens as f64,
                seq.max_gpu_ms / n_tokens as f64,
                seq_gb_s,
                0.0,
                0.0
            );
            println!(
                "{}\tmatmat_batch\t{}\t{}\t{}\t{}\t{:.4}\t{:.2}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.1}\t{:.4}\t{:.1}",
                group.name,
                n_tokens,
                group.n_in,
                group.max_out,
                group.weights.len(),
                weight_gb,
                batch_dispatch,
                batch.avg_wall_ms,
                batch.avg_gpu_ms,
                batch_per_tok,
                batch.p50_gpu_ms / n_tokens as f64,
                batch.p90_gpu_ms / n_tokens as f64,
                batch.max_gpu_ms / n_tokens as f64,
                batch_gb_s,
                save,
                pct
            );
            println!(
                "{}\tlayout_pack\t{}\t{}\t{}\t{}\t{:.4}\t{:.2}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.1}\t{:.4}\t{:.1}",
                group.name,
                n_tokens,
                group.n_in,
                group.max_out,
                group.input_pack_count,
                pack_gb,
                group.input_pack_count as f64,
                pack.avg_wall_ms,
                pack.avg_gpu_ms,
                pack.avg_gpu_ms / n_tokens as f64,
                pack.p50_gpu_ms / n_tokens as f64,
                pack.p90_gpu_ms / n_tokens as f64,
                pack.max_gpu_ms / n_tokens as f64,
                pack_gb_s,
                0.0,
                0.0
            );
            println!(
                "{}\tlayout_scatter\t{}\t{}\t{}\t{}\t{:.4}\t{:.2}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.1}\t{:.4}\t{:.1}",
                group.name,
                n_tokens,
                group.n_in,
                group.max_out,
                group.weights.len(),
                scatter_gb,
                group.weights.len() as f64,
                scatter.avg_wall_ms,
                scatter.avg_gpu_ms,
                scatter.avg_gpu_ms / n_tokens as f64,
                scatter.p50_gpu_ms / n_tokens as f64,
                scatter.p90_gpu_ms / n_tokens as f64,
                scatter.max_gpu_ms / n_tokens as f64,
                scatter_gb_s,
                0.0,
                0.0
            );
            println!(
                "{}\tmatmat_with_layout\t{}\t{}\t{}\t{}\t{:.4}\t{:.2}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.1}\t{:.4}\t{:.1}",
                group.name,
                n_tokens,
                group.n_in,
                group.max_out,
                group.weights.len(),
                weight_gb + pack_gb + scatter_gb,
                batch_dispatch + group.input_pack_count as f64 + group.weights.len() as f64,
                with_layout.avg_wall_ms,
                with_layout.avg_gpu_ms,
                with_layout_per_tok,
                with_layout.p50_gpu_ms / n_tokens as f64,
                with_layout.p90_gpu_ms / n_tokens as f64,
                with_layout.max_gpu_ms / n_tokens as f64,
                (weight_gb + pack_gb + scatter_gb) * n_tokens as f64
                    / (with_layout.avg_gpu_ms / 1e3),
                with_layout_save,
                with_layout_pct
            );
            sum_seq_gpu += seq.avg_gpu_ms;
            sum_batch_gpu += batch.avg_gpu_ms;
            total_weight_bytes += group.weight_bytes;
            total_weights += group.weights.len();
        }

        let seq_all = time_gpu_reps_stats(&ctx, warmup, iters, |enc| {
            for group in &groups {
                encode_projection_group_matvec_seq(&ctx, enc, group, n_tokens)?;
            }
            Ok(())
        })?;
        let batch_all = time_gpu_reps_stats(&ctx, warmup, iters, |enc| {
            for group in &groups {
                encode_projection_group_matmat_batch(&ctx, enc, group, n_tokens)?;
            }
            Ok(())
        })?;
        let with_layout_all = time_cmd_reps_stats(&ctx, warmup, iters, |cmd| {
            let blit = BlitEncoder::begin(cmd);
            for group in &groups {
                blit_projection_group_pack(&blit, group, n_tokens);
            }
            blit.end();
            let enc = KernelEncoder::begin(cmd);
            for group in &groups {
                encode_projection_group_matmat_batch(&ctx, &enc, group, n_tokens)?;
            }
            enc.end();
            let blit = BlitEncoder::begin(cmd);
            for group in &groups {
                blit_projection_group_scatter(&blit, group, n_tokens);
            }
            blit.end();
            Ok(())
        })?;

        let isolated_save = (sum_seq_gpu - sum_batch_gpu) / n_tokens as f64;
        let one_encoder_seq_per_tok = seq_all.avg_gpu_ms / n_tokens as f64;
        let one_encoder_batch_per_tok = batch_all.avg_gpu_ms / n_tokens as f64;
        let one_encoder_save = one_encoder_seq_per_tok - one_encoder_batch_per_tok;
        let with_layout_per_tok = with_layout_all.avg_gpu_ms / n_tokens as f64;
        let with_layout_save = one_encoder_seq_per_tok - with_layout_per_tok;
        let one_encoder_pct = if one_encoder_seq_per_tok > 0.0 {
            one_encoder_save / one_encoder_seq_per_tok * 100.0
        } else {
            0.0
        };
        let with_layout_pct = if one_encoder_seq_per_tok > 0.0 {
            with_layout_save / one_encoder_seq_per_tok * 100.0
        } else {
            0.0
        };
        let weight_gb = total_weight_bytes as f64 / 1e9;
        let layout_bytes: u64 = groups
            .iter()
            .map(|group| projection_group_pack_bytes(group) + projection_group_scatter_bytes(group))
            .sum();
        let layout_gb = layout_bytes as f64 / 1e9;
        let seq_gb_s = weight_gb * n_tokens as f64 / (seq_all.avg_gpu_ms / 1e3);
        let batch_gb_s = weight_gb * n_tokens as f64 / (batch_all.avg_gpu_ms / 1e3);
        println!(
            "aggregate_isolated\tsummed_groups\t{}\t0\t0\t{}\t{:.4}\t{:.2}\t0.0000\t{:.4}\t{:.4}\t0.0000\t0.0000\t0.0000\t0.0\t{:.4}\t0.0",
            n_tokens,
            total_weights,
            weight_gb,
            total_weights as f64,
            sum_seq_gpu,
            sum_seq_gpu / n_tokens as f64,
            isolated_save
        );
        println!(
            "aggregate_isolated\tsummed_matmat\t{}\t0\t0\t{}\t{:.4}\t{:.2}\t0.0000\t{:.4}\t{:.4}\t0.0000\t0.0000\t0.0000\t0.0\t{:.4}\t{:.1}",
            n_tokens,
            total_weights,
            weight_gb,
            total_weights as f64 / n_tokens as f64,
            sum_batch_gpu,
            sum_batch_gpu / n_tokens as f64,
            isolated_save,
            if sum_seq_gpu > 0.0 {
                (sum_seq_gpu - sum_batch_gpu) / sum_seq_gpu * 100.0
            } else {
                0.0
            }
        );
        println!(
            "aggregate_one_encoder\tmatvec_seq\t{}\t0\t0\t{}\t{:.4}\t{:.2}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.1}\t{:.4}\t0.0",
            n_tokens,
            total_weights,
            weight_gb,
            total_weights as f64,
            seq_all.avg_wall_ms,
            seq_all.avg_gpu_ms,
            one_encoder_seq_per_tok,
            seq_all.p50_gpu_ms / n_tokens as f64,
            seq_all.p90_gpu_ms / n_tokens as f64,
            seq_all.max_gpu_ms / n_tokens as f64,
            seq_gb_s,
            0.0
        );
        println!(
            "aggregate_one_encoder\tmatmat_batch\t{}\t0\t0\t{}\t{:.4}\t{:.2}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.1}\t{:.4}\t{:.1}",
            n_tokens,
            total_weights,
            weight_gb,
            total_weights as f64 / n_tokens as f64,
            batch_all.avg_wall_ms,
            batch_all.avg_gpu_ms,
            one_encoder_batch_per_tok,
            batch_all.p50_gpu_ms / n_tokens as f64,
            batch_all.p90_gpu_ms / n_tokens as f64,
            batch_all.max_gpu_ms / n_tokens as f64,
            batch_gb_s,
            one_encoder_save,
            one_encoder_pct
        );
        println!(
            "aggregate_one_encoder\tmatmat_with_layout\t{}\t0\t0\t{}\t{:.4}\t{:.2}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.1}\t{:.4}\t{:.1}",
            n_tokens,
            total_weights,
            weight_gb + layout_gb,
            total_weights as f64 / n_tokens as f64,
            with_layout_all.avg_wall_ms,
            with_layout_all.avg_gpu_ms,
            with_layout_per_tok,
            with_layout_all.p50_gpu_ms / n_tokens as f64,
            with_layout_all.p90_gpu_ms / n_tokens as f64,
            with_layout_all.max_gpu_ms / n_tokens as f64,
            (weight_gb + layout_gb) * n_tokens as f64 / (with_layout_all.avg_gpu_ms / 1e3),
            with_layout_save,
            with_layout_pct
        );
    }

    Ok(())
}

struct GdnLayerReplayScratch {
    h_pack: MetalTensor,
    qkv_pack: MetalTensor,
    z_pack: MetalTensor,
    normed_pack: MetalTensor,
    out_pack: MetalTensor,
}

impl GdnLayerReplayScratch {
    fn new(
        ctx: &MetalContext,
        max_tokens: usize,
        h: usize,
        conv_dim: usize,
        v_dim: usize,
    ) -> Result<Self> {
        Ok(Self {
            h_pack: MetalTensor::zeros_f32(ctx, vec![(max_tokens * h) as u64])?,
            qkv_pack: MetalTensor::zeros_f32(ctx, vec![(max_tokens * conv_dim) as u64])?,
            z_pack: MetalTensor::zeros_f32(ctx, vec![(max_tokens * v_dim) as u64])?,
            normed_pack: MetalTensor::zeros_f32(ctx, vec![(max_tokens * v_dim) as u64])?,
            out_pack: MetalTensor::zeros_f32(ctx, vec![(max_tokens * h) as u64])?,
        })
    }
}

fn fill_gdn_replay_inputs(ctx: &MetalContext, sessions: &[MetalSession]) -> Result<()> {
    let cmd = ctx.queue.commandBuffer().context("gdn replay fill cmd")?;
    let enc = KernelEncoder::begin(&cmd);
    for (slot, s) in sessions.iter().enumerate() {
        let v = 0.03125 + (slot as f32) * 0.0009765625;
        encode_fill_f32(ctx, &enc, &s.x, v)?;
    }
    enc.end();
    cmd.commit();
    cmd.waitUntilCompleted();
    Ok(())
}

fn read_f32_tensor(t: &MetalTensor) -> Vec<f32> {
    let n = t.n_elements() as usize;
    let mut xs = vec![0.0f32; n];
    unsafe {
        let src = (t.buffer.contents().as_ptr() as *const f32).add((t.offset / 4) as usize);
        std::ptr::copy_nonoverlapping(src, xs.as_mut_ptr(), n);
    }
    xs
}

fn read_f32_tensor_prefix(t: &MetalTensor, n: usize) -> Vec<f32> {
    let n = n.min(t.n_elements() as usize);
    let mut xs = vec![0.0f32; n];
    unsafe {
        let src = (t.buffer.contents().as_ptr() as *const f32).add((t.offset / 4) as usize);
        std::ptr::copy_nonoverlapping(src, xs.as_mut_ptr(), n);
    }
    xs
}

fn read_i32_tensor_prefix(t: &MetalTensor, n: usize) -> Vec<i32> {
    let n = n.min(t.n_elements() as usize);
    let mut xs = vec![0i32; n];
    unsafe {
        let src = (t.buffer.contents().as_ptr() as *const i32).add((t.offset / 4) as usize);
        std::ptr::copy_nonoverlapping(src, xs.as_mut_ptr(), n);
    }
    xs
}

fn fmt_i32_csv(xs: &[i32]) -> String {
    xs.iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

struct RouteFingerprint {
    idx: Vec<i32>,
    weight: Vec<f32>,
    shared_gate: f32,
    logit_margin: f32,
    logits: Vec<f32>,
}

fn topk_logit_margin(logits: &[f32], topk: usize) -> f32 {
    if topk == 0 || logits.len() <= topk {
        return 0.0;
    }
    let mut ranked: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    ranked[topk - 1].1 - ranked[topk].1
}

fn read_route_fingerprint(s: &MetalSession, topk: usize, n_expert: usize) -> RouteFingerprint {
    let idx = read_i32_tensor_prefix(&s.moe_topk_idx, topk);
    let weight = read_f32_tensor_prefix(&s.moe_topk_weight, topk);
    let shared_gate = read_f32_tensor_prefix(&s.moe_shared_gate, 1)
        .into_iter()
        .next()
        .unwrap_or(0.0);
    let logits = read_f32_tensor_prefix(&s.moe_router_probs, n_expert);
    RouteFingerprint {
        idx,
        weight,
        shared_gate,
        logit_margin: topk_logit_margin(&logits, topk),
        logits,
    }
}

fn f32_max_abs_delta(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn f32_rms_delta(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let ss = a
        .iter()
        .zip(b)
        .take(n)
        .map(|(x, y)| {
            let d = *x as f64 - *y as f64;
            d * d
        })
        .sum::<f64>();
    (ss / n as f64).sqrt()
}

fn route_weight_max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn same_i32_set(a: &[i32], b: &[i32]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut aa = a.to_vec();
    let mut bb = b.to_vec();
    aa.sort_unstable();
    bb.sort_unstable();
    aa == bb
}

fn cosine_max_abs(a: &[f32], b: &[f32]) -> (f64, f32) {
    let max_abs = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let cos = if na > 0.0 && nb > 0.0 {
        dot / (na * nb)
    } else {
        1.0
    };
    (cos, max_abs)
}

fn fresh_gdn_replay_sessions(
    ctx: &MetalContext,
    mm: &MetalModel,
    n: usize,
) -> Result<Vec<MetalSession>> {
    fresh_gdn_replay_sessions_with_capacity(ctx, mm, n, 32)
}

fn fresh_gdn_replay_sessions_with_capacity(
    ctx: &MetalContext,
    mm: &MetalModel,
    n: usize,
    kv_capacity: usize,
) -> Result<Vec<MetalSession>> {
    let mut sessions = Vec::with_capacity(n);
    for i in 0..n {
        sessions.push(
            MetalSession::fresh(ctx, mm, kv_capacity)
                .with_context(|| format!("fresh replay session {i}"))?,
        );
    }
    Ok(sessions)
}

fn encode_gdn_layer_baseline(
    ctx: &MetalContext,
    mf: &MetalForward<'_>,
    enc: &KernelEncoder,
    gb: &qwen_llm::metal_forward::MetalGdnBlock,
    gdn_i: usize,
    sessions: &mut [MetalSession],
) -> Result<()> {
    for s in sessions {
        encode_rms_norm_mul_f32(ctx, enc, &s.x, &gb.attn_norm, &s.h, RMS_EPS)?;
        mf.encode_gdn(enc, gb, gdn_i, s)?;
        encode_add_inplace_f32(ctx, enc, &s.x, &s.mixer_out)?;
        encode_rms_norm_mul_f32(ctx, enc, &s.x, &gb.post_attn_norm, &s.h, RMS_EPS)?;
    }
    Ok(())
}

fn encode_gdn_layer_replay(
    ctx: &MetalContext,
    mf: &MetalForward<'_>,
    cmd: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    gb: &qwen_llm::metal_forward::MetalGdnBlock,
    gdn_i: usize,
    sessions: &mut [MetalSession],
    scratch: &GdnLayerReplayScratch,
    h: usize,
    conv_dim: usize,
    v_dim: usize,
) -> Result<()> {
    let tokens = sessions.len();
    let enc = KernelEncoder::begin(cmd);
    for s in sessions.iter() {
        encode_rms_norm_mul_f32(ctx, &enc, &s.x, &gb.attn_norm, &s.h, RMS_EPS)?;
    }
    enc.end();

    let row_bytes = (h * std::mem::size_of::<f32>()) as u64;
    let blit = BlitEncoder::begin(cmd);
    for (tok, s) in sessions.iter().enumerate() {
        blit.copy_buffer(
            &s.h.buffer,
            s.h.offset,
            &scratch.h_pack.buffer,
            scratch.h_pack.offset + tok as u64 * row_bytes,
            row_bytes,
        );
    }
    blit.end();

    let enc = KernelEncoder::begin(cmd);
    let h_pack = scratch.h_pack.view_subrange(0, vec![(tokens * h) as u64]);
    let qkv_pack = scratch
        .qkv_pack
        .view_subrange(0, vec![(tokens * conv_dim) as u64]);
    let z_pack = scratch
        .z_pack
        .view_subrange(0, vec![(tokens * v_dim) as u64]);
    encode_mat_mat_dispatch(
        ctx,
        &enc,
        &gb.in_proj_qkv,
        &h_pack,
        &qkv_pack,
        h,
        conv_dim,
        tokens,
    )?;
    encode_mat_mat_dispatch(ctx, &enc, &gb.in_proj_z, &h_pack, &z_pack, h, v_dim, tokens)?;

    for (tok, s) in sessions.iter_mut().enumerate() {
        encode_mat_vec_dispatch(
            ctx,
            &enc,
            &gb.beta_proj,
            &s.h,
            &s.gdn_b,
            h,
            s.gdn_b.n_elements() as usize,
        )?;
        encode_mat_vec_dispatch(
            ctx,
            &enc,
            &gb.alpha_proj,
            &s.h,
            &s.gdn_a,
            h,
            s.gdn_a.n_elements() as usize,
        )?;
        encode_sigmoid_f32(ctx, &enc, &s.gdn_b, &s.gdn_beta)?;
        encode_gdn_decay_chain_f32(ctx, &enc, &s.gdn_a, &gb.dt_bias, &gb.a_log, &s.gdn_alpha)?;

        let qkv_row = scratch
            .qkv_pack
            .view_subrange((tok * conv_dim) as u64, vec![conv_dim as u64]);
        let z_row = scratch
            .z_pack
            .view_subrange((tok * v_dim) as u64, vec![v_dim as u64]);
        let normed_row = scratch
            .normed_pack
            .view_subrange((tok * v_dim) as u64, vec![v_dim as u64]);
        let alpha = s.gdn_alpha.clone();
        let beta = s.gdn_beta.clone();
        mf.encode_gdn_tail(
            &enc,
            gb,
            gdn_i,
            s,
            &qkv_row,
            &z_row,
            &alpha,
            &beta,
            &normed_row,
        )?;
    }

    let normed_pack = scratch
        .normed_pack
        .view_subrange(0, vec![(tokens * v_dim) as u64]);
    let out_pack = scratch.out_pack.view_subrange(0, vec![(tokens * h) as u64]);
    encode_mat_mat_dispatch(
        ctx,
        &enc,
        &gb.out_proj,
        &normed_pack,
        &out_pack,
        v_dim,
        h,
        tokens,
    )?;

    for (tok, s) in sessions.iter().enumerate() {
        let out_row = scratch
            .out_pack
            .view_subrange((tok * h) as u64, vec![h as u64]);
        encode_add_inplace_f32(ctx, &enc, &s.x, &out_row)?;
        encode_rms_norm_mul_f32(ctx, &enc, &s.x, &gb.post_attn_norm, &s.h, RMS_EPS)?;
    }
    enc.end();
    Ok(())
}

#[derive(Clone, Copy)]
struct SelectedGdnLayer<'a> {
    block_i: usize,
    gdn_i: usize,
    gb: &'a qwen_llm::metal_forward::MetalGdnBlock,
}

fn collect_gdn_layers(mm: &MetalModel) -> Vec<SelectedGdnLayer<'_>> {
    mm.blocks
        .iter()
        .enumerate()
        .filter_map(|(block_i, b)| match b {
            MetalBlock::Gdn(gb) => Some((block_i, gb)),
            MetalBlock::Attn(_) => None,
        })
        .enumerate()
        .map(|(gdn_i, (block_i, gb))| SelectedGdnLayer { block_i, gdn_i, gb })
        .collect()
}

fn run_decode_gdn_layer_replay(args: DecodeGdnLayerReplayArgs) -> Result<()> {
    let DecodeGdnLayerReplayArgs {
        model,
        mut tokens,
        block,
        mut gdn_indexes,
        sample_gdn_layers,
        iters,
        warmup,
        no_check,
    } = args;
    if iters == 0 {
        return Err(anyhow!("--iters must be >= 1"));
    }
    if tokens.is_empty() || tokens.iter().any(|&n| n == 0) {
        return Err(anyhow!("--tokens entries must be >= 1"));
    }
    if block.is_some() && (!gdn_indexes.is_empty() || sample_gdn_layers) {
        return Err(anyhow!(
            "--block conflicts with --gdn-index and --sample-gdn-layers"
        ));
    }
    if sample_gdn_layers && !gdn_indexes.is_empty() {
        return Err(anyhow!("--sample-gdn-layers conflicts with --gdn-index"));
    }
    tokens.sort_unstable();
    tokens.dedup();
    gdn_indexes.sort_unstable();
    gdn_indexes.dedup();
    let max_tokens = *tokens.last().expect("non-empty tokens");

    let ctx = MetalContext::new().context("init MetalContext")?;
    let g = GgufFile::open(&model).with_context(|| format!("open {}", model.display()))?;
    let m = Model::from_gguf(&g).context("parse model")?;
    let mm = MetalModel::load(&ctx, &g, &m).context("metal-load model")?;
    let mf = MetalForward::new(&ctx, &mm);
    let arch = &mm.arch;
    let h = arch.hidden_size as usize;
    let n_v = arch.gdn_n_v_heads as usize;
    let n_k = arch.gdn_n_k_heads as usize;
    let head_dim = arch.gdn_head_dim as usize;
    let conv_dim = (2 * n_k + n_v) * head_dim;
    let v_dim = n_v * head_dim;

    let gdn_layers = collect_gdn_layers(&mm);
    if gdn_layers.is_empty() {
        return Err(anyhow!("model has no GDN blocks"));
    }

    let selected: Vec<_> = if let Some(wanted) = block {
        vec![
            *gdn_layers
                .iter()
                .find(|layer| layer.block_i == wanted)
                .ok_or_else(|| anyhow!("block {wanted} is not a GDN block"))?,
        ]
    } else if !gdn_indexes.is_empty() {
        let mut xs = Vec::with_capacity(gdn_indexes.len());
        for &gdn_i in &gdn_indexes {
            xs.push(*gdn_layers.get(gdn_i).ok_or_else(|| {
                anyhow!(
                    "gdn-index {gdn_i} outside available 0..{}",
                    gdn_layers.len().saturating_sub(1)
                )
            })?);
        }
        xs
    } else if sample_gdn_layers {
        let mut idxs = vec![0, gdn_layers.len() / 2, gdn_layers.len() - 1];
        idxs.sort_unstable();
        idxs.dedup();
        idxs.into_iter().map(|i| gdn_layers[i]).collect()
    } else {
        vec![gdn_layers[0]]
    };

    println!(
        "[decode-gdn-layer-replay] model={} selected={} gdn_layers={} layers={} h={} conv_dim={} v_dim={} tokens={} warmup={} iters={}",
        model.display(),
        selected.len(),
        gdn_layers.len(),
        mm.blocks.len(),
        h,
        conv_dim,
        v_dim,
        tokens
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join(","),
        warmup,
        iters
    );
    println!(
        "mode\tblock\tgdn_i\ttokens\tencoders_per_batch\tdispatches_per_token\tavg_wall_ms\tavg_gpu_ms\tavg_gpu_ms_per_tok\tp50_gpu_ms_per_tok\tp90_gpu_ms_per_tok\tmax_gpu_ms_per_tok\tsaving_ms_per_tok\tsaving_pct"
    );

    for layer in selected {
        if !no_check {
            let check_tokens = max_tokens;
            let mut base = fresh_gdn_replay_sessions(&ctx, &mm, check_tokens)?;
            let mut replay = fresh_gdn_replay_sessions(&ctx, &mm, check_tokens)?;
            fill_gdn_replay_inputs(&ctx, &base)?;
            fill_gdn_replay_inputs(&ctx, &replay)?;
            let scratch = GdnLayerReplayScratch::new(&ctx, check_tokens, h, conv_dim, v_dim)?;

            let cmd = ctx
                .queue
                .commandBuffer()
                .context("gdn replay check baseline cmd")?;
            let enc = KernelEncoder::begin(&cmd);
            encode_gdn_layer_baseline(&ctx, &mf, &enc, layer.gb, layer.gdn_i, &mut base)?;
            enc.end();
            cmd.commit();
            cmd.waitUntilCompleted();

            let cmd = ctx
                .queue
                .commandBuffer()
                .context("gdn replay check replay cmd")?;
            encode_gdn_layer_replay(
                &ctx,
                &mf,
                &cmd,
                layer.gb,
                layer.gdn_i,
                &mut replay,
                &scratch,
                h,
                conv_dim,
                v_dim,
            )?;
            cmd.commit();
            cmd.waitUntilCompleted();

            let mut min_cos = 1.0f64;
            let mut max_abs_all = 0.0f32;
            let mut worst_slot = 0usize;
            for i in 0..check_tokens {
                let (cos, max_abs) =
                    cosine_max_abs(&read_f32_tensor(&base[i].h), &read_f32_tensor(&replay[i].h));
                if cos < min_cos || max_abs > max_abs_all {
                    worst_slot = i;
                }
                min_cos = min_cos.min(cos);
                max_abs_all = max_abs_all.max(max_abs);
            }
            println!(
                "check\t{}\t{}\t{}\tmin_cos_h={min_cos:.9}\tmax_abs_h={max_abs_all:.6}\tworst_slot={worst_slot}",
                layer.block_i, layer.gdn_i, check_tokens
            );
            if min_cos < 0.999 || max_abs_all > 1e-2 {
                return Err(anyhow!(
                    "GDN layer replay check failed: block={} gdn_i={} min_cos_h={min_cos:.9} max_abs_h={max_abs_all:.6}",
                    layer.block_i,
                    layer.gdn_i
                ));
            }
        }

        let mut baseline_sessions = fresh_gdn_replay_sessions(&ctx, &mm, max_tokens)?;
        let mut replay_sessions = fresh_gdn_replay_sessions(&ctx, &mm, max_tokens)?;
        fill_gdn_replay_inputs(&ctx, &baseline_sessions)?;
        fill_gdn_replay_inputs(&ctx, &replay_sessions)?;
        let scratch = GdnLayerReplayScratch::new(&ctx, max_tokens, h, conv_dim, v_dim)?;

        for &n_tokens in &tokens {
            let baseline = time_cmd_reps_stats(&ctx, warmup, iters, |cmd| {
                let enc = KernelEncoder::begin(cmd);
                encode_gdn_layer_baseline(
                    &ctx,
                    &mf,
                    &enc,
                    layer.gb,
                    layer.gdn_i,
                    &mut baseline_sessions[..n_tokens],
                )?;
                enc.end();
                Ok(())
            })?;
            let replay = time_cmd_reps_stats(&ctx, warmup, iters, |cmd| {
                encode_gdn_layer_replay(
                    &ctx,
                    &mf,
                    cmd,
                    layer.gb,
                    layer.gdn_i,
                    &mut replay_sessions[..n_tokens],
                    &scratch,
                    h,
                    conv_dim,
                    v_dim,
                )
            })?;

            let baseline_per_tok = baseline.avg_gpu_ms / n_tokens as f64;
            let replay_per_tok = replay.avg_gpu_ms / n_tokens as f64;
            let save = baseline_per_tok - replay_per_tok;
            let pct = if baseline_per_tok > 0.0 {
                save / baseline_per_tok * 100.0
            } else {
                0.0
            };
            println!(
                "baseline_seq\t{}\t{}\t{}\t1.00\t15.00\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t0.0000\t0.0",
                layer.block_i,
                layer.gdn_i,
                n_tokens,
                baseline.avg_wall_ms,
                baseline.avg_gpu_ms,
                baseline_per_tok,
                baseline.p50_gpu_ms / n_tokens as f64,
                baseline.p90_gpu_ms / n_tokens as f64,
                baseline.max_gpu_ms / n_tokens as f64
            );
            println!(
                "replay_batched_qkv_z_out\t{}\t{}\t{}\t3.00\t{:.2}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.1}",
                layer.block_i,
                layer.gdn_i,
                n_tokens,
                12.0 + 3.0 / n_tokens as f64,
                replay.avg_wall_ms,
                replay.avg_gpu_ms,
                replay_per_tok,
                replay.p50_gpu_ms / n_tokens as f64,
                replay.p90_gpu_ms / n_tokens as f64,
                replay.max_gpu_ms / n_tokens as f64,
                save,
                pct
            );
        }
    }

    Ok(())
}

fn encode_gdn_chain_baseline(
    ctx: &MetalContext,
    mf: &MetalForward<'_>,
    enc: &KernelEncoder,
    layers: &[SelectedGdnLayer<'_>],
    sessions: &mut [MetalSession],
) -> Result<()> {
    for layer in layers {
        encode_gdn_layer_baseline(ctx, mf, enc, layer.gb, layer.gdn_i, sessions)?;
    }
    Ok(())
}

fn encode_gdn_chain_replay(
    ctx: &MetalContext,
    mf: &MetalForward<'_>,
    cmd: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    layers: &[SelectedGdnLayer<'_>],
    sessions: &mut [MetalSession],
    scratch: &GdnLayerReplayScratch,
    h: usize,
    conv_dim: usize,
    v_dim: usize,
) -> Result<()> {
    for layer in layers {
        encode_gdn_layer_replay(
            ctx,
            mf,
            cmd,
            layer.gb,
            layer.gdn_i,
            sessions,
            scratch,
            h,
            conv_dim,
            v_dim,
        )?;
    }
    Ok(())
}

fn run_decode_gdn_chain_replay(args: DecodeGdnChainReplayArgs) -> Result<()> {
    let DecodeGdnChainReplayArgs {
        model,
        mut tokens,
        start_gdn,
        n_layers,
        iters,
        warmup,
        no_check,
    } = args;
    if iters == 0 {
        return Err(anyhow!("--iters must be >= 1"));
    }
    if n_layers == 0 {
        return Err(anyhow!("--layers must be >= 1"));
    }
    if tokens.is_empty() || tokens.iter().any(|&n| n == 0) {
        return Err(anyhow!("--tokens entries must be >= 1"));
    }
    tokens.sort_unstable();
    tokens.dedup();
    let max_tokens = *tokens.last().expect("non-empty tokens");

    let ctx = MetalContext::new().context("init MetalContext")?;
    let g = GgufFile::open(&model).with_context(|| format!("open {}", model.display()))?;
    let m = Model::from_gguf(&g).context("parse model")?;
    let mm = MetalModel::load(&ctx, &g, &m).context("metal-load model")?;
    let mf = MetalForward::new(&ctx, &mm);
    let arch = &mm.arch;
    let h = arch.hidden_size as usize;
    let n_v = arch.gdn_n_v_heads as usize;
    let n_k = arch.gdn_n_k_heads as usize;
    let head_dim = arch.gdn_head_dim as usize;
    let conv_dim = (2 * n_k + n_v) * head_dim;
    let v_dim = n_v * head_dim;

    let gdn_layers = collect_gdn_layers(&mm);
    let end_gdn = start_gdn
        .checked_add(n_layers)
        .ok_or_else(|| anyhow!("start_gdn + layers overflow"))?;
    if end_gdn > gdn_layers.len() {
        return Err(anyhow!(
            "GDN chain {}..{} outside available 0..{}",
            start_gdn,
            end_gdn,
            gdn_layers.len()
        ));
    }
    let selected = &gdn_layers[start_gdn..end_gdn];
    let block_list = selected
        .iter()
        .map(|layer| layer.block_i.to_string())
        .collect::<Vec<_>>()
        .join(",");

    println!(
        "[decode-gdn-chain-replay] model={} start_gdn={} chain_layers={} blocks={} gdn_layers={} layers={} h={} conv_dim={} v_dim={} tokens={} warmup={} iters={}",
        model.display(),
        start_gdn,
        n_layers,
        block_list,
        gdn_layers.len(),
        mm.blocks.len(),
        h,
        conv_dim,
        v_dim,
        tokens
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join(","),
        warmup,
        iters
    );

    if !no_check {
        let check_tokens = max_tokens;
        let mut base = fresh_gdn_replay_sessions(&ctx, &mm, check_tokens)?;
        let mut replay = fresh_gdn_replay_sessions(&ctx, &mm, check_tokens)?;
        fill_gdn_replay_inputs(&ctx, &base)?;
        fill_gdn_replay_inputs(&ctx, &replay)?;
        let scratch = GdnLayerReplayScratch::new(&ctx, check_tokens, h, conv_dim, v_dim)?;

        let cmd = ctx
            .queue
            .commandBuffer()
            .context("gdn chain check baseline cmd")?;
        let enc = KernelEncoder::begin(&cmd);
        encode_gdn_chain_baseline(&ctx, &mf, &enc, selected, &mut base)?;
        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();

        let cmd = ctx
            .queue
            .commandBuffer()
            .context("gdn chain check replay cmd")?;
        encode_gdn_chain_replay(
            &ctx,
            &mf,
            &cmd,
            selected,
            &mut replay,
            &scratch,
            h,
            conv_dim,
            v_dim,
        )?;
        cmd.commit();
        cmd.waitUntilCompleted();

        let mut min_cos = 1.0f64;
        let mut max_abs_all = 0.0f32;
        let mut worst_slot = 0usize;
        for i in 0..check_tokens {
            let (cos, max_abs) =
                cosine_max_abs(&read_f32_tensor(&base[i].h), &read_f32_tensor(&replay[i].h));
            if cos < min_cos || max_abs > max_abs_all {
                worst_slot = i;
            }
            min_cos = min_cos.min(cos);
            max_abs_all = max_abs_all.max(max_abs);
        }
        println!(
            "check\ttokens={check_tokens}\tmin_cos_h={min_cos:.9}\tmax_abs_h={max_abs_all:.6}\tworst_slot={worst_slot}"
        );
        if min_cos < 0.999 || max_abs_all > 5e-2 {
            return Err(anyhow!(
                "GDN chain replay check failed: min_cos_h={min_cos:.9} max_abs_h={max_abs_all:.6}"
            ));
        }
    }

    let mut baseline_sessions = fresh_gdn_replay_sessions(&ctx, &mm, max_tokens)?;
    let mut replay_sessions = fresh_gdn_replay_sessions(&ctx, &mm, max_tokens)?;
    fill_gdn_replay_inputs(&ctx, &baseline_sessions)?;
    fill_gdn_replay_inputs(&ctx, &replay_sessions)?;
    let scratch = GdnLayerReplayScratch::new(&ctx, max_tokens, h, conv_dim, v_dim)?;

    println!(
        "mode\ttokens\tchain_layers\tencoders_per_batch\tdispatches_per_token\tavg_wall_ms\tavg_gpu_ms\tavg_gpu_ms_per_tok\tp50_gpu_ms_per_tok\tp90_gpu_ms_per_tok\tmax_gpu_ms_per_tok\tsaving_ms_per_tok\tsaving_pct"
    );
    for &n_tokens in &tokens {
        let baseline = time_cmd_reps_stats(&ctx, warmup, iters, |cmd| {
            let enc = KernelEncoder::begin(cmd);
            encode_gdn_chain_baseline(
                &ctx,
                &mf,
                &enc,
                selected,
                &mut baseline_sessions[..n_tokens],
            )?;
            enc.end();
            Ok(())
        })?;
        let replay = time_cmd_reps_stats(&ctx, warmup, iters, |cmd| {
            encode_gdn_chain_replay(
                &ctx,
                &mf,
                cmd,
                selected,
                &mut replay_sessions[..n_tokens],
                &scratch,
                h,
                conv_dim,
                v_dim,
            )
        })?;

        let baseline_per_tok = baseline.avg_gpu_ms / n_tokens as f64;
        let replay_per_tok = replay.avg_gpu_ms / n_tokens as f64;
        let save = baseline_per_tok - replay_per_tok;
        let pct = if baseline_per_tok > 0.0 {
            save / baseline_per_tok * 100.0
        } else {
            0.0
        };
        println!(
            "baseline_seq\t{}\t{}\t1.00\t{:.2}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t0.0000\t0.0",
            n_tokens,
            n_layers,
            15.0 * n_layers as f64,
            baseline.avg_wall_ms,
            baseline.avg_gpu_ms,
            baseline_per_tok,
            baseline.p50_gpu_ms / n_tokens as f64,
            baseline.p90_gpu_ms / n_tokens as f64,
            baseline.max_gpu_ms / n_tokens as f64,
        );
        println!(
            "replay_batched_qkv_z_out\t{}\t{}\t{}\t{:.2}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.1}",
            n_tokens,
            n_layers,
            3 * n_layers,
            (12.0 + 3.0 / n_tokens as f64) * n_layers as f64,
            replay.avg_wall_ms,
            replay.avg_gpu_ms,
            replay_per_tok,
            replay.p50_gpu_ms / n_tokens as f64,
            replay.p90_gpu_ms / n_tokens as f64,
            replay.max_gpu_ms / n_tokens as f64,
            save,
            pct
        );
    }

    Ok(())
}

fn encode_block_slice_baseline(
    mf: &MetalForward<'_>,
    enc: &KernelEncoder,
    start_block: usize,
    n_blocks: usize,
    position: u32,
    sessions: &mut [MetalSession],
) -> Result<()> {
    for block_i in start_block..start_block + n_blocks {
        for s in sessions.iter_mut() {
            mf.encode_moe_block_by_index(enc, block_i, position, s)?;
        }
    }
    Ok(())
}

fn encode_block_slice_replay(
    ctx: &MetalContext,
    mf: &MetalForward<'_>,
    mm: &MetalModel,
    cmd: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    start_block: usize,
    n_blocks: usize,
    position: u32,
    sessions: &mut [MetalSession],
    scratch: &GdnLayerReplayScratch,
    gdn_layers: &[SelectedGdnLayer<'_>],
    h: usize,
    conv_dim: usize,
    v_dim: usize,
) -> Result<()> {
    for block_i in start_block..start_block + n_blocks {
        encode_one_block_replay(
            ctx, mf, mm, cmd, block_i, position, sessions, scratch, gdn_layers, h, conv_dim, v_dim,
        )?;
    }
    Ok(())
}

fn encode_one_block_replay(
    ctx: &MetalContext,
    mf: &MetalForward<'_>,
    mm: &MetalModel,
    cmd: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    block_i: usize,
    position: u32,
    sessions: &mut [MetalSession],
    scratch: &GdnLayerReplayScratch,
    gdn_layers: &[SelectedGdnLayer<'_>],
    h: usize,
    conv_dim: usize,
    v_dim: usize,
) -> Result<()> {
    match &mm.blocks[block_i] {
        MetalBlock::Gdn(gb) => {
            let gdn_i = gdn_layers
                .iter()
                .find(|layer| layer.block_i == block_i)
                .map(|layer| layer.gdn_i)
                .ok_or_else(|| anyhow!("missing GDN index for block {block_i}"))?;
            encode_gdn_layer_replay(
                ctx, mf, cmd, gb, gdn_i, sessions, scratch, h, conv_dim, v_dim,
            )?;
            let enc = KernelEncoder::begin(cmd);
            for s in sessions.iter_mut() {
                mf.encode_moe_ffn_after_mixer_by_index(&enc, block_i, s)?;
            }
            enc.end();
        }
        MetalBlock::Attn(_) => {
            let enc = KernelEncoder::begin(cmd);
            for s in sessions.iter_mut() {
                mf.encode_moe_block_by_index(&enc, block_i, position, s)?;
            }
            enc.end();
        }
    }
    Ok(())
}

fn run_decode_block_slice_replay(args: DecodeBlockSliceReplayArgs) -> Result<()> {
    let DecodeBlockSliceReplayArgs {
        model,
        mut tokens,
        start_block,
        n_blocks,
        position,
        iters,
        warmup,
        no_check,
    } = args;
    if iters == 0 {
        return Err(anyhow!("--iters must be >= 1"));
    }
    if n_blocks == 0 {
        return Err(anyhow!("--blocks must be >= 1"));
    }
    if tokens.is_empty() || tokens.iter().any(|&n| n == 0) {
        return Err(anyhow!("--tokens entries must be >= 1"));
    }
    tokens.sort_unstable();
    tokens.dedup();
    let max_tokens = *tokens.last().expect("non-empty tokens");

    let ctx = MetalContext::new().context("init MetalContext")?;
    let g = GgufFile::open(&model).with_context(|| format!("open {}", model.display()))?;
    let m = Model::from_gguf(&g).context("parse model")?;
    let mm = MetalModel::load(&ctx, &g, &m).context("metal-load model")?;
    if mm.arch.kind != qwen_llm::model::ArchKind::Moe {
        return Err(anyhow!(
            "decode-block-slice-replay currently requires an MoE model"
        ));
    }
    let end_block = start_block
        .checked_add(n_blocks)
        .ok_or_else(|| anyhow!("start_block + blocks overflow"))?;
    if end_block > mm.blocks.len() {
        return Err(anyhow!(
            "block slice {}..{} outside available 0..{}",
            start_block,
            end_block,
            mm.blocks.len()
        ));
    }
    let mf = MetalForward::new(&ctx, &mm);
    let arch = &mm.arch;
    let h = arch.hidden_size as usize;
    let n_v = arch.gdn_n_v_heads as usize;
    let n_k = arch.gdn_n_k_heads as usize;
    let head_dim = arch.gdn_head_dim as usize;
    let conv_dim = (2 * n_k + n_v) * head_dim;
    let v_dim = n_v * head_dim;
    let gdn_layers = collect_gdn_layers(&mm);
    let n_gdn_in_slice = (start_block..end_block)
        .filter(|&i| matches!(mm.blocks[i], MetalBlock::Gdn(_)))
        .count();
    let n_attn_in_slice = n_blocks - n_gdn_in_slice;
    let kv_capacity = (position as usize)
        .checked_add(32)
        .ok_or_else(|| anyhow!("position + KV slack overflow"))?;

    println!(
        "[decode-block-slice-replay] model={} start_block={} blocks={} gdn_blocks={} attn_blocks={} position={} kv_capacity={} h={} conv_dim={} v_dim={} tokens={} warmup={} iters={}",
        model.display(),
        start_block,
        n_blocks,
        n_gdn_in_slice,
        n_attn_in_slice,
        position,
        kv_capacity,
        h,
        conv_dim,
        v_dim,
        tokens
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join(","),
        warmup,
        iters
    );

    if !no_check {
        let check_tokens = max_tokens;
        let mut base =
            fresh_gdn_replay_sessions_with_capacity(&ctx, &mm, check_tokens, kv_capacity)?;
        let mut replay =
            fresh_gdn_replay_sessions_with_capacity(&ctx, &mm, check_tokens, kv_capacity)?;
        fill_gdn_replay_inputs(&ctx, &base)?;
        fill_gdn_replay_inputs(&ctx, &replay)?;
        let scratch = GdnLayerReplayScratch::new(&ctx, check_tokens, h, conv_dim, v_dim)?;

        let cmd = ctx
            .queue
            .commandBuffer()
            .context("block slice check baseline cmd")?;
        let enc = KernelEncoder::begin(&cmd);
        encode_block_slice_baseline(&mf, &enc, start_block, n_blocks, position, &mut base)?;
        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();

        let cmd = ctx
            .queue
            .commandBuffer()
            .context("block slice check replay cmd")?;
        encode_block_slice_replay(
            &ctx,
            &mf,
            &mm,
            &cmd,
            start_block,
            n_blocks,
            position,
            &mut replay,
            &scratch,
            &gdn_layers,
            h,
            conv_dim,
            v_dim,
        )?;
        cmd.commit();
        cmd.waitUntilCompleted();

        let mut min_cos = 1.0f64;
        let mut max_abs_all = 0.0f32;
        let mut worst_slot = 0usize;
        for i in 0..check_tokens {
            let (cos, max_abs) =
                cosine_max_abs(&read_f32_tensor(&base[i].x), &read_f32_tensor(&replay[i].x));
            if cos < min_cos || max_abs > max_abs_all {
                worst_slot = i;
            }
            min_cos = min_cos.min(cos);
            max_abs_all = max_abs_all.max(max_abs);
        }
        println!(
            "check\ttokens={check_tokens}\tmin_cos_x={min_cos:.9}\tmax_abs_x={max_abs_all:.6}\tworst_slot={worst_slot}"
        );
        if min_cos < 0.999 || max_abs_all > 5e-2 {
            return Err(anyhow!(
                "block slice replay check failed: min_cos_x={min_cos:.9} max_abs_x={max_abs_all:.6}"
            ));
        }
    }

    let mut baseline_sessions =
        fresh_gdn_replay_sessions_with_capacity(&ctx, &mm, max_tokens, kv_capacity)?;
    let mut replay_sessions =
        fresh_gdn_replay_sessions_with_capacity(&ctx, &mm, max_tokens, kv_capacity)?;
    fill_gdn_replay_inputs(&ctx, &baseline_sessions)?;
    fill_gdn_replay_inputs(&ctx, &replay_sessions)?;
    let scratch = GdnLayerReplayScratch::new(&ctx, max_tokens, h, conv_dim, v_dim)?;

    println!(
        "mode\ttokens\tblocks\tgdn_blocks\tattn_blocks\tavg_wall_ms\tavg_gpu_ms\tavg_gpu_ms_per_tok\tp50_gpu_ms_per_tok\tp90_gpu_ms_per_tok\tmax_gpu_ms_per_tok\tsaving_ms_per_tok\tsaving_pct"
    );
    for &n_tokens in &tokens {
        let baseline = time_cmd_reps_stats(&ctx, warmup, iters, |cmd| {
            let enc = KernelEncoder::begin(cmd);
            encode_block_slice_baseline(
                &mf,
                &enc,
                start_block,
                n_blocks,
                position,
                &mut baseline_sessions[..n_tokens],
            )?;
            enc.end();
            Ok(())
        })?;
        let replay = time_cmd_reps_stats(&ctx, warmup, iters, |cmd| {
            encode_block_slice_replay(
                &ctx,
                &mf,
                &mm,
                cmd,
                start_block,
                n_blocks,
                position,
                &mut replay_sessions[..n_tokens],
                &scratch,
                &gdn_layers,
                h,
                conv_dim,
                v_dim,
            )
        })?;

        let baseline_per_tok = baseline.avg_gpu_ms / n_tokens as f64;
        let replay_per_tok = replay.avg_gpu_ms / n_tokens as f64;
        let save = baseline_per_tok - replay_per_tok;
        let pct = if baseline_per_tok > 0.0 {
            save / baseline_per_tok * 100.0
        } else {
            0.0
        };
        println!(
            "baseline_seq\t{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t0.0000\t0.0",
            n_tokens,
            n_blocks,
            n_gdn_in_slice,
            n_attn_in_slice,
            baseline.avg_wall_ms,
            baseline.avg_gpu_ms,
            baseline_per_tok,
            baseline.p50_gpu_ms / n_tokens as f64,
            baseline.p90_gpu_ms / n_tokens as f64,
            baseline.max_gpu_ms / n_tokens as f64,
        );
        println!(
            "replay_gdn_batched\t{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.1}",
            n_tokens,
            n_blocks,
            n_gdn_in_slice,
            n_attn_in_slice,
            replay.avg_wall_ms,
            replay.avg_gpu_ms,
            replay_per_tok,
            replay.p50_gpu_ms / n_tokens as f64,
            replay.p90_gpu_ms / n_tokens as f64,
            replay.max_gpu_ms / n_tokens as f64,
            save,
            pct
        );
    }

    Ok(())
}

fn run_decode_block_slice_trace(args: DecodeBlockSliceTraceArgs) -> Result<()> {
    let DecodeBlockSliceTraceArgs {
        model,
        tokens,
        start_block,
        n_blocks,
        position,
    } = args;
    if tokens == 0 {
        return Err(anyhow!("--tokens must be >= 1"));
    }
    if n_blocks == 0 {
        return Err(anyhow!("--blocks must be >= 1"));
    }

    let ctx = MetalContext::new().context("init MetalContext")?;
    let g = GgufFile::open(&model).with_context(|| format!("open {}", model.display()))?;
    let m = Model::from_gguf(&g).context("parse model")?;
    let mm = MetalModel::load(&ctx, &g, &m).context("metal-load model")?;
    if mm.arch.kind != qwen_llm::model::ArchKind::Moe {
        return Err(anyhow!(
            "decode-block-slice-trace currently requires an MoE model"
        ));
    }
    let end_block = start_block
        .checked_add(n_blocks)
        .ok_or_else(|| anyhow!("start_block + blocks overflow"))?;
    if end_block > mm.blocks.len() {
        return Err(anyhow!(
            "block slice {}..{} outside available 0..{}",
            start_block,
            end_block,
            mm.blocks.len()
        ));
    }

    let mf = MetalForward::new(&ctx, &mm);
    let arch = &mm.arch;
    let h = arch.hidden_size as usize;
    let n_v = arch.gdn_n_v_heads as usize;
    let n_k = arch.gdn_n_k_heads as usize;
    let head_dim = arch.gdn_head_dim as usize;
    let conv_dim = (2 * n_k + n_v) * head_dim;
    let v_dim = n_v * head_dim;
    let topk = arch.expert_used_count.min(arch.expert_count) as usize;
    let n_expert = arch.expert_count as usize;
    let kv_capacity = (position as usize)
        .checked_add(32)
        .ok_or_else(|| anyhow!("position + KV slack overflow"))?;
    let gdn_layers = collect_gdn_layers(&mm);

    println!(
        "[decode-block-slice-trace] model={} start_block={} blocks={} position={} kv_capacity={} tokens={} topk={} experts={} h={} conv_dim={} v_dim={}",
        model.display(),
        start_block,
        n_blocks,
        position,
        kv_capacity,
        tokens,
        topk,
        n_expert,
        h,
        conv_dim,
        v_dim
    );

    let mut base = fresh_gdn_replay_sessions_with_capacity(&ctx, &mm, tokens, kv_capacity)?;
    let mut replay = fresh_gdn_replay_sessions_with_capacity(&ctx, &mm, tokens, kv_capacity)?;
    fill_gdn_replay_inputs(&ctx, &base)?;
    fill_gdn_replay_inputs(&ctx, &replay)?;
    let scratch = GdnLayerReplayScratch::new(&ctx, tokens, h, conv_dim, v_dim)?;

    println!(
        "block\tkind\tslot\troute_order_equal\troute_set_equal\tbase_idx\treplay_idx\tweight_max_abs\tshared_abs\tbase_margin\treplay_margin\tlogit_max_abs\tlogit_rms\th_cos\th_max_abs\tx_cos\tx_max_abs"
    );

    let mut route_order_mismatches = 0usize;
    let mut route_set_mismatches = 0usize;
    let mut first_order_mismatch: Option<(usize, usize)> = None;
    let mut first_set_mismatch: Option<(usize, usize)> = None;
    let mut min_h_cos = 1.0f64;
    let mut min_x_cos = 1.0f64;
    let mut max_h_abs = 0.0f32;
    let mut max_x_abs = 0.0f32;
    let mut max_logit_abs = 0.0f32;
    let mut max_logit_rms = 0.0f64;
    let mut min_base_margin = f32::INFINITY;
    let mut min_replay_margin = f32::INFINITY;
    let mut replay_margin_lt_1e3 = 0usize;
    let mut replay_margin_lt_5e3 = 0usize;

    for block_i in start_block..end_block {
        let kind = match &mm.blocks[block_i] {
            MetalBlock::Gdn(_) => "gdn",
            MetalBlock::Attn(_) => "attn",
        };

        let cmd = ctx
            .queue
            .commandBuffer()
            .context("block trace baseline cmd")?;
        let enc = KernelEncoder::begin(&cmd);
        encode_block_slice_baseline(&mf, &enc, block_i, 1, position, &mut base)?;
        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();

        let cmd = ctx
            .queue
            .commandBuffer()
            .context("block trace replay cmd")?;
        encode_one_block_replay(
            &ctx,
            &mf,
            &mm,
            &cmd,
            block_i,
            position,
            &mut replay,
            &scratch,
            &gdn_layers,
            h,
            conv_dim,
            v_dim,
        )?;
        cmd.commit();
        cmd.waitUntilCompleted();

        for slot in 0..tokens {
            let base_route = read_route_fingerprint(&base[slot], topk, n_expert);
            let replay_route = read_route_fingerprint(&replay[slot], topk, n_expert);
            let route_order_equal = base_route.idx == replay_route.idx;
            let route_set_equal = same_i32_set(&base_route.idx, &replay_route.idx);
            if !route_order_equal {
                route_order_mismatches += 1;
                first_order_mismatch.get_or_insert((block_i, slot));
            }
            if !route_set_equal {
                route_set_mismatches += 1;
                first_set_mismatch.get_or_insert((block_i, slot));
            }
            let weight_max_abs = route_weight_max_abs(&base_route.weight, &replay_route.weight);
            let shared_abs = (base_route.shared_gate - replay_route.shared_gate).abs();
            let logit_max_abs = f32_max_abs_delta(&base_route.logits, &replay_route.logits);
            let logit_rms = f32_rms_delta(&base_route.logits, &replay_route.logits);
            max_logit_abs = max_logit_abs.max(logit_max_abs);
            max_logit_rms = max_logit_rms.max(logit_rms);
            min_base_margin = min_base_margin.min(base_route.logit_margin);
            min_replay_margin = min_replay_margin.min(replay_route.logit_margin);
            if replay_route.logit_margin < 0.001 {
                replay_margin_lt_1e3 += 1;
            }
            if replay_route.logit_margin < 0.005 {
                replay_margin_lt_5e3 += 1;
            }
            let (h_cos, h_abs) = cosine_max_abs(
                &read_f32_tensor(&base[slot].h),
                &read_f32_tensor(&replay[slot].h),
            );
            let (x_cos, x_abs) = cosine_max_abs(
                &read_f32_tensor(&base[slot].x),
                &read_f32_tensor(&replay[slot].x),
            );
            min_h_cos = min_h_cos.min(h_cos);
            min_x_cos = min_x_cos.min(x_cos);
            max_h_abs = max_h_abs.max(h_abs);
            max_x_abs = max_x_abs.max(x_abs);
            println!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.9}\t{:.6}\t{:.9}\t{:.6}",
                block_i,
                kind,
                slot,
                route_order_equal,
                route_set_equal,
                fmt_i32_csv(&base_route.idx),
                fmt_i32_csv(&replay_route.idx),
                weight_max_abs,
                shared_abs,
                base_route.logit_margin,
                replay_route.logit_margin,
                logit_max_abs,
                logit_rms,
                h_cos,
                h_abs,
                x_cos,
                x_abs
            );
        }
    }

    let first_order = first_order_mismatch
        .map(|(block, slot)| format!("block={block},slot={slot}"))
        .unwrap_or_else(|| "none".to_string());
    let first_set = first_set_mismatch
        .map(|(block, slot)| format!("block={block},slot={slot}"))
        .unwrap_or_else(|| "none".to_string());
    println!(
        "summary\troute_order_mismatches={}\troute_set_mismatches={}\tfirst_order_mismatch={}\tfirst_set_mismatch={}\tmax_logit_abs={:.6}\tmax_logit_rms={:.6}\tmin_base_margin={:.6}\tmin_replay_margin={:.6}\treplay_margin_lt_1e3={}\treplay_margin_lt_5e3={}\tmin_h_cos={:.9}\tmax_h_abs={:.6}\tmin_x_cos={:.9}\tmax_x_abs={:.6}",
        route_order_mismatches,
        route_set_mismatches,
        first_order,
        first_set,
        max_logit_abs,
        max_logit_rms,
        min_base_margin,
        min_replay_margin,
        replay_margin_lt_1e3,
        replay_margin_lt_5e3,
        min_h_cos,
        max_h_abs,
        min_x_cos,
        max_x_abs
    );

    Ok(())
}

struct BlockSliceTraceSummary {
    route_order_mismatches: usize,
    route_set_mismatches: usize,
    first_set_mismatch: Option<(usize, usize)>,
    max_logit_abs: f32,
    max_logit_rms: f64,
    min_base_margin: f32,
    min_replay_margin: f32,
    replay_margin_lt_1e3: usize,
    replay_margin_lt_5e3: usize,
    min_x_cos: f64,
    max_x_abs: f32,
}

impl Default for BlockSliceTraceSummary {
    fn default() -> Self {
        Self {
            route_order_mismatches: 0,
            route_set_mismatches: 0,
            first_set_mismatch: None,
            max_logit_abs: 0.0,
            max_logit_rms: 0.0,
            min_base_margin: f32::INFINITY,
            min_replay_margin: f32::INFINITY,
            replay_margin_lt_1e3: 0,
            replay_margin_lt_5e3: 0,
            min_x_cos: 1.0,
            max_x_abs: 0.0,
        }
    }
}

fn trace_block_slice_summary(
    ctx: &MetalContext,
    mf: &MetalForward<'_>,
    mm: &MetalModel,
    start_block: usize,
    n_blocks: usize,
    position: u32,
    tokens: usize,
    h: usize,
    conv_dim: usize,
    v_dim: usize,
    topk: usize,
    n_expert: usize,
    gdn_layers: &[SelectedGdnLayer<'_>],
) -> Result<BlockSliceTraceSummary> {
    let kv_capacity = (position as usize)
        .checked_add(32)
        .ok_or_else(|| anyhow!("position + KV slack overflow"))?;
    let mut base = fresh_gdn_replay_sessions_with_capacity(ctx, mm, tokens, kv_capacity)?;
    let mut replay = fresh_gdn_replay_sessions_with_capacity(ctx, mm, tokens, kv_capacity)?;
    fill_gdn_replay_inputs(ctx, &base)?;
    fill_gdn_replay_inputs(ctx, &replay)?;
    let scratch = GdnLayerReplayScratch::new(ctx, tokens, h, conv_dim, v_dim)?;
    let mut out = BlockSliceTraceSummary::default();

    for block_i in start_block..start_block + n_blocks {
        let cmd = ctx
            .queue
            .commandBuffer()
            .context("block margin sweep baseline cmd")?;
        let enc = KernelEncoder::begin(&cmd);
        encode_block_slice_baseline(mf, &enc, block_i, 1, position, &mut base)?;
        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();

        let cmd = ctx
            .queue
            .commandBuffer()
            .context("block margin sweep replay cmd")?;
        encode_one_block_replay(
            ctx,
            mf,
            mm,
            &cmd,
            block_i,
            position,
            &mut replay,
            &scratch,
            gdn_layers,
            h,
            conv_dim,
            v_dim,
        )?;
        cmd.commit();
        cmd.waitUntilCompleted();

        for slot in 0..tokens {
            let base_route = read_route_fingerprint(&base[slot], topk, n_expert);
            let replay_route = read_route_fingerprint(&replay[slot], topk, n_expert);
            let route_order_equal = base_route.idx == replay_route.idx;
            let route_set_equal = same_i32_set(&base_route.idx, &replay_route.idx);
            if !route_order_equal {
                out.route_order_mismatches += 1;
            }
            if !route_set_equal {
                out.route_set_mismatches += 1;
                out.first_set_mismatch.get_or_insert((block_i, slot));
            }
            let logit_max_abs = f32_max_abs_delta(&base_route.logits, &replay_route.logits);
            let logit_rms = f32_rms_delta(&base_route.logits, &replay_route.logits);
            out.max_logit_abs = out.max_logit_abs.max(logit_max_abs);
            out.max_logit_rms = out.max_logit_rms.max(logit_rms);
            out.min_base_margin = out.min_base_margin.min(base_route.logit_margin);
            out.min_replay_margin = out.min_replay_margin.min(replay_route.logit_margin);
            if replay_route.logit_margin < 0.001 {
                out.replay_margin_lt_1e3 += 1;
            }
            if replay_route.logit_margin < 0.005 {
                out.replay_margin_lt_5e3 += 1;
            }
            let (x_cos, x_abs) = cosine_max_abs(
                &read_f32_tensor(&base[slot].x),
                &read_f32_tensor(&replay[slot].x),
            );
            out.min_x_cos = out.min_x_cos.min(x_cos);
            out.max_x_abs = out.max_x_abs.max(x_abs);
        }
    }

    Ok(out)
}

fn copy_recurrent_session_state(
    ctx: &MetalContext,
    src: &MetalSession,
    dst: &mut MetalSession,
) -> Result<()> {
    if src.gdn_conv.len() != dst.gdn_conv.len()
        || src.gdn_state.len() != dst.gdn_state.len()
        || src.kv_k.len() != dst.kv_k.len()
        || src.kv_v.len() != dst.kv_v.len()
    {
        return Err(anyhow!("session state vector lengths differ"));
    }
    let cmd = ctx
        .queue
        .commandBuffer()
        .context("copy session state cmd")?;
    let blit = BlitEncoder::begin(&cmd);
    for (a, b) in src.gdn_conv.iter().zip(dst.gdn_conv.iter()) {
        blit.copy_buffer(&a.buffer, a.offset, &b.buffer, b.offset, a.n_bytes());
    }
    for (a, b) in src.gdn_state.iter().zip(dst.gdn_state.iter()) {
        blit.copy_buffer(&a.buffer, a.offset, &b.buffer, b.offset, a.n_bytes());
    }
    for (a, b) in src.kv_k.iter().zip(dst.kv_k.iter()) {
        blit.copy_buffer(&a.buffer, a.offset, &b.buffer, b.offset, a.n_bytes());
    }
    for (a, b) in src.kv_v.iter().zip(dst.kv_v.iter()) {
        blit.copy_buffer(&a.buffer, a.offset, &b.buffer, b.offset, a.n_bytes());
    }
    blit.end();
    cmd.commit();
    cmd.waitUntilCompleted();
    dst.kv_n_pos.clone_from(&src.kv_n_pos);
    Ok(())
}

fn reset_block_slice_sessions(
    ctx: &MetalContext,
    src: &[MetalSession],
    dst: &mut [MetalSession],
) -> Result<()> {
    if src.len() != dst.len() {
        return Err(anyhow!(
            "session reset length mismatch: {} source versus {} destination",
            src.len(),
            dst.len()
        ));
    }

    for (a, b) in src.iter().zip(dst.iter()) {
        if a.gdn_conv.len() != b.gdn_conv.len()
            || a.gdn_state.len() != b.gdn_state.len()
            || a.kv_k.len() != b.kv_k.len()
            || a.kv_v.len() != b.kv_v.len()
            || a.x.n_bytes() != b.x.n_bytes()
        {
            return Err(anyhow!("block-slice session reset shape mismatch"));
        }
    }

    let cmd = ctx
        .queue
        .commandBuffer()
        .context("reset block-slice sessions cmd")?;
    let blit = BlitEncoder::begin(&cmd);
    for (a, b) in src.iter().zip(dst.iter()) {
        blit.copy_buffer(
            &a.x.buffer,
            a.x.offset,
            &b.x.buffer,
            b.x.offset,
            a.x.n_bytes(),
        );
        for (src_t, dst_t) in a.gdn_conv.iter().zip(b.gdn_conv.iter()) {
            blit.copy_buffer(
                &src_t.buffer,
                src_t.offset,
                &dst_t.buffer,
                dst_t.offset,
                src_t.n_bytes(),
            );
        }
        for (src_t, dst_t) in a.gdn_state.iter().zip(b.gdn_state.iter()) {
            blit.copy_buffer(
                &src_t.buffer,
                src_t.offset,
                &dst_t.buffer,
                dst_t.offset,
                src_t.n_bytes(),
            );
        }
        for (src_t, dst_t) in a.kv_k.iter().zip(b.kv_k.iter()) {
            blit.copy_buffer(
                &src_t.buffer,
                src_t.offset,
                &dst_t.buffer,
                dst_t.offset,
                src_t.n_bytes(),
            );
        }
        for (src_t, dst_t) in a.kv_v.iter().zip(b.kv_v.iter()) {
            blit.copy_buffer(
                &src_t.buffer,
                src_t.offset,
                &dst_t.buffer,
                dst_t.offset,
                src_t.n_bytes(),
            );
        }
    }
    blit.end();
    cmd.commit();
    cmd.waitUntilCompleted();
    for (a, b) in src.iter().zip(dst.iter_mut()) {
        b.kv_n_pos.clone_from(&a.kv_n_pos);
    }
    Ok(())
}

fn prepare_moe_session_to_block(
    ctx: &MetalContext,
    mf: &MetalForward<'_>,
    mm: &MetalModel,
    token_id: i32,
    position: u32,
    start_block: usize,
    session: &mut MetalSession,
    h: usize,
) -> Result<()> {
    unsafe {
        let ptr = session.ids_buf.buffer.contents().as_ptr() as *mut i32;
        *ptr = token_id;
    }
    let cmd = ctx
        .queue
        .commandBuffer()
        .context("prepare block-slice session cmd")?;
    let enc = KernelEncoder::begin(&cmd);
    encode_get_rows_f32(
        ctx,
        &enc,
        &mm.token_embd,
        &session.ids_buf,
        &session.x,
        1,
        h,
    )?;
    for block_i in 0..start_block {
        mf.encode_moe_block_by_index(&enc, block_i, position, session)?;
    }
    enc.end();
    cmd.commit();
    cmd.waitUntilCompleted();
    Ok(())
}

fn prepare_real_block_slice_sessions(
    ctx: &MetalContext,
    mf: &MetalForward<'_>,
    mm: &MetalModel,
    prefix_sessions: &[MetalSession],
    prompt_ids: &[Vec<i32>],
    slot_positions: &[usize],
    tokens: usize,
    start_block: usize,
    kv_capacity: usize,
    h: usize,
) -> Result<Vec<MetalSession>> {
    if slot_positions.len() < tokens {
        return Err(anyhow!(
            "slot_positions has {} entries, need {tokens}",
            slot_positions.len()
        ));
    }
    let mut sessions = fresh_gdn_replay_sessions_with_capacity(ctx, mm, tokens, kv_capacity)?;
    for slot in 0..tokens {
        copy_recurrent_session_state(ctx, &prefix_sessions[slot], &mut sessions[slot])?;
        let ids = if prompt_ids.len() == 1 {
            &prompt_ids[0]
        } else {
            &prompt_ids[slot]
        };
        let pos = slot_positions[slot];
        let token_id = ids[pos];
        prepare_moe_session_to_block(
            ctx,
            mf,
            mm,
            token_id,
            pos as u32,
            start_block,
            &mut sessions[slot],
            h,
        )?;
    }
    Ok(sessions)
}

struct ValidatedReplayStats {
    avg_wall_ms: f64,
    avg_gpu_ms: f64,
    p50_wall_ms: f64,
    p50_gpu_ms: f64,
    avg_fallback_slots: f64,
}

fn time_validated_block_slice_replay(
    ctx: &MetalContext,
    mf: &MetalForward<'_>,
    mm: &MetalModel,
    start_block: usize,
    n_blocks: usize,
    position: u32,
    seed_sessions: &[MetalSession],
    sessions: &mut [MetalSession],
    scratch: &GdnLayerReplayScratch,
    gdn_layers: &[SelectedGdnLayer<'_>],
    h: usize,
    conv_dim: usize,
    v_dim: usize,
    topk: usize,
    n_expert: usize,
    margin_threshold: f32,
    warmup: usize,
    iters: usize,
) -> Result<ValidatedReplayStats> {
    let mut wall_samples = Vec::with_capacity(iters);
    let mut gpu_samples = Vec::with_capacity(iters);
    let mut fallback_samples = Vec::with_capacity(iters);

    for rep in 0..warmup + iters {
        let timed = rep >= warmup;
        reset_block_slice_sessions(ctx, seed_sessions, sessions)?;
        let start = Instant::now();
        let mut gpu_ms = 0.0f64;
        let mut fallback_slots = vec![false; sessions.len()];

        for block_i in start_block..start_block + n_blocks {
            let cmd = ctx
                .queue
                .commandBuffer()
                .context("validated replay block cmd")?;
            encode_one_block_replay(
                ctx, mf, mm, &cmd, block_i, position, sessions, scratch, gdn_layers, h, conv_dim,
                v_dim,
            )?;
            cmd.commit();
            cmd.waitUntilCompleted();
            gpu_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;

            for (slot, session) in sessions.iter().enumerate() {
                let route = read_route_fingerprint(session, topk, n_expert);
                if route.logit_margin < margin_threshold {
                    fallback_slots[slot] = true;
                }
            }
        }

        if timed {
            wall_samples.push(start.elapsed().as_secs_f64() * 1e3);
            gpu_samples.push(gpu_ms);
            fallback_samples.push(fallback_slots.iter().filter(|&&v| v).count() as f64);
        }
    }

    let avg_wall_ms = wall_samples.iter().sum::<f64>() / iters as f64;
    let avg_gpu_ms = gpu_samples.iter().sum::<f64>() / iters as f64;
    wall_samples.sort_by(|a, b| a.total_cmp(b));
    gpu_samples.sort_by(|a, b| a.total_cmp(b));
    Ok(ValidatedReplayStats {
        avg_wall_ms,
        avg_gpu_ms,
        p50_wall_ms: percentile(&wall_samples, 0.50),
        p50_gpu_ms: percentile(&gpu_samples, 0.50),
        avg_fallback_slots: fallback_samples.iter().sum::<f64>() / iters as f64,
    })
}

fn trace_prepared_block_slice_summary(
    ctx: &MetalContext,
    mf: &MetalForward<'_>,
    mm: &MetalModel,
    start_block: usize,
    n_blocks: usize,
    position: u32,
    base: &mut [MetalSession],
    replay: &mut [MetalSession],
    scratch: &GdnLayerReplayScratch,
    h: usize,
    conv_dim: usize,
    v_dim: usize,
    topk: usize,
    n_expert: usize,
    gdn_layers: &[SelectedGdnLayer<'_>],
) -> Result<BlockSliceTraceSummary> {
    if base.len() != replay.len() {
        return Err(anyhow!("base/replay slot counts differ"));
    }
    let mut out = BlockSliceTraceSummary::default();
    for block_i in start_block..start_block + n_blocks {
        let cmd = ctx
            .queue
            .commandBuffer()
            .context("real margin baseline cmd")?;
        let enc = KernelEncoder::begin(&cmd);
        encode_block_slice_baseline(mf, &enc, block_i, 1, position, base)?;
        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();

        let cmd = ctx
            .queue
            .commandBuffer()
            .context("real margin replay cmd")?;
        encode_one_block_replay(
            ctx, mf, mm, &cmd, block_i, position, replay, scratch, gdn_layers, h, conv_dim, v_dim,
        )?;
        cmd.commit();
        cmd.waitUntilCompleted();

        for slot in 0..base.len() {
            let base_route = read_route_fingerprint(&base[slot], topk, n_expert);
            let replay_route = read_route_fingerprint(&replay[slot], topk, n_expert);
            let route_order_equal = base_route.idx == replay_route.idx;
            let route_set_equal = same_i32_set(&base_route.idx, &replay_route.idx);
            if !route_order_equal {
                out.route_order_mismatches += 1;
            }
            if !route_set_equal {
                out.route_set_mismatches += 1;
                out.first_set_mismatch.get_or_insert((block_i, slot));
            }
            let logit_max_abs = f32_max_abs_delta(&base_route.logits, &replay_route.logits);
            let logit_rms = f32_rms_delta(&base_route.logits, &replay_route.logits);
            out.max_logit_abs = out.max_logit_abs.max(logit_max_abs);
            out.max_logit_rms = out.max_logit_rms.max(logit_rms);
            out.min_base_margin = out.min_base_margin.min(base_route.logit_margin);
            out.min_replay_margin = out.min_replay_margin.min(replay_route.logit_margin);
            if replay_route.logit_margin < 0.001 {
                out.replay_margin_lt_1e3 += 1;
            }
            if replay_route.logit_margin < 0.005 {
                out.replay_margin_lt_5e3 += 1;
            }
            let (x_cos, x_abs) = cosine_max_abs(
                &read_f32_tensor(&base[slot].x),
                &read_f32_tensor(&replay[slot].x),
            );
            out.min_x_cos = out.min_x_cos.min(x_cos);
            out.max_x_abs = out.max_x_abs.max(x_abs);
        }
    }
    Ok(out)
}

fn run_decode_block_slice_margin_sweep(args: DecodeBlockSliceMarginSweepArgs) -> Result<()> {
    let DecodeBlockSliceMarginSweepArgs {
        model,
        tokens,
        mut start_blocks,
        n_blocks,
        mut positions,
    } = args;
    if tokens == 0 {
        return Err(anyhow!("--tokens must be >= 1"));
    }
    if n_blocks == 0 {
        return Err(anyhow!("--blocks must be >= 1"));
    }
    if positions.is_empty() {
        return Err(anyhow!("--position must include at least one entry"));
    }

    let ctx = MetalContext::new().context("init MetalContext")?;
    let g = GgufFile::open(&model).with_context(|| format!("open {}", model.display()))?;
    let m = Model::from_gguf(&g).context("parse model")?;
    let mm = MetalModel::load(&ctx, &g, &m).context("metal-load model")?;
    if mm.arch.kind != qwen_llm::model::ArchKind::Moe {
        return Err(anyhow!(
            "decode-block-slice-margin-sweep currently requires an MoE model"
        ));
    }
    if start_blocks.is_empty() {
        start_blocks = (0..mm.blocks.len())
            .step_by(n_blocks)
            .filter(|&i| i + n_blocks <= mm.blocks.len())
            .collect();
    }
    start_blocks.sort_unstable();
    start_blocks.dedup();
    positions.sort_unstable();
    positions.dedup();
    for &start_block in &start_blocks {
        let end_block = start_block
            .checked_add(n_blocks)
            .ok_or_else(|| anyhow!("start_block + blocks overflow"))?;
        if end_block > mm.blocks.len() {
            return Err(anyhow!(
                "block slice {}..{} outside available 0..{}",
                start_block,
                end_block,
                mm.blocks.len()
            ));
        }
    }

    let mf = MetalForward::new(&ctx, &mm);
    let arch = &mm.arch;
    let h = arch.hidden_size as usize;
    let n_v = arch.gdn_n_v_heads as usize;
    let n_k = arch.gdn_n_k_heads as usize;
    let head_dim = arch.gdn_head_dim as usize;
    let conv_dim = (2 * n_k + n_v) * head_dim;
    let v_dim = n_v * head_dim;
    let topk = arch.expert_used_count.min(arch.expert_count) as usize;
    let n_expert = arch.expert_count as usize;
    let gdn_layers = collect_gdn_layers(&mm);

    println!(
        "[decode-block-slice-margin-sweep] model={} windows={} positions={} tokens={} blocks={} topk={} experts={} h={} conv_dim={} v_dim={}",
        model.display(),
        start_blocks.len(),
        positions
            .iter()
            .map(|p| p.to_string())
            .collect::<Vec<_>>()
            .join(","),
        tokens,
        n_blocks,
        topk,
        n_expert,
        h,
        conv_dim,
        v_dim
    );
    println!(
        "start_block\tend_block\tposition\ttokens\troute_order_mismatches\troute_set_mismatches\tfirst_set_mismatch\tmax_logit_abs\tmax_logit_rms\tmin_base_margin\tmin_replay_margin\treplay_margin_lt_1e3\treplay_margin_lt_5e3\tmin_x_cos\tmax_x_abs"
    );

    for &position in &positions {
        for &start_block in &start_blocks {
            let summary = trace_block_slice_summary(
                &ctx,
                &mf,
                &mm,
                start_block,
                n_blocks,
                position,
                tokens,
                h,
                conv_dim,
                v_dim,
                topk,
                n_expert,
                &gdn_layers,
            )?;
            let first_set = summary
                .first_set_mismatch
                .map(|(block, slot)| format!("block={block},slot={slot}"))
                .unwrap_or_else(|| "none".to_string());
            println!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{}\t{}\t{:.9}\t{:.6}",
                start_block,
                start_block + n_blocks,
                position,
                tokens,
                summary.route_order_mismatches,
                summary.route_set_mismatches,
                first_set,
                summary.max_logit_abs,
                summary.max_logit_rms,
                summary.min_base_margin,
                summary.min_replay_margin,
                summary.replay_margin_lt_1e3,
                summary.replay_margin_lt_5e3,
                summary.min_x_cos,
                summary.max_x_abs,
            );
        }
    }

    Ok(())
}

fn run_decode_block_slice_real_margin(args: DecodeBlockSliceRealMarginArgs) -> Result<()> {
    let DecodeBlockSliceRealMarginArgs {
        model,
        file,
        tokens,
        mut slot_counts,
        mut contexts,
        stride,
        mut start_blocks,
        n_blocks,
        timing_iters,
        timing_warmup,
        margin_threshold,
    } = args;
    if tokens == 0 {
        return Err(anyhow!("--tokens must be >= 1"));
    }
    if slot_counts.is_empty() {
        slot_counts.push(tokens);
    }
    slot_counts.sort_unstable();
    slot_counts.dedup();
    if slot_counts.iter().any(|&count| count == 0) {
        return Err(anyhow!("--slot-counts must all be >= 1"));
    }
    let max_slots = *slot_counts.iter().max().expect("non-empty slot_counts");
    if max_slots > tokens {
        return Err(anyhow!(
            "largest --slot-counts value ({max_slots}) exceeds --tokens {tokens}; set --tokens to the maximum slots to prepare"
        ));
    }
    if stride == 0 {
        return Err(anyhow!("--stride must be >= 1"));
    }
    if file.is_empty() {
        return Err(anyhow!("--file must include at least one prompt"));
    }
    if file.len() > 1 && file.len() < max_slots {
        return Err(anyhow!(
            "got {} --file entries, need at least {max_slots} for the requested slot counts",
            file.len(),
        ));
    }
    if n_blocks == 0 {
        return Err(anyhow!("--blocks must be >= 1"));
    }
    if timing_iters == 0 && timing_warmup != 1 {
        return Err(anyhow!(
            "--timing-warmup is only meaningful when --timing-iters > 0"
        ));
    }
    if !(margin_threshold.is_finite() && margin_threshold >= 0.0) {
        return Err(anyhow!(
            "--margin-threshold must be a finite non-negative value"
        ));
    }
    if contexts.is_empty() {
        return Err(anyhow!("--context must include at least one entry"));
    }

    let ctx = MetalContext::new().context("init MetalContext")?;
    let g = GgufFile::open(&model).with_context(|| format!("open {}", model.display()))?;
    let m = Model::from_gguf(&g).context("parse model")?;
    let mm = MetalModel::load(&ctx, &g, &m).context("metal-load model")?;
    if mm.arch.kind != qwen_llm::model::ArchKind::Moe {
        return Err(anyhow!(
            "decode-block-slice-real-margin currently requires an MoE model"
        ));
    }
    if start_blocks.is_empty() {
        start_blocks = (0..mm.blocks.len())
            .step_by(n_blocks)
            .filter(|&i| i + n_blocks <= mm.blocks.len())
            .collect();
    }
    start_blocks.sort_unstable();
    start_blocks.dedup();
    contexts.sort_unstable();
    contexts.dedup();
    for &start_block in &start_blocks {
        let end_block = start_block
            .checked_add(n_blocks)
            .ok_or_else(|| anyhow!("start_block + blocks overflow"))?;
        if end_block > mm.blocks.len() {
            return Err(anyhow!(
                "block slice {}..{} outside available 0..{}",
                start_block,
                end_block,
                mm.blocks.len()
            ));
        }
    }

    let tok = NativeTokenizer::from_gguf(&g).context("open native GGUF tokenizer")?;
    let mut prompt_ids = Vec::with_capacity(file.len());
    for path in &file {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let ids = tok
            .encode(&text, false)
            .with_context(|| format!("tokenize {}", path.display()))?;
        prompt_ids.push(ids);
    }
    let max_context = *contexts.last().expect("non-empty contexts");
    let last_needed = if file.len() == 1 {
        max_context
            .checked_add((max_slots - 1).saturating_mul(stride))
            .ok_or_else(|| anyhow!("context + (tokens - 1) * stride overflow"))?
    } else {
        max_context
    };
    if file.len() == 1 {
        if prompt_ids[0].len() <= last_needed {
            return Err(anyhow!(
                "prompt {} has {} tokens, need at least {} for context+stride sweep",
                file[0].display(),
                prompt_ids[0].len(),
                last_needed + 1
            ));
        }
    } else {
        for (path, ids) in file.iter().zip(prompt_ids.iter()).take(max_slots) {
            if ids.len() <= last_needed {
                return Err(anyhow!(
                    "prompt {} has {} tokens, need at least {} for context sweep",
                    path.display(),
                    ids.len(),
                    last_needed + 1
                ));
            }
        }
    }

    let mf = MetalForward::new(&ctx, &mm);
    let arch = &mm.arch;
    let h = arch.hidden_size as usize;
    let n_v = arch.gdn_n_v_heads as usize;
    let n_k = arch.gdn_n_k_heads as usize;
    let head_dim = arch.gdn_head_dim as usize;
    let conv_dim = (2 * n_k + n_v) * head_dim;
    let v_dim = n_v * head_dim;
    let topk = arch.expert_used_count.min(arch.expert_count) as usize;
    let n_expert = arch.expert_count as usize;
    let gdn_layers = collect_gdn_layers(&mm);

    println!(
        "[decode-block-slice-real-margin] model={} files={} first_prompt_tokens={} contexts={} windows={} tokens={} slot_counts={} stride={} blocks={} topk={} experts={} h={} conv_dim={} v_dim={} timing_iters={} timing_warmup={} margin_threshold={:.6}",
        model.display(),
        file.len(),
        prompt_ids[0].len(),
        contexts
            .iter()
            .map(|p| p.to_string())
            .collect::<Vec<_>>()
            .join(","),
        start_blocks.len(),
        tokens,
        slot_counts
            .iter()
            .map(|p| p.to_string())
            .collect::<Vec<_>>()
            .join(","),
        stride,
        n_blocks,
        topk,
        n_expert,
        h,
        conv_dim,
        v_dim,
        timing_iters,
        timing_warmup,
        margin_threshold,
    );
    println!(
        "start_block\tend_block\tcontext\tslots\troute_order_mismatches\troute_set_mismatches\tfirst_set_mismatch\tmax_logit_abs\tmax_logit_rms\tmin_base_margin\tmin_replay_margin\treplay_margin_lt_1e3\treplay_margin_lt_5e3\tmin_x_cos\tmax_x_abs\tbaseline_wall_ms_per_tok\treplay_wall_ms_per_tok\tgross_wall_save_pct\tvalidated_wall_ms_per_tok\tvalidated_gpu_ms_per_tok\tfallback_slots_avg\tfallback_pct\tnet_wall_save_pct\tbaseline_p50_wall_ms_per_tok\treplay_p50_wall_ms_per_tok\tvalidated_p50_wall_ms_per_tok\tvalidated_p50_gpu_ms_per_tok\tp50_net_wall_save_pct"
    );

    for &context in &contexts {
        let kv_capacity = last_needed + 32;
        let slot_positions: Vec<usize> = (0..max_slots)
            .map(|slot| {
                if prompt_ids.len() == 1 {
                    context + slot * stride
                } else {
                    context
                }
            })
            .collect();
        let mut prefix_sessions = Vec::with_capacity(max_slots);
        if prompt_ids.len() == 1 {
            let ids = &prompt_ids[0];
            let mut prev_pos = slot_positions[0];
            let mut first = MetalSession::fresh(&ctx, &mm, kv_capacity)
                .context("real prefix session slot 0")?;
            for (p, &token_id) in ids.iter().take(prev_pos).enumerate() {
                let _ = mf.single_token_argmax_profiled(token_id, p as u32, &mut first)?;
            }
            prefix_sessions.push(first);
            for slot in 1..max_slots {
                let pos = slot_positions[slot];
                let mut s = MetalSession::fresh(&ctx, &mm, kv_capacity)
                    .with_context(|| format!("real prefix session slot {slot}"))?;
                copy_recurrent_session_state(&ctx, &prefix_sessions[slot - 1], &mut s)?;
                for p in prev_pos..pos {
                    let _ = mf.single_token_argmax_profiled(ids[p], p as u32, &mut s)?;
                }
                prefix_sessions.push(s);
                prev_pos = pos;
            }
        } else {
            for slot in 0..max_slots {
                let ids = &prompt_ids[slot];
                let pos = slot_positions[slot];
                let mut s = MetalSession::fresh(&ctx, &mm, kv_capacity)
                    .with_context(|| format!("real prefix session slot {slot}"))?;
                for (p, &token_id) in ids.iter().take(pos).enumerate() {
                    let _ = mf.single_token_argmax_profiled(token_id, p as u32, &mut s)?;
                }
                prefix_sessions.push(s);
            }
        }

        for &start_block in &start_blocks {
            let slice_has_attention = block_slice_has_attention(&mm, start_block, n_blocks)?;
            for &slot_count in &slot_counts {
                if prompt_ids.len() == 1 && slot_count > 1 && slice_has_attention {
                    return Err(anyhow!(
                        "single-file strided real-margin slots currently require GDN-only block slices; slice {}..{} contains attention",
                        start_block,
                        start_block + n_blocks
                    ));
                }
                let mut base = prepare_real_block_slice_sessions(
                    &ctx,
                    &mf,
                    &mm,
                    &prefix_sessions,
                    &prompt_ids,
                    &slot_positions,
                    slot_count,
                    start_block,
                    kv_capacity,
                    h,
                )?;
                let mut replay = prepare_real_block_slice_sessions(
                    &ctx,
                    &mf,
                    &mm,
                    &prefix_sessions,
                    &prompt_ids,
                    &slot_positions,
                    slot_count,
                    start_block,
                    kv_capacity,
                    h,
                )?;

                let scratch = GdnLayerReplayScratch::new(&ctx, slot_count, h, conv_dim, v_dim)?;
                let summary = trace_prepared_block_slice_summary(
                    &ctx,
                    &mf,
                    &mm,
                    start_block,
                    n_blocks,
                    context as u32,
                    &mut base,
                    &mut replay,
                    &scratch,
                    h,
                    conv_dim,
                    v_dim,
                    topk,
                    n_expert,
                    &gdn_layers,
                )?;
                let timing = if timing_iters > 0 {
                    let timing_base_seed = prepare_real_block_slice_sessions(
                        &ctx,
                        &mf,
                        &mm,
                        &prefix_sessions,
                        &prompt_ids,
                        &slot_positions,
                        slot_count,
                        start_block,
                        kv_capacity,
                        h,
                    )?;
                    let mut timing_base = prepare_real_block_slice_sessions(
                        &ctx,
                        &mf,
                        &mm,
                        &prefix_sessions,
                        &prompt_ids,
                        &slot_positions,
                        slot_count,
                        start_block,
                        kv_capacity,
                        h,
                    )?;
                    let timing_replay_seed = prepare_real_block_slice_sessions(
                        &ctx,
                        &mf,
                        &mm,
                        &prefix_sessions,
                        &prompt_ids,
                        &slot_positions,
                        slot_count,
                        start_block,
                        kv_capacity,
                        h,
                    )?;
                    let mut timing_replay = prepare_real_block_slice_sessions(
                        &ctx,
                        &mf,
                        &mm,
                        &prefix_sessions,
                        &prompt_ids,
                        &slot_positions,
                        slot_count,
                        start_block,
                        kv_capacity,
                        h,
                    )?;
                    let timing_validated_seed = prepare_real_block_slice_sessions(
                        &ctx,
                        &mf,
                        &mm,
                        &prefix_sessions,
                        &prompt_ids,
                        &slot_positions,
                        slot_count,
                        start_block,
                        kv_capacity,
                        h,
                    )?;
                    let mut timing_validated = prepare_real_block_slice_sessions(
                        &ctx,
                        &mf,
                        &mm,
                        &prefix_sessions,
                        &prompt_ids,
                        &slot_positions,
                        slot_count,
                        start_block,
                        kv_capacity,
                        h,
                    )?;
                    let timing_scratch =
                        GdnLayerReplayScratch::new(&ctx, slot_count, h, conv_dim, v_dim)?;
                    let baseline = time_cmd_reps_stats(&ctx, timing_warmup, timing_iters, |cmd| {
                        reset_block_slice_sessions(&ctx, &timing_base_seed, &mut timing_base)?;
                        let enc = KernelEncoder::begin(cmd);
                        encode_block_slice_baseline(
                            &mf,
                            &enc,
                            start_block,
                            n_blocks,
                            context as u32,
                            &mut timing_base,
                        )?;
                        enc.end();
                        Ok(())
                    })?;
                    let replay_stats =
                        time_cmd_reps_stats(&ctx, timing_warmup, timing_iters, |cmd| {
                            reset_block_slice_sessions(
                                &ctx,
                                &timing_replay_seed,
                                &mut timing_replay,
                            )?;
                            encode_block_slice_replay(
                                &ctx,
                                &mf,
                                &mm,
                                cmd,
                                start_block,
                                n_blocks,
                                context as u32,
                                &mut timing_replay,
                                &timing_scratch,
                                &gdn_layers,
                                h,
                                conv_dim,
                                v_dim,
                            )
                        })?;
                    let validated = time_validated_block_slice_replay(
                        &ctx,
                        &mf,
                        &mm,
                        start_block,
                        n_blocks,
                        context as u32,
                        &timing_validated_seed,
                        &mut timing_validated,
                        &timing_scratch,
                        &gdn_layers,
                        h,
                        conv_dim,
                        v_dim,
                        topk,
                        n_expert,
                        margin_threshold,
                        timing_warmup,
                        timing_iters,
                    )?;
                    let baseline_per_tok = baseline.avg_wall_ms / slot_count as f64;
                    let replay_per_tok = replay_stats.avg_wall_ms / slot_count as f64;
                    let validated_wall_per_tok = validated.avg_wall_ms / slot_count as f64;
                    let validated_gpu_per_tok = validated.avg_gpu_ms / slot_count as f64;
                    let baseline_p50_wall_per_tok = baseline.p50_wall_ms / slot_count as f64;
                    let replay_p50_wall_per_tok = replay_stats.p50_wall_ms / slot_count as f64;
                    let validated_p50_wall_per_tok = validated.p50_wall_ms / slot_count as f64;
                    let validated_p50_gpu_per_tok = validated.p50_gpu_ms / slot_count as f64;
                    let fallback_pct = validated.avg_fallback_slots / slot_count as f64;
                    let fallback_exact_per_tok = fallback_pct * baseline_per_tok;
                    let gross_save_pct = if baseline_per_tok > 0.0 {
                        (baseline_per_tok - replay_per_tok) / baseline_per_tok * 100.0
                    } else {
                        0.0
                    };
                    let net_save_pct = if baseline_per_tok > 0.0 {
                        (baseline_per_tok - validated_wall_per_tok - fallback_exact_per_tok)
                            / baseline_per_tok
                            * 100.0
                    } else {
                        0.0
                    };
                    let p50_net_save_pct = if baseline_p50_wall_per_tok > 0.0 {
                        (baseline_p50_wall_per_tok
                            - validated_p50_wall_per_tok
                            - fallback_exact_per_tok)
                            / baseline_p50_wall_per_tok
                            * 100.0
                    } else {
                        0.0
                    };
                    Some((
                        baseline_per_tok,
                        replay_per_tok,
                        gross_save_pct,
                        validated_wall_per_tok,
                        validated_gpu_per_tok,
                        validated.avg_fallback_slots,
                        fallback_pct * 100.0,
                        net_save_pct,
                        baseline_p50_wall_per_tok,
                        replay_p50_wall_per_tok,
                        validated_p50_wall_per_tok,
                        validated_p50_gpu_per_tok,
                        p50_net_save_pct,
                    ))
                } else {
                    None
                };
                let first_set = summary
                    .first_set_mismatch
                    .map(|(block, slot)| format!("block={block},slot={slot}"))
                    .unwrap_or_else(|| "none".to_string());
                let timing_cols = timing.map_or_else(
                    || "\t".repeat(13),
                    |(
                        baseline_per_tok,
                        replay_per_tok,
                        gross_save_pct,
                        validated_wall_per_tok,
                        validated_gpu_per_tok,
                        fallback_slots_avg,
                        fallback_pct,
                        net_save_pct,
                        baseline_p50_wall_per_tok,
                        replay_p50_wall_per_tok,
                        validated_p50_wall_per_tok,
                        validated_p50_gpu_per_tok,
                        p50_net_save_pct,
                    )| {
                        format!(
                            "\t{baseline_per_tok:.4}\t{replay_per_tok:.4}\t{gross_save_pct:.2}\t{validated_wall_per_tok:.4}\t{validated_gpu_per_tok:.4}\t{fallback_slots_avg:.2}\t{fallback_pct:.2}\t{net_save_pct:.2}\t{baseline_p50_wall_per_tok:.4}\t{replay_p50_wall_per_tok:.4}\t{validated_p50_wall_per_tok:.4}\t{validated_p50_gpu_per_tok:.4}\t{p50_net_save_pct:.2}"
                        )
                    },
                );
                println!(
                    "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{}\t{}\t{:.9}\t{:.6}{}",
                    start_block,
                    start_block + n_blocks,
                    context,
                    slot_count,
                    summary.route_order_mismatches,
                    summary.route_set_mismatches,
                    first_set,
                    summary.max_logit_abs,
                    summary.max_logit_rms,
                    summary.min_base_margin,
                    summary.min_replay_margin,
                    summary.replay_margin_lt_1e3,
                    summary.replay_margin_lt_5e3,
                    summary.min_x_cos,
                    summary.max_x_abs,
                    timing_cols,
                );
            }
        }
    }

    Ok(())
}

fn moe_for_block(mm: &MetalModel, block_idx: usize) -> Result<Option<&MetalMoeFfn>> {
    let block = mm
        .blocks
        .get(block_idx)
        .ok_or_else(|| anyhow!("block index {block_idx} >= {}", mm.blocks.len()))?;
    Ok(match block {
        MetalBlock::Gdn(b) => b.ffn_moe.as_ref(),
        MetalBlock::Attn(b) => b.ffn_moe.as_ref(),
    })
}

fn block_slice_has_attention(mm: &MetalModel, start_block: usize, n_blocks: usize) -> Result<bool> {
    let end_block = start_block
        .checked_add(n_blocks)
        .ok_or_else(|| anyhow!("start_block + blocks overflow"))?;
    if end_block > mm.blocks.len() {
        return Err(anyhow!(
            "block slice {}..{} outside available 0..{}",
            start_block,
            end_block,
            mm.blocks.len()
        ));
    }
    Ok(mm.blocks[start_block..end_block]
        .iter()
        .any(|block| matches!(block, MetalBlock::Attn(_))))
}

fn cpu_route_fingerprint(
    moe: &MetalMoeFfn,
    h_cpu: &[f32],
    topk: usize,
    n_expert: usize,
) -> RouteFingerprint {
    let mut logits = mat_vec_pub(&moe.gate_inp_cpu, h_cpu.len(), n_expert, h_cpu);
    let mut ranked: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    ranked.truncate(topk);
    let max_top = ranked.first().map(|&(_, v)| v).unwrap_or(f32::NEG_INFINITY);
    let mut sum = 0.0f32;
    let mut exp_vals = Vec::with_capacity(ranked.len());
    for &(_, v) in &ranked {
        let e = (v - max_top).exp();
        exp_vals.push(e);
        sum += e;
    }
    let inv = 1.0 / sum.max(6.103515625e-5);
    let shared_dot: f32 = h_cpu
        .iter()
        .zip(&moe.gate_inp_shexp_cpu[..h_cpu.len()])
        .map(|(a, b)| a * b)
        .sum();
    let idx = ranked.iter().map(|&(expert, _)| expert as i32).collect();
    let weight = exp_vals.into_iter().map(|v| v * inv).collect();
    let logit_margin = topk_logit_margin(&logits, topk);
    RouteFingerprint {
        idx,
        weight,
        shared_gate: 1.0 / (1.0 + (-shared_dot).exp()),
        logit_margin,
        logits: std::mem::take(&mut logits),
    }
}

fn seed_session_current_token(
    ctx: &MetalContext,
    mm: &MetalModel,
    session: &mut MetalSession,
    token_id: i32,
    h: usize,
) -> Result<()> {
    unsafe {
        let ptr = session.ids_buf.buffer.contents().as_ptr() as *mut i32;
        *ptr = token_id;
    }
    let cmd = ctx.queue.commandBuffer().context("router check seed cmd")?;
    let enc = KernelEncoder::begin(&cmd);
    encode_get_rows_f32(
        ctx,
        &enc,
        &mm.token_embd,
        &session.ids_buf,
        &session.x,
        1,
        h,
    )?;
    enc.end();
    cmd.commit();
    cmd.waitUntilCompleted();
    Ok(())
}

fn run_decode_moe_router_repack_check(args: DecodeMoeRouterRepackCheckArgs) -> Result<()> {
    let DecodeMoeRouterRepackCheckArgs {
        model,
        file,
        tokens,
        mut contexts,
    } = args;
    if tokens == 0 {
        return Err(anyhow!("--tokens must be >= 1"));
    }
    if file.is_empty() {
        return Err(anyhow!("--file must include at least one prompt"));
    }
    if file.len() < tokens {
        return Err(anyhow!(
            "got {} --file entries, need at least --tokens {tokens}",
            file.len()
        ));
    }
    if contexts.is_empty() {
        return Err(anyhow!("--context must include at least one entry"));
    }
    contexts.sort_unstable();
    contexts.dedup();

    let ctx = MetalContext::new().context("init MetalContext")?;
    let g = GgufFile::open(&model).with_context(|| format!("open {}", model.display()))?;
    let m = Model::from_gguf(&g).context("parse model")?;
    let mm = MetalModel::load(&ctx, &g, &m).context("metal-load model")?;
    if mm.arch.kind != qwen_llm::model::ArchKind::Moe {
        return Err(anyhow!(
            "decode-moe-router-repack-check currently requires an MoE model"
        ));
    }
    let mf = MetalForward::new(&ctx, &mm);
    let tok = NativeTokenizer::from_gguf(&g).context("open native GGUF tokenizer")?;
    let mut prompt_ids = Vec::with_capacity(file.len());
    for path in &file {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let ids = tok
            .encode(&text, false)
            .with_context(|| format!("tokenize {}", path.display()))?;
        prompt_ids.push(ids);
    }
    let max_context = *contexts.last().expect("non-empty contexts");
    for (path, ids) in file.iter().zip(prompt_ids.iter()).take(tokens) {
        if ids.len() <= max_context {
            return Err(anyhow!(
                "prompt {} has {} tokens, need at least {} for context sweep",
                path.display(),
                ids.len(),
                max_context + 1
            ));
        }
    }

    let h = mm.arch.hidden_size as usize;
    let topk = mm.arch.expert_used_count.min(mm.arch.expert_count) as usize;
    let n_expert = mm.arch.expert_count as usize;
    let first_moe = (0..mm.blocks.len())
        .find_map(|i| moe_for_block(&mm, i).ok().flatten())
        .ok_or_else(|| anyhow!("model has no MoE FFN blocks"))?;
    let router_dtype = first_moe.gate_inp.dtype;

    println!(
        "[decode-moe-router-repack-check] model={} files={} contexts={} slots={} blocks={} topk={} experts={} router_dtype={:?} env_QWEN_MOE_ROUTER_F16={}",
        model.display(),
        file.len(),
        contexts
            .iter()
            .map(|p| p.to_string())
            .collect::<Vec<_>>()
            .join(","),
        tokens,
        mm.blocks.len(),
        topk,
        n_expert,
        router_dtype,
        env_flag_enabled("QWEN_MOE_ROUTER_F16"),
    );
    println!(
        "context\tslots\troute_checks\troute_order_mismatches\troute_set_mismatches\tfirst_set_mismatch\tmax_logit_abs\tmax_logit_rms\tmin_f32_margin\tmin_gpu_margin\tmax_weight_abs\tmax_shared_abs"
    );

    for &context in &contexts {
        let kv_capacity = context + 32;
        let mut sessions = Vec::with_capacity(tokens);
        for slot in 0..tokens {
            let ids = &prompt_ids[slot];
            let mut s = MetalSession::fresh(&ctx, &mm, kv_capacity)
                .with_context(|| format!("router check session slot {slot}"))?;
            for (p, &token_id) in ids.iter().take(context).enumerate() {
                let _ = mf.single_token_argmax_profiled(token_id, p as u32, &mut s)?;
            }
            seed_session_current_token(&ctx, &mm, &mut s, ids[context], h)?;
            sessions.push(s);
        }

        let mut route_checks = 0usize;
        let mut order_mismatches = 0usize;
        let mut set_mismatches = 0usize;
        let mut first_set_mismatch: Option<(usize, usize)> = None;
        let mut max_logit_abs = 0.0f32;
        let mut max_logit_rms = 0.0f64;
        let mut min_f32_margin = f32::INFINITY;
        let mut min_gpu_margin = f32::INFINITY;
        let mut max_weight_abs = 0.0f32;
        let mut max_shared_abs = 0.0f32;

        for block_i in 0..mm.blocks.len() {
            let Some(moe) = moe_for_block(&mm, block_i)? else {
                continue;
            };
            for (slot, session) in sessions.iter_mut().enumerate() {
                let cmd = ctx.queue.commandBuffer().context("router mixer cmd")?;
                let enc = KernelEncoder::begin(&cmd);
                mf.encode_moe_mixer_prep_by_index(&enc, block_i, context as u32, session)?;
                enc.end();
                cmd.commit();
                cmd.waitUntilCompleted();

                let h_cpu = read_f32_tensor_prefix(&session.h, h);
                let f32_route = cpu_route_fingerprint(moe, &h_cpu, topk, n_expert);

                let cmd = ctx.queue.commandBuffer().context("router route cmd")?;
                let enc = KernelEncoder::begin(&cmd);
                mf.encode_moe_route_prepare_by_index(&enc, block_i, session)?;
                enc.end();
                cmd.commit();
                cmd.waitUntilCompleted();

                let gpu_route = read_route_fingerprint(session, topk, n_expert);
                route_checks += 1;
                if f32_route.idx != gpu_route.idx {
                    order_mismatches += 1;
                }
                if !same_i32_set(&f32_route.idx, &gpu_route.idx) {
                    set_mismatches += 1;
                    first_set_mismatch.get_or_insert((block_i, slot));
                }
                max_logit_abs =
                    max_logit_abs.max(f32_max_abs_delta(&f32_route.logits, &gpu_route.logits));
                max_logit_rms =
                    max_logit_rms.max(f32_rms_delta(&f32_route.logits, &gpu_route.logits));
                min_f32_margin = min_f32_margin.min(f32_route.logit_margin);
                min_gpu_margin = min_gpu_margin.min(gpu_route.logit_margin);
                max_weight_abs =
                    max_weight_abs.max(route_weight_max_abs(&f32_route.weight, &gpu_route.weight));
                max_shared_abs =
                    max_shared_abs.max((f32_route.shared_gate - gpu_route.shared_gate).abs());

                let cmd = ctx.queue.commandBuffer().context("router ffn cmd")?;
                let enc = KernelEncoder::begin(&cmd);
                mf.encode_moe_ffn_after_mixer_by_index(&enc, block_i, session)?;
                enc.end();
                cmd.commit();
                cmd.waitUntilCompleted();
            }
        }

        let first_set = first_set_mismatch
            .map(|(block, slot)| format!("block={block},slot={slot}"))
            .unwrap_or_else(|| "none".to_string());
        println!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}",
            context,
            tokens,
            route_checks,
            order_mismatches,
            set_mismatches,
            first_set,
            max_logit_abs,
            max_logit_rms,
            min_f32_margin,
            min_gpu_margin,
            max_weight_abs,
            max_shared_abs,
        );
    }

    Ok(())
}

fn run_moe_down_micro(args: MoeDownMicroArgs) -> Result<()> {
    let MoeDownMicroArgs {
        model,
        iters,
        warmup,
        tokens,
        fused_routed_q4q5,
        route_capture_ctx,
        route_capture_token_pattern,
        synthetic_f_exp,
        synthetic_h,
        synthetic_layers,
        k512_r2,
        legacy_k512,
        check_k512_r2,
    } = args;
    if iters == 0 {
        return Err(anyhow!("--iters must be >= 1"));
    }
    if tokens == 0 {
        return Err(anyhow!("--tokens must be >= 1"));
    }
    if fused_routed_q4q5 && tokens != 1 {
        return Err(anyhow!("--fused-routed-q4q5 currently requires --tokens 1"));
    }

    let ctx = MetalContext::new().context("init MetalContext")?;
    let g = GgufFile::open(&model).with_context(|| format!("open {}", model.display()))?;
    let m = Model::from_gguf(&g).context("parse model")?;
    let mm = MetalModel::load(&ctx, &g, &m).context("metal-load model")?;
    let arch = &mm.arch;
    if arch.kind != qwen_llm::model::ArchKind::Moe {
        return Err(anyhow!("moe-down-micro requires an MoE model"));
    }
    let h = arch.hidden_size as usize;
    let f_exp = arch.expert_feed_forward_length as usize;
    let n_expert = arch.expert_count as usize;
    let topk = arch.expert_used_count.min(arch.expert_count) as usize;
    let moe_blocks: Vec<_> = mm
        .blocks
        .iter()
        .filter_map(|b| match b {
            MetalBlock::Gdn(g) => g.ffn_moe.as_ref(),
            MetalBlock::Attn(a) => a.ffn_moe.as_ref(),
        })
        .collect();
    let q5_moes: Vec<_> = moe_blocks
        .iter()
        .copied()
        .filter(|moe| moe.down_exps.dtype == GgmlType::Q5_K)
        .collect();
    let q4q5_moes: Vec<_> = moe_blocks
        .iter()
        .copied()
        .filter(|moe| {
            moe.gate_exps.dtype == GgmlType::Q4_K
                && moe.up_exps.dtype == GgmlType::Q4_K
                && moe.down_exps.dtype == GgmlType::Q5_K
        })
        .collect();
    let bench_moes = if fused_routed_q4q5 {
        &q4q5_moes
    } else {
        &q5_moes
    };
    if bench_moes.is_empty() {
        let wanted = if fused_routed_q4q5 {
            "Q4_K/Q4_K/Q5_K routed expert banks"
        } else {
            "Q5_K routed down expert banks"
        };
        return Err(anyhow!("model has no eligible {wanted}"));
    }
    if synthetic_f_exp.is_some() != synthetic_h.is_some() {
        return Err(anyhow!(
            "--synthetic-f-exp and --synthetic-h must be passed together"
        ));
    }
    let synthetic = synthetic_f_exp.is_some();
    if fused_routed_q4q5 && synthetic {
        return Err(anyhow!(
            "--fused-routed-q4q5 requires real Q4/Q5 expert banks"
        ));
    }
    if synthetic && route_capture_ctx.is_some() {
        return Err(anyhow!(
            "--route-capture-ctx is only supported for real routed-down layers"
        ));
    }
    let f_run = synthetic_f_exp.unwrap_or(f_exp);
    let h_run = synthetic_h.unwrap_or(h);
    let layer_count = synthetic_layers.unwrap_or(bench_moes.len());
    if synthetic && layer_count == 0 {
        return Err(anyhow!("--synthetic-layers must be >= 1"));
    }
    if f_run % 256 != 0 {
        return Err(anyhow!("routed-down f_exp must be divisible by 256"));
    }
    let synthetic_weight = if synthetic {
        Some(MetalTensor::zeros_dtype(
            &ctx,
            vec![f_run as u64, h_run as u64, n_expert as u64, 1],
            GgmlType::Q5_K,
        )?)
    } else {
        None
    };

    let slots = tokens * topk;
    let inner = MetalTensor::zeros_f32(&ctx, vec![(slots * f_run) as u64])?;
    let x = MetalTensor::zeros_f32(&ctx, vec![(tokens * h_run) as u64])?;
    let topk_idx = MetalTensor::zeros_f32(&ctx, vec![slots as u64])?;
    let topk_w = MetalTensor::zeros_f32(&ctx, vec![slots as u64])?;
    let out = MetalTensor::zeros_f32(&ctx, vec![(tokens * h_run) as u64])?;
    unsafe {
        let x_ptr = x.buffer.contents().as_ptr() as *mut f32;
        for i in 0..(tokens * h_run) {
            *x_ptr.add(i) = ((i % 31) as f32 - 15.0) * 0.01;
        }
        let inner_ptr = inner.buffer.contents().as_ptr() as *mut f32;
        for i in 0..(slots * f_run) {
            *inner_ptr.add(i) = ((i % 17) as f32 - 8.0) * 0.0125;
        }
        let idx_ptr = topk_idx.buffer.contents().as_ptr() as *mut i32;
        let w_ptr = topk_w.buffer.contents().as_ptr() as *mut f32;
        for slot in 0..slots {
            *idx_ptr.add(slot) = ((slot * 17) % n_expert) as i32;
            *w_ptr.add(slot) = 1.0 / topk as f32;
        }
    }

    let mut captured_route_stats = None;
    let captured_routes: Option<(usize, Vec<(MetalTensor, MetalTensor, MetalTensor)>)> =
        if let Some(capture_ctx) = route_capture_ctx {
            let mf = MetalForward::new(&ctx, &mm);
            let mut capture_s = MetalSession::fresh(&ctx, &mm, capture_ctx + tokens + 16)
                .context("route-capture session")?;
            for pos in 0..capture_ctx {
                let _ = mf.single_token(0, pos as u32, &mut capture_s)?;
            }
            let mut all_routes_by_token = Vec::with_capacity(tokens);
            for tok in 0..tokens {
                let token_id =
                    capture_replay_token(route_capture_token_pattern, tok, arch.vocab_size);
                let all_routes = mf.capture_moe_gateup_replay_for_token(
                    token_id,
                    (capture_ctx + tok) as u32,
                    &mut capture_s,
                )?;
                if all_routes.len() != moe_blocks.len() {
                    return Err(anyhow!(
                        "captured {} MoE route rows, expected {}",
                        all_routes.len(),
                        moe_blocks.len()
                    ));
                }
                all_routes_by_token.push(all_routes);
            }
            let eligible_indices: Vec<_> = moe_blocks
                .iter()
                .enumerate()
                .filter_map(|(i, moe)| {
                    let eligible = if fused_routed_q4q5 {
                        moe.gate_exps.dtype == GgmlType::Q4_K
                            && moe.up_exps.dtype == GgmlType::Q4_K
                            && moe.down_exps.dtype == GgmlType::Q5_K
                    } else {
                        moe.down_exps.dtype == GgmlType::Q5_K
                    };
                    if eligible { Some(i) } else { None }
                })
                .collect();
            if eligible_indices.len() != bench_moes.len() {
                return Err(anyhow!(
                    "captured {} eligible route rows, expected {}",
                    eligible_indices.len(),
                    bench_moes.len()
                ));
            }
            captured_route_stats = Some(summarize_moe_route_batch(
                &all_routes_by_token,
                &eligible_indices,
                n_expert,
                topk,
            )?);

            let mut tensors = Vec::with_capacity(eligible_indices.len());
            for &moe_i in &eligible_indices {
                let idx = MetalTensor::zeros_f32(&ctx, vec![slots as u64])?;
                let weight = MetalTensor::zeros_f32(&ctx, vec![slots as u64])?;
                let hidden = MetalTensor::zeros_f32(&ctx, vec![(tokens * h_run) as u64])?;
                unsafe {
                    let idx_ptr = idx.buffer.contents().as_ptr() as *mut i32;
                    let w_ptr = weight.buffer.contents().as_ptr() as *mut f32;
                    let h_ptr = hidden.buffer.contents().as_ptr() as *mut f32;
                    for tok in 0..tokens {
                        let route = &all_routes_by_token[tok][moe_i];
                        if route.topk_idx.len() != topk || route.topk_weight.len() != topk {
                            return Err(anyhow!(
                                "captured route has idx={} weight={}, expected topk={topk}",
                                route.topk_idx.len(),
                                route.topk_weight.len()
                            ));
                        }
                        if route.hidden.len() != h_run {
                            return Err(anyhow!(
                                "captured hidden has {}, expected h={h_run}",
                                route.hidden.len()
                            ));
                        }
                        std::ptr::copy_nonoverlapping(
                            route.hidden.as_ptr(),
                            h_ptr.add(tok * h_run),
                            h_run,
                        );
                        for slot in 0..topk {
                            let expert = route.topk_idx[slot];
                            if expert < 0 || expert as usize >= n_expert {
                                return Err(anyhow!(
                                    "captured expert id {expert} outside n_expert={n_expert}"
                                ));
                            }
                            let out_slot = tok * topk + slot;
                            *idx_ptr.add(out_slot) = expert;
                            *w_ptr.add(out_slot) = route.topk_weight[slot];
                        }
                    }
                }
                tensors.push((idx, weight, hidden));
            }
            Some((capture_ctx, tensors))
        } else {
            None
        };

    let active_weight_bytes: f64 = if let Some(weight) = synthetic_weight.as_ref() {
        weight.n_bytes() as f64 * layer_count as f64 * tokens as f64 * topk as f64 / n_expert as f64
    } else if fused_routed_q4q5 {
        bench_moes
            .iter()
            .map(|moe| {
                (moe.gate_exps.n_bytes() + moe.up_exps.n_bytes() + moe.down_exps.n_bytes()) as f64
                    * topk as f64
                    * tokens as f64
                    / n_expert as f64
            })
            .sum()
    } else {
        bench_moes
            .iter()
            .map(|moe| {
                moe.down_exps.n_bytes() as f64 * tokens as f64 * topk as f64 / n_expert as f64
            })
            .sum()
    };
    let inner_bytes = (slots * f_run * std::mem::size_of::<f32>()) as f64;
    let out_bytes = (tokens * h_run * std::mem::size_of::<f32>()) as f64;
    if k512_r2 && legacy_k512 {
        return Err(anyhow!("--k512-r2 conflicts with --legacy-k512"));
    }
    let use_k512_r2 = k512_r2 || (f_run == 512 && !legacy_k512);
    if use_k512_r2 && f_run != 512 {
        return Err(anyhow!(
            "QWEN_MOE_DOWN_Q5_K512_R2 requires f_exp=512, got {f_run}"
        ));
    }
    if check_k512_r2 {
        if f_run != 512 {
            return Err(anyhow!(
                "QWEN_MOE_DOWN_Q5_K512_R2_CHECK requires f_exp=512, got {f_run}"
            ));
        }
        let ref_out = MetalTensor::zeros_f32(&ctx, vec![(tokens * h_run) as u64])?;
        let alt_out = MetalTensor::zeros_f32(&ctx, vec![(tokens * h_run) as u64])?;
        let cmd = ctx.queue.commandBuffer().context("moe-down check cmd")?;
        let enc = KernelEncoder::begin(&cmd);
        if let Some(weight) = synthetic_weight.as_ref() {
            encode_moe_down_weighted_sum_q5_K_f32_packed_slots(
                &ctx, &enc, weight, &inner, &topk_idx, &topk_w, &ref_out, f_run, h_run, n_expert,
                topk, tokens,
            )?;
            encode_moe_down_weighted_sum_q5_K_f32_packed_slots_k512_r2(
                &ctx, &enc, weight, &inner, &topk_idx, &topk_w, &alt_out, f_run, h_run, n_expert,
                topk, tokens,
            )?;
        } else {
            let moe = q5_moes[0];
            encode_moe_down_weighted_sum_q5_K_f32_packed_slots(
                &ctx,
                &enc,
                &moe.down_exps,
                &inner,
                &topk_idx,
                &topk_w,
                &ref_out,
                f_run,
                h_run,
                n_expert,
                topk,
                tokens,
            )?;
            encode_moe_down_weighted_sum_q5_K_f32_packed_slots_k512_r2(
                &ctx,
                &enc,
                &moe.down_exps,
                &inner,
                &topk_idx,
                &topk_w,
                &alt_out,
                f_run,
                h_run,
                n_expert,
                topk,
                tokens,
            )?;
        }
        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();
        let read = |t: &MetalTensor| -> Vec<f32> {
            let n = t.n_elements() as usize;
            let mut xs = vec![0.0f32; n];
            unsafe {
                let src = (t.buffer.contents().as_ptr() as *const f32).add((t.offset / 4) as usize);
                std::ptr::copy_nonoverlapping(src, xs.as_mut_ptr(), n);
            }
            xs
        };
        let a = read(&ref_out);
        let b = read(&alt_out);
        let max_abs = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        let dot: f64 = a.iter().zip(&b).map(|(x, y)| *x as f64 * *y as f64).sum();
        let na = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let nb = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let cos = if na > 0.0 && nb > 0.0 {
            dot / (na * nb)
        } else {
            1.0
        };
        println!("[moe-down-micro-check] max_abs={max_abs:.6} cos={cos:.9}");
    }

    let (wall, gpu) = time_gpu_reps(&ctx, warmup, iters, |enc| {
        if fused_routed_q4q5 {
            for (layer_i, moe) in bench_moes.iter().enumerate() {
                let (route_idx, route_w, layer_x) = captured_routes
                    .as_ref()
                    .map(|(_, routes)| (&routes[layer_i].0, &routes[layer_i].1, &routes[layer_i].2))
                    .unwrap_or((&topk_idx, &topk_w, &x));
                encode_moe_fused_routed_q4q5_token_f32(
                    &ctx,
                    enc,
                    &moe.gate_exps,
                    &moe.up_exps,
                    &moe.down_exps,
                    layer_x,
                    route_idx,
                    route_w,
                    &out,
                    h_run,
                    f_run,
                    n_expert,
                    topk,
                    tokens,
                )?;
            }
        } else if let Some(weight) = synthetic_weight.as_ref() {
            for _ in 0..layer_count {
                if use_k512_r2 {
                    encode_moe_down_weighted_sum_q5_K_f32_packed_slots_k512_r2(
                        &ctx, enc, weight, &inner, &topk_idx, &topk_w, &out, f_run, h_run,
                        n_expert, topk, tokens,
                    )?;
                } else {
                    encode_moe_down_weighted_sum_q5_K_f32_packed_slots(
                        &ctx, enc, weight, &inner, &topk_idx, &topk_w, &out, f_run, h_run,
                        n_expert, topk, tokens,
                    )?;
                }
            }
        } else {
            for (layer_i, moe) in bench_moes.iter().enumerate() {
                let (route_idx, route_w) = captured_routes
                    .as_ref()
                    .map(|(_, routes)| (&routes[layer_i].0, &routes[layer_i].1))
                    .unwrap_or((&topk_idx, &topk_w));
                if use_k512_r2 {
                    encode_moe_down_weighted_sum_q5_K_f32_packed_slots_k512_r2(
                        &ctx,
                        enc,
                        &moe.down_exps,
                        &inner,
                        route_idx,
                        route_w,
                        &out,
                        f_run,
                        h_run,
                        n_expert,
                        topk,
                        tokens,
                    )?;
                } else {
                    encode_moe_down_weighted_sum_q5_K_f32_packed_slots(
                        &ctx,
                        enc,
                        &moe.down_exps,
                        &inner,
                        route_idx,
                        route_w,
                        &out,
                        f_run,
                        h_run,
                        n_expert,
                        topk,
                        tokens,
                    )?;
                }
            }
        }
        Ok(())
    })?;

    let weight_gb = active_weight_bytes / 1e9;
    let activation_gb = if fused_routed_q4q5 {
        out_bytes * 2.0 * layer_count as f64 / 1e9
    } else {
        (inner_bytes + out_bytes) * layer_count as f64 / 1e9
    };
    let route_mode = captured_routes
        .as_ref()
        .map(|(ctx, _)| format!("captured(ctx={ctx},pattern={route_capture_token_pattern:?})"))
        .unwrap_or_else(|| "synthetic".to_string());
    println!(
        "[moe-down-micro] model={} mode={} route_mode={} kernel={} q5_layers={} layers={} h={} f_exp={} n_expert={} topk={} tokens={} warmup={} iters={}",
        model.display(),
        if synthetic { "synthetic" } else { "real" },
        route_mode,
        if fused_routed_q4q5 {
            "fused_q4q5"
        } else if use_k512_r2 {
            "k512_r2"
        } else {
            "default"
        },
        bench_moes.len(),
        layer_count,
        h_run,
        f_run,
        n_expert,
        topk,
        tokens,
        warmup,
        iters
    );
    if let Some(stats) = captured_route_stats {
        println!(
            "[moe-down-route-stats] layers={} tokens={} slots_per_layer={} avg_unique_experts={:.2} avg_max_slots={:.2} avg_reuse={:.2}",
            stats.layers,
            stats.tokens,
            stats.slots_per_layer,
            stats.avg_unique_experts,
            stats.avg_max_slots,
            stats.avg_reuse
        );
    }
    println!("phase\tactive_weight_gb\tactivation_gb\tavg_wall_ms\tavg_gpu_ms\tweight_gb_s");
    let phase = if fused_routed_q4q5 {
        "q4q5_fused_routed"
    } else {
        "q5_down_weighted_sum"
    };
    println!(
        "{phase}\t{weight_gb:.4}\t{activation_gb:.4}\t{wall:.4}\t{gpu:.4}\t{:.1}",
        weight_gb / (gpu / 1e3)
    );
    Ok(())
}

fn run_moe_batch_sweep(args: MoeBatchSweepArgs) -> Result<()> {
    let MoeBatchSweepArgs {
        model,
        iters,
        warmup,
        mut tokens,
        route_capture_ctx,
        route_capture_stride,
        file,
        route_capture_token_pattern,
        mut slot_orders,
    } = args;
    if iters == 0 {
        return Err(anyhow!("--iters must be >= 1"));
    }
    if tokens.is_empty() || tokens.iter().any(|&n| n == 0) {
        return Err(anyhow!("--tokens entries must be >= 1"));
    }
    if route_capture_stride == 0 {
        return Err(anyhow!("--route-capture-stride must be >= 1"));
    }
    if slot_orders.is_empty() {
        return Err(anyhow!("--slot-order must contain at least one entry"));
    }
    tokens.sort_unstable();
    tokens.dedup();
    slot_orders.sort_unstable();
    slot_orders.dedup();
    let max_tokens = *tokens.last().expect("non-empty tokens");

    let ctx = MetalContext::new().context("init MetalContext")?;
    let g = GgufFile::open(&model).with_context(|| format!("open {}", model.display()))?;
    let m = Model::from_gguf(&g).context("parse model")?;
    let mm = MetalModel::load(&ctx, &g, &m).context("metal-load model")?;
    let arch = &mm.arch;
    if arch.kind != qwen_llm::model::ArchKind::Moe {
        return Err(anyhow!("moe-batch-sweep requires an MoE model"));
    }
    let h = arch.hidden_size as usize;
    let f_exp = arch.expert_feed_forward_length as usize;
    let n_expert = arch.expert_count as usize;
    let topk = arch.expert_used_count.min(arch.expert_count) as usize;
    let moe_blocks: Vec<_> = mm
        .blocks
        .iter()
        .filter_map(|b| match b {
            MetalBlock::Gdn(g) => g.ffn_moe.as_ref(),
            MetalBlock::Attn(a) => a.ffn_moe.as_ref(),
        })
        .collect();
    let q4_moes: Vec<_> = moe_blocks
        .iter()
        .copied()
        .filter(|moe| moe.gate_exps.dtype == GgmlType::Q4_K && moe.up_exps.dtype == GgmlType::Q4_K)
        .collect();
    let q4_indices: Vec<_> = moe_blocks
        .iter()
        .enumerate()
        .filter_map(|(i, moe)| {
            if moe.gate_exps.dtype == GgmlType::Q4_K && moe.up_exps.dtype == GgmlType::Q4_K {
                Some(i)
            } else {
                None
            }
        })
        .collect();
    let q5_moes: Vec<_> = moe_blocks
        .iter()
        .copied()
        .filter(|moe| moe.down_exps.dtype == GgmlType::Q5_K)
        .collect();
    let q5_indices: Vec<_> = moe_blocks
        .iter()
        .enumerate()
        .filter_map(|(i, moe)| {
            if moe.down_exps.dtype == GgmlType::Q5_K {
                Some(i)
            } else {
                None
            }
        })
        .collect();
    if q4_moes.is_empty() || q5_moes.is_empty() {
        return Err(anyhow!(
            "moe-batch-sweep requires Q4_K gate/up and Q5_K down expert banks"
        ));
    }
    if f_exp % 256 != 0 {
        return Err(anyhow!("routed f_exp must be divisible by 256"));
    }
    let use_k512_r2 = f_exp == 512;

    let last_capture_pos = route_capture_ctx + (max_tokens - 1) * route_capture_stride;
    let prompt_ids = if file.is_empty() {
        Vec::new()
    } else {
        let tok = NativeTokenizer::from_gguf(&g).context("open native GGUF tokenizer")?;
        let mut all = Vec::with_capacity(file.len());
        for path in &file {
            let text = std::fs::read_to_string(path)
                .with_context(|| format!("read {}", path.display()))?;
            let ids = tok
                .encode(&text, false)
                .with_context(|| format!("tokenize {}", path.display()))?;
            all.push(ids);
        }
        all
    };
    if prompt_ids.len() == 1 {
        let need = last_capture_pos + 1;
        if prompt_ids[0].len() < need {
            return Err(anyhow!(
                "prompt file has {} tokens, need at least last capture position + 1 = {need}",
                prompt_ids[0].len()
            ));
        }
    } else if prompt_ids.len() > 1 {
        if prompt_ids.len() < max_tokens {
            return Err(anyhow!(
                "got {} --file entries, need at least max(tokens) = {max_tokens}",
                prompt_ids.len()
            ));
        }
        let need = route_capture_ctx + 1;
        for (path, ids) in file.iter().zip(prompt_ids.iter()) {
            if ids.len() < need {
                return Err(anyhow!(
                    "prompt file {} has {} tokens, need at least route_capture_ctx + 1 = {need}",
                    path.display(),
                    ids.len()
                ));
            }
        }
    }

    let mf = MetalForward::new(&ctx, &mm);
    let mut all_routes_by_token = Vec::with_capacity(max_tokens);
    if prompt_ids.len() > 1 {
        for tok in 0..max_tokens {
            let ids = &prompt_ids[tok];
            let mut capture_s = MetalSession::fresh(&ctx, &mm, route_capture_ctx + 17)
                .context("route-capture session")?;
            for (pos, &token_id) in ids.iter().take(route_capture_ctx).enumerate() {
                let _ = mf.single_token(token_id, pos as u32, &mut capture_s)?;
            }
            let all_routes = mf.capture_moe_gateup_replay_for_token(
                ids[route_capture_ctx],
                route_capture_ctx as u32,
                &mut capture_s,
            )?;
            if all_routes.len() != moe_blocks.len() {
                return Err(anyhow!(
                    "captured {} MoE route rows, expected {}",
                    all_routes.len(),
                    moe_blocks.len()
                ));
            }
            all_routes_by_token.push(all_routes);
        }
    } else {
        let mut capture_s = MetalSession::fresh(&ctx, &mm, last_capture_pos + 17)
            .context("route-capture session")?;
        for pos in 0..route_capture_ctx {
            let token_id = prompt_ids.first().map(|ids| ids[pos]).unwrap_or(0);
            let _ = mf.single_token(token_id, pos as u32, &mut capture_s)?;
        }
        let mut next_pos = route_capture_ctx;
        for tok in 0..max_tokens {
            let target_pos = route_capture_ctx + tok * route_capture_stride;
            for pos in next_pos..target_pos {
                let token_id = prompt_ids.first().map(|ids| ids[pos]).unwrap_or(0);
                let _ = mf.single_token(token_id, pos as u32, &mut capture_s)?;
            }
            let token_id = prompt_ids
                .first()
                .map(|ids| ids[target_pos])
                .unwrap_or_else(|| {
                    capture_replay_token(route_capture_token_pattern, tok, arch.vocab_size)
                });
            let all_routes = mf.capture_moe_gateup_replay_for_token(
                token_id,
                target_pos as u32,
                &mut capture_s,
            )?;
            if all_routes.len() != moe_blocks.len() {
                return Err(anyhow!(
                    "captured {} MoE route rows, expected {}",
                    all_routes.len(),
                    moe_blocks.len()
                ));
            }
            all_routes_by_token.push(all_routes);
            next_pos = target_pos + 1;
        }
    }

    let gateup_weight_gb_per_token: f64 = q4_moes
        .iter()
        .map(|moe| (moe.gate_exps.n_bytes() + moe.up_exps.n_bytes()) as f64)
        .sum::<f64>()
        * topk as f64
        / n_expert as f64
        / 1e9;
    let down_weight_gb_per_token: f64 = q5_moes
        .iter()
        .map(|moe| moe.down_exps.n_bytes() as f64)
        .sum::<f64>()
        * topk as f64
        / n_expert as f64
        / 1e9;

    println!(
        "[moe-batch-sweep] model={} route_mode={} slot_orders={} q4_layers={} q5_layers={} h={} f_exp={} n_expert={} topk={} warmup={} iters={}",
        model.display(),
        if file.len() > 1 {
            format!(
                "captured(ctx={},independent_files={})",
                route_capture_ctx,
                file.len()
            )
        } else if let Some(path) = file.first() {
            format!(
                "captured(ctx={},stride={},file={})",
                route_capture_ctx,
                route_capture_stride,
                path.display()
            )
        } else {
            format!(
                "captured(ctx={},stride={},pattern={:?})",
                route_capture_ctx, route_capture_stride, route_capture_token_pattern
            )
        },
        slot_orders
            .iter()
            .map(|order| order.label())
            .collect::<Vec<_>>()
            .join(","),
        q4_moes.len(),
        q5_moes.len(),
        h,
        f_exp,
        n_expert,
        topk,
        warmup,
        iters
    );
    println!(
        "slot_order\ttokens\tgateup_gpu_ms\tdown_gpu_ms\tcombined_ms_per_token\tgateup_gb_s\tdown_gb_s\tq4_unique\tq4_max\tq4_reuse\tq5_unique\tq5_max\tq5_reuse"
    );

    for &n_tokens in &tokens {
        let routes = &all_routes_by_token[..n_tokens];
        let q4_stats = summarize_moe_route_batch(routes, &q4_indices, n_expert, topk)?;
        let q5_stats = summarize_moe_route_batch(routes, &q5_indices, n_expert, topk)?;
        let slots = n_tokens * topk;
        for &slot_order in &slot_orders {
            let gateup_inputs =
                captured_gateup_tensors(&ctx, routes, &q4_indices, h, n_expert, topk, slot_order)?;
            let down_inputs =
                captured_down_tensors(&ctx, routes, &q5_indices, n_expert, topk, slot_order)?;
            let inner = MetalTensor::zeros_f32(&ctx, vec![(slots * f_exp) as u64])?;
            let down_out = MetalTensor::zeros_f32(&ctx, vec![(n_tokens * h) as u64])?;
            unsafe {
                let inner_ptr = inner.buffer.contents().as_ptr() as *mut f32;
                for i in 0..(slots * f_exp) {
                    *inner_ptr.add(i) = ((i % 17) as f32 - 8.0) * 0.0125;
                }
            }

            let (_gateup_wall, gateup_gpu) = time_gpu_reps(&ctx, warmup, iters, |enc| {
                for (layer_i, moe) in q4_moes.iter().enumerate() {
                    let (layer_x, route_idx) =
                        (&gateup_inputs[layer_i].0, &gateup_inputs[layer_i].1);
                    if n_tokens == 1 {
                        encode_moe_swiglu_q4_K_f32(
                            &ctx,
                            enc,
                            &moe.gate_exps,
                            &moe.up_exps,
                            layer_x,
                            route_idx,
                            &inner,
                            h,
                            f_exp,
                            n_expert,
                            topk,
                        )?;
                    } else {
                        encode_moe_swiglu_q4_K_f32_packed_slots(
                            &ctx,
                            enc,
                            &moe.gate_exps,
                            &moe.up_exps,
                            layer_x,
                            route_idx,
                            &inner,
                            h,
                            f_exp,
                            n_expert,
                            topk,
                            n_tokens,
                        )?;
                    }
                }
                Ok(())
            })?;

            let (_down_wall, down_gpu) = time_gpu_reps(&ctx, warmup, iters, |enc| {
                for (layer_i, moe) in q5_moes.iter().enumerate() {
                    let (route_idx, route_w) = (&down_inputs[layer_i].0, &down_inputs[layer_i].1);
                    if use_k512_r2 {
                        encode_moe_down_weighted_sum_q5_K_f32_packed_slots_k512_r2(
                            &ctx,
                            enc,
                            &moe.down_exps,
                            &inner,
                            route_idx,
                            route_w,
                            &down_out,
                            f_exp,
                            h,
                            n_expert,
                            topk,
                            n_tokens,
                        )?;
                    } else {
                        encode_moe_down_weighted_sum_q5_K_f32_packed_slots(
                            &ctx,
                            enc,
                            &moe.down_exps,
                            &inner,
                            route_idx,
                            route_w,
                            &down_out,
                            f_exp,
                            h,
                            n_expert,
                            topk,
                            n_tokens,
                        )?;
                    }
                }
                Ok(())
            })?;

            let combined_per_token = (gateup_gpu + down_gpu) / n_tokens as f64;
            let gateup_gb_s = gateup_weight_gb_per_token * n_tokens as f64 / (gateup_gpu / 1e3);
            let down_gb_s = down_weight_gb_per_token * n_tokens as f64 / (down_gpu / 1e3);
            println!(
                "{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.1}\t{:.1}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}",
                slot_order.label(),
                n_tokens,
                gateup_gpu,
                down_gpu,
                combined_per_token,
                gateup_gb_s,
                down_gb_s,
                q4_stats.avg_unique_experts,
                q4_stats.avg_max_slots,
                q4_stats.avg_reuse,
                q5_stats.avg_unique_experts,
                q5_stats.avg_max_slots,
                q5_stats.avg_reuse
            );
        }
    }

    Ok(())
}

fn run_moe_gateup_micro(args: MoeGateupMicroArgs) -> Result<()> {
    let MoeGateupMicroArgs {
        model,
        iters,
        warmup,
        tokens,
        route_capture_ctx,
        route_capture_token_pattern,
    } = args;
    if iters == 0 {
        return Err(anyhow!("--iters must be >= 1"));
    }
    if tokens == 0 {
        return Err(anyhow!("--tokens must be >= 1"));
    }

    let ctx = MetalContext::new().context("init MetalContext")?;
    let g = GgufFile::open(&model).with_context(|| format!("open {}", model.display()))?;
    let m = Model::from_gguf(&g).context("parse model")?;
    let mm = MetalModel::load(&ctx, &g, &m).context("metal-load model")?;
    let arch = &mm.arch;
    if arch.kind != qwen_llm::model::ArchKind::Moe {
        return Err(anyhow!("moe-gateup-micro requires an MoE model"));
    }
    let h = arch.hidden_size as usize;
    let f_exp = arch.expert_feed_forward_length as usize;
    let n_expert = arch.expert_count as usize;
    let topk = arch.expert_used_count.min(arch.expert_count) as usize;
    let moe_blocks: Vec<_> = mm
        .blocks
        .iter()
        .filter_map(|b| match b {
            MetalBlock::Gdn(g) => g.ffn_moe.as_ref(),
            MetalBlock::Attn(a) => a.ffn_moe.as_ref(),
        })
        .collect();
    let q4_moes: Vec<_> = moe_blocks
        .iter()
        .copied()
        .filter(|moe| moe.gate_exps.dtype == GgmlType::Q4_K && moe.up_exps.dtype == GgmlType::Q4_K)
        .collect();
    if q4_moes.is_empty() {
        return Err(anyhow!("model has no Q4_K routed gate/up expert banks"));
    }

    let slots = tokens * topk;
    let x = MetalTensor::zeros_f32(&ctx, vec![(tokens * h) as u64])?;
    let topk_idx = MetalTensor::zeros_f32(&ctx, vec![slots as u64])?;
    let inner = MetalTensor::zeros_f32(&ctx, vec![(slots * f_exp) as u64])?;
    unsafe {
        let x_ptr = x.buffer.contents().as_ptr() as *mut f32;
        for i in 0..(tokens * h) {
            *x_ptr.add(i) = ((i % 31) as f32 - 15.0) * 0.01;
        }
        let idx_ptr = topk_idx.buffer.contents().as_ptr() as *mut i32;
        for slot in 0..slots {
            *idx_ptr.add(slot) = ((slot * 17) % n_expert) as i32;
        }
    }

    let mut captured_route_stats = None;
    let captured_inputs: Option<(usize, Vec<(MetalTensor, MetalTensor)>)> =
        if let Some(capture_ctx) = route_capture_ctx {
            let mf = MetalForward::new(&ctx, &mm);
            let mut capture_s = MetalSession::fresh(&ctx, &mm, capture_ctx + tokens + 16)
                .context("route-capture session")?;
            for pos in 0..capture_ctx {
                let _ = mf.single_token(0, pos as u32, &mut capture_s)?;
            }
            let mut all_routes_by_token = Vec::with_capacity(tokens);
            for tok in 0..tokens {
                let token_id =
                    capture_replay_token(route_capture_token_pattern, tok, arch.vocab_size);
                let all_routes = mf.capture_moe_gateup_replay_for_token(
                    token_id,
                    (capture_ctx + tok) as u32,
                    &mut capture_s,
                )?;
                if all_routes.len() != moe_blocks.len() {
                    return Err(anyhow!(
                        "captured {} MoE route rows, expected {}",
                        all_routes.len(),
                        moe_blocks.len()
                    ));
                }
                all_routes_by_token.push(all_routes);
            }
            let q4_indices: Vec<_> = moe_blocks
                .iter()
                .enumerate()
                .filter_map(|(i, moe)| {
                    if moe.gate_exps.dtype == GgmlType::Q4_K && moe.up_exps.dtype == GgmlType::Q4_K
                    {
                        Some(i)
                    } else {
                        None
                    }
                })
                .collect();
            if q4_indices.len() != q4_moes.len() {
                return Err(anyhow!(
                    "captured {} Q4 route rows, expected {}",
                    q4_indices.len(),
                    q4_moes.len()
                ));
            }
            captured_route_stats = Some(summarize_moe_route_batch(
                &all_routes_by_token,
                &q4_indices,
                n_expert,
                topk,
            )?);

            let mut tensors = Vec::with_capacity(q4_indices.len());
            for &moe_i in &q4_indices {
                let hidden_t = MetalTensor::zeros_f32(&ctx, vec![(tokens * h) as u64])?;
                let idx = MetalTensor::zeros_f32(&ctx, vec![slots as u64])?;
                unsafe {
                    let dst = hidden_t.buffer.contents().as_ptr() as *mut f32;
                    let ptr = idx.buffer.contents().as_ptr() as *mut i32;
                    for tok in 0..tokens {
                        let route = &all_routes_by_token[tok][moe_i];
                        if route.topk_idx.len() != topk {
                            return Err(anyhow!(
                                "captured route has {} experts, expected topk={topk}",
                                route.topk_idx.len()
                            ));
                        }
                        if route.hidden.len() != h {
                            return Err(anyhow!(
                                "captured hidden has {} elements, expected h={h}",
                                route.hidden.len()
                            ));
                        }
                        std::ptr::copy_nonoverlapping(route.hidden.as_ptr(), dst.add(tok * h), h);
                        for (slot, &expert) in route.topk_idx.iter().enumerate() {
                            if expert < 0 || expert as usize >= n_expert {
                                return Err(anyhow!(
                                    "captured expert id {expert} outside n_expert={n_expert}"
                                ));
                            }
                            *ptr.add(tok * topk + slot) = expert;
                        }
                    }
                }
                tensors.push((hidden_t, idx));
            }
            Some((capture_ctx, tensors))
        } else {
            None
        };

    let active_weight_bytes: f64 = q4_moes
        .iter()
        .map(|moe| {
            (moe.gate_exps.n_bytes() + moe.up_exps.n_bytes()) as f64 * topk as f64 * tokens as f64
                / n_expert as f64
        })
        .sum();
    let activation_gb = ((tokens * h + slots * f_exp) * std::mem::size_of::<f32>()) as f64
        * q4_moes.len() as f64
        / 1e9;

    let (wall, gpu) = time_gpu_reps(&ctx, warmup, iters, |enc| {
        for (layer_i, moe) in q4_moes.iter().enumerate() {
            let (layer_x, route_idx) = captured_inputs
                .as_ref()
                .map(|(_, idx)| (&idx[layer_i].0, &idx[layer_i].1))
                .unwrap_or((&x, &topk_idx));
            if tokens == 1 {
                encode_moe_swiglu_q4_K_f32(
                    &ctx,
                    enc,
                    &moe.gate_exps,
                    &moe.up_exps,
                    layer_x,
                    route_idx,
                    &inner,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                )?;
            } else {
                encode_moe_swiglu_q4_K_f32_packed_slots(
                    &ctx,
                    enc,
                    &moe.gate_exps,
                    &moe.up_exps,
                    layer_x,
                    route_idx,
                    &inner,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                    tokens,
                )?;
            }
        }
        Ok(())
    })?;

    let weight_gb = active_weight_bytes / 1e9;
    let route_mode = captured_inputs
        .as_ref()
        .map(|(ctx, _)| format!("captured(ctx={ctx},pattern={route_capture_token_pattern:?})"))
        .unwrap_or_else(|| "synthetic".to_string());
    println!(
        "[moe-gateup-micro] model={} mode={} q4_layers={} h={} f_exp={} n_expert={} topk={} tokens={} warmup={} iters={}",
        model.display(),
        route_mode,
        q4_moes.len(),
        h,
        f_exp,
        n_expert,
        topk,
        tokens,
        warmup,
        iters
    );
    if let Some(stats) = captured_route_stats {
        println!(
            "[moe-gateup-route-stats] layers={} tokens={} slots_per_layer={} avg_unique_experts={:.2} avg_max_slots={:.2} avg_reuse={:.2}",
            stats.layers,
            stats.tokens,
            stats.slots_per_layer,
            stats.avg_unique_experts,
            stats.avg_max_slots,
            stats.avg_reuse
        );
    }
    println!("phase\tactive_weight_gb\tactivation_gb\tavg_wall_ms\tavg_gpu_ms\tweight_gb_s");
    println!(
        "q4_gateup_swiglu\t{weight_gb:.4}\t{activation_gb:.4}\t{wall:.4}\t{gpu:.4}\t{:.1}",
        weight_gb / (gpu / 1e3)
    );
    Ok(())
}

fn run_attn_front_micro(args: AttnFrontMicroArgs) -> Result<()> {
    let AttnFrontMicroArgs { model, rows } = args;
    let ctx = MetalContext::new().context("init MetalContext")?;
    let g = GgufFile::open(&model).with_context(|| format!("open {}", model.display()))?;
    let m = Model::from_gguf(&g).context("parse model")?;
    let mm = MetalModel::load(&ctx, &g, &m).context("metal-load model")?;
    let block = mm
        .blocks
        .iter()
        .find_map(|b| match b {
            MetalBlock::Attn(a) => Some(a),
            _ => None,
        })
        .ok_or_else(|| anyhow!("model has no full-attention block"))?;
    let h = mm.arch.hidden_size as usize;
    let head_dim = mm.arch.attn_head_dim as usize;
    let n_q = mm.arch.n_q_heads as usize;
    let n_kv = mm.arch.n_kv_heads as usize;
    let q_dim = n_q * head_dim;
    let kv_dim = n_kv * head_dim;
    let fused_out = 2 * q_dim + 2 * kv_dim;

    let x: Vec<f32> = (0..rows * h)
        .map(|i| ((i % 37) as f32 - 18.0) * 1e-2)
        .collect();
    let as_bytes = |xs: &[f32]| unsafe {
        std::slice::from_raw_parts(xs.as_ptr() as *const u8, std::mem::size_of_val(xs))
    };
    let x_t = MetalTensor::from_bytes(&ctx, as_bytes(&x), vec![(rows * h) as u64], GgmlType::F32)?;

    let read_tensor_bytes = |t: &MetalTensor| -> Vec<u8> {
        let n = t.n_bytes() as usize;
        let mut out = vec![0u8; n];
        unsafe {
            let src = (t.buffer.contents().as_ptr() as *const u8).add(t.offset as usize);
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), n);
        }
        out
    };
    let mut fused_bytes = Vec::new();
    fused_bytes.extend_from_slice(&read_tensor_bytes(&block.q));
    fused_bytes.extend_from_slice(&read_tensor_bytes(&block.k));
    fused_bytes.extend_from_slice(&read_tensor_bytes(&block.v));
    let fused_w = MetalTensor::from_bytes(
        &ctx,
        &fused_bytes,
        vec![h as u64, fused_out as u64],
        GgmlType::Q8_0,
    )?;

    let q_out = MetalTensor::zeros_f32(&ctx, vec![(rows * 2 * q_dim) as u64])?;
    let k_out = MetalTensor::zeros_f32(&ctx, vec![(rows * kv_dim) as u64])?;
    let v_out = MetalTensor::zeros_f32(&ctx, vec![(rows * kv_dim) as u64])?;
    let fused_out_t = MetalTensor::zeros_f32(&ctx, vec![(rows * fused_out) as u64])?;

    let t = Instant::now();
    {
        let cmd = ctx.queue.commandBuffer().context("separate cmd")?;
        let enc = KernelEncoder::begin(&cmd);
        encode_mat_mat_dispatch(&ctx, &enc, &block.q, &x_t, &q_out, h, 2 * q_dim, rows)?;
        encode_mat_mat_dispatch(&ctx, &enc, &block.k, &x_t, &k_out, h, kv_dim, rows)?;
        encode_mat_mat_dispatch(&ctx, &enc, &block.v, &x_t, &v_out, h, kv_dim, rows)?;
        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();
    }
    let separate_ms = t.elapsed().as_secs_f64() * 1e3;

    let t = Instant::now();
    {
        let cmd = ctx.queue.commandBuffer().context("fused cmd")?;
        let enc = KernelEncoder::begin(&cmd);
        encode_mat_mat_dispatch(&ctx, &enc, &fused_w, &x_t, &fused_out_t, h, fused_out, rows)?;
        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();
    }
    let fused_ms = t.elapsed().as_secs_f64() * 1e3;

    let read_back = |t: &MetalTensor| -> Vec<f32> {
        let n = t.n_elements() as usize;
        let mut out = vec![0.0f32; n];
        unsafe {
            let src = (t.buffer.contents().as_ptr() as *const f32).add((t.offset / 4) as usize);
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), n);
        }
        out
    };
    let q = read_back(&q_out);
    let k = read_back(&k_out);
    let v = read_back(&v_out);
    let fused = read_back(&fused_out_t);
    let row_q = 2 * q_dim;
    let row_kv = kv_dim;
    let row_f = fused_out;
    let mut max_abs = 0.0f32;
    for row in 0..rows {
        let q_off = row * row_q;
        let k_off = row * row_kv;
        let f_off = row * row_f;
        for i in 0..row_q {
            max_abs = max_abs.max((q[q_off + i] - fused[f_off + i]).abs());
        }
        for i in 0..row_kv {
            max_abs = max_abs.max((k[k_off + i] - fused[f_off + row_q + i]).abs());
            max_abs = max_abs.max((v[k_off + i] - fused[f_off + row_q + row_kv + i]).abs());
        }
    }
    println!(
        "[attn-front-micro] rows={} separate_ms={:.2} fused_ms={:.2} speedup={:.3} max|Δ|={:.2e}",
        rows,
        separate_ms,
        fused_ms,
        separate_ms / fused_ms,
        max_abs
    );
    Ok(())
}

fn run_attn_prefill_micro(args: AttnPrefillMicroArgs) -> Result<()> {
    let AttnPrefillMicroArgs {
        model,
        base_pos,
        rows,
        nwg,
        qt,
    } = args;
    let ctx = MetalContext::new().context("init MetalContext")?;
    let g = GgufFile::open(&model).with_context(|| format!("open {}", model.display()))?;
    let m = Model::from_gguf(&g).context("parse model")?;
    const HD: usize = 256;
    let N_Q: usize = m.arch.n_q_heads as usize;
    let N_KV: usize = m.arch.n_kv_heads as usize;
    let group_tile = match (N_Q, N_KV) {
        (16, 2) => 2,
        (32, 2) => 4,
        _ => 0,
    };
    if group_tile == 0 {
        anyhow::bail!(
            "attn-prefill-micro unsupported shape n_q={} n_kv={}",
            N_Q,
            N_KV
        );
    }
    if m.arch.attn_head_dim as usize != HD {
        anyhow::bail!(
            "attn-prefill-micro only supports head_dim=256, got {}",
            m.arch.attn_head_dim
        );
    }
    const TILE_C: usize = 64;
    let kv_dim = N_KV * HD;
    let n_pos = base_pos + rows;

    let q_rows: Vec<f32> = (0..rows * N_Q * HD)
        .map(|i| ((i % 31) as f32 - 15.0) * 1e-2)
        .collect();
    let k_f32: Vec<f32> = (0..n_pos * kv_dim)
        .map(|i| ((i % 23) as f32 - 11.0) * 1.5e-2)
        .collect();
    let v_f32: Vec<f32> = (0..n_pos * kv_dim)
        .map(|i| ((i % 17) as f32 - 8.0) * 2e-2)
        .collect();

    let as_bytes = |xs: &[f32]| unsafe {
        std::slice::from_raw_parts(xs.as_ptr() as *const u8, std::mem::size_of_val(xs))
    };

    let q_t = MetalTensor::from_bytes(
        &ctx,
        as_bytes(&q_rows),
        vec![(rows * N_Q * HD) as u64],
        GgmlType::F32,
    )?;
    let k_cache = MetalTensor::zeros_f16(&ctx, vec![(n_pos * kv_dim) as u64])?;
    let v_cache = MetalTensor::zeros_f16(&ctx, vec![(n_pos * kv_dim) as u64])?;
    for (src_f32, dst) in [(&k_f32, &k_cache), (&v_f32, &v_cache)] {
        let src_t = MetalTensor::from_bytes(
            &ctx,
            as_bytes(src_f32.as_slice()),
            vec![src_f32.len() as u64],
            GgmlType::F32,
        )?;
        let cmd = ctx.queue.commandBuffer().context("prep cmd")?;
        let enc = KernelEncoder::begin(&cmd);
        encode_scatter_offset_f32_to_f16(&ctx, &enc, &src_t, dst, 0, src_f32.len())?;
        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();
    }

    let out_baseline = MetalTensor::zeros_f32(&ctx, vec![(rows * N_Q * HD) as u64])?;
    let out_packed = MetalTensor::zeros_f32(&ctx, vec![(rows * N_Q * HD) as u64])?;
    let o_partial_row =
        MetalTensor::zeros_f32(&ctx, vec![(N_KV * nwg * (N_Q / N_KV) * HD) as u64])?;
    let ml_partial_row =
        MetalTensor::zeros_f32(&ctx, vec![(N_KV * nwg * (N_Q / N_KV) * 2) as u64])?;
    let o_partial_packed =
        MetalTensor::zeros_f32(&ctx, vec![(rows * N_KV * nwg * (N_Q / N_KV) * HD) as u64])?;
    let ml_partial_packed =
        MetalTensor::zeros_f32(&ctx, vec![(rows * N_KV * nwg * (N_Q / N_KV) * 2) as u64])?;

    let run_baseline = || -> Result<()> {
        let cmd = ctx.queue.commandBuffer().context("baseline cmd")?;
        let enc = KernelEncoder::begin(&cmd);
        with_attn_v4_group_tile_override(group_tile, || {
            for row in 0..rows {
                let q_row = q_t.view_subrange((row * N_Q * HD) as u64, vec![(N_Q * HD) as u64]);
                let out_row =
                    out_baseline.view_subrange((row * N_Q * HD) as u64, vec![(N_Q * HD) as u64]);
                encode_attn_decode_v4_f32(
                    &ctx,
                    &enc,
                    &q_row,
                    &k_cache,
                    &v_cache,
                    &o_partial_row,
                    &ml_partial_row,
                    &out_row,
                    N_Q,
                    N_KV,
                    HD,
                    base_pos + row + 1,
                    nwg,
                    TILE_C,
                )
                .expect("decode-shaped oracle attention");
            }
        });
        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();
        Ok(())
    };
    run_baseline()?;
    let t = Instant::now();
    run_baseline()?;
    let baseline_wall = t.elapsed().as_secs_f64() * 1e3;

    let run_packed = || -> Result<()> {
        let cmd = ctx.queue.commandBuffer().context("packed cmd")?;
        let enc = KernelEncoder::begin(&cmd);
        match (N_Q, N_KV, qt) {
            (16, 2, 2) => encode_attn_prefill_v4_g8_t2_q2_c64_f32(
                &ctx,
                &enc,
                &q_t,
                &k_cache,
                &v_cache,
                &o_partial_packed,
                &ml_partial_packed,
                &out_packed,
                rows,
                base_pos,
                nwg,
            )?,
            (16, 2, 4) => encode_attn_prefill_v4_g8_t2_q4_c64_f32(
                &ctx,
                &enc,
                &q_t,
                &k_cache,
                &v_cache,
                &o_partial_packed,
                &ml_partial_packed,
                &out_packed,
                rows,
                base_pos,
                nwg,
            )?,
            (32, 2, 2) => encode_attn_prefill_v4_g16_t4_q2_c64_f32(
                &ctx,
                &enc,
                &q_t,
                &k_cache,
                &v_cache,
                &o_partial_packed,
                &ml_partial_packed,
                &out_packed,
                rows,
                base_pos,
                nwg,
            )?,
            (32, 2, 4) => encode_attn_prefill_v4_g16_t4_q4_c64_f32(
                &ctx,
                &enc,
                &q_t,
                &k_cache,
                &v_cache,
                &o_partial_packed,
                &ml_partial_packed,
                &out_packed,
                rows,
                base_pos,
                nwg,
            )?,
            _ => anyhow::bail!(
                "attn-prefill-micro unsupported shape n_q={} n_kv={} qt={}",
                N_Q,
                N_KV,
                qt
            ),
        }
        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();
        Ok(())
    };
    run_packed()?;
    let t = Instant::now();
    run_packed()?;
    let packed_wall = t.elapsed().as_secs_f64() * 1e3;

    let read_back = |t: &MetalTensor| -> Vec<f32> {
        let n = t.n_elements() as usize;
        let mut out = vec![0.0f32; n];
        unsafe {
            let src = (t.buffer.contents().as_ptr() as *const f32).add((t.offset / 4) as usize);
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), n);
        }
        out
    };
    let baseline = read_back(&out_baseline);
    let packed = read_back(&out_packed);
    let max_abs = packed
        .iter()
        .zip(baseline.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let dot: f64 = packed
        .iter()
        .zip(baseline.iter())
        .map(|(a, b)| (*a as f64) * (*b as f64))
        .sum();
    let na: f64 = packed
        .iter()
        .map(|x| (*x as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let nb: f64 = baseline
        .iter()
        .map(|x| (*x as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let cos = dot / (na * nb);
    println!(
        "[attn-prefill-micro] base_pos={} rows={} nwg={} qt={} decode_loop_ms={:.2} packed_ms={:.2} speedup={:.3} max|Δ|={:.2e} cos={:.6}",
        base_pos,
        rows,
        nwg,
        qt,
        baseline_wall,
        packed_wall,
        baseline_wall / packed_wall,
        max_abs,
        cos
    );
    Ok(())
}

fn run_attn_layer_micro(args: AttnLayerMicroArgs) -> Result<()> {
    let AttnLayerMicroArgs {
        model,
        base_pos,
        rows,
        nwg,
        qt,
    } = args;
    let ctx = MetalContext::new().context("init MetalContext")?;
    let g = GgufFile::open(&model).with_context(|| format!("open {}", model.display()))?;
    let m = Model::from_gguf(&g).context("parse model")?;
    let mm = MetalModel::load(&ctx, &g, &m).context("metal-load model")?;
    let block = mm
        .blocks
        .iter()
        .find_map(|b| match b {
            MetalBlock::Attn(a) => Some(a),
            _ => None,
        })
        .ok_or_else(|| anyhow!("model has no full-attention block"))?;
    const HD: usize = 256;
    const TILE_C: usize = 64;
    let n_q = mm.arch.n_q_heads as usize;
    let n_kv = mm.arch.n_kv_heads as usize;
    let h = mm.arch.hidden_size as usize;
    let q_dim = n_q * HD;
    let kv_dim = n_kv * HD;
    let group = n_q / n_kv;
    let group_tile = match (n_q, n_kv) {
        (16, 2) => 2,
        (32, 2) => 4,
        _ => anyhow::bail!(
            "attn-layer-micro unsupported shape n_q={} n_kv={}",
            n_q,
            n_kv
        ),
    };
    if mm.arch.attn_head_dim as usize != HD {
        anyhow::bail!(
            "attn-layer-micro only supports head_dim=256, got {}",
            mm.arch.attn_head_dim
        );
    }
    if !matches!(qt, 2 | 4) {
        anyhow::bail!("attn-layer-micro only supports qt=2 or 4, got {qt}");
    }
    let n_rot = (HD as f32 * mm.arch.partial_rotary_factor) as usize;
    let n_pos = base_pos + rows;

    let h_rows: Vec<f32> = (0..rows * h)
        .map(|i| ((i % 37) as f32 - 18.0) * 1e-2)
        .collect();
    let prefix_k: Vec<f32> = (0..n_pos * kv_dim)
        .map(|i| ((i % 23) as f32 - 11.0) * 1.5e-2)
        .collect();
    let prefix_v: Vec<f32> = (0..n_pos * kv_dim)
        .map(|i| ((i % 17) as f32 - 8.0) * 2e-2)
        .collect();
    let as_bytes = |xs: &[f32]| unsafe {
        std::slice::from_raw_parts(xs.as_ptr() as *const u8, std::mem::size_of_val(xs))
    };
    let h_t = MetalTensor::from_bytes(
        &ctx,
        as_bytes(&h_rows),
        vec![(rows * h) as u64],
        GgmlType::F32,
    )?;

    #[derive(Clone)]
    struct StackScratch {
        q_full: MetalTensor,
        q: MetalTensor,
        gate: MetalTensor,
        q_normed: MetalTensor,
        k_now: MetalTensor,
        v_now: MetalTensor,
        k_normed: MetalTensor,
        attn_o: MetalTensor,
        mixer_out: MetalTensor,
        o_partial: MetalTensor,
        ml_partial: MetalTensor,
    }

    let make_baseline_scratch = || -> Result<StackScratch> {
        Ok(StackScratch {
            q_full: MetalTensor::zeros_f32(&ctx, vec![(rows * 2 * q_dim) as u64])?,
            q: MetalTensor::zeros_f32(&ctx, vec![(rows * q_dim) as u64])?,
            gate: MetalTensor::zeros_f32(&ctx, vec![(rows * q_dim) as u64])?,
            q_normed: MetalTensor::zeros_f32(&ctx, vec![(rows * q_dim) as u64])?,
            k_now: MetalTensor::zeros_f32(&ctx, vec![(rows * kv_dim) as u64])?,
            v_now: MetalTensor::zeros_f32(&ctx, vec![(rows * kv_dim) as u64])?,
            k_normed: MetalTensor::zeros_f32(&ctx, vec![(rows * kv_dim) as u64])?,
            attn_o: MetalTensor::zeros_f32(&ctx, vec![(rows * q_dim) as u64])?,
            mixer_out: MetalTensor::zeros_f32(&ctx, vec![(rows * h) as u64])?,
            o_partial: MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * group * HD) as u64])?,
            ml_partial: MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * group * 2) as u64])?,
        })
    };
    let make_packed_scratch = || -> Result<StackScratch> {
        Ok(StackScratch {
            q_full: MetalTensor::zeros_f32(&ctx, vec![(rows * 2 * q_dim) as u64])?,
            q: MetalTensor::zeros_f32(&ctx, vec![(rows * q_dim) as u64])?,
            gate: MetalTensor::zeros_f32(&ctx, vec![(rows * q_dim) as u64])?,
            q_normed: MetalTensor::zeros_f32(&ctx, vec![(rows * q_dim) as u64])?,
            k_now: MetalTensor::zeros_f32(&ctx, vec![(rows * kv_dim) as u64])?,
            v_now: MetalTensor::zeros_f32(&ctx, vec![(rows * kv_dim) as u64])?,
            k_normed: MetalTensor::zeros_f32(&ctx, vec![(rows * kv_dim) as u64])?,
            attn_o: MetalTensor::zeros_f32(&ctx, vec![(rows * q_dim) as u64])?,
            mixer_out: MetalTensor::zeros_f32(&ctx, vec![(rows * h) as u64])?,
            o_partial: MetalTensor::zeros_f32(&ctx, vec![(rows * n_kv * nwg * group * HD) as u64])?,
            ml_partial: MetalTensor::zeros_f32(&ctx, vec![(rows * n_kv * nwg * group * 2) as u64])?,
        })
    };

    let baseline = make_baseline_scratch()?;
    let packed = make_packed_scratch()?;
    let baseline_k_cache = MetalTensor::zeros_f16(&ctx, vec![(n_pos * kv_dim) as u64])?;
    let baseline_v_cache = MetalTensor::zeros_f16(&ctx, vec![(n_pos * kv_dim) as u64])?;
    let packed_k_cache = MetalTensor::zeros_f16(&ctx, vec![(n_pos * kv_dim) as u64])?;
    let packed_v_cache = MetalTensor::zeros_f16(&ctx, vec![(n_pos * kv_dim) as u64])?;

    let seed_cache = |dst: &MetalTensor, src_f32: &[f32]| -> Result<()> {
        let src_t = MetalTensor::from_bytes(
            &ctx,
            as_bytes(src_f32),
            vec![src_f32.len() as u64],
            GgmlType::F32,
        )?;
        let cmd = ctx.queue.commandBuffer().context("seed cache cmd")?;
        let enc = KernelEncoder::begin(&cmd);
        encode_scatter_offset_f32_to_f16(&ctx, &enc, &src_t, dst, 0, src_f32.len())?;
        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();
        Ok(())
    };
    for dst in [&baseline_k_cache, &packed_k_cache] {
        seed_cache(dst, &prefix_k)?;
    }
    for dst in [&baseline_v_cache, &packed_v_cache] {
        seed_cache(dst, &prefix_v)?;
    }

    let run_front = |enc: &KernelEncoder,
                     scratch: &StackScratch,
                     k_cache: &MetalTensor,
                     v_cache: &MetalTensor|
     -> Result<()> {
        encode_mat_mat_dispatch(
            &ctx,
            enc,
            &block.q,
            &h_t,
            &scratch.q_full,
            h,
            2 * q_dim,
            rows,
        )?;
        encode_mat_mat_dispatch(&ctx, enc, &block.k, &h_t, &scratch.k_now, h, kv_dim, rows)?;
        encode_mat_mat_dispatch(&ctx, enc, &block.v, &h_t, &scratch.v_now, h, kv_dim, rows)?;
        encode_split_q_gate_f32(
            &ctx,
            enc,
            &scratch.q_full,
            &scratch.q,
            &scratch.gate,
            rows * n_q,
            HD,
        )?;
        encode_rms_norm_batched_f32(
            &ctx,
            enc,
            &scratch.q,
            &block.q_norm,
            &scratch.q_normed,
            rows * n_q,
            HD,
            RMS_EPS,
        )?;
        encode_rms_norm_batched_f32(
            &ctx,
            enc,
            &scratch.k_now,
            &block.k_norm,
            &scratch.k_normed,
            rows * n_kv,
            HD,
            RMS_EPS,
        )?;
        encode_rope_neox_f32_packed_consecutive(
            &ctx,
            enc,
            &scratch.q_normed,
            rows,
            n_q,
            HD,
            n_rot,
            base_pos as u32,
            mm.arch.rope_theta,
        )?;
        encode_rope_neox_f32_packed_consecutive(
            &ctx,
            enc,
            &scratch.k_normed,
            rows,
            n_kv,
            HD,
            n_rot,
            base_pos as u32,
            mm.arch.rope_theta,
        )?;
        encode_scatter_offset_f32_to_f16_kv(
            &ctx,
            enc,
            &scratch.k_normed,
            &scratch.v_now,
            k_cache,
            v_cache,
            base_pos * kv_dim,
            rows * kv_dim,
        )?;
        Ok(())
    };

    let run_tail = |enc: &KernelEncoder, scratch: &StackScratch| -> Result<()> {
        encode_sigmoid_f32(&ctx, enc, &scratch.gate, &scratch.q)?;
        encode_mul_f32(&ctx, enc, &scratch.attn_o, &scratch.q, &scratch.attn_o)?;
        encode_mat_mat_dispatch(
            &ctx,
            enc,
            &block.o,
            &scratch.attn_o,
            &scratch.mixer_out,
            q_dim,
            h,
            rows,
        )?;
        Ok(())
    };

    let run_baseline = || -> Result<()> {
        let cmd = ctx.queue.commandBuffer().context("baseline layer cmd")?;
        let enc = KernelEncoder::begin(&cmd);
        run_front(&enc, &baseline, &baseline_k_cache, &baseline_v_cache)?;
        with_attn_v4_group_tile_override(group_tile, || {
            for row in 0..rows {
                let q_row = baseline
                    .q_normed
                    .view_subrange((row * q_dim) as u64, vec![q_dim as u64]);
                let out_row = baseline
                    .attn_o
                    .view_subrange((row * q_dim) as u64, vec![q_dim as u64]);
                encode_attn_decode_v4_f32(
                    &ctx,
                    &enc,
                    &q_row,
                    &baseline_k_cache,
                    &baseline_v_cache,
                    &baseline.o_partial,
                    &baseline.ml_partial,
                    &out_row,
                    n_q,
                    n_kv,
                    HD,
                    base_pos + row + 1,
                    nwg,
                    TILE_C,
                )
                .expect("baseline decode attention");
            }
        });
        run_tail(&enc, &baseline)?;
        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();
        Ok(())
    };

    let run_packed = || -> Result<()> {
        let cmd = ctx.queue.commandBuffer().context("packed layer cmd")?;
        let enc = KernelEncoder::begin(&cmd);
        run_front(&enc, &packed, &packed_k_cache, &packed_v_cache)?;
        match (n_q, n_kv, qt) {
            (16, 2, 2) => encode_attn_prefill_v4_g8_t2_q2_c64_f32(
                &ctx,
                &enc,
                &packed.q_normed,
                &packed_k_cache,
                &packed_v_cache,
                &packed.o_partial,
                &packed.ml_partial,
                &packed.attn_o,
                rows,
                base_pos,
                nwg,
            )?,
            (16, 2, 4) => encode_attn_prefill_v4_g8_t2_q4_c64_f32(
                &ctx,
                &enc,
                &packed.q_normed,
                &packed_k_cache,
                &packed_v_cache,
                &packed.o_partial,
                &packed.ml_partial,
                &packed.attn_o,
                rows,
                base_pos,
                nwg,
            )?,
            (32, 2, 2) => encode_attn_prefill_v4_g16_t4_q2_c64_f32(
                &ctx,
                &enc,
                &packed.q_normed,
                &packed_k_cache,
                &packed_v_cache,
                &packed.o_partial,
                &packed.ml_partial,
                &packed.attn_o,
                rows,
                base_pos,
                nwg,
            )?,
            (32, 2, 4) => encode_attn_prefill_v4_g16_t4_q4_c64_f32(
                &ctx,
                &enc,
                &packed.q_normed,
                &packed_k_cache,
                &packed_v_cache,
                &packed.o_partial,
                &packed.ml_partial,
                &packed.attn_o,
                rows,
                base_pos,
                nwg,
            )?,
            _ => anyhow::bail!(
                "attn-layer-micro unsupported shape n_q={} n_kv={} qt={}",
                n_q,
                n_kv,
                qt
            ),
        }
        run_tail(&enc, &packed)?;
        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();
        Ok(())
    };

    run_baseline()?;
    let t = Instant::now();
    run_baseline()?;
    let baseline_wall = t.elapsed().as_secs_f64() * 1e3;
    run_packed()?;
    let t = Instant::now();
    run_packed()?;
    let packed_wall = t.elapsed().as_secs_f64() * 1e3;

    let read_back = |t: &MetalTensor| -> Vec<f32> {
        let n = t.n_elements() as usize;
        let mut out = vec![0.0f32; n];
        unsafe {
            let src = (t.buffer.contents().as_ptr() as *const f32).add((t.offset / 4) as usize);
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), n);
        }
        out
    };
    let baseline_out = read_back(&baseline.mixer_out);
    let packed_out = read_back(&packed.mixer_out);
    let max_abs = packed_out
        .iter()
        .zip(baseline_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let dot: f64 = packed_out
        .iter()
        .zip(baseline_out.iter())
        .map(|(a, b)| (*a as f64) * (*b as f64))
        .sum();
    let na: f64 = packed_out
        .iter()
        .map(|x| (*x as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let nb: f64 = baseline_out
        .iter()
        .map(|x| (*x as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let cos = dot / (na * nb);
    println!(
        "[attn-layer-micro] base_pos={} rows={} nwg={} qt={} baseline_ms={:.2} packed_ms={:.2} speedup={:.3} max|Δ|={:.2e} cos={:.6}",
        base_pos,
        rows,
        nwg,
        qt,
        baseline_wall,
        packed_wall,
        baseline_wall / packed_wall,
        max_abs,
        cos,
    );
    Ok(())
}

fn run_tok(args: TokArgs) -> Result<()> {
    let TokArgs {
        model,
        prompt,
        file,
        messages,
        messages_max,
        messages_preserve_thinking,
        messages_strip_thinking,
        messages_no_generation_prompt,
        iters,
        add_special,
    } = args;
    if iters == 0 {
        anyhow::bail!("--iters must be > 0");
    }
    let (source, text) = match (prompt, file, messages) {
        (Some(prompt), None, None) => ("inline".to_string(), prompt),
        (None, Some(path), None) => {
            let text = std::fs::read_to_string(&path)
                .with_context(|| format!("read {}", path.display()))?;
            (format!("file:{}", path.display()), text)
        }
        (None, None, Some(path)) => (
            format!("messages:{}", path.display()),
            load_messages_prompt(
                &path,
                messages_max,
                messages_thinking_mode(messages_preserve_thinking, messages_strip_thinking),
                !messages_no_generation_prompt,
            )?,
        ),
        (None, None, None) => ("default".to_string(), default_tok_prompt()),
        _ => unreachable!("clap conflicts_with"),
    };

    let t0 = Instant::now();
    let ffi = LlamaCppTokenizer::open(&model).context("open llama.cpp FFI tokenizer")?;
    let ffi_load = t0.elapsed();

    let t0 = Instant::now();
    let gguf = GgufFile::open(&model).with_context(|| format!("open {}", model.display()))?;
    let native = NativeTokenizer::from_gguf(&gguf).context("open native GGUF tokenizer")?;
    let native_load = t0.elapsed();

    let ffi_ids = ffi.encode(&text, add_special).context("ffi encode")?;
    let native_ids = native.encode(&text, add_special).context("native encode")?;
    if ffi_ids != native_ids {
        let first = ffi_ids
            .iter()
            .zip(&native_ids)
            .position(|(a, b)| a != b)
            .unwrap_or_else(|| ffi_ids.len().min(native_ids.len()));
        anyhow::bail!(
            "native/ffi encode mismatch at token {first}: ffi_len={} native_len={} ffi={:?} native={:?}",
            ffi_ids.len(),
            native_ids.len(),
            ffi_ids.get(first),
            native_ids.get(first)
        );
    }
    let ffi_text = ffi.try_decode(&ffi_ids).context("ffi decode")?;
    let native_text = native.try_decode(&native_ids).context("native decode")?;
    if ffi_text != native_text {
        anyhow::bail!(
            "native/ffi decode mismatch: ffi_len={} native_len={}",
            ffi_text.len(),
            native_text.len()
        );
    }

    let t0 = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(ffi.encode(&text, add_special).context("ffi encode timed")?);
    }
    let ffi_encode = t0.elapsed();

    let t0 = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(
            native
                .encode(&text, add_special)
                .context("native encode timed")?,
        );
    }
    let native_encode = t0.elapsed();

    let t0 = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(ffi.try_decode(&ffi_ids).context("ffi decode timed")?);
    }
    let ffi_decode = t0.elapsed();

    let t0 = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(
            native
                .try_decode(&native_ids)
                .context("native decode timed")?,
        );
    }
    let native_decode = t0.elapsed();

    let n_tokens = ffi_ids.len() * iters;
    println!("[tok] model={}", model.display());
    println!(
        "[tok] source={} chars={} tokens={} iters={} add_special={}",
        source,
        text.len(),
        ffi_ids.len(),
        iters,
        add_special
    );
    println!(
        "[tok] load_ms: ffi={:.3} native={:.3}",
        ffi_load.as_secs_f64() * 1000.0,
        native_load.as_secs_f64() * 1000.0
    );
    print_tok_rate("ffi encode", ffi_encode, n_tokens);
    print_tok_rate("native encode", native_encode, n_tokens);
    print_tok_rate("ffi decode", ffi_decode, n_tokens);
    print_tok_rate("native decode", native_decode, n_tokens);
    Ok(())
}

#[derive(Debug, serde::Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Clone, Copy, Debug)]
enum MessagesThinkingMode {
    Auto,
    Preserve,
    Strip,
}

fn messages_thinking_mode(preserve: bool, strip: bool) -> MessagesThinkingMode {
    if preserve {
        MessagesThinkingMode::Preserve
    } else if strip {
        MessagesThinkingMode::Strip
    } else {
        MessagesThinkingMode::Auto
    }
}

fn load_messages_prompt(
    path: &PathBuf,
    max_messages: Option<usize>,
    thinking_mode: MessagesThinkingMode,
    append_generation_prompt: bool,
) -> Result<String> {
    let raw = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let value: serde_json::Value = serde_json::from_str(&raw)
        .with_context(|| format!("parse messages input {}", path.display()))?;
    let (mut messages, meta) = parse_messages_input(value)?;
    if let Some(max) = max_messages {
        messages.truncate(max);
    }
    if messages.is_empty() {
        anyhow::bail!("messages input {} contains no messages", path.display());
    }
    let preserve_thinking = match thinking_mode {
        MessagesThinkingMode::Preserve => true,
        MessagesThinkingMode::Strip => false,
        MessagesThinkingMode::Auto => messages_auto_preserve_thinking(&meta),
    };
    Ok(render_qwen_messages_prompt(
        &messages,
        preserve_thinking,
        append_generation_prompt,
    ))
}

fn messages_auto_preserve_thinking(meta: &serde_json::Value) -> bool {
    if meta
        .get("preserve_thinking")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        return true;
    }
    meta.get("model")
        .and_then(|v| v.as_str())
        .map(|s| s.to_ascii_lowercase().contains("qwen3.6"))
        .unwrap_or(false)
}

fn parse_messages_input(value: serde_json::Value) -> Result<(Vec<ChatMessage>, serde_json::Value)> {
    match value {
        serde_json::Value::Array(_) => {
            let messages: Vec<ChatMessage> =
                serde_json::from_value(value).context("parse bare messages array")?;
            Ok((messages, serde_json::Value::Null))
        }
        serde_json::Value::Object(mut obj) => {
            let messages_value = obj.remove("messages").ok_or_else(|| {
                anyhow!("wrapped messages input must contain a top-level `messages` array")
            })?;
            let messages: Vec<ChatMessage> =
                serde_json::from_value(messages_value).context("parse wrapped messages array")?;

            let mut merged = serde_json::Map::new();
            if let Some(meta_value) = obj.remove("meta") {
                match meta_value {
                    serde_json::Value::Object(map) => merged.extend(map),
                    serde_json::Value::Null => {}
                    other => {
                        merged.insert("meta".into(), other);
                    }
                }
            }
            for (key, value) in obj {
                merged.insert(key, value);
            }
            Ok((messages, serde_json::Value::Object(merged)))
        }
        other => Err(anyhow!(
            "messages input must be a message array or wrapped object, got {other}"
        )),
    }
}

fn render_qwen_messages_prompt(
    messages: &[ChatMessage],
    preserve_thinking: bool,
    append_generation_prompt: bool,
) -> String {
    let mut out = String::new();
    for msg in messages {
        out.push_str("<|im_start|>");
        out.push_str(&msg.role);
        out.push('\n');
        if msg.role == "assistant" && !preserve_thinking {
            out.push_str(&strip_think(&msg.content));
        } else {
            out.push_str(&msg.content);
        }
        out.push_str("<|im_end|>\n");
    }
    if append_generation_prompt {
        out.push_str("<|im_start|>assistant\n");
    }
    out
}

fn strip_think(text: &str) -> String {
    let trimmed = text.trim_start();
    if let Some(rest) = trimmed.strip_prefix("<think>") {
        if let Some((_, tail)) = rest.split_once("</think>") {
            return tail.trim().to_string();
        }
    }
    text.to_string()
}

fn default_tok_prompt() -> String {
    "<|im_start|>user\nHello, world!\n\n```rust\nfn main() { println!(\"hi\"); }\n```\n数字123 combining e\u{301} emoji🙂\u{fe0f}\n<|im_end|>"
        .to_string()
}

fn print_tok_rate(label: &str, elapsed: Duration, n_tokens: usize) {
    let secs = elapsed.as_secs_f64();
    let tok_s = n_tokens as f64 / secs.max(f64::MIN_POSITIVE);
    println!(
        "[tok] {label}: {:.3} ms total, {:.0} tok/s",
        secs * 1000.0,
        tok_s
    );
}

#[cfg(test)]
mod tok_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn auto_preserves_thinking_only_for_qwen36() {
        assert!(messages_auto_preserve_thinking(
            &json!({ "model": "/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf" })
        ));
        assert!(messages_auto_preserve_thinking(
            &json!({ "preserve_thinking": true, "model": "anything" })
        ));
        assert!(!messages_auto_preserve_thinking(
            &json!({ "model": "/Users/tito/models/Qwen3.5-27B-Q4_K_M.gguf" })
        ));
        assert!(!messages_auto_preserve_thinking(&serde_json::Value::Null));
    }

    #[test]
    fn strip_think_only_strips_leading_qwen_block() {
        assert_eq!(strip_think("<think>hidden</think>shown"), "shown");
        assert_eq!(strip_think("plain text"), "plain text");
        assert_eq!(strip_think("  plain text  "), "  plain text  ");
        assert_eq!(
            strip_think("prefix </think> shown"),
            "prefix </think> shown"
        );
    }

    #[test]
    fn render_messages_prompt_respects_thinking_mode() {
        let messages = vec![
            ChatMessage {
                role: "user".into(),
                content: "hi".into(),
            },
            ChatMessage {
                role: "assistant".into(),
                content: "<think>hidden</think>shown".into(),
            },
        ];
        let stripped = render_qwen_messages_prompt(&messages, false, true);
        let preserved = render_qwen_messages_prompt(&messages, true, true);
        assert!(stripped.contains("shown<|im_end|>"));
        assert!(!stripped.contains("hidden"));
        assert!(preserved.contains("<think>hidden</think>shown"));
        assert!(preserved.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn parse_messages_input_accepts_top_level_metadata() {
        let value = json!({
            "model": "/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf",
            "preserve_thinking": true,
            "messages": [
                {"role": "user", "content": "hi"}
            ]
        });
        let (messages, meta) = parse_messages_input(value).expect("parse messages");
        assert_eq!(messages.len(), 1);
        assert_eq!(
            meta.get("model").and_then(|v| v.as_str()),
            Some("/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf")
        );
        assert_eq!(
            meta.get("preserve_thinking").and_then(|v| v.as_bool()),
            Some(true)
        );
    }

    #[test]
    fn mtp_token_ids_require_output() {
        let missing_output = Args::try_parse_from([
            "qwen-bench",
            "mtp",
            "-m",
            "model.gguf",
            "--include-token-ids",
        ]);
        assert!(missing_output.is_err());

        let parsed = Args::try_parse_from([
            "qwen-bench",
            "mtp",
            "-m",
            "model.gguf",
            "--output",
            "fixture.json",
            "--include-token-ids",
        ])
        .expect("parse token fixture arguments");
        let Cmd::Mtp(args) = parsed.cmd else {
            panic!("expected mtp command");
        };
        assert!(args.include_token_ids);
        assert_eq!(args.output, Some(PathBuf::from("fixture.json")));
    }

    #[test]
    fn pld_terminal_window_counts_target_transitions() {
        let drafts = [11, 12, 13, 14, 15, 16, 17];
        let count = |emitted, stop| {
            pld_terminal_draft_count(&drafts, emitted, 128, stop).map(|window| window.count)
        };
        assert_eq!(count(1, &[99]), None);
        assert_eq!(count(121, &[99]), Some(7));
        assert_eq!(count(125, &[99]), Some(3));
        assert_eq!(count(1, &[14]), Some(4));

        for remaining in 1..=DRAFT_TOKENS {
            let window = pld_terminal_draft_count(&drafts, 128 - remaining, 128, &[99])
                .expect("output-limit window");
            assert_eq!(window.count, remaining);
            assert_eq!(window.cause, PldTerminalCause::OutputLimit);
        }
        for stop_index in 0..DRAFT_TOKENS {
            let window = pld_terminal_draft_count(&drafts, 1, 128, &[drafts[stop_index]])
                .expect("stop-token window");
            assert_eq!(window.count, stop_index + 1);
            assert_eq!(window.cause, PldTerminalCause::StopToken);
        }
        let output_first = pld_terminal_draft_count(&drafts, 125, 128, &[16])
            .expect("output limit precedes proposed stop");
        assert_eq!(output_first.count, 3);
        assert_eq!(output_first.cause, PldTerminalCause::OutputLimit);
    }
}

#[derive(Clone, Copy, Debug)]
struct MtpTargetStateAudit {
    resume_audit_pass: bool,
    kv_position_equal: bool,
    kv_payload_exact: bool,
    kv_payload_max_abs: f32,
    kv_payload_cosine: f64,
    reference_final_position: Option<usize>,
    candidate_final_position: Option<usize>,
    gdn_state_max_abs: f32,
    gdn_conv_max_abs: f32,
    continuation_argmax_equal: bool,
    continuation_token: i32,
    continuation_logits_max_abs: f32,
    continuation_logits_cosine: f64,
}

fn max_abs_f32_tensor_pairs(reference: &[MetalTensor], candidate: &[MetalTensor]) -> Result<f32> {
    anyhow::ensure!(
        reference.len() == candidate.len(),
        "state tensor count mismatch"
    );
    let mut max_abs = 0.0f32;
    for (a, b) in reference.iter().zip(candidate) {
        anyhow::ensure!(
            a.dtype == GgmlType::F32
                && b.dtype == GgmlType::F32
                && a.shape == b.shape
                && a.n_bytes() == b.n_bytes(),
            "state tensor shape or dtype mismatch"
        );
        unsafe {
            let pa =
                (a.buffer.contents().as_ptr() as *const u8).add(a.offset as usize) as *const f32;
            let pb =
                (b.buffer.contents().as_ptr() as *const u8).add(b.offset as usize) as *const f32;
            for i in 0..a.n_elements() as usize {
                let delta = (*pa.add(i) - *pb.add(i)).abs();
                if !delta.is_finite() {
                    return Ok(f32::INFINITY);
                }
                max_abs = max_abs.max(delta);
            }
        }
    }
    Ok(max_abs)
}

fn kv_bytes_metrics(reference: &[u8], candidate: &[u8], dtype: GgmlType) -> Result<(f32, f64)> {
    anyhow::ensure!(
        reference.len() == candidate.len(),
        "KV byte length mismatch"
    );
    let values: Box<dyn Iterator<Item = (f32, f32)> + '_> = match dtype {
        GgmlType::F16 => Box::new(
            reference
                .chunks_exact(2)
                .zip(candidate.chunks_exact(2))
                .map(|(a, b)| {
                    let a = half::f16::from_bits(u16::from_le_bytes([a[0], a[1]])).to_f32();
                    let b = half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32();
                    (a, b)
                }),
        ),
        GgmlType::F32 => Box::new(
            reference
                .chunks_exact(4)
                .zip(candidate.chunks_exact(4))
                .map(|(a, b)| {
                    let a = f32::from_le_bytes([a[0], a[1], a[2], a[3]]);
                    let b = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                    (a, b)
                }),
        ),
        other => anyhow::bail!("unsupported KV audit dtype {other:?}"),
    };
    let (mut max_abs, mut dot, mut reference_norm, mut candidate_norm) =
        (0.0f32, 0.0f64, 0.0f64, 0.0f64);
    for (a, b) in values {
        max_abs = max_abs.max((a - b).abs());
        dot += a as f64 * b as f64;
        reference_norm += (a as f64).powi(2);
        candidate_norm += (b as f64).powi(2);
    }
    let cosine = dot / (reference_norm.sqrt() * candidate_norm.sqrt() + f64::MIN_POSITIVE);
    Ok((max_abs, cosine))
}

fn audit_mtp_target_state(
    forward: &MetalForward<'_>,
    reference: &mut MetalSession,
    candidate: &mut MetalSession,
    expected_position: usize,
    pending_terminal_token: i32,
) -> Result<MtpTargetStateAudit> {
    let reference_position_valid = reference
        .kv_n_pos
        .iter()
        .all(|&position| position == expected_position);
    let candidate_position_valid = candidate
        .kv_n_pos
        .iter()
        .all(|&position| position == expected_position);
    let kv_position_equal = reference_position_valid
        && candidate_position_valid
        && reference.kv_n_pos == candidate.kv_n_pos;
    let reference_final_position = reference.kv_n_pos.first().copied();
    let candidate_final_position = candidate.kv_n_pos.first().copied();
    let gdn_state_max_abs = max_abs_f32_tensor_pairs(&reference.gdn_state, &candidate.gdn_state)?;
    let gdn_conv_max_abs = max_abs_f32_tensor_pairs(&reference.gdn_conv, &candidate.gdn_conv)?;
    let snapshot_identity = reference.snapshot_identity(0, 0);
    let reference_snapshot =
        reference.snapshot(snapshot_identity.clone(), vec![0; expected_position], None);
    let candidate_snapshot =
        candidate.snapshot(snapshot_identity, vec![0; expected_position], None);
    let kv_payload_exact = reference_snapshot.kv_k_arena == candidate_snapshot.kv_k_arena
        && reference_snapshot.kv_v_arena == candidate_snapshot.kv_v_arena;
    let kv_dtype = reference
        .kv_k
        .first()
        .map(|tensor| tensor.dtype)
        .context("target state has no KV layers")?;
    anyhow::ensure!(
        candidate.kv_k.first().map(|tensor| tensor.dtype) == Some(kv_dtype),
        "candidate KV dtype mismatch"
    );
    let (kv_k_max_abs, kv_k_cosine) = kv_bytes_metrics(
        &reference_snapshot.kv_k_arena,
        &candidate_snapshot.kv_k_arena,
        kv_dtype,
    )?;
    let (kv_v_max_abs, kv_v_cosine) = kv_bytes_metrics(
        &reference_snapshot.kv_v_arena,
        &candidate_snapshot.kv_v_arena,
        kv_dtype,
    )?;
    let kv_payload_max_abs = kv_k_max_abs.max(kv_v_max_abs);
    let kv_payload_cosine = kv_k_cosine.min(kv_v_cosine);
    let reference_logits =
        forward.single_token(pending_terminal_token, expected_position as u32, reference)?;
    let candidate_logits =
        forward.single_token(pending_terminal_token, expected_position as u32, candidate)?;
    anyhow::ensure!(
        reference_logits.len() == candidate_logits.len(),
        "continuation logits length mismatch"
    );
    let mut continuation_logits_max_abs = 0.0f32;
    let (mut dot, mut reference_norm, mut candidate_norm) = (0.0f64, 0.0f64, 0.0f64);
    for (&a, &b) in reference_logits.iter().zip(&candidate_logits) {
        continuation_logits_max_abs = continuation_logits_max_abs.max((a - b).abs());
        dot += a as f64 * b as f64;
        reference_norm += (a as f64).powi(2);
        candidate_norm += (b as f64).powi(2);
    }
    let continuation_logits_cosine =
        dot / (reference_norm.sqrt() * candidate_norm.sqrt() + f64::MIN_POSITIVE);
    let continuation_argmax_equal = argmax_i32(&reference_logits) == argmax_i32(&candidate_logits);
    let continuation_token = argmax_i32(&reference_logits);
    let resume_audit_pass = kv_position_equal
        && kv_payload_cosine >= 0.99999
        && continuation_argmax_equal
        && continuation_logits_max_abs <= 5e-2
        && continuation_logits_cosine >= 0.99999;
    Ok(MtpTargetStateAudit {
        resume_audit_pass,
        kv_position_equal,
        kv_payload_exact,
        kv_payload_max_abs,
        kv_payload_cosine,
        reference_final_position,
        candidate_final_position,
        gdn_state_max_abs,
        gdn_conv_max_abs,
        continuation_argmax_equal,
        continuation_token,
        continuation_logits_max_abs,
        continuation_logits_cosine,
    })
}

fn merge_target_state_audit(aggregate: &mut MtpTargetStateAudit, next: MtpTargetStateAudit) {
    aggregate.resume_audit_pass &= next.resume_audit_pass;
    aggregate.kv_position_equal &= next.kv_position_equal;
    aggregate.kv_payload_exact &= next.kv_payload_exact;
    aggregate.kv_payload_max_abs = aggregate.kv_payload_max_abs.max(next.kv_payload_max_abs);
    aggregate.kv_payload_cosine = aggregate.kv_payload_cosine.min(next.kv_payload_cosine);
    aggregate.reference_final_position = next.reference_final_position;
    aggregate.candidate_final_position = next.candidate_final_position;
    aggregate.gdn_state_max_abs = aggregate.gdn_state_max_abs.max(next.gdn_state_max_abs);
    aggregate.gdn_conv_max_abs = aggregate.gdn_conv_max_abs.max(next.gdn_conv_max_abs);
    aggregate.continuation_argmax_equal &= next.continuation_argmax_equal;
    aggregate.continuation_token = next.continuation_token;
    aggregate.continuation_logits_max_abs = aggregate
        .continuation_logits_max_abs
        .max(next.continuation_logits_max_abs);
    aggregate.continuation_logits_cosine = aggregate
        .continuation_logits_cosine
        .min(next.continuation_logits_cosine);
}

const MTP_CONTINUATION_AUDIT_STEPS: usize = 16;

fn audit_mtp_target_state_chain(
    forward: &MetalForward<'_>,
    reference: &mut MetalSession,
    candidate: &mut MetalSession,
    expected_position: usize,
    pending_terminal_token: i32,
) -> Result<MtpTargetStateAudit> {
    let mut audit = audit_mtp_target_state(
        forward,
        reference,
        candidate,
        expected_position,
        pending_terminal_token,
    )?;
    for offset in 1..MTP_CONTINUATION_AUDIT_STEPS {
        let next_token = audit.continuation_token;
        let next = audit_mtp_target_state(
            forward,
            reference,
            candidate,
            expected_position + offset,
            next_token,
        )?;
        merge_target_state_audit(&mut audit, next);
    }
    audit.resume_audit_pass &= audit.gdn_state_max_abs <= 1e-2;
    audit.resume_audit_pass &= audit.gdn_conv_max_abs <= 1e-1;
    Ok(audit)
}

#[derive(Default)]
struct PldStats {
    index_build_ms: f64,
    index_update_ms: f64,
    lookup_ms: f64,
    serial_ms: f64,
    verify_ms: f64,
    restore_ms: f64,
    decode_ms: f64,
    attempts: u32,
    abstentions: u32,
    verify_calls: u32,
    restore_calls: u32,
    accepted: u32,
    drafts_scored: u32,
    prompt_attempts: u32,
    self_attempts: u32,
    target_transitions: u32,
    final_effective_verify_n: u32,
    accept_histogram: [u32; DRAFT_TOKENS + 1],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
enum PldTerminalCause {
    StopToken,
    OutputLimit,
}

#[derive(Debug, serde::Serialize)]
struct PldEvent {
    event: usize,
    carry_index: usize,
    carry_token: i32,
    action: &'static str,
    source: Option<&'static str>,
    source_start: Option<usize>,
    source_end: Option<usize>,
    proposal: Option<[i32; DRAFT_TOKENS]>,
    accepted_prefix: usize,
    terminal_cause: Option<PldTerminalCause>,
    effective_verify_n: Option<usize>,
    n_keep: Option<usize>,
    restored: bool,
    resulting_position: usize,
    emitted_after: usize,
}

fn run_pld(args: PldArgs) -> Result<()> {
    let PldArgs {
        model,
        prompt,
        qwen_chat,
        system,
        disable_thinking,
        tokens,
        stop_tokens,
        no_warmup,
        output,
        trace_events,
    } = args;
    anyhow::ensure!(tokens > 0, "--tokens must be positive");
    if !qwen_chat && (system.is_some() || disable_thinking) {
        anyhow::bail!("`--system` and `--disable-thinking` require `--qwen-chat`");
    }

    let ctx = MetalContext::new().context("init MetalContext")?;
    eprintln!("[pld] device: {}", ctx.describe());
    let g = GgufFile::open(&model).with_context(|| format!("open {}", model.display()))?;
    let stops = resolve_stop_tokens(&g, stop_tokens)?;
    let m = Model::from_gguf(&g).context("parse model arch")?;
    let mm = MetalModel::load(&ctx, &g, &m).context("metal-load weights")?;
    let tok = Tokenizer::from_gguf(&g).context("open tokenizer")?;
    let rendered_prompt = if qwen_chat {
        render_qwen_single_turn_prompt(&prompt, system.as_deref(), !disable_thinking)
    } else {
        prompt.clone()
    };
    let prompt_ids = tok
        .encode(&rendered_prompt, false)
        .context("tokenize prompt")?;
    anyhow::ensure!(!prompt_ids.is_empty(), "prompt tokenized to zero tokens");
    eprintln!(
        "[pld] model={} prompt_tokens={} gen={} stop_tokens={stops:?}",
        model.display(),
        prompt_ids.len(),
        tokens,
    );

    let mf = MetalForward::new(&ctx, &mm);
    let cap = prompt_ids.len() + tokens + 16;
    if !no_warmup {
        let mut session = MetalSession::fresh(&ctx, &mm, cap).context("warmup session")?;
        let _ = mf.single_token(prompt_ids[0], 0, &mut session)?;
    }

    let mut ref_session = MetalSession::fresh(&ctx, &mm, cap).context("ref session")?;
    let mut ref_scratch = MetalDFlashLayerMajorScratch::fresh_prefill(&ctx, &mm, 16)
        .context("ref prefill scratch")?;
    let ref_total_start = Instant::now();
    let ref_prefill_start = Instant::now();
    let ref_last_logits = prefill_tokens_with_multi_hidden(
        &mf,
        &prompt_ids,
        0,
        &mut ref_session,
        &mut ref_scratch,
        &[],
        None,
    )
    .context("ref packed prefill")?;
    let ref_prefill_ms = ref_prefill_start.elapsed().as_secs_f64() * 1e3;
    let ref_decode_start = Instant::now();
    let mut ref_generated = Vec::with_capacity(tokens);
    let mut ref_carry = argmax_i32(&ref_last_logits);
    let mut ref_processed_pos = (prompt_ids.len() - 1) as u32;
    loop {
        ref_generated.push(ref_carry);
        if stops.contains(&ref_carry) || ref_generated.len() >= tokens {
            break;
        }
        let position = ref_processed_pos + 1;
        let logits = mf.single_token(ref_carry, position, &mut ref_session)?;
        ref_carry = argmax_i32(&logits);
        ref_processed_pos = position;
    }
    let ref_decode_ms = ref_decode_start.elapsed().as_secs_f64() * 1e3;
    let ref_total_ms = ref_total_start.elapsed().as_secs_f64() * 1e3;

    let mut candidate_session = MetalSession::fresh(&ctx, &mm, cap).context("candidate session")?;
    let mut candidate_prefill_scratch = MetalDFlashLayerMajorScratch::fresh_prefill(&ctx, &mm, 16)
        .context("candidate prefill scratch")?;
    let mut verify_scratch =
        MetalDFlashVerifyScratch::fresh(&ctx, &mm, 8, 0).context("PLD verify scratch")?;
    let mut layer_scratch =
        MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, 8).context("PLD layer scratch")?;
    let candidate_total_start = Instant::now();
    let index_start = Instant::now();
    let mut proposer = PromptLookupProposer::new(&prompt_ids);
    let mut stats = PldStats {
        index_build_ms: index_start.elapsed().as_secs_f64() * 1e3,
        ..PldStats::default()
    };
    let candidate_prefill_start = Instant::now();
    let candidate_last_logits = prefill_tokens_with_multi_hidden(
        &mf,
        &prompt_ids,
        0,
        &mut candidate_session,
        &mut candidate_prefill_scratch,
        &[],
        None,
    )
    .context("candidate packed prefill")?;
    let candidate_prefill_ms = candidate_prefill_start.elapsed().as_secs_f64() * 1e3;
    let decode_start = Instant::now();
    let mut candidate_generated = Vec::with_capacity(tokens);
    let mut carry = argmax_i32(&candidate_last_logits);
    let mut processed_pos = (prompt_ids.len() - 1) as u32;
    let mut events = Vec::new();
    let mut event_index = 0usize;
    'outer: loop {
        let carry_index = candidate_generated.len();
        let event_carry = carry;
        candidate_generated.push(carry);
        let update_start = Instant::now();
        proposer.commit_verified(&[carry]);
        stats.index_update_ms += update_start.elapsed().as_secs_f64() * 1e3;
        if stops.contains(&carry) || candidate_generated.len() >= tokens {
            if trace_events {
                events.push(PldEvent {
                    event: event_index,
                    carry_index,
                    carry_token: event_carry,
                    action: "terminal_emit",
                    source: None,
                    source_start: None,
                    source_end: None,
                    proposal: None,
                    accepted_prefix: 0,
                    terminal_cause: Some(if stops.contains(&carry) {
                        PldTerminalCause::StopToken
                    } else {
                        PldTerminalCause::OutputLimit
                    }),
                    effective_verify_n: None,
                    n_keep: None,
                    restored: false,
                    resulting_position: candidate_session.kv_n_pos[0],
                    emitted_after: candidate_generated.len(),
                });
            }
            break;
        }

        let lookup_start = Instant::now();
        let candidate = proposer.propose();
        stats.lookup_ms += lookup_start.elapsed().as_secs_f64() * 1e3;
        let Some(candidate) = candidate else {
            stats.abstentions += 1;
            let serial_start = Instant::now();
            let position = processed_pos + 1;
            let logits = mf.single_token(carry, position, &mut candidate_session)?;
            stats.serial_ms += serial_start.elapsed().as_secs_f64() * 1e3;
            stats.target_transitions += 1;
            carry = argmax_i32(&logits);
            processed_pos = position;
            if trace_events {
                events.push(PldEvent {
                    event: event_index,
                    carry_index,
                    carry_token: event_carry,
                    action: "abstain",
                    source: None,
                    source_start: None,
                    source_end: None,
                    proposal: None,
                    accepted_prefix: 0,
                    terminal_cause: None,
                    effective_verify_n: None,
                    n_keep: Some(1),
                    restored: false,
                    resulting_position: candidate_session.kv_n_pos[0],
                    emitted_after: candidate_generated.len(),
                });
            }
            event_index += 1;
            continue;
        };

        stats.attempts += 1;
        match candidate.source {
            ProposalSource::Prompt => stats.prompt_attempts += 1,
            ProposalSource::SelfOutput => stats.self_attempts += 1,
        }
        let terminal_window = terminal_draft_window(
            &candidate.proposal,
            candidate_generated.len(),
            tokens,
            &stops,
        );
        let mut verify_input = Vec::with_capacity(8);
        verify_input.push(carry);
        let (n_eff, n_drafts_scored) = if let Some(window) = terminal_window {
            let count = window.count;
            verify_input.extend_from_slice(&candidate.proposal[..count.saturating_sub(1)]);
            (count, count)
        } else {
            verify_input.extend_from_slice(&candidate.proposal);
            (8, DRAFT_TOKENS)
        };
        stats.drafts_scored += n_drafts_scored as u32;
        stats.final_effective_verify_n = n_eff as u32;
        let start_position = processed_pos + 1;
        let verify_start = Instant::now();
        let verify_argmax = qwen_llm::metal_dflash::encode_packed_verify_layer_major_inner(
            &mf,
            &[],
            &verify_input,
            start_position,
            &mut verify_scratch,
            &mut layer_scratch,
            &mut candidate_session,
            None,
            Some(n_eff as u32),
        )
        .context("PLD packed verify")?;
        stats.verify_ms += verify_start.elapsed().as_secs_f64() * 1e3;
        stats.verify_calls += 1;
        stats.target_transitions += n_eff as u32;

        let mut accepted_tokens = Vec::with_capacity(n_drafts_scored);
        let mut stop_now = false;
        for (&draft, &target) in candidate.proposal[..n_drafts_scored]
            .iter()
            .zip(&verify_argmax)
        {
            if draft != target {
                break;
            }
            accepted_tokens.push(draft);
            candidate_generated.push(draft);
            stats.accepted += 1;
            if stops.contains(&draft) || candidate_generated.len() >= tokens {
                stop_now = true;
                break;
            }
        }
        let n_accepted = accepted_tokens.len();
        stats.accept_histogram[n_accepted] += 1;
        let n_keep = if stop_now { n_accepted } else { 1 + n_accepted };
        let restored = n_keep < n_eff;
        if restored {
            let restore_start = Instant::now();
            qwen_llm::metal_dflash::encode_restore_after_partial_accept_inner(
                &mf,
                &verify_scratch,
                n_keep as u32,
                start_position,
                &mut candidate_session,
                Some(n_eff as u32),
            )
            .context("PLD restore after partial accept")?;
            stats.restore_ms += restore_start.elapsed().as_secs_f64() * 1e3;
            stats.restore_calls += 1;
        }
        let update_start = Instant::now();
        proposer.commit_verified(&accepted_tokens);
        stats.index_update_ms += update_start.elapsed().as_secs_f64() * 1e3;
        if !stop_now {
            processed_pos += 1 + n_accepted as u32;
            carry = verify_argmax[n_accepted];
        }
        if trace_events {
            events.push(PldEvent {
                event: event_index,
                carry_index,
                carry_token: event_carry,
                action: "attempt",
                source: Some(match candidate.source {
                    ProposalSource::Prompt => "prompt",
                    ProposalSource::SelfOutput => "self",
                }),
                source_start: Some(candidate.source_start),
                source_end: Some(candidate.source_end),
                proposal: Some(candidate.proposal),
                accepted_prefix: n_accepted,
                terminal_cause: terminal_window.map(|window| match window.cause {
                    PromptLookupTerminalCause::StopToken => PldTerminalCause::StopToken,
                    PromptLookupTerminalCause::OutputLimit => PldTerminalCause::OutputLimit,
                }),
                effective_verify_n: Some(n_eff),
                n_keep: Some(n_keep),
                restored,
                resulting_position: candidate_session.kv_n_pos[0],
                emitted_after: candidate_generated.len(),
            });
        }
        event_index += 1;
        if stop_now {
            break 'outer;
        }
    }
    stats.decode_ms = decode_start.elapsed().as_secs_f64() * 1e3;
    let candidate_total_ms = candidate_total_start.elapsed().as_secs_f64() * 1e3;

    let identical = ref_generated == candidate_generated;
    anyhow::ensure!(identical, "PLD generated tokens differ from serial target");
    let pending_terminal_token = *ref_generated.last().context("PLD emitted no tokens")?;
    let expected_position = prompt_ids.len() + ref_generated.len() - 1;
    const PLD_CONTINUATION_AUDIT_STEPS: usize = 16;
    let mut audit = audit_mtp_target_state(
        &mf,
        &mut ref_session,
        &mut candidate_session,
        expected_position,
        pending_terminal_token,
    )?;
    for offset in 1..PLD_CONTINUATION_AUDIT_STEPS {
        let next_token = audit.continuation_token;
        let next = audit_mtp_target_state(
            &mf,
            &mut ref_session,
            &mut candidate_session,
            expected_position + offset,
            next_token,
        )?;
        merge_target_state_audit(&mut audit, next);
    }
    audit.resume_audit_pass &= audit.gdn_state_max_abs <= 1e-2;
    audit.resume_audit_pass &= audit.gdn_conv_max_abs <= 1e-1;
    anyhow::ensure!(audit.resume_audit_pass, "PLD terminal resume audit failed");

    let decode_speedup = ref_decode_ms / stats.decode_ms;
    let total_speedup = ref_total_ms / candidate_total_ms;
    eprintln!(
        "[pld] ref prefill={ref_prefill_ms:.1} decode={ref_decode_ms:.1} total={ref_total_ms:.1} ms"
    );
    eprintln!(
        "[pld] pld index={:.3} prefill={candidate_prefill_ms:.1} decode={:.1} \
         total={candidate_total_ms:.1} ms",
        stats.index_build_ms, stats.decode_ms,
    );
    eprintln!(
        "[pld] speedup decode={decode_speedup:.3}x total={total_speedup:.3}x \
         attempts={} abstentions={} verifies={} restores={} accepted={}/{}",
        stats.attempts,
        stats.abstentions,
        stats.verify_calls,
        stats.restore_calls,
        stats.accepted,
        stats.drafts_scored,
    );
    eprintln!(
        "[pld] phases lookup={:.3} update={:.3} serial={:.1} verify={:.1} restore={:.1} ms",
        stats.lookup_ms, stats.index_update_ms, stats.serial_ms, stats.verify_ms, stats.restore_ms,
    );

    if let Some(output_path) = output {
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let event_trace = if trace_events {
            serde_json::to_value(&events)?
        } else {
            serde_json::Value::Null
        };
        let row = serde_json::json!({
            "schema_version": 1,
            "model": model.display().to_string(),
            "prompt_tokens": prompt_ids.len(),
            "generated_requested": tokens,
            "generated_emitted": ref_generated.len(),
            "stop_tokens": stops,
            "policy": {
                "sources": "prompt_and_committed_output",
                "selector": "most_recent",
                "match_tokens": 8,
                "draft_tokens": DRAFT_TOKENS,
                "physical_verify_n": 8,
            },
            "reference": {
                "prefill_ms": ref_prefill_ms,
                "decode_ms": ref_decode_ms,
                "total_ms": ref_total_ms,
            },
            "candidate": {
                "index_build_ms": stats.index_build_ms,
                "prefill_ms": candidate_prefill_ms,
                "decode_ms": stats.decode_ms,
                "total_ms": candidate_total_ms,
                "lookup_ms": stats.lookup_ms,
                "index_update_ms": stats.index_update_ms,
                "serial_ms": stats.serial_ms,
                "verify_ms": stats.verify_ms,
                "restore_ms": stats.restore_ms,
                "attempts": stats.attempts,
                "abstentions": stats.abstentions,
                "verify_calls": stats.verify_calls,
                "restore_calls": stats.restore_calls,
                "accepted": stats.accepted,
                "drafts_scored": stats.drafts_scored,
                "prompt_attempts": stats.prompt_attempts,
                "self_attempts": stats.self_attempts,
                "target_transitions": stats.target_transitions,
                "final_effective_verify_n": (stats.verify_calls > 0)
                    .then_some(stats.final_effective_verify_n),
                "accept_histogram": stats.accept_histogram,
            },
            "decode_speedup": decode_speedup,
            "total_speedup": total_speedup,
            "identical": identical,
            "events": event_trace,
            "target_state": {
                "continuation_steps": PLD_CONTINUATION_AUDIT_STEPS,
                "resume_audit_pass": audit.resume_audit_pass,
                "kv_position_equal": audit.kv_position_equal,
                "kv_payload_exact": audit.kv_payload_exact,
                "kv_payload_max_abs": audit.kv_payload_max_abs,
                "kv_payload_cosine": audit.kv_payload_cosine,
                "gdn_state_max_abs": audit.gdn_state_max_abs,
                "gdn_conv_max_abs": audit.gdn_conv_max_abs,
                "continuation_argmax_equal": audit.continuation_argmax_equal,
                "continuation_logits_max_abs": audit.continuation_logits_max_abs,
                "continuation_logits_cosine": audit.continuation_logits_cosine,
            },
        });
        std::fs::write(&output_path, serde_json::to_string(&row)? + "\n")?;
        eprintln!("[pld] wrote {}", output_path.display());
    }
    Ok(())
}

fn run_mtp(args: MtpArgs) -> Result<()> {
    let MtpArgs {
        model,
        prompt,
        qwen_chat,
        system,
        disable_thinking,
        spec_tokens,
        mtp_probe,
        mtp_physical_n,
        mtp_single_cb_draft,
        mtp_draft_token_embd_head,
        mtp_draft_lm_head_q4_1,
        mtp_draft_lm_head_q4_0,
        mtp_draft_lm_head_q4_affine64,
        mtp_recursive_hidden,
        mtp_base_hidden,
        mtp_history,
        mtp_rank_topk,
        output,
        include_token_ids,
        tokens,
        stop_tokens,
        no_warmup,
    } = args;

    let ctx = MetalContext::new().context("init MetalContext")?;
    eprintln!("[mtp-bench] device: {}", ctx.describe());
    let spec_token_limit = 15;
    if spec_tokens == 0 || spec_tokens > spec_token_limit {
        anyhow::bail!(
            "`--spec-tokens` must be in 1..={spec_token_limit} for probe {:?}",
            mtp_probe
        );
    }
    if mtp_physical_n.is_some() && spec_tokens == 1 {
        anyhow::bail!("--mtp-physical-n requires --spec-tokens 2 or higher");
    }
    if mtp_rank_topk.is_some() {
        if spec_tokens == 1 {
            anyhow::bail!("--mtp-rank-topk requires --spec-tokens 2 or higher");
        }
        if mtp_probe != MtpProbeMode::Normal {
            anyhow::bail!("--mtp-rank-topk is only supported with --mtp-probe normal");
        }
    }
    let draft_lm_head_override_count = usize::from(mtp_draft_token_embd_head)
        + usize::from(mtp_draft_lm_head_q4_1)
        + usize::from(mtp_draft_lm_head_q4_0)
        + usize::from(mtp_draft_lm_head_q4_affine64);
    if draft_lm_head_override_count > 1 {
        anyhow::bail!("choose at most one draft lm_head override");
    }
    let planned_verify_n = mtp_physical_n.unwrap_or(spec_tokens + 1);
    if spec_tokens >= 2 {
        if !(spec_tokens + 1..=16).contains(&planned_verify_n) {
            anyhow::bail!(
                "--mtp-physical-n must be in {}..=16 for --spec-tokens {spec_tokens}",
                spec_tokens + 1
            );
        }
    }

    let g = GgufFile::open(&model).with_context(|| format!("open {}", model.display()))?;
    let stops = resolve_stop_tokens(&g, stop_tokens)?;
    let m = Model::from_gguf(&g).context("parse model arch")?;
    let mtp_view = m.mtp.as_ref().ok_or_else(|| {
        anyhow!(
            "GGUF has no MTP head (\
             use brittlewis12/Qwen3.6-27B-MTP-GGUF or the 0.8B-MTP variant)"
        )
    })?;

    let mm = MetalModel::load(&ctx, &g, &m).context("metal-load weights")?;
    let mtp_head = MetalMtpHead::load(&ctx, &g, mtp_view).context("metal-load MTP head")?;
    let mtp_moe_bank_ledger = mtp_head.attn.ffn_moe.as_ref().map(|moe| {
        (
            [moe.gate_exps.dtype, moe.up_exps.dtype, moe.down_exps.dtype],
            moe.gate_exps.n_bytes() + moe.up_exps.n_bytes() + moe.down_exps.n_bytes(),
        )
    });
    let tok = Tokenizer::from_gguf(&g).context("open tokenizer")?;

    if !qwen_chat && (system.is_some() || disable_thinking) {
        anyhow::bail!("`--system` and `--disable-thinking` require `--qwen-chat`");
    }
    let rendered_prompt = if qwen_chat {
        render_qwen_single_turn_prompt(&prompt, system.as_deref(), !disable_thinking)
    } else {
        prompt.clone()
    };
    let prompt_ids = tok
        .encode(&rendered_prompt, false)
        .context("tokenize prompt")?;
    eprintln!(
        concat!(
            "[mtp-bench] model={} prompt={:?} rendered_mode={} thinking={} ",
            "spec_tokens={} probe={:?} physical_n={} single_cb_draft={} ",
            "draft_token_embd_head={} draft_lm_head_q4_1={} ",
            "draft_lm_head_q4_0={} draft_lm_head_q4_affine64={} ",
            "base_hidden={:?} recursive_hidden={:?} mtp_history={:?} ",
            "({} tokens) gen={} stop_tokens={:?}"
        ),
        model.display(),
        prompt,
        if qwen_chat { "qwen-chat" } else { "raw" },
        if qwen_chat && !disable_thinking {
            "on"
        } else if qwen_chat {
            "off"
        } else {
            "n/a"
        },
        spec_tokens,
        mtp_probe,
        planned_verify_n,
        mtp_single_cb_draft,
        mtp_draft_token_embd_head,
        mtp_draft_lm_head_q4_1,
        mtp_draft_lm_head_q4_0,
        mtp_draft_lm_head_q4_affine64,
        mtp_base_hidden,
        mtp_recursive_hidden,
        mtp_history,
        prompt_ids.len(),
        tokens,
        stops,
    );
    if let Some((dtypes, bytes)) = mtp_moe_bank_ledger {
        eprintln!(
            "[mtp-bench] MTP MoE banks: policy={:?} gate/up/down={:?}/{:?}/{:?} bytes={bytes}",
            mtp_head.moe_bank_policy, dtypes[0], dtypes[1], dtypes[2],
        );
    }

    let mf = MetalForward::new(&ctx, &mm);
    let draft_lm_head_override = if mtp_draft_lm_head_q4_1 {
        let t = Instant::now();
        eprintln!("[mtp-bench] quantizing output.weight -> draft Q4_1 lm_head");
        let q =
            quantize_lm_head_to_q4_1(&ctx, &mm.lm_head).context("quantize draft lm_head Q4_1")?;
        eprintln!(
            "[mtp-bench] draft Q4_1 lm_head ready in {:.1} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
        Some(q)
    } else if mtp_draft_lm_head_q4_0 {
        let t = Instant::now();
        eprintln!("[mtp-bench] quantizing output.weight -> draft Q4_0 lm_head");
        let q =
            quantize_lm_head_to_q4_0(&ctx, &mm.lm_head).context("quantize draft lm_head Q4_0")?;
        eprintln!(
            "[mtp-bench] draft Q4_0 lm_head ready in {:.1} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
        Some(q)
    } else {
        None
    };
    let draft_affine_q4_head_override = if mtp_draft_lm_head_q4_affine64 {
        let t = Instant::now();
        eprintln!("[mtp-bench] quantizing output.weight -> draft affine Q4 gs64 lm_head");
        let q = quantize_lm_head_to_affine_q4_gs64(&ctx, &mm.lm_head)
            .context("quantize draft affine Q4 gs64 lm_head")?;
        eprintln!(
            "[mtp-bench] draft affine Q4 gs64 lm_head ready in {:.1} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
        Some(q)
    } else {
        None
    };
    let cap = prompt_ids.len() + tokens + 16;

    if !no_warmup {
        let mut s = MetalSession::fresh(&ctx, &mm, cap).context("warmup session")?;
        let _ = mf.single_token(prompt_ids[0], 0, &mut s)?;
    }

    // ----- MTP=off: greedy baseline -----
    let mut ref_session = MetalSession::fresh(&ctx, &mm, cap).context("ref session")?;
    let mut ref_tokens = prompt_ids.clone();
    let t_ref_total = Instant::now();
    let t_ref_prefill = Instant::now();
    // v0.75.1: packed multi-token prefill (no hidden capture needed for
    // the no-spec ref). Block size 16 matches DFlash convention; chunk
    // boundaries don't affect ref correctness.
    let mut ref_layer_scratch =
        MetalDFlashLayerMajorScratch::fresh_prefill(&ctx, &mm, 16).context("ref layer scratch")?;
    let last_logits = prefill_tokens_with_multi_hidden(
        &mf,
        &prompt_ids,
        0,
        &mut ref_session,
        &mut ref_layer_scratch,
        &[],
        None,
    )?;
    let ref_prefill_ms = t_ref_prefill.elapsed().as_secs_f64() * 1e3;

    let t_ref_decode = Instant::now();
    let mut next_tok = argmax_i32(&last_logits);
    let mut pos = (prompt_ids.len() - 1) as u32;
    let mut ref_emitted = 0usize;
    for _ in 0..tokens {
        ref_tokens.push(next_tok);
        ref_emitted += 1;
        if stops.contains(&next_tok) || ref_emitted >= tokens {
            break;
        }
        pos += 1;
        let logits = mf.single_token(next_tok, pos, &mut ref_session)?;
        next_tok = argmax_i32(&logits);
    }
    let ref_decode_ms = t_ref_decode.elapsed().as_secs_f64() * 1e3;
    let ref_total_ms = t_ref_total.elapsed().as_secs_f64() * 1e3;
    let ref_decode_tps = ref_emitted as f64 / (ref_decode_ms / 1000.0);
    let ref_generated_vec: Vec<i32> = ref_tokens[prompt_ids.len()..].to_vec();

    // ----- MTP=on: speculative decode -----
    if spec_tokens == 1
        && matches!(
            mtp_probe,
            MtpProbeMode::ReplayCurrent | MtpProbeMode::BodyNoLmHead | MtpProbeMode::BridgeOnly
        )
    {
        anyhow::bail!("recorded MTP probes require --spec-tokens 2..=15");
    }

    let mut planned_state_audit: Option<MtpTargetStateAudit> = None;
    let mut normal_state_audit: Option<MtpTargetStateAudit> = None;
    let mut run_planned = |plan: PackedDraftPlan<'_>, label: &str| -> Result<DecodeOutput> {
        let audit_target_state = matches!(plan, PackedDraftPlan::Oracle(_));
        let mtp_session =
            MetalMtpSession::fresh(&ctx, &mtp_head, &m.arch, cap).context("MTP session")?;
        let mut spec_session = MetalSession::fresh(&ctx, &mm, cap).context("spec session")?;
        let mut spec = SpeculativeDecoder::new(&mf, &mtp_head, mtp_session);
        spec.set_draft_token_embd_head(mtp_draft_token_embd_head);
        spec.set_draft_lm_head_override(draft_lm_head_override.clone());
        spec.set_draft_affine_q4_head_override(draft_affine_q4_head_override.clone());
        spec.set_base_hidden_variant(mtp_base_hidden.into());
        spec.set_recursive_hidden_variant(mtp_recursive_hidden.into());
        spec.set_history_mode(mtp_history.into());
        let mut verify_scratch =
            MetalDFlashVerifyScratch::fresh(&ctx, &mm, planned_verify_n as u32, 1)
                .context("mtp packed verify scratch")?;
        let mut layer_scratch =
            MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, planned_verify_n as u32)
                .context("mtp packed layer scratch")?;
        let output = spec
            .decode_packed_n_planned(
                &prompt_ids,
                tokens,
                &stops,
                &mut spec_session,
                spec_tokens,
                &mut verify_scratch,
                &mut layer_scratch,
                plan,
            )
            .with_context(|| format!("spec decode packed-n {label}"))?;
        if audit_target_state {
            let generated = &ref_generated_vec[..ref_generated_vec.len().saturating_sub(1)];
            let pending_terminal_token = *ref_generated_vec
                .last()
                .context("oracle produced no terminal token")?;
            let mut serial_session =
                MetalSession::fresh(&ctx, &mm, cap).context("state-audit serial session")?;
            for (position, &token) in prompt_ids.iter().chain(generated).enumerate() {
                mf.single_token(token, position as u32, &mut serial_session)
                    .context("state-audit serial transition")?;
            }
            planned_state_audit = Some(audit_mtp_target_state_chain(
                &mf,
                &mut serial_session,
                &mut spec_session,
                prompt_ids.len() + generated.len(),
                pending_terminal_token,
            )?);
        }
        Ok(output)
    };

    let run_recorded_work =
        |trace: &[RecordedDraftStep], work: RecordedMtpWork, label: &str| -> Result<DecodeOutput> {
            let mtp_session =
                MetalMtpSession::fresh(&ctx, &mtp_head, &m.arch, cap).context("MTP session")?;
            let mut spec_session = MetalSession::fresh(&ctx, &mm, cap).context("spec session")?;
            let mut spec = SpeculativeDecoder::new(&mf, &mtp_head, mtp_session);
            spec.set_draft_token_embd_head(mtp_draft_token_embd_head);
            spec.set_draft_lm_head_override(draft_lm_head_override.clone());
            spec.set_draft_affine_q4_head_override(draft_affine_q4_head_override.clone());
            spec.set_base_hidden_variant(mtp_base_hidden.into());
            spec.set_recursive_hidden_variant(mtp_recursive_hidden.into());
            spec.set_history_mode(mtp_history.into());
            let mut verify_scratch =
                MetalDFlashVerifyScratch::fresh(&ctx, &mm, planned_verify_n as u32, 1)
                    .context("mtp packed verify scratch")?;
            let mut layer_scratch =
                MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, planned_verify_n as u32)
                    .context("mtp packed layer scratch")?;
            spec.decode_packed_n_recorded_mtp_work(
                &prompt_ids,
                tokens,
                &stops,
                &mut spec_session,
                spec_tokens,
                &mut verify_scratch,
                &mut layer_scratch,
                trace,
                work,
            )
            .with_context(|| format!("spec decode packed-n {label}"))
        };

    let mut mtp_rank_rows: Vec<MtpRankRow> = Vec::new();
    let result = match mtp_probe {
        MtpProbeMode::Normal => {
            let mtp_session =
                MetalMtpSession::fresh(&ctx, &mtp_head, &m.arch, cap).context("MTP session")?;
            let mut spec_session = MetalSession::fresh(&ctx, &mm, cap).context("spec session")?;
            let mut spec = SpeculativeDecoder::new(&mf, &mtp_head, mtp_session);
            spec.set_draft_token_embd_head(mtp_draft_token_embd_head);
            spec.set_draft_lm_head_override(draft_lm_head_override.clone());
            spec.set_draft_affine_q4_head_override(draft_affine_q4_head_override.clone());
            spec.set_base_hidden_variant(mtp_base_hidden.into());
            spec.set_recursive_hidden_variant(mtp_recursive_hidden.into());
            spec.set_history_mode(mtp_history.into());
            let output = if spec_tokens == 1 {
                spec.decode(&prompt_ids, tokens, &stops, &mut spec_session)
                    .context("spec decode")?
            } else {
                let mut verify_scratch =
                    MetalDFlashVerifyScratch::fresh(&ctx, &mm, planned_verify_n as u32, 1)
                        .context("mtp packed verify scratch")?;
                let mut layer_scratch =
                    MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, planned_verify_n as u32)
                        .context("mtp packed layer scratch")?;
                spec.decode_packed_n_recording(
                    &prompt_ids,
                    tokens,
                    &stops,
                    &mut spec_session,
                    spec_tokens,
                    &mut verify_scratch,
                    &mut layer_scratch,
                    None,
                    mtp_rank_topk.as_ref().map(|_| &mut mtp_rank_rows),
                    mtp_single_cb_draft,
                )
                .context("spec decode packed-n")?
            };
            if spec_tokens >= 2 {
                let generated = &ref_generated_vec[..ref_generated_vec.len().saturating_sub(1)];
                anyhow::ensure!(
                    output.tokens[prompt_ids.len()..] == ref_generated_vec,
                    "native MTP emitted tokens differ from serial target"
                );
                let pending_terminal_token = *ref_generated_vec
                    .last()
                    .context("native MTP produced no terminal token")?;
                let mut serial_session =
                    MetalSession::fresh(&ctx, &mm, cap).context("state-audit serial session")?;
                for (position, &token) in prompt_ids.iter().chain(generated).enumerate() {
                    mf.single_token(token, position as u32, &mut serial_session)
                        .context("state-audit serial transition")?;
                }
                let audit = audit_mtp_target_state_chain(
                    &mf,
                    &mut serial_session,
                    &mut spec_session,
                    prompt_ids.len() + generated.len(),
                    pending_terminal_token,
                )?;
                anyhow::ensure!(
                    audit.resume_audit_pass,
                    "native MTP terminal resume audit failed: {audit:?}"
                );
                normal_state_audit = Some(audit);
            }
            output
        }
        MtpProbeMode::Oracle => run_planned(PackedDraftPlan::Oracle(&ref_generated_vec), "oracle")?,
        MtpProbeMode::ReplayCurrent | MtpProbeMode::BodyNoLmHead | MtpProbeMode::BridgeOnly => {
            let mtp_session =
                MetalMtpSession::fresh(&ctx, &mtp_head, &m.arch, cap).context("MTP session")?;
            let mut spec_session = MetalSession::fresh(&ctx, &mm, cap).context("spec session")?;
            let mut spec = SpeculativeDecoder::new(&mf, &mtp_head, mtp_session);
            spec.set_draft_token_embd_head(mtp_draft_token_embd_head);
            spec.set_draft_lm_head_override(draft_lm_head_override.clone());
            spec.set_draft_affine_q4_head_override(draft_affine_q4_head_override.clone());
            spec.set_base_hidden_variant(mtp_base_hidden.into());
            spec.set_recursive_hidden_variant(mtp_recursive_hidden.into());
            spec.set_history_mode(mtp_history.into());
            let mut verify_scratch =
                MetalDFlashVerifyScratch::fresh(&ctx, &mm, planned_verify_n as u32, 1)
                    .context("mtp packed verify scratch")?;
            let mut layer_scratch =
                MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, planned_verify_n as u32)
                    .context("mtp packed layer scratch")?;
            let mut draft_trace = Vec::new();
            let recorded = spec
                .decode_packed_n_recording(
                    &prompt_ids,
                    tokens,
                    &stops,
                    &mut spec_session,
                    spec_tokens,
                    &mut verify_scratch,
                    &mut layer_scratch,
                    Some(&mut draft_trace),
                    None,
                    mtp_single_cb_draft,
                )
                .context("spec decode packed-n recording")?;
            let recorded_emitted = recorded.tokens.len() - prompt_ids.len();
            let recorded_tps = recorded_emitted as f64 / (recorded.stats.wall_ms / 1000.0);
            eprintln!(
                "[mtp-bench] replay-current source: {recorded_emitted} tokens, \
                 {:.1} ms, {:.1} t/s, steps={}, trace_rows={}",
                recorded.stats.wall_ms,
                recorded_tps,
                recorded.stats.steps,
                draft_trace.len(),
            );
            match mtp_probe {
                MtpProbeMode::ReplayCurrent => {
                    run_planned(PackedDraftPlan::Recorded(&draft_trace), "replay-current")?
                }
                MtpProbeMode::BodyNoLmHead => run_recorded_work(
                    &draft_trace,
                    RecordedMtpWork::BodyNoLmHead,
                    "body-no-lm-head",
                )?,
                MtpProbeMode::BridgeOnly => {
                    run_recorded_work(&draft_trace, RecordedMtpWork::BridgeOnly, "bridge-only")?
                }
                MtpProbeMode::Normal | MtpProbeMode::Oracle => unreachable!(),
            }
        }
    };

    if let Some(rank_path) = &mtp_rank_topk {
        if let Some(dir) = rank_path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let mut out = String::new();
        for row in &mtp_rank_rows {
            let top_tokens = row
                .top_tokens
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",");
            let top_logits = row
                .top_logits
                .iter()
                .map(|v| format!("{v:.6}"))
                .collect::<Vec<_>>()
                .join(",");
            out.push_str(&format!(
                "{{\"step\":{},\"depth\":{},\"rank\":{},\"accepted\":{},\"draft_tok\":{},\"target_tok\":{},\"target_logit\":{:.6},\"top_tokens\":[{}],\"top_logits\":[{}]}}\n",
                row.step,
                row.depth,
                row.rank,
                row.accepted,
                row.draft_tok,
                row.target_tok,
                row.target_logit,
                top_tokens,
                top_logits,
            ));
        }
        std::fs::write(rank_path, out)?;
        eprintln!(
            "[mtp-bench] MTP rank rows: {} -> {}",
            mtp_rank_rows.len(),
            rank_path.display()
        );
        eprintln!("[mtp-bench] depth\tn\tp1\tp2\tp4\tp8\tp16\tmean_rank\tmax_rank");
        let max_depth = mtp_rank_rows.iter().map(|r| r.depth).max().unwrap_or(0);
        for depth in 0..=max_depth {
            let ranks: Vec<usize> = mtp_rank_rows
                .iter()
                .filter(|r| r.depth == depth)
                .map(|r| r.rank)
                .collect();
            if ranks.is_empty() {
                continue;
            }
            let n = ranks.len() as f64;
            let pk = |k: usize| ranks.iter().filter(|&&r| r <= k).count() as f64 / n;
            let mean = ranks.iter().sum::<usize>() as f64 / n;
            let max_rank = ranks.iter().copied().max().unwrap_or(0);
            eprintln!(
                "[mtp-bench] {depth}\t{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.1}\t{}",
                ranks.len(),
                pk(1),
                pk(2),
                pk(4),
                pk(8),
                pk(16),
                mean,
                max_rank
            );
        }
    }

    let spec_emitted = result.tokens.len() - prompt_ids.len();
    let spec_total_ms = result.stats.wall_ms;
    let spec_decode_ms = result.stats.decode_ms;
    let spec_prefill_ms = result.stats.prefill_ms;
    let spec_decode_tps = spec_emitted as f64 / (spec_decode_ms / 1000.0).max(f64::MIN_POSITIVE);
    let transitions_per_verify = if result.stats.steps > 0 {
        spec_emitted.saturating_sub(1) as f64 / result.stats.steps as f64
    } else {
        0.0
    };

    // ----- Compare -----
    let ref_generated = &ref_tokens[prompt_ids.len()..];
    let spec_generated = &result.tokens[prompt_ids.len()..];
    let identical = ref_generated == spec_generated;
    let expected_target_transitions = ref_emitted.saturating_sub(1);
    let target_state_audit = if mtp_probe == MtpProbeMode::Oracle {
        planned_state_audit
    } else if mtp_probe == MtpProbeMode::Normal && spec_tokens >= 2 {
        normal_state_audit
    } else {
        None
    };
    if mtp_probe == MtpProbeMode::Oracle {
        let audit = target_state_audit.context("oracle target-state audit missing")?;
        anyhow::ensure!(identical, "oracle emitted tokens differ from serial target");
        anyhow::ensure!(
            result.stats.mtp_calls == 0,
            "oracle unexpectedly executed {} MTP calls",
            result.stats.mtp_calls
        );
        anyhow::ensure!(
            result.stats.target_transitions as usize == expected_target_transitions,
            "oracle target transitions {} != expected {}",
            result.stats.target_transitions,
            expected_target_transitions
        );
        anyhow::ensure!(
            audit.resume_audit_pass,
            "oracle terminal resume audit failed: {audit:?}"
        );
    }

    // Apples-to-apples reporting. The earlier version mixed phases —
    // comparing MTP=off decode-only t/s (excludes prefill) with
    // MTP=on overall t/s (includes prefill) made the regression look
    // worse than it was. Wall-time-vs-wall-time is the honest signal.
    let ref_decode_only_tps = ref_emitted as f64 / (ref_decode_ms / 1000.0);
    let ref_total_tps = ref_emitted as f64 / (ref_total_ms / 1000.0);
    let spec_total_tps = spec_emitted as f64 / (spec_total_ms / 1000.0);
    let _ = ref_decode_tps; // unused (replaced by ref_decode_only_tps)
    let _ = spec_decode_tps; // unused (replaced by spec_total_tps for clarity)

    eprintln!();
    eprintln!("[mtp-bench] === results ===");
    eprintln!(
        "[mtp-bench] MTP=off: {ref_emitted} tokens, prefill {ref_prefill_ms:.1} ms + \
         decode {ref_decode_ms:.1} ms = {ref_total_ms:.1} ms total"
    );
    eprintln!(
        "[mtp-bench]   t/s: decode-only {ref_decode_only_tps:.1} | total \
         {ref_total_tps:.1}"
    );
    eprintln!(
        "[mtp-bench] MTP=on : {spec_emitted} tokens, prefill {spec_prefill_ms:.1} ms + \
         decode {spec_decode_ms:.1} ms = {spec_total_ms:.1} ms total"
    );
    eprintln!("[mtp-bench]   t/s: decode-only {spec_decode_tps:.1} | total {spec_total_tps:.1}");
    eprintln!(
        "[mtp-bench]   α (acceptance rate) = {:.3}   steps={}  accepted={}",
        result.stats.acceptance_rate(),
        result.stats.steps,
        result.stats.accepted,
    );
    eprintln!(
        "[mtp-bench]   target transitions / verifier = {:.3} ({}/{})",
        transitions_per_verify,
        spec_emitted.saturating_sub(1),
        result.stats.steps,
    );
    eprintln!(
        "[mtp-bench]   base_calls={}  mtp_calls={} \
         (= prompt prefill + step-B drafts + step-E bridges)",
        result.stats.base_forward_calls, result.stats.mtp_calls,
    );
    let spec_phase_known_ms = result.stats.draft_ms
        + result.stats.verify_ms
        + result.stats.restore_ms
        + result.stats.bridge_ms;
    let spec_phase_other_ms = (spec_decode_ms - spec_phase_known_ms).max(0.0);
    eprintln!(
        "[mtp-bench]   decode phases ms: draft={:.1} verify={:.1} restore={:.1} \
         bridge={:.1} other={:.1}",
        result.stats.draft_ms,
        result.stats.verify_ms,
        result.stats.restore_ms,
        result.stats.bridge_ms,
        spec_phase_other_ms,
    );

    // Wall-time speedup: total-vs-total, the apples-to-apples ratio that
    // matches docs/H4-MTP.md §3.2's `1/(1+ε)` prediction. The earlier
    // 'decode-only-vs-total' phrasing was misleading — let people see
    // both interpretations.
    let total_speedup = ref_total_ms / spec_total_ms;
    eprintln!(
        "[mtp-bench]   speedup (total ms): {ref_total_ms:.1} / {spec_total_ms:.1} = \
         {total_speedup:.3}× (>1.0 means MTP wins)"
    );

    eprintln!(
        "[mtp-bench] equivalence: {} ({} vs {} emitted)",
        if identical {
            "PASS (identical sequences)"
        } else {
            "FAIL (sequences differ)"
        },
        spec_emitted,
        ref_emitted,
    );
    if let Some(audit) = target_state_audit {
        eprintln!(
            "[mtp-bench] terminal resume: PASS continuation_steps={} \
             kv_pos={:?} kv_max_abs={:.3e} kv_cos={:.10} \
             gdn_state_max_abs={:.3e} gdn_conv_max_abs={:.3e} \
             continuation_max_abs={:.3e} continuation_cos={:.10}",
            MTP_CONTINUATION_AUDIT_STEPS,
            audit.candidate_final_position,
            audit.kv_payload_max_abs,
            audit.kv_payload_cosine,
            audit.gdn_state_max_abs,
            audit.gdn_conv_max_abs,
            audit.continuation_logits_max_abs,
            audit.continuation_logits_cosine,
        );
    }
    if !identical {
        let n_show = 8usize.min(ref_generated.len()).min(spec_generated.len());
        eprintln!(
            "[mtp-bench]   ref[..{n_show}]:  {:?}",
            &ref_generated[..n_show]
        );
        eprintln!(
            "[mtp-bench]   spec[..{n_show}]: {:?}",
            &spec_generated[..n_show]
        );
    }

    if let Some(output_path) = &output {
        if let Some(dir) = output_path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let steps = result.stats.steps as f64;
        let emitted_per_step = if result.stats.steps > 0 {
            serde_json::json!(spec_emitted as f64 / steps)
        } else {
            serde_json::Value::Null
        };
        let accepted_per_step = if result.stats.steps > 0 {
            serde_json::json!(result.stats.accepted as f64 / steps)
        } else {
            serde_json::Value::Null
        };
        let drafts_per_step = if result.stats.steps > 0 {
            serde_json::json!(result.stats.drafts_attempted as f64 / steps)
        } else {
            serde_json::Value::Null
        };
        let accepted_prefix_histogram = result
            .stats
            .accepted_prefix_counts
            .iter()
            .enumerate()
            .filter(|(_, count)| **count > 0)
            .map(|(accepted, count)| serde_json::json!({"accepted": accepted, "steps": count}))
            .collect::<Vec<_>>();
        let speculative_phase_ms = serde_json::json!({
            "draft": result.stats.draft_ms,
            "verify": result.stats.verify_ms,
            "restore": result.stats.restore_ms,
            "bridge": result.stats.bridge_ms,
            "other": spec_phase_other_ms,
        });
        let mtp_moe_banks =
            mtp_moe_bank_ledger.map_or(serde_json::Value::Null, |(dtypes, bytes)| {
                serde_json::json!({
                    "policy": format!("{:?}", mtp_head.moe_bank_policy),
                    "gate_dtype": format!("{:?}", dtypes[0]),
                    "up_dtype": format!("{:?}", dtypes[1]),
                    "down_dtype": format!("{:?}", dtypes[2]),
                    "bytes": bytes,
                })
            });
        let target_state = target_state_audit.map_or(serde_json::Value::Null, |audit| {
            serde_json::json!({
                "continuation_steps": MTP_CONTINUATION_AUDIT_STEPS,
                "resume_audit_pass": audit.resume_audit_pass,
                "kv_position_equal": audit.kv_position_equal,
                "kv_payload_exact": audit.kv_payload_exact,
                "kv_payload_max_abs": audit.kv_payload_max_abs,
                "kv_payload_cosine": audit.kv_payload_cosine,
                "reference_final_position": audit.reference_final_position,
                "candidate_final_position": audit.candidate_final_position,
                "gdn_state_max_abs": audit.gdn_state_max_abs,
                "gdn_conv_max_abs": audit.gdn_conv_max_abs,
                "continuation_argmax_equal": audit.continuation_argmax_equal,
                "continuation_logits_max_abs": audit.continuation_logits_max_abs,
                "continuation_logits_cosine": audit.continuation_logits_cosine,
            })
        });
        let semantics = serde_json::json!({
            "verify_mode": if spec_tokens == 1 { "lazy_mtp1" } else { "packed_n" },
            "sampler": "greedy_argmax",
            "correction_accounting": "deferred_next_step_carry",
            "equivalence": if target_state_audit.is_some() {
                "target_greedy_sequence_and_terminal_resume_audit"
            } else {
                "target_greedy_sequence"
            },
        });
        let reference = serde_json::json!({
            "emitted": ref_emitted,
            "target_transitions": expected_target_transitions,
            "prefill_ms": ref_prefill_ms,
            "decode_ms": ref_decode_ms,
            "total_ms": ref_total_ms,
            "decode_only_tps": ref_decode_only_tps,
            "total_tps": ref_total_tps,
        });
        let planned_metrics = matches!(
            mtp_probe,
            MtpProbeMode::Oracle | MtpProbeMode::ReplayCurrent
        );
        let speculative = serde_json::json!({
            "emitted": spec_emitted,
            "prefill_ms": spec_prefill_ms,
            "decode_ms": spec_decode_ms,
            "decode_only_tps": spec_decode_tps,
            "total_ms": spec_total_ms,
            "total_tps": spec_total_tps,
            "steps": result.stats.steps,
            "accepted": result.stats.accepted,
            "drafts_attempted": result.stats.drafts_attempted,
            "acceptance_rate": result.stats.acceptance_rate(),
            "emitted_per_step": emitted_per_step,
            "target_transitions_per_verify": transitions_per_verify,
            "accepted_per_step": accepted_per_step,
            "drafts_per_step": drafts_per_step,
            "accepted_prefix_histogram": accepted_prefix_histogram,
            "base_forward_calls": result.stats.base_forward_calls,
            "mtp_calls": result.stats.mtp_calls,
            "target_transitions": planned_metrics.then_some(result.stats.target_transitions),
            "final_effective_verify_n": planned_metrics
                .then_some(result.stats.final_effective_verify_n),
            "rank_rows": mtp_rank_rows.len(),
            "phase_ms": speculative_phase_ms,
            "target_state": target_state,
        });
        let token_fixture = if include_token_ids {
            serde_json::json!({
                "schema_version": 1,
                "prompt_token_ids": prompt_ids,
                "target_generated_token_ids": ref_generated,
            })
        } else {
            serde_json::Value::Null
        };
        let row = serde_json::json!({
            "model": model.display().to_string(),
            "prompt_tokens": prompt_ids.len(),
            "generated_requested": tokens,
            "stop_tokens": stops,
            "spec_tokens": spec_tokens,
            "logical_verify_n": spec_tokens + 1,
            "physical_verify_n": planned_verify_n,
            "terminal_token_target_transition_consumed": if target_state_audit.is_some() {
                Some(false)
            } else {
                None
            },
            "probe": format!("{:?}", mtp_probe),
            "single_cb_draft": mtp_single_cb_draft,
            "draft_token_embd_head": mtp_draft_token_embd_head,
            "draft_lm_head_q4_1": mtp_draft_lm_head_q4_1,
            "draft_lm_head_q4_0": mtp_draft_lm_head_q4_0,
            "draft_lm_head_q4_affine64": mtp_draft_lm_head_q4_affine64,
            "base_hidden": format!("{:?}", mtp_base_hidden),
            "recursive_hidden": format!("{:?}", mtp_recursive_hidden),
            "mtp_history": format!("{:?}", mtp_history),
            "mtp_moe_banks": mtp_moe_banks,
            "rank_topk": mtp_rank_topk.as_ref().map(|p| p.display().to_string()),
            "no_warmup": no_warmup,
            "semantics": semantics,
            "token_fixture": token_fixture,
            "reference": reference,
            "speculative": speculative,
            "speedup_total_ms": total_speedup,
            "identical": identical,
        });
        std::fs::write(output_path, serde_json::to_string(&row)? + "\n")?;
        eprintln!("[mtp-bench] wrote {}", output_path.display());
    }

    if !identical {
        return Err(anyhow!(
            "MTP=on and MTP=off generated different token sequences"
        ));
    }
    Ok(())
}

fn run_dflash_lazy(args: DflashLazyArgs) -> Result<()> {
    let DflashLazyArgs {
        model,
        drafter,
        prompt,
        tokens,
        stop_tokens,
        effective_n,
        no_warmup,
        rank_topk,
        tree_sim,
        tree_chain_d,
        tree_sibling_depths,
        tree_b,
    } = args;
    if tree_sim && rank_topk.is_none() {
        return Err(anyhow!("--tree-sim requires --rank-topk"));
    }
    if tree_sim {
        let nodes = tree_chain_d + tree_sibling_depths * (tree_b - 1);
        if nodes > 15 {
            return Err(anyhow!(
                "tree topology exceeds the N=16 block budget: D={tree_chain_d} + R={tree_sibling_depths}*(B-1={}) = {nodes} > 15",
                tree_b - 1
            ));
        }
        eprintln!(
            "[dflash-lazy] tree-sim: chain D={tree_chain_d}, sibling sets at first {tree_sibling_depths} depths, B={tree_b} ({nodes}/15 nodes)"
        );
    }

    let ctx = MetalContext::new().context("init MetalContext")?;
    eprintln!("[dflash-lazy] device: {}", ctx.describe());

    let target_g =
        GgufFile::open(&model).with_context(|| format!("open target {}", model.display()))?;
    let stops = resolve_stop_tokens(&target_g, stop_tokens)?;
    let target_m = Model::from_gguf(&target_g).context("parse target arch")?;
    let drafter_g =
        GgufFile::open(&drafter).with_context(|| format!("open drafter {}", drafter.display()))?;
    let head = open_dflash_drafter(&drafter_g, &target_m).context("bind drafter")?;

    let mm = MetalModel::load(&ctx, &target_g, &target_m).context("metal-load target")?;
    let mhead = MetalDFlashHead::load(&ctx, &drafter_g, &head).context("metal-load drafter")?;
    let tok = Tokenizer::from_gguf(&target_g).context("open tokenizer")?;

    let prompt_ids = tok.encode(&prompt, false).context("tokenize prompt")?;
    let n_prompt = prompt_ids.len();
    let cfg = head.config;
    let n = cfg.block_size as usize; // 16
    let d = n - 1; // 15 candidate slots in the block (positions 1..N)
    let m = if effective_n == 0 {
        d
    } else {
        effective_n.min(d).max(1)
    };
    let h_target = target_m.arch.hidden_size as usize;
    let v = target_m.arch.vocab_size as usize;
    let vocab = v;
    let k_layers = head.target_layer_ids.len();
    let n_target_features = k_layers * h_target;
    // T0 rank rows: (draft depth j, rank of target argmax in drafter row,
    // accepted, post_rescue). Prefix-conditioned by construction: recorded
    // only along the walked path plus its terminal mismatch.
    let mut rank_rows: Vec<(usize, usize, bool, bool)> = Vec::new();
    let mut rescues_taken: u32 = 0;
    let mut post_rescue_accepts: u32 = 0;
    let mut post_rescue_attempts: u32 = 0;

    eprintln!(
        "[dflash-lazy] target={} drafter={}",
        model.display(),
        drafter.display()
    );
    eprintln!(
        "[dflash-lazy] prompt={prompt:?} ({n_prompt} tokens) gen={tokens} stop_tokens={stops:?} \
         block_size={n} D={d} effective_M={m}"
    );

    let mf = MetalForward::new(&ctx, &mm);

    if !no_warmup {
        let mut s =
            MetalSession::fresh(&ctx, &mm, n_prompt + tokens + 32).context("warmup session")?;
        let _ = mf.single_token(prompt_ids[0], 0, &mut s)?;
    }

    let cap = n_prompt + tokens + 32;
    let mut target_session = MetalSession::fresh(&ctx, &mm, cap).context("target session")?;
    let mut dsess = MetalDFlashSession::fresh(&ctx, &mhead, h_target as u64, v as u64, cap)
        .context("dflash session")?;

    // Per-prompt-token captured hidden buffer ([K · H] each). Used by
    // the per-decode-step append (line 652).
    let multi_hidden_dst =
        MetalTensor::zeros_f32(&ctx, vec![n_target_features as u64]).context("multi_hidden_dst")?;

    // v0.75.1: contiguous [T, K*H] hidden capture buffer + dedicated
    // layer scratch for the packed prefill path. layer_scratch is
    // local to the prefill phase; the lazy decode loop doesn't reuse it.
    let prefill_hidden_dst =
        MetalTensor::zeros_f32(&ctx, vec![(n_prompt * n_target_features) as u64])
            .context("prefill_hidden_dst")?;
    let mut prefill_layer_scratch =
        MetalDFlashLayerMajorScratch::fresh_prefill(&ctx, &mm, cfg.block_size)
            .context("prefill layer scratch")?;

    // ---------- Prompt prefill ----------
    let t_prefill = Instant::now();
    let last_logits = prefill_tokens_with_multi_hidden(
        &mf,
        &prompt_ids,
        0,
        &mut target_session,
        &mut prefill_layer_scratch,
        &head.target_layer_ids,
        Some(&prefill_hidden_dst),
    )
    .context("prefill_tokens_with_multi_hidden")?;
    dsess
        .append_target_ctx_columns_contiguous_now(
            &ctx,
            &prefill_hidden_dst,
            0,
            n_prompt,
            n_target_features,
        )
        .context("append prefill ctx columns (batched)")?;
    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1e3;
    eprintln!(
        "[dflash-lazy] prefill {n_prompt} tokens in {prefill_ms:.1} ms (packed mat-mat + batched append)"
    );

    // Bootstrap: argmax of last prompt logits is the first emit token (carry).
    let mut emitted: Vec<i32> = Vec::with_capacity(tokens);
    let mut carry_tok = argmax_i32(&last_logits);
    let mut processed_pos = (n_prompt - 1) as u32;

    // Per-position acceptance counters (length M).
    let mut accepts_at_pos: Vec<u32> = vec![0; m];
    let mut attempts_at_pos: Vec<u32> = vec![0; m];

    // Top-k ranks: for each draft position 0..M and each outer iter, record
    // the rank of target's argmax in the drafter's logits at that noise
    // position. Drafter logits are NOT exposed in v1; we approximate top-k
    // hit rate by tracking whether target_argmax matches the drafter's
    // top-1 (= α) and reserve top-k>1 for a future bench (would require
    // returning full logits from draft_block).
    // For now: track a simpler "ran out of accepts" distribution.

    let mut steps: u32 = 0;
    let mut accepted_total: u32 = 0;
    let mut drafter_calls: u32 = 0;
    let mut base_calls: u32 = 0;
    let t_decode = Instant::now();

    let mut decoder = DFlashDecoder::new(&mf, &mhead, dsess);

    loop {
        // Emit + stop checks happen inside the loop so EOS / max can short-circuit.
        if emitted.len() >= tokens {
            break;
        }
        // Emit carry (was selected last iter or by bootstrap; not yet emitted).
        emitted.push(carry_tok);
        if stops.contains(&carry_tok) {
            break;
        }
        if emitted.len() >= tokens {
            break;
        }

        // ---- Drafter ----
        let drafter_pos = processed_pos + 1; // noise_start_pos
        let mut draft_logits: Option<Vec<f32>> = None;
        let drafts: Vec<i32> = if rank_topk.is_some() {
            // T0: full drafter logits for rank measurement; drafts recomputed
            // host-side from the same logits (identical argmax by
            // construction).
            let logits = decoder
                .draft_block_with_logits(carry_tok, drafter_pos)
                .context("drafter draft_block_with_logits")?;
            let v = vocab;
            let d: Vec<i32> = (1..=m)
                .map(|row| argmax_i32(&logits[row * v..(row + 1) * v]))
                .collect();
            draft_logits = Some(logits);
            d
        } else {
            let argmaxes = decoder
                .draft_block(carry_tok, drafter_pos)
                .context("drafter draft_block")?;
            // Draft tokens come from positions 1..N.
            argmaxes[1..].iter().take(m).copied().collect()
        };
        drafter_calls += 1;

        // ---- Lazy verify ----
        // First, process carry_tok via target. Capture hidden + logits.
        let target_logits = mf
            .single_token_with_multi_hidden(
                carry_tok,
                drafter_pos,
                &mut target_session,
                &head.target_layer_ids,
                &multi_hidden_dst,
            )
            .context("verify base step (carry)")?;
        base_calls += 1;
        // Append carry's hidden to target_ctx.
        decoder
            .session
            .append_target_ctx_column_now(&ctx, &multi_hidden_dst, drafter_pos, n_target_features)
            .context("append carry ctx column")?;
        processed_pos += 1;
        let mut target_next = argmax_i32(&target_logits);

        // Walk the block. Chain mode: first mismatch ends the step.
        // Tree-sim mode: a mismatch at depth < R whose target argmax sits in
        // the drafter's top-B at that position is a RESCUE - the rescued
        // token (a target argmax) is emitted, processed via target, and the
        // walk CONTINUES against the same block's deeper rows (valid because
        // the block drafter never conditioned on its own intermediate
        // tokens). Budget/topology is static and pre-committed per step.
        let mut n_accepted_this_step = 0usize;
        steps += 1;
        let walk_max = if tree_sim { tree_chain_d.min(m) } else { m };
        let mut post_rescue = false;
        for j in 0..walk_max {
            attempts_at_pos[j] += 1;
            let mut rank = usize::MAX;
            if let Some(logits) = &draft_logits {
                let row = &logits[(j + 1) * vocab..(j + 2) * vocab];
                let t = target_next as usize;
                let tv = row[t];
                rank = 1 + row.iter().filter(|&&x| x > tv).count();
                rank_rows.push((j, rank, drafts[j] == target_next, post_rescue));
            }
            if post_rescue {
                post_rescue_attempts += 1;
            }
            let chain_hit = drafts[j] == target_next;
            let rescue_hit = tree_sim && !chain_hit && j < tree_sibling_depths && rank <= tree_b;
            if !chain_hit && !rescue_hit {
                break;
            }
            // Accepted: either the chain token or the rescued sibling (which
            // IS target's argmax, so the emitted stream stays target-greedy).
            let tok = if chain_hit { drafts[j] } else { target_next };
            if chain_hit {
                accepts_at_pos[j] += 1;
                accepted_total += 1;
                if post_rescue {
                    post_rescue_accepts += 1;
                }
            } else {
                rescues_taken += 1;
                post_rescue = true;
            }
            n_accepted_this_step += 1;
            emitted.push(tok);
            if emitted.len() >= tokens || stops.contains(&tok) {
                break;
            }
            // Process the accepted token via target for the next position.
            let logits = mf
                .single_token_with_multi_hidden(
                    tok,
                    processed_pos + 1,
                    &mut target_session,
                    &head.target_layer_ids,
                    &multi_hidden_dst,
                )
                .context("verify base step (walk)")?;
            base_calls += 1;
            decoder
                .session
                .append_target_ctx_column_now(
                    &ctx,
                    &multi_hidden_dst,
                    processed_pos + 1,
                    n_target_features,
                )
                .context("append walk ctx column")?;
            processed_pos += 1;
            target_next = argmax_i32(&logits);
        }

        // After loop: target_next holds what target wants AT processed_pos+1.
        // That becomes the new carry (will be emitted next iteration top).
        carry_tok = target_next;
        let _ = n_accepted_this_step; // (already counted)
    }

    let decode_ms = t_decode.elapsed().as_secs_f64() * 1e3;
    let total_ms = t_prefill.elapsed().as_secs_f64() * 1e3;

    // ---------- T0 rank artifact + p_k(depth) table ----------
    if let Some(rank_path) = &rank_topk {
        if let Some(dir) = rank_path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let mut out = String::new();
        for &(j, rank, accepted, post_rescue) in &rank_rows {
            out.push_str(&format!(
                "{{\"depth\":{},\"rank\":{},\"accepted\":{},\"post_rescue\":{}}}\n",
                j, rank, accepted, post_rescue
            ));
        }
        std::fs::write(rank_path, out)?;
        println!(
            "[dflash-lazy] T0 rank rows: {} -> {}",
            rank_rows.len(),
            rank_path.display()
        );
        // p_k(depth): P[rank <= k at depth j | prefix accepted to j-1].
        println!("[dflash-lazy] depth\tn\tp1\tp2\tp4\tp8\tp16");
        for j in 0..m {
            let at: Vec<usize> = rank_rows.iter().filter(|r| r.0 == j).map(|r| r.1).collect();
            if at.is_empty() {
                continue;
            }
            let nn = at.len() as f64;
            let pk = |k: usize| at.iter().filter(|&&r| r <= k).count() as f64 / nn;
            println!(
                "[dflash-lazy] {j}\t{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}",
                at.len(),
                pk(1),
                pk(2),
                pk(4),
                pk(8),
                pk(16)
            );
        }
        if tree_sim {
            println!(
                "[dflash-lazy] tree-sim: emitted/step = {} / {} = {:.3}; rescues={} post-rescue chain accepts {}/{} = {:.3}",
                emitted.len(),
                steps,
                emitted.len() as f64 / steps.max(1) as f64,
                rescues_taken,
                post_rescue_accepts,
                post_rescue_attempts,
                post_rescue_accepts as f64 / post_rescue_attempts.max(1) as f64
            );
        }
    }

    // ---------- Apples-to-apples no-spec baseline ----------
    eprintln!("[dflash-lazy] running MTP=off greedy baseline for comparison ...");
    let mut ref_session = MetalSession::fresh(&ctx, &mm, cap).context("ref session")?;
    let t_ref_total = Instant::now();
    let t_ref_prefill = Instant::now();
    // v0.75.1: packed multi-token prefill (no hidden capture).
    let mut ref_layer_scratch =
        MetalDFlashLayerMajorScratch::fresh_prefill(&ctx, &mm, 16).context("ref layer scratch")?;
    let last_logits_ref = prefill_tokens_with_multi_hidden(
        &mf,
        &prompt_ids,
        0,
        &mut ref_session,
        &mut ref_layer_scratch,
        &[],
        None,
    )?;
    let ref_prefill_ms = t_ref_prefill.elapsed().as_secs_f64() * 1e3;
    let mut next_tok = argmax_i32(&last_logits_ref);
    let mut ref_emitted: Vec<i32> = Vec::with_capacity(tokens);
    let mut pos = (n_prompt - 1) as u32;
    let t_ref_decode = Instant::now();
    for _ in 0..tokens {
        ref_emitted.push(next_tok);
        if stops.contains(&next_tok) {
            break;
        }
        pos += 1;
        let logits = mf.single_token(next_tok, pos, &mut ref_session)?;
        next_tok = argmax_i32(&logits);
    }
    let ref_decode_ms = t_ref_decode.elapsed().as_secs_f64() * 1e3;
    let ref_total_ms = t_ref_total.elapsed().as_secs_f64() * 1e3;

    // ---------- Report ----------
    eprintln!();
    eprintln!("[dflash-lazy] === results ===");
    eprintln!("[dflash-lazy] generated {} tokens", emitted.len());
    eprintln!(
        "[dflash-lazy] prefill {prefill_ms:.1} ms, decode {decode_ms:.1} ms, total {total_ms:.1} ms"
    );
    eprintln!(
        "[dflash-lazy]   throughput: total {:.2} t/s",
        emitted.len() as f64 / (total_ms / 1000.0)
    );
    eprintln!(
        "[dflash-lazy] no-spec ref: prefill {ref_prefill_ms:.1} ms, decode {ref_decode_ms:.1} ms, \
         total {ref_total_ms:.1} ms"
    );
    eprintln!(
        "[dflash-lazy]   throughput: decode-only {:.2} t/s | total {:.2} t/s",
        ref_emitted.len() as f64 / (ref_decode_ms / 1000.0),
        ref_emitted.len() as f64 / (ref_total_ms / 1000.0),
    );
    let speedup = ref_total_ms / total_ms;
    eprintln!(
        "[dflash-lazy]   speedup (total ms): {ref_total_ms:.1} / {total_ms:.1} = {speedup:.3}× \
         (lazy verify is correctness gate, not perf path; expect <1.0×)"
    );

    // Two ways to summarize α — both useful, neither alone is enough:
    //
    //  α_chain  = mean_accepted_drafts / steps       ∈ [0, M]
    //             "how many drafts make it past the chain check, on average"
    //             This is the speedup-relevant raw signal: tokens emitted
    //             per outer step = 1 + α_chain.
    //
    //  α_pos1   = accepts_at_pos[0] / attempts_at_pos[0]
    //             "rank-1 hit rate at the FIRST draft slot"
    //             vLLM/spiritbuun's reported "acceptance rate" is closest
    //             to this — if the first draft misses, the chain dies.
    //             This is the metric the GO/NO-GO gate compares against
    //             (z-lab claims ~93% on quicksort, ~38% on prose).
    //
    // Both are reported.
    let alpha_chain = if steps > 0 {
        accepted_total as f64 / steps as f64
    } else {
        0.0
    };
    let mean_emitted_per_step = 1.0 + alpha_chain;
    let alpha_pos1 = if attempts_at_pos.first().copied().unwrap_or(0) > 0 {
        accepts_at_pos[0] as f64 / attempts_at_pos[0] as f64
    } else {
        0.0
    };
    eprintln!();
    eprintln!("[dflash-lazy] === acceptance ===");
    eprintln!(
        "[dflash-lazy] outer steps={steps}  accepted_drafts={accepted_total}  drafter_calls={drafter_calls}  base_calls={base_calls}"
    );
    eprintln!(
        "[dflash-lazy] α_chain = mean_accepted_drafts / steps = {accepted_total} / {steps} = {alpha_chain:.3} drafts/step (max M={m})"
    );
    eprintln!(
        "[dflash-lazy] mean_emitted_per_step = 1 + α_chain = {mean_emitted_per_step:.3} tokens/step"
    );
    eprintln!(
        "[dflash-lazy] α_pos1 (rank-1 hit at first draft slot) = {} / {} = {alpha_pos1:.3}",
        accepts_at_pos[0], attempts_at_pos[0]
    );
    eprintln!("[dflash-lazy] per-position α (conditional on reaching that slot):");
    for j in 0..m {
        let attempts = attempts_at_pos[j];
        let accepts = accepts_at_pos[j];
        let alpha_j = if attempts > 0 {
            accepts as f64 / attempts as f64
        } else {
            0.0
        };
        eprintln!("[dflash-lazy]   position {j:2}: {accepts:>4}/{attempts:>4} = {alpha_j:.3}");
    }

    eprintln!();
    eprintln!("[dflash-lazy] GO/NO-GO gate (per docs/H5-DFLASH.md §H5.2.5):");
    eprintln!("[dflash-lazy]   α_pos1 ≥ 0.50 on code  → GO for H5.3 packed verify");
    eprintln!("[dflash-lazy]   α_pos1 ≥ 0.30 on prose → GO for H5.3 packed verify");
    eprintln!(
        "[dflash-lazy]   α_pos1 <  0.30 on prose → STOP. Debug drafter forward, SWA mask, hidden capture, quant, recipe."
    );
    eprintln!(
        "[dflash-lazy]   measured: α_pos1={alpha_pos1:.3} α_chain={alpha_chain:.3} on prompt {prompt:?} ({n_prompt}-token prefill, {} emitted)",
        emitted.len()
    );

    // Equivalence check (lazy verify is exact under greedy because we
    // only commit tokens equal to target_argmax).
    let identical = emitted == ref_emitted;
    eprintln!(
        "[dflash-lazy] equivalence vs no-spec greedy: {} ({} vs {} emitted)",
        if identical {
            "PASS (identical sequences — lazy verify is correct)"
        } else {
            "FAIL (sequences differ — bug in verify logic)"
        },
        emitted.len(),
        ref_emitted.len(),
    );
    if !identical {
        let n_show = 8.min(emitted.len()).min(ref_emitted.len());
        eprintln!("[dflash-lazy]   ours[..{n_show}]: {:?}", &emitted[..n_show]);
        eprintln!(
            "[dflash-lazy]   ref [..{n_show}]: {:?}",
            &ref_emitted[..n_show]
        );
        return Err(anyhow!(
            "lazy verify produced different tokens than no-spec greedy"
        ));
    }
    Ok(())
}

/// **H5.5 production DFlash decode** end-to-end bench.
///
/// Implements plan §1.3 algorithm:
///   per outer step:
///     drafts = draft_block(carry, processed_pos+1)[1..]
///     verify_argmax = packed_verify([carry, drafts[0..D-1]],
///                                    start_pos = processed_pos+1)
///     n_accepted = greedy match prefix
///     emit(carry); emit_all(drafts[0..n_accepted])
///     bonus = verify_argmax[n_accepted]; carry = bonus
///     append target_ctx with hidden_capture[0..=n_accepted]
///     if n_accepted < D: restore_after_partial_accept(n_accepted+1, ...)
///     processed_pos += 1 + n_accepted
///
/// EOS edge cases:
///   * EOS in carry → emit, stop, no drafter (handled at top of loop)
///   * EOS in accepted draft j → emit prefix through EOS, stop
///   * EOS as bonus → emit accepted prefix; bonus becomes next carry,
///     and the next iter's emit-then-stop fires
///   * EOS as draft at index ≥ n_accepted → bonus wins (verify says
///     not EOS); EOS not emitted
///
/// Compares vs DFlash=off baseline for greedy equivalence (token
/// sequences MUST match) and reports speedup.
fn run_dflash(args: DflashArgs) -> Result<()> {
    let DflashArgs {
        model,
        drafter,
        prompt,
        tokens,
        stop_tokens,
        no_warmup,
        skip_equivalence_check,
        profile,
        n_policy,
    } = args;
    let n_policy =
        NPolicy::parse(&n_policy).with_context(|| format!("invalid --n-policy={n_policy:?}"))?;

    let ctx = MetalContext::new().context("init MetalContext")?;
    eprintln!("[dflash] device: {}", ctx.describe());

    let target_g =
        GgufFile::open(&model).with_context(|| format!("open target {}", model.display()))?;
    let stops = resolve_stop_tokens(&target_g, stop_tokens)?;
    let target_m = Model::from_gguf(&target_g).context("parse target arch")?;
    let drafter_g =
        GgufFile::open(&drafter).with_context(|| format!("open drafter {}", drafter.display()))?;
    let head = open_dflash_drafter(&drafter_g, &target_m).context("bind drafter")?;
    let mm = MetalModel::load(&ctx, &target_g, &target_m).context("metal-load target")?;
    let mhead = MetalDFlashHead::load(&ctx, &drafter_g, &head).context("metal-load drafter")?;
    let tok = Tokenizer::from_gguf(&target_g).context("open tokenizer")?;

    let prompt_ids = tok.encode(&prompt, false).context("tokenize prompt")?;
    let n_prompt = prompt_ids.len();
    let cfg = head.config;
    let n_block = cfg.block_size as usize; // N=16
    let d = n_block - 1; // D=15
    let h_target = target_m.arch.hidden_size as usize;
    let v = target_m.arch.vocab_size as usize;
    let k_layers = head.target_layer_ids.len();
    let n_target_features = k_layers * h_target;

    eprintln!(
        "[dflash] target={} drafter={}",
        model.display(),
        drafter.display()
    );
    eprintln!(
        "[dflash] prompt={prompt:?} ({n_prompt} tokens) gen={tokens} stop_tokens={stops:?} \
         block_size={n_block} D={d}"
    );

    let mf = MetalForward::new(&ctx, &mm);

    if !no_warmup {
        let mut s =
            MetalSession::fresh(&ctx, &mm, n_prompt + tokens + 32).context("warmup session")?;
        let _ = mf.single_token(prompt_ids[0], 0, &mut s)?;
    }

    let cap = n_prompt + tokens + 32;
    let mut target_session = MetalSession::fresh(&ctx, &mm, cap).context("target session")?;
    let mut dsess = MetalDFlashSession::fresh(&ctx, &mhead, h_target as u64, v as u64, cap)
        .context("dflash session")?;

    // Production DFlash scratch buffers.
    let mut verify_scratch =
        MetalDFlashVerifyScratch::fresh(&ctx, &mm, cfg.block_size, k_layers as u32)
            .context("verify scratch")?;
    let mut layer_scratch =
        MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, cfg.block_size).context("layer scratch")?;

    // v0.75.1: contiguous [T, K*H] hidden capture buffer for the
    // packed prefill path. One allocation, one prefill call, one
    // batched append.
    let prefill_hidden_dst =
        MetalTensor::zeros_f32(&ctx, vec![(n_prompt * n_target_features) as u64])
            .context("prefill_hidden_dst")?;

    // ---------- Prompt prefill ----------
    let t_prefill = Instant::now();
    let last_logits = prefill_tokens_with_multi_hidden(
        &mf,
        &prompt_ids,
        0,
        &mut target_session,
        &mut layer_scratch,
        &head.target_layer_ids,
        Some(&prefill_hidden_dst),
    )
    .context("prefill_tokens_with_multi_hidden")?;
    dsess
        .append_target_ctx_columns_contiguous_now(
            &ctx,
            &prefill_hidden_dst,
            0,
            n_prompt,
            n_target_features,
        )
        .context("append prefill ctx columns (batched)")?;
    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1e3;
    eprintln!("[dflash] prefill {n_prompt} tokens in {prefill_ms:.1} ms");

    let mut emitted: Vec<i32> = Vec::with_capacity(tokens);
    let mut carry_tok = argmax_i32(&last_logits);
    let mut processed_pos = (n_prompt - 1) as u32;

    let mut steps: u32 = 0;
    let mut accepted_total: u32 = 0;
    let mut accepts_at_pos: Vec<u32> = vec![0; d];
    let mut attempts_at_pos: Vec<u32> = vec![0; d];
    let mut verify_calls: u32 = 0;
    let mut drafter_calls: u32 = 0;
    let mut restore_calls: u32 = 0;
    // v0.76 adaptive-N counters.
    let mut spec16_steps: u32 = 0;
    let mut spec8_steps: u32 = 0;
    let mut spec4_steps: u32 = 0;
    let mut off_steps: u32 = 0;
    // `Off` is terminal once entered (codex Q7: ctx is monotonic
    // within a generation, so a ctx that earned `Off` will never
    // cool back to favor `Spec`).
    let mut spec_disabled = false;

    let mut decoder = DFlashDecoder::new(&mf, &mhead, dsess);
    if profile {
        decoder.session.enable_phase_timers();
    }

    // H5.6 M1c: per-component wall accounting across the outer loop.
    // Terminal-step bias note: the loop can break mid-step (accept hits
    // `tokens`), so append/restore of the final step may be skipped; run
    // with --tokens >= 256 when using these numbers for economics.
    let mut acct_draft_ms = 0.0f64;
    let mut acct_verify_ms = 0.0f64;
    let mut acct_append_ms = 0.0f64;
    let mut acct_restore_ms = 0.0f64;

    let t_decode = Instant::now();
    'outer: loop {
        if emitted.len() >= tokens {
            break;
        }
        // Emit carry (selected last iter or by bootstrap; not yet in emitted).
        emitted.push(carry_tok);
        if stops.contains(&carry_tok) || emitted.len() >= tokens {
            break;
        }

        // ---- v0.76 adaptive-N: select VerifyMode for THIS step ----
        //
        // `processed_pos` here is the absolute KV position of the
        // carry's predecessor (incremented at the bottom of the loop
        // by `1 + n_accepted` per Spec step or by `1` per Off step).
        // The schedule keys on the upcoming verify's start position,
        // which is `processed_pos + 1`.
        let mode = if spec_disabled {
            VerifyMode::Off
        } else {
            n_policy.for_ctx((processed_pos + 1) as usize)
        };

        if matches!(mode, VerifyMode::Off) {
            // Off branch: no drafter, no packed_verify, no restore.
            // No drafter ctx update — drafter is permanently disabled
            // for the remainder of this generation.
            spec_disabled = true;
            off_steps += 1;
            steps += 1;
            let single_pos = processed_pos + 1;
            let logits = mf
                .single_token(carry_tok, single_pos, &mut target_session)
                .context("off-mode single_token")?;
            let next_tok = argmax_i32(&logits);
            // Advance cursors. carry_tok was already emitted at top of
            // the loop; next iter's carry is `next_tok`.
            processed_pos = single_pos;
            carry_tok = next_tok;
            continue;
        }

        // ---- Spec branch: existing drafter + packed_verify + restore ----
        let n_eff = match mode {
            VerifyMode::Spec { n_eff } => n_eff,
            VerifyMode::Off => unreachable!("Off handled above"),
        };
        match n_eff {
            16 => spec16_steps += 1,
            8 => spec8_steps += 1,
            4 => spec4_steps += 1,
            _ => {} // unexpected; bench on (we only schedule {16, 8, 4})
        }

        // ---- Drafter ----
        // The drafter always produces a full N=block_size chain
        // (block_size is GGUF-fixed metadata; can't change per call).
        // Adaptive-N truncates the VERIFY chain via `n_eff_override`
        // — drafter slots [n_eff..N) are computed but unused. This
        // wastes some drafter work at small n_eff; the alternative
        // (separate small-block drafters) is out of scope. Drafter
        // overhead is ~12% of decode wall after v0.74.2, so the
        // wasted fraction (1 - n_eff/N) of 12% is bounded.
        let drafter_pos = processed_pos + 1; // noise_start_pos
        let t_acct = Instant::now();
        let argmaxes = decoder
            .draft_block(carry_tok, drafter_pos)
            .context("drafter draft_block")?;
        acct_draft_ms += t_acct.elapsed().as_secs_f64() * 1e3;
        drafter_calls += 1;
        let drafts: Vec<i32> = argmaxes[1..].to_vec();
        debug_assert_eq!(drafts.len(), d);

        // ---- Packed verify ----
        // Input: [carry, drafts[0..n_eff-1]] of length n_eff. Truncate
        // to `n_eff` (≤ d=N-1, so we use drafts[..n_eff-1] to fit
        // carry + (n_eff-1) drafts = n_eff total tokens).
        let n_drafts_used = n_eff - 1; // carry + drafts = n_eff
        let mut verify_input: Vec<i32> = Vec::with_capacity(n_eff);
        verify_input.push(carry_tok);
        verify_input.extend_from_slice(&drafts[..n_drafts_used]);

        let t_acct = Instant::now();
        let verify_argmax = qwen_llm::metal_dflash::encode_packed_verify_layer_major_inner(
            decoder.base,
            &decoder.head.target_layer_ids,
            &verify_input,
            drafter_pos,
            &mut verify_scratch,
            &mut layer_scratch,
            &mut target_session,
            None,
            Some(n_eff as u32), // adaptive-N: truncate verify chain to n_eff
        )
        .context("packed_verify")?;
        acct_verify_ms += t_acct.elapsed().as_secs_f64() * 1e3;
        verify_calls += 1;
        debug_assert_eq!(verify_argmax.len(), n_eff);

        // ---- Greedy accept-prefix ----
        // n_accepted = number of DRAFT tokens accepted (∈ [0, D]).
        // Indexing invariant:
        //   verify_input = [carry, drafts[0], drafts[1], ..., drafts[d-1]]
        //   verify_argmax[i] = argmax of target's forward AT position
        //     drafter_pos + i, given input verify_input[i].
        // So verify_argmax[0] is target's prediction AFTER consuming
        // carry — i.e., what target says SHOULD come next. drafts[0]
        // is what drafter predicted for that same slot. Greedy
        // comparison: drafts[j] == verify_argmax[j] for j ∈ [0, d).
        // Stop at first mismatch. n_accepted = j.
        // Bonus = verify_argmax[n_accepted] (target's prediction at
        // the slot where the chain broke, or beyond the last accepted
        // draft if all were accepted).
        let mut n_accepted = 0usize;
        steps += 1;
        // accept-prefix iterates over the n_drafts_used draft positions
        // we actually verified (= n_eff - 1). Slots [n_drafts_used..d)
        // were never compared; their per-slot accept stats stay 0.
        for j in 0..n_drafts_used {
            attempts_at_pos[j] += 1;
            if drafts[j] != verify_argmax[j] {
                break;
            }
            accepts_at_pos[j] += 1;
            accepted_total += 1;
            n_accepted += 1;
            emitted.push(drafts[j]);
            if emitted.len() >= tokens {
                break 'outer;
            }
            if stops.contains(&drafts[j]) {
                // Emit-through-stop; halt.
                break 'outer;
            }
        }
        // Bonus is target's prediction at the slot where the chain
        // broke (or the slot beyond the last accepted draft if all
        // accepted).
        let bonus_tok = verify_argmax[n_accepted];

        // ---- Append target_ctx with hidden_capture columns ----
        // Per H5.3a contract: hidden_capture[n] (in [N, K, H] layout
        // post-v0.71) holds K-stacked target hiddens for verify
        // position n. We append columns 0..=n_accepted (carry +
        // accepted drafts) at absolute positions
        // drafter_pos..drafter_pos+n_accepted+1. Bonus position
        // (n_accepted+1 in verify) is NOT yet committed; it'll be
        // appended on the NEXT outer iter when bonus becomes carry.
        //
        // **v0.74.3** Batched commit: gather all columns into one
        // command buffer + one commit/wait via
        // `append_target_ctx_columns_now`. The per-column `_now`
        // variant created N CPU/GPU sync points per outer step; at
        // α_chain≈5.2 typical that's ~6 waits collapsed to 1.
        let mut append_columns: Vec<(qwen_llm::metal::MetalTensor, u32)> =
            Vec::with_capacity(n_accepted + 1);
        for n_idx in 0..=n_accepted {
            let n_slot = verify_scratch.hidden_capture_n_slot(n_idx as u32);
            let absolute_pos = drafter_pos + n_idx as u32;
            append_columns.push((n_slot, absolute_pos));
        }
        let columns_refs: Vec<(&qwen_llm::metal::MetalTensor, u32)> =
            append_columns.iter().map(|(t, p)| (t, *p)).collect();
        let t_acct = Instant::now();
        decoder
            .session
            .append_target_ctx_columns_now(&ctx, &columns_refs, n_target_features)
            .context("append packed ctx columns")?;
        acct_append_ms += t_acct.elapsed().as_secs_f64() * 1e3;

        // ---- Restore on partial accept ----
        // n_keep = 1 + n_accepted (carry + accepted drafts; bonus
        // position not yet committed). On FULL accept (n_accepted=D,
        // i.e. n_keep == N), rollback is a no-op: we kept all N
        // verify positions, so there's nothing to roll back. The
        // restore primitive is safe at n_keep=N (it would just blit
        // the latest checkpoint slot into itself + write the same
        // kv_n_pos back) but that's pure overhead — one BlitEncoder
        // commit + GPU wait + per-GDN-layer ckpt blits worth of work.
        // **v0.74.3** Skip restore entirely on full accept; reviewer's
        // round-2 lever item ("skip restore blits on full accept").
        // High α (which is typical for code prompts: α_pos1=1.000) makes
        // this fire often.
        let n_keep = (n_accepted + 1) as u32;
        let n_full = n_eff as u32; // adaptive-N: rollback boundary is n_eff, not n_block
        if n_keep < n_full {
            let t_acct = Instant::now();
            qwen_llm::metal_dflash::encode_restore_after_partial_accept_inner(
                decoder.base,
                &verify_scratch,
                n_keep,
                drafter_pos,
                &mut target_session,
                Some(n_eff as u32), // adaptive-N: same n_eff as the verify call
            )
            .context("restore_after_partial_accept")?;
            acct_restore_ms += t_acct.elapsed().as_secs_f64() * 1e3;
            restore_calls += 1;
        }

        // ---- Advance cursors ----
        processed_pos += 1 + n_accepted as u32;
        carry_tok = bonus_tok;
    }

    let decode_ms = t_decode.elapsed().as_secs_f64() * 1e3;
    let total_ms = t_prefill.elapsed().as_secs_f64() * 1e3;

    // ---------- Apples-to-apples DFlash=off baseline ----------
    let mut ref_emitted: Vec<i32> = Vec::with_capacity(tokens);
    let (ref_prefill_ms, ref_decode_ms, ref_total_ms) = if !skip_equivalence_check {
        eprintln!("[dflash] running DFlash=off greedy baseline for comparison...");
        let mut ref_session = MetalSession::fresh(&ctx, &mm, cap).context("ref session")?;
        let t_ref_total = Instant::now();
        let t_ref_prefill = Instant::now();
        // v0.75.1: packed multi-token prefill (no hidden capture).
        let mut ref_layer_scratch = MetalDFlashLayerMajorScratch::fresh_prefill(&ctx, &mm, 16)
            .context("ref layer scratch")?;
        let last_logits_ref = prefill_tokens_with_multi_hidden(
            &mf,
            &prompt_ids,
            0,
            &mut ref_session,
            &mut ref_layer_scratch,
            &[],
            None,
        )?;
        let ref_prefill_ms = t_ref_prefill.elapsed().as_secs_f64() * 1e3;
        let mut next_tok = argmax_i32(&last_logits_ref);
        let mut pos = (n_prompt - 1) as u32;
        let t_ref_decode = Instant::now();
        for _ in 0..tokens {
            ref_emitted.push(next_tok);
            if stops.contains(&next_tok) || ref_emitted.len() >= tokens {
                break;
            }
            pos += 1;
            let logits = mf.single_token(next_tok, pos, &mut ref_session)?;
            next_tok = argmax_i32(&logits);
        }
        let ref_decode_ms = t_ref_decode.elapsed().as_secs_f64() * 1e3;
        let ref_total_ms = t_ref_total.elapsed().as_secs_f64() * 1e3;
        (ref_prefill_ms, ref_decode_ms, ref_total_ms)
    } else {
        (0.0, 0.0, 0.0)
    };

    // ---------- Report ----------
    eprintln!();
    eprintln!("[dflash] === results ===");
    eprintln!("[dflash] generated {} tokens", emitted.len());
    eprintln!(
        "[dflash] prefill {prefill_ms:.1} ms, decode {decode_ms:.1} ms, total {total_ms:.1} ms"
    );
    eprintln!(
        "[dflash]   throughput: decode-only {:.2} t/s | total {:.2} t/s",
        emitted.len() as f64 / (decode_ms / 1000.0),
        emitted.len() as f64 / (total_ms / 1000.0),
    );

    let alpha_chain = if steps > 0 {
        accepted_total as f64 / steps as f64
    } else {
        0.0
    };
    let mean_emitted_per_step = 1.0 + alpha_chain;
    let alpha_pos1 = if attempts_at_pos.first().copied().unwrap_or(0) > 0 {
        accepts_at_pos[0] as f64 / attempts_at_pos[0] as f64
    } else {
        0.0
    };

    // H5.6 M1c: per-step component accounting (means over all outer steps;
    // see terminal-step caveat above — use --tokens >= 256 for economics).
    if steps > 0 {
        let s = steps as f64;
        let acct_sum = acct_draft_ms + acct_verify_ms + acct_append_ms + acct_restore_ms;
        eprintln!();
        eprintln!("[dflash] === step accounting (H5.6 M1c) ===");
        eprintln!(
            "[dflash] per-step means over {steps} steps: draft {:.1} ms | verify {:.1} ms | append {:.1} ms | restore {:.1} ms | unaccounted {:.1} ms | TOTAL {:.1} ms",
            acct_draft_ms / s,
            acct_verify_ms / s,
            acct_append_ms / s,
            acct_restore_ms / s,
            (decode_ms - acct_sum).max(0.0) / s,
            decode_ms / s,
        );
    }

    eprintln!();
    eprintln!("[dflash] === acceptance ===");
    eprintln!(
        "[dflash] outer steps={steps}  accepted_drafts={accepted_total}  \
         drafter_calls={drafter_calls}  verify_calls={verify_calls}  restore_calls={restore_calls}"
    );
    // v0.76 adaptive-N step distribution.
    eprintln!(
        "[dflash] n_policy={n_policy:?}  step distribution: \
         spec16={spec16_steps} spec8={spec8_steps} spec4={spec4_steps} off={off_steps}  \
         (spec_disabled={spec_disabled} terminally)"
    );
    eprintln!(
        "[dflash] α_chain = {accepted_total} / {steps} = {alpha_chain:.3} drafts/step (max D={d})"
    );
    eprintln!("[dflash] mean_emitted_per_step = 1 + α_chain = {mean_emitted_per_step:.3}");
    eprintln!(
        "[dflash] α_pos1 (rank-1 hit at first draft slot) = {} / {} = {alpha_pos1:.3}",
        accepts_at_pos[0], attempts_at_pos[0]
    );
    eprintln!("[dflash] per-position α (conditional on reaching that slot):");
    for j in 0..d {
        let attempts = attempts_at_pos[j];
        let accepts = accepts_at_pos[j];
        let alpha_j = if attempts > 0 {
            accepts as f64 / attempts as f64
        } else {
            0.0
        };
        eprintln!("[dflash]   position {j:2}: {accepts:>4}/{attempts:>4} = {alpha_j:.3}");
    }

    if profile {
        eprintln!();
        eprintln!("[dflash] === drafter phase profile (v0.72.3) ===");
        let timings = decoder.session.take_phase_timings();
        if timings.is_empty() {
            eprintln!("[dflash]   (no timings — was profile flag enabled?)");
        } else {
            // Aggregate same-name phases across all outer steps + layers.
            use std::collections::BTreeMap;
            let mut agg: BTreeMap<String, (f64, u32)> = BTreeMap::new();
            for (name, ms) in &timings {
                let e = agg.entry(name.clone()).or_insert((0.0, 0));
                e.0 += ms;
                e.1 += 1;
            }
            let has_phase3_split = agg.keys().any(|name| name.starts_with("phase3_split_"));
            let total_gpu_ms: f64 = agg
                .iter()
                .filter(|(name, _)| !name.starts_with("phase3_split_"))
                .map(|(_, (s, _))| *s)
                .sum();
            // Sort by descending sum.
            let mut sorted: Vec<_> = agg.iter().collect();
            sorted.sort_by(|a, b| b.1.0.partial_cmp(&a.1.0).unwrap());
            for (name, (sum_ms, count)) in &sorted {
                let avg = *sum_ms / (*count as f64);
                let pct = if total_gpu_ms > 0.0 {
                    100.0 * *sum_ms / total_gpu_ms
                } else {
                    0.0
                };
                eprintln!(
                    "[dflash]   {name:>40}  sum={sum_ms:>8.2} ms  ({pct:>5.1}%)  \
                     n={count:>4}  avg={avg:>6.2} ms"
                );
            }
            if has_phase3_split {
                eprintln!(
                    "[dflash]   (phase3_split_* rows are attribution-only and excluded from TOTAL_DRAFTER_GPU)"
                );
            }
            eprintln!(
                "[dflash]   {:>40}  sum={total_gpu_ms:>8.2} ms  (sum-of-phases drafter GPU time)",
                "TOTAL_DRAFTER_GPU"
            );
            eprintln!(
                "[dflash]   {:>40}  sum={:>8.2} ms  (drafter wall = phases + per-commit overhead)",
                "TOTAL_DECODE_WALL", decode_ms
            );
        }
    }

    if !skip_equivalence_check {
        eprintln!();
        eprintln!("[dflash] === DFlash=off baseline ===");
        eprintln!(
            "[dflash] no-spec ref: prefill {ref_prefill_ms:.1} ms, decode {ref_decode_ms:.1} ms, total {ref_total_ms:.1} ms"
        );
        eprintln!(
            "[dflash]   throughput: decode-only {:.2} t/s | total {:.2} t/s",
            ref_emitted.len() as f64 / (ref_decode_ms / 1000.0),
            ref_emitted.len() as f64 / (ref_total_ms / 1000.0),
        );
        let speedup_total = ref_total_ms / total_ms;
        let speedup_decode = ref_decode_ms / decode_ms;
        eprintln!();
        eprintln!("[dflash] === SPEEDUP vs DFlash=off ===");
        eprintln!("[dflash]   total wall: {ref_total_ms:.1} / {total_ms:.1} = {speedup_total:.3}×");
        eprintln!(
            "[dflash]   decode-only: {ref_decode_ms:.1} / {decode_ms:.1} = {speedup_decode:.3}×"
        );

        // ---------- Greedy equivalence check ----------
        let n_show = emitted.len().min(ref_emitted.len()).min(16);
        if emitted == ref_emitted {
            eprintln!();
            eprintln!(
                "[dflash] greedy equivalence: PASS — {} tokens identical to DFlash=off",
                emitted.len()
            );
        } else {
            eprintln!();
            eprintln!("[dflash] greedy equivalence: FAIL");
            eprintln!("[dflash]   dflash:  {:?}", &emitted[..n_show]);
            eprintln!("[dflash]   no-spec: {:?}", &ref_emitted[..n_show]);
            // v0.501: print the first divergence with context — the
            // first-16 prefix is often identical (near-tie argmax flips
            // happen mid-generation) and the index is the evidence that
            // matters for correctness triage.
            let div = emitted
                .iter()
                .zip(ref_emitted.iter())
                .position(|(a, b)| a != b);
            match div {
                Some(i) => {
                    let lo = i.saturating_sub(4);
                    let hi_a = (i + 4).min(emitted.len());
                    let hi_b = (i + 4).min(ref_emitted.len());
                    eprintln!(
                        "[dflash]   first divergence at index {i}: dflash[{lo}..{hi_a}]={:?} no-spec[{lo}..{hi_b}]={:?}",
                        &emitted[lo..hi_a],
                        &ref_emitted[lo..hi_b]
                    );
                }
                None => {
                    eprintln!(
                        "[dflash]   no positional divergence — length mismatch: dflash={} no-spec={}",
                        emitted.len(),
                        ref_emitted.len()
                    );
                }
            }
            return Err(anyhow!(
                "DFlash decode produced different tokens than DFlash=off greedy"
            ));
        }
    }

    Ok(())
}

fn run_pp(args: PpArgs) -> Result<()> {
    let PpArgs {
        model,
        n_prompt,
        prompt,
        file,
        messages,
        messages_max,
        messages_preserve_thinking,
        messages_strip_thinking,
        messages_no_generation_prompt,
        runs,
        no_warmup,
        prefill_chunk,
        with_tail,
        seed,
        output,
    } = args;
    if runs == 0 {
        return Err(anyhow!("--runs must be >= 1"));
    }
    let json_mode = matches!(output, OutputFormat::Json);
    macro_rules! text_log { ($($t:tt)*) => { if !json_mode { eprintln!($($t)*); } } }

    let runtime = Runtime::metal().context("init Runtime")?;
    text_log!("[pp] device: {}", runtime.describe());
    let power = capture_power_snapshot();
    text_log!("[pp] power: {}", power_snapshot_summary(power.as_ref()));

    let loaded = runtime
        .load_model(&model)
        .with_context(|| format!("load {}", model.display()))?;
    let ctx = loaded.context();
    let g = loaded.gguf();
    let mm = loaded.metal_model();
    let arch = loaded.arch();

    let (ids, source_label) = if let Some(prompt) = prompt {
        let tok = loaded.tokenizer().context("open tokenizer")?;
        let ids = tok.encode(&prompt, false).context("tokenize prompt")?;
        (ids, format!("text prompt ({} chars)", prompt.len()))
    } else if let Some(path) = file {
        let prompt =
            std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
        let tok = loaded.tokenizer().context("open tokenizer")?;
        let ids = tok.encode(&prompt, false).context("tokenize prompt file")?;
        (
            ids,
            format!("file prompt:{} ({} chars)", path.display(), prompt.len()),
        )
    } else if let Some(path) = messages {
        let prompt = load_messages_prompt(
            &path,
            messages_max,
            messages_thinking_mode(messages_preserve_thinking, messages_strip_thinking),
            !messages_no_generation_prompt,
        )?;
        let tok = loaded.tokenizer().context("open tokenizer")?;
        let ids = tok
            .encode(&prompt, false)
            .context("tokenize rendered messages prompt")?;
        (
            ids,
            format!("messages:{} ({} chars)", path.display(), prompt.len()),
        )
    } else {
        if n_prompt == 0 {
            return Err(anyhow!("--n-prompt must be >= 1"));
        }
        (
            synthetic_prompt_ids(n_prompt, arch.vocab_size, seed),
            format!("synthetic token ids (seed={seed})"),
        )
    };
    if ids.is_empty() {
        return Err(anyhow!("prompt tokenized to an empty sequence"));
    }

    let prefill_chunk =
        prefill_chunk.unwrap_or_else(|| default_prefill_chunk(arch.kind, ids.len()));
    if prefill_chunk == 0 {
        return Err(anyhow!("--prefill-chunk must be >= 1"));
    }

    let mf = loaded.forward();
    let cap = ids.len() + 16;
    text_log!(
        "[pp] model={} source={} n_prompt={} runs={} chunk={} tail={}",
        model.display(),
        source_label,
        ids.len(),
        runs,
        prefill_chunk,
        if with_tail { "final-logits" } else { "skip" }
    );
    if !json_mode {
        print_prefill_lowering_summary(mm);
    }

    let residency_guard = if env_flag_enabled("QWEN_PP_RESIDENCY_SET")
        && arch.kind == qwen_llm::model::ArchKind::Moe
    {
        let (guard, allocations, bytes) =
            pp_register_moe_residency_set(ctx, &mf).context("register MoE residency set")?;
        text_log!(
            "[pp] residency: registered {allocations} MoE expert-bank allocations ({:.2} GiB tracked)",
            bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        );
        Some(guard)
    } else {
        None
    };

    if residency_guard.is_some() && env_flag_enabled("QWEN_PP_WARM_MOE_BANKS") {
        text_log!("[pp] residency-set active; skipping QWEN_PP_WARM_MOE_BANKS touch pass");
    } else if env_flag_enabled("QWEN_PP_WARM_MOE_BANKS")
        && arch.kind == qwen_llm::model::ArchKind::Moe
    {
        let touched =
            pp_warm_moe_weight_banks(ctx, &mf).context("warm grouped MoE weight banks")?;
        text_log!("[pp] warmup: touched {touched} MoE expert-bank tensors via GPU residency pass");
    }

    if !no_warmup {
        let mut s = loaded
            .create_sequence(SequenceConfig::new(cap))
            .context("session warmup")?;
        let mut scratch = fresh_prefill_scratch_for_prompt(ctx, mm, prefill_chunk, ids.len())
            .context("warmup prefill scratch")?;
        if with_tail {
            let _ = prefill_tokens_with_multi_hidden(
                &mf,
                &ids,
                0,
                s.metal_session_mut(),
                &mut scratch,
                &[],
                None,
            )
            .context("warmup prefill with tail")?;
        } else {
            let _ = prefill_tokens_prompt_only_profiled(
                &mf,
                &ids,
                0,
                s.metal_session_mut(),
                &mut scratch,
            )
            .context("warmup prompt-only prefill")?;
        }
    }

    let mut wall_samples = Vec::with_capacity(runs);
    let mut gpu_samples = Vec::with_capacity(runs);
    let mut ts_samples = Vec::with_capacity(runs);
    for run_idx in 0..runs {
        let mut s = loaded
            .create_sequence(SequenceConfig::new(cap))
            .context("session run")?;
        let mut scratch = fresh_prefill_scratch_for_prompt(ctx, mm, prefill_chunk, ids.len())
            .context("timed prefill scratch")?;

        let t0 = Instant::now();
        let gpu_ms = if with_tail {
            let (_, gpu_ms) = prefill_tokens_with_multi_hidden_profiled(
                &mf,
                &ids,
                0,
                s.metal_session_mut(),
                &mut scratch,
                &[],
                None,
            )
            .context("timed prefill with tail")?;
            gpu_ms
        } else {
            prefill_tokens_prompt_only_profiled(&mf, &ids, 0, s.metal_session_mut(), &mut scratch)
                .context("timed prompt-only prefill")?
        };
        let wall_ms = t0.elapsed().as_secs_f64() * 1e3;
        let ts = ids.len() as f64 * 1000.0 / wall_ms;
        wall_samples.push(wall_ms);
        gpu_samples.push(gpu_ms);
        ts_samples.push(ts);
        text_log!(
            "[pp] run {:>2}: wall {:>8.1} ms  gpu {:>8.1} ms  {:>7.2} t/s",
            run_idx + 1,
            wall_ms,
            gpu_ms,
            ts
        );
    }

    let wall_mean = sample_mean(&wall_samples);
    let gpu_mean = sample_mean(&gpu_samples);
    let ts_mean = sample_mean(&ts_samples);
    let ts_sd = sample_stdev(&ts_samples);

    if json_mode {
        let (commit, dirty) = qwen_build_identity();
        let row = BenchRow {
            schema_version: BENCH_SCHEMA_VERSION,
            engine: "qwen-llm",
            build_commit: commit,
            build_dirty: dirty,
            build_identity: recorded_build_identity(),
            test_time: utc_iso8601_now(),
            model_filename: model.display().to_string(),
            model_size: model_weight_bytes(g),
            model_n_params: g
                .get_u64("general.parameter_count")
                .unwrap_or_else(|| g.tensors.iter().map(|t| t.n_elements()).sum()),
            arch_kind: match arch.kind {
                qwen_llm::model::ArchKind::Dense => "dense",
                qwen_llm::model::ArchKind::Moe => "moe",
            },
            test: format!("pp{}", ids.len()),
            n_tokens: ids.len(),
            n_repetitions: runs,
            avg_ts: ts_mean,
            stddev_ts: ts_sd,
            samples_ts: ts_samples.clone(),
            samples_ns: wall_samples.iter().map(|w| (*w * 1e6) as u64).collect(),
            avg_ns: (wall_mean * 1e6) as u64,
            avg_compute_ns: Some((wall_mean * 1e6) as u64),
            avg_session_alloc_ns: None,
            avg_scratch_alloc_ns: None,
            avg_gpu_ns: Some((gpu_mean * 1e6) as u64),
            kernel_trace_command_buffers_per_token: None,
            kernel_trace_encoders_per_token: None,
            kernel_trace_concurrent_encoders_per_token: None,
            kernel_trace_dispatches_per_token: None,
            // pp is not a steady-state-bandwidth measurement, so we don't
            // emit a derived GB/s for prefill rows. Digest tools can compute
            // their own if they want, but the canonical bandwidth comparison
            // is on decode.
            decode_gb_per_s: None,
            prefill_chunk: Some(prefill_chunk),
            decode_mode: None,
            prefill_mode: Some("packed"),
            power,
            qwen_env: capture_qwen_env(),
        };
        // Wrap in an array to match `llama-bench -o json`.
        let arr = vec![row];
        let json = serde_json::to_string(&arr).context("serialize pp bench row")?;
        println!("{json}");
    } else {
        eprintln!();
        eprintln!("[pp] === results ===");
        eprintln!(
            "[pp] prompt: {} tokens in {:.1} ms avg = {:.2} ms/token = {:.2} +/- {:.2} t/s",
            ids.len(),
            wall_mean,
            wall_mean / ids.len() as f64,
            ts_mean,
            ts_sd
        );
        eprintln!(
            "[pp] gpu:    {:.1} ms avg = {:.2} ms/token = {:.1}% of wall",
            gpu_mean,
            gpu_mean / ids.len() as f64,
            100.0 * gpu_mean / wall_mean.max(1e-9)
        );
        eprintln!(
            "[pp] note: session and scratch allocation are outside the timed interval; tail={}.",
            if with_tail {
                "included"
            } else {
                "skipped to match llama-bench pp logits policy"
            }
        );
    }

    Ok(())
}

fn run_pp_ffn_ab_once(
    ctx: &MetalContext,
    mm: &MetalModel,
    mf: &MetalForward<'_>,
    ids: &[i32],
    prefill_chunk: usize,
    fused: bool,
) -> Result<(f64, f64, f64)> {
    let mut s = MetalSession::fresh(ctx, mm, ids.len() + 16).context("session run")?;
    let mut scratch = fresh_prefill_scratch_for_prompt(ctx, mm, prefill_chunk, ids.len())
        .context("timed prefill scratch")?;

    let t0 = Instant::now();
    let gpu_ms = with_prefill_dense_ffn_fused_swiglu_q4_override(fused, || {
        prefill_tokens_prompt_only_profiled(mf, ids, 0, &mut s, &mut scratch)
    })
    .context("timed prompt-only prefill")?;
    let wall_ms = t0.elapsed().as_secs_f64() * 1e3;
    let ts = ids.len() as f64 * 1000.0 / wall_ms;
    Ok((wall_ms, gpu_ms, ts))
}

fn run_pp_ffn_ab(args: PpFfnAbArgs) -> Result<()> {
    let PpFfnAbArgs {
        model,
        n_prompt,
        prefill_chunk,
        pairs,
        no_warmup,
        seed,
    } = args;
    if n_prompt == 0 {
        return Err(anyhow!("--n-prompt must be >= 1"));
    }
    if pairs == 0 {
        return Err(anyhow!("--pairs must be >= 1"));
    }

    let ctx = MetalContext::new().context("init MetalContext")?;
    eprintln!("[pp-ffn-ab] device: {}", ctx.describe());
    let power = capture_power_snapshot();
    eprintln!(
        "[pp-ffn-ab] power: {}",
        power_snapshot_summary(power.as_ref())
    );

    let g = GgufFile::open(&model).with_context(|| format!("open {}", model.display()))?;
    let m = Model::from_gguf(&g).context("parse model arch from gguf")?;
    if m.arch.kind != qwen_llm::model::ArchKind::Dense {
        return Err(anyhow!("pp-ffn-ab is a dense FFN harness; got MoE model"));
    }
    let mm = MetalModel::load(&ctx, &g, &m).context("metal-load model weights")?;
    let mf = MetalForward::new(&ctx, &mm);
    let ids = synthetic_prompt_ids(n_prompt, m.arch.vocab_size, seed);
    let prefill_chunk =
        prefill_chunk.unwrap_or_else(|| default_prefill_chunk(m.arch.kind, ids.len()));
    if prefill_chunk == 0 {
        return Err(anyhow!("--prefill-chunk must be >= 1"));
    }

    eprintln!(
        "[pp-ffn-ab] model={} n_prompt={} pairs={} chunk={} warmup={}",
        model.display(),
        ids.len(),
        pairs,
        prefill_chunk,
        if no_warmup { "skip" } else { "base+fused" }
    );
    print_prefill_lowering_summary(&mm);

    if !no_warmup {
        for fused in [false, true] {
            let _ = run_pp_ffn_ab_once(&ctx, &mm, &mf, &ids, prefill_chunk, fused)
                .with_context(|| format!("warmup fused={fused}"))?;
        }
    }

    println!("pair\torder\tvariant\twall_ms\tgpu_ms\ttokens_s");
    for pair_idx in 0..pairs {
        let order = if pair_idx % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        };
        for (order_idx, fused) in order.into_iter().enumerate() {
            let (wall_ms, gpu_ms, ts) =
                run_pp_ffn_ab_once(&ctx, &mm, &mf, &ids, prefill_chunk, fused)
                    .with_context(|| format!("timed pair={pair_idx} fused={fused}"))?;
            println!(
                "{pair_idx}\t{order_idx}\t{}\t{wall_ms:.1}\t{gpu_ms:.1}\t{ts:.2}",
                if fused { "fused" } else { "base" }
            );
        }
    }

    Ok(())
}

fn run_tg(args: TgArgs) -> Result<()> {
    let TgArgs {
        model,
        n_gen,
        runs,
        no_warmup,
        pipelined,
        concurrent_gdn_proj,
        seed,
        output,
    } = args;
    if runs == 0 {
        return Err(anyhow!("--runs must be >= 1"));
    }
    if n_gen == 0 {
        return Err(anyhow!("--n-gen must be >= 1"));
    }
    if pipelined && concurrent_gdn_proj {
        return Err(anyhow!(
            "--pipelined and --concurrent-gdn-proj are separate bench-only decode experiments; use one at a time"
        ));
    }
    let json_mode = matches!(output, OutputFormat::Json);
    macro_rules! text_log { ($($t:tt)*) => { if !json_mode { eprintln!($($t)*); } } }
    let trace_counts = env_flag_enabled("QWEN_DECODE_TRACE_COUNTS");

    let runtime = Runtime::metal().context("init Runtime")?;
    text_log!("[tg] device: {}", runtime.describe());
    let power = capture_power_snapshot();
    text_log!("[tg] power: {}", power_snapshot_summary(power.as_ref()));

    let loaded = runtime
        .load_model(&model)
        .with_context(|| format!("load {}", model.display()))?;
    let ctx = loaded.context();
    let g = loaded.gguf();
    let mm = loaded.metal_model();
    let arch = loaded.arch();
    let mf = loaded.forward();

    let vocab = arch.vocab_size.max(1);
    // xorshift64* with the same `seed` controls the random tokens across
    // reps, so the bench is fully deterministic. Single shared state so
    // rep N+1 isn't reading the same tokens as rep N.
    let mut rng_state = if seed == 0 { 1u64 } else { seed };
    let mut next_rand_tok = || -> i32 {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        (rng_state % vocab as u64) as i32
    };

    text_log!(
        "[tg] model={} n_gen={} runs={} seed={} mode={}{}",
        model.display(),
        n_gen,
        runs,
        seed,
        if pipelined { "pipelined" } else { "default" },
        if concurrent_gdn_proj {
            "+concurrent_gdn"
        } else {
            ""
        }
    );

    let cap = n_gen + 16;
    let ids_ping = if pipelined {
        Some([
            MetalTensor::zeros_f32(ctx, vec![1]).context("tg pipelined ids ping0")?,
            MetalTensor::zeros_f32(ctx, vec![1]).context("tg pipelined ids ping1")?,
        ])
    } else {
        None
    };
    let argmax_ping = if pipelined {
        Some([
            MetalTensor::zeros_f32(ctx, vec![1]).context("tg pipelined argmax ping0")?,
            MetalTensor::zeros_f32(ctx, vec![1]).context("tg pipelined argmax ping1")?,
        ])
    } else {
        None
    };
    let run_once = |first_tok: i32,
                    ranges: &mut dyn FnMut() -> i32|
     -> Result<(f64, f64, Option<KernelTraceCounters>)> {
        let mut s = loaded
            .create_sequence(SequenceConfig::new(cap))
            .context("tg session")?;
        let _kernel_trace_guard = trace_counts.then(kernel_trace_begin);
        if !pipelined {
            let t0 = Instant::now();
            let mut tok = first_tok;
            // GPU-busy accumulator across the N decode steps. We report wall as
            // the headline t/s (matches lcpp's printer), and surface gpu as an
            // engine-specific field on the JSON row.
            let mut gpu_ms_acc = 0.0;
            for pos in 0..n_gen {
                // Use `single_token_argmax_profiled` (dispatches dense/MoE
                // internally) so we can sum per-step GPU time. The argmax i32 is
                // discarded; the next input is drawn from the seeded RNG, matching
                // lcpp's `test_gen` (random tokens, no logits coupling).
                let (_argmax, prof) = if concurrent_gdn_proj {
                    if mm.arch.kind == qwen_llm::model::ArchKind::Dense {
                        mf.single_token_argmax_profiled_concurrent_gdn_dense(
                            tok,
                            pos as u32,
                            s.metal_session_mut(),
                        )?
                    } else {
                        mf.single_token_argmax_profiled_concurrent_gdn_moe(
                            tok,
                            pos as u32,
                            s.metal_session_mut(),
                        )?
                    }
                } else {
                    mf.single_token_argmax_profiled(tok, pos as u32, s.metal_session_mut())?
                };
                gpu_ms_acc += prof.gpu_kernel_ms;
                tok = ranges();
            }
            let wall_ms = t0.elapsed().as_secs_f64() * 1e3;
            let counts = trace_counts.then(kernel_trace_snapshot);
            return Ok((wall_ms, gpu_ms_acc, counts));
        }

        let ids_ping = ids_ping.as_ref().expect("pipelined ids");
        let argmax_ping = argmax_ping.as_ref().expect("pipelined argmax");
        let mut inputs = Vec::with_capacity(n_gen.max(1));
        inputs.push(first_tok);
        for _ in 1..n_gen {
            inputs.push(ranges());
        }
        let _unused_next = ranges();

        let t0 = Instant::now();
        let mut gpu_ms_acc = 0.0;
        unsafe {
            let ptr = ids_ping[0].buffer.contents().as_ptr() as *mut i32;
            *ptr = inputs[0];
        }
        let first_cmd = ctx
            .queue
            .commandBuffer()
            .context("tg pipelined first command buffer")?;
        let first_enc = KernelEncoder::begin(&first_cmd);
        mf.encode_single_token_argmax(
            &first_enc,
            0,
            s.metal_session_mut(),
            &ids_ping[0],
            &argmax_ping[0],
        )?;
        first_enc.end();
        first_cmd.commit();
        let mut pending_cmd = first_cmd;
        let mut pending_slot = 0usize;

        for (pos, tok) in inputs.iter().copied().enumerate().skip(1) {
            let next_slot = pending_slot ^ 1;
            let next_cmd = ctx
                .queue
                .commandBuffer()
                .context("tg pipelined next command buffer")?;
            let next_enc = KernelEncoder::begin(&next_cmd);
            mf.encode_single_token_argmax(
                &next_enc,
                pos as u32,
                s.metal_session_mut(),
                &ids_ping[next_slot],
                &argmax_ping[next_slot],
            )?;
            next_enc.end();

            pending_cmd.waitUntilCompleted();
            gpu_ms_acc += (pending_cmd.GPUEndTime() - pending_cmd.GPUStartTime()) * 1e3;
            unsafe {
                let ptr = ids_ping[next_slot].buffer.contents().as_ptr() as *mut i32;
                *ptr = tok;
            }
            next_cmd.commit();
            pending_cmd = next_cmd;
            pending_slot = next_slot;
        }

        pending_cmd.waitUntilCompleted();
        gpu_ms_acc += (pending_cmd.GPUEndTime() - pending_cmd.GPUStartTime()) * 1e3;
        let wall_ms = t0.elapsed().as_secs_f64() * 1e3;
        let counts = trace_counts.then(kernel_trace_snapshot);
        Ok((wall_ms, gpu_ms_acc, counts))
    };

    if !no_warmup {
        let first = next_rand_tok();
        let _ = run_once(first, &mut next_rand_tok).context("tg warmup")?;
    }

    let mut wall_samples: Vec<f64> = Vec::with_capacity(runs);
    let mut gpu_samples: Vec<f64> = Vec::with_capacity(runs);
    let mut ts_samples: Vec<f64> = Vec::with_capacity(runs);
    let mut trace_samples: Vec<KernelTraceCounters> = Vec::with_capacity(runs);
    for run_idx in 0..runs {
        let first = next_rand_tok();
        let (wall_ms, gpu_ms, trace) = run_once(first, &mut next_rand_tok).context("tg run")?;
        let ts = n_gen as f64 * 1000.0 / wall_ms;
        wall_samples.push(wall_ms);
        gpu_samples.push(gpu_ms);
        ts_samples.push(ts);
        if let Some(trace) = trace {
            trace_samples.push(trace);
        }
        text_log!(
            "[tg] run {:>2}: wall {:>8.1} ms  gpu {:>8.1} ms  {:>7.2} t/s",
            run_idx + 1,
            wall_ms,
            gpu_ms,
            ts
        );
    }

    let wall_mean = sample_mean(&wall_samples);
    let gpu_mean = sample_mean(&gpu_samples);
    let ts_mean = sample_mean(&ts_samples);
    let ts_sd = sample_stdev(&ts_samples);
    let trace_per_token = if trace_counts && !trace_samples.is_empty() {
        let denom = (trace_samples.len() * n_gen) as f64;
        let encoders: u64 = trace_samples.iter().map(|t| t.encoders).sum();
        let concurrent_encoders: u64 = trace_samples.iter().map(|t| t.concurrent_encoders).sum();
        let dispatches: u64 = trace_samples.iter().map(|t| t.dispatches).sum();
        Some((
            1.0,
            encoders as f64 / denom,
            concurrent_encoders as f64 / denom,
            dispatches as f64 / denom,
        ))
    } else {
        None
    };

    if json_mode {
        let (commit, dirty) = qwen_build_identity();
        let arch_kind_str: &'static str = match arch.kind {
            qwen_llm::model::ArchKind::Dense => "dense",
            qwen_llm::model::ArchKind::Moe => "moe",
        };
        let model_size = model_weight_bytes(g);
        let model_n_params = g
            .get_u64("general.parameter_count")
            .unwrap_or_else(|| g.tensors.iter().map(|t| t.n_elements()).sum());
        // Bandwidth is only meaningful for dense; MoE active-param accounting
        // lives outside this schema today.
        let gb_per_s = if matches!(arch.kind, qwen_llm::model::ArchKind::Dense)
            && model_size > 0
            && wall_mean > 0.0
        {
            Some((model_size as f64 / 1e9) / (wall_mean / 1000.0 / n_gen as f64))
        } else {
            None
        };
        let row = BenchRow {
            schema_version: BENCH_SCHEMA_VERSION,
            engine: "qwen-llm",
            build_commit: commit,
            build_dirty: dirty,
            build_identity: recorded_build_identity(),
            test_time: utc_iso8601_now(),
            model_filename: model.display().to_string(),
            model_size,
            model_n_params,
            arch_kind: arch_kind_str,
            test: format!("tg{}", n_gen),
            n_tokens: n_gen,
            n_repetitions: runs,
            avg_ts: ts_mean,
            stddev_ts: ts_sd,
            samples_ts: ts_samples.clone(),
            samples_ns: wall_samples.iter().map(|w| (*w * 1e6) as u64).collect(),
            avg_ns: (wall_mean * 1e6) as u64,
            avg_compute_ns: Some((wall_mean * 1e6) as u64),
            avg_session_alloc_ns: None,
            avg_scratch_alloc_ns: None,
            avg_gpu_ns: Some((gpu_mean * 1e6) as u64),
            kernel_trace_command_buffers_per_token: trace_per_token.map(|t| t.0),
            kernel_trace_encoders_per_token: trace_per_token.map(|t| t.1),
            kernel_trace_concurrent_encoders_per_token: trace_per_token.map(|t| t.2),
            kernel_trace_dispatches_per_token: trace_per_token.map(|t| t.3),
            decode_gb_per_s: gb_per_s,
            prefill_chunk: None,
            decode_mode: Some(if pipelined {
                "apples-lcpp-pipelined"
            } else if concurrent_gdn_proj {
                "apples-lcpp-concurrent-gdn"
            } else {
                "apples-lcpp"
            }),
            prefill_mode: None,
            power,
            qwen_env: capture_qwen_env(),
        };
        let arr = vec![row];
        let json = serde_json::to_string_pretty(&arr).context("serialize tg bench row")?;
        println!("{json}");
    } else {
        eprintln!();
        eprintln!("[tg] === results ===");
        eprintln!(
            "[tg] gen: {n_gen} tokens × {runs} runs, avg {:.1} ms = {:.2} +/- {:.2} t/s",
            wall_mean, ts_mean, ts_sd
        );
        eprintln!(
            "[tg] gpu: {:.1} ms avg ({:.1}% of wall)",
            gpu_mean,
            100.0 * gpu_mean / wall_mean.max(1e-9)
        );
        if let Some((cmd_buffers, encoders, concurrent_encoders, dispatches)) = trace_per_token {
            eprintln!(
                "[tg] trace: {cmd_buffers:.1} command buffers/token, \
                 {encoders:.1} encoders/token ({concurrent_encoders:.1} concurrent), \
                 {dispatches:.1} dispatches/token"
            );
        }
        eprintln!(
            "[tg] note: empty KV per rep, random tokens, no logits readback — matches `llama-bench tg{n_gen}`."
        );
        if pipelined {
            eprintln!(
                "[tg] note: bench-only CPU/GPU overlap path; commands still commit serially."
            );
        } else if concurrent_gdn_proj {
            eprintln!(
                "[tg] note: bench-only concurrent GDN front-projection path; currently MoE-only."
            );
        }
    }

    Ok(())
}

fn arch_kind_label(kind: qwen_llm::model::ArchKind) -> &'static str {
    match kind {
        qwen_llm::model::ArchKind::Dense => "dense",
        qwen_llm::model::ArchKind::Moe => "moe",
    }
}

struct SuiteRowContext {
    build_commit: &'static str,
    build_dirty: u8,
    model_filename: String,
    model_size: u64,
    model_n_params: u64,
    arch_kind: &'static str,
    power: Option<PowerSnapshot>,
    qwen_env: std::collections::BTreeMap<String, String>,
}

fn run_suite_pp_row(
    loaded: &LoadedModel,
    row_ctx: &SuiteRowContext,
    n_prompt: usize,
    runs: usize,
    no_warmup: bool,
    prefill_chunk_override: Option<usize>,
    seed: u64,
) -> Result<BenchRow> {
    if n_prompt == 0 {
        return Err(anyhow!("--pp values must be >= 1"));
    }
    let arch = loaded.arch();
    let ctx = loaded.context();
    let mm = loaded.metal_model();
    let mf = loaded.forward();
    let ids = synthetic_prompt_ids(n_prompt, arch.vocab_size, seed);
    let prefill_chunk =
        prefill_chunk_override.unwrap_or_else(|| default_prefill_chunk(arch.kind, ids.len()));
    if prefill_chunk == 0 {
        return Err(anyhow!("--prefill-chunk must be >= 1"));
    }
    let cap = ids.len() + 16;

    if !no_warmup {
        let mut seq = loaded
            .create_sequence(SequenceConfig::new(cap))
            .context("suite pp warmup session")?;
        let mut scratch = fresh_prefill_scratch_for_prompt(ctx, mm, prefill_chunk, ids.len())
            .context("suite pp warmup scratch")?;
        let _ = prefill_tokens_prompt_only_profiled(
            &mf,
            &ids,
            0,
            seq.metal_session_mut(),
            &mut scratch,
        )
        .context("suite pp warmup")?;
    }

    let mut wall_samples = Vec::with_capacity(runs);
    let mut gpu_samples = Vec::with_capacity(runs);
    let mut ts_samples = Vec::with_capacity(runs);
    let mut session_alloc_samples = Vec::with_capacity(runs);
    let mut scratch_alloc_samples = Vec::with_capacity(runs);
    for _ in 0..runs {
        let session_t0 = Instant::now();
        let mut seq = loaded
            .create_sequence(SequenceConfig::new(cap))
            .context("suite pp session")?;
        let session_alloc_ms = session_t0.elapsed().as_secs_f64() * 1e3;
        let scratch_t0 = Instant::now();
        let mut scratch = fresh_prefill_scratch_for_prompt(ctx, mm, prefill_chunk, ids.len())
            .context("suite pp scratch")?;
        let scratch_alloc_ms = scratch_t0.elapsed().as_secs_f64() * 1e3;
        seq.ensure_can_append(ids.len())
            .context("suite pp sequence capacity")?;
        let t0 = Instant::now();
        let gpu_ms = prefill_tokens_prompt_only_profiled(
            &mf,
            &ids,
            0,
            seq.metal_session_mut(),
            &mut scratch,
        )
        .context("suite pp timed prefill")?;
        let wall_ms = t0.elapsed().as_secs_f64() * 1e3;
        seq.advance_by(ids.len())
            .context("suite pp advance sequence")?;
        wall_samples.push(wall_ms);
        gpu_samples.push(gpu_ms);
        ts_samples.push(ids.len() as f64 * 1000.0 / wall_ms);
        session_alloc_samples.push(session_alloc_ms);
        scratch_alloc_samples.push(scratch_alloc_ms);
    }

    let wall_mean = sample_mean(&wall_samples);
    let gpu_mean = sample_mean(&gpu_samples);
    let session_alloc_mean = sample_mean(&session_alloc_samples);
    let scratch_alloc_mean = sample_mean(&scratch_alloc_samples);
    Ok(BenchRow {
        schema_version: BENCH_SCHEMA_VERSION,
        engine: "qwen-llm",
        build_commit: row_ctx.build_commit,
        build_dirty: row_ctx.build_dirty,
        build_identity: recorded_build_identity(),
        test_time: utc_iso8601_now(),
        model_filename: row_ctx.model_filename.clone(),
        model_size: row_ctx.model_size,
        model_n_params: row_ctx.model_n_params,
        arch_kind: row_ctx.arch_kind,
        test: format!("pp{}", ids.len()),
        n_tokens: ids.len(),
        n_repetitions: runs,
        avg_ts: sample_mean(&ts_samples),
        stddev_ts: sample_stdev(&ts_samples),
        samples_ts: ts_samples,
        samples_ns: wall_samples.iter().map(|w| (*w * 1e6) as u64).collect(),
        avg_ns: (wall_mean * 1e6) as u64,
        avg_compute_ns: Some((wall_mean * 1e6) as u64),
        avg_session_alloc_ns: Some((session_alloc_mean * 1e6) as u64),
        avg_scratch_alloc_ns: Some((scratch_alloc_mean * 1e6) as u64),
        avg_gpu_ns: Some((gpu_mean * 1e6) as u64),
        kernel_trace_command_buffers_per_token: None,
        kernel_trace_encoders_per_token: None,
        kernel_trace_concurrent_encoders_per_token: None,
        kernel_trace_dispatches_per_token: None,
        decode_gb_per_s: None,
        prefill_chunk: Some(prefill_chunk),
        decode_mode: None,
        prefill_mode: Some("packed"),
        power: row_ctx.power.clone(),
        qwen_env: row_ctx.qwen_env.clone(),
    })
}

fn run_suite_tg_row(
    loaded: &LoadedModel,
    row_ctx: &SuiteRowContext,
    n_gen: usize,
    runs: usize,
    no_warmup: bool,
    seed: u64,
) -> Result<BenchRow> {
    if n_gen == 0 {
        return Err(anyhow!("--tg values must be >= 1"));
    }
    let arch = loaded.arch();
    let mf = loaded.forward();
    let cap = n_gen + 16;
    let vocab = arch.vocab_size.max(1);
    let trace_counts = env_flag_enabled("QWEN_DECODE_TRACE_COUNTS");
    let mut rng_state = if seed == 0 { 1u64 } else { seed };
    let mut next_rand_tok = || -> i32 {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        (rng_state % vocab as u64) as i32
    };
    let run_once = |first_tok: i32,
                    ranges: &mut dyn FnMut() -> i32|
     -> Result<(f64, f64, Option<KernelTraceCounters>, f64)> {
        let session_t0 = Instant::now();
        let mut seq = loaded
            .create_sequence(SequenceConfig::new(cap))
            .context("suite tg session")?;
        let session_alloc_ms = session_t0.elapsed().as_secs_f64() * 1e3;
        seq.ensure_can_append(n_gen)
            .context("suite tg sequence capacity")?;
        let _kernel_trace_guard = trace_counts.then(kernel_trace_begin);
        let t0 = Instant::now();
        let mut tok = first_tok;
        let mut gpu_ms_acc = 0.0;
        for pos in 0..n_gen {
            let (_argmax, prof) =
                mf.single_token_argmax_profiled(tok, pos as u32, seq.metal_session_mut())?;
            gpu_ms_acc += prof.gpu_kernel_ms;
            tok = ranges();
        }
        let wall_ms = t0.elapsed().as_secs_f64() * 1e3;
        seq.advance_by(n_gen).context("suite tg advance sequence")?;
        let counts = trace_counts.then(kernel_trace_snapshot);
        Ok((wall_ms, gpu_ms_acc, counts, session_alloc_ms))
    };

    if !no_warmup {
        let first = next_rand_tok();
        let _ = run_once(first, &mut next_rand_tok).context("suite tg warmup")?;
    }

    let mut wall_samples = Vec::with_capacity(runs);
    let mut gpu_samples = Vec::with_capacity(runs);
    let mut ts_samples = Vec::with_capacity(runs);
    let mut session_alloc_samples = Vec::with_capacity(runs);
    let mut trace_samples = Vec::with_capacity(runs);
    for _ in 0..runs {
        let first = next_rand_tok();
        let (wall_ms, gpu_ms, trace, session_alloc_ms) =
            run_once(first, &mut next_rand_tok).context("suite tg run")?;
        wall_samples.push(wall_ms);
        gpu_samples.push(gpu_ms);
        ts_samples.push(n_gen as f64 * 1000.0 / wall_ms);
        session_alloc_samples.push(session_alloc_ms);
        if let Some(trace) = trace {
            trace_samples.push(trace);
        }
    }

    let wall_mean = sample_mean(&wall_samples);
    let gpu_mean = sample_mean(&gpu_samples);
    let session_alloc_mean = sample_mean(&session_alloc_samples);
    let trace_per_token = if trace_counts && !trace_samples.is_empty() {
        let denom = (trace_samples.len() * n_gen) as f64;
        let encoders: u64 = trace_samples.iter().map(|t| t.encoders).sum();
        let concurrent_encoders: u64 = trace_samples.iter().map(|t| t.concurrent_encoders).sum();
        let dispatches: u64 = trace_samples.iter().map(|t| t.dispatches).sum();
        Some((
            1.0,
            encoders as f64 / denom,
            concurrent_encoders as f64 / denom,
            dispatches as f64 / denom,
        ))
    } else {
        None
    };
    let decode_gb_per_s = if matches!(arch.kind, qwen_llm::model::ArchKind::Dense)
        && row_ctx.model_size > 0
        && wall_mean > 0.0
    {
        Some((row_ctx.model_size as f64 / 1e9) / (wall_mean / 1000.0 / n_gen as f64))
    } else {
        None
    };

    Ok(BenchRow {
        schema_version: BENCH_SCHEMA_VERSION,
        engine: "qwen-llm",
        build_commit: row_ctx.build_commit,
        build_dirty: row_ctx.build_dirty,
        build_identity: recorded_build_identity(),
        test_time: utc_iso8601_now(),
        model_filename: row_ctx.model_filename.clone(),
        model_size: row_ctx.model_size,
        model_n_params: row_ctx.model_n_params,
        arch_kind: row_ctx.arch_kind,
        test: format!("tg{n_gen}"),
        n_tokens: n_gen,
        n_repetitions: runs,
        avg_ts: sample_mean(&ts_samples),
        stddev_ts: sample_stdev(&ts_samples),
        samples_ts: ts_samples,
        samples_ns: wall_samples.iter().map(|w| (*w * 1e6) as u64).collect(),
        avg_ns: (wall_mean * 1e6) as u64,
        avg_compute_ns: Some((wall_mean * 1e6) as u64),
        avg_session_alloc_ns: Some((session_alloc_mean * 1e6) as u64),
        avg_scratch_alloc_ns: None,
        avg_gpu_ns: Some((gpu_mean * 1e6) as u64),
        kernel_trace_command_buffers_per_token: trace_per_token.map(|t| t.0),
        kernel_trace_encoders_per_token: trace_per_token.map(|t| t.1),
        kernel_trace_concurrent_encoders_per_token: trace_per_token.map(|t| t.2),
        kernel_trace_dispatches_per_token: trace_per_token.map(|t| t.3),
        decode_gb_per_s,
        prefill_chunk: None,
        decode_mode: Some("apples-lcpp"),
        prefill_mode: None,
        power: row_ctx.power.clone(),
        qwen_env: row_ctx.qwen_env.clone(),
    })
}

fn run_suite(args: SuiteArgs) -> Result<()> {
    let SuiteArgs {
        model,
        pp,
        tg,
        runs,
        no_warmup,
        prefill_chunk,
        seed,
        output,
    } = args;
    if runs == 0 {
        return Err(anyhow!("--runs must be >= 1"));
    }
    if pp.is_empty() && tg.is_empty() {
        return Err(anyhow!("provide at least one --pp or --tg shape"));
    }
    let json_mode = matches!(output, OutputFormat::Json);
    macro_rules! text_log { ($($t:tt)*) => { if !json_mode { eprintln!($($t)*); } } }

    let runtime = Runtime::metal().context("init Runtime")?;
    text_log!("[suite] device: {}", runtime.describe());
    let power = capture_power_snapshot();
    text_log!("[suite] power: {}", power_snapshot_summary(power.as_ref()));

    let loaded = runtime
        .load_model(&model)
        .with_context(|| format!("load {}", model.display()))?;
    let g = loaded.gguf();
    let (commit, dirty) = qwen_build_identity();
    let row_ctx = SuiteRowContext {
        build_commit: commit,
        build_dirty: dirty,
        model_filename: model.display().to_string(),
        model_size: model_weight_bytes(g),
        model_n_params: g
            .get_u64("general.parameter_count")
            .unwrap_or_else(|| g.tensors.iter().map(|t| t.n_elements()).sum()),
        arch_kind: arch_kind_label(loaded.arch().kind),
        power,
        qwen_env: capture_qwen_env(),
    };

    text_log!(
        "[suite] model={} pp={:?} tg={:?} runs={} warmup={} seed={}",
        model.display(),
        pp,
        tg,
        runs,
        if no_warmup { "skip" } else { "on" },
        seed,
    );

    let mut rows = Vec::with_capacity(pp.len() + tg.len());
    for &n_prompt in &pp {
        let row = run_suite_pp_row(
            &loaded,
            &row_ctx,
            n_prompt,
            runs,
            no_warmup,
            prefill_chunk,
            seed,
        )?;
        text_log!(
            "[suite] {:>8}: {:>8.2} t/s  {:>8.2} ms/token",
            row.test,
            row.avg_ts,
            1000.0 / row.avg_ts.max(1e-9),
        );
        rows.push(row);
    }
    for &n_gen in &tg {
        let row = run_suite_tg_row(&loaded, &row_ctx, n_gen, runs, no_warmup, seed)?;
        text_log!(
            "[suite] {:>8}: {:>8.2} t/s  {:>8.2} ms/token",
            row.test,
            row.avg_ts,
            1000.0 / row.avg_ts.max(1e-9),
        );
        rows.push(row);
    }

    if json_mode {
        println!(
            "{}",
            serde_json::to_string(&rows).context("serialize suite rows")?
        );
    }
    Ok(())
}

fn run_pp_wait(args: PpWaitArgs) -> Result<()> {
    let PpWaitArgs {
        model,
        n_prompt,
        prefill_chunk,
        with_tail,
        seed,
        no_warmup,
        ready_file,
        go_file,
        output,
    } = args;
    if n_prompt == 0 {
        return Err(anyhow!("--n-prompt must be >= 1"));
    }
    let json_mode = matches!(output, OutputFormat::Json);
    macro_rules! text_log { ($($t:tt)*) => { if !json_mode { eprintln!($($t)*); } } }

    let ctx = MetalContext::new().context("init MetalContext")?;
    text_log!("[pp-wait] device: {}", ctx.describe());
    let power = capture_power_snapshot();
    text_log!(
        "[pp-wait] power: {}",
        power_snapshot_summary(power.as_ref())
    );

    let g = GgufFile::open(&model).with_context(|| format!("open {}", model.display()))?;
    let m = Model::from_gguf(&g).context("parse model arch from gguf")?;
    let mm = MetalModel::load(&ctx, &g, &m).context("metal-load model weights")?;
    let ids = synthetic_prompt_ids(n_prompt, m.arch.vocab_size, seed);
    let prefill_chunk =
        prefill_chunk.unwrap_or_else(|| default_prefill_chunk(m.arch.kind, ids.len()));
    if prefill_chunk == 0 {
        return Err(anyhow!("--prefill-chunk must be >= 1"));
    }

    let mf = MetalForward::new(&ctx, &mm);
    let cap = ids.len() + 16;
    text_log!(
        "[pp-wait] model={} n_prompt={} chunk={} tail={} pid={}",
        model.display(),
        ids.len(),
        prefill_chunk,
        if with_tail { "final-logits" } else { "skip" },
        std::process::id(),
    );
    if !json_mode {
        print_prefill_lowering_summary(&mm);
    }

    if !no_warmup {
        let mut s = MetalSession::fresh(&ctx, &mm, cap).context("session warmup")?;
        let mut scratch = fresh_prefill_scratch_for_prompt(&ctx, &mm, prefill_chunk, ids.len())
            .context("warmup prefill scratch")?;
        if with_tail {
            let _ = prefill_tokens_with_multi_hidden(&mf, &ids, 0, &mut s, &mut scratch, &[], None)
                .context("warmup prefill with tail")?;
        } else {
            let _ = prefill_tokens_prompt_only_profiled(&mf, &ids, 0, &mut s, &mut scratch)
                .context("warmup prompt-only prefill")?;
        }
    }

    if let Some(parent) = ready_file.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if let Some(parent) = go_file.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if go_file.exists() {
        std::fs::remove_file(&go_file)?;
    }
    std::fs::write(
        &ready_file,
        format!(
            "ready pid={} model={} n_prompt={} chunk={} tail={}\n",
            std::process::id(),
            model.display(),
            ids.len(),
            prefill_chunk,
            if with_tail { "final-logits" } else { "skip" }
        ),
    )?;
    text_log!("[pp-wait] ready; waiting for {:?}", go_file);
    while !go_file.exists() {
        std::thread::sleep(Duration::from_millis(25));
    }
    text_log!("[pp-wait] go signal received; running timed prefill");

    let mut s = MetalSession::fresh(&ctx, &mm, cap).context("session run")?;
    let mut scratch = fresh_prefill_scratch_for_prompt(&ctx, &mm, prefill_chunk, ids.len())
        .context("timed prefill scratch")?;
    let t0 = Instant::now();
    let gpu_ms = if with_tail {
        let (_, gpu_ms) = prefill_tokens_with_multi_hidden_profiled(
            &mf,
            &ids,
            0,
            &mut s,
            &mut scratch,
            &[],
            None,
        )
        .context("timed prefill with tail")?;
        gpu_ms
    } else {
        prefill_tokens_prompt_only_profiled(&mf, &ids, 0, &mut s, &mut scratch)
            .context("timed prompt-only prefill")?
    };
    let wall_ms = t0.elapsed().as_secs_f64() * 1e3;
    let ts = ids.len() as f64 * 1000.0 / wall_ms;

    if json_mode {
        let (commit, dirty) = qwen_build_identity();
        let row = BenchRow {
            schema_version: BENCH_SCHEMA_VERSION,
            engine: "qwen-llm",
            build_commit: commit,
            build_dirty: dirty,
            build_identity: recorded_build_identity(),
            test_time: utc_iso8601_now(),
            model_filename: model.display().to_string(),
            model_size: model_weight_bytes(&g),
            model_n_params: g
                .get_u64("general.parameter_count")
                .unwrap_or_else(|| g.tensors.iter().map(|t| t.n_elements()).sum()),
            arch_kind: match m.arch.kind {
                qwen_llm::model::ArchKind::Dense => "dense",
                qwen_llm::model::ArchKind::Moe => "moe",
            },
            test: format!("pp{}", ids.len()),
            n_tokens: ids.len(),
            n_repetitions: 1,
            avg_ts: ts,
            stddev_ts: 0.0,
            samples_ts: vec![ts],
            samples_ns: vec![(wall_ms * 1e6) as u64],
            avg_ns: (wall_ms * 1e6) as u64,
            avg_compute_ns: Some((wall_ms * 1e6) as u64),
            avg_session_alloc_ns: None,
            avg_scratch_alloc_ns: None,
            avg_gpu_ns: Some((gpu_ms * 1e6) as u64),
            kernel_trace_command_buffers_per_token: None,
            kernel_trace_encoders_per_token: None,
            kernel_trace_concurrent_encoders_per_token: None,
            kernel_trace_dispatches_per_token: None,
            decode_gb_per_s: None,
            prefill_chunk: Some(prefill_chunk),
            decode_mode: None,
            prefill_mode: Some("packed"),
            power,
            qwen_env: capture_qwen_env(),
        };
        println!(
            "{}",
            serde_json::to_string(&vec![row]).context("serialize pp-wait row")?
        );
    } else {
        eprintln!(
            "[pp-wait] run: wall {:>8.1} ms  gpu {:>8.1} ms  {:>7.2} t/s",
            wall_ms, gpu_ms, ts
        );
    }
    Ok(())
}

fn run_decode(args: DecodeArgs) -> Result<()> {
    let DecodeArgs {
        model,
        prompt,
        tokens,
        oracle,
        oracle_phase,
        no_warmup,
        sequential_prefill,
        prefill_chunk,
        kv_capacity,
        full_logits_decode,
        runs,
        output,
    } = args;
    if runs == 0 {
        return Err(anyhow!("--runs must be >= 1"));
    }
    let prompt =
        prompt.unwrap_or_else(|| "The quick brown fox jumps over the lazy dog".to_string());
    let json_mode = matches!(output, OutputFormat::Json);
    macro_rules! text_log { ($($t:tt)*) => { if !json_mode { eprintln!($($t)*); } } }

    let ctx = MetalContext::new().context("init MetalContext")?;
    text_log!("[bench] device: {}", ctx.describe());
    let power = capture_power_snapshot();
    text_log!("[bench] power: {}", power_snapshot_summary(power.as_ref()));

    let g = GgufFile::open(&model).with_context(|| format!("open {}", model.display()))?;
    let m = Model::from_gguf(&g).context("parse model arch from gguf")?;
    let mm = MetalModel::load(&ctx, &g, &m).context("metal-load model weights")?;
    let tok = Tokenizer::from_gguf(&g).context("open tokenizer")?;

    let ids = tok.encode(&prompt, false).context("tokenize prompt")?;
    if ids.is_empty() {
        return Err(anyhow!("prompt tokenized to an empty sequence"));
    }
    let prefill_chunk =
        prefill_chunk.unwrap_or_else(|| default_prefill_chunk(m.arch.kind, ids.len()));
    if prefill_chunk == 0 {
        return Err(anyhow!("--prefill-chunk must be >= 1"));
    }
    if tokens == 0 && oracle.is_some() && oracle_phase == OraclePhase::Final {
        return Err(anyhow!(
            "--oracle-phase final requires at least one decode token; use --oracle-phase prefill for prompt-only validation"
        ));
    }
    let mf = MetalForward::new(&ctx, &mm);
    let min_cap = ids.len() + tokens + 16;
    let cap = kv_capacity.unwrap_or(min_cap);
    if cap < min_cap {
        return Err(anyhow!(
            "--kv-capacity {cap} is too small; need at least prompt_tokens + tokens + 16 = {min_cap}"
        ));
    }
    eprintln!(
        "[bench] model={} prompt={:?} ({} tokens), gen={} tokens, kv_capacity={}",
        model.display(),
        prompt,
        ids.len(),
        tokens,
        cap
    );
    let use_packed_prefill = !sequential_prefill;
    let use_gpu_argmax_decode = !full_logits_decode;

    if !no_warmup {
        // One warmup pass to compile pipeline state objects + warm caches.
        let mut s = MetalSession::fresh(&ctx, &mm, cap).context("session warmup")?;
        let warmup_last_logits = if use_packed_prefill {
            let mut scratch = fresh_prefill_scratch_for_prompt(&ctx, &mm, prefill_chunk, ids.len())
                .context("packed prefill warmup scratch")?;
            prefill_tokens_with_multi_hidden(&mf, &ids, 0, &mut s, &mut scratch, &[], None)
                .context("packed prefill warmup")?
        } else {
            let mut logits = Vec::new();
            for (i, &tid) in ids.iter().enumerate() {
                logits = mf.single_token(tid, i as u32, &mut s)?;
            }
            logits
        };

        if tokens > 0 {
            let warmup_next = argmax_i32(&warmup_last_logits);
            let warmup_pos = ids.len() as u32;
            if use_gpu_argmax_decode {
                if m.arch.kind == qwen_llm::model::ArchKind::Moe {
                    let _ = mf.single_token_argmax_profiled(warmup_next, warmup_pos, &mut s)?;
                } else {
                    let _ = mf.single_token_argmax(warmup_next, warmup_pos, &mut s)?;
                }
            } else if m.arch.kind == qwen_llm::model::ArchKind::Moe {
                let _ = mf.single_token_profiled(warmup_next, warmup_pos, &mut s)?;
            } else {
                let _ = mf.single_token(warmup_next, warmup_pos, &mut s)?;
            }
        }
    }

    // Per-rep accumulators. The artifacts the oracle / text-output / MoE
    // profile blocks below need (last_logits, gen_ids, per_token_prof,
    // prefill_logits_for_oracle, s) all reflect the LAST timed rep — the
    // single-shot path is `--runs 1`, which preserves the old behavior.
    let want_prefill_oracle = oracle.is_some() && oracle_phase == OraclePhase::Prefill;
    let need_final_logits = oracle.is_some() && oracle_phase == OraclePhase::Final && tokens > 0;
    let mut prefill_walls: Vec<f64> = Vec::with_capacity(runs);
    let mut prefill_gpus: Vec<f64> = Vec::with_capacity(runs);
    let mut decode_walls: Vec<f64> = Vec::with_capacity(runs);
    let mut decode_steady_walls: Vec<f64> = Vec::with_capacity(runs);
    // last-rep artifacts; set inside the loop and consumed below.
    let mut s: MetalSession;
    let mut prefill_token_ms: Vec<f64> = Vec::with_capacity(ids.len());
    let mut decode_token_ms: Vec<f64> = Vec::with_capacity(tokens);
    let mut per_token_prof: Vec<qwen_llm::metal_forward::TokenProfile> =
        Vec::with_capacity(ids.len() + tokens);
    let mut last_logits: Vec<f32> = Vec::new();
    let mut prefill_logits_for_oracle: Option<Vec<f32>> = None;
    let mut gen_ids: Vec<i32> = Vec::with_capacity(tokens);

    for rep in 0..runs {
        // Fresh session per rep so we measure a steady-state cold-cache
        // prefill+decode pair, not the cumulative state of the previous rep.
        s = MetalSession::fresh(&ctx, &mm, cap).context("session run")?;
        prefill_token_ms.clear();
        decode_token_ms.clear();
        per_token_prof.clear();
        gen_ids.clear();
        last_logits.clear();
        let mut prefill_gpu_total_ms: Option<f64> = None;

        let t0 = Instant::now();
        if use_packed_prefill {
            let mut scratch = fresh_prefill_scratch_for_prompt(&ctx, &mm, prefill_chunk, ids.len())
                .context("packed prefill scratch")?;
            let (logits, gpu_total_ms) = prefill_tokens_with_multi_hidden_profiled(
                &mf,
                &ids,
                0,
                &mut s,
                &mut scratch,
                &[],
                None,
            )
            .context("packed prefill")?;
            last_logits = logits;
            prefill_gpu_total_ms = Some(gpu_total_ms);
            if want_prefill_oracle {
                prefill_logits_for_oracle = Some(last_logits.clone());
            }
        } else {
            for (i, &tid) in ids.iter().enumerate() {
                let tt = Instant::now();
                if m.arch.kind == qwen_llm::model::ArchKind::Moe {
                    let (logits, prof) = mf.single_token_profiled(tid, i as u32, &mut s)?;
                    last_logits = logits;
                    per_token_prof.push(prof);
                } else {
                    last_logits = mf.single_token(tid, i as u32, &mut s)?;
                }
                prefill_token_ms.push(tt.elapsed().as_secs_f64() * 1e3);
            }
            if want_prefill_oracle {
                prefill_logits_for_oracle = Some(last_logits.clone());
            }
        }
        let prefill_wall = t0.elapsed().as_secs_f64() * 1e3;
        prefill_walls.push(prefill_wall);
        if let Some(g_ms) = prefill_gpu_total_ms {
            prefill_gpus.push(g_ms);
        }

        // Decode loop: default greedy path uses GPU argmax so we don't read
        // back a full vocab row on every generated token. `--full-logits-decode`
        // forces the legacy path for A/B and debugging.
        let t1 = Instant::now();
        let mut next_tok = argmax_i32(&last_logits);
        for k in 0..tokens {
            let pos = ids.len() + k;
            let input_tok = next_tok;
            gen_ids.push(input_tok);
            let tt = Instant::now();
            let need_logits_this_step =
                !use_gpu_argmax_decode || (need_final_logits && k + 1 == tokens);
            if need_logits_this_step && m.arch.kind == qwen_llm::model::ArchKind::Moe {
                let (logits, prof) = mf.single_token_profiled(input_tok, pos as u32, &mut s)?;
                next_tok = argmax_i32(&logits);
                last_logits = logits;
                per_token_prof.push(prof);
            } else if need_logits_this_step {
                last_logits = mf.single_token(input_tok, pos as u32, &mut s)?;
                next_tok = argmax_i32(&last_logits);
            } else if m.arch.kind == qwen_llm::model::ArchKind::Moe {
                let (argmax, prof) =
                    mf.single_token_argmax_profiled(input_tok, pos as u32, &mut s)?;
                next_tok = argmax;
                per_token_prof.push(prof);
            } else {
                next_tok = mf.single_token_argmax(input_tok, pos as u32, &mut s)?;
            }
            let step_ms = tt.elapsed().as_secs_f64() * 1e3;
            decode_token_ms.push(step_ms);
        }
        let decode_wall = t1.elapsed().as_secs_f64() * 1e3;
        decode_walls.push(decode_wall);
        // Decode-only steady-state: skip the very first decode (cache-cold
        // for some downstream PSO + heavily warm-up sensitive).
        let steady_ms = if decode_token_ms.len() > 1 {
            decode_token_ms[1..].iter().sum::<f64>() / (decode_token_ms.len() - 1) as f64
        } else if decode_token_ms.len() == 1 {
            decode_wall
        } else {
            0.0
        };
        if !decode_token_ms.is_empty() {
            decode_steady_walls.push(steady_ms);
        }
        text_log!(
            "[bench] rep {:>2}: prefill {:>8.1} ms ({:.1} t/s)  decode {:>8.1} ms ({:.1} t/s)",
            rep + 1,
            prefill_wall,
            ids.len() as f64 * 1000.0 / prefill_wall,
            decode_wall,
            if tokens > 0 {
                tokens as f64 * 1000.0 / decode_wall
            } else {
                0.0
            }
        );
    }

    let prefill_wall = sample_mean(&prefill_walls);
    let prefill_avg = prefill_wall / ids.len() as f64;
    let prefill_gpu_total_ms = if prefill_gpus.is_empty() {
        None
    } else {
        Some(sample_mean(&prefill_gpus))
    };
    let decode_wall = if decode_walls.is_empty() {
        0.0
    } else {
        sample_mean(&decode_walls)
    };
    let total_wall = prefill_wall + decode_wall;
    let decode_avg_ms = if tokens > 0 {
        Some(decode_wall / tokens as f64)
    } else {
        None
    };
    let decode_steady_ms = if !decode_steady_walls.is_empty() {
        Some(sample_mean(&decode_steady_walls))
    } else {
        None
    };
    let prefill_ts_samples: Vec<f64> = prefill_walls
        .iter()
        .map(|w| ids.len() as f64 * 1000.0 / w)
        .collect();
    let decode_steady_ts_samples: Vec<f64> =
        decode_steady_walls.iter().map(|ms| 1000.0 / ms).collect();

    let decode_mode_label = if use_gpu_argmax_decode && need_final_logits {
        "gpu-argmax + final-logits-oracle"
    } else if use_gpu_argmax_decode {
        "gpu-argmax"
    } else {
        "full-logits"
    };

    if json_mode {
        let (commit, dirty) = qwen_build_identity();
        let model_size = model_weight_bytes(&g);
        let model_n_params = g
            .get_u64("general.parameter_count")
            .unwrap_or_else(|| g.tensors.iter().map(|t| t.n_elements()).sum());
        let arch_kind_str: &'static str = match m.arch.kind {
            qwen_llm::model::ArchKind::Dense => "dense",
            qwen_llm::model::ArchKind::Moe => "moe",
        };
        let qwen_env = capture_qwen_env();
        let mut rows: Vec<BenchRow> = Vec::new();
        // pp seed row: mean across reps.
        rows.push(BenchRow {
            schema_version: BENCH_SCHEMA_VERSION,
            engine: "qwen-llm",
            build_commit: commit,
            build_dirty: dirty,
            build_identity: recorded_build_identity(),
            test_time: utc_iso8601_now(),
            model_filename: model.display().to_string(),
            model_size,
            model_n_params,
            arch_kind: arch_kind_str,
            test: format!("pp{}", ids.len()),
            n_tokens: ids.len(),
            n_repetitions: runs,
            avg_ts: sample_mean(&prefill_ts_samples),
            stddev_ts: sample_stdev(&prefill_ts_samples),
            samples_ts: prefill_ts_samples.clone(),
            samples_ns: prefill_walls.iter().map(|w| (*w * 1e6) as u64).collect(),
            avg_ns: (prefill_wall * 1e6) as u64,
            avg_compute_ns: Some((prefill_wall * 1e6) as u64),
            avg_session_alloc_ns: None,
            avg_scratch_alloc_ns: None,
            avg_gpu_ns: prefill_gpu_total_ms.map(|g| (g * 1e6) as u64),
            kernel_trace_command_buffers_per_token: None,
            kernel_trace_encoders_per_token: None,
            kernel_trace_concurrent_encoders_per_token: None,
            kernel_trace_dispatches_per_token: None,
            decode_gb_per_s: None,
            prefill_chunk: if use_packed_prefill {
                Some(prefill_chunk)
            } else {
                None
            },
            decode_mode: None,
            prefill_mode: Some(if use_packed_prefill {
                "packed"
            } else {
                "sequential"
            }),
            power: power.clone(),
            qwen_env: qwen_env.clone(),
        });
        // tg row: post-first-token steady-state mean per rep, averaged
        // across reps. Matches lcpp's `tg<N>` printer (its `avg_ts` is the
        // mean of per-rep tokens/sec).
        if !decode_steady_ts_samples.is_empty() {
            let steady_mean_ms = decode_steady_ms.unwrap_or(0.0);
            let gb_per_s = if matches!(m.arch.kind, qwen_llm::model::ArchKind::Dense)
                && model_size > 0
                && steady_mean_ms > 0.0
            {
                Some((model_size as f64 / 1e9) / (steady_mean_ms / 1000.0))
            } else {
                None
            };
            rows.push(BenchRow {
                schema_version: BENCH_SCHEMA_VERSION,
                engine: "qwen-llm",
                build_commit: commit,
                build_dirty: dirty,
                build_identity: recorded_build_identity(),
                test_time: utc_iso8601_now(),
                model_filename: model.display().to_string(),
                model_size,
                model_n_params,
                arch_kind: arch_kind_str,
                test: format!("tg{}", tokens),
                n_tokens: tokens,
                n_repetitions: decode_steady_ts_samples.len(),
                avg_ts: sample_mean(&decode_steady_ts_samples),
                stddev_ts: sample_stdev(&decode_steady_ts_samples),
                samples_ts: decode_steady_ts_samples.clone(),
                samples_ns: decode_steady_walls
                    .iter()
                    .map(|m| (*m * 1e6) as u64)
                    .collect(),
                avg_ns: (steady_mean_ms * 1e6) as u64,
                avg_compute_ns: Some((steady_mean_ms * 1e6) as u64),
                avg_session_alloc_ns: None,
                avg_scratch_alloc_ns: None,
                avg_gpu_ns: None,
                kernel_trace_command_buffers_per_token: None,
                kernel_trace_encoders_per_token: None,
                kernel_trace_concurrent_encoders_per_token: None,
                kernel_trace_dispatches_per_token: None,
                decode_gb_per_s: gb_per_s,
                prefill_chunk: if use_packed_prefill {
                    Some(prefill_chunk)
                } else {
                    None
                },
                decode_mode: Some(if use_gpu_argmax_decode {
                    "gpu-argmax"
                } else {
                    "full-logits"
                }),
                prefill_mode: None,
                power,
                qwen_env,
            });
        }
        let json = serde_json::to_string_pretty(&rows).context("serialize decode bench rows")?;
        println!("{json}");
        // Still run the oracle check + generation print below in non-json
        // mode; in json mode we skip both since they're text-only.
        return Ok(());
    }

    eprintln!();
    eprintln!(
        "[bench] === results ({runs} run{}) ===",
        if runs == 1 { "" } else { "s" }
    );
    eprintln!("[bench] decode mode: {}", decode_mode_label);
    eprintln!(
        "[bench] prefill mode: {}",
        if use_packed_prefill {
            "packed layer-major"
        } else if m.arch.kind == qwen_llm::model::ArchKind::Moe {
            "sequential MoE"
        } else {
            "sequential dense"
        }
    );
    if use_packed_prefill {
        eprintln!("[bench] prefill chunk: {prefill_chunk}");
    }
    let prefill_ts_mean = sample_mean(&prefill_ts_samples);
    let prefill_ts_sd = sample_stdev(&prefill_ts_samples);
    eprintln!(
        "[bench] prefill: {} tokens in {prefill_wall:.1} ms avg = {prefill_avg:.2} ms/token = {prefill_ts_mean:.1} +/- {prefill_ts_sd:.1} t/s",
        ids.len()
    );
    if let Some(prefill_gpu_total_ms) = prefill_gpu_total_ms {
        eprintln!(
            "[bench] prefill gpu: {prefill_gpu_total_ms:.1} ms avg = {:.2} ms/token = {:.1}% of prefill wall",
            prefill_gpu_total_ms / ids.len() as f64,
            100.0 * prefill_gpu_total_ms / prefill_wall.max(1e-9)
        );
    }
    if let Some(avg_ms) = decode_avg_ms {
        eprintln!(
            "[bench] decode:  {tokens} tokens in {decode_wall:.1} ms avg = {avg_ms:.2} ms/token (avg) = {:.1} t/s",
            1000.0 * tokens as f64 / decode_wall
        );
    } else {
        eprintln!("[bench] decode:  0 tokens requested (no decode loop)");
    }
    if let Some(steady_ms) = decode_steady_ms {
        let steady_ts_mean = sample_mean(&decode_steady_ts_samples);
        let steady_ts_sd = sample_stdev(&decode_steady_ts_samples);
        eprintln!(
            "[bench] steady:  {steady_ms:.2} ms/token (excl. first decode) = {steady_ts_mean:.2} +/- {steady_ts_sd:.2} t/s",
        );
    } else {
        eprintln!("[bench] steady:  N/A (no decode tokens)");
    }
    eprintln!("[bench] total:   {total_wall:.1} ms wall (mean)");

    if use_packed_prefill {
        eprintln!("[bench] prefill per-token: packed mode (no sequential replay series)");
    } else if !prefill_token_ms.is_empty() {
        let n_show = 5usize.min(prefill_token_ms.len());
        eprintln!(
            "[bench] prefill per-token (first {n_show}): {:?}",
            &prefill_token_ms[..n_show]
        );
        if prefill_token_ms.len() > 2 * n_show {
            let n = prefill_token_ms.len();
            eprintln!(
                "[bench] prefill per-token (last  {n_show}): {:?}",
                &prefill_token_ms[n - n_show..]
            );
        }
    }
    if !decode_token_ms.is_empty() {
        let n_show = 5usize.min(decode_token_ms.len());
        eprintln!(
            "[bench] decode per-token (first {n_show}): {:?}",
            &decode_token_ms[..n_show]
        );
        if decode_token_ms.len() > 2 * n_show {
            let n = decode_token_ms.len();
            eprintln!(
                "[bench] decode per-token (last  {n_show}): {:?}",
                &decode_token_ms[n - n_show..]
            );
        }
    }

    if m.arch.kind == qwen_llm::model::ArchKind::Moe && !per_token_prof.is_empty() {
        let avg = |f: fn(&qwen_llm::metal_forward::TokenProfile) -> f64| {
            per_token_prof.iter().map(f).sum::<f64>() / per_token_prof.len() as f64
        };
        let avg_total = avg(|p| p.total_ms);
        let avg_enc = avg(|p| p.cpu_encode_ms);
        let avg_gpu = avg(|p| p.gpu_kernel_ms);
        let avg_wait = avg(|p| p.cpu_to_gpu_complete_ms);
        let avg_route = avg(|p| p.moe_cpu_route_ms);
        let avg_cmds = avg(|p| p.moe_cmd_count as f64);
        eprintln!();
        eprintln!("[bench] === moe profile ===");
        eprintln!(
            "[bench] avg/token: total {avg_total:.2} ms | cpu_encode {avg_enc:.2} ms | gpu_kernel {avg_gpu:.2} ms | commit+wait {avg_wait:.2} ms | cpu_route {avg_route:.2} ms | cmd_bufs {avg_cmds:.1}"
        );
        eprintln!(
            "[bench] sync overhead/token: {:.2} ms (= commit+wait - gpu_kernel)",
            avg_wait - avg_gpu
        );
    }

    if let Some(oracle_path) = oracle {
        let bytes = std::fs::read(&oracle_path)
            .with_context(|| format!("read oracle {}", oracle_path.display()))?;
        let oracle_logits: &[f32] = match oracle_phase {
            OraclePhase::Prefill => prefill_logits_for_oracle.as_deref().unwrap_or(&last_logits),
            OraclePhase::Final => &last_logits,
        };
        if bytes.len() % 4 != 0 || bytes.len() / 4 != oracle_logits.len() {
            return Err(anyhow!(
                "oracle size {} bytes ({} f32) != logits len {}",
                bytes.len(),
                bytes.len() / 4,
                oracle_logits.len()
            ));
        }
        let oracle: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let (cos, max_abs, argmax_ours, argmax_oracle) = compare_logits(oracle_logits, &oracle);
        eprintln!(
            "[bench] oracle ({:?}):  cos={cos:.6}  max|Δ|={max_abs:.4}  argmax: ours={argmax_ours} oracle={argmax_oracle} {}",
            oracle_phase,
            if argmax_ours == argmax_oracle {
                "✓"
            } else {
                "✗ MISMATCH"
            }
        );
    }

    if !gen_ids.is_empty() {
        let text = tok.try_decode(&gen_ids)?;
        eprintln!("[bench] generated: {:?}", text);
    }

    Ok(())
}

/// Built-in audit corpus, classified by category. Designed to stress
/// codex's predicted failure modes for vocab pruning: multilingual,
/// CJK, code, math, names, URLs, JSON.
const BUILTIN_AUDIT_CORPUS: &[(&str, &str)] = &[
    // English chat (codex prediction: <1-2% miss at K=32K)
    ("en-chat", "Hello! How are you doing today?"),
    ("en-chat", "Can you explain photosynthesis in simple terms?"),
    ("en-chat", "What's the difference between a cat and a dog?"),
    (
        "en-chat",
        "Tell me a short bedtime story about a brave squirrel.",
    ),
    (
        "en-chat",
        "I'm planning a trip to Japan next spring. Any tips?",
    ),
    (
        "en-chat",
        "Why is the sky blue and what makes it sometimes orange?",
    ),
    ("en-chat", "Recommend a good book about Roman history."),
    (
        "en-chat",
        "What's the most efficient way to learn a new language?",
    ),
    // Code generation (codex prediction: <1-2% miss at K=32K)
    (
        "code",
        "Write a Python function to compute the Fibonacci sequence iteratively.",
    ),
    (
        "code",
        "Implement a quicksort algorithm in Rust with detailed comments.",
    ),
    (
        "code",
        "Create a TypeScript interface for a RESTful user API.",
    ),
    (
        "code",
        "How do I parse JSON in Go using the standard library?",
    ),
    (
        "code",
        "Show me a SQL query to find duplicate rows in a table.",
    ),
    ("code", "Write a regex that matches IPv4 addresses."),
    // Math / scientific notation (codex prediction: could be >5% miss)
    (
        "math",
        "Solve the integral of x^2 * sin(x) dx using integration by parts.",
    ),
    (
        "math",
        "What is the eigenvalue decomposition of [[4, 1], [2, 3]]?",
    ),
    (
        "math",
        "Derive the formula for the area of a circle from first principles.",
    ),
    (
        "math",
        "Explain Bayes' theorem with a worked example using P(A) and P(B|A).",
    ),
    // Multilingual (codex prediction: well above 5% miss)
    (
        "multi-fr",
        "Bonjour, comment ça va aujourd'hui? Pouvez-vous me parler de la cuisine française?",
    ),
    (
        "multi-de",
        "Können Sie mir die Geschichte des Bauhaus-Stils erklären?",
    ),
    (
        "multi-es",
        "¿Cuál es la mejor manera de aprender programación desde cero?",
    ),
    (
        "multi-it",
        "Qual è la differenza tra il Rinascimento italiano e il Barocco?",
    ),
    // CJK (codex prediction: well above 5% miss)
    ("cjk-zh", "请用简体中文解释一下相对论的基本概念。"),
    ("cjk-zh", "中国传统建筑中,斗拱结构有什么作用?"),
    ("cjk-ja", "日本の茶道について簡単に説明してください。"),
    (
        "cjk-ja",
        "プログラミングを始めるにはどの言語がおすすめですか?",
    ),
    (
        "cjk-ko",
        "한국의 전통 음식 중 비빔밥의 유래를 설명해 주세요.",
    ),
    // Names / URLs / proper nouns (codex prediction: high miss)
    (
        "names",
        "Tell me about the careers of Mahalia Jackson, Sviatoslav Richter, and Hayao Miyazaki.",
    ),
    (
        "names",
        "Compare the philosophies of Friedrich Nietzsche and Søren Kierkegaard.",
    ),
    (
        "urls",
        "Visit https://www.example.com/path/to/resource?query=foo&other=bar for more info.",
    ),
    // JSON / structured (codex's constraint-driven idea applies here)
    (
        "json",
        "Output a JSON object with fields name, age, and email for a fictional person.",
    ),
    (
        "json",
        "Generate a JSON Schema for a blog post with title, body, and tags.",
    ),
];

/// Audit semantics for vocab pruning: would the greedy argmax change
/// if we pruned the vocab to the first `K` token ids?
///
/// Returns `(true, _)` if the argmax over `&logits[..K]` matches the
/// argmax over `&logits[..]` (i.e., pruning would be lossless at this K).
/// Returns `(false, full_argmax)` if pruning would change the result.
///
/// Two-pass O(N): one over full, one over the K-prefix.
fn would_prune_to_k_match(logits: &[f32], k: usize) -> (bool, usize) {
    let mut full_best = (0usize, f32::NEG_INFINITY);
    for (i, &v) in logits.iter().enumerate() {
        if v > full_best.1 {
            full_best = (i, v);
        }
    }
    let kk = k.min(logits.len());
    let mut pruned_best = (0usize, f32::NEG_INFINITY);
    for (i, &v) in logits[..kk].iter().enumerate() {
        if v > pruned_best.1 {
            pruned_best = (i, v);
        }
    }
    (full_best.0 == pruned_best.0, full_best.0)
}

fn run_vocab_audit(args: VocabAuditArgs) -> Result<()> {
    let VocabAuditArgs {
        model,
        prompts,
        tokens,
        ks,
        show_examples,
    } = args;

    let ctx = MetalContext::new()?;
    eprintln!("[vocab-audit] device: {}", ctx.describe());
    let g = GgufFile::open(&model)?;
    let m = Model::from_gguf(&g)?;
    let vocab_size = m.arch.vocab_size as usize;
    let mm = MetalModel::load(&ctx, &g, &m)?;
    let tok = Tokenizer::from_gguf(&g)?;
    let mf = MetalForward::new(&ctx, &mm);

    let mut ks = ks;
    ks.sort_unstable();
    eprintln!(
        "[vocab-audit] vocab={vocab_size}, tokens/prompt={tokens}, Ks={:?}",
        ks
    );

    // Load corpus.
    let corpus: Vec<(String, String)> = if let Some(p) = prompts {
        let bytes =
            std::fs::read_to_string(&p).with_context(|| format!("read prompts {}", p.display()))?;
        bytes
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(|l| ("user".to_string(), l.to_string()))
            .collect()
    } else {
        BUILTIN_AUDIT_CORPUS
            .iter()
            .map(|(c, p)| (c.to_string(), p.to_string()))
            .collect()
    };
    eprintln!("[vocab-audit] {} prompts in corpus", corpus.len());

    // Track step-level statistics globally and per-category.
    use std::collections::BTreeMap;
    #[derive(Default, Clone)]
    struct CatStats {
        n_steps: usize,
        // Per-K: count of steps where argmax was OUTSIDE the top-K
        misses: BTreeMap<usize, usize>,
        // Highest argmax rank seen (worst case)
        max_rank: usize,
        // Examples of out-of-K argmax tokens for inspection (per K)
        examples: BTreeMap<usize, Vec<(String, String, usize)>>, // (category, decoded_token, rank)
    }
    let mut by_cat: BTreeMap<String, CatStats> = BTreeMap::new();
    let mut global = CatStats::default();
    for &k in &ks {
        global.misses.insert(k, 0);
        global.examples.insert(k, vec![]);
    }

    let total_decode_steps = corpus.len() * tokens;
    eprintln!(
        "[vocab-audit] expected total decode steps: {total_decode_steps} ({} prompts × {tokens} tokens)",
        corpus.len()
    );

    // Warmup pass.
    {
        let mut s = MetalSession::fresh(&ctx, &mm, 64)?;
        let _ = mf.single_token(corpus[0].1.as_bytes()[0] as i32, 0, &mut s)?;
    }

    // v0.75.1: hoisted layer_scratch (reused across all prompts in
    // the corpus). Allocation is ~tens of MB; per-prompt re-alloc is
    // pure waste at corpus sizes ≥ 100.
    let mut eval_layer_scratch =
        MetalDFlashLayerMajorScratch::fresh_prefill(&ctx, &mm, 16).context("eval layer scratch")?;
    let t_total = Instant::now();
    for (i_prompt, (cat, prompt)) in corpus.iter().enumerate() {
        let cat_stats = by_cat.entry(cat.clone()).or_default();
        // Initialize miss counters / example bins for this category if new.
        for &k in &ks {
            cat_stats.misses.entry(k).or_insert(0);
            cat_stats.examples.entry(k).or_default();
        }

        let ids = tok.encode(prompt, false)?;
        let mut sess = MetalSession::fresh(&ctx, &mm, ids.len() + tokens + 16)?;

        // Prefill the prompt. v0.75.1: packed mat-mat prefill (no
        // hidden capture for this eval mode).
        let mut last_logits = prefill_tokens_with_multi_hidden(
            &mf,
            &ids,
            0,
            &mut sess,
            &mut eval_layer_scratch,
            &[],
            None,
        )?;

        // Decode `tokens` steps; for each, check if the greedy argmax
        // token id would still be selected under each K-prune.
        for step in 0..tokens {
            let pos = ids.len() + step;
            let argmax = argmax_i32(&last_logits) as usize;

            global.n_steps += 1;
            cat_stats.n_steps += 1;
            global.max_rank = global.max_rank.max(argmax);
            cat_stats.max_rank = cat_stats.max_rank.max(argmax);

            for &k in &ks {
                let (matches, _) = would_prune_to_k_match(&last_logits, k);
                if !matches {
                    *global.misses.get_mut(&k).unwrap() += 1;
                    *cat_stats.misses.get_mut(&k).unwrap() += 1;
                    if cat_stats.examples[&k].len() < show_examples {
                        let decoded = tok.try_decode_piece(argmax as i32)?;
                        cat_stats.examples.get_mut(&k).unwrap().push((
                            cat.clone(),
                            decoded.clone(),
                            argmax,
                        ));
                    }
                    if global.examples[&k].len() < show_examples * 2 {
                        let decoded = tok.try_decode_piece(argmax as i32)?;
                        global
                            .examples
                            .get_mut(&k)
                            .unwrap()
                            .push((cat.clone(), decoded, argmax));
                    }
                }
            }

            // Continue greedy decode.
            last_logits = mf.single_token(argmax as i32, pos as u32, &mut sess)?;
        }

        if i_prompt % 4 == 3 || i_prompt == corpus.len() - 1 {
            let elapsed = t_total.elapsed().as_secs_f64();
            let pct = (i_prompt + 1) as f64 / corpus.len() as f64 * 100.0;
            eprintln!(
                "[vocab-audit] progress: {}/{} prompts ({pct:.0}%) in {elapsed:.1}s",
                i_prompt + 1,
                corpus.len()
            );
        }
    }

    // ---- Report ----
    eprintln!();
    eprintln!(
        "[vocab-audit] === GLOBAL ({} steps, max rank seen={}) ===",
        global.n_steps, global.max_rank
    );
    eprintln!(
        "[vocab-audit] {:>6}   {:>10}  {:>10}  weight savings",
        "K", "miss%", "miss/total"
    );
    for &k in &ks {
        let n_miss = global.misses[&k];
        let pct = 100.0 * n_miss as f64 / global.n_steps as f64;
        let savings = 100.0 * (1.0 - k as f64 / vocab_size as f64);
        eprintln!(
            "[vocab-audit]   {k:>6}   {pct:>9.3}%   {n_miss:>4}/{:<4}  ({savings:.1}% lm_head smaller)",
            global.n_steps
        );
    }
    eprintln!();
    eprintln!("[vocab-audit] === PER CATEGORY ===");
    for (cat, stats) in by_cat.iter() {
        eprintln!(
            "[vocab-audit] {cat:<10} ({:>3} steps, max rank seen={})",
            stats.n_steps, stats.max_rank
        );
        for &k in &ks {
            let n_miss = stats.misses[&k];
            let pct = 100.0 * n_miss as f64 / stats.n_steps as f64;
            eprintln!(
                "[vocab-audit]   K={k:>6}: {pct:>6.2}% miss ({n_miss}/{} steps)",
                stats.n_steps
            );
        }
    }

    // Examples of out-of-K tokens for the smallest K (most demanding).
    let smallest_k = *ks.first().unwrap();
    eprintln!();
    eprintln!("[vocab-audit] === EXAMPLES of argmax > K={smallest_k} ===");
    for (cat, stats) in by_cat.iter() {
        let exs = &stats.examples[&smallest_k];
        if !exs.is_empty() {
            eprintln!("[vocab-audit] {cat}:");
            for (_, tok_str, rank) in exs.iter().take(show_examples) {
                eprintln!("[vocab-audit]   rank={rank:>6}: {:?}", tok_str);
            }
        }
    }

    // Codex's pass conditions.
    eprintln!();
    eprintln!("[vocab-audit] === codex's H3 falsification ===");
    let target_workloads: &[(&str, &[&str])] = &[
        ("en-chat-only", &["en-chat"]),
        ("en+code", &["en-chat", "code"]),
        ("everything", &[]),
    ];
    for (label, cats) in target_workloads {
        let (n, nm32, nm64) = if cats.is_empty() {
            (
                global.n_steps,
                global.misses.get(&32768).copied().unwrap_or(0),
                global.misses.get(&65536).copied().unwrap_or(0),
            )
        } else {
            let mut n = 0;
            let mut m32 = 0;
            let mut m64 = 0;
            for c in cats.iter() {
                if let Some(s) = by_cat.get(*c) {
                    n += s.n_steps;
                    m32 += s.misses.get(&32768).copied().unwrap_or(0);
                    m64 += s.misses.get(&65536).copied().unwrap_or(0);
                }
            }
            (n, m32, m64)
        };
        if n == 0 {
            continue;
        }
        let p32 = 100.0 * nm32 as f64 / n as f64;
        let p64 = 100.0 * nm64 as f64 / n as f64;
        let pass32 = if p32 < 1.0 { "PASS" } else { "FAIL" };
        let pass64 = if p64 < 1.0 { "PASS" } else { "FAIL" };
        eprintln!(
            "[vocab-audit] {label}: K=32768 miss={p32:.2}% [{pass32}], K=65536 miss={p64:.2}% [{pass64}]"
        );
    }

    Ok(())
}

fn run_ctx_sweep(args: CtxSweepArgs) -> Result<()> {
    let CtxSweepArgs {
        model,
        checkpoints,
        window,
        concurrent_gdn_proj,
        concurrent_attn_proj,
        fresh_per_checkpoint,
        prefill_warm,
    } = args;
    if prefill_warm && !fresh_per_checkpoint {
        return Err(anyhow!("--prefill-warm requires --fresh-per-checkpoint"));
    }
    let ctx = MetalContext::new()?;
    eprintln!("[bench] device: {}", ctx.describe());
    let g = GgufFile::open(&model)?;
    let m = Model::from_gguf(&g)?;
    let mm = MetalModel::load(&ctx, &g, &m)?;

    let max_n = *checkpoints
        .iter()
        .max()
        .ok_or_else(|| anyhow!("no checkpoints"))?;
    let mut s = MetalSession::fresh(&ctx, &mm, 32)?;
    let mf = MetalForward::new(&ctx, &mm);

    // Warmup pipeline state cache.
    for i in 0..3 {
        let _ = mf.single_token(0, i as u32, &mut s)?;
    }
    println!("[ctx-sweep] === per-token decode cost vs context ===");
    println!(
        "[ctx-sweep] allocation_mode={}",
        if fresh_per_checkpoint {
            "fresh-per-checkpoint"
        } else {
            "single-max-capacity"
        }
    );
    println!("[ctx-sweep] context  total_ms  gpu_ms  cpu_enc_ms  t/s");

    if fresh_per_checkpoint {
        for &target in &checkpoints {
            let mut s = MetalSession::fresh(&ctx, &mm, target + window + 16)?;
            if target > 0 && prefill_warm {
                let ids = vec![0i32; target];
                let chunk = default_prefill_chunk(mm.arch.kind, target);
                let mut scratch = fresh_prefill_scratch_for_prompt(&ctx, &mm, chunk, ids.len())
                    .context("prefill-warm scratch")?;
                let t0 = Instant::now();
                prefill_tokens_prompt_only_profiled(&mf, &ids, 0, &mut s, &mut scratch)
                    .context("prefill-warm")?;
                eprintln!(
                    "[ctx-sweep] prefill-warm to {target} in {:.1}s",
                    t0.elapsed().as_secs_f64()
                );
            } else if target > 0 {
                let _ = mf.single_token(0, 0, &mut s)?;
                for p in 1..(target as u32) {
                    let _ = mf.single_token(0, p, &mut s)?;
                }
            }

            let mut samples = Vec::with_capacity(window);
            for i in 0..window {
                let pos = target as u32 + i as u32;
                let (_, p) = if concurrent_gdn_proj && concurrent_attn_proj {
                    mf.single_token_profiled_concurrent_gdn_attn_dense(0, pos, &mut s)?
                } else if concurrent_gdn_proj {
                    mf.single_token_profiled_concurrent_gdn_dense(0, pos, &mut s)?
                } else if concurrent_attn_proj {
                    mf.single_token_profiled_concurrent_attn_dense(0, pos, &mut s)?
                } else {
                    mf.single_token_profiled(0, pos, &mut s)?
                };
                samples.push(p);
            }

            let avg_total = samples.iter().map(|p| p.total_ms).sum::<f64>() / window as f64;
            let avg_gpu = samples.iter().map(|p| p.gpu_kernel_ms).sum::<f64>() / window as f64;
            let avg_enc = samples.iter().map(|p| p.cpu_encode_ms).sum::<f64>() / window as f64;
            println!(
                "[ctx-sweep] {target:>7}  {avg_total:>8.2}  {avg_gpu:>6.2}  {avg_enc:>10.2}  {:>4.1}",
                1000.0 / avg_total
            );
        }
    } else {
        let mut s = MetalSession::fresh(&ctx, &mm, max_n + window + 16)?;
        // One pre-warmed token at position 0 to populate everything.
        let _ = mf.single_token(0, 0, &mut s)?;

        let mut prev_pos = 1u32;
        for &target in &checkpoints {
            for p in prev_pos..(target as u32) {
                let _ = mf.single_token(0, p, &mut s)?;
            }
            prev_pos = target as u32;

            let mut samples = Vec::with_capacity(window);
            for i in 0..window {
                let pos = prev_pos + i as u32;
                let (_, p) = if concurrent_gdn_proj && concurrent_attn_proj {
                    mf.single_token_profiled_concurrent_gdn_attn_dense(0, pos, &mut s)?
                } else if concurrent_gdn_proj {
                    mf.single_token_profiled_concurrent_gdn_dense(0, pos, &mut s)?
                } else if concurrent_attn_proj {
                    mf.single_token_profiled_concurrent_attn_dense(0, pos, &mut s)?
                } else {
                    mf.single_token_profiled(0, pos, &mut s)?
                };
                samples.push(p);
            }
            prev_pos += window as u32;

            let avg_total = samples.iter().map(|p| p.total_ms).sum::<f64>() / window as f64;
            let avg_gpu = samples.iter().map(|p| p.gpu_kernel_ms).sum::<f64>() / window as f64;
            let avg_enc = samples.iter().map(|p| p.cpu_encode_ms).sum::<f64>() / window as f64;
            println!(
                "[ctx-sweep] {target:>7}  {avg_total:>8.2}  {avg_gpu:>6.2}  {avg_enc:>10.2}  {:>4.1}",
                1000.0 / avg_total
            );
        }
    }

    Ok(())
}

fn run_phase(args: PhaseArgs) -> Result<()> {
    let PhaseArgs { model, ctx: target } = args;
    let mctx = MetalContext::new()?;
    let g = GgufFile::open(&model)?;
    let m = Model::from_gguf(&g)?;
    let mm = MetalModel::load(&mctx, &g, &m)?;

    let mf = MetalForward::new(&mctx, &mm);
    {
        let mut s = MetalSession::fresh(&mctx, &mm, 32)?;
        for i in 0..3 {
            let _ = mf.single_token(0, i as u32, &mut s)?;
        }
    }
    let mut s = MetalSession::fresh(&mctx, &mm, target + 16)?;
    for p in 0..(target as u32) {
        let _ = mf.single_token(0, p, &mut s)?;
    }
    let (_, wall_artifact, phases) = mf.single_token_phase_profiled(0, target as u32, &mut s)?;
    let phase_sum: f64 = phases.iter().map(|p| p.1).sum();
    let phase_mode = match std::env::var("QWEN_PHASE_MOE_FFN_SPLIT").as_deref() {
        Ok("2") | Ok("deep") | Ok("DEEP") => "deep serial FFN split diagnostic",
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES") => {
            "production-wave FFN split diagnostic"
        }
        _ => "production-realistic GPU",
    };
    println!(
        "[phase ctx={target}] phase_sum={phase_sum:.2} ms ({phase_mode})  \
         wall_artifact={wall_artifact:.2} ms (DO NOT use as prod ms/token)"
    );
    for (name, ms) in &phases {
        let pct = ms / phase_sum * 100.0;
        println!("[phase ctx={target}]   {name:25} {ms:7.2} ms  ({pct:5.1}%)");
    }
    Ok(())
}

fn run_attn_intra(args: AttnIntraArgs) -> Result<()> {
    let AttnIntraArgs {
        model,
        ctx: target,
        runs,
        block,
    } = args;
    if runs == 0 {
        return Err(anyhow!("--runs must be > 0"));
    }

    let mctx = MetalContext::new()?;
    let g = GgufFile::open(&model)?;
    let m = Model::from_gguf(&g)?;
    let mm = MetalModel::load(&mctx, &g, &m)?;
    let mf = MetalForward::new(&mctx, &mm);

    let total_attn = mm
        .blocks
        .iter()
        .filter(|b| matches!(b, MetalBlock::Attn(_)))
        .count();
    let mut attn_seen = 0usize;
    let mut selected: Option<(usize, usize)> = None;
    for (idx, b) in mm.blocks.iter().enumerate() {
        if matches!(b, MetalBlock::Attn(_)) {
            if block.map_or(true, |want| want == idx) {
                selected = Some((idx, attn_seen));
                break;
            }
            attn_seen += 1;
        }
    }
    let (block_idx, attn_idx_in_session) = selected.ok_or_else(|| match block {
        Some(idx) => anyhow!("block {idx} is not a full-attention block"),
        None => anyhow!("model has no full-attention blocks"),
    })?;
    let ab = match &mm.blocks[block_idx] {
        MetalBlock::Attn(a) => a,
        _ => unreachable!(),
    };

    let arch = &mm.arch;
    let h = arch.hidden_size as usize;
    let head_dim = arch.attn_head_dim as usize;
    let n_q = arch.n_q_heads as usize;
    let n_kv = arch.n_kv_heads as usize;
    let group = n_q / n_kv;
    let q_dim = n_q * head_dim;
    let kv_dim = n_kv * head_dim;
    let n_rot = (head_dim as f32 * arch.partial_rotary_factor) as usize;
    if head_dim != 256 || !matches!(group, 4 | 6 | 8 | 16) {
        return Err(anyhow!(
            "attn-intra only supports attn_v4 shapes; got head_dim={head_dim} group={group}"
        ));
    }

    {
        let mut s = MetalSession::fresh(&mctx, &mm, 32)?;
        for i in 0..3 {
            let _ = mf.single_token(0, i as u32, &mut s)?;
        }
    }
    let mut s = MetalSession::fresh(&mctx, &mm, target + runs + 16)?;
    if target > 1 {
        // prefill-warm (v0.494 pattern): production packed prefill instead of
        // the token-by-token decode ramp; validated gpu_ms parity at ctx16384.
        let ids = vec![0i32; target];
        let chunk = default_prefill_chunk(mm.arch.kind, target);
        let mut scratch = fresh_prefill_scratch_for_prompt(&mctx, &mm, chunk, ids.len())
            .context("attn-intra prefill scratch")?;
        let t0 = Instant::now();
        prefill_tokens_prompt_only_profiled(&mf, &ids, 0, &mut s, &mut scratch)
            .context("attn-intra prefill warm")?;
        eprintln!(
            "[attn-intra] prefill-warm to {target} in {:.1}s",
            t0.elapsed().as_secs_f64()
        );
    }

    let timed = |label: &str,
                 cb: &dyn Fn(&KernelEncoder) -> Result<()>,
                 phases: &mut Vec<(String, f64)>|
     -> Result<()> {
        let cmd = mctx.queue.commandBuffer().context("attn-intra cmd")?;
        let enc = KernelEncoder::begin(&cmd);
        cb(&enc)?;
        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();
        phases.push((
            label.to_string(),
            (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3,
        ));
        Ok(())
    };

    let mut agg: Vec<(String, f64)> = Vec::new();
    for run in 0..runs {
        let position = target as u32 + run as u32;
        let mut phases: Vec<(String, f64)> = Vec::new();
        timed(
            "pre_norm (rms_norm)",
            &|enc| {
                Ok(encode_rms_norm_mul_f32(
                    &mctx,
                    enc,
                    &s.x,
                    &ab.attn_norm,
                    &s.h,
                    RMS_EPS,
                )?)
            },
            &mut phases,
        )?;
        timed(
            "q_proj_2x (mat_vec)",
            &|enc| {
                Ok(encode_mat_vec_dispatch(
                    &mctx,
                    enc,
                    &ab.q,
                    &s.h,
                    &s.attn_q_full,
                    h,
                    2 * q_dim,
                )?)
            },
            &mut phases,
        )?;
        timed(
            "split_q_gate",
            &|enc| {
                Ok(encode_split_q_gate_f32(
                    &mctx,
                    enc,
                    &s.attn_q_full,
                    &s.attn_q,
                    &s.attn_gate,
                    n_q,
                    head_dim,
                )?)
            },
            &mut phases,
        )?;
        timed(
            "q_norm (batched rms)",
            &|enc| {
                Ok(encode_rms_norm_batched_f32(
                    &mctx,
                    enc,
                    &s.attn_q,
                    &ab.q_norm,
                    &s.attn_q_normed,
                    n_q,
                    head_dim,
                    RMS_EPS,
                )?)
            },
            &mut phases,
        )?;
        timed(
            "k_proj (mat_vec)",
            &|enc| {
                Ok(encode_mat_vec_dispatch(
                    &mctx,
                    enc,
                    &ab.k,
                    &s.h,
                    &s.attn_k_now,
                    h,
                    kv_dim,
                )?)
            },
            &mut phases,
        )?;
        timed(
            "v_proj (mat_vec)",
            &|enc| {
                Ok(encode_mat_vec_dispatch(
                    &mctx,
                    enc,
                    &ab.v,
                    &s.h,
                    &s.attn_v_now,
                    h,
                    kv_dim,
                )?)
            },
            &mut phases,
        )?;
        timed(
            "k_norm (batched rms)",
            &|enc| {
                Ok(encode_rms_norm_batched_f32(
                    &mctx,
                    enc,
                    &s.attn_k_now,
                    &ab.k_norm,
                    &s.attn_k_normed,
                    n_kv,
                    head_dim,
                    RMS_EPS,
                )?)
            },
            &mut phases,
        )?;
        timed(
            "rope Q",
            &|enc| {
                Ok(encode_rope_neox_f32(
                    &mctx,
                    enc,
                    &s.attn_q_normed,
                    n_q,
                    head_dim,
                    n_rot,
                    position,
                    arch.rope_theta,
                )?)
            },
            &mut phases,
        )?;
        timed(
            "rope K",
            &|enc| {
                Ok(encode_rope_neox_f32(
                    &mctx,
                    enc,
                    &s.attn_k_normed,
                    n_kv,
                    head_dim,
                    n_rot,
                    position,
                    arch.rope_theta,
                )?)
            },
            &mut phases,
        )?;
        timed(
            "kv scatter (fused)",
            &|enc| match s.kv_k[attn_idx_in_session].dtype {
                GgmlType::F16 => Ok(encode_scatter_offset_f32_to_f16_kv(
                    &mctx,
                    enc,
                    &s.attn_k_normed,
                    &s.attn_v_now,
                    &s.kv_k[attn_idx_in_session],
                    &s.kv_v[attn_idx_in_session],
                    (position as usize) * kv_dim,
                    kv_dim,
                )?),
                GgmlType::Q8_0 => Ok(encode_scatter_offset_f32_to_q8_0_kv(
                    &mctx,
                    enc,
                    &s.attn_k_normed,
                    &s.attn_v_now,
                    &s.kv_k[attn_idx_in_session],
                    &s.kv_v[attn_idx_in_session],
                    (position as usize) * kv_dim,
                    kv_dim,
                )?),
                other => Err(anyhow!("unsupported KV dtype for attn-intra: {other:?}")),
            },
            &mut phases,
        )?;
        s.kv_n_pos[attn_idx_in_session] = position as usize + 1;
        let n_pos = s.kv_n_pos[attn_idx_in_session];
        let nwg = attn_v4_choose_nwg(n_pos, group);
        let tile_c = attn_v4_choose_tile_c(n_pos, group);
        timed(
            "attn_decode_v4_main",
            &|enc| {
                Ok(encode_attn_decode_v4_main_only_f32(
                    &mctx,
                    enc,
                    &s.attn_q_normed,
                    &s.kv_k[attn_idx_in_session],
                    &s.kv_v[attn_idx_in_session],
                    &s.attn_v4_o_partial,
                    &s.attn_v4_ml_partial,
                    n_q,
                    n_kv,
                    head_dim,
                    n_pos,
                    nwg,
                    tile_c,
                )?)
            },
            &mut phases,
        )?;
        timed(
            "attn_decode_v4_reduce",
            &|enc| {
                Ok(encode_attn_decode_v4_reduce_only_f32(
                    &mctx,
                    enc,
                    &s.attn_v4_o_partial,
                    &s.attn_v4_ml_partial,
                    &s.attn_o,
                    n_q,
                    n_kv,
                    head_dim,
                    nwg,
                )?)
            },
            &mut phases,
        )?;
        timed(
            "gate sigmoid + mul",
            &|enc| {
                Ok(encode_sigmoid_mul_f32(
                    &mctx,
                    enc,
                    &s.attn_gate,
                    &s.attn_o,
                    &s.attn_o,
                )?)
            },
            &mut phases,
        )?;
        timed(
            "o_proj (mat_vec)",
            &|enc| {
                Ok(encode_mat_vec_dispatch(
                    &mctx,
                    enc,
                    &ab.o,
                    &s.attn_o,
                    &s.mixer_out,
                    q_dim,
                    h,
                )?)
            },
            &mut phases,
        )?;
        timed(
            "residual_add #1",
            &|enc| Ok(encode_add_inplace_f32(&mctx, enc, &s.x, &s.mixer_out)?),
            &mut phases,
        )?;
        timed(
            "post_norm (rms_norm)",
            &|enc| {
                Ok(encode_rms_norm_mul_f32(
                    &mctx,
                    enc,
                    &s.x,
                    &ab.post_attn_norm,
                    &s.h,
                    RMS_EPS,
                )?)
            },
            &mut phases,
        )?;

        if agg.is_empty() {
            agg = phases;
        } else {
            for ((_, total), (_, ms)) in agg.iter_mut().zip(phases) {
                *total += ms;
            }
        }
    }

    let n_pos_est = target + runs;
    let nwg = attn_v4_choose_nwg(n_pos_est, group);
    let tile_c = attn_v4_choose_tile_c(n_pos_est, group);
    let group_tile = attn_v4_choose_group_tile(n_pos_est, group);
    let subgroups = group / group_tile.max(1);
    let logical_kv_bytes = n_kv * n_pos_est * (head_dim + head_dim) * 2;
    let subgroup_kv_bytes = logical_kv_bytes * subgroups;
    let partial_bytes = n_kv * nwg * group * (head_dim * 4 + 2 * 4);
    let reduce_bytes = partial_bytes + n_q * head_dim * 4;

    let avgs: Vec<(String, f64)> = agg
        .into_iter()
        .map(|(name, ms)| (name, ms / runs as f64))
        .collect();
    let total: f64 = avgs.iter().map(|(_, ms)| *ms).sum();
    println!(
        "[attn-intra ctx={target}] block={block_idx} attn_idx={attn_idx_in_session} \
         attn_layers={total_attn} runs={runs} n_q={n_q} n_kv={n_kv} group={group} \
         group_tile={group_tile} nwg={nwg} tile_c={tile_c}"
    );
    println!(
        "[attn-intra ctx={target}] bytes_est main_gb={:.4} reduce_gb={:.4} \
         logical_kv_gb={:.4} subgroup_kv_gb={:.4}",
        (subgroup_kv_bytes + partial_bytes) as f64 / 1e9,
        reduce_bytes as f64 / 1e9,
        logical_kv_bytes as f64 / 1e9,
        subgroup_kv_bytes as f64 / 1e9,
    );
    println!(
        "[attn-intra ctx={target}] one_layer_avg={total:.4} ms extrapolated={:.4} ms",
        total * total_attn as f64
    );
    println!("phase\tavg_ms\tpct\textrapolated_ms\test_gb\test_gb_s");
    for (name, ms) in &avgs {
        let est_bytes = match name.as_str() {
            "attn_decode_v4_main" => subgroup_kv_bytes + partial_bytes,
            "attn_decode_v4_reduce" => reduce_bytes,
            _ => 0,
        };
        if est_bytes > 0 && *ms > 0.0 {
            let gb = est_bytes as f64 / 1e9;
            println!(
                "{name}\t{ms:.4}\t{:.2}\t{:.4}\t{gb:.4}\t{:.1}",
                ms / total * 100.0,
                ms * total_attn as f64,
                gb / (*ms / 1000.0)
            );
        } else {
            println!(
                "{name}\t{ms:.4}\t{:.2}\t{:.4}\t\t",
                ms / total * 100.0,
                ms * total_attn as f64
            );
        }
    }

    Ok(())
}

fn run_roofline(args: RooflineArgs) -> Result<()> {
    let RooflineArgs {
        stream_mib,
        fma_elements,
        fma_iters,
        mat_in,
        mat_out,
        mat_query,
        runs,
        output,
    } = args;
    if stream_mib == 0
        || fma_elements == 0
        || fma_iters == 0
        || mat_in == 0
        || mat_out == 0
        || mat_query == 0
        || runs == 0
    {
        return Err(anyhow!(
            "stream_mib, fma_elements, fma_iters, mat_in, mat_out, mat_query, and runs must all be > 0"
        ));
    }
    if mat_in % 256 != 0 || mat_out % 64 != 0 || mat_query % 64 != 0 {
        return Err(anyhow!(
            "Q4_K mat-mat calibration requires mat_in % 256 == 0, mat_out % 64 == 0, and mat_query % 64 == 0"
        ));
    }

    let ctx = MetalContext::new()?;
    let stream_bytes = stream_mib
        .checked_mul(1024 * 1024)
        .context("stream bytes overflow")?;
    let stream_elems = (stream_bytes / std::mem::size_of::<f32>()).max(1);
    let stream_x = MetalTensor::zeros_f32(&ctx, vec![stream_elems as u64])?;
    let stream_y = MetalTensor::zeros_f32(&ctx, vec![stream_elems as u64])?;
    let fma_x = MetalTensor::zeros_f32(&ctx, vec![fma_elements as u64])?;
    let fma_y = MetalTensor::zeros_f32(&ctx, vec![fma_elements as u64])?;
    let mat_x_elems = mat_query.checked_mul(mat_in).context("mat_x overflow")?;
    let mat_y_elems = mat_query.checked_mul(mat_out).context("mat_y overflow")?;
    let mat_w =
        MetalTensor::zeros_dtype(&ctx, vec![mat_in as u64, mat_out as u64], GgmlType::Q4_K)?;
    let mat_x = MetalTensor::zeros_f32(&ctx, vec![mat_x_elems as u64])?;
    let mat_y = MetalTensor::zeros_f32(&ctx, vec![mat_y_elems as u64])?;

    let init_cmd = ctx.queue.commandBuffer().context("roofline init cmd")?;
    let init_enc = KernelEncoder::begin(&init_cmd);
    encode_fill_f32(&ctx, &init_enc, &stream_x, 1.0)?;
    encode_fill_f32(&ctx, &init_enc, &stream_y, 2.0)?;
    encode_fill_f32(&ctx, &init_enc, &fma_x, 1.0)?;
    encode_fill_f32(&ctx, &init_enc, &fma_y, 0.0)?;
    encode_fill_f32(&ctx, &init_enc, &mat_x, 1.0)?;
    encode_fill_f32(&ctx, &init_enc, &mat_y, 0.0)?;
    init_enc.end();
    init_cmd.commit();
    init_cmd.waitUntilCompleted();

    fn timed_kernel(
        ctx: &MetalContext,
        runs: usize,
        mut encode: impl FnMut(&KernelEncoder) -> Result<()>,
    ) -> Result<Vec<f64>> {
        let mut samples = Vec::with_capacity(runs);
        for rep in 0..=runs {
            let cmd = ctx.queue.commandBuffer().context("roofline cmd")?;
            let enc = KernelEncoder::begin(&cmd);
            encode(&enc)?;
            enc.end();
            cmd.commit();
            cmd.waitUntilCompleted();
            if rep > 0 {
                samples.push((cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3);
            }
        }
        Ok(samples)
    }

    let stream_ms = timed_kernel(&ctx, runs, |enc| {
        encode_roofline_stream_f32(&ctx, enc, &stream_x, &stream_y, 1.0000001)?;
        Ok(())
    })?;
    let fma_ms = timed_kernel(&ctx, runs, |enc| {
        encode_roofline_fma_f32(&ctx, enc, &fma_x, &fma_y, fma_iters)?;
        Ok(())
    })?;
    let mat_ms = timed_kernel(&ctx, runs, |enc| {
        encode_mat_mat_dispatch(
            &ctx, enc, &mat_w, &mat_x, &mat_y, mat_in, mat_out, mat_query,
        )?;
        Ok(())
    })?;

    let stream_avg_ms = sample_mean(&stream_ms);
    let fma_avg_ms = sample_mean(&fma_ms);
    let mat_avg_ms = sample_mean(&mat_ms);
    let stream_nominal_bytes = (stream_elems as f64) * 3.0 * std::mem::size_of::<f32>() as f64;
    let fma_flops = (fma_elements as f64) * (fma_iters as f64) * 2.0;
    let mat_flops = (mat_in as f64) * (mat_out as f64) * (mat_query as f64) * 2.0;
    let stream_gb_s = stream_nominal_bytes / (stream_avg_ms * 1e-3) / 1e9;
    let fma_tflops = fma_flops / (fma_avg_ms * 1e-3) / 1e12;
    let mat_tflops = mat_flops / (mat_avg_ms * 1e-3) / 1e12;
    let power = capture_power_snapshot();

    if matches!(output, OutputFormat::Json) {
        let row = serde_json::json!({
            "schema_version": 1,
            "engine": "qwen-llm",
            "test": "roofline",
            "test_time": utc_iso8601_now(),
            "device": ctx.device.name().to_string(),
            "stream": {
                "elements": stream_elems,
                "nominal_bytes_per_rep": stream_nominal_bytes as u64,
                "avg_ms": stream_avg_ms,
                "stddev_ms": sample_stdev(&stream_ms),
                "samples_ms": stream_ms,
                "gb_s": stream_gb_s,
            },
            "fma": {
                "elements": fma_elements,
                "iters": fma_iters,
                "nominal_flops_per_rep": fma_flops as u64,
                "avg_ms": fma_avg_ms,
                "stddev_ms": sample_stdev(&fma_ms),
                "samples_ms": fma_ms,
                "tflops": fma_tflops,
            },
            "matmat_q4_k": {
                "n_in": mat_in,
                "n_out": mat_out,
                "n_query": mat_query,
                "dtype": "Q4_K",
                "nominal_flops_per_rep": mat_flops as u64,
                "avg_ms": mat_avg_ms,
                "stddev_ms": sample_stdev(&mat_ms),
                "samples_ms": mat_ms,
                "nominal_tflops": mat_tflops,
            },
            "power": power,
        });
        println!("{}", serde_json::to_string(&row)?);
    } else {
        println!("[roofline] device: {}", ctx.device.name());
        println!(
            "[roofline] power: {}",
            power_snapshot_summary(power.as_ref())
        );
        println!(
            "[roofline] stream elements={} nominal_bytes={} avg_ms={:.3} std_ms={:.3} GB/s={:.1}",
            stream_elems,
            stream_nominal_bytes as u64,
            stream_avg_ms,
            sample_stdev(&stream_ms),
            stream_gb_s
        );
        println!(
            "[roofline] fma elements={} iters={} nominal_flops={} avg_ms={:.3} std_ms={:.3} TFLOP/s={:.2}",
            fma_elements,
            fma_iters,
            fma_flops as u64,
            fma_avg_ms,
            sample_stdev(&fma_ms),
            fma_tflops
        );
        println!(
            "[roofline] matmat_q4_k n_in={} n_out={} n_query={} nominal_flops={} avg_ms={:.3} std_ms={:.3} nominal_TFLOP/s={:.2}",
            mat_in,
            mat_out,
            mat_query,
            mat_flops as u64,
            mat_avg_ms,
            sample_stdev(&mat_ms),
            mat_tflops
        );
    }
    Ok(())
}

fn run_decode_window(args: DecodeWindowArgs) -> Result<()> {
    let DecodeWindowArgs {
        model,
        target_ctx,
        prefill_warm,
        window,
        streams,
        stage_timestamps,
        stage_split_attn_route,
        stage_split_attn_detail,
        stage_split_gdn_after,
        ready_file,
        go_file,
        pipelined,
        concurrent_gdn_proj,
        concurrent_attn_proj,
    } = args;
    if streams == 0 {
        return Err(anyhow!("--streams must be >= 1"));
    }
    if streams > 1 && (pipelined || concurrent_gdn_proj || concurrent_attn_proj) {
        return Err(anyhow!(
            "--streams is a concurrency discriminator; combine it only with the default decode path"
        ));
    }
    if stage_timestamps && streams > 1 {
        return Err(anyhow!(
            "--stage-timestamps is a single-stream attribution probe; do not combine it with --streams"
        ));
    }
    if stage_timestamps && (pipelined || concurrent_gdn_proj || concurrent_attn_proj) {
        return Err(anyhow!(
            "--stage-timestamps profiles the default MoE decode shape; do not combine it with other decode-window experiments"
        ));
    }
    if stage_split_attn_route && !stage_timestamps {
        return Err(anyhow!(
            "--stage-split-attn-route requires --stage-timestamps"
        ));
    }
    if stage_split_attn_detail && !stage_timestamps {
        return Err(anyhow!(
            "--stage-split-attn-detail requires --stage-timestamps"
        ));
    }
    if stage_split_gdn_after && !stage_timestamps {
        return Err(anyhow!(
            "--stage-split-gdn-after requires --stage-timestamps"
        ));
    }
    let ctx = MetalContext::new()?;
    eprintln!("[decode-window] device: {}", ctx.describe());
    let g = GgufFile::open(&model)?;
    let m = Model::from_gguf(&g)?;
    let mm = MetalModel::load(&ctx, &g, &m)?;
    let mut s = MetalSession::fresh(&ctx, &mm, target_ctx + window + 16)?;
    let mf = MetalForward::new(&ctx, &mm);

    for i in 0..3 {
        let _ = mf.single_token(0, i as u32, &mut s)?;
    }
    let mut s = MetalSession::fresh(&ctx, &mm, target_ctx + window + 16)?;
    if prefill_warm && target_ctx > 1 {
        let ids = vec![0i32; target_ctx];
        let chunk = default_prefill_chunk(mm.arch.kind, target_ctx);
        let mut scratch = fresh_prefill_scratch_for_prompt(&ctx, &mm, chunk, ids.len())
            .context("prefill-warm scratch")?;
        let t0 = Instant::now();
        prefill_tokens_prompt_only_profiled(&mf, &ids, 0, &mut s, &mut scratch)
            .context("prefill-warm")?;
        eprintln!(
            "[decode-window] prefill-warm to {target_ctx} in {:.1}s",
            t0.elapsed().as_secs_f64()
        );
    } else {
        let _ = mf.single_token(0, 0, &mut s)?;
        for p in 1..(target_ctx as u32) {
            let _ = mf.single_token(0, p, &mut s)?;
        }
    }

    let mut s_opt = Some(s);
    let mut multi_stream_sessions = None;
    if streams > 1 {
        let mut sessions = Vec::with_capacity(streams);
        sessions.push(s_opt.take().expect("primary session present"));
        for stream_idx in 1..streams {
            eprintln!(
                "[decode-window] ramping stream {}/{} to ctx={}",
                stream_idx + 1,
                streams,
                target_ctx
            );
            let mut sx = MetalSession::fresh(&ctx, &mm, target_ctx + window + 16)?;
            let _ = mf.single_token(0, 0, &mut sx)?;
            for p in 1..(target_ctx as u32) {
                let _ = mf.single_token(0, p, &mut sx)?;
            }
            sessions.push(sx);
        }
        multi_stream_sessions = Some(sessions);
    }

    if let Some(parent) = ready_file.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if let Some(parent) = go_file.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if go_file.exists() {
        std::fs::remove_file(&go_file)?;
    }
    std::fs::write(
        &ready_file,
        format!("ready ctx={} window={}\n", target_ctx, window),
    )?;
    eprintln!(
        "[decode-window] ready at ctx={} waiting for {:?}",
        target_ctx, go_file
    );
    while !go_file.exists() {
        std::thread::sleep(Duration::from_millis(25));
    }
    eprintln!(
        "[decode-window] go signal received; running {} decode tokens",
        window
    );

    if pipelined && (concurrent_gdn_proj || concurrent_attn_proj) {
        return Err(anyhow!(
            "--pipelined is separate from the concurrent projection experiments; use one mode at a time"
        ));
    }

    if let Some(mut sessions) = multi_stream_sessions {
        return run_decode_window_multi_stream(&ctx, &mf, &mut sessions, target_ctx, window);
    }

    let mut s = s_opt.expect("single-stream session present");

    if stage_timestamps {
        if mm.arch.kind != qwen_llm::model::ArchKind::Moe {
            return Err(anyhow!(
                "--stage-timestamps currently supports MoE models only"
            ));
        }
        return run_decode_window_stage_timestamps(
            &mf,
            &mut s,
            target_ctx,
            window,
            stage_split_attn_route,
            stage_split_attn_detail,
            stage_split_gdn_after,
        );
    }

    fn median(values: &[f64]) -> f64 {
        let mut v = values.to_vec();
        v.sort_by(|a, b| a.total_cmp(b));
        let n = v.len();
        if n % 2 == 1 {
            v[n / 2]
        } else {
            (v[n / 2 - 1] + v[n / 2]) * 0.5
        }
    }

    fn p95(values: &[f64]) -> f64 {
        let mut v = values.to_vec();
        v.sort_by(|a, b| a.total_cmp(b));
        let idx = ((v.len() - 1) as f64 * 0.95).round() as usize;
        v[idx]
    }

    if pipelined {
        let ids_ping = [
            MetalTensor::zeros_f32(&ctx, vec![1])?,
            MetalTensor::zeros_f32(&ctx, vec![1])?,
        ];
        let argmax_ping = [
            MetalTensor::zeros_f32(&ctx, vec![1])?,
            MetalTensor::zeros_f32(&ctx, vec![1])?,
        ];

        let mut encode_ms = Vec::with_capacity(window);
        let mut wait_ms = Vec::with_capacity(window);
        let mut gpu_ms = Vec::with_capacity(window);
        let total_t = Instant::now();
        let mut next_tok = 0i32;
        let mut pos = target_ctx as u32;

        unsafe {
            let ptr = ids_ping[0].buffer.contents().as_ptr() as *mut i32;
            *ptr = next_tok;
        }
        let first_cmd = ctx.queue.commandBuffer().expect("command buffer");
        let first_encode_t = Instant::now();
        let first_enc = qwen_llm::metal::KernelEncoder::begin(&first_cmd);
        mf.encode_single_token_argmax(&first_enc, pos, &mut s, &ids_ping[0], &argmax_ping[0])?;
        first_enc.end();
        encode_ms.push(first_encode_t.elapsed().as_secs_f64() * 1e3);
        first_cmd.commit();
        let mut pending_cmd = first_cmd;
        let mut pending_slot = 0usize;
        pos += 1;

        for _step in 1..window {
            let next_slot = pending_slot ^ 1;
            let next_cmd = ctx.queue.commandBuffer().expect("command buffer");
            let next_encode_t = Instant::now();
            let next_enc = qwen_llm::metal::KernelEncoder::begin(&next_cmd);
            mf.encode_single_token_argmax(
                &next_enc,
                pos,
                &mut s,
                &ids_ping[next_slot],
                &argmax_ping[next_slot],
            )?;
            next_enc.end();
            encode_ms.push(next_encode_t.elapsed().as_secs_f64() * 1e3);

            let wait_t = Instant::now();
            pending_cmd.waitUntilCompleted();
            wait_ms.push(wait_t.elapsed().as_secs_f64() * 1e3);
            gpu_ms.push((pending_cmd.GPUEndTime() - pending_cmd.GPUStartTime()) * 1e3);
            next_tok = unsafe {
                let src = argmax_ping[pending_slot].buffer.contents().as_ptr() as *const i32;
                *src
            };
            unsafe {
                let ptr = ids_ping[next_slot].buffer.contents().as_ptr() as *mut i32;
                *ptr = next_tok;
            }
            next_cmd.commit();
            pending_cmd = next_cmd;
            pending_slot = next_slot;
            pos += 1;
        }

        let wait_t = Instant::now();
        pending_cmd.waitUntilCompleted();
        wait_ms.push(wait_t.elapsed().as_secs_f64() * 1e3);
        gpu_ms.push((pending_cmd.GPUEndTime() - pending_cmd.GPUStartTime()) * 1e3);

        let total_ms = total_t.elapsed().as_secs_f64() * 1e3;
        let avg_total = total_ms / window as f64;
        let avg_gpu = gpu_ms.iter().sum::<f64>() / window as f64;
        let avg_enc = encode_ms.iter().sum::<f64>() / window as f64;
        let _avg_wait = wait_ms.iter().sum::<f64>() / window as f64;
        let med_gpu = median(&gpu_ms);
        let med_enc = median(&encode_ms);
        let med_wait = median(&wait_ms);
        let p95_gpu = p95(&gpu_ms);
        let p95_enc = p95(&encode_ms);
        let p95_wait = p95(&wait_ms);
        eprintln!(
            "[decode-window] ctx={} window={} pipelined avg_total={:.2} ms avg_gpu={:.2} ms avg_cpu_enc={:.2} ms t/s={:.1}",
            target_ctx,
            window,
            avg_total,
            avg_gpu,
            avg_enc,
            1000.0 / avg_total
        );
        eprintln!(
            "[decode-window] pipelined med_gpu={:.2} ms med_cpu_enc={:.2} ms med_wait={:.2} ms gpu/total(avg)={:.1}%",
            med_gpu,
            med_enc,
            med_wait,
            100.0 * avg_gpu / avg_total
        );
        eprintln!(
            "[decode-window] pipelined p95_gpu={:.2} ms p95_cpu_enc={:.2} ms p95_wait={:.2} ms",
            p95_gpu, p95_enc, p95_wait,
        );
        return Ok(());
    }

    let mut samples = Vec::with_capacity(window);
    let mut prev_tok = 0i32;
    for i in 0..window {
        let pos = target_ctx as u32 + i as u32;
        let (logits, p) = match (mm.arch.kind, concurrent_gdn_proj, concurrent_attn_proj) {
            (qwen_llm::model::ArchKind::Dense, true, true) => {
                mf.single_token_profiled_concurrent_gdn_attn_dense(prev_tok, pos, &mut s)?
            }
            (qwen_llm::model::ArchKind::Dense, true, false) => {
                mf.single_token_profiled_concurrent_gdn_dense(prev_tok, pos, &mut s)?
            }
            (qwen_llm::model::ArchKind::Dense, false, true) => {
                mf.single_token_profiled_concurrent_attn_dense(prev_tok, pos, &mut s)?
            }
            (qwen_llm::model::ArchKind::Moe, true, false) => {
                mf.single_token_profiled_concurrent_gdn_moe(prev_tok, pos, &mut s)?
            }
            (qwen_llm::model::ArchKind::Moe, _, true) => {
                return Err(anyhow!(
                    "--concurrent-attn-proj decode-window is currently dense-only"
                ));
            }
            _ => mf.single_token_profiled(prev_tok, pos, &mut s)?,
        };
        samples.push(p);
        prev_tok = argmax_i32(&logits);
    }

    let avg_total = samples.iter().map(|p| p.total_ms).sum::<f64>() / window as f64;
    let avg_gpu = samples.iter().map(|p| p.gpu_kernel_ms).sum::<f64>() / window as f64;
    let avg_enc = samples.iter().map(|p| p.cpu_encode_ms).sum::<f64>() / window as f64;
    let avg_wait = samples
        .iter()
        .map(|p| p.cpu_to_gpu_complete_ms)
        .sum::<f64>()
        / window as f64;
    let totals: Vec<f64> = samples.iter().map(|p| p.total_ms).collect();
    let gpus: Vec<f64> = samples.iter().map(|p| p.gpu_kernel_ms).collect();
    let encs: Vec<f64> = samples.iter().map(|p| p.cpu_encode_ms).collect();
    let waits: Vec<f64> = samples.iter().map(|p| p.cpu_to_gpu_complete_ms).collect();
    let med_total = median(&totals);
    let med_gpu = median(&gpus);
    let med_enc = median(&encs);
    let med_wait = median(&waits);
    let p95_total = p95(&totals);
    let p95_gpu = p95(&gpus);
    let p95_enc = p95(&encs);
    let p95_wait = p95(&waits);
    eprintln!(
        "[decode-window] ctx={} window={}{}{}{} avg_total={:.2} ms avg_gpu={:.2} ms avg_cpu_enc={:.2} ms t/s={:.1}",
        target_ctx,
        window,
        if concurrent_gdn_proj {
            " concurrent_gdn"
        } else {
            ""
        },
        if concurrent_attn_proj {
            " concurrent_attn"
        } else {
            ""
        },
        if concurrent_gdn_proj && concurrent_attn_proj {
            "_both"
        } else {
            ""
        },
        avg_total,
        avg_gpu,
        avg_enc,
        1000.0 / avg_total
    );
    eprintln!(
        "[decode-window] med_total={:.2} ms med_gpu={:.2} ms med_cpu_enc={:.2} ms med_wait={:.2} ms gpu/total={:.1}%",
        med_total,
        med_gpu,
        med_enc,
        med_wait,
        100.0 * med_gpu / med_total
    );
    eprintln!(
        "[decode-window] p95_total={:.2} ms p95_gpu={:.2} ms p95_cpu_enc={:.2} ms p95_wait={:.2} ms",
        p95_total, p95_gpu, p95_enc, p95_wait,
    );
    eprintln!(
        "[decode-window] avg_wait={:.2} ms gpu/total(avg)={:.1}%",
        avg_wait,
        100.0 * avg_gpu / avg_total
    );
    Ok(())
}

#[derive(Default)]
struct StageAgg {
    count: usize,
    ms: f64,
}

fn run_decode_window_stage_timestamps(
    mf: &MetalForward,
    session: &mut MetalSession,
    target_ctx: usize,
    window: usize,
    split_attn_route: bool,
    split_attn_detail: bool,
    split_gdn_after: bool,
) -> Result<()> {
    let mut prev_tok = 0i32;
    let mut total_ms = 0.0f64;
    let mut gpu_ms = 0.0f64;
    let mut enc_ms = 0.0f64;
    let mut wait_ms = 0.0f64;
    let mut raw_cov = 0.0f64;
    let mut raw_span_ms = 0.0f64;
    let mut sampled_ticks = 0u64;
    let mut family = BTreeMap::<(String, String, bool), StageAgg>::new();
    let mut block =
        BTreeMap::<(String, String, Option<usize>, Option<usize>, bool), StageAgg>::new();

    for i in 0..window {
        let pos = target_ctx as u32 + i as u32;
        let (tok, profile) = mf.single_token_argmax_stage_profiled_concurrent_gdn_moe(
            prev_tok,
            pos,
            session,
            split_attn_route,
            split_attn_detail,
            split_gdn_after,
        )?;
        prev_tok = tok;
        total_ms += profile.token.total_ms;
        gpu_ms += profile.token.gpu_kernel_ms;
        enc_ms += profile.token.cpu_encode_ms;
        wait_ms += profile.token.cpu_to_gpu_complete_ms;
        raw_cov += profile.raw_coverage_assuming_ns;
        raw_span_ms += profile.raw_span_ms_assuming_ns;
        sampled_ticks = sampled_ticks.saturating_add(profile.sampled_span_ticks);

        for stage in profile.stages {
            let fam_key = (
                stage.block_kind.clone(),
                stage.family.clone(),
                stage.concurrent,
            );
            let fam = family.entry(fam_key).or_default();
            fam.count += 1;
            fam.ms += stage.duration_ms_scaled;

            let block_key = (
                stage.block_kind,
                stage.family,
                stage.block_index,
                stage.local_index,
                stage.concurrent,
            );
            let blk = block.entry(block_key).or_default();
            blk.count += 1;
            blk.ms += stage.duration_ms_scaled;
        }
    }

    let avg_total = total_ms / window as f64;
    let avg_gpu = gpu_ms / window as f64;
    let avg_enc = enc_ms / window as f64;
    let avg_wait = wait_ms / window as f64;
    eprintln!(
        "[decode-stage] ctx={} window={}{}{}{} avg_total={:.2} ms avg_gpu={:.2} ms avg_cpu_enc={:.2} ms t/s={:.1}",
        target_ctx,
        window,
        if split_attn_route {
            " split_attn_route"
        } else {
            ""
        },
        if split_attn_detail {
            " split_attn_detail"
        } else {
            ""
        },
        if split_gdn_after {
            " split_gdn_after"
        } else {
            ""
        },
        avg_total,
        avg_gpu,
        avg_enc,
        1000.0 / avg_total,
    );
    eprintln!(
        "[decode-stage] avg_wait={:.2} ms raw_coverage_assuming_ns={:.3} raw_span_ms={:.3} sampled_ticks={}",
        avg_wait,
        raw_cov / window as f64,
        raw_span_ms / window as f64,
        sampled_ticks,
    );

    println!(
        "row\tblock_kind\tfamily\tblock_index\tlocal_index\tconcurrent\tcount\tavg_ms\tpct_gpu"
    );
    for ((block_kind, family_name, concurrent), agg) in family {
        println!(
            "family\t{}\t{}\t\t\t{}\t{}\t{:.4}\t{:.2}",
            block_kind,
            family_name,
            concurrent,
            agg.count,
            agg.ms / window as f64,
            100.0 * agg.ms / gpu_ms,
        );
    }
    for ((block_kind, family_name, block_index, local_index, concurrent), agg) in block {
        let block_index = block_index.map(|v| v.to_string()).unwrap_or_default();
        let local_index = local_index.map(|v| v.to_string()).unwrap_or_default();
        println!(
            "block\t{}\t{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.2}",
            block_kind,
            family_name,
            block_index,
            local_index,
            concurrent,
            agg.count,
            agg.ms / window as f64,
            100.0 * agg.ms / gpu_ms,
        );
    }
    Ok(())
}

fn run_decode_window_multi_stream(
    ctx: &MetalContext,
    mf: &MetalForward,
    sessions: &mut [MetalSession],
    target_ctx: usize,
    window: usize,
) -> Result<()> {
    fn median(values: &[f64]) -> f64 {
        let mut v = values.to_vec();
        v.sort_by(|a, b| a.total_cmp(b));
        let n = v.len();
        if n % 2 == 1 {
            v[n / 2]
        } else {
            (v[n / 2 - 1] + v[n / 2]) * 0.5
        }
    }

    let streams = sessions.len();
    let mut queues = Vec::with_capacity(streams);
    let mut ids = Vec::with_capacity(streams);
    let mut argmax = Vec::with_capacity(streams);
    for _ in 0..streams {
        queues.push(
            ctx.device
                .newCommandQueue()
                .context("stream command queue")?,
        );
        ids.push(MetalTensor::zeros_f32(ctx, vec![1])?);
        argmax.push(MetalTensor::zeros_f32(ctx, vec![1])?);
    }

    let mut prev_tokens = vec![0i32; streams];
    let mut encode_ms = Vec::with_capacity(window);
    let mut wait_ms = Vec::with_capacity(window);
    let mut gpu_span_ms = Vec::with_capacity(window);
    let mut gpu_sum_ms = Vec::with_capacity(window);
    let total_t = Instant::now();

    for step in 0..window {
        let pos = target_ctx as u32 + step as u32;
        let encode_t = Instant::now();
        let mut cmds = Vec::with_capacity(streams);
        for stream_idx in 0..streams {
            unsafe {
                let ptr = ids[stream_idx].buffer.contents().as_ptr() as *mut i32;
                *ptr = prev_tokens[stream_idx];
            }
            let cmd = queues[stream_idx]
                .commandBuffer()
                .context("stream command buffer")?;
            let enc = KernelEncoder::begin(&cmd);
            mf.encode_single_token_argmax(
                &enc,
                pos,
                &mut sessions[stream_idx],
                &ids[stream_idx],
                &argmax[stream_idx],
            )?;
            enc.end();
            cmd.commit();
            cmds.push(cmd);
        }
        encode_ms.push(encode_t.elapsed().as_secs_f64() * 1e3);

        let wait_t = Instant::now();
        for cmd in &cmds {
            cmd.waitUntilCompleted();
        }
        wait_ms.push(wait_t.elapsed().as_secs_f64() * 1e3);

        let mut min_start = f64::INFINITY;
        let mut max_end = 0.0f64;
        let mut sum = 0.0f64;
        for cmd in &cmds {
            let start = cmd.GPUStartTime();
            let end = cmd.GPUEndTime();
            min_start = min_start.min(start);
            max_end = max_end.max(end);
            sum += (end - start) * 1e3;
        }
        gpu_span_ms.push((max_end - min_start) * 1e3);
        gpu_sum_ms.push(sum);

        for stream_idx in 0..streams {
            prev_tokens[stream_idx] = unsafe {
                let src = argmax[stream_idx].buffer.contents().as_ptr() as *const i32;
                *src
            };
        }
    }

    let total_ms = total_t.elapsed().as_secs_f64() * 1e3;
    let aggregate_tokens = streams * window;
    let avg_total = total_ms / window as f64;
    let avg_span = gpu_span_ms.iter().sum::<f64>() / window as f64;
    let avg_sum = gpu_sum_ms.iter().sum::<f64>() / window as f64;
    let avg_enc = encode_ms.iter().sum::<f64>() / window as f64;
    let avg_wait = wait_ms.iter().sum::<f64>() / window as f64;
    eprintln!(
        "[decode-window] ctx={} window={} streams={} avg_step={:.2} ms agg_t/s={:.1} per_stream_t/s={:.1}",
        target_ctx,
        window,
        streams,
        avg_total,
        aggregate_tokens as f64 / (total_ms * 1e-3),
        window as f64 / (total_ms * 1e-3)
    );
    eprintln!(
        "[decode-window] streams med_gpu_span={:.2} ms med_gpu_sum={:.2} ms med_cpu_enc={:.2} ms med_wait={:.2} ms",
        median(&gpu_span_ms),
        median(&gpu_sum_ms),
        median(&encode_ms),
        median(&wait_ms),
    );
    eprintln!(
        "[decode-window] streams avg_gpu_span={:.2} ms avg_gpu_sum={:.2} ms avg_cpu_enc={:.2} ms avg_wait={:.2} ms overlap_eff={:.2}x",
        avg_span,
        avg_sum,
        avg_enc,
        avg_wait,
        if avg_span > 0.0 {
            avg_sum / avg_span
        } else {
            0.0
        }
    );
    Ok(())
}

fn argmax_i32(logits: &[f32]) -> i32 {
    let mut best = (0usize, f32::NEG_INFINITY);
    for (i, &v) in logits.iter().enumerate() {
        if v > best.1 {
            best = (i, v);
        }
    }
    best.0 as i32
}

/// **v0.76 adaptive-N back-off**: per-outer-step verify-chain mode.
///
/// `Spec { n_eff }` runs the existing drafter + packed_verify path with
/// `n_eff` ∈ {16, 8, 4} (truncating the N=16 drafter's output to the
/// first `n_eff` tokens via `n_eff_override`). `Off` skips drafter
/// and packed_verify entirely, running a single `single_token` no-spec
/// step. Once entered, `Off` is terminal for the remainder of the
/// generation (codex Q7 rationale: ctx is monotonic within a
/// generation, so a ctx that earns `Off` will never cool back to
/// favor `Spec`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VerifyMode {
    Spec { n_eff: usize },
    Off,
}

/// **v0.76**: verify-chain length policy as selected by `--n-policy`.
///
/// `Adaptive` is the default — ctx-keyed schedule with `Off`-terminal.
/// The static variants exist for calibration sweeps + manual overrides.
#[derive(Clone, Copy, Debug)]
enum NPolicy {
    Adaptive,
    Static16,
    Static8,
    Static4,
    OffOnly,
}

impl NPolicy {
    fn parse(s: &str) -> Result<Self> {
        match s {
            "adaptive" => Ok(Self::Adaptive),
            "static-16" => Ok(Self::Static16),
            "static-8" => Ok(Self::Static8),
            "static-4" => Ok(Self::Static4),
            "off" => Ok(Self::OffOnly),
            other => anyhow::bail!(
                "unknown n-policy {other:?}; expected adaptive, static-16, \
                 static-8, static-4, or off"
            ),
        }
    }

    /// Choose `VerifyMode` for an outer step at given `kv_n_pos` (the
    /// session's current KV position, i.e. the absolute token position
    /// of the carry token's predecessor). The schedule is calibrated
    /// against M4 Max + 27B Q4_K_M; re-run the calibration sweep if
    /// hardware/quant changes (see `qwen-bench dflash --n-policy
    /// static-{16,8,4,off} --prompt ...` for sweep harness).
    fn for_ctx(self, kv_n_pos: usize) -> VerifyMode {
        match self {
            Self::Static16 => VerifyMode::Spec { n_eff: 16 },
            Self::Static8 => VerifyMode::Spec { n_eff: 8 },
            Self::Static4 => VerifyMode::Spec { n_eff: 4 },
            Self::OffOnly => VerifyMode::Off,
            Self::Adaptive => {
                // Calibrated schedule from v0.76 sweep (M4 Max, 27B
                // Q4_K_M, code prompts, 32-token gen, 2026-05-07).
                //
                // Decode tokens/sec by (ctx, mode):
                //
                //   ctx    static16  static8  static4   off    best
                //   ---  --------- -------- -------- ------  ------
                //     9     20.52    16.42    12.33  25.14    off
                //   181     32.71    20.08    12.60  24.98  spec16
                //   363     34.67    21.09    12.88  24.87  spec16
                //   727     24.89    16.80    11.05  24.58  spec16(tie)
                //  2055     11.36     9.52     7.11  24.25    off
                //  8223      3.65     3.34     2.95  22.35    off
                //
                // KEY FINDINGS:
                //  * Spec8 and Spec4 are NEVER the best mode for any
                //    ctx in {9, 181, 363, 727, 2055, 8223}. The action
                //    space collapses to {Spec16, Off} — binary choice.
                //  * Default ctx (~9 tokens) is OFF-favored: drafter +
                //    verify overhead at tiny ctx exceeds the
                //    amortization win. Surprising; pre-v0.76 we
                //    assumed Spec=16 was always best at small ctx.
                //  * Spec16 wins ctx ∈ [~64, ~1000) by 30-40% over
                //    off. Long-ctx (>=2K) Off wins by 2-7x.
                //  * Crossover ctx where Spec16 = Off is around
                //    ~727; above that, off pulls away fast as KV
                //    bandwidth scales with ctx and amplifies under
                //    N=16 verify-pass KV reads.
                //
                // SCHEDULE:
                //   ctx <   768: Spec(16) (the sweet spot for speculative
                //                gain at meaningful prompt sizes).
                //   ctx >=  768: Off (long-ctx collapse begins; off
                //                never loses again as ctx grows).
                //
                // The 768 threshold was validated by an additional
                // post-sweep measurement at ctx=1118 and ctx=1509:
                //
                //   ctx   static16  off    winner
                //  ---  --------- ------  ------
                //   727    24.89  24.58  spec16 (margin 1.3%)
                //  1118    18.08  24.72  off (margin 37%)
                //  1509    15.13  24.18  off (margin 60%)
                //
                // Crossover is between 727 and 1118; 768 is a
                // conservative round-power-of-2 cutoff that still
                // captures the marginal Spec16 win at ctx=727 and
                // hands off to Off well before the 1118 cliff. The
                // initial 1024 guess from interpolating {727, 2055}
                // was wrong: the long-ctx collapse starts well below
                // 1024.
                //
                // The 9-token-prompt regime where Off marginally beats
                // Spec(16) (25.14 vs 20.52 t/s) is INTENTIONALLY left
                // on Spec(16): real-world prompts almost always have
                // ≥ 100 tokens (system prompt + user input), and
                // entering Off at small ctx would break the
                // terminal-Off invariant when ctx grows past the
                // first crossover. The 18% slowdown on synthetic
                // tiny prompts is the cost of monotonicity.
                //
                // Re-run the sweep when KV-Q lands (v0.78+) — KV-Q
                // shifts the long-ctx crossover to higher ctx, and
                // possibly raises Spec's effective amortization range.
                if kv_n_pos < 768 {
                    VerifyMode::Spec { n_eff: 16 }
                } else {
                    VerifyMode::Off
                }
            }
        }
    }
}

fn compare_logits(ours: &[f32], oracle: &[f32]) -> (f64, f32, usize, usize) {
    debug_assert_eq!(ours.len(), oracle.len());
    let mut max_abs = 0.0f32;
    let mut argmax_ours = 0usize;
    let mut argmax_oracle = 0usize;
    let mut max_ours = f32::NEG_INFINITY;
    let mut max_oracle = f32::NEG_INFINITY;
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..ours.len() {
        max_abs = max_abs.max((ours[i] - oracle[i]).abs());
        if ours[i] > max_ours {
            max_ours = ours[i];
            argmax_ours = i;
        }
        if oracle[i] > max_oracle {
            max_oracle = oracle[i];
            argmax_oracle = i;
        }
        dot += ours[i] as f64 * oracle[i] as f64;
        na += (ours[i] as f64).powi(2);
        nb += (oracle[i] as f64).powi(2);
    }
    let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
    (cos, max_abs, argmax_ours, argmax_oracle)
}

/// **H2 falsification mode.** Compare cold-prefill TTFT vs snapshot-restore
/// TTFT for two requests sharing a token prefix.
///
/// Codex's H2 kill criteria (any failure → kill the experiment):
///   * 2nd-request TTFT ≥ 2× faster at prefix=64
///   * 2nd-request TTFT ≥ 5× faster at prefix=1024
///   * Restore p95 < 25 ms at prefix=4096
///   * (We also assert: cold-decoded-token == warm-decoded-token,
///      since both should produce identical greedy output.)
fn prefix_cache_prefill_logits(
    mf: &MetalForward<'_>,
    token_ids: &[i32],
    start_position: u32,
    session: &mut MetalSession,
    mode: PrefixCachePrefillMode,
    scratch: Option<&mut MetalDFlashLayerMajorScratch>,
) -> Result<Vec<f32>> {
    if token_ids.is_empty() {
        return Err(anyhow!("prefix-cache prefill token slice is empty"));
    }
    match mode {
        PrefixCachePrefillMode::Single => {
            let mut last_logits = Vec::new();
            for (i, &tid) in token_ids.iter().enumerate() {
                last_logits = mf.single_token(tid, start_position + i as u32, session)?;
            }
            Ok(last_logits)
        }
        PrefixCachePrefillMode::Packed => {
            let scratch =
                scratch.ok_or_else(|| anyhow!("packed prefix-cache prefill needs scratch"))?;
            Ok(prefill_tokens_with_multi_hidden(
                mf,
                token_ids,
                start_position,
                session,
                scratch,
                &[],
                None,
            )?)
        }
    }
}

fn run_prefix_cache(args: PrefixCacheArgs) -> Result<()> {
    let PrefixCacheArgs {
        model,
        prefix,
        target_prefix_len,
        suffix,
        tokens,
        prefill_mode,
        suffix_prefill_mode,
    } = args;

    let runtime = Runtime::metal()?;
    eprintln!("[prefix-cache] device: {}", runtime.describe());
    let loaded = runtime.load_model(&model)?;
    let ctx = loaded.context();
    let mm = loaded.metal_model();
    let tok = loaded.tokenizer()?;

    let mut prefix_ids = tok.encode(&prefix, false)?;
    if let Some(target) = target_prefix_len {
        // Pad with filler tokens to reach the target length.
        // Use a deterministic, semantically inert filler.
        let filler = " lorem ipsum dolor sit amet consectetur adipiscing elit";
        let filler_ids = tok.encode(filler, false)?;
        while prefix_ids.len() < target {
            for &id in &filler_ids {
                if prefix_ids.len() >= target {
                    break;
                }
                prefix_ids.push(id);
            }
        }
        prefix_ids.truncate(target);
    }
    let suffix_ids = tok.encode(&suffix, false)?;
    let total_len = prefix_ids.len() + suffix_ids.len();
    let effective_suffix_mode =
        choose_prefix_cache_suffix_mode(suffix_prefill_mode, suffix_ids.len());
    eprintln!(
        "[prefix-cache] prefix={} tokens, suffix={} tokens, total={} tokens prefill_mode={:?} suffix_prefill_mode={:?}->{:?}",
        prefix_ids.len(),
        suffix_ids.len(),
        total_len,
        prefill_mode,
        suffix_prefill_mode,
        effective_suffix_mode
    );

    let mf = MetalForward::new(&ctx, &mm);
    let cap = total_len + tokens + 16;

    let prefill_chunk = default_prefill_chunk(mm.arch.kind, total_len);
    let mut cold_scratch = if prefill_mode == PrefixCachePrefillMode::Packed {
        Some(fresh_prefill_scratch_for_prompt(
            &ctx,
            &mm,
            total_len,
            prefill_chunk,
        )?)
    } else {
        None
    };
    let mut prefix_scratch = if prefill_mode == PrefixCachePrefillMode::Packed {
        Some(fresh_prefill_scratch_for_prompt(
            &ctx,
            &mm,
            total_len,
            prefill_chunk,
        )?)
    } else {
        None
    };
    let mut suffix_scratch = if effective_suffix_mode == PrefixCachePrefillMode::Packed {
        Some(fresh_prefill_scratch_for_prompt(
            &ctx,
            &mm,
            total_len,
            prefill_chunk,
        )?)
    } else {
        None
    };

    // Warmup pass to compile pipeline state objects.
    {
        let mut s = loaded.create_sequence(SequenceConfig::new(32))?;
        if prefill_mode == PrefixCachePrefillMode::Packed {
            let mut warm_scratch = fresh_prefill_scratch_for_prompt(&ctx, &mm, 1, 1)?;
            let _ = prefill_tokens_with_multi_hidden(
                &mf,
                &[prefix_ids[0]],
                0,
                s.metal_session_mut(),
                &mut warm_scratch,
                &[],
                None,
            )?;
        } else {
            let _ = mf.single_token(prefix_ids[0], 0, s.metal_session_mut())?;
        }
    }

    // ---- COLD path: prefill (prefix + suffix), decode N tokens ----
    let cold_t0 = Instant::now();
    let mut seq_cold = loaded.create_sequence(SequenceConfig::new(cap))?;
    let full_ids: Vec<i32> = prefix_ids
        .iter()
        .chain(suffix_ids.iter())
        .copied()
        .collect();
    let last_logits = prefix_cache_prefill_logits(
        &mf,
        &full_ids,
        0,
        seq_cold.metal_session_mut(),
        prefill_mode,
        cold_scratch.as_mut(),
    )?;
    seq_cold.advance_by(full_ids.len())?;
    let cold_prefill_ms = cold_t0.elapsed().as_secs_f64() * 1e3;

    // First decoded token = TTFT-equivalent measurement.
    let cold_first_decode_t = Instant::now();
    let cold_first_id = argmax_i32(&last_logits);
    let _ = mf.single_token(
        cold_first_id,
        total_len as u32,
        seq_cold.metal_session_mut(),
    )?;
    seq_cold.advance_by(1)?;
    let cold_first_decode_ms = cold_first_decode_t.elapsed().as_secs_f64() * 1e3;

    let cold_ttft_ms = cold_prefill_ms + cold_first_decode_ms;
    eprintln!(
        "[prefix-cache] COLD: prefill {} tokens in {cold_prefill_ms:.1} ms, first-decode {cold_first_decode_ms:.1} ms, TTFT {cold_ttft_ms:.1} ms",
        total_len
    );

    // ---- WARM path: prefill prefix, snapshot. Then fresh session, restore, ----
    // ---- prefill suffix, decode 1 token. Time the second-request portion. ----
    let mut seq_pre = loaded.create_sequence(SequenceConfig::new(cap))?;
    let last_pre_logits = prefix_cache_prefill_logits(
        &mf,
        &prefix_ids,
        0,
        seq_pre.metal_session_mut(),
        prefill_mode,
        prefix_scratch.as_mut(),
    )?;
    seq_pre.advance_by(prefix_ids.len())?;
    let snap_t = Instant::now();
    let inserted =
        loaded.cache_sequence_prefix(&seq_pre, prefix_ids.clone(), Some(last_pre_logits))?;
    let snap_create_ms = snap_t.elapsed().as_secs_f64() * 1e3;
    let full_request: Vec<i32> = prefix_ids
        .iter()
        .chain(suffix_ids.iter())
        .copied()
        .collect();
    eprintln!(
        "[prefix-cache] (snapshot built: {:.1} MB in {snap_create_ms:.1} ms)",
        inserted.snapshot_bytes as f64 / 1e6
    );
    eprintln!(
        "[prefix-cache] cache after insert: entries={} bytes={:.1}/{:.1} MB",
        inserted.stats.entries,
        inserted.stats.total_bytes as f64 / 1e6,
        inserted.stats.max_bytes as f64 / 1e6
    );

    // Now simulate request 2 starting fresh and finding the cached prefix.
    let warm_t0 = Instant::now();
    let mut seq_warm = loaded.create_sequence(SequenceConfig::new(cap))?;
    let restore_t = Instant::now();
    let hit = loaded
        .restore_cached_prefix(&mut seq_warm, &full_request)?
        .ok_or_else(|| anyhow!("prefix cache lookup missed a freshly inserted prefix"))?;
    let restore_ms = restore_t.elapsed().as_secs_f64() * 1e3;
    if hit.matched_prefix_len != prefix_ids.len() {
        return Err(anyhow!(
            "prefix cache restored {} tokens, expected {}",
            hit.matched_prefix_len,
            prefix_ids.len()
        ));
    }
    eprintln!(
        "[prefix-cache] hit: matched_prefix={} exact={} exact_logits={} entries={} bytes={:.1}/{:.1} MB",
        hit.matched_prefix_len,
        hit.exact,
        hit.exact_final_logits.is_some(),
        hit.stats.entries,
        hit.stats.total_bytes as f64 / 1e6,
        hit.stats.max_bytes as f64 / 1e6
    );

    let last_warm_logits = prefix_cache_prefill_logits(
        &mf,
        &suffix_ids,
        prefix_ids.len() as u32,
        seq_warm.metal_session_mut(),
        effective_suffix_mode,
        suffix_scratch.as_mut(),
    )?;
    seq_warm.advance_by(suffix_ids.len())?;
    let warm_prefill_ms = warm_t0.elapsed().as_secs_f64() * 1e3;
    let warm_suffix_ms = warm_prefill_ms - restore_ms;

    let warm_first_decode_t = Instant::now();
    let warm_first_id = argmax_i32(&last_warm_logits);
    let _ = mf.single_token(
        warm_first_id,
        total_len as u32,
        seq_warm.metal_session_mut(),
    )?;
    seq_warm.advance_by(1)?;
    let warm_first_decode_ms = warm_first_decode_t.elapsed().as_secs_f64() * 1e3;

    let warm_ttft_ms = warm_prefill_ms + warm_first_decode_ms;
    eprintln!(
        "[prefix-cache] WARM: restore {restore_ms:.1} ms + suffix-prefill {} tokens in {warm_suffix_ms:.1} ms + first-decode {warm_first_decode_ms:.1} ms = TTFT {warm_ttft_ms:.1} ms",
        suffix_ids.len()
    );

    let speedup = cold_ttft_ms / warm_ttft_ms;
    eprintln!();
    eprintln!("[prefix-cache] === H2 falsification ===");
    eprintln!(
        "[prefix-cache] cold TTFT: {cold_ttft_ms:.1} ms  | warm TTFT: {warm_ttft_ms:.1} ms  | speedup: {speedup:.2}x"
    );

    // Codex's kill criteria check
    let prefix_len = prefix_ids.len();
    let required_speedup = if prefix_len >= 1024 {
        5.0
    } else if prefix_len >= 64 {
        2.0
    } else {
        1.0 // tiny prefix; only assert > 1×
    };
    let restore_ok = restore_ms < 25.0;
    let speedup_ok = speedup >= required_speedup;
    let first_token_match = cold_first_id == warm_first_id;

    eprintln!(
        "[prefix-cache] required_speedup_at_prefix_{prefix_len}: {required_speedup}x  → {} ({:.2}x measured)",
        if speedup_ok { "PASS" } else { "FAIL" },
        speedup
    );
    eprintln!(
        "[prefix-cache] restore_p95_under_25ms: {} ({restore_ms:.1} ms measured)",
        if restore_ok { "PASS" } else { "FAIL" }
    );
    eprintln!(
        "[prefix-cache] cold/warm first decoded token match: {} (cold={cold_first_id} warm={warm_first_id})",
        if first_token_match { "PASS" } else { "FAIL" }
    );

    // Decode a few more tokens on each path to confirm full convergence.
    if tokens > 1 {
        let mut cold_extra = vec![cold_first_id];
        let mut warm_extra = vec![warm_first_id];
        for k in 1..tokens {
            let pos = (total_len + k) as u32;
            let cold_logits = mf.single_token(
                *cold_extra.last().unwrap(),
                pos,
                seq_cold.metal_session_mut(),
            )?;
            let warm_logits = mf.single_token(
                *warm_extra.last().unwrap(),
                pos,
                seq_warm.metal_session_mut(),
            )?;
            seq_cold.advance_by(1)?;
            seq_warm.advance_by(1)?;
            cold_extra.push(argmax_i32(&cold_logits));
            warm_extra.push(argmax_i32(&warm_logits));
        }
        let same: Vec<bool> = cold_extra
            .iter()
            .zip(warm_extra.iter())
            .map(|(a, b)| a == b)
            .collect();
        let n_same = same.iter().filter(|x| **x).count();
        eprintln!(
            "[prefix-cache] cold/warm decoded sequence agreement: {}/{} tokens ({:.0}%)",
            n_same,
            tokens,
            100.0 * n_same as f64 / tokens as f64
        );
        let cold_text = tok.try_decode(&cold_extra)?;
        let warm_text = tok.try_decode(&warm_extra)?;
        eprintln!("[prefix-cache] cold generated: {:?}", cold_text);
        eprintln!("[prefix-cache] warm generated: {:?}", warm_text);
    }

    Ok(())
}
