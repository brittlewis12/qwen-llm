//! qwen-bench argument structs and the subcommand enum.

use super::*;

#[derive(Parser, Debug)]
#[command(
    name = "qwen-bench",
    version,
    about = "end-to-end throughput benchmark for qwen-llm"
)]
pub(crate) struct Args {
    /// Permit benchmarks from a known dirty source checkout. The dirty state
    /// remains recorded in every canonical JSON row.
    #[arg(long, global = true)]
    pub(crate) allow_dirty: bool,
    /// Permit a benchmark when the compiled binary cannot verify its source
    /// checkout. Commit mismatches are never overridable.
    #[arg(long, global = true)]
    pub(crate) allow_unverifiable_build: bool,
    #[command(subcommand)]
    pub(crate) cmd: Cmd,
}

#[derive(Subcommand, Debug)]
pub(crate) enum Cmd {
    /// Capture sparse true-long prefill attention tensors; this is not a timing benchmark.
    AttnCapture(attn_capture::AttnCaptureArgs),
    /// Price the fixed 32K compressed-KV matrix staging floor.
    AttnStageFloor(attn_stage_floor::AttnStageFloorArgs),
    /// Compare resident serialized execution with independent queue overlap
    /// across every supported model family. This is not yet layer batching.
    #[cfg(feature = "dsv4-diagnostics")]
    QueueOverlapProbe(batch_probe::QueueOverlapProbeArgs),
    /// Execute one complete dense GDN block over independent static slots,
    /// batching every large projection while retaining private recurrent state.
    DecodeDenseBlockBatch(dense_block_batch::DecodeDenseBlockBatchArgs),
    /// Execute one complete dense attention block over independent static
    /// slots, batching front and FFN projections while retaining private KV.
    DecodeDenseAttnBatch(dense_block_batch::DecodeDenseAttnBatchArgs),
    /// Execute a complete dense model over a fixed eight-slot cohort,
    /// comparing production serialization with layer-major weight reuse.
    DecodeDenseWholeBatch(dense_whole_batch::DecodeDenseWholeBatchArgs),
    /// Localize and repair Qwen MoE GDN replay schedule drift at B=16.
    DecodeMoeGdnRepair(moe_gdn_repair::DecodeMoeGdnRepairArgs),
    /// Report compiled and runtime source identity without initializing Metal
    /// or loading a model.
    BuildInfo(BuildInfoArgs),
    /// Report model-agnostic retained-GGUF geometry and byte coverage without loading weights.
    GgufStoragePlan(GgufStoragePlanArgs),
    /// Compare exact topology-preserving GGUF population primitives.
    GgufArenaFloor(gguf_arena_floor::GgufArenaFloorArgs),
    /// Decode N tokens after a prompt using the plain no-spec path.
    ///
    /// Packed prefill is the default no-spec path. `--sequential-prefill`
    /// keeps the legacy token-by-token prompt replay loop for A/B work.
    Decode(DecodeArgs),
    /// Benchmark one resident Muse Glimmer ATEM request without claiming
    /// llama-bench pp/tg comparability.
    MuseRequest(muse_glimmer_request_bench::MuseRequestArgs),
    /// Bounded native K2 raw request wall timings (not llama-bench pp/tg).
    K2Request(k2_request_bench::K2RequestArgs),
    /// Prompt-only prefill benchmark aligned with llama-bench pp semantics.
    Pp(PpArgs),
    /// Profile native DeepSeek V4 packed prefill with production policy.
    #[cfg(feature = "dsv4-diagnostics")]
    Dsv4Prefill(dsv4_prefill::Dsv4PrefillArgs),
    /// Decide the frozen K160 mHC representation-deletion ceiling.
    #[cfg(feature = "dsv4-diagnostics")]
    Dsv4MhcDelete(dsv4_mhc_delete::Dsv4MhcDeleteArgs),
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
    /// **v0.77** small-N mat-mat kernel-variant sweep on production
    /// shapes at N=8: times the v0.501 table pick against the generic
    /// tile, sequential mat-vec, nc8, and every N=8-capable mma8 variant
    /// per weight family (GDN qkv/z/out, attn q/o, FFN gate/down,
    /// lm_head). The 2026-08-19 verify microbench showed packed-verify
    /// cost is per-dispatch kernel efficiency c(n); this finds free wins
    /// before any kernel engineering.
    MatmatSmallnMicro(MatmatSmallnMicroArgs),
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
    /// Compare native, shared-head, and minimax RoPE kernels without a model.
    RopeMicro(rope_micro::RopeMicroArgs),
    /// Attribute the production-grid Q4_K N64 schedule with synthetic bounds.
    Q4MmaCeiling(q4_mma_ceiling::Q4MmaCeilingArgs),
    /// Measure the charged exact grammar-row Q6_K lm-head floor.
    GrammarLmHeadRowFloor(grammar_lm_head_row_floor::GrammarLmHeadRowFloorArgs),
    /// Measure one exact A3B request with integrated grammar-row heads.
    IntegratedGrammarRow(integrated_grammar_row::IntegratedGrammarRowArgs),
    /// Acquire the frozen v0.664 A3B lm-head screening packet.
    LmHeadScreeningOracle(lm_head_screening_oracle::LmHeadScreeningOracleArgs),
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
    #[command(hide = true)]
    PrefixCacheVtAb(prefix_cache_vt_ab::PrefixCacheVtAbArgs),
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
    /// Exact serial E1a sampled one-hot DFlash development oracle. E0 is
    /// explicitly unmeasured; this is not a performance benchmark.
    DflashSampledOracle(dflash_sampled_oracle::DflashSampledOracleArgs),
    /// Exact token-major E0 lockstep development harness for ordinary target
    /// decode versus multi-hidden capture. Grants no product authority.
    DflashE0Lockstep(dflash_e0_lockstep::DflashE0LockstepArgs),
    /// Emit one bounded DFlash K0-S selector lattice packet.
    #[cfg(feature = "dflash-k0s-diagnostics")]
    #[command(hide = true)]
    DflashK0sLattice(dflash_k0s::DflashK0sArgs),
    #[cfg(feature = "dflash-k0s-diagnostics")]
    #[command(hide = true)]
    DflashK0sInventory(dflash_k0s::DflashK0sInventoryArgs),
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
pub(crate) struct BuildInfoArgs {
    /// Output format. JSON emits one object rather than a benchmark-row array.
    #[arg(short = 'o', long, value_enum, default_value = "json")]
    pub(crate) output: OutputFormat,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum GgufStorageEmbeddingPolicy {
    ProductionAuto,
    ForceNativeIfSupported,
    Disabled,
}

impl GgufStorageEmbeddingPolicy {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::ProductionAuto => "production-auto",
            Self::ForceNativeIfSupported => "force-native-if-supported",
            Self::Disabled => "disabled",
        }
    }
}

#[derive(Parser, Debug)]
pub(crate) struct GgufStoragePlanArgs {
    /// Path to the first GGUF shard.
    #[arg(short = 'm', long)]
    pub(crate) model: PathBuf,
    /// Required Metal buffer binding alignment.
    #[arg(long, default_value = "32")]
    pub(crate) alignment: usize,
    /// Override the device's maxBufferLength for geometry falsification.
    #[arg(long)]
    pub(crate) max_buffer_length: Option<usize>,
    /// Token-embedding materialization policy used by the copied baseline.
    #[arg(long, value_enum, default_value = "production-auto")]
    pub(crate) embedding_policy: GgufStorageEmbeddingPolicy,
    /// Price the opt-in F16 MoE router conversion.
    #[arg(long)]
    pub(crate) router_f16: bool,
    /// `text` or `json`.
    #[arg(short = 'o', long, value_enum, default_value = "json")]
    pub(crate) output: OutputFormat,
}

#[derive(Parser, Debug)]
pub(crate) struct DecodeArgs {
    /// Path to a GGUF file.
    #[arg(short = 'm', long)]
    pub(crate) model: PathBuf,
    /// Prompt text. If absent, uses a fixed warmup prompt.
    #[arg(short = 'p', long)]
    pub(crate) prompt: Option<String>,
    /// Number of tokens to generate after the prompt.
    #[arg(long, default_value = "64")]
    pub(crate) tokens: usize,
    /// Optional oracle file (raw f32 logits at last position from
    /// llama.cpp/llm `--snapshot`). If provided, compares cos.
    #[arg(long)]
    pub(crate) oracle: Option<PathBuf>,
    /// Which logits row the oracle should validate.
    #[arg(long, value_enum, default_value = "final")]
    pub(crate) oracle_phase: OraclePhase,
    /// Skip the warmup pass (default is to do one warmup, then re-init
    /// the session for the timed run, exactly like the ignored tests).
    #[arg(long)]
    pub(crate) no_warmup: bool,
    /// Force the legacy sequential prompt replay loop. Useful for A/B timing
    /// against the dense packed prefill path.
    #[arg(long)]
    pub(crate) sequential_prefill: bool,
    /// Packed prefill chunk size for the layer-major path. If omitted, decode
    /// chooses a model-aware default (currently dense=256, MoE=16).
    #[arg(long)]
    pub(crate) prefill_chunk: Option<usize>,
    /// Override decode session KV capacity for capacity-sensitivity tests.
    /// Must be at least prompt_tokens + generation_tokens + 16.
    #[arg(long)]
    pub(crate) kv_capacity: Option<usize>,
    /// Force decode to read back full logits on every generated token instead
    /// of using the GPU argmax fast path.
    #[arg(long)]
    pub(crate) full_logits_decode: bool,
    /// Number of timed repetitions. Each rep re-tokenizes, re-prefills, and
    /// re-decodes from a fresh session. avg_ts / stddev_ts are over reps.
    #[arg(long, default_value = "1")]
    pub(crate) runs: usize,
    /// Print the exact initial token plus every timed transition result.
    #[arg(long)]
    pub(crate) generated_token_trace: bool,
    /// `text` or `json` (`llama-bench -o json` shape).
    #[arg(short = 'o', long, value_enum, default_value = "text")]
    pub(crate) output: OutputFormat,
}

#[derive(Parser, Debug)]
pub(crate) struct MetalCountersArgs {}

#[derive(Parser, Debug)]
pub(crate) struct MetalPipelinesArgs {
    /// Kernel names to inspect. If omitted, prints the hot decode audit set.
    #[arg(long = "kernel", value_delimiter = ',')]
    pub(crate) kernels: Vec<String>,
}

#[derive(Parser, Debug)]
pub(crate) struct DispatchCensusArgs {
    /// Path to a GGUF file (MoE arch; uses the stage-profiled decode entry).
    #[arg(short = 'm', long)]
    pub(crate) model: PathBuf,
    /// Context position to census at (KV warmed to this depth first).
    #[arg(long, default_value = "16384")]
    pub(crate) ctx: usize,
    /// Warm via the token-by-token decode ramp instead of packed prefill.
    #[arg(long)]
    pub(crate) decode_ramp_warm: bool,
    /// Output JSON path.
    #[arg(long, default_value = "target/profiles/dispatch-census/census.json")]
    pub(crate) out: PathBuf,
}

#[derive(Parser, Debug)]
pub(crate) struct TopologyProbeArgs {
    /// Arms to run: comma list of r,s,d or `all`. Arm D needs arm R results
    /// (or --grid-tgs) for conservative persistent-grid sizing.
    #[arg(long, default_value = "all")]
    pub(crate) arm: String,
    /// Timed repetitions per configuration (median reported).
    #[arg(long, default_value = "5")]
    pub(crate) runs: usize,
    /// Output directory for JSON artifacts.
    #[arg(long, default_value = "target/profiles/topology-probe")]
    pub(crate) out_dir: PathBuf,
    /// Override the persistent-grid TG count for arm D (default: arm R
    /// low-water p10 of the steady entry-alive distribution).
    #[arg(long)]
    pub(crate) grid_tgs: Option<usize>,
    /// Smoke mode: smaller grids/epochs/dwells for a fast end-to-end pass.
    #[arg(long)]
    pub(crate) quick: bool,
    /// Extra arm-R dwell points (us) for plateau confirmation, e.g.
    /// `--dwell-extend 15000`. Runs lo/no-traffic variants only.
    #[arg(long, value_delimiter = ',')]
    pub(crate) dwell_extend: Vec<f64>,
}

#[derive(Parser, Debug)]
pub(crate) struct PpArgs {
    /// Path to a GGUF file.
    #[arg(short = 'm', long)]
    pub(crate) model: PathBuf,
    /// Synthetic prompt token count, matching llama-bench's pp<N> shape.
    #[arg(
        short = 'p',
        long = "n-prompt",
        alias = "tokens",
        default_value = "320"
    )]
    pub(crate) n_prompt: usize,
    /// Optional real prompt text. If set, --n-prompt is ignored.
    #[arg(long, conflicts_with_all = ["file", "messages"])]
    pub(crate) prompt: Option<String>,
    /// Read prompt text from a file. If set, --n-prompt is ignored.
    #[arg(long, conflicts_with = "messages")]
    pub(crate) file: Option<PathBuf>,
    /// Render a JSON messages input into a Qwen chat-template prompt.
    ///
    /// Accepted shapes:
    /// - bare `[{ role, content }, ...]`
    /// - wrapped `{ messages: [...], ... }`
    #[arg(long)]
    pub(crate) messages: Option<PathBuf>,
    /// Use only the first N messages from `--messages` before rendering.
    #[arg(long)]
    pub(crate) messages_max: Option<usize>,
    /// Preserve assistant `<think>...</think>` history from `--messages`.
    #[arg(long)]
    pub(crate) messages_preserve_thinking: bool,
    /// Force stripping assistant `<think>...</think>` history from
    /// `--messages`, even if auto-detection would preserve it.
    #[arg(long, conflicts_with = "messages_preserve_thinking")]
    pub(crate) messages_strip_thinking: bool,
    /// Do not append a final `<|im_start|>assistant\n` generation marker for
    /// `--messages` prompts.
    #[arg(long)]
    pub(crate) messages_no_generation_prompt: bool,
    /// Number of timed repetitions after warmup.
    #[arg(long, default_value = "5")]
    pub(crate) runs: usize,
    /// Skip the warmup prefill pass.
    #[arg(long)]
    pub(crate) no_warmup: bool,
    /// Packed prefill chunk size. If omitted, uses the model-aware default.
    #[arg(long)]
    pub(crate) prefill_chunk: Option<usize>,
    /// Include final norm + lm_head + logits readback, like decode's prefill seed.
    #[arg(long)]
    pub(crate) with_tail: bool,
    /// Deterministic seed for synthetic token generation.
    #[arg(long, default_value = "1")]
    pub(crate) seed: u64,
    /// `text` or `json` (`llama-bench -o json` shape).
    #[arg(short = 'o', long, value_enum, default_value = "text")]
    pub(crate) output: OutputFormat,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum OraclePhase {
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
pub(crate) struct TgArgs {
    #[arg(short = 'm', long)]
    pub(crate) model: PathBuf,
    /// Number of tokens to generate per timed rep.
    #[arg(short = 'n', long = "n-gen", default_value = "128")]
    pub(crate) n_gen: usize,
    /// Number of timed reps after warmup.
    #[arg(long, default_value = "3")]
    pub(crate) runs: usize,
    /// Skip the warmup pass.
    #[arg(long)]
    pub(crate) no_warmup: bool,
    /// Bench-only CPU/GPU overlap path: encode token N+1 while token N is
    /// executing on the GPU. Commands are still committed serially.
    #[arg(long)]
    pub(crate) pipelined: bool,
    /// Bench-only GDN front-projection overlap path.
    #[arg(long)]
    pub(crate) concurrent_gdn_proj: bool,
    /// Deterministic seed for random token selection.
    #[arg(long, default_value = "1")]
    pub(crate) seed: u64,
    /// `text` or `json` (`llama-bench -o json` shape).
    #[arg(short = 'o', long, value_enum, default_value = "text")]
    pub(crate) output: OutputFormat,
}

/// Synthetic multi-shape suite that keeps one loaded model resident.
///
/// Each reported row still uses a fresh sequence/session for each measured
/// repetition. This removes repeated process/model-load overhead without
/// changing the steady-state pp/tg semantics used by the single-shape commands.
#[derive(Parser, Debug)]
pub(crate) struct SuiteArgs {
    #[arg(short = 'm', long)]
    pub(crate) model: PathBuf,
    /// Prompt-only prefill shapes. Accepts repeated flags or comma lists.
    #[arg(long = "pp", value_delimiter = ',')]
    pub(crate) pp: Vec<usize>,
    /// Generation-only decode shapes. Accepts repeated flags or comma lists.
    #[arg(long = "tg", value_delimiter = ',')]
    pub(crate) tg: Vec<usize>,
    /// Context depths (llama-bench `-d`): each pp/tg row runs at every depth
    /// after an untimed fill of that many tokens. Accepts comma lists.
    #[arg(
        short = 'd',
        long = "depth",
        value_delimiter = ',',
        default_value = "0"
    )]
    pub(crate) depth: Vec<usize>,
    /// Number of timed reps after each row's optional warmup.
    #[arg(long, default_value = "1")]
    pub(crate) runs: usize,
    /// Skip each row's warmup pass.
    #[arg(long)]
    pub(crate) no_warmup: bool,
    /// Fixed Qwen prefill chunk for pp rows. If omitted, the production
    /// allocator decides per row, as `qwen run` does (auto; MoE prompts over
    /// 1024 tokens take 2048-token chunks when admitted).
    #[arg(long)]
    pub(crate) prefill_chunk: Option<usize>,
    /// Deterministic seed for synthetic tokens.
    #[arg(long, default_value = "1")]
    pub(crate) seed: u64,
    /// `text` or `json` (`llama-bench -o json` shape). Defaults to JSON because
    /// suite output usually feeds scripts.
    #[arg(short = 'o', long, value_enum, default_value = "json")]
    pub(crate) output: OutputFormat,
}

#[derive(Parser, Debug)]
pub(crate) struct CtxSweepArgs {
    /// Path to a GGUF file.
    #[arg(short = 'm', long)]
    pub(crate) model: PathBuf,
    /// Comma-separated context checkpoints to measure at.
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "1,64,256,1024,4096,8192,16384"
    )]
    pub(crate) checkpoints: Vec<usize>,
    /// How many tokens to time at each checkpoint.
    #[arg(long, default_value = "5")]
    pub(crate) window: usize,
    /// Use the bench-only dense path that splits GDN blocks across encoders and
    /// runs the four front projections in a concurrent compute encoder.
    #[arg(long)]
    pub(crate) concurrent_gdn_proj: bool,
    /// Use the bench-only dense path that splits attention blocks across
    /// encoders and runs the q/k/v front projections in a concurrent compute
    /// encoder.
    #[arg(long)]
    pub(crate) concurrent_attn_proj: bool,
    /// Allocate a fresh right-sized session for each checkpoint instead of one
    /// max-capacity session for the whole sweep. Slower, but avoids large unused
    /// KV capacity poisoning earlier checkpoints on memory-pressure-sensitive
    /// models.
    #[arg(long)]
    pub(crate) fresh_per_checkpoint: bool,
    /// Warm each checkpoint with the production packed-prefill path instead of
    /// the token-by-token decode ramp (requires --fresh-per-checkpoint).
    #[arg(long)]
    pub(crate) prefill_warm: bool,
}

#[derive(Parser, Debug)]
pub(crate) struct PhaseArgs {
    /// Path to a GGUF file.
    #[arg(short = 'm', long)]
    pub(crate) model: PathBuf,
    /// Context length to profile at.
    #[arg(long, default_value = "4096")]
    pub(crate) ctx: usize,
}

#[derive(Parser, Debug)]
pub(crate) struct AttnIntraArgs {
    /// Path to a GGUF file.
    #[arg(short = 'm', long)]
    pub(crate) model: PathBuf,
    /// Context length to ramp before profiling one attention layer.
    #[arg(long, default_value = "32768")]
    pub(crate) ctx: usize,
    /// Timed single-layer repetitions after the ramp.
    #[arg(long, default_value = "3")]
    pub(crate) runs: usize,
    /// Optional absolute block index. Defaults to the first full-attention block.
    #[arg(long)]
    pub(crate) block: Option<usize>,
}

#[derive(Parser, Debug)]
pub(crate) struct GdnProjMicroArgs {
    /// Path to a GGUF file.
    #[arg(short = 'm', long)]
    pub(crate) model: PathBuf,
    /// Timed repetitions after warmup.
    #[arg(long, default_value = "20")]
    pub(crate) iters: usize,
    /// Untimed warmup repetitions.
    #[arg(long, default_value = "5")]
    pub(crate) warmup: usize,
    /// Synthetic token rows for the mat-mat batch path.
    #[arg(long, default_value = "1")]
    pub(crate) tokens: usize,
}

#[derive(Parser, Debug)]
pub(crate) struct MatmatSmallnMicroArgs {
    /// Path to a GGUF file.
    #[arg(short = 'm', long)]
    pub(crate) model: PathBuf,
    /// Timed repetitions after warmup.
    #[arg(long, default_value = "10")]
    pub(crate) iters: usize,
    /// Untimed warmup repetitions.
    #[arg(long, default_value = "3")]
    pub(crate) warmup: usize,
    /// Max tensors dispatched per family per rep (caps rep cost; the
    /// dense 27B has 48 GDN / 16 attn / 64 FFN instances per family).
    #[arg(long, default_value = "8")]
    pub(crate) tensors_per_family: usize,
}

#[derive(Parser, Debug)]
pub(crate) struct DecodeProjBatchArgs {
    /// Path to a GGUF file.
    #[arg(short = 'm', long)]
    pub(crate) model: PathBuf,
    /// Comma-separated token counts to replay through batched mat-mat kernels.
    #[arg(long, value_delimiter = ',', default_value = "1,2,4,8,16")]
    pub(crate) tokens: Vec<usize>,
    /// Timed repetitions after warmup.
    #[arg(long, default_value = "10")]
    pub(crate) iters: usize,
    /// Untimed warmup repetitions.
    #[arg(long, default_value = "3")]
    pub(crate) warmup: usize,
}

#[derive(Parser, Debug)]
pub(crate) struct DecodeGdnLayerReplayArgs {
    /// Path to a GGUF file.
    #[arg(short = 'm', long)]
    pub(crate) model: PathBuf,
    /// Comma-separated token counts to replay through one GDN layer.
    #[arg(long, value_delimiter = ',', default_value = "1,2,4,8,16")]
    pub(crate) tokens: Vec<usize>,
    /// Optional absolute block index. Defaults to the first GDN block.
    #[arg(long)]
    pub(crate) block: Option<usize>,
    /// GDN-layer indexes to measure. Accepts repeated flags or comma lists.
    #[arg(long = "gdn-index", value_delimiter = ',')]
    pub(crate) gdn_indexes: Vec<usize>,
    /// Measure first, middle, and last GDN layers in one model load.
    #[arg(long)]
    pub(crate) sample_gdn_layers: bool,
    /// Timed repetitions after warmup.
    #[arg(long, default_value = "5")]
    pub(crate) iters: usize,
    /// Untimed warmup repetitions.
    #[arg(long, default_value = "2")]
    pub(crate) warmup: usize,
    /// Skip the all-slot correctness comparison between baseline and replay.
    #[arg(long)]
    pub(crate) no_check: bool,
}

#[derive(Parser, Debug)]
pub(crate) struct DecodeGdnChainReplayArgs {
    /// Path to a GGUF file.
    #[arg(short = 'm', long)]
    pub(crate) model: PathBuf,
    /// Comma-separated token counts to replay through the GDN chain.
    #[arg(long, value_delimiter = ',', default_value = "8,16")]
    pub(crate) tokens: Vec<usize>,
    /// First GDN-layer index in the chain.
    #[arg(long, default_value = "0")]
    pub(crate) start_gdn: usize,
    /// Number of consecutive GDN layers to chain.
    #[arg(long = "layers", default_value = "4")]
    pub(crate) n_layers: usize,
    /// Timed repetitions after warmup.
    #[arg(long, default_value = "3")]
    pub(crate) iters: usize,
    /// Untimed warmup repetitions.
    #[arg(long, default_value = "1")]
    pub(crate) warmup: usize,
    /// Skip the all-slot correctness comparison between baseline and replay.
    #[arg(long)]
    pub(crate) no_check: bool,
}

#[derive(Parser, Debug)]
pub(crate) struct DecodeBlockSliceReplayArgs {
    /// Path to a GGUF file.
    #[arg(short = 'm', long)]
    pub(crate) model: PathBuf,
    /// Comma-separated token counts to replay through the block slice.
    #[arg(long, value_delimiter = ',', default_value = "8,16")]
    pub(crate) tokens: Vec<usize>,
    /// First absolute transformer block index in the slice.
    #[arg(long, default_value = "0")]
    pub(crate) start_block: usize,
    /// Number of consecutive absolute blocks in the slice.
    #[arg(long = "blocks", default_value = "4")]
    pub(crate) n_blocks: usize,
    /// Synthetic decode position for attention blocks in the slice.
    #[arg(long, default_value = "0")]
    pub(crate) position: u32,
    /// Timed repetitions after warmup.
    #[arg(long, default_value = "3")]
    pub(crate) iters: usize,
    /// Untimed warmup repetitions.
    #[arg(long, default_value = "1")]
    pub(crate) warmup: usize,
    /// Skip the all-slot correctness comparison between baseline and replay.
    #[arg(long)]
    pub(crate) no_check: bool,
}

#[derive(Parser, Debug)]
pub(crate) struct DecodeBlockSliceTraceArgs {
    /// Path to a GGUF file.
    #[arg(short = 'm', long)]
    pub(crate) model: PathBuf,
    /// Number of synthetic slots to trace.
    #[arg(long, default_value = "8")]
    pub(crate) tokens: usize,
    /// First absolute transformer block index in the slice.
    #[arg(long, default_value = "0")]
    pub(crate) start_block: usize,
    /// Number of consecutive absolute blocks in the slice.
    #[arg(long = "blocks", default_value = "4")]
    pub(crate) n_blocks: usize,
    /// Synthetic decode position for attention blocks in the slice.
    #[arg(long, default_value = "0")]
    pub(crate) position: u32,
}

#[derive(Parser, Debug)]
pub(crate) struct DecodeBlockSliceMarginSweepArgs {
    /// Path to a GGUF file.
    #[arg(short = 'm', long)]
    pub(crate) model: PathBuf,
    /// Number of synthetic slots per traced window.
    #[arg(long, default_value = "8")]
    pub(crate) tokens: usize,
    /// Start blocks to sweep. If omitted, uses a non-overlapping stride.
    #[arg(long = "start-block", value_delimiter = ',')]
    pub(crate) start_blocks: Vec<usize>,
    /// Number of consecutive absolute blocks per window.
    #[arg(long = "blocks", default_value = "4")]
    pub(crate) n_blocks: usize,
    /// Synthetic decode positions to sweep.
    #[arg(long = "position", value_delimiter = ',', default_value = "0,4096")]
    pub(crate) positions: Vec<u32>,
}

#[derive(Parser, Debug)]
pub(crate) struct DecodeBlockSliceRealMarginArgs {
    /// Path to a GGUF file.
    #[arg(short = 'm', long)]
    pub(crate) model: PathBuf,
    /// Read one or more real prompt token streams from text files.
    #[arg(long)]
    pub(crate) file: Vec<PathBuf>,
    /// Number of consecutive prompt positions to use as slots.
    #[arg(long, default_value = "4")]
    pub(crate) tokens: usize,
    /// Slot counts to measure after preparing the maximum slot prefix set.
    #[arg(long = "slot-counts", value_delimiter = ',')]
    pub(crate) slot_counts: Vec<usize>,
    /// Prompt context positions to sweep.
    #[arg(long = "context", value_delimiter = ',', default_value = "512")]
    pub(crate) contexts: Vec<usize>,
    /// Position stride between slots when one prompt file supplies multiple slots.
    #[arg(long, default_value = "1")]
    pub(crate) stride: usize,
    /// Start blocks to sweep. If omitted, uses a non-overlapping stride.
    #[arg(long = "start-block", value_delimiter = ',')]
    pub(crate) start_blocks: Vec<usize>,
    /// Number of consecutive absolute blocks per replay window.
    #[arg(long = "blocks", default_value = "4")]
    pub(crate) n_blocks: usize,
    /// Timed repetitions for optional real-window economics. Zero keeps this as
    /// a margin-only probe.
    #[arg(long, default_value = "0")]
    pub(crate) timing_iters: usize,
    /// Untimed warmup repetitions for optional real-window economics.
    #[arg(long, default_value = "1")]
    pub(crate) timing_warmup: usize,
    /// Replay-margin threshold used by optional economics fallback modeling.
    #[arg(long, default_value = "0.0003")]
    pub(crate) margin_threshold: f32,
}

#[derive(Parser, Debug)]
pub(crate) struct DecodeMoeRouterRepackCheckArgs {
    /// Path to a GGUF file.
    #[arg(short = 'm', long)]
    pub(crate) model: PathBuf,
    /// Read one or more real prompt token streams from text files.
    #[arg(long)]
    pub(crate) file: Vec<PathBuf>,
    /// Number of prompt files to use as independent slots.
    #[arg(long, default_value = "4")]
    pub(crate) tokens: usize,
    /// Prompt context positions to sweep.
    #[arg(long = "context", value_delimiter = ',', default_value = "512")]
    pub(crate) contexts: Vec<usize>,
}

#[derive(Parser, Debug)]
pub(crate) struct MoeDownMicroArgs {
    /// Path to a GGUF file.
    #[arg(short = 'm', long)]
    pub(crate) model: PathBuf,
    /// Timed repetitions after warmup.
    #[arg(long, default_value = "20")]
    pub(crate) iters: usize,
    /// Untimed warmup repetitions.
    #[arg(long, default_value = "5")]
    pub(crate) warmup: usize,
    /// Synthetic tokens routed through the packed-slots kernel.
    #[arg(long, default_value = "1")]
    pub(crate) tokens: usize,
    /// Time the existing one-token fused Q4/Q5 routed FFN monolith.
    #[arg(long)]
    pub(crate) fused_routed_q4q5: bool,
    /// Capture per-layer route ids/weights at this decode context.
    #[arg(long)]
    pub(crate) route_capture_ctx: Option<usize>,
    /// Token-id pattern used for captured replay tokens.
    #[arg(long, value_enum, default_value = "zero")]
    pub(crate) route_capture_token_pattern: CaptureTokenPattern,
    /// Override routed-down K dimension with a synthetic zero Q5_K bank.
    #[arg(long)]
    pub(crate) synthetic_f_exp: Option<usize>,
    /// Override routed-down output dimension with a synthetic zero Q5_K bank.
    #[arg(long)]
    pub(crate) synthetic_h: Option<usize>,
    /// Number of synthetic layer dispatches. Defaults to the real Q5 layer count.
    #[arg(long)]
    pub(crate) synthetic_layers: Option<usize>,
    /// Force the f_exp=512 two-row-per-simdgroup Q5 down kernel.
    #[arg(long)]
    pub(crate) k512_r2: bool,
    /// Use the legacy f_exp=512 Q5 down kernel instead of the production R2 path.
    #[arg(long)]
    pub(crate) legacy_k512: bool,
    /// Compare default vs --k512-r2 output before timing.
    #[arg(long)]
    pub(crate) check_k512_r2: bool,
}

#[derive(Parser, Debug)]
pub(crate) struct MoeGateupMicroArgs {
    /// Path to a GGUF file.
    #[arg(short = 'm', long)]
    pub(crate) model: PathBuf,
    /// Timed repetitions after warmup.
    #[arg(long, default_value = "20")]
    pub(crate) iters: usize,
    /// Untimed warmup repetitions.
    #[arg(long, default_value = "5")]
    pub(crate) warmup: usize,
    /// Tokens to replay through the packed-slots kernel.
    #[arg(long, default_value = "1")]
    pub(crate) tokens: usize,
    /// Capture per-layer hidden activations and top-k ids at this decode context.
    #[arg(long)]
    pub(crate) route_capture_ctx: Option<usize>,
    /// Token-id pattern used for captured replay tokens.
    #[arg(long, value_enum, default_value = "zero")]
    pub(crate) route_capture_token_pattern: CaptureTokenPattern,
}

#[derive(Parser, Debug)]
pub(crate) struct MoeBatchSweepArgs {
    /// Path to a GGUF file.
    #[arg(short = 'm', long)]
    pub(crate) model: PathBuf,
    /// Timed repetitions after warmup.
    #[arg(long, default_value = "10")]
    pub(crate) iters: usize,
    /// Untimed warmup repetitions.
    #[arg(long, default_value = "3")]
    pub(crate) warmup: usize,
    /// Comma-separated token counts to replay.
    #[arg(long, value_delimiter = ',', default_value = "1,2,4,8,16")]
    pub(crate) tokens: Vec<usize>,
    /// Capture per-layer hidden activations and top-k ids at this decode context.
    #[arg(long, default_value = "1024")]
    pub(crate) route_capture_ctx: usize,
    /// Position stride between captured replay tokens.
    #[arg(long, default_value = "1")]
    pub(crate) route_capture_stride: usize,
    /// Read one or more real prompt token streams from text files.
    #[arg(long)]
    pub(crate) file: Vec<PathBuf>,
    /// Token-id pattern used for captured replay tokens.
    #[arg(long, value_enum, default_value = "ramp")]
    pub(crate) route_capture_token_pattern: CaptureTokenPattern,
    /// Route slot order for packed replay. The expert-sorted mode is a
    /// perf-only locality upper bound, not a correctness-preserving replay.
    #[arg(
        long = "slot-order",
        value_enum,
        value_delimiter = ',',
        default_value = "exact"
    )]
    pub(crate) slot_orders: Vec<MoeBatchSlotOrder>,
}

#[derive(Parser, Debug)]
pub(crate) struct RooflineArgs {
    /// Per-buffer stream size in MiB. Stream bytes/rep are nominally 3x this.
    #[arg(long, default_value = "512")]
    pub(crate) stream_mib: usize,
    /// Elements for the compute-loop kernel.
    #[arg(long, default_value = "4194304")]
    pub(crate) fma_elements: usize,
    /// FMA iterations per element. Nominal FLOPs are 2*N*iters.
    #[arg(long, default_value = "4096")]
    pub(crate) fma_iters: usize,
    /// Q4_K mat-mat input width. Must be divisible by 256.
    #[arg(long, default_value = "4096")]
    pub(crate) mat_in: usize,
    /// Q4_K mat-mat output width. Use multiples of 64 for the tuned path.
    #[arg(long, default_value = "4096")]
    pub(crate) mat_out: usize,
    /// Q4_K mat-mat batch/query rows. Use multiples of 64 for the tuned path.
    #[arg(long, default_value = "1024")]
    pub(crate) mat_query: usize,
    /// Timed repetitions after one warmup dispatch per kernel.
    #[arg(long, default_value = "5")]
    pub(crate) runs: usize,
    /// `text` or compact `json`.
    #[arg(short = 'o', long, value_enum, default_value = "text")]
    pub(crate) output: OutputFormat,
}

#[derive(Parser, Debug)]
pub(crate) struct DecodeWindowArgs {
    /// Path to a GGUF file.
    #[arg(short = 'm', long)]
    pub(crate) model: PathBuf,
    /// Context length to ramp to before waiting.
    #[arg(long)]
    pub(crate) target_ctx: usize,
    /// Warm the KV/GDN state with the production packed-prefill path instead
    /// of the token-by-token decode ramp. Orders of magnitude faster to deep
    /// contexts; validate against a decode-ramp point before trusting new
    /// context regimes (v0.494 validation: ctx16384 matches within noise).
    #[arg(long)]
    pub(crate) prefill_warm: bool,
    /// Number of decode tokens to execute after the go signal.
    #[arg(long, default_value = "128")]
    pub(crate) window: usize,
    /// Independent decode streams to issue from separate command queues.
    /// Intended only for occupancy/counter discrimination; each stream owns
    /// separate KV/GDN state while sharing resident model weights.
    #[arg(long, default_value = "1")]
    pub(crate) streams: usize,
    /// Use Metal timestamp counter samples around existing decode-stage
    /// encoder boundaries. Bench-only attribution probe for single-stream MoE.
    #[arg(long)]
    pub(crate) stage_timestamps: bool,
    /// Under --stage-timestamps, split attention mixer work from MoE route prep.
    /// This is an attribution-only second-level probe and changes encoder shape.
    #[arg(long)]
    pub(crate) stage_split_attn_route: bool,
    /// Under --stage-timestamps, split attention blocks into pre-norm, front
    /// projections, attention body/output, residual+post-norm, and route prep.
    #[arg(long)]
    pub(crate) stage_split_attn_detail: bool,
    /// Under --stage-timestamps, split the serial GDN after-projection bucket
    /// into beta/alpha prep, GDN tail, output projection, post-norm, and route.
    #[arg(long)]
    pub(crate) stage_split_gdn_after: bool,
    /// File created when the process has reached `target_ctx` and is waiting.
    #[arg(long)]
    pub(crate) ready_file: PathBuf,
    /// File whose existence releases the process to run the decode window.
    #[arg(long)]
    pub(crate) go_file: PathBuf,
    /// Use a bench-only pipelined dense decode loop that overlaps CPU encoding of
    /// token N+1 with GPU execution of token N.
    #[arg(long)]
    pub(crate) pipelined: bool,
    /// Use a bench-only dense decode path that splits GDN blocks across multiple
    /// encoders and runs the four front projections in a concurrent compute
    /// encoder.
    #[arg(long)]
    pub(crate) concurrent_gdn_proj: bool,
    /// Use a bench-only dense decode path that splits attention blocks across
    /// encoders and runs q/k/v front projections in a concurrent compute
    /// encoder.
    #[arg(long)]
    pub(crate) concurrent_attn_proj: bool,
}

#[derive(Parser, Debug)]
pub(crate) struct PrefixCacheArgs {
    /// Path to a GGUF file.
    #[arg(short = 'm', long)]
    pub(crate) model: PathBuf,
    /// Shared prefix prompt (used for cold prefill of request 1, then
    /// cached). Pad with --prefix-pad-tokens to hit a target prefix len.
    #[arg(short = 'p', long, default_value = "You are a helpful assistant.")]
    pub(crate) prefix: String,
    /// Optionally pad the prefix to a target token count by repeating
    /// "lorem ipsum" filler. Used to hit specific prefix lengths
    /// (per codex's H2 kill criteria: 64, 256, 1024, 4096).
    #[arg(long)]
    pub(crate) target_prefix_len: Option<usize>,
    /// Suffix prompt for request 2 (concatenated to the cached prefix).
    #[arg(long, default_value = "\n\nUser: What time is it?\nAssistant:")]
    pub(crate) suffix: String,
    /// Decode tokens to generate after each request's prefill.
    #[arg(long, default_value = "8")]
    pub(crate) tokens: usize,
    /// Prefill implementation used by the cache probe.
    #[arg(long, value_enum, default_value_t = PrefixCachePrefillMode::Packed)]
    pub(crate) prefill_mode: PrefixCachePrefillMode,
    /// Prefill implementation for the post-hit suffix.
    #[arg(long, value_enum, default_value_t = PrefixCacheSuffixMode::Auto)]
    pub(crate) suffix_prefill_mode: PrefixCacheSuffixMode,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum PrefixCachePrefillMode {
    /// Product-shaped packed prefill path.
    Packed,
    /// Legacy per-token loop, retained as a diagnostic control.
    Single,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum PrefixCacheSuffixMode {
    /// Use the per-token path for short suffixes and packed path otherwise.
    Auto,
    /// Force product-shaped packed prefill for the suffix.
    Packed,
    /// Force the per-token loop for the suffix.
    Single,
}

pub(crate) fn choose_prefix_cache_suffix_mode(
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
pub(crate) struct MtpArgs {
    /// Path to an MTP-aware GGUF file.
    #[arg(short = 'm', long)]
    pub(crate) model: PathBuf,
    /// Prompt text. Use a non-trivial prompt for honest acceptance rates.
    #[arg(
        short = 'p',
        long,
        default_value = "The quick brown fox jumps over the lazy dog"
    )]
    pub(crate) prompt: String,
    /// Render the prompt through a Qwen chat template instead of treating
    /// `--prompt` as raw text. Useful for realistic thinking-mode evals.
    #[arg(long)]
    pub(crate) qwen_chat: bool,
    /// Optional system prompt for `--qwen-chat` rendering.
    #[arg(long)]
    pub(crate) system: Option<String>,
    /// For `--qwen-chat`, render the assistant generation prompt with an
    /// empty `<think>...</think>` block instead of an open thinking block.
    #[arg(long)]
    pub(crate) disable_thinking: bool,
    /// Experimental speculative depth. `1` defaults to the original H4
    /// lazy-verify path; combine it with `--mtp-physical-n 2` for ordinary
    /// packed D1/N2 verification. `2..=15` recursively chain MTP drafts and
    /// verify them with the packed base path.
    #[arg(long, default_value = "1")]
    pub(crate) spec_tokens: usize,
    /// Bench-only probe for pricing native-MTP draft overhead.
    #[arg(long, value_enum, default_value_t = MtpProbeMode::Normal)]
    pub(crate) mtp_probe: MtpProbeMode,
    /// Select packed verification and its physical N. For D1, pass `2` to
    /// compare standard packed speculation with the legacy lazy path. When N
    /// exceeds `1 + --spec-tokens`, padded positions are rolled back after the
    /// logical accept window.
    #[arg(long)]
    pub(crate) mtp_physical_n: Option<usize>,
    /// Chain all recursive MTP draft slots into one command buffer.
    #[arg(long)]
    pub(crate) mtp_single_cb_draft: bool,
    /// Use token_embd.weight as a cheap Q4 draft-only LM head. Bench falsifier;
    /// target verify still uses the real output.weight.
    #[arg(long)]
    pub(crate) mtp_draft_token_embd_head: bool,
    /// Quantize output.weight to Q4_1 at setup and use it as draft-only lm_head.
    /// Bench probe for MTPLX-style low-bit draft heads.
    #[arg(long)]
    pub(crate) mtp_draft_lm_head_q4_1: bool,
    /// Quantize output.weight to Q4_0 at setup and use it as draft-only lm_head.
    /// More aggressive bench probe for draft-head bandwidth/cost sensitivity.
    #[arg(long)]
    pub(crate) mtp_draft_lm_head_q4_0: bool,
    /// Quantize output.weight to affine Q4 group-size-64 for the draft lm_head.
    /// MTPLX-isomorphic bench probe; target verify still uses output.weight.
    #[arg(long)]
    pub(crate) mtp_draft_lm_head_q4_affine64: bool,
    /// Recursive MTP hidden fed into the next draft slot.
    #[arg(long, value_enum, default_value_t = MtpRecursiveHiddenArg::PostNorm)]
    pub(crate) mtp_recursive_hidden: MtpRecursiveHiddenArg,
    /// Base-model hidden variant fed into MTP prompt/bridge draft slots.
    #[arg(long, value_enum, default_value_t = MtpBaseHiddenArg::PostNorm)]
    pub(crate) mtp_base_hidden: MtpBaseHiddenArg,
    /// MTP KV history policy for packed native-MTP decode.
    #[arg(long, value_enum, default_value_t = MtpHistoryArg::Committed)]
    pub(crate) mtp_history: MtpHistoryArg,
    /// Write MTP target-rank rows as JSONL. This forces full draft-logit
    /// readback and is diagnostic-only, not a timing path.
    #[arg(long)]
    pub(crate) mtp_rank_topk: Option<PathBuf>,
    /// Per-packet candidate-vs-shadow-serial state trace (oracle probe
    /// only). Prints per-packet GDN state/conv max-abs deltas against an
    /// in-process serial shadow session. Diagnostic-only; wrecks timing.
    #[arg(long)]
    pub(crate) mtp_state_trace: bool,
    /// Write a compact JSON summary for MTPLX/profile-parity sweeps.
    #[arg(long)]
    pub(crate) output: Option<PathBuf>,
    /// Include exact prompt and product-target token IDs in `--output`.
    #[arg(long, requires = "output")]
    pub(crate) include_token_ids: bool,
    /// Number of tokens to generate after the prompt.
    #[arg(long, default_value = "64")]
    pub(crate) tokens: usize,
    /// Stop tokens for generation, comma-separated (e.g.
    /// `--stop-tokens 248046,248044`). When omitted, the stop set is
    /// resolved from the GGUF's declared `tokenizer.ggml.eos_token_id`
    /// (and `eot_token_id` if present) at runtime. There is no
    /// hardcoded fallback — a GGUF that declares no stops is an error.
    #[arg(long, value_delimiter = ',')]
    pub(crate) stop_tokens: Option<Vec<i32>>,
    /// Skip the warmup pass.
    #[arg(long)]
    pub(crate) no_warmup: bool,
}

#[derive(Parser, Debug)]
pub(crate) struct PldArgs {
    /// Path to a target GGUF file.
    #[arg(short = 'm', long)]
    pub(crate) model: PathBuf,
    /// Prompt text.
    #[arg(short = 'p', long)]
    pub(crate) prompt: String,
    /// Render the prompt through a Qwen chat template.
    #[arg(long)]
    pub(crate) qwen_chat: bool,
    /// Optional system prompt for `--qwen-chat` rendering.
    #[arg(long)]
    pub(crate) system: Option<String>,
    /// Render an empty thinking block for `--qwen-chat`.
    #[arg(long)]
    pub(crate) disable_thinking: bool,
    /// Number of tokens to generate after the prompt.
    #[arg(long, default_value = "128")]
    pub(crate) tokens: usize,
    /// Stop tokens for generation, comma-separated.
    #[arg(long, value_delimiter = ',')]
    pub(crate) stop_tokens: Option<Vec<i32>>,
    /// Skip the warmup pass.
    #[arg(long)]
    pub(crate) no_warmup: bool,
    /// Write a compact charged-path JSON summary.
    #[arg(long)]
    pub(crate) output: Option<PathBuf>,
    /// Include semantic per-step events in `--output`.
    #[arg(long, requires = "output")]
    pub(crate) trace_events: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum MtpProbeMode {
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
pub(crate) enum MtpRecursiveHiddenArg {
    /// Current qwen path: MTP residual stream before shared-head norm.
    PreNorm,
    /// MTPLX contract default: MTP shared-head-normalized hidden.
    PostNorm,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum MtpBaseHiddenArg {
    /// Current qwen path: base residual stream before final output norm.
    PreNorm,
    /// MTPLX contract default: base final-output-normalized hidden.
    PostNorm,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum MtpHistoryArg {
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

#[derive(Parser, Debug)]
pub(crate) struct DflashLazyArgs {
    /// Path to the target GGUF (e.g. Qwen3.6-27B-Q4_K_M.gguf).
    #[arg(short = 'm', long)]
    pub(crate) model: PathBuf,
    /// Path to the DFlash drafter GGUF (e.g.
    /// spiritbuun/Qwen3.6-27B-DFlash-GGUF / dflash-draft-3.6-q8_0.gguf).
    #[arg(long)]
    pub(crate) drafter: PathBuf,
    /// Prompt text. Use a meaningful prompt for honest acceptance rates.
    #[arg(
        short = 'p',
        long,
        default_value = "The quick brown fox jumps over the lazy dog"
    )]
    pub(crate) prompt: String,
    /// Number of tokens to generate after the prompt.
    #[arg(long, default_value = "32")]
    pub(crate) tokens: usize,
    /// Stop tokens for generation, comma-separated. When omitted, the
    /// stop set is resolved from the GGUF's declared
    /// `tokenizer.ggml.eos_token_id` (and `eot_token_id` if present).
    #[arg(long, value_delimiter = ',')]
    pub(crate) stop_tokens: Option<Vec<i32>>,
    /// Effective-N: only consider the first M draft positions per outer
    /// step (1 ≤ M ≤ block_size - 1). Reveals where α decays in the
    /// block; if α at M=8 is close to α at M=15, larger N is just paying
    /// for verify cost without recovering tokens. M=0 means use the
    /// full block_size - 1 from the GGUF.
    #[arg(long, default_value = "0")]
    pub(crate) effective_n: usize,
    /// Skip the warmup pass.
    #[arg(long)]
    pub(crate) no_warmup: bool,
    /// **T0 (Program T)**: record, for every prefix-conditioned draft
    /// position (accepted path + first mismatch), the RANK of target's
    /// argmax in the drafter's logits at that position. Emits a JSONL
    /// artifact for the comb-tree acceptance optimizer plus a p_k(depth)
    /// table (k in 1/2/4/8/16). Uses draft_block_with_logits (slower;
    /// measurement-only).
    #[arg(long)]
    pub(crate) rank_topk: Option<PathBuf>,
    /// **T0b tree-sim**: simulate ONE-block tree decode exactly (static
    /// topology: chain depth D plus rank<=B sibling sets at the first R
    /// depths; DFlash's block drafter makes deeper rows path-independent,
    /// so rescued paths keep verifying against the SAME block). Emitted
    /// stream remains target-greedy by construction. Requires --rank-topk.
    #[arg(long)]
    pub(crate) tree_sim: bool,
    /// Tree-sim chain depth D (node budget = D + R*(B-1) must be <= 15).
    #[arg(long, default_value = "6")]
    pub(crate) tree_chain_d: usize,
    /// Tree-sim sibling-set count R (rescues allowed at depths 0..R).
    #[arg(long, default_value = "3")]
    pub(crate) tree_sibling_depths: usize,
    /// Tree-sim sibling branching B (rescue taken when rank <= B).
    #[arg(long, default_value = "4")]
    pub(crate) tree_b: usize,
}

#[derive(Parser, Debug)]
pub(crate) struct DflashArgs {
    /// Path to the target GGUF (e.g. Qwen3.6-27B-Q4_K_M.gguf).
    #[arg(short = 'm', long)]
    pub(crate) model: PathBuf,
    /// Path to the DFlash drafter GGUF.
    #[arg(long)]
    pub(crate) drafter: PathBuf,
    /// Prompt text. Use a meaningful prompt for honest acceptance rates.
    #[arg(
        short = 'p',
        long,
        default_value = "The quick brown fox jumps over the lazy dog"
    )]
    pub(crate) prompt: String,
    /// Number of tokens to generate after the prompt.
    #[arg(long, default_value = "64")]
    pub(crate) tokens: usize,
    /// Stop tokens for generation, comma-separated. When omitted, the
    /// stop set is resolved from the GGUF's declared
    /// `tokenizer.ggml.eos_token_id` (and `eot_token_id` if present).
    #[arg(long, value_delimiter = ',')]
    pub(crate) stop_tokens: Option<Vec<i32>>,
    /// Skip the warmup pass.
    #[arg(long)]
    pub(crate) no_warmup: bool,
    /// Skip the equivalence check vs DFlash=off baseline (saves ~1×
    /// gen-time on the same prompt). Default: ON, because the bench
    /// is also a correctness gate.
    #[arg(long)]
    pub(crate) skip_equivalence_check: bool,
    /// **v0.72.3**: enable lightweight per-phase GPU timers in
    /// draft_block (phase1_ctx_fc_norm, phase2_embed,
    /// phase2_proj_norm_rope ×n_layer, phase3_attn_oproj_ffn_residuals
    /// ×n_layer, phase4_tail). Aggregated across all outer steps and
    /// reported at end. Used to confirm v0.72.4+ leverage map.
    #[arg(long)]
    pub(crate) profile: bool,
    /// **v0.76**: verify-chain length policy. One of:
    /// `adaptive` (default; ctx-keyed schedule with Off-terminal),
    /// `static-16` / `static-8` / `static-4` (fixed N, no Off ramp),
    /// `off` (no speculation; single_token decode loop), `cycle`
    /// (**v0.77** verify microbench: interleaves Spec(8/4/2/1) with Off
    /// reference steps and reports per-n_eff packed_verify wall stats).
    /// The static modes exist for the calibration sweep + as A/B
    /// comparators against `adaptive`. `static-16` matches pre-v0.76
    /// behavior.
    /// Current `adaptive` tuning is calibrated on M4 Max + 27B Q4_K_M
    /// code-prompt sweeps; treat it as a heuristic outside that regime.
    #[arg(long, default_value = "adaptive")]
    pub(crate) n_policy: String,
}

#[derive(Parser, Debug)]
pub(crate) struct VocabAuditArgs {
    /// Path to a GGUF file.
    #[arg(short = 'm', long)]
    pub(crate) model: PathBuf,
    /// Optional path to a prompt corpus (one prompt per line; empty
    /// lines and lines starting with '#' ignored). If absent, uses a
    /// built-in mixed-category corpus.
    #[arg(long)]
    pub(crate) prompts: Option<PathBuf>,
    /// Number of greedy-decode tokens per prompt.
    #[arg(long, default_value = "32")]
    pub(crate) tokens: usize,
    /// Comma-separated K thresholds to evaluate (vocab-prune sizes).
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "1024,4096,8192,16384,32768,49152,65536,98304"
    )]
    pub(crate) ks: Vec<usize>,
    /// Show this many "out of K" decoded tokens per K (for inspection).
    #[arg(long, default_value = "5")]
    pub(crate) show_examples: usize,
}

#[derive(Parser, Debug)]
pub(crate) struct TokArgs {
    /// Path to a Qwen 3.5 / 3.6 GGUF file.
    #[arg(short = 'm', long)]
    pub(crate) model: PathBuf,
    /// Prompt text. If absent, uses a fixed mixed tokenizer stress prompt.
    #[arg(short = 'p', long, conflicts_with_all = ["file", "messages"])]
    pub(crate) prompt: Option<String>,
    /// Read prompt text from a file.
    #[arg(long, conflicts_with = "messages")]
    pub(crate) file: Option<PathBuf>,
    /// Render a JSON messages input into a Qwen chat-template prompt.
    ///
    /// Accepted shapes:
    /// - bare `[{ role, content }, ...]`
    /// - wrapped `{ messages: [...], ... }`
    #[arg(long)]
    pub(crate) messages: Option<PathBuf>,
    /// Use only the first N messages from `--messages` before rendering.
    #[arg(long)]
    pub(crate) messages_max: Option<usize>,
    /// Preserve assistant `<think>...</think>` history from `--messages`.
    /// By default the bench preserves thinking only for wrapped Qwen3.6
    /// rollouts and strips it otherwise.
    #[arg(long)]
    pub(crate) messages_preserve_thinking: bool,
    /// Force stripping assistant `<think>...</think>` history from
    /// `--messages`, even if auto-detection would preserve it.
    #[arg(long, conflicts_with = "messages_preserve_thinking")]
    pub(crate) messages_strip_thinking: bool,
    /// Do not append a final `<|im_start|>assistant\n` generation marker for
    /// `--messages` prompts.
    #[arg(long)]
    pub(crate) messages_no_generation_prompt: bool,
    /// Timed encode/decode iterations for each backend.
    #[arg(long, default_value = "1000")]
    pub(crate) iters: usize,
    /// Pass add_special=true to both tokenizer backends.
    #[arg(long)]
    pub(crate) add_special: bool,
    /// Print exact token IDs and their canonical i32le SHA-256 digest.
    #[arg(long)]
    pub(crate) print_token_ids: bool,
}

#[derive(Parser, Debug)]
pub(crate) struct AttnPrefillMicroArgs {
    /// Path to a GGUF file.
    #[arg(
        short = 'm',
        long,
        default_value = "/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
    )]
    pub(crate) model: PathBuf,
    /// Absolute position of the first packed query row.
    #[arg(long, default_value = "16384")]
    pub(crate) base_pos: usize,
    /// Number of packed query rows.
    #[arg(long, default_value = "128")]
    pub(crate) rows: usize,
    /// Split-K partitions.
    #[arg(long, default_value = "64")]
    pub(crate) nwg: usize,
    /// Query rows processed per packed main-pass threadgroup.
    #[arg(long, default_value = "2")]
    pub(crate) qt: usize,
}

#[derive(Parser, Debug)]
pub(crate) struct AttnFrontMicroArgs {
    /// Path to a GGUF file.
    #[arg(short = 'm', long)]
    pub(crate) model: PathBuf,
    /// Number of packed rows.
    #[arg(long, default_value = "1024")]
    pub(crate) rows: usize,
}

#[derive(Parser, Debug)]
pub(crate) struct AttnLayerMicroArgs {
    /// Path to a GGUF file.
    #[arg(short = 'm', long)]
    pub(crate) model: PathBuf,
    /// Absolute position of the first packed query row.
    #[arg(long, default_value = "16384")]
    pub(crate) base_pos: usize,
    /// Number of packed query rows.
    #[arg(long, default_value = "4")]
    pub(crate) rows: usize,
    /// Forced split-K partitions for both baseline and packed body.
    #[arg(long, default_value = "64")]
    pub(crate) nwg: usize,
    /// Query rows processed per packed main-pass threadgroup.
    #[arg(long, default_value = "2")]
    pub(crate) qt: usize,
}

#[derive(Parser, Debug)]
pub(crate) struct PpFfnAbArgs {
    /// Path to a GGUF file.
    #[arg(short = 'm', long)]
    pub(crate) model: PathBuf,
    /// Synthetic prompt token count.
    #[arg(short = 'p', long, default_value = "4096")]
    pub(crate) n_prompt: usize,
    /// Packed prefill chunk size. If omitted, uses the model-aware default.
    #[arg(long)]
    pub(crate) prefill_chunk: Option<usize>,
    /// Number of base/fused pairs. Odd pairs run base->fused; even pairs reverse.
    #[arg(long, default_value = "2")]
    pub(crate) pairs: usize,
    /// Skip the unmeasured base and fused warmup passes.
    #[arg(long)]
    pub(crate) no_warmup: bool,
    /// Deterministic seed for synthetic token generation.
    #[arg(long, default_value = "1")]
    pub(crate) seed: u64,
}

#[derive(Parser, Debug)]
pub(crate) struct PpWaitArgs {
    /// Path to a GGUF file.
    #[arg(short = 'm', long)]
    pub(crate) model: PathBuf,
    /// Synthetic prompt token count.
    #[arg(short = 'p', long, default_value = "320")]
    pub(crate) n_prompt: usize,
    /// Packed prefill chunk size. If omitted, uses the model-aware default.
    #[arg(long)]
    pub(crate) prefill_chunk: Option<usize>,
    /// Include final norm + lm_head + logits readback.
    #[arg(long)]
    pub(crate) with_tail: bool,
    /// Deterministic seed for synthetic token generation.
    #[arg(long, default_value = "1")]
    pub(crate) seed: u64,
    /// Skip the warmup prefill pass before signaling ready.
    #[arg(long)]
    pub(crate) no_warmup: bool,
    /// File written once the model is loaded and warmup is complete.
    #[arg(long)]
    pub(crate) ready_file: PathBuf,
    /// File whose appearance triggers the timed run.
    #[arg(long)]
    pub(crate) go_file: PathBuf,
    /// Output format for the final timed run.
    #[arg(short = 'o', long, default_value = "json")]
    pub(crate) output: OutputFormat,
}
