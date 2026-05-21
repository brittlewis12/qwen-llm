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

use anyhow::{Context, Result, anyhow};
use clap::{Parser, Subcommand, ValueEnum};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSError, NSString};
use objc2_metal::{
    MTLAllocation, MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLDevice, MTLResidencySet,
    MTLResidencySetDescriptor,
};
use qwen_llm::{
    gguf::GgufFile,
    loader::{Model, open_dflash_drafter},
    metal::{
        KernelEncoder, MetalContext, MetalTensor, encode_attn_decode_v4_f32,
        encode_attn_prefill_v4_g8_t2_q2_c64_f32, encode_attn_prefill_v4_g8_t2_q4_c64_f32,
        encode_attn_prefill_v4_g16_t4_q2_c64_f32, encode_attn_prefill_v4_g16_t4_q4_c64_f32,
        encode_scatter_offset_f32_to_f16, encode_touch_bytes_f32, with_attn_v4_group_tile_override,
    },
    metal_dflash::{
        DFlashDecoder, MetalDFlashHead, MetalDFlashLayerMajorScratch, MetalDFlashSession,
        MetalDFlashVerifyScratch, prefill_tokens_prompt_only_profiled,
        prefill_tokens_with_multi_hidden, prefill_tokens_with_multi_hidden_profiled,
    },
    metal_forward::encode_mat_mat_dispatch,
    metal_forward::{MetalBlock, MetalForward, MetalModel, MetalSession},
    metal_mtp::{MetalMtpHead, MetalMtpSession, SpeculativeDecoder},
    prefix_cache::PrefixCache,
    tensor::GgmlType,
    tokenizer::{LlamaCppTokenizer, NativeTokenizer, Tokenizer},
};
use std::collections::HashSet;
use std::path::PathBuf;
use std::time::{Duration, Instant};

fn env_flag_enabled(name: &str) -> bool {
    matches!(
        std::env::var(name).as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
    )
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
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
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
    /// Sweep context length (ramp + measure window).
    CtxSweep(CtxSweepArgs),
    /// Phase-resolved profile at one context length (uses the
    /// `phase_sum` GPU time, NOT the per-phase-cmdbuf wall artifact).
    Phase(PhaseArgs),
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
    /// Bench-only GDN front-projection overlap path. Currently MoE-only in tg.
    #[arg(long)]
    concurrent_gdn_proj: bool,
    /// Deterministic seed for random token selection.
    #[arg(long, default_value = "1")]
    seed: u64,
    /// `text` or `json` (`llama-bench -o json` shape).
    #[arg(short = 'o', long, value_enum, default_value = "text")]
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
struct DecodeWindowArgs {
    /// Path to a GGUF file.
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Context length to ramp to before waiting.
    #[arg(long)]
    target_ctx: usize,
    /// Number of decode tokens to execute after the go signal.
    #[arg(long, default_value = "128")]
    window: usize,
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
    /// path. `2` and `3` use a bench-only MTP-N prototype that chains MTP
    /// drafts recursively and verifies them with the packed base path.
    #[arg(long, default_value = "1")]
    spec_tokens: usize,
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
const BENCH_SCHEMA_VERSION: u32 = 1;

/// One bench result row. Field names match `llama-bench`'s JSON schema where
/// the meaning is the same; engine-specific fields are `Option<T>` and
/// serialized as explicit `null` (NOT omitted) so downstream consumers can
/// rely on a stable field set.
#[derive(Debug, Clone, serde::Serialize)]
struct BenchRow {
    schema_version: u32,
    engine: &'static str,
    build_commit: &'static str,
    /// `1` if probed dirty at runtime via tracked-only git status.
    build_dirty: u8,
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
    avg_gpu_ns: Option<u64>,
    /// Effective decode bandwidth (GB/s). `None` for `pp<N>` rows and for
    /// MoE `tg<N>` rows — MoE active-param accounting is out of scope here,
    /// and the naive `model_size × t/s` overstates by ~10x for MoE.
    decode_gb_per_s: Option<f64>,
    prefill_chunk: Option<usize>,
    decode_mode: Option<&'static str>,
    prefill_mode: Option<&'static str>,
    qwen_env: std::collections::BTreeMap<String, String>,
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

fn default_prefill_chunk(kind: qwen_llm::model::ArchKind, prompt_len: usize) -> usize {
    match kind {
        qwen_llm::model::ArchKind::Moe => prompt_len.clamp(1, 1024),
        qwen_llm::model::ArchKind::Dense => prompt_len.clamp(1, 1024),
    }
}

/// Returns `(commit, dirty)` for stamping into JSON output.
///
/// Commit comes from `build.rs` (env at compile time → git → "unknown").
///
/// Dirty is probed at *runtime* via tracked-only
/// `git status --porcelain --untracked-files=no` because cargo's
/// `rerun-if-changed` directives only watch `.git/HEAD` and `.git/index`:
/// editing a tracked file without staging it does NOT invalidate the cached
/// build, so a stale `QWEN_BUILD_DIRTY=0` from the last clean compile would
/// otherwise lie about a dirty worktree. Untracked artifacts are ignored so the
/// benchmark harness does not poison its own provenance by writing output under
/// the repo. We fall back to the compile-time value when the runtime probe
/// fails (no git binary, not in a repo).
fn qwen_build_identity() -> (&'static str, u8) {
    let commit = env!("QWEN_BUILD_COMMIT");
    let baked_dirty = env!("QWEN_BUILD_DIRTY").parse::<u8>().unwrap_or(0);
    let runtime_dirty = std::process::Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=no"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()
        .map(|o| if o.stdout.is_empty() { 0u8 } else { 1u8 });
    (commit, runtime_dirty.unwrap_or(baked_dirty))
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

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();
    match args.cmd {
        Cmd::PrefixCache(a) => run_prefix_cache(a),
        Cmd::VocabAudit(a) => run_vocab_audit(a),
        Cmd::Decode(a) => run_decode(a),
        Cmd::Pp(a) => run_pp(a),
        Cmd::Tg(a) => run_tg(a),
        Cmd::CtxSweep(a) => run_ctx_sweep(a),
        Cmd::Phase(a) => run_phase(a),
        Cmd::DecodeWindow(a) => run_decode_window(a),
        Cmd::Mtp(a) => run_mtp(a),
        Cmd::DflashLazy(a) => run_dflash_lazy(a),
        Cmd::Dflash(a) => run_dflash(a),
        Cmd::Tok(a) => run_tok(a),
        Cmd::AttnPrefillMicro(a) => run_attn_prefill_micro(a),
        Cmd::AttnFrontMicro(a) => run_attn_front_micro(a),
    }
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
}

fn run_mtp(args: MtpArgs) -> Result<()> {
    let MtpArgs {
        model,
        prompt,
        qwen_chat,
        system,
        disable_thinking,
        spec_tokens,
        tokens,
        stop_tokens,
        no_warmup,
    } = args;

    let ctx = MetalContext::new().context("init MetalContext")?;
    eprintln!("[mtp-bench] device: {}", ctx.describe());
    if spec_tokens == 0 || spec_tokens > 3 {
        anyhow::bail!("`--spec-tokens` must be in 1..=3 for now");
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
        "[mtp-bench] model={} prompt={:?} rendered_mode={} thinking={} spec_tokens={} ({} tokens) gen={} stop_tokens={:?}",
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
        prompt_ids.len(),
        tokens,
        stops,
    );

    let mf = MetalForward::new(&ctx, &mm);
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
        if stops.contains(&next_tok) {
            break;
        }
        pos += 1;
        let logits = mf.single_token(next_tok, pos, &mut ref_session)?;
        next_tok = argmax_i32(&logits);
    }
    let ref_decode_ms = t_ref_decode.elapsed().as_secs_f64() * 1e3;
    let ref_total_ms = t_ref_total.elapsed().as_secs_f64() * 1e3;
    let ref_decode_tps = ref_emitted as f64 / (ref_decode_ms / 1000.0);

    // ----- MTP=on: speculative decode -----
    let mtp_session =
        MetalMtpSession::fresh(&ctx, &mtp_head, &m.arch, cap).context("MTP session")?;
    let mut spec_session = MetalSession::fresh(&ctx, &mm, cap).context("spec session")?;
    let mut spec = SpeculativeDecoder::new(&mf, &mtp_head, mtp_session);
    let result = if spec_tokens == 1 {
        spec.decode(&prompt_ids, tokens, &stops, &mut spec_session)
            .context("spec decode")?
    } else {
        let mut verify_scratch =
            MetalDFlashVerifyScratch::fresh(&ctx, &mm, (spec_tokens + 1) as u32, 1)
                .context("mtp packed verify scratch")?;
        let mut layer_scratch =
            MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, (spec_tokens + 1) as u32)
                .context("mtp packed layer scratch")?;
        spec.decode_packed_n(
            &prompt_ids,
            tokens,
            &stops,
            &mut spec_session,
            spec_tokens,
            &mut verify_scratch,
            &mut layer_scratch,
        )
        .context("spec decode packed-n")?
    };
    let spec_emitted = result.tokens.len() - prompt_ids.len();
    let spec_total_ms = result.stats.wall_ms;
    let spec_decode_tps = spec_emitted as f64 / (spec_total_ms / 1000.0);

    // ----- Compare -----
    let ref_generated = &ref_tokens[prompt_ids.len()..];
    let spec_generated = &result.tokens[prompt_ids.len()..];
    let identical = ref_generated == spec_generated;

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
        "[mtp-bench] MTP=on : {spec_emitted} tokens, {spec_total_ms:.1} ms total \
         (prefill + decode lumped — spec_decode internally streams MTP-KV \
         prefill alongside base prefill)"
    );
    eprintln!("[mtp-bench]   t/s: total {spec_total_tps:.1}");
    eprintln!(
        "[mtp-bench]   α (acceptance rate) = {:.3}   steps={}  accepted={}",
        result.stats.acceptance_rate(),
        result.stats.steps,
        result.stats.accepted,
    );
    eprintln!(
        "[mtp-bench]   base_calls={}  mtp_calls={} \
         (= prompt prefill + step-B drafts + step-E bridges)",
        result.stats.base_forward_calls, result.stats.mtp_calls,
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
    } = args;

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
    let k_layers = head.target_layer_ids.len();
    let n_target_features = k_layers * h_target;

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
        let argmaxes = decoder
            .draft_block(carry_tok, drafter_pos)
            .context("drafter draft_block")?;
        drafter_calls += 1;
        // Draft tokens come from positions 1..N.
        let drafts: Vec<i32> = argmaxes[1..].iter().take(m).copied().collect();

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

        // Now check each draft sequentially.
        let mut n_accepted_this_step = 0usize;
        steps += 1;
        for j in 0..m {
            attempts_at_pos[j] += 1;
            if drafts[j] != target_next {
                break;
            }
            // Accepted!
            accepts_at_pos[j] += 1;
            accepted_total += 1;
            n_accepted_this_step += 1;
            emitted.push(drafts[j]);
            if emitted.len() >= tokens || stops.contains(&drafts[j]) {
                // Note: we don't `return` here because we still want to
                // emit() through the outer loop. The outer-loop
                // `if emitted.len() >= tokens` check at the top of the
                // next iter handles the exit, so carry_tok doesn't
                // need to be touched here.
                break;
            }
            // Process drafts[j] via target to set up next verify step.
            let logits = mf
                .single_token_with_multi_hidden(
                    drafts[j],
                    processed_pos + 1,
                    &mut target_session,
                    &head.target_layer_ids,
                    &multi_hidden_dst,
                )
                .context("verify base step (draft)")?;
            base_calls += 1;
            decoder
                .session
                .append_target_ctx_column_now(
                    &ctx,
                    &multi_hidden_dst,
                    processed_pos + 1,
                    n_target_features,
                )
                .context("append draft ctx column")?;
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
    let mut layer_scratch = MetalDFlashLayerMajorScratch::fresh_prefill(&ctx, &mm, cfg.block_size)
        .context("layer scratch")?;

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
        let argmaxes = decoder
            .draft_block(carry_tok, drafter_pos)
            .context("drafter draft_block")?;
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
        decoder
            .session
            .append_target_ctx_columns_now(&ctx, &columns_refs, n_target_features)
            .context("append packed ctx columns")?;

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
            qwen_llm::metal_dflash::encode_restore_after_partial_accept_inner(
                decoder.base,
                &verify_scratch,
                n_keep,
                drafter_pos,
                &mut target_session,
                Some(n_eff as u32), // adaptive-N: same n_eff as the verify call
            )
            .context("restore_after_partial_accept")?;
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
            if stops.contains(&next_tok) {
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
            let total_gpu_ms: f64 = agg.values().map(|(s, _)| *s).sum();
            // Sort by descending sum.
            let mut sorted: Vec<_> = agg.iter().collect();
            sorted.sort_by(|a, b| b.1.0.partial_cmp(&a.1.0).unwrap());
            for (name, (sum_ms, count)) in &sorted {
                let avg = *sum_ms / (*count as f64);
                let pct = 100.0 * *sum_ms / total_gpu_ms;
                eprintln!(
                    "[dflash]   {name:>40}  sum={sum_ms:>8.2} ms  ({pct:>5.1}%)  \
                     n={count:>4}  avg={avg:>6.2} ms"
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

    let ctx = MetalContext::new().context("init MetalContext")?;
    text_log!("[pp] device: {}", ctx.describe());

    let g = GgufFile::open(&model).with_context(|| format!("open {}", model.display()))?;
    let m = Model::from_gguf(&g).context("parse model arch from gguf")?;
    let mm = MetalModel::load(&ctx, &g, &m).context("metal-load model weights")?;

    let (ids, source_label) = if let Some(prompt) = prompt {
        let tok = Tokenizer::from_gguf(&g).context("open tokenizer")?;
        let ids = tok.encode(&prompt, false).context("tokenize prompt")?;
        (ids, format!("text prompt ({} chars)", prompt.len()))
    } else if let Some(path) = file {
        let prompt =
            std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
        let tok = Tokenizer::from_gguf(&g).context("open tokenizer")?;
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
        let tok = Tokenizer::from_gguf(&g).context("open tokenizer")?;
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
            synthetic_prompt_ids(n_prompt, m.arch.vocab_size, seed),
            format!("synthetic token ids (seed={seed})"),
        )
    };
    if ids.is_empty() {
        return Err(anyhow!("prompt tokenized to an empty sequence"));
    }

    let prefill_chunk =
        prefill_chunk.unwrap_or_else(|| default_prefill_chunk(m.arch.kind, ids.len()));
    if prefill_chunk == 0 {
        return Err(anyhow!("--prefill-chunk must be >= 1"));
    }

    let mf = MetalForward::new(&ctx, &mm);
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
        print_prefill_lowering_summary(&mm);
    }

    let residency_guard = if env_flag_enabled("QWEN_PP_RESIDENCY_SET")
        && m.arch.kind == qwen_llm::model::ArchKind::Moe
    {
        let (guard, allocations, bytes) =
            pp_register_moe_residency_set(&ctx, &mf).context("register MoE residency set")?;
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
        && m.arch.kind == qwen_llm::model::ArchKind::Moe
    {
        let touched =
            pp_warm_moe_weight_banks(&ctx, &mf).context("warm grouped MoE weight banks")?;
        text_log!("[pp] warmup: touched {touched} MoE expert-bank tensors via GPU residency pass");
    }

    if !no_warmup {
        let mut s = MetalSession::fresh(&ctx, &mm, cap).context("session warmup")?;
        let mut scratch =
            MetalDFlashLayerMajorScratch::fresh_prefill(&ctx, &mm, prefill_chunk as u32)
                .context("warmup prefill scratch")?;
        if with_tail {
            let _ = prefill_tokens_with_multi_hidden(&mf, &ids, 0, &mut s, &mut scratch, &[], None)
                .context("warmup prefill with tail")?;
        } else {
            let _ = prefill_tokens_prompt_only_profiled(&mf, &ids, 0, &mut s, &mut scratch)
                .context("warmup prompt-only prefill")?;
        }
    }

    let mut wall_samples = Vec::with_capacity(runs);
    let mut gpu_samples = Vec::with_capacity(runs);
    let mut ts_samples = Vec::with_capacity(runs);
    for run_idx in 0..runs {
        let mut s = MetalSession::fresh(&ctx, &mm, cap).context("session run")?;
        let mut scratch =
            MetalDFlashLayerMajorScratch::fresh_prefill(&ctx, &mm, prefill_chunk as u32)
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
            n_repetitions: runs,
            avg_ts: ts_mean,
            stddev_ts: ts_sd,
            samples_ts: ts_samples.clone(),
            samples_ns: wall_samples.iter().map(|w| (*w * 1e6) as u64).collect(),
            avg_ns: (wall_mean * 1e6) as u64,
            avg_gpu_ns: Some((gpu_mean * 1e6) as u64),
            // pp is not a steady-state-bandwidth measurement, so we don't
            // emit a derived GB/s for prefill rows. Digest tools can compute
            // their own if they want, but the canonical bandwidth comparison
            // is on decode.
            decode_gb_per_s: None,
            prefill_chunk: Some(prefill_chunk),
            decode_mode: None,
            prefill_mode: Some("packed"),
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

    let ctx = MetalContext::new().context("init MetalContext")?;
    text_log!("[tg] device: {}", ctx.describe());

    let g = GgufFile::open(&model).with_context(|| format!("open {}", model.display()))?;
    let m = Model::from_gguf(&g).context("parse model arch from gguf")?;
    let mm = MetalModel::load(&ctx, &g, &m).context("metal-load model weights")?;
    let mf = MetalForward::new(&ctx, &mm);

    let vocab = m.arch.vocab_size.max(1);
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
            MetalTensor::zeros_f32(&ctx, vec![1]).context("tg pipelined ids ping0")?,
            MetalTensor::zeros_f32(&ctx, vec![1]).context("tg pipelined ids ping1")?,
        ])
    } else {
        None
    };
    let argmax_ping = if pipelined {
        Some([
            MetalTensor::zeros_f32(&ctx, vec![1]).context("tg pipelined argmax ping0")?,
            MetalTensor::zeros_f32(&ctx, vec![1]).context("tg pipelined argmax ping1")?,
        ])
    } else {
        None
    };
    let run_once = |first_tok: i32, ranges: &mut dyn FnMut() -> i32| -> Result<(f64, f64)> {
        let mut s = MetalSession::fresh(&ctx, &mm, cap).context("tg session")?;
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
                    if mm.arch.kind != qwen_llm::model::ArchKind::Moe {
                        return Err(anyhow!(
                            "--concurrent-gdn-proj tg mode is currently MoE-only"
                        ));
                    }
                    mf.single_token_argmax_profiled_concurrent_gdn_moe(tok, pos as u32, &mut s)?
                } else {
                    mf.single_token_argmax_profiled(tok, pos as u32, &mut s)?
                };
                gpu_ms_acc += prof.gpu_kernel_ms;
                tok = ranges();
            }
            let wall_ms = t0.elapsed().as_secs_f64() * 1e3;
            return Ok((wall_ms, gpu_ms_acc));
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
        mf.encode_single_token_argmax(&first_enc, 0, &mut s, &ids_ping[0], &argmax_ping[0])?;
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
                &mut s,
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
        Ok((wall_ms, gpu_ms_acc))
    };

    if !no_warmup {
        let first = next_rand_tok();
        let _ = run_once(first, &mut next_rand_tok).context("tg warmup")?;
    }

    let mut wall_samples: Vec<f64> = Vec::with_capacity(runs);
    let mut gpu_samples: Vec<f64> = Vec::with_capacity(runs);
    let mut ts_samples: Vec<f64> = Vec::with_capacity(runs);
    for run_idx in 0..runs {
        let first = next_rand_tok();
        let (wall_ms, gpu_ms) = run_once(first, &mut next_rand_tok).context("tg run")?;
        let ts = n_gen as f64 * 1000.0 / wall_ms;
        wall_samples.push(wall_ms);
        gpu_samples.push(gpu_ms);
        ts_samples.push(ts);
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

    if json_mode {
        let (commit, dirty) = qwen_build_identity();
        let arch_kind_str: &'static str = match m.arch.kind {
            qwen_llm::model::ArchKind::Dense => "dense",
            qwen_llm::model::ArchKind::Moe => "moe",
        };
        let model_size = model_weight_bytes(&g);
        let model_n_params = g
            .get_u64("general.parameter_count")
            .unwrap_or_else(|| g.tensors.iter().map(|t| t.n_elements()).sum());
        // Bandwidth is only meaningful for dense; MoE active-param accounting
        // lives outside this schema today.
        let gb_per_s = if matches!(m.arch.kind, qwen_llm::model::ArchKind::Dense)
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
            avg_gpu_ns: Some((gpu_mean * 1e6) as u64),
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
    eprintln!(
        "[bench] model={} prompt={:?} ({} tokens), gen={} tokens",
        model.display(),
        prompt,
        ids.len(),
        tokens
    );

    let mf = MetalForward::new(&ctx, &mm);
    let cap = ids.len() + tokens + 16;
    let use_packed_prefill = !sequential_prefill;
    let use_gpu_argmax_decode = !full_logits_decode;

    if !no_warmup {
        // One warmup pass to compile pipeline state objects + warm caches.
        let mut s = MetalSession::fresh(&ctx, &mm, cap).context("session warmup")?;
        let warmup_last_logits = if use_packed_prefill {
            let mut scratch =
                MetalDFlashLayerMajorScratch::fresh_prefill(&ctx, &mm, prefill_chunk as u32)
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
            let mut scratch =
                MetalDFlashLayerMajorScratch::fresh_prefill(&ctx, &mm, prefill_chunk as u32)
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
            avg_gpu_ns: prefill_gpu_total_ms.map(|g| (g * 1e6) as u64),
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
                avg_gpu_ns: None,
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
    } = args;
    let ctx = MetalContext::new()?;
    eprintln!("[bench] device: {}", ctx.describe());
    let g = GgufFile::open(&model)?;
    let m = Model::from_gguf(&g)?;
    let mm = MetalModel::load(&ctx, &g, &m)?;

    let max_n = *checkpoints
        .iter()
        .max()
        .ok_or_else(|| anyhow!("no checkpoints"))?;
    let mut s = MetalSession::fresh(&ctx, &mm, max_n + window + 16)?;
    let mf = MetalForward::new(&ctx, &mm);

    // Warmup pipeline state cache.
    for i in 0..3 {
        let _ = mf.single_token(0, i as u32, &mut s)?;
    }
    let mut s = MetalSession::fresh(&ctx, &mm, max_n + window + 16)?;
    // One pre-warmed token at position 0 to populate everything.
    let _ = mf.single_token(0, 0, &mut s)?;

    println!("[ctx-sweep] === per-token decode cost vs context ===");
    println!("[ctx-sweep] context  total_ms  gpu_ms  cpu_enc_ms  t/s");

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
    println!(
        "[phase ctx={target}] phase_sum={phase_sum:.2} ms (production-realistic GPU)  \
         wall_artifact={wall_artifact:.2} ms (DO NOT use as prod ms/token)"
    );
    for (name, ms) in &phases {
        let pct = ms / phase_sum * 100.0;
        println!("[phase ctx={target}]   {name:25} {ms:7.2} ms  ({pct:5.1}%)");
    }
    Ok(())
}

fn run_decode_window(args: DecodeWindowArgs) -> Result<()> {
    let DecodeWindowArgs {
        model,
        target_ctx,
        window,
        ready_file,
        go_file,
        pipelined,
        concurrent_gdn_proj,
        concurrent_attn_proj,
    } = args;
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
    let _ = mf.single_token(0, 0, &mut s)?;
    for p in 1..(target_ctx as u32) {
        let _ = mf.single_token(0, p, &mut s)?;
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
fn run_prefix_cache(args: PrefixCacheArgs) -> Result<()> {
    let PrefixCacheArgs {
        model,
        prefix,
        target_prefix_len,
        suffix,
        tokens,
    } = args;

    let ctx = MetalContext::new()?;
    eprintln!("[prefix-cache] device: {}", ctx.describe());
    let g = GgufFile::open(&model)?;
    let m = Model::from_gguf(&g)?;
    let mm = MetalModel::load(&ctx, &g, &m)?;
    let tok = Tokenizer::from_gguf(&g)?;

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
    eprintln!(
        "[prefix-cache] prefix={} tokens, suffix={} tokens, total={} tokens",
        prefix_ids.len(),
        suffix_ids.len(),
        total_len
    );

    let mf = MetalForward::new(&ctx, &mm);
    let cap = total_len + tokens + 16;

    // Warmup pass to compile pipeline state objects.
    {
        let mut s = MetalSession::fresh(&ctx, &mm, 32)?;
        let _ = mf.single_token(prefix_ids[0], 0, &mut s)?;
    }

    // ---- COLD path: prefill (prefix + suffix), decode N tokens ----
    let cold_t0 = Instant::now();
    let mut sess_cold = MetalSession::fresh(&ctx, &mm, cap)?;
    let mut last_logits = vec![];
    for (i, &tid) in prefix_ids.iter().chain(suffix_ids.iter()).enumerate() {
        last_logits = mf.single_token(tid, i as u32, &mut sess_cold)?;
    }
    let cold_prefill_ms = cold_t0.elapsed().as_secs_f64() * 1e3;

    // First decoded token = TTFT-equivalent measurement.
    let cold_first_decode_t = Instant::now();
    let cold_first_id = argmax_i32(&last_logits);
    let _ = mf.single_token(cold_first_id, total_len as u32, &mut sess_cold)?;
    let cold_first_decode_ms = cold_first_decode_t.elapsed().as_secs_f64() * 1e3;

    let cold_ttft_ms = cold_prefill_ms + cold_first_decode_ms;
    eprintln!(
        "[prefix-cache] COLD: prefill {} tokens in {cold_prefill_ms:.1} ms, first-decode {cold_first_decode_ms:.1} ms, TTFT {cold_ttft_ms:.1} ms",
        total_len
    );

    // ---- WARM path: prefill prefix, snapshot. Then fresh session, restore, ----
    // ---- prefill suffix, decode 1 token. Time the second-request portion. ----
    let mut sess_pre = MetalSession::fresh(&ctx, &mm, cap)?;
    let mut last_pre_logits = vec![];
    for (i, &tid) in prefix_ids.iter().enumerate() {
        last_pre_logits = mf.single_token(tid, i as u32, &mut sess_pre)?;
    }
    let identity = sess_pre.snapshot_identity(0xAA, 0xBB);
    let snap_t = Instant::now();
    let snap = sess_pre.snapshot(identity.clone(), prefix_ids.clone(), Some(last_pre_logits));
    let snap_create_ms = snap_t.elapsed().as_secs_f64() * 1e3;
    let snap_bytes = snap.n_bytes();
    let mut cache = PrefixCache::new();
    cache.insert(snap);
    let full_request: Vec<i32> = prefix_ids
        .iter()
        .chain(suffix_ids.iter())
        .copied()
        .collect();
    let hit = cache
        .lookup_longest(&identity, &full_request)
        .ok_or_else(|| anyhow!("prefix cache lookup missed a freshly inserted prefix"))?;
    eprintln!(
        "[prefix-cache] (snapshot built: {:.1} MB in {snap_create_ms:.1} ms)",
        snap_bytes as f64 / 1e6
    );

    // Now simulate request 2 starting fresh and finding the cached prefix.
    let warm_t0 = Instant::now();
    let mut sess_warm = MetalSession::fresh(&ctx, &mm, cap)?;
    let restore_t = Instant::now();
    sess_warm.restore_from(hit.snapshot)?;
    let restore_ms = restore_t.elapsed().as_secs_f64() * 1e3;

    let mut last_warm_logits = vec![];
    for (k, &tid) in suffix_ids.iter().enumerate() {
        let pos = (prefix_ids.len() + k) as u32;
        last_warm_logits = mf.single_token(tid, pos, &mut sess_warm)?;
    }
    let warm_prefill_ms = warm_t0.elapsed().as_secs_f64() * 1e3;
    let warm_suffix_ms = warm_prefill_ms - restore_ms;

    let warm_first_decode_t = Instant::now();
    let warm_first_id = argmax_i32(&last_warm_logits);
    let _ = mf.single_token(warm_first_id, total_len as u32, &mut sess_warm)?;
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
            let cold_logits = mf.single_token(*cold_extra.last().unwrap(), pos, &mut sess_cold)?;
            let warm_logits = mf.single_token(*warm_extra.last().unwrap(), pos, &mut sess_warm)?;
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
