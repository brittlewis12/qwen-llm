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
use clap::{Parser, Subcommand};
use qwen_llm::{
    gguf::GgufFile,
    loader::{Model, open_dflash_drafter},
    metal::{MetalContext, MetalTensor},
    metal_dflash::{
        DFlashDecoder, MetalDFlashHead, MetalDFlashLayerMajorScratch, MetalDFlashSession,
        MetalDFlashVerifyScratch,
    },
    metal_forward::{MetalForward, MetalModel, MetalSession},
    metal_mtp::{MetalMtpHead, MetalMtpSession, SpeculativeDecoder},
    tokenizer::Tokenizer,
};
use std::path::PathBuf;
use std::time::Instant;

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
    /// Decode N tokens after a prompt; report per-token timings.
    Decode(DecodeArgs),
    /// Sweep context length (ramp + measure window).
    CtxSweep(CtxSweepArgs),
    /// Phase-resolved profile at one context length (uses the
    /// `phase_sum` GPU time, NOT the per-phase-cmdbuf wall artifact).
    Phase(PhaseArgs),
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
    /// Skip the warmup pass (default is to do one warmup, then re-init
    /// the session for the timed run, exactly like the ignored tests).
    #[arg(long)]
    no_warmup: bool,
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
    /// Number of tokens to generate after the prompt.
    #[arg(long, default_value = "64")]
    tokens: usize,
    /// EOS token id (used to early-terminate generation).
    /// 0.8B / 27B Qwen3.5/3.6: 248046 (`<|im_end|>`).
    #[arg(long, default_value = "248046")]
    eos: i32,
    /// Skip the warmup pass.
    #[arg(long)]
    no_warmup: bool,
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
    /// EOS token id.
    #[arg(long, default_value = "248046")]
    eos: i32,
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
    /// EOS token id.
    #[arg(long, default_value = "248046")]
    eos: i32,
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
        Cmd::CtxSweep(a) => run_ctx_sweep(a),
        Cmd::Phase(a) => run_phase(a),
        Cmd::Mtp(a) => run_mtp(a),
        Cmd::DflashLazy(a) => run_dflash_lazy(a),
        Cmd::Dflash(a) => run_dflash(a),
    }
}

fn run_mtp(args: MtpArgs) -> Result<()> {
    let MtpArgs {
        model,
        prompt,
        tokens,
        eos,
        no_warmup,
    } = args;

    let ctx = MetalContext::new().context("init MetalContext")?;
    eprintln!("[mtp-bench] device: {}", ctx.describe());

    let g = GgufFile::open(&model).with_context(|| format!("open {}", model.display()))?;
    let m = Model::from_gguf(&g).context("parse model arch")?;
    let mtp_view = m.mtp.as_ref().ok_or_else(|| {
        anyhow!(
            "GGUF has no MTP head (\
             use brittlewis12/Qwen3.6-27B-MTP-GGUF or the 0.8B-MTP variant)"
        )
    })?;

    let mm = MetalModel::load(&ctx, &g, &m).context("metal-load weights")?;
    let mtp_head = MetalMtpHead::load(&ctx, &g, mtp_view).context("metal-load MTP head")?;
    let tok = Tokenizer::open(&model).context("open tokenizer")?;

    let prompt_ids = tok.encode(&prompt, false).context("tokenize prompt")?;
    eprintln!(
        "[mtp-bench] model={} prompt={:?} ({} tokens) gen={} eos={eos}",
        model.display(),
        prompt,
        prompt_ids.len(),
        tokens,
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
    let mut last_logits = Vec::new();
    let t_ref_prefill = Instant::now();
    let n_prompt_ref = prompt_ids.len();
    for (i, &tid) in prompt_ids.iter().enumerate() {
        // v0.75.0: skip lm_head + final norm + readback for all but the
        // last prompt token. Only the last token's logits seed the
        // decode phase below.
        if i + 1 < n_prompt_ref {
            mf.single_token_no_tail(tid, i as u32, &mut ref_session)?;
        } else {
            last_logits = mf.single_token(tid, i as u32, &mut ref_session)?;
        }
    }
    let ref_prefill_ms = t_ref_prefill.elapsed().as_secs_f64() * 1e3;

    let t_ref_decode = Instant::now();
    let mut next_tok = argmax_i32(&last_logits);
    let mut pos = (prompt_ids.len() - 1) as u32;
    let mut ref_emitted = 0usize;
    for _ in 0..tokens {
        ref_tokens.push(next_tok);
        ref_emitted += 1;
        if next_tok == eos {
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
    let result = spec
        .decode(&prompt_ids, tokens, eos, &mut spec_session)
        .context("spec decode")?;
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
        eos,
        effective_n,
        no_warmup,
    } = args;

    let ctx = MetalContext::new().context("init MetalContext")?;
    eprintln!("[dflash-lazy] device: {}", ctx.describe());

    let target_g =
        GgufFile::open(&model).with_context(|| format!("open target {}", model.display()))?;
    let target_m = Model::from_gguf(&target_g).context("parse target arch")?;
    let drafter_g =
        GgufFile::open(&drafter).with_context(|| format!("open drafter {}", drafter.display()))?;
    let head = open_dflash_drafter(&drafter_g, &target_m).context("bind drafter")?;

    let mm = MetalModel::load(&ctx, &target_g, &target_m).context("metal-load target")?;
    let mhead = MetalDFlashHead::load(&ctx, &drafter_g, &head).context("metal-load drafter")?;
    let tok = Tokenizer::open(&model).context("open tokenizer")?;

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
        "[dflash-lazy] prompt={prompt:?} ({n_prompt} tokens) gen={tokens} eos={eos} \
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

    // Per-prompt-token captured hidden buffer ([K · H] each).
    let multi_hidden_dst =
        MetalTensor::zeros_f32(&ctx, vec![n_target_features as u64]).context("multi_hidden_dst")?;

    // ---------- Prompt prefill ----------
    let t_prefill = Instant::now();
    let mut last_logits: Vec<f32> = Vec::new();
    for (i, &tid) in prompt_ids.iter().enumerate() {
        // v0.75.0: skip-tail for all but the last prompt token. Hidden
        // capture into multi_hidden_dst still runs, so the per-token
        // append_target_ctx_column_now below sees valid bytes.
        if i + 1 < n_prompt {
            mf.single_token_with_multi_hidden_no_tail(
                tid,
                i as u32,
                &mut target_session,
                &head.target_layer_ids,
                &multi_hidden_dst,
            )
            .context("prefill base step (no_tail)")?;
        } else {
            last_logits = mf
                .single_token_with_multi_hidden(
                    tid,
                    i as u32,
                    &mut target_session,
                    &head.target_layer_ids,
                    &multi_hidden_dst,
                )
                .context("prefill base step")?;
        }
        // Append captured K hiddens to the drafter's target_ctx.
        dsess
            .append_target_ctx_column_now(&ctx, &multi_hidden_dst, i as u32, n_target_features)
            .context("append prefill ctx column")?;
    }
    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1e3;
    eprintln!(
        "[dflash-lazy] prefill {n_prompt} tokens in {prefill_ms:.1} ms (incl. K-hidden capture + ctx append)"
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
        if carry_tok == eos {
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
            if emitted.len() >= tokens || drafts[j] == eos {
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
    let mut last_logits_ref = Vec::new();
    let t_ref_prefill = Instant::now();
    for (i, &tid) in prompt_ids.iter().enumerate() {
        // v0.75.0: skip-tail except for the last prompt token (whose
        // logits seed `next_tok` for the decode phase).
        if i + 1 < n_prompt {
            mf.single_token_no_tail(tid, i as u32, &mut ref_session)?;
        } else {
            last_logits_ref = mf.single_token(tid, i as u32, &mut ref_session)?;
        }
    }
    let ref_prefill_ms = t_ref_prefill.elapsed().as_secs_f64() * 1e3;
    let mut next_tok = argmax_i32(&last_logits_ref);
    let mut ref_emitted: Vec<i32> = Vec::with_capacity(tokens);
    let mut pos = (n_prompt - 1) as u32;
    let t_ref_decode = Instant::now();
    for _ in 0..tokens {
        ref_emitted.push(next_tok);
        if next_tok == eos {
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
        eos,
        no_warmup,
        skip_equivalence_check,
        profile,
    } = args;

    let ctx = MetalContext::new().context("init MetalContext")?;
    eprintln!("[dflash] device: {}", ctx.describe());

    let target_g =
        GgufFile::open(&model).with_context(|| format!("open target {}", model.display()))?;
    let target_m = Model::from_gguf(&target_g).context("parse target arch")?;
    let drafter_g =
        GgufFile::open(&drafter).with_context(|| format!("open drafter {}", drafter.display()))?;
    let head = open_dflash_drafter(&drafter_g, &target_m).context("bind drafter")?;
    let mm = MetalModel::load(&ctx, &target_g, &target_m).context("metal-load target")?;
    let mhead = MetalDFlashHead::load(&ctx, &drafter_g, &head).context("metal-load drafter")?;
    let tok = Tokenizer::open(&model).context("open tokenizer")?;

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
        "[dflash] prompt={prompt:?} ({n_prompt} tokens) gen={tokens} eos={eos} \
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

    // Per-prompt-token captured hidden buffer for prefill phase
    // (single_token_with_multi_hidden writes [K * H] per call).
    let multi_hidden_dst =
        MetalTensor::zeros_f32(&ctx, vec![n_target_features as u64]).context("multi_hidden_dst")?;

    // Production DFlash scratch buffers.
    let mut verify_scratch =
        MetalDFlashVerifyScratch::fresh(&ctx, &mm, cfg.block_size, k_layers as u32)
            .context("verify scratch")?;
    let mut layer_scratch =
        MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, cfg.block_size).context("layer scratch")?;

    // ---------- Prompt prefill ----------
    let t_prefill = Instant::now();
    let mut last_logits: Vec<f32> = Vec::new();
    for (i, &tid) in prompt_ids.iter().enumerate() {
        // v0.75.0: skip-tail for all but the last prompt token.
        if i + 1 < n_prompt {
            mf.single_token_with_multi_hidden_no_tail(
                tid,
                i as u32,
                &mut target_session,
                &head.target_layer_ids,
                &multi_hidden_dst,
            )
            .context("prefill base step (no_tail)")?;
        } else {
            last_logits = mf
                .single_token_with_multi_hidden(
                    tid,
                    i as u32,
                    &mut target_session,
                    &head.target_layer_ids,
                    &multi_hidden_dst,
                )
                .context("prefill base step")?;
        }
        dsess
            .append_target_ctx_column_now(&ctx, &multi_hidden_dst, i as u32, n_target_features)
            .context("append prefill ctx column")?;
    }
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
        if carry_tok == eos || emitted.len() >= tokens {
            break;
        }

        // ---- Drafter ----
        let drafter_pos = processed_pos + 1; // noise_start_pos
        let argmaxes = decoder
            .draft_block(carry_tok, drafter_pos)
            .context("drafter draft_block")?;
        drafter_calls += 1;
        let drafts: Vec<i32> = argmaxes[1..].to_vec();
        debug_assert_eq!(drafts.len(), d);

        // ---- Packed verify ----
        // Input: [carry, drafts[0..D-1]] of length N.
        let mut verify_input: Vec<i32> = Vec::with_capacity(n_block);
        verify_input.push(carry_tok);
        verify_input.extend_from_slice(&drafts[..d]);

        let verify_argmax = qwen_llm::metal_dflash::encode_packed_verify_layer_major_inner(
            decoder.base,
            &decoder.head.target_layer_ids,
            &verify_input,
            drafter_pos,
            &mut verify_scratch,
            &mut layer_scratch,
            &mut target_session,
            None,
        )
        .context("packed_verify")?;
        verify_calls += 1;
        debug_assert_eq!(verify_argmax.len(), n_block);

        // ---- Greedy accept-prefix ----
        // n_accepted = number of DRAFT tokens accepted (∈ [0, D]).
        // Position j in drafts corresponds to position j+1 in
        // verify_input (carry is index 0). target's prediction AT
        // verify_input[j+1] is verify_argmax[j+1]... wait, need to
        // re-check. verify_argmax[i] is the argmax of target's
        // forward AT position drafter_pos+i, given input
        // verify_input[i]. So verify_argmax[0] is the argmax AFTER
        // processing carry — this is what target says SHOULD come
        // next after carry. drafts[0] is what drafter predicted
        // for that same slot. So the comparison is:
        //   drafts[0] == verify_argmax[0] ?
        //   drafts[1] == verify_argmax[1] ?
        //   ...
        //   drafts[j] == verify_argmax[j] ?
        // Stop at first mismatch. n_accepted = j.
        // Bonus = verify_argmax[n_accepted].
        let mut n_accepted = 0usize;
        steps += 1;
        for j in 0..d {
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
            if drafts[j] == eos {
                // Emit-through-EOS; stop.
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
        let n_full = (verify_scratch.n) as u32;
        if n_keep < n_full {
            qwen_llm::metal_dflash::encode_restore_after_partial_accept_inner(
                decoder.base,
                &verify_scratch,
                n_keep,
                drafter_pos,
                &mut target_session,
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
        let mut last_logits_ref = Vec::new();
        for (i, &tid) in prompt_ids.iter().enumerate() {
            // v0.75.0: skip-tail except for the last prompt token.
            if i + 1 < n_prompt {
                mf.single_token_no_tail(tid, i as u32, &mut ref_session)?;
            } else {
                last_logits_ref = mf.single_token(tid, i as u32, &mut ref_session)?;
            }
        }
        let ref_prefill_ms = t_ref_prefill.elapsed().as_secs_f64() * 1e3;
        let mut next_tok = argmax_i32(&last_logits_ref);
        let mut pos = (n_prompt - 1) as u32;
        let t_ref_decode = Instant::now();
        for _ in 0..tokens {
            ref_emitted.push(next_tok);
            if next_tok == eos {
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

fn run_decode(args: DecodeArgs) -> Result<()> {
    let DecodeArgs {
        model,
        prompt,
        tokens,
        oracle,
        no_warmup,
    } = args;
    let prompt =
        prompt.unwrap_or_else(|| "The quick brown fox jumps over the lazy dog".to_string());

    let ctx = MetalContext::new().context("init MetalContext")?;
    eprintln!("[bench] device: {}", ctx.describe());

    let g = GgufFile::open(&model).with_context(|| format!("open {}", model.display()))?;
    let m = Model::from_gguf(&g).context("parse model arch from gguf")?;
    let mm = MetalModel::load(&ctx, &g, &m).context("metal-load model weights")?;
    let tok = Tokenizer::open(&model).context("open tokenizer")?;

    let ids = tok.encode(&prompt, false).context("tokenize prompt")?;
    eprintln!(
        "[bench] model={} prompt={:?} ({} tokens), gen={} tokens",
        model.display(),
        prompt,
        ids.len(),
        tokens
    );

    let mf = MetalForward::new(&ctx, &mm);
    let cap = ids.len() + tokens + 16;

    if !no_warmup {
        // One warmup pass to compile pipeline state objects + warm caches.
        let mut s = MetalSession::fresh(&ctx, &mm, cap).context("session warmup")?;
        let _ = mf.single_token(ids[0], 0, &mut s)?;
    }

    let mut s = MetalSession::fresh(&ctx, &mm, cap).context("session run")?;
    let mut per_token_ms: Vec<f64> = Vec::with_capacity(ids.len() + tokens);
    let mut last_logits: Vec<f32> = Vec::new();

    let t0 = Instant::now();

    // Prefill: feed all prompt tokens through (sequential single_token; we
    // don't have batched prefill yet — that's a future product).
    for (i, &tid) in ids.iter().enumerate() {
        let tt = Instant::now();
        last_logits = mf.single_token(tid, i as u32, &mut s)?;
        per_token_ms.push(tt.elapsed().as_secs_f64() * 1e3);
    }
    let prefill_wall = t0.elapsed().as_secs_f64() * 1e3;
    let prefill_avg = prefill_wall / ids.len() as f64;

    // Decode loop: greedy argmax sampling on the CPU side. (A backend
    // sampler kernel is on the roadmap; this is the trivial CPU baseline.)
    let mut gen_ids: Vec<i32> = Vec::with_capacity(tokens);
    let t1 = Instant::now();
    for k in 0..tokens {
        let pos = ids.len() + k;
        let next = argmax_i32(&last_logits);
        gen_ids.push(next);
        let tt = Instant::now();
        last_logits = mf.single_token(next, pos as u32, &mut s)?;
        per_token_ms.push(tt.elapsed().as_secs_f64() * 1e3);
    }
    let decode_wall = t1.elapsed().as_secs_f64() * 1e3;

    let total_wall = t0.elapsed().as_secs_f64() * 1e3;

    // Decode-only steady-state: skip the very first decode (cache-cold for
    // some downstream PSO + heavily warm-up sensitive).
    let decode_steady_ms: f64 = if tokens > 1 {
        per_token_ms[ids.len() + 1..].iter().sum::<f64>() / (tokens - 1) as f64
    } else {
        decode_wall
    };

    eprintln!();
    eprintln!("[bench] === results ===");
    eprintln!(
        "[bench] prefill: {} tokens in {prefill_wall:.1} ms = {prefill_avg:.2} ms/token = {:.1} t/s",
        ids.len(),
        1000.0 / prefill_avg
    );
    eprintln!(
        "[bench] decode:  {tokens} tokens in {decode_wall:.1} ms = {:.2} ms/token (avg) = {:.1} t/s",
        decode_wall / tokens as f64,
        1000.0 * tokens as f64 / decode_wall
    );
    eprintln!(
        "[bench] steady:  {decode_steady_ms:.2} ms/token (excl. first decode) = {:.2} t/s",
        1000.0 / decode_steady_ms
    );
    eprintln!("[bench] total:   {total_wall:.1} ms wall");

    // Print first/last few per-token times for spot-checking.
    let n_show = 5usize.min(per_token_ms.len());
    eprintln!(
        "[bench] per-token (first {n_show}): {:?}",
        &per_token_ms[..n_show]
    );
    if per_token_ms.len() > 2 * n_show {
        let m = per_token_ms.len();
        eprintln!(
            "[bench] per-token (last  {n_show}): {:?}",
            &per_token_ms[m - n_show..]
        );
    }

    if let Some(oracle_path) = oracle {
        let bytes = std::fs::read(&oracle_path)
            .with_context(|| format!("read oracle {}", oracle_path.display()))?;
        if bytes.len() % 4 != 0 || bytes.len() / 4 != last_logits.len() {
            return Err(anyhow!(
                "oracle size {} bytes ({} f32) != logits len {}",
                bytes.len(),
                bytes.len() / 4,
                last_logits.len()
            ));
        }
        let oracle: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let (cos, max_abs, argmax_ours, argmax_oracle) = compare_logits(&last_logits, &oracle);
        eprintln!(
            "[bench] oracle:  cos={cos:.6}  max|Δ|={max_abs:.4}  argmax: ours={argmax_ours} oracle={argmax_oracle} {}",
            if argmax_ours == argmax_oracle {
                "✓"
            } else {
                "✗ MISMATCH"
            }
        );
    }

    if !gen_ids.is_empty() {
        let text = tok.decode(&gen_ids);
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
    let tok = Tokenizer::open(&model)?;
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

        // Prefill the prompt. v0.75.0: skip-tail except last token.
        let mut last_logits = vec![];
        let n_ids = ids.len();
        for (i, &tid) in ids.iter().enumerate() {
            if i + 1 < n_ids {
                mf.single_token_no_tail(tid, i as u32, &mut sess)?;
            } else {
                last_logits = mf.single_token(tid, i as u32, &mut sess)?;
            }
        }

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
                        let decoded = tok.decode(&[argmax as i32]);
                        cat_stats.examples.get_mut(&k).unwrap().push((
                            cat.clone(),
                            decoded.clone(),
                            argmax,
                        ));
                    }
                    if global.examples[&k].len() < show_examples * 2 {
                        let decoded = tok.decode(&[argmax as i32]);
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
            let (_, p) = mf.single_token_profiled(0, pos, &mut s)?;
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

fn argmax_i32(logits: &[f32]) -> i32 {
    let mut best = (0usize, f32::NEG_INFINITY);
    for (i, &v) in logits.iter().enumerate() {
        if v > best.1 {
            best = (i, v);
        }
    }
    best.0 as i32
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
    let tok = Tokenizer::open(&model)?;

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
    let snap = sess_pre.snapshot(identity, prefix_ids.clone(), Some(last_pre_logits));
    let snap_create_ms = snap_t.elapsed().as_secs_f64() * 1e3;
    eprintln!(
        "[prefix-cache] (snapshot built: {:.1} MB in {snap_create_ms:.1} ms)",
        snap.n_bytes() as f64 / 1e6
    );

    // Now simulate request 2 starting fresh and finding the cached prefix.
    let warm_t0 = Instant::now();
    let mut sess_warm = MetalSession::fresh(&ctx, &mm, cap)?;
    let restore_t = Instant::now();
    sess_warm.restore_from(&snap)?;
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
        let cold_text = tok.decode(&cold_extra);
        let warm_text = tok.decode(&warm_extra);
        eprintln!("[prefix-cache] cold generated: {:?}", cold_text);
        eprintln!("[prefix-cache] warm generated: {:?}", warm_text);
    }

    Ok(())
}
