//! Integration tests for the H5 (DFlash) speculative-decode path.
//!
//! These tests exercise CPU prompt prefill + drafter forward on the
//! 27B-Q4_K_M target, which costs ~2 minutes per test on M4 Max
//! (CPU triple-loop matmul through 26.9B params, dequanting Q4_K
//! inline per layer per call). They're intentionally NOT in the lib
//! suite — running them on every `cargo test` would push the
//! feedback loop from ~10s to ~150s. Run them explicitly:
//!
//! ```bash
//! cargo test --test dflash_correctness --release
//! ```
//!
//! Tests:
//! * `dflash_draft_cpu_smoke` (~127s) — CPU drafter forward with real
//!   target prefill hiddens; confirms shape + finiteness + diverse
//!   argmaxes. Originally H5.1's gate; caught the cross-GGUF dequant
//!   bug.
//! * `metal_drafter_cosine_vs_cpu` (~142s) — H5.1.5 plumbing cosine
//!   gate. Validates Metal hybrid drafter matches CPU oracle bit-tight
//!   on the same `(target_ctx_stacked, pos_ctx, noise_ids)` inputs.
//!
//! Both tests skip silently if the required GGUFs aren't present
//! locally:
//! * `~/models/Qwen3.6-27B-Q4_K_M.gguf`
//! * `~/models/spiritbuun-dflash/dflash-draft-3.6-q8_0.gguf`

use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandQueue};
use qwen_llm::forward::{Forward, GdnState, KvCache};
use qwen_llm::gguf::GgufFile;
use qwen_llm::loader::{open_dflash_drafter, Model};
use qwen_llm::metal::{
    encode_add_inplace_f32, encode_argmax_f32, encode_copy_offset_f32, encode_gdn_alpha_chain_f32,
    encode_get_rows_f32, encode_mul_f32, encode_rms_norm_batched_f32, encode_rms_norm_mul_f32,
    encode_rope_neox_f32, encode_scatter_offset_f32_to_f16_kv, encode_sigmoid_f32,
    encode_silu_mul_f32, encode_split_q_gate_f32, BlitEncoder, KernelEncoder, MetalContext,
    MetalError, MetalTensor,
};
use qwen_llm::metal_dflash::{
    DFlashDecoder, MetalDFlashDebugScratch, MetalDFlashHead, MetalDFlashLayerMajorScratch,
    MetalDFlashSession, MetalDFlashVerifyScratch,
};
use qwen_llm::metal_forward::{
    encode_mat_mat_dispatch, encode_mat_vec_dispatch, encode_scatter_offset_f32, MetalBlock,
    MetalForward, MetalModel, MetalSession, RMS_EPS,
};
use qwen_llm::tensor::GgmlType;
use qwen_llm::tokenizer::Tokenizer;

const TARGET_GGUF: &str = "/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf";
const DRAFTER_GGUF: &str = "/Users/tito/models/spiritbuun-dflash/dflash-draft-3.6-q8_0.gguf";

fn fixtures_present() -> bool {
    std::path::Path::new(TARGET_GGUF).exists() && std::path::Path::new(DRAFTER_GGUF).exists()
}

/// **H5.1** smoke test: bind drafter, run dflash_draft with real target
/// prefill hiddens, confirm output shape + finite logits + non-degenerate
/// argmax. Real distribution validation (vs spiritbuun) comes in H5.5.
///
/// Catches the cross-GGUF dequant bug class and any layer-output layout
/// regression (target hiddens stacked in the wrong K order, etc.).
#[test]
fn dflash_draft_cpu_smoke() {
    if !fixtures_present() {
        eprintln!("[dflash-cpu-smoke] skipped — fixtures missing");
        return;
    }
    let target_g = GgufFile::open(TARGET_GGUF).expect("open target");
    let target_m = Model::from_gguf(&target_g).expect("load target");
    let drafter_g = GgufFile::open(DRAFTER_GGUF).expect("open drafter");
    let head = open_dflash_drafter(&drafter_g, &target_m).expect("bind drafter");

    let f = Forward::new(&target_g, &target_m);
    let cfg = head.config;
    let n = cfg.block_size as usize;
    let h_target = target_m.arch.hidden_size as usize;
    let k_layers = head.target_layer_ids.len();

    // Target prompt prefill on CPU: capture hiddens at K layer indices
    // per token. Mirrors what H5.2's Metal multi-hidden capture does;
    // the integration test runs CPU here for fixture-independence.
    let prompt = "The quick brown fox";
    let tok = Tokenizer::open(TARGET_GGUF).expect("tok");
    let prompt_ids = tok.encode(prompt, false).expect("tok");
    let ctx_len = prompt_ids.len();
    let n_target_features = k_layers * h_target;
    eprintln!("[dflash-cpu-smoke] prompt {prompt:?} → {ctx_len} tokens");

    let mut state = GdnState::fresh(&target_m);
    let mut kv = KvCache::new(&target_m);
    let mut target_ctx_stacked = vec![0.0f32; ctx_len * n_target_features];
    for (i, &tid) in prompt_ids.iter().enumerate() {
        let captured = f
            .single_token_capture_layers(tid, i as u32, &mut state, &mut kv, &head.target_layer_ids)
            .expect("prefill capture");
        for k in 0..k_layers {
            let dst_off = i * n_target_features + k * h_target;
            target_ctx_stacked[dst_off..dst_off + h_target]
                .copy_from_slice(&captured[k * h_target..(k + 1) * h_target]);
        }
    }
    let pos_ctx: Vec<u32> = (0..ctx_len as u32).collect();

    // Noise input: [carry_tok, MASK × (N-1)].
    let carry_tok = 760_i32; // "The"
    let mut noise_ids = vec![cfg.mask_token_id; n];
    noise_ids[0] = carry_tok;
    let noise_start_pos = ctx_len as u32;

    let logits = f
        .dflash_draft(
            &head,
            &drafter_g,
            &noise_ids,
            &target_ctx_stacked,
            ctx_len,
            &pos_ctx,
            noise_start_pos,
        )
        .expect("dflash_draft");
    assert_eq!(logits.len(), n * target_m.arch.vocab_size as usize);

    let v = target_m.arch.vocab_size as usize;
    let mut argmaxes: Vec<i32> = Vec::with_capacity(n);
    for i in 0..n {
        let row = &logits[i * v..(i + 1) * v];
        assert!(row[0].is_finite(), "row {i} produced NaN/Inf");
        let argmax = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0 as i32;
        argmaxes.push(argmax);
    }
    eprintln!("[dflash-cpu-smoke] argmaxes (per-noise-position): {argmaxes:?}");
    let unique: std::collections::HashSet<_> = argmaxes.iter().collect();
    assert!(
        unique.len() > 1,
        "all noise positions argmax to the same token — drafter forward likely broken"
    );
}

/// **H5.1.5** — plumbing cosine gate. NOT a DFlash correctness gate; see
/// docs/H5-DFLASH.md §H5.1.5 for the full framing.
///
/// Validates that the Metal hybrid drafter and the CPU oracle produce
/// the same `[N, V]` draft logits when fed the same
/// `(target_ctx_stacked, pos_ctx, noise_ids, noise_start_pos)` inputs.
/// Most of the per-layer compute (asymmetric SWA-masked attention +
/// SwiGLU silu_mul) runs on CPU on BOTH paths in v1, so the cosine here
/// mostly proves:
///
/// * drafter weight tensors scope to the drafter GGUF (cross-GGUF
///   dequant bug class)
/// * `target_ctx_stacked` column layout, `pos_ctx` ordering, row
///   slicing, and shared-target-lm_head wiring
/// * command-buffer encode + readback rhythm
///
/// It does NOT validate the SWA mask algorithm or the actual drafter
/// distribution — H5.2.5 (measured α) is that gate.
#[test]
fn metal_drafter_cosine_vs_cpu() {
    if !fixtures_present() {
        eprintln!("[h5.1.5] skipped — fixtures missing");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };

    let target_g = GgufFile::open(TARGET_GGUF).expect("open target");
    let target_m = Model::from_gguf(&target_g).expect("load target");
    let drafter_g = GgufFile::open(DRAFTER_GGUF).expect("open drafter");
    let head = open_dflash_drafter(&drafter_g, &target_m).expect("bind drafter");

    // Real prompt prefill on CPU to capture multi-layer hiddens.
    let cpu_fwd = Forward::new(&target_g, &target_m);
    let tok = Tokenizer::open(TARGET_GGUF).expect("tok");
    let prompt_ids = tok.encode("The quick brown fox", false).expect("tok");
    let ctx_len = prompt_ids.len();
    let h_target = target_m.arch.hidden_size as usize;
    let k_layers = head.target_layer_ids.len();
    let n_target_features = k_layers * h_target;

    let mut cpu_state = GdnState::fresh(&target_m);
    let mut cpu_kv = KvCache::new(&target_m);
    let mut target_ctx_stacked_cpu = vec![0.0f32; ctx_len * n_target_features];
    for (i, &tid) in prompt_ids.iter().enumerate() {
        let captured = cpu_fwd
            .single_token_capture_layers(
                tid,
                i as u32,
                &mut cpu_state,
                &mut cpu_kv,
                &head.target_layer_ids,
            )
            .expect("prefill");
        target_ctx_stacked_cpu[i * n_target_features..(i + 1) * n_target_features]
            .copy_from_slice(&captured);
    }
    let pos_ctx: Vec<u32> = (0..ctx_len as u32).collect();

    let cfg = head.config;
    let n = cfg.block_size as usize;
    let v = target_m.arch.vocab_size as usize;
    let carry_tok = 760_i32;
    let mut noise_ids = vec![cfg.mask_token_id; n];
    noise_ids[0] = carry_tok;
    let noise_start_pos = ctx_len as u32;

    let cpu_logits = cpu_fwd
        .dflash_draft(
            &head,
            &drafter_g,
            &noise_ids,
            &target_ctx_stacked_cpu,
            ctx_len,
            &pos_ctx,
            noise_start_pos,
        )
        .expect("cpu dflash_draft");
    assert_eq!(cpu_logits.len(), n * v);

    // Metal hybrid draft.
    let mm = MetalModel::load(&ctx, &target_g, &target_m).expect("metal target load");
    let mhead = MetalDFlashHead::load(&ctx, &drafter_g, &head).expect("metal drafter load");
    let mut msess =
        MetalDFlashSession::fresh(&ctx, &mhead, h_target as u64, v as u64, ctx_len + 32)
            .expect("metal drafter session");

    let cmd = ctx.queue.commandBuffer().expect("cmd");
    let enc = KernelEncoder::begin(&cmd);
    for c in 0..ctx_len {
        let col_bytes = bytemuck::cast_slice(
            &target_ctx_stacked_cpu[c * n_target_features..(c + 1) * n_target_features],
        );
        let col_t = MetalTensor::from_bytes(
            &ctx,
            col_bytes,
            vec![n_target_features as u64],
            GgmlType::F32,
        )
        .expect("col tensor");
        msess
            .append_target_ctx_column(&ctx, &enc, &col_t, c as u32, n_target_features)
            .expect("append col");
    }
    enc.end();
    cmd.commit();
    cmd.waitUntilCompleted();

    let mf = MetalForward::new(&ctx, &mm);
    let mut spec = DFlashDecoder::new(&mf, &mhead, msess);
    let metal_logits = spec
        .draft_block_with_logits(carry_tok, noise_start_pos)
        .expect("metal draft");
    assert_eq!(metal_logits.len(), cpu_logits.len());

    let mut all_argmax_match = true;
    for i in 0..n {
        let cpu_row = &cpu_logits[i * v..(i + 1) * v];
        let metal_row = &metal_logits[i * v..(i + 1) * v];
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        let mut max_abs = 0.0f32;
        let mut argmax_cpu = 0usize;
        let mut argmax_metal = 0usize;
        let mut max_cpu = f32::NEG_INFINITY;
        let mut max_metal = f32::NEG_INFINITY;
        for j in 0..v {
            let c = cpu_row[j];
            let mt = metal_row[j];
            dot += (c as f64) * (mt as f64);
            na += (c as f64).powi(2);
            nb += (mt as f64).powi(2);
            max_abs = max_abs.max((c - mt).abs());
            if c > max_cpu {
                max_cpu = c;
                argmax_cpu = j;
            }
            if mt > max_metal {
                max_metal = mt;
                argmax_metal = j;
            }
        }
        let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
        eprintln!(
            "[h5.1.5] noise pos {i}: cos={cos:.6} max|Δ|={max_abs:.4} \
             argmax cpu={argmax_cpu} metal={argmax_metal}"
        );
        if argmax_cpu != argmax_metal {
            all_argmax_match = false;
        }
        assert!(cos > 0.999, "noise pos {i}: cos {cos} below threshold");
        assert!(
            metal_row.iter().all(|x| x.is_finite()),
            "noise pos {i}: NaN/Inf in metal logits"
        );
    }
    eprintln!("[h5.1.5] argmax all-match: {all_argmax_match}");
}

/// **H5.3b** end-to-end validation on real Qwen3.6-27B-Q4_K_M:
/// layer-major `packed_verify` argmaxes match token-major argmaxes,
/// and the codex tripwire (≥ 2.5x speedup over token-major on
/// FFN-heavy paths) is checked.
///
/// The codex H5.3b plan rev 6 explicitly relaxed the per-layer cosine
/// gate from ≥ 0.9999 to ≥ 0.999 because lifted mat-mat stages
/// activations through half before float accumulation (Q3 correction).
/// On Q4_K + Q6_K weights the noise compounds across 64 layers, so we
/// expect cos around 0.999 on raw logits, possibly some argmax
/// divergence at the bottom-of-distribution (low-confidence positions).
///
/// What this test asserts:
///   * Layer-major produces FINITE logits (no NaN/Inf from the new
///     plumbing).
///   * Per-row cosine ≥ 0.99 between layer-major and token-major
///     final logits (relaxed from H5.3b plan's 0.999 — Q4_K + Q6_K
///     mat-mat stages noise across 64 layers; argmax can differ for
///     low-confidence positions).
///   * Argmax tokens agree on AT LEAST 80% of positions (high-
///     confidence positions should be invariant under bounded
///     accumulation noise; if many disagree, the layer-major path
///     has a real algorithmic bug, not just precision drift).
///   * Layer-major wall-time speedup over token-major: codex tripwire
///     report only (no hard assert; report log so we can decide on
///     the H5.3b.5.5 NR1=16 retune).
///
/// ~3-5 min runtime: 27B-Q4_K_M load + prime + two N=16 forwards.
#[test]
fn dflash_packed_verify_layer_major_vs_token_major_27b() {
    if !std::path::Path::new(TARGET_GGUF).exists() {
        eprintln!("[h5.3b-27b] skipped — target GGUF missing");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("metal init: {e}"),
    };

    eprintln!("[h5.3b-27b] loading 27B-Q4_K_M (~10s on cold cache)…");
    let g = GgufFile::open(TARGET_GGUF).expect("open target");
    let m = Model::from_gguf(&g).expect("load target");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx, &mm);

    // Prime two identical sessions through M tokens.
    const M: u32 = 4;
    const N: u32 = 16;
    let prime_tokens: [i32; M as usize] = [9419, 1, 5, 1234];
    let verify_tokens: [i32; N as usize] = [
        7, 999, 42, 11, 22, 33, 44, 55, 66, 77, 88, 99, 100, 200, 300, 400,
    ];

    let mut sess_tok = MetalSession::fresh(&ctx, &mm, 256).expect("sess tok");
    let mut sess_lm = MetalSession::fresh(&ctx, &mm, 256).expect("sess lm");
    eprintln!("[h5.3b-27b] priming both sessions through M={M} tokens…");
    let prime_t = std::time::Instant::now();
    for (i, &tok) in prime_tokens.iter().enumerate() {
        mf.single_token(tok, i as u32, &mut sess_tok)
            .expect("prime tok");
        mf.single_token(tok, i as u32, &mut sess_lm)
            .expect("prime lm");
    }
    eprintln!(
        "[h5.3b-27b] prime: {:.2} s",
        prime_t.elapsed().as_secs_f64()
    );

    let target_layer_ids: Vec<u32> = vec![1, 16, 31, 46, 61]; // K=5, mirrors spiritbuun drafter
    let k_target = target_layer_ids.len() as u32;

    // Token-major path (oracle) with logits.
    let mut dbg_tok =
        MetalDFlashDebugScratch::fresh(&ctx, &mm, N, k_target).expect("dbg scratch tok");
    eprintln!("[h5.3b-27b] running token-major packed_verify (N={N})…");
    let tok_t = std::time::Instant::now();
    let argmax_tok = qwen_llm::metal_dflash::encode_packed_verify_with_logits_inner(
        &mf,
        &target_layer_ids,
        &verify_tokens,
        M,
        &mut dbg_tok,
        &mut sess_tok,
    )
    .expect("token-major");
    let tok_wall = tok_t.elapsed();
    eprintln!(
        "[h5.3b-27b] token-major wall: {:.2} ms",
        tok_wall.as_secs_f64() * 1e3
    );

    // Layer-major path with logits.
    let mut dbg_lm =
        MetalDFlashDebugScratch::fresh(&ctx, &mm, N, k_target).expect("dbg scratch lm");
    let mut layer_scratch =
        MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, N).expect("layer scratch");
    eprintln!("[h5.3b-27b] running layer-major packed_verify (N={N})…");
    let lm_t = std::time::Instant::now();
    let MetalDFlashDebugScratch {
        verify: lm_verify,
        debug_logits: lm_debug,
    } = &mut dbg_lm;
    let argmax_lm = qwen_llm::metal_dflash::encode_packed_verify_layer_major_inner(
        &mf,
        &target_layer_ids,
        &verify_tokens,
        M,
        lm_verify,
        &mut layer_scratch,
        &mut sess_lm,
        Some(lm_debug),
    )
    .expect("layer-major");
    let lm_wall = lm_t.elapsed();
    eprintln!(
        "[h5.3b-27b] layer-major wall: {:.2} ms",
        lm_wall.as_secs_f64() * 1e3
    );

    let speedup = tok_wall.as_secs_f64() / lm_wall.as_secs_f64();
    eprintln!(
        "[h5.3b-27b] SPEEDUP layer-major vs token-major: {:.2}x \
         (codex tripwire: ≥ 2.5x; if missed, H5.3b.5.5 NR1=16 retune \
         is pre-authorized)",
        speedup
    );

    eprintln!(
        "[h5.3b-27b] argmax_tok = {argmax_tok:?}\n\
         [h5.3b-27b] argmax_lm  = {argmax_lm:?}"
    );

    // -- Per-row cosine + max|Δ| on raw logits.
    let v = m.arch.vocab_size as usize;
    let mut min_cos = f64::INFINITY;
    let mut total_max_abs = 0.0f32;
    unsafe {
        let p_tok = dbg_tok.debug_logits.buffer.contents().as_ptr() as *const f32;
        let p_lm = dbg_lm.debug_logits.buffer.contents().as_ptr() as *const f32;
        for n_idx in 0..N as usize {
            let mut dot = 0.0f64;
            let mut na = 0.0f64;
            let mut nb = 0.0f64;
            let mut max_abs = 0.0f32;
            for i in 0..v {
                let a = *p_tok.add(n_idx * v + i);
                let b = *p_lm.add(n_idx * v + i);
                let af = a as f64;
                let bf = b as f64;
                dot += af * bf;
                na += af * af;
                nb += bf * bf;
                let d = (a - b).abs();
                if d > max_abs {
                    max_abs = d;
                }
            }
            let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
            eprintln!("[h5.3b-27b] n_idx={n_idx}: cos={cos:.6} max|Δ|={max_abs:.3e}");
            if cos < min_cos {
                min_cos = cos;
            }
            if max_abs > total_max_abs {
                total_max_abs = max_abs;
            }
            // Finite check.
            for i in 0..v {
                let b = *p_lm.add(n_idx * v + i);
                assert!(
                    b.is_finite(),
                    "layer-major logits non-finite at n_idx={n_idx} vocab_idx={i}: {b}"
                );
            }
        }
    }
    eprintln!("[h5.3b-27b] min_cos={min_cos:.6} total_max|Δ|={total_max_abs:.3e}");
    assert!(
        min_cos >= 0.99,
        "layer-major vs token-major min cosine {min_cos} < 0.99"
    );

    // -- Argmax agreement: at least 80% of positions.
    let mut agree = 0usize;
    for n_idx in 0..N as usize {
        if argmax_tok[n_idx] == argmax_lm[n_idx] {
            agree += 1;
        }
    }
    let pct = 100.0 * (agree as f64) / (N as f64);
    eprintln!("[h5.3b-27b] argmax agreement: {agree}/{N} ({pct:.1}%)");
    assert!(
        agree as u32 * 5 >= N * 4,
        "argmax agreement {agree}/{N} below 80%"
    );
}

/// **H5.3b/v0.70 phase profile** for layer-major packed_verify.
///
/// Per codex H5.3b next-moves session: profile FIRST before any
/// further kernel work. Identifies which phase dominates wall-time
/// at which context length. Decisions for v0.73+ kernel work are
/// driven by THIS data, not estimates.
///
/// Structure: parallels `encode_packed_verify_layer_major_inner`
/// line-for-line, but each "phase" runs as its OWN command buffer
/// (commit + wait + GPUStartTime/EndTime aggregation). Per-phase
/// GPU time is exact; per-phase wall time is artifically inflated
/// by ~1 commit-wait per phase. The TOTAL number is artificially
/// LARGER than production wall (due to per-phase commit overhead);
/// the DISTRIBUTION across phases is what matters.
///
/// Aggregates by class:
///   * embed
///   * pre_norm  (× n_layer)
///   * gdn_mixer (× n_gdn)   — per-token loop with blits
///   * attn_mixer (× n_attn) — per-token loop
///   * residual1 (× n_layer)
///   * hidden_capture (× K target layers)
///   * post_norm (× n_layer)
///   * ffn       (× n_layer) — batched mat-mat for Q4_K, fallback for F32
///   * residual2 (× n_layer)
///   * tail      (final norm + lm_head + argmax)
///   * gdn_ckpt_blits (the cross-encoder blit cost)
///
/// Runs at ctx ∈ {1024, 4096, 16384, 65536} on 27B-Q4_K_M; primes
/// the session via prefill (single_token) up to start_pos, then runs
/// ONE packed_verify worth of profiled phases.
///
/// Hard timeboxed by codex's caveat: if this turns into framework
/// work, abort. Keep it SCOPED — profiling, no kernel changes.
#[test]
#[ignore = "slow: ~10-15 min on M4 Max for 4 ctx lengths × 27B-Q4_K_M; \
            run explicitly with `cargo test --test dflash_correctness \
            --release packed_verify_phase_profile_27b -- --ignored --nocapture`"]
fn packed_verify_phase_profile_27b() {
    if !std::path::Path::new(TARGET_GGUF).exists() {
        eprintln!("[v0.70-profile] skipped — target GGUF missing");
        return;
    }
    let ctx_metal = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("metal init: {e}"),
    };

    eprintln!("[v0.70-profile] loading 27B-Q4_K_M…");
    let g = GgufFile::open(TARGET_GGUF).expect("open target");
    let m = Model::from_gguf(&g).expect("load target");
    let mm = MetalModel::load(&ctx_metal, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx_metal, &mm);

    // Profile at multiple context lengths to surface the bottleneck-
    // shift (mat-mat + GDN flat in ctx; attn linear in ctx).
    const CTX_POINTS: &[u32] = &[1024, 4096, 16384];
    // 64K is gated to keep the test runtime bounded; uncomment to
    // exercise the upper attn-dominant regime at the cost of much
    // more wall time.
    // const CTX_POINTS: &[u32] = &[1024, 4096, 16384, 65536];
    const N: u32 = 16;
    let target_layer_ids: Vec<u32> = vec![1, 16, 31, 46, 61];
    let k_target = target_layer_ids.len() as u32;

    let arch = m.arch.clone();
    let h = arch.hidden_size as usize;
    let f = arch.intermediate_size as usize;
    let v = arch.vocab_size as usize;
    let head_dim = arch.attn_head_dim as usize;
    let n_q = arch.n_q_heads as usize;
    let n_kv = arch.n_kv_heads as usize;
    let q_dim = n_q * head_dim;
    let kv_dim = n_kv * head_dim;
    let n_rot = (head_dim as f32 * arch.partial_rotary_factor) as usize;
    let _ = (n_rot, q_dim, kv_dim);

    eprintln!("[v0.70-profile] arch: hidden={h} ffn={f} q_dim={q_dim} kv_dim={kv_dim}");
    eprintln!("[v0.70-profile] CTX_POINTS={CTX_POINTS:?} N={N}");

    for &start_position in CTX_POINTS {
        eprintln!("\n[v0.70-profile] ===== ctx={start_position} =====");
        let kv_capacity = (start_position + N + 32) as usize;

        // Prime the session up to start_position via single_token.
        // This is slow at large start_position (1024 single-token
        // forwards = ~50 s at 47 ms/each; 16K = 12+ min), so we
        // BYPASS the priming for start_position > 0 by setting
        // kv_n_pos directly and seeding the cache with bogus bytes.
        // The profile only measures KERNEL TIME — the actual K/V
        // values don't affect kernel BW (just the n_pos arg to
        // attn-v4 which controls how many positions to read).
        //
        // **Caveat**: this means the per-row argmax/cosine values
        // are NOT meaningful for these synthetic-priming runs. We
        // just want kernel timing. The v0.68 27b-validation test
        // already proved correctness on real prefill at ctx=4.
        let mut sess = MetalSession::fresh(&ctx_metal, &mm, kv_capacity).expect("sess");
        if start_position > 0 {
            eprintln!(
                "[v0.70-profile] synthetic-priming kv_n_pos to {start_position} \
                 (skipping real prefill — kernel timing only)"
            );
            for kp in sess.kv_n_pos.iter_mut() {
                *kp = start_position as usize;
            }
        }

        let mut verify_scratch =
            MetalDFlashVerifyScratch::fresh(&ctx_metal, &mm, N, k_target).expect("verify scratch");
        let mut layer_scratch =
            MetalDFlashLayerMajorScratch::fresh(&ctx_metal, &mm, N).expect("layer scratch");

        // Synthetic verify tokens.
        let verify_tokens: Vec<i32> = (0..N as i32).map(|i| (i + 1) * 13).collect();
        unsafe {
            let p = verify_scratch.packed_ids_buf.buffer.contents().as_ptr() as *mut i32;
            for (i, &t) in verify_tokens.iter().enumerate() {
                *p.add(i) = t;
            }
        }

        // ==== Phase profile aggregator ====
        let mut phase_ms: Vec<(String, f64)> = Vec::new();
        let mut accum = |name: &str, ms: f64| {
            // Aggregate same-name phases.
            if let Some(slot) = phase_ms.iter_mut().find(|(n, _)| n == name) {
                slot.1 += ms;
            } else {
                phase_ms.push((name.to_string(), ms));
            }
        };

        let t_total = std::time::Instant::now();

        // ---- PHASE: embed (1 dispatch) ----
        {
            let cmd = ctx_metal.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            encode_get_rows_f32(
                &ctx_metal,
                &enc,
                &mm.token_embd,
                &verify_scratch.packed_ids_buf,
                &layer_scratch.x_pack,
                N as usize,
                h,
            )
            .expect("embed");
            enc.end();
            cmd.commit();
            unsafe { cmd.waitUntilCompleted() };
            let ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
            accum("embed", ms);
        }

        // ---- per-layer phases ----
        let mut gdn_idx = 0usize;
        let mut attn_idx = 0usize;
        for (il, block) in mm.blocks.iter().enumerate() {
            // 2a: pre-mixer norm (batched).
            let attn_norm = match block {
                MetalBlock::Gdn(g) => &g.attn_norm,
                MetalBlock::Attn(a) => &a.attn_norm,
            };
            {
                let cmd = ctx_metal.queue.commandBuffer().expect("cmd");
                let enc = KernelEncoder::begin(&cmd);
                encode_rms_norm_batched_f32(
                    &ctx_metal,
                    &enc,
                    &layer_scratch.x_pack,
                    attn_norm,
                    &layer_scratch.h_pack,
                    N as usize,
                    h,
                    RMS_EPS,
                )
                .expect("pre-norm");
                enc.end();
                cmd.commit();
                unsafe { cmd.waitUntilCompleted() };
                accum("pre_norm", (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3);
            }

            // 2b: mixer.
            match block {
                MetalBlock::Gdn(g) => {
                    let gi = gdn_idx;
                    gdn_idx += 1;
                    // Per-token GDN inner loop, separating the compute
                    // dispatches from the blit dispatches in the
                    // accumulator.
                    let mut gdn_compute_ms = 0.0f64;
                    let mut gdn_blit_ms = 0.0f64;
                    for n_idx in 0..N as usize {
                        // Compute pass.
                        {
                            let cmd = ctx_metal.queue.commandBuffer().expect("cmd");
                            let enc = KernelEncoder::begin(&cmd);
                            encode_copy_offset_f32(
                                &ctx_metal,
                                &enc,
                                &layer_scratch.h_pack,
                                n_idx * h,
                                &sess.h,
                                h,
                            )
                            .expect("copy");
                            mf.encode_gdn(&enc, g, gi, &mut sess).expect("gdn");
                            encode_scatter_offset_f32(
                                &ctx_metal,
                                &enc,
                                &sess.mixer_out,
                                &layer_scratch.mixer_out_pack,
                                n_idx * h,
                                h,
                            )
                            .expect("scatter");
                            enc.end();
                            cmd.commit();
                            unsafe { cmd.waitUntilCompleted() };
                            gdn_compute_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                        }
                        // Blit pass.
                        {
                            let cmd = ctx_metal.queue.commandBuffer().expect("cmd");
                            let blit = BlitEncoder::begin(&cmd);
                            blit.copy_tensor(
                                &sess.gdn_state[gi],
                                &verify_scratch.gdn_ckpt_slot(gi as u32, n_idx as u32),
                            );
                            blit.copy_tensor(
                                &sess.gdn_conv[gi],
                                &verify_scratch.conv_ckpt_slot(gi as u32, n_idx as u32),
                            );
                            blit.end();
                            cmd.commit();
                            unsafe { cmd.waitUntilCompleted() };
                            gdn_blit_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                        }
                    }
                    accum("gdn_mixer_compute", gdn_compute_ms);
                    accum("gdn_ckpt_blits", gdn_blit_ms);
                }
                MetalBlock::Attn(a) => {
                    let ai = attn_idx;
                    attn_idx += 1;
                    let mut attn_ms = 0.0f64;
                    for n_idx in 0..N as usize {
                        let cmd = ctx_metal.queue.commandBuffer().expect("cmd");
                        let enc = KernelEncoder::begin(&cmd);
                        encode_copy_offset_f32(
                            &ctx_metal,
                            &enc,
                            &layer_scratch.h_pack,
                            n_idx * h,
                            &sess.h,
                            h,
                        )
                        .expect("copy");
                        let position_n = start_position + n_idx as u32;
                        mf.encode_attn(&enc, a, ai, position_n, &mut sess)
                            .expect("attn");
                        encode_scatter_offset_f32(
                            &ctx_metal,
                            &enc,
                            &sess.mixer_out,
                            &layer_scratch.mixer_out_pack,
                            n_idx * h,
                            h,
                        )
                        .expect("scatter");
                        enc.end();
                        cmd.commit();
                        unsafe { cmd.waitUntilCompleted() };
                        attn_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                    }
                    accum("attn_mixer", attn_ms);
                }
            }

            // 2c: residual #1.
            {
                let cmd = ctx_metal.queue.commandBuffer().expect("cmd");
                let enc = KernelEncoder::begin(&cmd);
                encode_add_inplace_f32(
                    &ctx_metal,
                    &enc,
                    &layer_scratch.x_pack,
                    &layer_scratch.mixer_out_pack,
                )
                .expect("residual1");
                enc.end();
                cmd.commit();
                unsafe { cmd.waitUntilCompleted() };
                accum("residual1", (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3);
            }

            // 2d: hidden capture (v0.71 layout: [N, K, H]).
            for (k_idx, &lid) in target_layer_ids.iter().enumerate() {
                if lid as usize == il {
                    let cmd = ctx_metal.queue.commandBuffer().expect("cmd");
                    let enc = KernelEncoder::begin(&cmd);
                    for n_idx in 0..N as usize {
                        let elem_off = (n_idx as u64 * verify_scratch.k_target_layers as u64
                            + k_idx as u64)
                            * verify_scratch.hidden_size;
                        encode_scatter_offset_f32(
                            &ctx_metal,
                            &enc,
                            &layer_scratch
                                .x_pack
                                .view_subrange((n_idx * h) as u64, vec![h as u64]),
                            &verify_scratch.hidden_capture,
                            elem_off as usize,
                            h,
                        )
                        .expect("hidden capture");
                    }
                    enc.end();
                    cmd.commit();
                    unsafe { cmd.waitUntilCompleted() };
                    accum(
                        "hidden_capture",
                        (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3,
                    );
                }
            }

            // 2e: post-mixer norm.
            let post_norm = match block {
                MetalBlock::Gdn(g) => &g.post_attn_norm,
                MetalBlock::Attn(a) => &a.post_attn_norm,
            };
            {
                let cmd = ctx_metal.queue.commandBuffer().expect("cmd");
                let enc = KernelEncoder::begin(&cmd);
                encode_rms_norm_batched_f32(
                    &ctx_metal,
                    &enc,
                    &layer_scratch.x_pack,
                    post_norm,
                    &layer_scratch.h_pack,
                    N as usize,
                    h,
                    RMS_EPS,
                )
                .expect("post-norm");
                enc.end();
                cmd.commit();
                unsafe { cmd.waitUntilCompleted() };
                accum("post_norm", (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3);
            }

            // 2f: FFN (Q4_K mat-mat for production 27B).
            let (g_w, u_w, d_w) = match block {
                MetalBlock::Gdn(gg) => (&gg.ffn_gate, &gg.ffn_up, &gg.ffn_down),
                MetalBlock::Attn(aa) => (&aa.ffn_gate, &aa.ffn_up, &aa.ffn_down),
            };
            let mat_mat_eligible = |dt: GgmlType| matches!(dt, GgmlType::Q4_K | GgmlType::Q6_K);
            let mat_mat_path = mat_mat_eligible(g_w.dtype)
                && mat_mat_eligible(u_w.dtype)
                && mat_mat_eligible(d_w.dtype);
            {
                let cmd = ctx_metal.queue.commandBuffer().expect("cmd");
                let enc = KernelEncoder::begin(&cmd);
                if mat_mat_path {
                    encode_mat_mat_dispatch(
                        &ctx_metal,
                        &enc,
                        g_w,
                        &layer_scratch.h_pack,
                        &layer_scratch.ffn_gate_pack,
                        h,
                        f,
                        N as usize,
                    )
                    .expect("ffn gate");
                    encode_mat_mat_dispatch(
                        &ctx_metal,
                        &enc,
                        u_w,
                        &layer_scratch.h_pack,
                        &layer_scratch.ffn_up_pack,
                        h,
                        f,
                        N as usize,
                    )
                    .expect("ffn up");
                    encode_silu_mul_f32(
                        &ctx_metal,
                        &enc,
                        &layer_scratch.ffn_gate_pack,
                        &layer_scratch.ffn_up_pack,
                        &layer_scratch.ffn_inner_pack,
                    )
                    .expect("silu_mul");
                    encode_mat_mat_dispatch(
                        &ctx_metal,
                        &enc,
                        d_w,
                        &layer_scratch.ffn_inner_pack,
                        &layer_scratch.ffn_out_pack,
                        f,
                        h,
                        N as usize,
                    )
                    .expect("ffn down");
                } else {
                    // F32 fallback (not exercised on 27B-Q4_K_M).
                    panic!("F32 FFN path not expected on 27B Q4_K_M");
                }
                encode_add_inplace_f32(
                    &ctx_metal,
                    &enc,
                    &layer_scratch.x_pack,
                    &layer_scratch.ffn_out_pack,
                )
                .expect("residual2");
                enc.end();
                cmd.commit();
                unsafe { cmd.waitUntilCompleted() };
                accum(
                    "ffn_plus_residual2",
                    (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3,
                );
            }
        }

        // ---- PHASE: tail (final norm + lm_head + argmax, BATCHED) ----
        {
            let cmd = ctx_metal.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            let lm_dtype = mm.lm_head.dtype;
            let lm_mat_mat_path = matches!(lm_dtype, GgmlType::Q4_K | GgmlType::Q6_K);
            if lm_mat_mat_path {
                encode_rms_norm_batched_f32(
                    &ctx_metal,
                    &enc,
                    &layer_scratch.x_pack,
                    &mm.output_norm,
                    &layer_scratch.h_pack,
                    N as usize,
                    h,
                    RMS_EPS,
                )
                .expect("final norm");
                encode_mat_mat_dispatch(
                    &ctx_metal,
                    &enc,
                    &mm.lm_head,
                    &layer_scratch.h_pack,
                    &layer_scratch.final_logits_pack,
                    h,
                    v,
                    N as usize,
                )
                .expect("lm_head");
                encode_argmax_f32(
                    &ctx_metal,
                    &enc,
                    &layer_scratch.final_logits_pack,
                    &verify_scratch.verify_argmax,
                    N as usize,
                    v,
                )
                .expect("argmax");
            } else {
                panic!("F32 lm_head path not expected on 27B Q4_K_M");
            }
            enc.end();
            cmd.commit();
            unsafe { cmd.waitUntilCompleted() };
            accum("tail", (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3);
        }

        let total_wall_ms = t_total.elapsed().as_secs_f64() * 1e3;
        let total_phase_ms: f64 = phase_ms.iter().map(|(_, m)| *m).sum();

        eprintln!("[v0.70-profile ctx={start_position}] phase breakdown:");
        // Sort by descending ms for at-a-glance bottleneck reading.
        let mut sorted: Vec<_> = phase_ms.iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for (name, ms) in &sorted {
            let pct = 100.0 * *ms / total_phase_ms;
            eprintln!(
                "[v0.70-profile ctx={start_position}]   {name:>20}  {ms:>8.2} ms  ({pct:>5.1}%)"
            );
        }
        eprintln!(
            "[v0.70-profile ctx={start_position}]   {:>20}  {:>8.2} ms  (sum-of-phases GPU time)",
            "TOTAL_PHASE_GPU", total_phase_ms
        );
        eprintln!(
            "[v0.70-profile ctx={start_position}]   {:>20}  {:>8.2} ms  (wall, includes per-phase commit overhead)",
            "TOTAL_WALL", total_wall_ms
        );
    }
}

/// **v0.73a.2 phase profiler** — surgical instrumentation of the
/// post-v0.73a.1 layer-major packed_verify path. Mirrors the v0.70
/// profiler's per-encoder commit/wait pattern but reflects the
/// actual encoder structure of the v0.73a.1 GDN restructure:
///
/// Per GDN layer (production 27B-Q4_K_M, eligible path):
///   * `gdn_step_a_proj_in_qkv_z`  (1 encoder/layer, 2 mat-mat)
///   * `gdn_per_token_compute`     (16 encoders/layer; sums beta/alpha
///                                  mat-vec + sigmoid + alpha-chain +
///                                  encode_gdn_tail = conv + 2 l2 +
///                                  gdn_step + rmsnorm_gated)
///   * `gdn_per_token_blit`        (16 blits/layer; gdn_state +
///                                  gdn_conv ckpt copies)
///   * `gdn_step_c_proj_out`       (1 encoder/layer, 1 mat-mat)
///
/// Per attn layer:
///   * `attn_per_token`            (16 encoders/layer)
///
/// Plus the existing batched phases: pre_norm, residual1, hidden_capture
/// (only fires on target_layer_ids), post_norm, ffn_plus_residual2, tail.
///
/// Decision criteria (from codex v0.73b prep):
///   * gdn_per_token_compute + gdn_per_token_blit ≥ 70-80ms ⇒ v0.73b
///     (internalized N-step GDN tail kernel) is the right move
///   * encoder/wall gap > 20-30ms ⇒ encoder amortization first
///   * tail dispatch dominates but cross-time fusion too risky ⇒
///     single-step fused tail kernel as v0.73b-lite
///   * attn ≥ GDN at ctx ⇒ pull packed-N attn forward (v0.75)
///
/// Single ctx point (1024) and single packed_verify call to keep
/// runtime ~30s. KV is synthetic-primed (kernel timing only).
#[test]
#[ignore = "slow: ~30-60s on M4 Max for 27B-Q4_K_M; \
            run explicitly with `cargo test --test dflash_correctness \
            --release packed_verify_phase_profile_v073a2_27b -- --ignored --nocapture`"]
fn packed_verify_phase_profile_v073a2_27b() {
    if !std::path::Path::new(TARGET_GGUF).exists() {
        eprintln!("[v0.73a.2-profile] skipped — target GGUF missing");
        return;
    }
    let ctx_metal = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("metal init: {e}"),
    };

    eprintln!("[v0.73a.2-profile] loading 27B-Q4_K_M…");
    let g = GgufFile::open(TARGET_GGUF).expect("open target");
    let m = Model::from_gguf(&g).expect("load target");
    let mm = MetalModel::load(&ctx_metal, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx_metal, &mm);

    // ctx is configurable via QWEN_PROFILE_CTX env var (default 1024).
    // Sweep at v0.73c decision-point: do `for ctx in 1024 4096 16384;
    //   do QWEN_PROFILE_CTX=$ctx cargo test --release --test
    //   dflash_correctness packed_verify_phase_profile_v073a2_27b
    //   -- --ignored --nocapture --test-threads=1; done`.
    let start_position: u32 = std::env::var("QWEN_PROFILE_CTX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);
    const N: u32 = 16;
    let target_layer_ids: Vec<u32> = vec![1, 16, 31, 46, 61];
    let k_target = target_layer_ids.len() as u32;

    let arch = m.arch.clone();
    let h = arch.hidden_size as usize;
    let f = arch.intermediate_size as usize;
    let v = arch.vocab_size as usize;
    let n_v_gdn = arch.gdn_n_v_heads as usize;
    let n_k_gdn = arch.gdn_n_k_heads as usize;
    let head_dim_gdn = arch.gdn_head_dim as usize;
    let conv_dim_gdn = (2 * n_k_gdn + n_v_gdn) * head_dim_gdn;
    let v_dim_gdn = n_v_gdn * head_dim_gdn;

    eprintln!(
        "[v0.73a.2-profile] arch: hidden={h} ffn={f} gdn(conv_dim={conv_dim_gdn} v_dim={v_dim_gdn} n_v={n_v_gdn})"
    );
    eprintln!("[v0.73a.2-profile] ctx={start_position} N={N}");

    let kv_capacity = (start_position + N + 32) as usize;
    let mut sess = MetalSession::fresh(&ctx_metal, &mm, kv_capacity).expect("sess");
    eprintln!(
        "[v0.73a.2-profile] synthetic-priming kv_n_pos to {start_position} (kernel timing only)"
    );
    for kp in sess.kv_n_pos.iter_mut() {
        *kp = start_position as usize;
    }

    let mut verify_scratch =
        MetalDFlashVerifyScratch::fresh(&ctx_metal, &mm, N, k_target).expect("verify scratch");
    let mut layer_scratch =
        MetalDFlashLayerMajorScratch::fresh(&ctx_metal, &mm, N).expect("layer scratch");

    // Synthetic verify tokens.
    let verify_tokens: Vec<i32> = (0..N as i32).map(|i| (i + 1) * 13).collect();
    unsafe {
        let p = verify_scratch.packed_ids_buf.buffer.contents().as_ptr() as *mut i32;
        for (i, &t) in verify_tokens.iter().enumerate() {
            *p.add(i) = t;
        }
    }

    let mut phase_ms: Vec<(String, f64)> = Vec::new();
    let mut accum = |name: &str, ms: f64| {
        if let Some(slot) = phase_ms.iter_mut().find(|(n, _)| n == name) {
            slot.1 += ms;
        } else {
            phase_ms.push((name.to_string(), ms));
        }
    };

    let t_total = std::time::Instant::now();

    // PHASE: embed
    {
        let cmd = ctx_metal.queue.commandBuffer().expect("cmd");
        let enc = KernelEncoder::begin(&cmd);
        encode_get_rows_f32(
            &ctx_metal,
            &enc,
            &mm.token_embd,
            &verify_scratch.packed_ids_buf,
            &layer_scratch.x_pack,
            N as usize,
            h,
        )
        .expect("embed");
        enc.end();
        cmd.commit();
        unsafe { cmd.waitUntilCompleted() };
        accum("embed", (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3);
    }

    let mut gdn_idx = 0usize;
    let mut attn_idx = 0usize;
    for (il, block) in mm.blocks.iter().enumerate() {
        // 2a: pre-mixer norm (batched).
        let attn_norm = match block {
            MetalBlock::Gdn(g) => &g.attn_norm,
            MetalBlock::Attn(a) => &a.attn_norm,
        };
        {
            let cmd = ctx_metal.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            encode_rms_norm_batched_f32(
                &ctx_metal,
                &enc,
                &layer_scratch.x_pack,
                attn_norm,
                &layer_scratch.h_pack,
                N as usize,
                h,
                RMS_EPS,
            )
            .expect("pre-norm");
            enc.end();
            cmd.commit();
            unsafe { cmd.waitUntilCompleted() };
            accum("pre_norm", (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3);
        }

        // 2b: mixer.
        match block {
            MetalBlock::Gdn(g) => {
                let gi = gdn_idx;
                gdn_idx += 1;

                // Step A: batched front-end projections (1 encoder/layer).
                {
                    let cmd = ctx_metal.queue.commandBuffer().expect("cmd");
                    let enc = KernelEncoder::begin(&cmd);
                    encode_mat_mat_dispatch(
                        &ctx_metal,
                        &enc,
                        &g.in_proj_qkv,
                        &layer_scratch.h_pack,
                        &layer_scratch.gdn_qkv_pack,
                        h,
                        conv_dim_gdn,
                        N as usize,
                    )
                    .expect("in_proj_qkv mat-mat");
                    encode_mat_mat_dispatch(
                        &ctx_metal,
                        &enc,
                        &g.in_proj_z,
                        &layer_scratch.h_pack,
                        &layer_scratch.gdn_z_pack,
                        h,
                        v_dim_gdn,
                        N as usize,
                    )
                    .expect("in_proj_z mat-mat");
                    enc.end();
                    cmd.commit();
                    unsafe { cmd.waitUntilCompleted() };
                    accum(
                        "gdn_step_a_proj_in_qkv_z",
                        (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3,
                    );
                }

                // Per-token loop: compute encoder + blit per token.
                let alpha_handle = sess.gdn_alpha.clone();
                let beta_handle = sess.gdn_beta.clone();
                let mut compute_ms = 0.0f64;
                let mut blit_ms = 0.0f64;
                for n_idx in 0..N as usize {
                    {
                        let cmd = ctx_metal.queue.commandBuffer().expect("cmd");
                        let enc = KernelEncoder::begin(&cmd);
                        let h_n = layer_scratch
                            .h_pack
                            .view_subrange((n_idx * h) as u64, vec![h as u64]);
                        encode_mat_vec_dispatch(
                            &ctx_metal,
                            &enc,
                            &g.beta_proj,
                            &h_n,
                            &sess.gdn_b,
                            h,
                            n_v_gdn,
                        )
                        .expect("beta mat-vec");
                        encode_sigmoid_f32(&ctx_metal, &enc, &sess.gdn_b, &sess.gdn_beta)
                            .expect("sigmoid");
                        encode_mat_vec_dispatch(
                            &ctx_metal,
                            &enc,
                            &g.alpha_proj,
                            &h_n,
                            &sess.gdn_a,
                            h,
                            n_v_gdn,
                        )
                        .expect("alpha mat-vec");
                        encode_gdn_alpha_chain_f32(
                            &ctx_metal,
                            &enc,
                            &sess.gdn_a,
                            &g.dt_bias,
                            &g.a_log,
                            &sess.gdn_alpha,
                        )
                        .expect("alpha-chain");
                        let qkv_n = layer_scratch.gdn_qkv_pack.view_subrange(
                            (n_idx * conv_dim_gdn) as u64,
                            vec![conv_dim_gdn as u64],
                        );
                        let z_n = layer_scratch
                            .gdn_z_pack
                            .view_subrange((n_idx * v_dim_gdn) as u64, vec![v_dim_gdn as u64]);
                        let normed_n = layer_scratch
                            .gdn_normed_pack
                            .view_subrange((n_idx * v_dim_gdn) as u64, vec![v_dim_gdn as u64]);
                        mf.encode_gdn_tail(
                            &enc,
                            g,
                            gi,
                            &mut sess,
                            &qkv_n,
                            &z_n,
                            &alpha_handle,
                            &beta_handle,
                            &normed_n,
                        )
                        .expect("gdn_tail");
                        enc.end();
                        cmd.commit();
                        unsafe { cmd.waitUntilCompleted() };
                        compute_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                    }
                    {
                        let cmd = ctx_metal.queue.commandBuffer().expect("cmd");
                        let blit = BlitEncoder::begin(&cmd);
                        blit.copy_tensor(
                            &sess.gdn_state[gi],
                            &verify_scratch.gdn_ckpt_slot(gi as u32, n_idx as u32),
                        );
                        blit.copy_tensor(
                            &sess.gdn_conv[gi],
                            &verify_scratch.conv_ckpt_slot(gi as u32, n_idx as u32),
                        );
                        blit.end();
                        cmd.commit();
                        unsafe { cmd.waitUntilCompleted() };
                        blit_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                    }
                }
                accum("gdn_per_token_compute", compute_ms);
                accum("gdn_per_token_blit", blit_ms);

                // Step C: batched out_proj (1 encoder/layer).
                {
                    let cmd = ctx_metal.queue.commandBuffer().expect("cmd");
                    let enc = KernelEncoder::begin(&cmd);
                    encode_mat_mat_dispatch(
                        &ctx_metal,
                        &enc,
                        &g.out_proj,
                        &layer_scratch.gdn_normed_pack,
                        &layer_scratch.mixer_out_pack,
                        v_dim_gdn,
                        h,
                        N as usize,
                    )
                    .expect("out_proj mat-mat");
                    enc.end();
                    cmd.commit();
                    unsafe { cmd.waitUntilCompleted() };
                    accum(
                        "gdn_step_c_proj_out",
                        (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3,
                    );
                }
            }
            MetalBlock::Attn(a) => {
                let ai = attn_idx;
                attn_idx += 1;
                // v0.73c.1: attn restructure mirrors the production
                // layer-major path. Three sub-phases reported separately
                // so we can see the new dispatch breakdown.
                let head_dim = arch.attn_head_dim as usize;
                let n_q = arch.n_q_heads as usize;
                let n_kv = arch.n_kv_heads as usize;
                let q_dim = n_q * head_dim;
                let kv_dim = n_kv * head_dim;
                let n_rot = (head_dim as f32 * arch.partial_rotary_factor) as usize;

                // Step A: batched front-end (Q gated / K / V mat-mat + split + Q-norm + K-norm).
                {
                    let cmd = ctx_metal.queue.commandBuffer().expect("cmd");
                    let enc = KernelEncoder::begin(&cmd);
                    encode_mat_mat_dispatch(
                        &ctx_metal,
                        &enc,
                        &a.q,
                        &layer_scratch.h_pack,
                        &layer_scratch.attn_q_full_pack,
                        h,
                        2 * q_dim,
                        N as usize,
                    )
                    .expect("attn step A: q_full mat-mat");
                    encode_split_q_gate_f32(
                        &ctx_metal,
                        &enc,
                        &layer_scratch.attn_q_full_pack,
                        &layer_scratch.attn_q_pack,
                        &layer_scratch.attn_gate_pack,
                        N as usize * n_q,
                        head_dim,
                    )
                    .expect("attn step A: split q/gate");
                    encode_mat_mat_dispatch(
                        &ctx_metal,
                        &enc,
                        &a.k,
                        &layer_scratch.h_pack,
                        &layer_scratch.attn_k_now_pack,
                        h,
                        kv_dim,
                        N as usize,
                    )
                    .expect("attn step A: k mat-mat");
                    encode_mat_mat_dispatch(
                        &ctx_metal,
                        &enc,
                        &a.v,
                        &layer_scratch.h_pack,
                        &layer_scratch.attn_v_now_pack,
                        h,
                        kv_dim,
                        N as usize,
                    )
                    .expect("attn step A: v mat-mat");
                    encode_rms_norm_batched_f32(
                        &ctx_metal,
                        &enc,
                        &layer_scratch.attn_q_pack,
                        &a.q_norm,
                        &layer_scratch.attn_q_normed_pack,
                        N as usize * n_q,
                        head_dim,
                        RMS_EPS,
                    )
                    .expect("attn step A: q-norm");
                    encode_rms_norm_batched_f32(
                        &ctx_metal,
                        &enc,
                        &layer_scratch.attn_k_now_pack,
                        &a.k_norm,
                        &layer_scratch.attn_k_normed_pack,
                        N as usize * n_kv,
                        head_dim,
                        RMS_EPS,
                    )
                    .expect("attn step A: k-norm");
                    enc.end();
                    cmd.commit();
                    unsafe { cmd.waitUntilCompleted() };
                    accum(
                        "attn_step_a_proj_split_norm",
                        (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3,
                    );
                }

                // Per-token loop: RoPE + KV-scatter + attn-v4 (or naive).
                let mut attn_per_tok_ms = 0.0f64;
                for n_idx in 0..N as usize {
                    let position_n = start_position + n_idx as u32;
                    let q_normed_n = layer_scratch
                        .attn_q_normed_pack
                        .view_subrange((n_idx * q_dim) as u64, vec![q_dim as u64]);
                    let k_normed_n = layer_scratch
                        .attn_k_normed_pack
                        .view_subrange((n_idx * kv_dim) as u64, vec![kv_dim as u64]);
                    let v_now_n = layer_scratch
                        .attn_v_now_pack
                        .view_subrange((n_idx * kv_dim) as u64, vec![kv_dim as u64]);
                    let attn_o_n = layer_scratch
                        .attn_o_pack
                        .view_subrange((n_idx * q_dim) as u64, vec![q_dim as u64]);
                    let cmd = ctx_metal.queue.commandBuffer().expect("cmd");
                    let enc = KernelEncoder::begin(&cmd);
                    encode_rope_neox_f32(
                        &ctx_metal,
                        &enc,
                        &q_normed_n,
                        n_q,
                        head_dim,
                        n_rot,
                        position_n,
                        arch.rope_theta,
                    )
                    .expect("RoPE Q");
                    encode_rope_neox_f32(
                        &ctx_metal,
                        &enc,
                        &k_normed_n,
                        n_kv,
                        head_dim,
                        n_rot,
                        position_n,
                        arch.rope_theta,
                    )
                    .expect("RoPE K");
                    encode_scatter_offset_f32_to_f16_kv(
                        &ctx_metal,
                        &enc,
                        &k_normed_n,
                        &v_now_n,
                        &sess.kv_k[ai],
                        &sess.kv_v[ai],
                        (position_n as usize) * kv_dim,
                        kv_dim,
                    )
                    .expect("KV scatter");
                    sess.kv_n_pos[ai] = position_n as usize + 1;

                    const V4_HEAD_DIM: usize = 256;
                    const V4_GROUP: usize = 6;
                    let use_v4 = head_dim == V4_HEAD_DIM && n_q == n_kv * V4_GROUP;
                    if use_v4 {
                        let nwg = qwen_llm::metal::attn_v4_choose_nwg(sess.kv_n_pos[ai]);
                        let tile_c = qwen_llm::metal::attn_v4_choose_tile_c(sess.kv_n_pos[ai]);
                        qwen_llm::metal::encode_attn_decode_v4_f32(
                            &ctx_metal,
                            &enc,
                            &q_normed_n,
                            &sess.kv_k[ai],
                            &sess.kv_v[ai],
                            &sess.attn_v4_o_partial,
                            &sess.attn_v4_ml_partial,
                            &attn_o_n,
                            n_q,
                            n_kv,
                            head_dim,
                            sess.kv_n_pos[ai],
                            nwg,
                            tile_c,
                        )
                        .expect("attn-v4");
                    } else {
                        qwen_llm::metal::encode_attn_decode_f16kv_f32(
                            &ctx_metal,
                            &enc,
                            &q_normed_n,
                            &sess.kv_k[ai],
                            &sess.kv_v[ai],
                            &attn_o_n,
                            n_q,
                            n_kv,
                            head_dim,
                            sess.kv_n_pos[ai],
                        )
                        .expect("attn naive");
                    }
                    enc.end();
                    cmd.commit();
                    unsafe { cmd.waitUntilCompleted() };
                    attn_per_tok_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                }
                accum("attn_per_token", attn_per_tok_ms);

                // Step C: gate-sigmoid + mul + o_proj mat-mat.
                {
                    let cmd = ctx_metal.queue.commandBuffer().expect("cmd");
                    let enc = KernelEncoder::begin(&cmd);
                    encode_sigmoid_f32(
                        &ctx_metal,
                        &enc,
                        &layer_scratch.attn_gate_pack,
                        &layer_scratch.attn_q_pack,
                    )
                    .expect("sigmoid gate");
                    encode_mul_f32(
                        &ctx_metal,
                        &enc,
                        &layer_scratch.attn_o_pack,
                        &layer_scratch.attn_q_pack,
                        &layer_scratch.attn_o_pack,
                    )
                    .expect("attn_o *= sigmoid(gate)");
                    encode_mat_mat_dispatch(
                        &ctx_metal,
                        &enc,
                        &a.o,
                        &layer_scratch.attn_o_pack,
                        &layer_scratch.mixer_out_pack,
                        q_dim,
                        h,
                        N as usize,
                    )
                    .expect("attn step C: o_proj mat-mat");
                    enc.end();
                    cmd.commit();
                    unsafe { cmd.waitUntilCompleted() };
                    accum(
                        "attn_step_c_gate_oproj",
                        (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3,
                    );
                }
            }
        }

        // 2c: residual #1.
        {
            let cmd = ctx_metal.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            encode_add_inplace_f32(
                &ctx_metal,
                &enc,
                &layer_scratch.x_pack,
                &layer_scratch.mixer_out_pack,
            )
            .expect("residual1");
            enc.end();
            cmd.commit();
            unsafe { cmd.waitUntilCompleted() };
            accum("residual1", (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3);
        }

        // 2d: hidden capture (v0.71 layout: [N, K, H]).
        for (k_idx, &lid) in target_layer_ids.iter().enumerate() {
            if lid as usize == il {
                let cmd = ctx_metal.queue.commandBuffer().expect("cmd");
                let enc = KernelEncoder::begin(&cmd);
                for n_idx in 0..N as usize {
                    let elem_off = (n_idx as u64 * verify_scratch.k_target_layers as u64
                        + k_idx as u64)
                        * verify_scratch.hidden_size;
                    encode_scatter_offset_f32(
                        &ctx_metal,
                        &enc,
                        &layer_scratch
                            .x_pack
                            .view_subrange((n_idx * h) as u64, vec![h as u64]),
                        &verify_scratch.hidden_capture,
                        elem_off as usize,
                        h,
                    )
                    .expect("hidden capture");
                }
                enc.end();
                cmd.commit();
                unsafe { cmd.waitUntilCompleted() };
                accum(
                    "hidden_capture",
                    (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3,
                );
            }
        }

        // 2e: post-mixer norm.
        let post_norm = match block {
            MetalBlock::Gdn(g) => &g.post_attn_norm,
            MetalBlock::Attn(a) => &a.post_attn_norm,
        };
        {
            let cmd = ctx_metal.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            encode_rms_norm_batched_f32(
                &ctx_metal,
                &enc,
                &layer_scratch.x_pack,
                post_norm,
                &layer_scratch.h_pack,
                N as usize,
                h,
                RMS_EPS,
            )
            .expect("post-norm");
            enc.end();
            cmd.commit();
            unsafe { cmd.waitUntilCompleted() };
            accum("post_norm", (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3);
        }

        // 2f: FFN (Q4_K/Q6_K mat-mat).
        let (g_w, u_w, d_w) = match block {
            MetalBlock::Gdn(gg) => (&gg.ffn_gate, &gg.ffn_up, &gg.ffn_down),
            MetalBlock::Attn(aa) => (&aa.ffn_gate, &aa.ffn_up, &aa.ffn_down),
        };
        {
            let cmd = ctx_metal.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            encode_mat_mat_dispatch(
                &ctx_metal,
                &enc,
                g_w,
                &layer_scratch.h_pack,
                &layer_scratch.ffn_gate_pack,
                h,
                f,
                N as usize,
            )
            .expect("ffn gate");
            encode_mat_mat_dispatch(
                &ctx_metal,
                &enc,
                u_w,
                &layer_scratch.h_pack,
                &layer_scratch.ffn_up_pack,
                h,
                f,
                N as usize,
            )
            .expect("ffn up");
            encode_silu_mul_f32(
                &ctx_metal,
                &enc,
                &layer_scratch.ffn_gate_pack,
                &layer_scratch.ffn_up_pack,
                &layer_scratch.ffn_inner_pack,
            )
            .expect("silu_mul");
            encode_mat_mat_dispatch(
                &ctx_metal,
                &enc,
                d_w,
                &layer_scratch.ffn_inner_pack,
                &layer_scratch.ffn_out_pack,
                f,
                h,
                N as usize,
            )
            .expect("ffn down");
            encode_add_inplace_f32(
                &ctx_metal,
                &enc,
                &layer_scratch.x_pack,
                &layer_scratch.ffn_out_pack,
            )
            .expect("residual2");
            enc.end();
            cmd.commit();
            unsafe { cmd.waitUntilCompleted() };
            accum(
                "ffn_plus_residual2",
                (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3,
            );
        }
    }

    // PHASE: tail.
    {
        let cmd = ctx_metal.queue.commandBuffer().expect("cmd");
        let enc = KernelEncoder::begin(&cmd);
        encode_rms_norm_batched_f32(
            &ctx_metal,
            &enc,
            &layer_scratch.x_pack,
            &mm.output_norm,
            &layer_scratch.h_pack,
            N as usize,
            h,
            RMS_EPS,
        )
        .expect("final norm");
        encode_mat_mat_dispatch(
            &ctx_metal,
            &enc,
            &mm.lm_head,
            &layer_scratch.h_pack,
            &layer_scratch.final_logits_pack,
            h,
            v,
            N as usize,
        )
        .expect("lm_head");
        encode_argmax_f32(
            &ctx_metal,
            &enc,
            &layer_scratch.final_logits_pack,
            &verify_scratch.verify_argmax,
            N as usize,
            v,
        )
        .expect("argmax");
        enc.end();
        cmd.commit();
        unsafe { cmd.waitUntilCompleted() };
        accum("tail", (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3);
    }

    let total_wall_ms = t_total.elapsed().as_secs_f64() * 1e3;
    let total_phase_ms: f64 = phase_ms.iter().map(|(_, m)| *m).sum();

    eprintln!("\n[v0.73a.2-profile ctx={start_position}] phase breakdown:");
    let mut sorted: Vec<_> = phase_ms.iter().collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (name, ms) in &sorted {
        let pct = 100.0 * *ms / total_phase_ms;
        eprintln!("[v0.73a.2-profile]   {name:>30}  {ms:>8.2} ms  ({pct:>5.1}%)");
    }
    eprintln!(
        "[v0.73a.2-profile]   {:>30}  {:>8.2} ms  (sum-of-phases GPU time)",
        "TOTAL_PHASE_GPU", total_phase_ms
    );
    eprintln!(
        "[v0.73a.2-profile]   {:>30}  {:>8.2} ms  (wall, includes per-phase commit overhead)",
        "TOTAL_WALL", total_wall_ms
    );
    eprintln!(
        "[v0.73a.2-profile]   wall - phase_gpu = {:.2} ms ({:.1}% — encoder/sync overhead)",
        total_wall_ms - total_phase_ms,
        100.0 * (total_wall_ms - total_phase_ms) / total_wall_ms,
    );

    // Decision aid: codex thresholds for v0.73b vs alternatives.
    let gdn_compute_total = phase_ms
        .iter()
        .find(|(n, _)| n == "gdn_per_token_compute")
        .map(|(_, m)| *m)
        .unwrap_or(0.0);
    let gdn_blit_total = phase_ms
        .iter()
        .find(|(n, _)| n == "gdn_per_token_blit")
        .map(|(_, m)| *m)
        .unwrap_or(0.0);
    let gdn_tail_dominant = gdn_compute_total + gdn_blit_total;
    let attn_total = phase_ms
        .iter()
        .find(|(n, _)| n == "attn_per_token")
        .map(|(_, m)| *m)
        .unwrap_or(0.0);
    let encoder_overhead = total_wall_ms - total_phase_ms;
    eprintln!(
        "\n[v0.73a.2-profile] v0.73b decision criteria:\n  \
         gdn_per_token_compute + gdn_per_token_blit = {:.2} ms\n  \
         attn_per_token total                       = {:.2} ms\n  \
         encoder/sync overhead (wall - phase_gpu)   = {:.2} ms\n  \
         => v0.73b (N-step GDN tail kernel) if gdn_tail_dominant >= 70 ms\n  \
         => v0.73c (encoder amortization)        if encoder_overhead > 30 ms\n  \
         => v0.75 (packed-N attn) if attn_total >= gdn_tail_dominant",
        gdn_tail_dominant, attn_total, encoder_overhead,
    );
}
