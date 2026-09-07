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
use qwen_llm::loader::{Model, open_dflash_drafter};
use qwen_llm::metal::{
    BlitEncoder, KernelEncoder, MetalContext, MetalError, MetalTensor, encode_add_inplace_f32,
    encode_argmax_f32, encode_copy_offset_f32, encode_gdn_alpha_chain_f32, encode_get_rows_f32,
    encode_mat_mat_mma8_dispatch, encode_mat_mat_mma8_variant, encode_mat_vec_nc_dispatch,
    encode_mat_vec_q4_k_nc2_rp4_f32, encode_mul_f32, encode_rms_norm_batched_f32,
    encode_rope_neox_f32, encode_scatter_offset_f32_to_f16_kv, encode_sigmoid_f32,
    encode_silu_mul_f32, encode_split_q_gate_f32,
};
use qwen_llm::metal_dflash::{
    DFlashDecoder, MetalDFlashDebugScratch, MetalDFlashHead, MetalDFlashLayerMajorScratch,
    MetalDFlashSession, MetalDFlashVerifyScratch, prefill_tokens_with_multi_hidden,
};
use qwen_llm::metal_forward::{
    MetalBlock, MetalForward, MetalModel, MetalSession, RMS_EPS, encode_mat_mat_dispatch,
    encode_mat_vec_dispatch, encode_scatter_offset_f32,
};
use qwen_llm::tensor::GgmlType;
use qwen_llm::tokenizer::Tokenizer;

const TARGET_GGUF: &str = qwen_llm::test_fixtures::QWEN36_27B_Q4_K_M.path();
const DRAFTER_GGUF: &str = qwen_llm::test_fixtures::DFLASH_DRAFT_36_Q8_0.path();

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
        None, // n_eff_override (test always uses full N)
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

    let arch = m.arch;
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

        let verify_scratch =
            MetalDFlashVerifyScratch::fresh(&ctx_metal, &mm, N, k_target).expect("verify scratch");
        let layer_scratch =
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
            cmd.waitUntilCompleted();
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
                cmd.waitUntilCompleted();
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
                            cmd.waitUntilCompleted();
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
                            cmd.waitUntilCompleted();
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
                        cmd.waitUntilCompleted();
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
                cmd.waitUntilCompleted();
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
                    cmd.waitUntilCompleted();
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
                cmd.waitUntilCompleted();
                accum("post_norm", (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3);
            }

            // 2f: FFN (Q4_K mat-mat for production 27B).
            let (g_w, u_w, d_w) = match block {
                MetalBlock::Gdn(gg) => (&gg.ffn_gate, &gg.ffn_up, &gg.ffn_down),
                MetalBlock::Attn(aa) => (&aa.ffn_gate, &aa.ffn_up, &aa.ffn_down),
            };
            let mat_mat_eligible = |dt: GgmlType| {
                matches!(
                    dt,
                    GgmlType::F32
                        | GgmlType::F16
                        | GgmlType::BF16
                        | GgmlType::Q2_K
                        | GgmlType::Q3_K
                        | GgmlType::Q4_0
                        | GgmlType::Q4_1
                        | GgmlType::Q4_K
                        | GgmlType::Q5_K
                        | GgmlType::Q6_K
                        | GgmlType::Q8_0
                        | GgmlType::IQ4_NL
                        | GgmlType::IQ4_XS
                )
            };
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
                cmd.waitUntilCompleted();
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
            let lm_mat_mat_path = matches!(
                lm_dtype,
                GgmlType::F32
                    | GgmlType::F16
                    | GgmlType::BF16
                    | GgmlType::Q2_K
                    | GgmlType::Q3_K
                    | GgmlType::Q4_0
                    | GgmlType::Q4_1
                    | GgmlType::Q4_K
                    | GgmlType::Q5_K
                    | GgmlType::Q6_K
                    | GgmlType::Q8_0
                    | GgmlType::IQ4_NL
                    | GgmlType::IQ4_XS
            );
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
            cmd.waitUntilCompleted();
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

    let arch = m.arch;
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

    let verify_scratch =
        MetalDFlashVerifyScratch::fresh(&ctx_metal, &mm, N, k_target).expect("verify scratch");
    let layer_scratch =
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
        cmd.waitUntilCompleted();
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
            cmd.waitUntilCompleted();
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
                    cmd.waitUntilCompleted();
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
                        cmd.waitUntilCompleted();
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
                        cmd.waitUntilCompleted();
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
                    cmd.waitUntilCompleted();
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
                    cmd.waitUntilCompleted();
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
                    let group = n_q / n_kv;
                    let use_v4 = head_dim == V4_HEAD_DIM && matches!(group, 4 | 6 | 8 | 16);
                    if use_v4 {
                        let nwg = qwen_llm::metal::attn_v4_choose_nwg(sess.kv_n_pos[ai], group);
                        let tile_c =
                            qwen_llm::metal::attn_v4_choose_tile_c(sess.kv_n_pos[ai], group);
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
                    cmd.waitUntilCompleted();
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
                    cmd.waitUntilCompleted();
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
            cmd.waitUntilCompleted();
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
                cmd.waitUntilCompleted();
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
            cmd.waitUntilCompleted();
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
            cmd.waitUntilCompleted();
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
        cmd.waitUntilCompleted();
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

/// **H5.6 M1b: pipelined packed-verify cost** — times the PRODUCTION
/// `encode_packed_verify_layer_major_inner` (one command buffer, one
/// commit+wait internally) as-is, versus a same-session `single_token`
/// decode step. Defaults to ctx {570, 1024, 4096} at N=16; long-context
/// verifier work can override `QWEN_DFLASH_VERIFY_AUDIT_MODEL`,
/// `QWEN_DFLASH_VERIFY_AUDIT_CTXS`, `QWEN_DFLASH_VERIFY_AUDIT_N`, and
/// `QWEN_DFLASH_VERIFY_AUDIT_REPS`.
///
/// This is the truth-source the per-phase profile above cannot give:
/// `packed_verify_phase_profile_v073a2_27b` commits a command buffer per
/// phase per layer, which both inflates wall (~2.9x) and distorts the GPU
/// pipelining. The verify:decode-step ratio here is the number the DFlash
/// step economics stand on (break-even at mean_emitted/step, roadmap item 6
/// bar: a removable villain must credibly clear >=1.25x decode).
///
/// KV positions are re-primed to the same ctx before every call (v0.440
/// lesson: immutable seed per rep — verify/decode otherwise advance
/// kv_n_pos and drift the shape).
///
/// `cargo test -p qwen-llm --release --test dflash_correctness \
///   packed_verify_pipelined_cost_27b -- --ignored --nocapture`
#[test]
#[ignore = "slow: loads 27B; run explicitly (see doc comment)"]
fn packed_verify_pipelined_cost_27b() {
    let target_path = std::env::var_os("QWEN_DFLASH_VERIFY_AUDIT_MODEL")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(TARGET_GGUF));
    if !target_path.exists() {
        eprintln!(
            "[h5.6-m1b] skipped — target GGUF missing: {}",
            target_path.display()
        );
        return;
    }
    let ctx_metal = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("metal init: {e}"),
    };

    eprintln!("[h5.6-m1b] loading {}", target_path.display());
    let g = GgufFile::open(&target_path).expect("open target");
    let m = Model::from_gguf(&g).expect("load target");
    let mm = MetalModel::load(&ctx_metal, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx_metal, &mm);

    let n: u32 = std::env::var("QWEN_DFLASH_VERIFY_AUDIT_N")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(16);
    assert!((1..=16).contains(&n));
    let target_layer_ids: Vec<u32> = vec![1, 16, 31, 46, 61];
    let k_target = target_layer_ids.len() as u32;
    let ctx_points: Vec<u32> = std::env::var("QWEN_DFLASH_VERIFY_AUDIT_CTXS")
        .ok()
        .map(|value| {
            value
                .split(',')
                .map(|part| part.trim().parse().expect("invalid verify audit context"))
                .collect()
        })
        .unwrap_or_else(|| vec![570, 1024, 4096]);
    assert!(!ctx_points.is_empty());
    let kv_capacity = (*ctx_points.iter().max().unwrap() + n + 32) as usize;

    let mut sess = MetalSession::fresh(&ctx_metal, &mm, kv_capacity).expect("sess");
    let mut verify_scratch =
        MetalDFlashVerifyScratch::fresh(&ctx_metal, &mm, n, k_target).expect("verify scratch");
    let mut layer_scratch =
        MetalDFlashLayerMajorScratch::fresh(&ctx_metal, &mm, n).expect("layer scratch");

    let verify_tokens: Vec<i32> = (0..n as i32).map(|i| (i + 1) * 13).collect();
    let reps: usize = std::env::var("QWEN_DFLASH_VERIFY_AUDIT_REPS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(8);
    assert!(reps > 0);

    for ctx_pos in ctx_points {
        let reprime = |sess: &mut MetalSession| {
            for kp in sess.kv_n_pos.iter_mut() {
                *kp = ctx_pos as usize;
            }
        };

        // Verify: warmup + REPS timed.
        reprime(&mut sess);
        qwen_llm::metal_dflash::encode_packed_verify_layer_major_inner(
            &mf,
            &target_layer_ids,
            &verify_tokens,
            ctx_pos,
            &mut verify_scratch,
            &mut layer_scratch,
            &mut sess,
            None,
            None,
        )
        .expect("verify warmup");
        let mut verify_ms = Vec::with_capacity(reps);
        for _ in 0..reps {
            reprime(&mut sess);
            let t = std::time::Instant::now();
            qwen_llm::metal_dflash::encode_packed_verify_layer_major_inner(
                &mf,
                &target_layer_ids,
                &verify_tokens,
                ctx_pos,
                &mut verify_scratch,
                &mut layer_scratch,
                &mut sess,
                None,
                None,
            )
            .expect("verify");
            verify_ms.push(t.elapsed().as_secs_f64() * 1e3);
        }

        // Single-token decode step: warmup + REPS timed.
        reprime(&mut sess);
        mf.single_token(1234, ctx_pos, &mut sess)
            .expect("decode warmup");
        let mut decode_ms = Vec::with_capacity(reps);
        for _ in 0..reps {
            reprime(&mut sess);
            let t = std::time::Instant::now();
            mf.single_token(1234, ctx_pos, &mut sess).expect("decode");
            decode_ms.push(t.elapsed().as_secs_f64() * 1e3);
        }

        let stats = |v: &[f64]| {
            let min = v.iter().cloned().fold(f64::INFINITY, f64::min);
            let mean = v.iter().sum::<f64>() / v.len() as f64;
            (min, mean)
        };
        let (v_min, v_mean) = stats(&verify_ms);
        let (d_min, d_mean) = stats(&decode_ms);
        eprintln!(
            "[h5.6-m1b ctx={ctx_pos:>6}] verify{n} min={v_min:7.2} mean={v_mean:7.2} ms | \
             decode1 min={d_min:6.2} mean={d_mean:6.2} ms | \
             ratio(min)={:.2}x  step-budget@4.267={:.0} ms",
            v_min / d_min,
            d_min * 4.267
        );
    }
}

/// **H5.6 M1a: skinny-N GEMM attribution** — times every packed-verify
/// projection shape (27B real weights) through `encode_mat_mat_dispatch`
/// at N in {2,4,8,16,32} against the N=1 `encode_mat_vec_dispatch` stream
/// reference. Reports per-call ms and effective weight-stream GB/s.
///
/// The M1b pipelined measurement shows verify16 = 5.2-5.5x a decode step
/// where theory says ~1.2-1.5x; the per-phase profile blames FFN (47%) and
/// GDN projections (~18%). This test pins WHICH shapes are off stream rate
/// and by how much, so the M2 kernel retune targets the right one.
///
/// `cargo test -p qwen-llm --release --test dflash_correctness \
///   packed_verify_skinny_gemm_micro_27b -- --ignored --nocapture`
#[test]
#[ignore = "slow: loads 27B; run explicitly (see doc comment)"]
fn packed_verify_skinny_gemm_micro_27b() {
    if !std::path::Path::new(TARGET_GGUF).exists() {
        eprintln!("[h5.6-m1a] skipped — target GGUF missing");
        return;
    }
    let ctx_metal = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("metal init: {e}"),
    };

    eprintln!("[h5.6-m1a] loading 27B-Q4_K_M…");
    let g = GgufFile::open(TARGET_GGUF).expect("open target");
    let m = Model::from_gguf(&g).expect("load target");
    let mm = MetalModel::load(&ctx_metal, &g, &m).expect("metal load");

    let h = mm.arch.hidden_size as usize;

    // One GDN block and one attn block donate their real weights.
    let gdn = mm
        .blocks
        .iter()
        .find_map(|b| match b {
            MetalBlock::Gdn(g) => Some(g),
            _ => None,
        })
        .expect("gdn block");
    let attn = mm
        .blocks
        .iter()
        .find_map(|b| match b {
            MetalBlock::Attn(a) => Some(a),
            _ => None,
        })
        .expect("attn block");

    // (label, weight, n_in); n_out derived from element count.
    let f = mm.arch.intermediate_size as usize;
    let shapes: Vec<(&str, &MetalTensor, usize)> = vec![
        ("ffn_gate  Q4K", &gdn.ffn_gate, h),
        ("ffn_down  Q6K", &gdn.ffn_down, f),
        ("gdn_qkv   ", &gdn.in_proj_qkv, h),
        ("gdn_z     ", &gdn.in_proj_z, h),
        ("gdn_out   ", &gdn.out_proj, {
            let n_out_elems = gdn.out_proj.n_elements() as usize;
            n_out_elems / h // out_proj: [v_dim -> h]; n_in = v_dim
        }),
        ("attn_q    ", &attn.q, h),
        ("attn_o    ", &attn.o, {
            let n_out_elems = attn.o.n_elements() as usize;
            n_out_elems / h // o: [q_dim -> h]; n_in = q_dim
        }),
    ];

    let timed = |encode: &dyn Fn(&KernelEncoder)| -> f64 {
        const ITERS: usize = 16;
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            let cmd = ctx_metal.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            for _ in 0..ITERS {
                encode(&enc);
            }
            enc.end();
            let t = std::time::Instant::now();
            cmd.commit();
            cmd.waitUntilCompleted();
            best = best.min(t.elapsed().as_secs_f64() / ITERS as f64);
        }
        best
    };

    for (label, w, n_in) in shapes {
        let n_out = w.n_elements() as usize / n_in;
        let w_bytes = w.buffer.length() as f64;
        let x_src: Vec<f32> = (0..32 * n_in)
            .map(|i| ((i % 13) as f32 - 6.0) * 1e-2)
            .collect();
        let x = MetalTensor::from_bytes(
            &ctx_metal,
            bytemuck::cast_slice(&x_src),
            vec![(32 * n_in) as u64],
            GgmlType::F32,
        )
        .expect("x");
        let y = MetalTensor::zeros_f32(&ctx_metal, vec![(32 * n_out) as u64]).expect("y");

        // N=1 mat-vec stream reference.
        let x1 = x.view_subrange(0, vec![n_in as u64]);
        let y1 = y.view_subrange(0, vec![n_out as u64]);
        let t_mv = timed(&|enc| {
            encode_mat_vec_dispatch(&ctx_metal, enc, w, &x1, &y1, n_in, n_out).expect("mat_vec");
        });
        let gbps_mv = w_bytes / t_mv / 1e9;
        eprint!(
            "[h5.6-m1a {label}] {n_in:>5}->{n_out:>5}  mv1 {:7.3} ms {gbps_mv:5.0} GB/s |",
            t_mv * 1e3
        );

        for &n_q in &[2usize, 4, 8, 16, 32] {
            let xn = x.view_subrange(0, vec![(n_q * n_in) as u64]);
            let yn = y.view_subrange(0, vec![(n_q * n_out) as u64]);
            let t_mm = timed(&|enc| {
                encode_mat_mat_dispatch(&ctx_metal, enc, w, &xn, &yn, n_in, n_out, n_q)
                    .expect("mat_mat");
            });
            let gbps = w_bytes / t_mm / 1e9;
            eprint!(" N{n_q}:{:6.3}ms/{gbps:4.0}", t_mm * 1e3);
        }
        eprintln!();
    }
}

/// **H5.6 M2-nc: multi-column GEMV experiment.** The M1a micro pins the
/// mat-mat tile family at 56-128 GB/s on skinny verify shapes with
/// N2..N32 costing identical wall time (per-tile floor + under-occupancy:
/// `n_out/64` threadgroups). This experiment keeps the mat-vec dispatch
/// geometry (`n_out/4` threadgroups, 283-436 GB/s at N=1 on these same
/// shapes) and amortizes the weight stream across NC in {2,4,8} activation
/// columns with register-staged quants (kernels/mat_vec_q4_k_nc.metal).
///
/// Gates:
///   * correctness (asserted): bit-exact per column vs `mv1` on every
///     Q4_K verify shape, real 27B weights.
///   * perf (pre-registered, reported): c(4) = t_nc4 / t_mv1 <= 1.4 on the
///     FFN gate shape reopens MTP-N / packed-verify economics
///     (v0.443/v0.444 reopen arm (c)); c(4) >= ~3 confirms the v0.444
///     ALU-cap for GEMV-shaped designs and closes the arm.
///
/// `cargo test -p qwen-llm --release --test dflash_correctness \
///   multicol_gemv_micro_27b -- --ignored --nocapture`
#[test]
#[ignore = "slow: loads 27B; run explicitly (see doc comment)"]
fn multicol_gemv_micro_27b() {
    if !std::path::Path::new(TARGET_GGUF).exists() {
        eprintln!("[h5.6-m2nc] skipped — target GGUF missing");
        return;
    }
    let ctx_metal = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("metal init: {e}"),
    };

    eprintln!("[h5.6-m2nc] loading 27B-Q4_K_M…");
    let g = GgufFile::open(TARGET_GGUF).expect("open target");
    let m = Model::from_gguf(&g).expect("load target");
    let mm = MetalModel::load(&ctx_metal, &g, &m).expect("metal load");

    let h = mm.arch.hidden_size as usize;
    let gdn = mm
        .blocks
        .iter()
        .find_map(|b| match b {
            MetalBlock::Gdn(g) => Some(g),
            _ => None,
        })
        .expect("gdn block");
    let attn = mm
        .blocks
        .iter()
        .find_map(|b| match b {
            MetalBlock::Attn(a) => Some(a),
            _ => None,
        })
        .expect("attn block");

    let f = mm.arch.intermediate_size as usize;
    let shapes: Vec<(&str, &MetalTensor, usize)> = vec![
        ("ffn_gate ", &gdn.ffn_gate, h),
        ("ffn_down ", &gdn.ffn_down, f),
        ("gdn_qkv  ", &gdn.in_proj_qkv, h),
        ("gdn_z    ", &gdn.in_proj_z, h),
        ("gdn_out  ", &gdn.out_proj, {
            gdn.out_proj.n_elements() as usize / h
        }),
        ("attn_q   ", &attn.q, h),
        ("attn_o   ", &attn.o, { attn.o.n_elements() as usize / h }),
    ];

    // best-of-5 x 32 iters: the mv1 reference is ~0.04-0.18 ms/dispatch and
    // showed +/-20% run-to-run at 3x16; the nc kernels were stable. Deeper
    // sampling keeps the c(N) ratios honest for the record.
    let timed = |encode: &dyn Fn(&KernelEncoder)| -> f64 {
        const ITERS: usize = 32;
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let cmd = ctx_metal.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            for _ in 0..ITERS {
                encode(&enc);
            }
            enc.end();
            let t = std::time::Instant::now();
            cmd.commit();
            cmd.waitUntilCompleted();
            best = best.min(t.elapsed().as_secs_f64() / ITERS as f64);
        }
        best
    };

    const NC_MAX: usize = 8;
    let mut bar_c4_ffn_gate: Option<f64> = None;

    let mut c2_summary: Vec<(String, f64)> = Vec::new();

    for (label, w, n_in) in shapes {
        if !matches!(w.dtype, GgmlType::Q4_K | GgmlType::Q6_K) {
            eprintln!(
                "[h5.6-m2nc {label}] skipped — dtype {:?} not in {{Q4_K, Q6_K}}",
                w.dtype
            );
            continue;
        }
        let n_out = w.n_elements() as usize / n_in;
        let w_bytes = w.buffer.length() as f64;

        let x_src: Vec<f32> = (0..NC_MAX * n_in)
            .map(|i| ((i % 13) as f32 - 6.0) * 1e-2)
            .collect();
        let x = MetalTensor::from_bytes(
            &ctx_metal,
            bytemuck::cast_slice(&x_src),
            vec![(NC_MAX * n_in) as u64],
            GgmlType::F32,
        )
        .expect("x");
        let y_nc = MetalTensor::zeros_f32(&ctx_metal, vec![(NC_MAX * n_out) as u64]).expect("y_nc");
        let y_ref =
            MetalTensor::zeros_f32(&ctx_metal, vec![(NC_MAX * n_out) as u64]).expect("y_ref");

        // --- Correctness: bit-exact per column vs mv1, per NC. ---
        for &nc in &[2usize, 4, 8] {
            let cmd = ctx_metal.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            let xn = x.view_subrange(0, vec![(nc * n_in) as u64]);
            let yn = y_nc.view_subrange(0, vec![(nc * n_out) as u64]);
            encode_mat_vec_nc_dispatch(&ctx_metal, &enc, w, &xn, &yn, n_in, n_out, nc)
                .expect("nc kernel");
            for c in 0..nc {
                let xc = x.view_subrange((c * n_in) as u64, vec![n_in as u64]);
                let yc = y_ref.view_subrange((c * n_out) as u64, vec![n_out as u64]);
                encode_mat_vec_dispatch(&ctx_metal, &enc, w, &xc, &yc, n_in, n_out)
                    .expect("mv1 ref");
            }
            enc.end();
            cmd.commit();
            cmd.waitUntilCompleted();

            let got = unsafe {
                std::slice::from_raw_parts(
                    y_nc.buffer.contents().as_ptr() as *const f32,
                    nc * n_out,
                )
            };
            let want = unsafe {
                std::slice::from_raw_parts(
                    y_ref.buffer.contents().as_ptr() as *const f32,
                    nc * n_out,
                )
            };
            // Exactness gate: E0 — the shipped nc body is
            // expression-identical to mv1 per column and MUST stay
            // bit-exact (the v0.498 convert-hoisted variant broke this
            // via fast-math reassociation and was reverted; this assert
            // is the tripwire). Cosine/rel_rms/max_abs are reported as
            // diagnostics for future variant work.
            let mut bit_mismatches = 0usize;
            let mut dot = 0f64;
            let mut n2_got = 0f64;
            let mut n2_want = 0f64;
            let mut err2 = 0f64;
            let mut max_abs = 0f64;
            for i in 0..nc * n_out {
                if got[i].to_bits() != want[i].to_bits() {
                    bit_mismatches += 1;
                }
                let (g, w) = (got[i] as f64, want[i] as f64);
                dot += g * w;
                n2_got += g * g;
                n2_want += w * w;
                let e = (g - w).abs();
                err2 += e * e;
                if e > max_abs {
                    max_abs = e;
                }
            }
            let cos = dot / (n2_got.sqrt() * n2_want.sqrt()).max(1e-30);
            let rel_rms = (err2 / n2_want.max(1e-30)).sqrt();
            eprintln!(
                "[h5.6-m2nc {label}] NC{nc}: bit-mismatches {bit_mismatches}/{} cos {cos:.9} rel_rms {rel_rms:.2e} max_abs {max_abs:.2e}",
                nc * n_out
            );
            assert_eq!(
                bit_mismatches, 0,
                "[h5.6-m2nc {label}] NC{nc}: {bit_mismatches} bit-mismatches vs per-column mv1 (E0 gate; cos {cos:.9} rel_rms {rel_rms:.2e})"
            );
        }

        // --- Bench: mv1 stream reference, then nc{2,4,8}. ---
        let x1 = x.view_subrange(0, vec![n_in as u64]);
        let y1 = y_ref.view_subrange(0, vec![n_out as u64]);
        let t_mv = timed(&|enc| {
            encode_mat_vec_dispatch(&ctx_metal, enc, w, &x1, &y1, n_in, n_out).expect("mat_vec");
        });
        let gbps_mv = w_bytes / t_mv / 1e9;
        eprint!(
            "[h5.6-m2nc {label}] {n_in:>5}->{n_out:>5}  mv1 {:7.3} ms {gbps_mv:5.0} GB/s |",
            t_mv * 1e3
        );
        for &nc in &[2usize, 4, 8] {
            let xn = x.view_subrange(0, vec![(nc * n_in) as u64]);
            let yn = y_nc.view_subrange(0, vec![(nc * n_out) as u64]);
            let t_nc = timed(&|enc| {
                encode_mat_vec_nc_dispatch(&ctx_metal, enc, w, &xn, &yn, n_in, n_out, nc)
                    .expect("nc kernel");
            });
            let gbps = w_bytes / t_nc / 1e9;
            let ratio = t_nc / t_mv;
            eprint!(" nc{nc}:{:6.3}ms/{gbps:4.0} c={ratio:4.2}", t_nc * 1e3);
            if nc == 4 && label.trim() == "ffn_gate" {
                bar_c4_ffn_gate = Some(ratio);
            }
            if nc == 2 {
                c2_summary.push((label.trim().to_string(), ratio));
            }
        }
        eprintln!();
    }

    let c4 = bar_c4_ffn_gate
        .expect("pre-registered bar shape (ffn_gate, Q4_K) was not measured — model/dtype drift?");
    let verdict = if c4 <= 1.4 {
        "PASS — reopen bar met"
    } else {
        "FAIL — under bar"
    };
    eprintln!("[h5.6-m2nc] pre-registered bar: c(4) on ffn_gate = {c4:.2} (<= 1.40) → {verdict}");
    // Shallow-chain (verify-2) economics read: c(2) per shape. Not a
    // pre-registered gate — reported for the MTP-1/2 pricing model.
    let c2s: Vec<String> = c2_summary
        .iter()
        .map(|(l, r)| format!("{l}={r:.2}"))
        .collect();
    eprintln!("[h5.6-m2nc] c(2) by shape: {}", c2s.join(" "));
}

/// **v0.498 follow-up: small-N MMA falsifier** (`smalln_mma_micro_27b`).
/// The recorded open lane after the M2-nc scalar cap replication:
/// simdgroup_matrix at mat-vec-grade occupancy (8-row x 8-padded-column
/// tiles, `n_out/8` single-SG threadgroups — 2176 at ffn_gate vs the
/// incumbent 64x32 tile's 272). On M4 there is no separate MMA pool; the
/// candidate win is dequant-once-per-weight + dense FMA encoding +
/// occupancy.
///
/// Pre-registered reads (cx design jam `019f38c6-f...`):
///   * PASS: c(2) = t_mma8/t_mv1 <= 1.25 on BOTH ffn_gate (Q4_K) and
///     ffn_down (Q6_K) with cos >= 0.999 per column → lane stays open,
///     next step whole-step integration.
///   * mechanism-real-but-under-kill: beats scalar nc2 (c < ~1.6) but
///     misses 1.25 → record and close unless whole-step composition
///     changes the math.
///   * diagnosis-falsified: fails to beat scalar nc2 → tiny-tile
///     overhead / shared-FP32-pipe pressure was the binding term, not
///     incumbent under-occupancy.
///
/// The kernel always computes 8 columns (padding is the caller's job);
/// correctness checks ALL 8 columns (distinct nonzero patterns — catches
/// transpose/stride bugs) at cos >= 0.999 vs per-column mv1 (E1: half
/// weight staging + MMA accumulation order; activations stay F32).
///
/// This is a FALSIFIER HARNESS, not a regression gate: the kill-line
/// verdict is REPORTED for the registry (a closed lane would otherwise
/// fail forever). Asserted invariants: per-column correctness, and that
/// both kill-line shapes were actually measured.
///
/// `cargo test -p qwen-llm --release --test dflash_correctness \
///   smalln_mma_micro_27b -- --ignored --nocapture`
#[test]
#[ignore = "slow: loads 27B; run explicitly (see doc comment)"]
fn smalln_mma_micro_27b() {
    if !std::path::Path::new(TARGET_GGUF).exists() {
        eprintln!("[h5.6-mma8] skipped — target GGUF missing");
        return;
    }
    let ctx_metal = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("metal init: {e}"),
    };

    eprintln!("[h5.6-mma8] loading 27B-Q4_K_M…");
    let g = GgufFile::open(TARGET_GGUF).expect("open target");
    let m = Model::from_gguf(&g).expect("load target");
    let mm = MetalModel::load(&ctx_metal, &g, &m).expect("metal load");

    let h = mm.arch.hidden_size as usize;
    let f = mm.arch.intermediate_size as usize;
    let gdn = mm
        .blocks
        .iter()
        .find_map(|b| match b {
            MetalBlock::Gdn(g) => Some(g),
            _ => None,
        })
        .expect("gdn block");
    let attn = mm
        .blocks
        .iter()
        .find_map(|b| match b {
            MetalBlock::Attn(a) => Some(a),
            _ => None,
        })
        .expect("attn block");

    // (label, weight, n_in, kill-line shape?)
    let shapes: Vec<(&str, &MetalTensor, usize, bool)> = vec![
        ("ffn_gate ", &gdn.ffn_gate, h, true),
        ("ffn_down ", &gdn.ffn_down, f, true),
        ("gdn_qkv  ", &gdn.in_proj_qkv, h, false),
        ("attn_q   ", &attn.q, h, false),
    ];

    let timed = |encode: &dyn Fn(&KernelEncoder)| -> f64 {
        const ITERS: usize = 32;
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let cmd = ctx_metal.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            for _ in 0..ITERS {
                encode(&enc);
            }
            enc.end();
            let t = std::time::Instant::now();
            cmd.commit();
            cmd.waitUntilCompleted();
            best = best.min(t.elapsed().as_secs_f64() / ITERS as f64);
        }
        best
    };

    let mut kill_line: Vec<(String, f64)> = Vec::new();

    for (label, w, n_in, is_kill_shape) in shapes {
        if !matches!(w.dtype, GgmlType::Q4_K | GgmlType::Q6_K) {
            eprintln!(
                "[h5.6-mma8 {label}] skipped — dtype {:?} not in {{Q4_K, Q6_K}}",
                w.dtype
            );
            continue;
        }
        let n_out = w.n_elements() as usize / n_in;
        let w_bytes = w.buffer.length() as f64;

        // 8 DISTINCT nonzero column patterns (catches transpose/stride
        // bugs; padded columns compute for real per the cx jam).
        let x_src: Vec<f32> = (0..8 * n_in)
            .map(|i| {
                let c = i / n_in;
                let k = i % n_in;
                (((k * 7 + c * 13) % 23) as f32 - 11.0) * 1e-2 + (c as f32 + 1.0) * 1e-3
            })
            .collect();
        let x = MetalTensor::from_bytes(
            &ctx_metal,
            bytemuck::cast_slice(&x_src),
            vec![(8 * n_in) as u64],
            GgmlType::F32,
        )
        .expect("x");
        let y_mma = MetalTensor::zeros_f32(&ctx_metal, vec![(8 * n_out) as u64]).expect("y_mma");
        let y_ref = MetalTensor::zeros_f32(&ctx_metal, vec![(8 * n_out) as u64]).expect("y_ref");

        // --- Correctness: all 8 columns vs per-column mv1, cos >= 0.999. ---
        {
            let cmd = ctx_metal.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            encode_mat_mat_mma8_dispatch(&ctx_metal, &enc, w, &x, &y_mma, n_in, n_out)
                .expect("mma8");
            for c in 0..8 {
                let xc = x.view_subrange((c * n_in) as u64, vec![n_in as u64]);
                let yc = y_ref.view_subrange((c * n_out) as u64, vec![n_out as u64]);
                encode_mat_vec_dispatch(&ctx_metal, &enc, w, &xc, &yc, n_in, n_out)
                    .expect("mv1 ref");
            }
            enc.end();
            cmd.commit();
            cmd.waitUntilCompleted();

            let got = unsafe {
                std::slice::from_raw_parts(
                    y_mma.buffer.contents().as_ptr() as *const f32,
                    8 * n_out,
                )
            };
            let want = unsafe {
                std::slice::from_raw_parts(
                    y_ref.buffer.contents().as_ptr() as *const f32,
                    8 * n_out,
                )
            };
            let mut min_cos = f64::INFINITY;
            let mut max_rel_rms = 0f64;
            for c in 0..8 {
                let (mut dot, mut n2g, mut n2w, mut err2) = (0f64, 0f64, 0f64, 0f64);
                for i in c * n_out..(c + 1) * n_out {
                    let (gv, wv) = (got[i] as f64, want[i] as f64);
                    dot += gv * wv;
                    n2g += gv * gv;
                    n2w += wv * wv;
                    err2 += (gv - wv) * (gv - wv);
                }
                let cos = dot / (n2g.sqrt() * n2w.sqrt()).max(1e-30);
                let rel_rms = (err2 / n2w.max(1e-30)).sqrt();
                min_cos = min_cos.min(cos);
                max_rel_rms = max_rel_rms.max(rel_rms);
            }
            eprintln!(
                "[h5.6-mma8 {label}] correctness: min_cos {min_cos:.6} max_rel_rms {max_rel_rms:.2e} (8/8 cols)"
            );
            assert!(
                min_cos >= 0.999,
                "[h5.6-mma8 {label}] min per-column cos {min_cos:.6} < 0.999 vs mv1 (E1 gate)"
            );
        }

        // --- Bench: mv1 reference vs mma8 (always 8 padded columns). ---
        let x1 = x.view_subrange(0, vec![n_in as u64]);
        let y1 = y_ref.view_subrange(0, vec![n_out as u64]);
        let t_mv = timed(&|enc| {
            encode_mat_vec_dispatch(&ctx_metal, enc, w, &x1, &y1, n_in, n_out).expect("mat_vec");
        });
        let t_mma = timed(&|enc| {
            encode_mat_mat_mma8_dispatch(&ctx_metal, enc, w, &x, &y_mma, n_in, n_out)
                .expect("mma8");
        });
        let gbps_mv = w_bytes / t_mv / 1e9;
        let gbps_mma = w_bytes / t_mma / 1e9;
        let tflops = (8.0 * 2.0 * n_in as f64 * n_out as f64) / t_mma / 1e12;
        let c_ratio = t_mma / t_mv;
        eprintln!(
            "[h5.6-mma8 {label}] {n_in:>5}->{n_out:>5}  mv1 {:7.3} ms {gbps_mv:5.0} GB/s | mma8 {:7.3} ms {gbps_mma:5.0} GB/s-eq {tflops:5.2} TF c={c_ratio:4.2} ({} TGs)",
            t_mv * 1e3,
            t_mma * 1e3,
            n_out / 8
        );
        if is_kill_shape {
            kill_line.push((label.trim().to_string(), c_ratio));
        }
    }

    assert_eq!(kill_line.len(), 2, "both kill-line shapes must be measured");
    let pass = kill_line.iter().all(|(_, c)| *c <= 1.25);
    let detail: Vec<String> = kill_line
        .iter()
        .map(|(l, c)| format!("{l}={c:.2}"))
        .collect();
    eprintln!(
        "[h5.6-mma8] kill line c(2) <= 1.25 on {{ffn_gate, ffn_down}}: {} → {}",
        detail.join(" "),
        if pass {
            "PASS — small-N MMA lane stays open"
        } else {
            "FAIL — shallow small-N lane closes per pre-registration"
        }
    );
}

/// **v0.500 small-N matmul selection sweep** (`smalln_selection_sweep_27b`).
/// Britt's ask: systematic config-permutation sweep so the small-N story
/// carries no stale kernel assumptions. cx-vetted axes/ranges/granularity
/// (session `019f393b-7...`): Family C first (selection permutations of
/// EXISTING kernels — the staleness audit found N in {2,3,4,8} verify
/// falls to the generic 32-wide tile because n16 requires n_query==16);
/// Family A staged mma8v variants (RT/CT/KS/SGS); Family B one TG-shape
/// point (nc2 rp4). Scope: MATMUL SELECTION ONLY (per-token GDN/attn/rope
/// staleness is a separate recorded scope).
///
/// Pre-registered reads: R1 reopen iff any config c(2) <= 1.25 on BOTH
/// ffn_gate and ffn_down; R2 recommend dispatcher change iff a config
/// beats the CURRENT selection >= 10% at some (N, shape) (production
/// flip needs repeat + e2e confirmation, separate gate); R3 updates the
/// verify(16) PROJECTION cost only; R4 else closure hardens to "swept
/// matmul neighborhood".
///
/// Rows are machine-parseable: `[sweep] shape= dtype= N= cfg= ms= gbps=
/// c= cos=`. Correctness asserted per (config, shape, N): live-column
/// cos >= 0.999 vs per-column mv1 (E0 families are separately
/// bit-asserted by their own gates); padded outputs finite-checked.
///
/// `cargo test -p qwen-llm --release --test dflash_correctness \
///   smalln_selection_sweep_27b -- --ignored --nocapture`
/// (run a second time with `QWEN_MATMAT_N16_V2=1` for the v2 rows —
/// the flag latches on first read.)
#[test]
#[ignore = "slow: loads 27B; run explicitly (see doc comment)"]
fn smalln_selection_sweep_27b() {
    if !std::path::Path::new(TARGET_GGUF).exists() {
        eprintln!("[sweep] skipped — target GGUF missing");
        return;
    }
    let ctx_metal = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("metal init: {e}"),
    };

    eprintln!("[sweep] loading 27B-Q4_K_M…");
    let g = GgufFile::open(TARGET_GGUF).expect("open target");
    let m = Model::from_gguf(&g).expect("load target");
    let mm = MetalModel::load(&ctx_metal, &g, &m).expect("metal load");

    let h = mm.arch.hidden_size as usize;
    let f = mm.arch.intermediate_size as usize;
    let gdn = mm
        .blocks
        .iter()
        .find_map(|b| match b {
            MetalBlock::Gdn(g) => Some(g),
            _ => None,
        })
        .expect("gdn block");
    let attn = mm
        .blocks
        .iter()
        .find_map(|b| match b {
            MetalBlock::Attn(a) => Some(a),
            _ => None,
        })
        .expect("attn block");

    let shapes: Vec<(&str, &MetalTensor, usize)> = vec![
        ("ffn_gate", &gdn.ffn_gate, h),
        ("ffn_down", &gdn.ffn_down, f),
        ("gdn_qkv", &gdn.in_proj_qkv, h),
        ("gdn_out", &gdn.out_proj, {
            gdn.out_proj.n_elements() as usize / h
        }),
        ("attn_q", &attn.q, h),
        ("attn_o", &attn.o, { attn.o.n_elements() as usize / h }),
        ("lm_head", &mm.lm_head, h),
    ];

    #[allow(clippy::type_complexity)]
    let timed = |encode: &dyn Fn(&KernelEncoder), iters: usize, reps: usize| -> f64 {
        let mut best = f64::INFINITY;
        for _ in 0..reps {
            let cmd = ctx_metal.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            for _ in 0..iters {
                encode(&enc);
            }
            enc.end();
            let t = std::time::Instant::now();
            cmd.commit();
            cmd.waitUntilCompleted();
            best = best.min(t.elapsed().as_secs_f64() / iters as f64);
        }
        best
    };

    let ns: [usize; 5] = [2, 3, 4, 8, 16];
    let mut best_c2_kill: Vec<(String, f64, String)> = Vec::new(); // (shape, best c, cfg)

    for (label, w, n_in) in shapes {
        let n_out = w.n_elements() as usize / n_in;
        let w_bytes = w.buffer.length() as f64;
        let dtype = w.dtype;
        // lm_head is ~15x the next-largest tensor; lighter sampling.
        let (iters, reps) = if n_out > 100_000 { (8, 3) } else { (32, 5) };

        // 16 distinct nonzero sentinel columns.
        let x_src: Vec<f32> = (0..16 * n_in)
            .map(|i| {
                let c = i / n_in;
                let k = i % n_in;
                (((k * 7 + c * 13) % 23) as f32 - 11.0) * 1e-2 + (c as f32 + 1.0) * 1e-3
            })
            .collect();
        let x = MetalTensor::from_bytes(
            &ctx_metal,
            bytemuck::cast_slice(&x_src),
            vec![(16 * n_in) as u64],
            GgmlType::F32,
        )
        .expect("x");
        let y_test = MetalTensor::zeros_f32(&ctx_metal, vec![(16 * n_out) as u64]).expect("y_test");
        let y_ref = MetalTensor::zeros_f32(&ctx_metal, vec![(16 * n_out) as u64]).expect("y_ref");

        // Reference: mv1 per column, all 16 columns.
        {
            let cmd = ctx_metal.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            for c in 0..16 {
                let xc = x.view_subrange((c * n_in) as u64, vec![n_in as u64]);
                let yc = y_ref.view_subrange((c * n_out) as u64, vec![n_out as u64]);
                encode_mat_vec_dispatch(&ctx_metal, &enc, w, &xc, &yc, n_in, n_out)
                    .expect("mv1 ref");
            }
            enc.end();
            cmd.commit();
            cmd.waitUntilCompleted();
        }
        let want_all = unsafe {
            std::slice::from_raw_parts(y_ref.buffer.contents().as_ptr() as *const f32, 16 * n_out)
        };

        // Anchor: mv1 at block start.
        let x1 = x.view_subrange(0, vec![n_in as u64]);
        let y1 = y_test.view_subrange(0, vec![n_out as u64]);
        let t_mv_a = timed(
            &|enc| {
                encode_mat_vec_dispatch(&ctx_metal, enc, w, &x1, &y1, n_in, n_out).expect("mv");
            },
            iters,
            reps,
        );

        // (cfg name, valid at N?, encode closure builder) — run per N.
        for &n in &ns {
            // Enumerate configs valid at this (N, dtype).
            #[allow(clippy::type_complexity)]
            let mut cfgs: Vec<(String, Box<dyn Fn(&KernelEncoder) + '_>, usize)> = Vec::new(); // (name, encode, cols_computed)

            // C1: current production selection at n_query=N.
            {
                let xn = x.view_subrange(0, vec![(n * n_in) as u64]);
                let yn = y_test.view_subrange(0, vec![(n * n_out) as u64]);
                let (ctx2, w2) = (&ctx_metal, w);
                cfgs.push((
                    "current".into(),
                    Box::new(move |enc| {
                        encode_mat_mat_dispatch(ctx2, enc, w2, &xn, &yn, n_in, n_out, n)
                            .expect("mm current");
                    }),
                    n,
                ));
            }
            // C2: padded-to-16 n16 (only meaningful when N < 16).
            if n < 16 {
                let xn = x.view_subrange(0, vec![(16 * n_in) as u64]);
                let yn = y_test.view_subrange(0, vec![(16 * n_out) as u64]);
                let (ctx2, w2) = (&ctx_metal, w);
                cfgs.push((
                    "pad16".into(),
                    Box::new(move |enc| {
                        encode_mat_mat_dispatch(ctx2, enc, w2, &xn, &yn, n_in, n_out, 16)
                            .expect("mm pad16");
                    }),
                    16,
                ));
            }
            // nc scalar family (Q4_K/Q6_K, N in {2,4,8}).
            if matches!(n, 2 | 4 | 8) && matches!(dtype, GgmlType::Q4_K | GgmlType::Q6_K) {
                let xn = x.view_subrange(0, vec![(n * n_in) as u64]);
                let yn = y_test.view_subrange(0, vec![(n * n_out) as u64]);
                let (ctx2, w2) = (&ctx_metal, w);
                cfgs.push((
                    "nc".into(),
                    Box::new(move |enc| {
                        encode_mat_vec_nc_dispatch(ctx2, enc, w2, &xn, &yn, n_in, n_out, n)
                            .expect("nc");
                    }),
                    n,
                ));
            }
            // B1: nc2 rp4 (Q4_K, N=2).
            if n == 2 && dtype == GgmlType::Q4_K {
                let xn = x.view_subrange(0, vec![(2 * n_in) as u64]);
                let yn = y_test.view_subrange(0, vec![(2 * n_out) as u64]);
                let (ctx2, w2) = (&ctx_metal, w);
                cfgs.push((
                    "nc2rp4".into(),
                    Box::new(move |enc| {
                        encode_mat_vec_q4_k_nc2_rp4_f32(ctx2, enc, w2, &xn, &yn, n_in, n_out)
                            .expect("nc2rp4");
                    }),
                    2,
                ));
            }
            // mma8 (8-col) at N <= 8; composed 2x at N=16.
            if matches!(dtype, GgmlType::Q4_K | GgmlType::Q6_K) && n_out.is_multiple_of(8) {
                if n <= 8 {
                    let xn = x.view_subrange(0, vec![(8 * n_in) as u64]);
                    let yn = y_test.view_subrange(0, vec![(8 * n_out) as u64]);
                    let (ctx2, w2) = (&ctx_metal, w);
                    cfgs.push((
                        "mma8".into(),
                        Box::new(move |enc| {
                            encode_mat_mat_mma8_dispatch(ctx2, enc, w2, &xn, &yn, n_in, n_out)
                                .expect("mma8");
                        }),
                        8,
                    ));
                    // Family A variants (8-col). Row-per-TG constraints
                    // (16 for r2c1k64/sg2) are re-checked below and in the
                    // encode fn; all swept shapes satisfy n_out % 16 == 0.
                    for v in ["r2c1k64", "r1c1k128", "r1c1k64_sg2", "r2c1k128", "r4c1k64"] {
                        if v == "r4c1k64" && !n_out.is_multiple_of(32) {
                            continue;
                        }
                        let xn = x.view_subrange(0, vec![(8 * n_in) as u64]);
                        let yn = y_test.view_subrange(0, vec![(8 * n_out) as u64]);
                        let (ctx2, w2) = (&ctx_metal, w);
                        cfgs.push((
                            format!("mma8v-{v}"),
                            Box::new(move |enc| {
                                encode_mat_mat_mma8_variant(
                                    ctx2, enc, w2, &xn, &yn, n_in, n_out, v,
                                )
                                .expect("mma8v");
                            }),
                            8,
                        ));
                    }
                } else {
                    // N=16: composed 2x mma8 and the 16-col variants.
                    let xa = x.view_subrange(0, vec![(8 * n_in) as u64]);
                    let ya = y_test.view_subrange(0, vec![(8 * n_out) as u64]);
                    let xb = x.view_subrange((8 * n_in) as u64, vec![(8 * n_in) as u64]);
                    let yb = y_test.view_subrange((8 * n_out) as u64, vec![(8 * n_out) as u64]);
                    let (ctx2, w2) = (&ctx_metal, w);
                    cfgs.push((
                        "mma8x2".into(),
                        Box::new(move |enc| {
                            encode_mat_mat_mma8_dispatch(ctx2, enc, w2, &xa, &ya, n_in, n_out)
                                .expect("mma8x2a");
                            encode_mat_mat_mma8_dispatch(ctx2, enc, w2, &xb, &yb, n_in, n_out)
                                .expect("mma8x2b");
                        }),
                        16,
                    ));
                    for v in ["r1c2k64", "r2c2k64", "r2c2k128"] {
                        let xn = x.view_subrange(0, vec![(16 * n_in) as u64]);
                        let yn = y_test.view_subrange(0, vec![(16 * n_out) as u64]);
                        let (ctx2, w2) = (&ctx_metal, w);
                        cfgs.push((
                            format!("mma8v-{v}"),
                            Box::new(move |enc| {
                                encode_mat_mat_mma8_variant(
                                    ctx2, enc, w2, &xn, &yn, n_in, n_out, v,
                                )
                                .expect("mma8v16");
                            }),
                            16,
                        ));
                    }
                }
            }

            for (cfg, encode, cols_computed) in cfgs {
                // Skip variants whose row constraint fails (encode returns Err
                // inside closure would panic; pre-check the common one).
                if (cfg == "mma8v-r2c1k64" || cfg == "mma8v-r1c1k64_sg2" || cfg == "mma8v-r2c2k64")
                    && !n_out.is_multiple_of(16)
                {
                    continue;
                }
                // Correctness once: zero y_test, run encode, compare live N
                // columns; finiteness of computed-but-padded columns.
                unsafe {
                    std::ptr::write_bytes(
                        y_test.buffer.contents().as_ptr() as *mut u8,
                        0,
                        16 * n_out * 4,
                    );
                }
                {
                    let cmd = ctx_metal.queue.commandBuffer().expect("cmd");
                    let enc = KernelEncoder::begin(&cmd);
                    encode(&enc);
                    enc.end();
                    cmd.commit();
                    cmd.waitUntilCompleted();
                }
                let got = unsafe {
                    std::slice::from_raw_parts(
                        y_test.buffer.contents().as_ptr() as *const f32,
                        16 * n_out,
                    )
                };
                let mut min_cos = f64::INFINITY;
                for c in 0..n {
                    let (mut dot, mut n2g, mut n2w) = (0f64, 0f64, 0f64);
                    for i in c * n_out..(c + 1) * n_out {
                        let (gv, wv) = (got[i] as f64, want_all[i] as f64);
                        dot += gv * wv;
                        n2g += gv * gv;
                        n2w += wv * wv;
                    }
                    min_cos = min_cos.min(dot / (n2g.sqrt() * n2w.sqrt()).max(1e-30));
                }
                for (i, value) in got
                    .iter()
                    .enumerate()
                    .take(cols_computed * n_out)
                    .skip(n * n_out)
                {
                    assert!(
                        value.is_finite(),
                        "[sweep {label}] {cfg} N{n}: padded output not finite at {i}"
                    );
                }
                assert!(
                    min_cos >= 0.999,
                    "[sweep {label}] {cfg} N{n}: min live-column cos {min_cos:.6} < 0.999"
                );

                let t = timed(&encode, iters, reps);
                let gbps = w_bytes / t / 1e9;
                let c_ratio = t / t_mv_a;
                eprintln!(
                    "[sweep] shape={label} dtype={dtype:?} N={n} cfg={cfg} ms={:.3} gbps={gbps:.0} c={c_ratio:.2} cos={min_cos:.6}",
                    t * 1e3
                );
                if n == 2 && (label == "ffn_gate" || label == "ffn_down") {
                    match best_c2_kill.iter_mut().find(|(l, _, _)| l == label) {
                        Some(entry) if c_ratio < entry.1 => {
                            entry.1 = c_ratio;
                            entry.2 = cfg.clone();
                        }
                        Some(_) => {}
                        None => best_c2_kill.push((label.to_string(), c_ratio, cfg.clone())),
                    }
                }
            }
        }

        // Anchor: mv1 at block end; drift check.
        let t_mv_b = timed(
            &|enc| {
                encode_mat_vec_dispatch(&ctx_metal, enc, w, &x1, &y1, n_in, n_out).expect("mv");
            },
            iters,
            reps,
        );
        let drift = (t_mv_b - t_mv_a).abs() / t_mv_a;
        eprintln!(
            "[sweep] shape={label} anchor mv1 start={:.3}ms end={:.3}ms drift={:.1}%{}",
            t_mv_a * 1e3,
            t_mv_b * 1e3,
            drift * 100.0,
            if drift > 0.03 {
                "  ** NOISY BLOCK **"
            } else {
                ""
            }
        );
    }

    // R1 read across everything measured.
    for (l, c, cfg) in &best_c2_kill {
        eprintln!("[sweep] best c(2) {l}: {c:.2} ({cfg})");
    }
    let reopen = best_c2_kill.len() == 2 && best_c2_kill.iter().all(|(_, c, _)| *c <= 1.25);
    eprintln!(
        "[sweep] R1 reopen read (c(2) <= 1.25 on both kill shapes): {}",
        if reopen { "REOPEN" } else { "stands closed" }
    );
}

/// **v0.75.1 27B integration correctness gate**: exercises the
/// mat-mat half-staging FFN/Q/K/V/O paths AND the per-token attn-v4
/// path (16 attn layers in the 27B Q4_K_M model — none in 0.8B-F32).
///
/// Oracle: sequential `single_token_with_multi_hidden` over T=24
/// prompt tokens with K=5 capture layers (mirroring the spiritbuun
/// drafter). Experimental: one `prefill_tokens_with_multi_hidden`
/// call with P=16 chunk size, exercising T==P+r (24=16+8).
///
/// Cosine equivalence ≥ 0.999 required on:
///   * final logits (last prompt token's vocab)
///   * accumulated multi-hidden capture across all 5 layers × 24 tokens
///   * GDN state (48 layers) and conv tensors after the call
///   * KV state for every attn layer (16 layers) over [0, T) range
///   * `kv_n_pos[ai] == T` exact for every attn layer
///
/// NOT bit-exact because mat-mat half-staging differs from per-token
/// mat-vec summation order. The 0.8B lib test gates bit-exact on the
/// F32 fallback path; this test gates the cosine gate on the Q4_K /
/// Q5_K / Q6_K mat-mat path.
///
/// Wall: ~3-5 min on M4 Max (oracle is 24 single_token forwards =
/// ~24 * 50 ms = 1.2 s GPU + load ~10 s; experimental is ~0.4 s).
#[test]
fn prefill_tokens_matches_single_token_loop_27b() {
    if !std::path::Path::new(TARGET_GGUF).exists() {
        eprintln!("[prefill-vs-single-27b] skipped — target GGUF missing");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("metal init: {e}"),
    };

    eprintln!("[prefill-vs-single-27b] loading 27B-Q4_K_M (~10s on cold cache)…");
    let g = GgufFile::open(TARGET_GGUF).expect("open target");
    let m = Model::from_gguf(&g).expect("load target");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx, &mm);
    let arch = &mm.arch;
    let h = arch.hidden_size as usize;

    // T=24 with P=16 → 2 chunks (T=16 + T=8). Exercises full chunk
    // AND short tail chunk.
    let total_n: usize = std::env::var("QWEN_TEST_27B_PREFILL_T")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(24);
    let p: usize = std::env::var("QWEN_TEST_27B_PREFILL_P")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16);
    let token_ids: Vec<i32> = (0..total_n)
        .map(|i| ((i * 17 + 11) % (arch.vocab_size as usize - 1)) as i32 + 1)
        .collect();

    // K=5 capture layers (mirrors the spiritbuun drafter).
    let capture_layers: Vec<u32> = vec![1, 16, 31, 46, 61];
    let k = capture_layers.len();

    // ---- Oracle path ----
    eprintln!("[prefill-vs-single-27b] running oracle (T={total_n} sequential single_token)…");
    let cap = total_n + 16;
    let mut sess_a = MetalSession::fresh(&ctx, &mm, cap).expect("sess A");
    let h_dst_a = MetalTensor::zeros_f32(&ctx, vec![(k * h) as u64]).expect("h_dst_a");
    let mut accum_a = vec![0.0f32; total_n * k * h];
    let oracle_t = std::time::Instant::now();
    let mut last_a = Vec::new();
    for (i, &tid) in token_ids.iter().enumerate() {
        last_a = mf
            .single_token_with_multi_hidden(tid, i as u32, &mut sess_a, &capture_layers, &h_dst_a)
            .expect("oracle forward");
        unsafe {
            let src = h_dst_a.buffer.contents().as_ptr() as *const f32;
            std::ptr::copy_nonoverlapping(
                src,
                accum_a[i * k * h..(i + 1) * k * h].as_mut_ptr(),
                k * h,
            );
        }
    }
    let oracle_ms = oracle_t.elapsed().as_secs_f64() * 1e3;
    eprintln!("[prefill-vs-single-27b] oracle wall: {oracle_ms:.1} ms");

    // ---- Experimental path ----
    eprintln!(
        "[prefill-vs-single-27b] running prefill_tokens (P={p}, chunks={})…",
        total_n.div_ceil(p)
    );
    let mut sess_b = MetalSession::fresh(&ctx, &mm, cap).expect("sess B");
    let mut layer_scratch = MetalDFlashLayerMajorScratch::fresh_prefill_with_matrix_max_pos(
        &ctx, &mm, p as u32, total_n,
    )
    .expect("layer scratch");
    let h_dst_b = MetalTensor::zeros_f32(&ctx, vec![(total_n * k * h) as u64]).expect("h_dst_b");
    let exp_t = std::time::Instant::now();
    let last_b = prefill_tokens_with_multi_hidden(
        &mf,
        &token_ids,
        0,
        &mut sess_b,
        &mut layer_scratch,
        &capture_layers,
        Some(&h_dst_b),
    )
    .expect("prefill");
    let exp_ms = exp_t.elapsed().as_secs_f64() * 1e3;
    eprintln!("[prefill-vs-single-27b] prefill wall: {exp_ms:.1} ms");
    let speedup = oracle_ms / exp_ms;
    eprintln!("[prefill-vs-single-27b] SPEEDUP: {speedup:.2}× (correctness, not bench)");

    // ---- Compare final logits (cos ≥ 0.999). ----
    assert_eq!(last_a.len(), last_b.len(), "logits len mismatch");
    let cos_logits = cosine_27b(&last_a, &last_b);
    eprintln!("[prefill-vs-single-27b] cos(final logits)={cos_logits:.6}");
    assert!(
        cos_logits >= 0.999,
        "logits cos={cos_logits} < 0.999 (mat-mat half-staging gate)"
    );

    // ---- Compare accumulated multi-hidden. ----
    let mut accum_b = vec![0.0f32; total_n * k * h];
    unsafe {
        let src = h_dst_b.buffer.contents().as_ptr() as *const f32;
        std::ptr::copy_nonoverlapping(src, accum_b.as_mut_ptr(), total_n * k * h);
    }
    let mut min_cos = f64::INFINITY;
    let mut worst_pos = (0usize, 0usize);
    for t in 0..total_n {
        for k_idx in 0..k {
            let off = (t * k + k_idx) * h;
            let c = cosine_27b(&accum_a[off..off + h], &accum_b[off..off + h]);
            if c < min_cos {
                min_cos = c;
                worst_pos = (t, k_idx);
            }
        }
    }
    eprintln!(
        "[prefill-vs-single-27b] hidden cos_min={min_cos:.6} (worst at token={}, capture_layer={})",
        worst_pos.0, worst_pos.1
    );
    assert!(
        min_cos >= 0.999,
        "hidden capture cos_min={min_cos} < 0.999 (worst at token={}, capture_layer={})",
        worst_pos.0,
        worst_pos.1
    );

    // ---- Compare GDN state + conv per layer (cos ≥ 0.999). ----
    assert_eq!(sess_a.gdn_state.len(), sess_b.gdn_state.len());
    let mut gdn_state_min_cos = f64::INFINITY;
    let mut gdn_conv_min_cos = f64::INFINITY;
    for gi in 0..sess_a.gdn_state.len() {
        let a_state = read_tensor_f32_27b(&sess_a.gdn_state[gi]);
        let b_state = read_tensor_f32_27b(&sess_b.gdn_state[gi]);
        let cs = cosine_27b(&a_state, &b_state);
        let a_conv = read_tensor_f32_27b(&sess_a.gdn_conv[gi]);
        let b_conv = read_tensor_f32_27b(&sess_b.gdn_conv[gi]);
        let cc = cosine_27b(&a_conv, &b_conv);
        gdn_state_min_cos = gdn_state_min_cos.min(cs);
        gdn_conv_min_cos = gdn_conv_min_cos.min(cc);
    }
    eprintln!(
        "[prefill-vs-single-27b] GDN state cos_min={gdn_state_min_cos:.6} \
         conv cos_min={gdn_conv_min_cos:.6}"
    );
    assert!(
        gdn_state_min_cos >= 0.999,
        "GDN state cos_min={gdn_state_min_cos} < 0.999"
    );
    assert!(
        gdn_conv_min_cos >= 0.999,
        "GDN conv cos_min={gdn_conv_min_cos} < 0.999"
    );

    // ---- KV state cosine (read F16 → F32 via codec). ----
    assert_eq!(sess_a.kv_n_pos.len(), sess_b.kv_n_pos.len());
    let mut kv_k_min_cos = f64::INFINITY;
    let mut kv_v_min_cos = f64::INFINITY;
    let kv_dim = (arch.n_kv_heads * arch.attn_head_dim) as usize;
    for ai in 0..sess_a.kv_n_pos.len() {
        // kv_n_pos must match exactly.
        assert_eq!(
            sess_a.kv_n_pos[ai], sess_b.kv_n_pos[ai],
            "kv_n_pos[{ai}] mismatch ({} vs {})",
            sess_a.kv_n_pos[ai], sess_b.kv_n_pos[ai]
        );
        assert_eq!(
            sess_a.kv_n_pos[ai], total_n,
            "expected kv_n_pos[{ai}]={total_n} after T={total_n} prefill, got {}",
            sess_a.kv_n_pos[ai]
        );
        // KV is F16-backed; read first kv_n_pos[ai] * kv_dim half-floats and convert.
        let a_k = read_kv_prefix_f16_to_f32(&sess_a.kv_k[ai], total_n * kv_dim);
        let b_k = read_kv_prefix_f16_to_f32(&sess_b.kv_k[ai], total_n * kv_dim);
        let a_v = read_kv_prefix_f16_to_f32(&sess_a.kv_v[ai], total_n * kv_dim);
        let b_v = read_kv_prefix_f16_to_f32(&sess_b.kv_v[ai], total_n * kv_dim);
        kv_k_min_cos = kv_k_min_cos.min(cosine_27b(&a_k, &b_k));
        kv_v_min_cos = kv_v_min_cos.min(cosine_27b(&a_v, &b_v));
    }
    eprintln!(
        "[prefill-vs-single-27b] KV K cos_min={kv_k_min_cos:.6} V cos_min={kv_v_min_cos:.6} \
         (over [0, {total_n}) for {} attn layers)",
        sess_a.kv_n_pos.len()
    );
    assert!(kv_k_min_cos >= 0.999, "KV K cos_min={kv_k_min_cos} < 0.999");
    assert!(kv_v_min_cos >= 0.999, "KV V cos_min={kv_v_min_cos} < 0.999");
}

#[test]
#[ignore]
fn prefill_tokens_matches_single_token_loop_27b_matrix_g6_prefix_gate() {
    if !std::path::Path::new(TARGET_GGUF).exists() {
        eprintln!("[prefill-vs-single-27b-g6-prefix] skipped — target GGUF missing");
        return;
    }
    if std::env::var("QWEN_PREFILL_ATTN_MATRIX_G6")
        .as_deref()
        .map(|v| matches!(v, "0" | "false" | "FALSE" | "no" | "NO"))
        .unwrap_or(false)
    {
        eprintln!("[prefill-vs-single-27b-g6-prefix] skipped — G6 matrix force-disabled");
        return;
    }

    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("metal init: {e}"),
    };

    eprintln!("[prefill-vs-single-27b-g6-prefix] loading 27B-Q4_K_M…");
    let g = GgufFile::open(TARGET_GGUF).expect("open target");
    let m = Model::from_gguf(&g).expect("load target");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx, &mm);
    let arch = &mm.arch;

    let total_n: usize = 8;
    let p: usize = 8;
    let prefix_len: usize = 4096;
    let n_total = prefix_len + total_n;
    let all_tokens: Vec<i32> = (0..n_total)
        .map(|i| ((i * 17 + 11) % (arch.vocab_size as usize - 1)) as i32 + 1)
        .collect();
    let prefix_tokens = &all_tokens[..prefix_len];
    let token_ids = &all_tokens[prefix_len..];

    eprintln!(
        "[prefill-vs-single-27b-g6-prefix] priming prefix={prefix_len}, then comparing T={total_n} P={p}"
    );

    let cap = n_total + 16;
    let mut sess_a = MetalSession::fresh(&ctx, &mm, cap).expect("sess A");
    for (i, &tid) in prefix_tokens.iter().enumerate() {
        mf.single_token(tid, i as u32, &mut sess_a)
            .expect("oracle prefix advance");
    }
    let oracle_t = std::time::Instant::now();
    let mut last_a = Vec::new();
    for (i, &tid) in token_ids.iter().enumerate() {
        last_a = mf
            .single_token(tid, (prefix_len + i) as u32, &mut sess_a)
            .expect("oracle forward");
    }
    let oracle_ms = oracle_t.elapsed().as_secs_f64() * 1e3;

    let mut sess_b = MetalSession::fresh(&ctx, &mm, cap).expect("sess B");
    for (i, &tid) in prefix_tokens.iter().enumerate() {
        mf.single_token(tid, i as u32, &mut sess_b)
            .expect("experimental prefix advance");
    }
    let mut layer_scratch = MetalDFlashLayerMajorScratch::fresh_prefill_with_matrix_max_pos(
        &ctx, &mm, p as u32, n_total,
    )
    .expect("layer scratch");
    let exp_t = std::time::Instant::now();
    let last_b = prefill_tokens_with_multi_hidden(
        &mf,
        token_ids,
        prefix_len as u32,
        &mut sess_b,
        &mut layer_scratch,
        &[],
        None,
    )
    .expect("prefill");
    let exp_ms = exp_t.elapsed().as_secs_f64() * 1e3;

    let cos_logits = cosine_27b(&last_a, &last_b);
    eprintln!(
        "[prefill-vs-single-27b-g6-prefix] oracle={oracle_ms:.1}ms prefill={exp_ms:.1}ms cos(logits)={cos_logits:.6}"
    );
    assert!(cos_logits >= 0.999, "logits cos={cos_logits} < 0.999");

    assert_eq!(sess_a.gdn_state.len(), sess_b.gdn_state.len());
    let mut gdn_state_min_cos = f64::INFINITY;
    let mut gdn_conv_min_cos = f64::INFINITY;
    for gi in 0..sess_a.gdn_state.len() {
        let a_state = read_tensor_f32_27b(&sess_a.gdn_state[gi]);
        let b_state = read_tensor_f32_27b(&sess_b.gdn_state[gi]);
        gdn_state_min_cos = gdn_state_min_cos.min(cosine_27b(&a_state, &b_state));
        let a_conv = read_tensor_f32_27b(&sess_a.gdn_conv[gi]);
        let b_conv = read_tensor_f32_27b(&sess_b.gdn_conv[gi]);
        gdn_conv_min_cos = gdn_conv_min_cos.min(cosine_27b(&a_conv, &b_conv));
    }
    eprintln!(
        "[prefill-vs-single-27b-g6-prefix] GDN state cos_min={gdn_state_min_cos:.6} conv cos_min={gdn_conv_min_cos:.6}"
    );
    assert!(
        gdn_state_min_cos >= 0.999,
        "GDN state cos_min={gdn_state_min_cos} < 0.999"
    );
    assert!(
        gdn_conv_min_cos >= 0.999,
        "GDN conv cos_min={gdn_conv_min_cos} < 0.999"
    );

    assert_eq!(sess_a.kv_k.len(), sess_b.kv_k.len());
    let kv_prefix_elems = n_total * (arch.n_kv_heads as usize * arch.attn_head_dim as usize);
    let mut kv_k_min_cos = f64::INFINITY;
    let mut kv_v_min_cos = f64::INFINITY;
    for ai in 0..sess_a.kv_k.len() {
        assert_eq!(
            sess_a.kv_n_pos[ai], n_total,
            "oracle kv_n_pos[{ai}] != total"
        );
        assert_eq!(
            sess_b.kv_n_pos[ai], n_total,
            "prefill kv_n_pos[{ai}] != total"
        );
        let a_k = read_kv_prefix_f16_to_f32(&sess_a.kv_k[ai], kv_prefix_elems);
        let b_k = read_kv_prefix_f16_to_f32(&sess_b.kv_k[ai], kv_prefix_elems);
        let a_v = read_kv_prefix_f16_to_f32(&sess_a.kv_v[ai], kv_prefix_elems);
        let b_v = read_kv_prefix_f16_to_f32(&sess_b.kv_v[ai], kv_prefix_elems);
        kv_k_min_cos = kv_k_min_cos.min(cosine_27b(&a_k, &b_k));
        kv_v_min_cos = kv_v_min_cos.min(cosine_27b(&a_v, &b_v));
    }
    eprintln!(
        "[prefill-vs-single-27b-g6-prefix] KV K cos_min={kv_k_min_cos:.6} V cos_min={kv_v_min_cos:.6}"
    );
    assert!(kv_k_min_cos >= 0.999, "KV K cos_min={kv_k_min_cos} < 0.999");
    assert!(kv_v_min_cos >= 0.999, "KV V cos_min={kv_v_min_cos} < 0.999");
}

#[test]
fn prefill_tokens_matches_single_token_loop_35b_a3b_moe() {
    let Some(model_path) = qwen_llm::test_fixtures::A3B_Q4_K_M.path_or_skip() else {
        eprintln!("[prefill-vs-single-a3b] skipped — target GGUF missing");
        return;
    };
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("metal init: {e}"),
    };

    eprintln!("[prefill-vs-single-a3b] loading 35B A3B…");
    let g = GgufFile::open(&model_path).expect("open target");
    let m = Model::from_gguf(&g).expect("load target");
    assert_eq!(m.arch.kind, qwen_llm::model::ArchKind::Moe);
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx, &mm);
    let arch = &mm.arch;

    let total_n_limit: Option<usize> = std::env::var("QWEN_A3B_MOE_TEST_TOTAL")
        .ok()
        .and_then(|s| s.parse().ok());
    let p: usize = std::env::var("QWEN_A3B_MOE_TEST_CHUNK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let cont_tokens: usize = std::env::var("QWEN_A3B_MOE_TEST_CONT_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let cont_cos_gate: Option<f64> = std::env::var("QWEN_A3B_MOE_TEST_CONT_COS_MIN")
        .ok()
        .and_then(|s| s.parse().ok());
    let logits_cos_gate: Option<f64> = std::env::var("QWEN_A3B_MOE_TEST_LOGITS_COS_MIN")
        .ok()
        .and_then(|s| s.parse().ok());
    let internal_cos_gate: Option<f64> = std::env::var("QWEN_A3B_MOE_TEST_INTERNAL_COS_MIN")
        .ok()
        .and_then(|s| s.parse().ok());
    let require_argmax_match_override = std::env::var("QWEN_A3B_MOE_TEST_REQUIRE_ARGMAX_MATCH")
        .ok()
        .map(|s| s != "0");
    let rank_escape_gate: Option<usize> = std::env::var("QWEN_A3B_MOE_TEST_RANK_ESCAPE_MAX")
        .ok()
        .and_then(|s| s.parse().ok());
    let (mut token_ids, source_label) =
        if let Ok(path) = std::env::var("QWEN_A3B_MOE_TEST_PROMPT_FILE") {
            let prompt = std::fs::read_to_string(&path).expect("read prompt file");
            let tok = Tokenizer::from_gguf(&g).expect("open tokenizer");
            let ids = tok.encode(&prompt, false).expect("tokenize prompt file");
            (ids, format!("file:{path}"))
        } else {
            let total_n = total_n_limit.unwrap_or(12);
            let ids: Vec<i32> = (0..total_n)
                .map(|i| ((i * 17 + 11) % (arch.vocab_size as usize - 1)) as i32 + 1)
                .collect();
            (ids, format!("synthetic:{total_n}"))
        };
    if let Some(max_n) = total_n_limit {
        token_ids.truncate(max_n);
    }
    assert!(!token_ids.is_empty(), "A3B test prompt tokenized empty");
    let total_n = token_ids.len();
    let require_argmax_match = require_argmax_match_override.unwrap_or(total_n <= 128);
    let cap = total_n + cont_tokens + 16;

    eprintln!("[prefill-vs-single-a3b] source={source_label} tokens={total_n}");
    eprintln!("[prefill-vs-single-a3b] running oracle…");
    let mut sess_a = MetalSession::fresh(&ctx, &mm, cap).expect("sess A");
    let oracle_t = std::time::Instant::now();
    let mut last_a = Vec::new();
    for (i, &tid) in token_ids.iter().enumerate() {
        last_a = mf
            .single_token(tid, i as u32, &mut sess_a)
            .expect("oracle forward");
    }
    let oracle_ms = oracle_t.elapsed().as_secs_f64() * 1e3;
    eprintln!("[prefill-vs-single-a3b] oracle wall: {oracle_ms:.1} ms");

    eprintln!(
        "[prefill-vs-single-a3b] running prefill_tokens (P={p}, chunks={})…",
        total_n.div_ceil(p)
    );
    let mut sess_b = MetalSession::fresh(&ctx, &mm, cap).expect("sess B");
    let mut layer_scratch = MetalDFlashLayerMajorScratch::fresh_prefill_with_matrix_max_pos(
        &ctx, &mm, p as u32, total_n,
    )
    .expect("layer scratch");
    let exp_t = std::time::Instant::now();
    let last_b = prefill_tokens_with_multi_hidden(
        &mf,
        &token_ids,
        0,
        &mut sess_b,
        &mut layer_scratch,
        &[],
        None,
    )
    .expect("prefill");
    let exp_ms = exp_t.elapsed().as_secs_f64() * 1e3;
    eprintln!("[prefill-vs-single-a3b] prefill wall: {exp_ms:.1} ms");
    eprintln!(
        "[prefill-vs-single-a3b] SPEEDUP: {:.2}×",
        oracle_ms / exp_ms
    );

    assert_eq!(last_a.len(), last_b.len(), "logits len mismatch");
    let cos_logits = cosine_27b(&last_a, &last_b);
    eprintln!("[prefill-vs-single-a3b] cos(final logits)={cos_logits:.6}");

    let gdn_blocks: Vec<usize> = mm
        .blocks
        .iter()
        .enumerate()
        .filter_map(|(i, block)| matches!(block, MetalBlock::Gdn(_)).then_some(i))
        .collect();
    let gdn_cos = compare_gdn_state_conv_27b(&sess_a, &sess_b);
    let state_block = gdn_blocks
        .get(gdn_cos.state_worst_layer)
        .copied()
        .unwrap_or(gdn_cos.state_worst_layer);
    let conv_block = gdn_blocks
        .get(gdn_cos.conv_worst_layer)
        .copied()
        .unwrap_or(gdn_cos.conv_worst_layer);
    eprintln!(
        "[prefill-vs-single-a3b] GDN state cos_min={:.6} worst_gdn={} blk={} max|Δ|={:.3e} \
         conv cos_min={:.6} worst_gdn={} blk={} max|Δ|={:.3e}",
        gdn_cos.state_min_cos,
        gdn_cos.state_worst_layer,
        state_block,
        gdn_cos.state_max_abs,
        gdn_cos.conv_min_cos,
        gdn_cos.conv_worst_layer,
        conv_block,
        gdn_cos.conv_max_abs,
    );

    let attn_blocks: Vec<usize> = mm
        .blocks
        .iter()
        .enumerate()
        .filter_map(|(i, block)| matches!(block, MetalBlock::Attn(_)).then_some(i))
        .collect();
    let kv_prefix_elems = total_n * (arch.n_kv_heads as usize * arch.attn_head_dim as usize);
    let kv_cos = compare_kv_prefix_27b(&sess_a, &sess_b, kv_prefix_elems, total_n);
    let k_block = attn_blocks
        .get(kv_cos.k_worst_layer)
        .copied()
        .unwrap_or(kv_cos.k_worst_layer);
    let v_block = attn_blocks
        .get(kv_cos.v_worst_layer)
        .copied()
        .unwrap_or(kv_cos.v_worst_layer);
    eprintln!(
        "[prefill-vs-single-a3b] KV K cos_min={:.6} worst_attn={} blk={} max|Δ|={:.3e} \
         V cos_min={:.6} worst_attn={} blk={} max|Δ|={:.3e}",
        kv_cos.k_min_cos,
        kv_cos.k_worst_layer,
        k_block,
        kv_cos.k_max_abs,
        kv_cos.v_min_cos,
        kv_cos.v_worst_layer,
        v_block,
        kv_cos.v_max_abs,
    );
    let mut cont_input = argmax_i32_27b(&last_a);
    let prefill_argmax = argmax_i32_27b(&last_b);
    let mut cont_min_cos = 1.0;
    let mut cont_worst_step = None;
    let mut cont_first_mismatch = None;
    if prefill_argmax != cont_input {
        cont_first_mismatch = Some(ContinuationMismatch::from_logits(
            usize::MAX,
            cont_input,
            prefill_argmax,
            &last_a,
            &last_b,
        ));
    }
    for step in 0..cont_tokens {
        let pos = total_n + step;
        let next_a = mf
            .single_token(cont_input, pos as u32, &mut sess_a)
            .expect("oracle continuation");
        let next_b = mf
            .single_token(cont_input, pos as u32, &mut sess_b)
            .expect("prefill continuation");
        let cos_next = cosine_27b(&next_a, &next_b);
        if cos_next < cont_min_cos {
            cont_min_cos = cos_next;
            cont_worst_step = Some(step);
        }
        let argmax_a = argmax_i32_27b(&next_a);
        let argmax_b = argmax_i32_27b(&next_b);
        if cont_first_mismatch.is_none() && argmax_a != argmax_b {
            cont_first_mismatch = Some(ContinuationMismatch::from_logits(
                step, argmax_a, argmax_b, &next_a, &next_b,
            ));
        }
        cont_input = argmax_a;
    }
    eprintln!(
        "[prefill-vs-single-a3b] continuation tokens={} cos_min={cont_min_cos:.6} worst_step={cont_worst_step:?} first_mismatch={cont_first_mismatch:?}",
        cont_tokens,
    );
    let effective_logits_cos_gate = logits_cos_gate.or_else(|| (total_n <= 128).then_some(0.999));
    let effective_cont_cos_gate = cont_cos_gate.or_else(|| (cont_tokens <= 1).then_some(0.999));
    let effective_internal_cos_gate =
        internal_cos_gate.or_else(|| (total_n <= 64).then_some(0.999));
    eprintln!(
        "[prefill-vs-single-a3b] gates logits_cos={effective_logits_cos_gate:?} continuation_cos={effective_cont_cos_gate:?} internal_cos={effective_internal_cos_gate:?} require_argmax_match={require_argmax_match} rank_escape={rank_escape_gate:?}"
    );
    if let Some(threshold) = effective_logits_cos_gate {
        assert!(
            cos_logits >= threshold,
            "logits cos={cos_logits} < {threshold}"
        );
    }
    if require_argmax_match {
        assert!(
            cont_first_mismatch.is_none(),
            "continuation argmax mismatch: {cont_first_mismatch:?}"
        );
    }
    if let (Some(max_rank), Some(mismatch)) = (rank_escape_gate, cont_first_mismatch) {
        assert!(
            mismatch.oracle_rank_in_prefill <= max_rank
                && mismatch.prefill_rank_in_oracle <= max_rank,
            "continuation rank escape > {max_rank}: {mismatch:?}"
        );
    }
    if let Some(threshold) = effective_cont_cos_gate {
        assert!(
            cont_min_cos >= threshold,
            "continuation logits cos_min={cont_min_cos} < {threshold}"
        );
    }
    if let Some(threshold) = effective_internal_cos_gate {
        assert!(
            gdn_cos.state_min_cos >= threshold,
            "GDN state cos_min={} < {} (worst_gdn={} blk={} max_abs={})",
            gdn_cos.state_min_cos,
            threshold,
            gdn_cos.state_worst_layer,
            state_block,
            gdn_cos.state_max_abs,
        );
        assert!(
            gdn_cos.conv_min_cos >= threshold,
            "GDN conv cos_min={} < {} (worst_gdn={} blk={} max_abs={})",
            gdn_cos.conv_min_cos,
            threshold,
            gdn_cos.conv_worst_layer,
            conv_block,
            gdn_cos.conv_max_abs,
        );
        assert!(
            kv_cos.k_min_cos >= threshold,
            "KV K cos_min={} < {} (worst_attn={} blk={} max_abs={})",
            kv_cos.k_min_cos,
            threshold,
            kv_cos.k_worst_layer,
            k_block,
            kv_cos.k_max_abs,
        );
        assert!(
            kv_cos.v_min_cos >= threshold,
            "KV V cos_min={} < {} (worst_attn={} blk={} max_abs={})",
            kv_cos.v_min_cos,
            threshold,
            kv_cos.v_worst_layer,
            v_block,
            kv_cos.v_max_abs,
        );
    }
}

#[test]
fn prefill_tokens_matches_single_token_loop_122b_a10b_moe_smoke() {
    let model_path = concat!(
        "/Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/",
        "Qwen3.5-122B-A10B-UD-Q4_K_XL-00001-of-00003.gguf"
    );
    if !std::path::Path::new(model_path).exists() {
        eprintln!("[prefill-vs-single-a10b] skipped — target GGUF missing");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("metal init: {e}"),
    };

    eprintln!("[prefill-vs-single-a10b] loading 122B A10B…");
    let g = GgufFile::open(model_path).expect("open target");
    let m = Model::from_gguf(&g).expect("load target");
    assert_eq!(m.arch.kind, qwen_llm::model::ArchKind::Moe);
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx, &mm);
    let arch = &mm.arch;

    let total_n: usize = 6;
    let p: usize = 4;
    let token_ids: Vec<i32> = (0..total_n)
        .map(|i| ((i * 19 + 7) % (arch.vocab_size as usize - 1)) as i32 + 1)
        .collect();
    let cap = total_n + 16;

    eprintln!("[prefill-vs-single-a10b] running oracle…");
    let mut sess_a = MetalSession::fresh(&ctx, &mm, cap).expect("sess A");
    let oracle_t = std::time::Instant::now();
    let mut last_a = Vec::new();
    for (i, &tid) in token_ids.iter().enumerate() {
        last_a = mf
            .single_token(tid, i as u32, &mut sess_a)
            .expect("oracle forward");
    }
    let oracle_ms = oracle_t.elapsed().as_secs_f64() * 1e3;
    eprintln!("[prefill-vs-single-a10b] oracle wall: {oracle_ms:.1} ms");

    eprintln!(
        "[prefill-vs-single-a10b] running prefill_tokens (P={p}, chunks={})…",
        total_n.div_ceil(p)
    );
    let mut sess_b = MetalSession::fresh(&ctx, &mm, cap).expect("sess B");
    let mut layer_scratch = MetalDFlashLayerMajorScratch::fresh_prefill_with_matrix_max_pos(
        &ctx, &mm, p as u32, total_n,
    )
    .expect("layer scratch");
    let exp_t = std::time::Instant::now();
    let last_b = prefill_tokens_with_multi_hidden(
        &mf,
        &token_ids,
        0,
        &mut sess_b,
        &mut layer_scratch,
        &[],
        None,
    )
    .expect("prefill");
    let exp_ms = exp_t.elapsed().as_secs_f64() * 1e3;
    eprintln!("[prefill-vs-single-a10b] prefill wall: {exp_ms:.1} ms");
    eprintln!(
        "[prefill-vs-single-a10b] SPEEDUP: {:.2}×",
        oracle_ms / exp_ms
    );

    assert_eq!(last_a.len(), last_b.len(), "logits len mismatch");
    let cos_logits = cosine_27b(&last_a, &last_b);
    eprintln!("[prefill-vs-single-a10b] cos(final logits)={cos_logits:.6}");

    assert_eq!(sess_a.gdn_state.len(), sess_b.gdn_state.len());
    let mut gdn_state_min_cos = f64::INFINITY;
    let mut gdn_conv_min_cos = f64::INFINITY;
    for gi in 0..sess_a.gdn_state.len() {
        let a_state = read_tensor_f32_27b(&sess_a.gdn_state[gi]);
        let b_state = read_tensor_f32_27b(&sess_b.gdn_state[gi]);
        gdn_state_min_cos = gdn_state_min_cos.min(cosine_27b(&a_state, &b_state));
        let a_conv = read_tensor_f32_27b(&sess_a.gdn_conv[gi]);
        let b_conv = read_tensor_f32_27b(&sess_b.gdn_conv[gi]);
        gdn_conv_min_cos = gdn_conv_min_cos.min(cosine_27b(&a_conv, &b_conv));
    }
    eprintln!(
        "[prefill-vs-single-a10b] GDN state cos_min={gdn_state_min_cos:.6} conv cos_min={gdn_conv_min_cos:.6}"
    );
    assert!(
        gdn_state_min_cos >= 0.999,
        "GDN state cos_min={gdn_state_min_cos} < 0.999"
    );
    assert!(
        gdn_conv_min_cos >= 0.999,
        "GDN conv cos_min={gdn_conv_min_cos} < 0.999"
    );

    assert_eq!(sess_a.kv_k.len(), sess_b.kv_k.len());
    let kv_prefix_elems = total_n * (arch.n_kv_heads as usize * arch.attn_head_dim as usize);
    let mut kv_k_min_cos = f64::INFINITY;
    let mut kv_v_min_cos = f64::INFINITY;
    for ai in 0..sess_a.kv_k.len() {
        assert_eq!(sess_a.kv_n_pos[ai], total_n, "oracle kv_n_pos[{ai}] != T");
        assert_eq!(sess_b.kv_n_pos[ai], total_n, "prefill kv_n_pos[{ai}] != T");
        let a_k = read_kv_prefix_f16_to_f32(&sess_a.kv_k[ai], kv_prefix_elems);
        let b_k = read_kv_prefix_f16_to_f32(&sess_b.kv_k[ai], kv_prefix_elems);
        let a_v = read_kv_prefix_f16_to_f32(&sess_a.kv_v[ai], kv_prefix_elems);
        let b_v = read_kv_prefix_f16_to_f32(&sess_b.kv_v[ai], kv_prefix_elems);
        kv_k_min_cos = kv_k_min_cos.min(cosine_27b(&a_k, &b_k));
        kv_v_min_cos = kv_v_min_cos.min(cosine_27b(&a_v, &b_v));
    }
    eprintln!(
        "[prefill-vs-single-a10b] KV K cos_min={kv_k_min_cos:.6} V cos_min={kv_v_min_cos:.6}"
    );
    assert!(cos_logits >= 0.999, "logits cos={cos_logits} < 0.999");
    assert!(kv_k_min_cos >= 0.999, "KV K cos_min={kv_k_min_cos} < 0.999");
    assert!(kv_v_min_cos >= 0.999, "KV V cos_min={kv_v_min_cos} < 0.999");
}

#[test]
#[ignore]
fn prefill_tokens_matches_single_token_loop_122b_a10b_moe_chunk128_boundary() {
    let model_path = concat!(
        "/Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/",
        "Qwen3.5-122B-A10B-UD-Q4_K_XL-00001-of-00003.gguf"
    );
    if !std::path::Path::new(model_path).exists() {
        eprintln!("[prefill-vs-single-a10b-boundary] skipped — target GGUF missing");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("metal init: {e}"),
    };

    eprintln!("[prefill-vs-single-a10b-boundary] loading 122B A10B…");
    let g = GgufFile::open(model_path).expect("open target");
    let m = Model::from_gguf(&g).expect("load target");
    assert_eq!(m.arch.kind, qwen_llm::model::ArchKind::Moe);
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx, &mm);
    let arch = &mm.arch;

    let total_n: usize = 129;
    let p: usize = 128;
    let token_ids: Vec<i32> = (0..total_n)
        .map(|i| ((i * 19 + 7) % (arch.vocab_size as usize - 1)) as i32 + 1)
        .collect();
    let cap = total_n + 16;

    eprintln!("[prefill-vs-single-a10b-boundary] running oracle…");
    let mut sess_a = MetalSession::fresh(&ctx, &mm, cap).expect("sess A");
    let oracle_t = std::time::Instant::now();
    let mut last_a = Vec::new();
    for (i, &tid) in token_ids.iter().enumerate() {
        last_a = mf
            .single_token(tid, i as u32, &mut sess_a)
            .expect("oracle forward");
    }
    let oracle_ms = oracle_t.elapsed().as_secs_f64() * 1e3;
    eprintln!("[prefill-vs-single-a10b-boundary] oracle wall: {oracle_ms:.1} ms");

    eprintln!(
        "[prefill-vs-single-a10b-boundary] running prefill_tokens (P={p}, chunks={})…",
        total_n.div_ceil(p)
    );
    let mut sess_b = MetalSession::fresh(&ctx, &mm, cap).expect("sess B");
    let mut layer_scratch = MetalDFlashLayerMajorScratch::fresh_prefill_with_matrix_max_pos(
        &ctx, &mm, p as u32, total_n,
    )
    .expect("layer scratch");
    let exp_t = std::time::Instant::now();
    let last_b = prefill_tokens_with_multi_hidden(
        &mf,
        &token_ids,
        0,
        &mut sess_b,
        &mut layer_scratch,
        &[],
        None,
    )
    .expect("prefill");
    if let Ok(raw_cap) = std::env::var("QWEN_PREFILL_ATTN_MATRIX_QUERY_CAP") {
        let query_cap: usize = raw_cap.parse().expect("valid query cap");
        assert_eq!(layer_scratch.attn_matrix_query_rows(), p.min(query_cap));
        if p > query_cap {
            assert!(
                layer_scratch.attn_matrix_tiled_layer_calls() > 0,
                "query-cap gate did not tile A10B matrix attention"
            );
        }
        let overlay_disabled = std::env::var("QWEN_PREFILL_ATTN_GDN_SCRATCH_OVERLAY")
            .as_deref()
            .map(|value| matches!(value, "0" | "false" | "FALSE" | "no" | "NO"))
            .unwrap_or(false);
        if !overlay_disabled {
            assert!(
                layer_scratch.prefill_scratch_overlay_stats().is_some(),
                "query-cap gate did not activate A10B scratch overlay"
            );
        }
    }
    let exp_ms = exp_t.elapsed().as_secs_f64() * 1e3;
    eprintln!("[prefill-vs-single-a10b-boundary] prefill wall: {exp_ms:.1} ms");
    eprintln!(
        "[prefill-vs-single-a10b-boundary] SPEEDUP: {:.2}×",
        oracle_ms / exp_ms
    );

    assert_eq!(last_a.len(), last_b.len(), "logits len mismatch");
    let cos_logits = cosine_27b(&last_a, &last_b);
    eprintln!("[prefill-vs-single-a10b-boundary] cos(final logits)={cos_logits:.6}");
    assert!(cos_logits >= 0.999, "logits cos={cos_logits} < 0.999");

    assert_eq!(sess_a.gdn_state.len(), sess_b.gdn_state.len());
    let mut gdn_state_min_cos = f64::INFINITY;
    let mut gdn_conv_min_cos = f64::INFINITY;
    for gi in 0..sess_a.gdn_state.len() {
        let a_state = read_tensor_f32_27b(&sess_a.gdn_state[gi]);
        let b_state = read_tensor_f32_27b(&sess_b.gdn_state[gi]);
        gdn_state_min_cos = gdn_state_min_cos.min(cosine_27b(&a_state, &b_state));
        let a_conv = read_tensor_f32_27b(&sess_a.gdn_conv[gi]);
        let b_conv = read_tensor_f32_27b(&sess_b.gdn_conv[gi]);
        gdn_conv_min_cos = gdn_conv_min_cos.min(cosine_27b(&a_conv, &b_conv));
    }
    eprintln!(
        "[prefill-vs-single-a10b-boundary] GDN state cos_min={gdn_state_min_cos:.6} conv cos_min={gdn_conv_min_cos:.6}"
    );
    assert!(
        gdn_state_min_cos >= 0.999,
        "GDN state cos_min={gdn_state_min_cos} < 0.999"
    );
    assert!(
        gdn_conv_min_cos >= 0.999,
        "GDN conv cos_min={gdn_conv_min_cos} < 0.999"
    );

    assert_eq!(sess_a.kv_k.len(), sess_b.kv_k.len());
    let kv_prefix_elems = total_n * (arch.n_kv_heads as usize * arch.attn_head_dim as usize);
    let mut kv_k_min_cos = f64::INFINITY;
    let mut kv_v_min_cos = f64::INFINITY;
    for ai in 0..sess_a.kv_k.len() {
        assert_eq!(sess_a.kv_n_pos[ai], total_n, "oracle kv_n_pos[{ai}] != T");
        assert_eq!(sess_b.kv_n_pos[ai], total_n, "prefill kv_n_pos[{ai}] != T");
        let a_k = read_kv_prefix_f16_to_f32(&sess_a.kv_k[ai], kv_prefix_elems);
        let b_k = read_kv_prefix_f16_to_f32(&sess_b.kv_k[ai], kv_prefix_elems);
        let a_v = read_kv_prefix_f16_to_f32(&sess_a.kv_v[ai], kv_prefix_elems);
        let b_v = read_kv_prefix_f16_to_f32(&sess_b.kv_v[ai], kv_prefix_elems);
        kv_k_min_cos = kv_k_min_cos.min(cosine_27b(&a_k, &b_k));
        kv_v_min_cos = kv_v_min_cos.min(cosine_27b(&a_v, &b_v));
    }
    eprintln!(
        "[prefill-vs-single-a10b-boundary] KV K cos_min={kv_k_min_cos:.6} V cos_min={kv_v_min_cos:.6}"
    );
    assert!(kv_k_min_cos >= 0.999, "KV K cos_min={kv_k_min_cos} < 0.999");
    assert!(kv_v_min_cos >= 0.999, "KV V cos_min={kv_v_min_cos} < 0.999");
}

#[test]
#[ignore]
fn prefill_tokens_matches_single_token_loop_122b_a10b_moe_packed_attn_active_shapes() {
    let model_path = concat!(
        "/Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/",
        "Qwen3.5-122B-A10B-UD-Q4_K_XL-00001-of-00003.gguf"
    );
    if !std::path::Path::new(model_path).exists() {
        eprintln!("[prefill-vs-single-a10b-packed] skipped — target GGUF missing");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("metal init: {e}"),
    };

    eprintln!("[prefill-vs-single-a10b-packed] loading 122B A10B…");
    let g = GgufFile::open(model_path).expect("open target");
    let m = Model::from_gguf(&g).expect("load target");
    assert_eq!(m.arch.kind, qwen_llm::model::ArchKind::Moe);
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx, &mm);
    let arch = &mm.arch;

    let run_scenario = |label: &str, total_n: usize, p: usize, prefix_len: usize| {
        let n_total_with_prefix = prefix_len + total_n;
        let all_tokens: Vec<i32> = (0..n_total_with_prefix)
            .map(|i| ((i * 19 + 7) % (arch.vocab_size as usize - 1)) as i32 + 1)
            .collect();
        let prefix_tokens = &all_tokens[..prefix_len];
        let token_ids = &all_tokens[prefix_len..];
        let cap = n_total_with_prefix + 16;

        eprintln!(
            "[prefill-vs-single-a10b-packed] {label}: prefix={prefix_len} T={total_n} P={p} chunks={}",
            total_n.div_ceil(p)
        );

        let mut sess_a = MetalSession::fresh(&ctx, &mm, cap).expect("sess A");
        for (i, &tid) in prefix_tokens.iter().enumerate() {
            mf.single_token(tid, i as u32, &mut sess_a)
                .expect("oracle prefix advance");
        }
        let mut last_a = Vec::new();
        for (i, &tid) in token_ids.iter().enumerate() {
            last_a = mf
                .single_token(tid, (prefix_len + i) as u32, &mut sess_a)
                .expect("oracle forward");
        }

        let mut sess_b = MetalSession::fresh(&ctx, &mm, cap).expect("sess B");
        for (i, &tid) in prefix_tokens.iter().enumerate() {
            mf.single_token(tid, i as u32, &mut sess_b)
                .expect("experimental prefix advance");
        }
        let mut layer_scratch = MetalDFlashLayerMajorScratch::fresh_prefill_with_matrix_max_pos(
            &ctx,
            &mm,
            p as u32,
            n_total_with_prefix,
        )
        .expect("layer scratch");
        let last_b = prefill_tokens_with_multi_hidden(
            &mf,
            token_ids,
            prefix_len as u32,
            &mut sess_b,
            &mut layer_scratch,
            &[],
            None,
        )
        .expect("prefill");

        assert_eq!(last_a.len(), last_b.len(), "{label}: logits len mismatch");
        let cos_logits = cosine_27b(&last_a, &last_b);
        eprintln!("[prefill-vs-single-a10b-packed] {label}: cos(logits)={cos_logits:.6}");
        assert!(
            cos_logits >= 0.999,
            "{label}: logits cos={cos_logits} < 0.999"
        );

        assert_eq!(sess_a.gdn_state.len(), sess_b.gdn_state.len());
        let mut gdn_state_min_cos = f64::INFINITY;
        let mut gdn_conv_min_cos = f64::INFINITY;
        for gi in 0..sess_a.gdn_state.len() {
            let a_state = read_tensor_f32_27b(&sess_a.gdn_state[gi]);
            let b_state = read_tensor_f32_27b(&sess_b.gdn_state[gi]);
            gdn_state_min_cos = gdn_state_min_cos.min(cosine_27b(&a_state, &b_state));
            let a_conv = read_tensor_f32_27b(&sess_a.gdn_conv[gi]);
            let b_conv = read_tensor_f32_27b(&sess_b.gdn_conv[gi]);
            gdn_conv_min_cos = gdn_conv_min_cos.min(cosine_27b(&a_conv, &b_conv));
        }
        assert!(
            gdn_state_min_cos >= 0.999,
            "{label}: GDN state cos_min={gdn_state_min_cos} < 0.999"
        );
        assert!(
            gdn_conv_min_cos >= 0.999,
            "{label}: GDN conv cos_min={gdn_conv_min_cos} < 0.999"
        );

        assert_eq!(sess_a.kv_n_pos.len(), sess_b.kv_n_pos.len());
        let kv_prefix_elems =
            n_total_with_prefix * (arch.n_kv_heads as usize * arch.attn_head_dim as usize);
        let mut kv_k_min_cos = f64::INFINITY;
        let mut kv_v_min_cos = f64::INFINITY;
        for ai in 0..sess_a.kv_k.len() {
            assert_eq!(
                sess_a.kv_n_pos[ai], sess_b.kv_n_pos[ai],
                "{label}: kv_n_pos[{ai}] mismatch ({} vs {})",
                sess_a.kv_n_pos[ai], sess_b.kv_n_pos[ai]
            );
            let a_k = read_kv_prefix_f16_to_f32(&sess_a.kv_k[ai], kv_prefix_elems);
            let b_k = read_kv_prefix_f16_to_f32(&sess_b.kv_k[ai], kv_prefix_elems);
            let a_v = read_kv_prefix_f16_to_f32(&sess_a.kv_v[ai], kv_prefix_elems);
            let b_v = read_kv_prefix_f16_to_f32(&sess_b.kv_v[ai], kv_prefix_elems);
            kv_k_min_cos = kv_k_min_cos.min(cosine_27b(&a_k, &b_k));
            kv_v_min_cos = kv_v_min_cos.min(cosine_27b(&a_v, &b_v));
        }
        assert!(
            kv_k_min_cos >= 0.999,
            "{label}: KV K cos_min={kv_k_min_cos} < 0.999"
        );
        assert!(
            kv_v_min_cos >= 0.999,
            "{label}: KV V cos_min={kv_v_min_cos} < 0.999"
        );
    };

    run_scenario("active-shape prefix4096 T4 P8", 4, 8, 4096);
    run_scenario("threshold-cross prefix4095 T2 P8", 2, 8, 4095);
    run_scenario("single-row prefix4096 T1 P8", 1, 8, 4096);
    run_scenario("full-tile prefix4096 T8 P8", 8, 8, 4096);
    run_scenario("multi-chunk prefix4096 T12 P8", 12, 8, 4096);
}

#[test]
fn prefill_tokens_moe_hidden_capture_matches_p1_oracle_35b_a3b() {
    let model_path = qwen_llm::test_fixtures::A3B_Q4_K_M.path();
    if !std::path::Path::new(model_path).exists() {
        eprintln!("[prefill-hidden-a3b] skipped — target GGUF missing");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("metal init: {e}"),
    };

    let g = GgufFile::open(model_path).expect("open target");
    let m = Model::from_gguf(&g).expect("load target");
    assert_eq!(m.arch.kind, qwen_llm::model::ArchKind::Moe);
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx, &mm);
    let arch = &mm.arch;
    let h = arch.hidden_size as usize;

    let total_n: usize = 13;
    let p: usize = 8;
    let token_ids: Vec<i32> = (0..total_n)
        .map(|i| ((i * 17 + 11) % (arch.vocab_size as usize - 1)) as i32 + 1)
        .collect();
    let capture_layers: Vec<u32> = vec![1, 3, 16, 27, 39];
    let k = capture_layers.len();
    let cap = total_n + 16;

    eprintln!("[prefill-hidden-a3b] running P=1 packed oracle…");
    let mut sess_a = MetalSession::fresh(&ctx, &mm, cap).expect("sess A");
    let mut scratch_a =
        MetalDFlashLayerMajorScratch::fresh_prefill_with_matrix_max_pos(&ctx, &mm, 1, total_n)
            .expect("scratch A");
    let h_dst_a = MetalTensor::zeros_f32(&ctx, vec![(k * h) as u64]).expect("h_dst_a");
    let mut accum_a = vec![0.0f32; total_n * k * h];
    let oracle_t = std::time::Instant::now();
    let mut last_a = Vec::new();
    for (i, &tid) in token_ids.iter().enumerate() {
        last_a = prefill_tokens_with_multi_hidden(
            &mf,
            &[tid],
            i as u32,
            &mut sess_a,
            &mut scratch_a,
            &capture_layers,
            Some(&h_dst_a),
        )
        .expect("p1 oracle");
        unsafe {
            let src = h_dst_a.buffer.contents().as_ptr() as *const f32;
            std::ptr::copy_nonoverlapping(
                src,
                accum_a[i * k * h..(i + 1) * k * h].as_mut_ptr(),
                k * h,
            );
        }
    }
    let oracle_ms = oracle_t.elapsed().as_secs_f64() * 1e3;
    eprintln!("[prefill-hidden-a3b] oracle wall: {oracle_ms:.1} ms");

    eprintln!(
        "[prefill-hidden-a3b] running packed prefill (P={p}, chunks={})…",
        total_n.div_ceil(p)
    );
    let mut sess_b = MetalSession::fresh(&ctx, &mm, cap).expect("sess B");
    let mut scratch_b = MetalDFlashLayerMajorScratch::fresh_prefill_with_matrix_max_pos(
        &ctx, &mm, p as u32, total_n,
    )
    .expect("scratch B");
    let h_dst_b = MetalTensor::zeros_f32(&ctx, vec![(total_n * k * h) as u64]).expect("h_dst_b");
    let exp_t = std::time::Instant::now();
    let last_b = prefill_tokens_with_multi_hidden(
        &mf,
        &token_ids,
        0,
        &mut sess_b,
        &mut scratch_b,
        &capture_layers,
        Some(&h_dst_b),
    )
    .expect("packed prefill");
    let exp_ms = exp_t.elapsed().as_secs_f64() * 1e3;
    eprintln!("[prefill-hidden-a3b] packed wall: {exp_ms:.1} ms");

    let cos_logits = cosine_27b(&last_a, &last_b);
    eprintln!("[prefill-hidden-a3b] cos(final logits)={cos_logits:.6}");
    assert!(cos_logits >= 0.999, "logits cos={cos_logits} < 0.999");

    let mut accum_b = vec![0.0f32; total_n * k * h];
    unsafe {
        let src = h_dst_b.buffer.contents().as_ptr() as *const f32;
        std::ptr::copy_nonoverlapping(src, accum_b.as_mut_ptr(), total_n * k * h);
    }
    let mut min_cos = f64::INFINITY;
    let mut worst_pos = (0usize, 0usize);
    for t in 0..total_n {
        for k_idx in 0..k {
            let off = (t * k + k_idx) * h;
            let c = cosine_27b(&accum_a[off..off + h], &accum_b[off..off + h]);
            if c < min_cos {
                min_cos = c;
                worst_pos = (t, k_idx);
            }
        }
    }
    eprintln!(
        "[prefill-hidden-a3b] hidden cos_min={min_cos:.6} (worst at token={}, capture_layer={})",
        worst_pos.0, worst_pos.1
    );
    assert!(min_cos >= 0.999, "hidden capture cos_min={min_cos} < 0.999");

    let kv_prefix_elems = total_n * (arch.n_kv_heads as usize * arch.attn_head_dim as usize);
    for ai in 0..sess_a.kv_k.len() {
        assert_eq!(sess_a.kv_n_pos[ai], total_n, "oracle kv_n_pos[{ai}] != T");
        assert_eq!(sess_b.kv_n_pos[ai], total_n, "packed kv_n_pos[{ai}] != T");
        let a_k = read_kv_prefix_f16_to_f32(&sess_a.kv_k[ai], kv_prefix_elems);
        let b_k = read_kv_prefix_f16_to_f32(&sess_b.kv_k[ai], kv_prefix_elems);
        let a_v = read_kv_prefix_f16_to_f32(&sess_a.kv_v[ai], kv_prefix_elems);
        let b_v = read_kv_prefix_f16_to_f32(&sess_b.kv_v[ai], kv_prefix_elems);
        assert!(
            cosine_27b(&a_k, &b_k) >= 0.999,
            "KV K mismatch at layer {ai}"
        );
        assert!(
            cosine_27b(&a_v, &b_v) >= 0.999,
            "KV V mismatch at layer {ai}"
        );
    }
}

fn cosine_27b(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..a.len() {
        dot += a[i] as f64 * b[i] as f64;
        na += (a[i] as f64).powi(2);
        nb += (b[i] as f64).powi(2);
    }
    dot / (na.sqrt() * nb.sqrt() + 1e-30)
}

struct GdnCosSummary {
    state_min_cos: f64,
    state_worst_layer: usize,
    state_max_abs: f32,
    conv_min_cos: f64,
    conv_worst_layer: usize,
    conv_max_abs: f32,
}

fn compare_gdn_state_conv_27b(sess_a: &MetalSession, sess_b: &MetalSession) -> GdnCosSummary {
    assert_eq!(sess_a.gdn_state.len(), sess_b.gdn_state.len());
    assert_eq!(sess_a.gdn_conv.len(), sess_b.gdn_conv.len());

    let mut summary = GdnCosSummary {
        state_min_cos: f64::INFINITY,
        state_worst_layer: 0,
        state_max_abs: 0.0,
        conv_min_cos: f64::INFINITY,
        conv_worst_layer: 0,
        conv_max_abs: 0.0,
    };

    for gi in 0..sess_a.gdn_state.len() {
        let a_state = read_tensor_f32_27b(&sess_a.gdn_state[gi]);
        let b_state = read_tensor_f32_27b(&sess_b.gdn_state[gi]);
        let state_cos = cosine_27b(&a_state, &b_state);
        if state_cos < summary.state_min_cos {
            summary.state_min_cos = state_cos;
            summary.state_worst_layer = gi;
            summary.state_max_abs = max_abs_delta_27b(&a_state, &b_state);
        }

        let a_conv = read_tensor_f32_27b(&sess_a.gdn_conv[gi]);
        let b_conv = read_tensor_f32_27b(&sess_b.gdn_conv[gi]);
        let conv_cos = cosine_27b(&a_conv, &b_conv);
        if conv_cos < summary.conv_min_cos {
            summary.conv_min_cos = conv_cos;
            summary.conv_worst_layer = gi;
            summary.conv_max_abs = max_abs_delta_27b(&a_conv, &b_conv);
        }
    }

    summary
}

fn max_abs_delta_27b(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn argmax_i32_27b(xs: &[f32]) -> i32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in xs.iter().enumerate() {
        if v > best_v {
            best = i;
            best_v = v;
        }
    }
    best as i32
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
struct ContinuationMismatch {
    step: usize,
    oracle_argmax: i32,
    prefill_argmax: i32,
    oracle_rank_in_prefill: usize,
    prefill_rank_in_oracle: usize,
    oracle_margin_over_prefill: f32,
    prefill_margin_over_oracle: f32,
    oracle_top2_margin: f32,
    prefill_top2_margin: f32,
}

impl ContinuationMismatch {
    fn from_logits(
        step: usize,
        oracle_argmax: i32,
        prefill_argmax: i32,
        oracle_logits: &[f32],
        prefill_logits: &[f32],
    ) -> Self {
        let oracle_idx = oracle_argmax as usize;
        let prefill_idx = prefill_argmax as usize;
        Self {
            step,
            oracle_argmax,
            prefill_argmax,
            oracle_rank_in_prefill: rank_of_token_27b(prefill_logits, oracle_idx),
            prefill_rank_in_oracle: rank_of_token_27b(oracle_logits, prefill_idx),
            oracle_margin_over_prefill: oracle_logits[oracle_idx] - oracle_logits[prefill_idx],
            prefill_margin_over_oracle: prefill_logits[prefill_idx] - prefill_logits[oracle_idx],
            oracle_top2_margin: top2_margin_27b(oracle_logits),
            prefill_top2_margin: top2_margin_27b(prefill_logits),
        }
    }
}

fn rank_of_token_27b(xs: &[f32], token_idx: usize) -> usize {
    let target = xs[token_idx];
    1 + xs.iter().filter(|&&v| v > target).count()
}

fn top2_margin_27b(xs: &[f32]) -> f32 {
    let mut best = f32::NEG_INFINITY;
    let mut second = f32::NEG_INFINITY;
    for &v in xs {
        if v > best {
            second = best;
            best = v;
        } else if v > second {
            second = v;
        }
    }
    best - second
}

struct KvCosSummary {
    k_min_cos: f64,
    k_worst_layer: usize,
    k_max_abs: f32,
    v_min_cos: f64,
    v_worst_layer: usize,
    v_max_abs: f32,
}

fn compare_kv_prefix_27b(
    sess_a: &MetalSession,
    sess_b: &MetalSession,
    kv_prefix_elems: usize,
    expected_n_pos: usize,
) -> KvCosSummary {
    assert_eq!(sess_a.kv_k.len(), sess_b.kv_k.len());
    assert_eq!(sess_a.kv_v.len(), sess_b.kv_v.len());

    let mut summary = KvCosSummary {
        k_min_cos: f64::INFINITY,
        k_worst_layer: 0,
        k_max_abs: 0.0,
        v_min_cos: f64::INFINITY,
        v_worst_layer: 0,
        v_max_abs: 0.0,
    };

    for ai in 0..sess_a.kv_k.len() {
        assert_eq!(
            sess_a.kv_n_pos[ai], expected_n_pos,
            "oracle kv_n_pos[{ai}] != T"
        );
        assert_eq!(
            sess_b.kv_n_pos[ai], expected_n_pos,
            "prefill kv_n_pos[{ai}] != T"
        );

        let a_k = read_kv_prefix_f16_to_f32(&sess_a.kv_k[ai], kv_prefix_elems);
        let b_k = read_kv_prefix_f16_to_f32(&sess_b.kv_k[ai], kv_prefix_elems);
        let k_cos = cosine_27b(&a_k, &b_k);
        if k_cos < summary.k_min_cos {
            summary.k_min_cos = k_cos;
            summary.k_worst_layer = ai;
            summary.k_max_abs = max_abs_delta_27b(&a_k, &b_k);
        }

        let a_v = read_kv_prefix_f16_to_f32(&sess_a.kv_v[ai], kv_prefix_elems);
        let b_v = read_kv_prefix_f16_to_f32(&sess_b.kv_v[ai], kv_prefix_elems);
        let v_cos = cosine_27b(&a_v, &b_v);
        if v_cos < summary.v_min_cos {
            summary.v_min_cos = v_cos;
            summary.v_worst_layer = ai;
            summary.v_max_abs = max_abs_delta_27b(&a_v, &b_v);
        }
    }

    summary
}

fn read_tensor_f32_27b(t: &MetalTensor) -> Vec<f32> {
    let n = t.n_elements() as usize;
    let mut out = vec![0.0f32; n];
    unsafe {
        let src = t.buffer.contents().as_ptr() as *const f32;
        std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), n);
    }
    out
}

fn read_kv_prefix_f16_to_f32(t: &MetalTensor, n_f16: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n_f16];
    unsafe {
        let src = t.buffer.contents().as_ptr() as *const u16;
        for (i, slot) in out.iter_mut().enumerate().take(n_f16) {
            let bits = *src.add(i);
            *slot = half::f16::from_bits(bits).to_f32();
        }
    }
    out
}
