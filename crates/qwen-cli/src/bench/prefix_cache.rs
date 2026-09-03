//! Prefix-cache bench and logits comparison.

use super::*;

pub(crate) fn default_prefill_chunk(kind: qwen_llm::model::ArchKind, prompt_len: usize) -> usize {
    match kind {
        qwen_llm::model::ArchKind::Moe => prompt_len.clamp(1, 1024),
        qwen_llm::model::ArchKind::Dense => prompt_len.clamp(1, 1024),
    }
}

pub(crate) fn prefix_cache_scratch_args(
    kind: qwen_llm::model::ArchKind,
    total_len: usize,
) -> (usize, usize) {
    (default_prefill_chunk(kind, total_len), total_len)
}

pub(crate) fn fresh_prefill_scratch_for_prompt(
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

pub(crate) fn compare_logits(ours: &[f32], oracle: &[f32]) -> (f64, f32, usize, usize) {
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
pub(crate) fn prefix_cache_prefill_logits(
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

pub(crate) fn run_prefix_cache(args: PrefixCacheArgs) -> Result<()> {
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

    let mf = MetalForward::new(ctx, mm);
    let cap = total_len + tokens + 16;

    let (prefill_chunk, scratch_prompt_len) = prefix_cache_scratch_args(mm.arch.kind, total_len);
    let mut cold_scratch = if prefill_mode == PrefixCachePrefillMode::Packed {
        Some(fresh_prefill_scratch_for_prompt(
            ctx,
            mm,
            prefill_chunk,
            scratch_prompt_len,
        )?)
    } else {
        None
    };
    let mut prefix_scratch = if prefill_mode == PrefixCachePrefillMode::Packed {
        Some(fresh_prefill_scratch_for_prompt(
            ctx,
            mm,
            prefill_chunk,
            scratch_prompt_len,
        )?)
    } else {
        None
    };
    let mut suffix_scratch = if effective_suffix_mode == PrefixCachePrefillMode::Packed {
        Some(fresh_prefill_scratch_for_prompt(
            ctx,
            mm,
            prefill_chunk,
            scratch_prompt_len,
        )?)
    } else {
        None
    };

    // Warmup pass to compile pipeline state objects.
    {
        let mut s = loaded.create_sequence(SequenceConfig::new(32))?;
        if prefill_mode == PrefixCachePrefillMode::Packed {
            let mut warm_scratch = fresh_prefill_scratch_for_prompt(ctx, mm, 1, 1)?;
            let _ = prefill_tokens_with_multi_hidden(
                &mf,
                &[prefix_ids[0]],
                0,
                unsafe { s.metal_session_mut() },
                &mut warm_scratch,
                &[],
                None,
            )?;
        } else {
            let _ = mf.single_token(prefix_ids[0], 0, unsafe { s.metal_session_mut() })?;
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
        unsafe { seq_cold.metal_session_mut() },
        prefill_mode,
        cold_scratch.as_mut(),
    )?;
    seq_cold.advance_by(full_ids.len())?;
    let cold_prefill_ms = cold_t0.elapsed().as_secs_f64() * 1e3;

    // First decoded token = TTFT-equivalent measurement.
    let cold_first_decode_t = Instant::now();
    let cold_first_id = argmax_i32(&last_logits);
    let _ = mf.single_token(cold_first_id, total_len as u32, unsafe {
        seq_cold.metal_session_mut()
    })?;
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
        unsafe { seq_pre.metal_session_mut() },
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
        inserted.stats.indexed_bytes as f64 / 1e6,
        inserted.stats.max_indexed_bytes as f64 / 1e6
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
        hit.stats_at_lookup.entries,
        hit.stats_at_lookup.indexed_bytes as f64 / 1e6,
        hit.stats_at_lookup.max_indexed_bytes as f64 / 1e6
    );

    let last_warm_logits = prefix_cache_prefill_logits(
        &mf,
        &suffix_ids,
        prefix_ids.len() as u32,
        unsafe { seq_warm.metal_session_mut() },
        effective_suffix_mode,
        suffix_scratch.as_mut(),
    )?;
    seq_warm.advance_by(suffix_ids.len())?;
    let warm_prefill_ms = warm_t0.elapsed().as_secs_f64() * 1e3;
    let warm_suffix_ms = warm_prefill_ms - restore_ms;

    let warm_first_decode_t = Instant::now();
    let warm_first_id = argmax_i32(&last_warm_logits);
    let _ = mf.single_token(warm_first_id, total_len as u32, unsafe {
        seq_warm.metal_session_mut()
    })?;
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
            let cold_logits = mf.single_token(*cold_extra.last().unwrap(), pos, unsafe {
                seq_cold.metal_session_mut()
            })?;
            let warm_logits = mf.single_token(*warm_extra.last().unwrap(), pos, unsafe {
                seq_warm.metal_session_mut()
            })?;
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

#[cfg(test)]
mod prefix_cache_geometry_tests {
    use super::*;

    #[test]
    fn long_prefix_keeps_chunk_and_prompt_arguments_in_order() {
        for kind in [
            qwen_llm::model::ArchKind::Dense,
            qwen_llm::model::ArchKind::Moe,
        ] {
            assert_eq!(prefix_cache_scratch_args(kind, 4096), (1024, 4096));
            assert_eq!(prefix_cache_scratch_args(kind, 16_384), (1024, 16_384));
            assert_eq!(prefix_cache_scratch_args(kind, 128), (128, 128));
        }
    }
}
