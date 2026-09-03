//! DFlash / DFlash2 decode bench and verify policy.

use super::*;

pub(crate) fn run_dflash_lazy(args: DflashLazyArgs) -> Result<()> {
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

    // v0.77: the DFlash path predates the prefetch subsystem (H5-era) —
    // apply the runtime's residency-gated ColdOnly warmer to BOTH files
    // before the copy loops (cold mmap page-in ~0.8 GB/s vs ~6 GB/s
    // parallel pread; no-op when already resident). Also keeps cold-start
    // bench runs from polluting decode numbers with page-in time.
    let prefetch_cfg = qwen_llm::runtime::LoadedModelConfig::default();
    let target_pf = qwen_llm::runtime::prefetch_opened_gguf(&target_g, &prefetch_cfg);
    let drafter_pf = qwen_llm::runtime::prefetch_opened_gguf(&drafter_g, &prefetch_cfg);
    eprintln!(
        "[dflash] prefetch: target {:.0} ms, drafter {:.0} ms ({:?})",
        target_pf.total_wall.as_secs_f64() * 1e3,
        drafter_pf.total_wall.as_secs_f64() * 1e3,
        drafter_pf.action,
    );

    let mm = MetalModel::load(&ctx, &target_g, &target_m).context("metal-load target")?;
    let t_head = Instant::now();
    let mhead = MetalDFlashHead::load(&ctx, &drafter_g, &head).context("metal-load drafter")?;
    eprintln!(
        "[dflash] drafter metal-load {:.0} ms",
        t_head.elapsed().as_secs_f64() * 1e3
    );
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
        shutdown::checkpoint()?;
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
pub(crate) fn run_dflash(args: DflashArgs) -> Result<()> {
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
    // v0.77: residency-gated prefetch for both files (see run_dflash_lazy
    // for rationale; the DFlash path predates the prefetch subsystem).
    let prefetch_cfg = qwen_llm::runtime::LoadedModelConfig::default();
    let target_pf = qwen_llm::runtime::prefetch_opened_gguf(&target_g, &prefetch_cfg);
    let drafter_pf = qwen_llm::runtime::prefetch_opened_gguf(&drafter_g, &prefetch_cfg);
    eprintln!(
        "[dflash] prefetch: target {:.0} ms, drafter {:.0} ms ({:?})",
        target_pf.total_wall.as_secs_f64() * 1e3,
        drafter_pf.total_wall.as_secs_f64() * 1e3,
        drafter_pf.action,
    );
    let mm = MetalModel::load(&ctx, &target_g, &target_m).context("metal-load target")?;
    let t_head = Instant::now();
    let mhead = MetalDFlashHead::load(&ctx, &drafter_g, &head).context("metal-load drafter")?;
    eprintln!(
        "[dflash] drafter metal-load {:.0} ms",
        t_head.elapsed().as_secs_f64() * 1e3
    );
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

    let target_cap = n_prompt + tokens + 32;
    let mut target_session =
        MetalSession::fresh(&ctx, &mm, target_cap).context("target session")?;

    // Production DFlash scratch buffers.
    let mut verify_scratch =
        MetalDFlashVerifyScratch::fresh(&ctx, &mm, cfg.block_size, k_layers as u32)
            .context("verify scratch")?;
    let mut layer_scratch =
        MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, cfg.block_size).context("layer scratch")?;

    // Prompt prefill has production geometry independent of the physical
    // verifier N. Reusing `layer_scratch` here would force N=8 prompt chunks
    // and lose matrix-attention coverage after the first chunk.
    let prefill_chunk = default_prefill_chunk(target_m.arch.kind, n_prompt);
    let mut prefill_scratch = fresh_prefill_scratch_for_prompt(&ctx, &mm, prefill_chunk, n_prompt)?;
    let capture_limit = qwen_llm::metal_dflash::dflash_capture_window_limit(&mhead);
    let (capture_start, capture_tokens) =
        qwen_llm::metal_dflash::dflash_capture_window_span(n_prompt, capture_limit);
    let dflash_cap = capture_tokens + tokens + 32;
    let mut dsess = MetalDFlashSession::fresh(&ctx, &mhead, h_target as u64, v as u64, dflash_cap)
        .context("dflash session")?;

    // Capture only the suffix visible to an all-SWA drafter. Full-attention
    // heads retain full-prompt capture through `capture_limit=usize::MAX`.
    let prefill_hidden_dst =
        MetalTensor::zeros_f32(&ctx, vec![(capture_tokens * n_target_features) as u64])
            .context("prefill_hidden_dst")?;

    // ---------- Prompt prefill ----------
    let t_prefill = Instant::now();
    if capture_start > 0 {
        prefill_tokens_prompt_only_profiled(
            &mf,
            &prompt_ids[..capture_start],
            0,
            &mut target_session,
            &mut prefill_scratch,
        )
        .context("plain prompt prefix prefill")?;
    }
    let last_logits = prefill_tokens_with_multi_hidden(
        &mf,
        &prompt_ids[capture_start..],
        capture_start as u32,
        &mut target_session,
        &mut prefill_scratch,
        &head.target_layer_ids,
        Some(&prefill_hidden_dst),
    )
    .context("prefill_tokens_with_multi_hidden")?;
    dsess
        .append_target_ctx_columns_contiguous_now(
            &ctx,
            &prefill_hidden_dst,
            capture_start as u32,
            capture_tokens,
            n_target_features,
        )
        .context("append prefill ctx columns (batched)")?;
    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1e3;
    drop(prefill_scratch);
    drop(prefill_hidden_dst);
    eprintln!(
        "[dflash] prefill {n_prompt} tokens in {prefill_ms:.1} ms (chunk={prefill_chunk}, capture_start={capture_start}, capture_tokens={capture_tokens})"
    );

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
    // v0.77 α-backoff state (adaptive + N≤8 only): trailing per-step
    // accepted-draft counts. Windowed mean below break-even → terminal Off.
    let mut alpha_window: Vec<usize> = Vec::with_capacity(DFLASH2_ALPHA_WINDOW + 1);
    let mut alpha_backoff_triggered = false;

    let mut decoder = DFlashDecoder::new(&mf, &mhead, dsess);
    if profile {
        decoder.session.enable_phase_timers();
    }

    // H5.6 M1c: per-component wall accounting across the outer loop.
    // Terminal-step bias note: the loop can break mid-step (accept hits
    // `tokens`), so append/restore of the final step may be skipped; run
    // with --tokens >= 256 when using these numbers for economics.
    let mut acct_draft_ms = 0.0f64;
    let mut acct_draft_first_ms = 0.0f64;
    let mut acct_verify_ms = 0.0f64;
    let mut acct_append_ms = 0.0f64;
    let mut acct_restore_ms = 0.0f64;
    // v0.77 verify-microbench samples: (n_eff, wall ms) per packed_verify
    // call, plus Off-step single_token wall times. Cheap to collect
    // unconditionally; reported only under `--n-policy cycle`.
    // v0.77: (n_eff, wall ms, kv position at the call). The ctx term is
    // recorded per sample so the ctx SLOPE can be fit WITHIN one process.
    // Comparing absolute rows across sessions is forbidden by PERF-TOOLS
    // and produced a phantom +24 ms/8.8K verify slope that survived into
    // the adaptive-policy break-even constant until 2026-08-20.
    let mut verify_samples: Vec<(usize, f64, usize)> = Vec::new();
    let mut off_single_ms: Vec<(f64, usize)> = Vec::new();

    let t_decode = Instant::now();
    'outer: loop {
        shutdown::checkpoint()?;
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
        } else if matches!(n_policy, NPolicy::Cycle) {
            // Verify microbench: fixed interleave; Off steps are the
            // in-process single_token reference.
            const CYCLE_PATTERN: [VerifyMode; 8] = [
                VerifyMode::Spec { n_eff: 8 },
                VerifyMode::Off,
                VerifyMode::Spec { n_eff: 4 },
                VerifyMode::Off,
                VerifyMode::Spec { n_eff: 2 },
                VerifyMode::Off,
                VerifyMode::Spec { n_eff: 1 },
                VerifyMode::Off,
            ];
            CYCLE_PATTERN[steps as usize % CYCLE_PATTERN.len()]
        } else {
            n_policy.for_ctx((processed_pos + 1) as usize, n_block)
        };

        if matches!(mode, VerifyMode::Off) {
            // Off branch: no drafter, no packed_verify, no restore.
            // No drafter ctx update. Terminal — except under Cycle, where
            // Off steps are reference measurements, not a policy decision.
            if !matches!(n_policy, NPolicy::Cycle) {
                spec_disabled = true;
            }
            off_steps += 1;
            steps += 1;
            let single_pos = processed_pos + 1;
            let t_single = Instant::now();
            let logits = mf
                .single_token(carry_tok, single_pos, &mut target_session)
                .context("off-mode single_token")?;
            off_single_ms.push((t_single.elapsed().as_secs_f64() * 1e3, single_pos as usize));
            let next_tok = argmax_i32(&logits);
            // Advance cursors. carry_tok was already emitted at top of
            // the loop; next iter's carry is `next_tok`.
            processed_pos = single_pos;
            carry_tok = next_tok;
            continue;
        }

        // ---- Spec branch: existing drafter + packed_verify + restore ----
        let n_eff = match mode {
            // Clamp to the drafter's block size: the N=16 policy schedules
            // were calibrated for the 3.6 drafter; the DFlash 2 drafter
            // ships block_size=8 and can't fill a 16-token verify chain.
            VerifyMode::Spec { n_eff } => n_eff.min(n_block),
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
        // v0.77: the FIRST draft_block call projects the entire prompt
        // through the drafter's fc + per-layer K/V caches (O(prompt) one-
        // time work; ~290 ms at ctx 8.8K). Folding it into the per-step
        // mean inflated the sweep's high-ctx draft numbers by 10-30%
        // (2026-08-19 adversarial review) — account it separately.
        let draft_elapsed_ms = t_acct.elapsed().as_secs_f64() * 1e3;
        if drafter_calls == 0 {
            acct_draft_first_ms = draft_elapsed_ms;
        } else {
            acct_draft_ms += draft_elapsed_ms;
        }
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
        let verify_elapsed_ms = t_acct.elapsed().as_secs_f64() * 1e3;
        acct_verify_ms += verify_elapsed_ms;
        verify_samples.push((n_eff, verify_elapsed_ms, drafter_pos as usize));
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

        // ---- v0.77 α-backoff (adaptive N≤8 schedule only) ----
        // Content-keyed complement to the ctx guard: the sweep showed
        // win/loss is dominated by acceptance (mean emitted 2.5–5.8 at
        // the SAME ctx), not by ctx. Bail to terminal Off when the
        // trailing window's mean emitted/step can't cover the ctx-keyed
        // draft+verify premium.
        if matches!(n_policy, NPolicy::Adaptive) && n_block <= 8 {
            alpha_window.push(n_accepted);
            if alpha_window.len() > DFLASH2_ALPHA_WINDOW {
                alpha_window.remove(0);
            }
            if alpha_window.len() == DFLASH2_ALPHA_WINDOW {
                let mean_emitted =
                    1.0 + alpha_window.iter().sum::<usize>() as f64 / DFLASH2_ALPHA_WINDOW as f64;
                let threshold =
                    dflash2_n8_breakeven(processed_pos as usize) - DFLASH2_ALPHA_OFF_MARGIN;
                if mean_emitted < threshold {
                    spec_disabled = true;
                    alpha_backoff_triggered = true;
                }
            }
        }

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
        let mut ref_session = MetalSession::fresh(&ctx, &mm, target_cap).context("ref session")?;
        let mut ref_layer_scratch =
            fresh_prefill_scratch_for_prompt(&ctx, &mm, prefill_chunk, n_prompt)
                .context("ref layer scratch")?;
        let t_ref_total = Instant::now();
        let t_ref_prefill = Instant::now();
        if capture_start > 0 {
            prefill_tokens_prompt_only_profiled(
                &mf,
                &prompt_ids[..capture_start],
                0,
                &mut ref_session,
                &mut ref_layer_scratch,
            )
            .context("ref plain prompt prefix prefill")?;
        }
        let last_logits_ref = prefill_tokens_with_multi_hidden(
            &mf,
            &prompt_ids[capture_start..],
            capture_start as u32,
            &mut ref_session,
            &mut ref_layer_scratch,
            &[],
            None,
        )?;
        let ref_prefill_ms = t_ref_prefill.elapsed().as_secs_f64() * 1e3;
        drop(ref_layer_scratch);
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
        let acct_sum =
            acct_draft_ms + acct_draft_first_ms + acct_verify_ms + acct_append_ms + acct_restore_ms;
        // Steady-state draft mean excludes the first call (one-time prompt
        // projection through the drafter caches; reported separately).
        let draft_steady_calls = drafter_calls.saturating_sub(1).max(1) as f64;
        eprintln!();
        eprintln!("[dflash] === step accounting (H5.6 M1c) ===");
        eprintln!(
            "[dflash] per-step means over {steps} steps: draft {:.1} ms (steady-state; first call {:.1} ms excluded) | verify {:.1} ms | append {:.1} ms | restore {:.1} ms | unaccounted {:.1} ms | TOTAL {:.1} ms",
            acct_draft_ms / draft_steady_calls,
            acct_draft_first_ms,
            acct_verify_ms / s,
            acct_append_ms / s,
            acct_restore_ms / s,
            (decode_ms - acct_sum).max(0.0) / s,
            decode_ms / s,
        );
    }

    // v0.77 verify microbench report (interleaved same-process samples).
    if matches!(n_policy, NPolicy::Cycle) {
        let stats = |xs: &mut Vec<f64>| -> (usize, f64, f64, f64) {
            xs.sort_by(|a, b| a.total_cmp(b));
            let n = xs.len();
            let mean = xs.iter().sum::<f64>() / n.max(1) as f64;
            let med = if n == 0 { 0.0 } else { xs[n / 2] };
            let min = xs.first().copied().unwrap_or(0.0);
            (n, mean, med, min)
        };
        // Least-squares slope of ms vs kv position, fit WITHIN this
        // process. Only valid where the samples actually span a ctx
        // range, so the span is reported alongside.
        let slope = |pts: &[(f64, usize)]| -> Option<(f64, f64, usize, usize)> {
            if pts.len() < 8 {
                return None;
            }
            let lo = pts.iter().map(|(_, c)| *c).min()?;
            let hi = pts.iter().map(|(_, c)| *c).max()?;
            if hi.saturating_sub(lo) < 256 {
                return None;
            }
            let n = pts.len() as f64;
            let mean_x = pts.iter().map(|(_, c)| *c as f64).sum::<f64>() / n;
            let mean_y = pts.iter().map(|(m, _)| *m).sum::<f64>() / n;
            let mut num = 0.0;
            let mut den = 0.0;
            for (m, c) in pts {
                let dx = *c as f64 - mean_x;
                num += dx * (*m - mean_y);
                den += dx * dx;
            }
            (den > 0.0).then(|| (num / den, mean_y, lo, hi))
        };
        eprintln!();
        eprintln!("[dflash] === verify microbench (--n-policy cycle) ===");
        for target_n in [8usize, 4, 2, 1] {
            let pts: Vec<(f64, usize)> = verify_samples
                .iter()
                .filter(|(ne, _, _)| *ne == target_n)
                .map(|(_, ms, ctx)| (*ms, *ctx))
                .collect();
            if pts.is_empty() {
                continue;
            }
            let mut xs: Vec<f64> = pts.iter().map(|(m, _)| *m).collect();
            let (n, mean, med, min) = stats(&mut xs);
            eprintln!(
                "[dflash] packed_verify n_eff={target_n}: samples={n} mean {mean:.1} ms | median {med:.1} | min {min:.1}"
            );
            if let Some((k, _, lo, hi)) = slope(&pts) {
                eprintln!(
                    "[dflash]   within-session ctx slope: {:+.2} ms/1K ctx over [{lo}, {hi}]",
                    k * 1000.0
                );
            }
        }
        if !off_single_ms.is_empty() {
            let mut xs: Vec<f64> = off_single_ms.iter().map(|(m, _)| *m).collect();
            let (n, mean, med, min) = stats(&mut xs);
            eprintln!(
                "[dflash] single_token (interleaved ref): samples={n} mean {mean:.1} ms | median {med:.1} | min {min:.1}"
            );
            if let Some((k, _, lo, hi)) = slope(&off_single_ms) {
                eprintln!(
                    "[dflash]   within-session ctx slope: {:+.2} ms/1K ctx over [{lo}, {hi}]",
                    k * 1000.0
                );
            }
        }
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
         (spec_disabled={spec_disabled} terminally, alpha_backoff={alpha_backoff_triggered})"
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
pub(crate) enum VerifyMode {
    Spec { n_eff: usize },
    Off,
}

/// **v0.77**: hard Spec(8)→Off ctx guard for the DFlash 2 N=8 adaptive
/// schedule. Unlike the N=16 schedule there is NO cliff inside the
/// calibrated range: the 2026-08-18 sweep (M4 Max, Qwen3.8-27B Q4_K_M +
/// incoai DFlash2 Q8_0, 64-tok gens) measured Spec(8) at 11.5 t/s even
/// at ctx 8.8K (the old N=16 path fell to 3.65 t/s by ctx 8.2K), with
/// break-even climbing only gently (see `dflash2_n8_breakeven`). The
/// guard sits past the calibrated range, where extrapolated break-even
/// (≈5.6) exceeds what even the best measured content sustains (5.8
/// peak, rare); the α-backoff below handles everything inside it.
pub(crate) const DFLASH2_N8_OFF_CTX: usize = 16384;

/// **v0.77** α-backoff for the DFlash 2 N=8 adaptive schedule: after
/// `DFLASH2_ALPHA_WINDOW` consecutive Spec steps, enter terminal Off if
/// the window's mean emitted tokens/step (1 + α_chain) falls below the
/// ctx-keyed break-even minus a noise margin.
///
/// Rationale: acceptance is strongly CONTENT-dependent — the same ctx
/// band measured mean-emitted 2.5 (code explanation) to 5.8 (repetitive
/// tensor-binding code continuation). A ctx-keyed schedule can't see
/// content; trailing α can. 2026-08-18 sweep, break-even =
/// (draft + verify) / t_single per ctx:
///
///   ctx    draft+verify   off ms/tok   break-even   sample mean-emitted
///   221        142 ms       42.6          3.33        2.9  (0.87×)
///   460        146 ms       41.5          3.55        3.6  (1.00×)
///   896        152 ms       38.8          3.93        2.9  (0.74×)
///  1358        171 ms       42.1          4.05        5.8  (1.43×)
///  2850        180 ms       41.7          4.32        2.7  (0.61×)
///  8835        222 ms       48.3          4.59        2.5  (0.56×)
///
/// Measured throughput ratio matched mean_emitted / break-even within a
/// few percent on every row — note this is a FIT-consistency check, not
/// independent validation (2026-08-19 adversarial review).
///
/// 2026-08-19 corrections from that review:
/// * The high-ctx draft means above include the FIRST `draft_block`
///   call, which projects the whole prompt through fc + per-layer K/V
///   (~290 ms at ctx 8.8K, amortized over ~25 steps ≈ +12 ms/step).
///   Steady-state draft plateaus at the SWA window (2048). The slope is
///   refit on first-call-corrected numbers: ctx/8000.
/// * Window/margin were statistically unsound (window 8 ⇒ SE of the
///   mean ≈ 0.8 emitted/step with per-step accepts σ≈2.3; margin 0.2 ≈
///   0.25 SE ⇒ ~18%/window false-trigger on content winning by +0.5,
///   compounding across overlapping windows). Window 16 (SE ≈ 0.57) +
///   margin 0.6 (≈ 1 SE) puts a +0.5 winner at z ≈ 1.9 ⇒ ~3%/window.
///   Terminal-Off re-probe remains future work.
pub(crate) const DFLASH2_ALPHA_WINDOW: usize = 16;

/// Break-even fit (mean emitted tokens/step at which Spec ties serial).
///
/// **2026-08-20 recalibration.** The previous form was
/// `2.9 + ctx/8000`, whose ctx term was fit on ABSOLUTE verify rows
/// compared ACROSS bench sessions — the exact procedure PERF-TOOLS
/// forbids, and the same phantom slope that produced (and then failed)
/// the V1 chunked-verify projection. Refit from a single-process
/// `--n-policy cycle` run (140+ samples per cell, ctx 464 -> 2062):
///
///   verify(8)    111.5 ms, within-session slope +0.81 ms/1K ctx (+0.73%/1K)
///   single_token  40.2 ms, within-session slope +0.40 ms/1K ctx (+1.00%/1K)
///   verify(1)     41.8 ms, within-session slope +0.28 ms/1K ctx
///
/// Break-even = (draft + verify) / single. Single-token cost grows
/// FASTER in relative terms than verify(8) does, because verify
/// amortizes one KV stream over 8 rows while serial decode re-reads it
/// every token. With the drafter's SWA window plateaued (>= 2048), the
/// derivative is `d(break-even)/d(1K ctx) = -0.011` — flat to slightly
/// DECLINING. Evaluated at both band ends the value is 3.12 / 3.10.
///
/// So the ctx term is dropped, not merely reduced: speculation does not
/// get harder with context on this architecture, it gets marginally
/// easier. The hard `*_OFF_CTX` guard and the content-aware α-backoff
/// remain the safety nets.
///
/// Owed: the within-session slope is only measured over 0.5K-2K. A
/// long-band (8K+) single-process confirmation is still outstanding;
/// until it lands, do not re-introduce a ctx term in either direction.
pub(crate) const DFLASH2_N8_BREAKEVEN_BASE: f64 = 3.1;

/// Trigger margin below break-even, sized ≈ 1 SE of the window mean.
pub(crate) const DFLASH2_ALPHA_OFF_MARGIN: f64 = 0.6;

/// Ctx-keyed Spec(8)-vs-Off break-even in mean emitted tokens/step.
pub(crate) fn dflash2_n8_breakeven(_kv_n_pos: usize) -> f64 {
    DFLASH2_N8_BREAKEVEN_BASE
}

/// **v0.76**: verify-chain length policy as selected by `--n-policy`.
///
/// `Adaptive` is the default — ctx-keyed schedule with `Off`-terminal.
/// The static variants exist for calibration sweeps + manual overrides.
#[derive(Clone, Copy, Debug)]
pub(crate) enum NPolicy {
    Adaptive,
    Static16,
    Static8,
    Static4,
    OffOnly,
    /// **v0.77 verify microbench**: interleave Spec(8)/Off/Spec(4)/Off/
    /// Spec(2)/Off/Spec(1)/Off within one process — same thermal state,
    /// same ctx band, Off steps double as in-process `single_token`
    /// reference timings. Decides the packed-verify fixed-vs-marginal
    /// cost split (2026-08-19 adversarial review, F5: the sweep's
    /// verify(4) > verify(8) anomaly survived an order-swap test, so
    /// only an interleaved same-process measurement settles it).
    /// Note: Off steps don't capture target hiddens, so the drafter's
    /// cross-ctx accumulates holes and α degrades slightly — irrelevant
    /// here because verify cost is target-side and independent of draft
    /// quality.
    Cycle,
}

impl NPolicy {
    pub(crate) fn parse(s: &str) -> Result<Self> {
        match s {
            "adaptive" => Ok(Self::Adaptive),
            "static-16" => Ok(Self::Static16),
            "static-8" => Ok(Self::Static8),
            "static-4" => Ok(Self::Static4),
            "off" => Ok(Self::OffOnly),
            "cycle" => Ok(Self::Cycle),
            other => anyhow::bail!(
                "unknown n-policy {other:?}; expected adaptive, static-16, \
                 static-8, static-4, off, or cycle"
            ),
        }
    }

    /// Choose `VerifyMode` for an outer step at given `kv_n_pos` (the
    /// session's current KV position, i.e. the absolute token position
    /// of the carry token's predecessor) and the drafter's block size
    /// (`n_block`, GGUF-fixed: 16 for the 3.6 DFlash 1 drafter, 8 for
    /// the 3.8 DFlash 2 drafter). The schedules are calibrated against
    /// M4 Max + 27B Q4_K_M; re-run the calibration sweep if hardware/
    /// quant changes (see `qwen-bench dflash --n-policy
    /// static-{16,8,4,off} --prompt ...` for sweep harness).
    pub(crate) fn for_ctx(self, kv_n_pos: usize, n_block: usize) -> VerifyMode {
        match self {
            Self::Static16 => VerifyMode::Spec { n_eff: 16 },
            Self::Static8 => VerifyMode::Spec { n_eff: 8 },
            Self::Static4 => VerifyMode::Spec { n_eff: 4 },
            Self::OffOnly => VerifyMode::Off,
            // Intercepted in the decode loop (pattern is keyed on the step
            // index, which for_ctx doesn't see); this arm is unreachable in
            // practice but kept total.
            Self::Cycle => VerifyMode::Spec { n_eff: 8 },
            Self::Adaptive if n_block <= 8 => {
                // v0.77 DFlash 2 (N=8) schedule — calibrated 2026-08-18
                // on M4 Max + Qwen3.8-27B Q4_K_M + incoai DFlash2 Q8_0
                // (see sweep table at `DFLASH2_N8_OFF_CTX`).
                if kv_n_pos < DFLASH2_N8_OFF_CTX {
                    VerifyMode::Spec { n_eff: 8 }
                } else {
                    VerifyMode::Off
                }
            }
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
