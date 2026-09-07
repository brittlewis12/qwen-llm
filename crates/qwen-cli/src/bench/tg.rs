//! Token-generation, decode, ctx-sweep, and phase benches.

use super::*;

pub(crate) fn run_tg(args: TgArgs) -> Result<()> {
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
    let trace_counts = env_flag_enabled("QWEN_DECODE_TRACE_COUNTS");

    let runtime = Runtime::metal().context("init Runtime")?;
    crate::text_log!(json_mode, "[tg] device: {}", runtime.describe());
    let power = capture_power_snapshot();
    crate::text_log!(json_mode, "[tg] power: {}", power_snapshot_summary(power.as_ref()));

    let loaded = runtime
        .load_model(&model)
        .with_context(|| format!("load {}", model.display()))?;
    shutdown::checkpoint()?;
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

    crate::text_log!(json_mode, 
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
            MetalTensor::zeros_i32(ctx, vec![1]).context("tg pipelined ids ping0")?,
            MetalTensor::zeros_i32(ctx, vec![1]).context("tg pipelined ids ping1")?,
        ])
    } else {
        None
    };
    let argmax_ping = if pipelined {
        Some([
            MetalTensor::zeros_i32(ctx, vec![1]).context("tg pipelined argmax ping0")?,
            MetalTensor::zeros_i32(ctx, vec![1]).context("tg pipelined argmax ping1")?,
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
                shutdown::checkpoint()?;
                // Use `single_token_argmax_profiled` (dispatches dense/MoE
                // internally) so we can sum per-step GPU time. The argmax i32 is
                // discarded; the next input is drawn from the seeded RNG, matching
                // lcpp's `test_gen` (random tokens, no logits coupling).
                let (_argmax, prof) = if concurrent_gdn_proj {
                    if mm.arch.kind == qwen_llm::model::ArchKind::Dense {
                        mf.single_token_argmax_profiled_concurrent_gdn_dense(
                            tok,
                            pos as u32,
                            unsafe { s.metal_session_mut() },
                        )?
                    } else {
                        mf.single_token_argmax_profiled_concurrent_gdn_moe(
                            tok,
                            pos as u32,
                            unsafe { s.metal_session_mut() },
                        )?
                    }
                } else {
                    mf.single_token_argmax_profiled(tok, pos as u32, unsafe {
                        s.metal_session_mut()
                    })?
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
            unsafe { s.metal_session_mut() },
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
                unsafe { s.metal_session_mut() },
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
        shutdown::checkpoint()?;
        let first = next_rand_tok();
        let (wall_ms, gpu_ms, trace) = run_once(first, &mut next_rand_tok).context("tg run")?;
        let ts = n_gen as f64 * 1000.0 / wall_ms;
        wall_samples.push(wall_ms);
        gpu_samples.push(gpu_ms);
        ts_samples.push(ts);
        if let Some(trace) = trace {
            trace_samples.push(trace);
        }
        crate::text_log!(json_mode, 
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

pub(crate) fn run_decode(args: DecodeArgs) -> Result<()> {
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
        generated_token_trace,
        output,
    } = args;
    if runs == 0 {
        return Err(anyhow!("--runs must be >= 1"));
    }
    let prompt =
        prompt.unwrap_or_else(|| "The quick brown fox jumps over the lazy dog".to_string());
    let json_mode = matches!(output, OutputFormat::Json);

    let ctx = MetalContext::new().context("init MetalContext")?;
    crate::text_log!(json_mode, "[bench] device: {}", ctx.describe());
    let power = capture_power_snapshot();
    crate::text_log!(json_mode, "[bench] power: {}", power_snapshot_summary(power.as_ref()));

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
    let mut request_walls: Vec<f64> = Vec::with_capacity(runs);
    // last-rep artifacts; set inside the loop and consumed below.
    let mut last_session: Option<MetalSession> = None;
    let mut prefill_token_ms: Vec<f64> = Vec::with_capacity(ids.len());
    let mut decode_token_ms: Vec<f64> = Vec::with_capacity(tokens);
    let mut per_token_prof: Vec<qwen_llm::metal_forward::TokenProfile> =
        Vec::with_capacity(ids.len() + tokens);
    let mut last_logits: Vec<f32> = Vec::new();
    let mut prefill_logits_for_oracle: Option<Vec<f32>> = None;
    let mut gen_ids: Vec<i32> = Vec::with_capacity(tokens);
    let mut final_next_token: Option<i32> = None;

    for rep in 0..runs {
        shutdown::checkpoint()?;
        // Fresh session per rep so we measure a steady-state cold-cache
        // prefill+decode pair, not the cumulative state of the previous rep.
        drop(last_session.take());
        prefill_token_ms.clear();
        decode_token_ms.clear();
        per_token_prof.clear();
        gen_ids.clear();
        last_logits.clear();
        let request_started = Instant::now();
        let mut s = MetalSession::fresh(&ctx, &mm, cap).context("session run")?;
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
            shutdown::checkpoint()?;
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
        let request_wall = request_started.elapsed().as_secs_f64() * 1e3;
        request_walls.push(request_wall);
        final_next_token = Some(next_tok);
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
        crate::text_log!(json_mode, 
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
        crate::text_log!(json_mode, 
            "[bench] rep {:>2} request {:>8.1} ms",
            rep + 1,
            request_wall
        );
        last_session = Some(s);
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
    let request_wall = sample_mean(&request_walls);
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
    eprintln!("[bench] request wall: {request_wall:.1} ms avg");
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

    if let Some(line) =
        generated_token_trace_line(generated_token_trace, &gen_ids, final_next_token)
    {
        eprintln!("{line}");
    }
    if !gen_ids.is_empty() {
        let text = tok.try_decode(&gen_ids)?;
        eprintln!("[bench] generated: {:?}", text);
    }

    Ok(())
}

pub(crate) fn generated_token_trace_line(
    enabled: bool,
    generated: &[i32],
    final_next_token: Option<i32>,
) -> Option<String> {
    if !enabled {
        return None;
    }
    let mut trace = Vec::with_capacity(generated.len() + usize::from(final_next_token.is_some()));
    trace.extend_from_slice(generated);
    if let Some(token) = final_next_token {
        trace.push(token);
    }
    let encoded = serde_json::to_string(&trace).expect("i32 token trace serialization");
    Some(format!("[bench] generated token trace: {encoded}"))
}

pub(crate) fn run_ctx_sweep(args: CtxSweepArgs) -> Result<()> {
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

pub(crate) fn run_phase(args: PhaseArgs) -> Result<()> {
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
