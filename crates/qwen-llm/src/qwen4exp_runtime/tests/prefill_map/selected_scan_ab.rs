//! One ordinary selected-prefill cell: 8192-prefix + 2048-suffix, context 32768.
//! FLASH_PREFILL_MODEL selects the artifact; FLASH_PREFILL_OUT must be a new JSONL.
//! Internal production benchmark lease. No planner/router override or timed census.
use super::*;
use crate::qwen4exp_qsa::with_packed_qsa_parallel_scan;

const START: usize = 8192;
const ROWS: usize = 2048;
const STOP: usize = START + ROWS;
const CONTEXT: usize = 32768;
const SCALAR: &str = "kernel_deepseek_v4_select_top_k_radix4_ids_f32";
const PARALLEL: &str = "kernel_qwen4exp_qsa_select_top_k_radix4_ids_parallel_scan_f32";

// Normalize only the substituted pipeline and diagnostic tags. Preserve dispatch
// order, encoder, concurrency, family and every grid/thread dimension.
fn normalized(r: &DispatchCensusRow) -> Value {
    let kernel = if r.kernel == PARALLEL {
        SCALAR
    } else {
        &r.kernel
    };
    json!({"family":r.family,"encoder":r.encoder_ordinal,"concurrent":r.encoder_concurrent,
        "kernel":kernel,"grid":[r.grid_width,r.grid_height,r.grid_depth],
        "threads":[r.threads_width,r.threads_height,r.threads_depth],
        "grid_tgs":r.grid_tgs,"tg_threads":r.tg_threads})
}

fn check_endpoint(
    out: &mut std::fs::File,
    label: &str,
    step: usize,
    a: &Endpoint,
    b: &Endpoint,
) -> PacketResult<()> {
    let metrics = comparison(a, b);
    emit(
        out,
        json!({"event":"comparison","label":label,"step":step,
        "reference":"A (warm_A for warm_B, A1 for measured arms)","comparison":metrics}),
    );
    require(
        !a.logits.is_empty()
            && a.logits.len() == b.logits.len()
            && metrics["finite"] == true
            && metrics["endpoint_and_state_concordant"] == true,
        "integer-scan substitution changed full logits or persistent state",
    )
}

fn run(out: &mut std::fs::File) -> PacketResult<()> {
    with_native_artifact(out, |ctx, gguf, out| {
        let config = Qwen4ExpConfig::from_gguf(gguf)?;
        let capacity = Qwen4ExpSessionCapacity::for_forward_limit(&config, CONTEXT)?;
        let tokenizer = Tokenizer::from_gguf(gguf)?;
        let mut text = String::new();
        let mut tokens = Vec::new();
        for _ in 0..64 {
            if !text.is_empty() {
                text.push_str("\n\n");
            }
            text.push_str(CORPUS);
            tokens = tokenizer
                .encode(&text, false)?
                .into_iter()
                .map(u32::try_from)
                .collect::<Result<_, _>>()?;
            if tokens.len() >= STOP + 3 {
                break;
            }
        }
        require(
            tokens.len() >= STOP + 3,
            "retained prose repetition too short",
        )?;
        tokens.truncate(STOP + 3);
        tokens.shrink_to_fit();
        emit(
            out,
            json!({"event":"prompt","source":"retained packed-natural-n512 prose, repeated as whole text",
            "source_sha256":sha256_bytes(CORPUS.as_bytes()),"text_sha256":sha256_bytes(text.as_bytes()),
            "tokenizer_identity":crate::tokenizer::qwen4exp_tokenizer_identity_sha256(gguf)?,
            "add_special_tokens":false,"token_ids":tokens,"token_ids_sha256":sha256_u32_le(b"", &tokens)}),
        );
        drop(text);
        drop(tokenizer);
        emit(
            out,
            json!({"event":"load_begin","context":CONTEXT,"packed_admission_extent":STOP}),
        );
        let mut loaded = Qwen4ExpLoadedModel::load_with_packed_prefill(ctx, gguf, capacity, STOP)?;
        emit(
            out,
            json!({"event":"loaded","admission":format!("{:?}", loaded.admission()),
            "weight_allocation_bytes":loaded.observed_weight_bytes(),"session_allocation_bytes":loaded.observed_session_bytes(),
            "packed_capacity":loaded.packed_prefill_capacity(),"selected_active":loaded.packed_selected_active()}),
        );
        require(
            loaded.packed_prefill_capacity() == Some(ROWS) && loaded.packed_selected_active(),
            "requires ordinary 2048-row selected-capable workspace",
        )?;
        let mut r = loaded.create_runner(ctx)?;
        let layers = r.workspace.geometry().qsa_layers().to_vec();
        require(!layers.is_empty(), "no QSA layers")?;
        for &layer in &layers {
            let g = crate::qwen4exp_qsa::QwenSparseAttentionMetalGeometry::from_config(
                &config,
                layer,
                capacity.qsa_physical_capacity(),
            )?;
            require(
                g.block_capacity() == 8192
                    && g.block_budget() == 512
                    && config.compress_ratios[layer as usize] == 4,
                "requires actual compressor capacity8192 and ratio4",
            )?;
        }
        let allocations = r.workspace.memory_plan().allocations();
        let scores = allocations
            .iter()
            .find(|a| a.name == "packed.qsa.selected.scores")
            .ok_or("missing selected scores allocation")?;
        require(
            scores.logical_bytes == 8192 * 32 * 4,
            "actual selected-score allocation differs",
        )?;
        // checkpoint_for_tests copies persistent allocations, logits and singleton
        // hyper. Endpoint comparisons retain A's four full vectors plus one current
        // vector; hyper readback is temporary. No capture banks or extra GPU buffers.
        let persistent = r.workspace.persistent_state_tensors();
        let checkpoint_bytes = persistent.iter().map(|t| t.n_bytes()).sum::<u64>()
            + (config.vocab_size as u64 + r.weights.geometry.hyper_width() as u64) * 4;
        let endpoint_bytes =
            (5 * config.vocab_size as u64 + r.weights.geometry.hyper_width() as u64) * 4;
        let cpu_payload = checkpoint_bytes + endpoint_bytes + tokens.capacity() as u64 * 4;
        let gate = crate::metal::evaluate_metal_memory_admission_with_cpu_bytes(
            0,
            cpu_payload,
            crate::qwen4exp_text_session::QWEN4EXP_TEXT_SESSION_DYNAMIC_RESERVE_BYTES,
            ctx.memory_signals(),
            true,
        );
        emit(
            out,
            json!({"event":"memory","checkpoint_payload_bytes":checkpoint_bytes,
            "endpoint_vector_peak_bytes":endpoint_bytes,"token_capacity_bytes":tokens.capacity()*4,
            "additional_cpu_payload_bytes":cpu_payload,"cpu_admission":format!("{gate:?}"),
            "accounting":"large CPU payloads; small JSON/hash/Vec metadata excluded; warm census storage measured separately; no fixed diagnostic headroom",
            "gpu_current_allocated_bytes":ctx.current_allocated_size(),
            "session_allocations":allocations.iter().map(|a| json!({"name":a.name,"logical_bytes":a.logical_bytes,"priced_bytes":a.priced_bytes})).collect::<Vec<_>>() }),
        );
        require(
            gate.admitted,
            "normal CPU admission refused checkpoint payload",
        )?;
        emit(
            out,
            json!({"event":"geometry","qsa_layers":layers,"context":CONTEXT,"compressor_capacity":8192,
            "ratio":4,"top_k_blocks":512,"actual_visible_counts_per_query":(START..STOP).map(|p|(p+1)/4).collect::<Vec<_>>(),
            "visible_count_source":"causal (absolute_position+1)/ratio, not allocated capacity; index-score grids checked by warm census",
            "query_band_limit":32,"bands_per_layer":ROWS/32}),
        );
        r.reset()?;
        zero_persistent_state(&r);
        r.prefill(&tokens[..START])?;
        require(r.next_position() == START, "prefix publication mismatch")?;
        let checkpoint = r.workspace.checkpoint_for_tests();
        let mut reference: Vec<Endpoint> = Vec::new();
        let mut warm_census: Vec<DispatchCensusRow> = Vec::new();
        let mut times = Vec::new();
        for (label, candidate, warm) in [
            ("warm_A", false, true),
            ("warm_B", true, true),
            ("A1", false, false),
            ("B1", true, false),
            ("B2", true, false),
            ("A2", false, false),
        ] {
            let clock = Instant::now();
            r.workspace.restore_checkpoint_for_tests(&checkpoint);
            require(
                r.next_position() == START,
                "checkpoint restore frontier mismatch",
            )?;
            let plan = normal_plan(&r, ROWS)?;
            require(
                plan.packed_ranges == vec![START..STOP]
                    && plan.scalar_start == STOP
                    && plan.contains_selection,
                "ordinary selected suffix must be one 2048-row command",
            )?;
            emit(
                out,
                json!({"event":"arm_begin","label":label,"parallel_scan":candidate,"measured":!warm,
                "restore_and_plan_wall_ms":clock.elapsed().as_secs_f64()*1e3,"restore_excluded":true,
                "gpu_current_allocated_bytes":ctx.current_allocated_size()}),
            );
            require(
                !crate::metal::dispatch_census_is_active(),
                "external census active",
            )?;
            if warm {
                crate::metal::dispatch_census_begin();
            }
            let clock = Instant::now();
            let (result, substitutions) = with_packed_qsa_parallel_scan(candidate, || {
                r.prefill_continuation_with_command_checkpoint(&tokens[START..STOP], || Ok(()))
                    .map(|_| ())
            });
            let wall = clock.elapsed().as_secs_f64() * 1e3;
            let census = if warm {
                crate::metal::dispatch_census_take()
            } else {
                Vec::new()
            };
            result?;
            let timing = r
                .last_prefill_timing()
                .ok_or("missing ordinary prefill timing")?;
            let gpu = timing.complete_gpu_ms();
            emit(
                out,
                json!({"event":"arm_complete","label":label,"measured":!warm,
                "ordinary_call_wall_ms":wall,"complete_gpu_ms":gpu,"gpu_samples":timing.gpu_samples,
                "command_count":timing.command_count,"packed_tokens":timing.packed_token_count,
                "substitutions":substitutions,"encode_cpu_ms":timing.encode_cpu_ms,"completion_wait_ms":timing.completion_wait_ms}),
            );
            require(
                r.next_position() == STOP
                    && timing.command_count == 1
                    && timing.gpu_samples == 1
                    && timing.token_count == ROWS
                    && timing.packed_token_count == ROWS
                    && timing.contains_selection
                    && gpu.is_some_and(|v| v.is_finite() && v > 0.0)
                    && wall.is_finite()
                    && wall > 0.0,
                "invalid ordinary timing/publication",
            )?;
            let expected = layers.len() * (ROWS / 32);
            require(
                substitutions == if candidate { expected } else { 0 },
                "selector substitution count mismatch",
            )?;
            if warm {
                let selectors: Vec<_> = census
                    .iter()
                    .filter(|r| r.kernel == SCALAR || r.kernel == PARALLEL)
                    .collect();
                let scores: Vec<_> = census
                    .iter()
                    .filter(|r| r.kernel == "kernel_qwen4exp_qsa_index_scores_packed_4x128_f16")
                    .collect();
                require(
                    selectors.len() == expected
                        && selectors.iter().all(|r| {
                            r.kernel == (if candidate { PARALLEL } else { SCALAR })
                                && r.grid_width == 32
                                && r.grid_height == 1
                                && r.grid_depth == 1
                                && r.tg_threads == 256
                        }),
                    "selector band census mismatch",
                )?;
                require(
                    scores.len() == expected
                        && scores.iter().enumerate().all(|(i, r)| {
                            r.grid_width
                                == ((START + (i % (ROWS / 32) + 1) * 32) / 4).div_ceil(8) as u64
                                && r.grid_height == 32
                        }),
                    "visible-count score grid mismatch",
                )?;
                let unchanged = !candidate
                    || warm_census.len() == census.len()
                        && warm_census
                            .iter()
                            .zip(&census)
                            .all(|(a, b)| normalized(a) == normalized(b));
                let census_bytes = |rows: &Vec<DispatchCensusRow>| {
                    rows.capacity() * size_of::<DispatchCensusRow>()
                        + rows
                            .iter()
                            .map(|r| {
                                r.kernel.capacity() + r.tag.as_ref().map_or(0, String::capacity)
                            })
                            .sum::<usize>()
                };
                emit(
                    out,
                    json!({"event":"warm_census","label":label,"timing_eligible":false,
                    "census":census_json(&census),"selector_dispatches":selectors.len(),"queries_per_dispatch":32,
                    "normalized_surrounding_dispatches_equal":unchanged,"normalization":"selector pipeline name and all diagnostic tags only",
                    "retained_census_heap_bytes":census_bytes(&warm_census)+census_bytes(&census),
                    "witness_reference_vectors_bytes":(selectors.capacity()+scores.capacity())*size_of::<&DispatchCensusRow>()}),
                );
                require(unchanged, "surrounding dispatch/order/geometry changed")?;
                if !candidate {
                    warm_census = census;
                } else {
                    warm_census = Vec::new();
                }
            }
            if label == "A1" {
                reference.clear();
            }
            let is_reference = label == "warm_A" || label == "A1";
            for step in 0..=if warm { 0 } else { 3 } {
                if step > 0 {
                    r.forward_token(tokens[STOP + step - 1])?;
                    emit(
                        out,
                        json!({"event":"continuation","label":label,"step":step,
                        "teacher_token":tokens[STOP+step-1],"position":r.next_position(),"timed":false}),
                    );
                }
                require(
                    r.next_position() == STOP + step,
                    "endpoint position mismatch",
                )?;
                let current = endpoint(&r, (step == 0).then_some(ROWS))?;
                emit_endpoint(out, &format!("{label}/step{step}"), &current);
                require(
                    !current.logits.is_empty() && current.logits.iter().all(|v| v.is_finite()),
                    "nonfinite endpoint",
                )?;
                if is_reference {
                    reference.push(current);
                } else {
                    check_endpoint(out, label, step, &reference[step], &current)?;
                }
            }
            if !warm {
                times.push((wall, gpu.unwrap()));
            }
        }
        let ratios = |v: Vec<f64>| {
            json!({"A1":v[0],"B1":v[1],"B2":v[2],"A2":v[3],
            "A_over_B_mean":(v[0]+v[3])/(v[1]+v[2]),"paired_A_over_B":[v[0]/v[1],v[3]/v[2]],
            "A2_over_A1":v[3]/v[0],"B2_over_B1":v[2]/v[1]})
        };
        emit(
            out,
            json!({"event":"abba","ordinary_call_wall_ms":ratios(times.iter().map(|t|t.0).collect()),
            "complete_gpu_ms":ratios(times.iter().map(|t|t.1).collect()),"decision":"unscored; no performance threshold"}),
        );
        Ok(())
    })
}

#[test]
#[ignore = "release, internal production lease; FLASH_PREFILL_MODEL and NEW FLASH_PREFILL_OUT"]
fn ordinary_selected_scan_ab() {
    run_packet(
        "flash.ordinary_selected_scan_ab.v1",
        include_bytes!("selected_scan_ab.rs"),
        json!({"prefix":START,"suffix":ROWS,"context":CONTEXT,"compressor_capacity":8192,
            "A":"production integer scan (override false)","B":"test-only parallel scan (override true)",
            "measurement":"ordinary continuation prefill wall and complete GPU sum; load, prefix, restore, warm census, readback/hash, JSON and three common teacher-forced steps excluded; NOT cold TTFT",
            "state":"same checkpoint restored before every arm, full persistent allocation hashes including inactive tails",
            "production_change":false}),
        run,
    );
}
