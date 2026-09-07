//! Single-turn Qwen request execution.

use super::*;

pub(crate) fn run_single_turn(
    model_path: &Path,
    gguf: GgufFile,
    args: &Args,
    staged_integrity: Option<StagedIntegrityMode>,
    drafter: Option<crate::drafter_policy::PreparedDrafter>,
) -> Result<()> {
    args.prefill_chunk.validate()?;
    ensure!(args.tokens > 0, "--tokens must be >= 1");
    let sampling = cli_sampling_config(args)?;
    validate_sampling_decode_policy(sampling, args.prompt_lookup)?;
    let durable_store = durable_checkpoint_store(args, staged_integrity)?;
    let durable_max_record_bytes = if durable_store.is_some() {
        durable_prefix_cache_max_entry_bytes(args)?
    } else {
        0
    };

    let mut timing_file = args
        .request_timings
        .as_ref()
        .map(|path| open_append_file(path, "request timings"))
        .transpose()?;
    let timing_enabled = timing_file.is_some();
    let arrival_ms = unix_epoch_ms()?;
    let load_t0 = Instant::now();
    let runtime = Runtime::metal().context("init Metal runtime")?;
    let loaded = runtime
        .load_open_model_for_disposable_single_turn_with_config(
            gguf,
            LoadedModelConfig {
                prefix_cache_max_bytes: prefix_cache_max_bytes(args)?,
                ..LoadedModelConfig::default()
            },
        )
        .with_context(|| format!("load model {}", model_path.display()))?;
    // v0.77 DFlash speculative decode. Policy and metadata binding were
    // settled pre-load by `drafter_policy`; only the GPU copy happens here.
    // The drafter mmap drops afterwards because `MetalDFlashHead` owns every
    // weight it needs (plus `config` and `target_layer_ids`).
    let dflash_head = match drafter.as_ref() {
        Some(prepared) => {
            let t0 = Instant::now();
            let head = prepared.load(loaded.context(), loaded.gguf())?;
            tracing::info!(
                target: "qwen_diag",
                drafter = %prepared.path().display(),
                block_size = head.config.block_size,
                dflash2 = head.config.selector_top_k > 0,
                load_ms = t0.elapsed().as_secs_f64() * 1e3,
                "dflash drafter loaded",
            );
            Some(head)
        }
        None => None,
    };
    drop(drafter);
    if args.sampling_attribution {
        let arch = loaded.arch();
        let lm_head = &loaded.metal_model().lm_head;
        ensure!(
            arch.kind == ArchKind::Moe
                && arch.n_layer == 40
                && arch.hidden_size == 2048
                && arch.vocab_size == 248_320
                && loaded.gguf().get_str("general.base_model.0.name") == Some("Qwen3.6 35B A3B")
                && loaded.gguf().get_u64("general.file_type") == Some(15)
                && lm_head.dtype == GgmlType::Q6_K
                && lm_head.shape.as_slice() == [2048, 248_320]
                && std::fs::metadata(model_path)
                    .is_ok_and(|metadata| metadata.len() == 22_134_528_992),
            "--sampling-attribution requires the frozen Qwen3.6 35B A3B profile"
        );
    }
    if args.sampled_structural {
        let vocab = usize::try_from(loaded.arch().vocab_size)
            .context("sampled structural vocabulary does not fit usize")?;
        ensure!(
            sampling.top_k < vocab,
            "--sampled-structural requires top-k smaller than vocabulary"
        );
        ensure!(
            loaded.gguf().get_str("general.base_model.0.name") == Some("Qwen3.6 35B A3B")
                && loaded.gguf().get_u64("general.file_type") == Some(15)
                && std::fs::metadata(model_path)
                    .is_ok_and(|metadata| metadata.len() == 22_134_528_992),
            "--sampled-structural requires the frozen Qwen3.6 35B A3B Q4_K_M profile"
        );
        loaded
            .forward()
            .ensure_sampled_structural_supported()
            .context("validate sampled structural decode organization")?;
    }
    let greedy_gpu_mode = configured_greedy_gpu_argmax_mode();
    let runtime_and_model_load_ms = load_t0.elapsed().as_secs_f64() * 1e3;
    if timing_enabled {
        loaded.context().set_pipeline_cache_metrics_enabled(true);
    }
    let process_model_ready_allocated =
        timing_enabled.then(|| loaded.context().current_allocated_size());
    let stdout_sink = if std::io::stdout().is_terminal() {
        "terminal"
    } else {
        "redirected"
    };
    let pair_id = if args.request_timing_warm_followup {
        Some(format!("{}-{}", std::process::id(), unix_epoch_ms_u64()?))
    } else {
        None
    };
    let sampling_clock_probe = args.sampling_attribution.then(measure_sampling_clock_probe);

    ensure!(
        loaded.prefix_cache_stats().entries == 0,
        "single-turn timing requires an empty prefix cache"
    );
    let first_pipeline_cache_start =
        timing_enabled.then(|| loaded.context().pipeline_cache_metrics());
    let first_request_t0 = Instant::now();
    let first_request_start_unix_ms = unix_epoch_ms_u64()?;
    let first_request_start_allocated =
        timing_enabled.then(|| loaded.context().current_allocated_size());
    let prompt_t0 = Instant::now();
    let (first_prompt, first_prompt_source, first_completed_checkpoint_eligible) =
        prompt_text(args)?;
    let first_prompt_acquisition_ms = prompt_t0.elapsed().as_secs_f64() * 1e3;
    if args.sampling_attribution {
        ensure!(
            first_prompt.len() == 1_891
                && format!("{:x}", Sha256::digest(first_prompt.as_bytes()))
                    == "e265de9742d1b22e566fc108ae26331ccf46166c6f071e73f211e0a1a7e8b474",
            "--sampling-attribution prompt byte identity changed"
        );
    }
    let tokenizer_t0 = Instant::now();
    let tokenizer = loaded.tokenizer().context("load tokenizer")?;
    let first_tokenizer_init_ms = tokenizer_t0.elapsed().as_secs_f64() * 1e3;
    let tokenization_t0 = Instant::now();
    let first_prompt_ids = tokenizer
        .encode(
            &first_prompt,
            prompt_add_special_tokens(args, first_prompt_source),
        )
        .context("tokenize prompt")?;
    let first_tokenization_ms = tokenization_t0.elapsed().as_secs_f64() * 1e3;
    if args.sampling_attribution {
        ensure!(
            first_prompt_ids.len() == 419
                && token_ids_sha256_i32le(&first_prompt_ids)
                    == "fb4bbb4dc66ca7d219099e2974e787ef976f80789cde3e48b8a905dceece1f9f",
            "--sampling-attribution prompt token identity changed"
        );
    }
    let first_prepared = PreparedRequest {
        request_start_unix_ms: first_request_start_unix_ms,
        request_t0: first_request_t0,
        request_start_allocated: first_request_start_allocated,
        pipeline_cache_start: first_pipeline_cache_start,
        prompt: first_prompt,
        prompt_source: first_prompt_source,
        completed_checkpoint_eligible: first_completed_checkpoint_eligible,
        prompt_ids: first_prompt_ids,
        prompt_acquisition_ms: first_prompt_acquisition_ms,
        tokenizer_init_ms: first_tokenizer_init_ms,
        tokenization_ms: first_tokenization_ms,
        tokenizer_reused: false,
    };
    let first = execute_single_turn_request(
        &loaded,
        &tokenizer,
        model_path,
        args,
        greedy_gpu_mode,
        runtime_and_model_load_ms,
        process_model_ready_allocated,
        pair_id.as_deref(),
        0,
        "first_post_model_load",
        first_prepared,
        stdout_sink,
        durable_store.as_ref(),
        durable_max_record_bytes,
        sampling_clock_probe.as_ref(),
        dflash_head.as_ref(),
    )?;
    let mut results = vec![first];

    if args.request_timing_warm_followup {
        ensure!(
            loaded.prefix_cache_stats().entries == 0,
            "warm follow-up requires an unused prefix cache"
        );
        let warm_pipeline_cache_start =
            timing_enabled.then(|| loaded.context().pipeline_cache_metrics());
        let warm_request_t0 = Instant::now();
        let warm_request_start_unix_ms = unix_epoch_ms_u64()?;
        let warm_request_start_allocated =
            timing_enabled.then(|| loaded.context().current_allocated_size());
        let prompt_t0 = Instant::now();
        let (warm_prompt, warm_prompt_source, warm_completed_checkpoint_eligible) =
            prompt_text(args)?;
        let warm_prompt_acquisition_ms = prompt_t0.elapsed().as_secs_f64() * 1e3;
        let tokenization_t0 = Instant::now();
        let warm_prompt_ids = tokenizer
            .encode(
                &warm_prompt,
                prompt_add_special_tokens(args, warm_prompt_source),
            )
            .context("retokenize warm follow-up prompt")?;
        let warm_tokenization_ms = tokenization_t0.elapsed().as_secs_f64() * 1e3;
        ensure!(
            warm_prompt_source == results[0].prompt_source
                && warm_prompt == results[0].prompt
                && warm_prompt_ids == results[0].prompt_ids
                && warm_completed_checkpoint_eligible == first_completed_checkpoint_eligible,
            "warm follow-up prompt bytes or token IDs differ from request 0"
        );
        let warm_prepared = PreparedRequest {
            request_start_unix_ms: warm_request_start_unix_ms,
            request_t0: warm_request_t0,
            request_start_allocated: warm_request_start_allocated,
            pipeline_cache_start: warm_pipeline_cache_start,
            prompt: warm_prompt,
            prompt_source: warm_prompt_source,
            completed_checkpoint_eligible: warm_completed_checkpoint_eligible,
            prompt_ids: warm_prompt_ids,
            prompt_acquisition_ms: warm_prompt_acquisition_ms,
            tokenizer_init_ms: 0.0,
            tokenization_ms: warm_tokenization_ms,
            tokenizer_reused: true,
        };
        let warm = execute_single_turn_request(
            &loaded,
            &tokenizer,
            model_path,
            args,
            greedy_gpu_mode,
            runtime_and_model_load_ms,
            process_model_ready_allocated,
            pair_id.as_deref(),
            1,
            "warm_followup",
            warm_prepared,
            stdout_sink,
            durable_store.as_ref(),
            durable_max_record_bytes,
            sampling_clock_probe.as_ref(),
            dflash_head.as_ref(),
        )?;
        let first_stop = results[0].row.as_ref().map(|row| row.stop_reason);
        let warm_stop = warm.row.as_ref().map(|row| row.stop_reason);
        ensure!(
            warm.generated == results[0].generated && warm_stop == first_stop,
            "warm follow-up generated tokens or stop reason differ from request 0"
        );
        results.push(warm);
        for result in &mut results {
            let row = result.row.as_mut().expect("paired timing row");
            row.pair_request_equal = Some(true);
            row.pair_generated_tokens_equal = Some(true);
        }
    }

    if let Some(file) = timing_file.as_mut() {
        let mut payload = Vec::new();
        for result in &results {
            serde_json::to_writer(&mut payload, result.row.as_ref().expect("timing row"))
                .context("serialize request timings")?;
            payload.push(b'\n');
        }
        file.write_all(&payload).context("write request timings")?;
        file.flush().context("flush request timings")?;
    }

    let cache_stats = loaded.prefix_cache_stats();
    for (index, result) in results.iter().enumerate() {
        let load_ms = if index == 0 {
            runtime_and_model_load_ms + result.tokenizer_init_ms
        } else {
            0.0
        };
        let stats_prefix = if args.request_timing_warm_followup {
            format!("stats[{index}]")
        } else {
            "stats".to_owned()
        };
        // Single-prompt stats line consumed by v0622/23/38/39/40. The
        // `qwen_diag` target keeps the bare `stats: prompt_tokens=…`
        // format that those scripts anchor `re.fullmatch` against.
        tracing::info!(
            target: "qwen_diag",
            concat!(
                "{}: prompt_tokens={} generated_tokens={} transitions={} stop_reason={} ",
                "load_ms={:.1} prefill_ms={:.1} ttft_ms={:.1} ",
                "decode_tps={:.2} transition_tps={:.2} cache_entries={} ",
                "cache_mib={:.1}/{:.1}",
            ),
            stats_prefix,
            result.prompt_ids.len(),
            result.generated.len(),
            result.transitions,
            result.stop_reason.as_str(),
            load_ms,
            result.prefill_ms,
            result.ttft_ms,
            result.decode_tps,
            result.transition_tps,
            cache_stats.entries,
            cache_stats.indexed_bytes as f64 / 1024.0 / 1024.0,
            cache_stats.max_indexed_bytes as f64 / 1024.0 / 1024.0,
        );
    }

    if let Some(path) = args.trace_request.as_ref() {
        append_request_trace(
            path,
            arrival_ms,
            results[0].prompt_ids.len(),
            results[0].generated.len(),
        )?;
    }
    if let Some(path) = args.request_stats_jsonl.as_ref() {
        // Legacy flat --messages renders the unpinned generic contract; `run`
        // carries the pinned protocol label on its prepared prompt.
        let template = args
            .prepared_prompt
            .as_ref()
            .and_then(|prompt| prompt.template)
            .or(Some("generic"));
        for (index, result) in results.iter().enumerate() {
            let load_ms = if index == 0 {
                runtime_and_model_load_ms + result.tokenizer_init_ms
            } else {
                0.0
            };
            let prefill_tps = if result.prefill_ms > 0.0 {
                result.prompt_ids.len() as f64 / (result.prefill_ms / 1e3)
            } else {
                0.0
            };
            let measured = RequestStatsMeasured {
                input_tokens: result.prompt_ids.len() as u64,
                output_tokens: result.generated.len() as u64,
                transitions: result.transitions as u64,
                stop_reason: result.stop_reason,
                tokenizer_ms: result.tokenization_ms,
                load_ms,
                prefill_ms: result.prefill_ms,
                prefill_tps,
                decode_ms: result.decode_ms,
                decode_tps: result.decode_tps,
                transition_tps: result.transition_tps,
                total_ms: result.total_ms,
                output_fingerprint: GeneratedTokenSha256Digest::of(&result.generated),
            };
            append_single_turn_stats_record(
                path,
                u32::try_from(index).context("request index exceeds u32")?,
                "qwen",
                request_stats_input(result.prompt_source, template),
                &measured,
                None,
            )?;
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_single_turn_request(
    loaded: &LoadedModel,
    tokenizer: &Tokenizer,
    model_path: &Path,
    args: &Args,
    greedy_gpu_mode: GreedyGpuArgmaxMode,
    runtime_and_model_load_ms: f64,
    process_model_ready_allocated: Option<u64>,
    pair_id: Option<&str>,
    request_index: usize,
    request_epoch: &'static str,
    prepared: PreparedRequest,
    stdout_sink: &'static str,
    durable_store: Option<&DurableCheckpointStore>,
    durable_max_record_bytes: u64,
    sampling_clock_probe: Option<&SamplingClockProbe>,
    dflash_head: Option<&MetalDFlashHead>,
) -> Result<SingleTurnResult> {
    let PreparedRequest {
        request_start_unix_ms,
        request_t0,
        request_start_allocated,
        pipeline_cache_start,
        prompt,
        prompt_source,
        completed_checkpoint_eligible,
        prompt_ids,
        prompt_acquisition_ms,
        tokenizer_init_ms,
        tokenization_ms,
        tokenizer_reused,
    } = prepared;
    let timing_enabled = process_model_ready_allocated.is_some();
    let validation_t0 = Instant::now();
    if prompt_ids.is_empty() {
        bail!("prompt tokenized to zero tokens");
    }
    let min_capacity = prompt_ids
        .len()
        .checked_add(args.tokens)
        .and_then(|v| v.checked_add(16))
        .context("sequence capacity overflow")?;
    let capacity = args.max_context_tokens.unwrap_or(min_capacity);
    ensure!(
        capacity >= prompt_ids.len() + args.tokens,
        "max context {} is smaller than prompt {} + generation {}",
        capacity,
        prompt_ids.len(),
        args.tokens
    );
    let capacity_validation_ms = validation_t0.elapsed().as_secs_f64() * 1e3;
    let allocated = allocate_prefill_request_state(
        loaded,
        args.prefill_chunk,
        prompt_ids.len(),
        capacity,
        durable_store.is_none(),
    )?;
    let chunk = allocated.chunk;
    let prefill_chunk_decision = allocated.decision;
    let mut scratch = allocated.scratch;
    let mut sequence = allocated.sequence;
    let scratch_allocation_ms = allocated.scratch_allocation_ms;
    let sequence_allocation_ms = allocated.sequence_allocation_ms;
    let after_scratch_allocated = timing_enabled.then_some(allocated.after_scratch_allocated);
    let after_sequence_allocated = timing_enabled.then_some(allocated.after_sequence_allocated);
    let forward = loaded.forward();
    if args.sampled_structural {
        forward
            .ensure_sampled_structural_session_supported(sequence.metal_session())
            .context("validate sampled structural session row before prefill")?;
    }

    let pipeline_cache_prefill_entry =
        timing_enabled.then(|| loaded.context().pipeline_cache_metrics());
    let durable_capture_policy = durable_store.map_or(DurableCapturePolicy::Disabled, |_| {
        selected_single_turn_durable_policy(args, prompt_ids.len(), completed_checkpoint_eligible)
    });
    let durable_prefix_len = durable_capture_policy.prompt_prefix_len();
    let mut durable_prepared: Option<PreparedCheckpoint> = None;
    let mut durable_capture_kind = None;
    let mut durable_capture_stop_reason = None;
    let mut durable_restore_ms = 0.0;
    let mut durable_capture_ms = 0.0;
    let mut prompt_logits = None;
    if let Some(store) = durable_store {
        let restore_t0 = Instant::now();
        let has_blobs = match store.has_managed_blobs() {
            Ok(has_blobs) => Some(has_blobs),
            Err(error) => {
                durable_restore_ms = restore_t0.elapsed().as_secs_f64() * 1e3;
                eprintln!(
                    "warning: durable prefix inventory failed after {:.1} ms; cold-prefilling: {error}",
                    durable_restore_ms,
                );
                None
            }
        };
        if has_blobs == Some(true) {
            let lookup_len = selected_single_turn_durable_lookup_len(args, prompt_ids.len());
            match loaded.restore_durable_prefix(
                store,
                &mut sequence,
                &prompt_ids[..lookup_len],
                durable_max_record_bytes,
            ) {
                Ok(attempt) => {
                    durable_restore_ms = restore_t0.elapsed().as_secs_f64() * 1e3;
                    eprintln!(
                        concat!(
                            "durable_prefix_cache: identity_cache={} hashed_bytes={} ",
                            "checkpoint_hit={} matched={} restored={} exact={} candidates={} ",
                            "corrupt_removed={} restore_total_ms={:.1}"
                        ),
                        identity_cache_outcome_label(attempt.compatibility.outcome),
                        attempt.compatibility.bytes_hashed,
                        attempt.hit.is_some(),
                        attempt.lookup.matched_prefix_len,
                        attempt.lookup.restored_prefix_len,
                        attempt.lookup.exact,
                        attempt.lookup.candidates_examined,
                        attempt.lookup.corrupt_entries_removed,
                        durable_restore_ms,
                    );
                    if let Some(hit) = attempt.hit {
                        prompt_logits = hit.exact_final_logits;
                    }
                }
                Err(RuntimeError::CheckpointStore(error)) => {
                    durable_restore_ms = restore_t0.elapsed().as_secs_f64() * 1e3;
                    eprintln!(
                        "warning: durable prefix lookup failed after {:.1} ms; cold-prefilling: {error}",
                        durable_restore_ms,
                    );
                }
                Err(error) => return Err(error.into()),
            }
        } else if has_blobs == Some(false) {
            durable_restore_ms = restore_t0.elapsed().as_secs_f64() * 1e3;
            eprintln!(
                "durable_prefix_cache: store_empty=true restore_total_ms={:.1}",
                durable_restore_ms,
            );
        }
    }

    let mut prefill_ms = 0.0;
    if let Some(prefix_len) = durable_prefix_len
        && prefix_len > sequence.position()
    {
        let position = sequence.position();
        let (logits, ms) = prefill_span(
            &forward,
            &mut sequence,
            &mut scratch,
            &prompt_ids[position..prefix_len],
            position,
        )?;
        prefill_ms += ms;
        let capture_t0 = Instant::now();
        let estimated =
            loaded.estimate_checkpoint_boundary_sizes(&sequence, prefix_len, false, true, 0)?;
        if estimated.record_bytes > durable_max_record_bytes {
            eprintln!(
                concat!(
                    "warning: durable prefix capture skipped: estimated_record_bytes={} ",
                    "estimated_snapshot_bytes={} max_entry_bytes={}"
                ),
                estimated.record_bytes, estimated.snapshot_bytes, durable_max_record_bytes,
            );
        } else {
            match loaded.prepare_checkpoint_boundary(
                &sequence,
                prompt_ids[..prefix_len].to_vec(),
                None,
                Some(logits.clone()),
                None,
                0,
            ) {
                Ok(prepared) => {
                    durable_prepared = Some(prepared);
                    durable_capture_kind = Some("prompt");
                }
                Err(RuntimeError::MetalModel(MfError::Snapshot(
                    SnapshotValidationError::AllocationFailed { .. },
                ))) => eprintln!(
                    "warning: durable prefix capture allocation failed; continuing without publication"
                ),
                Err(error) => return Err(error.into()),
            }
        }
        durable_capture_ms = capture_t0.elapsed().as_secs_f64() * 1e3;
        prompt_logits = Some(logits);
    }
    // v0.77: with a drafter loaded, the prompt prefill must also capture
    // the K target hidden layers the drafter conditions on. Windowed: an
    // all-SWA drafter observes only its SWA suffix, so only that window is
    // captured and seeded (F1 oracle, 2026-08-20); full-attention drafters
    // keep full-span capture via usize::MAX.
    let mut dflash_prefill_capture: Option<(MetalTensor, usize, usize, usize)> = None;
    if sequence.position() < prompt_ids.len() {
        let position = sequence.position();
        let (logits, ms) = match dflash_head {
            Some(head) => {
                let k_layers = head.target_layer_ids.len();
                let n_features = k_layers * loaded.arch().hidden_size as usize;
                let span = prompt_ids.len() - position;
                let window_limit = qwen_llm::metal_dflash::dflash_capture_window_limit(head);
                let (wstart_rel, window) =
                    qwen_llm::metal_dflash::dflash_capture_window_span(span, window_limit);
                let dst =
                    MetalTensor::zeros_f32(loaded.context(), vec![(window * n_features) as u64])
                        .context("allocate drafter prefill hidden capture")?;
                if wstart_rel > 0 {
                    let (_, plain_ms) = prefill_span(
                        &forward,
                        &mut sequence,
                        &mut scratch,
                        &prompt_ids[position..position + wstart_rel],
                        position,
                    )?;
                    prefill_ms += plain_ms;
                }
                let wstart_abs = position + wstart_rel;
                let out = prefill_span_with_capture(
                    &forward,
                    &mut sequence,
                    &mut scratch,
                    &prompt_ids[wstart_abs..],
                    wstart_abs,
                    &head.target_layer_ids,
                    &dst,
                )?;
                dflash_prefill_capture = Some((dst, wstart_abs, window, n_features));
                out
            }
            None => prefill_span(
                &forward,
                &mut sequence,
                &mut scratch,
                &prompt_ids[position..],
                position,
            )?,
        };
        prefill_ms += ms;
        prompt_logits = Some(logits);
    }
    let logits = prompt_logits.context("durable prefix restore did not produce prompt logits")?;
    let prefill_attention_query =
        (scratch.attn_matrix_tiled_layer_calls() > 0).then(|| PrefillAttentionQueryStats {
            outer_chunk_rows: chunk,
            query_rows: scratch.attn_matrix_query_rows(),
            tiled_layer_calls: scratch.attn_matrix_tiled_layer_calls(),
            query_tile_calls: scratch.attn_matrix_query_tile_calls(),
        });
    let prefill_scratch_overlay =
        scratch
            .prefill_scratch_overlay_stats()
            .map(|stats| PrefillScratchOverlayTimingStats {
                backing_bytes: stats.backing_bytes,
                attention_bytes: stats.attention_bytes,
                gdn_bytes: stats.gdn_bytes,
                saved_bytes: stats.saved_bytes,
            });
    let pipeline_cache_prefill_exit =
        timing_enabled.then(|| loaded.context().pipeline_cache_metrics());
    let after_prefill_allocated = timing_enabled.then(|| loaded.context().current_allocated_size());
    let mut scratch = Some(scratch);

    let stdout_handle = std::io::stdout();
    let mut stdout = stdout_handle.lock();
    let generation_start_ms = request_t0.elapsed().as_secs_f64() * 1e3;
    let mut first_delivery_ms = None;
    let mut first_callback_duration_ms = None;
    let mut first_delivery_allocated = None;
    let stop_tokens = loaded
        .gguf()
        .stop_token_ids()
        .context("load producer-declared stop tokens")?;
    let sampling_config = cli_sampling_config(args)?;
    let mut sampler = Sampler::new(sampling_config).context("initialize request sampler")?;
    let greedy_gpu_decision =
        resolve_greedy_gpu_decision(greedy_gpu_mode, sampling_config, args.prompt_lookup);
    let use_gpu_greedy = greedy_gpu_decision.enabled;
    #[allow(unused_assignments)]
    let mut dflash_stats: Option<DflashDecodeStats> = None;
    let (generation, prompt_lookup_stats, sampling_attribution, sampled_structural) =
        if let Some(head) = dflash_head {
            // v0.77 DFlash speculative decode. Seed the drafter's cross-context
            // with the captured prompt hiddens, then verify greedy or sampled
            // proposals against the packed target forward.
            // Windowed (2026-08-20): capacity covers the captured window plus
            // generated columns, not the whole prompt.
            let window_len = dflash_prefill_capture
                .as_ref()
                .map_or(0, |(_, _, window, _)| *window);
            let capacity = window_len + args.tokens + 16;
            let mut dsess = MetalDFlashSession::fresh(
                loaded.context(),
                head,
                loaded.arch().hidden_size as u64,
                loaded.arch().vocab_size as u64,
                capacity,
            )
            .context("allocate dflash drafter session")?;
            if let Some((dst, start_pos, window, n_features)) = dflash_prefill_capture.as_ref() {
                dsess
                    .append_target_ctx_columns_contiguous_now(
                        loaded.context(),
                        dst,
                        *start_pos as u32,
                        *window,
                        *n_features,
                    )
                    .context("seed drafter cross-context from prompt prefill")?;
            }
            let dflash_scratch =
                allocate_dflash_decode_scratch(loaded, head, sampling_config.temperature > 0.0)?;
            // Shadow-reference probe: a second sequence replays the committed
            // stream against the packed verifier; retain its prefill workspace.
            let shadow_probe = std::env::var_os("QWEN_DFLASH_SHADOW_PROBE").is_some();
            let mut shadow = if shadow_probe {
                let shadow_capacity = prompt_ids.len() + args.tokens + 16;
                let mut shadow_sequence = loaded
                    .create_sequence(SequenceConfig::new(shadow_capacity))
                    .context("allocate shadow probe sequence")?;
                crate::prefill_span(
                    &forward,
                    &mut shadow_sequence,
                    scratch.as_mut().expect("DFlash retains prefill scratch"),
                    &prompt_ids,
                    0,
                )
                .context("shadow probe prefill")?;
                Some(shadow_sequence)
            } else {
                None
            };
            let result = generate_dflash(
                loaded,
                &forward,
                head,
                dsess,
                dflash_scratch,
                sequence,
                logits,
                &mut sampler,
                args.tokens,
                &stop_tokens,
                None,
                None,
                shadow.as_mut(),
                |token| {
                    let callback_t0 = Instant::now();
                    write!(stdout, "{}", tokenizer.decode_piece(token))?;
                    stdout.flush().context("flush generated token")?;
                    if first_delivery_ms.is_none() {
                        first_delivery_ms = Some(request_t0.elapsed().as_secs_f64() * 1e3);
                        first_callback_duration_ms =
                            Some(callback_t0.elapsed().as_secs_f64() * 1e3);
                        first_delivery_allocated =
                            timing_enabled.then(|| loaded.context().current_allocated_size());
                    }
                    Ok(())
                },
            )?;
            sequence = result.sequence;
            let s = &result.stats;
            let steps = s.spec_steps.max(1) as f64;
            // Emitted tokens per verify step = 1 bonus + accepted drafts; the
            // economics of the whole mode reduce to this number vs the
            // ctx-keyed break-even.
            tracing::info!(
                target: "qwen_diag",
                concat!(
                    "dflash: off_ctx={} spec_steps={} off_steps={} accepted={}/{} ",
                    "mean_emitted={:.2} prefix_replay={}/{}/{}/{} alpha_backoff={} reason={} ",
                    "backoff_probes={} probe_ms={:.1} fallback={}/{:.1}ms ",
                    "draft_ms={:.1} draft_first_ms={:.1} verify_ms={:.1} ",
                    "read_ms={:.1} sample_ms={:.1} ",
                    "append_ms={:.1} restore_ms={:.1} serial_ms={:.1}",
                ),
                s.off_ctx,
                s.spec_steps,
                s.off_steps,
                s.accepted_drafts,
                s.drafts_scored,
                1.0 + s.accepted_drafts as f64 / steps,
                s.prefix_replay_steps,
                s.prefix_replay_accepted_drafts,
                s.prefix_replay_drafts_scored,
                s.prefix_replay_mismatches,
                s.alpha_backoff,
                s.backoff_reason.map_or("none", DflashBackoffReason::as_str),
                s.backoff_probe_steps,
                s.backoff_probe_ms,
                s.fallback_calls,
                s.fallback_ms,
                s.draft_ms / steps,
                s.draft_first_call_ms,
                s.verify_ms / steps,
                s.sampled_logits_read_ms / steps,
                s.sample_ms / steps,
                s.append_ms / steps,
                s.restore_ms / steps,
                s.serial_ms,
            );
            dflash_stats = Some(result.stats);
            (result.generation, None, None, None)
        } else if args.prompt_lookup {
            let result = generate_prompt_lookup(
                loaded,
                &forward,
                sequence,
                &prompt_ids,
                logits,
                args.tokens,
                &stop_tokens,
                |token| {
                    let callback_t0 = Instant::now();
                    write!(stdout, "{}", tokenizer.decode_piece(token))?;
                    stdout.flush().context("flush generated token")?;
                    if first_delivery_ms.is_none() {
                        first_delivery_ms = Some(request_t0.elapsed().as_secs_f64() * 1e3);
                        first_callback_duration_ms =
                            Some(callback_t0.elapsed().as_secs_f64() * 1e3);
                        first_delivery_allocated =
                            timing_enabled.then(|| loaded.context().current_allocated_size());
                    }
                    Ok(())
                },
            )?;
            sequence = result.sequence;
            (result.generation, Some(result.stats), None, None)
        } else {
            drop(scratch.take());
            let mut on_token = |token| {
                let callback_t0 = Instant::now();
                write!(stdout, "{}", tokenizer.decode_piece(token))?;
                stdout.flush().context("flush generated token")?;
                if first_delivery_ms.is_none() {
                    first_delivery_ms = Some(request_t0.elapsed().as_secs_f64() * 1e3);
                    first_callback_duration_ms = Some(callback_t0.elapsed().as_secs_f64() * 1e3);
                    first_delivery_allocated =
                        timing_enabled.then(|| loaded.context().current_allocated_size());
                }
                Ok(())
            };
            let (generation, sampling_attribution, sampled_structural) = if args
                .sampling_attribution
            {
                let (generation, sampler_attribution, transition_attribution) =
                    generate_serial_attributed(
                        logits,
                        args.tokens,
                        &stop_tokens,
                        &mut sampler,
                        &mut on_token,
                        |token| {
                            let position = sequence.position();
                            let next = forward
                                .single_token_sampled_attribution(
                                    token,
                                    u32::try_from(position).context("position does not fit u32")?,
                                    unsafe { sequence.metal_session_mut() },
                                )
                                .context("decode token with sampling attribution")?;
                            sequence.advance_by(1)?;
                            Ok(next)
                        },
                    )?;
                let clock_probe = sampling_clock_probe
                    .context("sampling attribution clock probe was not prepared")?
                    .clone();
                let attribution = finalize_sampling_attribution(
                    &prompt_ids,
                    clock_probe,
                    sampler_attribution,
                    transition_attribution,
                    generation.transition_ms,
                    generation.wall_ms,
                );
                (generation, Some(attribution), None)
            } else if args.sampled_structural {
                let (generation, telemetry) = generate_sampled_structural(
                    logits,
                    args.tokens,
                    &stop_tokens,
                    &mut sampler,
                    &mut on_token,
                    |token, trial, trial_telemetry| {
                        let position = sequence.position();
                        let (sampled, _profile, row) = forward
                            .single_token_sampled_structural(
                                token,
                                u32::try_from(position).context("position does not fit u32")?,
                                unsafe { sequence.metal_session_mut() },
                                trial,
                            )
                            .context("decode token with sampled structural path")?;
                        let state = match sampled {
                            Ok((sampled, evidence)) => {
                                ensure!(
                                    evidence.used_bounded_path,
                                    "sampled structural transition selection fell back"
                                );
                                trial_telemetry.record_transition(evidence, row)?;
                                SampledStructuralDecodeState::Selected(Ok(sampled))
                            }
                            Err(error) => SampledStructuralDecodeState::Selected(Err(error)),
                        };
                        sequence.advance_by(1)?;
                        Ok(state)
                    },
                )?;
                (generation, None, Some(telemetry))
            } else if use_gpu_greedy {
                (
                    generate_gpu_greedy(
                        logits,
                        args.tokens,
                        &stop_tokens,
                        &mut sampler,
                        &mut on_token,
                        |token| {
                            loaded
                                .decode_token_greedy(&mut sequence, token)
                                .context("decode token with GPU greedy selection")
                        },
                    )?,
                    None,
                    None,
                )
            } else {
                (
                    generate_serial(
                        logits,
                        args.tokens,
                        &stop_tokens,
                        &mut sampler,
                        &mut on_token,
                        |token| {
                            loaded
                                .decode_token(&mut sequence, token)
                                .context("decode token")
                        },
                    )?,
                    None,
                    None,
                )
            };
            (generation, None, sampling_attribution, sampled_structural)
        };
    let mut inference_complete_ms = request_t0.elapsed().as_secs_f64() * 1e3;
    let pipeline_cache_generation_exit =
        timing_enabled.then(|| loaded.context().pipeline_cache_metrics());
    let stop_reason = generation.stop_reason;
    let generated = generation.tokens;
    let completed_boundary = if durable_capture_policy == DurableCapturePolicy::AutomaticCompleted {
        Some(derive_completed_checkpoint_boundary(
            prompt_ids.len(),
            &generated,
            generation.transitions,
            sequence.position(),
        )?)
    } else {
        None
    };
    if !generated.is_empty() {
        writeln!(stdout)?;
        stdout.flush().context("flush final newline")?;
    }
    if first_delivery_ms.is_none() {
        let delivery_ms = request_t0.elapsed().as_secs_f64() * 1e3;
        first_delivery_ms = Some(delivery_ms);
        first_callback_duration_ms = Some(0.0);
        first_delivery_allocated =
            timing_enabled.then(|| loaded.context().current_allocated_size());
        inference_complete_ms = inference_complete_ms.max(delivery_ms);
    }
    let response_flushed_t0 = Instant::now();
    let total_request_ms = request_t0.elapsed().as_secs_f64() * 1e3;
    report_prefill_chunk_decision(prefill_chunk_decision.as_ref(), prompt_ids.len());
    let request_end_allocated = timing_enabled.then(|| loaded.context().current_allocated_size());
    drop(stdout);

    if let Some(boundary) = completed_boundary {
        let capture_t0 = Instant::now();
        let estimated = loaded.estimate_checkpoint_boundary_sizes(
            &sequence,
            boundary.consumed_prefix_len,
            true,
            false,
            0,
        )?;
        if estimated.record_bytes > durable_max_record_bytes {
            eprintln!(
                concat!(
                    "warning: completed durable checkpoint skipped: estimated_record_bytes={} ",
                    "estimated_snapshot_bytes={} max_entry_bytes={}"
                ),
                estimated.record_bytes, estimated.snapshot_bytes, durable_max_record_bytes,
            );
        } else {
            let consumed = boundary.consumed_tokens(&prompt_ids, &generated);
            match loaded.prepare_checkpoint_boundary(
                &sequence,
                consumed,
                Some(boundary.pending_token),
                None,
                None,
                0,
            ) {
                Ok(prepared) => {
                    durable_prepared = Some(prepared);
                    durable_capture_kind = Some("completed");
                    durable_capture_stop_reason = Some(stop_reason);
                }
                Err(RuntimeError::MetalModel(MfError::Snapshot(
                    SnapshotValidationError::AllocationFailed { .. },
                ))) => eprintln!(
                    "warning: completed durable checkpoint allocation failed; continuing without publication"
                ),
                Err(error) => return Err(error.into()),
            }
        }
        durable_capture_ms += capture_t0.elapsed().as_secs_f64() * 1e3;
    }

    let ttft_ms = first_delivery_ms.context("generation produced no first-token delivery")?;
    let first_token_ready_ms = generation_start_ms
        + generation
            .first_token_ready_ms
            .context("generation produced no first-token selection")?;
    let decode_tps = if generation.wall_ms > 0.0 {
        generated.len() as f64 / (generation.wall_ms / 1e3)
    } else {
        0.0
    };
    let transition_tps = if generation.transition_ms > 0.0 {
        generation.transitions as f64 / (generation.transition_ms / 1e3)
    } else {
        0.0
    };
    let model_identity = if timing_enabled {
        Some(loaded.snapshot_identity(&sequence)?)
    } else {
        None
    };
    let timing_values = timing_enabled.then(|| {
        (
            process_model_ready_allocated.expect("timing sample"),
            request_start_allocated.expect("timing sample"),
            after_scratch_allocated.expect("timing sample"),
            after_sequence_allocated.expect("timing sample"),
            after_prefill_allocated.expect("timing sample"),
            first_delivery_allocated.expect("timing sample"),
            request_end_allocated.expect("timing sample"),
        )
    });
    drop(sequence);
    drop(scratch);
    let after_state_drop_allocated =
        timing_enabled.then(|| loaded.context().current_allocated_size());
    if let (Some(store), Some(prepared)) = (durable_store, durable_prepared.as_ref()) {
        let publish_t0 = Instant::now();
        match loaded.publish_prepared_checkpoint(store, prepared, durable_max_record_bytes) {
            Ok(report) => {
                let publish_elapsed = publish_t0.elapsed();
                if store.staged_integrity_is_explicit() {
                    eprintln!(
                        concat!(
                            "durable_prefix_cache: publish={} capture={} matched_tokens={} ",
                            "restored_tokens={} pending={} stop_reason={} blob_bytes={} ",
                            "evicted={} staging_examined={} staging_removed={} ",
                            "staging_allocated_bytes_reclaimed={} ",
                            "staging_live={} staging_legacy={} staging_foreign={} ",
                            "staging_truncated={} identity={} staged_integrity={} ",
                            "staged_integrity_us={} capture_ms={:.1} publish_us={} ",
                            "post_response_us={}"
                        ),
                        publish_outcome_label(report.store.outcome),
                        durable_capture_kind.unwrap_or("unknown"),
                        prepared.matched_prefix_len(),
                        prepared.restored_prefix_len(),
                        prepared.has_pending_token(),
                        durable_capture_stop_reason.map_or("none", StopReason::as_str),
                        report.store.blob_bytes,
                        report.store.evicted_entries,
                        report.store.staging_entries_examined,
                        report.store.staging_entries_removed,
                        report.store.staging_allocated_bytes_reclaimed,
                        report.store.staging_live_entries,
                        report.store.staging_legacy_entries,
                        report.store.staging_foreign_entries,
                        report.store.staging_cleanup_truncated,
                        identity_cache_outcome_label(report.compatibility.outcome),
                        report.store.staged_integrity.mode.as_str(),
                        report.store.staged_integrity.elapsed.as_micros(),
                        durable_capture_ms,
                        publish_elapsed.as_micros(),
                        response_flushed_t0.elapsed().as_micros(),
                    );
                } else {
                    eprintln!(
                        concat!(
                            "durable_prefix_cache: publish={} capture={} matched_tokens={} ",
                            "restored_tokens={} pending={} stop_reason={} blob_bytes={} ",
                            "evicted={} staging_examined={} staging_removed={} ",
                            "staging_allocated_bytes_reclaimed={} ",
                            "staging_live={} staging_legacy={} staging_foreign={} ",
                            "staging_truncated={} identity={} capture_ms={:.1} publish_ms={:.1}"
                        ),
                        publish_outcome_label(report.store.outcome),
                        durable_capture_kind.unwrap_or("unknown"),
                        prepared.matched_prefix_len(),
                        prepared.restored_prefix_len(),
                        prepared.has_pending_token(),
                        durable_capture_stop_reason.map_or("none", StopReason::as_str),
                        report.store.blob_bytes,
                        report.store.evicted_entries,
                        report.store.staging_entries_examined,
                        report.store.staging_entries_removed,
                        report.store.staging_allocated_bytes_reclaimed,
                        report.store.staging_live_entries,
                        report.store.staging_legacy_entries,
                        report.store.staging_foreign_entries,
                        report.store.staging_cleanup_truncated,
                        identity_cache_outcome_label(report.compatibility.outcome),
                        durable_capture_ms,
                        publish_elapsed.as_secs_f64() * 1e3,
                    );
                }
            }
            Err(error) => {
                let publish_elapsed = publish_t0.elapsed();
                if store.staged_integrity_is_explicit() {
                    eprintln!(
                        concat!(
                            "durable_prefix_cache: publish=failed staged_integrity={} ",
                            "staged_integrity_us=none capture_ms={:.1} publish_us={} ",
                            "post_response_us={} error={}"
                        ),
                        store.staged_integrity_mode().as_str(),
                        durable_capture_ms,
                        publish_elapsed.as_micros(),
                        response_flushed_t0.elapsed().as_micros(),
                        error,
                    );
                } else {
                    eprintln!(
                        concat!(
                            "warning: durable prefix publication failed after response ",
                            "(restore_ms={:.1} capture_ms={:.1}): {}"
                        ),
                        durable_restore_ms, durable_capture_ms, error,
                    );
                }
            }
        }
    }

    let row = timing_values.map(|samples| {
        let model_identity = model_identity.expect("timing identity");
        let mut metal_allocated = metal_allocation_samples(
            samples.0,
            samples.1,
            samples.2,
            samples.3,
            samples.4,
            samples.5,
            samples.6,
            after_state_drop_allocated.expect("timing sample"),
        );
        if let Some(stats) = prompt_lookup_stats.as_ref() {
            metal_allocated.current_allocated_sampled_max_bytes = metal_allocated
                .current_allocated_sampled_max_bytes
                .max(stats.scratch_peak_allocated_bytes);
        }
        let _ = dflash_stats.as_ref();
        RequestTimingRow {
            schema_version: request_schema_version(
                args.prefill_chunk,
                args.prompt_lookup,
                prefill_attention_query.is_some(),
                prefill_scratch_overlay.is_some(),
                sampling_config.temperature > 0.0,
                args.sampling_attribution,
                args.sampled_structural,
            ),
            request_epoch,
            request_index,
            tokenizer_reused,
            pair_requested: pair_id.is_some(),
            pair_id: pair_id.map(str::to_owned),
            pair_request_equal: None,
            pair_generated_tokens_equal: None,
            prefix_cache_used: false,
            build_commit: env!("QWEN_BUILD_COMMIT"),
            build_dirty: env!("QWEN_BUILD_DIRTY"),
            build_source_state: env!("QWEN_BUILD_SOURCE_STATE"),
            model: model_path.display().to_string(),
            runtime_identity_kind: "metadata_compatibility_v1",
            runtime_model_id: format!("{:016x}", model_identity.model_id),
            runtime_tokenizer_id: format!("{:016x}", model_identity.tokenizer_id),
            greedy_gpu_selection_reason: greedy_gpu_decision.reason,
            request_start_unix_ms,
            runtime_and_model_load_ms,
            stdout_sink,
            ttft_endpoint: "stdout_flush_complete",
            prompt_source,
            prompt_bytes: prompt.len(),
            prompt_tokens: prompt_ids.len(),
            requested_tokens: args.tokens,
            generated_tokens: generated.len(),
            generated_token_sha256: generated_token_sha256(&generated),
            stop_reason,
            decode_policy: decode_policy_label(sampling_config, args.prompt_lookup, use_gpu_greedy),
            sampling: SamplingTelemetry::sampled(sampling_config, sampler.draws()),
            sampling_attribution,
            sampled_structural,
            terminal_token_target_transition_consumed: false,
            no_special_tokens: !prompt_add_special_tokens(args, prompt_source),
            prefill_chunk_requested: args.prefill_chunk,
            prefill_chunk_effective: chunk,
            prefill_chunk_decision,
            prefill_attention_query,
            prefill_scratch_overlay,
            max_context_tokens: capacity,
            prompt_acquisition_ms,
            tokenizer_init_ms,
            tokenization_ms,
            capacity_validation_ms,
            scratch_allocation_ms,
            sequence_allocation_ms,
            prefill_ms,
            first_token_selection_ms: generation.first_token_selection_ms,
            first_token_callback_duration_ms: first_callback_duration_ms
                .expect("first callback duration"),
            first_token_ready_ms,
            ttft_ms,
            generation_ms: generation.wall_ms,
            transition_count: generation.transitions,
            transition_ms: generation.transition_ms,
            transition_tps,
            inference_complete_ms,
            total_request_ms,
            pso_cache: pipeline_cache_phase_metrics(
                pipeline_cache_start.expect("timing PSO snapshot"),
                pipeline_cache_prefill_entry.expect("timing PSO snapshot"),
                pipeline_cache_prefill_exit.expect("timing PSO snapshot"),
                pipeline_cache_generation_exit.expect("timing PSO snapshot"),
            ),
            metal_allocated,
            prompt_lookup: prompt_lookup_stats,
        }
    });
    if let Some(row) = row.as_ref() {
        validate_request_timing_invariants(
            row.first_token_ready_ms,
            row.ttft_ms,
            row.inference_complete_ms,
            row.total_request_ms,
            row.generated_tokens,
            row.transition_count,
        )?;
        validate_sampling_attribution_row(row)?;
        validate_sampled_structural_row(row)?;
    }
    Ok(SingleTurnResult {
        row,
        prompt,
        prompt_source,
        prompt_ids,
        generated,
        transitions: generation.transitions,
        stop_reason,
        prefill_ms,
        ttft_ms,
        decode_tps,
        transition_tps,
        tokenizer_init_ms,
        tokenization_ms,
        decode_ms: generation.wall_ms,
        total_ms: total_request_ms,
    })
}
