use super::*;
use std::sync::Arc;

impl Settings {
    #[cfg(test)]
    pub(super) fn execute<S>(
        &self,
        request: &ServeRequest,
        prompt: &str,
        boundaries: Option<super::super::http::PromptBoundaries>,
        sink: &mut dyn GenerationSink,
        admit: impl FnMut(usize, usize, u64) -> std::result::Result<(), StateError>,
        new_state: impl FnOnce(usize) -> std::result::Result<S, ServeError>,
        reuse: &mut impl reuse::Reuse<S>,
        prefill: impl FnMut(
            &mut S,
            &[u32],
            &AtomicBool,
            bool,
        ) -> std::result::Result<Vec<f32>, ServeError>,
        step: impl FnMut(&mut S, u32, &AtomicBool) -> std::result::Result<Vec<f32>, String>,
        now: impl FnMut() -> Duration,
    ) -> std::result::Result<GenerationOutcome, BackendFailure> {
        let ceiling = super::super::output_memory::OutputByteCeiling::from_tokens_and_piece(
            self.limits.max_tokens(request),
            self.limits.max_piece_bytes,
        )
        .ok_or_else(|| {
            BackendFailure::Serve(ServeError::invalid_request(
                Some("max_output_tokens"),
                "V4.1 output byte budget overflow",
            ))
        })?;
        self.execute_with_ceiling(
            ceiling, request, prompt, boundaries, sink, admit, new_state, reuse, prefill, step, now,
        )
    }

    /// Drive one request to completion on the calling thread: the serial
    /// path every request takes unless the batch scheduler is enabled. The
    /// resumable phases below run in exactly this order.
    #[cfg(test)]
    pub(super) fn execute_with_ceiling<S>(
        &self,
        ceiling: super::super::output_memory::OutputByteCeiling,
        request: &ServeRequest,
        prompt: &str,
        boundaries: Option<super::super::http::PromptBoundaries>,
        sink: &mut dyn GenerationSink,
        admit: impl FnMut(usize, usize, u64) -> std::result::Result<(), StateError>,
        new_state: impl FnOnce(usize) -> std::result::Result<S, ServeError>,
        reuse: &mut impl reuse::Reuse<S>,
        prefill: impl FnMut(
            &mut S,
            &[u32],
            &AtomicBool,
            bool,
        ) -> std::result::Result<Vec<f32>, ServeError>,
        step: impl FnMut(&mut S, u32, &AtomicBool) -> std::result::Result<Vec<f32>, String>,
        now: impl FnMut() -> Duration,
    ) -> std::result::Result<GenerationOutcome, BackendFailure> {
        let mut engine = Closures {
            admit,
            new_state: Some(new_state),
            prefill,
            step,
            now,
        };
        self.execute_with_engine(
            ceiling,
            request,
            prompt,
            boundaries,
            sink,
            &mut engine,
            reuse,
        )
    }

    pub(super) fn execute_with_engine<S>(
        &self,
        ceiling: super::super::output_memory::OutputByteCeiling,
        request: &ServeRequest,
        prompt: &str,
        boundaries: Option<super::super::http::PromptBoundaries>,
        sink: &mut dyn GenerationSink,
        engine: &mut impl Engine<S>,
        reuse: &mut impl reuse::Reuse<S>,
    ) -> std::result::Result<GenerationOutcome, BackendFailure> {
        let mut run = self.begin(ceiling, request, prompt, boundaries, sink, 0, engine, reuse)?;
        while !run.prefilled() {
            self.prefill_slice(
                &mut run,
                crate::family_options::V41_PREFILL_ROWS,
                sink,
                engine,
                reuse,
            )?;
        }
        self.finish_prefill(&mut run, sink, engine, reuse)?;
        let flag = run.flag.clone();
        let cancel = cancel_of(&flag);
        let generation = {
            let Run {
                state,
                logits,
                consumed,
                sampler,
                reserve,
                ceiling,
                maximum,
                vocab,
                ..
            } = &mut run;
            let mut budgeted = BudgetedSink {
                sink,
                cancel,
                admit: || admit_output(reuse, *reserve, self.headroom),
                ceiling: *ceiling,
                emitted: 0,
            };
            let result = super::super::decode_loop::decode_serial(
                super::super::decode_loop::DecodeRequest {
                    family: "V4.1",
                    logits: logits.clone(),
                    max_tokens: *maximum,
                    stop_tokens: &[self.eos as i32],
                    vocab_size: *vocab as u32,
                },
                sampler,
                &self.tokenizer,
                &mut budgeted,
                |token| {
                    if cancelled(cancel) {
                        anyhow::bail!("V4.1 request cancelled");
                    }
                    let next = engine
                        .step(state, token, cancel)
                        .map_err(anyhow::Error::msg)?;
                    if next.len() != *vocab || !next.iter().all(|v| v.is_finite()) {
                        anyhow::bail!("V4.1 returned malformed logits");
                    }
                    consumed.push(token);
                    *logits = next.clone();
                    Ok(next)
                },
            );
            match result {
                Err(_) if cancelled(cancel) => return Err(BackendFailure::Aborted(interrupted())),
                result => result?,
            }
        };
        check(sink, cancel)?;
        Ok(self.finish(run, &generation, reuse, None))
    }

    /// Tokenize, plan checkpoint cuts, admit output, then take live/snapshot
    /// state or admit and allocate a cold one. `extra_reserve` adds other
    /// in-flight requests' caller reserves to this request's admissions
    /// (zero on the serial path).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn begin<S>(
        &self,
        ceiling: super::super::output_memory::OutputByteCeiling,
        request: &ServeRequest,
        prompt: &str,
        boundaries: Option<super::super::http::PromptBoundaries>,
        sink: &mut dyn GenerationSink,
        extra_reserve: u64,
        engine: &mut impl Engine<S>,
        reuse: &mut impl reuse::Reuse<S>,
    ) -> std::result::Result<Run<S>, BackendFailure> {
        let maximum = self.limits.max_tokens(request);
        // Check the output-only lower bound before tokenization or state allocation.
        super::super::decode_loop::required_forwards("V4.1", 1, maximum, self.context)?;
        let ids = super::super::decode_loop::encode_checked(
            &self.tokenizer,
            prompt,
            false,
            self.tokenizer.n_vocab(),
            "V4.1",
        )?;
        let mut checkpoint_cuts = Vec::new();
        let mut generation_cut = None;
        let mut shared_cut = None;
        if let Some(boundaries) = boundaries {
            for (name, offset) in [
                ("shared_prefix", boundaries.shared_prefix_end),
                ("generation_header", boundaries.generation_header_start),
            ] {
                let Some(offset) = offset else { continue };
                let prefix = prompt.get(..offset).and_then(|prefix| {
                    self.tokenizer
                        .encode(prefix, false)
                        .ok()
                        .map(|ids| ids.into_iter().map(|id| id as u32).collect::<Vec<_>>())
                });
                if let Some(prefix) = prefix.filter(|prefix| {
                    !prefix.is_empty() && prefix.len() < ids.len() && ids.starts_with(prefix)
                }) {
                    if name == "generation_header" {
                        generation_cut = Some(prefix.len());
                    } else {
                        shared_cut = Some(prefix.len());
                    }
                    checkpoint_cuts.push(prefix.len());
                } else {
                    tracing::warn!(
                        target: "qwen_diag",
                        "serve: deepseek_v41 {name} boundary at byte {offset} is not a token prefix of the prompt; not split there"
                    );
                }
            }
        }
        checkpoint_cuts.sort_unstable();
        checkpoint_cuts.dedup();
        // An aliased generation boundary still owns the shared-prefix cut.
        let fuse_header = self.fuse_header && generation_cut != shared_cut;
        let vocab = self.tokenizer.n_vocab() as usize;
        let required =
            super::super::decode_loop::required_forwards("V4.1", ids.len(), maximum, self.context)?;
        let peak = qwen_llm::tool_output_memory::checked_tool_block_peak_bytes(ceiling.get())
            .ok_or_else(|| super::super::output_memory::size_overflow("V4.1 output"))?;
        let reserve = u64::try_from(peak)
            .ok()
            .and_then(|n| caller_cpu_reserve(n, sink.transport_reserve_bytes(), vocab))
            .and_then(|n| n.checked_add(extra_reserve))
            .ok_or_else(|| super::super::output_memory::size_overflow("V4.1 transport"))?;
        let sampler = Sampler::new(sampling_with_defaults(request, self.sampling))
            .map_err(|e| ServeError::invalid_request(None, format!("V4.1 sampler: {e}")))?;
        let flag = sink.cancellation_flag();
        let cancel = cancel_of(&flag);
        check(sink, cancel)?;
        reuse.prepare(&ids);
        admit_output(reuse, reserve, self.headroom)?;
        let restore_start = engine.now();
        let restored = reuse.take(&ids, required, reserve)?;
        let restore_ms = (engine.now() - restore_start).as_secs_f64() * 1e3;
        let (state, logits, matched, reuse_source) = if let Some(restored) = restored {
            (
                restored.live.state,
                restored.live.logits,
                restored.live.tokens.len(),
                restored.source,
            )
        } else {
            super::super::memory_relief::retry(
                "deepseek_v41",
                engine.admit(required, ids.len(), reserve),
                |result| {
                    result
                        .as_ref()
                        .err()?
                        .memory_admission()
                        .map(|denied| (denied.reason(), u64::MAX))
                },
                |bytes| reuse.evict_for(bytes),
                || engine.admit(required, ids.len(), reserve),
            )
            .map_err(|error| match error.memory_admission() {
                Some(denied) => {
                    super::super::transport_memory::memory_refusal("V4.1 request", denied)
                }
                None => ServeError::server_error(format!("V4.1 request: {error}")),
            })?;
            let state = engine.new_state(required).map_err(|error| {
                if cancelled(cancel) {
                    BackendFailure::Aborted(io::Error::new(
                        io::ErrorKind::Interrupted,
                        error.message,
                    ))
                } else {
                    BackendFailure::Serve(error)
                }
            })?;
            (state, Vec::new(), 0, "none")
        };
        let capacity = reuse.capacity(&state, required);
        let prefill_start = engine.now();
        Ok(Run {
            position: matched,
            ids,
            checkpoint_cuts,
            generation_cut,
            fuse_header,
            header_fused: false,
            cold_ced: self.cold_ced && matched == 0,
            ced_ranges: Vec::new(),
            ced_target: None,
            ced_active: false,
            ced_episodes: 0,
            ced_yields: 0,
            conversation: super::super::checkpoint_ladder::conversation(request),
            ladder_attempted: false,
            ladder_capture_ms: 0.0,
            maximum,
            required,
            reserve,
            vocab,
            ceiling: ceiling.get(),
            sampler,
            flag,
            state,
            logits,
            matched,
            reuse_source,
            restore_ms,
            capacity,
            prefill_start,
            prefill_rows: 0,
            prefill_chunks: 0,
            prefill_calls: 0,
            prefill_ranges: Vec::new(),
            selected_rows: None,
            prefill_ms: 0.0,
            consumed: Vec::new(),
        })
    }

    /// One prefill call of at most `max_rows` prompt rows, never crossing a
    /// checkpoint cut. Only the call that reaches the prompt end computes logits.
    pub(super) fn prefill_slice<S>(
        &self,
        run: &mut Run<S>,
        max_rows: usize,
        sink: &mut dyn GenerationSink,
        engine: &mut impl Engine<S>,
        reuse: &mut impl reuse::Reuse<S>,
    ) -> std::result::Result<(), BackendFailure> {
        let flag = run.flag.clone();
        let cancel = cancel_of(&flag);
        let ids = &run.ids;
        let position = run.position;
        check(sink, cancel)?;
        admit_output(reuse, run.reserve, self.headroom)?;
        if run.ced_target.is_none() && run.cold_ced {
            // A request admitted under load keeps the existing small Full
            // path. Active CED episodes also accept loaded encoder quanta.
            if max_rows != crate::family_options::V41_PREFILL_ROWS {
                run.cold_ced = false;
            } else {
                let segment_end = run
                    .checkpoint_cuts
                    .iter()
                    .copied()
                    .find(|&cut| cut > position)
                    .unwrap_or(ids.len());
                run.ced_target = cold_episode_end(
                    position,
                    segment_end,
                    max_rows,
                    !run.ladder_attempted
                        && reuse.ladder_enabled()
                        && run.conversation.is_some()
                        && segment_end == run.generation_cut.unwrap_or(ids.len()),
                );
            }
        }
        let mut episode = false;
        let mut endpoint = run.next_endpoint(max_rows);
        if let Some(target) = run.ced_target {
            let begin = !run.ced_active;
            match engine
                .cold_prefill(
                    &mut run.state,
                    &ids[position..target],
                    cancel,
                    begin,
                    max_rows,
                )
                .map_err(|error| prefill_error(cancel, error))?
            {
                Some(progress) => {
                    run.ced_active = true;
                    run.ced_yields += 1;
                    let first = run.ced_ranges.last().map_or(position, |r| r[1]);
                    run.ced_ranges.push([first, first + progress.encoder_rows]);
                    check(sink, cancel)?;
                    if !progress.complete {
                        return Ok(());
                    }
                    run.ced_active = false;
                    run.ced_target = None;
                    run.ced_episodes += 1;
                    endpoint = target;
                    run.logits = progress.logits;
                    episode = true;
                }
                None => {
                    run.ced_target = None;
                    run.cold_ced = false;
                }
            }
        }
        let fused_header = run
            .generation_cut
            .is_some_and(|header| position < header && header < endpoint);
        let with_logits = endpoint == ids.len();
        if !episode {
            run.logits = engine
                .prefill(
                    &mut run.state,
                    &ids[position..endpoint],
                    cancel,
                    with_logits,
                )
                .map_err(|error| prefill_error(cancel, error))?;
        }
        check(sink, cancel)?;
        if with_logits
            && (run.logits.len() != run.vocab || !run.logits.iter().all(|v| v.is_finite()))
        {
            return Err(ServeError::server_error("V4.1 returned malformed logits").into());
        }
        let report = reuse.prefill_report(&run.state, position, endpoint - position);
        if report.start != position || report.tokens != endpoint - position {
            return Err(ServeError::server_error("V4.1 prefill report extent mismatch").into());
        }
        run.selected_rows = Some(report.selected_rows);
        run.prefill_calls += 1;
        if episode {
            run.prefill_chunks += run.ced_ranges.len();
            run.prefill_rows = run.prefill_rows.max(
                run.ced_ranges
                    .iter()
                    .map(|r| r[1] - r[0])
                    .max()
                    .unwrap_or(0),
            );
            run.prefill_ranges.append(&mut run.ced_ranges);
        } else {
            run.prefill_rows = run.prefill_rows.max(report.max_chunk_rows());
            run.prefill_chunks += report.chunk_count();
            run.prefill_ranges
                .extend(report.chunk_ranges().map(|range| [range.start, range.end]));
        }
        run.position = endpoint;
        run.header_fused |= fused_header;
        let segment_end = run
            .checkpoint_cuts
            .iter()
            .copied()
            .find(|&cut| cut > position)
            .unwrap_or(ids.len());
        if !run.ladder_attempted
            && segment_end == run.generation_cut.unwrap_or(ids.len())
            && reuse.ladder_enabled()
            && (ladder_candidate(position, endpoint, segment_end, max_rows)
                || (episode
                    && endpoint < segment_end
                    && segment_end - endpoint <= crate::family_options::V41_PREFILL_ROWS))
            && let Some(conversation) = run.conversation
        {
            // Full uses an actual call boundary; CED commits the endpoint
            // planned for this ladder before any encoder work. A loaded final
            // quantum must not discard that planned Ready checkpoint.
            run.ladder_attempted = true;
            let start = engine.now();
            reuse.ladder_checkpoint(conversation, &run.state, &ids[..endpoint], run.reserve);
            run.ladder_capture_ms += (engine.now() - start).as_secs_f64() * 1e3;
        }
        if run.checkpoint_cuts.contains(&endpoint) || fused_header {
            let logits = if fused_header { &run.logits[..] } else { &[] };
            reuse.checkpoint(&run.state, &ids[..endpoint], logits, run.reserve);
        }
        Ok(())
    }

    /// Validate the prompt-end logits and publish the uncut prompt checkpoint.
    pub(super) fn finish_prefill<S>(
        &self,
        run: &mut Run<S>,
        sink: &mut dyn GenerationSink,
        engine: &mut impl Engine<S>,
        reuse: &mut impl reuse::Reuse<S>,
    ) -> std::result::Result<(), BackendFailure> {
        check(sink, cancel_of(&run.flag))?;
        run.prefill_ms =
            (engine.now() - run.prefill_start).as_secs_f64() * 1e3 - run.ladder_capture_ms;
        if run.logits.len() != run.vocab || !run.logits.iter().all(|v| v.is_finite()) {
            return Err(ServeError::server_error("V4.1 returned malformed logits").into());
        }
        if run.checkpoint_cuts.is_empty() {
            reuse.checkpoint(&run.state, &run.ids, &run.logits, run.reserve);
        }
        run.consumed = run.ids.clone();
        Ok(())
    }

    /// Phases, shared stats and the live commit after a checked successful
    /// generation. `batch` adds the scheduler's per-request fields.
    pub(super) fn finish<S>(
        &self,
        run: Run<S>,
        generation: &crate::GenerationResult,
        reuse: &mut impl reuse::Reuse<S>,
        batch: Option<serde_json::Value>,
    ) -> GenerationOutcome {
        let Run {
            ids,
            matched,
            reuse_source,
            restore_ms,
            prefill_ms,
            ladder_capture_ms,
            conversation,
            generation_cut,
            checkpoint_cuts,
            header_fused,
            ced_episodes,
            ced_yields,
            prefill_rows,
            prefill_chunks,
            prefill_calls,
            prefill_ranges,
            selected_rows,
            required,
            capacity,
            state,
            consumed,
            logits,
            ..
        } = run;
        let retention_start = Instant::now();
        // Folding captures the completed prompt, never a state labelled as its
        // pre-header prefix. Exact prompt hits restore that endpoint and logits
        // without a new prefill, so they must retain the same key as well.
        let tip = if header_fused || matched == ids.len() {
            Some(ids.len())
        } else {
            generation_cut.or_else(|| checkpoint_cuts.is_empty().then_some(ids.len()))
        };
        if let Some(tip) = tip {
            reuse.retain_tip(conversation, &ids[..tip]);
        }
        let retention_ms = retention_start.elapsed().as_secs_f64() * 1e3;
        let transitions = generation.transitions;
        let transition_ms = generation.transition_ms;
        let prefill_mode = if prefill_chunks == 0 {
            "none"
        } else if ced_episodes > 0 {
            "shrinking_ced+full"
        } else {
            crate::family_options::V41_PREFILL_MODE
        };
        tracing::info!(target: "qwen_diag",
            "serve phases: family=deepseek_v41 prefill_ms={prefill_ms:.3} reused_tokens={matched} reuse_source={reuse_source} restore_ms={restore_ms:.3} prefill_tokens={} prefill_mode={prefill_mode} prefill_rows={prefill_rows} prefill_chunks={prefill_chunks} prefill_calls={prefill_calls} prefill_selected_rows={selected_rows:?} prefill_rows_max={} decode_ms={:.3} required_forwards={required} capacity={} transitions={transitions} transition_ms={transition_ms:.3} ced_episodes={ced_episodes} ced_yields={ced_yields} ladder_capture_ms={ladder_capture_ms:.3} retention_ms={retention_ms:.3}",
            ids.len() - matched, crate::family_options::V41_PREFILL_ROWS, generation.wall_ms, capacity);
        let mut phases = serde_json::json!({
            "family":"deepseek_v41", "prefill_ms":prefill_ms, "ladder_capture_ms":ladder_capture_ms, "retention_ms":retention_ms, "prefill_tokens":ids.len() - matched,
            "prefill_mode":prefill_mode, "prefill_rows":prefill_rows, "prefill_chunks":prefill_chunks,
            "prefill_rows_max":crate::family_options::V41_PREFILL_ROWS,
            "prefill_selected_rows":selected_rows, "prefill_calls":prefill_calls,
            "prefill_chunk_ranges":prefill_ranges,
            "prefill_header_fused":header_fused,
            "decode_ms":generation.wall_ms, "generated_tokens":generation.tokens.len(),
            "transitions":transitions, "transition_ms":transition_ms,
            "required_forwards":required, "capacity":capacity, "reused_tokens":matched, "reuse_source":reuse_source, "restore_ms":restore_ms, "ced":ced_episodes > 0, "ced_episodes":ced_episodes, "ced_yields":ced_yields,
            "cache_policy":"exact_prefix_ram",
            "reuse_selection_policy":"compatible_live_then_longest_snapshot_then_cold",
        });
        if let Some(batch) = batch {
            tracing::info!(target: "qwen_diag", "serve batch: family=deepseek_v41 {batch}");
            phases["batch"] = batch;
        }
        let mut outcome =
            super::super::outcome::finish_generation(ids.len(), generation, matched, restore_ms);
        outcome.stats.as_mut().expect("shared stats").phases = Some(phases);
        reuse.commit(reuse::Live {
            state,
            tokens: consumed,
            logits,
        });
        outcome
    }
}

/// The model operations one request needs. The serial path adapts its
/// closures; the batch scheduler implements it over the resident forward.
pub(super) trait Engine<S> {
    /// The engine's typed admission error, as the serial closures return it.
    #[allow(clippy::result_large_err)]
    fn admit(
        &mut self,
        positions: usize,
        prefill_tokens: usize,
        reserve: u64,
    ) -> std::result::Result<(), StateError>;
    fn new_state(&mut self, required: usize) -> std::result::Result<S, ServeError>;
    fn prefill(
        &mut self,
        state: &mut S,
        tokens: &[u32],
        cancel: &AtomicBool,
        with_logits: bool,
    ) -> std::result::Result<Vec<f32>, ServeError>;
    fn cold_prefill(
        &mut self,
        _: &mut S,
        _: &[u32],
        _: &AtomicBool,
        _: bool,
        _: usize,
    ) -> std::result::Result<Option<qwen_llm::deepseek_v41::forward::ColdPrefillSlice>, ServeError>
    {
        Ok(None)
    }
    fn step(
        &mut self,
        state: &mut S,
        token: u32,
        cancel: &AtomicBool,
    ) -> std::result::Result<Vec<f32>, String>;
    fn now(&mut self) -> Duration;
}

#[cfg(test)]
struct Closures<A, N, P, T, C> {
    admit: A,
    new_state: Option<N>,
    prefill: P,
    step: T,
    now: C,
}

#[cfg(test)]
impl<S, A, N, P, T, C> Engine<S> for Closures<A, N, P, T, C>
where
    A: FnMut(usize, usize, u64) -> std::result::Result<(), StateError>,
    N: FnOnce(usize) -> std::result::Result<S, ServeError>,
    P: FnMut(&mut S, &[u32], &AtomicBool, bool) -> std::result::Result<Vec<f32>, ServeError>,
    T: FnMut(&mut S, u32, &AtomicBool) -> std::result::Result<Vec<f32>, String>,
    C: FnMut() -> Duration,
{
    fn admit(
        &mut self,
        positions: usize,
        prefill_tokens: usize,
        reserve: u64,
    ) -> std::result::Result<(), StateError> {
        (self.admit)(positions, prefill_tokens, reserve)
    }
    fn new_state(&mut self, required: usize) -> std::result::Result<S, ServeError> {
        (self
            .new_state
            .take()
            .expect("one cold allocation per request"))(required)
    }
    fn prefill(
        &mut self,
        state: &mut S,
        tokens: &[u32],
        cancel: &AtomicBool,
        with_logits: bool,
    ) -> std::result::Result<Vec<f32>, ServeError> {
        (self.prefill)(state, tokens, cancel, with_logits)
    }
    fn step(
        &mut self,
        state: &mut S,
        token: u32,
        cancel: &AtomicBool,
    ) -> std::result::Result<Vec<f32>, String> {
        (self.step)(state, token, cancel)
    }
    fn now(&mut self) -> Duration {
        (self.now)()
    }
}

/// One request between its phases: everything `execute_with_ceiling` keeps
/// in locals, so the batch scheduler can resume it between other requests.
pub(super) struct Run<S> {
    pub(super) ids: Vec<u32>,
    checkpoint_cuts: Vec<usize>,
    generation_cut: Option<usize>,
    fuse_header: bool,
    header_fused: bool,
    cold_ced: bool,
    ced_ranges: Vec<[usize; 2]>,
    ced_target: Option<usize>,
    pub(super) ced_active: bool,
    ced_episodes: usize,
    ced_yields: usize,
    conversation: Option<super::super::checkpoint_ladder::Conversation>,
    ladder_attempted: bool,
    ladder_capture_ms: f64,
    pub(super) maximum: usize,
    required: usize,
    pub(super) reserve: u64,
    pub(super) vocab: usize,
    pub(super) ceiling: usize,
    pub(super) sampler: Sampler,
    pub(super) flag: Option<Arc<AtomicBool>>,
    pub(super) state: S,
    pub(super) logits: Vec<f32>,
    matched: usize,
    reuse_source: &'static str,
    restore_ms: f64,
    capacity: usize,
    prefill_start: Duration,
    pub(super) position: usize,
    prefill_rows: usize,
    prefill_chunks: usize,
    prefill_calls: usize,
    prefill_ranges: Vec<[usize; 2]>,
    selected_rows: Option<usize>,
    prefill_ms: f64,
    pub(super) consumed: Vec<u32>,
}

impl<S> Run<S> {
    /// Keep earlier cuts and loaded 512-row quanta. An idle owner's short
    /// append may include its terminal header in the first Full call.
    pub(super) fn next_endpoint(&self, max_rows: usize) -> usize {
        let cut = self
            .checkpoint_cuts
            .iter()
            .copied()
            .find(|&cut| cut > self.position)
            .unwrap_or(self.ids.len());
        let fold = self.fuse_header
            && self.matched > 0
            && self.position == self.matched
            && (512..=2048).contains(&(self.ids.len() - self.matched))
            && max_rows == crate::family_options::V41_PREFILL_ROWS
            && self.generation_cut == Some(cut)
            && (1..=8).contains(&(self.ids.len() - cut))
            && self.ids.len() - self.position <= max_rows;
        (self.position + max_rows)
            .min(if fold { self.ids.len() } else { cut })
            .min(self.ids.len())
    }
    pub(super) fn prefilled(&self) -> bool {
        self.position >= self.ids.len()
    }
    pub(super) fn cancel(&self) -> &AtomicBool {
        cancel_of(&self.flag)
    }
}

pub(super) fn cancel_of(flag: &Option<Arc<AtomicBool>>) -> &AtomicBool {
    flag.as_deref()
        .unwrap_or_else(|| crate::shutdown::cancellation_flag())
}

pub(super) fn check(
    sink: &mut dyn GenerationSink,
    cancel: &AtomicBool,
) -> std::result::Result<(), BackendFailure> {
    sink.tick().map_err(BackendFailure::Aborted)?;
    if cancel.load(Ordering::Acquire)
        || crate::shutdown::cancellation_flag().load(Ordering::Acquire)
    {
        return Err(BackendFailure::Aborted(io::Error::new(
            io::ErrorKind::Interrupted,
            "V4.1 request cancelled",
        )));
    }
    Ok(())
}

fn prefill_error(cancel: &AtomicBool, error: ServeError) -> BackendFailure {
    if cancelled(cancel) {
        BackendFailure::Aborted(io::Error::new(io::ErrorKind::Interrupted, error.message))
    } else {
        BackendFailure::Serve(error)
    }
}

pub(super) fn admit_output<S>(
    reuse: &mut impl reuse::Reuse<S>,
    reserve: u64,
    headroom: impl Fn() -> Option<HostMemoryHeadroom>,
) -> Result<(), ServeError> {
    let admit = || {
        super::super::transport_memory::admit_resident_transport_with_headroom(reserve, headroom())
    };
    super::super::memory_relief::retry(
        "deepseek_v41",
        admit(),
        |result| {
            result
                .as_ref()
                .err()
                .map(|denied| (denied.reason, u64::MAX))
        },
        |bytes| reuse.evict_for(bytes),
        admit,
    )
    .map_err(|denied| denied.error)
}

pub(super) fn cancelled(cancel: &AtomicBool) -> bool {
    cancel.load(Ordering::Acquire) || crate::shutdown::cancellation_flag().load(Ordering::Acquire)
}

pub(super) fn interrupted() -> io::Error {
    io::Error::new(io::ErrorKind::Interrupted, "V4.1 request cancelled")
}

pub(super) struct BudgetedSink<'a, A> {
    pub(super) sink: &'a mut dyn GenerationSink,
    pub(super) cancel: &'a AtomicBool,
    pub(super) admit: A,
    pub(super) ceiling: usize,
    pub(super) emitted: usize,
}

impl<A: FnMut() -> Result<(), ServeError>> GenerationSink for BudgetedSink<'_, A> {
    fn piece(&mut self, bytes: &[u8]) -> io::Result<()> {
        use super::super::output_memory;
        if cancelled(self.cancel) {
            return Err(interrupted());
        }
        if bytes.len() > self.ceiling.saturating_sub(self.emitted) {
            return Err(output_memory::refusal_error(
                output_memory::ceiling_exceeded("V4.1 decoded output", self.ceiling),
            ));
        }
        // This adapter retains no output buffer; emitted bytes are not credit
        // against resident ownership (the transport admits its own buffers).
        (self.admit)().map_err(output_memory::refusal_error)?;
        self.sink.piece(bytes)?;
        self.emitted += bytes.len();
        Ok(())
    }
    fn tick(&mut self) -> io::Result<()> {
        if cancelled(self.cancel) {
            return Err(interrupted());
        }
        self.sink.tick()
    }
}

/// One cold request plan: preserve mandatory cuts, and reconstruct at the
/// last existing full body boundary when the ladder will retain an interior.
/// Amortize the dependency halo over at least two idle encoder chunks.
fn cold_episode_end(start: usize, end: usize, rows: usize, ladder: bool) -> Option<usize> {
    let target = if ladder {
        start.checked_add((end.checked_sub(start + 1)? / rows) * rows)?
    } else {
        end
    };
    (target.checked_sub(start)? >= 2 * crate::family_options::V41_PREFILL_ROWS).then_some(target)
}

/// Last existing full call strictly before this segment's terminal boundary.
/// Using the actual call width also works when the scheduler interleaves 512
/// rows with another request's decode. No hypothetical grid authorizes capture.
fn ladder_candidate(start: usize, end: usize, segment_end: usize, rows: usize) -> bool {
    rows > 0
        && end.checked_sub(start) == Some(rows)
        && end < segment_end
        && segment_end - end <= rows
}

#[cfg(test)]
mod ladder_tests {
    use super::{cold_episode_end, ladder_candidate};
    #[test]
    fn cold_episodes_retain_real_ladder_boundaries_and_amortize_reconstruction() {
        assert_eq!(cold_episode_end(343, 64930, 4096, true), Some(61783));
        assert_eq!(cold_episode_end(343, 12631, 4096, true), Some(8535));
        assert_eq!(cold_episode_end(343, 16671, 4096, false), Some(16671));
        assert_eq!(cold_episode_end(0, 343, 4096, false), None);
        assert_eq!(cold_episode_end(61783, 64930, 4096, true), None);
        assert_eq!(cold_episode_end(0, 8192, 4096, false), Some(8192));
        assert_eq!(cold_episode_end(0, 8192, 4096, true), None);
    }

    #[test]
    fn last_committed_full_call_excludes_tip_and_small_appends() {
        assert!(ladder_candidate(100, 4196, 5000, 4096));
        assert!(!ladder_candidate(100, 4196, 9000, 4096));
        assert!(!ladder_candidate(100, 1100, 1102, 4096));
        assert!(!ladder_candidate(100, 4196, 4196, 4096));
        assert!(
            ladder_candidate(100, 4196, 8292, 4096),
            "aligned tip uses preceding full call"
        );
        assert!(
            ladder_candidate(66000, 66512, 67000, 512),
            "actual interleaved width"
        );
        assert!(!ladder_candidate(5, 4, 10, 512));
    }
}
