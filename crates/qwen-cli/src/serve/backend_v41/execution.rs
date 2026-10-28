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
        let mut run = self.begin(
            ceiling,
            request,
            prompt,
            boundaries,
            sink,
            0,
            &mut engine,
            reuse,
        )?;
        while !run.prefilled() {
            self.prefill_slice(
                &mut run,
                crate::family_options::V41_PREFILL_ROWS,
                sink,
                &mut engine,
                reuse,
            )?;
        }
        self.finish_prefill(&mut run, sink, &mut engine, reuse)?;
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
        let endpoint = run.next_endpoint(max_rows);
        let with_logits = endpoint == ids.len();
        run.logits = engine
            .prefill(
                &mut run.state,
                &ids[position..endpoint],
                cancel,
                with_logits,
            )
            .map_err(|error| prefill_error(cancel, error))?;
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
        run.prefill_rows = run.prefill_rows.max(report.max_chunk_rows());
        run.prefill_chunks += report.chunk_count();
        run.prefill_calls += 1;
        run.prefill_ranges
            .extend(report.chunk_ranges().map(|range| [range.start, range.end]));
        run.position = endpoint;
        if run.checkpoint_cuts.contains(&endpoint) {
            reuse.checkpoint(&run.state, &ids[..endpoint], &[], run.reserve);
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
        run.prefill_ms = (engine.now() - run.prefill_start).as_secs_f64() * 1e3;
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
        let transitions = generation.transitions;
        let transition_ms = generation.transition_ms;
        let prefill_mode = if prefill_chunks == 0 {
            "none"
        } else {
            crate::family_options::V41_PREFILL_MODE
        };
        tracing::info!(target: "qwen_diag",
            "serve phases: family=deepseek_v41 prefill_ms={prefill_ms:.3} reused_tokens={matched} reuse_source={reuse_source} restore_ms={restore_ms:.3} prefill_tokens={} prefill_mode={prefill_mode} prefill_rows={prefill_rows} prefill_chunks={prefill_chunks} prefill_calls={prefill_calls} prefill_selected_rows={selected_rows:?} prefill_rows_max={} decode_ms={:.3} required_forwards={required} capacity={} transitions={transitions} transition_ms={transition_ms:.3} ced=false",
            ids.len() - matched, crate::family_options::V41_PREFILL_ROWS, generation.wall_ms, capacity);
        let mut phases = serde_json::json!({
            "family":"deepseek_v41", "prefill_ms":prefill_ms, "prefill_tokens":ids.len() - matched,
            "prefill_mode":prefill_mode, "prefill_rows":prefill_rows, "prefill_chunks":prefill_chunks,
            "prefill_rows_max":crate::family_options::V41_PREFILL_ROWS,
            "prefill_selected_rows":selected_rows, "prefill_calls":prefill_calls,
            "prefill_chunk_ranges":prefill_ranges,
            "decode_ms":generation.wall_ms, "generated_tokens":generation.tokens.len(),
            "transitions":transitions, "transition_ms":transition_ms,
            "required_forwards":required, "capacity":capacity, "reused_tokens":matched, "reuse_source":reuse_source, "restore_ms":restore_ms, "ced":false,
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
    fn step(
        &mut self,
        state: &mut S,
        token: u32,
        cancel: &AtomicBool,
    ) -> std::result::Result<Vec<f32>, String>;
    fn now(&mut self) -> Duration;
}

struct Closures<A, N, P, T, C> {
    admit: A,
    new_state: Option<N>,
    prefill: P,
    step: T,
    now: C,
}

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
    /// Where a prefill call of at most `max_rows` from here would end.
    pub(super) fn next_endpoint(&self, max_rows: usize) -> usize {
        (self.position + max_rows)
            .min(
                self.checkpoint_cuts
                    .iter()
                    .copied()
                    .find(|&cut| cut > self.position)
                    .unwrap_or(self.ids.len()),
            )
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
