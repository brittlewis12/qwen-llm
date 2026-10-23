use super::*;

impl Settings {
    #[cfg(test)]
    pub(super) fn execute<S>(
        &self,
        request: &ServeRequest,
        prompt: &str,
        boundaries: Option<super::super::http::PromptBoundaries>,
        sink: &mut dyn GenerationSink,
        admit: impl FnMut(usize, u64) -> std::result::Result<(), StateError>,
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

    pub(super) fn execute_with_ceiling<S>(
        &self,
        ceiling: super::super::output_memory::OutputByteCeiling,
        request: &ServeRequest,
        prompt: &str,
        boundaries: Option<super::super::http::PromptBoundaries>,
        sink: &mut dyn GenerationSink,
        mut admit: impl FnMut(usize, u64) -> std::result::Result<(), StateError>,
        new_state: impl FnOnce(usize) -> std::result::Result<S, ServeError>,
        reuse: &mut impl reuse::Reuse<S>,
        mut prefill: impl FnMut(
            &mut S,
            &[u32],
            &AtomicBool,
            bool,
        ) -> std::result::Result<Vec<f32>, ServeError>,
        mut step: impl FnMut(&mut S, u32, &AtomicBool) -> std::result::Result<Vec<f32>, String>,
        mut now: impl FnMut() -> Duration,
    ) -> std::result::Result<GenerationOutcome, BackendFailure> {
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
            .ok_or_else(|| super::super::output_memory::size_overflow("V4.1 transport"))?;
        let mut sampler = Sampler::new(sampling_with_defaults(request, self.sampling))
            .map_err(|e| ServeError::invalid_request(None, format!("V4.1 sampler: {e}")))?;
        let flag = sink.cancellation_flag();
        let cancel = flag
            .as_deref()
            .unwrap_or_else(|| crate::shutdown::cancellation_flag());
        let check = |sink: &mut dyn GenerationSink| -> std::result::Result<(), BackendFailure> {
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
        };
        let prefill_error = |error: ServeError| {
            if cancel.load(Ordering::Acquire)
                || crate::shutdown::cancellation_flag().load(Ordering::Acquire)
            {
                BackendFailure::Aborted(io::Error::new(io::ErrorKind::Interrupted, error.message))
            } else {
                BackendFailure::Serve(error)
            }
        };
        check(sink)?;
        admit_output(reuse, reserve, self.headroom)?;
        let restore_start = now();
        let restored = reuse.take(&ids, required)?;
        let restore_ms = (now() - restore_start).as_secs_f64() * 1e3;
        let (mut state, mut logits, matched, reuse_source) = if let Some(restored) = restored {
            (
                restored.live.state,
                restored.live.logits,
                restored.live.tokens.len(),
                restored.source,
            )
        } else {
            super::super::memory_relief::retry(
                "deepseek_v41",
                admit(required, reserve),
                |result| {
                    result
                        .as_ref()
                        .err()?
                        .memory_admission()
                        .map(|denied| (denied.reason(), u64::MAX))
                },
                |bytes| reuse.evict_for(bytes),
                || admit(required, reserve),
            )
            .map_err(|error| match error.memory_admission() {
                Some(denied) => {
                    super::super::transport_memory::memory_refusal("V4.1 request", denied)
                }
                None => ServeError::server_error(format!("V4.1 request: {error}")),
            })?;
            let state = new_state(required).map_err(|error| {
                if cancel.load(Ordering::Acquire)
                    || crate::shutdown::cancellation_flag().load(Ordering::Acquire)
                {
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
        let prefill_start = now();
        let mut position = matched;
        let mut prefill_rows = 0;
        let mut prefill_chunks = 0;
        while position < ids.len() {
            check(sink)?;
            admit_output(reuse, reserve, self.headroom)?;
            let endpoint = (position + crate::family_options::V41_PREFILL_ROWS)
                .min(
                    checkpoint_cuts
                        .iter()
                        .copied()
                        .find(|&cut| cut > position)
                        .unwrap_or(ids.len()),
                )
                .min(ids.len());
            let with_logits = endpoint == ids.len();
            logits = prefill(&mut state, &ids[position..endpoint], cancel, with_logits)
                .map_err(&prefill_error)?;
            check(sink)?;
            if with_logits && (logits.len() != vocab || !logits.iter().all(|v| v.is_finite())) {
                return Err(ServeError::server_error("V4.1 returned malformed logits").into());
            }
            prefill_rows = prefill_rows.max(endpoint - position);
            prefill_chunks += 1;
            position = endpoint;
            if checkpoint_cuts.contains(&position) {
                reuse.checkpoint(&state, &ids[..position], &[], reserve);
            }
        }
        check(sink)?;
        let prefill_ms = (now() - prefill_start).as_secs_f64() * 1e3;
        if logits.len() != vocab || !logits.iter().all(|v| v.is_finite()) {
            return Err(ServeError::server_error("V4.1 returned malformed logits").into());
        }
        if checkpoint_cuts.is_empty() {
            reuse.checkpoint(&state, &ids, &logits, reserve);
        }
        let mut consumed = ids.clone();
        let generation = {
            let mut budgeted = BudgetedSink {
                sink,
                cancel,
                admit: || admit_output(reuse, reserve, self.headroom),
                ceiling: ceiling.get(),
                emitted: 0,
            };
            let result = super::super::decode_loop::decode_serial(
                super::super::decode_loop::DecodeRequest {
                    family: "V4.1",
                    logits: logits.clone(),
                    max_tokens: maximum,
                    stop_tokens: &[self.eos as i32],
                    vocab_size: vocab as u32,
                },
                &mut sampler,
                &self.tokenizer,
                &mut budgeted,
                |token| {
                    if cancelled(cancel) {
                        anyhow::bail!("V4.1 request cancelled");
                    }
                    let next = step(&mut state, token, cancel).map_err(anyhow::Error::msg)?;
                    if next.len() != vocab || !next.iter().all(|v| v.is_finite()) {
                        anyhow::bail!("V4.1 returned malformed logits");
                    }
                    consumed.push(token);
                    logits = next.clone();
                    Ok(next)
                },
            );
            match result {
                Err(_) if cancelled(cancel) => return Err(BackendFailure::Aborted(interrupted())),
                result => result?,
            }
        };
        check(sink)?;
        let transitions = generation.transitions;
        let transition_ms = generation.transition_ms;
        let prefill_mode = if prefill_chunks == 0 {
            "none"
        } else {
            crate::family_options::V41_PREFILL_MODE
        };
        tracing::info!(target: "qwen_diag",
            "serve phases: family=deepseek_v41 prefill_ms={prefill_ms:.3} reused_tokens={matched} reuse_source={reuse_source} restore_ms={restore_ms:.3} prefill_tokens={} prefill_mode={prefill_mode} prefill_rows={prefill_rows} prefill_chunks={prefill_chunks} decode_ms={:.3} required_forwards={required} capacity={} transitions={transitions} transition_ms={transition_ms:.3} ced=false",
            ids.len() - matched, generation.wall_ms, capacity);
        let phases = serde_json::json!({
            "family":"deepseek_v41", "prefill_ms":prefill_ms, "prefill_tokens":ids.len() - matched,
            "prefill_mode":prefill_mode, "prefill_rows":prefill_rows, "prefill_chunks":prefill_chunks,
            "prefill_rows_max":crate::family_options::V41_PREFILL_ROWS,
            "decode_ms":generation.wall_ms, "generated_tokens":generation.tokens.len(),
            "transitions":transitions, "transition_ms":transition_ms,
            "required_forwards":required, "capacity":capacity, "reused_tokens":matched, "reuse_source":reuse_source, "restore_ms":restore_ms, "ced":false,
            "cache_policy":"exact_prefix_ram",
        });
        let mut outcome =
            super::super::outcome::finish_generation(ids.len(), &generation, matched, restore_ms);
        outcome.stats.as_mut().expect("shared stats").phases = Some(phases);
        reuse.commit(reuse::Live {
            state,
            tokens: consumed,
            logits,
        });
        Ok(outcome)
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

fn cancelled(cancel: &AtomicBool) -> bool {
    cancel.load(Ordering::Acquire) || crate::shutdown::cancellation_flag().load(Ordering::Acquire)
}

fn interrupted() -> io::Error {
    io::Error::new(io::ErrorKind::Interrupted, "V4.1 request cancelled")
}

struct BudgetedSink<'a, A> {
    sink: &'a mut dyn GenerationSink,
    cancel: &'a AtomicBool,
    admit: A,
    ceiling: usize,
    emitted: usize,
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
