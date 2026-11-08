//! Completion-gated CED episodes for a cold serve request. Encoder calls yield
//! with provisional state; only the planned endpoint becomes snapshot/decode safe.
use super::*;

pub struct ColdPrefillSlice {
    pub encoder_rows: usize,
    pub complete: bool,
    pub logits: Vec<f32>,
}

pub(in super::super) struct Progress {
    plan: CedPlan,
    storage: gpu_step::ced::Storage,
    addresses: engram::AddressState,
    counters: CedCounters,
    next: usize,
}

impl Forward<'_> {
    /// Start before mutation, or decline for ordinary bounded Full execution.
    /// The caller must plan this endpoint over the whole cold request, not
    /// invoke CED after every ordinary encoder chunk. A long committed prefix
    /// is allowed only when all decoder reconstruction lies after that prefix.
    pub fn begin_cold_prefill_episode(
        &self,
        state: &mut State,
        tokens: usize,
    ) -> std::result::Result<bool, StateError> {
        op::require(
            state.is_ready() && state.cold_prefill.is_none(),
            "CED episode requires ready state",
        )?;
        op::require(
            Arc::ptr_eq(&state.identity, &self.identity),
            "state belongs to another model",
        )?;
        let start = state.position();
        let end = start
            .checked_add(tokens)
            .ok_or("CED episode endpoint overflow")?;
        let geometry =
            self.select_prefill(state, tokens, state.caller_cpu_reserve, state.prefill_rows)?;
        let plan = match CedPlan::from_prefix(
            &self.config,
            &self.schedule,
            &self.engram_hash,
            start,
            end,
            state.capacity(),
            geometry.selected_rows,
        ) {
            Ok(plan) => plan,
            Err(_) => return Ok(false),
        };
        let mut memory = estimate(self, state.capacity(), &plan)?;
        // Loaded quanta can create a genuine one-row final encoder chunk even
        // when the idle grid would not. Price both decoder command schedules.
        let mut scalar_plan = plan.clone();
        scalar_plan.encoder_chunks = Some(vec![end - 1..end]);
        let scalar_memory = estimate(self, state.capacity(), &scalar_plan)?;
        memory.incremental_metal_bytes = memory
            .incremental_metal_bytes
            .max(scalar_memory.incremental_metal_bytes);
        memory.incremental_cpu_bytes = memory
            .incremental_cpu_bytes
            .max(scalar_memory.incremental_cpu_bytes);
        // Progress and attention cursor retain the observed chunk descriptors.
        // Four times the worst count covers both Vec growth capacities.
        let chunk_cpu = (tokens.div_ceil(512) as u64)
            .checked_mul(4 * std::mem::size_of::<std::ops::Range<usize>>() as u64)
            .and_then(|n| n.checked_add(4096))
            .ok_or("CED chunk descriptor overflow")?;
        let decision = metal::evaluate_metal_memory_admission_with_cpu_bytes(
            memory.incremental_metal_bytes,
            memory
                .incremental_cpu_bytes
                .checked_add(chunk_cpu)
                .and_then(|n| n.checked_add(state.caller_cpu_reserve))
                .ok_or("CED caller reserve overflow")?,
            0,
            self.ctx.memory_signals(),
            true,
        );
        if let Some(reason) = decision.refusal() {
            if planning::pressure_reason(reason.reason()) {
                return Ok(false);
            }
            return Err(StateError::MemoryAdmissionDenied(reason));
        }
        let storage = gpu_step::ced::Storage::new(self, state.capacity(), &plan)?;
        let workspace = state.gpu_step.as_mut().ok_or("CED requires GPU state")?;
        workspace.start_ced_stream(&plan)?;
        state.prefill_rows = Some(geometry.selected_rows);
        state.phase = StatePhase::Provisional;
        state.cold_prefill = Some(Progress {
            plan,
            storage,
            addresses: state.addresses,
            counters: CedCounters::default(),
            next: start,
        });
        Ok(true)
    }

    /// One admitted encoder chunk, plus bounded decoder reconstruction at the
    /// final chunk. No command or CPU reader survives the yield. Remember each
    /// actual chunk so loaded 512-row quanta preserve Full's query-tile lineage.
    pub fn cold_prefill_episode_slice(
        &self,
        state: &mut State,
        tokens: &[u32],
        cancel: Option<&AtomicBool>,
        max_rows: usize,
    ) -> std::result::Result<ColdPrefillSlice, StateError> {
        let mut progress = state.cold_prefill.take().ok_or("no CED episode")?;
        // Any exit before committing or returning the provisional owner poisons
        // the state. Existing immutable snapshots are never touched.
        state.phase = StatePhase::Poisoned;
        op::require(
            Arc::ptr_eq(&state.identity, &self.identity),
            "state belongs to another model",
        )?;
        op::require(
            (512..=prefill::MAX_CHUNK_ROWS).contains(&max_rows),
            "CED scheduler quantum",
        )?;
        let plan = &mut progress.plan;
        op::require(
            tokens.len() == plan.prompt_rows - plan.start_position
                && state.position() == plan.start_position,
            "CED episode input extent changed",
        )?;
        op::require(
            tokens.iter().all(|&t| (t as usize) < self.token_map.len()),
            "token outside vocabulary",
        )?;
        cancelled(cancel)?;
        let start = progress.next;
        let end = (start + plan.full_chunk_rows.min(max_rows)).min(plan.prompt_rows);
        plan.encoder_chunks.as_mut().unwrap().push(start..end);
        let input = &tokens[start - plan.start_position..end - plan.start_position];
        let mut addresses = progress.addresses;
        let mut hashes = Vec::with_capacity(input.len());
        for &token in input {
            let (rows, next) = addresses
                .step_with_config(
                    engram::Token::Compressed(self.token_map[token as usize]),
                    &self.engram_hash,
                )
                .map_err(|e| format!("{e:?}"))?;
            hashes.push(rows);
            addresses = next;
        }
        let workspace = state.gpu_step.as_mut().ok_or("CED requires GPU state")?;
        workspace
            .attention_mut()
            .record_ced_stream_chunk(start..end)?;
        workspace.ced_stream_encoder(
            self,
            plan,
            start,
            input,
            &hashes,
            cancel,
            &mut progress.storage,
            &mut progress.counters,
        )?;
        progress.addresses = addresses;
        progress.next = end;
        cancelled(cancel)?;
        if end < plan.prompt_rows {
            state.phase = StatePhase::Provisional;
            state.cold_prefill = Some(progress);
            return Ok(ColdPrefillSlice {
                encoder_rows: input.len(),
                complete: false,
                logits: Vec::new(),
            });
        }
        let logits = workspace.finish_ced_stream(
            self,
            plan,
            cancel,
            Arithmetic::Production,
            &mut progress.storage,
            &mut progress.counters,
        )?;
        progress.counters.validate(plan)?;
        workspace.attention().validate_snapshot_position(end)?;
        cancelled(cancel)?;
        state.addresses = addresses;
        state.prefill_origin = PrefillExecution::ShrinkingCed;
        state.last_prefill = Some(PrefillGeometry {
            start: plan.start_position,
            tokens: tokens.len(),
            selected_rows: plan.full_chunk_rows,
        });
        state.phase = StatePhase::Ready;
        Ok(ColdPrefillSlice {
            encoder_rows: input.len(),
            complete: true,
            logits,
        })
    }
}
