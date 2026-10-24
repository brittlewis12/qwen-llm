use super::ced_plan::{CedPlan, CedRequest};
use super::*;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PrefillExecution {
    #[default]
    FullForward,
    ShrinkingCed,
}

impl PrefillExecution {
    pub fn name(self) -> &'static str {
        match self {
            Self::FullForward => "full_forward",
            Self::ShrinkingCed => "shrinking_ced",
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct CedLayerCounters {
    pub layer: usize,
    pub encoder_output_rows: usize,
    pub decoder_output_rows: usize,
    pub seed_rows: usize,
    pub publication_rows: usize,
    pub selection_producer_rows: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct CedCounters {
    pub encoder_output_rows: usize,
    pub publication_rows: usize,
    pub decoder_output_rows: usize,
    pub decoder_seed_rows: usize,
    pub encoded_commands: usize,
    pub completed_commands: usize,
    pub layers: Vec<CedLayerCounters>,
}

impl CedCounters {
    fn add(a: &mut usize, n: usize) -> Result<()> {
        *a = a.checked_add(n).ok_or("CED counter overflow")?;
        Ok(())
    }

    fn layer(&mut self, id: usize) -> &mut CedLayerCounters {
        if let Some(index) = self.layers.iter().position(|l| l.layer == id) {
            return &mut self.layers[index];
        }
        self.layers.push(CedLayerCounters {
            layer: id,
            ..CedLayerCounters::default()
        });
        self.layers.last_mut().unwrap()
    }

    pub(super) fn encoder(&mut self, layer: usize, rows: usize, selects: bool) -> Result<()> {
        Self::add(&mut self.encoder_output_rows, rows)?;
        let c = self.layer(layer);
        Self::add(&mut c.encoder_output_rows, rows)?;
        if selects {
            Self::add(&mut c.selection_producer_rows, rows)?;
        }
        Ok(())
    }

    pub(super) fn publication(&mut self, layer: usize, rows: usize) -> Result<()> {
        Self::add(&mut self.publication_rows, rows)?;
        Self::add(&mut self.layer(layer).publication_rows, rows)
    }

    pub(super) fn seed(&mut self, layer: usize, rows: usize) -> Result<()> {
        Self::add(&mut self.decoder_seed_rows, rows)?;
        Self::add(&mut self.layer(layer).seed_rows, rows)
    }

    pub(super) fn decoder(&mut self, layer: usize, rows: usize, selects: bool) -> Result<()> {
        Self::add(&mut self.decoder_output_rows, rows)?;
        let c = self.layer(layer);
        Self::add(&mut c.decoder_output_rows, rows)?;
        if selects {
            Self::add(&mut c.selection_producer_rows, rows)?;
        }
        Ok(())
    }

    pub(super) fn validate(&self, plan: &CedPlan) -> Result<()> {
        op::require(
            self.layers.len()
                == plan
                    .encoder_layers
                    .end
                    .checked_add(plan.decoder.len())
                    .ok_or("CED layer count overflow")?,
            "CED duplicate, missing or extraneous layer counters",
        )?;
        op::require(
            self.encoder_output_rows == plan.work.encoder_output_rows
                && self.publication_rows == plan.work.publication_rows
                && self.decoder_output_rows == plan.work.decoder_output_rows
                && self.decoder_seed_rows == plan.work.decoder_seed_rows
                && self.encoded_commands == self.completed_commands,
            "CED completed work does not match the dependency plan",
        )?;
        for layer in plan.encoder_layers.clone() {
            let count = self
                .layers
                .iter()
                .find(|count| count.layer == layer)
                .ok_or("missing CED encoder work counter")?;
            op::require(
                count.encoder_output_rows == plan.prompt_rows
                    && count.decoder_output_rows == 0
                    && count.seed_rows == 0
                    && count.publication_rows == 0,
                "CED encoder coverage mismatch",
            )?;
        }
        for layer in &plan.decoder {
            let count = self
                .layers
                .iter()
                .find(|count| count.layer == layer.layer)
                .ok_or("missing CED decoder work counter")?;
            op::require(
                count.decoder_output_rows == layer.output_rows.len()
                    && count.seed_rows == layer.seed_rows.len()
                    && count.encoder_output_rows == 0
                    && count.publication_rows
                        == if layer.layer == plan.publication_layer {
                            plan.prompt_rows
                        } else {
                            0
                        },
                "CED decoder coverage mismatch",
            )?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct CedMemoryEstimate {
    pub full_chunk_rows: usize,
    pub retained_residual_bytes: u64,
    pub retained_mix_bytes: u64,
    pub retained_selected_bytes: u64,
    pub retained_candidate_bytes: u64,
    pub priced_retained_bytes: u64,
    pub priced_chunk_bytes: u64,
    pub priced_half_chunk_bytes: u64,
    pub priced_publication_bytes: u64,
    pub priced_seed_bytes: u64,
    pub priced_batch_status_bytes: u64,
    pub batch_cpu_bytes: u64,
    pub priced_batch_diagnostic_bytes: u64,
    pub batch_encoding_reserve_bytes: u64,
    pub command_reserve_bytes: u64,
    pub incremental_metal_bytes: u64,
    pub incremental_cpu_bytes: u64,
    pub retained_control_cpu_bytes: u64,
    pub chunk_hash_cpu_bytes: u64,
    pub process_ru_maxrss_before_bytes: Option<u64>,
    pub process_ru_maxrss_after_bytes: Option<u64>,
    pub observation_method: &'static str,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct PrefillReport {
    pub execution: PrefillExecution,
    pub fallback: Option<String>,
    pub counters: Option<CedCounters>,
    pub memory: Option<CedMemoryEstimate>,
    /// Maximum of phase samples, not a continuous command/driver high-water.
    pub observed_incremental_metal_bytes: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Arithmetic {
    Production,
    #[cfg(test)]
    ScalarNative,
}

impl Arithmetic {
    pub(super) fn scalar(self) -> bool {
        match self {
            Self::Production => false,
            #[cfg(test)]
            Self::ScalarNative => true,
        }
    }
}

fn checked_add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b).ok_or_else(|| "CED memory overflow".into())
}

fn row_bytes(width: usize, rows: usize) -> Result<u64> {
    width
        .checked_mul(rows)
        .and_then(|n| n.checked_mul(4))
        .and_then(|n| u64::try_from(n).ok())
        .ok_or_else(|| "CED row storage overflow".into())
}

fn process_peak_rss_bytes() -> Option<u64> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } != 0 {
        return None;
    }
    let peak = u64::try_from(unsafe { usage.assume_init() }.ru_maxrss).ok()?;
    if cfg!(target_os = "macos") {
        Some(peak)
    } else {
        peak.checked_mul(1024)
    }
}

fn estimate(model: &Forward<'_>, capacity: usize, plan: &CedPlan) -> Result<CedMemoryEstimate> {
    let price = |bytes| -> Result<u64> {
        if bytes == 0 {
            return Ok(0);
        }
        model
            .ctx
            .price_shared_buffer_upper(bytes)
            .map(|p| p.priced_upper_bytes)
            .map_err(|e| e.to_string())
    };
    let mut estimate = estimate_from_parts(
        &model.config,
        model.layers.len(),
        &gpu_step::engram_projections(model),
        model.engram_hash.layers.len(),
        capacity,
        plan,
        metal::moe_grouped_tensor_selected(model.ctx),
        &price,
    )?;
    estimate.process_ru_maxrss_before_bytes = process_peak_rss_bytes();
    estimate.observation_method = "Metal allocation samples after allocation, encoding and completion; cumulative process ru_maxrss, not an isolated CED or continuous driver peak";
    Ok(estimate)
}

pub(super) fn estimate_from_parts(
    c: &Config,
    layers: usize,
    projections: &[(usize, usize)],
    hash_count: usize,
    capacity: usize,
    plan: &CedPlan,
    half_selected: bool,
    price: &impl Fn(u64) -> Result<u64>,
) -> Result<CedMemoryEstimate> {
    let rows = plan.prompt_rows.min(plan.full_chunk_rows);
    let mut retained = 0;
    for bytes in [
        plan.memory.boundary_residual_bytes,
        plan.memory.boundary_mix_bytes,
        plan.memory.selected_bytes,
        plan.memory.candidate_bytes,
    ] {
        retained = checked_add(retained, price(bytes)?)?;
    }
    let mut chunk = 0;
    for width in gpu_step::chunk_widths_from_geometry(c, layers, projections, capacity, rows)? {
        chunk = checked_add(chunk, price(row_bytes(width, rows)?)?)?;
    }
    let decoder_rows = plan
        .decoder
        .first()
        .ok_or("CED decoder is empty")?
        .output_rows
        .len()
        .min(prefill::DEFAULT_CHUNK_ROWS);
    let mut decoder = 0;
    for width in
        gpu_step::ced::decoder_chunk_widths_from_geometry(c, layers, capacity, decoder_rows)?
    {
        decoder = checked_add(decoder, price(row_bytes(width, decoder_rows)?)?)?;
    }
    let half_chunk = prefill::half_scratch_budget_with(c, rows, half_selected, price)?;
    chunk = checked_add(chunk.max(decoder), half_chunk)?;
    let publication = price(row_bytes(c.head_dim, rows)?)?;
    let seed_rows = plan
        .decoder
        .iter()
        .map(|l| l.seed_rows.len())
        .max()
        .unwrap_or(0)
        .min(prefill::DEFAULT_CHUNK_ROWS);
    let mut seed = 0;
    for width in [c.hidden, c.hidden, c.head_dim] {
        seed = checked_add(seed, price(row_bytes(width, seed_rows)?)?)?;
    }
    let batch_layers = gpu_step::ced::batch_layers();
    let batch_encoding_reserve = if batch_layers == 0 {
        0
    } else {
        super::ced_plan::batches::encoding_reserve_bytes(plan, batch_layers)
            .map_err(|e| e.to_string())?
    };
    let reserve = checked_add(16 * 1024 * 1024, batch_encoding_reserve)?;
    let batch_words = if batch_layers == 0 {
        0
    } else {
        super::ced_plan::batches::maximum_status_words(plan, batch_layers)
            .map_err(|e| e.to_string())?
    };
    let batch_status = price(row_bytes(1, batch_words)?)?;
    let batch_cpu = if batch_layers == 0 {
        0
    } else {
        gpu_step::ced::batch_cpu_bytes(plan, batch_layers)?
    };
    #[cfg(not(test))]
    let diagnostic = 0;
    #[cfg(test)]
    let diagnostic = if batch_layers == 0 {
        0
    } else {
        let mut maximum = 0;
        for group in plan.decoder.chunks(batch_layers) {
            let mut total = 0;
            for unit in super::ced_plan::batches::units(
                plan,
                group[0].layer..group.last().unwrap().layer + 1,
            )
            .map_err(|e| e.to_string())?
            {
                if unit.kind != super::ced_plan::batches::Kind::Seed {
                    total = checked_add(total, price(row_bytes(c.index_topk, unit.rows.len())?)?)?;
                    total = checked_add(
                        total,
                        price(row_bytes(
                            capacity.div_ceil(c.candidate_block),
                            unit.rows.len(),
                        )?)?,
                    )?;
                }
            }
            maximum = maximum.max(total);
        }
        checked_add(maximum, price(4)?)?
    };
    let metal = [
        retained,
        chunk,
        publication,
        seed,
        reserve,
        batch_status,
        diagnostic,
    ]
    .into_iter()
    .try_fold(0, checked_add)?;
    let controls_cpu = attention_gpu::QueryControls::cpu_bytes(plan.candidate_rows.clone())?;
    let hash_cpu = prefill::chunk_hash_cpu_bytes(hash_count, rows)?;
    let cpu = [
        prefill::upload_cpu_bytes(c, projections, rows)?,
        row_bytes(1, plan.prompt_rows)?,
        row_bytes(c.vocab, 2)?,
        hash_cpu,
        controls_cpu,
        batch_cpu,
        batch_encoding_reserve,
        16 * 1024 * 1024,
    ]
    .into_iter()
    .try_fold(0, checked_add)?;
    Ok(CedMemoryEstimate {
        full_chunk_rows: plan.full_chunk_rows,
        retained_residual_bytes: plan.memory.boundary_residual_bytes,
        retained_mix_bytes: plan.memory.boundary_mix_bytes,
        retained_selected_bytes: plan.memory.selected_bytes,
        retained_candidate_bytes: plan.memory.candidate_bytes,
        priced_retained_bytes: retained,
        priced_chunk_bytes: chunk,
        priced_half_chunk_bytes: half_chunk,
        priced_publication_bytes: publication,
        priced_seed_bytes: seed,
        priced_batch_status_bytes: batch_status,
        batch_cpu_bytes: batch_cpu,
        priced_batch_diagnostic_bytes: diagnostic,
        batch_encoding_reserve_bytes: batch_encoding_reserve,
        command_reserve_bytes: reserve,
        incremental_metal_bytes: metal,
        incremental_cpu_bytes: cpu,
        retained_control_cpu_bytes: controls_cpu,
        chunk_hash_cpu_bytes: hash_cpu,
        process_ru_maxrss_before_bytes: None,
        process_ru_maxrss_after_bytes: None,
        observation_method: "metadata-only preload plan; no execution or process RSS observations",
    })
}

struct Provisional<'s> {
    state: &'s mut State,
    committed: bool,
}

impl<'s> Provisional<'s> {
    fn begin(state: &'s mut State) -> Self {
        state.phase = StatePhase::Provisional;
        Self {
            state,
            committed: false,
        }
    }

    fn commit(mut self, addresses: engram::AddressState) {
        self.state.addresses = addresses;
        self.state.prefill_origin = PrefillExecution::ShrinkingCed;
        self.state.phase = StatePhase::Ready;
        self.committed = true;
    }
}

impl Drop for Provisional<'_> {
    fn drop(&mut self) {
        if !self.committed {
            self.state.phase = StatePhase::Poisoned;
        }
    }
}

fn full(
    model: &Forward<'_>,
    state: &mut State,
    tokens: &[u32],
    cancel: Option<&AtomicBool>,
    fallback: Option<String>,
) -> std::result::Result<(Vec<f32>, PrefillReport), StateError> {
    let logits = model.prefill(state, tokens, cancel)?;
    Ok((
        logits,
        PrefillReport {
            execution: PrefillExecution::FullForward,
            fallback,
            counters: None,
            memory: None,
            observed_incremental_metal_bytes: None,
        },
    ))
}

pub(super) fn run(
    model: &Forward<'_>,
    state: &mut State,
    tokens: &[u32],
    cancel: Option<&AtomicBool>,
    arithmetic: Arithmetic,
) -> std::result::Result<(Vec<f32>, PrefillReport), StateError> {
    op::require(state.is_ready(), "CED requires ready state")?;
    op::require(
        Arc::ptr_eq(&state.identity, &model.identity),
        "state belongs to another model",
    )?;
    op::require(state.gpu_step.is_some(), "CED requires GPU state")?;
    op::require(!tokens.is_empty(), "empty prefill suffix")?;
    op::require(
        tokens.len() <= state.capacity().saturating_sub(state.position()),
        "prefill exceeds state capacity",
    )?;
    op::require(
        tokens.iter().all(|&t| (t as usize) < model.token_map.len()),
        "token outside vocabulary",
    )?;
    cancelled(cancel)?;
    let mut request = CedRequest::cold(tokens.len(), state.capacity());
    request.full_chunk_rows = model.prefill_rows;
    request.start_position = state.position();
    let plan =
        match CedPlan::shrinking(&model.config, &model.schedule, &model.engram_hash, &request) {
            Ok(plan) => plan,
            Err(reason) => return full(model, state, tokens, cancel, Some(reason.to_string())),
        };
    state
        .gpu_step
        .as_ref()
        .unwrap()
        .attention()
        .validate_snapshot_position(0)?;
    let mut memory = estimate(model, state.capacity(), &plan)?;
    let admission = metal::evaluate_metal_memory_admission_with_cpu_bytes(
        memory.incremental_metal_bytes,
        memory.incremental_cpu_bytes,
        0,
        model.ctx.memory_signals(),
        true,
    );
    if let Some(reason) = admission.refusal() {
        return full(
            model,
            state,
            tokens,
            cancel,
            Some(format!("CED memory admission: {reason}")),
        );
    }
    let storage = gpu_step::ced::Storage::new(model, state.capacity(), &plan)?;
    cancelled(cancel)?;
    let guard = Provisional::begin(state);
    #[cfg(test)]
    tests::checkpoint("ced_begin", cancel)?;
    let mut addresses = guard.state.addresses;
    let result = guard.state.gpu_step.as_mut().unwrap().run_ced(
        model,
        &plan,
        tokens,
        &mut addresses,
        cancel,
        arithmetic,
        storage,
    )?;
    op::require(
        addresses.position() as usize == plan.prompt_rows,
        "CED address coverage mismatch",
    )?;
    result.counters.validate(&plan)?;
    guard
        .state
        .gpu_step
        .as_ref()
        .unwrap()
        .attention()
        .validate_snapshot_position(plan.prompt_rows)?;
    memory.process_ru_maxrss_after_bytes = process_peak_rss_bytes();
    cancelled(cancel)?;
    guard.commit(addresses);
    Ok((
        result.logits,
        PrefillReport {
            execution: PrefillExecution::ShrinkingCed,
            fallback: None,
            counters: Some(result.counters),
            memory: Some(memory),
            observed_incremental_metal_bytes: Some(result.observed_incremental_metal_bytes),
        },
    ))
}

impl Forward<'_> {
    /// Cold shrinking CED is opt-in. Unsupported requests and admission refusals
    /// select full forward before mutation and report that actual execution.
    pub fn prefill_with_execution(
        &self,
        state: &mut State,
        tokens: &[u32],
        cancel: Option<&AtomicBool>,
        execution: PrefillExecution,
    ) -> std::result::Result<(Vec<f32>, PrefillReport), StateError> {
        match execution {
            PrefillExecution::FullForward => full(self, state, tokens, cancel, None),
            PrefillExecution::ShrinkingCed => {
                run(self, state, tokens, cancel, Arithmetic::Production)
            }
        }
    }
}

#[cfg(test)]
pub(super) mod qualification;
#[cfg(test)]
pub(super) mod tests;
