//! Bounded layer-major prefill with NativeF32 backbone projections and
//! position-ordered causal attention transitions within each layer.
use super::*;

#[cfg(test)]
mod tests;

pub const DEFAULT_CHUNK_ROWS: usize = 256;

#[cfg(test)]
thread_local! {
    pub(super) static ROWWISE_LINEAR_ROWS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
fn with_rowwise_linears<T>(f: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            ROWWISE_LINEAR_ROWS.with(|flag| flag.set(self.0));
        }
    }
    let _restore = Restore(ROWWISE_LINEAR_ROWS.with(|flag| flag.replace(true)));
    f()
}

impl Linear<'_> {
    pub(super) fn encode_rows(
        &self,
        enc: &KernelEncoder,
        x: &MetalTensor,
        y: &MetalTensor,
        rows: usize,
    ) -> std::result::Result<(), metal::MetalError> {
        if rows == 0
            || x.dtype != GgmlType::F32
            || y.dtype != GgmlType::F32
            || self.input.checked_mul(rows).map(|n| n as u64) != Some(x.n_elements())
            || self.output.checked_mul(rows).map(|n| n as u64) != Some(y.n_elements())
        {
            return Err(metal::MetalError::BadShape {
                kernel: "v41_linear_rows",
                detail: "nonempty matching F32 row extents required".into(),
            });
        }
        let grouped = metal::grouped_projection_supported(self.tensor.dtype)
            && self.input.is_multiple_of(32)
            && self
                .tensor
                .dtype
                .storage_layout()
                .is_some_and(|(block, _)| (self.input as u64).is_multiple_of(block));
        #[cfg(test)]
        let grouped = grouped && !ROWWISE_LINEAR_ROWS.with(|flag| flag.get());
        if grouped {
            return metal::encode_mat_mat_grouped_f32(
                self.ctx,
                enc,
                &self
                    .tensor
                    .view_bytes(0, vec![self.input as u64, self.output as u64, 1]),
                &x.view_bytes(0, vec![self.input as u64, rows as u64]),
                &y.view_bytes(0, vec![self.output as u64, rows as u64]),
                self.input,
                self.output,
                1,
                rows,
            );
        }
        for row in 0..rows {
            self.encode(
                enc,
                &x.view_subrange((row * self.input) as u64, vec![self.input as u64]),
                &y.view_subrange((row * self.output) as u64, vec![self.output as u64]),
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
// Test-only token-major reference with batched Engram projections.
fn run_hybrid(
    model: &Forward<'_>,
    state: &mut State,
    tokens: &[u32],
    cancel: Option<&AtomicBool>,
    chunk_rows: usize,
) -> Result<Vec<f32>> {
    op::require(!tokens.is_empty(), "empty prefill suffix")?;
    op::require(
        (1..=DEFAULT_CHUNK_ROWS).contains(&chunk_rows),
        "invalid prefill chunk rows",
    )?;
    op::require(
        !state.poisoned,
        "sequence poisoned by a failed partial token",
    )?;
    op::require(
        Arc::ptr_eq(&state.identity, &model.identity),
        "state belongs to another model",
    )?;
    op::require(state.gpu_step.is_some(), "hybrid prefill requires GPU step")?;
    op::require(
        tokens.len() <= state.capacity().saturating_sub(state.position()),
        "prefill exceeds state capacity",
    )?;
    op::require(
        tokens.iter().all(|&t| (t as usize) < model.token_map.len()),
        "token outside vocabulary",
    )?;
    let mut logits = Vec::new();
    let final_position = state.position() + tokens.len();
    let rows = tokens.len().min(chunk_rows);
    let mut extra_metal = 0u64;
    let mut extra_cpu = 0u64;
    for hash in &model.engram_hash.layers {
        let linear = &model.layers[hash.layer]
            .engram
            .as_ref()
            .ok_or("missing Engram")?
            .projection;
        for width in [linear.input, linear.output] {
            let bytes = width
                .checked_mul(rows)
                .and_then(|n| n.checked_mul(4))
                .ok_or("prefill scratch extent overflow")? as u64;
            let priced = model
                .ctx
                .price_shared_buffer_upper(bytes)
                .map_err(|e| e.to_string())?;
            extra_metal = extra_metal
                .checked_add(priced.priced_upper_bytes)
                .ok_or("prefill scratch budget overflow")?;
            extra_cpu = extra_cpu.max(bytes);
        }
    }
    let admission = metal::evaluate_metal_memory_admission_with_cpu_bytes(
        extra_metal,
        extra_cpu,
        0,
        model.ctx.memory_signals(),
        true,
    );
    if let Some(reason) = admission.refusal() {
        return Err(StateError::MemoryAdmissionDenied(reason).to_string());
    }
    for chunk in tokens.chunks(chunk_rows) {
        cancelled(cancel)?;
        let mut addresses = state.addresses;
        let mut hashes = Vec::with_capacity(chunk.len());
        let mut next_addresses = Vec::with_capacity(chunk.len());
        for &token in chunk {
            let (rows, next) = addresses
                .step_with_config(
                    engram::Token::Compressed(model.token_map[token as usize]),
                    &model.engram_hash,
                )
                .map_err(|e| format!("{e:?}"))?;
            hashes.push(rows);
            next_addresses.push(next);
            addresses = next;
        }
        let mut projections = Vec::new();
        let mut inputs = Vec::new();
        for hash in &model.engram_hash.layers {
            let id = hash.layer;
            let linear = &model.layers[id]
                .engram
                .as_ref()
                .ok_or("missing Engram")?
                .projection;
            let mut values = Vec::with_capacity(chunk.len() * linear.input);
            for rows in &hashes {
                values.extend(
                    model
                        .engram_rows
                        .decode(id, rows.get(&id).ok_or("missing Engram hashes")?)?,
                );
            }
            inputs.push(upload(
                model.ctx,
                &values,
                vec![linear.input as u64, chunk.len() as u64],
            )?);
            projections.push((
                id,
                MetalTensor::zeros_dtype_unstaged(
                    model.ctx,
                    vec![linear.output as u64, chunk.len() as u64],
                    GgmlType::F32,
                )
                .map_err(|e| e.to_string())?,
            ));
        }
        if !projections.is_empty() {
            submit(
                model.ctx,
                "prefill_engram_rows",
                [chunk.len(), projections.len(), 0, 0],
                |enc| {
                    for ((id, output), input) in projections.iter().zip(&inputs) {
                        model.layers[*id]
                            .engram
                            .as_ref()
                            .unwrap()
                            .projection
                            .encode_rows(enc, input, output, chunk.len())?;
                    }
                    Ok(())
                },
            )?;
        }
        for (row, &token) in chunk.iter().enumerate() {
            cancelled(cancel)?;
            let position = state.position();
            let prepared = projections
                .iter()
                .map(|(id, tensor)| {
                    let width = model.layers[*id].engram.as_ref().unwrap().projection.output;
                    (
                        *id,
                        tensor.view_subrange((row * width) as u64, vec![width as u64]),
                    )
                })
                .collect::<Vec<_>>();
            state.poisoned = true;
            logits = state.gpu_step.as_mut().unwrap().run_prepared(
                model,
                position,
                token as usize,
                &hashes[row],
                cancel,
                None,
                None,
                position + 1 == final_position,
                Some(&prepared),
            )?;
            state.addresses = next_addresses[row];
            state.poisoned = false;
        }
    }
    Ok(logits)
}

pub(super) fn allocate(
    ctx: &MetalContext,
    width: usize,
    rows: usize,
    dtype: GgmlType,
) -> Result<MetalTensor> {
    MetalTensor::zeros_dtype_unstaged(ctx, vec![width as u64, rows as u64], dtype)
        .map_err(|e| e.to_string())
}

pub(super) fn row(t: &MetalTensor, width: usize, row: usize) -> MetalTensor {
    t.view_subrange((row * width) as u64, vec![width as u64])
}

pub(super) struct HcRows {
    pub(super) rows: usize,
    raw: MetalTensor,
    pub(super) pre: MetalTensor,
    pub(super) post: MetalTensor,
    pub(super) comb: MetalTensor,
    collapsed: MetalTensor,
    pub(super) normalized: MetalTensor,
}

impl HcRows {
    pub(super) fn widths(c: &Config) -> Vec<usize> {
        vec![24, 4, 4, 16, c.hidden, c.hidden]
    }
    pub(super) fn new(ctx: &MetalContext, c: &Config, rows: usize) -> Result<Self> {
        let f = |w| allocate(ctx, w, rows, GgmlType::F32);
        Ok(Self {
            rows,
            raw: f(24)?,
            pre: f(4)?,
            post: f(4)?,
            comb: f(16)?.view_bytes(0, vec![4, 4, rows as u64]),
            collapsed: f(c.hidden)?,
            normalized: f(c.hidden)?,
        })
    }
    pub(super) fn encode(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        c: &Config,
        projection: &Linear<'_>,
        constants: &[MetalTensor],
        x: &MetalTensor,
        incoming: &MetalTensor,
    ) -> std::result::Result<(), metal::MetalError> {
        projection.encode_rows(enc, x, &self.raw, self.rows)?;
        enc.set_pipeline(&ctx.pipeline("kernel_deepseek_v41_projected_rms")?);
        enc.set_bytes(0, &((4 * c.hidden) as u32));
        enc.set_bytes(1, &c.norm_eps);
        enc.set_tensor(2, x);
        enc.set_tensor(3, &self.raw);
        enc.dispatch(
            objc2_metal::MTLSize {
                width: self.rows,
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        metal::encode_mhc4_controls_rows(
            ctx,
            enc,
            self.rows,
            c.hc_eps,
            &self.raw,
            &constants[0],
            &constants[1],
            &self.pre,
            &self.post,
            &self.comb,
        )?;
        metal::encode_mhc4_collapse_rows(
            ctx,
            enc,
            c.hidden,
            self.rows,
            x,
            incoming,
            &self.collapsed,
        )?;
        metal::encode_rms_norm_mul_rows_f32(
            ctx,
            enc,
            &self.collapsed,
            &constants[2],
            &self.normalized,
            self.rows,
            c.hidden,
            c.norm_eps,
        )
    }
}

pub(super) fn run(
    model: &Forward<'_>,
    state: &mut State,
    tokens: &[u32],
    cancel: Option<&AtomicBool>,
    chunk_rows: usize,
    with_logits: bool,
) -> std::result::Result<Vec<f32>, StateError> {
    run_captured(model, state, tokens, cancel, chunk_rows, with_logits, None)
}

pub(super) fn run_captured(
    model: &Forward<'_>,
    state: &mut State,
    tokens: &[u32],
    cancel: Option<&AtomicBool>,
    chunk_rows: usize,
    with_logits: bool,
    mut capture: Option<&mut dyn capture::Capture>,
) -> std::result::Result<Vec<f32>, StateError> {
    op::require(!tokens.is_empty(), "empty prefill suffix")?;
    op::require(
        (1..=DEFAULT_CHUNK_ROWS).contains(&chunk_rows),
        "invalid prefill chunk rows",
    )?;
    op::require(
        !state.poisoned,
        "sequence poisoned by a failed partial token",
    )?;
    op::require(
        Arc::ptr_eq(&state.identity, &model.identity),
        "state belongs to another model",
    )?;
    op::require(state.gpu_step.is_some(), "prefill requires GPU step")?;
    op::require(
        tokens.len() <= state.capacity().saturating_sub(state.position()),
        "prefill exceeds state capacity",
    )?;
    op::require(
        tokens.iter().all(|&t| (t as usize) < model.token_map.len()),
        "token outside vocabulary",
    )?;
    cancelled(cancel)?;
    let rows = tokens.len().min(chunk_rows);
    if rows > 1 {
        let mut extra = 0u64;
        for width in gpu_step::chunk_widths(model, state.capacity(), rows)? {
            let bytes = width
                .checked_mul(rows)
                .and_then(|n| n.checked_mul(4))
                .ok_or("prefill scratch overflow")? as u64;
            let priced = model
                .ctx
                .price_shared_buffer_upper(bytes)
                .map_err(|e| e.to_string())?;
            extra = extra
                .checked_add(priced.priced_upper_bytes)
                .ok_or("prefill scratch overflow")?;
        }
        // State and resident weights already exist and are reflected in device signals.
        // Charge only the incremental chunk buffers and bounded upload temporaries.
        let upload_width = model
            .layers
            .iter()
            .filter_map(|l| l.engram.as_ref())
            .map(|e| e.projection.input)
            .max()
            .unwrap_or(0)
            .checked_add(4 * model.config.hidden)
            .ok_or("prefill upload overflow")?;
        let cpu = (rows as u64)
            .checked_mul(upload_width as u64)
            .and_then(|n| n.checked_mul(8))
            .ok_or("prefill upload overflow")?;
        let admission = metal::evaluate_metal_memory_admission_with_cpu_bytes(
            extra,
            cpu,
            0,
            model.ctx.memory_signals(),
            true,
        );
        if let Some(reason) = admission.refusal() {
            return Err(StateError::MemoryAdmissionDenied(reason));
        }
    }
    let end = state.position() + tokens.len();
    let mut logits = Vec::new();
    for chunk in tokens.chunks(chunk_rows) {
        cancelled(cancel)?;
        let start = state.position();
        let last = with_logits && start + chunk.len() == end;
        if chunk.len() == 1 {
            logits = model.execute_token(
                state,
                chunk[0],
                cancel,
                None,
                capture
                    .as_mut()
                    .map(|sink| &mut **sink as &mut dyn capture::Capture),
                last,
            )?;
            continue;
        }
        let mut next = state.addresses;
        let mut hashes = Vec::with_capacity(chunk.len());
        for &token in chunk {
            let (rows, addresses) = next
                .step_with_config(
                    engram::Token::Compressed(model.token_map[token as usize]),
                    &model.engram_hash,
                )
                .map_err(|e| format!("{e:?}"))?;
            hashes.push(rows);
            next = addresses;
        }
        state.poisoned = true;
        logits = state.gpu_step.as_mut().unwrap().run_chunk(
            model,
            start,
            chunk,
            &hashes,
            cancel,
            last,
            capture
                .as_mut()
                .map(|sink| &mut **sink as &mut dyn capture::Capture),
        )?;
        state.addresses = next;
        state.poisoned = false;
    }
    Ok(logits)
}
