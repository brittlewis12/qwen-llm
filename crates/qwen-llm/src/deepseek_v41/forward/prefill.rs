//! Bounded layer-major prefill with NativeF32 backbone projections and
//! position-ordered causal attention transitions within each layer.
use super::*;

#[cfg(test)]
mod tests;

pub const DEFAULT_CHUNK_ROWS: usize = 2048;
/// Global engine ceiling, not the width selected by request admission.
pub const MAX_CHUNK_ROWS: usize = 4096;

#[cfg(test)]
thread_local! {
    pub(super) static ROWWISE_LINEAR_ROWS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Tests holding the F32-operand (NativeF32) contract pin the scalar F32
    /// grouped projection; production prefers the shared half-staged MMA tiles.
    pub(super) static F32_LINEAR_ROWS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
pub(super) fn with_f32_linears<T>(f: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            F32_LINEAR_ROWS.with(|flag| flag.set(self.0));
        }
    }
    let _restore = Restore(F32_LINEAR_ROWS.with(|flag| flag.replace(true)));
    f()
}

/// Multi-row dense projections take the shared half-staged simdgroup-MMA
/// tiles for packed and BF16 weights. F32 weights (router, mHC projections)
/// keep the F32-operand grouped kernel: they are small and route-sensitive.
fn mma_linear_rows_selected(dtype: GgmlType, input: usize, rows: usize) -> bool {
    rows > 1
        && dtype != GgmlType::F32
        && crate::metal_forward::mat_mat_dispatch_supported(dtype)
        && input.is_multiple_of(32)
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

/// Half copies of prefill activations for the tensor tile (`_f16act`):
/// one scratch per row set, converted once per distinct input and read by
/// every tensor-routed projection of that input. The tile rounds its
/// operand to half either way, so the results are bitwise identical to
/// the F32-input form; this only removes the per-tile re-read and cast.
pub(super) struct HalfScratch {
    tensor: MetalTensor,
}

fn half_scratch_extent(factors: &[usize]) -> Result<usize> {
    let elements = factors.iter().try_fold(1usize, |n, &factor| {
        n.checked_mul(factor).ok_or("half scratch extent overflow")
    })?;
    HalfScratch::logical_bytes(elements)?;
    Ok(elements)
}

pub(super) fn attention_half_scratch_elements(c: &Config, rows: usize) -> Result<usize> {
    let grouped = half_scratch_extent(&[c.heads, c.head_dim])?;
    let low = half_scratch_extent(&[c.groups, c.output_rank])?;
    half_scratch_extent(&[c.hidden.max(c.query_rank).max(grouped).max(low), rows])
}

pub(super) fn ffn_half_scratch_elements(c: &Config, rows: usize) -> Result<[usize; 2]> {
    Ok([
        half_scratch_extent(&[c.hidden, rows])?,
        half_scratch_extent(&[c.intermediate, c.topk.max(1), rows])?,
    ])
}

/// Incremental half buffers owned by one attention/FFN row set. Price each
/// allocation separately, including buffers unused by arithmetic overrides.
pub(super) fn half_scratch_budget(ctx: &MetalContext, c: &Config, rows: usize) -> Result<u64> {
    if !HalfScratch::selected(ctx) || rows == 0 {
        return Ok(0);
    }
    let attention = attention_half_scratch_elements(c, rows)?;
    let [input, inner] = ffn_half_scratch_elements(c, rows)?;
    [attention, input, inner]
        .into_iter()
        .filter(|&elements| elements != 0)
        .try_fold(0u64, |total, elements| {
            let priced = ctx
                .price_shared_buffer_upper(HalfScratch::logical_bytes(elements)?)
                .map_err(|e| e.to_string())?;
            total
                .checked_add(priced.priced_upper_bytes)
                .ok_or_else(|| "half scratch budget overflow".into())
        })
}

impl HalfScratch {
    fn selected(ctx: &MetalContext) -> bool {
        metal::moe_grouped_tensor_selected(ctx)
    }

    fn logical_bytes(elements: usize) -> Result<u64> {
        u64::try_from(elements)
            .ok()
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| "half scratch byte overflow".into())
    }

    /// Scratch for up to `elements` activations when the tensor tier is
    /// selected on `ctx`; `None` otherwise (nothing would read it).
    pub(super) fn new(ctx: &MetalContext, elements: usize) -> Result<Option<Self>> {
        if !Self::selected(ctx) || elements == 0 {
            return Ok(None);
        }
        Self::logical_bytes(elements)?;
        Ok(Some(Self {
            tensor: allocate(ctx, elements, 1, GgmlType::F16)?,
        }))
    }

    /// Convert the first `elements` of `x` (F32) into the scratch and
    /// return the half view. Serial encoders only: every projection that
    /// reads the view must be encoded before the next `convert`.
    pub(super) fn convert(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        x: &MetalTensor,
        elements: usize,
    ) -> std::result::Result<MetalTensor, metal::MetalError> {
        if elements == 0 || (elements as u64) > self.tensor.n_elements() {
            return Err(metal::MetalError::BadShape {
                kernel: "v41_half_scratch",
                detail: format!(
                    "{elements} activations exceed the {} element half scratch",
                    self.tensor.n_elements()
                ),
            });
        }
        let src = x.view_subrange(0, vec![elements as u64]);
        let dst = self.tensor.view_subrange(0, vec![elements as u64]);
        metal::encode_scatter_offset_f32_to_f16(ctx, enc, &src, &dst, 0, elements)?;
        Ok(dst)
    }
}

impl Linear<'_> {
    /// True when `encode_rows` would take the dense tensor tile for `rows`.
    pub(super) fn tensor_rows_selected(&self, rows: usize) -> bool {
        let mma = mma_linear_rows_selected(self.tensor.dtype, self.input, rows);
        #[cfg(test)]
        let mma = mma
            && !ROWWISE_LINEAR_ROWS.with(|flag| flag.get())
            && !F32_LINEAR_ROWS.with(|flag| flag.get());
        mma && metal::moe_grouped_tensor_selected(self.ctx)
            && metal::moe_grouped_generic_layout(self.tensor.dtype)
                .is_some_and(|l| self.input.is_multiple_of(l.block_elems))
    }

    pub(super) fn encode_rows(
        &self,
        enc: &KernelEncoder,
        x: &MetalTensor,
        y: &MetalTensor,
        rows: usize,
    ) -> std::result::Result<(), metal::MetalError> {
        self.encode_rows_with(enc, x, None, y, rows)
    }

    /// `encode_rows` with an optional half copy of `x` (`x_half`, the first
    /// `input * rows` elements already converted) that the dense tensor
    /// tile reads instead of re-casting `x`; ignored on every other route.
    pub(super) fn encode_rows_with(
        &self,
        enc: &KernelEncoder,
        x: &MetalTensor,
        x_half: Option<&MetalTensor>,
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
        let mma = mma_linear_rows_selected(self.tensor.dtype, self.input, rows);
        #[cfg(test)]
        let mma = mma
            && !ROWWISE_LINEAR_ROWS.with(|flag| flag.get())
            && !F32_LINEAR_ROWS.with(|flag| flag.get());
        if mma {
            let x = x.view_bytes(0, vec![self.input as u64, rows as u64]);
            let y = y.view_bytes(0, vec![self.output as u64, rows as u64]);
            // The dense form of the Metal 4 tensor tile where the tier
            // exists: same half operands and F32 sums as the MMA tile.
            if self.tensor_rows_selected(rows) {
                let elements = (self.input * rows) as u64;
                let x_half = x_half.map(|h| {
                    if h.dtype == GgmlType::F16 && h.n_elements() >= elements {
                        Ok(h.view_bytes(0, vec![self.input as u64, rows as u64]))
                    } else {
                        Err(metal::MetalError::BadShape {
                            kernel: "v41_linear_rows",
                            detail: format!(
                                "half input must be F16 with >= {elements} elements, got {:?} x {}",
                                h.dtype,
                                h.n_elements()
                            ),
                        })
                    }
                });
                let x_half = x_half.transpose()?;
                return metal::encode_mat_mat_tensor_f32(
                    self.ctx,
                    enc,
                    &self.tensor,
                    x_half.as_ref().unwrap_or(&x),
                    &y,
                    self.input,
                    self.output,
                    rows,
                );
            }
            return crate::metal_forward::encode_mat_mat_dispatch_with_policy(
                self.ctx,
                enc,
                &self.tensor,
                &x,
                &y,
                self.input,
                self.output,
                rows,
                false,
            )
            .map_err(|error| match error {
                crate::metal_forward::MfError::Metal(error) => error,
                other => metal::MetalError::BadShape {
                    kernel: "v41_linear_rows",
                    detail: format!("mma projection: {other}"),
                },
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
        (1..=MAX_CHUNK_ROWS).contains(&chunk_rows),
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
    let tensor = MetalTensor::zeros_dtype_unstaged(ctx, vec![width as u64, rows as u64], dtype)
        .map_err(|e| e.to_string())?;
    #[cfg(test)]
    tests::lifetime::allocated(ctx, &tensor);
    Ok(tensor)
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
        (1..=MAX_CHUNK_ROWS).contains(&chunk_rows),
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
        // One summary per process: which routed-expert route each layer takes
        // at the production chunk width (artifact-dependent; see PERF-LOG).
        static ROUTE_CENSUS_ONCE: std::sync::Once = std::sync::Once::new();
        ROUTE_CENSUS_ONCE.call_once(|| {
            let census =
                super::ffn_gpu::routed_route_census(&model.config, &model.layers, rows);
            tracing::info!(
                target: "qwen_diag",
                "deepseek41: prefill rows={rows} routed experts: half_grouped={} f32_grouped={} native_rows={} per_row={}",
                census[0],
                census[1],
                census[2],
                census[3]
            );
        });
    }
    let cost = model.prefill_cost(state.capacity(), rows)?;
    planning::Cost {
        metal: cost.metal,
        cpu: cost
            .cpu
            .checked_add(state.caller_cpu_reserve)
            .ok_or("prefill reserve overflow")?,
    }
    .admit(model.ctx.memory_signals())?;
    op::require(
        state
            .prefill_rows
            .is_none_or(|selected| selected == chunk_rows),
        "prefill geometry is fixed for this request",
    )?;
    state.prefill_rows = Some(chunk_rows);
    state.last_prefill = None;
    let end = state.position() + tokens.len();
    let mut logits = Vec::new();
    for chunk in tokens.chunks(chunk_rows) {
        // Admission prices one chunk, not the whole suffix. Drop its completed
        // commands and their resource autoreleases before allocating the next.
        logits = objc2::rc::autoreleasepool(|_| -> std::result::Result<Vec<f32>, StateError> {
            cancelled(cancel)?;
            let start = state.position();
            let last = with_logits && start + chunk.len() == end;
            if chunk.len() == 1 {
                return Ok(model.execute_token(
                    state,
                    chunk[0],
                    cancel,
                    None,
                    capture
                        .as_mut()
                        .map(|sink| &mut **sink as &mut dyn capture::Capture),
                    last,
                )?);
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
            let logits = state.gpu_step.as_mut().unwrap().run_chunk(
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
            Ok(logits)
        })?;
        #[cfg(test)]
        tests::lifetime::completed_chunk(model.ctx, state, chunk.len());
    }
    state.last_prefill = Some(PrefillReport {
        start: end - tokens.len(),
        tokens: tokens.len(),
        selected_rows: chunk_rows,
    });
    Ok(logits)
}
