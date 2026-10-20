//! Optional, reusable GPU output storage for target features and row logits.
//! Captures are means of completed post-FFN mHC streams, before any norm.
//! These entry points advance the target; they do not accept drafts or rewind it.

use super::*;
use objc2_metal::MTLResource;
use std::ops::Range;

/// CPU-only plan. Tap order is the caller's order, not execution order.
/// Zero feature capacity requires no taps; zero logit capacity omits the head.
#[derive(Clone, Debug)]
pub struct Glm5NextRowBankPlan {
    taps: Vec<u32>,
    hidden: usize,
    vocab: usize,
    blocks: usize,
    feature_rows: usize,
    logit_rows: usize,
}

impl Glm5NextRowBankPlan {
    pub fn new(
        config: &Glm5NextConfig,
        taps: &[u32],
        feature_rows: usize,
        logit_rows: usize,
    ) -> Result<Self> {
        if config.hc_streams != 4 {
            return invalid("row captures require four mHC streams");
        }
        Self::from_dimensions(
            config.hidden_size as usize,
            config.vocab_size as usize,
            config.blocks.len(),
            taps,
            feature_rows,
            logit_rows,
        )
    }

    fn from_dimensions(
        hidden: usize,
        vocab: usize,
        blocks: usize,
        taps: &[u32],
        feature_rows: usize,
        logit_rows: usize,
    ) -> Result<Self> {
        if hidden == 0 || vocab == 0 || blocks == 0 {
            return invalid("row bank dimensions must be nonzero");
        }
        if taps.is_empty() != (feature_rows == 0) || (feature_rows == 0 && logit_rows == 0) {
            return invalid("row bank needs features with taps and/or logits");
        }
        for (i, &tap) in taps.iter().enumerate() {
            if tap as usize >= blocks || taps[..i].contains(&tap) {
                return invalid("capture taps must be unique executed blocks");
            }
        }
        let plan = Self {
            taps: taps.to_vec(),
            hidden,
            vocab,
            blocks,
            feature_rows,
            logit_rows,
        };
        for spec in plan.allocation_specs() {
            if spec.bytes().is_none_or(|n| n > isize::MAX as u64) {
                return invalid("row bank size exceeds addressable storage");
            }
        }
        Ok(plan)
    }

    pub fn taps(&self) -> &[u32] {
        &self.taps
    }

    pub fn feature_capacity(&self) -> usize {
        self.feature_rows
    }

    pub fn logit_capacity(&self) -> usize {
        self.logit_rows
    }

    /// The exact optional buffers allocated by this plan; ordinary sessions
    /// and ordinary packed prefill allocate none of them.
    pub fn allocation_specs(&self) -> Vec<BufferSpec> {
        let mut specs = Vec::new();
        for (name, shape, enabled) in [
            (
                "features",
                vec![
                    self.hidden as u64,
                    self.taps.len() as u64,
                    self.feature_rows as u64,
                ],
                self.feature_rows > 0,
            ),
            (
                "row_logits",
                vec![self.vocab as u64, self.logit_rows as u64],
                self.logit_rows > 0,
            ),
        ] {
            if enabled {
                specs.push(BufferSpec {
                    name,
                    dtype: BufferType::F32,
                    shape,
                    fill: 0.0,
                });
            }
        }
        specs
    }

    pub fn priced_bytes(&self, price: memory::BufferPricer<'_>) -> Result<u64> {
        Ok(memory::priced(&self.allocation_specs(), price)?)
    }

    fn matches(&self, config: &Glm5NextConfig) -> bool {
        config.hc_streams == 4
            && self.hidden == config.hidden_size as usize
            && self.vocab == config.vocab_size as usize
            && self.blocks == config.blocks.len()
    }

    fn ranges(&self, start: usize, rows: usize, all_logits: bool) -> Result<OutputRanges> {
        let end = start
            .checked_add(rows)
            .filter(|_| rows > 0)
            .ok_or_else(|| {
                Glm5NextMetalError::Invalid("empty or overflowing row output span".into())
            })?;
        if all_logits
            && (rows > self.logit_rows || (!self.taps.is_empty() && rows > self.feature_rows))
        {
            return invalid("verification rows exceed feature or all-row logit capacity");
        }
        let logit_rows = if all_logits {
            rows
        } else {
            usize::from(self.logit_rows > 0)
        };
        Ok(OutputRanges {
            input: start..end,
            features: end - rows.min(self.feature_rows)..end,
            logits: end - logit_rows..end,
        })
    }

    fn feature_offset(&self, row: usize, tap: usize) -> usize {
        (row * self.taps.len() + tap) * self.hidden
    }
}

/// A weights-instance/device-bound reusable bank, allocated separately from
/// session scratch. Each successful call replaces its contents; it is not a
/// rolling history cache. Consume/append returned rows before reusing it.
pub struct Glm5NextRowBank {
    plan: Glm5NextRowBankPlan,
    weights_instance: u64,
    device: u64,
    features: Option<MetalTensor>,
    logits: Option<MetalTensor>,
    priced_bytes: u64,
}

impl Glm5NextRowBank {
    /// Admit and allocate only the plan's additional storage. Resident target
    /// weights and session buffers are already reflected in device telemetry.
    /// Leave the session's dynamic command/allocator reserve unspent.
    pub fn new(
        ctx: &MetalContext,
        weights: &Glm5NextWeights,
        plan: Glm5NextRowBankPlan,
    ) -> Result<Self> {
        if !plan.matches(&weights.config)
            || weights.embedding.buffer.device().registryID() != ctx.device.registryID()
        {
            return invalid("row bank target geometry or device mismatch");
        }
        let _allocation = ctx.begin_allocation_transaction();
        let priced_bytes = plan.priced_bytes(&device_price(ctx))?;
        let admission = evaluate_metal_memory_admission(
            priced_bytes,
            memory::DYNAMIC_RESERVE_BYTES,
            ctx.memory_signals(),
            true,
        );
        if let Some(denied) = admission.refusal() {
            return Err(Glm5NextMetalError::MemoryAdmission {
                denied,
                budget_bytes: admission_budget(&admission),
                advice: CapacityAdvice::NotEvaluated,
            });
        }
        Self::allocate_admitted(ctx, weights, plan, priced_bytes)
    }

    /// Caller holds the allocation transaction and has admitted these exact
    /// buffers, either alone or as part of a complete request allocation.
    pub(super) fn allocate_admitted(
        ctx: &MetalContext,
        weights: &Glm5NextWeights,
        plan: Glm5NextRowBankPlan,
        priced_bytes: u64,
    ) -> Result<Self> {
        if !plan.matches(&weights.config)
            || weights.embedding.buffer.device().registryID() != ctx.device.registryID()
        {
            return invalid("row bank target geometry or device mismatch");
        }
        let mut buffers = SpecBuffers::allocate(ctx, "row_bank", &plan.allocation_specs())?;
        let features = (plan.feature_rows > 0)
            .then(|| buffers.take("features"))
            .transpose()?;
        let logits = (plan.logit_rows > 0)
            .then(|| buffers.take("row_logits"))
            .transpose()?;
        buffers.finish()?;
        Ok(Self {
            plan,
            weights_instance: weights.instance,
            device: ctx.device.registryID(),
            features,
            logits,
            priced_bytes,
        })
    }

    pub fn plan(&self) -> &Glm5NextRowBankPlan {
        &self.plan
    }

    pub fn priced_bytes(&self) -> u64 {
        self.priced_bytes
    }

    pub(super) fn validate(&self, ctx: &MetalContext, session: &Glm5NextSession<'_>) -> Result<()> {
        if self.device != ctx.device.registryID()
            || self.weights_instance != session.weights.instance
            || !self.plan.matches(&session.weights.config)
        {
            return invalid("row bank belongs to another target or device");
        }
        Ok(())
    }

    pub(super) fn validate_all_rows(&self, rows: usize) -> Result<()> {
        self.plan.ranges(0, rows, true)?;
        Ok(())
    }

    pub(super) fn completed_all_rows(
        &self,
        start: usize,
        rows: usize,
    ) -> Result<Glm5NextCapturedRows<'_>> {
        Ok(RowOutput {
            bank: self,
            ranges: self.plan.ranges(start, rows, true)?,
        }
        .completed())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct OutputRanges {
    input: Range<usize>,
    features: Range<usize>,
    logits: Range<usize>,
}

/// Completed row outputs, borrowing a bank until the caller consumes them.
/// The target has advanced through `input_positions`; nothing here constitutes
/// speculative acceptance. Logit row at position p predicts token p + 1.
/// Tensor views must not be retained across bank reuse.
pub struct Glm5NextCapturedRows<'b> {
    bank: &'b Glm5NextRowBank,
    ranges: OutputRanges,
    features: Option<MetalTensor>,
    logits: Option<MetalTensor>,
}

impl Glm5NextCapturedRows<'_> {
    pub fn input_positions(&self) -> Range<usize> {
        self.ranges.input.clone()
    }

    pub fn feature_positions(&self) -> Range<usize> {
        self.ranges.features.clone()
    }

    pub fn logit_positions(&self) -> Range<usize> {
        self.ranges.logits.clone()
    }

    pub fn taps(&self) -> &[u32] {
        self.bank.plan.taps()
    }

    /// F32 memory order `[token][tap][hidden]`, GGML shape `[hidden, taps, rows]`.
    /// No stream axis, target norm, or next-block norm is included.
    pub fn features(&self) -> Option<&MetalTensor> {
        self.features.as_ref()
    }

    /// F32 memory order `[token][vocabulary]`, GGML shape `[vocabulary, rows]`.
    pub fn logits(&self) -> Option<&MetalTensor> {
        self.logits.as_ref()
    }
}

pub(super) struct RowOutput<'b> {
    bank: &'b Glm5NextRowBank,
    ranges: OutputRanges,
}

impl<'b> RowOutput<'b> {
    pub(super) fn encode_capture(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        quarter: &MetalTensor,
        block: usize,
        start: usize,
        rows: usize,
        residual: &MetalTensor,
    ) -> Result<()> {
        let Some(tap) = self
            .bank
            .plan
            .taps
            .iter()
            .position(|&tap| tap as usize == block)
        else {
            return Ok(());
        };
        let Some(features) = &self.bank.features else {
            return Ok(());
        };
        let h = self.bank.plan.hidden;
        for position in
            start.max(self.ranges.features.start)..(start + rows).min(self.ranges.features.end)
        {
            let source =
                residual.view_subrange(((position - start) * 4 * h) as u64, vec![h as u64, 4]);
            let offset = self
                .bank
                .plan
                .feature_offset(position - self.ranges.features.start, tap);
            let destination = features.view_subrange(offset as u64, vec![h as u64]);
            encode_mhc4_collapse(ctx, enc, h, &source, quarter, &destination)?;
        }
        Ok(())
    }

    pub(super) fn logits_at(&self, position: usize) -> Option<MetalTensor> {
        if !self.ranges.logits.contains(&position) {
            return None;
        }
        let v = self.bank.plan.vocab;
        Some(self.bank.logits.as_ref()?.view_subrange(
            ((position - self.ranges.logits.start) * v) as u64,
            vec![v as u64],
        ))
    }

    fn completed(self) -> Glm5NextCapturedRows<'b> {
        let p = &self.bank.plan;
        let features = self.bank.features.as_ref().map(|t| {
            t.view_subrange(
                0,
                vec![
                    p.hidden as u64,
                    p.taps.len() as u64,
                    self.ranges.features.len() as u64,
                ],
            )
        });
        let logits = self
            .bank
            .logits
            .as_ref()
            .map(|t| t.view_subrange(0, vec![p.vocab as u64, self.ranges.logits.len() as u64]));
        Glm5NextCapturedRows {
            bank: self.bank,
            ranges: self.ranges,
            features,
            logits,
        }
    }
}

impl Glm5NextSession<'_> {
    /// Ordinary scalar decode with GPU-resident mean-stream captures and an
    /// optional final logit row. No diagnostic observers or feature readback.
    pub fn forward_with_captures<'b>(
        &mut self,
        ctx: &MetalContext,
        token: u32,
        bank: &'b mut Glm5NextRowBank,
    ) -> Result<Glm5NextCapturedRows<'b>> {
        self.validate_request(&[token])?;
        bank.validate(ctx, self)?;
        let output = RowOutput {
            bank,
            ranges: bank.plan.ranges(self.position, 1, false)?,
        };
        self.step_captured(ctx, token, &output)?;
        Ok(output.completed())
    }

    /// Prefill on the session's existing schedule, retaining only the trailing
    /// `bank.plan().feature_capacity()` rows of this call, and optionally its
    /// last logit row. Earlier prompt rows run without capture or head work.
    /// Packed/serial selection and arithmetic are unchanged. On failure,
    /// completed chunks stay advanced; no output view is published.
    pub fn prefill_with_captures<'b>(
        &mut self,
        ctx: &MetalContext,
        tokens: &[u32],
        bank: &'b mut Glm5NextRowBank,
    ) -> Result<Glm5NextCapturedRows<'b>> {
        self.prefill_with_captures_and_checkpoint(ctx, tokens, bank, &mut || Ok(()))
    }

    /// Capture prefill with a checkpoint before each native packed chunk (each
    /// token only when the session is already serial). A refusal returns typed
    /// `Cancelled`, preserving the healthy completed prefix and its position.
    /// No capture view is published on cancellation. Chunk widths, arithmetic
    /// and the whole-call trailing feature window match `prefill_with_captures`.
    pub fn prefill_with_captures_and_checkpoint<'b>(
        &mut self,
        ctx: &MetalContext,
        tokens: &[u32],
        bank: &'b mut Glm5NextRowBank,
        checkpoint: &mut dyn FnMut() -> std::result::Result<(), String>,
    ) -> Result<Glm5NextCapturedRows<'b>> {
        self.validate_request(tokens)?;
        bank.validate(ctx, self)?;
        let output = RowOutput {
            bank,
            ranges: bank.plan.ranges(self.position, tokens.len(), false)?,
        };
        if let Some(rows) = self.packed.as_ref().map(|p| p.rows()) {
            for chunk in tokens.chunks(rows) {
                checkpoint().map_err(Glm5NextMetalError::Cancelled)?;
                self.step_packed_with_row_output(ctx, chunk, false, Some(&output))?;
            }
        } else {
            for &token in tokens {
                checkpoint().map_err(Glm5NextMetalError::Cancelled)?;
                self.step_captured(ctx, token, &output)?;
            }
        }
        Ok(output.completed())
    }

    /// Execute one configured packed candidate block, exposing EVERY row's native
    /// output tail and all requested taps. Requires preallocated packed scratch
    /// and a bank large enough for the whole block (no trailing truncation).
    ///
    /// This is a verification foundation, not a speculative transaction: the
    /// entire input advances KDA/MLA state. The caller must checkpoint beforehand
    /// and restore/replay a rejected suffix; no acceptance/commit API is implied.
    /// Lineage and precision are unchanged. Exact preserves F32 operands but
    /// generic HC/non-Q8 latent reductions are
    /// not universally bitwise serial. Use numerical comparison/serial fallback
    /// before claiming greedy-equivalent speculation.
    pub fn verify_packed_with_captures<'b>(
        &mut self,
        ctx: &MetalContext,
        tokens: &[u32],
        bank: &'b mut Glm5NextRowBank,
    ) -> Result<Glm5NextCapturedRows<'b>> {
        self.validate_request(tokens)?;
        bank.validate(ctx, self)?;
        let output = RowOutput {
            bank,
            ranges: bank.plan.ranges(self.position, tokens.len(), true)?,
        };
        self.step_packed_with_row_output(ctx, tokens, false, Some(&output))?;
        Ok(output.completed())
    }

    pub(super) fn scalar_rows_with_captures<'b>(
        &mut self,
        ctx: &MetalContext,
        tokens: &[u32],
        bank: &'b mut Glm5NextRowBank,
        all_logits: bool,
    ) -> Result<Glm5NextCapturedRows<'b>> {
        self.validate_request(tokens)?;
        bank.validate(ctx, self)?;
        let output = RowOutput {
            bank,
            ranges: bank.plan.ranges(self.position, tokens.len(), all_logits)?,
        };
        for &token in tokens {
            self.step_captured(ctx, token, &output)?;
        }
        Ok(output.completed())
    }

    fn step_captured(
        &mut self,
        ctx: &MetalContext,
        token: u32,
        output: &RowOutput<'_>,
    ) -> Result<()> {
        let mut no_captures = |_: Glm5NextSiteCapture, _: &[f32]| {};
        let mut hooks = interventions::Hooks {
            module: &[],
            captures: &[],
            observer: &mut no_captures,
        };
        self.step_with_row_output(
            ctx,
            token,
            false,
            &[],
            &mut |_, _, _| {},
            &mut hooks,
            Some(output),
        )?;
        Ok(())
    }
}

#[cfg(test)]
#[path = "tests/capture.rs"]
mod tests;
