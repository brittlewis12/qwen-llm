//! Same-session append transactions. Only mutable recurrent state and pending
//! rings are copied; MLA latent/pooled prefixes stay in their original buffers.
//! An exclusive session borrow protects those prefixes for the transaction's
//! entire lifetime. No checkpoint can be restored outside that borrow.

use super::*;
use crate::metal::BlitEncoder;
use objc2_metal::MTLResource;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};

static SESSION_INSTANCES: AtomicU64 = AtomicU64::new(1);

pub(super) fn next_session_instance() -> Result<u64> {
    SESSION_INSTANCES
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
        .map_err(|_| Glm5NextMetalError::Invalid("session identity exhausted".into()))
}

fn mismatch<T>(message: &str) -> Result<T> {
    Err(Glm5NextMetalError::AppendCheckpointMismatch(message.into()))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Owner {
    session: u64,
    weights: u64,
    device: u64,
}

impl Owner {
    fn of(ctx: &MetalContext, session: &Glm5NextSession<'_>) -> Self {
        Self {
            session: session.instance,
            weights: session.weights.instance,
            device: ctx.device.registryID(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Stamp {
    owner: Owner,
    generation: u64,
    position: usize,
    end: usize,
    lineage: Option<PackedLineage>,
    precision: Option<FastPrecision>,
    pool: usize,
}

impl Stamp {
    fn validate(
        &self,
        active: Option<Self>,
        owner: Owner,
        position: usize,
        lineage: Option<PackedLineage>,
        precision: Option<FastPrecision>,
        pool: usize,
    ) -> Result<()> {
        if active != Some(*self) || owner != self.owner {
            return mismatch("stale or foreign append transaction");
        }
        if lineage != self.lineage || precision != self.precision || pool != self.pool {
            return mismatch("append transaction arithmetic or pool geometry changed");
        }
        if !(self.position..=self.end).contains(&position) {
            return mismatch("position left the append transaction's range");
        }
        Ok(())
    }
}

/// Numerical provenance, not a promise that packed reductions equal decode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Glm5NextVerificationArithmetic {
    Packed {
        lineage: PackedLineage,
        precision: Option<FastPrecision>,
    },
    Serial,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Ready,
    Running,
    Verified(Glm5NextVerificationArithmetic),
    /// A completed command failed a host-side kernel-status check.
    Failed,
    /// Completion/integrity is not established; compact restore cannot repair
    /// potentially damaged historical buffers omitted from the checkpoint.
    Fatal,
    Finished,
}

impl Phase {
    fn after_execution_error(error: &Glm5NextMetalError) -> Self {
        match error {
            Glm5NextMetalError::KernelValidation { .. } => Self::Failed,
            _ => Self::Fatal,
        }
    }

    fn can_restore(self) -> bool {
        matches!(self, Self::Ready | Self::Verified(_) | Self::Failed)
    }

    fn poisons_on_drop(self) -> bool {
        !matches!(self, Self::Ready | Self::Finished)
    }

    fn retained_inputs(
        self,
        count: usize,
        inputs: usize,
    ) -> Result<Glm5NextVerificationArithmetic> {
        let Self::Verified(arithmetic) = self else {
            return invalid("commit requires completed verification");
        };
        if count > inputs {
            return invalid("retained inputs exceed verified inputs");
        }
        Ok(arithmetic)
    }
}

enum Regions {
    Kda {
        conv: MetalTensor,
        state: MetalTensor,
    },
    Mla {
        pending: MetalTensor,
    },
}

fn region_specs(config: &Glm5NextConfig, mixer: MixerKind) -> Vec<BufferSpec> {
    match mixer {
        MixerKind::Kda => memory::kda_state_specs(config),
        MixerKind::Mla => vec![memory::mla_pending_spec(config)],
    }
}

/// Optional device storage for one session, reusable across append transactions.
/// It is not a portable snapshot or a standalone rewind capability. No contents
/// become restorable until the complete capture command has succeeded.
pub struct Glm5NextAppendCheckpoint {
    owner: Owner,
    generation: u64,
    active: Option<Stamp>,
    regions: Vec<Regions>,
    priced_bytes: u64,
}

impl Glm5NextAppendCheckpoint {
    /// Additional storage only, independent of context length and verify width.
    pub fn priced_bytes_for(
        config: &Glm5NextConfig,
        price: memory::BufferPricer<'_>,
    ) -> Result<u64> {
        let mut bytes = 0u64;
        for block in &config.blocks {
            bytes = bytes
                .checked_add(memory::priced(&region_specs(config, block.mixer), price)?)
                .ok_or(Glm5NextError::Overflow("append checkpoint prices"))?;
        }
        Ok(bytes)
    }

    pub fn new(ctx: &MetalContext, session: &Glm5NextSession<'_>) -> Result<Self> {
        if session.poisoned {
            return Err(Glm5NextMetalError::Poisoned);
        }
        if session.weights.embedding.buffer.device().registryID() != ctx.device.registryID() {
            return mismatch("checkpoint device differs from session");
        }
        let _allocation = ctx.begin_allocation_transaction();
        let config = &session.weights.config;
        let priced_bytes = Self::priced_bytes_for(config, &device_price(ctx))?;
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
        Self::allocate_admitted(ctx, session, priced_bytes)
    }

    /// Caller holds the allocation transaction and has admitted the full region
    /// inventory, including when it is part of an aggregate request admission.
    pub(super) fn allocate_admitted(
        ctx: &MetalContext,
        session: &Glm5NextSession<'_>,
        priced_bytes: u64,
    ) -> Result<Self> {
        if session.poisoned {
            return Err(Glm5NextMetalError::Poisoned);
        }
        if session.weights.embedding.buffer.device().registryID() != ctx.device.registryID() {
            return mismatch("checkpoint device differs from session");
        }
        let config = &session.weights.config;
        let mut regions = Vec::with_capacity(config.blocks.len());
        for block in &config.blocks {
            let mut buffers = SpecBuffers::allocate(
                ctx,
                "append_checkpoint",
                &region_specs(config, block.mixer),
            )?;
            regions.push(match block.mixer {
                MixerKind::Kda => Regions::Kda {
                    conv: buffers.take("conv")?,
                    state: buffers.take("state")?,
                },
                MixerKind::Mla => Regions::Mla {
                    pending: buffers.take("pending")?,
                },
            });
            buffers.finish()?;
        }
        let checkpoint = Self {
            owner: Owner::of(ctx, session),
            generation: 0,
            active: None,
            regions,
            priced_bytes,
        };
        checkpoint.validate_regions(ctx, session)?;
        Ok(checkpoint)
    }

    pub fn priced_bytes(&self) -> u64 {
        self.priced_bytes
    }

    fn pairs<'s, 'c>(
        &'c self,
        session: &'s Glm5NextSession<'_>,
    ) -> Result<Vec<(&'s MetalTensor, &'c MetalTensor)>> {
        if self.regions.len() != session.layers.len() {
            return mismatch("checkpoint layer count differs");
        }
        let mut pairs = Vec::new();
        for (layer, saved) in session.layers.iter().zip(&self.regions) {
            match (layer, saved) {
                (LayerState::Kda { conv, state }, Regions::Kda { conv: c, state: s }) => {
                    pairs.push((conv, c));
                    pairs.push((state, s));
                }
                (LayerState::Mla { pending, .. }, Regions::Mla { pending: p }) => {
                    pairs.push((pending, p))
                }
                _ => return mismatch("checkpoint layer kind differs"),
            }
        }
        Ok(pairs)
    }

    fn validate_regions(&self, ctx: &MetalContext, session: &Glm5NextSession<'_>) -> Result<()> {
        if self.owner != Owner::of(ctx, session) {
            return mismatch("checkpoint belongs to another session, weights or device");
        }
        for (live, saved) in self.pairs(session)? {
            if live.dtype != saved.dtype
                || live.shape != saved.shape
                || live.n_bytes() != saved.n_bytes()
            {
                return mismatch("checkpoint region geometry differs");
            }
            for tensor in [live, saved] {
                if !tensor.is_writable()
                    || tensor.buffer.device().registryID() != self.owner.device
                    || !tensor.offset.is_multiple_of(4)
                    || !tensor.n_bytes().is_multiple_of(4)
                    || tensor.n_bytes() == 0
                    || tensor
                        .offset
                        .checked_add(tensor.n_bytes())
                        .is_none_or(|end| end > tensor.buffer.length() as u64)
                {
                    return mismatch(
                        "checkpoint region is not writable aligned storage on this device",
                    );
                }
            }
            if live.buffer.contents().as_ptr() == saved.buffer.contents().as_ptr() {
                return mismatch("checkpoint aliases live state");
            }
        }
        Ok(())
    }

    fn copy_regions(
        &self,
        ctx: &MetalContext,
        session: &Glm5NextSession<'_>,
        restore: bool,
    ) -> Result<()> {
        self.validate_regions(ctx, session)?;
        let command = ctx
            .queue
            .commandBuffer()
            .ok_or_else(|| Glm5NextMetalError::Invalid("no checkpoint command buffer".into()))?;
        let blit = BlitEncoder::try_begin(&command)?;
        for (live, saved) in self.pairs(session)? {
            let (src, dst) = if restore {
                (saved, live)
            } else {
                (live, saved)
            };
            blit.copy_tensor(src, dst);
        }
        blit.end();
        command.commit();
        wait_completed(&command)?;
        Ok(())
    }
}

/// Tentative logits only. Hidden features cannot be published from this view.
/// Borrowing the transaction prevents restoration/reuse while this view lives.
/// Cloned tensor views must not be retained across transaction/bank reuse.
pub struct Glm5NextVerificationRows<'a> {
    logits: MetalTensor,
    positions: std::ops::Range<usize>,
    arithmetic: Glm5NextVerificationArithmetic,
    _transaction: PhantomData<&'a ()>,
}

impl Glm5NextVerificationRows<'_> {
    pub fn logits(&self) -> &MetalTensor {
        &self.logits
    }
    /// Explicit CPU copy for acceptance after verification has completed. Hidden
    /// captures stay device-resident; verification itself performs no logit copy.
    pub fn read_logits(&self) -> Result<Vec<f32>> {
        read_f32(&self.logits)
    }
    pub fn input_positions(&self) -> std::ops::Range<usize> {
        self.positions.clone()
    }
    pub fn arithmetic(&self) -> Glm5NextVerificationArithmetic {
        self.arithmetic
    }
}

/// Only retained features are published. Replay logits are state-repair outputs,
/// not permission to resample a correction/bonus already chosen by acceptance.
pub struct Glm5NextCommittedRows<'a> {
    rows: Option<Glm5NextCapturedRows<'a>>,
    verification_arithmetic: Glm5NextVerificationArithmetic,
    retained_arithmetic: Option<Glm5NextVerificationArithmetic>,
}

impl Glm5NextCommittedRows<'_> {
    /// None for zero retained inputs. Full retention exposes the verified rows;
    /// partial retention exposes replayed features and only the final replay logit.
    pub fn rows(&self) -> Option<&Glm5NextCapturedRows<'_>> {
        self.rows.as_ref()
    }

    pub fn verification_arithmetic(&self) -> Glm5NextVerificationArithmetic {
        self.verification_arithmetic
    }

    /// Full retention keeps verification arithmetic; partial retention is Serial.
    /// None means rollback with no retained inputs.
    pub fn retained_arithmetic(&self) -> Option<Glm5NextVerificationArithmetic> {
        self.retained_arithmetic
    }
}

/// Holds the only mutable session/checkpoint/bank borrows. Thus reset, snapshot
/// restoration, lineage changes and unrelated execution cannot occur between
/// capture and restore. Dropping unresolved work poisons without launching GPU
/// work; call `rollback` explicitly to leave a healthy baseline instead.
#[must_use = "verify and commit a prefix, or explicitly roll back the append"]
pub struct Glm5NextAppendTransaction<'a, 'w> {
    ctx: &'a MetalContext,
    session: &'a mut Glm5NextSession<'w>,
    checkpoint: &'a mut Glm5NextAppendCheckpoint,
    bank: &'a mut Glm5NextRowBank,
    inputs: &'a [u32],
    stamp: Stamp,
    phase: Phase,
    #[cfg(test)]
    fail_replay_completion: bool,
}

impl<'w> Glm5NextSession<'w> {
    /// Capture before one candidate append. `inputs[0]` is the anchor, remaining
    /// entries are draft inputs; never include a replacement/bonus to be emitted
    /// after verification. The checkpoint is single-session; a new transaction
    /// always recaptures, even at the same position after a reset/history edit.
    pub fn begin_append<'a>(
        &'a mut self,
        ctx: &'a MetalContext,
        checkpoint: &'a mut Glm5NextAppendCheckpoint,
        bank: &'a mut Glm5NextRowBank,
        inputs: &'a [u32],
    ) -> Result<Glm5NextAppendTransaction<'a, 'w>> {
        self.validate_request(inputs)?;
        checkpoint.validate_regions(ctx, self)?;
        bank.validate(ctx, self)?;
        bank.validate_all_rows(inputs.len())?;
        if checkpoint.active.is_some() {
            return mismatch("checkpoint already has an active transaction");
        }
        let generation = checkpoint
            .generation
            .checked_add(1)
            .ok_or_else(|| Glm5NextMetalError::Invalid("checkpoint generation exhausted".into()))?;
        let stamp = Stamp {
            owner: Owner::of(ctx, self),
            generation,
            position: self.position,
            end: self.position + inputs.len(),
            lineage: self.packed_lineage(),
            precision: self.effective_fast_precision(),
            pool: self.weights.config.indexer_pool as usize,
        };
        checkpoint.generation = generation;
        checkpoint.copy_regions(ctx, self, false)?;
        checkpoint.active = Some(stamp);
        Ok(Glm5NextAppendTransaction {
            ctx,
            session: self,
            checkpoint,
            bank,
            inputs,
            stamp,
            phase: Phase::Ready,
            #[cfg(test)]
            fail_replay_completion: false,
        })
    }
}

impl Glm5NextAppendTransaction<'_, '_> {
    pub fn base_position(&self) -> usize {
        self.stamp.position
    }

    /// A fatal execution error forbids all further transaction operations. A
    /// successful copy of compact state would not establish historical-cache
    /// integrity. The enclosing request must abort and reset/recreate the target.
    pub fn is_fatal(&self) -> bool {
        self.phase == Phase::Fatal
    }

    fn validate(&self) -> Result<()> {
        if self.is_fatal() {
            return Err(Glm5NextMetalError::Poisoned);
        }
        self.stamp.validate(
            self.checkpoint.active,
            Owner::of(self.ctx, self.session),
            self.session.position,
            self.session.packed_lineage(),
            self.session.effective_fast_precision(),
            self.session.weights.config.indexer_pool as usize,
        )?;
        self.checkpoint.validate_regions(self.ctx, self.session)
    }

    /// Full-row candidate logits using the configured packed lineage/precision,
    /// without claiming bitwise serial equivalence. Use `verify_serial` to replace
    /// these tentative decisions when the orchestration's numerical check fails.
    pub fn verify_packed(&mut self) -> Result<Glm5NextVerificationRows<'_>> {
        self.validate()?;
        if self.phase != Phase::Ready {
            return invalid("packed verify requires a restored baseline");
        }
        if self.inputs.len() > self.session.packed.as_ref().map_or(0, |p| p.rows()) {
            return invalid("packed append needs sufficient packed rows; use serial verify");
        }
        self.run_verify(Glm5NextVerificationArithmetic::Packed {
            lineage: self
                .session
                .packed_lineage()
                .expect("packed rows validated"),
            precision: self.session.effective_fast_precision(),
        })
    }

    /// Serial verification from the same captured baseline, including fallback
    /// after packed mismatch or a completed kernel-status failure. Command-buffer
    /// failures and other execution errors are fatal, never fallback candidates.
    pub fn verify_serial(&mut self) -> Result<Glm5NextVerificationRows<'_>> {
        self.validate()?;
        match self.phase {
            Phase::Ready => {}
            Phase::Verified(Glm5NextVerificationArithmetic::Packed { .. }) | Phase::Failed => {
                self.rollback()?
            }
            _ => return invalid("serial verify requires a baseline or packed/failed append"),
        }
        self.run_verify(Glm5NextVerificationArithmetic::Serial)
    }

    fn run_verify(
        &mut self,
        arithmetic: Glm5NextVerificationArithmetic,
    ) -> Result<Glm5NextVerificationRows<'_>> {
        if self.session.poisoned || self.session.position != self.stamp.position {
            return mismatch("verification baseline is not healthy and restored");
        }
        self.phase = Phase::Running;
        let result = match arithmetic {
            Glm5NextVerificationArithmetic::Packed { .. } => self
                .session
                .verify_packed_with_captures(self.ctx, self.inputs, self.bank),
            Glm5NextVerificationArithmetic::Serial => {
                self.session
                    .scalar_rows_with_captures(self.ctx, self.inputs, self.bank, true)
            }
        };
        match result {
            Ok(rows) => {
                let logits = rows.logits().expect("all-row bank validated").clone();
                self.phase = Phase::Verified(arithmetic);
                // Tentative state is not a usable session, even if the caller
                // forgets the guard instead of running its destructor.
                self.session.poisoned = true;
                Ok(Glm5NextVerificationRows {
                    logits,
                    positions: self.stamp.position..self.stamp.end,
                    arithmetic,
                    _transaction: PhantomData,
                })
            }
            Err(error) => {
                self.phase = Phase::after_execution_error(&error);
                self.session.poisoned = true;
                Err(error)
            }
        }
    }

    /// Restore KDA and pending rings and publish the baseline position only
    /// after successful device copies. Position determines latent visibility,
    /// completed pool count and ring phase. Historical cache rows are untouched;
    /// appended rows remain allocated but invisible. May recover completed
    /// kernel-status failures, not command failures, unwound/in-flight operations
    /// or finished transactions. A failed restore itself is terminal.
    pub fn rollback(&mut self) -> Result<()> {
        if self.is_fatal() {
            return Err(Glm5NextMetalError::Poisoned);
        }
        if !self.phase.can_restore() {
            return mismatch("append transaction is running or finished");
        }
        self.validate()?;
        if self.phase == Phase::Ready
            && !self.session.poisoned
            && self.session.position == self.stamp.position
        {
            return Ok(());
        }
        self.phase = Phase::Running;
        self.session.poisoned = true;
        if let Err(error) = self.checkpoint.copy_regions(self.ctx, self.session, true) {
            self.phase = Phase::Fatal;
            return Err(error);
        }
        self.session.position = self.stamp.position;
        self.session.poisoned = false;
        self.phase = Phase::Ready;
        Ok(())
    }

    /// Retain exactly `retain_input_count` verified inputs, including the anchor
    /// when nonzero. Pass the shared acceptance helper's count unchanged: an
    /// accepted terminal token may deliberately remain unconsumed.
    ///
    /// Full retention keeps verified device state and rows without any replay or
    /// device command. Partial retention restores then scalar-replays only the
    /// retained prefix; zero restores and publishes no rows. A replacement/bonus
    /// is never consumed or resampled here. Resolve numerical fallback BEFORE
    /// acceptance; replay does not retroactively change its arithmetic provenance.
    /// Failed replay publishes nothing. Only a completed kernel-status failure
    /// permits rollback; command failures remain poisoned and terminal.
    pub fn commit_retained_inputs(&mut self, count: usize) -> Result<Glm5NextCommittedRows<'_>> {
        self.validate()?;
        let verification_arithmetic = self.phase.retained_inputs(count, self.inputs.len())?;
        if self.session.position != self.stamp.end {
            return mismatch("verified append did not reach its endpoint");
        }
        let full = count == self.inputs.len();
        if !full {
            self.rollback()?;
        }
        if count == 0 {
            self.phase = Phase::Finished;
            self.checkpoint.active = None;
            return Ok(Glm5NextCommittedRows {
                rows: None,
                verification_arithmetic,
                retained_arithmetic: None,
            });
        }
        self.phase = Phase::Running;
        #[cfg(test)]
        let _replay_failure =
            (!full && self.fail_replay_completion).then(crate::metal::fail_next_completion_check);
        let result = if full {
            self.bank.completed_all_rows(self.stamp.position, count)
        } else {
            self.session.scalar_rows_with_captures(
                self.ctx,
                &self.inputs[..count],
                self.bank,
                false,
            )
        };
        match result {
            Ok(rows) => {
                self.session.poisoned = false;
                self.phase = Phase::Finished;
                self.checkpoint.active = None;
                Ok(Glm5NextCommittedRows {
                    rows: Some(rows),
                    verification_arithmetic,
                    retained_arithmetic: Some(if full {
                        verification_arithmetic
                    } else {
                        Glm5NextVerificationArithmetic::Serial
                    }),
                })
            }
            Err(error) => {
                self.phase = Phase::after_execution_error(&error);
                self.session.poisoned = true;
                Err(error)
            }
        }
    }
}

impl Drop for Glm5NextAppendTransaction<'_, '_> {
    fn drop(&mut self) {
        self.checkpoint.active = None;
        if self.phase.poisons_on_drop() {
            self.session.poisoned = true;
        }
    }
}

#[cfg(test)]
#[path = "tests/checkpoint.rs"]
mod tests;
