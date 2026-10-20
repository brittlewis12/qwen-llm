//! Optional full-vocabulary row readouts. This is not a rollback transaction.

use super::capture::validate_optional_owned_allocations;
use super::*;
use std::ops::Range;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen4ExpRowLogitsPlan {
    capacity: usize,
    vocab: usize,
}

impl Qwen4ExpRowLogitsPlan {
    pub fn new(
        geometry: &Qwen4ExpTextSessionMetalGeometry,
        capacity: usize,
    ) -> Result<Self, Qwen4ExpTextSessionError> {
        let vocab = geometry.vocab_size();
        let elements = capacity.checked_mul(vocab).ok_or_else(|| {
            Qwen4ExpTextSessionError::Invalid("row logits extent overflow".into())
        })?;
        if capacity == 0 || capacity > geometry.capacity() || elements > i32::MAX as usize {
            return invalid(
                "row logits capacity is empty, exceeds the session, or overflows shader indexing",
            );
        }
        Ok(Self { capacity, vocab })
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn vocab_size(&self) -> usize {
        self.vocab
    }
    pub fn logical_bytes(&self) -> u64 {
        (self.capacity * self.vocab) as u64 * 4
    }

    pub(crate) fn validate_rows(
        &self,
        start: usize,
        rows: usize,
        capture_capacity: usize,
    ) -> Result<Range<usize>, Qwen4ExpTextSessionError> {
        if rows == 0 || rows > self.capacity || rows > capture_capacity {
            return invalid(
                "all-row output requires nonempty rows fitting both logits and capture banks",
            );
        }
        let end = start.checked_add(rows).ok_or_else(|| {
            Qwen4ExpTextSessionError::Invalid("row logits position overflow".into())
        })?;
        if end > i32::MAX as usize {
            return invalid("row logits positions exceed i32");
        }
        Ok(start..end)
    }

    fn row_range(&self, row: usize, rows: usize) -> Option<Range<usize>> {
        if rows > self.capacity || row >= rows {
            return None;
        }
        let start = row * self.vocab;
        Some(start..start + self.vocab)
    }
}

/// Caller-owned optional scratch; ordinary prefill allocates none of it.
pub struct Qwen4ExpRowLogitsBank {
    plan: Qwen4ExpRowLogitsPlan,
    device: u64,
    logits: MetalTensor,
    allocation: Qwen4ExpTextSessionAllocation,
}

impl Qwen4ExpRowLogitsBank {
    /// Incremental admission after target residency, before allocation.
    pub fn allocate(
        ctx: &MetalContext,
        plan: Qwen4ExpRowLogitsPlan,
    ) -> Result<Self, Qwen4ExpTextSessionError> {
        let logical_bytes = plan.logical_bytes();
        let (priced_bytes, alignment) = price_session_allocation(
            "readout.row_logits",
            logical_bytes,
            ctx.shared_buffer_size_and_align(logical_bytes)?,
            host_page_size_bytes()? as u64,
            ctx.max_buffer_length() as u64,
        )?;
        ctx.pipeline("kernel_copy_offset_f32")?;
        let _transaction = ctx.begin_allocation_transaction();
        let admission = evaluate_metal_memory_admission(
            priced_bytes,
            QWEN4EXP_TEXT_SESSION_DYNAMIC_RESERVE_BYTES,
            ctx.memory_signals(),
            true,
        );
        if let Some(denied) = admission.refusal() {
            return Err(denied.into());
        }
        let before = ctx.current_allocated_size();
        let logits = MetalTensor::zeros_f32(ctx, vec![plan.vocab as u64, plan.capacity as u64])?;
        let allocation = Qwen4ExpTextSessionAllocation {
            name: "readout.row_logits".into(),
            logical_bytes,
            priced_bytes,
            alignment,
        };
        validate_optional_owned_allocations(
            ctx,
            &[&logits],
            std::slice::from_ref(&allocation),
            before,
        )?;
        Ok(Self {
            plan,
            device: ctx.device.registryID(),
            logits,
            allocation,
        })
    }

    pub fn plan(&self) -> &Qwen4ExpRowLogitsPlan {
        &self.plan
    }
    pub fn allocation(&self) -> &Qwen4ExpTextSessionAllocation {
        &self.allocation
    }

    pub(crate) fn validate(
        &self,
        ctx: &MetalContext,
        vocab: usize,
    ) -> Result<(), Qwen4ExpTextSessionError> {
        if ctx.device.registryID() != self.device || vocab != self.plan.vocab {
            return invalid("row logits bank device or vocabulary mismatch");
        }
        Ok(())
    }

    pub(crate) fn sink(
        &mut self,
        positions: Range<usize>,
    ) -> Result<Qwen4ExpRowLogitsSink<'_>, Qwen4ExpTextSessionError> {
        let positions =
            self.plan
                .validate_rows(positions.start, positions.len(), self.plan.capacity)?;
        Ok(Qwen4ExpRowLogitsSink {
            bank: self,
            positions,
        })
    }
}

pub(crate) struct Qwen4ExpRowLogitsSink<'a> {
    bank: &'a Qwen4ExpRowLogitsBank,
    positions: Range<usize>,
}

impl<'a> Qwen4ExpRowLogitsSink<'a> {
    pub(super) fn encode_row(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        position: usize,
        source: &MetalTensor,
    ) -> Result<(), Qwen4ExpTextSessionError> {
        if !self.positions.contains(&position) {
            return invalid("row logits position is outside this operation");
        }
        let row = position - self.positions.start;
        let range = self
            .bank
            .plan
            .row_range(row, self.positions.len())
            .ok_or_else(|| {
                Qwen4ExpTextSessionError::Invalid("row logits output extent is invalid".into())
            })?;
        let dst = self
            .bank
            .logits
            .view_subrange(range.start as u64, vec![range.len() as u64]);
        require_tensor(
            "row logits source",
            source,
            GgmlType::F32,
            &[range.len() as u64],
            false,
        )?;
        encode_copy_offset_f32(ctx, enc, source, 0, &dst, range.len())?;
        Ok(())
    }

    pub(crate) fn complete(self) -> Qwen4ExpRowLogits<'a> {
        let tensor = self.bank.logits.view_subrange(
            0,
            vec![self.bank.plan.vocab as u64, self.positions.len() as u64],
        );
        Qwen4ExpRowLogits {
            bank: self.bank,
            positions: self.positions,
            tensor,
        }
    }
}

/// Full target distributions, available only after successful command completion.
/// Each row predicts the token following that row's consumed input. CPU slices
/// borrow the bank; finish GPU reads before dropping this view and reusing it.
pub struct Qwen4ExpRowLogits<'a> {
    bank: &'a Qwen4ExpRowLogitsBank,
    positions: Range<usize>,
    tensor: MetalTensor,
}

impl Qwen4ExpRowLogits<'_> {
    pub fn tensor(&self) -> &MetalTensor {
        &self.tensor
    }
    pub fn positions(&self) -> Range<usize> {
        self.positions.clone()
    }
    pub fn vocab_size(&self) -> usize {
        self.bank.plan.vocab
    }

    /// Completed, contiguous `[position][vocabulary]` distributions.
    pub fn as_slice(&self) -> &[f32] {
        // SAFETY: completion and the bank borrow provide the same guarantees as row().
        unsafe {
            std::slice::from_raw_parts(
                self.tensor
                    .buffer
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .add(self.tensor.offset as usize)
                    .cast::<f32>(),
                self.positions.len() * self.bank.plan.vocab,
            )
        }
    }

    pub fn row(&self, row: usize) -> Option<&[f32]> {
        let range = self.bank.plan.row_range(row, self.positions.len())?;
        // SAFETY: private construction follows successful synchronous execution.
        // The bank borrow excludes reuse, and the checked row lies inside F32 storage.
        Some(unsafe {
            std::slice::from_raw_parts(
                self.tensor
                    .buffer
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .add(self.tensor.offset as usize)
                    .cast::<f32>()
                    .add(range.start),
                range.len(),
            )
        })
    }
}

#[cfg(test)]
impl Qwen4ExpTextSessionMetalWorkspace {
    /// Copy all final hyper rows from the just-completed native command. The
    /// singleton path uses different storage; packed rows must be read before
    /// another packed forward overwrites them. No capture/readout sink is used.
    pub(crate) fn completed_final_hyper_rows_for_tests(&self, command_rows: usize) -> Vec<f32> {
        self.require_idle().unwrap();
        self.logits().unwrap();
        assert!(command_rows > 0 && command_rows <= self.committed_length);
        let width = self.geometry.hyper_width();
        let tensor = if command_rows == 1 {
            self.hyper_residual.view_subrange(0, vec![width as u64])
        } else {
            let packed = self.packed_scratch().unwrap();
            assert!(command_rows <= packed.capacity);
            packed
                .hyper_residual
                .view_subrange(0, vec![width as u64, command_rows as u64])
        };
        // SAFETY: completed, idle workspace; the checked rows lie in F32 storage.
        unsafe {
            std::slice::from_raw_parts(
                tensor
                    .buffer
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .add(tensor.offset as usize)
                    .cast::<f32>(),
                width * command_rows,
            )
            .to_vec()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(capacity: usize) -> Result<Qwen4ExpRowLogitsPlan, Qwen4ExpTextSessionError> {
        let geometry = Qwen4ExpTextSessionMetalGeometry::from_config(
            &Qwen4ExpConfig::flash_next_reference(),
            8192,
        )?;
        Qwen4ExpRowLogitsPlan::new(&geometry, capacity)
    }

    #[test]
    fn row_logits_cpu_layout_has_full_vocab_per_row() {
        let p = plan(8).unwrap();
        assert_eq!(p.vocab_size(), 248320);
        assert_eq!(p.logical_bytes(), 8 * 248320 * 4);
        for row in 0..8 {
            assert_eq!(p.row_range(row, 8), Some(row * 248320..(row + 1) * 248320));
        }
        assert!(p.row_range(8, 8).is_none());
        assert!(p.row_range(0, 0).is_none());
        assert!(p.row_range(0, 9).is_none());
        assert!(p.row_range(2, 2).is_none());
    }

    #[test]
    fn row_logits_cpu_layout_refuses_truncation_and_invalid_requests() {
        let p = plan(8).unwrap();
        assert_eq!(p.validate_rows(3, 4, 4).unwrap(), 3..7);
        for (start, rows, captures) in [
            (0, 0, 8),
            (0, 9, 9),
            (0, 4, 3),
            (usize::MAX, 1, 8),
            (i32::MAX as usize, 1, 8),
        ] {
            assert!(p.validate_rows(start, rows, captures).is_err());
        }
        for capacity in [0, 8193, usize::MAX] {
            assert!(plan(capacity).is_err());
        }
    }
}
