//! Block-entry device checkpoints. Only bounded overwrite-in-place causal state
//! is copied; append-only QSA storage is recovered by restoring visibility.

use super::*;
use crate::metal::BlitEncoder;
use crate::qwen4exp_post_ple_block::Qwen4ExpCompactMixerState;
use objc2_metal::MTLCommandQueue;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct CheckpointLineage {
    session: u64,
    branch: u64,
    append_epoch: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Stamp {
    lineage: CheckpointLineage,
    length: usize,
}

impl CheckpointLineage {
    pub(super) fn fresh() -> Result<Self, Qwen4ExpTextSessionError> {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let session = NEXT
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| {
                Qwen4ExpTextSessionError::Invalid("checkpoint session identity exhausted".into())
            })?;
        Ok(Self {
            session,
            branch: 0,
            append_epoch: 0,
        })
    }

    pub(super) fn record_append(&mut self) -> Result<(), Qwen4ExpTextSessionError> {
        self.append_epoch = self.append_epoch.checked_add(1).ok_or_else(|| {
            Qwen4ExpTextSessionError::Invalid("checkpoint append epoch exhausted".into())
        })?;
        Ok(())
    }

    pub(super) fn next_branch(self) -> Result<Self, Qwen4ExpTextSessionError> {
        let branch = self.branch.checked_add(1).ok_or_else(|| {
            Qwen4ExpTextSessionError::Invalid("checkpoint branch epoch exhausted".into())
        })?;
        Ok(Self {
            branch,
            append_epoch: 0,
            ..self
        })
    }

    fn validate_restore(self, saved: Stamp, length: usize) -> Result<(), Qwen4ExpTextSessionError> {
        if self.session != saved.lineage.session
            || self.branch != saved.lineage.branch
            || saved.lineage.append_epoch > self.append_epoch
            || saved.length > length
        {
            return invalid(
                "device checkpoint belongs to another session, stale branch, or future append",
            );
        }
        Ok(())
    }
}

struct Entry {
    stamp: Stamp,
    history: PleHistory,
}

/// One separately admitted reusable device arena, bound to one workspace.
/// Capture replaces its saved entry. Successful restore consumes the entry and
/// invalidates every other checkpoint from that branch. No logits are saved.
pub struct Qwen4ExpDeviceCheckpointBank {
    session: u64,
    device: u64,
    arena: MetalTensor,
    region_elements: Vec<u64>,
    allocation: Qwen4ExpTextSessionAllocation,
    entry: Option<Entry>,
}

impl Qwen4ExpDeviceCheckpointBank {
    pub fn allocation(&self) -> &Qwen4ExpTextSessionAllocation {
        &self.allocation
    }

    /// Saved position, not a validity promise after target reset or rollback.
    pub fn captured_position(&self) -> Option<usize> {
        self.entry.as_ref().map(|e| e.stamp.length)
    }
}

impl Qwen4ExpTextSessionMetalGeometry {
    /// Fixed payload, independent of committed length and KV capacity.
    pub fn device_checkpoint_bytes(&self) -> u64 {
        let mut bytes = 2 * self.zero_one.layer_zero().gdn().snapshot_bytes()
            + self.zero_one.ple().conv_state_bytes();
        for block in &self.post_ple {
            bytes += match block.mixer() {
                Qwen4ExpPostPleMixerMetalGeometry::GatedDeltaNet(gdn) => gdn.snapshot_bytes(),
                Qwen4ExpPostPleMixerMetalGeometry::QwenSparseAttention(qsa) => {
                    qsa.checkpoint_pending_bytes()
                }
            };
        }
        bytes as u64
    }
}

impl Qwen4ExpTextSessionMetalWorkspace {
    /// Test oracle only: include visible append-only prefixes, never dirty suffixes.
    #[cfg(test)]
    pub(crate) fn checkpoint_visible_regions_for_tests(
        &self,
    ) -> Result<Vec<MetalTensor>, Qwen4ExpTextSessionError> {
        self.require_device_checkpoint_ready()?;
        let mut regions = self.zero_one.persistent_state_tensors();
        for block in &self.post_ple {
            regions.extend(block.snapshot_regions(self.committed_length)?);
        }
        Ok(regions)
    }

    fn require_device_checkpoint_ready(&self) -> Result<(), Qwen4ExpTextSessionError> {
        self.require_idle()?;
        if self.state_poisoned || self.encode_failed || self.pending_length.is_some() {
            return invalid(
                "device checkpoint requires healthy completed target state; GPU failures cannot be recovered",
            );
        }
        let expected_history = (self.committed_length > 0).then_some(self.committed_length as u64);
        if self.zero_one.next_position() != expected_history
            || self
                .qsa_committed_lengths()
                .iter()
                .any(|(_, n)| *n != self.committed_length)
        {
            return invalid("device checkpoint causal lengths disagree");
        }
        Ok(())
    }

    fn compact_checkpoint_regions(
        &self,
        ctx: &MetalContext,
    ) -> Result<Vec<MetalTensor>, Qwen4ExpTextSessionError> {
        self.require_device_checkpoint_ready()?;
        let mut regions = self.zero_one.compact_checkpoint_state()?;
        for block in &self.post_ple {
            match block.compact_checkpoint_state()? {
                Qwen4ExpCompactMixerState::GatedDeltaNet { recurrent } => regions.extend(recurrent),
                Qwen4ExpCompactMixerState::QwenSparseAttention { pending_index_keys } => {
                    regions.push(pending_index_keys)
                }
            }
        }
        let mut bytes = 0u64;
        for tensor in &regions {
            require_tensor(
                "checkpoint state",
                tensor,
                GgmlType::F32,
                &tensor.shape,
                true,
            )?;
            bytes = bytes.checked_add(tensor.n_bytes()).ok_or_else(|| {
                Qwen4ExpTextSessionError::Invalid("checkpoint region bytes overflow".into())
            })?;
        }
        if bytes != self.geometry.device_checkpoint_bytes() {
            return invalid("compact checkpoint region inventory differs from geometry");
        }
        let named: Vec<_> = regions.iter().map(|t| ("checkpoint state", t)).collect();
        require_same_device(ctx, &named)?;
        require_disjoint(&named)?;
        Ok(regions)
    }

    pub(crate) fn allocate_device_checkpoint_bank(
        &self,
        ctx: &MetalContext,
    ) -> Result<Qwen4ExpDeviceCheckpointBank, Qwen4ExpTextSessionError> {
        let regions = self.compact_checkpoint_regions(ctx)?;
        let logical_bytes = self.geometry.device_checkpoint_bytes();
        let (priced_bytes, alignment) = price_session_allocation(
            "checkpoint.causal_state",
            logical_bytes,
            ctx.shared_buffer_size_and_align(logical_bytes)?,
            host_page_size_bytes()? as u64,
            ctx.max_buffer_length() as u64,
        )?;
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
        let arena = MetalTensor::zeros_f32(ctx, vec![logical_bytes / 4])?;
        if ctx.current_allocated_size().saturating_sub(before) > priced_bytes {
            return invalid("checkpoint allocation exceeds priced upper bound");
        }
        Ok(Qwen4ExpDeviceCheckpointBank {
            session: self.checkpoint_lineage.session,
            device: ctx.device.registryID(),
            arena,
            region_elements: regions.iter().map(MetalTensor::n_elements).collect(),
            allocation: Qwen4ExpTextSessionAllocation {
                name: "checkpoint.causal_state".into(),
                logical_bytes,
                priced_bytes,
                alignment,
            },
            entry: None,
        })
    }

    fn validate_device_checkpoint_bank(
        &self,
        ctx: &MetalContext,
        bank: &Qwen4ExpDeviceCheckpointBank,
    ) -> Result<Vec<MetalTensor>, Qwen4ExpTextSessionError> {
        if bank.session != self.checkpoint_lineage.session || bank.device != ctx.device.registryID()
        {
            return invalid("checkpoint bank belongs to a different workspace or device");
        }
        let regions = self.compact_checkpoint_regions(ctx)?;
        if regions.len() != bank.region_elements.len()
            || regions
                .iter()
                .zip(&bank.region_elements)
                .any(|(t, &n)| t.n_elements() != n)
        {
            return invalid("checkpoint bank region layout differs from workspace");
        }
        require_tensor(
            "checkpoint arena",
            &bank.arena,
            GgmlType::F32,
            &[self.geometry.device_checkpoint_bytes() / 4],
            true,
        )?;
        require_same_device(ctx, &[("checkpoint arena", &bank.arena)])?;
        for region in &regions {
            require_disjoint(&[
                ("checkpoint state", region),
                ("checkpoint arena", &bank.arena),
            ])?;
        }
        Ok(regions)
    }

    pub(crate) fn capture_device_checkpoint(
        &mut self,
        ctx: &MetalContext,
        bank: &mut Qwen4ExpDeviceCheckpointBank,
    ) -> Result<(), Qwen4ExpTextSessionError> {
        let regions = self.validate_device_checkpoint_bank(ctx, bank)?;
        let entry = Entry {
            stamp: Stamp {
                lineage: self.checkpoint_lineage,
                length: self.committed_length,
            },
            history: self.zero_one.history().clone(),
        };
        let command = ctx.queue.commandBuffer().ok_or_else(|| {
            Qwen4ExpTextSessionError::Invalid("no checkpoint command buffer".into())
        })?;
        let blit = BlitEncoder::try_begin(&command)?;
        encode_state_copy(&blit, &regions, bank, false);
        blit.end();
        bank.entry = None;
        command.commit();
        if let Err(error) = crate::metal::wait_completed(&command) {
            self.state_poisoned = true;
            self.logits_ready = false;
            return Err(error.into());
        }
        bank.entry = Some(entry);
        Ok(())
    }

    pub(crate) fn restore_device_checkpoint(
        &mut self,
        ctx: &MetalContext,
        bank: &mut Qwen4ExpDeviceCheckpointBank,
    ) -> Result<usize, Qwen4ExpTextSessionError> {
        // Validate all lineage, state, lengths and storage before encoding writes.
        self.require_device_checkpoint_ready()?;
        let entry = bank.entry.as_ref().ok_or_else(|| {
            Qwen4ExpTextSessionError::Invalid("checkpoint bank has no saved entry".into())
        })?;
        self.checkpoint_lineage
            .validate_restore(entry.stamp, self.committed_length)?;
        let regions = self.validate_device_checkpoint_bank(ctx, bank)?;
        let length = entry.stamp.length;
        self.zero_one.validate_checkpoint_history(&entry.history)?;
        for block in &self.post_ple {
            block.validate_checkpoint_rewind(length)?;
        }
        let history = entry.history.clone();
        let next_lineage = self.checkpoint_lineage.next_branch()?;
        let command = ctx.queue.commandBuffer().ok_or_else(|| {
            Qwen4ExpTextSessionError::Invalid("no checkpoint restore command buffer".into())
        })?;
        let blit = BlitEncoder::try_begin(&command)?;
        encode_state_copy(&blit, &regions, bank, true);
        blit.end();

        bank.entry = None;
        self.checkpoint_lineage = next_lineage;
        self.state_poisoned = true;
        self.logits_ready = false;
        command.commit();
        crate::metal::wait_completed(&command)?;
        self.zero_one.rewind_checkpoint_history(history)?;
        for block in &mut self.post_ple {
            block.rewind_checkpoint_length(length)?;
        }
        self.committed_length = length;
        self.state_poisoned = false;
        Ok(length)
    }
}

fn encode_state_copy(
    blit: &BlitEncoder,
    regions: &[MetalTensor],
    bank: &Qwen4ExpDeviceCheckpointBank,
    restore: bool,
) {
    let mut offset = 0;
    for region in regions {
        let n = region.n_elements();
        let slot = bank.arena.view_subrange(offset, vec![n]);
        if restore {
            blit.copy_tensor(&slot, region);
        } else {
            blit.copy_tensor(region, &slot);
        }
        offset += n;
    }
}

#[cfg(test)]
mod tests;
