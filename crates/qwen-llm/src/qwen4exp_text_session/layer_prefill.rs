//! Row-preserving packed command checkpoints; no command survives a yield.
use super::*;
use objc2_metal::MTLCommandQueue;

struct Submitted {
    command: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    submitted: bool,
}
impl Drop for Submitted {
    fn drop(&mut self) {
        if self.submitted {
            crate::metal::wait_unchecked(&self.command);
        }
    }
}

pub(super) struct Progress {
    ids: Vec<u32>,
    start: usize,
    next_layer: usize,
    history: Option<PleHistory>,
}

impl Qwen4ExpTextSessionMetalWorkspace {
    pub(crate) fn layer_prefill_active(&self) -> bool {
        self.layer_prefill.is_some()
    }

    pub(crate) fn cancel_layer_prefill(&mut self) {
        if self.layer_prefill.take().is_some() || self.pending_length.is_some() {
            self.state_poisoned = true;
            self.encode_failed = true;
            self.logits_ready = false;
        }
    }

    pub(crate) fn packed_layer_slice(
        &mut self,
        ctx: &MetalContext,
        ids: &[u32],
        table: PleIq4NlTable<'_>,
        weights: &Qwen4ExpTextSessionMetalWeights<'_>,
        end_layer: usize,
    ) -> Result<(bool, f64), Qwen4ExpTextSessionError> {
        let layers = weights.post_ple.len() + 2;
        let next = self.layer_prefill.as_ref().map_or(0, |p| p.next_layer);
        if ids.len() < 2 || end_layer < 2 || end_layer > layers || end_layer <= next {
            return invalid("invalid packed layer boundary or scalar input");
        }
        if let Some(p) = &self.layer_prefill {
            if p.ids != ids || self.state_poisoned || self.committed_length != p.start {
                return invalid("packed layer input/state changed");
            }
        }
        let command = ctx
            .queue
            .commandBuffer()
            .ok_or_else(|| Qwen4ExpTextSessionError::Invalid("Metal command allocation".into()))?;
        let enc = KernelEncoder::begin(&command);
        let result = (|| -> Result<(bool, f64), Qwen4ExpTextSessionError> {
            let mut p = if let Some(p) = self.layer_prefill.take() {
                // Earlier slices have completed; keep the whole command's
                // pending length and checkpoint lineage until final validation.
                if self
                    .active_command
                    .as_ref()
                    .is_none_or(|c| c.status() != MTLCommandBufferStatus::Completed)
                {
                    return invalid("previous packed slice not complete");
                }
                self.active_command = Some(command.clone());
                p
            } else {
                let start = self.committed_length;
                let (history, rows) =
                    validate_and_preflight_packed(ctx, &enc, ids, start, table, weights, self)?;
                stage_packed_inputs(ids, &rows, table, self)?;
                reserve_command(self, &enc, start + ids.len())?;
                self.logits_ready = false;
                Progress {
                    ids: ids.to_vec(),
                    start,
                    next_layer: 0,
                    history: Some(history),
                }
            };
            unsafe {
                encode_packed_range(
                    ctx,
                    &enc,
                    p.start,
                    p.history.take(),
                    weights,
                    self,
                    ids.len(),
                    None,
                    None,
                    None,
                    p.next_layer..end_layer,
                )?;
            }
            p.next_layer = end_layer;
            self.layer_prefill = Some(p);
            Ok((end_layer == layers, 0.))
        })();
        enc.end();
        let (complete, _) = match result {
            Ok(v) => v,
            Err(e) => {
                self.state_poisoned = true;
                self.encode_failed = true;
                self.layer_prefill = None;
                return Err(e);
            }
        };
        let mut terminal = Submitted {
            command: command.clone(),
            submitted: false,
        };
        command.commit();
        terminal.submitted = true;
        if let Err(e) = crate::metal::wait_completed(&command) {
            self.cancel_layer_prefill();
            return Err(e.into());
        }
        let gpu = (command.GPUEndTime() - command.GPUStartTime()) * 1000.;
        if complete {
            self.layer_prefill = None;
            self.release_after()?;
        }
        Ok((complete, gpu))
    }
}
