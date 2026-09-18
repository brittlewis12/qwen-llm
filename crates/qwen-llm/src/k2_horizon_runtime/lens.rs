//! Forward-only hooks. Captures are observations, not reusable KV snapshots or
//! checkpoint-bound imported assets. Readouts deliberately carry no fit/backward API.

use super::*;
use crate::metal::encode_copy_offset_f32;

const WIDTH: usize = 4096;
const LAYERS: u32 = 36;

/// Final appended token, at selected zero-based post-block sites: after the full
/// FFN residual addition, before any following normalization. Rows are flattened
/// `[site][4096]` F32, in `post_block_layers` order. These are raw residuals, not
/// normalized states. `logits` is the ordinary final grouped-norm/untied-head output.
#[derive(Debug)]
pub struct K2CapturedForward {
    pub absolute_position: u32,
    pub post_block_layers: Vec<u32>,
    pub residuals: Vec<f32>,
    pub logits: Vec<f32>,
}

impl K2Session<'_, '_> {
    /// Capture only the last position of this append. Sites must be nonempty,
    /// sorted, unique, and in 0..36. No capture buffer is retained after the call.
    pub fn append_with_captures(
        &mut self,
        tokens: &[u32],
        post_block_layers: &[u32],
    ) -> Result<K2CapturedForward> {
        capture_bytes(post_block_layers)?;
        self.append_impl(tokens, post_block_layers)
    }

    /// Apply this model's final grouped norm and untied head to one arbitrary
    /// finite F32 residual. Does not read/write KV or advance the committed prefix.
    /// A submitted failure poisons the session; host rejection is retryable.
    pub fn readout(&mut self, residual: &[f32]) -> Result<Vec<f32>> {
        validate_residual(residual)?;
        self.model.plan.revalidate_source()?;
        let mut transaction = self.ledger.begin_readout()?;
        // Owned, checked shared F32[4096]; prior commands completed synchronously.
        unsafe {
            std::ptr::copy_nonoverlapping(
                residual.as_ptr(),
                self.buffers
                    .residual
                    .buffer
                    .contents()
                    .as_ptr()
                    .cast::<f32>(),
                WIDTH,
            );
        }
        let ctx = self.model.ctx;
        let command = ctx
            .queue
            .commandBuffer()
            .ok_or_else(|| invalid("cannot create readout command"))?;
        let encoder = KernelEncoder::begin(&command);
        let encoded = encode_readout(
            ctx,
            &encoder,
            &self.model.weights,
            &self.request,
            &self.buffers,
        );
        encoder.end();
        encoded?;
        transaction.submitting()?;
        command.commit();
        command.waitUntilCompleted();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(invalid(format!(
                "readout command failed: {:?}",
                command.error()
            )));
        }
        let logits = read_f32(&self.buffers.logits);
        if logits.iter().any(|value| !value.is_finite()) {
            return Err(invalid("nonfinite readout logits"));
        }
        self.model.plan.revalidate_source()?;
        let logits = logits.to_vec();
        transaction.checked()?;
        transaction.commit()?;
        Ok(logits)
    }
}

fn validate_residual(residual: &[f32]) -> Result<()> {
    if residual.len() != WIDTH || residual.iter().any(|value| !value.is_finite()) {
        return Err(invalid("readout requires exactly 4096 finite F32 values"));
    }
    Ok(())
}

fn capture_bytes(layers: &[u32]) -> Result<u64> {
    if layers.is_empty()
        || layers.len() > LAYERS as usize
        || layers.iter().any(|&layer| layer >= LAYERS)
        || layers.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(invalid(
            "capture sites must be nonempty sorted unique layers in 0..36",
        ));
    }
    Ok(layers.len() as u64 * WIDTH as u64 * 4)
}

pub(super) struct CaptureArena {
    layers: Vec<u32>,
    tensor: MetalTensor,
}

impl CaptureArena {
    pub(super) fn allocate(ctx: &MetalContext, layers: &[u32]) -> Result<Option<Self>> {
        if layers.is_empty() {
            return Ok(None);
        }
        let bytes = capture_bytes(layers)?;
        let price = price_buffers(ctx, &[bytes])?;
        let _transaction = ctx.begin_allocation_transaction();
        admit(ctx, price)?;
        let before = ctx.current_allocated_size();
        let shape = vec![WIDTH as u64, layers.len() as u64];
        let tensor = MetalTensor::zeros_dtype(ctx, shape.clone(), GgmlType::F32)?;
        if tensor.dtype != GgmlType::F32
            || tensor.shape != shape
            || tensor.n_bytes() != bytes
            || !tensor.is_writable()
        {
            return Err(invalid("capture allocation descriptor drift"));
        }
        validate_cpu_layout(
            tensor.buffer.storageMode() == MTLStorageMode::Shared,
            tensor.offset,
            bytes,
            tensor.buffer.length() as u64,
        )?;
        reconcile(ctx, before, price)?;
        Ok(Some(Self {
            layers: layers.to_vec(),
            tensor,
        }))
    }

    pub(super) fn encode(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        layer: u32,
        residual: &MetalTensor,
    ) -> Result<()> {
        if let Ok(index) = self.layers.binary_search(&layer) {
            let row = self
                .tensor
                .view_subrange((index * WIDTH) as u64, vec![WIDTH as u64]);
            encode_copy_offset_f32(ctx, enc, residual, 0, &row, WIDTH)?;
        }
        Ok(())
    }

    pub(super) fn read(&self) -> Result<Vec<f32>> {
        let rows = read_f32(&self.tensor);
        if rows.iter().any(|value| !value.is_finite()) {
            return Err(invalid("nonfinite captured residual"));
        }
        Ok(rows.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_sites_have_bounded_exact_storage_and_explicit_order() {
        assert_eq!(capture_bytes(&[35]).unwrap(), 16384);
        assert_eq!(capture_bytes(&(0..36).collect::<Vec<_>>()).unwrap(), 589824);
        for sites in [
            vec![],
            vec![0, 0],
            vec![35, 1],
            vec![36],
            vec![u32::MAX],
            vec![0; 37],
        ] {
            assert!(capture_bytes(&sites).is_err());
        }
    }

    #[test]
    fn readout_accepts_only_one_finite_hidden_row() {
        assert!(validate_residual(&vec![0.0; WIDTH]).is_ok());
        for length in [0, WIDTH - 1, WIDTH + 1] {
            assert!(validate_residual(&vec![0.0; length]).is_err());
        }
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut row = vec![0.0; WIDTH];
            row[WIDTH - 1] = value;
            assert!(validate_residual(&row).is_err());
        }
    }
}
