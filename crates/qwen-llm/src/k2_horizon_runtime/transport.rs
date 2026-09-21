//! Raw linear algebra only. External readers own digest verification, checkpoint
//! and source-site binding, target-layer semantics, and transfer authorization.

use super::lens::WIDTH;
use super::*;

const MATRIX_BYTES: usize = WIDTH * WIDTH * 2;

/// One finite 4096x4096 F16 matrix, little-endian row-major `[target, source]`.
/// No bias, normalization, implicit transpose, fitted-asset identity, or trust
/// claim. Ownership prevents coefficients changing after validation in safe Rust.
pub struct K2LinearF16 {
    bytes: Vec<u8>,
}

impl K2LinearF16 {
    pub fn from_target_source_le(bytes: Vec<u8>) -> Result<Self> {
        if bytes.len() != MATRIX_BYTES {
            return Err(invalid(
                "linear readout requires exactly 4096x4096 little-endian F16 coefficients",
            ));
        }
        if bytes
            .chunks_exact(2)
            .any(|pair| u16::from_le_bytes([pair[0], pair[1]]) & 0x7c00 == 0x7c00)
        {
            return Err(invalid("nonfinite linear F16 coefficient"));
        }
        Ok(Self { bytes })
    }

    pub fn byte_len(&self) -> usize {
        self.bytes.len()
    }

    pub(super) fn upload(&self, ctx: &MetalContext) -> Result<MetalTensor> {
        // Session scratch/output are already resident and admitted. Only this
        // per-call matrix is additional; keep the same reserve/current-use gate.
        let price = price_buffers(ctx, &[MATRIX_BYTES as u64])?;
        let _transaction = ctx.begin_allocation_transaction();
        admit(ctx, price)?;
        let before = ctx.current_allocated_size();
        let shape = vec![WIDTH as u64, WIDTH as u64];
        // This is a private upload destination, not a borrowed caller buffer.
        let tensor = MetalTensor::from_bytes(ctx, &self.bytes, shape.clone(), GgmlType::F16)?;
        if tensor.dtype != GgmlType::F16
            || tensor.shape != shape
            || tensor.n_bytes() != MATRIX_BYTES as u64
            || !tensor.is_writable()
        {
            return Err(invalid("linear matrix allocation descriptor drift"));
        }
        validate_cpu_layout(
            tensor.buffer.storageMode() == MTLStorageMode::Shared,
            tensor.offset,
            tensor.n_bytes(),
            tensor.buffer.length() as u64,
        )?;
        reconcile(ctx, before, price)?;
        Ok(tensor)
    }
}

/// One raw transported residual before final grouped norm, and its deployed logits.
pub struct K2LinearReadout {
    pub residual: Vec<f32>,
    pub logits: Vec<f32>,
}

impl K2Session<'_, '_> {
    /// `y[target] = sum_source M[target, source] * x[source]`, then the same final
    /// grouped norm/untied head as ordinary readout. F16 coefficients, F32 math.
    /// No KV access/prefix advance. A post-submit failure poisons the session.
    /// The owned per-call GPU matrix copy is admitted and lives through completion;
    /// ordinary readout does not allocate one. Asset policy is deliberately external.
    pub fn readout_linear_f16(
        &mut self,
        matrix: &K2LinearF16,
        residual: &[f32],
    ) -> Result<K2LinearReadout> {
        self.readout_impl(residual, Some(matrix))
    }
}

#[cfg(test)]
mod tests;
