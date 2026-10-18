//! Encode-only F32 Engram injection; constants are uploaded at bind time.
use super::*;
use objc2_metal::MTLSize;

#[cfg(test)]
mod tests;

pub(super) struct BoundGate {
    q: MetalTensor,
    k: MetalTensor,
    hidden: usize,
    eps: f32,
}

impl BoundGate {
    pub(super) fn new(ctx: &MetalContext, gate: &EngramGate, c: &Config) -> Result<Self> {
        op::require(
            c.hidden > 0
                && c.hidden <= u32::MAX as usize / 5
                && c.norm_eps.is_finite()
                && c.norm_eps > 0.0,
            "GPU Engram geometry/epsilon",
        )?;
        let n = 4 * c.hidden;
        let (q, k) = match gate {
            EngramGate::Separate { q, k } => (q.as_slice(), k.clone()),
            EngramGate::Folded(product) => (product.as_slice(), vec![1.0; n]),
        };
        op::require(q.len() == n && k.len() == n, "GPU Engram constants shape")?;
        Ok(Self {
            q: upload(ctx, q, vec![n as u64])?,
            k: upload(ctx, &k, vec![n as u64])?,
            hidden: c.hidden,
            eps: c.norm_eps,
        })
    }

    /// Active prefixes are row-major [rows, 4, hidden] and [rows, 5, hidden].
    /// Projected contains four keys followed by one shared value per row.
    /// Buffers must not overlap. No allocation, submission, or readback occurs.
    pub(super) fn encode(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        x: &MetalTensor,
        projected: &MetalTensor,
        rows: usize,
    ) -> Result<()> {
        let extent = rows
            .checked_mul(5 * self.hidden)
            .filter(|&n| n <= u32::MAX as usize)
            .ok_or("GPU Engram row extent overflow")?;
        let capacity = |t: &MetalTensor| t.shape.iter().try_fold(1u64, |n, &d| n.checked_mul(d));
        op::require(
            x.dtype == GgmlType::F32
                && projected.dtype == GgmlType::F32
                && capacity(x).is_some_and(|n| n >= (rows * 4 * self.hidden) as u64)
                && capacity(projected).is_some_and(|n| n >= extent as u64),
            "GPU Engram F32 activation shape",
        )?;
        if rows == 0 {
            return Ok(());
        }
        enc.set_pipeline(
            &ctx.pipeline("kernel_deepseek_v41_engram")
                .map_err(|e| e.to_string())?,
        );
        enc.set_bytes(0, &(self.hidden as u32));
        enc.set_bytes(1, &self.eps);
        for (slot, tensor) in [x, projected, &self.q, &self.k].iter().enumerate() {
            enc.set_tensor(slot + 2, tensor);
        }
        enc.dispatch(
            MTLSize {
                width: rows * 4,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }
}
