//! Completed-layer stream means for a bounded, token-major feature window.

use super::*;
use crate::metal::{
    encode_mhc4_collapse, encode_mhc4_collapse_rows, encode_scatter_rows_f32_unique,
};
use objc2_metal::MTLResource;
use std::ops::Range;

/// CPU-only capture layout. Layer IDs name completed zero-based blocks, in
/// feature order (not execution order). Layer zero is inside the fused PLE stage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen4ExpCapturePlan {
    layers: Vec<u32>,
    hidden: usize,
    layer_count: usize,
    capacity: usize,
    features: usize,
}

impl Qwen4ExpCapturePlan {
    pub fn new(
        geometry: &Qwen4ExpTextSessionMetalGeometry,
        layers: Vec<u32>,
        capacity: usize,
    ) -> Result<Self, Qwen4ExpTextSessionError> {
        if geometry.branch_count() != 4 || capacity == 0 || layers.is_empty() {
            return invalid("capture requires four streams, nonempty layers and nonzero capacity");
        }
        for (tap, &layer) in layers.iter().enumerate() {
            if layer == 0 || layer as usize >= geometry.layer_count() {
                return invalid(format!(
                    "capture layer {layer} is outside 1..{}",
                    geometry.layer_count()
                ));
            }
            if layers[..tap].contains(&layer) {
                return invalid(format!("duplicate capture layer {layer}"));
            }
        }
        let hidden = geometry.hidden_size();
        let features = hidden.checked_mul(layers.len()).ok_or_else(|| {
            Qwen4ExpTextSessionError::Invalid("capture feature width overflow".into())
        })?;
        let elements = features.checked_mul(capacity).ok_or_else(|| {
            Qwen4ExpTextSessionError::Invalid("capture bank extent overflow".into())
        })?;
        if elements > i32::MAX as usize || capacity > i32::MAX as usize / 4 {
            return invalid("capture bank exceeds signed shader indexing");
        }
        Ok(Self {
            layers,
            hidden,
            layer_count: geometry.layer_count(),
            capacity,
            features,
        })
    }

    pub fn layers(&self) -> &[u32] {
        &self.layers
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn feature_width(&self) -> usize {
        self.features
    }

    /// Tail of this operation only; a restored prefix is not implicitly captured.
    pub fn tail_range(
        &self,
        start: usize,
        rows: usize,
    ) -> Result<Range<usize>, Qwen4ExpTextSessionError> {
        let end = start
            .checked_add(rows)
            .ok_or_else(|| Qwen4ExpTextSessionError::Invalid("capture position overflow".into()))?;
        if rows == 0 || end > i32::MAX as usize {
            return invalid("capture requires nonempty positions within i32");
        }
        Ok(end - rows.min(self.capacity)..end)
    }

    fn allocations(&self) -> [(&'static str, u64); 4] {
        [
            (
                "capture.features",
                (self.capacity * self.features) as u64 * 4,
            ),
            ("capture.mean", (self.capacity * self.hidden) as u64 * 4),
            ("capture.quarters", self.capacity as u64 * 16),
            (
                "capture.indices",
                (self.capacity * self.layers.len()) as u64 * 4,
            ),
        ]
    }

    pub fn logical_bytes(&self) -> u64 {
        self.allocations().iter().map(|(_, bytes)| bytes).sum()
    }

    fn scatter_row(&self, tap: usize, row: usize) -> usize {
        row * self.layers.len() + tap
    }
}

/// Reusable caller-owned GPU storage, separate from target causal state. No
/// feature bytes are exposed until a synchronous capture operation succeeds.
pub struct Qwen4ExpCaptureBank {
    plan: Qwen4ExpCapturePlan,
    device: u64,
    features: MetalTensor,
    mean: MetalTensor,
    quarters: MetalTensor,
    indices: MetalTensor,
    allocations: Vec<Qwen4ExpTextSessionAllocation>,
    priced_bytes: u64,
}

pub(super) fn validate_optional_owned_allocations(
    ctx: &MetalContext,
    buffers: &[&MetalTensor],
    allocations: &[Qwen4ExpTextSessionAllocation],
    device_before: u64,
) -> Result<(), Qwen4ExpTextSessionError> {
    if buffers.len() != allocations.len() {
        return invalid("optional buffer allocation inventory length mismatch");
    }
    let device_after = ctx.current_allocated_size();
    // currentAllocatedSize includes allocator pools and driver resources. Small
    // shared buffers can consume an existing pool (zero delta) or grow it by a
    // slab larger than this bank. Validate ownership with each resource's size.
    for (tensor, allocation) in buffers.iter().zip(allocations) {
        let length = tensor.buffer.length() as u64;
        let owned = tensor.buffer.allocatedSize() as u64;
        if tensor.offset != 0
            || tensor.n_bytes() != allocation.logical_bytes
            || length != allocation.logical_bytes
            || length > allocation.priced_bytes
            || owned > allocation.priced_bytes
        {
            return invalid(format!(
                "optional buffer allocation violates priced inventory: buffer={} logical_bytes={} buffer_length={length} owned_allocated_bytes={owned} priced_upper_bytes={} device_delta={} device_before={device_before} device_after={device_after}",
                allocation.name,
                allocation.logical_bytes,
                allocation.priced_bytes,
                device_after.saturating_sub(device_before),
            ));
        }
    }
    // The global counter still governs admission. Recheck the reserve after
    // allocation so pool/validation growth cannot silently exhaust headroom.
    let admission = evaluate_metal_memory_admission(
        0,
        QWEN4EXP_TEXT_SESSION_DYNAMIC_RESERVE_BYTES,
        ctx.memory_signals(),
        true,
    );
    if let Some(denied) = admission.refusal() {
        return Err(denied.into());
    }
    Ok(())
}

impl Qwen4ExpCaptureBank {
    /// Prices and admits incremental storage against current device usage before
    /// allocating. The target workspace and weights must already be resident.
    pub fn allocate(
        ctx: &MetalContext,
        plan: Qwen4ExpCapturePlan,
    ) -> Result<Self, Qwen4ExpTextSessionError> {
        let mut allocations = Vec::new();
        let mut priced_bytes = 0u64;
        for (name, logical_bytes) in plan.allocations() {
            let (priced, alignment) = price_session_allocation(
                name,
                logical_bytes,
                ctx.shared_buffer_size_and_align(logical_bytes)?,
                host_page_size_bytes()? as u64,
                ctx.max_buffer_length() as u64,
            )?;
            priced_bytes = priced_bytes.checked_add(priced).ok_or_else(|| {
                Qwen4ExpTextSessionError::Invalid("capture price overflow".into())
            })?;
            allocations.push(Qwen4ExpTextSessionAllocation {
                name: name.into(),
                logical_bytes,
                priced_bytes: priced,
                alignment,
            });
        }
        for kernel in [
            "kernel_deepseek_v4_hc_collapse",
            "kernel_deepseek_v4_hc_collapse_batch",
            "kernel_scatter_rows_f32_unique",
        ] {
            ctx.pipeline(kernel)?;
        }
        let _allocation_transaction = ctx.begin_allocation_transaction();
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
        let features = MetalTensor::zeros_f32(ctx, vec![(plan.capacity * plan.features) as u64])?;
        let mean = MetalTensor::zeros_f32(ctx, vec![plan.hidden as u64, plan.capacity as u64])?;
        let quarters = MetalTensor::zeros_f32(ctx, vec![4, plan.capacity as u64])?;
        let indices =
            MetalTensor::zeros_i32(ctx, vec![(plan.capacity * plan.layers.len()) as u64])?;
        // SAFETY: fresh owned shared buffers, checked extents, no submitted work.
        unsafe {
            let q = quarters.buffer.contents().as_ptr().cast::<f32>();
            for i in 0..4 * plan.capacity {
                *q.add(i) = 0.25;
            }
            let ids = indices.buffer.contents().as_ptr().cast::<i32>();
            for tap in 0..plan.layers.len() {
                for row in 0..plan.capacity {
                    *ids.add(tap * plan.capacity + row) = plan.scatter_row(tap, row) as i32;
                }
            }
        }
        validate_optional_owned_allocations(
            ctx,
            &[&features, &mean, &quarters, &indices],
            &allocations,
            before,
        )?;
        Ok(Self {
            plan,
            device: ctx.device.registryID(),
            features,
            mean,
            quarters,
            indices,
            allocations,
            priced_bytes,
        })
    }

    pub fn plan(&self) -> &Qwen4ExpCapturePlan {
        &self.plan
    }
    pub fn allocations(&self) -> &[Qwen4ExpTextSessionAllocation] {
        &self.allocations
    }
    pub fn priced_upper_bytes(&self) -> u64 {
        self.priced_bytes
    }

    pub(crate) fn validate(
        &self,
        ctx: &MetalContext,
        geometry: &Qwen4ExpTextSessionMetalGeometry,
    ) -> Result<(), Qwen4ExpTextSessionError> {
        if self.device != ctx.device.registryID()
            || geometry.branch_count() != 4
            || self.plan.hidden != geometry.hidden_size()
            || self.plan.layer_count != geometry.layer_count()
        {
            return invalid("capture bank device or target geometry mismatch");
        }
        Ok(())
    }

    pub(crate) fn sink(&mut self, range: Range<usize>) -> Qwen4ExpCaptureSink<'_> {
        Qwen4ExpCaptureSink { bank: self, range }
    }
}

pub(crate) struct Qwen4ExpCaptureSink<'a> {
    bank: &'a Qwen4ExpCaptureBank,
    range: Range<usize>,
}

fn overlap(window: &Range<usize>, start: usize, rows: usize) -> Range<usize> {
    let lo = window.start.max(start);
    let hi = window.end.min(start + rows);
    lo..hi.max(lo)
}

impl<'a> Qwen4ExpCaptureSink<'a> {
    pub(super) fn encode(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        layer: u32,
        hyper: &MetalTensor,
        start: usize,
        rows: usize,
    ) -> Result<(), Qwen4ExpTextSessionError> {
        let p = &self.bank.plan;
        let Some(tap) = p.layers.iter().position(|&id| id == layer) else {
            return Ok(());
        };
        let selected = overlap(&self.range, start, rows);
        if selected.is_empty() {
            return Ok(());
        }
        let n = selected.len();
        let dst_row = selected.start - self.range.start;
        let src = hyper.view_subrange(
            ((selected.start - start) * p.hidden * 4) as u64,
            vec![p.hidden as u64, 4, n as u64],
        );
        if n == 1 {
            let src = src.view_subrange(0, vec![p.hidden as u64, 4]);
            let dst = self.bank.features.view_subrange(
                (p.scatter_row(tap, dst_row) * p.hidden) as u64,
                vec![p.hidden as u64],
            );
            encode_mhc4_collapse(
                ctx,
                enc,
                p.hidden,
                &src,
                &self.bank.quarters.view_subrange(0, vec![4]),
                &dst,
            )?;
        } else {
            let mean = self
                .bank
                .mean
                .view_subrange(0, vec![p.hidden as u64, n as u64]);
            let quarters = self.bank.quarters.view_subrange(0, vec![4, n as u64]);
            encode_mhc4_collapse_rows(ctx, enc, p.hidden, n, &src, &quarters, &mean)?;
            let indices = self
                .bank
                .indices
                .view_subrange((tap * p.capacity + dst_row) as u64, vec![n as u64]);
            encode_scatter_rows_f32_unique(
                ctx,
                enc,
                &mean,
                &indices,
                &self.bank.features,
                p.hidden,
                n,
            )?;
        }
        Ok(())
    }

    pub(crate) fn complete(self) -> Qwen4ExpCapturedFeatures<'a> {
        let features = self
            .bank
            .features
            .view_subrange(0, vec![(self.range.len() * self.bank.plan.features) as u64]);
        Qwen4ExpCapturedFeatures {
            bank: self.bank,
            range: self.range,
            features,
        }
    }
}

/// Borrowed completed GPU features. The bank cannot be reused while this view
/// lives. Flat storage is `[position][tap][hidden]`, directly appendable to the
/// drafter context; positions are consecutive and absolute.
/// Consumers must finish GPU reads before dropping the view and reusing its bank.
pub struct Qwen4ExpCapturedFeatures<'a> {
    bank: &'a Qwen4ExpCaptureBank,
    range: Range<usize>,
    features: MetalTensor,
}

impl Qwen4ExpCapturedFeatures<'_> {
    pub fn tensor(&self) -> &MetalTensor {
        &self.features
    }
    pub fn positions(&self) -> Range<usize> {
        self.range.clone()
    }
    pub fn layers(&self) -> &[u32] {
        self.bank.plan.layers()
    }
    pub fn feature_width(&self) -> usize {
        self.bank.plan.feature_width()
    }
}

#[cfg(test)]
mod tests;
