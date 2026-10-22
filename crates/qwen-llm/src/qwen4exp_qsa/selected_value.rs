//! Full-band selected-prefill scalar value reuse. Storage belongs to the
//! session's packed scratch; these prepared pipelines own no tensor allocation.

use super::*;
use crate::metal::Pipeline;

pub(super) const SOFTMAX: &str = "kernel_qwen4exp_qsa_selected_softmax_gqa4_f32";
pub(super) const VALUE: &str = "kernel_qwen4exp_qsa_selected_value_gqa12_f32";
const STATS_TGM: usize = 4 * ATTENTION_SCRATCH_FLOATS * size_of::<f32>();

pub(super) fn geometry_qualified(g: QwenSparseAttentionMetalGeometry) -> bool {
    (
        g.query_heads,
        g.kv_heads,
        g.head_dim,
        g.token_budget,
        g.ratio,
    ) == (24, 2, 256, 2048, 4)
}

pub(super) fn eligible(
    g: QwenSparseAttentionMetalGeometry,
    start_position: usize,
    query_count: usize,
    gqa4_enabled: bool,
) -> bool {
    gqa4_enabled
        && geometry_qualified(g)
        && query_count == DENSE_PACKED_QUERY_TILE
        && start_position >= 2051
        && g.capacity <= i32::MAX as usize
        && start_position
            .checked_add(query_count)
            .is_some_and(|end| end <= g.capacity)
}

pub(super) fn capacity_supported(
    simd_width: usize,
    max_threads: usize,
    static_bytes: usize,
    dynamic: usize,
    limit: usize,
) -> bool {
    simd_width == 32
        && max_threads >= ATTENTION_THREADS
        && static_bytes
            .checked_add(dynamic)
            .is_some_and(|n| n <= limit)
}

fn pipeline_supported(pso: &Pipeline, dynamic: usize, limit: usize) -> bool {
    capacity_supported(
        pso.threadExecutionWidth(),
        pso.maxTotalThreadsPerThreadgroup(),
        pso.staticThreadgroupMemoryLength(),
        dynamic,
        limit,
    )
}

fn optional_pipeline(ctx: &MetalContext, name: &str) -> Result<Option<Pipeline>, MetalError> {
    match ctx.pipeline(name) {
        Ok(pso) => Ok(Some(pso)),
        Err(MetalError::NoFunction(_) | MetalError::Pipeline(_, _)) => Ok(None),
        Err(error) => Err(error),
    }
}

pub(super) struct Pipelines {
    softmax: Pipeline,
    value: Pipeline,
    device: u64,
}

impl Pipelines {
    pub(super) fn prepare_for(
        ctx: &MetalContext,
        g: QwenSparseAttentionMetalGeometry,
        query_tile: usize,
        gqa4_enabled: bool,
    ) -> Result<Option<Self>, MetalError> {
        if !gqa4_enabled || !geometry_qualified(g) || query_tile != DENSE_PACKED_QUERY_TILE {
            return Ok(None);
        }
        Self::prepare(ctx)
    }

    /// Optional capability: absence or an unsupported PSO selects the incumbent
    /// before any encoding. Other errors, and all binding errors, propagate.
    pub(super) fn prepare(ctx: &MetalContext) -> Result<Option<Self>, MetalError> {
        let Some(softmax) = optional_pipeline(ctx, SOFTMAX)? else {
            return Ok(None);
        };
        let Some(value) = optional_pipeline(ctx, VALUE)? else {
            return Ok(None);
        };
        let limit = ctx.device.maxThreadgroupMemoryLength();
        if !pipeline_supported(&softmax, STATS_TGM, limit) || !pipeline_supported(&value, 0, limit)
        {
            return Ok(None);
        }
        Ok(Some(Self {
            softmax,
            value,
            device: ctx.device.registryID(),
        }))
    }

    pub(super) fn validate_support(&self, ctx: &MetalContext) -> Result<(), Qwen4ExpQsaError> {
        let limit = ctx.device.maxThreadgroupMemoryLength();
        if self.device != ctx.device.registryID()
            || !pipeline_supported(&self.softmax, STATS_TGM, limit)
            || !pipeline_supported(&self.value, 0, limit)
        {
            return invalid("selected value pipelines do not support this context");
        }
        Ok(())
    }

    /// The direct leaf accepts N1..32 for product-metallib qualification. Only
    /// `eligible` selects it in production, and that requires an actual N32 band.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn validate(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        gate: &MetalTensor,
        cache: &MetalTensor,
        ids: &MetalTensor,
        counts: &MetalTensor,
        status: &MetalTensor,
        logits: &MetalTensor,
        output: &MetalTensor,
        denominators: &MetalTensor,
        g: QwenSparseAttentionMetalGeometry,
        start: usize,
        n: usize,
    ) -> Result<(), Qwen4ExpQsaError> {
        if !geometry_qualified(g)
            || !(1..=32).contains(&n)
            || start < 2051
            || g.capacity > i32::MAX as usize
            || start.checked_add(n).is_none_or(|end| end > g.capacity)
        {
            return invalid("selected value leaf requires a bounded released selected band");
        }
        g.validate(g.capacity)?;
        validate_encoder(ctx, enc)?;
        self.validate_support(ctx)?;
        let rows = n as u64;
        let denominator_rows = match denominators.shape.as_slice() {
            [24, capacity] if (rows..=32).contains(capacity) => *capacity,
            _ => {
                return invalid("selected value denominator shape must cover N rows within 1..=32");
            }
        };
        for (name, tensor, dtype, shape, writable) in [
            (
                "selected value query/gate",
                gate,
                GgmlType::F32,
                &[12288, rows][..],
                false,
            ),
            (
                "selected value cache",
                cache,
                GgmlType::F16,
                &[256, 2, g.capacity as u64][..],
                false,
            ),
            (
                "selected value IDs",
                ids,
                GgmlType::I32,
                &[2051, rows][..],
                false,
            ),
            (
                "selected value counts",
                counts,
                GgmlType::I32,
                &[rows][..],
                false,
            ),
            (
                "selected value status",
                status,
                GgmlType::I32,
                &[rows][..],
                false,
            ),
            (
                "selected value logits",
                logits,
                GgmlType::F32,
                &[2051, 24, rows][..],
                true,
            ),
            (
                "selected value output",
                output,
                GgmlType::F32,
                &[6144, rows][..],
                true,
            ),
            (
                "selected value denominators",
                denominators,
                GgmlType::F32,
                &[24, denominator_rows][..],
                true,
            ),
        ] {
            require_tensor(name, tensor, dtype, shape, writable)?;
        }
        let tensors = [
            ("selected value query/gate", gate),
            ("selected value cache", cache),
            ("selected value IDs", ids),
            ("selected value counts", counts),
            ("selected value status", status),
            ("selected value logits", logits),
            ("selected value output", output),
            ("selected value denominators", denominators),
        ];
        require_same_device(ctx, &tensors)?;
        require_disjoint(&tensors)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        gate: &MetalTensor,
        cache: &MetalTensor,
        ids: &MetalTensor,
        counts: &MetalTensor,
        status: &MetalTensor,
        logits: &MetalTensor,
        output: &MetalTensor,
        denominators: &MetalTensor,
        g: QwenSparseAttentionMetalGeometry,
        start: usize,
        n: usize,
    ) -> Result<(), MetalError> {
        self.validate(
            ctx,
            enc,
            gate,
            cache,
            ids,
            counts,
            status,
            logits,
            output,
            denominators,
            g,
            start,
            n,
        )
        .map_err(|error| match error {
            Qwen4ExpQsaError::Metal(error) => error,
            error => MetalError::BadShape {
                kernel: VALUE,
                detail: error.to_string(),
            },
        })?;
        let args = packed_attention_args(g, start, n);
        enc.set_pipeline(&self.softmax);
        enc.set_bytes(0, &args);
        enc.set_tensor(1, counts);
        enc.set_tensor(2, status);
        enc.set_tensor(3, logits);
        enc.set_tensor(4, denominators);
        enc.set_threadgroup_memory(0, STATS_TGM);
        enc.dispatch(size(3, 2, n), size(256, 1, 1));
        enc.set_pipeline(&self.value);
        enc.set_bytes(0, &args);
        enc.set_tensor(1, gate);
        enc.set_tensor(2, cache);
        enc.set_tensor(3, ids);
        enc.set_tensor(4, counts);
        enc.set_tensor(5, status);
        enc.set_tensor(6, logits);
        enc.set_tensor(7, output);
        enc.set_tensor(8, denominators);
        enc.set_threadgroup_memory(0, 0);
        enc.dispatch(size(1, 2, n), size(256, 1, 1));
        Ok(())
    }
}

fn size(width: usize, height: usize, depth: usize) -> MTLSize {
    MTLSize {
        width,
        height,
        depth,
    }
}
