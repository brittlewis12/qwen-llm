use super::*;

crate::env_flag!(
    /// Wide 64-slot routed gate/up for qualified M2048 Flash-Next tensor prefill, default on. Reuses dead routed-down output for the incumbent half input values; no new GPU allocation. Set to 0 for the 32-slot rollback; the existing expert tensor controls still apply.
    default_on enabled,
    "QWEN_FN_PREFILL_EXPERT_WIDE"
);

crate::env_flag!(
    /// Default-on 64-slot routed gate/up for the 1024..=2047-row append band, alongside the original M2048 route. Same half operand, F32 sums and dead-scratch reuse; bitwise at 1000/1383/2000 rows. The isolated gate/up is 1.26x at 2000 rows and the whole append is 3.0% faster; at 1000 rows the win is 0.5%. Set to 0 to keep the 32-slot incumbent below 2048.
    default_on short_enabled,
    "QWEN_FN_PREFILL_EXPERT_WIDE_SHORT"
);

fn rows_qualified(short: bool, tokens: usize) -> bool {
    tokens == 2048 || (short && (1024..=2047).contains(&tokens))
}

pub(super) fn try_encode(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    gate: &MetalTensor,
    up: &MetalTensor,
    input: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    inner: &MetalTensor,
    dead_down_output: &MetalTensor,
    g: Qwen4ExpMoeMetalGeometry,
    tokens: usize,
) -> Result<bool, MetalError> {
    let rows_qualified = rows_qualified(short_enabled(), tokens);
    if !enabled()
        || !rows_qualified
        || !ctx.tensor_ops_status().is_available()
        || (
            g.hidden_size,
            g.routed_intermediate_size,
            g.expert_count,
            g.experts_per_token,
        ) != (2560, 640, 512, 10)
        || gate.dtype != up.dtype
        || !ROUTED_GATE_UP_DTYPES.contains(&gate.dtype)
    {
        return Ok(false);
    }
    for dtype in [gate.dtype, up.dtype] {
        let name = crate::metal::wide_probe_pipeline_name(dtype).unwrap();
        match ctx.tensor_pipeline(&name) {
            Ok(p)
                if p.threadExecutionWidth() == 32
                    && p.maxTotalThreadsPerThreadgroup() >= 128
                    && p.staticThreadgroupMemoryLength()
                        .checked_add(16384)
                        .is_some_and(|size| size <= ctx.device.maxThreadgroupMemoryLength()) => {}
            Ok(_) | Err(MetalError::NoFunction(_) | MetalError::Pipeline(_, _)) => {
                return Ok(false);
            }
            Err(e) => return Err(e),
        }
    }
    use crate::metal::checks::{bad_shape, check_disjoint, check_physical, require_serial};
    require_serial("fn_expert_wide", enc)?;
    check_physical(
        "fn_expert_wide",
        dead_down_output,
        16,
        true,
        "dead down output",
    )?;
    let elements = g.hidden_size * tokens;
    if dead_down_output.dtype != GgmlType::F32
        || dead_down_output.n_elements() < (elements / 2) as u64
    {
        return Err(bad_shape("fn_expert_wide", "insufficient F32 dead scratch"));
    }
    let mut half = dead_down_output.view_subrange(0, vec![(elements / 2) as u64]);
    half.dtype = GgmlType::F16;
    half.shape = vec![g.hidden_size as u64, tokens as u64];
    check_disjoint(
        "fn_expert_wide",
        &half,
        &[
            (gate, "gate"),
            (up, "up"),
            (input, "input"),
            (counts, "counts"),
            (ids, "ids"),
            (inner, "inner"),
        ],
    )?;
    ctx.pipeline("kernel_scatter_offset_f32_to_f16")?;
    crate::metal::encode_scatter_offset_f32_to_f16(ctx, enc, input, &half, 0, elements)?;
    crate::metal::encode_moe_swiglu_wide_unclamped(
        ctx,
        enc,
        gate,
        up,
        &half,
        counts,
        ids,
        inner,
        g.hidden_size,
        g.routed_intermediate_size,
        g.expert_count,
        g.experts_per_token,
        tokens,
    )?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::rows_qualified;

    #[test]
    fn wide_gate_up_rows_cover_the_measured_band_only() {
        for rows in [0, 1, 128, 256, 512, 1000, 1023, 2049, 4096] {
            assert!(!rows_qualified(true, rows), "rows={rows}");
        }
        for rows in [1024, 1383, 2000, 2047, 2048] {
            assert!(rows_qualified(true, rows), "rows={rows}");
        }
        assert!(rows_qualified(false, 2048));
        for rows in [1024, 1383, 2000, 2047] {
            assert!(!rows_qualified(false, rows), "rows={rows}");
        }
    }
}
