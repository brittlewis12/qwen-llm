use super::*;

pub(super) fn eligible(scratch: &GatedResidualMetalScratch, dtype: GgmlType) -> bool {
    (scratch.branch_count, scratch.hidden_size, scratch.low_rank) == (4, 2560, 320)
        && (dtype == GgmlType::Q8_0
            || (dtype == GgmlType::BF16 && bf16_enabled() && scratch.low.offset.is_multiple_of(16)))
}

/// `0` keeps BF16 HC up projections (GSQ-RCO) on the separate mat-vec and
/// gated-mean dispatches; by default they use the fused BF16 up-mix kernel
/// whenever the HC up-mix decode option is on.
const HC_UP_MIX_BF16_ENV: &str = "QWEN_FN_HC_UP_MIX_BF16";

fn bf16_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var(HC_UP_MIX_BF16_ENV).as_deref() != Ok("0"))
}

const Q8_KERNEL: &str = "kernel_qwen4exp_hc_up_mix_q8_k320";
const BF16_KERNEL: &str = "kernel_qwen4exp_hc_up_mix_bf16_k320";

pub(crate) fn preflight(ctx: &MetalContext) -> Result<(), Qwen4ExpMetalError> {
    for name in [Q8_KERNEL, BF16_KERNEL] {
        let pipeline = ctx.pipeline(name)?;
        if pipeline.threadExecutionWidth() != 32
            || pipeline.maxTotalThreadsPerThreadgroup() < 128
            || pipeline.staticThreadgroupMemoryLength() > ctx.device.maxThreadgroupMemoryLength()
        {
            return Err(invalid(
                "HC up-mix requires width32 and a128-thread cooperative pipeline",
            ));
        }
    }
    Ok(())
}

pub(crate) fn supported(ctx: &MetalContext) -> Result<bool, Qwen4ExpMetalError> {
    for name in [Q8_KERNEL, BF16_KERNEL] {
        let p = ctx.pipeline(name)?;
        if !(p.threadExecutionWidth() == 32
            && p.maxTotalThreadsPerThreadgroup() >= 128
            && p.staticThreadgroupMemoryLength() <= ctx.device.maxThreadgroupMemoryLength())
        {
            return Ok(false);
        }
    }
    Ok(true)
}

pub(super) fn encode(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    up: &MetalTensor,
    scratch: &GatedResidualMetalScratch,
) -> Result<(), MetalError> {
    let pipeline = ctx.pipeline(if up.dtype == GgmlType::BF16 {
        BF16_KERNEL
    } else {
        Q8_KERNEL
    })?;
    enc.set_pipeline(&pipeline);
    for (index, tensor) in [
        up,
        &scratch.low,
        &scratch.normalized,
        &scratch.raw_gate,
        &scratch.mixed,
    ]
    .into_iter()
    .enumerate()
    {
        enc.set_tensor(index, tensor);
    }
    enc.dispatch(
        MTLSize {
            width: 640,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use objc2_metal::{MTLCommandBuffer, MTLCommandQueue};

    /// The BF16 up-mix matches an F64 reference for the raw gates and the
    /// sigmoid-gated branch mean within F32 reduction-order error.
    #[test]
    fn bf16_up_mix_matches_f64_reference() {
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        const ROWS: usize = 4 * 2560;
        const RANK: usize = 320;
        let mut state = 0xbb67_ae85_84ca_a73bu64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % 20_001) as f32 * 1e-4 - 1.0
        };
        let weights: Vec<u16> = (0..ROWS * RANK)
            .map(|_| half::bf16::from_f32(next() * 0.2).to_bits())
            .collect();
        let low: Vec<f32> = (0..RANK).map(|_| next()).collect();
        let normalized: Vec<f32> = (0..ROWS).map(|_| next() * 3.0).collect();
        let w = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&weights),
            vec![ROWS as u64, RANK as u64],
            GgmlType::BF16,
        )
        .unwrap();
        let f32_tensor = |values: &[f32]| {
            MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(values),
                vec![values.len() as u64],
                GgmlType::F32,
            )
            .unwrap()
        };
        let (low_t, norm_t) = (f32_tensor(&low), f32_tensor(&normalized));
        let raw_t = MetalTensor::zeros_f32(&ctx, vec![ROWS as u64]).unwrap();
        let mixed_t = MetalTensor::zeros_f32(&ctx, vec![2560]).unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&command);
        enc.set_pipeline(&ctx.pipeline(BF16_KERNEL).unwrap());
        for (index, tensor) in [&w, &low_t, &norm_t, &raw_t, &mixed_t]
            .into_iter()
            .enumerate()
        {
            enc.set_tensor(index, tensor);
        }
        enc.dispatch(
            MTLSize {
                width: 640,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
        enc.end();
        command.commit();
        crate::metal::wait_completed(&command).expect("command buffer completed");
        let read = |t: &MetalTensor, n: usize| unsafe {
            std::slice::from_raw_parts(
                t.buffer
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .add(t.offset as usize)
                    .cast::<f32>(),
                n,
            )
            .to_vec()
        };
        let (raw, mixed) = (read(&raw_t, ROWS), read(&mixed_t, 2560));
        let reference_raw: Vec<f64> = (0..ROWS)
            .map(|r| {
                (0..RANK)
                    .map(|k| {
                        f64::from(half::bf16::from_bits(weights[r * RANK + k]).to_f32())
                            * f64::from(low[k])
                    })
                    .sum()
            })
            .collect();
        for r in 0..ROWS {
            assert!(
                (f64::from(raw[r]) - reference_raw[r]).abs() < 1e-4,
                "row {r}"
            );
        }
        for h in 0..2560 {
            let expected: f64 = (0..4)
                .map(|b| {
                    let row = b * 2560 + h;
                    f64::from(normalized[row]) / (1.0 + (-reference_raw[row]).exp())
                })
                .sum::<f64>()
                / 4.0;
            assert!((f64::from(mixed[h]) - expected).abs() < 1e-4, "hidden {h}");
        }
    }
}
