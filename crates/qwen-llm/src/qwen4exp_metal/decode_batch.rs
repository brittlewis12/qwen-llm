//! Independent HC owners, preserving each incumbent reduction and scratch.
use super::*;
crate::env_flag!(
    /// Combine independent M5 native-decode HC launches along a sequence axis.
    /// Uses the incumbent BF16 reduction and injection; 0 keeps member launches.
    default_on enabled, "QWEN_FN_NATIVE_HC_BATCH"
);
pub(crate) struct Member<'s, 'r> {
    pub input: &'r MetalTensor,
    pub output: &'r MetalTensor,
    pub scratch: &'s mut GatedResidualMetalScratch,
}
const NORM: &str = "kernel_qwen4exp_hc_rms_norm_batch";
const DOWN: &str = "kernel_qwen4exp_hc_down_inject_bf16_batch";
const UP: &str = "kernel_qwen4exp_hc_up_mix_bf16_batch";
crate::env_flag!(
    /// Span native independent members in the existing HC residual injection operation. 0 restores one injection launch per member. The prepared coefficients and F32 expression are unchanged.
    default_on combine_enabled, "QWEN_FN_BATCH_HC_INJECTION"
);

pub(crate) fn combine(
    reads: Vec<GatedResidualMetalRead<'_, '_, '_>>,
) -> Result<(), Qwen4ExpMetalError> {
    #[cfg(feature = "perf-ablations")]
    let selected = crate::metal::fn_batch_probe::axis("hc", combine_enabled());
    #[cfg(not(feature = "perf-ablations"))]
    let selected = combine_enabled();
    let eligible = enabled()
        && selected
        && (2..=4).contains(&reads.len())
        && reads.iter().all(|r| {
            r.injection_prepared
                && (r.scratch.branch_count, r.scratch.hidden_size) == (4, 2560)
                && std::ptr::eq(r.ctx, reads[0].ctx)
                && std::ptr::eq(r.encoder, reads[0].encoder)
        });
    if !eligible {
        for read in reads {
            read.encode_combine()?;
        }
        return Ok(());
    }
    let ctx = reads[0].ctx;
    let enc = reads[0].encoder;
    enc.set_pipeline(&ctx.pipeline("kernel_fn_batch_hc_inject")?);
    enc.set_bytes_slice(0, &[4u32, 2560]);
    for i in 0..4 {
        let r = &reads[i.min(reads.len() - 1)];
        enc.set_tensor(1 + i, r.block_output);
        enc.set_tensor(5 + i, &r.scratch.injection);
        enc.set_tensor(9 + i, r.hyper_input);
    }
    enc.dispatch(
        MTLSize {
            width: 10,
            height: 4,
            depth: reads.len(),
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}
pub(crate) fn encode<'s, 'r, 'p>(
    ctx: &'p MetalContext,
    enc: &'p KernelEncoder,
    eps: f32,
    weights: GatedResidualMetalReadWeights<'_>,
    inject: &'r MetalTensor,
    mut members: Vec<Member<'s, 'r>>,
) -> Result<Vec<GatedResidualMetalRead<'s, 'r, 'p>>, Qwen4ExpMetalError> {
    validate_encoder(ctx, enc)?;
    for m in &members {
        validate_and_preflight_gated_residual_mix(
            ctx, m.input, m.output, eps, weights, inject, m.scratch,
        )?;
    }
    let use_batch = enabled()
        && (2..=4).contains(&members.len())
        && hc_read::route_enabled()
        && hc_read::incumbent_tg_enabled()
        && hc_up::bf16_enabled()
        && hc_combine_wide()
        && weights.down.dtype == GgmlType::BF16
        && weights.up.dtype == GgmlType::BF16
        && inject.dtype == GgmlType::F32
        && weights.down.offset.is_multiple_of(16)
        && members.iter().all(|m| {
            (
                m.scratch.branch_count,
                m.scratch.hidden_size,
                m.scratch.low_rank,
            ) == (4, 2560, 320)
                && m.scratch.hc_up_mix
                && m.scratch.normalized.offset.is_multiple_of(16)
                && m.scratch.low.offset.is_multiple_of(16)
        });
    let incumbent_threads = ctx
        .pipeline("kernel_qwen4exp_hc_rms_norm_f32")?
        .maxTotalThreadsPerThreadgroup()
        .min(1024);
    let capable = use_batch
        && ctx
            .pipeline("kernel_qwen4exp_hc_combine_f32")?
            .maxTotalThreadsPerThreadgroup()
            >= 1024
        && [NORM, DOWN, UP].into_iter().all(|name| {
            ctx.pipeline(name).is_ok_and(|p| {
                p.threadExecutionWidth() == 32
                    && p.maxTotalThreadsPerThreadgroup()
                        >= if name == NORM { incumbent_threads } else { 128 }
                    && p.staticThreadgroupMemoryLength() <= ctx.device.maxThreadgroupMemoryLength()
            })
        });
    if !capable {
        return members
            .into_iter()
            .map(|m| {
                encode_gated_residual_mix(
                    ctx, enc, m.input, m.output, eps, weights, inject, m.scratch,
                )
            })
            .collect();
    }
    for m in &mut members {
        reserve_command(m.scratch, enc)?;
    }
    let bind_four = |first, which| {
        for i in 0..4 {
            let m = &members[i.min(members.len() - 1)];
            let t = match which {
                0 => m.input,
                1 => &m.scratch.normalized,
                2 => &m.scratch.low,
                3 => &m.scratch.injection,
                4 => &m.scratch.raw_gate,
                5 => &m.scratch.mixed,
                _ => unreachable!(),
            };
            enc.set_tensor(first + i, t);
        }
    };
    let dispatch = |width, threads| {
        enc.dispatch(
            MTLSize {
                width,
                height: members.len(),
                depth: 1,
            },
            MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            },
        )
    };
    enc.set_pipeline(&ctx.pipeline(NORM)?);
    enc.set_bytes(
        0,
        &HcNormArgs {
            branch_count: 4,
            hidden_size: 2560,
            eps,
        },
    );
    bind_four(1, 0);
    enc.set_tensor(5, weights.norm);
    bind_four(6, 1);
    enc.set_threadgroup_memory(0, (incumbent_threads.div_ceil(32) * 4).max(32));
    dispatch(4, incumbent_threads);
    enc.set_pipeline(&ctx.pipeline(DOWN)?);
    enc.set_tensor(0, weights.down);
    bind_four(1, 1);
    bind_four(5, 2);
    enc.set_tensor(9, inject);
    bind_four(10, 3);
    enc.set_threadgroup_memory(0, 0);
    dispatch(324, 128);
    enc.set_pipeline(&ctx.pipeline(UP)?);
    enc.set_tensor(0, weights.up);
    bind_four(1, 2);
    bind_four(5, 1);
    bind_four(9, 4);
    bind_four(13, 5);
    dispatch(640, 128);
    Ok(members
        .into_iter()
        .map(|m| GatedResidualMetalRead {
            injection_prepared: true,
            ctx,
            encoder: enc,
            hyper_input: m.input,
            block_output: m.output,
            inject,
            scratch: m.scratch,
        })
        .collect())
}
