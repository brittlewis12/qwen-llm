//! Routed-expert decode projections over every selected slot in one dispatch,
//! shared by DeepSeek V4 and GLM-5.3-Flash.
//!
//! Banks are `[n_in, n_out, experts]` (contiguous per-expert matrices). Expert
//! ids (I32 `[top_k]`) and route status (I32 `[1]`) come from
//! [`encode_route_learned`]. Slot activations are slot-strided: gate/up outputs
//! and down inputs are `[ffn, top_k]`, down outputs `[hidden, top_k]`. With an
//! unready route the DS4 kernels write zero rows; the IQ4_XS kernel leaves its
//! output unspecified, so callers must check route status before using results
//! (both families validate route records after each token's command).

use super::checks::{bad_shape, check_expert_bank, check_tensor, require_serial, to_u32};
use super::*;

/// Gate/up dtypes with a fused all-slot `silu(min(g, c)) * clamp(u, -c, c)`.
const GATE_UP_KERNELS: &[(GgmlType, &str)] = &[
    (
        GgmlType::IQ2_XS,
        "kernel_deepseek_v4_all_slots_swiglu_iq2_xs_f32_fast",
    ),
    (
        GgmlType::IQ2_S,
        "kernel_deepseek_v4_all_slots_swiglu_iq2_s_f32_fast",
    ),
    (
        GgmlType::IQ3_XXS,
        "kernel_deepseek_v4_all_slots_swiglu_iq3_xxs_f32_fast",
    ),
    (
        GgmlType::IQ3_S,
        "kernel_deepseek_v4_all_slots_swiglu_iq3_s_f32_fast",
    ),
    (
        GgmlType::Q3_K,
        "kernel_deepseek_v4_all_slots_swiglu_q3_K_f32",
    ),
];

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AllSlotsArgs {
    n_in: u32,
    n_out: u32,
    n_expert: u32,
    top_k: u32,
    clamp: f32,
}

fn check_route(
    kernel: &'static str,
    ids: &MetalTensor,
    status: &MetalTensor,
    top_k: usize,
) -> Result<(), MetalError> {
    if top_k == 0 || top_k > ROUTE_MAX_TOP_K {
        return Err(bad_shape(
            kernel,
            format!("top-k {top_k} outside 1..={ROUTE_MAX_TOP_K}"),
        ));
    }
    check_tensor(
        kernel,
        ids,
        GgmlType::I32,
        &[top_k as u64],
        false,
        "expert ids",
    )?;
    check_tensor(kernel, status, GgmlType::I32, &[1], false, "route status")
}

fn pipeline_for(
    ctx: &MetalContext,
    kernel: &'static str,
    threads: usize,
) -> Result<Pipeline, MetalError> {
    let pso = ctx.pipeline(kernel)?;
    if pso.threadExecutionWidth() != 32 || pso.maxTotalThreadsPerThreadgroup() < threads {
        return Err(bad_shape(
            kernel,
            format!(
                "needs SIMD width 32 and {threads} threads, pipeline has {} and {}",
                pso.threadExecutionWidth(),
                pso.maxTotalThreadsPerThreadgroup()
            ),
        ));
    }
    Ok(pso)
}

/// `output[:, s] = silu(min(gate_s x, clamp)) * clamp(up_s x, -clamp, clamp)` for
/// each routed slot `s`, where `gate_s`/`up_s` are slot `s`'s expert matrices.
#[allow(clippy::too_many_arguments)]
pub fn encode_all_slots_gate_up_swiglu(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    gate_bank: &MetalTensor,
    up_bank: &MetalTensor,
    input: &MetalTensor,
    ids: &MetalTensor,
    status: &MetalTensor,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    experts: usize,
    top_k: usize,
    clamp: f32,
) -> Result<(), MetalError> {
    const K: &str = "all_slots_gate_up_swiglu";
    require_serial(K, enc)?;
    if !clamp.is_finite() || clamp <= 0.0 {
        return Err(bad_shape(K, "clamp must be finite and positive"));
    }
    check_route(K, ids, status, top_k)?;
    check_expert_bank(K, gate_bank, n_in, n_out, experts, "gate bank")?;
    check_expert_bank(K, up_bank, n_in, n_out, experts, "up bank")?;
    check_tensor(K, input, GgmlType::F32, &[n_in as u64], false, "input")?;
    let out_shape = [n_out as u64, top_k as u64];
    check_tensor(K, output, GgmlType::F32, &out_shape, true, "output")?;
    if gate_bank.dtype != up_bank.dtype {
        return Err(bad_shape(K, "gate and up banks must share a dtype"));
    }
    let kernel = GATE_UP_KERNELS
        .iter()
        .find(|(dtype, _)| *dtype == gate_bank.dtype)
        .map(|(_, kernel)| *kernel)
        .ok_or_else(|| bad_shape(K, format!("no all-slot kernel for {:?}", gate_bank.dtype)))?;
    if !n_in.is_multiple_of(256) {
        return Err(bad_shape(
            K,
            format!("input width {n_in} is not a multiple of 256"),
        ));
    }
    let pso = pipeline_for(ctx, kernel, 64)?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &AllSlotsArgs {
            n_in: to_u32(K, n_in, "n_in")?,
            n_out: to_u32(K, n_out, "n_out")?,
            n_expert: to_u32(K, experts, "experts")?,
            top_k: top_k as u32,
            clamp,
        },
    );
    enc.set_tensor(1, gate_bank);
    enc.set_tensor(2, up_bank);
    enc.set_tensor(3, input);
    enc.set_tensor(4, ids);
    enc.set_tensor(5, status);
    enc.set_tensor(6, output);
    enc.set_threadgroup_memory(0, 16 * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(8),
            height: top_k,
            depth: 1,
        },
        MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// `output[:, s] = down_s input[:, s]` for each routed slot `s`.
#[allow(clippy::too_many_arguments)]
pub fn encode_all_slots_down(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    bank: &MetalTensor,
    input: &MetalTensor,
    ids: &MetalTensor,
    status: &MetalTensor,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    experts: usize,
    top_k: usize,
) -> Result<(), MetalError> {
    const K: &str = "all_slots_down";
    require_serial(K, enc)?;
    check_route(K, ids, status, top_k)?;
    check_expert_bank(K, bank, n_in, n_out, experts, "down bank")?;
    let in_shape = [n_in as u64, top_k as u64];
    let out_shape = [n_out as u64, top_k as u64];
    check_tensor(K, input, GgmlType::F32, &in_shape, false, "input")?;
    check_tensor(K, output, GgmlType::F32, &out_shape, true, "output")?;
    let args = AllSlotsArgs {
        n_in: to_u32(K, n_in, "n_in")?,
        n_out: to_u32(K, n_out, "n_out")?,
        n_expert: to_u32(K, experts, "experts")?,
        top_k: top_k as u32,
        clamp: 0.0,
    };
    if bank.dtype == GgmlType::IQ4_XS {
        // Qwen's all-slot IQ4_XS down: no status binding, 32-entry LUT.
        if !n_in.is_multiple_of(256) {
            return Err(bad_shape(
                K,
                format!("input width {n_in} is not a multiple of 256"),
            ));
        }
        let pso = pipeline_for(ctx, "kernel_moe_down_iq4_xs_f32_fast", 64)?;
        enc.set_pipeline(&pso);
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Iq4XsArgs {
            n_in: u32,
            n_out: u32,
            n_expert: u32,
            top_k: u32,
        }
        enc.set_bytes(
            0,
            &Iq4XsArgs {
                n_in: args.n_in,
                n_out: args.n_out,
                n_expert: args.n_expert,
                top_k: args.top_k,
            },
        );
        enc.set_tensor(1, bank);
        enc.set_tensor(2, input);
        enc.set_tensor(3, ids);
        enc.set_tensor(4, output);
        enc.set_threadgroup_memory(0, 32 * std::mem::size_of::<f32>());
        enc.dispatch(
            MTLSize {
                width: n_out.div_ceil(4),
                height: top_k,
                depth: 1,
            },
            MTLSize {
                width: 64,
                height: 1,
                depth: 1,
            },
        );
        return Ok(());
    }
    let (kernel, rows_per_group, threads, block) = match bank.dtype {
        GgmlType::IQ3_S => (
            "kernel_deepseek_v4_all_slots_down_iq3_s_f32_fast",
            8,
            64,
            256,
        ),
        GgmlType::IQ3_XXS => (
            "kernel_deepseek_v4_all_slots_down_iq3_xxs_f32_fast",
            8,
            64,
            256,
        ),
        GgmlType::MXFP4 => ("kernel_deepseek_v4_all_slots_down_mxfp4_f32", 4, 128, 32),
        GgmlType::Q4_K => ("kernel_deepseek_v4_all_slots_down_q4_K_f32", 4, 64, 256),
        dtype => return Err(bad_shape(K, format!("no all-slot kernel for {dtype:?}"))),
    };
    if !n_in.is_multiple_of(block) {
        return Err(bad_shape(
            K,
            format!("input width {n_in} is not a multiple of {block}"),
        ));
    }
    let pso = pipeline_for(ctx, kernel, threads)?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, bank);
    enc.set_tensor(2, input);
    enc.set_tensor(3, ids);
    enc.set_tensor(4, status);
    enc.set_tensor(5, output);
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(rows_per_group),
            height: top_k,
            depth: 1,
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{assert_moe_oracle_close, dequant_expert, offset_tensor};
    use super::*;

    const EXPERTS: usize = 12;
    const TOP_K: usize = 8;
    const HIDDEN: usize = 4096;
    const FFN: usize = 2048;
    const IDS: [i32; TOP_K] = [11, 3, 7, 0, 9, 5, 10, 2];

    /// Arbitrary quant payload bytes with a small finite F16 scale per block
    /// (every IQ2_S/IQ3_S/IQ4_XS byte pattern decodes).
    fn synthetic_bank(dtype: GgmlType, n_in: usize, n_out: usize, seed: u64) -> Vec<u8> {
        let (block, block_bytes) = dtype.storage_layout().unwrap();
        let blocks = EXPERTS * n_out * (n_in / block as usize);
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut bytes = vec![0u8; blocks * block_bytes as usize];
        for byte in bytes.iter_mut() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *byte = state as u8;
        }
        for (i, chunk) in bytes.chunks_exact_mut(block_bytes as usize).enumerate() {
            let d = if i % 2 == 0 { 1.0 } else { -1.0 } * ((i % 7) + 1) as f32 / 4096.0;
            chunk[..2].copy_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        }
        bytes
    }

    fn bank(
        ctx: &MetalContext,
        bytes: &[u8],
        dtype: GgmlType,
        n_in: usize,
        n_out: usize,
    ) -> MetalTensor {
        offset_tensor(
            ctx,
            256,
            bytes,
            64,
            vec![n_in as u64, n_out as u64, EXPERTS as u64],
            dtype,
        )
    }

    fn f32_tensor(ctx: &MetalContext, values: &[f32], shape: Vec<u64>) -> MetalTensor {
        offset_tensor(
            ctx,
            16,
            bytemuck::cast_slice(values),
            20,
            shape,
            GgmlType::F32,
        )
    }

    fn i32_tensor(ctx: &MetalContext, values: &[i32]) -> MetalTensor {
        offset_tensor(
            ctx,
            16,
            bytemuck::cast_slice(values),
            20,
            vec![values.len() as u64],
            GgmlType::I32,
        )
    }

    fn read_f32(tensor: &MetalTensor) -> Vec<f32> {
        super::super::test_support::tensor_f32_at_offset(tensor)
    }

    fn mat_vec(weights: &[f32], x: &[f32], n_out: usize) -> Vec<f32> {
        let n_in = x.len();
        (0..n_out)
            .map(|r| {
                weights[r * n_in..(r + 1) * n_in]
                    .iter()
                    .zip(x)
                    .map(|(w, v)| (*w as f64) * (*v as f64))
                    .sum::<f64>() as f32
            })
            .collect()
    }

    fn values(n: usize, seed: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (((i * 31 + seed * 17) % 97) as f32 - 48.0) * 0.021)
            .collect()
    }

    fn run(ctx: &MetalContext, encode: impl FnOnce(&KernelEncoder)) {
        let command = ctx.queue.commandBuffer().expect("command buffer");
        let encoder = KernelEncoder::begin(&command);
        encode(&encoder);
        encoder.end();
        command.commit();
        wait_completed(&command).expect("command buffer failed");
    }

    /// GLM-5.3 widths and top-8 over a reduced bank: fused gate/up clamp for
    /// IQ2_S and IQ3_S, against llama.cpp CPU dequantization.
    #[test]
    fn all_slots_gate_up_matches_dequantized_reference_at_glm_widths() {
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        let x = values(HIDDEN, 1);
        for dtype in [GgmlType::IQ2_S, GgmlType::IQ3_S] {
            let gate_bytes = synthetic_bank(dtype, HIDDEN, FFN, 3);
            let up_bytes = synthetic_bank(dtype, HIDDEN, FFN, 5);
            let gate = bank(&ctx, &gate_bytes, dtype, HIDDEN, FFN);
            let up = bank(&ctx, &up_bytes, dtype, HIDDEN, FFN);
            let input = f32_tensor(&ctx, &x, vec![HIDDEN as u64]);
            let ids = i32_tensor(&ctx, &IDS);
            let ready = i32_tensor(&ctx, &[ROUTE_STATUS_READY]);
            let unready = i32_tensor(&ctx, &[ROUTE_STATUS_PENDING]);
            let out = f32_tensor(
                &ctx,
                &vec![7.0; FFN * TOP_K],
                vec![FFN as u64, TOP_K as u64],
            );
            let zeroed = f32_tensor(
                &ctx,
                &vec![7.0; FFN * TOP_K],
                vec![FFN as u64, TOP_K as u64],
            );
            run(&ctx, |enc| {
                encode_all_slots_gate_up_swiglu(
                    &ctx, enc, &gate, &up, &input, &ids, &ready, &out, HIDDEN, FFN, EXPERTS, TOP_K,
                    10.0,
                )
                .unwrap();
                encode_all_slots_gate_up_swiglu(
                    &ctx, enc, &gate, &up, &input, &ids, &unready, &zeroed, HIDDEN, FFN, EXPERTS,
                    TOP_K, 10.0,
                )
                .unwrap();
            });
            let mut expected = Vec::with_capacity(FFN * TOP_K);
            for &expert in &IDS {
                let g = mat_vec(
                    &dequant_expert(&gate_bytes, dtype, HIDDEN, FFN, expert as usize),
                    &x,
                    FFN,
                );
                let u = mat_vec(
                    &dequant_expert(&up_bytes, dtype, HIDDEN, FFN, expert as usize),
                    &x,
                    FFN,
                );
                expected.extend(g.iter().zip(&u).map(|(&g, &u)| {
                    let g = g.min(10.0);
                    g / (1.0 + (-g).exp()) * u.clamp(-10.0, 10.0)
                }));
            }
            assert_moe_oracle_close(&format!("{dtype:?} gate/up"), &read_f32(&out), &expected);
            assert!(
                read_f32(&zeroed).iter().all(|&v| v == 0.0),
                "{dtype:?} unready route"
            );
        }
    }

    /// GLM-5.3 down widths and top-8: the new all-slot IQ3_S kernel and the
    /// bridged IQ4_XS kernel against llama.cpp CPU dequantization.
    #[test]
    fn all_slots_down_matches_dequantized_reference_at_glm_widths() {
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        let inner = values(FFN * TOP_K, 2);
        for dtype in [GgmlType::IQ3_S, GgmlType::IQ4_XS] {
            let bytes = synthetic_bank(dtype, FFN, HIDDEN, 7);
            let down = bank(&ctx, &bytes, dtype, FFN, HIDDEN);
            let input = f32_tensor(&ctx, &inner, vec![FFN as u64, TOP_K as u64]);
            let ids = i32_tensor(&ctx, &IDS);
            let status = i32_tensor(&ctx, &[ROUTE_STATUS_READY]);
            let out = f32_tensor(
                &ctx,
                &vec![7.0; HIDDEN * TOP_K],
                vec![HIDDEN as u64, TOP_K as u64],
            );
            run(&ctx, |enc| {
                encode_all_slots_down(
                    &ctx, enc, &down, &input, &ids, &status, &out, FFN, HIDDEN, EXPERTS, TOP_K,
                )
                .unwrap();
            });
            let mut expected = Vec::with_capacity(HIDDEN * TOP_K);
            for (slot, &expert) in IDS.iter().enumerate() {
                let w = dequant_expert(&bytes, dtype, FFN, HIDDEN, expert as usize);
                expected.extend(mat_vec(&w, &inner[slot * FFN..(slot + 1) * FFN], HIDDEN));
            }
            assert_moe_oracle_close(&format!("{dtype:?} down"), &read_f32(&out), &expected);
        }
        // The DS4 IQ3_S kernel writes zero rows for an unready route.
        let bytes = synthetic_bank(GgmlType::IQ3_S, FFN, HIDDEN, 7);
        let down = bank(&ctx, &bytes, GgmlType::IQ3_S, FFN, HIDDEN);
        let input = f32_tensor(&ctx, &inner, vec![FFN as u64, TOP_K as u64]);
        let ids = i32_tensor(&ctx, &IDS);
        let unready = i32_tensor(&ctx, &[ROUTE_STATUS_PENDING]);
        let out = f32_tensor(
            &ctx,
            &vec![7.0; HIDDEN * TOP_K],
            vec![HIDDEN as u64, TOP_K as u64],
        );
        run(&ctx, |enc| {
            encode_all_slots_down(
                &ctx, enc, &down, &input, &ids, &unready, &out, FFN, HIDDEN, EXPERTS, TOP_K,
            )
            .unwrap();
        });
        assert!(read_f32(&out).iter().all(|&v| v == 0.0));
    }
}
