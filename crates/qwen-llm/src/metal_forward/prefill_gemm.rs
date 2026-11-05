//! Dense 27B tensor prompt GEMMs. Decode and other model families retain
//! their incumbent dispatch. Q5_K already has its own tensor route.

use super::{MfError, encode_mat_mat_dispatch};
use crate::metal::{KernelEncoder, MetalContext, MetalTensor};
use crate::model::{Arch, QWEN3_27B};
use crate::tensor::GgmlType;
use std::sync::OnceLock;

/// Metal 4 half-operand tensor GEMMs for dense Qwen 27B Q4_K/Q6_K prompt projections at >=128 rows, enabled by default when the tensor tier is available. Set to 0 for A/B comparisons or rollback. Other devices, models, dtypes and small tails use incumbent dispatch; Q5_K keeps its existing tensor route and the endpoint head remains GEMV.
const QWEN27B_PREFILL_TENSOR_ENV: &str = "QWEN_27B_PREFILL_TENSOR";

fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| crate::env_flag::read_default_on(QWEN27B_PREFILL_TENSOR_ENV))
}

fn eligible(
    on: bool,
    tensor_available: bool,
    arch: &Arch,
    dtype: GgmlType,
    k: usize,
    n: usize,
    m: usize,
) -> bool {
    on && tensor_available
        && *arch == QWEN3_27B
        && matches!(dtype, GgmlType::Q4_K | GgmlType::Q6_K)
        && m >= 128
        && n >= 64
        && k > 0
        && k.is_multiple_of(256)
}

/// Selection proof for diagnostic callers. Availability is the context's
/// Apple10 + Metal4/library gate, not a device-name inference.
pub fn qwen27b_prefill_tensor_selected(
    ctx: &MetalContext,
    arch: &Arch,
    dtype: GgmlType,
    k: usize,
    n: usize,
    m: usize,
) -> bool {
    eligible(
        enabled(),
        ctx.tensor_ops_status().is_available(),
        arch,
        dtype,
        k,
        n,
        m,
    )
}

/// Prompt-only projection entry. Both operands already enter half staging in
/// the incumbent kernels; this changes the MMA primitive/reduction order.
/// Norm-fed inputs retain sqrt(H)*max|gamma| bounds; unclamped down inputs
/// retain the existing half-range/telemetry contract. No new narrowing site.
#[allow(clippy::too_many_arguments)]
pub fn encode_qwen27b_prefill_mat_mat(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    arch: &Arch,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    k: usize,
    n: usize,
    m: usize,
) -> Result<(), MfError> {
    if qwen27b_prefill_tensor_selected(ctx, arch, weight.dtype, k, n, m) {
        crate::metal::checks::check_disjoint(
            "qwen27b_prefill_tensor",
            y,
            &[(weight, "weight"), (x, "input")],
        )?;
        if k > n {
            crate::half_telemetry::record(
                ctx,
                enc,
                crate::half_telemetry::SITE_DENSE_DOWN,
                u32::MAX,
                x,
            );
        }
        crate::metal::encode_mat_mat_tensor_f32_narrow(ctx, enc, weight, x, y, k, n, m)?;
        Ok(())
    } else {
        encode_mat_mat_dispatch(ctx, enc, weight, x, y, k, n, m)
    }
}

#[cfg(test)]
#[test]
fn qwen27b_prefill_tensor_gate_preserves_fallbacks() {
    let a = QWEN3_27B;
    assert!(eligible(true, true, &a, GgmlType::Q4_K, 5120, 17408, 1024));
    assert!(eligible(true, true, &a, GgmlType::Q6_K, 17408, 5120, 129));
    assert!(!eligible(
        false,
        true,
        &a,
        GgmlType::Q4_K,
        5120,
        17408,
        1024
    ));
    assert!(!eligible(
        true,
        false,
        &a,
        GgmlType::Q4_K,
        5120,
        17408,
        1024
    ));
    for (k, n, m) in [
        (5120, 17408, 1),
        (5120, 17408, 127),
        (256, 1, 128),
        (257, 64, 128),
    ] {
        assert!(!eligible(true, true, &a, GgmlType::Q4_K, k, n, m));
    }
    for dtype in [GgmlType::F32, GgmlType::Q5_K, GgmlType::Q8_0] {
        assert!(!eligible(true, true, &a, dtype, 5120, 17408, 1024));
    }
    let mut other = a;
    other.kind = crate::model::ArchKind::Moe;
    assert!(!eligible(
        true,
        true,
        &other,
        GgmlType::Q4_K,
        5120,
        17408,
        1024
    ));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::{dispatch_census_begin, dispatch_census_take, wait_completed};
    use crate::tensor::TensorDesc;
    use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandQueue};

    fn random(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    fn weights(dtype: GgmlType, k: usize, n: usize) -> (Vec<u8>, Vec<f32>) {
        let block_bytes = if dtype == GgmlType::Q4_K { 144 } else { 210 };
        let mut packed = vec![0; k / 256 * n * block_bytes];
        let mut rng = 0x4e13_c9d2;
        for (ordinal, block) in packed.chunks_exact_mut(block_bytes).enumerate() {
            for byte in block.iter_mut() {
                *byte = random(&mut rng) as u8;
            }
            let d = half::f16::from_f32([0.0005, 0.003, 1.0 / 64.0][ordinal % 3]);
            if dtype == GgmlType::Q4_K {
                block[..2].copy_from_slice(&d.to_le_bytes());
                block[2..4].copy_from_slice(&half::f16::from_f32(d.to_f32() * 0.5).to_le_bytes());
            } else {
                block[208..210].copy_from_slice(&d.to_le_bytes());
                block[192..195].copy_from_slice(&[0, 128, 127]);
            }
        }
        let desc = TensorDesc {
            name: "prefill-projection-regression".into(),
            shape: vec![k as u64, n as u64],
            dtype,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: packed.len() as u64,
        };
        let decoded = crate::codec::dequant_to_f32(&desc, &packed).unwrap();
        (packed, decoded)
    }

    fn errors(values: &[f32], reference: &[f64]) -> (f64, f64) {
        let mut peak = 0.0f64;
        let mut squared = 0.0;
        let mut energy = 0.0;
        for (&value, &expected) in values.iter().zip(reference) {
            assert!(value.is_finite() && expected.is_finite());
            let difference = f64::from(value) - expected;
            peak = peak.max(difference.abs());
            squared += difference * difference;
            energy += expected * expected;
        }
        (peak, (squared / energy).sqrt())
    }

    #[test]
    fn qwen27b_prefill_tensor_gpu_regression() {
        // Unit contexts have isolated test locks; also retain the normal
        // production lease so this value test cannot overlap another lane.
        let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        const GUARD: usize = 16;
        const CANARY: u32 = 0x45ab_cdef;
        let mut cases = 0;
        let mut routed = 0;
        for dtype in [GgmlType::Q4_K, GgmlType::Q6_K] {
            // Full tiles, M/N tails, small-M and narrow-N fallbacks, and an
            // unsupported model. Quantized K must consist of whole blocks.
            for (m, n, k, other_model) in [
                (128, 64, 256, false),
                (129, 65, 512, false),
                (127, 65, 256, false),
                (3, 1, 256, false),
                (128, 64, 256, true),
            ] {
                let mut arch = QWEN3_27B;
                if other_model {
                    arch.kind = crate::model::ArchKind::Moe;
                }
                let (packed, decoded) = weights(dtype, k, n);
                let w = MetalTensor::from_bytes(&ctx, &packed, vec![k as u64, n as u64], dtype)
                    .unwrap();
                for amplitude in [2f32.powi(-24), 1.0, 2048.0, 65504.0] {
                    let mut rng = 0x5a31_89ef;
                    let input = (0..m * k)
                        .map(|_| {
                            ((random(&mut rng) >> 40) as f32 / (1u64 << 23) as f32 - 1.0)
                                * amplitude
                        })
                        .collect::<Vec<_>>();
                    let reference = (0..m * n)
                        .map(|i| {
                            let (row, col) = (i / n, i % n);
                            input[row * k..(row + 1) * k]
                                .iter()
                                .zip(&decoded[col * k..(col + 1) * k])
                                .map(|(&x, &w)| f64::from(x) * f64::from(w))
                                .sum::<f64>()
                        })
                        .collect::<Vec<_>>();
                    let x = MetalTensor::from_bytes(
                        &ctx,
                        bytemuck::cast_slice(&input),
                        vec![k as u64, m as u64],
                        GgmlType::F32,
                    )
                    .unwrap();
                    let mut outputs = Vec::new();
                    let selected = qwen27b_prefill_tensor_selected(&ctx, &arch, dtype, k, n, m);
                    for production in [false, true] {
                        let mut poison = vec![CANARY; m * n + 2 * GUARD];
                        poison[GUARD..GUARD + m * n].fill(f32::NAN.to_bits());
                        let backing = MetalTensor::from_bytes(
                            &ctx,
                            bytemuck::cast_slice(&poison),
                            vec![poison.len() as u64],
                            GgmlType::F32,
                        )
                        .unwrap();
                        let y = backing.view_subrange(GUARD as u64, vec![n as u64, m as u64]);
                        dispatch_census_begin();
                        let cmd = ctx.queue.commandBuffer().unwrap();
                        let enc = KernelEncoder::begin(&cmd);
                        if production {
                            encode_qwen27b_prefill_mat_mat(&ctx, &enc, &arch, &w, &x, &y, k, n, m)
                                .unwrap();
                        } else {
                            encode_mat_mat_dispatch(&ctx, &enc, &w, &x, &y, k, n, m).unwrap();
                        }
                        enc.end();
                        cmd.commit();
                        wait_completed(&cmd).unwrap();
                        let dispatches = dispatch_census_take();
                        if production && selected {
                            assert!(dispatches.iter().any(|d| d.kernel.ends_with("_f32_tensor")));
                        }
                        let data = unsafe {
                            std::slice::from_raw_parts(
                                backing.buffer.contents().as_ptr().cast::<u32>(),
                                poison.len(),
                            )
                        };
                        assert!(
                            data[..GUARD]
                                .iter()
                                .chain(&data[GUARD + m * n..])
                                .all(|&v| v == CANARY)
                        );
                        outputs.push(
                            data[GUARD..GUARD + m * n]
                                .iter()
                                .map(|&v| f32::from_bits(v))
                                .collect::<Vec<_>>(),
                        );
                    }
                    let incumbent = errors(&outputs[0], &reference);
                    let actual = errors(&outputs[1], &reference);
                    assert!(
                        actual.0 <= 1.5 * incumbent.0 && actual.1 <= 1.5 * incumbent.1,
                        "{dtype:?} M{m}/N{n}/K{k} amplitude={amplitude}: production={actual:?}, incumbent={incumbent:?}"
                    );
                    if !selected {
                        assert!(
                            outputs[0]
                                .iter()
                                .zip(&outputs[1])
                                .all(|(a, b)| a.to_bits() == b.to_bits())
                        );
                    }
                    cases += 1;
                    routed += usize::from(selected);
                }
            }
        }
        eprintln!(
            "qwen27b_prefill_tensor: {cases} F64 cases passed, {routed} tensor-selected, {} bitwise fallbacks; tier={:?}",
            cases - routed,
            ctx.tensor_ops_status()
        );
    }
}
