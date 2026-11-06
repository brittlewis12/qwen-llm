//! Tensor prompt projections for dense 27B and 35B-A3B, plus qualified
//! A3B grouped experts and Flash-Next prompt projections. Decode retains its incumbent dispatch.

use super::{MfError, encode_mat_mat_dispatch};
use crate::metal::{KernelEncoder, MetalContext, MetalTensor};
use crate::model::{Arch, ArchKind, QWEN3_27B};
use crate::tensor::GgmlType;
use std::sync::OnceLock;

/// Metal 4 half-operand tensor prompt projections for dense Qwen 27B Q4_K/Q6_K and Qwen 35B-A3B Q8_0 at >=128 rows, plus screened Flash-Next Q4_K/Q6_K/IQ4_XS/IQ4_NL projections at 2048 rows or 512..2047 rows with QWEN_FN_SHORT_APPEND_TENSOR, enabled by default when the tensor tier is available. Set to 0 for A/B or rollback. Qualified 27B FFN shapes at 1024 rows use a strict device-half 128-row tile with reused scratch. Other architectures, dtypes, shapes and small tails use incumbent dispatch; Q5_K retains its existing route and the endpoint head remains GEMV.

const DENSE_PREFILL_TENSOR_ENV: &str = "QWEN_DENSE_PREFILL_TENSOR";

crate::env_flag!(
    /// Wide 64-row Metal 4 tensor tile for the qualified Flash-Next dense projection cohorts (2048 rows, or 512..2047 with QWEN_FN_SHORT_APPEND_TENSOR), default on. Reuses the incumbent half operands and F32 accumulation with no new GPU buffers. Set to 0 to retain the 32-row tile; QWEN_DENSE_PREFILL_TENSOR=0 still disables both.
    default_on flash_next_dense_wide_enabled,
    "QWEN_FN_PREFILL_DENSE_WIDE"
);

/// Experimental row-range extensions for 27B FFN and A3B routed tensor prefill. Default off during qualification; original 1024/2048-row routes are unaffected. Set to 0 to retain the original short-append dispatches.
const SHORT_APPEND_TENSOR_ENV: &str = "QWEN_SHORT_APPEND_TENSOR";

fn short_append_tensor_enabled() -> bool {
    #[cfg(test)]
    if let Some(value) = short_append_probe::override_value() {
        return value;
    }
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| crate::env_flag::read_default_off(SHORT_APPEND_TENSOR_ENV))
}

#[cfg(test)]
mod short_append_probe;

/// Strict device-half 128-row tile for qualified 27B FFN projections at exactly 1024 rows, default on when QWEN_DENSE_PREFILL_TENSOR and the tensor tier are available. Set to 0 to retain the previous narrow B tile for A/B or rollback. Other shapes and A3B retain their existing tiles.
const FFN128_ENV: &str = "QWEN_27B_FFN_TENSOR128";

fn ffn128_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| crate::env_flag::read_default_on(FFN128_ENV))
}

/// Metal 4 Q4_K gate/up and Q5_K/Q6_K down grouped expert tiles for stock Qwen 35B-A3B prefill at exactly 2048 rows, enabled by default when the tensor tier is available. Set to 0 for A/B or rollback. Other architectures, quant combinations and chunk sizes retain their incumbent expert kernels.
const A3B_EXPERT_PREFILL_TENSOR_ENV: &str = "QWEN_A3B_EXPERT_PREFILL_TENSOR";

const A3B_ARCH: Arch = Arch {
    kind: ArchKind::Moe,
    n_layer: 40,
    hidden_size: 2048,
    intermediate_size: 0,
    vocab_size: 248_320,
    full_attention_interval: 4,
    n_q_heads: 16,
    n_kv_heads: 2,
    attn_head_dim: 256,
    rope_theta: 10_000_000.0,
    partial_rotary_factor: 0.25,
    gdn_n_v_heads: 32,
    gdn_n_k_heads: 16,
    gdn_head_dim: 128,
    gdn_conv_kernel: 4,
    expert_count: 256,
    expert_used_count: 8,
    expert_feed_forward_length: 512,
    expert_shared_feed_forward_length: 512,
    mtp_n_hidden_layers: 0,
};

fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| crate::env_flag::read_default_on(DENSE_PREFILL_TENSOR_ENV))
}

fn expert_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| crate::env_flag::read_default_on(A3B_EXPERT_PREFILL_TENSOR_ENV))
}

fn expert_eligible(
    on: bool,
    available: bool,
    arch: &Arch,
    gate: GgmlType,
    up: GgmlType,
    down: GgmlType,
    m: usize,
) -> bool {
    on && available
        && *arch == A3B_ARCH
        && (m == 2048 || (short_append_tensor_enabled() && (128..=2048).contains(&m)))
        && gate == GgmlType::Q4_K
        && up == GgmlType::Q4_K
        && matches!(down, GgmlType::Q5_K | GgmlType::Q6_K)
}

/// Selection proof for the independently controlled grouped-expert route.
pub fn qwen_a3b_expert_prefill_tensor_selected(
    ctx: &MetalContext,
    arch: &Arch,
    gate: GgmlType,
    up: GgmlType,
    down: GgmlType,
    m: usize,
) -> bool {
    expert_eligible(
        expert_enabled(),
        ctx.tensor_ops_status().is_available(),
        arch,
        gate,
        up,
        down,
        m,
    )
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
    if !on || !tensor_available || m < 128 {
        return false;
    }
    if *arch == QWEN3_27B {
        matches!(dtype, GgmlType::Q4_K | GgmlType::Q6_K)
            && n >= 64
            && k > 0
            && k.is_multiple_of(256)
    } else {
        *arch == A3B_ARCH
            && dtype == GgmlType::Q8_0
            && matches!(
                (k, n),
                (2048, 8192) | (2048, 4096) | (4096, 2048) | (2048, 512) | (512, 2048)
            )
    }
}

// Only real GSQ-RCO dtype/shape cohorts screened on M5. K/N here are
// projection dimensions, not architecture inference: callers are exclusively
// Flash-Next packed GDN, QSA and shared-expert motors.
fn flash_next_eligible(
    on: bool,
    available: bool,
    dtype: GgmlType,
    k: usize,
    n: usize,
    m: usize,
) -> bool {
    on && available
        && crate::qwen4exp_prefill_tensor::rows_qualified(m)
        && match dtype {
            GgmlType::Q4_K => matches!((k, n), (2560, 640 | 6144 | 10240 | 12288) | (6144, 2560)),
            GgmlType::Q6_K => matches!(
                (k, n),
                (2560, 512 | 640 | 6144 | 10240 | 12288) | (6144, 2560)
            ),
            GgmlType::IQ4_XS => matches!((k, n), (2560, 512 | 640 | 6144 | 10240 | 12288)),
            GgmlType::IQ4_NL => (k, n) == (640, 2560),
            _ => false,
        }
}

/// Flash-Next packed projections reuse the existing half-operand tensor tiles.
/// Both routes already stage half operands; existing caller down telemetry is
/// retained, along with the generic dispatcher's dense-down telemetry below.
pub(crate) fn encode_qwen4exp_prefill_mat_mat(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    k: usize,
    n: usize,
    m: usize,
) -> Result<(), MfError> {
    if crate::qwen4exp_prefill_tensor::small_f32::try_encode(ctx, enc, weight, x, y, k, n, m)? {
        return Ok(());
    }
    if flash_next_eligible(
        enabled(),
        ctx.tensor_ops_status().is_available(),
        weight.dtype,
        k,
        n,
        m,
    ) {
        crate::metal::checks::check_disjoint(
            "qwen4exp_prefill_tensor",
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
        crate::half_telemetry::record(
            ctx,
            enc,
            crate::half_telemetry::SITE_QWEN4EXP_TENSOR_INPUT,
            u32::MAX,
            x,
        );
        if flash_next_dense_wide_enabled() {
            crate::metal::encode_mat_mat_tensor_f32_flash_next_wide(
                ctx, enc, weight, x, y, k, n, m,
            )?;
        } else {
            crate::metal::encode_mat_mat_tensor_f32_narrow(ctx, enc, weight, x, y, k, n, m)?;
        }
        Ok(())
    } else {
        super::encode_mat_mat_dispatch_without_fewrow(ctx, enc, weight, x, y, k, n, m)
    }
}

#[test]
fn flash_next_prefill_tensor_gate_preserves_fallbacks() {
    for (dtype, shapes) in [
        (
            GgmlType::Q4_K,
            &[
                (2560, 640),
                (2560, 6144),
                (2560, 10240),
                (2560, 12288),
                (6144, 2560),
            ][..],
        ),
        (
            GgmlType::Q6_K,
            &[
                (2560, 512),
                (2560, 640),
                (2560, 6144),
                (2560, 10240),
                (2560, 12288),
                (6144, 2560),
            ][..],
        ),
        (
            GgmlType::IQ4_XS,
            &[
                (2560, 512),
                (2560, 640),
                (2560, 6144),
                (2560, 10240),
                (2560, 12288),
            ][..],
        ),
        (GgmlType::IQ4_NL, &[(640, 2560)][..]),
    ] {
        for &(k, n) in shapes {
            assert!(flash_next_eligible(true, true, dtype, k, n, 2048));
            assert!(!flash_next_eligible(false, true, dtype, k, n, 2048));
            assert!(!flash_next_eligible(true, false, dtype, k, n, 2048));
            for m in [512, 513, 769, 1009, 1024, 1536, 2047] {
                assert_eq!(
                    flash_next_eligible(true, true, dtype, k, n, m),
                    crate::qwen4exp_prefill_tensor::rows_qualified(m)
                );
            }
            for m in [0, 1, 16, 127, 128, 511, 2049, 4096] {
                assert!(!flash_next_eligible(true, true, dtype, k, n, m));
            }
        }
    }
    for dtype in [
        GgmlType::F32,
        GgmlType::F16,
        GgmlType::BF16,
        GgmlType::Q5_K,
        GgmlType::Q8_0,
        GgmlType::IQ3_S,
    ] {
        assert!(!flash_next_eligible(true, true, dtype, 2560, 6144, 2048));
    }
    for (k, n) in [
        (640, 512),
        (2048, 6144),
        (2560, 2560),
        (2560, 256),
        (6144, 512),
        (2560, 248320),
    ] {
        assert!(!flash_next_eligible(true, true, GgmlType::Q6_K, k, n, 2048));
    }
}

/// Selection proof for diagnostic callers. Availability is the context's
/// Apple10 + Metal4/library gate, not a device-name inference.
pub fn qwen_prefill_tensor_selected(
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
pub fn encode_qwen_prefill_mat_mat(
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
    if qwen_prefill_tensor_selected(ctx, arch, weight.dtype, k, n, m) {
        crate::metal::checks::check_disjoint(
            "qwen_prefill_tensor",
            y,
            &[(weight, "weight"), (x, "input")],
        )?;
        if weight.dtype != GgmlType::Q8_0 && k > n {
            crate::half_telemetry::record(
                ctx,
                enc,
                crate::half_telemetry::SITE_DENSE_DOWN,
                u32::MAX,
                x,
            );
        }
        if weight.dtype == GgmlType::Q8_0 {
            crate::metal::encode_mat_mat_tensor_f32_wide(ctx, enc, weight, x, y, k, n, m)?;
        } else {
            crate::metal::encode_mat_mat_tensor_f32_narrow(ctx, enc, weight, x, y, k, n, m)?;
        }
        Ok(())
    } else {
        encode_mat_mat_dispatch(ctx, enc, weight, x, y, k, n, m)
    }
}

fn ffn128_shape(dtype: GgmlType, k: usize, n: usize, m: usize) -> bool {
    matches!(
        (dtype, k, n),
        (GgmlType::Q4_K, 5120, 17408) | (GgmlType::Q4_K | GgmlType::Q6_K, 17408, 5120)
    ) && (m == 1024 || (short_append_tensor_enabled() && (128..=2048).contains(&m)))
}

/// Only the measured 27B FFN shapes select the 128-row device-half tile.
pub fn qwen_ffn128_selected(
    ctx: &MetalContext,
    arch: &Arch,
    dtype: GgmlType,
    k: usize,
    n: usize,
    m: usize,
) -> bool {
    ffn128_enabled()
        && *arch == QWEN3_27B
        && ffn128_shape(dtype, k, n, m)
        && qwen_prefill_tensor_selected(ctx, arch, dtype, k, n, m)
}

/// FFN-only entry. The caller lends scratch whose previous contents are dead:
/// inner before gate/up, up after SiLU for down. Reuse adds no allocated bytes
/// to the prefill scratch plan. Source activations already entered half in B.
#[allow(clippy::too_many_arguments)]
pub fn encode_qwen_ffn_prefill_mat_mat(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    arch: &Arch,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    dead_scratch: &MetalTensor,
    k: usize,
    n: usize,
    m: usize,
) -> Result<(), MfError> {
    if !qwen_ffn128_selected(ctx, arch, weight.dtype, k, n, m) {
        return encode_qwen_prefill_mat_mat(ctx, enc, arch, weight, x, y, k, n, m);
    }
    use crate::metal::checks::{
        bad_shape, check_disjoint, check_physical, check_tensor, require_serial,
    };
    const K: &str = "qwen_ffn128";
    // Dimensions are the three exact, bounded tuples above. Preflight every
    // binding and PSO before the first telemetry/conversion dispatch.
    require_serial(K, enc)?;
    check_tensor(
        K,
        weight,
        weight.dtype,
        &[k as u64, n as u64],
        false,
        "weight",
    )?;
    check_physical(K, x, 4, false, "input")?;
    check_physical(K, y, 4, true, "output")?;
    check_physical(K, dead_scratch, 16, true, "dead scratch")?;
    if x.dtype != GgmlType::F32
        || x.n_elements() != (m * k) as u64
        || y.dtype != GgmlType::F32
        || y.n_elements() != (m * n) as u64
        || dead_scratch.dtype != GgmlType::F32
        || dead_scratch.n_elements() < (m * k / 2) as u64
    {
        return Err(bad_shape(K, "F32 input/output or dead-scratch capacity mismatch").into());
    }
    // An exact-byte view preserves provenance and backing bounds while lending
    // only the required prefix as F16. No host write or new Metal buffer.
    let mut half = dead_scratch.view_subrange(0, vec![(m * k / 2) as u64]);
    half.dtype = GgmlType::F16;
    half.shape = vec![(m * k) as u64];
    check_disjoint(K, &half, &[(weight, "weight"), (x, "input"), (y, "output")])?;
    check_disjoint(K, y, &[(weight, "weight"), (x, "input")])?;
    let name = match weight.dtype {
        GgmlType::Q4_K => "kernel_qwen_ffn_q4_K_f16act_tensor128",
        GgmlType::Q6_K => "kernel_qwen_ffn_q6_K_f16act_tensor128",
        _ => unreachable!("shape gate"),
    };
    let pso = ctx.tensor_pipeline(name)?;
    ctx.pipeline("kernel_scatter_offset_f32_to_f16")?;
    if k > n {
        crate::half_telemetry::record(
            ctx,
            enc,
            crate::half_telemetry::SITE_DENSE_DOWN,
            u32::MAX,
            x,
        );
    }
    crate::metal::encode_scatter_offset_f32_to_f16(ctx, enc, x, &half, 0, m * k)?;
    enc.set_pipeline(&pso);
    let block_bytes = if weight.dtype == GgmlType::Q4_K {
        144
    } else {
        210
    };
    enc.set_bytes(
        0,
        &[m as u32, n as u32, k as u32, (k / 256 * block_bytes) as u32],
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, &half);
    enc.set_tensor(3, y);
    enc.dispatch(
        objc2_metal::MTLSize {
            width: m.div_ceil(128),
            height: n / 64,
            depth: 1,
        },
        objc2_metal::MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
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

    #[test]
    fn a3b_expert_prefill_tensor_gate_preserves_fallbacks() {
        for down in [GgmlType::Q5_K, GgmlType::Q6_K] {
            assert!(expert_eligible(
                true,
                true,
                &A3B_ARCH,
                GgmlType::Q4_K,
                GgmlType::Q4_K,
                down,
                2048
            ));
            assert!(!expert_eligible(
                false,
                true,
                &A3B_ARCH,
                GgmlType::Q4_K,
                GgmlType::Q4_K,
                down,
                2048
            ));
            assert!(!expert_eligible(
                true,
                false,
                &A3B_ARCH,
                GgmlType::Q4_K,
                GgmlType::Q4_K,
                down,
                2048
            ));
            for m in [1, 127, 128, 512, 1024, 2047, 2049, 4096] {
                assert!(!expert_eligible(
                    true,
                    true,
                    &A3B_ARCH,
                    GgmlType::Q4_K,
                    GgmlType::Q4_K,
                    down,
                    m
                ));
            }
            for (g, u) in [
                (GgmlType::Q5_K, GgmlType::Q4_K),
                (GgmlType::Q4_K, GgmlType::Q8_0),
            ] {
                assert!(!expert_eligible(true, true, &A3B_ARCH, g, u, down, 2048));
            }
        }
        assert!(!expert_eligible(
            true,
            true,
            &A3B_ARCH,
            GgmlType::Q4_K,
            GgmlType::Q4_K,
            GgmlType::Q8_0,
            2048
        ));
        let mut other = A3B_ARCH;
        other.n_layer = 38;
        assert!(!expert_eligible(
            true,
            true,
            &other,
            GgmlType::Q4_K,
            GgmlType::Q4_K,
            GgmlType::Q5_K,
            2048
        ));
    }

    #[test]
    fn a3b_prefill_tensor_gate_preserves_fallbacks() {
        for (k, n) in [
            (2048, 8192),
            (2048, 4096),
            (4096, 2048),
            (2048, 512),
            (512, 2048),
        ] {
            for m in [128, 129, 512, 2048] {
                assert!(eligible(true, true, &A3B_ARCH, GgmlType::Q8_0, k, n, m));
            }
            for m in [1, 2, 31, 127] {
                assert!(!eligible(true, true, &A3B_ARCH, GgmlType::Q8_0, k, n, m));
            }
            assert!(!eligible(
                false,
                true,
                &A3B_ARCH,
                GgmlType::Q8_0,
                k,
                n,
                2048
            ));
            assert!(!eligible(
                true,
                false,
                &A3B_ARCH,
                GgmlType::Q8_0,
                k,
                n,
                2048
            ));
        }
        for dtype in [
            GgmlType::F32,
            GgmlType::F16,
            GgmlType::Q4_K,
            GgmlType::Q5_K,
            GgmlType::Q6_K,
        ] {
            assert!(!eligible(true, true, &A3B_ARCH, dtype, 2048, 8192, 2048));
        }
        for (k, n) in [(0, 8192), (2049, 8192), (2048, 64), (512, 248_320)] {
            assert!(!eligible(true, true, &A3B_ARCH, GgmlType::Q8_0, k, n, 2048));
        }
        let mut other = A3B_ARCH;
        for variant in 0..5 {
            match variant {
                0 => other.kind = ArchKind::Dense,
                1 => other.n_layer = 48,
                2 => other.expert_count = 512,
                3 => other.expert_used_count = 4,
                _ => other.mtp_n_hidden_layers = 1,
            }
            assert!(!eligible(
                true,
                true,
                &other,
                GgmlType::Q8_0,
                2048,
                8192,
                2048
            ));
            other = A3B_ARCH;
        }
        assert!(!eligible(
            true,
            true,
            &crate::model::QWEN3_27B,
            GgmlType::Q8_0,
            2048,
            8192,
            2048
        ));
    }

    fn random(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    fn weights(dtype: GgmlType, k: usize, n: usize) -> (Vec<u8>, Vec<f32>) {
        let (block_elems, block_bytes) = match dtype {
            GgmlType::Q4_K => (256, 144),
            GgmlType::Q5_K => (256, 176),
            GgmlType::Q6_K => (256, 210),
            GgmlType::Q8_0 => (32, 34),
            _ => unreachable!(),
        };
        let mut packed = vec![0; k / block_elems * n * block_bytes];
        let mut rng = 0x4e13_c9d2;
        for (ordinal, block) in packed.chunks_exact_mut(block_bytes).enumerate() {
            for byte in block.iter_mut() {
                *byte = random(&mut rng) as u8;
            }
            let d = half::f16::from_f32([0.0005, 0.003, 1.0 / 64.0][ordinal % 3]);
            if matches!(dtype, GgmlType::Q4_K | GgmlType::Q5_K) {
                block[..2].copy_from_slice(&d.to_le_bytes());
                block[2..4].copy_from_slice(&half::f16::from_f32(d.to_f32() * 0.5).to_le_bytes());
            } else if dtype == GgmlType::Q8_0 {
                block[..2].copy_from_slice(&d.to_le_bytes());
                block[2..5].copy_from_slice(&[128, 0, 127]);
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

    struct Guarded {
        backing: MetalTensor,
        tensor: MetalTensor,
        len: usize,
    }

    impl Guarded {
        const GUARD: usize = 16;
        const CANARY: u32 = 0x45ab_cdef;
        fn new(ctx: &MetalContext, len: usize) -> Self {
            let mut data = vec![Self::CANARY; len + 2 * Self::GUARD];
            data[Self::GUARD..Self::GUARD + len].fill(f32::NAN.to_bits());
            let backing = MetalTensor::from_bytes(
                ctx,
                bytemuck::cast_slice(&data),
                vec![data.len() as u64],
                GgmlType::F32,
            )
            .unwrap();
            let tensor = backing.view_subrange(Self::GUARD as u64, vec![len as u64]);
            Self {
                backing,
                tensor,
                len,
            }
        }
        fn values(&self) -> &[f32] {
            let words = unsafe {
                std::slice::from_raw_parts(
                    self.backing.buffer.contents().as_ptr().cast::<u32>(),
                    self.len + 2 * Self::GUARD,
                )
            };
            assert!(
                words[..Self::GUARD]
                    .iter()
                    .chain(&words[Self::GUARD + self.len..])
                    .all(|&v| v == Self::CANARY)
            );
            let values = unsafe {
                std::slice::from_raw_parts(words.as_ptr().add(Self::GUARD).cast::<f32>(), self.len)
            };
            assert!(
                values.iter().all(|v| v.is_finite()),
                "nonfinite or unwritten output"
            );
            values
        }
    }

    fn sampled_axis(n: usize) -> Vec<usize> {
        (0..32.min(n))
            .map(|i| i * (n - 1) / (32.min(n) - 1).max(1))
            .collect()
    }

    #[test]
    fn a3b_dense_prefill_tensor_gpu_regression() {
        let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        for (k, n) in [
            (2048, 8192),
            (2048, 4096),
            (4096, 2048),
            (2048, 512),
            (512, 2048),
        ] {
            let (packed, decoded) = weights(GgmlType::Q8_0, k, n);
            let w =
                MetalTensor::from_bytes(&ctx, &packed, vec![k as u64, n as u64], GgmlType::Q8_0)
                    .unwrap();
            let mut geometries = vec![(1, false), (127, false), (128, false), (129, false)];
            if n == 512 && k == 2048 {
                geometries.extend([(2048, false), (128, true)]);
            }
            for (m, other_model) in geometries {
                let mut arch = A3B_ARCH;
                if other_model {
                    arch.n_layer = 38;
                }
                let selected = qwen_prefill_tensor_selected(&ctx, &arch, w.dtype, k, n, m);
                let cells: Vec<_> = sampled_axis(m)
                    .into_iter()
                    .flat_map(|row| sampled_axis(n).into_iter().map(move |col| (row, col)))
                    .collect();
                for amplitude in [2f32.powi(-24), 1.0, 2048.0, 65504.0] {
                    let mut rng = 0x5a31_89ef;
                    let mut input: Vec<f32> = (0..m * k)
                        .map(|_| {
                            ((random(&mut rng) >> 40) as f32 / (1u64 << 23) as f32 - 1.0)
                                * amplitude
                        })
                        .collect();
                    input[..4].copy_from_slice(&[amplitude, -amplitude, 0.0, 2f32.powi(-24)]);
                    let reference: Vec<f64> = cells
                        .iter()
                        .map(|&(row, col)| {
                            input[row * k..(row + 1) * k]
                                .iter()
                                .zip(&decoded[col * k..(col + 1) * k])
                                .map(|(&x, &w)| f64::from(x) * f64::from(w))
                                .sum()
                        })
                        .collect();
                    let x = MetalTensor::from_bytes(
                        &ctx,
                        bytemuck::cast_slice(&input),
                        vec![k as u64, m as u64],
                        GgmlType::F32,
                    )
                    .unwrap();
                    let a = Guarded::new(&ctx, m * n);
                    let b = Guarded::new(&ctx, m * n);
                    for (production, out) in [(false, &a), (true, &b)] {
                        dispatch_census_begin();
                        let cmd = ctx.queue.commandBuffer().unwrap();
                        let enc = KernelEncoder::begin(&cmd);
                        if production {
                            encode_qwen_prefill_mat_mat(
                                &ctx,
                                &enc,
                                &arch,
                                &w,
                                &x,
                                &out.tensor,
                                k,
                                n,
                                m,
                            )
                            .unwrap();
                        } else {
                            encode_mat_mat_dispatch(&ctx, &enc, &w, &x, &out.tensor, k, n, m)
                                .unwrap();
                        }
                        enc.end();
                        cmd.commit();
                        wait_completed(&cmd).unwrap();
                        let dispatches = dispatch_census_take();
                        if production && selected {
                            assert!(
                                dispatches
                                    .iter()
                                    .any(|d| d.kernel == "kernel_mat_mat_q8_0_f32_tensor_wide")
                            );
                        }
                    }
                    let (av, bv) = (a.values(), b.values());
                    let sample = |values: &[f32]| {
                        cells
                            .iter()
                            .map(|&(r, c)| values[r * n + c])
                            .collect::<Vec<_>>()
                    };
                    let incumbent = errors(&sample(av), &reference);
                    let actual = errors(&sample(bv), &reference);
                    assert!(
                        actual.0 <= 1.5 * incumbent.0 && actual.1 <= 1.5 * incumbent.1,
                        "A3B M{m}/N{n}/K{k} amplitude={amplitude}: actual={actual:?} incumbent={incumbent:?}"
                    );
                    if !selected {
                        assert!(av.iter().zip(bv).all(|(a, b)| a.to_bits() == b.to_bits()));
                    }
                }
            }
        }
    }

    fn expert_bank(
        ctx: &MetalContext,
        dtype: GgmlType,
        k: usize,
        n: usize,
        up: bool,
    ) -> (MetalTensor, Vec<u8>) {
        let (mut one, _) = weights(dtype, k, n);
        if up {
            assert_eq!(dtype, GgmlType::Q4_K);
            for block in one.chunks_exact_mut(144) {
                for byte in &mut block[16..] {
                    *byte ^= 0x5a;
                }
            }
        }
        let bytes = match dtype {
            GgmlType::Q4_K => 144,
            GgmlType::Q5_K => 176,
            GgmlType::Q6_K => 210,
            _ => unreachable!(),
        };
        let mut bank = Vec::with_capacity(one.len() * 256);
        for expert in 0..256 {
            let start = bank.len();
            bank.extend_from_slice(&one);
            for block in bank[start..].chunks_exact_mut(bytes) {
                let offsets: &[usize] = if dtype == GgmlType::Q6_K {
                    &[208]
                } else {
                    &[0, 2]
                };
                for &i in offsets {
                    let d = half::f16::from_le_bytes([block[i], block[i + 1]]).to_f32();
                    block[i..i + 2].copy_from_slice(
                        &half::f16::from_f32(d * (0.5 + expert as f32 / 256.0)).to_le_bytes(),
                    );
                }
            }
        }
        let tensor =
            MetalTensor::from_bytes(ctx, &bank, vec![k as u64, n as u64, 256], dtype).unwrap();
        (tensor, bank)
    }

    fn decoded_row(
        bank: &[u8],
        dtype: GgmlType,
        k: usize,
        n: usize,
        expert: usize,
        col: usize,
    ) -> Vec<f32> {
        let bytes = match dtype {
            GgmlType::Q4_K => 144,
            GgmlType::Q5_K => 176,
            GgmlType::Q6_K => 210,
            _ => unreachable!(),
        };
        let stride = k / 256 * bytes;
        let start = (expert * n + col) * stride;
        let desc = TensorDesc {
            name: "expert-regression-row".into(),
            shape: vec![k as u64],
            dtype,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: stride as u64,
        };
        crate::codec::dequant_to_f32(&desc, &bank[start..start + stride]).unwrap()
    }

    #[test]
    fn a3b_grouped_expert_tensor_gpu_regression() {
        let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        if !ctx.tensor_ops_status().is_available() {
            return;
        }
        const M: usize = 2048;
        const H: usize = 2048;
        const F: usize = 512;
        const E: usize = 256;
        const TOPK: usize = 8;
        let mut counts = vec![0i32; E];
        for e in [0, 1, 17, 63, 64, 127, 255] {
            counts[e] = M as i32;
        }
        for (e, count) in [
            (2, 1),
            (3, 7),
            (4, 15),
            (5, 16),
            (6, 31),
            (7, 47),
            (8, 48),
            (9, 63),
            (10, 128),
            (11, 256),
            (12, 1436),
        ] {
            counts[e] = count;
        }
        assert_eq!(counts.iter().sum::<i32>(), (M * TOPK) as i32);
        let mut ids = vec![-1i32; E * M];
        let mut ordinal = 0;
        let mut probes = Vec::new();
        for (e, &count) in counts.iter().enumerate() {
            for entry in 0..count as usize {
                ids[e * M + entry] = ((ordinal % M) * TOPK + ordinal / M) as i32;
                ordinal += 1;
            }
            if count > 0 {
                for entry in [0, count as usize - 1] {
                    probes.push((e, ids[e * M + entry] as usize));
                }
            }
        }
        let ct = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&counts),
            vec![E as u64],
            GgmlType::I32,
        )
        .unwrap();
        let it = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&ids),
            vec![(E * M) as u64],
            GgmlType::I32,
        )
        .unwrap();
        let mut rng = 0x5a31_89ef;
        let x: Vec<f32> = (0..M * H)
            .map(|_| ((random(&mut rng) >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * 0.02)
            .collect();
        let dx: Vec<f32> = (0..M * TOPK * F)
            .map(|_| (random(&mut rng) >> 40) as f32 / (1u64 << 23) as f32 - 1.0)
            .collect();
        let xt = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![(M * H) as u64],
            GgmlType::F32,
        )
        .unwrap();
        let dt = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&dx),
            vec![(M * TOPK * F) as u64],
            GgmlType::F32,
        )
        .unwrap();
        let (gate, gb) = expert_bank(&ctx, GgmlType::Q4_K, H, F, false);
        let (up, ub) = expert_bank(&ctx, GgmlType::Q4_K, H, F, true);
        for dtype in [GgmlType::Q5_K, GgmlType::Q6_K] {
            let (down, db) = expert_bank(&ctx, dtype, F, H, false);
            let moe = super::super::MetalMoeFfn {
                gate_inp: MetalTensor::zeros_f32(&ctx, vec![1]).unwrap(),
                gate_exps: gate.clone(),
                up_exps: up.clone(),
                down_exps: down,
                gate_inp_shexp: MetalTensor::zeros_f32(&ctx, vec![1]).unwrap(),
                gate_inp_cpu: vec![],
                gate_inp_shexp_cpu: vec![],
            };
            let dot = |a: &[f32], b: &[f32]| {
                a.iter()
                    .zip(b)
                    .map(|(&x, &w)| f64::from(x) * f64::from(w))
                    .sum::<f64>()
            };
            let mut gate_ref = Vec::new();
            let mut down_ref = Vec::new();
            for &(e, slot) in &probes {
                let row = &x[(slot / TOPK) * H..(slot / TOPK + 1) * H];
                for col in [0, 63, F - 1] {
                    let g = dot(row, &decoded_row(&gb, GgmlType::Q4_K, H, F, e, col));
                    let u = dot(row, &decoded_row(&ub, GgmlType::Q4_K, H, F, e, col));
                    gate_ref.push(g / (1.0 + (-g).exp()) * u);
                }
                for col in [0, 63, H - 1] {
                    down_ref.push(dot(
                        &dx[slot * F..(slot + 1) * F],
                        &decoded_row(&db, dtype, F, H, e, col),
                    ));
                }
            }
            let mut results = Vec::new();
            for production in [false, true] {
                let tensor = production
                    && qwen_a3b_expert_prefill_tensor_selected(
                        &ctx,
                        &A3B_ARCH,
                        GgmlType::Q4_K,
                        GgmlType::Q4_K,
                        dtype,
                        M,
                    );
                let inner = Guarded::new(&ctx, M * TOPK * F);
                let out = Guarded::new(&ctx, M * TOPK * H);
                dispatch_census_begin();
                let cmd = ctx.queue.commandBuffer().unwrap();
                let enc = KernelEncoder::begin(&cmd);
                crate::metal_dflash::encode_prefill_moe_grouped_swiglu(
                    &ctx,
                    &enc,
                    &moe,
                    &xt,
                    &ct,
                    &it,
                    &inner.tensor,
                    H,
                    F,
                    E,
                    TOPK,
                    M,
                    false,
                    Some(48),
                    tensor,
                )
                .unwrap();
                crate::metal_dflash::encode_prefill_moe_grouped_down(
                    &ctx,
                    &enc,
                    &moe.down_exps,
                    &dt,
                    &ct,
                    &it,
                    &out.tensor,
                    F,
                    H,
                    E,
                    M,
                    0,
                    tensor,
                )
                .unwrap();
                enc.end();
                cmd.commit();
                wait_completed(&cmd).unwrap();
                let dispatches = dispatch_census_take();
                if tensor {
                    assert_eq!(
                        dispatches
                            .iter()
                            .filter(|d| d.kernel.ends_with("grouped_slots_tensor"))
                            .count(),
                        3
                    );
                }
                let (g, d) = (inner.values(), out.values());
                let gs: Vec<_> = probes
                    .iter()
                    .flat_map(|&(_, slot)| [0, 63, F - 1].map(|col| g[slot * F + col]))
                    .collect();
                let ds: Vec<_> = probes
                    .iter()
                    .flat_map(|&(_, slot)| [0, 63, H - 1].map(|col| d[slot * H + col]))
                    .collect();
                results.push((errors(&gs, &gate_ref), errors(&ds, &down_ref)));
            }
            for (a, b) in [(results[0].0, results[1].0), (results[0].1, results[1].1)] {
                assert!(
                    b.0 <= 1.5 * a.0 && b.1 <= 1.5 * a.1,
                    "{dtype:?}: incumbent={a:?}, tensor={b:?}"
                );
            }
        }
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
                    let selected = qwen_prefill_tensor_selected(&ctx, &arch, dtype, k, n, m);
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
                            encode_qwen_prefill_mat_mat(&ctx, &enc, &arch, &w, &x, &y, k, n, m)
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

    #[test]
    fn qwen_ffn128_shape_gate() {
        for dtype in [GgmlType::Q4_K, GgmlType::Q6_K] {
            assert!(ffn128_shape(dtype, 17408, 5120, 1024));
            for m in [1, 127, 128, 512, 1023, 1025, 2048] {
                assert!(!ffn128_shape(dtype, 17408, 5120, m));
            }
        }
        assert!(ffn128_shape(GgmlType::Q4_K, 5120, 17408, 1024));
        assert!(!ffn128_shape(GgmlType::Q6_K, 5120, 17408, 1024));
        assert!(!ffn128_shape(GgmlType::Q8_0, 2048, 8192, 1024));
    }

    #[test]
    fn qwen_ffn128_gpu_regression() {
        let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        let mut cases = 0;
        let mut routed = 0;
        for (dtype, k, n) in [
            (GgmlType::Q4_K, 5120, 17408),
            (GgmlType::Q4_K, 17408, 5120),
            (GgmlType::Q6_K, 17408, 5120),
        ] {
            let (packed, decoded) = weights(dtype, k, n);
            let w =
                MetalTensor::from_bytes(&ctx, &packed, vec![k as u64, n as u64], dtype).unwrap();
            for m in [1024, 512, 1023] {
                let cells: Vec<_> = sampled_axis(m)
                    .into_iter()
                    .flat_map(|r| sampled_axis(n).into_iter().map(move |c| (r, c)))
                    .collect();
                let selected = qwen_ffn128_selected(&ctx, &QWEN3_27B, dtype, k, n, m);
                assert_eq!(
                    selected,
                    ctx.tensor_ops_status().is_available()
                        && enabled()
                        && ffn128_enabled()
                        && m == 1024
                );
                assert!(!qwen_ffn128_selected(&ctx, &A3B_ARCH, dtype, k, n, m));
                for amplitude in [2f32.powi(-24), 1.0, 2048.0, 65504.0] {
                    let mut rng = 0x5a31_89ef;
                    let mut input: Vec<f32> = (0..m * k)
                        .map(|_| {
                            ((random(&mut rng) >> 40) as f32 / (1u64 << 23) as f32 - 1.0)
                                * amplitude
                        })
                        .collect();
                    input[..4].copy_from_slice(&[amplitude, -amplitude, 0.0, 2f32.powi(-24)]);
                    let reference: Vec<f64> = cells
                        .iter()
                        .map(|&(r, c)| {
                            input[r * k..(r + 1) * k]
                                .iter()
                                .zip(&decoded[c * k..(c + 1) * k])
                                .map(|(&x, &w)| f64::from(x) * f64::from(w))
                                .sum()
                        })
                        .collect();
                    let x = MetalTensor::from_bytes(
                        &ctx,
                        bytemuck::cast_slice(&input),
                        vec![(m * k) as u64],
                        GgmlType::F32,
                    )
                    .unwrap();
                    let spare = Guarded::new(&ctx, m * k / 2);
                    let a = Guarded::new(&ctx, m * n);
                    let b = Guarded::new(&ctx, m * n);
                    // Deliberately follow B in the same serial encoder: this
                    // also exercises pipeline/TG binding state and ordering.
                    dispatch_census_begin();
                    let cmd = ctx.queue.commandBuffer().unwrap();
                    let enc = KernelEncoder::begin(&cmd);
                    encode_qwen_prefill_mat_mat(&ctx, &enc, &QWEN3_27B, &w, &x, &a.tensor, k, n, m)
                        .unwrap();
                    encode_qwen_ffn_prefill_mat_mat(
                        &ctx,
                        &enc,
                        &QWEN3_27B,
                        &w,
                        &x,
                        &b.tensor,
                        &spare.tensor,
                        k,
                        n,
                        m,
                    )
                    .unwrap();
                    enc.end();
                    cmd.commit();
                    wait_completed(&cmd).unwrap();
                    let dispatches = dispatch_census_take();
                    assert_eq!(
                        dispatches
                            .iter()
                            .any(|d| d.kernel.ends_with("f16act_tensor128")),
                        selected
                    );
                    let (av, bv) = (a.values(), b.values());
                    let sample =
                        |v: &[f32]| cells.iter().map(|&(r, c)| v[r * n + c]).collect::<Vec<_>>();
                    let ae = errors(&sample(av), &reference);
                    let be = errors(&sample(bv), &reference);
                    assert!(
                        be.0 <= 1.5 * ae.0 && be.1 <= 1.5 * ae.1,
                        "{dtype:?} M{m}: B={ae:?} C={be:?}"
                    );
                    if !selected {
                        assert!(av.iter().zip(bv).all(|(a, b)| a.to_bits() == b.to_bits()));
                    }
                    let words = unsafe {
                        std::slice::from_raw_parts(
                            spare.backing.buffer.contents().as_ptr().cast::<u32>(),
                            spare.len + 2 * Guarded::GUARD,
                        )
                    };
                    assert!(
                        words[..Guarded::GUARD]
                            .iter()
                            .chain(&words[Guarded::GUARD + spare.len..])
                            .all(|&v| v == Guarded::CANARY)
                    );
                    if selected {
                        let half = unsafe {
                            std::slice::from_raw_parts(
                                words.as_ptr().add(Guarded::GUARD).cast::<u16>(),
                                m * k,
                            )
                        };
                        for index in (0..m * k).step_by((m * k / 1024).max(1)).chain([m * k - 1]) {
                            let expected = half::f16::from_f32(input[index]).to_bits();
                            // Metal may flush half subnormals. The F64/B gate
                            // above still checks the resulting arithmetic.
                            if expected & 0x7c00 == 0 {
                                assert!(half[index] == expected || half[index] & 0x7fff == 0);
                            } else {
                                assert_eq!(half[index], expected);
                            }
                        }
                        // Refuse live-input/output scratch aliases before any
                        // encoder mutation, including the conversion dispatch.
                        for bad in [&x, &b.tensor] {
                            dispatch_census_begin();
                            let rejected = ctx.queue.commandBuffer().unwrap();
                            let e = KernelEncoder::begin(&rejected);
                            assert!(
                                encode_qwen_ffn_prefill_mat_mat(
                                    &ctx, &e, &QWEN3_27B, &w, &x, &b.tensor, bad, k, n, m
                                )
                                .is_err()
                            );
                            e.end();
                            assert!(dispatch_census_take().is_empty());
                        }
                        dispatch_census_begin();
                        let rejected = ctx.queue.commandBuffer().unwrap();
                        let concurrent = KernelEncoder::begin_concurrent(&rejected);
                        assert!(
                            encode_qwen_ffn_prefill_mat_mat(
                                &ctx,
                                &concurrent,
                                &QWEN3_27B,
                                &w,
                                &x,
                                &b.tensor,
                                &spare.tensor,
                                k,
                                n,
                                m
                            )
                            .is_err()
                        );
                        concurrent.end();
                        assert!(dispatch_census_take().is_empty());
                    } else {
                        assert!(
                            words[Guarded::GUARD..Guarded::GUARD + spare.len]
                                .iter()
                                .all(|&v| v == f32::NAN.to_bits())
                        );
                    }
                    cases += 1;
                    routed += usize::from(selected);
                }
            }
        }
        eprintln!(
            "qwen_ffn128: {cases} F64 cases passed, {routed} selected, {} bitwise fallbacks; reused-scratch guards and serial ordering checked",
            cases - routed
        );
    }
}
