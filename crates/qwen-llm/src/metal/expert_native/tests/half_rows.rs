use super::*;

fn half_operand(v: f32) -> f32 {
    half::f16::from_f32(v).to_f32()
}

// Deliberately scalar F32 accumulation, unlike both native SIMD and grouped MMA.
fn half_dot(w: &[f32], x: &[f32]) -> f32 {
    assert_eq!(w.len(), x.len());
    w.iter().zip(x).fold(0.0, |sum, (&w, &x)| {
        half_operand(w).mul_add(half_operand(x), sum)
    })
}

fn swiglu(g: f32, u: f32, clamp: f32) -> f32 {
    let g = g.min(clamp);
    g / (1.0 + (-g).exp()) * u.clamp(-clamp, clamp)
}

#[derive(Clone, Copy, Debug)]
enum Reduction {
    Native,
    Grouped,
    Scalar,
}

fn gamma(steps: usize, unit: f64) -> f64 {
    let nu = steps as f64 * unit;
    assert!(nu < 1.0);
    nu / (1.0 - nu)
}

#[derive(Clone, Copy, Debug, Default)]
struct Bounded {
    value: f64,
    error: f64,
}

impl Bounded {
    fn check(self, label: &str, index: usize, actual: f32) -> f64 {
        let error = (f64::from(actual) - self.value).abs();
        assert!(
            actual.is_finite()
                && self.value.is_finite()
                && self.error.is_finite()
                && self.error >= 0.0
        );
        assert!(
            error <= self.error,
            "{label} index {index}: actual={actual:e} F64={:e} abs_error={error:e} bound={:e}",
            self.value,
            self.error
        );
        if self.error == 0.0 {
            0.0
        } else {
            error / self.error
        }
    }
}

struct HalfDot {
    value: f64,
    l1: f64,
    k: usize,
}

impl HalfDot {
    fn new(w: &[f32], x: &[f32]) -> Self {
        assert_eq!(w.len(), x.len());
        let mut result = Self {
            value: 0.0,
            l1: 0.0,
            k: w.len(),
        };
        for (&w, &x) in w.iter().zip(x) {
            let w = f64::from(half_operand(w));
            let x = f64::from(half_operand(x));
            assert!(w.is_finite() && x.is_finite());
            // A finite half*half product is exact even in F32. The independent
            // GGML decoder supplies w; no shader decode/reduction is emulated.
            let product = w * x;
            result.value += product;
            result.l1 += product.abs();
        }
        result
    }

    fn bound(&self, reduction: Reduction) -> Bounded {
        let depth = match reduction {
            // At most 16 FMAs per 512-element stripe, then at most 31 adds.
            // Do not assume an undocumented balanced simd_sum tree.
            Reduction::Native => 16 * self.k.div_ceil(512) + 31,
            // Bound any F32 summation order of K exact half products, without
            // assuming the internal 8-wide MMA accumulator reduction tree.
            Reduction::Grouped | Reduction::Scalar => self.k,
        };
        // Metal permits RTZ as well as RNE arithmetic: use epsilon, not eps/2.
        // Include F64 reference-sum error and upward-enclose its measured L1.
        let reference_gamma = gamma(self.k, f64::EPSILON);
        let l1_upper = self.l1 / (1.0 - reference_gamma);
        Bounded {
            value: self.value,
            error: (gamma(depth, f64::from(f32::EPSILON)) + reference_gamma) * l1_upper,
        }
    }
}

fn silu64(g: f64) -> f64 {
    if g < 0.0 {
        let e = g.exp();
        g * e / (1.0 + e)
    } else {
        g / (1.0 + (-g).exp())
    }
}

fn swiglu_bound(g: Bounded, u: Bounded, clamp: f32) -> Bounded {
    let c = f64::from(clamp);
    let gc = g.value.min(c);
    let uc = u.value.clamp(-c, c);
    let glo = (g.value - g.error).min(c);
    let ghi = (g.value + g.error).min(c);
    let dg = (gc - glo).max(ghi - gc);
    let du = (uc - (u.value - u.error).clamp(-c, c)).max((u.value + u.error).clamp(-c, c) - uc);
    let s = silu64(gc);
    // |silu'(t)| <= 1 + 1/e: sigmoid <= 1 and
    // |t| sigmoid(t)(1-sigmoid(t)) <= |t| exp(-|t|) <= 1/e.
    // Clamps are nonexpansive; include the product cross term as well.
    let max_gate = glo.abs().max(ghi.abs());
    let global_slope = 1.0 + (-1.0f64).exp();
    // On a wholly negative interval, |silu'| <= (1+|t|) exp(t).
    // Preserve a tiny absolute bound for saturated negative gates rather than
    // letting a global Lipschitz bound admit visible nonzero outputs there.
    let slope = if ghi < 0.0 {
        global_slope.min((1.0 + max_gate) * ghi.exp())
    } else {
        global_slope
    };
    let ds = slope * dg;
    let propagated = uc.abs() * ds + s.abs() * du + ds * du;
    let max_output = (s.abs() + ds) * (uc.abs() + du);
    // MSL Numerical Compliance table 8.2 (fast math): exp(x) error is
    // <= 3+floor(|2*x|) ULP, division <= 2.5 ULP. Conservatively bound one
    // ULP by 2*epsilon*|value|, then include denominator addition and final
    // multiplication. This is separate from the cancellation-aware dot bound.
    let eps = f64::from(f32::EPSILON);
    let exp_relative = (3.0 + (2.0 * max_gate).floor()) * 2.0 * eps;
    assert!(exp_relative < 1.0);
    // exp(-g) contributes only sigmoid(-g) of the positive denominator.
    let exp_fraction = if glo >= 0.0 {
        let e = (-glo).exp();
        e / (1.0 + e)
    } else {
        1.0 / (1.0 + glo.exp())
    };
    let epilogue_relative =
        (1.0 + 5.0 * eps) * (1.0 + eps) / ((1.0 - exp_relative * exp_fraction) * (1.0 - eps)) - 1.0;
    // Outside the normal reciprocal/exp domain, allow only the vanishing
    // sigmoid tail (or its loss). Still reject nonfinite GPU results. This
    // explicitly covers the existing very-negative-gate fixtures, not a
    // general fast-math guarantee for overflow/underflow inputs.
    let tail = f64::from(f32::MIN_POSITIVE) * (1.0 + max_gate) * (1.0 + uc.abs() + du);
    Bounded {
        value: s * uc,
        error: propagated + max_output * (epilogue_relative + 32.0 * f64::EPSILON) + tail,
    }
}

#[test]
fn native_half_rows_cpu_cancellation_bounds() {
    // Pin the three grouped/native decode expressions, including scale/sign
    // evaluation order. Only the final destination cast may differ.
    let grouped = include_str!("../../../../../../kernels/quant_tiles.h");
    let native = include_str!("../../../../../../kernels/native_dequant_f32.h");
    let normalized = |source: &str, name: &str| {
        let body = &source[source.find(&format!("inline void {name}")).unwrap()..];
        body[..body.find("\n}").unwrap() + 2]
            .replace(name, "decode")
            .replace("half4x4", "float4x4")
            .replace("(half)", "(float)")
            .split_whitespace()
            .collect::<String>()
    };
    for dtype in ["iq2_s", "iq3_s", "iq4_xs"] {
        assert_eq!(
            normalized(grouped, &format!("qt_dequantize_{dtype}")),
            normalized(native, &format!("latent_dequantize_{dtype}"))
        );
    }
    let (k, m, expert, channel) = (768, 7, 1, 6);
    let x = values(k * 2, 9);
    let g = dequant_expert(
        &bank_bytes(GgmlType::IQ3_S, k, m, 3),
        GgmlType::IQ3_S,
        k,
        m,
        expert,
    );
    let u = dequant_expert(
        &bank_bytes(GgmlType::IQ3_S, k, m, 7),
        GgmlType::IQ3_S,
        k,
        m,
        expert,
    );
    let gd = HalfDot::new(&g[channel * k..][..k], &x[k..]);
    let ud = HalfDot::new(&u[channel * k..][..k], &x[k..]);
    assert!(ud.l1 / ud.value.abs() > 7000.0);
    // Recorded failure: neither GPU output is the exact oracle. Their pairwise
    // delta fails the old relative-output test despite legitimate dot errors.
    let (native_observed, grouped_observed) = (-2.1947422f32, -2.1950781f32);
    assert!((native_observed - grouped_observed).abs() > 8e-5 * grouped_observed.abs());
    swiglu_bound(
        gd.bound(Reduction::Native),
        ud.bound(Reduction::Native),
        100.0,
    )
    .check("recorded native", 20, native_observed);
    swiglu_bound(
        gd.bound(Reduction::Grouped),
        ud.bound(Reduction::Grouped),
        100.0,
    )
    .check("recorded grouped", 20, grouped_observed);
    for reduction in [Reduction::Native, Reduction::Grouped, Reduction::Scalar] {
        let b = swiglu_bound(gd.bound(reduction), ud.bound(reduction), 100.0);
        // Check the scalar CPU algorithm independently, too; it is not exact.
        let scalar = swiglu(
            half_dot(&g[channel * k..][..k], &x[k..]),
            half_dot(&u[channel * k..][..k], &x[k..]),
            100.0,
        );
        if matches!(reduction, Reduction::Scalar) {
            b.check("scalar", 20, scalar);
        }
        assert!((b.value + 2.0 * b.error - b.value).abs() > b.error);
        eprintln!(
            "{reduction:?}: gate={} gate_l1={} up={} up_l1={} F64={} bound={}",
            gd.value, gd.l1, ud.value, ud.l1, b.value, b.error
        );
    }
    // Exact zero products stay exact; cancellation does not add a fixed floor.
    assert_eq!(
        HalfDot::new(&[0.0; 256], &[1.0; 256])
            .bound(Reduction::Native)
            .error,
        0.0
    );
    // A one-product oracle must reject missing input OR coefficient narrowing.
    let (w, x) = (1.0003f32, 1.0003f32);
    let b = HalfDot::new(&[w], &[x]).bound(Reduction::Native);
    b.check("half basis", 0, half_operand(w) * half_operand(x));
    assert!((f64::from(w * half_operand(x)) - b.value).abs() > b.error);
    assert!((f64::from(half_operand(w) * x) - b.value).abs() > b.error);
    let saturated = swiglu_bound(
        Bounded {
            value: -100.0,
            error: 0.1,
        },
        Bounded {
            value: 4.0,
            error: 0.1,
        },
        1.0,
    );
    assert!(saturated.error < 1e-34);
    saturated.check("negative tail", 0, 0.0);
}

#[test]
fn native_half_rows_cpu_abi_capabilities_and_bounds() {
    assert_eq!(std::mem::size_of::<NativeArgs>(), 32);
    assert_eq!(std::mem::offset_of!(NativeArgs, mode), 24);
    assert_eq!(std::mem::offset_of!(NativeArgs, clamp), 28);
    assert_eq!(NATIVE_HALF_OPERANDS, 256);
    assert!(SOURCE.contains("EXPERT_NATIVE_HALF_OPERANDS = 1u << 8"));
    assert!(SOURCE.contains("EXPERT_NATIVE_PROJECTION_MASK = 3u"));
    for projection in 0..=2 {
        assert_eq!(NativeOperands::F32.mode(projection), projection);
        let mode = NativeOperands::Half.mode(projection);
        assert_eq!(mode & 3, projection);
        assert_eq!(mode & !3, NATIVE_HALF_OPERANDS);
    }
    for rows in 0..=17 {
        assert_eq!(
            check_half_rows("test", rows).is_ok(),
            (1..=8).contains(&rows)
        );
    }
    assert!(check_half_rows("test", usize::MAX).is_err());
    for raw in 0..=255 {
        let dtype = GgmlType::from_raw(raw);
        assert_eq!(
            all_slots_down_native_half_rows_supported(dtype),
            matches!(dtype, GgmlType::IQ3_S | GgmlType::IQ4_XS)
        );
        for other in (0..=255).map(GgmlType::from_raw) {
            assert_eq!(
                all_slots_gate_up_native_half_rows_supported(dtype, other),
                matches!(dtype, GgmlType::IQ2_S | GgmlType::IQ3_S)
                    && matches!(other, GgmlType::IQ2_S | GgmlType::IQ3_S)
            );
        }
    }
    for (k, m, e, top) in [
        (0, 7, 3, 6),
        (256, 0, 3, 6),
        (256, 7, 0, 6),
        (256, 7, 3, 0),
        (256, 7, 3, 17),
        (usize::MAX, 7, 3, 6),
    ] {
        assert!(dimensions("test", k, m, e, top).is_err());
    }
    assert_eq!(dimensions("test", 256, 7, 3, 16).unwrap().mode, 0);
    // New operand narrowing must not move before the route guards.
    assert!(
        SOURCE.find("if (expert < 0").unwrap()
            < SOURCE
                .find("const float value = expert_native_half_dot")
                .unwrap()
    );
    assert!(SOURCE.contains("partial = fma(float(w), float(activation), partial)"));
    // Existing exact-source regression remains applicable to all old modes.
    use sha2::{Digest, Sha256};
    let body = &SOURCE
        [SOURCE.find("    float sum = 0.0f;").unwrap()..SOURCE.find("\ntypedef decltype").unwrap()];
    assert_eq!(
        format!("{:x}", Sha256::digest(body.as_bytes())),
        "1bb4298433e54ba446cc5da2cc7d78d3b3431c7d54463763f9c7c55bef500b49"
    );
}

#[test]
fn native_half_rows_cpu_oracle_rounds_both_operands() {
    let x = values(768, 9);
    assert!(x.iter().any(|&v| half_operand(v) != v));
    for dtype in [GgmlType::IQ2_S, GgmlType::IQ3_S, GgmlType::IQ4_XS] {
        let bytes = bank_bytes(dtype, 768, 7, 3);
        let w = dequant_expert(&bytes, dtype, 768, 7, 2);
        assert!(w.iter().any(|&v| v != 0.0));
        assert!(w.iter().any(|&v| half_operand(v) != v));
        let mut differs_from_f32 = false;
        let mut needs_weight_rounding = false;
        let mut needs_input_rounding = false;
        for w in w.chunks_exact(768) {
            let expected = half_dot(w, &x);
            let f32_dot = w.iter().zip(&x).fold(0.0f32, |s, (&w, &x)| w.mul_add(x, s));
            let weight_only = w
                .iter()
                .zip(&x)
                .fold(0.0f32, |s, (&w, &x)| half_operand(w).mul_add(x, s));
            let input_only = w
                .iter()
                .zip(&x)
                .fold(0.0f32, |s, (&w, &x)| w.mul_add(half_operand(x), s));
            assert!(expected.is_finite());
            differs_from_f32 |= expected != f32_dot;
            needs_weight_rounding |= expected != input_only;
            needs_input_rounding |= expected != weight_only;
        }
        assert!(
            differs_from_f32 && needs_weight_rounding && needs_input_rounding,
            "{dtype:?}"
        );
    }
    // The gate clamp is upper-only and the inner store is not half-rounded.
    let inner = swiglu(-4.0, -4.0, 1.0);
    assert_eq!(inner, 4.0 / (1.0 + 4.0f32.exp()));
    assert_ne!(inner, half_operand(inner));
    assert_eq!(
        half_dot(&[1.0003], &[inner]),
        half_operand(1.0003) * half_operand(inner)
    );
}

const HALF_EXPERTS: usize = 19;
const TOP: usize = 8;

#[test]
#[ignore = "serialized Metal validation fixture; parent schedules device access"]
fn native_half_rows_metal_preflight_is_atomic() {
    let _lease = production_lease();
    let ctx = MetalContext::new().unwrap();
    let (k, m, top, n) = (256, 7, 6, 2);
    let make_bank = |dtype, seed| bank(&ctx, &bank_bytes(dtype, k, m, seed), dtype, k, m);
    let original = [
        make_bank(GgmlType::IQ2_S, 3),
        make_bank(GgmlType::IQ3_S, 7),
        f32_view(&ctx, &values(k * n, 9), &[k, n], 28),
        i32_view(&ctx, &vec![0; top * n]).view_subrange(0, vec![top as u64, n as u64]),
        i32_view(&ctx, &vec![1; n]),
        f32_view(&ctx, &vec![777.0; m * top * n], &[m, top * n], 28),
    ];
    let original_down = [
        make_bank(GgmlType::IQ4_XS, 3),
        f32_view(&ctx, &values(k * top * n, 5), &[k, top * n], 28),
        original[3].clone(),
        original[4].clone(),
        original[5].clone(),
    ];
    let before: Vec<_> = original
        .iter()
        .chain(&original_down)
        .map(tensor_backing_bytes)
        .collect();
    let command = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&command);
    let gate_up = |t: &[MetalTensor; 6], k, m, e, top, n, clamp| {
        encode_all_slots_gate_up_swiglu_native_half_rows(
            &ctx, &enc, &t[0], &t[1], &t[2], &t[3], &t[4], &t[5], k, m, e, top, n, clamp,
        )
    };
    let down = |t: &[MetalTensor; 5], k, m, e, top, n| {
        encode_all_slots_down_native_half_rows(
            &ctx, &enc, &t[0], &t[1], &t[2], &t[3], &t[4], k, m, e, top, n,
        )
    };
    let corrupt = |t: &mut MetalTensor, fault| match fault {
        0 => t.offset += 1,
        1 => t.offset = t.buffer.length() as u64,
        2 => t.offset = u64::MAX - 3,
        3 => t.dtype = GgmlType::F16,
        4 => t.shape = vec![u64::MAX, u64::MAX, 2],
        _ => t.shape = vec![1],
    };
    dispatch_census_begin();
    for which in 0..original.len() {
        for fault in 0..6 {
            let mut t = original.clone();
            corrupt(&mut t[which], fault);
            assert!(
                gate_up(&t, k, m, EXPERTS, top, n, 0.37).is_err(),
                "gate binding {which} fault {fault}"
            );
        }
    }
    for which in 0..original_down.len() {
        for fault in 0..6 {
            let mut t = original_down.clone();
            corrupt(&mut t[which], fault);
            assert!(
                down(&t, k, m, EXPERTS, top, n).is_err(),
                "down binding {which} fault {fault}"
            );
        }
    }
    // Valid extents on a shared backing force overlap validation, not size errors.
    let shared = ctx.buffer_from(&vec![0u8; 65536]).unwrap();
    for which in 0..5 {
        let mut t = original.clone();
        t[which].buffer = shared.clone();
        t[which].offset = 32;
        t[5].buffer = shared.clone();
        t[5].offset = 32;
        assert!(
            gate_up(&t, k, m, EXPERTS, top, n, 0.37)
                .unwrap_err()
                .to_string()
                .contains("aliases")
        );
    }
    for which in 0..4 {
        let mut t = original_down.clone();
        t[which].buffer = shared.clone();
        t[which].offset = 32;
        t[4].buffer = shared.clone();
        t[4].offset = 32;
        assert!(
            down(&t, k, m, EXPERTS, top, n)
                .unwrap_err()
                .to_string()
                .contains("aliases")
        );
    }
    let mut t = original.clone();
    t[5].provenance = MetalTensorProvenance::OwnedWeightReadOnly;
    assert!(gate_up(&t, k, m, EXPERTS, top, n, 0.37).is_err());
    let mut d = original_down.clone();
    d[4].provenance = MetalTensorProvenance::OwnedWeightReadOnly;
    assert!(down(&d, k, m, EXPERTS, top, n).is_err());
    for clamp in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, -0.0, -1.0] {
        assert!(gate_up(&original, k, m, EXPERTS, top, n, clamp).is_err());
    }
    for (k, m, e, top, rows) in [
        (0, m, EXPERTS, top, n),
        (k, 0, EXPERTS, top, n),
        (k, m, 0, top, n),
        (k, m, EXPERTS, 0, n),
        (k, m, EXPERTS, 17, n),
        (k, m, EXPERTS, top, 0),
        (k, m, EXPERTS, top, 3),
        (k, m, EXPERTS, top, 16),
        (k, m, EXPERTS, top, usize::MAX),
        (usize::MAX, m, EXPERTS, top, n),
        (k, usize::MAX, EXPERTS, top, n),
        (k, m, usize::MAX, top, n),
    ] {
        assert!(gate_up(&original, k, m, e, top, rows, 0.37).is_err());
        assert!(down(&original_down, k, m, e, top, rows).is_err());
    }
    // Full bank storage is insufficient when each K row is not block-aligned.
    let mut t = original.clone();
    t[0].shape[0] = 128;
    t[1].shape[0] = 128;
    t[2].shape[0] = 128;
    assert!(gate_up(&t, 128, m, EXPERTS, top, n, 0.37).is_err());
    let mut d = original_down.clone();
    d[0].shape[0] = 128;
    d[1].shape[0] = 128;
    assert!(down(&d, 128, m, EXPERTS, top, n).is_err());
    for dtype in [
        GgmlType::Q2_K,
        GgmlType::Q3_K,
        GgmlType::IQ4_XS,
        GgmlType::F32,
    ] {
        let mut t = original.clone();
        t[1].dtype = dtype;
        assert!(gate_up(&t, k, m, EXPERTS, top, n, 0.37).is_err());
    }
    for dtype in [GgmlType::Q3_K, GgmlType::IQ2_S, GgmlType::F32] {
        let mut d = original_down.clone();
        d[0].dtype = dtype;
        assert!(down(&d, k, m, EXPERTS, top, n).is_err());
    }
    let devices = objc2_metal::MTLCopyAllDevices();
    if let Some(device) = devices
        .iter()
        .find(|d| d.registryID() != ctx.device.registryID())
    {
        for which in 0..6 {
            let mut t = original.clone();
            t[which].buffer = device
                .newBufferWithLength_options(
                    t[which].buffer.length(),
                    MTLResourceOptions::StorageModeShared,
                )
                .unwrap();
            assert!(gate_up(&t, k, m, EXPERTS, top, n, 0.37).is_err());
        }
        for which in 0..5 {
            let mut d = original_down.clone();
            d[which].buffer = device
                .newBufferWithLength_options(
                    d[which].buffer.length(),
                    MTLResourceOptions::StorageModeShared,
                )
                .unwrap();
            assert!(down(&d, k, m, EXPERTS, top, n).is_err());
        }
    }
    enc.end();
    command.commit();
    command.waitUntilCompleted();
    assert!(dispatch_census_take().is_empty());
    for (t, bytes) in original.iter().chain(&original_down).zip(before) {
        assert_eq!(tensor_backing_bytes(t), bytes);
    }
    let command = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin_concurrent(&command);
    dispatch_census_begin();
    assert!(
        encode_all_slots_gate_up_swiglu_native_half_rows(
            &ctx,
            &enc,
            &original[0],
            &original[1],
            &original[2],
            &original[3],
            &original[4],
            &original[5],
            k,
            m,
            EXPERTS,
            top,
            n,
            0.37
        )
        .is_err()
    );
    assert!(
        encode_all_slots_down_native_half_rows(
            &ctx,
            &enc,
            &original_down[0],
            &original_down[1],
            &original_down[2],
            &original_down[3],
            &original_down[4],
            k,
            m,
            EXPERTS,
            top,
            n
        )
        .is_err()
    );
    enc.end();
    assert!(dispatch_census_take().is_empty());
}

fn route_ids(rows: usize) -> Vec<i32> {
    (0..rows)
        .flat_map(|r| [18, (r % 17) as i32, 18, -1, 19, i32::MIN, i32::MAX, 7])
        .collect()
}

#[test]
#[ignore = "characterize native SIMD versus grouped MMA; parent schedules GPU execution"]
fn native_half_rows_metal_grouped_reduction_deltas() {
    let _lease = production_lease();
    let ctx = MetalContext::new().unwrap();
    let (k, m, top) = (768, 7, 2);
    for (g, u, down) in [
        (GgmlType::IQ2_S, GgmlType::IQ2_S, false),
        (GgmlType::IQ3_S, GgmlType::IQ3_S, false),
        (GgmlType::IQ2_S, GgmlType::IQ3_S, false),
        (GgmlType::IQ3_S, GgmlType::IQ2_S, false),
        (GgmlType::IQ3_S, GgmlType::IQ3_S, true),
        (GgmlType::IQ4_XS, GgmlType::IQ4_XS, true),
    ] {
        let gb = bank_bytes(g, k, m, 3);
        let ub = bank_bytes(u, k, m, 7);
        let gate = bank(&ctx, &gb, g, k, m);
        let up = bank(&ctx, &ub, u, k, m);
        for rows in [1, 2, 3, 4, 7, 8] {
            // Grouped gate buckets have capacity rows per expert, so this
            // comparison uses unique experts within each row (not duplicates).
            let routes: Vec<_> = (0..rows)
                .flat_map(|r| [(r % EXPERTS) as i32, ((r + 1) % EXPERTS) as i32])
                .collect();
            let slots = rows * top;
            let capacity = if down { slots } else { rows };
            let mut counts = vec![0i32; EXPERTS];
            let mut bucket_ids = vec![-1i32; EXPERTS * capacity];
            for (slot, &id) in routes.iter().enumerate() {
                let e = id as usize;
                bucket_ids[e * capacity + counts[e] as usize] = slot as i32;
                counts[e] += 1;
            }
            let counts = i32_view(&ctx, &counts);
            let bucket_ids = i32_view(&ctx, &bucket_ids);
            let ids = i32_view(&ctx, &routes).view_subrange(0, vec![top as u64, rows as u64]);
            let status = i32_view(&ctx, &vec![1; rows]);
            let x = values(k * capacity, 9);
            // Mixed grouped kernels require 16-byte alignment; native-only
            // witnesses separately cover the looser four-byte contract.
            let aligned = |values: &[f32], shape: &[usize]| {
                offset_tensor(
                    &ctx,
                    32,
                    bytemuck::cast_slice(values),
                    28,
                    shape.iter().map(|&n| n as u64).collect(),
                    GgmlType::F32,
                )
            };
            let input = aligned(&x, &[k, capacity]);
            let native = aligned(&vec![f32::NAN; m * slots], &[m, slots]);
            let grouped = aligned(&vec![f32::NAN; m * slots], &[m, slots]);
            one_shot(&ctx, |enc| {
                if down {
                    encode_all_slots_down_native_half_rows(
                        &ctx, enc, &gate, &input, &ids, &status, &native, k, m, EXPERTS, top, rows,
                    )?;
                    encode_moe_down_f32_grouped_slots_generic(
                        &ctx,
                        enc,
                        &gate,
                        &input,
                        &counts,
                        &bucket_ids,
                        &grouped,
                        k,
                        m,
                        EXPERTS,
                        slots,
                    )
                } else {
                    encode_all_slots_gate_up_swiglu_native_half_rows(
                        &ctx, enc, &gate, &up, &input, &ids, &status, &native, k, m, EXPERTS, top,
                        rows, 100.0,
                    )?;
                    encode_moe_swiglu_clamped_f32_grouped_slots_generic(
                        &ctx,
                        enc,
                        &gate,
                        &up,
                        &input,
                        &counts,
                        &bucket_ids,
                        &grouped,
                        k,
                        m,
                        EXPERTS,
                        top,
                        rows,
                        100.0,
                    )
                }
            })
            .unwrap();
            let a = tensor_f32_at_offset(&native);
            let b = tensor_f32_at_offset(&grouped);
            let label = format!("native/grouped {g:?}/{u:?} down={down} rows={rows}");
            let max_abs = a
                .iter()
                .zip(&b)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let unequal_bits = a
                .iter()
                .zip(&b)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let (rel_l2, scaled_max, cosine) = crate::metal::test_support::f32_differential(&a, &b);
            eprintln!(
                "{label}: max_abs={max_abs:e} rel_l2={rel_l2:e} scaled_max={scaled_max:e} cosine={cosine} unequal_bits={unequal_bits}/{}",
                a.len()
            );
            let mut max_ratio = [0.0f64; 2];
            let mut max_error = [0.0f64; 2];
            let mut max_condition = 0.0f64;
            for (slot, &expert) in routes.iter().enumerate() {
                let gw = dequant_expert(&gb, g, k, m, expert as usize);
                let uw = dequant_expert(&ub, u, k, m, expert as usize);
                let start = if down { slot * k } else { slot / top * k };
                for channel in 0..m {
                    let gd = HalfDot::new(&gw[channel * k..][..k], &x[start..][..k]);
                    let ud = HalfDot::new(&uw[channel * k..][..k], &x[start..][..k]);
                    let index = slot * m + channel;
                    max_condition =
                        max_condition.max(gd.l1 / gd.value.abs().max(f64::MIN_POSITIVE));
                    if !down {
                        max_condition =
                            max_condition.max(ud.l1 / ud.value.abs().max(f64::MIN_POSITIVE));
                    }
                    for (branch, reduction, actual) in [
                        (0, Reduction::Native, a[index]),
                        (1, Reduction::Grouped, b[index]),
                    ] {
                        let oracle = if down {
                            gd.bound(reduction)
                        } else {
                            swiglu_bound(gd.bound(reduction), ud.bound(reduction), 100.0)
                        };
                        max_ratio[branch] = max_ratio[branch].max(oracle.check(
                            &format!("{label} {reduction:?}"),
                            index,
                            actual,
                        ));
                        max_error[branch] =
                            max_error[branch].max((f64::from(actual) - oracle.value).abs());
                    }
                }
            }
            eprintln!(
                "{label}: F64_max_abs(native,grouped)={max_error:?} max_error_over_bound={max_ratio:?} max_dot_condition={max_condition:e}"
            );
            assert_offset_guards(&native, 32, 28);
            assert_offset_guards(&grouped, 32, 28);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn half_case(
    ctx: &MetalContext,
    dtype: GgmlType,
    up_type: GgmlType,
    down: bool,
    rows: usize,
    k: usize,
    m: usize,
    partition: usize,
    reject_status: bool,
    clamp: f32,
) -> Vec<u32> {
    let gb = bank_bytes_for_experts(dtype, k, m, 3, HALF_EXPERTS);
    let ub = bank_bytes_for_experts(up_type, k, m, 7, HALF_EXPERTS);
    let make_bank = |bytes: &[u8], dtype| {
        offset_tensor(
            ctx,
            32,
            bytes,
            16,
            vec![k as u64, m as u64, HALF_EXPERTS as u64],
            dtype,
        )
    };
    let gate = make_bank(&gb, dtype);
    let up = make_bank(&ub, up_type);
    let width = k * if down { TOP } else { 1 };
    let mut x = values(width * rows, 9);
    if down {
        // Duplicate IDs still read independent slot activations.
        x[2 * k..3 * k].fill(0.0);
    }
    let input = f32_view(ctx, &x, &[k, if down { TOP * rows } else { rows }], 28);
    let routes = route_ids(rows);
    let ids = offset_tensor(
        ctx,
        12,
        bytemuck::cast_slice(&routes),
        20,
        vec![TOP as u64, rows as u64],
        GgmlType::I32,
    );
    let statuses: Vec<_> = (0..rows)
        .map(|r| {
            if reject_status {
                [0, -7, 2, 1][r % 4]
            } else {
                1
            }
        })
        .collect();
    let status = offset_tensor(
        ctx,
        12,
        bytemuck::cast_slice(&statuses),
        20,
        vec![rows as u64],
        GgmlType::I32,
    );
    let output = f32_view(ctx, &vec![f32::NAN; m * TOP * rows], &[m, TOP * rows], 28);
    let inputs = [&gate, &up, &input, &ids, &status];
    let before: Vec<_> = inputs.iter().map(|t| tensor_backing_bytes(t)).collect();
    dispatch_census_begin();
    one_shot(ctx, |enc| {
        for r in (0..rows).step_by(partition) {
            let n = partition.min(rows - r);
            let input = input.view_subrange(
                (r * width) as u64,
                vec![k as u64, (if down { TOP * n } else { n }) as u64],
            );
            let ids = ids.view_subrange((r * TOP) as u64, vec![TOP as u64, n as u64]);
            let status = status.view_subrange(r as u64, vec![n as u64]);
            let output =
                output.view_subrange((r * m * TOP) as u64, vec![m as u64, (TOP * n) as u64]);
            if down {
                encode_all_slots_down_native_half_rows(
                    ctx,
                    enc,
                    &gate,
                    &input,
                    &ids,
                    &status,
                    &output,
                    k,
                    m,
                    HALF_EXPERTS,
                    TOP,
                    n,
                )?;
            } else {
                encode_all_slots_gate_up_swiglu_native_half_rows(
                    ctx,
                    enc,
                    &gate,
                    &up,
                    &input,
                    &ids,
                    &status,
                    &output,
                    k,
                    m,
                    HALF_EXPERTS,
                    TOP,
                    n,
                    clamp,
                )?;
            }
        }
        Ok(())
    })
    .unwrap();
    let census = dispatch_census_take();
    let dispatches = if down { 1 } else { 2 };
    assert_eq!(census.len(), rows.div_ceil(partition) * dispatches);
    for (index, record) in census.iter().enumerate() {
        let chunk_rows = partition.min(rows - index / dispatches * partition);
        assert_eq!(
            (record.grid_width, record.grid_height, record.grid_depth),
            (m.div_ceil(4) as u64, TOP as u64, chunk_rows as u64)
        );
        assert_eq!(
            (
                record.threads_width,
                record.threads_height,
                record.threads_depth
            ),
            (128, 1, 1)
        );
    }
    let mut expected = vec![Bounded::default(); m * TOP * rows];
    for r in 0..rows {
        for slot in 0..TOP {
            let id = routes[r * TOP + slot];
            if statuses[r] != ROUTE_STATUS_READY || id < 0 || id as usize >= HALF_EXPERTS {
                continue;
            }
            let gw = dequant_expert(&gb, dtype, k, m, id as usize);
            let uw = dequant_expert(&ub, up_type, k, m, id as usize);
            let start = r * width + if down { slot * k } else { 0 };
            for channel in 0..m {
                let g = HalfDot::new(&gw[channel * k..][..k], &x[start..][..k])
                    .bound(Reduction::Native);
                expected[(r * TOP + slot) * m + channel] = if down {
                    g
                } else {
                    swiglu_bound(
                        g,
                        HalfDot::new(&uw[channel * k..][..k], &x[start..][..k])
                            .bound(Reduction::Native),
                        clamp,
                    )
                };
            }
        }
    }
    let actual = tensor_f32_at_offset(&output);
    let label = format!("half {dtype:?}/{up_type:?} down={down} rows={rows} K={k}");
    let mut max_ratio = 0.0f64;
    for (index, (&a, e)) in actual.iter().zip(&expected).enumerate() {
        // Rejected routes have the default zero bound: exact zero, no floor.
        max_ratio = max_ratio.max(e.check(&label, index, a));
    }
    let max_abs = actual
        .iter()
        .zip(&expected)
        .map(|(a, e)| (f64::from(*a) - e.value).abs())
        .fold(0.0f64, f64::max);
    eprintln!("{label} F64_max_abs={max_abs:e} max_error_over_bound={max_ratio:e}");
    assert_offset_guards(&output, 20, 28);
    for (t, bytes) in inputs.iter().zip(before) {
        assert_eq!(tensor_backing_bytes(t), bytes);
    }
    actual.iter().map(|v| v.to_bits()).collect()
}

#[test]
#[ignore = "serialized model-free Metal primitive; parent schedules GPU execution"]
fn native_half_rows_metal_oracle_partitions_routes_and_guards() {
    let _lease = production_lease();
    let ctx = MetalContext::new().unwrap();
    for (g, u, down) in [
        (GgmlType::IQ2_S, GgmlType::IQ2_S, false),
        (GgmlType::IQ2_S, GgmlType::IQ3_S, false),
        (GgmlType::IQ3_S, GgmlType::IQ2_S, false),
        (GgmlType::IQ3_S, GgmlType::IQ3_S, false),
        (GgmlType::IQ3_S, GgmlType::IQ3_S, true),
        (GgmlType::IQ4_XS, GgmlType::IQ4_XS, true),
    ] {
        for (k, m) in [(256, 1), (768, 7), (1280, 5)] {
            for rows in [1, 2, 3, 4, 7, 8] {
                for (reject, clamp) in [(false, 0.37), (false, 100.0), (true, 0.37)] {
                    let rowwise = half_case(&ctx, g, u, down, rows, k, m, 1, reject, clamp);
                    for partition in [2, 3, 4, 8].into_iter().filter(|&n| n <= rows) {
                        assert_eq!(
                            half_case(&ctx, g, u, down, rows, k, m, partition, reject, clamp),
                            rowwise,
                            "half row partition must be bit-identical: {g:?}/{u:?} down={down} rows={rows} chunk={partition}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "serialized model-free Metal primitive; parent schedules GPU execution"]
fn native_half_rows_metal_half_operand_basis() {
    let _lease = production_lease();
    let ctx = MetalContext::new().unwrap();
    let (k, m) = (256, 7);
    for dtype in [GgmlType::IQ2_S, GgmlType::IQ3_S, GgmlType::IQ4_XS] {
        let bytes = bank_bytes(dtype, k, m, 3);
        let weights = bank(&ctx, &bytes, dtype, k, m);
        let mut x = vec![0.0; k];
        x[0] = 1.0003;
        let input = f32_view(&ctx, &x, &[k, 1], 0);
        let ids = i32_view(&ctx, &[2]).view_subrange(0, vec![1, 1]);
        let status = i32_view(&ctx, &[1]);
        let output = f32_view(&ctx, &vec![f32::NAN; m], &[m, 1], 28);
        // Isolate the gate-only shader operation (IQ2_S is not a public down
        // capability). Public two-pass/one-pass validation is tested elsewhere.
        let mut args = dimensions("half basis", k, m, EXPERTS, 1).unwrap();
        let (pso, row_bytes) = bank_pipeline(&ctx, "half basis", &weights, &args, "bank").unwrap();
        args.row_bytes = row_bytes;
        args.mode = NativeOperands::Half.mode(0);
        one_shot(&ctx, |enc| {
            dispatch(
                enc, &pso, &args, &weights, &input, &ids, &status, &output, 1,
            );
            Ok(())
        })
        .unwrap();
        let w = dequant_expert(&bytes, dtype, k, m, 2);
        let mut sensitive_to_weight_cast = false;
        let mut sensitive_to_input_cast = false;
        for (channel, actual) in tensor_f32_at_offset(&output).into_iter().enumerate() {
            let w = w[channel * k];
            let expected = half_operand(w).mul_add(half_operand(x[0]), 0.0);
            // One exact half product plus zeros: no reduction/epilogue allowance.
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "{dtype:?} channel={channel}"
            );
            sensitive_to_weight_cast |= expected != w.mul_add(half_operand(x[0]), 0.0);
            sensitive_to_input_cast |= expected != half_operand(w).mul_add(x[0], 0.0);
        }
        assert!(
            sensitive_to_weight_cast && sensitive_to_input_cast,
            "{dtype:?}"
        );
        assert_offset_guards(&output, 20, 28);
    }
}

#[test]
#[ignore = "serialized model-free Metal primitive; parent schedules GPU execution"]
fn native_half_rows_metal_old_f32_basis_regression() {
    let _lease = production_lease();
    let ctx = MetalContext::new().unwrap();
    let (k, m) = (256, 7);
    for dtype in [GgmlType::IQ2_S, GgmlType::IQ3_S, GgmlType::IQ4_XS] {
        let bytes = bank_bytes(dtype, k, m, 3);
        let weights = bank(&ctx, &bytes, dtype, k, m);
        let mut x = vec![0.0; k * 2];
        x[0] = 70000.125;
        x[k] = -1.0003;
        let input = f32_view(&ctx, &x, &[k, 2], 0);
        let ids = i32_view(&ctx, &[2, 0]);
        let status = i32_view(&ctx, &[1]);
        let output = f32_view(&ctx, &vec![f32::NAN; m * 2], &[m, 2], 28);
        one_shot(&ctx, |enc| {
            encode_all_slots_down_native(
                &ctx, enc, &weights, &input, &ids, &status, &output, k, m, EXPERTS, 2,
            )
        })
        .unwrap();
        let actual = tensor_f32_at_offset(&output);
        for (slot, id) in [2, 0].into_iter().enumerate() {
            let w = dequant_expert(&bytes, dtype, k, m, id);
            for channel in 0..m {
                assert_eq!(
                    actual[slot * m + channel].to_bits(),
                    w[channel * k].mul_add(x[slot * k], 0.0).to_bits()
                );
            }
        }
        assert_offset_guards(&output, 20, 28);
    }
}
