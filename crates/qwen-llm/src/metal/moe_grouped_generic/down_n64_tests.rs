use super::*;
use crate::metal::test_support::{
    assert_offset_guards, dequant_expert, metal_test_context, offset_tensor, tensor_backing_bytes,
    tensor_f32_at_offset,
};

const SOURCE: &str = include_str!("../../../../../kernels/research/moe_down_n64.metal");
const COUNTS: [usize; 10] = [0, 1, 3, 17, 32, 33, 64, 65, 96, 97];
const SLOTS: usize = 512;
const SENTINEL: f32 = -777.0;

#[test]
fn down_n64_native_mapping_and_override_scope() {
    assert_eq!(std::mem::size_of::<GenericMmArgs>(), 36);
    assert_eq!(SOURCE.matches("template [[host_name(").count(), 17);
    for layout in MOE_GROUPED_GENERIC_LAYOUTS {
        let name = moe_down_n64_pipeline_name(layout.dtype).unwrap();
        let line = SOURCE
            .lines()
            .find(|l| l.contains(&format!("host_name(\"{name}\")")))
            .unwrap();
        let upper = layout.suffix.to_uppercase();
        assert!(line.contains(&format!(
            "<QT_{upper}_BYTES, QT_{upper}_NL, qt_dequantize_{}>",
            layout.suffix
        )));
    }
    assert!(moe_down_n64_pipeline_name(GgmlType::I32).is_none());
    assert!(!grouped_down_wide_enabled());
    let value = with_grouped_down_wide_override(true, || {
        assert!(grouped_down_wide_enabled());
        with_grouped_down_wide_override(false, || assert!(!grouped_down_wide_enabled()));
        assert!(grouped_down_wide_enabled());
        let _ = std::panic::catch_unwind(|| {
            with_grouped_down_wide_override(false, || panic!("restore"))
        });
        assert!(grouped_down_wide_enabled());
        std::thread::spawn(|| assert!(!grouped_down_wide_enabled()))
            .join()
            .unwrap();
        7
    });
    assert_eq!(value, 7);
    assert!(!grouped_down_wide_enabled());
}

fn bank_bytes(dtype: GgmlType, k: usize, m: usize, experts: usize) -> Vec<u8> {
    let (block, block_bytes) = dtype.storage_layout().unwrap();
    let mut bytes = vec![0; k * m * experts / block as usize * block_bytes as usize];
    let mut state = 0xdead_beefu32;
    for (ordinal, block) in bytes.chunks_exact_mut(block_bytes as usize).enumerate() {
        for b in &mut block[2..] {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *b = (state >> 24) as u8;
        }
        let sign = if ordinal % 3 == 0 { -1.0 } else { 1.0 };
        let d = sign * (ordinal % 7 + 1) as f32 / 4096.0;
        block[..2].copy_from_slice(&half::f16::from_f32(d).to_le_bytes());
    }
    bytes
}

/// Standard forward-error bound for K rounded F32 additions. Products of
/// these finite half operands fit exactly in F32; neither product overflow nor
/// underflow occurs in the fixture. u is unit roundoff, not f32::EPSILON.
fn half_dot_roundoff_bound(k: usize, sum_abs_products: f64) -> f64 {
    let ku = k as f64 * (f64::from(f32::EPSILON) / 2.0);
    assert!(ku < 1.0);
    (ku / (1.0 - ku)) * sum_abs_products
}

/// The originally failing dot: IQ3_S expert 6, count 64, position 54, slot 71.
/// Its Half operands match the CPU codec, but cancellation amplifies ordinary
/// F32 accumulation error beyond 1e-5 relative to the small final result.
#[test]
fn down_n64_half_dot_cancellation_regression() {
    let bytes = bank_bytes(GgmlType::IQ3_S, 256, 1, COUNTS.len());
    let w = dequant_expert(&bytes, GgmlType::IQ3_S, 256, 1, 6);
    let products: Vec<f64> = w
        .iter()
        .zip(71 * 256..72 * 256)
        .map(|(&w, i)| {
            let x = ((i * 17 + i / 23) % 107) as f32 / 51.0 - 1.0;
            half::f16::from_f32(w).to_f64() * half::f16::from_f32(x).to_f64()
        })
        .collect();
    let exact: f64 = products.iter().sum();
    let sum_abs: f64 = products.iter().map(|v| v.abs()).sum();
    assert_eq!(exact, -0.004048764705657959);
    assert_eq!(sum_abs, 21.779911175370216);
    assert!(products.iter().all(|&p| f64::from(p as f32) == p));
    let accumulated = products.iter().fold(0.0f32, |sum, &p| sum + p as f32);
    let error = (f64::from(accumulated) - exact).abs();
    assert!(
        error / exact.abs() > 1e-5,
        "reproduce the cancellation-sensitive rejection"
    );
    assert!(error <= half_dot_roundoff_bound(256, sum_abs));
    // The bound still rejects a materially wrong output independently of the
    // bitwise comparison against the incumbent.
    assert!((f64::from(accumulated) + 0.001 - exact).abs() > half_dot_roundoff_bound(256, sum_abs));
}

struct Fixture {
    k: usize,
    m: usize,
    n: usize,
    weight: MetalTensor,
    inner: MetalTensor,
    counts: MetalTensor,
    ids: MetalTensor,
    expected: Vec<f64>,
    sum_abs_products: Vec<f64>,
    touched: Vec<bool>,
}

impl Fixture {
    fn new(
        ctx: &MetalContext,
        dtype: GgmlType,
        k: usize,
        m: usize,
        n: usize,
        invalid: bool,
    ) -> Self {
        let bytes = bank_bytes(dtype, k, m, COUNTS.len());
        let x: Vec<f32> = (0..k * SLOTS)
            .map(|i| ((i * 17 + i / 23) % 107) as f32 / 51.0 - 1.0)
            .collect();
        let mut ids = vec![i32::MAX; n * COUNTS.len()];
        let mut expected = vec![SENTINEL as f64; m * SLOTS];
        let mut sum_abs_products = vec![0.0; m * SLOTS];
        let mut touched = vec![false; SLOTS];
        let mut next = 0;
        for (expert, &count) in COUNTS.iter().enumerate() {
            // CPU GGML codec, then independent Half operand conversion and F64
            // dot products. No GPU dequantizer or incumbent output is the oracle.
            let w = dequant_expert(&bytes, dtype, k, m, expert);
            let half_w: Vec<f64> = w
                .into_iter()
                .map(|v| half::f16::from_f32(v).to_f64())
                .collect();
            for j in 0..count {
                let slot = (next * 37 + 11) % SLOTS;
                next += 1;
                let id = if invalid {
                    match (count, j) {
                        (32, 31) | (65, 64) => -1,
                        (33, 32) | (97, 96) => SLOTS as i32,
                        (64, 32) | (96, 64) => i32::MAX,
                        _ => slot as i32,
                    }
                } else {
                    slot as i32
                };
                ids[expert * n + j] = id;
                if id < 0 || id >= SLOTS as i32 {
                    continue;
                }
                assert!(!touched[slot]);
                touched[slot] = true;
                for row in 0..m {
                    let products = half_w[row * k..(row + 1) * k]
                        .iter()
                        .zip(&x[slot * k..(slot + 1) * k])
                        .map(|(&w, &x)| w * half::f16::from_f32(x).to_f64());
                    expected[slot * m + row] = products.clone().sum();
                    sum_abs_products[slot * m + row] = products.map(f64::abs).sum();
                }
            }
        }
        Self {
            k,
            m,
            n,
            weight: offset_tensor(
                ctx,
                18,
                &bytes,
                22,
                vec![k as u64, m as u64, COUNTS.len() as u64],
                dtype,
            ),
            inner: offset_tensor(
                ctx,
                16,
                bytemuck::cast_slice(&x),
                20,
                vec![(k * SLOTS) as u64],
                GgmlType::F32,
            ),
            counts: offset_tensor(
                ctx,
                12,
                bytemuck::cast_slice(&COUNTS.map(|n| n as i32)),
                20,
                vec![COUNTS.len() as u64],
                GgmlType::I32,
            ),
            ids: offset_tensor(
                ctx,
                12,
                bytemuck::cast_slice(&ids),
                20,
                vec![(n * COUNTS.len()) as u64],
                GgmlType::I32,
            ),
            expected,
            sum_abs_products,
            touched,
        }
    }

    fn output(&self, ctx: &MetalContext) -> MetalTensor {
        offset_tensor(
            ctx,
            20,
            bytemuck::cast_slice(&vec![SENTINEL; self.m * SLOTS]),
            28,
            vec![(self.m * SLOTS) as u64],
            GgmlType::F32,
        )
    }

    fn prepare<'a>(
        &'a self,
        ctx: &MetalContext,
        enc: &'a KernelEncoder,
        out: &'a MetalTensor,
    ) -> Result<PreparedGroupedDown<'a>, MetalError> {
        prepare_moe_down_f32_grouped_slots_n64_split(
            ctx,
            enc,
            &self.weight,
            &self.inner,
            &self.counts,
            &self.ids,
            out,
            self.k,
            self.m,
            COUNTS.len(),
            self.n,
        )
    }

    fn assert_bits_equal(&self, label: &str, out: &MetalTensor, incumbent: &MetalTensor) {
        let actual = tensor_f32_at_offset(out);
        let baseline = tensor_f32_at_offset(incumbent);
        for (i, (&a, &b)) in actual.iter().zip(&baseline).enumerate() {
            let e = self.expected[i];
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{label} {:?} K={} M={} N={} slot={} row={}: actual={a:.12e}, incumbent={b:.12e}, CPU={e:.12e}, actual-CPU={:.12e}, incumbent-CPU={:.12e}, actual-incumbent={:.12e}",
                self.weight.dtype,
                self.k,
                self.m,
                self.n,
                i / self.m,
                i % self.m,
                f64::from(a) - e,
                f64::from(b) - e,
                f64::from(a) - f64::from(b)
            );
        }
    }

    fn assert_output(&self, label: &str, out: &MetalTensor, incumbent: &MetalTensor) {
        let actual = tensor_f32_at_offset(out);
        let baseline = tensor_f32_at_offset(incumbent);
        let (mut diff, mut norm) = (0.0, 0.0);
        let mut worst_relative = (0.0, 0usize);
        for (i, &a) in actual.iter().enumerate() {
            let slot = i / self.m;
            let row = i % self.m;
            let b = baseline[i];
            let e = self.expected[i];
            let error = f64::from(a) - e;
            let sum_abs = self.sum_abs_products[i];
            let bound = half_dot_roundoff_bound(self.k, sum_abs);
            let detail = || {
                format!(
                    "{label} {:?} K={} M={} N={} slot={slot} row={row}: actual={a:.12e}, incumbent={b:.12e}, CPU={e:.12e}, actual-CPU={error:.12e}, incumbent-CPU={:.12e}, bit_equal={}, sum_abs_products={sum_abs:.12e}, roundoff_bound={bound:.12e}",
                    self.weight.dtype,
                    self.k,
                    self.m,
                    self.n,
                    f64::from(b) - e,
                    a.to_bits() == b.to_bits()
                )
            };
            assert!(a.is_finite() && e.is_finite(), "{}", detail());
            if !self.touched[slot] {
                assert_eq!(a.to_bits(), SENTINEL.to_bits(), "{}", detail());
                continue;
            }
            // Half operands have exact F32 products in this fixture. F32
            // accumulation is bounded by gamma_K * sum(abs(w_half*x_half)).
            // Dividing by the output instead is ill-conditioned near cancellation
            // (especially M=1); it can reject valid F32 accumulation.
            assert!(error.abs() <= bound, "{}", detail());
            // Keep the original absolute check and 1e-5 relative RMS criterion,
            // with RMS over all written outputs rather than each scalar dot.
            assert!(error.abs() <= 3e-5 * (1.0 + e.abs()), "{}", detail());
            diff += error * error;
            norm += e * e;
            let relative = if e == 0.0 {
                if error == 0.0 { 0.0 } else { f64::INFINITY }
            } else {
                error.abs() / e.abs()
            };
            if relative > worst_relative.0 {
                worst_relative = (relative, i);
            }
        }
        let rms = if norm == 0.0 {
            diff.sqrt()
        } else {
            (diff / norm).sqrt()
        };
        let (relative, i) = worst_relative;
        eprintln!(
            "{label} {:?} K={} M={} N={}: CPU RMS={rms:.6e}, worst output-relative={relative:.6e} at slot={} row={}; actual={:.12e}, incumbent={:.12e}, CPU={:.12e}, abs_error={:.12e}, sum_abs_products={:.12e}, roundoff_bound={:.12e}",
            self.weight.dtype,
            self.k,
            self.m,
            self.n,
            i / self.m,
            i % self.m,
            actual[i],
            baseline[i],
            self.expected[i],
            (f64::from(actual[i]) - self.expected[i]).abs(),
            self.sum_abs_products[i],
            half_dot_roundoff_bound(self.k, self.sum_abs_products[i])
        );
        assert!(
            rms < 1e-5,
            "{label} {:?} K={} M={} N={}: aggregate CPU relative RMS={rms:.12e}, error L2={:.12e}, CPU L2={:.12e}",
            self.weight.dtype,
            self.k,
            self.m,
            self.n,
            diff.sqrt(),
            norm.sqrt()
        );
        assert_offset_guards(out, 20, 28);
    }
}

#[test]
fn down_n64_boundaries_match_incumbent_bits_and_independent_half_dot() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    for dtype in [
        GgmlType::IQ3_S,
        GgmlType::IQ3_XXS,
        GgmlType::IQ4_XS,
        GgmlType::Q2_0,
        GgmlType::Q8_0,
    ] {
        // Multiple storage blocks, M64 edge plus tail, and N512 sparse grid.
        for (k, m, n, invalid) in [
            (256, 1, 97, true),
            (512, 67, 512, true),
            (256, 65, 129, false),
        ] {
            // The normal test context must resolve N64 through the lazy research
            // fallback; the product metallib must remain free of this experiment.
            let name = moe_down_n64_pipeline_name(dtype).unwrap();
            assert!(
                ctx.library
                    .newFunctionWithName(&NSString::from_str(&name))
                    .is_none()
            );
            let f = Fixture::new(&ctx, dtype, k, m, n, invalid);
            let inputs = [&f.weight, &f.inner, &f.counts, &f.ids];
            let before: Vec<_> = inputs.iter().map(|t| tensor_backing_bytes(t)).collect();
            let candidate = f.output(&ctx);
            let incumbent = f.output(&ctx);
            let capped = f.output(&ctx);
            dispatch_census_begin();
            one_shot(&ctx, |enc| {
                let prepared = f.prepare(&ctx, enc, &candidate)?;
                assert_eq!(prepared.args.n as usize, n);
                assert_eq!(prepared.panels.len(), 2);
                // Both input staging and each sequential N32 output panel fit
                // the same 8 KiB overlay, including the N64 dispatch.
                assert!(prepared.panels.iter().all(|p| p.threadgroup_bytes == 8192));
                assert_eq!(
                    (
                        prepared.panels[0].min_count,
                        prepared.panels[0].max_count,
                        prepared.panels[0].grid_width
                    ),
                    (1, 32, 1)
                );
                assert_eq!(
                    (
                        prepared.panels[1].min_count,
                        prepared.panels[1].max_count,
                        prepared.panels[1].grid_width
                    ),
                    (33, n as u32, n.div_ceil(64))
                );
                prepared.encode();
                encode_moe_down_f32_grouped_slots_generic(
                    &ctx,
                    enc,
                    &f.weight,
                    &f.inner,
                    &f.counts,
                    &f.ids,
                    &incumbent,
                    k,
                    m,
                    COUNTS.len(),
                    n,
                )?;
                encode_moe_down_f32_grouped_slots_generic_capped(
                    &ctx,
                    enc,
                    &f.weight,
                    &f.inner,
                    &f.counts,
                    &f.ids,
                    &capped,
                    k,
                    m,
                    COUNTS.len(),
                    n,
                    97,
                )
            })
            .unwrap();
            let census = dispatch_census_take();
            assert_eq!(census.len(), 4);
            assert_eq!(census[0].grid_width, 1);
            assert_eq!(census[1].grid_width, n.div_ceil(64) as u64);
            assert_eq!(census[1].kernel, moe_down_n64_pipeline_name(dtype).unwrap());
            assert_eq!(census[2].grid_width, n.div_ceil(32) as u64);
            assert_eq!(census[3].grid_width, 4);
            assert!(census.iter().all(|r| r.tg_threads == 128));
            // Establish candidate/incumbent equivalence before evaluating any
            // oracle tolerance, so a CPU discrepancy cannot hide N64 drift.
            f.assert_bits_equal("candidate/incumbent", &candidate, &incumbent);
            f.assert_bits_equal("candidate/capped", &candidate, &capped);
            eprintln!(
                "{dtype:?} K={k} M={m} N={n}: candidate/incumbent/capped bitwise equal before CPU oracle"
            );
            for (label, output) in [
                ("candidate", &candidate),
                ("incumbent", &incumbent),
                ("capped", &capped),
            ] {
                f.assert_output(label, output, &incumbent);
            }
            // IQ4_XS production uses the specialized incumbent; compare that
            // separately on valid routes (its legacy kernel has no id guards).
            if dtype == GgmlType::IQ4_XS && !invalid {
                let specialized = f.output(&ctx);
                one_shot(&ctx, |enc| {
                    encode_moe_down_iq4_xs_f32_grouped_slots(
                        &ctx,
                        enc,
                        &f.weight,
                        &f.inner,
                        &f.counts,
                        &f.ids,
                        &specialized,
                        k,
                        m,
                        COUNTS.len(),
                        n,
                    )
                })
                .unwrap();
                f.assert_bits_equal("candidate/specialized", &candidate, &specialized);
                f.assert_output("specialized", &specialized, &incumbent);
            }
            for (tensor, before) in inputs.iter().zip(before) {
                assert_eq!(tensor_backing_bytes(tensor), before);
            }
        }
    }
}

fn with_fault<R>(missing: bool, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            DOWN_WIDE_PREFLIGHT_FAULT.with(|v| v.set(self.0));
        }
    }
    let _restore = Restore(DOWN_WIDE_PREFLIGHT_FAULT.with(|v| v.replace(Some(missing))));
    f()
}

#[test]
fn down_n64_preflights_both_pipelines_and_bindings_before_dispatch() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let f = Fixture::new(&ctx, GgmlType::IQ3_S, 256, 65, 512, true);
    let out = f.output(&ctx);
    let cmd = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&cmd);
    for missing in [true, false] {
        dispatch_census_begin();
        assert!(with_fault(missing, || f.prepare(&ctx, &enc, &out)).is_err());
        assert!(dispatch_census_take().is_empty());
    }
    let mut bad_out = out.clone();
    bad_out.provenance = MetalTensorProvenance::OwnedWeightReadOnly;
    let mut misaligned = f.inner.clone();
    misaligned.offset += 4;
    let mut bad_counts = f.counts.clone();
    bad_counts.dtype = GgmlType::F32;
    let mut bad_ids = f.ids.clone();
    bad_ids.offset = bad_ids.buffer.length() as u64;
    let shared = offset_tensor(
        &ctx,
        16,
        &vec![0; (f.k + f.m) * SLOTS * 4],
        20,
        vec![((f.k + f.m) * SLOTS) as u64],
        GgmlType::F32,
    );
    let shared_inner = shared.view_subrange(0, vec![(f.k * SLOTS) as u64]);
    let alias = shared.view_subrange(0, vec![(f.m * SLOTS) as u64]);
    for (inner, counts, ids, output, k, experts, n) in [
        (
            &f.inner,
            &f.counts,
            &f.ids,
            &bad_out,
            f.k,
            COUNTS.len(),
            f.n,
        ),
        (&misaligned, &f.counts, &f.ids, &out, f.k, COUNTS.len(), f.n),
        (&f.inner, &bad_counts, &f.ids, &out, f.k, COUNTS.len(), f.n),
        (&f.inner, &f.counts, &bad_ids, &out, f.k, COUNTS.len(), f.n),
        (
            &shared_inner,
            &f.counts,
            &f.ids,
            &alias,
            f.k,
            COUNTS.len(),
            f.n,
        ),
        (&f.inner, &f.counts, &f.ids, &out, 255, COUNTS.len(), f.n),
        (&f.inner, &f.counts, &f.ids, &out, 0, COUNTS.len(), f.n),
        (&f.inner, &f.counts, &f.ids, &out, f.k, usize::MAX, f.n),
        (
            &f.inner,
            &f.counts,
            &f.ids,
            &out,
            f.k,
            COUNTS.len(),
            i32::MAX as usize,
        ),
    ] {
        dispatch_census_begin();
        let result = encode_moe_down_f32_grouped_slots_n64_split(
            &ctx, &enc, &f.weight, inner, counts, ids, output, k, f.m, experts, n,
        );
        assert!(
            matches!(result, Err(MetalError::BadShape { .. })),
            "{result:?}"
        );
        assert!(dispatch_census_take().is_empty());
    }
    enc.end();
    let enc = KernelEncoder::begin_concurrent(&cmd);
    assert!(f.prepare(&ctx, &enc, &out).is_err());
    enc.end();
}

#[test]
fn down_n64_narrow_only_span_keeps_stride_and_needs_no_wide_pipeline() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let f = Fixture::new(&ctx, GgmlType::IQ3_S, 256, 65, 512, true);
    let out = f.output(&ctx);
    for n in [1usize, 32, 33] {
        let ids = offset_tensor(
            &ctx,
            12,
            bytemuck::cast_slice(&vec![-1i32; n * COUNTS.len()]),
            20,
            vec![(n * COUNTS.len()) as u64],
            GgmlType::I32,
        );
        let cmd = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&cmd);
        let result = with_fault(true, || {
            prepare_moe_down_f32_grouped_slots_n64_split(
                &ctx,
                &enc,
                &f.weight,
                &f.inner,
                &f.counts,
                &ids,
                &out,
                f.k,
                f.m,
                COUNTS.len(),
                n,
            )
        });
        if n <= 32 {
            let p = result.unwrap();
            assert_eq!(p.args.n, n as u32);
            assert_eq!(p.panels.len(), 1);
            assert_eq!(p.panels[0].grid_width, 1);
            assert_eq!(p.panels[0].threadgroup_bytes, 8192);
            assert_eq!(p.panels[0].max_count, n as u32);
        } else {
            assert!(result.is_err());
        }
        enc.end();
    }
}

/// Encode-only census: false must honor the supplied production SmallCounts
/// policy at N512, while true replaces DOWN only. F32 is untouched either way.
#[test]
fn down_n64_override_preserves_smallcounts_control_and_f32_selection() {
    use crate::metal::expert::{
        GroupedDownPolicy, encode_grouped_routed_experts_f32x,
        encode_grouped_routed_experts_with_down_policy,
    };
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let weight = offset_tensor(
        &ctx,
        16,
        &bank_bytes(GgmlType::IQ3_S, 256, 256, 2),
        20,
        vec![256, 256, 2],
        GgmlType::IQ3_S,
    );
    let zeros = |shape: Vec<u64>, dtype| {
        let len = shape.iter().product::<u64>() as usize;
        offset_tensor(&ctx, 16, &vec![0; len * 4], 20, shape, dtype)
    };
    let input = zeros(vec![256, 512], GgmlType::F32);
    let ids = zeros(vec![1, 512], GgmlType::I32);
    let weights = zeros(vec![1, 512], GgmlType::F32);
    let counts = zeros(vec![2], GgmlType::I32);
    let slots = zeros(vec![1024], GgmlType::I32);
    let inner = zeros(vec![256, 512], GgmlType::F32);
    let slot_out = zeros(vec![256, 512], GgmlType::F32);
    let output = zeros(vec![256, 512], GgmlType::F32);
    let b = GroupedExperts {
        gate_bank: &weight,
        up_bank: &weight,
        down_bank: &weight,
        input: &input,
        ids: &ids,
        weights: &weights,
        counts: &counts,
        slots: &slots,
        inner: &inner,
        slot_out: &slot_out,
        output: &output,
    };
    for f32_operands in [false, true] {
        for enabled in [false, true] {
            let cmd = ctx.queue.commandBuffer().unwrap();
            let enc = KernelEncoder::begin(&cmd);
            dispatch_census_begin();
            with_grouped_down_wide_override(enabled, || {
                if f32_operands {
                    encode_grouped_routed_experts_f32x(&ctx, &enc, &b, 256, 256, 2, 1, 512, 10.0)
                        .unwrap();
                } else {
                    crate::metal::iq3_s_down_retile::with_production(|| {
                        encode_grouped_routed_experts_with_down_policy(
                            &ctx,
                            &enc,
                            &b,
                            256,
                            256,
                            2,
                            1,
                            512,
                            10.0,
                            GroupedDownPolicy::Iq3SSmallCounts,
                        )
                        .unwrap()
                    });
                }
            });
            let census = dispatch_census_take();
            let down: Vec<_> = census
                .iter()
                .filter(|r| r.kernel.starts_with("kernel_moe_down_"))
                .collect();
            if f32_operands {
                assert_eq!(down.len(), 1);
                assert!(down[0].kernel.contains("_f32x_"));
            } else if enabled {
                assert_eq!(down.len(), 2);
                assert!(down[0].kernel.ends_with("_generic"));
                assert_eq!(down[0].grid_width, 1);
                assert!(down[1].kernel.ends_with("_n64"));
                assert_eq!(down[1].grid_width, 8);
            } else {
                assert_eq!(down.len(), 2);
                assert!(down[0].kernel.ends_with("_m128_n16_retile"));
                assert!(down[1].kernel.ends_with("_generic"));
                assert_eq!(down[1].grid_width, 16);
            }
            enc.end();
        }
    }
    // A missing N64 pipeline fails before routing, not after the narrow pass.
    let cmd = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&cmd);
    dispatch_census_begin();
    let result = with_grouped_down_wide_override(true, || {
        with_fault(true, || {
            encode_grouped_routed_experts_with_down_policy(
                &ctx,
                &enc,
                &b,
                256,
                256,
                2,
                1,
                512,
                10.0,
                GroupedDownPolicy::Iq3SSmallCounts,
            )
        })
    });
    assert!(result.is_err());
    assert!(dispatch_census_take().is_empty());
    enc.end();
}

#[test]
fn down_n64_skips_negative_and_oversized_counts() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let f = Fixture::new(&ctx, GgmlType::IQ3_XXS, 256, 65, 97, true);
    let invalid: Vec<i32> = (0..COUNTS.len())
        .map(|i| if i % 2 == 0 { -1 } else { 98 })
        .collect();
    let counts = offset_tensor(
        &ctx,
        12,
        bytemuck::cast_slice(&invalid),
        20,
        vec![COUNTS.len() as u64],
        GgmlType::I32,
    );
    let out = f.output(&ctx);
    one_shot(&ctx, |enc| {
        encode_moe_down_f32_grouped_slots_n64_split(
            &ctx,
            enc,
            &f.weight,
            &f.inner,
            &counts,
            &f.ids,
            &out,
            f.k,
            f.m,
            COUNTS.len(),
            f.n,
        )
    })
    .unwrap();
    assert!(tensor_f32_at_offset(&out).iter().all(|&v| v == SENTINEL));
    assert_offset_guards(&out, 20, 28);
}
