//! Research M32/N32 gate/up: bitwise incumbent comparison, independent CPU
//! decoded Half dots, routing/guard coverage and encode-only failure census.
use super::*;
use crate::metal::test_support::{
    assert_offset_guards, dequant_expert, metal_test_context, offset_tensor, tensor_backing_bytes,
    tensor_f32_at_offset,
};

const SOURCE: &str = include_str!("../../../../../kernels/research/moe_swiglu_m32n32.metal");
const COUNTS: [usize; 8] = [0, 1, 15, 16, 17, 31, 32, 33];
const TOPK: usize = 8;
const SENTINEL: f32 = -777.0;
const CLAMP: f32 = 0.25;

#[test]
fn gate_up_m32_mapping_and_scoped_override() {
    assert_eq!(std::mem::size_of::<GateUpM32Args>(), 36);
    assert_eq!(SOURCE.matches("template [[host_name(").count(), 2);
    for layout in MOE_GROUPED_GENERIC_LAYOUTS {
        let name = moe_swiglu_m32n32_pipeline_name(layout.dtype);
        assert_eq!(
            name.is_some(),
            clamped_swiglu_n32_pipeline_name(layout.dtype).is_some()
        );
        if let Some(name) = name {
            assert!(SOURCE.contains(&format!("host_name(\"{name}\")")));
        }
    }
    let enabled = || GROUPED_GATE_UP_M32.with(std::cell::Cell::get);
    assert!(!enabled());
    let result = with_grouped_gate_up_m32_override(true, || {
        assert!(enabled());
        with_grouped_gate_up_m32_override(false, || assert!(!enabled()));
        let _ = std::panic::catch_unwind(|| {
            with_grouped_gate_up_m32_override(false, || panic!("restore"));
        });
        assert!(enabled());
        std::thread::spawn(|| assert!(!GROUPED_GATE_UP_M32.with(std::cell::Cell::get)))
            .join()
            .unwrap();
        7
    });
    assert_eq!(result, 7);
    assert!(!enabled());
}

fn bank_bytes(dtype: GgmlType, k: usize, m: usize, mut state: u32) -> Vec<u8> {
    let (block, bytes) = dtype.storage_layout().unwrap();
    let mut bank = vec![0; k * m * COUNTS.len() / block as usize * bytes as usize];
    for (ordinal, b) in bank.chunks_exact_mut(bytes as usize).enumerate() {
        for v in &mut b[2..] {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = (state >> 24) as u8;
        }
        let sign = if ordinal % 3 == 0 { -1.0 } else { 1.0 };
        b[..2].copy_from_slice(
            &half::f16::from_f32(sign * (ordinal % 7 + 1) as f32 / 4096.0).to_le_bytes(),
        );
    }
    bank
}

fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

#[derive(Clone, Copy, Default)]
struct Oracle {
    gate: f64,
    up: f64,
    abs_gate: f64,
    abs_up: f64,
}
impl Oracle {
    fn value(self) -> f64 {
        silu(self.gate.min(CLAMP as f64)) * self.up.clamp(-(CLAMP as f64), CLAMP as f64)
    }
    fn bound(self, k: usize) -> f64 {
        // Established Half-dot bound: exact F32 products, K rounded additions.
        // Clamp is 1-Lipschitz; |silu'| < 2 everywhere. Use |clamped up| <= L
        // to propagate the gate error, and |silu(clamped CPU gate)| for up.
        let ku = k as f64 * (f64::from(f32::EPSILON) / 2.0);
        let gamma = ku / (1.0 - ku);
        let dot = 2.0 * CLAMP as f64 * gamma * self.abs_gate
            + silu(self.gate.min(CLAMP as f64)).abs() * gamma * self.abs_up;
        // Separate F32 exp/divide/multiply epilogue allowance. This does not
        // divide by a cancellation-sensitive final result.
        dot + 16.0 * f64::from(f32::EPSILON) * (1.0 + self.value().abs())
    }
}

struct Fixture {
    k: usize,
    m: usize,
    n: usize,
    gate: MetalTensor,
    up: MetalTensor,
    x: MetalTensor,
    counts: MetalTensor,
    ids: MetalTensor,
    oracle: Vec<Oracle>,
    touched: Vec<bool>,
}
impl Fixture {
    fn new(ctx: &MetalContext, dtype: GgmlType, k: usize, m: usize, n: usize) -> Self {
        let gate = bank_bytes(dtype, k, m, 0xdead_beef);
        let up = bank_bytes(dtype, k, m, 0x1234_5678);
        let x: Vec<f32> = (0..k * n)
            .map(|i| ((i * 17 + i / 23) % 107) as f32 / 51.0 - 1.0)
            .collect();
        let counts = COUNTS.map(|c| c.min(n) as i32);
        let slots = n * TOPK;
        let mut ids = vec![i32::MAX; n * COUNTS.len()];
        let mut oracle = vec![Oracle::default(); slots * m];
        let mut touched = vec![false; slots];
        let mut next = 0;
        for (expert, &count) in counts.iter().enumerate() {
            let g = dequant_expert(&gate, dtype, k, m, expert);
            let u = dequant_expert(&up, dtype, k, m, expert);
            for j in 0..count as usize {
                let slot = (next * 37 + 11) % slots;
                next += 1;
                // M=1 exercises invalid IDs at panel ends. Larger M cases
                // keep the 17th/32nd/33rd slots valid to cover tail scatter.
                let id = match (count, j, m == 1) {
                    (17, 16, true) | (17, 0, false) => -1,
                    (32, 31, true) | (32, 1, false) => slots as i32,
                    (33, 32, true) | (33, 1, false) => i32::MAX,
                    _ => slot as i32,
                };
                ids[expert * n + j] = id;
                if id < 0 || id >= slots as i32 {
                    continue;
                }
                assert!(!touched[slot]);
                touched[slot] = true;
                for row in 0..m {
                    let o = &mut oracle[slot * m + row];
                    for kk in 0..k {
                        let xh = half::f16::from_f32(x[(slot / TOPK) * k + kk]).to_f64();
                        let gp = half::f16::from_f32(g[row * k + kk]).to_f64() * xh;
                        let up = half::f16::from_f32(u[row * k + kk]).to_f64() * xh;
                        assert_eq!(f64::from(gp as f32), gp, "Half product must be exact F32");
                        assert_eq!(f64::from(up as f32), up, "Half product must be exact F32");
                        o.gate += gp;
                        o.up += up;
                        o.abs_gate += gp.abs();
                        o.abs_up += up.abs();
                    }
                }
            }
        }
        Self {
            k,
            m,
            n,
            gate: offset_tensor(
                ctx,
                18,
                &gate,
                22,
                vec![k as u64, m as u64, COUNTS.len() as u64],
                dtype,
            ),
            up: offset_tensor(
                ctx,
                22,
                &up,
                18,
                vec![k as u64, m as u64, COUNTS.len() as u64],
                dtype,
            ),
            x: offset_tensor(
                ctx,
                16,
                bytemuck::cast_slice(&x),
                20,
                vec![(k * n) as u64],
                GgmlType::F32,
            ),
            counts: offset_tensor(
                ctx,
                12,
                bytemuck::cast_slice(&counts),
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
            oracle,
            touched,
        }
    }
    fn output(&self, ctx: &MetalContext) -> MetalTensor {
        offset_tensor(
            ctx,
            20,
            bytemuck::cast_slice(&vec![SENTINEL; self.n * TOPK * self.m]),
            28,
            vec![(self.n * TOPK * self.m) as u64],
            GgmlType::F32,
        )
    }
    fn encode(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        out: &MetalTensor,
        enabled: bool,
        threshold: u32,
    ) -> Result<(), MetalError> {
        with_grouped_gate_up_m32_override(enabled, || {
            encode_moe_swiglu_clamped_f32_grouped_slots_split(
                ctx,
                enc,
                &self.gate,
                &self.up,
                &self.x,
                &self.counts,
                &self.ids,
                out,
                self.k,
                self.m,
                COUNTS.len(),
                TOPK,
                self.n,
                CLAMP,
                threshold,
            )
        })
    }
    fn assert_output(&self, candidate: &MetalTensor, incumbent: &MetalTensor) {
        let a = tensor_f32_at_offset(candidate);
        let b = tensor_f32_at_offset(incumbent);
        // Check bits first: failures distinguish changed kernel order from a
        // CPU decoder/rounding disagreement shared by both GPU paths.
        for (i, (&a, &b)) in a.iter().zip(&b).enumerate() {
            let o = self.oracle[i];
            let cpu = if self.touched[i / self.m] {
                o.value()
            } else {
                SENTINEL as f64
            };
            let detail = format!(
                "{:?} K={} M={} N={} slot={} row={} candidate={a:.12e} incumbent={b:.12e} CPU={cpu:.12e} candidate-CPU={:.12e} incumbent-CPU={:.12e} gate={} up={} abs-products={}/{} bound={:.12e}",
                self.gate.dtype,
                self.k,
                self.m,
                self.n,
                i / self.m,
                i % self.m,
                a as f64 - cpu,
                b as f64 - cpu,
                o.gate,
                o.up,
                o.abs_gate,
                o.abs_up,
                o.bound(self.k)
            );
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "MMA/half/clamp bit equivalence: {detail}"
            );
            if self.touched[i / self.m] {
                assert!(
                    a.is_finite() && (a as f64 - cpu).abs() <= o.bound(self.k),
                    "independent CPU Half-dot: {detail}"
                );
            } else {
                assert_eq!(a.to_bits(), SENTINEL.to_bits(), "unwritten slot: {detail}");
            }
        }
        assert_offset_guards(candidate, 20, 28);
        assert_offset_guards(incumbent, 20, 28);
    }
}

#[test]
fn gate_up_m32_boundaries_half_dot_incumbent_bits_and_guards() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    for dtype in [GgmlType::IQ2_S, GgmlType::IQ3_S] {
        let name = moe_swiglu_m32n32_pipeline_name(dtype).unwrap();
        assert!(
            ctx.library
                .newFunctionWithName(&NSString::from_str(name))
                .is_none(),
            "research entry leaked into product metallib"
        );
        for (k, m, n) in [
            (256, 1, 33),
            (256, 31, 33),
            (512, 32, 33),
            (512, 33, 33),
            (256, 65, 512),
        ] {
            let f = Fixture::new(&ctx, dtype, k, m, n);
            let inputs = [&f.gate, &f.up, &f.x, &f.counts, &f.ids];
            let before: Vec<_> = inputs.iter().map(|t| tensor_backing_bytes(t)).collect();
            let candidate = f.output(&ctx);
            let incumbent = f.output(&ctx);
            let n16 = f.output(&ctx);
            dispatch_census_begin();
            one_shot(&ctx, |enc| {
                // A supplied threshold other than 17 proves the experimental
                // fixed split is independent of the production policy.
                f.encode(&ctx, enc, &candidate, true, 32)?;
                f.encode(&ctx, enc, &incumbent, false, 17)?;
                f.encode(&ctx, enc, &n16, false, u32::MAX)
            })
            .unwrap();
            let census = dispatch_census_take();
            assert_eq!(census.len(), 5);
            assert_eq!(
                census[0].kernel,
                clamped_swiglu_pipeline_name(dtype).unwrap()
            );
            assert_eq!(census[0].grid_width, n.div_ceil(16) as u64);
            assert_eq!(census[0].grid_height, m.div_ceil(64) as u64);
            assert_eq!(census[1].kernel, name);
            assert_eq!(census[1].grid_width, n.div_ceil(32) as u64);
            assert_eq!(census[1].grid_height, m.div_ceil(32) as u64);
            assert_eq!(census[1].tg_threads, 128);
            assert_eq!(
                census[2].kernel,
                clamped_swiglu_pipeline_name(dtype).unwrap()
            );
            assert_eq!(
                census[3].kernel,
                clamped_swiglu_n32_pipeline_name(dtype).unwrap()
            );
            assert_eq!(census[3].grid_height, m.div_ceil(64) as u64);
            assert_eq!(
                census[4].kernel,
                clamped_swiglu_pipeline_name(dtype).unwrap()
            );
            f.assert_output(&candidate, &incumbent);
            f.assert_output(&candidate, &n16);
            for (t, bytes) in inputs.into_iter().zip(before) {
                assert_eq!(tensor_backing_bytes(t), bytes);
            }
            if m > 1 {
                assert!(f.oracle.iter().any(|o| o.gate > CLAMP as f64));
                assert!(f.oracle.iter().any(|o| o.up > CLAMP as f64));
                assert!(f.oracle.iter().any(|o| o.up < -(CLAMP as f64)));
                assert!(f.oracle.iter().any(|o| o.gate < -(CLAMP as f64)));
            }
        }
    }
}

fn with_fault<R>(fault: GateUpM32Fault, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<GateUpM32Fault>);
    impl Drop for Restore {
        fn drop(&mut self) {
            GATE_UP_M32_PREFLIGHT_FAULT.with(|v| v.set(self.0));
        }
    }
    let _restore = Restore(GATE_UP_M32_PREFLIGHT_FAULT.with(|v| v.replace(Some(fault))));
    f()
}

#[test]
fn gate_up_m32_span_edges_and_narrow_only_preflight() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    for n in [1, 16, 17, 32, 33] {
        let f = Fixture::new(&ctx, GgmlType::IQ3_S, 256, 33, n);
        let candidate = f.output(&ctx);
        let incumbent = f.output(&ctx);
        if n <= 16 {
            one_shot(&ctx, |enc| {
                with_fault(GateUpM32Fault::MissingPipeline, || {
                    f.encode(&ctx, enc, &candidate, true, 17)
                })
            })
            .unwrap();
        } else {
            one_shot(&ctx, |enc| f.encode(&ctx, enc, &candidate, true, 17)).unwrap();
        }
        one_shot(&ctx, |enc| f.encode(&ctx, enc, &incumbent, false, 17)).unwrap();
        f.assert_output(&candidate, &incumbent);
    }
}

#[test]
fn gate_up_m32_preflight_device_capacity_pipeline_and_bindings_are_atomic() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let f = Fixture::new(&ctx, GgmlType::IQ3_S, 256, 33, 33);
    let out = f.output(&ctx);
    let cmd = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&cmd);
    for fault in [
        GateUpM32Fault::MissingPipeline,
        GateUpM32Fault::Capacity,
        GateUpM32Fault::Device,
    ] {
        dispatch_census_begin();
        assert!(with_fault(fault, || f.encode(&ctx, &enc, &out, true, 17)).is_err());
        assert!(
            dispatch_census_take().is_empty(),
            "preflight must precede even the narrow dispatch"
        );
    }
    let mut read_only = out.clone();
    read_only.provenance = MetalTensorProvenance::OwnedWeightReadOnly;
    let mut misaligned = f.x.clone();
    misaligned.offset += 4;
    let mut wrong_counts = f.counts.clone();
    wrong_counts.dtype = GgmlType::F32;
    let mut outside = f.ids.clone();
    outside.offset = outside.buffer.length() as u64;
    let shared = offset_tensor(
        &ctx,
        16,
        &vec![0; (f.k * f.n + f.m * f.n * TOPK) * 4],
        20,
        vec![(f.k * f.n + f.m * f.n * TOPK) as u64],
        GgmlType::F32,
    );
    let alias_x = shared.view_subrange(0, vec![(f.k * f.n) as u64]);
    let alias_out = shared.view_subrange(0, vec![(f.m * f.n * TOPK) as u64]);
    for (x, counts, ids, out, k, experts, n, topk, clamp) in [
        (
            &f.x,
            &f.counts,
            &f.ids,
            &read_only,
            f.k,
            COUNTS.len(),
            f.n,
            TOPK,
            CLAMP,
        ),
        (
            &misaligned,
            &f.counts,
            &f.ids,
            &out,
            f.k,
            COUNTS.len(),
            f.n,
            TOPK,
            CLAMP,
        ),
        (
            &f.x,
            &wrong_counts,
            &f.ids,
            &out,
            f.k,
            COUNTS.len(),
            f.n,
            TOPK,
            CLAMP,
        ),
        (
            &f.x,
            &f.counts,
            &outside,
            &out,
            f.k,
            COUNTS.len(),
            f.n,
            TOPK,
            CLAMP,
        ),
        (
            &alias_x,
            &f.counts,
            &f.ids,
            &alias_out,
            f.k,
            COUNTS.len(),
            f.n,
            TOPK,
            CLAMP,
        ),
        (
            &f.x,
            &f.counts,
            &f.ids,
            &out,
            255,
            COUNTS.len(),
            f.n,
            TOPK,
            CLAMP,
        ),
        (
            &f.x,
            &f.counts,
            &f.ids,
            &out,
            0,
            COUNTS.len(),
            f.n,
            TOPK,
            CLAMP,
        ),
        (
            &f.x,
            &f.counts,
            &f.ids,
            &out,
            f.k,
            usize::MAX,
            f.n,
            TOPK,
            CLAMP,
        ),
        (
            &f.x,
            &f.counts,
            &f.ids,
            &out,
            f.k,
            COUNTS.len(),
            i32::MAX as usize,
            TOPK,
            CLAMP,
        ),
        (
            &f.x,
            &f.counts,
            &f.ids,
            &out,
            f.k,
            COUNTS.len(),
            f.n,
            i32::MAX as usize,
            CLAMP,
        ),
        (
            &f.x,
            &f.counts,
            &f.ids,
            &out,
            f.k,
            COUNTS.len(),
            f.n,
            TOPK,
            f32::NAN,
        ),
    ] {
        dispatch_census_begin();
        let result = encode_moe_swiglu_m32_split(
            &ctx, &enc, &f.gate, &f.up, x, counts, ids, out, k, f.m, experts, topk, n, clamp,
        );
        assert!(
            matches!(result, Err(MetalError::BadShape { .. })),
            "{result:?}"
        );
        assert!(dispatch_census_take().is_empty());
    }
    enc.end();
    let enc = KernelEncoder::begin_concurrent(&cmd);
    dispatch_census_begin();
    assert!(f.encode(&ctx, &enc, &out, true, 17).is_err());
    assert!(dispatch_census_take().is_empty());
    enc.end();
}

#[test]
fn gate_up_m32_override_leaves_exact_f32_and_other_recipes_unchanged() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    for dtype in [GgmlType::IQ2_S, GgmlType::IQ3_S] {
        let f = Fixture::new(&ctx, dtype, 256, 33, 33);
        let outputs = [f.output(&ctx), f.output(&ctx)];
        for (enabled, out) in [false, true].into_iter().zip(&outputs) {
            dispatch_census_begin();
            one_shot(&ctx, |enc| {
                with_grouped_gate_up_m32_override(enabled, || {
                    encode_moe_swiglu_clamped_f32x_grouped_slots(
                        &ctx,
                        enc,
                        &f.gate,
                        &f.up,
                        &f.x,
                        &f.counts,
                        &f.ids,
                        out,
                        f.k,
                        f.m,
                        COUNTS.len(),
                        TOPK,
                        f.n,
                        CLAMP,
                    )
                })
            })
            .unwrap();
            let census = dispatch_census_take();
            assert_eq!(census.len(), 1);
            assert!(census[0].kernel.contains("_f32x_"));
        }
        assert_eq!(
            tensor_backing_bytes(&outputs[0]),
            tensor_backing_bytes(&outputs[1])
        );
    }
    // Non-fast and mixed capability retain Down + UpClamped unchanged.
    for (gate_dtype, up_dtype) in [
        (GgmlType::IQ3_XXS, GgmlType::IQ3_XXS),
        (GgmlType::IQ2_S, GgmlType::IQ3_S),
    ] {
        let mut f = Fixture::new(&ctx, gate_dtype, 256, 33, 33);
        if up_dtype != gate_dtype {
            f.up = offset_tensor(
                &ctx,
                22,
                &bank_bytes(up_dtype, f.k, f.m, 0x1234_5678),
                18,
                vec![f.k as u64, f.m as u64, COUNTS.len() as u64],
                up_dtype,
            );
        }
        let outputs = [f.output(&ctx), f.output(&ctx)];
        let mut names = Vec::new();
        for (enabled, out) in [false, true].into_iter().zip(&outputs) {
            dispatch_census_begin();
            one_shot(&ctx, |enc| f.encode(&ctx, enc, out, enabled, 17)).unwrap();
            names.push(
                dispatch_census_take()
                    .into_iter()
                    .map(|r| r.kernel)
                    .collect::<Vec<_>>(),
            );
        }
        assert_eq!(names[0], names[1]);
        assert_eq!(names[0].len(), 2);
        assert!(names[0].iter().all(|n| !n.contains("m32n32")));
        assert_eq!(
            tensor_backing_bytes(&outputs[0]),
            tensor_backing_bytes(&outputs[1])
        );
    }
}

#[test]
fn gate_up_m32_invalid_counts_write_nothing() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    // Both narrow-only spans and the split reject counts beyond the IDs row.
    for n in [1, 16, 33] {
        let mut f = Fixture::new(&ctx, GgmlType::IQ2_S, 256, 33, n);
        let invalid = [
            -1,
            i32::MIN,
            i32::MAX,
            n as i32 + 1,
            0,
            -2,
            i32::MAX,
            n as i32 + 1,
        ];
        f.counts = offset_tensor(
            &ctx,
            12,
            bytemuck::cast_slice(&invalid),
            20,
            vec![COUNTS.len() as u64],
            GgmlType::I32,
        );
        let out = f.output(&ctx);
        let before = tensor_backing_bytes(&out);
        one_shot(&ctx, |enc| f.encode(&ctx, enc, &out, true, 17)).unwrap();
        assert_eq!(
            tensor_backing_bytes(&out),
            before,
            "invalid counts at N={n}"
        );
    }
}

#[test]
fn gate_up_m32_cpu_bound_handles_cancellation_but_rejects_material_error() {
    // The established IQ3_S cancellation example from the DOWN regression.
    // Propagate it through an unclamped negative gate and saturated up value.
    let oracle = Oracle {
        gate: -0.004048764705657959,
        up: 0.5,
        abs_gate: 21.779911175370216,
        abs_up: 1.0,
    };
    let ku = 256.0 * (f64::from(f32::EPSILON) / 2.0);
    let gate_bound = ku / (1.0 - ku) * oracle.abs_gate;
    for delta in [-gate_bound, gate_bound] {
        let rounded = silu(oracle.gate + delta) * CLAMP as f64;
        assert!((rounded - oracle.value()).abs() <= oracle.bound(256));
    }
    assert!(oracle.bound(256) < 0.001);
}
