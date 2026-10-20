use super::*;
use crate::metal::test_support::{
    assert_offset_guards, metal_test_context, offset_tensor, tensor_backing_bytes,
    tensor_f32_at_offset,
};

const PREFIX: usize = 20;
const SENTINEL: f32 = -777.0;

#[path = "grouped_f32_glm_bench.rs"]
mod glm_bench;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GroupedF32TestVariant {
    Scalar,
    Production,
}

// The control must not re-enter automatic selection or inherit a force-MMA
// scope. Both arms share the production validation, preflight and scalar tail.
pub(crate) fn encode_mat_mat_grouped_f32_for_test(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    input: &MetalTensor,
    output: &MetalTensor,
    k: usize,
    m: usize,
    groups: usize,
    rows: usize,
    variant: GroupedF32TestVariant,
) -> Result<(), MetalError> {
    if weight.dtype != GgmlType::F32 {
        return Err(bad_shape("grouped_f32_test", "requires F32 weights"));
    }
    match variant {
        GroupedF32TestVariant::Scalar => encode_mat_mat_grouped_f32_inner(
            ctx,
            enc,
            weight,
            input,
            output,
            k,
            m,
            groups,
            rows,
            false,
            Some(true),
        ),
        GroupedF32TestVariant::Production => {
            encode_mat_mat_grouped_f32(ctx, enc, weight, input, output, k, m, groups, rows)
        }
    }
}

#[test]
fn grouped_f32_mma_batched_f64_and_row_tail() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let variant = GroupedF32TestVariant::Production;
    for k in [256, 512, 640, 1024] {
        for groups in [1, 7, 8, 32] {
            exercise(
                &ctx,
                Shape {
                    k,
                    m: 512,
                    groups,
                    rows: 64,
                },
                1.0,
                true,
                28,
                variant,
            );
        }
    }
    for rows in [64, 65, 129] {
        for scale in [1.0, 131_072.0, 1.0 / 1_048_576.0] {
            exercise(
                &ctx,
                Shape {
                    k: 288,
                    m: 512,
                    groups: 8,
                    rows,
                },
                scale,
                false,
                0,
                variant,
            );
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Shape {
    k: usize,
    m: usize,
    groups: usize,
    rows: usize,
}

impl Shape {
    fn mma(self) -> bool {
        self.k >= 256
            && self.k % 32 == 0
            && self.m % 16 == 0
            && self.groups >= 8
            && self.rows >= 64
            && (self.m / 16) * (self.rows / 64) * self.groups >= 256
    }
}

#[test]
fn grouped_f32_mma_production_policy_floor_and_scalar_control() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    for (k, m, groups, rows) in [
        (256, 512, 1, 64),
        (256, 512, 7, 65),
        (256, 512, 8, 64),
        (256, 512, 8, 65),
        (256, 512, 32, 65),
        (1024, 512, 8, 65),
        (256, 512, 8, 63),
        (224, 512, 8, 64),
        (288, 513, 8, 65),
        (256, 16, 8, 64),
        (256, 496, 8, 65),
        (256, 256, 16, 65),
        (256, 16, 8, 2048),
    ] {
        let s = Shape { k, m, groups, rows };
        let mut f = Fixture::new(&ctx, s, 1.0, false, 28, GroupedF32TestVariant::Production);
        f.weight.shape = vec![k as u64, (m * groups) as u64];
        assert_eq!(grouped_projection_scalar_baseline(), None);
        f.checked_dispatch(&ctx, false);
        // Even an enclosing force-MMA diagnostic cannot corrupt this control.
        with_grouped_projection_scalar_baseline_override(false, || f.checked_dispatch(&ctx, true));
        f.check_values();
        for allow in [false, true] {
            dispatch_census_begin();
            one_shot(&ctx, |enc| {
                encode_mat_mat_grouped_f32_with_policy(
                    &ctx, enc, &f.weight, &f.input, &f.output, k, m, groups, rows, allow,
                )
            })
            .unwrap();
            assert_census(
                &dispatch_census_take(),
                s,
                if allow {
                    GroupedF32TestVariant::Production
                } else {
                    GroupedF32TestVariant::Scalar
                },
            );
            f.check_values();
        }
    }
}

#[test]
fn grouped_f32_mma_geometry_and_preflight_fail_closed() {
    let valid = GroupedProjectionArgs {
        k: 256,
        m: 512,
        groups: 8,
        rows: 64,
        row_bytes: 1024,
        input_stride: 2048,
        output_stride: 4096,
    };
    assert!(grouped_projection_f32_mma_eligible(GgmlType::F32, &valid));
    for dtype in GROUPED_PROJECTION_DTYPES
        .iter()
        .filter(|&&d| d != GgmlType::F32)
    {
        assert!(!grouped_projection_f32_mma_eligible(*dtype, &valid));
    }
    for args in [
        GroupedProjectionArgs { k: 255, ..valid },
        GroupedProjectionArgs { k: 257, ..valid },
        GroupedProjectionArgs { m: 0, ..valid },
        GroupedProjectionArgs { m: 511, ..valid },
        GroupedProjectionArgs { groups: 7, ..valid },
        GroupedProjectionArgs { rows: 63, ..valid },
        GroupedProjectionArgs {
            m: 496,
            rows: 127,
            ..valid
        },
        GroupedProjectionArgs {
            m: u32::MAX - 15,
            groups: u32::MAX,
            rows: u32::MAX,
            ..valid
        },
    ] {
        assert!(!grouped_projection_f32_mma_eligible(GgmlType::F32, &args));
    }
    // F32's zero dynamic scratch still requires the pipeline's static capacity.
    grouped_projection_scalar_capacity(32, 128, 0, 0).unwrap();
    grouped_projection_scalar_capacity(32, 256, 1024, 1024).unwrap();
    for (simd, threads, static_bytes, device_bytes) in [
        (16, 128, 0, 32768),
        (32, 127, 0, 32768),
        (32, 128, 1025, 1024),
    ] {
        assert!(
            grouped_projection_scalar_capacity(simd, threads, static_bytes, device_bytes).is_err()
        );
    }
    for error in [
        MetalError::NoFunction(GROUPED_F32_MMA_KERNEL.into()),
        MetalError::Pipeline(
            GROUPED_F32_MMA_KERNEL.into(),
            "unavailable candidate".into(),
        ),
    ] {
        assert!(
            grouped_projection_scalar_preflight(Err(error), 32768)
                .ok()
                .is_none()
        );
    }
    let source = include_str!("../../../../../kernels/latent_grouped_mma.metal");
    assert!(source.contains(GROUPED_F32_MMA_KERNEL));
    assert!(!source.contains("m8n32k32"));
    assert!(!source.contains("simdgroup_half"));
}

fn f32_tensor(ctx: &MetalContext, values: &[f32], shape: Vec<u64>, suffix: usize) -> MetalTensor {
    offset_tensor(
        ctx,
        PREFIX,
        bytemuck::cast_slice(values),
        suffix,
        shape,
        GgmlType::F32,
    )
}

struct Fixture {
    shape: Shape,
    variant: GroupedF32TestVariant,
    weights: Vec<f32>,
    x: Vec<f32>,
    weight: MetalTensor,
    input: MetalTensor,
    output: MetalTensor,
    scalar: MetalTensor,
    scale: f32,
    prefix: usize,
    suffix: usize,
}

impl Fixture {
    fn new(
        ctx: &MetalContext,
        s: Shape,
        scale: f32,
        positive: bool,
        suffix: usize,
        variant: GroupedF32TestVariant,
    ) -> Self {
        Self::with_prefix(ctx, s, scale, positive, PREFIX, suffix, variant)
    }

    fn with_prefix(
        ctx: &MetalContext,
        s: Shape,
        scale: f32,
        positive: bool,
        prefix: usize,
        suffix: usize,
        variant: GroupedF32TestVariant,
    ) -> Self {
        let f32_tensor = |ctx, values: &[f32], shape, suffix| {
            offset_tensor(
                ctx,
                prefix,
                bytemuck::cast_slice(values),
                suffix,
                shape,
                GgmlType::F32,
            )
        };
        let weights: Vec<f32> = (0..s.k * s.m * s.groups)
            .map(|i| {
                let lane = i % s.k;
                let channel = (i / s.k) % s.m;
                let group = i / (s.k * s.m);
                match channel % 4 {
                    // Both operands contain F32 distinctions below half's ULP.
                    0 => {
                        if lane % 2 == 0 {
                            1.0 + 1.0 / 8192.0
                        } else {
                            -1.0
                        }
                    }
                    1 => {
                        if lane % 2 == 0 {
                            1.0
                        } else {
                            -1.0
                        }
                    }
                    _ => {
                        ((lane * 17 + channel * 13 + group * 7) % 61) as f32 / 32.0 - 0.9375
                            + (group + 1) as f32 / 8192.0
                    }
                }
            })
            .collect();
        let x: Vec<f32> = (0..s.k * s.groups * s.rows)
            .map(|i| {
                let lane = i % s.k;
                let group = (i / s.k) % s.groups;
                let row = i / (s.k * s.groups);
                let base = 0.5 + ((lane / 2 * 7 + group * 11 + row * 3) % 17) as f32 / 32.0;
                let value = base + if lane % 2 == 0 { 1.0 / 8192.0 } else { 0.0 };
                let sign = if positive || (lane / 2 + row + group) % 3 != 0 {
                    1.0
                } else {
                    -1.0
                };
                // Positive mode resembles attention probabilities, without a
                // softmax dependency. Pairwise cancellation remains observable.
                value * sign * scale / if positive { s.k as f32 } else { 1.0 }
            })
            .collect();
        let weight = f32_tensor(
            ctx,
            &weights,
            vec![s.k as u64, s.m as u64, s.groups as u64],
            suffix,
        );
        let input = f32_tensor(
            ctx,
            &x,
            vec![(s.k * s.groups) as u64, s.rows as u64],
            suffix,
        );
        let output_shape = vec![(s.m * s.groups) as u64, s.rows as u64];
        let blank = vec![SENTINEL; s.m * s.groups * s.rows];
        let output = f32_tensor(ctx, &blank, output_shape.clone(), suffix);
        let scalar = f32_tensor(ctx, &blank, output_shape, suffix);
        Self {
            shape: s,
            variant,
            weights,
            x,
            weight,
            input,
            output,
            scalar,
            scale,
            prefix,
            suffix,
        }
    }

    fn encode(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        scalar: bool,
    ) -> Result<(), MetalError> {
        let s = self.shape;
        encode_mat_mat_grouped_f32_for_test(
            ctx,
            enc,
            &self.weight,
            &self.input,
            if scalar { &self.scalar } else { &self.output },
            s.k,
            s.m,
            s.groups,
            s.rows,
            if scalar {
                GroupedF32TestVariant::Scalar
            } else {
                self.variant
            },
        )
    }

    fn checked_dispatch(&self, ctx: &MetalContext, scalar: bool) {
        dispatch_census_begin();
        let result = one_shot(ctx, |enc| self.encode(ctx, enc, scalar));
        let census = dispatch_census_take();
        result.unwrap();
        assert_census(
            &census,
            self.shape,
            if scalar {
                GroupedF32TestVariant::Scalar
            } else {
                self.variant
            },
        );
    }

    fn check_values(&self) {
        let s = self.shape;
        let got = tensor_f32_at_offset(&self.output);
        let scalar = tensor_f32_at_offset(&self.scalar);
        assert_eq!(got.len(), s.m * s.groups * s.rows);
        for (i, (&a, &b)) in got.iter().zip(&scalar).enumerate() {
            assert!(a.is_finite() && b.is_finite(), "{s:?} index={i}: nonfinite");
            assert_ne!(a, SENTINEL, "{s:?} index={i}: unwritten candidate");
            assert_ne!(b, SENTINEL, "{s:?} index={i}: unwritten scalar");
            assert_close(
                &format!("{s:?} scalar comparison index={i}"),
                a,
                f64::from(b),
                self.scale,
            );
        }
        let tile = 64;
        // Bound debug-build CPU work: all channels on the final row, boundary
        // rows/channels elsewhere. Small grouped fixtures use the full oracle.
        for row in 0..s.rows {
            for group in 0..s.groups {
                for channel in 0..s.m {
                    if s.m > 32
                        && row != s.rows - 1
                        && !([0, 1, tile - 1, tile, tile + 1, s.rows / 2].contains(&row)
                            && [0, 1, 2, 3, 7, 8, 15, 16, s.m / 2, s.m - 1].contains(&channel))
                    {
                        continue;
                    }
                    let w = &self.weights[(group * s.m + channel) * s.k..][..s.k];
                    let x = &self.x[(row * s.groups + group) * s.k..][..s.k];
                    let expected: f64 = w
                        .iter()
                        .zip(x)
                        .map(|(&a, &b)| f64::from(a) * f64::from(b))
                        .sum();
                    let index = (row * s.groups + group) * s.m + channel;
                    let label = format!("{s:?} row={row} group={group} channel={channel}");
                    assert_close(&label, got[index], expected, self.scale);
                    assert_close(
                        &format!("scalar {label}"),
                        scalar[index],
                        expected,
                        self.scale,
                    );
                }
            }
        }
        assert_offset_guards(&self.output, self.prefix, self.suffix);
        assert_offset_guards(&self.scalar, self.prefix, self.suffix);
    }
}

fn assert_close(label: &str, got: f32, expected: f64, scale: f32) {
    assert!(
        got.is_finite() && expected.is_finite(),
        "{label}: nonfinite"
    );
    assert!(
        (f64::from(got) - expected).abs() <= 2e-5 * expected.abs().max(f64::from(scale)),
        "{label}: {got} vs {expected}, activation_scale={scale}"
    );
}

fn assert_census(census: &[DispatchCensusRow], s: Shape, variant: GroupedF32TestVariant) {
    let (tile_m, tile, kernel) = (16, 64, GROUPED_F32_MMA_KERNEL);
    let mma = variant == GroupedF32TestVariant::Production && s.mma();
    let remainder = if mma { s.rows % tile } else { s.rows };
    assert_eq!(
        census.len(),
        usize::from(mma) + usize::from(remainder > 0),
        "{s:?}: {census:?}"
    );
    if mma {
        // Do not treat missing/unsupported MMA as a successful device test.
        assert_eq!(census[0].kernel, kernel, "{s:?}: MMA required");
        assert_eq!(
            (
                census[0].grid_width,
                census[0].grid_height,
                census[0].grid_depth
            ),
            (
                (s.rows / tile) as u64,
                (s.m / tile_m) as u64,
                s.groups as u64
            )
        );
    }
    if remainder > 0 {
        let tail = &census[usize::from(mma)];
        assert_eq!(
            tail.kernel,
            grouped_projection_pipeline_name(GgmlType::F32).unwrap()
        );
        assert_eq!(
            (tail.grid_width, tail.grid_height, tail.grid_depth),
            (
                s.m.div_ceil(4) as u64,
                remainder.div_ceil(4) as u64,
                s.groups as u64
            )
        );
    }
    for dispatch in census {
        assert_eq!(
            (
                dispatch.threads_width,
                dispatch.threads_height,
                dispatch.threads_depth
            ),
            (128, 1, 1)
        );
        assert_eq!(dispatch.tg_threads, 128);
        assert!(!dispatch.encoder_concurrent);
    }
}

fn exercise(
    ctx: &MetalContext,
    s: Shape,
    scale: f32,
    positive: bool,
    suffix: usize,
    variant: GroupedF32TestVariant,
) {
    let f = Fixture::new(ctx, s, scale, positive, suffix, variant);
    let weight_before = tensor_backing_bytes(&f.weight);
    let input_before = tensor_backing_bytes(&f.input);
    f.checked_dispatch(ctx, false);
    f.checked_dispatch(ctx, true);
    f.check_values();
    assert_eq!(tensor_backing_bytes(&f.weight), weight_before);
    assert_eq!(tensor_backing_bytes(&f.input), input_before);
    let max_abs = crate::compare::assert_max_abs_diff_f32(
        "grouped F32 production versus scalar diagnostic",
        tensor_f32_at_offset(&f.output),
        tensor_f32_at_offset(&f.scalar),
    );
    eprintln!(
        "F32 correctness {variant:?} {s:?} scale={scale}: max_abs_vs_scalar={max_abs:e} (not performance)"
    );
}

#[test]
fn grouped_f32_mma_fixtures_detect_half_narrowing_of_either_operand() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let variant = GroupedF32TestVariant::Production;
    let s = Shape {
        k: 256,
        m: 512,
        groups: 8,
        rows: 64,
    };
    let f = Fixture::new(&ctx, s, 1.0, false, 0, variant);
    for (channel, narrow_weight) in [(0, true), (1, false)] {
        let w = &f.weights[channel * s.k..][..s.k];
        let x = &f.x[..s.k];
        let exact: f64 = w
            .iter()
            .zip(x)
            .map(|(&a, &b)| f64::from(a) * f64::from(b))
            .sum();
        let narrowed: f64 = w
            .iter()
            .zip(x)
            .map(|(&a, &b)| {
                if narrow_weight {
                    half::f16::from_f32(a).to_f64() * f64::from(b)
                } else {
                    f64::from(a) * half::f16::from_f32(b).to_f64()
                }
            })
            .sum();
        assert!(
            (exact - narrowed).abs() > 2e-5 * exact.abs().max(1.0),
            "fixture must reject half narrowing: channel={channel} exact={exact} half={narrowed}"
        );
    }
    f.checked_dispatch(&ctx, false);
    f.checked_dispatch(&ctx, true);
    f.check_values();
}

#[test]
fn grouped_f32_mma_bad_bindings_fail_before_any_prefix_dispatch() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    for variant in [
        GroupedF32TestVariant::Scalar,
        GroupedF32TestVariant::Production,
    ] {
        let tile = 64;
        let s = Shape {
            k: 256,
            m: 512,
            groups: 8,
            rows: tile + 1,
        };
        let f = Fixture::new(&ctx, s, 1.0, false, 0, variant);
        let original = [f.weight.clone(), f.input.clone(), f.output.clone()];
        let before = original.each_ref().map(tensor_backing_bytes);
        let command = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&command);
        let encode = |t: &[MetalTensor; 3], s: Shape| {
            encode_mat_mat_grouped_f32_for_test(
                &ctx, &enc, &t[0], &t[1], &t[2], s.k, s.m, s.groups, s.rows, variant,
            )
        };
        dispatch_census_begin();
        for which in 0..3 {
            for fault in 0..6 {
                let mut t = original.clone();
                match fault {
                    0 => t[which].offset += 1,
                    1 => t[which].offset = t[which].buffer.length() as u64,
                    2 => t[which].offset = u64::MAX - 3,
                    3 => t[which].dtype = GgmlType::I32,
                    4 => t[which].shape = vec![1],
                    _ => t[which].shape = vec![u64::MAX, u64::MAX, 2],
                }
                assert!(encode(&t, s).is_err(), "binding={which} fault={fault}");
            }
        }
        for which in [1, 2] {
            let mut t = original.clone();
            t[which].dtype = GgmlType::F16;
            assert!(encode(&t, s).is_err());
            // A complete prefix fits physically, but the declared scalar tail does
            // not. Whole-call validation must precede the first MMA dispatch.
            let width = if which == 1 {
                s.k * s.groups
            } else {
                s.m * s.groups
            };
            t[which] = f32_tensor(
                &ctx,
                &vec![0.0; width * tile],
                vec![width as u64, s.rows as u64],
                0,
            );
            assert!(encode(&t, s).is_err());
        }
        let mut t = original.clone();
        t[2].provenance = MetalTensorProvenance::OwnedWeightReadOnly;
        assert!(encode(&t, s).is_err());
        for source in [0, 1] {
            let mut t = original.clone();
            t[2].buffer = t[source].buffer.clone();
            t[2].offset = t[source].offset;
            assert!(encode(&t, s).is_err());
        }
        for invalid in [
            Shape { k: 0, ..s },
            Shape { k: 31, ..s },
            Shape { k: 33, ..s },
            Shape { m: 0, ..s },
            Shape { groups: 0, ..s },
            Shape { rows: 0, ..s },
            Shape {
                groups: usize::MAX,
                ..s
            },
            Shape {
                rows: usize::MAX,
                ..s
            },
            Shape {
                k: u32::MAX as usize + 1,
                ..s
            },
        ] {
            assert!(encode(&original, invalid).is_err(), "{invalid:?}");
        }
        enc.end();
        let census = dispatch_census_take();
        command.commit();
        wait_completed(&command).unwrap();
        assert!(census.is_empty(), "invalid bindings dispatched: {census:?}");
        for (tensor, bytes) in original.iter().zip(&before) {
            assert_eq!(&tensor_backing_bytes(tensor), bytes);
            assert_offset_guards(tensor, PREFIX, 0);
        }
        let command = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin_concurrent(&command);
        dispatch_census_begin();
        let result = f.encode(&ctx, &enc, false);
        enc.end();
        let census = dispatch_census_take();
        assert!(result.is_err());
        assert!(census.is_empty());
    }
}
