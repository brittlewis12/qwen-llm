use super::*;
use crate::metal::test_support::{
    assert_offset_guards, metal_test_context, offset_tensor, tensor_backing_bytes,
    tensor_f32_at_offset,
};
use crate::tensor::TensorDesc;

const VARIANTS: [Iq1MatMatVariant; 3] = [
    Iq1MatMatVariant::Auto,
    Iq1MatMatVariant::Scalar,
    Iq1MatMatVariant::Mma,
];

const DTYPES: [GgmlType; 2] = [GgmlType::IQ1_S, GgmlType::IQ1_M];

fn set_scale(b: &mut [u8], dtype: GgmlType, bits: u16) {
    if dtype == GgmlType::IQ1_S {
        b[..2].copy_from_slice(&bits.to_le_bytes());
    } else {
        for i in 0..4 {
            let old = u16::from_le_bytes([b[48 + 2 * i], b[49 + 2 * i]]);
            let word = (old & 0x0fff) | (((bits >> (4 * i)) & 15) << 12);
            b[48 + 2 * i..50 + 2 * i].copy_from_slice(&word.to_le_bytes());
        }
    }
}

fn payload(dtype: GgmlType, k: usize, m: usize) -> Vec<u8> {
    let stride = format_layout(dtype).unwrap().0;
    let mut bytes = vec![0u8; k / 256 * m * stride];
    for row in 0..m {
        for block in 0..k / 256 {
            let b = &mut bytes[(row * (k / 256) + block) * stride..][..stride];
            let d = if row == 2048 {
                0.0
            } else {
                let magnitude = ((row + block * 3) % 7 + 1) as f32 * 0.00137;
                if (row + block) % 3 == 0 {
                    -magnitude
                } else {
                    magnitude
                }
            };
            for group in 0..8 {
                let scale = ((row / 256 + block + group) & 7) as u16;
                let negative = ((row / 16 + block + group) & 1) as u16;
                if dtype == GgmlType::IQ1_S {
                    let mut qh = (scale << 12) | (negative << 15);
                    for l in 0..4 {
                        let code = ((row + 37 * block + 11 * group + 17 * l) & 2047) as u16;
                        b[2 + 4 * group + l] = code as u8;
                        qh |= (code >> 8) << (3 * l);
                    }
                    b[34 + 2 * group..36 + 2 * group].copy_from_slice(&qh.to_le_bytes());
                } else {
                    for l in 0..4 {
                        let code = ((row + 37 * block + 11 * group + 17 * l) & 2047) as u16;
                        b[4 * group + l] = code as u8;
                        let sign = (negative + l as u16) & 1;
                        b[32 + 2 * group + l / 2] |=
                            ((code >> 8) as u8 | ((sign as u8) << 3)) << (4 * (l % 2));
                    }
                    let i = group / 2;
                    let old = u16::from_le_bytes([b[48 + 2 * i], b[49 + 2 * i]]);
                    let scales = scale | (((scale + 3) & 7) << 3);
                    let word = old | (scales << (6 * (group % 2)));
                    b[48 + 2 * i..50 + 2 * i].copy_from_slice(&word.to_le_bytes());
                }
            }
            set_scale(b, dtype, half::f16::from_f32(d).to_bits());
        }
    }
    bytes
}

fn decode(dtype: GgmlType, bytes: &[u8], k: usize, m: usize) -> Vec<f32> {
    crate::codec::dequant_to_f32(
        &TensorDesc {
            name: "iq1_oracle".into(),
            shape: vec![k as u64, m as u64],
            dtype,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: bytes.len() as u64,
        },
        bytes,
    )
    .expect("independent llama.cpp IQ1 codec")
}

fn reference(w: &[f32], x: &[f32], k: usize, m: usize, n: usize) -> Vec<f64> {
    (0..m * n)
        .map(|i| {
            let (token, row) = (i / m, i % m);
            w[row * k..(row + 1) * k]
                .iter()
                .zip(&x[token * k..(token + 1) * k])
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum()
        })
        .collect()
}

fn assert_numerical(got: &[f32], expected: &[f64]) {
    assert_eq!(got.len(), expected.len());
    let (mut diff2, mut ref2, mut peak, mut worst) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (&a, &b) in got.iter().zip(expected) {
        assert!(a.is_finite() && b.is_finite(), "nonfinite result/reference");
        let delta = f64::from(a) - b;
        diff2 += delta * delta;
        ref2 += b * b;
        peak = peak.max(b.abs());
        worst = worst.max(delta.abs());
    }
    assert!(
        worst <= 3e-5 * peak.max(1.0),
        "max error={worst}, reference peak={peak}"
    );
    assert!(
        (diff2 / ref2.max(1e-30)).sqrt() <= 3e-5,
        "relative L2 error exceeds 3e-5"
    );
}

struct Fixture {
    k: usize,
    m: usize,
    n: usize,
    suffix: usize,
    weight: MetalTensor,
    input: MetalTensor,
    expected: Vec<f64>,
}

impl Fixture {
    fn new(
        ctx: &MetalContext,
        dtype: GgmlType,
        k: usize,
        m: usize,
        n: usize,
        prefix: usize,
        suffix: usize,
    ) -> Self {
        let bytes = payload(dtype, k, m);
        let x: Vec<f32> = (0..k * n)
            .map(|i| (((i * 37 + i / k * 11) % 251) as f32 - 125.0) * 0.0013)
            .collect();
        Self::from_values(ctx, dtype, k, m, n, prefix, suffix, &bytes, &x)
    }
    fn from_values(
        ctx: &MetalContext,
        dtype: GgmlType,
        k: usize,
        m: usize,
        n: usize,
        prefix: usize,
        suffix: usize,
        bytes: &[u8],
        x: &[f32],
    ) -> Self {
        let expected = reference(&decode(dtype, bytes, k, m), x, k, m, n);
        Self {
            k,
            m,
            n,
            suffix,
            expected,
            weight: offset_tensor(ctx, 18, bytes, suffix, vec![k as u64, m as u64], dtype),
            input: offset_tensor(
                ctx,
                prefix,
                bytemuck::cast_slice(x),
                suffix,
                vec![k as u64, n as u64],
                GgmlType::F32,
            ),
        }
    }
    fn output(&self, ctx: &MetalContext) -> MetalTensor {
        offset_tensor(
            ctx,
            20,
            bytemuck::cast_slice(&vec![-777.0f32; self.m * self.n]),
            self.suffix,
            vec![self.m as u64, self.n as u64],
            GgmlType::F32,
        )
    }
    fn encode(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        out: &MetalTensor,
        variant: Iq1MatMatVariant,
    ) -> Result<(), MetalError> {
        if matches!(variant, Iq1MatMatVariant::Auto) {
            return encode_mat_mat_iq1_f32(
                ctx,
                enc,
                &self.weight,
                &self.input,
                out,
                self.k,
                self.m,
                self.n,
            );
        }
        encode_mat_mat_iq1_f32_with_variant(
            ctx,
            enc,
            variant,
            &self.weight,
            &self.input,
            out,
            self.k,
            self.m,
            self.n,
        )
    }
    fn run(&self, ctx: &MetalContext, variant: Iq1MatMatVariant, expected_path: Path) -> Vec<f32> {
        let out = self.output(ctx);
        let before = [
            tensor_backing_bytes(&self.weight),
            tensor_backing_bytes(&self.input),
        ];
        dispatch_census_begin();
        let result = one_shot(ctx, |enc| self.encode(ctx, enc, &out, variant));
        let census = dispatch_census_take();
        result.unwrap();
        assert_eq!(census.len(), 1);
        let format = match self.weight.dtype {
            GgmlType::IQ1_S => "iq1_s",
            GgmlType::IQ1_M => "iq1_m",
            _ => unreachable!(),
        };
        assert_eq!(
            census[0].kernel,
            format!("{} [{format}]", expected_path.name())
        );
        assert_eq!(census[0].tag.as_deref(), Some(format));
        assert_eq!(census[0].threads_width, expected_path.threads() as u64);
        let (width, height) = match expected_path {
            Path::Gemv => (self.m.div_ceil(8), 1),
            Path::Scalar => (self.m, self.n.div_ceil(32)),
            Path::Gather => unreachable!(),
            Path::Mma => (self.m.div_ceil(16), self.n.div_ceil(8)),
        };
        assert_eq!(
            (
                census[0].grid_width,
                census[0].grid_height,
                census[0].grid_depth
            ),
            (width as u64, height as u64, 1)
        );
        let got = tensor_f32_at_offset(&out);
        assert_numerical(&got, &self.expected);
        assert_offset_guards(&out, 20, self.suffix);
        assert_eq!(tensor_backing_bytes(&self.weight), before[0]);
        assert_eq!(tensor_backing_bytes(&self.input), before[1]);
        got
    }
}

#[test]
fn iq1_all_grid_scale_delta_positions_match_codec_basis() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    for dtype in DTYPES {
        let (k, m, n) = (256, 2049, 256);
        let bytes = payload(dtype, k, m);
        let mut basis = vec![0.0f32; k * n];
        for i in 0..k {
            basis[i * k + i] = 1.0;
        }
        // The one-hot oracle is the codec output, transposed; avoid a cubic CPU dot.
        let decoded = decode(dtype, &bytes, k, m);
        let expected: Vec<f64> = (0..m * n)
            .map(|i| f64::from(decoded[(i % m) * k + i / m]))
            .collect();
        let f = Fixture {
            k,
            m,
            n,
            suffix: 0,
            expected,
            weight: offset_tensor(&ctx, 18, &bytes, 0, vec![k as u64, m as u64], dtype),
            input: offset_tensor(
                &ctx,
                16,
                bytemuck::cast_slice(&basis),
                0,
                vec![k as u64, n as u64],
                GgmlType::F32,
            ),
        };
        for variant in VARIANTS {
            let path = if matches!(variant, Iq1MatMatVariant::Scalar) {
                Path::Scalar
            } else {
                Path::Mma
            };
            let got = f.run(&ctx, variant, path);
            for (&a, &b) in got.iter().zip(&f.expected) {
                assert!(
                    (f64::from(a) - b).abs() <= 2e-6 * b.abs().max(1e-30),
                    "basis decoder mismatch {a} vs {b}"
                );
            }
        }
        let output = f.output(&ctx);
        let before = [
            tensor_backing_bytes(&f.weight),
            tensor_backing_bytes(&f.input),
        ];
        dispatch_census_begin();
        let result = one_shot(&ctx, |enc| {
            for token in 0..n {
                encode_mat_vec_iq1_f32(
                    &ctx,
                    enc,
                    &f.weight,
                    &f.input.view_subrange((token * k) as u64, vec![k as u64]),
                    &output.view_subrange((token * m) as u64, vec![m as u64]),
                    k,
                    m,
                )?;
            }
            Ok(())
        });
        let census = dispatch_census_take();
        result.unwrap();
        assert_eq!(census.len(), n);
        let format = if dtype == GgmlType::IQ1_S {
            "iq1_s"
        } else {
            "iq1_m"
        };
        assert!(
            census
                .iter()
                .all(|row| row.kernel == format!("{GEMV} [{format}]")
                    && row.tag.as_deref() == Some(format))
        );
        let got = tensor_f32_at_offset(&output);
        for (&a, &b) in got.iter().zip(&f.expected) {
            assert!(a.is_finite() && (f64::from(a) - b).abs() <= 2e-6 * b.abs().max(1e-30));
        }
        assert_offset_guards(&output, 20, 0);
        assert_eq!(tensor_backing_bytes(&f.weight), before[0]);
        assert_eq!(tensor_backing_bytes(&f.input), before[1]);
    }
}

#[test]
fn iq1_f32_paths_match_f64_tails_offsets_and_real_k() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    for dtype in DTYPES {
        let mut cases = Vec::new();
        for (i, n) in [1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 127, 128, 129]
            .into_iter()
            .enumerate()
        {
            cases.push((
                [256, 512, 768][i % 3],
                [1, 3, 4, 7, 8, 9, 15, 16, 17, 63, 64, 65][i % 12],
                n,
            ));
        }
        for k in [5120, 6144, 17408] {
            for n in [1, 2, 9, 33] {
                cases.push((k, 17, n));
            }
        }
        for (i, (k, m, n)) in cases.into_iter().enumerate() {
            let f = Fixture::new(&ctx, dtype, k, m, n, 16, if i % 2 == 0 { 0 } else { 28 });
            for variant in VARIANTS {
                let path = if n == 1 {
                    Path::Gemv
                } else if matches!(variant, Iq1MatMatVariant::Scalar) {
                    Path::Scalar
                } else {
                    Path::Mma
                };
                f.run(&ctx, variant, path);
            }
        }
    }
}

#[test]
fn iq1_f32_scale_extremes_do_not_stage_half() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    for dtype in DTYPES {
        let (k, m, n) = (256, 10, 9);
        let mut bytes = payload(dtype, k, m);
        for (row, bits) in [
            0u16, 0x8000, 0x0001, 0x03ff, 0x8001, 0x83ff, 0x0400, 0x3555, 0x7bff, 0xfbff,
        ]
        .into_iter()
        .enumerate()
        {
            let stride = format_layout(dtype).unwrap().0;
            set_scale(&mut bytes[row * stride..][..stride], dtype, bits);
        }
        let x: Vec<f32> = (0..k * n)
            .map(|i| ((i * 11 % 97) as f32 - 48.0) * 0.013)
            .collect();
        let f = Fixture::from_values(&ctx, dtype, k, m, n, 16, 28, &bytes, &x);
        assert!(f.expected.iter().any(|v| v.abs() > 65504.0));
        let check_rows = |got: &[f32]| {
            // Large scales must not hide loss of the subnormal-scale rows.
            for row in 0..m {
                let actual: Vec<f32> = (0..n).map(|token| got[token * m + row]).collect();
                let expected: Vec<f64> = (0..n).map(|token| f.expected[token * m + row]).collect();
                assert_numerical(&actual, &expected);
            }
        };
        for variant in VARIANTS {
            let got = f.run(
                &ctx,
                variant,
                if matches!(variant, Iq1MatMatVariant::Scalar) {
                    Path::Scalar
                } else {
                    Path::Mma
                },
            );
            check_rows(&got);
        }
        let out = f.output(&ctx);
        one_shot(&ctx, |enc| {
            for token in 0..n {
                encode_mat_vec_iq1_f32(
                    &ctx,
                    enc,
                    &f.weight,
                    &f.input.view_subrange((token * k) as u64, vec![k as u64]),
                    &out.view_subrange((token * m) as u64, vec![m as u64]),
                    k,
                    m,
                )?;
            }
            Ok(())
        })
        .unwrap();
        check_rows(&tensor_f32_at_offset(&out));
        assert_offset_guards(&out, 20, 28);
    }
}

fn with_fault<R>(path: Path, fault: PipelineFault, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<(Path, PipelineFault)>);
    impl Drop for Restore {
        fn drop(&mut self) {
            PIPELINE_FAULT.with(|s| s.set(self.0));
        }
    }
    let _restore = Restore(PIPELINE_FAULT.with(|s| s.replace(Some((path, fault)))));
    f()
}

fn with_mma_support<R>(supported: bool, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            MMA_SUPPORT.with(|s| s.set(self.0));
        }
    }
    let _restore = Restore(MMA_SUPPORT.with(|s| s.replace(Some(supported))));
    f()
}

#[test]
fn iq1_apple7_support_contract_and_scoped_injection() {
    check_mma_support(true).unwrap();
    assert!(check_mma_support(false).is_err());
    assert_eq!(MMA_SUPPORT.with(|s| s.get()), None);
    with_mma_support(false, || {
        assert_eq!(MMA_SUPPORT.with(|s| s.get()), Some(false));
        let unwind = std::panic::catch_unwind(|| {
            with_mma_support(true, || {
                assert_eq!(MMA_SUPPORT.with(|s| s.get()), Some(true));
                panic!("test capability scope restoration");
            });
        });
        assert!(unwind.is_err());
        assert_eq!(MMA_SUPPORT.with(|s| s.get()), Some(false));
        std::thread::spawn(|| assert_eq!(MMA_SUPPORT.with(|s| s.get()), None))
            .join()
            .unwrap();
    });
    assert_eq!(MMA_SUPPORT.with(|s| s.get()), None);
}

#[test]
fn iq1_unsupported_apple7_falls_back_before_pipeline_lookup() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    for dtype in DTYPES {
        let f = Fixture::new(&ctx, dtype, 512, 17, 9, 16, 28);
        with_mma_support(false, || {
            with_fault(Path::Mma, PipelineFault::Missing, || {
                f.run(&ctx, Iq1MatMatVariant::Auto, Path::Scalar);
                f.run(&ctx, Iq1MatMatVariant::Scalar, Path::Scalar);
                let out = f.output(&ctx);
                let before = tensor_backing_bytes(&out);
                let cmd = ctx.queue.commandBuffer().unwrap();
                let enc = KernelEncoder::begin(&cmd);
                dispatch_census_begin();
                let err = f
                    .encode(&ctx, &enc, &out, Iq1MatMatVariant::Mma)
                    .unwrap_err();
                assert!(err.to_string().contains("Apple7"), "{err}");
                enc.end();
                assert!(dispatch_census_take().is_empty());
                assert_eq!(tensor_backing_bytes(&out), before);
                let singleton = Fixture::new(&ctx, dtype, 256, 7, 1, 16, 0);
                singleton.run(&ctx, Iq1MatMatVariant::Auto, Path::Gemv);
            });
        });
        // Never enable an unsupported hardware path merely to exercise injection.
        if ctx.device.supportsFamily(objc2_metal::MTLGPUFamily::Apple7) {
            with_mma_support(true, || {
                f.run(&ctx, Iq1MatMatVariant::Auto, Path::Mma);
                f.run(&ctx, Iq1MatMatVariant::Mma, Path::Mma);
            });
        }
    }
}

#[test]
fn iq1_auto_preflights_capability_and_alignment_fallback() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    for dtype in DTYPES {
        let f = Fixture::new(&ctx, dtype, 512, 17, 9, 16, 28);
        for fault in [
            PipelineFault::Missing,
            PipelineFault::Width,
            PipelineFault::Threads,
            PipelineFault::Memory,
        ] {
            with_fault(Path::Mma, fault, || {
                f.run(&ctx, Iq1MatMatVariant::Auto, Path::Scalar);
                let out = f.output(&ctx);
                let cmd = ctx.queue.commandBuffer().unwrap();
                let enc = KernelEncoder::begin(&cmd);
                dispatch_census_begin();
                assert!(f.encode(&ctx, &enc, &out, Iq1MatMatVariant::Mma).is_err());
                let mut invalid = out.clone();
                invalid.offset += 1;
                assert!(
                    f.encode(&ctx, &enc, &invalid, Iq1MatMatVariant::Auto)
                        .is_err()
                );
                enc.end();
                assert!(dispatch_census_take().is_empty());
            });
        }
        let f = Fixture::new(&ctx, dtype, 512, 17, 9, 20, 0);
        with_fault(Path::Mma, PipelineFault::Missing, || {
            f.run(&ctx, Iq1MatMatVariant::Auto, Path::Scalar);
        });
        let out = f.output(&ctx);
        let cmd = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&cmd);
        dispatch_census_begin();
        assert!(f.encode(&ctx, &enc, &out, Iq1MatMatVariant::Mma).is_err());
        enc.end();
        assert!(dispatch_census_take().is_empty());
        // If even the fallback cannot run, fail without a partial dispatch.
        with_fault(Path::Scalar, PipelineFault::Memory, || {
            let out = f.output(&ctx);
            let cmd = ctx.queue.commandBuffer().unwrap();
            let enc = KernelEncoder::begin(&cmd);
            dispatch_census_begin();
            assert!(f.encode(&ctx, &enc, &out, Iq1MatMatVariant::Auto).is_err());
            enc.end();
            assert!(dispatch_census_take().is_empty());
        });
        let singleton = Fixture::new(&ctx, dtype, 256, 7, 1, 20, 0);
        with_fault(Path::Mma, PipelineFault::Missing, || {
            singleton.run(&ctx, Iq1MatMatVariant::Auto, Path::Gemv);
        });
    }
}

#[test]
fn iq1_capacity_device_ids_and_dimensions() {
    assert_eq!(format_layout(GgmlType::IQ1_S).unwrap(), (50, 0));
    assert_eq!(format_layout(GgmlType::IQ1_M).unwrap(), (56, 1));
    assert!(format_layout(GgmlType::F32).is_err());
    assert!(format_layout(GgmlType::IQ2_XXS).is_err());
    for path in [Path::Gemv, Path::Scalar, Path::Mma, Path::Gather] {
        check_capacity(path, 32, path.threads(), 0, path.scratch()).unwrap();
        assert!(check_capacity(path, 16, 256, 0, 32768).is_err());
        assert!(check_capacity(path, 32, path.threads() - 1, 0, 32768).is_err());
        assert!(check_capacity(path, 32, 256, 1, path.scratch()).is_err());
        assert!(check_capacity(path, 32, 256, usize::MAX, usize::MAX - 1).is_err());
    }
    assert!(super::super::checks::to_u32(KERNEL, u32::MAX as usize + 1, "test").is_err());
    check_devices(7, 7, [7; 3]).unwrap();
    assert!(check_devices(7, 8, [7; 3]).is_err());
    for i in 0..3 {
        let mut devices = [7; 3];
        devices[i] = 8;
        assert!(check_devices(7, 7, devices).is_err());
    }
    assert!(checked_moe_product(KERNEL, "test overflow", &[usize::MAX, 2]).is_err());
}

#[test]
fn iq1_concurrent_shared_inputs_distinct_outputs_match_serial() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    for dtype in DTYPES {
        let mut cases = vec![
            (1, true, Iq1MatMatVariant::Auto, 16),
            (1, false, Iq1MatMatVariant::Auto, 16),
            (9, false, Iq1MatMatVariant::Auto, 16),
            (9, false, Iq1MatMatVariant::Scalar, 16),
            (33, false, Iq1MatMatVariant::Auto, 20),
        ];
        if ctx.device.supportsFamily(objc2_metal::MTLGPUFamily::Apple7) {
            cases.push((9, false, Iq1MatMatVariant::Mma, 16));
        }
        for (n, vector, variant, prefix) in cases {
            let f = Fixture::new(&ctx, dtype, 512, 17, n, prefix, 28);
            let before = [
                tensor_backing_bytes(&f.weight),
                tensor_backing_bytes(&f.input),
            ];
            let encode = |enc: &KernelEncoder, out: &MetalTensor| {
                if vector {
                    let x = f.input.view_subrange(0, vec![f.k as u64]);
                    let y = out.view_subrange(0, vec![f.m as u64]);
                    encode_mat_vec_iq1_f32(&ctx, enc, &f.weight, &x, &y, f.k, f.m)
                } else {
                    f.encode(&ctx, enc, out, variant)
                }
            };
            let serial = [f.output(&ctx), f.output(&ctx)];
            dispatch_census_begin();
            one_shot(&ctx, |enc| {
                for out in &serial {
                    encode(enc, out)?;
                }
                Ok(())
            })
            .unwrap();
            let serial_census = dispatch_census_take();
            assert_eq!(serial_census.len(), 2);

            let concurrent = [f.output(&ctx), f.output(&ctx)];
            let cmd = ctx.queue.commandBuffer().unwrap();
            let enc = KernelEncoder::begin_concurrent(&cmd);
            dispatch_census_begin();
            for out in &concurrent {
                encode(&enc, out).unwrap();
            }
            #[cfg(debug_assertions)]
            {
                assert_eq!(enc.hazard_reads.borrow().len(), 4);
                assert_eq!(enc.hazard_writes.borrow().len(), 2);
            }
            enc.end();
            let concurrent_census = dispatch_census_take();
            assert_eq!(concurrent_census.len(), 2);
            cmd.commit();
            wait_completed(&cmd).unwrap();
            for i in 0..2 {
                assert_eq!(concurrent_census[i].kernel, serial_census[i].kernel);
                assert_eq!(concurrent_census[i].tag, serial_census[i].tag);
                let serial_values = tensor_f32_at_offset(&serial[i]);
                let concurrent_values = tensor_f32_at_offset(&concurrent[i]);
                assert_numerical(&serial_values, &f.expected);
                assert_numerical(&concurrent_values, &f.expected);
                let serial_reference: Vec<f64> = serial_values.into_iter().map(f64::from).collect();
                assert_numerical(&concurrent_values, &serial_reference);
                assert_offset_guards(&serial[i], 20, f.suffix);
                assert_offset_guards(&concurrent[i], 20, f.suffix);
            }
            assert_eq!(tensor_backing_bytes(&f.weight), before[0]);
            assert_eq!(tensor_backing_bytes(&f.input), before[1]);
        }
    }
}

#[test]
fn iq1_invalid_bindings_do_not_dispatch_on_serial_or_concurrent() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    for dtype in DTYPES {
        let f = Fixture::new(&ctx, dtype, 256, 17, 9, 16, 0);
        let out = f.output(&ctx);
        let original = [f.weight.clone(), f.input.clone(), out.clone()];
        for concurrent in [false, true] {
            let cmd = ctx.queue.commandBuffer().unwrap();
            let enc = if concurrent {
                KernelEncoder::begin_concurrent(&cmd)
            } else {
                KernelEncoder::begin(&cmd)
            };
            dispatch_census_begin();
            for variant in VARIANTS {
                let encode = |t: &[MetalTensor; 3], k, m, n| {
                    encode_mat_mat_iq1_f32_with_variant(
                        &ctx, &enc, variant, &t[0], &t[1], &t[2], k, m, n,
                    )
                };
                for i in 0..3 {
                    for fault in 0..6 {
                        let mut t = original.clone();
                        match fault {
                            0 => t[i].offset += 1,
                            1 => t[i].offset = t[i].buffer.length() as u64,
                            2 => t[i].offset = u64::MAX - 15,
                            3 => t[i].dtype = GgmlType::F16,
                            4 => t[i].shape = vec![1],
                            _ => t[i].shape.push(1),
                        }
                        assert!(encode(&t, f.k, f.m, f.n).is_err());
                    }
                }
                let mut t = original.clone();
                t[2].provenance = MetalTensorProvenance::OwnedWeightReadOnly;
                assert!(encode(&t, f.k, f.m, f.n).is_err());
                for source in [&f.weight, &f.input] {
                    let mut t = original.clone();
                    t[2].buffer = source.buffer.clone();
                    t[2].offset = source.offset.next_multiple_of(4);
                    assert!(encode(&t, f.k, f.m, f.n).is_err());
                }
                for (k, m, n) in [
                    (0, 17, 9),
                    (128, 17, 9),
                    (257, 17, 9),
                    (256, 0, 9),
                    (256, 17, 0),
                    (usize::MAX, 17, 9),
                    (256, usize::MAX, 9),
                    (256, 17, usize::MAX),
                    (i32::MAX as usize + 1, 17, 9),
                    (256, i32::MAX as usize + 1, 9),
                    (256, 17, i32::MAX as usize + 1),
                ] {
                    assert!(encode(&original, k, m, n).is_err());
                }
            }
            let x = f.input.view_subrange(0, vec![f.k as u64]);
            let y = out.view_subrange(0, vec![f.m as u64]);
            for i in 0..3 {
                let mut t = [f.weight.clone(), x.clone(), y.clone()];
                t[i].offset += 1;
                assert!(encode_mat_vec_iq1_f32(&ctx, &enc, &t[0], &t[1], &t[2], f.k, f.m).is_err());
            }
            #[cfg(debug_assertions)]
            {
                assert!(enc.hazard_reads.borrow().is_empty());
                assert!(enc.hazard_writes.borrow().is_empty());
            }
            enc.end();
            assert!(dispatch_census_take().is_empty());
        }
    }
}

#[test]
fn iq1_foreign_device_bindings_and_encoder_are_rejected_when_available() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    for dtype in DTYPES {
        let devices = objc2_metal::MTLCopyAllDevices();
        let Some(device) = devices
            .iter()
            .find(|d| d.registryID() != ctx.device.registryID())
        else {
            return;
        };
        let f = Fixture::new(&ctx, dtype, 256, 17, 9, 16, 0);
        let out = f.output(&ctx);
        let cmd = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&cmd);
        dispatch_census_begin();
        for which in 0..3 {
            let mut t = [f.weight.clone(), f.input.clone(), out.clone()];
            t[which].buffer = device
                .newBufferWithLength_options(
                    t[which].buffer.length(),
                    MTLResourceOptions::StorageModeShared,
                )
                .unwrap();
            assert!(
                encode_mat_mat_iq1_f32(&ctx, &enc, &t[0], &t[1], &t[2], f.k, f.m, f.n).is_err()
            );
        }
        enc.end();
        assert!(dispatch_census_take().is_empty());
        let queue = device.newCommandQueue().unwrap();
        let cmd = queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&cmd);
        dispatch_census_begin();
        assert!(f.encode(&ctx, &enc, &out, Iq1MatMatVariant::Auto).is_err());
        enc.end();
        assert!(dispatch_census_take().is_empty());
    }
}

#[test]
fn iq1_layout_grid_scale_delta_coverage() {
    assert_eq!(std::mem::size_of::<Args>(), 20);
    assert_eq!(GgmlType::IQ1_S.storage_layout(), Some((256, 50)));
    assert_eq!(GgmlType::IQ1_M.storage_layout(), Some((256, 56)));
    let source = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../kernels/iq1_grid.metalh"
    ));
    let table = source
        .split("constant ulong iq1_grid[2048] = {")
        .nth(1)
        .unwrap()
        .split("};")
        .next()
        .unwrap();
    let grid: Vec<u64> = table
        .split(',')
        .filter_map(|v| {
            let v = v.trim();
            if v.is_empty() {
                None
            } else {
                Some(u64::from_str_radix(v.trim_start_matches("0x"), 16).unwrap())
            }
        })
        .collect();
    assert_eq!(grid.len(), 2048);
    for dtype in DTYPES {
        let stride = format_layout(dtype).unwrap().0;
        let bytes = payload(dtype, 256, 2048);
        let decoded = decode(dtype, &bytes, 256, 2048);
        for group8 in 0..32 {
            let (mut grids, mut scales, mut deltas) = ([false; 2048], [false; 8], [false; 2]);
            for row in 0..2048 {
                let b = &bytes[row * stride..][..stride];
                let (bits, code, scale, negative) = if dtype == GgmlType::IQ1_S {
                    let qh = u16::from_le_bytes(
                        b[34 + 2 * (group8 / 4)..36 + 2 * (group8 / 4)]
                            .try_into()
                            .unwrap(),
                    );
                    (
                        u16::from_le_bytes([b[0], b[1]]),
                        usize::from(b[2 + group8])
                            | (usize::from((qh >> (3 * (group8 % 4))) & 7) << 8),
                        usize::from((qh >> 12) & 7),
                        usize::from(qh >> 15),
                    )
                } else {
                    let sc: Vec<u16> = b[48..]
                        .chunks_exact(2)
                        .map(|v| u16::from_le_bytes([v[0], v[1]]))
                        .collect();
                    let bits = (sc[0] >> 12)
                        | ((sc[1] >> 8) & 0xf0)
                        | ((sc[2] >> 4) & 0xf00)
                        | (sc[3] & 0xf000);
                    let qh = b[32 + group8 / 2] >> (4 * (group8 % 2));
                    (
                        bits,
                        usize::from(b[group8]) | (usize::from(qh & 7) << 8),
                        usize::from((sc[group8 / 8] >> (3 * ((group8 / 2) % 4))) & 7),
                        usize::from((qh >> 3) & 1),
                    )
                };
                grids[code] = true;
                scales[scale] = true;
                deltas[negative] = true;
                let dl = half::f16::from_bits(bits).to_f32() * (2 * scale + 1) as f32;
                let delta = if negative != 0 { -0.125 } else { 0.125 };
                for (j, v) in grid[code].to_le_bytes().into_iter().enumerate() {
                    assert!(matches!(v as i8, -1..=1));
                    assert_eq!(
                        dl * (f32::from(v as i8) + delta),
                        decoded[row * 256 + group8 * 8 + j]
                    );
                }
            }
            assert!(grids.into_iter().all(|v| v));
            assert!(scales.into_iter().all(|v| v));
            assert!(deltas.into_iter().all(|v| v));
        }
    }
}

#[test]
fn iq1_m_reconstructed_half_scale_all_finite_encodings_match_codec() {
    // Bounded batches: all four scale nibbles, signed zero, subnormals and
    // maximal finite half values, with no large persistent F32 test bank.
    let mut bytes = vec![0u8; 256 * 56];
    for high in 0..256u16 {
        for low in 0..256u16 {
            set_scale(
                &mut bytes[usize::from(low) * 56..][..56],
                GgmlType::IQ1_M,
                (high << 8) | low,
            );
        }
        let decoded = decode(GgmlType::IQ1_M, &bytes, 256, 256);
        for low in 0..256u16 {
            let d = half::f16::from_bits((high << 8) | low).to_f32();
            if !d.is_finite() {
                continue;
            }
            for &value in &decoded[usize::from(low) * 256..][..256] {
                assert_eq!(value, d * -0.875);
            }
        }
    }
}

fn gather_run(
    ctx: &MetalContext,
    embed: &MetalTensor,
    ids: &MetalTensor,
    out: &MetalTensor,
    tokens: usize,
    width: usize,
    concurrent: bool,
) {
    let cmd = ctx.queue.commandBuffer().unwrap();
    let enc = if concurrent {
        KernelEncoder::begin_concurrent(&cmd)
    } else {
        KernelEncoder::begin(&cmd)
    };
    dispatch_census_begin();
    encode_get_rows_iq1_m_f32(ctx, &enc, embed, ids, out, tokens, width).unwrap();
    enc.end();
    let census = dispatch_census_take();
    assert_eq!(census.len(), 1);
    assert_eq!(census[0].kernel, Path::Gather.name());
    assert_eq!(census[0].grid_width, (width / 256) as u64);
    assert_eq!(census[0].grid_height, tokens as u64);
    assert_eq!(census[0].threads_width, 256);
    cmd.commit();
    wait_completed(&cmd).unwrap();
}

#[test]
fn iq1_m_gather_codec_offsets_vocab_edges_invalid_ids_and_immutability() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    for (width, vocab, suffix) in [(256, 1, 0), (512, 17, 28), (5120, 7, 0), (17408, 3, 28)] {
        let bytes = payload(GgmlType::IQ1_M, width, vocab);
        let decoded = decode(GgmlType::IQ1_M, &bytes, width, vocab);
        let indices = [
            0i32,
            vocab as i32 - 1,
            (vocab / 2) as i32,
            vocab as i32 - 1,
            -1,
            vocab as i32,
            i32::MIN,
            i32::MAX,
        ];
        let tokens = indices.len();
        for flat in [false, true] {
            let embed = offset_tensor(
                &ctx,
                18,
                &bytes,
                suffix,
                if flat {
                    vec![(width * vocab) as u64]
                } else {
                    vec![width as u64, vocab as u64]
                },
                GgmlType::IQ1_M,
            );
            let ids = offset_tensor(
                &ctx,
                12,
                bytemuck::cast_slice(&indices),
                0,
                vec![tokens as u64],
                GgmlType::I32,
            );
            let out = offset_tensor(
                &ctx,
                20,
                bytemuck::cast_slice(&vec![-777.0f32; tokens * width]),
                suffix,
                if flat {
                    vec![(tokens * width) as u64]
                } else {
                    vec![width as u64, tokens as u64]
                },
                GgmlType::F32,
            );
            let before = [tensor_backing_bytes(&embed), tensor_backing_bytes(&ids)];
            gather_run(&ctx, &embed, &ids, &out, tokens, width, flat);
            let got = tensor_f32_at_offset(&out);
            for (token, id) in indices.into_iter().enumerate() {
                for j in 0..width {
                    let expected = if id >= 0 && (id as usize) < vocab {
                        decoded[id as usize * width + j]
                    } else {
                        0.0
                    };
                    let a = got[token * width + j];
                    assert!(
                        a.is_finite() && (a - expected).abs() <= 2e-6 * expected.abs().max(1e-30),
                        "gather mismatch {a} vs {expected}"
                    );
                }
            }
            assert_offset_guards(&out, 20, suffix);
            assert_eq!(tensor_backing_bytes(&embed), before[0]);
            assert_eq!(tensor_backing_bytes(&ids), before[1]);
        }
    }
}

#[test]
fn iq1_m_gather_reconstructed_scale_nibbles_and_extremes_match_codec() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let mut scales = vec![
        0u16, 0x8000, 0x0001, 0x8001, 0x03ff, 0x83ff, 0x0400, 0x7bff, 0xfbff,
    ];
    for nibble in 0..4 {
        for value in 0..16u16 {
            let bits = (0x3555 & !(15 << (4 * nibble))) | (value << (4 * nibble));
            if half::f16::from_bits(bits).is_finite() {
                scales.push(bits);
            }
        }
    }
    let vocab = scales.len();
    let mut bytes = payload(GgmlType::IQ1_M, 256, vocab);
    for (b, bits) in bytes.chunks_exact_mut(56).zip(scales) {
        set_scale(b, GgmlType::IQ1_M, bits);
    }
    let expected = decode(GgmlType::IQ1_M, &bytes, 256, vocab);
    let indices: Vec<i32> = (0..vocab as i32).collect();
    let embed = offset_tensor(
        &ctx,
        18,
        &bytes,
        0,
        vec![256, vocab as u64],
        GgmlType::IQ1_M,
    );
    let ids = offset_tensor(
        &ctx,
        4,
        bytemuck::cast_slice(&indices),
        0,
        vec![vocab as u64],
        GgmlType::I32,
    );
    let out = offset_tensor(
        &ctx,
        20,
        bytemuck::cast_slice(&vec![f32::NAN; vocab * 256]),
        28,
        vec![256, vocab as u64],
        GgmlType::F32,
    );
    gather_run(&ctx, &embed, &ids, &out, vocab, 256, false);
    for (a, b) in tensor_f32_at_offset(&out).into_iter().zip(expected) {
        assert!(
            a.is_finite() && (a - b).abs() <= 2e-6 * b.abs().max(1e-30),
            "scale mismatch {a} vs {b}"
        );
    }
    assert_offset_guards(&out, 20, 28);
}

#[test]
fn iq1_m_gather_rejects_invalid_bindings_devices_and_capabilities_before_dispatch() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let (width, vocab, tokens) = (256, 64, 2);
    let bytes = payload(GgmlType::IQ1_M, width, vocab);
    let embed = offset_tensor(
        &ctx,
        18,
        &bytes,
        0,
        vec![width as u64, vocab as u64],
        GgmlType::IQ1_M,
    );
    // Spare physical ID storage lets the alias case reach disjointness checks.
    let ids = offset_tensor(
        &ctx,
        12,
        bytemuck::cast_slice(&[0i32, 63]),
        4096,
        vec![2],
        GgmlType::I32,
    );
    let out = offset_tensor(
        &ctx,
        20,
        bytemuck::cast_slice(&vec![-777.0f32; tokens * width]),
        28,
        vec![width as u64, tokens as u64],
        GgmlType::F32,
    );
    let original = [embed, ids, out];
    let before = original.each_ref().map(tensor_backing_bytes);
    for concurrent in [false, true] {
        let cmd = ctx.queue.commandBuffer().unwrap();
        let enc = if concurrent {
            KernelEncoder::begin_concurrent(&cmd)
        } else {
            KernelEncoder::begin(&cmd)
        };
        let encode = |t: &[MetalTensor; 3], n, k| {
            encode_get_rows_iq1_m_f32(&ctx, &enc, &t[0], &t[1], &t[2], n, k)
        };
        dispatch_census_begin();
        for which in 0..3 {
            for fault in 0..7 {
                let mut t = original.clone();
                match fault {
                    0 => t[which].offset += 1,
                    1 => t[which].offset = t[which].buffer.length() as u64,
                    2 => t[which].offset = u64::MAX - 15,
                    3 => t[which].dtype = GgmlType::F32,
                    4 => t[which].shape = vec![1],
                    5 => t[which].shape = vec![u64::MAX, 2],
                    _ => t[which].dtype = GgmlType::F16,
                }
                // F32 is the correct output type; use I32 to make it invalid.
                if which == 2 && fault == 3 {
                    t[which].dtype = GgmlType::I32;
                }
                assert!(encode(&t, tokens, width).is_err());
            }
        }
        let mut t = original.clone();
        t[0].dtype = GgmlType::IQ1_S;
        assert!(encode(&t, tokens, width).is_err());
        let mut t = original.clone();
        t[2].provenance = MetalTensorProvenance::OwnedWeightReadOnly;
        assert!(encode(&t, tokens, width).is_err());
        for source in 0..2 {
            let mut t = original.clone();
            t[2].buffer = t[source].buffer.clone();
            t[2].offset = t[source].offset.next_multiple_of(4);
            assert!(encode(&t, tokens, width).is_err());
        }
        for (n, k) in [
            (0, 256),
            (2, 0),
            (2, 255),
            (2, 257),
            (usize::MAX, 256),
            (2, usize::MAX),
            (u32::MAX as usize + 1, 256),
            (2, u32::MAX as usize + 1),
        ] {
            assert!(encode(&original, n, k).is_err());
        }
        let mut t = original.clone();
        t[0].shape = vec![256, 63];
        assert!(encode(&t, 1, 512).is_err());
        for fault in [
            PipelineFault::Missing,
            PipelineFault::Width,
            PipelineFault::Threads,
            PipelineFault::Memory,
        ] {
            with_fault(Path::Gather, fault, || {
                assert!(encode(&original, tokens, width).is_err())
            });
        }
        let devices = objc2_metal::MTLCopyAllDevices();
        if let Some(device) = devices
            .iter()
            .find(|d| d.registryID() != ctx.device.registryID())
        {
            for which in 0..3 {
                let mut t = original.clone();
                t[which].buffer = device
                    .newBufferWithLength_options(
                        t[which].buffer.length(),
                        MTLResourceOptions::StorageModeShared,
                    )
                    .unwrap();
                assert!(encode(&t, tokens, width).is_err());
            }
            let queue = device.newCommandQueue().unwrap();
            let other_cmd = queue.commandBuffer().unwrap();
            let other_enc = KernelEncoder::begin(&other_cmd);
            assert!(
                encode_get_rows_iq1_m_f32(
                    &ctx,
                    &other_enc,
                    &original[0],
                    &original[1],
                    &original[2],
                    tokens,
                    width
                )
                .is_err()
            );
            other_enc.end();
        }
        #[cfg(debug_assertions)]
        {
            assert!(enc.hazard_reads.borrow().is_empty());
            assert!(enc.hazard_writes.borrow().is_empty());
        }
        enc.end();
        assert!(dispatch_census_take().is_empty());
    }
    for i in 0..3 {
        assert_eq!(tensor_backing_bytes(&original[i]), before[i]);
    }
}
