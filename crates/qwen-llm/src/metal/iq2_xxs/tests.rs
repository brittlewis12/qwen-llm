use super::*;
use crate::metal::test_support::{
    assert_offset_guards, metal_test_context, offset_tensor, tensor_backing_bytes,
    tensor_f32_at_offset,
};
use crate::tensor::TensorDesc;

const VARIANTS: [Iq2XxsMatMatVariant; 3] = [
    Iq2XxsMatMatVariant::Auto,
    Iq2XxsMatMatVariant::Scalar,
    Iq2XxsMatMatVariant::Mma,
];

fn payload(k: usize, m: usize) -> Vec<u8> {
    let mut bytes = vec![0u8; k / 256 * m * 66];
    for row in 0..m {
        for block in 0..k / 256 {
            let b = &mut bytes[(row * (k / 256) + block) * 66..][..66];
            let d = if row == 256 {
                0.0
            } else {
                let magnitude = ((row + block * 3) % 7 + 1) as f32 * 0.00137;
                if (row + block) % 3 == 0 {
                    -magnitude
                } else {
                    magnitude
                }
            };
            b[..2].copy_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
            for group in 0..8 {
                let mut metadata = (((row + 3 * block + group) & 15) as u32) << 28;
                for l in 0..4 {
                    b[2 + group * 8 + l] = ((row + 37 * block + 11 * group + 17 * l) & 255) as u8;
                    metadata |=
                        (((row * 13 + 19 * block + 7 * group + 31 * l) & 127) as u32) << (7 * l);
                }
                b[2 + group * 8 + 4..2 + group * 8 + 8].copy_from_slice(&metadata.to_le_bytes());
            }
        }
    }
    bytes
}

fn decode(bytes: &[u8], k: usize, m: usize) -> Vec<f32> {
    crate::codec::dequant_to_f32(
        &TensorDesc {
            name: "iq2_xxs_oracle".into(),
            shape: vec![k as u64, m as u64],
            dtype: GgmlType::IQ2_XXS,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: bytes.len() as u64,
        },
        bytes,
    )
    .expect("independent llama.cpp IQ2_XXS codec")
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
    fn new(ctx: &MetalContext, k: usize, m: usize, n: usize, prefix: usize, suffix: usize) -> Self {
        let bytes = payload(k, m);
        let x: Vec<f32> = (0..k * n)
            .map(|i| (((i * 37 + i / k * 11) % 251) as f32 - 125.0) * 0.0013)
            .collect();
        Self::from_values(ctx, k, m, n, prefix, suffix, &bytes, &x)
    }
    fn from_values(
        ctx: &MetalContext,
        k: usize,
        m: usize,
        n: usize,
        prefix: usize,
        suffix: usize,
        bytes: &[u8],
        x: &[f32],
    ) -> Self {
        let expected = reference(&decode(bytes, k, m), x, k, m, n);
        Self {
            k,
            m,
            n,
            suffix,
            expected,
            weight: offset_tensor(
                ctx,
                18,
                bytes,
                suffix,
                vec![k as u64, m as u64],
                GgmlType::IQ2_XXS,
            ),
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
        variant: Iq2XxsMatMatVariant,
    ) -> Result<(), MetalError> {
        if matches!(variant, Iq2XxsMatMatVariant::Auto) {
            return encode_mat_mat_iq2_xxs_f32(
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
        encode_mat_mat_iq2_xxs_f32_with_variant(
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
    fn run(
        &self,
        ctx: &MetalContext,
        variant: Iq2XxsMatMatVariant,
        expected_path: Path,
    ) -> Vec<f32> {
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
        assert_eq!(census[0].kernel, expected_path.name());
        assert_eq!(census[0].threads_width, expected_path.threads() as u64);
        let (width, height) = match expected_path {
            Path::Gemv => (self.m.div_ceil(8), 1),
            Path::Scalar => (self.m, self.n.div_ceil(32)),
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
fn iq2_xxs_layout_table_and_payload_coverage() {
    assert_eq!(GgmlType::IQ2_XXS.storage_layout(), Some((256, 66)));
    assert_eq!(std::mem::size_of::<Args>(), 16);
    let source = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../kernels/iq2_xxs_grid.metalh"
    ));
    let table = source
        .split("constant ulong iq2_xxs_grid[256] = {")
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
    assert_eq!(grid.len(), 256);
    let bytes = payload(256, 256);
    let decoded = decode(&bytes, 256, 256);
    for group in 0..8 {
        for l in 0..4 {
            let (mut grids, mut signs, mut scales) = ([false; 256], [false; 128], [false; 16]);
            for row in 0..256 {
                let b = &bytes[row * 66..][..66];
                let q = &b[2 + group * 8..][..8];
                let metadata = u32::from_le_bytes(q[4..8].try_into().unwrap());
                let code = q[l] as usize;
                let sign_code = ((metadata >> (7 * l)) & 127) as usize;
                let scale = (metadata >> 28) as usize;
                grids[code] = true;
                signs[sign_code] = true;
                scales[scale] = true;
                let sign_mask = sign_code | ((sign_code.count_ones() as usize & 1) << 7);
                let d = half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32();
                for j in 0..8 {
                    let value = d
                        * (0.5 + scale as f32)
                        * 0.25
                        * grid[code].to_le_bytes()[j] as f32
                        * if sign_mask & (1 << j) != 0 { -1.0 } else { 1.0 };
                    assert_eq!(value, decoded[row * 256 + group * 32 + l * 8 + j]);
                }
            }
            assert!(grids.into_iter().all(|v| v));
            assert!(signs.into_iter().all(|v| v));
            assert!(scales.into_iter().all(|v| v));
        }
    }
}

#[test]
fn iq2_xxs_all_grid_sign_scale_positions_match_codec_basis() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let (k, m, n) = (256, 257, 256);
    let bytes = payload(k, m);
    let mut basis = vec![0.0f32; k * n];
    for i in 0..k {
        basis[i * k + i] = 1.0;
    }
    // The one-hot oracle is the codec output, transposed; avoid a cubic CPU dot.
    let decoded = decode(&bytes, k, m);
    let expected: Vec<f64> = (0..m * n)
        .map(|i| f64::from(decoded[(i % m) * k + i / m]))
        .collect();
    let f = Fixture {
        k,
        m,
        n,
        suffix: 0,
        expected,
        weight: offset_tensor(
            &ctx,
            18,
            &bytes,
            0,
            vec![k as u64, m as u64],
            GgmlType::IQ2_XXS,
        ),
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
        let path = if matches!(variant, Iq2XxsMatMatVariant::Scalar) {
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
            encode_mat_vec_iq2_xxs_f32(
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
    assert!(census.iter().all(|row| row.kernel == GEMV));
    let got = tensor_f32_at_offset(&output);
    for (&a, &b) in got.iter().zip(&f.expected) {
        assert!(a.is_finite() && (f64::from(a) - b).abs() <= 2e-6 * b.abs().max(1e-30));
    }
    assert_offset_guards(&output, 20, 0);
    assert_eq!(tensor_backing_bytes(&f.weight), before[0]);
    assert_eq!(tensor_backing_bytes(&f.input), before[1]);
}

#[test]
fn iq2_xxs_f32_paths_match_f64_tails_offsets_and_real_k() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
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
        let f = Fixture::new(&ctx, k, m, n, 16, if i % 2 == 0 { 0 } else { 28 });
        for variant in VARIANTS {
            let path = if n == 1 {
                Path::Gemv
            } else if matches!(variant, Iq2XxsMatMatVariant::Scalar) {
                Path::Scalar
            } else {
                Path::Mma
            };
            f.run(&ctx, variant, path);
        }
    }
}

#[test]
fn iq2_xxs_f32_scale_extremes_do_not_stage_half() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let (k, m, n) = (256, 6, 9);
    let mut bytes = payload(k, m);
    for (row, bits) in [0u16, 0x8000, 0x0400, 0x3555, 0x7bff, 0xfbff]
        .into_iter()
        .enumerate()
    {
        bytes[row * 66..row * 66 + 2].copy_from_slice(&bits.to_le_bytes());
    }
    let x: Vec<f32> = (0..k * n)
        .map(|i| ((i * 11 % 97) as f32 - 48.0) * 0.013)
        .collect();
    let f = Fixture::from_values(&ctx, k, m, n, 16, 28, &bytes, &x);
    assert!(f.expected.iter().any(|v| v.abs() > 65504.0));
    for variant in VARIANTS {
        f.run(
            &ctx,
            variant,
            if matches!(variant, Iq2XxsMatMatVariant::Scalar) {
                Path::Scalar
            } else {
                Path::Mma
            },
        );
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
fn iq2_xxs_apple7_support_contract_and_scoped_injection() {
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
fn iq2_xxs_unsupported_apple7_falls_back_before_pipeline_lookup() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let f = Fixture::new(&ctx, 512, 17, 9, 16, 28);
    with_mma_support(false, || {
        with_fault(Path::Mma, PipelineFault::Missing, || {
            f.run(&ctx, Iq2XxsMatMatVariant::Auto, Path::Scalar);
            f.run(&ctx, Iq2XxsMatMatVariant::Scalar, Path::Scalar);
            let out = f.output(&ctx);
            let before = tensor_backing_bytes(&out);
            let cmd = ctx.queue.commandBuffer().unwrap();
            let enc = KernelEncoder::begin(&cmd);
            dispatch_census_begin();
            let err = f
                .encode(&ctx, &enc, &out, Iq2XxsMatMatVariant::Mma)
                .unwrap_err();
            assert!(err.to_string().contains("Apple7"), "{err}");
            enc.end();
            assert!(dispatch_census_take().is_empty());
            assert_eq!(tensor_backing_bytes(&out), before);
            let singleton = Fixture::new(&ctx, 256, 7, 1, 16, 0);
            singleton.run(&ctx, Iq2XxsMatMatVariant::Auto, Path::Gemv);
        });
    });
    // Never enable an unsupported hardware path merely to exercise injection.
    if ctx.device.supportsFamily(objc2_metal::MTLGPUFamily::Apple7) {
        with_mma_support(true, || {
            f.run(&ctx, Iq2XxsMatMatVariant::Auto, Path::Mma);
            f.run(&ctx, Iq2XxsMatMatVariant::Mma, Path::Mma);
        });
    }
}

#[test]
fn iq2_xxs_auto_preflights_capability_and_alignment_fallback() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let f = Fixture::new(&ctx, 512, 17, 9, 16, 28);
    for fault in [
        PipelineFault::Missing,
        PipelineFault::Width,
        PipelineFault::Threads,
        PipelineFault::Memory,
    ] {
        with_fault(Path::Mma, fault, || {
            f.run(&ctx, Iq2XxsMatMatVariant::Auto, Path::Scalar);
            let out = f.output(&ctx);
            let cmd = ctx.queue.commandBuffer().unwrap();
            let enc = KernelEncoder::begin(&cmd);
            dispatch_census_begin();
            assert!(
                f.encode(&ctx, &enc, &out, Iq2XxsMatMatVariant::Mma)
                    .is_err()
            );
            let mut invalid = out.clone();
            invalid.offset += 1;
            assert!(
                f.encode(&ctx, &enc, &invalid, Iq2XxsMatMatVariant::Auto)
                    .is_err()
            );
            enc.end();
            assert!(dispatch_census_take().is_empty());
        });
    }
    let f = Fixture::new(&ctx, 512, 17, 9, 20, 0);
    with_fault(Path::Mma, PipelineFault::Missing, || {
        f.run(&ctx, Iq2XxsMatMatVariant::Auto, Path::Scalar);
    });
    let out = f.output(&ctx);
    let cmd = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&cmd);
    dispatch_census_begin();
    assert!(
        f.encode(&ctx, &enc, &out, Iq2XxsMatMatVariant::Mma)
            .is_err()
    );
    enc.end();
    assert!(dispatch_census_take().is_empty());
    // If even the fallback cannot run, fail without a partial dispatch.
    with_fault(Path::Scalar, PipelineFault::Memory, || {
        let out = f.output(&ctx);
        let cmd = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&cmd);
        dispatch_census_begin();
        assert!(
            f.encode(&ctx, &enc, &out, Iq2XxsMatMatVariant::Auto)
                .is_err()
        );
        enc.end();
        assert!(dispatch_census_take().is_empty());
    });
    let singleton = Fixture::new(&ctx, 256, 7, 1, 20, 0);
    with_fault(Path::Mma, PipelineFault::Missing, || {
        singleton.run(&ctx, Iq2XxsMatMatVariant::Auto, Path::Gemv);
    });
}

#[test]
fn iq2_xxs_capacity_device_ids_and_dimensions() {
    for path in [Path::Gemv, Path::Scalar, Path::Mma] {
        check_capacity(path, 32, path.threads(), 0, path.scratch()).unwrap();
        assert!(check_capacity(path, 16, 128, 0, 32768).is_err());
        assert!(check_capacity(path, 32, path.threads() - 1, 0, 32768).is_err());
        assert!(check_capacity(path, 32, 128, 1, path.scratch()).is_err());
        assert!(check_capacity(path, 32, 128, usize::MAX, usize::MAX - 1).is_err());
    }
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
fn iq2_xxs_concurrent_shared_inputs_distinct_outputs_match_serial() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let mut cases = vec![
        (1, true, Iq2XxsMatMatVariant::Auto, 16),
        (1, false, Iq2XxsMatMatVariant::Auto, 16),
        (9, false, Iq2XxsMatMatVariant::Auto, 16),
        (9, false, Iq2XxsMatMatVariant::Scalar, 16),
        (33, false, Iq2XxsMatMatVariant::Auto, 20),
    ];
    if ctx.device.supportsFamily(objc2_metal::MTLGPUFamily::Apple7) {
        cases.push((9, false, Iq2XxsMatMatVariant::Mma, 16));
    }
    for (n, vector, variant, prefix) in cases {
        let f = Fixture::new(&ctx, 512, 17, n, prefix, 28);
        let before = [
            tensor_backing_bytes(&f.weight),
            tensor_backing_bytes(&f.input),
        ];
        let encode = |enc: &KernelEncoder, out: &MetalTensor| {
            if vector {
                let x = f.input.view_subrange(0, vec![f.k as u64]);
                let y = out.view_subrange(0, vec![f.m as u64]);
                encode_mat_vec_iq2_xxs_f32(&ctx, enc, &f.weight, &x, &y, f.k, f.m)
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

#[test]
fn iq2_xxs_invalid_bindings_do_not_dispatch_on_serial_or_concurrent() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let f = Fixture::new(&ctx, 256, 17, 9, 16, 0);
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
                encode_mat_mat_iq2_xxs_f32_with_variant(
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
            assert!(encode_mat_vec_iq2_xxs_f32(&ctx, &enc, &t[0], &t[1], &t[2], f.k, f.m).is_err());
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

#[test]
fn iq2_xxs_foreign_device_bindings_and_encoder_are_rejected_when_available() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let devices = objc2_metal::MTLCopyAllDevices();
    let Some(device) = devices
        .iter()
        .find(|d| d.registryID() != ctx.device.registryID())
    else {
        return;
    };
    let f = Fixture::new(&ctx, 256, 17, 9, 16, 0);
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
            encode_mat_mat_iq2_xxs_f32(&ctx, &enc, &t[0], &t[1], &t[2], f.k, f.m, f.n).is_err()
        );
    }
    enc.end();
    assert!(dispatch_census_take().is_empty());
    let queue = device.newCommandQueue().unwrap();
    let cmd = queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&cmd);
    dispatch_census_begin();
    assert!(
        f.encode(&ctx, &enc, &out, Iq2XxsMatMatVariant::Auto)
            .is_err()
    );
    enc.end();
    assert!(dispatch_census_take().is_empty());
}
