use super::*;
use crate::metal::test_support::{
    assert_offset_guards, metal_test_context, offset_tensor, tensor_backing_bytes,
    tensor_f32_at_offset,
};
use crate::tensor::TensorDesc;

const VARIANTS: [Iq2XsMatMatVariant; 3] = [
    Iq2XsMatMatVariant::Auto,
    Iq2XsMatMatVariant::Scalar,
    Iq2XsMatMatVariant::Mma,
];

fn payload(k: usize, m: usize) -> Vec<u8> {
    let mut bytes = vec![0u8; k / 256 * m * 74];
    for row in 0..m {
        for block in 0..k / 256 {
            let b = &mut bytes[(row * (k / 256) + block) * 74..][..74];
            let d = if row == 512 {
                0.0
            } else {
                let v = ((row + 3 * block) % 7 + 1) as f32 * 0.00137;
                if (row + block) % 3 == 0 { -v } else { v }
            };
            b[..2].copy_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
            for group in 0..8 {
                let low = (row + 3 * block + group) & 15;
                let high = (row / 16 + 5 * block + 3 * group) & 15;
                b[66 + group] = (low | (high << 4)) as u8;
                for l in 0..4 {
                    let grid = (row + 37 * block + 11 * group + 17 * l) & 511;
                    let sign = (13 * row + 19 * block + 7 * group + 31 * l) & 127;
                    let code = (grid | (sign << 9)) as u16;
                    let offset = 2 + 2 * (4 * group + l);
                    b[offset..offset + 2].copy_from_slice(&code.to_le_bytes());
                }
            }
        }
    }
    bytes
}

fn decode(bytes: &[u8], k: usize, m: usize) -> Vec<f32> {
    crate::codec::dequant_to_f32(
        &TensorDesc {
            name: "iq2_xs_dense_oracle".into(),
            shape: vec![k as u64, m as u64],
            dtype: GgmlType::IQ2_XS,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: bytes.len() as u64,
        },
        bytes,
    )
    .expect("independent llama.cpp IQ2_XS codec")
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
                GgmlType::IQ2_XS,
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
    ) -> Result<(), MetalError> {
        encode_mat_mat_iq2_xs_f32(
            ctx,
            enc,
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
        variant: Option<Iq2XsMatMatVariant>,
        expected_path: Path,
    ) -> Vec<f32> {
        let out = self.output(ctx);
        let before = [
            tensor_backing_bytes(&self.weight),
            tensor_backing_bytes(&self.input),
        ];
        dispatch_census_begin();
        let (result, substitutions) = with_iq2_xs_matmat_variant(variant, || {
            one_shot(ctx, |enc| self.encode(ctx, enc, &out))
        });
        let census = dispatch_census_take();
        result.unwrap();
        assert_eq!(substitutions, usize::from(expected_path == Path::Mma));
        assert_eq!(census.len(), 1);
        assert_eq!(census[0].kernel, expected_path.name());
        assert_eq!(census[0].threads_width, expected_path.threads() as u64);
        let (width, height) = match expected_path {
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

fn variants(ctx: &MetalContext) -> Vec<Option<Iq2XsMatMatVariant>> {
    let mut out = vec![
        None,
        Some(Iq2XsMatMatVariant::Scalar),
        Some(Iq2XsMatMatVariant::Auto),
    ];
    if ctx.device.supportsFamily(objc2_metal::MTLGPUFamily::Apple7) {
        out.push(Some(Iq2XsMatMatVariant::Mma));
    }
    out
}

fn expected_path(
    ctx: &MetalContext,
    variant: Option<Iq2XsMatMatVariant>,
    n: usize,
    offset: u64,
) -> Path {
    if n > 1
        && offset.is_multiple_of(16)
        && ctx.device.supportsFamily(objc2_metal::MTLGPUFamily::Apple7)
        && matches!(
            variant,
            None | Some(Iq2XsMatMatVariant::Auto | Iq2XsMatMatVariant::Mma)
        )
    {
        Path::Mma
    } else {
        Path::Scalar
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

fn with_support<R>(supported: bool, f: impl FnOnce() -> R) -> R {
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
fn iq2_xs_dense_layout_grid_sign_scale_coverage() {
    assert_eq!(GgmlType::IQ2_XS.storage_layout(), Some((256, 74)));
    assert_eq!(std::mem::size_of::<ScalarArgs>(), 12);
    assert_eq!(std::mem::size_of::<MmaArgs>(), 16);
    let header = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../kernels/iq2_xs_grid.metalh"
    ));
    let text = header
        .split("constant ulong mv_iq2xs_grid[512] = {")
        .nth(1)
        .unwrap()
        .split("};")
        .next()
        .unwrap();
    let grid: Vec<u64> = text
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
    assert_eq!(grid.len(), 512);
    let bytes = payload(256, 512);
    let decoded = decode(&bytes, 256, 512);
    for group in 0..8 {
        for l in 0..4 {
            let (mut grids, mut signs, mut scales) = ([false; 512], [false; 128], [false; 16]);
            for row in 0..512 {
                let b = &bytes[row * 74..][..74];
                let offset = 2 + 2 * (4 * group + l);
                let packed = u16::from_le_bytes([b[offset], b[offset + 1]]) as usize;
                let code = packed & 511;
                let sign = packed >> 9;
                let scale = usize::from((b[66 + group] >> (4 * (l / 2))) & 15);
                grids[code] = true;
                signs[sign] = true;
                scales[scale] = true;
                let sign_mask = sign | ((sign.count_ones() as usize & 1) << 7);
                let d = half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32();
                for (j, value) in grid[code].to_le_bytes().into_iter().enumerate() {
                    let expected = d
                        * (0.5 + scale as f32)
                        * 0.25
                        * f32::from(value)
                        * if sign_mask & (1 << j) == 0 { 1.0 } else { -1.0 };
                    assert_eq!(expected, decoded[row * 256 + group * 32 + l * 8 + j]);
                }
            }
            assert!(grids.into_iter().all(|v| v));
            assert!(signs.into_iter().all(|v| v));
            assert!(scales.into_iter().all(|v| v));
        }
    }
}

#[test]
fn iq2_xs_dense_scope_restores_production_counts_unwind_and_threads() {
    assert_eq!(STATE.with(|s| s.get()), (None, 0));
    assert!(auto_enabled());
    let (value, count) = with_iq2_xs_matmat_variant(Some(Iq2XsMatMatVariant::Scalar), || {
        assert!(!auto_enabled());
        STATE.with(|s| s.set((Some(Iq2XsMatMatVariant::Scalar), 7)));
        let (_, inner) = with_iq2_xs_matmat_variant(None, || {
            assert_eq!(STATE.with(|s| s.get()), (None, 0));
            assert!(auto_enabled());
        });
        assert_eq!(inner, 0);
        assert!(!auto_enabled());
        with_iq2_xs_matmat_variant(Some(Iq2XsMatMatVariant::Auto), || assert!(auto_enabled()));
        let unwind = std::panic::catch_unwind(|| {
            with_iq2_xs_matmat_variant(None, || {
                assert!(auto_enabled());
                panic!("scope restoration")
            })
        });
        assert!(unwind.is_err());
        assert!(!auto_enabled());
        assert_eq!(
            STATE.with(|s| s.get()),
            (Some(Iq2XsMatMatVariant::Scalar), 7)
        );
        std::thread::spawn(|| {
            assert_eq!(STATE.with(|s| s.get()), (None, 0));
            assert!(auto_enabled());
        })
        .join()
        .unwrap();
        42
    });
    assert_eq!((value, count), (42, 7));
    assert_eq!(STATE.with(|s| s.get()), (None, 0));
    assert!(auto_enabled());
}

#[test]
fn iq2_xs_dense_capacity_devices_and_address_boundaries() {
    check_mma_support(true).unwrap();
    assert!(check_mma_support(false).is_err());
    for path in [Path::Scalar, Path::Mma] {
        check_capacity(path, 32, path.threads(), 0, path.scratch()).unwrap();
        assert!(check_capacity(path, 16, 128, 0, 32768).is_err());
        assert!(check_capacity(path, 32, path.threads() - 1, 0, 32768).is_err());
        assert!(check_capacity(path, 32, 128, 1, path.scratch()).is_err());
        assert!(check_capacity(path, 32, 128, usize::MAX, usize::MAX).is_err());
    }
    check_devices(7, 7, [7; 3]).unwrap();
    assert!(check_devices(7, 8, [7; 3]).is_err());
    for i in 0..3 {
        let mut ids = [7; 3];
        ids[i] = 8;
        assert!(check_devices(7, 7, ids).is_err());
    }
    let validate = super::super::mat_mat::validate_iq2_xs_mat_mat_addressing;
    validate(256, 1, 16_777_216).unwrap();
    assert!(validate(256, 1, 16_777_217).is_err());
    validate(256, 1usize << 31, 2).unwrap();
    assert!(validate(256, (1usize << 31) + 1, 2).is_err());
    for dims in [
        (0, 1, 1),
        (256, 0, 1),
        (256, 1, 0),
        (257, 1, 1),
        (usize::MAX, 1, 1),
        (256, usize::MAX, 1),
    ] {
        assert!(validate(dims.0, dims.1, dims.2).is_err());
    }
}

#[test]
fn iq2_xs_dense_all_coefficients_match_codec_basis() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let (k, m, n) = (256, 513, 256);
    let bytes = payload(k, m);
    let decoded = decode(&bytes, k, m);
    let mut basis = vec![0.0f32; k * n];
    for i in 0..k {
        basis[i * k + i] = 1.0;
    }
    let f = Fixture {
        k,
        m,
        n,
        suffix: 0,
        expected: (0..m * n)
            .map(|i| f64::from(decoded[(i % m) * k + i / m]))
            .collect(),
        weight: offset_tensor(
            &ctx,
            18,
            &bytes,
            0,
            vec![k as u64, m as u64],
            GgmlType::IQ2_XS,
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
    for variant in variants(&ctx) {
        let got = f.run(&ctx, variant, expected_path(&ctx, variant, n, 16));
        for (a, b) in got.into_iter().zip(&f.expected) {
            assert!(
                (f64::from(a) - b).abs() <= 2e-6 * b.abs().max(1e-30),
                "coefficient mismatch {a} vs {b}"
            );
        }
    }
}

#[test]
fn iq2_xs_dense_f64_tails_offsets_flat_views_and_real_k() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let mut cases = Vec::new();
    for (i, n) in [1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 127, 128, 129, 512]
        .into_iter()
        .enumerate()
    {
        cases.push((
            [256, 512, 768][i % 3],
            [1, 3, 7, 8, 9, 15, 16, 17, 63, 64, 65][i % 11],
            n,
        ));
    }
    for k in [5120, 6144, 17408] {
        for n in [1, 2, 9, 33] {
            cases.push((k, 17, n));
        }
    }
    for (i, (k, m, n)) in cases.into_iter().enumerate() {
        let mut f = Fixture::new(&ctx, k, m, n, 16, if i % 2 == 0 { 0 } else { 28 });
        if i % 2 == 0 {
            f.input.shape = vec![(k * n) as u64];
        }
        for variant in variants(&ctx) {
            f.run(&ctx, variant, expected_path(&ctx, variant, n, 16));
        }
        let out = f.output(&ctx).view_subrange(0, vec![(m * n) as u64]);
        let (result, count) = with_iq2_xs_matmat_variant(Some(Iq2XsMatMatVariant::Auto), || {
            one_shot(&ctx, |enc| f.encode(&ctx, enc, &out))
        });
        result.unwrap();
        assert_eq!(
            count,
            usize::from(expected_path(&ctx, Some(Iq2XsMatMatVariant::Auto), n, 16) == Path::Mma)
        );
        assert_numerical(&tensor_f32_at_offset(&out), &f.expected);
        assert_offset_guards(&out, 20, f.suffix);
    }
}

#[test]
fn iq2_xs_dense_scale_extremes_keep_f32_per_row() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let bits = [
        0u16, 0x8000, 0x0001, 0x8001, 0x03ff, 0x83ff, 0x0400, 0x3555, 0x7bff, 0xfbff,
    ];
    let (k, m, n) = (256, bits.len(), 9);
    let mut bytes = payload(k, m);
    for (b, d) in bytes.chunks_exact_mut(74).zip(bits) {
        b[..2].copy_from_slice(&d.to_le_bytes());
    }
    let input: Vec<f32> = (0..k * n)
        .map(|i| ((i * 11 % 97) as f32 - 48.0) * 0.013)
        .collect();
    let f = Fixture::from_values(&ctx, k, m, n, 16, 28, &bytes, &input);
    assert!(f.expected.iter().any(|v| v.abs() > 65504.0));
    for variant in variants(&ctx) {
        let got = f.run(&ctx, variant, expected_path(&ctx, variant, n, 16));
        for row in 0..m {
            let actual: Vec<f32> = (0..n).map(|t| got[t * m + row]).collect();
            let reference: Vec<f64> = (0..n).map(|t| f.expected[t * m + row]).collect();
            assert_numerical(&actual, &reference);
        }
    }
}

fn rejected(f: &Fixture, ctx: &MetalContext, variant: Option<Iq2XsMatMatVariant>) -> MetalError {
    let out = f.output(ctx);
    let before = tensor_backing_bytes(&out);
    let cmd = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin_concurrent(&cmd);
    dispatch_census_begin();
    let (result, count) = with_iq2_xs_matmat_variant(variant, || f.encode(ctx, &enc, &out));
    let err = result.unwrap_err();
    assert_eq!(count, 0);
    #[cfg(debug_assertions)]
    {
        assert!(enc.hazard_reads.borrow().is_empty());
        assert!(enc.hazard_writes.borrow().is_empty());
    }
    enc.end();
    assert!(dispatch_census_take().is_empty());
    assert_eq!(tensor_backing_bytes(&out), before);
    err
}

#[test]
fn iq2_xs_dense_production_auto_preflight_fallback_and_strict_mma() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let f = Fixture::new(&ctx, 512, 17, 9, 16, 28);
    f.run(&ctx, None, expected_path(&ctx, None, f.n, f.input.offset));
    with_support(false, || {
        with_fault(Path::Mma, PipelineFault::Missing, || {
            f.run(&ctx, None, Path::Scalar);
            f.run(&ctx, Some(Iq2XsMatMatVariant::Auto), Path::Scalar);
            let err = rejected(&f, &ctx, Some(Iq2XsMatMatVariant::Mma));
            assert!(err.to_string().contains("Apple7"), "{err}");
        })
    });
    // Inject pipeline faults only when the real device permits reaching that pipeline.
    if ctx.device.supportsFamily(objc2_metal::MTLGPUFamily::Apple7) {
        for fault in [
            PipelineFault::Missing,
            PipelineFault::Width,
            PipelineFault::Threads,
            PipelineFault::Memory,
        ] {
            with_fault(Path::Mma, fault, || {
                f.run(&ctx, None, Path::Scalar);
                f.run(&ctx, Some(Iq2XsMatMatVariant::Auto), Path::Scalar);
                rejected(&f, &ctx, Some(Iq2XsMatMatVariant::Mma));
            });
        }
        f.run(&ctx, Some(Iq2XsMatMatVariant::Mma), Path::Mma);
    }
    let unaligned = Fixture::new(&ctx, 512, 17, 9, 20, 0);
    unaligned.run(&ctx, None, Path::Scalar);
    unaligned.run(&ctx, Some(Iq2XsMatMatVariant::Auto), Path::Scalar);
    assert!(
        rejected(&unaligned, &ctx, Some(Iq2XsMatMatVariant::Mma))
            .to_string()
            .contains("16-byte")
    );
    with_fault(Path::Scalar, PipelineFault::Memory, || {
        rejected(&unaligned, &ctx, Some(Iq2XsMatMatVariant::Auto));
        rejected(&unaligned, &ctx, None);
    });
    let singleton = Fixture::new(&ctx, 256, 17, 1, 20, 0);
    with_support(false, || {
        with_fault(Path::Mma, PipelineFault::Missing, || {
            singleton.run(&ctx, None, Path::Scalar);
            singleton.run(&ctx, Some(Iq2XsMatMatVariant::Mma), Path::Scalar);
        })
    });
}

#[test]
fn iq2_xs_dense_concurrent_shared_inputs_distinct_outputs_and_counts() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    for n in [1, 9, 33] {
        let f = Fixture::new(&ctx, 512, 17, n, 16, 28);
        let before = [
            tensor_backing_bytes(&f.weight),
            tensor_backing_bytes(&f.input),
        ];
        for variant in variants(&ctx) {
            let serial = f.run(&ctx, variant, expected_path(&ctx, variant, n, 16));
            let output = [f.output(&ctx), f.output(&ctx)];
            let cmd = ctx.queue.commandBuffer().unwrap();
            dispatch_census_begin();
            let enc = KernelEncoder::begin_concurrent(&cmd);
            let (result, count) = with_iq2_xs_matmat_variant(variant, || {
                for out in &output {
                    f.encode(&ctx, &enc, out)?;
                }
                let mut invalid = output[0].clone();
                invalid.offset += 1;
                assert!(f.encode(&ctx, &enc, &invalid).is_err());
                Ok::<(), MetalError>(())
            });
            result.unwrap();
            let path = expected_path(&ctx, variant, n, 16);
            assert_eq!(count, if path == Path::Mma { 2 } else { 0 });
            #[cfg(debug_assertions)]
            {
                assert_eq!(enc.hazard_reads.borrow().len(), 4);
                assert_eq!(enc.hazard_writes.borrow().len(), 2);
            }
            enc.end();
            let census = dispatch_census_take();
            assert_eq!(census.len(), 2);
            assert!(
                census
                    .iter()
                    .all(|r| r.kernel == path.name() && r.encoder_concurrent)
            );
            cmd.commit();
            wait_completed(&cmd).unwrap();
            let serial_reference: Vec<f64> = serial.into_iter().map(f64::from).collect();
            for out in output {
                let got = tensor_f32_at_offset(&out);
                assert_numerical(&got, &f.expected);
                assert_numerical(&got, &serial_reference);
                assert_offset_guards(&out, 20, 28);
            }
            assert_eq!(tensor_backing_bytes(&f.weight), before[0]);
            assert_eq!(tensor_backing_bytes(&f.input), before[1]);
        }
    }
}

#[test]
fn iq2_xs_dense_invalid_bindings_reject_without_dispatch_or_hazard_notes() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let f = Fixture::new(&ctx, 256, 17, 9, 16, 28);
    let original = [f.weight.clone(), f.input.clone(), f.output(&ctx)];
    let before = original.each_ref().map(tensor_backing_bytes);
    for concurrent in [false, true] {
        let cmd = ctx.queue.commandBuffer().unwrap();
        let enc = if concurrent {
            KernelEncoder::begin_concurrent(&cmd)
        } else {
            KernelEncoder::begin(&cmd)
        };
        dispatch_census_begin();
        for variant in [
            None,
            Some(Iq2XsMatMatVariant::Scalar),
            Some(Iq2XsMatMatVariant::Auto),
            Some(Iq2XsMatMatVariant::Mma),
        ] {
            let (_, count) = with_iq2_xs_matmat_variant(variant, || {
                let encode = |t: &[MetalTensor; 3], k, m, n| {
                    encode_mat_mat_iq2_xs_f32(&ctx, &enc, &t[0], &t[1], &t[2], k, m, n)
                };
                for which in 0..3 {
                    for fault in 0..6 {
                        let mut t = original.clone();
                        match fault {
                            0 => t[which].offset += 1,
                            1 => t[which].offset = t[which].buffer.length() as u64,
                            2 => t[which].offset = u64::MAX - 15,
                            3 => t[which].dtype = GgmlType::F16,
                            4 => t[which].shape = vec![1],
                            _ => t[which].shape = vec![u64::MAX, 2],
                        }
                        assert!(encode(&t, f.k, f.m, f.n).is_err());
                    }
                }
                let mut t = original.clone();
                t[0].shape.push(1);
                assert!(encode(&t, f.k, f.m, f.n).is_err());
                let mut t = original.clone();
                t[2].provenance = MetalTensorProvenance::OwnedWeightReadOnly;
                assert!(encode(&t, f.k, f.m, f.n).is_err());
                for source in 0..2 {
                    let mut t = original.clone();
                    t[2].buffer = t[source].buffer.clone();
                    t[2].offset = t[source].offset.next_multiple_of(4);
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
                    (256, 1, 16_777_217),
                    (256, 1usize << 31, 3),
                ] {
                    assert!(encode(&original, k, m, n).is_err());
                }
            });
            assert_eq!(count, 0);
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

#[test]
fn iq2_xs_dense_foreign_device_bindings_and_encoder_rejected_when_available() {
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
    let original = [f.weight.clone(), f.input.clone(), f.output(&ctx)];
    let cmd = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin_concurrent(&cmd);
    dispatch_census_begin();
    for variant in [
        None,
        Some(Iq2XsMatMatVariant::Auto),
        Some(Iq2XsMatMatVariant::Mma),
    ] {
        let (_, count) = with_iq2_xs_matmat_variant(variant, || {
            for which in 0..3 {
                let mut t = original.clone();
                t[which].buffer = device
                    .newBufferWithLength_options(
                        t[which].buffer.length(),
                        MTLResourceOptions::StorageModeShared,
                    )
                    .unwrap();
                assert!(
                    encode_mat_mat_iq2_xs_f32(&ctx, &enc, &t[0], &t[1], &t[2], f.k, f.m, f.n)
                        .is_err()
                );
            }
            let queue = device.newCommandQueue().unwrap();
            let other_cmd = queue.commandBuffer().unwrap();
            let other = KernelEncoder::begin(&other_cmd);
            assert!(f.encode(&ctx, &other, &original[2]).is_err());
            other.end();
        });
        assert_eq!(count, 0);
    }
    #[cfg(debug_assertions)]
    {
        assert!(enc.hazard_reads.borrow().is_empty());
        assert!(enc.hazard_writes.borrow().is_empty());
    }
    enc.end();
    assert!(dispatch_census_take().is_empty());
}

#[test]
fn iq2_xs_dense_forced_scalar_and_production_fallback_match_incumbent() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    for (n, prefix) in [1, 2, 9, 33]
        .into_iter()
        .flat_map(|n| [16, 20].map(|prefix| (n, prefix)))
    {
        let f = Fixture::new(&ctx, 512, 17, n, prefix, 28);
        let incumbent = f.output(&ctx);
        one_shot(&ctx, |enc| {
            super::super::mat_mat::encode_mat_mat_block256_f32(
                &ctx,
                enc,
                &f.weight,
                &f.input,
                &incumbent,
                f.k,
                f.m,
                f.n,
                GgmlType::IQ2_XS,
                SCALAR,
            )
        })
        .unwrap();
        let reference: Vec<u32> = tensor_f32_at_offset(&incumbent)
            .into_iter()
            .map(f32::to_bits)
            .collect();
        let mut comparisons = vec![Some(Iq2XsMatMatVariant::Scalar)];
        if expected_path(&ctx, None, n, f.input.offset) == Path::Scalar {
            comparisons.push(None);
        }
        for variant in comparisons {
            let got = f.run(&ctx, variant, Path::Scalar);
            // Same incumbent shader and bindings: this guards host-side routing/ABI,
            // and imposes no bitwise requirement on the new MMA arithmetic.
            assert_eq!(
                got.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
                reference
            );
        }
        for variant in std::iter::once(None).chain(VARIANTS.into_iter().map(Some)) {
            let out = f.output(&ctx);
            dispatch_census_begin();
            let (result, substitutions) = with_iq2_xs_matmat_variant(variant, || {
                one_shot(&ctx, |enc| {
                    encode_mat_mat_iq2_xs_f32_scalar(
                        &ctx, enc, &f.weight, &f.input, &out, f.k, f.m, f.n,
                    )
                })
            });
            let census = dispatch_census_take();
            result.unwrap();
            assert_eq!(substitutions, 0);
            assert_eq!(census.len(), 1);
            assert_eq!(census[0].kernel, SCALAR);
            assert_eq!(
                tensor_f32_at_offset(&out)
                    .into_iter()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>(),
                reference
            );
            assert_offset_guards(&out, 20, f.suffix);
        }
    }
}

#[test]
fn iq2_xs_dense_n1_controls_unchanged() {
    const CHILD: &str = "QWEN_IQ2_XS_N1_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        // Both existing flags are OnceLock-cached. Exercise each combination in
        // a fresh test process without mutating another test's environment.
        let qualified = concat!(module_path!(), "::iq2_xs_dense_n1_controls_unchanged");
        let test_name = qualified.split_once("::").unwrap().1;
        for shortcut in ["0", "1"] {
            for fast in ["0", "1"] {
                let status = std::process::Command::new(std::env::current_exe().unwrap())
                    .args(["--exact", test_name, "--nocapture", "--test-threads=1"])
                    .env(CHILD, "1")
                    .env("QWEN_MATMAT_N1_MATVEC", shortcut)
                    .env("QWEN_MATVEC_IQ2_XS_FAST", fast)
                    .status()
                    .unwrap();
                assert!(status.success(), "N1 shortcut={shortcut}, GEMV fast={fast}");
            }
        }
        return;
    }
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let shortcut = crate::env_flag::read_default_on("QWEN_MATMAT_N1_MATVEC");
    let fast = crate::env_flag::read_default_on("QWEN_MATVEC_IQ2_XS_FAST");
    let gemv_name = if fast {
        "kernel_mat_vec_iq2_xs_f32_fast"
    } else {
        "kernel_mat_vec_iq2_xs_f32"
    };
    let f = Fixture::new(&ctx, 512, 17, 1, 16, 28);
    let vector_x = f.input.view_subrange(0, vec![f.k as u64]);
    let baseline = [f.output(&ctx), f.output(&ctx)];
    one_shot(&ctx, |enc| {
        encode_mat_vec_iq2_xs_f32(&ctx, enc, &f.weight, &vector_x, &baseline[0], f.k, f.m)?;
        super::super::mat_mat::encode_mat_mat_block256_f32(
            &ctx,
            enc,
            &f.weight,
            &vector_x,
            &baseline[1],
            f.k,
            f.m,
            1,
            GgmlType::IQ2_XS,
            SCALAR,
        )
    })
    .unwrap();
    let expected: [Vec<u32>; 2] = baseline.each_ref().map(|t| {
        tensor_f32_at_offset(t)
            .into_iter()
            .map(f32::to_bits)
            .collect()
    });
    let before = [
        tensor_backing_bytes(&f.weight),
        tensor_backing_bytes(&f.input),
    ];
    for variant in std::iter::once(None).chain(VARIANTS.into_iter().map(Some)) {
        let out = [
            f.output(&ctx),
            f.output(&ctx),
            f.output(&ctx),
            f.output(&ctx),
        ];
        let cmd = ctx.queue.commandBuffer().unwrap();
        dispatch_census_begin();
        let enc = KernelEncoder::begin_concurrent(&cmd);
        let ((), count) = with_iq2_xs_matmat_variant(variant, || {
            // Direct matrix entry preserves its old scalar lineage at N=1.
            f.encode(&ctx, &enc, &out[0]).unwrap();
            crate::metal_forward::encode_mat_mat_dispatch_with_policy(
                &ctx, &enc, &f.weight, &vector_x, &out[1], f.k, f.m, 1, true,
            )
            .unwrap();
            crate::metal_forward::encode_mat_mat_dispatch_with_policy(
                &ctx, &enc, &f.weight, &vector_x, &out[2], f.k, f.m, 1, false,
            )
            .unwrap();
            crate::metal_forward::encode_mat_vec_dispatch(
                &ctx, &enc, &f.weight, &vector_x, &out[3], f.k, f.m,
            )
            .unwrap();
        });
        assert_eq!(count, 0);
        #[cfg(debug_assertions)]
        {
            assert_eq!(enc.hazard_reads.borrow().len(), 8);
            assert_eq!(enc.hazard_writes.borrow().len(), 4);
        }
        enc.end();
        let census = dispatch_census_take();
        assert_eq!(census.len(), 4);
        let names = [
            SCALAR,
            if shortcut { gemv_name } else { SCALAR },
            SCALAR,
            gemv_name,
        ];
        for (row, name) in census.iter().zip(names) {
            assert_eq!(row.kernel, name);
        }
        cmd.commit();
        wait_completed(&cmd).unwrap();
        for (i, t) in out.iter().enumerate() {
            let got = tensor_f32_at_offset(t);
            assert_numerical(&got, &f.expected);
            let reference = if i == 3 || (i == 1 && shortcut) {
                &expected[0]
            } else {
                &expected[1]
            };
            assert_eq!(
                &got.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
                reference
            );
            assert_offset_guards(t, 20, 28);
        }
        assert_eq!(tensor_backing_bytes(&f.weight), before[0]);
        assert_eq!(tensor_backing_bytes(&f.input), before[1]);
    }
}
