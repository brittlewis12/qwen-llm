//! `encode_dense_skinny_q5_k` (`kernel_dense_skinny_mma_q5_K_f32`): F32
//! agreement with the production one-row Q5_K kernel applied to each column,
//! validation before any dispatch, and an opt-in timing probe at the V4.1
//! decode shapes (`QWEN_DENSE_SKINNY_PROBE=1`, under the Metal lease).

use super::*;

/// Deterministic xorshift bytes: random quant/scale/high bits, with every
/// block's `d` and `dmin` replaced by small positive halves so the weights
/// stay finite.
fn random_q5_k(n_in: usize, n_out: usize, seed: u64) -> Vec<u8> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let blocks = n_in / 256 * n_out;
    let mut bytes: Vec<u8> = (0..blocks * 176).map(|_| next() as u8).collect();
    for block in bytes.as_chunks_mut::<176>().0 {
        for offset in [0, 2] {
            let v = 0.0005 + (next() % 1000) as f32 * 0.00002;
            block[offset..offset + 2].copy_from_slice(&half::f16::from_f32(v).to_le_bytes());
        }
    }
    bytes
}

fn random_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0xD1B5_4A32_D192_ED03) | 1;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        })
        .collect()
}

/// F32 tensor at offset 0 (16-byte aligned, as the skinny input needs).
fn f32_tensor(ctx: &MetalContext, values: &[f32], shape: &[usize]) -> MetalTensor {
    MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(values),
        shape.iter().map(|&v| v as u64).collect(),
        GgmlType::F32,
    )
    .unwrap()
}

fn weight(ctx: &MetalContext, bytes: &[u8], n_in: usize, n_out: usize) -> MetalTensor {
    MetalTensor::from_bytes(ctx, bytes, vec![n_in as u64, n_out as u64], GgmlType::Q5_K).unwrap()
}

/// The decode path's per-column projection: the production one-row native
/// Q5_K kernel (`encode_all_slots_down_native`, one slot per column).
#[allow(clippy::too_many_arguments)]
fn encode_per_column(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w: &MetalTensor,
    x: &MetalTensor,
    ids: &MetalTensor,
    status: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    cols: usize,
) -> Result<(), MetalError> {
    let bank = w.view_bytes(0, vec![n_in as u64, n_out as u64, 1]);
    encode_all_slots_down_native(ctx, enc, &bank, x, ids, status, y, n_in, n_out, 1, cols)
}

/// V4.1 decode shapes (shared gate/up and down, q_b, output_b) and a ragged
/// small one, B in {1, 2, 3, 4, 8}, random weights and activations.
#[test]
fn dense_skinny_q5_k_matches_per_column_kernel_to_f32_rounding() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let mut worst = 0f32;
    for (seed, (n_in, n_out)) in [
        (5120usize, 2304usize),
        (2304, 5120),
        (1280, 32768),
        (8192, 5120),
        (512, 40),
    ]
    .into_iter()
    .enumerate()
    {
        let w = weight(&ctx, &random_q5_k(n_in, n_out, seed as u64), n_in, n_out);
        for cols in [1usize, 2, 3, 4, 8] {
            let x = f32_tensor(
                &ctx,
                &random_f32(n_in * cols, (seed * 16 + cols) as u64),
                &[n_in, cols],
            );
            let ids = i32_view(&ctx, &vec![0; cols]);
            let status = i32_view(&ctx, &[ROUTE_STATUS_READY]);
            let want = f32_view(&ctx, &vec![f32::NAN; n_out * cols], &[n_out, cols], 0);
            let got = f32_view(&ctx, &vec![f32::NAN; n_out * cols], &[n_out, cols], 28);
            dispatch_census_begin();
            one_shot(&ctx, |enc| {
                encode_per_column(&ctx, enc, &w, &x, &ids, &status, &want, n_in, n_out, cols)?;
                encode_dense_skinny_q5_k(&ctx, enc, &w, &x, &got, n_in, n_out, cols)
            })
            .unwrap();
            let census = dispatch_census_take();
            assert_eq!(
                census.last().map(|r| r.kernel.as_str()),
                Some("kernel_dense_skinny_mma_q5_K_f32")
            );
            let (w_out, g_out) = (tensor_f32_at_offset(&want), tensor_f32_at_offset(&got));
            assert_offset_guards(&got, 20, 28);
            let scale = w_out.iter().fold(0f32, |m, v| m.max(v.abs()));
            assert!(scale > 0.0 && scale.is_finite());
            for (i, (p, q)) in w_out.iter().zip(&g_out).enumerate() {
                let err = (p - q).abs();
                assert!(
                    err <= 1e-5 * scale + 1e-6,
                    "n_in {n_in} n_out {n_out} cols {cols} element {i}: {q} vs {p} (scale {scale})"
                );
                worst = worst.max(err / scale);
            }
        }
    }
    eprintln!("[dense-skinny] max |skinny - per-column| / output scale {worst:.2e}");
}

#[test]
fn dense_skinny_q5_k_rejects_bad_bindings_before_dispatch() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let (n_in, n_out) = (512usize, 32usize);
    let bytes = random_q5_k(n_in, n_out, 7);
    let w = weight(&ctx, &bytes, n_in, n_out);
    let x = f32_tensor(&ctx, &random_f32(n_in * 2, 3), &[n_in, 2]);
    let y = f32_view(&ctx, &[0.0; 64], &[n_out, 2], 0);
    let q8 = MetalTensor::from_bytes(
        &ctx,
        &vec![0u8; n_in / 32 * 34 * n_out],
        vec![n_in as u64, n_out as u64],
        GgmlType::Q8_0,
    )
    .unwrap();
    let misaligned_x = offset_tensor(
        &ctx,
        20,
        bytemuck::cast_slice(&random_f32(n_in * 2, 3)),
        0,
        vec![n_in as u64, 2],
        GgmlType::F32,
    );
    let misaligned_w = offset_tensor(
        &ctx,
        4,
        &bytes,
        0,
        vec![n_in as u64, n_out as u64],
        GgmlType::Q5_K,
    );
    let wide_x = f32_tensor(&ctx, &random_f32(n_in * 9, 3), &[n_in, 9]);
    let wide_y = f32_view(&ctx, &vec![0.0; n_out * 9], &[n_out, 9], 0);
    let short_w = weight(&ctx, &random_q5_k(256, n_out, 1), 256, n_out);
    let x_rows = x.view_bytes(0, vec![n_out as u64, 2]);
    type Case<'a> = (
        &'static str,
        &'a MetalTensor,
        &'a MetalTensor,
        &'a MetalTensor,
        usize,
        usize,
        usize,
    );
    let cases: [Case<'_>; 9] = [
        ("Q5_K", &q8, &x, &y, n_in, n_out, 2),
        ("multiple of 256", &w, &x, &y, 300, n_out, 2),
        ("cols", &w, &x, &y, n_in, n_out, 0),
        ("cols", &w, &wide_x, &wide_y, n_in, n_out, 9),
        ("16-byte aligned", &w, &misaligned_x, &y, n_in, n_out, 2),
        ("16-byte aligned", &misaligned_w, &x, &y, n_in, n_out, 2),
        ("weight must have shape", &short_w, &x, &y, n_in, n_out, 2),
        ("output", &w, &x, &x, n_in, n_out, 2),
        ("aliases input", &w, &x, &x_rows, n_in, n_out, 2),
    ];
    for (needle, w, x, y, n_in, n_out, cols) in cases {
        dispatch_census_begin();
        let error = one_shot(&ctx, |enc| {
            encode_dense_skinny_q5_k(&ctx, enc, w, x, y, n_in, n_out, cols)
        })
        .unwrap_err()
        .to_string();
        assert!(dispatch_census_take().is_empty(), "{needle}: dispatched");
        assert!(error.contains(needle), "expected {needle:?} in {error}");
    }
}

/// Set (any value) to run `dense_skinny_q5_k_decode_probe`, a timing probe
/// of the skinny Q5_K kernel against the per-column kernel; take the Metal
/// lease.
const DENSE_SKINNY_PROBE_ENV: &str = "QWEN_DENSE_SKINNY_PROBE";

/// Opt-in timing at the V4.1 decode Q5_K shapes: for each B, the per-column
/// production kernel (one dispatch, B slots) against the skinny kernel, over
/// rotating weight copies (>= 384 MiB per shape) so reads stream from DRAM,
/// interleaved after warming every copy. Medians of per-dispatch GPU time go
/// to stderr; nothing is asserted beyond completion.
#[test]
fn dense_skinny_q5_k_decode_probe() {
    if std::env::var_os(DENSE_SKINNY_PROBE_ENV).is_none() {
        return;
    }
    let _lease = production_lease();
    let ctx = metal_test_context().expect("Metal device");
    let run = |encode: &dyn Fn(&KernelEncoder)| -> f64 {
        let cmd = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&cmd);
        encode(&enc);
        enc.end();
        cmd.commit();
        wait_completed(&cmd).unwrap();
        (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e6
    };
    for (name, n_in, n_out) in [
        ("shared_gate_up", 5120usize, 2304usize),
        ("shared_down", 2304, 5120),
        ("q_b", 1280, 32768),
        ("output_b", 8192, 5120),
    ] {
        let bytes = n_out * n_in / 256 * 176;
        let copies = (384usize << 20).div_ceil(bytes).max(4);
        let banks: Vec<MetalTensor> = (0..copies)
            .map(|c| weight(&ctx, &random_q5_k(n_in, n_out, c as u64), n_in, n_out))
            .collect();
        let x = f32_tensor(&ctx, &random_f32(8 * n_in, 1), &[n_in, 8]);
        let ids = i32_view(&ctx, &[0; 8]);
        let status = i32_view(&ctx, &[ROUTE_STATUS_READY]);
        let y = f32_tensor(&ctx, &vec![0.0; 8 * n_out], &[n_out, 8]);
        let view = |cols: usize| {
            (
                x.view_bytes(0, vec![n_in as u64, cols as u64]),
                y.view_bytes(0, vec![n_out as u64, cols as u64]),
            )
        };
        let (x8, y8) = view(8);
        let start = std::time::Instant::now();
        let mut it = 0;
        while it < copies || start.elapsed() < std::time::Duration::from_millis(400) {
            let w = &banks[it % copies];
            run(&|enc| encode_dense_skinny_q5_k(&ctx, enc, w, &x8, &y8, n_in, n_out, 8).unwrap());
            it += 1;
        }
        let per = copies.min(8);
        let med = |mut v: Vec<f64>| {
            v.sort_by(|p, q| p.partial_cmp(q).unwrap());
            v[v.len() / 2]
        };
        for cols in [1usize, 2, 3, 4, 8] {
            let (xc, yc) = view(cols);
            let (x1, y1) = view(1);
            let ids = ids.view_subrange(0, vec![cols as u64]);
            // 0: per-column kernel on `cols`, 1: skinny on `cols`, 2: one row.
            let mut times = [Vec::new(), Vec::new(), Vec::new()];
            for rep in 0..10 {
                for (arm, samples) in times.iter_mut().enumerate() {
                    let base = (rep * 3 + arm) * per;
                    samples.push(
                        run(&|enc| {
                            for j in 0..per {
                                let w = &banks[(base + j) % copies];
                                match arm {
                                    0 => encode_per_column(
                                        &ctx, enc, w, &xc, &ids, &status, &yc, n_in, n_out, cols,
                                    ),
                                    1 => encode_dense_skinny_q5_k(
                                        &ctx, enc, w, &xc, &yc, n_in, n_out, cols,
                                    ),
                                    _ => encode_per_column(
                                        &ctx,
                                        enc,
                                        w,
                                        &x1,
                                        &ids.view_subrange(0, vec![1]),
                                        &status,
                                        &y1,
                                        n_in,
                                        n_out,
                                        1,
                                    ),
                                }
                                .unwrap();
                            }
                        }) / per as f64,
                    );
                }
            }
            let [per_col, skinny, one] = times.map(med);
            eprintln!(
                "[dense-skinny] {name} n_in={n_in} n_out={n_out} B={cols} (us): per-column {per_col:.1}, \
                 skinny {skinny:.1}, one-row {one:.1} | skinny / one-row {:.2}x, \
                 skinny / per-column {:.2}x, skinny {:.0} GB/s",
                skinny / one,
                skinny / per_col,
                bytes as f64 / skinny / 1e3,
            );
        }
    }
}
