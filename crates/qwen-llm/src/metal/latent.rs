//! Shared-latent (MLA) attention over a linear F16 cache whose rows are both
//! key and value, as in GLM-5.3-Flash's absorbed NoPE MLA.
//!
//! For each of `rows` consecutive query rows at absolute positions
//! `start + r`, every head attends causally to cache rows `0..=start + r`:
//! `o[h] = softmax_j(q[h] . c_j * scale) . c`, accumulated in F32. Sinks enter
//! the softmax as one extra logit per head; pass [`LATENT_NO_SINK`] for none
//! (a large finite value: the metallib builds with fast math). Reuses DS4's
//! grouped online kernel (64 heads x 512, eight heads share staged rows) with
//! the ring window widened to the cache capacity and no compressed rows.

use super::checks::{bad_shape, check_alignment, check_disjoint, check_tensor, require_serial};
use super::*;

/// Sink logit that contributes negligible softmax mass whenever some visible
/// row scores far above it (always true for real scores, which are O(10));
/// finite because the metallib builds with fast math.
pub const LATENT_NO_SINK: f32 = -1.0e30;
pub const LATENT_HEADS: usize = 64;
pub const LATENT_WIDTH: usize = 512;

/// `queries` and `output` are F32 `[512, 64, rows]`; `cache` is F16 `[512,
/// capacity]` with rows `0..start + rows` written; `sinks` is F32 `[64]`.
#[allow(clippy::too_many_arguments)]
pub fn encode_latent_attention(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    queries: &MetalTensor,
    cache: &MetalTensor,
    sinks: &MetalTensor,
    output: &MetalTensor,
    start: usize,
    rows: usize,
    scale: f32,
) -> Result<(), MetalError> {
    const K: &str = "latent_attention";
    require_serial(K, enc)?;
    if rows == 0 || !scale.is_finite() || scale <= 0.0 {
        return Err(bad_shape(
            K,
            "rows must be positive and scale finite and positive",
        ));
    }
    let (heads, width) = (LATENT_HEADS as u64, LATENT_WIDTH as u64);
    let shape = [width, heads, rows as u64];
    check_tensor(K, queries, GgmlType::F32, &shape, false, "queries")?;
    check_tensor(K, output, GgmlType::F32, &shape, true, "output")?;
    check_alignment(K, queries, 16, "queries")?;
    check_alignment(K, output, 16, "output")?;
    if cache.dtype != GgmlType::F16 || cache.shape.len() != 2 || cache.shape[0] != width {
        return Err(bad_shape(
            K,
            format!(
                "cache must be F16 [512, capacity], got {:?} {:?}",
                cache.dtype, cache.shape
            ),
        ));
    }
    let capacity = cache.shape[1];
    check_tensor(K, cache, GgmlType::F16, &[width, capacity], false, "cache")?;
    check_alignment(K, cache, 16, "cache")?;
    let end = start
        .checked_add(rows)
        .filter(|&end| end as u64 <= capacity && u32::try_from(end).is_ok())
        .ok_or_else(|| {
            bad_shape(
                K,
                format!("positions {start}..{start}+{rows} exceed capacity {capacity}"),
            )
        })?;
    let _ = end;
    check_tensor(K, sinks, GgmlType::F32, &[heads], false, "sinks")?;
    check_disjoint(
        K,
        output,
        &[(queries, "queries"), (cache, "cache"), (sinks, "sinks")],
    )?;
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        head_count: u32,
        head_dim: u32,
        n_tokens: u32,
        compression_ratio: u32,
        start_position: u32,
        window: u32,
        raw_cache_is_chunk: u32,
        scale: f32,
    }
    let pso = ctx.pipeline("kernel_deepseek_v4_grouped_online_dense_sink_attention_f16")?;
    if pso.threadExecutionWidth() != 32 || pso.maxTotalThreadsPerThreadgroup() < 256 {
        return Err(bad_shape(K, "needs 32-lane simdgroups and 256 threads"));
    }
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            head_count: heads as u32,
            head_dim: width as u32,
            n_tokens: rows as u32,
            compression_ratio: 0,
            start_position: start as u32,
            // A window covering the whole cache makes ring addressing linear.
            window: u32::try_from(capacity).map_err(|_| bad_shape(K, "capacity exceeds u32"))?,
            raw_cache_is_chunk: 0,
            scale,
        },
    );
    enc.set_tensor(1, queries);
    enc.set_tensor(2, cache);
    enc.set_tensor(3, cache);
    enc.set_tensor(4, cache);
    enc.set_tensor(5, sinks);
    enc.set_tensor(6, output);
    // 16 staged rows of 512 halves.
    enc.set_threadgroup_memory(0, 16 * LATENT_WIDTH * 2);
    enc.dispatch(
        MTLSize {
            width: rows,
            height: LATENT_HEADS / 8,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Group-axis Q8_0 mat-mat over `rows` tokens with F32 activations and F32
/// simdgroup accumulation: for each token and group `g`,
/// `output[g * n_out..][..n_out] = W_g * input[g * n_in..][..n_in]`, with
/// weights `[n_in, n_out, groups]` (one contiguous matrix per group) and
/// token-major activations `[n_in * groups, rows]` / `[n_out * groups, rows]`.
/// Requires `n_in % 64 == 0`, `n_out % 16 == 0` and `rows % 128 == 0`.
#[allow(clippy::too_many_arguments)]
pub fn encode_mat_mat_q8_0_grouped_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    input: &MetalTensor,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    groups: usize,
    rows: usize,
) -> Result<(), MetalError> {
    const K: &str = "mat_mat_q8_0_grouped";
    require_serial(K, enc)?;
    if weight.dtype != GgmlType::Q8_0
        || groups == 0
        || rows == 0
        || !n_in.is_multiple_of(64)
        || !n_out.is_multiple_of(16)
        || !rows.is_multiple_of(128)
    {
        return Err(bad_shape(
            K,
            format!(
                "needs Q8_0, n_in % 64, n_out % 16 and rows % 128; got {:?} {n_in} {n_out} {rows}",
                weight.dtype
            ),
        ));
    }
    let overflow = || bad_shape(K, "dimension product overflows");
    let in_width = n_in.checked_mul(groups).ok_or_else(overflow)?;
    let out_width = n_out.checked_mul(groups).ok_or_else(overflow)?;
    // `groups` contiguous [n_in, n_out] Q8_0 matrices, viewed per group
    // ([n_in, n_out, groups]) or flat ([n_in, n_out * groups]), inside the
    // weight's buffer. The kernel reads each block's half scale through a
    // typed pointer, so the base must be 2-byte aligned.
    let (block, block_bytes) = GgmlType::Q8_0.storage_layout().ok_or_else(overflow)?;
    let shapes = [
        [n_in as u64, n_out as u64, groups as u64].to_vec(),
        [n_in as u64, out_width as u64].to_vec(),
    ];
    if !shapes.contains(&weight.shape) {
        return Err(bad_shape(
            K,
            format!(
                "weight must be {:?} or {:?}, got {:?}",
                shapes[0], shapes[1], weight.shape
            ),
        ));
    }
    let bytes = (n_in as u64 / block)
        .checked_mul(block_bytes)
        .and_then(|row| row.checked_mul(out_width as u64))
        .ok_or_else(overflow)?;
    let end = weight.offset.checked_add(bytes);
    if weight.n_bytes() != bytes || end.is_none_or(|end| end > weight.buffer.length() as u64) {
        return Err(bad_shape(
            K,
            format!("weight is not {groups} contiguous Q8_0 matrices inside its buffer"),
        ));
    }
    check_alignment(K, weight, 2, "weight")?;
    check_tensor(
        K,
        input,
        GgmlType::F32,
        &[in_width as u64, rows as u64],
        false,
        "input",
    )?;
    check_tensor(
        K,
        output,
        GgmlType::F32,
        &[out_width as u64, rows as u64],
        true,
        "output",
    )?;
    check_disjoint(K, output, &[(input, "input"), (weight, "weight")])?;
    let pso = ctx.pipeline("kernel_mat_mat_q8_0_f32_r2c16k64_grouped")?;
    if pso.threadExecutionWidth() != 32 || pso.maxTotalThreadsPerThreadgroup() < 128 {
        return Err(bad_shape(K, "needs four 32-lane simdgroups"));
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        m: u32,
        n: u32,
        k: u32,
        groups: u32,
        nb01: u32,
        stride_b: u32,
        stride_c: u32,
    }
    let u = |v: usize, name: &str| super::checks::to_u32(K, v, name);
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            m: u(n_out, "n_out")?,
            n: u(rows, "rows")?,
            k: u(n_in, "n_in")?,
            groups: u(groups, "groups")?,
            nb01: u(n_in / 32 * 34, "row bytes")?,
            stride_b: u(in_width, "input width")?,
            stride_c: u(out_width, "output width")?,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, input);
    enc.set_tensor(3, output);
    enc.set_threadgroup_memory(0, 4_096);
    enc.dispatch(
        MTLSize {
            width: rows / 128,
            height: n_out / 16,
            depth: groups,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{offset_tensor, synthetic_q8_0_bank, tensor_f32_at_offset};
    use super::*;

    const H: usize = LATENT_HEADS;
    const W: usize = LATENT_WIDTH;

    fn f32_tensor(ctx: &MetalContext, values: &[f32], shape: Vec<u64>) -> MetalTensor {
        offset_tensor(
            ctx,
            32,
            bytemuck::cast_slice(values),
            32,
            shape,
            GgmlType::F32,
        )
    }

    fn run(ctx: &MetalContext, encode: impl FnOnce(&KernelEncoder)) {
        let command = ctx.queue.commandBuffer().expect("command buffer");
        let encoder = KernelEncoder::begin(&command);
        encode(&encoder);
        encoder.end();
        command.commit();
        wait_completed(&command).expect("command buffer failed");
    }

    fn reference(
        q: &[f32],
        cache: &[half::f16],
        start: usize,
        rows: usize,
        scale: f32,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; W * H * rows];
        for r in 0..rows {
            let visible = start + r + 1;
            for h in 0..H {
                let qh = &q[(r * H + h) * W..][..W];
                let scores: Vec<f64> = (0..visible)
                    .map(|j| {
                        qh.iter()
                            .zip(&cache[j * W..(j + 1) * W])
                            .map(|(q, c)| *q as f64 * c.to_f64())
                            .sum::<f64>()
                            * scale as f64
                    })
                    .collect();
                let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                let total: f64 = weights.iter().sum();
                for d in 0..W {
                    let o: f64 = (0..visible)
                        .map(|j| weights[j] * cache[j * W + d].to_f64())
                        .sum();
                    out[(r * H + h) * W + d] = (o / total) as f32;
                }
            }
        }
        out
    }

    /// Visible lengths covering staging tails (16-row tiles) and the dense
    /// frontier, distinct heads, poisoned rows past the visible range, and a
    /// packed causal block.
    #[test]
    fn latent_attention_matches_reference_without_sinks() {
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        const CAPACITY: usize = 2064;
        let scale = 1.0 / 16.0;
        let mut cache: Vec<half::f16> = (0..CAPACITY * W)
            .map(|i| half::f16::from_f32((((i * 37 + i / 513) % 199) as f32 - 99.0) * 0.011))
            .collect();
        let sinks = f32_tensor(&ctx, &[LATENT_NO_SINK; H], vec![H as u64]);
        for (start, rows) in [
            (0, 1),
            (14, 1),
            (15, 1),
            (16, 1),
            (99, 1),
            (2050, 1),
            (12, 3),
        ] {
            // Rows past the last visible position are NaN; reading them would poison.
            let visible_end = start + rows;
            for value in &mut cache[visible_end * W..] {
                *value = half::f16::NAN;
            }
            let q: Vec<f32> = (0..W * H * rows)
                .map(|i| (((i * 29 + (i / W) * 131) % 97) as f32 - 48.0) * 0.02)
                .collect();
            let cache_t = offset_tensor(
                &ctx,
                64,
                bytemuck::cast_slice(&cache),
                64,
                vec![W as u64, CAPACITY as u64],
                GgmlType::F16,
            );
            let q_t = f32_tensor(&ctx, &q, vec![W as u64, H as u64, rows as u64]);
            let out_t = f32_tensor(
                &ctx,
                &vec![0.0; W * H * rows],
                vec![W as u64, H as u64, rows as u64],
            );
            run(&ctx, |enc| {
                encode_latent_attention(
                    &ctx, enc, &q_t, &cache_t, &sinks, &out_t, start, rows, scale,
                )
                .unwrap();
            });
            let expected = reference(&q, &cache, start, rows, scale);
            let actual = tensor_f32_at_offset(&out_t);
            let worst = actual
                .iter()
                .zip(&expected)
                .map(|(a, e)| (a - e).abs())
                .fold(0.0f32, f32::max);
            assert!(
                actual.iter().all(|v| v.is_finite()),
                "start {start}: non-finite"
            );
            assert!(worst <= 2e-4, "start {start} rows {rows}: max abs {worst}");
            // Restore poisoned rows with finite data for the next case.
            for (i, value) in cache.iter_mut().enumerate().skip(visible_end * W) {
                *value = half::f16::from_f32((((i * 37 + i / 513) % 199) as f32 - 99.0) * 0.011);
            }
        }
        // One visible row returns that row (rounded to F16) for every head.
        let one = reference(&vec![1.0; W * H], &cache, 0, 1, scale);
        assert!(one.chunks(W).all(|h| {
            h.iter()
                .zip(&cache[..W])
                .all(|(o, c)| (o - c.to_f32()).abs() < 1e-6)
        }));
    }

    /// GLM-5.3 absorption shapes through the existing grouped Q8_0 GEMV:
    /// `attn_k_b` (256 -> 512 per head) and `attn_v_b` (512 -> 256 per head).
    #[test]
    fn grouped_q8_absorption_matches_reference_at_glm_shapes() {
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        for (n_in, n_out) in [(256usize, 512usize), (512, 256)] {
            let (bytes, decoded) = synthetic_q8_0_bank(n_in, n_out * H);
            let weight = offset_tensor(
                &ctx,
                256,
                &bytes,
                64,
                vec![n_in as u64, n_out as u64, H as u64],
                GgmlType::Q8_0,
            );
            let x: Vec<f32> = (0..n_in * H)
                .map(|i| (((i * 17 + (i / n_in) * 53) % 89) as f32 - 44.0) * 0.03)
                .collect();
            let x_t = f32_tensor(&ctx, &x, vec![n_in as u64, H as u64]);
            let y_t = f32_tensor(&ctx, &vec![0.0; n_out * H], vec![n_out as u64, H as u64]);
            run(&ctx, |enc| {
                encode_mat_vec_q8_0_grouped_f32(&ctx, enc, &weight, &x_t, &y_t, n_in, n_out, H)
                    .unwrap();
            });
            let actual = tensor_f32_at_offset(&y_t);
            for h in 0..H {
                for r in 0..n_out {
                    let row = &decoded[(h * n_out + r) * n_in..][..n_in];
                    let expected: f64 = row
                        .iter()
                        .zip(&x[h * n_in..(h + 1) * n_in])
                        .map(|(w, v)| *w as f64 * *v as f64)
                        .sum();
                    let got = actual[h * n_out + r] as f64;
                    assert!(
                        (got - expected).abs() <= 1e-4 * (1.0 + expected.abs()),
                        "{n_in}->{n_out} head {h} row {r}: {got} vs {expected}"
                    );
                }
            }
        }
    }

    /// Grouped Q8_0 mat-mat over 128 rows equals the per-row grouped GEMV at
    /// both GLM absorption shapes (F32 accumulation, different order).
    #[test]
    fn grouped_q8_mat_mat_matches_per_row_gemv_at_glm_shapes() {
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        const ROWS: usize = 128;
        for (n_in, n_out) in [(256usize, 512usize), (512, 256)] {
            let (bytes, _) = synthetic_q8_0_bank(n_in, n_out * H);
            let weight = offset_tensor(
                &ctx,
                256,
                &bytes,
                64,
                vec![n_in as u64, n_out as u64, H as u64],
                GgmlType::Q8_0,
            );
            let x: Vec<f32> = (0..n_in * H * ROWS)
                .map(|i| (((i * 17 + (i / n_in) * 53) % 89) as f32 - 44.0) * 0.03)
                .collect();
            let x_t = f32_tensor(&ctx, &x, vec![(n_in * H) as u64, ROWS as u64]);
            let y_t = f32_tensor(
                &ctx,
                &vec![0.0; n_out * H * ROWS],
                vec![(n_out * H) as u64, ROWS as u64],
            );
            run(&ctx, |enc| {
                encode_mat_mat_q8_0_grouped_f32(
                    &ctx, enc, &weight, &x_t, &y_t, n_in, n_out, H, ROWS,
                )
                .unwrap();
            });
            let grouped = tensor_f32_at_offset(&y_t);
            for row in [0usize, 1, 63, 127] {
                let xr = x_t.view_subrange((row * n_in * H) as u64, vec![(n_in * H) as u64]);
                let yr = f32_tensor(&ctx, &vec![0.0; n_out * H], vec![(n_out * H) as u64]);
                run(&ctx, |enc| {
                    encode_mat_vec_q8_0_grouped_f32(&ctx, enc, &weight, &xr, &yr, n_in, n_out, H)
                        .unwrap();
                });
                let gemv = tensor_f32_at_offset(&yr);
                let got = &grouped[row * n_out * H..(row + 1) * n_out * H];
                let scale = gemv.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let worst = got
                    .iter()
                    .zip(&gemv)
                    .map(|(a, e)| (a - e).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    worst <= 1e-5 * scale.max(1.0),
                    "{n_in}->{n_out} row {row}: {worst} (scale {scale})"
                );
            }
        }
    }

    /// The grouped Q8_0 weight contract: per-group or flat (DS4) views of
    /// the same bytes give identical results; any other shape, a view past
    /// its buffer, a misaligned base and overflowing dimensions are refused.
    #[test]
    fn grouped_q8_mat_mat_refuses_invalid_weight_views() {
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        const ROWS: usize = 128;
        let (n_in, n_out) = (256usize, 512usize);
        let (bytes, _) = synthetic_q8_0_bank(n_in, n_out * H);
        let view = |prefix: usize, suffix: usize, shape: Vec<u64>| {
            offset_tensor(&ctx, prefix, &bytes, suffix, shape, GgmlType::Q8_0)
        };
        let grouped = view(256, 64, vec![n_in as u64, n_out as u64, H as u64]);
        let flat = view(256, 64, vec![n_in as u64, (n_out * H) as u64]);
        let x: Vec<f32> = (0..n_in * H * ROWS)
            .map(|i| ((i * 29 % 97) as f32 - 48.0) * 0.02)
            .collect();
        let x_t = f32_tensor(&ctx, &x, vec![(n_in * H) as u64, ROWS as u64]);
        let output = || {
            f32_tensor(
                &ctx,
                &vec![0.0; n_out * H * ROWS],
                vec![(n_out * H) as u64, ROWS as u64],
            )
        };
        let (y_grouped, y_flat) = (output(), output());
        run(&ctx, |enc| {
            for (weight, y) in [(&grouped, &y_grouped), (&flat, &y_flat)] {
                encode_mat_mat_q8_0_grouped_f32(&ctx, enc, weight, &x_t, y, n_in, n_out, H, ROWS)
                    .unwrap();
            }
        });
        assert!(tensor_f32_at_offset(&y_grouped) == tensor_f32_at_offset(&y_flat));

        let y = output();
        let command = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&command);
        let refuse = |weight: &MetalTensor, groups: usize, needle: &str| {
            let err = encode_mat_mat_q8_0_grouped_f32(
                &ctx, &enc, weight, &x_t, &y, n_in, n_out, groups, ROWS,
            )
            .unwrap_err()
            .to_string();
            assert!(err.contains(needle), "{err}");
        };
        refuse(
            &view(256, 64, vec![(n_in * H) as u64, n_out as u64]),
            H,
            "weight must be",
        );
        let past_end = MetalTensor {
            offset: grouped.offset + 128,
            ..grouped.clone()
        };
        refuse(&past_end, H, "inside its buffer");
        refuse(
            &view(257, 64, vec![n_in as u64, n_out as u64, H as u64]),
            H,
            "not 2-byte aligned",
        );
        refuse(&grouped, usize::MAX / 2, "overflow");
        enc.end();
    }
}
