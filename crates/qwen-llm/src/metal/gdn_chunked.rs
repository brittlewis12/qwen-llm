//! Tensor-tier prompt recurrence; scratch borrows the dead QKV projection.
use super::*;

/// F32 chunked GDN prompt recurrence for the qualified Hk=16, Hv=32/48, D=128 shapes at >=256 tokens on the Metal 4 tensor tier, enabled by default. Set to 0 for A/B or rollback. Decode, checkpointed verification, other head shapes and devices retain the packed serial recurrence. Reuses dead QKV projection scratch without increasing the admission budget.
const CHUNKED_PREFILL_ENV: &str = "QWEN_GDN_CHUNKED_PREFILL";

fn enabled() -> bool {
    #[cfg(test)]
    if let Some(value) = NUMERIC_CONTROL.with(|s| s.get()) {
        return value;
    }
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| crate::env_flag::read_default_on(CHUNKED_PREFILL_ENV))
}

#[cfg(test)]
thread_local! {
    static NUMERIC_CONTROL: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Test-only F32 arithmetic control, without changing the normal chunk policy.
#[cfg(test)]
pub(crate) fn with_gdn_prefill_numeric_control<R>(enabled: bool, work: impl FnOnce() -> R) -> R {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            NUMERIC_CONTROL.with(|s| s.set(self.0));
        }
    }
    let _restore = Restore(NUMERIC_CONTROL.with(|s| s.replace(Some(enabled))));
    work()
}

fn chunk_size(
    on: bool,
    tensor: bool,
    tokens: usize,
    hv: usize,
    hk: usize,
    dim: usize,
) -> Option<usize> {
    if !on || !tensor || tokens < 256 || hk != 16 || !matches!(hv, 32 | 48) || dim != 128 {
        return None;
    }
    Some(if hv == 48 && tokens >= 2048 { 32 } else { 64 })
}

/// Ordinary prefill only. `dead_qkv` is no longer consumed after conv/prep and
/// must be disjoint from all recurrence inputs, outputs and persistent state.
/// Full internal chunks use TensorOps; a short suffix carries their final F32
/// state through the existing serial recurrence in the same command buffer.
pub fn encode_gdn_step_decay_packed_prefill_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q: &MetalTensor,
    k: &MetalTensor,
    v: &MetalTensor,
    decay: &MetalTensor,
    beta: &MetalTensor,
    state: &MetalTensor,
    out: &MetalTensor,
    dead_qkv: &MetalTensor,
    tokens: usize,
    hv: usize,
    hk: usize,
    dim: usize,
) -> Result<(), MetalError> {
    let Some(c) = chunk_size(
        enabled(),
        ctx.tensor_ops_status().is_available(),
        tokens,
        hv,
        hk,
        dim,
    ) else {
        return encode_gdn_step_decay_packed_f32(
            ctx, enc, q, k, v, decay, beta, state, out, tokens, hv, hk, dim,
        );
    };
    encode_chunked(
        ctx, enc, q, k, v, decay, beta, state, out, dead_qkv, tokens, hv, hk, c,
    )
}

fn encode_chunked(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q: &MetalTensor,
    k: &MetalTensor,
    v: &MetalTensor,
    decay: &MetalTensor,
    beta: &MetalTensor,
    state: &MetalTensor,
    out: &MetalTensor,
    scratch: &MetalTensor,
    tokens: usize,
    hv: usize,
    hk: usize,
    c: usize,
) -> Result<(), MetalError> {
    const KERNEL: &str = "gdn_chunked_prefill";
    let bad = |detail: String| MetalError::BadShape {
        kernel: KERNEL,
        detail,
    };
    // Device indexing is uint throughout both kernels. Refuse overflow before
    // making views or encoding either dispatch.
    let qn = tokens
        .checked_mul(hk * 128)
        .ok_or_else(|| bad("Q/K size overflow".into()))?;
    let vn = tokens
        .checked_mul(hv * 128)
        .ok_or_else(|| bad("V size overflow".into()))?;
    let hn = tokens
        .checked_mul(hv)
        .ok_or_else(|| bad("decay size overflow".into()))?;
    if u32::try_from(vn).is_err() || u32::try_from(qn).is_err() {
        return Err(bad("device element index exceeds u32".into()));
    }
    let tensors = [
        (q, qn),
        (k, qn),
        (v, vn),
        (decay, hn),
        (beta, hn),
        (state, hv * 128 * 128),
        (out, vn),
    ];
    for (t, want) in tensors {
        if t.dtype != GgmlType::F32 || t.n_elements() as usize != want {
            return Err(bad(format!("expected F32 tensor with {want} elements")));
        }
        if super::mat_vec::tensor_ranges_overlap(scratch, t) {
            return Err(bad("QKV scratch overlaps live recurrence storage".into()));
        }
    }
    if enc.is_concurrent() {
        return Err(bad("chunked recurrence requires a serial encoder".into()));
    }
    let prefix = tokens / c * c;
    let matrix = prefix * hv * c;
    let scale = prefix * hv * 2;
    let needed = matrix * 2 + scale;
    if scratch.dtype != GgmlType::F32 || scratch.n_elements() < needed as u64 {
        return Err(bad(format!("dead QKV scratch needs {needed} F32 elements")));
    }
    enc.note_write(scratch);
    enc.note_write(state);
    enc.note_write(out);
    // Resolve both pipelines before appending either dispatch.
    let (pre_name, scan_name) = if c == 32 {
        (
            "kernel_gdn_chunk_precompute32_f32",
            "kernel_gdn_chunk_scan32_f32",
        )
    } else {
        (
            "kernel_gdn_chunk_precompute64_f32",
            "kernel_gdn_chunk_scan64_f32",
        )
    };
    let pre = ctx.tensor_pipeline(pre_name)?;
    let scan = ctx.tensor_pipeline(scan_name)?;
    let p = scratch.view_subrange(0, vec![matrix as u64]);
    let inv = scratch.view_subrange(matrix as u64, vec![matrix as u64]);
    let scales = scratch.view_subrange((matrix * 2) as u64, vec![scale as u64]);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        tokens: u32,
        hv: u32,
        hk: u32,
        unused: u32,
    }
    enc.set_pipeline(&pre);
    enc.set_bytes(
        0,
        &Args {
            tokens: prefix as u32,
            hv: hv as u32,
            hk: hk as u32,
            unused: 0,
        },
    );
    enc.set_tensor(1, q);
    enc.set_tensor(2, k);
    enc.set_tensor(3, decay);
    enc.set_tensor(4, beta);
    enc.set_tensor(5, &p);
    enc.set_tensor(6, &inv);
    enc.set_tensor(7, &scales);
    enc.dispatch(
        MTLSize {
            width: prefix / c,
            height: hk,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    enc.set_pipeline(&scan);
    enc.set_tensor(3, v);
    enc.set_tensor(8, state);
    enc.set_tensor(9, out);
    enc.dispatch(
        MTLSize {
            width: hv,
            height: 4,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    if prefix != tokens {
        let tail = |t: &MetalTensor, width: usize| {
            t.view_subrange(
                (prefix * width) as u64,
                vec![((tokens - prefix) * width) as u64],
            )
        };
        encode_gdn_step_decay_packed_f32(
            ctx,
            enc,
            &tail(q, hk * 128),
            &tail(k, hk * 128),
            &tail(v, hv * 128),
            &tail(decay, hv),
            &tail(beta, hv),
            state,
            &tail(out, hv * 128),
            tokens - prefix,
            hv,
            hk,
            128,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_preserves_serial_devices_decode_and_other_shapes() {
        assert_eq!(chunk_size(true, true, 2048, 48, 16, 128), Some(32));
        assert_eq!(chunk_size(true, true, 2048, 32, 16, 128), Some(64));
        assert_eq!(chunk_size(true, true, 1024, 48, 16, 128), Some(64));
        assert_eq!(chunk_size(true, true, 257, 32, 16, 128), Some(64));
        for (on, tensor, n, hv, hk, d) in [
            (false, true, 2048, 48, 16, 128),
            (true, false, 2048, 48, 16, 128),
            (true, true, 1, 48, 16, 128),
            (true, true, 255, 48, 16, 128),
            (true, true, 2048, 16, 16, 128),
            (true, true, 2048, 48, 8, 128),
            (true, true, 2048, 48, 16, 64),
        ] {
            assert_eq!(chunk_size(on, tensor, n, hv, hk, d), None);
        }
    }

    fn tensor(ctx: &MetalContext, xs: &[f32]) -> MetalTensor {
        MetalTensor::from_bytes(
            ctx,
            bytemuck::cast_slice(xs),
            vec![xs.len() as u64],
            GgmlType::F32,
        )
        .unwrap()
    }

    fn check(label: &str, actual: &[f32], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        let (mut max, mut dot, mut aa, mut bb) = (0.0f64, 0.0, 0.0, 0.0);
        for (&a, &b) in actual.iter().zip(expected) {
            assert!(a.is_finite() && b.is_finite(), "{label} non-finite");
            max = max.max((a as f64 - b).abs());
            dot += a as f64 * b;
            aa += (a as f64).powi(2);
            bb += b * b;
        }
        let cos = dot / (aa * bb).sqrt();
        eprintln!("{label}: max_abs={max:.9e} cosine={cos:.12}");
        assert!(
            max <= 1e-5 && cos >= 0.999999,
            "{label}: max={max} cos={cos}"
        );
    }

    #[test]
    fn f64_output_state_outer_carry_reset_decay_and_serial_tail() {
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        if !ctx.tensor_ops_status().is_available() {
            return;
        }
        let mut seed = 0x9a73000012345678u64;
        let mut rand = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            ((seed >> 40) as i32 - 8388608) as f32 / 8388608.0
        };
        for (hv, n, c, regime) in [
            (32, 321, 64, 0),
            (48, 1057, 64, 0),
            (48, 2081, 32, 1),
            (32, 321, 64, 2),
        ] {
            const HK: usize = 16;
            const D: usize = 128;
            let mut q = vec![0.0f32; n * HK * D];
            let mut k = q.clone();
            for row in 0..n * HK {
                for j in 0..D {
                    q[row * D + j] = rand();
                    k[row * D + j] = if regime == 1 {
                        1.0 + 0.01 * rand()
                    } else {
                        rand()
                    };
                }
                let nq = (1e-6
                    + q[row * D..(row + 1) * D]
                        .iter()
                        .map(|x| (*x as f64).powi(2))
                        .sum::<f64>())
                .sqrt();
                let nk = (1e-6
                    + k[row * D..(row + 1) * D]
                        .iter()
                        .map(|x| (*x as f64).powi(2))
                        .sum::<f64>())
                .sqrt();
                for j in 0..D {
                    q[row * D + j] /= nq as f32;
                    k[row * D + j] /= nk as f32;
                }
            }
            let v: Vec<f32> = (0..n * hv * D)
                .map(|_| rand() * if regime == 1 { 3.0 } else { 1.0 })
                .collect();
            let g: Vec<f32> = (0..n * hv)
                .map(|i| match regime {
                    1 => 0.99999,
                    2 if (i / hv) % 67 == 0 => 0.0,
                    2 => 0.95,
                    _ => (-((2.0 * rand()).exp()) * (3.0 * rand()).exp().ln_1p()).exp(),
                })
                .collect();
            let b: Vec<f32> = (0..n * hv)
                .map(|_| {
                    if regime == 1 {
                        0.999
                    } else {
                        1.0 / (1.0 + (-4.0 * rand()).exp())
                    }
                })
                .collect();
            let initial: Vec<f32> = (0..hv * D * D).map(|_| rand() * 0.1).collect();
            let mut sr: Vec<f64> = initial.iter().map(|x| *x as f64).collect();
            let mut yr = vec![0.0f64; n * hv * D];
            for h in 0..hv {
                for dv in 0..D {
                    let row = &mut sr[(h * D + dv) * D..(h * D + dv + 1) * D];
                    for t in 0..n {
                        let kt = &k[(t * HK + h % HK) * D..(t * HK + h % HK + 1) * D];
                        let qt = &q[(t * HK + h % HK) * D..(t * HK + h % HK + 1) * D];
                        let mut sk = 0.0;
                        for j in 0..D {
                            row[j] *= g[t * hv + h] as f64;
                            sk += row[j] * kt[j] as f64;
                        }
                        let delta = (v[(t * hv + h) * D + dv] as f64 - sk) * b[t * hv + h] as f64;
                        let mut y = 0.0;
                        for j in 0..D {
                            row[j] += delta * kt[j] as f64;
                            y += row[j] * qt[j] as f64;
                        }
                        yr[(t * hv + h) * D + dv] = y;
                    }
                }
            }
            let (q, k, v, g, b) = (
                tensor(&ctx, &q),
                tensor(&ctx, &k),
                tensor(&ctx, &v),
                tensor(&ctx, &g),
                tensor(&ctx, &b),
            );
            let state = tensor(&ctx, &initial);
            let out = MetalTensor::zeros_f32(&ctx, vec![(n * hv * D) as u64]).unwrap();
            // Offset views and sentinel guards catch an inverse slice writing
            // outside scratch (including inactive C=32 simdgroups).
            let scratch_len = n * (2 * HK + hv) * D;
            let backing = tensor(&ctx, &vec![123.5; scratch_len + 64]);
            let scratch = backing.view_subrange(32, vec![scratch_len as u64]);
            // Two outer calls exercise global state carry; both have a serial
            // suffix, and the second starts from a nonzero transformed state.
            let split = if n > 2000 { 2049 } else { 257 };
            one_shot(&ctx, |enc| {
                let view = |t: &MetalTensor, start: usize, count: usize, width: usize| {
                    t.view_subrange((start * width) as u64, vec![(count * width) as u64])
                };
                for (start, count) in [(0, split), (split, n - split)] {
                    if count == 0 {
                        continue;
                    }
                    if count >= c {
                        encode_chunked(
                            &ctx,
                            enc,
                            &view(&q, start, count, HK * D),
                            &view(&k, start, count, HK * D),
                            &view(&v, start, count, hv * D),
                            &view(&g, start, count, hv),
                            &view(&b, start, count, hv),
                            &state,
                            &view(&out, start, count, hv * D),
                            &scratch,
                            count,
                            hv,
                            HK,
                            c,
                        )?;
                    } else {
                        encode_gdn_step_decay_packed_f32(
                            &ctx,
                            enc,
                            &view(&q, start, count, HK * D),
                            &view(&k, start, count, HK * D),
                            &view(&v, start, count, hv * D),
                            &view(&g, start, count, hv),
                            &view(&b, start, count, hv),
                            &state,
                            &view(&out, start, count, hv * D),
                            count,
                            hv,
                            HK,
                            D,
                        )?;
                    }
                }
                Ok(())
            })
            .unwrap();
            check(
                &format!("Hv{hv} C{c} regime{regime} output"),
                &read_back_f32(&out.buffer, n * hv * D),
                &yr,
            );
            check(
                &format!("Hv{hv} C{c} regime{regime} state"),
                &read_back_f32(&state.buffer, hv * D * D),
                &sr,
            );
            let guard = read_back_f32(&backing.buffer, scratch_len + 64);
            assert!(
                guard[..32]
                    .iter()
                    .chain(&guard[scratch_len + 32..])
                    .all(|x| *x == 123.5)
            );
        }
    }
}
