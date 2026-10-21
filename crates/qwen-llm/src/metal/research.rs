//! Research-only probes: roofline and touch-bytes kernels (bench-only).

use super::*;
use objc2_metal::{MTLBuffer, MTLResource};

pub fn encode_touch_bytes_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    src: &MetalTensor,
    sink: &MetalTensor,
    stride_bytes: usize,
) -> Result<(), MetalError> {
    let sink_n = sink.n_elements() as usize;
    if sink_n == 0 {
        return Err(MetalError::BadShape {
            kernel: "touch_bytes",
            detail: "sink must have at least one element".into(),
        });
    }
    let n_bytes = src.n_bytes() as usize;
    if n_bytes == 0 || stride_bytes == 0 {
        return Err(MetalError::BadShape {
            kernel: "touch_bytes",
            detail: format!("n_bytes={n_bytes} stride_bytes={stride_bytes} must both be > 0"),
        });
    }
    let n_steps = n_bytes.div_ceil(stride_bytes);
    let pso = ctx.pipeline("kernel_touch_bytes_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_steps: u32,
        stride_bytes: u32,
        n_bytes: u64,
    }
    enc.set_bytes(
        0,
        &Args {
            n_steps: n_steps as u32,
            stride_bytes: stride_bytes as u32,
            n_bytes: n_bytes as u64,
        },
    );
    enc.set_tensor(1, src);
    enc.set_tensor(2, sink);
    enc.dispatch(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: sink_n.min(256),
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub fn encode_roofline_stream_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    y: &MetalTensor,
    alpha: f32,
) -> Result<(), MetalError> {
    let n = x.n_elements() as usize;
    if y.n_elements() as usize != n {
        return Err(MetalError::BadShape {
            kernel: "roofline_stream",
            detail: format!("y.n_elements={} != x.n_elements={n}", y.n_elements()),
        });
    }
    let pso = ctx.pipeline("kernel_roofline_stream_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n: u32,
        alpha: f32,
    }
    enc.set_bytes(0, &Args { n: n as u32, alpha });
    enc.set_tensor(1, x);
    enc.set_tensor(2, y);
    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    enc.dispatch(
        MTLSize {
            width: n.div_ceil(tg_threads),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg_threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub fn encode_roofline_fma_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    y: &MetalTensor,
    iters: usize,
) -> Result<(), MetalError> {
    let n = x.n_elements() as usize;
    if y.n_elements() as usize != n || iters == 0 {
        return Err(MetalError::BadShape {
            kernel: "roofline_fma",
            detail: format!(
                "y.n_elements={} x.n_elements={n} iters={iters}",
                y.n_elements()
            ),
        });
    }
    let pso = ctx.pipeline("kernel_roofline_fma_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n: u32,
        iters: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n: n as u32,
            iters: iters as u32,
        },
    );
    enc.set_tensor(1, x);
    enc.set_tensor(2, y);
    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    enc.dispatch(
        MTLSize {
            width: n.div_ceil(tg_threads),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg_threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Fat Q8_0 decode GEMV: `R` output rows per simdgroup, `SG` simdgroups per
/// threadgroup. `weight` is `[n_in, n_out]` Q8_0; `x` is F32 `[n_in]`; `y` is
/// F32 `[n_out]`. This is a research probe until end-to-end evidence promotes
/// it. The supported `(R, SG)` pairs match the kernel instantiations.
#[allow(clippy::too_many_arguments)]
pub fn encode_fat_gemv_q8_0_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    rows_per_simdgroup: usize,
    simdgroups: usize,
) -> Result<(), MetalError> {
    const K: &str = "fat_gemv_q8_0";
    let bad = |detail: String| MetalError::BadShape { kernel: K, detail };
    if !matches!(rows_per_simdgroup, 2 | 4 | 8) || !matches!(simdgroups, 1 | 2 | 4) {
        return Err(bad(format!(
            "unsupported geometry R={rows_per_simdgroup} SG={simdgroups}; R in 2/4/8, SG in 1/2/4"
        )));
    }
    if n_in == 0 || !n_in.is_multiple_of(32) || n_out == 0 {
        return Err(bad(format!(
            "n_in={n_in} must be a positive multiple of 32 and n_out={n_out} positive"
        )));
    }
    let expected_weight = n_in
        .checked_mul(n_out)
        .ok_or_else(|| bad("weight element count overflow".into()))?;
    if weight.dtype != GgmlType::Q8_0
        || weight.n_elements() as usize != expected_weight
        || x.dtype != GgmlType::F32
        || x.n_elements() as usize != n_in
        || y.dtype != GgmlType::F32
        || y.n_elements() as usize != n_out
    {
        return Err(bad(format!(
            "expected Q8_0 [{n_in}, {n_out}] and F32 x[{n_in}] -> y[{n_out}], got {:?} x={} y={}",
            weight.dtype,
            x.n_elements(),
            y.n_elements()
        )));
    }
    let name = format!("kernel_fat_gemv_q8_0_f32_r{rows_per_simdgroup}_sg{simdgroups}");
    let pso = ctx.pipeline(&name)?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &[n_in as u32, n_out as u32]);
    enc.set_tensor(1, weight);
    enc.set_tensor(2, x);
    enc.set_tensor(3, y);
    let rows_per_tg = rows_per_simdgroup * simdgroups;
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(rows_per_tg),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: simdgroups * 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Read back a shared F32 tensor. Research/bench use only.
pub fn read_f32_tensor(t: &MetalTensor) -> Result<Vec<f32>, MetalError> {
    let n = t.n_elements() as usize;
    let bytes = (n as u64)
        .checked_mul(4)
        .ok_or_else(|| MetalError::BadShape {
            kernel: "read_f32_tensor",
            detail: "byte extent overflow".into(),
        })?;
    let end = t
        .offset
        .checked_add(bytes)
        .ok_or_else(|| MetalError::BadShape {
            kernel: "read_f32_tensor",
            detail: "offset overflow".into(),
        })?;
    if t.dtype != GgmlType::F32
        || t.offset % 4 != 0
        || end > t.buffer.length() as u64
        || t.buffer.storageMode() != objc2_metal::MTLStorageMode::Shared
    {
        return Err(MetalError::BadShape {
            kernel: "read_f32_tensor",
            detail: "expected a shared F32 tensor range".into(),
        });
    }
    let ptr = t
        .buffer
        .contents()
        .as_ptr()
        .cast::<f32>()
        .wrapping_add((t.offset / 4) as usize);
    Ok(unsafe { std::slice::from_raw_parts(ptr, n).to_vec() })
}

#[cfg(test)]
mod fat_gemv_tests {
    use super::*;

    fn q8_0_weights(n_in: usize, n_out: usize) -> Vec<u8> {
        let nb = n_in / 32;
        let mut bytes = vec![0u8; nb * 34 * n_out];
        let mut state = 0x1234_5678u32;
        for row in 0..n_out {
            for b in 0..nb {
                let off = (row * nb + b) * 34;
                bytes[off] = 0x00;
                bytes[off + 1] = 0x3C; // half 1.0
                for i in 0..32 {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    bytes[off + 2 + i] = ((state >> 24) as i8).wrapping_rem(7) as u8;
                }
            }
        }
        bytes
    }

    #[test]
    fn fat_gemv_q8_0_matches_tuned_for_all_geometries() {
        let ctx = crate::test_fixtures::metal_context_or_skip().unwrap();
        let (n_in, n_out) = (256usize, 24usize);
        let weights = q8_0_weights(n_in, n_out);
        let xv: Vec<f32> = (0..n_in)
            .map(|i| ((i as f32) * 0.017).sin() * 0.5)
            .collect();
        let w = MetalTensor::from_bytes(
            &ctx,
            &weights,
            vec![n_in as u64, n_out as u64],
            GgmlType::Q8_0,
        )
        .unwrap();
        let x = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&xv),
            vec![n_in as u64],
            GgmlType::F32,
        )
        .unwrap();
        let y_ref = MetalTensor::from_bytes(
            &ctx,
            &vec![0u8; n_out * 4],
            vec![n_out as u64],
            GgmlType::F32,
        )
        .unwrap();
        let cmd = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&cmd);
        encode_mat_vec_q8_0_f32(&ctx, &enc, &w, &x, &y_ref, n_in, n_out).unwrap();
        enc.end();
        cmd.commit();
        crate::metal::wait_completed(&cmd).unwrap();
        let reference = read_f32_tensor(&y_ref).unwrap();

        for rows in [2usize, 4, 8] {
            for simdgroups in [1usize, 2, 4] {
                let y = MetalTensor::from_bytes(
                    &ctx,
                    &vec![0u8; n_out * 4],
                    vec![n_out as u64],
                    GgmlType::F32,
                )
                .unwrap();
                let cmd = ctx.queue.commandBuffer().unwrap();
                let enc = KernelEncoder::begin(&cmd);
                encode_fat_gemv_q8_0_f32(&ctx, &enc, &w, &x, &y, n_in, n_out, rows, simdgroups)
                    .unwrap();
                enc.end();
                cmd.commit();
                crate::metal::wait_completed(&cmd).unwrap();
                let actual = read_f32_tensor(&y).unwrap();
                let max_abs = reference
                    .iter()
                    .zip(&actual)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    max_abs < 2.0e-4,
                    "R={rows} SG={simdgroups} max_abs={max_abs:e}"
                );
            }
        }
    }
}
