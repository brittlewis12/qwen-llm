//! Exact-copy and charged layer-bank screen; no production transpose selector.

use super::*;
use objc2_metal::MTLCommandBufferStatus;
use serde_json::json;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Args {
    base: u32,
    rows: u32,
    kv_dim: u32,
    kv_stride: u32,
    vt_stride: u32,
}

fn tiled(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    src: &MetalTensor,
    dst: &MetalTensor,
    args: &Args,
) {
    assert_eq!(src.dtype, GgmlType::F16);
    assert_eq!(dst.dtype, GgmlType::F16);
    assert!(args.rows > 0 && args.kv_dim > 0);
    let end = args.base as usize + args.rows as usize;
    assert!(args.vt_stride as usize >= end && args.kv_stride >= args.kv_dim);
    assert!(
        src.n_elements() as usize >= (end - 1) * args.kv_stride as usize + args.kv_dim as usize
    );
    assert!(
        dst.n_elements() as usize >= (args.kv_dim as usize - 1) * args.vt_stride as usize + end
    );
    let pso = ctx.pipeline("kernel_vt_transpose_u16_tiled").unwrap();
    enc.set_pipeline(&pso);
    enc.set_bytes(0, args);
    enc.set_tensor(1, src);
    enc.set_tensor(2, dst);
    enc.set_threadgroup_memory(0, 32 * 33 * 2);
    enc.dispatch(
        MTLSize {
            width: (args.kv_dim as usize).div_ceil(32),
            height: (args.rows as usize).div_ceil(32),
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
}

fn tensor(ctx: &MetalContext, values: &[u16]) -> MetalTensor {
    MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(values),
        vec![values.len() as u64],
        GgmlType::F16,
    )
    .unwrap()
}

fn read(t: &MetalTensor) -> Vec<u16> {
    assert_eq!(t.buffer.storageMode(), objc2_metal::MTLStorageMode::Shared);
    unsafe {
        std::slice::from_raw_parts(
            (t.buffer.contents().as_ptr() as *const u8).add(t.offset as usize) as *const u16,
            t.n_elements() as usize,
        )
        .to_vec()
    }
}

#[test]
#[ignore = "serial Metal raw-bit transpose oracle including NaNs and subspans"]
fn vt_tiled_raw_bits_and_subspans() {
    let ctx = MetalContext::new().unwrap();
    for (kv_dim, base, rows, n_pos, k_pad, v_pad) in [
        (1usize, 0usize, 1usize, 1usize, 0usize, 3usize),
        (38, 7, 33, 53, 5, 11),
        (1024, 3, 65, 79, 7, 13),
        (1024, 31, 1025, 1089, 1, 5),
        (1024, 0, 32768, 32768, 0, 0),
    ] {
        let kv_stride = kv_dim + k_pad;
        let vt_stride = n_pos + v_pad;
        let source: Vec<u16> = (0..n_pos * kv_stride)
            .map(|i| i.wrapping_mul(40503) as u16)
            .collect();
        if source.len() >= 65536 {
            let mut seen = vec![false; 65536];
            for &bits in &source[..65536] {
                seen[bits as usize] = true;
            }
            assert!(seen.into_iter().all(|present| present));
        }
        let mut expected = vec![0x7e55u16; kv_dim * vt_stride + 16];
        for pos in base..base + rows {
            for component in 0..kv_dim {
                expected[8 + component * vt_stride + pos] = source[pos * kv_stride + component];
            }
        }
        let src = tensor(&ctx, &source);
        let a_storage = tensor(&ctx, &vec![0x7e55; expected.len()]);
        let b_storage = tensor(&ctx, &vec![0x7e55; expected.len()]);
        let a = a_storage.view_subrange(8, vec![(kv_dim * vt_stride) as u64]);
        let b = b_storage.view_subrange(8, vec![(kv_dim * vt_stride) as u64]);
        let args = Args {
            base: base as u32,
            rows: rows as u32,
            kv_dim: kv_dim as u32,
            kv_stride: kv_stride as u32,
            vt_stride: vt_stride as u32,
        };
        one_shot(&ctx, |enc| {
            encode_attn_matrix_transpose_v_f16_mode(
                &ctx, enc, &src, &a, base, rows, n_pos, kv_stride, vt_stride, 1, kv_dim, true,
            )?;
            tiled(&ctx, enc, &src, &b, &args);
            Ok(())
        })
        .unwrap();
        assert_eq!(read(&a_storage), expected, "incumbent raw-bit span");
        assert_eq!(read(&b_storage), expected, "tiled raw-bit span");
        assert_eq!(read(&src), source);
        println!(
            "VT_TILED_JSON {}",
            json!({"kind":"oracle", "kv_dim":kv_dim, "base":base,
            "rows":rows, "n_pos":n_pos, "kv_stride":kv_stride, "vt_stride":vt_stride, "bitwise":true})
        );
    }
}

#[test]
#[ignore = "serial Metal 16 distinct-layer restored-prefix transpose screen"]
fn vt_tiled_layer_bank_screen() {
    let ctx = MetalContext::new().unwrap();
    const LAYERS: usize = 16;
    const KV_DIM: usize = 1024;
    for rows in [8840usize, 32752] {
        let before = ctx.current_allocated_size();
        let mut banks = Vec::new();
        for layer in 0..LAYERS {
            let source: Vec<u16> = (0..rows * KV_DIM)
                .map(|i| (i ^ (layer * 7919)) as u16)
                .collect();
            banks.push(tensor(&ctx, &source));
        }
        let dst = MetalTensor::zeros_f16(&ctx, vec![(rows * KV_DIM) as u64]).unwrap();
        let args = Args {
            base: 0,
            rows: rows as u32,
            kv_dim: KV_DIM as u32,
            kv_stride: KV_DIM as u32,
            vt_stride: rows as u32,
        };
        println!(
            "VT_TILED_JSON {}",
            json!({"kind":"allocation", "rows":rows,
            "banks":LAYERS, "allocated_bytes":ctx.current_allocated_size()-before})
        );
        for (index, arm) in ["A", "B", "B", "A", "A", "B", "B", "A"]
            .into_iter()
            .enumerate()
        {
            let start = std::time::Instant::now();
            let cmd = ctx.queue.commandBuffer().unwrap();
            for src in &banks {
                let enc = KernelEncoder::begin(&cmd);
                if arm == "A" {
                    encode_attn_matrix_transpose_v_f16_mode(
                        &ctx, &enc, src, &dst, 0, rows, rows, KV_DIM, rows, 4, 256, true,
                    )
                    .unwrap();
                } else {
                    tiled(&ctx, &enc, src, &dst, &args);
                }
                enc.end();
            }
            cmd.commit();
            cmd.waitUntilCompleted();
            let wall_ms = start.elapsed().as_secs_f64() * 1e3;
            assert_eq!(cmd.status(), MTLCommandBufferStatus::Completed);
            assert!(cmd.error().is_none());
            let gpu_ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
            assert!(gpu_ms.is_finite() && gpu_ms > 0.0);
            println!(
                "VT_TILED_JSON {}",
                json!({"kind":if index<4 {"warmup"} else {"sample"},
                "rows":rows, "arm":arm, "index":index, "wall_ms":wall_ms, "gpu_ms":gpu_ms,
                "logical_read_write_bytes":LAYERS * rows * KV_DIM * 4})
            );
        }
    }
}
