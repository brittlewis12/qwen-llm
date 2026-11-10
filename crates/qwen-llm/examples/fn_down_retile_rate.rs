//! Isolated IQ4_NL expert-down rate probe (research-only).
//!
//! Times the incumbent `kernel_expert_native_iq4_nl_f32_k640_r4` against a
//! repacked-layout twin on the real decode grid (n_out=2560, K=640, top_k=10,
//! rows=4) with synthetic weights. The repacked bank keeps each 32-weight
//! block's 16 qs bytes contiguous and 16-byte aligned and moves the half scales
//! to a separate array; the per-row arithmetic and reduction order are
//! identical, so the probe also checks the outputs bytewise.
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLSize};
use qwen_llm::metal::{KernelEncoder, MetalContext, MetalTensor, commit_and_wait};
use qwen_llm::tensor::GgmlType;

const K: usize = 640;
const N_OUT: usize = 2560;
const TOP_K: usize = 10;
const ROWS: usize = 4;
const BLOCKS: usize = K / 32;
const ROW_BYTES: usize = BLOCKS * 18;
const QS_ROW_BYTES: usize = BLOCKS * 16;
const D_ROW_BYTES: usize = BLOCKS * 2;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ExpertArgs {
    row_bytes: u64,
    n_in: u32,
    n_out: u32,
    experts: u32,
    top_k: u32,
    mode: u32,
    clamp: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RetileArgs {
    qs_row_bytes: u64,
    d_row_bytes: u64,
    n_in: u32,
    n_out: u32,
    experts: u32,
    top_k: u32,
}

fn prng(state: &mut u64) -> u8 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state >> 24) as u8
}

fn encode<T: bytemuck::Pod>(
    ctx: &MetalContext,
    pso: &qwen_llm::metal::Pipeline,
    args: &T,
    tensors: &[&MetalTensor],
    grid: MTLSize,
    threads: MTLSize,
    reps: usize,
) -> Result<f64, Box<dyn std::error::Error>> {
    let command = ctx.queue.commandBuffer().ok_or("no command buffer")?;
    let enc = KernelEncoder::begin(&command);
    for _ in 0..reps {
        enc.set_pipeline(pso);
        enc.set_bytes(0, args);
        for (i, t) in tensors.iter().enumerate() {
            enc.set_tensor(1 + i, t);
        }
        enc.dispatch(grid, threads);
    }
    enc.end();
    commit_and_wait(&command)?;
    Ok((command.GPUEndTime() - command.GPUStartTime()) * 1000.0)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let experts: usize = std::env::args()
        .nth(1)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(64);
    let slots = ROWS * TOP_K;
    let ctx = MetalContext::new()?;
    let mut seed = 0x9e3779b97f4a7c15u64;

    // Incumbent bank: 18-byte blocks with a fixed valid half scale.
    let mut bank = vec![0u8; experts * N_OUT * ROW_BYTES];
    for block in bank.chunks_exact_mut(18) {
        block[0] = 0x00;
        block[1] = 0x2c; // half 0.0625
        for b in &mut block[2..] {
            *b = prng(&mut seed);
        }
    }
    // Repacked twins: 16-byte qs rows, separate half scales.
    let mut qs = vec![0u8; experts * N_OUT * QS_ROW_BYTES];
    let mut ds = vec![0u8; experts * N_OUT * D_ROW_BYTES];
    for e in 0..experts {
        for row in 0..N_OUT {
            let src = (e * N_OUT + row) * ROW_BYTES;
            let qbase = (e * N_OUT + row) * QS_ROW_BYTES;
            let dbase = (e * N_OUT + row) * D_ROW_BYTES;
            for block in 0..BLOCKS {
                let s = src + block * 18;
                ds[dbase + block * 2] = bank[s];
                ds[dbase + block * 2 + 1] = bank[s + 1];
                qs[qbase + block * 16..qbase + block * 16 + 16]
                    .copy_from_slice(&bank[s + 2..s + 18]);
            }
        }
    }
    let inner: Vec<f32> = (0..slots * K)
        .map(|_| (prng(&mut seed) as f32 - 128.0) / 128.0)
        .collect();
    let ids: Vec<i32> = (0..slots)
        .map(|slot| ((slot * experts) / slots) as i32)
        .collect();
    let status = vec![1i32; ROWS];
    let distinct = ids
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>()
        .len();

    let bank_t = MetalTensor::from_bytes(
        &ctx,
        &bank,
        vec![K as u64, N_OUT as u64, experts as u64],
        GgmlType::IQ4_NL,
    )?;
    let qs_t = MetalTensor::from_bytes(&ctx, &qs, vec![qs.len() as u64], GgmlType::I8)?;
    let ds_t = MetalTensor::from_bytes(&ctx, &ds, vec![ds.len() as u64], GgmlType::I8)?;
    let inner_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&inner),
        vec![(slots * K) as u64],
        GgmlType::F32,
    )?;
    let ids_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&ids),
        vec![slots as u64],
        GgmlType::I32,
    )?;
    let status_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&status),
        vec![ROWS as u64],
        GgmlType::I32,
    )?;
    let out_inc = MetalTensor::zeros_f32(&ctx, vec![(slots * N_OUT) as u64])?;
    let out_new = MetalTensor::zeros_f32(&ctx, vec![(slots * N_OUT) as u64])?;

    let inc_pso = ctx.pipeline("kernel_expert_native_iq4_nl_f32_k640_r4")?;
    let new_pso = ctx.pipeline("kernel_fn_down_retile_iq4nl_k640_r4")?;
    let inc_args = ExpertArgs {
        row_bytes: ROW_BYTES as u64,
        n_in: K as u32,
        n_out: N_OUT as u32,
        experts: experts as u32,
        top_k: TOP_K as u32,
        mode: 2,
        clamp: f32::MAX,
    };
    let new_args = RetileArgs {
        qs_row_bytes: QS_ROW_BYTES as u64,
        d_row_bytes: D_ROW_BYTES as u64,
        n_in: K as u32,
        n_out: N_OUT as u32,
        experts: experts as u32,
        top_k: TOP_K as u32,
    };
    let grid = MTLSize {
        width: N_OUT / 16,
        height: TOP_K,
        depth: ROWS,
    };
    let threads = MTLSize {
        width: 128,
        height: 1,
        depth: 1,
    };

    // Heat the GPU and interleave variants so both run at settled clocks.
    const BATCH: usize = 16;
    for _ in 0..8 {
        for which in [0, 1] {
            if which == 0 {
                encode(
                    &ctx,
                    &inc_pso,
                    &inc_args,
                    &[&bank_t, &inner_t, &ids_t, &status_t, &out_inc],
                    grid,
                    threads,
                    BATCH,
                )?;
            } else {
                encode(
                    &ctx,
                    &new_pso,
                    &new_args,
                    &[&qs_t, &ds_t, &inner_t, &ids_t, &status_t, &out_new],
                    grid,
                    threads,
                    BATCH,
                )?;
            }
        }
    }
    let mut inc_ms = Vec::new();
    let mut new_ms = Vec::new();
    for rep in 0..12 {
        let order = if rep % 2 == 0 { [0, 1] } else { [1, 0] };
        for which in order {
            let t = if which == 0 {
                encode(
                    &ctx,
                    &inc_pso,
                    &inc_args,
                    &[&bank_t, &inner_t, &ids_t, &status_t, &out_inc],
                    grid,
                    threads,
                    BATCH,
                )?
            } else {
                encode(
                    &ctx,
                    &new_pso,
                    &new_args,
                    &[&qs_t, &ds_t, &inner_t, &ids_t, &status_t, &out_new],
                    grid,
                    threads,
                    BATCH,
                )?
            } / BATCH as f64;
            if which == 0 {
                inc_ms.push(t);
            } else {
                new_ms.push(t);
            }
        }
    }
    let median = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let inc = median(&mut inc_ms);
    let new = median(&mut new_ms);
    let bytes = (distinct * N_OUT * ROW_BYTES) as f64;
    let gbs = |ms: f64| bytes / (ms / 1e3) / 1e9;

    let read = |t: &MetalTensor| unsafe {
        std::slice::from_raw_parts(
            t.buffer
                .contents()
                .as_ptr()
                .add(t.offset as usize)
                .cast::<f32>(),
            slots * N_OUT,
        )
    };
    let bitwise = bytemuck::cast_slice::<f32, u8>(read(&out_inc))
        == bytemuck::cast_slice::<f32, u8>(read(&out_new));
    println!(
        "{}",
        serde_json::json!({
            "kind": "fn_down_retile_rate",
            "experts": experts,
            "slots": slots,
            "distinct": distinct,
            "bytes": bytes,
            "incumbent_ms": inc,
            "incumbent_gbs": gbs(inc),
            "retile_ms": new,
            "retile_gbs": gbs(new),
            "gain": inc / new - 1.0,
            "bitwise": bitwise,
        })
    );
    Ok(())
}
