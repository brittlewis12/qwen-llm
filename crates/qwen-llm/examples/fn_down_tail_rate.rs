//! Isolated ragged-tail expert-down rate probe (research-only).
//!
//! Times the incumbent `kernel_expert_native_{iq4_nl,q2_0}_f32_k640_r4` against
//! its `_f32_ragged_k640_r4` twin on the real Flash-Next decode grid
//! (n_out=2560, K=640, top_k=10, rows=4) with synthetic weights. The ragged
//! kernel runs one full 32-fragment pass and then spreads the 8-fragment tail
//! over all 32 lanes (one fragment/channel dot per lane), so its cross-lane
//! F32 reduction is regrouped; the probe reports both the rate and the output
//! delta against the incumbent.
//!
//! Usage: fn_down_tail_rate [iq4_nl|q2_0] [experts]
//! Grid/heating match `fn_down_retile_rate`: 40 distinct experts, 16
//! dispatches per command, eight interleaved heat rounds, 12 interleaved
//! measured repetitions.
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLSize};
use qwen_llm::metal::{KernelEncoder, MetalContext, MetalTensor, commit_and_wait};
use qwen_llm::tensor::GgmlType;

const K: usize = 640;
const N_OUT: usize = 2560;
const TOP_K: usize = 10;
const ROWS: usize = 4;

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
    let dtype = std::env::args().nth(1).unwrap_or_else(|| "iq4_nl".into());
    let (ggml, inc_name, new_name, block, nl) = match dtype.as_str() {
        // 18-byte IQ4_NL block: half scale + 16 code bytes (32 weights).
        "iq4_nl" => (
            GgmlType::IQ4_NL,
            "kernel_expert_native_iq4_nl_f32_k640_r4",
            "kernel_expert_native_iq4_nl_f32_ragged_k640_r4",
            18usize,
            2usize,
        ),
        // 18-byte Q2_0 block: half scale + 16 packed 2-bit bytes (64 weights).
        "q2_0" => (
            GgmlType::Q2_0,
            "kernel_expert_native_q2_0_f32_k640_r4",
            "kernel_expert_native_q2_0_f32_ragged_k640_r4",
            18usize,
            4usize,
        ),
        other => return Err(format!("unknown dtype {other:?}; use iq4_nl or q2_0").into()),
    };
    let experts: usize = std::env::args()
        .nth(2)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(64);
    let row_bytes = (K / (16 * nl)) * block;
    let slots = ROWS * TOP_K;
    let ctx = MetalContext::new()?;
    let mut seed = 0x9e3779b97f4a7c15u64;

    let mut bank = vec![0u8; experts * N_OUT * row_bytes];
    for blk in bank.chunks_exact_mut(block) {
        blk[0] = 0x00;
        blk[1] = 0x2c; // half 0.0625
        for b in &mut blk[2..] {
            *b = prng(&mut seed);
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
        ggml,
    )?;
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

    let inc_pso = ctx.pipeline(inc_name)?;
    let new_pso = ctx.pipeline(new_name)?;
    let args = ExpertArgs {
        row_bytes: row_bytes as u64,
        n_in: K as u32,
        n_out: N_OUT as u32,
        experts: experts as u32,
        top_k: TOP_K as u32,
        mode: 2,
        clamp: f32::MAX,
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
                    &args,
                    &[&bank_t, &inner_t, &ids_t, &status_t, &out_inc],
                    grid,
                    threads,
                    BATCH,
                )?;
            } else {
                encode(
                    &ctx,
                    &new_pso,
                    &args,
                    &[&bank_t, &inner_t, &ids_t, &status_t, &out_new],
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
                    &args,
                    &[&bank_t, &inner_t, &ids_t, &status_t, &out_inc],
                    grid,
                    threads,
                    BATCH,
                )?
            } else {
                encode(
                    &ctx,
                    &new_pso,
                    &args,
                    &[&bank_t, &inner_t, &ids_t, &status_t, &out_new],
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
    let bytes = (distinct * N_OUT * row_bytes) as f64;
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
    let (inc_v, new_v) = (read(&out_inc), read(&out_new));
    let bitwise = bytemuck::cast_slice::<f32, u8>(inc_v) == bytemuck::cast_slice::<f32, u8>(new_v);
    let max_abs = inc_v
        .iter()
        .zip(new_v)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let l2_ref = inc_v.iter().map(|v| v * v).sum::<f32>().sqrt();
    let l2_diff = inc_v
        .iter()
        .zip(new_v)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>()
        .sqrt();
    println!(
        "{}",
        serde_json::json!({
            "kind": "fn_down_tail_rate",
            "dtype": dtype,
            "experts": experts,
            "slots": slots,
            "distinct": distinct,
            "bytes": bytes,
            "incumbent_ms": inc,
            "incumbent_gbs": gbs(inc),
            "ragged_ms": new,
            "ragged_gbs": gbs(new),
            "gain": inc / new - 1.0,
            "bitwise": bitwise,
            "max_abs_diff": max_abs,
            "rel_l2_diff": l2_diff / l2_ref.max(f32::MIN_POSITIVE),
        })
    );
    Ok(())
}
