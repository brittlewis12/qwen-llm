#![cfg_attr(not(feature = "perf-ablations"), allow(unused))]
//! Isolated Flash-Next HC up-mix re-grid rate probe.
//! Requires `--features perf-ablations`; run under the canonical Metal lease.
use objc2_metal::{MTLCommandBuffer as _, MTLCommandQueue as _};
use qwen_llm::{metal::*, tensor::GgmlType};
use std::io::Write;

const HIDDEN: usize = 2560;
const RANK: usize = 320;
const ROWS: usize = 4 * HIDDEN;

fn emit(v: serde_json::Value) {
    println!("{v}");
    std::io::stdout().flush().unwrap();
}

fn thermal() -> u32 {
    std::process::Command::new("notifyutil")
        .args(["-g", "com.apple.system.thermalpressurelevel"])
        .output()
        .ok()
        .and_then(|o| {
            String::from_utf8_lossy(&o.stdout)
                .split_whitespace()
                .last()?
                .parse()
                .ok()
        })
        .unwrap_or(u32::MAX)
}

fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * 0.2
        })
        .collect()
}

fn run(
    ctx: &MetalContext,
    name: &str,
    rows: usize,
    bank: &MetalTensor,
    low: &MetalTensor,
    norm: &MetalTensor,
    raw: &MetalTensor,
    mixed: &MetalTensor,
) -> f64 {
    let cmd = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&cmd);
    let pso = ctx.pipeline(name).unwrap();
    enc.set_pipeline(&pso);
    enc.set_tensor(0, bank);
    enc.set_tensor(1, low);
    enc.set_tensor(2, norm);
    enc.set_tensor(3, raw);
    enc.set_tensor(4, mixed);
    enc.dispatch(
        objc2_metal::MTLSize {
            width: HIDDEN.div_ceil(rows),
            height: 1,
            depth: 1,
        },
        objc2_metal::MTLSize {
            width: rows * 32,
            height: 1,
            depth: 1,
        },
    );
    enc.end();
    cmd.commit();
    wait_completed(&cmd).unwrap();
    (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e6
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() -> anyhow::Result<()> {
    assert!(std::env::var_os("QWEN_METAL_LEASE_FD").is_some());
    let ctx = MetalContext::new()?;
    let words: Vec<u16> = values(ROWS * RANK, 1)
        .into_iter()
        .map(|v| half::bf16::from_f32(v).to_bits())
        .collect();
    let bank = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&words),
        vec![ROWS as u64, RANK as u64],
        GgmlType::BF16,
    )?;
    let low: Vec<f32> = values(RANK, 2);
    let low = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&low),
        vec![RANK as u64],
        GgmlType::F32,
    )?;
    let norm: Vec<f32> = values(ROWS, 3).into_iter().map(|v| v * 3.0).collect();
    let norm = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&norm),
        vec![ROWS as u64],
        GgmlType::F32,
    )?;
    let raw = MetalTensor::zeros_f32(&ctx, vec![ROWS as u64])?;
    let mixed = MetalTensor::zeros_f32(&ctx, vec![HIDDEN as u64])?;
    let variants = [
        ("kernel_hc_up_regrid_r2", 2usize),
        ("kernel_hc_up_regrid_r4", 4),
        ("kernel_hc_up_regrid_r8", 8),
        ("kernel_hc_up_regrid_r16", 16),
    ];
    emit(
        serde_json::json!({"kind":"header","thermal":thermal(),"rows":ROWS,"rank":RANK,"device":ctx.device_facts()}),
    );
    // Warm compile and cache.
    for (name, rows) in variants {
        for _ in 0..8 {
            run(&ctx, name, rows, &bank, &low, &norm, &raw, &mixed);
        }
    }
    let mut times = vec![Vec::new(); variants.len()];
    for rep in 0..12 {
        let order: Vec<usize> = if rep % 2 == 0 {
            (0..variants.len()).collect()
        } else {
            (0..variants.len()).rev().collect()
        };
        for i in order {
            let (name, rows) = variants[i];
            times[i].push(run(&ctx, name, rows, &bank, &low, &norm, &raw, &mixed));
        }
    }
    let base = median(times[1].clone());
    for (i, (name, rows)) in variants.iter().enumerate() {
        let m = median(times[i].clone());
        emit(
            serde_json::json!({"kind":"rate","name":name,"rows_per_tg":rows,"threads":rows*32,"us":m,"gbps":(ROWS*RANK*2) as f64/m/1e3,"vs_r4":base/m,"thermal":thermal()}),
        );
    }
    Ok(())
}
