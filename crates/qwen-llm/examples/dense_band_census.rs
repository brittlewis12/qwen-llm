//! Isolated per-shape rate census for the V4.1 batched dense (Q5_K/Q8_0) band.
//!
//! Usage:
//!   dense_band_census [--cols 1,2,3,4] [--reps 12] [--target-mib 384]
//!                     [--shapes q_b,output_a,...] [--include-head]
//!
//! For each projection shape it times, at `cols` activation columns:
//!   * the production per-column native GEMV path (one dispatch per column),
//!   * the production skinny MMA dispatch with the automatic K split, and
//!   * the same skinny dispatch swept over explicit splits.
//!
//! Weights rotate from a set of copies totalling at least `--target-mib` so
//! reads stream from DRAM. Medians of per-dispatch GPU time are emitted as
//! JSON lines. This is a probe: it requires `perf-ablations` and the canonical
//! Metal lease.

use anyhow::{Context, Result, ensure};
use objc2_metal::{MTLBuffer as _, MTLCommandBuffer as _, MTLCommandQueue as _};
use qwen_llm::metal::*;
use qwen_llm::tensor::GgmlType;
use std::io::Write;
use std::process::Command;

const Q5_K_BYTES: usize = 176;
const Q8_0_BYTES: usize = 34;

fn emit(v: serde_json::Value) {
    println!("{v}");
    std::io::stdout().flush().unwrap();
}

fn thermal() -> u32 {
    Command::new("notifyutil")
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

fn random_bytes(len: usize, seed: u64) -> Vec<u8> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    (0..len).map(|_| next() as u8).collect()
}

/// Q5_K bytes with finite per-block scale/min (small positive halves).
fn random_q5_k(n_in: usize, n_out: usize, seed: u64) -> Vec<u8> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let blocks = n_in / 256 * n_out;
    let mut bytes = random_bytes(blocks * Q5_K_BYTES, seed);
    for block in bytes.as_chunks_mut::<Q5_K_BYTES>().0 {
        for offset in [0, 2] {
            let v = 0.0005 + (next() % 1000) as f32 * 0.00002;
            block[offset..offset + 2].copy_from_slice(&half::f16::from_f32(v).to_le_bytes());
        }
    }
    bytes
}

/// Q8_0 bytes with finite per-block scale (small positive half at offset 0).
fn random_q8_0(n_in: usize, n_out: usize, seed: u64) -> Vec<u8> {
    let mut state = seed.wrapping_mul(0xD1B5_4A32_D192_ED03) | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let blocks = n_in / 32 * n_out;
    let mut bytes = random_bytes(blocks * Q8_0_BYTES, seed);
    for block in bytes.as_chunks_mut::<Q8_0_BYTES>().0 {
        let v = 0.0005 + (next() % 1000) as f32 * 0.00002;
        block[0..2].copy_from_slice(&half::f16::from_f32(v).to_le_bytes());
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

fn f32_tensor(ctx: &MetalContext, values: &[f32], shape: &[usize]) -> MetalTensor {
    MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(values),
        shape.iter().map(|&v| v as u64).collect(),
        GgmlType::F32,
    )
    .unwrap()
}

fn i32_tensor(ctx: &MetalContext, values: &[i32]) -> MetalTensor {
    MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(values),
        vec![values.len() as u64],
        GgmlType::I32,
    )
    .unwrap()
}

fn run(ctx: &MetalContext, encode: &dyn Fn(&KernelEncoder) -> Result<()>) -> Result<f64> {
    let cmd = ctx.queue.commandBuffer().context("command buffer")?;
    let enc = KernelEncoder::begin(&cmd);
    encode(&enc)?;
    enc.end();
    cmd.commit();
    wait_completed(&cmd)?;
    Ok((cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e6)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|p, q| p.partial_cmp(q).unwrap());
    v[v.len() / 2]
}

/// Divisors of `n` at most 8, in ascending order.
fn splits(n: usize) -> Vec<usize> {
    (1..=8).filter(|&d| n.is_multiple_of(d)).collect()
}

/// The production split heuristic, mirrored for reporting.
fn auto_split(blocks: usize, channel_groups: usize) -> usize {
    let mut split = 1;
    for d in (1..=8).filter(|&d| blocks.is_multiple_of(d)) {
        split = d;
        if channel_groups * d >= 2560 {
            break;
        }
    }
    split
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Dense,
    Grouped,
}

struct Spec {
    name: &'static str,
    kind: Kind,
    n_in: usize,
    n_out: usize,
    groups: usize,
}

/// A measured arm: encode the path once (per-column arms encode `cols`
/// dispatches internally, matching the production per-column path).
enum Arm {
    PerColumn,
    Auto,
    Split(usize),
    /// Work-split Q5_K with `cols` accumulators (probe kernel).
    SplitCols,
}

impl Arm {
    fn label(&self) -> String {
        match self {
            Arm::PerColumn => "per_column".into(),
            Arm::Auto => "auto".into(),
            Arm::Split(s) => format!("split{s}"),
            Arm::SplitCols => "split_cols".into(),
        }
    }
}

fn census_dense(
    ctx: &MetalContext,
    spec: &Spec,
    cols: &[usize],
    reps: usize,
    target: u64,
) -> Result<()> {
    let row_bytes = spec.n_in / 256 * Q5_K_BYTES;
    let bytes = (row_bytes * spec.n_out * spec.groups) as u64;
    let copies = ((target.div_ceil(bytes)).max(4)) as usize;
    let banks: Vec<MetalTensor> = (0..copies)
        .map(|c| {
            let b: Vec<u8> = (0..spec.groups)
                .flat_map(|g| random_q5_k(spec.n_in, spec.n_out, (c * spec.groups + g) as u64))
                .collect();
            MetalTensor::from_bytes(
                ctx,
                &b,
                vec![spec.n_in as u64, spec.n_out as u64, spec.groups as u64],
                GgmlType::Q5_K,
            )
            .unwrap()
        })
        .collect();
    let ids = i32_tensor(ctx, &(0..spec.groups as i32).collect::<Vec<_>>());
    let status = i32_tensor(ctx, &[1]);
    let x = f32_tensor(
        ctx,
        &random_f32(spec.n_in * spec.groups * 8, 1),
        &[spec.n_in * spec.groups, 8],
    );
    let y = f32_tensor(
        ctx,
        &vec![0.0; spec.n_out * spec.groups * 8],
        &[spec.n_out * spec.groups, 8],
    );

    let blocks = spec.n_in / 256;
    let channel_groups = spec.n_out.div_ceil(16) * spec.groups;
    let auto = auto_split(blocks, channel_groups);
    let mut arms = vec![Arm::PerColumn, Arm::Auto, Arm::SplitCols];
    for s in splits(blocks) {
        if s != auto {
            arms.push(Arm::Split(s));
        }
    }
    let per = 4usize;
    emit(serde_json::json!({
        "event": "shape", "name": spec.name, "kind": "q5_k", "n_in": spec.n_in,
        "n_out": spec.n_out, "groups": spec.groups, "weight_bytes": bytes,
        "copies": copies, "blocks": blocks, "channel_groups": channel_groups,
        "auto_split": auto, "splits": splits(blocks),
    }));

    for &c in cols {
        let xs = x.view_bytes(0, vec![(spec.n_in * spec.groups) as u64, c as u64]);
        let ys = y.view_bytes(0, vec![(spec.n_out * spec.groups) as u64, c as u64]);
        let mut times: Vec<(String, Vec<f64>)> =
            arms.iter().map(|a| (a.label(), Vec::new())).collect();
        for rep in 0..reps {
            for (ai, arm) in arms.iter().enumerate() {
                let base = (rep * arms.len() + ai) * per;
                let t = run(ctx, &|enc| {
                    for j in 0..per {
                        let bank = &banks[(base + j) % copies];
                        match arm {
                            Arm::PerColumn => {
                                for col in 0..c {
                                    encode_all_slots_down_native(
                                        ctx,
                                        enc,
                                        bank,
                                        &x.view_bytes(
                                            (col * spec.n_in * spec.groups * 4) as u64,
                                            vec![spec.n_in as u64, spec.groups as u64],
                                        ),
                                        &ids,
                                        &status,
                                        &y.view_bytes(
                                            (col * spec.n_out * spec.groups * 4) as u64,
                                            vec![spec.n_out as u64, spec.groups as u64],
                                        ),
                                        spec.n_in,
                                        spec.n_out,
                                        spec.groups,
                                        spec.groups,
                                    )?;
                                }
                            }
                            Arm::Auto => {
                                if spec.kind == Kind::Grouped {
                                    encode_grouped_skinny_q5_k(
                                        ctx,
                                        enc,
                                        bank,
                                        &xs,
                                        &ys,
                                        spec.n_in,
                                        spec.n_out,
                                        spec.groups,
                                        c,
                                    )?;
                                } else {
                                    let flat = bank
                                        .view_bytes(0, vec![spec.n_in as u64, spec.n_out as u64]);
                                    encode_dense_skinny_q5_k(
                                        ctx, enc, &flat, &xs, &ys, spec.n_in, spec.n_out, c,
                                    )?;
                                }
                            }
                            Arm::Split(s) => {
                                if spec.kind == Kind::Grouped {
                                    encode_grouped_skinny_q5_k_split_probe(
                                        ctx,
                                        enc,
                                        bank,
                                        &xs,
                                        &ys,
                                        spec.n_in,
                                        spec.n_out,
                                        spec.groups,
                                        c,
                                        *s,
                                    )?;
                                } else {
                                    let flat = bank
                                        .view_bytes(0, vec![spec.n_in as u64, spec.n_out as u64]);
                                    encode_dense_skinny_q5_k_split_probe(
                                        ctx, enc, &flat, &xs, &ys, spec.n_in, spec.n_out, c, *s,
                                    )?;
                                }
                            }
                            Arm::SplitCols => {
                                encode_q5_k_split_cols_probe(
                                    ctx,
                                    enc,
                                    bank,
                                    &xs,
                                    &ys,
                                    spec.n_in,
                                    spec.n_out,
                                    spec.groups,
                                    c,
                                )?;
                            }
                        }
                    }
                    Ok(())
                })?;
                times[ai].1.push(t / per as f64);
            }
        }
        for (label, samples) in &times {
            let med = median(samples.clone());
            emit(serde_json::json!({
                "event": "timing", "name": spec.name, "kind": "q5_k", "cols": c,
                "arm": label, "gpu_us": med, "GBps": bytes as f64 / med / 1e3,
                "auto_split": auto,
            }));
        }
    }
    witness_split_cols(ctx, spec, &banks[0], &x, &y, &ids, &status, cols)?;
    Ok(())
}

/// Compare the multi-column work split against the production per-column
/// split kernel on the same bank and activations, at the B=3 column count when
/// it was requested. The per-column chain is the same lane map and summation
/// order, so agreement is expected to be bitwise.
fn witness_split_cols(
    ctx: &MetalContext,
    spec: &Spec,
    bank: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    ids: &MetalTensor,
    status: &MetalTensor,
    cols: &[usize],
) -> Result<()> {
    let check = if cols.contains(&3) {
        3
    } else {
        *cols.iter().max().unwrap_or(&1)
    }
    .min(4);
    let xs = x.view_bytes(0, vec![(spec.n_in * spec.groups) as u64, check as u64]);
    let ys = y.view_bytes(0, vec![(spec.n_out * spec.groups) as u64, check as u64]);
    run(ctx, &|enc| {
        for col in 0..check {
            encode_all_slots_down_native(
                ctx,
                enc,
                bank,
                &x.view_bytes(
                    (col * spec.n_in * spec.groups * 4) as u64,
                    vec![spec.n_in as u64, spec.groups as u64],
                ),
                ids,
                status,
                &y.view_bytes(
                    (col * spec.n_out * spec.groups * 4) as u64,
                    vec![spec.n_out as u64, spec.groups as u64],
                ),
                spec.n_in,
                spec.n_out,
                spec.groups,
                spec.groups,
            )?;
        }
        Ok(())
    })?;
    let reference = read_f32_tensor(y);
    run(ctx, &|enc| {
        Ok(encode_q5_k_split_cols_probe(
            ctx,
            enc,
            bank,
            &xs,
            &ys,
            spec.n_in,
            spec.n_out,
            spec.groups,
            check,
        )?)
    })?;
    let got = read_f32_tensor(y);
    let n = check * spec.n_out * spec.groups;
    let bitwise = (0..n)
        .filter(|&i| reference[i].to_bits() == got[i].to_bits())
        .count();
    let max_abs = qwen_llm::compare::report_max_abs_diff_f32(&reference[..n], &got[..n]);
    emit(serde_json::json!({
        "event": "witness", "name": spec.name, "cols": check,
        "values": n, "bitwise": bitwise, "max_abs": max_abs,
    }));
    Ok(())
}

fn read_f32_tensor(tensor: &MetalTensor) -> Vec<f32> {
    let mut values = vec![0.0f32; tensor.n_elements() as usize];
    unsafe {
        let source = (tensor.buffer.contents().as_ptr() as *const u8).add(tensor.offset as usize)
            as *const f32;
        std::ptr::copy_nonoverlapping(source, values.as_mut_ptr(), values.len());
    }
    values
}

fn census_pair(ctx: &MetalContext, cols: usize, reps: usize, target: u64) -> Result<()> {
    let (n_in, n_a, n_b) = (5120usize, 1280usize, 512usize);
    let row_bytes = n_in / 32 * Q8_0_BYTES;
    let bytes = (row_bytes * (n_a + n_b)) as u64;
    let copies = ((target.div_ceil(bytes)).max(4)) as usize;
    let q8 = |n: usize, seed: usize| {
        MetalTensor::from_bytes(
            ctx,
            &random_q8_0(n_in, n, seed as u64),
            vec![n_in as u64, n as u64],
            GgmlType::Q8_0,
        )
        .unwrap()
    };
    let pairs: Vec<(MetalTensor, MetalTensor)> = (0..copies)
        .map(|c| (q8(n_a, 2 * c), q8(n_b, 2 * c + 1)))
        .collect();
    let ids = i32_tensor(ctx, &[0]);
    let status = i32_tensor(ctx, &[1]);
    let x = f32_tensor(ctx, &random_f32(n_in * 8, 2), &[n_in, 8]);
    let ya = f32_tensor(ctx, &vec![0.0; n_a * 8], &[n_a, 8]);
    let yb = f32_tensor(ctx, &vec![0.0; n_b * 8], &[n_b, 8]);
    let blocks = n_in / 32;
    let channel_groups = (n_a + n_b).div_ceil(16);
    let auto = auto_split(blocks, channel_groups);
    let mut arms = vec![Arm::PerColumn, Arm::Auto];
    for s in splits(blocks) {
        if s != auto {
            arms.push(Arm::Split(s));
        }
    }
    let per = 4usize;
    emit(serde_json::json!({
        "event": "shape", "name": "q8_pair", "kind": "q8_0", "n_in": n_in,
        "n_a": n_a, "n_b": n_b, "weight_bytes": bytes, "copies": copies,
        "blocks": blocks, "channel_groups": channel_groups, "auto_split": auto,
        "splits": splits(blocks),
    }));
    let xs = x.view_bytes(0, vec![n_in as u64, cols as u64]);
    let yac = ya.view_bytes(0, vec![n_a as u64, cols as u64]);
    let ybc = yb.view_bytes(0, vec![n_b as u64, cols as u64]);
    let mut times: Vec<(String, Vec<f64>)> = arms.iter().map(|a| (a.label(), Vec::new())).collect();
    for rep in 0..reps {
        for (ai, arm) in arms.iter().enumerate() {
            let base = (rep * arms.len() + ai) * per;
            let t = run(ctx, &|enc| {
                for j in 0..per {
                    let (a, b) = &pairs[(base + j) % copies];
                    match arm {
                        Arm::PerColumn => {
                            for col in 0..cols {
                                for (w, n, out) in [(a, n_a, &ya), (b, n_b, &yb)] {
                                    encode_all_slots_down_native(
                                        ctx,
                                        enc,
                                        &w.view_bytes(0, vec![n_in as u64, n as u64, 1]),
                                        &x.view_bytes(
                                            (col * n_in * 4) as u64,
                                            vec![n_in as u64, 1],
                                        ),
                                        &ids,
                                        &status,
                                        &out.view_bytes((col * n * 4) as u64, vec![n as u64, 1]),
                                        n_in,
                                        n,
                                        1,
                                        1,
                                    )?;
                                }
                            }
                        }
                        Arm::Auto => {
                            encode_dense_pair_skinny_q8_0(
                                ctx,
                                enc,
                                a,
                                b,
                                &xs,
                                &yac,
                                &ybc,
                                (n_in, n_a, n_b),
                                cols,
                            )?;
                        }
                        Arm::Split(s) => {
                            encode_dense_pair_skinny_q8_0_split_probe(
                                ctx,
                                enc,
                                a,
                                b,
                                &xs,
                                &yac,
                                &ybc,
                                (n_in, n_a, n_b),
                                cols,
                                *s,
                            )?;
                        }
                        Arm::SplitCols => unreachable!("split_cols is a Q5_K arm"),
                    }
                }
                Ok(())
            })?;
            times[ai].1.push(t / per as f64);
        }
    }
    for (label, samples) in &times {
        let med = median(samples.clone());
        emit(serde_json::json!({
            "event": "timing", "name": "q8_pair", "kind": "q8_0", "cols": cols,
            "arm": label, "gpu_us": med, "GBps": bytes as f64 / med / 1e3,
            "auto_split": auto,
        }));
    }
    Ok(())
}

fn main() -> Result<()> {
    let argv: Vec<String> = std::env::args().collect();
    let mut cols: Vec<usize> = vec![2, 3, 4];
    let mut reps = 12usize;
    let mut target_mib = 384u64;
    let mut selected: Option<Vec<String>> = None;
    let mut include_head = false;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--cols" => {
                i += 1;
                cols = argv[i]
                    .split(',')
                    .map(str::parse)
                    .collect::<Result<_, _>>()?;
            }
            "--reps" => {
                i += 1;
                reps = argv[i].parse()?;
            }
            "--target-mib" => {
                i += 1;
                target_mib = argv[i].parse()?;
            }
            "--shapes" => {
                i += 1;
                selected = Some(argv[i].split(',').map(str::to_string).collect());
            }
            "--include-head" => include_head = true,
            other => anyhow::bail!("unknown argument {other}"),
        }
        i += 1;
    }
    let target = target_mib << 20;
    let mut specs = vec![
        Spec {
            name: "shared_gate_up",
            kind: Kind::Dense,
            n_in: 5120,
            n_out: 2304,
            groups: 1,
        },
        Spec {
            name: "shared_down",
            kind: Kind::Dense,
            n_in: 2304,
            n_out: 5120,
            groups: 1,
        },
        Spec {
            name: "q_b",
            kind: Kind::Dense,
            n_in: 1280,
            n_out: 32768,
            groups: 1,
        },
        Spec {
            name: "output_b",
            kind: Kind::Dense,
            n_in: 8192,
            n_out: 5120,
            groups: 1,
        },
        Spec {
            name: "output_a",
            kind: Kind::Grouped,
            n_in: 4096,
            n_out: 1024,
            groups: 8,
        },
        Spec {
            name: "engram",
            kind: Kind::Dense,
            n_in: 6144,
            n_out: 25600,
            groups: 1,
        },
    ];
    if include_head {
        specs.push(Spec {
            name: "head",
            kind: Kind::Dense,
            n_in: 5120,
            n_out: 129280,
            groups: 1,
        });
    }
    if let Some(want) = &selected {
        specs.retain(|s| want.iter().any(|w| w == s.name));
        let only_pair = want.iter().all(|w| w == "q8_pair");
        ensure!(
            only_pair || !specs.is_empty(),
            "no matching shapes in {want:?}"
        );
    }
    let ctx = MetalContext::new()?;
    emit(serde_json::json!({
        "event": "start", "argv": argv, "cols": cols, "reps": reps,
        "target_mib": target_mib, "thermal": thermal(),
    }));
    let want_pair = selected
        .as_ref()
        .is_none_or(|w| w.iter().any(|s| s == "q8_pair"));
    for s in &specs {
        if let Err(error) = census_dense(&ctx, s, &cols, reps, target) {
            emit(serde_json::json!({
                "event": "shape_error", "name": s.name, "error": format!("{error:#}"),
            }));
        }
    }
    if want_pair {
        for &c in &cols {
            if let Err(error) = census_pair(&ctx, c, reps, target) {
                emit(serde_json::json!({
                    "event": "shape_error", "name": "q8_pair", "cols": c,
                    "error": format!("{error:#}"),
                }));
            }
        }
    }
    emit(serde_json::json!({"event": "complete", "thermal": thermal()}));
    Ok(())
}
