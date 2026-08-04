//! T8a: V=2 / cheap-V1 decode-code search on the validated synthetic oracle.
//! Preregistration + gates: docs/bench/2026-07-19-trellis3-code-search/.
//!
//! Screens candidate state->value(s) maps (controls, register-shuffle LUTs,
//! nonlinear mixes, imul-free codes, B8 sub-scales) at 64 groups, then
//! rescreens every row within 1.10x of the V1 baseline at 256 groups x 2
//! seeds. CPU-only. Run niced:
//!   nice -n 19 env QWEN_T8_THREADS=8 cargo run --release -p qwen-llm \
//!     --example trellis3_code_search

use qwen_llm::trellis_offline::{TrellisCode, encode_group, encode_group_sub};
use std::time::Instant;

const NSTATES: usize = 1 << 16;
const LCG_A: u32 = 89_226_354;
const LCG_B: u32 = 64_248_484;
const MASK: u32 = 0x8FFF_8FFF;
const FIXED: u32 = 0x3B60_3B60 & !MASK;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn gaussian(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    let mut out = Vec::with_capacity(n + 1);
    while out.len() < n {
        let u1 = ((splitmix64(&mut s) >> 11) as f64 / (1u64 << 53) as f64).max(1e-15);
        let u2 = (splitmix64(&mut s) >> 11) as f64 / (1u64 << 53) as f64;
        let r = (-2.0 * u1.ln()).sqrt();
        out.push((r * (2.0 * std::f64::consts::PI * u2).cos()) as f32);
        out.push((r * (2.0 * std::f64::consts::PI * u2).sin()) as f32);
    }
    out.truncate(n);
    out
}

fn f16r(x: f32) -> f32 {
    half::f16::from_f32(x).to_f32()
}

fn halves(bits: u32) -> (f32, f32) {
    (
        half::f16::from_bits((bits & 0xFFFF) as u16).to_f32(),
        half::f16::from_bits((bits >> 16) as u16).to_f32(),
    )
}

/// Inverse normal CDF (Acklam) — for Gaussian-quantile tables.
fn inv_phi(p: f64) -> f64 {
    let a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383_577_518_672_69e2,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    let b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    let c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    let d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    let plow = 0.02425;
    if p < plow {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= 1.0 - plow {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        -inv_phi(1.0 - p)
    }
}

fn q32_table() -> [f32; 32] {
    let mut q = [0f32; 32];
    for (i, v) in q.iter_mut().enumerate() {
        *v = f16r(inv_phi((i as f64 + 0.5) / 32.0) as f32);
    }
    q
}

/// k-means on 2D samples; returns k centroids.
fn kmeans2d(samples: &[(f32, f32)], k: usize, iters: usize, seed: u64) -> Vec<(f32, f32)> {
    let mut s = seed;
    let mut cent: Vec<(f32, f32)> = (0..k)
        .map(|_| samples[(splitmix64(&mut s) as usize) % samples.len()])
        .collect();
    for _ in 0..iters {
        let mut sum = vec![(0f64, 0f64, 0f64); k];
        for &(x, y) in samples {
            let mut bi = 0;
            let mut bd = f32::INFINITY;
            for (i, &(cx, cy)) in cent.iter().enumerate() {
                let d = (x - cx) * (x - cx) + (y - cy) * (y - cy);
                if d < bd {
                    bd = d;
                    bi = i;
                }
            }
            sum[bi].0 += x as f64;
            sum[bi].1 += y as f64;
            sum[bi].2 += 1.0;
        }
        for (c, acc) in cent.iter_mut().zip(&sum) {
            if acc.2 > 0.0 {
                *c = ((acc.0 / acc.2) as f32, (acc.1 / acc.2) as f32);
            }
        }
    }
    cent
}

fn hash(st: u32) -> u32 {
    st.wrapping_mul(LCG_A).wrapping_add(LCG_B)
}

struct Row {
    name: String,
    code: TrellisCode,
    n_sub: usize,
    bpw: f64,
    kernel_ok: bool,
}

fn build_rows() -> Vec<Row> {
    let mut rows = Vec::new();
    let q32 = q32_table();

    // Baselines.
    rows.push(Row {
        name: "V1 maskor (baseline)".into(),
        code: TrellisCode::v1_maskor(),
        n_sub: 1,
        bpw: 3.0625,
        kernel_ok: true,
    });
    rows.push(Row {
        name: "V2 split (baseline)".into(),
        code: TrellisCode::v2_split_maskor(),
        n_sub: 1,
        bpw: 3.0625,
        kernel_ok: true,
    });

    // Controls: RPTC (random Gaussian labels), topology bound.
    {
        let mut s = 0xC0_47u64;
        let g: Vec<f32> = gaussian(2 * NSTATES, splitmix64(&mut s));
        let vx1: Vec<f32> = g[..NSTATES].iter().map(|v| f16r(*v)).collect();
        rows.push(Row {
            name: "RPTC-V1 (control)".into(),
            code: TrellisCode::from_tables(1, vx1, Vec::new()),
            n_sub: 1,
            bpw: 3.0625,
            kernel_ok: false,
        });
        let vx2: Vec<f32> = g[..NSTATES].iter().map(|v| f16r(*v)).collect();
        let vy2: Vec<f32> = g[NSTATES..].iter().map(|v| f16r(*v)).collect();
        rows.push(Row {
            name: "RPTC-V2 (control)".into(),
            code: TrellisCode::from_tables(2, vx2, vy2),
            n_sub: 1,
            bpw: 3.0625,
            kernel_ok: false,
        });
    }

    // Control: HYB-512 (tuned small alphabet, two sign flips).
    {
        let samp: Vec<(f32, f32)> = gaussian(200_000, 0x11B5)
            .chunks(2)
            .map(|c| (c[0], c[1]))
            .collect();
        let cent = kmeans2d(&samp, 512, 15, 0x5EED);
        let mut vx = vec![0f32; NSTATES];
        let mut vy = vec![0f32; NSTATES];
        for st in 0..NSTATES as u32 {
            let h = hash(st);
            let (cx, cy) = cent[((h >> 6) & 511) as usize];
            let sx = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
            let sy = if h & 0x8000_0000 != 0 { -1.0 } else { 1.0 };
            vx[st as usize] = f16r(cx.abs() * sx);
            vy[st as usize] = f16r(cy.abs() * sy);
        }
        rows.push(Row {
            name: "HYB-512 2sign (control)".into(),
            code: TrellisCode::from_tables(2, vx, vy),
            n_sub: 1,
            bpw: 3.0625,
            kernel_ok: false,
        });
    }

    // C1 cart32: two 5-bit quantile lookups from one hash.
    for (s0, s1) in [(0u32, 5u32), (3, 9), (5, 10), (2, 11)] {
        let mut vx = vec![0f32; NSTATES];
        let mut vy = vec![0f32; NSTATES];
        for st in 0..NSTATES as u32 {
            let h = hash(st);
            vx[st as usize] = q32[((h >> s0) & 31) as usize];
            vy[st as usize] = q32[((h >> s1) & 31) as usize];
        }
        rows.push(Row {
            name: format!("C1 cart32 S{s0}/{s1}"),
            code: TrellisCode::from_tables(2, vx, vy),
            n_sub: 1,
            bpw: 3.0625,
            kernel_ok: true,
        });
    }

    // C2 joint32 (+ C3 swap variant at S=6): 32 positive-quadrant pair
    // centroids, two sign bits (and optionally one swap bit).
    {
        let samp: Vec<(f32, f32)> = gaussian(200_000, 0x2D2D)
            .chunks(2)
            .map(|c| (c[0].abs(), c[1].abs()))
            .collect();
        let cent = kmeans2d(&samp, 32, 20, 0xA5A5);
        for s in [3u32, 6, 9] {
            let mut vx = vec![0f32; NSTATES];
            let mut vy = vec![0f32; NSTATES];
            for st in 0..NSTATES as u32 {
                let h = hash(st);
                let (cx, cy) = cent[((h >> s) & 31) as usize];
                let sx = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
                let sy = if h & 0x8000_0000 != 0 { -1.0 } else { 1.0 };
                vx[st as usize] = f16r(cx * sx);
                vy[st as usize] = f16r(cy * sy);
            }
            rows.push(Row {
                name: format!("C2 joint32 S{s}"),
                code: TrellisCode::from_tables(2, vx, vy),
                n_sub: 1,
                bpw: 3.0625,
                kernel_ok: true,
            });
        }
        // C3: S=6 with hash-bit coordinate swap.
        let mut vx = vec![0f32; NSTATES];
        let mut vy = vec![0f32; NSTATES];
        for st in 0..NSTATES as u32 {
            let h = hash(st);
            let (mut cx, mut cy) = cent[((h >> 6) & 31) as usize];
            if h & (1 << 20) != 0 {
                std::mem::swap(&mut cx, &mut cy);
            }
            let sx = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
            let sy = if h & 0x8000_0000 != 0 { -1.0 } else { 1.0 };
            vx[st as usize] = f16r(cx * sx);
            vy[st as usize] = f16r(cy * sy);
        }
        rows.push(Row {
            name: "C3 joint32+swap S6".into(),
            code: TrellisCode::from_tables(2, vx, vy),
            n_sub: 1,
            bpw: 3.0625,
            kernel_ok: true,
        });
    }

    // C4 sum/product.
    for c in [0.5f32, 1.0, 2.0] {
        let mut vx = vec![0f32; NSTATES];
        let mut vy = vec![0f32; NSTATES];
        for st in 0..NSTATES as u32 {
            let hb = (hash(st) & MASK) | FIXED;
            let (zx, zy) = halves(hb);
            vx[st as usize] = f16r(zx + zy);
            vy[st as usize] = f16r(c * zx * zy);
        }
        rows.push(Row {
            name: format!("C4 sum/prod C={c}"),
            code: TrellisCode::from_tables(2, vx, vy),
            n_sub: 1,
            bpw: 3.0625,
            kernel_ok: true,
        });
    }

    // C5 dual-3INST (quality ceiling; over kernel budget).
    for r in [7u32, 13] {
        let mut vx = vec![0f32; NSTATES];
        let mut vy = vec![0f32; NSTATES];
        for st in 0..NSTATES as u32 {
            let h = hash(st);
            let g = h ^ h.rotate_left(r);
            let (ax, ay) = halves((h & MASK) | FIXED);
            let (bx, by) = halves((g & MASK) | FIXED);
            vx[st as usize] = f16r(ax + ay);
            vy[st as usize] = f16r(bx + by);
        }
        rows.push(Row {
            name: format!("C5 dual3inst R{r} (ceiling)"),
            code: TrellisCode::from_tables(2, vx, vy),
            n_sub: 1,
            bpw: 3.0625,
            kernel_ok: false,
        });
    }

    // C5b (iteration-2 candidate): dual code with CHEAP second word
    // hb_b = ((h >> 3) & MASK) | FIXED — saves the rotl vs C5.
    {
        let mut vx = vec![0f32; NSTATES];
        let mut vy = vec![0f32; NSTATES];
        for st in 0..NSTATES as u32 {
            let h = hash(st);
            let (ax, ay) = halves((h & MASK) | FIXED);
            let (bx, by) = halves(((h >> 3) & MASK) | FIXED);
            vx[st as usize] = f16r(ax + ay);
            vy[st as usize] = f16r(bx + by);
        }
        rows.push(Row {
            name: "C5b dual shr3 (cheap)".into(),
            code: TrellisCode::from_tables(2, vx, vy),
            n_sub: 1,
            bpw: 3.0625,
            kernel_ok: true,
        });
    }

    // V1-RLUT32 (imul-free): raw state bits or xor-folded index.
    for s in [3u32, 7, 11] {
        let mut vx = vec![0f32; NSTATES];
        for st in 0..NSTATES as u32 {
            vx[st as usize] = q32[((st >> s) & 31) as usize];
        }
        rows.push(Row {
            name: format!("V1-RLUT32 S{s} (imul-free)"),
            code: TrellisCode::from_tables(1, vx, Vec::new()),
            n_sub: 1,
            bpw: 3.0625,
            kernel_ok: true,
        });
    }
    {
        let mut vx = vec![0f32; NSTATES];
        for st in 0..NSTATES as u32 {
            let q = st ^ (st >> 5) ^ (st >> 10);
            vx[st as usize] = q32[(q & 31) as usize];
        }
        rows.push(Row {
            name: "V1-RLUT32 xorfold (imul-free)".into(),
            code: TrellisCode::from_tables(1, vx, Vec::new()),
            n_sub: 1,
            bpw: 3.0625,
            kernel_ok: true,
        });
    }

    // V1-ARX (imul-free packed u16 add/rotate).
    for r in [5u32, 9] {
        let mut vx = vec![0f32; NSTATES];
        for st in 0..NSTATES as u32 {
            let u0 = (st as u16).wrapping_add(0x9E37);
            let u1 = (st as u16).rotate_left(r).wrapping_add(0x7F4A);
            let hb = (((u1 as u32) << 16) | u0 as u32) & MASK | FIXED;
            let (zx, zy) = halves(hb);
            vx[st as usize] = f16r(zx + zy);
        }
        rows.push(Row {
            name: format!("V1-ARX R{r} (imul-free)"),
            code: TrellisCode::from_tables(1, vx, Vec::new()),
            n_sub: 1,
            bpw: 3.0625,
            kernel_ok: true,
        });
    }

    // B8 sub-scale rows on the two baselines.
    for (n_sub, bpw) in [(2usize, 3.125f64), (4, 3.25)] {
        rows.push(Row {
            name: format!("B8 V2 split /{}", 256 / n_sub),
            code: TrellisCode::v2_split_maskor(),
            n_sub,
            bpw,
            kernel_ok: true,
        });
        rows.push(Row {
            name: format!("B8 V1 maskor /{}", 256 / n_sub),
            code: TrellisCode::v1_maskor(),
            n_sub,
            bpw,
            kernel_ok: true,
        });
    }

    rows
}

fn run_row(x: &[f32], row: &Row, n_threads: usize) -> f64 {
    let groups: Vec<&[f32]> = x.chunks(256).collect();
    let err: f64 = std::thread::scope(|scope| {
        let chunk = groups.len().div_ceil(n_threads);
        let handles: Vec<_> = groups
            .chunks(chunk)
            .map(|gs| {
                scope.spawn(move || {
                    gs.iter()
                        .map(|g| {
                            if row.n_sub == 1 {
                                encode_group(g, &row.code).sq_err
                            } else {
                                encode_group_sub(g, &row.code, row.n_sub)
                            }
                        })
                        .sum::<f64>()
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).sum()
    });
    let norm2: f64 = x.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    (err / norm2).sqrt()
}

fn main() {
    let n_threads = std::env::var("QWEN_T8_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    let rows = build_rows();

    // Phase 1: screen at 64 groups, seed A.
    let xa = gaussian(64 * 256, 0x007E_1115);
    println!("== T8a screen (64 groups, seed A) ==");
    let mut screen: Vec<(usize, f64)> = Vec::new();
    let mut v1_base = 0f64;
    for (i, row) in rows.iter().enumerate() {
        let t0 = Instant::now();
        let e = run_row(&xa, row, n_threads);
        if i == 0 {
            v1_base = e;
        }
        println!(
            "{:<32} bpw={:.4} rel-err={:.5}  vsV1={:.3}x  [{}] ({:.1}s)",
            row.name,
            row.bpw,
            e,
            e / v1_base,
            if row.kernel_ok { "kernel" } else { "control" },
            t0.elapsed().as_secs_f32()
        );
        screen.push((i, e));
    }

    // Phase 2: confirm every row within 1.10x of V1 at 256 groups x 2 seeds.
    println!("\n== T8a confirm (256 groups x seeds A,B) for rows <= 1.10x V1 ==");
    let xa2 = gaussian(256 * 256, 0x007E_1115);
    let xb2 = gaussian(256 * 256, 0xB0B_CAFE);
    let mut confirm_v1 = 0f64;
    let mut results: Vec<(String, f64, bool, f64)> = Vec::new();
    for (i, e_screen) in &screen {
        let row = &rows[*i];
        if *e_screen > 1.10 * v1_base && *i != 0 {
            continue;
        }
        let t0 = Instant::now();
        let ea = run_row(&xa2, row, n_threads);
        let eb = run_row(&xb2, row, n_threads);
        let e = 0.5 * (ea + eb);
        if *i == 0 {
            confirm_v1 = e;
        }
        println!(
            "{:<32} rel-err={:.5} (A {:.5} / B {:.5})  vsV1={:.3}x  ({:.0}s)",
            row.name,
            e,
            ea,
            eb,
            e / confirm_v1,
            t0.elapsed().as_secs_f32()
        );
        results.push((row.name.clone(), e, row.kernel_ok, row.bpw));
    }

    println!("\n== gate evaluation (docs/bench/2026-07-19-trellis3-code-search) ==");
    for (name, e, kernel_ok, _bpw) in &results {
        let r = e / confirm_v1;
        if *kernel_ok && name.starts_with('C') && r <= 1.03 {
            println!("PROMOTE-V2: {name} at {r:.3}x V1");
        }
        if *kernel_ok && name.starts_with("V1-") && r <= 1.03 {
            println!("PROMOTE-V1-CHEAP: {name} at {r:.3}x V1");
        }
    }
    if let Some((_, e_rptc2, _, _)) = results.iter().find(|r| r.0.starts_with("RPTC-V2")) {
        let r = e_rptc2 / confirm_v1;
        println!(
            "RPTC-V2 topology bound: {r:.3}x V1 {}",
            if r > 1.03 {
                "-> STOP gate fires (deficit is topological)"
            } else {
                "(V2 closable in principle)"
            }
        );
    }
    if let (Some((_, e_v2_128, _, _)), Some((_, e_v2, _, _))) = (
        results.iter().find(|r| r.0 == "B8 V2 split /128"),
        results.iter().find(|r| r.0 == "V2 split (baseline)"),
    ) {
        println!(
            "B8 per-128 vs per-256 (V2): {:.3}x {}",
            e_v2_128 / e_v2,
            if e_v2_128 / e_v2 <= 0.92 {
                "-> B8-FLIP fires"
            } else {
                "(below flip margin)"
            }
        );
    }
}
