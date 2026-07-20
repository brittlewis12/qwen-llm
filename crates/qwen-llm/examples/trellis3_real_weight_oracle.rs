//! T7a: real-weight fidelity oracle.
//! Preregistration + gates: docs/bench/2026-07-19-trellis3-real-weight-oracle/.
//!
//! Samples rows from real Qwen tensors (BF16/F32 source), quantizes them
//! with the exact shipped trellis decode functions (offline Viterbi via
//! qwen_llm::trellis_offline, input-side incoherence = column signs +
//! blockwise H128), and compares relative Frobenius error against
//! Lloyd-Max-RHT (TQ-class), Q4_K, Q3_K, IQ3_XXS (ggml, no imatrix).
//! Because the rotation is orthonormal, error measured in rotated space
//! equals error in original space — the metric is exact.
//!
//! Run: cargo run --release -p qwen-llm --example trellis3_real_weight_oracle

use qwen_llm::gguf::GgufFile;
use qwen_llm::trellis_offline::{TrellisCode, encode_group, fwht128_blocks};
use std::time::Instant;

const LM8: [f32; 8] = [
    -1.996684, -1.291398, -0.740341, -0.247508, 0.230106, 0.725222, 1.277503, 1.988943,
];

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn f16r(x: f32) -> f32 {
    half::f16::from_f32(x).to_f32()
}

/// Per-32-block Lloyd-Max-8 with fitted fp16 scale (TQ3_1S-class, 3.5 bpw).
fn lm8_sq_err(rows: &[f32]) -> f64 {
    let mut err = 0f64;
    for blk in rows.chunks(32) {
        let mut s = f16r(
            ((blk.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / blk.len() as f64).sqrt()
                / 1.0295) as f32, // rms of LM8 centroid distribution ~ 1.0295
        );
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            let mut num = 0f64;
            let mut den = 0f64;
            let mut e = 0f64;
            for xx in blk {
                let mut bc = LM8[0];
                let mut bd = f32::INFINITY;
                for &c in &LM8 {
                    let d = (xx - s * c).abs();
                    if d < bd {
                        bd = d;
                        bc = c;
                    }
                }
                num += (*xx as f64) * (bc as f64);
                den += (bc as f64) * (bc as f64);
                let dd = *xx as f64 - s as f64 * bc as f64;
                e += dd * dd;
            }
            best = best.min(e);
            s = f16r(if den > 0.0 { (num / den) as f32 } else { s });
        }
        err += best;
    }
    err
}

fn ggml_sq_err(rows: &[f32], dtype: u32, n_per_row: usize) -> f64 {
    let nrows = rows.len() / n_per_row;
    unsafe {
        llama_cpp_sys_2::ggml_quantize_init(dtype);
        let mut buf = vec![0u8; rows.len() * 2];
        let sz = llama_cpp_sys_2::ggml_quantize_chunk(
            dtype,
            rows.as_ptr(),
            buf.as_mut_ptr() as *mut std::ffi::c_void,
            0,
            nrows as i64,
            n_per_row as i64,
            std::ptr::null(),
        );
        buf.truncate(sz);
        let traits = llama_cpp_sys_2::ggml_get_type_traits(dtype);
        let to_float = (*traits).to_float.expect("to_float");
        let mut back = vec![0f32; rows.len()];
        to_float(
            buf.as_ptr() as *const std::ffi::c_void,
            back.as_mut_ptr(),
            rows.len() as i64,
        );
        rows.iter()
            .zip(back.iter())
            .map(|(a, b)| {
                let d = (*a - *b) as f64;
                d * d
            })
            .sum()
    }
}

fn trellis_sq_err(rot_rows: &[f32], code: &TrellisCode) -> f64 {
    // QWEN_T7_THREADS caps worker threads (co-tenant politeness).
    let n_threads = std::env::var("QWEN_T7_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|v| v.get())
                .unwrap_or(8)
        });
    let groups: Vec<&[f32]> = rot_rows.chunks(256).collect();
    std::thread::scope(|scope| {
        let chunk = groups.len().div_ceil(n_threads);
        let handles: Vec<_> = groups
            .chunks(chunk)
            .map(|gs| {
                scope.spawn(move || gs.iter().map(|g| encode_group(g, code).sq_err).sum::<f64>())
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).sum()
    })
}

fn main() {
    let model = std::env::var("QWEN_T7_MODEL")
        .unwrap_or_else(|_| "/Users/tito/models/Qwen3.5-9B-BF16.gguf".to_string());
    let g = GgufFile::open(&model).expect("open model");
    println!("model: {model}");

    // Candidate classes: 2D weights from one GDN block (blk.0) and one
    // full-attn block (blk.3), n_in divisible by 256, largest first.
    let mut cands: Vec<_> = g
        .tensors
        .iter()
        .filter(|t| {
            (t.name.starts_with("blk.0.") || t.name.starts_with("blk.3."))
                && t.name.ends_with(".weight")
                && t.shape.len() == 2
                && t.shape[0] % 256 == 0
                && t.shape[0] % 128 == 0
        })
        .collect();
    cands.sort_by_key(|t| std::cmp::Reverse(t.n_bytes));
    cands.truncate(6);

    let code_v1 = TrellisCode::v1_maskor();
    let code_v2 = TrellisCode::v2_split_maskor();
    // T8a winner probe: dual-3INST R13 (V=2, ~6.5 ops/weight), built from
    // explicit tables; 0.977x V1 on the synthetic oracle.
    let code_d = {
        const MASK: u32 = 0x8FFF_8FFF;
        const FIXED: u32 = 0x3B60_3B60 & !MASK;
        let mut vx = vec![0f32; 1 << 16];
        let mut vy = vec![0f32; 1 << 16];
        let hv = |bits: u32| {
            (
                half::f16::from_bits((bits & 0xFFFF) as u16).to_f32(),
                half::f16::from_bits((bits >> 16) as u16).to_f32(),
            )
        };
        for st in 0..(1u32 << 16) {
            let h = st.wrapping_mul(89_226_354).wrapping_add(64_248_484);
            let g = h ^ h.rotate_left(13);
            let (ax, ay) = hv((h & MASK) | FIXED);
            let (bx, by) = hv((g & MASK) | FIXED);
            vx[st as usize] = f16r(ax + ay);
            vy[st as usize] = f16r(bx + by);
        }
        TrellisCode::from_tables(2, vx, vy)
    };

    println!(
        "\n{:<28} {:>7} | {:>9} {:>9} {:>9} | {:>9} {:>9} {:>9}",
        "class", "n_in", "t3_v2", "t3_v1", "lm8_rht", "q4_k", "q3_k", "iq3_xxs"
    );
    println!(
        "{:<28} {:>7} | {:>9} {:>9} {:>9} | {:>9} {:>9} {:>9}",
        "(rel Frobenius error)", "", "3.06bpw", "3.06bpw", "3.5bpw", "4.5bpw", "3.44bpw", "3.06bpw"
    );

    let mut v2_beats_q3k = 0usize;
    let mut v2_beats_iq3 = 0usize;
    let mut n_classes = 0usize;
    let mut v2_v1_ratios: Vec<f64> = Vec::new();

    for t in &cands {
        let t0 = Instant::now();
        let n_in = t.shape[0] as usize;
        let n_rows_total = t.shape[1] as usize;
        let groups_per_row = n_in / 256;
        let rows = (1024usize.div_ceil(groups_per_row))
            .clamp(8, 64)
            .min(n_rows_total);

        let full = qwen_llm::codec::dequant_to_f32(t, g.try_slice(t).expect("slice")).expect("dequant");

        // Seeded distinct row sample.
        let mut s = 0x9B_2026u64 ^ (n_in as u64);
        let mut picked = std::collections::BTreeSet::new();
        while picked.len() < rows {
            picked.insert((splitmix64(&mut s) as usize) % n_rows_total);
        }
        let mut orig: Vec<f32> = Vec::with_capacity(rows * n_in);
        for r in &picked {
            orig.extend_from_slice(&full[r * n_in..(r + 1) * n_in]);
        }
        drop(full);

        let norm2: f64 = orig.iter().map(|v| (*v as f64) * (*v as f64)).sum();

        // Input-side incoherence: column signs + H128 blocks, per row.
        let mut sgn = vec![1f32; n_in];
        let mut ss = 0x51_6E5u64 ^ (n_in as u64);
        for v in sgn.iter_mut() {
            if splitmix64(&mut ss) & 1 == 1 {
                *v = -1.0;
            }
        }
        let mut rot = orig.clone();
        for row in rot.chunks_mut(n_in) {
            for (v, sg) in row.iter_mut().zip(&sgn) {
                *v *= *sg;
            }
            fwht128_blocks(row);
        }

        let rel = |e: f64| (e / norm2).sqrt();
        let e_v2 = rel(trellis_sq_err(&rot, &code_v2));
        let e_v1 = rel(trellis_sq_err(&rot, &code_v1));
        let e_d = rel(trellis_sq_err(&rot, &code_d));
        let e_lm = rel(lm8_sq_err(&rot));
        let e_q4 = rel(ggml_sq_err(&orig, 12, n_in)); // Q4_K
        let e_q3 = rel(ggml_sq_err(&orig, 11, n_in)); // Q3_K
        let e_iq3 = rel(ggml_sq_err(&orig, 18, n_in)); // IQ3_XXS

        let label = t.name.trim_end_matches(".weight");
        println!(
            "{label:<28} {n_in:>7} | {e_v2:>9.5} {e_v1:>9.5} {e_lm:>9.5} | {e_q4:>9.5} {e_q3:>9.5} {e_iq3:>9.5} | d3inst {e_d:>9.5}   ({} rows, {:.0}s)",
            rows,
            t0.elapsed().as_secs_f32()
        );

        n_classes += 1;
        if e_v2 <= e_q3 {
            v2_beats_q3k += 1;
        }
        if e_v2 <= e_iq3 {
            v2_beats_iq3 += 1;
        }
        v2_v1_ratios.push(e_v2 / e_v1);
    }

    println!("\n== gate evaluation (docs/bench/2026-07-19-trellis3-real-weight-oracle) ==");
    println!(
        "v2 <= q3_k on {v2_beats_q3k}/{n_classes}; v2 <= iq3_xxs on {v2_beats_iq3}/{n_classes}"
    );
    let pass = v2_beats_q3k * 6 >= 5 * n_classes && v2_beats_iq3 * 6 >= 5 * n_classes;
    let kill = n_classes - v2_beats_q3k >= 3;
    println!(
        "verdict: {}",
        if pass {
            "PASS — 3.06 bpw tier lives on real weights"
        } else if kill {
            "KILL — 3.06 bpw tier loses to Q3_K broadly"
        } else {
            "MIXED — neither pass nor kill condition met"
        }
    );
    let mean_ratio = v2_v1_ratios.iter().sum::<f64>() / v2_v1_ratios.len().max(1) as f64;
    println!(
        "real-weight V2/V1 error ratio per class: {:?} (mean {:.3}; T5 synthetic prior 1.24x MSE => ~1.11x rel-err)",
        v2_v1_ratios
            .iter()
            .map(|r| (r * 1000.0).round() / 1000.0)
            .collect::<Vec<_>>(),
        mean_ratio
    );
}
