//! T9 Stage A arms runner (docs/bench/2026-07-20-trellis3-t9-ldlq-pilot/).
//!
//! Arms over the 6 0.8B FFN classes (blk.{0,3} x gate/up/down):
//!   A0: T7a-recipe baseline — input-side incoherence (fixed splitmix
//!       signs + block-H128) + plain per-span V1 encode.
//!   A1: two-sided RHT (per-seed signs both sides) + plain encode.
//!   A2: T7a input basis + BlockLDLQ (H rotated into the same basis).
//!   A3: two-sided RHT + BlockLDLQ (H rotated with the seed's input side).
//! DECLARED AMENDMENTS vs the frozen prereg text (recorded before any
//! gate was evaluated; see close notes): (1) A0/A2 use the T7a
//! input-side incoherence recipe — the prereg's literal "A2 ... H in
//! original basis" would have confounded LDLQ's increment with
//! REMOVING the tier's existing incoherence step; (2) T7a-style seeded
//! ROW SAMPLING (~1024 spans/case) — the first cut ran full tensors
//! and measured 448 s/arm-tensor (4 Viterbi passes/span; ~6 h for the
//! matrix), so the T7a sampling precedent is restored. Rows are the
//! independent unit — LDLQ span chains within sampled rows are intact.
//!
//! Metrics per arm-tensor: plain rel-Frobenius P (orthogonal-invariant,
//! scored in the encode domain) and held-out r_H via test-Gram traces:
//! r_H^2 = tr(E Ht E^T) / tr(W Ht W^T), output-side rotation cancels in
//! the trace; Ht is rotated into the arm's input basis in f64.
//!
//! Env: T9_DATA (capture dir), T9_MODEL, T9_OUT, T9_SEEDS ("11,22,33"),
//! T9_THREADS (8), T9_NSUB (1), T9_DAMP (0.01).

use qwen_llm::gguf::GgufFile;
use qwen_llm::trellis_ldlq::{
    LdlqRowResult, SPAN, block_unit_lower_a, cholesky_lower, ldlq_quantize_row, rel_frobenius,
    rht_sign_vec, rotate_hessian_input_f64,
};
use qwen_llm::trellis_offline::{TrellisCode, fwht128_blocks};
use std::io::Write;
use std::time::Instant;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// T7a's exact input-side sign recipe (trellis3_real_weight_oracle.rs).
fn t7a_signs(n_in: usize) -> Vec<f32> {
    let mut sgn = vec![1f32; n_in];
    let mut ss = 0x51_6E5u64 ^ (n_in as u64);
    for v in sgn.iter_mut() {
        if splitmix64(&mut ss) & 1 == 1 {
            *v = -1.0;
        }
    }
    sgn
}

fn env_or(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn read_f64_file(path: &str, n: usize) -> Vec<f64> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    assert_eq!(bytes.len(), n * 8, "{path} size");
    bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn read_f32_file(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// tr(M Ht M^T) threaded over rows.
fn trace_m_h_mt(m: &[f32], n_out: usize, d: usize, ht: &[f64], n_threads: usize) -> f64 {
    let rows: Vec<&[f32]> = m.chunks(d).collect();
    assert_eq!(rows.len(), n_out);
    let chunk = n_out.div_ceil(n_threads.max(1));
    std::thread::scope(|scope| {
        let handles: Vec<_> = rows
            .chunks(chunk)
            .map(|rs| {
                scope.spawn(move || {
                    let mut acc = 0f64;
                    let mut hx = vec![0f64; d];
                    for r in rs {
                        for (j, s) in hx.iter_mut().enumerate() {
                            let mut v = 0f64;
                            let hrow = &ht[j * d..(j + 1) * d];
                            for (k, x) in r.iter().enumerate() {
                                v += hrow[k] * *x as f64;
                            }
                            *s = v;
                        }
                        let mut e = 0f64;
                        for (j, x) in r.iter().enumerate() {
                            e += *x as f64 * hx[j];
                        }
                        acc += e;
                    }
                    acc
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).sum()
    })
}

struct ArmScore {
    p: f64,
    r_h: f64,
    encode_s: f64,
    traces: Option<Vec<(f64, f64)>>, // mean (rms, kurt) per reverse span idx
}

#[allow(clippy::too_many_arguments)]
fn run_arm(
    w_basis: &[f32], // rows already in the arm's encode basis
    n_out: usize,
    d: usize,
    a: Option<&[f64]>,
    ht_rot: &[f64], // test Gram in the same input basis
    code: &TrellisCode,
    n_sub: usize,
    n_threads: usize,
) -> ArmScore {
    let t0 = Instant::now();
    let mut w_hat = vec![0f32; n_out * d];
    let n_spans = d / SPAN;
    let mut trace_acc = vec![(0f64, 0f64); n_spans];
    {
        let rows_in: Vec<&[f32]> = w_basis.chunks(d).collect();
        let rows_out: Vec<&mut [f32]> = w_hat.chunks_mut(d).collect();
        let mut pairs: Vec<(usize, (&[f32], &mut [f32]))> =
            rows_in.into_iter().zip(rows_out).enumerate().collect();
        let chunk = n_out.div_ceil(n_threads.max(1));
        let results: Vec<Vec<(usize, LdlqRowResult)>> = std::thread::scope(|scope| {
            let handles: Vec<_> = pairs
                .chunks_mut(chunk)
                .map(|prs| {
                    scope.spawn(move || {
                        let mut out = Vec::with_capacity(prs.len());
                        for (ri, (rin, rout)) in prs.iter_mut() {
                            let res = ldlq_quantize_row(rin, a, d, code, n_sub);
                            rout.copy_from_slice(&res.w_hat);
                            out.push((*ri, res));
                        }
                        out
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for group in results {
            for (_ri, res) in group {
                for (si, (rms, kurt)) in res.span_traces.iter().enumerate() {
                    trace_acc[si].0 += rms;
                    trace_acc[si].1 += kurt;
                }
            }
        }
    }
    for t in trace_acc.iter_mut() {
        t.0 /= n_out as f64;
        t.1 /= n_out as f64;
    }
    let p = rel_frobenius(w_basis, &w_hat);
    let mut e = vec![0f32; n_out * d];
    for (i, v) in e.iter_mut().enumerate() {
        *v = w_basis[i] - w_hat[i];
    }
    let num = trace_m_h_mt(&e, n_out, d, ht_rot, n_threads);
    let den = trace_m_h_mt(w_basis, n_out, d, ht_rot, n_threads);
    ArmScore {
        p,
        r_h: (num / den.max(1e-300)).sqrt(),
        encode_s: t0.elapsed().as_secs_f64(),
        traces: if a.is_some() { Some(trace_acc) } else { None },
    }
}

/// Apply input-side basis to weight rows: signs + fwht per row.
fn rotate_rows_input(w: &mut [f32], d: usize, signs: &[f32]) {
    for row in w.chunks_mut(d) {
        for (v, s) in row.iter_mut().zip(signs) {
            *v *= *s;
        }
        fwht128_blocks(row);
    }
}

/// Apply output-side basis: signs + fwht along columns.
fn rotate_cols_output(w: &mut [f32], n_out: usize, d: usize, signs: &[f32]) {
    let mut col = vec![0f32; n_out];
    for j in 0..d {
        for (i, c) in col.iter_mut().enumerate() {
            *c = w[i * d + j];
        }
        for (v, s) in col.iter_mut().zip(signs) {
            *v *= *s;
        }
        fwht128_blocks(&mut col);
        for (i, c) in col.iter().enumerate() {
            w[i * d + j] = *c;
        }
    }
}

fn main() {
    let data = std::env::var("T9_DATA").expect("T9_DATA (capture dir)");
    let model = std::env::var("T9_MODEL")
        .unwrap_or_else(|_| "/Users/tito/models/Qwen3.5-0.8B.F32.gguf".into());
    let out_dir = std::env::var("T9_OUT").expect("T9_OUT");
    let seeds: Vec<u64> = std::env::var("T9_SEEDS")
        .unwrap_or_else(|_| "11,22,33".into())
        .split(',')
        .map(|s| s.trim().parse().expect("seed"))
        .collect();
    let n_threads = env_or("T9_THREADS", 8);
    let n_sub = env_or("T9_NSUB", 1);
    let damp: f64 = std::env::var("T9_DAMP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.01);
    std::fs::create_dir_all(&out_dir).expect("mkdir");

    let g = GgufFile::open(&model).expect("open model");
    let code = TrellisCode::v1_maskor();

    // (tensor name, space, d_in, blk)
    let classes = [
        ("blk.0.ffn_gate.weight", "blk0_h", 1024usize),
        ("blk.0.ffn_up.weight", "blk0_h", 1024),
        ("blk.0.ffn_down.weight", "blk0_inner", 3584),
        ("blk.3.ffn_gate.weight", "blk3_h", 1024),
        ("blk.3.ffn_up.weight", "blk3_h", 1024),
        ("blk.3.ffn_down.weight", "blk3_inner", 3584),
    ];

    // Load weights (F32 raw) with T7a-style seeded distinct-row sampling
    // targeting ~T9_SPAN_BUDGET spans per case.
    let span_budget = env_or("T9_SPAN_BUDGET", 1024);
    let mut weights: Vec<(usize, Vec<f32>, usize, usize)> = Vec::new(); // (class idx, w, n_rows, d)
    for (ci, (name, _space, d)) in classes.iter().enumerate() {
        let t = g
            .tensors
            .iter()
            .find(|t| t.name == *name)
            .unwrap_or_else(|| panic!("tensor {name}"));
        assert_eq!(t.shape[0] as usize, *d, "{name} input dim");
        let n_out = t.shape[1] as usize;
        let full =
            qwen_llm::codec::dequant_to_f32(t, g.try_slice(t).expect("slice")).expect("dequant");
        let spans_per_row = *d / SPAN;
        // Round the row sample up to a multiple of 128 so the
        // output-side block-H128 transform is well-defined on the
        // sampled submatrix (A1/A3 rotate the sampled rows as a unit —
        // a legitimate two-sided RHT of the submatrix we quantize and
        // score).
        let rows = span_budget
            .div_ceil(spans_per_row)
            .div_ceil(128)
            .saturating_mul(128)
            .min(n_out);
        let mut s = 0x9B_2026u64 ^ (*d as u64) ^ ((ci as u64) << 32);
        let mut picked = std::collections::BTreeSet::new();
        while picked.len() < rows {
            picked.insert((splitmix64(&mut s) as usize) % n_out);
        }
        let mut w: Vec<f32> = Vec::with_capacity(rows * *d);
        for r in &picked {
            w.extend_from_slice(&full[r * *d..(r + 1) * *d]);
        }
        eprintln!(
            "[t9-stage-a] {name}: sampled {rows}/{n_out} rows ({} spans)",
            rows * spans_per_row
        );
        weights.push((ci, w, rows, *d));
    }

    // Load train Grams + build test Grams per space.
    let spaces = ["blk0_h", "blk0_inner", "blk3_h", "blk3_inner"];
    let space_dim = |s: &str| -> usize { if s.ends_with("_h") { 1024 } else { 3584 } };
    let mut gram_train: Vec<Vec<f64>> = Vec::new();
    let mut gram_test: Vec<Vec<f64>> = Vec::new();
    for s in &spaces {
        let d = space_dim(s);
        gram_train.push(read_f64_file(&format!("{data}/gram_{s}.f64"), d * d));
        let xs = read_f32_file(&format!("{data}/test_{s}.f32"));
        assert_eq!(xs.len() % d, 0);
        let mut acc = qwen_llm::trellis_ldlq::GramAccumulator::new(d);
        acc.add_batch(&xs, n_threads);
        eprintln!(
            "[t9-stage-a] test gram {s}: d={d} n={} tokens",
            xs.len() / d
        );
        gram_test.push(acc.h);
    }
    let space_idx = |s: &str| spaces.iter().position(|x| x == &s).unwrap();

    // Basis contexts: t7a (fixed signs) + per-seed.
    // For each basis we need, per space: rotated damped-LDL A (train) and
    // rotated test Gram.
    #[derive(Clone)]
    struct BasisCtx {
        label: String,
        in_signs: Vec<Vec<f32>>, // per space
        a_mats: Vec<Option<Vec<f64>>>,
        ht_rot: Vec<Vec<f64>>,
        out_signs_seed: Option<u64>,
    }

    let build_basis =
        |label: &str, signs_for: &dyn Fn(usize) -> Vec<f32>, out_seed: Option<u64>| {
            let t0 = Instant::now();
            let mut in_signs = Vec::new();
            let mut a_mats = Vec::new();
            let mut ht_rot = Vec::new();
            for (si, s) in spaces.iter().enumerate() {
                let d = space_dim(s);
                let signs = signs_for(d);
                // Rotated damped train H -> Cholesky -> block A.
                let mut h = {
                    let acc = qwen_llm::trellis_ldlq::GramAccumulator {
                        d,
                        n_samples: 0,
                        h: gram_train[si].clone(),
                    };
                    acc.damped(damp)
                };
                rotate_hessian_input_f64(&mut h, d, &signs);
                match cholesky_lower(&mut h, d) {
                    Ok(()) => a_mats.push(Some(block_unit_lower_a(&h, d))),
                    Err(p) => {
                        eprintln!("[t9-stage-a] WARN cholesky failed {label}/{s} pivot {p}");
                        a_mats.push(None);
                    }
                }
                let mut ht = gram_test[si].clone();
                rotate_hessian_input_f64(&mut ht, d, &signs);
                ht_rot.push(ht);
                in_signs.push(signs);
            }
            eprintln!(
                "[t9-stage-a] basis {label}: LDL+rotations {:.1}s",
                t0.elapsed().as_secs_f64()
            );
            BasisCtx {
                label: label.to_string(),
                in_signs,
                a_mats,
                ht_rot,
                out_signs_seed: out_seed,
            }
        };

    let basis_t7a = build_basis("t7a", &t7a_signs, None);
    let basis_seeds: Vec<BasisCtx> = seeds
        .iter()
        .map(|&sd| {
            build_basis(
                &format!("seed{sd}"),
                &move |d| rht_sign_vec(d, sd ^ 0x1157),
                Some(sd),
            )
        })
        .collect();

    // Run arms.
    let mut json_rows: Vec<String> = Vec::new();
    let mut results: Vec<(String, String, u64, f64, f64)> = Vec::new(); // (arm, class, seed, P, r_H)
    let mut trace_lines: Vec<String> = Vec::new();

    let mut run_case = |arm: &str,
                        seed: u64,
                        basis: &BasisCtx,
                        use_ldlq: bool,
                        results: &mut Vec<(String, String, u64, f64, f64)>,
                        json_rows: &mut Vec<String>,
                        trace_lines: &mut Vec<String>| {
        for (ci, w, n_out, d) in weights.iter() {
            let (name, space, _) = classes[*ci];
            let si = space_idx(space);
            let mut wb = w.clone();
            rotate_rows_input(&mut wb, *d, &basis.in_signs[si]);
            if let Some(osd) = basis.out_signs_seed {
                let osigns = rht_sign_vec(*n_out, osd ^ 0x2263);
                rotate_cols_output(&mut wb, *n_out, *d, &osigns);
            }
            let a = if use_ldlq {
                basis.a_mats[si].as_deref()
            } else {
                None
            };
            if use_ldlq && a.is_none() {
                eprintln!("[t9-stage-a] SKIP {arm}/{name}: no LDL factor");
                continue;
            }
            let score = run_arm(
                &wb,
                *n_out,
                *d,
                a,
                &basis.ht_rot[si],
                &code,
                n_sub,
                n_threads,
            );
            eprintln!(
                "[t9-stage-a] {arm} seed={seed} {name:<24} P={:.5} r_H={:.5} ({:.0}s)",
                score.p, score.r_h, score.encode_s
            );
            results.push((arm.to_string(), name.to_string(), seed, score.p, score.r_h));
            json_rows.push(format!(
                "{{\"arm\":{arm:?},\"class\":{name:?},\"seed\":{seed},\"basis\":{:?},\
                 \"P\":{:.6},\"r_H\":{:.6},\"encode_s\":{:.1}}}",
                basis.label, score.p, score.r_h, score.encode_s
            ));
            if let Some(tr) = score.traces {
                trace_lines.push(format!(
                    "{arm} seed={seed} {name} spans(rms,kurt)={:?}",
                    tr.iter()
                        .map(|(r, k)| ((r * 1e4).round() / 1e4, (k * 100.0).round() / 100.0))
                        .collect::<Vec<_>>()
                ));
            }
        }
    };

    // Anchor scoring (additive measurement, declared at close: the
    // prereg's anchors were plain-only from T7a; the LDLQ arms trade P
    // for r_H, so the anchors need r_H on the SAME sampled rows to make
    // the scorecard decision-grade). Anchors quantize the ORIGINAL
    // (unrotated) rows; their E lives in the original basis, so r_H
    // uses the UNROTATED test Gram — cross-basis r_H comparison is
    // valid because both numerator and denominator are orthogonal-
    // invariant norms of the same underlying operators.
    for (dtype, label) in [(12u32, "q4_k"), (11u32, "q3_k")] {
        for (ci, w, n_rows, d) in weights.iter() {
            let (name, space, _) = classes[*ci];
            let si = space_idx(space);
            let mut back = vec![0f32; w.len()];
            unsafe {
                llama_cpp_sys_2::ggml_quantize_init(dtype);
                let mut buf = vec![0u8; w.len() * 2];
                let sz = llama_cpp_sys_2::ggml_quantize_chunk(
                    dtype,
                    w.as_ptr(),
                    buf.as_mut_ptr() as *mut std::ffi::c_void,
                    0,
                    *n_rows as i64,
                    *d as i64,
                    std::ptr::null(),
                );
                buf.truncate(sz);
                let traits = llama_cpp_sys_2::ggml_get_type_traits(dtype);
                let to_float = (*traits).to_float.expect("to_float");
                to_float(
                    buf.as_ptr() as *const std::ffi::c_void,
                    back.as_mut_ptr(),
                    w.len() as i64,
                );
            }
            let p = rel_frobenius(w, &back);
            let mut e = vec![0f32; w.len()];
            for (i, v) in e.iter_mut().enumerate() {
                *v = w[i] - back[i];
            }
            // Unrotated test Gram: damp=0 copy of gram_test.
            let ht = &gram_test[si];
            let num = trace_m_h_mt(&e, *n_rows, *d, ht, n_threads);
            let den = trace_m_h_mt(w, *n_rows, *d, ht, n_threads);
            let r_h = (num / den.max(1e-300)).sqrt();
            eprintln!("[t9-stage-a] ANCHOR {label} {name:<24} P={p:.5} r_H={r_h:.5}");
            json_rows.push(format!(
                "{{\"arm\":\"anchor_{label}\",\"class\":{name:?},\"seed\":0,\
                 \"P\":{p:.6},\"r_H\":{r_h:.6}}}"
            ));
            results.push((format!("anchor_{label}"), name.to_string(), 0, p, r_h));
        }
    }

    run_case(
        "A0",
        0,
        &basis_t7a,
        false,
        &mut results,
        &mut json_rows,
        &mut trace_lines,
    );
    run_case(
        "A2",
        0,
        &basis_t7a,
        true,
        &mut results,
        &mut json_rows,
        &mut trace_lines,
    );
    for (bi, basis) in basis_seeds.iter().enumerate() {
        run_case(
            "A1",
            seeds[bi],
            basis,
            false,
            &mut results,
            &mut json_rows,
            &mut trace_lines,
        );
        run_case(
            "A3",
            seeds[bi],
            basis,
            true,
            &mut results,
            &mut json_rows,
            &mut trace_lines,
        );
    }

    // Aggregation: per arm per class, median across seeds; then macro-mean.
    let agg =
        |arm: &str, metric: &dyn Fn(&(String, String, u64, f64, f64)) -> f64| -> (f64, Vec<f64>) {
            let mut per_class = Vec::new();
            for (name, _, _) in classes.iter() {
                let mut vals: Vec<f64> = results
                    .iter()
                    .filter(|r| r.0 == arm && r.1 == *name)
                    .map(metric)
                    .collect();
                vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
                if !vals.is_empty() {
                    per_class.push(vals[vals.len() / 2]);
                }
            }
            let mean = per_class.iter().sum::<f64>() / per_class.len().max(1) as f64;
            (mean, per_class)
        };

    println!("\n== T9 Stage A summary (seed-median then macro-mean over 6 classes) ==");
    let mut summary = Vec::new();
    for arm in ["anchor_q4_k", "anchor_q3_k", "A0", "A1", "A2", "A3"] {
        let (p_mean, p_per) = agg(arm, &|r| r.3);
        let (rh_mean, rh_per) = agg(arm, &|r| r.4);
        println!("{arm}: P={p_mean:.5} r_H={rh_mean:.5}  per-class P={p_per:?} r_H={rh_per:?}");
        summary.push((arm, p_mean, rh_mean, rh_per));
    }
    let idx = |arm: &str| summary.iter().position(|s| s.0 == arm).expect("arm row");
    let a0_rh = summary[idx("A0")].2;
    let a0_p = summary[idx("A0")].1;
    let a3_rh = summary[idx("A3")].2;
    let a3_p = summary[idx("A3")].1;
    let delta_h = 1.0 - a3_rh / a0_rh;
    let a0_rh_per = summary[idx("A0")].3.clone();
    let a3_rh_per = summary[idx("A3")].3.clone();
    let improved = a0_rh_per
        .iter()
        .zip(&a3_rh_per)
        .filter(|(a0, a3)| a3 < a0)
        .count();
    println!(
        "\nStage A -> B gate: Delta_H(A3) = {:.1}% (need >= 20%), P(A3) {:.5} vs P(A0) {:.5} \
         (need no worse than +2%), r_H improved on {improved}/6 (need >= 5)",
        delta_h * 100.0,
        a3_p,
        a0_p
    );
    let gate = delta_h >= 0.20 && a3_p <= a0_p * 1.02 && improved >= 5;
    println!(
        "VERDICT: {}",
        if gate {
            "STAGE-A PASS -> Stage B authorized"
        } else {
            "STAGE-A FAIL -> consult prereg (INCONCLUSIVE/KILL ladder)"
        }
    );

    let mut jf =
        std::fs::File::create(format!("{out_dir}/stage_a_results.jsonl")).expect("json out");
    for r in &json_rows {
        writeln!(jf, "{r}").unwrap();
    }
    std::fs::write(
        format!("{out_dir}/stage_a_traces.txt"),
        trace_lines.join("\n"),
    )
    .unwrap();
    eprintln!("[t9-stage-a] wrote {out_dir}/stage_a_results.jsonl and traces");
}
