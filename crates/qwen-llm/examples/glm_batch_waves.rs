//! MODEL [DEPTH] [QUALITY_STEPS]. Whole-round ABBA first, then output/noise witness.
//! --features perf-ablations; canonical lease required; all heads remain in RAM.
#[cfg(not(feature = "perf-ablations"))]
fn main() {
    panic!("requires perf-ablations");
}
#[cfg(feature = "perf-ablations")]
mod experiment {
    use qwen_llm::{gguf::GgufFile, glm5_next_metal::*, metal::*};
    use serde_json::{Value, json};
    use std::{io::Write, time::Instant};
    fn emit(v: Value) {
        println!("{v}");
        std::io::stdout().flush().unwrap();
    }
    fn metrics(a: &[f32], b: &[f32], target: usize) -> (bool, f64, f64, f64) {
        assert_eq!(a.len(), b.len());
        assert!(a.iter().chain(b).all(|v| v.is_finite()), "nonfinite logits");
        let bits = a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
        if bits {
            return (true, 0., 0., 0.);
        }
        let norm = |v: &[f32]| {
            let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            m + v.iter().map(|v| (*v as f64 - m).exp()).sum::<f64>().ln()
        };
        let (za, zb) = (norm(a), norm(b));
        let kl = a
            .iter()
            .zip(b)
            .map(|(x, y)| {
                let logp = *x as f64 - za;
                logp.exp() * (logp - (*y as f64 - zb))
            })
            .sum::<f64>();
        (
            false,
            (zb - b[target] as f64) - (za - a[target] as f64),
            kl,
            qwen_llm::compare::report_max_abs_diff_f64(a, b),
        )
    }
    pub fn run() -> anyhow::Result<()> {
        let args: Vec<_> = std::env::args().collect();
        let depth = args
            .get(2)
            .map(|v| v.parse())
            .transpose()?
            .unwrap_or(8192usize);
        let quality = args
            .get(3)
            .map(|v| v.parse())
            .transpose()?
            .unwrap_or(128usize);
        let mode = args.get(4).map(String::as_str).unwrap_or("all");
        anyhow::ensure!(matches!(mode, "all" | "timing" | "quality"));
        assert!(std::env::var_os("QWEN_METAL_LEASE_FD").is_some());
        let cohort = args.get(5).map(String::as_str).unwrap_or("distinct");
        anyhow::ensure!(matches!(cohort, "distinct" | "forked"));
        let ctx = MetalContext::new()?;
        let g = GgufFile::open(&args[1])?;
        let w = Glm5NextWeights::load(&ctx, &g)?;
        anyhow::ensure!(
            evaluate_metal_memory_admission(12u64 << 30, 512 << 20, ctx.memory_signals(), true)
                .is_admitted(),
            "normal staging admission"
        );
        let f: Value = serde_json::from_slice(include_bytes!(
            "../../../scripts/reference/glm53/quality-v1.json"
        ))?;
        let docs: Vec<Vec<u32>> = f["documents"]
            .as_array()
            .unwrap()
            .iter()
            .map(|d| {
                d["tokens"]
                    .as_str()
                    .unwrap()
                    .split_whitespace()
                    .map(|t| t.parse().unwrap())
                    .collect()
            })
            .collect();
        let prefix = [0, 8, 16, 9];
        let suffix = [1, 10, 17, 11];
        let token = |r: usize, p: usize| docs[suffix[r]][p % docs[suffix[r]].len()];
        let mut ss: Vec<_> = (0..4)
            .map(|i| {
                Glm5NextSession::with_prefill_rows(
                    &ctx,
                    &w,
                    depth + 512,
                    if i == 0 { 2048 } else { 4 },
                )
                .unwrap()
            })
            .collect();
        let mut batch = Glm5NextBatch::new(&ctx, &w, 4)?;
        let empty = ss[0].capture_snapshot()?;
        let mut banks = Vec::new();
        emit(
            json!({"kind":"header","depth":depth,"quality":quality,"cohort":cohort,"device":ctx.device_facts(),"arithmetic":"causal dependency waves, no arithmetic fusion","source":std::process::Command::new("git").args(["rev-parse","HEAD"]).output().map(|o|String::from_utf8_lossy(&o.stdout).trim().to_string()).ok()}),
        );
        for r in 0..4 {
            if r > 0 && cohort == "forked" {
                ss[0].restore_snapshot(&banks[0])?;
                ss[0].prefill_packed(
                    &ctx,
                    &docs[prefix[r]]
                        .iter()
                        .copied()
                        .cycle()
                        .take(r * 17)
                        .collect::<Vec<_>>(),
                )?;
            } else {
                ss[0].restore_snapshot(&empty)?;
                ss[0].prefill_packed(
                    &ctx,
                    &docs[prefix[r]]
                        .iter()
                        .copied()
                        .cycle()
                        .take(depth + r * 17)
                        .collect::<Vec<_>>(),
                )?;
            }
            banks.push(ss[0].capture_snapshot()?);
            emit(json!({"kind":"prefix","member":r}));
        }
        drop(empty);
        for r in 0..4 {
            ss[r].restore_snapshot(&banks[r])?;
        }
        drop(banks);
        for p in 0..64 {
            with_batch_waves(false, || {
                batch.step(
                    &ctx,
                    &mut ss.iter_mut().collect::<Vec<_>>(),
                    &(0..4).map(|r| token(r, p)).collect::<Vec<_>>(),
                )
            })?;
        }
        let entries: Vec<_> = ss.iter().map(|s| s.capture_snapshot().unwrap()).collect();
        if mode != "quality" {
            for width in [4, 2, 3] {
                for (arm, on) in [false, true, true, false, true, false, false, true]
                    .into_iter()
                    .enumerate()
                {
                    for r in 0..width {
                        ss[r].restore_snapshot(&entries[r])?;
                    }
                    for p in 0..4 {
                        with_batch_waves(on, || {
                            batch.step(
                                &ctx,
                                &mut ss[..width].iter_mut().collect::<Vec<_>>(),
                                &(0..width).map(|r| token(r, 64 + p)).collect::<Vec<_>>(),
                            )
                        })?;
                    }
                    let mut times = Vec::new();
                    for p in 0..16 {
                        let t = Instant::now();
                        let out = with_batch_waves(on, || {
                            batch.step(
                                &ctx,
                                &mut ss[..width].iter_mut().collect::<Vec<_>>(),
                                &(0..width).map(|r| token(r, 68 + p)).collect::<Vec<_>>(),
                            )
                        })?;
                        std::hint::black_box(out);
                        times.push(t.elapsed().as_secs_f64() * 1e3);
                    }
                    emit(json!({"kind":"timing","width":width,"arm":arm,"on":on,"wall_ms":times}));
                }
            }
        }
        let quality = if mode == "timing" { 0 } else { quality };
        let mut reference = Vec::new();
        let mut noise_vectors = 0;
        let mut moved = 0;
        for pass in 0..3 {
            for r in 0..4 {
                ss[r].restore_snapshot(&entries[r])?;
            }
            for p in 0..quality {
                let out = with_batch_waves(pass == 2, || {
                    batch.step(
                        &ctx,
                        &mut ss.iter_mut().collect::<Vec<_>>(),
                        &(0..4).map(|r| token(r, 64 + p)).collect::<Vec<_>>(),
                    )
                })?;
                if pass == 0 {
                    reference.push(out);
                } else {
                    for r in 0..4 {
                        let (bits, dnll, kl, max_abs) =
                            metrics(&reference[p][r], &out[r], token(r, 65 + p) as usize);
                        if !bits && pass == 2 {
                            moved += 1;
                        }
                        emit(
                            json!({"kind":if pass==1{"noise"}else{"quality"},"position":p,"member":r,"bits":bits,"dnll":dnll,"kl":kl,"max_abs":max_abs}),
                        );
                    }
                    if pass == 1 {
                        noise_vectors += out.len();
                    }
                }
            }
        }
        if quality > 0 {
            let mut lifecycle = Vec::new();
            let mut final_state = Vec::new();
            for pass in 0..2 {
                for r in 0..4 {
                    ss[r].restore_snapshot(&entries[r])?;
                }
                for p in 0..32 {
                    let width = [4, 2, 3, 1][p % 4];
                    let mut ordered: Vec<_> = ss.iter_mut().enumerate().collect();
                    ordered.sort_by_key(|(r, _)| (r + 4 - p % 4) % 4);
                    ordered.truncate(width);
                    let inputs: Vec<_> = ordered.iter().map(|(r, _)| token(*r, 64 + p)).collect();
                    let out = with_batch_waves(pass == 1, || {
                        batch.step(
                            &ctx,
                            &mut ordered.into_iter().map(|(_, s)| s).collect::<Vec<_>>(),
                            &inputs,
                        )
                    })?;
                    if pass == 0 {
                        lifecycle.push(out);
                    } else {
                        for r in 0..width {
                            let (bits, dnll, kl, max_abs) =
                                metrics(&lifecycle[p][r], &out[r], inputs[r] as usize);
                            emit(
                                json!({"kind":"lifecycle","phase":"width/order/workspace","round":p,"member":r,"bits":bits,"dnll":dnll,"kl":kl,"max_abs":max_abs}),
                            );
                        }
                    }
                }
                for r in 0..4 {
                    let out = ss[r].prefill_packed(
                        &ctx,
                        &(0..13).map(|p| token(r, 128 + p)).collect::<Vec<_>>(),
                    )?;
                    if pass == 0 {
                        lifecycle.push(vec![out]);
                        final_state.push(ss[r].capture_snapshot()?);
                    } else {
                        let (bits, dnll, kl, max_abs) =
                            metrics(&lifecycle[32 + r][0], &out, token(r, 141) as usize);
                        let state_bits = final_state[r].same_state(&ss[r].capture_snapshot()?);
                        emit(
                            json!({"kind":"lifecycle","phase":"append","member":r,"bits":bits,"state_bits":state_bits,"dnll":dnll,"kl":kl,"max_abs":max_abs}),
                        );
                    }
                }
            }
        }
        emit(
            json!({"kind":"complete","quality_vectors":quality*4,"moved_vectors":moved,"noise_vectors":noise_vectors}),
        );
        Ok(())
    }
}
#[cfg(feature = "perf-ablations")]
fn main() -> anyhow::Result<()> {
    experiment::run()
}
