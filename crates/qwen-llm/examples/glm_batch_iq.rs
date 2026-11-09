//! MODEL [DEPTH] [QUALITY_STEPS] [all|timing|quality] [distinct|forked] [VARIANTS] [WIDTH].
//! Whole B=WIDTH round (shipped causal waves) across GLM batch IQ load-schedule
//! variants, forward and reverse arm orders, thermal per arm; then
//! teacher-forced output witness of each variant against the shipped kernels.
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
    fn thermal() -> i64 {
        let out = std::process::Command::new("notifyutil")
            .args(["-g", "com.apple.system.thermalpressurelevel"])
            .output()
            .expect("notifyutil");
        String::from_utf8_lossy(&out.stdout)
            .split_whitespace()
            .last()
            .and_then(|v| v.parse().ok())
            .unwrap_or(-1)
    }
    fn metrics(a: &[f32], b: &[f32], target: usize) -> (bool, f64, f64, f64) {
        assert_eq!(a.len(), b.len());
        assert!(a.iter().chain(b).all(|v| v.is_finite()), "nonfinite logits");
        if a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()) {
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
        let arg = |i: usize| args.get(i).map(String::as_str);
        let depth: usize = arg(2).map(str::parse).transpose()?.unwrap_or(8192);
        let quality: usize = arg(3).map(str::parse).transpose()?.unwrap_or(32);
        let mode = arg(4).unwrap_or("all");
        anyhow::ensure!(matches!(mode, "all" | "timing" | "quality"));
        let cohort = arg(5).unwrap_or("distinct");
        anyhow::ensure!(matches!(cohort, "distinct" | "forked"));
        let variants: Vec<usize> = match arg(6) {
            Some(v) => v
                .split(',')
                .map(str::parse)
                .collect::<std::result::Result<_, _>>()?,
            None => (0..GLM_BATCH_IQ_VARIANTS.len()).collect(),
        };
        anyhow::ensure!(
            variants.first() == Some(&0),
            "variant list starts with shipped 0"
        );
        anyhow::ensure!(variants.iter().all(|v| *v < GLM_BATCH_IQ_VARIANTS.len()));
        let width: usize = arg(7).map(str::parse).transpose()?.unwrap_or(4);
        anyhow::ensure!((2..=4).contains(&width), "width 2..=4");
        assert!(std::env::var_os("QWEN_METAL_LEASE_FD").is_some());
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
        let mut ss: Vec<_> = (0..width)
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
            json!({"kind":"header","depth":depth,"quality":quality,"width":width,"cohort":cohort,"variants":variants.iter().map(|v|GLM_BATCH_IQ_VARIANTS[*v]).collect::<Vec<_>>(),"device":ctx.device_facts(),"source":std::process::Command::new("git").args(["rev-parse","HEAD"]).output().map(|o|String::from_utf8_lossy(&o.stdout).trim().to_string()).ok()}),
        );
        for r in 0..width {
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
        }
        drop(empty);
        for r in 0..width {
            ss[r].restore_snapshot(&banks[r])?;
        }
        drop(banks);
        let mut step = |ss: &mut [Glm5NextSession], v: usize, p: usize| {
            with_glm_batch_iq(v, || {
                batch.step(
                    &ctx,
                    &mut ss.iter_mut().collect::<Vec<_>>(),
                    &(0..width).map(|r| token(r, p)).collect::<Vec<_>>(),
                )
            })
        };
        for p in 0..64 {
            step(&mut ss, 0, p)?;
        }
        let entries: Vec<_> = ss.iter().map(|s| s.capture_snapshot().unwrap()).collect();
        if mode != "quality" {
            let reverse: Vec<_> = variants.iter().rev().copied().collect();
            let order: Vec<usize> = [&variants, &reverse, &reverse, &variants]
                .into_iter()
                .flatten()
                .copied()
                .collect();
            for (arm, v) in order.into_iter().enumerate() {
                for r in 0..width {
                    ss[r].restore_snapshot(&entries[r])?;
                }
                let before = thermal();
                for p in 0..4 {
                    step(&mut ss, v, 64 + p)?;
                }
                let mut times = Vec::new();
                for p in 0..16 {
                    let t = Instant::now();
                    std::hint::black_box(step(&mut ss, v, 68 + p)?);
                    times.push(t.elapsed().as_secs_f64() * 1e3);
                }
                emit(
                    json!({"kind":"timing","width":width,"arm":arm,"variant":GLM_BATCH_IQ_VARIANTS[v],"wall_ms":times,"thermal":[before,thermal()]}),
                );
            }
        }
        let quality = if mode == "timing" { 0 } else { quality };
        let mut reference = Vec::new();
        let mut final_state = Vec::new();
        for (index, pass) in std::iter::once(0)
            .chain(variants.iter().copied())
            .enumerate()
        {
            if quality == 0 {
                break;
            }
            for r in 0..width {
                ss[r].restore_snapshot(&entries[r])?;
            }
            let first = index == 0;
            for p in 0..quality {
                let out = step(&mut ss, pass, 64 + p)?;
                if first {
                    reference.push(out);
                    continue;
                }
                for r in 0..width {
                    let (bits, dnll, kl, max_abs) =
                        metrics(&reference[p][r], &out[r], token(r, 65 + p) as usize);
                    emit(
                        json!({"kind":if pass==0{"noise"}else{"quality"},"variant":GLM_BATCH_IQ_VARIANTS[pass],"position":p,"member":r,"bits":bits,"dnll":dnll,"kl":kl,"max_abs":max_abs}),
                    );
                }
            }
            let finals: Vec<_> = ss.iter().map(|s| s.capture_snapshot().unwrap()).collect();
            if first {
                final_state = finals;
            } else {
                let state_bits: Vec<_> = final_state
                    .iter()
                    .zip(&finals)
                    .map(|(a, b)| a.same_state(b))
                    .collect();
                emit(
                    json!({"kind":"state","variant":GLM_BATCH_IQ_VARIANTS[pass],"index":index,"state_bits":state_bits}),
                );
            }
        }
        emit(json!({"kind":"complete","quality_vectors":quality*width}));
        Ok(())
    }
}
#[cfg(feature = "perf-ablations")]
fn main() -> anyhow::Result<()> {
    experiment::run()
}
