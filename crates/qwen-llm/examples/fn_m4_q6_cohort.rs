//! Teacher-forced Q6 row-packing A/B with paired GPU-loop timing.
//! MODEL CODE PROSE DEPTHS(csv) TARGETS DOCS PAIRS.
//! Incumbent = one row per simdgroup (QWEN_Q6K_R1 default); candidate = two.
//! Build with perf-ablations; run under the canonical Metal lease.
use qwen_llm::{
    gguf::GgufFile,
    metal::{MetalContext, diagnostic_q6k_r1},
    qwen4exp::Qwen4ExpConfig,
    qwen4exp_runtime::{Qwen4ExpLoadedModel, Qwen4ExpSessionCapacity},
    sampling::{Sampler, SamplingConfig},
    tokenizer::Tokenizer,
};
use serde_json::json;
use std::{io::Write, process::Command, time::Instant};
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
fn compilers() -> usize {
    Command::new("ps")
        .args(["-axo", "comm="])
        .output()
        .map(|o| {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .filter(|l| {
                    matches!(
                        l.trim().rsplit('/').next().unwrap_or(""),
                        "rustc" | "clang" | "clang++" | "ld" | "metal"
                    )
                })
                .count()
        })
        .unwrap_or(usize::MAX)
}
fn mode_of(arm: &str) -> Option<u8> {
    match arm {
        "incumbent" | "incumbent_repeat" => Some(1),
        "candidate" => Some(2),
        _ => None,
    }
}
fn lse(v: &[f32]) -> f64 {
    let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    m + v
        .iter()
        .map(|&x| (f64::from(x) - m).exp())
        .sum::<f64>()
        .ln()
}
fn argmax(v: &[f32]) -> usize {
    (0..v.len()).fold(0, |b, i| if v[i] > v[b] { i } else { b })
}
fn compare(x: &[f32], y: &[f32], target: usize) -> (f64, f64, f64) {
    assert!(x.iter().chain(y).all(|q| q.is_finite()));
    let (zx, zy) = (lse(x), lse(y));
    let kl = x
        .iter()
        .zip(y)
        .map(|(&p, &q)| (f64::from(p) - zx).exp() * ((f64::from(p) - zx) - (f64::from(q) - zy)))
        .sum::<f64>();
    let max = qwen_llm::compare::report_max_abs_diff_f64(x, y);
    assert!(max.is_finite());
    (
        (zy - f64::from(y[target])) - (zx - f64::from(x[target])),
        kl,
        max,
    )
}
fn fnv(rows: &[Vec<f32>]) -> String {
    let mut h = 0xcbf29ce484222325u64;
    for x in rows.iter().flatten() {
        for b in x.to_bits().to_le_bytes() {
            h = (h ^ u64::from(b)).wrapping_mul(0x100000001b3);
        }
    }
    format!("{h:016x}")
}
fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let depths: Vec<usize> = a[4].split(',').map(str::parse).collect::<Result<_, _>>()?;
    let (targets, docs, pairs): (usize, usize, usize) =
        (a[5].parse()?, a[6].parse()?, a[7].parse()?);
    let ctx = MetalContext::new()?;
    let g = GgufFile::open(&a[1])?;
    let c = Qwen4ExpConfig::from_gguf(&g)?;
    let tok = Tokenizer::from_gguf(&g)?;
    let mut texts = Vec::new();
    for (name, path) in [("code", &a[2]), ("prose", &a[3])] {
        let ids = tok.encode(&std::fs::read_to_string(path)?, false)?;
        texts.push((name, ids.into_iter().map(|t| t as u32).collect::<Vec<_>>()));
    }
    let limit = depths.iter().max().unwrap() + targets + 192;
    let cap = Qwen4ExpSessionCapacity::for_forward_limit(&c, limit)?;
    let mut m = Qwen4ExpLoadedModel::load_with_packed_prefill(&ctx, &g, cap, limit)?;
    let mut r = m.create_runner(&ctx)?;
    r.configure_causal_executor(true);
    let prefix: Vec<u32> = (0..64u32).collect();
    r.prefill(&prefix)?;
    for &depth in &depths {
        let mut last = None;
        for (name, tokens) in &texts {
            let span = depth + targets + 1;
            anyhow::ensure!(tokens.len() >= span * docs, "{name} too short");
            for d in 0..docs {
                let start = d * (tokens.len() - span) / docs.max(1);
                let w = &tokens[start..start + span];
                r.reset()?;
                let t = Instant::now();
                r.prefill(&w[..depth])?;
                let prefill_ms = t.elapsed().as_secs_f64() * 1e3;
                let entry = r.capture_snapshot()?;
                let entry_logits = r.logits()?.to_vec();
                let arms = if d % 2 == 0 {
                    ["incumbent", "candidate", "incumbent_repeat", "production"]
                } else {
                    ["candidate", "incumbent", "production", "incumbent_repeat"]
                };
                let mut rows = std::collections::BTreeMap::new();
                for arm in arms {
                    r.restore_snapshot_with_logits(&entry, &entry_logits)?;
                    let _guard = mode_of(arm).map(diagnostic_q6k_r1);
                    let mut v = Vec::with_capacity(targets);
                    for &t in &w[depth..depth + targets] {
                        v.push(r.forward_token(t)?.to_vec());
                    }
                    rows.insert(arm, v);
                }
                let reference = &rows["incumbent"];
                let nll: f64 = reference
                    .iter()
                    .zip(&w[depth + 1..])
                    .map(|(row, &t)| lse(row) - f64::from(row[t as usize]))
                    .sum();
                let mut out = json!({"kind":"cohort","depth":depth,"cohort":name,"doc":d,"start_token":start,"targets":targets,"prefill_ms":prefill_ms,"incumbent_mean_nll":nll/targets as f64,"order":arms});
                for arm in ["incumbent_repeat", "candidate", "production"] {
                    let (mut dnll, mut kl, mut max, mut flips) = (Vec::new(), 0., 0f64, 0);
                    for ((x, y), &t) in reference.iter().zip(&rows[arm]).zip(&w[depth + 1..]) {
                        let (d, k, mv) = compare(x, y, t as usize);
                        dnll.push(d);
                        kl += k;
                        max = max.max(mv);
                        flips += usize::from(argmax(x) != argmax(y));
                    }
                    out[arm] = json!({"signed_dNLL":dnll.iter().sum::<f64>()/targets as f64,"mean_KL":kl/targets as f64,"max_abs_token_dNLL":dnll.iter().fold(0f64,|m,x|m.max(x.abs())),"argmax_flips":flips,"max_logit_error_diagnostic":max,"per_target_dNLL":dnll});
                }
                for (arm, v) in &rows {
                    out["hash"][*arm] = json!(fnv(v));
                }
                emit(out);
                last = Some((entry, entry_logits));
            }
        }
        let (entry, entry_logits) = last.unwrap();
        let (mut p, mut redo) = (0, 0);
        while p < pairs {
            let waited = Instant::now();
            while thermal() != 0 && waited.elapsed().as_secs() < 120 {
                std::thread::sleep(std::time::Duration::from_secs(2));
            }
            let cool_s = waited.elapsed().as_secs_f64();
            let order = if p % 2 == 0 {
                ["incumbent", "candidate"]
            } else {
                ["candidate", "incumbent"]
            };
            let mut arms = Vec::new();
            for arm in order {
                let (t0, c0) = (thermal(), compilers());
                r.restore_snapshot_with_logits(&entry, &entry_logits)?;
                let _guard = mode_of(arm).map(diagnostic_q6k_r1);
                r.generate_gpu_tokens(
                    &mut Sampler::new(SamplingConfig::default())?,
                    2,
                    &[],
                    |_| true,
                )?;
                r.restore_snapshot_with_logits(&entry, &entry_logits)?;
                let t = r.generate_gpu_tokens(
                    &mut Sampler::new(SamplingConfig::default())?,
                    16,
                    &[],
                    |_| true,
                )?;
                let fwd = t.forwarded_tokens as f64;
                arms.push(json!({"arm":arm,"wall_ms_per_token":t.transition_ms/fwd,"gpu_ms_per_token":(t.gpu_ms-t.first_token_selection_ms)/fwd,"thermal_before":t0,"thermal_after":thermal(),"compilers_before":c0,"compilers_after":compilers()}));
            }
            let hot = arms.iter().any(|x| {
                x["thermal_before"].as_u64() >= Some(2) || x["thermal_after"].as_u64() >= Some(2)
            });
            let rejected = hot && redo < 2 * pairs;
            emit(
                json!({"kind":"pair","depth":depth,"pair":p,"cool_wait_s":cool_s,"arms":arms,"rejected_thermal":rejected}),
            );
            if rejected {
                redo += 1;
            } else {
                p += 1;
            }
        }
    }
    Ok(())
}
