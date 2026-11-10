//! CPU-only NextN acceptance reduction from a native capture and the
//! reference head probe. Requires --features perf-ablations.
//!
//! Reports one-step speculative acceptance (full-vocabulary `1 - TV` and the
//! agent-filtered `sum min(P,Q)`), greedy agreement, and a conditional second
//! step. The capture is an on-policy T=1 / top_p=0.95 trajectory; the head
//! probe emits one logit row per scored position conditioned on the captured
//! token. The second-step `chain` value therefore assumes the first draft
//! followed the captured trajectory; the `on_trajectory` subset only counts
//! rows where the head's greedy first draft actually equals the captured token.
//!
//! ```text
//! cargo run --release -p qwen-llm --features perf-ablations \
//!   --example spec_nextn_accept -- CAPTURE_DIR PROBE_DIR [OUT.json]
//! ```
#[cfg(not(feature = "perf-ablations"))]
fn main() {
    panic!("requires --features perf-ablations");
}
#[cfg(feature = "perf-ablations")]
mod probe {
    use qwen_llm::sampling::{Sampler, SamplingConfig, SamplingUniform};
    use std::{io::Write, path::Path, time::Instant};

    fn read_f32(path: &Path) -> anyhow::Result<Vec<f32>> {
        let bytes = std::fs::read(path)?;
        anyhow::ensure!(bytes.len() % 4 == 0, "f32 extent");
        Ok(bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect())
    }
    fn read_u32(path: &Path) -> anyhow::Result<Vec<u32>> {
        let bytes = std::fs::read(path)?;
        anyhow::ensure!(bytes.len() % 4 == 0, "u32 extent");
        Ok(bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect())
    }
    fn softmax(v: &[f32]) -> Vec<f64> {
        let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let mut out: Vec<f64> = v.iter().map(|&x| (x as f64 - m).exp()).collect();
        let s: f64 = out.iter().sum();
        for x in &mut out {
            *x /= s;
        }
        out
    }
    /// 1 - total variation distance between the two softmax distributions.
    fn tv_acceptance(p: &[f32], q: &[f32]) -> f64 {
        let (p, q) = (softmax(p), softmax(q));
        p.iter().zip(&q).map(|(a, b)| a.min(*b)).sum()
    }
    /// Agent-filtered overlap: sum of min mass over the head's filtered support.
    fn filtered_overlap(
        sampler: &Sampler,
        uniform: &SamplingUniform,
        p: &[f32],
        q: &[f32],
    ) -> anyhow::Result<f64> {
        let (pd, qd) = (
            sampler.diagnose(p, *uniform)?,
            sampler.diagnose(q, *uniform)?,
        );
        let mut mass = vec![0.0f64; p.len()];
        for c in &pd.candidates {
            mass[c.token as usize] = c.weight / pd.total_weight;
        }
        let mut overlap = 0.0;
        for c in &qd.candidates {
            overlap += mass[c.token as usize].min(c.weight / qd.total_weight);
        }
        Ok(overlap.min(1.0))
    }
    fn greedy(s: &Sampler, uniform: &SamplingUniform, v: &[f32]) -> anyhow::Result<u32> {
        Ok(s.diagnose(v, *uniform)?.sampled.token as u32)
    }

    fn median(mut v: Vec<f64>) -> f64 {
        if v.is_empty() {
            return f64::NAN;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    }

    pub fn run() -> anyhow::Result<()> {
        let args: Vec<_> = std::env::args().collect();
        let capture = Path::new(args.get(1).expect("CAPTURE_DIR PROBE_DIR [OUT.json]"));
        let probe = Path::new(&args[2]);
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(capture.join("manifest.json"))?)?;
        let report: serde_json::Value =
            serde_json::from_slice(&std::fs::read(probe.join("report.json"))?)?;
        anyhow::ensure!(
            manifest["status"] == "complete" && report["status"] == "complete",
            "incomplete capture or probe"
        );
        let vocab = manifest["vocab_size"].as_u64().unwrap() as usize;
        let t = manifest["score_count"].as_u64().unwrap() as usize;
        let start = manifest["score_start"].as_u64().unwrap() as usize;
        let tokens = read_u32(&capture.join("tokens.u32"))?;
        let target = read_f32(&capture.join("target-logits.f32"))?;
        let draft = read_f32(&probe.join("draft-logits.f32"))?;
        anyhow::ensure!(
            target.len() == t * vocab && draft.len() == t * vocab,
            "logit extents"
        );
        let sampling = SamplingConfig::glm5_next(42);
        let filtered = Sampler::new(sampling)?;
        let greedy_sampler = Sampler::new(SamplingConfig::default())?;
        let uniform = SamplingUniform::new(0.0)?;

        let row = |i: usize| {
            (
                &target[i * vocab..(i + 1) * vocab],
                &draft[i * vocab..(i + 1) * vocab],
            )
        };
        let mut a1_tv = Vec::new();
        let mut a1_filt = Vec::new();
        let mut agree = Vec::new();
        let mut a2_tv = Vec::new();
        let mut a2_filt = Vec::new();
        let mut a2_on_traj_tv = Vec::new();
        let mut a2_on_traj_filt = Vec::new();
        for i in 0..t {
            let (p, q) = row(i);
            a1_tv.push(tv_acceptance(p, q));
            a1_filt.push(filtered_overlap(&filtered, &uniform, p, q)?);
            let (gp, gq) = (
                greedy(&greedy_sampler, &uniform, p)?,
                greedy(&greedy_sampler, &uniform, q)?,
            );
            agree.push(f64::from(gp == gq));
            if i + 1 < t {
                let (p2, q2) = row(i + 1);
                a2_tv.push(tv_acceptance(p2, q2));
                a2_filt.push(filtered_overlap(&filtered, &uniform, p2, q2)?);
                // Exact-on-trajectory subset: the head's greedy first draft is
                // the captured next token, so q2 conditions on the same token
                // as p2 (for the greedy first-step path).
                if gq == tokens[start + i + 1] {
                    a2_on_traj_tv.push(tv_acceptance(p2, q2));
                    a2_on_traj_filt.push(filtered_overlap(&filtered, &uniform, p2, q2)?);
                }
            }
        }
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
        // Cost of the agent-default draw itself, which the native head step
        // timing excludes: one top-p diagnose on a 154K-vocabulary row.
        let time_diagnose = |logits: &[f32]| -> anyhow::Result<f64> {
            let _ = filtered.diagnose(logits, uniform)?;
            let mut samples = Vec::with_capacity(64);
            for _ in 0..64 {
                let started = Instant::now();
                let _ = filtered.diagnose(logits, uniform)?;
                samples.push(started.elapsed().as_secs_f64() * 1e3);
            }
            Ok(median(samples))
        };
        let sampler_target_ms = time_diagnose(row(0).0)?;
        let sampler_draft_ms = time_diagnose(row(0).1)?;
        let step_ms: Vec<f64> = report["score_steps"]
            .as_array()
            .unwrap()
            .iter()
            .skip(1)
            .map(|s| s["full_head_step_ms"].as_f64().unwrap())
            .collect();
        let out = serde_json::json!({
            "event":"summary",
            "prompt_tokens": manifest["prompt_count"],
            "scored_rows": t,
            "sampling": manifest["sampling"],
            "one_step_tv": mean(&a1_tv),
            "one_step_filtered": mean(&a1_filt),
            "greedy_agreement": mean(&agree),
            "second_step_chain_tv": mean(&a2_tv),
            "second_step_chain_filtered": mean(&a2_filt),
            "second_step_on_trajectory_rows": a2_on_traj_tv.len(),
            "second_step_on_trajectory_tv": mean(&a2_on_traj_tv),
            "second_step_on_trajectory_filtered": mean(&a2_on_traj_filt),
            "chain_product_filtered": mean(&a1_filt) * mean(&a2_filt),
            "head_step_ms_median_excluding_first": median(step_ms.clone()),
            "head_step_ms_mean_excluding_first": mean(&step_ms),
            "head_steps": step_ms.len(),
            "sampler_diagnose_ms_target_median": sampler_target_ms,
            "sampler_diagnose_ms_draft_median": sampler_draft_ms,
            "sampler_diagnose_iters": 64,
        });
        println!("{}", serde_json::to_string_pretty(&out)?);
        if let Some(path) = args.get(3) {
            std::fs::File::create(path)?
                .write_all(serde_json::to_string_pretty(&out)?.as_bytes())?;
        }
        Ok(())
    }
}
#[cfg(feature = "perf-ablations")]
fn main() -> anyhow::Result<()> {
    probe::run()
}
