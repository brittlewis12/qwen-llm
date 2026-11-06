//! Bounded production-allocator append witness. Run under build.lock + Metal lease.
//! Restores prefix checkpoints outside the measured append; no profiling in timings.
#![allow(dead_code)]
#[path = "../src/qwen/prefill_plan.rs"]
mod prefill_plan;

use anyhow::{Result, ensure};
use clap::Parser;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandQueue};
use qwen_llm::{
    metal::*,
    metal_dflash::with_prefill_attn_tensor_flash_override as scoped,
    runtime::{LoadedModel, Runtime},
    tensor::GgmlType,
};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: Option<PathBuf>,
    #[arg(long, default_value = "kill")]
    mode: String,
    #[arg(long, default_value_t = 32768)]
    depth: usize,
    #[arg(long, default_value_t = 512)]
    rows: usize,
    #[arg(long, default_value_t = 2)]
    reps: usize,
    /// Use automatic production policy for the on arm; otherwise force on.
    #[arg(long)]
    automatic: bool,
    #[arg(long, default_value = "code")]
    doc: String,
    #[arg(long, default_value = "2ebe7e57")]
    corpus_rev: String,
}
fn emit(v: serde_json::Value) {
    println!("{v}");
}
fn hash(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
fn ids(n: usize, vocab: u32, mut state: u64) -> Vec<i32> {
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % u64::from(vocab)) as i32
        })
        .collect()
}
fn policy<T>(on: bool, automatic: bool, f: impl FnOnce() -> T) -> T {
    if on && automatic { f() } else { scoped(on, f) }
}
fn thermal(arm: &str, position: &str) -> Result<()> {
    let o = std::process::Command::new("notifyutil")
        .args(["-g", "com.apple.system.thermalpressurelevel"])
        .output()?;
    let raw = String::from_utf8(o.stdout)?;
    emit(json!({"record":"thermal","arm":arm,"where":position,"value":raw.trim()}));
    ensure!(
        o.status.success() && raw.split_whitespace().last() == Some("0"),
        "nonzero thermal pressure; stop"
    );
    Ok(())
}
fn census(rows: Vec<DispatchCensusRow>) -> Vec<serde_json::Value> {
    let mut a = BTreeMap::<(String, u64, u64, u64), usize>::new();
    for r in rows {
        *a.entry((r.kernel, r.grid_width, r.grid_height, r.grid_depth))
            .or_default() += 1;
    }
    a.into_iter()
        .map(|((kernel, x, y, z), count)| json!({"kernel":kernel,"grid":[x,y,z],"count":count}))
        .collect()
}
fn text(rev: &str, path: &str) -> Result<String> {
    let o = std::process::Command::new("git")
        .args(["show", &format!("{rev}:{path}")])
        .output()?;
    ensure!(o.status.success(), "frozen corpus missing");
    Ok(String::from_utf8(o.stdout)?)
}
fn f32s(t: &MetalTensor) -> &[f32] {
    unsafe {
        std::slice::from_raw_parts(
            t.buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(t.offset as usize)
                .cast::<f32>(),
            t.n_elements() as usize,
        )
    }
}
fn readout(
    model: &LoadedModel,
    hidden: &MetalTensor,
    rows: usize,
    endpoint: &[f32],
) -> Result<MetalTensor> {
    let ctx = model.context();
    let mm = model.metal_model();
    let h = mm.arch.hidden_size as usize;
    let v = mm.arch.vocab_size as usize;
    let out = MetalTensor::zeros_f32(ctx, vec![(rows * v) as u64])?;
    let norm = MetalTensor::zeros_f32(ctx, vec![h as u64])?;
    let cmd = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&cmd);
    for r in 0..rows {
        let x = hidden.view_subrange((r * h) as u64, vec![h as u64]);
        let y = out.view_subrange((r * v) as u64, vec![v as u64]);
        encode_rms_norm_mul_f32(
            ctx,
            &enc,
            &x,
            &mm.output_norm,
            &norm,
            qwen_llm::metal_forward::RMS_EPS,
        )?;
        qwen_llm::metal_forward::encode_mat_vec_dispatch(ctx, &enc, &mm.lm_head, &norm, &y, h, v)?;
    }
    enc.end();
    cmd.commit();
    wait_completed(&cmd)?;
    ensure!(
        f32s(&out).iter().all(|v| v.is_finite()),
        "nonfinite readout"
    );
    let peak = qwen_llm::compare::assert_max_abs_diff_f32(
        "native depth readout",
        &f32s(&out)[(rows - 1) * v..],
        endpoint,
    );
    ensure!(peak == 0., "native endpoint mismatch");
    emit(json!({"record":"readout","rows":rows,"max_abs":peak}));
    Ok(out)
}
#[derive(Default)]
struct Score {
    n: usize,
    off: f64,
    on: f64,
    kl: f64,
    max_kl: f64,
    agree: usize,
}
impl Score {
    fn add(&mut self, s: &Self) {
        self.n += s.n;
        self.off += s.off;
        self.on += s.on;
        self.kl += s.kl;
        self.max_kl = self.max_kl.max(s.max_kl);
        self.agree += s.agree;
    }
    fn report(&self) -> serde_json::Value {
        json!({"targets":self.n,"nll_off":self.off/self.n as f64,"nll_on":self.on/self.n as f64,"delta_nll":(self.on-self.off)/self.n as f64,"mean_kl":self.kl/self.n as f64,"max_kl":self.max_kl,"argmax_agree":self.agree})
    }
    fn check(&self) -> Result<()> {
        ensure!(
            self.n > 0
                && ((self.on - self.off) / self.n as f64).abs() <= 0.005
                && self.kl / self.n as f64 <= 0.01,
            "distribution cell failed: {}",
            self.report()
        );
        Ok(())
    }
}
fn score(a: &[f32], b: &[f32], targets: &[i32], v: usize) -> Result<Score> {
    ensure!(
        a.len() == targets.len() * v && a.len() == b.len(),
        "logit extents"
    );
    let mut s = Score::default();
    for (r, &target) in targets.iter().enumerate() {
        let x = &a[r * v..(r + 1) * v];
        let y = &b[r * v..(r + 1) * v];
        ensure!(x.iter().chain(y).all(|x| x.is_finite()), "nonfinite logits");
        let arg = |z: &[f32]| {
            z.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0
        };
        let (ax, ay) = (arg(x), arg(y));
        let (mx, my) = (f64::from(x[ax]), f64::from(y[ay]));
        let zx = x.iter().map(|&x| (f64::from(x) - mx).exp()).sum::<f64>();
        let zy = y.iter().map(|&x| (f64::from(x) - my).exp()).sum::<f64>();
        let (lx, ly) = (mx + zx.ln(), my + zy.ln());
        let kl = x
            .iter()
            .zip(y)
            .map(|(&a, &b)| (f64::from(a) - mx).exp() / zx * (f64::from(a) - f64::from(b)))
            .sum::<f64>()
            + ly
            - lx;
        ensure!(kl.is_finite() && kl >= -1e-7, "invalid KL");
        s.n += 1;
        s.off += lx - f64::from(x[target as usize]);
        s.on += ly - f64::from(y[target as usize]);
        s.kl += kl.max(0.);
        s.max_kl = s.max_kl.max(kl);
        s.agree += usize::from(ax == ay);
    }
    Ok(s)
}
fn main() -> Result<()> {
    let a = Args::parse();
    let runtime = Runtime::metal()?;
    let ctx = runtime.context();
    emit(
        json!({"record":"source","head":env!("QWEN_BUILD_COMMIT"),"device":ctx.device_facts().name,"tensor_tier":format!("{:?}",ctx.tensor_ops_status()),"mode":a.mode,"depth":a.depth,"rows":a.rows,"automatic":a.automatic}),
    );
    if !ctx.tensor_ops_status().is_available() {
        emit(json!({"record":"skip","reason":"tensor tier unavailable"}));
        return Ok(());
    }
    if a.mode == "check-tier" {
        return Ok(());
    }
    let model = runtime.load_model(
        a.model
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("--model required"))?,
    )?;
    ensure!(
        (
            model.arch().n_q_heads,
            model.arch().n_kv_heads,
            model.arch().attn_head_dim
        ) == (16, 2, 256),
        "A3B geometry required"
    );
    let quality = a.mode == "quality";
    let append = if quality { 2048 } else { a.rows };
    let total = a.depth + append;
    let tok = model.tokenizer()?;
    let (prefix_ids, append_ids) = if quality {
        let (p, t) = match a.doc.as_str() {
            "prose" => ("docs/PERF-LOG.md", "docs/APPLE-GPU-OPTIMIZATION.md"),
            "code" => (
                "crates/qwen-llm/src/metal_forward/mod.rs",
                "crates/qwen-llm/src/metal_dflash.rs",
            ),
            _ => anyhow::bail!("unknown doc"),
        };
        let p = text(&a.corpus_rev, p)?;
        let t = text(&a.corpus_rev, t)?;
        let pi = tok.encode(&p, false)?;
        let ti = tok.encode(&t, false)?;
        ensure!(ti.len() > append, "corpus too short");
        let pi = pi.iter().copied().cycle().take(a.depth).collect::<Vec<_>>();
        emit(
            json!({"record":"corpus","doc":a.doc,"rev":a.corpus_rev,"prefix_text_sha256":hash(p.as_bytes()),"target_text_sha256":hash(t.as_bytes()),"prefix_ids_sha256":hash(bytemuck::cast_slice(&pi)),"teacher_ids_sha256":hash(bytemuck::cast_slice(&ti[..append+1]))}),
        );
        (pi, ti[..append + 1].to_vec())
    } else {
        (
            ids(a.depth, model.arch().vocab_size, 1 ^ 0x9e37_79b9_7f4a_7c15),
            ids(append, model.arch().vocab_size, 1),
        )
    };
    let policy_prefix = a.automatic || quality;
    let mut snapshots = Vec::new();
    for on in if policy_prefix {
        vec![false, true]
    } else {
        vec![false]
    } {
        thermal(&format!("prefix-{on}"), "before")?;
        let mut p = prefill_plan::allocate_prefill_request_state(
            &model,
            prefill_plan::PrefillChunkArg::Auto,
            total,
            total,
            true,
        )?;
        policy(on, a.automatic, || {
            model.prefill(&mut p.sequence, &mut p.scratch, &prefix_ids)
        })?;
        snapshots.push(model.prepare_checkpoint_boundary(
            &p.sequence,
            prefix_ids.clone(),
            None,
            None,
            None,
            0,
        )?);
        thermal(&format!("prefix-{on}"), "after")?;
        emit(json!({"record":"prefix","candidate":on,"chunk":p.chunk,"decision":p.decision}));
    }
    let restore = |on: bool| -> Result<_> {
        let mut p = prefill_plan::allocate_prefill_request_state(
            &model,
            prefill_plan::PrefillChunkArg::Auto,
            total,
            total,
            true,
        )?;
        let mut all = prefix_ids.clone();
        all.extend_from_slice(&append_ids[..append]);
        model.restore_prepared_checkpoint(
            &snapshots[usize::from(on && policy_prefix)],
            &mut p.sequence,
            &all,
        )?;
        ensure!(p.sequence.position() == a.depth, "restored position");
        Ok(p)
    };
    if quality {
        let mut off = restore(false)?;
        let mut on = restore(true)?;
        let mut done = 0;
        let mut pooled = Score::default();
        for rows in [512, 512, 1024] {
            let s = objc2::rc::autoreleasepool(|_| -> Result<Score> {
                let mut outputs = Vec::new();
                for (candidate, p) in [(false, &mut off), (true, &mut on)] {
                    let hidden = MetalTensor::zeros_f32(
                        ctx,
                        vec![(rows * model.arch().hidden_size as usize) as u64],
                    )?;
                    dispatch_census_begin();
                    let endpoint = policy(candidate, a.automatic, || {
                        model.prefill_with_hidden_capture(
                            &mut p.sequence,
                            &mut p.scratch,
                            &append_ids[done..done + rows],
                            &[model.arch().n_layer - 1],
                            &hidden,
                        )
                    })?;
                    emit(
                        json!({"record":"dispatches","candidate":candidate,"rows":rows,"position":a.depth+done,"dispatches":census(dispatch_census_take())}),
                    );
                    outputs.push(readout(&model, &hidden, rows, &endpoint)?);
                }
                score(
                    f32s(&outputs[0]),
                    f32s(&outputs[1]),
                    &append_ids[done + 1..done + rows + 1],
                    model.arch().vocab_size as usize,
                )
            })?;
            emit(
                json!({"record":"quality_slice","depth":a.depth,"doc":a.doc,"position":a.depth+done,"rows":rows,"score":s.report()}),
            );
            pooled.add(&s);
            done += rows;
        }
        emit(json!({"record":"quality_cell","depth":a.depth,"doc":a.doc,"score":pooled.report()}));
        pooled.check()?;
        return Ok(());
    }
    if a.mode == "profile" || a.mode == "census" {
        for on in [false, true] {
            std::thread::sleep(std::time::Duration::from_secs(30));
            thermal(&format!("profile-{on}"), "before")?;
            let mut p = restore(on)?;
            dispatch_census_begin();
            let profile = if a.mode == "profile" {
                Some(dispatch_profile_begin(ctx, 8192)?)
            } else {
                None
            };
            policy(on, a.automatic, || {
                model.prefill(&mut p.sequence, &mut p.scratch, &append_ids)
            })?;
            if let Some(g) = profile {
                let r = dispatch_profile_finish(ctx, g)?;
                ensure!(
                    r.overflow == 0 && r.rows.iter().all(|x| x.gpu_ns.is_some()),
                    "incomplete profile"
                );
                emit(json!({"record":"profile","candidate":on,"rows":r.rows}));
            }
            emit(
                json!({"record":"dispatches","candidate":on,"dispatches":census(dispatch_census_take())}),
            );
            thermal(&format!("profile-{on}"), "after")?;
        }
        return Ok(());
    }
    ensure!(a.mode == "kill" || a.mode == "e2e", "unknown mode");
    for (arm, on) in [false, true, true, false].into_iter().enumerate() {
        std::thread::sleep(std::time::Duration::from_secs(30));
        thermal(&format!("{arm}-{on}"), "before")?;
        for sample in 0..=a.reps {
            objc2::rc::autoreleasepool(|_| -> Result<()> {
                let mut p = restore(on)?;
                let chunk = p.chunk;
                let start = Instant::now();
                let logits = policy(on, a.automatic, || {
                    model.prefill(&mut p.sequence, &mut p.scratch, &append_ids)
                })?;
                let ms = start.elapsed().as_secs_f64() * 1e3;
                ensure!(
                    p.sequence.position() == total && logits.iter().all(|x| x.is_finite()),
                    "invalid completed append"
                );
                emit(
                    json!({"record":"sample","arm":arm,"candidate":on,"warmup":sample==0,"sample":sample,"depth":a.depth,"rows":append,"chunk":chunk,"ms":ms,"tok_s":append as f64*1000./ms,"endpoint_sha256":hash(bytemuck::cast_slice(&logits))}),
                );
                Ok(())
            })?;
        }
        thermal(&format!("{arm}-{on}"), "after")?;
    }
    Ok(())
}
