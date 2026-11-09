//! V4.1 independent decode at agent depth. CPU inventory: MODEL inventory.
//! Timing: MODEL DEPTH BATCH STEPS [dump=TEMP_LOGITS]. Run under the canonical
//! Metal lease; build unleased. Corpus regenerates from committed Git blobs.
use anyhow::{Result, ensure};
use qwen_llm::{
    deepseek_v41::{
        forward::{Forward, PrefillExecution, State},
        gguf::{ResidentWeights, resident_storage_plan_with_target},
    },
    gguf::GgufFile,
    metal::{self, MetalContext},
    tokenizer::NativeTokenizer,
};
use std::{
    io::{BufRead, Read, Write},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};
static CANCEL: AtomicBool = AtomicBool::new(false);
extern "C" fn cancel(_: libc::c_int) {
    CANCEL.store(true, Ordering::Relaxed);
}
fn thermal() -> Result<()> {
    let o = std::process::Command::new("notifyutil")
        .args(["-g", "com.apple.system.thermalpressurelevel"])
        .output()?;
    ensure!(
        String::from_utf8(o.stdout)?.split_whitespace().last() == Some("0"),
        "thermal pressure: stop"
    );
    Ok(())
}
fn hashes(v: &[Vec<f32>]) -> Result<Vec<String>> {
    v.iter()
        .map(|x| {
            ensure!(x.iter().all(|v| v.is_finite()), "finite vocabulary");
            let b: Vec<_> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
            Ok(blake3::hash(&b).to_hex().to_string())
        })
        .collect()
}
fn step(
    f: &Forward<'_>,
    states: &mut [State],
    tokens: &[u32],
    batch: bool,
) -> Result<Vec<Vec<f32>>> {
    if batch {
        f.step_batch(
            &mut states.iter_mut().collect::<Vec<_>>(),
            tokens,
            Some(&CANCEL),
        )
        .map_err(anyhow::Error::msg)
    } else {
        states
            .iter_mut()
            .zip(tokens)
            .map(|(s, &t)| f.step(s, t, Some(&CANCEL)).map_err(anyhow::Error::msg))
            .collect()
    }
}
fn main() -> Result<()> {
    unsafe {
        libc::signal(libc::SIGINT, cancel as *const () as libc::sighandler_t);
        libc::signal(libc::SIGTERM, cancel as *const () as libc::sighandler_t);
    }
    let args: Vec<_> = std::env::args().collect();
    if args.get(1).is_some_and(|v| v == "compare") {
        return compare(&args[2], &args[3]);
    }
    thermal()?;
    let g = Arc::new(GgufFile::open(&args[1])?);
    if args.get(2).is_some_and(|s| s == "inventory") {
        let plan = qwen_llm::deepseek_v41::gguf::plan(&g)?;
        println!(
            "{}",
            serde_json::json!({"kind":"inventory", "config":format!("{:?}",plan.config), "kv_owners":plan.schedule.kv_source_layers,"index_layers":plan.schedule.index_source_layers,"ratios":plan.schedule.compression_ratios,"tensors":g.tensors.iter().map(|t|serde_json::json!({"name":t.name,"dtype":format!("{:?}",t.dtype),"shape":t.shape,"bytes":t.n_bytes})).collect::<Vec<_>>()})
        );
        return Ok(());
    }
    let ctx = MetalContext::new()?;
    let storage = resident_storage_plan_with_target(
        &g,
        metal::host_page_size_bytes()?,
        ctx.max_buffer_length(),
        Some(4_294_967_296),
    )?;
    let w = ResidentWeights::load_with_storage_plan(&ctx, g.clone(), &storage)?;
    let f = Forward::from_resident(&ctx, &w, w.token_map.clone(), w.raw_pad)
        .map_err(anyhow::Error::msg)?;
    let tok = NativeTokenizer::from_gguf(&g)?;
    let prose = args.get(6).is_some_and(|v| v == "prose");
    let text = corpus(prose)?;
    let mut repeats = 1;
    let common = loop {
        let x = tok.encode(&text.repeat(repeats), false)?;
        if x.len() > 70000 {
            break x.into_iter().map(|v| v as u32).collect::<Vec<_>>();
        }
        repeats *= 2;
    };
    let streams = (0..4)
        .map(|i| common[1000 + i * 5000..9192 + i * 5000].to_vec())
        .collect::<Vec<_>>();
    ensure!(
        streams.iter().all(|s| s.len() > 4500),
        "distinct natural suffix extent"
    );
    let mut base = f.new_state(98304)?;
    let depth: usize = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(32768);
    ensure!([32768, 65536].contains(&depth), "depth 32K/64K");
    for depth in [depth] {
        thermal()?;
        let target = depth - 4096;
        let p = base.position();
        let t = Instant::now();
        if p == 0 {
            f.prefill_with_execution(
                &mut base,
                &common[..target],
                Some(&CANCEL),
                PrefillExecution::ShrinkingCed,
            )?;
        } else {
            ensure!(
                f.begin_cold_prefill_episode(&mut base, target - p)?,
                "CDF prefix eligible"
            );
            loop {
                if f.cold_prefill_episode_slice(&mut base, &common[p..target], Some(&CANCEL), 4096)?
                    .complete
                {
                    break;
                }
            }
        }
        let snap = f.capture_snapshot(&base)?;
        let mut states = Vec::new();
        for i in 0..4 {
            let mut state = f.new_state(98304)?;
            f.restore_snapshot(&mut state, &snap)?;
            f.prefill(&mut state, &streams[i][..4096 + i * 17], Some(&CANCEL))?;
            states.push(state);
        }
        for j in 0..16 {
            let tokens: Vec<_> = (0..4).map(|i| streams[i][4096 + i * 17 + j]).collect();
            step(&f, &mut states, &tokens, true)?;
        }
        let entries = states
            .iter()
            .map(|s| f.capture_snapshot(s))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        println!(
            "{}",
            serde_json::json!({"kind":"prepared","depth":depth,"actual_position":states[0].position(),"prepare_ms":t.elapsed().as_secs_f64()*1000.,"suffix_rows":4096,"warm_steps":16})
        );
        let batch: usize = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(3);
        ensure!((1..=4).contains(&batch), "batch 1..4");
        for batch in [batch] {
            thermal()?;
            for (s, snap) in states.iter_mut().zip(&entries) {
                f.restore_snapshot(s, snap)?;
            }
            for j in 0..8 {
                let tokens: Vec<_> = (0..batch).map(|i| streams[i][4300 + j]).collect();
                step(&f, &mut states[..batch], &tokens, true)?;
            }
            let period = w
                .schedule
                .compression_ratios
                .iter()
                .copied()
                .max()
                .unwrap_or(1);
            let n = period.max(args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(128));
            ensure!(
                n <= 512 && streams.iter().all(|s| s.len() > 4310 + n),
                "bounded complete compression period"
            );
            let start = Instant::now();
            let mut heads = Vec::with_capacity(n);
            for j in 0..n {
                let tokens: Vec<_> = (0..batch).map(|i| streams[i][4310 + j]).collect();
                heads.push(step(&f, &mut states[..batch], &tokens, true)?);
            }
            let ms = start.elapsed().as_secs_f64() * 1000.;
            let mut digests: Vec<_> = (0..batch).map(|_| blake3::Hasher::new()).collect();
            for head in &heads {
                for (h, digest) in head.iter().zip(&mut digests) {
                    ensure!(h.iter().all(|x| x.is_finite()), "finite full vocabulary");
                    digest.update(bytemuck::cast_slice(h));
                }
            }
            println!(
                "{}",
                serde_json::json!({"kind":"kill","depth":depth,"batch":batch,"rounds":n,"compression_period":period,"vocab":heads[0][0].len(),"ms":ms,"ms_round":ms/n as f64,"aggregate_tps":batch as f64*n as f64*1000./ms,"hashes":digests.iter().map(|d|d.finalize().to_hex().to_string()).collect::<Vec<_>>(),"first_hashes":hashes(&heads[0])?,"last_hashes":hashes(heads.last().unwrap())?})
            );
            if let Some(path) = args.get(5).and_then(|s| s.strip_prefix("dump=")) {
                let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
                writeln!(
                    file,
                    "{}",
                    serde_json::json!({"rounds":n,"batch":batch,"vocab":heads[0][0].len(),"targets":(0..n).map(|j|(0..batch).map(|i|streams[i][4311+j]).collect::<Vec<_>>()).collect::<Vec<_>>()})
                )?;
                for h in &heads {
                    for x in h {
                        file.write_all(bytemuck::cast_slice(x))?;
                    }
                }
            }
            if batch == 3 {
                metal::dispatch_census_begin();
                let tokens: Vec<_> = (0..batch).map(|i| streams[i][4310 + n]).collect();
                step(&f, &mut states[..batch], &tokens, true)?;
                let rows = metal::dispatch_census_take();
                let mut counts = std::collections::BTreeMap::<String, usize>::new();
                for row in &rows {
                    *counts.entry(row.kernel.clone()).or_default() += 1;
                }
                println!(
                    "{}",
                    serde_json::json!({"kind":"dispatch_counts","depth":depth,"batch":batch,"count":rows.len(),"kernels":counts})
                );
            }
            thermal()?;
        }
    }
    Ok(())
}

/// Reproduce the standing's corpus without a private fixture or source archive.
fn corpus(prose: bool) -> Result<String> {
    let mut text = String::new();
    let paths = if prose {
        vec!["docs/SERVE.md"]
    } else {
        vec![
            "crates/qwen-llm/src/deepseek_v41/forward/gpu_step.rs",
            "crates/qwen-llm/src/deepseek_v41/forward/attention_gpu.rs",
            "crates/qwen-llm/src/deepseek_v41/forward/ffn_gpu.rs",
            "crates/qwen-llm/src/glm5_next_metal.rs",
            "crates/qwen-llm/src/qwen4exp_runtime.rs",
        ]
    };
    for path in paths {
        let source = std::process::Command::new("git")
            .args([
                "show",
                &format!("dabb97021e65f192a15057b3f226281cbe1ad81f:{path}"),
            ])
            .output()?;
        ensure!(source.status.success(), "corpus Git blob {path}");
        text.push_str(&format!("\n\n// SOURCE: {path}\n\n"));
        text.push_str(&String::from_utf8(source.stdout)?);
    }
    Ok(text)
}

/// CPU-only paired full-vocabulary comparison. Logits are temporary, regenerable
/// bulk; keep this compact result and remove both input files after comparison.
fn compare_stats(off: &str, on: &str) -> Result<serde_json::Value> {
    let mut a = std::io::BufReader::new(std::fs::File::open(off)?);
    let mut b = std::io::BufReader::new(std::fs::File::open(on)?);
    let header = |r: &mut std::io::BufReader<std::fs::File>| -> Result<serde_json::Value> {
        let mut line = String::new();
        r.read_line(&mut line)?;
        Ok(serde_json::from_str(&line)?)
    };
    let ha = header(&mut a)?;
    let hb = header(&mut b)?;
    ensure!(ha == hb, "paired targets/geometry");
    let vocab = ha["vocab"]
        .as_u64()
        .ok_or_else(|| anyhow::anyhow!("vocabulary"))? as usize;
    let targets = ha["targets"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("targets"))?;
    let mut ba = vec![0u8; vocab * 4];
    let mut bb = vec![0u8; vocab * 4];
    let (mut na, mut nb, mut kl_sum, mut max_kl, mut max_abs) = (0f64, 0f64, 0f64, 0f64, 0f64);
    let (mut n, mut argmax_equal) = (0usize, 0usize);
    for row in targets {
        ensure!(!CANCEL.load(Ordering::Relaxed), "CPU comparison cancelled");
        for target in row
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("target row"))?
        {
            let target = target
                .as_u64()
                .ok_or_else(|| anyhow::anyhow!("target id"))? as usize;
            ensure!(target < vocab, "target in vocabulary");
            a.read_exact(&mut ba)?;
            b.read_exact(&mut bb)?;
            let va: Vec<f64> = ba
                .chunks_exact(4)
                .map(|s| f32::from_le_bytes(s.try_into().unwrap()) as f64)
                .collect();
            let vb: Vec<f64> = bb
                .chunks_exact(4)
                .map(|s| f32::from_le_bytes(s.try_into().unwrap()) as f64)
                .collect();
            ensure!(
                va.iter().chain(&vb).all(|x| x.is_finite()),
                "finite vocabulary"
            );
            let ma = va.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mb = vb.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let (mut sa, mut sb, mut weighted) = (0f64, 0f64, 0f64);
            for (&x, &y) in va.iter().zip(&vb) {
                let e = (x - ma).exp();
                sa += e;
                sb += (y - mb).exp();
                weighted += e * (x - y);
            }
            max_abs = max_abs.max(qwen_llm::compare::assert_max_abs_diff_f64(
                "paired logits",
                &va,
                &vb,
            ));
            let za = ma + sa.ln();
            let zb = mb + sb.ln();
            let kl = weighted / sa - za + zb;
            ensure!(
                kl.is_finite() && kl >= -1e-9,
                "finite nonnegative KL to F64 error"
            );
            na += za - va[target];
            nb += zb - vb[target];
            kl_sum += kl;
            max_kl = max_kl.max(kl);
            n += 1;
            argmax_equal +=
                usize::from(va.iter().position(|&x| x == ma) == vb.iter().position(|&x| x == mb));
        }
    }
    ensure!(
        n > 0 && a.read(&mut [0u8; 1])? == 0 && b.read(&mut [0u8; 1])? == 0,
        "exact paired byte counts"
    );
    let dnll = (nb - na) / n as f64;
    let mean_kl = kl_sum / n as f64;
    Ok(
        serde_json::json!({"kind":"distribution","targets":n,"nll_off":na/n as f64,"nll_on":nb/n as f64,"delta_nll":dnll,"mean_kl":mean_kl,"max_kl":max_kl,"max_abs_logit_diff":max_abs,"argmax_equal":argmax_equal,"qualified":dnll.abs()<=0.005 && mean_kl<=0.01}),
    )
}
fn compare(off: &str, on: &str) -> Result<()> {
    let result = compare_stats(off, on)?;
    println!("{result}");
    ensure!(
        result["qualified"].as_bool() == Some(true),
        "distribution gate"
    );
    Ok(())
}

#[cfg(test)]
mod comparison_tests {
    use super::*;
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    struct Fixture(std::path::PathBuf);
    impl Fixture {
        fn new(values: &[f32], target: usize) -> Self {
            let id = NEXT.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir()
                .join(format!("qwen-v41-quality-{}-{id}.f32", std::process::id()));
            let mut f = std::fs::File::create(&path).unwrap();
            writeln!(
                f,
                "{}",
                serde_json::json!({"rounds":1,"batch":1,"vocab":values.len(),"targets":[[target]]})
            )
            .unwrap();
            f.write_all(bytemuck::cast_slice(values)).unwrap();
            Self(path)
        }
        fn path(&self) -> &str {
            self.0.to_str().unwrap()
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    #[test]
    fn additive_logit_shift_preserves_nll_and_kl() {
        let a = Fixture::new(&[-2., 0.25, 1.125], 1);
        let b = Fixture::new(&[-1.875, 0.375, 1.25], 1);
        let r = compare_stats(a.path(), b.path()).unwrap();
        assert!(r["delta_nll"].as_f64().unwrap().abs() < 1e-12);
        assert!(r["mean_kl"].as_f64().unwrap().abs() < 1e-12);
        assert_eq!(r["qualified"], true);
        assert_eq!(r["max_abs_logit_diff"], 0.125);
    }
    #[test]
    fn unequal_distribution_matches_analytic_kl_and_fails_gate() {
        let a = Fixture::new(&[0., 0., 0.], 0);
        let b = Fixture::new(&[0., 0., 1.], 0);
        let r = compare_stats(a.path(), b.path()).unwrap();
        let expected = ((2. + std::f64::consts::E) / 3.).ln() - 1. / 3.;
        assert!((r["mean_kl"].as_f64().unwrap() - expected).abs() < 1e-12);
        assert_eq!(r["qualified"], false);
    }
    #[test]
    fn nonfinite_or_mismatched_targets_cannot_qualify() {
        let a = Fixture::new(&[0., 1., 2.], 0);
        let b = Fixture::new(&[0., f32::NAN, 2.], 0);
        assert!(compare_stats(a.path(), b.path()).is_err());
        let c = Fixture::new(&[0., 1., 2.], 1);
        assert!(compare_stats(a.path(), c.path()).is_err());
    }
    #[test]
    fn truncated_or_extra_logits_cannot_qualify() {
        let a = Fixture::new(&[0., 1., 2.], 0);
        let b = Fixture::new(&[0., 1., 2.], 0);
        std::fs::OpenOptions::new()
            .write(true)
            .open(&b.0)
            .unwrap()
            .set_len(std::fs::metadata(&b.0).unwrap().len() - 1)
            .unwrap();
        assert!(compare_stats(a.path(), b.path()).is_err());
        let c = Fixture::new(&[0., 1., 2.], 0);
        std::fs::OpenOptions::new()
            .append(true)
            .open(&c.0)
            .unwrap()
            .write_all(&[0])
            .unwrap();
        assert!(compare_stats(a.path(), c.path()).is_err());
    }
}
