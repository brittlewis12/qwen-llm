//! V4.1 independent decode at agent depth. CPU inventory: MODEL inventory.
//! Timing: MODEL DEPTH BATCH STEPS [dump=TEMP_LOGITS]. Run under the canonical
//! Metal lease; builds follow the host's campaign policy. Corpus regenerates
//! from committed Git blobs. `nodes` exports isolated node times and span edges.
use anyhow::{Result, ensure};
use qwen_llm::{
    deepseek_v41::{
        forward::{Forward, PrefillExecution, State},
        gguf::{ResidentWeights, resident_storage_plan_with_target},
    },
    gguf::GgufFile,
    metal::{self, GpuThreadQos, MetalContext},
    pid_metrics::{PidDelta, PidSnapshot},
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
fn thermal_level() -> Result<u32> {
    let o = std::process::Command::new("notifyutil")
        .args(["-g", "com.apple.system.thermalpressurelevel"])
        .output()?;
    let level = String::from_utf8(o.stdout)?
        .split_whitespace()
        .last()
        .unwrap()
        .parse()?;
    ensure!(level < 2, "thermal pressure >=2: redo");
    Ok(level)
}
fn thermal() -> Result<()> {
    thermal_level().map(|_| ())
}

struct Load {
    stop: Arc<AtomicBool>,
    threads: Vec<std::thread::JoinHandle<()>>,
}
impl Load {
    fn start(n: usize) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let threads = (0..n)
            .map(|_| {
                let stop = stop.clone();
                std::thread::spawn(move || {
                    while !stop.load(Ordering::Relaxed) {
                        std::hint::spin_loop();
                    }
                })
            })
            .collect();
        Self { stop, threads }
    }
}
impl Drop for Load {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        while let Some(thread) = self.threads.pop() {
            let _ = thread.join();
        }
    }
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
    // A host-load probe deliberately runs hot; the paired on/off arms share
    // whatever pressure is present.
    let probe_load: usize = args.get(8).and_then(|v| v.parse().ok()).unwrap_or(0);
    if probe_load == 0 {
        thermal()?;
    }
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
    ensure!([32768, 49152, 65536].contains(&depth), "depth 32K/48K/64K");
    for depth in [depth] {
        if probe_load == 0 {
            thermal()?;
        }
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
        let is_bounds = args.get(5).is_some_and(|s| s == "bounds");
        let is_waves = args.get(5).is_some_and(|s| s == "waves");
        let is_causal = args.get(5).is_some_and(|s| s == "causal");
        let is_profile = args.get(5).is_some_and(|s| s == "profile");
        let is_hostgap = args.get(5).is_some_and(|s| s == "hostgap");
        let is_nodes = args.get(5).is_some_and(|s| s == "nodes");
        let is_graph = is_nodes || args.get(5).is_some_and(|s| s == "graph");
        let screen_batch: usize = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(3);
        let members = if is_waves {
            3
        } else if is_bounds || is_graph {
            ensure!((1..=4).contains(&screen_batch), "bounds/graph width 1..4");
            screen_batch
        } else if is_causal {
            ensure!((2..=4).contains(&screen_batch), "causal screen width 2..=4");
            screen_batch
        } else if is_profile || is_hostgap {
            ensure!(
                (1..=4).contains(&screen_batch),
                "profile/hostgap width 1..4"
            );
            screen_batch
        } else {
            4
        };
        let mut states = Vec::new();
        for i in 0..members {
            let mut state = f.new_state(98304)?;
            f.restore_snapshot(&mut state, &snap)?;
            f.prefill(&mut state, &streams[i][..4096 + i * 17], Some(&CANCEL))?;
            states.push(state);
        }
        for j in 0..16 {
            let tokens: Vec<_> = (0..members)
                .map(|i| streams[i][4096 + i * 17 + j])
                .collect();
            step(&f, &mut states, &tokens, true)?;
        }
        let entries = states
            .iter()
            .map(|s| f.capture_snapshot(s))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        if is_causal {
            #[cfg(feature = "perf-ablations")]
            return causal_screen(
                &f,
                &mut states,
                &entries,
                &streams,
                args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(128),
            );
            #[cfg(not(feature = "perf-ablations"))]
            anyhow::bail!("causal requires perf-ablations");
        }
        if is_hostgap {
            return host_gap(
                &f,
                &mut states,
                &entries,
                &streams,
                args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(16),
                args.get(7).is_some_and(|v| v == "1"),
                args.get(8).and_then(|v| v.parse().ok()).unwrap_or(0),
            );
        }
        if is_profile {
            #[cfg(feature = "perf-ablations")]
            return dispatch_profile(
                &f,
                &mut states,
                &entries,
                &streams,
                args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(8),
            );
            #[cfg(not(feature = "perf-ablations"))]
            anyhow::bail!("profile requires perf-ablations");
        }
        if is_waves {
            #[cfg(feature = "perf-ablations")]
            return waves(&ctx, &f, &mut states, &streams);
            #[cfg(not(feature = "perf-ablations"))]
            anyhow::bail!("waves requires perf-ablations");
        }
        if is_graph {
            #[cfg(feature = "perf-ablations")]
            return causal_graph(
                &ctx,
                &f,
                &mut states,
                &entries,
                &streams,
                args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(8),
                is_nodes,
            );
            #[cfg(not(feature = "perf-ablations"))]
            anyhow::bail!("graph requires perf-ablations");
        }
        if is_bounds {
            #[cfg(feature = "perf-ablations")]
            return bounds(
                &ctx,
                &f,
                &mut states,
                &entries,
                &streams,
                args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(32),
            );
            #[cfg(not(feature = "perf-ablations"))]
            anyhow::bail!("bounds requires --features perf-ablations");
        }
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
            let io_before = PidSnapshot::now().ok();
            let start = Instant::now();
            let mut heads = Vec::with_capacity(n);
            for j in 0..n {
                let tokens: Vec<_> = (0..batch).map(|i| streams[i][4310 + j]).collect();
                heads.push(step(&f, &mut states[..batch], &tokens, true)?);
            }
            let ms = start.elapsed().as_secs_f64() * 1000.;
            let disk = match (io_before, PidSnapshot::now().ok()) {
                (Some(before), Some(after)) => Some(PidDelta::between(before, after)),
                _ => None,
            };
            let mut digests: Vec<_> = (0..batch).map(|_| blake3::Hasher::new()).collect();
            for head in &heads {
                for (h, digest) in head.iter().zip(&mut digests) {
                    ensure!(h.iter().all(|x| x.is_finite()), "finite full vocabulary");
                    digest.update(bytemuck::cast_slice(h));
                }
            }
            println!(
                "{}",
                serde_json::json!({"kind":"kill","depth":depth,"batch":batch,"rounds":n,"compression_period":period,"vocab":heads[0][0].len(),"ms":ms,"ms_round":ms/n as f64,"aggregate_tps":batch as f64*n as f64*1000./ms,"disk_read_bytes":disk.map(|d|d.diskio_bytesread),"disk_bytes_per_token":disk.map(|d|d.diskio_bytesread as f64/n as f64),"pageins":disk.map(|d|d.pageins),"hashes":digests.iter().map(|d|d.finalize().to_hex().to_string()).collect::<Vec<_>>(),"first_hashes":hashes(&heads[0])?,"last_hashes":hashes(heads.last().unwrap())?})
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

/// Wall time minus the command GPU span per round: the host/encode/commit gap
/// the counterfactual modes do not model. Uses a plain profile scope (command
/// start/end, not per-dispatch), so timing is near-production.
fn host_gap(
    f: &Forward<'_>,
    states: &mut [State],
    entries: &[qwen_llm::deepseek_v41::forward::Snapshot],
    streams: &[Vec<u32>],
    n: usize,
    probe_qos: bool,
    probe_load: usize,
) -> Result<()> {
    ensure!((1..=64).contains(&n), "1..64 host-gap rounds");
    let width = states.len();
    for (s, snap) in states.iter_mut().zip(entries) {
        f.restore_snapshot(s, snap)?;
    }
    for j in 0..4 {
        let tokens: Vec<_> = (0..width).map(|i| streams[i][4300 + j]).collect();
        step(f, states, &tokens, width > 1)?;
    }
    let _load = (probe_load > 0).then(|| Load::start(probe_load));
    let _qos = GpuThreadQos::enter_if(probe_qos);
    let scope =
        qwen_llm::deepseek_v41::forward::profile::Scope::begin().map_err(anyhow::Error::msg)?;
    let start = Instant::now();
    for j in 0..n {
        let tokens: Vec<_> = (0..width).map(|i| streams[i][4310 + j]).collect();
        step(f, states, &tokens, width > 1)?;
    }
    let wall = start.elapsed().as_secs_f64() * 1000.;
    let report = scope.finish();
    let name = if width > 1 {
        "gpu_step_batch"
    } else {
        "gpu_step"
    };
    let gpu = report
        .operations
        .iter()
        .find(|o| o.operation == name)
        .and_then(|o| o.gpu_ms)
        .unwrap_or(f64::NAN);
    println!(
        "{}",
        serde_json::json!({"kind":"host_gap","width":width,"rounds":n,
            "wall_ms_round": wall / n as f64,
            "gpu_ms_round": gpu / n as f64,
            "gap_ms_round": (wall - gpu) / n as f64,
            "encode_ms": report.operations.iter().find(|o| o.operation==name).map(|o| o.encode_ms),
            "commit_wait_ms": report.operations.iter().find(|o| o.operation==name).map(|o| o.commit_wait_ms),
            "qos": probe_qos, "load": probe_load,
            "embedding_ms_round": report.operations.iter().find(|o| o.operation=="batch_embedding").map(|o| o.wall_ms / n as f64).unwrap_or(0.0),
            "engram_rows_ms_round": report.operations.iter().find(|o| o.operation=="decode_engram_rows").map(|o| o.wall_ms / n as f64).unwrap_or(0.0),
            "metal_encode_ms_round": report.operations.iter().find(|o| o.operation=="batch_metal_encode").map(|o| o.wall_ms / n as f64).unwrap_or(0.0),
            "causal_finish_ms_round": report.operations.iter().find(|o| o.operation=="batch_causal_finish").map(|o| o.wall_ms / n as f64).unwrap_or(0.0),
            "cmd_alloc_ms_round": report.operations.iter().find(|o| o.operation=="batch_cmd_alloc").map(|o| o.wall_ms / n as f64).unwrap_or(0.0),
            "encoder_end_ms_round": report.operations.iter().find(|o| o.operation=="batch_encoder_end").map(|o| o.wall_ms / n as f64).unwrap_or(0.0),
            "commit_ms_round": report.operations.iter().find(|o| o.operation=="batch_commit").map(|o| o.wall_ms / n as f64).unwrap_or(0.0)})
    );
    Ok(())
}

/// Per-dispatch isolated GPU service at the requested width. Each dispatch runs
/// in its own sampled encoder (concurrency removed), so the aggregate is kernel
/// service time by (label, kernel); compare with an unprofiled round for gaps
/// and host time. Rows are per decoded round.
#[cfg(feature = "perf-ablations")]
fn dispatch_profile(
    f: &Forward<'_>,
    states: &mut [State],
    entries: &[qwen_llm::deepseek_v41::forward::Snapshot],
    streams: &[Vec<u32>],
    n: usize,
) -> Result<()> {
    ensure!((1..=16).contains(&n), "1..16 profile rounds");
    let width = states.len();
    for (s, snap) in states.iter_mut().zip(entries) {
        f.restore_snapshot(s, snap)?;
    }
    for j in 0..2 {
        let tokens: Vec<_> = (0..width).map(|i| streams[i][4300 + j]).collect();
        step(f, states, &tokens, width > 1)?;
    }
    let scope = qwen_llm::deepseek_v41::forward::profile::Scope::begin_dispatches()
        .map_err(anyhow::Error::msg)?;
    let start = Instant::now();
    let mut heads = Vec::with_capacity(n);
    for j in 0..n {
        let tokens: Vec<_> = (0..width).map(|i| streams[i][4310 + j]).collect();
        heads.push(step(f, states, &tokens, width > 1)?);
    }
    let wall = start.elapsed().as_secs_f64() * 1000.;
    let report = scope.finish();
    let mut digests: Vec<_> = (0..width).map(|_| blake3::Hasher::new()).collect();
    for head in &heads {
        for (h, d) in head.iter().zip(&mut digests) {
            ensure!(h.iter().all(|v| v.is_finite()), "finite profile head");
            d.update(bytemuck::cast_slice(h));
        }
    }
    let mut counts = std::collections::BTreeMap::<(String, String), (usize, u64)>::new();
    for row in &report.dispatches {
        let key = (row.label.to_string(), row.kernel.clone());
        let entry = counts.entry(key).or_default();
        entry.0 += 1;
        entry.1 += row.gpu_ns.unwrap_or(0);
    }
    let rows: Vec<_> = counts
        .into_iter()
        .map(|((label, kernel), (count, ns))| {
            serde_json::json!({"label": label, "kernel": kernel, "count": count,
                "ms_round": ns as f64 / 1e6 / n as f64,
                "us_per_dispatch": ns as f64 / 1e3 / count as f64})
        })
        .collect();
    let service: f64 = rows.iter().map(|r| r["ms_round"].as_f64().unwrap()).sum();
    println!(
        "{}",
        serde_json::json!({"kind":"dispatch_profile","width":width,"rounds":n,
            "wall_ms_round": wall / n as f64,
            "token_wall_ms_total": report.token_wall_ms,
            "overflow": report.dispatch_overflow,
            "gpu_service_ms_round": service,
            "hashes": digests.iter().map(|d| d.finalize().to_hex().to_string()).collect::<Vec<_>>(),
            "rows": rows})
    );
    Ok(())
}

/// Same prepared B=3 trajectory for every counterfactual. Capture is excluded;
/// the read control preserves real expert IDs, grids and exact normal outputs.
#[cfg(feature = "perf-ablations")]
fn bounds(
    ctx: &MetalContext,
    f: &Forward<'_>,
    states: &mut [State],
    entries: &[qwen_llm::deepseek_v41::forward::Snapshot],
    streams: &[Vec<u32>],
    n: usize,
) -> Result<()> {
    use objc2_metal::{MTLCommandBuffer, MTLCommandQueue};
    use qwen_llm::metal::decode_bounds::{Experiment, Mode};
    ensure!((8..=64).contains(&n), "8..64 bounded probe steps");
    let width = states.len();
    ensure!(width <= streams.len(), "width has a stream");
    let experiment = Experiment::default();
    // Price every hold on the production causal schedule rather than the
    // native encoder cuts the bound used to force.
    let _causal = metal::decode_bounds::batch_causal_scope(true);
    let restore = |states: &mut [State]| -> Result<()> {
        for (s, e) in states.iter_mut().zip(entries) {
            f.restore_snapshot(s, e)?;
        }
        Ok(())
    };
    let run = |mode,
               states: &mut [State]|
     -> Result<(f64, Vec<String>, qwen_llm::metal::decode_bounds::Stats)> {
        restore(states)?;
        let scope = experiment.scope(mode)?;
        let start = Instant::now();
        let mut heads = Vec::with_capacity(n);
        for j in 0..n {
            let tokens: Vec<_> = (0..width).map(|i| streams[i][4310 + j]).collect();
            heads.push(step(f, states, &tokens, width > 1)?);
        }
        let ms = start.elapsed().as_secs_f64() * 1000. / n as f64;
        let stats = scope.finish();
        let mut digests: Vec<_> = (0..width).map(|_| blake3::Hasher::new()).collect();
        for row in &heads {
            for (h, d) in row.iter().zip(&mut digests) {
                ensure!(h.iter().all(|v| v.is_finite()), "finite bound head");
                d.update(bytemuck::cast_slice(h));
            }
        }
        Ok((
            ms,
            digests
                .iter()
                .map(|d| d.finalize().to_hex().to_string())
                .collect(),
            stats,
        ))
    };
    let (_, reference, _) = run(Mode::Capture, states)?;
    println!(
        "{}",
        serde_json::json!({"kind":"bound_capture","causal":true,"frames":n,"calls":experiment.captured_calls(),"routed_bytes_per_round":experiment.captured_bytes()/n as u64})
    );
    // Compile/warm the substituted pipeline outside the measured cells.
    let (_, check, _) = run(Mode::RoutedRead, states)?;
    ensure!(
        check == reference,
        "held expert outputs preserve exact normal trajectory"
    );
    for mode in [
        Mode::Reference,
        Mode::AttentionIo,
        Mode::PostControls,
        Mode::RoutedRead,
        Mode::RoutedHold,
        Mode::DenseHold,
        Mode::AttentionCoreHold,
        Mode::FlashPartialHold,
        Mode::FlashCombineHold,
        Mode::IndexHold,
        Mode::Reference,
        Mode::AllHc,
    ] {
        thermal()?;
        let (ms, hashes, stats) = run(mode, states)?;
        if matches!(mode, Mode::Reference | Mode::RoutedRead | Mode::RoutedHold) {
            ensure!(hashes == reference, "normal/stream endpoint bits");
        }
        println!(
            "{}",
            serde_json::json!({"kind":"bound","causal":true,"mode":mode,"rounds":n,"ms_round":ms,"stats":stats,"hashes":hashes})
        );
    }
    for read in [true, false, true, false] {
        thermal()?;
        let cmd = ctx
            .queue
            .commandBuffer()
            .ok_or_else(|| anyhow::anyhow!("command"))?;
        let enc = metal::KernelEncoder::begin(&cmd);
        experiment.encode_streams(ctx, &enc, read)?;
        enc.end();
        cmd.commit();
        metal::wait_completed(&cmd)?;
        let ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1000. / n as f64;
        println!(
            "{}",
            serde_json::json!({"kind":"expert_stream","read":read,"frames":n,"ms_per_round":ms,"logical_gb_s":read.then_some((experiment.captured_bytes() as f64/n as f64)/ms/1e6)})
        );
    }
    Ok(())
}

/// One production causal round with every wave boundary timestamped: the
/// kernel names per wave and the GPU span each wave takes. Waves are separated
/// by buffer barriers, so the spans sum to the scope critical path.
#[cfg(feature = "perf-ablations")]
fn causal_graph(
    ctx: &MetalContext,
    f: &Forward<'_>,
    states: &mut [State],
    entries: &[qwen_llm::deepseek_v41::forward::Snapshot],
    streams: &[Vec<u32>],
    n: usize,
    nodes: bool,
) -> Result<()> {
    ensure!((1..=8).contains(&n), "1..8 graph rounds");
    let width = states.len();
    for (s, snap) in states.iter_mut().zip(entries) {
        f.restore_snapshot(s, snap)?;
    }
    for j in 0..8 {
        let tokens: Vec<_> = (0..width).map(|i| streams[i][4300 + j]).collect();
        step(f, states, &tokens, true)?;
    }
    let mut reference_ms = Vec::new();
    for _ in 0..3 {
        for (s, snap) in states.iter_mut().zip(entries) {
            f.restore_snapshot(s, snap)?;
        }
        let started = Instant::now();
        for j in 0..8 {
            let tokens: Vec<_> = (0..width).map(|i| streams[i][4310 + j]).collect();
            step(f, states, &tokens, width > 1)?;
        }
        reference_ms.push(started.elapsed().as_secs_f64() * 1e3 / 8.0);
    }
    for (s, snap) in states.iter_mut().zip(entries) {
        f.restore_snapshot(s, snap)?;
    }
    let mut expected = Vec::new();
    for j in 0..n {
        let tokens: Vec<_> = (0..width).map(|i| streams[i][4310 + j]).collect();
        expected.push(step(f, states, &tokens, width > 1)?);
    }
    for (s, snap) in states.iter_mut().zip(entries) {
        f.restore_snapshot(s, snap)?;
    }
    let guard = if nodes {
        metal::causal_node_graph_begin(ctx, 16384)?
    } else {
        metal::causal_graph_begin(ctx, 2000)?
    };
    let scope =
        qwen_llm::deepseek_v41::forward::profile::Scope::begin().map_err(anyhow::Error::msg)?;
    let start = Instant::now();
    let mut heads = Vec::with_capacity(n);
    for j in 0..n {
        let tokens: Vec<_> = (0..width).map(|i| streams[i][4310 + j]).collect();
        heads.push(step(f, states, &tokens, width > 1)?);
    }
    let wall_ms = start.elapsed().as_secs_f64() * 1000. / n as f64;
    let report = scope.finish();
    let gpu_ms = report
        .operations
        .iter()
        .find(|o| o.operation == "gpu_step_batch")
        .and_then(|o| o.gpu_ms)
        .map(|v| v / n as f64)
        .unwrap_or(f64::NAN);
    let mut value = metal::causal_graph_finish(ctx, guard)?;
    ensure!(
        expected
            .iter()
            .flatten()
            .flatten()
            .zip(heads.iter().flatten().flatten())
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "node capture changed V4.1 logits"
    );
    value["reference_wall_ms_round"] = serde_json::json!(reference_ms);
    value["bitwise"] = serde_json::json!(true);
    value["width"] = serde_json::json!(width);
    value["rounds"] = serde_json::json!(n);
    value["wall_ms_round"] = serde_json::json!(wall_ms);
    value["gpu_ms_round"] = serde_json::json!(gpu_ms);
    value["hashes"] = serde_json::json!(
        heads
            .iter()
            .map(|row| {
                let mut d = blake3::Hasher::new();
                for h in row {
                    d.update(bytemuck::cast_slice(h));
                }
                d.finalize().to_hex().to_string()
            })
            .collect::<Vec<_>>()
    );
    println!("{value}");
    Ok(())
}

#[cfg(feature = "perf-ablations")]
fn waves(
    ctx: &MetalContext,
    f: &Forward<'_>,
    states: &mut [State],
    streams: &[Vec<u32>],
) -> Result<()> {
    use objc2_foundation::NSString;
    use objc2_metal::{
        MTLBinding, MTLCommandBuffer, MTLCommandQueue, MTLDevice, MTLLibrary, MTLPipelineOption,
    };
    let tokens: Vec<_> = (0..3).map(|i| streams[i][4310]).collect();
    let entries = states
        .iter()
        .map(|s| f.capture_snapshot(s))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    native_wave_calibration(f, states, &entries, &tokens, "before")?;
    for (s, snap) in states.iter_mut().zip(&entries) {
        f.restore_snapshot(s, snap)?;
    }
    let guard = metal::decode_capture_begin();
    let native = step(f, states, &tokens, true)?;
    let graph = metal::decode_capture_finish(guard);
    let expected = hashes(&native)?;
    println!(
        "{}",
        serde_json::json!({"kind":"wave_capture","metadata":graph.metadata(),"hashes":expected})
    );
    let mut access = std::collections::BTreeMap::new();
    for kernel in graph.kernels() {
        let function = ctx
            .library
            .newFunctionWithName(&NSString::from_str(&kernel))
            .unwrap();
        let mut reflection = None;
        unsafe {
            ctx.device
                .newComputePipelineStateWithFunction_options_reflection_error(
                    &function,
                    MTLPipelineOption::BindingInfo,
                    Some(&mut reflection),
                )
        }
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let rows = reflection
            .unwrap()
            .bindings()
            .iter()
            .filter(|b| b.r#type().0 == 0 && b.isUsed())
            .map(|b| (b.index(), b.access().0 != 0))
            .collect::<Vec<_>>();
        access.insert(kernel, rows);
    }
    let flush_src = metal::MetalTensor::zeros_f32(ctx, vec![134217728])?;
    let flush_dst = metal::MetalTensor::zeros_f32(ctx, vec![134217728])?;
    let flush = || -> Result<()> {
        let cmd = ctx.queue.commandBuffer().unwrap();
        let enc = metal::KernelEncoder::begin(&cmd);
        metal::encode_copy_offset_f32(ctx, &enc, &flush_src, 0, &flush_dst, 134217728)?;
        enc.end();
        cmd.commit();
        metal::wait_completed(&cmd)?;
        Ok(())
    };
    for rep in 0..4 {
        for dag in if rep % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        } {
            thermal()?;
            flush()?;
            let mut gpu_ms = 0.;
            let t = Instant::now();
            for segment in 0..graph.segments() {
                gpu_ms += graph.replay_segment(ctx, segment, dag.then_some(&access));
            }
            ensure!(
                hashes(&graph.outputs(3))? == expected,
                "causal replay mismatch"
            );
            println!(
                "{}",
                serde_json::json!({"kind":"wave_causal","dag":dag,"rep":rep,"gpu_ms":gpu_ms,"wall_ms":t.elapsed().as_secs_f64()*1000.,"bitwise":true})
            );
        }
    }
    let frozen = graph.freeze(ctx, &access);
    ensure!(
        hashes(&graph.outputs(3))? == expected,
        "frozen input oracle mismatch"
    );
    println!(
        "{}",
        serde_json::json!({"kind":"wave_frozen_metadata","metadata":frozen.metadata()})
    );
    for rep in 0..6 {
        let modes = if rep % 2 == 0 { [0, 1, 2] } else { [2, 1, 0] };
        for mode in modes {
            thermal()?;
            frozen.restore();
            flush()?;
            let cmd = ctx.queue.commandBuffer().unwrap();
            let enc = if mode == 0 {
                metal::KernelEncoder::begin(&cmd)
            } else {
                metal::KernelEncoder::begin_concurrent(&cmd)
            };
            frozen.replay(&enc, mode == 1);
            enc.end();
            cmd.commit();
            metal::wait_completed(&cmd)?;
            ensure!(cmd.error().is_none(), "frozen command error");
            ensure!(frozen.check(), "frozen reflected output mismatch");
            println!(
                "{}",
                serde_json::json!({"kind":"wave_frozen","mode":(["serial","branches","global"][mode]),"rep":rep,"gpu_ms":(cmd.GPUEndTime()-cmd.GPUStartTime())*1000.,"all_outputs_bitwise":true})
            );
        }
    }
    native_wave_calibration(f, states, &entries, &tokens, "after")?;
    Ok(())
}

#[cfg(feature = "perf-ablations")]
fn native_wave_calibration(
    f: &Forward<'_>,
    states: &mut [State],
    entries: &[qwen_llm::deepseek_v41::forward::Snapshot],
    tokens: &[u32],
    label: &str,
) -> Result<()> {
    let mut expected = None;
    for rep in 0..6 {
        thermal()?;
        for (s, snap) in states.iter_mut().zip(entries) {
            f.restore_snapshot(s, snap)?;
        }
        let scope =
            qwen_llm::deepseek_v41::forward::profile::Scope::begin().map_err(anyhow::Error::msg)?;
        let start = Instant::now();
        let outputs = step(f, states, tokens, true)?;
        let wall_ms = start.elapsed().as_secs_f64() * 1000.;
        let report = scope.finish();
        let h = hashes(&outputs)?;
        if let Some(ref r) = expected {
            ensure!(*r == h, "native fixed-frame mismatch");
        } else {
            expected = Some(h.clone());
        }
        let op = report
            .operations
            .iter()
            .find(|o| o.operation == "gpu_step_batch")
            .unwrap();
        println!(
            "{}",
            serde_json::json!({"kind":"wave_native","label":label,"rep":rep,"wall_ms":wall_ms,"gpu_ms":op.gpu_ms,"hashes":h,"pageins":report.process_pageins,"disk_read_bytes":report.process_disk_read_bytes})
        );
    }
    Ok(())
}

#[cfg(feature = "perf-ablations")]
fn causal_screen(
    f: &Forward<'_>,
    states: &mut [State],
    entries: &[qwen_llm::deepseek_v41::forward::Snapshot],
    streams: &[Vec<u32>],
    n: usize,
) -> Result<()> {
    ensure!((32..=256).contains(&n), "bounded causal screen");
    let width = states.len();
    ensure!((2..=4).contains(&width), "causal screen width 2..=4");
    let mut reference = None;
    for (arm, enabled) in [false, true, true, false].into_iter().enumerate() {
        let thermal_before = thermal_level()?;
        for (s, snap) in states.iter_mut().zip(entries) {
            f.restore_snapshot(s, snap)?;
        }
        let guard = metal::decode_bounds::batch_causal_scope(enabled);
        for j in 0..8 {
            let tokens: Vec<_> = (0..width).map(|i| streams[i][4300 + j]).collect();
            step(f, states, &tokens, true)?;
        }
        let unix = || {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64()
        };
        let started_unix = unix();
        let begin = Instant::now();
        let mut heads = Vec::with_capacity(n);
        for j in 0..n {
            let tokens: Vec<_> = (0..width).map(|i| streams[i][4310 + j]).collect();
            heads.push(step(f, states, &tokens, true)?);
        }
        let ms_round = begin.elapsed().as_secs_f64() * 1000. / n as f64;
        let finished_unix = unix();
        let thermal_after = thermal_level()?;
        drop(guard);
        let mut digest = blake3::Hasher::new();
        for h in &heads {
            for v in h {
                ensure!(v.iter().all(|x| x.is_finite()), "finite causal logits");
                digest.update(bytemuck::cast_slice(v));
            }
        }
        let logits = digest.finalize().to_hex().to_string();
        let payload = states
            .iter()
            .map(|s| f.capture_snapshot(s).map(|x| x.diagnostic_digest()))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut follow = Vec::new();
        for j in 0..8 {
            let tokens: Vec<_> = (0..width).map(|i| streams[i][4310 + n + j]).collect();
            follow.push(step(f, states, &tokens, false)?);
        }
        let follow: Vec<_> = follow.iter().map(|h| hashes(h)).collect::<Result<_>>()?;
        let bits = (logits.clone(), payload.clone(), follow.clone());
        if let Some(ref expected) = reference {
            ensure!(
                *expected == bits,
                "causal arithmetic or persistent state changed"
            );
        } else {
            reference = Some(bits);
        }
        println!(
            "{}",
            serde_json::json!({"kind":"causal_screen","width":width,"started_unix":started_unix,"finished_unix":finished_unix,"thermal_before":thermal_before,"thermal_after":thermal_after,"arm":arm,"enabled":enabled,"rounds":n,"ms_round":ms_round,"logits":logits,"state":payload,"follow":follow,"bitwise":true})
        );
    }
    Ok(())
}
