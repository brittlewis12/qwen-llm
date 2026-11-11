//! V4.1 live-append CED probe; build with `--features perf-ablations`.
//!
//! Builds a `DEPTH`-token prefix once through the same cold-CED streaming path
//! serve uses, snapshots it, then for each append size compares:
//!   - Full:  the ordinary warm-append prefill;
//!   - CED(k): a live CED episode whose encoder slices are capped at `k` rows.
//! The CED plan keeps `full_chunk_rows` from the restored state (the request's
//! selected geometry), so a smaller `k` changes only the submission width.
//!
//! Per arm it records wall time, encoder slice count and per-slice wall, whether
//! the native planner declined (the decoder dependency halo is 2541 rows), and
//! final-logit exactness against Full (bitwise, mean KL, signed dNLL at the
//! corpus continuation). One common post-append decode step compares state.
//!
//! No production change. Usage:
//! `v41_live_ced_probe <model.gguf> <corpus> <depth> [--widths 4096,2048,1024,512]
//!  [--sizes 405,1499,2260,2600,3000,5000,8191,20650] [--no-state]`

#[cfg(not(feature = "perf-ablations"))]
fn main() {
    panic!("requires perf-ablations");
}

#[cfg(feature = "perf-ablations")]
mod probe {
    use anyhow::{Context, Result, ensure};
    use qwen_llm::{
        deepseek_v41::{
            forward::{Forward, State},
            gguf::{ResidentWeights, resident_storage_plan_with_target},
        },
        gguf::GgufFile,
        metal::{self, MetalContext},
        tokenizer::NativeTokenizer,
    };
    use serde_json::json;
    use std::{
        io::Write,
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

    fn thermal() -> Result<u32> {
        let out = std::process::Command::new("notifyutil")
            .args(["-g", "com.apple.system.thermalpressurelevel"])
            .output()?;
        let level = String::from_utf8(out.stdout)?
            .split_whitespace()
            .last()
            .context("thermal output")?
            .parse()?;
        ensure!(level < 2, "thermal pressure >=2");
        Ok(level)
    }

    fn emit(v: serde_json::Value) {
        println!("{v}");
        std::io::stdout().flush().unwrap();
    }

    fn log_softmax(x: &[f32]) -> Vec<f64> {
        let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        let sum: f64 = x.iter().map(|v| (*v as f64 - max).exp()).sum();
        let logsum = max + sum.ln();
        x.iter().map(|v| *v as f64 - logsum).collect()
    }

    fn mean_kl(reference: &[f32], other: &[f32]) -> f64 {
        let (p, q) = (log_softmax(reference), log_softmax(other));
        p.iter().zip(&q).map(|(lp, lq)| lp.exp() * (lp - lq)).sum()
    }

    fn log_prob(logits: &[f32], token: u32) -> f64 {
        log_softmax(logits)
            .get(token as usize)
            .copied()
            .unwrap_or(f64::NEG_INFINITY)
    }

    fn top1(logits: &[f32]) -> usize {
        let mut best = 0;
        for (i, v) in logits.iter().enumerate() {
            if *v > logits[best] {
                best = i;
            }
        }
        best
    }

    fn exact(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    fn max_abs(a: &[f32], b: &[f32]) -> f64 {
        qwen_llm::compare::report_max_abs_diff_f32(a, b) as f64
    }

    fn corpus(path: &str) -> Result<String> {
        std::fs::read_to_string(path).with_context(|| format!("corpus {path}"))
    }

    fn build_prefix(f: &Forward<'_>, depth: usize, tokens: &[u32]) -> Result<State> {
        let mut state = f.new_state(98_304)?;
        ensure!(
            f.begin_cold_prefill_episode(&mut state, depth)?,
            "cold prefix eligible for CED"
        );
        loop {
            let slice =
                f.cold_prefill_episode_slice(&mut state, &tokens[..depth], Some(&CANCEL), 4096)?;
            if slice.complete {
                break;
            }
        }
        ensure!(state.position() == depth, "prefix position");
        Ok(state)
    }

    fn full_append(
        f: &Forward<'_>,
        snap: &qwen_llm::deepseek_v41::forward::Snapshot,
        append: &[u32],
    ) -> Result<(f64, Vec<f32>)> {
        let mut state = f.new_state(98_304)?;
        f.restore_snapshot(&mut state, snap)?;
        let start = Instant::now();
        let logits = f.prefill(&mut state, append, Some(&CANCEL))?;
        Ok((start.elapsed().as_secs_f64() * 1e3, logits))
    }

    struct CedArm {
        declined: bool,
        wall_ms: f64,
        slices: Vec<(usize, f64)>,
        logits: Vec<f32>,
    }

    fn ced_append(
        f: &Forward<'_>,
        snap: &qwen_llm::deepseek_v41::forward::Snapshot,
        append: &[u32],
        max_rows: usize,
    ) -> Result<CedArm> {
        let mut state = f.new_state(98_304)?;
        f.restore_snapshot(&mut state, snap)?;
        if !f.begin_cold_prefill_episode(&mut state, append.len())? {
            return Ok(CedArm {
                declined: true,
                wall_ms: 0.0,
                slices: Vec::new(),
                logits: Vec::new(),
            });
        }
        let start = Instant::now();
        let mut slices = Vec::new();
        let logits = loop {
            let t = Instant::now();
            let slice =
                f.cold_prefill_episode_slice(&mut state, append, Some(&CANCEL), max_rows)?;
            slices.push((slice.encoder_rows, t.elapsed().as_secs_f64() * 1e3));
            if slice.complete {
                break slice.logits;
            }
        };
        Ok(CedArm {
            declined: false,
            wall_ms: start.elapsed().as_secs_f64() * 1e3,
            slices,
            logits,
        })
    }

    pub fn run() -> Result<()> {
        unsafe {
            libc::signal(libc::SIGINT, cancel as *const () as libc::sighandler_t);
            libc::signal(libc::SIGTERM, cancel as *const () as libc::sighandler_t);
        }
        let args: Vec<String> = std::env::args().collect();
        ensure!(args.len() >= 4, "model corpus depth");
        let model = &args[1];
        let corpus_path = &args[2];
        let depth: usize = args[3].parse()?;
        let mut widths = vec![4096usize, 2048, 1024, 512];
        let mut sizes = vec![405usize, 1499, 2260, 2600, 3000, 5000, 8191, 20650];
        let mut check_state = true;
        let mut i = 4;
        while i < args.len() {
            match args[i].as_str() {
                "--widths" => {
                    widths = args
                        .get(i + 1)
                        .context("--widths value")?
                        .split(',')
                        .map(|v| v.parse::<usize>())
                        .collect::<Result<_, _>>()?;
                    i += 2;
                }
                "--sizes" => {
                    sizes = args
                        .get(i + 1)
                        .context("--sizes value")?
                        .split(',')
                        .map(|v| v.parse::<usize>())
                        .collect::<Result<_, _>>()?;
                    i += 2;
                }
                "--no-state" => {
                    check_state = false;
                    i += 1;
                }
                other => anyhow::bail!("unknown argument {other}"),
            }
        }
        ensure!(
            widths.iter().all(|w| (1..=4096).contains(w)),
            "width 1..=4096"
        );
        let thermal_level = thermal()?;
        let g = Arc::new(GgufFile::open(model)?);
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
        let mut text = corpus(corpus_path)?;
        let longest = sizes.iter().copied().max().unwrap_or(0);
        while tok.encode(&text, false)?.len() < depth + longest + 1 {
            text.push('\n');
            text.push_str(&text.clone());
            ensure!(
                text.len() < 8 << 20,
                "corpus too short for depth {depth} + {longest}"
            );
        }
        let ids: Vec<u32> = tok
            .encode(&text, false)?
            .into_iter()
            .map(|v| v as u32)
            .collect();
        ensure!(ids.len() > depth + longest, "token extent");
        emit(
            json!({"kind":"load","depth":depth,"corpus_tokens":ids.len(),"thermal":thermal_level,"widths":widths,"sizes":sizes}),
        );

        let build = Instant::now();
        let base = build_prefix(&f, depth, &ids)?;
        let selected_rows = base.selected_prefill_rows();
        let snap = f.capture_snapshot(&base)?;
        drop(base);
        emit(
            json!({"kind":"snapshot","depth":depth,"selected_prefill_rows":selected_rows,"build_ms":build.elapsed().as_secs_f64()*1e3}),
        );

        for &n in &sizes {
            let append = &ids[depth..depth + n];
            let next = ids[depth + n];
            let (full_ms, full_logits) = full_append(&f, &snap, append)?;
            let full_top = top1(&full_logits);
            let mut ced_steps = Vec::new();
            for &width in &widths {
                let arm = ced_append(&f, &snap, append, width)?;
                let exact_logits = !arm.declined && exact(&arm.logits, &full_logits);
                let kl = if arm.declined {
                    None
                } else {
                    Some(mean_kl(&full_logits, &arm.logits))
                };
                let dnll = if arm.declined {
                    None
                } else {
                    Some(log_prob(&arm.logits, next) - log_prob(&full_logits, next))
                };
                let max_abs_diff = if arm.declined {
                    None
                } else {
                    Some(max_abs(&arm.logits, &full_logits))
                };
                let top1_agree = !arm.declined && top1(&arm.logits) == full_top;
                // One common post-append decode step compares resulting state.
                let state_check = if arm.declined || !check_state {
                    None
                } else {
                    let mut s = f.new_state(98_304)?;
                    f.restore_snapshot(&mut s, &snap)?;
                    // Re-run the arm on a fresh state so the step starts from it.
                    if f.begin_cold_prefill_episode(&mut s, append.len())? {
                        loop {
                            let slice =
                                f.cold_prefill_episode_slice(&mut s, append, Some(&CANCEL), width)?;
                            if slice.complete {
                                break;
                            }
                        }
                        let step_logits = f
                            .step(&mut s, next, Some(&CANCEL))
                            .map_err(anyhow::Error::msg)?;
                        let mut ref_state = f.new_state(98_304)?;
                        f.restore_snapshot(&mut ref_state, &snap)?;
                        let _ = f.prefill(&mut ref_state, append, Some(&CANCEL))?;
                        let step_ref = f
                            .step(&mut ref_state, next, Some(&CANCEL))
                            .map_err(anyhow::Error::msg)?;
                        Some(json!({
                            "state_bitwise": exact(&step_logits, &step_ref),
                            "state_kl": mean_kl(&step_ref, &step_logits),
                            "state_max_abs": max_abs(&step_ref, &step_logits),
                        }))
                    } else {
                        None
                    }
                };
                let slice_walls: Vec<f64> = arm.slices.iter().map(|(_, ms)| *ms).collect();
                ced_steps.push(json!({
                    "max_rows": width,
                    "declined": arm.declined,
                    "wall_ms": arm.wall_ms,
                    "slices": arm.slices.len(),
                    "encoder_rows": arm.slices.iter().map(|(r, _)| *r).collect::<Vec<_>>(),
                    "slice_walls_ms": slice_walls,
                    "max_slice_wall_ms": arm.slices.iter().map(|(_, ms)| *ms).fold(0.0, f64::max),
                    "logits_bitwise": exact_logits,
                    "top1_agree": top1_agree,
                    "mean_kl": kl,
                    "signed_dnll": dnll,
                    "max_abs": max_abs_diff,
                    "state": state_check,
                    "thermal": thermal().unwrap_or(u32::MAX),
                }));
            }
            emit(json!({
                "kind":"append",
                "depth": depth,
                "new_tokens": n,
                "full_ms": full_ms,
                "full_top1": full_top,
                "ced": ced_steps,
            }));
        }
        emit(json!({"kind":"done","thermal":thermal()?}));
        Ok(())
    }
}

#[cfg(feature = "perf-ablations")]
fn main() -> anyhow::Result<()> {
    probe::run()
}
