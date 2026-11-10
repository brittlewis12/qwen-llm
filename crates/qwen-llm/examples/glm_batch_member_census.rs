//! GLM per-member width census: MODEL DEPTH [HOST_STEPS=32].
//! --features perf-ablations; canonical lease required. B=1 and B=4 dispatch
//! profiles plus a B=3 host-bubble measurement. No production change.
#[cfg(not(feature = "perf-ablations"))]
fn main() {
    panic!("requires perf-ablations");
}
#[cfg(feature = "perf-ablations")]
mod probe {
    use qwen_llm::{
        gguf::GgufFile,
        glm5_next_metal::*,
        metal::*,
        sampling::{Sampler, SamplingConfig},
    };
    use serde_json::{Value, json};
    use std::{collections::BTreeMap, io::Write, time::Instant};
    fn emit(v: Value) {
        println!("{v}");
        std::io::stdout().flush().unwrap();
    }
    fn thermal() -> i32 {
        let s = std::process::Command::new("notifyutil")
            .args(["-g", "com.apple.system.thermalpressurelevel"])
            .output()
            .unwrap();
        String::from_utf8(s.stdout)
            .unwrap()
            .split_whitespace()
            .last()
            .unwrap()
            .parse()
            .unwrap()
    }
    fn family(kernel: &str) -> &'static str {
        if kernel.contains("q6_K") || kernel.contains("q6_k") {
            "q6_projection"
        } else if kernel.contains("q8_0") {
            "q8_projection"
        } else if kernel.contains("expert_native_gate_up") || kernel.contains("glm_batch_gate_up") {
            "routed_gate_up"
        } else if kernel.contains("expert_native_down") || kernel.contains("glm_batch_down") {
            "routed_down"
        } else if kernel.contains("weighted_sum") || kernel.contains("finalizer") {
            "routed_reduce"
        } else if kernel.contains("mat_vec") || kernel.contains("matvec") {
            "matvec"
        } else if kernel.contains("indexer_append") || kernel.contains("scatter_offset") {
            "index_cache"
        } else if kernel.contains("indexer") || kernel.contains("index_score") {
            "index_scores"
        } else if kernel.contains("select_top_k") || kernel.contains("expand_selection") {
            "selection"
        } else if kernel.contains("selected_attention")
            || kernel.contains("latent_attention")
            || kernel.contains("attention")
        {
            "attention"
        } else if kernel.contains("kda") {
            "kda"
        } else if kernel.contains("hc_") || kernel.contains("mhc") {
            "hc"
        } else if kernel.contains("route_learned") {
            "router"
        } else if kernel.contains("rms_norm")
            || kernel.contains("l2_norm")
            || kernel.contains("swiglu")
            || kernel.contains("silu")
            || kernel.contains("argmax")
            || kernel.contains("copy")
            || kernel.contains("fill")
            || kernel.contains("add")
            || kernel.contains("scale")
        {
            "pointwise"
        } else {
            "other"
        }
    }
    pub fn run() -> anyhow::Result<()> {
        let args: Vec<_> = std::env::args().collect();
        let depth: usize = args[2].parse()?;
        let host_steps: usize = args
            .get(3)
            .map(|v| v.parse())
            .transpose()?
            .unwrap_or(32usize);
        assert!(std::env::var_os("QWEN_METAL_LEASE_FD").is_some());
        assert!(thermal() < 2, "thermal level 2 at start");
        let ctx = MetalContext::new()?;
        let g = GgufFile::open(&args[1])?;
        let w = Glm5NextWeights::load(&ctx, &g)?;
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
        emit(json!({
            "kind":"header","depth":depth,"host_steps":host_steps,
            "device":ctx.device_facts(),
            "commit":std::process::Command::new("git").args(["rev-parse","HEAD"]).output()
                .map(|o|String::from_utf8_lossy(&o.stdout).trim().to_string()).ok()
        }));
        for r in 0..4 {
            if r > 0 {
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
                        .take(depth)
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
            batch.step(
                &ctx,
                &mut ss.iter_mut().collect::<Vec<_>>(),
                &(0..4).map(|r| token(r, p)).collect::<Vec<_>>(),
            )?;
        }
        let entries: Vec<_> = ss.iter().map(|s| s.capture_snapshot().unwrap()).collect();
        let mut profiles = BTreeMap::<usize, BTreeMap<String, (usize, u64)>>::new();
        for width in [1usize, 4usize] {
            for r in 0..width {
                ss[r].restore_snapshot(&entries[r])?;
            }
            for p in 0..8 {
                batch.step(
                    &ctx,
                    &mut ss[..width].iter_mut().collect::<Vec<_>>(),
                    &(0..width).map(|r| token(r, 64 + p)).collect::<Vec<_>>(),
                )?;
            }
            let tokens: Vec<_> = (0..width).map(|r| token(r, 72)).collect();
            let guard = dispatch_profile_begin(&ctx, 8192)?;
            let t = Instant::now();
            batch.step(
                &ctx,
                &mut ss[..width].iter_mut().collect::<Vec<_>>(),
                &tokens,
            )?;
            let wall_ms = t.elapsed().as_secs_f64() * 1e3;
            let report = dispatch_profile_finish(&ctx, guard)?;
            assert_eq!(report.overflow, 0);
            let mut rows = BTreeMap::<String, (usize, u64)>::new();
            for row in &report.rows {
                let v = rows.entry(row.kernel.clone()).or_default();
                v.0 += 1;
                v.1 += row.gpu_ns.unwrap_or(0);
            }
            emit(json!({
                "kind":"profile","width":width,"wall_ms":wall_ms,"dispatches":report.rows.len(),
                "kernels":rows.iter().map(|(k,(c,ns))| json!({"kernel":k,"family":family(k),"calls":c,"gpu_ms":*ns as f64/1e6})).collect::<Vec<_>>()
            }));
            profiles.insert(width, rows);
        }
        let empty = BTreeMap::new();
        let b1 = profiles.get(&1).unwrap_or(&empty);
        let b4 = profiles.get(&4).unwrap_or(&empty);
        let mut names: Vec<_> = b1.keys().chain(b4.keys()).cloned().collect();
        names.sort();
        names.dedup();
        for name in names {
            let (c1, n1) = b1.get(&name).copied().unwrap_or((0, 0));
            let (c4, n4) = b4.get(&name).copied().unwrap_or((0, 0));
            emit(json!({
                "kind":"pair","kernel":name,"family":family(&name),
                "b1_calls":c1,"b1_ms":n1 as f64/1e6,
                "b4_calls":c4,"b4_ms":n4 as f64/1e6,
                "call_ratio":if c1>0 {c4 as f64/c1 as f64} else {0.0},
                "delta_ms":(n4 as f64 - n1 as f64)/1e6,
            }));
        }
        let mut fam = BTreeMap::<&'static str, (usize, f64, usize, f64, usize, f64)>::new();
        for (name, (c1, n1)) in b1.iter() {
            let e = fam.entry(family(name)).or_default();
            e.0 += c1;
            e.1 += *n1 as f64 / 1e6;
        }
        for (name, (c4, n4)) in b4.iter() {
            let e = fam.entry(family(name)).or_default();
            e.2 += c4;
            e.3 += *n4 as f64 / 1e6;
            e.4 += 1;
            e.5 += 0.;
        }
        for (name, (c1, t1, c4, t4, _, _)) in fam {
            emit(json!({
                "kind":"family","family":name,
                "b1_calls":c1,"b1_ms":t1,"b4_calls":c4,"b4_ms":t4,
                "delta_ms":t4-t1,
                "m_us_per_row":(t4-t1)*1e3/3.0,
            }));
        }
        // Host bubble at B=3: wall vs GPU span, encode, remainder and sampling.
        for r in 0..3 {
            ss[r].restore_snapshot(&entries[r])?;
        }
        for p in 0..8 {
            batch.step(
                &ctx,
                &mut ss[..3].iter_mut().collect::<Vec<_>>(),
                &(0..3).map(|r| token(r, 64 + p)).collect::<Vec<_>>(),
            )?;
        }
        let mut sampler = Sampler::new(SamplingConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.95,
            min_p: 0.0,
            seed: 19,
        })?;
        let mut sums = [0f64; 6];
        for p in 0..host_steps {
            let wall = Instant::now();
            let out = batch.step(
                &ctx,
                &mut ss[..3].iter_mut().collect::<Vec<_>>(),
                &(0..3).map(|r| token(r, 64 + p)).collect::<Vec<_>>(),
            )?;
            let step_wall = wall.elapsed().as_secs_f64() * 1e3;
            let (encode, gpu) = take_last_step_timing().unwrap_or((0., 0.));
            let sample = Instant::now();
            for row in &out {
                std::hint::black_box(sampler.sample(row)?.token);
            }
            let sample_ms = sample.elapsed().as_secs_f64() * 1e3;
            sums[0] += step_wall;
            sums[1] += gpu;
            sums[2] += encode;
            sums[3] += step_wall - gpu;
            sums[4] += step_wall - gpu - encode;
            sums[5] += sample_ms;
        }
        let q = host_steps as f64;
        emit(json!({
            "kind":"host_bubble","width":3,"steps":host_steps,
            "wall_ms":sums[0]/q,"gpu_ms":sums[1]/q,"encode_ms":sums[2]/q,
            "outside_gpu_ms":sums[3]/q,"outside_gpu_minus_encode_ms":sums[4]/q,
            "sample_ms":sums[5]/q,"thermal_end":thermal(),
        }));
        emit(json!({"kind":"complete"}));
        Ok(())
    }
}
#[cfg(feature = "perf-ablations")]
fn main() -> anyhow::Result<()> {
    probe::run()
}
