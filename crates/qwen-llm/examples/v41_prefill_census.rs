//! Direct cold-CED/root and deep Full-append census. No row cycling.
//! model corpus depth rows [native|stages|dispatch] [reps=1]
//! Dispatch sampling serializes kernels: use native for wall claims.
use anyhow::{Result, ensure};
use qwen_llm::{
    deepseek_v41::{
        forward::{Forward, State, profile},
        gguf::{ResidentWeights, resident_storage_plan_with_target},
    },
    gguf::GgufFile,
    metal::{self, MetalContext},
    tokenizer::NativeTokenizer,
};
use serde_json::json;
use std::{io::Write, sync::Arc, time::Instant};

fn emit(v: serde_json::Value) {
    println!("{v}");
    std::io::stdout().flush().unwrap();
}
fn thermal() -> Result<u32> {
    let out = std::process::Command::new("notifyutil")
        .args(["-g", "com.apple.system.thermalpressurelevel"])
        .output()?;
    let level = String::from_utf8(out.stdout)?
        .split_whitespace()
        .last()
        .ok_or_else(|| anyhow::anyhow!("thermal output"))?
        .parse()?;
    ensure!(level < 3, "thermal pressure >=3");
    Ok(level)
}
fn cold(f: &Forward<'_>, s: &mut State, ids: &[u32]) -> Result<Vec<f32>> {
    cold_profiled(f, s, ids, None, 0)
}
fn emit_dispatches(
    ctx: &MetalContext,
    guard: metal::DispatchProfileGuard,
    rep: usize,
) -> Result<()> {
    let report = metal::dispatch_profile_finish(ctx, guard)?;
    emit(json!({"kind":"dispatch_summary","overflow":report.overflow,"count":report.rows.len()}));
    ensure!(report.overflow == 0, "dispatch sample capacity exceeded");
    for row in report.rows {
        #[cfg(feature = "perf-ablations")]
        let bindings = row.bindings().to_vec();
        #[cfg(feature = "perf-ablations")]
        let constants = row.constants().to_vec();
        #[cfg(not(feature = "perf-ablations"))]
        let bindings: Vec<(usize, serde_json::Value)> = Vec::new();
        #[cfg(not(feature = "perf-ablations"))]
        let constants: Vec<(usize, Vec<u32>)> = Vec::new();
        emit(
            json!({"kind":"dispatch","rep":rep,"row":row,"bindings":bindings,"constants":constants}),
        );
    }
    Ok(())
}
fn cold_profiled(
    f: &Forward<'_>,
    s: &mut State,
    ids: &[u32],
    ctx: Option<&MetalContext>,
    rep: usize,
) -> Result<Vec<f32>> {
    ensure!(f.begin_cold_prefill_episode(s, ids.len())?, "CED eligible");
    let target = (ids.len() / 4096).saturating_sub(2) * 4096;
    let mut offset = 0;
    loop {
        thermal()?;
        let guard = if offset == target {
            ctx.map(|ctx| metal::dispatch_profile_begin(ctx, 65_536))
                .transpose()?
        } else {
            None
        };
        let slice = f.cold_prefill_episode_slice(s, ids, None, 4096)?;
        if let Some(guard) = guard {
            emit(
                json!({"kind":"chunk_dispatch","start":offset,"rows":slice.encoder_rows,"root_rows":ids.len(),"rep":rep}),
            );
            emit_dispatches(ctx.unwrap(), guard, rep)?;
        }
        offset += slice.encoder_rows;
        if slice.complete {
            return Ok(slice.logits);
        }
    }
}
fn execute(
    f: &Forward<'_>,
    s: &mut State,
    ids: &[u32],
    cold_root: bool,
    experiment: &str,
    control_for: &str,
    dispatch_ctx: Option<&MetalContext>,
    rep: usize,
) -> Result<Vec<f32>> {
    let run = || {
        if cold_root {
            cold_profiled(f, s, ids, dispatch_ctx, rep)
        } else {
            Ok(f.prefill(s, ids, None)?)
        }
    };
    #[cfg(feature = "perf-ablations")]
    {
        use qwen_llm::deepseek_v41::forward::{PrefillExperiment as E, with_prefill_experiment};
        let e = match experiment {
            "baseline" => match control_for {
                "masked-reindex-mid" => E::MaskedReindexMid(false),
                "attention-bundle" => E::AttentionBundle(false),
                "cooperative-experts" => E::CooperativeExperts(false),
                _ => E::Production,
            },
            "masked-reindex-mid" => E::MaskedReindexMid(true),
            "attention-bundle" => E::AttentionBundle(true),
            "cooperative-experts" => E::CooperativeExperts(true),
            _ => anyhow::bail!("experiment"),
        };
        with_prefill_experiment(e, run)
    }
    #[cfg(not(feature = "perf-ablations"))]
    {
        let _ = control_for;
        ensure!(
            experiment == "baseline",
            "experiment requires perf-ablations"
        );
        let mut run = run;
        run()
    }
}
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    ensure!(
        a.len() >= 5,
        "model corpus depth rows [native|stages|dispatch] [reps]"
    );
    let depth: usize = a[3].parse()?;
    let rows: usize = a[4].parse()?;
    let mode = a.get(5).map_or("native", String::as_str);
    let reps: usize = a.get(6).map_or(Ok(1), |v| v.parse())?;
    let experiment = a.get(7).map_or("baseline", String::as_str);
    let paired = experiment.starts_with("paired-");
    let candidate = experiment.strip_prefix("paired-").unwrap_or(experiment);
    ensure!(
        matches!(
            candidate,
            "baseline" | "masked-reindex-mid" | "attention-bundle" | "cooperative-experts"
        ),
        "experiment"
    );
    ensure!(
        matches!(mode, "native" | "stages" | "dispatch"),
        "profile mode"
    );
    ensure!(
        rows > 0 && depth + rows < 98_304 && reps <= 8,
        "bounded shape"
    );
    thermal()?;
    let ctx = MetalContext::new()?;
    let g = Arc::new(GgufFile::open(&a[1])?);
    let tok = NativeTokenizer::from_gguf(&g)?;
    let text = std::fs::read_to_string(&a[2])?;
    let ids: Vec<u32> = tok
        .encode(&text, false)?
        .into_iter()
        .map(|v| v as u32)
        .collect();
    ensure!(ids.len() > depth + rows, "uncycled corpus too short");
    let storage = resident_storage_plan_with_target(
        &g,
        metal::host_page_size_bytes()?,
        ctx.max_buffer_length(),
        Some(4_294_967_296),
    )?;
    let w = ResidentWeights::load_with_storage_plan(&ctx, g, &storage)?;
    let f = Forward::from_resident(&ctx, &w, w.token_map.clone(), w.raw_pad)
        .map_err(anyhow::Error::msg)?;
    emit(
        json!({"kind":"load", "depth":depth,"rows":rows,"mode":mode,"experiment":experiment,
        "corpus_tokens":ids.len(),"thermal":thermal()?,"device":ctx.device_facts().name,
        "config":format!("{:?}",w.config),"ratios":w.schedule.compression_ratios,
        "index_layers":w.schedule.index_source_layers,"kv_layers":w.schedule.kv_source_layers,
        "prefill_rows":f.prefill_chunk_rows()}),
    );
    let mut prefix = f.new_state(98_304)?;
    if depth > 0 {
        let t = Instant::now();
        cold(&f, &mut prefix, &ids[..depth])?;
        emit(
            json!({"kind":"prefix", "wall_ms":t.elapsed().as_secs_f64()*1e3,"thermal":thermal()?}),
        );
    }
    let snap = f.capture_snapshot(&prefix)?;
    drop(prefix);
    #[cfg(feature = "perf-ablations")]
    let mut expected = None;
    for rep in 0..reps * if paired { 2 } else { 1 } {
        let arm = if paired && rep / 2 % 2 == 1 {
            1 - rep % 2
        } else {
            rep % 2
        };
        let policy = if paired && arm == 0 {
            "baseline"
        } else {
            candidate
        };
        let mut s = f.new_state(98_304)?;
        f.restore_snapshot(&mut s, &snap)?;
        let scope = if mode == "stages" {
            profile::Scope::begin_stages().map_err(anyhow::Error::msg)?
        } else {
            profile::Scope::begin().map_err(anyhow::Error::msg)?
        };
        let guard = if mode == "dispatch" && !(depth == 0 && rows > 4096) {
            Some(metal::dispatch_profile_begin(&ctx, 65_536)?)
        } else {
            None
        };
        let t = Instant::now();
        let logits = execute(
            &f,
            &mut s,
            &ids[depth..depth + rows],
            depth == 0 && rows > 4096,
            policy,
            candidate,
            (mode == "dispatch").then_some(&ctx),
            rep,
        )?;
        let wall = t.elapsed().as_secs_f64() * 1e3;
        let report = scope.finish();
        emit(
            json!({"kind":"cell", "rep":rep,"depth":depth,"rows":rows,"mode":mode,"policy":policy,
            "wall_ms":wall,"thermal":thermal()?,"profile":report,
            "logit_digest":blake3::hash(bytemuck::cast_slice(&logits)).to_hex().to_string()}),
        );
        if let Some(guard) = guard {
            emit_dispatches(&ctx, guard, rep)?;
        }
        #[cfg(feature = "perf-ablations")]
        if paired {
            let state = f.capture_snapshot(&s)?.diagnostic_digest();
            let next = f
                .step(&mut s, ids[depth + rows], None)
                .map_err(anyhow::Error::msg)?;
            let signature = (
                blake3::hash(bytemuck::cast_slice(&logits))
                    .to_hex()
                    .to_string(),
                state,
                blake3::hash(bytemuck::cast_slice(&next))
                    .to_hex()
                    .to_string(),
            );
            if expected.is_none() {
                expected = Some(signature.clone());
            }
            let exact = expected.as_ref() == Some(&signature);
            emit(
                json!({"kind":"exact","rep":rep,"policy":policy,"exact":exact,"signature":signature}),
            );
            ensure!(exact, "paired logits/causal state/following decode differ");
        }
    }
    Ok(())
}
