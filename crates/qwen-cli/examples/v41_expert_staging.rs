//! Paired NativeF32 experiment; caller records original source/revision and tokens.
use anyhow::{Context, Result, anyhow, ensure};
use qwen_llm::deepseek_v41::{
    forward::{Forward, profile},
    loader::{LoadLimits, ResidentWeights, SourceStore},
};
use qwen_llm::metal::MetalContext;
use qwen_llm::pid_metrics::PidSnapshot;
use serde_json::json;
use std::{path::PathBuf, sync::Arc};

fn timing(report: &profile::Report) -> serde_json::Value {
    let submit: f64 = report
        .operations
        .iter()
        .map(|o| o.encode_ms + o.commit_wait_ms)
        .sum();
    let gpu: Option<f64> = report
        .operations
        .iter()
        .filter_map(|o| o.gpu_ms)
        .reduce(|a, b| a + b);
    json!({"wall_ms":report.token_wall_ms,"gpu_ms":gpu,"submit_wall_ms":submit,
        "cpu_outside_submit_wall_ms":report.token_wall_ms-submit,"operations":report.operations})
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        (3..=4).contains(&args.len()),
        "usage: v41_expert_staging ROOT TOKEN_IDS_COMMA_SEPARATED CONTEXT [BACKING_DIR]"
    );
    let root = PathBuf::from(&args[0]);
    let tokens: Vec<u32> = args[1]
        .split(',')
        .map(str::parse)
        .collect::<std::result::Result<_, _>>()?;
    let context: usize = args[2].parse()?;
    ensure!(
        !tokens.is_empty() && tokens.len() <= context,
        "tokens must fit context"
    );
    let backing = args
        .get(3)
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    std::fs::create_dir_all(&backing)?;
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("config.json"))?)?;
    let tokenizer = tokenizers::Tokenizer::from_file(root.join("tokenizer.json"))
        .map_err(|e| anyhow!(e.to_string()))?;
    let map =
        qwen_cli::deepseek_v41_tokenizer::CompressedTokenMap::for_config(&tokenizer, &config)?;
    ensure!(
        tokens.iter().all(|&t| (t as usize) < map.lookup.len()),
        "token outside vocabulary"
    );
    let source = SourceStore::open(&root).map_err(|e| anyhow!(e))?;
    let engram = Arc::new(SourceStore::open_engram(&root).map_err(|e| anyhow!(e))?);
    let plan = ResidentWeights::plan(&source, &engram, &config).map_err(|e| anyhow!(e))?;
    let ctx = MetalContext::new()?;
    let weights = ResidentWeights::load(
        &ctx,
        &source,
        engram,
        &config,
        LoadLimits {
            resident_bytes: plan.resident_bytes(),
        },
        &backing,
    )
    .map_err(|e| anyhow!(e))?;
    drop(source);
    let pad = u32::try_from(
        config["text_config"]["engram_pad_token_id"]
            .as_u64()
            .context("missing pad ID")?,
    )?;
    let model = Forward::from_resident(&ctx, &weights, map.lookup, pad).map_err(|e| anyhow!(e))?;
    let legacy_estimate = model
        .estimate_request_memory(tokens.len())
        .map_err(|e| anyhow!(e))?;
    let staged_estimate = model
        .estimate_request_memory_with_expert_staging(tokens.len())
        .map_err(|e| anyhow!(e))?;
    let admit = || -> Result<()> {
        let decision = qwen_llm::metal::evaluate_metal_memory_admission_with_cpu_bytes(
            legacy_estimate
                .metal_bytes
                .checked_add(staged_estimate.metal_bytes)
                .context("paired Metal size overflow")?,
            legacy_estimate
                .cpu_bytes
                .checked_add(staged_estimate.cpu_bytes)
                .context("paired CPU size overflow")?,
            0,
            ctx.memory_signals(),
            true,
        );
        if let Some(reason) = decision.refusal() {
            return Err(anyhow!("paired request admission: {reason}"));
        }
        Ok(())
    };
    admit()?;
    let mut legacy = model.new_state(context).map_err(|e| anyhow!(e))?;
    let mut staged = model
        .new_state_with_expert_staging(context)
        .map_err(|e| anyhow!(e))?;
    println!(
        "{}",
        json!({"experiment":"NativeF32 active expert staging","root":root,"backing_dir":backing,
        "tokens":tokens,"context":context,"source_provenance":"caller must record original checkpoint revision",
        "legacy_request_bytes":model.estimated_request_bytes(tokens.len()).map_err(|e| anyhow!(e))?,
        "staged_request_bytes":model.estimate_request_memory_with_expert_staging(tokens.len()).map_err(|e| anyhow!(e))?.total_bytes})
    );
    for (position, &token) in tokens.iter().enumerate() {
        let run = |state: &mut _| -> Result<_> {
            admit()?;
            let before = PidSnapshot::now()?;
            let scope = profile::Scope::begin().map_err(|e| anyhow!(e))?;
            let logits = model.step(state, token, None).map_err(|e| anyhow!(e))?;
            let profile = scope.finish();
            let after = PidSnapshot::now()?;
            Ok((
                logits,
                profile,
                after
                    .diskio_bytesread
                    .saturating_sub(before.diskio_bytesread),
                after.pageins.saturating_sub(before.pageins),
            ))
        };
        let (a, b) = if position % 2 == 0 {
            (run(&mut legacy)?, run(&mut staged)?)
        } else {
            let b = run(&mut staged)?;
            (run(&mut legacy)?, b)
        };
        let top1 = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0
        };
        println!(
            "{}",
            json!({"position":position,"token":token,"order":if position%2==0 {"AB"} else {"BA"},
            "legacy":timing(&a.1),"staged":timing(&b.1),
             "legacy_disk_read_bytes":a.2,"staged_disk_read_bytes":b.2,
             "legacy_pageins":a.3,"staged_pageins":b.3,
             "logit_maxabs":qwen_llm::compare::assert_max_abs_diff_f32("expert staging", a.0.iter(), b.0.iter()),
            "bit_identity":a.0.iter().zip(&b.0).all(|(a,b)|a.to_bits()==b.to_bits()),
            "top1_legacy":top1(&a.0),"top1_staged":top1(&b.0)})
        );
    }
    Ok(())
}
