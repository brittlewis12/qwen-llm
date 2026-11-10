//! Read-only GLM DFlash2 acceptance + component-cost probe on frozen code.
//!
//! Runs the production GLM DFlash packet runner over frozen quality-v1 code
//! document tokens at an explicit sampling configuration, and reports
//! acceptance and per-component costs. No engine changes.
//!
//! ```text
//! QWEN_METAL_LEASE_WAIT=1 scripts/bench/with-metal-lease --task spec-econ --for 15m -- \
//!   target/release/examples/spec_dflash_accept TARGET DRAFTER DEPTH TOKENS TEMP SEED [DOC]
//! ```
//!
//! `TEMP=0` is greedy; `TEMP=1` uses GLM's released temperature/nucleus
//! defaults for the target and a temperature-only proposal sampler.
use qwen_llm::{
    gguf::GgufFile,
    glm5_next_metal::*,
    loader::{DFlashTargetShape, open_dflash_drafter_for_target},
    metal::*,
    metal_dflash::MetalDFlashHead,
    sampling::{Sampler, SamplingConfig},
};
use std::{io::Write, time::Instant};

fn emit(v: serde_json::Value) {
    println!("{v}");
    std::io::stdout().flush().unwrap();
}

fn ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1e3
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

fn main() -> anyhow::Result<()> {
    assert!(
        std::env::var_os("QWEN_METAL_LEASE_FD").is_some(),
        "canonical Metal lease required"
    );
    let args: Vec<_> = std::env::args().collect();
    let target = args
        .get(1)
        .expect("TARGET DRAFTER DEPTH TOKENS TEMP SEED [DOC]");
    let drafter = &args[2];
    let depth: usize = args[3].parse()?;
    let new_tokens: usize = args[4].parse()?;
    let temperature: f32 = args[5].parse()?;
    let seed: u64 = args[6].parse()?;
    let doc_index: usize = args.get(7).map(|s| s.parse()).transpose()?.unwrap_or(8);

    let ctx = MetalContext::new()?;
    let target_gguf = GgufFile::open(target)?;
    let weights = Glm5NextWeights::load(&ctx, &target_gguf)?;
    let drafter_gguf = GgufFile::open(drafter)?;
    let description = open_dflash_drafter_for_target(
        &drafter_gguf,
        DFlashTargetShape {
            hidden_size: weights.config.hidden_size,
            vocab_size: weights.config.vocab_size,
            n_layer: weights.config.blocks.len() as u32,
        },
    )?;
    let head = MetalDFlashHead::load(&ctx, &drafter_gguf, &description)?;

    let frozen = include_bytes!("../../../scripts/reference/glm53/quality-v1.json");
    let f: serde_json::Value = serde_json::from_slice(frozen)?;
    let doc: Vec<u32> = f["documents"][doc_index]["tokens"]
        .as_str()
        .unwrap()
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    let prompt: Vec<u32> = doc.iter().copied().cycle().take(depth).collect();
    let block = head.config.block_size as usize;
    let capacity = prompt.len() + new_tokens + block;
    // Prompt prefill chunk width; must also cover the packed verify block.
    let rows = capacity.min(2048).max(block);

    let target_sampler = if temperature <= 0.0 {
        Sampler::new(SamplingConfig::default())?
    } else {
        Sampler::new(SamplingConfig::glm5_next(seed))?
    };
    let proposal = (temperature > 0.0)
        .then(|| {
            Sampler::new(SamplingConfig {
                temperature,
                top_k: 0,
                top_p: 1.0,
                min_p: 0.0,
                seed: seed ^ 0x9e37_79b9_7f4a_7c15,
            })
        })
        .transpose()?;

    emit(serde_json::json!({
        "event": "header",
        "depth": depth,
        "new_tokens": new_tokens,
        "temperature": temperature,
        "seed": seed,
        "doc_index": doc_index,
        "block": block,
        "target_layers": head.target_layer_ids,
        "selector": head.selector.is_some(),
        "device": ctx.device_facts(),
        "thermal_before": thermal(),
    }));

    let mut session = Glm5NextSession::with_prefill_rows(&ctx, &weights, capacity, rows)?;
    let options = Glm5NextDFlashOptions {
        prompt_tokens: prompt.len(),
        max_new_tokens: new_tokens,
        stop_ids: Vec::new(),
        verification: Glm5NextDFlashVerification::ConfiguredPacked,
    };
    let t = Instant::now();
    let mut runner =
        Glm5NextDFlashRunner::new(&ctx, &mut session, &head, options, target_sampler, proposal)?;
    let first = runner.prefill(&prompt, &mut || false)?;
    let prefill_ms = t.elapsed().as_secs_f64() * 1e3;

    let mut emitted = first.emitted_tokens.len();
    let mut accepted = first.accepted_drafts;
    let mut packets = 1usize;
    let mut carry = first.next_carry.map(|c| c.token());
    let mut stop = first.stop_reason;
    let mut draft = ms(first.costs.draft);
    let mut verify = ms(first.costs.verify);
    let mut checkpoint = ms(first.costs.checkpoint);
    let mut commit = ms(first.costs.commit);
    let mut append = ms(first.costs.append);
    emit(serde_json::json!({
        "event":"packet","step":0,"role":"initial","emitted":emitted,
        "accepted":accepted,"carry":carry.is_some(),
        "draft_ms":ms(first.costs.draft),"verify_ms":ms(first.costs.verify),
        "checkpoint_ms":ms(first.costs.checkpoint),"commit_ms":ms(first.costs.commit),
        "append_ms":ms(first.costs.append),"total_ms":ms(first.costs.total),
    }));

    let mut steps = Vec::new();
    while stop.is_none() {
        let Some(c) = carry else { break };
        let p = runner.step(c, &mut || false)?;
        emitted += p.emitted_tokens.len();
        accepted += p.accepted_drafts;
        packets += 1;
        draft += ms(p.costs.draft);
        verify += ms(p.costs.verify);
        checkpoint += ms(p.costs.checkpoint);
        commit += ms(p.costs.commit);
        append += ms(p.costs.append);
        steps.push(serde_json::json!({
            "step":steps.len()+1,
            "emitted":p.emitted_tokens.len(),
            "accepted":p.accepted_drafts,
            "rows":p.retain_input_count,
            "anchor_only":p.anchor_only,
            "draft_ms":ms(p.costs.draft),"verify_ms":ms(p.costs.verify),
            "checkpoint_ms":ms(p.costs.checkpoint),"commit_ms":ms(p.costs.commit),
            "append_ms":ms(p.costs.append),"total_ms":ms(p.costs.total),
        }));
        carry = p.next_carry.map(|c| c.token());
        stop = p.stop_reason;
    }

    let c = runner.counters();
    emit(serde_json::json!({
        "event":"summary",
        "prefill_ms":prefill_ms,
        "packets":packets,
        "emitted":emitted,
        "accepted_drafts":accepted,
        "drafted_tokens":c.drafted_tokens,
        "verified_rows":c.verified_rows,
        "serial_verifications":c.serial_verifications,
        "anchor_only_packets":c.anchor_only_packets,
        "emitted_per_packet":emitted as f64 / packets as f64,
        "alpha_per_draft": if c.drafted_tokens>0 { accepted as f64 / c.drafted_tokens as f64 } else { 0.0 },
        "draft_ms":draft,"verify_ms":verify,"checkpoint_ms":checkpoint,
        "commit_ms":commit,"append_ms":append,
        "thermal_after":thermal(),
    }));
    for s in steps {
        emit(serde_json::json!({"event":"step","detail":s}));
    }
    emit(serde_json::json!({"event":"complete"}));
    Ok(())
}
