//! Request wall measurements, not GPU-only pp/tg timings.
use anyhow::{Context, Result, ensure};
use clap::{ArgGroup, Parser};
use qwen_llm::gguf::GgufFile;
use qwen_llm::k2_horizon_runtime::{K2ArtifactLayout, K2LoadedModel, K2RuntimePlan};
use qwen_llm::metal::{MetalContext, host_page_size_bytes};
use qwen_llm::model_family::ModelFamily;
use qwen_llm::sampling::{SAMPLER_ALGORITHM_VERSION, Sampler, SamplingConfig};
use qwen_llm::tokenizer::{NativeTokenizer, token_ids_sha256_i32le};
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(group(ArgGroup::new("input").required(true).multiple(false).args(["raw_prompt", "token_ids"])))]
pub struct K2RequestArgs {
    /// Dense K2 7B GGUF; other families are rejected before Metal.
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Raw text, without a chat template; native special insertion by default.
    #[arg(long, allow_hyphen_values = true)]
    raw_prompt: Option<String>,
    /// Literal token IDs; no automatic specials.
    #[arg(long, value_delimiter = ',')]
    token_ids: Option<Vec<i32>>,
    #[arg(long, requires = "raw_prompt", conflicts_with = "token_ids")]
    no_special_tokens: bool,
    /// Maximum sampled tokens, including EOS; prompt+tokens-1 must fit model context.
    #[arg(long)]
    tokens: usize,
    /// Fresh session capacity; defaults to prompt+tokens-1, subject to memory admission.
    #[arg(long)]
    capacity: Option<usize>,
    /// Timed repetitions; each creates fresh KV and a fresh greedy sampler.
    #[arg(long, default_value_t = 3)]
    runs: usize,
    /// Omit the single full-request warmup; no steady-state claim either way.
    #[arg(long)]
    no_warmup: bool,
    #[arg(short = 'o', long, value_enum, default_value = "text")]
    output: super::OutputFormat,
}

fn budget(prompt: usize, sampled: usize, capacity: Option<usize>, context: u32) -> Result<usize> {
    ensure!(
        prompt > 0 && sampled > 0,
        "K2 requires nonempty input and positive --tokens"
    );
    let required = prompt
        .checked_add(sampled - 1)
        .context("K2 forward budget overflow")?;
    let capacity = capacity.unwrap_or(required);
    ensure!(
        required <= capacity && capacity <= context as usize,
        "K2 requires {required} forwards; capacity {capacity} must fit checkpoint context {context}"
    );
    Ok(capacity)
}

fn checked_id(id: i32, vocab: u32) -> Result<u32> {
    ensure!(
        id >= 0 && (id as u32) < vocab,
        "K2 token ID {id} outside vocabulary {vocab}"
    );
    Ok(id as u32)
}

#[cfg(test)]
use qwen_llm::k2_horizon_runtime::validate_generation_stops as stops;

fn elapsed(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

#[derive(Debug)]
struct Generation {
    sampled: Vec<i32>,
    emitted: Vec<i32>,
    wall_ns: u64,
    transition_forwards: usize,
    transition_wall_ns: u64,
    first_sample_ready_request_ns: u64,
    termination: &'static str,
}

#[allow(clippy::too_many_arguments)]
fn generate(
    mut logits: Vec<f32>,
    limit: usize,
    vocab: u32,
    request_started: Instant,
    mut select: impl FnMut(&[f32]) -> Result<i32>,
    mut transition: impl FnMut(u32) -> Result<Vec<f32>>,
    mut emit: impl FnMut(i32) -> Result<()>,
    mut checkpoint: impl FnMut() -> Result<()>,
) -> Result<Generation> {
    ensure!(limit > 0, "invalid K2 sample limit");
    let started = Instant::now();
    let mut sampled = Vec::with_capacity(limit);
    let mut emitted = Vec::with_capacity(limit);
    let mut transition_forwards = 0;
    let mut transition_wall_ns = 0;
    let mut first_sample_ready_request_ns = None;
    let termination = loop {
        checkpoint()?;
        let token = select(&logits)?;
        first_sample_ready_request_ns.get_or_insert_with(|| elapsed(request_started));
        let id = checked_id(token, vocab)?;
        sampled.push(token);
        if token == 1 {
            break "eos";
        }
        emit(token)?;
        emitted.push(token);
        if sampled.len() == limit {
            break "token_limit";
        }
        checkpoint()?;
        let transition_started = Instant::now();
        logits = transition(id)?;
        transition_wall_ns += elapsed(transition_started);
        transition_forwards += 1;
    };
    Ok(Generation {
        sampled,
        emitted,
        wall_ns: elapsed(started),
        transition_forwards,
        transition_wall_ns,
        first_sample_ready_request_ns: first_sample_ready_request_ns.unwrap(),
        termination,
    })
}

#[derive(Debug, PartialEq, Eq, Serialize)]
struct Outcome {
    sampled_token_ids: Vec<i32>,
    emitted_token_ids: Vec<i32>,
    sampled_token_ids_sha256_i32le: String,
    emitted_bytes_sha256: String,
    emitted_bytes_hex: String,
    emitted_text_lossy: String,
    emitted_utf8_valid: bool,
    termination: &'static str,
    transition_forwards: usize,
}

#[derive(Debug, Serialize)]
struct Sample {
    repetition: usize,
    session_allocation_wall_ns: u64,
    sampler_setup_wall_ns: u64,
    prefill_wall_ns: u64,
    generation_wall_ns: u64,
    transition_forward_wall_ns: u64,
    first_sample_ready_request_wall_ns: u64,
    request_wall_ns: u64,
    prompt_forwards: usize,
    committed_positions: u32,
    observed_session_metal_allocation_delta_bytes: u64,
    outcome: Outcome,
}

fn iteration(
    repetition: usize,
    model: &K2LoadedModel<'_>,
    ctx: &MetalContext,
    tokenizer: &NativeTokenizer,
    prompt: &[u32],
    limit: usize,
) -> Result<Sample> {
    super::shutdown::checkpoint()?;
    let request_started = Instant::now();
    let started = Instant::now();
    let before = ctx.current_allocated_size();
    let mut session = model.create_session(0)?;
    let allocation_delta = ctx.current_allocated_size().saturating_sub(before);
    let session_ns = elapsed(started);
    let started = Instant::now();
    let mut logits = Vec::new();
    let mut chunks = prompt
        .chunks(model.prefill_info(prompt.len()).chunk_tokens)
        .peekable();
    while let Some(chunk) = chunks.next() {
        super::shutdown::checkpoint()?;
        if chunks.peek().is_none() {
            logits = session.append(chunk)?;
        } else {
            session.advance(chunk)?;
        }
    }
    let prefill_ns = elapsed(started);
    let started = Instant::now();
    let mut sampler = Sampler::new(SamplingConfig::default())?;
    let sampler_ns = elapsed(started);
    let mut bytes = Vec::new();
    let generation = generate(
        logits,
        limit,
        model.config().vocab_size,
        request_started,
        |logits| Ok(sampler.sample(logits)?.token),
        |id| Ok(session.append(&[id])?),
        |id| {
            bytes.extend_from_slice(tokenizer.try_decode_piece_bytes_exact(id)?);
            Ok(())
        },
        super::shutdown::checkpoint,
    )?;
    let request_ns = elapsed(request_started);
    let committed_positions = session.committed_len();
    drop(session);
    ensure!(
        generation.transition_forwards + 1 == generation.sampled.len(),
        "K2 sample/transition accounting drift"
    );
    ensure!(
        committed_positions as usize == prompt.len() + generation.transition_forwards,
        "K2 committed prefix accounting drift"
    );
    let outcome = Outcome {
        sampled_token_ids_sha256_i32le: token_ids_sha256_i32le(&generation.sampled),
        emitted_bytes_sha256: format!("{:x}", Sha256::digest(&bytes)),
        emitted_bytes_hex: bytes.iter().map(|b| format!("{b:02x}")).collect(),
        emitted_text_lossy: String::from_utf8_lossy(&bytes).into_owned(),
        emitted_utf8_valid: std::str::from_utf8(&bytes).is_ok(),
        sampled_token_ids: generation.sampled,
        emitted_token_ids: generation.emitted,
        termination: generation.termination,
        transition_forwards: generation.transition_forwards,
    };
    Ok(Sample {
        repetition,
        session_allocation_wall_ns: session_ns,
        sampler_setup_wall_ns: sampler_ns,
        prefill_wall_ns: prefill_ns,
        generation_wall_ns: generation.wall_ns,
        transition_forward_wall_ns: generation.transition_wall_ns,
        first_sample_ready_request_wall_ns: generation.first_sample_ready_request_ns,
        request_wall_ns: request_ns,
        prompt_forwards: prompt.len(),
        committed_positions,
        observed_session_metal_allocation_delta_bytes: allocation_delta,
        outcome,
    })
}

fn rate(count: usize, ns: u128) -> Option<f64> {
    (count > 0 && ns > 0).then(|| count as f64 * 1e9 / ns as f64)
}

fn aggregate(warmup: Option<&Sample>, samples: &[Sample]) -> (bool, Option<Value>) {
    let Some(first) = samples.first() else {
        return (false, None);
    };
    let consistent = samples
        .iter()
        .chain(warmup)
        .all(|s| s.outcome == first.outcome && s.committed_positions == first.committed_positions);
    let aggregate = consistent.then(|| json!({
        "transition_forwards_per_second":rate(samples.iter().map(|s| s.outcome.transition_forwards).sum(), samples.iter().map(|s| s.transition_forward_wall_ns as u128).sum()),
        "prompt_forwards_per_second":rate(samples.iter().map(|s| s.prompt_forwards).sum(), samples.iter().map(|s| s.prefill_wall_ns as u128).sum()),
        "mean_request_wall_ns":samples.iter().map(|s| s.request_wall_ns as f64).sum::<f64>() / samples.len() as f64,
        "mean_first_sample_ready_request_wall_ns":samples.iter().map(|s| s.first_sample_ready_request_wall_ns as f64).sum::<f64>() / samples.len() as f64,
    }));
    (consistent, aggregate)
}

pub fn run(args: K2RequestArgs) -> Result<()> {
    ensure!(args.runs > 0, "--runs must be positive");
    ensure!(args.tokens > 0, "--tokens must be positive");
    let started = Instant::now();
    let gguf = GgufFile::open(&args.model)?;
    let open_ns = elapsed(started);
    ensure!(
        ModelFamily::detect(&gguf) == Some(ModelFamily::K2Horizon),
        "k2-request requires a dense K2 Horizon model"
    );
    let started = Instant::now();
    let layout = K2ArtifactLayout::inspect(&gguf)?;
    let bind_ns = elapsed(started);
    let started = Instant::now();
    let artifact = layout.prepare_tokenizer()?;
    artifact.generation_stops()?;
    let config = artifact.config().clone();
    let tokenizer = artifact.into_tokenizer();
    let tokenizer_ns = elapsed(started);
    let started = Instant::now();
    let ids = if let Some(text) = &args.raw_prompt {
        ensure!(!text.is_empty(), "--raw-prompt must not be empty");
        tokenizer.encode(text, !args.no_special_tokens)?
    } else {
        ensure!(
            !args.no_special_tokens,
            "--no-special-tokens requires --raw-prompt"
        );
        args.token_ids
            .clone()
            .context("raw prompt or literal token IDs required")?
    };
    let tokenization_ns = elapsed(started);
    let prompt = ids
        .iter()
        .map(|&id| checked_id(id, config.vocab_size))
        .collect::<Result<Vec<_>>>()?;
    let capacity = budget(
        prompt.len(),
        args.tokens,
        args.capacity,
        config.context_length,
    )?;
    let started = Instant::now();
    K2RuntimePlan::inspect(
        &gguf,
        u32::try_from(capacity)?,
        host_page_size_bytes()?,
        usize::MAX,
    )?;
    let preflight_plan_ns = elapsed(started);
    let head = gguf
        .tensors
        .iter()
        .find(|tensor| tensor.name == "output.weight")
        .context("missing bound K2 output head")?;
    let stamps = gguf.revalidate_retained_shard_stamps()?;
    let source_stamps = stamps.iter().map(|s| json!({"shard":s.shard_idx,"path":s.path,"device":s.device,"inode":s.inode,
        "size":s.size,"mtime_sec":s.mtime_sec,"mtime_nsec":s.mtime_nsec,"ctime_sec":s.ctime_sec,"ctime_nsec":s.ctime_nsec})).collect::<Vec<_>>();
    let power = super::capture_power_snapshot();
    super::shutdown::checkpoint()?;
    let started = Instant::now();
    let ctx = MetalContext::new()?;
    let context_ns = elapsed(started);
    let started = Instant::now();
    let plan = K2RuntimePlan::inspect(
        &gguf,
        u32::try_from(capacity)?,
        host_page_size_bytes()?,
        ctx.max_buffer_length(),
    )?;
    let plan_ns = preflight_plan_ns.saturating_add(elapsed(started));
    let started = Instant::now();
    let before = ctx.current_allocated_size();
    let model = K2LoadedModel::load(&ctx, &gguf, u32::try_from(capacity)?)?;
    let load_ns = elapsed(started);
    let weight_delta = ctx.current_allocated_size().saturating_sub(before);
    let warmup = if args.no_warmup {
        None
    } else {
        Some(iteration(
            0,
            &model,
            &ctx,
            &tokenizer,
            &prompt,
            args.tokens,
        )?)
    };
    let samples = (0..args.runs)
        .map(|i| iteration(i + 1, &model, &ctx, &tokenizer, &prompt, args.tokens))
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        gguf.revalidate_retained_shard_stamps()? == stamps,
        "K2 model source changed during benchmark"
    );
    let (consistent, aggregates) = aggregate(warmup.as_ref(), &samples);
    let report = json!({"schema":"qwen.k2_horizon.request_benchmark", "schema_version":1,
        "build_identity":super::recorded_build_identity(), "device":ctx.describe(), "qwen_env":super::capture_qwen_env(), "power":power,
        "instrumentation":{"debug_assertions":cfg!(debug_assertions),"MTL_DEBUG_LAYER":std::env::var("MTL_DEBUG_LAYER").ok(),"MTL_SHADER_VALIDATION":std::env::var("MTL_SHADER_VALIDATION").ok()},
        "model":{"path":args.model,"architecture":"k2-horizon","layers":config.layer_count,"hidden_size":config.hidden_size,"vocab_size":config.vocab_size,
            "checkpoint_context_length":config.context_length,"rope_theta":config.rope_theta,"kv_storage":"f16","start_position":0,
            "execution_capacity":capacity,"output_head_dtype":format!("{:?}",head.dtype),"output_head_shape":head.shape,
            "query_heads":config.query_head_count,"kv_heads":config.kv_head_count,"head_dim":config.key_head_dim,"norm_groups":config.norm_groups,
            "weight_dtype_counts":gguf.tensors.iter().fold(std::collections::BTreeMap::<String,usize>::new(), |mut counts,t| { *counts.entry(format!("{:?}",t.dtype)).or_default() += 1; counts }),
            "tokenizer_metadata_id":format!("{:016x}",qwen_llm::runtime::tokenizer_metadata_identity(&gguf)),"source_stamps":source_stamps,"identity_scope":"file_freshness_not_content_authentication",
            "weight_payload_bytes":plan.weight_payload_bytes(),"planned_retained_window_bytes":plan.weight_buffer_bytes()?.iter().sum::<u64>(),
            "planned_session_buffer_bytes":plan.session_buffer_bytes().iter().sum::<u64>(),"logical_kv_bytes":147456 * capacity as u64,
            "observed_weight_metal_allocation_delta_bytes":weight_delta},
        "request":{"input_kind":if args.raw_prompt.is_some(){"raw_prompt"}else{"token_ids"},"add_special_tokens":args.raw_prompt.as_ref().map(|_|!args.no_special_tokens),
            "prompt_token_ids":ids,"prompt_token_ids_sha256_i32le":token_ids_sha256_i32le(&ids),"capacity":capacity,"maximum_sampled_tokens":args.tokens,"stops":[1]},
        "sampling":{"method":"greedy","algorithm_version":SAMPLER_ALGORITHM_VERSION,"temperature":0,"top_k":0,"top_p":1,"min_p":0,"seed":0},
        "setup":{"gguf_open_ns":open_ns,"profile_bind_ns":bind_ns,"tokenizer_open_ns":tokenizer_ns,"tokenization_ns":tokenization_ns,
            "runtime_plan_ns":plan_ns,"metal_context_ns":context_ns,"resident_model_load_ns":load_ns},
        "method":{"clock":"host_monotonic_wall_ns","session_scope":"fresh_allocation_each_repetition",
            "prefill":"bounded_cancellable_chunk_appends","prefill_execution":model.prefill_info(prompt.len()),
            "readout":"final_prompt_only_one_head_and_logits_download",
            "warmup_repetitions":usize::from(warmup.is_some()),"timed_repetitions":args.runs,
            "append_includes":"encoding_submission_wait_residual_kv_finite_checks_source_checks_optional_final_logits_readback",
            "request_boundary":"before_session_allocation_to_after_generation_before_cleanup_hashes_serialization",
            "generation_includes":"sampling_native_piece_decoding_transition_appends_checkpoint_checks",
            "first_sample_ready":"direct_request_clock_after_first_selection_not_network_ttft",
            "transition_rate":"actual_transition_appends_per_transition_append_wall_second_excludes_first_sample",
            "prompt_rate":"prompt_positions_per_append_wall_second"},
        "qualification":{"all_repetitions_identical":consistent,"steady_state_qualified":false,"llama_bench_comparable":false,"kernel_only":false,
            "runtime_scope":"k2_dense_request_wall","capacity_policy":"checkpoint_context_and_device_memory","performance_claim":false},
        "warmup":warmup,"samples":samples,"aggregate":aggregates});
    match args.output {
        super::OutputFormat::Json => println!("{}", serde_json::to_string_pretty(&report)?),
        super::OutputFormat::Text => {
            println!(
                "K2 request wall: prompt={} max_sampled={} capacity={} runs={} identical={consistent}",
                prompt.len(),
                args.tokens,
                capacity,
                args.runs
            );
            println!("{}", serde_json::to_string_pretty(&report["aggregate"])?);
            println!(
                "Includes host guards/readback; not GPU-only or llama-bench comparable. Use --output json for full evidence."
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
