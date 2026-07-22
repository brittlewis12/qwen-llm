//! CPU-only census for loader source coverage and reconstructed prompt reuse.

#[allow(dead_code)]
mod messages;

use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, ValueEnum};
use messages::{
    ChatMessage, MessagesThinkingMode, messages_auto_preserve_thinking, parse_messages_input,
    render_qwen_messages_prompt, strip_think,
};
use qwen_llm::gguf::GgufFile;
use qwen_llm::loader::Model;
use qwen_llm::metal_forward::{
    ModelWeightStorageKind, ModelWeightStorageRequest, gguf_descriptor_layout_digest,
    model_weight_storage_inventory_digest, model_weight_storage_requests,
    mtp_weight_source_descriptors, production_native_quant_embedding_storage_enabled,
};
use qwen_llm::tokenizer::Tokenizer;
use serde_json::{Value, json};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::{Path, PathBuf};

const SCHEMA_VERSION: u32 = 1;
const DEFAULT_PAGE_SIZE: u64 = 16 * 1024;
const SCREENING_NUMERATOR: u64 = 9;
const SCREENING_DENOMINATOR: u64 = 10;

#[derive(Parser, Debug)]
#[command(
    name = "qwen-census",
    version,
    about = "price loader source coverage and reconstructed prompt reuse"
)]
struct Args {
    /// Qwen GGUF whose base-weight plan and tokenizer define the census.
    #[arg(short = 'm', long)]
    model: PathBuf,

    /// Messages JSON file. Repeat to price cross-file prefix reuse.
    #[arg(long = "messages", value_name = "FILE")]
    messages: Vec<PathBuf>,

    /// Page granularity used for rounded source coverage.
    #[arg(long, default_value_t = DEFAULT_PAGE_SIZE)]
    page_size: u64,

    /// Model the opt-in F16 MoE router storage plan.
    #[arg(long)]
    router_f16: bool,

    /// Thinking policy used by the shared Qwen messages renderer.
    #[arg(long, value_enum, default_value_t = ThinkingPolicy::Auto)]
    thinking: ThinkingPolicy,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ThinkingPolicy {
    Auto,
    Preserve,
    Strip,
}

impl ThinkingPolicy {
    fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Preserve => "preserve",
            Self::Strip => "strip",
        }
    }

    fn messages_mode(self) -> MessagesThinkingMode {
        match self {
            Self::Auto => MessagesThinkingMode::Auto,
            Self::Preserve => MessagesThinkingMode::Preserve,
            Self::Strip => MessagesThinkingMode::Strip,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Interval {
    start: u64,
    end: u64,
}

impl Interval {
    fn len(self) -> u64 {
        self.end - self.start
    }
}

#[derive(Debug)]
struct GapStats {
    count: usize,
    bytes: u64,
    largest_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CheckpointKind {
    Prompt,
    CompletedProxy,
}

impl CheckpointKind {
    fn label(self) -> &'static str {
        match self {
            Self::Prompt => "reconstructed_prompt",
            Self::CompletedProxy => "retokenized_completed_proxy",
        }
    }

    fn confidence(self) -> &'static str {
        match self {
            Self::Prompt => "reconstructed_request",
            Self::CompletedProxy => "proxy_not_authoritative_generated_tokens",
        }
    }

    fn tie_rank(self) -> u8 {
        match self {
            Self::Prompt => 0,
            Self::CompletedProxy => 1,
        }
    }
}

#[derive(Debug)]
struct ConversationTurn {
    file_ordinal: usize,
    file: PathBuf,
    turn_index: usize,
    assistant_message_index: usize,
    prompt_text: String,
    prompt_tokens: Vec<i32>,
    completed_text: String,
    completed_tokens: Vec<i32>,
}

#[derive(Debug)]
struct ConversationFile {
    ordinal: usize,
    path: PathBuf,
    metadata_model: Option<String>,
    metadata_model_matches_loaded_path: Option<bool>,
    preserve_thinking: bool,
    thinking_resolution: String,
    transformed_assistant_messages: usize,
    role_anomalies: Vec<String>,
    turns: Vec<ConversationTurn>,
}

#[derive(Clone, Copy)]
struct Candidate<'a> {
    turn: &'a ConversationTurn,
    kind: CheckpointKind,
}

impl<'a> Candidate<'a> {
    fn text(self) -> &'a str {
        match self.kind {
            CheckpointKind::Prompt => &self.turn.prompt_text,
            CheckpointKind::CompletedProxy => &self.turn.completed_text,
        }
    }

    fn tokens(self) -> &'a [i32] {
        match self.kind {
            CheckpointKind::Prompt => &self.turn.prompt_tokens,
            CheckpointKind::CompletedProxy => &self.turn.completed_tokens,
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.page_size == 0 {
        bail!("--page-size must be nonzero");
    }

    let gguf = GgufFile::open(&args.model)
        .with_context(|| format!("open GGUF {}", args.model.display()))?;
    let model = Model::from_gguf(&gguf).context("bind model")?;
    let native_embedding = production_native_quant_embedding_storage_enabled(&model);
    let requests = model_weight_storage_requests(&model, native_embedding, args.router_f16)
        .context("construct base-model storage request plan")?;
    let loader = loader_census(
        &gguf,
        &model,
        &requests,
        native_embedding,
        args.router_f16,
        args.page_size,
    )?;

    let conversations = if args.messages.is_empty() {
        Value::Null
    } else {
        let tokenizer = Tokenizer::from_gguf(&gguf).context("construct native GGUF tokenizer")?;
        conversation_census(&args.model, &args.messages, args.thinking, &tokenizer)?
    };

    let ignored_environment_overrides = ["QWEN_NATIVE_QUANT_EMBED", "QWEN_MOE_ROUTER_F16"]
        .into_iter()
        .filter_map(|name| std::env::var_os(name).map(|_| name))
        .collect::<Vec<_>>();
    let document = json!({
        "schema_version": SCHEMA_VERSION,
        "tool": {
            "name": "qwen-census",
            "package_version": env!("CARGO_PKG_VERSION"),
            "build_commit": env!("QWEN_BUILD_COMMIT"),
            "build_dirty": env!("QWEN_BUILD_DIRTY"),
        },
        "semantics": {
            "intervals": "half_open",
            "page_rounding": "round_each_merged_interval_outward_then_clip_per_shard",
            "page_size_bytes": args.page_size,
            "screening_threshold": {
                "comparison": "rounded_tensor_region_bytes * 10 > tensor_region_bytes * 9",
                "strict": true,
                "numerator": SCREENING_NUMERATOR,
                "denominator": SCREENING_DENOMINATOR,
                "drop_label": "drop_range_only_warmer",
            },
            "tokenizer_backend": "native_gguf_qwen35",
            "tokenizer_add_special": false,
            "prompt_checkpoint_scope": "full_reconstructed_prompt_not_historical_publication_proof",
            "completed_checkpoint_scope": "retokenized_proxy_not_authoritative_generated_tokens",
            "thinking_policy_requested": args.thinking.label(),
            "ignored_environment_overrides": ignored_environment_overrides,
        },
        "model": {
            "path": args.model,
            "architecture": gguf.architecture(),
            "descriptor_layout_digest": format!("{:#018x}", gguf_descriptor_layout_digest(&gguf)),
        },
        "loader": loader,
        "conversations": conversations,
    });
    serde_json::to_writer_pretty(std::io::stdout().lock(), &document)
        .context("write census JSON")?;
    println!();
    Ok(())
}

fn loader_census(
    gguf: &GgufFile,
    model: &Model<'_>,
    requests: &[ModelWeightStorageRequest<'_>],
    native_embedding: bool,
    router_f16: bool,
    page_size: u64,
) -> Result<Value> {
    let mut shards = Vec::with_capacity(gguf.shards.len());
    for (shard_idx, shard) in gguf.shards.iter().enumerate() {
        let file_bytes = u64::try_from(shard.mmap_len()).context("shard length exceeds u64")?;
        let shard_requests = requests
            .iter()
            .filter(|request| request.desc.shard_idx == shard_idx)
            .copied()
            .collect::<Vec<_>>();
        let exact = merged_request_intervals(&shard_requests, shard.tensor_data_start, file_bytes)?;
        let rounded = round_intervals(&exact, page_size, file_bytes)?;
        let rounded_tensor = intersect_intervals(
            &rounded,
            Interval {
                start: shard.tensor_data_start,
                end: file_bytes,
            },
        );
        let tensor_region_bytes = file_bytes
            .checked_sub(shard.tensor_data_start)
            .ok_or_else(|| anyhow!("tensor data starts beyond shard {shard_idx}"))?;
        let exact_bytes = interval_bytes(&exact)?;
        let rounded_source_bytes = interval_bytes(&rounded)?;
        let rounded_tensor_bytes = interval_bytes(&rounded_tensor)?;
        let logical_source_bytes = sum_request_source_bytes(&shard_requests)?;
        let logical_resident_bytes = sum_request_resident_bytes(&shard_requests)?;
        let exact_gaps = gap_stats(
            &exact,
            Interval {
                start: shard.tensor_data_start,
                end: file_bytes,
            },
        )?;
        let rounded_gaps = gap_stats(
            &rounded_tensor,
            Interval {
                start: shard.tensor_data_start,
                end: file_bytes,
            },
        )?;
        let drop = above_screening_threshold(rounded_tensor_bytes, tensor_region_bytes)?;
        shards.push(json!({
            "shard_idx": shard_idx,
            "path": shard.path,
            "file_bytes": file_bytes,
            "pre_tensor_data_bytes": shard.tensor_data_start,
            "tensor_region_bytes": tensor_region_bytes,
            "request_count": shard_requests.len(),
            "logical_source_bytes": logical_source_bytes,
            "logical_resident_bytes": logical_resident_bytes,
            "unique_source_union_bytes": exact_bytes,
            "repeated_source_bytes": logical_source_bytes.checked_sub(exact_bytes)
                .ok_or_else(|| anyhow!("unique source union exceeds logical source bytes"))?,
            "rounded_source_bytes": rounded_source_bytes,
            "rounded_tensor_region_bytes": rounded_tensor_bytes,
            "savings_vs_whole_file_bytes": file_bytes.checked_sub(rounded_source_bytes)
                .ok_or_else(|| anyhow!("rounded source bytes exceed file bytes"))?,
            "rounded_tensor_region_coverage": ratio(rounded_tensor_bytes, tensor_region_bytes),
            "exact_interval_count": exact.len(),
            "rounded_interval_count": rounded.len(),
            "exact_tensor_region_complement": gap_json(&exact_gaps),
            "rounded_tensor_region_complement": gap_json(&rounded_gaps),
            "screening_verdict": if drop { "drop_range_only_warmer" } else { "retain_range_only_warmer" },
        }));
    }

    let logical_source_bytes = sum_request_source_bytes(requests)?;
    let logical_resident_bytes = sum_request_resident_bytes(requests)?;
    let unique_source_union_bytes = sum_json_u64(&shards, "unique_source_union_bytes")?;
    let rounded_source_bytes = sum_json_u64(&shards, "rounded_source_bytes")?;
    let rounded_tensor_bytes = sum_json_u64(&shards, "rounded_tensor_region_bytes")?;
    let file_bytes = sum_json_u64(&shards, "file_bytes")?;
    let tensor_region_bytes = sum_json_u64(&shards, "tensor_region_bytes")?;
    let repeated_source_bytes = logical_source_bytes
        .checked_sub(unique_source_union_bytes)
        .ok_or_else(|| anyhow!("global unique source union exceeds logical source bytes"))?;

    let bound_sources = requests
        .iter()
        .map(|request| source_key(request.desc))
        .collect::<HashSet<_>>();
    let mtp_descriptors = mtp_weight_source_descriptors(model);
    let mtp_sources = mtp_descriptors
        .iter()
        .map(|desc| source_key(desc))
        .collect::<HashSet<_>>();
    let mtp_source_bytes = mtp_descriptors.iter().try_fold(0u64, |sum, desc| {
        sum.checked_add(desc.n_bytes)
            .ok_or_else(|| anyhow!("MTP source byte accounting overflow"))
    })?;
    let unbound = gguf
        .tensors
        .iter()
        .filter(|desc| {
            let key = source_key(desc);
            !bound_sources.contains(&key) && !mtp_sources.contains(&key)
        })
        .collect::<Vec<_>>();
    let unbound_source_bytes = unbound.iter().try_fold(0u64, |sum, desc| {
        sum.checked_add(desc.n_bytes)
            .ok_or_else(|| anyhow!("unbound source byte accounting overflow"))
    })?;

    Ok(json!({
        "scope": "base_model_weight_requests",
        "plan_semantics": "production_default_with_explicit_overrides",
        "embedding_policy_requested": "production_auto",
        "native_embedding_resolved": native_embedding,
        "router_f16": router_f16,
        "inventory_digest": model_weight_storage_inventory_digest(requests),
        "request_count": requests.len(),
        "logical_source_bytes": logical_source_bytes,
        "logical_resident_bytes": logical_resident_bytes,
        "unique_source_union_bytes": unique_source_union_bytes,
        "repeated_source_bytes": repeated_source_bytes,
        "rounded_source_bytes": rounded_source_bytes,
        "rounded_tensor_region_bytes": rounded_tensor_bytes,
        "file_bytes": file_bytes,
        "tensor_region_bytes": tensor_region_bytes,
        "savings_vs_whole_file_bytes": file_bytes.checked_sub(rounded_source_bytes)
            .ok_or_else(|| anyhow!("global rounded source bytes exceed file bytes"))?,
        "rounded_tensor_region_coverage": ratio(rounded_tensor_bytes, tensor_region_bytes),
        "screening_verdict": if above_screening_threshold(rounded_tensor_bytes, tensor_region_bytes)? {
            "drop_range_only_warmer"
        } else {
            "retain_range_only_warmer"
        },
        "storage_kinds": storage_kind_census(requests, gguf.shards.len())?,
        "repeated_source_groups": repeated_source_groups(requests),
        "mtp": {
            "present": model.mtp.is_some(),
            "descriptor_count": mtp_descriptors.len(),
            "logical_source_bytes": mtp_source_bytes,
            "excluded_from_base_plan": true,
        },
        "unbound_gguf": {
            "descriptor_count": unbound.len(),
            "logical_source_bytes": unbound_source_bytes,
            "excludes_mtp": true,
        },
        "shards": shards,
    }))
}

fn storage_kind_census(
    requests: &[ModelWeightStorageRequest<'_>],
    shard_count: usize,
) -> Result<Value> {
    let rows = [
        ModelWeightStorageKind::Direct,
        ModelWeightStorageKind::ConvertedF32,
        ModelWeightStorageKind::ConvertedF16,
    ]
    .into_iter()
    .map(|kind| -> Result<(String, Value)> {
        let selected = requests
            .iter()
            .filter(|request| request.kind == kind)
            .copied()
            .collect::<Vec<_>>();
        let mut unique = 0u64;
        for shard_idx in 0..shard_count {
            let intervals = selected
                .iter()
                .filter(|request| request.desc.shard_idx == shard_idx)
                .map(|request| checked_source_interval(request.desc))
                .collect::<Result<Vec<_>>>()?;
            unique = unique
                .checked_add(interval_bytes(&merge_intervals(intervals))?)
                .ok_or_else(|| anyhow!("storage-kind unique byte accounting overflow"))?;
        }
        Ok((
            storage_kind_label(kind).to_string(),
            json!({
                "request_count": selected.len(),
                "logical_source_bytes": sum_request_source_bytes(&selected)?,
                "logical_resident_bytes": sum_request_resident_bytes(&selected)?,
                "unique_source_union_bytes": unique,
                "unique_unions_may_overlap_other_kinds": true,
            }),
        ))
    })
    .collect::<Result<serde_json::Map<String, Value>>>()?;
    Ok(Value::Object(rows))
}

fn repeated_source_groups(requests: &[ModelWeightStorageRequest<'_>]) -> Vec<Value> {
    let mut groups: BTreeMap<(usize, u64, u64), Vec<&ModelWeightStorageRequest<'_>>> =
        BTreeMap::new();
    for request in requests {
        groups
            .entry(source_key(request.desc))
            .or_default()
            .push(request);
    }
    groups
        .into_iter()
        .filter(|(_, requests)| requests.len() > 1)
        .map(|((shard_idx, data_offset, n_bytes), requests)| {
            let names = requests
                .iter()
                .map(|request| request.desc.name.as_str())
                .collect::<BTreeSet<_>>();
            let kinds = requests
                .iter()
                .map(|request| storage_kind_label(request.kind))
                .collect::<BTreeSet<_>>();
            json!({
                "shard_idx": shard_idx,
                "data_offset": data_offset,
                "n_bytes": n_bytes,
                "request_count": requests.len(),
                "names": names,
                "storage_kinds": kinds,
            })
        })
        .collect()
}

fn conversation_census(
    model_path: &Path,
    paths: &[PathBuf],
    policy: ThinkingPolicy,
    tokenizer: &Tokenizer,
) -> Result<Value> {
    let mut files = Vec::with_capacity(paths.len());
    for (ordinal, path) in paths.iter().enumerate() {
        files.push(load_conversation_file(
            model_path, ordinal, path, policy, tokenizer,
        )?);
    }

    let file_rows = files
        .iter()
        .map(|file| {
            let turn_rows = file
                .turns
                .iter()
                .map(|turn| turn_json(turn, &files))
                .collect::<Vec<_>>();
            json!({
                "input_ordinal": file.ordinal,
                "path": file.path,
                "metadata_model": file.metadata_model,
                "metadata_model_matches_loaded_path": file.metadata_model_matches_loaded_path,
                "thinking_preserved": file.preserve_thinking,
                "thinking_resolution": file.thinking_resolution,
                "assistant_messages_transformed_by_renderer": file.transformed_assistant_messages,
                "role_anomalies": file.role_anomalies,
                "request_count": file.turns.len(),
                "adjacent_reuse_summary": transition_summary(std::slice::from_ref(file)),
                "requests": turn_rows,
            })
        })
        .collect::<Vec<_>>();

    Ok(json!({
        "input_count": files.len(),
        "comparison_semantics": {
            "prompt": "reconstructed_from_current_shared_renderer",
            "completed": "retokenized_proxy_without_authoritative_generated_token_ids",
            "best_local_prior": "earlier_assistant_turns_in_same_file",
            "best_corpus_prefix": "earlier_same_file_or_any_distinct_cross_file_candidate_not_causal_ancestry",
            "boundary_tokens_lost": "candidate_token_count_minus_lcp_tokens",
            "request_suffix_tokens": "request_token_count_minus_lcp_tokens",
        },
        "adjacent_reuse_summary": transition_summary(&files),
        "files": file_rows,
    }))
}

fn transition_summary(files: &[ConversationFile]) -> Value {
    let mut prompt_suffixes = Vec::new();
    let mut completed_suffixes = Vec::new();
    let mut saved_replay_tokens = Vec::new();
    let mut prompt_exact = 0usize;
    let mut completed_exact = 0usize;
    for file in files {
        for pair in file.turns.windows(2) {
            let prior = &pair[0];
            let target = &pair[1];
            let prompt_lcp = token_lcp(&prior.prompt_tokens, &target.prompt_tokens);
            let completed_lcp = token_lcp(&prior.completed_tokens, &target.prompt_tokens);
            prompt_exact += usize::from(prompt_lcp == prior.prompt_tokens.len());
            completed_exact += usize::from(completed_lcp == prior.completed_tokens.len());
            let prompt_suffix = target.prompt_tokens.len() - prompt_lcp;
            let completed_suffix = target.prompt_tokens.len() - completed_lcp;
            prompt_suffixes.push(prompt_suffix as u64);
            completed_suffixes.push(completed_suffix as u64);
            saved_replay_tokens.push(prompt_suffix as i64 - completed_suffix as i64);
        }
    }
    let prompt_total: u64 = prompt_suffixes.iter().sum();
    let completed_total: u64 = completed_suffixes.iter().sum();
    json!({
        "transition_count": prompt_suffixes.len(),
        "prompt_checkpoint_exact_prefix_count": prompt_exact,
        "completed_proxy_exact_prefix_count": completed_exact,
        "prompt_checkpoint_request_suffix_tokens": distribution_u64(&prompt_suffixes),
        "completed_proxy_request_suffix_tokens": distribution_u64(&completed_suffixes),
        "completed_proxy_replay_tokens_avoided": distribution_i64(&saved_replay_tokens),
        "aggregate_suffix_replay_ratio_prompt_over_completed":
            ratio(prompt_total, completed_total),
    })
}

fn load_conversation_file(
    model_path: &Path,
    ordinal: usize,
    path: &Path,
    policy: ThinkingPolicy,
    tokenizer: &Tokenizer,
) -> Result<ConversationFile> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("read messages input {}", path.display()))?;
    let value: Value = serde_json::from_str(&raw)
        .with_context(|| format!("parse messages input {}", path.display()))?;
    let (messages, meta) = parse_messages_input(value)?;
    if messages.is_empty() {
        bail!("messages input {} contains no messages", path.display());
    }
    let (preserve_thinking, thinking_resolution) = resolve_thinking(policy, &meta);
    let metadata_model = meta
        .get("model")
        .and_then(Value::as_str)
        .map(str::to_string);
    let metadata_model_matches_loaded_path = metadata_model
        .as_deref()
        .map(|saved| paths_equivalent(Path::new(saved), model_path));
    let transformed_assistant_messages = if preserve_thinking {
        0
    } else {
        messages
            .iter()
            .filter(|message| {
                message.role == "assistant" && strip_think(&message.content) != message.content
            })
            .count()
    };
    let role_anomalies = role_anomalies(&messages);
    let mut turns = Vec::new();
    for (assistant_message_index, message) in messages.iter().enumerate() {
        if message.role != "assistant" {
            continue;
        }
        let prompt_text = render_qwen_messages_prompt(
            &messages[..assistant_message_index],
            preserve_thinking,
            true,
        );
        let completed_text = render_qwen_messages_prompt(
            &messages[..=assistant_message_index],
            preserve_thinking,
            false,
        );
        let prompt_tokens = tokenizer
            .encode(&prompt_text, false)
            .with_context(|| format!("tokenize request {} in {}", turns.len(), path.display()))?;
        let completed_tokens = tokenizer.encode(&completed_text, false).with_context(|| {
            format!(
                "tokenize completed proxy {} in {}",
                turns.len(),
                path.display()
            )
        })?;
        turns.push(ConversationTurn {
            file_ordinal: ordinal,
            file: path.to_path_buf(),
            turn_index: turns.len(),
            assistant_message_index,
            prompt_text,
            prompt_tokens,
            completed_text,
            completed_tokens,
        });
    }
    Ok(ConversationFile {
        ordinal,
        path: path.to_path_buf(),
        metadata_model,
        metadata_model_matches_loaded_path,
        preserve_thinking,
        thinking_resolution,
        transformed_assistant_messages,
        role_anomalies,
        turns,
    })
}

fn turn_json(turn: &ConversationTurn, files: &[ConversationFile]) -> Value {
    let current = &turn.prompt_tokens;
    let previous = files[turn.file_ordinal]
        .turns
        .get(turn.turn_index.wrapping_sub(1));
    let immediate_prompt = previous.map(|prior| {
        comparison_json(
            Candidate {
                turn: prior,
                kind: CheckpointKind::Prompt,
            },
            turn,
        )
    });
    let immediate_completed = previous.map(|prior| {
        comparison_json(
            Candidate {
                turn: prior,
                kind: CheckpointKind::CompletedProxy,
            },
            turn,
        )
    });
    let best_local = best_candidate(turn, files, false).map(candidate_match_json);
    let best_corpus = best_candidate(turn, files, true).map(candidate_match_json);
    json!({
        "turn_index": turn.turn_index,
        "assistant_message_index": turn.assistant_message_index,
        "request_rendered_bytes": turn.prompt_text.len(),
        "request_tokens": current.len(),
        "completed_proxy_rendered_bytes": turn.completed_text.len(),
        "completed_proxy_tokens": turn.completed_tokens.len(),
        "immediate_prior_prompt": immediate_prompt,
        "immediate_prior_completed_proxy": immediate_completed,
        "best_local_prior_checkpoint": best_local,
        "best_corpus_prefix_checkpoint": best_corpus,
    })
}

fn comparison_json(candidate: Candidate<'_>, target: &ConversationTurn) -> Value {
    let lcp = token_lcp(candidate.tokens(), &target.prompt_tokens);
    json!({
        "candidate_kind": candidate.kind.label(),
        "candidate_confidence": candidate.kind.confidence(),
        "candidate_tokens": candidate.tokens().len(),
        "candidate_rendered_bytes": candidate.text().len(),
        "rendered_bytes_are_prefix": target.prompt_text.starts_with(candidate.text()),
        "lcp_tokens": lcp,
        "exact_token_prefix": lcp == candidate.tokens().len(),
        "boundary_tokens_lost": candidate.tokens().len() - lcp,
        "request_suffix_tokens": target.prompt_tokens.len() - lcp,
    })
}

fn best_candidate<'a>(
    target: &'a ConversationTurn,
    files: &'a [ConversationFile],
    include_cross_file: bool,
) -> Option<Candidate<'a>> {
    let mut best = None;
    for file in files {
        for source in &file.turns {
            let allowed = if source.file_ordinal == target.file_ordinal {
                source.turn_index < target.turn_index
            } else {
                include_cross_file
            };
            if !allowed {
                continue;
            }
            for kind in [CheckpointKind::Prompt, CheckpointKind::CompletedProxy] {
                let candidate = Candidate { turn: source, kind };
                if candidate.tokens().len() > target.prompt_tokens.len()
                    || token_lcp(candidate.tokens(), &target.prompt_tokens)
                        != candidate.tokens().len()
                {
                    continue;
                }
                if best.is_none_or(|existing| candidate_order(candidate, existing).is_lt()) {
                    best = Some(candidate);
                }
            }
        }
    }
    best
}

fn candidate_order(left: Candidate<'_>, right: Candidate<'_>) -> Ordering {
    right
        .tokens()
        .len()
        .cmp(&left.tokens().len())
        .then_with(|| left.kind.tie_rank().cmp(&right.kind.tie_rank()))
        .then_with(|| left.turn.file_ordinal.cmp(&right.turn.file_ordinal))
        .then_with(|| left.turn.turn_index.cmp(&right.turn.turn_index))
}

fn candidate_match_json(candidate: Candidate<'_>) -> Value {
    json!({
        "source_file": candidate.turn.file,
        "source_input_ordinal": candidate.turn.file_ordinal,
        "source_turn_index": candidate.turn.turn_index,
        "candidate_kind": candidate.kind.label(),
        "candidate_confidence": candidate.kind.confidence(),
        "matched_tokens": candidate.tokens().len(),
    })
}

fn resolve_thinking(policy: ThinkingPolicy, meta: &Value) -> (bool, String) {
    match policy.messages_mode() {
        MessagesThinkingMode::Preserve => (true, "explicit_preserve".to_string()),
        MessagesThinkingMode::Strip => (false, "explicit_strip".to_string()),
        MessagesThinkingMode::Auto => {
            let preserve = messages_auto_preserve_thinking(meta);
            let reason = if meta
                .get("preserve_thinking")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                "metadata_preserve_thinking"
            } else if meta
                .get("model")
                .and_then(Value::as_str)
                .is_some_and(|model| model.to_ascii_lowercase().contains("qwen3.6"))
            {
                "metadata_model_qwen36"
            } else if meta.get("model").is_none() {
                "absent_model_metadata_default_strip"
            } else {
                "metadata_default_strip"
            };
            (preserve, reason.to_string())
        }
    }
}

fn role_anomalies(messages: &[ChatMessage]) -> Vec<String> {
    let mut anomalies = Vec::new();
    for (index, pair) in messages.windows(2).enumerate() {
        if pair[0].role == pair[1].role {
            anomalies.push(format!(
                "consecutive role {:?} at message indices {} and {}",
                pair[0].role,
                index,
                index + 1
            ));
        }
    }
    for (index, message) in messages.iter().enumerate() {
        if message.role == "assistant"
            && index
                .checked_sub(1)
                .and_then(|prior| messages.get(prior))
                .is_none_or(|prior| prior.role != "user")
        {
            anomalies.push(format!(
                "assistant at message index {index} is not immediately preceded by user"
            ));
        }
    }
    anomalies
}

fn paths_equivalent(left: &Path, right: &Path) -> bool {
    match (left.canonicalize(), right.canonicalize()) {
        (Ok(left), Ok(right)) => left == right,
        _ => left == right,
    }
}

fn merged_request_intervals(
    requests: &[ModelWeightStorageRequest<'_>],
    tensor_data_start: u64,
    file_bytes: u64,
) -> Result<Vec<Interval>> {
    let intervals = requests
        .iter()
        .map(|request| {
            let interval = checked_source_interval(request.desc)?;
            if interval.start < tensor_data_start || interval.end > file_bytes {
                bail!(
                    "tensor {:?} source interval [{}, {}) is outside tensor region [{}, {})",
                    request.desc.name,
                    interval.start,
                    interval.end,
                    tensor_data_start,
                    file_bytes
                );
            }
            Ok(interval)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(merge_intervals(intervals))
}

fn checked_source_interval(desc: &qwen_llm::tensor::TensorDesc) -> Result<Interval> {
    let end = desc
        .data_offset
        .checked_add(desc.n_bytes)
        .ok_or_else(|| anyhow!("tensor {:?} source interval overflows", desc.name))?;
    Ok(Interval {
        start: desc.data_offset,
        end,
    })
}

fn merge_intervals(mut intervals: Vec<Interval>) -> Vec<Interval> {
    intervals.sort_by_key(|interval| (interval.start, interval.end));
    let mut merged: Vec<Interval> = Vec::with_capacity(intervals.len());
    for interval in intervals {
        if let Some(last) = merged.last_mut()
            && interval.start <= last.end
        {
            last.end = last.end.max(interval.end);
        } else {
            merged.push(interval);
        }
    }
    merged
}

fn round_intervals(
    intervals: &[Interval],
    page_size: u64,
    file_bytes: u64,
) -> Result<Vec<Interval>> {
    let mut rounded = Vec::with_capacity(intervals.len());
    for interval in intervals {
        let start = interval.start / page_size * page_size;
        let end = interval
            .end
            .checked_add(page_size - 1)
            .ok_or_else(|| anyhow!("page rounding overflow"))?
            / page_size
            * page_size;
        rounded.push(Interval {
            start: start.min(file_bytes),
            end: end.min(file_bytes),
        });
    }
    Ok(merge_intervals(rounded))
}

fn intersect_intervals(intervals: &[Interval], domain: Interval) -> Vec<Interval> {
    intervals
        .iter()
        .filter_map(|interval| {
            let start = interval.start.max(domain.start);
            let end = interval.end.min(domain.end);
            (start < end).then_some(Interval { start, end })
        })
        .collect()
}

fn gap_stats(intervals: &[Interval], domain: Interval) -> Result<GapStats> {
    let mut cursor = domain.start;
    let mut count = 0usize;
    let mut bytes = 0u64;
    let mut largest_bytes = 0u64;
    for interval in intervals {
        let start = interval.start.max(domain.start);
        let end = interval.end.min(domain.end);
        if start >= end {
            continue;
        }
        if start > cursor {
            let gap = start - cursor;
            count += 1;
            bytes = bytes
                .checked_add(gap)
                .ok_or_else(|| anyhow!("gap byte accounting overflow"))?;
            largest_bytes = largest_bytes.max(gap);
        }
        cursor = cursor.max(end);
    }
    if cursor < domain.end {
        let gap = domain.end - cursor;
        count += 1;
        bytes = bytes
            .checked_add(gap)
            .ok_or_else(|| anyhow!("gap byte accounting overflow"))?;
        largest_bytes = largest_bytes.max(gap);
    }
    Ok(GapStats {
        count,
        bytes,
        largest_bytes,
    })
}

fn interval_bytes(intervals: &[Interval]) -> Result<u64> {
    intervals.iter().try_fold(0u64, |sum, interval| {
        sum.checked_add(interval.len())
            .ok_or_else(|| anyhow!("interval byte accounting overflow"))
    })
}

fn above_screening_threshold(numerator: u64, denominator: u64) -> Result<bool> {
    let left = numerator
        .checked_mul(SCREENING_DENOMINATOR)
        .ok_or_else(|| anyhow!("screening numerator overflow"))?;
    let right = denominator
        .checked_mul(SCREENING_NUMERATOR)
        .ok_or_else(|| anyhow!("screening denominator overflow"))?;
    Ok(left > right)
}

fn sum_request_source_bytes(requests: &[ModelWeightStorageRequest<'_>]) -> Result<u64> {
    requests.iter().try_fold(0u64, |sum, request| {
        sum.checked_add(request.desc.n_bytes)
            .ok_or_else(|| anyhow!("logical source byte accounting overflow"))
    })
}

fn sum_request_resident_bytes(requests: &[ModelWeightStorageRequest<'_>]) -> Result<u64> {
    requests.iter().try_fold(0u64, |sum, request| {
        sum.checked_add(request.resident_bytes)
            .ok_or_else(|| anyhow!("logical resident byte accounting overflow"))
    })
}

fn sum_json_u64(rows: &[Value], field: &str) -> Result<u64> {
    rows.iter().try_fold(0u64, |sum, row| {
        let value = row
            .get(field)
            .and_then(Value::as_u64)
            .ok_or_else(|| anyhow!("missing numeric field {field:?}"))?;
        sum.checked_add(value)
            .ok_or_else(|| anyhow!("field {field:?} sum overflow"))
    })
}

fn source_key(desc: &qwen_llm::tensor::TensorDesc) -> (usize, u64, u64) {
    (desc.shard_idx, desc.data_offset, desc.n_bytes)
}

fn storage_kind_label(kind: ModelWeightStorageKind) -> &'static str {
    match kind {
        ModelWeightStorageKind::Direct => "direct",
        ModelWeightStorageKind::ConvertedF32 => "converted_f32",
        ModelWeightStorageKind::ConvertedF16 => "converted_f16",
    }
}

fn gap_json(stats: &GapStats) -> Value {
    json!({
        "gap_count": stats.count,
        "gap_bytes": stats.bytes,
        "largest_gap_bytes": stats.largest_bytes,
    })
}

fn ratio(numerator: u64, denominator: u64) -> Option<f64> {
    (denominator != 0).then_some(numerator as f64 / denominator as f64)
}

fn distribution_u64(values: &[u64]) -> Value {
    if values.is_empty() {
        return json!({
            "count": 0,
            "sum": 0,
            "min": null,
            "median": null,
            "max": null,
        });
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    json!({
        "count": sorted.len(),
        "sum": sorted.iter().sum::<u64>(),
        "min": sorted.first(),
        "median": sorted[sorted.len() / 2],
        "max": sorted.last(),
    })
}

fn distribution_i64(values: &[i64]) -> Value {
    if values.is_empty() {
        return json!({
            "count": 0,
            "sum": 0,
            "min": null,
            "median": null,
            "max": null,
        });
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    json!({
        "count": sorted.len(),
        "sum": sorted.iter().sum::<i64>(),
        "min": sorted.first(),
        "median": sorted[sorted.len() / 2],
        "max": sorted.last(),
    })
}

fn token_lcp(left: &[i32], right: &[i32]) -> usize {
    left.iter()
        .zip(right)
        .take_while(|(left, right)| left == right)
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interval_union_merges_overlap_adjacency_and_duplicates() {
        let merged = merge_intervals(vec![
            Interval { start: 30, end: 40 },
            Interval { start: 10, end: 20 },
            Interval { start: 20, end: 25 },
            Interval { start: 12, end: 18 },
        ]);
        assert_eq!(
            merged,
            vec![
                Interval { start: 10, end: 25 },
                Interval { start: 30, end: 40 }
            ]
        );
    }

    #[test]
    fn page_rounding_clips_and_remerges() {
        let rounded = round_intervals(
            &[
                Interval { start: 17, end: 31 },
                Interval { start: 32, end: 49 },
            ],
            16,
            45,
        )
        .expect("round intervals");
        assert_eq!(rounded, vec![Interval { start: 16, end: 45 }]);
    }

    #[test]
    fn complement_is_scoped_to_tensor_region() {
        let stats = gap_stats(
            &[
                Interval { start: 16, end: 24 },
                Interval { start: 32, end: 40 },
            ],
            Interval { start: 20, end: 44 },
        )
        .expect("gap stats");
        assert_eq!(stats.count, 2);
        assert_eq!(stats.bytes, 12);
        assert_eq!(stats.largest_bytes, 8);
    }

    #[test]
    fn token_comparison_exposes_boundary_loss() {
        assert_eq!(token_lcp(&[1, 2, 3], &[1, 2, 4, 5]), 2);
        assert_eq!(token_lcp(&[1, 2], &[1, 2, 3]), 2);
    }

    #[test]
    fn screening_threshold_is_strict() {
        assert!(!above_screening_threshold(90, 100).expect("threshold"));
        assert!(above_screening_threshold(91, 100).expect("threshold"));
    }
}
