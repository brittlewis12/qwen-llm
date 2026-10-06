//! GLM-5.3-Flash: bounded serial generation up to the checkpoint context
//! (dense attention below visible length 2052, sparse DSA selection from
//! there on) within device memory, from a raw prompt or a text chat
//! rendered by `qwen_llm::glm5_next_chat`; chat reasoning streams to stderr
//! and the answer to stdout (`qwen serve`: `serve/backend_glm5_next.rs`;
//! `qwen-lens read-full --logit-lens`: `plain_logit_lens/glm5_next.rs`).
//! Tools, lens transport and fitting are not implemented;
//! every other surface refuses rather than falling through to Qwen
//! protocols.

use super::*;
use crate::lane_timing::{LanePhases, LaneTiming};
use crate::prompt_template::{InputCapability, Support};
use qwen_llm::glm5_next::{
    Glm5NextAdmissionError, Glm5NextArtifactLayout, Glm5NextPreparedArtifact,
};
use qwen_llm::glm5_next_chat::{self as chat, Effort, Message, RenderOptions};
use qwen_llm::glm5_next_metal::{
    DEFAULT_PREFILL_ROWS, Glm5NextMetalError, Glm5NextSession, Glm5NextWeights,
    prefetch_retained_with_cancel, preflight_session,
};
use serde_json::{Value, json};

const FAMILY: &str = "GLM-5.3-Flash";

/// Request phases of the GLM run lane. Setup (artifact, tokenizer, chat
/// profile, input, rendering, device, memory preflight, prefetch) stays out
/// of the loaded request, which is request preparation, encoding and
/// resident execution.
#[derive(Clone, Copy)]
enum Phase {
    RequestPreparation,
    ArtifactLayout,
    TokenizerConstruction,
    ArtifactVerification,
    InputAcquisition,
    Rendering,
    Encoding,
    DeviceSetup,
    MemoryPreflight,
    Prefetch,
    ModelLoad,
    SessionSetup,
    ResidentExecution,
}

impl LanePhases for Phase {
    const FAMILY: &'static str = "GLM-5.3";
    const PHASES: &'static [(Self, &'static str)] = &[
        (Phase::RequestPreparation, "request_preparation"),
        (Phase::ArtifactLayout, "artifact_layout"),
        (Phase::TokenizerConstruction, "tokenizer_construction"),
        (Phase::ArtifactVerification, "artifact_verification"),
        (Phase::InputAcquisition, "input_acquisition"),
        (Phase::Rendering, "rendering"),
        (Phase::Encoding, "encoding"),
        (Phase::DeviceSetup, "device_setup"),
        (Phase::MemoryPreflight, "memory_preflight"),
        (Phase::Prefetch, "prefetch"),
        (Phase::ModelLoad, "model_load"),
        (Phase::SessionSetup, "session_setup"),
        (Phase::ResidentExecution, "resident_execution"),
    ];
    const LOADED_REQUEST: &'static [Self] = &[
        Phase::Encoding,
        Phase::RequestPreparation,
        Phase::ResidentExecution,
    ];
    const LOAD: &'static [Self] = &[Phase::ModelLoad, Phase::SessionSetup];
    const ENCODING: Self = Phase::Encoding;
    const LOADED_REQUEST_POLICY: &'static str =
        "sum_encoding_request_preparation_resident_execution_not_continuous_wall";
    const END_TO_END_BOUNDARY: &'static str = "glm5_next_run_entry_through_generator_return_excludes_initial_gguf_open_final_formatting_stats_serialization";
    fn index(self) -> usize {
        self as usize
    }
}

type Timing = LaneTiming<Phase>;

/// Forward-artifact preparation (binding, execution coverage, tokenizer)
/// without generation policy: the library call the plain lens admits with
/// (`qwen-lens`, `plain_logit_lens::glm5_next`), from which the capability
/// projection derives lens support, so the two cannot disagree. Stop and chat
/// requirements apply only to the lanes that generate.
pub(crate) fn forward_admission(
    gguf: &GgufFile,
) -> std::result::Result<Glm5NextPreparedArtifact<'_>, Glm5NextAdmissionError> {
    Glm5NextPreparedArtifact::inspect(gguf)
}

/// CPU preparation shared by run, info and bench: binding, execution
/// coverage, tokenizer and stop set, with stable refusal codes.
fn admission(
    gguf: &GgufFile,
) -> std::result::Result<(Glm5NextPreparedArtifact<'_>, Vec<i32>), Glm5NextAdmissionError> {
    let prepared = Glm5NextPreparedArtifact::inspect(gguf)?;
    let stops = prepared.generation_stops()?;
    Ok((prepared, stops))
}

type ChatVerdict = std::result::Result<chat::VerifiedChatProfile, chat::ChatError>;

/// The artifact's admission and, once admitted, its chat profile verdict.
fn verdict(
    gguf: &GgufFile,
) -> std::result::Result<(Glm5NextPreparedArtifact<'_>, ChatVerdict), Glm5NextAdmissionError> {
    let (prepared, _) = admission(gguf)?;
    let profile = prepared.chat_profile();
    Ok((prepared, profile))
}

/// Which input forms render for this artifact: raw once it is admitted;
/// text chat and function tools once its template profile verifies.
fn input_support(
    verdict: &std::result::Result<
        (Glm5NextPreparedArtifact<'_>, ChatVerdict),
        Glm5NextAdmissionError,
    >,
) -> InputCapability {
    match verdict {
        Err(error) => InputCapability::none(error.code(), error.to_string()),
        Ok((_, Ok(_))) => InputCapability {
            raw: Support::Supported,
            user: Support::Supported,
            messages: Support::Supported,
            tools: Support::Supported,
        },
        Ok((_, Err(error))) => InputCapability {
            tools: Support::Unsupported {
                code: error.code(),
                message: format!("{FAMILY} tools need a verified chat template: {error}"),
            },
            ..InputCapability::raw_only(error.code(), format!("{error}; use --raw-prompt"))
        },
    }
}

/// `capabilities.input` for callers outside the family profile.
pub(crate) fn input_capability(gguf: &GgufFile) -> InputCapability {
    input_support(&verdict(gguf))
}

/// `capabilities.template` for callers outside the family profile.
pub(crate) fn template_projection(gguf: &GgufFile) -> Value {
    capability_projection(gguf)
        .map(|projection| projection["template"].clone())
        .unwrap_or_else(|error| json!({"status": "unresolved", "message": error.to_string()}))
}

pub(crate) fn capability_projection(gguf: &GgufFile) -> Result<Value> {
    let verdict = verdict(gguf);
    let run = match &verdict {
        Ok((prepared, profile)) => json!({
            "status": "conditional", "implementation_status": "partial",
            "scope": if profile.is_ok() { "raw_and_text_chat" } else { "raw_only" },
            "artifact_admission": {"status": "passed"},
            "capacity_policy": format!("checkpoint_context_{}_and_device_memory", prepared.config().context_length),
            "attention": "dense_below_2052_sparse_dsa_from_2052",
            "prefill": if prepared.packed_prefill() { "packed_fast" } else { "serial" },
            "native_tokenizer": true, "latent_cache": "f16",
            "output": "raw_literal_or_reasoning_stderr_answer_stdout",
            "sampling_default": "release_generation_config_temperature_1_top_p_0.95",
        }),
        Err(error) => json!({
            "status": "unsupported", "implementation_status": "partial",
            "artifact_admission": {"status": "rejected", "code": error.code(), "message": error.to_string()},
        }),
    };
    let bench = match &verdict {
        Ok(_) => json!({"status": "conditional", "implementation_status": "partial",
            "command": "qwen-bench suite", "scope": "packed_prefill_and_serial_decode_rows"}),
        Err(_) => json!({"status": "unsupported", "implementation_status": "partial"}),
    };
    let serve = match &verdict {
        Ok((_, Ok(_))) => json!({"status": "conditional", "implementation_status": "partial",
            "endpoint": "/v1/responses", "input": "verified_chat_and_function_tool_items",
            "capacity_policy": "explicit_max_context_tokens_and_max_tokens_within_device_memory",
            "prefix_reuse": "live_session_exact_extension", "snapshot_cache": false,
            "tools": "function_tools_auto_or_allowed_strict_refused",
            "idle_residency": "default_60s_idle_residency_secs",
            "sampling_default": "release_generation_config_temperature_1_top_p_0.95"}),
        Ok((_, Err(error))) => json!({"status": "unsupported", "implementation_status": "partial",
            "code": error.code(), "message": format!("{FAMILY} serve renders verified text chat only: {error}")}),
        Err(error) => json!({"status": "unsupported", "implementation_status": "partial",
            "code": error.code(), "message": error.to_string()}),
    };
    // Lens support follows forward preparation only (generation stops are
    // a generation-lane requirement): re-derived only when generation
    // admission failed, since its success implies forward preparation's.
    let forward = match &verdict {
        Ok(_) => Ok(()),
        Err(_) => forward_admission(gguf).map(|_| ()),
    };
    let lens = match &forward {
        Ok(()) => json!({"status": "partial", "command": "qwen-lens read-full --logit-lens",
            "scope": "raw_plain_logit_lens_post_block_residual_streams",
            "output_tail": "native_four_stream_mean_rmsnorm_untied_head",
            "capacity_policy": "selected_position_plus_one_within_device_memory",
            "content_identity": "not_computed", "transport": "unsupported", "cli_interventions": false}),
        Err(error) => json!({"status": "unsupported", "implementation_status": "partial",
            "code": error.code(), "message": error.to_string()}),
    };
    let refused = |code: &str, message: String| {
        let refused = json!({"status": "unsupported", "code": code, "message": message});
        json!({"levels": [], "fallback": null, "no_thinking": refused, "thinking": refused})
    };
    let (reasoning, template) = match &verdict {
        Ok((_, Ok(profile))) => (
            json!({"levels": Effort::LEVELS, "fallback": Effort::default().as_str(),
                "no_thinking": {"status": "unsupported", "code": "glm5_next_no_non_thinking_mode",
                    "message": format!("the {FAMILY} template always opens reasoning; use reasoning effort low")},
                "thinking": {"status": "supported"}}),
            json!({"status": "identified", "rendered_as": chat::RENDERER, "profile": profile}),
        ),
        Ok((_, Err(error))) => (
            refused(error.code(), error.to_string()),
            json!({"status": "unverified", "rendered_as": null, "code": error.code(), "message": error.to_string()}),
        ),
        Err(error) => (
            refused(error.code(), error.to_string()),
            json!({"status": "unsupported", "rendered_as": null, "message": error.to_string()}),
        ),
    };
    Ok(json!({
        "execution": {"run": run, "bench": bench, "serve": serve, "lens": lens,
            "request_device": {"status": "not_evaluated",
                "requires": ["request_options_and_token_budget", "live_memory_admission"]}},
        "input": input_support(&verdict),
        "reasoning": reasoning,
        "template": template,
    }))
}

/// What the request asks for, settled before the artifact is inspected or
/// any input is read.
enum Request {
    /// `--raw-prompt`, or the legacy prompt flags.
    Raw(cli::Invocation),
    /// `--user` (with `--system`) or `--messages`.
    Chat {
        run: cli::RunInvocation,
        effort: Effort,
    },
}

fn admit(
    invocation: cli::Invocation,
    args: &Args,
    explicit: ExplicitCliOptions,
) -> Result<Request> {
    admission::GLM5_NEXT_SINGLE_TURN.admit(&admission::supplied(args, explicit))?;
    ensure!(
        args.drafter.is_none(),
        "{FAMILY} does not support --drafter"
    );
    ensure!(
        args.messages.is_none(),
        "{FAMILY} reads chat documents with `qwen run --messages`; the legacy --messages flag is unsupported"
    );
    match invocation {
        cli::Invocation::Run(run) if matches!(run.input, cli::RunInput::RawPrompt(_)) => {
            ensure!(
                !run.no_thinking && run.reasoning_effort.is_none(),
                "{FAMILY} raw input does not accept reasoning controls; the prompt is the exact model input"
            );
            Ok(Request::Raw(cli::Invocation::Run(run)))
        }
        cli::Invocation::Run(run) => {
            ensure!(
                !run.no_thinking,
                "{FAMILY} has no --no-thinking mode: its template always opens reasoning; use --reasoning-effort low for the least"
            );
            let effort = Effort::parse(run.reasoning_effort.as_deref())?;
            Ok(Request::Chat { run, effort })
        }
        cli::Invocation::Legacy => Ok(Request::Raw(cli::Invocation::Legacy)),
        _ => bail!("{FAMILY} generation is a single-turn run path"),
    }
}

fn raw_input(invocation: cli::Invocation, args: &Args) -> Result<(String, PromptSource)> {
    match invocation {
        cli::Invocation::Run(cli::RunInvocation {
            input: cli::RunInput::RawPrompt(text),
            ..
        }) => Ok((text, PromptSource::Inline)),
        cli::Invocation::Legacy => {
            let (text, source, _) = prompt_text(args)?;
            Ok((text, source))
        }
        _ => bail!("{FAMILY} raw input requires --raw-prompt"),
    }
}

/// The conversation, its `clear_thinking` and its declared tools from
/// `--user`/`--system` or a `--messages` document.
fn chat_messages(
    input: cli::AcquiredRunInput,
) -> Result<(Vec<Message>, bool, Vec<chat::ToolDefinition>)> {
    match input {
        cli::AcquiredRunInput::User { system, user } => Ok((
            system
                .map(Message::System)
                .into_iter()
                .chain([Message::User(user)])
                .collect(),
            false,
            Vec::new(),
        )),
        cli::AcquiredRunInput::Messages { document, source } => {
            let document = chat::parse_document(document.as_bytes())
                .with_context(|| format!("read {FAMILY} chat document from {source}"))?;
            Ok((
                document.messages,
                document.clear_thinking.unwrap_or(false),
                document.tools,
            ))
        }
        cli::AcquiredRunInput::RawPrompt(_) => bail!("{FAMILY} raw input is not a chat"),
    }
}

/// Forwards for the request, bounded by the checkpoint context (device
/// memory is admitted when the session is created).
fn capacity(args: &Args, prompt_tokens: usize, context: u32) -> Result<usize> {
    let required = required_forwards(FAMILY, prompt_tokens, args.tokens, None)?;
    let capacity = args.max_context_tokens.unwrap_or(required);
    ensure!(
        capacity >= required,
        "{FAMILY} requires {required} forwards; requested capacity {capacity} is smaller"
    );
    ensure!(
        capacity <= context as usize,
        "{FAMILY} supports at most {context} positions (checkpoint context); this request needs {capacity}"
    );
    Ok(capacity)
}

/// What to do when a session of `required` positions (prompt plus
/// generation) does not fit and `fitting` positions would.
fn admission_advice(fitting: Option<u64>, required: usize) -> String {
    match fitting {
        Some(n) if n as usize >= required => format!("pass --max-context-tokens {n} or less"),
        Some(n) => format!(
            "this request needs {required} positions (prompt plus generation); shorten the prompt or --max-tokens to fit {n}"
        ),
        None => "free device memory or use a smaller artifact".to_string(),
    }
}

/// Admits the session against current device memory before prefetch and
/// load; on refusal, says what fits.
fn preflight(
    ctx: &MetalContext,
    gguf: &GgufFile,
    prepared: &Glm5NextPreparedArtifact<'_>,
    args: &Args,
    prompt_tokens: usize,
    capacity: usize,
    prefill_rows: usize,
) -> Result<()> {
    match preflight_session(ctx, gguf, prepared.model(), capacity, prefill_rows) {
        Ok(_) => Ok(()),
        Err(
            error @ Glm5NextMetalError::MemoryAdmission {
                fitting_capacity, ..
            },
        ) => {
            let required = required_forwards(FAMILY, prompt_tokens, args.tokens, None)?;
            bail!("{error}; {}", admission_advice(fitting_capacity, required))
        }
        Err(error) => Err(error).context("admit GLM-5.3 session"),
    }
}

pub(crate) fn run(
    gguf: &GgufFile,
    args: &Args,
    explicit: ExplicitCliOptions,
    invocation: cli::Invocation,
) -> Result<()> {
    let lane_t0 = Instant::now();
    let mut timing = Timing::default();
    let request = timing.measure(Phase::RequestPreparation, || {
        admit(invocation, args, explicit)
    })?;
    let sampling =
        release_sampling_config(SamplingConfig::glm5_next(args.seed), args, explicit, FAMILY)?;
    let layout = timing.measure(Phase::ArtifactLayout, || {
        Glm5NextArtifactLayout::inspect(gguf).with_context(|| format!("admit {FAMILY} artifact"))
    })?;
    let prepared = timing.measure(Phase::TokenizerConstruction, || {
        layout
            .prepare_tokenizer()
            .context("build GLM-5.3 native tokenizer")
    })?;
    let stops = prepared.generation_stops()?;
    let vocab_size = prepared.config().vocab_size;
    let mut tools: Vec<chat::ToolDefinition> = Vec::new();
    let mut chat_effort = None;
    let (text, source, mut chat_record) = match request {
        Request::Raw(invocation) => {
            let (text, source) =
                timing.measure(Phase::InputAcquisition, || raw_input(invocation, args))?;
            (text, source, None)
        }
        Request::Chat { run, effort } => {
            // Bind the artifact's template before reading a file or waiting on stdin.
            let profile = timing.measure(Phase::ArtifactVerification, || {
                prepared
                    .chat_profile()
                    .map_err(|error| anyhow!("{error}; use --raw-prompt"))
            })?;
            shutdown::checkpoint()?;
            let input = timing.measure(Phase::InputAcquisition, || run.acquire_input())?;
            let (text, clear_thinking) = timing.measure(Phase::Rendering, || {
                let (messages, clear_thinking, definitions) = chat_messages(input)?;
                let text = chat::render_with_tools(
                    &messages,
                    &definitions,
                    RenderOptions::generate(effort, clear_thinking),
                )?;
                tools = definitions;
                Ok((text, clear_thinking))
            })?;
            chat_effort = Some(effort);
            let record = json!({
                "profile": profile, "reasoning_effort": effort, "clear_thinking": clear_thinking,
                "stops": chat::CHAT_STOPS, "prefix_owner": "renderer",
                "output": if tools.is_empty() { "reasoning_stderr_answer_stdout" } else { "responses_json" },
                "tools": tools.iter().map(chat::ToolDefinition::name).collect::<Vec<_>>(),
            });
            (text, PromptSource::Messages, Some(record))
        }
    };
    // The glm4 tokenizer never inserts BOS; [gMASK]<sop> belongs in the
    // text, and the chat renderer writes it.
    let add_special = chat_record.is_none() && !args.no_special_tokens;
    let tokens = timing.measure(Phase::Encoding, || {
        prepared
            .tokenizer()
            .encode(&text, add_special)?
            .into_iter()
            .enumerate()
            .map(|(i, id)| checked_token_id(id, vocab_size, &format!("prompt[{i}]")))
            .collect::<Result<Vec<_>>>()
    })?;
    if let Some(record) = &mut chat_record {
        let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        record["prompt_token_ids_sha256_i32le"] =
            json!(qwen_llm::tokenizer::token_ids_sha256_i32le(&ids));
    }
    let (capacity, prefill_rows) = timing.measure(Phase::RequestPreparation, || {
        let capacity = capacity(args, tokens.len(), prepared.config().context_length)?;
        let rows = if prepared.packed_prefill() {
            tokens.len().min(DEFAULT_PREFILL_ROWS)
        } else {
            0
        };
        Ok((capacity, rows))
    })?;
    let mut sampler = Sampler::new(sampling)?;
    shutdown::checkpoint()?;

    let ctx = timing.measure(Phase::DeviceSetup, || {
        MetalContext::new().context("initialize Metal for GLM-5.3")
    })?;
    timing.measure(Phase::MemoryPreflight, || {
        preflight(
            &ctx,
            gguf,
            &prepared,
            args,
            tokens.len(),
            capacity,
            prefill_rows,
        )
    })?;
    // Warm cold retained windows with parallel reads before the zero-copy
    // weights are first touched; demand paging would otherwise land in the
    // first prefill at a fraction of the storage bandwidth.
    let prefetch = timing.measure(Phase::Prefetch, || {
        prefetch_retained_with_cancel(&ctx, gguf, 0.98, &|| shutdown::checkpoint().is_err())
            .context("prefetch GLM-5.3 retained windows")
    })?;
    eprintln!(
        "glm5_next prefetch: windows={} cold_windows={} bytes_read={} wall_ms={:.1}",
        prefetch.windows,
        prefetch.cold_windows,
        prefetch.bytes_read,
        prefetch.wall.as_secs_f64() * 1e3
    );
    let weights = timing.measure(Phase::ModelLoad, || {
        Glm5NextWeights::load(&ctx, gguf).context("load GLM-5.3 weights")
    })?;
    let mut session = timing.measure(Phase::SessionSetup, || {
        Glm5NextSession::with_prefill_rows(&ctx, &weights, capacity, prefill_rows)
            .context("create GLM-5.3 session")
    })?;

    let resident_t0 = Instant::now();
    let logits = session.prefill_packed_with_checkpoint(&ctx, &tokens, &mut || {
        shutdown::checkpoint().map_err(|e| e.to_string())
    })?;
    let prefill_ms = resident_t0.elapsed().as_secs_f64() * 1e3;
    let stdout = std::io::stdout();
    let mut stdout = stdout.lock();
    let mut stderr = std::io::stderr();
    // With declared tools the output is one Responses JSON object (the
    // serve shape, function_call items included), printed at the end.
    let mut tool_partition = (!tools.is_empty())
        .then(|| -> Result<_> {
            let max_bytes = args
                .tokens
                .checked_mul(prepared.tokenizer().max_decoded_piece_bytes())
                .and_then(|n| n.checked_mul(3))
                .context("tool output byte bound overflows")?;
            Ok(crate::serve::output_partition::OutputPartition::new(
                crate::serve::output_partition::OutputProtocol::Glm5NextTools {
                    definitions: tools.clone(),
                    max_bytes,
                },
            ))
        })
        .transpose()?;
    let mut tool_events = Vec::new();
    let mut partition = chat_record
        .as_ref()
        .filter(|_| tool_partition.is_none())
        .map(|_| crate::serve::render_glm5_next::partition());
    let mut visible = false;
    let generation = generate_serial(
        logits,
        args.tokens,
        &stops,
        &mut sampler,
        |token| {
            let bytes = prepared.tokenizer().try_decode_piece_bytes_exact(token)?;
            if let Some(partition) = &mut tool_partition {
                partition.push(bytes, &mut tool_events);
            } else if let Some(partition) = &mut partition {
                let mut events = Vec::new();
                partition.push(bytes, &mut events);
                crate::chat_output::write_chat_events(
                    &events,
                    &mut stdout,
                    &mut stderr,
                    &mut visible,
                    FAMILY,
                )?;
            } else {
                stdout.write_all(bytes)?;
                stdout.flush()?;
                visible |= !bytes.is_empty();
            }
            Ok(())
        },
        |token| {
            shutdown::checkpoint()?;
            let token = checked_token_id(token, vocab_size, "generated")?;
            session.forward(&ctx, token).map_err(anyhow::Error::from)
        },
    );
    let generation = match generation {
        Ok(generation) => generation,
        Err(error) => {
            // Leave the terminal on a fresh line; the original error stands.
            if partition.is_some() {
                let _ = writeln!(stderr);
            }
            if visible {
                let _ = writeln!(stdout);
                let _ = stdout.flush();
            }
            return Err(error);
        }
    };
    timing.record(Phase::ResidentExecution, resident_t0.elapsed())?;
    let report = timing.finish(lane_t0.elapsed())?;
    if let Some(partition) = tool_partition {
        let (stop, end) = crate::serve::outcome::generation_end(&generation);
        partition
            .finish(end, &mut tool_events)
            .map_err(|error| anyhow!(error.message))?;
        let effort = chat_effort.context("tools imply a chat request")?;
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?;
        let request = crate::serve::items::ServeRequest {
            model: gguf.get_str("general.name").unwrap_or(FAMILY).into(),
            model_request: crate::model_request::ModelRequest {
                tools: tools
                    .iter()
                    .map(|tool| crate::model_request::ToolDefinition {
                        name: tool.name().into(),
                        description: tool.description().map(str::to_owned),
                        parameters: tool
                            .parameters()
                            .cloned()
                            .unwrap_or(serde_json::Value::Null),
                        strict: None,
                    })
                    .collect(),
                ..Default::default()
            },
            allowed_tools: tools.iter().map(|tool| tool.name().to_owned()).collect(),
            parallel_tool_calls: true,
            tool_choice: json!("auto"),
            reasoning: Some(json!({"effort": effort.as_str()})),
            max_output_tokens: Some(args.tokens),
            temperature_echo: Some(f64::from(sampling.temperature)),
            top_p_echo: Some(f64::from(sampling.top_p)),
            ..Default::default()
        };
        let response = crate::serve::events::build_response_object(
            &request,
            format!("resp_glm_cli_{}_{}", std::process::id(), now.as_nanos()),
            now.as_secs(),
            &tool_events,
            stop,
            crate::serve::events::Usage {
                input_tokens: tokens.len(),
                output_tokens: generation.tokens.len(),
                cached_tokens: 0,
            },
            None,
        )?;
        serde_json::to_writer(&mut stdout, &response)?;
        writeln!(stdout)?;
        stdout.flush()?;
    }
    if let Some(partition) = partition {
        let mut events = Vec::new();
        let closed_before = partition.closed();
        let result = partition.finish(
            crate::serve::outcome::generation_end(&generation).1,
            &mut events,
        );
        let closed = closed_before
            || events.iter().any(|event| {
                matches!(
                    event,
                    crate::serve::partition::PartitionEvent::ReasoningClosed
                )
            });
        crate::chat_output::write_chat_events(
            &events,
            &mut stdout,
            &mut stderr,
            &mut visible,
            FAMILY,
        )?;
        writeln!(stderr)?;
        if let Some(record) = &mut chat_record {
            record["reasoning_closed"] = json!(closed);
        }
        result.map_err(|error| anyhow!(error.message))?;
        if !closed {
            writeln!(
                stderr,
                "glm5_next: incomplete response: token budget exhausted before reasoning closed; no final answer"
            )?;
        }
    }
    if visible {
        writeln!(stdout)?;
        stdout.flush()?;
    }

    let prefill_mode = if prefill_rows > 0 {
        "packed_fast"
    } else {
        "serial"
    };
    let prefill_tps = tokens.len() as f64 / (prefill_ms / 1e3).max(f64::MIN_POSITIVE);
    let decode_tps =
        generation.tokens.len() as f64 / (generation.wall_ms / 1e3).max(f64::MIN_POSITIVE);
    eprintln!(
        "glm5_next: input={} prompt_tokens={} generated_tokens={} transitions={} stop={} capacity={capacity} prefill={prefill_mode} prefill_rows={prefill_rows} setup_prefetch_ms={:.1} load_ms={:.1} prefill_ms={prefill_ms:.1} prefill_tps={prefill_tps:.2} decode_tps={decode_tps:.2} loaded_request_ms={:.1} end_to_end_ms={:.1}",
        chat_record
            .as_ref()
            .map_or("raw".to_string(), |record| format!(
                "chat effort={}",
                record["reasoning_effort"].as_str().unwrap_or("?")
            )),
        tokens.len(),
        generation.tokens.len(),
        generation.transitions,
        generation.stop_reason.as_str(),
        prefetch.wall.as_secs_f64() * 1e3,
        report.load_ms,
        report.loaded_request_ms,
        report.json["end_to_end_lane_ms"].as_f64().unwrap_or(0.0),
    );
    if let Some(path) = args.request_stats_jsonl.as_ref() {
        let measured = RequestStatsMeasured {
            input_tokens: tokens.len() as u64,
            output_tokens: generation.tokens.len() as u64,
            transitions: generation.transitions as u64,
            stop_reason: generation.stop_reason,
            tokenizer_ms: report.encoding_ms,
            load_ms: report.load_ms,
            prefill_ms,
            prefill_tps,
            decode_ms: generation.wall_ms,
            decode_tps,
            transition_tps: generation.transitions as f64
                / (generation.transition_ms / 1e3).max(f64::MIN_POSITIVE),
            total_ms: report.loaded_request_ms,
            output_fingerprint: GeneratedTokenSha256Digest::of(&generation.tokens),
        };
        append_single_turn_stats_record(
            path,
            0,
            ModelFamily::Glm5Next.record_label(),
            request_stats_input(source, chat_record.as_ref().map(|_| chat::RENDERER)),
            &measured,
            Some(RequestStatsDiagnostics {
                deepseek_v4: None,
                k2_horizon: None,
                glm5_next: Some(RequestStatsGlm5NextDiagnostics {
                    schema_version: 2,
                    prefill_mode,
                    prefill_rows: prefill_rows as u64,
                    capacity: capacity as u64,
                    prefetch: json!({
                        "windows": prefetch.windows,
                        "cold_windows": prefetch.cold_windows,
                        "bytes_read": prefetch.bytes_read,
                    }),
                    timing: report.json,
                    sampling: sampling_json(sampling),
                    chat: chat_record,
                }),
            }),
        )?;
    }
    Ok(())
}

fn sampling_json(config: SamplingConfig) -> Value {
    json!({"temperature": config.temperature, "top_k": config.top_k, "top_p": config.top_p,
        "min_p": config.min_p, "seed": config.seed})
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(argv: &[&str]) -> (Args, ExplicitCliOptions, cli::Invocation) {
        let (mut args, explicit) = Args::parse_with_explicit(argv);
        let invocation = cli::normalize(&mut args);
        invocation.apply_option_overrides(&mut args);
        (args, explicit, invocation)
    }

    #[test]
    fn setup_phases_never_reach_the_loaded_request() {
        use std::time::Duration;
        let example = |extra: Option<Phase>| {
            let mut timing = Timing::default();
            for phase in [
                Phase::Encoding,
                Phase::RequestPreparation,
                Phase::ResidentExecution,
            ] {
                timing.record(phase, Duration::from_millis(10)).unwrap();
            }
            let mut wall = Duration::from_millis(31);
            if let Some(phase) = extra {
                timing.record(phase, Duration::from_secs(2)).unwrap();
                wall += Duration::from_secs(2);
            }
            timing.finish(wall).unwrap()
        };
        let base = example(None);
        assert_eq!(base.loaded_request_ms, 30.0);
        assert_eq!(base.encoding_ms, 10.0);
        for phase in [
            Phase::InputAcquisition,
            Phase::ArtifactLayout,
            Phase::TokenizerConstruction,
            Phase::DeviceSetup,
            Phase::MemoryPreflight,
            Phase::Prefetch,
            Phase::ModelLoad,
            Phase::SessionSetup,
        ] {
            let report = example(Some(phase));
            assert_eq!(report.loaded_request_ms, 30.0);
            assert_eq!(report.encoding_ms, 10.0);
            assert_eq!(report.json["unclassified_host_overhead_ms"], 1.0);
        }
        assert_eq!(example(Some(Phase::ModelLoad)).load_ms, 2000.0);
        assert_eq!(example(Some(Phase::SessionSetup)).load_ms, 2000.0);
        assert_eq!(example(Some(Phase::Prefetch)).load_ms, 0.0);
        assert_eq!(
            example(Some(Phase::Prefetch)).json["phases_ms"]["prefetch"],
            2000.0
        );
        // Phases that exceed the lane wall are refused.
        let mut timing = Timing::default();
        timing
            .record(Phase::Encoding, Duration::from_secs(2))
            .unwrap();
        assert!(timing.finish(Duration::from_secs(1)).is_err());
    }

    #[test]
    fn admission_advice_names_what_fits() {
        assert_eq!(
            admission_advice(Some(649_040), 4_000),
            "pass --max-context-tokens 649040 or less"
        );
        assert!(admission_advice(Some(3_000), 4_000).contains("needs 4000 positions"));
        assert!(admission_advice(None, 10).contains("free device memory"));
    }

    fn run_args(extra: &[&str]) -> Vec<String> {
        ["qwen", "run", "-m", "m.gguf"]
            .iter()
            .chain(extra)
            .map(|s| s.to_string())
            .collect()
    }

    fn admitted(extra: &[&str]) -> Result<Request> {
        let argv = run_args(extra);
        let argv: Vec<&str> = argv.iter().map(String::as_str).collect();
        let (args, explicit, invocation) = parse(&argv);
        admit(invocation, &args, explicit)
    }

    #[test]
    fn chat_and_raw_requests_settle_their_controls_before_any_input() {
        for (extra, effort) in [
            (&["--user", "hi"][..], Effort::Max),
            (&["--user", "hi", "--reasoning-effort", "low"], Effort::Low),
            (
                &["--user", "hi", "--reasoning-effort", "high"],
                Effort::High,
            ),
            (
                &[
                    "--messages",
                    "/nonexistent.json",
                    "--reasoning-effort",
                    "max",
                ],
                Effort::Max,
            ),
        ] {
            match admitted(extra).unwrap() {
                Request::Chat { effort: got, .. } => assert_eq!(got, effort, "{extra:?}"),
                Request::Raw(_) => panic!("{extra:?} is chat"),
            }
        }
        for (extra, needle) in [
            (
                &["--user", "hi", "--reasoning-effort", "medium"][..],
                "low, high or max",
            ),
            (
                &["--user", "hi", "--reasoning-effort", "none"],
                "no non-thinking mode",
            ),
            (&["--user", "hi", "--no-thinking"], "no --no-thinking mode"),
            (&["--user", "hi", "--drafter", "d.gguf"], "--drafter"),
        ] {
            let error = format!("{:#}", admitted(extra).err().unwrap());
            assert!(error.contains(needle), "{extra:?}: {error}");
        }
        let Request::Raw(invocation) = admitted(&["--raw-prompt", "[gMASK]<sop>hi"]).unwrap()
        else {
            panic!("raw prompt is raw");
        };
        let (args, ..) = parse(&["qwen", "run", "-m", "m.gguf", "--raw-prompt", "x"]);
        assert_eq!(raw_input(invocation, &args).unwrap().0, "[gMASK]<sop>hi");
    }

    #[test]
    fn user_and_system_flags_become_the_conversation() {
        let (messages, clear, _) = chat_messages(cli::AcquiredRunInput::User {
            system: Some("Be brief.".into()),
            user: "hi".into(),
        })
        .unwrap();
        assert!(!clear);
        assert_eq!(
            messages,
            [
                Message::System("Be brief.".into()),
                Message::User("hi".into())
            ]
        );
        let (messages, clear, _) = chat_messages(cli::AcquiredRunInput::Messages {
            document: r#"{"messages":[{"role":"user","content":"q"}],"clear_thinking":true}"#
                .into(),
            source: "test".into(),
        })
        .unwrap();
        assert!(clear);
        assert_eq!(messages, [Message::User("q".into())]);
        let (messages, _, tools) = chat_messages(cli::AcquiredRunInput::Messages {
            document: r#"{"messages":[{"role":"user","content":"q"}],"tools":[]}"#.into(),
            source: "test".into(),
        })
        .unwrap();
        assert_eq!(messages, [Message::User("q".into())]);
        assert!(tools.is_empty());
        let (_, _, tools) = chat_messages(cli::AcquiredRunInput::Messages {
            document: r#"{"messages":[{"role":"user","content":"q"}],"tools":[{"name":"f"}]}"#
                .into(),
            source: "test".into(),
        })
        .unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name(), "f");
        let error = chat_messages(cli::AcquiredRunInput::Messages {
            document: r#"{"messages":[{"role":"user","content":"q"}],"tools":[{"name":"f","strict":true}]}"#
                .into(),
            source: "test".into(),
        })
        .unwrap_err();
        assert!(format!("{error:#}").contains("strict"), "{error:#}");
    }

    #[test]
    fn sampling_defaults_to_the_release_config_and_flags_override_it() {
        let config = |extra: &[&str]| {
            let argv = run_args(extra);
            let argv: Vec<&str> = argv.iter().map(String::as_str).collect();
            let (args, explicit, _) = parse(&argv);
            release_sampling_config(
                SamplingConfig::glm5_next(args.seed),
                &args,
                explicit,
                FAMILY,
            )
            .unwrap()
        };
        let release = config(&["--user", "hi"]);
        assert_eq!(release, SamplingConfig::glm5_next(42));
        assert_eq!((release.temperature, release.top_p), (1.0, 0.95));
        assert_eq!((release.top_k, release.min_p), (0, 0.0));
        assert_eq!(config(&["--raw-prompt", "x"]), release);
        let greedy = config(&["--raw-prompt", "x", "--temp", "0", "--seed", "7"]);
        assert_eq!(
            greedy,
            SamplingConfig {
                temperature: 0.0,
                seed: 7,
                ..release
            }
        );
        let custom = config(&[
            "--user", "hi", "--top-k", "40", "--min-p", "0.05", "--top-p", "1",
        ]);
        assert_eq!((custom.top_k, custom.min_p, custom.top_p), (40, 0.05, 1.0));
    }

    #[test]
    fn capacity_stays_within_the_checkpoint_context() {
        let (args, ..) = parse(&[
            "qwen",
            "run",
            "-m",
            "m.gguf",
            "--raw-prompt",
            "x",
            "-n",
            "16",
        ]);
        assert_eq!(capacity(&args, 10, 1 << 20).unwrap(), 25);
        // Past the sparse frontier is admitted; past the checkpoint is not.
        assert_eq!(capacity(&args, 4000, 1 << 20).unwrap(), 4015);
        let error = capacity(&args, 2040, 2048).unwrap_err().to_string();
        assert!(error.contains("at most 2048"), "{error}");
        let (args, ..) = parse(&[
            "qwen",
            "run",
            "-m",
            "m.gguf",
            "--raw-prompt",
            "x",
            "-n",
            "16",
            "--max-context-tokens",
            "20",
        ]);
        assert!(capacity(&args, 10, 1 << 20).is_err());
    }
}
