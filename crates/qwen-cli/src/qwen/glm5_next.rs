//! GLM-5.3-Flash: bounded raw generation up to the checkpoint context
//! (dense attention below visible length 2052, sparse DSA selection from
//! there on) within device memory. Chat templating, serve and lens are not
//! implemented; every other surface refuses rather than falling through to
//! Qwen protocols.

use super::*;
use crate::lane_timing::{LanePhases, LaneTiming};
use qwen_llm::glm5_next::{
    Glm5NextAdmissionError, Glm5NextArtifactLayout, Glm5NextPreparedArtifact,
};
use qwen_llm::glm5_next_metal::{
    DEFAULT_PREFILL_ROWS, Glm5NextMetalError, Glm5NextSession, Glm5NextWeights,
    prefetch_retained_with_cancel, preflight_session,
};
use serde_json::{Value, json};

const FAMILY: &str = "GLM-5.3-Flash";

/// Request phases of the GLM run lane. Setup (artifact, tokenizer, device,
/// memory preflight, prefetch) stays out of the loaded request, which is
/// encoding, request preparation and resident execution.
#[derive(Clone, Copy)]
enum Phase {
    InputAcquisition,
    ArtifactLayout,
    TokenizerConstruction,
    Encoding,
    RequestPreparation,
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
        (Phase::InputAcquisition, "input_acquisition"),
        (Phase::ArtifactLayout, "artifact_layout"),
        (Phase::TokenizerConstruction, "tokenizer_construction"),
        (Phase::Encoding, "encoding"),
        (Phase::RequestPreparation, "request_preparation"),
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

/// CPU preparation shared by run, info and bench: binding, execution
/// coverage, tokenizer and stop set, with stable refusal codes.
fn admission(
    gguf: &GgufFile,
) -> std::result::Result<(Glm5NextPreparedArtifact<'_>, Vec<i32>), Glm5NextAdmissionError> {
    let prepared = Glm5NextPreparedArtifact::inspect(gguf)?;
    let stops = prepared.generation_stops()?;
    Ok((prepared, stops))
}

pub(crate) fn capability_projection(gguf: &GgufFile) -> Result<Value> {
    let admission = admission(gguf);
    let run = match &admission {
        Ok((prepared, _)) => json!({
            "status": "conditional", "implementation_status": "partial", "scope": "raw_only",
            "artifact_admission": {"status": "passed"},
            "capacity_policy": format!("checkpoint_context_{}_and_device_memory", prepared.config().context_length),
            "attention": "dense_below_2052_sparse_dsa_from_2052",
            "prefill": if prepared.packed_prefill() { "packed_fast" } else { "serial" },
            "native_tokenizer": true, "latent_cache": "f16",
        }),
        Err(error) => json!({
            "status": "unsupported", "implementation_status": "partial",
            "artifact_admission": {"status": "rejected", "code": error.code(), "message": error.to_string()},
        }),
    };
    let bench = match &admission {
        Ok(_) => json!({"status": "conditional", "implementation_status": "partial",
            "command": "qwen-bench suite", "scope": "packed_prefill_and_serial_decode_rows"}),
        Err(_) => json!({"status": "unsupported", "implementation_status": "partial"}),
    };
    let unsupported = |lane: &str| {
        json!({"status": "unsupported", "implementation_status": "unsupported",
            "code": "glm5_next_lane_unimplemented",
            "message": format!("{FAMILY} has no {lane} implementation yet")})
    };
    let raw = match &admission {
        Ok(_) => json!({"status": "supported"}),
        Err(error) => {
            json!({"status": "unsupported", "code": error.code(), "message": error.to_string()})
        }
    };
    let templated = json!({"status": "unsupported", "code": "glm5_next_chat_unimplemented",
        "message": format!("{FAMILY} chat rendering is not implemented; use --raw-prompt")});
    Ok(json!({
        "execution": {"run": run, "bench": bench, "serve": unsupported("serve"), "lens": unsupported("lens"),
            "request_device": {"status": "not_evaluated",
                "requires": ["request_options_and_token_budget", "live_memory_admission"]}},
        "input": {"raw": raw, "user": templated, "messages": templated, "tools": templated},
        "reasoning": {"levels": [], "fallback": null,
            "no_thinking": {"status": "unsupported", "code": "glm5_next_chat_unimplemented", "message": "raw input has no reasoning controls"},
            "thinking": {"status": "unsupported", "code": "glm5_next_chat_unimplemented", "message": "raw input has no reasoning controls"}},
        "template": {"status": "unsupported", "rendered_as": null,
            "message": format!("{FAMILY} raw input has no template renderer yet")},
    }))
}

fn prepare_raw(
    invocation: cli::Invocation,
    args: &Args,
    explicit: ExplicitCliOptions,
) -> Result<(String, PromptSource)> {
    admission::GLM5_NEXT_RAW_SINGLE_TURN.admit(&admission::supplied(args, explicit))?;
    ensure!(
        args.drafter.is_none(),
        "{FAMILY} does not support --drafter"
    );
    ensure!(
        args.messages.is_none(),
        "{FAMILY} currently supports raw input only"
    );
    match invocation {
        cli::Invocation::Run(run) => {
            ensure!(
                !run.no_thinking && run.reasoning_effort.is_none(),
                "{FAMILY} raw input does not accept reasoning controls"
            );
            match run.input {
                cli::RunInput::RawPrompt(text) => Ok((text, PromptSource::Inline)),
                _ => bail!(
                    "{FAMILY} chat rendering is not implemented; use --raw-prompt (include [gMASK]<sop> for the release prefix)"
                ),
            }
        }
        cli::Invocation::Legacy => {
            let (text, source, _) = prompt_text(args)?;
            Ok((text, source))
        }
        _ => bail!("{FAMILY} raw preparation is a single-turn generation path"),
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

pub(crate) fn run_raw(
    gguf: &GgufFile,
    args: &Args,
    explicit: ExplicitCliOptions,
    invocation: cli::Invocation,
) -> Result<()> {
    let lane_t0 = Instant::now();
    let mut timing = Timing::default();
    let (text, source) = timing.measure(Phase::InputAcquisition, || {
        prepare_raw(invocation, args, explicit)
    })?;
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
    // The glm4 tokenizer never inserts BOS; [gMASK]<sop> belongs in the text.
    let tokens = timing.measure(Phase::Encoding, || {
        prepared
            .tokenizer()
            .encode(&text, !args.no_special_tokens)?
            .into_iter()
            .enumerate()
            .map(|(i, id)| checked_token_id(id, vocab_size, &format!("prompt[{i}]")))
            .collect::<Result<Vec<_>>>()
    })?;
    let (capacity, prefill_rows) = timing.measure(Phase::RequestPreparation, || {
        let capacity = capacity(args, tokens.len(), prepared.config().context_length)?;
        let rows = if prepared.packed_prefill() {
            tokens.len().min(DEFAULT_PREFILL_ROWS)
        } else {
            0
        };
        Ok((capacity, rows))
    })?;
    let mut sampler = Sampler::new(cli_sampling_config(args)?)?;
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
    let generation = generate_serial(
        logits,
        args.tokens,
        &stops,
        &mut sampler,
        |token| {
            stdout.write_all(prepared.tokenizer().try_decode_piece_bytes_exact(token)?)?;
            stdout.flush()?;
            Ok(())
        },
        |token| {
            shutdown::checkpoint()?;
            let token = checked_token_id(token, vocab_size, "generated")?;
            session.forward(&ctx, token).map_err(anyhow::Error::from)
        },
    )?;
    if !generation.tokens.is_empty() {
        writeln!(stdout)?;
        stdout.flush()?;
    }
    timing.record(Phase::ResidentExecution, resident_t0.elapsed())?;
    let report = timing.finish(lane_t0.elapsed())?;

    let prefill_mode = if prefill_rows > 0 {
        "packed_fast"
    } else {
        "serial"
    };
    let prefill_tps = tokens.len() as f64 / (prefill_ms / 1e3).max(f64::MIN_POSITIVE);
    let decode_tps =
        generation.tokens.len() as f64 / (generation.wall_ms / 1e3).max(f64::MIN_POSITIVE);
    eprintln!(
        "glm5_next: prompt_tokens={} generated_tokens={} transitions={} stop={} capacity={capacity} prefill={prefill_mode} prefill_rows={prefill_rows} setup_prefetch_ms={:.1} load_ms={:.1} prefill_ms={prefill_ms:.1} prefill_tps={prefill_tps:.2} decode_tps={decode_tps:.2} loaded_request_ms={:.1} end_to_end_ms={:.1}",
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
            request_stats_input(source, None),
            &measured,
            Some(RequestStatsDiagnostics {
                deepseek_v4: None,
                k2_horizon: None,
                glm5_next: Some(RequestStatsGlm5NextDiagnostics {
                    schema_version: 1,
                    prefill_mode,
                    prefill_rows: prefill_rows as u64,
                    capacity: capacity as u64,
                    prefetch: json!({
                        "windows": prefetch.windows,
                        "cold_windows": prefetch.cold_windows,
                        "bytes_read": prefetch.bytes_read,
                    }),
                    timing: report.json,
                }),
            }),
        )?;
    }
    Ok(())
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

    #[test]
    fn raw_lane_refuses_templated_input() {
        let (args, explicit, invocation) =
            parse(&["qwen", "run", "-m", "m.gguf", "--user", "hello"]);
        let error = prepare_raw(invocation, &args, explicit)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("chat rendering is not implemented"),
            "{error}"
        );
        let (args, explicit, invocation) = parse(&[
            "qwen",
            "run",
            "-m",
            "m.gguf",
            "--raw-prompt",
            "[gMASK]<sop>hi",
        ]);
        assert_eq!(
            prepare_raw(invocation, &args, explicit).unwrap().0,
            "[gMASK]<sop>hi"
        );
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
