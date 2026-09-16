//! Qwen3.8 Flash-Next single-turn path and profiles.

use super::*;

pub(crate) const QWEN4EXP_LAYER_PROFILE_ENV: &str = "QWEN4EXP_LAYER_PROFILE";

pub(crate) const QWEN4EXP_PACKED_PREFILL_PROFILE_ENV: &str = "QWEN4EXP_PACKED_PREFILL_PROFILE";

pub(crate) const QWEN4EXP_PACKED_SELECTED_QSA_ENV: &str = "QWEN4EXP_PACKED_SELECTED_QSA";

pub(crate) const QWEN4EXP_FULL_SHARD_PREFETCH_ENV: &str = "QWEN4EXP_FULL_SHARD_PREFETCH";
pub(crate) const QWEN4EXP_QSA_SPLIT_DECODE_ENV: &str = "QWEN4EXP_QSA_SPLIT_DECODE";
pub(crate) const QWEN4EXP_HC_UP_MIX_ENV: &str = "QWEN4EXP_HC_UP_MIX";
pub(crate) const QWEN4EXP_GUARDED_TOPK_ENV: &str = "QWEN4EXP_GUARDED_TOPK";

pub(crate) fn parse_qwen4exp_split_decode(value: Option<&std::ffi::OsStr>) -> Result<bool> {
    parse_qwen4exp_decode_flag(value, QWEN4EXP_QSA_SPLIT_DECODE_ENV)
}

pub(crate) fn parse_qwen4exp_decode_flag(
    value: Option<&std::ffi::OsStr>,
    name: &str,
) -> Result<bool> {
    match value {
        None => Ok(true),
        Some(value) if value == "0" => Ok(false),
        Some(value) if value == "1" => Ok(true),
        _ => bail!("{name} must be 0 or 1"),
    }
}

pub(crate) const QWEN4EXP_MAX_STOP_TOKENS: usize = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Qwen4ExpPromptCapabilityFailure {
    Architecture,
    TokenizerModel,
    Pretokenizer,
    ChatTemplate,
}

impl Qwen4ExpPromptCapabilityFailure {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Architecture => "general.architecture",
            Self::TokenizerModel => "tokenizer.ggml.model",
            Self::Pretokenizer => "tokenizer.ggml.pre",
            Self::ChatTemplate => "tokenizer.chat_template",
        }
    }
}

pub(crate) fn qwen4exp_prompt_capability_failure(
    family: ModelFamily,
    gguf: &GgufFile,
) -> Option<Qwen4ExpPromptCapabilityFailure> {
    classify_qwen4exp_prompt_capability(
        family,
        gguf.get_str("tokenizer.ggml.model"),
        gguf.get_str("tokenizer.ggml.pre"),
        gguf.get_str("tokenizer.chat_template")
            .is_some_and(qwen4exp_chat_template_matches),
    )
}

pub(crate) fn classify_qwen4exp_prompt_capability(
    family: ModelFamily,
    tokenizer_model: Option<&str>,
    tokenizer_pre: Option<&str>,
    supported_chat_template: bool,
) -> Option<Qwen4ExpPromptCapabilityFailure> {
    if family != ModelFamily::Qwen4Exp {
        return Some(Qwen4ExpPromptCapabilityFailure::Architecture);
    }
    if tokenizer_model != Some("gpt2") {
        return Some(Qwen4ExpPromptCapabilityFailure::TokenizerModel);
    }
    if tokenizer_pre != Some("qwen35") {
        return Some(Qwen4ExpPromptCapabilityFailure::Pretokenizer);
    }
    if !supported_chat_template {
        return Some(Qwen4ExpPromptCapabilityFailure::ChatTemplate);
    }
    None
}

pub(crate) fn qwen4exp_chat_template_matches(template: &str) -> bool {
    messages::qwen4exp_chat_template_matches(template)
}

pub(crate) fn validate_qwen4exp_generation_mode(
    args: &Args,
    explicit: ExplicitCliOptions,
) -> Result<()> {
    let mut unsupported = serial_lane_unsupported_options(args, explicit);
    ensure_no_deepseek_v4_only_options(args, explicit, &mut unsupported)?;
    ensure!(
        unsupported.is_empty(),
        "Qwen3.8-Flash-Next currently supports request-shaped serial single-turn generation only; unsupported options: {}",
        unsupported.join(", ")
    );
    ensure!(
        has_single_turn_input(args),
        "Qwen3.8-Flash-Next generation requires --prompt, --prompt-file, --messages, or `qwen run --user`"
    );
    ensure!(
        args.prepared_prompt.is_some() || args.messages.is_none(),
        "Qwen3.8-Flash-Next legacy --messages rendering is not supported; use `qwen run --messages`"
    );
    Ok(())
}

pub(crate) fn validate_qwen4exp_stop_tokens(stop_tokens: &[i32], vocab_size: u32) -> Result<()> {
    ensure!(
        !stop_tokens.is_empty(),
        "Qwen3.8-Flash-Next producer must declare at least one stop token"
    );
    ensure!(
        stop_tokens.len() <= QWEN4EXP_MAX_STOP_TOKENS,
        "Qwen3.8-Flash-Next producer declares {} stop tokens, above the supported maximum {QWEN4EXP_MAX_STOP_TOKENS}",
        stop_tokens.len()
    );
    for &token in stop_tokens {
        checked_token_id(token, vocab_size, "stop")?;
    }
    Ok(())
}

#[derive(Default)]
pub(crate) struct Qwen4ExpTimingTotals {
    pub(crate) forwards: usize,
    pub(crate) encode_cpu_ms: f64,
    pub(crate) completion_wait_ms: f64,
    pub(crate) gpu_ms: f64,
    pub(crate) gpu_samples: usize,
    pub(crate) total_wall_ms: f64,
}

impl Qwen4ExpTimingTotals {
    pub(crate) fn record(&mut self, timing: Qwen4ExpTokenTiming) {
        self.forwards += 1;
        self.encode_cpu_ms += timing.encode_cpu_ms;
        self.completion_wait_ms += timing.completion_wait_ms;
        self.total_wall_ms += timing.total_wall_ms;
        if let Some(gpu_ms) = timing.gpu_ms {
            self.gpu_ms += gpu_ms;
            self.gpu_samples += 1;
        }
    }

    pub(crate) fn record_prefill(&mut self, timing: Qwen4ExpPrefillTiming) {
        self.forwards += timing.command_count;
        self.encode_cpu_ms += timing.encode_cpu_ms;
        self.completion_wait_ms += timing.completion_wait_ms;
        self.total_wall_ms += timing.total_wall_ms;
        self.gpu_ms += timing.gpu_ms;
        self.gpu_samples += timing.gpu_samples;
    }

    pub(crate) fn outside_gpu_ms(&self) -> Option<f64> {
        self.complete_gpu_ms()
            .map(|gpu_ms| (self.total_wall_ms - gpu_ms).max(0.0))
    }

    pub(crate) fn complete_gpu_ms(&self) -> Option<f64> {
        (self.gpu_samples == self.forwards).then_some(self.gpu_ms)
    }
}

pub(crate) fn emit_qwen4exp_layer_profile(profile: &Qwen4ExpLayerProfile) {
    let mut gdn_ms = 0.0;
    let mut gdn_layers = 0;
    let mut qsa_ms = 0.0;
    let mut qsa_layers = 0;
    for stage in &profile.stages {
        let (name, layer, mixer) = match stage.stage {
            Qwen4ExpLayerStage::LayersZeroOne => ("layers_zero_one", None, "bootstrap"),
            Qwen4ExpLayerStage::PostPle { layer, mixer } => match mixer {
                qwen_llm::qwen4exp::MixerKind::GatedDeltaNet => {
                    gdn_ms += stage.gpu_ms;
                    gdn_layers += 1;
                    ("post_ple", Some(layer), "gdn")
                }
                qwen_llm::qwen4exp::MixerKind::QwenSparseAttention => {
                    qsa_ms += stage.gpu_ms;
                    qsa_layers += 1;
                    ("post_ple", Some(layer), "qsa")
                }
            },
            Qwen4ExpLayerStage::Tail => ("tail", None, "final_hc_logits"),
        };
        eprintln!(
            "qwen4exp layer_profile_stage: position={} stage={} layer={layer:?} mixer={} gpu_ms={:.3} fraction={:.6} ticks={}",
            profile.token.position,
            name,
            mixer,
            stage.gpu_ms,
            stage.fraction_of_gpu,
            stage.duration_ticks,
        );
    }
    eprintln!(
        "qwen4exp layer_profile: position={} command_gpu_ms={:.3} sampled_span_ticks={} encoder_boundary_ms={:.3} gdn_layers={} gdn_ms={:.3} qsa_layers={} qsa_ms={:.3}",
        profile.token.position,
        profile.token.gpu_ms.unwrap_or(0.0),
        profile.sampled_span_ticks,
        profile.encoder_boundary_ms,
        gdn_layers,
        gdn_ms,
        qsa_layers,
        qsa_ms,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_qwen4exp_packed_profile(
    outcome: &Qwen4ExpPackedProfileOutcome,
    first: Qwen4ExpPrefillTiming,
    warm: Qwen4ExpPrefillTiming,
    first_copy_ms: f64,
    warm_copy_ms: f64,
    profiled_copy_ms: f64,
    first_reset_ms: f64,
    warm_reset_ms: f64,
    packet_ms: f64,
) {
    for (pass, timing, copy_ms, reset_ms) in [
        ("first", first, first_copy_ms, Some(first_reset_ms)),
        ("warm", warm, warm_copy_ms, Some(warm_reset_ms)),
    ] {
        eprintln!(
            "qwen4exp packed_profile_pass: pass={pass} tokens={} commands={} encode_cpu_ms={:.3} completion_wait_ms={:.3} gpu_ms={:?} gpu_samples={}/{} outside_gpu_ms={:?} runtime_wall_ms={:.3} logits_copy_ms={copy_ms:.3} reset_ms={reset_ms:?}",
            timing.token_count,
            timing.command_count,
            timing.encode_cpu_ms,
            timing.completion_wait_ms,
            timing.complete_gpu_ms(),
            timing.gpu_samples,
            timing.command_count,
            timing.outside_gpu_ms(),
            timing.total_wall_ms,
        );
    }
    let observer_gpu_ratio = warm
        .complete_gpu_ms()
        .zip(outcome.token.gpu_ms)
        .map(|(baseline, profiled)| profiled / baseline);
    let observer_wall_ratio = outcome.token.total_wall_ms / warm.total_wall_ms;
    let observer_gpu_accepted =
        observer_gpu_ratio.is_some_and(|ratio| (0.985..=1.015).contains(&ratio));
    let observer_wall_accepted = (0.98..=1.02).contains(&observer_wall_ratio);
    let observer_accepted = observer_gpu_accepted && observer_wall_accepted;
    eprintln!(
        "qwen4exp packed_profile: sampling={} sampling_fallback={:?} position={} command_gpu_ms={:?} runtime_wall_ms={:.3} encode_cpu_ms={:.3} preflight_ms={:.3} stage_inputs_ms={:.3} graph_encode_ms={:.3} encode_unattributed_ms={:.3} commit_return_ms={:.3} root_wait_ms={:.3} child_publication_ms={:.3} root_publish_ms={:.3} release_total_ms={:.3} observer_gpu_ratio={observer_gpu_ratio:?} observer_wall_ratio={observer_wall_ratio:.6} observer_accepted={observer_accepted} logits_copy_ms={profiled_copy_ms:.3} packet_ms={packet_ms:.3}",
        outcome.sampling.as_str(),
        outcome.sampling_fallback.as_deref(),
        outcome.token.position,
        outcome.token.gpu_ms,
        outcome.token.total_wall_ms,
        outcome.token.encode_cpu_ms,
        outcome.encode.preflight_ms,
        outcome.encode.stage_inputs_ms,
        outcome.encode.graph_encode_ms,
        outcome.encode.unattributed_ms,
        outcome.command.commit_return_ms,
        outcome.command.root_wait_ms,
        outcome.command.child_publication_ms,
        outcome.command.root_publish_ms,
        outcome.command.release_total_ms,
    );
    if !observer_accepted {
        eprintln!(
            "qwen4exp packed_profile_warning: observer acceptance failed; gpu_ratio={observer_gpu_ratio:?} wall_ratio={observer_wall_ratio:.6}"
        );
    }
}

pub(crate) fn emit_qwen4exp_packed_timestamp_profile(profile: &Qwen4ExpPackedPrefillProfile) {
    let coarse_gpu_ms = profile
        .stages
        .iter()
        .filter(|stage| stage.label.scope == Qwen4ExpPackedProfileScope::Coarse)
        .map(|stage| stage.gpu_ms)
        .sum::<f64>();
    let raw_accepted = (0.995..=1.005).contains(&profile.raw_coverage_assuming_ns);
    eprintln!(
        "qwen4exp packed_profile_timestamps: sampling={} sample_count={} sampled_span_ticks={} raw_span_ms_assuming_ns={:.3} raw_coverage_assuming_ns={:.6} raw_accepted={raw_accepted} coarse_gpu_ms={coarse_gpu_ms:.3}",
        profile.sampling.as_str(),
        profile.sample_count,
        profile.sampled_span_ticks,
        profile.raw_span_ms_assuming_ns,
        profile.raw_coverage_assuming_ns,
    );
    if !raw_accepted {
        eprintln!(
            "qwen4exp packed_profile_warning: raw timestamp coverage failed; coverage={:.6}",
            profile.raw_coverage_assuming_ns,
        );
    }
    if profile.sampling == Qwen4ExpPackedProfileSampling::EncoderStage {
        for parent in profile.stages.iter().filter(|stage| {
            (stage.label.scope == Qwen4ExpPackedProfileScope::Coarse
                && stage.label.name == "post_ple_layer")
                || (stage.label.scope == Qwen4ExpPackedProfileScope::Detail
                    && matches!(stage.label.name, "block.moe" | "moe.routing"))
        }) {
            emit_qwen4exp_packed_detail_residual(profile, parent);
        }
    }
    for stage in &profile.stages {
        let scope = stage.label.scope.as_str();
        eprintln!(
            "qwen4exp packed_profile_stage: scope={scope} depth={} name={} layer={:?} mixer={:?} samples={}..{} ticks={} gpu_ms={:.3} fraction={:.6}",
            stage.depth,
            stage.label.name,
            stage.label.layer,
            stage.label.mixer,
            stage.start_sample,
            stage.end_sample,
            stage.duration_ticks,
            stage.gpu_ms,
            stage.fraction_of_gpu,
        );
    }
}

pub(crate) fn emit_qwen4exp_packed_detail_residual(
    profile: &Qwen4ExpPackedPrefillProfile,
    parent: &Qwen4ExpPackedProfileStageTiming,
) {
    let details = profile
        .stages
        .iter()
        .filter(|stage| {
            stage.label.scope == Qwen4ExpPackedProfileScope::Detail
                && stage.depth == parent.depth + 1
                && stage.label.layer == parent.label.layer
                && stage.label.mixer == parent.label.mixer
                && stage.start_sample >= parent.start_sample
                && stage.end_sample <= parent.end_sample
        })
        .try_fold((0_u64, 0.0_f64), |(ticks, gpu_ms), stage| {
            ticks
                .checked_add(stage.duration_ticks)
                .map(|ticks| (ticks, gpu_ms + stage.gpu_ms))
        });
    match details {
        Some((0, _)) => {}
        Some((detail_ticks, detail_gpu_ms)) => {
            match parent.duration_ticks.checked_sub(detail_ticks) {
                Some(residual_ticks) => {
                    let residual_gpu_ms = parent.gpu_ms - detail_gpu_ms;
                    if parent.label.scope == Qwen4ExpPackedProfileScope::Coarse {
                        eprintln!(
                            "qwen4exp packed_profile_detail_residual: layer={:?} mixer={:?} coarse_ticks={} detail_ticks={detail_ticks} residual_ticks={residual_ticks} coarse_gpu_ms={:.3} detail_gpu_ms={detail_gpu_ms:.3} residual_gpu_ms={residual_gpu_ms:.3}",
                            parent.label.layer,
                            parent.label.mixer,
                            parent.duration_ticks,
                            parent.gpu_ms,
                        );
                    } else if parent.label.name == "block.moe" {
                        eprintln!(
                            "qwen4exp packed_profile_moe_residual: layer={:?} mixer={:?} parent_ticks={} detail_ticks={detail_ticks} residual_ticks={residual_ticks} parent_gpu_ms={:.3} detail_gpu_ms={detail_gpu_ms:.3} residual_gpu_ms={residual_gpu_ms:.3}",
                            parent.label.layer,
                            parent.label.mixer,
                            parent.duration_ticks,
                            parent.gpu_ms,
                        );
                    } else if parent.label.name == "moe.routing" {
                        eprintln!(
                            "qwen4exp packed_profile_routing_residual: layer={:?} mixer={:?} parent_ticks={} detail_ticks={detail_ticks} residual_ticks={residual_ticks} parent_gpu_ms={:.3} detail_gpu_ms={detail_gpu_ms:.3} residual_gpu_ms={residual_gpu_ms:.3}",
                            parent.label.layer,
                            parent.label.mixer,
                            parent.duration_ticks,
                            parent.gpu_ms,
                        );
                    } else {
                        eprintln!(
                            "qwen4exp packed_profile_warning: unsupported residual parent={} layer={:?} mixer={:?}",
                            parent.label.name, parent.label.layer, parent.label.mixer,
                        );
                    }
                }
                None => eprintln!(
                    "qwen4exp packed_profile_warning: detail ticks exceed parent ticks for parent={} layer={:?} mixer={:?}",
                    parent.label.name, parent.label.layer, parent.label.mixer,
                ),
            }
        }
        None => eprintln!(
            "qwen4exp packed_profile_warning: detail tick sum overflowed for parent={} layer={:?} mixer={:?}",
            parent.label.name, parent.label.layer, parent.label.mixer,
        ),
    }
}

pub(crate) fn qwen4exp_logits_bitwise_equal(left: &[f32], right: &[f32]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

pub(crate) fn qwen4exp_prefill_execution_mode(
    packed_profile_enabled: bool,
    layer_profile_enabled: bool,
    packed_fallback: bool,
    packed_tokens: usize,
    scalar_tail_commands: usize,
    contains_selection: bool,
) -> &'static str {
    if packed_profile_enabled {
        "packed_profile"
    } else if layer_profile_enabled {
        "scalar_profiled"
    } else if contains_selection {
        "packed_contains_selection"
    } else if packed_tokens != 0 && scalar_tail_commands != 0 {
        "packed_then_scalar"
    } else if packed_tokens != 0 {
        "packed_dense"
    } else if packed_fallback {
        "scalar_fallback"
    } else {
        "scalar"
    }
}

pub(crate) fn validate_qwen4exp_full_shard_prefetch_scope(
    enabled: bool,
    shard_count: usize,
) -> Result<()> {
    if !enabled {
        return Ok(());
    }
    ensure!(
        shard_count == 3,
        "{QWEN4EXP_FULL_SHARD_PREFETCH_ENV} requires the released three-shard asset, got {shard_count} shards"
    );
    Ok(())
}

pub(crate) fn validate_qwen4exp_full_shard_prefetch_report(
    expected_shards: usize,
    reported_shards: usize,
    prefetched_shards: usize,
    skipped_shards: usize,
    mapped_bytes: u64,
    bytes_returned: u64,
) -> Result<()> {
    ensure!(
        reported_shards == expected_shards,
        "Flash-Next full-shard prefetch reported {reported_shards} shards, expected {expected_shards}"
    );
    ensure!(
        prefetched_shards == expected_shards && skipped_shards == 0,
        "Flash-Next full-shard prefetch completed {prefetched_shards} of {expected_shards} shards and skipped {skipped_shards}"
    );
    ensure!(
        bytes_returned == mapped_bytes,
        "Flash-Next full-shard prefetch returned {bytes_returned} bytes for {mapped_bytes} mapped bytes"
    );
    Ok(())
}

pub(crate) fn run_qwen4exp_single_turn(
    model_path: &Path,
    gguf: &GgufFile,
    args: &Args,
    explicit: ExplicitCliOptions,
) -> Result<()> {
    validate_qwen4exp_generation_mode(args, explicit)?;
    let request_t0 = Instant::now();
    let sampling = cli_sampling_config(args)?;
    let (prompt, prompt_source, _) = prompt_text(args)?;
    let tokenizer_t0 = Instant::now();
    let tokenizer = Tokenizer::from_gguf(gguf).context("load Qwen3.8-Flash-Next tokenizer")?;
    let encode_t0 = Instant::now();
    let prompt_ids = tokenizer
        .encode(&prompt, prompt_add_special_tokens(args, prompt_source))
        .context("tokenize Qwen3.8-Flash-Next prompt")?;
    let encode_ms = encode_t0.elapsed().as_secs_f64() * 1e3;
    let tokenizer_ms = tokenizer_t0.elapsed().as_secs_f64() * 1e3;
    let required_forwards =
        required_forwards("Qwen3.8-Flash-Next", prompt_ids.len(), args.tokens, None)?;
    let config =
        Qwen4ExpConfig::from_gguf(gguf).context("bind Qwen3.8-Flash-Next request geometry")?;
    ensure!(
        config == Qwen4ExpConfig::flash_next_reference(),
        "Qwen3.8-Flash-Next runtime requires the released architecture contract"
    );
    ensure!(
        tokenizer.n_vocab() == config.vocab_size,
        "Qwen3.8-Flash-Next tokenizer vocabulary {} differs from model vocabulary {}",
        tokenizer.n_vocab(),
        config.vocab_size
    );
    let logical_forward_limit = args.max_context_tokens.unwrap_or(required_forwards);
    ensure!(
        logical_forward_limit >= required_forwards,
        "Qwen3.8-Flash-Next request requires {required_forwards} forwards, beyond --max-context-tokens {logical_forward_limit}"
    );
    let capacity = Qwen4ExpSessionCapacity::for_forward_limit(&config, logical_forward_limit)
        .context("derive Qwen3.8-Flash-Next request-shaped session capacity")?;
    let vocab_size = tokenizer.n_vocab();
    let prompt_tokens = prompt_ids
        .iter()
        .enumerate()
        .map(|(index, &token)| checked_token_id(token, vocab_size, &format!("prompt[{index}]")))
        .collect::<Result<Vec<_>>>()?;
    let stop_tokens = gguf
        .stop_token_ids()
        .context("load producer-declared Qwen3.8-Flash-Next stop tokens")?;
    validate_qwen4exp_stop_tokens(&stop_tokens, vocab_size)?;
    let decode_options = qwen_llm::qwen4exp_runtime::Qwen4ExpDecodeOptions {
        guarded_topk: parse_qwen4exp_decode_flag(
            std::env::var_os(QWEN4EXP_GUARDED_TOPK_ENV).as_deref(),
            QWEN4EXP_GUARDED_TOPK_ENV,
        )?,
        hc_up_mix: parse_qwen4exp_decode_flag(
            std::env::var_os(QWEN4EXP_HC_UP_MIX_ENV).as_deref(),
            QWEN4EXP_HC_UP_MIX_ENV,
        )?,
        split_qsa: parse_qwen4exp_split_decode(
            std::env::var_os(QWEN4EXP_QSA_SPLIT_DECODE_ENV).as_deref(),
        )?,
    };
    let layer_profile_enabled = qwen_llm::env_flag::read_default_off(QWEN4EXP_LAYER_PROFILE_ENV);
    let packed_profile_enabled =
        qwen_llm::env_flag::read_default_off(QWEN4EXP_PACKED_PREFILL_PROFILE_ENV);
    let full_shard_prefetch_enabled =
        qwen_llm::env_flag::read_default_off(QWEN4EXP_FULL_SHARD_PREFETCH_ENV);
    ensure!(
        !layer_profile_enabled || !packed_profile_enabled,
        "{QWEN4EXP_LAYER_PROFILE_ENV} and {QWEN4EXP_PACKED_PREFILL_PROFILE_ENV} are mutually exclusive"
    );
    ensure!(
        !packed_profile_enabled || prompt_tokens.len() >= 2,
        "{QWEN4EXP_PACKED_PREFILL_PROFILE_ENV} requires at least two prompt tokens"
    );
    validate_qwen4exp_full_shard_prefetch_scope(full_shard_prefetch_enabled, gguf.shard_count())?;
    let packed_prefill_requested =
        !layer_profile_enabled && (packed_profile_enabled || prompt_tokens.len() > 1);
    let prefill_request = if packed_profile_enabled {
        "packed_profile"
    } else if packed_prefill_requested {
        "packed"
    } else if layer_profile_enabled {
        "scalar_profiled"
    } else {
        "scalar"
    };

    eprintln!(
        "qwen4exp: loading {} for text generation; prompt_tokens={} max_generated_tokens={} forward_limit={} qsa_physical_capacity={} prefill_request={prefill_request}",
        model_path.display(),
        prompt_tokens.len(),
        args.tokens,
        capacity.forward_limit(),
        capacity.qsa_physical_capacity(),
    );
    if full_shard_prefetch_enabled {
        // This explicit whole-file operation includes the 28.8 GB CPU PLE
        // region; it is neither an automatic policy nor a range-prefetch seam.
        let feature_t0 = Instant::now();
        let process_before = PidSnapshot::now().ok();
        let prefetch_config = LoadedModelConfig {
            prefetch_policy: PrefetchPolicy::Always,
            ..LoadedModelConfig::default()
        };
        let report = prefetch_opened_gguf(gguf, &prefetch_config);
        let process_delta = process_before
            .zip(PidSnapshot::now().ok())
            .map(|(before, after)| PidDelta::between(before, after));
        let mapped_bytes = u64::try_from(gguf.total_mapped_len())
            .context("Flash-Next mapped byte count exceeds u64")?;
        let bytes_returned = report.bytes_returned_total();
        let feature_wall_ms = feature_t0.elapsed().as_secs_f64() * 1e3;
        for (index, shard) in report.shards.iter().enumerate() {
            eprintln!(
                "qwen4exp full_shard_prefetch_shard: schema=1 index={index} path={:?} bytes_returned={} wall_ms={:.3} skipped={} reason={:?}",
                shard.path,
                shard.bytes_returned,
                shard.wall.as_secs_f64() * 1e3,
                shard.skipped,
                shard.skipped_reason.as_deref(),
            );
        }
        eprintln!(
            "qwen4exp full_shard_prefetch: schema=1 policy=always shards_reported={} shards_prefetched={} shards_skipped={} mapped_bytes={mapped_bytes} bytes_returned={bytes_returned} report_wall_ms={:.3} feature_wall_ms={feature_wall_ms:.3} physical_read_bytes={}",
            report.shards.len(),
            report.shards_prefetched(),
            report.shards_skipped(),
            report.total_wall.as_secs_f64() * 1e3,
            process_delta
                .map(|delta| delta.diskio_bytesread.to_string())
                .unwrap_or_else(|| "unavailable".to_string()),
        );
        validate_qwen4exp_full_shard_prefetch_report(
            gguf.shard_count(),
            report.shards.len(),
            report.shards_prefetched(),
            report.shards_skipped(),
            mapped_bytes,
            bytes_returned,
        )?;
    }
    let load_t0 = Instant::now();
    let ctx = MetalContext::new().context("initialize Metal for Qwen3.8-Flash-Next")?;
    let (mut loaded, packed_fallback) = if packed_prefill_requested {
        match Qwen4ExpLoadedModel::load_with_decode_options(
            &ctx,
            gguf,
            capacity,
            Some(prompt_tokens.len()),
            decode_options,
        ) {
            Ok(loaded) => (loaded, false),
            Err(packed_error) => {
                if packed_profile_enabled {
                    return Err(anyhow!(
                        "load Qwen3.8-Flash-Next packed profile session failed ({packed_error})"
                    ));
                }
                eprintln!(
                    "qwen4exp: packed prefill unavailable ({packed_error}); retrying scalar admission"
                );
                match Qwen4ExpLoadedModel::load_with_decode_options(
                    &ctx,
                    gguf,
                    capacity,
                    None,
                    decode_options,
                ) {
                    Ok(loaded) => (loaded, true),
                    Err(scalar_error) => {
                        return Err(anyhow!(
                            "load Qwen3.8-Flash-Next packed session failed ({packed_error}); scalar fallback also failed ({scalar_error})"
                        ));
                    }
                }
            }
        }
    } else {
        (
            Qwen4ExpLoadedModel::load_with_decode_options(
                &ctx,
                gguf,
                capacity,
                None,
                decode_options,
            )
            .context("load admitted Qwen3.8-Flash-Next weights and text session")?,
            false,
        )
    };
    let load_ms = load_t0.elapsed().as_secs_f64() * 1e3;
    eprintln!(
        "qwen4exp: qsa_split_decode={} eligible_ids=2048..2051 rollback={QWEN4EXP_QSA_SPLIT_DECODE_ENV}=0",
        loaded.split_decode_enabled()
    );
    eprintln!(
        "qwen4exp: hc_up_mix={} eligible=Q8_0/4x2560/K320 rollback={QWEN4EXP_HC_UP_MIX_ENV}=0",
        loaded.hc_up_mix_enabled()
    );
    eprintln!(
        "qwen4exp: guarded_topk={} eligible=N512/K10 rollback={QWEN4EXP_GUARDED_TOPK_ENV}=0",
        loaded.guarded_topk_enabled()
    );
    let admission = loaded.admission();
    let packed_prefill_capacity = loaded.packed_prefill_capacity();
    let packed_selected_requested = loaded.packed_selected_requested();
    let packed_selected_capable = loaded.packed_selected_capable();
    let packed_selected_active = loaded.packed_selected_active();
    if !packed_selected_requested {
        eprintln!(
            "qwen4exp: selected-range packed QSA disabled via {QWEN4EXP_PACKED_SELECTED_QSA_ENV}=0; selected rows past the dense shoulder will scalarize"
        );
    }
    eprintln!(
        "qwen4exp: resident on {} in {:.1} ms; packed_prefill_capacity={packed_prefill_capacity:?} packed_selected_requested={packed_selected_requested} packed_selected_capable={packed_selected_capable} packed_selected_active={packed_selected_active} aggregate_required={:?} weight_observed={} session_observed={} session_required={:?}",
        ctx.describe(),
        load_ms,
        admission.aggregate.required_bytes,
        loaded.observed_weight_bytes(),
        loaded.observed_session_bytes(),
        admission.session.required_bytes,
    );
    let mut runner = loaded
        .create_runner(&ctx)
        .context("bind Qwen3.8-Flash-Next execution graph")?;

    let prefill_t0 = Instant::now();
    let mut measured_prefill_ms = None;
    let mut prefill_timing = Qwen4ExpTimingTotals::default();
    let mut prefill_packed_tokens = 0;
    let mut prefill_scalar_tail_commands = prompt_tokens.len();
    let mut prefill_contains_selection = false;
    let logits = if layer_profile_enabled {
        let mut logits = None;
        for (index, &token) in prompt_tokens.iter().enumerate() {
            shutdown::checkpoint()?;
            let (next_logits, timing) = if index + 1 == prompt_tokens.len() {
                let outcome = runner
                    .forward_token_layer_profiled(token)
                    .with_context(|| {
                        format!("profile Qwen3.8-Flash-Next prompt token {index} by layer")
                    })?;
                let next_logits = runner.logits()?.to_vec();
                let timing = outcome.token;
                match outcome.profile {
                    Ok(profile) => emit_qwen4exp_layer_profile(&profile),
                    Err(error) => eprintln!("qwen4exp layer_profile_warning: {error}"),
                }
                (next_logits, timing)
            } else {
                let next_logits = runner
                    .forward_token(token)
                    .with_context(|| format!("forward Qwen3.8-Flash-Next prompt token {index}"))?
                    .to_vec();
                let timing = runner
                    .last_token_timing()
                    .expect("successful Flash-Next forward records timing");
                (next_logits, timing)
            };
            prefill_timing.record(timing);
            logits = Some(next_logits);
        }
        logits.expect("nonempty prompt produced endpoint logits")
    } else if packed_profile_enabled {
        let packed_capacity =
            packed_prefill_capacity.expect("packed profile mode has an admitted packed workspace");
        ensure!(
            prompt_tokens.len() <= packed_capacity,
            "{QWEN4EXP_PACKED_PREFILL_PROFILE_ENV} requires the complete prompt to fit packed capacity {packed_capacity}; got {} tokens",
            prompt_tokens.len()
        );
        let packet_t0 = Instant::now();

        shutdown::checkpoint()?;
        let first_completed = runner
            .prefill(&prompt_tokens)
            .context("run first unprofiled Flash-Next packed prefill")?;
        let first_copy_t0 = Instant::now();
        let first_logits = first_completed.to_vec();
        let first_copy_ms = first_copy_t0.elapsed().as_secs_f64() * 1e3;
        let first_timing = runner
            .last_prefill_timing()
            .expect("successful first packed prefill records timing");
        let reset_t0 = Instant::now();
        runner
            .reset()
            .context("reset Flash-Next after first packed profile pass")?;
        let first_reset_ms = reset_t0.elapsed().as_secs_f64() * 1e3;

        shutdown::checkpoint()?;
        let warm_completed = runner
            .prefill(&prompt_tokens)
            .context("run warm unprofiled Flash-Next packed prefill")?;
        let warm_copy_t0 = Instant::now();
        let warm_logits = warm_completed.to_vec();
        let warm_copy_ms = warm_copy_t0.elapsed().as_secs_f64() * 1e3;
        let warm_timing = runner
            .last_prefill_timing()
            .expect("successful warm packed prefill records timing");
        let reset_t0 = Instant::now();
        runner
            .reset()
            .context("reset Flash-Next before profiled packed pass")?;
        let warm_reset_ms = reset_t0.elapsed().as_secs_f64() * 1e3;

        shutdown::checkpoint()?;
        let profiled_t0 = Instant::now();
        let outcome = runner
            .prefill_packed_profiled(&prompt_tokens)
            .context("profile Flash-Next packed prefill")?;
        let profiled_copy_t0 = Instant::now();
        let logits = runner.logits()?.to_vec();
        let profiled_copy_ms = profiled_copy_t0.elapsed().as_secs_f64() * 1e3;
        let profiled_ms = profiled_t0.elapsed().as_secs_f64() * 1e3;
        ensure!(
            qwen4exp_logits_bitwise_equal(&first_logits, &warm_logits)
                && qwen4exp_logits_bitwise_equal(&warm_logits, &logits),
            "Flash-Next packed profile passes produced different endpoint logits"
        );
        let timing = runner
            .last_prefill_timing()
            .expect("successful profiled packed prefill records timing");
        debug_assert_eq!(timing.token_count, prompt_tokens.len());
        prefill_packed_tokens = timing.packed_token_count;
        prefill_scalar_tail_commands = timing.token_count - timing.packed_token_count;
        prefill_contains_selection = timing.contains_selection;
        prefill_timing.record_prefill(timing);
        let packet_ms = packet_t0.elapsed().as_secs_f64() * 1e3;
        emit_qwen4exp_packed_profile(
            &outcome,
            first_timing,
            warm_timing,
            first_copy_ms,
            warm_copy_ms,
            profiled_copy_ms,
            first_reset_ms,
            warm_reset_ms,
            packet_ms,
        );
        match &outcome.profile {
            Ok(profile) => emit_qwen4exp_packed_timestamp_profile(profile),
            Err(error) => eprintln!("qwen4exp packed_profile_warning: {error}"),
        }
        measured_prefill_ms = Some(profiled_ms);
        logits
    } else {
        let logits = runner
            .prefill_with_command_checkpoint(&prompt_tokens, || {
                shutdown::checkpoint()
                    .map_err(|error| Qwen4ExpRuntimeError::Checkpoint(error.to_string()))
            })
            .context("prefill Qwen3.8-Flash-Next prompt")?
            .to_vec();
        let timing = runner
            .last_prefill_timing()
            .expect("successful Flash-Next prefill records timing");
        debug_assert_eq!(timing.token_count, prompt_tokens.len());
        prefill_packed_tokens = timing.packed_token_count;
        prefill_scalar_tail_commands = timing
            .token_count
            .checked_sub(timing.packed_token_count)
            .expect("packed prefill token count cannot exceed the prompt");
        prefill_contains_selection = timing.contains_selection;
        prefill_timing.record_prefill(timing);
        logits
    };
    let prefill_mode = qwen4exp_prefill_execution_mode(
        packed_profile_enabled,
        layer_profile_enabled,
        packed_fallback,
        prefill_packed_tokens,
        prefill_scalar_tail_commands,
        prefill_contains_selection,
    );
    let prefill_ms =
        measured_prefill_ms.unwrap_or_else(|| prefill_t0.elapsed().as_secs_f64() * 1e3);
    let mut sampler = Sampler::new(sampling).context("initialize Qwen3.8-Flash-Next sampler")?;
    let stdout_handle = std::io::stdout();
    let mut stdout = stdout_handle.lock();
    let mut decode_timing = Qwen4ExpTimingTotals::default();
    let generation = generate_serial(
        logits,
        args.tokens,
        &stop_tokens,
        &mut sampler,
        |token| {
            let piece = tokenizer
                .try_decode_piece_bytes_exact(token)
                .with_context(|| format!("decode Qwen3.8-Flash-Next token {token}"))?;
            stdout
                .write_all(piece)
                .with_context(|| format!("write Qwen3.8-Flash-Next token {token}"))?;
            stdout.flush().context("flush Qwen3.8-Flash-Next token")?;
            Ok(())
        },
        |token| {
            let token = checked_token_id(token, vocab_size, "generated")?;
            let next_logits = runner
                .forward_token(token)
                .context("forward generated Qwen3.8-Flash-Next token")?
                .to_vec();
            decode_timing.record(
                runner
                    .last_token_timing()
                    .expect("successful Flash-Next forward records timing"),
            );
            Ok(next_logits)
        },
    )?;
    if !generation.tokens.is_empty() {
        writeln!(stdout)?;
        stdout
            .flush()
            .context("flush Qwen3.8-Flash-Next final newline")?;
    }
    let prefill_tps = if prefill_ms > 0.0 {
        prompt_tokens.len() as f64 / (prefill_ms / 1e3)
    } else {
        0.0
    };
    let decode_tps = if generation.wall_ms > 0.0 {
        generation.tokens.len() as f64 / (generation.wall_ms / 1e3)
    } else {
        0.0
    };
    eprintln!(
        "qwen4exp stats: prompt_tokens={} generated_tokens={} transitions={} stop_reason={} tokenizer_ms={:.1} load_ms={:.1} prefill_mode={} prefill_ms={:.1} prefill_tps={:.2} prefill_commands={} prefill_packed_tokens={} prefill_scalar_tail_commands={} prefill_encode_cpu_ms={:.1} prefill_completion_wait_ms={:.1} prefill_gpu_ms={:?} prefill_gpu_samples={}/{} prefill_outside_gpu_ms={:?} generation_ms={:.1} decode_tps={:.2} decode_encode_cpu_ms={:.1} decode_completion_wait_ms={:.1} decode_gpu_ms={:?} decode_gpu_samples={}/{} decode_outside_gpu_ms={:?} total_ms={:.1}",
        prompt_tokens.len(),
        generation.tokens.len(),
        generation.transitions,
        generation.stop_reason.as_str(),
        tokenizer_ms,
        load_ms,
        prefill_mode,
        prefill_ms,
        prefill_tps,
        prefill_timing.forwards,
        prefill_packed_tokens,
        prefill_scalar_tail_commands,
        prefill_timing.encode_cpu_ms,
        prefill_timing.completion_wait_ms,
        prefill_timing.complete_gpu_ms(),
        prefill_timing.gpu_samples,
        prefill_timing.forwards,
        prefill_timing.outside_gpu_ms(),
        generation.wall_ms,
        decode_tps,
        decode_timing.encode_cpu_ms,
        decode_timing.completion_wait_ms,
        decode_timing.complete_gpu_ms(),
        decode_timing.gpu_samples,
        decode_timing.forwards,
        decode_timing.outside_gpu_ms(),
        request_t0.elapsed().as_secs_f64() * 1e3,
    );
    if let Some(path) = args.request_stats_jsonl.as_ref() {
        let transition_tps = if generation.transition_ms > 0.0 {
            generation.transitions as f64 / (generation.transition_ms / 1e3)
        } else {
            0.0
        };
        let measured = RequestStatsMeasured {
            input_tokens: prompt_tokens.len() as u64,
            output_tokens: generation.tokens.len() as u64,
            transitions: generation.transitions as u64,
            stop_reason: generation.stop_reason,
            // Record semantics: encode-only tokenization; total without load.
            tokenizer_ms: encode_ms,
            load_ms,
            prefill_ms,
            prefill_tps,
            decode_ms: generation.wall_ms,
            decode_tps,
            transition_tps,
            total_ms: request_t0.elapsed().as_secs_f64() * 1e3 - load_ms,
            output_fingerprint: GeneratedTokenSha256Digest::of(&generation.tokens),
        };
        append_single_turn_stats_record(
            path,
            0,
            ModelFamily::Qwen4Exp.record_label(),
            request_stats_input(prompt_source, Some("qwen38")),
            &measured,
            None,
        )?;
    }
    Ok(())
}
