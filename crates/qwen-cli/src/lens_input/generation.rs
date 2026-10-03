//! Typed assistant prefills and their retained exact-input context.

use super::*;
use crate::messages::{MessageRenderChannel, MessageRenderSpan};
use crate::model_request::prefill::{AssistantPrefill, AssistantPrefillChannel, qwen_transition};
use crate::open_responses::render::QwenGeneration;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GenerationInputRecord {
    kind: String,
    token_ids: Vec<i32>,
    rendering: LensInputRendering,
    prompt_text: String,
    prompt_bytes: Vec<u8>,
    prompt_digest: String,
    template: String,
    assistant_prefill: AssistantPrefill,
    output_initial_state: String,
}

fn initial_state(
    protocol: QwenPromptTemplate,
    mode: ResolvedMessageMode,
) -> Result<QwenGeneration> {
    ensure!(
        matches!(
            protocol,
            QwenPromptTemplate::Qwen36 | QwenPromptTemplate::Qwen38
        ),
        "assistant prefill is unsupported for this prompt protocol"
    );
    match mode {
        ResolvedMessageMode::Qwen36(QwenGenerationMode::Auto | QwenGenerationMode::Thinking)
        | ResolvedMessageMode::Qwen38(Qwen38GenerationMode::Thinking(_)) => {
            Ok(QwenGeneration::PreOpen)
        }
        ResolvedMessageMode::Qwen36(QwenGenerationMode::NoThinking)
        | ResolvedMessageMode::Qwen38(Qwen38GenerationMode::NoThinking) => {
            Ok(QwenGeneration::PreClosed)
        }
        _ => bail!("unsupported typed generation mode"),
    }
}

fn state_name(state: QwenGeneration) -> &'static str {
    match state {
        QwenGeneration::Bare => "bare",
        QwenGeneration::PreOpen => "pre_open",
        QwenGeneration::PreClosed => "pre_closed",
    }
}

fn render_generation(
    messages: &[ChatMessage],
    protocol: QwenPromptTemplate,
    mode: Option<LensMessageMode>,
    prefill: &AssistantPrefill,
) -> Result<(AnnotatedMessageRender, ResolvedMessageMode, QwenGeneration)> {
    let (mut rendered, _, mode) = render_qwen_structured_messages(messages, protocol, mode)?;
    let (transition, output_state) =
        qwen_transition(protocol, initial_state(protocol, mode)?, prefill)?;
    let mut append = |text: &str, kind, channel| {
        if text.is_empty() {
            return;
        }
        let byte_start = rendered.text.len();
        rendered.text.push_str(text);
        rendered.spans.push(MessageRenderSpan {
            kind,
            message_index: None,
            role: Some("assistant".into()),
            channel,
            byte_start,
            byte_end: rendered.text.len(),
        });
    };
    for &(text, kind, channel) in transition {
        append(text, kind, channel);
    }
    append(
        &prefill.text,
        MessageRenderSpanKind::AssistantPrefillContent,
        (prefill.channel == AssistantPrefillChannel::Reasoning)
            .then_some(MessageRenderChannel::Thinking),
    );
    Ok((rendered, mode, output_state))
}

pub(crate) fn prepare_qwen_model_generation_input(
    spec: LensInputSpec<'_>,
    prefill: &AssistantPrefill,
    family: ModelFamily,
    gguf: &GgufFile,
    tokenizer: &Tokenizer,
) -> Result<(PreparedLensInput, GenerationInputRecord)> {
    validate_lens_input_spec(spec)?;
    ensure!(
        matches!(family, ModelFamily::Qwen35 | ModelFamily::Qwen35Moe),
        "typed assistant prefill supports ordinary Qwen only"
    );
    ensure!(
        spec.user.is_some() || spec.messages.is_some(),
        "typed generation input requires --user or --messages"
    );
    let messages = acquire_structured_messages(spec.user, spec.system, spec.messages)?;
    let protocol = detect_qwen_message_protocol(family, gguf)?;
    let prepared = prepare_generation(
        &messages,
        protocol,
        spec.message_mode,
        Some(prefill),
        tokenizer,
    )?;
    let record = record(
        &prepared.input,
        prepared.prompt_text,
        prefill.clone(),
        prepared.output_state,
    );
    validate_record(
        &record,
        &prepared.input.token_ids,
        &prepared.input.rendering,
    )?;
    Ok((prepared.input, record))
}

pub(crate) struct PreparedLensGenerationInput {
    pub(crate) input: PreparedLensInput,
    prompt_text: String,
    output_state: QwenGeneration,
}

impl PreparedLensGenerationInput {
    #[allow(dead_code)] // Native publication is in the separate qwen binary.
    pub(crate) fn record<'a>(
        &'a self,
        prefill: Option<&'a AssistantPrefill>,
    ) -> impl Serialize + 'a {
        #[derive(Serialize)]
        struct Record<'a> {
            kind: &'static str,
            token_ids: &'a [i32],
            rendering: &'a LensInputRendering,
            prompt_text: &'a str,
            prompt_bytes: &'a [u8],
            prompt_digest: String,
            template: &'a str,
            assistant_prefill: Option<&'a AssistantPrefill>,
            output_initial_state: &'static str,
        }
        Record {
            kind: "prepared_input",
            token_ids: &self.input.token_ids,
            rendering: &self.input.rendering,
            prompt_text: &self.prompt_text,
            prompt_bytes: self.prompt_text.as_bytes(),
            prompt_digest: qwen_llm::tokenizer::token_ids_sha256_i32le(&self.input.token_ids),
            template: &self.input.rendering.renderer,
            assistant_prefill: prefill,
            output_initial_state: state_name(self.output_state),
        }
    }
}

#[allow(dead_code)] // Native ingress is in the separate qwen binary.
pub(crate) fn prepare_qwen_generation_messages_bytes(
    bytes: &[u8],
    source: &str,
    mode: Option<LensMessageMode>,
    prefill: Option<&AssistantPrefill>,
    protocol: QwenPromptTemplate,
    tokenizer: &Tokenizer,
) -> Result<PreparedLensGenerationInput> {
    let text = std::str::from_utf8(bytes).context("messages must be UTF-8")?;
    let messages = parse_strict_ordinary_chat_input(text, source)?;
    prepare_generation(&messages, protocol, mode, prefill, tokenizer)
}

fn prepare_generation(
    messages: &[ChatMessage],
    protocol: QwenPromptTemplate,
    mode: Option<LensMessageMode>,
    prefill: Option<&AssistantPrefill>,
    tokenizer: &Tokenizer,
) -> Result<PreparedLensGenerationInput> {
    let (rendered, mode, output_state) = if let Some(prefill) = prefill {
        render_generation(messages, protocol, mode, prefill)?
    } else {
        let (rendered, _, mode) = render_qwen_structured_messages(messages, protocol, mode)?;
        (rendered, mode, initial_state(protocol, mode)?)
    };
    ensure!(
        rendered.text.len() <= MAX_OPEN_RESPONSES_BYTES,
        "generation input exceeds retained-text limit"
    );
    // Tokenize the complete prompt once; separately encoded suffixes can change
    // BPE boundaries and must not be appended as if they were exact token IDs.
    let token_ids = tokenizer
        .encode(&rendered.text, false)
        .context("tokenize exact generation prompt")?;
    let spans = align_rendered_message_spans(tokenizer, &rendered, &token_ids)?;
    let input = PreparedLensInput {
        source: "messages",
        add_special_tokens: Some(false),
        token_ids,
        rendering: LensInputRendering {
            renderer: protocol.renderer_name().into(),
            generation_mode: Some(mode.artifact_name().into()),
            spans,
        },
    };
    Ok(PreparedLensGenerationInput {
        input,
        prompt_text: rendered.text,
        output_state,
    })
}

fn record(
    input: &PreparedLensInput,
    prompt_text: String,
    prefill: AssistantPrefill,
    state: QwenGeneration,
) -> GenerationInputRecord {
    GenerationInputRecord {
        kind: "prepared_input".into(),
        token_ids: input.token_ids.clone(),
        rendering: input.rendering.clone(),
        prompt_bytes: prompt_text.as_bytes().to_vec(),
        prompt_text,
        prompt_digest: qwen_llm::tokenizer::token_ids_sha256_i32le(&input.token_ids),
        template: input.rendering.renderer.clone(),
        assistant_prefill: prefill,
        output_initial_state: state_name(state).into(),
    }
}

pub(crate) fn validate_generation_input(
    record: Option<&GenerationInputRecord>,
    schema_version: u32,
    runtime: &str,
    input_source: &str,
    add_special_tokens: Option<bool>,
    token_ids: &[i32],
    rendering: Option<&LensInputRendering>,
) -> Result<()> {
    let Some(record) = record else {
        if let Some(rendering) = rendering {
            ensure!(
                !rendering
                    .spans
                    .iter()
                    .any(|span| span.kind == "assistant_prefill_content"),
                "prefill span requires generation_input"
            );
            if matches!(
                rendering.renderer.as_str(),
                "qwen3.6_messages_v1" | "qwen3.8_messages_v1"
            ) && rendering.generation_mode.as_deref() != Some("no_thinking")
            {
                let tail = rendering
                    .spans
                    .iter()
                    .skip_while(|span| span.kind != "generated_assistant_start_marker");
                ensure!(
                    !tail
                        .into_iter()
                        .any(|span| span.kind == "thinking_channel_end_marker"),
                    "closed thinking generation requires generation_input"
                );
            }
        }
        return Ok(());
    };
    ensure!(
        schema_version == 5
            && runtime == "ordinary_qwen"
            && input_source == "messages"
            && add_special_tokens == Some(false),
        "generation_input requires v5 ordinary Qwen messages without added specials"
    );
    validate_record(
        record,
        token_ids,
        rendering.context("generation_input requires rendering metadata")?,
    )
}

fn retained_mode(record: &GenerationInputRecord) -> Result<(QwenPromptTemplate, LensMessageMode)> {
    let protocol = match record.template.as_str() {
        "qwen3.6_messages_v1" => QwenPromptTemplate::Qwen36,
        "qwen3.8_messages_v1" => QwenPromptTemplate::Qwen38,
        _ => bail!("generation input template is unsupported"),
    };
    let mode = match record.rendering.generation_mode.as_deref() {
        Some("auto") if protocol == QwenPromptTemplate::Qwen36 => LensMessageMode::Auto,
        Some("thinking") if protocol == QwenPromptTemplate::Qwen36 => LensMessageMode::Thinking,
        Some("thinking_low") if protocol == QwenPromptTemplate::Qwen38 => LensMessageMode::Low,
        Some("thinking_medium") if protocol == QwenPromptTemplate::Qwen38 => {
            LensMessageMode::Medium
        }
        Some("thinking_xhigh") if protocol == QwenPromptTemplate::Qwen38 => LensMessageMode::Xhigh,
        Some("no_thinking") => LensMessageMode::NoThinking,
        _ => bail!("generation input mode is unsupported for its template"),
    };
    Ok((protocol, mode))
}

fn validate_record(
    record: &GenerationInputRecord,
    tokens: &[i32],
    rendering: &LensInputRendering,
) -> Result<()> {
    ensure!(
        record.kind == "prepared_input",
        "invalid generation input kind"
    );
    ensure!(
        record.token_ids == tokens && &record.rendering == rendering,
        "generation input disagrees with run tokens/rendering"
    );
    ensure!(
        record.prompt_text.len() <= MAX_OPEN_RESPONSES_BYTES
            && record.prompt_bytes == record.prompt_text.as_bytes(),
        "generation input text/bytes mismatch or limit exceeded"
    );
    ensure!(
        record.prompt_digest == qwen_llm::tokenizer::token_ids_sha256_i32le(tokens),
        "generation input token digest mismatch"
    );
    ensure!(
        record.template == rendering.renderer,
        "generation input template mismatch"
    );
    let mut previous = 0;
    let mut previous_token_end = 0;
    for span in &rendering.spans {
        let valid_tokens = match (span.token_start, span.token_end) {
            (None, None) => !is_structural_lens_span(&span.kind),
            (Some(start), Some(end)) => {
                let valid = start < end && start >= previous_token_end && end <= tokens.len();
                previous_token_end = end;
                valid
            }
            _ => false,
        };
        ensure!(
            span.byte_start == previous
                && span.byte_start < span.byte_end
                && record
                    .prompt_text
                    .get(span.byte_start..span.byte_end)
                    .is_some()
                && valid_tokens
                && valid_lens_span_metadata(&rendering.renderer, span),
            "invalid generation input span coverage/attribution"
        );
        previous = span.byte_end;
    }
    ensure!(
        previous == record.prompt_text.len(),
        "generation input spans do not cover retained text"
    );
    let (protocol, mode) = retained_mode(record)?;
    let (expected, _, state) =
        render_generation(&[], protocol, Some(mode), &record.assistant_prefill)?;
    ensure!(
        record.output_initial_state == state_name(state),
        "generation input parser state mismatch"
    );
    let expected_start = expected
        .spans
        .iter()
        .position(|span| span.kind == MessageRenderSpanKind::GeneratedAssistantStartMarker)
        .context("missing expected generation header")?;
    let starts = rendering
        .spans
        .iter()
        .enumerate()
        .filter(|(_, span)| span.kind == "generated_assistant_start_marker")
        .collect::<Vec<_>>();
    ensure!(
        starts.len() == 1,
        "generation input must have one generation header"
    );
    let (start, header) = starts[0];
    let offset = expected.spans[expected_start].byte_start;
    ensure!(
        record.prompt_text[header.byte_start..] == expected.text[offset..],
        "generation input transition/prefill suffix mismatch"
    );
    let actual = &rendering.spans[start..];
    let expected = &expected.spans[expected_start..];
    ensure!(
        actual.len() == expected.len(),
        "generation input suffix span count mismatch"
    );
    for (actual, expected) in actual.iter().zip(expected) {
        ensure!(
            actual.kind == expected.kind.as_str()
                && actual.message_index.is_none()
                && actual.role.as_deref() == Some("assistant")
                && actual.channel.as_deref() == expected.channel.map(|channel| channel.as_str())
                && actual.byte_start - header.byte_start == expected.byte_start - offset
                && actual.byte_end - header.byte_start == expected.byte_end - offset,
            "generation input suffix attribution mismatch"
        );
    }
    ensure!(
        !rendering.spans[..start]
            .iter()
            .any(|span| span.kind == "assistant_prefill_content"),
        "prefill content appears in history"
    );
    Ok(())
}

#[cfg(test)]
pub(crate) fn byte_token_prefill_fixture(
    prefill: &AssistantPrefill,
) -> (PreparedLensInput, GenerationInputRecord) {
    let (input, record) = byte_token_generation_fixture(
        QwenPromptTemplate::Qwen36,
        Some(LensMessageMode::Thinking),
        Some(prefill),
    );
    (input, record.unwrap())
}

#[cfg(test)]
pub(crate) fn byte_token_generation_fixture(
    protocol: QwenPromptTemplate,
    mode: Option<LensMessageMode>,
    prefill: Option<&AssistantPrefill>,
) -> (PreparedLensInput, Option<GenerationInputRecord>) {
    let messages = parse_strict_ordinary_chat_input(
        r#"[{"role":"user","content":"first"},{"role":"assistant","reasoning_content":"old reasoning","content":"old answer"},{"role":"user","content":"next"}]"#,
        "fixture",
    ).unwrap();
    let (rendered, mode, state) = if let Some(prefill) = prefill {
        render_generation(&messages, protocol, mode, prefill).unwrap()
    } else {
        let (rendered, _, mode) =
            render_qwen_structured_messages(&messages, protocol, mode).unwrap();
        (rendered, mode, initial_state(protocol, mode).unwrap())
    };
    // These token IDs stand for byte positions, not a qualified model tokenizer.
    let pieces = rendered.text.as_bytes().chunks(1).collect::<Vec<_>>();
    let input = PreparedLensInput {
        source: "messages",
        add_special_tokens: Some(false),
        token_ids: (0..rendered.text.len() as i32).collect(),
        rendering: LensInputRendering {
            renderer: protocol.renderer_name().into(),
            generation_mode: Some(mode.artifact_name().into()),
            spans: map_rendered_message_spans(&rendered, &pieces).unwrap(),
        },
    };
    let record = prefill.map(|prefill| record(&input, rendered.text, prefill.clone(), state));
    (input, record)
}

#[cfg(test)]
mod tests;
