use super::*;

fn messages() -> Vec<ChatMessage> {
    parse_strict_ordinary_chat_input(r#"[{"role":"user","content":"first"},{"role":"assistant","content":"old answer"},{"role":"user","content":"next"}]"#, "prefill fixture").unwrap()
}

#[test]
fn channel_mode_matrix_preserves_history_and_exact_prefix_bytes() {
    for protocol in [QwenPromptTemplate::Qwen36, QwenPromptTemplate::Qwen38] {
        for mode in [
            None,
            Some(LensMessageMode::Auto),
            Some(LensMessageMode::Thinking),
            Some(LensMessageMode::NoThinking),
            Some(LensMessageMode::Low),
            Some(LensMessageMode::Medium),
            Some(LensMessageMode::High),
            Some(LensMessageMode::Xhigh),
        ] {
            let base = render_qwen_structured_messages(&messages(), protocol, mode);
            for channel in [
                AssistantPrefillChannel::Reasoning,
                AssistantPrefillChannel::Final,
            ] {
                for text in [
                    "",
                    " ",
                    "\n\n",
                    "  Answer:\n",
                    "a < b; <section>prose</section>",
                    "\u{1f642}",
                ] {
                    let prefill = AssistantPrefill {
                        channel,
                        text: text.into(),
                    };
                    let rendered = render_generation(&messages(), protocol, mode, &prefill);
                    if base.is_err()
                        || (mode == Some(LensMessageMode::NoThinking)
                            && channel == AssistantPrefillChannel::Reasoning)
                    {
                        assert!(rendered.is_err());
                        continue;
                    }
                    let (rendered, _, state) = rendered.unwrap();
                    let base = &base.as_ref().unwrap().0;
                    let transition = if channel == AssistantPrefillChannel::Final
                        && mode != Some(LensMessageMode::NoThinking)
                    {
                        "\n</think>\n\n"
                    } else {
                        ""
                    };
                    assert_eq!(rendered.text, format!("{}{transition}{text}", base.text));
                    assert_eq!(&rendered.spans[..base.spans.len()], base.spans);
                    assert_eq!(
                        state,
                        if channel == AssistantPrefillChannel::Reasoning {
                            QwenGeneration::PreOpen
                        } else {
                            QwenGeneration::PreClosed
                        }
                    );
                    let pieces = rendered.text.as_bytes().chunks(1).collect::<Vec<_>>();
                    let spans = map_rendered_message_spans(&rendered, &pieces).unwrap();
                    assert!(
                        spans
                            .iter()
                            .all(|span| valid_lens_span_metadata(protocol.renderer_name(), span))
                    );
                    let content = spans
                        .iter()
                        .filter(|span| span.kind == "assistant_prefill_content")
                        .collect::<Vec<_>>();
                    assert_eq!(content.len(), usize::from(!text.is_empty()));
                    for span in content {
                        assert_eq!(
                            rendered.text.get(span.byte_start..span.byte_end),
                            Some(text)
                        );
                        assert!(span.message_index.is_none());
                        assert_eq!(span.token_start, Some(span.byte_start));
                    }
                }
            }
        }
    }
}

#[test]
fn structural_markers_unknown_fields_and_unsupported_protocols_fail_closed() {
    for marker in [
        "<|im_start|>",
        "<|im_end|>",
        "<|endoftext|>",
        "<think>",
        "</think>",
        "<tool_call>",
        "</tool_call>",
        "<tool_response>",
        "</tool_response>",
    ] {
        let prefill = AssistantPrefill {
            channel: AssistantPrefillChannel::Final,
            text: format!("quoted `{marker}`"),
        };
        assert!(
            render_generation(&messages(), QwenPromptTemplate::Qwen36, None, &prefill).is_err()
        );
    }
    for protocol in [
        QwenPromptTemplate::Qwen35,
        QwenPromptTemplate::Qwen4Next,
        QwenPromptTemplate::UnknownChatMl,
    ] {
        assert!(
            render_generation(
                &messages(),
                protocol,
                None,
                &AssistantPrefill {
                    channel: AssistantPrefillChannel::Final,
                    text: String::new()
                }
            )
            .is_err()
        );
    }
    for value in [
        r#"{"channel":"thinking","text":""}"#,
        r#"{"channel":"final","text":"","unknown":1}"#,
        r#"{"channel":"final"}"#,
    ] {
        assert!(serde_json::from_str::<AssistantPrefill>(value).is_err());
    }
}

#[test]
fn exact_input_validation_rejects_inconsistent_records_and_missing_context() {
    let (input, record) = byte_token_prefill_fixture(&AssistantPrefill {
        channel: AssistantPrefillChannel::Final,
        text: "  Answer:\n".into(),
    });
    let validate = |record: Option<&GenerationInputRecord>, version, runtime, source, specials| {
        validate_generation_input(
            record,
            version,
            runtime,
            source,
            specials,
            &input.token_ids,
            Some(&input.rendering),
        )
    };
    validate(Some(&record), 5, "ordinary_qwen", "messages", Some(false)).unwrap();
    assert!(validate(None, 5, "ordinary_qwen", "messages", Some(false)).is_err());
    for (version, runtime, source, specials) in [
        (4, "ordinary_qwen", "messages", Some(false)),
        (5, "flash_next", "messages", Some(false)),
        (5, "ordinary_qwen", "prompt", Some(false)),
        (5, "ordinary_qwen", "messages", Some(true)),
    ] {
        assert!(validate(Some(&record), version, runtime, source, specials).is_err());
    }
    for key in [
        "kind",
        "prompt_text",
        "prompt_digest",
        "template",
        "output_initial_state",
    ] {
        let mut value = serde_json::to_value(&record).unwrap();
        value[key] = "changed".into();
        let changed = serde_json::from_value(value).unwrap();
        assert!(
            validate(Some(&changed), 5, "ordinary_qwen", "messages", Some(false)).is_err(),
            "{key}"
        );
    }
    let mut changed = record.clone();
    let closing = changed.prompt_text.rfind("</think>").unwrap();
    changed
        .prompt_text
        .replace_range(closing..closing + "</think>".len(), "<xthink>");
    changed.prompt_bytes = changed.prompt_text.as_bytes().to_vec();
    assert!(validate(Some(&changed), 5, "ordinary_qwen", "messages", Some(false)).is_err());
    let mut changed = record.clone();
    changed.token_ids[0] += 1;
    assert!(validate(Some(&changed), 5, "ordinary_qwen", "messages", Some(false)).is_err());
    let mut changed = record.clone();
    changed.rendering.spans.last_mut().unwrap().channel = Some("thinking".into());
    assert!(validate_record(&changed, &changed.token_ids, &changed.rendering).is_err());
}

#[test]
fn empty_final_prefill_retains_its_transition_without_a_content_span() {
    let (input, record) = byte_token_prefill_fixture(&AssistantPrefill {
        channel: AssistantPrefillChannel::Final,
        text: String::new(),
    });
    assert!(
        !input
            .rendering
            .spans
            .iter()
            .any(|span| span.kind == "assistant_prefill_content")
    );
    assert!(record.prompt_text.ends_with("<think>\n\n</think>\n\n"));
    validate_record(&record, &input.token_ids, &input.rendering).unwrap();
    assert!(
        validate_generation_input(
            None,
            5,
            "ordinary_qwen",
            "messages",
            Some(false),
            &input.token_ids,
            Some(&input.rendering)
        )
        .is_err()
    );
}

#[test]
fn prefill_token_boundaries_remain_nullable_without_invented_alignment() {
    let prefill = AssistantPrefill {
        channel: AssistantPrefillChannel::Final,
        text: "Answer".into(),
    };
    let (rendered, mode, state) =
        render_generation(&messages(), QwenPromptTemplate::Qwen36, None, &prefill).unwrap();
    let boundary = rendered.spans.last().unwrap().byte_start;
    let bytes = rendered.text.as_bytes();
    let pieces = bytes[..boundary - 1]
        .chunks(1)
        .chain(std::iter::once(&bytes[boundary - 1..boundary + 1]))
        .chain(bytes[boundary + 1..].chunks(1))
        .collect::<Vec<_>>();
    let input = PreparedLensInput {
        source: "messages",
        add_special_tokens: Some(false),
        token_ids: (0..pieces.len() as i32).collect(),
        rendering: LensInputRendering {
            renderer: QwenPromptTemplate::Qwen36.renderer_name().into(),
            generation_mode: Some(mode.artifact_name().into()),
            spans: map_rendered_message_spans(&rendered, &pieces).unwrap(),
        },
    };
    let record = record(&input, rendered.text, prefill, state);
    let last = record.rendering.spans.last().unwrap();
    assert_eq!((last.token_start, last.token_end), (None, None));
    validate_record(&record, &input.token_ids, &input.rendering).unwrap();
    for (start, end) in [
        (Some(0), None),
        (None, Some(1)),
        (Some(0), Some(1)),
        (Some(1), Some(usize::MAX)),
    ] {
        let mut changed = record.clone();
        let last = changed.rendering.spans.last_mut().unwrap();
        last.token_start = start;
        last.token_end = end;
        assert!(validate_record(&changed, &changed.token_ids, &changed.rendering).is_err());
    }
    let mut changed = record.clone();
    changed.rendering.spans[0].token_start = None;
    changed.rendering.spans[0].token_end = None;
    assert!(validate_record(&changed, &changed.token_ids, &changed.rendering).is_err());
}

#[test]
fn historical_reasoning_content_requires_assistant_attribution() {
    let (input, _) = byte_token_generation_fixture(QwenPromptTemplate::Qwen36, None, None);
    let span = input
        .rendering
        .spans
        .iter()
        .find(|span| span.kind == "message_content" && span.channel.as_deref() == Some("thinking"))
        .unwrap();
    assert!(span.message_index.is_some());
    assert!(valid_lens_span_metadata(&input.rendering.renderer, span));
    for role in [None, Some("system"), Some("user"), Some("tool")] {
        let mut changed = span.clone();
        changed.role = role.map(str::to_owned);
        assert!(!valid_lens_span_metadata(
            &input.rendering.renderer,
            &changed
        ));
    }
    let mut changed = span.clone();
    changed.message_index = None;
    assert!(!valid_lens_span_metadata(
        &input.rendering.renderer,
        &changed
    ));
}
