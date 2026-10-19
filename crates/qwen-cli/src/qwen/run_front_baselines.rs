use super::*;
use serde_json::{Value, json};
use std::path::Path;

const INFO_BASELINES: &str = include_str!("../../tests/fixtures/run-front-info-baselines-v1.json");
const FRONT_BASELINES: &str = include_str!("../../tests/fixtures/run-front-contracts-v1.json");
const CHAT_BASELINES: &str = include_str!("../../tests/fixtures/run-front-chat-baselines-v1.json");
const K2_STATS_BASELINE: &str = include_str!("../../tests/fixtures/run-baselines/k2-real-v1.json");
const RAW_PROMPT: &str = "  exact raw bytes\nsecond line  ";

fn family_info_baselines() -> Value {
    let mut families = serde_json::Map::new();
    for family in ModelFamily::ALL {
        let path = std::env::temp_dir().join(format!(
            "qwen-run-front-{}-{}.gguf",
            std::process::id(),
            family.architecture_name()
        ));
        crate::linear_transport::cpu_fixture::write_cpu_gguf(
            &path,
            family.architecture_name(),
            2,
            "run-front-baseline",
            false,
        );
        let gguf = GgufFile::open(&path).unwrap();
        let mut projection = project_info(Path::new("<model>"), &gguf).unwrap();
        projection["model"] = json!("<model>");
        if !projection["weight_storage"].is_null() {
            projection["weight_storage"]["mode"] = json!("<device-derived>");
            projection["weight_storage"]["copied_bytes_estimate"] = json!("<device-derived>");
            projection["weight_storage"]["recommended_working_set_bytes"] =
                json!("<device-derived>");
            projection["weight_storage"]["reason"] = json!("<device-derived>");
        }
        families.insert(
            family.architecture_name().to_owned(),
            json!({
                "architecture": family.architecture_name(),
                "json": projection,
                "text_capability_lines": capability_lines(&projection),
            }),
        );
        std::fs::remove_file(path).unwrap();
    }
    Value::Object(families)
}

static V41_ADMISSION: admission::LaneAdmission = admission::LaneAdmission {
    envelope: "DeepSeek V4.1 supports serial single-turn generation without prefix caching",
    supported: &[
        admission::LegacyOption::MaxContextTokens,
        admission::LegacyOption::RequestStatsJsonl,
    ],
};

fn admission_for(family: ModelFamily) -> &'static admission::LaneAdmission {
    match family {
        ModelFamily::Qwen35 | ModelFamily::Qwen35Moe => &admission::ORDINARY_QWEN_SINGLE_TURN,
        ModelFamily::Qwen4Exp => &admission::FLASH_NEXT_SINGLE_TURN,
        ModelFamily::DeepSeek4 => &admission::DEEPSEEK_V4_SINGLE_TURN,
        ModelFamily::DeepSeekV41 => &V41_ADMISSION,
        ModelFamily::MuseGlimmer => &admission::MUSE_GLIMMER_SINGLE_TURN,
        ModelFamily::K2Horizon => &admission::K2_RAW_SINGLE_TURN,
        ModelFamily::Glm5Next => &admission::GLM5_NEXT_SINGLE_TURN,
    }
}

fn family_run_front_baselines() -> Value {
    let stats: Value = serde_json::from_str(K2_STATS_BASELINE).unwrap();
    let mut families = serde_json::Map::new();
    for family in ModelFamily::ALL {
        let path = std::env::temp_dir().join(format!(
            "qwen-run-front-contract-{}-{}.gguf",
            std::process::id(),
            family.architecture_name()
        ));
        crate::linear_transport::cpu_fixture::write_cpu_gguf(
            &path,
            family.architecture_name(),
            2,
            "run-front-baseline",
            false,
        );
        let gguf = GgufFile::open(&path).unwrap();
        let argv = [
            "qwen",
            "run",
            "--model",
            path.to_str().unwrap(),
            "--raw-prompt",
            RAW_PROMPT,
        ];
        let (mut args, _) = Args::parse_with_explicit(argv);
        let invocation = cli::normalize(&mut args);
        invocation.apply_option_overrides(&mut args);
        let (prepared_text, source) = if matches!(
            family,
            ModelFamily::DeepSeekV41
                | ModelFamily::MuseGlimmer
                | ModelFamily::K2Horizon
                | ModelFamily::Glm5Next
        ) {
            // These families enter through their own request adapter before
            // the common Qwen prompt adapter; raw input passes through.
            (RAW_PROMPT.to_owned(), PromptSource::Inline)
        } else {
            let cli::Invocation::Run(run) = invocation else {
                panic!("run command normalized to a non-run invocation");
            };
            let prepared = prepare_modern_run_prompt(run, Some(*family), &gguf, &args).unwrap();
            (prepared.text, prepared.source)
        };
        assert_eq!(prepared_text, RAW_PROMPT, "{}", family.architecture_name());
        assert_eq!(source, PromptSource::Inline);

        let lane = admission_for(*family);
        let bad_options = [
            admission::LegacyOption::Concurrency,
            admission::LegacyOption::BatchSize,
        ];
        let refusal = lane.refuse(&lane.unsupported(&bad_options)).unwrap_err();
        let (raw_bos, chat_bos, template) = match family {
            ModelFamily::Qwen35 | ModelFamily::Qwen35Moe => (true, false, "qwen36"),
            ModelFamily::Qwen4Exp => (true, false, "qwen38"),
            ModelFamily::DeepSeek4 => (false, false, "deepseek_v4"),
            ModelFamily::DeepSeekV41 => (true, false, "deepseek_recipe_dsv41"),
            ModelFamily::MuseGlimmer => (true, false, "atem"),
            ModelFamily::K2Horizon => (true, true, "k2_horizon_ifm_no_tools_v1"),
            ModelFamily::Glm5Next => (true, false, "glm53_flash_chat_v2"),
        };
        families.insert(
            family.architecture_name().to_owned(),
            json!({
                "raw_prompt_utf8": prepared_text,
                "raw_prompt_source": match source { PromptSource::Inline => "inline", PromptSource::File => "file", PromptSource::Messages => "messages" },
                "bos": {"raw":raw_bos,"chat":chat_bos},
                "refusal_precedence": {
                    "supplied": ["--concurrency", "--batch-size"],
                    "message": refusal.to_string(),
                },
                "stats_jsonl_top_level_keys": stats["stats_jsonl_keys"],
                "stats_family_label": family.record_label(),
                "stats_chat_template_label": template,
            }),
        );
        std::fs::remove_file(path).unwrap();
    }
    Value::Object(families)
}

fn family_chat_prompt_baselines() -> Value {
    use qwen_llm::model_family::ModelFamily as F;
    let user = "Add 2 and 3.";
    let system = Some("You are terse.");
    let mut families = serde_json::Map::new();
    for family in ModelFamily::ALL {
        let prompt = match family {
            F::Qwen35 | F::Qwen35Moe => {
                let path = std::env::temp_dir().join(format!(
                    "qwen-run-chat-{}-{}.gguf",
                    std::process::id(),
                    family.architecture_name()
                ));
                crate::linear_transport::cpu_fixture::write_cpu_gguf(
                    &path,
                    family.architecture_name(),
                    2,
                    "qwen-release:Qwen3.6",
                    false,
                );
                let gguf = GgufFile::open(&path).unwrap();
                let protocol = QwenUserPromptProtocol::resolve(*family, &gguf)
                    .unwrap()
                    .unwrap();
                let prompt = protocol
                    .render(user, system, QwenReasoningControls::default())
                    .unwrap();
                std::fs::remove_file(path).unwrap();
                prompt
            }
            F::Qwen4Exp => {
                render_qwen38_single_turn_prompt(user, system, Qwen38GenerationMode::default())
            }
            F::DeepSeek4 => render_deepseek_v4_0731_single_turn_prompt(
                user,
                system,
                DeepSeekV4EncodeOptions::default(),
            )
            .unwrap(),
            F::DeepSeekV41 => {
                let profile = crate::serve::request_profile::RequestProfile::DeepSeekV41 {
                    limits: crate::serve::request_profile::OutputLimits {
                        default_max_tokens: 32,
                        max_piece_bytes: 4096,
                    },
                    sampling: None,
                };
                let body = json!({
                    "model": "v41",
                    "instructions": system.unwrap(),
                    "input": [{"role":"user", "content":user}],
                });
                let mut request = profile.parse(&body).unwrap();
                profile.normalize(&mut request).unwrap();
                profile.render(&request).unwrap()
            }
            F::MuseGlimmer => qwen_llm::muse_glimmer_request::MuseGlimmerRequest::single_turn(
                user,
                system.map(str::to_owned),
            )
            .render(
                qwen_llm::muse_glimmer::MuseGlimmerChatTemplateProfile::UnslothLaunch,
                None,
            )
            .unwrap(),
            F::K2Horizon => qwen_llm::k2_horizon_chat::render(
                &[
                    qwen_llm::k2_horizon_chat::Message::text("system", system.unwrap().to_owned()),
                    qwen_llm::k2_horizon_chat::Message::text("user", user.to_owned()),
                ],
                qwen_llm::k2_horizon_chat::Effort::High,
            )
            .unwrap(),
            F::Glm5Next => qwen_llm::glm5_next_chat::render(
                &[
                    qwen_llm::glm5_next_chat::Message::System(system.unwrap().to_owned()),
                    qwen_llm::glm5_next_chat::Message::User(user.to_owned()),
                ],
                qwen_llm::glm5_next_chat::RenderOptions::generate(
                    qwen_llm::glm5_next_chat::Effort::Max,
                    true,
                ),
            )
            .unwrap(),
        };
        let (add_special_tokens, rendered_bos_marker, bos_owner) = match family {
            F::Qwen35 | F::Qwen35Moe | F::Qwen4Exp => (false, false, "none"),
            F::DeepSeek4 | F::DeepSeekV41 | F::MuseGlimmer => (false, true, "renderer"),
            F::K2Horizon => (true, false, "native_tokenizer"),
            F::Glm5Next => (false, false, "glm_generation_prefix"),
        };
        families.insert(
            family.architecture_name().to_owned(),
            json!({
                "rendered_prompt_utf8":prompt,
                "bos_handling": {
                    "add_special_tokens": add_special_tokens,
                    "rendered_bos_marker": rendered_bos_marker,
                    "owner": bos_owner,
                }
            }),
        );
    }
    Value::Object(families)
}

#[test]
fn every_family_chat_prompt_matches_checked_in_baseline() {
    let expected: Value = serde_json::from_str(CHAT_BASELINES).unwrap();
    assert_eq!(family_chat_prompt_baselines(), expected);
}

#[test]
fn every_family_info_json_and_text_matches_synthetic_baseline() {
    let expected: Value = serde_json::from_str(INFO_BASELINES).unwrap();
    assert_eq!(family_info_baselines(), expected);
}

#[test]
fn every_family_run_front_matches_checked_in_baseline() {
    let expected: Value = serde_json::from_str(FRONT_BASELINES).unwrap();
    assert_eq!(family_run_front_baselines(), expected);
}
