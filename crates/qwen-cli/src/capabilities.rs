//! Header-level run capability decisions shared by the CLI and model info.
//!
//! Family-owned renderers supply only their narrow projection hooks; the
//! family routing and common release decisions live here so callers do not
//! need a callback through the command binary.

use crate::prompt_template::{self, InputCapability, QwenUserPromptProtocol};
use qwen_llm::gguf::GgufFile;
use qwen_llm::model_family::ModelFamily;
use serde_json::{Value, json};

/// Which request forms can be rendered for a family and inspected GGUF.
pub fn input_for(
    family: Option<ModelFamily>,
    gguf: &GgufFile,
    qwen4exp_failure: impl FnOnce(&GgufFile) -> Option<String>,
    glm5_next: impl FnOnce(&GgufFile) -> InputCapability,
) -> InputCapability {
    match family {
        Some(family @ (ModelFamily::Qwen35 | ModelFamily::Qwen35Moe)) => {
            match QwenUserPromptProtocol::resolve(family, gguf) {
                Ok(Some(protocol)) => protocol.input_capability(),
                Ok(None) => unreachable!("ordinary Qwen resolves a protocol"),
                Err(error) => InputCapability::raw_only("template_unresolved", error.to_string()),
            }
        }
        Some(ModelFamily::Qwen4Exp) => match qwen4exp_failure(gguf) {
            None => InputCapability::all_supported(),
            Some(failure) => InputCapability::raw_only(
                "prompt_protocol_unsupported",
                format!(
                    "Qwen3.8-Flash-Next chat rendering does not support the declared {failure}; use --raw-prompt for untemplated input"
                ),
            ),
        },
        Some(
            ModelFamily::DeepSeekV41 | ModelFamily::DeepSeek4 | ModelFamily::MuseGlimmer,
        ) => InputCapability::all_supported(),
        Some(ModelFamily::K2Horizon) => {
            match qwen_llm::k2_horizon_runtime::K2PreparedArtifact::inspect(gguf)
                .and_then(|artifact| artifact.generation_stops())
            {
                Ok(_) => InputCapability::raw_only(
                    "k2_native_chat_renderer_required",
                    "K2 structured chat uses its native marker and template checks; compatible checkpoints retain raw input".into(),
                ),
                Err(error) => InputCapability::none(error.code(), error.to_string()),
            }
        }
        Some(ModelFamily::Glm5Next) => glm5_next(gguf),
        None => InputCapability::none(
            "unknown_family",
            "templated input requires a recognised architecture".into(),
        ),
    }
}

/// `capabilities.template`: renderer identity and release identification.
pub fn template_for(
    family: Option<ModelFamily>,
    gguf: &GgufFile,
    glm5_next: impl FnOnce(&GgufFile) -> Value,
) -> Value {
    match family {
        Some(ModelFamily::Qwen35 | ModelFamily::Qwen35Moe | ModelFamily::Qwen4Exp) => {
            match prompt_template::identify_qwen_release_for_gguf(gguf) {
                Ok(identity) if family == Some(ModelFamily::Qwen4Exp) => {
                    qwen4exp_template_projection(&identity)
                }
                Ok(identity) => {
                    let mut value =
                        serde_json::to_value(&identity.status).expect("serialize release status");
                    value["rendered_as"] = json!(identity.template.renderer_name());
                    value
                }
                Err(error) => json!({"status":"unresolved", "message":error.to_string()}),
            }
        }
        Some(ModelFamily::DeepSeekV41) => {
            json!({"status":"identified", "rendered_as":"deepseek_recipe_dsv41"})
        }
        Some(ModelFamily::DeepSeek4) => json!({
            "status":"identified", "version":"deepseek_v4_0731",
            "source":"general.architecture", "rendered_as":"deepseek_v4_0731"
        }),
        Some(ModelFamily::MuseGlimmer) => json!({
            "status":"identified", "version":"muse_glimmer",
            "source":"general.architecture", "rendered_as":"muse_glimmer"
        }),
        Some(ModelFamily::K2Horizon) => json!({
            "status":"unsupported", "rendered_as":null,
            "message":"K2 Horizon raw input has no template renderer"
        }),
        Some(ModelFamily::Glm5Next) => glm5_next(gguf),
        None => json!({"status":"unresolved", "message":"no recognised architecture"}),
    }
}

pub fn qwen4exp_template_projection(identity: &prompt_template::QwenReleaseIdentity) -> Value {
    let mut value = serde_json::to_value(&identity.status).expect("serialize release status");
    if matches!(
        &identity.status,
        prompt_template::QwenReleaseStatus::Identified { .. }
    ) {
        value["rendered_as"] = json!(identity.template.renderer_name());
    }
    value
}
