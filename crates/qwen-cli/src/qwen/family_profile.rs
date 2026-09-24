use anyhow::Result;
use qwen_llm::gguf::GgufFile;
use qwen_llm::model_family::ModelFamily;
use serde_json::{Value, json};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DrafterSupport {
    Dense,
    MoeCliSerial,
    Unsupported(&'static str),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FixedCohort {
    None,
    Dense8,
    Moe16,
}

/// How `qwen serve` keeps a family's prefixes warm between requests.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ServeWarmth {
    /// RAM snapshot cache (`--snapshot-cache-mib`) plus the durable disk tier
    /// (`--durable-snapshot-*`).
    SnapshotsDurable,
    /// RAM snapshot cache only; nothing survives a restart.
    SnapshotsRam,
    /// Reuse of the resident session's live prefix; no snapshots at all.
    LiveSession,
}

pub(crate) struct FamilyProfile {
    pub(crate) family: ModelFamily,
    pub(crate) display: &'static str,
    pub(crate) drafter: DrafterSupport,
    pub(crate) fixed_cohort: FixedCohort,
    /// A `GenerationBackend` exists; not artifact or device admission.
    pub(crate) serve_backend: bool,
    pub(crate) serve_warmth: ServeWarmth,
    pub(crate) capabilities: fn(&GgufFile) -> Result<Value>,
}

static QWEN35: FamilyProfile = FamilyProfile {
    family: ModelFamily::Qwen35,
    display: "Qwen",
    drafter: DrafterSupport::Dense,
    fixed_cohort: FixedCohort::Dense8,
    serve_backend: true,
    serve_warmth: ServeWarmth::SnapshotsDurable,
    capabilities: qwen35_capabilities,
};

static QWEN35_MOE: FamilyProfile = FamilyProfile {
    family: ModelFamily::Qwen35Moe,
    display: "Qwen MoE",
    drafter: DrafterSupport::MoeCliSerial,
    fixed_cohort: FixedCohort::Moe16,
    serve_backend: true,
    serve_warmth: ServeWarmth::SnapshotsDurable,
    capabilities: qwen35_moe_capabilities,
};

static QWEN4EXP: FamilyProfile = FamilyProfile {
    family: ModelFamily::Qwen4Exp,
    display: "Qwen3.8-Flash-Next",
    drafter: DrafterSupport::Unsupported("family_no_speculation"),
    fixed_cohort: FixedCohort::None,
    serve_backend: true,
    serve_warmth: ServeWarmth::SnapshotsRam,
    capabilities: qwen4exp_capabilities,
};

static DEEPSEEK4: FamilyProfile = FamilyProfile {
    family: ModelFamily::DeepSeek4,
    display: "DeepSeek V4",
    drafter: DrafterSupport::Unsupported("family_no_speculation"),
    fixed_cohort: FixedCohort::None,
    serve_backend: true,
    serve_warmth: ServeWarmth::SnapshotsDurable,
    capabilities: deepseek4_capabilities,
};

static MUSE_GLIMMER: FamilyProfile = FamilyProfile {
    family: ModelFamily::MuseGlimmer,
    display: "Muse Glimmer",
    drafter: DrafterSupport::Unsupported("family_no_speculation"),
    fixed_cohort: FixedCohort::None,
    serve_backend: true,
    serve_warmth: ServeWarmth::LiveSession,
    capabilities: muse_glimmer_capabilities,
};

static K2_HORIZON: FamilyProfile = FamilyProfile {
    family: ModelFamily::K2Horizon,
    display: "K2 Horizon",
    drafter: DrafterSupport::Unsupported("family_no_speculation"),
    fixed_cohort: FixedCohort::None,
    serve_backend: true,
    serve_warmth: ServeWarmth::LiveSession,
    capabilities: k2_capabilities,
};

pub(crate) fn profile(family: ModelFamily) -> &'static FamilyProfile {
    let profile = match family {
        ModelFamily::Qwen35 => &QWEN35,
        ModelFamily::Qwen35Moe => &QWEN35_MOE,
        ModelFamily::Qwen4Exp => &QWEN4EXP,
        ModelFamily::DeepSeek4 => &DEEPSEEK4,
        ModelFamily::MuseGlimmer => &MUSE_GLIMMER,
        ModelFamily::K2Horizon => &K2_HORIZON,
    };
    debug_assert_eq!(profile.family, family);
    profile
}

fn assemble(family: ModelFamily, gguf: &GgufFile, reasoning: Value) -> Result<Value> {
    Ok(json!({
        "reasoning": reasoning,
        "input": serde_json::to_value(crate::input_capability_for(Some(family), gguf))?,
        "template": crate::template_projection(Some(family), gguf),
    }))
}

fn qwen_capabilities(family: ModelFamily, gguf: &GgufFile) -> Result<Value> {
    let reasoning = match crate::QwenUserPromptProtocol::resolve(family, gguf) {
        Ok(Some(protocol)) => serde_json::to_value(protocol.reasoning_capability())?,
        Ok(None) => unreachable!("ordinary Qwen resolves a protocol"),
        Err(error) => json!({
            "status": "unsupported",
            "code": "template_unresolved",
            "message": error.to_string(),
        }),
    };
    assemble(family, gguf, reasoning)
}

fn qwen35_capabilities(gguf: &GgufFile) -> Result<Value> {
    qwen_capabilities(ModelFamily::Qwen35, gguf)
}

fn qwen35_moe_capabilities(gguf: &GgufFile) -> Result<Value> {
    qwen_capabilities(ModelFamily::Qwen35Moe, gguf)
}

fn qwen4exp_capabilities(gguf: &GgufFile) -> Result<Value> {
    let reasoning = if crate::supports_qwen38_prompt_protocol(ModelFamily::Qwen4Exp, gguf) {
        serde_json::to_value(crate::prompt_template::ReasoningCapability {
            levels: crate::Qwen38GenerationMode::level_names(),
            fallback: Some("xhigh"),
            no_thinking: crate::prompt_template::Support::Supported,
            thinking: crate::prompt_template::Support::Supported,
        })?
    } else {
        let failure = crate::qwen4exp_prompt_capability_failure(ModelFamily::Qwen4Exp, gguf)
            .expect("unsupported Flash-Next prompt has a capability failure");
        json!({
            "status": "unsupported",
            "code": "prompt_protocol_unsupported",
            "message": format!("Qwen3.8-Flash-Next chat rendering does not support the declared {}", failure.as_str()),
        })
    };
    assemble(ModelFamily::Qwen4Exp, gguf, reasoning)
}

fn deepseek4_capabilities(gguf: &GgufFile) -> Result<Value> {
    let config = match qwen_llm::deepseek_v4::DeepSeekV4Config::from_gguf(gguf)
        .and_then(|config| config.validate_flash_0731_profile().map(|()| config))
    {
        Ok(config) => config,
        Err(error) => {
            return unverified_profile_capabilities(
                "deepseek4_release_profile_unverified",
                error.to_string(),
            );
        }
    };
    if let Err(error) = crate::deepseek_v4_generation_stops(gguf, config.vocab_size) {
        return unverified_profile_capabilities(
            "deepseek4_release_profile_unverified",
            error.to_string(),
        );
    }
    let reasoning = serde_json::to_value(crate::prompt_template::ReasoningCapability {
        levels: crate::DeepSeekV4Reasoning::level_names(),
        fallback: Some("none"),
        no_thinking: crate::prompt_template::Support::Supported,
        thinking: crate::prompt_template::Support::Unsupported {
            code: "thinking_unsupported",
            message: "DeepSeek V4 selects thinking through reasoning effort; there is no explicit thinking toggle".into(),
        },
    })?;
    assemble(ModelFamily::DeepSeek4, gguf, reasoning)
}

fn muse_glimmer_capabilities(gguf: &GgufFile) -> Result<Value> {
    let config = match qwen_llm::muse_glimmer::MuseGlimmerConfig::from_gguf(gguf) {
        Ok(config) => config,
        Err(error) => {
            return unverified_profile_capabilities(
                "muse_release_profile_unverified",
                error.to_string(),
            );
        }
    };
    let stop_tokens = match gguf.stop_token_ids() {
        Ok(stop_tokens) => stop_tokens,
        Err(error) => {
            return unverified_profile_capabilities(
                "muse_release_profile_unverified",
                error.to_string(),
            );
        }
    };
    if let Err(error) = config.validate_stop_tokens(&stop_tokens) {
        return unverified_profile_capabilities(
            "muse_release_profile_unverified",
            error.to_string(),
        );
    }
    assemble(
        ModelFamily::MuseGlimmer,
        gguf,
        serde_json::to_value(crate::muse_glimmer_reasoning_capability())?,
    )
}

fn unverified_profile_capabilities(code: &'static str, reason: String) -> Result<Value> {
    let message = format!("release profile is not verified: {reason}");
    let unsupported = crate::prompt_template::Support::Unsupported {
        code,
        message: message.clone(),
    };
    Ok(json!({
        "reasoning": crate::prompt_template::ReasoningCapability {
            levels: Vec::<&'static str>::new(),
            fallback: None,
            no_thinking: unsupported.clone(),
            thinking: unsupported,
        },
        "input": crate::prompt_template::InputCapability::none(code, message),
        "template": {
            "status": "unknown",
            "reason": reason,
        },
        "execution": {
            "status": "rejected",
            "code": code,
            "message": reason,
        },
    }))
}

fn k2_capabilities(gguf: &GgufFile) -> Result<Value> {
    let capabilities = crate::k2_horizon::capability_projection(gguf)?;
    Ok(json!({
        "reasoning": capabilities["reasoning"].clone(),
        "input": capabilities["input"].clone(),
        "template": capabilities["template"].clone(),
        "execution": capabilities["execution"].clone(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn every_family_has_a_round_tripping_unique_profile() {
        let mut displays = BTreeSet::new();
        for family in ModelFamily::ALL {
            let profile = profile(*family);
            assert_eq!(profile.family, *family);
            assert!(displays.insert(profile.display), "{}", profile.display);
        }
    }

    #[test]
    fn unverified_profile_projection_is_rejected_and_explicit() {
        let capabilities = unverified_profile_capabilities(
            "muse_release_profile_unverified",
            "invalid metadata key \"muse-glimmer.block_count\": expected 52, got 1".into(),
        )
        .unwrap();
        assert_eq!(capabilities["template"]["status"], "unknown");
        assert!(capabilities["template"].get("fields_consulted").is_none());
        for form in ["raw", "user", "messages", "tools"] {
            assert_eq!(capabilities["input"][form]["status"], "unsupported");
            assert_eq!(
                capabilities["input"][form]["code"],
                "muse_release_profile_unverified"
            );
        }
        assert_eq!(
            capabilities["reasoning"]["thinking"]["code"],
            "muse_release_profile_unverified"
        );
        assert_eq!(capabilities["execution"]["status"], "rejected");
        assert_eq!(
            capabilities["execution"]["code"],
            "muse_release_profile_unverified"
        );
        assert_eq!(
            capabilities["execution"]["message"],
            "invalid metadata key \"muse-glimmer.block_count\": expected 52, got 1"
        );
    }
}
