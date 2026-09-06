use anyhow::{Context, Result, bail, ensure};
use qwen_llm::gguf::GgufFile;
use qwen_llm::muse_glimmer::{
    ARCHITECTURE_NAME as MUSE_GLIMMER_ARCHITECTURE, MuseGlimmerChatTemplateProfile,
    MuseGlimmerConfig,
};
use sha2::{Digest, Sha256};
use std::fmt::Write;

// Exact tokenizer.chat_template metadata identities. These hash only the
// small template string, never model weights. Qwen3.5 admits the two variants
// observed across its released dense and MoE GGUFs. Qwen3.6 accepts the
// canonical 27B template and the alternate packaged with the 35B A3B GGUF;
// both Qwen3.6 variants implement the same renderer thinking transition.
const QWEN35_CHAT_TEMPLATE_SHA256: [u8; 32] = [
    0x7f, 0x0e, 0x52, 0x90, 0x32, 0xc2, 0x51, 0x83, 0xbc, 0xd6, 0x6c, 0x7f, 0x23, 0x8d, 0xa2, 0xd3,
    0x77, 0xf4, 0x3b, 0xe7, 0x54, 0xa9, 0x4e, 0x27, 0x25, 0xa5, 0x8c, 0x4e, 0x16, 0xd2, 0xed, 0x67,
];
const QWEN35_ALTERNATE_CHAT_TEMPLATE_SHA256: [u8; 32] = [
    0xe6, 0x0d, 0xf4, 0x14, 0x81, 0xb6, 0xad, 0x20, 0x57, 0x1c, 0xda, 0x7e, 0x2e, 0x12, 0x90, 0xcf,
    0xee, 0x25, 0x67, 0x0f, 0x1b, 0x50, 0x97, 0x3f, 0x74, 0x57, 0x6c, 0x93, 0x54, 0xc6, 0x63, 0x36,
];
const QWEN36_CHAT_TEMPLATE_SHA256: [u8; 32] = [
    0xe8, 0x4f, 0x32, 0xa2, 0x3f, 0xdd, 0xa2, 0x76, 0x89, 0xf8, 0x68, 0xaa, 0x4a, 0x1a, 0x56, 0x21,
    0xf4, 0x11, 0x33, 0xe5, 0x1a, 0x48, 0xd7, 0xf3, 0xef, 0xcb, 0xea, 0x28, 0x39, 0x57, 0x42, 0x59,
];
const QWEN36_ALTERNATE_CHAT_TEMPLATE_SHA256: [u8; 32] = [
    0x55, 0xd4, 0x93, 0x14, 0x33, 0xfe, 0x50, 0x2b, 0x79, 0x42, 0x26, 0xee, 0x7f, 0x4d, 0x20, 0x6a,
    0x6b, 0xdd, 0x43, 0x6a, 0xc9, 0xf8, 0x0e, 0xb7, 0xd8, 0xeb, 0xb4, 0xc6, 0x39, 0xf9, 0xea, 0x0c,
];
const QWEN38_CHAT_TEMPLATE_SHA256: [u8; 32] = [
    0x70, 0x1b, 0xa1, 0x3a, 0x08, 0x5c, 0x0c, 0x1b, 0x5e, 0x05, 0x41, 0x4d, 0xec, 0x1a, 0xa3, 0x06,
    0x99, 0x04, 0xf9, 0x62, 0xbe, 0xee, 0x36, 0xf0, 0x89, 0x9e, 0x44, 0x17, 0x20, 0xb8, 0x39, 0x74,
];
const QWEN4NEXT_CHAT_TEMPLATE_SHA256: [u8; 32] = [
    0x12, 0x82, 0x7f, 0x24, 0xb7, 0x42, 0xea, 0x4e, 0x80, 0xcd, 0xc1, 0x2d, 0xbc, 0xf9, 0x62, 0x22,
    0x27, 0x05, 0x6b, 0x9f, 0x79, 0x72, 0x52, 0xa3, 0x14, 0x92, 0x63, 0xd4, 0xf9, 0xaa, 0xad, 0xce,
];
const DEEPSEEK_V4_0731_CHAT_TEMPLATE_SHA256: [u8; 32] = [
    0xe6, 0x43, 0xc3, 0x1f, 0xce, 0xc1, 0x7f, 0x34, 0x2f, 0x72, 0x29, 0x6e, 0x02, 0xc4, 0x6d, 0x35,
    0x84, 0x6b, 0xf4, 0xc7, 0x0f, 0x6a, 0x02, 0x71, 0xf2, 0x3b, 0xad, 0x73, 0xfd, 0x4e, 0xb6, 0x45,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum QwenPromptTemplate {
    Qwen35,
    Qwen36,
    Qwen38,
    Qwen4Next,
    UnverifiedChatMl,
}

impl QwenPromptTemplate {
    pub(crate) const fn renderer_name(self) -> &'static str {
        match self {
            Self::Qwen35 => "qwen3.5_messages_v1",
            Self::Qwen36 => "qwen3.6_messages_v1",
            Self::Qwen38 => "qwen3.8_messages_v1",
            Self::Qwen4Next => "qwen4next_messages_v1",
            Self::UnverifiedChatMl => "qwen_chatml_messages_v1",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ModelPromptTemplate {
    Qwen(QwenPromptTemplate),
    MuseGlimmer(MuseGlimmerChatTemplateProfile),
    DeepSeekV4_0731,
}

pub(crate) fn resolve_model_prompt_template(gguf: &GgufFile) -> Result<ModelPromptTemplate> {
    let architecture = gguf
        .architecture()
        .context("model is missing general.architecture")?;
    match architecture.as_str() {
        "qwen35" | "qwen35moe" | "qwen4exp" => {
            resolve_qwen_prompt_template(&architecture, gguf).map(ModelPromptTemplate::Qwen)
        }
        MUSE_GLIMMER_ARCHITECTURE => {
            let config =
                MuseGlimmerConfig::from_gguf(gguf).context("bind Muse Glimmer prompt template")?;
            Ok(ModelPromptTemplate::MuseGlimmer(
                config.chat_template_profile,
            ))
        }
        "deepseek4" => {
            let digest = chat_template_digest(gguf)?;
            ensure!(
                digest == DEEPSEEK_V4_0731_CHAT_TEMPLATE_SHA256,
                "DeepSeek V4 structured input requires the released 0731 chat template; found SHA-256 {}",
                digest_hex(digest)
            );
            Ok(ModelPromptTemplate::DeepSeekV4_0731)
        }
        other => bail!("unsupported prompt-template architecture {other:?}"),
    }
}

/// Serve/run rendering template for a Qwen-family GGUF. Qwen3.8 keeps its
/// metadata-validated identity gate (`supports_qwen38_prompt_protocol`);
/// among the rest, only digest-pinned Qwen3.5/3.6 templates render with
/// released-exact bytes, and anything unpinned keeps the legacy generic
/// ChatML contract rather than guessing.
pub(crate) fn serve_qwen_template(
    family: qwen_llm::model_family::ModelFamily,
    gguf: &GgufFile,
) -> Result<crate::open_responses::items::QwenTemplate> {
    use crate::open_responses::items::QwenTemplate;
    if crate::messages::supports_qwen38_release_prompt_protocol(family, gguf) {
        return Ok(QwenTemplate::Qwen38);
    }
    Ok(
        match resolve_qwen_prompt_template(family.architecture_name(), gguf)? {
            QwenPromptTemplate::Qwen35 => QwenTemplate::Qwen35,
            QwenPromptTemplate::Qwen36 => QwenTemplate::Qwen36,
            QwenPromptTemplate::Qwen38
            | QwenPromptTemplate::Qwen4Next
            | QwenPromptTemplate::UnverifiedChatMl => QwenTemplate::Generic,
        },
    )
}

/// Template for any loaded GGUF: Qwen families resolve by digest; other
/// families render nothing through the Qwen renderer and get `Generic`.
pub(crate) fn qwen_template_for_gguf(
    gguf: &GgufFile,
) -> Result<crate::open_responses::items::QwenTemplate> {
    match qwen_llm::model_family::ModelFamily::detect(gguf) {
        Some(
            family @ (qwen_llm::model_family::ModelFamily::Qwen35
            | qwen_llm::model_family::ModelFamily::Qwen35Moe
            | qwen_llm::model_family::ModelFamily::Qwen4Exp),
        ) => serve_qwen_template(family, gguf),
        _ => Ok(crate::open_responses::items::QwenTemplate::Generic),
    }
}

fn resolve_qwen_prompt_template(architecture: &str, gguf: &GgufFile) -> Result<QwenPromptTemplate> {
    let tokenizer_model = gguf.get_str("tokenizer.ggml.model");
    let tokenizer_pre = gguf.get_str("tokenizer.ggml.pre");
    if tokenizer_model != Some("gpt2") || tokenizer_pre != Some("qwen35") {
        ensure!(
            architecture != "qwen4exp",
            "Qwen4Next structured input requires tokenizer model/pre gpt2/qwen35"
        );
        return Ok(QwenPromptTemplate::UnverifiedChatMl);
    }

    let digest = chat_template_digest(gguf)?;
    let resolved = if let Some(template) = classify_qwen_digest(architecture, digest) {
        template
    } else {
        if architecture == "qwen4exp" {
            bail!(
                "Qwen4Next structured input requires the released chat template; found SHA-256 {}. Use --prompt/--raw-prompt or --token-ids for exact untemplated input",
                digest_hex(digest)
            );
        }
        if [
            gguf.get_str("general.name"),
            gguf.get_str("general.base_model.0.name"),
        ]
        .into_iter()
        .flatten()
        .any(|name| name.to_ascii_lowercase().contains("qwen3.8"))
        {
            bail!(
                "model declares Qwen3.8 but has an unrecognized chat template SHA-256 {}. Use --prompt/--raw-prompt or --token-ids for exact untemplated input",
                digest_hex(digest)
            );
        }
        QwenPromptTemplate::UnverifiedChatMl
    };
    Ok(resolved)
}

fn classify_qwen_digest(architecture: &str, digest: [u8; 32]) -> Option<QwenPromptTemplate> {
    match (architecture, digest) {
        (
            "qwen35" | "qwen35moe",
            QWEN35_CHAT_TEMPLATE_SHA256 | QWEN35_ALTERNATE_CHAT_TEMPLATE_SHA256,
        ) => Some(QwenPromptTemplate::Qwen35),
        (
            "qwen35" | "qwen35moe",
            QWEN36_CHAT_TEMPLATE_SHA256 | QWEN36_ALTERNATE_CHAT_TEMPLATE_SHA256,
        ) => Some(QwenPromptTemplate::Qwen36),
        ("qwen35", QWEN38_CHAT_TEMPLATE_SHA256) => Some(QwenPromptTemplate::Qwen38),
        ("qwen4exp", QWEN4NEXT_CHAT_TEMPLATE_SHA256) => Some(QwenPromptTemplate::Qwen4Next),
        _ => None,
    }
}

fn chat_template_digest(gguf: &GgufFile) -> Result<[u8; 32]> {
    let template = gguf
        .get_str("tokenizer.chat_template")
        .context("model is missing tokenizer.chat_template")?;
    Ok(Sha256::digest(template.as_bytes()).into())
}

fn digest_hex(digest: [u8; 32]) -> String {
    let mut output = String::with_capacity(64);
    for byte in digest {
        write!(&mut output, "{byte:02x}").expect("write digest to String");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen_release_digests_are_architecture_scoped() {
        assert_eq!(
            classify_qwen_digest("qwen35", QWEN35_CHAT_TEMPLATE_SHA256),
            Some(QwenPromptTemplate::Qwen35)
        );
        assert_eq!(
            classify_qwen_digest("qwen35moe", QWEN35_ALTERNATE_CHAT_TEMPLATE_SHA256),
            Some(QwenPromptTemplate::Qwen35)
        );
        for architecture in ["qwen35", "qwen35moe"] {
            for digest in [
                QWEN36_CHAT_TEMPLATE_SHA256,
                QWEN36_ALTERNATE_CHAT_TEMPLATE_SHA256,
            ] {
                assert_eq!(
                    classify_qwen_digest(architecture, digest),
                    Some(QwenPromptTemplate::Qwen36)
                );
            }
        }
        assert_eq!(
            classify_qwen_digest("qwen35", QWEN38_CHAT_TEMPLATE_SHA256),
            Some(QwenPromptTemplate::Qwen38)
        );
        assert_eq!(
            classify_qwen_digest("qwen4exp", QWEN4NEXT_CHAT_TEMPLATE_SHA256),
            Some(QwenPromptTemplate::Qwen4Next)
        );
        assert_eq!(
            classify_qwen_digest("qwen35moe", QWEN38_CHAT_TEMPLATE_SHA256),
            None
        );
        assert_eq!(
            classify_qwen_digest("qwen35", QWEN4NEXT_CHAT_TEMPLATE_SHA256),
            None
        );
    }

    #[test]
    fn tracked_template_oracles_keep_their_pinned_digests() {
        for (template, expected) in [
            (
                include_str!("../tests/fixtures/templates/qwen36_a3b_chat_template.jinja"),
                QWEN36_ALTERNATE_CHAT_TEMPLATE_SHA256,
            ),
            (
                include_str!("../tests/fixtures/templates/qwen38_27b_chat_template.jinja"),
                QWEN38_CHAT_TEMPLATE_SHA256,
            ),
            (
                include_str!("../tests/fixtures/templates/ds4_flash_chat_template.jinja"),
                DEEPSEEK_V4_0731_CHAT_TEMPLATE_SHA256,
            ),
        ] {
            assert_eq!(
                <[u8; 32]>::from(Sha256::digest(template.as_bytes())),
                expected
            );
        }
    }

    #[test]
    #[ignore = "requires QWEN_PROMPT_GGUF and QWEN_PROMPT_TEMPLATE"]
    fn local_qwen_metadata_resolves_to_expected_template() {
        let path = std::env::var("QWEN_PROMPT_GGUF").expect("set QWEN_PROMPT_GGUF");
        let expected = match std::env::var("QWEN_PROMPT_TEMPLATE")
            .expect("set QWEN_PROMPT_TEMPLATE")
            .as_str()
        {
            "qwen35" => QwenPromptTemplate::Qwen35,
            "qwen36" => QwenPromptTemplate::Qwen36,
            "qwen38" => QwenPromptTemplate::Qwen38,
            "qwen4next" => QwenPromptTemplate::Qwen4Next,
            "unverified" => QwenPromptTemplate::UnverifiedChatMl,
            other => panic!("unknown expected Qwen prompt template {other:?}"),
        };
        let gguf = GgufFile::open(path).expect("open Qwen GGUF");
        let digest = chat_template_digest(&gguf).expect("read chat template digest");
        assert_eq!(
            resolve_model_prompt_template(&gguf).unwrap(),
            ModelPromptTemplate::Qwen(expected),
            "tokenizer model/pre={:?}/{:?}, chat template SHA-256 {}",
            gguf.get_str("tokenizer.ggml.model"),
            gguf.get_str("tokenizer.ggml.pre"),
            digest_hex(digest),
        );
    }
}
