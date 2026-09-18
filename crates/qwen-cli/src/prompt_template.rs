use crate::messages::{CapabilityError, Qwen38GenerationMode, QwenGenerationMode};
use crate::open_responses::items::QwenTemplate;
use anyhow::{Context, Result, bail};
use qwen_llm::gguf::GgufFile;
use qwen_llm::model_family::ModelFamily;
use qwen_llm::muse_glimmer::{
    ARCHITECTURE_NAME as MUSE_GLIMMER_ARCHITECTURE, MuseGlimmerChatTemplateProfile,
    MuseGlimmerConfig,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum QwenPromptTemplate {
    Qwen35,
    Qwen36,
    Qwen38,
    Qwen4Next,
    /// No identified release; the legacy ChatML contract.
    UnknownChatMl,
}

impl QwenPromptTemplate {
    pub(crate) const fn renderer_name(self) -> &'static str {
        match self {
            Self::Qwen35 => "qwen3.5_messages_v1",
            Self::Qwen36 => "qwen3.6_messages_v1",
            Self::Qwen38 => "qwen3.8_messages_v1",
            Self::Qwen4Next => "qwen4next_messages_v1",
            Self::UnknownChatMl => "qwen_chatml_messages_v1",
        }
    }

    pub(crate) fn serve_template(self) -> QwenTemplate {
        match self {
            Self::Qwen35 => QwenTemplate::Qwen35,
            Self::Qwen36 => QwenTemplate::Qwen36,
            Self::Qwen38 | Self::Qwen4Next => QwenTemplate::Qwen38,
            Self::UnknownChatMl => QwenTemplate::Generic,
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
        "qwen35" | "qwen35moe" | "qwen4exp" => Ok(ModelPromptTemplate::Qwen(
            identify_qwen_release(&QwenHeaderFacts::from_gguf(&architecture, gguf)?).template,
        )),
        MUSE_GLIMMER_ARCHITECTURE => {
            let config =
                MuseGlimmerConfig::from_gguf(gguf).context("bind Muse Glimmer prompt template")?;
            Ok(ModelPromptTemplate::MuseGlimmer(
                config.chat_template_profile,
            ))
        }
        // The renderer is pinned to the vLLM/SGLang 0731 encoders, not to
        // any GGUF template; repacks embed three different templates.
        "deepseek4" => Ok(ModelPromptTemplate::DeepSeekV4_0731),
        other => bail!("unsupported prompt-template architecture {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Qwen release identity. The renderer contract follows the release version
// (3.5 / 3.6 / 3.8); the version is carried only by the header's name
// fields (Qwen ships identical architectures across releases at 27B and
// 35B-A3B, and every other header key is either constant or a converter
// artefact). `tokenizer.chat_template` is never consulted.
// ---------------------------------------------------------------------------

/// The tokenizer every supported Qwen3.x release ships; a deviation means
/// the eos/special-token assumptions the renderer relies on do not hold.
const QWEN3_TOKENIZER_MODEL: &str = "gpt2";
const QWEN3_TOKENIZER_PRE: &str = "qwen35";
const QWEN3_TOKEN_COUNT: usize = 248_320;

pub(crate) const QWEN_NAME_FIELDS: [&str; 5] = [
    "general.name",
    "general.basename",
    "general.base_model.0.name",
    "general.base_model.0.repo_url",
    "general.license.link",
];

/// Header facts consulted for release identity, in the order they are
/// reported.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct QwenHeaderFacts<'a> {
    pub(crate) architecture: &'a str,
    pub(crate) tokenizer_model: Option<&'a str>,
    pub(crate) tokenizer_pre: Option<&'a str>,
    pub(crate) token_count: Option<usize>,
    /// Parallel to `QWEN_NAME_FIELDS`.
    pub(crate) names: [Option<&'a str>; 5],
}

impl<'a> QwenHeaderFacts<'a> {
    pub(crate) fn from_gguf(architecture: &'a str, gguf: &'a GgufFile) -> Result<Self> {
        Ok(Self {
            architecture,
            tokenizer_model: gguf.get_str("tokenizer.ggml.model"),
            tokenizer_pre: gguf.get_str("tokenizer.ggml.pre"),
            token_count: gguf
                .get_array_len("tokenizer.ggml.tokens")
                .context("read tokenizer.ggml.tokens")?,
            names: QWEN_NAME_FIELDS.map(|key| gguf.get_str(key)),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub(crate) enum QwenReleaseStatus {
    Identified {
        version: &'static str,
        source: &'static str,
    },
    Unknown {
        reason: String,
        fields_consulted: Vec<&'static str>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct QwenReleaseIdentity {
    pub(crate) template: QwenPromptTemplate,
    pub(crate) status: QwenReleaseStatus,
}

impl QwenReleaseIdentity {
    pub(crate) fn warning(&self) -> Option<String> {
        match &self.status {
            QwenReleaseStatus::Identified { .. } => None,
            QwenReleaseStatus::Unknown { reason, .. } => Some(format!(
                "Qwen release not identified ({reason}); rendering the {} contract, which matches no released template; thinking controls and tools are unavailable (`qwen info --json` reports capabilities.template)",
                self.template.serve_template().label()
            )),
        }
    }
}

/// A tokenizer that is not the Qwen3.x release tokenizer, by header key.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum QwenTokenizerMismatch {
    Model,
    Pretokenizer,
    TokenCount,
}

impl QwenTokenizerMismatch {
    pub(crate) fn header_key(self) -> &'static str {
        match self {
            Self::Model => "tokenizer.ggml.model",
            Self::Pretokenizer => "tokenizer.ggml.pre",
            Self::TokenCount => "tokenizer.ggml.tokens",
        }
    }
}

pub(crate) fn qwen_tokenizer_gate(
    facts: &QwenHeaderFacts<'_>,
) -> Result<(), QwenTokenizerMismatch> {
    if facts.tokenizer_model != Some(QWEN3_TOKENIZER_MODEL) {
        return Err(QwenTokenizerMismatch::Model);
    }
    if facts.tokenizer_pre != Some(QWEN3_TOKENIZER_PRE) {
        return Err(QwenTokenizerMismatch::Pretokenizer);
    }
    if facts.token_count != Some(QWEN3_TOKEN_COUNT) {
        return Err(QwenTokenizerMismatch::TokenCount);
    }
    Ok(())
}

/// `qwen`, optionally one of ` -_`, then `3.` and exactly one of 5/6/8 not
/// followed by another digit. Case-insensitive.
fn qwen_versions_in(text: &str) -> Vec<&'static str> {
    let lower = text.to_ascii_lowercase();
    let bytes = lower.as_bytes();
    let mut found = Vec::new();
    let mut at = 0;
    while let Some(offset) = lower[at..].find("qwen") {
        let mut i = at + offset + 4;
        if matches!(bytes.get(i), Some(b' ' | b'-' | b'_')) {
            i += 1;
        }
        if bytes.get(i..i + 2) == Some(b"3.") {
            let minor = bytes.get(i + 2).copied();
            let next_is_digit = bytes.get(i + 3).is_some_and(u8::is_ascii_digit);
            let version = match (minor, next_is_digit) {
                (Some(b'5'), false) => Some("qwen3.5"),
                (Some(b'6'), false) => Some("qwen3.6"),
                (Some(b'8'), false) => Some("qwen3.8"),
                _ => None,
            };
            if let Some(version) = version
                && !found.contains(&version)
            {
                found.push(version);
            }
        }
        at = at + offset + 4;
    }
    found
}

pub(crate) fn identify_qwen_release(facts: &QwenHeaderFacts<'_>) -> QwenReleaseIdentity {
    let consulted = || {
        let mut fields = vec![
            "general.architecture",
            "tokenizer.ggml.model",
            "tokenizer.ggml.pre",
            "tokenizer.ggml.tokens",
        ];
        fields.extend(QWEN_NAME_FIELDS);
        fields
    };
    let unknown = |reason: String| QwenReleaseIdentity {
        template: QwenPromptTemplate::UnknownChatMl,
        status: QwenReleaseStatus::Unknown {
            reason,
            fields_consulted: consulted(),
        },
    };
    if let Err(mismatch) = qwen_tokenizer_gate(facts) {
        return unknown(format!(
            "{} is not the Qwen3.x release tokenizer",
            mismatch.header_key()
        ));
    }
    // Flash-Next is the only `qwen4exp` product; the architecture is the identity.
    if facts.architecture == "qwen4exp" {
        return QwenReleaseIdentity {
            template: QwenPromptTemplate::Qwen4Next,
            status: QwenReleaseStatus::Identified {
                version: "qwen3.8",
                source: "general.architecture",
            },
        };
    }
    let mut versions: Vec<(&'static str, &'static str)> = Vec::new();
    for (field, value) in QWEN_NAME_FIELDS.iter().zip(facts.names) {
        for version in value.map(qwen_versions_in).unwrap_or_default() {
            if !versions.iter().any(|(_, seen)| *seen == version) {
                versions.push((field, version));
            }
        }
    }
    match versions.as_slice() {
        [] => unknown("no Qwen release version in any name field".into()),
        [(source, version)] => QwenReleaseIdentity {
            template: match *version {
                "qwen3.5" => QwenPromptTemplate::Qwen35,
                "qwen3.6" => QwenPromptTemplate::Qwen36,
                _ => QwenPromptTemplate::Qwen38,
            },
            status: QwenReleaseStatus::Identified { version, source },
        },
        many => unknown(format!(
            "conflicting Qwen release versions: {}",
            many.iter()
                .map(|(field, version)| format!("{field}={version}"))
                .collect::<Vec<_>>()
                .join(", ")
        )),
    }
}

pub(crate) fn identify_qwen_release_for_gguf(gguf: &GgufFile) -> Result<QwenReleaseIdentity> {
    let architecture = gguf
        .architecture()
        .context("model is missing general.architecture")?;
    Ok(identify_qwen_release(&QwenHeaderFacts::from_gguf(
        &architecture,
        gguf,
    )?))
}

/// Serve/run rendering template plus identity status for a Qwen-family GGUF.
pub(crate) fn resolve_serve_qwen_template(
    _family: qwen_llm::model_family::ModelFamily,
    gguf: &GgufFile,
) -> Result<QwenReleaseIdentity> {
    identify_qwen_release_for_gguf(gguf)
}

pub(crate) fn serve_qwen_template(
    family: qwen_llm::model_family::ModelFamily,
    gguf: &GgufFile,
) -> Result<QwenTemplate> {
    if family == ModelFamily::K2Horizon {
        bail!("K2 Horizon has no Qwen prompt or output protocol");
    }
    Ok(resolve_serve_qwen_template(family, gguf)?
        .template
        .serve_template())
}

/// Template for any loaded GGUF; non-Qwen families get `Generic`.
pub(crate) fn qwen_template_for_gguf(gguf: &GgufFile) -> Result<QwenTemplate> {
    match qwen_llm::model_family::ModelFamily::detect(gguf) {
        Some(
            family @ (qwen_llm::model_family::ModelFamily::Qwen35
            | qwen_llm::model_family::ModelFamily::Qwen35Moe
            | qwen_llm::model_family::ModelFamily::Qwen4Exp),
        ) => serve_qwen_template(family, gguf),
        Some(ModelFamily::K2Horizon) => bail!("K2 Horizon has no Qwen prompt or output protocol"),
        _ => Ok(QwenTemplate::Generic),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    fn facts<'a>(architecture: &'a str, names: [Option<&'a str>; 5]) -> QwenHeaderFacts<'a> {
        QwenHeaderFacts {
            architecture,
            tokenizer_model: Some("gpt2"),
            tokenizer_pre: Some("qwen35"),
            token_count: Some(248_320),
            names,
        }
    }

    fn template(architecture: &str, names: [Option<&str>; 5]) -> QwenPromptTemplate {
        identify_qwen_release(&facts(architecture, names)).template
    }

    /// Rows from the local header inventory (2026-09-17): same shapes across
    /// releases, version only in names.
    #[test]
    fn release_identity_follows_name_fields_not_shape_or_template() {
        let released = [
            (
                "qwen35",
                ["Qwen3.5-27B", "Qwen3.5-27B", "Qwen3.5 27B"],
                QwenPromptTemplate::Qwen35,
            ),
            (
                "qwen35",
                ["Qwen3.6-27B", "Qwen3.6-27B", "Qwen3.6 27B"],
                QwenPromptTemplate::Qwen36,
            ),
            (
                "qwen35",
                ["Qwen3.8-27B", "Qwen3.8-27B", "Qwen3.8 27B"],
                QwenPromptTemplate::Qwen38,
            ),
            (
                "qwen35moe",
                ["Qwen3.5-35B-A3B", "Qwen3.5-35B-A3B", "Qwen3.5 35B A3B"],
                QwenPromptTemplate::Qwen35,
            ),
            (
                "qwen35moe",
                ["Qwen3.6-35B-A3B", "Qwen3.6-35B-A3B", "Qwen3.6 35B A3B"],
                QwenPromptTemplate::Qwen36,
            ),
            (
                "qwen35",
                ["Qwen3.5 0.8B", "Qwen3.5", "Qwen3.5 0.8B Base"],
                QwenPromptTemplate::Qwen35,
            ),
        ];
        for (architecture, [name, basename, base], expected) in released {
            assert_eq!(
                template(
                    architecture,
                    [Some(name), Some(basename), Some(base), None, None]
                ),
                expected,
                "{name}"
            );
        }
        // Derivatives and repacks that the digest gate refused or misfiled.
        assert_eq!(
            template(
                "qwen35",
                [Some("Qwen3.8 27B Bf16"), Some("Qwen3.8"), None, None, None]
            ),
            QwenPromptTemplate::Qwen38,
            "ridge"
        );
        assert_eq!(
            template(
                "qwen35",
                [
                    Some("Qwen3.8 27B Abliterated"),
                    Some("Qwen3.8"),
                    None,
                    None,
                    None
                ]
            ),
            QwenPromptTemplate::Qwen38,
            "uncensored"
        );
        assert_eq!(
            template(
                "qwen35",
                [
                    Some("Qwen3.6 27B"),
                    Some("Qwen3.6"),
                    None,
                    None,
                    Some("https://huggingface.co/Qwen/Qwen3.6-27B/blob/main/LICENSE")
                ]
            ),
            QwenPromptTemplate::Qwen36,
            "3.6-27B-MTP shares the 3.8-27B shape"
        );
        assert_eq!(
            template(
                "qwen4exp",
                [Some("Qwen3.8 Flash Next"), None, None, None, None]
            ),
            QwenPromptTemplate::Qwen4Next
        );
        assert_eq!(
            template("qwen4exp", [Some("renamed"), None, None, None, None]),
            QwenPromptTemplate::Qwen4Next,
            "qwen4exp architecture implies the release"
        );
    }

    #[test]
    fn release_identity_is_unknown_on_absence_conflict_or_foreign_tokenizer() {
        let identity = identify_qwen_release(&facts(
            "qwen35",
            [Some("MyModel-27B"), None, None, None, None],
        ));
        assert_eq!(identity.template, QwenPromptTemplate::UnknownChatMl);
        let QwenReleaseStatus::Unknown {
            reason,
            fields_consulted,
        } = &identity.status
        else {
            panic!("{:?}", identity.status);
        };
        assert!(reason.contains("no Qwen release version"), "{reason}");
        assert!(fields_consulted.contains(&"general.base_model.0.name"));
        assert!(identity.warning().unwrap().contains("generic"));

        let conflict = identify_qwen_release(&facts(
            "qwen35",
            [Some("Qwen3.6-27B"), None, Some("Qwen3.8 27B"), None, None],
        ));
        assert!(
            matches!(&conflict.status, QwenReleaseStatus::Unknown { reason, .. } if reason.contains("general.name=qwen3.6") && reason.contains("general.base_model.0.name=qwen3.8"))
        );
        let two_in_one = identify_qwen_release(&facts(
            "qwen35",
            [
                Some("Qwen3.6 distilled from Qwen3.8"),
                None,
                None,
                None,
                None,
            ],
        ));
        assert!(matches!(
            two_in_one.status,
            QwenReleaseStatus::Unknown { .. }
        ));

        for (model, pre, count, key) in [
            (
                Some("llama"),
                Some("qwen35"),
                Some(248_320),
                "tokenizer.ggml.model",
            ),
            (
                Some("gpt2"),
                Some("qwen2"),
                Some(248_320),
                "tokenizer.ggml.pre",
            ),
            (
                Some("gpt2"),
                Some("qwen35"),
                Some(248_400),
                "tokenizer.ggml.tokens",
            ),
        ] {
            let mut facts = facts("qwen35", [Some("Qwen3.8-27B"), None, None, None, None]);
            facts.tokenizer_model = model;
            facts.tokenizer_pre = pre;
            facts.token_count = count;
            let identity = identify_qwen_release(&facts);
            assert_eq!(
                identity.template,
                QwenPromptTemplate::UnknownChatMl,
                "{key}"
            );
            assert!(
                matches!(&identity.status, QwenReleaseStatus::Unknown { reason, .. } if reason.contains(key))
            );
        }
        let mut flash = facts(
            "qwen4exp",
            [Some("Qwen3.8 Flash Next"), None, None, None, None],
        );
        flash.tokenizer_pre = Some("qwen2");
        assert_eq!(
            identify_qwen_release(&flash).template,
            QwenPromptTemplate::UnknownChatMl
        );
    }

    #[test]
    fn version_scan_boundaries() {
        assert_eq!(qwen_versions_in("Qwen3.85-9B"), Vec::<&str>::new());
        assert_eq!(qwen_versions_in("Qwen-3.8"), vec!["qwen3.8"]);
        assert_eq!(qwen_versions_in("qwen_3.6"), vec!["qwen3.6"]);
        assert_eq!(qwen_versions_in("Qwen 3.5"), vec!["qwen3.5"]);
        assert_eq!(qwen_versions_in("Qwen3.7-27B"), Vec::<&str>::new());
        assert_eq!(
            qwen_versions_in("https://huggingface.co/Qwen/Qwen3.6-27B"),
            vec!["qwen3.6"]
        );
        assert_eq!(
            qwen_versions_in("Qwen3.6 distilled from Qwen3.8"),
            vec!["qwen3.6", "qwen3.8"]
        );
        assert_eq!(qwen_versions_in("Qwen3.6 and qwen3.6"), vec!["qwen3.6"]);
    }

    #[test]
    fn identified_status_serializes_with_source() {
        let identity =
            identify_qwen_release(&facts("qwen35", [None, Some("Qwen3.8"), None, None, None]));
        assert_eq!(
            serde_json::to_value(&identity.status).unwrap(),
            serde_json::json!({"status": "identified", "version": "qwen3.8", "source": "general.basename"})
        );
    }

    /// Oracle fixture provenance. Not consulted for identity. The DS4 file
    /// is Unsloth's patched 0731 template (the one in the local UD repack);
    /// the DS4 renderer itself is pinned to the vLLM/SGLang encoders.
    #[test]
    fn tracked_template_oracles_keep_their_released_digests() {
        const DS4_FLASH_0731_UNSLOTH: [u8; 32] = [
            0xe6, 0x43, 0xc3, 0x1f, 0xce, 0xc1, 0x7f, 0x34, 0x2f, 0x72, 0x29, 0x6e, 0x02, 0xc4,
            0x6d, 0x35, 0x84, 0x6b, 0xf4, 0xc7, 0x0f, 0x6a, 0x02, 0x71, 0xf2, 0x3b, 0xad, 0x73,
            0xfd, 0x4e, 0xb6, 0x45,
        ];
        const QWEN36_A3B: [u8; 32] = [
            0x55, 0xd4, 0x93, 0x14, 0x33, 0xfe, 0x50, 0x2b, 0x79, 0x42, 0x26, 0xee, 0x7f, 0x4d,
            0x20, 0x6a, 0x6b, 0xdd, 0x43, 0x6a, 0xc9, 0xf8, 0x0e, 0xb7, 0xd8, 0xeb, 0xb4, 0xc6,
            0x39, 0xf9, 0xea, 0x0c,
        ];
        const QWEN38_27B: [u8; 32] = [
            0x70, 0x1b, 0xa1, 0x3a, 0x08, 0x5c, 0x0c, 0x1b, 0x5e, 0x05, 0x41, 0x4d, 0xec, 0x1a,
            0xa3, 0x06, 0x99, 0x04, 0xf9, 0x62, 0xbe, 0xee, 0x36, 0xf0, 0x89, 0x9e, 0x44, 0x17,
            0x20, 0xb8, 0x39, 0x74,
        ];
        for (template, expected) in [
            (
                include_str!("../tests/fixtures/templates/qwen36_a3b_chat_template.jinja"),
                QWEN36_A3B,
            ),
            (
                include_str!("../tests/fixtures/templates/qwen38_27b_chat_template.jinja"),
                QWEN38_27B,
            ),
            (
                include_str!("../tests/fixtures/templates/ds4_flash_chat_template.jinja"),
                DS4_FLASH_0731_UNSLOTH,
            ),
        ] {
            assert_eq!(
                <[u8; 32]>::from(Sha256::digest(template.as_bytes())),
                expected
            );
        }
    }

    /// Same-shape releases differ only in name-bearing and converter-artefact
    /// keys (Qwen3.5-27B vs Qwen3.6-27B; whole-header diff, 2026-09-17).
    #[test]
    #[ignore = "requires QWEN_IDENTITY_GGUF_A and QWEN_IDENTITY_GGUF_B"]
    fn same_shape_releases_differ_only_in_names() {
        let open = |var: &str| GgufFile::open(std::env::var(var).expect(var)).expect("open GGUF");
        let (a, b) = (open("QWEN_IDENTITY_GGUF_A"), open("QWEN_IDENTITY_GGUF_B"));
        let arch = a.architecture().unwrap();
        assert_eq!(arch, b.architecture().unwrap());
        for key in ["tokenizer.ggml.tokens", "tokenizer.ggml.merges"] {
            assert_eq!(
                a.get_array_len(key).unwrap(),
                b.get_array_len(key).unwrap(),
                "{key}"
            );
        }
        let ia = identify_qwen_release_for_gguf(&a).unwrap();
        let ib = identify_qwen_release_for_gguf(&b).unwrap();
        assert_ne!(ia.template, ib.template, "{ia:?} vs {ib:?}");
    }
}

// ---------------------------------------------------------------------------
// Ordinary-Qwen user-prompt protocol: the family-owned binding of a `user`
// (+ `system`) request and its reasoning controls to released template
// bytes. Transport-neutral (no clap, no HTTP): `qwen run --user`, batch
// `user` rows, and any future lane hand it plain strings and bools.
// ---------------------------------------------------------------------------

/// The rendering protocol for ordinary-Qwen `user`(+`system`) requests,
/// resolved once per model from the GGUF header, so there is exactly one
/// implementation of the released template bytes across lanes.
#[derive(Clone, Debug)]
pub(crate) struct QwenUserPromptProtocol {
    identity: QwenReleaseIdentity,
}

/// Transport-neutral reasoning controls for one request.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct QwenReasoningControls<'a> {
    /// `reasoning_effort` spelling as supplied; `None` when omitted.
    pub(crate) effort: Option<&'a str>,
    /// The released non-thinking transition was requested.
    pub(crate) no_thinking: bool,
}

impl QwenUserPromptProtocol {
    #[cfg(test)]
    pub(crate) fn for_test(qwen38: bool, template: QwenTemplate) -> Self {
        let release = match (qwen38, template) {
            (true, _) => QwenPromptTemplate::Qwen38,
            (false, QwenTemplate::Qwen35) => QwenPromptTemplate::Qwen35,
            (false, QwenTemplate::Qwen36) => QwenPromptTemplate::Qwen36,
            (false, QwenTemplate::Qwen38) => QwenPromptTemplate::Qwen38,
            (false, QwenTemplate::Generic) => QwenPromptTemplate::UnknownChatMl,
        };
        Self {
            identity: QwenReleaseIdentity {
                template: release,
                status: QwenReleaseStatus::Identified {
                    version: "test",
                    source: "test",
                },
            },
        }
    }

    /// `None` for families that do not render ordinary-Qwen chat.
    pub(crate) fn resolve(family: ModelFamily, gguf: &GgufFile) -> Result<Option<Self>> {
        if !matches!(family, ModelFamily::Qwen35 | ModelFamily::Qwen35Moe) {
            return Ok(None);
        }
        Ok(Some(Self {
            identity: identify_qwen_release_for_gguf(gguf)?,
        }))
    }

    pub(crate) fn is_qwen38(&self) -> bool {
        matches!(
            self.identity.template,
            QwenPromptTemplate::Qwen38 | QwenPromptTemplate::Qwen4Next
        )
    }

    pub(crate) fn status(&self) -> &QwenReleaseStatus {
        &self.identity.status
    }

    pub(crate) fn warning(&self) -> Option<String> {
        self.identity.warning()
    }

    pub(crate) fn template(&self) -> QwenTemplate {
        self.identity.template.serve_template()
    }

    /// Stable protocol label for records.
    pub(crate) fn label(&self) -> &'static str {
        self.template().label()
    }

    /// What this model accepts as reasoning controls: the effort levels (in
    /// release order, empty when the model has no effort control), the
    /// fallback when omitted, and whether no-thinking exists. Derived from
    /// the same tables `bind` parses with, so the advertisement cannot drift
    /// from the parser.
    pub(crate) fn reasoning_capability(&self) -> ReasoningCapability {
        if self.is_qwen38() {
            ReasoningCapability {
                levels: Qwen38GenerationMode::level_names(),
                fallback: Some("xhigh"),
                no_thinking: Support::Supported,
                thinking: Support::Supported,
            }
        } else {
            let identified = self.template().verified();
            let unsupported = || Support::Unsupported {
                code: "release_unknown",
                message: RELEASE_UNKNOWN_MESSAGE.into(),
            };
            ReasoningCapability {
                levels: Vec::new(),
                fallback: None,
                no_thinking: if identified {
                    Support::Supported
                } else {
                    unsupported()
                },
                thinking: if identified {
                    Support::Supported
                } else {
                    unsupported()
                },
            }
        }
    }

    /// Plain chat on an unidentified release is the legacy bare ChatML
    /// contract; the tool block exists only per release.
    pub(crate) fn input_capability(&self) -> InputCapability {
        InputCapability {
            raw: Support::Supported,
            user: Support::Supported,
            messages: Support::Supported,
            tools: qwen_tools_support(self.template()),
        }
    }

    /// Bind the controls to this model's generation mode. Qwen3.8 identities
    /// bind effort levels (including `none`); every other ordinary Qwen has
    /// no effort control and binds only the no-thinking transition.
    pub(crate) fn bind(
        &self,
        controls: QwenReasoningControls<'_>,
    ) -> Result<QwenBoundGeneration, CapabilityError> {
        if self.is_qwen38() {
            return Ok(QwenBoundGeneration::Qwen38(Qwen38GenerationMode::parse(
                controls.effort,
                controls.no_thinking,
            )?));
        }
        if let Some(effort) = controls.effort {
            return Err(CapabilityError {
                code: "reasoning_effort_unsupported",
                message: format!(
                    "reasoning effort {effort:?} applies to Qwen3.8 (low/medium/xhigh), DeepSeek V4 (none/low/high/max), and Muse Glimmer; this model has no reasoning-effort control"
                ),
            });
        }
        if controls.no_thinking && !self.template().verified() {
            return Err(CapabilityError {
                code: "no_thinking_unsupported",
                message: format!(
                    "no-thinking {RELEASE_UNKNOWN_MESSAGE}; omit it to use the default generation behavior"
                ),
            });
        }
        Ok(QwenBoundGeneration::Template(if controls.no_thinking {
            QwenGenerationMode::NoThinking
        } else {
            QwenGenerationMode::Auto
        }))
    }

    /// Render one user turn with optional system text under bound controls.
    pub(crate) fn render(
        &self,
        user: &str,
        system: Option<&str>,
        controls: QwenReasoningControls<'_>,
    ) -> Result<String> {
        match self.bind(controls)? {
            QwenBoundGeneration::Qwen38(mode) => Ok(
                crate::messages::render_qwen38_single_turn_prompt(user, system, mode),
            ),
            QwenBoundGeneration::Template(mode) => Ok(
                crate::messages::render_qwen_single_turn_prompt_for_template(
                    user,
                    system,
                    self.template(),
                    mode,
                ),
            ),
        }
    }
}

/// The bound generation decision for an ordinary-Qwen request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum QwenBoundGeneration {
    Qwen38(Qwen38GenerationMode),
    Template(QwenGenerationMode),
}

/// Whether a control exists for this model. `Unsupported` carries a stable
/// code and the message a lane would refuse with.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub(crate) enum Support {
    Supported,
    Unsupported { code: &'static str, message: String },
}

impl Support {
    /// The refusal a lane raises when a request uses an unsupported control:
    /// the same code and message the capability advertises.
    pub(crate) fn require(&self) -> Result<(), CapabilityError> {
        match self {
            Self::Supported => Ok(()),
            Self::Unsupported { code, message } => Err(CapabilityError {
                code,
                message: message.clone(),
            }),
        }
    }
}

pub(crate) const RELEASE_UNKNOWN_MESSAGE: &str = "requires an identified Qwen release (Qwen3.5/3.6/3.8 in the model's name metadata); this model's release is unknown";

/// The tool block is a per-release contract; one rule for `run --messages`
/// and serve.
pub(crate) fn qwen_tools_support(template: QwenTemplate) -> Support {
    if template.verified() {
        Support::Supported
    } else {
        Support::Unsupported {
            code: "tools_require_known_release",
            message: format!("tools and tool history {RELEASE_UNKNOWN_MESSAGE}"),
        }
    }
}

/// Projection of a family's input contract: which request forms its
/// template renders. A family-level fact from the header alone; a lane that
/// has not implemented a form reports that gap itself.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub(crate) struct InputCapability {
    /// Untemplated text (`--raw-prompt`, batch `prompt`/`prompt_file`).
    pub(crate) raw: Support,
    /// One user turn with optional system text (`--user`, batch `user`).
    pub(crate) user: Support,
    /// A chat document (`--messages`, serve items).
    pub(crate) messages: Support,
    /// Function-tool definitions and tool-call history inside a document.
    pub(crate) tools: Support,
}

impl InputCapability {
    /// A claim, not a default: a family may declare this only when every
    /// form has an oracle-backed renderer (DeepSeek V4: dual-source chat
    /// fixtures incl. tools; Muse Glimmer: released ATEM prompt tests).
    pub(crate) fn all_supported() -> Self {
        Self {
            raw: Support::Supported,
            user: Support::Supported,
            messages: Support::Supported,
            tools: Support::Supported,
        }
    }

    /// Raw text always renders; every templated form shares one refusal.
    pub(crate) fn raw_only(code: &'static str, message: String) -> Self {
        let unsupported = Support::Unsupported { code, message };
        Self {
            raw: Support::Supported,
            user: unsupported.clone(),
            messages: unsupported.clone(),
            tools: unsupported,
        }
    }

    /// Nothing renders (no recognised architecture).
    pub(crate) fn none(code: &'static str, message: String) -> Self {
        let unsupported = Support::Unsupported { code, message };
        Self {
            raw: unsupported.clone(),
            user: unsupported.clone(),
            messages: unsupported.clone(),
            tools: unsupported,
        }
    }
}

/// Projection of a family's reasoning contract: what `reasoning_effort`
/// accepts, what applies when omitted, and whether the explicit thinking
/// controls exist. Built from the same tables the binders parse with.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub(crate) struct ReasoningCapability {
    pub(crate) levels: Vec<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) fallback: Option<&'static str>,
    pub(crate) no_thinking: Support,
    pub(crate) thinking: Support,
}
