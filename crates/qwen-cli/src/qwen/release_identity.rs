//! Which released model a GGUF is, for decisions keyed by release.
//!
//! Detection only: every heuristic that reads header facts to decide what a
//! model is lives here, and decisions consume the resulting identity (see
//! `release_sampling`). Converter-written fields that restate a decision,
//! such as `general.sampling.*`, are deliberately not identity inputs.
//! Unknown releases may still use them as sampling defaults.

use crate::prompt_template::{QwenReleaseStatus, identify_qwen_release_for_gguf};
use qwen_llm::gguf::GgufFile;
use qwen_llm::model_family::ModelFamily;

/// Qwen release lineage, from the header's name fields
/// (`prompt_template::identify_qwen_release`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum QwenVersion {
    Qwen35,
    Qwen36,
    Qwen38,
    Unknown,
}

/// Released Qwen3.x shapes that release-keyed decisions distinguish, from
/// header geometry. Qwen ships identical architectures across versions at a
/// given size, so shape and version are independent axes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum QwenShape {
    /// 64 layers at width 5120 (65 when the MTP layer is counted).
    Dense27b,
    /// 40 layers at width 2048, 256 experts, 8 routed.
    Moe35bA3b,
    /// 48 layers at width 3072, 256 experts, 8 routed.
    Moe122bA10b,
    Other,
}

impl QwenShape {
    pub(crate) fn classify(
        family: ModelFamily,
        block_count: Option<u64>,
        embedding_length: Option<u64>,
        expert_count: Option<u64>,
        expert_used_count: Option<u64>,
    ) -> Self {
        let experts = (expert_count, expert_used_count);
        match (family, block_count, embedding_length) {
            (ModelFamily::Qwen35, Some(64 | 65), Some(5120)) => Self::Dense27b,
            (ModelFamily::Qwen35Moe, Some(40), Some(2048)) if experts == (Some(256), Some(8)) => {
                Self::Moe35bA3b
            }
            (ModelFamily::Qwen35Moe, Some(48), Some(3072)) if experts == (Some(256), Some(8)) => {
                Self::Moe122bA10b
            }
            _ => Self::Other,
        }
    }

    fn from_header(family: ModelFamily, gguf: &GgufFile) -> Self {
        let key = |suffix: &str| gguf.get_u64(&format!("{}.{suffix}", family.architecture_name()));
        Self::classify(
            family,
            key("block_count"),
            key("embedding_length"),
            key("expert_count"),
            key("expert_used_count"),
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReleaseIdentity {
    Qwen {
        version: QwenVersion,
        shape: QwenShape,
    },
    FlashNext,
    Qwen4Unknown,
    DeepSeekV4,
    DeepSeekV41,
    MuseGlimmer,
    K2Horizon,
    Glm53Flash,
}

impl ReleaseIdentity {
    /// Infallible: a header too malformed to read a release from is an
    /// unidentified release, so detection alone never fails a run, a serve
    /// startup or `qwen info` (whatever needs those facts fails on its own).
    pub(crate) fn detect(family: ModelFamily, gguf: &GgufFile) -> Self {
        match family {
            ModelFamily::Qwen35 | ModelFamily::Qwen35Moe => Self::Qwen {
                version: match identify_qwen_release_for_gguf(gguf).map(|id| id.status) {
                    Ok(QwenReleaseStatus::Identified {
                        version: "qwen3.5", ..
                    }) => QwenVersion::Qwen35,
                    Ok(QwenReleaseStatus::Identified {
                        version: "qwen3.6", ..
                    }) => QwenVersion::Qwen36,
                    Ok(QwenReleaseStatus::Identified {
                        version: "qwen3.8", ..
                    }) => QwenVersion::Qwen38,
                    _ => QwenVersion::Unknown,
                },
                shape: QwenShape::from_header(family, gguf),
            },
            ModelFamily::Qwen4Exp => match identify_qwen_release_for_gguf(gguf).map(|id| id.status)
            {
                Ok(QwenReleaseStatus::Identified {
                    version: "qwen3.8", ..
                }) => Self::FlashNext,
                _ => Self::Qwen4Unknown,
            },
            ModelFamily::DeepSeek4 => Self::DeepSeekV4,
            ModelFamily::DeepSeekV41 => Self::DeepSeekV41,
            ModelFamily::MuseGlimmer => Self::MuseGlimmer,
            ModelFamily::K2Horizon => Self::K2Horizon,
            ModelFamily::Glm5Next => Self::Glm53Flash,
        }
    }

    /// Stable label for `qwen info -m MODEL --json` and diagnostics.
    pub(crate) fn label(self) -> String {
        match self {
            Self::Qwen { version, shape } => {
                let version = match version {
                    QwenVersion::Qwen35 => "qwen3.5",
                    QwenVersion::Qwen36 => "qwen3.6",
                    QwenVersion::Qwen38 => "qwen3.8",
                    QwenVersion::Unknown => "qwen_unidentified",
                };
                let shape = match shape {
                    QwenShape::Dense27b => "dense_27b",
                    QwenShape::Moe35bA3b => "moe_35b_a3b",
                    QwenShape::Moe122bA10b => "moe_122b_a10b",
                    QwenShape::Other => "other_shape",
                };
                format!("{version}/{shape}")
            }
            Self::FlashNext => "qwen3.8_flash_next".into(),
            Self::Qwen4Unknown => "qwen4_unidentified".into(),
            Self::DeepSeekV4 => "deepseek_v4_0731".into(),
            Self::DeepSeekV41 => "deepseek_v41".into(),
            Self::MuseGlimmer => "muse_glimmer".into(),
            Self::K2Horizon => "k2_horizon".into(),
            Self::Glm53Flash => "glm_5.3_flash".into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen4_release_labels_sampling_and_prompt_identity_share_qualification() {
        use crate::prompt_template::resolve_qwen_contract_for_gguf;
        use crate::qwen_contract_fixture::{Value, gguf, qwen4_reference_gguf};
        use crate::release_sampling::{gguf_sampling_defaults, release_sampling_defaults};
        let check = |gguf: &GgufFile, qualified: bool| {
            let public = identify_qwen_release_for_gguf(gguf).unwrap();
            let prompt = resolve_qwen_contract_for_gguf(gguf).unwrap();
            assert_eq!(
                matches!(
                    public.status,
                    QwenReleaseStatus::Identified {
                        version: "qwen3.8",
                        ..
                    }
                ),
                matches!(
                    prompt.status,
                    QwenReleaseStatus::Identified {
                        version: "qwen3.8",
                        ..
                    }
                ),
            );
            assert_eq!(
                matches!(
                    public.status,
                    QwenReleaseStatus::Identified {
                        version: "qwen3.8",
                        ..
                    }
                ),
                qualified
            );
            let runtime = ReleaseIdentity::detect(ModelFamily::Qwen4Exp, gguf);
            assert_eq!(
                runtime,
                if qualified {
                    ReleaseIdentity::FlashNext
                } else {
                    ReleaseIdentity::Qwen4Unknown
                }
            );
            assert_eq!(
                runtime.label(),
                if qualified {
                    "qwen3.8_flash_next"
                } else {
                    "qwen4_unidentified"
                }
            );
            assert_eq!(
                gguf_sampling_defaults(runtime, gguf),
                release_sampling_defaults(runtime)
            );
        };
        for name in ["Qwen3.8-Flash-Next", "Qwen3.8-Flash-Next GSQ-RCO"] {
            check(&qwen4_reference_gguf("qwen4exp", name, |_| {}), true);
        }
        for name in [
            "Distilled Qwen3.8-Flash-Next",
            "renamed model",
            "Qwen4-Flash",
        ] {
            check(&qwen4_reference_gguf("qwen4exp", name, |_| {}), false);
        }
        check(
            &qwen4_reference_gguf("qwen4exp", "Qwen3.8-Flash-Next", |c| {
                c.context_length = 1_048_576
            }),
            false,
        );
        check(
            &qwen4_reference_gguf("qwen4exp", "Qwen3.8-Flash-Next", |c| c.hidden_size = 4096),
            false,
        );
        check(
            &qwen4_reference_gguf("qwen4", "Qwen3.8-Flash-Next", |_| {}),
            false,
        );
        let incomplete = gguf(
            "Qwen3.8-Flash-Next",
            "",
            &[
                ("general.architecture", Value::Text("qwen4exp")),
                ("general.sampling.temp", Value::Float(0.37)),
            ],
        );
        let unknown = ReleaseIdentity::detect(ModelFamily::Qwen4Exp, &incomplete);
        assert_eq!(unknown, ReleaseIdentity::Qwen4Unknown);
        // Unknown Qwen4 uses validated header defaults, rather than silently
        // taking a release-specific policy merely from its architecture.
        assert_eq!(
            gguf_sampling_defaults(unknown, &incomplete).temperature,
            0.37
        );
        assert_eq!(
            gguf_sampling_defaults(ReleaseIdentity::FlashNext, &incomplete).temperature,
            1.0
        );
    }

    #[test]
    fn qwen_shapes_classify_from_released_geometry_only() {
        let dense = |blocks, width| {
            QwenShape::classify(ModelFamily::Qwen35, Some(blocks), Some(width), None, None)
        };
        let moe = |blocks, width, experts, used| {
            QwenShape::classify(
                ModelFamily::Qwen35Moe,
                Some(blocks),
                Some(width),
                Some(experts),
                Some(used),
            )
        };
        // Qwen3.5/3.6-27B count 64 blocks; Qwen3.8-27B headers count its MTP layer.
        assert_eq!(dense(64, 5120), QwenShape::Dense27b);
        assert_eq!(dense(65, 5120), QwenShape::Dense27b);
        // 0.8B, 2B, 4B, 9B.
        for (blocks, width) in [(24, 1024), (24, 2048), (32, 2560), (32, 4096)] {
            assert_eq!(dense(blocks, width), QwenShape::Other);
        }
        assert_eq!(moe(40, 2048, 256, 8), QwenShape::Moe35bA3b);
        assert_eq!(moe(48, 3072, 256, 8), QwenShape::Moe122bA10b);
        assert_eq!(moe(48, 3072, 128, 8), QwenShape::Other);
        // A dense family never takes a MoE shape, and missing facts never match.
        assert_eq!(
            QwenShape::classify(
                ModelFamily::Qwen35,
                Some(40),
                Some(2048),
                Some(256),
                Some(8)
            ),
            QwenShape::Other
        );
        assert_eq!(
            QwenShape::classify(ModelFamily::Qwen35, None, Some(5120), None, None),
            QwenShape::Other
        );
    }
}
