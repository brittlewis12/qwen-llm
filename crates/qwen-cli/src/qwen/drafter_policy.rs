//! Pre-load drafter admission.
//!
//! Decides, from header-derived model facts and the execution lane, whether a
//! `--drafter` request is permitted *before* any Metal allocation. Until this
//! module existed the same question was answered in five places with three
//! different outcomes: rejected pre-load (Muse, Flash-Next, DeepSeek serve),
//! rejected after a full weight load (serve MoE target), or silently ignored
//! (DeepSeek CLI). `run` and `serve` now consume one resolver, and the
//! remaining post-load checks are defensive duplicates only.
//!
//! The decision is a projection of what the implementation supports; it is
//! not a capability table maintained by hand. Lane differences that are real
//! (ordinary MoE targets are accepted serially on the CLI but refused by
//! serve) stay explicit rather than being flattened into a global rule.

use std::fmt;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use qwen_llm::gguf::GgufFile;
use qwen_llm::loader::{Model, open_dflash_drafter};
use qwen_llm::metal::MetalContext;
use qwen_llm::metal_dflash::MetalDFlashHead;
use qwen_llm::model_family::ModelFamily;

/// Execution lane the drafter would serve. Batch lanes (`--requests-jsonl`,
/// `--batch-size`, `--concurrency`) are rejected earlier by the cross-flag
/// validator and never reach this resolver.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Lane {
    CliSingleTurn,
    Serve,
}

impl Lane {
    fn label(self) -> &'static str {
        match self {
            Lane::CliSingleTurn => "run",
            Lane::Serve => "serve",
        }
    }
}

/// Outcome of resolving a drafter request against model facts and lane.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DrafterDecision {
    /// No `--drafter` given; nothing to admit.
    NotRequested,
    /// The lane and family support DFlash speculation for this target shape.
    Permitted(DrafterTarget),
    /// The request cannot be honoured; the reason is stable and machine-legible.
    Unsupported(DrafterUnsupported),
}

/// Target shape the drafter will be bound against.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DrafterTarget {
    /// Dense Qwen3.5/3.6/3.8 target: the qualified DFlash lane.
    Dense,
    /// MoE Qwen target on the CLI: accepted today (binder enforces hidden
    /// size), but speculation runs serially and is unqualified. Serve refuses
    /// this shape; the CLI keeps its historical acceptance.
    MoeCliSerial,
}

/// Why a drafter request is refused. `code()` is the stable identifier for
/// machine consumers; `Display` is the human message.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DrafterUnsupported {
    /// The family has no speculative-decode implementation in this lane.
    FamilyNoSpeculation { family: ModelFamily, lane: Lane },
    /// Serve speculation is qualified for dense targets only.
    ServeMoeTarget,
    /// Architecture is not one this binary recognises.
    UnknownFamily,
}

impl DrafterUnsupported {
    pub(crate) fn code(self) -> &'static str {
        match self {
            DrafterUnsupported::FamilyNoSpeculation { .. } => "family_no_speculation",
            DrafterUnsupported::ServeMoeTarget => "serve_moe_target",
            DrafterUnsupported::UnknownFamily => "unknown_family",
        }
    }
}

impl fmt::Display for DrafterUnsupported {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DrafterUnsupported::FamilyNoSpeculation { family, lane } => {
                let family_label = match family {
                    ModelFamily::DeepSeek4 => "DeepSeek V4",
                    ModelFamily::MuseGlimmer => "Muse Glimmer",
                    ModelFamily::Qwen4Exp => "Qwen3.8-Flash-Next",
                    ModelFamily::Qwen35 | ModelFamily::Qwen35Moe => "this Qwen",
                };
                let note = match family {
                    ModelFamily::MuseGlimmer => " (Muse DFlash2 integration is not active yet)",
                    _ => "",
                };
                write!(
                    f,
                    "--drafter is not supported for {family_label} {}{note}",
                    lane.label()
                )
            }
            DrafterUnsupported::ServeMoeTarget => write!(
                f,
                "serve DFlash speculation currently supports dense targets only; this MoE model would run serially, so omit --drafter"
            ),
            DrafterUnsupported::UnknownFamily => {
                write!(
                    f,
                    "--drafter requires a recognised Qwen target architecture"
                )
            }
        }
    }
}

/// Pure policy: no I/O, no Metal. Safe to call from a metadata-only
/// capabilities query as well as from dispatch.
pub(crate) fn resolve_drafter(
    family: Option<ModelFamily>,
    lane: Lane,
    requested: bool,
) -> DrafterDecision {
    if !requested {
        return DrafterDecision::NotRequested;
    }
    match family {
        Some(ModelFamily::Qwen35) => DrafterDecision::Permitted(DrafterTarget::Dense),
        Some(ModelFamily::Qwen35Moe) => match lane {
            Lane::CliSingleTurn => DrafterDecision::Permitted(DrafterTarget::MoeCliSerial),
            Lane::Serve => DrafterDecision::Unsupported(DrafterUnsupported::ServeMoeTarget),
        },
        Some(
            family @ (ModelFamily::DeepSeek4 | ModelFamily::MuseGlimmer | ModelFamily::Qwen4Exp),
        ) => DrafterDecision::Unsupported(DrafterUnsupported::FamilyNoSpeculation { family, lane }),
        None => DrafterDecision::Unsupported(DrafterUnsupported::UnknownFamily),
    }
}

/// A drafter whose GGUF has been opened and whose metadata has been bound
/// against the target's header, all before the target's weights exist on the
/// GPU. Holding the opened file means the later Metal load reuses the same
/// inspected asset rather than reopening a path that may have changed.
pub(crate) struct PreparedDrafter {
    path: PathBuf,
    gguf: GgufFile,
}

impl PreparedDrafter {
    /// Resolve policy, then open and bind. Returns `Ok(None)` when no drafter
    /// was requested. Fails before any Metal allocation on policy refusal,
    /// unreadable drafter, or metadata that does not bind to this target.
    pub(crate) fn prepare(
        path: Option<&Path>,
        target: &GgufFile,
        family: Option<ModelFamily>,
        lane: Lane,
    ) -> Result<Option<Self>> {
        let Some(path) = path else {
            return Ok(None);
        };
        match resolve_drafter(family, lane, true) {
            DrafterDecision::NotRequested => unreachable!("drafter path present"),
            DrafterDecision::Unsupported(reason) => bail!("{reason}"),
            DrafterDecision::Permitted(_) => {}
        }
        let gguf =
            GgufFile::open(path).with_context(|| format!("open drafter {}", path.display()))?;
        let target_model =
            Model::from_gguf(target).context("parse target arch for drafter binding")?;
        open_dflash_drafter(&gguf, &target_model)
            .with_context(|| format!("bind drafter {}", path.display()))?;
        Ok(Some(Self {
            path: path.to_path_buf(),
            gguf,
        }))
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    /// Bind again (header-only, cheap) and copy the drafter to the GPU. The
    /// target `GgufFile` must be the same asset `prepare` bound against.
    pub(crate) fn load(&self, ctx: &MetalContext, target: &GgufFile) -> Result<MetalDFlashHead> {
        qwen_llm::runtime::prefetch_opened_gguf(
            &self.gguf,
            &qwen_llm::runtime::LoadedModelConfig::default(),
        );
        let target_model =
            Model::from_gguf(target).context("parse target arch for drafter binding")?;
        let bound = open_dflash_drafter(&self.gguf, &target_model)
            .with_context(|| format!("bind drafter {}", self.path.display()))?;
        MetalDFlashHead::load(ctx, &self.gguf, &bound).context("metal-load drafter")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absent_request_is_not_requested_for_every_family_and_lane() {
        for family in [
            None,
            Some(ModelFamily::Qwen35),
            Some(ModelFamily::Qwen35Moe),
            Some(ModelFamily::Qwen4Exp),
            Some(ModelFamily::DeepSeek4),
            Some(ModelFamily::MuseGlimmer),
        ] {
            for lane in [Lane::CliSingleTurn, Lane::Serve] {
                assert_eq!(
                    resolve_drafter(family, lane, false),
                    DrafterDecision::NotRequested
                );
            }
        }
    }

    #[test]
    fn dense_qwen_is_permitted_in_both_lanes() {
        for lane in [Lane::CliSingleTurn, Lane::Serve] {
            assert_eq!(
                resolve_drafter(Some(ModelFamily::Qwen35), lane, true),
                DrafterDecision::Permitted(DrafterTarget::Dense)
            );
        }
    }

    #[test]
    fn moe_qwen_keeps_the_lane_distinction() {
        assert_eq!(
            resolve_drafter(Some(ModelFamily::Qwen35Moe), Lane::CliSingleTurn, true),
            DrafterDecision::Permitted(DrafterTarget::MoeCliSerial)
        );
        assert_eq!(
            resolve_drafter(Some(ModelFamily::Qwen35Moe), Lane::Serve, true),
            DrafterDecision::Unsupported(DrafterUnsupported::ServeMoeTarget)
        );
    }

    #[test]
    fn families_without_speculation_are_refused_in_both_lanes() {
        for family in [
            ModelFamily::DeepSeek4,
            ModelFamily::MuseGlimmer,
            ModelFamily::Qwen4Exp,
        ] {
            for lane in [Lane::CliSingleTurn, Lane::Serve] {
                let decision = resolve_drafter(Some(family), lane, true);
                assert_eq!(
                    decision,
                    DrafterDecision::Unsupported(DrafterUnsupported::FamilyNoSpeculation {
                        family,
                        lane
                    }),
                    "{family:?} {lane:?}"
                );
            }
        }
    }

    #[test]
    fn unknown_architecture_is_refused() {
        assert_eq!(
            resolve_drafter(None, Lane::CliSingleTurn, true),
            DrafterDecision::Unsupported(DrafterUnsupported::UnknownFamily)
        );
    }

    #[test]
    fn reason_codes_are_stable_and_distinct() {
        let codes = [
            DrafterUnsupported::FamilyNoSpeculation {
                family: ModelFamily::DeepSeek4,
                lane: Lane::Serve,
            }
            .code(),
            DrafterUnsupported::ServeMoeTarget.code(),
            DrafterUnsupported::UnknownFamily.code(),
        ];
        assert_eq!(
            codes,
            [
                "family_no_speculation",
                "serve_moe_target",
                "unknown_family"
            ]
        );
    }

    #[test]
    fn messages_preserve_the_previous_per_family_wording() {
        let muse = DrafterUnsupported::FamilyNoSpeculation {
            family: ModelFamily::MuseGlimmer,
            lane: Lane::Serve,
        };
        assert_eq!(
            muse.to_string(),
            "--drafter is not supported for Muse Glimmer serve (Muse DFlash2 integration is not active yet)"
        );
        let ds4 = DrafterUnsupported::FamilyNoSpeculation {
            family: ModelFamily::DeepSeek4,
            lane: Lane::CliSingleTurn,
        };
        assert_eq!(
            ds4.to_string(),
            "--drafter is not supported for DeepSeek V4 run"
        );
        assert!(
            DrafterUnsupported::ServeMoeTarget
                .to_string()
                .contains("dense targets only")
        );
    }
}
