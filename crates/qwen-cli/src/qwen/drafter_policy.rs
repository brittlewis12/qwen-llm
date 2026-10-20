//! Pre-load drafter admission.
//!
//! Decides, from header-derived model facts and the execution lane, whether a
//! `--drafter` request is permitted *before* any Metal allocation. Until this
//! module existed the same question was answered in five places with three
//! different outcomes: rejected pre-load (Muse, Flash-Next, DeepSeek serve),
//! rejected after a full weight load (serve MoE target), or silently ignored
//! (DeepSeek CLI). `run` and `serve` now consume one resolver, settled once
//! before family dispatch so no family lane can accept and ignore the flag
//! (Muse's dispatch had skipped it).
//!
//! The decision is a projection of what the implementation supports; it is
//! not a capability table maintained by hand. Ordinary dense and MoE Qwen
//! targets share capture, verification, and rollback in both execution lanes.

use std::fmt;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use qwen_llm::gguf::GgufFile;
use qwen_llm::loader::{DFlashTargetShape, Model, open_dflash_drafter_for_target};
use qwen_llm::metal::MetalContext;
use qwen_llm::metal_dflash::MetalDFlashHead;
use qwen_llm::model_family::ModelFamily;

use crate::family_profile::{DrafterSupport, profile};

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
    /// Ordinary MoE Qwen target using packed verification and recurrent rollback.
    Moe,
}

/// Why a drafter request is refused. `code()` is the stable identifier for
/// machine consumers; `Display` is the human message.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DrafterUnsupported {
    /// The family has no speculative-decode implementation in this lane.
    FamilyNoSpeculation { family: ModelFamily, lane: Lane },
    /// Architecture is not one this binary recognises.
    UnknownFamily,
}

impl DrafterUnsupported {
    pub(crate) fn code(self) -> &'static str {
        match self {
            DrafterUnsupported::FamilyNoSpeculation { family, .. } => {
                match profile(family).drafter {
                    DrafterSupport::Unsupported(reason) => reason,
                    _ => unreachable!("family refusal requires unsupported profile"),
                }
            }
            DrafterUnsupported::UnknownFamily => "unknown_family",
        }
    }
}

impl fmt::Display for DrafterUnsupported {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DrafterUnsupported::FamilyNoSpeculation { family, lane } => {
                let family_label = profile(*family).display;
                let note = match family {
                    ModelFamily::MuseGlimmer => " (Muse DFlash2 integration is not active yet)",
                    _ => "",
                };
                write!(
                    f,
                    "--drafter is not supported for {family_label} {}{note} ({})",
                    lane.label(),
                    match profile(*family).drafter {
                        DrafterSupport::Unsupported(reason) => reason,
                        _ => unreachable!("family refusal requires unsupported profile"),
                    }
                )
            }
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
        Some(family) => match profile(family).drafter {
            DrafterSupport::Dense => DrafterDecision::Permitted(DrafterTarget::Dense),
            DrafterSupport::Moe => DrafterDecision::Permitted(DrafterTarget::Moe),
            DrafterSupport::Unsupported(_) => {
                DrafterDecision::Unsupported(DrafterUnsupported::FamilyNoSpeculation {
                    family,
                    lane,
                })
            }
        },
        None => DrafterDecision::Unsupported(DrafterUnsupported::UnknownFamily),
    }
}

/// A drafter whose GGUF has been opened and whose metadata has been bound
/// against validated target dimensions, before the target's GPU weights exist.
/// Holding the opened file means the later Metal load reuses the same
/// inspected asset rather than reopening a path that may have changed.
pub(crate) struct PreparedDrafter {
    path: PathBuf,
    gguf: GgufFile,
    target_shape: DFlashTargetShape,
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
        Self::prepare_with_shape(path, family, lane, || {
            let target_model =
                Model::from_gguf(target).context("parse target arch for drafter binding")?;
            Ok(DFlashTargetShape {
                hidden_size: target_model.arch.hidden_size,
                vocab_size: target_model.arch.vocab_size,
                n_layer: target_model.arch.n_layer,
            })
        })
    }

    /// Prepare using dimensions from a caller-validated target configuration,
    /// without parsing an ordinary Qwen model. This still enforces lane/family
    /// policy; dimensions alone do not establish generation support.
    pub(crate) fn prepare_for_target(
        path: Option<&Path>,
        target_shape: DFlashTargetShape,
        family: Option<ModelFamily>,
        lane: Lane,
    ) -> Result<Option<Self>> {
        Self::prepare_with_shape(path, family, lane, || Ok(target_shape))
    }

    fn prepare_with_shape(
        path: Option<&Path>,
        family: Option<ModelFamily>,
        lane: Lane,
        target_shape: impl FnOnce() -> Result<DFlashTargetShape>,
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
        let target_shape = target_shape()?;
        open_dflash_drafter_for_target(&gguf, target_shape)
            .with_context(|| format!("bind drafter {}", path.display()))?;
        Ok(Some(Self {
            path: path.to_path_buf(),
            gguf,
            target_shape,
        }))
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    pub(crate) fn bound_head(&self) -> Result<qwen_llm::loader::DFlashHead<'_>> {
        open_dflash_drafter_for_target(&self.gguf, self.target_shape)
            .with_context(|| format!("bind drafter {}", self.path.display()))
    }

    pub(crate) fn priced_metal_bytes(&self, ctx: &MetalContext) -> Result<u64> {
        let head = self.bound_head()?;
        MetalDFlashHead::priced_metal_weight_bytes(ctx, &head)
            .context("price drafter Metal weights")
    }

    /// Rebind the retained drafter against the prepared dimensions and copy it
    /// to the GPU. No target weights or target metadata parsing are needed.
    pub(crate) fn load(&self, ctx: &MetalContext) -> Result<MetalDFlashHead> {
        qwen_llm::runtime::prefetch_opened_gguf(
            &self.gguf,
            &qwen_llm::runtime::LoadedModelConfig::default(),
        );
        let bound = self.bound_head()?;
        MetalDFlashHead::load(ctx, &self.gguf, &bound).context("metal-load drafter")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preparation_without_a_drafter_skips_target_dimensions() {
        for family in std::iter::once(None).chain(ModelFamily::ALL.iter().copied().map(Some)) {
            for lane in [Lane::CliSingleTurn, Lane::Serve] {
                assert!(
                    PreparedDrafter::prepare_with_shape(None, family, lane, || {
                        panic!("absent drafter must not parse target dimensions")
                    })
                    .unwrap()
                    .is_none()
                );
                assert!(
                    PreparedDrafter::prepare_for_target(
                        None,
                        DFlashTargetShape {
                            hidden_size: 0,
                            vocab_size: 0,
                            n_layer: 0,
                        },
                        family,
                        lane,
                    )
                    .unwrap()
                    .is_none()
                );
            }
        }
    }

    #[test]
    fn preparation_refuses_policy_before_drafter_io_or_target_dimensions() {
        let shape = DFlashTargetShape {
            hidden_size: 32,
            vocab_size: 64,
            n_layer: 4,
        };
        for family in std::iter::once(None).chain(ModelFamily::ALL.iter().copied().map(Some)) {
            for lane in [Lane::CliSingleTurn, Lane::Serve] {
                let DrafterDecision::Unsupported(reason) = resolve_drafter(family, lane, true)
                else {
                    continue;
                };
                let error =
                    PreparedDrafter::prepare_with_shape(Some(Path::new("")), family, lane, || {
                        panic!("refused drafter must not parse target dimensions")
                    })
                    .err()
                    .expect("unsupported family must be refused");
                assert_eq!(error.to_string(), reason.to_string());
                let error =
                    PreparedDrafter::prepare_for_target(Some(Path::new("")), shape, family, lane)
                        .err()
                        .expect("validated dimensions must not bypass policy");
                assert_eq!(error.to_string(), reason.to_string());
            }
        }
    }

    #[test]
    fn unreadable_drafter_still_fails_before_ordinary_target_parsing() {
        let error = PreparedDrafter::prepare_with_shape(
            Some(Path::new("")),
            Some(ModelFamily::Qwen35),
            Lane::CliSingleTurn,
            || panic!("unreadable drafter must fail before parsing the target"),
        )
        .err()
        .expect("empty drafter path cannot be opened");
        assert_eq!(error.to_string(), "open drafter ");
    }

    #[test]
    fn absent_request_is_not_requested_for_every_family_and_lane() {
        for family in [
            None,
            Some(ModelFamily::Qwen35),
            Some(ModelFamily::Qwen35Moe),
            Some(ModelFamily::Qwen4Exp),
            Some(ModelFamily::DeepSeek4),
            Some(ModelFamily::MuseGlimmer),
            Some(ModelFamily::K2Horizon),
            Some(ModelFamily::Glm5Next),
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
    fn moe_qwen_is_permitted_in_both_lanes() {
        for lane in [Lane::CliSingleTurn, Lane::Serve] {
            assert_eq!(
                resolve_drafter(Some(ModelFamily::Qwen35Moe), lane, true),
                DrafterDecision::Permitted(DrafterTarget::Moe)
            );
        }
    }

    #[test]
    fn families_without_speculation_are_refused_in_both_lanes() {
        for family in [
            ModelFamily::DeepSeek4,
            ModelFamily::MuseGlimmer,
            ModelFamily::Qwen4Exp,
            ModelFamily::K2Horizon,
            ModelFamily::Glm5Next,
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
            DrafterUnsupported::UnknownFamily.code(),
        ];
        assert_eq!(codes, ["family_no_speculation", "unknown_family"]);
    }

    #[test]
    fn messages_preserve_the_previous_per_family_wording() {
        let muse = DrafterUnsupported::FamilyNoSpeculation {
            family: ModelFamily::MuseGlimmer,
            lane: Lane::Serve,
        };
        assert_eq!(
            muse.to_string(),
            "--drafter is not supported for Muse Glimmer serve (Muse DFlash2 integration is not active yet) (family_no_speculation)"
        );
        let ds4 = DrafterUnsupported::FamilyNoSpeculation {
            family: ModelFamily::DeepSeek4,
            lane: Lane::CliSingleTurn,
        };
        assert_eq!(
            ds4.to_string(),
            "--drafter is not supported for DeepSeek V4 run (family_no_speculation)"
        );
    }

    #[test]
    fn shared_refusal_covers_k2_and_glm() {
        for (family, label) in [
            (ModelFamily::K2Horizon, "K2 Horizon"),
            (ModelFamily::Glm5Next, "GLM-5.3-Flash"),
        ] {
            let DrafterDecision::Unsupported(reason) =
                resolve_drafter(Some(family), Lane::CliSingleTurn, true)
            else {
                panic!("{family:?} must share the family refusal");
            };
            assert_eq!(
                reason.to_string(),
                format!("--drafter is not supported for {label} run (family_no_speculation)")
            );
        }
    }
}
