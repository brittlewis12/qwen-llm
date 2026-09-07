//! Shared discovery for the local model fixtures that GPU-gated tests use.
//!
//! Before this module each test file resolved fixtures its own way: one
//! release GGUF was reachable through thirteen different environment
//! variables, the A3B path literal appeared dozens of times, and the
//! `QWEN_REQUIRE_METAL_TESTS` gate had six identical copies. This module owns
//! the fixture inventory, the environment aliases (all preserved), and the
//! two questions every such test asks: "is the fixture available or should I
//! skip?" and "is Metal available or should I skip?"
//!
//! Distinctions kept deliberately:
//! - *Unavailable* (no alias set, default path absent) is a skip. *Explicitly
//!   requested but unreadable* (an alias set to a missing file) is a failure —
//!   a caller asked for something specific and did not get it.
//! - Aliases are consulted in the order listed; setting two aliases to
//!   different values is a conflict and fails rather than silently picking one.
//! - Metal being required (`QWEN_REQUIRE_METAL_TESTS` truthy) turns a
//!   would-be skip into a failure. It never changes what a fixture resolves to.
//!
//! This module is compiled into the library so the crate's integration tests
//! (a separate crate) can reach it; it has no dependencies beyond `std` and
//! the Metal context constructor.

use std::path::{Path, PathBuf};

use crate::env_flag::env_value_truthy;
use crate::metal::{MetalContext, MetalError};

/// One local model fixture: a stable id, the environment aliases that may
/// override its location, and the path it lives at on the reference machine.
#[derive(Clone, Copy, Debug)]
pub struct Fixture {
    pub id: &'static str,
    /// Aliases in precedence order. All historical names are kept.
    pub env: &'static [&'static str],
    pub default_path: &'static str,
}

/// Qwen3.6-35B-A3B UD-Q4_K_M (MoE hybrid; the most-referenced fixture).
pub const A3B_Q4_K_M: Fixture = Fixture {
    id: "qwen3.6-35b-a3b-ud-q4_k_m",
    env: &["QWEN_A3B_MOE_MODEL"],
    default_path: "/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
};

/// Qwen3.6-27B Q4_K_M (dense).
pub const QWEN36_27B_Q4_K_M: Fixture = Fixture {
    id: "qwen3.6-27b-q4_k_m",
    env: &[],
    default_path: "/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf",
};

/// Qwen3.5-0.8B F32 (smallest dense; CPU-oracle comparisons).
pub const QWEN35_0_8B_F32: Fixture = Fixture {
    id: "qwen3.5-0.8b-f32",
    env: &[],
    default_path: "/Users/tito/models/Qwen3.5-0.8B.F32.gguf",
};

/// Qwen3.5-122B-A10B UD-Q4_K_XL (large MoE; residency and prefetch tests).
pub const A10B_Q4_K_XL: Fixture = Fixture {
    id: "qwen3.5-122b-a10b-ud-q4_k_xl",
    env: &[],
    default_path: "/Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf",
};

/// Qwen3.5-35B-A3B Q3_K_M (older A3B quant used by MoE kernel tests).
pub const A3B_Q3_K_M: Fixture = Fixture {
    id: "qwen3.5-35b-a3b-q3_k_m",
    env: &["QWEN_A3B_Q3_MODEL"],
    default_path: "/Users/tito/models/Qwen3.5-35B-A3B-Q3_K_M.gguf",
};

/// Qwen3.5-35B-A3B UD-IQ4_XS.
pub const A3B_IQ4_XS: Fixture = Fixture {
    id: "qwen3.5-35b-a3b-ud-iq4_xs",
    env: &["QWEN_A3B_UDIQ4XS_MODEL"],
    default_path: "/Users/tito/models/Qwen3.5-35B-A3B-UD-IQ4_XS.gguf",
};

/// Muse Glimmer 30B Q8_0.
pub const MUSE_GLIMMER_Q8_0: Fixture = Fixture {
    id: "muse-glimmer-30b-q8_0",
    env: &["MUSE_GLIMMER_Q8_GGUF"],
    default_path: "/Users/tito/models/muse-glimmer/Muse-Glimmer-30B-Q8_0.gguf",
};

/// DeepSeek V4 Flash 0731 UD-IQ3_XXS (first shard).
pub const DEEPSEEK_V4_IQ3_XXS: Fixture = Fixture {
    id: "deepseek-v4-flash-0731-ud-iq3_xxs",
    env: &[],
    default_path: "/Users/tito/models/deepseek-v4-flash-0731/UD-IQ3_XXS/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf",
};

/// Qwen3.8-Flash-Next UD-Q3_K_XL (first shard). One file, historically
/// thirteen environment names; every one still works.
pub const QWEN4EXP_Q3_K_XL: Fixture = Fixture {
    id: "qwen3.8-flash-next-ud-q3_k_xl",
    env: &[
        "QWEN4EXP_Q3_K_XL_GGUF",
        "QWEN4EXP_Q3_K_XL_RUNTIME_GGUF",
        "QWEN4EXP_Q3_K_XL_TEXT_SESSION_GGUF",
        "QWEN4EXP_Q3_K_XL_QSA_GGUF",
        "QWEN4EXP_Q3_K_XL_PLE_GGUF",
        "QWEN4EXP_Q3_K_XL_MOE_GGUF",
        "QWEN4EXP_Q3_K_XL_GDN_GGUF",
        "QWEN4EXP_Q3_K_XL_LAYER_ZERO_GGUF",
        "QWEN4EXP_Q3_K_XL_LAYERS_ZERO_ONE_GGUF",
        "QWEN4EXP_Q3_K_XL_LAYERS_ZERO_THREE_GGUF",
        "QWEN4EXP_Q3_K_XL_REALIZE_GGUF",
        "QWEN4EXP_TOKENIZER_GGUF",
        "QWEN4EXP_METADATA_GGUF",
    ],
    default_path: "/Users/tito/models/Qwen3.8-Flash-Next-UD-Q3_K_XL/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf",
};

/// Spiritbuun DFlash drafter for Qwen3.6 (v1 drafter convention).
pub const DFLASH_DRAFT_36_Q8_0: Fixture = Fixture {
    id: "dflash-draft-3.6-q8_0",
    env: &[],
    default_path: "/Users/tito/models/spiritbuun-dflash/dflash-draft-3.6-q8_0.gguf",
};

/// Where a fixture came from, for messages and for tests that care.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FixtureSource {
    /// An environment alias named it.
    Env(&'static str),
    /// The reference-machine default.
    Default,
}

#[derive(Clone, Debug)]
pub struct ResolvedFixture {
    pub path: PathBuf,
    pub source: FixtureSource,
}

impl Fixture {
    /// The reference-machine path as a literal. Type-preserving replacement
    /// for the string literals tests used to carry; consults no environment.
    pub const fn path(&self) -> &'static str {
        self.default_path
    }

    /// Consult aliases in order. Returns `Err` on a conflict between two set
    /// aliases; `Ok(None)` when no alias is set.
    fn env_override(&self) -> Result<Option<ResolvedFixture>, String> {
        let mut chosen: Option<(&'static str, PathBuf)> = None;
        for &name in self.env {
            let Some(value) = std::env::var_os(name) else {
                continue;
            };
            let value = PathBuf::from(value);
            match &chosen {
                None => chosen = Some((name, value)),
                Some((first, first_value)) if *first_value != value => {
                    return Err(format!(
                        "fixture {}: {first}={} conflicts with {name}={}",
                        self.id,
                        first_value.display(),
                        value.display()
                    ));
                }
                Some(_) => {}
            }
        }
        Ok(chosen.map(|(name, path)| ResolvedFixture {
            path,
            source: FixtureSource::Env(name),
        }))
    }

    /// Resolve to a readable path, or `None` when the fixture is simply not
    /// present on this machine. Panics when an alias names an unreadable
    /// file (requested-but-invalid) or two aliases disagree.
    pub fn path_or_skip(&self) -> Option<PathBuf> {
        match self.env_override() {
            Err(conflict) => panic!("{conflict}"),
            Ok(Some(resolved)) => {
                assert!(
                    resolved.path.is_file(),
                    "fixture {}: {} names {} which is not a readable file",
                    self.id,
                    match resolved.source {
                        FixtureSource::Env(name) => name,
                        FixtureSource::Default => "default",
                    },
                    resolved.path.display()
                );
                Some(resolved.path)
            }
            Ok(None) => {
                let default = Path::new(self.default_path);
                default.is_file().then(|| default.to_path_buf())
            }
        }
    }

    /// Resolve or fail with a message naming the fixture and every alias
    /// that could have supplied it.
    pub fn required(&self) -> PathBuf {
        self.path_or_skip().unwrap_or_else(|| {
            panic!(
                "fixture {} is required: set one of [{}] or place it at {}",
                self.id,
                self.env.join(", "),
                self.default_path
            )
        })
    }
}

/// Whether `QWEN_REQUIRE_METAL_TESTS` demands that Metal-gated tests run.
pub fn metal_tests_required() -> bool {
    std::env::var("QWEN_REQUIRE_METAL_TESTS")
        .map(|value| env_value_truthy(&value))
        .unwrap_or(false)
}

/// A Metal context for a test, or `None` when this machine has no usable
/// device and the test may skip. Any other initialization error, or a
/// missing device when Metal tests are required, is a failure.
pub fn metal_context_or_skip() -> Option<MetalContext> {
    match MetalContext::new() {
        Ok(ctx) => Some(ctx),
        Err(MetalError::EmptyLibrary | MetalError::NoDevice) => {
            assert!(!metal_tests_required(), "Metal is required but unavailable");
            None
        }
        Err(error) => panic!("Metal context: {error}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PROBE: Fixture = Fixture {
        id: "probe",
        env: &["QWEN_TEST_FIXTURE_PROBE_A", "QWEN_TEST_FIXTURE_PROBE_B"],
        default_path: "/definitely/not/present.gguf",
    };

    // Environment mutation is process-global; serialize the tests that touch it.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn with_env<T>(pairs: &[(&str, Option<&str>)], body: impl FnOnce() -> T) -> T {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for (name, value) in pairs {
            // SAFETY: guarded by ENV_LOCK; tests in this module are the only
            // readers of these probe names.
            unsafe {
                match value {
                    Some(value) => std::env::set_var(name, value),
                    None => std::env::remove_var(name),
                }
            }
        }
        let result = body();
        for (name, _) in pairs {
            unsafe { std::env::remove_var(name) };
        }
        result
    }

    #[test]
    fn absent_everywhere_is_a_skip() {
        with_env(
            &[
                ("QWEN_TEST_FIXTURE_PROBE_A", None),
                ("QWEN_TEST_FIXTURE_PROBE_B", None),
            ],
            || assert_eq!(PROBE.path_or_skip(), None),
        );
    }

    #[test]
    fn first_alias_wins_and_agreeing_aliases_are_fine() {
        let file = std::env::temp_dir().join("qwen-test-fixture-probe.gguf");
        std::fs::write(&file, b"x").unwrap();
        let path = file.to_str().unwrap();
        with_env(
            &[
                ("QWEN_TEST_FIXTURE_PROBE_A", Some(path)),
                ("QWEN_TEST_FIXTURE_PROBE_B", Some(path)),
            ],
            || {
                let resolved = PROBE.env_override().unwrap().unwrap();
                assert_eq!(
                    resolved.source,
                    FixtureSource::Env("QWEN_TEST_FIXTURE_PROBE_A")
                );
                assert_eq!(PROBE.path_or_skip().as_deref(), Some(file.as_path()));
            },
        );
        std::fs::remove_file(&file).ok();
    }

    #[test]
    fn disagreeing_aliases_are_a_conflict() {
        with_env(
            &[
                ("QWEN_TEST_FIXTURE_PROBE_A", Some("/a.gguf")),
                ("QWEN_TEST_FIXTURE_PROBE_B", Some("/b.gguf")),
            ],
            || {
                let err = PROBE.env_override().unwrap_err();
                assert!(err.contains("conflicts with"), "{err}");
            },
        );
    }

    #[test]
    fn explicitly_requested_but_missing_is_a_failure_not_a_skip() {
        let outcome = with_env(
            &[
                ("QWEN_TEST_FIXTURE_PROBE_A", Some("/nope/never.gguf")),
                ("QWEN_TEST_FIXTURE_PROBE_B", None),
            ],
            || std::panic::catch_unwind(|| PROBE.path_or_skip()),
        );
        assert!(outcome.is_err());
    }

    #[test]
    fn required_message_names_every_alias() {
        let outcome = with_env(
            &[
                ("QWEN_TEST_FIXTURE_PROBE_A", None),
                ("QWEN_TEST_FIXTURE_PROBE_B", None),
            ],
            || std::panic::catch_unwind(|| PROBE.required()),
        );
        let payload = outcome.unwrap_err();
        let message = payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap();
        assert!(
            message.contains("QWEN_TEST_FIXTURE_PROBE_A, QWEN_TEST_FIXTURE_PROBE_B"),
            "{message}"
        );
    }

    #[test]
    fn inventory_ids_are_unique_and_paths_are_gguf() {
        let all = [
            A3B_Q4_K_M,
            QWEN36_27B_Q4_K_M,
            QWEN35_0_8B_F32,
            A10B_Q4_K_XL,
            A3B_Q3_K_M,
            A3B_IQ4_XS,
            MUSE_GLIMMER_Q8_0,
            DEEPSEEK_V4_IQ3_XXS,
            QWEN4EXP_Q3_K_XL,
            DFLASH_DRAFT_36_Q8_0,
        ];
        let mut ids = all.iter().map(|f| f.id).collect::<Vec<_>>();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), all.len());
        assert!(all.iter().all(|f| f.default_path.ends_with(".gguf")));
    }
}
