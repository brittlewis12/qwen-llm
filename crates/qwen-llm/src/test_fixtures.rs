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
//! - *Unavailable* (no alias set, filename not found) is a skip. *Explicitly
//!   requested but unreadable* (an alias set to a missing file) is a failure —
//!   a caller asked for something specific and did not get it.
//! - Aliases are consulted in the order listed; setting two aliases to
//!   different values is a conflict and fails rather than silently picking one.
//! - `QWEN_REQUIRE_FIXTURES` turns unavailable fixtures into failures. Metal
//!   being required (`QWEN_REQUIRE_METAL_TESTS` truthy) turns a
//!   would-be skip into a failure. It never changes what a fixture resolves to.
//!
//! This module is compiled into the library so the crate's integration tests
//! (a separate crate) can reach it; it has no dependencies beyond `std` and
//! the Metal context constructor.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use crate::env_flag::env_value_truthy;
use crate::metal::{MetalContext, MetalError};

crate::env_flag!(
    /// Fail a test when a model fixture is unavailable instead of skipping it.
    default_off require_fixtures, "QWEN_REQUIRE_FIXTURES"
);

/// One local model fixture: a stable id, the environment aliases that may
/// override its location, and its published filename.
#[derive(Clone, Copy, Debug)]
pub struct Fixture {
    pub id: &'static str,
    /// Aliases in precedence order. All historical names are kept.
    pub env: &'static [&'static str],
    pub filename: &'static str,
}

/// Qwen3.6-35B-A3B UD-Q4_K_M (MoE hybrid; the most-referenced fixture).
pub const A3B_Q4_K_M: Fixture = Fixture {
    id: "qwen3.6-35b-a3b-ud-q4_k_m",
    env: &["QWEN_A3B_MOE_MODEL"],
    filename: "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
};

/// Qwen3.6-27B Q4_K_M (dense).
pub const QWEN36_27B_Q4_K_M: Fixture = Fixture {
    id: "qwen3.6-27b-q4_k_m",
    env: &[],
    filename: "Qwen3.6-27B-Q4_K_M.gguf",
};

/// Qwen3.5-0.8B F32 (smallest dense; CPU-oracle comparisons).
pub const QWEN35_0_8B_F32: Fixture = Fixture {
    id: "qwen3.5-0.8b-f32",
    env: &[],
    filename: "Qwen3.5-0.8B.F32.gguf",
};

/// Qwen3.5-122B-A10B UD-Q4_K_XL (large MoE; residency and prefetch tests).
pub const A10B_Q4_K_XL: Fixture = Fixture {
    id: "qwen3.5-122b-a10b-ud-q4_k_xl",
    env: &[],
    filename: "Qwen3.5-122B-A10B-UD-Q4_K_XL-00001-of-00003.gguf",
};

/// Qwen3.5-35B-A3B Q3_K_M (older A3B quant used by MoE kernel tests).
pub const A3B_Q3_K_M: Fixture = Fixture {
    id: "qwen3.5-35b-a3b-q3_k_m",
    env: &["QWEN_A3B_Q3_MODEL"],
    filename: "Qwen3.5-35B-A3B-Q3_K_M.gguf",
};

/// Qwen3.5-35B-A3B UD-IQ4_XS.
pub const A3B_IQ4_XS: Fixture = Fixture {
    id: "qwen3.5-35b-a3b-ud-iq4_xs",
    env: &["QWEN_A3B_UDIQ4XS_MODEL"],
    filename: "Qwen3.5-35B-A3B-UD-IQ4_XS.gguf",
};

/// Muse Glimmer 30B Q8_0.
pub const MUSE_GLIMMER_Q8_0: Fixture = Fixture {
    id: "muse-glimmer-30b-q8_0",
    env: &["MUSE_GLIMMER_Q8_GGUF"],
    filename: "Muse-Glimmer-30B-Q8_0.gguf",
};

/// DeepSeek V4 Flash 0731 UD-IQ3_XXS (first shard).
pub const DEEPSEEK_V4_IQ3_XXS: Fixture = Fixture {
    id: "deepseek-v4-flash-0731-ud-iq3_xxs",
    env: &[],
    filename: "DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf",
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
    filename: "Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf",
};

/// GLM-5.3-Flash UD-IQ3_XXS (first, metadata-only shard of four).
pub const GLM53_FLASH_UD_IQ3_XXS: Fixture = Fixture {
    id: "glm-5.3-flash-ud-iq3_xxs",
    env: &["GLM53_GGUF"],
    filename: "GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00004.gguf",
};

/// Spiritbuun DFlash drafter for Qwen3.6 (v1 drafter convention).
pub const DFLASH_DRAFT_36_Q8_0: Fixture = Fixture {
    id: "dflash-draft-3.6-q8_0",
    env: &[],
    filename: "dflash-draft-3.6-q8_0.gguf",
};

/// Where a fixture came from, for messages and for tests that care.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FixtureSource {
    /// An environment alias named it.
    Env(&'static str),
    /// A directory from `QWEN_MODEL_SEARCH_PATH`.
    SearchPath,
}

#[derive(Clone, Debug)]
pub struct ResolvedFixture {
    pub path: PathBuf,
    pub source: FixtureSource,
}

impl Fixture {
    /// Published filename used to discover this fixture.
    pub const fn filename(&self) -> &'static str {
        self.filename
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

    /// Resolve to a readable path, or skip when the fixture is not present.
    /// Panics when an alias names an unreadable
    /// file (requested-but-invalid) or two aliases disagree.
    pub fn path_or_skip(&self) -> Option<PathBuf> {
        match self.env_override() {
            Err(conflict) => panic!("{conflict}"),
            Ok(Some(resolved)) => {
                if let Err(error) = readable_file(&resolved.path) {
                    panic!(
                        "fixture {}: {} names {} which is not a readable file: {error}",
                        self.id,
                        match resolved.source {
                            FixtureSource::Env(name) => name,
                            FixtureSource::SearchPath => "QWEN_MODEL_SEARCH_PATH",
                        },
                        resolved.path.display()
                    );
                }
                Some(resolved.path)
            }
            Ok(None) => self.search_path().or_else(|| {
                let message = format!(
                    "fixture {} ({}) unavailable; set QWEN_MODEL_SEARCH_PATH or one of [{}]",
                    self.id,
                    self.filename,
                    self.env.join(", ")
                );
                if require_fixtures() {
                    panic!("{message}");
                }
                eprintln!("skipping: {message}");
                None
            }),
        }
    }

    /// Resolve or fail with a message naming the fixture and every alias
    /// that could have supplied it.
    pub fn required(&self) -> &'static PathBuf {
        let path = self.path_or_skip().unwrap_or_else(|| {
            panic!(
                "fixture {} ({}) is required: set one of [{}] or include it in QWEN_MODEL_SEARCH_PATH",
                self.id,
                self.filename,
                self.env.join(", ")
            )
        });
        static REQUIRED_PATHS: OnceLock<
            Mutex<std::collections::HashMap<(&'static str, PathBuf), &'static PathBuf>>,
        > = OnceLock::new();
        let paths = REQUIRED_PATHS.get_or_init(|| Mutex::new(std::collections::HashMap::new()));
        let mut paths = paths
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *paths
            .entry((self.id, path.clone()))
            .or_insert_with(|| Box::leak(Box::new(path)))
    }

    fn search_path(&self) -> Option<PathBuf> {
        // QWEN_MODEL_SEARCH_PATH is a colon-separated list of directories. For example:
        // `QWEN_MODEL_SEARCH_PATH="$MODELS/primary:$MODELS/archive"`.
        let search = std::env::var_os("QWEN_MODEL_SEARCH_PATH")?;
        for root in std::env::split_paths(&search) {
            if let Some(path) = find_filename(&root, self.filename, 3) {
                return Some(path);
            }
        }
        None
    }
}

/// Configured roots in `QWEN_MODEL_SEARCH_PATH` order.
pub fn model_search_roots() -> Vec<PathBuf> {
    std::env::var_os("QWEN_MODEL_SEARCH_PATH")
        .map(|value| std::env::split_paths(&value).collect())
        .unwrap_or_default()
}

/// Search a directory and at most three directory levels below it.
fn find_filename(root: &Path, filename: &str, depth: usize) -> Option<PathBuf> {
    let entries = std::fs::read_dir(root).ok()?;
    let mut paths = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .collect::<Vec<_>>();
    paths.sort();
    for path in &paths {
        if path.file_name().is_some_and(|name| name == filename) && readable_file(path).is_ok() {
            return Some(path.clone());
        }
    }
    if depth == 0 {
        return None;
    }
    paths
        .into_iter()
        .filter(|path| path.is_dir())
        .find_map(|path| find_filename(&path, filename, depth - 1))
}

/// "Readable" means what the loaders need: a regular file that opens.
fn readable_file(path: &Path) -> std::io::Result<()> {
    let metadata = std::fs::metadata(path)?;
    if !metadata.is_file() {
        return Err(std::io::Error::other("not a regular file"));
    }
    std::fs::File::open(path).map(|_| ())
}

/// Whether `QWEN_REQUIRE_METAL_TESTS` demands that Metal-gated tests run.
pub fn metal_tests_required() -> bool {
    std::env::var("QWEN_REQUIRE_METAL_TESTS")
        .map(|value| env_value_truthy(&value))
        .unwrap_or(false)
}

/// A Metal context for a test, or `None` when the GPU is not available to
/// this process (no device, or another qwen process holds the lease —
/// dependent crates' test binaries take the real lease, not the
/// per-process test lease) and the test may skip. Any other initialization
/// error, or an unavailable GPU when Metal tests are required, is a failure.
pub fn metal_context_or_skip() -> Option<MetalContext> {
    match MetalContext::new() {
        Ok(ctx) => Some(ctx),
        Err(error @ (MetalError::EmptyLibrary | MetalError::NoDevice)) => {
            assert!(
                !metal_tests_required(),
                "Metal is required but unavailable: {error}"
            );
            eprintln!("skipping: {error}");
            None
        }
        Err(error @ MetalError::ProcessLease(_)) => {
            assert!(
                !metal_tests_required(),
                "Metal is required but leased: {error}"
            );
            eprintln!("skipping: {error}");
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
        filename: "not-present.gguf",
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
            GLM53_FLASH_UD_IQ3_XXS,
            DFLASH_DRAFT_36_Q8_0,
        ];
        let mut ids = all.iter().map(|f| f.id).collect::<Vec<_>>();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), all.len());
        assert!(all.iter().all(|f| f.filename.ends_with(".gguf")));
    }

    #[test]
    #[ignore = "prints fixture search results without opening model files"]
    fn list_fixture_resolutions_without_loading_models() {
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
            GLM53_FLASH_UD_IQ3_XXS,
            DFLASH_DRAFT_36_Q8_0,
        ];
        for fixture in all {
            match fixture.path_or_skip() {
                Some(path) => eprintln!("fixture {}: {}", fixture.id, path.display()),
                None => eprintln!("fixture {}: unavailable", fixture.id),
            }
        }
    }

    #[test]
    fn searches_filename_to_bounded_depth_in_directory_order() {
        let root = std::env::temp_dir().join(format!("qwen-fixture-search-{}", std::process::id()));
        let file = root.join("one/two/three/probe.gguf");
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        std::fs::write(&file, b"fixture").unwrap();
        let fixture = Fixture {
            id: "search-probe",
            env: &[],
            filename: "probe.gguf",
        };
        with_env(
            &[("QWEN_MODEL_SEARCH_PATH", Some(root.to_str().unwrap()))],
            || {
                assert_eq!(fixture.path_or_skip().as_deref(), Some(file.as_path()));
            },
        );
        std::fs::remove_dir_all(root).unwrap();
    }
}
