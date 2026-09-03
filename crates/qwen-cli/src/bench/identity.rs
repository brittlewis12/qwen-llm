//! Build identity capture and validation for bench rows.

use super::*;

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BuildIdentity {
    pub(crate) schema_version: u32,
    pub(crate) build_commit: String,
    pub(crate) build_commit_short: String,
    pub(crate) build_dirty: Option<bool>,
    pub(crate) build_source_state: Option<String>,
    pub(crate) stamp_source: String,
    pub(crate) stamp_error: Option<String>,
    pub(crate) runtime_commit: Option<String>,
    pub(crate) runtime_dirty: Option<bool>,
    pub(crate) runtime_source_state: Option<String>,
    pub(crate) status: String,
    pub(crate) problems: Vec<String>,
    pub(crate) overrides: Vec<String>,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct BuildIdentityPolicy {
    pub(crate) allow_dirty: bool,
    pub(crate) allow_unverifiable: bool,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct RuntimeGitIdentity {
    pub(crate) commit: Option<String>,
    pub(crate) dirty: Option<bool>,
    pub(crate) source_state: Option<String>,
}

pub(crate) static BUILD_IDENTITY: OnceLock<BuildIdentity> = OnceLock::new();

pub(crate) static BUILD_IDENTITY_POLICY: OnceLock<BuildIdentityPolicy> = OnceLock::new();

pub(crate) fn runtime_git_identity() -> RuntimeGitIdentity {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
    let commit = source_identity::git_text(repo, &["rev-parse", "HEAD"])
        .map(|value| value.to_ascii_lowercase())
        .filter(|value| source_identity::full_object_id(value));
    RuntimeGitIdentity {
        commit,
        dirty: source_identity::git_dirty(repo),
        source_state: source_identity::tracked_source_state(repo),
    }
}

pub(crate) fn classify_build_identity(
    build_commit: &str,
    build_dirty: Option<bool>,
    build_source_state: Option<&str>,
    stamp_source: &str,
    stamp_error: Option<&str>,
    runtime: RuntimeGitIdentity,
) -> BuildIdentity {
    let normalized_build = build_commit.to_ascii_lowercase();
    let mut problems = Vec::new();
    if !source_identity::full_object_id(&normalized_build) {
        problems.push("build_commit_unknown".to_string());
    }
    if build_dirty.is_none() {
        problems.push("build_dirty_unknown".to_string());
    }
    let normalized_build_state = build_source_state
        .filter(|state| source_identity::valid_source_state(state))
        .map(str::to_ascii_lowercase);
    if normalized_build_state.is_none() {
        problems.push("build_source_state_unknown".to_string());
    }
    if let Some(error) = stamp_error.filter(|error| *error != "none") {
        problems.push(error.to_string());
    }
    if runtime.commit.is_none() {
        problems.push("runtime_commit_unknown".to_string());
    }
    if runtime.dirty.is_none() {
        problems.push("runtime_dirty_unknown".to_string());
    }
    if runtime.source_state.is_none() {
        problems.push("runtime_source_state_unknown".to_string());
    }
    if source_identity::full_object_id(&normalized_build)
        && let Some(runtime_commit) = runtime.commit.as_deref()
        && normalized_build != runtime_commit
    {
        problems.push("commit_mismatch".to_string());
    }
    if let (Some(build_dirty), Some(runtime_dirty)) = (build_dirty, runtime.dirty)
        && build_dirty != runtime_dirty
    {
        problems.push("dirty_state_mismatch".to_string());
    }
    if let (Some(build_state), Some(runtime_state)) = (
        normalized_build_state.as_deref(),
        runtime.source_state.as_deref(),
    ) && build_state != runtime_state
    {
        problems.push("source_state_mismatch".to_string());
    }
    if build_dirty == Some(true) || runtime.dirty == Some(true) {
        problems.push("dirty".to_string());
    }

    let status = if problems
        .iter()
        .any(|problem| problem.ends_with("_mismatch"))
    {
        "mismatch"
    } else if problems.iter().any(|p| p != "dirty") {
        "unverifiable"
    } else if problems.iter().any(|p| p == "dirty") {
        "dirty"
    } else {
        "match"
    };
    let short = if source_identity::full_object_id(&normalized_build) {
        normalized_build[..9].to_string()
    } else {
        "unknown".to_string()
    };

    BuildIdentity {
        schema_version: 2,
        build_commit: normalized_build,
        build_commit_short: short,
        build_dirty,
        build_source_state: normalized_build_state,
        stamp_source: stamp_source.to_string(),
        stamp_error: stamp_error
            .filter(|error| *error != "none")
            .map(str::to_string),
        runtime_commit: runtime.commit,
        runtime_dirty: runtime.dirty,
        runtime_source_state: runtime.source_state,
        status: status.to_string(),
        problems,
        overrides: Vec::new(),
    }
}

pub(crate) fn qwen_build_identity_packet() -> &'static BuildIdentity {
    BUILD_IDENTITY.get_or_init(|| {
        let build_dirty = match env!("QWEN_BUILD_DIRTY") {
            "0" => Some(false),
            "1" => Some(true),
            _ => None,
        };
        classify_build_identity(
            env!("QWEN_BUILD_COMMIT"),
            build_dirty,
            Some(env!("QWEN_BUILD_SOURCE_STATE")),
            env!("QWEN_BUILD_STAMP_SOURCE"),
            Some(env!("QWEN_BUILD_STAMP_ERROR")),
            runtime_git_identity(),
        )
    })
}

pub(crate) fn validate_build_identity(
    identity: &BuildIdentity,
    policy: BuildIdentityPolicy,
) -> Result<()> {
    if identity.status == "mismatch" {
        return Err(anyhow!(
            "benchmark binary/source identity mismatch ({:?}): binary={} source={}; rebuild qwen-bench from the current checkout",
            identity.problems,
            identity.build_commit,
            identity.runtime_commit.as_deref().unwrap_or("unknown")
        ));
    }
    let unverifiable = identity.status == "unverifiable";
    if unverifiable && !policy.allow_unverifiable {
        return Err(anyhow!(
            "benchmark build identity is unverifiable ({:?}); rebuild in a Git checkout or pass --allow-unverifiable-build for a non-canonical run",
            identity.problems
        ));
    }
    let dirty = identity.problems.iter().any(|p| p == "dirty");
    if dirty && !policy.allow_dirty {
        return Err(anyhow!(
            "benchmark source/build is dirty; commit the changes or pass --allow-dirty for a non-canonical run"
        ));
    }
    Ok(())
}

pub(crate) fn recorded_build_identity() -> BuildIdentity {
    let mut identity = qwen_build_identity_packet().clone();
    let policy = BUILD_IDENTITY_POLICY.get().copied().unwrap_or_default();
    if policy.allow_dirty && identity.problems.iter().any(|p| p == "dirty") {
        identity.overrides.push("allow_dirty".to_string());
    }
    if policy.allow_unverifiable && identity.status == "unverifiable" {
        identity
            .overrides
            .push("allow_unverifiable_build".to_string());
    }
    identity
}

/// Legacy aliases retained for llama-bench-compatible row consumers.
pub(crate) fn qwen_build_identity() -> (&'static str, u8) {
    let identity = qwen_build_identity_packet();
    let dirty =
        u8::from(identity.build_dirty == Some(true) || identity.runtime_dirty == Some(true));
    (env!("QWEN_BUILD_COMMIT_SHORT"), dirty)
}

#[cfg(test)]
mod build_identity_tests {
    use super::*;

    const A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const STATE_A: &str = concat!(
        "git-source-sha256-v2:",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    );
    const STATE_B: &str = concat!(
        "git-source-sha256-v2:",
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    );

    fn identity(
        build_commit: &str,
        build_dirty: Option<bool>,
        runtime_commit: Option<&str>,
        runtime_dirty: Option<bool>,
    ) -> BuildIdentity {
        classify_build_identity(
            build_commit,
            build_dirty,
            Some(STATE_A),
            "test",
            None,
            RuntimeGitIdentity {
                commit: runtime_commit.map(str::to_string),
                dirty: runtime_dirty,
                source_state: Some(STATE_A.to_string()),
            },
        )
    }

    #[test]
    fn clean_matching_identity_passes() {
        let id = identity(A, Some(false), Some(A), Some(false));
        assert_eq!(id.status, "match");
        assert!(validate_build_identity(&id, BuildIdentityPolicy::default()).is_ok());
    }

    #[test]
    fn commit_mismatch_is_never_overridable() {
        let id = identity(A, Some(false), Some(B), Some(false));
        assert_eq!(id.status, "mismatch");
        assert!(
            validate_build_identity(
                &id,
                BuildIdentityPolicy {
                    allow_dirty: true,
                    allow_unverifiable: true,
                },
            )
            .is_err()
        );
    }

    #[test]
    fn dirty_identity_requires_explicit_override() {
        let id = identity(A, Some(true), Some(A), Some(true));
        assert_eq!(id.status, "dirty");
        assert!(validate_build_identity(&id, BuildIdentityPolicy::default()).is_err());
        assert!(
            validate_build_identity(
                &id,
                BuildIdentityPolicy {
                    allow_dirty: true,
                    allow_unverifiable: false,
                },
            )
            .is_ok()
        );
    }

    #[test]
    fn unknown_runtime_identity_requires_unverifiable_override() {
        let id = identity(A, Some(false), None, None);
        assert_eq!(id.status, "unverifiable");
        assert!(validate_build_identity(&id, BuildIdentityPolicy::default()).is_err());
        assert!(
            validate_build_identity(
                &id,
                BuildIdentityPolicy {
                    allow_dirty: false,
                    allow_unverifiable: true,
                },
            )
            .is_ok()
        );
    }

    #[test]
    fn invalid_compile_stamp_is_unverifiable() {
        let id = classify_build_identity(
            "unknown",
            None,
            None,
            "environment-invalid",
            Some("identity_override_triple_required"),
            RuntimeGitIdentity {
                commit: Some(A.to_string()),
                dirty: Some(false),
                source_state: Some(STATE_A.to_string()),
            },
        );
        assert_eq!(id.status, "unverifiable");
        assert!(
            id.problems
                .iter()
                .any(|p| p == "identity_override_triple_required")
        );
    }

    #[test]
    fn dirty_state_disagreement_is_never_overridable() {
        let id = identity(A, Some(false), Some(A), Some(true));
        assert_eq!(id.status, "mismatch");
        assert!(id.problems.iter().any(|p| p == "dirty_state_mismatch"));
        assert!(
            validate_build_identity(
                &id,
                BuildIdentityPolicy {
                    allow_dirty: true,
                    allow_unverifiable: true,
                },
            )
            .is_err()
        );
    }

    #[test]
    fn source_state_disagreement_is_never_overridable() {
        let id = classify_build_identity(
            A,
            Some(true),
            Some(STATE_A),
            "test",
            None,
            RuntimeGitIdentity {
                commit: Some(A.to_string()),
                dirty: Some(true),
                source_state: Some(STATE_B.to_string()),
            },
        );
        assert_eq!(id.status, "mismatch");
        assert!(id.problems.iter().any(|p| p == "source_state_mismatch"));
        assert!(
            validate_build_identity(
                &id,
                BuildIdentityPolicy {
                    allow_dirty: true,
                    allow_unverifiable: true,
                },
            )
            .is_err()
        );
    }
}
