//! Cached environment-flag helpers.
//!
//! The engine has grown ~160 `QWEN_*` boolean knobs, and until now each one
//! was a hand-rolled 8-line `OnceLock` block. That pattern has three costs:
//! sheer line count, *invisible polarity* (you must read the `matches!` arm
//! to learn whether a flag is opt-in or opt-out), and drift risk (three
//! files carried three slightly different spellings of the truthy/falsy
//! sets). This module centralizes the parse and makes polarity part of the
//! declaration site.
//!
//! Two polarities, matching the two existing idioms exactly:
//!
//! * **default-off / opt-in** (`env_flag!(default_off ...)`) — the flag is
//!   `true` only when the variable is set to an explicitly truthy value
//!   (`1`, `true`, `TRUE`, `yes`, `YES`). Used for diagnostics, no-op
//!   ablations, and experimental sidecars.
//! * **default-on / opt-out** (`env_flag!(default_on ...)`) — the flag is
//!   `true` unless the variable is set to an explicitly falsy value
//!   (`0`, `false`, `FALSE`, `no`, `NO`). Used for shipped defaults whose
//!   env var exists as a rollback lever.
//!
//! Semantics are latch-on-first-read via `OnceLock`, identical to the
//! hand-rolled blocks this replaces: changing the environment after the
//! first read has no effect for the life of the process. Tests that need
//! per-call variation should use thread-local override shims (see
//! `with_matmat_bf16_bfloat_act_override` in `metal_forward.rs`), not env
//! mutation.
//!
//! New or touched `env_flag!` declarations carry a `///` line inside the
//! invocation stating the setting's observable behavior.
//!
//! NOTE: a value outside both sets (e.g. a typo like `QWEN_FOO=ture`)
//! silently resolves to the default, exactly as before. That's a deliberate
//! bug-compatibility choice for this refactor; a warn-on-unrecognized pass
//! can be layered later without touching call sites.

/// Explicitly truthy values for opt-in flags.
#[inline]
pub fn env_value_truthy(v: &str) -> bool {
    matches!(v, "1" | "true" | "TRUE" | "yes" | "YES")
}

/// Explicitly falsy values for opt-out flags.
#[inline]
pub fn env_value_falsy(v: &str) -> bool {
    matches!(v, "0" | "false" | "FALSE" | "no" | "NO")
}

/// Read + parse an opt-in flag (default off) without caching.
#[inline]
pub fn read_default_off(name: &str) -> bool {
    std::env::var(name).as_deref().is_ok_and(env_value_truthy)
}

/// Read + parse an opt-out flag (default on) without caching.
#[inline]
pub fn read_default_on(name: &str) -> bool {
    !std::env::var(name).as_deref().is_ok_and(env_value_falsy)
}

/// The one authoritative list for computation-skipping attribution settings.
pub const PERF_ABLATION_ENV: &[&str] = &[
    "QWEN_DECODE_GDN_NOOP_ALPHA",
    "QWEN_DECODE_GDN_NOOP_BETA",
    "QWEN_DECODE_GDN_NOOP_FRONT",
    "QWEN_DECODE_GDN_NOOP_OUT",
    "QWEN_DECODE_GDN_NOOP_QKV",
    "QWEN_DECODE_GDN_NOOP_Z",
    "QWEN_DECODE_MOE_NOOP_ROUTE",
    "QWEN_DECODE_MOE_NOOP_ROUTED_GATEUP",
    "QWEN_DECODE_MOE_NOOP_ROUTED_DOWN",
    "QWEN_PREFILL_MOE_GROUPED_ZERO_FILL",
    "QWEN_PREFILL_NOOP_ATTN_BODY",
    "QWEN_PREFILL_NOOP_FFN",
    "QWEN_PREFILL_NOOP_GDN_BODY",
    "QWEN_PREFILL_NOOP_MOE_GROUPED_DOWN",
    "QWEN_PREFILL_NOOP_MOE_GROUPED_REDUCE",
    "QWEN_PREFILL_NOOP_MOE_GROUPED_SWIGLU",
    "QWEN_PREFILL_NOOP_MOE_ROUTED",
    "QWEN_PREFILL_NOOP_MOE_SHARED",
];

/// Const string equality used to keep each `perf_ablation` declaration in sync
/// with `PERF_ABLATION_ENV` at compile time.
pub const fn is_perf_ablation_env(name: &str) -> bool {
    let mut index = 0;
    while index < PERF_ABLATION_ENV.len() {
        if const_str_eq(PERF_ABLATION_ENV[index], name) {
            return true;
        }
        index += 1;
    }
    false
}

const fn const_str_eq(left: &str, right: &str) -> bool {
    let left = left.as_bytes();
    let right = right.as_bytes();
    if left.len() != right.len() {
        return false;
    }
    let mut index = 0;
    while index < left.len() {
        if left[index] != right[index] {
            return false;
        }
        index += 1;
    }
    true
}

/// Return a refusal message for the listed settings present in an injected
/// environment-name iterator. Used by the process check and its unit test.
pub fn perf_ablation_refusal_for<I, S>(set_names: I) -> Option<String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let set_names: std::collections::HashSet<String> = set_names
        .into_iter()
        .map(|name| name.as_ref().to_string())
        .collect();
    let found: Vec<&str> = PERF_ABLATION_ENV
        .iter()
        .copied()
        .filter(|name| set_names.contains(*name))
        .collect();
    (!found.is_empty()).then(|| {
        format!(
            "perf-ablation settings are refused in this product build: {}; rebuild with --features perf-ablations",
            found.join(", ")
        )
    })
}

/// Refuse a product-binary invocation before doing work if any ablation
/// setting is present in its environment.
pub fn check_perf_ablation_environment() -> Result<(), String> {
    #[cfg(feature = "perf-ablations")]
    {
        Ok(())
    }
    #[cfg(not(feature = "perf-ablations"))]
    {
        let names: Vec<String> = std::env::vars_os()
            .filter_map(|(name, _)| name.into_string().ok())
            .collect();
        perf_ablation_refusal_for(names.iter().map(String::as_str)).map_or(Ok(()), Err)
    }
}

#[cfg(feature = "perf-ablations")]
#[inline]
pub fn perf_ablation_enabled(name: &str) -> bool {
    read_default_off(name)
}

#[cfg(not(feature = "perf-ablations"))]
#[inline(always)]
pub const fn perf_ablation_enabled(_name: &str) -> bool {
    false
}

#[cfg(feature = "perf-ablations")]
pub fn any_perf_ablation_enabled_with_prefixes(prefixes: &[&str]) -> bool {
    PERF_ABLATION_ENV
        .iter()
        .filter(|name| prefixes.iter().any(|prefix| name.starts_with(prefix)))
        .any(|name| perf_ablation_enabled(name))
}

#[cfg(feature = "perf-ablations")]
pub fn any_perf_ablation_enabled_with_prefixes_and_suffixes(
    prefixes: &[&str],
    suffixes: &[&str],
) -> bool {
    PERF_ABLATION_ENV
        .iter()
        .filter(|name| {
            prefixes.iter().any(|prefix| name.starts_with(prefix))
                && suffixes.iter().any(|suffix| name.ends_with(suffix))
        })
        .any(|name| perf_ablation_enabled(name))
}

#[cfg(not(feature = "perf-ablations"))]
#[inline(always)]
pub const fn any_perf_ablation_enabled_with_prefixes(_prefixes: &[&str]) -> bool {
    false
}

#[cfg(not(feature = "perf-ablations"))]
#[inline(always)]
pub const fn any_perf_ablation_enabled_with_prefixes_and_suffixes(
    _prefixes: &[&str],
    _suffixes: &[&str],
) -> bool {
    false
}

/// Declare a cached env-flag accessor function. New declarations should carry
/// a doc line inside the invocation stating the setting's observable behavior.
///
/// ```ignore
/// env_flag!(default_on  concurrent_gdn_dense_decode_enabled, "QWEN_DECODE_DENSE_CONCURRENT_GDN");
/// env_flag!(perf_ablation $fn_name, $env);
/// ```
///
/// Expands to a `fn $name() -> bool` backed by a private `OnceLock<bool>`,
/// byte-for-byte equivalent to the hand-rolled blocks it replaces. The
/// polarity keyword is load-bearing documentation: it is now impossible to
/// read a flag declaration without learning its default.
#[macro_export]
macro_rules! env_flag {
    ($(#[$meta:meta])* default_on $fn_name:ident, $env:literal) => {
        $(#[$meta])*
        fn $fn_name() -> bool {
            static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ENABLED.get_or_init(|| $crate::env_flag::read_default_on($env))
        }
    };
    ($(#[$meta:meta])* default_off $fn_name:ident, $env:literal) => {
        $(#[$meta])*
        fn $fn_name() -> bool {
            static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ENABLED.get_or_init(|| $crate::env_flag::read_default_off($env))
        }
    };
    ($(#[$meta:meta])* perf_ablation $fn_name:ident, $env:literal) => {
        const _: () = assert!($crate::env_flag::is_perf_ablation_env($env));
        $(#[$meta])*
        #[inline(always)]
        #[cfg(feature = "perf-ablations")]
        fn $fn_name() -> bool {
            static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ENABLED.get_or_init(|| $crate::env_flag::perf_ablation_enabled($env))
        }
        $(#[$meta])*
        #[inline(always)]
        #[cfg(not(feature = "perf-ablations"))]
        const fn $fn_name() -> bool {
            false
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    // No attributes at all: the original invocation syntax.
    crate::env_flag!(default_off un_attributed_flag, "QWEN_TEST_UNATTRIBUTED_FLAG");
    crate::env_flag!(
        /// Returns false while its test environment variable is unset.
        #[allow(dead_code)]
        default_off attributed_flag,
        "QWEN_TEST_ATTRIBUTED_FLAG"
    );
    #[test]
    fn startup_check_refuses_listed_variables_from_injected_environment() {
        let decode_route = PERF_ABLATION_ENV[6];
        let prefill_ffn = PERF_ABLATION_ENV[11];
        let error = perf_ablation_refusal_for([prefill_ffn, "PATH", decode_route]).unwrap();
        assert!(error.contains(decode_route));
        assert!(error.contains(prefill_ffn));
        assert!(error.contains("--features perf-ablations"));
        assert!(!error.contains("PATH"));
        assert!(perf_ablation_refusal_for(["PATH"]).is_none());
    }

    #[test]
    fn truthy_and_falsy_sets_are_disjoint_and_exact() {
        for v in ["1", "true", "TRUE", "yes", "YES"] {
            assert!(env_value_truthy(v), "{v} must be truthy");
            assert!(!env_value_falsy(v), "{v} must not be falsy");
        }
        for v in ["0", "false", "FALSE", "no", "NO"] {
            assert!(env_value_falsy(v), "{v} must be falsy");
            assert!(!env_value_truthy(v), "{v} must not be truthy");
        }
        // Unrecognized values resolve to the default in both polarities —
        // the historical (bug-compatible) behavior.
        for v in ["ture", "on", "off", "", "2"] {
            assert!(!env_value_truthy(v));
            assert!(!env_value_falsy(v));
        }
    }

    #[test]
    fn unset_var_resolves_to_declared_default() {
        // Use names that can't collide with real knobs.
        assert!(!read_default_off("QWEN_TEST_FLAG_THAT_IS_NEVER_SET"));
        assert!(read_default_on("QWEN_TEST_FLAG_THAT_IS_NEVER_SET"));
    }

    #[test]
    fn env_flag_attributed_accessor_behaves_like_unattributed_accessor() {
        assert_eq!(attributed_flag(), un_attributed_flag());
        assert!(!attributed_flag());
    }
}
