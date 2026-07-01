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

/// Declare a cached env-flag accessor function.
///
/// ```ignore
/// env_flag!(default_on  concurrent_gdn_dense_decode_enabled, "QWEN_DECODE_DENSE_CONCURRENT_GDN");
/// env_flag!(default_off decode_gdn_noop_front_enabled,       "QWEN_DECODE_GDN_NOOP_FRONT");
/// ```
///
/// Expands to a `fn $name() -> bool` backed by a private `OnceLock<bool>`,
/// byte-for-byte equivalent to the hand-rolled blocks it replaces. The
/// polarity keyword is load-bearing documentation: it is now impossible to
/// read a flag declaration without learning its default.
#[macro_export]
macro_rules! env_flag {
    (default_on $fn_name:ident, $env:literal) => {
        fn $fn_name() -> bool {
            static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ENABLED.get_or_init(|| $crate::env_flag::read_default_on($env))
        }
    };
    (default_off $fn_name:ident, $env:literal) => {
        fn $fn_name() -> bool {
            static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ENABLED.get_or_init(|| $crate::env_flag::read_default_off($env))
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
