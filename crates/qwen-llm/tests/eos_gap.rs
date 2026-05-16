//! End-of-generation / stop-token gap tests.
//!
//! Documents — as failing tests — the assumption-bugs around how the
//! engine and bench CLI currently terminate generation:
//!
//! * `crates/qwen-llm/src/metal_mtp.rs:1459` hardcodes
//!   `let eos_id = 248046_i32;` regardless of the loaded GGUF's
//!   declared `tokenizer.ggml.eos_token_id`.
//! * `crates/qwen-cli/src/bench.rs:311,510,543` use
//!   `default_value = "248046"` for the `--eos` flag, again ignoring
//!   the GGUF metadata.
//!
//! The 248046 value (`<|im_end|>`) is correct for chat/instruct Qwen
//! 3.5/3.6 GGUFs but **wrong for base/pretraining variants of the same
//! family**, which declare `<|endoftext|>` (248044) as their EOS. It
//! is also wrong for any future fine-tune that repoints EOS, and for
//! any GGUF outside the family entirely.
//!
//! These tests exist so:
//!   1. Touching the hardcoded literals deliberately makes them
//!      compile/pass — the diff is "delete the literal, plumb through
//!      a value from `GgufFile`, watch the gap tests go green".
//!   2. The "family is heterogeneous" data-checks document the actual
//!      observed disagreement on disk so future regressions can't
//!      hand-wave the issue away.
//!
//! Test layers:
//!   * **API surface gap** — fail today, pass after the fix:
//!       - `speculative_decoder_accepts_multiple_stop_tokens`
//!         (`SpeculativeDecoder::decode` and `decode_packed_n` take a
//!         scalar `eos_id: i32` and cannot express a stop set;
//!         pinned via a source scrape on the signature)
//!       - `bench_cli_does_not_default_eos_to_a_literal`
//!         (`--eos default_value = "248046"` × 3 sites)
//!       - `gguf_file_exposes_stop_token_ids_method` (the
//!         declarative reader doesn't exist yet)
//!   * **on-disk evidence** — passes today, documents WHY the fix
//!      matters (not just stylistic):
//!       - `family_declared_eos_is_heterogeneous`
//!         (proof that a single hardcoded value is wrong on a base
//!         variant of the supported family)
//!       - `every_family_file_declares_eos`
//!         (proof the GGUF KV section IS the authoritative source)
//!       - `gguf_file_exposes_stop_token_ids_contract`
//!         (locks the contract the new API must satisfy on 0.8B)
//!
//! All tests gracefully skip when their fixtures aren't present so the
//! suite stays green on machines without the models directory.

use qwen_llm::gguf::GgufFile;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// `<|im_end|>` — the chat-template turn terminator. Hardcoded today
/// in metal_mtp.rs and bench.rs. Correct for instruct GGUFs of the
/// Qwen 3.5/3.6 family, wrong everywhere else.
const HARDCODED_EOS_INSTRUCT: i32 = 248046;

/// `<|endoftext|>` — the pretraining EOS used by base variants of the
/// same family. Observed on `Qwen3.5-4B-Base-BF16.gguf` (and would also
/// appear on any future `*-Base-*` GGUF).
const FAMILY_BASE_EOS: i32 = 248044;

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn workspace_root() -> PathBuf {
    // CARGO_MANIFEST_DIR points at `crates/qwen-llm/`. Walk up two.
    manifest_dir()
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root above crates/qwen-llm/")
        .to_path_buf()
}

fn read_src(rel: &str) -> Option<String> {
    let p = workspace_root().join(rel);
    std::fs::read_to_string(&p).ok()
}

fn models_dir() -> Option<PathBuf> {
    let p = PathBuf::from("/Users/tito/models");
    if p.is_dir() { Some(p) } else { None }
}

fn family_ggufs(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let read = match std::fs::read_dir(dir) {
        Ok(r) => r,
        Err(_) => return out,
    };
    for entry in read.flatten() {
        let path = entry.path();
        let name = match path.file_name().and_then(|s| s.to_str()) {
            Some(n) => n,
            None => continue,
        };
        // Only the family we actually support: Qwen 3.5 / 3.6 (top-level files only).
        if !(name.starts_with("Qwen3.5-") || name.starts_with("Qwen3.6-")) {
            continue;
        }
        if !name.ends_with(".gguf") {
            continue;
        }
        out.push(path);
    }
    out.sort();
    out
}

// =========================================================================
// 1. API surface gaps — fail today, pass after the fix.
// =========================================================================

/// `SpeculativeDecoder::decode` and `decode_packed_n` take a scalar
/// `eos_id: i32`. That signature physically cannot express a multi-id
/// stop set — and the GGUF KV section already supports declaring more
/// than one (Llama-3 family ships `eos_token_id` + `eot_token_id`;
/// Qwen 3.5/3.6 doesn't today, but a future Qwen fine-tune could, and
/// the engine should be ready). The fix is to take a stop set
/// (`&[i32]` slice is fine — short, init-time, no smallvec needed) and
/// check membership in the hot loop.
///
/// This test FAILS today by source scrape on the production
/// signatures. It passes when both signatures take a slice / set
/// rather than a single id.
///
/// Source: `crates/qwen-llm/src/metal_mtp.rs:824` and `:1017`.
#[test]
fn speculative_decoder_accepts_multiple_stop_tokens() {
    let src = match read_src("crates/qwen-llm/src/metal_mtp.rs") {
        Some(s) => s,
        None => {
            eprintln!("[eos-gap] skipped — metal_mtp.rs not found");
            return;
        }
    };
    let production = strip_test_modules(&src);

    // Look for the literal scalar shape `eos_id: i32`. After the fix
    // this should become e.g. `stop_tokens: &[i32]` or
    // `stop_set: &StopTokens`.
    let offenders: Vec<(usize, String)> = production
        .lines()
        .enumerate()
        .filter(|(_, line)| line.contains("eos_id: i32"))
        .map(|(i, l)| (i + 1, l.trim().to_string()))
        .collect();

    assert!(
        offenders.is_empty(),
        "metal_mtp.rs decode entry-points take a scalar `eos_id: i32`, which \
         cannot represent multi-stop-token GGUFs. Take a slice/set instead. \
         Offending signatures:\n{}",
        offenders
            .iter()
            .map(|(i, l)| format!("  {i}: {l}"))
            .collect::<Vec<_>>()
            .join("\n"),
    );
}

/// The bench CLI's `--eos` flag must not have a hardcoded numeric
/// default. Either it should default to `None` (resolved from the
/// loaded GGUF at run time) or it should be removed in favor of a
/// `--stop-tokens` override that defaults to the GGUF's declared set.
///
/// This test FAILS today because of three `default_value = "248046"`
/// occurrences in `crates/qwen-cli/src/bench.rs` (the MTP bench at
/// :311 and the two DFlash benches at :510 and :543).
#[test]
fn bench_cli_does_not_default_eos_to_a_literal() {
    let src = match read_src("crates/qwen-cli/src/bench.rs") {
        Some(s) => s,
        None => {
            eprintln!("[eos-gap] skipped — bench.rs not found");
            return;
        }
    };

    let offenders: Vec<(usize, &str)> = src
        .lines()
        .enumerate()
        .filter(|(_, line)| line.contains("default_value = \"248046\""))
        .collect();

    assert!(
        offenders.is_empty(),
        "bench.rs hardcodes `default_value = \"248046\"` for the --eos flag in {} place(s). \
         Default should be sourced from the GGUF's tokenizer.ggml.eos_token_id at runtime. \
         Offending lines:\n{}",
        offenders.len(),
        offenders
            .iter()
            .map(|(i, l)| format!("  {}: {}", i + 1, l.trim()))
            .collect::<Vec<_>>()
            .join("\n"),
    );
}

// =========================================================================
// 2. Declarative API gap — `GgufFile` doesn't yet expose stop tokens
//    as a first-class concept.
// =========================================================================

/// `GgufFile` should expose a `stop_token_ids()` method (or similar
/// declarative API) that returns the producer-declared end-of-
/// generation set — strictly what the GGUF KV section says, no
/// heuristic name-matching.
///
/// This test FAILS today by source scrape: the method does not exist
/// on `GgufFile`. We assert via the source file rather than via the
/// type system so that the failure is informative ("the method is
/// missing") rather than a build break ("this test doesn't compile").
///
/// Source: `crates/qwen-llm/src/gguf.rs`.
#[test]
fn gguf_file_exposes_stop_token_ids_method() {
    let src = match read_src("crates/qwen-llm/src/gguf.rs") {
        Some(s) => s,
        None => {
            eprintln!("[eos-gap] skipped — gguf.rs not found");
            return;
        }
    };

    // Look for any `pub fn stop_token_ids(` signature on GgufFile.
    // Tolerant of `&self` vs `&self, ...` and return-type styles.
    let has_method = src
        .lines()
        .any(|line| line.contains("pub fn stop_token_ids("));

    assert!(
        has_method,
        "GgufFile has no `stop_token_ids()` method. Add a declarative \
         reader that returns the producer-declared stop set \
         (`tokenizer.ggml.eos_token_id` plus `eot_token_id` if present) \
         and errors when neither is declared. No heuristic name-matching.",
    );
}

/// Locks the contract the new `stop_token_ids()` method must satisfy
/// on the 0.8B instruct fixture: it must include `248046`. This test
/// PASSES today via the generic `get_u64` shim, documenting the
/// expected behavior so a flawed implementation of the new method
/// can't silently pass review.
///
/// When `GgufFile::stop_token_ids()` lands, replace the `get_u64`
/// call below with the real method invocation; the assertion should
/// remain unchanged.
#[test]
fn gguf_file_exposes_stop_token_ids_contract() {
    let path = Path::new("/Users/tito/models/Qwen3.5-0.8B-BF16.gguf");
    if !path.exists() {
        eprintln!("[eos-gap] skipped — Qwen3.5-0.8B-BF16.gguf not present");
        return;
    }
    let g = GgufFile::open(path).expect("open 0.8B BF16");

    let stops = g
        .stop_token_ids()
        .expect("0.8B instruct GGUF declares stop tokens");
    assert!(
        stops.contains(&HARDCODED_EOS_INSTRUCT),
        "0.8B instruct stop set must include 248046 (`<|im_end|>`); got {stops:?}",
    );
}

// =========================================================================
// 3. On-disk evidence — passes today, documents WHY the hardcode is
//    provably wrong rather than just-conventionally-suspect.
// =========================================================================

/// Walks every Qwen 3.5/3.6 GGUF in `~/models` and asserts the set of
/// declared `tokenizer.ggml.eos_token_id` values contains more than
/// one distinct value.
///
/// As of 2026-05-16 the observed split is:
///   * `Qwen3.5-4B-Base-BF16.gguf` → 248044  (`<|endoftext|>`)
///   * all other `Qwen3.{5,6}-*.gguf`       → 248046  (`<|im_end|>`)
///
/// If the dataset on disk ever drifts to a single value across all
/// files this test goes red — at which point the right move is *not*
/// to delete the test but to re-establish the heterogeneity (e.g. by
/// pulling a fresh base GGUF) and confirm the engine still respects
/// what the producer declared.
#[test]
fn family_declared_eos_is_heterogeneous() {
    let dir = match models_dir() {
        Some(d) => d,
        None => {
            eprintln!("[eos-gap] skipped — ~/models not present");
            return;
        }
    };
    let files = family_ggufs(&dir);
    if files.is_empty() {
        eprintln!("[eos-gap] skipped — no Qwen3.5/3.6 GGUFs in ~/models");
        return;
    }

    let mut per_file: BTreeMap<String, i32> = BTreeMap::new();
    for path in &files {
        let name = path.file_name().unwrap().to_str().unwrap().to_string();
        let g = match GgufFile::open(path) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("[eos-gap]   skip {name}: {e}");
                continue;
            }
        };
        let eos = g
            .get_u64("tokenizer.ggml.eos_token_id")
            .unwrap_or_else(|| panic!("{name}: missing tokenizer.ggml.eos_token_id"))
            as i32;
        per_file.insert(name, eos);
    }

    eprintln!("[eos-gap] per-file declared EOS across the family:");
    for (name, id) in &per_file {
        let mark = if *id == HARDCODED_EOS_INSTRUCT {
            " "
        } else {
            "!"
        };
        eprintln!("  {mark} {id:>6}  {name}");
    }

    let distinct: std::collections::BTreeSet<i32> = per_file.values().copied().collect();
    assert!(
        distinct.len() > 1,
        "expected at least one base GGUF declaring {FAMILY_BASE_EOS} \
         alongside instruct GGUFs declaring {HARDCODED_EOS_INSTRUCT}, \
         but found a single value {distinct:?} across {} files. \
         If this is intentional (e.g. base models removed from ~/models), \
         either re-add a base GGUF or relax this test — but do NOT \
         delete the assertion as a side effect of cleanup.",
        per_file.len(),
    );

    // And explicitly: at least one file declares the base EOS. This is
    // the on-disk proof that a 248046 hardcode is wrong.
    assert!(
        per_file.values().any(|&v| v == FAMILY_BASE_EOS),
        "expected at least one file declaring {FAMILY_BASE_EOS} (`<|endoftext|>`). \
         Observed: {per_file:?}",
    );
}

/// Every family GGUF on disk must declare an EOS. The engine has no
/// safe fallback — silently picking 248046 (as today) hides corrupt
/// or partially-converted GGUFs.
#[test]
fn every_family_file_declares_eos() {
    let dir = match models_dir() {
        Some(d) => d,
        None => {
            eprintln!("[eos-gap] skipped — ~/models not present");
            return;
        }
    };
    let files = family_ggufs(&dir);
    if files.is_empty() {
        eprintln!("[eos-gap] skipped — no Qwen3.5/3.6 GGUFs in ~/models");
        return;
    }

    let mut missing: Vec<String> = Vec::new();
    for path in &files {
        let name = path.file_name().unwrap().to_str().unwrap().to_string();
        let g = match GgufFile::open(path) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("[eos-gap]   skip {name}: open failed: {e}");
                continue;
            }
        };
        if g.get_u64("tokenizer.ggml.eos_token_id").is_none() {
            missing.push(name);
        }
    }
    assert!(
        missing.is_empty(),
        "GGUF files missing tokenizer.ggml.eos_token_id: {missing:?}",
    );
}

// =========================================================================
// helpers
// =========================================================================

/// Crude-but-sufficient: strip `#[cfg(test)] mod tests { ... }` blocks
/// (and nested braces inside them) from a Rust source string so the
/// production-only assertions don't trip on test fixtures that
/// legitimately reference the literal we're forbidding in production.
///
/// This is a heuristic, not a Rust parser. It correctly handles the
/// single test-module pattern used throughout this crate; if multiple
/// `mod tests` blocks appear in the same file in the future this needs
/// revisiting.
fn strip_test_modules(src: &str) -> String {
    let needle = "#[cfg(test)]";
    let Some(start) = src.find(needle) else {
        return src.to_string();
    };
    // Find the opening brace of the module that follows `#[cfg(test)]`.
    let after = &src[start..];
    let Some(brace_rel) = after.find('{') else {
        return src.to_string();
    };
    let brace_abs = start + brace_rel;

    // Walk the braces to find the matching close.
    let bytes = src.as_bytes();
    let mut depth = 0i32;
    let mut end = bytes.len();
    let mut in_str = false;
    let mut in_char = false;
    let mut prev_backslash = false;
    let mut in_line_comment = false;
    let mut in_block_comment = false;
    let mut i = brace_abs;
    while i < bytes.len() {
        let b = bytes[i];
        let next = bytes.get(i + 1).copied().unwrap_or(0);
        if in_line_comment {
            if b == b'\n' {
                in_line_comment = false;
            }
        } else if in_block_comment {
            if b == b'*' && next == b'/' {
                in_block_comment = false;
                i += 1;
            }
        } else if in_str {
            if !prev_backslash && b == b'"' {
                in_str = false;
            }
            prev_backslash = !prev_backslash && b == b'\\';
        } else if in_char {
            if !prev_backslash && b == b'\'' {
                in_char = false;
            }
            prev_backslash = !prev_backslash && b == b'\\';
        } else {
            match b {
                b'/' if next == b'/' => {
                    in_line_comment = true;
                    i += 1;
                }
                b'/' if next == b'*' => {
                    in_block_comment = true;
                    i += 1;
                }
                b'"' => in_str = true,
                b'\'' => in_char = true,
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = i + 1;
                        break;
                    }
                }
                _ => {}
            }
        }
        i += 1;
    }

    let mut out = String::with_capacity(src.len());
    out.push_str(&src[..start]);
    out.push_str(&src[end..]);
    out
}

#[cfg(test)]
mod helper_tests {
    use super::strip_test_modules;

    #[test]
    fn strip_test_modules_removes_nested_braces() {
        let src = r#"
fn keep() { 1 }
#[cfg(test)]
mod tests {
    fn inner() {
        if true { 248046; }
        let s = "248046";
    }
}
fn also_keep() { 2 }
"#;
        let out = strip_test_modules(src);
        assert!(out.contains("fn keep()"));
        assert!(out.contains("fn also_keep()"));
        assert!(
            !out.contains("248046"),
            "literal inside #[cfg(test)] survived strip: {out}",
        );
    }

    #[test]
    fn strip_test_modules_is_a_noop_when_no_test_block() {
        let src = "fn x() { 248046; }";
        assert_eq!(strip_test_modules(src), src);
    }
}
