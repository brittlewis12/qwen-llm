use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

const COMPARISON_MODULE: &str = "crates/qwen-llm/src/compare.rs";

// The GLM Metal migration belongs to a separate lane. Remove each path when
// that lane migrates its comparisons.
const TEMPORARY_GLM5_METAL_FILES: &[&str] = &[
    "crates/qwen-llm/src/glm5_next_metal/tests.rs",
    "crates/qwen-llm/src/glm5_next_metal/packed.rs",
    "crates/qwen-llm/src/glm5_next_metal/tests/intervention_gates.rs",
];

#[derive(Clone, Copy)]
struct Exception {
    path: &'static str,
    function: &'static str,
    reason: &'static str,
}

// These reductions compute magnitudes for quantization scales or activation
// envelopes; they do not compare two independently produced values.
const MAGNITUDE_EXCEPTIONS: &[Exception] = &[
    Exception {
        path: "crates/qwen-llm/src/deepseek_v4_oracle.rs",
        function: "attention_fp8_nope_bf16_rope_roundtrip_in_place",
        reason: "quantization scale magnitude",
    },
    Exception {
        path: "crates/qwen-llm/src/deepseek_v4_oracle.rs",
        function: "attention_fp8_nope_roundtrip_in_place",
        reason: "quantization scale magnitude",
    },
    Exception {
        path: "crates/qwen-llm/src/deepseek_v4_oracle.rs",
        function: "fp4_activation_roundtrip_in_place",
        reason: "quantization scale magnitude",
    },
    Exception {
        path: "crates/qwen-llm/src/deepseek_v4_oracle.rs",
        function: "pack_indexer_fp4_row",
        reason: "quantization scale magnitude",
    },
    Exception {
        path: "crates/qwen-llm/src/k2_horizon_metal/compact/tests.rs",
        function: "quantize",
        reason: "quantization scale magnitude",
    },
    Exception {
        path: "crates/qwen-llm/src/k2_horizon_metal/compact/tests.rs",
        function: "quantizer_contract_has_exact_ties_canonical_zero_and_visible_failures",
        reason: "quantization error bound scale",
    },
    Exception {
        path: "crates/qwen-llm/src/k2_horizon_metal/compact/tests.rs",
        function: "gpu_q8_store_bytes_and_inline_attention_match_independent_controls",
        reason: "quantization scale magnitude",
    },
    Exception {
        path: "crates/qwen-llm/src/muse_glimmer_lens_fit/tests.rs",
        function: "real_q8_block_51_replay_and_vjp_smoke",
        reason: "expected-gradient magnitude scale",
    },
    Exception {
        path: "crates/qwen-llm/src/metal/moe_grouped_generic/tests.rs",
        function: "assert_close",
        reason: "reference magnitude scale",
    },
    Exception {
        path: "crates/qwen-llm/src/metal/moe_grouped_generic/tests.rs",
        function: "generic_grouped_moe_down_q5_k_matches_specialized",
        reason: "reference magnitude scale",
    },
    Exception {
        path: "crates/qwen-llm/src/metal/moe_grouped_generic/tests.rs",
        function: "generic_grouped_moe_gate_up_q4_k_q5_k_match_specialized",
        reason: "reference magnitude scale",
    },
    Exception {
        path: "crates/qwen-llm/src/metal/mhc.rs",
        function: "mhc4_pre_q8_0_directed_numerics_and_widths",
        reason: "summed absolute activation envelope",
    },
    Exception {
        path: "crates/qwen-llm/src/metal/elementwise.rs",
        function: "scatter_kv_q8_matches_ref_quant",
        reason: "Q8 block quantization scale magnitude",
    },
    Exception {
        path: "crates/qwen-llm/src/metal/kda.rs",
        function: "assert_close",
        reason: "reference magnitude for relative tolerance",
    },
    Exception {
        path: "crates/qwen-llm/src/metal/latent.rs",
        function: "grouped_q8_mat_mat_matches_per_row_gemv_at_glm_shapes",
        reason: "reference row magnitude for relative tolerance",
    },
    Exception {
        path: "crates/qwen-llm/src/metal/latent.rs",
        function: "split_selected_attention_directed_cases_match_an_independent_softmax",
        reason: "reference magnitude for relative tolerance",
    },
    Exception {
        path: "crates/qwen-llm/src/metal/expert.rs",
        function: "assert_moe_oracle_close_loose",
        reason: "reference magnitude for relative tolerance",
    },
    Exception {
        path: "crates/qwen-llm/src/metal/test_support.rs",
        function: "assert_moe_oracle_close",
        reason: "reference magnitude in checked shared assertion helper",
    },
    Exception {
        path: "crates/qwen-llm/src/metal/mat_mat.rs",
        function: "q8_0_fewrow_matches_cpu_reference_at_tile_edges",
        reason: "reference output magnitude for relative tolerance",
    },
    Exception {
        path: "crates/qwen-llm/src/metal/mat_vec.rs",
        function: "mat_vec_trellis3_matches_cpu",
        reason: "reference output magnitude for relative tolerance",
    },
    Exception {
        path: "crates/qwen-llm/src/glm5_next/oracle.rs",
        function: "kda_contract_matches_fla_naive_recurrence",
        reason: "reference output magnitude for relative tolerance",
    },
    Exception {
        path: "crates/qwen-llm/src/muse_tiled_numerical_diagnostic.rs",
        function: "tiled_prefill_numerical_diagnostic",
        reason: "worst-row index selected after checked comparison metrics",
    },
];

#[derive(Debug, PartialEq, Eq)]
struct Finding {
    path: String,
    line: usize,
    function: String,
}

fn mask_range(bytes: &mut [u8], start: usize, end: usize) {
    for byte in &mut bytes[start..end] {
        if *byte != b'\n' && *byte != b'\r' {
            *byte = b' ';
        }
    }
}

fn raw_string_end(bytes: &[u8], start: usize) -> Option<usize> {
    let mut quote = start;
    if bytes.get(quote) == Some(&b'b') {
        quote += 1;
    }
    if bytes.get(quote) != Some(&b'r') {
        return None;
    }
    quote += 1;
    let mut hashes = 0;
    while bytes.get(quote) == Some(&b'#') {
        hashes += 1;
        quote += 1;
    }
    if bytes.get(quote) != Some(&b'"') {
        return None;
    }
    let mut end = quote + 1;
    while end < bytes.len() {
        if bytes[end] == b'"'
            && bytes.get(end + 1..end + 1 + hashes) == Some(&vec![b'#'; hashes][..])
        {
            return Some(end + 1 + hashes);
        }
        end += 1;
    }
    None
}

// Preserve byte offsets and line breaks so findings still point into the
// original file while comments and literals cannot create matches.
fn strip_comments_and_strings(source: &str) -> Result<String, String> {
    let mut bytes = source.as_bytes().to_vec();
    let mut i = 0;
    while i < bytes.len() {
        if bytes.get(i..i + 2) == Some(b"//") {
            let start = i;
            i += 2;
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            mask_range(&mut bytes, start, i);
        } else if bytes.get(i..i + 2) == Some(b"/*") {
            let start = i;
            i += 2;
            let mut depth = 1usize;
            while i < bytes.len() && depth > 0 {
                match bytes.get(i..i + 2) {
                    Some(b"/*") => {
                        depth += 1;
                        i += 2;
                    }
                    Some(b"*/") => {
                        depth -= 1;
                        i += 2;
                    }
                    _ => i += 1,
                }
            }
            if depth != 0 {
                return Err("unterminated block comment".to_string());
            }
            mask_range(&mut bytes, start, i);
        } else if let Some(end) = raw_string_end(&bytes, i) {
            mask_range(&mut bytes, i, end);
            i = end;
        } else if bytes[i] == b'"' || (bytes[i] == b'b' && bytes.get(i + 1) == Some(&b'"')) {
            let start = i;
            if bytes[i] == b'b' {
                i += 1;
            }
            i += 1;
            let mut closed = false;
            while i < bytes.len() {
                match bytes[i] {
                    b'\\' => i = (i + 2).min(bytes.len()),
                    b'"' => {
                        i += 1;
                        closed = true;
                        break;
                    }
                    _ => i += 1,
                }
            }
            if !closed {
                return Err("unterminated string literal".to_string());
            }
            mask_range(&mut bytes, start, i);
        } else if bytes[i] == b'\'' {
            let start = i;
            let mut end = i + 1;
            if bytes.get(end) == Some(&b'\\') {
                end += 2;
            } else {
                end += 1;
            }
            if bytes.get(end) == Some(&b'\'') {
                end += 1;
                mask_range(&mut bytes, start, end);
                i = end;
            } else {
                i += 1;
            }
        } else {
            i += 1;
        }
    }
    String::from_utf8(bytes).map_err(|error| format!("masked source is not UTF-8: {error}"))
}

fn is_ident(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

fn identifier_at(source: &str, index: usize) -> Option<(usize, usize)> {
    let bytes = source.as_bytes();
    if index >= bytes.len() || !is_ident(bytes[index]) {
        return None;
    }
    let mut start = index;
    let mut end = index + 1;
    while start > 0 && is_ident(bytes[start - 1]) {
        start -= 1;
    }
    while end < bytes.len() && is_ident(bytes[end]) {
        end += 1;
    }
    Some((start, end))
}

fn function_at(source: &str, offset: usize) -> String {
    let prefix = &source[..offset];
    let Some(fn_offset) = prefix.rfind("fn ") else {
        return "<module>".to_string();
    };
    let name_start = fn_offset + 3;
    let Some((start, end)) = identifier_at(source, name_start) else {
        return "<module>".to_string();
    };
    if start != name_start {
        return "<module>".to_string();
    }
    source[start..end].to_string()
}

fn call_end(source: &str, open: usize) -> Result<usize, String> {
    let bytes = source.as_bytes();
    let mut depth = 0usize;
    for (index, byte) in bytes.iter().enumerate().skip(open) {
        match byte {
            b'(' => depth += 1,
            b')' => {
                depth = depth
                    .checked_sub(1)
                    .ok_or_else(|| "unbalanced close parenthesis".to_string())?;
                if depth == 0 {
                    return Ok(index + 1);
                }
            }
            _ => {}
        }
    }
    Err("unterminated comparison reduction call".to_string())
}

fn contains_abs(source: &str) -> bool {
    source.contains(".abs(") || source.contains("map(f32::abs)") || source.contains("map(f64::abs)")
}

fn precomputed_difference_names(source: &str) -> HashSet<String> {
    let mut names = HashSet::new();
    for statement in source.split(';') {
        if !contains_abs(statement) || !statement.contains('-') {
            continue;
        }
        let Some(let_offset) = statement.rfind("let ") else {
            continue;
        };
        let name_start = let_offset + 4;
        let Some((start, end)) = identifier_at(statement, name_start) else {
            continue;
        };
        if start == name_start {
            names.insert(statement[start..end].to_string());
        }
    }
    names
}

fn is_reduction(name: &str) -> bool {
    matches!(name, "fold" | "reduce" | "max_by")
}

fn has_maximum_operator(source: &str) -> bool {
    source.contains("f32::max")
        || source.contains("f64::max")
        || source.contains(".max(")
        || source.contains("max_by")
}

fn scan_unfiltered(path: &str, source: &str) -> Result<Vec<Finding>, String> {
    let source = strip_comments_and_strings(source)?;
    let names = precomputed_difference_names(&source);
    let bytes = source.as_bytes();
    let mut findings = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'.' {
            i += 1;
            continue;
        }
        let Some((start, end)) = identifier_at(&source, i + 1) else {
            i += 1;
            continue;
        };
        let name = &source[start..end];
        if !is_reduction(name) || bytes.get(end) != Some(&b'(') {
            i = end;
            continue;
        }
        let call_close = call_end(&source, end)?;
        let statement_start = source[..start]
            .rfind([';', '{', '}'])
            .map_or(0, |boundary| boundary + 1);
        let context = &source[statement_start..call_close];
        let call = &source[start..call_close];
        let has_precomputed = names.iter().any(|name| {
            context.match_indices(name).any(|(at, _)| {
                let before = context.as_bytes().get(at.wrapping_sub(1)).copied();
                let after = context.as_bytes().get(at + name.len()).copied();
                !before.is_some_and(is_ident) && !after.is_some_and(is_ident)
            })
        });
        let is_abs_comparison =
            has_maximum_operator(call) && (contains_abs(context) || contains_abs(call));
        if is_abs_comparison || (has_precomputed && has_maximum_operator(call)) {
            let line = source[..start]
                .bytes()
                .filter(|byte| *byte == b'\n')
                .count()
                + 1;
            findings.push(Finding {
                path: path.to_string(),
                line,
                function: function_at(&source, start),
            });
        }
        i = call_close;
    }
    Ok(findings)
}

fn scan_source(path: &str, source: &str) -> Result<Vec<Finding>, String> {
    if path == COMPARISON_MODULE || TEMPORARY_GLM5_METAL_FILES.contains(&path) {
        return Ok(Vec::new());
    }
    let findings = scan_unfiltered(path, source)?;
    Ok(findings
        .into_iter()
        .filter(|finding| {
            !MAGNITUDE_EXCEPTIONS.iter().any(|exception| {
                exception.path == path
                    && exception.function == finding.function
                    && !exception.reason.is_empty()
            })
        })
        .collect())
}

fn scan_tree(root: &Path, directory: &Path, findings: &mut Vec<Finding>) -> Result<(), String> {
    let entries = fs::read_dir(directory)
        .map_err(|error| format!("read {}: {error}", directory.display()))?;
    for entry in entries {
        let entry =
            entry.map_err(|error| format!("read entry in {}: {error}", directory.display()))?;
        let path = entry.path();
        if path.is_dir() {
            scan_tree(root, &path, findings)?;
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            let source = fs::read_to_string(&path)
                .map_err(|error| format!("read {}: {error}", path.display()))?;
            let relative = path
                .strip_prefix(root)
                .map_err(|error| {
                    format!(
                        "source path {} is outside {}: {error}",
                        path.display(),
                        root.display()
                    )
                })?
                .to_str()
                .ok_or_else(|| format!("non-UTF-8 source path: {}", path.display()))?
                .replace('\\', "/");
            findings.extend(scan_source(&relative, &source)?);
        }
    }
    Ok(())
}

fn scan_text(path: &str, source: &str) -> Vec<Finding> {
    scan_source(path, source).expect("scanner fixture must be valid")
}

#[test]
fn workspace_has_no_unchecked_absolute_difference_max_reductions() {
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let workspace = workspace.canonicalize().expect("workspace root");
    let mut findings = Vec::new();
    scan_tree(&workspace, &workspace.join("crates"), &mut findings)
        .unwrap_or_else(|error| panic!("comparison tripwire scan failed closed: {error}"));
    assert!(
        findings.is_empty(),
        "unchecked absolute-difference max reductions remain; use crate::compare::assert_max_abs_diff_f32/f64 for tests or qwen_llm::compare::report_max_abs_diff_f32/f64 for reporters:\n{}",
        findings
            .iter()
            .map(|finding| format!("{}:{}", finding.path, finding.line))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn tripwire_detects_pairwise_and_precomputed_difference_variants() {
    let variants = [
        "fn sample() { let delta = a.iter().zip(b).map(|(x,y)| (x-y).abs()).fold(0.0, f32::max); }",
        "fn sample() { let delta = a.iter().zip(b).map(|(x,y)| (x-y).abs()).reduce(f64::max); }",
        "fn sample() { let delta = a.iter().zip(b).map(|(x,y)| (x-y).abs()).max_by(|a,b| a.total_cmp(b)); }",
        "fn sample() { let delta = a.iter().zip(b).max_by(|(x,y),(u,v)| (x-y).abs().total_cmp(&(u-v).abs())); }",
        "fn sample() { let delta = a.iter().zip(b).map(|(x,y)| (x-y).abs()).fold(0.0, |m,d| m.max(d)); }",
        "fn sample() { let delta = a.iter().zip(b).map(|(x,y)| (x-y).abs()).reduce(|m,d| m.max(d)); }",
        "fn sample() { let delta = a.iter().zip(b).map(f32::abs).fold(0.0, f32::max); }",
        "fn sample() { let delta = a.iter().zip(b).map(f64::abs).reduce(f64::max); }",
        "fn sample() { let differences = a.iter().zip(b).map(|(x,y)| (x-y).abs()).collect::<Vec<_>>(); let delta = differences.iter().fold(0.0, f32::max); }",
    ];
    for sample in variants {
        assert_eq!(
            scan_text("crates/example.rs", sample).len(),
            1,
            "missed: {sample}"
        );
    }
}

#[test]
fn tripwire_ignores_commented_and_quoted_examples() {
    let source = r##"
        // a.iter().zip(b).map(|(x,y)| (x-y).abs()).fold(0.0, f32::max);
        /* a.iter().zip(b).map(|(x,y)| (x-y).abs()).reduce(f64::max); */
        let text = ".abs().reduce(f64::max)";
        let raw = r#".abs().max_by(f32::max)"#;
    "##;
    assert!(scan_text("crates/example.rs", source).is_empty());
}

#[test]
fn magnitude_fixture_is_detected_and_only_its_named_exception_is_allowed() {
    let magnitude =
        "fn quantize() { let scale = values.iter().map(f32::abs).fold(0.0, f32::max); }";
    let raw = scan_unfiltered(
        "crates/qwen-llm/src/k2_horizon_metal/compact/tests.rs",
        magnitude,
    )
    .unwrap();
    assert_eq!(raw.len(), 1, "magnitude fixture must exercise the scanner");
    assert!(
        scan_text(
            "crates/qwen-llm/src/k2_horizon_metal/compact/tests.rs",
            magnitude
        )
        .is_empty()
    );
    assert_eq!(scan_text("crates/example.rs", magnitude).len(), 1);
    let deferred = "fn old() { a.iter().zip(b).map(|(x,y)| (x-y).abs()).fold(0.0, f32::max); }";
    assert!(scan_text("crates/qwen-llm/src/glm5_next_metal/tests.rs", deferred).is_empty());
}

#[test]
fn scanner_errors_are_returned_instead_of_discarded() {
    assert!(scan_unfiltered("crates/example.rs", "/* unterminated").is_err());
    assert!(scan_unfiltered("crates/example.rs", "let text = \"unterminated").is_err());
}
