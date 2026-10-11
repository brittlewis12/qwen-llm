use std::collections::{HashMap, HashSet};
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
    expression: &'static str,
    reason: &'static str,
}

// This diagnostic selects the greatest differing row after `compare_logits`
// has already validated and measured the whole compared slice.
const COMPARISON_EXCEPTIONS: &[Exception] = &[Exception {
    path: "crates/qwen-llm/src/muse_tiled_numerical_diagnostic.rs",
    function: "tiled_prefill_numerical_diagnostic",
    expression: "online.iter().zip(candidate).enumerate().max_by(|(_,(a,b)),(_,(c,d))|(*a-*b).abs().total_cmp(&(*c-*d).abs()))",
    reason: "worst-row index selected after checked comparison metrics",
}];

#[derive(Debug, PartialEq, Eq)]
struct Finding {
    path: String,
    line: usize,
    function: String,
    expression: String,
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

fn is_operand_end(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b')' | b']' | b'}')
}

fn is_operand_start(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'(' | b'[' | b'&' | b'*')
}

fn has_binary_subtraction(source: &str) -> bool {
    let bytes = source.as_bytes();
    for index in 0..bytes.len() {
        if bytes[index] != b'-'
            || bytes.get(index + 1) == Some(&b'>')
            || bytes.get(index + 1) == Some(&b'=')
        {
            continue;
        }
        let mut before = index;
        while before > 0 && bytes[before - 1].is_ascii_whitespace() {
            before -= 1;
        }
        let mut after = index + 1;
        while after < bytes.len() && bytes[after].is_ascii_whitespace() {
            after += 1;
        }
        if before == 0 || after == bytes.len() || !is_operand_end(bytes[before - 1]) {
            continue;
        }
        if !is_operand_start(bytes[after]) {
            continue;
        }
        let previous = bytes[before - 1];
        if matches!(previous, b'e' | b'E')
            && before >= 2
            && (bytes[before - 2].is_ascii_digit() || bytes[before - 2] == b'.')
        {
            continue;
        }
        return true;
    }
    false
}

fn precomputed_difference_names(
    source: &str,
) -> (
    HashMap<String, HashSet<String>>,
    HashMap<String, HashSet<String>>,
) {
    let mut names = HashMap::<String, HashSet<String>>::new();
    let mut absolute_names = HashMap::<String, HashSet<String>>::new();
    let mut statement_start = 0;
    for (separator, _) in source.match_indices(';') {
        let statement = &source[statement_start..separator];
        if !has_binary_subtraction(statement) {
            statement_start = separator + 1;
            continue;
        }
        let Some(let_offset) = statement.rfind("let ") else {
            statement_start = separator + 1;
            continue;
        };
        let mut name_start = let_offset + 4;
        while statement
            .as_bytes()
            .get(name_start)
            .is_some_and(u8::is_ascii_whitespace)
        {
            name_start += 1;
        }
        if statement[name_start..].starts_with("mut ") {
            name_start += 4;
        }
        while statement
            .as_bytes()
            .get(name_start)
            .is_some_and(u8::is_ascii_whitespace)
        {
            name_start += 1;
        }
        let Some((start, end)) = identifier_at(statement, name_start) else {
            statement_start = separator + 1;
            continue;
        };
        if start == name_start {
            let function = function_at(source, statement_start + start);
            names
                .entry(function.clone())
                .or_default()
                .insert(statement[start..end].to_string());
            if contains_abs(&statement[end..]) {
                absolute_names
                    .entry(function)
                    .or_default()
                    .insert(statement[start..end].to_string());
            }
        }
        statement_start = separator + 1;
    }
    (names, absolute_names)
}

fn name_is_referenced(context: &str, name: &str) -> bool {
    context.match_indices(name).any(|(at, _)| {
        let before = context.as_bytes().get(at.wrapping_sub(1)).copied();
        let after = context.as_bytes().get(at + name.len()).copied();
        let suffix = context[at + name.len()..].trim_start();
        !before.is_some_and(is_ident)
            && before != Some(b'.')
            && before != Some(b':')
            && !after.is_some_and(is_ident)
            && (suffix.starts_with(".iter(")
                || suffix.starts_with(".into_iter(")
                || suffix.starts_with(".as_slice("))
    })
}

fn normalize_expression(source: &str) -> String {
    source
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect()
}

fn expression_start(source: &str, target: usize) -> usize {
    let bytes = source.as_bytes();
    let mut delimiters = Vec::new();
    for &byte in &bytes[..target] {
        match byte {
            b'(' | b'[' | b'{' => delimiters.push(byte),
            b')' | b']' | b'}' => {
                delimiters.pop();
            }
            _ => {}
        }
    }
    let target_depth = delimiters.len();

    delimiters.clear();
    let mut boundary = 0;
    for (index, &byte) in bytes.iter().enumerate().take(target) {
        let depth = delimiters.len();
        if depth == target_depth {
            if matches!(byte, b';' | b',') {
                boundary = index + 1;
            } else if byte == b'=' {
                let previous = index.checked_sub(1).and_then(|i| bytes.get(i));
                let next = bytes.get(index + 1);
                if previous != Some(&b'=')
                    && previous != Some(&b'!')
                    && previous != Some(&b'<')
                    && previous != Some(&b'>')
                    && next != Some(&b'=')
                    && next != Some(&b'>')
                {
                    boundary = index + 1;
                }
            }
        }
        match byte {
            b'(' | b'[' | b'{' => {
                delimiters.push(byte);
                if depth + 1 == target_depth {
                    boundary = index + 1;
                }
            }
            b')' | b']' | b'}' => {
                delimiters.pop();
            }
            _ => {}
        }
    }
    boundary
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
    let (names, absolute_names) = precomputed_difference_names(&source);
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
        let statement_start = expression_start(&source, start);
        let context = &source[statement_start..call_close];
        let call = &source[start..call_close];
        let function = function_at(&source, start);
        let has_precomputed = names
            .get(&function)
            .is_some_and(|names| names.iter().any(|name| name_is_referenced(context, name)));
        let has_precomputed_absolute = absolute_names
            .get(&function)
            .is_some_and(|names| names.iter().any(|name| name_is_referenced(context, name)));
        let has_abs = contains_abs(context) || contains_abs(call);
        let has_comparison_abs = has_precomputed_absolute
            || (has_abs && (has_binary_subtraction(context) || has_precomputed));
        if has_comparison_abs && has_maximum_operator(call) {
            let line = source[..start]
                .bytes()
                .filter(|byte| *byte == b'\n')
                .count()
                + 1;
            findings.push(Finding {
                path: path.to_string(),
                line,
                function,
                expression: normalize_expression(context),
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
            !COMPARISON_EXCEPTIONS.iter().any(|exception| {
                exception.path == path
                    && exception.function == finding.function
                    && exception.expression == finding.expression
                    && !exception.reason.is_empty()
            })
        })
        .collect())
}

fn scan_tree(
    root: &Path,
    directory: &Path,
    findings: &mut Vec<Finding>,
    apply_exceptions: bool,
) -> Result<(), String> {
    let entries = fs::read_dir(directory)
        .map_err(|error| format!("read {}: {error}", directory.display()))?;
    for entry in entries {
        let entry =
            entry.map_err(|error| format!("read entry in {}: {error}", directory.display()))?;
        let path = entry.path();
        if path.is_dir() {
            scan_tree(root, &path, findings, apply_exceptions)?;
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
            if apply_exceptions {
                findings.extend(scan_source(&relative, &source)?);
            } else if relative != COMPARISON_MODULE {
                findings.extend(scan_unfiltered(&relative, &source)?);
            }
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
    scan_tree(&workspace, &workspace.join("crates"), &mut findings, true)
        .unwrap_or_else(|error| panic!("comparison tripwire scan failed closed: {error}"));
    assert!(
        findings.is_empty(),
        "unchecked max reductions over two-sided absolute differences remain; use crate::compare::assert_max_abs_diff_f32/f64 for tests or qwen_llm::compare::report_max_abs_diff_f32/f64 for reporters:\n{}",
        findings
            .iter()
            .map(|finding| format!("{}:{} ({})", finding.path, finding.line, finding.expression))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn every_expression_exception_matches_a_current_finding() {
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let workspace = workspace.canonicalize().expect("workspace root");
    let mut findings = Vec::new();
    scan_tree(&workspace, &workspace.join("crates"), &mut findings, false)
        .unwrap_or_else(|error| panic!("comparison tripwire scan failed closed: {error}"));
    for exception in COMPARISON_EXCEPTIONS {
        assert!(
            findings.iter().any(|finding| {
                finding.path == exception.path
                    && finding.function == exception.function
                    && finding.expression == exception.expression
            }),
            "stale comparison exception {}::{} ({})",
            exception.path,
            exception.function,
            exception.reason
        );
    }
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
    assert!(
        scan_unfiltered("crates/example.rs", magnitude)
            .unwrap()
            .is_empty()
    );
    let deviation =
        "fn sample() { let peak = values.iter().map(|x| (*x - mean).abs()).fold(0.0, f32::max); }";
    assert_eq!(scan_text("crates/example.rs", deviation).len(), 1);
    let return_type_only = "fn sample() { let peak = values.iter().map(|value| -> f32 { value.abs() }).fold(0.0, f32::max); }";
    assert!(scan_text("crates/example.rs", return_type_only).is_empty());
    let deferred = "fn old() { a.iter().zip(b).map(|(x,y)| (x-y).abs()).fold(0.0, f32::max); }";
    assert!(scan_text("crates/qwen-llm/src/glm5_next_metal/tests.rs", deferred).is_empty());
}

#[test]
fn block_closures_and_mutable_precomputed_differences_are_detected() {
    let block_closure = "fn sample() { let peak = actual.iter().zip(expected).map(|(x, y)| { (x - y).abs() }).fold(0.0, f32::max); }";
    assert_eq!(scan_text("crates/example.rs", block_closure).len(), 1);

    let mutable_precomputed = "fn sample() { let mut differences = actual.iter().zip(expected).map(|(a, b)| (a - b).abs()).collect::<Vec<_>>(); let peak = differences.iter().fold(0.0, f32::max); }";
    assert_eq!(scan_text("crates/example.rs", mutable_precomputed).len(), 1);

    let mapped_precomputed = "fn sample() { let mut differences = actual.iter().zip(expected).map(|(a, b)| a - b).collect::<Vec<_>>(); let peak = differences.iter().map(f32::abs).fold(0.0, f32::max); }";
    assert_eq!(scan_text("crates/example.rs", mapped_precomputed).len(), 1);
}

#[test]
fn an_additional_comparison_in_an_exempt_function_is_not_exempted() {
    let source = "fn tiled_prefill_numerical_diagnostic() { let index = online.iter().zip(candidate).enumerate().max_by(|(_, (a, b)), (_, (c, d))| (*a - *b).abs().total_cmp(&(*c - *d).abs())); let peak = actual.iter().zip(expected).map(|(a,b)| (a-b).abs()).fold(0.0, f32::max); }";
    let findings = scan_text(
        "crates/qwen-llm/src/muse_tiled_numerical_diagnostic.rs",
        source,
    );
    assert_eq!(findings.len(), 1);
    assert!(
        findings[0]
            .expression
            .contains("actual.iter().zip(expected)")
    );
}

#[test]
fn scanner_errors_are_returned_instead_of_discarded() {
    assert!(scan_unfiltered("crates/example.rs", "/* unterminated").is_err());
    assert!(scan_unfiltered("crates/example.rs", "let text = \"unterminated").is_err());
}
