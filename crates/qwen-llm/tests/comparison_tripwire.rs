use fancy_regex::Regex;
use std::fs;
use std::path::{Path, PathBuf};

const COMPARISON_MODULE: &str = "crates/qwen-llm/src/compare.rs";
const EXCEPTION_MARKER: &str = "comparison-tripwire:";

// Temporary deferrals owned by the GLM Metal lane; remove each entry when it migrates.
const TEMPORARY_GLM5_METAL_EXCEPTIONS: &[&str] = &[
    "crates/qwen-llm/src/glm5_next_metal/tests.rs",
    "crates/qwen-llm/src/glm5_next_metal/packed.rs",
    "crates/qwen-llm/src/glm5_next_metal/tests/intervention_gates.rs",
];

#[derive(Debug, PartialEq, Eq)]
struct Finding {
    path: String,
    line: usize,
}

fn unchecked_abs_max_regex() -> Regex {
    Regex::new(r"(?s)\.abs\(\)(?:(?!;).)*?\.fold\((?:(?!;).)*?(?:f32::max|f64::max|max_by\s*\()")
        .expect("comparison tripwire regex")
}

fn scan_source(path: &str, source: &str, pattern: &Regex) -> Vec<Finding> {
    if path == COMPARISON_MODULE || TEMPORARY_GLM5_METAL_EXCEPTIONS.contains(&path) {
        return Vec::new();
    }
    pattern
        .find_iter(source)
        .filter_map(Result::ok)
        .filter_map(|matched| {
            let fold_offset = source[matched.start()..matched.end()].find(".fold(")?;
            let fold_line_start = source[..matched.start() + fold_offset]
                .rfind('\n')
                .map_or(0, |offset| offset + 1);
            let fold_line_end = source[matched.start() + fold_offset..]
                .find('\n')
                .map_or(source.len(), |offset| {
                    matched.start() + fold_offset + offset
                });
            if source[fold_line_start..fold_line_end].contains(EXCEPTION_MARKER) {
                return None;
            }
            let line = source[..matched.start()]
                .bytes()
                .filter(|&byte| byte == b'\n')
                .count()
                + 1;
            Some(Finding {
                path: path.to_string(),
                line,
            })
        })
        .collect()
}

fn scan_tree(root: &Path, directory: &Path, pattern: &Regex, findings: &mut Vec<Finding>) {
    let entries = fs::read_dir(directory).expect("read source directory");
    for entry in entries {
        let path = entry.expect("read source entry").path();
        if path.is_dir() {
            scan_tree(root, &path, pattern, findings);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            let source = fs::read_to_string(&path).expect("read Rust source");
            let relative = path
                .strip_prefix(root)
                .expect("source path beneath workspace")
                .to_string_lossy()
                .replace('\\', "/");
            findings.extend(scan_source(&relative, &source, pattern));
        }
    }
}

fn scan_text(path: &str, source: &str) -> Vec<Finding> {
    scan_source(path, source, &unchecked_abs_max_regex())
}

#[test]
fn workspace_has_no_unchecked_absolute_difference_max_folds() {
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let workspace = workspace.canonicalize().expect("workspace root");
    let mut findings = Vec::new();
    scan_tree(
        &workspace,
        &workspace.join("crates"),
        &unchecked_abs_max_regex(),
        &mut findings,
    );
    assert!(
        findings.is_empty(),
        "unchecked absolute-difference max folds remain; use crate::compare::assert_max_abs_diff_f32/f64 for tests or qwen_llm::compare::report_max_abs_diff_f32/f64 for reporters:\n{}",
        findings
            .iter()
            .map(|finding| format!("{}:{}", finding.path, finding.line))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn tripwire_names_a_reintroduced_multiline_site() {
    let sample = format!(
        "fn regression() {{\n  let error = actual.iter().zip(expected)\n    .map(|(a, b)| (a - b).{}())\n    .{}(0.0f32, f32::max);\n}}\n",
        "abs", "fold"
    );
    assert_eq!(
        scan_text("crates/example.rs", &sample),
        vec![Finding {
            path: "crates/example.rs".to_string(),
            line: 3,
        }]
    );
}

#[test]
fn marked_magnitude_and_temporary_glm_sites_are_explicit_exceptions() {
    assert!(scan_text(
        "crates/example.rs",
        "let peak = values.iter().map(f32::abs).fold(0.0, f32::max); // comparison-tripwire: magnitude only\n"
    )
    .is_empty());
    let deferred = format!(
        "a.iter().zip(b).map(|(x, y)| (x - y).{}()).{}(0.0, f32::max);\n",
        "abs", "fold"
    );
    assert!(scan_text("crates/qwen-llm/src/glm5_next_metal/tests.rs", &deferred).is_empty());
}
