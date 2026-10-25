#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///
"""Guard production Rust against unreviewed artifact-identity gates.

This is a source-text heuristic, not a Rust parser. It strips comments and
inline cfg(test) modules while retaining string literals for the checks.
"""

from __future__ import annotations

import pathlib
import re
import sys
import bisect
from dataclasses import dataclass


ROOT = pathlib.Path(__file__).resolve().parents[2]
ALLOWLIST = ROOT / "scripts/reference/identity_gates.allow"
CATEGORIES = {
    "disk_integrity",
    "durable_binding",
    "lens_binding",
    "provenance",
    "frozen_experiment",
    "renderer_format",
    "temporary",
}
RAW_STRING_START = re.compile(r"(?:br|r)(#{0,})\"")


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    token: str
    kind: str
    matched: str


def blank_non_code(text: str, *, keep_strings: bool) -> str:
    """Blank Rust comments and optionally literals, preserving offsets/newlines."""
    out = list(text)
    i = 0
    n = len(text)
    while i < n:
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            for k in range(i, j):
                out[k] = " "
            i = j
            continue
        if text.startswith("/*", i):
            depth = 1
            j = i + 2
            while j < n and depth:
                if text.startswith("/*", j):
                    depth += 1
                    j += 2
                elif text.startswith("*/", j):
                    depth -= 1
                    j += 2
                else:
                    j += 1
            for k in range(i, j):
                if text[k] != "\n":
                    out[k] = " "
            i = j
            continue
        # Raw strings: r###"..."### and byte raw strings br###"..."###.
        raw = RAW_STRING_START.match(text, i)
        if raw:
            hashes = raw.group(1)
            start = i
            end_marker = '"' + hashes
            end = text.find(end_marker, i + raw.end())
            j = n if end < 0 else end + len(end_marker)
            if not keep_strings:
                for k in range(start, j):
                    if text[k] != "\n":
                        out[k] = " "
            i = j
            continue
        # Normal string and character literals.
        if text[i] == "'":
            # Rust lifetimes are not quoted strings. A character literal is
            # short, including one escaped codepoint.
            close = text.find("'", i + 1, min(n, i + 16))
            interior = text[i + 1 : close] if close >= 0 else ""
            is_char = bool(
                close >= 0
                and "\n" not in interior
                and (
                    (len(interior) == 1 and interior != "\\")
                    or re.fullmatch(
                        r"\\(?:['\"\\nrt0]|x[0-9a-fA-F]{2}|u\{[0-9a-fA-F_]+\})",
                        interior,
                    )
                )
            )
            if not is_char:
                i += 1
                continue
        if text[i] in ('"', "'"):
            quote = text[i]
            start = i
            j = i + 1
            while j < n:
                if text[j] == "\\":
                    j += 2
                elif text[j] == quote:
                    j += 1
                    break
                else:
                    j += 1
            if not keep_strings:
                for k in range(start, min(j, n)):
                    if text[k] != "\n":
                        out[k] = " "
            i = j
            continue
        i += 1
    return "".join(out)


def strip_inline_test_modules(text: str) -> str:
    code = blank_non_code(text, keep_strings=False)
    pattern = re.compile(
        r"#\s*\[\s*cfg\s*\(\s*test\s*\)\s*\]\s*mod\s+[A-Za-z_][A-Za-z_0-9]*\s*\{"
    )
    ranges: list[tuple[int, int]] = []
    for match in pattern.finditer(code):
        brace = code.find("{", match.start(), match.end())
        line_start = code.rfind("\n", 0, match.start()) + 1
        indent = re.match(r"[ \t]*", code[line_start : match.start()]).group(0)
        # Rustfmt keeps a module's final brace at its declaration indentation.
        # This handles brace characters in Rust macro strings without needing
        # a full lexer; retain counting for compact synthetic modules.
        closing = None
        pos = text.find("\n", brace)
        if pos >= 0:
            for line_match in re.finditer(r"(?m)^([ \t]*)\}[^\n]*$", text[pos + 1 :]):
                if line_match.group(1) == indent:
                    closing = pos + 1 + line_match.end()
                    break
        if closing is not None:
            ranges.append((match.start(), closing))
            continue
        depth = 1
        pos = brace + 1
        while pos < len(code) and depth:
            if code[pos] == "{":
                depth += 1
            elif code[pos] == "}":
                depth -= 1
            pos += 1
        if depth == 0:
            ranges.append((match.start(), pos))
    chars = list(text)
    for start, end in ranges:
        for pos in range(start, end):
            if chars[pos] != "\n":
                chars[pos] = " "
    return "".join(chars)


def rust_files() -> list[pathlib.Path]:
    files = []
    for base in sorted((ROOT / "crates").glob("*/src")):
        all_rs = sorted(base.rglob("*.rs"))
        test_only_external: set[pathlib.Path] = set()
        declaration = re.compile(
            r"#\s*\[\s*cfg\s*\(\s*test\s*\)\s*\]\s*"
            r"(?:#\s*\[\s*path\s*=\s*\"([^\"]+)\"\s*\]\s*)?"
            r"mod\s+([A-Za-z_][A-Za-z_0-9]*)\s*;"
        )
        for parent in all_rs:
            source = blank_non_code(parent.read_text(), keep_strings=True)
            for match in declaration.finditer(source):
                child = (
                    (parent.parent / match.group(1))
                    if match.group(1)
                    else (parent.parent / f"{match.group(2)}.rs")
                )
                if child.is_file():
                    test_only_external.add(child.resolve())
                elif not match.group(1):
                    nested = parent.parent / match.group(2) / "mod.rs"
                    if nested.is_file():
                        test_only_external.add(nested.resolve())
        for path in all_rs:
            if path.name in {"tests.rs"} or path.name.endswith("_tests.rs"):
                continue
            if any(
                part == "tests" or part.endswith("_tests")
                for part in path.relative_to(base).parts
            ):
                continue
            if path.resolve() in test_only_external:
                continue
            files.append(path)
    return files


def file_level_review_module(path: str) -> bool:
    """Research/evidence/lens files whose whole content may be reviewed once."""
    candidate = pathlib.PurePosixPath(path)
    name = candidate.name
    if name in {
        "dflash_k0s.rs",
        "dsv4_mhc_delete.rs",
        "gguf_arena_floor.rs",
        "grammar_lm_head_row_floor.rs",
        "grammar_oracle.rs",
        "grammar_row_runtime.rs",
        "integrated_grammar_row.rs",
        "prefix_cache_vt_ab.rs",
        "qwen4exp.rs",
        "qwen4exp_moe_observe.rs",
        "deepseek_v4_census.rs",
    }:
        return True
    if name.endswith("_probe.rs") or name.endswith("_packet.rs"):
        return True
    if "full_lens" in candidate.parts or "full_lens" in name:
        return True
    if name.startswith("muse_lens") or name in {"lens.rs", "plain_logit_lens.rs"}:
        return True
    return path == "crates/qwen-llm/src/metal_dflash/tiled_vt_prefill_pilot.rs"


def scan_text(path: str, text: str) -> list[Finding]:
    text = strip_inline_test_modules(text)
    visible = blank_non_code(text, keep_strings=True)
    code_only = blank_non_code(text, keep_strings=False)
    findings: list[Finding] = []
    line_starts = [0] + [match.end() for match in re.finditer("\n", text)]

    def line_at(offset: int) -> int:
        return bisect.bisect_right(line_starts, offset)

    # String literals, including raw strings. Extract contents by matching on
    # original text at offsets that remain identical after comment masking.
    string_re = re.compile(r'(?:br|r)(#{0,})".*?"\1|"(?:\\.|[^"\\])*"', re.S)
    for match in string_re.finditer(visible):
        raw = match.group(0)
        if raw.startswith(("r", "br")):
            quote = raw.find('"')
            content = raw[quote + 1 : -1 - len(raw[:quote].split('"')[0])]
        else:
            content = raw[1:-1]
        line = line_at(match.start())
        # Find long hexadecimal literals; 16-character IDs are flagged only
        # when the containing source line compares them.
        for hex_match in re.finditer(
            r"(?i)(?<![0-9a-f])[0-9a-f]{16,}(?![0-9a-f])", content
        ):
            token = hex_match.group(0)
            if len(token) < 32:
                prefix = visible[max(0, match.start() - 48) : match.start()].rstrip()
                suffix = visible[match.end() : match.end() + 48].lstrip()
                compared = (
                    re.search(r"(?:==|!=|\.eq\s*\(|\.ne\s*\()$", prefix) is not None
                    or re.match(r"(?:==|!=)", suffix) is not None
                    or re.search(r"assert_(?:eq|ne)!\s*\([^\n]*$", prefix) is not None
                )
                if not compared:
                    continue
            literal = f'"{content}"'
            findings.append(Finding(path, line, token, "digest", literal))

        for dev in re.finditer(
            r"(?i)(?:Apple\s+M\d+(?:\s+(?:Max|Pro|Ultra|Studio))?|\bM\d+\s+(?:Max|Pro|Ultra|Studio))\b",
            content,
        ):
            token = dev.group(0)
            findings.append(Finding(path, line, token, "device", f'"{content}"'))

    lines = code_only.splitlines()
    for match in re.finditer(
        r"(?<![\w.])(?:0x[0-9a-fA-F]+|[0-9][0-9_]*)(?![\w.])", code_only
    ):
        raw = match.group(0)
        if raw.lower().startswith("0x"):
            continue
        try:
            value = int(raw.replace("_", ""))
        except ValueError:
            continue
        if value < 1_000_000_000:
            continue
        line = line_at(match.start())
        source_line = lines[line - 1] if line <= len(lines) else ""
        previous_const = code_only.rfind("const ", 0, match.start())
        if previous_const >= 0:
            declaration = code_only[previous_const : match.start()]
            if re.match(
                r"const\s+(?:[A-Za-z_][A-Za-z_0-9]*_)?MULTIPLIERS?\b", declaration
            ):
                continue
        # Skip a literal used by a MAX-style constant and literals in shifts.
        before = code_only[max(0, match.start() - 180) : match.start()]
        if re.search(
            r"\b(?:const|static)\s+[A-Za-z_][A-Za-z_0-9]*MAX[A-Za-z_0-9]*\s*:[^;=]*=",
            before,
            re.I,
        ):
            continue
        after = code_only[match.end() : match.end() + 30]
        if re.match(r"\s*<<", after) or re.search(r"<<\s*$", before[-30:]):
            continue
        findings.append(Finding(path, line, raw, "byte_total", raw))

    call_re = re.compile(
        r"\b((?:verified_)?checkpoint_content_identity[A-Za-z0-9_]*)\s*\("
    )
    for match in call_re.finditer(code_only):
        prefix = code_only[max(0, match.start() - 5) : match.start()]
        if re.search(r"\bfn\s*$", prefix):
            continue
        line = line_at(match.start())
        token = match.group(1)
        findings.append(Finding(path, line, token, "identity_call", token))

    # Same path/token/kind is a single review entry; retain first line.
    unique: dict[tuple[str, str, str], Finding] = {}
    for finding in findings:
        unique.setdefault((finding.path, finding.token, finding.kind), finding)
    return sorted(unique.values(), key=lambda f: (f.line, f.kind, f.token))


def scan_sources() -> list[Finding]:
    findings = []
    for path in rust_files():
        rel = path.relative_to(ROOT).as_posix()
        findings.extend(scan_text(rel, path.read_text()))
    return sorted(findings, key=lambda f: (f.path, f.line, f.kind, f.token))


def parse_allowlist(text: str) -> dict[tuple[str, str], tuple[str, str]]:
    entries = {}
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split("\t", 3)
        if len(fields) != 4:
            raise ValueError(
                f"allowlist line {number}: expected path<TAB>token<TAB>category<TAB>reason"
            )
        path, token, category, reason = fields
        if category not in CATEGORIES:
            raise ValueError(f"allowlist line {number}: invalid category {category!r}")
        if not reason.strip():
            raise ValueError(f"allowlist line {number}: reason is required")
        if category == "temporary" and not reason.lower().startswith("trigger:"):
            raise ValueError(
                f"allowlist line {number}: temporary entry must name a removal trigger"
            )
        if token == "*" and category not in {"lens_binding", "frozen_experiment"}:
            raise ValueError(
                f"allowlist line {number}: file-level entries require lens_binding or frozen_experiment"
            )
        key = (path, token)
        if key in entries:
            raise ValueError(
                f"allowlist line {number}: duplicate entry for {path} {token}"
            )
        entries[key] = (category, reason)
    return entries


def check(findings: list[Finding], allow_text: str) -> tuple[list[str], list[str]]:
    entries = parse_allowlist(allow_text)
    found = {(f.path, f.token) for f in findings}
    matched_entries: set[tuple[str, str]] = set()
    missing = []
    for path, token in found:
        key = (path, token)
        wildcard = (path, "*")
        if key in entries:
            matched_entries.add(key)
        elif wildcard in entries:
            if not file_level_review_module(path):
                missing.append(f"unreviewable file-level entry: {path}: *")
            else:
                matched_entries.add(wildcard)
        else:
            missing.append(f"unallowlisted: {path}: {token}")
    for path, token in entries:
        if token == "*":
            if not file_level_review_module(path):
                missing.append(f"unreviewable file-level entry: {path}: *")
            elif not any(f.path == path for f in findings):
                pass
        elif (path, token) in found:
            matched_entries.add((path, token))
    stale = sorted(set(entries) - matched_entries)
    return missing, [f"stale allowlist entry: {p}: {t}" for p, t in stale]


def self_test() -> None:
    synthetic = """
const DIGEST: &str = "0123456789abcdef0123456789abcdef";
if tokenizer == "51ebd8140ea2abd9" { }
const HUGE: u64 = 1_000_000_000;
const MULTIPLIERS: [u64; 1] = [9_223_372_036_854_775_807];
let product = "Apple M4 Max";
verified_checkpoint_content_identity(source);
#[cfg(test)] mod tests { const TEST: &str = "ffffffffffffffffffffffffffffffff"; }
"""
    findings = scan_text("crates/example/src/lib.rs", synthetic)
    kinds = {f.kind for f in findings}
    assert kinds == {"digest", "byte_total", "device", "identity_call"}, findings
    assert not any(f.token == "9_223_372_036_854_775_807" for f in findings), findings
    assert not any("ffffffff" in f.token for f in findings), findings
    sample = findings[0]
    entry = f"{sample.path}\t{sample.token}\tprovenance\tsynthetic self-test\n"
    errors, stale = check([sample], entry)
    assert not errors and not stale
    errors, _ = check([f for f in findings if f != sample], entry)
    assert errors
    _, stale = check(
        [sample],
        entry + "crates/example/src/old.rs\tdeadbeef\tprovenance\tstale test\n",
    )
    assert stale
    wildcard_hit = Finding(
        "crates/qwen-cli/src/full_lens.rs", 1, "abc", "digest", '"abc"'
    )
    wildcard_entry = "crates/qwen-cli/src/full_lens.rs\t*\tlens_binding\twhole lens module reviewed\n"
    assert check([wildcard_hit], wildcard_entry) == ([], [])
    assert check(
        [Finding("crates/qwen-cli/src/main.rs", 1, "abc", "digest", '"abc"')],
        "crates/qwen-cli/src/main.rs\t*\tfrozen_experiment\tbad wildcard\n",
    )[0]
    test_only_fixture = (
        ROOT / "crates/qwen-llm/src/deepseek_v4_metal/prefill/k160_n128_floor.rs"
    )
    if test_only_fixture.exists():
        assert test_only_fixture not in rust_files()
    registry = ALLOWLIST.read_text()
    for path, token in [
        (
            "crates/qwen-llm/src/qwen4exp_runtime/strict_inject_probe.rs",
            "874537119c68f6c566c4288ba17c1099694416edb001c4003249570894438e97",
        ),
        ("crates/qwen-llm/src/metal/moe/hc4_strict.rs", "1_013_904_223"),
        (
            "crates/qwen-llm/src/deepseek_v41/forward/ced/qualification/provenance.rs",
            "1b2bf6b0384435d7cb551f4df7444beb52da51ed62bd52f3e2626bf0f1c675c5",
        ),
        (
            "crates/qwen-llm/src/deepseek_v41/forward/ced/qualification/archive.rs",
            "84c465db06384a4c87da7a7304a98e9b479c150b",
        ),
        ("crates/qwen-llm/src/deepseek_v41/gguf/resident_storage.rs", "4_294_967_296"),
    ]:
        entries = parse_allowlist(registry)
        assert (path, token) in entries and (path, "*") not in entries
        findings = scan_text(path, (ROOT / path).read_text())
        assert any(f.token == token for f in findings)
        assert not check(findings, registry)[0]
        assert check(
            findings + [Finding(path, 1, "unreviewed", "digest", "test")], registry
        )[0]
    print("identity_gates self-tests: passed")


def main(argv: list[str]) -> int:
    if argv == ["--self-test"]:
        self_test()
        return 0
    if argv not in ([], ["--check"]):
        print("usage: identity_gates.py [--check|--self-test]", file=sys.stderr)
        return 2
    findings = scan_sources()
    try:
        errors, stale = check(findings, ALLOWLIST.read_text())
    except (OSError, ValueError) as error:
        print(f"identity_gates: {error}", file=sys.stderr)
        return 1
    for finding in findings:
        print(f"{finding.path}:{finding.line}: {finding.matched}")
    if errors or stale:
        for error in errors + stale:
            print(f"identity_gates: {error}", file=sys.stderr)
        return 1
    print(
        f"identity_gates: {len(findings)} reviewed findings across {len({f.path for f in findings})} files"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
