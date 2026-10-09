#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Generate the reference of environment knobs read by the Rust engine."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
# These belong to the host, toolchain, or logging setup rather than the engine.
# Keep exclusions explicit so new environment reads are visible by default.
EXCLUDED_VARIABLES = {
    "HOME",
    "PATH",
    "TMPDIR",
    "TERM",
    "NO_COLOR",
    "RUST_LOG",
    "CARGO",
    "OUT_DIR",
}
EXCLUDED_PREFIXES = ("MTL_", "CARGO_")
FAMILIES = (
    "deepseek_v4",
    "qwen4exp",
    "glm",
    "k2",
    "muse",
    "dflash",
    "serve",
    "lens",
    "bench",
    "metal",
    "cli",
)


@dataclass
class Knob:
    variable: str
    kind: str
    path: str
    line: int
    description: str
    families: set[str] = field(default_factory=set)
    polarities: set[str] = field(default_factory=set)
    scopes: set[str] = field(default_factory=set)


def line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def strip_comments(text: str) -> str:
    """Blank comments while preserving offsets and line numbers."""
    out: list[str] = []
    i = 0
    block_depth = 0
    while i < len(text):
        if block_depth:
            if text.startswith("/*", i):
                block_depth += 1
                out.extend("  ")
                i += 2
            elif text.startswith("*/", i):
                block_depth -= 1
                out.extend("  ")
                i += 2
            elif text[i] == "\n":
                out.append("\n")
                i += 1
            else:
                out.append(" ")
                i += 1
        elif text.startswith("//", i):
            while i < len(text) and text[i] != "\n":
                out.append(" ")
                i += 1
        elif text.startswith("/*", i):
            block_depth = 1
            out.extend("  ")
            i += 2
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


@dataclass(frozen=True)
class RustToken:
    kind: str
    value: str
    offset: int


def rust_tokens(text: str) -> list[RustToken]:
    """Tokenize enough Rust syntax to skip comments/strings and balance scopes."""
    tokens: list[RustToken] = []
    i = 0
    while i < len(text):
        if text[i].isspace():
            i += 1
        elif text.startswith("//", i):
            end = text.find("\n", i)
            i = len(text) if end < 0 else end + 1
        elif text.startswith("/*", i):
            depth = 1
            i += 2
            while i < len(text) and depth:
                if text.startswith("/*", i):
                    depth += 1
                    i += 2
                elif text.startswith("*/", i):
                    depth -= 1
                    i += 2
                else:
                    i += 1
        elif text[i] == '"':
            start = i
            i += 1
            value: list[str] = []
            while i < len(text):
                if text[i] == "\\" and i + 1 < len(text):
                    value.append(text[i + 1])
                    i += 2
                elif text[i] == '"':
                    i += 1
                    break
                else:
                    value.append(text[i])
                    i += 1
            tokens.append(RustToken("string", "".join(value), start))
        elif text.startswith('r"', i) or re.match(r"r#*\"", text[i:]):
            start = i
            match = re.match(r"r(#+)?\"", text[i:])
            assert match
            hashes = match.group(1) or ""
            i += len(match.group(0))
            close = '"' + hashes
            end = text.find(close, i)
            value = text[i:] if end < 0 else text[i:end]
            i = len(text) if end < 0 else end + len(close)
            tokens.append(RustToken("string", value, start))
        elif text[i].isalpha() or text[i] == "_":
            start = i
            i += 1
            while i < len(text) and (text[i].isalnum() or text[i] == "_"):
                i += 1
            tokens.append(RustToken("ident", text[start:i], start))
        elif text.startswith("::", i):
            tokens.append(RustToken("punct", "::", i))
            i += 2
        else:
            tokens.append(RustToken("punct", text[i], i))
            i += 1
    return tokens


def matching(
    tokens: list[RustToken], start: int, opening: str, closing: str
) -> int | None:
    if start >= len(tokens) or tokens[start].value != opening:
        return None
    depth = 0
    for index in range(start, len(tokens)):
        if tokens[index].value == opening:
            depth += 1
        elif tokens[index].value == closing:
            depth -= 1
            if depth == 0:
                return index
    return None


def split_arguments(tokens: list[RustToken]) -> list[list[RustToken]]:
    parts: list[list[RustToken]] = []
    start = 0
    depths = {"(": 0, "[": 0, "{": 0}
    closes = {")": "(", "]": "[", "}": "{"}
    for index, token in enumerate(tokens):
        value = token.value
        if value in depths:
            depths[value] += 1
        elif value in closes:
            depths[closes[value]] -= 1
        elif value == "," and not any(depths.values()):
            parts.append(tokens[start:index])
            start = index + 1
    if start < len(tokens):
        parts.append(tokens[start:])
    return parts


def parse_env_flag(
    text: str, tokens: list[RustToken], opening: int, closing: int
) -> tuple[str, str, str] | None:
    """Parse the exact env_flag! declaration grammar and its in-call docs."""
    args = split_arguments(tokens[opening + 1 : closing])
    if len(args) != 2:
        return None
    declaration, variable = args
    index = 0
    while index < len(declaration) and declaration[index].value == "#":
        if index + 1 >= len(declaration) or declaration[index + 1].value != "[":
            return None
        attr_end = matching(declaration, index + 1, "[", "]")
        if attr_end is None:
            return None
        index = attr_end + 1
    if (
        index + 1 >= len(declaration)
        or declaration[index].value not in ("default_on", "default_off")
        or declaration[index + 1].kind != "ident"
        or len(declaration) != index + 2
        or len(variable) != 1
        or variable[0].kind != "string"
    ):
        return None
    polarity = declaration[index].value
    docs_start = tokens[opening].offset + 1
    docs_end = declaration[index].offset
    docs = [
        re.sub(r"^\s*/// ?", "", line).strip()
        for line in text[docs_start:docs_end].splitlines()
        if re.match(r"^\s*///", line)
    ]
    purpose = " ".join(part for part in docs if part) or "undocumented"
    return polarity, variable[0].value, purpose


def register_env_flag_declaration(
    declared: dict[str, tuple[str, int, str]],
    variable: str,
    path: str,
    line: int,
    purpose: str,
) -> tuple[bool, tuple[str, str, str, int, str] | None]:
    """Keep the first declaration for each variable and report later ones."""
    if variable in declared:
        first_path, first_line, _first_purpose = declared[variable]
        return False, (
            variable,
            "duplicate env_flag! declaration",
            f"first at {first_path}:{first_line}",
            line,
            path,
        )
    declared[variable] = (path, line, purpose)
    return True, None


def self_test() -> None:
    def parse(source: str) -> tuple[str, str, str] | None:
        tokens = rust_tokens(source)
        start = next(i for i, token in enumerate(tokens) if token.value == "(")
        end = matching(tokens, start, "(", ")")
        assert end is not None
        return parse_env_flag(source, tokens, start, end)

    assert parse(
        'env_flag!(\n    /// Purpose from inside the invocation.\n'
        '    default_off switch, "QWEN_SELF_TEST");'
    ) == ("default_off", "QWEN_SELF_TEST", "Purpose from inside the invocation.")
    assert parse(
        'env_flag!(\n    /// First line.\n    /// Second line.\n'
        '    #[allow(dead_code)]\n    default_on switch, "QWEN_SELF_TEST");'
    ) == ("default_on", "QWEN_SELF_TEST", "First line. Second line.")
    assert parse(
        '/// Adjacent comments are not invocation documentation.\n'
        'env_flag!(\n    #[allow(dead_code)]\n    default_on switch, "QWEN_SELF_TEST");'
    ) == ("default_on", "QWEN_SELF_TEST", "undocumented")
    assert parse('env_flag!(default_on, "QWEN_SELF_TEST");') is None
    assert parse('env_flag!(default_on switch, NAME);') is None

    declared: dict[str, tuple[str, int, str]] = {}
    first, first_conflict = register_env_flag_declaration(
        declared, "QWEN_DUP", "one.rs", 1, "first purpose"
    )
    second, second_conflict = register_env_flag_declaration(
        declared, "QWEN_DUP", "two.rs", 2, "second purpose"
    )
    assert first and first_conflict is None
    assert not second and second_conflict == (
        "QWEN_DUP",
        "duplicate env_flag! declaration",
        "first at one.rs:1",
        2,
        "two.rs",
    )
    assert declared["QWEN_DUP"][2] == "first purpose"
    print("env_knobs scanner self-tests: passed")


def cfg_test_ranges(tokens: list[RustToken]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for index in range(len(tokens) - 4):
        if [token.value for token in tokens[index : index + 5]] != [
            "#",
            "[",
            "cfg",
            "(",
            "test",
        ]:
            continue
        close_attr = matching(tokens, index + 1, "[", "]")
        if close_attr is None:
            continue
        item = close_attr + 1
        while item < len(tokens) and tokens[item].value in (
            "pub",
            "async",
            "unsafe",
            "const",
        ):
            item += 1
        if item >= len(tokens) or tokens[item].value not in (
            "mod",
            "fn",
            "impl",
            "static",
            "const",
            "type",
        ):
            continue
        opening = next(
            (n for n in range(item + 1, len(tokens)) if tokens[n].value in ("{", ";")),
            None,
        )
        if opening is None:
            continue
        if tokens[opening].value == ";":
            continue
        end = matching(tokens, opening, "{", "}")
        if end is not None:
            ranges.append((tokens[opening].offset, tokens[end].offset))
    return ranges


def in_ranges(offset: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start <= offset <= end for start, end in ranges)


def module_base_path(rel: str) -> tuple[str, ...]:
    path = Path(rel)
    parts = list(path.parts)
    if "src" in parts:
        base = parts[parts.index("src") + 1 :]
    elif "tests" in parts:
        base = parts[parts.index("tests") + 1 :]
    else:
        base = parts[1:]
    if base and base[-1].endswith(".rs"):
        filename = Path(base.pop()).stem
        if filename not in ("lib", "main", "mod"):
            base.append(filename)
    else:
        base = [Path(part).stem for part in base]
    return tuple(base)


def module_scopes(tokens: list[RustToken]) -> list[tuple[int, int, str]]:
    scopes: list[tuple[int, int, str]] = []
    for index in range(len(tokens) - 2):
        if tokens[index].value != "mod" or tokens[index + 1].kind != "ident":
            continue
        opening = next(
            (n for n in range(index + 2, len(tokens)) if tokens[n].value in ("{", ";")),
            None,
        )
        if opening is None or tokens[opening].value != "{":
            continue
        closing = matching(tokens, opening, "{", "}")
        if closing is not None:
            scopes.append(
                (
                    tokens[opening].offset,
                    tokens[closing].offset,
                    tokens[index + 1].value,
                )
            )
    return scopes


def path_is_test(rel: str, test_files: set[str]) -> bool:
    return "/tests/" in f"/{rel}" or rel.endswith("/tests.rs") or rel in test_files


def crate_name(rel: str) -> str:
    parts = Path(rel).parts
    package = parts[1] if len(parts) > 1 and parts[0] == "crates" else "unknown"
    return package.replace("-", "_")


def use_paths(tokens: list[RustToken]) -> list[tuple[int, list[str], str | None]]:
    """Return (offset, imported path, local alias) entries from simple Rust uses."""
    result: list[tuple[int, list[str], str | None]] = []

    def expand(prefix: list[str], part: list[RustToken], offset: int) -> None:
        brace = next((n for n, token in enumerate(part) if token.value == "{"), None)
        if brace is not None:
            end = matching(part, brace, "{", "}")
            if end is None:
                return
            base = prefix + [
                token.value for token in part[:brace] if token.kind == "ident"
            ]
            for item in split_arguments(part[brace + 1 : end]):
                expand(base, item, offset)
            return
        alias = None
        if any(token.value == "as" for token in part):
            as_index = next(n for n, token in enumerate(part) if token.value == "as")
            alias = next(
                (
                    token.value
                    for token in part[as_index + 1 :]
                    if token.kind == "ident"
                ),
                None,
            )
            part = part[:as_index]
        names = [
            token.value for token in part if token.kind == "ident" or token.value == "*"
        ]
        result.append((offset, prefix + names, alias))

    for index, token in enumerate(tokens):
        if token.value != "use":
            continue
        end = next(
            (n for n in range(index + 1, len(tokens)) if tokens[n].value == ";"), None
        )
        if end is not None:
            expand([], tokens[index + 1 : end], token.offset)
    return result


def helper_string_parameters(
    tokens: list[RustToken],
) -> list[tuple[str, int, set[int]]]:
    """Return string parameter positions actually passed to env readers."""
    helpers: list[tuple[str, int, set[int]]] = []
    imported_flags = {
        alias or parts[-1]
        for _, parts, alias in use_paths(tokens)
        if len(parts) >= 2
        and parts[-2] == "env_flag"
        and parts[-1] in ("read_default_on", "read_default_off")
    }
    for index, token in enumerate(tokens[:-1]):
        if token.value != "fn" or tokens[index + 1].kind != "ident":
            continue
        name = tokens[index + 1].value
        opening = next(
            (n for n in range(index + 2, len(tokens)) if tokens[n].value in ("(", "{")),
            None,
        )
        if opening is None or tokens[opening].value != "(":
            continue
        close = matching(tokens, opening, "(", ")")
        if close is None:
            continue
        params = split_arguments(tokens[opening + 1 : close])
        string_params: dict[str, int] = {}
        for position, param in enumerate(params):
            colon = next((n for n, item in enumerate(param) if item.value == ":"), None)
            if colon is None or not any(
                item.value == "str" for item in param[colon + 1 :]
            ):
                continue
            param_name = next(
                (item.value for item in param[:colon] if item.kind == "ident"), None
            )
            if param_name:
                string_params[param_name] = position
        body = next(
            (n for n in range(close + 1, len(tokens)) if tokens[n].value in ("{", ";")),
            None,
        )
        if body is None or tokens[body].value != "{":
            continue
        body_end = matching(tokens, body, "{", "}")
        if body_end is None:
            continue
        body_tokens = tokens[body : body_end + 1]
        env_params: set[int] = set()
        for n in range(len(body_tokens) - 4):
            if (
                body_tokens[n].value != "env"
                or body_tokens[n + 1].value != "::"
                or body_tokens[n + 2].value not in ("var", "var_os")
                or body_tokens[n + 3].value != "("
            ):
                continue
            args_end = matching(body_tokens, n + 3, "(", ")")
            if args_end is None:
                continue
            args = split_arguments(body_tokens[n + 4 : args_end])
            if args and args[0] and args[0][0].value in string_params:
                env_params.add(string_params[args[0][0].value])
        # A wrapper can delegate its string parameter to the shared env flag
        # helpers instead of calling std::env directly. Require the helper's
        # actual env_flag module path or a matching `use` path in this file.
        for n, item in enumerate(body_tokens[:-1]):
            if (
                item.value not in ("read_default_on", "read_default_off")
                or body_tokens[n + 1].value != "("
            ):
                continue
            start = n
            while (
                start >= 2
                and body_tokens[start - 1].value == "::"
                and body_tokens[start - 2].kind == "ident"
            ):
                start -= 2
            path = [
                part.value
                for part in body_tokens[start : n + 1]
                if part.kind == "ident"
            ]
            resolved = (
                "env_flag" in path and path[-1] == item.value
            ) or item.value in imported_flags
            if not resolved:
                continue
            close = matching(body_tokens, n + 1, "(", ")")
            args = (
                split_arguments(body_tokens[n + 2 : close]) if close is not None else []
            )
            if args and args[0] and args[0][0].value in string_params:
                env_params.add(string_params[args[0][0].value])
        if env_params:
            helpers.append((name, token.offset, env_params))
    return helpers


def description(lines: list[str], line: int) -> str:
    i = line - 2
    while i >= 0 and not lines[i].strip():
        i -= 1
    if i < 0:
        return "undocumented"
    candidate = lines[i].strip()
    if not candidate.startswith("//"):
        return "undocumented"
    candidate = re.sub(r"^//[/!]?\s?", "", candidate).strip()
    return candidate or "undocumented"


def family(path: str) -> str:
    lower = path.lower()
    for name in FAMILIES:
        if name in lower or (
            name == "deepseek_v4" and ("dsv4" in lower or "deepseek" in lower)
        ):
            return name
    return "cli" if "/qwen-cli/" in path else "metal"


def is_knob(variable: str) -> bool:
    return variable not in EXCLUDED_VARIABLES and not variable.startswith(
        EXCLUDED_PREFIXES
    )


def test_only_location(rel: str, masked: str, offset: int) -> bool:
    """Recognize conventional test files and inline #[cfg(test)] blocks."""
    if "/tests/" in f"/{rel}" or rel.endswith("/tests.rs"):
        return True
    for attribute in re.finditer(r"#\s*\[\s*cfg\s*\(\s*test\s*\)\s*\]", masked):
        opening = masked.find("{", attribute.end())
        if opening < 0 or opening >= offset:
            continue
        depth = 0
        for index in range(opening, len(masked)):
            if masked[index] == "{":
                depth += 1
            elif masked[index] == "}":
                depth -= 1
                if depth == 0:
                    if opening <= offset <= index:
                        return True
                    break
    return False


def collect() -> tuple[dict[str, Knob], list[tuple[str, str, str, int, str]]]:
    knobs: dict[str, Knob] = {}
    conflicts: list[tuple[str, str, str, int, str]] = []
    files = sorted(Path(ROOT / "crates").rglob("*.rs"))
    sources: list[
        tuple[str, str, str, list[str], list[RustToken], list[tuple[int, int]]]
    ] = []

    for source in files:
        rel = source.relative_to(ROOT).as_posix()
        text = source.read_text(encoding="utf-8")
        tokens = rust_tokens(text)
        sources.append(
            (
                rel,
                text,
                strip_comments(text),
                text.splitlines(),
                tokens,
                cfg_test_ranges(tokens),
            )
        )

    module_bases = {rel: module_base_path(rel) for rel, *_ in sources}
    module_ranges = {rel: module_scopes(tokens) for rel, _, _, _, tokens, _ in sources}

    # `#[path = "..."] mod name;` changes the file location, not the Rust
    # module path. Apply those declared paths so sibling imports resolve to
    # their actual module names rather than the directory names on disk.
    for rel, _, _, _, tokens, _ in sources:
        for index in range(len(tokens) - 7):
            if [item.value for item in tokens[index : index + 3]] != ["#", "[", "path"]:
                continue
            close = matching(tokens, index + 1, "[", "]")
            if (
                close is None
                or close + 3 >= len(tokens)
                or tokens[index + 3].value != "="
                or tokens[index + 4].kind != "string"
            ):
                continue
            mod_index = next(
                (
                    n
                    for n in range(close + 1, min(close + 5, len(tokens)))
                    if tokens[n].value == "mod"
                ),
                None,
            )
            if mod_index is None or mod_index + 1 >= len(tokens):
                continue
            target = (Path(rel).parent / tokens[index + 4].value).as_posix()
            parent_module = module_bases[rel] + tuple(
                name
                for start, end, name in module_ranges[rel]
                if start <= tokens[mod_index].offset <= end
            )
            module_bases[target] = parent_module + (tokens[mod_index + 1].value,)

    def module_at(rel: str, offset: int) -> tuple[str, ...]:
        active = [
            name for start, end, name in module_ranges[rel] if start <= offset <= end
        ]
        return module_bases[rel] + tuple(active)

    # Propagate cfg(test) through external module declarations, including
    # descendants of a test-only module whose own declarations have no gate.
    test_files = {
        rel
        for rel, *_ in sources
        if "/tests/" in f"/{rel}" or rel.endswith("/tests.rs")
    }
    source_by_path = {rel: (tokens, ranges) for rel, _, _, _, tokens, ranges in sources}

    def module_children(rel: str, module: str) -> list[str]:
        parent = Path(rel)
        if parent.name == "mod.rs":
            base = parent.parent / module
        elif parent.stem == module and parent.suffix == ".rs":
            return []
        else:
            base = parent.parent / parent.stem / module
        return [
            candidate.as_posix()
            for candidate in (base.with_suffix(".rs"), base / "mod.rs")
            if (ROOT / candidate).exists()
        ]

    changed = True
    while changed:
        changed = False
        for rel, _, _, _, tokens, ranges in sources:
            for index, token in enumerate(tokens[:-2]):
                if (
                    token.value != "mod"
                    or tokens[index + 1].kind != "ident"
                    or tokens[index + 2].value != ";"
                ):
                    continue
                gated = in_ranges(token.offset, ranges)
                for attr in range(max(0, index - 8), index):
                    if [item.value for item in tokens[attr : attr + 5]] != [
                        "#",
                        "[",
                        "cfg",
                        "(",
                        "test",
                    ]:
                        continue
                    close_attr = matching(tokens, attr + 1, "[", "]")
                    if close_attr is not None and close_attr < index:
                        gated = True
                if not gated and rel not in test_files:
                    continue
                for child in module_children(rel, tokens[index + 1].value):
                    if child not in test_files:
                        test_files.add(child)
                        changed = True

    crate_names = {crate_name(rel) for rel, *_ in sources}
    imports: dict[
        tuple[str, tuple[str, ...], str], tuple[str, tuple[str, ...], str | None]
    ] = {}
    glob_imports: dict[
        tuple[str, tuple[str, ...]], list[tuple[str, tuple[str, ...]]]
    ] = {}

    def resolve_module(
        rel: str, current: tuple[str, ...], parts: list[str]
    ) -> tuple[str, tuple[str, ...]]:
        own_crate = crate_name(rel)
        if not parts:
            return own_crate, current
        first, *rest = parts
        if first == "crate":
            return own_crate, tuple(rest)
        if first == "self":
            return own_crate, current + tuple(rest)
        if first == "super":
            return own_crate, current[:-1] + tuple(rest)
        if first in crate_names:
            return first, tuple(rest)
        return own_crate, tuple(parts)

    for rel, _, _, _, tokens, _ in sources:
        own_crate = crate_name(rel)
        for offset, parts, alias in use_paths(tokens):
            current = module_at(rel, offset)
            if not parts:
                continue
            if parts[-1] == "*":
                target = resolve_module(rel, current, parts[:-1])
                glob_imports.setdefault((own_crate, current), []).append(target)
                continue
            target_crate, target_path = resolve_module(rel, current, parts[:-1])
            local_name = alias or parts[-1]
            imports[(own_crate, current, local_name)] = (
                target_crate,
                target_path,
                parts[-1],
            )

    def resolve_symbol(
        rel: str, current: tuple[str, ...], parts: list[str]
    ) -> tuple[str, tuple[str, ...], str]:
        own_crate = crate_name(rel)
        if not parts:
            return own_crate, current, ""
        if len(parts) == 1:
            imported = imports.get((own_crate, current, parts[0]))
            if imported and imported[2] is not None:
                return imported[0], imported[1], imported[2]
            return own_crate, current, parts[0]
        imported = imports.get((own_crate, current, parts[0]))
        if imported and imported[2] is None:
            return imported[0], imported[1] + tuple(parts[1:-1]), parts[-1]
        target_crate, target_module = resolve_module(rel, current, parts[:-1])
        return target_crate, target_module, parts[-1]

    helper_defs: dict[tuple[str, tuple[str, ...], str], set[int]] = {}
    const_defs: dict[tuple[str, tuple[str, ...], str], list[tuple[str, str, int]]] = {}
    for rel, text, _, _, tokens, _ in sources:
        own_crate = crate_name(rel)
        for name, offset, indices in helper_string_parameters(tokens):
            key = (own_crate, module_at(rel, offset), name)
            helper_defs.setdefault(key, set()).update(indices)
        for index in range(len(tokens) - 6):
            if tokens[index].value != "const" or tokens[index + 1].kind != "ident":
                continue
            name = tokens[index + 1].value
            equals = next(
                (
                    n
                    for n in range(index + 2, min(index + 12, len(tokens)))
                    if tokens[n].value in ("=", ";")
                ),
                None,
            )
            if (
                equals is None
                or tokens[equals].value != "="
                or equals + 1 >= len(tokens)
                or tokens[equals + 1].kind != "string"
            ):
                continue
            value = tokens[equals + 1].value
            if re.fullmatch(r"[A-Z][A-Z0-9_]*", value):
                key = (own_crate, module_at(rel, tokens[index].offset), name)
                const_defs.setdefault(key, []).append(
                    (value, rel, line_number(text, tokens[index].offset))
                )

    def find_constant(
        crate: str,
        module: tuple[str, ...],
        name: str,
        seen: set[tuple[str, tuple[str, ...]]],
    ) -> tuple[str, str, int] | None:
        key = (crate, module, name)
        definitions = const_defs.get(key, [])
        if definitions:
            return definitions[-1]
        if (crate, module) in seen:
            return None
        seen.add((crate, module))
        for imported in glob_imports.get((crate, module), []):
            found = find_constant(imported[0], imported[1], name, seen)
            if found:
                return found
        return None

    def find_helper(
        crate: str,
        module: tuple[str, ...],
        name: str,
        seen: set[tuple[str, tuple[str, ...]]],
    ) -> set[int] | None:
        found = helper_defs.get((crate, module, name))
        if found:
            return found
        if (crate, module) in seen:
            return None
        seen.add((crate, module))
        imported = imports.get((crate, module, name))
        if imported and imported[2] is not None:
            found = find_helper(imported[0], imported[1], imported[2], seen)
            if found:
                return found
        for imported in glob_imports.get((crate, module), []):
            found = find_helper(imported[0], imported[1], name, seen)
            if found:
                return found
        return None

    def add(
        variable: str,
        kind: str,
        rel: str,
        line: int,
        lines: list[str],
        polarity: str | None = None,
        scope: str = "runtime",
        purpose: str | None = None,
    ) -> None:
        if not is_knob(variable):
            return
        item = knobs.get(variable)
        desc = purpose if purpose is not None else description(lines, line)
        if item is None or (
            item.description == "undocumented" and desc != "undocumented"
        ):
            if item is None:
                item = Knob(variable, kind, rel, line, desc)
                knobs[variable] = item
        if kind == "build-time" or polarity:
            if polarity and not item.polarities:
                item.path = rel
                item.line = line
                if desc != "undocumented":
                    item.description = desc
            item.kind = kind
        item.families.add(family(rel))
        item.scopes.add(scope)
        if polarity:
            item.polarities.add(polarity)

    declared_flags: dict[str, tuple[str, int, str]] = {}
    for rel, text, _, lines, tokens, test_ranges in sources:
        scope_at = (
            lambda offset: "test-only"
            if path_is_test(rel, test_files) or in_ranges(offset, test_ranges)
            else "runtime"
        )

        def value_for(
            argument: list[RustToken], at: int
        ) -> tuple[str, str, int] | None:
            if not argument:
                return None
            if argument[0].kind == "string":
                return argument[0].value, "value", argument[0].offset
            names = [token.value for token in argument if token.kind == "ident"]
            if not names:
                return None
            current_path = module_at(rel, at)
            target_crate, target_module, name = resolve_symbol(rel, current_path, names)
            definition = find_constant(target_crate, target_module, name, set())
            if definition is None and len(names) == 1:
                imported = imports.get((crate_name(rel), current_path, name))
                if imported and imported[2] is not None:
                    definition = find_constant(
                        imported[0], imported[1], imported[2], set()
                    )
                if definition is None:
                    candidates = [
                        value
                        for (
                            candidate_crate,
                            _,
                            candidate_name,
                        ), values in const_defs.items()
                        if candidate_crate == crate_name(rel) and candidate_name == name
                        for value in values
                    ]
                    if len(candidates) == 1:
                        definition = candidates[0]
            if definition:
                variable, _, offset = definition
                return variable, "value", offset
            return None

        for index, token in enumerate(tokens):
            # env!("NAME") and option_env!("NAME") are compile-time reads.
            if (
                token.value in ("env", "option_env")
                and index + 3 < len(tokens)
                and tokens[index + 1].value == "!"
                and tokens[index + 2].value == "("
            ):
                close = matching(tokens, index + 2, "(", ")")
                if (
                    close is not None
                    and index + 3 < close
                    and tokens[index + 3].kind == "string"
                ):
                    variable = tokens[index + 3].value
                    add(
                        variable,
                        "build-time",
                        rel,
                        line_number(text, token.offset),
                        lines,
                        scope="build",
                    )

            # The env_flag! declaration is the source of polarity and the
            # variable name; its expansion uses the general helper below.
            if (
                token.value == "env_flag"
                and index + 2 < len(tokens)
                and tokens[index + 1].value == "!"
                and tokens[index + 2].value == "("
            ):
                close = matching(tokens, index + 2, "(", ")")
                if close is not None:
                    parsed_flag = parse_env_flag(text, tokens, index + 2, close)
                    if parsed_flag:
                        polarity, variable, purpose = parsed_flag
                        declaration_line = line_number(text, token.offset)
                        first_declaration, duplicate = register_env_flag_declaration(
                            declared_flags,
                            variable,
                            rel,
                            declaration_line,
                            purpose,
                        )
                        if duplicate:
                            conflicts.append(duplicate)
                        if not first_declaration:
                            continue
                        add(
                            variable,
                            "bool " + polarity.replace("_", "-"),
                            rel,
                            declaration_line,
                            lines,
                            polarity,
                            scope_at(token.offset),
                            purpose,
                        )

            # Direct env::var/var_os calls, including fully qualified paths.
            if (
                token.value == "env"
                and index + 3 < len(tokens)
                and tokens[index + 1].value == "::"
                and tokens[index + 2].value in ("var", "var_os")
                and tokens[index + 3].value == "("
            ):
                close = matching(tokens, index + 3, "(", ")")
                if close is not None:
                    args = split_arguments(tokens[index + 4 : close])
                    parsed = value_for(args[0], token.offset) if args else None
                    if parsed:
                        variable, kind, _ = parsed
                        add(
                            variable,
                            kind,
                            rel,
                            line_number(text, token.offset),
                            lines,
                            scope=scope_at(token.offset),
                        )

            # Resolve helper calls locally or through the module's `use` paths.
            if token.kind != "ident" or (index and tokens[index - 1].value == "fn"):
                continue
            opening = index + 1
            if opening >= len(tokens) or tokens[opening].value != "(":
                continue
            close = matching(tokens, opening, "(", ")")
            if close is None:
                continue
            start = index
            while (
                start >= 2
                and tokens[start - 1].value == "::"
                and tokens[start - 2].kind == "ident"
            ):
                start -= 2
            callee = [
                item.value for item in tokens[start : index + 1] if item.kind == "ident"
            ]
            current_path = module_at(rel, token.offset)
            helper_crate, helper_module, helper_name = resolve_symbol(
                rel, current_path, callee
            )
            helper_indices = find_helper(
                helper_crate, helper_module, helper_name, set()
            )
            if helper_indices is None:
                continue
            args = split_arguments(tokens[opening + 1 : close])
            for arg_index in helper_indices:
                if arg_index >= len(args):
                    continue
                parsed = value_for(args[arg_index], token.offset)
                if parsed:
                    variable, kind, _ = parsed
                    add(
                        variable,
                        kind,
                        rel,
                        line_number(text, token.offset),
                        lines,
                        scope=scope_at(token.offset),
                    )

    # Test-fixture aliases live in the `test_fixtures` inventory as
    # `env: &["A", "B"]` arrays rather than literal var_os calls. Each alias
    # is a `fixture path` knob described by the fixture's id.
    for rel, text, masked, lines, _, _ in sources:
        if not rel.endswith("/test_fixtures.rs"):
            continue
        for fixture in re.finditer(
            r"Fixture\s*\{\s*id:\s*\"([^\"]+)\"\s*,\s*env:\s*&\[([^\]]*)\]",
            masked,
            re.DOTALL,
        ):
            fixture_id, aliases = fixture.groups()
            for alias in re.finditer(r"\"([A-Z][A-Z0-9_]*)\"", aliases):
                variable = alias.group(1)
                if not is_knob(variable):
                    continue
                line = line_number(masked, fixture.start(2) + alias.start())
                item = knobs.get(variable)
                desc = f"test fixture alias for `{fixture_id}` (see `test_fixtures`)"
                if item is None:
                    knobs[variable] = Knob(variable, "fixture path", rel, line, desc)
                elif item.description == "undocumented":
                    item.description = desc
                knobs[variable].families.add("test-fixtures")
                knobs[variable].scopes.add("test-only")

    for variable, item in knobs.items():
        if len(item.polarities) > 1:
            conflicts.append(
                (
                    variable,
                    ", ".join(sorted(item.polarities)),
                    ", ".join(sorted(item.families)),
                    item.line,
                    item.path,
                )
            )
    return knobs, conflicts


def render(knobs: dict[str, Knob], revision: str) -> str:
    rows = [
        "# Environment knobs",
        "",
        f"Generated by [`scripts/reference/env_knobs.py`](../scripts/reference/env_knobs.py) at source revision `{revision}`; run `uv run scripts/reference/env_knobs.py` to refresh it. The script scans Rust source text for named `env::var`/`var_os` reads, calls to env-reading helpers, `env_flag!` declarations and `env!`/`option_env!` build-time reads. It is a heuristic scan, not a compiler pass: helper resolution follows direct aliased imports and parent glob imports, but complex re-exports or computed helper paths can still be missed. Test-only scope is inferred from `cfg(test)` gates, so a few rows may be mislabelled. Names computed at runtime, including names found by `std::env::vars()` scans, are not listed. Host, toolchain and logging variables in the script's exclusion constants are omitted.",
        "",
        "| Variable | Kind | Scope | Defining file | Description | Crate/module family |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for variable in sorted(knobs):
        item = knobs[variable]
        rendered_description = item.description
        if item.polarities:
            polarity = next(iter(item.polarities))
            if polarity == "default_off":
                parse_rule = "on for `1`/`true`/`TRUE`/`yes`/`YES`; any other value uses the default"
            else:
                parse_rule = "off for `0`/`false`/`FALSE`/`no`/`NO`; any other value uses the default"
            rendered_description += f" Parse rule: {parse_rule}; read once per process."
        escaped_description = rendered_description.replace("|", "\\|")
        scope = (
            "build"
            if "build" in item.scopes
            else ("test-only" if item.scopes == {"test-only"} else "runtime")
        )
        rows.append(
            f"| `{variable}` | {item.kind} | {scope} | `{item.path}` | {escaped_description} | {', '.join(sorted(item.families))} |"
        )
    return "\n".join(rows) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--self-test", action="store_true", help="run scanner parser self-tests"
    )
    parser.add_argument(
        "--check", action="store_true", help="fail if docs/ENV.md is stale"
    )
    parser.add_argument(
        "--output", type=Path, help="write the generated reference to this path"
    )
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    knobs, conflicts = collect()
    revision = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
    ).strip()
    rendered = render(knobs, revision)
    target = args.output or ROOT / "docs/ENV.md"
    if args.check:
        existing = target.read_text(encoding="utf-8") if target.exists() else ""
        normalize_revision = lambda value: re.sub(
            r"at source revision `[0-9a-f]+`", "at source revision `<revision>`", value
        )
        if normalize_revision(existing) != normalize_revision(rendered):
            print("docs/ENV.md is stale; run the generator", file=sys.stderr)
            return 1
    else:
        target.write_text(rendered, encoding="utf-8")
    print(f"knobs: {len(knobs)}")
    print(
        "kinds:",
        ", ".join(
            f"{kind}={sum(item.kind == kind for item in knobs.values())}"
            for kind in (
                "bool default-on",
                "bool default-off",
                "value",
                "build-time",
                "fixture path",
            )
        ),
    )
    print(
        f"undocumented: {sum(item.description == 'undocumented' for item in knobs.values())}"
    )
    print("polarity conflicts:", "none" if not conflicts else conflicts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
