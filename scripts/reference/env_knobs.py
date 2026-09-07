#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Generate the reference of environment knobs read by the Rust engine."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PREFIXES = ("QWEN_", "QWEN4EXP_", "DSV4_", "MUSE_")
FAMILIES = ("metal", "dflash", "deepseek_v4", "qwen4exp", "muse", "serve", "lens", "bench", "cli")


@dataclass
class Knob:
    variable: str
    kind: str
    path: str
    line: int
    description: str
    families: set[str] = field(default_factory=set)
    polarities: set[str] = field(default_factory=set)


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
        if name in lower or (name == "deepseek_v4" and ("dsv4" in lower or "deepseek" in lower)):
            return name
    return "cli" if "/qwen-cli/" in path else "metal"


def is_knob(variable: str) -> bool:
    return variable.startswith(PREFIXES)


def collect() -> tuple[dict[str, Knob], list[tuple[str, str, str, int, str]]]:
    knobs: dict[str, Knob] = {}
    conflicts: list[tuple[str, str, str, int, str]] = []
    consts: dict[str, tuple[str, str, int, str, str]] = {}
    files = sorted(Path(ROOT / "crates").rglob("*.rs"))
    sources: list[tuple[str, str, str, str, list[str]]] = []

    for source in files:
        rel = source.relative_to(ROOT).as_posix()
        text = source.read_text(encoding="utf-8")
        sources.append((rel, text, strip_comments(text), source.as_posix(), text.splitlines()))
        masked = sources[-1][2]
        for match in re.finditer(r"\bconst\s+([A-Za-z_]\w*_ENV)\s*:\s*&str\s*=\s*\"([A-Z][A-Z0-9_]*)\"", masked):
            value = match.group(2)
            if is_knob(value):
                consts[match.group(1)] = (value, rel, line_number(masked, match.start()), family(rel), text.splitlines())

    def add(variable: str, kind: str, rel: str, line: int, lines: list[str], polarity: str | None = None) -> None:
        if not is_knob(variable):
            return
        item = knobs.get(variable)
        desc = description(lines, line)
        if item is None or (item.description == "undocumented" and desc != "undocumented"):
            if item is None:
                item = Knob(variable, kind, rel, line, desc)
                knobs[variable] = item
        item.families.add(family(rel))
        if polarity:
            item.polarities.add(polarity)

    for rel, text, masked, _, lines in sources:
        for match in re.finditer(r"(?:[A-Za-z_]\w*::)*env_flag!\s*\(\s*(default_on|default_off)\s+[A-Za-z_]\w*\s*,\s*\"([A-Z][A-Z0-9_]*)\"", masked, re.DOTALL):
            polarity, variable = match.groups()
            add(variable, "bool " + polarity.replace("_", "-"), rel, line_number(masked, match.start()), lines, polarity)

        for match in re.finditer(r"(?:std::)?env::var(?:_os)?\s*\(\s*([A-Za-z_]\w*|\"([A-Z][A-Z0-9_]*)\")", masked):
            argument, literal = match.groups()
            if literal:
                add(literal, "value", rel, line_number(masked, match.start()), lines)
            elif argument in consts:
                variable, const_rel, const_line, _, const_lines = consts[argument]
                add(variable, "value", const_rel, const_line, const_lines)

    # Constants are often passed through a small parser/helper before that
    # helper calls var/var_os, so retain every qualifying *_ENV definition.
    for variable, rel, line, _, lines in consts.values():
        add(variable, "value", rel, line, lines)

    # Test-fixture aliases live in the `test_fixtures` inventory as
    # `env: &["A", "B"]` arrays rather than literal var_os calls. Each alias
    # is a `fixture path` knob described by the fixture's id.
    for rel, text, masked, _, lines in sources:
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

    for variable, item in knobs.items():
        if len(item.polarities) > 1:
            conflicts.append((variable, ", ".join(sorted(item.polarities)), ", ".join(sorted(item.families)), item.line, item.path))
    return knobs, conflicts


def render(knobs: dict[str, Knob]) -> str:
    rows = [
        "# Environment knobs",
        "",
        "Generated by [`scripts/reference/env_knobs.py`](../scripts/reference/env_knobs.py). Run `uv run scripts/reference/env_knobs.py` to refresh it.",
        "",
        "| Variable | Kind | Defining file:line | Description | Crate/module family |",
        "| --- | --- | --- | --- | --- |",
    ]
    for variable in sorted(knobs):
        item = knobs[variable]
        escaped_description = item.description.replace("|", "\\|")
        rows.append(f"| `{variable}` | {item.kind} | `{item.path}:{item.line}` | {escaped_description} | {', '.join(sorted(item.families))} |")
    return "\n".join(rows) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if docs/ENV.md is stale")
    args = parser.parse_args()
    knobs, conflicts = collect()
    rendered = render(knobs)
    target = ROOT / "docs/ENV.md"
    if args.check:
        if not target.exists() or target.read_text(encoding="utf-8") != rendered:
            print("docs/ENV.md is stale; run the generator", file=sys.stderr)
            return 1
    else:
        target.write_text(rendered, encoding="utf-8")
    print(f"knobs: {len(knobs)}")
    print("kinds:", ", ".join(f"{kind}={sum(item.kind == kind for item in knobs.values())}" for kind in ("bool default-on", "bool default-off", "value", "fixture path")))
    print(f"undocumented: {sum(item.description == 'undocumented' for item in knobs.values())}")
    print("polarity conflicts:", "none" if not conflicts else conflicts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
