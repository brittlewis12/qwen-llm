#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["jinja2==3.1.4"]
# ///
"""Render Qwen chat-template oracle cases with jinja2 (the engine Transformers
uses for `apply_chat_template`) and emit a fixture JSON.

Usage:
  scripts/reference/render_qwen_chat_template.py TEMPLATE.jinja CASES.json > fixture.json
  scripts/reference/render_qwen_chat_template.py TEMPLATE.jinja CASES.json --check fixture.json

Each case is {"id", "messages", "add_generation_prompt", and optional
"enable_thinking", "preserve_thinking", "tools"}. The rendered string is stored
verbatim; `raise_exception` cases store {"error": message}.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import jinja2
from jinja2 import sandbox


def raise_exception(message: str) -> None:
    raise jinja2.TemplateError(message)


def tojson(value, **kwargs):
    # Transformers' tojson: json.dumps with ensure_ascii=False, default separators.
    return json.dumps(value, ensure_ascii=False, **kwargs)


def build_env() -> sandbox.ImmutableSandboxedEnvironment:
    env = sandbox.ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
    )
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    return env


def render(env, source: str, case: dict) -> dict:
    template = env.from_string(source)
    variables = {k: v for k, v in case.items() if k != "id"}
    try:
        return {"rendered": template.render(**variables)}
    except jinja2.TemplateError as error:
        return {"error": str(error)}


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    template_path = Path(argv[1])
    cases_path = Path(argv[2])
    source = template_path.read_text(encoding="utf-8")
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    env = build_env()
    fixture = {
        "schema": "qwen.chat_template_oracle",
        "schema_version": 1,
        "engine": f"jinja2 {jinja2.__version__} ImmutableSandboxedEnvironment(trim_blocks, lstrip_blocks)",
        "template": template_path.name,
        "template_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "cases": [
            {
                "id": case["id"],
                "input": {k: v for k, v in case.items() if k != "id"},
                **render(env, source, case),
            }
            for case in cases
        ],
    }
    if len(argv) >= 5 and argv[3] == "--check":
        expected = json.loads(Path(argv[4]).read_text(encoding="utf-8"))
        if expected != fixture:
            print("fixture drift detected", file=sys.stderr)
            return 1
        print("fixture matches", file=sys.stderr)
        return 0
    json.dump(fixture, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
