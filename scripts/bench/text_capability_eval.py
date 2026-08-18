#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Fixed Qwen 27B text-capability and reasoning-effort packet."""

from __future__ import annotations

import argparse
import ast
import fcntl
import hashlib
import json
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter, deque
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, NoReturn

from family import source_identity, validate_qwen_identity


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "target" / "qualitative" / "qwen27b-text-capability-v1"
DEFAULT_QWEN = ROOT / "target" / "release" / "qwen"
DEFAULT_QWEN_BENCH = ROOT / "target" / "release" / "qwen-bench"
RUN_LOCK = ROOT / "target" / "qualitative" / ".qwen-text-capability-v1.lock"
BATTERY_ID = "qwen-27b-text-capability-v1"
PACKET_SEMANTIC_SHA256 = (
    "87152253314e0ae06877948dfc6dde9652388cb0c3d55048d25c5ab0fdee3a0c"
)
SEED = 42
SAMPLING = {
    "temperature": 0.0,
    "top_k": 200,
    "top_p": 1.0,
    "min_p": 0.05,
    "seed": SEED,
}
MODES = ("no-thinking", "low", "medium", "xhigh")
EFFORT_TASK_IDS = ("c3", "c4", "c6", "c8", "r1", "r4", "j2", "j3")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
ARM_RE = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")

QWEN38_REASONING_EFFORT_XHIGH = (
    "Reasoning effort is set to xhigh. Please think carefully through the task, "
    "validate key assumptions, consider plausible alternatives, and prioritize "
    "correctness, consistency, and clarity in the final answer."
)
QWEN38_REASONING_EFFORT_LOW = (
    "Reasoning effort is set to low. Keep your thinking brief and focused, moving "
    "directly to the conclusion without unnecessary elaboration."
)

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "qwen36-q4": {
        "family": "qwen36",
        "display": "Qwen3.6 27B Q4_K_M",
        "path": Path.home() / "models" / "Qwen3.6-27B-Q4_K_M.gguf",
        "bytes": 16_817_244_384,
        "sha256": "5ed60d0af4650a854b1755bd392f9aef4872643dc25a254bc68043fa638392a0",
        "producer": "local pinned regression anchor",
    },
    "qwen38-q4": {
        "family": "qwen38",
        "display": "Qwen3.8 27B Q4_K_M",
        "path": Path.home() / "models" / "Qwen3.8-27B-Q4_K_M.gguf",
        "bytes": 17_106_773_984,
        "sha256": "7b2aec3b9ababdfd75aa17552ee95607d866e44decf547f6f12fcef85cc89f1b",
        "producer": "unsloth/Qwen3.8-27B-GGUF",
    },
    "ridge": {
        "family": "qwen38",
        "display": "Qwen3.8 27B Ridge 3.7bpw",
        "path": Path.home()
        / "models"
        / "qwen38-27b-ridge"
        / "Qwen3.8-27B-Ridge-3.7bpw.gguf",
        "bytes": 12_599_187_008,
        "sha256": "95580dbdaad579582ee898257116abc18d7f3625a00c16a15735d41444a09f5e",
        "producer": "empero-ai/Qwen3.8-27B-Ridge-GGUF@578362007e830185e1a03ff3454309c8590bad5f",
    },
}

CHILD_ENV_ALLOWLIST = {
    "CARGO_HOME",
    "COMMAND_MODE",
    "DEVELOPER_DIR",
    "HOME",
    "LANG",
    "LOGNAME",
    "MACOSX_DEPLOYMENT_TARGET",
    "PATH",
    "RUSTUP_HOME",
    "SDKROOT",
    "SHELL",
    "TERM",
    "TMPDIR",
    "USER",
}

CODE_SUFFIX = (
    "\n\nReturn only the requested Python function definition as plain text. No "
    "Markdown, tests, imports, classes, decorators, global variables, I/O, or "
    "explanations. Use only Python built-ins. Inputs contain only inert built-in "
    "values."
)
EXACT_SUFFIX = (
    "\n\nReturn exactly the requested answer token and nothing else. No explanation "
    "or Markdown."
)
JSON_SUFFIX = (
    "\n\nReturn one raw JSON value only: no Markdown fence, prose, comments, trailing "
    "commas, or extra keys. This is a data-serialization task, not native tool use "
    "or a function call."
)


@dataclass(frozen=True)
class Task:
    task_id: str
    category: str
    prompt: str
    tokens: int
    grader: str
    expected: Any = None


def die(message: str, code: int = 2) -> NoReturn:
    print(f"[text-capability] error: {message}", file=sys.stderr)
    raise SystemExit(code)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def reject_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant {value!r}")


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def parse_json(text: str) -> Any:
    return json.loads(
        text,
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_constant,
    )


def read_json(path: Path) -> Any:
    try:
        return parse_json(path.read_text())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        die(f"cannot read strict JSON from {path}: {error}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text().splitlines()
    except OSError as error:
        die(f"cannot read {path}: {error}")
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = parse_json(line)
        except (ValueError, json.JSONDecodeError) as error:
            die(f"invalid JSON at {path}:{line_no}: {error}")
        if not isinstance(row, dict):
            die(f"expected an object at {path}:{line_no}")
        rows.append(row)
    return rows


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=True) + "\n").encode()


def jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, json_bytes(value))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write(path, jsonl_bytes(rows))


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_sha256(value: Any) -> str:
    return sha256_bytes(
        json.dumps(
            value, sort_keys=True, ensure_ascii=True, separators=(",", ":")
        ).encode()
    )


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def ref_merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    normalized = sorted(
        (min(a, b), max(a, b)) for a, b in spans if min(a, b) != max(a, b)
    )
    merged: list[tuple[int, int]] = []
    for start, end in normalized:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def ref_split_escaped(text: str, sep: str = "|", esc: str = "\\") -> list[str]:
    fields: list[str] = []
    current: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char == esc:
            index += 1
            if index == len(text):
                current.append(esc)
                break
            current.append(text[index])
        elif char == sep:
            fields.append("".join(current))
            current = []
        else:
            current.append(char)
        index += 1
    fields.append("".join(current))
    return fields


def parse_data_path(path: str) -> list[tuple[str, Any]] | None:
    if not path:
        return None
    parts: list[tuple[str, Any]] = []
    index = 0
    while index < len(path):
        if path[index] != "[":
            key: list[str] = []
            while index < len(path) and path[index] not in ".[":
                if path[index] == "\\":
                    index += 1
                    if index >= len(path):
                        return None
                    key.append(path[index])
                elif path[index] == "]":
                    return None
                else:
                    key.append(path[index])
                index += 1
            if not key:
                return None
            parts.append(("key", "".join(key)))

        while index < len(path) and path[index] == "[":
            close = path.find("]", index + 1)
            if close < 0:
                return None
            raw = path[index + 1 : close]
            if not raw or not raw.isascii() or not raw.isdecimal():
                return None
            parts.append(("index", int(raw)))
            index = close + 1

        if index == len(path):
            break
        if path[index] != ".":
            return None
        index += 1
        if index == len(path) or path[index] in ".[":
            return None
    return parts or None


def ref_deep_get(data: Any, path: str) -> tuple[bool, Any]:
    parts = parse_data_path(path)
    if parts is None:
        return (False, None)
    current = data
    for kind, value in parts:
        if kind == "key":
            if not isinstance(current, dict) or value not in current:
                return (False, None)
            current = current[value]
        else:
            if not isinstance(current, list) or value >= len(current):
                return (False, None)
            current = current[value]
    return (True, current)


def ref_topo_layers(graph: dict[str, list[str]]) -> list[list[str]] | None:
    nodes = set(graph)
    for prerequisites in graph.values():
        nodes.update(prerequisites)
    remaining = {node: set(graph.get(node, [])) for node in nodes}
    completed: set[str] = set()
    layers: list[list[str]] = []
    while len(completed) < len(nodes):
        layer = sorted(
            node
            for node, prerequisites in remaining.items()
            if node not in completed and prerequisites <= completed
        )
        if not layer:
            return None
        layers.append(layer)
        completed.update(layer)
    return layers


def ref_window_max(values: list[int], k: int) -> list[int]:
    if k <= 0 or k > len(values):
        return []
    queue: deque[int] = deque()
    output: list[int] = []
    for index, value in enumerate(values):
        while queue and queue[0] <= index - k:
            queue.popleft()
        while queue and values[queue[-1]] <= value:
            queue.pop()
        queue.append(index)
        if index >= k - 1:
            output.append(values[queue[0]])
    return output


def valid_list_index(value: Any, length: int) -> bool:
    return type(value) is int and 0 <= value < length


def ref_apply_patch(doc: Any, ops: list[tuple[Any, ...]]) -> Any:
    result = deepcopy(doc)
    try:
        for operation in ops:
            if not isinstance(operation, (tuple, list)) or len(operation) not in {2, 3}:
                return None
            kind, path = operation[0], operation[1]
            if kind not in {"set", "del"} or not isinstance(path, list) or not path:
                return None
            if (kind == "set") != (len(operation) == 3):
                return None
            parent = result
            for component in path[:-1]:
                if isinstance(parent, dict):
                    if component not in parent:
                        return None
                    parent = parent[component]
                elif isinstance(parent, list):
                    if not valid_list_index(component, len(parent)):
                        return None
                    parent = parent[component]
                else:
                    return None
            final = path[-1]
            if isinstance(parent, dict):
                if kind == "set":
                    parent[final] = deepcopy(operation[2])
                elif final in parent:
                    del parent[final]
                else:
                    return None
            elif isinstance(parent, list):
                if not valid_list_index(final, len(parent)):
                    return None
                if kind == "set":
                    parent[final] = deepcopy(operation[2])
                else:
                    del parent[final]
            else:
                return None
    except (KeyError, TypeError, IndexError):
        return None
    return result


def ref_ledger_balance(events: list[tuple[Any, ...]]) -> int | None:
    balance = 0
    seen: dict[str, tuple[str, int]] = {}
    for event in events:
        if not isinstance(event, (tuple, list)) or len(event) != 3:
            return None
        event_id, kind, amount = event
        if (
            not isinstance(event_id, str)
            or kind not in {"credit", "debit"}
            or type(amount) is not int
            or amount <= 0
        ):
            return None
        payload = (kind, amount)
        if event_id in seen:
            if seen[event_id] != payload:
                return None
            continue
        seen[event_id] = payload
        if kind == "credit":
            balance += amount
        elif amount > balance:
            return None
        else:
            balance -= amount
    return balance


def ref_minimal_cover(text: str, required: str) -> tuple[int, int] | None:
    if not required:
        return (0, 0)
    need = Counter(required)
    best: tuple[int, int] | None = None
    for start in range(len(text)):
        have: Counter[str] = Counter()
        for end in range(start, len(text)):
            have[text[end]] += 1
            if all(have[char] >= count for char, count in need.items()):
                candidate = (start, end + 1)
                if best is None or (candidate[1] - candidate[0], candidate[0]) < (
                    best[1] - best[0],
                    best[0],
                ):
                    best = candidate
                break
    return best


def natural_runs(value: str) -> list[tuple[bool, str]]:
    if not value:
        return []
    runs: list[tuple[bool, str]] = []
    start = 0
    digit = value[0].isdigit()
    for index, char in enumerate(value[1:], 1):
        if char.isdigit() != digit:
            runs.append((digit, value[start:index]))
            start = index
            digit = char.isdigit()
    runs.append((digit, value[start:]))
    return runs


def ref_natural_compare(a: str, b: str) -> int:
    left = natural_runs(a)
    right = natural_runs(b)
    for left_run, right_run in zip(left, right):
        left_digit, left_text = left_run
        right_digit, right_text = right_run
        if left_digit != right_digit:
            return -1 if left_text < right_text else 1
        if left_digit:
            left_value, right_value = int(left_text), int(right_text)
            if left_value != right_value:
                return -1 if left_value < right_value else 1
            if len(left_text) != len(right_text):
                return -1 if len(left_text) < len(right_text) else 1
        elif left_text != right_text:
            return -1 if left_text < right_text else 1
    if len(left) == len(right):
        return 0
    return -1 if len(left) < len(right) else 1


NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def ref_render_template(template: str, values: dict[str, Any]) -> str | None:
    output: list[str] = []
    index = 0
    while index < len(template):
        if template.startswith("{{", index):
            output.append("{")
            index += 2
        elif template.startswith("}}", index):
            output.append("}")
            index += 2
        elif template[index] == "{":
            close = template.find("}", index + 1)
            if close < 0:
                return None
            name = template[index + 1 : close]
            if not NAME_RE.fullmatch(name) or name not in values:
                return None
            value = values[name]
            if not isinstance(value, (str, int)) or isinstance(value, bool):
                return None
            output.append(str(value))
            index = close + 1
        elif template[index] == "}":
            return None
        else:
            output.append(template[index])
            index += 1
    return "".join(output)


def code_cases(task_id: str) -> list[tuple[tuple[Any, ...], Any]]:
    rng = random.Random(0x51A7 + int(task_id[1:]))
    if task_id == "c1":
        raw = [
            [],
            [(5, 1), (2, 4), (8, 8), (7, 9), (9, 10)],
            [(3, 1), (1, 3)],
            [(-1, -4), (-6, -5), (-5, -4)],
        ]
        for _ in range(48):
            raw.append(
                [
                    (rng.randint(-20, 20), rng.randint(-20, 20))
                    for _ in range(rng.randrange(12))
                ]
            )
        return [((spans,), ref_merge_spans(spans)) for spans in raw]
    if task_id == "c2":
        cases: list[tuple[tuple[Any, ...], Any]] = []
        fixed = [
            (("a\\|b||c\\\\d|e\\",), ["a|b", "", "c\\d", "e\\"]),
            (("|x|",), ["", "x", ""]),
            (("",), [""]),
            (("\\||", "|", "\\"), ["|", ""]),
            (("a,,b", ",", "~"), ["a", "", "b"]),
        ]
        cases.extend(fixed)
        for _ in range(60):
            text = "".join(rng.choice("ab|\\") for _ in range(rng.randrange(20)))
            cases.append(((text,), ref_split_escaped(text)))
        return cases
    if task_id == "c3":
        data = {
            "a.b": [{"x": 3}, {"x": None}],
            "a": {"b": [9, {"q.r": [4]}]},
            "br[k]": 7,
            "slash\\key": {"v": 2},
            "none": None,
        }
        paths = [
            "a\\.b[1].x",
            "a.b[0]",
            "a.b[1].q\\.r[0]",
            "br\\[k\\]",
            "slash\\\\key.v",
            "none",
            "a.b[3]",
            "a.b[-1]",
            "a.b[0",
            "a..b",
            "a.b[]",
            "a.b[1]x",
            "a.b[1].",
            "\\",
            "",
        ]
        return [((data, path), ref_deep_get(data, path)) for path in paths]
    if task_id == "c4":
        graphs: list[dict[str, list[str]]] = [
            {
                "ship": ["test", "pack"],
                "test": ["build"],
                "pack": ["build"],
                "build": [],
            },
            {"a": ["b"], "b": ["a"]},
            {"z": ["x"]},
            {"b": [], "a": []},
            {"self": ["self"]},
        ]
        for _ in range(35):
            count = rng.randint(1, 9)
            nodes = [chr(ord("a") + index) for index in range(count)]
            graph: dict[str, list[str]] = {}
            for index, node in enumerate(nodes):
                prerequisites = [prior for prior in nodes[:index] if rng.random() < 0.3]
                if prerequisites and rng.random() < 0.2:
                    prerequisites.append(prerequisites[0])
                if rng.random() < 0.85:
                    graph[node] = prerequisites
            graphs.append(graph)
        return [((graph,), ref_topo_layers(graph)) for graph in graphs]
    if task_id == "c5":
        raw = [
            ([1, 4, 2, 3, 5, 0], 3),
            ([-1, -1], 2),
            ([3, 2, 1], 1),
            ([], 1),
            ([1, 2], 3),
            ([1, 2], 0),
        ]
        for _ in range(50):
            values = [rng.randint(-20, 20) for _ in range(rng.randrange(30))]
            k = rng.randint(-2, len(values) + 3)
            raw.append((values, k))
        return [((values, k), ref_window_max(values, k)) for values, k in raw]
    if task_id == "c6":
        raw = [
            (
                {"a": [10, 20], "b": {"x": 1}},
                [("set", ["a", 1], 25), ("del", ["b", "x"]), ("set", ["b", "y"], 2)],
            ),
            ({"a": [1]}, [("set", ["a", 1], 2)]),
            ({"a": [1, 2]}, [("del", ["a", 0]), ("set", ["a", 0], 9)]),
            ({}, [("set", [], 1)]),
            ({"a": {}}, [("del", ["a", "missing"])]),
            ({"a": [1]}, [("set", ["a", True], 3)]),
            ({"a": {"b": 1}}, [("set", ["a", "c"], [1, 2])]),
            ({"a": 1}, [("set", ["a", "x"], 2)]),
            ({"a": [0]}, [("wat", ["a", 0])]),
        ]
        return [((doc, ops), ref_apply_patch(doc, ops)) for doc, ops in raw]
    if task_id == "c7":
        events = [
            [("a", "credit", 10), ("b", "debit", 4), ("a", "credit", 10)],
            [("a", "credit", 1), ("a", "debit", 1)],
            [("a", "debit", 1)],
            [("a", "credit", True)],
            [("a", "credit", 3), ("b", "debit", 3)],
            [("a", "credit", 0)],
            [("a", "other", 3)],
            [(1, "credit", 3)],
        ]
        for sample in range(45):
            balance = 0
            row: list[tuple[Any, ...]] = []
            for index in range(rng.randrange(1, 12)):
                if balance == 0 or rng.random() < 0.6:
                    amount = rng.randint(1, 12)
                    row.append((f"{sample}-{index}", "credit", amount))
                    balance += amount
                else:
                    amount = rng.randint(1, balance)
                    row.append((f"{sample}-{index}", "debit", amount))
                    balance -= amount
                if rng.random() < 0.2:
                    row.append(row[-1])
            events.append(row)
        return [((row,), ref_ledger_balance(row)) for row in events]
    if task_id == "c8":
        raw = [
            ("ADOBECODEBANC", "ABC"),
            ("aaabcbc", "aabc"),
            ("abdcab", "ab"),
            ("abc", "zz"),
            ("abc", ""),
        ]
        for _ in range(65):
            text = "".join(rng.choice("abcd") for _ in range(rng.randrange(18)))
            required = "".join(rng.choice("abcde") for _ in range(rng.randrange(7)))
            raw.append((text, required))
        return [
            ((text, required), ref_minimal_cover(text, required))
            for text, required in raw
        ]
    if task_id == "c9":
        raw = [
            ("v2x10", "v2x9"),
            ("a01", "a1"),
            ("x9a", "x09"),
            ("A2", "a2"),
            ("file", "file"),
            ("x", "x0"),
            ("", "0"),
        ]
        alphabet = "abAB019"
        for _ in range(80):
            left = "".join(rng.choice(alphabet) for _ in range(rng.randrange(15)))
            right = "".join(rng.choice(alphabet) for _ in range(rng.randrange(15)))
            raw.append((left, right))
            raw.append((right, left))
        return [
            ((left, right), ref_natural_compare(left, right)) for left, right in raw
        ]
    if task_id == "c10":
        raw: list[tuple[str, dict[str, Any]]] = [
            ("Hi {{ {name} }}", {"name": "Ada"}),
            ("{x}{x}", {"x": 3}),
            ("}}{{", {}),
            ("{}", {}),
            ("{missing}", {}),
            ("{x}", {"x": True}),
            ("a}b", {}),
            ("{{{x}}}", {"x": "v"}),
            ("{_x1}", {"_x1": -2}),
            ("{1x}", {"1x": 2}),
        ]
        for index in range(40):
            values = {"x": index, "name": f"n{index}"}
            template = rng.choice(
                ["{{{name}}}:{x}", "{name}-{{x}}-{x}", "{{{{{x}}}}}", "plain"]
            )
            raw.append((template, values))
        return [
            ((template, values), ref_render_template(template, values))
            for template, values in raw
        ]
    raise ValueError(f"unknown code task {task_id}")


def build_tasks() -> list[Task]:
    code_specs = [
        (
            "c1",
            384,
            "Implement merge_spans(spans). Each item is an integer pair (a, b). Normalize it to (min(a,b), max(a,b)), discard zero-length spans, sort the remaining half-open spans, and merge spans that overlap or touch. Return a list of (start, end) tuples. Do not mutate spans.",
        ),
        (
            "c2",
            384,
            'Implement split_escaped(text, sep="|", esc="\\\\"). sep and esc are distinct one-character strings. An unescaped separator starts a new field. An escape consumes the next character and inserts it literally. A final dangling escape is inserted literally. Preserve empty fields.',
        ),
        (
            "c3",
            768,
            "Implement deep_get(data, path). A path contains dot-separated dictionary keys and zero-based list indexes written as [N], where N is a nonnegative decimal integer. A backslash in a key escapes the next character, allowing literal dot, brackets, or backslash. Dictionary lookup and list indexing may alternate. Return (True, value) if the complete path exists, including when value is None; otherwise return (False, None). Malformed paths also return (False, None).",
        ),
        (
            "c4",
            512,
            "Implement topo_layers(graph). graph maps a node string to a list of prerequisite node strings. Nodes appearing only as prerequisites must also be included. Return topological layers as a list of lexicographically sorted lists. A layer contains every node whose prerequisites were completed in earlier layers, not the current layer. Return None if there is a cycle. Do not mutate graph.",
        ),
        (
            "c5",
            384,
            "Implement window_max(values, k). Return the maximum of every contiguous window of length k. Return [] if k <= 0 or k > len(values). The implementation must handle 200,000 values within a short timeout, so repeated full-window scans are not acceptable.",
        ),
        (
            "c6",
            768,
            'Implement apply_patch(doc, ops). doc contains dictionaries, lists, and scalar JSON-like values. Each operation is ("set", path, value) or ("del", path), where path is a nonempty list of dictionary keys or list indexes. Apply operations sequentially to a deep copy. Dictionary set may create the final key; earlier components must exist. Every list index must already be in range and negative or boolean indexes are invalid. del requires an existing final component. If any operation or path is invalid, return None. Never mutate doc or values in ops.',
        ),
        (
            "c7",
            384,
            'Implement ledger_balance(events). Each event is (event_id, kind, amount), where event_id is a string, kind is "credit" or "debit", and amount is a positive integer but not a boolean. Start at zero. Repeating an ID with exactly the same kind and amount is an idempotent duplicate. Repeating it with different data is invalid. A debit that would make the balance negative is invalid. Return the final balance, or None for invalid input.',
        ),
        (
            "c8",
            512,
            "Implement minimal_cover(text, required). Return (start, end) for the shortest half-open substring of text containing every character in required with at least its required multiplicity. Break equal-length ties by smallest start. Return (0, 0) when required is empty and None when no cover exists.",
        ),
        (
            "c9",
            512,
            "Implement natural_compare(a, b). Split each ASCII string into maximal digit and non-digit runs. Compare two non-digit runs by ordinary Python string order. Compare two digit runs by integer value; if values are equal, the shorter digit run sorts first. If one corresponding run is digits and the other is not, compare their raw run texts by ordinary Python string order. If all compared runs are equal, the string with fewer runs sorts first. Return -1, 0, or 1.",
        ),
        (
            "c10",
            512,
            "Implement render_template(template, values). {{ emits { and }} emits }. A placeholder is {name}, where name matches ASCII [A-Za-z_][A-Za-z0-9_]*. Every placeholder must exist in values, and its value must be a string or an integer other than a boolean. Insert str(value). Return None for unmatched braces, invalid names, missing values, or invalid value types.",
        ),
    ]
    tasks = [
        Task(task_id, "code", prompt + CODE_SUFFIX, tokens, "python_tests")
        for task_id, tokens, prompt in code_specs
    ]
    reasoning = [
        (
            "r1",
            "Find the smallest three-digit integer n such that n mod 7 = 3, n mod 11 = 5, and the decimal digits of n sum to 15. Return only the integer.",
            "654",
        ),
        (
            "r2",
            "Jobs J, K, L, and M occupy positions 1 through 4 exactly once. K is neither first nor last. M is immediately after K. L is last. J is before L. Return the four letters in position order with no separators.",
            "JKML",
        ),
        (
            "r3",
            "An undirected graph has edges A-B:4, A-C:2, C-B:1, B-D:3, C-D:5, B-E:6, D-E:1, and C-E:8. Find the minimum-cost simple path from A to E. If tied, choose the lexicographically smallest comma-separated node sequence. Return path|cost.",
            "A,C,B,D,E|7",
        ),
        (
            "r4",
            "Treat 0xD3 as exactly eight bits. Rotate it left by three bit positions, then XOR the result with 0xB6. Return the result as uppercase 0xNN.",
            "0x28",
        ),
        (
            "r5",
            "Among 60 items, 32 have P, 28 have Q, and 25 have R. Counts for P intersection Q, P intersection R, and Q intersection R are 15, 14, and 12. Seven have all three. Return exactly_one=X;none=Y.",
            "exactly_one=24;none=9",
        ),
        (
            "r6",
            "Define a OP b = 2a - b. OP has higher precedence than + and is left-associative. Evaluate 8 OP 3 + 5 OP 4 OP 2. Return only the integer.",
            "23",
        ),
        (
            "r7",
            "Intervals are half-open: [0,4), [2,7), [4,6), [5,8), [6,9), [7,10). Find the maximum number active simultaneously and the earliest integer time at which that maximum occurs. Return maximum@time.",
            "3@5",
        ),
        (
            "r8",
            "Start with x=7 and process ABACCBAB left to right. A means x=(3*x+1) mod 17; B means x=x XOR 5; C means x=(x*x+2) mod 17. Return the final decimal value.",
            "12",
        ),
    ]
    tasks.extend(
        Task(task_id, "reasoning", prompt + EXACT_SUFFIX, 32, "exact", expected)
        for task_id, prompt, expected in reasoning
    )
    json_specs: list[tuple[str, str, Any]] = [
        (
            "j1",
            'Plan rollout waves for services: db prerequisites [] risk 3; api prerequisites [db] risk 2; worker prerequisites [db] risk 3; ui prerequisites [api] risk 1; reports prerequisites [db,worker] risk 2. A wave has at most two services and total risk at most 4. Prerequisites finish in earlier waves. At each wave, sort currently ready services lexicographically and greedily add each if it still fits. Risk above 4 is deferred. Return exactly {"waves":[...],"deferred":[...]}; names inside arrays are lexicographically ordered.',
            {"waves": [["db"], ["api"], ["ui", "worker"], ["reports"]], "deferred": []},
        ),
        (
            "j2",
            'Schedule nonpreemptive tasks: A duration 3 priority 2 prerequisites []; B duration 2 priority 1 prerequisites []; C duration 2 priority 3 prerequisites [A]; D duration 1 priority 2 prerequisites [A]; E duration 2 priority 2 prerequisites [B,C]. Workers are 0 and 1. At each integer time first finish ending tasks, sort ready tasks by descending priority then ID, and assign in that order to the lowest-numbered free worker. Return {"assignments":[...],"makespan":N}. Each assignment has exactly task, worker, start, end; order by start then worker.',
            {
                "assignments": [
                    {"task": "A", "worker": 0, "start": 0, "end": 3},
                    {"task": "B", "worker": 1, "start": 0, "end": 2},
                    {"task": "C", "worker": 0, "start": 3, "end": 5},
                    {"task": "D", "worker": 1, "start": 3, "end": 4},
                    {"task": "E", "worker": 0, "start": 5, "end": 7},
                ],
                "makespan": 7,
            },
        ),
        (
            "j3",
            'Choose actions under budget 9: contain cost 3 value 5 prerequisites []; snapshot cost 2 value 4 prerequisites []; patch cost 4 value 8 prerequisites [snapshot]; notify cost 1 value 1 prerequisites [contain]; rotate cost 3 value 6 prerequisites [contain]. Selecting an action requires all prerequisites. Maximize total value, then minimize total cost, then choose the lexicographically smallest sorted selected-ID array. For execution order repeatedly choose the lexicographically smallest selected action whose prerequisites executed. Return exactly {"selected":[],"order":[],"cost":N,"value":N}.',
            {
                "selected": ["contain", "patch", "snapshot"],
                "order": ["contain", "snapshot", "patch"],
                "cost": 9,
                "value": 17,
            },
        ),
        (
            "j4",
            'Pack jobs into bins of capacity 6; each bin contains one zone. Jobs: A size 4 east, B size 3 west, C size 2 east, D size 3 east, E size 1 west, F size 2 west. Process descending size, breaking ties by ID. Put each job in the lowest-index compatible bin with room, otherwise create a bin. Preserve processing order within bins. Return {"bins":[{"zone":...,"jobs":[...],"load":...},...]} in bin-index order.',
            {
                "bins": [
                    {"zone": "east", "jobs": ["A", "C"], "load": 6},
                    {"zone": "west", "jobs": ["B", "F", "E"], "load": 6},
                    {"zone": "east", "jobs": ["D"], "load": 3},
                ]
            },
        ),
    ]
    tasks.extend(
        Task(task_id, "json", prompt + JSON_SUFFIX, 384, "strict_json", expected)
        for task_id, prompt, expected in json_specs
    )
    return tasks


def render_no_thinking(user: str) -> str:
    return (
        f"<|im_start|>user\n{user.strip()}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def render_qwen38_effort(user: str, mode: str) -> str:
    if mode == "no-thinking":
        return render_no_thinking(user)
    if mode not in {"low", "medium", "xhigh"}:
        raise ValueError(f"unknown Qwen3.8 effort mode {mode!r}")
    instruction = {
        "low": QWEN38_REASONING_EFFORT_LOW,
        "medium": None,
        "xhigh": QWEN38_REASONING_EFFORT_XHIGH,
    }[mode]
    system = f"<|im_start|>system\n{instruction}<|im_end|>\n" if instruction else ""
    return (
        f"{system}<|im_start|>user\n{user.strip()}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"
    )


def effort_tokens(task: Task) -> int:
    if task.category == "reasoning":
        return 256
    if task.category == "json":
        return 512
    return max(task.tokens, 768)


def task_manifest(task: Task) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": task.task_id,
        "category": task.category,
        "grader": task.grader,
        "tokens": task.tokens,
        "prompt": task.prompt,
        "prompt_sha256": sha256_bytes(task.prompt.encode()),
    }
    if task.category == "code":
        cases = code_cases(task.task_id)
        row["test_case_count"] = len(cases)
        row["test_cases_sha256"] = sha256_bytes(repr(cases).encode())
    else:
        row["expected"] = task.expected
    return row


def build_packet() -> tuple[dict[str, Any], dict[str, bytes]]:
    tasks = build_tasks()
    by_id = {task.task_id: task for task in tasks}
    if len(by_id) != len(tasks):
        raise RuntimeError("duplicate task IDs")
    capability = [
        {
            "id": task.task_id,
            "prompt": render_no_thinking(task.prompt),
            "tokens": task.tokens,
            "sampling": SAMPLING,
        }
        for task in tasks
    ]
    effort: list[dict[str, Any]] = []
    effort_order: list[dict[str, Any]] = []
    for task_index, task_id in enumerate(EFFORT_TASK_IDS):
        task = by_id[task_id]
        rotated = MODES[task_index % len(MODES) :] + MODES[: task_index % len(MODES)]
        effort_order.append({"task_id": task_id, "modes": list(rotated)})
        for mode in rotated:
            effort.append(
                {
                    "id": f"{task_id}--{mode}",
                    "prompt": render_qwen38_effort(task.prompt, mode),
                    "tokens": effort_tokens(task),
                    "sampling": SAMPLING,
                }
            )
    request_rows = {"capability": capability, "effort": effort}
    request_bytes = {
        profile: jsonl_bytes(rows) for profile, rows in request_rows.items()
    }
    packet = {
        "schema_version": 1,
        "battery_id": BATTERY_ID,
        "interpretation": {
            "capability": "fixed no-thinking text surface; not a general model ceiling",
            "effort": "single interleaved discovery pass; policy direction, not a timing promotion",
            "json": "ordinary strict text serialization, not native tool invocation",
        },
        "sampling": SAMPLING,
        "capability_prompt_mode": "byte-identical preclosed empty think across all model arms",
        "effort_modes": list(MODES),
        "effort_task_ids": list(EFFORT_TASK_IDS),
        "effort_request_order": effort_order,
        "tasks": [task_manifest(task) for task in tasks],
        "request_sets": {
            profile: {
                "path": f"requests-{profile}.jsonl",
                "count": len(request_rows[profile]),
                "sha256": sha256_bytes(content),
            }
            for profile, content in request_bytes.items()
        },
        "models": {
            key: {
                name: (str(value) if isinstance(value, Path) else value)
                for name, value in spec.items()
            }
            for key, spec in MODEL_SPECS.items()
        },
        "safety": {
            "whole_model_residency": "disabled",
            "model_prefetch": "off",
            "prefix_cache": "disabled",
            "execution": "serial",
        },
    }
    return packet, request_bytes


def verify_contract() -> tuple[dict[str, Any], dict[str, bytes]]:
    packet, request_bytes = build_packet()
    actual = semantic_sha256(packet)
    if PACKET_SEMANTIC_SHA256 != "TODO" and actual != PACKET_SEMANTIC_SHA256:
        die(
            f"maintained packet identity changed: expected={PACKET_SEMANTIC_SHA256} actual={actual}"
        )
    return packet, request_bytes


def managed_prepare_paths(output: Path) -> list[Path]:
    return [
        output / "packet.json",
        output / "requests-capability.jsonl",
        output / "requests-effort.jsonl",
    ]


def prepare(output: Path, force: bool) -> None:
    packet, request_bytes = verify_contract()
    existing = [path for path in managed_prepare_paths(output) if path.exists()]
    if existing and not force:
        die(f"refusing to replace prepared packet under {output}; use --force")
    if force and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "packet.json", packet)
    for profile, content in request_bytes.items():
        atomic_write(output / f"requests-{profile}.jsonl", content)
    print(f"prepared {BATTERY_ID} under {output}")
    print(f"packet {semantic_sha256(packet)}")
    for profile, content in request_bytes.items():
        print(f"{profile} requests {sha256_bytes(content)}")


def verify_prepared(output: Path) -> dict[str, Any]:
    expected, request_bytes = verify_contract()
    actual = read_json(output / "packet.json")
    if actual != expected:
        die(f"prepared packet drifted: {output / 'packet.json'}")
    for profile, content in request_bytes.items():
        path = output / f"requests-{profile}.jsonl"
        if not path.is_file() or path.read_bytes() != content:
            die(f"prepared request set drifted: {path}")
    return expected


def arm_path(output: Path, arm: str, suffix: str) -> Path:
    if not ARM_RE.fullmatch(arm):
        die(f"invalid arm {arm!r}")
    return output / f"{arm}{suffix}"


def binary_identity(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        die(f"binary is not executable: {path}")
    stat = path.stat()
    return {
        "path": str(path),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(path),
    }


def model_locator(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "bytes": stat.st_size,
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
    }


def verified_model_identity(output: Path, model_key: str) -> dict[str, Any]:
    spec = MODEL_SPECS[model_key]
    path = Path(spec["path"]).expanduser().resolve()
    if not path.is_file():
        die(f"model is missing: {path}")
    locator = model_locator(path)
    if locator["bytes"] != spec["bytes"]:
        die(
            f"model size mismatch for {model_key}: expected={spec['bytes']} actual={locator['bytes']}"
        )
    cache_path = output / "model-identities.json"
    cache = read_json(cache_path) if cache_path.is_file() else {}
    cached = cache.get(model_key) if isinstance(cache, dict) else None
    if (
        isinstance(cached, dict)
        and cached.get("locator") == locator
        and cached.get("sha256") == spec["sha256"]
    ):
        return {
            "model_key": model_key,
            "family": spec["family"],
            "display": spec["display"],
            "producer": spec["producer"],
            "locator": locator,
            "sha256": spec["sha256"],
        }
    print(
        f"verifying full model SHA-256 for {model_key} ({locator['bytes']} bytes)",
        flush=True,
    )
    digest = sha256_file(path)
    if digest != spec["sha256"]:
        die(
            f"model digest mismatch for {model_key}: expected={spec['sha256']} actual={digest}"
        )
    record = {
        "model_key": model_key,
        "family": spec["family"],
        "display": spec["display"],
        "producer": spec["producer"],
        "locator": locator,
        "sha256": digest,
    }
    cache = cache if isinstance(cache, dict) else {}
    cache[model_key] = record
    write_json(cache_path, cache)
    return record


def child_environment() -> tuple[dict[str, str], dict[str, Any]]:
    inherited = {
        key: value
        for key, value in os.environ.items()
        if key in CHILD_ENV_ALLOWLIST or key.startswith("LC_")
    }
    environment = dict(inherited)
    environment.update(
        {
            "QWEN_DSV4_PREFETCH": "off",
            "QWEN_DSV4_RESIDENCY_SET": "0",
            # `warn` for tracing generally, but keep the `qwen_diag`
            # bare-body load diagnostics (`[metal-load-ledger]` etc.) at
            # `info` so failure-artifact stderr tails still capture them.
            "RUST_LOG": "warn,qwen_diag=info",
        }
    )
    qwen_controls = sorted(key for key in environment if key.startswith("QWEN_"))
    if qwen_controls != ["QWEN_DSV4_PREFETCH", "QWEN_DSV4_RESIDENCY_SET"]:
        raise RuntimeError(f"unsafe QWEN child controls: {qwen_controls!r}")
    record = {
        "policy": "allowlisted base environment plus fixed safety overlays",
        "removed_qwen_keys": sorted(
            key for key in os.environ if key.startswith("QWEN_")
        ),
        "overlay": {
            "QWEN_DSV4_PREFETCH": "off",
            "QWEN_DSV4_RESIDENCY_SET": "0",
            "RUST_LOG": "warn,qwen_diag=info",
        },
    }
    return environment, record


def capture_text(command: list[str]) -> str:
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    return (process.stdout or process.stderr).strip() or f"exit {process.returncode}"


def host_context() -> dict[str, str]:
    return {
        "thermal": capture_text(["pmset", "-g", "therm"]),
        "memory_pressure": capture_text(["memory_pressure", "-Q"]),
        "swap": capture_text(["sysctl", "vm.swapusage"]),
    }


def qwen_build_identity(
    qwen_bench: Path,
    source_commit: str,
    source_dirty: bool,
    source_state: str,
) -> dict[str, Any]:
    qwen_bench = qwen_bench.expanduser().resolve()
    process = subprocess.run(
        [str(qwen_bench), "build-info", "--allow-dirty"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        die(f"qwen-bench build-info failed: {process.stderr.strip()}")
    try:
        identity = parse_json(process.stdout)
    except (ValueError, json.JSONDecodeError) as error:
        die(f"qwen-bench build-info emitted invalid JSON: {error}")
    if not isinstance(identity, dict):
        die("qwen-bench build-info did not emit an object")
    validate_qwen_identity(
        identity,
        source_commit,
        source_dirty,
        source_state,
        allow_dirty=True,
    )
    return identity


@contextmanager
def exclusive_run_lock() -> Iterator[None]:
    RUN_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with RUN_LOCK.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            die(f"another text-capability run owns {RUN_LOCK}")
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} started={utc_now()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def terminate_and_reap(process: subprocess.Popen[str]) -> int:
    if process.poll() is None:
        try:
            process.terminate()
        except ProcessLookupError:
            pass
    while process.poll() is None:
        try:
            process.wait()
        except KeyboardInterrupt:
            continue
    assert process.returncode is not None
    return process.returncode


def run_command(
    qwen: Path,
    model: Path,
    requests: Path,
    stats: Path,
) -> list[str]:
    return [
        str(qwen),
        "--model",
        str(model),
        "--requests-jsonl",
        str(requests),
        "--execution-mode",
        "serial",
        "--model-prefetch",
        "off",
        "--request-stats",
        str(stats),
        "--tokens",
        "768",
        "--temp",
        str(SAMPLING["temperature"]),
        "--top-k",
        str(SAMPLING["top_k"]),
        "--top-p",
        str(SAMPLING["top_p"]),
        "--min-p",
        str(SAMPLING["min_p"]),
        "--seed",
        str(SEED),
        "--no-special-tokens",
        "--prefill-chunk",
        "1024",
        "--prefix-cache-max-mib",
        "0",
        "--cache-prefix-auto-min-tokens",
        "0",
    ]


def index_unique(
    rows: list[dict[str, Any]], label: str, key: str = "id"
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        request_id = row.get(key)
        if not isinstance(request_id, str) or request_id in indexed:
            die(f"{label} has invalid or duplicate {key} {request_id!r}")
        indexed[request_id] = row
    return indexed


def validate_engine_artifacts(
    requests: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
    stats: list[dict[str, Any]],
    model_record: dict[str, Any],
    source_commit: str,
    source_dirty: bool,
    source_state: str,
) -> None:
    request_order = [row.get("id") for row in requests]
    output_order = [row.get("id") for row in outputs]
    stats_order = [row.get("id") for row in stats]
    if output_order != request_order:
        die("output order differs from the frozen serial request order")
    if stats_order != request_order:
        die("stats order differs from the frozen serial request order")
    request_by_id = index_unique(requests, "requests")
    output_by_id = index_unique(outputs, "outputs")
    stats_by_id = index_unique(stats, "stats")
    expected_ids = set(request_by_id)
    for label, observed in (("outputs", output_by_id), ("stats", stats_by_id)):
        if set(observed) != expected_ids:
            missing = sorted(expected_ids - set(observed))
            extra = sorted(set(observed) - expected_ids)
            die(f"{label} IDs differ: missing={missing[:1]} extra={extra[:1]}")
    model_path = Path(model_record["locator"]["path"]).resolve()
    for request_id, request in request_by_id.items():
        output = output_by_id[request_id]
        stat = stats_by_id[request_id]
        if output.get("generated_token_sha256") != stat.get("generated_token_sha256"):
            die(f"generated-token hash mismatch for {request_id}")
        if output.get("generated_tokens") != stat.get("generated_tokens"):
            die(f"generated-token count mismatch for {request_id}")
        if output.get("prompt_tokens") != stat.get("prompt_tokens"):
            die(f"prompt-token count mismatch for {request_id}")
        if output.get("stop_reason") != stat.get("stop_reason"):
            die(f"stop-reason mismatch for {request_id}")
        if stat.get("requested_tokens") != request.get("tokens"):
            die(f"requested-token budget mismatch for {request_id}")
        if stat.get("request_stats_contract") != "qwen_jsonl_v2":
            die(f"request-stats contract mismatch for {request_id}")
        if Path(str(stat.get("model"))).resolve() != model_path:
            die(f"model path mismatch for {request_id}")
        if stat.get("build_commit") != source_commit:
            die(f"qwen build commit mismatch for {request_id}")
        if stat.get("build_dirty") is not source_dirty:
            die(f"qwen build dirty bit mismatch for {request_id}")
        if stat.get("build_source_state") != source_state:
            die(f"qwen build source state mismatch for {request_id}")
        if stat.get("model_prefetch_policy") != "off":
            die(f"model prefetch was not off for {request_id}")
        if stat.get("model_prefetch_bytes_returned") != 0:
            die(f"model prefetch returned bytes for {request_id}")
        if stat.get("cache_max_bytes") != 0 or stat.get("cache_hit") is not False:
            die(f"prefix cache was active for {request_id}")
        if output.get("thinking_partition") != stat.get("thinking_partition"):
            die(f"thinking partition mismatch for {request_id}")
        token_hash = output.get("generated_token_sha256")
        if not isinstance(token_hash, str) or not SHA256_RE.fullmatch(token_hash):
            die(f"invalid generated-token identity for {request_id}")


def verify_run_artifacts(
    output: Path,
    arm: str,
    allowed_statuses: set[str],
    *,
    require_scores: bool = False,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    packet = verify_prepared(output)
    run_path = arm_path(output, arm, ".run.json")
    run = read_json(run_path)
    if not isinstance(run, dict) or run.get("status") not in allowed_statuses:
        die(
            f"arm {arm} status is not scoreable/publishable: "
            f"{run.get('status') if isinstance(run, dict) else None!r}"
        )
    profile = run.get("profile")
    model_key = run.get("model_key")
    if profile not in packet["request_sets"] or model_key not in MODEL_SPECS:
        die(f"run metadata for arm {arm} has invalid profile/model")
    requests_path = output / packet["request_sets"][profile]["path"]
    outputs_path = arm_path(output, arm, ".outputs.jsonl")
    stats_path = arm_path(output, arm, ".stats.jsonl")
    for path in (requests_path, outputs_path, stats_path):
        if not path.is_file():
            die(f"arm {arm} is missing artifact {path}")
    expected_hashes = {
        "packet_sha256": sha256_file(output / "packet.json"),
        "requests_sha256": sha256_file(requests_path),
        "outputs_sha256": sha256_file(outputs_path),
        "stats_sha256": sha256_file(stats_path),
    }
    for field, actual in expected_hashes.items():
        if run.get(field) != actual:
            die(f"arm {arm} {field} differs from run metadata")

    source = run.get("source")
    model = run.get("model")
    if not isinstance(source, dict) or not isinstance(model, dict):
        die(f"arm {arm} lacks source/model provenance")
    source_commit = source.get("commit")
    source_dirty = source.get("dirty")
    source_state = source.get("source_state")
    if (
        not isinstance(source_commit, str)
        or type(source_dirty) is not bool
        or not isinstance(source_state, str)
    ):
        die(f"arm {arm} has malformed source provenance")
    spec = MODEL_SPECS[model_key]
    locator = model.get("locator")
    if (
        model.get("model_key") != model_key
        or model.get("family") != spec["family"]
        or model.get("display") != spec["display"]
        or model.get("producer") != spec["producer"]
        or model.get("sha256") != spec["sha256"]
        or not isinstance(locator, dict)
        or locator.get("path") != str(Path(spec["path"]).expanduser().resolve())
        or locator.get("bytes") != spec["bytes"]
    ):
        die(f"arm {arm} model provenance differs from the frozen model specification")
    build_identity = run.get("build_identity")
    if (
        not isinstance(build_identity, dict)
        or build_identity.get("build_commit") != source_commit
        or build_identity.get("runtime_commit") != source_commit
        or build_identity.get("build_source_state") != source_state
        or build_identity.get("runtime_source_state") != source_state
        or build_identity.get("build_dirty") is not source_dirty
        or build_identity.get("runtime_dirty") is not source_dirty
    ):
        die(f"arm {arm} build identity is not bound to its source record")

    requests = read_jsonl(requests_path)
    outputs = read_jsonl(outputs_path)
    stats = read_jsonl(stats_path)
    validate_engine_artifacts(
        requests,
        outputs,
        stats,
        model,
        source_commit,
        source_dirty,
        source_state,
    )
    if require_scores:
        scored_path = arm_path(output, arm, ".scored.jsonl")
        summary_path = arm_path(output, arm, ".summary.json")
        if not scored_path.is_file() or not summary_path.is_file():
            die(f"arm {arm} has no retained scores")
        if run.get("scored_sha256") != sha256_file(scored_path):
            die(f"arm {arm} scored rows differ from run metadata")
        if run.get("summary_sha256") != sha256_file(summary_path):
            die(f"arm {arm} summary differs from run metadata")
        summary = read_json(summary_path)
        if (
            not isinstance(summary, dict)
            or summary.get("arm") != arm
            or summary.get("profile") != profile
            or summary.get("model_key") != model_key
            or summary.get("packet_sha256") != expected_hashes["packet_sha256"]
            or summary.get("outputs_sha256") != expected_hashes["outputs_sha256"]
            or summary.get("stats_sha256") != expected_hashes["stats_sha256"]
        ):
            die(f"arm {arm} summary provenance is inconsistent")
    return run, packet, requests, outputs, stats


def run_arm(
    output: Path,
    profile: str,
    arm: str,
    model_key: str,
    qwen: Path,
    qwen_bench: Path,
    force: bool,
) -> None:
    with exclusive_run_lock():
        _run_arm_locked(
            output,
            profile,
            arm,
            model_key,
            qwen,
            qwen_bench,
            force,
        )


def _run_arm_locked(
    output: Path,
    profile: str,
    arm: str,
    model_key: str,
    qwen: Path,
    qwen_bench: Path,
    force: bool,
) -> None:
    packet = verify_prepared(output)
    if profile not in packet["request_sets"]:
        die(f"unknown profile {profile!r}")
    if model_key not in MODEL_SPECS:
        die(f"unknown model key {model_key!r}")
    if profile == "effort" and MODEL_SPECS[model_key]["family"] != "qwen38":
        die("the effort profile requires a validated Qwen3.8 model")
    requests_path = output / packet["request_sets"][profile]["path"]
    requests = read_jsonl(requests_path)
    outputs_path = arm_path(output, arm, ".outputs.jsonl")
    stats_path = arm_path(output, arm, ".stats.jsonl")
    stderr_path = arm_path(output, arm, ".stderr.log")
    run_path = arm_path(output, arm, ".run.json")
    scored_path = arm_path(output, arm, ".scored.jsonl")
    summary_path = arm_path(output, arm, ".summary.json")
    existing = [
        path
        for path in (
            outputs_path,
            stats_path,
            stderr_path,
            run_path,
            scored_path,
            summary_path,
        )
        if path.exists()
    ]
    if existing and not force:
        die(f"refusing to replace artifacts for arm {arm!r}; use --force")
    for path in existing:
        path.unlink()

    qwen = qwen.expanduser().resolve()
    qwen_bench = qwen_bench.expanduser().resolve()
    model_record = verified_model_identity(output, model_key)
    model = Path(model_record["locator"]["path"])
    source_commit, source_dirty, source_state = source_identity(ROOT)
    build_identity = qwen_build_identity(
        qwen_bench, source_commit, source_dirty, source_state
    )
    qwen_record = binary_identity(qwen)
    qwen_bench_record = binary_identity(qwen_bench)
    environment, environment_record = child_environment()
    command = run_command(qwen, model, requests_path, stats_path)
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "battery_id": BATTERY_ID,
        "status": "running",
        "profile": profile,
        "arm": arm,
        "model_key": model_key,
        "started_at": utc_now(),
        "source": {
            "commit": source_commit,
            "dirty": source_dirty,
            "source_state": source_state,
        },
        "build_identity": build_identity,
        "qwen_binary": qwen_record,
        "qwen_bench_binary": qwen_bench_record,
        "model": model_record,
        "command": command,
        "environment": environment_record,
        "packet_sha256": sha256_file(output / "packet.json"),
        "requests_sha256": sha256_file(requests_path),
        "expected_rows": len(requests),
        "safety": packet["safety"],
        "host_before": host_context(),
    }
    write_json(run_path, metadata)
    completed = 0
    parse_errors: list[str] = []
    started = time.perf_counter()
    process: subprocess.Popen[str] | None = None
    # The public entry point owns the lock across cache mutation, artifact
    # replacement, execution, scoring, and the final metadata transition.
    with nullcontext():
        try:
            with (
                outputs_path.open("w") as output_handle,
                stderr_path.open("w") as stderr_handle,
            ):
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=stderr_handle,
                    text=True,
                    encoding="utf-8",
                )
                metadata["child_pid"] = process.pid
                write_json(run_path, metadata)
                assert process.stdout is not None
                for line in process.stdout:
                    output_handle.write(line)
                    output_handle.flush()
                    try:
                        row = parse_json(line)
                        request_id = row.get("id") if isinstance(row, dict) else None
                    except (ValueError, json.JSONDecodeError) as error:
                        parse_errors.append(str(error))
                        request_id = "invalid-json"
                    completed += 1
                    print(
                        f"[{completed:02d}/{len(requests)}] {arm} {request_id}",
                        flush=True,
                    )
                returncode = process.wait()
        except BaseException as error:
            child_returncode = (
                terminate_and_reap(process) if process is not None else None
            )
            metadata.update(
                {
                    "status": "interrupted"
                    if isinstance(error, KeyboardInterrupt)
                    else "failed",
                    "finished_at": utc_now(),
                    "outer_wall_s": time.perf_counter() - started,
                    "completed_rows": completed,
                    "error_type": type(error).__name__,
                    "child_returncode": child_returncode,
                    "child_shutdown": "cooperative_sigterm_and_wait"
                    if process is not None
                    else "not_started",
                }
            )
            write_json(run_path, metadata)
            raise

    source_after = source_identity(ROOT)
    metadata.update(
        {
            "finished_at": utc_now(),
            "outer_wall_s": time.perf_counter() - started,
            "completed_rows": completed,
            "child_returncode": returncode,
            "parse_errors": parse_errors,
            "source_after": {
                "commit": source_after[0],
                "dirty": source_after[1],
                "source_state": source_after[2],
            },
            "host_after": host_context(),
            "stderr_sha256": sha256_file(stderr_path),
            "outputs_sha256": sha256_file(outputs_path),
            "stats_sha256": sha256_file(stats_path) if stats_path.is_file() else None,
        }
    )
    if (
        returncode != 0
        or completed != len(requests)
        or parse_errors
        or source_after != (source_commit, source_dirty, source_state)
        or not stats_path.is_file()
    ):
        metadata["status"] = "failed"
        write_json(run_path, metadata)
        die(
            f"arm {arm} failed: exit={returncode} rows={completed}/{len(requests)} "
            f"parse_errors={len(parse_errors)} source_changed={source_after != (source_commit, source_dirty, source_state)}"
        )
    outputs = read_jsonl(outputs_path)
    stats = read_jsonl(stats_path)
    validate_engine_artifacts(
        requests,
        outputs,
        stats,
        model_record,
        source_commit,
        source_dirty,
        source_state,
    )
    metadata["status"] = "scoring"
    write_json(run_path, metadata)
    try:
        score_arm(output, arm, force=False)
    except BaseException as error:
        metadata = read_json(run_path)
        metadata["status"] = "score_failed"
        metadata["score_error_type"] = type(error).__name__
        write_json(run_path, metadata)
        raise
    metadata = read_json(run_path)
    metadata["status"] = "complete"
    write_json(run_path, metadata)


FORBIDDEN_AST = (
    ast.Import,
    ast.ImportFrom,
    ast.ClassDef,
    ast.AsyncFunctionDef,
    ast.Global,
    ast.Nonlocal,
    ast.Await,
    ast.Yield,
    ast.YieldFrom,
)
FORBIDDEN_NAMES = {
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "globals",
    "help",
    "input",
    "locals",
    "memoryview",
    "open",
    "print",
    "setattr",
    "vars",
}


def validate_candidate_source(
    source: str, function_name: str
) -> tuple[str | None, str | None]:
    if not source.strip():
        return None, "empty_response"
    if source != source.strip():
        source = source.strip()
    try:
        module = ast.parse(source, mode="exec")
    except SyntaxError as error:
        return None, f"syntax_error:{error.msg}"
    if len(module.body) != 1 or not isinstance(module.body[0], ast.FunctionDef):
        return None, "module_must_contain_exactly_one_function"
    function = module.body[0]
    if function.name != function_name:
        return None, f"wrong_function:{function.name}"
    if function.decorator_list:
        return None, "decorators_forbidden"
    for node in ast.walk(module):
        if isinstance(node, FORBIDDEN_AST):
            return None, f"forbidden_ast:{type(node).__name__}"
        if isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES or "__" in node.id:
                return None, f"forbidden_name:{node.id}"
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            return None, f"private_attribute:{node.attr}"
    return ast.unparse(module), None


def code_function_name(task_id: str) -> str:
    return {
        "c1": "merge_spans",
        "c2": "split_escaped",
        "c3": "deep_get",
        "c4": "topo_layers",
        "c5": "window_max",
        "c6": "apply_patch",
        "c7": "ledger_balance",
        "c8": "minimal_cover",
        "c9": "natural_compare",
        "c10": "render_template",
    }[task_id]


def code_special_test(task_id: str) -> str:
    if task_id == "c5":
        return """
values = [(index * 7919) % 100003 for index in range(200000)]
k = 257
expected = []
queue = []
head = 0
for index, value in enumerate(values):
    while head < len(queue) and queue[head] <= index - k:
        head += 1
    while len(queue) > head and values[queue[-1]] <= value:
        queue.pop()
    queue.append(index)
    if index >= k - 1:
        expected.append(values[queue[head]])
started = time.process_time()
actual = fn(values, k)
elapsed = time.process_time() - started
if not strict_equal(actual, expected):
    raise AssertionError("large linear case returned wrong output")
if elapsed > 2.5:
    raise AssertionError(f"large linear case exceeded 2.5 CPU seconds: {elapsed:.3f}s")
special_tests += 1
"""
    if task_id == "c6":
        return """
doc = {"x": []}
value = [1]
ops = [("set", ["x"], value)]
actual = fn(doc, ops)
if not strict_equal(actual, {"x": [1]}) or not strict_equal(doc, {"x": []}) or not strict_equal(value, [1]):
    raise AssertionError("deep-copy contract failed")
actual["x"].append(2)
if value != [1] or ops != [("set", ["x"], [1])]:
    raise AssertionError("result aliases operation value")
special_tests += 1
"""
    return ""


def run_code_grader_process(
    script: Path,
    directory: str,
    *,
    timeout_s: float = 7.0,
    max_rss_kib: int = 512 * 1024,
) -> dict[str, Any]:
    process = subprocess.Popen(
        [sys.executable, "-I", str(script)],
        cwd=directory,
        env={"PYTHONHASHSEED": "0", "LANG": "C", "LC_ALL": "C"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    started = time.monotonic()
    failure: str | None = None
    while process.poll() is None:
        if time.monotonic() - started > timeout_s:
            failure = "test_timeout"
            break
        rss_probe = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(process.pid)],
            capture_output=True,
            text=True,
            check=False,
        )
        if rss_probe.returncode != 0:
            if process.poll() is None:
                failure = "rss_probe_failed"
                break
        else:
            try:
                rss_kib = int(rss_probe.stdout.strip())
            except ValueError:
                failure = "rss_probe_invalid"
                break
            if rss_kib > max_rss_kib:
                failure = f"rss_limit_exceeded:{rss_kib}>{max_rss_kib}"
                break
        time.sleep(0.01)
    if failure is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    stdout, stderr = process.communicate()
    return {
        "returncode": process.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "failure": failure,
    }


def grade_code(task_id: str, response: str) -> dict[str, Any]:
    function_name = code_function_name(task_id)
    source, validation_error = validate_candidate_source(response, function_name)
    if source is None:
        return {
            "format_compliant": False,
            "semantic_correct": False,
            "error": validation_error,
            "tests_run": 0,
        }
    cases = code_cases(task_id)
    special = code_special_test(task_id)
    child = f"""\
import builtins
import copy
import json
import resource
import time
import traceback

limit_failures = []
for limit_name, soft, hard in [
    ("RLIMIT_CPU", 4, 4),
    ("RLIMIT_FSIZE", 1024 * 1024, 1024 * 1024),
    ("RLIMIT_NOFILE", 32, 32),
]:
    if not hasattr(resource, limit_name):
        limit_failures.append(f"missing:{{limit_name}}")
        continue
    try:
        resource.setrlimit(getattr(resource, limit_name), (soft, hard))
        observed = resource.getrlimit(getattr(resource, limit_name))
        if observed[0] != soft or observed[1] != hard:
            limit_failures.append(f"mismatch:{{limit_name}}:{{observed!r}}")
    except (OSError, ValueError) as error:
        limit_failures.append(f"failed:{{limit_name}}:{{error}}")
if limit_failures:
    print(json.dumps({{"ok": False, "tests_run": 0, "error_type": "SandboxLimitError", "error": ";".join(limit_failures)}}))
    raise SystemExit(0)

def strict_equal(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return len(actual) == len(expected) and all(
            key in actual and type(next(candidate for candidate in actual if candidate == key)) is type(key)
            and strict_equal(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(
            strict_equal(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected

safe_names = [
    "Exception", "ValueError", "TypeError", "abs", "all", "any", "bool",
    "dict", "enumerate", "filter", "float", "int", "isinstance", "iter",
    "len", "list", "map", "max", "min", "next", "range", "reversed",
    "set", "sorted", "str", "sum", "tuple", "type", "zip"
]
safe_builtins = {{name: getattr(builtins, name) for name in safe_names}}
namespace = {{"__builtins__": safe_builtins}}
source = {source!r}
cases = {cases!r}
try:
    exec(compile(source, "<candidate>", "exec"), namespace, namespace)
    fn = namespace[{function_name!r}]
    tests_run = 0
    special_tests = 0
    for case_index, (arguments, expected) in enumerate(cases):
        before = copy.deepcopy(arguments)
        actual = fn(*arguments)
        if not strict_equal(actual, expected):
            raise AssertionError(
                f"case {{case_index}} mismatch: expected={{expected!r}} actual={{actual!r}}"
            )
        if not strict_equal(arguments, before):
            raise AssertionError(f"case {{case_index}} mutated its inputs")
        tests_run += 1
{special}
    print(json.dumps({{"ok": True, "tests_run": tests_run + special_tests}}))
except BaseException as error:
    print(json.dumps({{
        "ok": False,
        "tests_run": locals().get("tests_run", 0),
        "error_type": type(error).__name__,
        "error": str(error)[:1000],
    }}))
"""
    with tempfile.TemporaryDirectory(prefix="qwen-code-grade-") as directory:
        script = Path(directory) / "grade.py"
        script.write_text(child)
        process = run_code_grader_process(script, directory)
        if process["failure"] is not None:
            return {
                "format_compliant": True,
                "semantic_correct": False,
                "error": process["failure"],
                "tests_run": 0,
            }
    lines = [line for line in process["stdout"].splitlines() if line.strip()]
    if process["returncode"] != 0 or len(lines) != 1:
        return {
            "format_compliant": True,
            "semantic_correct": False,
            "error": f"test_process_failed:{process['returncode']}",
            "stderr": process["stderr"][-1000:],
            "tests_run": 0,
        }
    try:
        report = parse_json(lines[0])
    except (ValueError, json.JSONDecodeError) as error:
        return {
            "format_compliant": True,
            "semantic_correct": False,
            "error": f"invalid_test_report:{error}",
            "tests_run": 0,
        }
    return {
        "format_compliant": True,
        "semantic_correct": report.get("ok") is True,
        "error": report.get("error"),
        "error_type": report.get("error_type"),
        "tests_run": report.get("tests_run", 0),
    }


def split_generated_response(
    generated_text: str,
    mode: str,
    stop_reason: str,
    generated_tokens: int,
    partition: Any,
) -> dict[str, Any]:
    if mode == "no-thinking":
        protocol_ok = partition is None and "</think>" not in generated_text
        return {
            "hidden_text": "",
            "visible_text": generated_text,
            "protocol_separator_removed": False,
            "protocol_complete": protocol_ok,
            "reasoning_tokens": 0,
            "delimiter_tokens": 0,
            "visible_tokens": generated_tokens,
            "stop_complete": stop_reason == "eos",
        }
    hidden, separator, visible = generated_text.partition("</think>")
    protocol_separator_removed = separator == "</think>" and visible.startswith("\n\n")
    if protocol_separator_removed:
        visible = visible[2:]
    valid_partition = (
        separator == "</think>"
        and isinstance(partition, dict)
        and partition.get("delimiter") == "</think>"
        and partition.get("delimiter_token_aligned") is True
        and type(partition.get("reasoning_tokens")) is int
        and type(partition.get("delimiter_tokens")) is int
        and type(partition.get("visible_tokens")) is int
        and partition["reasoning_tokens"]
        + partition["delimiter_tokens"]
        + partition["visible_tokens"]
        == generated_tokens
    )
    return {
        "hidden_text": hidden,
        "visible_text": visible if separator else "",
        "protocol_separator_removed": protocol_separator_removed,
        "protocol_complete": valid_partition,
        "reasoning_tokens": partition.get("reasoning_tokens")
        if isinstance(partition, dict)
        else None,
        "delimiter_tokens": partition.get("delimiter_tokens")
        if isinstance(partition, dict)
        else None,
        "visible_tokens": partition.get("visible_tokens")
        if isinstance(partition, dict)
        else None,
        "stop_complete": stop_reason == "eos",
    }


def grade_visible(task: Task, visible: str) -> dict[str, Any]:
    if task.grader == "python_tests":
        return grade_code(task.task_id, visible)
    if task.grader == "exact":
        format_compliant = visible == str(task.expected)
        return {
            "format_compliant": format_compliant,
            "semantic_correct": format_compliant,
            "expected": task.expected,
        }
    if task.grader == "strict_json":
        try:
            parsed = parse_json(visible)
            format_compliant = visible == visible.strip()
            semantic_correct = canonical_json(parsed) == canonical_json(task.expected)
            error = None
        except (ValueError, json.JSONDecodeError) as parse_error:
            parsed = None
            format_compliant = False
            semantic_correct = False
            error = str(parse_error)
        return {
            "format_compliant": format_compliant,
            "semantic_correct": semantic_correct,
            "expected": task.expected,
            "parsed": parsed,
            "error": error,
        }
    raise ValueError(f"unknown grader {task.grader}")


def score_rows(
    packet: dict[str, Any],
    profile: str,
    arm: str,
    model_key: str,
    output_rows: list[dict[str, Any]],
    stats_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tasks = {task.task_id: task for task in build_tasks()}
    outputs = index_unique(output_rows, "outputs")
    stats = index_unique(stats_rows, "stats")
    request_rows = read_jsonl(
        Path(packet["_output_dir"]) / packet["request_sets"][profile]["path"]
    )
    expected_ids = [row["id"] for row in request_rows]
    if set(outputs) != set(expected_ids) or set(stats) != set(expected_ids):
        die(f"arm {arm} scoring IDs differ from prepared {profile} requests")
    scored: list[dict[str, Any]] = []
    for request_id in expected_ids:
        if profile == "capability":
            task_id, mode = request_id, "no-thinking"
        else:
            task_id, mode = request_id.rsplit("--", 1)
        task = tasks[task_id]
        output = outputs[request_id]
        stat = stats[request_id]
        split = split_generated_response(
            output["generated_text"],
            mode,
            output["stop_reason"],
            output["generated_tokens"],
            output.get("thinking_partition"),
        )
        grade = grade_visible(task, split["visible_text"])
        complete = split["protocol_complete"] and split["stop_complete"]
        correct = bool(
            complete and grade["format_compliant"] and grade["semantic_correct"]
        )
        scored.append(
            {
                "schema_version": 1,
                "battery_id": BATTERY_ID,
                "profile": profile,
                "arm": arm,
                "model_key": model_key,
                "request_id": request_id,
                "task_id": task_id,
                "category": task.category,
                "mode": mode,
                "complete": complete,
                "correct": correct,
                "format_compliant": bool(grade["format_compliant"]),
                "semantic_correct": bool(grade["semantic_correct"]),
                "hidden_text": split["hidden_text"],
                "visible_text": split["visible_text"],
                "reasoning_tokens": split["reasoning_tokens"],
                "delimiter_tokens": split["delimiter_tokens"],
                "visible_tokens": split["visible_tokens"],
                "grade": grade,
                "engine_output": output,
                "engine_stats": stat,
            }
        )
    return scored, summarize_rows(scored, profile, arm, model_key)


def median(values: list[float | int]) -> float | None:
    return statistics.median(values) if values else None


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    generated = [row["engine_output"]["generated_tokens"] for row in rows]
    reasoning = [
        row["reasoning_tokens"] for row in rows if row["reasoning_tokens"] is not None
    ]
    visible = [
        row["visible_tokens"] for row in rows if row["visible_tokens"] is not None
    ]
    totals = [row["engine_stats"]["total_ms"] for row in rows]
    ttft = [row["engine_stats"]["model_ttft_ms"] for row in rows]
    correct = sum(row["correct"] for row in rows)
    generated_total = sum(generated)
    return {
        "tasks": len(rows),
        "correct": correct,
        "complete": sum(row["complete"] for row in rows),
        "format_compliant": sum(row["format_compliant"] for row in rows),
        "semantic_correct": sum(row["semantic_correct"] for row in rows),
        "generated_tokens_total": generated_total,
        "generated_tokens_median": median(generated),
        "reasoning_tokens_available": len(reasoning),
        "reasoning_tokens_total": sum(reasoning)
        if len(reasoning) == len(rows)
        else None,
        "reasoning_tokens_median": median(reasoning),
        "visible_tokens_total": sum(visible) if len(visible) == len(rows) else None,
        "visible_tokens_median": median(visible),
        "total_ms_sum": sum(totals),
        "total_ms_median": median(totals),
        "model_ttft_ms_median": median(ttft),
        "correct_per_1000_generated_tokens": (
            correct * 1000.0 / generated_total if generated_total else 0.0
        ),
        "failed_tasks": [row["task_id"] for row in rows if not row["correct"]],
    }


def summarize_rows(
    rows: list[dict[str, Any]], profile: str, arm: str, model_key: str
) -> dict[str, Any]:
    categories = sorted({row["category"] for row in rows})
    modes = [mode for mode in MODES if any(row["mode"] == mode for row in rows)]
    return {
        "schema_version": 1,
        "battery_id": BATTERY_ID,
        "profile": profile,
        "arm": arm,
        "model_key": model_key,
        "interpretation": (
            "fixed no-thinking text surface; report categories separately"
            if profile == "capability"
            else "single interleaved policy discovery pass; not promotion-grade timing"
        ),
        "overall": summarize_group(rows),
        "by_category": {
            category: summarize_group(
                [row for row in rows if row["category"] == category]
            )
            for category in categories
        },
        "by_mode": {
            mode: summarize_group([row for row in rows if row["mode"] == mode])
            for mode in modes
        },
        "output_token_stream_sha256": sha256_bytes(
            b"".join(
                bytes.fromhex(row["engine_output"]["generated_token_sha256"])
                for row in rows
            )
        ),
    }


def score_arm(output: Path, arm: str, force: bool) -> None:
    run, packet, _, output_rows, stats_rows = verify_run_artifacts(
        output, arm, {"scoring", "complete"}
    )
    profile = run.get("profile")
    model_key = run.get("model_key")
    scored_path = arm_path(output, arm, ".scored.jsonl")
    summary_path = arm_path(output, arm, ".summary.json")
    existing = [path for path in (scored_path, summary_path) if path.exists()]
    if existing and not force:
        die(f"refusing to replace scores for arm {arm}; use --force")
    for comparison_path in output.glob("*-comparison.json"):
        comparison_path.unlink()
    packet_for_score = dict(packet)
    packet_for_score["_output_dir"] = str(output)
    scored, summary = score_rows(
        packet_for_score,
        profile,
        arm,
        model_key,
        output_rows,
        stats_rows,
    )
    summary["packet_sha256"] = sha256_file(output / "packet.json")
    summary["outputs_sha256"] = sha256_file(arm_path(output, arm, ".outputs.jsonl"))
    summary["stats_sha256"] = sha256_file(arm_path(output, arm, ".stats.jsonl"))
    write_jsonl(scored_path, scored)
    write_json(summary_path, summary)
    run["scored_sha256"] = sha256_file(scored_path)
    run["summary_sha256"] = sha256_file(summary_path)
    run["scored_at"] = utc_now()
    write_json(arm_path(output, arm, ".run.json"), run)
    overall = summary["overall"]
    print(
        f"{arm}: {overall['correct']}/{overall['tasks']} correct, "
        f"{overall['complete']}/{overall['tasks']} complete"
    )
    for category, row in summary["by_category"].items():
        print(f"  {category}: {row['correct']}/{row['tasks']}")
    for mode, row in summary["by_mode"].items():
        print(
            f"  {mode}: {row['correct']}/{row['tasks']} "
            f"tokens={row['generated_tokens_total']} wall_ms={row['total_ms_sum']:.1f}"
        )


def compare_arms(output: Path, profile: str, arms: list[str], force: bool) -> None:
    if len(set(arms)) != len(arms):
        die("comparison arms must be unique")
    if len(arms) < 2 and profile == "capability":
        die("capability comparison requires at least two arms")
    runs = {
        arm: verify_run_artifacts(output, arm, {"complete"}, require_scores=True)[0]
        for arm in arms
    }
    runtime_identities = {
        canonical_json(
            {
                "source": runs[arm].get("source"),
                "build_identity": runs[arm].get("build_identity"),
                "qwen_binary_sha256": runs[arm].get("qwen_binary", {}).get("sha256"),
                "qwen_bench_binary_sha256": runs[arm]
                .get("qwen_bench_binary", {})
                .get("sha256"),
            }
        )
        for arm in arms
    }
    if len(runtime_identities) != 1:
        die("comparison arms do not share one source/build/binary identity")
    model_keys = [runs[arm]["model_key"] for arm in arms]
    if len(set(model_keys)) != len(model_keys):
        die("comparison arms must use distinct model keys")
    if profile == "capability" and len(arms) == len(MODEL_SPECS):
        if set(model_keys) != set(MODEL_SPECS):
            die("three-arm capability comparison requires all frozen model keys")
    summaries = {arm: read_json(arm_path(output, arm, ".summary.json")) for arm in arms}
    scored_by_arm = {
        arm: index_unique(
            read_jsonl(arm_path(output, arm, ".scored.jsonl")),
            arm,
            "request_id",
        )
        for arm in arms
    }
    for arm, summary in summaries.items():
        if summary.get("profile") != profile:
            die(f"arm {arm} is not profile {profile}")
    ids = set(next(iter(scored_by_arm.values())))
    if any(set(rows) != ids for rows in scored_by_arm.values()):
        die("comparison arms have different request IDs")
    pairwise: list[dict[str, Any]] = []
    for left_index, left in enumerate(arms):
        for right in arms[left_index + 1 :]:
            disagreements = []
            token_stream_matches = 0
            visible_matches = 0
            for request_id in sorted(ids):
                left_row = scored_by_arm[left][request_id]
                right_row = scored_by_arm[right][request_id]
                if left_row["correct"] != right_row["correct"]:
                    disagreements.append(
                        {
                            "request_id": request_id,
                            "left_correct": left_row["correct"],
                            "right_correct": right_row["correct"],
                        }
                    )
                if (
                    left_row["engine_output"]["generated_token_sha256"]
                    == right_row["engine_output"]["generated_token_sha256"]
                ):
                    token_stream_matches += 1
                if left_row["visible_text"] == right_row["visible_text"]:
                    visible_matches += 1
            pairwise.append(
                {
                    "left": left,
                    "right": right,
                    "correctness_disagreements": disagreements,
                    "token_stream_matches": token_stream_matches,
                    "visible_text_matches": visible_matches,
                    "tasks": len(ids),
                }
            )
    comparison = {
        "schema_version": 1,
        "battery_id": BATTERY_ID,
        "profile": profile,
        "arms": arms,
        "summary_sha256": {
            arm: sha256_file(arm_path(output, arm, ".summary.json")) for arm in arms
        },
        "scored_sha256": {
            arm: sha256_file(arm_path(output, arm, ".scored.jsonl")) for arm in arms
        },
        "summaries": summaries,
        "pairwise": pairwise,
    }
    path = output / f"{profile}-comparison.json"
    if path.exists() and not force:
        die(f"refusing to replace {path}; use --force")
    write_json(path, comparison)
    print(f"wrote {path}")


def publish(output: Path, destination: Path, arms: list[str], force: bool) -> None:
    verify_prepared(output)
    destination = destination.resolve()
    if destination.exists() and not destination.is_dir():
        die(f"publish destination is not a directory: {destination}")
    if destination.exists() and any(destination.iterdir()) and not force:
        die(
            f"refusing to replace nonempty publish directory {destination}; use --force"
        )
    if len(set(arms)) != len(arms):
        die("publish arms must be unique")
    runs = {
        arm: verify_run_artifacts(output, arm, {"complete"}, require_scores=True)[0]
        for arm in arms
    }
    runtime_identities = {
        canonical_json(
            {
                "source": runs[arm].get("source"),
                "build_identity": runs[arm].get("build_identity"),
                "qwen_binary_sha256": runs[arm].get("qwen_binary", {}).get("sha256"),
                "qwen_bench_binary_sha256": runs[arm]
                .get("qwen_bench_binary", {})
                .get("sha256"),
            }
        )
        for arm in arms
    }
    if len(runtime_identities) != 1:
        die("published arms do not share one source/build/binary identity")
    arms_by_profile: dict[str, list[str]] = {}
    for arm in arms:
        arms_by_profile.setdefault(runs[arm]["profile"], []).append(arm)
    capability_arms = arms_by_profile.get("capability", [])
    if capability_arms and {runs[arm]["model_key"] for arm in capability_arms} != set(
        MODEL_SPECS
    ):
        die("published capability packet requires all three frozen model keys")
    for arm in arms_by_profile.get("effort", []):
        if MODEL_SPECS[runs[arm]["model_key"]]["family"] != "qwen38":
            die("published effort arms must use Qwen3.8 models")

    comparison_paths: list[Path] = []
    for profile, profile_arms in arms_by_profile.items():
        comparison_path = output / f"{profile}-comparison.json"
        comparison = read_json(comparison_path)
        expected_summary_hashes = {
            arm: sha256_file(arm_path(output, arm, ".summary.json"))
            for arm in profile_arms
        }
        expected_scored_hashes = {
            arm: sha256_file(arm_path(output, arm, ".scored.jsonl"))
            for arm in profile_arms
        }
        if (
            not isinstance(comparison, dict)
            or comparison.get("battery_id") != BATTERY_ID
            or comparison.get("profile") != profile
            or comparison.get("arms") != profile_arms
            or comparison.get("summary_sha256") != expected_summary_hashes
            or comparison.get("scored_sha256") != expected_scored_hashes
        ):
            die(f"{profile} comparison does not match the selected publish arms")
        comparison_paths.append(comparison_path)
    selected = [
        output / "packet.json",
        output / "requests-capability.jsonl",
        output / "requests-effort.jsonl",
        output / "model-identities.json",
    ]
    for arm in arms:
        selected.extend(
            arm_path(output, arm, suffix)
            for suffix in (
                ".outputs.jsonl",
                ".stats.jsonl",
                ".stderr.log",
                ".run.json",
                ".scored.jsonl",
                ".summary.json",
            )
        )
    selected.extend(comparison_paths)
    selected = sorted(set(selected))
    missing = [path for path in selected if not path.is_file()]
    if missing:
        die(f"cannot publish; first missing artifact: {missing[0]}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.publish-", dir=destination.parent)
    )
    backup = destination.parent / f".{destination.name}.previous-{os.getpid()}"
    try:
        for source in selected:
            shutil.copy2(source, staging / source.name)
        manifest = {
            "schema_version": 1,
            "battery_id": BATTERY_ID,
            "published_at": utc_now(),
            "arms": arms,
            "model_keys": {arm: runs[arm]["model_key"] for arm in arms},
            "files": {
                path.name: {
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in selected
            },
        }
        write_json(staging / "artifact-manifest.json", manifest)
        if backup.exists():
            shutil.rmtree(backup)
        if destination.exists():
            os.replace(destination, backup)
        os.replace(staging, destination)
        if backup.exists():
            shutil.rmtree(backup)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        if backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    print(f"published {len(selected) + 1} artifacts to {destination}")


def check_contract() -> None:
    packet, request_bytes = verify_contract()
    print(f"battery {BATTERY_ID}")
    print(f"packet {semantic_sha256(packet)}")
    print(f"tasks {len(packet['tasks'])}")
    for profile, content in request_bytes.items():
        print(
            f"{profile} requests {packet['request_sets'][profile]['count']} "
            f"{sha256_bytes(content)}"
        )
    for task in build_tasks():
        if task.category == "code":
            cases = code_cases(task.task_id)
            if not cases:
                raise RuntimeError(f"no code cases for {task.task_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check")

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    prepare_parser.add_argument("--force", action="store_true")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    run_parser.add_argument(
        "--profile", choices=("capability", "effort"), required=True
    )
    run_parser.add_argument("--arm", required=True)
    run_parser.add_argument("--model-key", choices=tuple(MODEL_SPECS), required=True)
    run_parser.add_argument("--qwen", type=Path, default=DEFAULT_QWEN)
    run_parser.add_argument("--qwen-bench", type=Path, default=DEFAULT_QWEN_BENCH)
    run_parser.add_argument("--force", action="store_true")

    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    score_parser.add_argument("--arm", required=True)
    score_parser.add_argument("--force", action="store_true")

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    compare_parser.add_argument(
        "--profile", choices=("capability", "effort"), required=True
    )
    compare_parser.add_argument("--arms", nargs="+", required=True)
    compare_parser.add_argument("--force", action="store_true")

    publish_parser = subparsers.add_parser("publish")
    publish_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    publish_parser.add_argument("--destination", type=Path, required=True)
    publish_parser.add_argument("--arms", nargs="+", required=True)
    publish_parser.add_argument("--force", action="store_true")

    args = parser.parse_args()
    if args.command == "check":
        check_contract()
    elif args.command == "prepare":
        with exclusive_run_lock():
            prepare(args.output_dir.resolve(), args.force)
    elif args.command == "run":
        run_arm(
            args.output_dir.resolve(),
            args.profile,
            args.arm,
            args.model_key,
            args.qwen,
            args.qwen_bench,
            args.force,
        )
    elif args.command == "score":
        with exclusive_run_lock():
            score_arm(args.output_dir.resolve(), args.arm, args.force)
    elif args.command == "compare":
        with exclusive_run_lock():
            compare_arms(args.output_dir.resolve(), args.profile, args.arms, args.force)
    elif args.command == "publish":
        with exclusive_run_lock():
            publish(args.output_dir.resolve(), args.destination, args.arms, args.force)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
