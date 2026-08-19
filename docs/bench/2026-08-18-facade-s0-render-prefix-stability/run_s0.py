# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""S0 runner: render prefix stability and durable-checkpoint economics.

One stateless `qwen` process per turn, per cell. See README.md in this
directory for the preregistered matrix, predictions, and decision rules.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PACKET_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKET_DIR.parents[2]
DEFAULT_BINARY = REPO_ROOT / "target" / "release" / "qwen"
DEFAULT_MODELS_DIR = Path.home() / "models"

MODELS = {
    "smoke": "Qwen3.5-0.8B-Q8_0.gguf",
    "a3b": "Qwen3.6-35B-A3B-UD-Q4_K_S.gguf",
    "q38": "Qwen3.8-27B-Q4_K_M.gguf",
}

CELLS = {
    "smoke": ("smoke", "preserve", "scripted", 3),
    "smoke-strip": ("smoke", "strip", "scripted", 3),
    "a3b-preserve-scripted": ("a3b", "preserve", "scripted", 10),
    "a3b-strip-scripted": ("a3b", "strip", "scripted", 10),
    "q38-preserve-scripted": ("q38", "preserve", "scripted", 10),
    "q38-strip-scripted": ("q38", "strip", "scripted", 10),
    "a3b-preserve-live": ("a3b", "preserve", "live", 5),
    "a3b-strip-live": ("a3b", "strip", "live", 5),
}
PRIMARY = [name for name in CELLS if name != "smoke"]

SYSTEM_PROMPT = """You are Sable, an autonomous software engineering agent operating inside a
sandboxed repository workspace. You investigate defects, propose minimal
patches, and verify them with the project's own test suite before reporting.
You never invent file contents you have not read, never fabricate tool
results, and never claim a test passed without running it in this session.

## Operating loop

Work in strict observe-orient-act cycles. In every assistant turn you first
reason about the latest evidence inside a thinking block, then state one
short conclusion in plain prose, then issue exactly one tool call. You stop
issuing tool calls only when the task's acceptance criteria are met, at
which point you produce a final report instead of a tool call. Keep prose
outside the thinking block terse: one or two sentences before the call.

## Tools

You may call exactly one tool per turn using a `<tool_call>` block whose
body is a single JSON object with `name` and `arguments` keys. The
available tools and their argument schemas are:

{"tools": [
  {"name": "fs.list", "description": "List directory entries with sizes",
   "arguments": {"type": "object", "properties": {"path": {"type":
   "string", "description": "Repository-relative directory path"}},
   "required": ["path"]}},
  {"name": "fs.read", "description": "Read a UTF-8 text file",
   "arguments": {"type": "object", "properties": {"path": {"type":
   "string"}, "start_line": {"type": "integer", "minimum": 1},
   "end_line": {"type": "integer"}}, "required": ["path"]}},
  {"name": "fs.write", "description": "Replace an exact string in a file",
   "arguments": {"type": "object", "properties": {"path": {"type":
   "string"}, "old": {"type": "string"}, "new": {"type": "string"}},
   "required": ["path", "old", "new"]}},
  {"name": "shell.run", "description": "Run a shell command in the
   workspace and return stdout, stderr, and exit code", "arguments":
   {"type": "object", "properties": {"cmd": {"type": "string"},
   "timeout_s": {"type": "integer", "default": 120}}, "required":
   ["cmd"]}}
]}

Tool results arrive in the next user turn as a JSON object with keys
`tool`, `ok`, and `result`. Treat them as ground truth for this session.
If `ok` is false, diagnose before retrying; do not repeat a failed call
unchanged more than once.

## Constraints

- Minimal diffs: change the fewest lines that fix the defect. Preserve
  formatting, comment style, and import order of the surrounding code.
- Evidence discipline: quote line numbers and exact strings from files you
  have read in this session when justifying a patch. If two hypotheses
  remain, name the discriminating observation you will gather next.
- Test discipline: after any write, rerun the narrowest failing test
  first, then the file's full test module, then the suite if both pass.
- Safety: never run commands that leave the workspace (no network, no
  package installation, no git push). Destructive commands require the
  defect report to have explicitly authorized them.
- Reporting: the final report states the root cause in one sentence, lists
  each file changed with a one-line rationale, and cites the passing test
  command output that verifies the fix.

## Failure protocol

If a tool result is malformed JSON, missing required keys, or reports an
infrastructure fault rather than a domain result, say so explicitly, cite
the offending fragment verbatim, and issue one diagnostic call (`fs.list`
of the working directory or `shell.run` of `pwd`) before resuming the
plan. If the same infrastructure fault repeats twice, halt and produce an
incident note instead of a report: what you attempted, the exact error
text, and the smallest reproduction command an operator could run. Never
silently reinterpret a faulty result as a domain observation, and never
let a halt discard evidence you have already gathered in this session.

## Workspace briefing

The workspace is a Python monorepo for a telemetry ingestion service.
Layout highlights: `ingest/` holds the service (parsers, batching, sinks),
`ingest/tests/` its pytest suite, `tools/` operational scripts, and
`vendor/` pinned third-party snapshots you must not modify. The suite is
invoked with `python -m pytest -q`. Continuous integration currently
reports one failing test that did not fail last week; the repository has
had three merges since the last green run. Your task each session is
scoped by the first user message; do not expand scope beyond it."""

# (user, assistant) exchange script. Turn t submits history through user_t;
# assistant_t is appended for later turns. The final user turn requests the
# report, so exactly len(EXCHANGES) turns are runnable and turn 10 has no
# scripted assistant reply.
EXCHANGES = [
    (
        "Task: `ingest/tests/test_batcher.py::test_flush_on_max_bytes` fails "
        "on CI with `AssertionError: expected 2 batches, got 1`. Find the "
        "root cause and fix it. Acceptance: that test and its module pass.",
        "<think>\nOne failing batcher test, three recent merges. Before "
        "reading any code I want the shape of the ingest package so I can "
        "map the test to its subject module. The batcher likely lives in "
        "ingest/batching or similar; listing the tree is the cheapest "
        "orienting observation and cannot bias me toward a hypothesis.\n"
        "</think>\n\nOrienting: mapping the package layout first.\n\n"
        '<tool_call>\n{"name": "fs.list", "arguments": {"path": "ingest"}}\n'
        "</tool_call>",
    ),
    (
        '{"tool": "fs.list", "ok": true, "result": {"entries": ['
        '{"name": "__init__.py", "bytes": 71}, {"name": "batching.py", '
        '"bytes": 6488}, {"name": "parsers.py", "bytes": 9114}, '
        '{"name": "sinks.py", "bytes": 4732}, {"name": "config.py", '
        '"bytes": 2210}, {"name": "tests", "bytes": 0}]}}',
        "<think>\nbatching.py is 6.5 KB, a single module. The failing "
        "assertion counts flushed batches under a max-bytes threshold, so "
        "the suspect logic is the size accounting at append time. I will "
        "read the whole module; at this size a full read beats a windowed "
        "one and gives me exact line numbers for any patch.\n</think>\n\n"
        "Reading the batcher implementation.\n\n"
        '<tool_call>\n{"name": "fs.read", "arguments": {"path": '
        '"ingest/batching.py"}}\n</tool_call>',
    ),
    (
        '{"tool": "fs.read", "ok": true, "result": {"path": '
        '"ingest/batching.py", "lines": 180, "content": "…lines 1-180; '
        "key region: 88: def append(self, record: bytes) -> None: 89:     "
        "projected = self._bytes + len(record) 90:     if projected > "
        "self.max_bytes and self._records: 91:         self.flush() 92:     "
        'self._records.append(record) 93:     self._bytes = projected"}}',
        "<think>\nLine 93 is wrong after a flush: when line 91 flushes, "
        "_bytes resets inside flush(), but line 93 then assigns `projected`, "
        "which still includes the pre-flush total. Subsequent appends "
        "therefore think the new batch is nearly full-sized and stop "
        "flushing at the right boundaries — matching `expected 2 batches, "
        "got 1` if the second boundary is never reached. Before patching I "
        "will reproduce the failure to confirm the symptom is current, not "
        "a stale CI artifact.\n</think>\n\nReproducing the failure locally "
        "before patching.\n\n"
        '<tool_call>\n{"name": "shell.run", "arguments": {"cmd": "python '
        "-m pytest -q ingest/tests/test_batcher.py::"
        'test_flush_on_max_bytes"}}\n</tool_call>',
    ),
    (
        '{"tool": "shell.run", "ok": true, "result": {"exit_code": 1, '
        '"stdout": "F\\n=== FAILURES ===\\ntest_flush_on_max_bytes\\n'
        'AssertionError: expected 2 batches, got 1\\n1 failed in 0.31s", '
        '"stderr": ""}}',
        "<think>\nReproduced. The fix: after a flush, the running total "
        "must be just the new record's size, not the stale projection. "
        "Minimal patch is to recompute on line 93 from self._bytes (reset "
        "by flush) plus len(record) — equivalently, move the accounting "
        "after the conditional and derive it from current state. I will "
        "check flush() resets _bytes before writing the patch; the read "
        "content summarized lines 88-93 only, so one more targeted read of "
        "the flush method guards against a wrong assumption.\n</think>\n\n"
        "Verifying flush() resets the byte counter before I patch.\n\n"
        '<tool_call>\n{"name": "fs.read", "arguments": {"path": '
        '"ingest/batching.py", "start_line": 100, "end_line": 118}}\n'
        "</tool_call>",
    ),
    (
        '{"tool": "fs.read", "ok": true, "result": {"path": '
        '"ingest/batching.py", "content": "100: def flush(self) -> None: '
        "101:     if not self._records: 102:         return 103:     "
        "self._sink.emit(list(self._records)) 104:     "
        'self._records.clear() 105:     self._bytes = 0"}}',
        "<think>\nConfirmed: flush() zeroes _bytes at line 105. So after a "
        "conditional flush, line 93's `self._bytes = projected` reinstates "
        "the pre-flush total — the defect. The minimal correct statement "
        "is `self._bytes += len(record)` placed after the append, which is "
        "right in both the flushed and unflushed paths. Patch now, then "
        "rerun the narrowest test per discipline.\n</think>\n\nApplying "
        "the one-line accounting fix.\n\n"
        '<tool_call>\n{"name": "fs.write", "arguments": {"path": '
        '"ingest/batching.py", "old": "    self._records.append(record)\\n'
        '    self._bytes = projected", "new": "    '
        'self._records.append(record)\\n    self._bytes += len(record)"}}\n'
        "</tool_call>",
    ),
    (
        '{"tool": "fs.write", "ok": true, "result": {"path": '
        '"ingest/batching.py", "replaced": 1}}',
        "<think>\nPatch applied. Narrowest verification first: the single "
        "failing test. If it passes, widen to the module as the acceptance "
        "criteria require. No other call site mutates _bytes directly per "
        "the earlier full read, so I expect no collateral failures.\n"
        "</think>\n\nRerunning the failing test.\n\n"
        '<tool_call>\n{"name": "shell.run", "arguments": {"cmd": "python '
        "-m pytest -q ingest/tests/test_batcher.py::"
        'test_flush_on_max_bytes"}}\n</tool_call>',
    ),
    (
        '{"tool": "shell.run", "ok": true, "result": {"exit_code": 0, '
        '"stdout": ".\\n1 passed in 0.28s", "stderr": ""}}',
        "<think>\nTarget test green. Acceptance requires the whole module "
        "to pass — the batcher has adjacent tests for flush-on-count and "
        "flush-on-close that exercise the same counter, so they are the "
        "next cheapest falsifiers of my patch.\n</think>\n\nWidening to "
        "the full test module.\n\n"
        '<tool_call>\n{"name": "shell.run", "arguments": {"cmd": "python '
        '-m pytest -q ingest/tests/test_batcher.py"}}\n</tool_call>',
    ),
    (
        '{"tool": "shell.run", "ok": true, "result": {"exit_code": 0, '
        '"stdout": "........\\n8 passed in 0.94s", "stderr": ""}}',
        "<think>\nModule green, eight tests. Acceptance is met, but test "
        "discipline says run the suite when the narrow and module layers "
        "both pass, since parsers and sinks import the batcher and one of "
        "the three recent merges touched sinks. A full quiet run is cheap "
        "insurance against a coupled regression.\n</think>\n\nRunning the "
        "full suite as final verification.\n\n"
        '<tool_call>\n{"name": "shell.run", "arguments": {"cmd": "python '
        '-m pytest -q"}}\n</tool_call>',
    ),
    (
        '{"tool": "shell.run", "ok": true, "result": {"exit_code": 0, '
        '"stdout": "...........................\\n27 passed in 3.12s", '
        '"stderr": ""}}',
        "<think>\nEntire suite passes: 27 tests. Root cause, patch, and "
        "verification chain are all in hand. Remaining obligation is the "
        "final report with the required citations: root cause sentence, "
        "changed file with rationale, and the passing command output.\n"
        "</think>\n\nAll layers green; preparing the final report.\n\n"
        '<tool_call>\n{"name": "shell.run", "arguments": {"cmd": "git '
        'diff --stat"}}\n</tool_call>',
    ),
    (
        '{"tool": "shell.run", "ok": true, "result": {"exit_code": 0, '
        '"stdout": " ingest/batching.py | 2 +-\\n 1 file changed, '
        '1 insertion(+), 1 deletion(-)", "stderr": ""}}\n\nProduce the '
        "final report now.",
        None,
    ),
]

DIAG_RE = re.compile(r"(\w+)=([^\s]+)")


def parse_stderr(text: str) -> dict:
    out: dict = {"restore": None, "publish": None, "stats": None, "warnings": []}
    for line in text.splitlines():
        if line.startswith("durable_prefix_cache: "):
            pairs = dict(DIAG_RE.findall(line))
            if "checkpoint_hit" in pairs or "store_empty" in pairs:
                out["restore"] = pairs
            elif "publish" in pairs:
                out["publish"] = pairs
        elif line.startswith("stats: "):
            out["stats"] = dict(DIAG_RE.findall(line))
        elif line.startswith("warning:"):
            out["warnings"].append(line)
    return out


def build_messages(turn: int, assistants: list[str | None]) -> list[dict]:
    """History through user turn `turn` (1-indexed)."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for i in range(turn):
        msgs.append({"role": "user", "content": EXCHANGES[i][0]})
        if i < turn - 1:
            reply = assistants[i]
            assert reply is not None, f"missing assistant reply for turn {i + 1}"
            msgs.append({"role": "assistant", "content": reply})
    return msgs


def run_turn(binary, model_path, cache_dir, mode, msgs, gen_tokens, log_path):
    save = {
        "meta": {"model": str(model_path), "preserve_thinking": mode == "preserve"},
        "messages": msgs,
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="s0_turn_", delete=False
    ) as fh:
        json.dump(save, fh)
        tmp = fh.name
    cmd = [
        str(binary),
        "--model",
        str(model_path),
        "--messages",
        tmp,
        "--messages-preserve-thinking"
        if mode == "preserve"
        else "--messages-strip-thinking",
        "--temp",
        "0",
        "--seed",
        "42",
        "--top-k",
        "200",
        "--top-p",
        "1.0",
        "--min-p",
        "0.05",
        "--tokens",
        str(gen_tokens),
        "--max-context-tokens",
        "32768" if gen_tokens > 32 else "16384",
        "--durable-prefix-cache",
        str(cache_dir),
        "--durable-prefix-cache-max-mib",
        "8192",
        "--durable-prefix-cache-max-entry-mib",
        "2048",
    ]
    env = os.environ.copy()
    env["RUST_LOG"] = "warn,qwen_diag=info"
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=1800, env=env
        )
    finally:
        os.unlink(tmp)
    log_path.write_text(proc.stderr)
    if proc.returncode != 0:
        head = next(
            (l for l in proc.stderr.strip().splitlines() if l.strip()), "<no stderr>"
        )
        raise RuntimeError(f"qwen exited {proc.returncode}: {head}")
    return proc.stdout, parse_stderr(proc.stderr)


def run_cell(name, binary, models_dir, artifacts_root, turns_override=None):
    model_key, mode, lane, turns = CELLS[name]
    turns = turns_override or turns
    model_path = models_dir / MODELS[model_key]
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    cell_dir = artifacts_root / name
    cache_dir = cell_dir / "cache"
    if cell_dir.exists():
        shutil.rmtree(cell_dir)
    cache_dir.mkdir(parents=True)
    results_path = cell_dir / "results.jsonl"

    assistants: list[str | None] = [a for _, a in EXCHANGES]
    if lane == "live":
        assistants = [None] * len(EXCHANGES)

    print(f"\n=== {name} ({model_path.name}, {mode}, {lane}, {turns} turns) ===")
    header = (
        f"{'turn':>4} {'prompt':>7} {'hit':>5} {'matched':>8} {'restored':>8} "
        f"{'tail':>6} {'restore_ms':>10} {'capture':>9} {'prefill_ms':>10} "
        f"{'ttft_ms':>9} {'stop':>11}"
    )
    print(header)
    rows = []
    prev_prompt_tokens = None
    with results_path.open("w") as results:
        for t in range(1, turns + 1):
            msgs = build_messages(t, assistants)
            gen_tokens = 4096 if lane == "live" else 32
            stdout, parsed = run_turn(
                binary,
                model_path,
                cache_dir,
                mode,
                msgs,
                gen_tokens,
                cell_dir / f"turn-{t:02d}.stderr.log",
            )
            if lane == "live" and t <= len(assistants):
                assistants[t - 1] = stdout[:-1] if stdout.endswith("\n") else stdout
            stats = parsed["stats"] or {}
            restore = parsed["restore"] or {}
            publish = parsed["publish"] or {}
            prompt_tokens = int(stats.get("prompt_tokens", 0))
            restored = int(restore.get("restored", 0))
            row = {
                "cell": name,
                "turn": t,
                "lane": lane,
                "mode": mode,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": int(stats.get("generated_tokens", 0)),
                "stop_reason": stats.get("stop_reason"),
                "checkpoint_hit": restore.get("checkpoint_hit"),
                "matched": int(restore.get("matched", 0)),
                "restored": restored,
                "exact": restore.get("exact"),
                "store_empty": restore.get("store_empty"),
                "restore_total_ms": float(restore.get("restore_total_ms", 0.0)),
                "publish": publish.get("publish"),
                "capture": publish.get("capture"),
                "publish_matched_tokens": int(publish.get("matched_tokens", 0)),
                "capture_ms": float(publish.get("capture_ms", 0.0)),
                "tail": prompt_tokens - restored,
                "prev_prompt_tokens": prev_prompt_tokens,
                "load_ms": float(stats.get("load_ms", 0.0)),
                "prefill_ms": float(stats.get("prefill_ms", 0.0)),
                "ttft_ms": float(stats.get("ttft_ms", 0.0)),
                "warnings": parsed["warnings"],
            }
            rows.append(row)
            results.write(json.dumps(row) + "\n")
            results.flush()
            print(
                f"{t:>4} {row['prompt_tokens']:>7} {str(row['checkpoint_hit']):>5} "
                f"{row['matched']:>8} {row['restored']:>8} {row['tail']:>6} "
                f"{row['restore_total_ms']:>10.1f} {str(row['capture']):>9} "
                f"{row['prefill_ms']:>10.1f} {row['ttft_ms']:>9.1f} "
                f"{str(row['stop_reason']):>11}"
            )
            if t == 1 and prompt_tokens < 1024:
                print(
                    f"  WARNING: turn-1 prompt is {prompt_tokens} tokens "
                    "(< durable min 1024); checkpoints will be skipped",
                    file=sys.stderr,
                )
            prev_prompt_tokens = prompt_tokens
    return rows


def main():
    ap = argparse.ArgumentParser(description="S0 prefix-stability runner")
    ap.add_argument(
        "--cells",
        default=",".join(PRIMARY),
        help="comma-separated cell names, or 'smoke'",
    )
    ap.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    ap.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    ap.add_argument("--artifacts", type=Path, default=PACKET_DIR / "artifacts")
    ap.add_argument(
        "--turns", type=int, default=None, help="override turn count (smoke/debug)"
    )
    args = ap.parse_args()

    if not args.binary.is_file() or not os.access(args.binary, os.X_OK):
        sys.exit(f"binary not executable: {args.binary}")
    for name in args.cells.split(","):
        if name not in CELLS:
            sys.exit(f"unknown cell {name!r}; known: {', '.join(CELLS)}")
    args.artifacts.mkdir(parents=True, exist_ok=True)
    for name in args.cells.split(","):
        run_cell(name, args.binary, args.models_dir, args.artifacts, args.turns)


if __name__ == "__main__":
    main()
