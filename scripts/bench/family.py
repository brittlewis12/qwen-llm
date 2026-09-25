#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Family scoreboard sweep — qwen-llm vs llama.cpp.

Usage:
    scripts/bench/family.py                          # full family
    scripts/bench/family.py --tag 27B                # one model
    scripts/bench/family.py --shapes pp512,tg128     # exact narrow shapes
    scripts/bench/family.py --runs 5
    scripts/bench/family.py --no-digest              # raw json only

Output: docs/bench/<stamp>[-tag][-run-tag]-family/ with manifest.json,
lcpp-*.json, qwen-*.json, and an auto-generated README.md.

Sequential only — see docs/PERF-ROADMAP.md guardrails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import socket
import stat
import subprocess
import sys
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import NoReturn

from llama_cpp import (
    LOCK_PATH as DEFAULT_LLAMA_CPP_LOCK,
    load_lock,
    lock_summary,
    missing_tool_message,
    resolve_tool,
    validate_probe_row,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE_STATE_PREFIX = "git-source-sha256-v2:"
DEFAULT_QWEN_BENCH = ROOT / "target" / "release" / "qwen-bench"
DEFAULT_MODELS_TOML = ROOT / "scripts" / "bench" / "models.toml"
DEFAULT_DIGEST = ROOT / "scripts" / "bench" / "digest.py"

SHARD_RE = re.compile(r"-(\d{5})-of-\d{5}\.gguf$")


def die(msg: str, code: int = 2) -> NoReturn:
    print(f"[family] error: {msg}", file=sys.stderr)
    sys.exit(code)


def expand(path: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(path)))


def load_registry(toml_path: Path, tag_filter: str | None) -> list[dict]:
    """Parse models.toml and filter by tag if requested."""
    with toml_path.open("rb") as f:
        data = tomllib.load(f)
    rows = data.get("models", [])
    if tag_filter:
        rows = [r for r in rows if r.get("tag") == tag_filter]
        if not rows:
            die(f"no model matches --tag {tag_filter}")
    if not rows:
        die(f"no models found in {toml_path}")
    return rows


def resolve_model(row: dict) -> tuple[Path, int]:
    """Return (path, total_bytes). Sums shard siblings for split GGUFs."""
    path = expand(row["path"])
    if not path.is_file():
        die(f"model file not found for tag={row['tag']}: {path}")
    shard_match = SHARD_RE.search(path.name)
    if shard_match:
        # Replace the -NNNNN-of-NNNNN.gguf suffix with a glob and sum the
        # matching siblings. The 122B family ships three shards; only the
        # entry point loads, but the total weight bytes are the sum.
        glob_pat = SHARD_RE.sub("-*-of-*.gguf", path.name)
        shards = sorted(path.parent.glob(glob_pat))
        total = sum(s.stat().st_size for s in shards)
        return path, total
    return path, path.stat().st_size


def run_json_value(
    cmd: list[str | Path], *, stderr_filter: re.Pattern | None = None
) -> object:
    """Run a command that emits JSON on stdout. Stream stderr to
    our own stderr, filtering noisy lines so the operator sees only
    interesting output."""
    proc = subprocess.run(
        [str(c) for c in cmd],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        die(
            f"command failed (exit {proc.returncode}): {shlex.join(str(c) for c in cmd)}"
        )
    if proc.stderr:
        if stderr_filter is None:
            sys.stderr.write(proc.stderr)
        else:
            for line in proc.stderr.splitlines():
                if not stderr_filter.search(line):
                    print(line, file=sys.stderr)
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        sys.stderr.write(proc.stdout)
        die(f"non-JSON output from: {shlex.join(str(c) for c in cmd)} ({e})")


def run_json(
    cmd: list[str | Path], *, stderr_filter: re.Pattern | None = None
) -> list[dict]:
    value = run_json_value(cmd, stderr_filter=stderr_filter)
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        die(f"expected JSON row array from: {shlex.join(str(c) for c in cmd)}")
    return value


def run_json_object(
    cmd: list[str | Path], *, stderr_filter: re.Pattern | None = None
) -> dict:
    value = run_json_value(cmd, stderr_filter=stderr_filter)
    if not isinstance(value, dict):
        die(f"expected JSON object from: {shlex.join(str(c) for c in cmd)}")
    return value


def capture_text(command: list[str]) -> str:
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    out = proc.stdout.strip()
    err = proc.stderr.strip()
    if proc.returncode != 0:
        return err or out or f"exit {proc.returncode}"
    return out if out else err


def sample_run_context() -> dict:
    return {
        "thermal": capture_text(["pmset", "-g", "therm"]),
        "memory_pressure": capture_text(["memory_pressure", "-Q"]),
    }


def run_json_measured(
    cmd: list[str | Path],
    *,
    stderr_filter: re.Pattern | None = None,
    cooldown_seconds: float = 0.0,
) -> tuple[list[dict], dict]:
    if cooldown_seconds > 0:
        time.sleep(cooldown_seconds)
    before = sample_run_context()
    wall_start = time.perf_counter()
    rows = run_json(cmd, stderr_filter=stderr_filter)
    outer_wall_s = time.perf_counter() - wall_start
    after = sample_run_context()
    return rows, {
        "cmd": [str(c) for c in cmd],
        "cooldown_seconds": cooldown_seconds,
        "outer_wall_s": outer_wall_s,
        "thermal_before": before["thermal"],
        "thermal_after": after["thermal"],
        "memory_before": before["memory_pressure"],
        "memory_after": after["memory_pressure"],
    }


def write_manifest(out_dir: Path, manifest: dict) -> None:
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def probe_lcpp(
    llama_bench: Path,
    sample_model: Path,
    *,
    lock: dict | None,
    allow_unpinned: bool,
) -> dict:
    rows = run_json(
        [llama_bench, "-m", sample_model, "-p", "1", "-n", "0", "-r", "1", "-o", "json"]
    )
    r = rows[0]
    if lock is not None:
        err = validate_probe_row(r, lock, allow_unpinned=allow_unpinned)
        if err:
            die(err)
    return {
        "binary": str(llama_bench),
        "build_commit": r.get("build_commit", "unknown"),
        "build_number": r.get("build_number", 0),
        "backends": r.get("backends", "unknown"),
        "gpu_info": r.get("gpu_info", "unknown"),
        "cpu_info": r.get("cpu_info", "unknown"),
    }


def git_bytes(root: Path, *args: str) -> bytes:
    proc = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, check=False
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode(errors="replace").strip()
        stdout = proc.stdout.decode(errors="replace").strip()
        die(
            f"git {' '.join(args)} failed in {root}: "
            f"{stderr or stdout or f'exit {proc.returncode}'}"
        )
    return proc.stdout


def git_stdout(root: Path, *args: str) -> str:
    output = git_bytes(root, *args)
    try:
        value = output.decode().strip()
    except UnicodeDecodeError as exc:
        die(f"git {' '.join(args)} returned non-UTF-8 output in {root}: {exc}")
    if not value:
        die(f"git {' '.join(args)} returned no value in {root}")
    return value.lower()


def hash_section(digest: hashlib._Hash, label: bytes, value: bytes) -> None:
    digest.update(len(label).to_bytes(8, "big"))
    digest.update(label)
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def hash_worktree_entry(
    digest: hashlib._Hash, root: Path, scope: bytes, relative: bytes
) -> None:
    hash_section(digest, b"entry-scope", scope)
    hash_section(digest, b"entry-path", relative)
    path = os.path.join(os.fsencode(root), relative)
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        hash_section(digest, b"entry-kind", b"missing")
        return
    if stat.S_ISLNK(metadata.st_mode):
        hash_section(digest, b"entry-kind", b"symlink")
        hash_section(digest, b"entry-content", os.readlink(path))
        return
    if not stat.S_ISREG(metadata.st_mode):
        die(f"unsupported source entry type: {os.fsdecode(path)}")
    hash_section(digest, b"entry-kind", b"file")
    hash_section(
        digest,
        b"entry-executable",
        bytes([int(bool(metadata.st_mode & 0o111))]),
    )
    content = hashlib.sha256()
    size = 0
    with open(path, "rb") as source:
        while chunk := source.read(64 * 1024):
            content.update(chunk)
            size += len(chunk)
    hash_section(digest, b"entry-size", size.to_bytes(8, "big"))
    hash_section(digest, b"entry-content-sha256", content.digest())


def tracked_source_state(root: Path) -> str:
    """Mirror of `source_identity::tracked_source_state` (Rust): tracked
    content only. Untracked files (bench output, work-in-progress docs) do
    not change a binary's identity (32bacc9a); hashing them here made every
    sweep since 2026-08-17 refuse a freshly built binary."""
    head = git_bytes(root, "rev-parse", "HEAD")
    index = git_bytes(root, "ls-files", "--stage", "-z")
    index_flags = git_bytes(root, "ls-files", "-v", "-z")
    tracked = git_bytes(root, "ls-files", "-z")
    digest = hashlib.sha256()
    digest.update(b"qwen-git-source-state-v2\0")
    hash_section(digest, b"head", head)
    hash_section(digest, b"index", index)
    hash_section(digest, b"index-flags", index_flags)
    hash_section(digest, b"tracked-paths", tracked)
    for path in tracked.split(b"\0"):
        if path:
            hash_worktree_entry(digest, root, b"tracked", path)
    return f"{SOURCE_STATE_PREFIX}{digest.hexdigest()}"


def source_identity(root: Path) -> tuple[str, bool, str]:
    commit = git_stdout(root, "rev-parse", "HEAD")
    flags = git_bytes(root, "ls-files", "-v", "-z")
    hidden = any(
        entry[:1].islower() or entry.startswith(b"S")
        for entry in flags.split(b"\0")
        if entry
    )
    dirty = bool(git_bytes(root, "status", "--porcelain", "--untracked-files=all"))
    dirty = dirty or hidden
    return commit, dirty, tracked_source_state(root)


def validate_qwen_identity(
    identity: dict,
    source_commit: str,
    source_dirty: bool,
    source_state: str,
    *,
    allow_dirty: bool,
) -> None:
    required = {
        "schema_version",
        "build_commit",
        "build_commit_short",
        "build_dirty",
        "build_source_state",
        "stamp_source",
        "stamp_error",
        "runtime_commit",
        "runtime_dirty",
        "runtime_source_state",
        "status",
        "problems",
        "overrides",
    }
    missing = sorted(required - identity.keys())
    if missing:
        die(f"qwen build-info missing required fields: {', '.join(missing)}")
    if identity["schema_version"] != 2:
        die(
            "qwen build-info schema mismatch: "
            f"expected=2 got={identity['schema_version']!r}"
        )
    build_commit = identity["build_commit"]
    runtime_commit = identity["runtime_commit"]
    if not isinstance(build_commit, str) or not re.fullmatch(
        r"[0-9a-f]{40}|[0-9a-f]{64}", build_commit
    ):
        die(f"qwen build-info reported invalid build_commit: {build_commit!r}")
    if not isinstance(runtime_commit, str) or not re.fullmatch(
        r"[0-9a-f]{40}|[0-9a-f]{64}", runtime_commit
    ):
        die(f"qwen build-info reported invalid runtime_commit: {runtime_commit!r}")
    if build_commit != source_commit or runtime_commit != source_commit:
        die(
            "qwen binary/source identity mismatch: "
            f"binary={build_commit} runtime={runtime_commit} family_root={source_commit}; "
            "rebuild qwen-bench from this checkout"
        )
    if identity["build_commit_short"] != source_commit[:9]:
        die(
            "qwen build-info short commit mismatch: "
            f"expected={source_commit[:9]} got={identity['build_commit_short']!r}"
        )
    build_state = identity["build_source_state"]
    runtime_state = identity["runtime_source_state"]
    state_pattern = rf"{re.escape(SOURCE_STATE_PREFIX)}[0-9a-f]{{64}}"
    if not isinstance(build_state, str) or not re.fullmatch(state_pattern, build_state):
        die(f"qwen build-info reported invalid build_source_state: {build_state!r}")
    if not isinstance(runtime_state, str) or not re.fullmatch(
        state_pattern, runtime_state
    ):
        die(f"qwen build-info reported invalid runtime_source_state: {runtime_state!r}")
    if build_state != source_state or runtime_state != source_state:
        die(
            "qwen binary/source state mismatch: "
            f"binary={build_state} runtime={runtime_state} family_root={source_state}; "
            "rebuild qwen-bench from this exact source state"
        )
    build_dirty = identity["build_dirty"]
    runtime_dirty = identity["runtime_dirty"]
    if not isinstance(build_dirty, bool) or not isinstance(runtime_dirty, bool):
        die(
            "qwen build-info reported an unverifiable dirty state: "
            f"build={build_dirty!r} runtime={runtime_dirty!r}"
        )
    if build_dirty != source_dirty or runtime_dirty != source_dirty:
        die(
            "qwen binary/source dirty-state disagreement: "
            f"build={build_dirty} runtime={runtime_dirty} family_root={source_dirty}; "
            "rebuild qwen-bench after the latest source changes"
        )
    if identity["stamp_source"] not in {"git", "environment-verified"}:
        die(f"qwen build-info has invalid stamp_source: {identity['stamp_source']!r}")
    if identity["stamp_error"] is not None:
        die(f"qwen build-info has stamp_error: {identity['stamp_error']!r}")
    expected_status = "dirty" if source_dirty else "match"
    expected_problems = ["dirty"] if source_dirty else []
    if (
        identity["status"] != expected_status
        or identity["problems"] != expected_problems
    ):
        die(
            "qwen build identity classification disagreement: "
            f"status={identity['status']!r} problems={identity['problems']!r} "
            f"expected_status={expected_status!r} expected_problems={expected_problems!r}"
        )
    if identity["overrides"] != []:
        die(
            f"qwen build-info must report raw identity without overrides: {identity['overrides']!r}"
        )
    if source_dirty and not allow_dirty:
        die(
            "qwen-llm source is dirty. Commit the changes or "
            "rerun with --allow-dirty for a non-canonical sweep."
        )


def probe_qwen(qwen_bench: Path, *, allow_dirty: bool) -> dict:
    identity = run_json_object([qwen_bench, "build-info", "-o", "json"])
    source_commit, source_dirty, source_state = source_identity(ROOT)
    validate_qwen_identity(
        identity,
        source_commit,
        source_dirty,
        source_state,
        allow_dirty=allow_dirty,
    )
    return {
        "binary": str(qwen_bench),
        "build_commit": identity["build_commit_short"],
        "build_commit_full": identity["build_commit"],
        "build_dirty": int(source_dirty),
        "source_state": source_state,
        "build_identity": identity,
    }


def assert_source_identity(expected: dict) -> None:
    current_commit, current_dirty, current_state = source_identity(ROOT)
    if (
        current_commit != expected["build_commit_full"]
        or int(current_dirty) != expected["build_dirty"]
        or current_state != expected["source_state"]
    ):
        die(
            "qwen source identity changed during the family sweep; "
            "discard this artifact and restart from a stable checkout"
        )


def validate_qwen_rows(
    rows: list[dict],
    expected_identity: dict,
    *,
    allow_dirty: bool,
    expected_tests: set[str],
) -> None:
    stable_fields = (
        "schema_version",
        "build_commit",
        "build_commit_short",
        "build_dirty",
        "build_source_state",
        "stamp_source",
        "stamp_error",
        "runtime_commit",
        "runtime_dirty",
        "runtime_source_state",
        "status",
        "problems",
        "overrides",
    )
    expected_row_identity = dict(expected_identity)
    expected_row_identity["overrides"] = (
        ["allow_dirty"]
        if allow_dirty
        and (
            expected_identity["build_dirty"] is True
            or expected_identity["runtime_dirty"] is True
        )
        else []
    )
    actual_tests = [row.get("test") for row in rows]
    if len(actual_tests) != len(set(actual_tests)):
        die(f"qwen suite emitted duplicate test rows: {actual_tests!r}")
    if set(actual_tests) != expected_tests:
        die(
            "qwen suite row set mismatch: "
            f"expected={sorted(expected_tests)!r} got={sorted(actual_tests)!r}"
        )
    for row in rows:
        identity = row.get("build_identity")
        if not isinstance(identity, dict):
            die(f"qwen row {row.get('test')!r} has no build_identity packet")
        missing = [field for field in stable_fields if field not in identity]
        if missing:
            die(
                f"qwen row {row.get('test')!r} identity missing fields: "
                f"{', '.join(missing)}"
            )
        for field in stable_fields:
            if identity[field] != expected_row_identity[field]:
                die(
                    f"qwen identity changed during sweep for row {row.get('test')!r}: "
                    f"field {field} expected {expected_row_identity[field]!r}, "
                    f"got {identity[field]!r}"
                )
        expected_dirty = int(
            identity["build_dirty"] is True or identity["runtime_dirty"] is True
        )
        aliases = {
            "schema_version": 3,
            "engine": "qwen-llm",
            "build_commit": identity["build_commit_short"],
            "build_dirty": expected_dirty,
        }
        for field, expected in aliases.items():
            if row.get(field) != expected:
                die(
                    f"qwen row {row.get('test')!r} has inconsistent {field}: "
                    f"expected {expected!r}, got {row.get(field)!r}"
                )


def capture_host() -> dict:
    try:
        load = subprocess.check_output(["uptime"], text=True).strip()
        loadavg = re.sub(r".*load averages?: ", "", load)
    except Exception:
        loadavg = "unknown"
    try:
        top = subprocess.check_output(
            ["ps", "-A", "-o", "%cpu,comm", "-r"], text=True
        ).splitlines()[1:6]
    except Exception:
        top = []
    return {
        "hostname": socket.gethostname().split(".")[0],
        "os": " ".join(os.uname()[:3]) + " " + os.uname().machine,
        "loadavg": loadavg,
        "top_cpu_at_start": top,
    }


def capture_qwen_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k.startswith(("QWEN_", "QWEN4EXP_"))}


def parse_shapes(
    arg: str | None, pp_default: list[int], tg_default: list[int]
) -> tuple[list[int], list[int]]:
    """`--shapes pp128,pp512,tg32` runs exactly those shapes.
    Bare numbers default to pp. Without `--shapes`, use the defaults."""
    if not arg:
        return pp_default, tg_default
    pp: list[int] = []
    tg: list[int] = []
    for piece in arg.split(","):
        s = piece.strip()
        if s.startswith("pp"):
            pp.append(int(s[2:]))
        elif s.startswith("tg"):
            tg.append(int(s[2:]))
        else:
            pp.append(int(s))
    return pp, tg


def parse_int_list(arg: str, name: str) -> list[int]:
    try:
        values = [int(piece) for piece in arg.split(",") if piece.strip()]
    except ValueError:
        die(f"{name} must be a comma list of integers, got {arg!r}")
    if not values or any(v < 0 for v in values):
        die(f"{name} must list non-negative integers, got {arg!r}")
    return values


def shape_test(kind: str, n: int, depth: int) -> str:
    """qwen-bench's `test` vocabulary: `pp512`, `tg128@d8192`."""
    return f"{kind}{n}" if depth == 0 else f"{kind}{n}@d{depth}"


def lcpp_test(row: dict) -> str:
    n_prompt = row.get("n_prompt") or 0
    n_gen = row.get("n_gen") or 0
    depth = row.get("n_depth") or 0
    if n_prompt > 0 and n_gen == 0:
        return shape_test("pp", n_prompt, depth)
    if n_gen > 0 and n_prompt == 0:
        return shape_test("tg", n_gen, depth)
    return f"pp{n_prompt}+tg{n_gen}"


def merge_blocks(blocks: list[list[dict]], key) -> list[dict]:
    """Merge one engine's rows from several blocks: samples concatenate and
    the summary statistics are recomputed over all of them. Each row keeps
    its per-block means in `block_avg_ts` so drift stays visible."""
    merged: dict = {}
    for rows in blocks:
        for row in rows:
            k = key(row)
            if k not in merged:
                merged[k] = dict(row)
                merged[k]["samples_ts"] = list(row.get("samples_ts") or [])
                merged[k]["samples_ns"] = list(row.get("samples_ns") or [])
                merged[k]["block_avg_ts"] = [row.get("avg_ts")]
            else:
                merged[k]["samples_ts"] += list(row.get("samples_ts") or [])
                merged[k]["samples_ns"] += list(row.get("samples_ns") or [])
                merged[k]["block_avg_ts"].append(row.get("avg_ts"))
    for row in merged.values():
        ts = row["samples_ts"]
        ns = row["samples_ns"]
        if ts:
            mean = sum(ts) / len(ts)
            row["avg_ts"] = mean
            row["stddev_ts"] = (
                (sum((x - mean) ** 2 for x in ts) / (len(ts) - 1)) ** 0.5
                if len(ts) > 1
                else 0.0
            )
        if ns:
            row["avg_ns"] = int(sum(ns) / len(ns))
        row["n_repetitions"] = len(ts)
        row["n_blocks"] = len(row["block_avg_ts"])
    return list(merged.values())


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--tag", help="only run this model tag")
    ap.add_argument("--shapes", help="comma list of pp<N>/tg<N> shapes")
    ap.add_argument(
        "--runs", type=int, default=3, help="timed reps per engine command per block"
    )
    ap.add_argument(
        "--blocks",
        type=int,
        default=2,
        help="paired blocks per model; engine order alternates (ABBA) and "
        "each engine's block samples merge into one row per test",
    )
    ap.add_argument(
        "--depths",
        default="0",
        help="comma list of context depths for tg shapes (llama-bench -d); "
        "a model's `tg_depths` in models.toml overrides it",
    )
    ap.add_argument(
        "--lcpp-ubatch",
        default="512,2048",
        help="llama.cpp micro-batch sizes for pp shapes; the largest is the "
        "tuned comparator (lcpp-<tag>.json), the rest are written as "
        "lcpp_ub<N>-<tag>.json (512 is llama.cpp's default)",
    )
    ap.add_argument(
        "--cooldown-seconds",
        type=float,
        default=0.0,
        help="sleep before each measured engine command after the first",
    )
    ap.add_argument("--run-tag", default="", help="suffix on the output dir name")
    ap.add_argument("--no-digest", action="store_true")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="allow reusing a non-empty output directory; deletes lcpp-*.json, "
        "qwen-*.json, manifest.json, README.md before running",
    )
    ap.add_argument(
        "--allow-dirty",
        action="store_true",
        help="permit source changes for a non-canonical sweep; the "
        "dirty identity remains recorded in every qwen row",
    )
    ap.add_argument("--qwen-bench", type=Path, default=DEFAULT_QWEN_BENCH)
    ap.add_argument(
        "--llama-bench",
        type=Path,
        help="explicit llama-bench path; default resolves the pinned llama.cpp lock",
    )
    ap.add_argument("--llama-cpp-lock", type=Path, default=DEFAULT_LLAMA_CPP_LOCK)
    ap.add_argument(
        "--allow-unpinned-lcpp",
        action="store_true",
        help="permit a llama.cpp binary whose build_commit/backends do not match the lock",
    )
    ap.add_argument("--models-toml", type=Path, default=DEFAULT_MODELS_TOML)
    ap.add_argument("--digest-script", type=Path, default=DEFAULT_DIGEST)
    args = ap.parse_args()

    qwen_bench = args.qwen_bench
    lcpp_lock = load_lock(args.llama_cpp_lock)
    llama_bench, llama_bench_locked = resolve_tool(
        "llama-bench",
        explicit=args.llama_bench,
        env_var="LLAMA_BENCH",
        lock=lcpp_lock,
    )
    if not os.access(qwen_bench, os.X_OK):
        die(f"qwen-bench not built or not executable: {qwen_bench}")
    if not os.access(llama_bench, os.X_OK):
        die(missing_tool_message(llama_bench))
    if not args.models_toml.is_file():
        die(f"models registry missing: {args.models_toml}")

    pp_shapes, tg_shapes = parse_shapes(args.shapes, [128, 512, 1024], [32, 128])
    if args.runs < 1:
        die("--runs must be >= 1")
    if args.blocks < 1:
        die("--blocks must be >= 1")
    if args.cooldown_seconds < 0:
        die("--cooldown-seconds must be >= 0")
    default_depths = parse_int_list(args.depths, "--depths")
    ubatches = sorted(set(parse_int_list(args.lcpp_ubatch, "--lcpp-ubatch")))
    if 0 in ubatches:
        die("--lcpp-ubatch values must be >= 1")
    tuned_ubatch = ubatches[-1]

    # Fail closed before resolving or loading any model. qwen-bench independently
    # compares its compiled stamp with its source checkout; this driver also
    # compares both against ROOT so copied/stale binaries cannot bless a sweep.
    qwen_engine = probe_qwen(qwen_bench, allow_dirty=args.allow_dirty)

    registry = load_registry(args.models_toml, args.tag)
    # Resolve every model up front so we fail fast on missing files.
    resolved = [(row, *resolve_model(row)) for row in registry]
    # Probe llama.cpp on a model it implements (a registry may lead with one
    # it does not, e.g. K2 Horizon).
    lcpp_models = [path for row, path, _size in resolved if row.get("lcpp", True)]
    sample_path = lcpp_models[0] if lcpp_models else resolved[0][1]

    # Stderr noise filters — lcpp prints ggml_metal_* setup lines on every
    # run; qwen-bench prints its own [pp]/[bench]/[tg]/[suite] progress lines. None
    # of that is useful in sweep mode.
    qwen_noise = re.compile(r"^(\[(pp|bench|tg|suite)\]|ggml_metal_|\s*$)")
    lcpp_noise = re.compile(r"^(ggml_metal_|\s*$)")

    lcpp_engine = probe_lcpp(
        llama_bench,
        sample_path,
        lock=lcpp_lock,
        allow_unpinned=args.allow_unpinned_lcpp,
    )
    lcpp_engine["locked"] = llama_bench_locked

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M")
    suffix = f"-{args.tag}" if args.tag else ""
    if args.run_tag:
        suffix += f"-{args.run_tag}"
    out_dir = ROOT / "docs" / "bench" / f"{stamp}{suffix}-family"
    # Stale-file guard. Same-minute reruns (e.g. interrupted sweep, narrowed
    # shape grid) would otherwise leak old rows into the digest input set.
    if out_dir.exists() and any(out_dir.iterdir()):
        known = sorted(
            p
            for p in out_dir.iterdir()
            if p.name == "manifest.json"
            or p.name == "README.md"
            or p.name.startswith(("lcpp-", "lcpp_ub", "qwen-"))
        )
        if not args.overwrite:
            existing = ", ".join(p.name for p in known) or "(unrelated files)"
            die(
                f"output dir already populated: {out_dir}\n"
                f"  existing: {existing}\n"
                f"  rerun with --overwrite to remove the generated files first, "
                f"or wait a minute for a new timestamp"
            )
        for p in known:
            p.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[family] output: {out_dir}", file=sys.stderr)

    manifest = {
        "schema_version": 2,
        "stamp": stamp,
        "host": capture_host(),
        "engines": {"qwen_llm": qwen_engine, "llama_cpp": lcpp_engine},
        "llama_cpp_lock": lock_summary(lcpp_lock),
        "sweep": {
            "pp_shapes": pp_shapes,
            "tg_shapes": tg_shapes,
            "runs": args.runs,
            "cooldown_seconds": args.cooldown_seconds,
            "allow_dirty": args.allow_dirty,
            "tag_filter": args.tag or "",
            # Driver runs `lcpp(model_i); qwen(model_i)` for each model so
            # crash recovery is local and lcpp's baseline for each model is
            # taken minutes (not hours) before our number.
            "blocks": args.blocks,
            "default_tg_depths": default_depths,
            "lcpp_ubatch": ubatches,
            "lcpp_tuned_ubatch": tuned_ubatch,
            # Per model, `blocks` paired blocks; within a block both engines
            # run pp (depth 0) then tg (at the model's depths). The first
            # engine alternates by block and by model (ABBA), and each
            # engine's samples merge across blocks into one row per test.
            "engine_order": "abba_per_model_alternating",
        },
        "qwen_env_at_start": capture_qwen_env(),
        "command_records": [],
        "models": [
            {
                "tag": row["tag"],
                "display": row["display"],
                "kind": row["kind"],
                "quant": row["quant"],
                "path": str(path),
                # On-disk file footprint (sum across shards). NOT
                # the same as a row's `model_size`, which is weight
                # tensor bytes only.
                "file_size_bytes": size,
                "family": row.get("family", "qwen"),
                # False when upstream llama.cpp has no implementation of
                # the architecture: the cell is reported, not measured.
                "lcpp": row.get("lcpp", True),
                "tg_depths": row.get("tg_depths", default_depths),
            }
            for row, path, size in resolved
        ],
    }
    write_manifest(out_dir, manifest)

    # Sweep. Never run engines in parallel. Per model, `blocks` paired
    # blocks alternate which engine goes first (ABBA), and the first engine
    # also alternates by model, so linear thermal drift cancels instead of
    # always favoring one engine.
    lcpp_count = 0
    qwen_count = 0
    first_measured = True

    def measured(cmd: list[str | Path], noise: re.Pattern, what: dict) -> list:
        nonlocal first_measured
        cooldown = 0.0 if first_measured else args.cooldown_seconds
        first_measured = False
        assert_source_identity(qwen_engine)
        rows, record = run_json_measured(
            cmd, stderr_filter=noise, cooldown_seconds=cooldown
        )
        assert_source_identity(qwen_engine)
        record.update(what)
        manifest["command_records"].append(record)
        write_manifest(out_dir, manifest)
        return rows

    for model_index, (row, path, _size) in enumerate(resolved):
        tag = row["tag"]
        display = row["display"]
        lcpp_supported = row.get("lcpp", True)
        depths = row.get("tg_depths", default_depths)
        print(f"[family] === {tag} ({display}) ===", file=sys.stderr)
        if not lcpp_supported:
            print(
                f"[family] {tag}: llama.cpp has no upstream implementation; "
                "measuring qwen-llm only",
                file=sys.stderr,
            )

        def run_lcpp(block: int) -> list[dict]:
            rows: list[dict] = []
            if pp_shapes:
                cmd: list[str | Path] = [llama_bench, "-m", path]
                cmd += ["-p", ",".join(str(p) for p in pp_shapes), "-n", "0"]
                cmd += ["-ub", ",".join(str(u) for u in ubatches)]
                cmd += ["-r", str(args.runs), "-o", "json"]
                rows += measured(
                    cmd,
                    lcpp_noise,
                    {"engine": "llama.cpp", "tag": tag, "test": "pp", "block": block},
                )
            if tg_shapes:
                cmd = [llama_bench, "-m", path, "-p", "0"]
                cmd += ["-n", ",".join(str(n) for n in tg_shapes)]
                cmd += ["-d", ",".join(str(d) for d in depths)]
                cmd += ["-ub", str(tuned_ubatch)]
                cmd += ["-r", str(args.runs), "-o", "json"]
                rows += measured(
                    cmd,
                    lcpp_noise,
                    {"engine": "llama.cpp", "tag": tag, "test": "tg", "block": block},
                )
            for lcpp_row in rows:
                lcpp_row["test"] = lcpp_test(lcpp_row)
            return rows

        def run_qwen(block: int) -> list[dict]:
            rows: list[dict] = []
            base: list[str | Path] = [qwen_bench]
            if args.allow_dirty:
                base.append("--allow-dirty")
            base += ["suite", "-m", path, "--runs", str(args.runs), "-o", "json"]
            invocations = []
            if pp_shapes:
                invocations.append(
                    (
                        "pp",
                        ["--pp", ",".join(str(p) for p in pp_shapes)],
                        {shape_test("pp", p, 0) for p in pp_shapes},
                    )
                )
            if tg_shapes:
                invocations.append(
                    (
                        "tg",
                        [
                            "--tg",
                            ",".join(str(n) for n in tg_shapes),
                            "--depth",
                            ",".join(str(d) for d in depths),
                        ],
                        {shape_test("tg", n, d) for n in tg_shapes for d in depths},
                    )
                )
            for test, extra, expected in invocations:
                got = measured(
                    base + extra,
                    qwen_noise,
                    {"engine": "qwen", "tag": tag, "test": test, "block": block},
                )
                validate_qwen_rows(
                    got,
                    qwen_engine["build_identity"],
                    allow_dirty=args.allow_dirty,
                    expected_tests=expected,
                )
                rows += got
            return rows

        lcpp_blocks: list[list[dict]] = []
        qwen_blocks: list[list[dict]] = []
        for block in range(args.blocks):
            order = (
                ["lcpp", "qwen"] if (model_index + block) % 2 == 0 else ["qwen", "lcpp"]
            )
            if not lcpp_supported:
                order = ["qwen"]
            for engine in order:
                print(f"[family] -> {tag} block {block} {engine}", file=sys.stderr)
                if engine == "lcpp":
                    lcpp_blocks.append(run_lcpp(block))
                else:
                    qwen_blocks.append(run_qwen(block))

        if lcpp_supported:
            lcpp_rows = merge_blocks(
                lcpp_blocks, key=lambda r: (r["test"], r.get("n_ubatch"))
            )
            tuned = [
                r
                for r in lcpp_rows
                if r["test"].startswith("tg") or r.get("n_ubatch") == tuned_ubatch
            ]
            (out_dir / f"lcpp-{tag}.json").write_text(
                json.dumps(tuned, indent=2) + "\n"
            )
            for ubatch in ubatches:
                if ubatch == tuned_ubatch:
                    continue
                variant = [
                    r
                    for r in lcpp_rows
                    if r["test"].startswith("pp") and r.get("n_ubatch") == ubatch
                ]
                (out_dir / f"lcpp_ub{ubatch}-{tag}.json").write_text(
                    json.dumps(variant, indent=2) + "\n"
                )
            lcpp_count += 1

        for bench_row in merge_blocks(qwen_blocks, key=lambda r: r["test"]):
            test = bench_row.get("test", "")
            if not re.fullmatch(r"(?:pp|tg)\d+(?:@d\d+)?", test):
                die(f"unexpected qwen suite test name for tag={tag}: {test!r}")
            qwen_out = out_dir / f"qwen-{test}-{tag}.json"
            qwen_out.write_text(json.dumps([bench_row], indent=2) + "\n")
            qwen_count += 1

    print(
        f"[family] sweep done: lcpp={lcpp_count} models, qwen={qwen_count} bench files",
        file=sys.stderr,
    )

    if not args.no_digest:
        if not os.access(args.digest_script, os.X_OK):
            print(
                "[family] note: digest.py not executable, skipping README.md",
                file=sys.stderr,
            )
        else:
            print("[family] generating digest...", file=sys.stderr)
            readme = subprocess.run(
                [str(args.digest_script), str(out_dir)],
                check=True,
                capture_output=True,
                text=True,
            )
            (out_dir / "README.md").write_text(readme.stdout)
            print(f"[family] README.md: {out_dir / 'README.md'}", file=sys.stderr)

    print(out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
