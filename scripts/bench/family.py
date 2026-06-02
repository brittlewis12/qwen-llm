#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Family scoreboard sweep — qwen-llm vs llama.cpp.

Usage:
    scripts/bench/family.py                          # full family
    scripts/bench/family.py --tag 27B                # one model
    scripts/bench/family.py --shapes pp512,tg128     # narrow shapes
    scripts/bench/family.py --runs 5
    scripts/bench/family.py --no-digest              # raw json only

Output: docs/bench/<stamp>[-tag][-run-tag]-family/ with manifest.json,
lcpp-*.json, qwen-*.json, and an auto-generated README.md.

Sequential only — see docs/PERF-ROADMAP.md guardrails.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import NoReturn

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QWEN_BENCH = ROOT / "target" / "release" / "qwen-bench"
DEFAULT_LLAMA_BENCH = (
    Path.home() / "code" / "llama.cpp" / "build" / "bin" / "llama-bench"
)
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


def run_json(
    cmd: list[str | Path], *, stderr_filter: re.Pattern | None = None
) -> list[dict]:
    """Run a command that emits a JSON array on stdout. Stream stderr to
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


def probe_lcpp(llama_bench: Path, sample_model: Path) -> dict:
    rows = run_json(
        [llama_bench, "-m", sample_model, "-p", "1", "-n", "0", "-r", "1", "-o", "json"]
    )
    r = rows[0]
    return {
        "binary": str(llama_bench),
        "build_commit": r.get("build_commit", "unknown"),
        "build_number": r.get("build_number", 0),
        "backends": r.get("backends", "unknown"),
        "gpu_info": r.get("gpu_info", "unknown"),
        "cpu_info": r.get("cpu_info", "unknown"),
    }


def probe_qwen(qwen_bench: Path, sample_model: Path) -> dict:
    rows = run_json(
        [
            qwen_bench,
            "pp",
            "-m",
            sample_model,
            "-p",
            "1",
            "--runs",
            "1",
            "--no-warmup",
            "-o",
            "json",
        ]
    )
    r = rows[0]
    return {
        "binary": str(qwen_bench),
        "build_commit": r.get("build_commit", "unknown"),
        "build_dirty": int(r.get("build_dirty", 0)),
    }


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
    return {k: v for k, v in os.environ.items() if k.startswith("QWEN_")}


def parse_shapes(
    arg: str | None, pp_default: list[int], tg_default: list[int]
) -> tuple[list[int], list[int]]:
    """`--shapes pp128,pp512,tg32` overrides either set; unspecified side
    keeps the default. Bare numbers default to pp."""
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
    return (pp or pp_default), (tg or tg_default)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--tag", help="only run this model tag")
    ap.add_argument("--shapes", help="comma list of pp<N>/tg<N> shapes")
    ap.add_argument("--runs", type=int, default=3)
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
        help="permit qwen-llm builds with uncommitted worktree changes "
        "(default: refuse, since the resulting artifact's `build_commit` "
        "stamp points at the parent commit and lies about provenance)",
    )
    ap.add_argument("--qwen-bench", type=Path, default=DEFAULT_QWEN_BENCH)
    ap.add_argument("--llama-bench", type=Path, default=DEFAULT_LLAMA_BENCH)
    ap.add_argument("--models-toml", type=Path, default=DEFAULT_MODELS_TOML)
    ap.add_argument("--digest-script", type=Path, default=DEFAULT_DIGEST)
    args = ap.parse_args()

    qwen_bench = args.qwen_bench
    llama_bench = args.llama_bench
    if not os.access(qwen_bench, os.X_OK):
        die(f"qwen-bench not built or not executable: {qwen_bench}")
    if not os.access(llama_bench, os.X_OK):
        die(f"llama-bench missing or not executable: {llama_bench}")
    if not args.models_toml.is_file():
        die(f"models registry missing: {args.models_toml}")

    pp_shapes, tg_shapes = parse_shapes(args.shapes, [128, 512, 1024], [32, 128])
    if args.runs < 1:
        die("--runs must be >= 1")
    if args.cooldown_seconds < 0:
        die("--cooldown-seconds must be >= 0")

    registry = load_registry(args.models_toml, args.tag)
    # Resolve every model up front so we fail fast on missing files.
    resolved = [(row, *resolve_model(row)) for row in registry]
    sample_path = resolved[0][1]

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
            or p.name.startswith(("lcpp-", "qwen-"))
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

    # Stderr noise filters — lcpp prints ggml_metal_* setup lines on every
    # run; qwen-bench prints its own [pp]/[bench]/[tg] progress lines. None
    # of that is useful in sweep mode.
    qwen_noise = re.compile(r"^(\[(pp|bench|tg)\]|ggml_metal_|\s*$)")
    lcpp_noise = re.compile(r"^(ggml_metal_|\s*$)")

    lcpp_engine = probe_lcpp(llama_bench, sample_path)
    qwen_engine = probe_qwen(qwen_bench, sample_path)

    # Provenance guard: a dirty qwen-llm build will stamp its rows with the
    # *parent* commit (the last one cargo saw via .git/HEAD), so the artifact
    # silently claims to be from a commit that doesn't contain the changes
    # actually running. cx-round-2 #1.
    if qwen_engine.get("build_dirty") and not args.allow_dirty:
        die(
            "qwen-llm build is dirty (uncommitted changes in worktree). "
            "The resulting `build_commit` stamp will point at the parent "
            "commit and misrepresent provenance. Commit your changes and "
            "rebuild, or rerun with --allow-dirty to override."
        )

    manifest = {
        "stamp": stamp,
        "host": capture_host(),
        "engines": {"qwen_llm": qwen_engine, "llama_cpp": lcpp_engine},
        "sweep": {
            "pp_shapes": pp_shapes,
            "tg_shapes": tg_shapes,
            "runs": args.runs,
            "cooldown_seconds": args.cooldown_seconds,
            "tag_filter": args.tag or "",
            # Driver runs `lcpp(model_i); qwen(model_i)` for each model so
            # crash recovery is local and lcpp's baseline for each model is
            # taken minutes (not hours) before our number.
            "engine_order": "per_model_lcpp_then_qwen",
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
            }
            for row, path, size in resolved
        ],
    }
    write_manifest(out_dir, manifest)

    # Sweep. lcpp first per model so a mid-sweep crash still leaves a fresh
    # baseline. Never run engines in parallel.
    lcpp_count = 0
    qwen_count = 0
    first_measured = True
    for row, path, _size in resolved:
        tag = row["tag"]
        display = row["display"]
        print(f"[family] === {tag} ({display}) ===", file=sys.stderr)

        lcpp_out = out_dir / f"lcpp-{tag}.json"
        print(f"[family] -> {lcpp_out}", file=sys.stderr)
        cmd: list[str | Path] = [llama_bench, "-m", path]
        for p in pp_shapes:
            cmd += ["-p", str(p)]
        for n in tg_shapes:
            cmd += ["-n", str(n)]
        cmd += ["-r", str(args.runs), "-o", "json"]
        cooldown = 0.0 if first_measured else args.cooldown_seconds
        first_measured = False
        lcpp_rows, record = run_json_measured(
            cmd, stderr_filter=lcpp_noise, cooldown_seconds=cooldown
        )
        record.update({"engine": "llama.cpp", "tag": tag, "test": "all"})
        manifest["command_records"].append(record)
        write_manifest(out_dir, manifest)
        lcpp_out.write_text(json.dumps(lcpp_rows, indent=2) + "\n")
        lcpp_count += 1

        # qwen-bench pp, one file per shape.
        for p in pp_shapes:
            qpp_out = out_dir / f"qwen-pp{p}-{tag}.json"
            print(f"[family] -> {qpp_out}", file=sys.stderr)
            cmd = [
                qwen_bench,
                "pp",
                "-m",
                path,
                "-p",
                str(p),
                "--runs",
                str(args.runs),
                "-o",
                "json",
            ]
            cooldown = 0.0 if first_measured else args.cooldown_seconds
            first_measured = False
            rows, record = run_json_measured(
                cmd,
                stderr_filter=qwen_noise,
                cooldown_seconds=cooldown,
            )
            record.update({"engine": "qwen", "tag": tag, "test": f"pp{p}"})
            manifest["command_records"].append(record)
            write_manifest(out_dir, manifest)
            qpp_out.write_text(json.dumps(rows, indent=2) + "\n")
            qwen_count += 1

        # qwen-bench tg (apples-to-apples lcpp semantics), one file per shape.
        for n in tg_shapes:
            qtg_out = out_dir / f"qwen-tg{n}-{tag}.json"
            print(f"[family] -> {qtg_out}", file=sys.stderr)
            cmd = [
                qwen_bench,
                "tg",
                "-m",
                path,
                "-n",
                str(n),
                "--runs",
                str(args.runs),
                "-o",
                "json",
            ]
            cooldown = 0.0 if first_measured else args.cooldown_seconds
            first_measured = False
            rows, record = run_json_measured(
                cmd,
                stderr_filter=qwen_noise,
                cooldown_seconds=cooldown,
            )
            record.update({"engine": "qwen", "tag": tag, "test": f"tg{n}"})
            manifest["command_records"].append(record)
            write_manifest(out_dir, manifest)
            qtg_out.write_text(json.dumps(rows, indent=2) + "\n")
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
