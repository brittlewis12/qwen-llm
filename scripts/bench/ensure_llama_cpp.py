#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from llama_cpp import (
    LOCK_PATH,
    build_dir,
    cache_root,
    expected_build_commit,
    load_lock,
    lock_summary,
    locked_bin,
    mirror_dir,
    source_dir,
    validate_probe_row,
)


def run(cmd: list[str | Path], *, cwd: Path | None = None) -> str:
    printable = shlex.join(str(c) for c in cmd)
    print(f"[ensure-llama-cpp] $ {printable}", file=sys.stderr)
    proc = subprocess.run(
        [str(c) for c in cmd],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        if proc.stdout:
            sys.stderr.write(proc.stdout)
        if proc.stderr:
            sys.stderr.write(proc.stderr)
        raise SystemExit(f"command failed ({proc.returncode}): {printable}")
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    return proc.stdout.strip()


def is_executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def ensure_mirror(lock: dict, root: Path) -> Path:
    mirror = mirror_dir(lock, root)
    mirror.parent.mkdir(parents=True, exist_ok=True)
    if mirror.exists():
        run(["git", f"--git-dir={mirror}", "fetch", "--tags", "origin"])
    else:
        cmd = ["git", "clone", "--mirror"]
        reference = Path.home() / "code" / "llama.cpp"
        if reference.exists():
            cmd.extend(["--reference-if-able", reference])
        cmd.extend([str(lock["repo"]), mirror])
        run(cmd)
    run(
        ["git", f"--git-dir={mirror}", "cat-file", "-e", f"{lock['commit']}^{{commit}}"]
    )
    return mirror


def ensure_source(lock: dict, root: Path) -> Path:
    mirror = ensure_mirror(lock, root)
    src = source_dir(lock, root)
    src.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        head = run(["git", "-C", src, "rev-parse", "HEAD"])
        status = run(["git", "-C", src, "status", "--porcelain"])
        if status:
            raise SystemExit(f"cached llama.cpp worktree is dirty: {src}")
        if head != lock["commit"]:
            raise SystemExit(
                f"cached llama.cpp worktree has unexpected HEAD {head}: {src}"
            )
    else:
        run(
            [
                "git",
                f"--git-dir={mirror}",
                "worktree",
                "add",
                "--detach",
                src,
                lock["commit"],
            ]
        )
    return src


def build(lock: dict, root: Path) -> Path:
    src = ensure_source(lock, root)
    bld = build_dir(lock, root)
    bld.mkdir(parents=True, exist_ok=True)
    cmake_args = [
        str(x) for x in lock.get("cmake_args", ["-DCMAKE_BUILD_TYPE=Release"])
    ]
    run(["cmake", "-S", src, "-B", bld, *cmake_args])
    targets = [str(x) for x in lock.get("targets", ["llama-bench"])]
    jobs = str(os.cpu_count() or 1)
    run(["cmake", "--build", bld, "--target", *targets, "-j", jobs])
    meta = {
        "lock": lock_summary(lock),
        "source_dir": str(src),
        "build_dir": str(bld),
    }
    (bld / "qwen-llama-cpp-meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return bld


def smoke(lock: dict, root: Path, model: str | None) -> None:
    for target in lock.get("targets", []):
        path = locked_bin(str(target), lock, root)
        if not is_executable(path):
            raise SystemExit(f"expected target missing after build: {path}")
    run([locked_bin("llama-bench", lock, root), "--help"])
    run([locked_bin("llama-tokenize", lock, root), "--help"])
    run([locked_bin("llama-cli", lock, root), "--version"])
    if model:
        out = run(
            [
                locked_bin("llama-bench", lock, root),
                "-m",
                model,
                "-p",
                "1",
                "-n",
                "0",
                "-r",
                "1",
                "-o",
                "json",
            ]
        )
        rows = json.loads(out)
        err = validate_probe_row(rows[0], lock)
        if err:
            raise SystemExit(err)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the pinned llama.cpp benchmark target used by qwen scripts."
    )
    parser.add_argument("--lock", type=Path, default=LOCK_PATH)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument(
        "--smoke-model", help="optional GGUF for build_commit/backend smoke"
    )
    parser.add_argument(
        "--print-bin",
        choices=("llama-bench", "llama-tokenize", "llama-cli"),
        help="print one resolved binary path after ensuring the build",
    )
    args = parser.parse_args()

    lock = load_lock(args.lock)
    root = cache_root(args.cache_root)
    print(
        f"[ensure-llama-cpp] lock {lock.get('ref') or lock['commit']} "
        f"({expected_build_commit(lock)})",
        file=sys.stderr,
    )
    print(f"[ensure-llama-cpp] cache {root}", file=sys.stderr)
    build(lock, root)
    smoke(lock, root, args.smoke_model)
    if args.print_bin:
        print(locked_bin(args.print_bin, lock, root))
    else:
        print(build_dir(lock, root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
