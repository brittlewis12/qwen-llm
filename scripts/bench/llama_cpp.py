from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


LOCK_PATH = Path(__file__).with_name("llama-cpp.lock.json")
CACHE_ENV = "QWEN_LLAMA_CPP_CACHE"
DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "qwen-llm" / "llama.cpp"


def expand_path(value: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(value))))


def cache_root(raw: str | Path | None = None) -> Path:
    if raw is not None:
        return expand_path(raw)
    return expand_path(os.environ.get(CACHE_ENV, DEFAULT_CACHE_ROOT))


def load_lock(path: str | Path | None = None) -> dict[str, Any]:
    lock_path = expand_path(path or LOCK_PATH)
    try:
        lock = json.loads(lock_path.read_text())
    except FileNotFoundError as exc:
        raise RuntimeError(f"llama.cpp lock file missing: {lock_path}") from exc
    for key in ("repo", "commit", "targets"):
        if key not in lock:
            raise RuntimeError(f"llama.cpp lock file missing `{key}`: {lock_path}")
    commit = str(lock["commit"])
    if len(commit) < 8:
        raise RuntimeError(f"llama.cpp lock commit is too short: {commit!r}")
    return lock


def lock_summary(lock: dict[str, Any]) -> dict[str, Any]:
    return {
        "repo": lock.get("repo"),
        "ref": lock.get("ref"),
        "commit": lock.get("commit"),
        "cmake_args": lock.get("cmake_args", []),
        "targets": lock.get("targets", []),
        "expected_build_commit": expected_build_commit(lock),
        "expected_backends": lock.get("expected_backends", []),
    }


def cache_key(lock: dict[str, Any]) -> str:
    return str(lock["commit"])[:12]


def mirror_dir(lock: dict[str, Any], root: str | Path | None = None) -> Path:
    del lock
    return cache_root(root) / "mirror.git"


def source_dir(lock: dict[str, Any], root: str | Path | None = None) -> Path:
    return cache_root(root) / "src" / cache_key(lock)


def build_dir(lock: dict[str, Any], root: str | Path | None = None) -> Path:
    return cache_root(root) / "build" / cache_key(lock)


def locked_bin(
    tool: str,
    lock: dict[str, Any] | None = None,
    root: str | Path | None = None,
) -> Path:
    lock = lock or load_lock()
    return build_dir(lock, root) / "bin" / tool


def expected_build_commit(lock: dict[str, Any]) -> str:
    return str(lock.get("expected_build_commit") or str(lock["commit"])[:9])


def resolve_tool(
    tool: str,
    *,
    explicit: str | Path | None = None,
    env_var: str | None = None,
    lock: dict[str, Any] | None = None,
    root: str | Path | None = None,
) -> tuple[Path, bool]:
    if explicit is not None:
        return expand_path(explicit), False
    if env_var:
        raw = os.environ.get(env_var)
        if raw:
            return expand_path(raw), False
    lock = lock or load_lock()
    return locked_bin(tool, lock, root), True


def validate_probe_row(
    row: dict[str, Any],
    lock: dict[str, Any],
    *,
    allow_unpinned: bool = False,
) -> str | None:
    if allow_unpinned:
        return None

    actual_commit = str(row.get("build_commit") or "")
    expected_commit = expected_build_commit(lock)
    full_commit = str(lock["commit"])
    if not actual_commit:
        return (
            "llama.cpp row did not report build_commit; rerun with "
            "--allow-unpinned-lcpp to override"
        )
    if not (
        full_commit.startswith(actual_commit)
        or actual_commit.startswith(expected_commit)
    ):
        return (
            "llama.cpp build_commit mismatch: "
            f"binary reports {actual_commit}, lock expects {expected_commit} "
            f"({full_commit}). Run scripts/bench/ensure_llama_cpp.py or pass "
            "--allow-unpinned-lcpp for an intentional one-off."
        )

    backends = str(row.get("backends") or "")
    missing = [b for b in lock.get("expected_backends", []) if b not in backends]
    if missing:
        return (
            "llama.cpp backend mismatch: "
            f"binary reports {backends!r}, missing {missing}. "
            "This usually means the pinned build did not pick up expected "
            "llama.cpp CMake defaults on this host."
        )
    return None


def missing_tool_message(path: Path) -> str:
    return (
        f"llama.cpp binary missing or not executable: {path}\n"
        "  Build the pinned benchmark target with:\n"
        "    uv run scripts/bench/ensure_llama_cpp.py\n"
        "  Or pass --allow-unpinned-lcpp with an explicit --llama-bench/--lcpp-bin."
    )
