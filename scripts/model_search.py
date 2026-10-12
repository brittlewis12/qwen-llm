"""Resolve readable regular files by published filename in search-path order.

Filename is the artifact's lookup identity, not a content identity. Consumers
that authenticate an artifact must verify its content after resolving it.
"""

from __future__ import annotations

import fnmatch
import os
import stat
from pathlib import Path

MAX_DEPTH = 3


def search_roots() -> tuple[Path, ...]:
    """Return expanded search roots in the order configured by the caller."""
    value = os.environ.get("QWEN_MODEL_SEARCH_PATH", "")
    return tuple(_expand_home(item) for item in value.split(os.pathsep) if item)


def _expand_home(value: str) -> Path:
    home = os.environ.get("HOME")
    if home and value == "~":
        return Path(home)
    if home and value.startswith("~/"):
        return Path(home) / value[2:]
    return Path(value)


def resolve_model(filename: str) -> Path:
    """Find the first readable regular file with this published filename."""
    for root in search_roots():
        path = _find(root, filename, MAX_DEPTH)
        if path is not None:
            return path
    raise FileNotFoundError(
        f"model fixture {filename} not found; set QWEN_MODEL_SEARCH_PATH"
    )


def models_matching(pattern: str) -> tuple[Path, ...]:
    """List matching readable GGUF files, preserving same-name distinct paths."""
    matches: dict[Path, Path] = {}
    for root in search_roots():
        for path in _walk(root, MAX_DEPTH):
            if fnmatch.fnmatch(path.name, pattern) and _readable_regular_file(path):
                resolved = path.resolve()
                matches.setdefault(resolved, resolved)
    return tuple(matches[path] for path in sorted(matches))


def _readable_regular_file(path: Path) -> bool:
    try:
        if not stat.S_ISREG(path.stat().st_mode):
            return False
        with path.open("rb"):
            return True
    except OSError:
        return False


def _find(root: Path, filename: str, depth: int) -> Path | None:
    try:
        entries = sorted(root.iterdir(), key=lambda path: path.name)
    except OSError:
        return None
    for path in entries:
        if path.name == filename and _readable_regular_file(path):
            return path
    if depth == 0:
        return None
    for path in entries:
        try:
            is_dir = path.is_dir()
        except OSError:
            continue
        if is_dir:
            match = _find(path, filename, depth - 1)
            if match is not None:
                return match
    return None


def _walk(root: Path, depth: int):
    try:
        entries = sorted(root.iterdir(), key=lambda path: path.name)
    except OSError:
        return
    for path in entries:
        if _readable_regular_file(path):
            yield path
        elif depth:
            try:
                is_dir = path.is_dir()
            except OSError:
                continue
            if is_dir:
                yield from _walk(path, depth - 1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    print(resolve_model(parser.parse_args().filename))
