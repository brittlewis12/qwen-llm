"""Resolve published model filenames from QWEN_MODEL_SEARCH_PATH."""

from __future__ import annotations

import os
import fnmatch
from pathlib import Path

MAX_DEPTH = 3


def search_roots() -> tuple[Path, ...]:
    """Return search roots in the order configured by the caller."""
    value = os.environ.get("QWEN_MODEL_SEARCH_PATH", "")
    return tuple(Path(item).expanduser() for item in value.split(os.pathsep) if item)


def resolve_model(filename: str) -> Path:
    """Find a published model filename below the configured search roots."""
    for root in search_roots():
        path = _find(root, filename, MAX_DEPTH)
        if path is not None:
            return path
    raise FileNotFoundError(
        f"model fixture {filename} not found; set QWEN_MODEL_SEARCH_PATH"
    )


def resolve_directory(dirname: str) -> Path:
    """Find a named model directory below the configured search roots."""
    for root in search_roots():
        path = _find(root, dirname, MAX_DEPTH)
        if path is not None and path.is_dir():
            return path
    raise FileNotFoundError(
        f"model directory {dirname} not found; set QWEN_MODEL_SEARCH_PATH"
    )


def models_matching(pattern: str) -> tuple[Path, ...]:
    """List matching GGUF files below configured roots without duplicates."""
    matches: dict[str, Path] = {}
    for root in search_roots():
        for path in _walk(root, MAX_DEPTH):
            if path.is_file() and fnmatch.fnmatch(path.name, pattern):
                matches.setdefault(path.name, path)
    return tuple(matches[name] for name in sorted(matches))


def _find(root: Path, filename: str, depth: int) -> Path | None:
    try:
        entries = sorted(root.iterdir(), key=lambda path: path.name)
    except OSError:
        return None
    for path in entries:
        if path.name == filename:
            return path
    if depth == 0:
        return None
    for path in entries:
        if path.is_dir():
            match = _find(path, filename, depth - 1)
            if match is not None:
                return match
    return None


def _walk(root: Path, depth: int):
    try:
        entries = sorted(root.iterdir(), key=lambda path: path.name)
    except OSError:
        return
    yield from entries
    if depth:
        for path in entries:
            if path.is_dir():
                yield from _walk(path, depth - 1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    print(resolve_model(parser.parse_args().filename))
