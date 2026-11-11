#!/usr/bin/env python3
"""Assemble the frozen Metal-kernel prompt from two quality-v1 witnesses."""

from __future__ import annotations

import argparse
from pathlib import Path

SOURCES = (Path("kernels/moe.metal"), Path("kernels/mat_vec.metal"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    pieces = []
    for source in SOURCES:
        pieces.append(f"// SOURCE: {source.as_posix()}\n")
        pieces.append(source.read_text(encoding="utf-8").rstrip())
        pieces.append("\n")
    args.output.write_text("".join(pieces), encoding="utf-8")
    print(f"wrote {args.output}: {args.output.stat().st_size} UTF-8 bytes")


if __name__ == "__main__":
    main()
