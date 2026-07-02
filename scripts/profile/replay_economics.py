#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReplayRow:
    path: Path
    start_block: int
    blocks: int
    tokens: int
    baseline_ms_per_tok: float
    replay_ms_per_tok: float
    gross_save_pct: float


@dataclass(frozen=True)
class FallbackRow:
    threshold: str
    fallback_pct: float


def format_float(value: float) -> str:
    return f"{value:.4g}"


def parse_replay(path: Path) -> list[ReplayRow]:
    start_block: int | None = None
    blocks: int | None = None
    header: list[str] | None = None
    baseline_by_tokens: dict[int, float] = {}
    rows: list[ReplayRow] = []
    for line in path.read_text().splitlines():
        if line.startswith("[decode-block-slice-replay]"):
            start_match = re.search(r"\bstart_block=(\d+)", line)
            blocks_match = re.search(r"\bblocks=(\d+)", line)
            if start_match is None or blocks_match is None:
                raise SystemExit(f"missing start_block/blocks in {path}")
            start_block = int(start_match.group(1))
            blocks = int(blocks_match.group(1))
            continue
        if not line or line.startswith("check\t"):
            continue
        parts = line.split("\t")
        if parts[0] == "mode":
            header = parts
            continue
        if header is None or len(parts) != len(header):
            continue
        row = dict(zip(header, parts, strict=True))
        tokens = int(row["tokens"])
        ms_per_tok = float(row["avg_gpu_ms_per_tok"])
        if row["mode"] == "baseline_seq":
            baseline_by_tokens[tokens] = ms_per_tok
        elif row["mode"] == "replay_gdn_batched":
            if start_block is None or blocks is None:
                raise SystemExit(f"missing banner before rows in {path}")
            baseline = baseline_by_tokens.get(tokens)
            if baseline is None:
                raise SystemExit(f"missing baseline tokens={tokens} in {path}")
            rows.append(
                ReplayRow(
                    path=path,
                    start_block=start_block,
                    blocks=blocks,
                    tokens=tokens,
                    baseline_ms_per_tok=baseline,
                    replay_ms_per_tok=ms_per_tok,
                    gross_save_pct=float(row["saving_pct"]),
                )
            )
    if not rows:
        raise SystemExit(f"no replay rows found in {path}")
    return rows


def parse_margin_summary(path: Path) -> list[FallbackRow]:
    rows: list[FallbackRow] = []
    header: list[str] | None = None
    for line in path.read_text().splitlines():
        if not line:
            continue
        parts = line.split("\t")
        if parts[0] == "threshold":
            header = parts
            continue
        if header == ["threshold", "fallback_rows", "fallback_pct"]:
            if len(parts) != len(header):
                continue
            row = dict(zip(header, parts, strict=True))
            rows.append(
                FallbackRow(
                    threshold=row["threshold"],
                    fallback_pct=float(row["fallback_pct"]),
                )
            )
    if not rows:
        raise SystemExit(f"no fallback threshold rows found in {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Model block-slice replay net savings after exact fallback."
    )
    parser.add_argument("--margin-summary", type=Path, required=True)
    parser.add_argument("--replay", type=Path, action="append", required=True)
    args = parser.parse_args()

    fallback_rows = parse_margin_summary(args.margin_summary)
    replay_rows: list[ReplayRow] = []
    for path in args.replay:
        replay_rows.extend(parse_replay(path))

    print(
        "path\tstart_block\tblocks\ttokens\tbaseline_ms_per_tok"
        "\treplay_ms_per_tok\tgross_save_pct\tthreshold\tfallback_pct"
        "\tnet_save_pct_post_fallback"
    )
    for row in sorted(replay_rows, key=lambda r: (r.blocks, r.start_block, r.tokens)):
        for fallback in fallback_rows:
            net = row.gross_save_pct - fallback.fallback_pct
            print(
                f"{row.path}\t{row.start_block}\t{row.blocks}\t{row.tokens}"
                f"\t{format_float(row.baseline_ms_per_tok)}"
                f"\t{format_float(row.replay_ms_per_tok)}"
                f"\t{row.gross_save_pct:.2f}\t{fallback.threshold}"
                f"\t{fallback.fallback_pct:.2f}\t{net:.2f}"
            )


if __name__ == "__main__":
    main()
