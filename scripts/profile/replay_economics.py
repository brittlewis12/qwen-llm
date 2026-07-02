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
    context: int | None
    baseline_ms_per_tok: float
    replay_ms_per_tok: float
    gross_save_pct: float
    validated_ms_per_tok: float | None
    fallback_slots_avg: float | None
    net_save_pct: float | None


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
                    context=None,
                    baseline_ms_per_tok=baseline,
                    replay_ms_per_tok=ms_per_tok,
                    gross_save_pct=float(row["saving_pct"]),
                    validated_ms_per_tok=None,
                    fallback_slots_avg=None,
                    net_save_pct=None,
                )
            )
    if not rows:
        raise SystemExit(f"no replay rows found in {path}")
    return rows


def parse_real_margin(path: Path) -> list[ReplayRow]:
    header: list[str] | None = None
    rows: list[ReplayRow] = []
    for line in path.read_text().splitlines():
        if not line or line.startswith("["):
            continue
        parts = line.split("\t")
        if parts[0] == "start_block":
            header = parts
            continue
        if header is None or len(parts) != len(header):
            continue
        row = dict(zip(header, parts, strict=True))
        baseline = row.get("baseline_wall_ms_per_tok", "")
        if baseline == "":
            continue
        start_block = int(row["start_block"])
        end_block = int(row["end_block"])
        rows.append(
            ReplayRow(
                path=path,
                start_block=start_block,
                blocks=end_block - start_block,
                tokens=int(row["slots"]),
                context=int(row["context"]),
                baseline_ms_per_tok=float(baseline),
                replay_ms_per_tok=float(row["replay_wall_ms_per_tok"]),
                gross_save_pct=float(row["gross_wall_save_pct"]),
                validated_ms_per_tok=float(row["validated_wall_ms_per_tok"]),
                fallback_slots_avg=float(row["fallback_slots_avg"]),
                net_save_pct=float(row["net_wall_save_pct"]),
            )
        )
    if not rows:
        raise SystemExit(f"no timed real-margin rows found in {path}")
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


def parse_occupancy(raw: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        slot, weight = item.split("=", 1)
        out[int(slot)] = float(weight)
    total = sum(out.values())
    if total <= 0.0:
        raise SystemExit("--occupancy weights must sum above zero")
    return {slot: weight / total for slot, weight in out.items()}


def observed_save(row: ReplayRow) -> float:
    return row.net_save_pct if row.net_save_pct is not None else row.gross_save_pct


def adjusted_save(row: ReplayRow, fallback_pct: float) -> float:
    return observed_save(row) - fallback_pct


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def print_rows(rows: list[ReplayRow], fallback_rows: list[FallbackRow]) -> None:
    print(
        "path\tcontext\tstart_block\tblocks\ttokens\tbaseline_ms_per_tok"
        "\treplay_ms_per_tok\tgross_save_pct\tvalidated_ms_per_tok"
        "\tfallback_slots_avg\tnet_save_pct\tthreshold\tfallback_pct"
        "\tadjusted_net_save_pct"
    )
    scenarios = fallback_rows or [FallbackRow(threshold="observed", fallback_pct=0.0)]
    for row in sorted(rows, key=lambda r: (r.tokens, r.context or -1, r.start_block)):
        for fallback in scenarios:
            print(
                f"{row.path}\t{row.context if row.context is not None else ''}"
                f"\t{row.start_block}\t{row.blocks}\t{row.tokens}"
                f"\t{format_float(row.baseline_ms_per_tok)}"
                f"\t{format_float(row.replay_ms_per_tok)}"
                f"\t{row.gross_save_pct:.2f}"
                f"\t{format_float(row.validated_ms_per_tok or 0.0)}"
                f"\t{format_float(row.fallback_slots_avg or 0.0)}"
                f"\t{observed_save(row):.2f}\t{fallback.threshold}"
                f"\t{fallback.fallback_pct:.2f}"
                f"\t{adjusted_save(row, fallback.fallback_pct):.2f}"
            )


def print_policy(
    rows: list[ReplayRow], fallback_rows: list[FallbackRow], occupancy: str
) -> None:
    weights = parse_occupancy(occupancy)
    by_tokens: dict[int, list[ReplayRow]] = {}
    for row in rows:
        by_tokens.setdefault(row.tokens, []).append(row)

    max_slot = max(weights)
    min_slot_candidates = sorted(set(weights) | set(by_tokens))
    scenarios = fallback_rows or [FallbackRow(threshold="observed", fallback_pct=0.0)]
    print(
        "policy_min_slots\tthreshold\tworkload_weight_replayed"
        "\tbaseline_ms_per_tok\tcharged_ms_per_tok\tblended_save_pct"
    )
    for min_slots in min_slot_candidates:
        if min_slots > max_slot:
            continue
        for fallback in scenarios:
            baseline_total = 0.0
            charged_total = 0.0
            replayed_weight = 0.0
            for slots, weight in weights.items():
                slot_rows = by_tokens.get(slots)
                if not slot_rows:
                    continue
                baseline = mean([row.baseline_ms_per_tok for row in slot_rows])
                save = 0.0
                if slots >= min_slots:
                    replayed_weight += weight
                    save = mean(
                        [adjusted_save(row, fallback.fallback_pct) for row in slot_rows]
                    )
                baseline_total += weight * baseline
                charged_total += weight * baseline * (1.0 - save / 100.0)
            if baseline_total <= 0.0:
                continue
            blended = (baseline_total - charged_total) / baseline_total * 100.0
            print(
                f"{min_slots}\t{fallback.threshold}\t{replayed_weight:.4f}"
                f"\t{baseline_total:.4f}\t{charged_total:.4f}\t{blended:.2f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Model block-slice replay net savings after validation and fallback."
    )
    parser.add_argument("--margin-summary", type=Path)
    parser.add_argument("--replay", type=Path, action="append", default=[])
    parser.add_argument("--real-margin", type=Path, action="append", default=[])
    parser.add_argument(
        "--occupancy",
        help="optional token-step occupancy mix, for example '4=0.2,6=0.3,8=0.5'",
    )
    args = parser.parse_args()

    fallback_rows = (
        parse_margin_summary(args.margin_summary) if args.margin_summary else []
    )
    replay_rows: list[ReplayRow] = []
    for path in args.replay:
        replay_rows.extend(parse_replay(path))
    for path in args.real_margin:
        replay_rows.extend(parse_real_margin(path))
    if not replay_rows:
        raise SystemExit("pass at least one --replay or --real-margin path")

    print_rows(replay_rows, fallback_rows)
    if args.occupancy:
        print_policy(replay_rows, fallback_rows, args.occupancy)


if __name__ == "__main__":
    main()
