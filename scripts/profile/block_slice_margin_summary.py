#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


DEFAULT_THRESHOLDS = (1e-4, 3e-4, 1e-3, 3e-3, 5e-3)


@dataclass(frozen=True)
class MarginRow:
    path: Path
    start_block: int
    end_block: int
    context: int
    slots: int
    route_order_mismatches: int
    route_set_mismatches: int
    first_set_mismatch: str
    max_logit_abs: float
    max_logit_rms: float
    min_base_margin: float
    min_replay_margin: float
    replay_margin_lt_1e3: int
    replay_margin_lt_5e3: int
    min_x_cos: float
    max_x_abs: float


def parse_rows(path: Path) -> list[MarginRow]:
    header: list[str] | None = None
    rows: list[MarginRow] = []
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
        rows.append(
            MarginRow(
                path=path,
                start_block=int(row["start_block"]),
                end_block=int(row["end_block"]),
                context=int(row["context"]),
                slots=int(row["slots"]),
                route_order_mismatches=int(row["route_order_mismatches"]),
                route_set_mismatches=int(row["route_set_mismatches"]),
                first_set_mismatch=row["first_set_mismatch"],
                max_logit_abs=float(row["max_logit_abs"]),
                max_logit_rms=float(row["max_logit_rms"]),
                min_base_margin=float(row["min_base_margin"]),
                min_replay_margin=float(row["min_replay_margin"]),
                replay_margin_lt_1e3=int(row["replay_margin_lt_1e3"]),
                replay_margin_lt_5e3=int(row["replay_margin_lt_5e3"]),
                min_x_cos=float(row["min_x_cos"]),
                max_x_abs=float(row["max_x_abs"]),
            )
        )
    return rows


def parse_thresholds(raw: str) -> list[float]:
    values: list[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    return values


def format_float(value: float) -> str:
    return f"{value:.6g}"


def format_score(value: float) -> str:
    return f"{value:.9f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize block-slice route-margin sweep rows."
    )
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument(
        "--thresholds",
        default=",".join(format_float(v) for v in DEFAULT_THRESHOLDS),
        help="comma-separated window fallback thresholds for min_replay_margin",
    )
    parser.add_argument(
        "--rows",
        action="store_true",
        help="also print parsed rows with per-threshold fallback markers",
    )
    args = parser.parse_args()

    thresholds = parse_thresholds(args.thresholds)
    rows: list[MarginRow] = []
    for path in args.paths:
        parsed = parse_rows(path)
        if not parsed:
            raise SystemExit(f"no block-slice margin rows found in {path}")
        rows.extend(parsed)

    total = len(rows)
    set_mismatch_rows = sum(1 for row in rows if row.route_set_mismatches > 0)
    order_mismatch_rows = sum(1 for row in rows if row.route_order_mismatches > 0)
    set_mismatches = sum(row.route_set_mismatches for row in rows)
    order_mismatches = sum(row.route_order_mismatches for row in rows)
    min_margin = min(row.min_replay_margin for row in rows)
    min_x_cos = min(row.min_x_cos for row in rows)
    max_x_abs = max(row.max_x_abs for row in rows)
    max_logit_abs = max(row.max_logit_abs for row in rows)
    low_1e3 = sum(row.replay_margin_lt_1e3 for row in rows)
    low_5e3 = sum(row.replay_margin_lt_5e3 for row in rows)

    print(
        "rows\tset_mismatch_rows\tset_mismatches\torder_mismatch_rows"
        "\torder_mismatches\tmin_replay_margin\tmin_x_cos\tmax_x_abs"
        "\tmax_logit_abs\tlow_margin_lt_1e3\tlow_margin_lt_5e3"
    )
    print(
        f"{total}\t{set_mismatch_rows}\t{set_mismatches}\t{order_mismatch_rows}"
        f"\t{order_mismatches}\t{format_float(min_margin)}"
        f"\t{format_score(min_x_cos)}\t{format_float(max_x_abs)}"
        f"\t{format_float(max_logit_abs)}\t{low_1e3}\t{low_5e3}"
    )

    print("threshold\tfallback_rows\tfallback_pct")
    for threshold in thresholds:
        fallback_rows = sum(1 for row in rows if row.min_replay_margin < threshold)
        fallback_pct = 100.0 * fallback_rows / max(1, total)
        print(f"{format_float(threshold)}\t{fallback_rows}\t{fallback_pct:.2f}")

    if args.rows:
        extra = "".join(f"\tfallback_lt_{format_float(th)}" for th in thresholds)
        print(
            "path\tcontext\tslots\tstart_block\tend_block\tset_mismatches"
            "\torder_mismatches\tmin_replay_margin\tmin_x_cos\tmax_x_abs" + extra
        )
        for row in rows:
            markers = "".join(
                "\t" + ("1" if row.min_replay_margin < threshold else "0")
                for threshold in thresholds
            )
            print(
                f"{row.path}\t{row.context}\t{row.slots}\t{row.start_block}"
                f"\t{row.end_block}\t{row.route_set_mismatches}"
                f"\t{row.route_order_mismatches}"
                f"\t{format_float(row.min_replay_margin)}"
                f"\t{format_score(row.min_x_cos)}\t{format_float(row.max_x_abs)}"
                f"{markers}"
            )


if __name__ == "__main__":
    main()
