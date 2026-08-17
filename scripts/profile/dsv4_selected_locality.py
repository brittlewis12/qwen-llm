#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Selection:
    source: str
    position: int
    layer: int
    visible_rows: int
    selected_ids: tuple[int, ...]


def parse_csv_ints(raw: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in raw.split(",") if item.strip())
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def stable_top_k(scores: list[float], k: int) -> tuple[int, ...]:
    ranked = sorted(range(len(scores)), key=lambda row: (-scores[row], row))
    return tuple(sorted(ranked[:k]))


def validate_selection(
    source: Path,
    layer: int,
    scores: list[float],
    selected_ids: list[int],
    require_cache_order: bool,
) -> tuple[int, ...]:
    if not scores or not all(math.isfinite(score) for score in scores):
        raise ValueError(f"{source}: layer {layer} has invalid visible scores")
    if require_cache_order and any(
        left >= right for left, right in zip(selected_ids, selected_ids[1:])
    ):
        raise ValueError(
            f"{source}: layer {layer} cache-order selected IDs are not strictly ascending"
        )
    selected = tuple(selected_ids if require_cache_order else sorted(selected_ids))
    if len(selected) != len(set(selected)):
        raise ValueError(f"{source}: layer {layer} selected IDs are not unique")
    if not selected or selected[0] < 0 or selected[-1] >= len(scores):
        raise ValueError(f"{source}: layer {layer} selected ID is outside visibility")
    expected = stable_top_k(scores, len(selected))
    if selected != expected:
        difference = len(set(selected).symmetric_difference(expected))
        raise ValueError(
            f"{source}: layer {layer} selected set differs from stable top-k "
            f"by {difference} IDs"
        )
    return selected


def parse_native(path: Path, root: dict[str, Any]) -> list[Selection]:
    position = int(root["position"])
    selections: list[Selection] = []
    for record in root["layers"]:
        csa = record.get("csa")
        if csa is None:
            continue
        layer = int(record["layer"])
        scores = [float(score) for score in csa["visible_scores"]]
        selected = validate_selection(
            path,
            layer,
            scores,
            [int(row) for row in csa["cache_order_selected_ids"]],
            True,
        )
        if int(csa["selected_count"]) != len(selected):
            raise ValueError(f"{path}: layer {layer} selected count mismatch")
        selections.append(Selection(str(path), position, layer, len(scores), selected))
    return selections


def parse_injected(path: Path, root: dict[str, Any]) -> list[Selection]:
    position = int(root["position"])
    tensors = {tensor["name"]: tensor for tensor in root["tensors"]}
    layers = sorted(
        int(name.removeprefix("lid_score_masked-"))
        for name in tensors
        if name.startswith("lid_score_masked-")
    )
    selections: list[Selection] = []
    for layer in layers:
        raw_scores = tensors[f"lid_score_masked-{layer}"]["values"]
        scores = [float(score) for score in raw_scores if score is not None]
        selected = validate_selection(
            path,
            layer,
            scores,
            [int(row) for row in tensors[f"lid_top_k-{layer}"]["values"]],
            False,
        )
        selections.append(Selection(str(path), position, layer, len(scores), selected))
    return selections


def load_selections(path: Path) -> list[Selection]:
    root = json.loads(path.read_text())
    if root.get("schema_version") in {1, 2, 3} and "layers" in root:
        selections = parse_native(path, root)
    elif root.get("schema") == "deepseek_v4_injected_decision_transcript/v1":
        selections = parse_injected(path, root)
    else:
        raise ValueError(f"{path}: unsupported decision transcript schema")
    if not selections:
        raise ValueError(f"{path}: no CSA selections found")
    return selections


def run_lengths(ids: tuple[int, ...]) -> list[int]:
    lengths: list[int] = []
    start = previous = ids[0]
    for row in ids[1:]:
        if row != previous + 1:
            lengths.append(previous - start + 1)
            start = row
        previous = row
    lengths.append(previous - start + 1)
    return lengths


def merged_span_cost(ids: tuple[int, ...], max_gap: int) -> tuple[int, int]:
    spans = 1
    loaded_rows = 1
    previous = ids[0]
    for row in ids[1:]:
        gap = row - previous - 1
        if gap <= max_gap:
            loaded_rows += row - previous
        else:
            spans += 1
            loaded_rows += 1
        previous = row
    return spans, loaded_rows


def page_cost(
    ids: tuple[int, ...], visible_rows: int, page_rows: int
) -> tuple[int, int]:
    pages = sorted({row // page_rows for row in ids})
    loaded_rows = sum(min(page_rows, visible_rows - page * page_rows) for page in pages)
    return len(pages), loaded_rows


def percentile(values: list[float], numerator: int, denominator: int) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * numerator / denominator) - 1)
    return ordered[index]


def summary(values: list[float]) -> dict[str, float | int]:
    return {
        "samples": len(values),
        "min": min(values),
        "mean": sum(values) / len(values),
        "median": percentile(values, 1, 2),
        "p95": percentile(values, 95, 100),
        "max": max(values),
    }


def analyze(
    selections: list[Selection],
    page_rows: tuple[int, ...],
    merge_gaps: tuple[int, ...],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for selection in selections:
        lengths = run_lengths(selection.selected_ids)
        selected_count = len(selection.selected_ids)
        row: dict[str, Any] = {
            **asdict(selection),
            "selected_ids": None,
            "selected_count": selected_count,
            "selection_density": selected_count / selection.visible_rows,
            "runs": len(lengths),
            "mean_run_length": selected_count / len(lengths),
            "max_run_length": max(lengths),
            "contiguous_adjacency_fraction": (selected_count - len(lengths))
            / max(1, selected_count - 1),
            "id_bytes": selected_count * 4,
            "run_descriptor_bytes": len(lengths) * 8,
            "pages": {},
            "merged_spans": {},
        }
        for size in page_rows:
            pages, loaded = page_cost(
                selection.selected_ids, selection.visible_rows, size
            )
            row["pages"][str(size)] = {
                "pages": pages,
                "loaded_rows": loaded,
                "payload_amplification": loaded / selected_count,
            }
        for gap in merge_gaps:
            spans, loaded = merged_span_cost(selection.selected_ids, gap)
            row["merged_spans"][str(gap)] = {
                "spans": spans,
                "loaded_rows": loaded,
                "payload_amplification": loaded / selected_count,
            }
        rows.append(row)

    aggregate: dict[str, Any] = {
        "selection_lists": len(rows),
        "selection_density": summary([row["selection_density"] for row in rows]),
        "runs": summary([float(row["runs"]) for row in rows]),
        "mean_run_length": summary([row["mean_run_length"] for row in rows]),
        "max_run_length": summary([float(row["max_run_length"]) for row in rows]),
        "contiguous_adjacency_fraction": summary(
            [row["contiguous_adjacency_fraction"] for row in rows]
        ),
        "run_descriptor_to_id_bytes": summary(
            [row["run_descriptor_bytes"] / row["id_bytes"] for row in rows]
        ),
        "pages": {},
        "merged_spans": {},
    }
    for size in page_rows:
        aggregate["pages"][str(size)] = {
            "pages": summary([float(row["pages"][str(size)]["pages"]) for row in rows]),
            "payload_amplification": summary(
                [row["pages"][str(size)]["payload_amplification"] for row in rows]
            ),
        }
    for gap in merge_gaps:
        aggregate["merged_spans"][str(gap)] = {
            "spans": summary(
                [float(row["merged_spans"][str(gap)]["spans"]) for row in rows]
            ),
            "payload_amplification": summary(
                [row["merged_spans"][str(gap)]["payload_amplification"] for row in rows]
            ),
        }
    return {"schema_version": 1, "aggregate": aggregate, "layers": rows}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate DSV4 top-k decisions and measure selected-ID locality."
    )
    parser.add_argument("transcripts", nargs="+", type=Path)
    parser.add_argument("--page-rows", type=parse_csv_ints, default=(8, 16, 32, 64))
    parser.add_argument("--merge-gaps", type=parse_csv_ints, default=(1, 3, 7, 15))
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    selections = [
        selection
        for transcript in args.transcripts
        for selection in load_selections(transcript)
    ]
    report = analyze(selections, args.page_rows, args.merge_gaps)
    aggregate = report["aggregate"]

    print(
        "lists\tdensity_mean\truns_median\truns_p95\tmean_run_median"
        "\trun_desc_vs_ids_mean"
    )
    print(
        f"{aggregate['selection_lists']}"
        f"\t{aggregate['selection_density']['mean']:.6f}"
        f"\t{aggregate['runs']['median']:.1f}"
        f"\t{aggregate['runs']['p95']:.1f}"
        f"\t{aggregate['mean_run_length']['median']:.3f}"
        f"\t{aggregate['run_descriptor_to_id_bytes']['mean']:.3f}"
    )
    print("page_rows\tpages_median\tpayload_amp_mean\tpayload_amp_p95")
    for size in args.page_rows:
        page = aggregate["pages"][str(size)]
        print(
            f"{size}\t{page['pages']['median']:.1f}"
            f"\t{page['payload_amplification']['mean']:.3f}"
            f"\t{page['payload_amplification']['p95']:.3f}"
        )
    print("max_gap\tspans_median\tpayload_amp_mean\tpayload_amp_p95")
    for gap in args.merge_gaps:
        span = aggregate["merged_spans"][str(gap)]
        print(
            f"{gap}\t{span['spans']['median']:.1f}"
            f"\t{span['payload_amplification']['mean']:.3f}"
            f"\t{span['payload_amplification']['p95']:.3f}"
        )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
