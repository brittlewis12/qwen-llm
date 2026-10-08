# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Summarize all retained expert-down attempts without promotion decisions."""

import json
import sys
from collections import defaultdict
from pathlib import Path

for path in sys.argv[1:]:
    events = [json.loads(line) for line in Path(path).read_text().splitlines()]
    cells = defaultdict(dict)
    for event in events:
        if event["event"] == "leaf_attempt":
            key = (
                event["rows"],
                event["block"],
                event["control"],
                event["candidate_policy"],
            )
            cells[key][event["label"]] = event["command_gpu_ms"]
    print(path)
    print(
        "rows layer control policy incumbent_ms candidate_ms saved_pct pair1_pct pair2_pct"
    )
    for key, times in sorted(cells.items()):
        if not all(times.get(k) is not None for k in ("A1", "B1", "B2", "A2")):
            print(*key, "INCOMPLETE_OR_INVALID")
            continue
        a, b = (times["A1"] + times["A2"]) / 2, (times["B1"] + times["B2"]) / 2
        print(
            *key,
            f"{a:.4f}",
            f"{b:.4f}",
            f"{100 * (1 - b / a):.2f}",
            f"{100 * (1 - times['B1'] / times['A1']):.2f}",
            f"{100 * (1 - times['B2'] / times['A2']):.2f}",
        )
    whole = defaultdict(dict)
    for event in events:
        if event["event"] == "prefill_attempt" and not event["capture_on"]:
            key = (
                event.get("stream", "historical"),
                event["rows"],
                event.get("candidate_policy", event["phase"]),
            )
            whole[key][event["label"]] = event
    print(
        "stream rows policy metric incumbent_ms candidate_ms saved_pct pair1_pct pair2_pct"
    )
    for key, attempts in sorted(whole.items()):
        for metric in ("command_gpu_ms", "wall_ms"):
            times = {label: event[metric] for label, event in attempts.items()}
            if not all(times.get(k) is not None for k in ("A1", "B1", "B2", "A2")):
                print(*key, metric, "INCOMPLETE_OR_INVALID")
                continue
            a = (times["A1"] + times["A2"]) / 2
            b = (times["B1"] + times["B2"]) / 2
            print(
                *key,
                metric,
                f"{a:.4f}",
                f"{b:.4f}",
                f"{100 * (1 - b / a):.2f}",
                f"{100 * (1 - times['B1'] / times['A1']):.2f}",
                f"{100 * (1 - times['B2'] / times['A2']):.2f}",
            )
    model_diffs = [
        e["comparison"]
        if e["event"] == "domain_output_comparison"
        else e["comparison_to_A1"]
        for e in events
        if e["event"] == "domain_output_comparison"
        or (e["event"] == "whole_output" and e["comparison_to_A1"] is not None)
    ]
    outputs = [e for e in events if e["event"] == "leaf_output"]
    diffs = [
        e[k]["comparison_to_A1"]
        for e in outputs
        for k in ("slots", "weighted")
        if e[k] and e[k]["comparison_to_A1"] is not None
    ]
    print(
        json.dumps(
            {
                "leaf_outputs": len(outputs),
                "all_structural_finite": bool(outputs)
                and all(e["structural_and_finite_success"] for e in outputs),
                "max_abs_diff": max((e["max_abs"] for e in diffs), default=None),
                "max_relative_l2": max((e["relative_l2"] for e in diffs), default=None),
                "completion": [
                    e
                    for e in events
                    if e["event"] in ("leaf_complete", "complete", "whole_skipped")
                ],
                "model_comparisons": len(model_diffs),
                "model_max_abs": max((e["max_abs"] for e in model_diffs), default=None),
                "model_max_relative_l2": max(
                    (e["relative_l2"] for e in model_diffs), default=None
                ),
                "model_max_kl": max(
                    (e["kl_reference_candidate"] for e in model_diffs), default=None
                ),
                "model_top1_mismatches": sum(
                    e["reference_top1"] != e["candidate_top1"] for e in model_diffs
                ),
                "domain_cells": [
                    e
                    for e in events
                    if e["event"] in ("domain_cell_summary", "domain_cell_skipped")
                ],
            },
            indent=2,
        )
    )
