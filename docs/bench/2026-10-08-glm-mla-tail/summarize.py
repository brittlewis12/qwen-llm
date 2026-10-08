# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Summarize retained MLA tail attempts, including unchanged controls."""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

for filename in sys.argv[1:]:
    events = [json.loads(line) for line in Path(filename).read_text().splitlines()]
    cells = defaultdict(dict)
    for e in events:
        if e["event"] == "leaf_attempt" or (
            e["event"] == "prefill_attempt" and e["phase"] == "whole_target"
        ):
            key = (e["stream"], e["prefix"], e["rows"], e.get("direction", "whole"))
            cells[key][e["label"]] = e
    print(filename)
    print(
        "stream prefix rows direction metric A_ms B_ms saved_pct pair1_pct pair2_pct A_drift_pct"
    )
    for key, arms in sorted(cells.items()):
        for metric in ("command_gpu_ms", "wall_ms"):
            t = [arms.get(label, {}).get(metric) for label in ("A1", "B1", "A2", "B2")]
            if not all(v is not None and math.isfinite(v) and v > 0 for v in t):
                print(*key, metric, "INCOMPLETE_OR_INVALID")
                continue
            a1, b1, a2, b2 = t
            a, b = (a1 + a2) / 2, (b1 + b2) / 2
            print(
                *key,
                metric,
                *(f"{v:.4f}" for v in (a, b)),
                *(
                    f"{v:.2f}"
                    for v in (
                        100 * (1 - b / a),
                        100 * (1 - b1 / a1),
                        100 * (1 - b2 / a2),
                        100 * (a2 / a1 - 1),
                    )
                ),
            )
    comparisons = defaultdict(list)
    for e in events:
        if (
            e["event"] in ("leaf_output", "output_comparison")
            and e.get("difference") is not None
        ):
            key = (
                e["stream"],
                e["prefix"],
                e["rows"],
                e.get("direction", "whole"),
                "repeat" if e["label"] == "A2" else "candidate",
            )
            comparisons[key].append(e)
    print(
        "stream prefix rows direction comparison count max_abs max_relative_l2 max_bidirectional_kl flips max_regret"
    )
    for key, rows in sorted(comparisons.items()):
        print(
            *key,
            len(rows),
            max(e["difference"]["max_abs"] for e in rows),
            max(e["difference"]["relative_l2"] for e in rows),
            max(
                (
                    max(
                        e.get("kl_reference_actual", 0), e.get("kl_actual_reference", 0)
                    )
                    for e in rows
                )
            ),
            sum(e.get("reference_top1") != e.get("actual_top1") for e in rows),
            max(
                (
                    max(
                        e.get("reference_choice_regret", 0),
                        e.get("actual_choice_regret", 0),
                    )
                    for e in rows
                )
            ),
        )
    print(
        json.dumps(
            {
                "attempts": sum(
                    e["event"] in ("leaf_attempt", "prefill_attempt") for e in events
                ),
                "invalid_attempts": sum(
                    e.get("command_gpu_valid") is False for e in events
                ),
                "nonfinite": sum(e.get("nonfinite", 0) for e in events),
                "topology_mismatches": sum(
                    e["substitutions"] != e["expected_substitutions"]
                    for e in events
                    if "substitutions" in e
                ),
                "completion": [
                    e for e in events if e["event"] in ("complete", "cell_skipped")
                ],
            },
            indent=2,
        )
    )
