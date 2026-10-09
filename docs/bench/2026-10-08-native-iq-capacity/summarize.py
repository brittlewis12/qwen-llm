# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Report all native-IQ primitive cells; completion is not qualification."""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

for filename in sys.argv[1:]:
    events = [json.loads(line) for line in Path(filename).read_text().splitlines()]
    print(filename)
    header = next(e for e in events if e["event"] == "header")
    print(
        "dtype",
        header["dtype"],
        "tensors",
        header["cohort_count"],
        "avoided_logical_GiB",
        header["cohort_avoidable_persistent_logical_bytes"] / 2**30,
    )
    cells = defaultdict(dict)
    for e in events:
        if e["event"] == "performance_attempt":
            cells[(e["tensor"], e["N"])][e["label"]] = e
    print("tensor N metric f32_ms native_ms speedup pair1 pair2")
    for key, arms in sorted(cells.items()):
        for metric in ("command_gpu_ms", "wall_ms"):
            values = [arms.get(a, {}).get(metric) for a in ("A1", "B1", "A2", "B2")]
            if not all(v is not None and math.isfinite(v) and v > 0 for v in values):
                print(*key, metric, "INVALID_OR_INCOMPLETE")
                continue
            a1, b1, a2, b2 = values
            print(
                *key,
                metric,
                f"{(a1 + a2) / 2:.4f}",
                f"{(b1 + b2) / 2:.4f}",
                f"{(a1 + a2) / (b1 + b2):.3f}",
                f"{a1 / b1:.3f}",
                f"{a2 / b2:.3f}",
            )
    differences = defaultdict(list)
    for e in events:
        if e["event"] == "output_comparison":
            differences[(e["phase"], e["arm"])].append(e)
    print("phase arm outputs nonfinite max_oracle_abs max_oracle_relative_l2")
    for key, rows in sorted(differences.items()):
        print(
            *key,
            len(rows),
            sum(e["nonfinite"] for e in rows),
            max(e["cpu_codec_f64_difference"]["max_abs"] for e in rows),
            max(e["cpu_codec_f64_difference"]["relative_l2"] for e in rows),
        )
    print(
        json.dumps(
            {
                "invalid_timings": sum(
                    e.get("command_gpu_valid") is False for e in events
                ),
                "admission_refusals": sum(e.get("admitted") is False for e in events),
                "full_output_max_abs_difference": max(
                    (
                        e["difference"]["max_abs"]
                        for e in events
                        if e["event"] == "output_comparison"
                        and e["phase"] == "representative"
                        and e.get("difference") is not None
                    ),
                    default=None,
                ),
                "full_output_max_relative_l2_difference": max(
                    (
                        e["difference"]["relative_l2"]
                        for e in events
                        if e["event"] == "output_comparison"
                        and e["phase"] == "representative"
                        and e.get("difference") is not None
                    ),
                    default=None,
                ),
                "completion": [e for e in events if e["event"] == "complete"],
            },
            indent=2,
        )
    )
