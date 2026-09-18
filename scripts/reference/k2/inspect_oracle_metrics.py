# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Read-only summary of a K2 extended oracle evidence directory; no model/GPU use."""

import argparse
import json
import math
from pathlib import Path


def read_json(path):
    with path.open("rb") as stream:
        payload = stream.read(16 * 1024 * 1024 + 1)
    if len(payload) > 16 * 1024 * 1024:
        raise ValueError(f"oversized evidence JSON: {path}")
    return json.loads(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    manifest = read_json(args.directory / "manifest.json")
    cases = read_json(args.directory / "metrics.json")["cases"]
    thresholds = manifest["thresholds"]
    groups = {}
    for row in cases:
        groups.setdefault((row["corpus"], row["base"]), []).append(row)
    summaries = []
    for (name, base), rows in groups.items():
        failures = {key: [] for key in ["max_abs", "rmse", "cosine", "top1"]}
        for row in rows:
            metrics = row["metrics"]
            for key in ["max_abs", "rmse", "cosine"]:
                if not math.isfinite(metrics[key]):
                    raise ValueError(f"nonfinite metric: {row}")
            for key, failed in [
                ("max_abs", metrics["max_abs"] >= thresholds["max_abs_exclusive"]),
                ("rmse", metrics["rmse"] >= thresholds["rmse_exclusive"]),
                ("cosine", metrics["cosine"] <= thresholds["cosine_exclusive_min"]),
                ("top1", metrics["actual_top1"] != metrics["reference_top1"]),
            ]:
                if failed:
                    failures[key].append(row["visible_length"])
        summaries.append(
            {
                "corpus": name,
                "base": base,
                "rows": len(rows),
                "max_abs": max(r["metrics"]["max_abs"] for r in rows),
                "max_rmse": max(r["metrics"]["rmse"] for r in rows),
                "min_cosine": min(r["metrics"]["cosine"] for r in rows),
                "gate_failure_counts": {k: len(v) for k, v in failures.items()},
                "first_failed_lengths": {
                    k: min(v) if v else None for k, v in failures.items()
                },
                "top1_mismatch_lengths": failures["top1"],
            }
        )
    print(json.dumps({"thresholds": thresholds, "corpora": summaries}, indent=2))


if __name__ == "__main__":
    main()
