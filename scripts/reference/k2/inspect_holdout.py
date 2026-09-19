# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Read-only summary and reference-score inspection of frozen K2 holdout evidence."""

import argparse
import hashlib
import heapq
import json
import math
import struct

from pathlib import Path
from inspect_oracle_metrics import read_json


def reference_disagreement(directory, row):
    lane = "ordinary" if row["lane"] == "teacher_forced" else "greedy"
    path = (
        directory
        / f"{row['corpus']}-{row['base']}"
        / lane
        / f"reference-{row['base']}.f32"
    )
    vocab = 250624
    row_bytes = 8 + vocab * 4
    if path.stat().st_size != 24 + 256 * row_bytes:
        raise ValueError("wrong reference extent")
    with path.open("rb") as stream:
        header = stream.read(24)
        if header[:8] != b"K2REF001" or struct.unpack("<4I", header[8:]) != (
            vocab,
            256,
            row["base"],
            16,
        ):
            raise ValueError("wrong reference header")
        stream.seek(24 + (row["visible_length"] - 1) * row_bytes)
        coordinates = struct.unpack("<2I", stream.read(8))
        if coordinates != (row["base"] + row["visible_length"] - 1, row["token"]):
            raise ValueError("wrong reference coordinates")
        logits = struct.unpack(f"<{vocab}f", stream.read(vocab * 4))
    if not all(math.isfinite(v) for v in logits):
        raise ValueError("nonfinite reference row")
    top = heapq.nlargest(2, range(vocab), key=lambda i: (logits[i], i))
    actual = row["metrics"]["logits"]["actual_top1"]
    expected = row["metrics"]["logits"]["reference_top1"]
    if top[0] != expected or not 0 <= actual < vocab:
        raise ValueError("top-1 evidence mismatch")
    denominator = sum(math.exp(v - logits[expected]) for v in logits)
    result = {
        "corpus": row["corpus"],
        "base": row["base"],
        "lane": row["lane"],
        "visible_length": row["visible_length"],
        "reference_top_two_ids": top,
        "native_choice": actual,
        "reference_top_two_gap": logits[top[0]] - logits[top[1]],
        "reference_logit_regret_of_native_choice": logits[expected] - logits[actual],
        "reference_probability_gap_to_native_choice": (
            1 - math.exp(logits[actual] - logits[expected])
        )
        / denominator,
    }
    if "ranking" in row["metrics"]:
        result["recorded_live_ranking_witness"] = row["metrics"]["ranking"]
        result["accepted_ranking_indeterminate"] = row["accepted_ranking_indeterminate"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    summary = read_json(args.directory / "summary.json")
    manifest = read_json(args.directory / "manifest.json")
    reference_directory = args.directory
    if "retained_directory" in manifest:
        reference_directory = Path(manifest["retained_directory"])
        for name, key in [
            ("manifest.json", "retained_manifest_sha256"),
            ("references.json", "retained_references_sha256"),
        ]:
            with (reference_directory / name).open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != manifest[key]:
                raise ValueError("retained reference metadata digest changed")
    expected_policy = {
        "k2.guarded-context-holdout.v1": "65a517ad27cab60a7a38989ce8cf3499902940fb94f82c26d83dcca86f5f6889",
        "k2.guarded-context-holdout.v2": "44bd53a7dcf72ae6afb8e1f9921bf461df3f5cf9cc14c0c7705f46188418a1fe",
    }[manifest["policy"]["schema"]]
    if (
        manifest["policy_sha256"] != expected_policy
        or summary["policy_sha256"] != expected_policy
    ):
        raise ValueError("not the declared frozen policy")
    if summary["public_cap_promoted"] is not False:
        raise ValueError("holdout results cannot promote public capacity")
    evidence = read_json(args.directory / "metrics.json")
    rows, captures = evidence["rows"], evidence["captures"]
    if len(rows) != 4096 or len(captures) != 16 or summary["rows"] != len(rows):
        raise ValueError("incomplete frozen experiment")
    for row in rows:
        for metrics in row["metrics"].values():
            if not all(math.isfinite(v) for v in metrics.values()):
                raise ValueError("nonfinite metric")
    failures = [row for row in rows + captures if row["failed_gates"]]
    if len(failures) != summary["failed_records"]:
        raise ValueError("failure-count mismatch")
    if summary["candidate_envelope_passed"] != (not failures):
        raise ValueError("recorded verdict contradicts failures")
    disagreements = [
        reference_disagreement(reference_directory, row)
        for row in rows
        if row["metrics"]["logits"]["actual_top1"]
        != row["metrics"]["logits"]["reference_top1"]
    ]
    result = {
        "experiment_claim": manifest.get("claim", "frozen_holdout_execution"),
        "recorded_verdict": summary,
        "max_abs": max(r["metrics"]["logits"]["max_abs"] for r in rows),
        "max_rmse": max(r["metrics"]["logits"]["rmse"] for r in rows),
        "max_centered_rmse": max(
            r["metrics"]["distribution"]["centered_rmse"] for r in rows
        ),
        "min_cosine": min(r["metrics"]["logits"]["cosine"] for r in rows),
        "max_kl_reference_to_actual": max(
            r["metrics"]["distribution"]["kl_reference_to_actual"] for r in rows
        ),
        "max_total_variation": max(
            r["metrics"]["distribution"]["total_variation"] for r in rows
        ),
        "max_capture_relative_l2": max(r["relative_l2"] for r in captures),
        "min_capture_cosine": min(r["metrics"]["cosine"] for r in captures),
        "generated_tail_failures": [
            r
            for r in failures
            if r.get("lane") == "reference_argmax_trajectory"
            and r["visible_length"] >= 241
        ],
        "top1_disagreements": disagreements,
    }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
