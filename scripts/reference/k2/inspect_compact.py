# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Read-only audit of native F16-versus-Q8 diagnostics, not independent qualification."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct

from inspect_oracle_metrics import read_json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_manifest(manifest):
    policy_path = Path(__file__).with_name("holdout-256-v2.json")
    require(
        manifest["schema"] == "k2.compact_cache_diagnostic.v1",
        "wrong diagnostic schema",
    )
    require(
        manifest["claim"]
        == "native_q8_vs_native_f16_not_independent_oracle_or_new_holdout",
        "wrong scope",
    )
    require(
        manifest["policy_sha256"]
        == digest(policy_path)
        == "44bd53a7dcf72ae6afb8e1f9921bf461df3f5cf9cc14c0c7705f46188418a1fe",
        "policy digest drift",
    )
    require(manifest["policy"] == read_json(policy_path), "policy payload drift")
    for key in ["model_sha256", "tokenizer_metadata_id"]:
        require(manifest[key] == manifest["policy"][key], f"{key} drift")
    require(
        manifest["compact_policy_sha256"]
        == digest(Path(__file__).with_name("COMPACT-KV-POLICY.md")),
        "compact policy drift",
    )
    expected = [
        (c["name"], base)
        for c, base in zip(
            manifest["policy"]["corpora"],
            manifest["policy"]["continuation"]["bases_by_corpus"],
            strict=True,
        )
    ]
    require(
        [
            (name, base)
            for name, base, _, selected in manifest["inputs"]
            if selected is True
        ]
        == expected
        and len(manifest["inputs"]) == 4,
        "selected corpus/base drift",
    )


def inspect(directory):
    manifest = read_json(directory / "manifest.json")
    summary = read_json(directory / "summary.json")
    evidence = read_json(directory / "metrics.json")
    controls = read_json(directory / "controls.json")
    memory = read_json(directory / "memory.json")
    validate_manifest(manifest)
    for record in (manifest, summary):
        require(
            record["public_default_changed"] is False
            and record["performance_claim"] is False,
            "unsupported promotion/claim",
        )
    inputs = {
        (name, base): tokens
        for name, base, tokens, selected in manifest["inputs"]
        if selected
    }
    require(len(inputs) == len(controls) == 4, "incomplete controls")
    tails = {}
    for control in controls:
        key = (control["corpus"], control["base"])
        require(key in inputs and key not in tails, "wrong/duplicate control")
        tokens = inputs[key]
        corpus = next(c for c in manifest["policy"]["corpora"] if c["name"] == key[0])
        require(
            len(tokens) == 256
            and hashlib.sha256(struct.pack("<256i", *tokens)).hexdigest()
            == corpus["prefix_sha256_i32le"],
            "input digest drift",
        )
        tails[key] = control["tail_ids"]
        require(
            len(tails[key]) == 16 and tails[key][0] == tokens[240],
            "wrong trajectory prefix",
        )
        for field, hash_field, count, base in [
            ("file", "sha256", 256, key[1]),
            ("tail_file", "tail_sha256", 16, key[1] + 240),
        ]:
            path = directory / Path(control[field]).name
            require(digest(path) == control[hash_field], "control digest drift")
            require(
                path.stat().st_size == 24 + count * (8 + 250624 * 4),
                "wrong control extent",
            )
            with path.open("rb") as stream:
                require(
                    stream.read(24)
                    == b"K2REF001" + struct.pack("<4I", 250624, count, base, 16),
                    "wrong control header",
                )
        path = directory / Path(control["file"]).with_suffix(".captures.f32le").name
        require(
            path.stat().st_size == 4 * 4096 * 4
            and digest(path) == control["captures_sha256"],
            "capture digest/extent drift",
        )
    rows, captures = evidence["rows"], evidence["captures"]
    require(
        len(rows) == summary["rows"] == 1088
        and len(captures) == summary["capture_sites"] == 16,
        "incomplete evidence",
    )
    expected = {
        (name, base, "teacher_forced", n): token
        for (name, base), tokens in inputs.items()
        for n, token in enumerate(tokens, 1)
    }
    expected.update(
        {
            (name, base, "f16_derived_trajectory", n): token
            for (name, base), tokens in tails.items()
            for n, token in enumerate(tokens, 241)
        }
    )
    seen = set()
    for row in rows:
        key = (row["corpus"], row["base"], row["lane"], row["visible_length"])
        require(
            key in expected and key not in seen and row["token"] == expected[key],
            "wrong/duplicate row",
        )
        seen.add(key)
        require(
            all(
                math.isfinite(v)
                for group in row["metrics"].values()
                for v in group.values()
            ),
            "nonfinite row metric",
        )
        mismatch = (
            row["metrics"]["logits"]["actual_top1"]
            != row["metrics"]["logits"]["reference_top1"]
        )
        exact = row["lane"] == "f16_derived_trajectory"
        require(
            row["exact_top1_mismatch"] == mismatch
            and row["exact_predictor_required"] == exact,
            "ranking label drift",
        )
        require(
            not (exact and mismatch) or "top1" in row["v2_failed_gates"],
            "missing exact predictor failure",
        )
    require(
        {(c["corpus"], c["base"], c["layer"]) for c in captures}
        == {(name, base, layer) for name, base in inputs for layer in [0, 11, 23, 35]},
        "capture coverage drift",
    )
    require(
        all(
            math.isfinite(v)
            for c in captures
            for v in [c["relative_l2"], *c["metrics"].values()]
        ),
        "nonfinite capture metric",
    )
    failed_rows = sum(bool(r["v2_failed_gates"]) for r in rows)
    failed_captures = sum(bool(c["failed_gates"]) for c in captures)
    require(
        summary["failed_quality_rows"] == failed_rows
        and summary["failed_quality_capture_sites"] == failed_captures,
        "failure-count drift",
    )
    require(
        summary["v2_quality_envelope_passed"] == (failed_rows + failed_captures == 0),
        "verdict drift",
    )
    require(
        summary["top1_mismatches"] == sum(r["exact_top1_mismatch"] for r in rows),
        "top1-count drift",
    )
    require(
        summary["exact_trajectory_predictors"] == 64
        and summary["trajectory_predictor_mismatches"]
        == sum(r["exact_top1_mismatch"] for r in rows if r["exact_predictor_required"]),
        "trajectory-count drift",
    )
    require(
        summary["invariants_passed"] is True
        and summary["q8_bitwise_partition_controls"]
        == summary["q8_identity_transport_controls"]
        == 4,
        "incomplete invariant controls",
    )
    for mode, size in [("F16", 147456), ("Q8_0", 78336)]:
        require(
            memory[mode]["actual_logical_cache_bytes"]
            == memory[mode]["cache_buffer_length"]
            == size * 256,
            "wrong cache size",
        )
    require(
        memory["Q8_0"]["cache_allocated_bytes"]
        < memory["F16"]["cache_allocated_bytes"],
        "no measured allocation saving",
    )
    return {
        "claim": manifest["claim"],
        "recorded_verdict": summary,
        "memory": memory,
        "row_gate_failure_counts": dict(
            Counter(g for r in rows for g in r["v2_failed_gates"])
        ),
        "capture_gate_failure_counts": dict(
            Counter(g for c in captures for g in c["failed_gates"])
        ),
        "extrema": {
            f"{direction}_{metric}": reducer(r["metrics"][group][metric] for r in rows)
            for direction, reducer, group, metric in [
                ("max", max, "logits", "max_abs"),
                ("max", max, "logits", "rmse"),
                ("min", min, "logits", "cosine"),
                ("max", max, "distribution", "centered_rmse"),
                ("max", max, "distribution", "total_variation"),
                ("max", max, "distribution", "kl_reference_to_actual"),
            ]
        },
        "max_capture_relative_l2": max(c["relative_l2"] for c in captures),
        "min_capture_cosine": min(c["metrics"]["cosine"] for c in captures),
        "failed_rows_by_lane": dict(
            Counter(r["lane"] for r in rows if r["v2_failed_gates"])
        ),
        "top1_disagreements": [
            {
                k: r[k]
                for k in [
                    "corpus",
                    "base",
                    "lane",
                    "visible_length",
                    "metrics",
                    "v2_failed_gates",
                ]
            }
            for r in rows
            if r["exact_top1_mismatch"]
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect(args.directory), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
