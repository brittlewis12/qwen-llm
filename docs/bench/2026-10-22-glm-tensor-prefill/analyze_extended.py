# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Descriptive paired tensor-prefill screen; no numerical promotion rule."""

import argparse
import hashlib
import json
from pathlib import Path
import runpy

import numpy as np


def analyze(report, repo):
    reference = runpy.run_path(
        str(repo / "scripts/reference/glm53/quality_analysis.py")
    )
    fixture_bytes = (repo / "scripts/reference/glm53/quality-v1.json").read_bytes()
    fixture = json.loads(fixture_bytes)
    rows = [json.loads(line) for line in report.read_text().splitlines()]
    assert rows[0]["schema"] == "glm53.tensor_extended_quality.v1"
    assert rows[0]["fixture_sha256"] == hashlib.sha256(fixture_bytes).hexdigest()
    assert rows[0]["artifact_layout"] == fixture["artifact_layout"]
    assert rows[-1]["event"] == "complete" and rows[-1]["execution_complete"]
    expected = {
        (d["path"], max(d["prefix_lengths"])): d["stratum"]
        for d in fixture["documents"]
    }
    targets = {
        (d["path"], max(d["prefix_lengths"])): [int(t) for t in d["tokens"].split()][
            max(d["prefix_lengths"]) : max(d["prefix_lengths"]) + 64
        ]
        for d in fixture["documents"]
    }
    assert len(expected) == 22
    arms = {arm: {} for arm in ("old", "tensor")}
    tokens = {}
    for row in rows:
        if row["event"] == "document_arm":
            key = (row["path"], row["prefix"])
            assert key in expected and row["stratum"] == expected[key]
            assert key not in arms[row["arm"]]
            assert row["logit_rows"] == 64 and row["actual_prefill_rows"] == 2048
            assert row["final_position"] == row["prefix"] + 63
            arms[row["arm"]][key] = row
        elif row["event"] == "document_token":
            key = (row["arm"], row["path"], row["prefix"])
            tokens.setdefault(key, []).append(row)
    assert all(set(arm) == set(expected) for arm in arms.values())
    for arm, documents in arms.items():
        for key, row in documents.items():
            values = tokens[(arm, *key)]
            assert [v["index"] for v in values] == list(range(64))
            assert all(v["target_position"] == key[1] + v["index"] for v in values)
            assert [v["target_token"] for v in values] == targets[key]
            assert row["continuation_tokens"] == targets[key]
            assert all(v["logits_finite"] for v in values)
            assert np.isfinite([v["nll"] for v in values]).all()
            assert np.isclose(np.mean([v["nll"] for v in values]), row["mean_nll"])
            assert sum(v["top1_hit"] for v in values) == row["top1_hits"]
    documents = []
    for key, stratum in expected.items():
        old, new = (arms[arm][key] for arm in ("old", "tensor"))
        documents.append(
            {
                "path": key[0],
                "prefix": key[1],
                "stratum": stratum,
                "old_nll": old["mean_nll"],
                "tensor_nll": new["mean_nll"],
                "nll_delta": new["mean_nll"] - old["mean_nll"],
                "old_hits": old["top1_hits"],
                "tensor_hits": new["top1_hits"],
                "top1_delta": (new["top1_hits"] - old["top1_hits"]) / 64,
                "argmax_agreements": new["top1_baseline_agreements"],
                "mean_kl": new["mean_kl_baseline_to_arm"],
                "max_kl": new["max_kl_baseline_to_arm"],
            }
        )
    summaries = {}
    for group in ("overall", "prose", "code", "long"):
        selected = [d for d in documents if group == "overall" or d["stratum"] == group]
        strata_of = {d["path"]: d["stratum"] for d in selected}
        summary = {"documents": len(selected), "targets": 64 * len(selected)}
        for metric in ("nll_delta", "top1_delta"):
            by_doc = {d["path"]: [d[metric]] for d in selected}
            samples = reference["bootstrap"](by_doc, strata_of, True, np.mean)
            summary[metric] = {
                "observed_mean": float(np.mean([d[metric] for d in selected])),
                "central90": np.percentile(samples, [5, 95]).tolist(),
            }
        for field in ("old_nll", "tensor_nll", "mean_kl"):
            summary[field] = float(np.mean([d[field] for d in selected]))
        for field in ("old_hits", "tensor_hits", "argmax_agreements"):
            summary[field] = sum(d[field] for d in selected)
        summary["max_kl"] = max(d["max_kl"] for d in selected)
        summaries[group] = summary
    frozen_tasks = {t["id"]: t for t in fixture["tool_tasks"] if t["context_level"] > 0}
    task_rows = [r for r in rows if r["event"] == "task_generation"]
    assert len(task_rows) == 16
    seen = set()
    tool_results = []
    for row in task_rows:
        key = (row["id"], row["arm"], row["generation"]["mode"])
        assert key not in seen and row["id"] in frozen_tasks
        seen.add(key)
        task = frozen_tasks[row["id"]]
        assert row["expected"] == task["expected"]
        assert row["prompt_tokens"] == task["prompt_tokens"]
        assert row["generation"]["mode"] in ("greedy", "sampled")
        if row["generation"]["mode"] == "sampled":
            assert row["generation"]["seed"] == 11
        tool_results.append(
            {
                "id": row["id"],
                "arm": row["arm"],
                "mode": row["generation"]["mode"],
                "strict_correct": reference["tool_correct"](
                    row, task["expected"], row["id"]
                ),
                "stop": row["stop"],
                "emitted_tokens": row["emitted_tokens"],
                "parse_error": row["parse_error"],
                "calls": row["calls"],
            }
        )
    assert seen == {
        (task, arm, mode)
        for task in frozen_tasks
        for arm in ("old", "tensor")
        for mode in ("greedy", "sampled")
    }
    return {
        "schema": "glm53.tensor_extended_descriptive.v1",
        "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        "fixture_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
        "method": "paired document bootstrap from quality_analysis.py; 20,000 resamples, seed 20261008, stratified; one max-prefix item per document; central90 intervals are descriptive, not a noninferiority verdict",
        "overlap": "includes the original three bounded cases; not independent replication",
        "summaries": summaries,
        "documents": documents,
        "tools": tool_results,
        "quality_verdict": None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.report, args.repo)
    with args.out.open("x") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")
    print(json.dumps(result["summaries"], indent=2))
    print(json.dumps(result["tools"], indent=2))
