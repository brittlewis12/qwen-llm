#!/usr/bin/env python3
"""Pair per-position artifact NLL rows and bootstrap over prompt clusters."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from pathlib import Path
from typing import Any

RESAMPLES = 20_000
SEED = 1_196_182_867


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    at = (len(ordered) - 1) * p
    lo = int(at)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (at - lo)


def bootstrap(rows: list[float], rng: random.Random) -> dict[str, float]:
    estimate = sum(rows) / len(rows)
    samples = [
        sum(rows[rng.randrange(len(rows))] for _ in rows) / len(rows)
        for _ in range(RESAMPLES)
    ]
    return {
        "mean": estimate,
        "ci90_low": percentile(samples, 0.05),
        "ci90_high": percentile(samples, 0.95),
        "upper95_one_sided": percentile(samples, 0.95),
        "prompt_means": rows,
    }


def sign_test(rows: list[float]) -> dict[str, int | float]:
    """Exact two-sided sign test over independent prompt-cluster means."""
    nonzero = [value for value in rows if value != 0.0]
    positive = sum(value > 0.0 for value in nonzero)
    negative = len(nonzero) - positive
    n = len(nonzero)
    tail = min(positive, negative)
    p = min(1.0, 2.0 * sum(math.comb(n, k) for k in range(tail + 1)) / (2**n)) if n else 1.0
    return {
        "positive_prompts": positive,
        "negative_prompts": negative,
        "zero_prompts": len(rows) - n,
        "nonzero_prompts": n,
        "two_sided_p": p,
    }


def load_artifact(spec: str) -> tuple[str, dict[str, Any]]:
    name, raw_path = spec.split("=", 1)
    value = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    if value.get("kind") != "artifact_quality":
        raise ValueError(f"{name}: expected artifact_quality JSON in {raw_path}")
    return name, value


def prompt_map(artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {prompt["id"]: prompt for prompt in artifact["prompts"]}


def compare_pair(
    left_name: str,
    left: dict[str, Any],
    right_name: str,
    right: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    if left["family"] != right["family"]:
        raise ValueError(f"cross-family comparison: {left_name} / {right_name}")
    if left["tokenizer_sha256"] != right["tokenizer_sha256"]:
        raise ValueError(f"tokenizer mismatch: {left_name} / {right_name}")
    left_prompts, right_prompts = prompt_map(left), prompt_map(right)
    if left_prompts.keys() != right_prompts.keys():
        raise ValueError(f"prompt cohorts differ: {left_name} / {right_name}")

    per_prompt: list[dict[str, Any]] = []
    paired_positions: list[dict[str, Any]] = []
    for prompt_id in left_prompts:
        a, b = left_prompts[prompt_id], right_prompts[prompt_id]
        if a["token_ids_sha256_i32le"] != b["token_ids_sha256_i32le"]:
            raise ValueError(f"token IDs differ in prompt {prompt_id}")
        if a["targets"] != b["targets"]:
            raise ValueError(f"target count differs in prompt {prompt_id}")
        ao, bo = a["observations"], b["observations"]
        if len(ao) != len(bo):
            raise ValueError(f"observation count differs in prompt {prompt_id}")
        dnll: list[float] = []
        agreements = 0
        for x, y in zip(ao, bo, strict=True):
            if x["position"] != y["position"] or x["target_id"] != y["target_id"]:
                raise ValueError(f"target alignment differs in prompt {prompt_id}")
            signed_dnll = x["nll"] - y["nll"]
            top1_agreement = x["top1_id"] == y["top1_id"]
            dnll.append(signed_dnll)
            agreements += top1_agreement
            paired_positions.append(
                {
                    "prompt_id": prompt_id,
                    "position": x["position"],
                    "target_id": x["target_id"],
                    "left_nll": x["nll"],
                    "right_nll": y["nll"],
                    "left_minus_right_signed_dNLL": signed_dnll,
                    "left_top1_id": x["top1_id"],
                    "right_top1_id": y["top1_id"],
                    "top1_agreement": top1_agreement,
                }
            )
        per_prompt.append(
            {
                "id": prompt_id,
                "stratum": a["stratum"],
                "left_mean_nll": sum(row["nll"] for row in ao) / len(ao),
                "right_mean_nll": sum(row["nll"] for row in bo) / len(bo),
                "left_minus_right_signed_dNLL": sum(dnll) / len(dnll),
                "top1_agreement": agreements / len(dnll),
                "targets": len(dnll),
            }
        )
    code = [row["left_minus_right_signed_dNLL"] for row in per_prompt if row["stratum"] == "code"]
    all_prompts = [row["left_minus_right_signed_dNLL"] for row in per_prompt]
    agreement_num = sum(
        row["top1_agreement"] * row["targets"] for row in per_prompt
    )
    total_targets = sum(row["targets"] for row in per_prompt)
    return {
        "left_minus_right_sign": "positive means left has worse (higher) NLL",
        "left": left_name,
        "right": right_name,
        "equal_prompt_all": bootstrap(all_prompts, rng),
        "equal_prompt_code_only": bootstrap(code, rng),
        "exact_prompt_sign_test_all": sign_test(all_prompts),
        "exact_prompt_sign_test_code_only": sign_test(code),
        "pooled_top1_agreement": agreement_num / total_targets,
        "per_prompt": per_prompt,
        "paired_positions": paired_positions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", action="append", required=True, help="NAME=quality.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    artifacts = dict(load_artifact(spec) for spec in args.artifact)
    families: dict[str, list[str]] = {}
    for name, artifact in artifacts.items():
        families.setdefault(artifact["family"], []).append(name)
    rng = random.Random(SEED)
    comparisons = []
    for family, names in families.items():
        baseline = "A" if family == "deepseek-v4.1-flash-1017" else "D"
        if baseline not in names:
            raise ValueError(f"missing required {family} current default {baseline}")
        candidates = [name for name in names if name != baseline]
        oriented = [(name, baseline) for name in candidates]
        oriented.extend(itertools.combinations(candidates, 2))
        for left_name, right_name in oriented:
            comparisons.append(
                compare_pair(
                    left_name,
                    artifacts[left_name],
                    right_name,
                    artifacts[right_name],
                    rng,
                )
            )
    report = {
        "schema_version": 1,
        "kind": "artifact_choice_analysis",
        "bootstrap": {
            "unit": "prompt cluster",
            "resamples": RESAMPLES,
            "seed": SEED,
            "intervals": "central 90%; upper95_one_sided is the one-sided 95% worsening bound",
        },
        "artifacts": {
            name: {
                "family": artifact["family"],
                "tokenizer_sha256": artifact["tokenizer_sha256"],
                "depth": artifact["depth"],
                "targets_per_prompt": artifact["targets_per_prompt"],
                "prompt_ids": [prompt["id"] for prompt in artifact["prompts"]],
            }
            for name, artifact in artifacts.items()
        },
        "pairwise": comparisons,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
