#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy"]
# ///
"""Reduce a native GLM NextN capture and head-probe logits to quality and
acceptance statistics.

Usage: nextn_quality.py CAPTURE_DIR PROBE_DIR [OUT.json]

CAPTURE_DIR holds manifest.json, tokens.u32 and target-logits.f32 from the
engine capture; PROBE_DIR holds draft-logits.f32 and report.json from the
head probe. Reports signed teacher-forced dNLL/KL (head minus target) plus
full-vocabulary and agent-filtered (T=1, top_p 0.95) one-step and second-step
acceptance. The filtered path replicates sampler-v1's top-p keep rule.
"""

import json
import sys
from pathlib import Path

import numpy as np

TOP_P = 0.95


def stat(x):
    x = np.asarray(x, dtype=np.float64)
    return {
        "mean": float(x.mean()),
        "sem": float(x.std(ddof=1) / np.sqrt(len(x))),
        "median": float(np.median(x)),
        "p10": float(np.percentile(x, 10)),
        "p90": float(np.percentile(x, 90)),
    }


def filtered(logits):
    # temperature 1.0, top_k 0, min_p 0; sampler-v1 keeps the smallest
    # prefix whose cumulative mass is >= top_p * total.
    z = logits - logits.max()
    w = np.exp(z)
    w /= w.sum()
    order = np.argsort(-w)
    ws = w[order]
    cumulative = np.cumsum(ws)
    keep = int(np.searchsorted(cumulative, TOP_P * cumulative[-1], side="left") + 1)
    keep = min(keep, len(ws))
    out = np.zeros(len(w))
    out[order[:keep]] = ws[:keep] / ws[:keep].sum()
    return out


def main():
    capture = Path(sys.argv[1])
    probe = Path(sys.argv[2])
    manifest = json.loads((capture / "manifest.json").read_text())
    report = json.loads((probe / "report.json").read_text())
    vocab = int(manifest["vocab_size"])
    rows = int(manifest["score_count"])
    start = int(manifest["score_start"])
    tokens = np.fromfile(capture / "tokens.u32", dtype=np.uint32)
    target = np.fromfile(capture / "target-logits.f32", dtype=np.float32)
    draft = np.fromfile(probe / "draft-logits.f32", dtype=np.float32)
    target = target.reshape(rows, vocab).astype(np.float64)
    draft = draft.reshape(rows, vocab).astype(np.float64)

    def logsoftmax(x):
        z = x - x.max(axis=1, keepdims=True)
        return z - np.log(np.exp(z).sum(axis=1, keepdims=True))

    lp, lq = logsoftmax(target), logsoftmax(draft)
    p, q = np.exp(lp), np.exp(lq)
    n = min(rows - 1, len(tokens) - start - 1)
    captured = tokens[start + 1 : start + 1 + n].astype(np.int64)
    index = np.arange(n)
    dnll = (-lq[index, captured]) - (-lp[index, captured])
    tv = 0.5 * np.abs(p[:n] - q[:n]).sum(axis=1)

    pp = np.array([filtered(target[i]) for i in range(rows)])
    qq = np.array([filtered(draft[i]) for i in range(rows)])
    one_step = np.array([np.minimum(pp[i], qq[i]).sum() for i in range(rows)])
    head_greedy = draft.argmax(axis=1)
    on_trajectory = head_greedy[:n] == captured
    second = np.array(
        [np.minimum(pp[i + 1], qq[i + 1]).sum() for i in range(n)]
    )
    steps = [s["full_head_step_ms"] for s in report["score_steps"][1:]]
    out = {
        "prompt_tokens": int(manifest["prompt_count"]),
        "scored_rows": rows,
        "sampling": manifest["sampling"],
        "nll_target": stat(-lp[index, captured]),
        "nll_head": stat(-lq[index, captured]),
        "dnll_head_minus_target": stat(dnll),
        "kl_target_head": stat((p[:n] * (lp[:n] - lq[:n])).sum(axis=1)),
        "kl_head_target": stat((q[:n] * (lq[:n] - lp[:n])).sum(axis=1)),
        "tv_target_head": stat(tv),
        "one_step_tv": stat(1.0 - tv),
        "one_step_filtered": stat(one_step),
        "greedy_agreement": float((head_greedy == target.argmax(axis=1)).mean()),
        "second_step_chain_filtered": stat(second),
        "second_step_on_trajectory_rows": int(on_trajectory.sum()),
        "second_step_on_trajectory_filtered": stat(second[on_trajectory]),
        "head_step_ms_median_excluding_first": float(np.median(steps)),
        "head_step_ms_mean_excluding_first": float(np.mean(steps)),
        "head_step_first_ms": float(report["score_steps"][0]["full_head_step_ms"]),
        "head_history_init_ms": report["head_history_init_ms"],
    }
    text = json.dumps(out, indent=2)
    print(text)
    if len(sys.argv) > 3:
        Path(sys.argv[3]).write_text(text + "\n")


if __name__ == "__main__":
    main()
