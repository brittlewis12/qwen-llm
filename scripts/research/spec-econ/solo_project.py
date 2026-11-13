#!/usr/bin/env python3
"""Price current solo NextN from native JSONL, head report and quality JSON.

Usage: solo_project.py NATIVE.jsonl HEAD-report.json QUALITY.json OUT.json
Reports warm-head steady-state and fresh-history amortization. No serve speed
claim: the zero-host-cost ratio is an optimistic delivered ceiling. The ring
charge is K target copies plus an expected rejection restore; head ring and
acceptance bookkeeping are free in this ceiling.
"""
import json
from pathlib import Path
import statistics
import sys


def main():
    native = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line.startswith("{")]
    head = json.loads(Path(sys.argv[2]).read_text())
    quality = json.loads(Path(sys.argv[3]).read_text())
    if not native or native[-1].get("event") != "complete" or head.get("status") != "complete":
        raise RuntimeError("incomplete native cost or head run")
    base = statistics.median(r["ms_token"] for r in native if r["event"] == "gpu_loop" and r["sampling"]["temperature"] == 1)
    ring = next(r for r in native if r["event"] == "ring")
    distribution = next(r["median"] for r in native if r["event"] == "sampler" and r["distribution"])
    sample = next(r["median"] for r in native if r["event"] == "sampler" and not r["distribution"])
    head_ms = statistics.median(r["full_head_step_ms"] for r in head["score_steps"][1:])
    recursive = quality.get("recursive_two_step")
    a1 = recursive["first_step_aligned"]["mean"] if recursive else quality["one_step_filtered"]["mean"]
    a2 = recursive["conditional_second_estimate"] if recursive else quality["second_step_on_trajectory_filtered"]["mean"]
    second_head_ms = statistics.median(head["recursive_head_steps_ms"]) if head.get("recursive_head_steps_ms") else head_ms
    rewinds = head.get("head_position_ring", [])
    if rewinds and not all(r["replay_bitwise"] for r in rewinds):
        raise RuntimeError("head position rewind failed qualification")
    head_rewind_ms = statistics.median(statistics.median(r["restore_ms"]) for r in rewinds) if rewinds else 0
    # Explicit unmeasured allowances, above measured per-row state copies.
    # History catch-up can process accepted unseen head inputs in the next batch.
    history_catchup_budget = 0.25
    adapter_budget = 0.5
    out = {"baseline_gpu_loop_ms":base,"head_ms":head_ms,"second_head_ms":second_head_ms,
           "head_rewind_ms":head_rewind_ms,"history_catchup_budget_ms":history_catchup_budget,
           "adapter_budget_ms":adapter_budget,
           "head_history_ms":head["head_history_init_ms"],"a1":a1,"a2_estimate":a2,
           "second_step_kind":"recursive_importance_estimate" if recursive else "target_hidden_greedy_subset_proxy",
           "sample_ms":sample,"distribution_ms":distribution,"ring":ring,"cases":[]}
    for k in (1,2):
        v = statistics.median(r["median"] for r in native if r["event"] == "verify" and r["causal"] and r["rows"]==k+1)
        accepted = a1 if k==1 else a1*a2
        emitted = 1+a1+(a1*a2 if k==2 else 0)
        # Retain K proposal distributions. Compute target distributions only
        # through the first rejection, or the bonus draw if all drafts survive:
        # E[target distributions] = E[emitted] for either K.
        # The fast-retention budget assumes the ordinary ordered sampler can
        # also retain its weights for free; it is an optimistic unbuilt path.
        # A GPU acceptance/residual implementation has a separate zero-cost
        # sampling ceiling, not a measured implementation price.
        head_cost = head_ms + (second_head_ms if k == 2 else 0)
        # K=1 head endpoint is never ahead of the retained target anchor.
        # K=2 needs one metadata rewind if its first proposed token is rejected.
        head_restore = (1-a1)*head_rewind_ms if k == 2 else 0
        cost = v+head_cost+k*ring["copy_median"]+(1-accepted)*ring["restore_median"]+head_restore
        cost += history_catchup_budget+adapter_budget
        for sampling, extra in [("free_sampling_bound",0),
                                ("fast_retention_budget",(k+emitted)*sample),
                                ("exact_cpu_distributions",(k+emitted)*distribution)]:
            round_ms = cost+extra
            per_token = round_ms/emitted
            out["cases"].append({"k":k,"sampling_cost":sampling,"emitted":emitted,
                "expected_distribution_rows":k+emitted,"verify_ms":v,"round_ms":round_ms,"per_token_ms":per_token,
                "warm_zero_host_speedup":base/per_token,
                "with_host_ms":{str(h):(base+h)/(per_token+h) for h in (0.5,1,2)},
                "fresh_history_speedup":{str(n):n*base/(n*per_token+head["head_history_init_ms"]) for n in (128,256,512,1024,4096)}})
    Path(sys.argv[4]).write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out,indent=2))


if __name__ == "__main__":
    main()
