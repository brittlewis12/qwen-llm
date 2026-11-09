#!/usr/bin/env python3
"""Compact cache/prefetch results and reconciled cohort/dispatch marginals."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics as st


def stats(xs):
    return {"median": st.median(xs), "mean": st.mean(xs), "min": min(xs), "max": max(xs), "n": len(xs)}


def shapley(values, count):
    result = []
    for i in range(count):
        total = 0
        for mask in range(1 << count):
            if mask & (1 << i):
                continue
            k = mask.bit_count()
            weight = math.factorial(k) * math.factorial(count-k-1) / math.factorial(count)
            total += weight * (values[mask | (1 << i)] - values[mask])
        result.append(total)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    rows = [json.loads(x) for x in a.input.read_text().splitlines()]
    result = {"working_set": {}, "matvec_cache": [], "prefetch": [], "attribution": []}
    for mib in sorted({r["mib"] for r in rows if r["kind"] == "working_set"}):
        samples = [r for r in rows if r["kind"] == "working_set" and r["mib"] == mib]
        result["working_set"][str(mib)] = {"hot_gb_s": stats([r["hot_gb_s"] for r in samples]), "cold_us": stats([r["cold_us"] for r in samples]), "hot_us": stats([r["hot_us"] for r in samples])}
    if result["working_set"]:
        dram = st.median(v["hot_gb_s"]["median"] for k,v in result["working_set"].items() if int(k) >= 384)
        cache = [int(k) for k,v in result["working_set"].items() if v["hot_gb_s"]["median"] >= 1.5*dram]
        result["cache_reference"] = {"large_set_gb_s": dram, "largest_set_over_1_5x_dram_mib": max(cache) if cache else None, "note": "Effective read-cache knee, not a physical SLC size/counter measurement; small sets can include GPU-local caches."}
    sets = {(r["phase"],r["current"],r["next"],r["scope"]): r["bytes"] for r in rows if r["kind"] == "prefetch_set"}
    pair = defaultdict(dict)
    for r in rows:
        if r["kind"] == "prefetch_pair":
            pair[r["phase"],r["current"],r["next"],r["scope"],r["placement"]][r["rep"]] = r["gpu_us"]
    for key, values in sorted(pair.items()):
        phase,current,nxt,scope,placement = key
        baseline = pair[phase,current,nxt,scope,"none"]
        delta = [baseline[rep]-v for rep,v in values.items()]
        forward = [baseline[rep]-v for rep,v in values.items() if rep%2 == 0]
        reverse = [baseline[rep]-v for rep,v in values.items() if rep%2 != 0]
        result["prefetch"].append({"phase":phase,"current":current,"next":nxt,"scope":scope,"placement":placement,"bytes":sets[phase,current,nxt,scope],"pair_us":stats(list(values.values())),"saved_us":stats(delta),"forward_saved_us":st.mean(forward),"reverse_saved_us":st.mean(reverse),"both_orders_positive":min(st.mean(forward),st.mean(reverse)) > 0})
    for key in sorted(sets):
        phase,current,nxt,scope = key
        cold = [r["gpu_us"] for r in rows if r["kind"] == "matvec_cache" and (r["phase"],r["current"],r["next"],r["scope"]) == key and not r["warm"]]
        warm = [r["gpu_us"] for r in rows if r["kind"] == "matvec_cache" and (r["phase"],r["current"],r["next"],r["scope"]) == key and r["warm"]]
        if cold and warm:
            result["matvec_cache"].append({"phase":phase,"current":current,"next":nxt,"scope":scope,"bytes":sets[key],"cold_us":stats(cold),"hot_us":stats(warm),"saved_us":st.median(cold)-st.median(warm)})
    for meta in [r for r in rows if r["kind"] == "attribution_metadata"]:
        phase = meta["phase"]
        names = meta["metadata"]["groups"]
        masks = {}
        for copy in [False,True]:
            masks[copy] = {mask: st.median([r["gpu_us"] for r in rows if r["kind"] == "attribution_mask" and r["phase"] == phase and r["copy_only"] == copy and r["mask"] == mask]) for mask in range(64)}
        phi = [shapley(masks[copy],6) for copy in [False,True]]
        groups = []
        for i,name in enumerate(names):
            steps = [s for s in meta["metadata"]["steps"] if s["group"] == i]
            logical = sum(s["logical_bytes"] for s in steps)
            work = phi[0][i]+phi[1][i]
            groups.append({"name":name,"calls":len(steps),"logical_bytes":logical,"byte_floor_us":logical/1.17e6,"allocated_work_us":work,"excess_us":work-logical/1.17e6})
        dispatches = []
        for step in meta["metadata"]["steps"]:
            i = step["index"]
            omitted = {copy: st.median([r["gpu_us"] for r in rows if r["kind"] == "dispatch_marginal" and r["phase"] == phase and r["copy_only"] == copy and r["omit"] == i]) for copy in [False,True]}
            marginal = masks[False][63]-omitted[False]+masks[True][63]-omitted[True]
            dispatches.append({**step,"marginal_work_us":marginal,"marginal_excess_us":marginal-step["logical_bytes"]/1.17e6})
        full = masks[False][63]
        command = masks[True][0]
        result["attribution"].append({"phase":phase,"full_us":full,"command_control_us":command,"groups":groups,"reconciliation_us":full-command-sum(g["allocated_work_us"] for g in groups),"dispatches":dispatches})
    # Exploratory fixed-policy extrapolation over the actual 47 layer handoffs.
    by_policy = defaultdict(dict)
    for r in result["prefetch"]:
        if r["placement"] != "none":
            by_policy[r["current"],r["next"],r["scope"],r["placement"]][r["phase"]] = r
    winners = []
    for pair_key,count in [((2,3),12),((3,4),11),((4,5),24)]:
        options = []
        for (current,nxt,scope,placement),phases in by_policy.items():
            if (current,nxt) != pair_key or set(phases) != {0,3}:
                continue
            delta = 0.75*phases[0]["saved_us"]["median"]+0.25*phases[3]["saved_us"]["median"]
            positive = all(r["both_orders_positive"] for r in phases.values())
            options.append({"current":current,"next":nxt,"scope":scope,"placement":placement,"weighted_saved_us":delta,"both_orders_both_phases":positive,"count":count})
        if options:
            # Also retain the unconstrained winner to expose negative/noisy cells.
            eligible = [r for r in options if r["both_orders_both_phases"]]
            best = max(eligible,key=lambda r:r["weighted_saved_us"]) if eligible else {"current":pair_key[0],"next":pair_key[1],"weighted_saved_us":0,"count":count,"scope":"none","placement":"none"}
            winners.append({"selected":best,"best_observed":max(options,key=lambda r:r["weighted_saved_us"])})
    if len(winners)==3:
        gain = sum(r["selected"]["count"]*r["selected"]["weighted_saved_us"] for r in winners)/1000
        gpu = 10.82919146271403-gain
        result["projection"] = {"policies":winners,"prefetch_saved_ms":gain,"current_host_tps":1000/(gpu+1.3971296819122152),"resident_feedback_upper_tps":1000/(gpu+0.060818196971013),"note":"Exploratory sum of cold-pair deltas over 24 GG/12 GQ/11 QG handoffs, on top of the prior best-executor extrapolation. Cross-handoff cache interactions and multiple-policy selection require whole-model confirmation; no integrated gain is claimed."}
    result["caveat"] = "Cohort work is a copy-charged Shapley allocation preserving DAG waves, not exclusive hardware timestamps. Dispatch leave-one-out marginals are non-additive. Cache-hot matvec sets use recorded real inputs and private outputs; end-to-end pair time charges all read-touch overhead."
    text = json.dumps(result,indent=2)+"\n"
    if a.output:
        a.output.write_text(text)
    print(text,end="")


if __name__=="__main__":
    main()
