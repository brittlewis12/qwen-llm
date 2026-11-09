#!/usr/bin/env python3
"""Summarize native, causal replay and deliberately noncausal predictors."""
import argparse
import json
from pathlib import Path
import statistics as st


def distribution(values):
    return {"mean": st.mean(values), "median": st.median(values), "min": min(values), "max": max(values), "n": len(values)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--logical-bytes", type=float, default=5603241000)
    a = p.parse_args()
    rows = [json.loads(line) for line in a.input.read_text().splitlines()]
    native = [s for r in rows if r["kind"] == "native" for s in r["samples"]]
    after = [s for r in rows if r["kind"] == "after" for s in r["samples"]]
    cap = next(r["metadata"] for r in rows if r["kind"] == "capture")
    # Two initial stream probes excluded by protocol, not on their performance.
    bw = [r["gb_s"] for r in rows if r["kind"] == "bandwidth" and r["rep"] >= 2]
    result = {
        "native": {k: distribution([s[k] for s in native]) for k in ["wall_ms", "gpu_ms", "encode_ms"]},
        "after": {k: distribution([s[k] for s in after]) for k in ["wall_ms", "gpu_ms", "encode_ms"]} if after else None,
        "ordinary_dispatches": len(cap["steps"]),
        "bandwidth_gb_s": distribution(bw),
        "logical_bytes": a.logical_bytes,
        "byte_floor_ms": a.logical_bytes / (st.median(bw) * 1e6),
    }
    for kind, group in [("replay", "mode"), ("frozen", "mode"), ("launch_floor", "calls")]:
        groups = sorted(set(r[group] for r in rows if r["kind"] == kind))
        result[kind] = {str(g): distribution([r["gpu_ms"] for r in rows if r["kind"] == kind and r[group] == g]) for g in groups}
    ns = [int(n) for n in result["launch_floor"]]
    times = [result["launch_floor"][str(n)]["median"] for n in ns]
    slope = sum((x-st.mean(ns))*(y-st.mean(times)) for x,y in zip(ns,times)) / sum((x-st.mean(ns))**2 for x in ns)
    intercept = st.mean(times)-slope*st.mean(ns)
    result["launch_fit"] = {"ms_per_dispatch": slope, "intercept_ms": intercept}
    wall = result["native"]["wall_ms"]["mean"]
    gpu = result["native"]["gpu_ms"]["mean"]
    command = slope * result["ordinary_dispatches"] + intercept
    result["decomposition_ms"] = {
        "logical_byte_floor": result["byte_floor_ms"],
        "command_control_model": command,
        "kernel_excess_model": gpu-command-result["byte_floor_ms"],
        "host_total_nonoverlap": wall-gpu,
    }
    result["projection"] = {
        "launches_to_48_ms": wall-slope*(result["ordinary_dispatches"]-48),
        "launches_to_48_tps": 1000/(wall-slope*(result["ordinary_dispatches"]-48)),
    }
    if result["frozen"]:
        frozen_serial = result["frozen"]["serial"]["median"]
        for mode in ["segments", "all"]:
            frozen_time = result["frozen"][mode]["median"]
            delta = frozen_serial-frozen_time
            result["projection"][f"{mode}_overlap_saved_ms"] = delta
            result["projection"][f"{mode}_delta_adjusted_ms"] = wall-delta
            result["projection"][f"{mode}_delta_adjusted_tps"] = 1000/(wall-delta)
            result["projection"][f"{mode}_direct_ms"] = wall-gpu+frozen_time
            result["projection"][f"{mode}_direct_tps"] = 1000/(wall-gpu+frozen_time)
            encode = result["native"]["encode_ms"]["mean"]
            result["projection"][f"{mode}_ideal_encode_hidden_tps_range"] = sorted([
                1000/(wall-gpu+frozen_time-encode), 1000/(wall-delta-encode)
            ])
    result["caveat"] = "Command cost is a synthetic allocation; kernel excess also includes unpacking and scratch traffic. Frozen concurrency breaks causality and is a predictor only; do not add its savings to launch savings. Logical traffic is the prior census floor, not DRAM counters."
    text = json.dumps(result, indent=2)+"\n"
    if a.output:
        a.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
