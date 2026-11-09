#!/usr/bin/env python3
"""Summarize causal layer timings, F64 screens, byte floors and projections."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics as st


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    rows = [json.loads(line) for line in a.input.read_text().splitlines()]
    layers = {(r["phase"], r["layer"]): r for r in rows if r["kind"] == "layer"}
    f64 = {(r["phase"], r["layer"], r["variant"]): r["metrics"] for r in rows if r["kind"] == "f64_operator"}
    timings = defaultdict(list)
    for r in rows:
        if r["kind"] == "timing":
            timings[r["phase"], r["layer"], r["variant"]].append(r)
    result = {"layers": [], "gpu_feedback_us": {}}
    for (phase, layer), meta in sorted(layers.items()):
        variants = {}
        for (ph, ly, name), samples in timings.items():
            if (ph, ly) != (phase, layer):
                continue
            values = [r["gpu_us"] for r in samples]
            check = f64[phase, layer, name]
            base, cand = check["incumbent"], check["candidate"]
            no_worse = all(cand[k] <= base[k] * (1 + 1e-9) for k in ["rms", "max_abs"])
            if check["state"]:
                sb, sc = check["state"]["incumbent"], check["state"]["candidate"]
                no_worse &= all(sc[k] <= sb[k] * (1 + 1e-9) for k in ["rms", "max_abs"])
            variants[name] = {
                "median_us": st.median(values), "mean_us": st.mean(values), "min_us": min(values), "max_us": max(values), "n": len(values),
                "hyper_max_abs": max(r["check"]["hyper_max_abs"] for r in samples),
                "hyper_rel_l2": max(r["check"]["hyper_rel_l2"] for r in samples),
                "f64_screen_no_worse": no_worse, "f64": check,
            }
        fastest = min(variants, key=lambda k: variants[k]["median_us"])
        eligible = [k for k, v in variants.items() if v["f64_screen_no_worse"]]
        fastest_f64 = min(eligible, key=lambda k: variants[k]["median_us"]) if eligible else None
        cell = {
            "phase": phase, "layer": layer, "type": meta["type"], "gate_us": meta["gate_us"],
            "byte_floor_us": meta["metadata"]["floor_us_at_1170_GBs"], "logical_bytes": meta["metadata"]["logical_bytes"],
            "variants": variants, "fastest": fastest, "fastest_f64_screen": fastest_f64,
            "gate_pass_fastest": variants[fastest]["median_us"] <= meta["gate_us"],
            "gate_pass_f64_screen": fastest_f64 is not None and variants[fastest_f64]["median_us"] <= meta["gate_us"],
        }
        result["layers"].append(cell)
    for phase in sorted({r["phase"] for r in rows if r["kind"] == "gpu_feedback"}):
        values = [r["us_per_argmax_and_embedding"] for r in rows if r["kind"] == "gpu_feedback" and r["phase"] == phase]
        result["gpu_feedback_us"][str(phase)] = {"median": st.median(values), "samples": values}
    # Anchor deltas to the measured complete native token, so representative
    # layer byte/type differences do not silently replace the full model.
    first = {r["layer"]: r for r in result["layers"] if r["phase"] == 0}
    publication = {r["layer"]: r for r in result["layers"] if r["phase"] == 3}
    weights = {0: 0.75, 3: 0.25} if set(publication) == {2, 3} else {0: 1.0}
    result["projection_phase_weights"] = weights
    result["period_weighted"] = []
    for layer, cell in first.items():
        variants = {}
        for name in cell["variants"]:
            total = 0
            eligible = True
            for phase, weight in weights.items():
                variant = (first if phase == 0 else publication)[layer]["variants"][name]
                total += weight * variant["median_us"]
                eligible &= variant["f64_screen_no_worse"]
            variants[name] = {"median_weighted_us": total, "f64_screen_no_worse": eligible}
        result["period_weighted"].append({"layer": layer, "variants": variants})
    weighted = {r["layer"]: r["variants"] for r in result["period_weighted"]}
    if set(first) == {2, 3}:
        feedback = sum(weight * result["gpu_feedback_us"][str(phase)]["median"] for phase, weight in weights.items()) / 1000
        result["projection"] = {}
        for selection in ["fastest", "fastest_f64_screen"]:
            gain = 0
            chosen = {}
            for layer, count in [(2, 36), (3, 12)]:
                cell = first[layer]
                available = [k for k, v in weighted[layer].items() if selection == "fastest" or v["f64_screen_no_worse"]]
                name = min(available, key=lambda k: weighted[layer][k]["median_weighted_us"]) if available else "incumbent_serial"
                candidate = weighted[layer][name]["median_weighted_us"]
                base = weighted[layer]["incumbent_serial"]["median_weighted_us"]
                gain += count * (base - candidate) / 1000
                chosen[str(layer)] = name
            gpu_ms = 11.993878474337786 - gain
            result["projection"][selection] = {
                "chosen": chosen, "saved_gpu_ms": gain, "projected_gpu_ms": gpu_ms,
                "current_host_tps": 1000 / (gpu_ms + 1.3971296819122152),
                "encode_hidden_tps": 1000 / (gpu_ms + 0.5342065256622152),
                "resident_loop_upper_tps": 1000 / (gpu_ms + feedback),
            }
    result["caveat"] = "Projections extrapolate layers 2/3 across the 36/12 mix and anchor deltas to the prior full-token GPU measurement. They are not whole-model candidate throughput. F64 checks cover one actual input per phase, not distribution qualification. GPU feedback is greedy argmax plus embedding only; PLE gathers and dynamic loop arguments remain unimplemented. Byte floors count useful weights/state, excluding scratch and redundant reads."
    text = json.dumps(result, indent=2) + "\n"
    if a.output:
        a.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
