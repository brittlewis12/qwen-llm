# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Summarize retained native XS leaf/model evidence; no numerical promotion gates."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from statistics import median


def variant(row):
    return "A" if row["label"].startswith(("A", "warm_A")) else "B"


def location(row):
    return {key: row[key] for key in ("tensor", "N", "round", "label")}


def maxima(rows, field):
    rows = [row for row in rows if row.get(field) is not None]
    return {
        metric: {"value": worst[field][metric], **location(worst)}
        for metric in ("max_abs", "relative_l2")
        if rows
        for worst in [max(rows, key=lambda row: row[field][metric])]
    }


def summarize_leaf(path):
    payload = path.read_bytes()
    rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
    events = Counter(row["event"] for row in rows)
    problems = []

    def check(ok, message):
        if not ok:
            problems.append(message)

    headers = [r for r in rows if r["event"] == "header"]
    if len(headers) != 1:
        raise ValueError("expected exactly one header")
    header = headers[0]
    check(header["schema"] == "iq2_xs.native_mma.leaf.v1", "unknown schema")
    check(events["complete"] == 1 and rows[-1]["event"] == "complete", "missing final completion")
    attempts = [r for r in rows if r["event"] == "leaf_attempt"]
    witnesses = [r for r in rows if r["event"] == "witness_attempt"]
    outputs = [r for r in rows if r["event"] == "output_comparison"]
    cells = []
    for rep in header["representatives"]:
        for n in header["widths"]:
            name = rep["name"]
            measured = [r for r in attempts if r["tensor"] == name and r["N"] == n]
            expected_keys = {(i, label) for i in range(1, header["rounds"] + 1)
                             for label in ("A1", "B1", "B2", "A2")}
            check(len(measured) == len(expected_keys) and
                  {(r["round"], r["label"]) for r in measured} == expected_keys,
                  f"incomplete/duplicate ABBA: {name} N{n}")
            valid = [r for r in measured if r.get("command_gpu_valid") is True
                     and not r.get("error") and r.get("measured") is True
                     and r.get("census_observer") is False
                     and not r.get("dispatch_census")
                     and all(isinstance(r.get(k), (int, float)) and math.isfinite(r[k])
                             and r[k] > 0 for k in ("command_gpu_ms", "wall_ms"))]
            check(len(valid) == len(measured), f"invalid timed attempt: {name} N{n}")
            if len(valid) != len(expected_keys):
                continue  # Explicitly marked incomplete above, never silently average a partial cell.
            arms = {v: [r for r in valid if variant(r) == v] for v in ("A", "B")}
            timing = {v: {k: median(r[k] for r in arm) for k in ("command_gpu_ms", "wall_ms")}
                      for v, arm in arms.items()}
            drift, pair_gpu, pair_wall = [], [], []
            for i in range(1, header["rounds"] + 1):
                arm = {r["label"]: r for r in valid if r["round"] == i}
                drift.append({"round": i, **{k: arm["A2"][k] / arm["A1"][k]
                                             for k in ("command_gpu_ms", "wall_ms")}})
                for a, b in (("A1", "B1"), ("A2", "B2")):
                    pair_gpu.append(1 - arm[b]["command_gpu_ms"] / arm[a]["command_gpu_ms"])
                    pair_wall.append(1 - arm[b]["wall_ms"] / arm[a]["wall_ms"])
            cell_outputs = [r for r in outputs if r["tensor"] == name and r["N"] == n]
            expected_output_keys = expected_keys | {(0, "warm_A_witness"), (0, "warm_B_witness")}
            check(len(cell_outputs) == len(expected_output_keys) and
                  {(r["round"], r["label"]) for r in cell_outputs} == expected_output_keys,
                  f"incomplete/duplicate outputs: {name} N{n}")
            hashes = {v: len({r["output_sha256_f32_le"] for r in cell_outputs if variant(r) == v})
                      for v in ("A", "B")}
            control_equal = len({r["output_sha256_f32_le"] for r in cell_outputs}) == 1 if n == 1 else None
            cells.append({"tensor": name, "N": n, "samples_per_arm": len(arms["A"]),
                          "median_ms": timing,
                          "gpu_speedup": timing["A"]["command_gpu_ms"] / timing["B"]["command_gpu_ms"],
                          "wall_speedup": timing["A"]["wall_ms"] / timing["B"]["wall_ms"],
                          "gpu_range_ms": {v: [min(r["command_gpu_ms"] for r in arm),
                                               max(r["command_gpu_ms"] for r in arm)] for v, arm in arms.items()},
                          "A2_over_A1": drift, "paired_gpu_savings": pair_gpu,
                          "paired_wall_savings": pair_wall, "distinct_hashes": hashes,
                          "N1_all_hashes_equal": control_equal})
    for row in attempts + witnesses:
        check(not row.get("error"), f"attempt error: {row.get('error')}")
        if row.get("expected_mma_substitutions") is not None:
            check(row["mma_substitutions"] == row["expected_mma_substitutions"],
                  f"substitution mismatch: {row['tensor']} N{row['N']} {row['label']}")
    for row in witnesses:
        check(row["measured"] is False and row["census_observer"] is True,
              "witness incorrectly timed")
        check(len(row["dispatch_census"]) == 1, "expected one leaf dispatch per witness")
    for row in rows:
        if row["event"] == "admission":
            check(row["admitted"], f"admission failed: {row['phase']}")
    expected_cells = len(header["representatives"]) * len(header["widths"])
    check(len(attempts) == expected_cells * header["rounds"] * 4, "unexpected timed count")
    check(len(witnesses) == expected_cells * 2, "unexpected witness count")
    if rows[-1]["event"] == "complete":
        check(rows[-1]["timed_attempts"] == len(attempts) and
              rows[-1]["untimed_witness_attempts"] == len(witnesses), "completion count mismatch")
    return {"file": str(path), "sha256": hashlib.sha256(payload).hexdigest(),
            "events": dict(events), "device": header["device"],
            "source_binding": header["source_binding"], "packet_problems": problems,
            "witness_kernels": dict(Counter(d["kernel"] for r in witnesses for d in r["dispatch_census"])),
            "mma_substitutions": {"witness": sum(r["mma_substitutions"] for r in witnesses),
                                  "timed": sum(r["mma_substitutions"] for r in attempts)},
            "nonfinite_output_elements": sum(r["nonfinite"] for r in outputs),
            "cpu_oracle": {v: maxima([r for r in outputs if variant(r) == v], "cpu_codec_f64_difference")
                           for v in ("A", "B")},
            "full_output_B_vs_A1": maxima([r for r in outputs if variant(r) == "B"], "difference"),
            "full_output_A2_vs_A1": maxima([r for r in outputs if r["label"] == "A2"], "difference"),
            "cells": cells}


def model_metrics(rows):
    keys = ("max_abs", "relative_l2", "kl_reference_actual", "kl_actual_reference",
            "reference_choice_regret", "actual_choice_regret")
    return {"comparisons": len(rows),
            "top1_disagreements": sum(r["metrics"]["reference_top1"] != r["metrics"]["actual_top1"] for r in rows),
            "maxima": {key: {"value": worst["metrics"][key],
                             **{k: worst[k] for k in ("stream", "N", "round", "label", "teacher_forced_steps")}}
                       for key in keys if rows for worst in [max(rows, key=lambda r: r["metrics"][key])]}}


def summarize_model(path):
    payload = path.read_bytes()
    rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
    events = Counter(r["event"] for r in rows)
    problems = []

    def check(ok, message):
        if not ok:
            problems.append(message)

    def unique(event):
        found = [r for r in rows if r["event"] == event]
        if len(found) != 1:
            raise ValueError(f"expected exactly one {event}")
        return found[0]

    header = unique("header")
    check(header["schema"] == "iq2_xs.native_mma.model.v1", "unknown schema")
    selection = header.get("selection_policy", {
        "candidate": "mma", "A_scope": "Some(Scalar)", "B_scope": "Some(Mma)",
        "provenance": "historical revision1 forced-MMA packet; production remained scalar",
    })
    check(events["complete"] == 1 and rows[-1]["event"] == "complete", "missing final completion")
    attempts = [r for r in rows if r["event"] == "whole_attempt"]
    outputs = [r for r in rows if r["event"] == "output"]
    comparisons = [r for r in rows if r["event"] == "comparison"]
    continuations = [r for r in rows if r["event"] == "continuation"]
    witnesses = [r for r in rows if r["event"] == "warm_kernel_witness"]
    streams = [r["name"] for r in rows if r["event"] == "stream"]
    projection_count = unique("xs_projection_inventory")["per_eligible_chunk"]
    cells = []
    expected_arms = {(0, "warm_A"), (0, "warm_B")} | {
        (i, label) for i in range(1, header["rounds"] + 1) for label in ("A1", "B1", "B2", "A2")}
    steps = range(header["continuations"] + 1)
    for stream in streams:
        for n in header["widths"]:
            matches = lambda r: r["stream"] == stream and r["N"] == n
            cell_attempts = [r for r in attempts if matches(r)]
            check(len(cell_attempts) == len(expected_arms) and
                  {(r["round"], r["label"]) for r in cell_attempts} == expected_arms,
                  f"incomplete/duplicate trajectories: {stream} N{n}")
            for group, label, expected in (
                (outputs, "outputs", {(i, arm, s) for i, arm in expected_arms for s in steps}),
                (continuations, "continuations", {(i, arm, s) for i, arm in expected_arms for s in steps if s}),
                (comparisons, "comparisons", {(i, arm, s) for i, arm in expected_arms for s in steps
                                               if arm not in ("warm_A", "A1")}),
            ):
                selected = [r for r in group if matches(r)]
                check(len(selected) == len(expected) and
                      {(r["round"], r["label"], r["teacher_forced_steps"]) for r in selected} == expected,
                      f"incomplete/duplicate {label}: {stream} N{n}")
            chunk = min(n, header["configured_chunk"])
            expected_subs = projection_count * sum(min(chunk, n - start) > 1 for start in range(0, n, chunk))
            singleton_chunks = sum(min(chunk, n - start) == 1 for start in range(0, n, chunk))
            measured = []
            for r in cell_attempts:
                warm = r["round"] == 0
                expected = 0 if variant(r) == "A" else expected_subs
                check(r["mma_substitutions"] == r["expected_mma_substitutions"] == expected,
                      f"substitution mismatch: {stream} N{n} {r['label']}")
                check(r["chunk_rows"] == chunk and r["chunks"] == (n + chunk - 1) // chunk,
                      f"chunk mismatch: {stream} N{n}")
                if header.get("packet_revision", 1) >= 2:
                    check(r["singleton_chunks"] == singleton_chunks and r["expected_singleton_mma_substitutions"] == 0,
                          f"singleton chunk mismatch: {stream} N{n}")
                check(r["measured"] == (not warm) and r["census_observer"] == warm,
                      f"observer scope mismatch: {stream} N{n}")
                if not warm:
                    valid = r["aggregate_gpu_valid"] and all(
                        isinstance(r.get(k), (int, float)) and math.isfinite(r[k]) and r[k] > 0
                        for k in ("gpu_ms", "wall_ms"))
                    check(valid, f"invalid timing: {stream} N{n} {r['label']}")
                    if valid:
                        measured.append(r)
            cell_witnesses = [r for r in witnesses if matches(r)]
            check(len(cell_witnesses) == 2 and {r["label"] for r in cell_witnesses} == {"warm_A", "warm_B"},
                  f"missing/duplicate witnesses: {stream} N{n}")
            for r in cell_witnesses:
                a = variant(r) == "A"
                check(r["iq2_xs_scalar_gemm"] == (expected_subs if a else 0) and
                      r["iq2_xs_mma_gemm"] == (0 if a else expected_subs) and
                      r["mma_substitutions"] == (0 if a else expected_subs), "warm dispatch mismatch")
                if header.get("packet_revision", 1) >= 2:
                    check(r["singleton_chunks"] == singleton_chunks and
                          r["iq2_xs_gemv"] == r["expected_singleton_gemv"] == singleton_chunks * projection_count,
                          f"singleton GEMV witness mismatch: {stream} N{n}")
            other_topology = [{(k["kernel"], k["concurrent"]): k["count"] for k in r["kernels"]
                               if k["kernel"] not in ("kernel_mat_mat_iq2_xs_f32", "kernel_mat_mat_iq2_xs_f32_mma")}
                              for r in cell_witnesses]
            check(len(other_topology) == 2 and other_topology[0] == other_topology[1], "other warm topology differs")
            if len(measured) != header["rounds"] * 4:
                continue
            medians = {v: {k: median(r[k] for r in measured if variant(r) == v) for k in ("gpu_ms", "wall_ms")}
                       for v in ("A", "B")}
            paired = []
            for i in range(1, header["rounds"] + 1):
                arm = {r["label"]: r for r in measured if r["round"] == i}
                paired.append({"round": i,
                               "savings": {k: [1 - arm[b][k] / arm[a][k] for a, b in (("A1", "B1"), ("A2", "B2"))]
                                           for k in ("gpu_ms", "wall_ms")},
                               "A2_over_A1": {k: arm["A2"][k] / arm["A1"][k] for k in ("gpu_ms", "wall_ms")}})
            cell_outputs = [r for r in outputs if matches(r)]
            cells.append({"stream": stream, "N": n, "median_ms": medians,
                          "median_savings": {k: 1 - medians["B"][k] / medians["A"][k] for k in ("gpu_ms", "wall_ms")},
                          "pairs": paired, "B_substitutions_per_prefill": expected_subs,
                          "singleton_chunks": singleton_chunks,
                          "singleton_witnesses": [{k: r.get(k) for k in ("label", "selector_scope", "iq2_xs_gemv",
                                                    "expected_singleton_gemv", "singleton_attribution")}
                                                  for r in cell_witnesses if "iq2_xs_gemv" in r],
                          "B_vs_A1": model_metrics([r for r in comparisons if matches(r) and variant(r) == "B"]),
                          "hash_counts": {v: [len({r["sha256_f32_le"] for r in cell_outputs
                                                   if variant(r) == v and r["teacher_forced_steps"] == s}) for s in steps]
                                          for v in ("A", "B")}})
    for r in rows:
        check(not r.get("error"), f"recorded error: {r.get('error')}")
        if "selection_policy" in header and r["event"] in (
            "whole_attempt", "warm_kernel_witness", "output", "comparison", "continuation"
        ):
            check(r.get("selector_scope") == selection[f"{variant(r)}_scope"],
                  f"selector scope mismatch: {r['event']} {r['label']}")
        if r["event"] == "admission":
            check(r["admitted"], f"admission failed: {r['phase']}")
            if "session_bytes" in r["components"]:
                check(r["dynamic_reserve_bytes"] == 2 * 1024**3, "trajectory dynamic reserve changed")
    for r in continuations:
        check(r["mma_substitutions"] == 0 and not r["measured"], "decode unexpectedly substituted/timed")
    audit = unique("realized_summary")
    bindings = [r for r in rows if r["event"] == "realized_binding"]
    ledger = audit["ledger"]
    check(audit["reconciled"] and all(r["source_dtype"] == r["actual_dtype"] and r["planned_kind"] == "Direct"
                                     for r in bindings), "native binding audit failed")
    check([len(bindings), sum(r["logical_bytes"] for r in bindings)] == ledger["direct_copy"] == ledger["source"],
          "direct-copy ledger mismatch")
    check(all(not any(ledger[k]) for k in ("converted", "derived", "direct_view", "direct_alias", "tail_fallback")),
          "unexpected conversions/derived/views/aliases/fallbacks")
    plan = unique("prepared_policy")["plan"]
    planned = {r["name"]: r for r in plan}
    check(len(plan) == len(planned) == len(bindings) == len({r["name"] for r in bindings}),
          "prepared/realized count or unique-name mismatch")
    for r in bindings:
        p = planned.get(r["name"], {})
        check(p.get("kind") == r["planned_kind"] and p.get("source_dtype") == r["source_dtype"]
              and p.get("shape") == r["shape"] and p.get("resident_logical_bytes") == r["logical_bytes"],
              f"prepared/realized binding mismatch: {r['name']}")
    check(unique("conversion_plan")["all_conversions_eliminated"], "conversion plan not native")
    if rows[-1]["event"] == "complete":
        complete = rows[-1]
        check(complete["cells"] == len(cells) and complete["timed_prefills"] == sum(r["measured"] for r in attempts)
              and complete["warm_prefills"] == sum(not r["measured"] for r in attempts)
              and complete["teacher_forced_tokens"] == len(continuations), "completion count mismatch")
    pricing = {json.dumps(r["components"], sort_keys=True) for r in rows
               if r["event"] == "admission" and "session_bytes" in r["components"]}
    candidate_comparisons = [r for r in comparisons if variant(r) == "B"]
    return {"kind": "model", "file": str(path), "sha256": hashlib.sha256(payload).hexdigest(),
            "events": dict(events), "device": header["device"], "source_binding": header["source_binding"],
            "packet_revision": header.get("packet_revision", 1), "selection_policy": selection,
            "timing_policy": header["timing_policy"], "packet_problems": problems, "cells": cells,
            "native_ledger": ledger, "xs_projection_count": projection_count,
            "nonfinite_output_elements": sum(r["nonfinite"] for r in outputs),
            "candidate_comparisons": model_metrics(candidate_comparisons),
            "candidate_by_step": {s: model_metrics([r for r in candidate_comparisons if r["teacher_forced_steps"] == s])
                                  for s in steps},
            "A2_vs_A1": model_metrics([r for r in comparisons if r["label"] == "A2"]),
            "mma_substitutions": {"timed": sum(r["mma_substitutions"] for r in attempts if r["measured"]),
                                  "warm": sum(r["mma_substitutions"] for r in attempts if not r["measured"]),
                                  "continuation": sum(r["mma_substitutions"] for r in continuations)},
            "session_pricing": [json.loads(p) for p in sorted(pricing)],
            "trajectory_dynamic_reserve_bytes": sorted({r["dynamic_reserve_bytes"] for r in rows
                                                        if r["event"] == "admission" and "session_bytes" in r["components"]}),
            "session_measured_allocation": None,
            "session_measurement_note": "Only loaded and post-session-drop footprint samples; no isolated session allocation delta."}


def summarize(path):
    with path.open() as f:
        header = next(json.loads(line) for line in f if line.strip())
    if header.get("schema") == "iq2_xs.native_mma.model.v1":
        return summarize_model(path)
    return summarize_leaf(path)


def render_model(result):
    print(f"Packet: {result['file']}\nSHA256: {result['sha256']}\nDevice: {result['device']}")
    print("Counts:", result["events"], "\nPacket problems:", result["packet_problems"])
    print("Selection policy:", result["selection_policy"])
    print("\n| Stream | N | GPU median A/B ms | Saving | Wall median A/B ms | Saving | B substitutions |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for c in result["cells"]:
        a, b = c["median_ms"]["A"], c["median_ms"]["B"]
        print(f"| {c['stream']} | {c['N']} | {a['gpu_ms']:.3f}/{b['gpu_ms']:.3f} | {100*c['median_savings']['gpu_ms']:.2f}% "
              f"| {a['wall_ms']:.3f}/{b['wall_ms']:.3f} | {100*c['median_savings']['wall_ms']:.2f}% | {c['B_substitutions_per_prefill']} |")
        for pair in c["pairs"]:
            print(f"  Round {pair['round']}: " + json.dumps(pair))
        print("  Repeat hash counts A/B, endpoint through continuation4:", c["hash_counts"])
        print("  Singleton chunks:", c["singleton_chunks"], "witnesses:", c["singleton_witnesses"])
    for key in ("candidate_comparisons", "candidate_by_step", "A2_vs_A1", "mma_substitutions",
                "native_ledger", "nonfinite_output_elements", "session_pricing", "session_measurement_note", "timing_policy"):
        print(f"\n{key}:\n{json.dumps(result[key], indent=2)}")


def render(result):
    if result.get("kind") == "model":
        return render_model(result)
    print(f"Packet: {result['file']}\nSHA256: {result['sha256']}\nDevice: {result['device']}")
    print("Counts:", json.dumps(result["events"], sort_keys=True))
    print("Packet problems:", result["packet_problems"])
    print("Witness kernels:", result["witness_kernels"], "substitutions:", result["mma_substitutions"])
    print("Nonfinite output elements:", result["nonfinite_output_elements"])
    print("N1 controls:", [{"tensor": c["tensor"], "all_hashes_equal": c["N1_all_hashes_equal"]}
                           for c in result["cells"] if c["N"] == 1])
    print("Unstable per-variant output hashes:", [{"tensor": c["tensor"], "N": c["N"],
                                                 "distinct_hashes": c["distinct_hashes"]}
                                                for c in result["cells"]
                                                if any(n != 1 for n in c["distinct_hashes"].values())])
    print("\nMedians pool measured samples per arm across ABBA rounds; N1 is an unchanged GEMM control.")
    print("\n| Tensor | N | GPU A/B ms | GPU A/B ratio | Wall A/B ms | A2/A1 GPU (rounds) |")
    print("|---|---:|---:|---:|---:|---|")
    for c in result["cells"]:
        a, b = c["median_ms"]["A"], c["median_ms"]["B"]
        drift = ", ".join(f"{r['command_gpu_ms']:.3f}" for r in c["A2_over_A1"])
        print(f"| {c['tensor']} | {c['N']} | {a['command_gpu_ms']:.4f}/{b['command_gpu_ms']:.4f} "
              f"| {c['gpu_speedup']:.3f} | {a['wall_ms']:.4f}/{b['wall_ms']:.4f} | {drift} |")
    for key in ("cpu_oracle", "full_output_B_vs_A1", "full_output_A2_vs_A1"):
        print(f"\n{key}:\n{json.dumps(result[key], indent=2)}")
    print("\nDescriptive drift flags (>10% A2/A1 change or >25% within-arm range; not qualification gates):")
    for c in result["cells"]:
        if any(abs(r[k] - 1) > 0.1 for r in c["A2_over_A1"] for k in ("command_gpu_ms", "wall_ms")) or any(
            hi / lo > 1.25 for lo, hi in c["gpu_range_ms"].values()
        ):
            print(json.dumps({k: c[k] for k in ("tensor", "N", "gpu_range_ms", "A2_over_A1")}))
    print("\nLeaf evidence only. CPU oracle is sampled; full-output differences are relative to scalar A1, not an oracle.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path, nargs="?", default=Path(__file__).with_name("leaf.jsonl"))
    parser.add_argument("--json", action="store_true", help="include per-cell pairs, ranges and wall drift")
    args = parser.parse_args()
    result = summarize(args.packet)
    if args.json:
        print(json.dumps(result, indent=2, allow_nan=False))
    else:
        render(result)
    raise SystemExit(1 if result["packet_problems"] else 0)
