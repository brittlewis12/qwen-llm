# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Audit frontier JSONL; recompute timings/hash comparisons, aggregate recorded numerics.

Full logits/state values are not retained: KL, L2, maxabs and regrets below are
source-computed observations, not independently recalculated from their hashes.
No performance or numerical promotion threshold is imposed.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median


def summarize(path):
    payload = path.read_bytes()
    rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
    events = Counter(r["event"] for r in rows)
    problems = []

    def check(ok, message):
        if not ok:
            problems.append(message)

    def event(name):
        return [r for r in rows if r["event"] == name]

    header, = event("header")
    check(header["schema"] == "flash.frontier_schedule.v1", "unknown schema")
    declared_rounds = header["details"].get("rounds")
    valid_rounds = type(declared_rounds) is int and declared_rounds > 0
    check(valid_rounds, "missing/invalid declared round count")
    expected_rounds = declared_rounds if valid_rounds else 0
    # Older v1 packets declare two rounds but do not record scoped overrides.
    # Do not reinterpret their missing override fields as production None.
    candidate_mode = header["details"].get("candidate_mode")
    if candidate_mode is not None:
        check(candidate_mode in ("forced", "production"), "unknown candidate mode")
        check(header["details"].get("A_schedule_override") is False,
              "incumbent override is not explicit false")
        expected_override = None if candidate_mode == "production" else True
        check("B_schedule_override" in header["details"]
              and header["details"]["B_schedule_override"] is expected_override,
              "candidate header override disagrees with mode")
    check(events["complete"] == 1 and rows[-1]["event"] == "complete"
          and rows[-1].get("execution_complete") is True
          and rows[-1].get("error") is None, "missing/failed final completion")
    check(not any(r.get("error") or r["event"].endswith("_error") for r in rows),
          "packet contains an error")
    check(any(r.get("wired_gate_passed") is True for r in event("production_lease_acquired")),
          "production lease/gate not witnessed")
    check(any(r.get("admitted") is True for r in event("diagnostic_memory_gate")),
          "diagnostic admission missing/refused")
    check(any(r.get("unchanged") is True for r in event("artifact_revalidation")),
          "retained shard revalidation missing/failed")

    begins = {r["label"]: r for r in event("arm_begin")}
    ends = {r["label"]: r for r in event("arm_complete")}
    endpoints = {r["label"]: r for r in event("endpoint")}
    check(len(begins) == events["arm_begin"] and len(ends) == events["arm_complete"],
          "duplicate arm label")
    check(begins.keys() == ends.keys(), "incomplete arm")
    check(len(endpoints) == events["endpoint"], "duplicate endpoint label")
    for label, row in ends.items():
        begin = begins[label]
        if candidate_mode is not None:
            expected_override = (None if candidate_mode == "production" else True) if begin["candidate_schedule"] else False
            check(begin.get("candidate_mode") == candidate_mode
                  and "schedule_override" in begin
                  and begin["schedule_override"] is expected_override,
                  f"arm policy witness mismatch: {label}")
        ranges = begin["absolute_ranges"]
        expected = ([[2048, 4096]] if begin["candidate_schedule"] else
                    [[2048, 2051], [2051, 4096]])
        if begin["start"] == 0:
            expected = [[0, 2048]] + expected
        check(ranges == expected, f"unexpected ranges: {label}")
        check(row["gpu_valid"] and row["gpu_samples"] == row["command_count"] == len(ranges)
              and row["packed_tokens"] == 4096 - begin["start"] and row["contains_selection"],
              f"invalid GPU coverage/publication: {label}")
        check(all(isinstance(row.get(k), (float, int)) and math.isfinite(row[k]) and row[k] > 0
                  for k in ("complete_gpu_ms", "ordinary_call_wall_ms")),
              f"invalid timing: {label}")
        check(row["timing_eligible"] == begin["timing_eligible"] == (not begin["census"]),
              f"observer/timing mismatch: {label}")
        steps = range(5) if row["timing_eligible"] else range(1)
        for step in steps:
            key = label + (f"/continuation{step}" if step else "")
            check(key in endpoints, f"missing endpoint: {key}")
            if key in endpoints:
                check(endpoints[key]["state"]["position"] == 4096 + step,
                      f"endpoint position mismatch: {key}")
    check(all(r["nonfinite_logits"] == 0 and r["logit_count"] > 0
              for r in endpoints.values()), "nonfinite/empty endpoint logits")

    measured = [r for r in ends.values() if r["timing_eligible"]]
    cells = defaultdict(list)
    for row in measured:
        cell, round_name, arm_name = row["label"].rsplit("/", 2)
        cells[cell].append((int(round_name.removeprefix("round")), arm_name, row))
    corpora = ["prose", "ssh_repeated"] if header["details"]["corpora"] == "both" else ["prose"]
    stage = header["details"]["stage"]
    stages = (["suffix_at2048"] if stage == "suffix" else ["whole4096"] if stage == "whole"
              else ["suffix_at2048", "whole4096"])
    check(set(cells) == {f"{corpus}/{part}" for corpus in corpora for part in stages},
          "missing/unexpected prompt or stage cell")
    reports = []
    abba_records = {(r["label"], r["round"]): r for r in event("abba")}
    check(len(abba_records) == events["abba"] and set(abba_records) ==
          {(cell, i) for cell in cells for i in range(1, expected_rounds + 1)},
          "missing/duplicate/unexpected ABBA records")
    for cell, attempts in sorted(cells.items()):
        expected = [(i, name) for i in range(1, expected_rounds + 1)
                    for name in ("A1", "B1", "B2", "A2")]
        check([(i, name) for i, name, _ in attempts] == expected,
              f"incomplete/out-of-order ABBA: {cell}")
        rounds, aggregates = [], {}
        for i in sorted({i for i, _, _ in attempts}):
            arms = {name: r for j, name, r in attempts if j == i}
            if set(arms) != {"A1", "B1", "B2", "A2"}:
                continue
            recorded_abba = abba_records.get((cell, i), {})
            check(recorded_abba.get("rounds", declared_rounds) == declared_rounds,
                  f"ABBA round-count record disagrees with header: {cell}/{i}")
            if candidate_mode is not None:
                check(recorded_abba.get("rounds") == declared_rounds
                      and recorded_abba.get("candidate_mode") == candidate_mode,
                      f"ABBA options record disagrees with header: {cell}/{i}")
            timing = {}
            for key in ("complete_gpu_ms", "ordinary_call_wall_ms"):
                v = {name: r[key] for name, r in arms.items()}
                recorded_key = "gpu_ms" if key == "complete_gpu_ms" else key
                recorded_timing = recorded_abba.get(recorded_key, {})
                check(all(name in recorded_timing and math.isclose(recorded_timing[name], value,
                           rel_tol=1e-12, abs_tol=1e-9) for name, value in v.items()),
                      f"ABBA timings disagree with ordinary arm records: {cell}/{i}/{key}")
                timing[key] = {
                    "arms_ms": v,
                    "pairs": [{"A": a, "B": b, "saved_ms": v[a] - v[b],
                               "saving_percent": 100 * (1 - v[b] / v[a])}
                              for a, b in (("A1", "B1"), ("A2", "B2"))],
                    "mean_saved_ms": (v["A1"] + v["A2"] - v["B1"] - v["B2"]) / 2,
                    "mean_saving_percent": 100 * (1 - (v["B1"] + v["B2"]) / (v["A1"] + v["A2"])),
                    "A2_vs_A1_drift_percent": 100 * (v["A2"] / v["A1"] - 1),
                    "B2_vs_B1_drift_percent": 100 * (v["B2"] / v["B1"] - 1),
                }
            rounds.append({"round": i, **timing})
        for key in ("complete_gpu_ms", "ordinary_call_wall_ms"):
            a = [r[key] for _, name, r in attempts if name.startswith("A")]
            b = [r[key] for _, name, r in attempts if name.startswith("B")]
            aggregates[key] = {"A_mean_ms": mean(a), "B_mean_ms": mean(b),
                               "mean_saved_ms": mean(a) - mean(b),
                               "mean_saving_percent": 100 * (1 - mean(b) / mean(a)),
                               "A_median_ms": median(a), "B_median_ms": median(b),
                               "median_saving_percent": 100 * (1 - median(b) / median(a))}
        reports.append({"cell": cell, "timed_attempts": len(attempts),
                        "aggregate": aggregates, "rounds": rounds})
    check(bool(reports), "no timed cells")
    check(events["continuation"] == 4 * len(measured), "missing/duplicate continuations")
    check(events["comparison"] == len(cells) * (1 + 20 * expected_rounds),
          "missing/duplicate comparison records")

    quality = defaultdict(list)
    persistent = []
    for row in event("comparison"):
        a, b, step = row["reference"], row["candidate"], row["step"]
        x, y = (endpoints[label + (f"/continuation{step}" if step else "")]
                for label in (a, b))
        same = begins[a]["candidate_schedule"] == begins[b]["candidate_schedule"]
        warm = not begins[a]["timing_eligible"] or not begins[b]["timing_eligible"]
        category = "warm_cross" if warm else ("same_schedule" if same else "cross_schedule")
        metrics = row["comparison"]["metrics"]
        values = {"relative_l2": metrics["relative_l2"], "max_abs": metrics["max_abs"],
                  **{key: row[key] for key in (
                      "kl_reference_candidate", "kl_candidate_reference",
                      "candidate_choice_regret_reference_logits",
                      "reference_choice_regret_candidate_logits")}}
        check(row["comparison"]["finite"] and all(math.isfinite(v) for v in values.values()),
              f"invalid numerical comparison: {a} {b} step{step}")
        metadata_equal = all(x["state"][k] == y["state"][k]
                             for k in ("position", "qsa_lengths", "ple_prior_tokens"))
        check(metadata_equal and row["causal_metadata_equal"], f"causal metadata differs: {a} {b}")
        check((x["state"] == y["state"]) == row["comparison"]["state_digests_equal"],
              f"state equality record disagrees: {a} {b}")
        tx, ty = x["state"]["tensors"], y["state"]["tensors"]
        check([(t["index"], t["dtype"], t["shape"], t["bytes"]) for t in tx] ==
              [(t["index"], t["dtype"], t["shape"], t["bytes"]) for t in ty],
              f"persistent tensor layouts differ: {a} {b}")
        changed = [u["index"] for u, v in zip(tx, ty) if u["sha256"] != v["sha256"]]
        persistent.append({"reference": a, "candidate": b, "step": step, "category": category,
                           "tensor_count": len(tx), "changed_indices": changed,
                           "hyper_hash_equal": x["state"]["hyper_sha256_f32_le"] == y["state"]["hyper_sha256_f32_le"]})
        cell = a.split("/round")[0] if not warm else a.rsplit("/", 1)[0]
        quality[(cell, category, step)].append({**values, "top1_equal": metrics["reference_top1"] == metrics["candidate_top1"],
                                         "top1_pair": [metrics["reference_top1"], metrics["candidate_top1"]],
                                         "logits_bits_equal": row["comparison"]["logits_bits_equal"],
                                         "state_digests_equal": row["comparison"]["state_digests_equal"]})
    numerical = []
    for (cell, category, step), group in sorted(quality.items()):
        numerical.append({"cell": cell, "category": category, "step": step, "comparisons": len(group),
                          "maxima": {k: max(r[k] for r in group) for k in values},
                          "top1_pairs": sorted({tuple(r["top1_pair"]) for r in group}),
                          **{k: all(r[k] for r in group) for k in
                             ("top1_equal", "logits_bits_equal", "state_digests_equal")}})
    witnesses = [{k: v for k, v in r.items() if k != "all_kernels"}
                 for r in event("warm_dispatch_witness")]
    check(len(witnesses) == 2 * len(cells) and all(r["valid"] for r in witnesses),
          "missing/invalid warm router witnesses")
    for row in event("warm_dispatch_witness"):
        begin = begins[row["label"]]
        if candidate_mode is not None:
            check("schedule_override" in row
                  and "schedule_override" in begin
                  and row["schedule_override"] is begin["schedule_override"],
                  f"census policy witness disagrees with arm: {row['label']}")
        check([r["absolute_range"] for r in row["commands"]] == begin["absolute_ranges"],
              f"census command ranges disagree with planner: {row['label']}")
        for command in row["commands"]:
            start, end = command["absolute_range"]
            strict_calls = 0 if end - start == 3 else 48
            check(command["rows"] == end - start
                  and command["strict_router_calls"] == command["expected_strict_router_calls"] == strict_calls
                  and command["generic_n3_router_shape_calls"] == (48 if end - start == 3 else 0),
                  f"router command witness mismatch: {row['label']}/{start}")
        strict = row["all_kernels"]["kernel_counts"].get("kernel_mat_mat_f32_f32_router_e8p32_strict", 0)
        expected = 96 if begins[row["label"]]["start"] == 0 else 48
        check(strict == row["strict_router_calls"] == expected,
              f"strict router aggregate census mismatch: {row['label']}")
    repetitions = defaultdict(list)
    for label, row in begins.items():
        cell = label.split("/round")[0] if row["timing_eligible"] else label.rsplit("/", 1)[0]
        for step in range(5) if row["timing_eligible"] else range(1):
            key = label + (f"/continuation{step}" if step else "")
            if key in endpoints:
                repetitions[(cell, row["candidate_schedule"], step)].append(endpoints[key])
    repetition_hashes = []
    for (cell, candidate, step), group in sorted(repetitions.items()):
        repetition_hashes.append({"cell": cell, "candidate_schedule": candidate, "step": step,
                                  "observations": len(group), "includes_warm": step == 0,
                                  "logits_hash_equal": len({r["logits_sha256_f32_le"] for r in group}) == 1,
                                  "state_equal": all(r["state"] == group[0]["state"] for r in group)})
    # Compare the same schedule across restored suffix and ordinary fresh whole
    # execution. Report equality as evidence, never make it a promotion gate.
    across_stages = defaultdict(list)
    for (cell, candidate, step), group in repetitions.items():
        corpus, stage_name = cell.split("/", 1)
        across_stages[(corpus, candidate, step)].append((stage_name, group))
    stage_hashes = []
    for (corpus, candidate, step), stage_groups in sorted(across_stages.items()):
        if len(stage_groups) < 2:
            continue
        group = [row for _, members in stage_groups for row in members]
        stage_hashes.append({"corpus": corpus, "candidate_schedule": candidate, "step": step,
                             "stages": sorted(stage for stage, _ in stage_groups),
                             "observations": len(group),
                             "logits_hash_equal": len({r["logits_sha256_f32_le"] for r in group}) == 1,
                             "state_equal": all(r["state"] == group[0]["state"] for r in group)})

    return {"path": str(path), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
            "events": events, "problems": problems, "header": header,
            "rounds_validated_from_header": declared_rounds,
            "candidate_mode_witness": candidate_mode or "legacy_not_explicitly_recorded",
            "executable_binding": event("executable_binding"), "artifact": event("artifact"),
            "loaded": event("loaded"), "diagnostic_memory": event("diagnostic_memory_bound"),
            "timed_attempts": len(measured), "warm_attempts_excluded": len(ends) - len(measured),
            "finite_endpoints": sum(r["nonfinite_logits"] == 0 for r in endpoints.values()),
            "endpoint_count": len(endpoints), "timing": reports, "router_witnesses": witnesses,
            "numerical_metric_origin": "Recorded f64 harness comparisons of full logits; JSONL retains hashes, not logit vectors.",
            "numerics": numerical, "persistent_hash_comparisons": persistent,
            "same_schedule_repetition_hashes": repetition_hashes,
            "same_schedule_cross_stage_hashes": stage_hashes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packets", nargs="+", type=Path,
                        help="explicit completed packets only; never auto-discover running JSONL files")
    args = parser.parse_args()
    reports = [summarize(path) for path in args.packets]
    print(json.dumps(reports, indent=2, allow_nan=False))
    raise SystemExit(1 if any(r["problems"] for r in reports) else 0)


if __name__ == "__main__":
    main()
