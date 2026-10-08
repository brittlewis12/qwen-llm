# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Summarize unscored prefill diagnostics; never infer promotion from exit status."""

import json
import statistics
import sys
from pathlib import Path


def mean(values):
    return statistics.mean(values)


def glm(events):
    result = {}
    for rows in (32, 128, 512):
        attempts = {
            e["label"]: e
            for e in events
            if e["event"] == "prefill_attempt" and e["rows"] == rows
        }
        if not all(k in attempts for k in ("A1", "B1", "B2", "A2")):
            continue
        a = mean(attempts[k]["wall_ms"] for k in ("A1", "A2"))
        b = mean(attempts[k]["wall_ms"] for k in ("B1", "B2"))
        leaf = {
            e["label"]: e["gpu_ms_per_dispatch"]
            for e in events
            if e["event"] == "leaf_attempt" and e["rows"] == rows
        }
        comparisons = [
            e["comparison"]
            for e in events
            if e["event"] == "prefill_comparison"
            and e["rows"] == rows
            and e["label"] in ("B1", "B2")
            and e["reference"] == "A1"
        ]
        routes = comparisons[0]["routes"]
        counts = [r["reference_occupancy"]["counts_by_expert"] for r in routes]
        panels = {
            n: sum((c + n - 1) // n for layer in counts for c in layer)
            for n in (16, 32)
        }
        item = {
            "incumbent_wall_ms": a,
            "candidate_wall_ms": b,
            "latency_saved_percent": 100 * (1 - b / a),
            "paired_saved_ms": [
                attempts["A1"]["wall_ms"] - attempts["B1"]["wall_ms"],
                attempts["A2"]["wall_ms"] - attempts["B2"]["wall_ms"],
            ],
            "incumbent_spread_percent": 100
            * abs(attempts["A1"]["wall_ms"] - attempts["A2"]["wall_ms"])
            / a,
            "max_endpoint_kl": max(
                c["endpoint_logits"]["distribution"]["kl_reference_candidate"]
                for c in comparisons
            ),
            "route_set_rows_changed": [
                sum(r["route_set_rows_changed"] for r in c["routes"])
                for c in comparisons
            ],
            "mean_active_experts": mean(
                r["reference_occupancy"]["active_experts"] for r in routes
            ),
            "useful_n16_lanes_fraction": rows * 8 * len(routes) / (16 * panels[16]),
            "useful_n32_lanes_fraction": rows * 8 * len(routes) / (32 * panels[32]),
            "m128n16_over_m64n32_panel_ratio": panels[16] / (2 * panels[32]),
        }
        if all(k in leaf for k in ("A1", "B1", "B2", "A2")):
            item["cache_hot_leaf_incumbent_ms"] = mean(leaf[k] for k in ("A1", "A2"))
            item["cache_hot_leaf_candidate_ms"] = mean(leaf[k] for k in ("B1", "B2"))
        gpu = [attempts[k].get("command_gpu_ms") for k in ("A1", "B1", "B2", "A2")]
        if all(v is not None for v in gpu):
            item["command_gpu_ms_abba"] = gpu
            item["gpu_latency_saved_percent"] = 100 * (
                1 - (gpu[1] + gpu[2]) / (gpu[0] + gpu[3])
            )
        result[rows] = item
    return result


def flash(events):
    interesting = {
        "loaded",
        "diagnostic_memory_gate",
        "run_complete",
        "observer",
        "observer_shape",
        "suffix_abba_summary",
        "factorial",
        "complete",
        "error",
        "model_error",
    }
    output = {"screen": [e for e in events if e["event"] in interesting]}
    output["normal_commands"] = [
        {"label": e["label"], "range": e["absolute_range"], **e["timing"]}
        for e in events
        if e["event"] == "command_complete" and e["label"].startswith("normal/")
    ]
    output["comparisons"] = [e for e in events if e["event"] == "comparison"]
    output["profiles"] = []
    for e in events:
        if e["event"] != "command_complete" or not e.get("profile"):
            continue
        profile = e["profile"]["profile"]
        groups = {}
        for s in profile.get("stages") or []:
            key = (
                s["depth"],
                s["scope"],
                s["label"],
                s["mixer"],
                s["layer"] if s["depth"] else None,
            )
            groups[key] = groups.get(key, 0) + s["gpu_ms"]
        peers = [
            r
            for r in events
            if r["event"] == "command_complete" and r["label"] == e["label"]
        ]
        index = peers.index(e)
        width = int(e["label"].split("/")[1])
        observer = next(
            (
                r
                for r in events
                if r["event"] == "observer"
                and r["width"] == width
                and r["command_index"] == index
            ),
            None,
        )
        output["profiles"].append(
            {
                "label": e["label"],
                "range": e["absolute_range"],
                "error": profile.get("error"),
                "raw_coverage": profile.get("raw_coverage_assuming_ns"),
                "raw_coverage_accepted": profile.get("raw_coverage_accepted"),
                "sampling": e["profile"].get("sampling"),
                "sampling_fallback": e["profile"].get("sampling_fallback"),
                "observer": observer,
                "usable_attribution": not profile.get("error")
                and bool(profile.get("raw_coverage_accepted"))
                and bool(observer and observer["accepted"]),
                "grouped_inclusive_spans_not_additive": [
                    {"depth_scope_label_mixer_layer": k, "ms": v}
                    for k, v in groups.items()
                ],
            }
        )
    return output


for arg in sys.argv[1:]:
    text = Path(arg).read_text()
    if text.lstrip().startswith("["):
        rows = json.loads(text)
        print(
            json.dumps(
                {
                    "file": arg,
                    "suite": [
                        {
                            "test": r["test"],
                            "depth": r["n_depth"],
                            "avg_ms": r["avg_ns"] / 1e6,
                            "samples_ms": [v / 1e6 for v in r["samples_ns"]],
                            "tokens_per_second": r["avg_ts"],
                            "prefill_chunk_reported": r["prefill_chunk"],
                        }
                        for r in rows
                    ],
                },
                indent=2,
            )
        )
        continue
    events = [json.loads(line) for line in text.splitlines() if line.strip()]
    schema = next(e["schema"] for e in events if e["event"] == "header")
    print(
        json.dumps(
            {
                "file": arg,
                "summary": glm(events) if schema.startswith("glm53") else flash(events),
            },
            indent=2,
        )
    )
