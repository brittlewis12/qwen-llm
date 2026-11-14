#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///
"""Rank wasted GPU work in captured causal node graphs.

Inputs are the ``nodes`` captures exported by ``v41_batch_depth``,
``glm_solo_bounds`` and ``fn_solo_bounds`` (see
``docs/bench/2026-11-11-two-queue-causal.md``).  Each capture contains one
plan per causal scope with exact read/write buffer spans and one isolated GPU
timestamp pair per dispatch.  This tool attributes that GPU time to

  * dead writes (written, then overwritten or dropped without a read),
  * recomputation (same kernel, arguments, grid and input spans within a round),
  * pure data movement (copies, fills and F32<->half conversions),
  * memory round-trip fusion candidates,
  * possible unneeded rows/head work, and
  * per-member weight duplication at B>1.

It is a diagnostic ranking, not a correctness proof: buffer names are not in
the capture, so persistent state is inferred from cross-round reads and real
outputs from kernel names.  Every item carries its dispatch evidence.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

MOVEMENT = re.compile(
    r"(?i)(^kernel_copy|copy_offset|blit|round_trip|(^|_)scatter|"
    r"(^|_)fill(_|$)|(^|_)zero(_|$)|(^|_)pack(_|$)|unpack|broadcast|"
    r"convert|f32_to_f16|f16_to_f32|_to_half|_half_to_)"
)
ELEMENTWISE = re.compile(
    r"(?i)(add|mul|axpy|swiglu|silu|gelu|relu|scale|clamp|norm|rms|"
    r"reduce|sum|weighted_sum|combine|inject|epilogue|sigmoid|softmax|rope|"
    r"copy_offset|scatter|convert)"
)
OUTPUT = re.compile(
    r"(?i)(logit|lm_head|_head$|head_|argmax|sample|status|route|count|"
    r"topk|token|valid|select_status)"
)
PERSISTENT = re.compile(r"(?i)(kv_|cache|state|ring|indexer|append|store|history|conv)")
HEAD = re.compile(r"(?i)(logit|lm_head|_head$|head_|argmax|sample)")


def overlap(lo, hi, lo2, hi2):
    return lo < hi2 and lo2 < hi


def covered_bytes(a_lo, a_hi, b_lo, b_hi):
    return max(0, min(a_hi, b_hi) - max(a_lo, b_lo))


def capture_records(path: Path):
    records = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.startswith("{")
    ]
    captures = [r for r in records if r.get("kind") == "causal_graph"]
    if not captures:
        raise SystemExit(f"{path}: no causal_graph record")
    return captures


class Capture:
    def __init__(self, path: Path, raw=None):
        self.path = path
        self.raw = capture_records(path)[-1] if raw is None else raw
        if self.raw.get("mode") != "nodes":
            raise SystemExit(f"{path}: capture is {self.raw.get('mode')!r}, need nodes")
        if not self.raw.get("bitwise", True):
            raise SystemExit(f"{path}: capture is not bitwise")
        self.width = int(self.raw.get("width", 1) or 1)
        self.rounds = int(self.raw.get("rounds", 1) or 1)
        self.phase = self.raw.get("phase") or (
            "decode" if self.raw.get("width") else "decode"
        )
        self.rows = self.raw.get("capture_rows")
        if self.rows is None and self.phase == "decode":
            self.rows = self.width
        self.plans = self.raw["plans"]
        self.node_times = {}
        for row in self.raw.get("rows", []):
            if row.get("node") is not None:
                self.node_times[(row["scope"], row["node"])] = int(
                    row.get("gpu_ns") or 0
                )
        if len(self.plans) % self.rounds:
            raise SystemExit(
                f"{path}: {len(self.plans)} scopes for {self.rounds} rounds"
            )
        self.scopes_per_round = len(self.plans) // self.rounds

    def plan_round(self, index):
        return index // self.scopes_per_round

    def events(self):
        """Yield (round, plan_index, node_index, node, writes, reads)."""
        for pi, plan in enumerate(self.plans):
            scope = plan["scope"]
            for ni, node in enumerate(plan["nodes"]):
                writes = [s for s in node["spans"] if s["write"]]
                reads = [s for s in node["spans"] if not s["write"]]
                yield (
                    self.plan_round(pi),
                    pi,
                    ni,
                    node,
                    writes,
                    reads,
                )

    def gpu_ns(self, pi, ni):
        plan = self.plans[pi]
        return self.node_times.get((plan["scope"], ni), 0)

    def share(self, pi, ni, start_byte, total_bytes):
        if total_bytes <= 0:
            return 0.0
        return self.gpu_ns(pi, ni) * (start_byte / total_bytes)

    def summary_wall(self):
        for key in (
            "reference_wall_ms_chunk",
            "reference_wall_ms_token",
            "reference_wall_ms_round",
            "wall_ms_round",
        ):
            value = self.raw.get(key)
            if isinstance(value, list) and value:
                return statistics.mean(value)
            if isinstance(value, (int, float)) and value:
                return float(value)
        return None

    def unit(self):
        if self.phase in ("prefill", "append"):
            return "chunk"
        return "round" if self.width > 1 else "token"

    def gpu_ms_total(self):
        return sum(self.gpu_ns(pi, ni) for _, pi, ni, _, _, _ in self.events()) / 1e6

    def per_unit_ms(self):
        return self.gpu_ms_total() / max(1, self.rounds)


def index_accesses(cap: Capture):
    """Per-buffer ordered access list across the captured rounds."""
    by_buffer = defaultdict(list)
    for rnd, pi, ni, node, writes, reads in cap.events():
        for kind, spans in (("w", writes), ("r", reads)):
            for s in spans:
                by_buffer[s["buffer"]].append(
                    (rnd, pi, ni, kind, s["lo"], s["hi"], node["kernel"])
                )
    for rows in by_buffer.values():
        rows.sort(key=lambda x: (x[0], x[1], x[2]))
    return by_buffer


def ordered_events(cap: Capture):
    return list(cap.events())


def writes_between(events, start, end, spans):
    """True if any node after `start` and before `end` overwrites an input."""
    for _rnd, pi, ni, _node, writes, _reads in events:
        pos = (pi, ni)
        if pos <= start or pos >= end:
            continue
        for w in writes:
            for s in spans:
                if s["buffer"] == w["buffer"] and overlap(
                    s["lo"], s["hi"], w["lo"], w["hi"]
                ):
                    return True
    return False


def dead_writes(cap: Capture):
    """Writes overwritten later *in the same round* without an intervening read.

    Cross-round accesses are deliberately ignored: scratch buffers recur at the
    same offsets every token, and outputs (logits, KV, recurrent state) are
    consumed on the host or in later rounds rather than by a later capture
    dispatch.  Only a same-round overwrite proves the prior write was useless.
    """
    items = []
    for rnd, pi, ni, node, writes, _reads in cap.events():
        kernel = node["kernel"]
        total = sum(s["hi"] - s["lo"] for s in writes)
        for s in writes:
            lo, hi = s["lo"], s["hi"]
            buf = s["buffer"]
            first_overlap = None
            for r2, p2, n2, node2, _w2, reads2 in cap.events():
                if r2 != rnd or (p2, n2) <= (pi, ni):
                    continue
                for r in reads2:
                    if r["buffer"] == buf and overlap(lo, hi, r["lo"], r["hi"]):
                        first_overlap = (p2, n2, "r", r, node2["kernel"])
                        break
                if first_overlap is not None:
                    break
                for w in _w2:
                    if w["buffer"] == buf and overlap(lo, hi, w["lo"], w["hi"]):
                        first_overlap = (p2, n2, "w", w, node2["kernel"])
                        break
                if first_overlap is not None:
                    break
            if first_overlap is None or first_overlap[2] != "w":
                continue
            _, n2, _, w, akernel = first_overlap
            dead = covered_bytes(lo, hi, w["lo"], w["hi"])
            if dead <= 0:
                continue
            items.append(
                {
                    "category": "dead_write",
                    "family_file": cap.path.name,
                    "phase": cap.phase,
                    "kernel": kernel,
                    "buffer": buf,
                    "lo": lo,
                    "hi": hi,
                    "dead_bytes": dead,
                    "span_bytes": hi - lo,
                    "gpu_ms": cap.share(pi, ni, dead, total) / 1e6,
                    "round": rnd,
                    "scope": cap.plans[pi]["scope"],
                    "node": ni,
                    "overwritten_by": akernel,
                    "overwrite_scope": cap.plans[p2]["scope"],
                    "overwrite_node": n2,
                    "evidence": "overwritten in the same round before any read",
                }
            )
    return items


def recomputation(cap: Capture):
    items = []
    events = ordered_events(cap)
    for rnd in range(cap.rounds):
        seen = {}
        for pi in range(rnd * cap.scopes_per_round, (rnd + 1) * cap.scopes_per_round):
            # Scratch buffers are reused across layers, so identical pointers in
            # different scopes are not the same data.  Only compare within one
            # causal scope (one layer/stage owner).
            seen = {}
            for ni, node in enumerate(cap.plans[pi]["nodes"]):
                reads = tuple(
                    sorted(
                        (s["buffer"], s["lo"], s["hi"])
                        for s in node["spans"]
                        if not s["write"]
                    )
                )
                key = (
                    node["kernel"],
                    tuple(node["grid"]),
                    tuple(node["threads"]),
                    reads,
                    tuple((slot, tuple(data)) for slot, data in node.get("constants", [])),
                    tuple(tuple(x) for x in node.get("threadgroup_memory", [])),
                    node.get("pipeline"),
                )
                if key in seen and reads:
                    first = seen[key]
                    if writes_between(
                        events,
                        (first[0], first[1]),
                        (pi, ni),
                        [{"buffer": b, "lo": lo, "hi": hi} for b, lo, hi in reads],
                    ):
                        seen[key] = (pi, ni)
                        continue
                    previous = cap.plans[first[0]]["nodes"][first[1]]
                    verified_args = all(
                        field in node and field in previous
                        for field in ("constants", "threadgroup_memory", "pipeline")
                    )
                    items.append(
                        {
                            "category": "recomputation" if verified_args else "unverified_recomputation",
                            "family_file": cap.path.name,
                            "phase": cap.phase,
                            "kernel": node["kernel"],
                            "gpu_ms": cap.gpu_ns(pi, ni) / 1e6,
                            "round": rnd,
                            "scope": cap.plans[pi]["scope"],
                            "node": ni,
                            "duplicate_of_scope": cap.plans[first[0]]["scope"],
                            "duplicate_of_node": first[1],
                            "input_bytes": sum(
                                s["hi"] - s["lo"]
                                for s in node["spans"]
                                if not s["write"]
                            ),
                            "evidence": (
                                "identical kernel/arguments/grid/threads/input spans"
                                if verified_args else
                                "legacy capture omits argument/pipeline identity; reuse is not proven"
                            ),
                        }
                    )
                else:
                    seen[key] = (pi, ni)
    return items


def movement(cap: Capture):
    items = []
    for rnd, pi, ni, node, writes, reads in cap.events():
        if not MOVEMENT.search(node["kernel"]):
            continue
        written = sum(s["hi"] - s["lo"] for s in writes)
        read = sum(s["hi"] - s["lo"] for s in reads)
        items.append(
            {
                "category": "data_movement",
                "family_file": cap.path.name,
                "phase": cap.phase,
                "kernel": node["kernel"],
                "gpu_ms": cap.gpu_ns(pi, ni) / 1e6,
                "bytes": written or read,
                "round": rnd,
                "scope": cap.plans[pi]["scope"],
                "node": ni,
                "grid": node["grid"],
                "threads": node["threads"],
                "evidence": "copy/convert/fill/scatter class kernel",
            }
        )
    return items


def _consumers(cap: Capture, rnd, pi, ni, writes):
    """Nodes that read a written range before the next write covers it."""
    events = ordered_events(cap)
    consumers = set()
    for w in writes:
        pos = (pi, ni)
        for r2, p2, n2, _node2, writes2, reads2 in events:
            if r2 != rnd or (p2, n2) <= pos:
                continue
            covered = False
            for s in reads2:
                if s["buffer"] == w["buffer"] and overlap(
                    w["lo"], w["hi"], s["lo"], s["hi"]
                ):
                    consumers.add((p2, n2))
            for x in writes2:
                if x["buffer"] == w["buffer"] and overlap(
                    w["lo"], w["hi"], x["lo"], x["hi"]
                ):
                    covered = True
                    break
            if covered:
                break
    return consumers


def fusable_chains(cap: Capture):
    items = []
    for rnd, pi, ni, node, writes, _reads in cap.events():
        if not writes:
            continue
        inter = sum(s["hi"] - s["lo"] for s in writes)
        if inter < 4096:
            continue
        consumers = _consumers(cap, rnd, pi, ni, writes)
        if len(consumers) != 1:
            continue
        p2, n2 = next(iter(consumers))
        consumer = cap.plans[p2]["nodes"][n2]
        if not (
            ELEMENTWISE.search(node["kernel"]) or ELEMENTWISE.search(consumer["kernel"])
        ):
            continue
        if OUTPUT.search(node["kernel"]) or PERSISTENT.search(node["kernel"]):
            continue
        items.append(
            {
                "category": "fusable_chain",
                "family_file": cap.path.name,
                "phase": cap.phase,
                "producer": node["kernel"],
                "consumer": consumer["kernel"],
                "gpu_ms": cap.gpu_ns(pi, ni) / 1e6,
                "intermediate_bytes": inter,
                "round": rnd,
                "scope": cap.plans[pi]["scope"],
                "node": ni,
                "consumer_scope": cap.plans[p2]["scope"],
                "consumer_node": n2,
                "evidence": "single consumer, elementwise/reduction round trip",
            }
        )
    return items


def per_member(cap: Capture):
    if cap.width <= 1:
        return []
    items = []
    for rnd in range(cap.rounds):
        for pi in range(rnd * cap.scopes_per_round, (rnd + 1) * cap.scopes_per_round):
            seen = defaultdict(list)
            for ni, node in enumerate(cap.plans[pi]["nodes"]):
                reads = [s for s in node["spans"] if not s["write"]]
                if not reads:
                    continue
                weight = max(reads, key=lambda s: s["hi"] - s["lo"])
                if weight["hi"] - weight["lo"] < 65536:
                    continue
                key = (node["kernel"], weight["buffer"], weight["lo"], weight["hi"])
                seen[key].append((ni, node, weight))
            for key, rows in seen.items():
                if len(rows) < 2:
                    continue
                outputs = {
                    tuple(
                        (s["buffer"], s["lo"], s["hi"])
                        for s in node["spans"]
                        if s["write"]
                    )
                    for _, node, _ in rows
                }
                if len(outputs) < 2:
                    continue
                nbytes = rows[0][2]["hi"] - rows[0][2]["lo"]
                dup = nbytes * (len(rows) - 1)
                extra_ms = sum(cap.gpu_ns(pi, ni) for ni, _, _ in rows[1:]) / 1e6
                items.append(
                    {
                        "category": "per_member_duplication",
                        "family_file": cap.path.name,
                        "phase": cap.phase,
                        "kernel": rows[0][1]["kernel"],
                        "gpu_ms": extra_ms,
                        "duplicated_bytes": dup,
                        "copies": len(rows),
                        "round": rnd,
                        "scope": cap.plans[pi]["scope"],
                        "node": rows[0][0],
                        "evidence": "same large weight span read by distinct outputs",
                    }
                )
    return items


def unneeded_rows(cap: Capture):
    items = []
    rows = cap.rows or (cap.width if cap.phase == "decode" else None)
    for rnd, pi, ni, node, writes, _reads in cap.events():
        if not HEAD.search(node["kernel"]) or not writes:
            continue
        out = sum(s["hi"] - s["lo"] for s in writes)
        items.append(
            {
                "category": "head_output",
                "family_file": cap.path.name,
                "phase": cap.phase,
                "kernel": node["kernel"],
                "gpu_ms": cap.gpu_ns(pi, ni) / 1e6,
                "written_bytes": out,
                "rows": rows,
                "round": rnd,
                "scope": cap.plans[pi]["scope"],
                "node": ni,
                "evidence": "full-vocabulary head dispatch; check row count",
            }
        )
    return items


def analyze(path: Path, raw=None):
    cap = Capture(path, raw)
    items = []
    items += dead_writes(cap)
    items += recomputation(cap)
    items += movement(cap)
    items += fusable_chains(cap)
    items += per_member(cap)
    items += unneeded_rows(cap)
    wall = cap.summary_wall()
    per_unit = cap.per_unit_ms()
    return {
        "file": str(path),
        "cell": cap.phase,
        "phase": cap.phase,
        "width": cap.width,
        "rounds": cap.rounds,
        "rows": cap.rows,
        "unit": cap.unit(),
        "gpu_ms_total": cap.gpu_ms_total(),
        "gpu_ms_per_unit": per_unit,
        "reference_wall_ms_unit": wall,
        "dispatches": sum(len(p["nodes"]) for p in cap.plans),
        "items": items,
    }


def ledger(results):
    units = defaultdict(
        lambda: {
            "b1_ms_token": 0.0,
            "b3_ms_round": 0.0,
            "prefill_ms_chunk": 0.0,
            "append_ms_chunk": 0.0,
        }
    )
    key_map = {
        ("decode", 1): "b1_ms_token",
        ("decode", 3): "b3_ms_round",
        ("prefill", 1): "prefill_ms_chunk",
        ("append", 1): "append_ms_chunk",
    }
    detailed = []
    for result in results:
        mult = result["gpu_ms_per_unit"]
        total = result["gpu_ms_total"]
        if total <= 0:
            continue
        for item in result["items"]:
            if item["gpu_ms"] <= 0:
                continue
            key = key_map.get((result["phase"], result["width"]))
            if key is None:
                key = "b3_ms_round" if result["width"] > 1 else "b1_ms_token"
            cat = item["category"]
            bucket = units[(cat, key)]
            bucket[key] += item["gpu_ms"]
            share = item["gpu_ms"] / total
            detailed.append(
                {
                    **item,
                    "per_unit_share": share,
                    "share_percent": 100 * share,
                    "capture_unit_ms": mult,
                    "capture_rounds": result["rounds"],
                    "cell_wall_ms": result["reference_wall_ms_unit"],
                    "capture_unit": result["unit"],
                }
            )
    ranked = sorted(detailed, key=lambda x: -x["gpu_ms"])
    return ranked


def self_test():
    def span(buf, lo, hi, write):
        return {"buffer": buf, "lo": lo, "hi": hi, "write": write}

    # Round 0: node0 writes scratch 0..1024 dead (overwritten by node1);
    # node2 and node3 recompute on the same read span; node4 is a copy.
    plan = {
        "scope": 1,
        "nodes": [
            {
                "id": 0,
                "kernel": "kernel_rms_norm_f32",
                "label": "",
                "grid": [1, 1, 1],
                "threads": [64, 1, 1],
                "predecessors": [],
                "spans": [
                    span(1, 0, 1024, False),
                    span(2, 0, 1024, True),
                ],
            },
            {
                "id": 1,
                "kernel": "kernel_add_f32",
                "label": "",
                "grid": [1, 1, 1],
                "threads": [64, 1, 1],
                "predecessors": [],
                "spans": [
                    span(1, 0, 1024, False),
                    span(2, 0, 1024, True),
                ],
            },
            {
                "id": 2,
                "kernel": "kernel_mul_f32",
                "label": "",
                "grid": [1, 1, 1],
                "threads": [64, 1, 1],
                "predecessors": [],
                "spans": [
                    span(1, 0, 1024, False),
                    span(3, 0, 1024, True),
                ],
            },
            {
                "id": 3,
                "kernel": "kernel_mul_f32",
                "label": "",
                "grid": [1, 1, 1],
                "threads": [64, 1, 1],
                "predecessors": [],
                "spans": [
                    span(1, 0, 1024, False),
                    span(4, 0, 1024, True),
                ],
            },
            {
                "id": 4,
                "kernel": "kernel_copy_offset_f32",
                "label": "",
                "grid": [1, 1, 1],
                "threads": [64, 1, 1],
                "predecessors": [],
                "spans": [
                    span(3, 0, 1024, False),
                    span(5, 0, 1024, True),
                ],
            },
            {
                "id": 5,
                "kernel": "kernel_lm_head_f32_splitk",
                "label": "",
                "grid": [1, 1, 1],
                "threads": [64, 1, 1],
                "predecessors": [],
                "spans": [
                    span(5, 0, 1024, False),
                    span(6, 0, 4096, True),
                ],
            },
        ],
        "waves": [[0, 2, 3, 4], [1], [5]],
    }
    for node in plan["nodes"]:
        node["constants"] = []
        node["threadgroup_memory"] = []
        node["pipeline"] = 1
    # Same spans and launch, different row0: not a duplicate.
    altered = dict(plan["nodes"][3], id=len(plan["nodes"]), constants=[(7,[1,0,0,0])])
    plan["nodes"].append(altered)
    nodes = []
    for scope_id, p in [(1, plan)]:
        for n in p["nodes"]:
            nodes.append(
                {
                    "scope": scope_id,
                    "node": n["id"],
                    "gpu_ns": 1_000_000,
                    "gpu_start": 0,
                    "gpu_end": 1_000_000,
                }
            )
    capture = {
        "kind": "causal_graph",
        "mode": "nodes",
        "bitwise": True,
        "width": 1,
        "rounds": 1,
        "phase": "decode",
        "plans": [plan],
        "rows": nodes,
    }
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps(capture) + "\n")
        name = f.name
    result = analyze(Path(name))
    Path(name).unlink()
    assert not any(i["category"] == "recomputation" and i.get("node") == altered["id"] for i in result["items"]), "different inline row0 is not recomputation"
    cats = {i["category"] for i in result["items"]}
    assert "dead_write" in cats, cats
    assert "recomputation" in cats, cats
    assert "data_movement" in cats, cats
    assert any(
        i["category"] == "dead_write" and i["dead_bytes"] == 1024
        for i in result["items"]
    )
    assert any(
        i["category"] == "recomputation" and i["kernel"] == "kernel_mul_f32"
        for i in result["items"]
    )
    for node in plan["nodes"]:
        node.pop("constants", None)
    legacy = recomputation(Capture(Path("legacy.jsonl"), capture))
    assert legacy and all(i["category"] == "unverified_recomputation" for i in legacy)
    assert all(i["gain_ms"] == 0 for i in _aggregate({"items": legacy}))
    print("waste_audit self-tests: passed")


FIX = {
    "dead_write": "drop the write, or stop overwriting before a reader",
    "recomputation": "verify output lifetime, then reuse the first result",
    "unverified_recomputation": "capture and compare inline constants before considering reuse",
    "data_movement": "have the producer write the consumer format / bind the source",
    "fusable_chain": "fuse the producer into its single consumer",
    "per_member_duplication": "share one weight stream across members",
    "head_output": "keep only the rows that can be sampled",
}


def _label(item):
    if item.get("producer"):
        return f"{item['producer']} -> {item['consumer']}"
    return item.get("kernel", item["category"])


FUSION_BW_GBPS = 800.0


def _aggregate(result):
    groups = {}
    for item in result["items"]:
        label = (
            f"{item['producer']} -> {item['consumer']}"
            if item.get("producer")
            else item.get("kernel", item["category"])
        )
        key = (item["category"], label)
        entry = groups.setdefault(
            key,
            {
                "category": item["category"],
                "label": label,
                "ms": 0.0,
                "count": 0,
                "bytes": 0,
                "scope": item.get("scope"),
                "node": item.get("node"),
            },
        )
        entry["ms"] += item["gpu_ms"]
        entry["count"] += 1
        entry["bytes"] += (
            item.get("intermediate_bytes")
            or item.get("dead_bytes")
            or item.get("duplicated_bytes")
            or item.get("bytes")
            or 0
        )
    for entry in groups.values():
        if entry["category"] == "fusable_chain":
            entry["gain_ms"] = min(
                entry["ms"], 2 * entry["bytes"] / (FUSION_BW_GBPS * 1e6)
            )
        elif entry["category"] == "unverified_recomputation":
            entry["gain_ms"] = 0.0
        else:
            entry["gain_ms"] = entry["ms"]
    return sorted(groups.values(), key=lambda x: -x["ms"])


def markdown_tables(results, top):
    lines = ["# Waste audit tables", ""]
    for result in results:
        wall = result["reference_wall_ms_unit"]
        lines.append(
            f"## `{Path(result['file']).name}`: {result['phase']}, "
            f"width {result['width']}, {result['rows']} rows"
        )
        lines.append("")
        lines.append(
            f"GPU service {result['gpu_ms_total']:.3f} ms "
            f"({result['gpu_ms_per_unit']:.3f} ms/{result['unit']}); "
            f"uninstrumented wall {wall if wall else 'n/a'} ms/{result['unit']}"
        )
        lines.append("")
        lines.append(
            "| category | node / kernel | ms/call | share of cell wall | "
            "bytes | proposed fix | expected gain |"
        )
        lines.append("|---|---|---:|---:|---:|---|---|")
        for entry in _aggregate(result)[:top]:
            per_call = entry["ms"] / max(1, result["rounds"])
            gain = entry["gain_ms"] / max(1, result["rounds"])
            share = 100 * per_call / wall if wall else float("nan")
            gain_share = 100 * gain / wall if wall else float("nan")
            label = f"{entry['label']} (x{entry['count']})"
            node = f"[{entry['scope']}:{entry['node']}] {label}"
            lines.append(
                f"| {entry['category']} | `{node}` | {per_call:.4f} | "
                f"{share:.2f}% | {entry['bytes']} | "
                f"{FIX.get(entry['category'], 'review')} | "
                f"{gain:.4f} ms/call ({gain_share:.2f}% wall) |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", nargs="*", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--json", action="store_true", help="emit full JSON")
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.captures:
        parser.error("no captures")
    results = [
        analyze(path, raw)
        for path in args.captures
        for raw in capture_records(path)
    ]
    ranked = ledger(results)
    out = {"captures": results, "ledger": ranked[: args.top]}
    if args.markdown:
        args.markdown.write_text(markdown_tables(results, args.top))
    if args.json:
        print(json.dumps(out, indent=2))
        return
    for result in results:
        print(
            f"\n{result['file']}: {result['phase']} width={result['width']} "
            f"rounds={result['rounds']} rows={result['rows']} "
            f"gpu={result['gpu_ms_total']:.3f} ms "
            f"({result['gpu_ms_per_unit']:.3f} ms/{result['unit']}) "
            f"wall={result['reference_wall_ms_unit']}"
        )
        by_cat = defaultdict(lambda: [0, 0.0])
        for item in result["items"]:
            by_cat[item["category"]][0] += 1
            by_cat[item["category"]][1] += item["gpu_ms"]
        for cat, (count, ms) in sorted(by_cat.items(), key=lambda x: -x[1][1]):
            print(f"  {cat:24s} n={count:5d}  {ms:9.4f} ms")
        for item in sorted(result["items"], key=lambda x: -x["gpu_ms"])[: args.top]:
            detail = item.get("kernel") or (
                f"{item.get('producer')} -> {item.get('consumer')}"
            )
            print(
                f"    {item['gpu_ms']:9.5f} ms  {item['category']:22s} "
                f"{detail}  [{item.get('scope')}:{item.get('node')}]"
            )
    print("\nCombined ranked ledger (ms per capture unit):")
    for item in ranked[: args.top]:
        print(json.dumps(item))


if __name__ == "__main__":
    main()
