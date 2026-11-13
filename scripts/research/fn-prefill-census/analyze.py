#!/usr/bin/env python3
"""Flash-Next prefill census: stage times, achieved TFLOP/s and GB/s.

Reads the JSONL emitted by the scratch `fn_prefill_census` probe. The in-tree
profiler samples layer 5 (GDN) and layer 7 (QSA) in detail; totals scale those
detail stages by the number of layers of each mixer. MoE/HC stages appear in
both detailed layers and are averaged before scaling to all layers.
"""

import argparse
import collections
import json
import math
import statistics as st
from pathlib import Path

PEAK = 131.676e12
BW = 1175.47e9

GROUPS = {
    "moe.router": "routed",
    "moe.topk": "routed",
    "moe.bucket": "routed",
    "moe.routed_gate_up": "routed",
    "moe.routed_down": "routed",
    "moe.routed_reduce": "routed",
    "moe.shared_gate_up": "shared",
    "moe.shared_down": "shared",
    "moe.shared_merge": "shared",
    "gdn.projections": "gdn_proj",
    "gdn.output": "gdn_proj",
    "gdn.decay": "gdn_other",
    "gdn.prep": "gdn_other",
    "gdn.norm": "gdn_other",
    "gdn.recurrence": "gdn_recur",
    "gdn.gate_norm": "gdn_other",
    "qsa.index_projection": "qsa_index",
    "qsa.index_state": "qsa_index",
    "qsa.visible_blocks": "qsa_index",
    "qsa.qkv_projections": "qsa_qkv",
    "qsa.norm_rope": "qsa_attn",
    "qsa.cache_scatter": "qsa_attn",
    "qsa.attention": "qsa_attn",
    "qsa.gate": "qsa_attn",
    "qsa.output": "qsa_qkv",
    "block.attention_hc": "hc",
    "block.ffn_hc": "hc",
    "block.attention_combine": "hc",
    "block.ffn_combine": "hc",
    "block.mixer_bridge": "hc",
    "block.moe_bridge": "hc",
}
ORDER = [
    "routed",
    "shared",
    "gdn_proj",
    "gdn_recur",
    "gdn_other",
    "qsa_qkv",
    "qsa_attn",
    "qsa_index",
    "hc",
    "other",
]


def leaves(stages):
    out = []
    for s in stages:
        if s["scope"] != "detail":
            continue
        nested = any(
            t is not s
            and t["depth"] > s["depth"]
            and t["start"] >= s["start"]
            and t["end"] <= s["end"]
            for t in stages
        )
        if not nested:
            out.append(s)
    return out


def tensor_map(inventory):
    m = {}
    for t in inventory:
        m[t["name"]] = t
    return m


def sum_bytes(tensors, layer, suffixes):
    total = 0
    for suffix in suffixes:
        t = tensors.get(f"blk.{layer}.{suffix}")
        if t:
            total += t["bytes"]
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    recs = [json.loads(x) for x in a.input.read_text().splitlines() if x.strip()]
    ident = next(x for x in recs if x["kind"] == "identity")
    cfg = ident["config"]
    tensors = tensor_map(ident["inventory"])
    layers = cfg["layers"]
    qsa_layers = math.ceil(layers / cfg["full_attention_interval"])
    gdn_layers = layers - qsa_layers
    H = cfg["hidden"]
    F = cfg["expert_intermediate"]
    S = cfg["shared_intermediate"]
    topk = cfg["topk"]
    gdn_n = (
        2 * cfg["gdn_key_heads"] * cfg["gdn_key_dim"]
        + cfg["gdn_value_heads"] * cfg["gdn_value_dim"]
    )
    gdn_v = cfg["gdn_value_heads"] * cfg["gdn_value_dim"]
    qkv_gdn = (
        2 * cfg["gdn_key_heads"] * cfg["gdn_key_dim"]
        + cfg["gdn_value_heads"] * cfg["gdn_value_dim"]
    )
    ph = cfg["hyper_hidden"] if "hyper_hidden" in cfg else 0
    # GDN layer example: use layer 5; QSA: layer 7.
    g_layer = next(
        (i for i in range(layers) if (i + 1) % cfg["full_attention_interval"] != 0), 5
    )
    q_layer = next(
        (i for i in range(layers) if (i + 1) % cfg["full_attention_interval"] == 0), 7
    )

    def layer_bytes(layer, suffixes):
        return sum_bytes(tensors, layer, suffixes)

    # Per-layer weight bytes (all layers identical for a mixer family).
    gdn_b = layer_bytes(
        g_layer,
        ["attn_qkv.weight", "attn_gate.weight", "ssm_beta.weight", "ssm_alpha.weight"],
    )
    gdn_out_b = layer_bytes(g_layer, ["ssm_out.weight"])
    qsa_b = layer_bytes(q_layer, ["attn_q.weight", "attn_k.weight", "attn_v.weight"])
    qsa_out_b = layer_bytes(q_layer, ["attn_output.weight"])
    qsa_idx_b = layer_bytes(q_layer, ["indexer.q_proj.weight", "indexer.k_proj.weight"])
    shared_b = layer_bytes(
        g_layer,
        ["ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"],
    )
    hc_b = layer_bytes(
        g_layer,
        [
            "hc_attn_down.weight",
            "hc_attn_up.weight",
            "hc_attn_inject.weight",
            "hc_ffn_down.weight",
            "hc_ffn_up.weight",
            "hc_ffn_inject.weight",
        ],
    )

    results = []
    for rec in recs:
        if rec["kind"] != "profile":
            continue
        rows = rec["rows"]
        depth = rec["depth"]
        lv = leaves(rec["stages"])
        by = collections.defaultdict(lambda: [0.0, 0])
        per_layer = collections.defaultdict(float)
        for s in lv:
            g = GROUPS.get(s["name"], "other")
            by[g][0] += s["gpu_ms"]
            by[g][1] += 1
            per_layer[(s["layer"], s["name"])] += s["gpu_ms"]
        # Scale detail stages. GDN/QSA stages are seen only in their layer;
        # MoE/HC are seen in both and averaged.
        scaled = collections.defaultdict(float)
        seen = collections.defaultdict(set)
        for (layer, name), ms in per_layer.items():
            g = GROUPS.get(name, "other")
            seen[g].add(layer)
            if name.startswith("gdn."):
                scaled[g] += ms * gdn_layers
            elif name.startswith("qsa."):
                scaled[g] += ms * qsa_layers
            else:
                scaled[g] += (
                    ms
                    * layers
                    / max(
                        1,
                        len({l for (l, n) in per_layer if GROUPS.get(n, "other") == g}),
                    )
                )
        # Routes for expert bytes.
        route = next(
            (
                r
                for r in recs
                if r["kind"] == "routes"
                and r.get("cell") == rec.get("cell")
                and r.get("depth") == depth
                and r.get("rows") == rows
            ),
            None,
        )
        expert_b = 0.0
        if route:
            for row in route["layers"]:
                counts = row["counts"]
                bank = row["bank_bytes"]
                active = sum(1 for c in counts if c > 0)
                frac = active / max(1, len(counts))
                expert_b += sum(bank) * frac
        routed_flops = 6 * rows * topk * H * F
        shared_flops = 6 * rows * H * S
        gdn_proj_flops = (
            2 * rows * H * (qkv_gdn + gdn_v + 2 * cfg["gdn_value_heads"]) * gdn_layers
        )
        gdn_out_flops = 2 * rows * gdn_v * H * gdn_layers
        qsa_qkv_flops = (
            2
            * rows
            * H
            * (
                2 * cfg["qsa_heads"] * cfg["qsa_head_dim"]
                + 2 * cfg["qsa_kv_heads"] * cfg["qsa_head_dim"]
            )
            * qsa_layers
        )
        qsa_out_flops = (
            2 * rows * cfg["qsa_heads"] * cfg["qsa_head_dim"] * H * qsa_layers
        )
        # HC projections: two roles/layer, down+up+inject.
        hyper_hidden = tensors.get(f"blk.{g_layer}.hc_attn_down.weight", {}).get(
            "shape", [0, 0]
        )[0]
        hyper_rank = tensors.get(f"blk.{g_layer}.hc_attn_down.weight", {}).get(
            "shape", [0, 0]
        )[1]
        hyper_count = tensors.get(f"blk.{g_layer}.hc_attn_inject.weight", {}).get(
            "shape", [0, 0]
        )[1]
        hc_flops = (
            2
            * rows
            * (2 * hyper_hidden * hyper_rank + hyper_hidden * hyper_count)
            * 2
            * layers
        )
        rowsd = {
            "routed": (
                routed_flops,
                expert_b + rows * topk * H * 4 + rows * topk * F * 2 * 2,
            ),
            "shared": (shared_flops, shared_b * layers + rows * H * 4 * 2),
            "gdn_proj": (
                gdn_proj_flops + gdn_out_flops,
                (gdn_b + gdn_out_b) * gdn_layers + rows * H * 4 * 2,
            ),
            "gdn_recur": (0, 0),
            "gdn_other": (0, 0),
            "qsa_qkv": (
                qsa_qkv_flops + qsa_out_flops,
                (qsa_b + qsa_out_b) * qsa_layers + rows * H * 4 * 2,
            ),
            "qsa_attn": (0, 0),
            "qsa_index": (0, qsa_idx_b * qsa_layers),
            "hc": (hc_flops, hc_b * layers),
            "other": (0, 0),
        }
        stages = []
        for g in ORDER:
            ms = scaled.get(g, 0.0)
            flops, bts = rowsd[g]
            if g in ("gdn_proj",):
                flops = gdn_proj_flops
                bts = (gdn_b + gdn_out_b) * gdn_layers
            if g == "qsa_qkv":
                flops = qsa_qkv_flops + qsa_out_flops
                bts = (qsa_b + qsa_out_b) * qsa_layers
            roof_ms = max(flops / PEAK, bts / BW) * 1e3
            stages.append(
                {
                    "group": g,
                    "modeled_gpu_ms": ms,
                    "share": None,
                    "flops": flops,
                    "bytes": bts,
                    "tflops": flops / ms / 1e9 if ms else None,
                    "gb_s": bts / ms / 1e6 if ms else None,
                    "roof_ms": roof_ms,
                    "distance": ms / roof_ms if roof_ms else None,
                }
            )
        total = sum(s["modeled_gpu_ms"] for s in stages)
        for s in stages:
            s["share"] = s["modeled_gpu_ms"] / total if total else None
        token = rec["token"]
        results.append(
            {
                "cell": rec.get("cell"),
                "depth": depth,
                "rows": rows,
                "wall_ms": token["total_wall_ms"],
                "gpu_ms": token["gpu_ms"],
                "host_encode_ms": rec["encode"],
                "command": rec["command"],
                "profile_sampling": rec["sampling"],
                "sampling_fallback": rec["sampling_fallback"],
                "detail_layers": [g_layer, q_layer],
                "mixer_layers": {"gdn": gdn_layers, "qsa": qsa_layers},
                "stages": stages,
                "leaf_sum_ms": total,
                "gpu_coverage": total / token["gpu_ms"] if token["gpu_ms"] else None,
            }
        )

    out = {
        "identity": {k: ident[k] for k in ("depths", "sizes", "root_rows", "config")},
        "profiles": results,
    }
    if a.out:
        a.out.write_text(json.dumps(out, indent=2) + "\n")
    for r in results:
        print(
            f"\n=== {r['cell']} depth={r['depth']} rows={r['rows']} "
            f"wall={r['wall_ms']:.1f}ms gpu={r['gpu_ms']:.1f}ms ==="
        )
        print(
            f"{'group':12} {'ms':>9} {'share':>6} {'TFLOP/s':>8} {'GB/s':>8} "
            f"{'roof ms':>8} {'dist':>6}"
        )
        for s in sorted(r["stages"], key=lambda x: -x["modeled_gpu_ms"]):
            if s["modeled_gpu_ms"] <= 0:
                continue
            tf = f"{s['tflops']:.1f}" if s["tflops"] else "-"
            gb = f"{s['gb_s']:.0f}" if s["gb_s"] else "-"
            dist = f"{s['distance']:.2f}" if s["distance"] else "-"
            print(
                f"{s['group']:12} {s['modeled_gpu_ms']:9.2f} {s['share'] * 100:5.1f}% "
                f"{tf:>8} {gb:>8} {s['roof_ms']:8.2f} {dist:>6}"
            )


if __name__ == "__main__":
    main()
