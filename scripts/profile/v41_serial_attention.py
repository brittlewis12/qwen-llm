#!/usr/bin/env python3
"""Attribute v41_prefill_census dispatch captures to serial attention.

All rates use isolated dispatch service, never native wall. Traffic is logical
payload: keys once per query (not once per head), and the selector's five
score/mask passes. Caches and redundant SIMD loads prevent treating this as
measured DRAM traffic. FLOPs count useful matrix dots and ordinary arithmetic;
integer selection/comparisons and special functions have no tensor FLOP rate.
"""
import argparse
import collections
import json
from pathlib import Path
import statistics

LABELS = {"attention_serial", "attn_owner", "attn_index"}
TENSOR_TF = 131.7
PAYLOAD_GBS = 800.0


def resources(record):
    """Return (logical bytes, ordinary FLOPs, accounting caveat)."""
    row = record["row"]
    k = row["kernel"].removeprefix("kernel_")
    a = dict(record["constants"])
    if k in {
        "deepseek_v41_attention_index_heads_rows_mma",
        "deepseek_v41_attention_index_heads_rows_mma_masked",
    }:
        start, ratio, _, _ = a[0]
        dim, heads, _, _ = a[1]
        _, n, block, _ = a[2]
        first = a[7][0]
        visible = [(start + first + i + 1) // ratio for i in range(n)]
        scored = sum(visible)
        masked = k.endswith("_masked")
        if masked:
            # The publisher keeps 2048 blocks, including the forced newest
            # block; only that last block can be partial.
            scored = sum(2047 * block + ((v - 1) % block + 1) for v in visible)
        payload = 4 * (n * heads * (dim + 1) + scored * dim + sum(visible))
        if masked:
            payload += 4 * sum((v + block - 1) // block for v in visible)
        return payload, 2 * dim * heads * scored, "dot FLOPs only; masked cardinality from publisher contract"
    if k.endswith("select_radix_rows"):
        start, ratio, topk, block = a[0]
        _, _, n, mask = a[1]
        _, first, candidate_k, publish = a[2]
        visible = [(start + first + i + 1) // ratio for i in range(n)]
        lengths = [(v + block - 1) // block if publish else v for v in visible]
        masked = bool(mask and not publish)
        kept = sum(candidate_k * block - ((-v) % block) for v in visible) if masked else sum(lengths)
        payload = 5 * 4 * kept
        if masked:
            payload += 5 * 4 * sum(lengths)
        payload += 4 * (sum(lengths) if publish else n * topk)
        return payload, None, "five score/mask passes; histogram atomics and on-chip traffic excluded"
    if k.endswith("select_block_scores_rows"):
        start, ratio, block, _ = a[0]
        _, _, n, first = a[1]
        visible = [(start + first + i + 1) // ratio for i in range(n)]
        return 4 * sum(v + (v + block - 1) // block for v in visible), None, "max comparisons; no matrix FLOPs"
    if k.endswith("selection_prepare_rows"):
        start, ratio, topk, block = a[0]
        _, _, n, publish = a[1]
        first = a[4][0]
        visible = [(start + first + i + 1) // ratio for i in range(n)]
        payload = 4 * sum((v if v <= topk else 0) + (publish * ((v + block - 1) // block)) for v in visible)
        return payload, None, "pruned non-publisher writes nothing; otherwise ID/mask fill"
    if k.endswith("attention_copy"):
        return 8 * a[0][0], 0, "F32 read plus write"
    if k.endswith("attention_pool"):
        return 20 * a[0][0], 8 * a[0][0], "also two exp per element, excluded from ordinary FLOPs"
    if k == "rms_norm_mul_f32":
        return 12 * a[0][0], 4 * a[0][0], "also one rsqrt; reduction overhead excluded"
    if k == "mat_vec_bf16_f32_tg":
        ni, no = a[0]
        return 2 * ni * no + 4 * (ni + no), 2 * ni * no, "logical vector once; issued vector loads repeat per output"
    if k.endswith("compress_publish"):
        kd, ld, pairs, _ = a[0]
        return 4 * (3 * kd + 2 * ld + pairs), 4 * kd + 14 * pairs, "also rsqrt and sin/cos; ordinary arithmetic estimate"
    if k.endswith("compress_rows"):
        ratio, _, _, pairs, _ = a[0]
        groups = row["grid"][0]
        payload = groups * (4 * ratio * 512 * (2 if ratio == 2 else 1) + 2 * 512 * 128 + 4 * (512 + 128 + pairs + 640))
        flops = groups * (2 * 512 * 128 + 4 * (512 + 128) + 14 * pairs + (8 * 512 if ratio == 2 else 0))
        return payload, flops, "weights/norms repeat per group logically; exp/rsqrt/sin/cos excluded"
    raise ValueError(f"unaccounted serial kernel: {k}")


def attribute(path):
    captures = [json.loads(line) for line in path.read_text().splitlines()]
    load = next(r for r in captures if r["kind"] == "load")
    grouped = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in captures:
        if r["kind"] == "dispatch" and r["rep"] > 0 and r["row"]["label"] in LABELS:
            grouped[r["row"]["kernel"]][r["rep"]].append(r)
    kernels = []
    for name, reps in grouped.items():
        times, sizes, ops, counts = [], [], [], []
        for rows in reps.values():
            metrics = [resources(r) for r in rows]
            times.append(sum(r["row"]["gpu_ns"] for r in rows) / 1e6)
            sizes.append(sum(x[0] for x in metrics))
            ops.append(sum(x[1] or 0 for x in metrics) if all(x[1] is not None for x in metrics) else None)
            counts.append(len(rows))
        ms = statistics.mean(times)
        size = statistics.mean(sizes)
        flops = statistics.mean(ops) if all(x is not None for x in ops) else None
        kernels.append(dict(kernel=name, calls=statistics.mean(counts), ms=ms, per_rep_ms=times,
                            logical_bytes=size, flops=flops,
                            logical_GBs=size / (ms * 1e6),
                            logical_bandwidth_roof_fraction=size / (ms * 1e6) / PAYLOAD_GBS,
                            useful_TFLOPs=flops / (ms * 1e9) if flops is not None else None,
                            tensor_peak_fraction=flops / (ms * 1e9) / TENSOR_TF if flops is not None else None,
                            caveat=metrics[0][2]))
    kernels.sort(key=lambda x: -x["ms"])
    cells = [r for r in captures if r["kind"] == "cell"]
    assert len(cells) == 3
    assert len({r["logit_digest"] for r in cells}) == 1
    return dict(file=path.name, depth=load["depth"], rows=load["rows"],
                warmup_plus_two_reps=True, logits_bitwise=True,
                isolated_serial_service_ms=sum(k["ms"] for k in kernels), kernels=kernels)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("captures", nargs="+", type=Path)
    a = p.parse_args()
    print(json.dumps([attribute(path) for path in a.captures], indent=2))
