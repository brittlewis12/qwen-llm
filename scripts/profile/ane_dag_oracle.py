#!/usr/bin/env python3
"""P0b DAG makespan oracle for the ANE prefill co-processor ladder.

Preregistered in docs/ANE-ORACLE.md rev 2. Replays a serialized
prefill phase trace (QWEN_PREFILL_TRACE_{LAYER,ATTN,FFN}_PHASES) in traced
execution order, moves chosen job classes to a serial ANE queue with
explicit producer/consumer edges, and reports projected whole-prefill
ceilings against the untraced baseline wall.

Timing model (current engine order; chunks sequential, layers sequential):
  - GPU: one serial resource executing every non-offloaded phase in traced
    order. ANE: one serial queue.
  - Offloaded job j: release = finish(producer(j)); ANE start =
    max(ane_free, release + in_stage + sync); consumer of j cannot start
    before ANE finish + out_stage + sync.
  - Optimistic row: sync = 0, staging fully overlapped (compute time only).
  - Pessimistic row: sync = 250 us/job, staging serial at 13.55 GB/s over
    f32 byte counts, input staging deduped per (chunk, layer, producer).
  - ANE service time = FLOPs / A_class prior (docs/ANE-ORACLE.md table),
    replaced by P1 measurements when available via --rates-json.

Projection onto product wall W (untraced median): distortion D =
traced_baseline_makespan / W_gpu; saved = base - makespan(S);
projected = W - saved / D; ceiling = W / projected. The proportional-
scaling assumption is declared in the prereg and printed per row.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

LAYER_RE = re.compile(
    r"\[prefill-layer-phase\] chunk=(?P<chunk>\d+) start=(?P<start>\d+) "
    r"layer=(?P<layer>\d+) kind=(?P<kind>\S+) phase=(?P<phase>\S+) "
    r"gpu_ms=(?P<gpu_ms>[0-9.]+)"
)
ATTN_RE = re.compile(
    r"\[prefill-attn-phase\] chunk=(?P<chunk>\d+) start=(?P<start>\d+) "
    r"layer=(?P<layer>\d+) phase=(?P<phase>\S+) gpu_ms=(?P<gpu_ms>[0-9.]+)"
)

STAGE_RATE = 13.55e9  # B/s, fused f32<->fp16 conversion prior (rustane)
SYNC_PESS = 250e-6  # s per ANE job, CPU rendezvous + encoder split proxy

# Class -> ANE TFLOP/s prior (docs/ANE-ORACLE.md rev 2 table).
ANE_RATES = {
    "OFF-EXP": 7.3e12,
    "OFF-RED": 3.2e12,
    "OFF-SHARED": 3.2e12,
    "OFF-SKINNY": 3.2e12,
    "OFF-ATTN-E": 7.3e12,
    "OFF-ATTN-R": 3.2e12,
    "DENSE-FFN-E": 7.3e12,
    "DENSE-FFN-R": 3.2e12,
}

# Per model: phase -> (n_in, n_out, class, producer_phase, consumer_phase).
# producer/consumer are phase names within the same (chunk, layer) group;
# consumer None means "join at next layer boundary" (end of this layer's
# record block). Shapes from the GGUF tensor tables (gguf -t).
MODELS = {
    "a3b": {
        "gdn_qkv": (2048, 8192, "OFF-EXP", "pre_norm", "gdn_prep_conv"),
        "gdn_z": (2048, 4096, "OFF-EXP", "pre_norm", "gdn_gated"),
        "gdn_back": (4096, 2048, "OFF-RED", "gdn_gated", "mixer_resid"),
        "gdn_beta_alpha": (2048, 64, "OFF-SKINNY", "pre_norm", "gdn_step"),
        "shared_packed": (2048, 2048, "OFF-SHARED", "post_norm", None),
        "proj": (2048, 9216, "OFF-ATTN-E", "norm", "rope_scatter"),
        "attn_back": (4096, 2048, "OFF-ATTN-R", "body_matrix_kqv", "mixer_resid"),
    },
    "d27b": {
        "gdn_qkv": (5120, 10240, "OFF-EXP", "pre_norm", "gdn_prep_conv"),
        "gdn_z": (5120, 6144, "OFF-EXP", "pre_norm", "gdn_gated"),
        "gdn_back": (6144, 5120, "OFF-RED", "gdn_gated", "mixer_resid"),
        "gdn_beta_alpha": (5120, 96, "OFF-SKINNY", "pre_norm", "gdn_step"),
        "proj": (5120, 14336, "OFF-ATTN-E", "norm", "rope_scatter"),
        "attn_back": (6144, 5120, "OFF-ATTN-R", "body_matrix_kqv", "mixer_resid"),
        "ffn_gate": (5120, 17408, "DENSE-FFN-E", "ffn_norm", "ffn_swiglu"),
        "ffn_up": (5120, 17408, "DENSE-FFN-E", "ffn_norm", "ffn_swiglu"),
        "ffn_down": (17408, 5120, "DENSE-FFN-R", "ffn_swiglu", "ffn_resid"),
    },
}
# shared_packed is three mats (gate, up: 2048->512; down: 512->2048); its
# FLOPs are overridden below rather than derived from the (n_in, n_out) pair.
SHARED_PACKED_FLOPS_PER_TOKEN = 2 * 3 * 2048 * 512


def parse_trace(paths: list[Path]):
    records = []
    pass_index = 0
    last_pos = None
    for path in paths:
        for line in path.open():
            m = LAYER_RE.search(line)
            if m:
                source, kind, phase = "layer", m.group("kind"), m.group("phase")
            else:
                m = ATTN_RE.search(line)
                if not m:
                    continue
                source, kind, phase = "attn-detail", "attn", m.group("phase")
            pos = (int(m.group("chunk")), int(m.group("start")), int(m.group("layer")))
            if last_pos is not None and pos < last_pos:
                pass_index += 1
            last_pos = pos
            records.append(
                dict(
                    p=pass_index,
                    chunk=pos[0],
                    start=pos[1],
                    layer=pos[2],
                    source=source,
                    kind=kind,
                    phase=phase,
                    ms=float(m.group("gpu_ms")),
                )
            )
    return records


def median_timeline(records, n_timed=3):
    """Median per ordered slot across the last n_timed passes."""
    n_pass = max(r["p"] for r in records) + 1
    timed = [p for p in range(max(0, n_pass - n_timed), n_pass)]
    per_pass = defaultdict(list)  # pass -> ordered records
    for r in records:
        if r["p"] in timed:
            per_pass[r["p"]].append(r)
    lengths = {p: len(v) for p, v in per_pass.items()}
    if len(set(lengths.values())) != 1:
        print(f"WARN: unequal pass lengths {lengths}; using min", file=sys.stderr)
    n = min(lengths.values())
    base = per_pass[timed[0]][:n]
    out = []
    for i, r in enumerate(base):
        vals = []
        for p in timed:
            rr = per_pass[p][i]
            key_match = (rr["chunk"], rr["layer"], rr["source"], rr["phase"]) == (
                r["chunk"],
                r["layer"],
                r["source"],
                r["phase"],
            )
            vals.append(rr["ms"] if key_match else r["ms"])
        out.append({**r, "ms": statistics.median(vals)})
    return out


def dedupe_attn(timeline):
    """Drop aggregate layer-source 'attn' rows where attn-detail exists."""
    detail_layers = {
        (r["chunk"], r["layer"]) for r in timeline if r["source"] == "attn-detail"
    }
    return [
        r
        for r in timeline
        if not (
            r["source"] == "layer"
            and r["phase"] == "attn"
            and (r["chunk"], r["layer"]) in detail_layers
        )
    ]


def chunk_tokens(timeline, n_prompt):
    starts = sorted({r["start"] for r in timeline})
    sizes = {}
    for i, s in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else n_prompt
        sizes[s] = end - s
    return sizes


def simulate(
    timeline, model, offload_classes, n_prompt, pessimistic, rates, layer_coverage="all"
):
    """Walk the traced order; return makespan (seconds)."""
    shapes = MODELS[model]
    sizes = chunk_tokens(timeline, n_prompt)
    gpu = 0.0
    ane_free = 0.0
    finish = {}  # (chunk, layer, phase) -> gpu-side availability time
    ane_out = {}  # (chunk, layer, phase) -> ANE result availability time
    staged_inputs = set()  # (chunk, layer, producer) staging dedupe
    sync = SYNC_PESS if pessimistic else 0.0

    def offloadable(r):
        if r["phase"] not in shapes:
            return False
        n_in, n_out, cls, prod, cons = shapes[r["phase"]]
        if cls not in offload_classes:
            return False
        if layer_coverage == "alt" and r["layer"] % 2 == 1:
            return False
        return True

    # Pre-index consumers: for each offloaded job, remember it under its
    # consumer slot so the walk can apply the wait when it reaches it.
    waiting = defaultdict(list)  # (chunk, layer, consumer_phase) -> jobs

    for r in timeline:
        key = (r["chunk"], r["layer"], r["phase"])
        # Apply any ANE-result waits that gate this phase.
        for job in waiting.pop(key, []):
            gpu = max(gpu, job)
        if offloadable(r):
            n_in, n_out, cls, prod, cons = shapes[r["phase"]]
            n = sizes[r["start"]]
            flops = (
                n * SHARED_PACKED_FLOPS_PER_TOKEN
                if r["phase"] == "shared_packed"
                else 2.0 * n * n_in * n_out
            )
            a = flops / rates[cls]
            in_b, out_b = 4.0 * n * n_in, 4.0 * n * n_out
            pkey = (r["chunk"], r["layer"], prod)
            in_st = 0.0
            if pessimistic:
                if pkey not in staged_inputs:
                    in_st = in_b / STAGE_RATE
                    staged_inputs.add(pkey)
                out_st = out_b / STAGE_RATE
            else:
                out_st = 0.0
            release = finish.get((r["chunk"], r["layer"], prod), gpu)
            start = max(ane_free, release + in_st + sync)
            done = start + a + out_st + sync
            ane_free = start + a + out_st  # queue occupancy
            if cons is None:
                # Join at layer boundary: gate the next layer's first phase.
                ane_out[(r["chunk"], r["layer"], "__layer_end__")] = done
                waiting[("__next_layer__", r["chunk"], r["layer"])].append(done)
            else:
                waiting[(r["chunk"], r["layer"], cons)].append(done)
            finish[key] = gpu  # phase itself consumed no GPU time
            continue
        # Layer-boundary joins from shared_packed-style jobs.
        prev = (r["chunk"], r["layer"] - 1)
        for job in waiting.pop(("__next_layer__", prev[0], prev[1]), []):
            gpu = max(gpu, job)
        gpu += r["ms"] / 1e3
        finish[key] = gpu

    # Drain any unconsumed ANE results (end of trace).
    tail = [t for jobs in waiting.values() for t in jobs]
    return max([gpu, ane_free] + tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--base-json", type=Path, required=True)
    ap.add_argument("--n-prompt", type=int, required=True)
    ap.add_argument("--rates-json", type=Path, help="P1 measured rates override")
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    rates = dict(ANE_RATES)
    if args.rates_json:
        rates.update(json.loads(args.rates_json.read_text()))

    records = parse_trace([args.trace])
    timeline = dedupe_attn(median_timeline(records))
    base = json.loads(args.base_json.read_text())
    row = base[0] if isinstance(base, list) else base
    w_wall = statistics.median([args.n_prompt / ts for ts in row["samples_ts"]])
    w_gpu = row["avg_gpu_ns"] / 1e9

    baseline = simulate(timeline, args.model, set(), args.n_prompt, True, rates)
    distortion = baseline / w_gpu

    classes = sorted({MODELS[args.model][p][2] for p in MODELS[args.model]})
    results = []
    for rsz in range(1, len(classes) + 1):
        for subset in itertools.combinations(classes, rsz):
            for cov in ("all", "alt"):
                mk_p = simulate(
                    timeline, args.model, set(subset), args.n_prompt, True, rates, cov
                )
                mk_o = simulate(
                    timeline, args.model, set(subset), args.n_prompt, False, rates, cov
                )
                proj_p = w_wall - (baseline - mk_p) / distortion
                proj_o = w_wall - (baseline - mk_o) / distortion
                results.append(
                    dict(
                        subset="+".join(subset),
                        coverage=cov,
                        ceiling_pess=w_wall / proj_p,
                        ceiling_opt=w_wall / proj_o,
                    )
                )
    results.sort(key=lambda r: r["ceiling_pess"], reverse=True)

    header = dict(
        model=args.model,
        n_prompt=args.n_prompt,
        w_wall_s=round(w_wall, 4),
        w_gpu_s=round(w_gpu, 4),
        traced_baseline_s=round(baseline, 4),
        distortion=round(distortion, 4),
        note="ceilings assume proportional scaling of serialized shares onto W",
    )
    if args.json:
        print(json.dumps(dict(**header, rows=results[: args.top]), indent=1))
        return
    print(json.dumps(header))
    print(f"{'subset':<58} {'cov':<4} {'pess':>7} {'opt':>7}")
    for r in results[: args.top]:
        print(
            f"{r['subset']:<58} {r['coverage']:<4} "
            f"{r['ceiling_pess']:>7.4f} {r['ceiling_opt']:>7.4f}"
        )


if __name__ == "__main__":
    main()
