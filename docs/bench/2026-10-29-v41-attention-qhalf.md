# V4.1 Prefill Q Half Mirror: +2.1%, Bitwise

2026-10-29. Source `da5d404e` on `perf/v41-attn-dataflow`, based on main
`c7c7cb12`. Artifact: GSQ-RCO IQ2_XS, 40 layers, 8K CED, M5 Ultra. Raw:
`~/qwen-runs/2026-10-29-v41-attn-qhalf/`.

## Decision

The tensor64 score kernel re-converted the full F32 query for every 32-key
panel. Staging the post-RoPE query once in the shared half scratch and
reading it from `score_tensor64_qhalf` is **exact** and improves the
same-binary CED A/B by **2.08% / 89 ms** (1,863.2 -> 1,901.9 tok/s; the pair's
CED arm is 1,827.6). The 8K pair endpoints are unchanged at
`a2db19afd7c377c54160e90bdefc52dc04ab13a69b0f929abb34eab099da6edf`. Default
on; `QWEN_V41_ATTN_QHALF=0` restores the F32 query. No new allocation: the
half scratch is already sized for the `heads*head_dim` output-projection half
and `finish_rows` overwrites it after the score read.

## What Changed

`encode_rows_prepare` stages the post-RoPE query once in the shared
`HalfScratch`; `sink_tile` selects `score_tensor64_qhalf` and binds the
half view for each query tile. The score kernel reads Q as half and K/V as
F32; the K/V mirrors are a separate premise.

## Why

A production-shaped probe (512-query tiles, K=640, window 128, topk 512,
three ABBA repeats, four launches, GPU timestamps) measured:

| variant | non-ragged | ragged |
| --- | ---: | ---: |
| K/V half | 1.022x | 1.022x |
| Q half | 1.113x | 1.115x |
| Q + K/V half | 1.166x | 1.170x |

All variants were bitwise-identical to the F32 tile on every finite lane.
The earlier half-KV negative used at most 32 queries per dispatch and ran
far below the GPU's threadgroup occupancy; at production occupancy the Q
reload is the larger cost. The query is about two thirds of the logical
Q+K/V loads (2.5 vs 1.25 MiB per query) and is re-converted for every
32-key panel.

## Evidence

8K pair (`--runs 4`, production timing, full-blocks-first):

| policy | samples (tok/s) | mean |
| --- | --- | ---: |
| Full | 1,146.91 / 1,145.86 / 1,145.85 / 1,145.83 | 1,146.1 |
| CED | 1,822.36 / 1,829.21 / 1,830.30 / 1,828.45 | **1,827.6** |

All 14 endpoints share the hash above.

Same-binary CED A/B, `O S S O`, two runs per arm, 180 s apart:

| arm | samples (tok/s) | mean |
| --- | --- | ---: |
| off | 1,857.59 / 1,855.94 | 1,856.8 |
| on | 1,897.92 / 1,904.04 | **1,901.0** |
| on | 1,896.29 / 1,909.19 | **1,902.7** |
| off | 1,868.91 / 1,870.32 | 1,869.6 |

Off mean 1,863.2 (wall 4.3968 s); on mean 1,901.9 (wall 4.3074 s);
+2.08% / 89.4 ms.

Private `current_pair` at `da5d404e`: PASS in 85.43 s, complete/current_pair
true, production self-lease; 257 positions, 257 full-vocabulary vectors per
arm, 56 causal regions, 256 teacher-forced steps. Report:
`~/qwen-runs/2026-10-29-v41-attn-qhalf/qualify-8k-da5d404e.json`.

## Limits

One artifact, one prompt, one session. The probe uses the fixture window 128
and topk 512; the production path is limited to the tensor tier and the
64-head/512-dim geometry. The half scratch reuse is only safe because the
score read is encoded before `finish_rows` overwrites the scratch. Burst
(cooldown) runs and continuous runs remain separate observations.

## Next

The K/V half mirrors (history and owner) add roughly 5% of the tile in the
combined probe but need new buffers and ledger pricing. The larger dataflow
premise is SIMD-owned attention with cooperative softmax/score-to-PV reuse.
