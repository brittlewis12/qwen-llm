# V4.1 Query-Batched Attention

Measured baseline `81d6fdf1` and candidate `c959e511`, REAP-272 MXFP4/Q8 Engram,
80-core M5 Ultra, macOS 27.0.1. The baseline adds qualification tests to
`d302d083`; it changes no production arithmetic. The implementation is
`3b04af60`, rebased unchanged as `34d1b72c` onto serving maintenance `54824458`.
The candidate remains off main: sustained 10K repetitions expose unexplained
slowdown that also affects the following decode control.

## Matched Throughput

Five fresh-process pairs alternate arm order, baseline first in odd pairs.
Each process runs the shared suite with pp1024/2048, seed 1, normal model/row
warmups and one measured repetition per shape. Weights are resident, KV is
fresh, capacity is 2048 and prefill chunks are 256. Metal API/shader validation
are unset. Every row has matching clean source/build identity, no QWEN
overrides and no recorded thermal/performance warning. Values are tokens/s.

| Pair | Baseline pp1024 | Candidate pp1024 | Baseline pp2048 | Candidate pp2048 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 119.337 | 223.532 | 115.284 | 206.912 |
| 2 | 119.399 | 223.687 | 115.305 | 207.055 |
| 3 | 119.508 | 223.557 | 115.240 | 207.077 |
| 4 | 119.430 | 223.637 | 115.252 | 207.081 |
| 5 | 119.384 | 223.551 | 115.314 | 206.918 |

Median paired throughput gains are **87.25%** at 1024 and **79.57%** at 2048;
all pairs lie within 87.07-87.34% and 79.44-79.69%, respectively. Separate arm
medians are 119.399/223.557 and 115.284/207.055. These are warm prefill results,
not cold first-turn latency, decode acceleration or task quality.

### Sustained 10K Check

A separate combined pp10000/tg32 suite uses capacity 10000, three measured
repetitions after normal warmups, otherwise the same settings. Candidate runs
first, then baseline; these are not interleaved pairs. TG starts at depth zero
with fresh state, not populated 10K context. Values are tokens/s.

| Arm / shape | Samples | Mean +/- SD |
| --- | --- | ---: |
| Baseline pp10000 | 108.799, 108.730, 108.706 | 108.745 +/- 0.048 |
| Candidate pp10000 | 183.829, 171.687, 165.659 | 173.725 +/- 9.255 |
| Baseline tg32 | 26.699, 26.688, 26.665 | 26.684 +/- 0.018 |
| Candidate tg32 | 26.489, 25.027, 23.568 | 25.028 +/- 1.461 |

The candidate's 10K mean is 59.75% higher, but both candidate rows degrade
monotonically across repetitions. The decode mean is 6.21% lower. Allocation
growth, scheduling and thermal causes are unproven; absent OS warning flags do
not exclude throttling. These samples do not establish sustained steady-state
throughput or decode non-regression. No other Qwen workload was observed.

## Executed Change

Tiles of at most 32 queries batch RoPE, scoring, softmax, value packing and
reduction. Prepared tensor views replace per-row staging copies; the existing
HC projected-RMS reduction also batches rows. Old-window plus chunk scratch
preserves layer-local raw history before publishing the final ring slots.
Compressor, index selection and candidate transitions remain sequential.

Precision stays NativeF32. Scalar and tiled entry points share the numerical
operator bodies. Tiles split at the existing per-query PV threshold. The
64-head grouped value reduction still uses the shared scalar F32 kernel, not
MMA; GLM's shared dispatch is unchanged. Real-geometry 256-row scratch adds
54.76 MiB, included in admission accounting.

The real ring witness captures calls 127 then 257, internally 127 then 256+1.
Actual census requires 480 tiled score and 480 tiled softmax launches, followed
by 40 scalar launches each in the singleton tail's separate encoder. All
pipeline/grid/threadgroup combinations are retained; complete dispatch totals
are 7,529 and 15,948 per call. Counts establish execution, not GPU-time shares.

## Numerical Evidence

Baseline and candidate run with Metal API validation and required Metal tests.
Independent GGUF attention replay rebuilds history from captured normalized
inputs, rather than native caches. This is conditional attention qualification,
not independent long full-model/logit acceptance.

| Witness | Attention rows | Candidate comparisons | Pruned rows | Worst output error |
| --- | ---: | ---: | ---: | ---: |
| Ring/pair suffix, 127+257 | 72 | 45 | 0 | 3.52859e-5 |
| Odd suffix, 16383+3 | 56 | 35 | 10 | 2.93255e-5 |

All selected IDs, ownership and candidate sets match exactly, with no reported
boundary ties. Output bounds remain atol 1e-4, rtol 0. Baseline and candidate
maximum errors coincide; this does not establish cross-revision bitwise equality.
The wrapped 384-token snapshot is restored, both states grow to capacity 768,
and a three-token continuation gives bitwise-equal endpoint logits: native
self-parity only. Independent seven-token execution also passes. Internal
1,025-token serial/batched parity has maximum logit error 1.95503e-5,
KL 2.16867e-12, TV 7.17955e-7 and matching top-1.

At measured `c959e511`, all 140 model-free V4.1 tests pass on M4 and M5;
`check-contracts.sh --full` and the release workspace pass, with 3,098 passed,
zero failed and 648 opt-in/fixture-dependent skips in the full workspace.
Four nested filtered subprocess summaries are excluded from that total.
Astra reviews cover source, boundary witnesses, census and measured claims.

Evidence root: `~/qwen-runs/2026-10-19-v41-f32-mma/`. Raw pairs are
`query-c959e511-pairs/pair-{1..5}-{baseline,candidate}.{json,log}`;
the sustained check is `query-c959e511-long/{baseline,candidate}.{json,log}`;
qualification logs are `suffix-81d6fdf1-*` and `query-c959e511-*`. Logs identify
retained captures, independent reports, continuation self-parity and actual
dispatch census. The separate Q8 half-prefill experiment remains off main.
