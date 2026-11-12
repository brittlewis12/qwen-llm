# Flash-Next sustained layer-slice gate: opt-in only

**Do not enable by default.** The worst observed peer gap drops **1754 → 670 ms
(61.8%)**, but throughput non-inferiority is unproven: mean paired difference
**−1.29%, 95% CI [−8.36%, +5.77%]**. The half-width is 7.06 percentage points,
well beyond the requested ±2%. No arm was discarded or repeated.

Grand-mean delivered rates are 131.017 off / 129.298 on tok/s. Those means
alone do not establish neutrality. Pooled ITL p50 is 19.564 / 19.542 ms;
p99 increases 25.645 / 130.332 ms. Pooled append TTFT p50 is 977 / 1009 ms,
p99 4555 / 4363 ms. Partial-prefill totals over each policy's 540 measured
seconds are 5.778 / 6.029 s (means 1.926 / 2.010 s per arm).

`QWEN_SERVE_FN_LAYER_SLICE_MS` stays **0**. Opt in with **200**, minimum predicted
append wall **500 ms**. MS=0 retains the incumbent native commands, row/frontier
plan and decode-credit policy. New diagnostic prefill accounting is common to
both arms. The [native price and exactness note](2026-11-12-fn-layer-slice-price.md)
describes the implementation; the [results JSON](2026-11-12-fn-layer-slice-serve.json)
contains full-precision metrics.

## Fixed protocol

Frozen source **3218067d3**, one release binary and tokenizer, nice 10, GSQ-RCO
IQ3_S. Three independent 49152-token roots, context 196608, production max-active
4, native feedback batch loop on. Release sampling defaults: temperature 1.0,
top-p 0.95; request temperature/top-p fields are absent. The existing harness
sets request seed 20261109, also used for workload draws. This is seeded
sampling, not greedy. That explicit seed is accepted by Flash-Next's native
sampler. V4.1's separate fast sampler is opt-in and was unset in its packet;
therefore the seed did not select a different sampler from the current default.

Order **A1 B1 B2 A2 A3 B3** (ABBAAB), paired A1/B1, A2/B2, A3/B3. Each arm has
15 s ramp and a complete 180 s measurement. Every arm checks the same binary
checksum and source revision before loading; no other model was resident.
Every build and load ran through the queued Metal lease, each hold ≤15 minutes,
with one serve arm per hold. The final guarded build and A1 shared a hold with
an explicit remaining-time budget; no earlier binary was measured.

Warm-up and measurement allow pressure 2, aborting at 3. Each arm requires a
continuous 60 s at ≤1 before loading and again after warming. These are new
opt-in harness controls; defaults are unchanged. Validation checked both
settling intervals, source, seed, sampling fields and full thermal exposure.

## Per-arm results

All time columns are milliseconds except partial-prefill seconds. ITL uses
nearest-rank content-delta intervals with both endpoints in-window. Append
TTFT includes offered-in-window requests that reached first output before the
cutoff; censored calls remain in the raw records.

| arm | MS | tok/s | ITL p50 / p99 / max | append TTFT p50 / p99 | partial_prefill_s | layer slices |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| A1 | 0 | 129.222 | 19.43 / 25.89 / 1754.22 | 999.09 / 4555.09 | 1.8980 | 0 |
| B1 | 200 | 128.928 | 19.72 / 130.85 / 364.20 | 1134.39 / 4362.68 | 2.1630 | 138 |
| B2 | 200 | 132.200 | 19.39 / 122.85 / 669.82 | 1398.24 / 4238.44 | 2.1002 | 138 |
| A2 | 0 | 131.067 | 19.77 / 25.70 / 1555.22 | 923.65 / 3553.61 | 1.9431 | 0 |
| A3 | 0 | 132.761 | 19.45 / 25.17 / 1569.03 | 969.93 / 3840.52 | 1.9372 | 0 |
| B3 | 200 | 126.767 | 19.56 / 132.82 / 400.76 | 885.73 / 4219.99 | 1.7659 | 162 |

Partial accounting includes active prefill paid by subsequently cancelled
turns and excludes peer service. The harness estimates overlap with the
measurement window from these active-work records; it does not treat peer
rounds as appender GPU work.

## Thermal exposure

Sample-and-hold seconds clipped to each 180 s window. Unknown exposure is zero.
All six windows remained at 0/1; pressure 2 was allowed and occurred during
A1 warm-up. These observations do not isolate throttling as the source of
throughput uncertainty.

| arm | level 0 | level 1 | level 2 |
| --- | ---: | ---: | ---: |
| A1 | 51.46 | 128.54 | 0.00 |
| B1 | 93.72 | 86.28 | 0.00 |
| B2 | 117.59 | 62.41 | 0.00 |
| A2 | 90.41 | 89.59 | 0.00 |
| A3 | 0.00 | 180.00 | 0.00 |
| B3 | 0.00 | 180.00 | 0.00 |

Whole model-process observation, including warming/ramp and shutdown:

| arm | level 0 | level 1 | level 2 |
| --- | ---: | ---: | ---: |
| A1 | 195.24 | 178.18 | 20.58 |
| B1 | 176.80 | 172.63 | 0.00 |
| B2 | 285.67 | 62.14 | 0.00 |
| A2 | 256.45 | 89.88 | 0.00 |
| A3 | 70.87 | 279.14 | 0.00 |
| B3 | 79.28 | 270.52 | 0.00 |

Pre-load exposure is separately retained in `settle.json`. All six arms have
the compiler-overlap flag; those flags and traces are retained. No flagged arm
was excluded.

## Paired uncertainty and remaining latency

The 95% Student-t interval uses three independent arm pairs (df=2), not tokens,
as replicates. Relative differences use the corresponding off-arm denominator.

| pair | order | on-minus-off tok/s | relative difference |
| --- | --- | ---: | ---: |
| A1/B1 | AB | -0.2944 | -0.228% |
| A2/B2 | BA | +1.1333 | +0.865% |
| A3/B3 | AB | -5.9944 | -4.515% |

The absolute mean is −1.719 tok/s, 95% CI [−11.087, +7.650]. The first two
pairs were close to neutral; the third was −4.52%. A3 and B3 both spent all
180 measured seconds at pressure 1. B3 offered 22 append TTFT samples versus
A3's 16, and its accounted prefill was 22.158 versus 15.494 s. Turn turnover,
EOS, scheduler order and compiler exposure vary; this packet cannot attribute
the spread solely to thermal throttling or establish a ≤2% throughput loss.

The live maximum is not bounded by the native 170 ms result. B2's 670 ms
peer gap spans completion of a 4326-row append and the next queued append's
work. The first prefill completes 104 ms into that gap; another 311-row append
follows immediately. This is an adjacent-append case absent from the native
single-appender probe. No final-handoff or I/O workaround was added mid-packet.

The admission model also prices each original body/header command separately.
Thus a short append can qualify when it pays two streaming floors. For
example, a 245-row append is predicted at 553 ms and is sliced, although its
observed incumbent active prefill was 464 ms. This coarse prediction near the
threshold and extra peer service are relevant to TTFT; the table reports the
observed distribution rather than promising all short appends are untouched.

## Exactness, coherence and validation

The native regression passes 60/60 paired warm/cold cases at 1/2/4/8 cuts,
including full finite logits, causal-state digests, next decode and both peers.
Ten cancel/dispose/replay cases are exact. Eighteen sampled completions across
all six serve arms remain coherent source-specific Rust/Metal reviews, with
no response errors or replacement characters. Output-limit truncation is
expected; this is a qualitative coherence check, not a factual audit of reviews.

The final frozen source passes 698 serve tests, with 42 pre-existing ignored
tests unchanged; the 23 shared Driver tests include cancellation, one-peer-round
handoff, group-loop rejoin and unequal-geometry fallback. Three new policy and
accounting tests pass. Contracts passed on 3218067d3; records-only validation is
repeated on the final commit. No new ignored tests, roadmap changes or merge.

Raw evidence is retained at
`metagross:~/qwen-runs/2026-11-12-v41-layer-slice/protocol-v2/fn-serve/<arm>/`,
including launch, settle, thermal, native lines, calls, events and responses.
The price probes and build/checksum records are in the parent directory.
Build output, private M5 worktree and transfer bundles are removed after use.
