# Vector4 Weight-To-Half: Exact, Whole-FFN Continuation Gate Missed

The vector4 producer's operator signal does not clear the fixed actual-routing
1K whole-FFN gate. All eleven native replays pass their exactness contracts,
but both ABBA blocks miss the required 1% GPU latency reduction and regress
inclusive host latency. Close this continuation; production defaults and
allocation remain unchanged. No GPU rerun or favorable-window selection occurs.

## Frozen Source And Workload

- M5 Ultra, 256 GiB, native TensorOps tier; release Rust build, Metal compiler
  32023.921 and SDK 27.0, with the existing Metal build policy.
- Research source `102fb825f23db9de5286920999855ec84995a21d`, tree
  `05eef0d7e91f1930af35260be65d4b9fbfab0bc0`, incorporating maintained source
  through `421f2054`. Later batched-decode and serving work is not part of this
  frozen executable.
- Frozen executable SHA-256:
  `ae370c3d7bb59aa505bf8ee2a6cd5b6710090395743a205dc34427035152d82b`.
  Build, freeze, discovery, all 385 classified CPU checks and the exact CPU-only
  header fixture pass without a GPU lease; discovery finds 2,946 library tests.
- Existing DeepSeek V4.1 Flash GSQ-RCO IQ2_XS mixed-quant artifact and its
  IQ4_NL Engram companion; no replacement weights or dtype substitution.
  Layer 8 uses IQ2_XXS Gate/Up and IQ2_S Down, with hidden 5,120,
  intermediate 2,304, 384 experts and top-6 routing.
- One successful production FullForward capture at start 0, using the first
  1,024 prompt IDs. Actual arithmetic is `Ordinary`, scratch capacity is 1,024,
  and the stock selected planner width remains 4,096. The incoming HC value is
  current attention `hc.pre`, not a lagged incoming tensor.

Selected IDs have SHA-256, over little-endian u32 words,
`628c172eb1fc489e4393393e3eec1c039b4a475fcf89ef281f7c37098b9f97df`.
The frozen source JSON contains 8,192 prompt and 256 continuation IDs and has
SHA-256 `aaf5840de38220f6ed4579ffcf83dc828a28e7d71f8e1f2517077ee77020eca2`;
no continuation tokens enter this experiment. These hashes identify the retained
inputs, not an independently available public corpus.

The same new executable first repeats the ten-cohort vector4 GPU oracle over
141,560,517 half words. It passes before model variables are supplied to the
native child. This is correctness evidence, not a timing sample.

## Comparison And Exactness

A runs the original F32 weighted intermediate followed by half conversion.
B replaces only those two adjacent MID dispatches with the qualified vector4
producer. Both use one captured boundary, the same immutable banks and the same
physical Replay buffers. The router and remaining FFN are recomputed on every
call; saved routes serve as references, not substituted routing inputs.

B intentionally does not materialize F32 `b13`. Its entire working payload
must retain A5 poison, while the captured `b13` reference stays sealed and
immutable. The verifier explicitly checks 15 materialized FFN states for A
and 14 for B, plus six HC states, both half outputs and final output. It does
not call the elided intermediate computed or claim all 15 FFN states match B.
All other defined stage bits, status, input/control data, spare capacity and
guards pass for every call. Raw bank hashes match before and after replay;
captured references remain sealed and immutable.

The actual live population contains 6,144 routed slots and 121 zero-count
experts, with unique complete slot ownership checked. It is freshly captured
at 1K, not inferred by scaling a 4K population.

Before each compute, a separate GPU command restores all working payloads to
raw A5 words and target status to Pending. Checked completion and full reset
validation precede compute. All eleven reset commands are recorded separately
and remain outside both compute timers. The F32 allocation remains present in
both arms; this experiment establishes no allocation reduction.

## Fixed Economics

Order is counter-free diagnostic A, census-bearing warm A/B, then uncensused
`A B B A`, `A B B A`. All eleven compute commands and eleven resets complete
serially with no command error. Warm census proves exactly the two-to-one MID
substitution, actual narrow-1K Gate/Up and Down geometry, and identical other
ordered dispatches. Primaries have no stage counters, census or profiler.

GPU latency is command end minus start. Inclusive host latency starts before
command creation and ends after checked completion. Reset, validation, hashing
and reporting are outside those timers but inside the native phase budgets.
Each block independently requires `B_mean_gpu <= 0.99 * A_mean_gpu` and
`B_mean_host <= A_mean_host`. The fixed 5 us cross-API containment allowance
does not relax either comparative gate.

| Block | A GPU Mean, ms | B GPU Mean, ms | GPU Reduction | A Host Mean, ms | B Host Mean, ms | Host Regression |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 20.099563 | 20.130125 | -0.152055% | 358.229146 | 368.506979 | 2.869067% |
| 2 | 20.159917 | 20.025812 | 0.665202% | 361.540146 | 363.387125 | 0.510864% |

Both blocks fail both conditions. Native and independent reductions agree on
`continuation_passed=false`, `performance_qualified=false`, `decision=close_leaf`.
The native packet completes in 56.46095325 s; capture, diagnostic/warmup,
primaries and postflight stay within their fixed 40/25/35/20 s phase caps and
120 s native bound.

CPU splits place approximately 0.356-0.369 s per compute in the checked-wait
envelope, versus roughly 57-66 us creating/encoding and 10-13 us committing.
This locates an API envelope, not the cause of its gap from GPU timestamps.
GPU reset does not establish an explanation or cure for the earlier host/GPU
gap, and the old paired-Down negative is not reinterpreted.

## Original Wrapper Failure And Offline Analysis

The native child exits successfully and is reaped with the canonical descriptor
retained through drain; no cancellation signal is sent. The original outer
parser then fails on its CPU diagnostic split containment check. For the last
call, the serialized decimal seconds are exactly:

```text
0.000059959 + 0.000010333 + 0.357715292 = 0.357785584
```

Python's ordinary binary-float summation exceeds the recorded host value by
`5.551115123125783e-17` s. A separate copied adapter uses exact decimal
arithmetic for this diagnostic comparison only. All other assertions and the
SHA-bound public reducer remain unchanged. All 93 offline controls pass,
including rejection of genuine 1 ns split overruns; no timing tolerance,
performance gate, sample or raw packet changes.

That separate analysis validates the completed native negative, including all
stage/elision, reset, census, provenance, bank and native postflight checks.
It does not synthesize original outer acceptance. The original campaign
remains failed, its final outer postflight and 140 s acceptance are not
established, and 56.46095325 s is native elapsed time only. A later unleased
CPU postflight independently matches the pre-GPU binding; it is not relabeled
as the missing original outer postflight.

Native packet SHA-256:
`8f6f16bcc0d6baf4ff6296ba335f1f7012b8c70b7c663c081fe83477a6ac74ab`.
Original failures, attempt latches, raw samples and the isolated analysis remain
retained separately. A first compiler failure in a CPU census fixture is also
preserved; its sole source repair changes fixture parameters from `usize` to
the dispatch record's `u64`, before any GPU attempt.

## Decision

Do not promote vector4 on its operator numbers or select only the better
whole-FFN block. This experiment establishes no Full/CED parity, logits,
likelihood, sustained throughput, DRAM traffic or model-speed improvement.
The next distinct hypothesis is token K32-major input layout shared by Gate
and Up, with its pack cost included on every candidate call. Its bit proof and
separate actual-route economics must qualify independently.

Operator evidence: `docs/bench/2026-10-28-v41-vector4-weight-half.md`.
The frozen research protocol is
`docs/bench/2026-10-28-v41-vector4-ffn-protocol.md` at source `102fb825`.
