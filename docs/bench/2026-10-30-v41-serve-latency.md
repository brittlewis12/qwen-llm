# V4.1 Serve Latency: Idle Driver Delay, Long Tail Still Open

## Status

The reported 13-26-second cached-prefill stalls are real in the earlier serve
logs, but the controlled reload sequence did not reproduce them. They are not
declared fixed. Opt-in unsplit native-call timing can attribute a recurring
stall without guessing from TTFT alone.

Diagnostic source: `c6e1d4c4`, based on `4accd031`. M5 Ultra, 256 GiB, SDK 27.0,
canonical release. The wrapper changes no arithmetic, batching, command graph,
residency policy, or checkpoint policy. It calls `Scope::begin()`, not stage or
per-dispatch sampling. The frozen executable SHA256 is
`3c68224e8ae4e9208e2ff579b7a10637ea32293e84a015f388b36b326582f56d`.

## Measurements

One fixed saved request per model retains its original 32,768-token output
reservation. The reload experiment stops client consumption after 16 primary
delta events; duplicate reasoning event aliases are excluded. Consequently
these are bounded diagnostic requests, not complete-turn throughput tests.
All V4.1 phases use the same saved request; first deltas include reasoning.

| Phase | Cold TTFT, seconds | Cached TTFT, seconds |
| --- | ---: | --- |
| A0: V4.1 reload | 15.446 | 1.538, 1.556, 1.522 |
| A1: V4.1 reload again | 13.488 | 1.607, 1.650, 1.634 |
| GLM-5.3-Flash Q4 | 21.891 | 0.822 |
| Qwen3.8-Flash-Next IQ3_S | 22.283 | 0.419 |
| B: V4.1 after both other models | 17.208 | 1.428, 1.643, 1.684 |

There is no long-tail reproduction in these samples. The fixed V4.1 cold
prompt has 15,564 tokens, so its cold TTFT must not be compared with the
earlier incident's shorter cold opener. Shortened generation also differs
from the original multi-turn GLM and Flash Next usage; this experiment does
not prove that switching models cannot trigger the condition.

The original early V4.1 turns were then replayed with their original sampling
seed, reasoning policy and user messages, preserving the newly generated
assistant reasoning exactly as the client does. The first three generations
completed before the next turn; only the fourth was cut short.

| Turn | Prompt tokens | Matched tokens | Reuse | New tokens | Capacity | Prefill seconds | TTFT seconds |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 0 | 6250 | 6246 | RAM snapshot | 4 | 39017 | 1.519 | 1.974 |
| 1 | 6445 | 6413 | Live session | 32 | 78034 | 1.633 | 1.837 |
| 2 | 7454 | 7427 | Live session | 27 | 78034 | 1.626 | 1.774 |

The generations contain 164, 983 and 582 tokens, matching the earlier trace's
counts. Turn 1 exercises the same capacity doubling as the earlier 13.367-second
prefill. Thus neither prefix loss nor that growth alone explains the historical
stall. The fourth request's TTFT is 1.815 seconds. These results do not erase the
earlier bad observations or establish that every live session is unaffected.

## Reproducible Idle Cost

An identical cached two-token request was sent after five seconds idle, then
immediately repeated. This is a workload discriminator, not a deployed fix.

| Observation | After five seconds idle | Immediate repeat |
| --- | ---: | ---: |
| Client TTFT, ms | 1434.556 | 244.537 |
| Native-call wall, ms | 1287.136 | 150.512 |
| Encode, ms | 1.502 | 0.841 |
| Commit/wait, ms | 1284.877 | 149.266 |
| GPU envelope, ms | 144.873 | 144.638 |
| Metal driver scheduling interval, ms | 1138.789 | 4.410 |

GPU time remains effectively unchanged while the Metal-reported driver
interval changes sharply. Other cached controls show approximately 1.1-1.26
seconds in that interval after idle, with roughly 145 ms GPU time. The driver,
GPU and inclusive wait intervals are not assumed to be an additive partition.
In particular, wait minus GPU is not pure application queue time.

This makes idle/resource-residency handling a concrete performance target.
It does not prove that the historical 13-26-second tail has the same cause.
Earlier retained-window experiments observed much larger driver intervals on
another artifact; that remains supporting context, not a diagnosis of this run.

## Diagnostic Hook And Handoff

`QWEN_SERVE_V41_PREFILL_TIMING=1` emits one `serve native prefill timing:` JSON
record per native call in both continuous and serial serving. Records contain
position, rows, logits mode, success, call wall time and the existing native
profile, including scratch/host preparation and unsplit command intervals.
The default is off. Diagnostic setup failure warns and runs the original call;
native results and errors are preserved. No stage splitting is introduced.

At the diagnostic handoff, V4.1 was restored with its original endpoint and
limits and this diagnostic hook enabled. Saved game files were not changed.
The original executable and incident log are retained privately. Expected client-disconnect
messages result from bounded probes, not an observed server failure.

The next actual long stall should be matched to its native-call record and
outer `serve phases` timer. That will distinguish a native allocation/encode,
driver/GPU wait, or caller-side checkpoint delay. The diagnostic campaign did
not enable persistent idle keep-alive or smaller retained windows.

## Opt-In Implementation Checkpoint

Two independent serving controls now exist; both preserve existing defaults:

- `QWEN_SERVE_V41_WEIGHT_WINDOW_MIB=4096` requests a 4 GiB soft coalescing
  target through the existing storage planner and checked loader. Unset/0 uses
  the ordinary loader. Oversized singletons remain eligible; this is not a fit
  cap. The startup report describes mapping exposure, not physical residency.
- `--idle-residency-secs` or `QWEN_SERVE_IDLE_RESIDENCY_SECS` enables the shared
  resource-aware pulse. V4.1 defaults to zero, unlike the shared 60-second
  default. Pulses name deduplicated bank/dense backing buffers, never Engram.
  Continuous batching observes synchronous compute quanta rather than
  overlapping connection lifetimes. Native errors close the window even if
  cancellation masks the HTTP error. Successful startup decode warm-up opens
  an explicitly enabled window; loading alone does not.

Focused CPU checks cover parsing, overflow, sparse mapping geometry, activity
attribution, late completions and cancellation-masked faults. The full CLI run
has 1028 passes, 50 explicit ignores and one info-baseline failure independently
reproduced on untouched `adad4461`. Eleven existing CPU soft-window checks pass;
the extended actual-resource GPU witness was ignored and unexecuted at this
checkpoint.
Independent source re-review has no remaining P1/P2 findings.
`cargo check --locked --offline --workspace --bins`, formatting and the generated
environment-reference checks pass. The local build uses the available Metal 4.0
tier; no Metal 4.1 tier or M5 execution is inferred from CPU checks.

No current-artifact M5 performance comparison or deployment was included in the
implementation checkpoint. Test each control independently, with the same
artifact, matched prefix/suffix, capacity, output reservation and sampling.
Record ordinary TTFT, native wall, GPU/driver intervals, checkpoint/restore,
memory pressure and all samples. Distinguish immediate and idle requests; do
not merely move a request stall into an expensive pulse. The original long-tail
cause remains open.

## Measured Controls And Completed Expiry Follow-Up

The subsequent [serving-controls packet](2026-10-30-v41-serve-controls/README.md)
records the frozen `5bf7e4e5` M5 release, with separate sanitized cached CSVs.
The first six-arm cohort has 30 complete, byte-matching 534-token generations
(24 cached). Five-second TTFT medians are 744.982 ms ordinary, 734.716 ms soft
4 GiB and 220.555 ms with a ten-second idle window. Mapping remains negative.

The completed A2/E0 ordinary-logging follow-up (`RUST_LOG=info`, no per-quantum
debug) adds ten complete matching generations, eight cached, not pooled with
the first cohort. Five-second TTFT is 745.873 -> 220.004 ms, with driver delay
531.559 -> 5.907 ms and idle GPU envelopes about 145 ms. E0 at 15 seconds
returns to 738.144 ms TTFT / 524.174 ms driver delay, supporting expiry at this
narrow gap. No obvious decode regression or recorded request/native/pulse/
cleanup failure appears. This supports ten-second keep-alive for the measured
workload, not a global default. Power, pressure and arbitrary conversations
remain unmeasured; the original 15-40-second tails are not explained or fixed.
