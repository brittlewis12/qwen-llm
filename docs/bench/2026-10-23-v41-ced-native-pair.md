# Native V4.1 Full/CED Pair: Short-Prompt Loss, 2K Timing Rejected

## Scope

Experimental source `6896f87d` compares production Full forward and shrinking
CED inside one resident Maya-S model, with fresh KV state for every call. The
compiled/runtime identity matches and both are clean. This is synthetic cold-KV
prefill, not model loading, sampled generation, artifact quality or bounded128.
The implementation and further diagnostics remain in `perf/v41-ced-bench`;
there is no default-policy promotion.

Native timing includes prefill planning/admission, temporary allocation, CPU
Engram work, GPU encoding/completion and endpoint logits copy. State creation,
head fingerprinting, diagnostic serialization and teardown are outside that
timer. It is not GPU-only time or total request-lifecycle time.

## Protocol And Admission

- Warm A/B once, run an excluded ABBA pilot with two samples per policy, then
  retain a balanced BAAB block with two samples per policy. A is Full; B is CED.
- Check finite positive native samples and max/min <=1.5 separately per policy,
  including the pilot baseline, before scheduling further cases. Do not pool
  the intended difference between A and B into that stability ratio.
- Require the requested execution without fallback, completed CED commands,
  observed incremental Metal allocation no larger than its accepted estimate,
  and finite complete endpoint heads matching warm Full bitwise.
- Keep debug validation, the production lease and ordinary memory admission.
  Continuous secondary host supervision remains required; it does not replace
  core admission. Failure stops the sequence without automatic retries.

Public metadata-only cold planning now shares runtime geometry and prices
resident views/private shadows, binding allocations, one request state, the
sequential Full/CED workspace envelope and caller CPU storage. Binding controls
use actual validated tensor types and shared projection descriptors. The
256-layer all-owner ratio-two Q8_0 fixture needs 4,097 separate 8-byte controls:
at 16 KiB resource pricing, 64 MiB plus one page, not one aggregate allocation.

The caller envelope supports at most 16 retained samples per policy, 256 layers
and 1,048,576 logits. Its isolated maximum-size CPU regression charges 33.89 MiB
against the 64 MiB allowance, including live head/bit-vector capacities, both
aggregates, duplicated chronology and final JSON serialization. This RSS
observation is a regression witness, not a formal heap proof.

## Measurements

M5 Ultra, 256 GiB; Maya-S plans 165.950 GiB resident views and zero private
shadow. Results concern this artifact, not the larger IQ3 artifact or its quality.

| Prompt | Full retained native samples | CED retained native samples | Verdict |
| --- | --- | --- | --- |
| 128 | 304.631, 305.713 ms | 462.065, 466.234 ms | Stable; CED latency is 52.1% higher |
| 2048 | Incomplete | Incomplete | Rejected; no comparative timing result |

The 128-token run completes all ten chronological calls and all endpoint
fingerprints agree. Minimum sampled conservative host availability is 73.66 GiB;
maximum sampled wired plus compressor usage is 174.18 GiB. The short-prompt
result does not predict long-prompt CED behavior.

At 2048, the pilot Full calls take 17.05 and 16.96 seconds. Its later retained
calls take 14.91 and 2.35 seconds, so Full's pilot-plus-main max/min reaches
7.272. The CLI rejects the sequence immediately after that ninth chronological
call, before scheduling the final CED case. Neither a partial mean nor the
single fast sample is a qualified comparison. No 8K timing follows this failure.

All recorded 2K endpoint heads still match. CED completes all 38 commands in
each executed case, with sampled incremental Metal allocation 2,981,986,304
bytes versus an accepted 3,004,235,776-byte estimate. Minimum sampled host
availability is 70.53 GiB; maximum wired plus compressor usage is 177.06 GiB.
There is no unhealthy host sample or watchdog incident in this sequence. These
samples are not continuous memory peaks or an OS reservation.

## What Remains Unknown

One resident object and one bound forward survive all cases; mappings, binding
and control-vector decoding are not recreated per case. Completed-work pools
and state teardown remain intrinsic. Engram still performs per-request reads
and dequantization, and pipelines are lazily cached, so retained ownership does
not imply constant physical residency, I/O cost or complete pipeline prewarming.
None of those facts attributes the observed phase shift.

CED also samples allocation through full memory signals inside its execution,
unlike Full. That instrumentation asymmetry is real but its cost is unmeasured;
it is not an explanation for the discontinuity or the 128-token difference.

Localize the 2K shift with explicitly instrumented case-level process I/O,
pipeline-cache and command encode/GPU/wait evidence, retaining the same stop
rules. Keep preparation/Engram and teardown outside command-timing attribution.
Instrumented results are diagnostic, not a substitute for a subsequent stable
unchanged native protocol. Do not infer thermal or external-job interference
from post-run observations alone.

## Verification And Next Kernel Bet

M4 verification passes 71 selected policy/planning/CED CPU tests, 1,513 CLI
library/binary tests and affected release all-target checks. The three bounded
metadata/runtime parity fixtures plus Full lifetime and half-allocation fixtures
pass on M4 and M5; corrected host-reserve boundary fixtures and explicitly
self-leased test entry points are recorded in `cc0a05c1`.

The earlier strict current-Full/CED equivalence still passes at 128/2048/8192:
all 257 full-vocabulary vectors bitwise and all 56 causal regions at every
position. See the [memory and qualification packet](2026-10-23-v41-memory-safety.md).

Exact hierarchical top512 selection above 4K remains an independent bounded
experiment (`perf/v41-select-hier`). Partition-local top512 plus exact merge
targets the source-derived 83,880 legacy sort/chronological dispatches in 8K CED.
It preserves signed-zero/tie/NaN behavior, causal visibility and absolute IDs,
with capacity-priced tile scratch and an incumbent rollback. Review does not
establish GPU correctness or wall-time improvement; production-tile reuse and
borrowed CED control views need bounded execution proof before model trials.
