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
with capacity-priced tile scratch and an incumbent rollback. At the initial
checkpoint, review alone does not establish GPU correctness or wall-time
improvement. Subsequent bounded and end-to-end proofs are linked in the follow-up.

## Follow-Up: Phase Planning And Attribution

The experimental planner separates full-future capacity feasibility from
availability at the actual phase boundary. It preserves the 80% ownership
ceiling, 20% host margin, positive process/device constraints and private/import
debt. Before import it admits resident views, binding and caller storage; after
binding and before each fresh state it reassesses the complete runtime envelope
with fresh signals. No active-cache credit or telemetry exception is introduced.
The original strict cold-peak wrapper remains available for qualification.

This avoids requiring every future allocation to fit a pre-import availability
snapshot, without assuming shared-cache reuse will occur. If runtime availability
does not suffice afterward, no new case state/prefill starts. Later production
CED fallback remains guarded Full execution and still invalidates a paired case.

Source `c1fdca25` passes a fitting supervised 2K diagnostic through those phase
checks, then rejects a 2.509x CED spread after its fifth chronological case.
Recorded GPU durations remain approximately 2.00 s Full / 1.75 s CED; CED's
completion-wall sum changes from 6.57 to 18.70 s while its GPU sum stays near
1.75 s. Warmed process reads/page-ins and recorded pipeline misses are zero.
These counters do not rule out all I/O or driver work, and the wait interval
does not identify scheduling, validation, residency or completion delivery.

The new aggregate block-row allocation/initialization span measures about
1.18 s across CED encoder/decoder constructors. It overlaps the encoder's host
preparation span; it is host wall, not additive CPU work. Request-local storage
reuse is a separate measured-code-path bet. Replacing allocation observations'
full host/task queries with the same raw device getter does not establish a
native speedup: a separate pp128 pair remains approximately 306 ms Full /
474 ms CED. Do not attribute the observed short-prompt loss to those queries.

The combined M4 checkpoint passes 156 forward, 17 FFN, 11 bounded CED, four
selector GPU and 1,525 CLI tests plus release all-target checks. Independent
reviews find no remaining staged-admission or lifecycle correctness issue.
All-current/frozen 8K head/state evidence is recorded in the
[exact-selector packet](2026-10-23-v41-select-hier.md).

Measurement regimes need explicit separation: recent V4.1 production-throughput
packets leave API/shader validation unset, whereas this initial pair contract
requires API validation. Historical rows prove that API validation was requested;
they do not record every shader-validation override or prove effective API-only
instrumentation. A separately identified production-timing regime is in flight;
correctness tests keep validation enabled and all memory/lease/stop checks stay.
No evidence yet shows that validation caused the completion-wall variability.

Whole-model explicit residency remains closed under the standing
[failure decision](../PERF-LOG.md#2026-08-13-whole-model-residency-optimization-lane-closed).
Do not reinterpret the wait gap as permission to pin, pre-wire, use `mlock`,
force cache cleanup or relax admission.
