# Native Exact CED Host-Range Discovery

Measured source: `27d56bc08810fa0781b84216ae8289f451d54178`, including the
qualified `ca7930a9` FFN/mHC, Metal zero-fill and exact CED baseline. Hardware
is M5 Ultra; SDK 27.0, Rust 1.99.1, canonical opt3/fat-LTO/codegen1 release
lib-test executable, assertions disabled, no diagnostic features.

Binary SHA256:
`d575cda3c7905dff7c79aedf82effdcd481c59cae4ed40410c10e73edcbd8f79`.
The executable was frozen before releasing its build lease. Build wall is
283.563 s within its 330 s cap. No model or GPU work is part of that build.

## Scope

A retains the original per-token host binding/validation chain. B advances
validated command-free prepared ranges. Both use the same compiled kernels,
publication/selection choices and common checked cursor primitives. There is
no FFN kernel, precision, artifact, batch-depth or execution-policy A/B.

This is a short same-binary discovery, not a sustained or formal performance
qualification. Inputs are the existing frozen private IDs, not a recreated
MLX-Serve prompt or a new common cross-engine workload. Absolute rates must
not replace a scoreboard's differently sampled or sustained controls.

## N2048 Actual Model

One resident GSQ import/binding, one live KV state, capacity 2050, common None
storage mapping and production-selected 4096-row geometry: one actual encoder
chunk at N2048. RequiredFullCed joint admission remains active and every arm
executes its requested mode without fallback. Resident import takes 7.910 s;
the complete controller takes 29.709 s within the 60 s process cap.

All chronological prefill API timings are retained:

| Phase | Execution | Host Path | Seconds |
| --- | --- | --- | --- |
| Warm reference | Full | Range | 3.418272625 |
| Warm A | Exact CED | Row chain | 2.143088709 |
| Warm B | Exact CED | Range | 2.060198250 |
| Primary A | Exact CED | Row chain | 2.085035291 |
| Primary B | Exact CED | Range | 2.002377042 |
| Primary B | Exact CED | Range | 1.984059417 |
| Primary A | Exact CED | Row chain | 2.027567083 |

Primary means are 2.056301187 -> 1.993218230 s: 3.068% lower latency,
3.165% higher throughput when derived from mean latency. Both B observations
are faster than both A observations. However, A improves by about 2.8% in
throughput across the block, and there are only two primary observations per
arm. Retain that drift and sample-size limitation; do not treat this as a
stable sustained gain or filter the controls.

Clocks include the production prefill API's allocation, hashing, command
completion, status and endpoint copy. Fixture state creation, teacher forcing,
external state/logit hashes, report/raw-file I/O and teardown are outside the
prefill clocks, but inside the controller's process deadline.

All seven prefills and fourteen teacher-forced scalar steps complete. All
21 finite full-vocabulary vectors match Full float bits, including positions
N, N+1 and N+2. Subsequent CED observations match the old-CED reference's
56 causal region hashes, logical address history, cursor, readiness, origin
and status at every position. Full/CED inactive-state equality is not used as
a requirement. All 21 vectors are retained as a 10,859,520-byte F32LE sidecar.

## Native HOST-Only Discovery

This separate fixture has one ownerless/non-index-producing layer, N128/4096,
query rank 1280, 64 heads, latent width 512, RoPE width 64, BF16 weights,
hidden/output widths 32 and one output group. Prepared payloads are stable
synthetic zeros, not imported model activations. Both arms time the complete
serial API including validation, but neither submits or dispatches GPU work.

All primary samples are in microseconds; each cell contains both observations
from its chronological block:

| Rows | Block | A Samples (us) | B Samples (us) |
| --- | --- | --- | --- |
| 128 | ABBA 1 | 93.583, 94.083 | 0.833, 0.750 |
| 128 | BAAB 1 | 94.333, 93.958 | 0.750, 0.750 |
| 128 | ABBA 2 | 94.250, 94.292 | 0.750, 0.750 |
| 128 | BAAB 2 | 93.792, 94.667 | 0.750, 0.708 |
| 4096 | ABBA 1 | 2805.459, 2780.792 | 0.792, 0.791 |
| 4096 | BAAB 1 | 2787.458, 2620.541 | 0.708, 1.000 |
| 4096 | ABBA 2 | 2721.625, 2742.542 | 0.750, 0.709 |
| 4096 | BAAB 2 | 2648.916, 2766.417 | 0.750, 0.917 |

Warm A/B observations are separately 97.542/1.250 us at N128 and
2929.916/1.042 us at N4096. All 36 samples have empty GPU dispatch and Metal
allocation censuses, unchanged original bindings and identical cursor
outcomes. The whole fixture takes 0.226 s. Its priced Metal storage is
1,962,917,888 bytes plus a 536,870,912-byte CPU reserve, below 4 GiB.

These host-slice ratios are not engine throughput multipliers. The earlier
M4 optimized-test-profile packet remains distinct and is not relabeled as
native M5 or release evidence.

## Correctness And Evidence

The canonical image also passes 337 selected CPU tests. Closed library
discovery matches all 531 classified witnesses and 93 ignore states. Two
DEBUG1 correctness witnesses pass in 0.592 s combined, including H64/N128
attention-route coverage, exact old-host/range outputs/controls/pending state,
ordered dispatches and poisoned CED replay cache immutability.

Private packet directory: `2026-10-27-v41-range-execution` on Metagross,
with local counterpart `v41-range-execution-2026-10-27`. Retained evidence
includes `build-27d5-v1.*`, `range-native-27d5-core.identity.json`, frozen
binary/source bundle, `cpu-27d5-v2.*`, `parity-27d5-v1.*`, `host-27d5-v1.*`,
and `model2k-27d5-v1.*` plus its raw sidecar. All admission waits and busy
refusals remain separate; waiting is not GPU execution time. No failed state
or partially executed prefix is reused, and there is no sample filtering.
