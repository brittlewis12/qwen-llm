# Native Exact CED Host-Range Discovery

N2048/HOST measured source: `27d56bc08810fa0781b84216ae8289f451d54178`.
The N8192 follow-up uses `1e0190168d21265fe9ff24b072fa15ec39ac6981`, adding
canonical inherited-lease adoption and runner descriptor forwarding without
changing model computation. Both include the qualified `ca7930a9` FFN/mHC,
Metal zero-fill and exact CED baseline. Hardware is M5 Ultra; SDK 27.0, Rust
1.99.1, canonical opt3/fat-LTO/codegen1 release lib-test executable, assertions
disabled, no diagnostic features.

Binary SHA256:
`d575cda3c7905dff7c79aedf82effdcd481c59cae4ed40410c10e73edcbd8f79`.
That initial executable was frozen before releasing a GPU lease unnecessarily
used for build exclusion. Build wall is 283.563 s within its 330 s cap. No
model or GPU work is part of that build. The follow-up corrects the mechanism:
an isolated CPU Cargo target and separate CPU build lock, with no Metal lease
during compilation or CPU tests.

## Scope

A retains the original per-token host binding/validation chain. B advances
validated command-free prepared ranges. Both use the same compiled kernels,
publication/selection choices and common checked cursor primitives. There is
no FFN kernel, precision, artifact, batch-depth or execution-policy A/B.

This is a short same-binary discovery, not a sustained or formal performance
qualification. Inputs are the existing frozen private IDs, not a recreated
MLX-Serve prompt or a new common cross-engine workload. Absolute rates must
not replace a scoreboard's differently sampled or sustained controls.

## N8192 Actual Model Follow-Up

Binary SHA256:
`4ac824276b6fba97d1504dd9e43450e0aeaa5cd0b510e91f15ea17dcd247f65b`.
The CPU-only build/freeze takes 285.842 s within a 600 s cap. Cached
dependencies are APFS-cloned into a private target; only that target's qwen-llm
package is cleaned/rebuilt. The executable is copied and hashed outside the
target while a separate CPU flock remains held. Its identity records
`gpu_lease_used: false` and `frozen_under_cpu_build_lock: true`.

All 338 selected CPU tests pass in 2.664 s without a GPU lease. Closed library
discovery matches 532 classified witnesses and 93 ignore states. One canonical
Metal lease then covers DEBUG1 parity, native HOST and one N8192 model run,
without releasing/reacquiring between phases. Admission waits 796.934 s without
owning the GPU; the complete admitted transaction takes 49.886 s. Parity and
HOST controllers take 0.412 and 0.296 s, respectively. Model controller wall
is 48.531 s within its 90 s cap; test wall is 48.36 s. The model records
`lease_mode: inherited_adopted`, FD 3 and WAIT 0. No build occurs under this lease.

One resident GSQ import/binding, one live KV state, capacity 8194 and common
None storage mapping remain unchanged. Resident import takes 0.064 s; filesystem
caches are not reset, so this is not cold-load evidence. Strict RequiredFullCed
admission selects the normal 4096-row width and two actual encoder chunks, with no
pressure-demotion override or fallback.

All chronological prefill API timings are retained:

| Phase | Execution | Host Path | Seconds |
| --- | --- | --- | --- |
| Warm reference | Full | Range | 9.092846875 |
| Warm A | Exact CED | Row chain | 5.534206209 |
| Warm B | Exact CED | Range | 5.073475666 |
| Primary A | Exact CED | Row chain | 5.286332833 |
| Primary B | Exact CED | Range | 5.053403958 |
| Primary B | Exact CED | Range | 4.947243416 |
| Primary A | Exact CED | Row chain | 5.099267417 |

Primary means are 5.192800125 -> 5.000323687 s: 192.476 ms / 3.707% lower
latency, or 3.849% higher throughput derived from mean latency. Both B samples
are faster than both A samples. A improves by 3.668% in throughput across the
block, however, and there are only two primary samples per arm. No sample is
excluded. This is preliminary same-binary evidence, not a sustained gain or a
replacement for the differently sampled 1,662.9 tok/s baseline scoreboard.

All seven prefills, fourteen teacher-forced scalar steps and 21 complete finite
vocabulary vectors pass. Every CED vector equals Full float bits at N, N+1 and
N+2. All old-host/range CED observations match the 56 causal region hashes,
logical address history, cursor, readiness, origin and status at each position.
Each CED arm encodes/completes all 12 production commands, with 163840 encoder
output rows, 8192 publication rows, 24150 decoder output rows and 2540 seed rows.
All vectors remain in the 10,859,520-byte F32LE sidecar. A separate local
reduction revalidates every vector's bytes/hash against Full, all CED state
observations and every receipt's evidence hashes.

The repeated HOST fixture retains all 36 samples/warmups: primary N4096
row-chain 2.774250-2.852000 ms versus range 0.708-1.125 us; N128 row-chain
94.334-95.000 us versus range 0.750-0.875 us. Warm A/B are
3.007292 ms/1.208 us at N4096 and 104.500/1.166 us at N128. All samples have
zero dispatches, unchanged bindings and matching cursors; no command is
submitted. These ratios remain host-slice observations, not engine multipliers.

Evidence is under the private packet's `native-1e01-cpu-lock/` subdirectory:
`build-1e01-v1.*`, `range-native-1e01-core.identity.json`, `cpu-1e01-v1.*`,
`cpu-discovery*`, `gpu-1e01-v1.*`, and its `-parity`, `-host`, `-model8k`
children, including raw vectors. The source bundle and frozen executable are
retained remotely; reports/raw vectors and the independent reduction are also
retained locally. Model report SHA256:
`9e4f2de318382e839bda26ebf746a50c3c4ed916a32e7a1e5a7c622bd0bbfc2f`.
Earlier admission-only busy refusals never reached context/import/prefill and
remain separate evidence. This admitted packet executes the prefix only once
per declared arm, without retries or win filtering. `performance_qualified`
remains false. The inherited publisher rollback/pruned eligibility caveat is
unchanged; no alternative publisher policy is qualified here.

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
