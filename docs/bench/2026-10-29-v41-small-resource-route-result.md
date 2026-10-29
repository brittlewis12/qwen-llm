# Two-SIMD small-resource GateUp: exact, below the real-route gate

## Decision

Close this implementation without promotion or another timing window. The
complete, accepted M5 Ultra packet passes exactness and full postflight, but
neither independent ABBA block reaches the predeclared 10% GPU-latency reduction.
Host latency improves in both blocks. This is a captured-operand GateUp-only
result, not a router rerun, whole-FFN, FullCED, logits, serving or default result.

| Block | A GPU ms | B GPU ms | GPU reduction | A host ms | B host ms | Host reduction | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 13.650646 | 13.072000 | 4.238963% | 197.431063 | 195.063979 | 1.198942% | fail |
| 1 | 13.549854 | 13.393604 | 1.153149% | 196.277750 | 193.373938 | 1.479440% | fail |

Each row uses two A and two B arithmetic means. Each block independently requires
`B_GPU <= 0.90 * A_GPU` and `B_host <= A_host`; no pooling, sample filtering,
threshold change, selecting the better block, or retry. The native decision is
`complete=true`, `continuation_passed=false`, `performance_qualified=false`,
`decision=close_implementation`. Outer acceptance classifies a completed
negative, not a performance win; no parser repair or replacement acceptance was
needed.

## Mechanism and workload

- A always runs the incumbent Gate and clamped Up: two dispatches. B always runs
  SmallGate, LargeGate, SmallUp, LargeUp: four dispatches, including empty classes.
- Counts 1..16 use dedicated 16-slot x 32-output x K32 kernels, two SIMD groups,
  64 threads and 3072 dynamic shared bytes. Counts 17..1024 retain the incumbent
  32-slot x 64-output x K32 kernels, four groups, 128 threads and 8192 bytes.
  The two count classes are disjoint and exhaustive for nonempty experts.
- Canonical IQ2 decode, F32-to-half rounding, ascending K32/F32 accumulation,
  token-major F16 activations, expert-major ID stride1024, slot-major F32 output,
  in-place Up and model clamp10 remain unchanged. No pack, scatter, paired Down,
  vector4, adaptive four-SIMD or decode composition is selected.
- One fresh FullForward Ordinary capture uses the original
  DeepSeek-V4.1-Flash-1017-GSQ-RCO IQ2_XS backbone and IQ4_NL Engram artifact,
  first 1024 frozen prompt IDs, layer8/start0. Gate and Up banks are IQ2_XXS,
  each `[5120,2304,384]`; hidden5120, intermediate2304, top6. The unchanged
  planner selects4096, but actual scratch/capture extent is1024.
- The capture validates 6144 unique active assignments, six distinct experts per
  token, 121 empty experts and maximum count319. Of the 263 nonempty experts,
  169 have counts 1..16, covering 956 assignments. Saved normalized F16 operands,
  counts and IDs feed the GateUp comparison; the router is not rerun per arm.
- All compute calls write the same F32 `[2304,6144]` buffer. The immutable
  production final GateUp capture is the final oracle; an existing replay buffer
  holds a separately computed, finite, hash-sealed incumbent raw Gate oracle.
  There is no extra oracle payload or CPU payload clone.

These are dedicated resource/geometry changes, not measured register pressure,
occupancy, DRAM traffic or issued-work savings. The earlier adaptive four-SIMD
and token-K32 negatives remain closed. The standalone correctness precursor is
`docs/bench/2026-10-28-v41-small-resource-proof.md`; its old binary and synthetic
clamp7 did not qualify this new binary, actual clamp10 workload or economics.

## Verification and timing

The frozen release test binary passes 406 actual CPU controls and the CPU-only
exact1K metadata header without a GPU lease or context. Before model execution,
the same binary repeats and passes all eight standalone GPU cohorts, including
raw Gate/final Up bits, live-store witnesses, poison/guards, input/control/bank
integrity, actual PSO facts and ordered A2/B4 census.

The native comparison completes all 32 post-construction driver submissions:
seven untimed proofs, fifteen targeted resets and ten compute calls. Proofs
cover both arms' raw Gate and final Up over all 14,155,776 active output words,
plus separate Up-only `f32::MAX` live-store witnesses bounded by 100. The first
targeted reset fills output and oracle; later resets fill only output. Every
checked completion validates defined captures, immutable undefined control
tails, physical guards/slack, untouched replay payloads, sealed oracle and
captured READY status. Source, original banks, artifact/input identity and the
retained input descriptor pass full postflight.

Warm AB records a complete ordered A2/B4 census. The eight primary calls are
uncensused `ABBAABBA`, with observers absent. All 32 absolute GPU intervals are
finite, positive and serial without overlap; all command completions succeed.
GPU time is command-buffer end minus start, including all four B dispatches.
Inclusive host time starts before command creation and ends after checked
completion. Reset, proofs, verification, capture, hashing and reporting are
outside timed compute but inside cooperative phase accounting. The 5us
cross-API containment allowance changes neither comparative gate nor seriality.

Normal resident-model admission remains active. The pack-free adapter baseline
is 1,441,988,608 logical bytes plus 25,600 guards, before device pricing/slack and
fixed pipeline/CPU envelopes. No external FFN workspace or packing payload is
introduced. The native report exposes and the reader verifies the actual cold
load sum: core Metal 155,225,341,952 + extra Metal 1,509,900,288 = combined
Metal 156,735,242,240 bytes. Exposing the core term fixes auditability, not the
already enforced admission calculation or numerical/timing contract.

Native elapsed time is 83.481128625s: capture 11.496469208/40s,
oracle/proofs/warm 31.571092083/45s, primaries 32.7776205/45s and
postflight 7.635910583/20s. The composite accepts its PRE-WRITE
verification-decision at 127.27134275s against 250s. Queue wait and final durable
receipt I/O are excluded; this is not a hard completion deadline. Cooperative
signal handlers, continuously inherited lease FD, checked GPU completion and
context teardown retain safe-drain ownership. Both children exit normally,
without cancellation. CPU build, discovery, header and offline checks hold no
GPU lease.

## Reproduction identities

- Research source: `2258eba6c68db8ee3e2b1a0ac538f73f7662c73d`;
  tree `fb00f57c3aea2315a9c272c9f037744d3764e742`. Test-only source remains
  isolated; this report does not land a production selector or default.
- Hardware/software: Apple M5 Ultra, 256GiB unified memory, Rust 1.99.1 release
  library tests, macOS SDK 27.0, Apple Metal 32023.921. Tensor sources compile as
  Metal 4.0 with target 26.0. Cargo build is offline/locked.
- Frozen binary SHA256:
  `659a8d6fb3009b7db5c3500efa6137b2ac8961b0100313368066fec3a0e82113`.
- Selected 1024 IDs, SHA256 of little-endian u32 words:
  `628c172eb1fc489e4393393e3eec1c039b4a475fcf89ef281f7c37098b9f97df`.
  Source JSON SHA256:
  `aaf5840de38220f6ed4579ffcf83dc828a28e7d71f8e1f2517077ee77020eca2`.
  Input provenance is unknown; no retokenization, alternate window or continuation
  is executed.
- Backbone tensor-layout SHA256:
  `c4a6897d16299794d4727a182dceb5cd731aa61b94b7d09f898929b862897a6e`;
  metadata SHA256:
  `96c677da55cc53effd7596865ad5946434b0730e3b38450b6e1da16d31bcc92f`.
  Retained artifact stamps, metadata/layout and bank seals detect observed drift;
  they are not a fresh full-artifact content hash or authenticity proof.
- Native report SHA256:
  `780dc9f1b141e08148d613de01c9f62d7eda6cbb56d23c9f927f88a12893eced`.
  Accepted composite receipt SHA256:
  `f0a41788dbe9ea331c673d5bfa34eac3194b24e75e49b87bcb6791c2126738fb`.
  Preserved 104-file attempt manifest SHA256:
  `42cd75270efd49563650560df870b607b80cf4dae3998316c6ed3100453c49b2`.

At that research source, the native selector is
`small_resource_route::real_route_native_abba`; the CPU metadata selector is
`small_resource_route::exact1k_header`. The same-binary prerequisite is
`small_resource::gpu_small_resource_gate_up_bits`. The protocol and offline
reader are `docs/bench/2026-10-28-v41-small-resource-route-protocol.md` and
`scripts/reference/deepseek_v41/small_resource_route.py`, schema
`v41-small-resource-real-route-gate-up-v1`. The reader independently recomputes
the published means/gates and checks recorded assertions, not hardware
authenticity. Preserve the complete packet, including both failed blocks; do
not infer whole-model speedup from this operator result.
