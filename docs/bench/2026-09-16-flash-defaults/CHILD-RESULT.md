# Child Attribution And Validation: Bounded Closure

No production optimization or new switch is delivered by this packet. Guarded
top-k and HC remain qualified defaults; retired split QSA stays retired. The
qualified parent observation is unchanged, but local child GPU attribution is
INCONCLUSIVE_NO_RETRY. No additional segmentation/replay/timing sweep is justified.

## Native Observer

Fresh cx Luna01a0ac00-145e-7392-8465-480c3c01eb28 correctly challenged isolated
timing: it loses native command position/cache conditions. We omitted unused
isolated capture/replay infrastructure and chose native source markers instead.
The attempted dispatch observer then stopped on an unsupported device capability
before model load or any forward (FAIL0.03s). Implementation fe20770e and protocol
f57da2d8 preserve this failure. We did NOT put manual markers on a stage-only
counter buffer. Unsupported counter code is removed from the current test path.

The versioned native host/source packet uses no GPU subgroup markers or buffers.
It completes five forwards after one2179-token prefix, preserving full logits,
hyper residual and all121 terminal tensors bitwise across all five arms. Dispatch
sequence and shapes across the full graph also match. Raw rows/state, host records,
cohort metadata, census and source-group membership persist before gates.

| Axis | Ordinary before | Profiled native | Ordinary after | Control drift | Observer delta |
|---|---:|---:|---:|---:|---:|
| GPU ms |48.203500|45.569459|49.821083|3.30036%|-7.02443%|
| Executor wall ms |54.078625|51.553000|55.756292|3.05489%|-6.12639%|

Control<=5% and wall observer within10% pass. GPU observer outside5% FAILS even
though the profiled arm is faster. Harness completion21.72s is not performance
qualification. No retry, adaptive warmup or subgroup GPU ranking follows.

## Structural Findings

Eleven QSA-containing blocks, layers3..43, have identical named weight dtypes and
shapes, plus identical40-dispatch kernel sequences/shapes. Layer47 differs: Q8
routed down replaces IQ4_NL and its separate weighted sum, yielding39 dispatches.
Thus layer39 is a structural sentinel for11 blocks, not every QSA block's timing.

Sixteen source groups cover layer39's40 dispatches: two four-kernel HC reads;
four index-preparation kernels; one periodic pool publication; three index
score/radix-select/expand kernels; six QKV/norm/publication kernels; two incumbent
attention kernels; one output projection; two copies; two two-kernel combines;
three router/shared-gate kernels; three routed-expert kernels; two shared-expert
kernels; one final MoE accumulation. Full source membership is saved in each
profiled source-groups file. Counting launches does not rank GPU costs.

Raw host block spans are0.117458ms warm and0.113041ms measured; measured leaves
sum0.107665ms with0.005376ms uncovered and no overlap. Largest recorded host leaf
is QKV/norm/publication0.015666ms. These are unqualified encoding diagnostics,
not GPU durations, precision guarantees for tiny leaves, or measured savings.
Position2179 also fires one-in-four pool publication; do not price it every token.

BF16 index projections are K2560 by512+128 outputs:3,276,800 weight bytes per
QSA layer,39,321,600 across12. `kernel_mat_vec_bf16_f32` already reads vectorized
ushort4/float4 operands with independent ordered row accumulation. The bytes
are not deletable merely by fusing the projections. QKV/output Q8 weights total
52,920,320 bytes per layer; simple fusion still reads those weights. These are
logical traffic counts, NOT achieved DRAM traffic or timing ceilings. No source
fact here supports another BF16/QKV/copy candidate above the whole-forward gate.

## Independent Validation Ceiling

Source audit finds repeated deep validation in session, block and public child
entry points. But runner weights are bound once: per-token weight rebinding is a
rejected hypothesis. CPU leaf spans omit parts of root validation, so we used one
separate CPU-only ceiling packet rather than pretending the child profile measured it.

Production lease/real wired gate/API validation, real metadata and empty scalar
workspace; an empty serial encoder is never committed. Two warm calls and32 timed
root validations, census disabled during timing. Warm census empty, persistent
bytes/position/logit readiness unchanged. PASS0.19s with median0.1718545ms,
mean0.1682669ms, min0.157542ms, max0.176583ms. Median misses the frozen>=1ms design
gate: HOLD_NO_OPTIMIZATION. Even the entire pass is not safely removable; dynamic
state/ownership/poison/publication checks remain mandatory. No contract cache or
validation bypass is added. Do not multiply nesting counts into a claimed saving.

Sol01a0ac1c-f130-7ad2-baae-17b9ad220fe7 timed out without a final response. Its
persisted commentary independently found the repeated-validation structure, but
is partial source feedback, not final approval or timing evidence. Final Luna
review agrees to close the packets; its suggested summary incorrectly says
per-token weight binding was confirmed. Source shows it is already bound once;
we reject that sentence rather than propagate a review error.

## Continuous-Decode And Donor Rechart

The bounded source walk finds one production command/serial encoder per token,
one root completion wait, child publication after completion, and a copied logits
row passed to the serial sampler. Weights persist in the runner. Checkpoint restore,
full-state readback and artifact writes belong to qualification harnesses, not the
continuous CLI path. Their idle/cache effects may contribute to parent front-loading;
do not turn that pattern into a clock/warmup optimization claim.

Read-only GitHub refresh compares antirez/ds4 from pinned9139e2ae58a41503968a500f36f75895c1ba63fc
to8db1d1d155cb0400a86a86b9c62d0defb3a6148b:37 commits. Most changes are CUDA,
multi-session batching or speculative batching, secondary to this serial BS=1 lane.
Do not treat C16 aggregate throughput as local inter-token capability.
Scope: commit-summary screening, not a complete audit of every changed kernel;
the PLE scheduling patch below was read in full.

One new BS=1 mechanism merits recording: donor d0b74340bf1467cfde054f1b1e80dd6f0f8519ca
parallelizes16 on-demand, uncached320-byte BF16 PLE preads, changing the Apple
worker threshold from256 rows to2. The author reports stage1.31->0.25ms and plain
51.3->54.5token/s on its own setup; those are not local M4 measurements. The actual
patch only changes scheduling thresholds/reader count, not model arithmetic.
Local PLE uses mmap-backed IQ4_NL selected rows,16x90=1440 packed bytes, copied
serially to bounded staging, then native dequantization. It is not the donor's
uncached pread path. A small temporary allocation/copy is visible but has no
credible millisecond deletion ceiling. No worker pool, alternate I/O policy,
range warming or further storage experiment is introduced.

Pinned sources: [donor comparison](https://github.com/antirez/ds4/compare/9139e2ae58a41503968a500f36f75895c1ba63fc...8db1d1d155cb0400a86a86b9c62d0defb3a6148b),
[on-demand PLE scheduling patch](https://github.com/antirez/ds4/commit/d0b74340bf1467cfde054f1b1e80dd6f0f8519ca).

Current queue: no justified new local kernel/runtime candidate from these packets.
Source-only screening of genuinely new mechanisms may continue, but must show
local reachability and recoverable work before another experiment. This is not
global optimization exhaustion. Cold-storage/default-prefetch/range-warming,
retired attention, tiny-copy and previous dtype/mixer lanes remain closed.

## Artifacts

- `target/profiles/2026-09-16-native-qsa39-child-ledger.log`: capability failure.
- `target/profiles/qwen4exp-native-child-host-35634/` and
  `target/profiles/2026-09-16-native-qsa39-host-source.log`: exact structural packet,
  inconclusive observer, raw host/source evidence.
- `target/profiles/qwen4exp-validation-ceiling-40224/` and
  `target/profiles/2026-09-16-scalar-validation-cpu-ceiling.log`: CPU-only ceiling.

No new weights, server restart, remote push or weakened safety/quality contract.
