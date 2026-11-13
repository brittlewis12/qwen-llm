# V4.1 cooperative experts: default-off preparation

The M32/M64 prototype from `03bbba12c`, with the scoped device policy from
`dff2b125c`, is replayed on the attention-qualified base. It stays on the
separate `perf/v41-cooperative-prepared` branch and is not part of the
attention landing. `QWEN_V41_EXPERT_COOPERATIVE` remains default-off. The
benchmark enum keeps all current attention controls plus the cooperative
experiment; two source conflicts were resolved without restoring older
standalone PV controls. The complete mlx-serve MIT notice in
`kernels/tensor/v41_sorted_experts.metalh` remains intact.

Eight SIMD groups / 256 threads share M32 or M64 GU N64 and down N128, K64
updates and the existing native-IQ dequantizer. <=32 live rows choose M32.
The 64-row run plan and 18,432-byte shared operand allocation remain. No
expanded bank, requantization or persistent GPU allocation is introduced.
The same half operands, volatile F32 activation and weighted rounding points,
and F32 down result are retained. No physical register-allocation or speed
claim follows from the source layout.

## Differential and compile checks

The existing GPU differential is extended in place; no new ignored test or
test name is added. Compact H/F=256 covers all four supported IQ formats and
same/mixed down formats. A header-only GGUF read confirms real H=5120,
expert F=2304, 384 experts, top-6; those K/N lengths are additionally tested
with IQ2_XS GU and IQ2_XS/IQ2_S down.

Rows 64/128/129 with top-2 and eight synthetic experts exercise means
16/32 rows per expert, uniform and skewed routes, empty runs, 31/32/33,
63/64/65 boundaries, and 129-row hot runs with a one-row final panel.
Both modes preserve packing, plan, dead-scratch alias and guard assertions.
There are 90 cases per mode and 73,048,064 strict inner/down components in
total, including actual-dimension cases; GPU execution must confirm these
counts. This is functional coverage, not a model routing histogram or speed
measurement. The later stage gate uses actual 384-expert populations.

Local iter census type-check and test-profile no-run compile pass. Required
M5 execution uses the existing exact differential with MTL debug and Metal
required, plus existing binding-refusal coverage. Until its result is
recorded below, no functional/exactness claim is made.

## Predeclared next stage gate (no isolation run in this stage)

1. Functional/exactness first: all strict components, guards, aliases and
   packing assertions pass; no tolerance is introduced. Then the full
   paired state/logit/following-step signatures must be bitwise on model
   cells before any wall claim. A failure cannot be treated as an isolation
   win. No approximate or non-inferior claim is inferred from compilation.
2. Use the qualified attention path in both arms and one iter binary with
   only the cooperative selector changed. Capture real bank formats and
   row-count histograms for 2048 rows at 48K and 1024 at 80K: means are
   2048*6/384=32 and 1024*6/384=16. Retain min/median/p95/max, empty counts,
   and M32/M64/full/tail-panel counts rather than replacing the routes with
   uniform synthetic GEMMs. Include a cold-root CED dispatch witness to
   confirm the same expert geometry.
3. Warmup plus three retained interleaved pairs per cell. Require >=1.25x
   combined GU+down GPU isolation at each real population; report GU and
   down separately. Baseline cooperative dispatch count must be zero; every
   eligible candidate call must use the cooperative family, with no overflow
   or hidden fallback. Dispatch profiling is not a native wall claim.
4. Report the inclusive expert-input conversion + plan/pack + GU/down chain
   alongside math time. A representation variant must include all of its
   preparation work. Only after isolation passes proceed to native paired
   root/append/PP and served gates under separately frozen rules. No such
   performance or promotion run is authorized for this bounded stage.

## Expert-input movement design

The waste audit prices all `scatter_offset_f32_to_f16` plus
`v41_sorted_pack` at 39.1577+32.2980=71.4557 ms GPU per 2048-row chunk,
and 19.2433+15.0799=34.3232 ms for the 1K append. The 280 scatter calls
include indexer and other consumers; **the full 71 ms is an upper bound, not
an expert-only saving**. Map the expert-input calls by bindings and stage
before attributing a removed dispatch.

The current cooperative GU already consumes packed F16 directly, and its
weighted Half output directly feeds down. It retains both upstream movement
passes. No removal of those passes is claimed by this prototype.

A follow-on direct-input variant can gather the original normalized F32
through existing expert IDs/plan into a shared Half A tile, explicitly
retaining the current float-to-half rounding. TensorOps then consumes that
shared tile with the same K64 update order. M64 needs at most another
64*72*2=9216 bytes, for 27,648 bytes with the current weight slabs; M32 needs
4608. Guard total resources against the actual device and pipeline limits,
and measure occupancy rather than assuming the extra scratch is free.
There is no expanded bank or persistent allocation, preserving the three
48K-session memory requirement. A shared-input TensorOps path may change
reduction order, so it requires the full strict differential again.

This can eliminate sorted-pack traffic and only the expert-owned half
conversion that becomes dead. If `input_half` is already produced for a
shared or indexer consumer, that conversion remains charged to that other
consumer and is not counted as removed. The source F32 buffer must stay live
and disjoint through GU; down's permitted alias with packed scratch stays
unchanged. The fallback is the current packed-F16 path.

A smaller alternative is F32-to-Half conversion inside the existing pack
pass. That removes a separate expert conversion only where it is actually
owned, and reads F32 for each route rather than F16, so it can increase read
traffic. Benchmark it, do not infer a gain from one fewer pass. Normalization
cannot simply write expert-major output: router IDs depend on that normalized
input and are not available at its producer. The gather variant respects
that dependency.

[Waste-audit evidence](2026-11-13-waste-audit.md). No direct-F32 path is
implemented or timed in this stage; its exactness, movement and inclusive
GPU-time gates remain separate from the current cooperative tile screen.
