# DeepSeek V4 Packed Q8 F32 Matrix Falsifier

Status: model-free `GO`. One fixed F32 `R2C4K64` Q8 matrix geometry removes
both F16 operand conversions from the killed packed attention-output schedule.
It clears every frozen numerical, resource, stationarity, median, p95, wall,
and absolute-time gate on the qualified Apple M4 Max.

At this checkpoint, the packet authorized only drafting and statically reviewing
one canonical position-zero 128+12 current-asset gate. That gate subsequently
received a separate static GO and produced a decisive full-model `KILL`; see
`../2026-08-06-dsv4-packed-f32-q8-output-kill/README.md`. The later result closes
the live output family while preserving this model-free arithmetic evidence.

## Question

The prior Q8 matrix schedule reduced the synthetic N=128 output stage from
8.912 to 2.181 ms, but it rounded both reconstructed Q8 weights and F32
activations through F16. Its small local A+B error accumulated through 43
layers: position-140 hidden relative RMS reached 0.070735848 and consumed MoE
routes changed.

The reopen condition required materially different arithmetic that preserved
more activation information, repaired the local error mechanism by at least
fourfold, and retained enough of the attributed output-stage ceiling to support
the frozen 20% first-chunk pre-expert target.

## Fixed Candidate

The candidate uses exactly one geometry:

- one 32-thread SIMD group per threadgroup;
- 16 output rows (`R2`) by 32 activation rows (`C4`);
- K-step 64;
- eight `simdgroup_float8x8` accumulators;
- 4 KiB of threadgroup memory for a 16x64 F32 weight tile; and
- grid `ceil(N/32) x (M/16)` for `1 <= N <= 128`.

For every Q8 block, the kernel reads the stored F16 scale and signed int8
quants, then materializes each weight as exactly one F32
`float(scale) * float(quant)` expression. Activations load directly as F32.
There are no half matrix operands or `fast::` operations. K tiles and eight-wide
subtiles execute in ascending order; output matrices publish through a guarded
threadgroup scatter.

The arithmetic remains numerical versus deployed Q8 GEMV because the F32
matrix reduction order differs from block-scaled scalar GEMV. It is not an
exact-output candidate.

The packed output topology remains unchanged: eight group
pack/projection/scatter chains for output A and one output-B projection. Exact,
killed half/half, and F32 A-only/B-only/A+B policies exist only under tests.
Ordinary Rust has no candidate route, API, allocation, switch, or fallback
change. The dormant Metal symbol is compiled into the metallib.

## Contracts

Host preflight requires Q8_0 weight storage, exact aligned F32 `[K,N]` and
`[M,N]` tensors, `K % 64 == 0`, `M % 16 == 0`, bounded N, writable output,
physical input backing through `ceil(N/32)`, no output overlap with either the
weight or complete padded-input footprint, a serial encoder, 4 KiB TGM,
execution width 32, and at least 32 threads.

Active reduced tests cover K=64/128, M=16/32, and N=1/31/32/33/128. Fixtures
include signed zero, F32 subnormals, half-rounding boundaries, ordinary finite
activations, Q8 quants -128/-127/0/126/127, alternating signs, and F16 scale
subnormal/normal boundaries. Every candidate output is finite, repeated runs
are bit-identical, error is lower than the killed half/half matrix, and all
guards survive.

Negative tests reject unpadded tail backing, logical input/output overlap,
output wholly inside padded input backing, malformed input shape, and wrong
weight dtype before command submission.

## Frozen Gate

The production fixture reuses the exact synthetic weights and activations from
the prior Q8 KILL packet:

- output A: Q8_0 `[4096,8192]`, interpreted as eight `[4096,1024]` groups;
- output B: Q8_0 `[8192,4096]`;
- attention: F32 `[32768,128]`; and
- low rank/output: F32 `[8192,128]` and `[4096,128]`.

Numerical qualification compares deployed exact GEMV, the killed half/half
matrix, and the F32 candidate for A-only, B-only, and A+B. Every final vector,
plus A low rank, must improve cosine deficit, relative RMS, and maximum absolute
error by at least fourfold. Additional caps are:

- A low-rank relative RMS at most 0.00020 and max absolute at most 0.001;
- A+B final cosine at least 0.9999998;
- A+B final relative RMS at most 0.00030; and
- A+B final max absolute at most 0.008.

Timing creates no fixture buffers, pipelines, first touches, or readbacks in the
measured interval. After five alternating untimed warms per arm, it records 24
exact controls, 24 F32 A+B candidates, and 24 trailing exact controls. Control
and candidate-half median drift must each remain within 5%. Against the faster
control, candidate gates are:

- GPU median saving at least 58%;
- GPU p95 saving at least 55%;
- wall median saving at least 55%;
- wall p95 saving at least 50%; and
- absolute candidate GPU median at most 3.75 ms.

The 58% GPU gate follows from the prior 377.528 ms output attribution: saving
20% of the faster 1,084.159 ms first-chunk pre-expert control requires 57.4% of
that family.

## Result

The F32 candidate repairs the local error by far more than the required 4x:

| Arm/vector | Half/half rel RMS | F32 rel RMS | Repair | Half/half max abs | F32 max abs |
|---|---:|---:|---:|---:|---:|
| A low rank | 0.000660414 | 0.000001241 | 531.9x | 0.003462195 | 0.000013828 |
| A-only output | 0.000894324 | 0.000005257 | 170.1x | 0.023438454 | 0.000122070 |
| B-only output | 0.000840433 | 0.000001992 | 421.9x | 0.024706840 | 0.000103951 |
| A+B output | 0.001230151 | 0.000005573 | 220.7x | 0.032827854 | 0.000164986 |

A+B cosine is `0.9999999999843242`. All A-only, B-only, and A+B F32
outputs repeat bit-for-bit. Exact and candidate paths each execute one serial
encoder and 25 dispatches; this result is arithmetic acceleration, not dispatch
collapse.

| Timing metric | Result | Gate |
|---|---:|---:|
| Candidate GPU median | 2.485250 ms | <=3.75 ms |
| GPU median saving | 72.2179% | >=58% |
| GPU p95 saving | 72.2189% | >=55% |
| Wall median saving | 70.8481% | >=55% |
| Wall p95 saving | 70.5607% | >=50% |
| GPU control drift | 0.0936% | <=5% |
| Wall control drift | 0.0922% | <=5% |
| GPU candidate drift | 0.0252% | <=5% |
| Wall candidate drift | 0.1676% | <=5% |

Exact and candidate output bits remain stable before and after timing. Attention,
low-rank, final-output, group-input, and group-output guards all survive.

## Decision

Promote only the model-free arithmetic and ceiling result. Preserve the fixed
test-only kernel, exact reduced contracts, and sealed profiler. Do not retile,
default-enable, or infer model quality from synthetic vectors.

The result authorizes drafting and static review of one canonical current-asset
packet: position-zero matrix A+B for N=128, exact N=12 tail, positions 128 and
140, canonical snapshot restore, one singleton continuation, exact packed and
continuation route transcripts, the prior cosine/RMS/argmax/state gates, and the
frozen first-chunk performance floor. Running that packet requires a separate
reviewed checkpoint.

## Provenance

- Base revision: `263d1e5852384a51dcce2d371c45fe83ec3c3f00`.
- Device: Apple M4 Max.
- OS: macOS 15.6.1 (24G90).
- Rust/Cargo: 1.97.1.
- Xcode: 26.2 (17C52).
- Metal: Apple metal 32023.864, target `air64-apple-darwin24.6.0`.
- Executed `prefill.rs` SHA-256:
  `d7ab61e513ea5e1e8eea854d870672a3cb932f149880b03f4c21a7549365eee5`.
- Executed and final `mat_mat_q8_0.metal` SHA-256:
  `8e01e038765bb333ffdb46249c2ff352ae73dc3b62f14487aba7ba059f5983ca`.
- Raw `profile.log` SHA-256:
  `8f0344efc45e04bcb69e615324f36673bf192131cca05c15e63aa0652a89115f`.
- Validation-complete `prefill.rs` SHA-256:
  `0027c9d44afad0c1bec3f392cdb05b433060dd4758fe1716f12748fc074b1df0`.
- Validation-complete release test binary SHA-256:
  `4579004bcfd42654bdc8ab2f2fbdc5bce79e0fa39164ccd89efb4e0c8ffd79b3`.

The executed test-binary hash was not captured before a post-run host-validation
repair rebuilt the same target. That provenance gap is explicit: the archived
packet records the reconstructed executed Rust hash and unchanged Metal hash,
while the binary hash above binds final validation-complete source.

The post-run delta does not alter the kernel, valid fixture, encoded arguments,
dispatch geometry, or timing calculation. It replaces element-count checks with
exact aligned tensor validation, extends overlap rejection through padded input
backing, adds two fail-before-dispatch tests, and seals the ignored profiler.
The frozen profiler is not rerun.

CX review session: `019fcf7d-e9d4-7150-b496-e70a31958e80`.
