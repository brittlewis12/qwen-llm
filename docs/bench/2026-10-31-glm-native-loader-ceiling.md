# GLM Native Loader Ceiling: K64 Rejected

## Decision

Do not promote the cooperative, double-buffered native Q4 K64 loader. It passes
the coefficient and complete-operator witnesses on M5 Ultra, but loses to both
the maintained routed TensorOps path (C) and the simpler N64 reference. No model
adapter, expanded weight cache, metadata sidecar or quality campaign is funded
by this result. The qualified C production default is unchanged.

This rejects one delivery design, not native GGUF execution. All three tested
paths already consume native compressed Q4 weights and reconstruct Half operands
on the GPU; none stores an expanded expert-weight bank.

## Frozen Experiment

- Source: `fe7256421656e6d9d8fbc10a859dc5e62d5b6d69`, based on N64 checkpoint
  `4ace43502db2236927252eac94c848bbfcab43c7`.
- Canonical release libtest SHA256:
  `d49bf78458ead83c61cf0565993b1ee874aceccaef0e1bc3d7d1633e21c2a4f8`.
- Ordinary Fat LTO, one codegen unit, opt-level 3; SDK 26.2, Metal 32023.864,
  Metal 4.0. No profile, toolchain or deployment-target overrides.
- Actual device: M5 Ultra. Validation uses the Metal debug layer; timing disables
  it. Each campaign acquires the canonical lease fail-fast under an external
  timeout and releases it on completion. No service or model is loaded.

The new tile computes 64 outputs by 64 routed rows with K64 delivery and 128
threads. Paired loaders fetch both native nibble planes, reuse superblock
metadata, and prefetch into registers. Ping-pong operand slabs reduce four loop
barriers per K64 to one while retaining two K32 MMA calls and F32 accumulation.
Gate and up remain separate dispatches with the incumbent clamped epilogue.

Observed pipeline capacity accepts SIMD32, 128 threads and zero static
threadgroup bytes for all arms. Dynamic threadgroup storage is 8 KiB for C,
16 KiB for N64 and 32 KiB for the cooperative candidate. All report a 1024-thread
pipeline maximum. These facts do not measure registers, occupancy or the reason
for the regression.

## Correctness And Admission

All six exact ignored GPU entries pass: canonical coefficient bits; inherited
N64 composition and preflight witnesses; cooperative pair bits/guards/preflight;
small-pair timing; and the separately admitted full-shape timing fixture.

Coefficient checks include native block boundaries, both nibble planes, packed
scale/min extremes, signed/zero/subnormal/large scales, row tails, three banks
and offsets 0/16/144/160. Complete pair checks compare every finite F32 output
bit across C, N64 and the candidate, plus bounded Half-operand F64 dot samples,
clamps, immutable inputs, guards, dispatch geometry and preflight non-mutation.
The malformed-route/count controls also pass. There is no tolerance relaxation.

The full fixture is H4096/F2048/E288/top8/R2048. Live device pricing admits
3,173,744,640 GPU bytes, 1,762,112,000 CPU/upload/readback/report bytes and a
536,870,912-byte reserve: 5,472,727,552 bytes, about 5.097 GiB. Native banks are
uploaded sequentially; there is no full expanded oracle. This is an explicit
fixture quote, not a measured process/driver high-water bound.

## Complete-Pair Timing

Each comparison records four warmups followed by eight uninterrupted ABBA/BAAB
samples. Timed work includes preflight, F32-to-Half conversion, route bucketing,
both projections and checked completion. Oracles, hashing and JSON are outside
the continuous sample block. All samples are retained.

Full-shape GPU medians, milliseconds:

| Synthetic routes | Reference | Reference ms | K64 ms | K64 latency change |
| --- | --- | ---: | ---: | ---: |
| Balanced | C | 10.373 | 11.095 | +6.96% |
| Balanced | N64 | 8.015 | 11.214 | +39.90% |
| Skewed | C | 11.145 | 14.171 | +27.15% |
| Skewed | N64 | 9.806 | 14.131 | +44.12% |

Balanced C controls improve 4.15% between the two four-sample halves, so its
pooled +6.96% is not a stable effect-size estimate. The candidate still loses
in both halves (+4.55% and +8.91%). All other full-shape GPU control drift is at
most 1.70%, and each comparison's two halves agree on the negative decision.
Host medians also lose: +6.67%/+38.32% balanced and +26.64%/+43.29% skewed.

At the smaller H4096/F256/E16/R2048 shape, K64 loses about 16.7-18.3% to C and
15.5-16.1% to N64 in GPU time. The two E16 route recipes yield the same count
histogram, so they are not independent skew evidence. Those native banks are
only 18 MiB per pair; the full 2.53125-GiB pair was therefore tested separately
rather than treating the smaller case as proof of streaming behavior.

Balanced full-shape routes issue 576 N32 versus 288 N64 panels; skewed routes
issue 632 versus 400. Even with fewer panels, less duplicate codec work and
fewer source barriers, this K64 design is slower. Source-level load counts are
not ISA issue counts or DRAM traffic. Doubled LDS and prefetched live state are
plausible costs, not measured attribution.

## Remaining Leverage

The simpler N64 reference has potential operator headroom, but C and N64 were
each counterbalanced against K64 in separate blocks, not directly against each
other. Their difference is not an application speedup. Multiplying an apparent
2-3 ms pair difference by 41 eligible Q4 layers suggests only order 80-120 ms
against a roughly 1.98-second 2K application baseline; real route counts,
in-stream overlap and actual weights remain unmeasured here.

Any further model falsifier must stay short and independently budgeted. A new
structural loader should remove an evidenced operand-delivery or occupancy
constraint, not initiate another barrier/tile sweep. For an application affected
fraction f and operator speedup s, savings are f*(1-1/s); neither f nor a useful
compiled native-codec ceiling is established by these synthetic measurements.

Routes and compact weight recipes are synthetic, not captured model traffic.
No DRAM, cache-miss, register-count, whole-model, quality or production-selection
claim follows. Existing native decoders, down projections and V4 are unchanged.

## Retained Evidence

Private build receipts, raw reports, logs and the independent analysis script are
retained. Public report SHA256 identifiers:

| Report | SHA256 |
| --- | --- |
| Canonical Half bits | `97c127bff90b9954804f9ca6e543d78332725729f654390db36ade134dc77f37` |
| Pair bits/guards/preflight | `7e1ac004a90ef5eadf329f2b69829a55633ca384e0d3a39096054e65de648876` |
| Small pair timing | `335bb72488b131781a3df4a27c81d0d01c3642fb4858c54d8290b0130b871a69` |
| Full-shape pair timing | `025e6410183331a06a5cc6e77beb7fd44741d3bb988645f0da81f7b7256c5a5a` |
