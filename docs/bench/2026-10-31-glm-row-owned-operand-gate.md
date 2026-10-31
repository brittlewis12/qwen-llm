# GLM Row-Owned Mixed-Input Operand Gate

## Decision And Scope

The exact mixed-input API and bounded M5 operand-delivery gates pass. This funds
source preparation of one native Q4 complete-pair experiment, not a production
selector, timing claim or model-quality campaign.

The rejected [double-buffered K64 loader](2026-10-31-glm-native-loader-ceiling.md)
staged both operands in 32 KiB of LDS. This alternative assigns each of four
SIMD32 cohorts 16 routed rows and all 64 output columns. Activations use a
cooperative Half left operand; weights use a threadgroup Half right operand;
the cooperative accumulator is F32. Weight-only storage is 16 KiB. There is no
activation LDS slab or extra expanded weight bank.

This changes MMA ownership and operand delivery, not merely tile/barrier counts.
Physical register residency, spills and occupancy remain unmeasured.

## Frozen Gate

Source checkpoint `6cef7ad700feda65b9261ac7cc6b25fd962d369d` adds only a standalone
probe shader, validation host and protocol, outside Cargo/production selection.
The exact operation is `matmul2d(16,64,32)` with `execution_simdgroup`, mixed
cooperative/threadgroup inputs and non-relaxed precision. Configured SDK 26.2,
Metal 32023.864 and Metal 4.0 compile and link it without a geometry or precision
fallback.

Each of two pattern cases dispatches one 128-thread group and four ordered K32
operations over K128. Both weight slabs are loaded before computation: this gate
does **not** demonstrate pipelining, overlap, native dequantization or speed.

On M5 Ultra the actual PSO accepts SIMD32, 128 maximum threads, 16,384 static LDS
bytes and zero dynamic LDS against the device's 32,768-byte limit. The host
checks all five physical bindings, aliases, capacity and admission before writes,
then waits for checked completion before reading or dumping results.

Actual explicit fixture pricing is 770,048 Metal bytes plus 5,242,880 CPU bytes,
total 6,012,928 bytes below the 8 MiB explicit cap. A separate 128 MiB CPU-only
framework/control policy reserve is not charged to Metal WSS and is not a
measured driver bound. This is not a continuous process/driver peak measurement.

## Hardware Result And Independent Replay

Both cases pass unchanged validation. Independent CPU replay decodes the retained
guarded physical buffers rather than trusting report booleans or the host oracle:

- All ten guard regions and all 16,384 Half input values per case match the fixed
  recipes; inputs remain unchanged.
- Actual left/result per-lane capacities are 16/32. All 128 SIMD/lane tags occur
  exactly once.
- All 8,192 valid left records per case match the dumped activation Half bits;
  every coordinate occurs exactly once within each step/cohort.
- All 4,096 output coordinates per case occur exactly once and equal scalar F64
  dots calculated from the dumped activation and weight buffers, with zero error.
- No unwritten sentinels, holes, replicated coordinates, malformed unused records
  or nonfinite results occur. No tolerance or validator relaxation is used.

The canonical lease is acquired fail-fast under an external 90-second timeout,
with Metal validation enabled and wired-memory bypass unset. Earlier launcher
refusals occurred before GPU execution and do not supply hardware evidence.

## Next Proof Obligation

A native Q4 adaptation must retain canonical Half coefficient bits, K32 F32
accumulation, separate gate/up projections and the incumbent clamped epilogue.
All threads must still join weight loads/barriers when an empty 16-row cohort
omits MMA. Tails, malformed routes/counts, complete physical preflight, independent
C/N64 references and bounded CPU/Metal fixture pricing remain mandatory.

Only a material complete-operator ceiling funds model conversion. The screening
target remains roughly 5-6 ms per actual-shape pair, not a small source-load
improvement. Actual application affected fraction and real-route conversion are
still unknown. Neither this gate nor the target establishes an application gain.

## Retained Evidence

Private source/build receipts, reports, physical dumps and replay are retained.

| Artifact | SHA256 |
| --- | --- |
| Validation executable | `9b87b766c9145beb5de2e6992d7df5255a30b256137ca8e1b467670a79f938b0` |
| Frozen metallib | `435388cb7c5be16d1ca62985ec72291f8e0ec2caf19895c02f3fcb62c368a0af` |
| Completed hardware report | `bfe3040859e423910001b1679c97b88dbbdc7903f408e8bd4a7616f84f8142a0` |
| Case 0 guarded physical dump | `fccd71939a1cfb15a8a8f4eda5a2c792c0af297d4e6191ad6b66f2575c906a4a` |
| Case 1 guarded physical dump | `d5c616332808b0ad42b333048251e4220cbbaab52453ab9eb982ee79ddf0383b` |
| Independent replay script | `00ce04c2d72ba1f3311d0a14a8622676a1c6e3e7c59576bcb2fa2c5fa7ea0865` |
| Independent replay result | `79a8ab53f348b066dd451b13acae464503c52f7b5685359c2b886ac64785a832` |
