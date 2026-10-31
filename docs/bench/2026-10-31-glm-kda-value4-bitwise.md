# GLM KDA Value4: Fixed M5 Bitwise Witness

## Result And Scope

The fixed four-value-row-per-SIMD-group candidate matches production CURRENT
bitwise on one rich synthetic M5 DEBUG1 case: all **16,777,216 output words** and
**1,048,576 final-state words** agree exactly, including signed zeros. Both arms
are entirely finite. Independent replay accepts the unchanged actual capture.

This observes no division-versus-reciprocal mismatch for this case. It does NOT
establish universal arithmetic equivalence, actual prepared model/nonempty-prefix
or chunk-carry correctness, timing, speedup, default eligibility or model quality.
No performance mode is implemented in the witness.

Native source is `3dfd034e850b5aa6a1ebd7c4384ed3cafcdae7cd` on experimental
`probe/glm-kda-value4`. Checker-only amendment `66f7604d514a24d62124ab5a41b9f5f2a5a0608e`
does not change the native binary, candidate shader, inputs or numerical rules.
Production code remains unchanged.

## Why Actual Bits Matter

The candidate reuses q/k/decay/beta across four independent value rows instead
of one per SIMD group. Each row retains the written decay, dot/simd_sum, delta,
FMA, readout and final-store sequence. Lane-owned state increases from one
float4 to four; grid changes from `[64,32,1]` to `[64,8,1]`, both 128 threads.
Logical q/k/decay loads fall from 24 GiB to 6 GiB, not measured DRAM traffic.
State was already register-resident: initial/final state traffic stays 8 MiB.

However, normal O3/fast-math AIR lowers the isolated unchanged control's delta
to direct division, while value4 shares a reciprocal and multiplies four deltas.
A normal IEEE CPU witness differs by one bit. That proves the forms CAN differ,
not that the actual M5 kernels must differ. Source syntax and CPU row-grouping
tests alone therefore cannot clear GPU bitwise equality.

The real baseline here is `MetalContext::pipeline` from the EXISTING production
PRIMARY library, not the isolated control metallib used for AIR inspection.
The exact 6,726,818 PRIMARY bytes are verified inside the pinned production rlib
and witness executable. The private prepared-stage method is not assumed public:
this explicitly scoped GPU-API witness uses source-verified direct Args16/bindings
and grids, not a claimed public whole-KDA operator call or native application test.

## Fixed Input And Commands

The single fixed represented F32 dataset has 2048 rows, 64 heads and 128
key/value dimensions. Q/K are normalized, V is bounded, decay is nonuniform and
below one, and incoming state is nonzero/asymmetric with head/value/key tags.
Raw beta has 457 distinct values, including 19,186 exact 0.125 entries near the
CPU witness region. Independent pre-GPU review reconstructs all 273,158,160
asset bytes and checks every Q/K/V/decay vector. This is not model data or a
GPU dot/exp emulator.

Exactly two checked commands run, CURRENT then value4. EACH command restores
the SAME immutable 4 MiB GPU incoming state by tracked BLIT, ends the blit, then
dispatches one serial compute kernel. Nine guarded physical buffers are captured
and synced before work-state/output reuse. There is no warmup, repeat, timing,
unmatched evolving state or free reset outside a claimed timer.

| Actual Comparison | Words | Bit Differences | Nonfinite In Either Arm |
| --- | ---: | ---: | ---: |
| Output | 16777216 | 0 | 0 |
| Final state | 1048576 | 0 | 0 |

The complete 344,530,960-byte raw files are byte-identical, not just their
outputs' hashes. Independent standard-library parsing checks every output/state
word, 147,456 guard bytes and 546,308,128 immutable input/frozen-state bytes.
Exact signed-zero comparison remains in force; matching NaNs/Infs/poison fail.
The source/CPU recipe review, authored offline replay and independent actual
byte replay remain distinct evidence.

## Checker Interface Amendment

The first offline invocation stops before raw comparison because the frozen Rust
collector serializes its capture filenames as `{"Unix":[u8,...]}`, while the
original Python checker incorrectly expects strings. The failed result is retained
as an interface refusal, not an arithmetic failure.

The checker-only fix validates exactly that Darwin wire union: one Unix key,
bounded actual integer bytes, matching the independently constructed fixed arm
basename. Strings, bool bytes, other variants, wrong arms and path substitutions
reject. It opens only the constructed expected path. Numerical, finiteness,
full-recipe, parent-proof, guard and bitwise rules do not change.

All twenty CPU tests pass after the amendment, including literal actual filename
unions and prior numerical negative controls. The original source packet, failed
result, binary, libraries, asset and raw pair remain unchanged. Replaying that
same capture through the corrected checker passes; independent byte parsing and
a separate reviewer rerun agree. There is no GPU rerun or tolerance relaxation.

## Execution And Resource Evidence

The DEBUG1 process exits in five seconds under the external 90-second bound and
20-second TERM-to-KILL grace. Normal WAIT0 self-leasing ends naturally; post-exit
inspection finds no openers. All math acceptance runs after process/lease end.
Those supervisor observations are trusted parent assertions, not attestation.

Both PSOs are SIMD32/static0/dynamic0 and launch 128 threads. CURRENT reports
max1024, candidate max128; these fields are not register or occupancy evidence.
Actual device prices total B344604672 and bound buffer growth344588288.
Separate Metal policy100663296, CPU explicit1062354976, shared R33554432 once
and separate CPU framework134217728 give quoted demand1,675,395,104 bytes,
below the fresh 2 GiB cap. Asset/map pages and both raw-file caches are charged;
there is no incumbent credit, old-cap increase or formal peak claim.

The configured opt1/noLTO collector links cached rlib4d30; all 223 cached files
and five original candidate files remain unchanged. Candidate MSL3.1/O3/fast-math
flags and geometry are unchanged. No application build, model, endpoint,
installation, PATH, SDK selection or production-default change occurs.

## Frozen Evidence

| Evidence | SHA256 |
| --- | --- |
| Native witness | `2f16def97eed7e859f4eecfc3f17274c2984dc60c458f2b16c0d2d92888532ad` |
| Candidate library | `e7380f68c8a94f4ff5d463c37b3f55aff6ff4956bacad32de50ffc799a62fc0c` |
| Synthetic input asset | `c70f4453078a07148cad7926a3fca487bbc31c31bb8131641b09e28578822461` |
| Actual report | `cbab6d9b64a9df03ee59e743972eb00a1a91f45b1c9d333baac7626c5e34d16b` |
| Each complete raw arm | `6e2740d280036c5b2f7f46924a464579273529ecaea1c9278820fb00790fd22d` |
| Amended checker | `0cf084e4725433a42662bcc01f01db7d06a038dcdeadcd5477fa0909512253d4` |
| Authored actual replay | `cc03ba0b4e6700ee02a2596cf56517247d0fda6ad44d85b5aadb2442766d85e1` |
| Independent actual replay/rerun | `0406ec17375f39c2ef7bac47fa648df10eb0da6dfe3b0703322b9334ae4707da` |

Next is a separately reviewed actual-prepared-input capture with nonempty prefix
and continuation proof, not automatic timing. The future recurrence screen still
requires >=30% repeatable improvement and reset-inclusive composition qualification
before any integration. This one synthetic exact match waives none of those gates.
