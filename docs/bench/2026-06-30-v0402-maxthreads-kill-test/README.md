# v0.402 MoE `max_total_threads` Kill-Test

Goal: test the external-audit hypothesis that hot 64-thread MoE kernels may be
register-constrained because they lack Metal
`[[max_total_threads_per_threadgroup(64)]]` annotations.

Probe: temporarily annotated these exact kernels, then built and measured the
dirty tree. The patch was not kept.

- `kernel_moe_swiglu_q4_K_f32`
- `kernel_moe_swiglu_q4_K_f32_packed_slots`
- `kernel_moe_down_weighted_sum_q5_K_f32_packed_slots`
- `kernel_moe_down_weighted_sum_q5_K_f32_packed_slots_k512_r2`

Validation:

- `cargo check -p qwen-cli --bin qwen-bench`
- `cargo build --release -p qwen-cli --bin qwen-bench`
- A3B/A10B independent-file `moe-batch-sweep` at `ctx512`
- A3B/A10B `qwen-bench phase --ctx 512` with MoE FFN split
- `cx ask` probe ranking, session `019f1bcd-144e-73b0-9f07-b6008d46c1d4`

Artifacts:

- `target/profiles/v0402-dirty-a3b-independent8-ctx512-maxthreads.out`
- `target/profiles/v0402-dirty-a10b-independent8-ctx512-maxthreads.out`
- `target/profiles/v0402-dirty-a3b-ctx512-phase-maxthreads.out`
- `target/profiles/v0402-dirty-a10b-ctx512-phase-maxthreads.out`

## Results

Compared against the immediately prior v0.400 exact independent-file rows.

| Model | Slots | v0.400 exact | Dirty annotated | Read |
| --- | ---: | ---: | ---: | --- |
| A3B Q4 | 1 | `1.8244` | `1.8279` | flat/slower |
| A3B Q4 | 4 | `1.2968` | `1.2974` | flat |
| A3B Q4 | 8 | `1.2211` | `1.2201` | flat |
| A10B Q4_XL | 1 | `5.7233` | `5.7328` | flat/slower |
| A10B Q4_XL | 4 | `4.8401` | `4.7915` | small batch-only win |
| A10B Q4_XL | 8 | `5.1352` | `4.9861` | small batch-only win |

A3B phase at `ctx512` stays flat: `phase_sum 9.32 -> 9.34 ms`, gate/up
`1.20 -> 1.20 ms`, down `0.86 -> 0.86 ms`. A10B dirty phase shows no obvious
single-token discontinuity: gate/up `3.69 ms`, down `2.61 ms`, with no paired
clean phase win large enough to justify keeping the patch.

## Decision

Kill this annotation probe. It does not move the single-token A3B rows and does
not clear the `2-3%` captured micro gate on the primary single-token path. The
A10B batch-only improvement is not enough to reopen batching after v0.400/v0.401:
`b8` remains slower than `b2`, and single-token decode is flat.

Do not blanket-add `max_total_threads_per_threadgroup` annotations without a
fresh exact-kernel micro win or manual shader-profiler evidence of register
pressure. Keep future annotation tests scoped to one named active kernel with a
paired micro gate.
