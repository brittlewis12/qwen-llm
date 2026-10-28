# Vector4 Weight-To-Half: Exact Operator Continuation Signal

The vector4 research producer passes its exactness proof and all four
predeclared 1K/4K operator-economics gates. This is a continuation signal for
whole-FFN qualification, not a production default or model-throughput result.
The earlier scalar negative remains closed and unchanged.

## Source And Contract

- M5 Ultra, native TensorOps tier, release Rust 1.99.1 and SDK 27.0; existing
  `-O3 -ffast-math` Metal build policy.
- Source `a0e60724e027642e013b88cb3d6f1f66b4e7448b`, tree
  `dbd82710cbd5da1e173ffc422c7256483cf9debf`.
- Both exactness and economics use frozen executable SHA-256
  `9cac53ec5445e97cb38c3115e87bae44da34405aaea2fb7bf52646679e2d8c9b`.
  Each phase independently passes all 344 classified CPU checks.

One thread handles four adjacent channels, shares one status and route-weight
lookup, loads an aligned `float4`, computes F32 products and stores `half4`.
It requires width divisible by four, actual input alignment of 16 bytes and
output alignment of eight bytes. Unsupported layouts refuse before pipeline
lookup rather than falling back silently. The padded-thread guard precedes
`tid * 4`, including near the uint limit. Partial quads use valid-only scalar
accesses; every non-READY output is positive zero, not a NaN-times-zero mask.

The original scalar shader stays unchanged. This is a new implementation
hypothesis, not a rerun of that scalar variant or a claimed explanation of its
two disagreeing 1K blocks.

## Exactness

The unchanged production F32 weight kernel plus `HalfScratch::convert` is the
GPU oracle. Ten fixed cohorts compare 141,560,517 raw half words, with all
input/status integrity, offsets, guards and spare capacity checked.

Coverage includes both F32/half double-rounding witnesses, signed zero,
subnormals, half overflow and negative weights. READY and non-READY one- and
two-lane tails both run; the large minus-one cases retain READY three-lane
tails. Review adds the two missing READY-tail cohorts before the first GPU
attempt, without removing any earlier case. The READY tail operands include
`65504*1 -> 0x7bff` and `65520*1 -> 0x7c00`.

All ten cohorts pass. The selector reports 0.54 s, which is correctness
execution time, not performance evidence. Largest priced buffers plus CPU
reserve are 679,788,544 bytes; fixture admission retains its 768 MiB cap and
1 GiB device-plus-admission bound. This finite proof does not establish every
floating-point input or compiler/device combination.

## Separate Economics

Each geometry uses finite nonzero synthetic activations, all READY statuses
and positive top-6 weights summing to 1.5. Shapes are 1,024 and 4,096 rows,
width 2,304. Neither is a model-routing fixture.

A runs the unchanged F32 producer and half converter. B runs vector4. Both
write the same physical half-output range and retain the same F32 scratch
allocation. A GPU oracle freezes before B; every warm and primary call checks
all half bits, inputs, status, guards and capacity. A checks its F32 products;
B must leave the entire unused F32 temporary poisoned.

One oracle, warm A/B and primary `A B B A`, `A B B A` run per geometry. Warm
census checks two A dispatches versus one vector4 B dispatch, including B's
quarter-sized logical thread grid. Primaries have no census or profiler.
Allocation, reset, copies, verification and reporting are outside both timed
intervals. All 22 command envelopes are serial and valid. A fixed 5 us
cross-API allowance does not relax the comparative gate.

Every block at both shapes must reduce mean producer GPU latency by at least
10%, with no mean inclusive-host regression. The independent parser uses the
same predeclared estimator, `B_mean_gpu <= 0.9 * A_mean_gpu`, without filtering
samples or inferring a win from libtest success.

| Rows | Block | A GPU Mean, us | B GPU Mean, us | GPU Reduction | A Host Mean, us | B Host Mean, us |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 1 | 527.417 | 378.979 | 28.1442% | 1827.146 | 1619.687 |
| 1,024 | 2 | 580.104 | 413.146 | 28.7807% | 2214.938 | 1979.229 |
| 4,096 | 1 | 2692.292 | 1307.500 | 51.4354% | 4186.209 | 2720.167 |
| 4,096 | 2 | 2668.687 | 1168.458 | 56.2160% | 3927.813 | 2701.355 |

All four blocks pass both conditions. The timing selector reports 1.16 s;
the raw samples and independently recomputed decisions remain retained.
The frozen vector proof is revalidated before and after economics, and both
phases use the identical executable. The economics receipt explicitly allows
only a later whole-FFN gate, with production promotion and engine-gain claims
both false.

## Limits And Next Gate

Build, freeze and CPU acceptance hold no GPU lease. Each GPU packet uses one
continuously inherited canonical lease, a 30 s child cancellation trigger and
45 s outer acceptance limit. Safe drain can exceed those limits while retaining
the descriptor; late work cannot qualify. Completed adversarial review and
source/config/binary/CPU/helper bindings are checked before queueing, across the
queue wait, before native execution and after postflight.

Removing the temporary's write/read would eliminate `8 * N` logical bytes and
one dispatch in an eligible integration: 432 MiB at 4K. Both experiment arms
retain that allocation. No measured DRAM, allocation or end-to-end speedup
follows from these numbers, and operator time savings cannot be added to
nonexclusive production stage envelopes.

The next qualification is a fixed actual-routing 1K whole-FFN replay. Its
reference must explicitly distinguish the deliberately elided F32 temporary
from every materialized stage, final half backing, status and output. Normal
production dispatch and defaults remain unchanged pending that work and later
model-level checks.

Scalar evidence: `docs/bench/2026-10-28-v41-weight-half-producer.md`.
