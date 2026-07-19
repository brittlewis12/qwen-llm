# v0.606 Q4_K F16-Activation Floor

Status: **KILL**. Packing the normalized FFN activation to F16 once and reusing
it across gate/up is bit-exact, but the best charged N64 candidate improves the
pair only `1.03419x`, projecting `1.01365x` whole 27B prefill. The required gates
were `1.10x` primitive continuation and `1.05x` charged whole prefill.

The default-off measurement sidecar was removed after the decision.

## Baseline And Ceiling

Production Q4_K N64 on the exact `[5120,17408]` FFN-gate shape is highly stable:

| Query rows | Mean | Nominal throughput | Sample SD |
| ---: | ---: | ---: | ---: |
| 1024 | `13.80553 ms` | `13.22195 TFLOP/s` | `0.00384 ms` |
| 4096 | `55.06145 ms` | `13.26054 TFLOP/s` | `0.00520 ms` |

A timed 27B pp1024 phase trace records:

- `ffn_gate = 884.04 ms`, or 21.902% of traced GPU time;
- `ffn_up = 883.66 ms`, or 21.893%;
- gate+up = `1767.70 ms`, 43.795% of GPU time and 40.754% of the
  `4337.494 ms` whole prefill;
- a charged gate+up candidate therefore needs about `1.133x` to project
  `1.05x` whole prefill.

Straight A-only or A+B ping-pong was rejected before implementation. It raises
dynamic threadgroup memory from 8 KiB to 12/16 KiB and lowers the 32 KiB
threadgroup-memory residency ceiling from four groups/core to two. Current code
also computes the next A dequant tile in registers before its reuse barrier, so
naive double buffering advances only stores and the barrier tail, not all A work.

## Falsifier

The sidecar instead rounded `[N,5120]` F32 activations to row-major F16 once,
using the same half conversion already performed inside every N64 threadgroup.
Two bit-exact candidates then consumed the packed activations:

1. **Staged** retained `sb`, halved its logical source bytes, and removed repeated
   F32-to-F16 conversion while preserving cross-simdgroup B sharing.
2. **Direct** removed `sb` and loaded F16 fragments from device, reducing dynamic
   threadgroup memory to 4 KiB but duplicating B reads across the two M halves.

Both candidates preserve Q4-to-F16 values, activation-half bits, FP32 MMA K
order, grid, and output layout. Real `blk.0.ffn_gate.weight` checks are bit-exact
against production at N64 and N1024 (`max_abs=0`).

## Result

Twenty-sample P1024 actual-shape rows:

| Arm | Single kernel | Single ratio | Charged gate+up | Pair ratio |
| --- | ---: | ---: | ---: | ---: |
| Production | `13.80711 ms` | — | `27.61210 ms` | — |
| Staged F16 | `13.31596 ms` | `1.03688x` | `26.69917 ms` | `1.03419x` |
| Direct F16 | `14.15607 ms` | `0.97507x` | `28.37748 ms` | `0.97304x` |

The F32-to-F16 pack costs only `0.05895-0.05902 ms`; it is not the limiter. The
staged pair saves `0.91293 ms/layer`, or `58.427 ms` across 64 layers. Against
the measured whole prefill this is only `1.01365x`.

## Interpretation

- Repeated activation conversion plus B staging account for only about 3.4% of
  this kernel, despite their large operation count.
- `sb` cross-simdgroup sharing is valuable: deleting it turns the gain into a
  2.7% regression even though dynamic threadgroup memory falls by half.
- The remaining A-store/barrier overlap cannot credibly bridge the gap from
  `1.034x` to the required `1.133x`; A ping-pong is not authorized.
- Production N64 is principally bounded by Q4 dequantization and MMA rather than
  the removable B conversion/staging path. Do not reopen activation packing,
  direct B, or two-slab A staging without a new execution primitive.

Adversarial design review used `cx ask` session
`019f77aa-b0e3-7f11-ae13-1c9d6b45ac26`.
