# Muse Packed-Q8 Sidecar KILL

Decision: **KILL** the block-major, address-only C16/Q128/K64 Q8 VJP
candidate. Packing Q8 records by input block preserves exact output but yields
only a low-single-digit gain, even when pack cost is excluded.

## Predeclared Gate

The private candidate copied each complete 34-byte Q8_0 record from
`[output_row,input_block]` to `[input_block,output_row]`. Its VJP retained the
incumbent C16/Q128/K64 arithmetic and accumulation order and changed only the
record address.

At each released FFN direction and `n_query=512`, a candidate train packed into
a preallocated sidecar and executed eight packed VJPs; its paired control
executed eight incumbent row-major VJPs. Both used one serial encoder and the
same enclosing command-GPU clock. Five pairs alternated arm order after warmup.
Promotion required complete output equality and at least 25% median command-GPU
time saving on both directions. A packed-only diagnostic priced the free-pack
ceiling. The invocation was externally capped at 180 seconds and loaded no
model asset.

## Result

Gate/up stopped the packet:

| Arm | Samples | Median |
| --- | --- | ---: |
| pack + packed VJP | `105.035709, 104.823333, 104.710250, 105.049334, 105.058125 ms` | `105.035709 ms` |
| row-major control | `107.868000, 107.885083, 107.863125, 107.872125, 107.870958 ms` | `107.870958 ms` |
| packed VJP only | `104.205333, 104.183500, 104.205958, 104.203875, 103.794875 ms` | `104.203875 ms` |

Pack-inclusive execution is `1.026993x`, a `2.6284%` command-GPU time saving.
Packed arithmetic without pack cost is `1.035191x`, a `3.3995%` command-GPU
time saving. Both miss the `80.903219 ms` promotion threshold decisively. The
`0.831834 ms` difference between candidate medians is not a directly measured
pack duration.

The complete `512x6656` output tensor, or 3,407,872 F32 values, matches the
incumbent bitwise. A separate small oracle verifies every packed record byte.
The stopped invocation finished in `2.06 s`. Down was skipped because both
directions were mandatory and gate/up alone falsified promotion.

## Interpretation

Block-major Q8 record locality is not a material lever for the incumbent
address-only C16/Q128/K64 body at gate/up. Free packing remains more than 23 ms
outside the threshold, so neither tiled packing nor persistent sidecar
amortization can rescue this unchanged packed body. This result does not close
other arithmetic, dequantization, or backend changes.

The private pack kernel, packed VJP, wrappers, correctness test, and profile
harness were removed in full. Production remains on row-major C16/Q128/K64.

## Next Gate

Keep row-major Q8, C16/Q128/K64 geometry, output traversal, and F32
accumulators. Test half-staged dequantized weights and cotangents with
half-input/F32-accumulate MMA against the incumbent at `6656x19968` gate/up and
`19968x6656` down with `n_query=512`. Use deterministic nonzero, block-varying
Q8 records and cotangents with separate outputs. Require finite complete output,
relative L2 at or below `5e-5`, scaled max
`max_abs(candidate-control) / max(max_abs(control), 1)` at or below `2e-4`, and
cosine at or above `0.9999999`. After one warmup per arm, require at least 25%
median command-GPU time saving over five order-alternating pairs on both
shapes. Assert retrospective elapsed time at or below 180 seconds and apply an
external 180-second cap. No model asset is needed.

Adversarial KILL and leverage review:
`01a0516b-475c-7291-92c4-7017a79fa3d8`.
