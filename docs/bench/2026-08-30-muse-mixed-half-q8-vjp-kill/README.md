# Muse Mixed-Half Q8 VJP KILL

Decision: **KILL** half-staged operands for the C16/Q128/K64 Q8 VJP. The
released gate/up shape meets its numerical envelope but takes 30.431% more
command-GPU time than the incumbent F32-operand kernel.

## Predeclared Gate

The private candidate retained row-major Q8 records, C16/Q128/K64 geometry,
output traversal, and F32 accumulators. It staged the dequantized 16x64 weight
tile and 128x64 cotangent tile as half, then used half-input/F32-accumulate MMA.
The resulting threadgroup footprint was 18,432 bytes.

At `n_query=512`, promotion required finite complete output, relative L2 at or
below `5e-5`, scaled max at or below `2e-4`, cosine at or above `0.9999999`, and
at least 25% median command-GPU time saving at both `6656x19968` gate/up and
`19968x6656` down. Each arm warmed once before five order-alternating command
pairs. The invocation was externally capped at 180 seconds and asserted the
same retrospective limit. It loaded no model asset.

## Result

Gate/up stopped the packet:

| Arm | Samples | Median |
| --- | --- | ---: |
| mixed-half | `17.560000, 17.564000, 17.563417, 17.573583, 17.564958 ms` | `17.564000 ms` |
| F32-operand control | `13.445000, 13.466125, 13.480208, 13.495042, 13.455708 ms` | `13.466125 ms` |

Incumbent/candidate is `0.766689x`; candidate latency is `1.30431x`, or
`30.431%` higher. The 25% promotion threshold was `10.099594 ms`, which the
candidate missed by `7.464406 ms`.

All 3,407,872 released-shape output values are finite. Released gate/up passes
its numerical limits with relative L2 `1.9372947e-5`, scaled max
`max_abs(candidate-control) / max(max_abs(control), 1)` of `2.8259445e-5`, and
cosine `0.999999999968477`. The auxiliary `96x128` screen did not pass relative
L2: `1.7739649e-4`, or `3.54793x` its `5e-5` limit, while its other metrics
passed. Released synthetic numerical acceptance therefore does not establish
real-block or chained full-transport quality.

The stopped invocation's retrospective elapsed time was `0.33546 s`. Down was
not measured because gate/up alone falsified the conjunctive two-shape gate.

## Interpretation

Half-input MMA does not repay the candidate's F32-to-half cotangent staging,
F32 dequantization followed by half staging/rounding, and 18 KiB threadgroup
footprint in this VJP shape. The regression is consistent with those added
costs, but no counter evidence attributes it to one specific cause.

B64, Q256, C32, block-major records, and mixed-half operands now cover the
locally motivated outer-bank, query-width, input-width, locality, and precision
retile hypotheses. This does not prove every Q8 kernel impossible; it does mean
the immediate common-Q8 microkernel search has no remaining demonstrated
material ceiling and should pivot.

The private kernel, wrapper, small screen, and ignored profile harness were
removed in full. Production remains on the F32-operand C16/Q128/K64 kernel.

## Next Gate

Attribute the incumbent model-free target-51 B32/T16 R bank with four legal,
descriptor-backed serial encoders inside one command buffer. Bind feed-forward
reverse to `scratch.hidden[2]`, attention-output reverse to `scratch.query[0]`,
causal-GQA reverse to `scratch.query[1]`, `scratch.kv[0]`, `scratch.kv[1]`, and
`scratch.query[2]`, and attention-input reverse to `grad_input`. Keep the
ordinary path on its existing single encoder. Forbid manual counter calls,
dispatch-boundary samples, and rotating or cumulative-prefix subtraction; check
command error before resolving timestamps. Require identity output and matching
dispatch topology, raw coverage within 0.5%, signed transition ambiguity at or
below 2.5%, perturbation at or below 10%, within-arm drift at or below 5%, at
least two of three sampled arms valid, and accepted stage shares repeating
within two percentage points before authorizing another candidate.

Adversarial KILL and corrected leverage review:
`01a0516b-475c-7291-92c4-7017a79fa3d8`.
