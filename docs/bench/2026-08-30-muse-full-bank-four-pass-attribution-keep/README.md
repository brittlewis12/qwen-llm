# Muse Full-Bank Four-Pass Attribution KEEP

Decision: **KEEP** the diagnostics-only four-pass observer and its model-free
target-51 B32/T16 attribution. Feed-forward reverse owns 79.8554% of the
attributed command and is the only immediate material stage.

## Contract

The production bank body was mechanically extracted into four helpers with
explicit owned boundaries:

1. feed-forward reverse to `scratch.hidden[2]`;
2. attention-output reverse to `scratch.query[0]`;
3. causal-GQA reverse to `scratch.query[1]`, `scratch.kv[0]`,
   `scratch.kv[1]`, and `scratch.query[2]`;
4. attention-input reverse to `grad_input`.

Ordinary execution calls all four through its existing single serial encoder
and preserves dispatch order. The sampled arm opens four descriptor-backed
serial encoders inside one command buffer. It uses legal stage-boundary
attachments only: no dispatch-boundary samples, manual counter calls, rotating
prefixes, or cumulative subtraction.

The model-free fixture uses zero-Q8 released geometry, target block 51, B32,
T16, and the R rule. It warms ordinary and sampled arms once, then runs
`O/S/O/S/O/S/O`. Every arm must preserve residual-identity output bits and the
complete ordered kernel/grid/threadgroup sequence. Sampled arms require raw
coverage within 0.5%, transition ambiguity at or below 2.5%, interpolated
ordinary perturbation at or below 10%, and at least two valid samples. Both
arms require drift at or below 5%; accepted stage shares must repeat within two
percentage points.

## Result

All three sampled arms are valid:

| Measure | Values |
| --- | --- |
| ordinary command | `50.491625, 50.512042, 50.797875, 50.793750 ms` |
| sampled command | `50.350542, 50.678625, 50.619791 ms` |
| sampled perturbation | `-0.2996%, +0.0467%, -0.3465%` |
| raw coverage | `1.000000820, 0.9999999999, 1.000000010` |
| transition ambiguity | `2.9791e-5, 2.8770e-5, 3.2932e-5` |
| encoder gaps | `0.001500, 0.001458, 0.001667 ms` |
| encoder overlaps | `0, 0, 0 ms` |

Ordinary drift is `0.605%`; sampled drift is `0.650%`. Output bits, dispatch
count, kernel order, grid geometry, and threadgroup geometry are exact across
all arms. Encoder count is the only topology difference: one ordinary versus
four sampled, all serial.

Median attributed stages:

| Stage | Median | Share |
| --- | ---: | ---: |
| feed-forward reverse | `40.414000 ms` | `79.8554%` |
| attention-output reverse | `2.788833 ms` | `5.5106%` |
| causal-GQA reverse | `1.057417 ms` | `2.0894%` |
| attention-input reverse | `6.348708 ms` | `12.5446%` |

The medians sum to `50.608958 ms`; independent stage medians need not equal a
sampled command median. Maximum accepted share spread is below `0.030`
percentage points. B8 samples are `13.906125, 13.922625, 13.906875 ms`, for a
`13.911875 ms` mean; B32/B8 is `3.641`, retaining B32. The command output
records `0.088 s` of model preparation, and the complete invocation finishes in
`0.65 s`. No model asset, identity path, artifact, or weight hash is involved.

The existing nonzero tiny J/R scalar-equivalence gate also passes after helper
extraction, with scaled max at or below `3.21e-7`.

## Interpretation

Cross-packet standalone Q512 controls sum to
`2 * 13.420500 + 12.945792 = 39.786792 ms`, or 98.448% of the 40.414 ms FF
median; the four-pass observer does not internally split that stage.
Attention-output, GQA, and attention-input together expose at most 20.1446%
bank-time saving, equivalent to `1.2523x` if eliminated, so another unpriced
retile there is not justified. The next candidate must change the FF
representation rather than another C16/Q128/K64 schedule detail.

## Next Gate

Add a `cfg(test)` seam that packs down to F16 `[H,F]` and gate/up to F16 `[F,H]`,
then calls `encode_mat_mat_f16_half_act_f32` directly. Use one serial command per
arm. The candidate command contains three packs followed by eight complete FF
reverses; control contains eight incumbent Q8 FF reverses. Compare each
command's `GPUEndTime-GPUStartTime`; keep allocation and differential readback
outside both intervals. Require the full Muse numerical envelope for every
intermediate and final output and at least 15% median whole-FF saving.
Production remains Q8 pending a separate promotion.

Adversarial promotion and leverage review:
`01a0516b-475c-7291-92c4-7017a79fa3d8`.
