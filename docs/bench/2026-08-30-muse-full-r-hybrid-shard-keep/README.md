# Muse Full-R Hybrid Shard KEEP

Decision: **KEEP** the fixed-scratch full-attention bank inside the complete
Muse full-transport fitter. The integrated path preserves the prior all-source
row result while reducing measured VJP wall by `2.8068x`.

## Contract

- released `Muse-Glimmer-30B-Q8_0.gguf` on the internal SSD;
- release binary from clean merge commit `126b4e79`;
- R rule, target 51, sources 0 through 50;
- one explicit 16-token prompt with no special-token insertion;
- rows `[0,32)`, inner query batch 32;
- one invocation, no warm bracket, no corpus fit.

The command completed in `25.24 s` process wall. The first forward measured
`11.232441 s`, so process wall and forward are cold directional observations,
not promotion comparisons. VJP wall is the comparable mechanism result.

## Result

| Measure | Prior CPU-attention path | Integrated hybrid | Change |
| --- | ---: | ---: | ---: |
| VJP wall | `14.079706 s` | `5.016234 s` | `2.8068x` |
| replay | `1.958641 s` | `1.097529 s` | `1.7846x` |
| feed-forward reverse | `2.512375 s` | `1.697464 s` | `1.4801x` |
| attention-output reverse | `0.218561 s` | `0.132495 s` | `1.6496x` |
| CPU attention reverse | `4.503687 s` | `0.429134 s` | `10.495x` |
| attention-input reverse | `2.810513 s` | `0.630659 s` | `4.4565x` |
| full-attention bank | n/a | `0.703416 s` | attributed |

The six named hybrid components account for `4.690697 s`, or `93.51%` of VJP
wall and `98.02%` of the nested `4.785256 s` component timer. The remaining
`0.325538 s` lies outside that nested timer.

At 208 B32 batches for all 6,656 output rows, this result projects the
25-prompt VJP-only floor at `7.246 h`. Capture, artifact publication, and
checkpoint I/O are excluded; this is not an end-to-end fit forecast.

## Correctness And Identity

The schema-v2 payload has shape `[51,32,6656]`. Comparing all `10,862,592`
F32 values with the prior all-source artifact gives:

- max absolute error `1.86264515e-7`;
- RMS error `1.03075124e-8`;
- scaled max error `1.80459926e-7`;
- identical replay diagnostics apart from expected last-bit variation.

Identity resolved as `DeclaredAndStored`, authenticated the released asset,
and hashed zero model-weight bytes. The payload comparison read only the two
43.45 MB result files.

## Disposition

The next gate is source- and model-free construction of a 256-row block-major
engine that prepares each prompt/layer replay once and runs eight B32 reverse
banks against it. One bounded no-publication R256 A/B may follow only after the
equivalence gate passes. Its predeclared mechanism floor is a 15% wall saving;
the timing model predicts `40.130 s` control versus `32.447 s` candidate.

Adversarial integration review: `01a05132-50ff-7d22-8dc6-3b47a6dfc54a`.
Source-only leverage review: `01a0516b-475c-7291-92c4-7017a79fa3d8`.
