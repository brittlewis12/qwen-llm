# GLM Q4 Decode Fusion: Exactness Passed, Conversion Parked

## Decision

Park candidate D default OFF and off main production. Actual-Q4 M5 validation
passed the bounded bitwise contract, but ordinary native conversion supplies
only suggestive evidence of an approximately 1% decode benefit, not a robust
wide-M5 default-promotion result. M4 is untested. The deep pooled latency
decrease is not a valid 4.8% performance headline because the late unchanged
decode control drifted substantially.

Prioritize C, the separate roughly 20-25% prefill optimization, rather than
another D encoder, geometry or control-repeat campaign. D's exactness and
timing results do not establish C's prefill quality or qualify a combination
with C. This closeout authorizes no new implementation, CPU build or GPU run.

This standalone document can be cherry-picked onto maintained main
`266c2825` or the C lane without the candidate implementation or preparation
README. This documentation checkpoint changes no production default.

## Source and Mechanism

Corrected witness source: `591f3712a10d5b576b1c2163fec6239283595841`.
Canonical native source: `64283a794b4bea0613a6ae19b24d1d81c1f86dee`.
Only the witness test and its documentation differ between those commits;
production is unchanged. The older commit's test artifact was superseded
before GPU qualification: the corrected witness pins actual KDA Q/K/V/O
storage as Q8_0 and requires zero dense-Q6 tensor calls. It does not copy the
synthetic release fixture's Q6 expectations into actual-artifact identity.

`QWEN_GLM53_Q4_DECODE_FUSED_GATE_UP=1` selects the existing native fused
gate/up helper for equal Q4_K banks at the released GLM serial-decode
geometry `(hidden, ffn, experts, top_k) = (4096, 2048, 288, 8)`. It replaces
two native projection dispatches with one for 41 routed pairs. Block 11's
Q5 pair remains on its fallback; packed prefill is untouched.

The decoded weights, activations, reduction and inner output remain F32.
The clamped SwiGLU formula, unweighted inner, existing down projection and
post-down routing weights are unchanged. There is no new shader, precision
substitution, scratch allocation or admission exception. Preload admission
prices the full Metal demand and unchanged CPU capture reserve separately.

The mechanical saving is 41 dispatches per token and about 5.125 MiB of
intermediate F32 gate writes/reads, not quantized weight traffic. Both dot
products still execute; the selected Q4 gate/up banks alone still stream
about 2.88 GiB/token across 41 blocks. Cache/intermediate traffic savings do
not imply a 2x decode or bandwidth improvement.

## Bounded Actual-Model Validation

On an M5 Ultra with 80 GPU cores and 256 GiB, the tiny witness completed in
0.11 seconds and the pinned six-shard GLM-5.3-Flash UD-Q4_K_XL model witness
in 54.38 seconds, with Metal validation ON (`MTL_DEBUG_LAYER=1`):

- One weight load, 12 fresh sessions, depths 512 and 4096, and 32 continuous
  teacher-forced decode steps per session.
- All 396 full-vocabulary rows checked bit-exact, including 66 baseline
  self-capture rows; all 24 common-prefix/final causal-state captures exact,
  including four baseline self-captures.
- 256 plain validation-ON decode steps; warm census work is separate from
  those timing samples. Retained shard stamps remained unchanged.
- Returned full-vocabulary rows are retained independently and compared
  AFTER each continuous loop, not through a same-buffer self-comparison.
  Persistent state is compared after the common prefix and final step, not
  captured between timed steps.
- Causal captures cover KDA conv/S, visible MLA latents, completed pools and
  the defined pending ring, excluding undefined scratch and capacity padding.
  Ordinary route/sparse checks, position/ledger checks and admission remain.

Both arms use common CURRENT Fast/Half packed prefill with core Q8-wide and
Q5-down ON and D OFF. Only the decode selector differs. The census preserves
the fixed decode inventory: 82 native Q4 passes versus 41 fused passes per
token, with unchanged Q5/Q6 down. At deep prefill it requires Q8-wide activity,
78 Q5-down tensor calls, and zero dense-Q6 tensor calls.

Validation-ON timings are diagnostic only, not canonical native performance:

| Depth | Baseline ms/token | Fused ms/token | Observed latency change |
| --- | ---: | ---: | ---: |
| 512 | 24.80895 | 24.42420 | -1.551% |
| 4096 | 25.57970 | 25.32210 | -1.007% |

This establishes the bitwise contract only for the exercised artifact,
prefixes and trajectories, not all inputs, artifacts or hardware. Identity
checks cover pinned shard names/sizes, metadata-only primary SHA, 660 named
mixed-role/KDA/MLA dtype-and-shape pins and retained file stamps. The actual
weight payload was NOT fully hashed; unchanged stamps are not payload hashes.

## Canonical Native Campaign

The native build and runtime checkout both identify
`64283a794b4bea0613a6ae19b24d1d81c1f86dee`: identity MATCH, clean source,
features `[]`, no overrides, SDK 26.2. Execution used an M5 Ultra with 80 GPU
cores and 256 GiB on macOS 27, with Metal validation OFF.

Canonical benchmark binary SHA-256:

```text
bf2d836ec43d3ef91ca3ecdc57766ce2aceb4329578b59c010b26d8a0cdca60e
```

The bracket order was A0, F0, F1, A1, with A = D OFF and F = D ON.
`QWEN_EXPERT_NATIVE_FUSED_GATE_UP=1`, core Q8-wide and Q5-down were ON in
both arms; the new D flag alone changed 0/1. Other experimental selectors
remained OFF. The campaign used unchanged `pp512` controls and `tg256` at
depths 512/4096, two runs per row and 15-second gaps. It completed in
398.094 seconds, retaining all 32 samples across 16 rows. Raw reports remain
private; no private report/model paths, prompts or token IDs are published.

All `tg256` elapsed totals in milliseconds, rounded to three decimals
(two samples per cell):

| Depth | A0 | F0 | F1 | A1 |
| --- | --- | --- | --- | --- |
| 512 | 5666.539, 5668.126 | 5606.210, 5605.764 | 5612.014, 5605.388 | 5694.366, 5697.433 |
| 4096 | 5426.539, 5424.127 | 5358.836, 5369.836 | 5381.386, 5388.795 | 5786.934, 5945.919 |

Pooling all four A samples and all four F samples, then dividing elapsed
totals by 256 tokens, gives the following descriptive arithmetic. Values
are calculated from the original nanosecond samples, not the rounded table:

| Depth | Pooled A ms/token | Pooled F ms/token | Observed latency change |
| --- | ---: | ---: | ---: |
| 512 | 22.19381 | 21.90369 | -1.307% |
| 4096 | 22.05422 | 20.99497 | -4.803% (invalid as a benefit headline) |

Unchanged control drift from A0 to A1:

| Depth | `pp512` elapsed-time drift | Decode elapsed-time drift |
| --- | ---: | ---: |
| 512 | +6.756% | +0.504% |
| 4096 | +9.727% | +8.130% |

The late deep A1 slowdown inflates the pooled baseline. Comparing the same
F0/F1 candidate average against only early A0 gives -1.059% at depth 512 and
-0.933% at depth 4096. This is a drift-sensitivity comparison, not a second
controlled replication or a substitute causal estimate. Together the
measurements suggest an approximately 1% effect, but do not isolate it
robustly from campaign nonstationarity. The deep pooled 4.803% decrease
cannot be presented as an attributable gain.

The unchanged prefill and baseline-decode movement is not evidence that D
causes a prefill regression or a causal decode regression. D changes no
packed-prefill operation. Nor do validation-ON timings establish the native
effect size, and this decode campaign supplies no separate prefill-quality
verdict.

## Closeout

Keep D default OFF and off main production. The bounded math result is
positive; the ordinary conversion is only suggestive around 1%, lacks a
robust wide-M5 promotion case and says nothing about untested M4. Do not run
another encoder-order, geometry or control-repeat campaign for D on this
evidence. Highest-leverage focus moves to C's separate roughly 20-25%
prefill result and its own quality/qualification evidence, not a combined
claim derived from D's bitwise witness or drift-sensitive timings.
