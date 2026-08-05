# DeepSeek V4 FP4 No-Double-Score Falsifier

Date: 2026-08-04

Status: bounded instrumented falsifier `GO`; collapsed one-command engineering
is authorized. Ordinary-token, terminal-speed, production selection, cache
migration, and snapshot v2 remain `HOLD`.

## Question

The real-weight sidecar established that official FP4 selector exchanges preserve
downstream quality, but every candidate token still computed both the retained
F16 Lightning scorer and the packed FP4 scorer. This experiment asks the smallest
next question: when attention consumes the same FP4 IDs, can the F16 score and
selection pipeline be omitted without changing one output or causal-state bit,
and does that omission save enough GPU time to justify collapsed-command work?

The predeclared gate requires:

- exact packed, singleton-audit, and final logits across paired A, FP4-only,
  and paired B;
- exact causal-state digests, consumed-ID traces, and committed-token
  transcripts across all three arms;
- exact decision transcripts including routes and FP4 reports across all arms
  at the singleton audit; paired-control packed reports exact and no timed
  reports;
- an encode-time ledger proving zero F16 score/selector pipeline invocations
  on FP4-only positions;
- paired-control drift at most 5%;
- FP4-only median at least 1.0 ms below the faster paired control.

## Implementation

The diagnostics session now seals one exhaustive authority mode:

- `F16Authoritative` derives F16-only execution, or paired observation when a
  report is armed;
- `PairedCounterfactual` always computes both and consumes FP4;
- `Fp4OnlyExperimental` computes only FP4 unless an explicit audit is armed.

Singleton and packed indexers split common query/head-weight preparation from
the F16 score/selection pipeline. FP4-only executes the common preparation once,
then the existing pack/preflight/matrix-score/selector pipeline once. It still
publishes and reads the F16 selected-attention cache; this experiment changes
selection scoring only.

An atomic singleton audit validates and arms the existing F16 decision transcript
and FP4 report before either state changes. Direct decision arming is rejected in
FP4-only mode. Packed execution retains its one-sparse-query restriction and
rejects wider geometry before token staging or causal mutation.

Every successful instrumented token exposes an honest operation ledger:
common-preparation invocations, composite F16 score/selector invocations, and
composite FP4 pipeline invocations. It does not claim individual Metal kernel
dispatch counts. The consumed trace now binds selection source in addition to
execution kind, position, layer, visibility, and all 512 IDs.

## Identity

- Base revision:
  `cdf4e0a570989c9f4c9f0fc8e125ba1b3de2a1e0`.
- Six-file campaign source SHA-256:
  `dda85443ac46af09e6d5de912b1144e6ce923db6186c52739cbada37eccb0d56`.
- DeepSeek Metal source SHA-256:
  `bd6e7ce08c84dbfad93999ef6008310153fccf5dca6c848e1ccae9d332747b8a`.
- Embedded metallib SHA-256:
  `45c446a9151bae8a217d223554b28b5c6e3a1042d5faa6fe7b01ba72db28ded0`.
- Test executable SHA-256:
  `6e1cbc5dbc7c6467359a6f0e5fcceb8a0be904796941cc1e85abae1af84f1333`.
- Refreshed model-content BLAKE3:
  `ae11d1ea13ccfd98509d248705a589384412cd67c502450158f84a8bd143b5e2`.
- Token transcript SHA-256:
  `5ef390b5ff3dcb4e14c0fb1dc18d703bea7db08f1c2c20ac12a93f56cbb40125`.
- Hardware: Apple M4 Max, registry ID `4294968482`.
- Rust: `rustc 1.97.1 (8bab26f4f 2026-07-14)`, LLVM 22.1.6.
- Metal: Apple metal 32023.864, target `air64-apple-darwin24.6.0`.
- SDK: macOS 26.2.

The schema-v2 source packet is 1,078,035 bytes at SHA-256
`8829f365a7f50c38c54a145cd14a4768f151911a2fd7ca87361f69008e879a85`.
`summary.json` retains the complete provenance, request, memory, hashes,
timings, dispatch ledgers, and every audit cutoff exchange without the full
score vectors. Its SHA-256 is
`d88655c5a964cfa293f2eb22e6bc535a3ed07c507b342fff24c83d81890c8950`.

## Protocol

One residency runs three fresh 2,061-forward sessions:

1. paired counterfactual A;
2. FP4-only experimental candidate;
3. paired counterfactual B.

All advance `[35, 201, 200, 34] * 512` under packed execution. Position 2,051
is one packed sparse query. In the candidate it is FP4-only. Position 2,052 is
an explicit paired singleton audit. Positions 2,053-2,060 are unreported timed
tokens: controls remain paired by sealed mode and the candidate remains FP4-only.
No timing arm performs report construction or readback.

Exact command:

```bash
env -u DSV4_FP4_OBSERVE_ONLY \
  DSV4_FP4_NO_DOUBLE_SCORE=1 \
  DSV4_MODEL=/Users/tito/models/deepseek-v4-flash-0731/UD-IQ3_XXS/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf \
  DSV4_FP4_PACKET=/Users/tito/code/qwen-llm/target/dsv4-current-fp4-no-double-score-final.json \
  cargo test --release -p qwen-llm --features dsv4-diagnostics \
  --test deepseek_v4_position_zero_live \
  current_deepseek_v4_fp4_selection_counterfactual_packet \
  -- --ignored --exact --nocapture
```

The campaign completed in 201.28 seconds.

## Exactness

All three arms produce identical F32 logit vectors:

- packed position 2,051:
  `f08efb321a4972d9de5d3b1ad53fd3cc961369f14e8d0a7c25e419dde6b7978c`;
- paired singleton audit at 2,052:
  `f7e1ecd79f260be8b53c7f17d330ad3b9d3c6b1fc87f217672e1cec1559924e5`;
- final timed position 2,060:
  `99726fc307b0068e05cd0dd7fec1910752e8709d95a5cfeac827580a90c1f7bd`.

Causal-state digests are likewise identical at packed, audit, and final
endpoints. The final digest is
`68122e6f8c0ff2ab4b6265ed7993b28e25d29a92a79b8c6fa338f34977732d5c`.
All arms consume the same 210 CSA-layer selections under final trace BLAKE3
`28ddb115041232ba85e6a03fbd6a36646a6677a6a001f2d6d88a0f5d237b3ce3`.

The candidate's singleton paired audit reproduces the previous real-weight
packet: 21/21 Q/K-ready layers, 11 exact masks, ten reciprocal rank-512/rank-513
exchanges, median score relative RMS 0.09809, and maximum 0.24306. Its F16
decision transcript and full report are exact to both paired controls.

## Dispatch Evidence

Every recorded sparse position covers all 21 CSA layers:

| Arm/position | Common | F16 score/select | FP4 pipeline | Consumed |
|---|---:|---:|---:|---|
| paired packed 2,051 | 21 | 21 | 21 | FP4 |
| FP4-only packed 2,051 | 21 | 0 | 21 | FP4 |
| candidate audit 2,052 | 21 | 21 | 21 | FP4 |
| paired timing 2,053-2,060 | 21/token | 21/token | 21/token | FP4 |
| FP4-only timing 2,053-2,060 | 21/token | 0 | 21/token | FP4 |

Ledger schema, execution kind, position, plan, source, and counts are asserted
for every arm and every timed token before packet publication.

## Timing

| Arm | Repeated singleton GPU median, ms |
|---|---:|
| paired A | 46.738 |
| FP4-only | 45.291 |
| paired B | 46.802 |

Control gap is 0.064 ms and drift is 0.136%. FP4-only saves 1.479 ms against
the control midpoint and 1.447 ms against the faster control, clearing the
frozen 1.0 ms worst-control gate by 45%.

This measures the instrumented one-command-per-layer schedule at 513-515 visible
rows. It proves that omitted F16 work survives the complete layer graph and has
a measurable ceiling. It does not transfer that saving to the collapsed
one-command token or terminal history.

## Decision

Promote bounded A as a successful falsifier and proceed to collapsed B. Retain
FP4 IDs in layer-addressed slices, validate every consumed source/count/status/
visibility/ID record after the single command, and hash traces in layer order
before token commit. Reuse query, score, and mask scratch serially; do not retain
full scores or masks per layer.

Paired audits may remain on the instrumented path. After collapsed exactness
passes, pay one real-weight prefix to position 3,070, snapshot before sealing,
and compare restored F16 controls against the original sidecar-bearing candidate.
No snapshot v2, paged K, multi-query FP4, or production switch is authorized.
