# Native dense IQ2 residency: Saluki capacity milestone

Decision: enable native IQ2_XS and IQ2_XXS for dense-model projections. Keep
MoE-model projections, routed banks, embeddings and output heads on their prior
policies. This is a partial capacity fix, not a claim of near-file-size RSS.

## Realized capacity

Artifact: `Underdog-Saluki-27B-1.0-IQ2-mix.gguf`, architecture qwen35, 851
tensors, 7.356 GiB file. The original report's roughly 75 GiB RSS / 7 tok/s
was not reproduced as a baseline. Source inspection instead identified four
converted cohorts, not IQ2_XXS alone; see `docs/IQ-QUANT-NATIVE-CAPACITY.md`.

| Cohort | Tensors | Previously planned F32 GiB | Native GiB | Saved GiB |
| --- | ---: | ---: | ---: | ---: |
| IQ2_XS | 39 | 8.047 | 0.582 | 7.465 |
| IQ2_XXS | 119 | 34.277 | 2.209 | 32.068 |
| Combined | 158 | 42.324 | 2.791 | **39.533** |

The whole packet loads one model under ordinary admission and reconciles every
source binding against its prepared plan: 784 direct copies and 67 conversions,
zero aliases/views/tail fallbacks/derived allocations. Actual logical weight
storage is **26.539263 GiB**, versus the old descriptor-derived 66.072687 GiB.
IQ1_S/IQ1_M still expand 1,128,038,400 bytes to 21,736,980,480 bytes. Their
remaining avoidable expansion is approximately 19.194 GiB, including the IQ1_M
embedding. No expanded full-model A arm was allocated.

Loaded point samples (not peaks): Metal allocation 28,499,705,856 bytes,
RSS 36,608,933,888 bytes and physical footprint 28,705,466,720 bytes. These
measure different things; RSS includes mapped source pages. Do not report
logical weights as RSS, or infer a measured 75-to-27 GiB RSS comparison.

## Implementation

- Role-aware storage policy is shared by planning and loading. The resolved
  dense-IQ choice is frozen during preparation, including diagnostic rollback;
  changing a thread-local test scope afterward cannot invalidate admission.
- Legacy global native coverage stays unchanged. A rank-two shape or tensor
  name cannot grant an embedding, head or MoE bank dense-projection admission.
- Existing IQ2_XS GEMV/GEMM gains native residency, physical-range validation
  and N-dependent activation addressing checks. Weight offsets remain ulong;
  no unnecessary u32 weight-bank size limit is imposed.
- IQ2_XXS has canonical 256-value/66-byte decoding, native F32 GEMV, scalar
  arbitrary-N GEMM and F32 register MMA with a 16-output-by-8-token tile.
  Neither operands nor decoded weights are staged through half precision.
- Apple7 capability, alignment and pipeline checks select MMA before dispatch;
  unsupported capability falls back to compressed scalar execution, never an
  unplanned persistent F32 weight. Invalid bindings remain errors.
- Shared dispatch makes checked zero-copy activation prefix views for packed
  buffers. Independent projections support ordinary concurrent encoders, with
  hazard notes recorded once after validation. No concurrency rollback is used.

The canonical table and fragment mapping cite llama.cpp's MIT-licensed source.
IQ2_XS and IQ2_XXS are distinct encodings; their decoders are not interchangeable.

## Primitive evidence

`iq2-xs-primitive.jsonl` and `iq2-xxs-primitive.jsonl` sample first/middle/last
physical rows from every one of the 39/119 tensors, at N1/N3. Inputs are fixed
synthetic activations, not captured model activations. CPU codec/F64 reference
max relative L2 is 2.16e-6 / 3.11e-6. All outputs are finite.

Two full representative matrices per format compare native execution against
canonical CPU-decoded F32 weights. Warm A/B then BAAB, N1/2/8/32/128/512;
ordinary command GPU/wall timings, warm census only. Both formats coexist for
this operator comparison, admitted with CPU reference and bounded conversion
staging (about460 MiB maximum incremental allowance). No whole CPU F32 weight
copy is retained. All admissions and timestamps succeed.

N1 GPU speedup is 1.97-2.28x for XS and about2.08x for XXS. XS N32-512 is
2.77-3.59x; XXS N2-512 means are 19.39-35.11x. The latter compares F32 MMA to
the **current scalar F32 matrix fallback**, not an optimized F32 GEMM or a
whole-model speedup. N1 uses the existing F32 GEMV. These timings exclude
load-time CPU conversion. XS small-N results are lower than its packed results.

Retain drift: XXS gate N512 B1/B2 takes19.04/9.12 ms, paired speedups
14.23x/30.18x. Full-output native-versus-F32 maximum absolute difference reaches
1.41e-4; relative L2 remains2.59e-6. Sampled oracle errors do not bound every
full-output absolute error. `summarize.py` reports both types of comparison.

## Whole-model qualification

`saluki-native-model.jsonl` uses one admitted native load and four sequential
trajectories: serial-token native GEMV reference versus ordinary packed128,
on prose N129 and code N131. Both receive four identical teacher-forced
continuations. Sessions are dropped between trajectories; packed tails are
one and three tokens. Normal load/session admission prices copied destinations,
source residency, prefetch workers, conversions, scratch and output references.

All 20 output records are finite. All 10 comparisons have matching top1 and
zero choice regret. Maximum bidirectional KL is5.72769e-7, relative logit L2
2.67510e-4 and absolute logit difference0.00490808. Position and token histories
reconcile. These compare execution modes on the same compressed weights, not
an independent whole-model implementation or persistent-state equality.

Witnesses show both native dtypes in packed matrices and ordinary decode.
Each sampled decode/continuation has15 concurrent XS and20 concurrent XXS
dispatches. Packed itself is serial. Captured execution overrides are absent;
only lease waiting is set. The recorded trajectory wall times are explicitly
nonbenchmark: census/readout scopes differ. No whole-model speedup, decode-rate,
long-context or broad task-quality claim is made from them.

Admissions report process-budget telemetry omitted; this is normal admission,
not affirmative independent process-headroom measurement. Review found a small
existing shared session-pricer undercount: actual193,314,816 versus estimated
192,719,648 bytes (595,168 difference), covered by the retained2 GiB reserve.
Correct that upper-bound estimate separately; do not erase the observed gap.

## Checks and reproduction

- Final Metal API validation:13 IQ2_XXS primitive/dispatch tests,11 dense-role
  tests including actual tiny-GGUF loader realization across opposite scopes;
  existing XS offset/codec test and invalid-binding checks also pass.
- Canonical tests cover all codebook/sign/scale positions, real K dimensions,
  ragged M/N, offsets, exact physical ends, immutability, device/capability
  fallback and concurrent independent projections.
- CPU bookkeeping/addressing checks, release test build, workspace check with
  tests, non-test release check, formatting and whitespace checks pass.
- Independent cx review approves this partial default after source, raw
  evidence, admission, per-binding realization and numerical review.

Build `cargo test -p qwen-llm --release --offline -j2 --lib --no-run`.
All diagnostics require a new output filename, release mode, normal lease
and memory admission. Set `QWEN_METAL_LEASE_WAIT=1`; unset `MTL_DEBUG_LAYER`
for timings. Optional `IQ_CAPACITY_GGUF` selects the artifact.

- Primitive: `metal_forward::tests::iq_capacity::iq_capacity_packet`,
  `IQ_CAPACITY_DTYPE=iq2_xs` or `iq2_xxs`, `IQ_CAPACITY_OUT=<new.jsonl>`.
- Whole: `metal_forward::tests::iq_capacity_model::iq_capacity_model_packet`,
  `IQ_CAPACITY_MODEL_OUT=<new.jsonl>`.
- Select only that test with `--ignored --exact --nocapture --test-threads=1`.
  Use `uv run summarize.py <primitive.jsonl> ...` for operator summaries.

Raw primitive packets bind the earlier serial-only primitive API and XS-only
storage integration. Whole binds final roles/concurrency/dispatch; arithmetic
shader hashes are unchanged. Original evidence is never rewritten.
The commit removes a trailing blank line from `iq2_xxs_grid.metalh` after
qualification; its recorded source hash therefore predates that whitespace-only edit.

Next: IQ1_M matrix/embedding and IQ1_S matrix coverage to remove the remaining
expansion. The shared session-pricer shortfall is a separate small accounting
follow-up. No MoE or new output-head admission is implied by this milestone.

Jams: cx `01a11893-3352-75b1-9ec7-2ed940154a10` (roles),
`01a11528-973c-7e40-ba8c-d4bece18c93b` (kernels),
`01a11893-3352-79d3-8d8d-8343c0d8f293` (harness); independent adversarial
review `01a118ca-c0f6-7c00-a0dd-23ae962883a5`.
