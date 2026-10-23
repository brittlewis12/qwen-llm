# GLM-5.3-Flash dense tensor prefill

## Decision

Enable the bounded M5 route within the existing Fast/Half policy. Repeatable
ordinary-prefill and saved-request gains convert without changing operands,
context, buffers, or strict precision modes. The expanded text screen has
a small positive loss estimate with an interval spanning zero; all eight
paired tool generations are correct and token-identical. This supports the
Fast/Half optimization, not a proof of model-wide quality noninferiority.

## Scope

Reuse the existing Metal tensor-ops dense tile for ordinary GLM Fast/Half
Q6_K and Q8_0 projections at 1024 or more active rows. The family-local
selector preserves Exact, F32-sensitive stages, short packets, grouped MLA
absorption, other storage, and unavailable tensor tiers. Both old and new
paths use Half operands and F32 sums; reduction order changes. This is not
an exact-logit optimization or a new precision mode.

The route adds no buffers, activation copies, or memory-policy exceptions.
Binding extent, alignment, device, writability, alias, integer, serial-encoder,
and pipeline-capacity checks precede encoder mutation. The existing
`QWEN_EXPERT_TENSOR_TILE=0` rollback keeps the incumbent. M4 declines before
new binding checks or tensor pipeline preparation.

## Method

- M5 Ultra, 256 GiB; Metal API validation enabled throughout. Every GPU
  process holds the production lease. Compilation also holds the lease,
  outside measured inference.
- Ordinary request baseline: clean `8b7f2f00`; candidate: clean `7a7012f7`.
  Both frozen binaries have matching build/runtime source identity. The
  baseline and candidate source roots remain pinned independently.
- Native same-binary probe: `babdeff3`, old/tensor thread-local overrides.
  `QWEN_MATMAT_Q6_K_N64_MIN_N=1024` makes the unit-test incumbent match the
  production Q6 N64 threshold; ordinary CLI runs have no threshold override.
- Untimed warmup/census for each arm, followed by plain ABBA then BAAB.
  No profiler, trace, or census runs during reported plain samples.
  Session allocation is outside native prefill wall; GPU waits and logits
  readback remain inside it.
- Full-model/session preflight precedes weight loading. Retained comparison
  logits, census growth, and analysis scratch are additionally charged
  against both working-set and reported process headroom, including the
  standard omitted-process-budget case.

## Ordinary Prefill

Four samples per arm and prompt length, always a 2048-row workspace. Frozen
quality-v1 long-document token IDs supply both prompt lengths.

| Prompt Tokens | Old Mean ms | Tensor Mean ms | Wall Change | Throughput Change |
| --- | ---: | ---: | ---: | ---: |
| 2048 | 3604.874 | 3061.376 | -15.08% | +17.75% |
| 4096 | 7613.441 | 6524.326 | -14.31% | +16.69% |

Untimed census confirms 507 dense substitutions per 2048-token first chunk:
288 Q6_K and 219 Q8_0. At 4096 tokens there are 1025 substitutions: 576 Q6_K
and 449 Q8_0. Total dispatch counts stay 1342 and 6597 respectively. All
non-dense kernel counts match; this census does not establish equal routing
data or equal logits.

## Saved Request

Native gameplay messages, 11,105 input tokens, 256 output tokens, capacity
11,360, admitted/frozen 2048-row workspace. Sampling is temperature 1,
top-p 0.95, top-k 0, min-p 0, seed 42; no speculation. Both arms receive the
same messages and request settings. All eight measured requests report
zero cold prefetch windows and zero prefetch bytes.

| Bracket | Old Prefill s | Tensor Prefill s | Old Request s | Tensor Request s |
| --- | ---: | ---: | ---: | ---: |
| ABBA | 22.377 | 19.638 | 28.522 | 25.809 |
| BAAB | 23.452 | 20.366 | 29.611 | 26.524 |
| All Samples | 22.915 | 20.002 | 29.067 | 26.167 |

The combined prefill wall falls 12.71%; loaded-request wall falls 9.98%.
The second bracket has slower controls and a slower final tensor sample;
all samples remain included. Decode kernels are unchanged, with observed
means 6.151 versus 6.164 seconds; no decode gain is claimed.

Generated token fingerprints are stable within each arm but differ between
arms. Both generations hit the 256-token cap in reasoning, so this replay
demonstrates latency conversion and repeatability, not equal answers or a
completed gameplay quality score. Private messages and generated text are
not committed.

## Bounded Continuations

The initial screen uses the first frozen prose/code/long documents at
2048/2048/4352 tokens, each followed by 64 identical true held-out tokens.
Only prefix execution differs; continuation uses the unchanged scalar
decode path. All full-vocabulary logit rows are finite and target-aligned.

| Stratum | Old Mean NLL | Tensor Mean NLL | Old Hits | Tensor Hits | Top-1 Agreement | Mean KL(old \|\| tensor) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prose | 1.730009 | 1.746312 | 38/64 | 38/64 | 63/64 | 0.005672 |
| Code | 1.239073 | 1.224034 | 46/64 | 46/64 | 64/64 | 0.005030 |
| Long | 2.957709 | 2.985913 | 26/64 | 27/64 | 61/64 | 0.018241 |

This mixed, nonzero change motivates the full frozen document and
long-context tool screen rather than an arbitrary exact-bit or KL cutoff.
The expanded screen includes these three cases, not an independent
replication.

## Expanded Quality

Source `5289a3dd` fixes all 22 frozen documents in fixture order, each at its
maximum frozen prefix: eight prose and eight code at 2048, six long at 4352.
There are 1408 held targets per arm. No short, unaffected 512-token items
dilute this comparison.

| Stratum | Old Mean NLL | Tensor Mean NLL | Delta | Central 90% Interval | Old Hits | Tensor Hits |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Overall | 1.464350 | 1.468351 | +0.004001 | [-0.000263, +0.008127] | 970/1408 | 969/1408 |
| Prose | 2.577239 | 2.581650 | +0.004411 | [-0.004290, +0.012235] | 239/512 | 238/512 |
| Code | 0.466902 | 0.466785 | -0.000117 | [-0.004116, +0.003162] | 457/512 | 456/512 |
| Long | 1.310428 | 1.319373 | +0.008944 | [-0.000746, +0.018407] | 274/384 | 275/384 |

Intervals use the existing reference's document bootstrap: 20,000 paired
resamples, seed 20261008, stratified overall. There is one maximum-prefix
item per document, rather than the original Fast-versus-Exact comparison's
38 items. No numerical noninferiority verdict is applied. In particular,
the overall upper bound does not establish a 0.005-nat margin. Top-1 change
is -0.071 percentage points, central 90% [-0.497, +0.355]; old/tensor argmax
agreement is 1363/1408. Mean KL is 0.006033, worst token 0.112870.

All four frozen long-context tool tasks T09-T12 run greedy and seed 11 in
both arms: eight generations per arm, 16 total. The existing
`quality_analysis.py::tool_correct` strict scorer finds 8/8 correct in each
arm, with identical generated IDs in all eight pairs and the required
observation stop. This is a finite weather/currency screen, not general
agent or engineering-task qualification.

`extended-documents.csv`, `extended-tokens.csv`, `extended-tools.csv`, and
`extended-summary.json` retain the complete sanitized cohort results.
`native-prefill.csv` and `saved-request.csv` retain every timing sample.
`evidence.json` records source, binary, fixture, and raw-report digests.
Private request inputs and device filesystem metadata stay outside the
repository.

## Correctness And Fallbacks

- Six CPU tests cover arithmetic/stage/type/row eligibility, checked geometry,
  pipeline limits, and nested/thread-local override restoration.
- M5 witnesses cover 22 native Q6_K/Q8_0 shape/width cases, F64 full/half
  operand references, true M=65 and N=1025 tails, poisoned outputs, guarded
  nonzero offsets, invalid bindings before mutation, and fresh-process global
  rollback. The available tensor tier is required, not silently skipped.
- Actual M4 fallback declines without encoder/output mutation and runs the
  incumbent. Twenty-two existing precision/native-dispatch tests and all
  seven short-packet policy/dispatch witnesses pass with validation enabled.
- The production CLI library passes adaptive R2048 cold/snapshot/live reuse,
  cancellation/retry, short suffix, and deep off-grid checkpoint restore.
  Positions, history, width, full-vocabulary logits, and emitted bytes match
  their uninterrupted/captured schedule. The serial CLI unit run passes
  957 tests; model/GPU tests remain explicitly selected rather than counted
  as passing through the default ignored set.

## Reproduction

Ignored native tests live in
`glm5_next_metal::tests::quality::tensor_prefill`. Use a release test binary,
`MTL_DEBUG_LAYER=1`, `QWEN_METAL_LEASE_WAIT=1`,
`QWEN_MATMAT_Q6_K_N64_MIN_N=1024`, and `GLM53_GGUF` naming the stock
UD-IQ3_XXS first shard through its standard shard path. Do not canonicalize
the entrypoint to a backing blob. Leave the tensor and Q6 N64 rollbacks on.

```sh
# Each test acquires the production lease itself; do not wrap it in another.
GLM53_TENSOR_TIMING_REPORT=new-timing.jsonl "$TEST_BINARY" \
  glm5_next_metal::tests::quality::tensor_prefill::tensor_prefill_plain_abba \
  --ignored --exact --nocapture --test-threads=1
GLM53_TENSOR_QUALITY_REPORT=new-bounded.jsonl "$TEST_BINARY" \
  glm5_next_metal::tests::quality::tensor_prefill::tensor_prefill_quality_screen \
  --ignored --exact --nocapture --test-threads=1
GLM53_TENSOR_EXTENDED_REPORT=new-extended.jsonl "$TEST_BINARY" \
  glm5_next_metal::tests::quality::tensor_prefill::tensor_prefill_extended_quality \
  --ignored --exact --nocapture --test-threads=1
uv run docs/bench/2026-10-22-glm-tensor-prefill/analyze_extended.py \
  new-extended.jsonl --out new-analysis.json
```

The frozen `scripts/reference/glm53/quality-v1.json` IDs remain authoritative.
These measurements do not claim general model noninferiority, a tokenizer
compatibility proof, or a new context limit.
