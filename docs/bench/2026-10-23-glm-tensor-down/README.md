# GLM-5.3-Flash M5 Tensor-Down Prefill

## Decision

Enable the bounded down-only route within the existing Fast/Half policy.
Dense-fixed plain prefill improves 9.1%/8.7% at 2K/4K; the saved native request
converts to 6.1% less prefill time and 4.5% less loaded-request time. Both
quality references remain fixed, with near-neutral aggregate NLL versus the
original reference, mixed stratum changes, and correct identical tool outputs.
Native shape, refusal, fallback, reuse and checkpoint witnesses pass.

This supports the Fast/Half optimization, not exact logits or model-wide
quality noninferiority. Strict precision modes and decode kernels are unchanged.

## Arms And Scope

| Arm | Dense tensor | Down tensor | Interpretation |
| --- | --- | --- | --- |
| `original` | OFF | OFF | Cumulative-quality reference |
| `dense_only` | ON | OFF | Incremental-down reference |
| `dense_and_down` | ON | ON | Candidate |

The independent overrides apply during prefill only; decode is unchanged.
These are explicit experimental arms, not aliases for mutable production
defaults. Both quality baselines are retained. The plain timing comparison
contains only `dense_only` and `dense_and_down`.

Arithmetic is Fast/Half on the actual available tensor tier, with workspace
rows 2048, Q6_K N64 minimum N 1024, and Metal validation enabled. The candidate
selects ordinary IQ3_S down at exactly 1024 or 2048 committed rows; IQ4_XS down
and fused gate/up remain unchanged. The artifact has hidden size 4096, expert
FFN size 2048, 288 experts and top-8 routing. The JSON preserves the recorded
tensor-shape counts and artifact layout; the CSVs retain schema, source/binary
hashes where recorded, flags, geometry and timing scope.

Measurements use M5 Ultra, 256 GiB, Metal API validation, the production GPU
lease and normal memory gates. Compilation also holds the lease, outside
inference measurements. Quality retains two full-vocabulary 64-row baselines;
CPU comparison/census storage is charged against combined working-set and
process headroom before loading weights, including an omitted process budget.

The frozen native probe is `0d0e23eb`, with explicit independent overrides.
The final default-on runtime and repaired refusal fixture are `5dc38d30`.
The saved-request baseline is the dense-only checkpoint `7a7012f7`, not the
pre-dense original. Both ordinary CLI binaries have matching clean
build/runtime source identity; their source roots remain pinned separately.

## Available Evidence

All four supplied reports end in a unique `complete` record:

- `plain-v2r.jsonl`: 16 samples, four per arm at each of 2048 and 4096 tokens,
  ordered ABBA then BAAB at each width. All 16 appear in `plain-samples.csv`.
- `stock-profile-v2r.jsonl`: warm call, two ordinary calls, one stage-profiled
  call; all recorded runs have finite logits and bit equality to the warm run.
- `quality-v2r.jsonl`: three frozen documents, 64 teacher-forced continuation
  tokens per document/arm, 576 token records, nine document-arm summaries;
  four tool tasks times two modes times three arms = 24 generations.
- `extended-v2r.jsonl`: all 22 frozen documents at their maximum frozen
  prefixes, 66 document-arm summaries, 4224 target-token records, and the same
  24 tool generations. The full report passes schema, fixture/hash,
  completeness, alignment, reference-arithmetic and strict-tool validation.

The extended schema is `glm53.tensor_down_extended_quality.v1`. No partial
report is analyzed. The 22-document cohort includes the original three:
it is not independent replication. These reports use the frozen same-binary
probe and explicit overrides; they are not native-CLI measurements of the
final default-on branch candidate.

## Plain Timing

| Prompt Tokens | Dense Only Mean (ms) | Dense And Down Mean (ms) | Change |
| --- | ---: | ---: | ---: |
| 2048 | 3059.0625 | 2779.5827 | -9.136% |
| 4096 | 6521.7729 | 5955.7569 | -8.679% |

Change is `100 * (candidate mean / dense_only mean - 1)`, not throughput
change. These are ordinary packed-prefill wall times including GPU wait and
logits readback; session allocation, checks and JSON are excluded. There is
no profiler, trace or census in the timed calls. Metal validation is on.
Four observations per arm in one run are diagnostic, not independent workload
replications; no timing confidence interval or performance verdict is assigned.

The separate dense-only stage profile assigns 56.8745% (about 57%) of its
scaled stage-span sum to routed experts. Profiled wall time is 3264.2951 ms
versus a two-call plain mean of 3060.3910 ms: **6.6627% profiling overhead**
(about 6.7%). Encoder splitting and sample resolution perturb execution.
These shares are not independent leaf costs and are not ordinary absolute
time; they cannot isolate the cost of IQ3_S down or predict an additive saving.

Untimed dispatch attribution records 39 IQ3_S down calls per full chunk,
switching generic to tensor only in the candidate; three IQ4_XS down calls and
84 fused gate/up dispatches per chunk remain. The 4096-token dense tensor
count is the recorded 1025, not an assumed doubling of the 2048-token 507.
Counts establish selection/geometry, not routing-data or numerical identity.

## Saved Native Request

Saved gameplay messages, 11,105 input and 256 output tokens, capacity 11,360,
admitted/frozen R2048, no speculation. Sampling is temperature 1, top-p 0.95,
top-k 0, min-p 0, seed 42. Ordinary CLI runs have no Q6 threshold override,
profiler, trace or dispatch census. Untimed warmups precede ABBA and BAAB;
all eight measured calls report zero cold prefetch windows and bytes read.

| Bracket | Dense-Only Prefill s | Candidate Prefill s | Dense-Only Request s | Candidate Request s |
| --- | ---: | ---: | ---: | ---: |
| ABBA | 20.109 | 19.008 | 26.335 | 25.263 |
| BAAB | 20.412 | 19.058 | 26.631 | 25.303 |
| All Samples | 20.261 | 19.033 | 26.483 | 25.283 |

Combined prefill wall falls 6.06%; loaded-request wall falls 4.53%. Controls
and candidates vary within the run, including the slower second candidate
in ABBA; no samples are dropped. Decode kernels are unchanged, with observed
means 6.221 versus 6.249 seconds (+0.44%); no decode gain is claimed.

Generated token fingerprints are stable within each arm and differ between
arms. Both hit the cap in reasoning. This demonstrates latency conversion
and repeatability, not equal answers or a completed gameplay quality score.
Private messages and generated text are not committed. `saved-request.csv`
retains every timing sample; `saved-request-summary.json` records bracket
means, fingerprints and raw-record digests.

## Three-Document Quality Screen

Each arm has 192 scored target tokens: 64 prose, 64 code and 64 long-context.
The table reports the mean of the document means, which here also equals the
token mean. Positive NLL delta means worse continuation prediction.

| Arm | Mean NLL (nats/token) | Top-1 Hits | Top-1 Accuracy |
| --- | ---: | ---: | ---: |
| `original` | 1.975596763 | 110/192 | 57.2917% |
| `dense_only` | 1.985419893 | 111/192 | 57.8125% |
| `dense_and_down` | 1.973896762 | 109/192 | 56.7708% |

| Comparison | NLL Delta | Top-1 Accuracy Delta (pp) | Argmax Agreement | Mean KL(ref \|\| arm) | Max Token KL |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense only vs original | +0.009823131 | +0.5208 | 97.9167% | 0.009647404 | 0.107418782 |
| Dense and down vs original | -0.001700001 | -0.5208 | 98.4375% | 0.007124092 | 0.071313396 |
| Dense and down vs dense only | -0.011523132 | -1.0417 | 98.4375% | 0.007102408 | 0.062081671 |

KL direction is always `KL(reference || arm)` over the recorded full
154880-token vocabulary. KL is not symmetric; the JSON also preserves the
reverse original-versus-dense-only comparison and self references. Argmax
agreement is distinct from accuracy against the real target token. NLL and
top-1 need not move in the same direction. These small observations do not
establish general quality equivalence.

The three-document screen has one document per stratum: **no bootstrap
intervals are emitted**.

## Extended Quality

Each arm has 1408 teacher-forced targets: eight prose and eight code documents
at prefix 2048, six long documents at prefix 4352, with 64 targets per document.
Means weight documents equally. `Dense` means `dense_only`; `Candidate` means
`dense_and_down`. All brackets in this section are descriptive central 90%
intervals, not thresholds or a formal quality verdict.

| Stratum | Documents | Original Mean NLL | Dense Mean NLL | Candidate Mean NLL | Original Hits | Dense Hits | Candidate Hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall | 22 | 1.464349584 | 1.468350565 | 1.463744706 | 970/1408 | 969/1408 | 968/1408 |
| Prose | 8 | 2.577238555 | 2.581649562 | 2.570045884 | 239/512 | 238/512 | 239/512 |
| Code | 8 | 0.466901572 | 0.466784934 | 0.469177853 | 457/512 | 456/512 | 457/512 |
| Long | 6 | 1.310428306 | 1.319372745 | 1.314765607 | 274/384 | 275/384 | 272/384 |

Paired NLL delta is arm minus reference, in nats/token; positive is worse.

| Stratum | Dense vs Original Delta [Central90] | Candidate vs Original Delta [Central90] | Candidate vs Dense Delta [Central90] |
| --- | --- | --- | --- |
| Overall | +0.004001 [-0.000263, +0.008127] | -0.000605 [-0.003375, +0.002132] | -0.004606 [-0.007796, -0.001604] |
| Prose | +0.004411 [-0.004290, +0.012235] | -0.007193 [-0.013622, -0.000916] | -0.011604 [-0.015514, -0.007570] |
| Code | -0.000117 [-0.004116, +0.003162] | +0.002276 [+0.000676, +0.004223] | +0.002393 [-0.000836, +0.006280] |
| Long | +0.008944 [-0.000746, +0.018407] | +0.004337 [-0.000683, +0.009358] | -0.004607 [-0.014473, +0.003688] |

Top-1 accuracy is 68.8920%, 68.8210%, 68.7500% for original, dense and
candidate, respectively. Accuracy deltas below are percentage points;
argmax agreement compares predictions, not correctness on the real target.

| Comparison | Stratum | Top-1 Accuracy Delta pp [Central90] | Argmax Agreement |
| --- | --- | --- | ---: |
| Dense vs original | Overall | -0.0710 [-0.4972, +0.3551] | 1363/1408 |
| Dense vs original | Prose | -0.1953 [-1.1719, +0.9766] | 480/512 |
| Dense vs original | Code | -0.1953 [-0.5859, +0.0000] | 510/512 |
| Dense vs original | Long | +0.2604 [-0.5208, +1.0417] | 373/384 |
| Candidate vs original | Overall | -0.1420 [-0.4261, +0.2131] | 1368/1408 |
| Candidate vs original | Prose | +0.0000 [-0.5859, +0.5859] | 485/512 |
| Candidate vs original | Code | +0.0000 [+0.0000, +0.0000] | 511/512 |
| Candidate vs original | Long | -0.5208 [-1.3021, +0.2604] | 372/384 |
| Candidate vs dense | Overall | -0.0710 [-0.4972, +0.3551] | 1367/1408 |
| Candidate vs dense | Prose | +0.1953 [-0.5859, +0.9766] | 482/512 |
| Candidate vs dense | Code | +0.1953 [+0.0000, +0.5859] | 510/512 |
| Candidate vs dense | Long | -0.7812 [-1.8229, +0.2604] | 375/384 |

KL is `KL(reference || arm)` in nats over the full vocabulary. Mean-KL
intervals resample document means; maxima are descriptive worst-token values
without intervals.

| Comparison | Stratum | Mean KL [Central90] | Max Token KL |
| --- | --- | --- | ---: |
| Dense vs original | Overall | 0.006033 [0.004818, 0.007421] | 0.112870 |
| Dense vs original | Prose | 0.009366 [0.007692, 0.010970] | 0.112870 |
| Dense vs original | Code | 0.002452 [0.001288, 0.003669] | 0.107419 |
| Dense vs original | Long | 0.006365 [0.002889, 0.010711] | 0.093542 |
| Candidate vs original | Overall | 0.005799 [0.004648, 0.007019] | 0.122392 |
| Candidate vs original | Prose | 0.009545 [0.007245, 0.011949] | 0.117667 |
| Candidate vs original | Code | 0.001952 [0.000996, 0.002945] | 0.068371 |
| Candidate vs original | Long | 0.005932 [0.003366, 0.008690] | 0.122392 |
| Candidate vs dense | Overall | 0.004971 [0.003840, 0.006240] | 0.117836 |
| Candidate vs dense | Prose | 0.008574 [0.006143, 0.011353] | 0.117836 |
| Candidate vs dense | Code | 0.001804 [0.000953, 0.002735] | 0.040978 |
| Candidate vs dense | Long | 0.004391 [0.002410, 0.007219] | 0.088045 |

The candidate's aggregate NLL estimate is lower than both references, but
that does not establish general quality equivalence: code NLL increases
against original, and long-context top-1 loses two hits versus original and
three versus dense. The zero-width code accuracy-delta interval versus
original reflects zero observed per-document hit deltas, not absence of
population uncertainty or identical predictions. No formal pass/fail or
non-inferiority verdict is assigned from any of these intervals.

### Dense Checkpoint Reproduction

The `original` and `dense_only` arms reproduce the previous dense-prefill
checkpoint in `docs/bench/2026-10-22-glm-tensor-prefill` exactly under the
mapping `original -> old`, `dense_only -> tensor`. The analyzer compares all
2816 overlapping token records and 44 document-arm summaries, not just
rounded aggregate means: NLL, top-1 hits/argmax/agreement, target alignment,
and original-reference KL match the earlier exports exactly. This reproduces
the checkpoint's overall NLL 1.464349584 vs 1.468350565, dense NLL delta
+0.004000981, central90 [-0.000263121, +0.008126716], and 970 vs 969 hits.

`analysis.json` records the prior checkpoint file digests, raw-report digest
and recorded binary digest alongside the new report provenance. This is
exact reproduction of recorded observations, not an assertion that unseen
full logits or all model behavior are identical. It also does not turn the
overlapping cohort into independent quality replication.

### Interval Method

For the completed extended cohort only, the analyzer imports `bootstrap`
from `quality_analysis.py`: 20,000 document resamples,
seed 20261008, stratified by prose/code/long, reporting percentiles 5 and 95
as central 90% intervals. One maximum-prefix item is used per document;
documents have equal weight, giving overall stratum weights 8/22, 8/22, 6/22
(not the older 38-item cohort weights). Whole documents, not individual tokens,
are resampled. Overall and within-stratum means, accuracy, paired deltas,
argmax agreement and mean KL get descriptive intervals; max-token KL does not.
Intervals are marginal, not simultaneous, and do not account for choosing
this repository cohort or hardware/run variation. No old margin or verdict
logic is reused.

## Strict Tool Screen

The analyzer imports `tool_correct` from `quality_analysis.py` and scores the
recorded calls, parse error and stop against the task expectation in frozen
`quality-v1.json`. It does not trust `lenient_first_call_match` or a
report-provided replacement expectation. Correctness requires exactly one
valid expected call and observation stop ID 154829.

In each of the bounded and extended reports, all three arms are correct on
4/4 greedy and 4/4 sampled (seed 11) tasks: 24/24 generations per report.
These repeated tasks are not pooled as independent evidence. Sampled IDs
including stop, emitted IDs excluding stop, and recorded stops are identical
in all 24 task/mode/arm-pair
comparisons. Per-generation output hashes and pairwise identity results are
retained, not generated text or call arguments.

- T09/T10: one committed 1455-row chunk each, zero down-eligible chunks.
  These are unaffected controls only for `dense_only` vs `dense_and_down`;
  `original` still differs in dense policy. The workspace allocation is 2048
  rows, but the actual committed chunk is 1455 rows.
- T11/T12: committed chunks `[2048,2048,912]`, two down-eligible chunks each.
  The candidate's expected IQ3_S tensor-down count is 78 per prefill. Tool
  chunk eligibility is recorded committed geometry, not a dispatch census.

This is a finite, low-power screen: only two eligible tasks and one sampled
seed per task. Identical outputs on it do not imply general agent quality or
bitwise-identical logits.

## Correctness And Fallbacks

- Five CPU tests cover arithmetic/stage/type/actual-row eligibility, checked
  geometry, pipeline capacity, and nested/thread-local/unwind-safe overrides.
  The two established down-policy tests, six dense-policy tests, 22 native
  precision/dispatch tests and seven short-packet tests also pass.
- Five M5 GPU witnesses pass on `5dc38d30`: native and older expert shapes,
  generic/tensor/F64 references, post-down weighted output, sparse/empty
  buckets, invalid IDs, slot-limit tails, guarded active views over larger
  backing allocations, and complete fused gate/up -> down -> weighted-sum
  composition. Invalid bindings and injected pipeline/capacity failures leave
  the encoder and outputs unmodified. The refusal sweep exceeds the command
  queue capacity without retaining unsubmitted command buffers.
- A fresh M5 process with `QWEN_EXPERT_TENSOR_TILE=0` retains the generic path
  despite the family override. Actual M4 hardware declines before tensor
  pipeline preparation and runs the unchanged generic composition. No new
  buffers, activation conversions, shader, memory bypass or context change
  is introduced.
- The production CLI passes adaptive R2048 cold/snapshot/live continuation,
  cancellation/retry, and short suffix checks. Its default deep off-grid
  checkpoint test preserves full-vocabulary logits, output and final state
  through capture/restore, including denied optional capture.
- All 1526 CLI/integration tests pass serially; the default suite leaves 70
  model/GPU cases explicitly ignored rather than counting them as qualified.
  The named native tests above run separately with Metal validation and the
  production lease, without externally wrapping already-leased processes.

## Reproduction And Files

The ignored native tests live in
`glm5_next_metal::tests::quality::tensor_prefill`. Use a release test binary,
`MTL_DEBUG_LAYER=1`, `QWEN_METAL_LEASE_WAIT=1`,
`QWEN_MATMAT_Q6_K_N64_MIN_N=1024`, and `GLM53_GGUF` naming the standard stock
UD-IQ3_XXS first-shard path. Keep the tensor and Q6 N64 rollbacks enabled.
Each native test acquires the production lease itself:

```sh
GLM53_TENSOR_DOWN_TIMING_REPORT=new-plain.jsonl "$TEST_BINARY" \
  glm5_next_metal::tests::quality::tensor_prefill::tensor_down_plain_abba \
  --ignored --exact --nocapture --test-threads=1
GLM53_TENSOR_DOWN_QUALITY_REPORT=new-quality.jsonl "$TEST_BINARY" \
  glm5_next_metal::tests::quality::tensor_prefill::tensor_down_quality_screen \
  --ignored --exact --nocapture --test-threads=1
GLM53_TENSOR_DOWN_EXTENDED_REPORT=new-extended.jsonl "$TEST_BINARY" \
  glm5_next_metal::tests::quality::tensor_prefill::tensor_down_extended_quality \
  --ignored --exact --nocapture --test-threads=1
```

Set `ARCHIVE` to the durable local archive directory containing all four
source JSONLs. From the repository root, regenerate this full CPU-only packet:

```sh
uv run scripts/reference/glm53/tensor_down_analysis.py \
  --archive "$ARCHIVE" --extended "$ARCHIVE/extended-v2r.jsonl" \
  --out-dir docs/bench/2026-10-23-glm-tensor-down
```

The previous dense checkpoint defaults to the maintained repository packet
`docs/bench/2026-10-22-glm-tensor-prefill`; `--dense-checkpoint DIR` can select
another copy of that checkpoint's exports. Its recorded fixture/report
identity and all overlapping exported token observations are checked before
outputs are written.

The analyzer is a uv script with NumPy 2.4.4 metadata. It validates all supplied
reports before writing any outputs and does not contact the run host, build,
or execute a GPU job. Re-running it overwrites only its named analysis JSON
and CSV outputs, not this README or later native replay material.

- `analysis.json`: descriptive aggregates overall and by stratum, tool counts,
  completion status, report/analyzer/helper hashes and provenance.
- `plain-samples.csv`: all 16 ordered timing samples.
- `profile-stages.csv`: all 16 recorded stage shares and scaled spans, with
  their perturbed timing scope.
- `quality-documents.csv`: all nine document-arm results against both
  baselines (18 rows), geometry and diagnostic prefill/decode timings.
- `quality-tools.csv`: all 24 strictly scored generations and token hashes.
- `quality-token-identity.csv`: all 24 pairwise token identity comparisons.
- `extended-documents.csv`: all 66 document-arm results against both baselines
  (132 rows), including geometry and diagnostic prefill/decode timings.
- `extended-tools.csv`: all 24 strictly scored extended-report generations.
- `extended-token-identity.csv`: all 24 extended pairwise token comparisons.
- `dispatch-census.csv`: ten untimed down-attribution records across plain,
  bounded quality and extended reports.
- `test_analysis.py`: CPU-only regression tests, including actual extended
  acceptance and completeness corruption checks.
- `saved-request.csv` and `saved-request-summary.json`: all eight ordinary
  CLI requests, bracket means, output fingerprints and raw-record digests.
- `verification.json`: native/local test counts and log digests, final probe
  fingerprints, and matching clean native baseline/candidate build identities.

All generated CSV/JSON files use LF newlines. Curated exports omit private
model/binary filesystem paths, shard device/inode and host identities,
generated text, and reservation details. Full raw report hashes identify the
durable inputs without copying their private metadata into this packet.

Validation includes unique header/final completion records, exact schemas and
counts, explicit flags, fixture/artifact-layout and frozen token hashes,
unique row matching, target alignment, both reference NLL/top-1 values and
summary arithmetic, finite recorded NLL/KL/logits markers, prompt/output token
hashes, sampled/emitted/stop consistency, tool expectations, and untimed
dispatch-count attribution. Recorded source/binary hashes are preserved, not
claimed independently re-hashed from an executable. The stock-profile schema
does not contain a binary hash; none is inferred. Artifact layout/tensor-table
identity is not a full model-content hash. Raw logits are not archived here,
so full-vocabulary KL cannot be independently recomputed by this analyzer.

Run the regression checks with the same NumPy version:

```sh
TENSOR_DOWN_ARCHIVE="$ARCHIVE" uv run --with numpy==2.4.4 python -m unittest \
  discover -s docs/bench/2026-10-23-glm-tensor-down -p test_analysis.py -v
uvx ruff check scripts/reference/glm53/tensor_down_analysis.py \
  docs/bench/2026-10-23-glm-tensor-down/test_analysis.py
```

The 24 tests cover all four actual complete archives, corruption/missing/duplicate
rows and hashes, reference arithmetic, strict scoring, control eligibility,
all plain samples, helper seed/stratification, interval-output plumbing and
LF CSV output. The added actual-extended test validates all 4224 tokens,
66 document-arm summaries and 24 tools, checks exact prior-checkpoint
reproduction, and recomputes both candidate paired-NLL intervals with the real
20,000-resample helper. It rejects missing/duplicate document, token and tool
rows, missing completion and a wrong completion count. The separate plumbing
test still uses mocked bootstrap output; helper seed/stratification is also
tested on a deterministic nondegenerate example. All ten generated JSON/CSV
files reproduce byte-for-byte, use LF newlines, and pass the privacy allowlist
check. Native request and correctness exports are separate from these ten
analyzer-generated files.
