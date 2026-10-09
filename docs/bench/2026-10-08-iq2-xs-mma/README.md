# Native IQ2_XS F32 MMA: Saluki leaf and whole-prefill evidence

Decision: **guarded dense IQ2_XS MMA is promoted and the family-scope regression
is resolved and revalidated**. Production `None` follows guarded Auto for N>1,
preserving support/alignment scalar fallback and unchanged N1/decode paths. A
test-only forced-Scalar comparator is retained; there is no runtime force switch.
Both natural corpora improve in every measured GPU
and wall pair, with finite logit differences and unchanged endpoint/continuation
choices. No new bit-match policy is imposed. This documentation records the
promotion; it changes no production selection.

## Resolved family-scope correction

The owner reports that the broader `iq2_xs` suite exposed **three DeepSeek
scalar-lineage failures**, caused by a real selection-scope leak through shared
`encode_batch_projection`. The fix adds a checked explicit scalar entry and pins
IQ2_XS to that entry in the family helper. The existing exact tests and tolerances
are unchanged, and a new scope regression was added. The owner reports the broad
`iq2_xs` suite passing in **both release and debug with Metal API validation:
22 passed, 0 failed, 3 ignored in each run**. Both runs execute the three formerly
failing DeepSeek tests and the new scope case. The ignored physical-bindings test
was separately executed in debug and passed. The other two ignored tests are the
retained leaf/whole packets executed earlier; they are not unexecuted coverage.

All 12 promoted dense tests also passed debug hazard tracking. The four pricing
tests pass in both debug and release, and the final non-test cargo check passes.
An independent reviewer approved the scope fix: the dense body still uses
`scalar_only=false`, and shader/metallib are unchanged. The reviewer concluded
that no whole-packet repeat is needed for this family-scope correction.

All three retained performance packets precede this latest fix. In particular,
`production-129.jsonl` is now historical evidence of the postpromotion default
path before the family-scope correction, not a source binding for the current
tree. Raw packets remain untouched; no current-source hash equality is asserted.

## Historical postpromotion default-path confirmation (before family-scope fix)

`production-129.jsonl` is the unchanged 453,751-byte packet copied from
`/tmp/saluki-iq2-xs-production-129-20261008.jsonl`:

```text
SHA256 3acba5121a80542ba75dc7e1212224302ce7f1efecb3e4afb8b08e3811f3b364
```

This is schema `iq2_xs.native_mma.model.v1`, packet revision2: widths and candidate
are configurable and recorded in the header. Here N129/chunk128 runs one warm
A/B pair and one measured ABBA round per corpus. A is `Some(Scalar)`; **B is the
actual production `None` scope**, not forced MMA. It confirms 128+1 execution and
four ordinary continuation tokens. The owner reports all 12 promoted kernel tests
passing under Metal API validation, and this packet completing in 13.78 seconds.
The packet duration includes setup and is not a per-request timing.

| Stream | GPU median A / B ms | GPU saving | Wall median A / B ms | Wall saving |
| --- | ---: | ---: | ---: | ---: |
| Technical prose | 1072.010 / 679.068 | 36.65% | 1076.671 / 683.621 | 36.51% |
| Code review | 1085.631 / 697.327 | 35.77% | 1091.902 / 702.171 | 35.69% |

These medians have two measurements per arm. All four pairs improve:

| Stream | GPU pair savings, B1/A1 then B2/A2 | Wall pair savings | A2/A1 GPU | A2/A1 wall |
| --- | --- | --- | ---: | ---: |
| Technical prose | 36.673%, 36.636% | 36.522%, 36.491% | 1.000051 | 1.000105 |
| Code review | 35.327%, 36.200% | 35.158%, 36.217% | 1.018815 | 1.021638 |

Four warm censuses show 39 matrix calls per prefill: scalar in A and MMA in B.
**Every warm arm also has 39 XS GEMV calls for its singleton chunk**, with zero
extra MMA substitutions. Attribution is aggregate GEMV census plus total MMA
count, without splitting commands. Timed B scopes total 156 substitutions; warm B
scopes 78. All 48 continuation steps have zero substitutions. Other warm kernel
counts/concurrency match between variants.

All 60 output records are finite. The 30 B/reference comparisons repeat 10 distinct
corpus/step positions; all top1 choices match and regrets are zero. Maximum
bidirectional KL is 1.830412e-6, relative L2 is 3.051437e-4 and absolute logit
difference is 0.006623626, all on prose continuation2. Endpoint-only maxima are
KL 1.744805e-7, relative L2 1.387621e-4 and absolute 0.002601877. All 10 scalar
A2/A1 comparisons have zero difference; hashes are stable within each variant
and position. The code timing drift (1.88% GPU / 2.16% wall) is retained and is
small compared with its paired savings. This is a short default-path/singleton
confirmation, not a repeated postpromotion 4K or broad quality campaign.

The loader again reconciles 851 direct copies / 7,887,374,336 logical bytes with
zero conversions. Capacity133 session pricing is 193,593,344 bytes, with scratch
84,869,120 bytes and per-call IDs 16,384 bytes; the 2 GiB reserve is retained. These
remain admission bounds, not measured live-session allocation deltas.

### Source-binding chronology

- `leaf.jsonl`: historical prepromotion source; forced Scalar/MMA leaves.
- `model.jsonl`: historical prepromotion source; forced Scalar/MMA whole128/4096.
- `production-129.jsonl`: historical postpromotion selector/wrapper and revised
  harness, before the family-scope fix; forced Scalar versus production `None`,
  whole129/chunk128.
- Current family-scope correction: broad release/debug validation and independent
  review complete; no new whole packet, as the reviewer found a repeat unnecessary.

The leaf/main whole selector and matrix-wrapper hashes match each other and
differ from the historical production129 confirmation. Scalar/MMA shader and
metallib hashes match across those three recorded builds. The main whole and
historical production129 confirmation also retain
identical loader, residency, prefill and session-pricer bindings. Historical
headers saying production remained scalar describe their measurement build;
they have not been rewritten to claim default-path execution after promotion or
execution of the later family-scope fix. These comparisons are between retained
packet headers, not between a packet and current source.

## Historical prepromotion whole-prefill evidence

`model.jsonl` is a byte-for-byte copy of
`/tmp/saluki-iq2-xs-mma-model-20261008.jsonl` (650,449 bytes):

```text
SHA256 7e437a47e470ff024903eff05eb8ec0599c6a0419f2b4bbae19d0dd93501aa27
```

One copied all-native Saluki model on Apple M4 Max; technical prose and Rust
code-review streams, encoded by the artifact tokenizer using `encode(false)`.
Each stream retains 4,100 natural token IDs plus full-text/token hashes; no
padding, corpus repetition or chat template is used. Source inputs are
`docs/GLM53-FLASH-PLAN.md` and the GLM backend/rendering Rust files. The tokenizer
API correction preceded the measured build; its source hash is in the header.

Each stream runs fresh 128 and fresh 4096, with warm A/B and two ABBA rounds.
A forces native scalar XS GEMM; B forces native F32 XS MMA. Actual chunk rows
are 128 and 1024 respectively (one or four chunks). Every trajectory continues
its actual prefill session through four common teacher-forced corpus tokens;
there is no restore/reset between its prefill and continuations. Weight loading
and session/scratch construction are outside timing; resident weights are reused.
The owner's reported 656-second packet duration is not a model latency sample.

The packet contains **32 measured prefills, eight warm prefills, 160 continuation
steps, 200 finite output records and 140 comparisons**. All 32 measured prefills
have finite positive aggregate GPU times. The existing ordinary API sums command
timestamps, including the final norm/head command; individual raw command
timestamps are not exposed, so their separate validity cannot be audited here.
Although the API is named `prefill_tokens_with_multi_hidden_profiled`, it is the
same implementation used by the ordinary wrapper: there is no diagnostic phase
splitting. Census is enabled only for the eight warm calls.
Leaf and whole headers have identical selector, matrix-wrapper, scalar/MMA shader
and metallib hashes, binding the whole result to the qualified leaf implementation.

**Whole wall time includes ordinary per-call IDs allocation**, encoding, waits
and final logits readback. It excludes session/scratch construction, census and
continuations, not allocation generally. These are resident ordinary-prefill
measurements, not placement-cold or HTTP TTFT.

### Whole timing medians

Four measured samples per arm per cell; savings are computed from the arm medians.

| Stream | Tokens | GPU A / B ms | GPU saving | Wall A / B ms | Wall saving |
| --- | ---: | ---: | ---: | ---: | ---: |
| Technical prose | 128 | 1038.669 / 644.946 | 37.91% | 1042.659 / 648.858 | 37.77% |
| Code review | 128 | 1081.466 / 688.168 | 36.37% | 1085.558 / 692.482 | 36.21% |
| Technical prose | 4096 | 38985.761 / 23361.167 | 40.08% | 39012.626 / 23389.031 | 40.05% |
| Code review | 4096 | 39155.356 / 23415.997 | 40.20% | 39182.457 / 23443.718 | 40.17% |

All paired results follow. Each pair column lists B1 versus A1, then B2 versus A2;
positive percentages mean lower latency. Drift columns are A2/A1, not savings.

| Stream / tokens | Round | GPU pair savings | Wall pair savings | A2/A1 GPU | A2/A1 wall |
| --- | ---: | --- | --- | ---: | ---: |
| Prose128 | 1 | 37.901%, 37.912% | 37.761%, 37.777% | 1.000008 | 1.000128 |
| Prose128 | 2 | 37.898%, 37.382% | 37.752%, 37.239% | 1.010135 | 1.010113 |
| Code128 | 1 | 35.946%, 37.314% | 35.784%, 37.177% | 1.007624 | 1.007618 |
| Code128 | 2 | 36.315%, 35.610% | 36.154%, 35.484% | 0.999823 | 0.999936 |
| Prose4096 | 1 | 41.969%, 40.064% | 41.940%, 40.035% | 1.008951 | 1.008911 |
| Prose4096 | 2 | 40.092%, 39.940% | 40.060%, 39.913% | 1.009930 | 1.009973 |
| Code4096 | 1 | 40.819%, 39.977% | 40.791%, 39.946% | 1.001301 | 1.001250 |
| Code4096 | 2 | 40.026%, 39.864% | 39.997%, 39.837% | 1.009765 | 1.009846 |

All 16 GPU pairs and all 16 wall pairs improve. Scalar A2 drift is at most
about 1.014%, far smaller than the observed gain. Prose4096's first pair is
somewhat stronger than later pairs; retain it without making41.97% the headline.
The measured 4K saving is about 15.6–15.7 seconds, or 1.67x throughput for this fixed
token count. Do not add the leaf savings to these whole savings or infer an exact
XS stage share from the observed reduction.

### Whole dispatch and numerical audit

The realized model has 39 eligible XS projections. Each B128 prefill substitutes
39 calls; each B4096 substitutes 156. All A counts are zero. Eight warm censuses
confirm scalar-to-MMA replacement and unchanged other kernel/count/concurrency
topology. Timed scopes total 1,560 substitutions; warm scopes 390. All 160 ordinary
decode continuations have zero substitutions, preserving the GEMV route.

There are 100 B-versus-reference logit comparisons (warm B plus both measured B
arms), and 40 scalar A2/A1 repeat comparisons. The 100 comparisons repeat 20 distinct
corpus/length/step positions; they are not 100 independent quality examples.
All top1 choices agree and both choice regrets are zero. Full logit hashes are
stable within each variant/position across warm and measured repeats. A2/A1
logit differences and both KL directions are zero throughout.

| Position after prefill | Maximum bidirectional KL | Maximum relative logit L2 | Maximum absolute logit difference |
| --- | ---: | ---: | ---: |
| Endpoint | 7.536860e-8 | 3.202715e-4 | 0.004953861 |
| Continuation1 | 8.003690e-7 | 1.230533e-3 | 0.024718046 |
| Continuation2 | 4.109420e-8 | 1.478897e-4 | 0.005227566 |
| Continuation3 | 2.612037e-6 | 3.699422e-4 | 0.008121014 |
| Continuation4 | 2.635750e-7 | 2.509747e-4 | 0.010776043 |

Column maxima can come from different cells. Largest KL occurs on prose128
continuation3; largest relative L2/absolute difference occurs on prose4096
continuation1. These are finite arithmetic differences, not model bit identity.
No task-quality verdict follows from matching choices at 20 positions. Four
teacher-forced steps exercise the handoff but do not establish free-running or
long-context quality; full persistent-state tensors are not compared. These
limits did not reveal a dense numerical blocker in that packet. The subsequent
cross-family failures, fix and successful reruns are recorded separately above.

### Native storage and session pricing

All 851 source bindings reconcile to **851 direct copies / 7,887,374,336 logical
bytes (7.345690 GiB)**, with zero conversions, derived weights, views, aliases or
tail fallbacks. All source and actual dtypes agree, including 39 native XS tensors;
the 353 source F32 tensors are direct source storage. The summarizer checks the
binding totals against the loader ledger. No expanded baseline is allocated.

All 42 admissions succeed. Each of 40 trajectories preserves the 2 GiB dynamic
reserve and prices session, scratch, per-call IDs and diagnostic CPU storage:

| Prefill / capacity | Session upper bytes | Scratch upper bytes | IDs upper bytes |
| --- | ---: | ---: | ---: |
| 128 / 132 | 193,593,344 | 84,836,352 | 16,384 |
| 4096 / 4100 | 453,640,192 | 982,761,472 | 16,384 |

These are **modeled admission bounds**. Footprints are sampled after loading and
after each session/scratch pair is dropped; there is no isolated live-session
allocation delta, so this packet cannot supply a new priced-versus-measured
session comparison. The earlier capacity135 regression remains historical.

## Retained leaf evidence and method

`leaf.jsonl` is a byte-for-byte copy of
`/tmp/saluki-iq2-xs-mma-leaf-20261008.jsonl` (361,517 bytes):

```text
SHA256 1d733610e0116af45d2979920fc1e46b43a7cfb136bb65f1213165bad7508907
```

Apple M4 Max; actual Saluki IQ2_XS compressed weights; fixed synthetic F32
activations. A is the incumbent native scalar GEMM; B requests the test-only
strict F32 MMA path. Both share the same compressed weight/input/output buffers.
There is no expanded F32 weight bank or full-model load. The representatives are
`blk.22.ffn_gate.weight` [K5120,M17408] and `blk.9.ssm_out.weight` [K6144,M5120].

Each representative runs N1/2/8/31/32/33/127/128/129/512/1024. Every cell has
untimed warm A/B census witnesses followed by two A1/B1/B2/A2 rounds. Tables use
the median of four measurements per arm; speedup is median A divided by median B,
not the median of paired ratios. GPU is a completed-command timestamp interval.
Wall includes command creation, encoding, commit and completion checking.
Allocation, output initialization, hashing, CPU references and readback are outside
the timed interval. Timed arms have census disabled and are not profiled.

The retained packet has 522 records, a final completion, **176 timed attempts,
44 untimed witnesses and 220 output records**. All 176 timed attempts have valid GPU
timestamps and no recorded errors. All three admissions succeed through the
normal production lease/memory path. Their reason records that the process-budget
signal was unavailable/omitted; it does not mean unlimited memory was assumed.
Source, shader, codec, GGUF reader and metallib hashes are embedded in the header;
the owner corrected source bindings before the measured build. Artifact stamps,
header hash, representative payload hashes and sampled-row hashes are retained.
Only selected payloads are hashed, not the entire artifact.

## Initial timing results

| Representative | N | GPU A / B ms | GPU speedup | Wall A / B ms |
| --- | ---: | ---: | ---: | ---: |
| FFN gate | 32 | 4.661 / 0.595 | 7.84x | 5.307 / 1.272 |
| FFN gate | 128 | 18.787 / 2.312 | 8.13x | 19.538 / 3.076 |
| FFN gate | 512 | 90.548 / 20.505 | 4.42x | 91.151 / 21.120 |
| FFN gate | 1024 | 192.141 / 30.122 | 6.38x | 192.749 / 30.728 |
| SSM out | 32 | 1.546 / 0.222 | 6.96x | 2.144 / 0.884 |
| SSM out | 128 | 6.426 / 0.828 | 7.76x | 7.107 / 1.529 |
| SSM out | 512 | 28.322 / 3.257 | 8.70x | 29.068 / 4.004 |
| SSM out | 1024 | 69.554 / 6.576 | 10.58x | 70.330 / 7.343 |

All 80 GPU pairs and all 80 wall pairs at N>1 improve (B1 versus A1 and B2
versus A2, two rounds). The smallest paired reductions are 74.82% GPU and
55.19% wall. The summarizer emits all 22 cells, including ragged widths, rather
than selecting only these primary widths. These are two synthetic-input leaves,
not all 39 native XS projections and not model-stage shares.

Dispatch witnesses contain **24 scalar kernels and 20 MMA kernels**: all 22 A
witnesses are scalar; 20 B witnesses at N>1 use MMA, while both N1 B witnesses
stay scalar. Timed scoped counters record 80 MMA substitutions; these counters
are not a timed dispatch census. Expected substitution counts agree everywhere.

## Controls and anomalies

- **N1 is unchanged scalar GEMM in both arms**, not the optimized GEMV decode
  path. Gate GPU medians are 2.823/2.778 ms and wall 2.998/3.004 ms. SSM-out
  GPU medians are 1.678/1.973 ms and wall 1.875/2.176 ms. The latter looks like
  a candidate regression despite identical kernel selection and output hashes:
  its A2/A1 GPU ratios are 0.796 and 0.875, and all attempts drift from roughly
  2.886 to 0.927 ms. This control exposes timing instability; do not infer an
  N1 implementation effect or decode-throughput result from the pooled medians.
- Gate N512 candidate GPU spans **17.722–22.945 ms**; scalar spans
  89.303–91.117 ms. Its weaker 4.42x median gain and non-linear scaling against
  N128/N1024 deserve a targeted repeat or later leaf attribution. This packet
  does not establish whether clocks, cache, scheduling or kernel behavior cause it.
- Gate N31 scalar A2/A1 GPU ratios are **0.845 and 0.867**; scalar decreases
  from 7.517 to 4.646 ms across the two rounds. Gate N8 also has an 11.8%
  second-round A2/A1 increase. The very large N8 ratios (21.62x gate, 18.28x
  SSM-out) should not be extrapolated to wider matrices or model speedups.
- Within each representative/N/variant, output hashes are stable across warm
  and measured repetitions. Timing drift therefore does not coincide with
  observed output instability. The mechanism of timing drift remains unproven.

The summarizer prints both GPU and wall A2/A1 ratios in JSON, individual paired
savings and within-arm GPU ranges. Its >10% drift / >25% range annotations are
descriptive flags, not qualification or promotion thresholds.

## Numerical scope

All 220 full-output records are finite. Sampled CPU references decode the real
first/middle/last weight rows using the canonical codec and accumulate F64 dots
at first/middle/last token positions (three unique points at N1, six at N2,
nine at larger widths). These are sampled references, not a full F64 matrix.

| Comparison | Maximum absolute difference | Maximum relative L2 |
| --- | ---: | ---: |
| Scalar A versus sampled CPU codec/F64 | 3.297407e-6 | 2.390569e-6 |
| Candidate B versus sampled CPU codec/F64 | 2.389697e-6 | 1.482806e-6 |
| Full candidate B versus same-round scalar A1 | 5.149841e-5 | 1.576166e-6 |
| Full scalar A2 versus same-round A1 | 0 | 0 |

Column maxima need not occur in the same cell. The full B/A1 absolute maximum
is SSM-out N512; its relative-L2 maximum is SSM-out N2. The sampled A maxima
occur at SSM-out N512 and sampled B maxima at SSM-out N33. A is a comparator,
not an exact oracle; slightly smaller sampled B error does not establish superior
model quality. N1 full A/B hashes agree. No new bitwise or numerical threshold
policy is imposed. Model logits and continuations are outside this leaf packet;
the separate whole evidence above supplies that qualification at128/4096.

## Pricing and validation

Final owner-reported validation:

- Broad `iq2_xs`, release with Metal API validation: **22 passed, 0 failed,
  3 ignored**.
- Broad `iq2_xs`, debug with Metal API validation: **22 passed, 0 failed,
  3 ignored**. Both broad runs include the three previously failing executing
  DeepSeek tests and the new scope regression; exact tolerances remain unchanged.
- Ignored physical-bindings test: separately executed in debug, **passed**.
  The other two ignored tests are the leaf/whole packets retained here and
  executed earlier, not additional runs counted in the broad totals.
- Promoted dense tests: **12 passed** under Metal API validation, also **12
  passed** in debug with hazard tracking (overlapping coverage, not additive).
- Session-pricing CPU tests: **4 passed in debug and 4 passed in release**.
- Whole-packet CPU chunk-count test: **passed**. Final non-test cargo check:
  **passed**.

Independent review approves the family-scope fix and confirms unchanged dense
arithmetic (`scalar_only=false`) and unchanged shader/metallib; no whole repeat
is required. Retained packet source bindings remain historical.
Those test results are owner-reported; their logs are not embedded in `leaf.jsonl`.
Timing itself ran without the API validation layer, as required by the harness.

The session-pricing regression uses the actual 165-buffer inventory with an
injected page-rounded device quote. At capacity135 it models 193,593,344 bytes,
covering the historical 193,314,816-byte allocation delta. This is a **modeled
upper bound**, not a new measured session allocation. The dynamic 2 GiB reserve
is unchanged. Leaf admission byte quotes likewise are bounds, not peak RSS.

## Reproduction

Exact leaf invocation (the output path must be new; the retained original already
exists). The artifact defaults to the Saluki path recorded in the JSONL:

```sh
env -u MTL_DEBUG_LAYER QWEN_METAL_LEASE_WAIT=1 \
  IQ2_XS_MMA_CANDIDATE=mma IQ2_XS_MMA_ROUNDS=2 \
  IQ2_XS_MMA_OUT=/tmp/saluki-iq2-xs-mma-leaf-repeat.jsonl \
  cargo test --release -p qwen-llm --lib \
  metal_forward::tests::iq2_xs_mma::iq2_xs_mma_packet \
  -- --ignored --exact --nocapture --test-threads=1
```

Set `IQ2_XS_MMA_GGUF` only to override the artifact. The 12-test kernel filter
is `metal::iq2_xs::tests::`; use `MTL_DEBUG_LAYER=1` and the normal lease for
kernel validation. The four CPU-test filter is `qwen_queue2::pricing_tests::`.
No tests, builds or GPU execution were performed while retaining this evidence.

```sh
uv run --no-project docs/bench/2026-10-08-iq2-xs-mma/summarize.py
uv run --no-project docs/bench/2026-10-08-iq2-xs-mma/summarize.py --json
uv run --no-project docs/bench/2026-10-08-iq2-xs-mma/summarize.py docs/bench/2026-10-08-iq2-xs-mma/model.jsonl
uv run --no-project docs/bench/2026-10-08-iq2-xs-mma/summarize.py docs/bench/2026-10-08-iq2-xs-mma/model.jsonl --json
```

The dependency-free script accepts leaf or whole JSONL paths. It checks
completion, ABBA coverage, timing validity, output coverage, admissions and
substitution counts; malformed/incomplete packets are not silently averaged.
Successful summarization is an integrity result, not a promotion decision.

Whole reproduction, using a new output path and the corrected prebuild harness:

```sh
env -u MTL_DEBUG_LAYER QWEN_METAL_LEASE_WAIT=1 \
  IQ2_XS_MMA_MODEL_ROUNDS=2 IQ2_XS_MMA_MODEL_CHUNK=1024 \
  IQ2_XS_MMA_MODEL_OUT=/tmp/saluki-iq2-xs-mma-model-repeat.jsonl \
  cargo test --release -p qwen-llm --lib \
  metal_forward::tests::iq2_xs_mma_model::iq2_xs_mma_model_packet \
  -- --ignored --exact --nocapture --test-threads=1
```

CPU chunk-count filter:
`metal_forward::tests::iq2_xs_mma_model::iq2_xs_mma_model_cpu_chunk_counts`.

Postpromotion production/singleton confirmation (new output path required):

```sh
env -u MTL_DEBUG_LAYER QWEN_METAL_LEASE_WAIT=1 \
  IQ2_XS_MMA_MODEL_CANDIDATE=production IQ2_XS_MMA_MODEL_WIDTHS=129 \
  IQ2_XS_MMA_MODEL_ROUNDS=1 IQ2_XS_MMA_MODEL_CHUNK=128 \
  IQ2_XS_MMA_MODEL_OUT=/tmp/saluki-iq2-xs-production-129-repeat.jsonl \
  cargo test --release -p qwen-llm --lib \
  metal_forward::tests::iq2_xs_mma_model::iq2_xs_mma_model_packet \
  -- --ignored --exact --nocapture --test-threads=1

uv run --no-project docs/bench/2026-10-08-iq2-xs-mma/summarize.py docs/bench/2026-10-08-iq2-xs-mma/production-129.jsonl --json
```

The summarizer takes widths/rounds/chunking and candidate scope from each header,
including both revisions of model schema v1. Revision2 checks singleton chunk/GEMV
witness counts and per-record selector scopes. Revision1 retains its historical
forced-MMA interpretation; absent singleton observations are not invented.
