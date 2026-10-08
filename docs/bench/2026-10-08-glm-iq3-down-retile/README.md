# GLM IQ3_S expert-down retile

Decision: enable SmallCounts for GLM Fast, IQ3_S down, K2048/M4096/E288,
actual command rows 32..=512. Domain and final production-path confirmation
pass, with normal shared Metal lease acquisition throughout.
Exact, other families, IQ4_XS layers, and widths outside this range retain
their incumbent path. This is kernel eligibility, not a context limit.

## Mechanism

Sparse expert buckets waste much of the incumbent M64/N32 tile at short
prompts. The new M128/N16/K32 kernel reuses the established Flash IQ4_NL
layout with canonical IQ3_S decoding from `quant_tiles.h`. It consumes the
same compressed bank and materialized slots; weighted reduction is unchanged.
No full-bank dequantization, new production buffers, or CPU route readback.

SmallCounts sends counts 1..16 to one N16 panel and counts 17..MAX to generic
down. Both dispatches are included in timings. Keeping dense experts on the
incumbent protects against the extra weight traversal of a blanket retile.
The bucket stride remains the actual prompt width, not 16. Both pipelines,
bindings and SIMD32/TG128/TGM requirements are checked before any grouped
dispatch; unsupported capabilities fall back without partially encoding down.

The GLM family owns selection. Shared callers explicitly request incumbent.
Test-only scopes distinguish forced incumbent from no override (production),
with substitution counters; the shader and checked encoder are production code.

## Evidence progression

All timings use the normal production lease and memory admission, release
builds, and no Metal API validation. GPU duration includes execution stalls.
Router selection is production in every arm. Allocation, output diagnostics,
route readback and continuation are outside whole-prefill timing.

1. `leaf-screen.jsonl`: actual inputs/routes from layers 4 and 21 at N128/512,
   replayed into existing slot scratch. N128 natural down saves 25-33%.
   N512 is mixed/noisy; leaf timing alone does not establish model conversion.
2. `whole-screen.jsonl`: repeats leaf controls then ordinary whole-prefill
   ABBA. N128 Blanket/SmallCounts save 8.36%/8.70%; N512 1.68%/3.86%.
   Separate policy blocks drift, so 8.70 versus 8.36 does not rank them.
3. `domain-screen.jsonl`: no capture/leaf setup; two fixed streams, seven
   widths and normal eight-chunk 4K requests. Warm A/B then B1/A1/A2/B2
   reverses the previous order. All 96 prefills complete; all 32 measured GPU
   pairs and all 32 wall pairs improve. The weakest GPU pair saves 3.00%.

| Prompt rows | Qualification GPU reduction | Code-review GPU reduction |
| ---: | ---: | ---: |
| 32 | 9.41% | 10.93% |
| 64 | 8.75% | 10.17% |
| 127 | 8.13% | 10.15% |
| 128 | 9.79% | 10.88% |
| 129 | 8.82% | 9.58% |
| 256 | 7.34% | 10.00% |
| 512 | 4.15% | 3.44% |
| 4096, packed512 | 4.05% | 3.84% |

The default-width/long-request headline is approximately **3-4% lower warm
prefill latency**, not 7-11% universally. The 4K domain packet saves 871/843 ms
GPU, with corresponding ordinary-call wall savings. This is not HTTP TTFT,
cold storage startup, or a decode-throughput claim. Earlier router savings
are already in the baseline and must not be added to these percentages.

`production-screen.jsonl` confirms the guarded default (B has no override)
against forced incumbent A, with the same streams and BAAB method:

| Rows | Qualification GPU reduction | Code-review GPU reduction |
| ---: | ---: | ---: |
| 32 | 10.69% | 10.73% |
| 128 | 8.73% | 10.89% |
| 512 | 4.25% | 4.33% |
| 4096, packed512 | 3.61% | 2.71% |

All 48 prefills and all eight cells pass; all 16 measured GPU/wall pairs
improve. Code-review 4K has drift: paired GPU savings are 1.11% and 4.30%,
so retain its lower 2.71% mean rather than quoting only its favorable pair.
Final 4K GPU savings are 777/593 ms (ordinary-call wall 777/587 ms).
All 96 endpoint/continuation comparisons match: 64 candidate comparisons
and 32 incumbent-repeat controls. Expected topology and timestamps pass.

Predeclared domain screen: valid timestamps in every attempt; both paired
GPU savings positive, both wall pairs nonregressing, mean GPU saving >=2%.
Each width needs both streams. All sampled widths pass. The contiguous
32-512 policy is an engineering inference from samples, not proof of every
interior width. Tiny prompts and wider commands are deliberately unchanged.

## Numerics and controls

Domain comparisons include endpoint logits and four identical teacher-forced
tokens after N128/N512/N4096. All 120 comparisons have zero max-absolute,
relative-L2 and bidirectional-KL difference: 80 candidate comparisons and 40
incumbent-repeat controls. All 240 output hashes agree within corresponding
stream/width/step groups, including warmups. All recorded route IDs/counts
agree. Long route observations cover only the final chunk. These checks
exercise decode handoff, not every persistent state element or general quality.
Bit identity is an observation here, not a new numerical policy for this lane.

All 39 IQ3_S down layers substitute per eligible command; IQ4_XS layers
11, 12 and 44 remain unchanged. Long requests report 312 substitutions.
Every domain admission succeeds; the recorded reason is
`admitted_process_budget_omitted`, not a measurement of process headroom.

Negative controls remain in both leaf packets. Concentrated N128 routes make
Blanket about 30-34% slower; SmallCounts mostly reduces this to dispatch
overhead, not guaranteed zero regression. Empty N512's roughly 1.5 ms is
the incumbent 294,912-group early-exit floor, not generic command-launch cost.
Do not infer a full-model speedup from utilization or isolated empty grids.

Historical leaf/whole prompts prepend `[gMASK]<sop>` twice. Domain/final
prompts use the existing prefix once. Historical paired measurements remain
valid, but domain qualification is not an identical-input repeat. Short code
prefixes mostly cover the request and source comments; the 4K stream includes
substantive backend and renderer source. These are two fixed streams on one
device/artifact, not a broad held-out quality corpus.

## Verification and reproduction

Release test build and non-test `cargo check -p qwen-llm --release --offline`
pass. Domain evidence, arm-selection and family-policy CPU tests pass.
All eight final primitive tests pass with Metal API validation, covering
independent codec/f64 reference, tails, offsets, sparse buckets, immutable
inputs, disjoint count ranges, rejected unsafe bindings, and production
fallback/counter/shared-caller isolation. The independent observer CPU test
also passes. GPU work waited for another owner's long quality evaluation;
no lease was bypassed or unrelated process terminated.

Integration with concurrent main preserves its per-stage Exact attribution and
typed-error changes. Workspace check including tests, release test rebuild,
formatting, five policy/observer CPU tests and five typed-error/HTTP mapping
tests pass after the merge resolution. The measured packets retain their
premerge source bindings; later runs bind any changed backend-derived corpus
and must compare token hashes before claiming identical input.

Build using `cargo test -p qwen-llm --release --offline -j2 --lib --no-run`.
Run only `glm5_next_metal::packed::expert_down::expert_down_packet` with
`--ignored --exact --nocapture --test-threads=1`. The test acquires the lease.
Set `QWEN_METAL_LEASE_WAIT=1`, a new `GLM53_EXPERT_DOWN_OUT`, and:

- `GLM53_EXPERT_DOWN_DOMAIN=1`: full diagnostic domain sweep.
- `GLM53_EXPERT_DOWN_DOMAIN_LONG=1`: also test admitted 4096+4 capacity.
- `GLM53_EXPERT_DOWN_DOMAIN=1 GLM53_EXPERT_DOWN_PRODUCTION=1`: focused
  N32/128/512, plus optional4096;
  B uses no override, A forces incumbent. 48 prefills with long enabled.
- Without DOMAIN, optional `GLM53_EXPERT_DOWN_FULL=both` reproduces the
  historical capture/leaf/whole experiment rather than the production packet.

Unset `MTL_DEBUG_LAYER` for timings. The fixture is
`GLM53_FLASH_UD_IQ3_XXS`, optionally overridden with `GLM53_GGUF`. Source,
metallib, shard descriptors/stamps, token streams, every attempt, admission,
topology, continuation and completion are retained in raw JSONL. No historical
raw evidence is rewritten. Use `uv run summarize.py <packet.jsonl> ...`.

## Next force-rank

1. Flash frontier scheduling on the promoted-router baseline: compare ordinary
   2048+3+2045 against 2048+2048. Old 199/282 ms router-disabled savings are
   not current residuals. Require repeatable whole-call conversion (~1% or more)
   before deeper qualification; preserve numerical/publication diagnostics.
2. GLM ragged absorption/expansion: retile leaves N127/N129 about 65-82 ms
   slower than N128. Isolate these bindings; this is not yet causal MLA
   attribution or evidence of an aligned-4K saving.
3. Flash selected-QSA leaves: historical GSQ layer7 mixer grows from 33.87 ms
   dense to 134.61 ms selected. Attribute projection, selection, QK and value
   before choosing a kernel; inclusive mixer time is not a leaf budget.
4. GLM gate/up and Flash experts by actual dtype cohort. Neither retained Flash
   artifact has IQ3_S down banks; direct transfer of this shader has no coverage.
5. GDN/KDA preparation and state update before recurrence redesign.

For requests wholly beyond the dense frontier, selected QSA outranks scheduling;
for fresh short prompts, expert width coverage outranks it. Existing sibling-
head QSA reuse and Flash IQ4_NL retile are already enabled where qualified.

Independent `cx` review: `01a118ca-c0f6-7c00-a0dd-23ae962883a5`; kernel jam
`01a11528-973c-7e40-ba8c-d4bece18c93b`; GLM harness
`01a11893-3352-79d3-8d8d-8343c0d8f293`; Flash prioritization
`01a11893-3352-75b1-9ec7-2ed940154a10`. Review approved production source and
domain evidence, and caught historical
summary event/schema mismatches fixed without changing raw data.
