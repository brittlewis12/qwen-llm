# Three-Way GLM Quality Comparison

## Status

Completed: all three verified artifacts ran all eight frozen cases, and all 24
responses ended naturally. Two fresh blind reviews are complete. The pilot does
not demonstrate a quality advantage for Q4; Maya and stock IQ3 remain competitive.
This is not a universal ranking or independent numerical qualification.

The direct downloader uses `huggingface_hub` 2.2.0 and `hf_xet` 1.7.0, with
`HF_XET_HIGH_PERFORMANCE=1` and four file workers. Subsequent acquisition pins
those observed versions. Raw transfer logs remain private.

## Results

| Recipe | Reviewer A / B, each out of 32 | Exact-fact cases / 8 | Strict JSON cases / 8 | Generation tok/s | Prefill tok/s | Eight-request total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Maya-M | 23 / 23 | 6 | 7 | 47.25 | 499.91 | 177.28 s |
| Stock IQ3 | 23 / 24 | 5 | 5 | 49.14 | 506.06 | 169.16 s |
| Stock Q4 | 22 / 21 | 5 | 7 | 42.42 | 485.25 | 179.22 s |

Rates are weighted totals: emitted tokens divided by summed native generation
time, and input tokens divided by summed native prefill time. Output-token counts
include stop tokens. These are real generated responses, not fixed-token compute
benchmarks. Outputs differ in length and content: Maya produced 6316 tokens,
IQ3 6195, and Q4 5696. Q4's shorter outputs mostly offset slower generation in
whole-request time; do not equate the rate difference with an identical increase
in end-to-end latency. Checksum, startup, and acquisition time are excluded.

Every arm received the same 21715 rendered input tokens, case by case. The long
cases are 9654 and 9810 input tokens; their native prefills took approximately
18.5-19.2 seconds across all three models. All actual selected prefill widths
were 2048, all cache-reuse counters were zero, and all three arms used exactly
the committed runner/checker/prompt bytes. Swap usage was zero before and after
each arm. Existing compressor occupancy was nonzero (approximately 146-158 MiB)
and stayed flat or declined; the cumulative compression count did not increase
during any arm. These are endpoint observations, not continuous peak-memory
measurements. No arm restarted or exhausted its token budget.

Reviewer A used the `astra` CX profile; reviewer B used `sol`, in independent
fresh read-only sessions with reversed candidate presentation order. Both full
24-record grades were saved and validated before the private map was opened.
They agreed on 22 of 24 numerical scores; the remaining disagreements concern
severity, not different identified bugs. Scores are retained separately.

| Case | Maya A/B | Stock IQ3 A/B | Stock Q4 A/B |
| --- | ---: | ---: | ---: |
| Single-pass lookahead | 2/2 | 2/2 | 2/1 |
| Shared-load cancellation | 2/2 | 2/2 | 2/2 |
| Generation-fenced cache | 4/4 | 2/2 | 3/3 |
| SQL NULL/bag semantics | 4/4 | 4/4 | 3/3 |
| Framed-log recovery | 4/4 | 4/4 | 4/4 |
| Incremental UTF-8 lines | 1/1 | 2/3 | 2/2 |
| Long archive recovery | 4/4 | 4/4 | 4/4 |
| Long rollout/base resolution | 2/2 | 3/3 | 2/2 |

Examples from static review, not executed-code tests:

- Q4's lookahead class does not cache exhaustion; its rollout answer incorrectly
  substitutes L41 for the explicitly named L40 base, producing wrong settings.
- Maya's UTF-8 implementation mixes a bytes buffer with decoded strings and
  would raise TypeError. Its rollout facts are substantively on track but the
  final answer violates the requested JSON structure.
- IQ3's cache implementation can raise KeyError on a normal later hit because
  an initial successful load never inserts the corresponding generation key.
- All three have material issues in the shared-load cancellation task. None is
  a blanket correctness winner merely because its model file is larger.

Exact-fact checks are deliberately strict about types and strings. Some failures
are formatting-only, such as quoted numbers or an explanatory record-ID string.
The checker tolerates and reports an outer JSON fence for fact extraction, but
the strict-JSON column excludes fenced answers. These counts are not general
reasoning accuracy, and the blind scores are judgments rather than test execution.

**Decision:** this eight-case, low-reasoning pilot does not justify switching the
default to Q4. It uses roughly 72-77 GiB more main-weight storage (74-78 GiB
larger files) than the smaller recipes and generated about 10-14% fewer tokens per second on this task mix,
without a demonstrated quality gain. Keep Maya as the incumbent; stock IQ3 is
competitive but not a clearly established quality upgrade. Different tasks,
reasoning effort, or repeated sampling could change that assessment.

`results/` contains final answers, exact checks, usage/timing summaries, both
blind packets and review prompts, validated grades, the subsequently revealed
map, and calculated metrics. Reasoning traces, full SSE streams, and server logs
remain in the private run directory. Private terminal-response hashes are kept
in the public summaries for traceability. No hostnames or private connection
identifiers are published. The worker is idle after owned-server cleanup.

## Scope and identity

Compare Maya-M IQ2_S, Unsloth UD-IQ3_XXS, and Unsloth UD-Q4_K_XL using the same
native qwen binary, machine, prompts, chat settings, and output budgets. The
binary is pinned to `d7b2a05b5ad1b9706486c69718a1e11dbbc58937`; its SHA-256 and
every shard's size/full SHA-256 are in the three artifact manifests.

Maya is a calibration/quantization recipe, **not a fine-tune**, according to its
[pinned model card](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF/blob/f16ee4ba50d8fd2bb726ea1862ca6ca14ba188da/README.md).
It declares quantization from Z.ai's FP8 release with per-expert statistics and
error-feedback rounding. The two stock quantizations are pinned to the same
Unsloth repository revision. This is a comparison of complete quantization
recipes in this engine, not a controlled bit-width-only experiment. Exact
upstream source-weight revision equality across publishers is not established.

Header inspection finds matching 1,412 tensor names/shapes and identical
tokenizer/chat-template metadata. All have 45 main blocks plus one NextN block,
4096 hidden width, and 288 experts with top-8 routing. NextN is not executed.
Filename quant labels do not describe all tensors; stock UD-IQ3_XXS, for
example, has no tensors actually encoded as IQ3_XXS.

The stock manifests select their publisher-provided rewritten first shards.
These contain metadata only: the architecture namespace becomes `glm5-next`
and the shared-index NextN metadata flag is explicit. Payload shards are
unchanged. Entrypoints retain the standard `.gguf` split filename alongside
their siblings, even when backed by the publisher's `.gguf_file` metadata file.

| Recipe | Complete files, bytes | Main tensor bytes | NextN tensor bytes |
| --- | ---: | ---: | ---: |
| Maya-M | 116003563168 | 112401648888 | 3592392832 |
| Stock IQ3 | 120367571840 | 117558078712 | 2799972480 |
| Stock Q4 | 199707321472 | 195118640376 | 4579160192 |

Q4's roughly 181.72 GiB of main weights is not its total runtime footprint.
Native live memory admission stays enabled. No wired-limit or device-name
overrides are authorized by this protocol.

## Frozen cases and settings

`cases.json` freezes all eight cases, references, and rubrics before generation.
`cases.py` deterministically generates them and implements partial exact-fact
checks. The questions cover single-pass lookahead, shared-task cancellation,
generation-fenced caching, SQL NULL/bag semantics, framed-log recovery,
incremental UTF-8 decoding, and two long incident archives.

Each long archive puts its authoritative record between 24 preceding and 24
following draft notes, with an early base and contradictory stale tail. Text
length is not a token-count claim: actual rendered input tokens come from each
request's native usage. These are synthetic engineering cases, not private user
workloads or a broad benchmark suite.

- Native `qwen serve`, one resident model per arm, one request at a time.
- Identical case order; temperature 0, seed 0, top-p 1, top-k 0, min-p 0.
- GLM low reasoning, Fast prefill, explicit Half arithmetic, no drafter/MTP.
- 32768 context capacity; per-case output budget 4096 or 6144, frozen in JSON.
- Prefix reuse and Fast snapshots disabled; snapshot budget zero; no disk tier.
- Native Metal lease enabled. Measured arms must not overlap weight transfers
  or other deliberately launched GPU work.
- Model and binary SHA-256 validation before loading, stat guards during/after;
  full checksum reads warm the filesystem cache. Loading is not a cold-disk test.
- Startup deadline 600 seconds; request deadline 900 seconds. Cleanup signals
  only the owned server process. No failed arm silently changes its settings.
- Token-limit exhaustion is a recorded quality outcome, not an infrastructure
  error or a reason to continue generation with extra budget.
- Narrowly identified malformed model stops are retained as quality outcomes,
  not dropped along with their arm. The failed case is not retried; the owned
  server restarts before the next case with identical settings. Unknown server,
  transport, or integrity failures still abort and require investigation.

Native admission may select a smaller prefill width. The runner records startup
`prefill_rows_max`, capacity, and precision separately from the actual selected
width in `serve: glm5_next prefill plan rows=...` diagnostics. A missing plan
record is explicitly unavailable, not inferred from the maximum. Policy-selected
width differences are accepted as part of each complete runtime configuration;
identical Fast/Half settings are not a claim of identical batch widths or a pure
quantization-error experiment.

Acquisition and model order are fixed before generation: finish the direct Q4
download, evaluate Maya, evaluate Q4, acquire stock IQ3 on the worker, then
evaluate IQ3. Each model gets a fresh server; each request starts without reused
context. This is not counterbalanced hardware timing. The choice prioritizes
the user's requested larger-quant comparison while keeping downloads outside
measured inference windows. Runner/checker source hashes are recorded per arm.

The native SSE stream, terminal response, request, event timestamps, server
logs, and before/after memory observations are retained privately. Cache
counters must be explicitly zero in both HTTP and server telemetry. The runner
separates first emitted content (reasoning included), first visible answer,
whole request latency, native prefill/decode timing, and token counts.
For malformed model termination, unavailable token counts and native timings
remain null, not zero. Raw failure evidence and client wall time remain recorded;
server restart/startup cost is separate from the following request.

## Preregistered grading

Exact fact checks are reported separately as the number of cases with every
requested fact correct. Fact presence, valid JSON, or code presence alone is
not a full quality pass. Generated code is not automatically executed.

Two independent fresh reviewer sessions receive only the frozen task/rubric and
anonymous final answers. Candidate labels/order are shuffled independently per
case; model names, reasoning traces, token counts, latency, and recipe-to-label
mapping are withheld. Reviewers cannot consult other result files. Mapping is
revealed only after both grading outputs have been saved.

For each answer, reviewers assign one functional score:

- 0: no usable final answer or unrelated response.
- 1: central implementation or reasoning fails the requested contract.
- 2: partially correct but at least one material edge-case/contract defect.
- 3: correct core behavior, with only minor explanation or presentation defects.
- 4: meets the complete stated contract and explains the relevant edge cases.

Reviewers must cite concrete response evidence and distinguish definite bugs
from unresolved concerns. Report both scores and disagreements rather than
silently averaging away disputes. Show per-case outcomes, aggregate descriptive
scores (maximum 32 per reviewer/recipe), format compliance, and truncations.
Eight cases are eight case-level observations, not dozens of independent trials
because each question asks multiple facts. Do not claim statistical significance
or universal superiority from this pilot.

Latency is reported alongside quality, never included in blind functional
scores. Startup, checksum time, and downloads are separate. The native backend
has not thereby been independently numerically qualified on every M5 artifact;
observed differences belong to the full tested configuration.

## Running

Use private paths and dedicated connections; do not put hostnames or connection
identifiers in public filenames, prose, logs, or commit messages.

```sh
uv run --offline --no-project python -B -m unittest discover \
  -s docs/bench/2026-10-16-glm-quality-comparison -p 'test_*.py'

uv run --script run.py --binary "$QWEN" --model "$MODEL" \
  --manifest "$MANIFEST" --cases cases.json --arm "$ARM" --out "$NEW_OUTPUT"
```

The parent verifies the worker source checkout matches the pinned clean commit
before launch. Output directories must be new. Raw results require privacy
review before publication; the public packet will identify hardware and model
artifacts, not private machines. The preparation checkpoint was `b57ce0b8`,
committed before any scored responses; the results above are the subsequent run.

`worker-queue.sh` performs the source check before each arm, using caller-supplied
paths. `stage.py` creates only canonical shard links, refusing conflicting files;
actual payload integrity is verified by the runner before GPU loading. The queue
must be launched only after the separate Q4 downloader exits successfully.

`blind.py` packages all three completed arms for two fresh reviewers. An arm can
contain incomplete or malformed-model-output outcomes; those are retained, not
excluded from grading. The label mapping stays in a separate private directory
until both reviewer outputs pass `validate_grades` and are saved.

Preparation validation: 46 CPU tests and shellcheck pass. Two separate native
Maya integration diagnostics completed eight tiny requests each, including
token-limit outcomes and cache-isolation checks. They did not use the evaluation
prompts, ran during acquisition, and are excluded from quality/latency results.
The actual diagnostic log confirms startup capacity 32768, maximum rows 2048,
Half precision, and selected rows 2048; parsing this real log also passed.

Frozen prompt-file SHA-256:
`af75387cd0f9c885c95e435953683e2e3cadfe0ec619117e679fd6a617b4b0a7`.

All three measured arms report runner SHA-256
`b0fceee7a3ea98868065e905cc934aea26a302181cb45b96c80e89571d1d502e`
and checker SHA-256
`495c6f5e1840a49c80a425e1022e9eb7693b592539880d9af73a130ac063ef9a`.
Preparation structural contracts passed for tree
`6af7264ef04c72e8410a6717c43f122ab517b372`.

Final evidence validation: 51 CPU tests and shellcheck pass. Result tests
recompute scores, facts, rates, mappings, source hashes, input equality, cache
isolation, and the memory claims from retained artifacts. An independent
publication audit checked the report against the outputs; its compressor
accounting correction is reflected above.
