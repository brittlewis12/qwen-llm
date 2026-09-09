# Muse resident optimized math delivery

## Delivered scope

`5b23b814` exposes existing runtime options through two independent startup-only
variables: `QWEN_SERVE_MUSE_MATRIX_PREFILL` and `QWEN_SERVE_MUSE_SPLIT_DECODE`.
Unset/0 disables each,1 enables, invalid values fail before Metal initialization.
The CLI-only math variables do not affect serve. Defaults, Q8_0/Apple M4 Max
eligibility, model-context/capacity admission, native ATEM, sampling, and stop
handling remain unchanged. Split scratch is already priced by the runtime.

Matrix prefill selects the delivered tiled fullN128 kernel and online packed
remainders. The resident runner already uses generated-token execution for split
decode. Live-prefix reuse remains a separate opt-in with immutable math options
for the loaded model's lifetime. Exact token matching does not imply bitwise
warm/reset arithmetic when packing boundaries or generated history change.

## Numerical and lifecycle composition

`ee374e03` core optimized warm/reset test PASS51.56s. Six cohorts compute
1/1/17/129/1153/16 tokens after reusing1168/1151/1040/1040/0/1154 respectively.
These cover exact retry, shorter prompt, changed suffixes across16/N128, unrelated
history, and a split-generated token subsequently becoming packed prompt history.
All active KV is compared, retained prefix immutable; one reference-selected
continuation and its newly written KV row are checked per cohort.

Established endpoint cosine>0.99999/RMS<0.002/maxabs<0.1 and continuation
>0.99999/<0.006/<0.3 gates pass. Worst endpoint maxdelta0.010140061, activeKV
RMS0.0000764883 and new generated-row KV RMS0.000343223. All six endpoint top1
comparisons happen to agree; those are observations, not an independent rollout.
KV gates remain aggregate cosine>0.9999/RMS<0.01, not per-element bounds.

The actual resident-backend packet PASS21.62s on `09d70c1c`: native1158-token
prompt/four outputs witnesses468 tiled dispatches and156 split main plus156
reductions. Exact retry reports1157 cached tokens and preserves emitted bytes.
Authored followup warm/reset outputs and consumed IDs agree. Capacity rejection
preserves history. A second-checkpoint abort after128 prefill positions and a
decode-piece abort both clear history; recovery is cold. A separate short
temperature0.7/seed99 retry cohort agrees, not a general sampled-exact claim.

Retained attempt01 failed its dispatch assertion after6.97s: the census followed
the last PSO lookup, so preloading both split PSOs mislabeled main dispatches as
reduce. `09d70c1c` explicitly tags both bindings without changing pipeline objects,
arguments or dispatch order. Attempt02 records both expected counts. This is an
observer repair, not evidence that the first attempt omitted split execution.
Review also caught a zero-prefix hash panic and a missing129-row suffix case
before execution; both were repaired without changing numerical gates.

## Actual HTTP packet

Clean release binary from `09d70c1c`, SHA256
`8bbb2ba5c8875aef74d6f2b58ed4ec6af02e820c01f362d2125c1168bd16c785`.
Serial, loopback-only processes, capacity2048,17-output requests, temperature0,
reasoning high. Body `seed:42` is ignored by the existing HTTP parser; effective
backend seed is its default42, and the requests are greedy. Startup-to-ready is
separate. These are first requests after resident startup, NOT OS-cold trials.

| Request | Input / cached tokens | Client wall ms | Prefill ms | Generation loop ms / forwards |
| --- | --- | ---: | ---: | --- |
| Original math, first | 1158 /0 | 33903.013 | 32594.8 | 1301.4 /16 |
| Optimized math, first | 1158 /0 | 7165.973 | 6156.0 | 1004.8 /16 |
| Optimized identical SSE retry | 1158 /1157 | 1095.618 | 81.8 | 1007.1 /16 |
| Authored followup, warm | 1198 /1159 | 1795.621 | 782.1 | 1007.2 /16 |
| Followup recovery after detected abort | 1198 /0 | 7894.798 | 6878.6 | 1009.2 /16 |
| Followup retry | 1198 /1197 | 1101.399 | 82.4 | 1013.4 /16 |

Default and optimized first responses have equal semantic output after removing
generated item IDs. Optimized nonstream/SSE retry and warm/reset/retry followup
outputs agree. All complete requests return200 with17 output tokens and token-limit
incompletion. SSE has exactly one terminal response and `[DONE]`. The retry's first
nonempty model delta arrives at277.322ms and is reasoning; no output-text delta
occurs in this short response. Heartbeats are not counted as model output.

While the optimized first request is active, another connection receives503 and
`Retry-After:1`. A separate256-output streaming request is reset after its first
model delta; the server logs Broken pipe, then recovery reports zero reuse and
matches the prior warm followup. This checks detected generation cancellation,
not immediate interruption of a running GPU command. Backend history publication
still precedes final HTTP framing: a late framing/transport failure after backend
completion may retain valid consumed history. That existing behavior is documented,
not changed or claimed covered by the earlier-generation disconnect witness.

Startup-to-ready386.719ms original /391.415ms optimized. The original process is
deliberately given CLI-only math flags=1 yet logs serving math=false, proving
namespace isolation. Both owned servers exit130 through normal SIGINT teardown.
No parent lease is layered around a CLI process; production lease ownership stays
with the server. CPU serve regressions:109 PASS,12 ignored in0.35s.

The33.90s versus7.17s client observations are NOT a balanced speedup qualification:
original runs first, and only optimized receives the busy probe. They demonstrate
product reachability and complete HTTP flow, complementing existing kernel and
whole-phase authorities. Decode telemetry counts17 generated/sampled tokens; the table
separately reports16 actual forwards. Warm timing includes avoided work, not a
claim about faster fresh inference.

Raw evidence: `target/profiles/muse-live-prefix/{optimized-reuse-01*,serve-math-01*,serve-math-02*,build-serve-math-*}`
and `target/profiles/muse-serve-math/{events-01.jsonl,summary-01.json,*.raw,*-01.log,cpu-tests-01.log}`.

## Next leverage

Further PV polishing is held after positive but below-budget F32/half-P screens.
The next independent research question is actual decode FFN gate/up/product
distribution and block-aligned removable work. Small gates alone do not bound
products; post-up observations do not establish avoidable up computation. Preserve
the measured65% FFN share as a stage share, not a sparsity or DRAM roofline claim.
