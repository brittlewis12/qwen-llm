# GLM solo NextN re-price, 2026-11-12

## Final decision: kill; retain the record

The delivered gate failed: **1.0533x geometric mean**, below the required
**1.15x**. All product/runtime changes, temporary public APIs, the C++ bridge,
extra build dependency and four Rust diagnostic executables have been removed.
The released serve path stays at main `f691af6e1`; no default was enabled.
The capture fixture's explicit prompt-depth control and research records remain.

| pair | order | baseline tok/s | K=2 tok/s | ratio |
| --- | --- | ---: | ---: | ---: |
| 0 | baseline, K=2 | 50.200 | 52.583 | 1.04748x |
| 1 | K=2, baseline | 50.917 | 53.683 | 1.05434x |
| 2 | baseline, K=2 | 50.467 | 53.400 | 1.05812x |

This is the preregistered single-agent `serve_realistic.py` workload: 49,152
warm source tokens, agent sampling (T=1 / top_p=.95 / top_k=0), three paired
60-second windows, A-B-B-A-A-B order and the same iter binary for all six arms.
The run used actual recorded OpenCode call sizes/tool gaps and authentic Rust
source. Each sequence continued its own generated history. Before loading and
after warming, each arm settled for 60 continuous seconds at pressure <=1;
all six measured windows stayed at **pressure 0**. Background compiler work
was left running and its overlap/load was recorded. Each paired lease held
work for 8.6–8.9 minutes. `served-paired-result.json` validates the shared binary
hash, sampling, seed, context, corpus and settling; individual summaries retain
rates, thermal traces/budgets and host load.

The favorable native copying cohort reached 1.256x, but the real serve review
accepted only about 0.61 drafts per proposal, about 2.2 emitted tokens per
packet. Logs confirm that head history was reused: measured-window catch-up
work was small; cold initialization was paid during warming. The output remains
a coherent engineering review. The evidence does not support a default-on solo
path at the requested threshold. K=1 was built/qualified first; K=2 was selected
for the served gate after its stronger native result.

Correctness passed before removal: statistical exact rejection sampling,
greedy equality, per-row target/head state restoration, cancellation, native
terminal IDs, max_tokens, peer arrival, and the 20-second three-stream smoke
with zero NextN activity. Custom stop-string fields are ignored by the existing
Responses parser, and that behavior remained identical. Functional serve checks
used release; native state tests used the test profile with Metal validation.

Prototype code remains recoverable from this branch at `5e0e6d339` (runtime
source `9fd036671`); the checkpoints below describe that historical prototype.
The final checkout contains no speculative product path. Compact decision
artifacts live in this directory; the large raw reports
(`agent48-report.json`, `agent48-full-restore-report.json`,
`recursive-code-report.json`, `head-reprice-report.json`,
`agent-head-report.json`) remain on branch `perf/glm-solo-nextn` at
`d2b565951`, with full logs, request/response bodies and run metadata in the
M5 run directory `metagross:~/qwen-runs/2026-11-12-glm-nextn/`.

## Historical repricing and prototype checkpoints

Stage 1 starts from main `f691af6e1`. Gate: projected delivered solo gain at
48K and T=1 / top_p=0.95 / top_k=0 must reach 1.15x before product work.

`spec_solo_nextn_cost` measures scalar and GPU-loop B=1, independent batch
rows, causal rows through the batch kernels, and the compact target state copy.
The causal diagnostic shares persistent caches, serializes KDA and indexer
updates, and keeps row visibility and batch projections/experts. It is compiled
only with `perf-ablations`; serve has no call site. It deliberately has no
acceptance or partial-commit API. A future-input perturbation checks that earlier
rows are bitwise unchanged; scalar comparisons report reduction differences.

Measurements and the build/no-build decision follow in the results checkpoint.

## Current native costs (checkpoint; decision pending)

One `iter` binary (`14b75787e`, `perf-ablations`), M5 Ultra, stock
UD-IQ3_XXS, 49,152 tokens, 12 timed repetitions per shape and pool phase.
All-row verify costs include CPU logit retrieval. A scalar control uses the
same weight instance and prefill trajectory.

| component | measured ms |
| --- | ---: |
| B=1 scalar forward with logit readback | 18.421 |
| B=1 GPU loop, T=1 / top_p=.95 / top_k=0, median of three 64-forward runs | 17.996 |
| B=1 ordinary CPU draw | 0.956 |
| B=1 draw retaining filtered probabilities | 0.891 |
| causal V(2), median across four pool phases | 24.125 |
| causal V(3), median across four pool phases | 28.776 |
| target compact state copy, 152,813,568 bytes | 0.631 |
| target compact state restore after actual verification | 0.650 |
| head step, 47,697-token reference, median excluding first | 1.613 |
| head-history initialization, same reference | 3536.6 |
| head model load and context creation | 689.4 |

The shared-cache path uses existing multi-query attention and score kernels;
only KDA recurrence and pending-indexer appends serialize. No alias guard is
removed. Four consecutive starting positions exercise all pooling phases.
Eight future-input perturbation checks pass bitwise on earlier logits; all
20 checked greedy tokens match scalar. Maximum scalar logit difference is
9.25e-5. These are causal-probe checks, **not distribution qualification**.

The earlier per-query fallback costs 24.553/29.111 ms and also passes its
causal checks. The first failed probe was refused by the independent-cache
alias guard before causal execution; no timings from that failed causal call
are used. An initial no-op rollback timing was corrected to measure restoration
after actual verification.

The head timing is an upstream MTP reference using the same extracted native
block-45 tensors, not a shipped Rust head. The M5 upstream source is an archive
without Git metadata; the updated probe labels it `archive-no-git` and retains
library/header/wrapper hashes. Its first recursive attempt exposed an immutable
batch-input binding and was fixed; that incomplete run is excluded.

## Acceptance and delivered-cost method

The first fresh current-target capture conditions on a recorded OpenCode GLM
transcript plus source context, then samples a technical review at T=1,
top_p=.95, top_k=0. Its actual prompt is **56,916 tokens**, with 128 scored rows:
one-step overlap **0.90595**, second-step target-hidden on-trajectory proxy
**0.92844**. This is a separate longer-context cohort, not a 48K result.
`agent-input.json` identifies the transcript and message hashes; private raw
messages, hidden rows and logits remain on M5. The exact-depth capture completed with **49,152 prompt tokens and 256 scored
rows**, retaining the first 32 and tail tokens of the same single-user render.
Its manifest records that source-context truncation explicitly. Head analysis
of this exact-depth cohort is pending.

The old .814 second-step estimate uses target hidden states. The new probe
instead feeds the first head's normalized `h_nextn` into the second head step.
For captured next token x sampled from P, the weight `min(1,Q(x)/P(x))` times
the recursive next-row overlap estimates joint acceptance of both drafts.
This accounts for the accepted-first-token distribution without pretending
that the second head has the target's hidden state. The reduction's analytic
self-test compares that importance estimate with the explicit categorical sum.
The recursive result and head-state rewind/copy prices remain pending.

`solo_project.py` charges K proposal distributions and E target distributions,
where E is expected emitted tokens, rather than one target draw per packet.
It reports current exact CPU costs, a hypothetical faster retention path, a
free-sampling ceiling, host overhead sensitivity and fresh-history amortization.
Warm-head numbers require preserving or rebuilding the head state across turns.
Current native prefill captures can supply the needed final hidden rows, so
head-history work can be fused after each target chunk; it still adds GPU work.
Current production snapshots contain only target state, so head reuse is not
implemented by the baseline.

## Protocol and status

Every M5 build and model load queues through `with-metal-lease --task glm-nextn`,
with holds declared at 15 minutes or less. Builds and model processes use
`nice -n 10`. Outputs are under `~/qwen-runs/2026-11-12-glm-nextn/`.
The native driver requires 60 continuous seconds at pressure <=1 before load,
records pressure throughout, allows 2 and aborts at 3. This is a native price,
not the delivered paired serve gate. The shared-cache target run stayed at 0
through settling and measurement. The head-reference run encountered pressure 2
before its continuous settling interval completed; measurement stayed at 0. Background ghost work is left untouched;
process listings and load averages are preserved in the M5 run metadata.

No product path is built. The 1.15x decision waits for the recursive and
exact-depth acceptance results and the head-state price. Stage-2 distribution,
terminal/cancellation, paired served throughput and B=3 smoke gates have not run.

## Stage 1 decision: proceed to K=1, then K=2

The exact-depth trace-conditioned cohort is coherent: the 256-token
continuation reviews the task and reconstructs the relevant Rust guard code.
It has 136 distinct tokens; it is not a repeated-token acceptance artifact.
Its unusually high acceptance reflects copying code from the supplied context,
so it must not be generalized to arbitrary code generation.

| exact-depth cohort | one-step overlap | recursive conditional second estimate | expected tokens, K=2 |
| --- | ---: | ---: | ---: |
| current-target traced review, 49,152 prompt tokens, 256 scored rows | 0.98111 | 0.93455 | 2.89800 |
| earlier code capture, 47,697 prompt tokens, 128 scored rows | 0.79759 | 0.67702 | 2.33757 |

The recursive estimate importance-weights the actual next target token; it is
not the earlier target-hidden/greedy-subset .814 proxy. Row SEM for joint
acceptance is 0.01391 on the trace and 0.03137 on code. This measures one
recursive block with canonical head history; sustained runtime acceptance still
needs measurement.

Head-state qualification found a critical implementation distinction. Full
ON_DEVICE serialization costs **4.045 ms to capture and 12.110 ms to restore**
on the code reference, and also makes subsequent head calls slower. Instead,
the head's append-only caches permit a **logical position ring**: removing a
one-to-four-row suffix costs **0.051–0.068 ms** and reproduces the final logits
bitwise at all four pooling phases (16 timed repetitions each). The exact-depth
trace passes too, at about 0.07 ms. Target state still needs its 153 MB compact
ring. No retained target prefix is replayed by this design.

The final trace head run uses that position rewind between recursive probes.
Its first/second head calls cost **2.295/1.836 ms** median. The first price is
conservative for K=1: it includes the private normalized-hidden extraction and
follows a rewind on every probe. The one-step-only reference was 1.613 ms.
One- and two-step logit files are **byte-identical** between full-state restore
and position rewind (`rewind-logit-sha256.txt`), so the faster restore did not
change the acceptance result. `agent48-continuation.txt` holds the coherence
sample; `agent48-quality.json` holds the compact measurements, and the full
`agent48-report.json` remains on branch `perf/glm-solo-nextn` at `d2b565951`
under `metagross:~/qwen-runs/2026-11-12-glm-nextn/`.

`agent48-projection.json` uses the final 48K target prices, separate first and
second head prices, exact CPU distribution construction, target restores on
partial acceptance, and a head position rewind on first rejection at K=2.
It also charges **0.25 ms/packet for head-history catch-up** and **0.50 ms/packet
for adapter work**, both explicit estimates rather than measurements.

| case | warm native ceiling | with 1 ms unchanged host cost/token | host sensitivity, 0–2 ms/token |
| --- | ---: | ---: | ---: |
| K=1, current exact CPU distributions | 1.170x | 1.160x | 1.151–1.170x |
| K=2, current exact CPU distributions | 1.326x | 1.303x | 1.284–1.326x |

The code cohort loses the 1.15x screen with either K, using its measured
recursive estimate. It remains a required sensitivity, rather than being
replaced by the favorable copying cohort. Initial head history is about
3.5–3.8 s and would erase the gain on short fresh sessions; preserving or
fusing head history and reusing it across served turns is mandatory. These
numbers are projections, not a served throughput claim.

**Decision:** the coherent, actual-transcript-conditioned 48K case clears
1.15x even with the stated allowances. Build K=1 first, then K=2 because it
projects better. Keep the native/delivered distinction and apply all stage-2
correctness, terminal, cancellation, concurrency and paired served gates before
making the product path default-on. No serve/product behavior has changed at
this checkpoint.


## Stage 2 native packet checkpoint

Source `d360239e2`, same-process iter A/B order A-B-B-A-A-B at 49,152
prompt tokens and T=1/top_p=.95/top_k=0. Each arm emits 128 tokens.
This is a warm native ceiling, not the delivered serve gate.

| K | baseline ms/token (three arms) | first speculative arm | later speculative arms | warmed ratio |
| --- | --- | --- | --- | --- |
| 1 | 18.048, 17.940, 17.947 | 17.177 | 15.271, 15.379 | 1.173x |
| 2 | 18.024, 17.905, 17.915 | 14.276 | 14.270, 14.317 | 1.256x |

Sustained acceptance falls below the trace-conditioned projection: K=1 accepts
61 of 67 proposals; K=2 accepts 82 of 92 proposals and emits 127 post-anchor
tokens in 46 packets. Both continuations coherently review the supplied Rust
transcript. The first head invocation adds setup work; the raw first arm is
retained rather than discarded. Head-history initialization is over four
seconds in this wrapper and is outside the timed packet arms.

The native target ring passes at all four pooling phases for both K values,
including retaining either one or two inputs. Tests use the test profile and
Metal API validation. Anchor/restored-prefix logit differences remain small,
greedy selections match, invalid input causes no mutation, unresolved packets
poison the session, and explicit snapshot recovery works. No retained prefix
is replayed. Raw witnesses are native-ring-k1.jsonl and native-ring-k2.jsonl.

The exact sampler's synthetic Markov test checks 300,000 transitions per arm
at the requested nucleus settings for each K, against both analytical target
probabilities and serial sampling. Greedy ties, terminal tokens and limits pass.
A separate cursor publication test checks cancellation, limits, terminal IDs,
peer arrival and backpressure against serial publication. All three tests in
each module pass. The earlier committed structural contract suite passed 646
CPU inventory tests; the new adapter witnesses will enter the next check.

The prototype retains normalized target rows and the live head across turns,
and includes host head state in snapshots. It extracts the 32 intrinsic/fallback
tensors directly from the stock split GGUF without changing their bytes. A
one-agent mode was added to serve_realistic.py. These changes remain opt-in
until native history, served exactness, paired throughput and B=3 gates pass.


The subsequent test-profile history witnesses pass at 64 and 8,192 tokens with
Metal API validation. The stock-head subset loads, target prefill logits remain
bitwise identical with feature export, and greedy GPU-loop tokens match.
Same-session and fresh-session head restoration reproduce logits bitwise;
fresh-session preparation after restore costs 11.37/14.04 ms, with no prefix
replay. Solo and three-member GPU-loop feature histories stay contiguous and
reset recovers the 64-row prefix. Raw witnesses: history-dense.jsonl and
history-sparse.jsonl. These are correctness checks, not iter throughput prices.

The Responses parser currently ignores a custom `stop` field. Preserve that
existing behavior and check native GLM terminal IDs through the ordinary
cursor; do not claim custom stop-string support. Live HTTP checks are pending.
The HTTP harness preserves the stock publisher's split-GGUF entrypoint name:
resolving its symlink changes the suffix and is rejected by the model loader.


## Live serve correctness checkpoint

`serve-smoke.json` records the release-binary HTTP check with runtime rollback
0 and K=2. The 64-token greedy response text is identical; limits 1, 2, 3, 4
and 17 all return the requested token count. Custom stop fields remain ignored
in both arms. A disconnected stream is followed by a healthy request. A second
request submitted mid-turn joins a native batch of width two in both arms.

With K=2 enabled, three streams remain open for 20.024 seconds and deliver
861/772/725 content events. There are zero NextN activity reports in that
window, while six reports confirm solo engagement elsewhere in the same check.
This is a concurrency smoke, not a three-agent throughput A/B. The actual
served solo speed gate remains the preregistered three interleaved iter pairs.
