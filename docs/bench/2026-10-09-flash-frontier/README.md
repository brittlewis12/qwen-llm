Flash frontier scheduling: UD and GSQ suffix/whole4096, 2026-10-09
================================================================

Actual production-default confirmation — 2026-10-09 (current decision)
-------------------------------------------------------------------

**Narrow promotion is integrated on main `83c91cec`; actual `None` execution
is confirmed on UD and GSQ, and final independent review approved with no fixes.**
The subsequent pinned-Metal-target confirmation preserves every sampled
trajectory; see the [integration appendix](#integration-appendix-pinned-metal-target-at-83c91cec).
The joint quality basis remains v1 NLL PASS on both artifacts plus fresh,
separately frozen v2 retrieval PASS on both artifacts.
GSQ's v1 retrieval remains INCONCLUSIVE (correct facts, missing required
marker); it has not been rescored or converted into a v1 pass. The earlier
"parked / v2 under review" decisions below are retained history, superseded
by this joint review and actual-default confirmation. Validation is limited to
the filters and checks below; this is not a full-suite claim.

**Pre-integration final validation, owner-reported:**

| Scope | Release | Debug |
|---|---|---|
| `qwen4exp_runtime::tests` | 28 passed, 0 failed, 32 ignored | 28 passed, 0 failed, 32 ignored |
| HC capture CPU tests | 3 passed | 3 passed |
| GDN capture CPU tests | Not reported in this final update | 3 passed |

The `qwen-cli` production check also passes. Ignored benchmark acquisitions
were executed separately as documented in the retained release packets; the
32 ignored entries are not counted as passes in either filtered run. Final
review verified the historical policy pins and requested no fixes. These
results were supplied by the owner; this docs update ran no tests or builds.
Integration subsequently completed at `83c91cec`; the owner also reports six
planner CPU tests passed after the target-pinned rebuild. These six are an
additional focused result, not a rerun claim for the entire table above.

The source guard is exact: `selected_enabled`, packed capacity 2048,
`dense_end=2051`, and `(start,count)` equal to **`(0,4096)` or `(2048,2048)`**.
Those ordinary calls now group into `2048+2048` or a single suffix 2048 command.
Other geometries retain the existing planner path. This changes command
grouping, **not the arithmetic policy**: production router, quant kernels and
BF16 activation-dispatch rules remain in effect. Grouping changes which
width-dependent arithmetic is reached, so GSQ's deterministic schedule-dependent
numerical differences remain part of the qualified behavior. The diagnostic
uniform/narrow HC F32 interventions are not promoted.

The new completed packets are
[ud-production-confirmation.jsonl](ud-production-confirmation.jsonl) and
[gsq-production-confirmation.jsonl](gsq-production-confirmation.jsonl). Each
records **one ABBA round**, both natural corpora and both suffix/whole stages.
A explicitly uses `Some(false)` for the incumbent planner; B explicitly uses
**`None`**, exercising the actual production default, not forced `Some(true)`.
All 24 arm records per artifact and all eight census witnesses per artifact
bind the appropriate override and expected absolute ranges. The generic raw
`diagnostic_only_no_promotion` / `production_change=false` fields describe the
harness's non-mutating measurement role; they are not evidence that B uses the
old planner. The explicit mode/override records and retained source guard
identify the policy executed. Raw headers are preserved as emitted.

**Complete timing evidence, without changing the headline to fit this run.**
Warm census arms are excluded. Each artifact has 16 timed arms, eight excluded
warm arms, and complete valid GPU coverage for every timed command. Wall time
is the ordinary prefill/continuation call; reset/restore, JSON, endpoint
readback/hashing and four continued tokens are outside it. All 16 adjacent GPU
pairs and all 16 wall pairs across both packets favor B. Suffix and whole
savings overlap and must not be added.

| Artifact / corpus / stage | Mean GPU A→B ms | GPU saved ms / % | Mean wall A→B ms | Wall saved ms / % |
|---|---:|---:|---:|---:|
| UD / prose/suffix_at2048 | 4799.08→4586.07 | 213.01 / 4.439% | 4815.01→4607.16 | 207.85 / 4.317% |
| UD / prose/whole4096 | 8368.24→8138.22 | 230.03 / 2.749% | 8389.45→8163.51 | 225.94 / 2.693% |
| UD / ssh_repeated/suffix_at2048 | 4796.68→4605.33 | 191.35 / 3.989% | 4812.41→4625.46 | 186.94 / 3.885% |
| UD / ssh_repeated/whole4096 | 8313.50→7978.09 | 335.42 / 4.035% | 8333.24→8003.76 | 329.48 / 3.954% |
| GSQ / prose/suffix_at2048 | 5152.68→4963.14 | 189.54 / 3.678% | 5175.66→5010.15 | 165.51 / 3.198% |
| GSQ / prose/whole4096 | 8503.12→8138.47 | 364.65 / 4.288% | 8529.53→8175.21 | 354.32 / 4.154% |
| GSQ / ssh_repeated/suffix_at2048 | 4945.88→4678.57 | 267.31 / 5.405% | 4968.03→4710.72 | 257.31 / 5.179% |
| GSQ / ssh_repeated/whole4096 | 8434.62→8116.08 | 318.54 / 3.777% | 8461.46→8153.35 | 308.11 / 3.641% |

All pairs and drift below are from the sole declared round. P1=A1−B1;
P2=A2−B2. Pair entries are saved milliseconds / percent. Drift entries are
100×(A2/A1−1) / 100×(B2/B1−1). Complete raw arm timings, median summaries and
numerical comparisons are in
[production-confirmation-summary.json](production-confirmation-summary.json).

| Artifact / corpus / stage | GPU P1; P2 saved ms / % | Wall P1; P2 saved ms / % | GPU drift A / B % | Wall drift A / B % |
|---|---:|---:|---:|---:|
| UD / prose/suffix_at2048 | 231.05 / 4.797%; 194.97 / 4.077% | 219.78 / 4.554%; 195.92 / 4.078% | -0.719% / +0.032% | -0.438% / +0.059% |
| UD / prose/whole4096 | 231.53 / 2.759%; 228.53 / 2.738% | 218.21 / 2.597%; 233.68 / 2.790% | -0.531% / -0.510% | -0.334% / -0.532% |
| UD / ssh_repeated/suffix_at2048 | 191.17 / 3.986%; 191.53 / 3.993% | 180.81 / 3.762%; 193.08 / 4.007% | +0.024% / +0.017% | +0.277% / +0.022% |
| UD / ssh_repeated/whole4096 | 443.29 / 5.265%; 227.55 / 2.773% | 432.15 / 5.124%; 226.82 / 2.755% | -2.523% / +0.042% | -2.378% / +0.060% |
| GSQ / prose/suffix_at2048 | 235.62 / 4.383%; 143.45 / 2.910% | 199.92 / 3.712%; 131.10 / 2.640% | -8.308% / -6.896% | -7.789% / -6.762% |
| GSQ / prose/whole4096 | 373.43 / 4.405%; 355.87 / 4.173% | 349.09 / 4.111%; 359.54 / 4.196% | +0.593% / +0.837% | +0.909% / +0.819% |
| GSQ / ssh_repeated/suffix_at2048 | 267.31 / 5.403%; 267.32 / 5.406% | 245.55 / 4.953%; 269.07 / 5.404% | -0.050% / -0.053% | +0.434% / -0.042% |
| GSQ / ssh_repeated/whole4096 | 363.96 / 4.292%; 273.13 / 3.256% | 341.77 / 4.023%; 274.45 / 3.257% | -1.086% / -0.016% | -0.803% / -0.011% |

Retain the earlier repeatable **UD ~2.4% warmed whole4K GPU/wall** headline,
not the new 4.035% SSH GPU mean: that cell's A falls 2.523% within the round and
its paired gains are 5.265% versus 2.773%. UD prose here saves 2.749% GPU / 2.693%
wall with both pairs positive. GSQ whole prose saves 4.288% GPU / 4.154% wall
and SSH 3.777% / 3.641% in this one round. These are positive execution-confirmation
observations, not a new stable 4% GSQ claim. Earlier prose nonstationarity and
this packet's 8.308%/6.896% same-arm GPU drift in prose *suffix* remain visible.
Do not pool selected historical cells, replace the earlier headline, or infer
cold placement, short-prompt, 8K or other-depth savings. Review supports the
narrow promotion with GSQ's benefit direction positive and magnitude uncertain.

**Default-policy and dispatch witnesses.** All normal production-lease,
diagnostic memory-admission, completion and artifact-revalidation gates pass.
Every suffix warm arm witnesses 48 strict-router calls; every whole warm arm
witnesses 96 (48 prefix  + 48 suffix). A additionally witnesses 48 generic N3
router calls. The N2045 command in A remains on its promoted strict router;
this comparator is not the historical generic-router baseline. No router gain
is added to these scheduling results.

| Artifact / stage, either corpus | BF16 bfloat matmuls A / B | BF16 F32-activation matmuls A / B |
|---|---:|---:|
| GSQ suffix | 194 / 194 | 230 / 24 |
| GSQ whole | 388 / 388 | 242 / 36 |
| UD suffix | 0 / 0 | 36 / 24 |
| UD whole | 0 / 0 | 48 / 36 |

These counts confirm the original arithmetic paths, including GSQ bfloat
activation dispatch, remain active. They do not imply schedule bit equivalence.
All 180 retained endpoint records are finite (90 per artifact, including warm
and prefix checkpoints). Within each schedule, repeated endpoints/state and
four teacher-forced continuations agree exactly; the same schedule also agrees
between restored suffix and fresh whole execution. Actual token arrays match
the corresponding earlier `*-both-restart.jsonl`, and **all 90 matching endpoint
records per artifact reproduce exactly**, including logits/state hashes.
Thus actual `None` B reproduces the previously measured forced-B trajectory.

UD's A/B logits and persistent state remain exact on these streams. GSQ still
has 120/121 differing persistent hashes across schedules, with matching
position/QSA/PLE metadata and top1 choices. Endpoint-plus-four-continuation
maxima remain the historical values: prose KL(A‖B)/KL(B‖A)
0.0429139/0.0950802, relative L2 0.137688, maxabs 2.77734; SSH
0.000550126/0.00102399, relative L2 0.0720061, maxabs 2.65525. Choice regrets
are zero. These logit metrics are retained source-computed observations,
not recomputed from unavailable full logits; timings and hash/metadata
comparisons are independently audited. These continued tokens are confirmation
controls, not a rerun of the frozen NLL/retrieval quality studies.

**Quality decision and remaining work.** Independent review accepts the joint
[v1 NLL evidence](../2026-10-09-flash-frontier-quality/README.md) and
[v2 retrieval evidence](../2026-10-09-flash-frontier-retrieval-v2/README.md)
as support for this narrowly guarded promotion. V2 was frozen before its GPU
acquisition with new facts and a new strict raw-JSON contract; both artifacts
parse and answer both before/after-frontier cases correctly in both schedules.
Its motivation was the v1 formatting issue, so it is a fresh contract check,
not an independent replication or retroactive repair of v1. GSQ v1 remains
INCONCLUSIVE. No quality acquisition was rerun for this docs update. The scoped
final checks above pass and the reviewer gives final approval with no fixes;
integration subsequently completed at `83c91cec`. Next Flash performance work
is selected-QSA leaf attribution, not another layer0 bit-identity intervention or
an enlarged scheduling speed claim.

**Historical harnesses and source bindings.** The historical router packet and
`prefill_map` now explicitly pin `Some(false)` for their full invocations,
covering both planning and execution. The router packet preserves its N2045
router comparison under the old schedule; `prefill_map` separately preserves
its older generic-N2045 router baseline. Shared native scaffolding stays
unpinned, so this actual-default confirmation uses current production routing
and B=None. Those test-only pins do not change production behavior. Their
current helper source hashes may postdate these binaries; no historical raw
header is rebound to current HEAD. The executable in both new packets is
47,118,952 bytes, SHA256
`1f85ca7827e9a122bebe05d760c5a09a3561fd76675cf523c05f27a59ca974e1`.

**Analysis validation and reproduction.** `summarizer.py` validates a positive
round count from each header, exact ABBA ordering/count, each ABBA record's
optional legacy/new declared count, and agreement of recorded arm timings.
New explicit mode/override fields are checked in every arm and census witness;
legacy missing override fields are not reinterpreted as production None.
All seven retained timing packets audit cleanly: five historical two-round
packets and these two one-round packets. Four synthetic CPU-negative controls
reject a false header count, false ABBA count, forced-true B in production mode,
and a missing B override. They exercise the analyzer only, not model/GPU code.

```sh
UV_CACHE_DIR=/tmp/flash-frontier-uv-cache uv run --no-project \
  docs/bench/2026-10-09-flash-frontier/summarizer.py \
  docs/bench/2026-10-09-flash-frontier/ud-production-confirmation.jsonl \
  docs/bench/2026-10-09-flash-frontier/gsq-production-confirmation.jsonl \
  > /tmp/flash-production-confirmation-summary.json
```

| Raw binding | Bytes | SHA256 |
|---|---:|---|
| ud-production-confirmation.jsonl | 1,729,914 | `eb7a2f0f310a0b785e7f3ef4d970fcbde6f17216e1a663f00b9f00feeeb6d6f8` |
| gsq-production-confirmation.jsonl | 1,735,328 | `48fb67024a80842ebfa144bf84d9b8f96d9ac5963e56814bf06673a7fc18f5b6` |

**Git retention recommendation.** Keep all completed raw JSONLs unchanged,
the derived summaries, analysis scripts and README. Keep these verified
lossless sidecar archives with their matching `.manifest.json` files:

| Archive | Bytes | Original payload represented |
|---|---:|---|
| `gsq-layer0-sidecars.tar.xz` | 20,792,856 | 28,500,992 bytes |
| `gsq-layer0-bf16-f32-sidecars.tar.xz` | 13,364,996 | 28,500,992 bytes |
| `gsq-hc-oracles-sidecars.tar.xz` | 33,354,224 | 92,719,104 bytes; shared HC weights stored once |

The manifests bind every original filename and hash. Keep raw `.f32le`,
`.f64le` and `.weights.bin` sidecars locally without also adding those duplicate
payloads to git. The owner added `*.f64le` and `*.weights.bin` ignore rules;
raw sidecars have not been deleted. No new binary archive is needed
for these JSONL-only production confirmations; do not replace readable raw
JSONLs with another compressed copy. Existing archive hashes were rechecked;
prior full member round-trip verification remains recorded in the manifests.
No GPU work, build, model-source edit or commit was performed for this update.

Earlier whole-only performance confirmation (retained history)
-------------------------------------------------------------

Whole-only performance confirmation — 2026-10-09
-------------------------------------------------------

**Bounded headline: UD repeats about 2.4% whole-prefill GPU and wall savings
on both natural corpora. GSQ improves in every pair, but does not establish a
stable whole-prefill saving across both corpora; keep the GSQ headline and
cross-artifact promotion parked.** GSQ prose drift is comparable to the claimed
saving, so its 4.73% aggregate is not a reliable performance estimate. No round
or pair was discarded, and these new runs are not pooled with selected older
cells to rescue a headline.

The completed raw packets are [ud-whole-confirm.jsonl](ud-whole-confirm.jsonl)
and [gsq-whole-confirm.jsonl](gsq-whole-confirm.jsonl). Owner-reported process
completion was 258 s UD / 336 s GSQ; those durations include work outside the
prefill timer and are not the performance metric. Each packet contains two
corpora, two warmed ABBA rounds per corpus, 16 timed whole4096 attempts and
four census warm arms excluded from every timing summary below. All 32 timed
attempts have valid complete GPU coverage (80 command samples in total).
The normal lease, diagnostic memory gate, completion and shard-revalidation
checks pass; the existing uv summarizer reports zero audit problems.

A is ordinary fresh `2048+3+2045`; B is ordinary fresh `2048+2048`, both from
position 0 with **current production routers and original arithmetic**. There
is no restored-suffix timing, HC F32 intervention or profiled timed arm in this
confirmation. Wall time encloses the ordinary `prefill` call. Preparation/reset,
endpoint hashing/readback, JSON emission and four subsequent teacher-forced
continuations are outside that timer. GPU time is the complete ordinary command
sum: three commands for A, two for B. The harness retains validity and sample
counts, not individual raw command timestamps.

**Every cell's full two-round aggregate.** Means use all four A and four B
observations in each cell. Savings are `(A−B)/A`; percentages compare means
(or medians), rather than averaging pair percentages. These are descriptive
summaries of repeated execution on two token streams, not confidence bounds
or independent quality samples.

| Artifact / corpus | Mean GPU A→B ms | GPU saved ms / % | Mean wall A→B ms | Wall saved ms / % | Median GPU / wall saving % |
|---|---:|---:|---:|---:|---:|
| UD / prose | 8176.49→7976.00 | 200.50 / 2.452% | 8205.64→8007.66 | 197.99 / 2.413% | 2.430% / 2.428% |
| UD / ssh_repeated | 8177.70→7978.95 | 198.75 / 2.430% | 8208.12→8012.65 | 195.46 / 2.381% | 2.435% / 2.405% |
| GSQ / prose | 9623.39→9168.42 | 454.97 / 4.728% | 9677.25→9225.48 | 451.77 / 4.668% | 4.834% / 4.890% |
| GSQ / ssh_repeated | 8388.56→8133.67 | 254.89 / 3.039% | 8420.33→8167.96 | 252.37 / 2.997% | 3.297% / 3.323% |

**All 16 adjacent ABBA pairs.** P1 compares A1−B1; P2 compares A2−B2 (B2 runs
before A2). Positive means B is faster. All 16 GPU pairs and all 16 wall pairs
are positive; GSQ prose's smallest pair is only 0.508% GPU / 0.482% wall.

| Artifact / corpus | Round / pair | GPU saved ms / % | Wall saved ms / % |
|---|---|---:|---:|
| UD / prose | 1 / P1 | 205.80 / 2.515% | 188.33 / 2.298% |
| UD / prose | 1 / P2 | 195.01 / 2.386% | 197.99 / 2.412% |
| UD / prose | 2 / P1 | 205.86 / 2.517% | 207.71 / 2.529% |
| UD / prose | 2 / P2 | 195.30 / 2.390% | 197.93 / 2.412% |
| UD / ssh_repeated | 1 / P1 | 201.15 / 2.460% | 182.02 / 2.222% |
| UD / ssh_repeated | 1 / P2 | 213.53 / 2.608% | 215.68 / 2.623% |
| UD / ssh_repeated | 2 / P1 | 187.35 / 2.292% | 189.35 / 2.306% |
| UD / ssh_repeated | 2 / P2 | 192.96 / 2.362% | 194.80 / 2.374% |
| GSQ / prose | 1 / P1 | 446.45 / 4.813% | 425.11 / 4.576% |
| GSQ / prose | 1 / P2 | 724.40 / 7.440% | 736.12 / 7.505% |
| GSQ / prose | 2 / P1 | 48.44 / 0.508% | 46.33 / 0.482% |
| GSQ / prose | 2 / P2 | 600.58 / 6.044% | 599.51 / 5.996% |
| GSQ / ssh_repeated | 1 / P1 | 277.42 / 3.308% | 257.60 / 3.066% |
| GSQ / ssh_repeated | 1 / P2 | 293.35 / 3.497% | 298.51 / 3.542% |
| GSQ / ssh_repeated | 2 / P1 | 170.25 / 2.030% | 172.42 / 2.047% |
| GSQ / ssh_repeated | 2 / P2 | 278.54 / 3.319% | 280.92 / 3.333% |

**Every round's mean saving and same-arm drift.** Drift is 100×(A2/A1−1) or
100×(B2/B1−1). A large difference between these drifts warns that ABBA has not
removed nonstationarity; there is no retrospective stability cutoff or removal
of inconvenient samples here. Complete arm timings are in
[whole-confirm-summary.json](whole-confirm-summary.json).

| Artifact / corpus / round | GPU mean saved ms / % | Wall mean saved ms / % | GPU drift A / B % | Wall drift A / B % |
|---|---:|---:|---:|---:|
| UD / prose / 1 | 200.41 / 2.451% | 193.16 / 2.355% | -0.126% / +0.006% | +0.143% / +0.025% |
| UD / prose / 2 | 200.58 / 2.453% | 202.82 / 2.470% | -0.066% / +0.064% | -0.063% / +0.058% |
| UD / ssh_repeated / 1 | 207.34 / 2.534% | 198.85 / 2.423% | +0.115% / -0.038% | +0.380% / -0.031% |
| UD / ssh_repeated / 2 | 190.15 / 2.327% | 192.08 / 2.340% | -0.050% / -0.121% | -0.037% / -0.106% |
| GSQ / prose / 1 | 585.42 / 6.158% | 580.61 / 6.080% | +4.961% / +2.064% | +5.581% / +2.340% |
| GSQ / prose / 2 | 324.51 / 3.332% | 322.92 / 3.293% | +4.118% / -1.675% | +4.015% / -1.748% |
| GSQ / ssh_repeated / 1 | 285.38 / 3.403% | 278.06 / 3.305% | +0.017% / -0.179% | +0.310% / -0.183% |
| GSQ / ssh_repeated / 2 | 224.40 / 2.674% | 226.67 / 2.690% | +0.051% / -1.266% | +0.055% / -1.259% |

UD's four round means span 2.327–2.534% GPU and 2.340–2.470% wall. Absolute
same-arm drift stays below 0.127% GPU and 0.381% wall, substantially smaller
than every paired gain. A roughly 0.20 s / 2.4% whole-call saving is supported
for these two natural 4K streams under this confirmation's warmed conditions.
This does not establish short-prompt, cold-placement or deeper-context savings.

GSQ prose's GPU round means fall from 6.158% to 3.332%, while A drifts +4.961%
and +4.118% within the two rounds. Wall A drift reaches +5.581%. GPU pair gains
range 0.508–7.440%, wall 0.482–7.505%. The packet does not explain the cause of
this timing drift, and a positive mean/median does not resolve it. GSQ SSH is
more encouraging: all pairs save 2.030–3.497% GPU / 2.047–3.542% wall, and round
means are 3.403% then 2.674% GPU. However B still drifts −1.266% in round 2; do not
promote the SSH mean into a stable all-corpus GSQ claim. The old noisy whole
packets and their later stable cells remain below as history, not additional
selected observations in these aggregates.

**Router, arithmetic and numerical controls.** All eight census warm witnesses
show 96 strict-router calls per whole prefill in each arm: 48 in the common
prefix and 48 in the suffix, including production N2045 in A. A also has 48
N3 generic-router calls. These are additional scheduling savings after the
landed router transfer; the old 7% router win and router-disabled 199/282 ms
residuals are not additive budgets. GSQ's ordinary whole census retains 388
BF16 bfloat-activation calls in both arms, with 242/36 BF16 F32-activation
matmuls in A/B; this is the production width-dependent arithmetic, not either
HC intervention. UD records 48/36 BF16 F32 matmuls and no bfloat matmuls.

All 168 recorded endpoints/continuations (84 per artifact, including warm
endpoints) are finite. Same-schedule repetitions, including warm endpoint
versus ordinary timed endpoint and all four continuations, have identical
logit and persistent-state hashes in both artifacts. Both packets' actual
4100-token arrays and all 84 whole endpoint records also exactly reproduce
the corresponding earlier `*-both-restart.jsonl` records. Timing drift here
is not accompanied by a changed recorded numerical trajectory.

UD has exact A/B logit/state agreement at both corpus endpoints and all four
continuations. GSQ retains the known A/B differences: 120/121 persistent
allocation hashes differ, while causal metadata agrees. Across endpoint plus
four continuations, prose reaches KL(A‖B)/KL(B‖A) 0.0429139/0.0950802,
relative L2 0.137688 and maxabs 2.77734; SSH reaches 0.000550126/0.00102399,
relative L2 0.0720061 and maxabs 2.65525. All sampled top1 choices agree and
choice regrets are zero. These logit metrics are source-computed observations
(the raw JSONLs retain hashes, not full vectors); hash comparisons and timings
were independently recomputed. Matching top1 and finite logits do not establish
quality acceptance.

**Quality status is separate and remains bounded.** The completed
[v1 quality analysis](../2026-10-09-flash-frontier-quality/README.md) reports
**both NLL screens PASS; retrieval UD PASS / GSQ INCONCLUSIVE**. GSQ emits the
correct record/code facts in both schedules but omits the required
`FINAL_JSON:` marker. Under the frozen v1 parser and incumbent-parse-failure
rule, those are invalid fixtures/inconclusive, not passes and not evidence of
a candidate-specific factual regression. There is no content-only rescore.
The [fresh retrieval v2 contract](../2026-10-09-flash-frontier-retrieval-v2/README.md)
was frozen before its GPU acquisition and is **under review**; no v2 verdict
is claimed here. It is a separately versioned follow-up motivated by the v1
format failure, not an independent replication or a rewrite of v1. Do not
summarize the present state as “quality both pass.”

**Decision.** UD's modest performance benefit is repeatable in this bounded
confirmation, and its v1 quality verdict is PASS. GSQ does not yet support a
stable both-corpus performance headline, independently of its inconclusive v1
retrieval verdict. Keep cross-artifact promotion parked. Do not repeat the
whole matrix or choose only favorable GSQ rounds to manufacture stability.
Reopening GSQ performance requires a prospectively bounded prose confirmation
that resolves the drift, plus separate owner judgment on the frozen quality
contract; it does not require another HC bit-identity intervention. No
production change follows from this analysis.

Reproduce the complete audit (both raw packets remain unchanged):

```sh
UV_CACHE_DIR=/tmp/flash-frontier-uv-cache uv run --no-project \
  docs/bench/2026-10-09-flash-frontier/summarizer.py \
  docs/bench/2026-10-09-flash-frontier/ud-whole-confirm.jsonl \
  docs/bench/2026-10-09-flash-frontier/gsq-whole-confirm.jsonl \
  > /tmp/flash-whole-confirm-summary.json
```

| Binding | Bytes | SHA256 |
|---|---:|---|
| ud-whole-confirm.jsonl | 1,608,728 | `39ef5292448a9bc0f9cf78df5d98e7c6c5cdbc0ab91cae68a823a7e429de0792` |
| gsq-whole-confirm.jsonl | 1,612,805 | `db40abd6cc0aa3f607238fe30d2e5b9e313bd54541a2fa6ea4a6733045df9b47` |
| Both packet executables | 46,825,400 | `e29c8d92921e033d954fda6dc8df308964a9cf023aea1e49af2af50f297cffb7` |

Compiled source and metallib hashes stay bound to these packet headers, not to
an asserted current/future checkout HEAD. Only docs and derived analysis were
written for this update; no GPU, build, model-source edit or commit was made.

Earlier HC/oracle checkpoint (retained history)
---------------------------------------------

**Narrow layer0 attention HC + independent oracle checkpoint.** The
first captured production mismatch is consistent with the intended
width-dependent activation precision: equal normalized F32 inputs enter HC
down; A's N3 uses F32 activations and B's N2048 uses BF16-RNE activations.
Independent native-weight dots match each GPU path's own arithmetic closely.
There is no kernel-bug signal in these captured projections. This does not
establish model accuracy, validate every BF16 kernel shape, or justify schedule
promotion. Production scheduling and arithmetic remain unchanged.

[gsq-hc-production.jsonl](gsq-hc-production.jsonl) and
[gsq-hc-f32downup.jsonl](gsq-hc-f32downup.jsonl) completed under the normal lease,
memory admission and artifact-revalidation gates. The owner reports 93 s and
621 s process completion and six passing new CPU tests. Those elapsed durations
include untimed diagnostic work and are **not a performance comparison**.
Both packets have exact observer-off/on endpoint, persistent-state and original
dispatch concordance. No builds, GPU work or model-source edits were performed
for this analysis. There are no new continuation or quality measurements.

**First divergence, independently read from the raw sidecars.** Positions are
absolute, zero-based. Hyper input and post-RMS-normalization input match for
all eight retained rows, across both schedules and both modes. Initial GDN
convolution and delta state also match across schedules and modes.

| Production stage, rows 2048–2055 | First differing rows | A/B relative L2 (eight-row aggregate) | Maxabs |
|---|---|---:|---:|
| HC hyper input | none | 0 | 0 |
| HC normalized input | none | 0 | 0 |
| HC down, before SiLU | 2048–2050 only | 0.000177662266 | 0.0703735352 |
| HC low, after SiLU(x/4), actual up input | 2048–2050 only | 0.000166063537 | 0.00967121124 |
| HC raw up, before gated mix | 2048–2050 only | 0.00132279423 | 0.206161499 |
| Mixed output entering GDN | 2048–2050 only | 0.000562197690 | 0.0251445770 |
| GDN output | all eight rows | 0.00103500165 | 0.00143986940 |

All 960 down/low values and 30,720 raw-up values in the first three rows differ;
rows 2051–2055 are identical at those HC stages. Thus normalization is not the
first captured divergence, and SiLU inherits a difference already present in
down. This is localization, not a separate SiLU/mix accuracy oracle.

With only layer0 attention HC down+up forced to their existing F32-activation
path, all five HC sections and all 17 GDN sections are A/B byte-identical,
including all eight rows and delta-state snapshots after each of the first
three tokens. Production delta state after token 2050 had relative L2
0.000343748815 and maxabs0.00293636322; narrow mode has zero difference.
The new production GDN sidecars exactly reproduce the earlier production
capture. The narrow GDN sidecars exactly reproduce the earlier *broad* suffix
BF16-class intervention, SHA256
`1b8459106c945f9cd4fae1da83a7f0821c6f5b6962f57013bf8bbbb6116547c1`.
This narrows the sampled seed mechanism to the attention HC pair. It does not
separately measure a down-only intervention. Narrow A preserves production A's
first-three-row values, but changes its later rows too; this is not an unchanged
A comparator at every suffix position.

**Native-weight oracle.** Actual attention HC down/up are BF16
`[10240,320]` / `[320,10240]`; norm is F32 `[10240]`. Their retained native bytes
are identical between modes. The independent script decodes each BF16 word
exactly and evaluates binary64 `math.fsum` dot products with either the original
captured F32 input or that input rounded to BF16, round-to-nearest ties-to-even.
Weight layout is contiguous K within each output row. Both modes retain the
same 13,148,160-byte weight file; no F32 weight inflation is used as evidence.

The script recomputed all 675,840 saved f64 oracle elements (24 distinct
projection-input rows, reused only after exact byte-hash equality). Largest
absolute discrepancy from the producer's sequential-f64 references is
1.705303e-13. All captured floats, decoded weights and saved/recomputed oracles
are finite. All 996 HC/oracle metric fields and 1,608 GDN metric fields checked
across the two packets agree within summation rounding. Full per-row/per-arm
RMS, relative L2 and maxabs against **both** references are retained in
[hc-summary.json](hc-summary.json); A/B L2 normalizes by A, while oracle L2
normalizes by its reference.

| GPU cohort / captured rows | Reference matching dispatch | Down per-row relative L2 range | Up per-row relative L2 range |
|---|---|---:|---:|
| Production A, 2048–2050 | Original F32 input | 1.269e-6–1.601e-6 | 2.060e-7–2.104e-7 |
| Production A, 2051–2055 | BF16-RNE input | 1.395e-6–1.733e-6 | 1.903e-7–3.397e-7 |
| Production B, 2048–2055 | BF16-RNE input | 1.322e-6–1.733e-6 | 1.761e-7–3.397e-7 |
| Narrow A and B, 2048–2055 | Original F32 input | 1.269e-6–1.601e-6 | 2.060e-7–3.345e-7 |

Matched-reference maxabs is at most 0.000830940189 for down and0.000021632003
for up. By contrast, production B against the original-F32 reference has
per-row relative errors 0.0002745–0.0003909 down and0.0004161–0.0020088 up.
Production A's first three rows disagree with the BF16-RNE reference at
0.0002743–0.0003709 down and0.0011623–0.0023060 up. The small residuals against
the dispatch-matched references support ordinary reduction rounding around
the intended activation-precision change; an independent oracle is not expected
to be bit-identical to Metal's reduction order.

**Up inputs already differ upstream.** Every up reference uses that arm's actual
captured `hc.low`. Production A and B do not share that input on rows 2048–2050.
The raw-up A/B gap therefore combines the down/SiLU input change with the up
activation-rounding change. These conditional oracles cannot assign the entire
raw-up gap to the up kernel alone. Nor does closer agreement with a local
F32-input dot establish better whole-model accuracy.

**Scope witnesses.** Target counts are four total calls in A (two per role), two
in B (one per role), independently checked in observer-off and observer-on.
The scope identifies both actual layer0 attention HC weight buffers, offsets,
shapes and dtypes. It excludes FFN HC, injection, all other layers and prefix.
The header remains `bf16_activation_mode=production` in both packets; only
`hc_mode` changes from `production` to `f32downup`.

| Each observer off/on suffix | Target down/up policy | All BF16 bfloat calls | All BF16 F32-activation calls | Non-target bfloat calls |
|---|---|---:|---:|---:|
| Production A: N3 then N2045 | F32 then bfloat | 194 | 230 | 192 |
| Production B: N2048 | bfloat | 194 | 24 | 192 |
| Narrow A | F32 for both widths | 192 | 232 | 192 |
| Narrow B | F32 | 192 | 26 | 192 |

Both prefixes retain 194 bfloat / 12 F32-activation calls. Both suffix schedules
retain 48 strict production-router calls; A still has its N3 generic router
calls. Tagged GDN kernels/grids/thread shapes are unchanged per schedule
between modes. This intervenes on HC arithmetic without overriding the
N64 quant-kernel or routing policies. The capture fits its 15,319,040-byte GPU
upper quote (observed 15,237,120); its CPU upper quote is 371,521,536 bytes.
These are retained admission/accounting results, not measured latency savings.

**Endpoint limitation.** The JSONL records the following finite endpoint logits;
full logits are not sidecars, so these metrics are reported from the packet,
not independently recomputed. Persistent hashes are independently compared.

| Position 4096 metric | Production | Narrow attention HC down+up |
|---|---:|---:|
| KL(A‖B) | 0.00502608521 | 0.00656984370 |
| KL(B‖A) | 0.00645067039 | 0.00514656820 |
| Relative logit L2 | 0.0783402310 | 0.0580027486 |
| Maxabs | 1.23584795 | 1.10804147 |
| Shared top1 | 63280 | 63280 |
| Differing persistent hashes /121 | 120 | 117 |

Narrow mode makes indices 0–3 equal, while 4–120 remain different; production
only index 0 matches. Position, QSA lengths and PLE token metadata match in both
modes. Full-allocation hash differences include inactive tails and are not
numerical state-error magnitudes. Forward KL increases even as layer0 becomes
identical and relative logit L2 decreases. Eliminating this seed neither removes
all downstream schedule effects nor establishes a quality improvement. The
broader BF16-class intervention's smaller KL is a different intervention, not
an additive or production-ready result.

**Decision / next budget.** Retain this checkpoint and close the specific
"unexplained early GDN seed / possible HC leaf bug" lead for the captured GSQ
case: dispatch-dependent activation rounding explains the first down mismatch,
and the narrow pair is sufficient to remove the sampled layer0 propagation.
Do not chase layer0 bit identity as the performance objective, promote this
F32 intervention, or use matching top1 as a quality gate. The useful next
qualification is a bounded natural GSQ held-out quality/continuation screen of
the **production-arithmetic** mixed schedule against the incumbent, with
predeclared task/NLL acceptance criteria. Existing stable whole cells suggest
roughly 2–3% incremental savings, with substantial drift elsewhere; no new timing
or larger ceiling comes from these observers. If quality fails or whole savings
do not repeat, park the schedule and return to selected-QSA leaf attribution.
Further layer-by-layer localization is justified only by a concrete failed
quality case, not by nonzero KL alone. UD's matching sampled continuations are
a useful control, not GSQ quality authority.

**Reproduction and lossless retention.** No third-party Python packages:

```sh
UV_CACHE_DIR=/tmp/flash-frontier-uv-cache uv run --no-project \
  docs/bench/2026-10-09-flash-frontier/summarize_hc.py \
  docs/bench/2026-10-09-flash-frontier/gsq-hc-production.jsonl \
  docs/bench/2026-10-09-flash-frontier/gsq-hc-f32downup.jsonl \
  --output /tmp/flash-hc-summary.json
```

The retained run additionally supplied
`--archive docs/bench/2026-10-09-flash-frontier/gsq-hc-oracles-sidecars.tar.xz`.
Archive creation refuses overwrite. The archive uses standard tar hardlinks
for byte-identical files, including the shared native-weight file, then
XZ/LZMA2 with a 32 MiB dictionary. All 14 original filenames, lengths and SHA256s
round-trip exactly; the originals remain untouched. Raw total 92,719,104 bytes
becomes 33,354,224 bytes (35.97% retained). Unique uncompressed payload after
hardlink deduplication is 62,965,248 bytes. Retain the archive and
[manifest](gsq-hc-oracles-sidecars.manifest.json) for git, instead of additionally
adding the raw `.f32le`, `.f64le` or `.weights.bin` duplicates. The JSONLs stay raw.
Extraction into an empty directory with `tar -xJf` recreates both modes' names.

| Retained binding | Bytes | SHA256 |
|---|---:|---|
| Production JSONL | 222,764 | `7453a76f0af80e244eafc76c3c7b46607cc8e8ea266fe5f5620cb9b671db52a9` |
| Narrow JSONL | 216,699 | `3dd37b8ec21d45f742760228d310a7f9bb29fc2fd6ea395ce4161070c7904caa` |
| Shared native HC weights, raw | 13,148,160 | `4ef0e8f82276e3899153de27997b93ce4230e9c65b29ab71855daf5f73fcb4e8` |
| Combined sidecar archive | 33,354,224 | `b2f677bb49e331a9cf4a1db674b005fb820099fe17afbd3c73a80952b20bc645` |
| Packet executable, both modes | 46,011,064 | `dbbc1a3c562ece5a173dc132bb9af21cb42ba5fd727f728cea263cbd4e78b5fc` |

Source/metallib bindings remain the packet's historical bindings, retained in
the raw headers and generated summary; they are not asserted to equal a future
HEAD. The observer integration, HC scope and CPU-oracle sources are
`qwen4exp_runtime/tests/prefill_map/frontier_layer0.rs`,
`qwen4exp_metal/frontier_hc.rs` and
`qwen4exp_runtime/tests/prefill_map/frontier_hc_oracle.rs` under
`crates/qwen-llm/src/`. All files below describe earlier checkpoints; their
proposed narrow-HC follow-up is now completed above.

Earlier broad-class and GDN checkpoints (retained history)
--------------------------------------------------------

Checkpoint validation: all eight focused CPU tests pass in release and debug;
`cargo check --offline -j2 -p qwen-cli` passes. Both resumed artifact packets and
both layer0 observer packets completed under the normal lease and admission.
Independent adversarial review approves retaining this diagnostic checkpoint,
not promoting the schedule. Production scheduling and arithmetic defaults stay
unchanged; the existing BF16 override helper now restores its state on unwind.
Raw float sidecars remain local and git-ignored; verified lossless archives and
original-hash manifests are retained alongside the JSONL evidence.

**Uniform BF16 suffix intervention: layer0 divergence removed, endpoint divergence
remains.** The new GSQ prose packet disables the BF16 bfloat-activation dispatch
class only inside each complete suffix, equally in A and B and across all suffix
layers. The prefix/checkpoint stays production. Both intervention sidecars are
byte-identical: all 17 sections, eight captured rows and three delta-state
snapshots match. This is strong bounded evidence that the dispatch class
participates in the earlier layer0 seed mismatch. It is not a single-projection
causal isolation, an accuracy oracle, a production proposal, or a speed result.

[gsq-layer0-bf16-f32.jsonl](gsq-layer0-bf16-f32.jsonl) is retained unchanged:
184,245 bytes, SHA256
`7b892df015722f9b3bd25351dde17c663a07865b6468538a1a4388abdb1b71a8`.
The owner reports a 71-second completion and passing RAII/mode CPU tests; this
analysis did not rerun them. Header mode is `f32`, configured through
`FLASH_FRONTIER_LAYER0_BF16_ACT=f32`. Weights, command widths, quant-kernel
eligibility and router policy receive no override. The F32 label refers to the
activation path for BF16-weight matmuls, not conversion of BF16 weights to F32.
All four suffix calls are untimed diagnostics; there are no continuation steps.

| Scope witness | BF16 bfloat-activation kernel calls | BF16 F32-activation kernel calls |
|---|---:|---:|
| Production prefix2048 | 194 | 12 |
| A/off and A/on, each | 0 | 424 |
| B/off and B/on, each | 0 | 218 |

The prefix witness confirms production bfloat-activation use; suffix witnesses
confirm complete suppression only during the scoped calls. Suffix counts agree
with the retained census. Each suffix still has 48 strict-router calls, and A
retains the 48 generic N3 calls. The tagged layer0 GDN dispatch sequences,
including kernels, grids and thread shapes, exactly match the baseline for
each schedule. Both observer-off/on gates pass endpoint/state and original
dispatch concordance. These checks validate the diagnostic scope, not quality.

`summarize_layer0.py` now accepts `--baseline` to independently audit both packets
and recompute same-arm changes from the original F32LE files. It also checks the
new BF16 witnesses against retained suffix census counts. Reproduce:

```sh
UV_CACHE_DIR=/tmp/flash-frontier-uv-cache uv run --no-project \
  docs/bench/2026-10-09-flash-frontier/summarize_layer0.py \
  docs/bench/2026-10-09-flash-frontier/gsq-layer0-bf16-f32.jsonl \
  --baseline docs/bench/2026-10-09-flash-frontier/gsq-layer0.jsonl
```

Both audits pass. All 7,125,248 intervention values are finite. All 804 recorded
A/B metric fields agree with independent recomputation; every A/B maxabs, RMS,
relative L2 and different-bit count is zero. Specifically, input, QKV/gate,
alpha/beta/decay, prepared Q/K/V, recurrence, normalized output, final GDN output,
initial states and states after tokens 2048/2049/2050 all agree.

The baseline comparison matters: both initial states also match their baseline
counterparts exactly, confirming the captured state entering the suffix was
unchanged. All first-three-row tensors and state snapshots in intervention A
equal baseline A. Intervention B now equals those same values. But from 2051
onward, the intervention changes both arms relative to their production-path
baseline; it has not merely corrected B to an unchanged whole suffix A.

| Same-arm baseline → intervention, eight-row aggregate | A relative L2 / maxabs | B relative L2 / maxabs |
|---|---:|---:|
| GDN input | 0.000355647 / 0.0118227 | 0.000665218 / 0.0251446 |
| QKV | 0.000427484 / 0.0422668 | 0.000766133 / 0.0681438 |
| GDN output | 0.000572320 / 0.000922024 | 0.001182714 / 0.00143987 |
| Delta state after first3 | 0 / 0 | 0.000343741 / 0.00293636 |

Cross-packet relative L2 uses the corresponding baseline arm as denominator;
it is not the original within-packet A-normalized metric. The new A/B zero-error
result applies only to the retained layer0 rows/states and matched final layer0
state hashes, not all model activations.

**Full-model endpoint under the hybrid prefix/suffix policy.** At position 4096,
the A/B output logits remain different even with the entire suffix BF16 class
forced to the F32-activation path. This is neither a fresh whole-prefill
uniform-fallback experiment nor a timed whole-performance result.

| A/B endpoint metric | Production suffix baseline | Uniform F32-activation suffix |
|---|---:|---:|
| KL(A‖B) | 0.00502608521 | 0.000158449810 |
| KL(B‖A) | 0.00645067039 | 0.000172168944 |
| Relative logit L2 | 0.0783402310 | 0.0723610708 |
| Max absolute logit difference | 1.23584795 | 1.32744884 |
| Shared top1 | 63280 | 63280 |
| Differing persistent allocation hashes /121 | 120 | 117 |

KL decreases by 96.85%/97.33%, but L2 decreases only 7.63% and maxabs increases
7.41%. These are consistency changes between two new trajectories, not measured
accuracy gains against a reference. Finite logits and position/QSA/PLE metadata
agree. In the intervention, persistent indices 0–3 match: layer0 convolution,
layer0 delta state, PLE state and layer1 convolution. Index 4 (layer1 delta state)
is the first differing allocation in retained ordering; indices 4–120 and final
hyper differ. This is not a trace of the first temporal divergence. Raw endpoint
logit vectors are not retained, so endpoint KL is the harness result and
cross-packet baseline-versus-intervention endpoint KL cannot be recomputed.

**HC metadata now confirms the plausible upstream class.** Both layer0 attention
and FFN HC down weights are BF16 `[10240,320]`; both up weights are BF16
`[320,10240]`. Their norms and injection weights are F32. GDN itself remains
Q6_K QKV/output, Q4_K gate and F32 alpha/beta; no BF16 GDN projection was changed.
These metadata plus the disappearance of the mixed-input seed strengthen the
HC arithmetic hypothesis. However, all eligible suffix layers were changed at
once. The packet does not distinguish attention HC down versus up, prove that
these two projections alone explain the endpoint effect, or exclude another
upstream interaction. N64-versus-non-N64 GDN quant dispatches remain different
between schedules yet produce equal captured outputs on the now-equal inputs.
That is useful bounded negative evidence against blaming those GDN quant kernels
for the original captured seed.

**Checkpoint recommendation.** This is a useful stopping point for the broad
class intervention: it answers whether that class can remove the captured
layer0 mismatch. Do not spend another whole-timing matrix or infer promotion
from lower KL. If one more causal screen is desired, keep production routing,
prefix and all other suffix math, and restrict F32-activation dispatch to the
identified layer0 attention HC down+up weights in both schedules. Reuse the
eight-row/tape observer and within-arm concordance. If that reproduces the seed
removal, narrow down/up individually only if the distinction would change an
implementation decision. If the residual full-model difference becomes the
priority, separately inspect layer1 input/state boundaries rather than assuming
the remaining 117 hash differences all originate in QSA. A broader natural
quality/continuation evaluation remains necessary before any production policy
decision; no arbitrary cross-schedule bit-identity gate is introduced.

**Bindings and compression.** This is a new observer executable, 45,782,456 bytes,
SHA256 `f4fb777333f739b68e1f247ce4ce7a8acfa8eda5e5ecec0b229ee1732be5b518`.
Packet and dispatch source hashes changed for the intervention/RAII work;
capture, GDN and metallib hashes match the preceding layer0 capture. Preserve
each packet's own bindings. Both raw 14,250,496-byte sidecars have SHA256
`1b8459106c945f9cd4fae1da83a7f0821c6f5b6962f57013bf8bbbb6116547c1`.
Their equality is verified from bytes, not only from the reported metrics.
The verified standard
[XZ archive](gsq-layer0-bf16-f32-sidecars.tar.xz) is 13,364,996 bytes (12.75 MiB),
**53.11% smaller** than the 28,500,992 raw bytes; gzip saves only 5.67%.
The 32 MiB XZ dictionary reuses the identical second bank. The
[manifest](gsq-layer0-bf16-f32-sidecars.manifest.json) retains original member
names/sizes/hashes and archive hash
`019d0aba6f6f244da33abc5779994beabbce8de296dfb07af3bbb54fa32aca88`.
Both members round-trip exactly. All raw evidence remains untouched; no GPU,
builds, model-source edits, deletions or commits were performed for this analysis.

**Baseline layer0 capture (before the class intervention): first recorded
divergence is upstream of GDN.** The
untimed GSQ prose capture starts with exactly equal convolution and delta states,
but its mixed GDN input already differs at positions 2048–2050. Thus these data do
not identify GDN recurrence, GDN quant projection, or selected QSA as the first
source. The next bounded localization should move one stage earlier into the
layer0 HC/residual mix. No production promotion or quality acceptance follows
from this observer packet.

[gsq-layer0.jsonl](gsq-layer0.jsonl) is retained unchanged (187,165 bytes, SHA256
`9db580f381943324463181fa81176ab14016e2e15f11abfa14fab39e37dbe94f`).
The owner reports 47 seconds and six passing CPU tests; neither was rerun here.
The packet executes one prefix 2048 and restored Aoff/Aon/Boff/Bon suffixes with
unchanged ordinary N3+N2045 versus N2048 commands and production routing. It has
**no timed performance arms or continuation/quality screen**. The observer
captures eight absolute rows 2048..2055 plus initial states and delta states after
each of the first three tokens, without splitting B's recurrence into commands.

[summarize_layer0.py](summarize_layer0.py) independently reads little-endian F32
sidecars, verifies whole-file and all 34 section hashes/shapes/offsets, checks
every captured value for finiteness, and recomputes aggregate/per-row metrics
using Python binary64 and `math.fsum`. It checks 804 recorded metric fields;
all agree, with largest relative summation-order discrepancy 1.521e-12. This
audit tolerance only accommodates summation order, not model-quality error.
Both original sidecars contain 3,562,624 floats (14,250,496 bytes) each;
**all 7,125,248 captured values are finite**. Audit reports zero problems.

```sh
UV_CACHE_DIR=/tmp/flash-frontier-uv-cache uv run --no-project \
  docs/bench/2026-10-09-flash-frontier/summarize_layer0.py \
  docs/bench/2026-10-09-flash-frontier/gsq-layer0.jsonl
```

Initial delta state is identical across all 786,432 elements; initial convolution
history is identical across all 30,720 elements. All 2560 GDN input elements
differ in each of the first three rows. Input row 2048 has relative L2
0.001070762 and maxabs 0.02514458. Input, QKV, gate, beta-after-sigmoid, alpha, and
decay become bit-identical at every captured row 2051–2055. Convolved Q/K/V and
their normalized Q/K remain different through 2053 and match at 2054–2055, consistent
with the four-tap convolution carrying earlier differences. Recurrence, gated
normalization and output remain different through the last captured row 2055.
The observer is bounded to these eight rows: it does not establish when the
entire later sequence reconverges or which stage dominates the endpoint KL.

| Absolute row | Input relative L2 | QKV relative L2 | Normalized Q relative L2 | Recurrent relative L2 | GDN output relative L2 |
|---|---:|---:|---:|---:|---:|
| 2048 | 0.001070762 | 0.001524267 | 0.001703428 | 0.000600960 | 0.001963227 |
| 2049 | 0.000425623 | 0.000454904 | 0.000764366 | 0.000426729 | 0.001087686 |
| 2050 | 0.000980894 | 0.001001946 | 0.001437275 | 0.000404609 | 0.001413681 |
| 2051 | 0 | 0 | 0.000381105 | 0.000091061 | 0.000275666 |
| 2052 | 0 | 0 | 0.000241262 | 0.000046180 | 0.000182561 |
| 2053 | 0 | 0 | 0.000193219 | 0.000058533 | 0.000228390 |
| 2054 | 0 | 0 | 0 | 0.000022836 | 0.000043641 |
| 2055 | 0 | 0 | 0 | 0.000022778 | 0.000038277 |

At 2055, recurrent maxabs is 0.000324249 and output maxabs is 0.000026254;
different-bit counts are 4388/6144 and 2557/2560 respectively. Near-ubiquitous bit
differences alone are not an error-magnitude or quality criterion. The three
full delta-state snapshots establish a state difference already after the first
row, starting from the equal initial state:

| State after processing | Different elements / 786432 | Relative L2 | Maxabs | RMS error |
|---|---:|---:|---:|---:|
| 2048 (first token) | 785656 | 0.000550261 | 0.004459143 | 0.000024978 |
| 2049 (second token) | 786186 | 0.000387096 | 0.002370834 | 0.000017711 |
| 2050 (after three; next position 2051) | 786247 | 0.000343749 | 0.002936363 | 0.000015717 |

These snapshots do not isolate a recurrence arithmetic defect: Q/K/V, decay and
beta already differ before the recurrence sees them. Matching initial state
rules out an initial-state mismatch in this capture, not every possible
upstream/input-path cause.

**Actual layer0 dtypes and kernel evidence.** QKV is Q6_K `[2560,10240]`, gate
Q4_K `[2560,6144]`, beta and alpha F32 `[2560,48]`, and output Q6_K `[6144,2560]`.
Convolution, A, dt-bias and norm are F32. There are **no BF16 projections inside
the captured GDN**. A's N3 and N2045 calls each use the non-N64 Q6_K/Q4_K
matmuls; B uses their N64 kernels at N2048. The two F32 projection calls and
`kernel_gdn_step_decay_packed_nsg4_f32` remain the same kernel families. Recurrence
grid is `[32,48,1]`, threads `[32,4,1]` for all three command widths; loop length
and history still differ. Full tagged GDN kernel/grid/thread sequences are in
the JSONL and independent report.

The quant kernel switch is real, but QKV and gate outputs at 2051–2055 are exact
on equal captured inputs despite that switch. This is bounded evidence against
calling N64 projection itself the demonstrated source of the first difference;
it does not prove equivalence for arbitrary inputs or later layers. The earlier
BF16 hypothesis must move upstream: width-dependent BF16 rounding inside HC is
plausible, but this packet does not retain HC weight dtypes, intermediate values,
or a causal counterfactual to establish it.

[Layer0 composition](../../../crates/qwen-llm/src/qwen4exp_layers_zero_one.rs#L922)
passes `attention.mixed()` from
[packed gated residual mix](../../../crates/qwen-llm/src/qwen4exp_metal.rs#L718)
directly into GDN. Next capture the same eight rows at hyper-input, HC-normalized
input, down projection/low activation, up projection/raw gate, and mixed output;
retain actual HC dtypes and tagged dispatches. First establish equal pre-HC
inputs. Then a narrow diagnostic arithmetic override at the first divergent
projection can test causality on identical inputs, preserving full command
widths and current routing. Do not change GDN recurrence or QSA based solely on
this packet. Any later promotion still needs a natural continuation/quality
screen; this capture adds localization, not quality authority.

**Observer/admission/source limits.** Both schedules pass exact observer-off/on
endpoint and persistent/hyper-state concordance. The analysis independently
checks the retained endpoint hashes/state records. The harness also asserts the
original dispatch sequence, geometry and order match after excluding 30 A / 16 B
tagged copy dispatches; complete original dispatch-order equivalence is a harness
observation, not reconstructed from only the retained aggregate census. The
checkpoint recurrence uses the existing kernel/grid and emits three state
snapshots without adding command boundaries. Agreement in this run does not
make observation overhead a performance result. Cross-schedule endpoint KL
0.00502609/0.00645067 and relative L2 0.0783402 reproduce the earlier GSQ prose
endpoint; these logit metrics remain recorded, not recomputed from GDN sidecars.

Capture admission prices 14,303,232 GPU bytes for 14,250,496 logical bytes and
352,172,032 CPU upper bytes including the checkpoint, two CPU capture copies,
and margin. Observed capture allocation 14,254,080 bytes is below its price.
Normal lease, model/session/capture memory gates and shard revalidation pass.
This is a newly compiled observer binary: executable 45,760,040 bytes, SHA256
`37863f8a7ec5ef380b60be39f2963b6b6069ab6a6805584f226293ec344df476`.
The packet/capture/GDN source bindings differ from the timing binary; the
metallib hash is unchanged. The old performance packets retain their own
historical bindings. No executable was copied into this evidence directory.

**Lossless retention.** The raw sidecars total **28,500,992 bytes (27.18 MiB)**.
A deterministic tar with gzip9 is 26,884,221 bytes, saving only 5.67%; a standard
tar with XZ/LZMA2 preset6 and 32 MiB dictionary is **20,792,856 bytes (19.83 MiB)**,
saving **27.05%**. Both compression trials were decompressed and every extracted
member SHA256 verified against the original JSONL. The recommended
[gsq-layer0-sidecars.tar.xz](gsq-layer0-sidecars.tar.xz) and
[hash manifest](gsq-layer0-sidecars.manifest.json) are retained durably. Originals
are still present and unchanged; nothing was deleted or committed. For eventual
git retention, use the archive plus manifest/JSONL/script rather than staging
both compressed and uncompressed copies. This is a moderate reduction, not an
order-of-magnitude compression claim.

| Original member | Bytes | SHA256 |
|---|---:|---|
| gsq-layer0.jsonl.A.f32le | 14250496 | `f0c9ed5c60236f2ac5ccc15d9e42a505b85bec38ff06fdd6d80b68220a379266` |
| gsq-layer0.jsonl.B.f32le | 14250496 | `9496952d7fb564b7c7180b15da9dbf29bccb25be57082adc6c2f2183f8134664` |

Archive SHA256 is
`cafa88738b1479cadcaff7ccd7a45a7f10a435d78bb07475e81a8e6ee391e2bc`.
To independently repeat compression/roundtrip checks, add
`--compression-dir /tmp/flash-layer0-new-compression-trial` to the analysis command;
archive filenames must not already exist. To analyze an archive-only checkout,
extract its two members beside the JSONL first. No custom float codec or lossy
transformation is used.

**Earlier timing and qualification evidence (retained).**

Both artifacts now support a useful residual scheduling opportunity, but the
gain should be described using the paired results and drift, not pooled means.
UD's stable second prose whole round saves **197.11 ms GPU / 2.412%** and
**198.59 ms wall / 2.424%**. GSQ's stable second SSH whole round saves
**257.06 ms GPU / 3.070%** and **259.07 ms wall / 3.085%**. These are explicitly
selected low-drift observations alongside every retained attempt below, not a
new filtered benchmark score. Suffix and whole savings overlap; the old router
transfer saving cannot be added to them.

UD is an informative numerical control: its A/B logits and persistent state
match exactly on both streams, both stages, endpoint and four continuations.
GSQ retains reproducible differences beginning before QSA. **Prioritize the
early-GDN observer and quality screen; no production promotion.** Exact UD
agreement is an observation, not a new acceptance requirement for other artifacts.

**Completed UD restart.** [ud-both-restart.jsonl](ud-both-restart.jsonl) completed
after the GSQ-only analysis, directly in durable storage. The owner reports
406 seconds; the packet does not measure process-total elapsed time. Raw bytes
are unchanged: **3,166,239 bytes**, SHA256
`3f201d32619ecc6b22ad0bbe1b443ebeb6e76ff05d38c366240d355d099d488e`.
Compiled-source bindings and executable SHA256 match both completed GSQ packets.
The same 4100 token IDs are used for each corresponding corpus across artifacts.
All three UD shard stamps revalidate unchanged. Native admission passes with
61,175,349,248 observed weight bytes, 2,161,442,816 session bytes, selected
capacity2048/extent4100, and the same checkpoint bound and reserve as GSQ.
Normal lease and diagnostic memory gates pass; process-budget admission remains
`AdmittedProcessBudgetOmitted`, not a peak-RSS guarantee.

```sh
UV_CACHE_DIR=/tmp/flash-frontier-uv-cache uv run --no-project \
  docs/bench/2026-10-09-flash-frontier/summarizer.py \
  docs/bench/2026-10-09-flash-frontier/ud-both-restart.jsonl
```

Audit passes without problems: 32 timed attempts, eight excluded warm witnesses,
170 finite logit vectors, 128 continuation steps, 164 finite comparison records,
and matching causal metadata. The existing multi-case summarizer handles this
packet unchanged. All timings have complete command GPU coverage. The same
ordinary-driver and outside-timer preparation contract described for GSQ applies.

| UD cell | GPU A → B mean ms | GPU saving ms (%) | Wall A → B mean ms | Wall saving ms (%) |
|---|---:|---:|---:|---:|
| Prose suffix | 4951.977 → 4937.014 | 14.963 (0.302%) | 4972.745 → 4959.954 | 12.791 (0.257%) |
| Prose whole4096 | 8206.183 → 7974.814 | 231.370 (2.819%) | 8227.307 → 7995.706 | 231.602 (2.815%) |
| Repeated SSH suffix | 4795.294 → 4605.217 | 190.077 (3.964%) | 4811.068 → 4621.048 | 190.020 (3.950%) |
| Repeated SSH whole4096 | 9218.821 → 9029.745 | 189.075 (2.051%) | 9260.484 → 9066.959 | 193.525 (2.090%) |

UD median GPU savings are -2.606%, 2.412%, 3.967%, and 2.223% in the table's
order; wall medians save -2.723%, 2.424%, 3.989%, and 2.358%. The prose suffix
mean/median disagreement is a warning against treating the aggregate as a clean
mechanism estimate. All attempts remain included in the report.

Paired savings below are A1/B1 then A2/B2; drift is A2/A1 then B2/B1, all in
percent. Execution order is A1, B1, B2, A2. All pairs, including losses, are shown.

| UD cell / round | Paired GPU savings | Paired wall savings | GPU A/B drift | Wall A/B drift |
|---|---:|---:|---:|---:|
| Prose suffix 1 | +3.533 / -11.266 | +3.254 / -11.248 | -12.169 / +1.306 | -12.178 / +0.986 |
| Prose suffix 2 | +4.328 / +4.135 | +4.397 / +4.176 | -0.278 / -0.077 | -0.307 / -0.076 |
| Prose whole 1 | +4.143 / +2.287 | +4.091 / +2.300 | -1.837 / +0.064 | -1.771 / +0.062 |
| Prose whole 2 | +2.412 / +2.412 | +2.421 / +2.426 | +0.012 / +0.012 | +0.017 / +0.011 |
| SSH suffix 1 | +3.927 / +3.980 | +3.788 / +3.994 | +0.071 / +0.016 | +0.233 / +0.019 |
| SSH suffix 2 | +3.938 / +4.010 | +3.973 / +4.042 | +0.058 / -0.017 | +0.061 / -0.011 |
| SSH whole 1 | +4.805 / -0.658 | +4.743 / -0.716 | +15.866 / +22.515 | +16.112 / +22.766 |
| SSH whole 2 | +2.740 / +1.697 | +2.864 / +1.844 | -1.692 / -0.638 | -1.695 / -0.663 |

UD has **14/16 improving GPU pairs and 14/16 improving wall pairs**, not all-pair
qualification. Prose suffix round1 loses 173.02 ms GPU on average while A speeds
up by 12.17%; round2 saves 202.95 ms / 4.232% with much smaller drift. SSH whole
round1 contains a 22.52% B slowdown and one losing pair; it is not a stable gain
estimate. Its second round saves 208.84 ms GPU / 2.223% and 222.91 ms wall /
2.358%, but A's 1.69% drift is still appreciable relative to that saving.
UD prose whole round2 and both SSH suffix rounds give cleaner evidence of a
roughly 190–200 ms residual; the unstable results neither erase those observations
nor qualify an artifact-wide percentage.

**UD numerics, state and routing.** All 80 timed cross-schedule step comparisons
have exactly zero KL in both directions, relative L2, maxabs and choice regret;
logit bits and complete state digests agree. The four warm A/B endpoint
comparisons agree too. No persistent allocation hashes differ (0/121), and final
hyper matches, at every compared step. All 80 same-schedule A1/A2 and B1/B2 step
comparisons also agree. Independently checked hashes match across rounds, warm
versus measured endpoints, and suffix versus whole (20 corpus/arm/step groups).
All positions, QSA lengths and PLE prior-token metadata agree. These results are
bounded to the two retained repeated-text streams and four continuations.

| UD corpus | Shared top1 IDs: endpoint, then continuation1–4 | Worst bidirectional KL / L2 / maxabs / regret |
|---|---|---:|
| Prose | 63280, 10993, 11, 321, 37715 | All zero |
| Repeated SSH | 7854, 539, 4924, 20653, 279 | All zero |

Both UD suffix arms witness 48 strict-router calls; both whole arms witness 96.
A additionally has the 48 generic N3-router calls. Warm dispatch totals are
9838→8021 suffix and 13983→12166 whole for both corpora. Unlike GSQ, the UD suffix
census contains no N64 quant matmuls or bfloat-activation BF16 calls in either
arm; ordinary BF16 matmul calls are 36→24. Actual expert-down cohorts are
43 IQ4_NL and five Q8_0; gate/up are 47 IQ3_XXS and one IQ4_XS. Thus the exact UD
result is a useful control for the common scheduling/checkpoint path and points
toward artifact-specific projection arithmetic as a GSQ localization priority.
It does not isolate a single causal kernel: weight values/dtypes differ as well.
No token-level route-equivalence observation is retained.

**Repeat budget and next action.** Another full `both` stages/`both` corpora packet
is not worth running before the early-GDN observer: both artifacts already show
a usable low-drift whole-cell signal, and repeating timings will not explain
GSQ's quality difference. Do the GSQ early-GDN localization below first; use UD
as an optional control if it helps discriminate a proposed cause. Keep routing
production in every arm and keep all captures outside performance timing.

After localization, if the candidate remains viable, the useful timing repeats
are **GSQ prose whole** and **UD SSH whole**, the least settled whole cells.
The existing harness supports `FLASH_FRONTIER_STAGE=whole` with
`FLASH_FRONTIER_CORPORA=prose` or `both`; it cannot select SSH alone. Without a
harness change, use GSQ whole/prose and UD whole/both, retaining the prose control
rather than rerunning suffixes. Two warmed ABBA rounds remain a bounded screen.
Current traces contain about 87 seconds of GSQ prose whole calls and 171 seconds
of UD both-corpus whole calls including warm calls; loading, state inspection,
continuations and other overhead are additional. Budget several minutes, not
another full suffix-plus-whole matrix. Do not repeat until a favorable subset
appears: predeclare the two rounds, retain every pair, and if drift remains of
the same order as the gain, report timing magnitude unresolved and investigate
the execution conditions. No arbitrary bit-identity or new numerical cutoff
is implied. Production promotion remains deferred.

**GSQ restart analysis (retained below).**

The completed restart confirms a useful **ordinary whole4096 performance signal
on both GSQ corpora**, with substantial timing drift that limits the headline
mean. All 16 GPU and all 16 wall pairs improve across suffix and whole cells.
The stable second SSH whole round saves **257.06 ms GPU / 3.070%** and
**259.07 ms wall / 3.085%**. Do not promote yet: **early layer0 GDN localization
and a quality screen are the next steps regardless of matching top1**. The
candidate changes width-dependent arithmetic as well as command scheduling.
Production remains the incumbent split planner.

**Recovery provenance.** The owner reported a computer shutdown after the initial
retained GSQ prose suffix run. Worktree, compiled binary, and that retained packet
survived; incomplete GSQ/UD files under `/tmp` disappeared and no benchmark
process survived. Those interrupted attempts have no retained results and are
not scored, silently reconstructed, or merged into this packet. After recovery,
the owner reported AC power, mounted drive, a passing scope CPU test, and a new
GSQ `both` stages / `both` corpora run completing in 403 seconds. Elapsed process
time and AC status are owner reports, not fields measured by the JSONL.

[gsq-both-restart.jsonl](gsq-both-restart.jsonl) was written directly to this
durable directory and is retained unchanged: **3,174,540 bytes**, SHA256
`2bd1efe0efdc641c42d728af66859c45272666867e2cce06c8362501d9566660`.
Its complete compiled-source binding and executable SHA256 exactly match the
initial retained packet. The new packet completed with unchanged shard stamps,
normal lease and memory gates, native selected-capable extent4100, and current
production routing in both arms. Its weight/session/checkpoint allocation counts
match the initial packet below. The recorded recommended working-set signal
changed from 118 GiB to 96 GiB after recovery; normal admission still passed.
This does not identify the cause of timing drift. No clock/thermal telemetry is
retained. At the GSQ-only analysis checkpoint UD was running and its partial
durable file was **not read**. Its subsequent completed analysis is above.

Reproduce this analysis, naming only the completed file:

```sh
UV_CACHE_DIR=/tmp/flash-frontier-uv-cache uv run --no-project \
  docs/bench/2026-10-09-flash-frontier/summarizer.py \
  docs/bench/2026-10-09-flash-frontier/gsq-both-restart.jsonl
```

The summarizer now requires explicit input paths, preventing automatic discovery
of a benchmark's growing JSONL. Multi-case audit passes with **zero problems**:
32 timed attempts, eight excluded warm census attempts, 170 finite logit vectors,
128 continuation steps, and 164 finite comparison records with matching causal
metadata. All timed GPU aggregates have complete command coverage. Suffix A/B
have two/one commands; ordinary whole A/B have three/two. Whole calls begin from
reset and zeroed persistent state in one resident session. Preparation, logging,
endpoint inspection and continuations remain outside timing; this is not cold
placement/TTFT. The schema does not retain individual command GPU timestamps.

| Restart cell | GPU A → B mean ms | GPU saving ms (%) | Wall A → B mean ms | Wall saving ms (%) |
|---|---:|---:|---:|---:|
| Prose suffix | 5459.719 → 5127.139 | 332.580 (6.092%) | 5475.777 → 5143.482 | 332.295 (6.068%) |
| Prose whole4096 | 8914.343 → 8346.396 | 567.947 (6.371%) | 8934.083 → 8366.192 | 567.891 (6.356%) |
| Repeated SSH suffix | 5077.117 → 4729.870 | 347.247 (6.839%) | 5093.394 → 4746.507 | 346.887 (6.811%) |
| Repeated SSH whole4096 | 8632.610 → 8160.614 | 471.996 (5.468%) | 8653.189 → 8180.577 | 472.611 (5.462%) |

All attempts remain included. Median GPU A→B is 5466.589→5122.357 ms for prose
suffix (6.297%), 8879.583→8387.530 ms for prose whole (5.541%),
4948.890→4718.667 ms for SSH suffix (4.652%), and 8386.648→8117.503 ms for SSH
whole (3.209%). Corresponding wall median savings are 6.338%, 5.509%, 4.659%,
and 3.220%. Means/medians are descriptive, not a calibrated confidence interval.

In the following table, paired savings are A1/B1 then A2/B2; drift is A2/A1 then
B2/B1. Each entry is a percentage. Execution order remains A1, B1, B2, A2.

| Cell / round | Paired GPU savings | Paired wall savings | GPU A/B drift | Wall A/B drift |
|---|---:|---:|---:|---:|
| Prose suffix 1 | 2.686 / 6.691 | 2.573 / 6.645 | +8.313 / +3.856 | +8.466 / +3.933 |
| Prose suffix 2 | 7.436 / 7.386 | 7.471 / 7.409 | -3.495 / -3.443 | -3.504 / -3.438 |
| Prose whole 1 | 9.629 / 3.366 | 9.550 / 3.381 | -9.201 / -2.909 | -9.095 / -2.895 |
| Prose whole 2 | 1.113 / 10.569 | 1.135 / 10.562 | +11.599 / +0.928 | +11.577 / +0.938 |
| SSH suffix 1 | 12.149 / 3.981 | 12.022 / 3.965 | -9.376 / -0.951 | -9.214 / -0.900 |
| SSH suffix 2 | 5.327 / 5.332 | 5.356 / 5.353 | -0.015 / -0.020 | -0.024 / -0.021 |
| SSH whole 1 | 11.688 / 3.288 | 11.624 / 3.312 | -10.731 / -2.240 | -10.646 / -2.242 |
| SSH whole 2 | 3.226 / 2.913 | 3.241 / 2.929 | -0.289 / +0.033 | -0.277 / +0.045 |

The full prose mean is particularly uncertain: A changes -9.2% then +11.6%
within rounds; individual whole GPU savings range 94.88–1005.40 ms. SSH's first
whole A1 is also slow. Its second whole round has much smaller drift and saves
257.06 ms on average; second suffix round saves 262.88 ms GPU / 5.329% and
265.15 ms wall / 5.354%. These stable-round observations support the original
roughly quarter-second residual, but are not grounds for discarding other
attempts. Whole and suffix savings overlap and must never be added; nor should
the new larger means be called an improvement over the earlier packet.

**Restart numerics and repeatability.** Every recorded cross-schedule comparison
agrees in top1 with zero regret in both directions. All 80 same-schedule A1/A2
and B1/B2 step comparisons have zero KL/L2/maxabs and identical logits/state.
The summarizer additionally verifies hashes across both rounds and warm versus
measured endpoints, and across **suffix versus fresh whole** for each arm and
corpus (20 step/arm/corpus groups, all equal). Thus restoring the prefix is not
creating the observed numerical difference in this evidence.

For prose, all endpoint and four-continuation metrics exactly reproduce the
initial table below, in both stages: worst KL(A‖B)/KL(B‖A) is
0.0429139/0.0950802 at continuation4; worst relative L2 0.137688 and maxabs
2.777336 occur at continuation3. SSH is a distinct second stream: 2211 retained
tokens are decoded with the identity-checked tokenizer, round-tripped exactly,
then the whole text is repeated twice and truncated to 4100 tokens. Used-token
SHA256 is `675aa61981cb1e69719263a3b10793b53e2fffc343d58e6602c7e7c82d695329`.
It is not an uninterrupted natural 4K excerpt. SSH observations are identical
across suffix and whole stages:

| SSH step | KL(A‖B) | KL(B‖A) | Relative L2 | Max absolute logit difference | Shared top1 ID |
|---|---:|---:|---:|---:|---:|
| Prefill endpoint | 0.0000111972 | 0.0000132917 | 0.0720061 | 2.655254 | 7854 |
| Continuation 1 | 0.0000426542 | 0.0000465782 | 0.0503001 | 0.624812 | 539 |
| Continuation 2 | 0.0000219934 | 0.0000281090 | 0.0672371 | 1.528862 | 4924 |
| Continuation 3 | 0.0000113826 | 0.0000135694 | 0.0431042 | 0.951006 | 20653 |
| Continuation 4 | 0.000550126 | 0.00102399 | 0.0511982 | 1.788044 | 279 |

Cross-schedule persistent hashes still differ at 120/121 allocations at every
step of both corpora/stages, including layer0 GDN delta state. Only layer0
convolution state (index0) agrees, and final hyper differs. Metadata agrees.
These are hash observations, not state-error magnitudes, element counts or a
state-finiteness scan. Matching greedy choices and much smaller SSH KL do not
remove the need to understand the pre-QSA seed difference. Numeric metrics come
from harness comparisons; raw logit/state values are not retained for independent
recalculation.

**Mechanism and next qualification.** Both suffix arms witness 48 strict-router
calls; both whole arms witness 96. The N3 shoulder adds 48 generic router calls
to A. Reviewer findings are corroborated by source/census: changing the command
width also changes eligible projection arithmetic. In the prose suffix census,
A has no N64 quant matmuls; B has 47 Q4_K, 35 Q5_K, and 128 Q6_K N64 calls.
N2048 is divisible by64, N2045 is not; see
[quant selection](../../../crates/qwen-llm/src/metal/mat_mat.rs#L2004).
Eligible BF16 projections use the bfloat-activation path at N≥16, while N3 uses
the other BF16 path; see
[BF16 dispatch](../../../crates/qwen-llm/src/metal_forward/dispatch.rs#L531).
Suffix A/B census counts are 194/194 bfloat-activation calls and 230/24 other
BF16 matmul calls. These totals show path changes, not measured stage shares.
The layer0 GDN delta-state difference precedes QSA, so later selection cannot
be the sole source of divergence; it may amplify an earlier difference.

The next bounded diagnostic should restore the same prefix and compare layer0
input, QKV/gate/beta/alpha projections, decay, prepared normalized Q/K/V,
recurrent output and delta state, aligned by absolute row across A's concatenated
3+2045 and B's 2048. Begin with the first three rows, the 2051 boundary, and final
state. Verify matching input/state before attributing a projection error. The
[packed GDN sequence](../../../crates/qwen-llm/src/qwen4exp_gdn.rs#L880) supplies
the existing stage boundaries. An intermediate recurrence state at position2051
inside B may require a narrow observer hook; final-state hashes alone do not
localize the first divergence. Admit any capture memory and keep observer runs
outside performance arms. If inputs to recurrence already differ, isolate the
width-dependent BF16/quant projection path before blaming the recurrence; if
they agree, test recurrence across the split using identical prepared inputs.
Do not assume equal final convolution-state hashes establish equality at every
earlier row.

Require this early-GDN localization **now, regardless of top1**, then a natural
quality/continuation screen covering the identified arithmetic change before
promotion. Compare finite values, meaningful error magnitudes, KL and choice
regret; do not invent a bit-identity gate. Retain the original rollback/production
schedule while qualifying. The completed UD evidence above adds bounded artifact
transfer but does not resolve GSQ's numerical difference. If a credible whole
saving disappears on stable repeats, wall
regresses, or the quality screen reveals consequential degradation, demote or
rework the candidate. No claims yet cover placement-cold behavior, 8K/deeper
requests, or short prompts below the frontier. No GPU, builds, production-source
edits or commits were performed for this analysis.

**Initial retained prose suffix screen (pre-shutdown; historical).**

The current-production-router suffix screen supports **expanding to ordinary
whole4096 confirmation and UD**, not promoting the schedule yet. Across two
warmed ABBA rounds, merging the dense shoulder saves **266.63 ms GPU (5.382% of
the measured suffix)** and **265.86 ms ordinary-call wall (5.348%)**. Every paired
GPU and wall comparison improves. Numerical differences are deterministic and
top1 agrees through four continuations, but distribution/state differences are
material enough to retain a qualification flag.

This is a diagnostic-only scheduling scope. Production remains the incumbent
split planner. The owner reported the scope CPU test passed and the leased
packet completed in about 225 seconds; this analysis did not run Rust tests,
builds, or GPU work. The JSONL establishes successful completion, lease/admission,
and the observations below; it does not contain total process elapsed time.

**Retained evidence and reproduction.** [gsq-prose-suffix.jsonl](gsq-prose-suffix.jsonl)
is a byte-for-byte copy of `/tmp/flash-frontier-gsq-suffix-20261009.jsonl`:
828,656 bytes, SHA256
`577c021c08f5d177067f9b80800559a994b4ef8a01d747f418166cc727dd751d`.
No raw fields were rewritten. Recompute the report from the repository root:

```sh
UV_CACHE_DIR=/tmp/flash-frontier-uv-cache uv run --no-project \
  docs/bench/2026-10-09-flash-frontier/summarizer.py \
  docs/bench/2026-10-09-flash-frontier/gsq-prose-suffix.jsonl
```

The dependency-free summarizer recalculates timings from individual
`arm_complete` records, excluding both warm census attempts. It checks complete
ABBA coverage, valid command GPU coverage, endpoint positions, finite reported
logits, causal metadata, persistent hash comparisons, router census totals,
admission, shard revalidation, and final completion. It exits nonzero on audit
problems. On this packet: **zero audit problems**. KL/L2/maxabs/regrets are
aggregated from harness-computed full-logit comparisons: raw logits and state
values are not retained, so their hashes cannot independently reproduce those
numerical metrics. Hash equality/differences are independently recomputed from
the retained records.

**Measurement contract.** The test is
`qwen4exp_runtime::tests::prefill_map::frontier_schedule::native_frontier_schedule`,
with `FLASH_FRONTIER_STAGE=suffix`, `FLASH_FRONTIER_CORPORA=prose`,
`FLASH_PREFILL_MODEL` selecting the first shard below, and a new
`FLASH_PREFILL_OUT`. The packet records Apple M4 Max, a production benchmark lease
with wired gate passed, release execution, and no Metal debug layer. Both router
and selected-QSA environments were unset/default-on; the strict router pipeline
was supported. There is no N2045 router override.

```text
/Volumes/wdblack/weights-archive/qwen3.8-flash-next-gsq-rco-iq3_s/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
```

The source prose is the retained natural-n512 roadmap text, repeated whole
eight times with two-newline separators and tokenized without added specials.
The first 4100 token IDs are retained, covering prefill4096 plus four handoff
tokens. Tokenizer identity matches the released fixture identity. Used-token
SHA256 (u32 little endian) is
`0cc71705ddd73b629e2e0f1bf60bbaa8ecf9b4b3ff32e469719644f5737ce7aa`.
This is one repeated natural stream, not eight independent quality cases.

A common normal prefix2048 is executed once, then checkpointed. Each arm restores
that checkpoint before its timer. A executes `[2048,2051)` then `[2051,4096)`;
B executes `[2048,4096)`. Both use the ordinary runtime driver. GPU is the sum of
complete valid command intervals; wall surrounds the ordinary continuation call
and includes its normal staging/encoding/completion/publication. Restore,
checkpoint, logging, readback/hashing, and four continuation steps are outside
timing. There are no profilers/censuses inside timed arms. Eight timed attempts
have valid GPU coverage (A: two samples each; B: one). Individual raw GPU start/end
timestamps are not exposed by this schema.

| Timing | A mean ms | B mean ms | Saved ms | Saving | A median ms | B median ms |
|---|---:|---:|---:|---:|---:|---:|
| Suffix GPU | 4954.404 | 4687.774 | 266.630 | 5.382% | 4950.883 | 4677.395 |
| Ordinary-call wall | 4971.642 | 4705.783 | 265.859 | 5.348% | 4965.479 | 4695.052 |

Percentages use `1 - B/A`; aggregate percentages use the ratio of arm means,
not an unweighted average of pair percentages. Median reductions are 5.524%
GPU and 5.446% wall.

| Round / paired arms | GPU A → B ms | GPU saved ms (%) | Wall A → B ms | Wall saved ms (%) |
|---|---:|---:|---:|---:|
| 1 A1/B1 | 4965.647 → 4689.788 | 275.858 (5.555%) | 4974.634 → 4706.678 | 267.956 (5.386%) |
| 1 A2/B2 | 4936.120 → 4665.002 | 271.118 (5.493%) | 4956.325 → 4683.426 | 272.900 (5.506%) |
| 2 A1/B1 | 4988.413 → 4740.587 | 247.826 (4.968%) | 5009.291 → 4757.477 | 251.814 (5.027%) |
| 2 A2/B2 | 4927.437 → 4655.720 | 271.717 (5.514%) | 4946.318 → 4675.551 | 270.767 (5.474%) |

Actual execution order is A1, B1, B2, A2. Round means save 273.488 / 259.771 ms
GPU (5.524% / 5.240%) and 270.428 / 261.291 ms wall (5.446% / 5.249%).
GPU A2/A1 drift is -0.595% / -1.222%; B2/B1 is -0.529% / -1.790%.
Wall drift is -0.368% / -1.257% for A and -0.494% / -1.722% for B.
Round2 B1 is the slowest candidate; the residual survives that variation.
Two rounds in one session establish a useful screen, not a population confidence
interval. Checkpoint capture took 12.721 ms; timed-arm restores took 3.160–3.251 ms,
all excluded from prefill timing.

**Routing and attribution.** Warm witnesses show **48 strict-router calls in
each suffix arm**. A also has 48 generic-router-shape calls in N3; its N2045
command is strict. B's N2048 command is strict. Total witnessed dispatches fall
from 9908 to 8021. These are observations of the untimed census arms; they do not
attribute milliseconds to individual kernels or prove token-level expert routes
match. The artifact's expert-down cohorts are 39 IQ4_NL and nine Q2_0, not IQ3_S
despite the artifact name. Gate/up cohorts are 20 IQ2_S, 17 IQ3_XXS, ten IQ3_S,
and one IQ4_XS.

The measured win is scheduling/rebatching on the already-promoted router baseline,
including removal of the tiny complete model pass and its command boundary.
It must not be added to old mixed-schedule gains or treated as another router
transfer. Historical strict-disabled 199/282 ms are not a current baseline.
Dividing this 266.63 ms residual by the historical GSQ whole4096 8388.22 ms gives
about **3.18% as a scale estimate only**; whole timing and denominator were not
measured here, and the historical session had a different admitted extent.

**Numerics and state.** All 43 observed full-logit vectors (248,320 elements each)
report zero nonfinite logits: one prefix checkpoint, two warm endpoints, eight
timed endpoints, and 32 continuations. All 41 comparison records have finite
metrics and matching position/QSA-length/PLE-token metadata. The table gives
the observed A→B comparison at endpoint and after each common continuation;
all four timed cross-schedule observations at each step have identical metrics.
Relative L2 is uncentered and normalized by A's logit norm; KL is in nats.

| Step | KL(A‖B) | KL(B‖A) | Relative L2 | Max absolute logit difference | Shared top1 ID |
|---|---:|---:|---:|---:|---:|
| Prefill endpoint | 0.00502609 | 0.00645067 | 0.0783402 | 1.235848 | 63280 |
| Continuation 1 | 0.0109660 | 0.0117731 | 0.0451948 | 0.986737 | 10993 |
| Continuation 2 | 0.0235677 | 0.0294748 | 0.0753667 | 1.415041 | 11 |
| Continuation 3 | 0.00976645 | 0.0147785 | 0.137688 | 2.777336 | 321 |
| Continuation 4 | 0.0429139 | 0.0950802 | 0.102496 | 2.252436 | 37715 |

Both choice-regret directions are zero at all steps. The four teacher tokens
are `[63280,10993,11,321]`, also the common preceding greedy choices in this case.
This remains a four-token, one-stream check, not broad generation/distribution
qualification. Final KL grows beyond endpoint KL, although the progression is
not monotonic. The endpoint L2/maxabs reproduce the older GSQ mixed-schedule
observation; current production routing has not removed that numerical effect.

Same-schedule A1/A2 and B1/B2 comparisons are exact in all 20 recorded step
comparisons: zero KL/L2/maxabs, equal logit bits and complete state digests.
The summarizer also verifies matching same-schedule hashes across both rounds
and between the warm endpoint and measured endpoint. Thus the observed A/B
differences repeat deterministically in this run rather than appearing as
within-schedule drift.

Cross-schedule hashes differ for **120 of 121 persistent allocations at every
step**, plus final hyper. Only index0, layer0 GDN convolution state, matches;
index1, layer0 GDN delta state, already differs. This rules out describing the
state difference as exclusively a later selected-QSA phenomenon. Allocation
hashes include inactive cache tails and say neither how many elements differ
nor by how much; no persistent-state finiteness/value scan is retained. The
shared restored storage and deterministic repeats help interpret the result,
but these hashes do not prove corruption or establish a numerical tolerance.
There is no captured per-token routing or selected-block equivalence witness.

**Admission and binding.** Native combined admission and the diagnostic CPU gate
passed without a scalar retry. Observed native weights are 55,110,090,752 bytes;
session allocations are 2,161,442,816 bytes. Selected capability and selection
are active, packed capacity2048, admitted extent4100. The checkpoint bound is
223,007,744 bytes plus 100,663,296 bytes diagnostic margin; the normal Flash
536,870,912-byte dynamic reserve is preserved. Admission reports
`AdmittedProcessBudgetOmitted`: the process-budget signal was omitted by the
normal gate, not a separately measured process-memory guarantee. Allocation
counts are not peak RSS measurements. Both retained shard stamps revalidated
unchanged at completion; no new full artifact content hash was computed.

The JSONL header retains SHA256 of compiled runtime, packet, shared helper,
session, checkpoint, QSA, MoE, GDN, dispatch, profile and metallib inputs. Key
bindings are packet `894db81b0b30513536b8d0f5306be3680536da322c7b5487e24257bd6d7d434a`,
runtime `c93855b590f6553b61c95c0768abc12726492a023c04d43819581d77c919b175`,
and executable `e463b081cbb9fdfb3f42ad4c6fd7a386874dd65d29aa79b060f7f4940e322da0`.
The recorded executable is under `/Users/tito/code/qwen-llm/target/release/deps/`;
analysis used `/Users/tito/code/qwen-llm-quant-compat`, HEAD `2a322a65`, with the
owner's uncommitted diagnostic source changes. Compiled hashes, not that HEAD
alone or a later checkout, bind this historical run.

The initial recommendation to expand measurements has now been exercised for
GSQ by the completed restart above. Its results supersede the historical 3.18%
whole-time scale estimate; they do not turn that earlier estimate into a
measurement. Early-GDN localization and quality qualification are now required
before promotion, regardless of unchanged top1.

Integration appendix: pinned Metal target at 83c91cec
--------------------------------------------------

Main is integrated at `83c91cec25cd9d006bb8d24ea9e9a0497bd3987e`; the owner
reports six planner CPU tests passed after rebuilding. New [UD](ud-merged-target-confirmation.jsonl)
and [GSQ](gsq-merged-target-confirmation.jsonl) packets each contain one whole4096
ABBA round per corpus, A=Some(false), production B=None. Normal lease/admission,
router and complete-GPU-timing checks pass. The subsequent harness-metadata
addition is separately source-bound; this is not a clean-commit binary claim.

**Every sampled trajectory matches the prior production confirmation.** Per
artifact: four warm endpoints plus eight timed arms ×(endpoint + four continued
tokens), 44 whole records, both schedules, with no observations excluded.

| Artifact | Logit hashes | Hyper hashes | Persistent tensor records (layout + hash) | Causal metadata / complete endpoint records | Teacher-token continuation records |
|---|---:|---:|---:|---:|---:|
| UD | 44/44 exact | 44/44 exact | 5,324/5,324 exact | 44/44 exact | 32/32 exact |
| GSQ | 44/44 exact | 44/44 exact | 5,324/5,324 exact | 44/44 exact | 32/32 exact |

All 88 endpoints are finite. Persistent equality includes all 121 records'
indices, dtypes, shapes, byte counts and hashes; causal equality covers position,
QSA lengths and PLE tokens. Both 4100-token arrays, complete prompt/text records,
tokenizer identity, loaded-model metadata and MoE dtype cohorts match per
artifact. Prompt/text hashes are independently recomputed. All five archived
shard stamps match (path/device/inode/size/mtime/ctime), with successful runtime
revalidation. This is stat/binding comparison, not new GGUF content hashing.

**Compiled target:** both headers record `metal3.2`, product/research deployment
targets `15.0`, from compiled constants. Bound `build.rs` passes the matching
`-std`/`-mmacosx-version-min` flags and checks the effective compiler triple.
This does **not** establish a macOS 15 host test; packets identify Apple M4 Max.

| Binding | SHA256 |
|---|---|
| New executable, 47,228,568 bytes | `78db7dab29c36b801022d577672d05a320ce1b6e38ddd1409f984149d9b5f953` |
| Prior executable, 47,118,952 bytes | `1f85ca7827e9a122bebe05d760c5a09a3561fd76675cf523c05f27a59ca974e1` |
| Embedded product metallib, both builds | `fbbb7bd18a6c6a74cf0be61060159645b134e7e2670038e7120c81ea09677023` |
| New compiled build.rs | `d1fa951c0acb15bf3f66942e5f945ab345cbc4f60f9df46030ee6c9d837c69b3` |
| New frontier_schedule.rs | `e63c0e428d9795513e776f00d89824dc8421421f2c88f8fc09529f992a753308` |
| New shared prefill_map.rs | `30d11df8d4de967451049e5e9000f61d024c11ae3ff255f2821eea37537927c1` |
| Runtime, both builds | `c70552d890673af77eb4576254124b62349cd2070298243b12506cbc64934342` |

The product metallib hash and runtime/session/checkpoint/QSA/MoE/GDN/dispatch/
profile source hashes match the prior build. Only `packet_rs` and
`shared_packet_rs` differ among pre-existing source-binding fields; target
metadata is new. All prior hashes remain preserved. These matching
trajectories and bindings support carrying the prior joint v1 NLL / v2
retrieval qualification through integration without a full quality rerun.
This is bounded evidence on retained streams, not a new quality study. GSQ's
original A/B differences and INCONCLUSIVE v1 retrieval verdict remain intact.

**New timings, not replacement estimates.** Warm arms excluded; all eight
GPU/wall pairs favor B. Means use both observations per arm.

| Artifact / corpus | Mean GPU A→B ms; saved % | Mean wall A→B ms; saved % | GPU pair savings % | Wall pair savings % |
|---|---:|---:|---:|---:|
| UD / prose | 8257.34→8107.89; 1.810% | 8282.74→8140.99; 1.711% | 0.953% / 2.683% | 0.717% / 2.721% |
| UD / ssh_repeated | 8775.66→7983.74; 9.024% | 8801.58→8019.33; 8.888% | 13.550% / 3.998% | 13.326% / 3.974% |
| GSQ / prose | 8536.43→8099.37; 5.120% | 8555.87→8121.93; 5.072% | 6.588% / 3.604% | 6.469% / 3.631% |
| GSQ / ssh_repeated | 8382.86→8113.41; 3.214% | 8402.93→8137.72; 3.156% | 3.260% / 3.169% | 3.137% / 3.175% |

UD SSH A drifts −9.943% GPU / −9.677% wall; its 9.024% mean saving is not a new
headline. UD prose B drifts −3.523% GPU, GSQ prose A −3.152%; GSQ SSH is steadier
(A/B −0.222%/−0.128%). Keep the historical UD ~2.4% result and GSQ's positive
but uncertain magnitude. No recorded trajectory regression; these samples do
not establish a compiler-induced performance change.

Reproduce the complete CPU-only comparison, including ordinary timing audits:

```sh
UV_CACHE_DIR=/tmp/flash-frontier-uv-cache uv run --no-project \
  docs/bench/2026-10-09-flash-frontier/compare_integration.py \
  docs/bench/2026-10-09-flash-frontier/ud-production-confirmation.jsonl \
  docs/bench/2026-10-09-flash-frontier/ud-merged-target-confirmation.jsonl \
  docs/bench/2026-10-09-flash-frontier/gsq-production-confirmation.jsonl \
  docs/bench/2026-10-09-flash-frontier/gsq-merged-target-confirmation.jsonl \
  > /tmp/flash-merged-target-comparison.json
```

The retained [comparison report](merged-target-comparison.json) includes every
endpoint result, full model/stat bindings, source hashes and unrounded timings;
both audits report zero problems. New raw packet bindings:

| Packet | Bytes | SHA256 |
|---|---:|---|
| ud-merged-target-confirmation.jsonl | 891,120 | `7f2acc511f0a3d3e5a8483f0476972e7b29c989e7233f9e9be9b5f11f939bed7` |
| gsq-merged-target-confirmation.jsonl | 893,827 | `06cc458c7540757f2012634b84a23cee0c2332bd8f9b89ef3c2a534525e09922` |

Raw packets are unchanged. This analysis performed no GPU work, build, quality
rerun, model-source edit or commit. Prior binary sidecar archives/manifests are
unmodified; these new JSONL-only confirmations need no duplicate binary archive.
