# Tier-1 serve agent standing, 2026-11-08

Clean main **`3a36b6e8396bc410139dcbf0d810c3b026804651`**, M5 Ultra / 256 GiB,
macOS 27.0.1. Unmodified source build in an owned M5 worktree. This reruns the
[November 7 standing](2026-11-07-tier1-serve-agents.md), whose source was
`dabb97021e65f192a15057b3f226281cbe1ad81f`. It measures the cumulative landed
changes, not individual feature ablations. No engine or client application changes.

## Protocol and limits

The original frozen Rust corpus and per-model tokenizer chunks are unchanged.
One serial `/v1/responses` session/model: four ~16K tool-call/result buildup
turns, three ~1.1–1.9K appends at ~64–70K, then truncate to the third build
request (~48K) and edit its last question. Actual new reasoning/message items
are replayed faithfully. GLM/Flash-Next also repeat the original return-from-fork
control that omits historical reasoning (`Return*`). This combines two changes;
it does not isolate reasoning omission or identify the installed OpenCode client.

Same production artifacts: four-shard DeepSeek-V4.1-Flash-1017-GSQ-RCO IQ2_XS
with its IQ4_NL Engram companion; four-shard GLM-5.3-Flash UD-IQ3_XXS; two-shard
Qwen3.8-Flash-Next GSQ-RCO IQ3_S. Same context ceiling **98304**, no drafter,
greedy/seed 1, default serve precision/cache/ladder/residency controls. V4.1
uses default chat; GLM/Flash-Next use low reasoning. Buildup replies request
64 tokens; all other replies request 200. All reach those limits.

TTFT ends at the first nonempty primary text/reasoning/tool-argument delta;
heartbeats and duplicate aliases do not count. Decode is the server's generated
count/decode-wall statistic. These are single observations per turn, not paired
switch experiments or confidence intervals. Group means average different turn
sizes. The cold cell is **16K B1**: B4 is a live extension, not a cold 64K probe.
The separate cold-64K **59.1 → 36.9 s** CED evidence remains in PERF-LOG.

Every build/campaign holds the shared build lock at nice 10. Each model has a
separate canonical Metal lease with wait enabled, including its native controls;
one engine is resident at a time. Each API turn cools 25 seconds plus three zero
readings five seconds apart. One-second monitoring stops a campaign on nonzero
thermal pressure. Every before/during/after sample is **0**. Leases are released
between models, with an additional cooling gap so the routed-tile lane can alternate.
No model/kernel instrumentation was added; the original V4.1 unsplit native timing
flag is retained. All six native build/runtime identities match clean `3a36b6e8`.

## Before → after, seconds

| Model | Cold B1 | Live B2–B4 mean | Deep A1–A3 mean | Older-question fork | Return* |
| --- | ---: | ---: | ---: | ---: | ---: |
| V4.1 | 14.666 → **11.172** | 15.169 → **14.755** | 2.967 → **1.728** | 15.116 → **3.853** | — |
| GLM | 17.240 → **16.105** | 17.337 → **16.313** | 2.887 → **1.873** | 34.339 → **2.159** | 55.953 → **54.916** |
| FN | 9.526 → **9.543** | 9.442 → **9.499** | 1.640 → **1.708** | 9.037 → **1.907** | 28.856 → **29.159** |

Deep-append TTFT change: **V4.1 41.8% lower**, **GLM 35.1% lower**, **FN 4.1% higher**. Fork reductions: **V4.1 74.5%**, **GLM 93.7%**, **FN 78.9%**.

## Per-turn measurements

Processed/reused columns are current-main counts; prompt length is their sum.
`Reuse` is the measured backend origin (`cold`, `live`, or RAM `snapshot`).
TTFT and decode arrows compare the original standing with this run.

### V4.1 GSQ-RCO IQ2_XS + IQ4_NL Engram

| Turn | Processed | Reused | TTFT s, old → new | Decode tok/s, old → new | Reuse |
| --- | ---: | ---: | ---: | ---: | --- |
| B1 | 16,460 | 0 | 14.666 → **11.172** | 75.54 → **75.11** | cold |
| B2 | 16,095 | 16,523 | 14.368 → **13.957** | 75.22 → **75.29** | live |
| B3 | 16,094 | 32,681 | 15.202 → **14.873** | 74.60 → **74.68** | live |
| B4 | 16,094 | 48,838 | 15.936 → **15.437** | 73.44 → **74.15** | live |
| A1 | 1,118 | 64,995 | 2.395 → **1.310** | 73.57 → **73.30** | live |
| A2 | 1,629 | 66,312 | 3.234 → **1.676** | 73.44 → **73.08** | live |
| A3 | 1,886 | 68,140 | 3.273 → **2.199** | 73.10 → **65.49** | live |
| Fork | 3,805 | 44,969 | 15.116 → **3.853** | 74.05 → **73.92** | snapshot |

### GLM-5.3-Flash UD-IQ3_XXS

| Turn | Processed | Reused | TTFT s, old → new | Decode tok/s, old → new | Reuse |
| --- | ---: | ---: | ---: | ---: | --- |
| B1 | 16,322 | 0 | 17.240 → **16.105** | 57.13 → **57.11** | cold |
| B2 | 16,060 | 16,385 | 16.895 → **15.860** | 56.79 → **56.83** | live |
| B3 | 16,060 | 32,508 | 17.344 → **16.321** | 56.16 → **56.18** | live |
| B4 | 16,060 | 48,631 | 17.771 → **16.758** | 55.65 → **55.99** | live |
| A1 | 1,084 | 64,754 | 2.194 → **1.419** | 55.20 → **55.17** | live |
| A2 | 1,596 | 66,037 | 3.018 → **1.961** | 55.14 → **55.17** | live |
| A3 | 1,852 | 67,832 | 3.449 → **2.240** | 55.05 → **55.10** | live |
| Fork | 1,723 | 46,844 | 34.339 → **2.159** | 55.66 → **55.67** | snapshot |
| Return* | 54,922 | 14,573 | 55.953 → **54.916** | 54.88 → **55.02** | snapshot |

### Flash-Next GSQ-RCO IQ3_S

| Turn | Processed | Reused | TTFT s, old → new | Decode tok/s, old → new | Reuse |
| --- | ---: | ---: | ---: | ---: | --- |
| B1 | 16,493 | 0 | 9.526 → **9.543** | 69.08 → **69.28** | cold |
| B2 | 16,093 | 16,556 | 9.982 → **10.024** | 68.64 → **68.34** | live |
| B3 | 16,094 | 32,712 | 9.121 → **9.165** | 67.62 → **67.39** | live |
| B4 | 16,092 | 48,869 | 9.223 → **9.308** | 66.79 → **66.20** | live |
| A1 | 1,117 | 65,024 | 1.353 → **1.418** | 66.18 → **65.50** | live |
| A2 | 1,630 | 66,340 | 1.699 → **1.753** | 66.05 → **65.62** | live |
| A3 | 1,886 | 68,169 | 1.868 → **1.952** | 65.93 → **65.30** | live |
| Fork | 1,757 | 47,048 | 9.037 → **1.907** | 67.07 → **66.81** | snapshot |
| Return* | 53,006 | 16,387 | 28.856 → **29.159** | 65.98 → **65.05** | snapshot |

All V4.1 requests have the same structured contents as the original session.
GLM replies change under the newly qualified numerical routes, so later requests
replay those new replies; identical token counts do not imply identical text.
The source chunks, instructions, request construction and output limits stay fixed.
All nine Flash-Next requests match the originals as well. V4.1 emits visible text
at TTFT. GLM's first visible answer on A1–A3 follows its first reasoning delta
by **0.347–0.550 s**. Flash-Next's three appends emit only reasoning within
200 tokens, so their TTFT is not first-visible-answer latency.
The GLM `Return*` prompt is one token shorter than before; its 14,573-token reuse
ceiling is unchanged. Flash-Next `Return*` has identical input but restores
**16387** rather than **16488**, adding 101 replay rows after older-tip retention.
Its much larger history mismatch remains. Fork processing falls from **16158/33994/16093** to the
current counts above (V4.1/GLM/Flash-Next).

V4.1 A3 is retained as observed: **65.49 tok/s**, versus 73.08 in A2 and 73.92
in the following fork, all pressure zero. Its native driver spans rise to
**359 ms**, versus **35/46 ms** in A1/A2. This is not evidence of a persistent
decode regression; no outlier was discarded.

## Same-build native bridge

Fresh native A2 controls match its processed rows/reused depth for pp and its
complete prompt depth for tg200. One retained sample/cell, `--no-warmup --runs 1
--seed 1`; prefix fill/loading/allocation are untimed. Native synthetic IDs differ
from the served code/replies, and idle/input regimes differ. The excess below
is descriptive, not an isolated transport overhead. The old extra GLM 2048-row
shape probe is not repeated: that route-gate diagnosis has already been resolved.

| Model | PP rows / depth | Native PP s | Serve A2 s | Excess s, old → new | Native TG200 / serve tok/s | TG depth |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V4.1 | 1629 / 66312 | 1.597 | 1.676 | 1.712 → **0.078** | 71.98 / 73.08 | 67941 |
| GLM | 1596 / 66037 | 1.918 | 1.961 | 0.091 → **0.043** | 56.55 / 55.17 | 67633 |
| FN | 1630 / 66340 | 1.249 | 1.753 | 0.447 → **0.504** | 68.64 / 65.62 | 67970 |

The original 32K PERF-LOG tg64 references (V4.1 74.81–74.94, GLM 57.65,
Flash-Next 71.10 tok/s) have shorter context and output lengths. They are not
serve overhead baselines. Current GLM native pp **1.918 s** agrees with the
landed arbitrary-row 1596/66037 receipt (**1.912 s**). V4.1's serve A2 is in the
range of the landed folded-header/residency receipt (**1.750 s**, different
1626/64995 shape). The native/serve gaps no longer support another second of
V4.1 framework-only append savings.

## What actually engaged

- **V4.1:** B1 logs one shrinking-CED episode with three encoder yields;
  B2–B4 and every append/fork remain Full. All appends use `live_session`,
  **one native call**, and `prefill_header_fused=true`. No loaded-prefill calls,
  width max/mean **1**, 199 serial decode transitions: batching is not exercised
  by this single-client protocol. A1's **47.5 ms** live-state growth is included;
  A2/A3 reuse costs are ~0.02 ms. A1/A2 GPU **1111/1502 ms** and valid driver
  **35/46 ms** compare with old GPU **1244/1639 ms**, driver **993/1435 ms**.
  These overlapping spans are not additive wall budgets. The fork restores
  **44969** instead of **32616**, retaining a **3805-row** suffix and a separate
  two-row header. See [cold-only eligibility](../../crates/qwen-cli/src/serve/backend_v41/execution.rs#L289),
  [header policy](../../crates/qwen-cli/src/serve/backend_v41.rs#L56),
  [phase/call counters](../../crates/qwen-cli/src/serve/backend_v41/execution.rs#L549)
  and [Ready CED/Full reuse](../../crates/qwen-cli/src/serve/backend_v41/reuse.rs#L127).
- **GLM:** all normal extensions log live Fast/Half reuse. Actual-trajectory
  captures **30721/46844/62967** are present; the fork restores **46844** in
  `fast_trajectory_checkpoint_v1`, not the old first-turn **14573** fallback.
  Real **1084/1596/1852-row** append commands meet widened IQ/KDA/Q6 predicates.
  Sparse tensor QK/PV still requires 512/1024/2048 command rows, so these appends
  retain split attention. IQ/KDA/Q6 dispatch is a source eligibility inference,
  not a measured per-kernel counter. No extra prefill checkpoint copy is logged
  on the three appends; A2 pays **109.6 ms after decode** to advance the ladder
  tip. See [actual command rows](../../crates/qwen-llm/src/glm5_next_metal/packed.rs#L1852),
  [IQ row gate](../../crates/qwen-llm/src/glm5_next_metal/tensor_gate_up.rs#L51),
  [attention gate](../../crates/qwen-llm/src/glm5_next_metal/prefill_attention.rs#L15),
  [trajectory capture](../../crates/qwen-cli/src/serve/backend_glm5_next.rs#L1223)
  and [post-decode capture](../../crates/qwen-cli/src/serve/backend_glm5_next.rs#L1401).

- **Flash-Next:** all appends log `live_session`, **0 ms restore**, **0 ms
  completed copy**, endpoint checkpoints with logits, and **two native calls**.
  Prompt-state capture is **178.3–208.6 ms**. The fork restores
  **47048** instead of **32712** and processes **1757**;
  its reuse/restore phase is **225.1 ms**, including the **166.7 ms** deferred
  departing-state copy. Those spans overlap. The small packed header
  is still separate. See [live selection](../../crates/qwen-cli/src/serve/backend_qwen4exp.rs#L477),
  [departure copy](../../crates/qwen-cli/src/serve/backend_qwen4exp.rs#L506),
  [body/header calls](../../crates/qwen-cli/src/serve/backend_qwen4exp.rs#L575)
  and [prompt endpoint capture](../../crates/qwen-cli/src/serve/backend_qwen4exp.rs#L627).

Flash-Next's quiet-repeat append mean is **1.708 s**, versus **1.640 s** in
the original standing (**4.1% higher**). A2's phase difference accounts for it:
prefill **1535.1 → 1545.9 ms**, restore **37.6 → 0 ms**, prompt capture
**98.7 → 178.3 ms**, tokenize **23.1 → 24.3 ms**. Their net **+54.0 ms**
tracks client TTFT **+53.9 ms**. Live reuse is engaged; the observed slowdown
comes from a larger copy span. Copy timing already varied widely in the
checkpoint qualification. This cross-session standing does not replace the
separate same-binary 6–11% feature off/on receipt or establish that the feature
causes the copy increase. Profile the copy/allocation path in the actual
multi-turn workload before another rollout claim.

Per-kernel route counters are not exposed by ordinary serve for these backends.
Only the logged scheduler/prefill/reuse/capture observations above are measured
engagement; tensor-kernel eligibility is a source inference. The serial protocol
cannot assess multi-agent batching or drafter performance.

## Next gaps, ranked

These are engineering targets, **not measured patch gains**, and are not additive.
The cold CED, basic ladder, GLM IQ/KDA/Q6 widening and Flash-Next live turns are
already deployed. The native bridge argues against more large generic serve
append-overhead work on V4.1/GLM.

| Rank | Work | Expected wall value in this workload | Rough cost / constraint |
| ---: | --- | --- | --- |
| 1 | Verify the deployed client's historical reasoning replay; fix it only if absent | Roughly **50 s GLM / 25 s Flash-Next** on an affected deep return; **zero** if already patched | **Half-day audit, 1–2 days fix/tests**. `Return*` is not a clean omission-only A/B. Existing cheap faithful-return receipts and the early prefix break support the target; capture the real client's request before attributing this to OpenCode. Never alias changed transcripts to old snapshots. |
| 2 | Flash-Next prompt-state transfer, then its tiny-header kernel | Recover **0.08–0.11 s/append** from this copy-cost increase; broader target **0.1–0.25 s/append** | **1–3 days** to profile/stabilize/reuse copy storage; **3–7 days** with a header kernel. Preserve body/header arithmetic. Earlier combined/scalar header substitutions failed the quality gate; optimize that kernel/copy instead of simply fusing calls. This session pays 0.18–0.21 s before first token for the prompt copy. |
| 3 | V4.1 CED for large **live** suffixes | Target **2–4 s per 16K buildup turn**; no projected benefit for the 1–2K appends here | **1–2 weeks** of native episode/ready-state, admission, cancellation and exact restore qualification. Serve currently enables it only at zero/shared-prelude origin; cold CED equality does not itself qualify a warm-origin extension. |
| 4 | Edit-local checkpoint coverage for branch-heavy sessions | Target **1–3 s V4.1**, **0.5–1.5 s GLM/Flash-Next per fork** | **3–7 days plus memory/capture budget**. Current committed widths leave 3.8K/1.7K replay. More retained budget alone cannot invent a missing committed state; new cuts need numerical qualification and must repay ordinary-turn copy/call cost. |
| 5 | GLM arbitrary-row sparse attention with better operand precision | Target **0.1–0.3 s per 1–2K append** | **About a week** for an F32-aware route and signed NLL/KL/F64 checks. The present Half-query/probability tensor widening was explicitly rejected; changing its row predicate is insufficient. |
| 6 | V4.1 routed expert tiles, already in flight | A **15–30%** cut in the prior ~0.67 s routed-expert census projects **0.10–0.20 s per ~2K append** | Existing routed-tile lane; roughly **3–7 days** including qualification. Do not duplicate it. This is an engine gain and should reach serve now that large scheduling excess is gone. |

The remaining 200-token decode takes approximately **2.7 s V4.1 / 3.6 s GLM /
3.1 s Flash-Next**, generally longer than warm prefill. A qualified
10% native decode improvement would save ~0.25–0.33 s/turn. Different served and
synthetic tokens prevent treating the native/serve decode difference as a
transport bug. Existing native GLM/Flash-Next drafting serve paths cold-prefill
and skip cache reuse, so enabling a drafter is not a free continuation win
([serve contract](../SERVE.md)).

## Evidence and checks

Raw material on metagross and the local mirror:
`$HOME/qwen-runs/2026-11-08-serve-agents/` — `session-initial.py`, `session.py`,
`bench-initial.py`, `bench.py`, frozen source/chunks, all request/event/response
bodies, phase/native timing logs, one-second thermal traces, build/toolchain
identities, `comparisons.json`, `validation.json`, and `manifest.sha256`.
The initial Flash-Next attempt is retained in `fn-contended/`; an unrelated
~500% CPU AOT worker overlapped its deep appends. That attempt was stopped and
is **excluded from every table**, not pooled with the full quiet repeat.
The repeat also monitors that worker's absence throughout server/native runs.
No nonzero thermal sample occurred in either attempt.

Docs-only validation: `git diff --check`, documentation path normalizer and
local-link/data consistency checks. No code tests are rerun for these two
Markdown changes. All owned server/native children are stopped/reaped; owned
M5 source/target scratch and transfer bundle are removed. The local docs
worktree is removed after committing, preserving the branch; no merge.
