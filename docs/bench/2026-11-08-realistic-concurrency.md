# Warm concurrent coding-agent standing, 2026-11-08

Source: clean main **`09a3a521cc907711d556f4e59da6daa5592eb457`**, Apple M5 Ultra,
**256 GiB**. Production Rust/shaders are unchanged. Exact artifacts are the same
V4.1 GSQ-RCO IQ2_XS + IQ4_NL Engram, GLM UD-IQ3_XXS and Flash-Next GSQ-RCO IQ3_S
as the earlier [16K-root standing](2026-11-08-tier1-concurrency.md).

**Neither own-history A/B passes the requested exact-output-work gate.** One V4.1
reply ended 12 tokens early; the final GLM reply ended 390 early. The harness
refuses both same-work ratios. The table below reports observed wall time and
actual-output throughput, with that exception explicit. It is useful capacity
and latency evidence, not a qualified identical-work speedup. Flash-Next is an
EOS-censored serial baseline only. No model was retried to obtain a passing
length outcome.

## Primary observations

Three concurrent client chains are offered in every arm. MAX_ACTIVE=1 controls
owner admission; the default is four for V4.1/GLM. GLM's original serial owner
also keeps one live session, while its default retains four; its policy delta
therefore includes residency as well as native batching.

| Family | Calls/arm | Serial wall s | Default wall s | Serial / default output tokens | Actual-output aggregate tok/s | Observed throughput ratio | Exact-work gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| V4.1 | 25 | 240.730 | 211.717 | 14,622 / 14,610 | 60.74 / 69.01 | 1.136x | Failed: 12 fewer outputs |
| GLM-5.3 | 33 | 319.129 | 220.578 | 12,410 / 12,020 | 38.89 / 54.49 | 1.401x | Failed: 390 fewer outputs |
| Flash-Next | 14 | 203.601 | — | 12,041 / — | 59.14 / — | — | Serial only; two EOS stops |

V4.1's wall difference is **29.0 s**, with **0.082%** fewer output tokens; GLM's
is **98.6 s**, with **3.143%** fewer. These are observed differences, not matched
work savings. A rough conversion of the missing tokens at the serial decode
pace is 0.17 / 7.1 s, but that is not a counterfactual measurement. The exact
failures are V4.1 **a1-t8: 218 requested / 206 emitted**, and GLM
**a2-t7: 3822 / 3432**. Flash-Next emits **12041 / 14448 planned**:
**a1-t0: 276 / 81**, **a0-t2: 2247 / 35**. Other primary calls reach their limits.
The API's `max_output_tokens` is a ceiling; EOS still stops generation.

The older 16K-cold-root standing and its subsequent fair-prefill qualifications
are different workloads. Their 1.12x V4.1 / 1.225x GLM ratios must not be treated
as directly comparable matched-work improvements here. This warm study exposes
long-reply head-of-line waits and state-pool effects that those roots obscure.

## Per-agent latency, decode and queueing

TTFT and native decode rate include service interference. `Native tok/s` uses
sum(generated)/sum(native decode wall), rather than averaging short-call rates.
Native decode wall includes interleaved work on other requests. Header wait is
submission to response headers, a dispatch-wait proxy; it is not all owner/GPU
queueing. Native owner-only `batch.queue_ms` is separately retained in the CSV.

| Family / arm | Agent | Calls | TTFT p50 / p95 s | Native tok/s | Header wait sum s |
| --- | ---: | ---: | ---: | ---: | ---: |
| V4.1 serial | 0 | 4 | 5.991 / 22.151 | 72.59 | 32.465 |
| V4.1 serial | 1 | 15 | 1.896 / 144.523 | 71.57 | 190.208 |
| V4.1 serial | 2 | 6 | 6.710 / 128.384 | 73.18 | 166.339 |
| V4.1 default | 0 | 4 | 0.852 / 3.251 | 51.31 | 0.035 |
| V4.1 default | 1 | 15 | 0.996 / 2.190 | 27.41 | 0.171 |
| V4.1 default | 2 | 6 | 0.877 / 13.436 | 27.44 | 0.040 |
| GLM-5.3 serial | 0 | 12 | 14.572 / 82.261 | 55.37 | 224.118 |
| GLM-5.3 serial | 1 | 13 | 11.942 / 81.231 | 55.23 | 223.721 |
| GLM-5.3 serial | 2 | 8 | 10.466 / 50.950 | 55.22 | 121.008 |
| GLM-5.3 default | 0 | 12 | 0.544 / 6.596 | 21.68 | 0.181 |
| GLM-5.3 default | 1 | 13 | 0.758 / 8.448 | 23.49 | 0.179 |
| GLM-5.3 default | 2 | 8 | 0.599 / 1.688 | 31.02 | 0.079 |
| Flash-Next serial | 0 | 5 | 18.404 / 54.897 | 68.10 | 109.209 |
| Flash-Next serial | 1 | 6 | 9.740 / 47.491 | 69.19 | 125.831 |
| Flash-Next serial | 2 | 3 | 12.229 / 16.505 | 69.41 | 37.213 |

The short V4.1 chain's p95 TTFT falls **144.5 -> 2.19 s**; GLM's first two agents
fall **82.3 / 81.2 -> 6.60 / 8.45 s**. Large file reads remain in those tails:
V4.1's ~20.7K-row append needs **13.44 s** to first token. Its live exact CED
route engages in both arms (`shrinking_ced+full`, one episode/five yields).
There is no transport parking in the batched primary arms.

## Where cohort wall goes

**Time widths below count simultaneously open post-first-token streams**, not
native GPU batch width. They include interleaved appends/stalls. These exclusive
buckets sum to cohort wall; predecode-only includes restore, append, dispatch and
queue time when no stream is decoding. Do not add per-request prefill/decode
walls: shared work overlaps. Production stats expose exact native width counts,
but not exclusive per-step service time by width; no per-kernel attribution is
claimed. The [20-second trace](2026-11-08-realistic-concurrency-trace.csv) retains
width overlap, content-event intervals, thermal samples and host availability.

| Family / arm | Stream decode width 1 / 2 / 3 s | Predecode-only s | Idle s | Native member transitions at B=1 / 2 / 3 |
| --- | ---: | ---: | ---: | ---: |
| V4.1 serial | 201.65 / 0.00 / 0.00 | 36.77 | 2.31 | 14597 / 0 / 0 |
| V4.1 default | 115.10 / 48.55 / 47.45 | 0.58 | 0.04 | 8149 / 1390 / 5046 |
| GLM-5.3 serial | 224.71 / 0.00 / 0.00 | 94.41 | 0.01 | 12377 / 0 / 0 |
| GLM-5.3 default | 74.95 / 44.55 / 96.56 | 4.43 | 0.09 | 3765 / 698 / 7524 |
| Flash-Next serial | 174.38 / 0.00 / 0.00 | 29.20 | 0.03 | 12027 / 0 / 0 |

V4.1 has **55.9%** of native member transitions at B=1, largely its long tail;
GLM has **31.4%** at B=1 and **62.8%** at B=3. Default native forward counts are
**10526 V4.1 / 6622 GLM**, after dividing shared member counts by their width.
These are counts, not time shares. Serial cohorts spend **83.8% / 70.4% / 85.6%**
of wall inside the observed decode span for V4.1 / GLM / Flash-Next.

| Family / arm | Content-gap p95 / max ms | Minimum conservative host availability GiB |
| --- | ---: | ---: |
| V4.1 serial | 13.9 / 676.7 | 22.76 |
| V4.1 default | 28.2 / 3025.5 | 21.44 |
| GLM-5.3 serial | 18.3 / 36.6 | 57.52 |
| GLM-5.3 default | 38.5 / 2396.1 | 51.87 |
| Flash-Next serial | 15.0 / 917.1 | 128.86 |

All one-second thermal samples are **0**, including warm-up and diagnostics.
Solo stream medians stay roughly **13.6–13.7 ms V4.1, 18.1–18.3 ms GLM,
14.3–14.4 ms Flash-Next** across the sustained traces. There is no observed
progressive median-rate collapse. Default early medians are about 28 / 38 ms
while streams share decode; late solo medians return to the serial scale.
Multi-second gaps align with interleaved appends. Pressure=0 alone is not a GPU
clock measurement. Host availability uses the engine's conservative free plus
reclaimable-inactive-minus-anonymous estimate, not a spare Metal entitlement;
sampled RSS is not total model/Metal memory.

## Reuse and next levers

| Family / arm | Processed prompt rows | Live / snapshot hits |
| --- | ---: | ---: |
| V4.1 serial | 39,846 | 25 / 0 |
| V4.1 default | 39,872 | 25 / 0 |
| GLM-5.3 serial | 68,391 | 0 / 33 |
| GLM-5.3 default | 27,192 | 31 / 2 |
| Flash-Next serial | 39,065 | 0 / 14 |

V4.1's default owns all 25 live extensions. GLM owns 31/33; its first two timed
turns restore the 47343-row cold chunk checkpoint and process 1016/999 rows.
Later turns stay live. The serial GLM control restores on all 33 turns and
re-reads **41199 more rows** than default; this residency effect is part of the
requested MAX_ACTIVE=1 comparison. Spaced optional checkpoints are not a rolling
tip: the [4096-row spacing gate](../../crates/qwen-cli/src/serve/checkpoint_ladder.rs#L152)
helps explain short-turn capture ineligibility. With three retained live
contexts, more ordinary tip copies are not the first default-path priority.

Planning estimates below are **not measured optimizations**. Native stage
attribution is still needed before selecting a specific kernel.

1. **Flash-Next: independent decode batching plus idle conversation residency.**
   It serializes 174.4 s of decode in this 203.6 s baseline. If native multi-row
   decode amortizes weights similarly to the qualified GLM route, a practical
   target is **30–45 s/cohort (~15–22%)**, below the fully shared ideal. Rough
   cost **3–5 days**, including per-sequence QSA/GDN ownership, admission,
   cancellation, restore and likelihood checks; that lane is already active.
2. **V4.1: B=1 native decode, alongside the routed-expert tile lane.** The long
   tail leaves ~115 s with one open decoder. A 10% service-cost reduction there
   has an approximately **10–12 s/cohort** ceiling; gains in shared rounds could
   add a few seconds. Rough cost **one day of native stage census, then 2–4 days**
   of targeted kernel work and qualification. More cold-root prefill work has
   little value in the timed portion of this protocol.
3. **GLM: B=2/3 native decode efficiency.** The 31 live extensions already avoid
   most checkpoint replay. A 10% native decode-service reduction offers roughly
   **10–20 s/cohort** as a planning ceiling, not all open-stream wall. Rough cost
   **one day of attribution, then 2–4 days** on the confirmed dense/routed path,
   retaining its qualified arithmetic/likelihood and exact-restore contracts.

For strict future qualification, a **bench-only minimum-output/EOS suppression
control** is needed if sampled lengths must be guaranteed. Current API ceilings
cannot guarantee that. Approximate size: **0.5–1 day plus stop/cancellation and
sampling tests**; no product speedup is claimed. The current comparator instead
fails when actual output work differs. Client/OpenCode changes are not required
for the server-side performance levers above.

## Protocol, diagnostics and reproduction

The [reusable stdlib harness](../../scripts/bench/serve_realistic.py) samples
paired (new content budget, reply budget, tool-union gap) observations from the
[real OpenCode calls](2026-11-08-real-opencode-agent-calls.csv), seed **20261108**,
with replacement. Eligible primary live-prefix calls: **54 / 31 / 25**. Cold
roots, title helpers, effort-change cold roots and snapshot fallback replays are
excluded. Draws go to the agent with least estimated source-call work until
about 240 serial seconds are planned. Call counts are **4/15/6 V4.1,
12/13/8 GLM, 5/6/3 Flash-Next**. The complete plans and exact per-call source IDs
are retained. Real long tails and large file reads remain.

Three distinct **48000-token corpus slices** prime each fresh server sequentially
with one generated token per root. Warm roots include template overhead:
~48250–48390 tokens. The source corpus is the November 7 frozen Rust file,
SHA-256 **`cf2059d77a3b3e70d8116b3e445c94e00aa515e5d792e94e6966def38e142969`**.
Templates/tool framing add to the sampled content counts; native processed and
reused rows are separately reported. Tool execution is a fixed sleep of the
source call's tool-union duration. No actual tool or OpenCode instance runs.
This models API workload geometry, not coding quality or client/SQLite overhead.
V4.1 uses released default effort; GLM/Flash-Next use low effort. All are greedy,
seeded, drafting off, maximum context **98304**. One-second thermal/memory
monitoring covers each server lifetime; a nonzero pressure sample aborts.

| Family / arm | Startup s | Untimed warm-up s | Timed wall s |
| --- | ---: | ---: | ---: |
| V4.1 serial | 1.780 | 81.930 | 240.730 |
| V4.1 default | 16.456 | 75.570 | 211.717 |
| GLM-5.3 serial | 5.801 | 140.550 | 319.129 |
| GLM-5.3 default | 4.919 | 140.418 | 220.578 |
| Flash-Next serial | 7.432 | 80.479 | 203.601 |

Arms take build lock/nice 10 and the canonical Metal lease with wait enabled.
A 25 s cooling gap and three nominal readings five seconds apart precede each
fresh server, bound only to **127.0.0.1:18737**. Other lanes/outside jobs own the
lease between arms. The GLM serial arm runs 19 s beyond the approximate
five-minute target because of checkpoint replay. About **46 minutes of owned
lease time** includes warm-up, cooling and two diagnostic default arms; shared
slot waits are separate. Build once from source: **138.1 s**, offline release,
`qwen` + `qwen-tok` in an independent APFS-cloned target directory.

An initial **canonical frozen-history diagnostic** replayed the complete serial
request bodies. Batched rounding changed replies, so the next canonical request
edited the candidate's live history. It added re-reading and does not represent
a client replaying its own output. Those runs (`v41-default`, `glm-default`) are
excluded from the primary standing, retained as `scope=diagnostic-frozen` in the
CSV. Their exact-body/output matched throughput ratios are **1.112x / 1.184x**,
with **43010 / 61781 processed rows**, versus own-history **39872 / 27192**.
Only default arms were repeated to correct this; serial controls already used
their own histories. The final harness defaults to own history and rejects
mislabeling a frozen candidate as an own-history qualification.

```sh
python3 scripts/bench/serve_realistic.py plan --family v41 --out plan.json
# Under the build lock, nice 10 and with-metal-lease, wait enabled:
python3 scripts/bench/serve_realistic.py run --arm serial \
  --binary target/release/qwen --tokenizer target/release/qwen-tok \
  --model MODEL.gguf --corpus CORPUS.txt --plan plan.json --out serial
# Fresh server, same corpus roots/new input and sampled pacing, own replies:
python3 scripts/bench/serve_realistic.py run --arm default \
  --binary target/release/qwen --model MODEL.gguf --plan plan.json \
  --reference serial --out default
python3 scripts/bench/serve_realistic.py compare serial default
```

Replace the family with `glm`; use serial only for `fn` on the pinned source.
`run --history frozen` and `compare --history frozen` are explicit diagnostics.
The primary comparator requires identical plans/tool gaps, warm-root bytes,
fresh tool-call/result bytes and generation settings, and actual output counts.
It reports historical prompt-count deltas from independently generated replies.
For this packet, own-history comparisons **exit 1** with the exact EOS failures
above; the reporting script catches and publishes those failures, not a passing
ratio. The [calls CSV](2026-11-08-realistic-concurrency-calls.csv) retains all
per-call budgets, usage, times, native width counts, reuse and request hashes.
`analyze` writes per-agent rates and 20-second stream/thermal traces offline.

Raw evidence on both machines:
**`$HOME/qwen-runs/2026-11-08-realistic-concurrency/`**. It includes every request,
response and SSE event; native logs; plans; one-second thermals/memory; actual
runtime harness versions; final analysis source; comparisons and conservation
checks. Production binary SHA-256:
**`5964efa203269067497e1e9d8b88d988d3cb420f9298ae0ef652801903a1d387`**.
All arms use that one binary. There is no confidence interval or universal fleet
speedup claim from one finite seeded cohort.

## Validation and cleanup

Seven CPU harness checks pass: paired seeded sampling/exclusions; bounded-window
wall conservation; native member-width census; actual output-count equality
(including matched EOS outcomes); replayable partial reasoning; own-history
comparison preserving fresh work; changed tool gaps rejected. Full contracts
and final documentation/path/CSV checks are recorded with the final commit.

All owned servers exit through their finally blocks (SIGINT, code 130), and all
owned clients/lease wrappers finish. The M5 process audit finds no owned process;
the clean source worktree and transfer bundle are removed. The shared
ControlMaster and other jobs remain untouched. No OpenCode process, config,
database or port 3456 is accessed. No merge or push.
