# GLM three-agent served TTFT: existing-record decomposition

**Analysis only; no GPU run.** The primary evidence is the six counted arms
from `metagross:~/qwen-runs/2026-11-12-glm-prefill/kda-parallel-prep/serve-180/`:
three reference/candidate pairs, release binary, same seed and 180 s plan, real
agent sampling, 48K root warm, three agents. The spare `reference-1` arm is
excluded. `reference-2` and `candidate-1` each have one right-censored request;
those requests have no TTFT and are absent from the class percentiles below.
The run record says nice 0, compiler overlap in all six arms, thermal pressure
peaking at 2, and configured `max_active=4` with 3 sample workers.

The figures use nearest-rank p50/p95 over all six counted arms. “New rows” is
the request's `prompt - reused` token count. The root class uses the three
48K warm-up requests per arm, before the measured window; one had no prefix
reuse and two reused 239 tokens. It is not a cold-process measurement.

## Per-class measurements

| Call class | n | Client TTFT p50 / p95 | Server prefill p50 / p95 | `batch.queue_ms` p50 / p95 | TTFT minus prefill p50 / p95 |
|---|---:|---:|---:|---:|---:|
| 48K roots, warm-up | 18 | 45.84 / 49.39 s | 45.72 / 49.33 s | not retained per root | not paired |
| Short, <512 new rows | 134 | 0.724 / 1.456 s | 0.384 / 0.725 s | 0.0019 / 0.0055 ms | 0.371 / 1.065 s |
| About 1K, 1,239–1,509 rows | 20 | 1.931 / 2.254 s | 1.387 / 1.645 s | 0.0020 / 0.0047 ms | 0.540 / 0.740 s |
| Long, 3,984–7,454 rows | 10 | 5.295 / 9.594 s | 4.092 / 7.783 s | 0.0018 / 0.0067 ms | 1.252 / 1.950 s |

The last column is a residual, not a measured first-token or transport stage.
It includes time between prefill slices, first-token selection/callback,
unmeasured HTTP admission, and delivery. For the root calls, client TTFT and
prefill percentiles are from separate records and cannot be subtracted per
request.

`layer_gpu_ms` is p50/p95 `0 / 0.569 s` for the short group, but only 28 of 134
calls have a nonzero layer observer; zero means that observer was not armed,
not that the GPU did no work. It is `1.348 / 1.600 s` for the about-1K calls
and `3.980 / 7.654 s` for the long calls.

The overall per-agent p95s in the campaign range from about 2.0 to 9.6 s. The
long 4K–7.5K append class explains the upper tail; it is easy to miss if the
corpus is summarized as only “short” and “1K” appends. The separate one-agent
GLM NextN baseline has append TTFT p50 about 0.595–0.598 s and p95 1.583–1.589
s across three windows, but it uses a different plan and is context only
(`docs/bench/2026-11-12-glm-nextn/`). Older Maya reuse screens are also single
client and use different prompt sizes.

## What was active during TTFT

The server reported the same policy in the six arms: fair prefill, 2,048
loaded rows, 10% decode budget for unsliced prefills, 200 ms target layer
slices with a 500 ms minimum predicted append, and four active slots. The
three-agent traffic reached a mean decode width near 3.

| Call class | Sent while another decode response was active | TTFT interval overlapped another prefill | Layer slices p50 / p95 |
|---|---:|---:|---:|
| Short, <512 rows | 125 / 134 | 19 / 134 | 0 / 4 |
| About 1K | 12 / 20 | 2 / 20 | 8 / 16 |
| Long, 4K–7.5K | 10 / 10 | 1 / 10 | 32 / 56 |

For short calls, those whose TTFT overlapped another agent's prefill had
TTFT p50/p95 `0.974 / 2.479 s`, versus `0.704 / 1.251 s` with no such overlap.
This is evidence of correlation, not exact causal attribution: requests enter
at variable points in other agents' prefill and decode schedules.

The long requests were sliced into 32–56 pieces. In `batching/driver.rs`,
each unfinished layer slice with active decode members is followed by one
decode batch round; the 10% decode budget is not used on that sliced branch.
The per-arm summary estimates 31.8–32.7 ms per decode round. That predicts
roughly 1.0–1.8 s of interleaved
decode service across 31–55 rounds, close to the observed 1.25–1.95 s residual.
For the about-1K calls, 7–15 rounds predict 0.22–0.49 s, also material against
the observed 0.54–0.74 s residual. These are estimates from the schedule
counters, not direct per-request GPU wait timers.

`batch.queue_ms` is measured from backend submission until a free active slot
starts the request; its p95 is under 0.007 ms in every append class. It does
not include HTTP parsing/rendering before submission. The harness field
`queue_exposure_s` is not queue time: `serve_realistic.py` sets it to
`first - sent` for delivered calls, so it repeats client TTFT.

## Variance sources and logging gaps

- Request sizes range from 27 to 7,454 new rows; output sizes and tool gaps
  also vary substantially. The 4K–7.5K calls account for the largest TTFTs.
- Work overlaps: 93% of short calls were sent while another response was
  streaming, and every long append began while another response was active.
  The long prefill path itself is split by layer slices, with a decode round
  after each unfinished slice.
- All six arms reported compiler overlap and thermal pressure 2 at peak. The
  recorded three-value load averages before arms ranged from 12 to 70; after
  arms they ranged from 19 to 63 where recorded. Two TTFTs were right-censored.
  These add noise to small per-arm populations (3–4 about-1K calls and 1–3
  long calls per arm).
- The client records `sent`, response-header time and first-delta time inside
  `request()`, but normalization drops the response-header time; root warm-up
  events keep only TTFT. `x_qwen.phases` contains prefill, decode and batch
  facts, but `ServeStats` does not include GLM's internal
  `first_token_selection_ms`, `first_token_ready_ms` or
  `first_token_callback_ms`. Server first-byte flush time is not recorded.

The smallest useful trace addition is to keep `headers` and `first` in every
client call record, add request-received-to-driver-submit admission time, and
include the existing GLM first-token selection/ready/callback durations in
`x_qwen.phases`. Add one server timestamp for the first flushed response delta
so local client/server timestamps can separate callback/first-token work from
transport. Keep the existing `batch.queue_ms`; it measures a different,
post-submit wait.

## Scheduling levers, ranked

1. **Layer-slice target: 200 → 400 ms, with the 500 ms minimum unchanged.**
   Each unfinished slice currently incurs a decode round. Halving 8–16
   about-1K slices should save roughly 0.13–0.25 s of decode interleave; halving
   32–56 long slices should save roughly 0.5–0.9 s. Expect the largest p50/p95
   change on long appends. A longer slice can add up to 200 ms of blocking for
   a short request arriving during that slice, so measure short-call p95 beside
   long-call TTFT. This is the recommended first A/B because the existing
   `QWEN_SERVE_GLM_LAYER_SLICE_MS` knob tests the mechanism directly.

2. **Rotate among waiting prefills at slice boundaries.** The current driver
   selects the first active prefill each pump. The 19/134 short calls that
   overlapped another prefill show a short-call tail opportunity: the observed
   overlap cohort is 0.27 s slower at p50 and 1.23 s at p95 than the no-overlap
   cohort. A round-robin prefill choice could reduce that tail, but may delay
   long append completion; measure both classes and throughput.

3. **Change the 10% decode budget for unsliced prefills.** Its effect is at
   decode-budget boundaries and is likely on the order of one or two
   32 ms rounds. It does not remove the mandatory one-round-per-slice service
   on the long calls above.

4. **Admission and active-slot count.** With three agents and four configured
   slots, measured backend slot-queue p95 is below 7 microseconds. There is no
   evidence for changing `QWEN_SERVE_GLM_MAX_ACTIVE` or optimizing slot
   admission before a new workload shows saturation.

The common HTTP `ResponseStream` / `ServeStats` surface is usable for shared
timing fields. GLM uses `serve::batching::driver::Driver` (also used by
Qwen4Exp); V4.1 has a separate scheduler in
`serve/backend_v41/scheduler.rs`. Logging can be shared, but a scheduling
policy change needs a V4.1 adapter or a deliberately shared scheduler change.

## Sources

- KDA three-agent campaign summary and arm quantiles:
  [`2026-11-13-glm-kda-parallel-prep.md`](2026-11-13-glm-kda-parallel-prep.md)
  and raw M5 records under
  `metagross:~/qwen-runs/2026-11-12-glm-prefill/kda-parallel-prep/serve-180/`.
- One-agent comparison: [`2026-11-12-glm-nextn/`](2026-11-12-glm-nextn/).
- Client timing/normalization: `scripts/bench/serve_realistic.py:370-420, 760-801`.
- Server queue and slice behavior: `crates/qwen-cli/src/serve/batching/driver.rs:681-818`;
  GLM slice policy and phase export: `crates/qwen-cli/src/serve/backend_glm5_next/scheduler.rs:11-30, 333-365, 605-626`.
- Shared response stats: `crates/qwen-cli/src/serve/events.rs:91-118` and
  `crates/qwen-cli/src/serve/outcome.rs:37-75`; generation already carries
  first-token durations in `crates/qwen-cli/src/qwen/decode.rs:391-401`.
