# Tier-1 serve concurrency standing, 2026-11-08

Clean main **`842083d5ff0161a822a3981de184b37142035edb`**, M5 Ultra / 256 GiB.
Direct loopback Responses API only; no OpenCode, tunnel or interaction with port
3456. Unmodified source build, one model resident at a time, no drafting.

Three independent clients each issue three sequential turns, resending actual
reasoning/message output. Greedy, seed 1, **128 tokens/turn**; all 63 requests
reach that limit. Each first prompt contains a distinct frozen ~16K-token Rust
excerpt and unique first-user identity; later turns append a ~1024-token tool
read plus wrappers. Actual prompts grow from ~16.3–16.5K to ~18.8–19K. Context
limit **32768**. This adds realistic cold roots to the October 28 protocol;
its old absolute timings/1.25x ratio are not a matched baseline.

Fresh server per arm, no client warm-up. Wall runs from dispatch of the three
agents to the last completion; startup-to-ready is reported separately. Default
V4.1 startup warm-up is included in startup, not request wall. Initial pipeline/
weight caches vary across processes; these are short descriptive observations.

V4.1 has three offered clients in both arms: `c1` sets only
`QWEN_SERVE_V41_MAX_ACTIVE=1`, retaining **four live states**; `c3` uses defaults
**4 active / 4 live**, with at most three active here. The original entry's
**1 active / 1 live** control was not run. GLM/FN `c1` permits one client POST
at a time in fair round-robin order, preserving three-agent interleaving; `c3`
offers three independent chains. Their native server remains serial in both.

Build lock, nice 10, canonical Metal lease with wait enabled cover campaigns.
Each arm cools 25 seconds plus three zero readings five seconds apart. A one-second
watcher stops on nonzero pressure. All samples are zero. The optional reverse
V4.1 pair completed only its default arm; its serial repeat and the legacy
one-live-state control were skipped for the 30-minute budget. M5 build/model
scratch and owned processes were gone at **20:26:02 UTC**, about 28 minutes
from the first host check. No measurements remain running.

## Arm totals

`Ready TTFT` includes client-throttle wait; `Wire TTFT` starts when the POST is
sent. All entries are nine requests / 1152 generated tokens. Memory is GiB.

| Cell | Wall s | Startup s | Mean wire / ready TTFT s | Max client wait s | Max HTTP-header delay s | Peak footprint | Min host availability |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fn-c1 | 52.044 | 8.08 | 3.97 / 14.52 | 22.47 | 0.00 | 14.25 | 94.03 |
| fn-c3 | 51.845 | 0.84 | 14.46 / 14.46 | 0.00 | 22.38 | 14.19 | 102.60 |
| glm-c1 | 85.412 | 9.58 | 7.25 / 24.64 | 35.30 | 0.00 | 6.32 | 63.84 |
| glm-c3 | 85.626 | 1.36 | 24.70 / 24.70 | 0.00 | 35.34 | 6.31 | 63.67 |
| v41-c1 | 60.187 | 1.51 | 17.39 / 17.39 | 0.00 | 31.22 | 19.96 | 40.67 |
| v41-c3 | 71.885 | 2.18 | 11.68 / 11.68 | 0.00 | 0.95 | 19.35 | 38.99 |
| v41-c3-r2 | 71.914 | 15.22 | 11.63 / 11.63 | 0.00 | 0.55 | 19.36 | 31.34 |

## Per-agent results

Each slash-separated triple is **turn 0 / turn 1 / turn 2**. TTFT starts at POST;
for single-client GLM/FN, add the separately recorded client wait to obtain
ready-to-first-content latency. Decode is the server's generated-token/decode-wall
rate; under batching it includes time interleaved with other requests' prefill.
[All requests, ready TTFT, queue fields, reuse and prefill counters](2026-11-08-tier1-concurrency.csv).

| Cell | Agent | Wire TTFT s, T0/T1/T2 | Decode tok/s, T0/T1/T2 |
| --- | ---: | ---: | ---: |
| fn-c1 | 0 | 9.45 / 1.27 / 1.26 | 70.7 / 70.6 / 70.9 |
| fn-c1 | 1 | 9.40 / 1.25 / 1.25 | 70.8 / 70.4 / 70.6 |
| fn-c1 | 2 | 9.38 / 1.24 / 1.25 | 71.1 / 70.7 / 70.8 |
| fn-c3 | 0 | 31.76 / 7.33 / 7.33 | 70.3 / 71.0 / 71.0 |
| fn-c3 | 1 | 9.39 / 23.63 / 7.31 | 70.7 / 70.7 / 70.9 |
| fn-c3 | 2 | 20.57 / 15.49 / 7.32 | 70.7 / 71.0 / 70.8 |
| glm-c1 | 0 | 15.78 / 3.96 / 2.50 | 57.1 / 57.0 / 57.1 |
| glm-c1 | 1 | 15.04 / 3.95 / 2.51 | 57.1 / 57.0 / 57.1 |
| glm-c1 | 2 | 15.02 / 3.94 / 2.51 | 57.1 / 57.0 / 57.0 |
| glm-c3 | 0 | 33.09 / 27.51 / 13.51 | 57.0 / 56.9 / 56.9 |
| glm-c3 | 1 | 50.40 / 16.41 / 12.06 | 56.9 / 56.8 / 56.8 |
| glm-c3 | 2 | 15.76 / 38.62 / 14.95 | 57.0 / 56.9 / 56.9 |
| v41-c1 | 0 | 29.50 / 15.88 / 6.79 | 74.6 / 74.2 / 74.1 |
| v41-c1 | 1 | 17.57 / 24.95 / 6.81 | 74.1 / 74.2 / 74.1 |
| v41-c1 | 2 | 41.33 / 6.89 / 6.77 | 74.5 / 73.9 / 73.9 |
| v41-c3 | 0 | 51.78 / 1.80 / 1.80 | 18.7 / 18.9 / 44.8 |
| v41-c3 | 1 | 11.17 / 1.85 / 1.79 | 3.0 / 18.9 / 18.8 |
| v41-c3 | 2 | 31.27 / 1.81 / 1.82 | 5.2 / 18.9 / 25.8 |
| v41-c3-r2 | 0 | 51.72 / 1.82 / 1.80 | 18.6 / 18.8 / 44.6 |
| v41-c3-r2 | 1 | 10.95 / 1.87 / 1.81 | 3.0 / 18.7 / 18.8 |
| v41-c3-r2 | 2 | 31.05 / 1.82 / 1.81 | 5.1 / 18.8 / 25.7 |

## Interpretation and limits

- **V4.1:** serial **60.187 s**, default **71.885 / 71.914 s**: default is
  **19.4% longer** on this cold-root-heavy cohort (0.84x throughput). Mean wire
  TTFT nevertheless falls **17.39 → 11.68/11.63 s**. All later requests retain
  live sessions. Serial uses **three CED episodes**, default only **one**;
  each default arm logs **74 loaded-prefill calls**, with 512-row quanta, versus
  zero serially. Width reaches **3**, maximum admission `queue_ms` is **0.004 ms** in the first arm and
  **545 ms** in the repeat,
  and no slow-client parking is observed. Low admission queue does not remove
  time spent interleaving GPU work after admission. The cold-root scheduling/
  CED tradeoff agrees with the longer wall; this is not a warm-only throughput
  qualification or evidence to disable batching globally. Later default TTFTs
  are about **1.8 s**, while serially offered clients wait behind peers.
  [Loaded-prefill scheduling](../SERVE.md#deepseek-v41-continuous-batching).
- **GLM:** **85.412 → 85.626 s**. **Flash-Next:** **52.044 → 51.845 s**.
  No meaningful throughput scaling appears: both execute serially. End-to-end
  ready TTFT barely changes (**24.64 → 24.70 s GLM**, **14.52 → 14.46 s FN**).
  Waiting moves from the client throttle to the server: up to **35.3 s GLM /
  22.4 s FN** before HTTP response headers. No 503/retry or admission failure
  occurred. Headers measure queue plus dispatch/parse, not a pure scheduler
  counter; exact native admission queue is exposed only by V4.1 batching.
- Per-agent decode slows under V4.1 mixed prefill/decode load; it must not be
  compared directly to unloaded tg128. Aggregate campaign wall includes cold
  prefill, queueing and continuation work. Different arrival orders decide
  which agent waits first. Batched rounding/loaded prefill can change greedy
  text; actual replies are replayed, with small suffix-count differences.

Memory is sampled at one-second cadence, including startup, not a true transient
high-water mark. `proc_pid_rusage(RUSAGE_INFO_V2)` supplies RSS and physical
footprint; both are retained in raw summaries. Footprint is **not total model
size or Metal working-set allocation**, especially for mapped/no-copy weights.
Host availability uses the engine's conservative formula, free pages plus
max(inactive minus anonymous pages, 0), excluding active file cache. It is a
host pressure signal, **not exact Metal admission headroom**. All configurations
fit; the lowest observed host availability was **31.34 GiB** in the repeated
V4.1 default arm. [Budget formula](../../crates/qwen-llm/src/metal/host_budget.rs#L90).

## Evidence and cleanup

`$HOME/qwen-runs/2026-11-08-tier1-concurrency/` on both hosts retains source/build
identity, protocol/harness, launches, each request/reply/SSE trace, timed server
logs, thermal/memory samples, per-agent metrics, validation and cleanup receipts,
and a SHA-256 manifest. The original frozen excerpts are in the November 7
serve-agent evidence directory on both hosts. One serial/default pair per model,
plus the V4.1 default repeat; no confidence-interval claim. The unrun cells remain
explicit budget skips. Docs/path/link/data checks pass. No engine changes, merge,
OpenCode invocation, or retained scratch worktree; only the docs branch remains.
