# Shared serve TTFT lifecycle instrumentation

Bounded stage started 2026-11-14 09:17 UTC, two-hour limit. Instrumentation only; no scheduling, tokenization, cache or kernel optimization. Fresh `perf/serve-ttft-instrumentation` from main `50d5397ac`.

`QWEN_SERVE_TTFT_TIMING=1` enables a shared per-request trace. Default-off: disabled requests allocate no trace and read no trace clocks. Enabled capture uses monotonic clock reads, atomic first-event fields and typed span records; JSON is built when emitting diagnostics. A local CPU diagnostic of 20 spans and a snapshot measured 28.916 microseconds/request over 100 iterations, not an M5 overhead claim. Global history is bounded to 8,192 quanta; per-request spans to 8,192 with an explicit drop counter. Diagnostic metadata is reserved in the transport CPU budget (16 MiB/request); typical capture is much smaller. No prompt/token contents are recorded.

HTTP accept/receive/body, parse/validate, render, full tokenization, enqueue/dequeue, submission/admission, first sample/raw piece, first SSE flush and first **visible** flush have distinct markers. V4.1 adds full/backend boundary tokenization, prefix match/restore, output admission, state allocation, checkpoint and each prefill slice. Request IDs join the standard response ID retained by the client. Prefill/decode quantum IDs and bounded history account for work before a request reaches the scheduler.

A heartbeat or response-created event is not client TTFT. The socket flush-return stamp follows the actual SSE write/flush; first nonempty text/reasoning/function-argument delta is recorded separately. Each first-visible event emits `serve ttft timing:` immediately, including cancelled-at-window-end requests; final emission retains the same first-visible cutoff and completed metadata. The common harness now retains headers, response IDs and warm-call timestamps.

Shared APIs: `BatchSink::lifecycle()`, `Trace::enter/mark/extent`, `Span::current(name, position, rows)`, and `Round::new`. The common batching driver sets admission/prefill/initial-sample scopes, so GLM can add precise slice positions and cache subspans with these same calls. Its existing policy is unchanged. V4.1 kernels and row partitions remain identical to main. No changes to luna's GLM worktree or A/B.

CPU tests cover disabled capture, cross-thread publication/scope restoration, quantum linkage/clipping, bounded metadata/capture cost, and unchanged real loopback SSE bytes with control versus visible flush separation. Existing transport lifecycle tests also pass: 23 tests total, zero failures/ignored. New CPU tests are in the acceptance inventory; no ignored tests added.

## Pinned source and controls

The measured Rust source is `0a54d231622131bc37e5c41ae6e97357e7f9a72c`, tree `3a761d17bd0c621e44f57047608c82397f26d63a`, release profile. Rust sources remain identical after the harness-only retention fixes (`300bd75cb`). No changes to `qwen-llm` or kernels relative to main `50d5397ac`. Release build used `nice -n 10`, one <=15-minute lease, and took 4m09s. The executables were copied out of target before measurement; their complete hashes and all three metallib identities are retained in the pin manifest.

- `qwen`: `bc0e4fe375a4b377204d17d10f827689efb87088675ce56db0b1aea789d3ea6e`.
- `qwen-tok`: `29f4edce59d54fe79c99b4668a3e250037270b3b86dfee706f3e2b768f9c9ac0`.

File A is the production GSQ-RCO IQ2_XS DeepSeek V4.1 Flash 1017 GGUF plus streamed IQ4_NL Engram. Three agents, greedy, steady-own-history v2 plan, seed 20261112, 120-second window after a 15-second ramp, 49,152-token disjoint warm roots with 32-token replies, context 196,608. Existing native prefill timing remains enabled. MAX_ACTIVE=4, SAMPLE_WORKERS=3, PREFILL_FAIR=1. Actual server niceness is checked as 0. Engram must reach zero resident pages before every load. Thermal settling requires <=1 continuously for 60 seconds, allows 2 during operation and aborts at 3. No admission limits, scheduling policy, input geometry or output sampling were changed.

The shared APIs are sufficient for GLM to add a precise slice event at its backend boundary:

```rust
let timing = sink.lifecycle();
let _request = timing.enter();
let _slice = Span::current("backend_layer_slice", position, rows);
```

The common batching driver already supplies the request scope, admission/prefill events and round IDs, so a backend called inside it needs only the last line. Missing backend-specific stages remain null rather than being filled with estimates.

## First attempt: memory-admission failure, retained separately

The measurement lease started 10:20:39 UTC. Thermal pressure initially 2 fell to <=1 before loading; the preload settle and Engram purge passed (0 resident pages, msync return 0). Server PID 82223 ran at nice 0. All three warm roots completed, client TTFT 29.433 / 29.655 / 28.893 seconds. [Root-only joined measurements](2026-11-14-serve-ttft-instrumentation/failed-roots-summary.json) (N=3) give native prefill p50/p95 29,333.382 / 29,564.228 ms, outside-native 92.504 / 99.976 ms, tokenization 31.716 / 49.397 ms, and checkpointing 25.001 / 26.395 ms. Root p95 is the maximum of three sequential warm calls, not a contended-root estimate.

The append window stopped on `memory_admission_denied` / `host_availability_insufficient`: transport required 399,532,032 bytes, reported host remaining 289,210,368 bytes. The run's thermal maximum was 0; maximum live process disk-read counter 193,836,449,792 bytes, including model loading/warmup. Three Ghost scenes accounted for roughly 37.4 GiB resident memory in the post-stop snapshot. No foreign process was stopped and no memory guard was relaxed.

This is a failed/incomplete run, not a latency qualification. Its first-visible server traces survive (21 request traces, no parse failures or dropped spans), but the old harness discarded append client timelines on an exception. Root timelines survive. The harness now writes all raw client rows plus planned window boundaries in `finally`, so an interrupted run can be analyzed honestly. One retry uses the same pinned Rust binaries, limits and traffic; only this retention behavior changes. First-attempt append server-only observations are diagnostic and excluded from the primary client quantiles.

## Primary component measurement

The retry completed the full 120-second window (10:47:33.460–10:49:33.460 UTC), one lease 10:42:22–10:49:36 UTC (7m14s). Server PID 91330, verified nice 0; purge 0→0 pages; no competing model at load. Total process disk-read maximum **189,521,977,344 bytes** (model load, roots and window together; 0.5-second samples). Load was about 9–13 during the run; compiler processes were observed during the window (this lane’s build had finished at 10:06 UTC). Thermal maximum 2 was permitted by the frozen protocol: about 87.4 s at 0, 1.0 s at 1 and 31.6 s at 2 during the window, none at 3. The harness's stricter historical `thermal_flagged` bit is true at 2; that does not mean the stated allow-2/abort-3 policy failed. Aggregate delivered rate **90.325 tok/s** is context, not an A/B or an instrumentation-overhead estimate.

All **32 eligible TTFTs** join by response ID: 19 short calls (70–487 actual new rows), 9 ~1K calls (608–1,798 rows), three warm roots (49,164–49,509 rows), and one 2,002-row control retained in the ledger. Two append generations were cancelled at window end after their first visible delta; their TTFT and native prefill are fully observed and included. No TTFT-censored eligible calls, missing joins, parse errors or dropped spans. Three carry-in calls and ramp-only calls are excluded from the primary appendix. Every per-call non-overlapping lifecycle partition balances to client TTFT within 0.001 ms. Nearest-rank p95 equals the maximum in these small class samples; this is a component diagnostic, not a stable tail estimate.

[Derived calls and peer request/round IDs](2026-11-14-serve-ttft-instrumentation/calls.jsonl), [full distributions](2026-11-14-serve-ttft-instrumentation/summary.json), [controls](2026-11-14-serve-ttft-instrumentation/controls.json), [binary/metallib pin](2026-11-14-serve-ttft-instrumentation/pin-manifest.json), and [raw input hashes](2026-11-14-serve-ttft-instrumentation/input-manifest.json). Raw runs remain at `metagross:~/qwen-runs/2026-11-12-v41-prefill/ttft-instrumentation/`, with the clean arm under `measurement/`. First-attempt root data and failed server observations remain separate.

### Per-class costs, p50 / p95 milliseconds

Quantiles of components do not add. The first table shows the broad accounting; the following tables split its work and wait intervals. Parent and child spans must not be added together.

| Component | Short, N=19 | ~1K, N=9 | Roots, N=3 |
|---|---:|---:|---:|
| Client TTFT | 497.202 / 1622.344 | 1108.693 / 1551.415 | 25017.426 / 25188.427 |
| Native prefill wall | 368.163 / 658.084 | 1028.846 / 1468.336 | 24939.136 / 25083.699 |
| Outside native | 105.692 / 1154.966 | 85.773 / 99.808 | 78.290 / 104.728 |
| Logged ready waits | 25.679 / 1100.053 | 18.099 / 27.930 | 0.286 / 0.304 |
| Outside native after removing logged waits | 79.153 / 90.709 | 65.895 / 86.098 | 78.097 / 104.424 |
| Tokenization: full + boundary encodes | 52.086 / 61.514 | 39.495 / 56.179 | 30.570 / 53.613 |
| Checkpoint capture (inside slice, outside native engine) | 21.649 / 24.458 | 21.585 / 24.138 | 19.411 / 20.713 |

### Request preparation and cache work

| Component | Short, N=19 | ~1K, N=9 | Roots, N=3 |
|---|---:|---:|---:|
| Client send → TCP accept | -0.206 / 2.837 | -0.197 / -0.112 | 0.144 / 0.201 |
| HTTP body read | 0.136 / 0.279 | 0.106 / 0.235 | 0.091 / 0.173 |
| Body received → parsed/validated | 0.265 / 0.338 | 0.250 / 0.340 | 0.217 / 1.013 |
| Chat render | 0.085 / 0.126 | 0.082 / 0.099 | 0.041 / 0.352 |
| Full prompt tokenize on HTTP worker | 30.966 / 37.328 | 24.577 / 34.185 | 18.566 / 34.508 |
| Boundary tokenization, inclusive parent | 20.804 / 25.080 | 14.918 / 21.994 | 12.004 / 19.105 |
| Generation-prefix encode, child of boundary tokenize | 20.580 / 24.859 | 14.614 / 21.775 | 11.824 / 18.924 |
| Shared-prefix encode, child | 0.228 / 0.319 | 0.222 / 0.318 | 0.180 / 0.228 |
| Prepared-token copy | 0.005 / 0.006 | 0.005 / 0.006 | 0.004 / 0.005 |
| Prefix preparation | 0.004 / 0.004 | 0.004 / 0.004 | 0.003 / 0.013 |
| Prefix take, inclusive parent | 0.017 / 0.021 | 0.017 / 0.019 | 2.379 / 2.439 |
| Live prefix match, child | 0.006 / 0.007 | 0.005 / 0.007 | 0.000 / 0.000 |
| Restore/capacity, child | 0.009 / 0.011 | 0.009 / 0.010 | 0.754 / 0.777 |
| Output memory admission | 0.018 / 0.022 | 0.015 / 0.020 | 0.018 / 0.038 |
| Cold state admit/allocate | not logged | not logged | 1.709 / 1.709 |
| Checkpoint capture | 21.649 / 24.458 | 21.585 / 24.138 | 19.411 / 20.713 |

Root restore has N=2 (the first root is cold); state allocation may have fewer observations. Full metric counts are in the JSON. A missing backend fallback encode is absent, not an estimated zero.

### Queueing, prefill and delivery

| Component | Short, N=19 | ~1K, N=9 | Roots, N=3 |
|---|---:|---:|---:|
| TCP accept → HTTP worker start | 15.191 / 1082.155 | 6.440 / 12.353 | 0.062 / 0.133 |
| Prepared job enqueue → dequeue | 10.602 / 19.264 | 11.647 / 22.078 | 0.007 / 0.017 |
| Dequeue → scheduler submit | 0.000 / 0.001 | 0.000 / 0.002 | 0.001 / 0.012 |
| Scheduler slot wait | 0.005 / 0.057 | 0.005 / 0.019 | 0.005 / 0.006 |
| Admission work, inclusive parent | 21.201 / 25.473 | 15.295 / 22.387 | 14.746 / 21.478 |
| Admitted → first prefill slice | 0.002 / 0.002 | 0.002 / 0.002 | 0.003 / 0.017 |
| Prefill slices incl. checkpoint/validation | 377.856 / 680.136 | 1044.502 / 1490.672 | 24979.864 / 25126.351 |
| Between own prefill slices | 0.000 / 0.000 | 0.000 / 0.000 | 0.120 / 0.210 |
| Last slice → finish bookkeeping end | 0.046 / 0.049 | 0.047 / 0.049 | 0.042 / 0.048 |
| Finish → first token sampled | 0.438 / 0.440 | 0.439 / 0.455 | 0.453 / 0.459 |
| First sample → raw piece queued | 0.021 / 0.025 | 0.024 / 0.028 | 0.017 / 0.018 |
| Raw piece → first visible SSE flush | 0.859 / 1.427 | 1.029 / 1.303 | 0.886 / 0.980 |
| Visible flush-return → client delta read | 0.633 / 0.736 | 0.576 / 0.762 | 0.237 / 0.252 |
| Ready wait overlapping peer prefill | 0.000 / 1082.090 | 0.000 / 0.000 | 0.000 / 0.000 |
| Ready wait overlapping peer decode | 22.706 / 33.196 | 17.959 / 27.813 | 0.000 / 0.000 |

The fixed residual is mostly **CPU tokenization, checkpoint capture, and waiting for the GPU-feeding loop around decode rounds**. Render, live-prefix lookup/restore, first sampling and response transport are small. Short tokenization p50 is 52.086 ms, including 30.966 ms full-prompt encoding on the HTTP worker and 20.580 ms re-encoding nearly the same prompt through the generation-header boundary during admission. ~1K tokenization is 39.495 ms (24.577 / 14.614 ms). Checkpointing adds about 21.6 ms to both classes. The ordinary ready-wait p50 is 25.679 / 18.099 ms, mostly crossing peer decode quanta; slot admission itself is about 0.005 ms.

Prompt lengths differ by class: short 52,816–63,881 (median 58,355), ~1K 51,944–62,992 (median 56,838). Tokenization wall also includes host descheduling; the class difference is not a measured consequence of append size alone. The fixed-cost variance comes from full-history length, host scheduling/compiler load and checkpoint copies. The large tail variance is the residual duration of an already-running peer prefill. Root differences between attempts also include cold-file/native I/O and host state; no causal cross-attempt comparison is made.

This does not identify pure CPU cycles versus host descheduling inside a span. Likewise native wall includes Engram preparation and GPU/driver waits, not just arithmetic. No extra native transition was needed for the first visible text in these observed calls. The existing native profiles retain Engram/API/GPU details; full process disk bytes must not be assigned wholly to Engram or a single request.

### The short tail, now with explicit request IDs

Three of 19 short calls crossed a peer prefill during a logged wait; none of the nine ~1K calls did. The largest two are:

| Waiting call | New rows | Client TTFT | Own native | TCP accept → HTTP start | Blocking peer / round | Prefill overlap |
|---|---:|---:|---:|---:|---|---:|
| `a1-t8` | 249 | 1622.344 | 467.378 | 1082.155 | `a0-t18`, request48, round3591 | 1082.090 |
| `a1-t6` | 462 | 1459.120 | 637.032 | 739.100 | `a0-t11`, request39, round2177 | 734.905 |

The first request was already TCP-accepted while the shared loop remained inside the peer's synchronous prefill; its HTTP worker had not started. It therefore was not a parsed, tokenized scheduler job yet. After that, ordinary mailbox waits crossed peer decode rounds. This resolves the earlier ambiguity: the p95 peer-prefill tail is predominantly **before HTTP processing**, not scheduler slot queueing. The peer prefill can contain Engram I/O internally; that does not make it a second independently additive wait. Per-call peer IDs/rounds are retained in the ledger. The current span hooks capture logical prefill slices; lower-level kernels inside a synchronous slice retain their existing native profile.

## Recommendation: exact tokenization reuse first

Measure **template-aware reuse of the worker's full tokenization for generation-boundary cuts**, with the current independent prefix encode as the fallback at any unproven boundary. Today V4.1 re-encodes almost the entire 52–64K prompt on the GPU-feeding thread after the HTTP worker has already encoded the full prompt. The duplicated generation-prefix encode alone prices **20.6 ms short / 14.6 ms ~1K p50**; total tokenization is the larger 52.1 / 39.5 ms upper opportunity bound, not a promised saving. The tokenizer trait currently returns IDs without offsets, so this needs an exact boundary-aware API or proven special-token segment reuse. Raw BPE prefix concatenation is insufficient.

Freeze exact full token IDs and every checkpoint/generation cut against today's path over real chat/tool/Unicode boundaries; use the same fallback when a cut is not a token prefix. Then paired served TTFT with this instrumentation, including TPS and decode-gap controls. Moving the same work to another thread alone does not remove its client TTFT cost.

Next, consider moving the **final Ready checkpoint** after first-token publication but before the first state-mutating decode transition: it prices another ~21.6 ms per append. Earlier semantic checkpoints cannot be deferred across a state advance. This is a design suggestion, not implemented or proven here. A row-preserving prefill yield remains the larger **short-p95** lever (up to the measured 1.08 s peer exposure in this run), with latency transferred to other work; it does not explain or remove the tokenization cost paid by every call. Do not treat the 52 ms tokenization bucket or 1.08 s overlap as guaranteed recoverable time.

No scheduling, cache, tokenization or kernel optimization was implemented in this stage.

## Validation and closure

CPU lifecycle/wire/transport tests: 23 passed, zero failed/ignored. Full standard contracts pass on the source and harness-retention commits; ENV regenerated, new CPU tests inventoried. The measured Rust sources remain byte-identical to the final tip. M5 functional witness: all three roots and the full three-agent window completed with opt-in tracing, no dropped events, and balanced client/server accounting. Total diagnostic on/off overhead was not measured; the knob stays default-off.

M5 process audit found no task processes after the clean arm. The task worktree, compiled target, pinned executables, bundles and transfer archives were then removed; raw runs and manifests remain. No process was killed for cleanup. The local process audit also found no task processes. Final committed-tree contracts are recorded in the handoff.

To reproduce from retained raw evidence:

```sh
python3 docs/bench/2026-11-14-serve-ttft-instrumentation/analyze.py "$RUN/measurement/serve" --out "$OUT/primary"
python3 docs/bench/2026-11-14-serve-ttft-instrumentation/analyze.py "$RUN/serve" --warm-only --out "$OUT/failed-roots"
```

The maintained script reproduces the committed primary distributions and ledger byte-for-byte. `--warm-only` explicitly avoids inventing missing append client rows in the failed first attempt.
