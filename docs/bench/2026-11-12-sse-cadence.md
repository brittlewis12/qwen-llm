# Serve SSE cadence: raw-socket audit, 2026-11-12

Main **`619e3f501`**, M5 Ultra (`metagross`), release `qwen` built from this
worktree. The 2026-11-11 solo census reported that the V4.1 output partition
buffered SSE deltas until completion (client inter-delta p50/p99
0.015/0.018 ms) and that Flash-Next did the same. This run re-measured the
wire with a plain-socket client: it parses HTTP framing and SSE events itself,
records every `recv` chunk and every complete event, and uses no buffering
HTTP library. **No server-side buffering was found.** The raw socket sees
generated text arrive at the server's token cadence. The old all-at-once
distribution was a client-side collection artifact; the output partition
emits each token as it is produced.

## Method

- **Client.** `scripts/bench/sse_cadence.py`, stdlib only. One HTTP/1.1
  `POST /v1/responses`, `stream: true`, no sampling fields, `x_qwen.stats`
  on, OpenCode-style `instructions`/`tools`/`input` items. The client reads
  headers, then parses close-delimited (or chunked) body bytes directly.
  Event time is the `recv` that completed the `\n\n` frame.
- **Workload.** 12K-token distinct warm roots from
  `docs/bench/2026-11-08-real-opencode-agent-calls.csv`'s corpus source,
  then continuing tool-loop history with 512-token replies for a 40 s window
  (one or three agents). A separate single-shot trio sends one 800-token
  request per agent after warm-up, so no mid-request turn prefill competes.
- **Families.** V4.1 (`DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS`), GLM
  (`GLM-5.3-Flash-UD-IQ3_XXS`), Flash-Next
  (`Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S`); production defaults
  (`QWEN_SERVE_V41_MAX_ACTIVE` unset).
- **Cadence comparison.** Canonical deltas only (`output_text`,
  `reasoning`, `function_call_arguments`), excluding the compatibility
  reasoning alias. Server ITL is per response: `transition_ms/transitions`
  when present, else `decode_ms/stepped_transitions` (GLM/FN). "Steady"
  client cadence drops intervals before every measured request has received
  its first delta plus 0.5 s, which removes concurrent prefill startup.
- **Thermal / lease.** Every model load ran under
  `with-metal-lease --task sse`; 60 s at pressure <= 1 before load and after
  warm-up, abort at 3. Other lanes shared the GPU; queue time is excluded
  from arm windows.

## Continuous workload (40 s, 512-token replies)

| Case | Req | tok/s | client p50/p95/p99 ms | steady p99 ms | server ITL p50/p99 ms | first delta p50 s | server prefill p50 s | max no-byte gap ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V4.1 solo | 6 | 62.21 | 15.57 / 16.39 / 17.27 | 17.18 | 14.65 / 14.79 | 0.80 | 0.76 | 668 (tool span) |
| V4.1 trio | 11 | 94.18 | 27.89 / 29.20 / 30.10 | 29.90 | 27.71 / 28.07 | 0.78 | 0.73 | 1012 (tool span) |
| GLM solo | 5 | 58.52 | 17.14 / 18.59 / 18.91 | 18.90 | 17.25 / 17.27 | 0.91 | 0.91 | 41 |
| GLM trio | 9 | 90.53 | 27.19 / 29.38 / 148.32 | 113.19 | 30.06 / 36.18 | 1.42 | 0.90 | 262 (turn prefill) |
| Flash-Next solo | 9 | 81.08 | 11.52 / 15.70 / 16.94 | 16.79 | 11.86 / 12.22 | 0.66 | 0.63 | 216 (tool span) |
| Flash-Next trio | 14 | 130.53 | 18.52 / 23.71 / 121.85 | 116.13 | 20.37 / 25.71 | 1.01 | 0.64 | 617 (tool span) |

The p50/p95 cadence tracks the server. The p99 tails are one of two
intentional/known pauses:

- **Atomic tool block.** V4.1/GLM/FN hold the `<tool_call>`/DSML/ATEM block
  until the turn ends, validate it, then emit the completed call. The largest
  visible gap is the visible-text -> `function_call_arguments.delta` jump
  (0.67/1.01 s solo/trio V4.1, 0.22 s FN solo, 0.62 s FN trio). This is the
  documented "no partial calls" contract, not per-response buffering.
- **Turn-boundary prefill.** In the continuous trio window a finishing agent
  starts its next prompt while the others decode; GLM/FN p99 picks up those
  ~0.26/0.55 s prefill pauses. The single-shot steady-state runs below remove
  that contention.

## Single-shot trio (one 800-token request per agent)

| Case | steady client p50/p95/p99 ms | server ITL p50/p99 ms | first delta p50 s | server prefill p50 s |
| --- | ---: | ---: | ---: | ---: |
| V4.1 | 25.88 / 31.92 / 36.85 | 26.66 / 31.57 | 2.56 | 1.23 |
| GLM | 27.51 / 29.65 / 30.15 | 31.12 / 34.62 | 8.02 | 2.89 |
| Flash-Next | 16.38 / 20.95 / 21.65 | 19.18 / 19.37 | 2.56 | 0.79 |

With no mid-request turn prefill, p99 tracks server p99 ITL within
1.17x / 0.87x / 1.12x (V4.1 / GLM / FN). First content delta is the
request's own queued prefill plus first-token selection; it is not delayed
by the output path.

## The cited 2026-11-11 case

The old run was V4.1, `QWEN_SERVE_V41_MAX_ACTIVE=1`, 31,918-token prompt,
64 generated tokens, real sampling, release CLI. Re-run with the raw socket:
p50 **16.10 ms** (not 0.015 ms), server ITL **14.74 ms**, first content
delta 0.63 s against 0.54 s prefill, 15 deltas over 54 transitions with the
last 0.63 s in the held tool block. The old client's ~0.015 ms intervals are
the time to walk an already-received body, not wire arrival times.

## Where the output path stands

- `OutputPartition::push` -> family partition -> `ResponseStream::on_partition`
  -> `SseWriter::event` writes and flushes one SSE event; no response-level
  accumulation (`crates/qwen-cli/src/serve/output_partition.rs`,
  `events.rs`, `http.rs`).
- V4.1's recipe `StateMachine::feed` returns output actions per fed chunk;
  the partition applies them immediately. Reasoning/visible splitting emits
  on each push and holds only the longest ambiguous tag prefix.
- GPU token loops call `sink.piece` and `sink.tick` per token, and
  `flush_ready` per token in a burst; the V4.1 scheduler publishes each
  member's queued piece after every step. The 4096-byte transport chunking
  only changes delta boundaries.
- The one deliberate hold is the tool block, which is parsed whole before a
  `function_call` item is published.

## Verification

No production code changed; only the probe was added. With main unchanged:

- `cargo test -p qwen-llm --lib`: **2495 passed, 0 failed, 678 ignored**.
- `cargo test -p qwen-cli`: **1729 passed, 0 failed, 73 ignored**.
- `scripts/check-contracts.sh`: PASS.

## Evidence

`metagross:~/qwen-runs/2026-11-12-sse/`:
`raw/{base,shot,cite}-*/{summary,events,server-lines}.json`,
`logs/`, `analyze_sse2.py`, `analyze_steady.py`, and the preserved
`bin/baseline-qwen{,-tok}`. Local worktree `/Users/tito/code/qwen-llm-sse`,
branch `fix/serve-sse`; no merge.
