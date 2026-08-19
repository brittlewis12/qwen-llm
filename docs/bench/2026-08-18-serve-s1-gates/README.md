# S1 gate execution record

Status: in progress, 2026-08-18. Binary: `serve/s1` @ 9f79dac (release).
Host: Apple M4 Max. Contract: docs/SERVE.md. Client artifact:
`~/code/llm/game/play_serve.py` (thin serve client sharing play.py's save
format); `play.py` repaired in passing (`RUST_LOG=warn,qwen_diag=info` —
the 1ecf208 tracing migration had silently broken its stop_reason gate).

## Gate 1 — thin client, byte-identical saves: PASS (one finding, one letter-miss)

Scripted 5-turn session, The Current ring0 v0.2 (~7.3k-token system),
Qwen3.6-35B-A3B-UD-Q4_K_S, temp 0, seed 42, preserve-thinking,
`QWEN_GREEDY_GPU_ARGMAX=0` on the CLI arm for sampler parity.

- **Main phase: 11/11 messages byte-identical** between the serve items
  path (`play_serve.py`) and the legacy CLI path (`play.py` +
  `--durable-prefix-cache`), including full `<think>` reasoning
  reconstructed from items. Meta equal except `created_at` (timestamps
  excluded per gate definition).
- **Fork phase: byte-identical under matched restore topology.** The
  uncontrolled first comparison diverged mid-generation (char 229 of the
  new turn; prior history identical) — **F7, engine determinism
  finding**: CLI restored from a mid-prompt durable checkpoint while the
  restarted server cold-prefilled; different restore partitionings
  perturb logits enough to flip a temp-0 near-tie. The cold-vs-cold
  control run is byte-identical, proving harness equivalence. Recorded
  consequence: transcripts are byte-deterministic *conditional on
  checkpoint-restore topology*; no surface should promise byte identity
  across differing cache states at temp 0.
- **Letter miss:** client is 187 lines (formatter-expanded) against the
  frozen <150; 62% reduction from play.py's 493 with zero process/model
  management and zero stderr scraping. Deviation recorded rather than
  gamed; a follow-up trim decides whether the number or the file wins.

## Gate 2 — warm turn-2 TTFT <150 ms @8k (A3B): FAIL as measured

Client-measured request-to-first-content-delta, resident server, RAM
completed-checkpoint hits covering the full prior context every turn:

| turn | context (tok) | matched | restore_ms | ttft_ms |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 7,371 | 7,371 | 11.5 | 583 |
| 3 | 8,085 | 8,085 | 14.3 | 643 |
| 4 | 8,511 | 8,511 | 12.7 | 659 |
| 5 | 8,989 | 8,989 | 13.1 | 683 |

Restore is ~2% of TTFT; the checkpoint design is not the bottleneck.
Suspects, in estimated order: per-request sequence/scratch allocation at
~13k capacity, prompt-boundary snapshot capture sitting pre-decode in
the TTFT path (~185 MB memcpy at these lengths), 30k-char tokenization,
tail prefill, first decode step. Next action: instrument the backend
stats line with per-phase timings, then attack the largest term
(candidates: defer/skip prompt-boundary capture when completed capture
will cover it — matching the CLI's shadowing policy k3 already endorsed
in R2 — and pool sequence/scratch allocations across requests in the
resident server).

## Gate 3 — heartbeat <1 s: mechanism verified, formal cell pending

Admission heartbeat is the first SSE byte (`: ping` precedes
`response.created`; pinned by unit test and observed live). Tick-driven
heartbeats between prefill chunks are rate-limited to 1 Hz. Formal
cold-prefill measurement cell pending.

## Gate 4 — render goldens: PASS (committed at f4e8bcc/2910512)

## Gate 5 — conformance suite: PENDING

## Gate 6 — cancellation <250 ms next-admission: mechanism verified, cell pending

Client disconnect during streaming surfaces as write failure and aborts
decode between token steps (observed live: `serve: connection aborted:
Broken pipe` from a `head`-truncated stream). Formal
abort-to-next-`response.created` measurement pending. Known limitation
(recorded at 9f79dac): TERM while parked in `accept()` unwinds only on
the next connection.

## Gate 7 — no regression: PASS (bin suite 219 green, clippy clean)

## Operational finding (unnumbered): GPU lease contention

The engine's Metal process lease serializes serve against concurrent
bench sessions machine-wide (observed live against two different
OpenCode sessions' `qwen-bench` runs; the lease message names the owning
session). A resident server therefore blocks bench work and vice versa;
`QWEN_METAL_LEASE_WAIT=1` queues politely. This is exactly the
reap-and-requeue workflow the stateless substrate was built for, but S4
docs must state it: serve and bench are mutually exclusive tenants.
