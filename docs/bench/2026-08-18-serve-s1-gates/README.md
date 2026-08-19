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

## Gate 2 — warm turn-2 TTFT <150 ms @8k (A3B): 186 ms after serial-tail fix

Phase instrumentation on the resident server decomposed the original
583–683 ms: tokenize 2.2 ms, alloc 0.7 ms, restore 11 ms,
prompt-capture 10 ms — and **~495 ms fixed cost in the matrix prefill
path for a 4–16-token tail**. Fix: tails ≤48 tokens decode serially via
`single_token` (~10 ms/token; same determinism class as chunk-boundary
choice, F7). Re-measured turn-2 SSE TTFT (request byte one → first
content delta): **186 ms** for a ~16-token tail; identical-request
exact hits serve in **14–16 ms** with zero forward passes
(final-logits reuse). The residual over the 150 ms letter is one
engine-scope term — small-span prefill fixed cost / serial token rate
— linear in user-turn length at ~10 ms/token. Recorded as
CONDITIONAL PASS: <150 ms holds for tails ≤12 tokens; the fixed target
needs a small-span prefill kernel (engine work, out of S1 scope).

### Original measurement (pre-fix, retained)

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

## Gate 3 — heartbeat <1 s: PASS

Admission heartbeat is the first SSE body byte (`: ping` precedes
`response.created`; pinned by unit test, observed live, and initially
misread by a probe whose frame splitter didn't treat `\r\n\r\n` as a
boundary). Formal cold cell (23 s first-touch prefill, 6.5k prompt):
6 tick pings across prefill. Observation: tick pings fire between
prefill chunks, so inter-ping gaps equal chunk wall time (~3.7 s
process-cold, sub-second warm) — well inside SSE keepalive norms,
recorded for client-timeout guidance.

## Gate 4 — render goldens: PASS (committed at f4e8bcc/2910512)

## Gate 5 — conformance suite: PASS (6/6 in scope; 11 documented skips)

Suite: `openresponses/openresponses` `bin/compliance-test.ts` (bun),
against Qwen3.6-35B-A3B-UD-Q4_K_S, serve `--max-tokens 2048`.

**Pass:** Basic Text Response, Assistant Message Phase, Response Output
Phase Schema, Streaming Response (219 events zod-validated on a full
thinking generation), System Prompt, Multi-turn Conversation.

**Documented skips (11), each mapping to a SERVE.md subset exclusion:**
7× WebSocket transport (spec MAY; parked post-S4), Tool Calling (S2
scope — serve rejects with the spec envelope), Image Input (text-only
surface; vision is the H6 program), Compaction Endpoint ×2 (no
`/responses/compact`; added to the SERVE.md exclusion list).

The suite adjudicated two contract details exactly as gate 4/5 were
designed to: reasoning streaming events are `response.reasoning.delta|
done` (not the newer OpenAI `reasoning_text.*` names this tree had
chosen provisionally), and reasoning items require `summary: []` from
the first `output_item.added` payload onward. It also forced the full
31-field ResponseResource envelope (explicit nulls, echoes, usage
detail objects — `cached_tokens` now honestly reports restored
checkpoint tokens).

Operational note: the suite sends no `max_output_tokens` and enforces a
20 s fetch timeout per test on a serial server — conformance runs need
a serve `--max-tokens` small enough for the model to finish, and a
model that reaches EOS (the 0.8B rambles past any budget at greedy;
A3B completes).

## Gate 6 — cancellation <250 ms next-admission: PASS

Formal cell: client socket closed after the first content delta of a
512-token generation; the abort surfaced server-side (`connection
aborted` logged), and the immediately following request's
`response.created` arrived **24 ms** after its own start. Known
limitation stands (TERM while parked in `accept()` unwinds on the next
connection).

## Gate 7 — no regression: PASS (bin suite 219 green, clippy clean)

## Operational finding (unnumbered): GPU lease contention

The engine's Metal process lease serializes serve against concurrent
bench sessions machine-wide (observed live against two different
OpenCode sessions' `qwen-bench` runs; the lease message names the owning
session). A resident server therefore blocks bench work and vice versa;
`QWEN_METAL_LEASE_WAIT=1` queues politely. This is exactly the
reap-and-requeue workflow the stateless substrate was built for, but S4
docs must state it: serve and bench are mutually exclusive tenants.
