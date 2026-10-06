# qwen serve — contract and program

Status: S0–S3 functionality is present in the current working tree. Gate
evidence is mixed and is recorded separately below; implementation status must
not be read as a blanket gate pass.

Evidence base: S0 packet
(`docs/bench/2026-08-18-facade-s0-render-prefix-stability/`), the S1/S2/S3
gate records under `docs/bench/`, the successful real OpenCode session
`ses_fe307ea3effefOzYcDBgTbYiie` (2026-08-20, 5 h overnight / 58 requests /
133 k tokens), adversarial reviews
(`ses_fe8ee25c6ffe`), and the full investigation session (OpenCode Recall).

## Scope

Private, single-box engine serving local clients over loopback.

K2 Horizon dense 7B has separate raw-string and verified native chat/tools
`/v1/responses` subsets: see [K2 Horizon Raw Profile](#k2-horizon-raw-profile)
and [K2 Horizon Verified Chat](#k2-horizon-verified-chat). It does not inherit the
Qwen/DeepSeek chat, reasoning, tools, snapshots, or long-context capabilities.

Operating goals, in order:

1. **Foundations that hold** — durable continuity across restarts,
   honest behaviour when busy, and DeepSeek V4 that is actually usable.
2. **Concurrency / batch-serving responsiveness** (continuous batching).
3. **Long-context performance stability.**

## Program arc (decision record)

| Unit              | Contents                                                                                                                                                                                                                                                                     | Consumer                                    | Gate                                                                                                                                                                                                                                  |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| S0                | Prefix-stability falsifier                                                                                                                                                                                                                                                   | measurement                                 | DONE — see packet RESULTS                                                                                                                                                                                                             |
| S1                | Resident serial server, Open Responses subset, no tools                                                                                                                                                                                                                      | the game (thin HTTP client)                 | Functional evidence passed, but not the literal full gate: client was 187 lines versus `<150`, and the `<150 ms` TTFT gate failed at 186 ms. Conformance was 6/6 in scope and cancellation measured 24 ms — docs/bench/2026-08-18-serve-s1-gates/ |
| S2                | Tool items (XML-parameter form from the template oracle), exact declared/allowed-tool enforcement, continuation rendering                                                                                                                                                    | OpenCode via stock `@ai-sdk/open-responses` | Provider gate: 10/10 requests checkpoint-hit (94–100% restored), 5/5 turns tool-called. `allowed_tools` subsequently landed. Production evidence: real OpenCode session `ses_fe307ea3effefOzYcDBgTbYiie` ran successfully for 5 h overnight — docs/bench/2026-08-19-s2-agent-gate/ |
| S3                | Pre-opened (headless) reasoning support, DeepSeek V4 family backend, DFlash drafter integration. `encrypted_content` is **not** implemented.                                                                                                                                    | OpenCode, DS4 clients                       | Core cells 1–5 passed: warm snapshot hits, CLI byte identity including CJK/emoji, 91% reasoning replay restore, clean headless partition, fail-closed behavior. Live cells 6–9 still require rerun — docs/bench/2026-08-19-s3-ds4-gate/ |

## Remaining program (2026-08-20)

**F1 — Durable continuity: implemented for Qwen and DS4 (2026-09-23), GPU
validated (2026-09-24: a 1.51 GB 27B snapshot, larger than the write queue,
persisted at shutdown and restored after restart in 862 ms).** See
[Durable snapshots](#durable-snapshots-cross-restart-warmth) for the behavior
and flags. Since 2026-09-26 Qwen also publishes the latest continuation
boundary after 30 s idle, so a crash after an idle period keeps the session
(27B Q8, 66K: restored from disk after SIGKILL). A crash before the first
idle publication still loses what was only in RAM.

**F2 — Per-family publication policy (the one F1 implements).** Publication
cost is not uniform, and this decides the policy:

| model | KV per token | snapshot at 133 k |
| --- | ---: | ---: |
| DeepSeek V4 | 6.9 KB | 0.94 GB |
| Qwen3.6 35B A3B | 20.5 KB | 2.7 GB |
| Qwen3.8 27B | 93 KB | 12.4 GB |

DeepSeek V4 can afford DwarfStar-style periodic writes during
generation plus a shutdown flush. A dense 27B cannot: 12 GB per turn is
unwritable, so Qwen families get **publish on graceful shutdown and idle
only** until content-addressed delta chains exist (S0 F6 — consecutive
snapshots share nearly all their bytes). Both halves are implemented
(idle since 2026-09-26). The 12.4 GB row above overstates today's F16
cache: the 27B's 16 attention layers hold 64 KiB per token, so 133K is about
9 GB (4.5 GB measured at 66K).

**F3 — Honest behaviour when busy: implemented, live rerun pending.** Serve
remains single-flight, but a separate acceptor now fails concurrent connections
fast with `503 Service Unavailable`, `Retry-After: 1`, and a `server_busy`
envelope instead of leaving them silent in the listen backlog.

**F4 — DeepSeek V4 tool support: implemented.** DS4 serve renders declared
tools with the 0731 chat template (`render_ds4.rs`, pinned against the
template's `tools_chat_one_round` case) and parses DSML tool calls into
Responses `function_call` items.

**F5 — Memory admission: implemented, live rerun pending.** Qwen requests are
priced before execution and fail with `503` when denied; a process-memory
shortfall first evicts unpinned RAM snapshots (keeping the one the request
restores) and re-checks once. Qwen and DS4 boundary
capture is memory-admitted and best-effort: denial or capture failure skips the
snapshot while the request continues. Both caches are byte-bounded; DS4 session
construction returns ownership on failure so residency remains recoverable.

**F6 — Concurrency / continuous batching.** The engine already provides
`admit_independent_queue2`, `create_dense_batch8_executor`,
`create_moe_batch16_executor`, and `concurrent_jsonl.rs` runs width-2
overlapped decode today. Route serve through them: wiring and
scheduling, with F3 as the on-ramp.

**F7 — Long-context performance stability.** Measured overnight
(2026-08-20, 27B Q8): decode fell 14.5 → 8.4 tok/s from 31 k → 133 k, and
restore grew to ~1.5 s. Resolved by the 2026-08-22 attention retune:
serve on 27B Q8 now decodes 14.0 tok/s at 66 k (engine 14.0; 12.2 at
128 k, 1.59x llama.cpp), with or without a drafter loaded (PERF-LOG
2026-09-26). Cold prefill (≈176 tok/s at 64 k) is now the long-session
cost; snapshot reuse is its lever.

Also open: selector `emit_completion` telemetry (k3 R1.4), and fresh execution
of S3 live cells 6–9 (LRU eviction, DS4 cancellation/heartbeat, DS4 TTFT @8k,
memory envelope). Engine optimization candidates live in `docs/PERF-ROADMAP.md`.

## Parked, with reasons

- **WebSocket transport / connection-local continuation.** Optional in
  the spec. On loopback, posting 133 k of history costs ~16–23 ms of
  tokenization and ~1 ms of bandwidth against a ~1.5 s restore and tens
  of seconds of generation. Its value is skipping that restore by
  holding session state on the connection, which requires the server to
  stay up between turns.
- **`previous_response_id` stored mode** — same reasoning.
- **`encrypted_content` opaque reasoning** — the S2 capture showed plain
  reasoning content already replays verbatim through the stock provider,
  so this is hardening, not a requirement.
- **CC (chat-completions) shim, items npm provider, `/responses/compact`,
  installers** — no consumer on this box.
- **`tool_choice` `required` / `none` / forced-function** — unused.

## Current behaviour

One new subcommand:

```sh
qwen serve -m MODEL [--addr 127.0.0.1:8737] [--max-tokens N] \
  [--max-context-tokens N] [--snapshot-cache-mib auto|MIB] [--drafter GGUF] \
  [--snapshot-idle-ttl-secs 3600] [--snapshot-max-age-secs 86400] \
  [--snapshot-half-life-secs 600] [--durable-snapshot-dir PATH|off] \
  [--durable-snapshot-max-mib auto|MIB] [--durable-snapshot-min-tokens 1024] \
  [--durable-idle-publish-secs 30] [--durable-shutdown-secs 30] \
  [--lens-data-dir PATH] [--web-root web/dist]
# Qwen: without --max-context-tokens the admission ceiling is the smaller of the
# 262,144 hard default and the GGUF's declared context length; --drafter is
# accepted for dense targets only (an MoE target fails startup rather than
# silently running serially)
# DeepSeek V4 additionally requires --max-context-tokens (startup-fixed forward budget)
# Muse Glimmer and Qwen3.8-Flash-Next require both --max-context-tokens and
# --max-tokens (resident session capacity is fixed at load); admitted capacity
# may extend through the model's declared context.
# K2 requires explicit --max-context-tokens (within checkpoint context) and
# --max-tokens; --snapshot-cache-mib is ignored (live-session prefix reuse).
# GLM-5.3-Flash requires both too (device memory admits the session at
# startup and names the capacity that fits); --snapshot-cache-mib only warns.
```

**Durable diagnostic workbench.** `--lens-data-dir PATH` enables saved Lens history
on every serving family. Qualified ordinary House
Qwen3.6/3.8 deployments accept messages, typed assistant prefills and explicit
sampling; other deployments can read history without claiming native execution
support. `--web-root web/dist` independently serves the prebuilt Bun client;
Rust never invokes Bun. Supported passive heads offer plain original-forward
readouts; explicit registered fitted assets enable fitted heads and directions.
Scoped ordered interventions, retained source/full scores and whole-site before/after
pairs are recovered. Compatible saved diagnostics remain inspectable.
Ordinary `/v1/responses` remains available, sharing one execution reservation.
Accepted native jobs survive client disconnect; history reads never run inference.
See [Lens Workbench](LENS-WEB.md) for limits, overload, recovery and qualification.
This job directory is separate from reusable model snapshots.

**Snapshot cache policy (Qwen and DS4).** Both RAM caches share one policy
(`qwen_llm::snapshot_policy`), logged on the `serve limits:` line:

- **Budget.** `auto` (default) = min(25% of physical RAM, 50% of the Metal
  recommended working set left after the model loads), at least 1 GiB; an
  integer is MiB, taken verbatim (0 disables capture).
- **Eviction** when over budget removes the entry with the lowest
  `score * reusable_tokens / bytes`, where a hit adds 1 to a score that decays
  with `--snapshot-half-life-secs` (ties: older access). `0` is exact LRU.
- **Expiry.** Entries unused for `--snapshot-idle-ttl-secs` or older than
  `--snapshot-max-age-secs` are dropped (0 disables each), lazily on lookup and
  insert and from the idle admission loop, so memory is returned while idle.
- **Pins.** The request's transcript entry is pinned while its completed
  boundary is inserted; pinned bytes count against the budget, so a completed
  boundary that would need to evict it is skipped.
- **Memory pressure.** A capture denied for process-footprint headroom evicts
  by the same rank for the deficit and re-checks once. Metal-headroom denials
  are not retried: snapshots are CPU arenas, invisible to the Metal signal.

### Idle residency (no-copy families)

Metal wires no-copy weights only while commands use them, and unwires them
about 2 s after the GPU goes idle (observed on macOS; not an API
guarantee). The next request then pays to re-wire every page it touches.
`--idle-residency-secs SECS` (or `QWEN_SERVE_IDLE_RESIDENCY_SECS`) keeps a
family's no-copy weights wired for that long after each request, using
ordinary keep-alive commands: 500 ms pulses that mark the weights used. It
is not a residency set.

- **Window.** It opens at the first activity: the warm-up (GLM) or the
  first request that submitted GPU compute work (other families), so a
  pulse never faults in a cold model. It reopens when such a request
  finishes, even if its client aborted it after submission. A request is
  judged by what it submitted: the backend snapshots the process's
  compute-encoder count when the request starts and compares it when the
  connection finishes. A server-side failure (5xx, including GPU and
  command-buffer faults and memory-admission refusals) closes the window
  at once, so no pulse follows a fault. These never open or renew it:
  - model lists, malformed or refused requests, and disconnects;
  - memory-admission refusals and session or runner allocation failures;
  - a cancellation before the first command;
  - the keep-alive pulses themselves (a snapshot left unmatched when the
    server goes idle is dropped before any pulse).

  Blit-only work does not count. `RUST_LOG=info,qwen_diag=debug` logs
  each finish (`finished compute_encoders=N window=renewed|unchanged|closed`);
  `scripts/reference/serve_residency_poll_check.py` checks polling, a
  refusal, a completed request and a client abort after submission
  against those lines.
- **Default and bounds.** Default 60 s; 0 disables.
- **Safety.**
  - Pulsing is suspended while the host reports memory pressure of
    warning or worse. It resumes on the next idle tick after the pressure
    clears, within the same window and with no cooldown, and an
    already-submitted pulse is not cancelled. A host that cannot report
    pressure counts as unpressured (logged once). This is suspension, not
    protection against re-pinning under oscillating pressure.
  - It stops for the server's lifetime on any pulse failure, including a
    pulse still running after 10 s.
  - The last pulse settles at shutdown.
  - A 1 GiB kill check recovered wired memory after SIGKILL between pulses
    and with a command in flight
    (`docs/bench/2026-10-05-residency-kill-check/`).
- **Eligible families.** Every family whose serve weights are no-copy GGUF
  windows: GLM-5.3-Flash, K2 Horizon, DeepSeek V4, Muse Glimmer and
  Qwen3.8-Flash-Next. Each backend names its buffers (`retained_buffers()`).
  - Flash-Next's CPU-read PLE table stays in the page cache, outside the
    pulse.
  - DeepSeek V4 turns the keep-alive off when its opt-in
    `QWEN_DSV4_RESIDENCY_SET` already holds the weights.
- **Refused.** Qwen's serve backend does not name its weight buffers yet,
  and refuses a nonzero value. Its default weights are Metal-allocated
  copies, which stay wired anyway. The opt-in `QWEN_GGUF_NO_COPY` storage
  would need the hooks on both its request and native-inference paths.

### Durable snapshots (cross-restart warmth)

Qwen (3.5/3.6/3.8 dense and MoE) and DeepSeek V4 keep warm prefixes across
restarts in a disk tier under the RAM cache. Flash-Next, Muse Glimmer, K2 and
GLM-5.3-Flash have no durable tier: an explicit `--durable-snapshot-*` value is
refused at startup (`--durable-snapshot-dir off` is accepted), and the default
logs that the tier does not apply. Muse, K2 and GLM reuse their live session
instead of snapshots, so `--snapshot-cache-mib` only draws a warning there.

- **Flags.** `--durable-snapshot-dir PATH|off` (default
  `~/.cache/qwen-llm/serve-checkpoints`; each family uses its own subdirectory
  `qwen/` or `deepseek_v4/`, and so its own budget). `--durable-snapshot-max-mib
  auto|MIB`: `auto` = min(64 GiB, 10% of that volume's free space), `0` disables.
  `--durable-snapshot-min-tokens 1024`: shorter prefixes are neither written nor
  looked up. The resolved plan is logged as a `serve durable:` line at startup.
- **Stores.** Qwen uses `DurableCheckpointStore` (QWENCKP v1 blobs) and DS4
  `DeepSeekV4CheckpointStore`: staged temp file, fsync, decode-verify, flock,
  hard-link publish, corrupt-blob self-heal, and LRU-by-mtime eviction to the
  byte budget (a disk hit touches mtime). Disk ranking is plain LRU, not the
  RAM frecency. Publishers serialize on `writer.lock`; each sizes its record
  exactly, then requires record + 2 GiB free on the volume, evicting its
  oldest blobs if other activity filled the disk and refusing (logged) before
  writing if that is not enough. A publication that may not evict (Qwen's
  ranked shutdown writes) is refused instead, for the volume reserve and the
  budget alike, under the writer lock. A valid record already at the final
  key has its directory fsync'd before it is reported as existing, so one
  left by a failed post-link sync is never acknowledged as durable. Lookups never wait more than ~100 ms on a
  publisher (a busy store is a logged miss), skip and keep records they cannot
  use (`unusable_skipped`), and each publish sweeps one other model's
  directory for staging files a crash left behind.
- **Identity.** Records are keyed by the model's strong content identity (the
  CLI's `checkpoint_identity`: full ordered GGUF hash, or Hugging Face sidecar
  SHA-256s, cached under the store's `identity/`). It resolves on a background
  thread over a second open of the loaded files (checked to be the same inodes
  and timestamps). **The first start on a model hashes every shard** (seconds for
  a 16 GB 27B; roughly a minute or more for ~97 GB DS4, disk-bound) unless
  sidecars exist; later starts hit the identity cache. Until it resolves the
  durable tier is inactive; `serve durable: ... tier active after N ms` marks the
  switch. DS4 sessions bind a process-local stand-in identity until then; RAM
  hits captured under it are re-attributed to the strong identity afterwards.
- **Writes** all happen on one background thread with a byte-bounded queue
  (a quarter of the RAM budget, clamped to 1–16 GiB); one snapshot larger than
  that goes alone when the queue is idle (long-context sessions); otherwise a
  snapshot that does not fit is dropped with a once-only warning, and the
  request path never waits on encoding or fsync. Entries leaving RAM are
  queued as soon as a capture or lookup releases them.
  - *Qwen* (dense snapshots are GBs). Nothing is written per request.
    - **Idle publication:** once no request has arrived for
      `--durable-idle-publish-secs` (default 30, 0 = off; any request, even a
      failed one, restarts the clock), the latest completed request's
      transcript boundary is written.
      - The transcript boundary is what every later turn reuses. The
        completed boundary only extends when the next prompt re-renders the
        turn exactly, so after a restart just that turn's tokens re-prefill.
      - If the target left RAM, the ranked entries are tried best first and
        the first one not already on disk or in flight is written.
      - It happens once per idle period, only when the writer is empty, and
        only with memory headroom for the write's second copy (the staged
        decode check); otherwise it is deferred and logged.
      - Agent loops send requests seconds apart, so this is usually one
        write per human turn (1-9 GB each for long 27B sessions). A restart
        or crash after it keeps the session.
    - **Spill:** an entry is written when it leaves RAM through budget
      eviction or idle/age expiry, unless its record is known to be on disk
      (see below). It is queued only with memory headroom for its write's
      decode copy; otherwise it is dropped (`spill dropped
      reason=memory_headroom`) and its payload freed.
    - **Graceful shutdown** (SIGINT/SIGTERM): one shared budget,
      `--durable-shutdown-secs` (default 30; was 10).
      - Jobs already queued finish first, then the continuation target, then
        the other ranked entries, one write at a time.
      - A ranked write never evicts a record, the target's among them. The
        store refuses it under its writer lock, before staging, when it would
        fit only by eviction (store budget or volume free-space reserve), and
        it is logged as `no_room`.
      - Each write is logged as `serve durable: ... shutdown write
        role=target|ranked outcome=published|already_on_disk|failed|
        timed_out|no_room`. The summary reports `target=` and, checked
        again at the end, `target_on_disk=`.
      - A write already running is not cancelled at the deadline.
    - Writes are acknowledged by the writer before an entry is marked
      durable. Marking uses payload identity, so an equivalent replacement is
      never marked.
    - A write is skipped only when this process has validated the record:
      its own write was acknowledged, or it was promoted from disk. The
      record must also still be on disk as a regular file of the exact
      record size. Otherwise it is written, and the store itself validates
      or repairs any record already there.
    - A job already in flight is not queued again, matched by payload or by
      record name (length, mode, digest), even before the model identity
      resolves.
    - Memory:
      - Memory-pressure evictions are not spilled (that memory is needed
        now), and they skip entries also held by a write or restore in
        flight, because evicting those frees nothing.
      - Each queued write reserves one decoded copy (its snapshot size: the
        staged decode check, or validation of a record already at its key,
        in any integrity mode) from enqueue until the writer acknowledges
        it. Request admission (prefill and decode), snapshot capture,
        promotion, spills and idle publication all count it; spills and
        idle publication also require headroom for their own copy. While the
        copy is resident the memory signals count it too, which errs toward
        refusing.
    - Measured on 27B Q8 at 66K:
      - The idle write of the 66,110-token transcript boundary took 5.5 s,
        2.1 s of it for the staged decode check.
      - After SIGKILL, the restart restored it from disk and prefilled 152
        tokens: 13.0 s wall vs 381 s cold.
      - A later SIGTERM, in 11.9 s, wrote the new target and one ranked
        entry and reported the disk-promoted entry as `already_on_disk`
        (`target_on_disk=true persisted=3`).
      - Sessions above ~100K are not yet measured.
  - *DS4* (small snapshots): every captured prompt/completed boundary is written
    behind as it is captured; shutdown waits up to `--durable-shutdown-secs`
    for the queue.
- **Reads.** Before the RAM lookup, a disk record strictly longer than the best
  RAM match is looked up (longest first; each candidate is gated by the strict
  RAM budget and the capture memory admission before it is read), decoded, and
  promoted into the RAM cache; the request then restores it like any RAM hit.
  Restored tokens count in `matched_tokens`/`cached_tokens` and `restore_ms`, and
  the `serve phases:` line reports `restore_source=ram|disk|none`.

The listener rejects every resolved non-loopback address and is bound before
the model loads, so an unresolvable or busy address fails startup immediately;
connections that arrive during the load wait in the backlog and are answered
once the accept loop starts. Request bodies are
limited to 16 MiB. The whole request read has a 30 s absolute deadline; the
socket read timeout is 35 s so it cannot preempt that mapping, and writes have a
30 s timeout. For an optional trace,
prepare a private directory rather than using a predictable shared `/tmp` name:

```sh
trace_dir="${XDG_CACHE_HOME:-$HOME/.cache}/qwen-llm/traces"
install -d -m 700 "$trace_dir"
qwen serve -m MODEL --trace-sse "$trace_dir/serve-$(date +%Y%m%d-%H%M%S).jsonl"
```

- **Residency:** model loads once; the process is the warm tier. S0's F2
  finding makes this the TTFT mechanism (per-process paging floor is
  5–8 s on A3B even with checkpoint hits; `load_ms` does not cover
  first-touch). Warm prefixes survive a restart through the
  [durable tier](#durable-snapshots-cross-restart-warmth) for Qwen and DS4
  (closing k3 review, D3); the process remains the fastest tier. The finite
  default Qwen context ceiling is 262,144 tokens; an omitted limit sizes each
  request to need without making the ceiling unbounded. Muse keeps one fixed
  resident session and resets it between requests by default; its separate
  live-prefix opt-in reports reused tokens without snapshot restores.
  Its synthesized system prompt is stamped with the current
  UTC date for each request; when callers provide `instructions` or a system
  item, that explicit system text owns any date policy instead.
- **Speculative decode (`--drafter`, v0.77 DFlash):** a request
  speculates after a cold prefill or when a restored RAM-cache entry carries a
  compatible target-hidden capture tail. Missing or malformed tails fall back
  to serial decode while refreshing the tail for the next turn. Greedy requests
  use target-verified accept-prefix;
  sampled DFlash2 requests sample the selector's sparse top-16 distribution
  and use maximal coupling against the packed target distribution. The packed
  forward is numerically close to, but not bit-identical with, serial
  token-major arithmetic. The per-request
  `serve phases:` line reports `decode_path=dflash|serial`, with a
  `serve dflash:` line carrying acceptance and backoff counters.
  `QWEN_DFLASH_PREFIX_REPLAY=1` additionally enables a default-off,
  single-process experiment that verifies repeated exact-prompt completions
  before falling back to DFlash; sampled use requires two distinct-seed outputs
  with a 32-token consensus prefix. It is not a multi-tenant cache contract.
  `QWEN_DFLASH_OFF_CTX=<tokens>` overrides the default 16,384-token hard stop
  for explicit long-context canaries. Above the default boundary, admission
  backs off on low acceptance or dense exact fallback and prices recovery probes
  conservatively; the override is not a default-on long-context promotion.
  DFlash starts only after the required full-prompt or trailing SWA window has
  been captured and the target sequence is at prompt length. The old 25-token performance run is
  retracted: its short-prompt serial-tail path did not seed prompt hiddens, so it
  cannot support a DFlash performance or output-equivalence claim. The corrected
  path has since passed scoped GPU validation: release 27B prefill and
  packed-verify gates passed (24-token final-logit cosine 1.0, minimum hidden
  cosine 0.999998, and packed verify 16/16 argmax with minimum cosine 1.0), and a
  live cold 19-token Qwen3.8-27B Q8_0 + DFlash2 Q8_0 request logged
  `decode_path=dflash` and matched serial output (`orange`, EOS after two
  generated tokens). This is correctness evidence for that cell, not a revived
  performance claim or blanket output-equivalence claim. Optional DFlash
  admission prices prompt capture, drafter session, verify scratch, and
  layer-major scratch; allocation, capture, or seeding failure restarts or falls
  back to serial generation rather than rejecting an otherwise viable request.
- **Determinism scope (F7):** no serve surface promises temp-0 byte
  identity across differing checkpoint-restore topologies; transcripts
  are byte-deterministic conditional on restore partitioning.
- **Stats contracts (frozen):** the stderr `qwen_diag` line carries
  `version=serve_stats_v1`; the `x_qwen` response echo's field names
  `matched_tokens`, `restore_ms`, `prompt_tokens` are frozen;
  `usage.input_tokens_details.cached_tokens` reports tokens restored
  from checkpoints.
- **Serial generation, fail-fast admission.** `std::net`, one request in
  flight; the acceptor rejects other connections immediately with `503` and
  `Retry-After: 1`. HTTP/1.1 with
  `Connection: close`; hand-rolled request parse (loopback threat model;
  request bodies are `Content-Length` JSON).
- **Implemented families: Qwen3.5/3.6/3.8, Qwen3.8-Flash-Next, DeepSeek
  V4, Muse Glimmer, and K2 Horizon raw/verified native chat/tools.** DS4 runs its own session and snapshot stack
  (`serve/backend_ds4.rs`) with a startup-fixed forward budget and a
  serve-owned snapshot cache (`serve/snapshot_cache.rs`; DS4 has no
  engine-side RAM prefix cache). Flash-Next (`serve/backend_qwen4exp.rs`, since 2026-09-17) holds
  one text-session workspace sized at load and hands it back reset after
  every request: the Qwen3.8 contract (effort levels, thinking, tools).
  Since 2026-09-23 it reuses prefixes through the shared serve snapshot
  cache: a request restores the longest cached strictly-shorter token prefix
  into the reset workspace and prefills the rest, with the Qwen
  transcript-boundary split below. A snapshot holds only state later tokens
  read — 36 GDN conv+delta states, PLE conv and n-gram history, and per QSA
  layer its pending index-key block, `n/4` compressed index keys, and `n` F16
  K/V rows: `118,063,104 + 24,576·n + 3,072·⌊n/4⌋` bytes (≈113 MiB + 24.75
  KiB/token; ≈0.95 GB at 32K). `--snapshot-cache-mib` configures the Qwen,
  Flash-Next, and DS4 caches (default `auto`; see the cache policy above). Muse
  reuses its one live session's longest common token prefix by default (no
  snapshots; `QWEN_MUSE_PREFIX_REUSE=0` disables).
  K2 keeps one live session and reuses its longest common token prefix (see
  [K2 Horizon Raw Profile](#k2-horizon-raw-profile)); it ignores the snapshot
  budget and has no drafter. Its verified chat profile separates
  reasoning and final answers without inheriting another family's parser.
- **Stdout is never written.** All diagnostics via the existing stderr
  tracing surface; per-request `qwen_diag` stats line retained and
  extended with `matched_tokens` and `restore_ms` (the S2/S3 gates are
  defined on this log, never on "the session completed").
- **Opt-in asynchronous SSE trace:** `--trace-sse PATH` appends one JSON object per line
  for each `/v1/responses` request, plus each streamed response's heartbeat,
  event (including its exact JSON `data` payload), and terminal `[DONE]` marker.
  Every record has a `trace_request_id`, allocated before JSON decoding and shared
  by that request's records; it is separate from the response's protocol ID.
  It is disabled by default and may contain prompts, tool definitions, and
  generated text. The file is opened append-only with `O_NOFOLLOW|O_CLOEXEC`,
  using nonblocking open so a FIFO cannot hang startup; it must be a regular file
  owned by the current user and is forced to mode 0600.
  One background writer serves all request subscribers. Its queue holds at most
  eight records (not a byte budget); a full queue or writer failure disables
  tracing for all subscribers rather than blocking serving. Records retain their
  per-request order; records from different requests may interleave. Trace loss
  does not alter response bytes, and this optional log is not durable job history.
  Shutdown gives the writer 250 ms to drain, then detaches it so a stalled
  filesystem cannot hold process exit; queued trace events may be lost in that
  case.

## HTTP Ownership And Admission

Serving still admits one ordinary HTTP connection at a time. A CPU worker owns
request reading, family-profile parsing/rendering, output partitioning and response
writing. The resident backend remains on its original owner thread, including K2's
borrowed model/session. There is no additional service, worker-pool option or
multi-request queue. Busy connections still receive a pre-header 503; readiness
does not reopen until generation and response cleanup have both finished.

The prepared request and prompt are shared without deep copies. Generated bytes
cross a two-slot channel in chunks of at most 4096 bytes. Backpressure waits are
cancellation-aware, not scheduling-dependent immediate refusals. Each original
piece is acknowledged only after downstream processing completes, so the owner
cannot overlap its next model step with processing that piece. Nonstream collection
preserves its original per-piece allocation/grouping. Streaming deltas may be
split differently; concatenated text, UTF-8, reasoning/tools, usage and terminal
semantics are preserved. Exact delta boundaries are not a compatibility promise.

The incremental CPU allowance is the worker's configured 2 MiB stack plus 32 KiB
for channel/producer/consumer chunks and metadata. This is conservative, not a
total request/output memory budget or pre-spawn admission: the worker, parsed
request and channels already exist when generation admission runs. Existing
whole-response collection and record-count-only tracing retain their limitations.
Qwen includes the allowance alongside durable reservations in initial admission,
optional-tail fallback and snapshot-eviction retry. Muse and Flash-Next check CPU
headroom before binding resident runners. K2 also includes it in combined admission
when allocating a fresh/replacement session. DS4 checks before and after fresh
session construction, restoring residency on a returned post-construction refusal;
this does not price or qualify the construction peak. A reported zero process
budget retains the runtime's omitted-budget convention, not proof of headroom.

Worker exit signals cancellation independently of activity accounting. Owner work
retains activity until it unwinds, and each handled connection produces exactly
one completion callback. CPU preparation and response writing inhibit idle
publication. Stop/error paths wake socket and channel waits and join the HTTP
worker; after acceptance stops and activity settles, the owner runs the backend's
existing bounded shutdown. Synchronous generation observes process signals at its
cooperative checkpoints, including bridge waits. This does not imply preemption of
an in-flight Metal command. The optional trace writer retains its separate 250 ms
detach-on-stall policy.

The bridge has CPU protocol, ownership, cancellation and admission-policy coverage.
An isolated CPU subprocess also verifies real SIGTERM during observed bridge
backpressure while the client stays connected and unread: generation aborts,
one completion callback runs, workers settle, and the listener closes before
backend shutdown. The fixture checks the termination error and exits through the
test harness; it does not qualify the production CLI's signal-derived exit code.
Live Metal pressure, performance, cache continuity and model-backed signal/durable
shutdown remain pending. This earlier ordinary-bridge milestone does not establish
Lens qualification. The subsequent [durable baseline slice](LENS-WEB.md) adds
concurrent history/native routes and the capability-gated Bun browser client.

## K2 Horizon Raw Profile

K2 dense 7B supports completion-style raw text on `POST /v1/responses`, not a
new `/v1/completions` endpoint. Choose an unused loopback address:

The verified final artifact also supports [HTTP chat](#k2-horizon-verified-chat)
and local `qwen run --user/--messages`. Raw strings never opt into templating:
their EOS-only stop set and literal output contract below are unchanged.

```sh
qwen serve -m "$HOME/models/K2-Horizon-7B-Q8_0.gguf" \
  --addr 127.0.0.1:8795 --max-context-tokens 256 --max-tokens 8
```

Both limits must be explicit for resident memory planning and the response default.
Capacity must be positive and fit the checkpoint's declared context and actual
device/memory admission; the default output limit is 1..=capacity. There is no
K2-specific 256-forward/response or 7168-context cap. Drafters fail startup;
a snapshot budget is ignored with a stderr note. K2 config, runtime storage plan, native
tokenizer, and EOS metadata are checked before listener binding or Metal setup.
The normal production lease and memory gate apply; listener binding still
precedes weight loading so busy addresses fail cheaply.

```json
{
  "model": "K2-Horizon-7B-Q8_0",
  "input": "The capital of France is",
  "max_output_tokens": 8,
  "stream": false
}
```

The model ID is the filename stem, as returned by `GET /v1/models`. Input must be
a nonempty string. The family parser retains it explicitly and applies no chat
template; native tokenizer NFC normalization and special-token recognition still
apply. Native BOS insertion defaults on. For already serialized input, use
`"x_k2":{"add_special_tokens":false}`; there is no authored-token deduplication.
For example, that option with input
`"<|ifm|begin_of_text|>The capital of France is"` matches the plain example's IDs.

Accepted fields are `model`, `input`, `stream`, `max_output_tokens`, `temperature`,
`top_p`, `store`, `truncation`, `x_qwen`, and `x_k2` only:

- Omitted `max_output_tokens` uses the explicit startup default. The full
  `prompt_tokens + max_output_tokens - 1` budget must fit capacity; no truncation.
- Sampling defaults are temperature 0, top-p 1, top-k 0, min-p 0, seed 0.
  `x_qwen` accepts only `seed`, `top_k`, `min_p`, and `stats`.
- `x_k2` accepts only boolean `add_special_tokens`; other families reject that
  extension. If supplied, `store` must be false and `truncation` must be `"disabled"`.
- In raw mode, message/item arrays, token-ID arrays, instructions, tools, reasoning/history,
  previous-response controls, and unknown fields fail, including null-valued
  unsupported fields. Accepted fields cannot be null either. The whitelist uses
  the existing parsed JSON representation; it adds no duplicate-key guarantee.

EOS 1 is the only stopping token: sampled/countable, but never emitted or forwarded.
The final non-EOS budget token is emitted without an extra forward. JSON and SSE
both emit literal protocol text: think/tool/IFM marker strings are not stripped,
partitioned, or executed. Incremental UTF-8 assembly preserves split characters;
invalid bytes or a final incomplete sequence use replacement characters. This is
not a byte-preserving binary HTTP format. Budget exhaustion uses the ordinary
incomplete-response terminal semantics, not an EOS success.
An over-budget nonstream request returns HTTP 400. If SSE headers have already
been sent before backend tokenization, the same refusal is a `response.failed`
event after HTTP 200, with no generated text or request-session allocation.

The model and one capacity-sized session stay resident; each request gets a fresh
sampler. Full attention truncates at any token, so each request (raw or chat)
reuses the longest common token prefix of the session's consumed history (prompt
plus forwarded outputs of the last successful request), capped to recompute the
final prompt row, rewinds the session in O(1), and prefills only the suffix. No
snapshots. History is taken before the session moves and republished only on
success, so any abort or error clears it and the retry starts from zero; a
poisoned session is discarded and recreated. A late HTTP write failure after
success keeps history, which is sound because it is exactly what the session
committed. Prefill appends run in spans that are whole multiples of the model's
prefill chunk (at least 64 tokens) with cancellation ticks between spans. By
default every admitted weight dtype (Q4_K/Q5_K/Q6_K/Q8_0/F16/BF16/F32) uses the
general batched prefill: 256-token commands with tiled mat-mat projections and
row-parallel norm/RoPE/KV store/causal attention (chunk shrinks under memory
admission; the load logs `k2 prefill: mode=... reason=...` on `qwen_diag`).
`QWEN_K2_PREFILL=q8_lcpp` selects the bitwise-serial-equal Q8 lcpp lineage
(32-token commands; load fails if the weights are not all Q8_0), and
`QWEN_K2_PREFILL=serial` one command per token. A fresh serve prefill runs
the same command partition as `qwen run`. Reused tokens are
reported as `cached_tokens`/`matched_tokens` with `restore_ms=0`.
`QWEN_K2_PREFIX_REUSE=0` (or `false`/`no`) disables reuse. Warm and fresh
prefill are token-identical but can differ numerically (different chunk
partitions reorder tiled projection accumulation), so this carries no bitwise
or sampled-exact claim. The borrowing
backend stays on the accept-loop thread without self-referential/leaked model
storage. `x_qwen.stats` is opt-in; the response echo has no tools/reasoning and
disables parallel tool calls.

Validation covers CPU wire/UTF-8/EOS controls and actual-Q8 borrowed-backend
JSON/SSE, BOS, abort isolation, and requested-capacity boundaries on ephemeral sockets.
Admission bounds are distinct from the lengths covered by numerical fixtures.
Numerical evidence covers the pinned final Q8_0-weight checkpoint with F16 KV on
M4 Max, including the frozen v2 holdout; it does not qualify F16 weights or every
compatible intermediate checkpoint. Serial online attention retains the full
history without a context-sized score buffer. Numerical fixtures above predate
the general batched prefill (GPU validation pending; gated by
`k2_general_batched_prefill_matches_serial_*`). Beyond 256 visible positions,
attention merges fixed-size
online summaries in registers to reduce accumulation error without dropping history.
Stored F16 KV is unchanged; compact KV is not enabled. At 524288 positions the
logical cache alone is 72 GiB; declared context does not promise that it fits a
particular device or is economical to run.
This is not sustained-service, full-context, tool, or cross-checkpoint
numerical qualification. Test reproduction is in
[`scripts/reference/k2/README.md`](../scripts/reference/k2/README.md); development
evidence remains in `K2-HORIZON-PLAN.md` and `K2-HORIZON-REVIEW.md`.

## K2 Horizon Verified Chat

For verified final Q8_0 and standard Q4_K_M artifacts, array `input` selects the pinned IFM
renderer; string `input` always remains raw. Startup verifies retained checkpoint
bytes once, before accepting requests, and owns an opaque chat capability. An
unverified artifact or verification failure emits a diagnostic and remains raw-only.
No request field can authorize chat or trigger rehashing. Profile details and
provenance limits are documented in `CLI-UX.md#k2-horizon-verified-chat`.

```json
{
  "model": "K2-Horizon-7B-Q8_0",
  "input": [{"role":"user","content":"What is 2+2? Answer briefly."}],
  "reasoning": {"effort":"low"},
  "max_output_tokens": 128,
  "stream": true
}
```

Use adequate server capacity and output defaults for reasoning workloads. The raw
profile's residency, sampling, cancellation, and no-truncation policies also apply.
Chat additionally accepts string `instructions` and `reasoning` containing only
`effort` (`high` default, `medium`, or `low`). Native BOS is mandatory; false
`x_k2.add_special_tokens` is rejected. Chat stops at EOS 1 or `im_end` 250019.

History permits one leading system (or `instructions`, not both), user and
assistant messages, and must end in a user turn. Message content is a string or
`input_text`/`output_text` parts matching the role. An assistant's reasoning
item directly precedes it, with string content or `reasoning_text` parts; an
assistant without one renders with empty reasoning, like every family (see
"Missing reasoning is empty reasoning"; the native renderer and its upstream
oracle still require the explicit field, which serve supplies). Generic replayed reasoning uses IFM's canonical `reasoning`
alias (high/base history tag); the request effort controls only the new suffix.
Completed response items can be replayed verbatim, including string IDs, completed
status, empty reasoning `summary`, and empty output-text `annotations`. Nonempty
summaries/annotations and incomplete history are rejected rather than discarded.
Developer roles, multimodal data, unsupported fields and null controls are rejected
before transcript normalization. Nonempty tools select the extension below. K2
wire JSON rejects duplicate keys and nesting beyond 128; typed containers and
number lexemes are preserved, including Serde-internal-looking object keys. These
are parser safety rules, not inference-context limits; other families are unchanged.

JSON and SSE separate `reasoning` and final `message` items. Generation begins in
preopened reasoning; the first exact `</ifm|think>`, `</ifm|think_fast>` or
`</ifm|think_faster>` switches to final text. Effort is the requested prompt control,
not a promise that the emitted close matches it. An optional matching opener at
byte zero is removed. Tool-looking, Qwen and later IFM markers remain literal in
ordinary no-tools chat. Empty reasoning still emits a completed
reasoning item so history can be replayed. Budget exhaustion inside reasoning is
incomplete, never an answer; EOS before its close is a protocol failure. Abort
discards ambiguous buffered delimiter/UTF-8 bytes and never synthesizes completion.
Unlike vLLM's K2 parser, this no-tools subset does not use tool-call markers as
fallback boundaries or reinterpret unterminated reasoning as visible text.

Leased actual-Q8 and Q4_K_M checks cover all efforts, short incomplete reasoning, completed
low-effort answers, stop-aware raw-prefix parity, JSON/SSE equality, CLI answer
parity, and fresh sessions after cancellation. These are wiring/termination checks,
not reasoning-quality, sustained-service, tool, or full-context qualification.
Total output-token usage includes stop tokens; the existing shared
`reasoning_tokens: 0` detail is not a measured per-channel token count.

## K2 Horizon Tools

Verified final Q8_0/Q4_K_M chat accepts standard flat function definitions in
`tools`, `function_call` history with JSON-string `arguments`, and matching
`function_call_output` items. Tool outputs may be strings, objects or arrays;
objects/arrays render as native JSON data, not multimodal content. Every
pending ID needs exactly one result. Results may arrive out of order and render in original call
order. IDs remain wire metadata, not tokens inserted into the IFM prompt.
A call group replayed without reasoning opens an assistant turn with empty
reasoning; its parallel calls attach to it.

```json
{
  "model": "K2-Horizon-7B-Q4_K_M",
  "input": [{"role":"user","content":"Call lookup_code with key orbital."}],
  "tools": [{"type":"function","name":"lookup_code","parameters":{
    "type":"object","properties":{"key":{"type":"string"}},"required":["key"]
  }}],
  "reasoning": {"effort":"low"},
  "x_k2": {"tool_call_format":"json"},
  "max_output_tokens": 512
}
```

`x_k2.tool_presentation_format` accepts `markdown` (default), `xml`, or `json`;
`x_k2.tool_call_format` accepts `xml` (default), `json`, or `xml_typed`. The response
echo records these requested formats. Only omitted/`auto` tool choice and
omitted/`true` parallel calls are supported: the native template has no forced,
disabled, named or narrowed choice control. `strict: true` is rejected; schema
presentation is not constrained generation or complete argument validation.
Empty tools normalize to ordinary chat without tool-format echoes or marker parsing.

JSON and SSE publish completed `function_call` items with stable per-response call
IDs. Append the response's `output` items and caller-owned results to `input` for
continuation. qwen never runs the tool. Calls publish only after a whole terminal
block is valid; incomplete or malformed blocks never leak partial calls. A closed
valid call at the output budget may publish while the overall response stays
`incomplete`; EOS before closure fails. Cancellation drops pending calls.

XML is IFM's verbatim tagged text, not entity-escaped XML. Types must be unambiguous
under the definitions; use JSON for ambiguous values. Requested typed XML may
accept a fully valid untyped XML block when the model omits labels, but never
contradictory labels, mixed malformed structure, trailing text or ambiguous types.
The strict library dialect parser remains strict. No arbitrary token cap or
heuristic reasoning closure is introduced; buffering uses a checked byte bound
derived from output budget and native tokenizer piece size.

Actual Q4 checks exercise call -> caller-supplied result -> final answer in all
three requested formats through CLI, HTTP JSON and SSE. They demonstrate product
wiring, not universal tool reliability, schema compliance or answer quality.
Reproduction: `scripts/reference/k2/README.md`.

## GLM-5.3-Flash Verified Chat And Tools

```sh
qwen serve -m GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00004.gguf \
  --max-context-tokens 16384 --max-tokens 2048
```

Startup admits the artifact, requires its chat profile (embedded template
digest pinned to the upstream or unsloth GGUF template, every marker a single
released token, the released stops), and refuses to serve otherwise: there is
no raw serve lane. Before listening it admits the session against device
memory (a refusal names the `--max-context-tokens` that fits), prefetches the
retained windows, loads, and runs a two-token warm-up prefill so the first
request is not charged the weights' first GPU use (`serve startup:` logs
`load_ms` and `warm_up_ms`).

Requests use the shared Open Responses parser; the pinned
`glm53_flash_chat_v2` renderer (`crates/qwen-llm/src/glm5_next_chat.rs`)
renders them:

- `input` string is one user message; items are system/developer (or
  `instructions`), user and assistant messages with `reasoning` items,
  `function_call` and `function_call_output` items, ending in a user turn or
  tool results. Tools are below.
- `reasoning.effort` is `low`, `high` or `max` (default `max`, the
  template's); anything else is a 400 on `reasoning.effort`, where the
  template would silently use Max. The template always opens reasoning:
  `x_qwen.no_thinking: true` is refused, `x_qwen.thinking: true` is a no-op.
- History renders as generated. The template keeps past reasoning by
  default; an assistant without a reasoning item renders with empty
  reasoning, never through the template's inline `</think>` split.
  `x_qwen.history_thinking: "strip"` is the template's own `clear_thinking`
  (reasoning at or before the last user turn dropped). `template_style` is
  refused at startup and per request, as for K2 and Muse.
- Sampling defaults are the release `generation_config.json`: temperature
  1.0, top-p 0.95, top-k and min-p off, seed 42; `max_output_tokens` defaults
  to `--max-tokens` and must be within capacity.

Output partitions on the pre-opened reasoning: bytes are a `reasoning` item
until the first `</think>`, then the `message`. Stops are `<|endoftext|>`,
`<|user|>` and `<|observation|>`. Budget exhaustion inside reasoning is
incomplete; a stop before `</think>` is a protocol failure.

**Tools.** Function tools follow the template's own format; serve renders
and parses it, and never runs a tool.

- **Definitions.** They render in request order as one `<|system|>` tools
  block of `{"name", "description", "parameters"}` objects (Python `tojson`).
  `strict: true` is refused (no constrained decoding). So are
  `defer_loading: true` and any key other than `type`, `name`,
  `description`, `parameters` and `strict`: upstream would hide deferred
  functions and print other keys, which the shared parser does not keep.
  `tool_choice` is `auto` or `allowed_tools` (shared rules below).
  Generated calls to functions outside `allowed_tools`, and extra calls when
  `parallel_tool_calls` is false, are suppressed from the response, not
  prevented by decoding.
- **History.**
  - A replayed `function_call` becomes its assistant turn's
    `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value>…</tool_call>`:
    string values are written verbatim, others through `tojson`, and
    `arguments` must be a JSON object without duplicate keys (decoded
    losslessly).
  - `function_call_output` items render under one `<|observation|>` as
    `<tool_response>…</tool_response>`, in call order.
  - Each call needs exactly one output before the next turn.
  - Renders are byte-exact against the template oracle's tool cases
    (`crates/qwen-llm/tests/fixtures/glm53_chat_hf.json`).
- **Output.** After `</think>`, visible text is released until the first
  `<tool_call>`. The block is buffered within `max_output_tokens` × the
  longest decoded token × 3 bytes, and calls are published together as
  `function_call` items when the turn stops (`<|observation|>`).
  - Each argument's outer type comes from the declared schema (`type`,
    `enum`, `const`, `anyOf`/`oneOf`/`allOf`, local `$ref`). This is not
    full JSON Schema validation, and an unresolvable `$ref` refuses the call.
    - A schema that admits no string: the value must decode losslessly as
      JSON of an admitted, renderable type.
    - A schema that admits a string: the text is the string, unless it also
      decodes as another admitted type (`3` under `{}` or `["string",
      "integer"]`). The template writes both identically, so such a call is
      refused as ambiguous; declare a narrower schema.
  - Memory: the block is capped at its byte bound and admitted before it
    grows. Its capacity grows in 64 KiB steps, and each step must find
    process headroom for the block's whole outstanding peak at the new
    capacity, less what it already holds. That peak is 192 bytes per block
    byte: the text, the parsed values with their container overhead, and
    the serialized arguments kept and embedded in events and the response.
    The factor is measured against a counting allocator over
    container-heavy shapes; the worst, arrays nested to the decoder's
    128-level limit, costs about 152
    (`crates/qwen-llm/tests/glm53_tool_block_peak.rs`). Headroom is read
    afresh at every step and once more before the parse, so an earlier step
    reserves nothing. A failed check ends the turn with
    `memory_admission_denied` (503). Nothing is reserved upfront at the
    worst case (the longest piece is 512 bytes; the 99.9th percentile
    is 52).
  - Text after a call, an undeclared function, a repeated key or an
    ill-typed value is a server error, never text.
  - A block still open at the token limit is incomplete with no call.
  - A tool marker inside reasoning stays reasoning.
- **Live session.** A tool loop extends the live session: the model ends
  the call turn by sampling `<|observation|>`, which opens the results.
  Live, UD-IQ3_XXS, effort low
  (`docs/bench/2026-10-05-glm53-tools/serve-tool-loop.json`):
  - the first turn returned `get_weather {"city":"Paris","days":2}`;
  - the replayed call and output reused 218 of 236 tokens (401 ms
    prefill vs 1.5 s cold), and the answer used the output;
  - JSON and SSE behaved alike.
- **`qwen run --messages`.** A document with a nonempty `tools` list prints
  one Responses JSON object, `function_call` items included, instead of
  streaming text.

**Idle residency.** On by default (60 s; see
[Idle residency](#idle-residency-no-copy-families)). Without it, a request
after a pause pays ~1 s to re-wire GLM's 109.5 GiB: a 27-token prefill takes
1.6-1.7 s instead of 0.6 s. The window opens after the warm-up.

**Live session.** KDA recurrent state cannot rewind, so the one resident
session is reused only when a request's prompt strictly extends exactly the
tokens it consumed; anything else drops it and prefills a fresh session
(about 8 ms to allocate). A replayed conversation extends: the model ends a
turn by sampling `<|user|>`, the next turn's opener, and the renderer writes
history back byte-for-byte when the client replays the reasoning and answer
items. Re-tokenization that differs from the sampled tokens, a changed
effort or system message, `strip`, or whitespace the template strips from an
answer all fall back to a fresh session. Every client-caused refusal
(budget, capacity, input) precedes the session, so the cache survives it. A
cancellation during prefill keeps the committed chunks (a retry of the same
prompt resumes from them); an abort during decode or any engine failure
clears the session. `QWEN_GLM_PREFIX_REUSE=0` disables reuse.

Checked under `MTL_DEBUG_LAYER=1` on UD-IQ3_XXS
(`serve::backend_glm5_next::tests::gpu_live_session_extends_resumes_and_resets`):
cold bytes equal the run lane's; a replayed turn reuses the whole history;
under the Exact packed lineage a warm continuation's output bytes, and the
logits at the join, equal a cold run's bit for bit; a two-chunk prompt
cancelled between chunks resumes with the cold run's bytes; aborts after
prefill, on the first piece and after several forwards clear the session and
their retries equal an uninterrupted run; over-capacity is refused with no
ticks and the history kept; JSON and SSE carry the reasoning item and the
answer. A new session is admitted together with the request's transport
allowance. These are wiring and cache-correctness checks, not
reasoning-quality or sustained-service qualification.

## Wire subset (Open Responses)

This section describes the Qwen/DeepSeek/Muse chat profiles; K2's narrower raw and
verified-chat contracts and GLM's text chat are specified separately above.

The parser, Qwen capability binding, and prompt renderer are one shared pure
module. `qwen-lens --open-responses FILE|-` uses that same path for offline
readout/intervention prompts and records renderer-authored role, reasoning, and
tool-channel spans. Lens does not emulate HTTP routing or response filtering:
its `--model` and sampler flags remain authoritative, so request sampling fields
and narrowed `allowed_tools` fail closed there.

`POST /v1/responses` accepting. For Qwen and DS4, omitted
`max_output_tokens` defaults to 65536 unless overridden with `--max-tokens` at
startup. Muse requires an explicit startup default:

- Supported non-null fields are type-checked strictly. Known standard controls
  outside this subset (including `max_tool_calls`, `text`, `metadata`, and
  `stream_options`) fail closed rather than being silently ignored. Other
  unknown fields are ignored; request, `reasoning`, and `x_qwen` scopes log each
  unknown name once, while input-item and tool-object extras are silently
  normalized away. Unknown item/content types fail closed.
- `model` — must equal the loaded model id; else `model_not_found`.
- `input` — string (one user message) or item array in the subset:
  `message` (roles `system`|`developer` (system-equivalent, documented
  mapping)|`user`|`assistant`, content string or `input_text` parts),
  `reasoning` (plain `content`; `encrypted_content` is unsupported).
  **Item-sequence validation,
  not turn grammar** (review defect 1: the spec's `input` is an item
  list; stock AI SDK traffic legally contains consecutive user
  messages and, in S2, interleaved tool items): system/developer/
  `instructions` at head only, `reasoning` items must immediately
  precede their assistant message or function call, and the final item must be
  a `user` message or `function_call_output`. Unknown item _types_ →
  `invalid_request`; unknown _fields_ are
  ignored (top-level request fields logged once per name; `id`/`status`
  on replayed input items accepted and ignored).
- `instructions` — optional system text (exclusive with a system item).
- `max_output_tokens`, `temperature`, `top_p` — standard.
- `reasoning.effort` — validated Qwen3.8 identities accept `none`, `low`,
  `medium`, and `xhigh`; absent defaults to `xhigh`. It is rejected on generic
  Qwen identities. DS4 applies its separate renderer rules. Muse accepts
  `low`, `medium`, `high`, and `xhigh`, defaulting to `high`.
- `x_qwen` extension object — `seed`, `top_k`, and `min_p` are generation
  controls for every served family. `no_thinking` renders the released
  preclosed suffix on any identified Qwen release (Qwen3.5/3.6/3.8), the same
  rule as `qwen run --no-thinking`; `thinking: true` requests the released
  `<think>\n` opener on templates whose default is no-thinking (Qwen3.5) and
  is a no-op where thinking is already the default. DS4 uses
  `reasoning.effort` instead, while Muse directs callers to effort `low`.
  `template_style: "house" | "upstream"` overrides the deployment's
  `--template-style` for this request (identified Qwen releases and DS4 only;
  refused elsewhere). `history_thinking: "preserve" | "strip"` overrides
  either style's history-reasoning rule on Qwen; DS4 refuses `strip`, and
  `preserve` matters there only under `upstream` in a thinking tier. This is
  a documented implementor extension.
- `stream` — SSE when true, single JSON response otherwise.
- `store` — `false`, `null`, or absent; `true` → `invalid_request`.
- `previous_response_id` — → error code `previous_response_not_found`.
- `truncation` — only `"disabled"` (default). The engine already fails
  closed on context overflow (S0 F3); serve maps that to the spec error
  instead of a process exit.
- `tools` — uniquely named function tools are supported on identified Qwen
  releases (Qwen3.5/3.6/3.8), DeepSeek V4 (DSML), and Muse Glimmer's ATEM
  protocol; known definition fields have strict types, and `strict:true` is
  rejected because schema enforcement is unsupported. Hosted tool types fail
  closed. Definitions render into the selected family tool block, byte-pinned
  to its renderer contract. An unidentified Qwen release refuses tools and
  replayed tool turns with code `tools_require_known_release` — the same
  family rule as `qwen run --messages`, advertised by `qwen info --json`
  under `capabilities.input.tools`.
  Function names must match `[A-Za-z0-9_.-]{1,64}` (the grammar `qwen run
  --messages` admits); replay `call_id` values are limited to 64 bytes.
- `tool_choice` — `"auto"` (default) or an `allowed_tools` object.
  `allowed_tools` accepts only mode `"auto"`, requires a non-empty unique list
  of declared function names, and defines the exact executable set. Narrowing
  is enforced as a hard constraint on emitted calls while
  leaving rendered bytes identical, so prompt prefixes and their
  checkpoints stay valid across tool-menu changes (the spec's
  cache-preserving intent, test-pinned). Enforcement is post-generation:
  a suppressed call has already consumed tokens and remains in the
  completed-turn checkpoint key, so that continuation will not hit.
  `required`, `none`, and forced-function are not implemented.
- Replayed `function_call_output` items must link by unique `call_id` to every
  call in the immediately preceding call batch; unknown, duplicate, or missing
  links fail closed. Argument delta/done events carry that same `call_id`.
- `/responses/compact` — not implemented (404); compaction is outside the
  S1–S4 arc and revisits with the WebSocket transport question.
- `x_qwen.stats: true` — echoes `{matched_tokens, restore_ms,
prompt_tokens}` into the response object, so thin clients (the game)
  get per-request checkpoint stats without correlating server stderr
  (review R4). `usage` is always populated (agent clients budget on
  it).

Provenance notes: `top_k`/`min_p` pass through normal sampling semantics;
`no_thinking` is identity-gated to the validated `qwen run` surface;
`stream: false` exists for the stock provider's `doGenerate` path in S2,
not for S1's consumer.

Output items: `reasoning` (when the model emits thinking) then `message` with
`output_text`, followed by typed function calls when present. Qwen/DS4 use the
`</think>` seam; Muse parses exact `to=self`, `to=user`, and ATEM recipient
segments from generated bytes. Family parsers own all model syntax, so ATEM or
Qwen-looking text in another family's visible response cannot become a call.
Muse rejects undeclared recipients, malformed controls, invalid UTF-8, and tool
values that cannot be rendered back into ATEM history without changing
structure; partial controls and calls are discarded on truncation or abort.
Truncated thinking (S0 F4) yields `reasoning` with `status:
"incomplete"` and `response.status = "incomplete"` with
`incomplete_details.reason = "max_output_tokens"`.

Response envelopes truthfully echo normalized, validated `instructions`, `tools`,
`tool_choice`, `reasoning`, `parallel_tool_calls`, sampling values, and output
limit rather than emitting fixed placeholders.

Streaming events: `response.created`, `response.in_progress`,
`response.output_item.added`, `response.content_part.added`,
`response.reasoning.delta|done` (the gate-5 contract), plus one
`response.reasoning_text.delta` compatibility alias per reasoning delta for the
stock AI SDK provider, `response.output_text.delta|done`,
`response.content_part.done`, `response.output_item.done`,
`response.completed|incomplete|failed`, terminal `[DONE]`. Tool turns
add `function_call` output items with
`response.function_call_arguments.delta|done` (S2). **Every
event carries a monotonic `sequence_number`** (review defect 4 — the
conformance suite asserts ordering; retrofitting into a hand-rolled SSE
writer later costs more). One SSE comment heartbeat (`: ping`) is written
immediately, then idle heartbeats are attempted between prefill chunks; this is
not a `>=1 Hz` guarantee because a chunk may run longer than one second.
Parse, validation, model-id, and render failures occur before SSE and return an
HTTP error. Backend failures after SSE begins emit `response.failed` with the
spec error envelope; disconnect/write failures cannot emit a terminal event.

`GET /v1/models` returns the single loaded model (trivial, ships in S1
because client model-pickers probe it).

## Render and checkpoint contract

- Items→tokens rendering reuses the existing family renderers
  (messages.rs); the S1 tree adds golden byte fixtures asserting
  prefix-stability across turn append, including the future
  tool-continuation shape (designed now, wired in S2), and documenting
  the intentional divergence points per family.
- Reasoning items render into the think block for the preserve-mode
  families; assistant `message` content containing inline `<think>` is
  rejected (`invalid_request`) — reasoning travels as items, never
  inline (F1: verbatim echo is what makes preserve reuse exact, and
  items make echo structural).
- Qwen serve snapshots the **transcript boundary**: the last `<|im_start|>`,
  where the generation header begins. Qwen templates re-render a prior
  assistant turn differently from the generation suffix the model consumed
  (`<think>\n` versus a preclosed `<think>\n\n</think>`, or reasoning dropped by
  the client), so prompt-end and completed snapshots stop prefixing the next
  request one token after `<think>`, and a hybrid's recurrent state cannot be
  rewound to recover. Prefill stops at the boundary, captures, then finishes
  the header. Without it, thinking-mode chat and tool loops whose client drops
  reasoning reused 0 tokens on every turn. A captured transcript boundary
  replaces the prompt-end capture (an exact resend re-prefills only the header),
  and the transcript entry is pinned while the completed boundary is inserted
  (skipped if it cannot fit beside it). A resend or regeneration that restores
  exactly at the transcript boundary treats the restored entry the same way
  (phases: `transcript_restored=true`), so its captures cannot evict it.
  With a DFlash drafter loaded, the capture window starts a few columns early
  so the transcript snapshot also carries the drafter's window ending at the
  boundary, then rebases onto the full-prompt window; restored thinking turns
  keep speculating (27B + DFlash2, turn 2: 1559/1628 tokens reused, 1.83 s vs
  8.41 s before, greedy output unchanged). Flash-Next applies the same split,
  resend handling and pin. DS4 captures prompt and completed boundaries and
  skips a completed boundary with no transition or a truncation inside open
  reasoning. Eligible boundaries enter the **RAM** prefix cache (8–42 ms each
  per S0). Evidence: PERF-LOG 2026-09-23 transcript-boundary entry.
  Capture is best-effort and admitted against cache bytes plus Metal/process
  headroom; denial or failure is logged and generation continues. Both family
  caches enforce the configured byte budget.
  Prompt remainders (fresh, or after a restore) prefill one token at a time
  only up to 12 tokens on dense and 20 on MoE; anything longer is chunked.
  Both paths are correct for every model; the limit is where serial stops
  being cheaper (M4 Max, 2026-09-23: ~41-48 ms per serial token on 27B vs a
  ~390-640 ms chunk floor; ~10-11 ms vs ~170-300 ms on 35B-A3B). The old fixed
  48 made 20-47 token remainders 2-3x slower. Restored 7-32-token remainders
  on the 27B dense geometry (Qwen3.5/3.6/3.8, any weight dtype, greedy or
  sampled, no drafter) prefill in one packed block when its scratch price fits
  128 MiB, otherwise chunked; packed is bitwise equal to the chunked plan. The
  phases line reports `prefill_path=serial_tail|chunked|single_chunk|exact`
  and `prefill_chunk=N`. Chunked prefill on any Qwen MoE uses 2048-token
  chunks for prompts over 1,024 tokens (4096 for the pinned 122B-A10B profile)
  when memory admission passes; dense models use 1024. A declined larger chunk
  is logged with its reason (PERF-LOG 2026-09-24 auto prefill chunk entry).
  `QWEN_SERVE_FRESH_PACKED=1` additionally opts validated Qwen3.8/Q8 dense-27
  **fresh cache misses** of 19-48 prompt tokens into bounded packed prefill,
  greedy requests without a drafter only. Unset, `0` and invalid values disable
  this fresh path; Q4 is unsupported even with `1`. The same 128 MiB scratch cap
  and chunked fallback apply (the chunked path already removes most of the
  serial cost this opt-in was measured against). Q8 passes balanced endpoint gates, but automatic
  enablement remains held after an unpaired first-request outlier under substantial
  global compression. No first-request guarantee or sampled-distribution claim.
  Evidence: `docs/bench/2026-09-07-fresh-serving-http/RESULT.md`.
  Muse defaults to fresh packed prefill in superchunks of up to128 tokens
  with a16-token packing quantum, followed by a scalar remainder. By default
  it reuses the exact consumed-token prefix of the last
  completed backend generation in its resident session, without snapshot copies
  (`QWEN_MUSE_PREFIX_REUSE=0`, `false` or `no` disables). It always
  recomputes at least the final prompt row, reports only actually reused tokens
  as cached/matched, and keeps `restore_ms=0`. Capacity rejection preserves the
  prior history; history is taken before the session moves and republished only
  on success, so any abort or error clears it. A late transport/framing failure
  after backend success retains history, which is sound: it is exactly what the
  session consumed. GPU poison remains
  fail-stop. This is one serial resident history, not durable or cross-process
  caching; it does not accelerate fresh prompts or per-token decode. Evidence:
   `docs/bench/2026-09-08-muse-live-prefix/RESULT.md`.
   Muse Q8_0 on unified Apple M4 Max defaults to optimized matrix prefill and
   split decode. `QWEN_SERVE_MUSE_MATRIX_PREFILL=0` and
   `QWEN_SERVE_MUSE_SPLIT_DECODE=0` independently roll back to original math.
   Each accepts only unset/`0`/`1`, is resolved once at startup, and keeps the
   existing model-context/capacity and device admission checks. Unset or `1`
   permits only qualified execution; BF16 and other devices retain original math.
   Matrix prefill uses tiled full128 chunks and online packed remainders; scalar
   tails remain. Split decode admits528 KiB scratch and requires1024 visible KV
   positions. CLI-only math variables remain independent. Native ATEM, sampling,
   reasoning and terminal-token accounting do not change. Token-prefix identity
   is exact, but changed chunk boundaries and generated-versus-prompt history
   make optimized warm/reset arithmetic numerical, not bitwise or sampled-exact.
   Diagnostics report resolved options, planned tiled/online token counts, and
   actual generation transitions; planned counts are not dispatch measurements.
  Serve's durable tier (see
  [Durable snapshots](#durable-snapshots-cross-restart-warmth)) spills Qwen
  entries on eviction/expiry and shutdown rather than publishing per turn.
  `--durable-dual-publish` is
  designed but unbuilt (review R2: at q38's ~90 KB/token, dual durable publish
  of a 32k context is ~6 GB/turn — LRU churn that evicts the prefixes it
  is meant to protect, and publish time serializes the next request on a
  serial server). Live cache-pressure behavior remains part of the S3 rerun.
- **Replay-fidelity coverage (review R6):** render→items→render byte
  identity is asserted by unit tests (`render.rs` split/render inverse,
  `tool_parse.rs` `qwen36_raw_echo_identity`, `render_ds4.rs` preserved
  history round-trip) rather than by a JSON fixture case.
- Tool-continuation golden fixtures are **written before the renderer**
  (review R3), so the renderer is fit to the fixture, never the reverse.
- Release identity: the renderer contract follows the Qwen release
  (3.5/3.6/3.8) identified from the GGUF's name metadata (`general.name`,
  `basename`, `base_model.0.name`, `base_model.0.repo_url`, `license.link`)
  plus the Qwen3.x tokenizer gate (`gpt2`/`qwen35`/248320 tokens);
  `qwen4exp` architecture implies the Qwen3.8 contract; `deepseek4` implies
  the V4-Flash-0731 encoder contract. The GGUF's `tokenizer.chat_template`
  is never consulted: per supported release there is one correct contract
  — the full-capability one (tools, thinking, effort where the release has
  them) — and embedded-template variance across repacks reflects
  simplification, not intent (local DS4 repacks embed three different
  templates: Unsloth-patched 0731, upstream 0731, and the pre-0731
  original). Until 2026-09-17 the digest selected the release and refused
  unknown digests, which rejected derivatives whose template differed by a
  no-op. For an identified
  release the renderer follows the released Jinja byte for byte (oracle fixture
  `tests/fixtures/qwen36_chat_template_oracle_v1.json`): content is trimmed,
  the generation suffix is always `<think>\n` or the preclosed block, and
  preserved reasoning replays as `<think>\n{reasoning}\n</think>\n\n{content}`.
  Qwen3.6 thinks unless `x_qwen.no_thinking`; Qwen3.5's released default is
  no-thinking (its template has no `preserve_thinking` at all, so preserve is
  a serve policy there). Documented divergences: a second system item is
  rejected rather than merged, and Qwen3.5 preserves history reasoning in both
  thinking and no-thinking sessions (a serve policy; the template strips it
  before the last user query). Trimming
  follows Python `str.strip()`. An unidentified release (no version token in
  any name field, conflicting versions, or a foreign tokenizer) keeps the
  legacy generic ChatML contract (bare suffix, verbatim content) rather than
  guessing, logs a startup warning, and reports `capabilities.template`
  `{status: unknown, reason, fields_consulted}` in `qwen info --json`.
  `qwen run`, `qwen-lens`, `qwen-bench`, and `qwen-census` render through the
  same code.
- Tools on pinned templates render the way every released client feeds
  `tool | tojson`: the OpenAI-shaped `{"type": "function", "function": {...}}`
  object with Python's `", "`/`": "` separators (Transformers and llama.cpp
  agree), numbers in Python float repr (`10000000000.0`, `1e-07`), and
  replayed call arguments follow each template's own value rule: Qwen3.6
  passes non-container scalars through Jinja `string` (`True`/`None`),
  Qwen3.5 and Qwen3.8 pass every non-string through `tojson` (`true`/`null`).
  Every rule is pinned by the per-template jinja2 oracle fixtures. The released `last_query_index` rule applies:
  assistant turns after the final user query keep their think block (empty
  if no reasoning) even under strip. An unidentified release refuses tools
  (`tools_require_known_release`); the compact flat form
  `serve_tool_render_fixtures_v1.json` once froze for it was serve-invented
  and is superseded (2026-09-07). Function names on every lane match
  `[A-Za-z0-9_.-]{1,64}` — dotted names are released-protocol shapes
  (Qwen3.6 oracle `fs.list`, Muse ATEM namespaces).
- Preserve/strip rendering policy: preserve is the default for the validated
  Qwen3.6 identity (Amendment 1 of S0; economics measured in S0 G2), and strip remains available there via `x_qwen`. Validated Qwen3.8 and
  Flash-Next follow their released template, which preserves by default:
  plain and tool turns alike replay their reasoning (until 2026-09-25 plain
  turns always rendered the empty block and dropped it).
  DS4 rejects strip mode; under house style it renders each past turn as
  generated (see below). Muse rejects strip mode and preserves structured ATEM
  reasoning/tool history.
- **Missing reasoning: one predictable rendering per family.** Clients may
  replay history without its reasoning items (conforming Responses usage).
  An assistant turn (a message with its attached calls, or a call-only group)
  that arrives without one renders exactly as an explicit empty reasoning item
  would on Qwen3.5/3.6/3.8 (the empty think block), Muse (no ATEM `to=self`
  record) and K2 (the explicit empty field its native renderer requires). DS4
  house style is the exception: there a reasoning item's presence is the
  turn's provenance, so a turn without one renders as a chat turn,
  `</think>content`, which is also the release encoder's form for dropped
  reasoning. Strip applies to it as to any other turn. Only the unidentified
  generic Qwen contract keeps history verbatim. Thinking requests that relied
  on this log `serve: history_reasoning_missing=N` before generation (not DS4
  house style, where absence is provenance). This cannot restore reasoning the
  client discarded, so the transcript boundary above is still what keeps such
  loops warm.
- **Principle: serve does not rewrite past turns because of a new request's
  controls (2026-09-25).** The client owns the transcript; thinking mode and
  effort apply to the turn being generated. Where a release template
  re-renders history in the current mode, serve's `house` template style
  deliberately departs from it, and `--template-style upstream` (or
  `x_qwen.template_style: "upstream"`) follows the release rules instead.
  Effort instructions stay where each template puts them (the Qwen3.8 system
  block, DS4 after BOS), so changing effort still changes the prompt head;
  that is a deliberate edit, like editing the system prompt or forking a
  transcript, and serve renders it as sent.
  - `upstream` for Qwen restores each template's history-reasoning default:
    Qwen3.5 and 3.6 drop reasoning before the last user query, Qwen3.8 keeps
    it. Qwen history is otherwise already mode-independent in the releases.
  - `upstream` for DS4 is the release encoder within the supported subset:
    the whole transcript in the current tier; chat drops all past reasoning;
    thinking keeps it only with declared tools or an explicit
    `history_thinking: "preserve"`.
  - `upstream` is refused on the generic Qwen contract, Muse and K2 (at
    startup for the flag, per request for the override).
- **Qwen history renders as generated, independent of the generation mode
  (2026-09-25).** On identified Qwen releases (3.5/3.6/3.8, Flash-Next),
  thinking controls (`x_qwen.no_thinking`, `x_qwen.thinking`,
  `reasoning.effort`) change only the new turn's generation suffix. History
  keeps each turn's supplied reasoning in every mode. A turn generated without
  thinking has no reasoning item and replays as the empty block it consumed,
  byte-identical to the preclosed suffix. A mode switch therefore no longer
  changes earlier turns' bytes.
  - This removes only mode-induced changes. Replay is byte-identical for
    canonically formatted output (as the released templates render it).
    Other output still re-renders canonically: for example, `</think>`
    emitted without its preceding newline, surrounding whitespace, or
    reasoning truncated by the token limit. A completed snapshot hit also
    still needs matching token ids and a retained snapshot.
  - Previously a no-thinking request refused reasoning history
    (`x_qwen.no_thinking`) or re-rendered every prior turn with the preclosed
    block (Qwen3.8 `none`, Qwen3.5 default).
  - Only an explicit `x_qwen.history_thinking: "strip"` removes history
    reasoning, with the released last-query rule. Stripped turns before the
    last user query show no block even if they were generated without
    thinking, as in the released templates.
  - Qwen3.8 `low`/`xhigh` effort instructions lead the system block, so
    changing to or from those tiers still changes the prompt head.
  - The unidentified generic contract has no per-turn provenance. It keeps
    rendering history in the current mode and refuses nonempty reasoning
    history under `x_qwen.no_thinking` rather than drop it.
  - Pinned by the per-template Jinja oracle cases and
    `qwen_history_replays_as_generated_across_mode_switches` (serve
    partition, response items, admission and rendering, for every mode pair
    and for plain, call-only and prose-plus-call turns, with nonempty and
    immediately closed reasoning).
  - Serve replay, Qwen3.8-27B Q4_K_M, efforts medium, none, medium, none, with
    items replayed verbatim; each turn shows prompt tokens / matched tokens:
    before 1865/0, 1925/1860, 1987/1960, 2054/1982 (the turn-1 reasoning was
    dropped, and each switch re-prefilled the prior assistant turn); after
    1865/0, 1977/1944, 2039/2012, 2156/2134 (completed-snapshot hits). No
    no-thinking turn leaked a think tag.
- **DS4 history renders as generated under house style (2026-09-25).** The
  release encoder renders every past assistant transition in the current
  tier. House style renders each past turn from its own items: a reasoning
  item, even empty, means the turn thought (`<think>reasoning</think>`); none
  means it was a chat turn (`</think>`). Only the new turn's transition
  follows the current tier.
  - Serve emits an explicit empty reasoning item when a thinking turn closes
    its block immediately (every family; identified Qwen releases render it
    exactly like a missing item, the generic contract as the `<think></think>`
    it generated), and a DS4 chat generation never opens a reasoning item
    (its prompt already closed the block): any `<think>` it writes is
    content, admitted and replayed verbatim. Qwen binding still refuses
    inline `<think>` in assistant content; DS4 and Muse accept it as literal
    text. Replay is exact for clients that return these items, including
    empty ones.
  - Tier switches between `none` and `low` keep every earlier turn's bytes.
    `high` and `max` insert their effort prompt after BOS, so switching to or
    from them still changes the head.
  - Not guaranteed: legacy or client-omitted reasoning (renders as chat), a
    turn cut off inside its reasoning (replays closed), and a thinking turn
    that ends immediately after the opened `<think>` (no items, so the replay
    has consecutive user turns, which DS4 refuses).
- **A reasoning-only turn replays (Qwen, Flash-Next, DS4, Muse).** A thinking turn cut off
  by the token limit returns only an incomplete reasoning item. Replayed
  before the next user message, it is admitted as an assistant turn with
  empty visible text and its reasoning closed (Qwen
  `<think>\n{reasoning}\n</think>\n\n`, DS4 `<think>{reasoning}</think>`), a
  canonical completion rather than the unclosed bytes. Before 2026-09-26
  admission refused it, so the session could not continue. A trailing
  reasoning item, or one before a `function_call_output`, is still refused.
  K2 has its own parsers and still refuses both incomplete items and
  reasoning before a user message.
  - Pinned by `ds4_history_replays_as_generated_across_tier_switches` (tier
    chains through the partition, response items, admission and renderer)
    and `reasoning_item_presence_is_history_provenance`.
  - Serve replay, DS4 UD-IQ3_XXS, efforts low, none, low, none, items
    replayed verbatim, prompt / matched tokens per turn: before 1863/0,
    1903/0, 1977/1912, 1985/1927 (the switch to chat re-prefilled the whole
    history, 8.0 s at 1.9K tokens); after 1863/0, 1934/1912, 1976/1958,
    2058/2046 (completed-snapshot hits, 0.9-1.4 s prefill). Chat turns that
    followed visible past reasoning stayed chat: no reasoning item, no think
    tag in visible text, and correct answers.

## Cancellation

SIGINT/SIGTERM is checked around expensive startup phases, before admission,
while idle on a 100 ms admission poll, and at generation sink/chunk checkpoints.
The dedicated acceptor uses nonblocking accept with a 50 ms poll and is joined
during unwind, so a signal pending before the accept loop or arriving while idle
does not require a connection to wake the listener. Active-request shutdown is
bounded by the next checkpoint rather than immediate preemption.

For streaming requests, a disconnect is observed on an SSE write or a prefill
chunk heartbeat. Byte-oriented reasoning/tool partitioning can buffer token
boundaries that do not produce a write, so cancellation is write/chunk bounded,
not universally token- or time-bounded. The S1 cell measured 24 ms to the
following admission in its tested mid-decode case, but serve makes no universal
`<250 ms` claim. A non-stream response has no required write before completion;
graceful disconnect detection before that first write is inherently best-effort.

## S1 gate definitions (executed; see gate record)

1. **Thin client:** play.py rewritten against serve (direct HTTP + SSE,
   `x_qwen.seed`) at <150 lines. Artifact-named check: save-file bytes
   byte-identical to the CLI harness on a scripted 5-turn session plus
   one fork-by-truncation, temp 0, same seed; stderr explicitly outside
   the comparison.
2. **TTFT:** warm (resident, RAM-cache) turn-2 TTFT <150 ms at 8k
   context on Qwen3.6-35B-A3B; measured from request byte one to the
   first byte of the first `response.reasoning.delta` or
   `response.output_text.delta` event. Comments and lifecycle events
   excluded (review defect 5: heartbeats are SSE bytes; `response.created`
   fires at admission and measures nothing).
3. **Heartbeat:** first `: ping` within 1 s of request admission on a
   cold-prefill request (the client-timeout defense, gated separately).
4. **Prefix-stability goldens:** render fixtures pass — turn-append
   stability per family with documented divergence points,
   tool-continuation shape (fixture frozen before renderer), and the
   render→items→render replay-fidelity identity.
5. **Conformance:** Open Responses acceptance tests executed; every
   skipped test maps to a documented subset exclusion in this file; any
   failure or unlisted skip fails the gate.
6. **Cancellation:** mid-decode client abort → disconnect detected and
   next request's `response.created` emitted <250 ms after abort.
7. **No regression:** existing CLI surfaces untouched (parser tests,
   fixture tests, and a scripted legacy invocation matrix pass
   unchanged).

## Named risks

- Hand-rolled HTTP: scope strictly to loopback + Content-Length bodies +
  SSE writes; any request outside that shape gets a 4xx and a closed
  connection. If AI SDK fetch behavior (S2) demands more (keep-alive,
  chunked request bodies), prefer a minimal vendored parser over an
  async runtime — decision deferred to evidence.
- Dual-boundary capture doubles per-turn snapshot cost (S0: capture
  8–42 ms each, RAM); budget honestly in the stats line. Durable
  dual-publish doubles blob churn; F6 dedupe work may be pulled earlier
  if eviction pressure shows up in S2 agent loops.
- The serve stats surface is a new pinned contract; version it from day
  one (`serve_stats_v1`).

## Adversarial review record

k3 review (`ses_fe8ee25c6ffe`, 2026-08-18) returned REVISE with six
defects — item-list grammar vs turn grammar, `developer` role, unknown-
field policy, `sequence_number`, TTFT gate gameable by heartbeats, and
two interpretation-passable gates — all incorporated above, plus: dual
durable publish demoted to a default-off flag, replay-fidelity golden
pulled into S1, fixture-before-renderer ordering, and `x_qwen.stats`
response echo. Named riskiest assumption: stock-provider reasoning-item
replay fidelity (mitigated by the S1 golden; measured by the S2 gate's
`matched_tokens` definition).
