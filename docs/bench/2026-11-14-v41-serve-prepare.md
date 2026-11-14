# V4.1 concurrent HTTP preparation and exact token-prefix reuse

Bounded stage began 2026-11-14 11:02:34 UTC; stop/report by 13:02:34 UTC. Branch `perf/serve-ttft-instrumentation`, starting tip `e5b163c8a`. This is one composed diagnostic run against the previous clean reference, not a paired promotion gate. Both candidates remain default-off with independent `=0` rollbacks. Checkpoint capture is unchanged.

## Serialization point and local fix

At baseline `e5b163c8a`, `crates/qwen-cli/src/serve/mod.rs:817` forwards an accepted socket as an `Incoming` event. The shared loop runs synchronous `backend.pump()` at `mod.rs:1173`, then handles that event and calls `Connection::start_batched` at `mod.rs:1206`. `transport.rs:498` delegates to the worker-spawning path. Therefore the HTTP thread does not exist while a peer's pump is inside its blocking prefill. There is no tokenizer/prefix-cache mutex held across that prefill and no async executor pool to enlarge. This is dispatch ordering on the inference thread.

`QWEN_SERVE_ASYNC_PREPARE=1` snapshots only immutable CPU request/profile/tokenizer state and the inference thread's wake handle before accepting. The accept thread starts a bounded HTTP worker immediately and sends the connection through the same arrival channel. GPU state never leaves the inference thread. Rendering, tokenization and token-prefix matching can proceed during a peer's native wait. Starting the native forward still requires the inference thread.

At most `max_active` eager workers exist; when their permits are occupied, sockets retain the existing bounded FIFO path. Existing active workers remain bounded by `max_active`, so the combined maximum is twice that count, also capped by accepted connection depth. Permits follow the connection lifetime and release on failure/shutdown. Each early caller conservatively adds `max_active * 128 MiB` to its existing CPU transport reserve for preparations not yet submitted to the scheduler; these are admission allowances, not allocations. Three 48K sessions retain the same engine representation. The test suite checks overlap while a fake inference pump is blocked, cap fallback preserving the original socket, and permit release on shutdown.

Mutable live-KV lookup/restore still consumes inference state in `backend_v41/reuse.rs:181`; its previous append lookup cost was about 0.006 ms. It is not the cause of handler-start delay and has no lock held across peer GPU work. The concurrent prefix matching added here is exact text/token-prefix matching. An asynchronous KV-state consume/restore would need a separate state handoff design; no GPU state race is introduced for a microsecond lookup.

## Exact tokenization reuse

`QWEN_SERVE_TOKEN_PREFIX_CACHE=1` retains the previous full tokenization for four conversation families (first-user-turn hash, as in checkpoint retention), at most 8 MiB of text/IDs/cut payload plus fixed metadata. Exact bytes, never that hash, authorize reuse. Encoding and text scans happen outside the short cache mutex. Concurrent old-entry snapshots are bounded by the bounded HTTP worker count. Oversized entries are not retained; eviction or a missing proof uses the incumbent full encode.

The released JoyAI tokenizer partitions text at explicit special tokens, then pretokenizes/BPE-encodes each ordinary fragment independently. The new `NativeTokenizer::encode_partitioned` records the byte/token offsets at those fragment boundaries without changing the incumbent `encode` implementation. For an extension or fork it finds the exact unchanged byte prefix and resumes only at a recorded boundary at least one maximum-special-token length before the first changed byte. This guard prevents a completed or longer special token from changing the earlier partition. Ordinary BPE/Unicode boundaries are never guessed. Records carry an immutable tokenizer-instance identity, so another vocabulary cannot reuse them.

A prefix ending at a recorded special-fragment boundary has exactly the independent prefix encoding. The V4.1 admission path can therefore get shared/generation cut IDs from the just-prepared full encoding, preserving the existing nonempty/length/`starts_with` checks and falling back to the independent encode at any other cut. No prompt IDs, CED cuts, sampler or numerical kernel is intentionally changed.

CPU property checks use recorded V4.1 primary plan sizes, actual repository source excerpts and reconstructed tool-call transcripts passed through the current V4.1 renderer. They compare full token IDs and generation cuts, with adversarial partial/overlapping special markers, BPE merges, forks, Unicode combining marks/ZWJ, whitespace and tokenizer identity changes. Synthetic native vocabularies include a real `a b -> ab` merge and overlapping control tokens. The M5 CPU witness also uses the released file-A tokenizer with the existing source corpus, comparing full IDs and prefix IDs before any served measurement.

## Frozen measurement

One composed arm, same production file-A IQ2_XS + streamed IQ4_NL Engram, release profile, greedy, three agents, seed 20261112, steady v2 plan, 49,152-token sequential disjoint warm roots with 32-token replies, 15-second ramp and 120-second window. Context 196,608; MAX_ACTIVE=4, SAMPLE_WORKERS=3, PREFILL_FAIR=1; both existing timing probes enabled. Both new knobs are 1. Nice 0 is checked on the actual server. Engram purge must reach zero resident pages. Thermal settle <=1 continuously 60 s; allow 2, abort 3. Each M5 build/model phase uses a waiting lease <=15 minutes, inherited FD through Python. Process disk bytes, load, compiler and thermal flags are retained.

Reference: the clean `e5b163c8a` record, measured Rust source `0a54d2316`, [previous decomposition](2026-11-14-serve-ttft-instrumentation.md). Compare per-class component distributions and common call labels/output bytes. Small-sample p95 and unpaired wall/throughput ratios are descriptive; a full paired gate is next stage if justified.

## Checkpoint async note

The current checkpoint reads a live GPU state that decode will mutate. Running its copy independently while advancing that state is unsafe without an explicit copy-before-write dependency or immutable/COW snapshot. The command queue could overlap independent copies only after establishing those dependencies. Publishing the first token before a final Ready checkpoint, then completing that checkpoint before the next state mutation, is a separate ordering candidate. Earlier semantic checkpoints cannot move across their state advance. No checkpoint code is changed here.

## Validation and results

CPU and M5 results pending. No promotion claim; keep both rollbacks.
