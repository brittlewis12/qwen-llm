# Maya real repository context - 2026-10-12

One 64K-capacity serving run completed a 41,420-token repository-source request
and its follow-up. The follow-up reused 43,032 tokens from the live session and
prefilled 106 tokens. Both responses ended normally, but independent review
found material factual errors in their operator guidance. This is measured live
reuse and completed generation, not a task-quality pass or evidence of restoring
a 41K-document snapshot. Raw answers are retained unchanged; there was no rerun.

Follow-up: [document checkpoints](../2026-10-12-glm-document-checkpoint/README.md)
now reuse a deep prefix on an edited final question. That separate comparison
and default promotion supersede the shared-only cache behavior described here,
not this run's measurements or answer review.

## Configuration and source packet

CLI build `ec96cc45`, driver/launch checkout `9cd2915e`; [report.json](report.json)
records binary, driver and helper SHA-256 identities separately. One resident
Maya-M IQ2_S model, default Fast Half, automatic workspace/cache selection,
loopback, context capacity 65,536, output limit 4,096 per request, low reasoning
effort. Responses report temperature 1.0 and top_p 0.95. Sampling was fresh,
without a seed or sampling override; this was not a paired deterministic test.
The only recorded inherited overrides were `QWEN_METAL_LEASE_WAIT=1` and
`RUST_LOG=info`; Metal validation was not requested.

The first user message contained actual source, followed by an operator decision
table question requiring paths and function/test citations. There were no tools,
canned results, repeated padding or external model calls. The second request
replayed the actual first reasoning and answer, then asked about the earlier
11,104-token snapshot/303-token prefill observation and prefill/decode cancellation.
That earlier observation was supplied only in the follow-up question.

[packet.txt](packet.txt) preserves the exact first-user packet;
[packet.json](packet.json) records selected ranges, full-file and selected-byte
hashes, instructions, questions and prior-report provenance. At capture:

| Source | Included lines |
| --- | ---: |
| `crates/qwen-cli/src/serve/backend_glm5_next.rs` | 1-1229, complete |
| `crates/qwen-cli/src/serve/backend_glm5_next/prefill_plan.rs` | 1-440, complete |
| `crates/qwen-cli/src/serve/backend_glm5_next/tests.rs` | 1-1699, complete |
| `crates/qwen-cli/src/serve/render_glm5_next.rs` | 1-650, complete |
| `crates/qwen-llm/src/glm5_next_metal/tests/reset.rs` | 1-244, complete |
| `docs/SERVE.md` | 774-1037, GLM section |

Selected source is 178,313 bytes / 4,526 lines; the packet with labels and question
is 179,515 bytes. Packet SHA-256 is
`e6e72a87bbb973c72e5e7dc11519f405296a790d0cda7f0d65b54503fb70a2d4`.
The 41,420-token figure is actual rendered prompt usage, not a tokenizer estimate.

## Measured serving behavior

| Metric | Repository request | Follow-up |
| --- | ---: | ---: |
| Input tokens | 41,420 | 43,138 |
| Cached/reused tokens | 0 | 43,032 |
| Prefilled tokens | 41,420 | 106 |
| Reuse source | none | live |
| First streamed delta, seconds | 187.2918 | 1.2314 |
| First answer delta, seconds | 188.5597 | 6.3004 |
| Client wall, seconds | 245.4655 | 43.0456 |
| Output tokens | 1,613 | 1,164 |
| Server prefill, milliseconds | 186,981.2 | 1,211.4 |
| Server decode, milliseconds | 58,175.0 | 41,815.5 |
| Server decode tokens/second | 27.73 | 27.84 |

Both responses have status `completed`, no incomplete details, and server stop
reason `eos`; neither was truncated. First delta includes reasoning and is not
first answer availability. Client wall runs through stream EOF, includes terminal
evidence writing, and excludes startup and the 0.5-second inter-request settle.
No request or teardown errors were recorded; the owned server exited 130 on
SIGINT. See [events.jsonl](events.jsonl) and [serve.log](serve.log).

Startup cold load was **68.265 s** and warm-up **36.573 s**, both before requests.
The log records three cold prefetch windows. These startup times are separate
from request latency; this packet does not establish why warm-up took that long.

Actual admitted workspace was **2,048 rows**, with session ledger
**3,548,266,496 bytes**. The automatic cache budget was **7,149,666,304 bytes**
(about 6.66 GiB), not allocated cache usage. Fast selected only the shared
instruction cut at **51 tokens** and captured one **153,264,332-byte** entry.
Capture cost was 15.8 ms; neither request restored a snapshot (`restore_ms=0`).
The large source document lived in user content after the shared cut. Its reuse
on the follow-up came from the retained live session, not that small snapshot.

## Answer review and corrected operator guidance

The answers organized much of the relevant code and mostly cited real names,
but these material errors prevent a quality-pass claim. Corrections below are
from `generate_with` in the captured
[backend source](../../../crates/qwen-cli/src/serve/backend_glm5_next.rs) and its
[tests](../../../crates/qwen-cli/src/serve/backend_glm5_next/tests.rs), not new GPU
results from this run.

- **Decode cancellation clears live state, not the snapshot cache.** The
  post-prefill abort and decode-failure branches drop `self.session`; they do
  not clear `self.cache`. Compatible snapshots captured earlier can remain
  available, subject to normal eviction/admission policy. A retry therefore
  need not be fully cold. The older `gpu_live_session_extends_resumes_and_resets`
  cold-retry cases do not establish a universal no-snapshot rule.
- **Fast canonicality checks the segment start, not its endpoint.** Before
  encoding a segment, the code applies `canonical &= position % rows == 0`.
  A segment starting on the canonical grid may end at an off-grid verified
  shared cut and publish that cut. Continuing a new segment from that off-grid
  position clears canonicality for later Fast capture. Prefill cancellation
  retains the committed prefix and that computed provenance; it is not a new
  endpoint-modulo test in the cancellation handler.
- **The tick-5 test stops at the first cut.** In
  `gpu_snapshots_continue_the_captured_trajectory`, it asserts
  `backend.history.len() == s1`: cancellation occurs before work in the next
  segment, not past the cut. The subsequent request can reuse `s1` live but
  cannot publish the later `s2` Fast cut from that off-grid segment start.
- **Exact can restore a transcript-specific suffix.** Fast uses only its own
  verified shared cut; Exact also captures at the generation-header boundary
  and selects the longest compatible prefix. The same snapshot test explicitly
  checks `cached == a_len - 2` for an Exact retry of A. The answers' claims that
  a restore never covers a conversation's unique suffix are false for Exact.
- **The 2,048-token cancellation proof has its own test.** It is
  `gpu_adaptive_prefill_cold_hit_and_continuation`, which asserts committed
  position/history 2,048 and canonical retry. The older
  `gpu_live_session_extends_resumes_and_resets` case resumes 512 tokens.

For the earlier agent screen specifically, Fast restored the shared 11,104-token
instructions/tools prefix and computed the remaining 303 conversation tokens.
That explains that observation without generalizing it into a restriction on
Exact snapshots. In this new run, `reuse_source=live` directly distinguishes the
43,032-token live continuation from the unchanged 51-token snapshot entry.

This is one freshly sampled source-understanding example, not broader quality
evidence. HTTP completion, plausible citations and long-context retention do not
make incorrect operator advice correct. Keep the raw answers and these limits.

## Follow-up

The bounded repository task and follow-up are complete. The separate
[document-checkpoint A/B and promotion](../2026-10-12-glm-document-checkpoint/README.md)
addressed branch reuse using this frozen packet. This original run measured no
branch and supports no branch-speedup claim. Its raw answers and factual errors
remain unchanged; no rerun to obtain a better answer is planned.
