# Output-proportional CPU memory in serve (2026-10-06, map #14 survey)

This survey is from reading the code at `e0bce097`. Byte figures are
estimates, except where they cite a measurement. It maps every place
`qwen serve` holds CPU memory in proportion to a response's output, so
#14 can price them.

## Admission today

- **Only one output buffer is admitted against memory:** GLM's tool block
  (`partition_glm5_next.rs`: 64 KiB capacity steps, each checked for the
  block's whole outstanding peak at 256 bytes per block byte against fresh
  headroom, and again before the parse). Its peak was measured end to end
  (`docs/bench/2026-10-05-glm53-tools/` and
  `serve/http/tests/tool_block_memory_tests.rs`).
- **The transport reserve is fixed.** It is about 2 MiB of buffers, plus
  the control reserve: 16 MiB, or 512 MiB with Lens history. It does not
  read `max_output_tokens`.
  - Every resident backend checks it once per request
    (`admit_resident_transport`): Muse, K2, Qwen4exp, DS4 and GLM.
  - Ordinary Qwen combines it with its durable reserve.

## Retention by path

| Item | Where | Bytes per output byte (worst, estimated) | Bound | Admitted |
|---|---|---|---|---|
| Qwen/DS4 `tool_buffer` and its `serde_json` parse | `output_partition.rs` (`tool_buffer`), `open_responses/tool_parse.rs` | ~150+ for the block (parse dominated, as GLM measured) | output length only, no byte cap | no |
| K2 `ToolOutputStream` and parse | `k2_horizon_chat/tools/output.rs`; `render_k2/tools.rs` `byte_budget` | ~150+ for the block | `byte_budget`, checked only for overflow or zero | no |
| Muse tool-phase `pending` and calls | `partition_muse.rs` | up to ~150 for the block | output length only | no |
| GLM tool block, including non-streaming pieces | `partition_glm5_next.rs` | ≤ 256 (measured 205 worst) | `tool_byte_budget`, plus admission steps | **yes** |
| Non-streaming pieces (`CollectSink.pieces`, one allocation each) and per-piece `partition_events` | `http.rs` (`CollectSink`, non-stream arm of `handle_responses`) | ~40-64 for pieces of one byte (64 measured for GLM), plus about 100 per piece for events | output length | no |
| Terminal envelope and `serde_json::to_vec` body | `events.rs` (`finish`), `http.rs` (`write_json_response`) | ~2-7 (escaping; up to ~12 with growth) | output length | no |
| Streaming `ResponseStream` text, plus done/terminal copies | `events.rs` (`reasoning_text`, `visible_text`, `close_*`, `finish`) | ~5-6 | output length | no |
| SSE trace queue | `trace.rs` (8 records, not bytes) | ~8 for the terminal events' text | 8 records | no |
| Token vectors and backend histories | `ordinary_executor.rs`, `backend_{k2,glm5_next,muse}.rs` | 4-8 per token | capacity | no |

Bounded and priced elsewhere: completed-turn KV snapshots (budgeted, with
headroom admission), durable writes, native Lens (`CPU_UPPER_BYTES` and
readout pricing), and the transport channel (2 × 4 KiB chunks).

## Most important unpriced items

1. **Non-streaming collection, all families.** One allocation per piece,
   plus one event per piece. Both are held until the body is written.
2. **Qwen/DS4 `tool_buffer`.** No byte cap and no admission; the parse is
   into arbitrary-precision, order-preserving `serde_json` values.
3. **K2 `ToolOutputStream`.** Its budget is arithmetic only, and the
   parse is unpriced.
4. **Muse tool phase.** Unbounded `pending`, and calls accumulated until
   finish.
5. **Terminal-envelope copies and the trace queue.** The trace queue is
   bounded by record count, not bytes.

## Next steps for #14

- Price each family's tool buffer the way GLM does: admit before growth
  and price the whole outstanding peak. Measure each family end to end
  with the test allocator (`crates/qwen-cli/src/test_alloc.rs`).
- Replace non-streaming per-piece retention with a contiguous buffer plus
  boundaries, or partition during generation. That layout is pinned by
  `nonstream_fragments_preserve_one_allocation_per_original_piece`, so the
  pin's intent must be kept.
- Bound the trace queue by bytes.
- Separately: K2's non-strict `schema_kinds` has no visit budget, so
  shared-reference fans expand exponentially within its depth limit. It
  should be bounded without changing K2's pinned resolution convention.
