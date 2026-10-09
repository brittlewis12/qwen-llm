# Request timing fields

What `qwen run` timing fields measure today. A field name can measure a
different interval in each execution path (lane). Values are host wall-clock
milliseconds unless stated otherwise. This records current behaviour before
any decision about comparability; changing a field's meaning is a request-stats
contract change. The common record builder writes schema version 2.

Families: Qwen (Qwen3.5/3.6/3.8 dense and MoE, single turn); Flash-Next
(Qwen3.8-Flash-Next); DS4 (DeepSeek V4); Muse (Muse Glimmer); K2 (Kimi K2
Horizon); GLM (GLM-5.3). “Prefill” means processing prompt tokens. “Transition”
means feeding a selected non-stop token back through the model to get the next
logits. “Resident execution” means the measured prefill-through-generator-return
phase in K2/GLM diagnostics. “Unclassified” is the lane wall minus the sum of
recorded phases.

Scope: this page covers single-turn `--request-stats-jsonl` records and stderr
stats, plus DS4 `--requests-jsonl` result/timing diagnostics. It excludes
Qwen's separate `--request-timings` format and the DFlash and prompt-lookup
decode paths; their timing semantics are not established here.

## Events and clocks

| Item | Current meaning |
|---|---|
| Initial GGUF open | Opening the model file initially; not timed. Later file opens are separate. |
| Input | In modern `qwen run`, Qwen, Flash-Next and DS4 acquire/render input before lane dispatch; cloning prepared text is timed. Legacy file/message acquisition occurs inside lane clocks. |
| Lane clock | Flash-Next, DS4 and Muse start after initial validation. K2 and GLM start at lane entry and include admission work. Qwen's request clock starts after model load. Batch has no matching single-request lane clock. |
| Device / weights / session | Metal setup, model-weight buffer setup, and per-request state allocation, where that family has those phases. Buffer setup is distinct from physical page residency. `weights_first_use` means first GPU use in the measured pass. |
| Tokenizer / encode | Tokenizer construction and contract validation are distinct from prompt encoding; family-specific fields below say which are included. |
| Prefill interval | Family-specific measured interval(s), detailed below; not a universal pair of forward boundaries. |
| First output / decode end | Qwen TTFT ends at the first non-stop-token stdout callback flush; if no callback runs, it ends at final-newline flush. Decode ends when its generator returns at stop or token limit. |
| Output / stats / record | Output work is conditional and lane-specific. A field's clock sample is taken at its implementation boundary; stats-line printing and record emission are outside the sampled interval unless stated below. |

## Stderr stats

| Field | Qwen | Flash-Next | DS4 | Muse | K2 | GLM |
|---|---|---|---|---|---|---|
| `tokenizer_ms` | — | tokenizer construction + encoding | construction only | tokenizer construction + tokenizer-contract validation + encoding | encoding | encoding + token-ID check |
| `load_ms` | runtime/model initialization through post-load checks and configuration; includes tokenizer construction on request 0 | device through weights and session | device through session; `prefetch_ms` is separately reported | device through weights and session | Metal context creation and model loading, plus session creation | weights plus session; no device setup or prefetch |
| `prefill_ms` | Sum of measured prefill calls; excludes restore/capture | Prompt processing through timing accounting and mode selection; packed-profile mode uses its separate third-pass sample | Prompt processing, including restore, memory reconciliation, logits copy, and optional debug output | Prompt-processing interval | Prompt-processing interval | Prompt-processing interval |
| `ttft_ms` | Request start to first written boundary above | — | — | — | — | — |
| `generation_ms` | — | Decode loop wall time | Decode loop wall time | Decode loop wall time | Decode loop wall time | Decode loop wall time |
| `total_ms` | — | Clock sample before stats-line printing | — | Clock sample before stats-line printing | Lane clock sample before stats-line printing | Lane clock sample after output assembly, before stats-line printing |

Other stderr fields:

- Qwen prints `stats:` per completed request. With warm-follow-up enabled it
  runs two requests and labels their lines `stats[0]:` and `stats[1]:`; the
  second has `load_ms=0`.
- Flash-Next adds `prefill_` and `decode_` versions of `encode_cpu_ms`,
  `completion_wait_ms`, `gpu_ms`, `gpu_samples`, and `outside_gpu_ms`.
  Encoding is host time from command preparation through encoder end.
  Completion wait includes command commit and completion. GPU time comes from
  Metal command-buffer timestamps and can overlap host completion wait.
  `gpu_samples` is valid timestamp samples / command count, not a time value.
  These values sum command measurements; decode command sums exclude sampling
  and output callbacks.
  `outside_gpu_ms = max(command wall sum - GPU sum, 0)` only when every command
  has a GPU sample; otherwise GPU and outside-GPU values are unavailable.
  `prefill_ms` is the measured prefill interval. Packed-profile mode reports
  its third warmed pass; layer-profile mode measures first use. Both warm-up
  passes, resets and profile reporting remain in the total clock.
- DS4 `prefetch_ms` is prefetch wall time. In single-turn DS4, explicit
  snapshot capture/publication occurs inside prefill. Durable prompt capture
  occurs inside prefill; durable publication, for prompt or completed-turn
  captures, occurs after decode. The initial durable-store probe is before
  load. `total_ms` is not printed for this lane.
- GLM `setup_prefetch_ms` is outside `load_ms` and `loaded_request_ms`;
  `loaded_request_ms` is the record's `timing_ms.total`.
- DS4 batch rejects `--request-stats-jsonl`. It still writes result JSONL;
  concurrency timing diagnostics use separate structured stderr records.
  Per-request `session_ms` is outside `prefill_ms`.

## Request-stats JSONL records

| Field | Qwen | Flash-Next, DS4, Muse | K2, GLM |
|---|---|---|---|
| `timing_ms.tokenization` | Encoding | Encoding only | Encoding (GLM also includes token-ID check) |
| `timing_ms.prefill` | Sum of measured prefill calls | Same family-specific prefill interval as above | Same family-specific prefill interval as above |
| `timing_ms.decode` | Decode loop wall | Decode loop wall | Decode loop wall |
| `timing_ms.total` | Encoding + request preparation + resident execution | Encoding + request preparation + resident execution | Sum of the same three non-overlapping phases. |
| `throughput_tps.prefill` / `.decode` | Present | Present | Present |

Schema v2 changes `timing_ms.total` to the sum of encoding, request
preparation, and resident execution for every lane. A v1 total from Qwen,
Flash-Next, DS4, or Muse used a different wall-clock calculation and is not
comparable to that lane's v2 total. K2 and GLM totals retain their v1 values.

The encoding span is the existing prompt encode interval. Qwen preparation is
the span covering its empty-prompt and capacity checks, stop-token loading,
sampling-option resolution, and sampler initialization. Flash-Next preparation
is one span covering forward-capacity, model geometry, token-range, stop-token,
decode-option, and profile-option checks; it ends before optional shard
prefetch and model loading. DS4 preparation sums two spans: required-forward
and prompt-token range checks, plus stop-token range checks. Its durable-store
admission and identity work are outside those spans. Muse preparation is one
span covering token-range, capacity, stop-token, and runtime-option checks.
Resident execution starts at prefill and ends immediately after the decode
generator returns, before trailing output, completed-checkpoint capture/publication,
stats output, or record writing. Qwen starts resident execution immediately
after request-state allocation returns; this excludes session allocation and
includes durable restore before its prefill timer starts. Qwen's warm-followup
request has its own three spans and total. Flash-Next packed-profile totals
include the reported third prefill pass and its generation only; both warm-up
passes are excluded.

`load_ms` and `transition_tps` are not common record fields. DS4 records them
under `diagnostics.deepseek_v4`; K2 and GLM include phase timings,
`end_to_end_lane_ms` and `unclassified_host_overhead_ms` under
`diagnostics.<family>.timing`. There is no common TTFT, load, or transition-rate
field. K2/GLM end-to-end lane wall and resident execution end immediately
after generation, before final newline or Responses assembly, stats printing,
and record emission. Their stderr `total_ms` is sampled later. For raw K2/GLM
output, a final newline is written only if bytes were written; DS4 has no
trailing newline after generation.

## Rates and overlaps

- Throughput is `1000 × tokens / milliseconds`. Prefill rates use the full
  prompt length, except concurrent DS4 uses `evaluated_prefill_tokens`.
  Decode counts generated tokens including a stop token.
- Decode wall time includes allocation, checks, selection, callbacks, and
  transitions. Batch callbacks buffer output bytes; K2/GLM Responses callbacks
  parse and buffer output, with stdout serialization afterward.
- `transition_tps` is printed for Qwen and DS4 stderr: transitions divided by
  transition time. A transition interval includes the whole transition
  callback (including validation/logits copying) and the following checkpoint,
  not only GPU forwards.
- Qwen request 0 tokenizer construction overlaps `load_ms`, TTFT and record
  total. DS4 prefetch is inside load; its tokenizer and record tokenization do
  not overlap. K2/GLM recorded phases do not overlap; prefill and generation
  lie within resident execution.
- For K2/GLM rates, the denominator is clamped to `f64::MIN_POSITIVE`, the
  smallest positive normal `f64`. Record serialization replaces negative or
  non-finite results with zero and preserves finite nonnegative results.

## Cache and first use

- A fresh process does not imply cold storage; there is no common cache-state
  field. GLM reports partial evidence in `prefetch.cold_windows` and
  `prefetch.bytes_read`, derived from sampled page residency.
- Flash-Next, DS4, Muse, K2 and GLM use no-copy weights; first GPU use is in
  prefill for ordinary single-turn execution. `weights_first_use` describes
  that measured pass, not disk-cache coldness. Flash-Next packed-profile mode
  reports a warmed third pass; layer profiling remains first-use.
- Qwen copies weights during load and request 0 also pays pipeline creation.
  The warm-follow-up request reuses tokenizer and pipelines.
- DS4 batch's first weight use occurs in its first executed prefill. With
  file-root sharing, this can be shared-prefix setup before a request's
  reported prefill.

## DS4 batch concurrency

- Paired requests prepare sequentially, then workers are released for concurrent
  decode; measured decode intervals can overlap. Short decodes need not overlap,
  and some requests run serially.
- Pair diagnostics report `model_prefill_ms` as the sum of lane prefill
  measurements; `prepare_ms` runs from before the first preparation message is
  built and sent until both workers report ready. `concurrent_generation_ms`
  runs from before generation signals through both worker joins. `pair_wall_ms`
  encloses pair execution, excluding earlier file-root setup and later output
  emission. Pair records include `file_root_restore_ms` and pair prefix,
  snapshot, restore and private-prefill timings. File-root `prefill_ms` and
  `snapshot_ms` are reported separately in `concurrency_file_root`.
  Aggregate rates are `1000 × summed count / concurrent_generation_ms`.
- Shared-source prefill sums prefix and suffix execution, excluding snapshot
  capture. Restored-request prefill measures suffix execution only, excluding
  restoration. Restored exact-logits reuse reports zero prefill and zero
  evaluated tokens; shared-source exact-logits reuse still counts prefix
  execution and evaluated tokens.
