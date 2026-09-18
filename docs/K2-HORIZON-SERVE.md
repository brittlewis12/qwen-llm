# K2 Horizon raw serving

Dense 7B K2 uses a bounded completion-style subset of the existing
`POST /v1/responses` endpoint. This is not chat/tool support, an OpenAI
`/v1/completions` endpoint, or a claim of full-context qualification.

Choose an unused loopback address; do not replace an existing server:

```sh
qwen serve -m "$HOME/models/K2-Horizon-7B-Q8_0.gguf" \
  --addr 127.0.0.1:8795 --max-context-tokens 32 --max-tokens 8 \
  --snapshot-cache-mib 0
```

Both token limits must be explicit at startup. Capacity is 1..=32 and must fit
the checkpoint context. The default output limit is 1..=capacity. A nonzero
snapshot budget or any drafter is rejected. Config, tensor/runtime plan, native
tokenizer, and exact EOS 1 metadata are checked before binding the listener or
acquiring Metal. Loopback address binding then precedes weight loading, so a busy
address fails cheaply. Normal production lease and memory admission remain active.

## Requests

```json
{
  "model": "K2-Horizon-7B-Q8_0",
  "input": "The capital of France is",
  "max_output_tokens": 8,
  "stream": false
}
```

The model ID is the loaded filename stem, also returned by `GET /v1/models`.
`input` must be a nonempty string and is passed to the native tokenizer without
any chat template. String contents are retained explicitly through request
normalization. Native NFC normalization and special-token recognition still apply.

Native BOS insertion is on by default, just as in raw CLI/benchmark input. For
already serialized input, opt out explicitly; there is no token-ID deduplication:

```json
{
  "model": "K2-Horizon-7B-Q8_0",
  "input": "<|ifm|begin_of_text|>The capital of France is",
  "x_k2": {"add_special_tokens": false},
  "max_output_tokens": 1,
  "stream": true
}
```

Accepted top-level fields are `model`, `input`, `stream`, `max_output_tokens`,
`temperature`, `top_p`, `store`, `truncation`, `x_qwen`, and `x_k2` only.

- `max_output_tokens` may be omitted to use the explicit startup default.
- Defaults are greedy: temperature 0, top-p 1, top-k 0, min-p 0, seed 0.
- `x_qwen` accepts only `seed`, `top_k`, `min_p`, and `stats`; accepted sampling
  controls are passed to the shared native sampler, not ignored.
- `x_k2` accepts only boolean `add_special_tokens`; other families reject this
  extension instead of silently treating it as a Qwen option.
- If supplied, `store` must be false and `truncation` must be `"disabled"`.
- Message/item arrays, token-ID arrays, instructions, tools, reasoning, history,
  previous-response controls, unknown fields, and unsupported extension keys fail.
  Unsupported fields fail even when null; accepted fields cannot be null either.

The family whitelist operates on the existing parsed JSON representation. It does
not introduce a new duplicate-key rejection guarantee for the HTTP JSON decoder.

After tokenization, the whole `prompt_tokens + max_output_tokens - 1` budget must
fit capacity. Requests are rejected rather than truncated. EOS 1 is the only
stopping token. It counts toward output usage but is not emitted or forwarded;
the final non-EOS budget token is emitted without an extra forward.

## Output And Lifetime

Both non-stream JSON and SSE use a literal-text output protocol. `<think>`, tool
syntax, and IFM marker text are not stripped, partitioned, or interpreted as
executable calls. Byte pieces pass through incremental UTF-8 assembly; genuinely
invalid bytes or a final incomplete sequence use the existing replacement-character
policy. This is raw protocol text, not a byte-preserving binary HTTP format.

The model remains resident and is borrowed by the serial backend. Each request
creates fresh KV and a fresh sampler; no snapshots, prefix reuse, reset, or hidden
conversation history exist. The backend stays on the accept-loop thread without
leaked/self-referential model storage. Prefill uses real single-token appends with
cancellation ticks between them. Every command finishes synchronously; an aborted
request drops its entire session, and the next request starts fresh.

`x_qwen.stats` is opt-in; cached/matched tokens are always zero. The response echo
has no tools/reasoning and disables parallel tool calls. Output-token exhaustion
uses the existing incomplete-response terminal semantics, not an EOS success.

## Evidence

CPU coverage includes field admission, exact raw rendering, nondefault sampling,
startup limits, EOS accounting, literal markers under all chunk sizes, split/
invalid UTF-8, and JSON/SSE transport. Shared Qwen/Muse/DeepSeek HTTP, parsing,
rendering, startup, and CLI regressions pass.

The opt-in actual-Q8 test borrows the production model, uses the production lease
and real wired-memory gate with API validation, and exercises both direct backend
calls and ephemeral loopback HTTP connections. It verifies raw-run output parity,
automatic/serialized BOS behavior, JSON/SSE output, stats gating, budget refusal,
and request isolation after prefill/generation aborts. It does not start a separate
long-running server or operate an existing service.

```sh
MTL_DEBUG_LAYER=1 K2_GGUF="$HOME/models/K2-Horizon-7B-Q8_0.gguf" \
  cargo test -p qwen-cli --bin qwen \
  serve::backend_k2::tests::gpu_borrowed_backend_matches_raw_run_and_discards_aborted_requests \
  -- --ignored --exact --nocapture --test-threads=1
```

This CLI test links `qwen-llm` as a non-test dependency, so `MetalContext::new`
already owns the production lease. Do not acquire an outer/second lease. This
differs from the library's isolated unit-test context behavior. Evidence remains
scoped to the final Q8 artifact and short contexts; no sustained-service,
full-context, chat/tool, or cross-checkpoint numerical qualification is implied.
