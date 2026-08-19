# Using `qwen serve` from opencode

`qwen serve` speaks the Open Responses subset (docs/SERVE.md), so
opencode drives it through the **stock** `@ai-sdk/open-responses`
provider — no first-party provider package, no OpenAI-compatible shim.

## 1. Start the server

```sh
qwen serve -m ~/models/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf \
  --addr 127.0.0.1:8737 --max-tokens 4096
```

Notes:

- The model stays resident; the first request pays first-touch paging,
  later turns restore from checkpoints (~4 ms in the S2 agent gate).
- Serve holds the engine's Metal process lease, so `qwen-bench` runs and
  serve are mutually exclusive tenants. Use `QWEN_METAL_LEASE_WAIT=1` to
  queue politely behind a running bench.
- Diagnostics go to stderr; `RUST_LOG=warn,qwen_diag=info` keeps the
  per-request `serve stats: version=serve_stats_v1 …` line (the
  `matched_tokens` field is the checkpoint-hit metric).

## 2. Configure the provider

In `opencode.json` (project or global):

```jsonc
{
  "provider": {
    "qwen-serve": {
      "npm": "@ai-sdk/open-responses",
      "name": "qwen serve (local)",
      "options": {
        // createOpenResponses takes a full endpoint URL, not a baseURL.
        "url": "http://127.0.0.1:8737/v1/responses"
      },
      "models": {
        "Qwen3.6-35B-A3B-UD-Q4_K_S": {
          "name": "Qwen3.6 35B A3B (local)",
          "tool_call": true,
          "reasoning": true,
          "limit": { "context": 32768, "output": 4096 }
        }
      }
    }
  }
}
```

The model id must equal the GGUF file stem the server reports at
`GET /v1/models`; serve rejects mismatches with `model_not_found`.

`@ai-sdk/open-responses` is not in opencode's bundled provider map, so
it is installed on demand into opencode's provider cache the first time
the provider loads.

## 3. What works, and what to expect

Verified live (docs/bench/2026-08-19-s2-agent-gate/): multi-turn tool
loops through the stock provider, reasoning items round-tripping
verbatim, `store:false` throughout, and 10/10 requests hitting
checkpoints with 94–100 % of each prompt restored.

Current limits worth knowing:

- One request in flight (serial). opencode serializes per session, so
  this is normally invisible; a second concurrent session waits.
- Reasoning is plain `content` in S1/S2; `encrypted_content` opaque
  round-trip is S3.
- `tool_choice` supports `"auto"` and `allowed_tools` narrowing;
  `required` / `none` / forced-function are not implemented.
- No image input, no `/responses/compact`, no WebSocket transport.
- Qwen3.5/3.6-family models only; DeepSeek V4 serve support is S3.

## 4. Checking checkpoint reuse

Per response, with `x_qwen: {"stats": true}` the envelope carries
`x_qwen.matched_tokens` / `restore_ms`; without it,
`usage.input_tokens_details.cached_tokens` reports the same restored
token count through the spec field. Server-side, grep the stats line:

```sh
grep 'serve stats' server.log | tail
```
