# Using `qwen serve` from opencode

`qwen serve` speaks the Open Responses subset (docs/SERVE.md), so
opencode drives it through the **stock** `@ai-sdk/open-responses`
provider — no first-party provider package, no OpenAI-compatible shim.

## 1. Start the server

```sh
qwen serve -m ~/models/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf \
  --addr 127.0.0.1:8737 --max-tokens 4096 --snapshot-cache-mib 4096
```

Notes:

- The model stays resident; the first request pays first-touch paging,
  later turns restore from checkpoints (~4 ms in the S2 agent gate).
- Serve enforces loopback binding and one request in flight. A concurrent
  connection now receives fail-fast `503` plus `Retry-After: 1`; OpenCode may
  retry instead of waiting silently behind another session.
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

Production evidence is separate from that provider gate: real OpenCode session
`ses_fe307ea3effefOzYcDBgTbYiie` ran successfully for five hours overnight on
2026-08-20 (58 requests, reaching 133k prompt tokens). This proves sustained use
for that session, not every model-family or deferred live gate.

Current limits worth knowing:

- One request runs at a time. A second concurrent OpenCode session receives
  `503` and must retry.
- Reasoning round-trips as plain `content`; `encrypted_content` is not
  implemented or advertised.
- `tool_choice` supports `"auto"` and exact `allowed_tools` narrowing with
  mode `"auto"` only. Names must be declared; `strict:true`, `required`,
  `none`, and forced-function choices are unsupported.
- No image input, no `/responses/compact`, no WebSocket transport.
- Validated Qwen3.8 27B identities support text, `none|low|medium|xhigh`
  reasoning effort, no-thinking, and Qwen tool rendering. This is not a vision
  surface; unvalidated identities fail family-specific controls closed.
- DeepSeek V4 requires an explicit startup `--max-context-tokens`, supports
  chat plus `none|low|high|max` reasoning effort, and preserves reasoning
  history for non-`none` thinking tiers. It does not support tools,
  `x_qwen.no_thinking`, or DFlash. Its core
  S3 cells 1–5 passed; live cells 6–9 still require rerun.

## 4. Checking checkpoint reuse

Per response, with `x_qwen: {"stats": true}` the envelope carries
`x_qwen.matched_tokens` / `restore_ms`; without it,
`usage.input_tokens_details.cached_tokens` reports the same restored
token count through the spec field. Server-side, grep the stats line:

```sh
grep 'serve stats' server.log | tail
```
