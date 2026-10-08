# Using `qwen serve` from opencode

`qwen serve` exposes the Open Responses subset described in [SERVE.md](SERVE.md).
OpenCode can connect through its bundled `@ai-sdk/open-responses` provider.

## Start the server

```sh
qwen serve -m ~/models/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf \
  --addr 127.0.0.1:8737 --max-tokens 4096
```

Set `--max-context-tokens` for families other than ordinary Qwen. Set
`--max-tokens` for every family except DeepSeek V4. The family contracts and
defaults are in [SERVE.md](SERVE.md).

The server binds to loopback and runs one request at a time. A concurrent
request receives `503` with `Retry-After: 1`. Diagnostics go to stderr; use
`RUST_LOG=warn,qwen_diag=info` to include the `serve stats:` line.

## Configure the provider

In `opencode.json`:

```jsonc
{
  "provider": {
    "qwen-serve": {
      "npm": "@ai-sdk/open-responses",
      "name": "qwen serve (local)",
      "options": {
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

The model id must match the GGUF filename stem reported by `GET /v1/models`.
The endpoint is `POST /v1/responses`.

## Reasoning replay

The stock bundled `@ai-sdk/open-responses` 1.0.35 does not send prior reasoning
items back to the server. Families that keep past reasoning therefore render
that reasoning empty on later steps. A patched OpenCode build replays the items;
the CPU-only capture and patched result are in
[`docs/bench/2026-10-06-opencode-reasoning-replay/`](bench/2026-10-06-opencode-reasoning-replay/).

## Request fields and limits

`x_qwen.stats: true` adds `x_qwen.matched_tokens` and `x_qwen.restore_ms` to
the response. Without it, restored tokens are reported as
`usage.input_tokens_details.cached_tokens`. `x_qwen.seed` sets the sampling
seed. The supported request fields and family-specific limits are documented
in [SERVE.md](SERVE.md). DeepSeek V4 supports declared tools, renders them as
DSML calls, and parses generated calls into response `function_call` items.
