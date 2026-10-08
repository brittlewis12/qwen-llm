# GLM-5.3-Flash serve reuse for an agent-shaped conversation (map #13, 2026-10-07)

Question: with the patched opencode client's replay shape, does each
continued agent step extend GLM's live session, and what do branches, new
sessions sharing the prefix, and returns to an earlier session cost?

Answer: in one scripted continued step, the request extends the live session
exactly (11,185 of 11,284 tokens reused; 1.25 s to the first token instead of
~60 s). Every request that does not strictly extend it replays the whole
~11K-token prefix of instructions and tool schemas at ~185 tok/s, about a
minute each. That exposes roughly a minute of potentially avoidable replay
per matching-prefix hit for a RAM snapshot at the system/tools boundary
(#15), less capture, restore and suffix costs.

## Method

`scripts/reference/glm53/serve_agent_reuse_screen.py` (no opencode process,
no external server): a `qwen serve` it launched (`/tmp/qwen_release_9d426a6e`,
built from `9d426a6e`; `--max-context-tokens 32768`, prefill rows 512, Fast
lineage, no API validation) and a scripted client sending
`scripts/reference/glm53/opencode-shape-v1.json` (the patched client's
instructions and 12 tool schemas, sanitized) with reasoning effort low,
replaying reasoning as `reasoning_text`, text as `output_text`, calls as
`function_call`, canned tool results. Weights warm (prefetch read 0 cold
windows).

## Results (`report.json`, `serve.log`)

| Request | Kind | Items sent (reasoning) | Prompt tokens | Reused | Prefilled | Prefill | First delta | Wall |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A1 | cold | 1 (0) | 11,129 | 0 | 11,129 | 60.1 s | 60.1 s | 62.1 s |
| A2 | continued tool step | 6 (1) | 11,284 | 11,185 | 99 | 1.23 s | 1.25 s | 3.5 s |
| B | branch (edited first turn) | 1 (0) | 11,128 | 0 | 11,128 | 61.1 s | 61.2 s | 62.4 s |
| C1 | new session, same prefix | 1 (0) | 11,123 | 0 | 11,123 | 61.3 s | 61.3 s | 66.4 s |
| A' | return to session A | 9 (2) | 11,366 | 0 | 11,366 | 62.8 s | 62.8 s | 66.1 s |

A1 called two tools; A2 answered, so session A had one continued step. The
report labels C1 `continued`; it is the new session (the label is fixed in
the script after this run). Decode ran at 28.5-29.3 tok/s.

## Reading

- The patched client's replay shape keeps a continued step on the live
  session (one scripted step with serve's own items; not an end-to-end run
  of the client).
- GLM reuses only exact extensions of one live session, so a branch, a new
  session, a subagent with the same instructions and tools, or switching
  back pays the full prefix every time. For this prompt shape that is ~61 s
  per request.
- Scope: one scripted conversation sizes cost; it is not a quality or
  default-policy decision. Effort low shortens decode, not prefill.
