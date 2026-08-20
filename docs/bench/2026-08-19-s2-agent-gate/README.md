# S2 live gate: stock provider agent loop

Executed 2026-08-19. Server: `qwen serve` @ 937e712 (release),
Qwen3.6-35B-A3B-UD-Q4_K_S, `--max-tokens 1024`. Client:
`scripts/serve/provider-capture/agent_gate.ts` — a real AI SDK agent
loop (`generateText` + `stopWhen(stepCountIs(4))`, two executing tools)
through the **stock `@ai-sdk/open-responses@2.0.29` provider**, no
custom provider package, `store:false` throughout.

## Gate: ≥8/10 turns checkpoint-hit — PASS (10/10)

| request | prompt tok | matched | % | restore |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 338 | 338 | 100.0 | 3.5 ms |
| 2 | 386 | 386 | 100.0 | 3.8 ms |
| 3 | 431 | 407 | 94.4 | 3.7 ms |
| 4 | 485 | 459 | 94.6 | 3.9 ms |
| 5 | 517 | 501 | 96.9 | 3.8 ms |
| 6 | 561 | 543 | 96.8 | 4.0 ms |
| 7 | 600 | 575 | 96.8 | 4.0 ms |
| 8 | 653 | 629 | 96.3 | 4.0 ms |
| 9 | 688 | 667 | 96.9 | 4.3 ms |
| 10 | 736 | 714 | 97.0 | 4.5 ms |

Every request in the five-turn / ten-request loop hit a checkpoint;
tails are 16–26 tokens (the new user turn plus tool-result glue).
Restores stay ~4 ms. This is the agent-loop monotonic-prefix-append
economics predicted in the original jam, measured on real traffic.

## Functional result: 5/5 turns tool-called correctly

Every turn issued a real `function_call`, executed it client-side, and
grounded its answer in the returned data (`/var` → `log`, revenue 42,
notes summary, file count 2). Reasoning items, function_call items, and
function_call_output continuation all round-tripped through the stock
provider with no serve-side special-casing.

## Findings

1. **F-S2.6 (harness, not serve):** the AI SDK's
   `result.response.messages` carries only the *final* step's messages,
   so appending it across turns silently drops tool calls/results. With
   that history, the model saw a transcript of direct answers and
   stopped calling tools by turn 3 — it hallucinated `/var` contents
   rather than listing them. Accumulating `step.response.messages`
   across `result.steps` fixed it: 5/5 turns tool-called. Recorded
   because any client integrating this way (including future opencode
   config work) can silently degrade agent behavior with no server-side
   error — and the failure looks like a model quality problem.
2. **F-S2.7:** checkpoint hit rate is insensitive to tool-item replay
   shape here because the provider replays reasoning, calls, and outputs
   verbatim (F-S2.2/F-S2.3), so rendered history stays byte-prefix
   stable across the whole loop. The named R6 risk is now measured,
   not assumed.
3. **F-S2.8:** `usage.input_tokens_details.cached_tokens` equals
   `matched_tokens`, so clients see checkpoint reuse through the spec
   field without reading server logs.

## Status

The recorded provider gate passed. Since this capture, `allowed_tools` has
landed: mode is `auto` only, names must be a unique subset of declared function
tools, and the resulting set is the exact executable set without changing the
rendered tool block. `strict:true` remains unsupported. Replayed call batches
also enforce complete, unique `call_id` linkage to their outputs.

Production evidence is now available separately: real OpenCode session
`ses_fe307ea3effefOzYcDBgTbYiie` ran successfully for five hours overnight on
2026-08-20 (58 requests, reaching 133k prompt tokens). That demonstrates the
stock-provider configuration under sustained real use; it does not retroactively
turn S1's line-count/TTFT deviations or S3's deferred cells into passes.
