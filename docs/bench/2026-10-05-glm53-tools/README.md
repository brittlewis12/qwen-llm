# GLM-5.3-Flash native tools, live (2026-10-05)

Leverage map #11. UD-IQ3_XXS, M4 Max, effort low. Release builds: serve at
`2c9285ff`, the run lane from the `f00e5230` source.

- `serve-tool-loop.json`: `scripts/reference/serve_tool_loop_check.py`, two
  loops (non-streamed, then streamed). One `get_weather` tool. Turn 1 asks
  for the Paris forecast; turn 2 replays the reasoning, the call and an
  output.
  - **Turn 1:** `[reasoning, function_call get_weather
    {"city":"Paris","days":2}]` in both loops. The integer is typed by its
    schema.
  - **Turn 2:** `[reasoning, message]`, a forecast using the tool output.
  - **Reuse:** turn 2 reused 218 of 236 prompt tokens and prefilled 18
    (400-401 ms vs 1,509-1,531 ms cold).
  - **Streamed events** included `function_call_arguments.delta/done`.
- `run-lane-document.json` / `run-lane-response.json`: `qwen run --messages`
  with a tools document printed one Responses object: `[reasoning,
  function_call get_weather {"city":"Rome","days":2}]`, status completed.

Correctness of rendering and parsing is pinned on CPU:
- the jinja oracle, 79 cases, 56 byte-exact renders, with native token
  ids equal to HF's;
- the parser's every-byte-split, round-trip and schema-typing tests.

This packet shows the model emits the format and the loop closes through
both lanes. It is not a tool-use quality evaluation.
