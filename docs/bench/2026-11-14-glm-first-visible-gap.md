# GLM first visible piece: tool-call parser hold

## Result

No exact parser change was made. The long raw-piece-to-first-visible interval
is GLM's validated tool-call path. GLM streams ordinary reasoning immediately;
tool-call arguments remain buffered until the terminal block is parsed and
validated. Moving argument deltas earlier would change failure behavior for
incomplete, malformed, undeclared, duplicate-key, schema-invalid, or suffixed
tool output. Matching token IDs does not remove that semantic boundary.

## Emitted pieces and hold rule

One representative 963-row append produced these first pieces (index, token
ID, decoded bytes):

| Index | Token ID | Piece |
|---:|---:|---|
| 0 | 154842 | `</think>` |
| 1 | 154843 | `<tool_call>` |
| 2–3 | 878, 2458 | `read`, `_file` |
| 4 | 154847 | `<arg_key>` |
| 5 | 2343 | `path` |
| 6 | 154848 | `</arg_key>` |
| 7 | 154849 | `<arg_value>` |
| 8 | 3548 | `src` |
| 9 | 14 | `/` |
| 10 | 8091 | `agent` |
| 11 | 16 | `1` |
| 12 | 25734 | `/ex` |
| 13 | 33863 | `cerpt` |
| 14 | 17 | `2` |
| 15 | 25559 | `.rs` |
| 16 | 154850 | `</arg_value>` |
| 17 | 154844 | `</tool_call>` |
| 18 | 154829 | empty decoded piece |

The first client event was `response.function_call_arguments.delta` with
`{"path":"src/agent1/excerpt2.rs"}`. The trace records 19 pieces through the
closing tool marker and stop; raw-piece to first-visible flush was 468.12 ms.
There was no whitespace or prose before the call. Both `</think>` and
`<tool_call>` were each one token in this trace, so textual-prefix ambiguity
was not the delay.

The prompt pre-opens reasoning. `PreopenedPartition` withholds only a suffix
that could still become `</think>` and releases safe reasoning bytes as they
arrive ([partition_preopened.rs](../../crates/qwen-cli/src/serve/partition_preopened.rs:88)).
After that close marker, GLM routes visible pieces through
`Glm5NextToolsPartition::route`; empty tool-stream output is intentionally not
published ([partition_glm5_next.rs](../../crates/qwen-cli/src/serve/partition_glm5_next.rs:84)).
`ToolOutputStream::push_visible` starts a block at `<tool_call>` and buffers
every later piece ([tool_output.rs](../../crates/qwen-llm/src/glm5_next_chat/tool_output.rs:279)).
At finish, the stream parses the whole block; parsing checks the declared
function, argument markers and keys, schema-compatible types, repeated keys,
and complete call structure before the call is returned
([tool_output.rs](../../crates/qwen-llm/src/glm5_next_chat/tool_output.rs:65),
[tool_output.rs](../../crates/qwen-llm/src/glm5_next_chat/tool_output.rs:335)).
GLM's call grammar also allows adjacent calls and rejects non-call suffixes,
so publishing a seemingly complete first call before the terminal boundary
could expose output from a turn that currently fails validation.

Ordinary reasoning does not share this hold: in the same run the first event
was `response.reasoning_text.delta` and its raw-to-visible gap was around 1 ms.
The residual is therefore an intentional tool-call validation delay, rather
than a thinking-prefix or transport delay. No parser change or CPU parser tests
were appropriate because no exact earlier-publication rule was found.

## One instrumented serve run

The run used the release binary from `fcb344ac1` with the default 400 ms GLM
slice, three greedy agents, a 48K warm-root plan (seed `20261109`), and a
60-second measured window. Piece capture was enabled by the new diagnostic
knob. The shared analyzer joined 52 calls, found no missing joins, parse
errors, or dropped spans, and marked two requests as censored before a visible
event. Nearest-rank p50/p95, milliseconds:

| Class | Calls | Client TTFT | Native prefill | Outside native | Raw→visible | Visible→client |
|---|---:|---:|---:|---:|---:|---:|
| Short (≤511 new rows) | 41 | 1,141.758 / 2,310.174 | 386.977 / 741.766 | 749.706 / 1,798.064 | 402.920 / 801.676 | 1.253 / 1.412 |
| About 1K (512–2,000 rows) | 6 | 2,721.311 / 6,210.332 | 1,350.290 / 1,595.988 | 1,371.021 / 4,614.344 | 1.067 / 887.050 | 1.231 / 1.641 |

First-event mix was 22 tool calls / 19 reasoning deltas for short, and 3 / 3
for about 1K. Conditional raw-to-visible was 466.549 / 950.346 ms for short
tool calls (n=22), versus 1.098 / 1.545 ms for reasoning (n=19). For about
1K, tool calls were 885.606 / 887.050 ms (n=3), versus 0.967 / 1.067 ms for
reasoning (n=3). The mixed about-1K p50 selects the reasoning half under
nearest-rank quantiles; the conditional values show the tool-path cost.

The pre-load and post-warm thermal settles both completed at pressure 0 for
at least 60 seconds, the measurement window remained at pressure 0, and the
arm maximum was 2 (allowed; no pressure-3 event). The harness's historical
`thermal_flagged` field is true at pressure 2. This was a single diagnostic
arm, not a latency qualification. `scripts/check-contracts.sh` and
`cargo check -p qwen-cli --bin qwen` passed on the instrumentation commit.

## Recommendation

Keep the parser unchanged. An exact earlier flush would require preserving the
current all-or-error tool-output semantics, which means waiting until the
terminal block can be validated. For the separate scheduler residual, reduce
time waiting in the shared work mailbox by allowing admission between GLM
layer slices; that does not remove the tool-call validation interval.

Run data and pinned binaries are under
`metagross:~/qwen-runs/2026-11-14-glm-first-visible-gap/`. Release binary SHA256:
`5fec04fbfdf4d7d056ca9a49f5c33b8eae8dc8f2e03201e0780ee54f1d1204d3`.
