# Maya conversation reuse screen - 2026-10-12

Historical screen: the later [document-checkpoint promotion](../2026-10-12-glm-document-checkpoint/README.md)
adds deeper Fast snapshot reuse. The shared-only behavior and measurements below
describe this run and are retained unchanged.

Current Maya serving reuses live continuations and a shared instruction/tool
prefix across branches and conversations. A, B and the return to A produced
useful, fixture-correct answers. C remained unfinished; completed HTTP responses
do not establish completed tasks.

## Usable configuration

```sh
qwen serve -m /path/to/GLM-5.3-Flash-Maya-M-IQ2_S-00001-of-00003.gguf \
  --addr 127.0.0.1:8797 --max-context-tokens 32768 --max-tokens 2048
```

Use default Fast Half and automatic workspace/cache selection. This run admitted
2048 workspace rows. Send requests to `POST /v1/responses`; the screen used low
reasoning effort. Keep shared instructions and tool definitions stable, and
replay the full history, including reasoning, assistant text, function calls and
their results. Keep the server running to retain RAM snapshots. Fast return
restores the shared prefix and prefills the remaining conversation; it does not
restore a full conversation snapshot. See [serving details](../../SERVE.md#glm-53-flash-verified-chat-and-tools).

## Observed behavior

CLI `ec96cc45`, screen script `f36f52b1`; the report separately records the binary
SHA and launch checkout revision. The scripted client uses the opencode request
shape and synthetic `tally-canned-v2` tools. No opencode process or real tool
commands ran. The harness refresh passed 10 mock tests.

Cold startup load was **50.711 s**, followed by **1.129 s** warm-up, before the
server listened. These are separate from the request times below.

| Request | Reuse | Reused tokens | Prefilled tokens | First delta (s) |
| --- | --- | ---: | ---: | ---: |
| A1, first task | none | 0 | 11,129 | 50.016 |
| A2, tool continuation | live | 11,173 | 140 | 1.473 |
| B, edited task | snapshot | 11,104 | 24 | 0.622 |
| C1, new task | snapshot | 11,104 | 19 | 0.538 |
| A return | snapshot | 11,104 | 303 | 2.051 |

First delta includes reasoning or tool arguments, not necessarily answer text.
There was a 0.5 s settle between requests. This is one practical screen, not a
paired speedup or throughput measurement. Historical UD results used a different
binary before snapshots and are not a speedup baseline.

One 11,104-token shared entry occupied **272.36 MiB** (285,592,960 bytes).
The automatic **6.66 GiB** cache budget was a limit, not an allocation of that
amount. The same single entry served B, C and A return.

- A listed the fixture's source files and summarized `tally`; B listed its
  dependencies after one continuation. A return selected `src/main.rs` as the
  next useful file. These answers agree with the canned fixture.
- C asked about error handling. The fixture returned errors for unsupported
  `task`, an `rg` shell command and reading `src/main.rs`. After three
  continuations, C4 requested `cat /work/project/src/main.rs`; that call remained
  pending at the step limit, with no final answer. The failures are retained;
  this task is not a quality pass.
- All 11 generated argument objects parsed. All nine HTTP responses completed,
  with no request or teardown errors. The owned server exited 130 on SIGINT;
  the log's termination-signal message records that shutdown.

## Evidence and next step

[Report](report.json), [events](events.jsonl) and [server log](serve.log) retain
the original outcomes without a rerun. The prior cx scan found no credentials;
raw evidence includes local paths. The
[screen](../../../scripts/reference/glm53/serve_agent_reuse_screen.py) and
[mock tests](../../../scripts/reference/glm53/test_serve_agent_reuse_screen.py)
record timing, replay, partial failures and bounded owned-process teardown.

The normalized F32 HC candidate was parked and removed from active builds at
`ec96cc45`; source is archived at `08ec6267`. The
[64K-capacity real repository task and follow-up](../2026-10-12-glm-maya-repository-context/README.md)
are now complete: 41,420 initial prompt tokens, then 43,032 live-reused tokens
and 106 prefilled tokens. Both responses completed without truncation, but
review found material factual errors; this is not a task-quality pass. The new
run captured only a 51-token shared snapshot, so it demonstrates long live reuse,
not restoration of the document. No further kernel search follows from these
screens; a possible long-document branch lever still needs a concrete mechanism
and measured branch cost.
