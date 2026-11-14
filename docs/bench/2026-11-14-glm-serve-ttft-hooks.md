# GLM served TTFT component read

## Protocol and trace coverage

This is one instrumented arm on the current-main source plus shared lifecycle
hooks: source commit `a0b721678a3e7b1d4312e3595b37415eeee0ec29`, rebased from
sol's instrumentation commit `e5b163c8a` onto main `e70da3f8a`. The M5 clone's
main ref was stale, so the detached M5 worktree was checked out at the exact
rebased source commit. The GLM layer-slice setting was left unset; the server
log confirmed the 400 ms default and 500 ms minimum.

The release build and the three-agent greedy serve ran at nice 10 and nice 0,
respectively. One <=14-minute lease covered model load and the 180-second
window. It used the stock GLM-5.3 Flash UD-IQ3_XXS model, 48K warm roots,
`serve-plan-180.json` (seed `20261109`), and the metal-code corpus. Thermal
pressure settled at 0 for 60.8 seconds before load; it peaked at 1 in the
measurement window and 2 over the whole arm. There was no model resident at
load. Load averages were 22.3/24.5/26.0 before load and 16.5/39.0/35.8
afterward. Shader compilation overlapped the arm.

The trace contained 124 lifecycle records, joined to 119 calls: 3 roots,
91 short calls (90 completed), 16 about-1K calls (15 completed), and 9 long
calls (all completed). There were no missing joins, first-visible censored
calls, parse errors, or dropped spans. The analysis uses the shared harness's
nearest-rank p50/p95. Durations below are milliseconds; each cell is p50/p95.
`other` in the analyzer maps to long here: the observed row counts were
3,997–6,705.

## Component distributions

| Component | Root n=3 | Short n=91 | About 1K n=16 | Long n=9 |
|---|---:|---:|---:|---:|
| Client TTFT | 46,630/48,118 | 1,154/3,141 | 2,511/9,631 | 9,598/16,048 |
| Native server prefill | 46,542/48,052 | 381/1,529 | 1,529/1,678 | 7,197/7,365 |
| Outside native prefill | 88/126 | 773/2,264 | 945/8,082 | 2,355/8,851 |
| Accept to HTTP worker | 0.079/0.097 | 170/499 | 163/1,599 | 162/570 |
| Receive body | 0.082/0.248 | 0.116/0.259 | 0.098/0.140 | 0.104/0.180 |
| Body parse | 0.135/0.548 | 0.281/0.561 | 0.330/0.706 | 0.330/0.497 |
| Parsed to enqueue | 7.794/11.068 | 7.955/10.221 | 8.075/10.017 | 8.579/9.801 |
| Owner mailbox wait | 0.014/0.018 | 166/349 | 175/2,077 | 176/326 |
| Dequeue to submit | 0.002/0.002 | 0.000/0.002 | 0.000/0.001 | 0.000/0.086 |
| Slot queue wait | 0.007/0.007 | 0.002/0.006 | 0.002/0.057 | 0.004/0.023 |
| Admission span | 3.578/28.616 | 0.390/0.478 | 0.390/0.513 | 0.409/0.516 |
| Render | 0.022/0.061 | 0.053/0.130 | 0.050/0.183 | 0.065/0.223 |
| Tokenization total | 5.009/7.011 | 5.417/7.390 | 5.547/7.248 | 5.957/7.193 |
| Boundary tokenization | 0.078/0.084 | 0.063/0.082 | 0.063/0.084 | 0.064/0.109 |
| Prefix match | 0.007/0.012 | 0.040/0.060 | 0.041/0.054 | 0.043/0.068 |
| Snapshot prefix restore | 3.197/28.144 (n=2) | — (n=0) | — (n=0) | — (n=0) |
| Prefill-slice scheduler span | 46,592/48,104 | 370/626 | 1,529/1,678 | 7,320/7,453 |
| GLM layer-slice backend span | — (n=0) | 541/696 (n=29) | 1,547/1,678 (n=15) | 7,320/7,453 (n=9) |
| Admitted to first slice | 0.003/0.017 | 0.001/2,488 | 0.001/7,737 | 0.001/7,713 |
| Interslice wait | 0.224/0.336 | 0.000/44 | 73/195 | 490/634 |
| Finish-prefill span | 0.013/0.013 | 0.011/0.014 | 0.012/0.023 | 0.011/0.026 |
| Finish to first sample | 0.549/0.600 | 0.517/0.565 | 0.519/3.242 | 0.523/0.605 |
| First sample to raw piece | 0.001/0.002 | 0.001/0.002 | 0.001/0.002 | 0.001/0.002 |
| Raw piece to first visible flush | 0.501/0.876 | 426/1,065 | 423/5,065 | 911/5,037 |
| Visible flush to client | 0.434/0.583 | 1.526/2.230 | 1.558/2.942 | 1.506/2.179 |

In this run the first SSE flush preceded first visible content; the first SSE
can be a heartbeat or another non-visible event. From acceptance to first SSE
flush was root 5/8 ms, short 177/504 ms, about 1K 170/1,605 ms, and long
167/576 ms. The subsequent first-SSE to first-visible interval was root
46,625/48,112 ms, short 978/3,104 ms, about 1K 2,340/9,496 ms, and long
9,562/15,861 ms.

The `prefill-slice scheduler span` is the generic owner quantum. The additional
`GLM layer-slice backend span` brackets one `prefill_slice_run`, including
chunk preparation and checkpoint commit; it is not a GPU-only kernel timer.

## Peer-round waits and attribution

The shared round history shows peer rounds overlapping request-ready waits.
The counts below are p50/p95 peer rounds per request, followed by the total
overlapped peer-round time from the table above.

| Class | Peer decode rounds | Peer prefill rounds | Any peer-round overlap |
|---|---:|---:|---:|
| Root | 0/0 | 0/0 | 0/0 ms |
| Short | 2/9 | 0/9 | 380/2,720 ms |
| About 1K | 5/34 | 0/28 | 489/8,072 ms |
| Long | 29/55 | 0/28 | 911/8,700 ms |

For short appends, the largest single measured residual component is the
owner raw-piece to first-visible-flush gap: **426 ms p50**. Scheduler waits
overlapped peer rounds for **380 ms p50**, while first visible flush reached
the client in only **1.5 ms p50**. About-1K calls show the same two large
pieces: **423 ms** raw-to-visible and **489 ms** peer-overlapped ready wait.
Long calls rise to **911 ms** raw-to-visible and **911 ms** peer-overlapped
wait. These are per-call distributions; their percentiles should not be
summed as if they were one synthetic request.

GLM render, tokenization, prefix match, snapshot restore, and the admission
span are small in this run. Boundary tokenization is about 0.06–0.08 ms and
prefix matching about 0.04 ms for appends. No append restored a snapshot.
The shared intake path also has visible cost: accept-to-worker and owner
mailbox wait are each about 160–176 ms p50. Tokenization reuse alone has a
small ceiling here.

`send_to_accept` is about -1 ms for append classes because it subtracts client
wall-clock time from server monotonic-epoch time; treat that boundary as clock
alignment noise. The analyzer reports a few negative `last_slice_to_finish`
values (~0.002 ms) from nested-span ordering; the explicit finish-prefill
span above is the meaningful duration.

## Recommendation

The next measured lever should target the **shared first-piece-to-visible
event handoff**. Its short-append p50 (426 ms) is larger than any single
render, tokenize, prefix, admission, or first-sample phase; the actual
visible-flush-to-client delay is only about 1.5 ms. Keep sol's intake
serialization fix in the shared lane: this run also measures roughly
160–176 ms p50 in accept-to-worker and owner-mailbox wait. Re-measure the
first-piece handoff after that shared intake change before implementing a
GLM-only scheduling adjustment. No performance lever was changed in this
stage.

Run artifacts are under
`metagross:~/qwen-runs/2026-11-14-glm-serve-ttft-hooks/`; they contain the
release binary/hash, input hashes, raw serve logs, joined call traces, and
analysis summary.
