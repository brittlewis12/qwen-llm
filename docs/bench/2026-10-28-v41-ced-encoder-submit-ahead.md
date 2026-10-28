# V4.1 CED Encoder Submit-Ahead: +3.3%, 146 ms

2026-10-28. Source: `181ad0d1` on `perf/v41-engram-prefetch-main`, rebased
onto main `421f2054` (Engram producer + decoder submit-ahead + cross-chunk
Engram prefetch + encoder submit-ahead). Main has since moved (serve
batching), so integration needs a fresh rebase and source-identity
qualification. Artifact: GSQ-RCO IQ2_XS, 40 layers, 4,096-row chunks, M5
Ultra. Raw: `~/qwen-runs/2026-10-28-v41-encoder-pipeline/`.

## Decision

The encoder submit-ahead is exact and keeps the encoder queue fed across the
chunk boundary. The 8K pair is bitwise (all 14 endpoints share
`a2db19afd7c377c54160e90bdefc52dc04ab13a69b0f929abb34eab099da6edf`); the
same-binary CED A/B improves **3.81% / 166.6 ms** (1,803.8 -> 1,872.4 tok/s;
wall 4.5417 -> 4.3752 s). The private `current_pair` qualification passes at
N=8192: 257 positions, 257 full-vocabulary vectors per arm, 56 causal regions
per position, 256 teacher-forced steps, bitwise logits and causal-state
SHA256, 170.62 s. Candidate for promotion; the harness's
`performance_qualified` remains false by design, and the bounded fault matrix
below is still the promotion gate.

## What Changed

The corrected 8K stage timeline (hot run, stage sampling inflates GPU spans;
use it for ratios, not absolute rates) put the encoder boundary cost at
**130.7 ms of host preparation for two chunks** (nested Engram prepare
70.3 ms, late join 5.2 ms total) and **92/106 ms of completion tail per
encoder command**. The decoder already hid the same tail with its own
submit-ahead; the encoder did not.

Encoder chunk work is split into prepare, submit, and finish. Chunk N+1 is
prepared while chunk N's command runs, then submitted before chunk N's
completion is observed. The attention cursor keeps one older submitted
encoder intent in `inflight` and the newest in `pending`: a speculative begin
requires the previous encoder to be pending and contiguous, completion
requires the completed frontier and pops FIFO, and the error path waits the
outstanding command before unwinding. Command count (12), kernel arguments,
and arithmetic are unchanged. `QWEN_V41_CED_ENCODER_PIPELINE=0` restores the
serialized wait.

The pipeline keeps a second chunk scratch, publication buffer and command
reserve live, so the ledger charges
`priced_pipeline_chunk_bytes = chunk + publication + reserve` when the mode is
eligible, plus the second chunk's CPU upload staging
(`priced_pipeline_cpu_bytes`). The charge is gated on prefetch and a nonscalar
second chunk, so single-chunk and scalar-tail requests do not pay for overlap
they never perform. Observed incremental Metal is 11.202 GB against a priced
11.254 GB; the device ceiling is 240 GiB.

## Hardening

The checkpoint review found two P1s in the first version; both are fixed at
`e789bdec`, which re-passes the 8K pair bitwise at the same endpoint hash:

- Every committed encoder command is owned by a drop guard immediately after
  commit, before the prefetch worker is spawned, and before the older command
  is retired. The guard waits it on error or unwind, so neither a failed
  retirement nor a thread-spawn panic can leave an unwaited command or free
  the buffers it reads.
- The shared attention-status read is deferred to the final encoder chunk,
  after every writer has drained. Reading it while the newer command was still
  committing was unsynchronized and could misattribute a failure. Per-chunk
  FFN status remains owned by the chunk scratch, and the sticky status still
  rejects a failed run at the final checked completion.
- Speculative begin validates the new geometry and requires the previous
  intent to be fully encoded before moving it to the in-flight queue.
- A three-chunk cursor test covers re-entering the speculative state after
  retiring a front intent.

## Evidence

8K pair (`--runs 4`, P0 production timing, full-blocks-first):

| policy | samples (tok/s) | mean |
| --- | --- | ---: |
| Full | 1,121.03 / 1,119.31 / 1,119.70 / 1,119.85 | 1,120.0 |
| CED | 1,795.09 / 1,725.94 / 1,798.42 / 1,762.19 | **1,770.4** |

All 14 cases share the endpoint hash above.

Same-binary CED A/B, `O S S O`, two runs per arm, 180 s between arms:

| arm | samples (tok/s) | mean | wall |
| --- | --- | ---: | ---: |
| off | 1,802.72 / 1,793.13 | 1,797.9 | 4.5564 |
| on | 1,874.34 / 1,873.67 | **1,874.0** | 4.3714 |
| on | 1,874.21 / 1,867.36 | **1,870.8** | 4.3789 |
| off | 1,810.60 / 1,808.54 | 1,809.6 | 4.5270 |

The four off samples average 1,803.8 (wall 4.5417); the four on samples
average 1,872.4 (wall 4.3752). The machine state differs from earlier
packets (off arm was 1,774.98 in the prefetch packet), so only the
same-binary delta is claimed.

## Qualification

`181ad0d1`, private `current_vs_ced_paired_full_model` at N=8192,
`MTL_DEBUG_LAYER=1`, production self-lease, one test:

- `test result: ok. 1 passed; 0 failed` in 170.62 s.
- `complete: true`, `current_pair: true`, `performance_qualified: false`,
  `formal_quality_verdict: null`.
- Coverage: 257 state positions and 257 full-vocabulary vectors per arm,
  129,280 logits per vector, 56 causal regions per position, 256
  teacher-forced steps.
- Report: `~/qwen-runs/2026-10-28-v41-encoder-pipeline/qualify-8k-181ad0d1.json`.

## Limits

- One prompt, one artifact, one session. The pair's CED arm (1,799.5) is below
  the CED-only A/B arms (1,869.1) because the pair runs Full first; report
  both and use the A/B for the delta.
- The stage timeline that motivated the change was collected on a hot machine
  with stage sampling on; its GPU spans are inflated. Only the host/tail
  ratios are used, and the A/B is the timing claim.
- The pipeline doubles the encoder's peak chunk scratch (11.2 GB). The ledger
  prices it, but there is no automatic pressure downgrade yet; a
  near-capacity resident should set `QWEN_V41_CED_ENCODER_PIPELINE=0` until
  admission can choose the serialized path itself.
- Qualification covers the standard private witness only; no chunk-size/
  position boundary sweep, snapshot/growth, cancellation, or throughput
  qualification.
- Admission does not yet retry serialized CED automatically when the priced
  pipeline mode is refused under pressure; near-capacity residents must set
  `QWEN_V41_CED_ENCODER_PIPELINE=0`.
- A fault injection after the newer command commits (completion, FFN,
  attention, cancellation, panic) and a physical three-chunk prompt remain
  the minimum promotion tests.

## Next

- Adversarial review of the cursor/aliasing/lifetime argument and the memory
  ledger at this checkpoint.
- Decoder setup overlap (hide the last encoder tail), KVMP PV-axis composition,
  a single 8,192-row encoder chunk (engine cap/admission), and parallelism for
  the chunk-0 Engram dequant. Beyond those, the exact GPU floor is about
  4.0 s and the next frontier is reduced work, not more kernel squeezing.
