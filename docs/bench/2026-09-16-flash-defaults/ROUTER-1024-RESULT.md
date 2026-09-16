# Exact N1024 Packed Router: KEEP

Admit exact1024 packed rows to the existing strict-order E8P32 router scope:
`{512,527,1024,2048}`, Apple M4 Max, H2560/E512/F32 router only. It is ON through
the existing default; `QWEN4EXP_PACKED_ROUTER_E8P32_STRICT=0` rolls back. No new
kernel, environment switch, allocation, planner change or speculative routing.
Other widths remain generic. This is prefill coverage, not a singleton decode gain.

## Why This Is New Leverage

Closed child timing and validation probes do not imply that every existing win is
reached by useful request shapes. Source inspection finds an unqualified1024-row
island between existing strict-router widths. The unchanged shader shares each
activation across8 output rows while retaining scalar K accumulation. N1024 is a
directly reachable one-command prompt. It requires qualification, not new shader work.

Fresh cx Luna01a0ac41-e142-7f72-a725-b980fbf06e2a challenges scope, fixture, state
coverage, census-free timing and precise wall gates, then confirms narrow promotion.
We correct its first planner example:3072 selected prompt tokens produce
2048+3+1021, not2048+1024. We do not pad a fixture or change the planner to manufacture
the target. The canonical natural-SSH fixture's first1024 IDs roundtrip through
the released tokenizer before execution; the original next4 IDs supply continuations.

## Qualification

All Metal work: production-exclusive benchmark lease, real wired-memory gate,
API validation and serial execution. Existing UD-Q3_K_XL model, no download.

- CPU exact-scope test checks all1..2049 widths and unsupported device/geometry/
  dtype; planner proves a direct1024 command and the real3072 shoulder split.
- Model-free N1024 differential PASS0.06s: complete router logits, top-k IDs and
  weights, shared scale, route counts and slots match bitwise; kernel/grid witness.
  Inputs are the existing deterministic finite fixture, not exhaustive values.
- Native packet uses SHA874537119c68f6c566c4288ba17c1099694416edb001c4003249570894438e97,
  first1024 tokens, next4 original teacher-forced IDs. A/B correctness compares
  full-vocabulary logits, hyper rows, all121 persistent tensors, QSA lengths and
  PLE history at prefill and EACH continuation step. All match bitwise.
- Prefill census proves exactly48 generic-to-strict substitutions and every other
  dispatch unchanged. Continuation census witnesses existing guarded top-k/HC and
  incumbent QSA; singleton algorithms are unaffected by the packed scope change.
- Same loaded model, session plan and empty checkpoint in every arm. Two fixed
  A/B correctness passes warm the paths; one measured prefill ABBA has no census
  or timestamp instrumentation. All timed endpoint/state replays match.

| Axis /1024-row prefill | A1 ms | B1 ms | B2 ms | A2 ms | Mean saved | Control spread |
|---|---:|---:|---:|---:|---:|---:|
| GPU |2107.862625|1794.884875|1797.552375|2107.145625|14.77034%|0.03402%|
| Executor wall |2117.348166|1805.355792|1807.597875|2117.032375|14.67574%|0.01492%|

Explicit verdict KEEP:>=1% mean GPU saving, positive BOTH GPU pairs,<=5% GPU/wall
control spread, candidate mean wall<=baseline with zero regression tolerance.
Native packet36.76s. Process success alone is not promotion authority; the harness
also permits logged HOLD outcomes. No timing retry or shape sweep was used.

## Resource Failure Preserved

First native invocation FAIL0.13s before any prefill or forward at the<1GiB snapshot
budget assertion. The initial design retained all five full baseline states.
The harness repair keeps the limit and every state comparison: retain prefill
state, persist each continuation state, then compare one saved baseline tensor at
a time. The bound is four snapshots plus retained logits. No numerical or timing
data preceded that repair; the only actual timing bracket is native02.
An earlier type-inference compile error also produced no execution evidence.

## Actual CLI Delivery

The roundtripped raw1024-token prompt runs with the strict flag UNSET and with
rollback0. Both statusok,16 outputs,15 transitions, token_limit (not EOS), one packed
prefill command, and identical stdout plus generated-token fingerprint:
`2d0306464915685e7be88853d00be95cdb8a51d04b787d36521966133b2319ac`.
The default logs strict routing active; rollback does not. Guarded top-k and HC
remain default on. No new known-answer claim is made for this partial manual-page
prompt, and the two CLI runs are delivery checks, not a second performance bracket.

## Evidence And Limits

Logs under `target/profiles/`:
- `2026-09-16-router-n1024-component.log`.
- `2026-09-16-router-n1024-native.log`: pre-execution memory-budget failure.
- `2026-09-16-router-n1024-native-02.log`: sole native KEEP packet.
- `qwen4exp-router-n1024-46237/`: full raw endpoints/state, per-step causal metadata,
  census, exact prompt text and ordinary timings, persisted before gates.
- `2026-09-16-router-n1024-{default,rollback}-cli.*`: real delivery evidence.

Baselineb57f6be2; qualified one-width implementation7f9b4821. This establishes a
bounded warm native prefill improvement at1024, not process-cold TTFT, general
request speedup, other widths, sampled equivalence or universal semantic quality.
No automatic promotion for1021,1023,1025,1536 or any interval follows. The previous
negative child/validation results and closed storage/attention/dtype lanes remain
valid. Further policy coverage needs its own source/reachability case and frozen
qualification; do not immediately launch another width sweep.
