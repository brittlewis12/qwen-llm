# 2026-11-11 — GLM production layer-sliced prefill

Landing branch: `perf/glm-layer-sliced-prefill`. Production integration:
`dbff76d0b` (original benchmarked commit `39519c6eb`, rebased without
production-source changes onto `2aa654ed7`). **Default stays off:** the reverse
serve pair is throughput-neutral and cuts the maximum gap by 88%, but its
global gap p99 worsens. This extends the native design/pricing work in
`2026-11-11-layer-sliced-prefill.md`.

## Behavior and rollback

`QWEN_SERVE_GLM_LAYER_SLICE_MS` is a startup integer, 0..10000 milliseconds.
Zero uses the original whole-row-chunk forward and decode-percentage policy.
A positive value enables layer quanta only with a ready peer decoder, on the
supported GLM Fast/Half batching path. With no peer, an unstarted row chunk
uses the legacy call; an already parked cursor drains its remaining layers
in one command. Kernel selection and the original row partition never change.
The shipped default is **0**. Set **100** to opt into the tested policy. The
strict global-p99 ship gate below failed; no alternate metric or policy was
substituted.

The nominal 100ms plan seeds layer endpoints from the measured M5 row regimes,
rounding the slice count up to a power of two, capped by the layer count. That
reproduces 4/8/16 slices at 32/342/1509 rows. A GPU command slower than 1.5 times
the target reduces subsequent layer quanta, with at least one whole layer per
command. Whole-layer skew and cold host/residency cost are outside a strict
100ms bound.

A cursor belongs to one session. It holds the original tokens and next layer;
MLA ordinal is reconstructed from completed blocks. Token IDs, embedding and
sparse controls are staged only once. A slice completes its command before
returning. The appender stays poisoned/exclusive until every layer and the
head finish and normal route/selector validation succeeds. Only then is its
position advanced and its logits exposed. No extra Metal activation buffers
are allocated; the existing residual/workspace stays with that session.

## Scheduler and state interactions

- The shared Driver's optional sliced-prefill hook defaults to the old path
  for other families. One unfinished layer quantum forces exactly one peer
  round, then another prefill quantum. That round replaces percentage credit;
  it does not add a second post-chunk service budget.
- A parked appender has no decode cursor, so it cannot enter `advance_group`.
  Peer rounds use ordinary batching and normal sampling. Once all members are
  decoding, the default-on pipelined GLM group loop resumes unchanged. A lone
  peer uses the existing singleton path; batch refusal before mutation keeps
  the existing per-member fallback.
- A nested `GpuThreadQos` guard covers sliced prefill submission, completion
  and validation. The existing Driver guards cover peer decoding. Host-only
  tests check nesting and restoration on early return.
- Only a fully committed original row chunk invokes the existing checkpoint
  observer. Canonical/shared/trajectory namespaces, ladder targets and row
  cuts are unchanged. Partial layers cannot be snapshotted or reused.
- Cancellation between callbacks drops the partial session with its request;
  no partial recurrent state reaches the idle pool. A request-scoped accounting
  guard emits its partial-prefill record even when Driver cancellation happens
  before the next backend callback. Completed checkpoints remain valid.
- For sliced requests, `prefill_ms` charges active prefill submission/wait/
  validation time and excludes interleaved peer decode. Capture time remains
  separate. The knob-zero accounting stays unchanged. Completed requests also
  report `layer_slices` and `layer_gpu_ms` in their phases.

## Correctness and CPU validation

- All 23 Driver tests pass, including exactly one peer round, group-loop
  re-entry, mid-append disposal and a refused peer batch followed by singleton
  fallback. The original rollback tests remain unchanged.
- The serve suite passes 688 tests; its 42 pre-existing ignored GPU/fixture
  cases remain explicit. No ignored tests were added.
- Three model-free layer-plan tests, two partial-accounting tests, the QoS
  nesting/early-return test and all 17 harness tests pass. CPU acceptance is
  extended by explicit full test names/classes, not inferred discovery.
- `scripts/check-contracts.sh` passes on integration commit `39519c6eb`,
  including 381 library, 33 oracle, 54 bench and 165 CLI CPU acceptance tests,
  plus the comparison tripwire. The existing leased checkpoint cancellation
  model test was extended and passed on M5 (17.76s): cancel inside a partial
  row chunk, verify no cache publication/pin leak/reusable partial state, then
  restore the shared checkpoint with bitwise full logits and identical greedy
  output. No new ignored test was introduced. Tests use the test profile; executable builds
  use `nice -n 10 cargo build --profile iter`.

The production API regression mode is:

```
layer_slice_probe <corpus> 4 2 1 runtime 15000
```

It executes warm 48K and fresh-prefix appends at 32/342/1509 rows, comparing all
logits, all persistent cache bytes, a following decode, and peer logits/state
against ordinary unsplit execution. All 24 cases pass, including the 15s idle
case. A further full **96-pair** 1/2/4/8-slice regression passes all four bitwise
gates after the production refactor (thermal 0–2; timing is descriptive, not
the served gate; mean paired GPU delta -0.08%..+0.08%). A second runtime pass
also checks the dense-to-sparse frontier inside a row chunk, peers at unequal
depths, and restoring a full snapshot over a partial cursor. All pass.

It also verifies that partial layers refuse normal decode/snapshot/reset/
workspace release, that cancellation disposal is safe, and that an unequal
activation geometry declines before mutating either peer.

## Cold limit

The idle case is a real remaining limit: after 15s idle, the first 32-row
slice causes a **794.5ms** peer gap and **1073.9ms** appender TTFT, although
all append GPU command spans total **305.8ms**. About 700ms is outside the
recorded command GPU spans. Later warm repeats give **102.8ms** maximum gaps
and ~368ms TTFT. This was reported before any workaround; the layer mechanism
is unchanged. A nominal GPU quantum cannot preempt a cold host/driver cost
before the next submission completes.

Fresh-prefix (position zero) native max gaps are 96 / 90 / 123ms for
32 / 342 / 1509 rows. Warm 48K max gaps, excluding that idle arm, are
103 / 110 / 150ms. These native bounds exclude sampling/transport and the
cold limit above; the serve measurements below take precedence for delivery.

## Served comparison

Raw directory:
`metagross:~/qwen-runs/2026-11-11-layer-slice-production/`.

Same `iter` executable in every arm; SHA-256:
`f8f4c94bda321d429ed066bc794e0dd83135d294201563ef4561db3954ffbea3`.
`source-state.json` records hashes of all eleven production source files;
those hashes match integration commit `39519c6eb` and its rebased equivalent
`dbff76d0b`. The final landing adds startup/completion diagnostics and clarifies
knob documentation; it keeps the benchmarked scheduling and arithmetic. The M5 checkout starts at
`7c6062f18` with that source overlay. Only the explicit slice-ms environment
value differs between arms. The batch GPU loop stays enabled.

Workload: the same seeded three-agent real-call population (`20261109`), three
disjoint 48000-token real code roots, context capacity 196608, released agent
sampling defaults (GLM temperature 1.0/top-p 0.95), 15s ramp and 30s measurement
windows. Bulk requests, completions, event timestamps and native phase lines
are retained. No greedy substitution or fixed token-count drain.

The initial `off1` attempt aborted at thermal pressure 2 after its second warm
root, before the measurement window. The `on1` pilot completed at thermal 1
but overlapped unrelated compiler work: **77.17 tok/s**, streamed gap p50/p99/
max **28.7 / 144.3 / 309.6ms**, append-turn TTFT p50/p99 **1231 / 3115ms**
(n=7, no censored TTFT), `partial_prefill_s=2.3066`. These pilot outcomes are
retained separately. The longer-cooled baseline `off2` also aborted before its window: thermal
pressure rose to 2 while 72 unrelated compiler processes consumed about
3200% CPU. The harness gained opt-in warm-up-only thermal tolerance and a
continuous settling interval. Defaults stay unchanged. The final matched
ABBA uses 60s initial cooling, 45s between warm roots, permits at most 2 during
warm-up, then requires 15 continuous seconds at <=1 before the ramp. Ramp and
window still abort above 1. All four accepted windows remain at 0–1; their
full-arm maxima also happened to remain at 0–1. Failed warm-ups contribute no
timing result. All windows observed shared-host compiler activity; neither
arm's result is presented as an uncontended-host limit.

The requested results follow. Gaps use the existing content-delta timestamp
metric within the shared window; p99 is nearest-rank over the same events used
for harness p50/max. TTFT covers appending turns offered inside the window;
carry-in/ramp turns are excluded and censored cases are reported separately.
All throughput values are delivered text/reasoning/tool-argument tokens,
retokenized as in the existing harness.

| order / arm | tok/s | gap p50 / p99 / max ms | append TTFT p50 / p99 ms | TTFT n / censored | partial_prefill_s | timing thermal |
|---|---:|---:|---:|---:|---:|---:|
| A / off4 | 44.10 | 27.4 / 560.9 / 2120.1 | 1336 / 6540 | 18 / 1 | 2.0272 | 0 |
| B / on4 | 75.47 | 27.6 / 133.4 / 252.3 | 1085 / 2974 | 8 / 0 | 2.3896 | 0 |
| B / on5 | 73.40 | 28.1 / 138.1 / 252.8 | 1097 / 3050 | 8 / 0 | 2.4156 | 1 |
| A / off5 | 73.77 | 28.2 / 65.2 / 2111.8 | 844 / 1869 | 8 / 0 | 2.3791 | 1 |

### Decision: production available, default off

The comparable reverse pair (`on5`/`off5`) has the same number of turns and
one sampled tool call in each arm:

- Delivered throughput **73.40 vs 73.77 tok/s**, **-0.50%**, within observed
  run variation (the two candidate windows differ by 2.8%).
- Maximum gap **2112 → 253ms**, **-88.0%**, about **8.4× shorter**.
- **Global p99 65.2 → 138.1ms, +112%**. The few very long stalls become more
  numerous ~100ms gaps. Append-local p99 from the native probe is not the
  global percentile of the served stream.
- Append-turn TTFT p50 **844 → 1097ms** and p99 **1869 → 3050ms**; the
  appender explicitly gives time to its peers. Partial-prefill accounting is
  approximately unchanged (**2.3791 → 2.4156s**).

Thus the requested rule requiring material improvement in p99 **and** maximum
fails, even though throughput is neutral and the maximum improves strongly.
`QWEN_SERVE_GLM_LAYER_SLICE_MS` remains **0 by default**, with **100** available
for callers choosing the shorter worst stall. The result was reported before
any workaround. No smaller quantum, changed row partition, sampling mode or
substitute percentile was used to turn the gate into a pass.

Do not infer a 71% engine speedup from the forward pair. In `off4`, agent 1
repeatedly asks for more files: there are **13 sampled tool calls and 22
turns**, compared with **1 tool call and 11 turns** in each candidate and the
reverse baseline. Its prefill estimate is 15.68s rather than ~7.8s. This
endogenous turn churn dominates the forward-pair rate. The reverse pair is
the more useful throughput comparison; both orders consistently show the
large reduction in maximum gap.

### Output coherence

Read the longest streamed completion for each agent in each accepted arm
(12 samples), plus the pilot samples. They contain on-topic Rust/Metal review
text, concrete function/workspace invariants and valid file-request JSON; no
garbled or cross-agent text was observed in those samples. `on4` and `on5`
produce closely matching review continuations, with one tool call each. The
reverse baseline is likewise coherent, with one tool call. `off4` has the
repeated-file-request loop noted above, contrary to the prompt's no-more-tools
instruction. That is a workload/coherence limitation and is retained in the
report, rather than classified as an engine throughput gain.

Sample references: `a0-t2`, `a1-t4`, `a2-t3` in `on4`, `on5` and `off5`;
`a0-t2`, `a1-t2`, `a2-t3` in `off4`. All raw deltas/responses remain in the
arm directories.

### Accounting and reproducibility

`partial_prefill_s` is the existing harness estimate from prefill records,
including request-drop records. Sliced requests charge active prefill work,
excluding interleaved decoder time; the rollback's original accounting is
unchanged and can include percentage-credit service between whole row chunks.
The phase-based estimate is not a trace of every individual GPU interval.

`analysis.json`, source/data hashes, lease records, build logs, the exact
commands (`run-abba-v2.py`) and the original failed/pilot arms remain in the
raw directory. Every GPU hold was below seven minutes; waits were retained
in the canonical lease history. The M5 worktree and temporary branch are
removed after final validation. Nothing is merged.
