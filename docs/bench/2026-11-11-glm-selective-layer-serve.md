# 2026-11-11 — Selective GLM layer slicing: one 500ms / 200ms variant

Branch: `perf/glm-layer-sliced-prefill`. Policy source: `0971a85c2`.
**Decision: keep opt-in.** Exactly one variant was tested in one attempted
30s real-sampling ABBA packet. Three windows were accepted; the last baseline
was thermally rejected during the ramp. The one complete A/B pair already
fails the global-p99 gate. There was no replacement run or parameter retune. The
previous all-append 100ms result is in `2026-11-11-glm-layer-slice-serve.md`.

## Fixed policy

- `QWEN_SERVE_GLM_LAYER_SLICE_MS=200`: nominal GPU quantum for selected
  appends. Zero retains the original whole-row-chunk/credit behavior.
- `QWEN_SERVE_GLM_LAYER_SLICE_MIN_MS=500`: enter the layer cursor only when
  the append's predicted original prefill wall **strictly exceeds 500ms**.
  Zero restores all-append slicing for a positive slice target.
- Prediction sums the existing row chunks and checkpoint cuts, using the
  same measured floor/per-row function as the layer planner. Zero remainder
  rows cost zero. The threshold is checked before any cursor, QoS change or
  session mutation; rejected appends use the legacy prefill/credit path.
- An armed cursor continues through final validation regardless of later peer
  readiness. Row geometry, weights, arithmetic, checkpoint observers and
  cancellation handling are unchanged.

At the qualified M5 Fast/Half 48K geometry, 32 and 67 rows predict 308 and
413ms and stay unsliced. 342/951/1509 rows qualify; at a 200ms target they
start with 4/16/8 layer slices. The estimate is a warm-work seed, not a cold
residency or hard latency bound. There is no parameter search after the packet.

## Controls and validation

Both arms include current main `f03637fa2`, including accepted-connection
TCP_NODELAY. Results from the earlier Nagle-enabled packet are not controls.
The same iter executable is used for all four arms; only slice-ms differs:
0 / 200 / 200 / 0. The minimum is explicitly 500 and the default-on batch GPU
loop explicitly stays enabled in every arm.

Same real three-agent corpus and seeded plan (`20261109`): three disjoint
48000-token roots, 196608 capacity, released agent sampling defaults, 15s
ramp and 30s measurement windows. Each arm takes its own canonical lease,
with 60s initial cooling, 45s between warm roots, optional warm-up limit 2,
then 15 continuous seconds <=1 before the ramp. Ramp/window still abort above
1. Full events, native phases and sampled responses are retained.

Local checks: 693 serve tests pass (42 existing ignored), four layer-plan
tests pass, and the selective threshold/row-plan tests pass. The original
cursor/bitwise/cancellation implementations are unchanged. Integration
`check-contracts.sh` passes on `5ec7e2ec7`. No ignored tests were added.

M5 `df -h ~` was checked before building: 41 GiB initially and 48 GiB before
the cached iter build. One CLI build passed; space reached 36 GiB during the
build, then recovered to 40–46 GiB as other shared activity changed. No M5
test-profile build, additional policy probe build or cache garbage collection
was performed for this packet. Benchmark artifacts are small relative to
that free space. Free disk later reached 18 GiB amid shared activity; no
further builds were performed. Our M5 worktree and temporary branch were
removed after the packet; free disk then read 19 GiB. No other worktree or
shared cache was removed.

Raw: `metagross:~/qwen-runs/2026-11-11-glm-selective-layer/`, including exact
commands in `run-abba.py`, `source-state.json` and per-arm `launch.json`.
Binary SHA-256:
`9f3144831d6622ba587dbe104413221a2142c272e31ca9224ab60190715b5bbe`.

## Results and ship decision

The table uses the existing content-delta gap metric and nearest-rank p99.
TTFT covers appending turns offered inside the common window, excluding
carry-in/ramp turns. All accepted TTFTs are uncensored. The shared host had
compiler activity in every accepted window; no isolated-host claim is made.

| order / arm | delivered tok/s | gap p50 / p99 / max ms | append TTFT p50 / p99 ms | TTFT n | partial_prefill_s | window thermal |
|---|---:|---:|---:|---:|---:|---:|
| A / off-a | 73.07 | 29.3 / 58.4 / 1794.1 | 943 / 1940 | 7 | 2.3622 | 1 |
| B / on-a | 77.13 | 28.1 / 211.5 / 535.6 | 955 / 2064 | 8 | 2.4041 | 1 |
| B / on-b | 71.87 | 28.7 / 197.5 / 704.9 | 1007 / 2150 | 8 | 2.4240 | 1 |
| A / off-b | rejected | pressure 2 during ramp | no accepted window | — | — | — |

The last baseline's trace rises to pressure 2 at 20:36:14 UTC, after thermal
settling but before its measurement window, with 95 compiler processes
consuming about 2678% CPU. Its partial event/log files and `failed.json` are
retained. It contributes no timing summary. The requested packet was not
extended to replace it, since the valid pair already fails the gate.

In the complete forward pair:

- Delivered throughput **+5.6%**; the second candidate measured 71.87 tok/s,
  so the accepted candidate range brackets the 73.07 control and does not
  establish a repeatable throughput gain without the reverse baseline.
- Maximum gap **1794 → 536ms**, **-70.1%**, with **705ms** in the second
  candidate. This materially removes the seconds-long tail in accepted runs.
- Global p99 **58.4 → 211.5ms, +262%**; second candidate **197.5ms**. This
  is a material regression, not a default-on pass.
- Append TTFT p50 **943 → 955ms, +1.2%**, near neutral in that pair; second
  candidate **1007ms**. The incomplete reverse control limits a stronger
  non-inferiority conclusion. Partial prefill is **2.3622 → 2.4041s**.

**Default remains `QWEN_SERVE_GLM_LAYER_SLICE_MS=0`.** The tested variant is
available with **MS=200 / MIN_MS=500**; MIN_MS=0 retains all-append slicing.
No policy, metric, sampling mode or threshold was substituted after failure.

## Mechanism and small-append audit

Phase records confirm that every reported 32/45/67-row append in both
candidate arms has **`layer_slices=0`**. Completed selected examples have the
expected **342→4, 951→16, 1509→8** slices. A large uncontended append can
also have zero slices because no peer decoder is ready; that existing rule
is unchanged. `policy-audit.json` retains the per-call observations.

Removing the small forward's own slice tax does not remove queuing behind a
selected append. For completed small calls with in-window TTFT and known row
counts, nearest-rank TTFT p50 is **744ms off (n=2)** and **955 / 886ms on
(n=3 each)**. These tiny, different cohorts do not establish small-turn TTFT
non-inferiority. The forwarding path is preserved; end-to-end latency is not
claimed to be identical.

The event distribution explains the remaining p99 regression:

| complete pair | content gaps | >100ms | >200ms | >1s |
|---|---:|---:|---:|---:|
| off-a | 2181 | 18 | 13 | 4 |
| on-a | 2292 | 68 | 23 | 0 |

The selective policy eliminates the >1s gaps in this pair, but enough shorter
gaps remain to occupy the global top percentile. A 200ms layer target also
does not bound an intentionally unsliced small forward to 200ms: the gate
allows up to roughly 500ms of predicted work, plus model error and host/cold
costs. Thus the observed ~0.5–0.7s maximum is compatible with the policy.
The known cold-idle limitation remains outside the warm estimate.

## Coherence and reproducibility

Read the longest streamed completion for each agent in each accepted arm
(nine samples). They are on-topic Rust/Metal review continuations with
concrete functions, invariants and test observations; no garbled or
cross-agent text was observed in these samples. `off-a` has ten turns and
zero sampled tool calls; each candidate has eleven turns and one valid
file-request call. Outputs are sampled, so a fixed-output speedup is not
inferred. Samples and raw responses remain in `coherence-samples.json` and
the arm directories.

All four lease acquisitions and waits are retained in `lease-records.json`:
holds **382 / 375 / 373 / 326s**, waits **0 / 41 / 39 / 67s**, all below the
15-minute limit. `analysis.json`, `decision.json`, exact commands, the rejected
arm and source/data hashes in `manifest.json` remain in the raw directory.
The branch contains a PERF-LOG entry and remains unmerged.
