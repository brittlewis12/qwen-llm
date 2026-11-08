# 2026-11-07: budgeted conversation checkpoints through serve

Initial measurements: `a838cd13`, based on main `eb02cb0b`. Final implementation
`293c7c62` also selects V4.1's last body segment when the system prefix is
large and moves tip promotion/pruning after decode. Neither adds a numerical
operation or changes the measured checkpoint boundaries. Final on values below
use the confirmations where applicable.
GLM rollback control was built from `878c2ac7`; its disabled ladder follows the
same original capture/lookup and numerical execution paths.
V4.1 and Flash-Next final on confirmations use `293c7c62`; their rollback
controls use `a838cd13`. No prefill cuts, workspace widths, kernels or operand
precision change. The ladder retains copied committed state in RAM.

The session uses the previous standing's real Rust corpus in synthetic
`read_file` call/result history over `/v1/responses`, faithfully replaying
reasoning and message content. Four ~16K turns generate 64 tokens each, three
1–2K appends generate 200 each, then edited-question forks target ~48K and ~64K.
The final return resends the unedited deep turn (64 output tokens). Greedy,
seed 1, no drafter, context 98304; default engine precision/cache/idle settings.
One initial pair per model, two final on confirmations and a warm-only rollback;
this is a wall-time screen, not a timing confidence interval. TTFT is the first streamed reasoning/text delta.

Forks run in that order. In GLM rollback, the first fork creates a canonical
47,341-token checkpoint reused by the second; the 64K control is therefore a
warmer fork. Processed/reused counts below preserve this distinction.

Whole campaigns use the build lock, nice 10 and canonical Metal lease with
`QWEN_METAL_LEASE_WAIT=1`, one resident engine at a time. Nominal cooling gaps
are 25 seconds (45 before forks), including three thermal-zero observations.
Only turns with thermal pressure 0 immediately before and after qualify.

## GLM-5.3 UD-IQ3_XXS

| Edited prompt | TTFT off / on, s | Processed off / on | Reused off / on | Saved, s |
| --- | ---: | ---: | ---: | ---: |
| 48,567 | 34.354 / 3.239 | 33,994 / 1,723 | 14,573 / 46,844 | 31.115 |
| 64,690 | 18.984 / 3.282 | 17,349 / 1,723 | 47,341 / 62,967 | 15.702 |

Live extensions published 30,721, 46,844 and 62,967-token points. Their extra
capture costs were 49.0 / 68.7 / 83.9 ms. Append TTFT off/on was
2.177/2.174, 3.023/3.023 and 3.446/3.448 s. The middle append additionally
captured a 67,832-token endpoint **after decode** in 94.443 ms; the other two
made no extra copy (eligibility probes 0.034 / 0.023 ms). Return to the deep
turn restored that endpoint, processed 1,852 tokens and reached TTFT 3.515 s.
Peak optional retention: four points, 3,105,154,480 bytes (2.89 GiB), below the
8 GiB auxiliary ceiling. All seven pre-fork replies' content was identical
between arms (server-generated item IDs excluded).

`fast_trajectory_checkpoint_v1` identifies actual committed live states under
the same shared cut, frozen workspace rows, Fast precision and lineage. It
never seeds or satisfies the existing cold-canonical `ChunkOrigin` proof.
Restore is bit exact relative to the captured trajectory; Fast's existing
partition dependence means this is not a claim of cold-canonical equality.
The final-build model witness under Metal validation matched all prompt logits,
teacher-forced continuation logits and persistent end-state bytes.

## V4.1 GSQ-RCO IQ2_XS + IQ4_NL Engram

| Edited prompt | TTFT off / on, s | Processed off / on | Reused off / on | Saved, s |
| --- | ---: | ---: | ---: | ---: |
| 48,774 | 15.994 / 5.326 | 16,158 / 3,805 | 32,616 / 44,969 | 10.669 |
| 64,931 | 16.609 / 6.429 | 16,158 / 3,805 | 48,773 / 61,126 | 10.180 |

Existing committed Full calls published 12,631 / 28,811 / 44,969 / 61,126-token
interiors, costing 6.016 / 14.135 / 20.196 / 42.184 ms in the final confirmation. Every small append
reported zero interior capture time. Final TTFT off/on was 2.467/2.686,
3.119/3.401 and 8.046/3.555 s. The earlier on pass measured
2.604/3.186/3.313 s; neither pass adds small-append state copies. The last control was a driver-scheduling
outlier: its native GPU intervals sum to 1.817 s versus 1.816 s on, while
scheduling intervals sum to 6.109 s versus 1.371 s. These intervals can overlap;
they are diagnostic spans, not an additive attribution. No append speedup is
credited to the ladder. The final first two TTFTs vary by +219/+283 ms; one paired session
is insufficient to separate small effects from the observed scheduling drift. No new
small-append state copies or numerical calls were added.

Return to the deep turn processed two header tokens in both arms, TTFT
1.231/1.393 s in the final pass. Post-decode tip retention itself costs only
0.024–0.029 ms on the small appends. All seven pre-fork replies' content matched. The four interior
publication records account for 986,824,004 optional bytes (0.92 GiB); this is
not a peak-retention claim because later metadata-only tip promotions are not
logged by V4.1. The same 8 GiB auxiliary and parent-cache admission limits
bound those promotions. Serving remains Full; shrinking CED is not enabled.

## Flash-Next GSQ-RCO IQ3_S

| Edited prompt | TTFT off / on, s | Processed off / on | Reused off / on | Saved, s |
| --- | ---: | ---: | ---: | ---: |
| 48,805 | 9.038 / 1.868 | 16,093 / 1,757 | 32,712 / 47,048 | 7.171 |
| 64,960 | 9.159 / 1.944 | 16,091 / 1,755 | 48,869 / 63,205 | 7.215 |

The final on arm uses `293c7c62`: tip promotion/pruning runs after decode and
is timed separately as `retention_ms` (0.071–0.077 ms on the three appends). It adds no new model operation. Peak
logged optional retention is five points, 6,306,209,636 bytes (5.87 GiB).
All seven pre-fork replies' content matched rollback. The three small appends
make no additional interior copy. Capture/restore timing varied across the
long shared-machine session. Final on append TTFT was 1.525/1.847/2.027 s;
the same-binary late rollback measured 1.554/1.901/2.122 s, all thermal 0.
End-to-end wall was 4.647/4.973/5.207 s on versus 4.684/5.053/5.269 s off.
The earlier rollback measured 1.361/1.705/1.864 s. Late rollback also reproduced
the higher copy/restore costs, so the earlier difference cannot be credited to
pruning: instrumentation shows pruning is negligible, not the ~70 ms initially
suspected. No append slowdown is observed against the late control.

Long-turn prefill capture totals on were 80.6/126.5/214.1/231.0 ms versus
earlier rollback 49.8/67.9/91.4/108.6 ms (the on total includes one extra
interior copy). Small append totals were 154.9/103.9/112.1 ms on versus
193.1/150.7/215.7 ms in late rollback; these are existing transcript captures,
not extra ladder copies. Return to the deep turn processed five header tokens,
TTFT 0.471 s on. All retained points remain subject to the parent RAM budget.

## Source and exactness

The common retention/admission layer is in
[`checkpoint_ladder.rs`](../../crates/qwen-cli/src/serve/checkpoint_ladder.rs)
and [`snapshot_cache.rs`](../../crates/qwen-cli/src/serve/snapshot_cache.rs).
GLM's canonical and actual-trajectory namespaces remain separate in
[`backend_glm5_next.rs`](../../crates/qwen-cli/src/serve/backend_glm5_next.rs)
and its committed-command observer in
[`chunk_checkpoint.rs`](../../crates/qwen-cli/src/serve/backend_glm5_next/chunk_checkpoint.rs).
V4.1 uses the existing ready-Full capture/restore in
[`reuse.rs`](../../crates/qwen-cli/src/serve/backend_v41/reuse.rs), observing
successful scheduler calls in
[`execution.rs`](../../crates/qwen-cli/src/serve/backend_v41/execution.rs).
Flash-Next's read-only observer is in
[`qwen4exp_runtime.rs`](../../crates/qwen-llm/src/qwen4exp_runtime.rs), with
capture/admission in
[`backend_qwen4exp.rs`](../../crates/qwen-cli/src/serve/backend_qwen4exp.rs).

Exact model witnesses are
`glm5_next_metal::tests::snapshot_gates::live_trajectory_observer_restore_is_exact`
and `qwen4exp_runtime::tests::snapshot`. V4.1 reuses its existing bitwise
wrapped-ring, restore/growth and Full-prefill snapshot witnesses; the new
fake-state serve test also checks live publication, identical call partitions
and an older-message fork. The CPU inventory explicitly classifies the new
retention tests and excludes the leased model witnesses.

## Retention and controls

`QWEN_SERVE_CHECKPOINT_LADDER=0` restores the original capture/lookup behavior.
`QWEN_SERVE_CHECKPOINT_LADDER_MIB=auto` defaults to 8192 MiB, capped at half the
existing snapshot-cache budget; explicit zero disables the ladder. Optional
points and normal tips share that parent cache, admission, expiry and pressure
eviction. Five spaced optional points per first-user-turn conversation family
are an upper bound, not a guaranteed reservation. Exact token keys and backend
compatibility validate restore; the family hash affects retention only.

Already-paid turn tips can advance coverage without another copy. The recent
and deepest tip pairs survive older-message forks. Superseded tips are pruned
where covered; a budget too small for coverage preserves legacy fallback.
V4.1 continues to reject CED-origin or poisoned states. Drafting paths retain
their original cold-only snapshot behavior. No cross-restart durability is added.

Full release library suite: 2,469 passed, 741 ignored, zero failures; the library
and kernels are unchanged by subsequent CLI-only fixes. Exact CPU contracts
passed for final implementation tree `5a4eefc229298f71fa50a88b27e5524e9f339272`
(`293c7c62`). All 21 pre-fork reply contents match rollback, and all primary
turns and late control turns have thermal pressure 0 before/after.
The Flash-Next observer model witness passed with the tier-1 GSQ IQ3_S
artifact. An additional existing witness explicitly declared for the released
UD-Q3_K_XL fixture was explored with that GSQ override. Its prefix-13 bitwise
restore logits and all persistent-tensor comparisons passed, then its separate
cold-versus-split comparison failed max-absolute tolerance: 0.22040343 versus
0.2 (same argmax, cosine 0.99990017, relative RMS 0.01429373). The check compares
different prefill partitions, not restoration; no tolerance was changed. Its
raw failed result is retained in `fn-exact.log`. This is not a claim that GSQ
passes the UD fixture's cold/split qualification.

Raw requests, responses, SSE events, per-turn phase logs, corpus and scripts:
`~/qwen-runs/2026-11-07-serve-checkpoints/`.
