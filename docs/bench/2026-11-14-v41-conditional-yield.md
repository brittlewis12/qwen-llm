# V4.1 conditional prefill yielding

Bounded stage:2026-11-14 16:52:42–19:22:42 UTC. Branch
`perf/v41-prefill-yield-conditional`, based on `cabbb8a24` / main `1522261bc`.
The previous unconditional experiment stays recorded separately.

## Policy frozen before the first arm

Default-off `QWEN_SERVE_V41_PREFILL_YIELD=1`, independent `=0` rollback.
Conditional mode uses **only incumbent completed-command drains**: Full
appends poll after original Engram-overlap segments; CED polls after its
original encoder quantum drains, leaving the decoder pipeline untouched.
There are no imposed timed GPU cuts and no decoder rounds at empty polls.
Positive `LAYER_SLICE_MS` enables polls;0 disables them. Its400 default in
conditional mode is a legacy enable value, not a claimed time bound.
`LAYER_SLICE_MIN_MS` defaults700 in this mode; all calls below512 rows stay
unsliced even if the minimum is lowered.

At a drain, collect prepared HTTP work. A read-only compatible-live-prefix
hint identifies short pending suffixes before allocating/restoring a state;
unknown long prompts wait for ordinary admission. A waiting short request
finishes its original checkpoint/header calls as **one** service event,
without recursive yields. Suspended slots/reserves remain charged. Ready
short work also takes priority at ordinary pumps. Normal post-quantum decode
credit includes actual peer prefill service; empty polls retain incumbent
credit. No numerical kernels, precision, row partition, layer order, Engram
representation or checkpoint capture rules change.

Per-request polls, actual conditional yields and native-call counts are
recorded; explicit yield-event lines retain parent/child request IDs. Report
full-arm and measured-window fire counts separately from ordinary short-priority
dispatches. No work is called a yield solely because a poll occurred.

## Exactness and checks

CPU cases cover no decode inside empty polls, preserved post-quantum credit,
no short slicing, no long admission at an empty poll, read-only prefix hints,
bounded suspended admission/reserves and cancellation. No new ignored tests.
The M5 release witness compares baseline, empty polls and actual nested short
service at48K CED root and1K@80K: identical logits, causal payload digest and
next logits, plus peer logits/state. Empty polls must also preserve the exact
GPU command-operation/dimension/count topology. `perf-ablations` is used by the
witness for existing diagnostics only; serve is the ordinary release build,
and both metallib identities must match.

## Frozen serve rule

Same file-A/corpus and plan as the preceding gate: seed20261112, greedy,
three49,152-token sequential roots /32-token replies,15s ramp,120s window,
context196608, FAIR1 / MAX_ACTIVE4 / SAMPLE_WORKERS3. Same pinned release binary
for both switches. Run order **A1 B1 B2 A2 A3 B3**, each arm its own waiting
lease<=15min. Actual nice0 checked by `ps`; builds nice10. Engram companion
purged to0 resident pages before every arm; no other resident model. Settle
thermal<=1 continuously60s, allow2, abort3. Symmetric CPU boundary tests and
root/ramp warmup in both arms. Host load, disk reads and compiler presence stay
in the record; no speed/load-based arm exclusions. Latest arm start19:00 UTC
leaves cleanup/reporting time before19:22:42.

The **unchanged** previous reduction uses label/spec/target, exact prompt/token
hashes and actual new-row equality for calls offered in both windows. Short
means1..511 new rows; about1K512..2000. Matched observed TTFT includes partial
responses. Completion additionally requires both responses complete and equal
generated-token counts; pool all three matched pair samples before p50.
Percentiles are nearest rank. Report all exclusions and censored counts.

- Short client TTFT p95: geomean(candidate/baseline)<1.00.
- Pooled matched completion p50: candidate/baseline<=1.00.
- Aggregate TPS: geomean(candidate/baseline)>=0.98.
- Root client TTFT, root native, short native and about1K native: each
  geomean(baseline/candidate)>=0.98. Root uses each arm's mean of three roots;
  append-native ratios use sums over matched completed calls.
- SSE delta-event gap p99: geomean(candidate/baseline)<=1.25; both consecutive
  event endpoints inside the window, as in the harness. This is an event-gap
  proxy, not exact token-level ITL.

All three pairs and bitwise checks are required. Pass=>default-on with0
rollback; fail=>default-off and record whether the line merits more work.
If the queue prevents completion, report the completed measurements and keep
the pins. No pushes.

## Results

**INCOMPLETE**. The packet is incomplete; no promotion, default-off, pins retained.

Native witness: **bitwise PASS** for 48K CED root and 1K@80K, both empty polls and actual nested short service. Parent logits, causal payload digest and next-token logits match baseline; the nested 150-row peer matches logits/state. Empty polls also preserve GPU command-operation/dimension/submit-count topology. Ordinary release serve and perf-ablations witness metallibs are byte-identical. Runtime source `e41e037e2`; see [pins](2026-11-14-v41-conditional-yield/pin-manifest.json) and [native witness](2026-11-14-v41-conditional-yield/witness/result.json).

### Frozen ratios

B/A means candidate/baseline; A/B is the native/client latency speed ratio. All three pairs are required. No cells or arms were excluded based on speed, load or temperature within the allowed range.

| Metric | Pair1 AB | Pair2 BA | Pair3 AB | Geomean | Rule | Result |
|---|---:|---:|---:|---:|---|---|
| Short client TTFT p95 B/A | 0.498751 | pending | 0.510646 | pending | <1.00 | pending |
| Aggregate TPS B/A | 1.019409 | pending | 1.053079 | pending | >=0.98 | pending |
| Root native A/B | 1.275547 | pending | 1.516237 | pending | >=0.98 | pending |
| Root client TTFT A/B | 1.274856 | pending | 1.515347 | pending | >=0.98 | pending |
| Short native A/B | 1.003269 | pending | 1.030795 | pending | >=0.98 | pending |
| ~1K native A/B | 1.032551 | pending | 1.043115 | pending | >=0.98 | pending |
| SSE delta-gap p99 B/A | 1.108685 | pending | 0.925024 | pending | <=1.25 | pending |

| Pair | Matched short TTFT n | A p95 ms | B p95 ms | Completion matched n | Input/spec/row/window exclusions |
|---|---:|---:|---:|---:|---:|
| 1 | 19 | 1506.110 | 751.174 | 25 | 2 |
| 3 | 19 | 1523.330 | 777.883 | 25 | 2 |

### Stream stalls and process I/O

Event gaps use the harness rule: both consecutive delta timestamps inside the 120s window. These are SSE-event gaps, not exact token ITL. Total process reads include startup/roots/ramp; window reads are the first-to-last in-window sampled counter delta. All model niceness values are 0.

| Arm | TPS | p99 ms | Max gap ms | Gaps>100ms / all | Process disk-read bytes | Window sampled bytes | Startup s | Thermal max | Compiler overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A1 | 88.875 | 31.016 | 2109.397 | 55/10596 | 191623065600 | 3435954176 | 18.120 | 2 | True |
| B1 | 90.600 | 34.387 | 1697.450 | 54/10805 | 39881265152 | 3449520128 | 1.994 | 2 | True |
| B2 | 84.900 | 33.174 | 2485.538 | 55/10127 | 34786672640 | 2913976320 | 2.020 | 2 | True |
| A3 | 85.250 | 32.448 | 2270.496 | 55/10162 | 191374704640 | 3056746496 | 24.516 | 2 | True |
| B3 | 89.775 | 30.015 | 1711.028 | 54/10704 | 190758088704 | 3229302784 | 19.053 | 1 | True |

### Client and native components

Each cell is p50/p95 in milliseconds over all eligible observed TTFTs. These descriptive distributions include unmatched calls and have different denominators from the matched gate; n=19 versus n=20 can change which observation is p95. All component fields and censoring/exclusions are retained in the JSON.

| Arm | Class | n | Client TTFT | Native prefill | Outside native |
|---|---|---:|---:|---:|---:|
| A1 | short | 19 | 436.539/1506.110 | 378.843/664.141 | 55.424/1004.990 |
| A1 | 1k | 8 | 1100.620/1575.362 | 1045.043/1529.850 | 52.762/79.440 |
| A1 | root | 3 | 29228.723/42882.700 | 29145.650/42808.097 | 74.603/83.073 |
| B1 | short | 19 | 450.347/751.174 | 372.968/695.302 | 46.194/237.053 |
| B1 | 1k | 9 | 1081.064/1616.853 | 1026.317/1472.211 | 50.620/702.671 |
| B1 | root | 3 | 25017.506/27980.225 | 24947.041/27895.947 | 70.465/84.278 |
| B2 | short | 18 | 438.902/785.967 | 378.793/716.154 | 50.844/103.802 |
| B2 | 1k | 8 | 1107.551/1636.966 | 1071.114/1587.291 | 52.212/83.473 |
| B2 | root | 3 | 30454.821/32084.826 | 30378.860/32012.685 | 72.142/75.961 |
| A3 | short | 19 | 456.346/1523.330 | 392.596/699.706 | 55.830/1004.172 |
| A3 | 1k | 8 | 1091.977/1660.825 | 1069.848/1594.974 | 55.348/91.206 |
| A3 | root | 3 | 39658.533/52506.550 | 39568.765/52404.039 | 89.768/102.511 |
| B3 | short | 19 | 441.372/777.883 | 380.223/669.733 | 47.843/255.285 |
| B3 | 1k | 9 | 1109.309/1677.813 | 1037.336/1515.623 | 54.738/732.975 |
| B3 | root | 3 | 26689.288/29160.556 | 26617.063/29076.082 | 73.242/84.474 |

| Arm | Class | Tokenize | Work mailbox wait | Admission | Prefix checkpoint | Inside-prefill residual | Flush to client |
|---|---|---:|---:|---:|---:|---:|---:|
| A1 | short | 2.650/3.607 | 17.498/804.652 | 0.427/0.531 | 27.580/45.889 | 27.627/45.943 | 0.789/4.480 |
| A1 | 1k | 3.026/3.794 | 14.886/20.173 | 0.480/0.683 | 26.856/54.032 | 26.897/54.086 | 0.748/0.916 |
| A1 | root | 20.069/31.297 | 0.015/0.017 | 3.027/3.188 | 20.829/20.855 | 43.565/44.300 | 0.258/0.344 |
| B1 | short | 2.491/3.330 | 14.421/170.689 | 0.398/0.582 | 22.918/26.047 | 22.974/26.088 | 0.734/0.916 |
| B1 | 1k | 2.813/3.580 | 14.706/21.670 | 0.397/0.438 | 23.245/27.992 | 23.301/679.582 | 0.676/0.887 |
| B1 | root | 18.330/29.117 | 0.008/0.015 | 2.837/2.863 | 21.283/23.279 | 42.791/45.776 | 0.248/0.329 |
| B2 | short | 2.684/4.747 | 8.821/22.610 | 0.403/0.544 | 29.129/67.044 | 29.174/67.104 | 0.782/1.168 |
| B2 | 1k | 3.150/4.353 | 11.101/22.622 | 0.406/0.524 | 28.627/48.806 | 28.681/48.878 | 0.706/0.871 |
| B2 | root | 17.057/22.048 | 0.011/0.020 | 2.982/3.906 | 21.784/21.865 | 44.620/45.047 | 0.258/0.350 |
| A3 | short | 3.011/4.789 | 11.752/770.913 | 0.447/0.547 | 28.676/67.640 | 28.718/67.687 | 0.969/1.197 |
| A3 | 1k | 3.967/4.668 | 14.082/21.119 | 0.408/0.517 | 28.685/65.154 | 28.726/65.200 | 0.899/1.300 |
| A3 | root | 18.303/30.458 | 0.042/0.205 | 5.870/6.298 | 28.932/29.677 | 56.089/60.953 | 0.331/0.381 |
| B3 | short | 2.695/3.829 | 13.323/174.239 | 0.401/0.547 | 25.846/48.963 | 25.888/49.020 | 0.700/0.837 |
| B3 | 1k | 3.084/3.792 | 18.192/22.313 | 0.419/1.190 | 25.547/51.428 | 25.600/714.649 | 0.658/0.828 |
| B3 | root | 21.244/28.173 | 0.012/0.013 | 2.823/3.158 | 21.019/21.422 | 44.137/44.396 | 0.203/0.292 |

Inside-prefill residual is inclusive prefill-span time minus native work: peer service plus own host/checkpoint bookkeeping. It is not labelled entirely as GPU or queue time. Child-prefill nesting and decode-round widths are retained separately; ordinary prefix checkpoints stay synchronous and unchanged. Slot-queue/HTTP/render/sample/transport subcomponents remain in the full per-arm JSON.

### Every root and host load

Root native and client times are separate seconds; host load is sampled 1-minute load during that root, min/p50/max.

| Arm | Root | Native s | Client s | Selected rows | Load min/p50/max |
|---|---|---:|---:|---:|---:|
| A1 | a0-warm | 29.145650 | 29.228723 | 4096 | 17.28/17.51/18.11 |
| A1 | a1-warm | 42.808097 | 42.882700 | 4096 | 17.86/18.94/20.64 |
| A1 | a2-warm | 26.404162 | 26.476336 | 4096 | 20.64/21.44/21.57 |
| B1 | a0-warm | 27.895947 | 27.980225 | 4096 | 12.73/13.47/13.84 |
| B1 | a1-warm | 24.947041 | 25.017506 | 4096 | 11.26/12.05/12.87 |
| B1 | a2-warm | 24.267413 | 24.334731 | 4096 | 10.03/10.31/11.26 |
| B2 | a0-warm | 30.378860 | 30.454821 | 4096 | 16.78/17.78/17.88 |
| B2 | a1-warm | 32.012685 | 32.084826 | 4096 | 17.70/18.29/20.79 |
| B2 | a2-warm | 24.958622 | 25.030720 | 4096 | 19.16/20.44/20.83 |
| A3 | a0-warm | 52.404039 | 52.506550 | 4096 | 18.82/36.11/39.60 |
| A3 | a1-warm | 39.568765 | 39.658533 | 4096 | 35.88/37.76/39.09 |
| A3 | a2-warm | 31.343023 | 31.426804 | 4096 | 31.91/34.66/36.45 |
| B3 | a0-warm | 29.076082 | 29.160556 | 4096 | 29.07/31.81/35.32 |
| B3 | a1-warm | 26.617063 | 26.689288 | 4096 | 26.24/27.54/29.07 |
| B3 | a2-warm | 25.637014 | 25.710256 | 4096 | 23.61/24.27/25.50 |
| A2 invalid | a0-warm | 29.782115 | 29.864321 | 4096 | 32.43/34.51/37.60 |
| A2 invalid | a1-warm | 28.958923 | 29.029920 | 4096 | 30.22/31.08/32.43 |
| A2 invalid | a2-warm | 27.274107 | 27.354318 | 4096 | 29.24/29.40/30.22 |

### Conditional service frequency

Actual peer-prefill events are counted independently of empty polls. Window counts use the child prefill round start. Completed-response poll counts include warm roots and ramp; censored requests without phases are excluded. Root and measured-window poll counts are retained separately in JSON. Ordinary priority dispatches happen at the regular pump, rather than inside a suspended parent.

| Arm | Yields full/window | Short priority full/window | Completed-response polls |
|---|---:|---:|---:|
| A1 | 0/0 | 0/0 | 0 |
| B1 | 2/2 | 0/0 | 58 |
| B2 | 0/0 | 0/0 | 58 |
| A3 | 0/0 | 0/0 | 0 |
| B3 | 2/2 | 0/0 | 58 |

### Invalid baseline control and incomplete gate

Baseline A2 aborted during the measured window with `memory_admission_denied`: transport admission required 2,046,812,160 bytes, while reported host headroom was 2,022,670,336 bytes. Its failed calls, logs, timings, purge and process evidence remain in the raw packet; [failure and roots](2026-11-14-v41-conditional-yield/invalid-A2/failure.json). This is an invalid control, not a completed slow arm removed to obtain a pass. Pair 2 is pending; no three-pair verdict or promotion is claimed.

The valid completed pair numbers are 1, 3. The ratio table preserves the missing pair 2 column. All pins are retained.

Descriptive partial-packet values, without applying the frozen three-pair rule: short TTFT p95 B/A 0.504664, SSE gap p99 B/A 1.012699, pooled matched completion p50 B/A 0.955758 (50 samples per arm).

### Interpretation

The line merits finishing qualification. Pairs 1 and 3 each fired two actual short-prefill services. Matched short TTFT p95 ratios were 0.498751 and 0.510646, with SSE gap p99 ratios 1.108685 and 0.925024. Across these two valid pairs, pooled matched completion p50 was 0.955758 (50 samples per arm). These are descriptive partial results, not a three-pair pass. B2 had no actual yields, demonstrating that useful arrivals depend on timing at the incumbent drains. Empty-poll command topology is proven identical; the unconditional extra decode rounds from the preceding failed design are absent.

Root speed ratios should not be credited to conditional yielding. There were no peer services during the sequential warm roots. A1/A3 read about 191.6/191.4 GB over their full processes and took 18.1/24.5 seconds to start, whereas B1/B2 read 39.9/34.8 GB and started in about 2 seconds. Engram companion residency was purged to zero in every arm, but per-file disk reads are not logged, so the remaining I/O/cache attribution is unresolved. Every root, load, thermal result, compiler overlap and process-read counter is retained.

Baseline A2 failed transport memory admission during its window. No completed slow control was dropped to obtain a pass, and there is no full three-pair verdict. Keep the candidate default-off and preserve the pinned release binaries. Recommend repeating the BA pair with these pins when the host has adequate headroom, then applying the unchanged frozen rule before any promotion; make no further scheduler changes first.

CPU evidence: 55 selected shared serve tests and 719 broader serve tests passed (42 pre-existing ignored); contracts and fmt passed on the measured source. No new ignored tests, numerical kernels, precision changes, row repartition, checkpoint policy changes or pushes. Raw data remain at `metagross:~/qwen-runs/2026-11-12-v41-prefill/conditional-yield/`. Cleanup and final record-tree checks are recorded before reporting.

Cleanup: M5 worktree, build target, build temporary directory and source bundle were removed. All 30 registered M5 PIDs are absent; no task processes remain. The pinned binaries remain in the run directory because the gate is incomplete. See [cleanup audit](2026-11-14-v41-conditional-yield/cleanup.json). No task processes remain locally; the local transfer bundle was deleted.
