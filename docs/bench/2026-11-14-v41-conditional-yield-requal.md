# V4.1 conditional yield: replacement BA pair

Initial preparation started 2026-11-14 19:12:35 UTC. The lead stopped the
RAM-before-queue attempt and reset this stage at 19:38:30 UTC; stop/report by
21:08:30 UTC.
The lead authorized repeating only the invalid BA pair with the retained
release pins, then combining it with valid standing pairs 1 and 3. The
original B2 and failed A2 remain recorded; no slow valid pair is discarded.

## Frozen protocol

No performance threshold, call matching, percentile, warmup, purge, sampling,
profile or measurement niceness changes from the [standing protocol](2026-11-14-v41-conditional-yield.md).
New arm names are B2r then A2r, each its own waiting lease of at most 15 minutes.
The same archived qwen/qwen-tok/test binaries are used; runtime e41e037e2.
No M5 rebuild. Valid pairs 1 and 3 stand as measured. Native root/append
bitwise and empty-command-topology witnesses stand, with byte-identical
serve/witness metallibs. Branch is already based on main 1522261bc.

## Host-memory preflight

The lead requires recording free/available memory before each arm and waiting
when admission would fail. Availability uses the existing server formula:
`(free_pages + max(inactive_pages - anonymous_pages, 0)) * page_bytes`.
Speculative and active file-cache pages receive no extra credit.

The initial RAM-before-queue wait starved the lane: active lease holders could
keep occupying RAM before this lane joined the FIFO. The lead stopped that
attempt and killed its exact waiter PIDs 15561/15563 and SSH dispatch. No
model was loaded, and no measured arm was attempted under that ordering.

**Corrected order: join the Metal FIFO first.** Check RAM only after acquiring
the lease, waiting up to five minutes before releasing and requeueing.

A second preparation error was in this lane's guard: it added shared Metal
allocation bytes to CPU availability demand. The [native admission policy](../../crates/qwen-llm/src/metal/context.rs#L1049)
explicitly bounds shared Metal by physical capacity and Metal's working-set
limit, separately from the cache-blind CPU availability lower bound. The
196 GiB preload guard therefore wrongly blocked cache-heavy hosts. At one
baseline preparation, file-backed cache exceeded 215 GB while conservative
CPU availability was about 150 GB; no baseline model had loaded.

At 20:13 UTC this lane stopped only its own preparation controller and
children, retaining their records, and corrected the guard to **40 GiB**
(42,949,672,960 bytes), sustained for 15 seconds. That rounds up the observed
maximum process footprint (32,601,915,016), transport reserve (2,046,812,160),
and 4 GiB of margin. Process footprint can itself include shared storage,
so this remains a conservative CPU bound. The retained server's native Metal
capacity/working-set and CPU admission checks are unchanged.

B2r completed under the stricter original guard and also satisfies this CPU
bound. A2r had no measured run under that overcounted guard; only preparations
were deferred. No valid measured arm was discarded, and no performance rule
was changed. Ghost processes were never modified.

If CPU headroom remains insufficient after five minutes, or if too little
hold time remains for a complete arm, release before model load and requeue.
Deferred preparation remains recorded. Holds stay below 15 minutes. No memory
polling delays or retries are added to the measured request window. Memory
continues to be sampled during each arm to expose later pressure changes.

First-arm latest start 20:25 UTC; second-arm latest start 20:44 UTC leaves
cleanup/check/report time. An incomplete packet retains all pins.

## CPU readiness

`QWEN_SERVE_V41_PREFILL_YIELD=1 cargo test -p qwen-cli` passed all 12
targets: 1,748 passed, 0 failed, 73 pre-existing ignored. The legacy scheduler
fixture now explicitly chooses rollback settings, independent of the process
default. Production was default-off during measurement. The full gate now passes;
production is default-on with `QWEN_SERVE_V41_PREFILL_YIELD=0` rollback.

## Results

**PASS**. The full frozen gate passed.

Native witness: **bitwise PASS** for 48K CED root and 1K@80K, both empty polls and actual nested short service. Parent logits, causal payload digest and next-token logits match baseline; the nested 150-row peer matches logits/state. Empty polls also preserve GPU command-operation/dimension/submit-count topology. Ordinary release serve and perf-ablations witness metallibs are byte-identical. Runtime source `e41e037e2`; see [pins](2026-11-14-v41-conditional-yield-requal/pin-manifest.json) and [native witness](2026-11-14-v41-conditional-yield-requal/witness/result.json).

### Frozen ratios

Canonical pair-2 labels A2/B2 below refer to the replacement A2r/B2r arms.
The [mapping](2026-11-14-v41-conditional-yield-requal/reduction-mapping.json)
keeps standing pairs 1 and 3 unchanged.

B/A means candidate/baseline; A/B is the native/client latency speed ratio. All three pairs are required. No cells or arms were excluded based on speed, load or temperature within the allowed range.

| Metric | Pair1 AB | Pair2 BA | Pair3 AB | Geomean | Rule | Result |
|---|---:|---:|---:|---:|---|---|
| Short client TTFT p95 B/A | 0.498751 | 0.529547 | 0.510646 | 0.512825 | <1.00 | PASS |
| Aggregate TPS B/A | 1.019409 | 1.075793 | 1.053079 | 1.049171 | >=0.98 | PASS |
| Root native A/B | 1.275547 | 1.095640 | 1.516237 | 1.284430 | >=0.98 | PASS |
| Root client TTFT A/B | 1.274856 | 1.095515 | 1.515347 | 1.283898 | >=0.98 | PASS |
| Short native A/B | 1.003269 | 1.065914 | 1.030795 | 1.033009 | >=0.98 | PASS |
| ~1K native A/B | 1.032551 | 1.105976 | 1.043115 | 1.060058 | >=0.98 | PASS |
| SSE delta-gap p99 B/A | 1.108685 | 0.985475 | 0.925024 | 1.003542 | <=1.25 | PASS |

Pooled matched completion p50: A4005.238ms, B3860.216ms; B/A **0.963792**, 75 matched samples per arm, rule<=1.00: **PASS**.

| Pair | Matched short TTFT n | A p95 ms | B p95 ms | Completion matched n | Input/spec/row/window exclusions |
|---|---:|---:|---:|---:|---:|
| 1 | 19 | 1506.110 | 751.174 | 25 | 2 |
| 2 | 19 | 1397.045 | 739.801 | 25 | 1 |
| 3 | 19 | 1523.330 | 777.883 | 25 | 2 |

### Stream stalls and process I/O

Event gaps use the harness rule: both consecutive delta timestamps inside the 120s window. These are SSE-event gaps, not exact token ITL. Total process reads include startup/roots/ramp; window reads are the first-to-last in-window sampled counter delta. All model niceness values are 0.

| Arm | TPS | p99 ms | Max gap ms | Gaps>100ms / all | Process disk-read bytes | Window sampled bytes | Startup s | Thermal max | Compiler overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A1 | 88.875 | 31.016 | 2109.397 | 55/10596 | 191623065600 | 3435954176 | 18.120 | 2 | True |
| B1 | 90.600 | 34.387 | 1697.450 | 54/10805 | 39881265152 | 3449520128 | 1.994 | 2 | True |
| B2 | 90.958 | 31.931 | 1694.056 | 54/10846 | 193305767936 | 3122184192 | 23.938 | 2 | True |
| A2 | 84.550 | 32.401 | 2030.856 | 55/10081 | 189367013376 | 2463055872 | 18.230 | 2 | True |
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
| B2 | short | 19 | 481.982/739.801 | 370.963/656.880 | 45.840/254.615 |
| B2 | 1k | 9 | 1078.486/1607.599 | 1023.303/1467.422 | 42.426/691.555 |
| B2 | root | 3 | 25048.541/25525.104 | 24979.404/25442.317 | 69.137/82.787 |
| A2 | short | 19 | 437.836/1397.045 | 387.909/722.736 | 47.005/898.693 |
| A2 | 1k | 8 | 1161.253/1723.948 | 1123.154/1683.776 | 40.172/59.674 |
| A2 | root | 3 | 26973.763/29291.789 | 26900.795/29204.476 | 72.968/87.313 |
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
| B2 | short | 2.635/3.916 | 10.022/180.527 | 0.399/0.515 | 22.303/53.347 | 22.343/53.387 | 0.731/2.161 |
| B2 | 1k | 3.103/4.280 | 8.290/21.064 | 0.397/0.468 | 23.057/28.769 | 23.392/678.262 | 0.718/2.242 |
| B2 | root | 16.616/28.429 | 0.009/0.016 | 2.702/2.925 | 21.378/21.804 | 43.136/44.013 | 0.228/0.271 |
| A2 | short | 2.641/3.712 | 10.822/735.807 | 0.401/0.457 | 24.685/27.782 | 24.732/27.842 | 1.139/1.340 |
| A2 | 1k | 3.189/3.753 | 12.422/23.842 | 0.400/0.511 | 25.952/27.451 | 25.995/27.492 | 1.044/1.275 |
| A2 | root | 17.616/28.935 | 0.018/0.027 | 3.117/3.684 | 22.208/23.738 | 44.818/47.260 | 0.239/0.372 |
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
| B2 | a0-warm | 25.442317 | 25.525104 | 4096 | 12.25/12.65/12.88 |
| B2 | a1-warm | 24.211686 | 24.279250 | 4096 | 12.72/13.30/14.07 |
| B2 | a2-warm | 24.979404 | 25.048541 | 4096 | 12.56/12.77/13.10 |
| A2 | a0-warm | 29.204476 | 29.291789 | 4096 | 31.67/31.98/32.42 |
| A2 | a1-warm | 26.900795 | 26.973763 | 4096 | 32.09/32.84/33.33 |
| A2 | a2-warm | 25.666057 | 25.736919 | 4096 | 32.90/33.15/33.48 |
| A3 | a0-warm | 52.404039 | 52.506550 | 4096 | 18.82/36.11/39.60 |
| A3 | a1-warm | 39.568765 | 39.658533 | 4096 | 35.88/37.76/39.09 |
| A3 | a2-warm | 31.343023 | 31.426804 | 4096 | 31.91/34.66/36.45 |
| B3 | a0-warm | 29.076082 | 29.160556 | 4096 | 29.07/31.81/35.32 |
| B3 | a1-warm | 26.617063 | 26.689288 | 4096 | 26.24/27.54/29.07 |
| B3 | a2-warm | 25.637014 | 25.710256 | 4096 | 23.61/24.27/25.50 |

### Conditional service frequency

Actual peer-prefill events are counted independently of empty polls. Window counts use the child prefill round start. Completed-response poll counts include warm roots and ramp; censored requests without phases are excluded. Root and measured-window poll counts are retained separately in JSON. Ordinary priority dispatches happen at the regular pump, rather than inside a suspended parent.

| Arm | Yields full/window | Short priority full/window | Completed-response polls |
|---|---:|---:|---:|
| A1 | 0/0 | 0/0 | 0 |
| B1 | 2/2 | 0/0 | 58 |
| B2 | 2/2 | 0/0 | 60 |
| A2 | 0/0 | 0/0 | 0 |
| A3 | 0/0 | 0/0 | 0 |
| B3 | 2/2 | 0/0 | 58 |

### Interpretation

The unchanged frozen scheduling rule passes all eight checks. Short client TTFT p95 improved in every pair: candidate/baseline 0.498751, 0.529547 and 0.510646 (geomean 0.512825). Pooled matched completion p50 is 0.963792 with 75 samples per arm; aggregate TPS geomean is 1.049171, SSE gap p99 is 1.003542. Each candidate arm fired two actual in-window yields, all single native short calls; no priority-only dispatches occurred.

The first and third standing pairs remain unchanged. Only the invalid BA pair was replaced, as the lead requested. The original complete B2 and failed baseline A2 remain in the standing record. No performance threshold, percentile, matching rule, warmup, purge, sampling or measured request pacing changed. CPU/RAM preparation deferrals did not load a model or supply performance samples.

Root and native ratios are non-regression controls, not claimed kernel speedups. There were no peer services during the sequential roots, and the kernels, rows and layer order are unchanged. Startup, I/O, load and thermal drift remain visible in the per-arm tables. The queue benefit is supported by the repeated useful yields and short-call outside-native/mailbox reductions.

Production is default-on after the full gate, with QWEN_SERVE_V41_PREFILL_YIELD=0 restoring the incumbent pump. The measured release code is unchanged except for this default and its documentation; the test-only scheduler fixture explicitly selects rollback behavior. All numerical and Metal kernel sources are identical to the standing bitwise witness. Final clean-tree contracts and serve tests must pass before the branch is ready to land.

CPU readiness: full qwen-cli suite with yielding enabled passed 1,748 tests across 12 targets, zero failures, 73 pre-existing ignored. Final rollout validation is recorded separately. No new ignored tests, numerical kernels, precision changes, row repartition, checkpoint policy changes or pushes. Raw data remain at `metagross:~/qwen-runs/2026-11-12-v41-prefill/conditional-yield/`. Cleanup and final record-tree checks are recorded before reporting.

### Memory observations and preparation history

GB below is decimal; raw byte values and five-second time series are retained.
The pre-load readings are before the companion purge. All measured models
ran at niceness 0. Native Metal admission remained active in both arms.

| Arm | Pre-load free GB | Pre-load conservative available GB | Window free minimum GB | Window available minimum/maximum GB |
|---|---:|---:|---:|---:|
| B2r | 193.724 | 236.169 | 43.666 | 43.666/49.858 |
| A2r | 230.675 | 238.619 | 2.283 | 2.283/9.068 |

The [FIFO acquisition history](2026-11-14-v41-conditional-yield-requal/lease-history.json)
and [preparation logs](2026-11-14-v41-conditional-yield-requal/preparation/retry-attempts.json)
retain the five holds: B2r preparation deferred, B2r measured successfully,
A2r preparation deferred, A2r preparation stopped before model load to fix
this lane's CPU/Metal overcount, and A2r measured successfully. The earlier
RAM-before-FIFO wait was stopped by the lead. These were preparation
corrections, with no valid measured arm removed and no threshold override.

### Rollout and cleanup

`QWEN_SERVE_V41_PREFILL_YIELD` is default-on; `=0` restores the incumbent pump.
The release runtime differs from measured `e41e037e2` only in that default and
its documentation. The scheduler's test-only fixture explicitly selects
rollback settings. All library numerical/Metal sources are identical to the
standing witness. No M5 rebuild or extra model witness was needed.

The actual-default serve suite passed 719 tests (zero failures); the complete
qwen-cli suite previously passed 1,748 tests with yielding enabled. No new
ignored tests. ENV regeneration and fmt pass. Clean-tree contracts must pass
on the final rebased commit before the ready-to-land report.

[Cleanup audit](2026-11-14-v41-conditional-yield-requal/cleanup.json): M5 worktree,
pins and transient bytecode/build artifacts removed; no task processes or
registered PIDs remain. Four pins were archived and SHA-verified at
`/Users/tito/qwen-runs/2026-11-12-v41-prefill/conditional-yield-local-pins/`
before M5 deletion. Raw runs remain under the original M5 run directory.
No pushes. Final reporting is bounded by the 21:08:30 UTC limit.
