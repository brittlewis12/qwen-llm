# V4.1 row-preserving prefill yielding

Stage start 2026-11-14 14:54:10 UTC; stop/report by 16:54:10 UTC. Fresh branch
`perf/v41-prefill-yield` from main `1522261bc`. Default-off prototype;
`QWEN_SERVE_V41_PREFILL_YIELD=1` enables it, `=0` restores the incumbent.

## Design and exactness

V4.1 already has drained-command layer callbacks, but they service decoders
only and do not return to HTTP intake. The new hook collects already-prepared
HTTP jobs at those boundaries, admits at most one within the existing active
limit, and executes one waiting short prefill call (remaining rows<512).
The original parent is suspended on the same stack/thread, excluded from
decode/snapshot scans; its active slot and output reserve remain charged.
Peer prefill does not recursively interleave. Raw HTTP/control events and
worker joins stay in the ordinary loop. Native work excludes peer service;
`peer_prefills` is retained with phase timings.

The scheduling switch selects existing400ms layer targets (minimum predicted
work500ms); explicit `QWEN_SERVE_V41_LAYER_SLICE_MS=0` disables boundaries.
Rows, layer order, kernels, precision, CED encoder/replay geometry, Engram
representation and checkpoint rules are unchanged. Shared tracing stays opt-in.
The GLM resumable scheduler informed the policy; the V4.1 implementation uses
its existing stack-preserving callbacks instead of moving native state.

CPU tests cover short service before the parent finishes, identical scripted
outputs, suspended admission/reserves, cancellation, and actual loopback
prepared-job submission before the parent pump returns. The native witness
uses production layer APIs at48K CED root and1K@80K, comparing logits, full
causal-payload digests, next-token logits and a nested 150-row peer's logits/state.
`perf-ablations` is needed by the witness only for the existing snapshot digest;
serve uses the ordinary release build. Both metallib identities are recorded.

## Frozen serve protocol, before the first arm

One pinned release binary; A=`PREFILL_YIELD=0`, B=`PREFILL_YIELD=1`, each
without a layer-MS override (A default0, B default400). Other settings fixed:
FAIR1, MAX_ACTIVE4, SAMPLE_WORKERS3, greedy; file A/corpus and seed20261112;
three sequential49,152-token roots/32-token replies,15s ramp,120s window,
context196608. Run order **A1 B1 B2 A2 A3 B3** (AB/BA/AB). Each arm gets its
own waiting lease<=15min. Engram companion purged to0 resident pages before
each load. Actual serve nice0; build nice10; no resident model before loading.
Settle thermal<=1 continuously60s, allow2, abort3. Background load/compiler
presence and process/window disk-read counters are retained for every arm.
Identical CPU boundary tests plus three-root/ramp warmup in both arms; all GPU
pipeline shapes are incumbent shapes. No arm exclusions based on speed/load.

Nearest-rank percentiles, geometric means of three paired ratios, with
matching by label, request spec, exact input-token IDs/prompt hash and actual
new-row count. Calls must be offered inside both windows. Short means1..511
new rows, about1K means512..2000. TTFT includes every matched observed first
token, even if the response is partial; unobserved/censored counts are reported.
Completion matching additionally requires completion in both arms and equal
generated-token counts; completion is end-sent. Raw counts and exclusions remain.

Frozen scheduling rule:

- Short client TTFT p95: geomean(candidate/baseline)<1.00.
- Pooled matched-call completion p50: candidate/baseline<=1.00, pooling the
  three pairs' matched completed samples before taking each median.
- Aggregate TPS: geomean(candidate/baseline)>=0.98.
- Root client TTFT, root native prefill, short native prefill, and about1K
  native prefill: each geomean(baseline/candidate)>=0.98. Native append ratios
  use sums over matched completed calls, preserving the previous reduction.
- Client-visible SSE delta-event gap p99: geomean(candidate/baseline)<=1.25,
  using each arm's harness window gaps. This is an event-gap proxy for ITL.

All three pairs and the bitwise witness are required for promotion. If the
queue prevents completion by the cutoff, report incomplete and keep pins.
Pass: default-on with0 rollback. Fail: default-off, record retained. The prior
served A/A control already demonstrated batch-composition variation; this
stage's exactness claim is same-input/rows/order native logits and state.

## Results

**FAIL**. The frozen gate failed; the candidate stays default-off.

Native witness: **bitwise PASS**, 48K CED root 53 callbacks and 1K@80K 2 callbacks. Parent logits, causal payload digest and next-token logits match unsliced execution; a nested 150-row peer matches logits/state. Serve and witness metallibs are byte-identical. Runtime source `f33834a2c`; ordinary release serve binary SHA256 `493817a1522dff23032eb00b4ce1a793c89c3b71b9b644c6b8cdb5ad18b3c3b6`. See [pins](2026-11-14-v41-prefill-yield/pin-manifest.json) and [native witness](2026-11-14-v41-prefill-yield/witness/result.json).

### Frozen ratios

B/A means candidate/baseline; A/B is the native/client latency speed ratio. All three pairs are required. No cells or arms were excluded based on speed, load or temperature within the allowed range.

| Metric | Pair1 AB | Pair2 BA | Pair3 AB | Geomean | Rule | Result |
|---|---:|---:|---:|---:|---|---|
| Short client TTFT p95 B/A | 0.630671 | 0.616161 | 0.617656 | 0.621462 | <1.00 | PASS |
| Aggregate TPS B/A | 0.990435 | 1.021708 | 0.977466 | 0.996364 | >=0.98 | PASS |
| Root native A/B | 1.071740 | 1.177001 | 1.097601 | 1.114560 | >=0.98 | PASS |
| Root client TTFT A/B | 1.071360 | 1.176798 | 1.097184 | 1.114224 | >=0.98 | PASS |
| Short native A/B | 1.001653 | 1.020244 | 0.986349 | 1.002653 | >=0.98 | PASS |
| ~1K native A/B | 1.004663 | 1.014261 | 0.976972 | 0.998506 | >=0.98 | PASS |
| SSE delta-gap p99 B/A | 9.307927 | 6.975229 | 9.381581 | 8.476746 | <=1.25 | FAIL |

Pooled matched completion p50: A3768.408ms, B3979.043ms; B/A **1.055895**, 76 matched samples per arm, rule<=1.00: **FAIL**.

| Pair | Matched short TTFT n | A p95 ms | B p95 ms | Completion matched n | Input/spec/row/window exclusions |
|---|---:|---:|---:|---:|---:|
| 1 | 19 | 1416.201 | 893.157 | 26 | 2 |
| 2 | 19 | 1466.275 | 903.461 | 25 | 2 |
| 3 | 19 | 1382.081 | 853.651 | 25 | 5 |

### Stream stalls and process I/O

Event gaps use the harness rule: both consecutive delta timestamps inside the 120s window. These are SSE-event gaps, not exact token ITL. Total process reads include startup/roots/ramp; window reads are the first-to-last in-window sampled counter delta. All model niceness values are 0.

| Arm | TPS | p99 ms | Max gap ms | Gaps>100ms / all | Process disk-read bytes | Window sampled bytes | Startup s | Thermal max | Compiler overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A1 | 91.475 | 29.845 | 1679.421 | 58/10906 | 34395430912 | 2714861568 | 2.276 | 0 | True |
| B1 | 90.600 | 277.795 | 1196.196 | 112/10804 | 189385408512 | 2469003264 | 17.215 | 2 | True |
| B2 | 89.817 | 274.251 | 1209.874 | 108/10712 | 34457112576 | 2768896000 | 1.897 | 2 | True |
| A2 | 87.908 | 39.318 | 1719.648 | 54/10480 | 189809594368 | 2658861056 | 19.138 | 2 | True |
| A3 | 92.083 | 29.445 | 1661.969 | 62/10977 | 189416144896 | 2488467456 | 17.541 | 0 | True |
| B3 | 90.008 | 276.242 | 1174.186 | 108/10738 | 189371465728 | 2458468352 | 17.553 | 2 | True |

### Client and native components

Each cell is p50/p95 in milliseconds over all eligible observed TTFTs. These descriptive distributions include unmatched calls and have different denominators from the matched gate; n19 versus20 can change which observation is p95. All component fields and censoring/exclusions are retained in the JSON.

| Arm | Class | n | Client TTFT | Native prefill | Outside native |
|---|---|---:|---:|---:|---:|
| A1 | short | 20 | 400.624/1315.474 | 366.074/651.494 | 39.214/677.872 |
| A1 | 1k | 9 | 1073.290/1515.419 | 1029.120/1480.624 | 42.233/53.083 |
| A1 | root | 3 | 26986.376/28984.961 | 26912.858/28902.648 | 76.456/82.312 |
| B1 | short | 19 | 434.563/893.157 | 380.993/660.537 | 57.768/333.851 |
| B1 | 1k | 9 | 1148.938/1660.083 | 1026.357/1471.764 | 122.698/736.167 |
| B1 | root | 3 | 24912.283/27144.082 | 24825.582/27059.902 | 84.180/86.701 |
| B2 | short | 19 | 436.709/903.461 | 380.128/658.499 | 53.983/329.323 |
| B2 | 1k | 8 | 1166.029/1689.320 | 1037.457/1497.307 | 125.994/754.312 |
| B2 | root | 3 | 24346.888/26617.162 | 24274.185/26539.552 | 72.703/77.610 |
| A2 | short | 19 | 419.125/1466.275 | 368.703/674.835 | 49.623/952.400 |
| A2 | 1k | 8 | 1102.897/1569.425 | 1064.503/1514.969 | 51.160/68.924 |
| A2 | root | 3 | 29014.897/31838.325 | 28932.689/31747.613 | 82.208/90.712 |
| A3 | short | 22 | 400.752/1276.901 | 369.458/647.823 | 39.253/649.560 |
| A3 | 1k | 9 | 1043.970/1496.585 | 1023.167/1459.619 | 30.911/38.569 |
| A3 | root | 3 | 26516.518/28603.901 | 26443.714/28518.550 | 72.804/85.351 |
| B3 | short | 19 | 443.353/853.651 | 367.684/687.302 | 57.753/328.586 |
| B3 | 1k | 8 | 1221.331/1681.394 | 1066.430/1506.093 | 128.201/744.808 |
| B3 | root | 3 | 24443.872/24998.112 | 24359.774/24922.279 | 75.833/84.098 |

| Arm | Class | Tokenize | Work mailbox wait | Admission | Prefix checkpoint | Inside-prefill residual | Flush to client |
|---|---|---:|---:|---:|---:|---:|---:|
| A1 | short | 2.325/3.542 | 8.970/557.421 | 0.394/0.413 | 22.762/25.203 | 22.802/25.255 | 0.964/1.161 |
| A1 | 1k | 2.878/3.774 | 11.529/20.998 | 0.395/0.413 | 22.087/26.519 | 22.128/26.559 | 0.865/1.167 |
| A1 | root | 20.449/28.973 | 0.025/0.039 | 3.099/3.110 | 21.273/21.356 | 43.960/45.534 | 0.257/0.350 |
| B1 | short | 2.571/3.778 | 12.422/304.601 | 0.461/0.619 | 21.747/24.938 | 22.687/46.351 | 1.015/1.217 |
| B1 | 1k | 3.240/3.951 | 13.498/18.253 | 0.428/0.920 | 22.745/24.540 | 69.305/698.463 | 0.977/1.276 |
| B1 | root | 19.035/30.076 | 0.015/0.018 | 2.821/2.980 | 20.216/25.693 | 44.237/54.458 | 0.221/0.446 |
| B2 | short | 2.598/3.399 | 12.279/254.741 | 0.431/0.559 | 24.209/28.716 | 24.690/50.555 | 0.867/1.103 |
| B2 | 1k | 3.165/3.769 | 8.560/20.370 | 0.424/0.513 | 26.740/28.497 | 73.067/720.934 | 0.772/0.978 |
| B2 | root | 19.922/23.840 | 0.013/0.015 | 3.643/4.436 | 19.942/20.849 | 42.411/43.012 | 0.246/0.327 |
| A2 | short | 2.790/3.468 | 15.141/761.934 | 0.398/0.689 | 25.315/46.994 | 25.355/47.082 | 1.155/1.362 |
| A2 | 1k | 3.142/3.914 | 9.100/20.127 | 0.398/0.437 | 25.573/55.184 | 25.620/55.246 | 1.067/1.267 |
| A2 | root | 22.617/28.970 | 0.014/0.016 | 3.115/4.168 | 22.668/23.225 | 48.019/49.069 | 0.239/0.441 |
| A3 | short | 2.199/2.890 | 10.098/533.547 | 0.394/0.408 | 20.697/23.224 | 20.746/23.264 | -0.856/-0.662 |
| A3 | 1k | 2.357/3.141 | 1.023/8.916 | 0.393/0.408 | 21.461/22.892 | 21.501/22.930 | -0.910/-0.749 |
| A3 | root | 20.618/31.316 | 0.013/0.038 | 3.012/3.297 | 19.811/20.663 | 41.373/42.163 | 0.236/0.329 |
| B3 | short | 2.591/3.738 | 15.360/268.266 | 0.399/0.533 | 22.772/25.206 | 23.150/48.907 | 0.928/1.685 |
| B3 | 1k | 3.152/3.442 | 12.482/23.667 | 0.404/0.518 | 23.461/29.477 | 77.990/704.979 | 0.867/1.078 |
| B3 | root | 20.301/27.994 | 0.050/0.111 | 2.782/4.130 | 20.375/22.184 | 44.725/45.788 | 0.241/0.450 |

Inside-prefill residual is inclusive prefill-span time minus native work: peer service plus own host/checkpoint bookkeeping. It is not labelled entirely as GPU or queue time. Child-prefill nesting and decode-round widths are retained separately; ordinary prefix checkpoints stay synchronous and unchanged. Slot-queue/HTTP/render/sample/transport subcomponents remain in the full per-arm JSON.

### Every root and host load

Root native and client times are separate seconds; host load is sampled 1-minute load during that root, min/p50/max.

| Arm | Root | Native s | Client s | Selected rows | Load min/p50/max |
|---|---|---:|---:|---:|---:|
| A1 | a0-warm | 28.902648 | 28.984961 | 4096 | 22.79/24.69/24.99 |
| A1 | a1-warm | 26.912858 | 26.986376 | 4096 | 20.17/21.21/22.79 |
| A1 | a2-warm | 25.285098 | 25.361554 | 4096 | 18.84/19.43/20.17 |
| B1 | a0-warm | 24.825582 | 24.912283 | 4096 | 13.11/13.85/14.27 |
| B1 | a1-warm | 23.786400 | 23.859203 | 4096 | 13.70/13.88/14.32 |
| B1 | a2-warm | 27.059902 | 27.144082 | 4096 | 14.21/17.64/28.44 |
| B2 | a0-warm | 26.539552 | 26.617162 | 4096 | 17.52/18.93/19.94 |
| B2 | a1-warm | 24.274185 | 24.346888 | 4096 | 19.14/19.53/19.63 |
| B2 | a2-warm | 23.885913 | 23.958405 | 4096 | 18.61/18.93/19.45 |
| A2 | a0-warm | 31.747613 | 31.838325 | 4096 | 22.16/27.73/35.04 |
| A2 | a1-warm | 28.932689 | 29.014897 | 4096 | 35.04/40.47/44.19 |
| A2 | a2-warm | 27.241237 | 27.315410 | 4096 | 44.19/46.36/46.41 |
| A3 | a0-warm | 28.518550 | 28.603901 | 4096 | 13.55/13.84/13.91 |
| A3 | a1-warm | 26.443714 | 26.516518 | 4096 | 13.80/14.12/14.32 |
| A3 | a2-warm | 25.147182 | 25.216687 | 4096 | 13.57/13.85/14.34 |
| B3 | a0-warm | 24.359774 | 24.443872 | 4096 | 17.73/18.07/20.08 |
| B3 | a1-warm | 24.922279 | 24.998112 | 4096 | 22.00/25.09/28.67 |
| B3 | a2-warm | 23.703895 | 23.779168 | 4096 | 29.82/32.37/35.13 |

### Interpretation and next measured stage

The new policy trades rare long stalls for more frequent shorter ones. In pair1, gaps>100ms rose 58/10,906→112/10,804, crossing the 1% boundary; maximum gap improved 1.679→1.196s while p99 worsened 29.845→277.795ms. Only three full-run parent calls actually nested a peer prefill (two in the measured window). Unconditional decode rounds at otherwise empty layer boundaries add pauses and delay the prefilling caller; the tracing shows that delay inside its prefill span, rather than as native compute.

Recommend a more selective follow-up, not promotion of this variant. The smallest existing-knob experiment is `QWEN_SERVE_V41_LAYER_SLICE_MIN_MS=700`: all short(<512-row) calls remain unsliced because their maximum predicted wall is 683.8ms, while the observed 611/1,149-row window parents still qualify. Report pause frequency and maximum as well as p99. A stronger follow-up would service actual waiting short work at drained boundaries and preserve incumbent decode credit when no such work exists, avoiding unconditional extra decode rounds. Neither change was made or measured in this frozen packet.

CPU evidence: 26 selected scheduler/loopback tests and 51 broader shared serve tests passed; contracts passed on `f33834a2c`, tree `3fa1540ae5db5f96a8aab24f0b76e481a004e2d8`. No new ignored tests, shader changes, precision changes, row repartition or pushes. Raw data remain at `metagross:~/qwen-runs/2026-11-12-v41-prefill/row-yield/`. Final record-tree contracts and exact-PID cleanup are completed before reporting.

Root ratios use the mean of the three sequential warm-root native/client timings. The non-regression results are not attributed to faster kernels; all numerical kernels are unchanged and root host load/startup/disk state varies as recorded.


### Cleanup and reproducibility

The [M5 exact-PID audit](2026-11-14-v41-prefill-yield/results/cleanup-m5.json)
found no live task process and none of the 30 registered build, witness, arm,
harness, warmup or dispatch PIDs still present. No kills were needed. The M5
worktree, build target, temporary build directory, bundle and closed-gate pins
were removed; raw measurement records remain. The compressed pin archive was
also removed from M5 after transfer.

All four executables are archived locally at
`/Users/tito/qwen-runs/2026-11-12-v41-prefill/row-yield-local-pins/`, outside
Cargo target directories. Every archived SHA256 matches the pin manifest, so a
subsequent existing-knob campaign can reuse these executables without rebuilding.
The local Git bundle was deleted. No pushes.

All production source remains identical to measured `f33834a2c`; only records
and generated ENV documentation changed after measurement. Shared hook files
`serve/mod.rs`, `http.rs` and `transport.rs` overlap the separate GLM mailbox
work conceptually; compose that API explicitly before any landing.
