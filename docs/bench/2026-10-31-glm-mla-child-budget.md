# GLM MLA Child Budget: Reproducible Leaves, Unstable Controls

## Decision

The unchanged grouped query/output transforms are a real but modest profiled
cost, not the dominant MLA lever. Their summed intervals are57.680/57.800ms in
two actual-model2K diagnostics; dense attention is240.766/241.033ms.

Do not treat either run as a stable ordinary-latency denominator. Plain controls
drift by19.22% and19.46%, including the single cooldown-separated repeat. Retain
both results and stop this profiling repetition. No cause, clock/thermal
attribution, model-speedup or exclusive affected fraction follows.

This narrows the structural decision: a lifted256 graph must win through its
complete attention/dataflow economics, not a claim that grouped transforms
dominate or disappear. The fixed full-F32/Half D512 design remains
[closed](2026-10-31-glm-mla-fused-2k-negative.md).

## Unchanged Operators, New Diagnostic Scope

Frozen source is `809210e6ef0123556b41a33e0996dfca90a18d81`, following reviewed
budget lineage `05822b52`. Nineteen scoped source comparisons agree with
maintained `c87be851`; this is not whole maintained-build parity. No KDA capture,
FFN child boundary or V4 transition/support work is imported.

The new V2 packet replaces each combined MLA interval with six children on the
same retained command buffer: projection prefix, query absorption, indexer,
dense attention, value expansion and output projection. Child hooks activate
only for the fixed test recorder; inactive/plain and legacy profiling paths
do not split encoders. Following operators bind their required arguments and
resources, with no new dispatch, submission, wait or arithmetic.

The actual six-shard stock UD-Q4 model uses position0/capacity2048/rows2048,
Fast/Half, C ON, KDA Half staging OFF and qualified Q8-wide/Q5-down controls.
All rows are dense below frontier2052. Absorption256-to512 and expansion512-to256
remain grouped Q8 with64 groups and F32 activations/accumulation. Warm census
checks eleven calls of each geometry, alternating in model order, both128 threads.
Total census is1384 dispatches across29 names.

## Actual Profiled Intervals

Totals across the eleven executed MLA sites, milliseconds:

| Child | First Run | Cooldown Repeat |
| --- | ---: | ---: |
| Projection prefix | 18.306 | 18.355 |
| Query absorption | 24.238 | 24.335 |
| Indexer | 37.871 | 36.552 |
| Dense attention | 240.766 | 241.033 |
| Value expansion | 33.441 | 33.466 |
| Output projection | 40.111 | 39.814 |
| Absorption plus expansion | 57.680 | 57.800 |

First-run per-block transform pairs are5.229-5.260ms, median5.248ms; repeat
pairs are5.245-5.269ms. All547 spans/1094 timestamps, stage aggregates and block
summaries per run pass independent recomputation. Transform totals change0.209%
and dense attention0.111%. The grouped intervals are measured directly; no
synthetic attention subtraction or elevenfold leaf-gain extrapolation is used.

Encoder boundaries, cache effects and unused inherited resource bindings remain
interventions. Grouped Q8 explicitly binds its required4096-byte LDS slot in
both ordinary and sampled encoders; that does not prove every unused slot or
per-child cost is identical. Aggregate plain/profile comparisons cannot correct
each child. These are actual profiled, nonexclusive diagnostic spans.

## Controls And Limits

Each process retains weights and executes four fresh sessions: excluded
warm+census, plain1, profiled, plain2. Ordinary call wall includes its own wait,
status checks and readback; allocation, sampler preparation/resolution, hashes
and report serialization are outside it.

| Run | Plain1 ms | Profiled ms | Plain2 ms | Plain Drift |
| --- | ---: | ---: | ---: | ---: |
| First | 2003.618 | 2347.050 | 2388.809 | +19.225% |
| Repeat | 1999.715 | 2554.006 | 2388.821 | +19.458% |

The repeat starts1096 unleased seconds after the first process completes.
Profile/plain-mean ratios are1.068680/1.163945; neither isolates instrumentation
overhead in these drifting brackets. All four finite full-vocabulary endpoint
hashes per run match, also across runs. Those words are checked by the frozen
native code, not retained for independent endpoint-word rescanning.

First-run command GPU duration is2337.541ms, span sum1985.951ms and timestamp
union1947.845ms. Multiplicity excess38.107ms describes overlap, not proven
concurrent execution. Uncovered time inside the counter envelope is389.697ms;
two gaps account for384.409ms, at block33-to34 and block44-to-head. Their cause
is unresolved. Uncovered time is not an identified idle/overhead category.
The repeat has span sum1984.631ms, union1947.856ms and reported command GPU
duration2544.580ms, with overlap excess36.775ms and uncovered envelope time
596.724ms. The post-profile plain slowdown repeats under this harness; neither
this pattern nor the gap comparison identifies its cause.

The earlier [coarse budget](2026-10-31-glm-current-post-c-2k-budget.md) retains
its own accepted stable controls. Do not combine its plain denominator with
these child intervals to manufacture an exclusive fraction or speed forecast.

## Build, Admission And Execution

Canonical opt3/fat-LTO/codegen1/default-feature compilation uses configured
rustc1.99.1, SDK26.2 and Metal32023.864. All370 exact library CPU tests and strict
600 classifications/106 ignored entries pass unleased;397 dependency hashes and
the scoped source comparisons remain unchanged. No alternate SDK/PATH/profile
or diagnostic feature is introduced.

V2 explicitly prices1095MiB in each diagnostic domain,110MiB more than V1.
The ordinary512MiB reserve remains once, and opaque sampler allocations receive
no credit. Actual cold Metal is199,380,714,624 bytes; fresh Metal4,262,035,456;
CPU1,727,221,848. Transactions enclose signal sampling and factories. These
policies/quotes are not measured or formal driver peaks.

Both exact ignored tests pass in about10.5 seconds, under external90-second
bounds plus20-second termination grace. Timing leaves validation disabled with
the required unset MTL debug variable. Canonical WAIT0 leases end naturally.
Unknown command statuses retain resources and retry until known terminal; the
external process bound remains essential. No installation, service or default
changes occur, and no GPU slot remains reserved by this work.

## Frozen Evidence

| Evidence | SHA256 |
| --- | --- |
| Canonical libtest | `bf5137e1deaa5204d65bfbc75a3e278155c2bc9fb490ad38ce9438716019b46b` |
| Canonical build/CPU receipt | `18e1e7b7ef7b0c1c0d8b54faa49298bee973e9e72fdea6dbae0265925e77b6ab` |
| First actual report | `7bf444a0f3d63d5b5dfc9669c084584faf95edfd42db5a76c18c45a9b0837e02` |
| First independent recomputation | `e7cf14e50e7a8a2f59e6c06f9a665e68b85a9086219158c7e445e710ff140090` |
| Repeat actual report | `9a5c4bcbd6ab95419269ea763f9099449777be81dc5c7c9c9b880a29d72737dd` |
| Repeat independent recomputation | `ca0f3257ba7ac44ea72ef9b03b173443e1e3058a69d8b3f98dbf77012b1b26de` |
| Independent two-run comparison | `a246ded016b6366b2b9896e1ff30be448ff470163045c5302e95d0cfa52bd135` |

Private records retain raw pairs, overlaps, controls, census hashes and trusted
parent exit/lease observations. The latter are not remote attestation. A distinct
attention proposal still requires numerical/resource review and complete-operator
cost; these measurements qualify no new graph or production policy.
