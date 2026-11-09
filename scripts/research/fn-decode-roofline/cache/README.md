# Decode cache and next-layer read-touch physics

Step 1 only: measure effective read-cache capacity/bandwidth, real next-layer
prefetch overlap, and the best executor's remaining QSA time. Production sources
remain unchanged in the committed tree. The best causal executor and reusable
GPU token loop remain authorized integration work after this decision packet.

In a clean isolated worktree:

```sh
python3 scripts/research/fn-decode-roofline/cache/prepare.py
lockf -k "$HOME/.cache/qwen-llm/locks/build.lock" nice -n 10 \
  cargo build --release -p qwen-llm --example fn_cache_probe
python3 scripts/research/fn-decode-roofline/cache/run.py \
  --model "$MODEL" --binary target/release/examples/fn_cache_probe \
  --out "$HOME/qwen-runs/fn-cache-prefetch" --mode all
python3 scripts/research/fn-decode-roofline/cache/summarize.py \
  "$HOME/qwen-runs/fn-cache-prefetch/metrics.jsonl" \
  --output "$HOME/qwen-runs/fn-cache-prefetch/summary.json"
```

Keep the commit, commands and compact metrics. The runner creates and deletes
its combined Metal source in a temporary directory. It does not seal binaries,
source copies or snapshots. Snapshots/recorded outputs exist only in RAM during
the process. Remove the example binaries after the run; ordinary build caches
are not evidence. The old decomposition/layer packets have also been reduced to
metrics and summaries; their source commits reproduce their removed bulk.

## Working-set sweep

Checked UInt32 reads cover 1/2/4/8/12/16/24/32/48/64/96/128/192/256/384/512/768/
1024 MiB working sets, four alternating-order rounds. One 256-thread group reads
64 KiB. At least 1024 groups keep occupancy from collapsing for small sets;
groups wrap over the working set, giving at least 64 MiB of logical reads per
dispatch. Each cold pass follows a separate checked 1 GiB flush. The hot result
averages sixteen consecutive serial dispatches, with every checksum verified.
Report this effective cache knee; small sets can include GPU-local caches, so
it is not an architectural SLC-size claim or a hardware DRAM counter.

## Known-weight overlap

Use the incumbent 8K fixed-input stream and the best causal executor from the
layer experiment: dependency waves, FFN scalar epilogue fusion and copy elision.
No new arithmetic or predicted expert route is introduced. Probe real adjacent
layers 2->3 (GDN->QSA), 3->4 (QSA->GDN) and 4->5 (GDN->GDN), at ordinary and
fourth-token publication phases. Input/state restoration and full-prefix work
stay outside timing; the current and next layers execute causally in one
concurrent encoder, without restoring a future layer input between them.

Read-touch packets use an argument buffer with at most 32 actual model tensor
spans. They read all selected bytes and check the complete wrapping UInt32 sum;
they never write weights. Four sets:

- `hc`: the next layer's first HC read weights;
- `early`: those HC weights and early attention/GDN projections;
- `proj_shared`: attention projections, router/shared projections and shared down;
- `all_known`: all known nonrouted weights, including both HC reads and controls.

Compare no prefetch, serial warmup, overlap in selection/recurrence, publication,
router, routed-expert activation or tail waves, and reverse-order 4 MiB chunks spread over low-bandwidth
waves. Global barriers retain the incumbent dependencies. A prefetch must finish
with its wave: do not assume resource-scoped barriers allow it to outlive that
boundary. There are no polling or inter-threadgroup spin barriers.

The primary gate is the unsampled current+next-layer GPU span, including every
read-touch dispatch and any contention. Six samples/cell alternate order.
Separate sampled-encoder diagnostics split current/next time; they are not the
performance gate. Fixed-input, private-output next-layer projection probes also
measure immediate cold/hot cache benefit, four samples/condition. Every projected
output must match its real reference. These bounds do not hide prefetch overhead
from the paired measurement. Expert prediction is optional and is not tested by
this first known-weight packet.

## QSA attribution

Build output-preserving substitutes for every dispatch in the best QSA executor,
retaining its actual dependency waves. Six cohorts: HC/glue, attention
projections, index/selection chain, attention/cache work, FFN shared/router, and
routed FFN/epilogue. Exhaust all 64 masks, three repetitions per mask, in each
of two phases: original-versus-recorded-copy and copy-versus-minimal-dispatch.
All original/copy graphs must preserve architectural boundary bytes.

The two Shapley allocations charge copy costs back and reconcile cohort work
plus command/control to the whole layer. Subtract each cohort's own useful
logical bytes / 1170 GB/s to allocate excess. Cache/occupancy interactions make
this a model allocation, not exclusive hardware timestamps. Per-dispatch
leave-one-out marginals, with the same copy correction, rank individual leaves;
they must not be summed as independent wall-time costs.

Whole-token prefetch projections use the actual 47 layer handoffs: 24 GDN->GDN,
12 GDN->QSA and 11 QSA->GDN. They sum measured pair deltas on top of the prior
best-executor estimate and remain exploratory until integration measures the
whole model; consecutive handoffs can change cache residency and contention.

## Coordination and integration boundary

Read the batching standing at `perf/fn-serve-batch` (`0bdb9ca5` at probe start):
the old few-row vehicle was killed, and no native seam was merged there yet.
The new batching lane can still be developing its seam. This probe changes no
batch/serve files or public runtime API. A future read-touch plan is model-owned
and should execute once per layer/round, not once per independent sequence.
Default/rollback and GPU-token-loop integration must preserve that seam and be
qualified on whole-model timing and signed teacher-forced distributions.

## Measured verdict, 2026-11-08

Measured source **`335e5445`**, with the resource-array shader fix in
`c121b543`. Production remains `09a3a521`. Raw command, metrics, analysis and
build/thermal logs: `~/qwen-runs/2026-11-09-fn-cache-prefetch/`, mirrored locally.
The successful packet is `second/`; `first/` failed shader compilation before
any timing. No measured sample was dropped. There are 1,152 primary paired
layer timings, 192 matvec checks, 72 sweep measurements, 768 attribution masks
and 390 per-dispatch marginal measurements. All boundary/checksum checks pass;
all 80 monitor readings and 16 in-probe thermal checks are zero.

| Working set, MiB | Effective hot GB/s |
| --- | ---: |
| 1 / 2 / 4 | 4,731 / 4,707 / 3,620 |
| 8 / 12 / 16 / 24 | 3,290 / 2,228 / 2,629 / 2,002 |
| 32 / 48 / 64 / 96 | 1,919 / 1,924 / 1,898 / 1,782 |
| 128 / 192 / 256 | 1,307 / 1,200 / 1,172 |
| 384 / 512 / 768 / 1024 | 1,172 / 1,176 / 1,179 / 1,184 |

Effective cache knee is **96–128 MiB**, versus **1.177 TB/s** large-set median.
The useful 32–64 MiB region is about **1.9 TB/s**, roughly 1.6x DRAM, rather
than the small-set 4.7 TB/s peak. This measures read reuse, not physical SLC
capacity independently of the rest of the cache hierarchy.

Positive deltas below mean less **current+next** GPU time, including prefetch.
The fixed policy is the best observed common placement across both phases;
ordinary/publication medians are weighted 3:1. These are exploratory choices
from a sweep, not independently confirmed whole-model wins.

| Handoff | Best fixed packet / overlap wave | Ordinary saved us | Publication saved us | Weighted saved us |
| --- | --- | ---: | ---: | ---: |
| GDN 2 -> QSA 3 | first HC / router | -0.938 | 1.812 | **-0.250** |
| QSA 3 -> GDN 4 | first HC / selection | 1.812 | 2.958 | **2.099** |
| GDN 4 -> GDN 5 | first HC / router | -1.000 | 0.875 | **-0.531** |

The HC packet is **13.312 MB**. Only QSA->GDN has positive mean deltas in both
order directions in both phases. Publication-wave placement saves **5.167 us**
on the quarter of tokens that publish, but its ordinary fallback loses
**6.688 us**. Selecting selection/publication by phase would save about
**2.65 us per QSA handoff**, still only about **0.029 ms/token**.

Larger packets lose even at their best placement: `early` **16–20 us**,
`proj_shared` **22–29 us**, `all_known` **35–44 us** slower per pair. Known
weight sets are 35–48 / 39–56 / 66–84 MB respectively. Immediate cache-hot
projection probes do save time: all-known cold->hot **113.96->88.77**,
**113.56->87.04**, **110.87->83.48 us** for targets 3/4/5 at ordinary entries.
But the causal pair charges loading those bytes and competing with current
work. A sampled diagnostic illustrates the trade: HC/selection at QSA->GDN
takes **5.21 us** more current-layer time and **6.67 us** less next-layer time;
the six-sample unsampled pairs above are the gate, not that one diagnostic.

Select no touch on GDN handoffs, HC/selection on the 11 QSA handoffs: projected
extra saving **0.023 ms/token**, **81.79->81.95 tok/s** with the original host
loop, or ideal GPU feedback **91.83->92.02 tok/s**. Phase-aware selection gives
the same rounded ~82/~92. Gains below 0.3%, multiple-policy selection and
cross-handoff cache interactions do not justify changing the executor design.
Do not integrate broad prefetch. The narrow QSA HC touch can remain a cheap
whole-model A/B after the already-authorized executor/token-loop work.

### QSA time remaining

The copy-charged allocation measures **272.458 us** for the best causal QSA
layer, against its **76.938 us** useful byte floor: **195.520 us excess**.
This is consistent with the previous 274.219 us layer result.

| Cohort | Useful byte floor us | Excess allocation us |
| --- | ---: | ---: |
| Command/control | 0 | **46.688** |
| HC/glue | 22.756 | **13.855** |
| Attention projections | 26.484 | **8.284** |
| Index/selection chain | 0.452 | **35.443** |
| Attention/cache | 3.590 | **61.223** |
| FFN shared/router | 6.809 | **6.351** |
| FFN routed/epilogue | 16.848 | **24.114** |
| Reconciliation | 0 | **-0.438** |

Dispatch marginals rank split merge at **19.59 us**, selector **18.95 us**,
split PV **16.04 us**, QK **13.21 us**, and routed gate/up **16.76 us**.
Pool publication is **40.08 us per publishing token**, about **10.02 us**
amortized. These are copy-corrected leave-one-out effects, not exclusive
kernel times; the raw per-dispatch table includes cache/copy-sensitive leaves
and must not be summed. In particular, selection + attention/cache + control
dominate while attention-projection excess is only 8.3 us. Warming projection
weights cannot recover most of the QSA gap with this executor.

Step 1 stops here. Preserve the **81.8/~92 tok/s** executor/feedback candidates
for whole-model integration and distribution qualification; the failed 120
target does not reject those gains. No production rollout, merge or GPU worker
is left by this packet. Frozen binaries, source copies and snapshots from this
lane's older packets were removed; compact timing/numerical evidence and source
commits remain. This retention policy supersedes those historical log claims.
