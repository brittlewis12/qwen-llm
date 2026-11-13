# V4.1 bitwise attention bundle qualification, 2026-11-13

Stage starts17:37:08 UTC. Target report19:37, hard stop19:52. Measurement
cutoff19:37 leaves cleanup/report time. Base main9877b6f25 with full header
fold. Reapply only short masked index from backup/v41-short-bounded-measured
and wide PV from its archived source659974115; retain mlx-serve MIT notice.
Both default-on with separate QWEN_V41_MASKED_REINDEX_SHORT=0 and
QWEN_V41_ATTN_PV_WIDE=0 rollbacks. No payload allocation or precision change.

## Frozen gates, before the first arm

Functional: existing masked/full-MMA strict score/selection/guard GPU test
and existing tensor64 sink differential, extended with15 strict wide PV and
inverse-RoPE cases. Required Metal/debug first witness; no new ignored tests.

Native: one iter binary; bundle(false) is current main with both flags off,
bundle(true) both on, MID unchanged. Common production prefix, declared first
warmup pair plus3 interleaved retained pairs. Cells:48K cold CED root,
150/256/1024 rows at81920 depth, cold4096 and8192. Existing probe uses CED
for depth0 rows>4096;4096 is direct Full. All8 complete endpoint/state/next
teacher-forced signatures per cell must be bitwise. Report all cells.

Serve: same release binary, greedy/cap32, three48K agents, fixed60s windows,
seed20261112, active4/sampleworkers3/fair1, ordersAB/BA/AB. A sets both flags0;
B sets both1. Before EVERY arm, run both existing GPU differentials at nice0
without the debug layer: they dispatch full/masked index and original/wide PV,
normalization and inverse-RoPE, including actual query-storage aliasing.
Require both test successes and all15 wide cases. This identical model-free
warmup primes both kernel families in both arms. Exit warmup, then purge
Engram with msync(MS_SYNC|MS_INVALIDATE), require0 resident pages, then launch
fresh server. All three root requests retained. Warm roots/ramp further warm
the active serving path before the measured steady window. Record compiler
activity and any overlap rather than discarding slow observations.

For this bitwise bundle, serve acceptance is non-regression. Across3 pairs,
geometric means of pair ratios must satisfy aggregate TPS>=0.98 and root
prefill>=0.98; short and1K-class append prefill AND client TTFT each>=1.00.
Bins use actual prefill rows128..511 and512..2000 respectively, equal-new-row
matched completed offered-in-window calls, no carry-in or CED. Require at
least one matched and one observed TTFT in each bin per pair; report every
count, missing TTFT and matched row. Null TTFT is not fabricated. No filtering
by order, timing, I/O, load or compiler activity. Per-arm process lifetime
physical-read counters and complete native/per-call/window evidence retained.

Nice0 verified by ps for models/warmups; builds nice10. Settle<=1 for60s before
each model load and after root warming, allow2/abort3. All M5 builds/loads
leased<=15m; Python children inherit pass_fds. Ghost load recorded. No other
model process may be resident. If root gate fails again, drop wide PV and
retain short evidence; no sub-bisection in this stage. Engram-only main
landings need rebase/contracts, no new attention qualification. No pushes.
