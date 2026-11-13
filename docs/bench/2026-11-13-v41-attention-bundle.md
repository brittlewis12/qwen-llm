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


CPU contracts pass source67b3ebaf4. Shader compiler flags are profile-independent;
the fixture and release build use the same tensor shader functions. Before
any served arm, additionally enforce pressure<3 throughout model-free warmup,
record its maximum and verify nice0 via ps. This guard changes only Python;
binaries remain pinned to67b3ebaf4 and one SHA per profile. No served result
has been inspected; thresholds and timing windows remain frozen.


## M5 functional witness

Builds67b3ebaf4 complete, same source for each profile. Iter SHA256
`19d9adf78d3a53f7f8d209386a1c06e0d4da10162524cdcba123cabfc1facaef`;
release`5c538677b90c074f36c40ec44b9616942e71ed392ff3368154dbb72a153d575f`.
Python-only warmup guard27caf33d8 is applied after all builds finish.
Both existing GPU witnesses pass at nice0 with Metal API validation and
required-Metal mode after60s settled<=1; observed pressure0. Wide PV covers
15 cases /13,434,880 strictly compared normalized/inverse-RoPE components,
including production query aliasing and guards. The masked-MMA witness
checks allowed-score bits, selection, offsets, tails, all/none masks and guards.
`2026-11-13-v41-attention-bundle/data/functional-summary.json` keeps raw hashes.

Build commands use nice-n10. The manifest's immediate post-Popen `nice` sample
for build commands occurs before exec of the nice wrapper; it records0 and
is not a ps witness of cargo. Functional and native children inherit nice0;
serve models/warmups additionally verify ps-o nice=. No build timing claim.
The warmup CPU smoke checks both witnesses, inherited lease FD, debug-layer
removal, nice0 and pressure3 refusal. Archived reducer replay preserves every
matched row and produces both predeclared append bins. Native cells pending.


Main and tensor metallib hashes are identical across iter/release/test;
tensor SHA256`43ae770545fa60f1f51d2bb3220a93c0aaf3bfc13d382a99ea2345920eaf184c`.
Thus model-free warmup dispatches the exact shader binaries used by serving.
The restored wide shader and extended fixture are byte-identical to archive
659974115. Per-profile sizes/hashes remain in data/shader-profile-hashes.json.


## Symmetric weight-residency warmup amendment, before any served arm

At18:16 UTC no served arm has begun. The discarded native root warmup pair
is45.701/31.422s and bitwise, showing a possible weight-residency/startup
confound larger than the candidate. Add identical paired4K AB model warmup
before EVERY served arm, using the same iter binary with both scoped policies.
Require2 strict complete signatures, nice0 via ps, and thermal60s<=1 before
that load; allow2/abort3. Exit the warmup model, then execute both GPU fixtures,
then purge Engram to0 pages, then load the fresh measured server after its
normal60s thermal settle. No KV/history survives into a measured process.
This warms GGUF weights symmetrically while preserving the requested cold
Engram condition. Record warmup PID, hash, thermal and disk bytes separately
from measured-server physical reads. Main pair budget860s plus outer885s
remains below the15-minute lease; all windows, gate thresholds, bins and
three-pair orders are unchanged. Do not attribute the discarded warmup ratio
to attention speed. If resource time is insufficient, report incomplete.


## Native cold packet, completed18:18:40 UTC

One iter SHA19d9adf78d3a; nice0, warmup plus3 retained interleaved pairs;
all8 complete signatures per cell bitwise. Keep every retained pair.

| Cell | Baseline ms | Bundle ms | Ratio | Positive pairs | Pressure max |
| --- | ---: | ---: | ---: | ---: | ---: |
| 49152 | 26848.591 | 30309.481 | 0.885815x | 1/3 | 1 |
| 4096 | 2855.960 | 2797.941 | 1.020736x | 2/3 | 0 |
| 8192 | 4009.679 | 3906.816 | 1.026329x | 2/3 | 0 |

The48K root regresses in this native packet and is not hidden by averaging
across cells. Retained root pairs A/B ms:21480.729/30623.624,
28663.776/30132.464,30401.268/30172.355; ratios0.70144/0.95126/1.00759.
Its exposed Engram preparation+encode ratio is0.70755x; that component's
large movement is recorded without assigning a cause. Native root is not
purged; the declared purged served-root non-regression gate remains decisive.
No rerun or sub-bisection. Deep150/256/1K packet queued.

Python warmup7ed354ad9 is installed between native packets, before any
served process; Rust/shaders are unchanged. The weight-warmup CPU smoke
passes inherited FD, paired4096 command, ps nice0,2 exact signatures, live
PID/I/O evidence and pressure3 refusal. No served result has been inspected.


## Native deep packet, completed18:33:17 UTC

Same iter19d9adf78d3a, nice0, one warmup plus3 retained interleaved pairs.
All8 signatures per cell bitwise; all six requested cells are now complete.

| Rows at80K | Baseline ms | Bundle ms | Ratio | Positive pairs | Pressure max |
| --- | ---: | ---: | ---: | ---: | ---: |
| 150 | 406.423 | 377.757 | 1.075886x | 3/3 | 0 |
| 256 | 520.083 | 498.179 | 1.043968x | 3/3 | 1 |
| 1024 | 1060.764 | 1050.128 | 1.010128x | 3/3 | 1 |

All raw hashes are verified against local copies. First150 baseline warmup
was1429.194ms versus380.895ms bundle; it is excluded as predeclared, retained
in raw data. No native reruns. The negative48K root is retained alongside
positive append/pp cells. Serve pair1AB queued18:33:17 behind GLM.
Contracts with the full symmetric-weight-warmup driver pass sourcee51141aba,
tree645d2d7c41d9979f76d86e2d6db01e4080a5625a. Main remains9877b6f25.


## Purged serve pair1AB, complete18:56:05 UTC

Short prefill1.053301x and TTFT1.032870x,6/6 observed;1K-class prefill
1.045457x and TTFT1.058349x,3/3 observed. Root prefill1.166676x. Aggregate
TPS87.35/88.3167. Root TTFT A33.049/32.029/29.379s;
B28.110/27.191/25.680s. All six roots retained.

Both arms: paired4K warmup2 exact signatures, full+masked index and
original+wide PV witnesses,15 strict wide cases; warm/model ps nice0;
purge0 pages; maximum measured pressure1/0. Main/tensor binaries identical
across fixture/native/release. A/B measured lifetime sampled physical reads
52,267,646,976 /33,156,030,464B; warmup process159,335,890,944 /496,418,816B.
The root gain coincides with a substantial I/O difference despite equal
Engram residency at start; do not attribute the entire difference to attention.
Arma sampled steady-window read delta3,161,849,856B; load-after[15.86181640625, 20.939453125, 28.1015625].
Armb sampled steady-window read delta1,464,713,216B; load-after[12.43994140625, 16.3271484375, 23.3330078125].

Counters are max live values sampled every0.5s; full per-call and raw sample
records remain. Complete comparison, calls, roots, nullable TTFT and all
protocol proofs retained. No gate decision from one pair.

## Authorized scheduling grace

At18:58:40 UTC, pair2 was still queued with no warmup/model loaded. Stop
verified exact controller18008, reap exact lease child67196; preserve the
old queue log and every completed pair1 artifact. Resume only remaining
pairs with a19:44 measurement cutoff, leaving8 minutes for cleanup/report
before the user's19:52 hard stop. Target remains19:37. Same release/iter,
plans, windows, ordering and metric thresholds; no partial/complete arm is
rerun or overwritten. `grace-reschedule.json` preserves the exact-PID facts.
