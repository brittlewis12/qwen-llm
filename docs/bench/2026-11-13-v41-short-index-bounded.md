# Short-index bounded qualification, 2026-11-13

Stage starts 15:47:46 UTC; stop and report before 17:47 UTC. Only short
masked-index qualification is active. MID is already landed. The old lane
history is preserved in `backup/v41-prefill-before-bounded-stage`; its net
changes were rebased onto rewritten main `56a1e8fe5`. Rebase conflicts in
MID declarations, generated ENV and the shared bench note were resolved
without dropping the short candidate or main evidence. Main MID excerpt
is retained as `2026-11-12-v41-prefill/MID-LANDED.md`.

The cooperative expert prototype is deferred and removed from this branch.
The previous cache-priming controller is superseded. Historical nice10
measurements and unpurged serve results remain records, not new acceptance.

## Corrected protocol

- Build with nice10; measurements and serve at nice0, verified with `ps -o nice=`.
- Same `iter` binary for native A/B, release for serve.
- Before EVERY serve arm, target-invalidate only the IQ4_NL Engram using
  the residency study's `msync(MS_SYNC|MS_INVALIDATE)` helper, and require
  zero resident pages afterward. No priming process.
- Three interleaved pairs, orders AB / BA / AB, same release binary, plan,
  cap32 and three 48K agents. Use greedy sampling in both arms to reduce
  stochastic history/I/O variation, as in the residency study. Keep every
  matched observation and root. Fixed active4, sample-workers3, fair1.
- Record live process disk-read counters, native per-call I/O, load,
  compiler activity and exact model niceness for every arm.
- Settle <=1 for 60 continuous seconds per load, allow2, abort3.
- Each lease <=15min; Python children inherit the lease via `pass_fds`.

Earlier completed boundaries:128 rows 1.01678x (2/3 positive, all8bitwise),
511 rows1.04877x (3/3 positive, all8bitwise), nice10/pressure0. New nice0
controls include150/256 target cells,127/512/1024 inactive cells and two
root shapes:49152 and49509. The latter matches the first served root and
can exercise a357-row CED tail; do not call every served root inactive.

Serve promotion requires >=3 eligible equal-row short append matches in
each pair, improved mean and median prefill and client TTFT across the
three pairs, and no material root or aggregate latency regression. Report
each order separately and all disk bytes; a failed gate removes the short
candidate. Optional wide-PV served-root recheck only if this finishes within
~90 GPU minutes; it is evidence only and cannot promote in this stage.

Before the first served arm, freeze the decision rule: require at least3
eligible prefill matches and3 observed eligible TTFT pairs in each of the3
arm pairs. Empty-EOS calls remain in the complete record; absent TTFT is
never invented. Require improved geometric mean of pair mean ratios and
pooled median for both eligible prefill and client TTFT. Root prefill and
aggregate throughput geometric means must each remain within2% of baseline.
Report every pair/order, matched call, missing-TTFT count and disk counter.

At16:42 UTC, before any served arm/model started, amend the fixed steady
window from120s to60s for all6 arms to fit the shared lease queue inside the
two-hour stage. Stop only the waiting controller by its recorded exact PID;
use fresh `bounded-short60-serve-pair*` tags. Keep the3-pair and>=3 observed
eligible-match requirements. Earlier recorded runs' first60s contain5/6
eligible matches, a planning check only. No new served result was inspected.

## Rebased build and nice0 deep gate

Source `4ba3c377b` based on rewritten main `56a1e8fe5`. Full contracts pass.
Release SHA256 `94cadc3065c78572ed92cd396d800e22b7efb4035f28c649fa6b8143f5dc370c`;
iter `524444e298fd63c481cfa3ea29e9a208df884d5ba4e218d0a3c31aa38ad2cf42`.
Builds nice10; native children record nice0. Both arms share one iter binary,
with one declared warmup plus three interleaved retained pairs.

| Rows at80K | Baseline ms | Candidate ms | Ratio | Positive pairs |
| --- | ---: | ---: | ---: | ---: |
| 150 | 389.204 | 381.099 | 1.02127x | 3/3 |
| 256 | 503.816 | 490.679 | 1.02677x | 3/3 |
| 127 inactive | 383.647 | 381.334 | 1.00607x | 2/3 |
| 512 inactive | 649.756 | 649.588 | 1.00026x | 2/3 |
| 1024 inactive | 1012.154 | 1012.723 | 0.99944x | 2/3 |

All8 signatures per cell are bitwise; all are pressure0. No speedup claim for
inactive controls. Per-pair times and raw hashes are retained in
`2026-11-12-v41-prefill/data/bounded-short-deep-nice0-summary.json`.
Root49152/49509 and purged serve gates remain pending. No default promotion.

Root controls complete at nice0:49152 is **21495.799/21647.937ms,0.99297x**;
49509 is **21982.811/21954.227ms,1.00130x**. Positive pairs1/3 and2/3;
all8 signatures in each cell are bitwise. Allowed pressure2 is retained.
No root speedup claim. `data/bounded-short-roots-nice0-summary.json` under
the earlier bench directory keeps every pair/hash. All7 rebased native
target/control cells pass the numerical gate. Purged serve remains pending.


## Current-main composition and first purged pair

Main advanced during this stage to `9877b6f25`, landing the header fold for
1..=2048-row appends (previously512..=2048). That overlaps this short band.
The branch was rebased again onto that main; generated ENV conflicts were
resolved by regeneration. The measured binaries remain pinned to source
`4ba3c377b` on main`56a1e8fe5`. These measurements cannot establish the
combined server configuration without a composition check.

Purged three-agent serve pair1, orderAB, completed16:59:37 UTC. All6 eligible
short appends have observed TTFT. Mean prefill ratio1.013515x; TTFT1.013062x;
root prefill0.995840x. Aggregate TPS87.0167/86.9833. Both arms nice0 and
pressure0; both purges returned0 and left0 resident Engram pages.

Process lifetime physical disk reads were32,785,104,896 /32,762,953,728B;
direct sampled steady-window deltas1,187,528,704 /1,168,949,248B. The
per-call native attribution method gives1,287,405,568 /2,209,792,000B because
its first-token/window boundaries differ; do not interchange those metrics.
All counters, matched calls, roots, missing-TTFT counts and thermal evidence
remain in the pair summary/comparison under the earlier bench data directory.
One pair is insufficient for promotion. Pair2BA acquired after queueing.
