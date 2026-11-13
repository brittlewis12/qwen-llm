# V4.1 attention bundle: purged root and remaining serve gate

Start19:42:05 UTC, hard report before21:42 UTC. No extension. Measurement
cutoff21:30 reserves cleanup/contracts/report time. Baseline pinned to main
9877b6f25; source67b3ebaf4. Rebuild only to restore deleted artifacts, at the
original worktree and target paths, and require every original SHA before use.
No kernel or Rust probe changes. Pair1 from the previous stage stands.

## Frozen native root protocol

48K cold CED, warmupAB then retainedBA/AB/BA, three paired observations.
Each run uses a fresh single-run invocation of the same iter binary. Before
EVERY launch: settle<=1 continuously60s; check no other resident model;
use invalidate_file_cache.py on the IQ4_NL Engram and require0 resident pages.
No separate model/weight warmup. Both flags0 for main, both1 for bundle,
MASKED_REINDEX and MID remain1. Native mode retains the production encoder
pipeline. Nice0 verified by ps; allow2/abort3. Per-run load, process I/O and
exposed Engram prepare wall time are recorded; also retain completed-submit
encode time so the earlier combined host component can be compared.

The pinned probe emits full state/following-step hashes only in its two-run
mode. Fresh single-run processes guarantee purge before every timed run;
check each endpoint digest against the prior golden digest. The existing
full-state/following-step bitwise witness from the exact same binary stands.
All eight runs and their timing/prep/load values remain in the record.
Native decision uses the existing ratio of retained mean A/B wall times,
>=0.98x; report geometric pair ratio too, without changing the decision rule.
Try one<=15m hold; checkpoint at pair boundaries. If remaining pairs cannot
fit the envelope, use one lease per remaining pair, preserving completed runs.

## Frozen served protocol

Pair1AB remains exactly as recorded. Run fresh pair2BA and pair3AB; the old
incomplete BA packet remains archival and is not substituted. Each arm has
its OWN<=15m lease and deadline, using --only-arm on the existing driver.
Symmetric paired4K model warmup, both GPU witnesses, zero-page Engram purge,
fresh server, greedy/cap32,3x48K agents,60s window,15s ramp, active4,
sample-workers3/fair1, nice0 and existing thermal protocol are unchanged.
No redesign of warming or measurement. Refuse overwriting arm artifacts;
second-arm metadata, flags, plan hash and ordering must match first arm.

Across the three complete pairs: geomean TPS B/A>=0.98, root prefill A/B>=0.98;
short(actual128..511) and1K-class(actual512..2000) prefill AND client TTFT
geomeans each>=1.00. Retain every matched equal-new-row completed offered
call, counts/null TTFT, all roots, compiler overlap, load and per-arm disk
bytes. At least one observed match per bin per pair, as frozen previously.

Pass plus purged native root>=0.98 => ready to land, both default-on with
independent0 rollbacks. A purged root failure drops wide PV. Short stays
on only if its own150/256 cells still hold; no broader lever exploration.
For a fresh short-only check with the pinned probe, PV=0/SHORT=1 and scoped
MID false/true at80K with sub512 rows is exactly main versus short scoring:
MID affects only the depth eligibility here, while the main512..2048 band is
inactive. Existing paired mode then supplies all8 strict complete signatures.
Engram-only main landing requires rebase/contracts, no attention remeasurement.
If queue/budget prevents completion, report completed work without extending.

M5 outputs under qwen-runs/2026-11-12-v41-prefill/attention-finish; restored
build target uses the old attention-bundle/target path for byte reproduction.
Every build/model load leased, holds<=15m, Python children pass_fds. Raw
records survive; remove target, worktree and bundles at stage end. No pushes.
