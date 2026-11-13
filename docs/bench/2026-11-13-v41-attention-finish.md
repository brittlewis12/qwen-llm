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


## Executable pin conflict: no new model measurements

Contracts pass cf6652fd1, treee7b03964c79d423c83dbcfb14fc0286e22c76809.
The restored source is exactly67b3ebaf4, original worktree and target paths,
locked dependencies and matching profiles. Rebuild completes but fails the
exact executable SHA guard before any model or GPU witness loads:

| Artifact | Required SHA prefix | Rebuilt SHA prefix |
| --- | --- | --- |
| iter | 19d9adf78d3a | 1fa66184f9a1 |
| release | 5c538677b90c | 78131445f024 |
| test fixture | a83fa4a6639a | b253930bc372 |
| tokenizer | eb94be08eb27 | eb94be08eb27 |

Every main/research/tensor metallib across iter/release/test is byte-identical
to the previous pinned build. Executable differences are not localized;
identical GPU libraries alone do not establish identical host executables.
No mbx tool/cache is configured on the M5. Bounded checks found no executable
archives in the recorded run directories. The original executables were
removed with the prior build output; no separate copies were retained.
A text-input question requests an archive location. Do not silently replace
the requested pins with new hashes or use elapsed waiting as authorization.
No native-root run or served arm has started; pair1 remains unchanged.

The one-arm helper preserves warmup/purge/window semantics, checks flags,
order, sampling, window length, warm reply cap and plan hash, and refuses
any arm overwrite. CPU mock tests pass BA accumulation and duplicate-arm
refusal. A separate model-free mock verifies eight one-run processes with
purge immediately before each launch, inherited lease FD, ps nice0, root
shape/digest checks, prep/encode fields and pair checkpoints. Synthetic
scratch data is removed and is not represented as performance evidence.

Complete expected/actual hashes, source/compiler details and rebuilt Mach-O
section digests are in data/pinned-binary-conflict.json. Existing full-state
bitwise/native/served evidence is unchanged. No performance or readiness
verdict can be added without the originals or an explicit pin clarification.
