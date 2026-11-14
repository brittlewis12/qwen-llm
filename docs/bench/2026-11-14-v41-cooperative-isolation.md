# V4.1 cooperative experts: real-population stage gate

Bounded stage starts 2026-11-13 23:56:18 UTC; report by 2026-11-14 01:56.
No changes to `perf/v41-prefill`; the landing lane handles that branch.

## Candidates and frozen decision

Compare incumbent A independently against packed cooperative B and direct
F32-gather cooperative C. Both remain default-off. Use real model traffic
at 2048 rows around 48K and 256/1024-row appends at 80K. Mean routed rows
per expert are 32, 4 and 16; retain all per-layer counts and their actual
distributions instead of substituting uniform rows. No complete existing
route vectors were found in the checked reports, so a dispatch-only observer
copies current GPU counts before their scratch is reused and reads them
after forward completion. It is disabled for native wall measurements.

At 256 rows the actual incumbent is rectangular grouped experts, not the
sorted 1024..4096 path. Experimental eligibility extends to 256 rows only
for the selected candidate. A remains unchanged.

One iter binary; scoped paired controls leave attention and prefix creation
identical. Warmup pair plus three retained interleaved pairs per candidate
and cell, actual nice 0. Require all eight logit/state/following-step
signatures bitwise, no counter overflow or missing timestamps, candidate
active in every eligible routed layer, baseline cooperative count zero,
and unchanged route vectors across paired runs. Any inexact candidate is
rejected before considering its timings.

**Gate: >=1.25x combined routed GU + down GPU isolation at every requested
population.** GU includes its required SwiGLU/weight/narrow epilogue, whether
fused or separate. Report GU and down separately, and expert input conversion
and plan/pack separately. Candidate C's gather/convert is included inside its
GU time. Input conversion retained for shared FFN is not counted as removed.
The audit's entire 71.4557ms includes indexer copies and is not a claimed
saving. Interleaved arithmetic-mean stage sums determine the ratio; retain
all allowed pressure-2 observations. No isolated dispatch sum is a native
wall claim. If neither candidate passes all cells, remove candidate code
and retain the record. Winners alone proceed to three paired native 48K
root/root-chunk and 1K@80K cells; serve is a later stage.

## Direct F32 input

C reads original normalized F32 through the existing route IDs into a
threadgroup Half A slab. It preserves the old GPU float-to-half conversion,
K64 update order, both F32 epilogue storage rounding points, and Half input
to down. No requantization or persistent GPU allocation. Maximum dynamic
shared operands are 27,648 bytes (18,432 weights +9,216 A); the actual PSO
static plus dynamic resource check remains enforced. Sorted pack is skipped.
Only an otherwise dead routed-input half copy can disappear; a copy needed
by the shared FFN remains. The full MIT notice remains in the shader header.

The existing GPU differential gains a third mode, F32 input perturbations
and half-boundary/subnormal values. Reference input narrowing uses the
actual GPU scatter rather than CPU approximation. Expected: 90 cases per
mode, 270 total and 109,572,096 strict inner/down components, plus existing
packing/alias/guard and binding refusals. No new ignored tests are added.
Compiler checks first caught the diagnostic observer's missing MTLBuffer
trait and String/MetalError boundary; those were fixed before any M5 run.

## Protocol and artifacts

Every M5 build/model load uses the waiting Metal lease with <=15m holds;
Python children inherit the lease FD explicitly. Builds nice 10; measured
processes nice 0 checked with ps. Settle continuously <=1 for 60s, allow 2,
abort at 3. Check resident models and record Ghost/load/thermal. Artifacts
under `~/qwen-runs/2026-11-12-v41-prefill/cooperative-stage/`, never temporary
system output directories. Copy iter and test executables outside target;
keep them until their qualification gate closes. No push or merge.
