# V4.1 cooperative experts: real-population stage gate

**FAIL: both variants are withdrawn from active source.** Neither reaches
1.25x in any cell. No native or serve run follows.


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

## Measured results and independent audit

Functional passes 270 cases / **109,572,096 strict components**, including
F32 gather and actual H5120/F2304, plus binding refusals. Six model cells
retain all eight bitwise logit/state/following-step signatures each:
**48 full signatures, 18 retained timing pairs**. No overflow, missing GPU
timestamp, hidden fallback or route-count change is accepted. Every measured
PID is nice 0. All allowed pressure-2 observations remain in the raw evidence.

The independent local reducer reproduces all primary gate values, components,
integer GPU times, quality and populations exactly after JSON key normalization.
Derived inclusive ratios differ by at most 2e-16 between Python versions; their
presentation-only aggregation is checked within 1e-14. No model-output or gate
tolerance is changed. Ratios are incumbent/candidate.

| Rows / prefix | Variant | GU baseline / candidate ms | Down baseline / candidate ms | Combined ratio | Inclusive ratio |
|---|---|---:|---:|---:|---:|
| 2048 / 47104 | cooperative-experts | 399.499 / 622.105 | 199.719 / 281.860 | 0.662878x | 0.677382x |
| 256 / 81920 | cooperative-experts | 80.681 / 235.540 | 163.263 / 104.638 | 0.717106x | 0.705615x |
| 1024 / 81920 | cooperative-experts | 278.742 / 424.683 | 139.649 / 191.860 | 0.678607x | 0.687189x |
| 2048 / 47104 | cooperative-gather | 399.838 / 742.972 | 199.709 / 282.622 | 0.584585x | 0.621559x |
| 256 / 81920 | cooperative-gather | 82.816 / 231.712 | 162.273 / 104.805 | 0.728312x | 0.725387x |
| 1024 / 81920 | cooperative-gather | 279.017 / 471.072 | 139.917 / 190.942 | 0.632817x | 0.656827x |

The 2048-row cell spans 47104..49151, ending at 48K. The 256/1024-row cells
start at 81920. All use real weights, activations and routes from the existing
natural code corpus; no synthetic row distribution is substituted. Counts
cover all 40 routed layers, 384 experts each, and are identical across arms
and variants for a given traffic cell.

| Rows | Mean | Median | p95 | Maximum | Empty / 15360 observations | M32 / M64 panels |
|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 32 | 7 | 149 | 1695 | 3213 | 10062 / 6680 |
| 256 | 4 | 1 | 18 | 172 | 7369 | 7749 / 354 |
| 1024 | 16 | 3 | 75 | 847 | 4363 | 9734 / 2894 |

The actual distributions are strongly skewed. Mean rows/expert alone did
not predict the outcome. Packed root ratios are 0.662164/0.663327/0.663143x;
all retained pairs regress. These are isolated dispatch GPU sums, not native
wall claims.

## Attributable movement

| Rows | Variant | Input copy baseline / candidate ms | Sorted pack baseline / candidate ms |
|---:|---|---:|---:|
| 2048 | cooperative-experts | 5.731 / 5.672 | 34.964 / 35.705 |
| 256 | cooperative-experts | 0.320 / 0.318 | 0.000 / 4.115 |
| 1024 | cooperative-experts | 1.162 / 1.167 | 14.997 / 15.104 |
| 2048 | cooperative-gather | 5.695 / 5.692 | 35.098 / 0.000 |
| 256 | cooperative-gather | 0.312 / 0.312 | 0.000 / 0.000 |
| 1024 | cooperative-gather | 1.164 / 1.150 | 15.076 / 0.000 |

Gather removes sorted-pack dispatches but retains the input Half copy needed
by the shared FFN. Root attribution is about 5.7ms input +35ms pack, not
the full 71.4557ms scatter+pack audit total, which includes indexer/other
consumers. The slower cooperative math overwhelms movement savings. At 256
the rectangular incumbent does not pack; packed cooperative adds that work.
The inclusive ratios expose both effects. No whole-audit saving is claimed.

## Sources and withdrawal

Measured source `0534619d53d6b9fd4a6812e6e5cb3b8d7b134220`; all arms use one
archived iter executable. The test profile supplies functional comparisons.
[summary.json](2026-11-14-v41-cooperative-isolation-data/summary.json) retains
all pairs, components, raw-file hashes and executable/metallib identities.
[populations.json](2026-11-14-v41-cooperative-isolation-data/populations.json)
retains every route count. Full raw data remains at
`metagross:~/qwen-runs/2026-11-12-v41-prefill/cooperative-stage/`.

Main landed the attention bundle and witness at `0b0ea5f12`. The cooperative
branch rebased without conflicts; main's production source matched the
qualified attention base. Measured source is preserved at
`backup/v41-cooperative-stage-measured`. Both candidate families, selectors,
temporary route observer, expanded 256-row eligibility, test-only candidate
extensions and scoped identity exceptions are removed by restoring landed
main's source. ENV is regenerated without the two experimental knobs.

No new ignored tests, requantization, approximation, native timing, serve
timing, push or merge. Final contracts and M5 cleanup are in the closing raw
logs. Both candidate gates are closed, so their archived executables can be
deleted after records are complete. A later lever may retain incumbent math
and reduce pack lookup/movement; no such implementation is included here.
