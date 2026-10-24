# Exact Hierarchical V4.1 Top512 Selection

## Experiment

The batched unpruned selector previously returned to per-query global sorting
above 4096 visible keys. At 8192, the default shrinking CED plan retains 6,990
such producer queries: 83,880 legacy sort/chronological dispatches by source
census. This motivates an exact dispatch-reduction experiment, not a wall-time
attribution or a measured speedup.

For top512 and 4097..=16384 visible keys, two/four independent 4096-entry
partitions retain their best 512 score/absolute-ID pairs. One exact merge picks
the global 512 and emits chronological IDs. Both partition work and merge use
one dispatch each per selection tile. An item excluded locally already has 512
better items in its own partition, so it cannot enter the global top512.

The comparator remains numerical descending, equal-value bits ascending, then
absolute ID ascending; +0 precedes -0. Every visible NaN, including a losing
one, becomes -Inf and sticky-ORs status bit 1. Infinities remain sortable;
future scores are never read. Existing <=4K selection, pruning above 16384,
source-layer mask ownership and arithmetic tile lineage remain unchanged.
Presence of `QWEN_V41_SELECT_NO_HIER` restores incumbent long-prefix routing;
the existing per-row/split-sort/2K-only controls also retain their old routes.

## Shared Scratch Geometry

Allocation and Full/CED admission share checked tile/capacity geometry and
price the score and pair allocations separately. The pair slab is
`min(rows, tile_rows) * partitions * 512 * 8` bytes: 4/8 MiB at 512 rows.
Capacity 8448 reserves four partitions and 8 MiB even when an 8192-token
prefill dispatch uses only two. Rollback does not silently change allocation.

## Bounded Proof

Frozen candidate `7089b312` passes six pure selector/buffer-bound tests and
four self-leased GPU fixtures on both M4 and M5, with debug validation and
ordinary memory admission. The largest pre-allocation device-priced fixture
bound is 97.844 MiB, below the 128 MiB explicit-buffer ceiling. Context,
pipelines, driver overhead and process RSS are not covered by that ceiling.

Coverage includes exact legacy-GPU and independent CPU IDs around 4K/8K/16K,
ratio two, nonzero origins, ties, signed zeros, infinities, losing NaNs, future
poison, canaries, scratch reuse, unsupported routes and the pruning boundary.
The full-width fixture processes 513 rows: a 512-row P2 tile and a one-row P4
tail reuse the same slabs inside one serial encoder. A second arm borrows CED
controls at nonzero selected/candidate offsets and verifies untouched prefix,
suffix and future-mask cells. These are selector/control-binding witnesses,
not full CED transactions.

The broader forward regression exposes a newly added census assertion that
incorrectly demands block-score/keep publication from every index producer.
Only the scheduled candidate source performs that work; consumers reuse its
mask. `208124cd` corrects that test without changing execution, preserving exact
ID/state comparisons and asserting consumers do not republish. The combined
M4 checkpoint `18052d01` passes the corrected 156-test forward suite, 17 FFN
tests, 11 ignored bounded CED tests, all four selector GPU fixtures, 1,525 CLI
tests and affected release all-target checks.

## End-To-End Causal Equivalence

A single supervised fitting Maya-S 8192-token qualification completes at
candidate `7089b312`. Current candidate Full and CED match across all 257
complete vocabulary vectors (129,280 logits each) and all 56 causal regions
at every position, including 256 teacher-forced continuation steps.

An independent offline comparison against frozen incumbent `01983b7b` also
matches every complete state observation and all four arms' head hashes:
incumbent Full/CED and candidate Full/CED. Input selection, retained artifact
metadata/layout/dtypes and token-map identities match. This is not a fresh
whole-shard content hash or an artifact-quality evaluation.

The run imports one resident model and holds one KV state at a time. Minimum
sampled conservative host availability is 69.39 GiB; maximum sampled wired
plus compressor usage is 177.15 GiB. No memory bypass, unhealthy sample or
watchdog incident occurs. Samples are not continuous peaks or OS reservations.

## Performance Still Open

Ordered qualification timings remain explicitly unqualified. The selector's
source dispatch prediction and bounded census are not throughput evidence.
The separate [native-pair packet](2026-10-23-v41-ced-native-pair.md) records a
stable short-prompt CED loss and a rejected 2K timing phase shift. Diagnose that
shift on the separate incumbent-selector build, then use stable unchanged
native samples before comparing long-prompt performance. No default-policy
promotion or CED speedup is established by these correctness results.
