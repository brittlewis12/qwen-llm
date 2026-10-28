# V4.1 Long-Selection Short-Tail Correctness

## Result

Long-context append and final-chunk scratch must hold at least one complete
selection score row before the executor skips per-row selection. If it cannot,
retain the existing per-row ILP/radix path. Fitting long batches keep their
existing tile sizes. No allocation sizes, memory prices, kernels, scoring
arithmetic, environment defaults, or ordinary unpruned selectors change.

This is a correctness repair, not a performance or model-quality result.

## Regression

The long-batch path introduced in `9abf844e` uses a scratch slab containing
`min(chunk_rows, tile_rows) * min(capacity, 16384)` F32 elements at the released
candidate geometry. It requires one full visible-key score row, plus block
scores at the candidate source. Previously, routing selected that path without
checking whether even one row fit.

For a 32,768-token prefix and two-token append, the slab contains 32,768
elements. A ratio-one candidate source needs 32,770 key scores and 4,097 block
scores: 36,867 elements. The former routing skipped per-row selection and then
returned `long selection score width`. The production request guard leaves an
unsuccessful chunk poisoned. A cold 32,770-token prefill with 4K chunks reaches
the same two-row tail. This original request-level failure is source-established;
no old full-model request was run to reproduce it.

The shared `long_selection_tile` preflight now checks this complete-row extent
before deciding whether to skip per-row selection. Its checked addition also
rejects extent overflow. Only the candidate source pays for block-score rows;
consumers need just the key-score row. The fallback uses already-allocated
workspace scratch, the incumbent F32 ILP scorer, and radix key selection.
Candidate publication uses the existing per-row block-sort/keep chain.

## GPU Witness

The new ignored, self-leased witness is
`deepseek_v41::forward::attention_gpu::tests::query_batch::selection_tail::long_selection_short_tails_match_prepared_per_row_and_continue`.

It uses the production **32-head by 128-dimension indexer**, top-k 512,
8-key candidate blocks, and 2,048 kept blocks. Hidden/attention dimensions and
weights are small synthetic fixtures, not the full model. Its three-layer
schedule contains a ratio-two owner, a ratio-one candidate-source owner, and a
masked index-producing follower sharing the latter cache. Prefix state is
seeded directly; no long prefix or model import is performed.

| Prefix | Rows | Score Elements | Ratio-Two Owner | Candidate Source | Follower |
| ---: | ---: | ---: | --- | --- | --- |
| 32768 | 2 | 32768 | batch, tile 1 | per-row | per-row |
| 32769 | 2 | 32768 | batch, tile 1 | per-row | per-row |
| 32768 | 3 | 49152 | batch, tiles 2+1 | batch, tile 1 | batch, tile 1 |
| 65536 | 2 | 32768 | per-row | per-row | per-row |
| 65536 | 5 | 81920 | batch, tiles 2+2+1 | batch, tile 1 | batch, tile 1 |

Both arms use identical batched prepared projections and attention/output
arithmetic. The reference explicitly encodes the prepared serial transitions
per row; the candidate invokes production routing. Assertions check actual ILP,
block-score, radix and preparation dispatch dimensions, zero serial Metal buffer
allocations, bitwise outputs and selected IDs, visible candidate masks,
authoritative cache reuse, pending pooling operands, causal regions, clean
status, and cursor completion. An unmasked selection from the follower's actual
scores must contain an otherwise-winning excluded key. One subsequent ordinary
attention decode checks exact matching output/control/state and readiness.

The odd prefix exercises pending ratio-two ordering; the three- and five-row
cases exercise partial final subtiles. CPU coverage additionally distinguishes
source-only undercapacity: width 29,127 plus block scores exactly fits 32,768
elements; width 29,128 needs 32,769 at the source but still fits a consumer.
The CPU matrix covers 32K through 512K prefixes and retains large-chunk batching.

## Verification

- Qualified source: `fb4fadb113d2ad988f86c3da7c92afa864ff1e76`;
  tree `eb4c9cdf64b599260a3bd6d2aec82094fc68ba92`, based on main `1cef0778`.
- M5 Ultra, Rust 1.99.1, SDK 27.0, canonical release, no optional features or
  Rustflags. Fresh isolated target, CPU build lock, frozen executable outside
  the target, clean-source and binary-hash checks: **460.250 seconds, no GPU
  lease during compilation or CPU checks**.
- Frozen libtest SHA256:
  `b7cb28e323aff69b3c09024423aac6de92337a44c6f38a0e82421db1dfb0beef`.
- **340 classified CPU tests pass**; discovery matches 538 maintained entries
  and 94 ignored witnesses out of 2,877 total library tests. Discovery does not
  classify or authorize GPU execution.
- **Four GPU tests pass, zero failed/ignored**, in the first actual GPU packet:
  the new short-tail witness and all three maintained radix/ILP tests. DEBUG
  validation is enabled. One canonical outer lease is continuously inherited
  on FD 3 through the supervised child. Libtest reports 0.33 seconds; the
  checked controller span is 0.378 seconds, not an independently measured total
  lease duration or a performance sample.
- The short-tail witness's largest priced Metal allocation total is 256 MiB;
  adding its 256 MiB Metal reserve and 512 MiB CPU reserve gives 1 GiB, below its
  2 GiB admission cap. This is a conservative admission price, not measured
  RSS, wired memory, or a formal driver-allocation bound.
- 33 maintained pilot and 14 reference Python tests, inventory corruption
  controls, formatting/whitespace and ENV polarity checks pass. Thirty private
  offline controls cover provenance rejection and actual owned CPU-child
  signal cancellation, descriptor retention, heartbeat draining and reaping.

Two local M4 CPU attempts are retained, not filtered: both stop at the unchanged
4 GiB available-memory health floor with normal OS pressure. The first also
reports a test-only ambiguous integer receiver, repaired by the explicit
`usize` annotation in `fb4fadb1`. No GPU test ran in either attempt. Compilation
then moved to metagross rather than relaxing the guard or interrupting other
work. No owned local compiler survivor remained at the later process check.

Adversarial `cx` session `01a17fc9-0f07-7d72-b508-231f6222db52` reviewed the
production fix, actual regression and private runner. Its two private-runner
blockers were repaired before GPU execution: acceptance now follows provenance
postflight, and SIGINT/SIGTERM retain supervised child ownership through drain.
The 90-second execution budget triggers cancellation; it is not a hard exit
ceiling if a child survives bounded INT/TERM. Such a child keeps the controller
and inherited descriptor until reaped; SIGKILL/crash recovery is not guaranteed.
The final adversarial result review verifies all 45 packet hashes, source/binary
identity, actual test names/counts and lease/validation receipts, with no P1/P2
blocker or overreach in this narrow qualification claim.

Private evidence is retained at
`~/Library/Application Support/qwen-llm/benchmarks/v41-selection-tail-2026-10-27/`,
with the remote source, binary and 45-file receipt closure copied to `m5-final/`.
`m5-final.sha256.json` has SHA256
`5614c1089fae833623c9adb3d6707278d80cc450e56e8694f6c6a126fb1bd9c6`.
The remote packet is `metagross:~/qwen-runs/2026-10-27-v41-selection-tail/`.

## Limits

These checks qualify the narrow routing repair at the fixed synthetic boundary
cases, not arbitrary-context Full/CED equivalence, complete logits, teacher-forced
model continuation, sustained throughput, or a newer branch composition. The
existing ILP-versus-old-scorer test uses its stated F32 tolerance; it does not
establish old-model bitwise equivalence. This repair does not change the
long-context scorer's numerical contract or inherit qualification of unlanded
Engram/submit-ahead work. No model campaign, installation, push or deployment
is part of this packet.
