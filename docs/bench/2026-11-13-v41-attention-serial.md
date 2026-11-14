# V4.1 serial attention: depth decomposition and compact-block experiment

Stage started 2026-11-14 01:07 UTC; bounded to two hours. File A GSQ-RCO
IQ2_XS with disk-streamed IQ4_NL Engram, M5 Ultra. This stage concerns the
`prefill_attention_serial` interval, not the sparse-attention QK/PV tiles.

## Measurement and provenance

- Baseline source `3fcffa721` on main `a77d48b97`, archived locally as
  `backup/v41-serial-measured`. Subsequent rebase onto main `f8398b4b4`
  incorporates the cooperative-experts negative record; no shader change.
  The generated ENV header conflict was resolved by regeneration.
- Baseline iter executable SHA256
  `8c93ead862d8eb92f271fd8b19be0ab7aa8c9f10b74b0623d3b4708a45471162`;
  test-profile executable
  `62f70bf2b958041535a615a47ceebfdfca03987a05fde99161f4603d749c948d`.
  Iter/test metallibs agree: main
  `d52890108ebc1e3ab428b6d65d5a586158415efe1d697a720aa3f10395b95836`,
  tensor `43ae770545fa60f1f51d2bb3220a93c0aaf3bfc13d382a99ea2345920eaf184c`,
  research `be63c8ae9332529da45c21328333191834bd767917f5f9bc55d94eb3702b557f`.
- `v41_prefill_census`: uncycled natural-code prefix at exactly 81,920 or
  131,072 tokens, then 256 or 1,024 Full-append tokens. One warmup plus two
  profiled repetitions per baseline cell. All three logits digests agree
  bitwise in every cell. Capacity rounds `(depth + rows + 1)` up to 4K, with
  a 98,304 minimum. The example now supports the requested 128K depths.
- Separate dispatch and stage packets use the same archived iter executable.
  Dispatch mode disables the CED encoder pipeline and samples separate
  encoders. Its sum is GPU service attribution, not native wall or an additive
  Amdahl estimate. Stage mode retains production scheduling; GPU intervals
  can overlap. No subtraction of these different modes is called a gap.
- Every build/load has a waiting lease, each hold at most 15 minutes.
  Builds use nice 10; every measured/test child is verified nice 0 with
  `ps -o nice=` as well as `getpriority`. Drivers forward the lease FD with
  `pass_fds`, check for other resident models, settle at pressure <=1 for
  60 seconds, allow 2 and abort at 3. Load and background processes are in
  the monitors. Pinned executables live outside `target/` until gate closure.
- Raw: `metagross:~/qwen-runs/2026-11-12-v41-prefill/attention-serial/`.
  `pins.json`, `monitor.jsonl`, `dispatch-{depth}-{rows}.jsonl`,
  `stages-{depth}-{rows}.jsonl`; compact experiment under `compact/`.

## Baseline decomposition

Means below exclude the warmup. The small baseline stage packet is a census,
not a paired performance comparison with the earlier depth-attribution probe.
Its 1K serial intervals reproduce that probe closely (122.73/191.32 versus
122.81/191.43 ms). The 256-row intervals are lower (40.18/61.15 versus
54.07/86.23 ms); both records stand, with different probe/capacity protocols.

| depth / rows | serial stage ms | isolated service sum ms | append wall ms | host prepare / encode ms |
|---|---:|---:|---:|---:|
| 81,920 / 256 | 40.180 | 46.129 | 444.846 | 7.604 / 3.880 |
| 81,920 / 1,024 | 122.729 | 128.676 | 953.023 | 6.956 / 2.141 |
| 131,072 / 256 | 61.146 | 69.362 | 462.204 | 7.525 / 3.924 |
| 131,072 / 1,024 | 191.317 | 198.442 | 1007.323 | 6.790 / 2.151 |

Host preparation and command encode are inclusive CPU intervals outside the
sampled serial interval; they are not serial-span sub-kernels. There is no
CPU readback, host top-k, per-token command commit or host wait within
`encode_rows_serial`. The active global layers are 2/8/14/20/24/28/32/36.
Other layers advance state without index GPU work.

Each entry is summed isolated service, with dispatch count in parentheses.

| kernel | 80K / 256 ms | 80K / 1K ms | 128K / 256 ms | 128K / 1K ms |
|---|---:|---:|---:|---:|
| index_heads_rows_mma | 15.600 (15) | 62.641 (30) | 25.627 (25) | 100.246 (46) |
| index_heads_rows_mma_masked | 11.446 (24) | 45.369 (44) | 15.859 (36) | 62.543 (68) |
| select_radix_rows | 5.273 (45) | 16.337 (86) | 12.508 (71) | 31.083 (133) |
| compress_publish | 3.940 (640) | — | 3.928 (640) | — |
| mat_vec_bf16_f32_tg | 2.767 (640) | — | 2.771 (640) | — |
| copy | 2.406 (768) | — | 2.416 (768) | — |
| rms_norm_mul_f32 | 2.318 (640) | — | 2.378 (640) | — |
| pool | 1.230 (384) | — | 1.231 (384) | — |
| selection_prepare_rows | 1.080 (39) | 3.802 (74) | 2.518 (61) | 3.841 (114) |
| select_block_scores_rows | 0.068 (6) | 0.326 (12) | 0.127 (10) | 0.550 (19) |
| compress_rows | — | 0.202 (4) | — | 0.178 (4) |

At 1K rows, the 80K->128K increase in summed service is 69.765 ms:
37.605 ms dense MMA (53.9%), 17.174 ms masked MMA (24.6%), 14.747 ms radix
(21.1%). At 256 rows the increase is 23.233 ms: 10.027 ms dense MMA
(43.2%), 7.235 ms radix (31.1%), 4.413 ms masked MMA (19.0%). Publication
is flat. At 256 rows it uses the per-row path (640 projections/norms/publishes,
768 copies and 384 pools); at 1K, four batched `compress_rows` calls replace
that chain. Changing that publication arithmetic is outside this experiment.

### Work, traffic and roofs

The full per-kernel/per-cell ledger is [attribution.json](2026-11-13-v41-attention-serial/attribution.json),
generated by `scripts/profile/v41_serial_attention.py`; it includes calls,
two individual service times, bytes, FLOPs, GB/s and reference-roof fractions
for every kernel above. [stage-spans.json](2026-11-13-v41-attention-serial/stage-spans.json)
keeps individual stage repetitions, host intervals, wall and disk bytes.

These are **logical payload bytes, not measured DRAM transactions**. The MMA
model loads keys once per query rather than once per head; real SIMD loads
repeat some data, while caches reuse keys and queries. Radix counts five
score/mask passes and output IDs/masks; it excludes on-chip histograms and
atomics. Rates above the prior waste audit's 800 GB/s pricing reference are
logical request rates; caches can serve repeated loads. They are not measured
DRAM bandwidth or a measurement of its hardware roof. Hardware transaction
counters were not captured. FP32 SIMD-group MMA is also not the Half tensor
tier whose supplied reference peak is 131.7 TFLOP/s.

**81,920 depth / 256 rows**

| kernel | logical GB | useful GFLOP | logical GB/s | useful TFLOP/s (% tensor reference) |
|---|---:|---:|---:|---:|
| index_heads_rows_mma | 27.11 | 430.2 | 1738 | 27.57 (20.9%) |
| index_heads_rows_mma_masked | 8.983 | 137.4 | 784.8 | 12.01 (9.12%) |
| select_radix_rows | 3.133 | — | 594.2 | — |
| compress_publish | 0.003686 | 0.0006144 | 0.9355 | 0.0001559 (0.000118%) |
| mat_vec_bf16_f32_tg | 0.08552 | 0.08389 | 30.91 | 0.03031 (0.023%) |
| copy | 0.003146 | 0 | 1.307 | 0 (0%) |
| rms_norm_mul_f32 | 0.003932 | 0.001311 | 1.696 | 0.0005655 (0.000429%) |
| pool | 0.003932 | 0.001573 | 3.196 | 0.001278 (0.000971%) |
| selection_prepare_rows | 0.0105 | — | 9.726 | — |
| select_block_scores_rows | 0.09452 | — | 1396 | — |

**81,920 depth / 1,024 rows**

| kernel | logical GB | useful GFLOP | logical GB/s | useful TFLOP/s (% tensor reference) |
|---|---:|---:|---:|---:|
| index_heads_rows_mma | 109 | 1729 | 1739 | 27.6 (21%) |
| index_heads_rows_mma_masked | 35.94 | 549.6 | 792.2 | 12.11 (9.2%) |
| select_radix_rows | 12.59 | — | 770.4 | — |
| selection_prepare_rows | 0.04221 | — | 11.1 | — |
| select_block_scores_rows | 0.3799 | — | 1164 | — |
| compress_rows | 0.3637 | 0.3495 | 1800 | 1.73 (1.31%) |

**131,072 depth / 256 rows**

| kernel | logical GB | useful GFLOP | logical GB/s | useful TFLOP/s (% tensor reference) |
|---|---:|---:|---:|---:|
| index_heads_rows_mma | 43.34 | 687.9 | 1691 | 26.84 (20.4%) |
| index_heads_rows_mma_masked | 9.21 | 137.4 | 580.7 | 8.665 (6.58%) |
| select_radix_rows | 4.807 | — | 384.3 | — |
| compress_publish | 0.003686 | 0.0006144 | 0.9386 | 0.0001564 (0.000119%) |
| mat_vec_bf16_f32_tg | 0.08552 | 0.08389 | 30.87 | 0.03027 (0.023%) |
| selection_prepare_rows | 0.01679 | — | 6.67 | — |
| copy | 0.003146 | 0 | 1.302 | 0 (0%) |
| rms_norm_mul_f32 | 0.003932 | 0.001311 | 1.653 | 0.0005511 (0.000418%) |
| pool | 0.003932 | 0.001573 | 3.195 | 0.001278 (0.00097%) |
| select_block_scores_rows | 0.1511 | — | 1186 | — |

**131,072 depth / 1,024 rows**

| kernel | logical GB | useful GFLOP | logical GB/s | useful TFLOP/s (% tensor reference) |
|---|---:|---:|---:|---:|
| index_heads_rows_mma | 173.9 | 2760 | 1735 | 27.53 (20.9%) |
| index_heads_rows_mma_masked | 36.85 | 549.6 | 589.1 | 8.788 (6.67%) |
| select_radix_rows | 19.28 | — | 620.3 | — |
| selection_prepare_rows | 0.06737 | — | 17.54 | — |
| select_block_scores_rows | 0.6063 | — | 1102 | — |
| compress_rows | 0.3637 | 0.3495 | 2041 | 1.961 (1.49%) |


Index FLOPs are useful `2 * 128 * 32 * scored_keys`; they exclude ReLU,
weight/head reductions and duplicated output columns in the weight MMA.
Masked cardinality is 2,048 kept eight-key blocks per query, including the
newest partial block. Dense useful throughput stays near 27 TFLOP/s across
depth. Masked useful work stays fixed, but its service grows because the
incumbent still launches `ceil(visible_keys / 128)` groups per query, loads
query/weight tiles and writes the dense score domain for rejected panels.
Thus low masked utilization is partly avoidable launch/domain work.

Selection and fill kernels use integer operations/comparisons, not tensor
FLOPs. Pool/norm/publication also use exp, rsqrt or trigonometric functions;
their ledger's ordinary-operation counts exclude those special functions.
They are small, flat or dependent-dispatch limited, so a tensor-peak ranking
alone would be misleading. Even deleting all radix service gives only
1.129/1.145x at 80K (256/1K) and 1.220/1.186x at 128K; selection alone
cannot meet the complete-span 1.25x gate.

### Dependencies versus ordering choices

- KV publication must precede the scoring that reads those new cache rows.
  Ratio-2 groups pool values/gates before normalization, BF16 key projection
  and normalized/rotated publication. Position and cache state transitions
  are real dependencies; they do not imply that every GPU query must run
  singly. The production 1K path already batches completed groups.
- Query-row score computations are independent after publication, but the
  bounded dense score scratch forces sub-tiles. Reusing that scratch requires
  each sub-tile's selection to finish before the next score overwrite.
- Layer 20 must score keys, derive eight-key block maxima and publish its
  candidate mask before later layers 24/28/32/36 use it. Those layers also
  depend on the preceding layer's hidden/FFN result. Cross-layer scoring is
  not independent work that can simply be put on another queue.
- Within layer 20, its own unmasked key top-k and block-candidate selection
  have separate outputs and could be scheduled independently after scoring.
  Their current order is a scheduling choice, but only a small fraction of
  the whole span. Value gathers and sparse QK/PV/sinks are in `attention_tiles`,
  outside this serial span.
- For these already-pruned rows, non-publishing `selection_prepare_rows`
  calls write nothing. The publishing layer's initial all-ones mask is then
  overwritten by block radix before a consumer reads it. This is a smaller
  exact pass-removal opportunity, not the missing depth-scale solution.

## The three alleged duplicates are required work

The historical append capture ends at 81,920; its actual prefix is 80,896
tokens followed by 1,024 rows. For its ratio-2 index layers,
`width=40960`, score capacity is 8,388,608 floats and tile size is 204 rows.
The six score calls cover row offsets 0/204/408/612/816/1020 (last tile four
rows). The fourth and fifth calls both round up to grid `[320,204,1]`.

Old audit matches are scope/node 2:103 against 2:100, 2:506 against 2:503,
and 3:40 against 3:37, totaling 7.533875 ms. They share buffer spans, but
`row0` at buffer slot 7 differs: 612 versus 816. Their maximum visible
length also differs. The kernel reads query/weight row `row0 + local` and
writes local score scratch; these are different computations and outputs.
The reported 75 MB is reflected buffer-span extent, including the writable
score scratch, not a measured set of invariant input bytes.

The existing MMA GPU differential now includes a strict offset witness:
full evaluation versus tiled evaluation of the same rows is bitwise, while
the same buffers/grid with row0=0 versus 2 produce different scores. It passed
on M5 with Metal validation and `QWEN_REQUIRE_METAL_TESTS=1`, emitting
`MMA_SUBTILE_REUSE_PROOF full_vs_tile_bits_exact=true row0_0_vs_2_differ=true`.
There is no safe runtime reuse to ship for these matches.

The causal recorder now exports inline bytes, threadgroup-memory lengths and
pipeline identity. `waste_audit.py` includes them in recomputation keys;
legacy captures without constants are labeled `unverified_recomputation`,
with no reuse recommendation. Its self-test covers differing inline args.
[The original audit](2026-11-13-waste-audit.md) is corrected at the source.

## Compact masked-block prototype and frozen gate

Measured prototype source `62e0bf495` (test helper path correction atop
`fca83a16d`). Default off, experimental `QWEN_V41_MASKED_INDEX_COMPACT`.
It compacts the published mask into 2,048 chronological block IDs per query,
then launches exactly 128 groups of 16 kept blocks per query. The existing
long-selection pair scratch holds those IDs; no new allocation or requantizing.
Each kept eight-key block retains the incumbent FP32 MMA, tail-window rule
and head-sum order. Rejected score locations are unused by masked radix.
The implementation derives only from this repository's incumbent kernels;
MIT and Apache-2.0 licenses were checked and source provenance noted in the
shader. No external source was adapted.

The strict M5 GPU differential passes at 80K/128K with nonzero row offsets
and partial tails: compact IDs, allowed score bits, chronological selected
IDs and status all match. Metal validation is enabled. Candidate hashes and
metallib hashes are retained in `compact/pins.json`.

Frozen before the first measured arm in `compact/plan.json`: one iter binary,
AB warmup, then BA/AB/BA; the geometric mean of three paired baseline/candidate
serial GPU intervals must be >=1.25 in **each** of 80K/128K x 256/1K cells.
All eight logits/causal-state/following-decode signatures must be bitwise in
every cell. No serve or default-on promotion is authorized by this gate.

**FAIL.** All 32 logits/state/following-decode endpoint signatures are bitwise.
Three cells miss the 1.25x floor; only 1K at 128K passes.

| depth / rows | baseline serial ms | compact serial ms | paired speedups | geometric mean | verdict |
|---|---:|---:|---|---:|---|
| 81,920 / 256 | 40.132597 | 34.282666 | 1.171983 / 1.170533 / 1.169398 | 1.170638 | fail |
| 81,920 / 1,024 | 122.675167 | 100.032209 | 1.237062 / 1.212312 / 1.229961 | 1.226401 | fail |
| 131,072 / 256 | 61.181888 | 51.293167 | 1.189710 / 1.194115 / 1.194551 | 1.192790 | fail |
| 131,072 / 1,024 | 191.139361 | 151.415263 | 1.262650 / 1.261631 / 1.262775 | 1.262352 | pass |

All arms: nice 0, thermal 0, process disk-read bytes **0** in every append
window (including warmups). Gate one-minute host load ranges 3.49–4.45;
the baseline census ranges 3.32–20.74. The workload monitors retain the
background process lists and per-run load samples. No cache-purge serve gate
was run or claimed in this stage.

Informational stage-profiled append walls (mean baseline/compact): 444.279/438.394 ms
at 80K/256; 952.332/938.152 at 80K/1K; 458.032/448.646 at 128K/256;
1,005.957/966.798 at 128K/1K. These are not an unprofiled wall gate.

The candidate, GPU test and environment knob were removed from active source.
Only the measurement tools, strict sub-tile witness, audit correction and
record remain. Prototype archive: `backup/v41-serial-compact-measured` at
`62e0bf495`. The complete paired arms, signatures and disk bytes are in
[compact-gate.json](2026-11-13-v41-attention-serial/compact-gate.json).

Candidate iter SHA256 `49c78442d3c87c81793213e29cdfabb7a8314810155b32d5b160277742a299fa`;
test `dc3523ac93a03317fbfe29bcecb37e8baf9e32a1036726f84f68c76506fcb3fc`.
Both profiles have main metallib `adbace11c5eecbc44e324eeb5141d364ef4546eeb989c2080b4e320b163a9fb6`;
tensor/research metallibs retain the baseline hashes above.

**Next:** isolate an exact two-query F32 dense-index MMA tile: reuse each
loaded key matrix for two independent accumulators while retaining each
query/key accumulation order. Dense MMA is the largest remaining cost and
depth slope. Compact masked launches alone cannot meet the full-span gate;
retain their archive as evidence for a future combined experiment, without
relaxing this failed gate. No additional GPU lever ran in this stage.

## Validation and closeout

- Iter check and test-profile compilation pass locally; both archived M5
  profiles build. Strict existing-MMA offset witness and prototype GPU
  differential each pass with Metal validation; no ignored test was added.
- Audit self-tests pass, including different inline `row0` values and legacy
  captures. Legacy matches now receive **zero priced gain**, not a reuse
  recommendation. The original three legacy matches and corrected pricing
  are retained in [audit-correction.json](2026-11-13-v41-attention-serial/audit-correction.json).
- Preliminary contracts attempts were invalidated by concurrent ENV
  regeneration; neither is counted as a pass. Final committed-tree
  `scripts/check-contracts.sh` output is retained in the closing run log.
  ENV is regenerated after candidate withdrawal (830 knobs).
- A first candidate build was interrupted when its private driver was updated
  while executing. No model measurement used it. The corrected driver and
  test module paths were fixed before functional/gate measurements; the
  successful source and executable/metallib hashes are pinned above.
- At gate closure all four archived executable hashes were reverified, then
  both pin directories, the M5 build target, all three transfer bundles and
  the M5 worktree were deleted. `cleanup.json` reports **14/14 model/test PIDs
  exited**, no remaining task processes, and no task process was killed.
  Raw data, manifests and drivers remain in the run directory. Final local
  process checks accompany the closing contracts log.
- No pushes or merges; no edits to `docs/PERF-ROADMAP.md`. SRV-2K0Y notes
  `T68EVR8M3XY6ZK8K` and `M41R8DNZ97GCEW2R` record the stage progression.

