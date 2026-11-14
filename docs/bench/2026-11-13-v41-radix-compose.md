# V4.1 compact + radix coalescing: frozen gate fails, default-off controls retained

The 256-row cells missed the stage gate, and compact plus radix coalescing was promoted to wall qualification by lead judgment.

Bounded stage started 2026-11-14 04:09 UTC; hard stop 06:09 UTC. Compact is
restored unchanged from its earlier record. Its standalone frozen-gate failure
stands. M2 remains removed. This stage measures the composition under the
same >=1.25x-every-cell serial-span rule, with compact-only controls in the
same packet. A failed performance gate keeps both components default-off for
the lead/Britt decision; it does not promote either component.

## Exact change and controls

`select_radix_rows_coalesced` derives from this repository's incumbent radix
kernel under MIT/Apache-2.0; the shader records that provenance. For each
digit pass it retains the existing score-to-key mapping, mask checks and
first-pass NaN reporting. Matching masked `-Inf` keys accumulate a private
integer count; `simd_sum` combines 32 counts, followed by one histogram atomic
per simdgroup. All other bins retain per-key atomics. Counts are bounded by
the row length and add exactly. Threshold search, signed-zero order, equal-key
tie selection, chronological output placement and candidate publication are
unchanged. No floating arithmetic, precision or persistent allocation changes.

The host selects this variant only for masked, non-publishing row selection.
The baseline kernel remains intact. Independent controls, both default-off:

- `QWEN_V41_MASKED_INDEX_COMPACT=0`: restore full-domain masked scoring.
- `QWEN_V41_RADIX_NEG_INF_COALESCE=0`: restore per-key histogram atomics.

The `perf-ablations` probe scopes explicit arms; production rollback flags
retain their usual independent semantics. Prefix construction uses production
with both experimental flags zero. M2 code and its knob are absent.

## Frozen three-arm packet

One pinned iter binary; A=baseline, B=compact+coalescing, C=compact alone.
Warm A/B/C, then measured triplets B/A/C, C/A/B, B/A/C. Primary A/B pairs
remain adjacent and interleave BA/AB/BA. Compact comparisons also reverse
order across the same triplets. No model samples precede this recorded plan.

At each of 256/1K rows x 80K/128K depth, require geometric mean A/B >=1.25
on the sum of `prefill_attention_serial` GPU intervals over the eight active
global layers. Every logits/causal-state/following-decode signature must be
bitwise across all twelve arms, including warmups. No pair is omitted.
Compact's contribution is paired A-C milliseconds; the incremental radix
contribution is C-B. These are measured differences in one span, not separate
kernel-service times or an Amdahl estimate.

Pass proceeds to native 1K/256 @80K and 1K @128K: one binary, AB warmup,
three BA/AB/BA pairs per cell. Then stop. No serve/default-on promotion is
part of this stage. Future serve warmup must exercise the eligible deep-index
pipelines; a shallow 4K-only warmup does not necessarily do so.

## Source, strict witness and environment

Initial source `682ca699b`, based on pre-stage `2a0e533aa` and main
`f8398b4b4`. Worktree: `~/code/qwen-llm-v41-radix-compose` on M5; local
branch `perf/v41-attention-serial`. No edits to the preparation/finish paths.

The strict differential compares old/new selected IDs, complete candidate
masks, status and canaries. Small cases also compare against the independent
CPU sort order. Cases include real 256/1K row populations at 80K/128K,
nonzero row/buffer offsets, top-k crossings, all/none/2,048-block masks,
ties, NaNs, signed zeros, infinities and all-`-Inf` thresholds. The restored
compact scoring/selection differential runs separately.

All builds use nice 10 and every measured/test child nice 0, verified by
`ps -o nice=` and `getpriority`. Waiting leases hold at most 15 minutes;
Python forwards the locked FD with `pass_fds`, checks for other resident
models, settles <=1 for 60 seconds, permits 2 and aborts at 3. Background
load and each arm's disk-read bytes are retained. Pinned executables are
copied outside `target/` before measurement and kept for pending review.

Raw: `metagross:~/qwen-runs/2026-11-12-v41-prefill/radix-compose/`:
`plan.json`, `pins.json`, `monitor.jsonl`, strict-witness outputs and
`stages-{depth}-{rows}.jsonl`. Source/driver versions and the analysis script
are retained beside the raw records.

Both M5 witnesses pass with Metal validation and
`QWEN_REQUIRE_METAL_TESTS=1`. Coalescing: **13 cases, 1,332,736 selected-buffer
words**, complete masks/status and guards, with independent CPU references.
Diagnostic old/new GPU timings (median five buffers after warmup, three
encodings per buffer; not a formal isolation gate):

| depth / rows | incumbent us | coalesced us | speedup |
|---|---:|---:|---:|
| 80K / 256 | 733.903 | 482.583 | 1.520779x |
| 80K / 1K | 1521.208 | 1483.528 | 1.025399x |
| 128K / 256 | 794.375 | 726.875 | 1.092863x |
| 128K / 1K | 2311.833 | 2228.139 | 1.037562x |

[strict.json](2026-11-13-v41-radix-compose/strict.json) retains the points.
Iter SHA256 `b1cad0ae6e96bd6e011951c07339cb3dcb8e5f1696576ea9213619310f435643`;
test `e95618fcf666735415d131639ab0391cdf486c4e3bf27935a94e2a437d54660e`.
Both profiles have main metallib `35ac100a2ceb16fdcfdbc435fb7d941910183e873570c50a2044c52cce168c7b`,
tensor `43ae770545fa60f1f51d2bb3220a93c0aaf3bfc13d382a99ea2345920eaf184c`,
research `be63c8ae9332529da45c21328333191834bd767917f5f9bc55d94eb3702b557f`.
The same archived iter executable runs every A/B/C arm.


## Serial-span gate and component contributions

**FAIL under the unchanged every-cell rule.** Both 1K cells pass; both
256-row cells miss 1.25x. All **48** logits/state/following-decode endpoints
are bitwise. No pair is removed or reweighted.

| depth / rows | baseline A ms | compact C ms | composed B ms | A/C geomean | A/B geomean |
|---|---:|---:|---:|---:|---:|
| 81,920 / 256 | 40.121389 | 34.268513 | 33.892529 | 1.170794 | 1.183782 |
| 81,920 / 1,024 | 122.700528 | 99.230152 | 97.829987 | 1.236525 | 1.254222 |
| 131,072 / 256 | 62.306457 | 52.100583 | 50.939806 | 1.195885 | 1.223085 |
| 131,072 / 1,024 | 191.520000 | 152.397695 | 149.147153 | 1.256752 | 1.284101 |

| depth / rows | compact saved ms | radix incremental saved ms | C/B geomean | compact / radix share of saved span |
|---|---:|---:|---:|---:|
| 81,920 / 256 | 5.852875 | 0.375984 | 1.011093 | 93.96% / 6.04% |
| 81,920 / 1,024 | 23.470376 | 1.400166 | 1.014312 | 94.37% / 5.63% |
| 131,072 / 256 | 10.205874 | 1.160777 | 1.022745 | 89.79% / 10.21% |
| 131,072 / 1,024 | 39.122305 | 3.250542 | 1.021762 | 92.33% / 7.67% |

Compact-only A/C, incremental C/B and composed A/B are **3/3 positive in
every cell**. Compact accounts for roughly 90–94% of the observed span saving;
coalescing adds about 0.38/1.40 ms at 80K and 1.16/3.25 ms at 128K (256/1K).

Every measured window has **0 process disk-read bytes**; warmup reads remain
in [gate.json](2026-11-13-v41-radix-compose/gate.json). All test/model children
are nice 0. Thermal pressure reaches **2** during the 128K/256 run, allowed
by the frozen protocol; no observed sample reaches 3. Its three pairs remain.
The 128K/1K load settles again before starting. Model-run one-minute load
ranges 15.77–40.94; background work is retained in the monitor.

All eight active global layers (2/8/14/20/24/28/32/36) have valid timestamps
in every arm. The other 32 serial stages have no GPU work and unset timestamp
placeholders. No active sample is dropped. The full nine measured arms per
cell, their order, layer times, component deltas, disk bytes and signatures
are preserved in the linked gate JSON.

## Native packet

Not run: the all-cell serial gate failed. No wall-based criterion was
substituted. Both controls remain **default-off on the branch**, as explicitly
requested; neither component is removed or promoted. Native/serve
qualification and any alternate wall criterion await the lead/Britt decision.
The measured source is also tagged `backup/v41-radix-compose-measured`.


## Validation and closeout

- Local iter/test-profile compilation and both M5 profiles build. Strict
  GPU witnesses pass; real-model A/B/C signatures are 48/48 bitwise.
- Initial CPU preflight correctly rejects the two unclassified GPU tests.
  `acceptance_inventory.json` now classifies both by full name as `gpu`,
  `ignored=false`; corrected preflight contracts pass at `a77bffed5`.
  Final committed-tree contracts output is retained as `contracts-final.log`.
  ENV was regenerated: **832 knobs**, no polarity conflicts. No ignored test
  was added. M2 remains absent from active source and knobs.
- Queue contention is recorded: the initial Ghost streaming lease declared
  3 minutes ran past 15. Build waited approximately 18m35s, witness 13m05s,
  and model packet 2m29s. Work stayed queued; no bypass or other-lane signal.
  Our build/witness/model holds were each below 15 minutes.
- Both executable hashes were reverified at closeout. The iter/test pins
  and three byte-identical profile metallibs are retained under `pins/`
  for pending review; [retained-pins.json](2026-11-13-v41-radix-compose/retained-pins.json)
  records their hashes. They link only to system libraries. The archived
  driver support and lease wrapper work without the deleted worktree.
- M5 worktree, bulk build target and transfer bundle are deleted.
  `cleanup.json` records **6/6 observed model/test exits** and no task
  processes; no task process was killed. Final local/M5 process checks
  accompany the closing log. Local transfer bundles are deleted as well.
- No pushes/merges and no roadmap edits. SRV-2K0Y notes `JE6H18W8ENXTNBMV`,
  `EEQ8VXHZFVJQPY4Y` and `288S2CWGYFBHVKAX` record source, scheduling and
  strict-witness checkpoints; the closing note records the final tip.

