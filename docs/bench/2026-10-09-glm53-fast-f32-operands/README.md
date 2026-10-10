# GLM-5.3-Flash Fast with F32 matrix operands (map #12 accuracy lane, 2026-10-09)

Fast packed prefill stages both activations and dequantized weights in half
precision inside its batched mat-mat tiles; Exact runs the decode kernels
row by row in F32. This packet replaces the half staging with F32-operand
tiles, selected per stage family in tests only (`packed::F32Stages`,
`StageMode::FastF32`; product Fast is unchanged), and measures how close
each configuration gets to Exact and what it costs.

Commits: `f973ddde` (F32-operand Q8_0 tile, shared with DS4), `a64cf699`
(padded activation backing), `a9c96c4b` (F32-operand Q6_K tile
`kernel_mat_mat_q6_K_f32_mm64x32`), `56e558e1` (F32-operand routed experts,
`kernels/moe_grouped_f32.metal`), `a369d266` and `3eca56e3` (review fixes).
The JSON here was produced at `8e3a6047`, the tree of `a369d266` before
rebasing onto later main commits (the JSON keeps the recorded hash): debug
with Metal API validation for accuracy and gates, release without
validation for timing.

## Configurations

| Arm | F32-operand calls | Still half-staged |
|---|---|---|
| `fast` | none (product Fast) | every quantized matrix |
| `fast_f32_q8` | Q8_0 dense projections | Q6_K projections, routed experts |
| `fast_f32_dense` | Q8_0 and Q6_K dense projections (KDA, MLA, indexer, shared expert, dense FFN) | routed experts |
| `fast_f32_all` | dense projections and routed experts (gate/up and the SwiGLU output into down) | none |

The router and MLA absorption were F32 already. A census records every
Fast-lineage matrix call by stage family, weight kind and path, and the
probe asserts each selection's census against the half-staged run's on the
same tokens: the same calls, every enabled kind on its tile, nothing left
half-staged in a complete selection, unselected stages unchanged.

## Accuracy against Exact (`probe.json`)

All six frozen natural cases (`scripts/reference/glm53/reuse-natural-v1.json`,
sha256 recorded), 512-row chunks, Exact on the same frozen continuation as
the reference. Diagnostic: no bounds. Per case, mean / worst per-position
KL(Exact || arm):

| Case (prompt tokens) | Fast | Q8 only | Dense F32 | All F32 |
|---|---:|---:|---:|---:|
| H1 short chat (157) | 3.7e-3 / 4.1e-2 | 4.1e-3 / 3.7e-2 | 1.0e-2 / 1.1e-1 | 3.6e-3 / 3.0e-2 |
| H2 code context (1,755) | 2.6e-2 / 1.1e-1 | 2.5e-2 / 1.4e-1 | 1.6e-2 / 1.4e-1 | 1.1e-2 / 1.2e-1 |
| H3 tool, dense (351) | 2.5e-3 / 3.6e-2 | 6.2e-4 / 5.8e-3 | 5.0e-4 / 7.4e-3 | 1.7e-8 / 1.8e-7 |
| H4 tool across the frontier (2,279) | 1.1e-3 / 2.0e-2 | 1.1e-3 / 9.0e-3 | 5.0e-4 / 6.7e-3 | 2.1e-4 / 4.0e-3 |
| H5 long chat, sparse (4,962) | 1.3e-2 / 9.5e-2 | 1.7e-2 / 1.8e-1 | 6.1e-3 / 3.6e-2 | 1.0e-2 / 5.1e-2 |
| H6 max sampled tool (333) | 2.5e-3 / 2.7e-2 | 2.6e-3 / 2.0e-2 | 1.8e-3 / 1.2e-2 | 6.4e-9 / 5.5e-8 |
| **Equal-case mean of mean KL** | **8.28e-3** | 8.40e-3 | 5.83e-3 | **4.16e-3** |
| Median / largest worst KL | 3.8e-2 / 0.113 | 2.8e-2 / 0.177 | 2.4e-2 / 0.137 | 1.7e-2 / 0.116 |

Top-1 flips (position: Exact-side / arm-side regret): Fast H2 21 (0.22 /
0.03); all-F32 H1 24 (0.010 / 0.30), H2 22 (0.84 / 0.25), H5 0 (0.23 /
0.33). The Q8 and dense arms share the H1 24 and H5 0 flips and add H2 10.

- F32 for the Q8_0 projections alone does not move the summary (8.40e-3);
  partial removal of perturbation sources is not monotone on single cases.
- Dense F32 lowers the mean on five of six cases (H1 worse) and the summary
  by 30%.
- All F32 halves the summary. H3 and H6 become indistinguishable from Exact
  at this resolution (worst KL 1.8e-7 and 5.5e-8: reduction-order
  differences only). H1, H2 and H5 keep drift of Fast's order at their
  worst positions, and the largest worst KL (0.116, H2) is not lower than
  Fast's (0.113).

This is closer agreement with Exact on these cases, not a demonstrated
quality improvement; quality claims need the preregistered holdout.

## Routing under all-F32 (`routes.json`)

For the single-chunk cases, the expert set each MoE block (42) routed every
prompt row to, against Exact (sets only: slot order and route weights are
not compared):

| Case | Fast: rows changed (first block) | All F32: rows changed (first block) | All-F32 prompt-end KL |
|---|---:|---:|---:|
| H1 | 887 (3) | 654 (7) | 6.2e-4 |
| H3 | 6,077 (3) | 14 (29) | 1.3e-8 |
| H6 | 2,933 (3) | 13 (26) | 3.2e-8 |

Half staging changes routes from the first MoE block on. Under all-F32 the
two near-exact cases change one row in a few late blocks; H1 changes one row
at block 7, and the changes then spread to 654 row-blocks. This is
consistent with a near-tie routing change amplified through later tokens,
but does not establish the mechanism: Fast's H3 changes 6,077 row-blocks
with a prompt-end KL of only 1.7e-4, so changed routes need not mean large
drift. Replaying Exact's routes inside the candidate would separate cause
from consequence.

## Bitwise properties under all-F32

- `fast_f32_operands_keep_chunk_identities`: 512 == 128 and 64 == 97 rows
  bitwise through decode; the census at each chunking matches the half
  run's calls with no fallback (600-token prompt in 512-row chunks: KDA
  projections Q6_K 272 / Q8_0 204, KDA expansions 136, MLA 40 / 48, indexer
  44, shared expert 246 / 6, dense FFN 18, routed experts 84 calls on F32
  tiles; the F32 router's 42 on its own kernel).
- `f32_operand_snapshot_restore_and_frontier_chunkings` (`snapshot_gates.rs`):
  capture -> restore -> suffix equals the uninterrupted run in logits and end
  state at n = 97-100 and 2,050-2,054 (every pool residue, around the
  2,052-token sparse frontier); a capture at a cancelled prefill's committed
  boundary continues with equal logits; 512- and 128-row chunkings of a
  2,352-token prompt agree in logits and end state. (`3eca56e3` adds the
  end-state comparison to the cancellation case; rerun at integration.)

## Kernel-level checks (unit tests, Metal validation)

- Q6_K tile: basis-vector readback equals an independent ggml-order decoder
  for every weight (signed and extreme scales, q 31-33, partial 64-row
  tile); relative RMS 4.3e-7 against an F64 product versus 2.3e-4 for the
  half tile (K = 1,024, outlier channels at 50x); per-token outputs bitwise
  independent of the token count (1-512), tile position and poisoned
  neighbouring rows.
- Routed experts: against the per-row decode composition 1.7-1.8e-6
  relative RMS versus 4.7-6.1e-4 for the half path; two slots against an
  F64 product within 1.8e-6; per-token outputs bitwise equal in a 64-row
  dispatch and in 1/17/33/13-row sub-dispatches (IQ2_S and IQ3_S gate/up,
  IQ3_S and IQ4_XS down); buckets of 0/1/17/33 slots, invalid slot ids
  skipped; refusals for unsupported types, bad bindings and absurd
  geometry.

## Cost (`cost.json`, release, A-B-C-C-B-A twice after warm-ups)

| Span | Fast | Dense F32 | All F32 |
|---|---:|---:|---:|
| Fresh 2,048-token prompt | 10,465 ms | 10,935 ms (+4.5%) | 11,791 ms (+12.7%) |
| 1-token suffix at 2,048 | 92 ms | 170 ms (+84%) | 164 ms (+78%) |
| 17-token suffix at 2,048 | 398 ms | 405 ms (+1.9%) | 415 ms (+4.3%) |
| 64-token suffix at 2,048 | 772 ms | 757 ms (-2.0%) | 893 ms (+15.6%) |
| 64-token suffix at 2,564 (sparse) | 897 ms | 882 ms (-1.7%) | 1,036 ms (+15.5%) |

An earlier run at `e8640a0f` (the tree of `56e558e1` before rebasing) measured +4.6% (dense) and
+16.1% (all) on the fresh prompt. The one-row cost comes from running a
32-token tile for one token where half Fast uses mat-vec; a narrower tile
with the same per-token arithmetic would address it without giving up the
chunking identities.

## Reading

- The half staging of matrix operands accounts for most of Fast's drift on
  these cases: with every quantized matrix operand in F32, two cases match
  Exact to reduction-order level and the summary halves.
- What remains is concentrated in a few positions of three cases and
  coincides with discrete routing changes; it is not removed by F32
  operands alone.
- Dense F32 is nearly free on prompts of 17 tokens or more; all F32 costs
  about 13-16% of prefill wall. Both pay about 75 ms on one-token suffixes.

Next (cx 01a10cc): a one-row path that keeps each candidate's per-token
arithmetic; one bounded packet on the F32 expert tiles' cost; then freeze
the surviving configurations and run a preregistered fresh holdout against
Exact under the deployed split schedule, with the selection rule fixed
before viewing results.
