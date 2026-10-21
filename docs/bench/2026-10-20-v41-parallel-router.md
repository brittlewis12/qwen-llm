# V4.1: Parallel Router, 1024-Row Chunks, Half-Staged Expert Tiles, MMA Dense Rows

Candidate `perf/v41-router-land` (measured at wip tip `19f3c2f3`; the landing
series is the same code with the no-effect native-GEMV edit reverted and the
probe trimmed) against main `4b1fca04`. M5 Ultra 80-core, Unsloth
DeepSeek-V4.1-Flash-1017 UD-IQ3_XXS, default wired limit, production
process lease, `qwen-bench suite --pp 2048 --tg 32 -d 0,2048 --runs 3 --seed 1`,
clean binaries, A/B/B/A. Raw: `metagross:~/qwen-runs/2026-10-20-flashnext/abba-v41-router/`.

## Throughput

| shape | main `4b1fca04` | candidate | change |
|---|---:|---:|---:|
| pp2048 | 192.1 (190.5-193.7) | 251.0 (250.9-251.0) | +30.6% |
| tg32 | 22.3 (22.3-22.3) | 32.8 (32.7-32.9) | +46.9% |
| pp2048@d2048 | 170.5 (163.0-178.1) | 227.4 (227.4-227.5) | +33.4% |
| tg32@d2048 | 23.2 (23.0-23.4) | 35.3 (35.2-35.3) | +51.8% |

tok/s, mean of two arms with the arm range in parentheses. Decode goes from
44.8 to 30.5 ms per token at depth 0.

Earlier in the lane, chunk 1024 + half-staged expert tiles alone (wip
`9758c06b`, same protocol): pp2048 190 -> 239 (+25.5%), tg32 unchanged
(`abba-v41/iq3-arm*.json`). The DS lane's chunk pilot had measured the chunk
change alone at 166 -> 220-228 on its own base.

## Where the decode time was

The DS lane's sampled attribution (`docs/bench/2026-10-20-v41-complete-row-batching.md`,
private decode FFN pilot) put the routed-expert stage at 0.49-0.54 ms per
layer for every expert storage cohort, about five times its byte floor and
independent of quant type. The opt-in probe
(`metal::expert_native::tests::decode_probe`, `QWEN_EXPERT_DECODE_PROBE=1`)
times the stage's dispatches in isolation at the decode shape
(K=5120, M=2304, top-6, 64-expert banks with rotating IDs, GPU heated,
variants interleaved per iteration, median of 12):

| dispatch set | median us | effective GB/s |
|---|---:|---:|
| production gate+up+down, Q3_K/Q3_K/Q3_K | 162 | 563 |
| production gate+up+down, Q3_K/Q3_K/MXFP4 | 164 | 599 |
| production gate+up+down, IQ2_XS/IQ2_XS/MXFP4 | 129 | 610 |
| production gate+up+down, Q8_0 x3 | 207 | 1,090 |
| Q3_K variant: no activation loads at all | 150 | 608 |
| Q3_K variant: float4 activation loads | 162 | 565 |
| Q3_K variant: 4 rows per simdgroup | 170 | 536 |
| production router (384 experts, top-6), serial, standalone / back-to-back | 641 / 400 | - |
| parallel router, standalone / back-to-back | 35 / 19-28 | - |

The expert GEMVs were already near the memory system's limit (the Q8_0
figure is above the 854 GB/s single-kernel stream number this repo has used
as the M5 Ultra roofline; the DRAM ceiling is nearer 1.09 TB/s). The
single-thread router accounted for ~0.4 ms of the ~0.5 ms stage, ~16 ms per
token across 40 layers.

Two measurement traps found on the way and built into the probe: command
buffers issued within the first ~100 ms after seconds of CPU work (fresh
fixture construction) read about 2x slower until the GPU clock settles, and
a fresh 800 MB buffer's first GPU touch is slower than its steady state. A
first pass without heating attributed a 2.2x "gain" to vectorized
activation loads; the heated, interleaved rerun shows no difference, and
that kernel edit is not part of the landing series. Decode-token-sized CPU
gaps (0.5-2 ms) before a command buffer cost nothing; 5 ms gaps do.

## Parity and tests

- `parallel_router_is_bitwise_equal_to_the_serial_reference`: ids, weights
  and status bitwise equal to the retained serial kernel over 96 rows (ties,
  duplicates, extreme logits, non-finite logits/biases at equal and
  different indices; 8/257/272/384 experts; top-1/2/6/8; three
  temperatures). The existing CPU-oracle routing test and the dispatch
  census tests pass at the new 256-wide threadgroup.
- 166 V4.1 lib tests pass on M5 (release) with all four changes.
- `linear_rows_mma_q8_and_bf16_with_offsets` checks the MMA dense route
  against the CPU oracle at a half-activation tolerance with the same
  offset and untouched-row guards; the F32-operand contract test pins the
  grouped kernel through a test-only flag.

## Held-text witness

The DS lane's bounded text-likelihood helper (v1: six frozen repository
documents, three prose and three code, 512-token prefix, 64 teacher-forced
targets each, F64 logsumexp over the full F32 vocabulary; helper carried at
`49eb469f` on `perf/v41-half-experts` with the chunk-256 pin replaced by a
manifest field, identical source on both arms), IQ3_XXS, each arm scored
twice with identical results:

| document | NLL main | NLL candidate | delta | top-1 main | top-1 candidate |
|---|---:|---:|---:|---:|---:|
| prose_1 | 2.60890 | 2.60648 | -0.0024 | 0.609 | 0.578 |
| prose_2 | 4.20139 | 4.18945 | -0.0119 | 0.219 | 0.203 |
| prose_3 | 2.24882 | 2.26673 | +0.0179 | 0.469 | 0.453 |
| code_4 | 1.25146 | 1.24928 | -0.0022 | 0.719 | 0.703 |
| code_5 | 1.58460 | 1.57317 | -0.0114 | 0.688 | 0.656 |
| code_6 | 0.98299 | 0.99666 | +0.0137 | 0.766 | 0.766 |
| mean | 2.14636 | 2.14696 | +0.0006 | 0.578 | 0.560 |

Mean NLL +0.03% with per-document changes of both signs; top-1 agreement
with the targets 222 -> 215 of 384. The router is bitwise; the difference is
the half-operand expert tiles and MMA dense rows (near-tie flips, with the
held-text likelihood unchanged). Raw:
`metagross:~/qwen-runs/2026-10-20-flashnext/v41-witness/`.

Task screen: the DS lane's 22-task battery
(`~/qwen-runs/2026-10-20-v41-unsloth-profile/v41-grouped-task-screen.py`,
`qwen serve`, max 768 output tokens) on IQ3_XXS: 9/22 on both arms with the
identical per-task pass/fail/truncation pattern (code 8/10, reasoning 1/8,
JSON 0/4; r1 truncated on both), and 21 of 22 visible answers
character-identical; c7 diverges at character 582 and passes on both. Raw:
`metagross:~/qwen-runs/2026-10-20-flashnext/v41-task-screen/`.

## Limits

- One artifact (IQ3_XXS) on one machine; the router change is
  artifact-independent, the half-tile coverage is not (10/40 layers on
  IQ3_XXS, 20/40 on Q2_K_XL; MXFP4 `ffn_down` has no generic half layout yet).
- The dense MMA rows round operands to half with F32 accumulation; the
  witness above and the task screen are the quality evidence, not
  baseline exactness.
- No Q2_K_XL or REAP-272 throughput rerun with the router; the earlier Q2
  chunk-only arms are partial (one B arm) and are not claimed.
