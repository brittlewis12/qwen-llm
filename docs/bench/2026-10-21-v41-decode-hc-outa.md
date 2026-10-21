# V4.1: Fused mHC Pre-Block, Grouped output_a, 2048-Row Chunks

Candidate `perf/v41-land2-series` against main `e0c3ac0f` (the previous
V4.1 landing). M5 Ultra 80-core, Unsloth DeepSeek-V4.1-Flash-1017
UD-IQ3_XXS, default wired limit, production lease,
`qwen-bench suite --pp 2048 --tg 32 -d 0,2048 --runs 3 --seed 1`, clean
binaries, A/B/B/A. Raw: `metagross:~/qwen-runs/2026-10-20-flashnext/abba-v41-land2/`
(arms 1,4 = A; arms 5,6 = B; arm 7 = A).

## Throughput

| shape | main `e0c3ac0f` | candidate `32babac5` | change |
|---|---:|---:|---:|
| pp2048 | 251.9 (246.4-254.7) | 266.0 (266.0-266.0) | +5.6% |
| tg32 | 32.7 (32.5-32.9) | 40.8 (40.4-41.2) | +24.7% |
| pp2048@d2048 | 230.0 (229.2-230.5) | 239.5 (239.5-239.6) | +4.2% |
| tg32@d2048 | 33.8 (31.3-35.1) | 41.6 (39.4-43.8) | +23.2% |

tok/s, arm means with ranges (A: three arms, B: two; the first two B arms
of this protocol were voided by a dirty candidate checkout and rerun).
Decode at depth 0 goes from 30.6 to 24.5 ms per token. Cumulative over the
two V4.1 landings today, from main `4b1fca04`: tg32 22.3 -> 40.8 (+83%),
pp2048 192 -> 266 (+39%).

## Where the decode time went (sampled stage envelopes, depth 2048)

The DS lane's private sampled-encoder attribution (`f08bdcea`, carried onto
main for measurement only on `perf/v41-decode-timeline`; ~3% window
inflation; spans are inclusive, nonadditive envelopes) across the three
steps of this pass, ms per token summed over 40 layers:

| span | after router (`e0c3ac0f`) | + fused mHC pre | + grouped output_a |
|---|---:|---:|---:|
| attention mHC pre | 3.29 | 0.87 | 0.88 |
| attention operator composite | 10.53 | 10.63 | 9.72 |
| attention mHC post | 0.16 | 0.16 | 0.16 |
| FFN composite incl. mHC (30 layers) | 9.03 | 7.28 | 7.29 |
| FFN mHC pre (10 split layers) | 0.79 | 0.21 | 0.21 |
| FFN router + shared (10) | 0.61 | 0.61 | 0.61 |
| FFN routed (10) | 1.62 | 1.62 | 1.62 |
| head projection | 0.61 | 0.61 | 0.61 |
| Engram projections + gates | 0.48 | 0.51 | 0.52 |
| **control ms/token** | **28.05** | **23.26** | **22.30** |

- mHC pre: 0.079 -> 0.021 ms per site. The 4*hidden -> 24 F32 projection
  ran through the generic one-simdgroup-per-row mat-vec (24 simdgroups, a
  160-deep load chain each); the controls kernel did its Sinkhorn iterations
  as single-thread device read-modify-writes. Now one 1024-thread
  threadgroup per logit for the projection, and one fused threadgroup for
  projected RMS, controls, collapse and RMS norm.
- Attention operator: 0.237 ms per plain layer = ~0.123 ms of Q8_0 bytes at
  1.09 TB/s plus ~17 dispatches; grouping the eight output_a GEMVs into one
  native-expert dispatch removed seven of them (-0.9 ms per token). Owner
  and producer layers add 0.11-0.14 ms each for compressor and index work.
- Remaining per-token budget at 22.3 ms: ~10.2 ms of bytes (11.1 GiB at
  the ~1.09 TB/s the Q8_0 GEMV probe reaches; the part is listed at
  1.2 TB/s), ~8 ms of dispatch floor (~1,600 dispatches), ~4 ms of kernel
  inefficiency spread across attention core, routed stage glue, Engram.

## Prefill attribution (2048 tokens, IQ3_XXS)

Four sampled spans per layer (`QWEN_V41_PREFILL_STAGES=1`; the splits
inflate wall time, so only shares are meaningful), summed over 40 layers
and two 1024-row chunks:

| span | GPU s | share |
|---|---:|---:|
| attention HC rows | 0.10 | 1% |
| attention rows (projections, owner/producer per-row loops, sink tiles) | 3.07 | 39% |
| attention finish/post (output_a/b MMA rows, mHC post) | 0.54 | 7% |
| FFN (HC rows, router, shared, routed experts) | 4.08 | 52% |

The routed expert GEMMs are ~35 TFLOP per 2048 tokens and take at most
~4 s: ~8.7 TFLOPS effective on half-staged MMA tiles. Two negative results
bound the cheap options: moving the 25 remaining MXFP4-down layers onto the
half-staged generic tiles (new MXFP4 generic layout, 35/40 layers half;
branch `perf/v41-mxfp4-half`, CPU-codec equivalence tests pass) gained
+1.6% pp2048 (251.0 -> 254.9 against the same base), so the half tiles are
not materially faster than the F32-operand tiles here; 2048-row chunks
(32 rows per expert instead of 16) gained +5.6% (254.9 -> 269.3). The
expert tile's own efficiency and the scalar-F32 attention core (score,
softmax, PV, per-row owner/producer loops) are the prefill levers left,
and both overlap other lanes' work (Metal 4 TensorOps tier; MLX-style
expert GEMM). The MXFP4 layout is not landed: it also widens DFlash and GLM
grouped eligibility on MXFP4 artifacts, which those lanes have not measured.

## Held-text witness and task screen

Same helper and protocol as the previous packet (six frozen documents,
512-token prefix, 64 teacher-forced targets each, IQ3_XXS; helper
`49eb469f` on both arms, main at `e0c3ac0f` versus the candidate):

| document | NLL main | NLL candidate | delta | top-1 main | top-1 candidate |
|---|---:|---:|---:|---:|---:|
| prose_1 | 2.60648 | 2.60648 | -0.00000 | 0.578 | 0.578 |
| prose_2 | 4.18945 | 4.18944 | -0.00001 | 0.203 | 0.203 |
| prose_3 | 2.26673 | 2.26673 | +0.00000 | 0.453 | 0.453 |
| code_4 | 1.24928 | 1.24928 | +0.00000 | 0.703 | 0.703 |
| code_5 | 1.57317 | 1.57311 | -0.00006 | 0.656 | 0.656 |
| code_6 | 0.99666 | 0.99666 | +0.00000 | 0.766 | 0.766 |
| mean | 2.14696 | 2.14695 | -0.00001 | 0.560 | 0.560 |

Differences are F32 reduction order in the fused mHC kernels; top-1 is
unchanged on all 384 targets. The 22-task screen is 9/22 on both arms with
the identical per-task pattern and 22/22 character-identical answers.
Raw: `metagross:~/qwen-runs/2026-10-20-flashnext/land2-quality/`.

## Limits

- One artifact, one machine. The mHC and output_a changes are
  artifact-independent; the chunk change scales scratch with rows
  (admission prices it).
- Sampled attribution is inclusive and nonadditive; the control windows
  are the throughput claims, the ABBA is the landing evidence.
- Decode is now ~46% bytes; the next decode levers are each <= 1 ms
  (fused attention core, routed gate/up fusion, rope/copy fusion) and the
  artifact's Q8_0 attention projections (5.1 GB/token).
