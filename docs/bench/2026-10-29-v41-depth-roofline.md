# V4.1 Depth Matrix On The Q-Cache Build

2026-10-29. Build `8f455225` (main `034b0c01` Q-cache plus docs; the later
test-only PV probe commits do not change the production path). M5 Ultra
80-core, 256 GiB, GSQ-RCO IQ2_XS, production lease. Q-cache default on, CED
prefill. Raw: `~/qwen-runs/2026-10-29-v41-encoder-depth/`.

This supersedes the earlier Q-half matrix (`cf82e551`). The Q-cache A/B on
the same binary is 1,991.1 on vs 1,901.2 off tok/s at 8K (+4.7%, two tight
samples per arm, 180 s cooldown).

## Decision

The cold-prefill curve peaks at 32K, not at 128K, and the append curve falls
much harder. Cold CED prefill is 2K 1,231.1, 8K 1,988.3, 32K 2,301.7 and
128K 1,980.8 tok/s. A fixed 1K append is 856.5 at d0, peaks at 955.1 at d8K,
then falls to 895.8 at d32K and 586.0 at d128K. At 128K the append costs
**3.4x** the cold-prefill per-token time.

The 8K shape is now close to the 32K shape, but 128K is a real saturation
failure, not just a ramp: the deep-context selection/index path eats the
gain. The cold 128K point is also thermally fragile; a two-rep run's second
sample was 1,689 after the 60 s fill.

## Protocol

`qwen-bench suite` with the synthetic uniform-vocabulary stream, CED prefill
(`QWEN_V41_PREFILL_EXECUTION=shrinking_ced`), cooled points: 180 s between
2K/8K/32K, two reps each; 128K two separate single-rep runs after 300 s
cooldowns. The 1K append is two reps per depth after the untimed fill,
180 s between depths. The 2K cold point uses a half-full 4,096-row chunk
because the production allocator selected 4,096 rows; 8K and deeper use full
chunks. Single-run CED above 8K is outside the pair gate: mapping points,
not promotion evidence.

## Cold prefill

| prompt | tok/s | wall ms | samples |
| ---: | ---: | ---: | --- |
| 2K | 1,231.1 | 1,664.6 | 1,200.7 / 1,261.4 |
| 8K | 1,988.3 | 4,120.1 | 1,987.4 / 1,989.1 |
| 32K | 2,301.7 | 14,236.6 | 2,302.6 / 2,300.7 |
| 128K | 1,980.8 | 66,173.0 | 1,987.0 / 1,974.6 (cooled singles) |

The 128K two-rep run gave 1,995.7 / 1,689.3 (mean 1,842.5): the second
sample is a thermal after-effect of the first 60 s prefill. Use the cooled
singles.

## Fixed 1K append

| existing depth | tok/s | wall ms | ms per appended token | samples |
| ---: | ---: | ---: | ---: | --- |
| 0 | 856.5 | 1,195.7 | 1.168 | 845.8 / 867.3 |
| 8K | 955.1 | 1,072.1 | 1.047 | 955.9 / 954.4 |
| 32K | 895.8 | 1,143.1 | 1.116 | 896.1 / 895.4 |
| 128K | 586.0 | 1,747.4 | 1.706 | 587.9 / 584.2 |

The 1K append is fastest at d8K, then falls; at 128K it is 3.4x the cold
128K per-token cost. The cold curve hides this because it amortizes the
late-context selection over 131,072 rows instead of 1,024.

## Decode

`tg32`, two reps after the untimed fill. The d0 and d128K rows are the
Q-cache build; d8K/d32K are from the preceding build, whose decode path is
unchanged by the prefill Q-cache. They bracket a flat curve to 32K.

| depth | tok/s | ms/token | samples |
| ---: | ---: | ---: | --- |
| 0 | 77.68 | 12.87 | 77.71 / 77.66 |
| 8K | 73.23 | 13.66 | preceding build |
| 32K | 72.46 | 13.80 | preceding build |
| 128K | 68.79 | 14.54 | 70.31 / 67.27 |

Decode is flat to 32K and about 11% lower at 128K. The weight-byte floor is
7.17 ms/token at 1.09 TB/s; d0 is 1.79x that floor. The gap is not all
recoverable bandwidth: routed IQ2 accumulation has measured compute-bound in
isolation, so a streaming/cache-resident/loads-only control is still owed
before another decode-bandwidth claim.

## What changed the earlier matrix

- The Q-cache (`score_tensor64_qregister`) is real: 8K CED +4.7% same-binary.
  It raises 8K/32K and leaves the 128K selection path unchanged.
- The old 128K 2,128 tok/s point was a single uncooldowned run; the cooled
  Q-cache point is 1,980.8. The cold curve peaks at 32K.
- The cold 128K 1K append remains the worst cell: 586 tok/s, 3.4x the cold
  per-token cost. The wide prefill top-k selector was exact but only +1.8%
  end-to-end and is parked; the index score and per-row chain are next.

## Next

1. Revisit the 8192-row encoder chunk with DEWM if prioritized. The sibling
   MLX-Serve V4.1 plugin sets `MOE_ROWS=8192` and notes that 2048-row chunks
   stream every expert bank 4x per 8K prompt; our 4096-row chunk pays twice.
   This is the cleanest explanation of the 8K-to-32K ramp and the biggest
   exact prefill lever in that stack.
2. Dense dequant+GEMM at >=1024 rows (MLX-Serve `DENSE_ROWS=1024`): start
   with Q8_0 `q_a`, then Q5_K `output_a/b` and shared expert. Their codes are
   lossless in BF16; ours are not, so quality-screen the Q5_K path.
3. Compare the Q-cache/attnGather layout with DEWM's fused softmax/PV
   prototype; the sibling attention core is BF16 QK, F16 probabilities, keys
   read in place, grouped output.
4. Decode: run the streaming/cache-resident/loads-only control before another
   bandwidth claim, then judge a PLD/MTP verifier on predictable and novel
   text separately.

## Limits

Synthetic tokens exercise uniform routing and Engram novelty. Deep points are
single reps after cooldowns; the 128K decode spread and the 128K prefill
second sample show the thermal sensitivity. Cold 128K averages early and late
positions; the 1K append is the late-position probe. No quality claim is made
for any scheduling, half-attention or dense-GEMM change.
