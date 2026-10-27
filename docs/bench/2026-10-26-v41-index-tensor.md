# V4.1 Tensor Index-Score Rows

2026-10-26. Source: `4c187158` (on `e0230244`, `f191c303`, `ddcec6b6`).
Artifact: GSQ-RCO IQ2_XS, 40 layers, 4,096-row chunks, M5 Ultra. Raw:
`metagross:~/qwen-runs/2026-10-26-v41-index-tensor/`.

## Decision

Promote. The wide index path now runs one threadgroup per query row: the 32x128
query is staged once, then each 64-key tile runs one 32x64x128 tensor matmul
and the relu/weighted head sum writes the same F32 score layout. Same-binary 8K
with a 180 s inter-arm cooldown, N C C N x2: new 1,022.03/1,021.55 and
1,022.08/1,020.90 tok/s, control 918.66/918.01 and 915.78/914.82,
**+11.43%**, every sample stable. The 2K greedy endpoint is byte-identical
(45 bytes). `QWEN_V41_INDEX_TENSOR=0` restores `index_heads_rows`.

## Why

`index_heads_rows` gave each simdgroup eight keys and reloaded the row's query
for every key group; the q traffic dominated. The tensor form stages q once,
loops the key tiles inside the threadgroup, and uses the 32x64x128 tensor
matmul for the per-head dots. The profiled `prefill_attention_serial` span
falls from 2,145.9 to 325.4 ms (the same two-run stage protocol), **-84.8%**.
The remaining serial is the top-k selection sort, not the scores. This is the
largest single prefill win of the session and the direct analogue of
MLX-Serve's tiled `indexScores` kernel.

## Correctness

- A raw GPU witness compares the tensor scores with `index_heads_rows` over the
  fixture rows; the tensor stages q/keys as half but the weighted head sum and
  score layout are unchanged.
- The 2K greedy endpoint is byte-identical to the simdgroup path
  (sha256 `b47cc0f104b62d4c7c30bcd68fd8e67613e287dc4ad8c310ef10cbadea9c4380`).
- The selection sort and every 3-bit/4-bit weight path are untouched.

## Campaign

Campaign lease, 120 s pre-campaign cooldown, discarded stage-profiled warm-up,
then N C C N with 2 runs per arm and **180 s between arms**:

| arm | samples (tok/s) | mean |
| --- | --- | ---: |
| N | 1,022.03, 1,021.55 | 1,021.79 |
| C | 918.66, 918.01 | 918.33 |
| C | 915.78, 914.82 | 915.30 |
| N | 1,022.08, 1,020.90 | 1,021.49 |

The control drift is 0.3%; the new arms differ by 0.03%. The 180 s
inter-arm cooldown is now part of the protocol after the publication campaign's
9% thermal soak.

## Limits

One artifact and one host session. The kernel requires the Metal 4 tensor tier
and 32 index heads x 128 at `last_length > top_k`; other geometry keeps the
scalar/simd path. The remaining serial is the top-k sort; the score precision is
half-staged, but the endpoint and the raw witness cover this fixture. The
publication and attention tile are independent toggles and are on in both arms;
this campaign isolates the index scores.
