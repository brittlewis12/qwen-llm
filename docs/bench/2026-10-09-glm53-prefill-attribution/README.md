# GLM-5.3-Flash packed-prefill attribution and chunk size (2026-10-09)

Where Fast prompt reading spends its time, and what larger chunks buy.
Release builds without Metal validation, M4 Max, UD-IQ3_XXS, Fast with the
default half precision unless stated.

## Stage attribution (`packed_prefill_stage_attribution`)

`Glm5NextSession::prefill_chunk_stage_profiled` encodes one packed chunk as
one command with a timestamp-sampled encoder per stage of every block (the
decode recorder's method); its logits equal an unprofiled chunk's bitwise
(asserted). One 512-row chunk, spans summed over blocks:

| Stage | depth 0 | depth 0, `f32` precision | depth 1,536 | depth 4,096 |
|---|---:|---:|---:|---:|
| routed experts | 1,374 ms (61.2%) | 1,749 ms (63.3%) | 1,532 ms (55.8%) | 1,510 ms (53.4%) |
| KDA mixer | 488 ms (21.7%) | 578 ms (20.9%) | 543 ms (19.8%) | 551 ms (19.5%) |
| MLA mixer (incl. attention) | 153 ms (6.8%) | 179 ms (6.5%) | 416 ms (15.2%) | 509 ms (18.0%) |
| shared expert | 95 ms (4.2%) | 112 ms (4.0%) | 108 ms (3.9%) | 107 ms (3.8%) |
| dense FFN (blocks 0-2) | 37 ms | 41 ms | 41 ms | 41 ms |
| router | 25 ms | 26 ms | 25 ms | 27 ms |
| mHC pre and post (4 stages) | 73 ms | 76 ms | 77 ms | 78 ms |
| head, embed | 2 ms | 3 ms | 2 ms | 2 ms |
| **unprofiled wall** | **2,214 ms (231 tok/s)** | **2,707 ms (189 tok/s)** | **2,798 ms (183 tok/s)** | **2,684 ms (191 tok/s)** |

- Routed experts dominate: ~2.7 ms per token at depth 0. With 512 rows and
  top-8 of 288 experts, each expert sees ~14 tokens per chunk, so every
  quantized weight tile is dequantized for few tokens.
- MLA (attention included) grows with depth; KDA is flat.
- In `f32` precision the expert stage carries most of the added cost
  (+375 ms of +518 ms per chunk).

## Chunk size (`packed_prefill_rows_cost`, `rows-cost.json`)

Fresh prompts, A-B-C-C-B-A per length after warm-ups:

| Prompt | 512-row chunks | 1,024 | 2,048 |
|---|---:|---:|---:|
| 2,048 tokens | 10,570 ms (194 tok/s) | 9,947 ms (206 tok/s, +6%) | 9,255 ms (221 tok/s, +14%) |
| 4,096 tokens | 21,247 ms (193 tok/s) | 19,570 ms (209 tok/s, +9%) | 18,313 ms (224 tok/s, +16%) |
| packed scratch | 0.60 GiB | 1.17 GiB | 2.30 GiB |

Larger chunks give each expert more tokens per chunk; the gain is bounded
by the grouped gate/up tile's 16-token width (each 16-token tile
dequantizes its weights again). Chunking changes Fast's arithmetic
schedule (as the shared-prefix split does), not its kernels or precision.

## Decode experts: two quick ideas falsified

The decode routed-expert stage is 10 ms of 30.5 ms per token (IQ2_S gate/up
at ~293 GB/s, IQ3_S down at ~347 GB/s; `routed_expert_block_dispatch_costs`,
42 blocks per command, A-B-B-A):

- Codebooks copied into threadgroup memory (outputs bitwise equal): 258
  vs 234 us per block, slower.
- Gate/up epilogue in registers instead of a threadgroup round trip and
  barrier (bitwise equal): 144.8 vs 144.7 us, no change.

Neither was kept.
