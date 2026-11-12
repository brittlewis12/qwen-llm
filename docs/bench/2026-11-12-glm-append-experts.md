# 2026-11-12 — GLM routed expert streaming floor and bucket kernel

Status: Step 1 pricing complete. Step 2 failed the isolated routed-expert speed gate; the streaming kernel implementation was removed. No quality, ordinary follow-up, or serve campaign was run.

GLM-5.3-Flash stock UD-IQ3_XXS, M5 Ultra, `code.txt` (150,273 tokens), 48K and 80K roots, 2,048-row packed-prefill session, capacity 98,304. Actual route IDs were read from `Glm5NextSession::perf_packed_route_ids` after each production-routed append. The M5 capture is `metagross:~/qwen-runs/2026-11-12-glm-append-experts/occupancy.jsonl`; the first occupancy call started at thermal pressure 0. The required gate widths had pressure 0 at both depths; only the final 80K / 1023-row record reached pressure 2.

## 1. Byte floor

The captured model has 42 routed-MoE blocks. Exact bank sizes show 41 IQ2_S gate/up blocks plus one IQ3_S gate/up block; 39 IQ3_S down blocks plus three IQ4_XS down blocks. The width table counts an expert once per block if at least one routed row selects it. `rows p50/p90/max` is the number of append rows routed to an active expert, pooled over all 42 blocks. `stage GB/s` divides this distinct-expert union by the `routed_experts` stage; the gate/up and down columns use their respective profiled expert kernel times.

Streaming references come from the existing 48K B=4 decode-batch census, which runs 32 routed slots per expert block. The floor weights each type by its own measured B=4 rate:

| Weight bank | Logical B=4 bytes | B=4 kernel time | Measured rate |
| --- | ---: | ---: | ---: |
| IQ2_S gate/up | 7.051 GB | 8.134 ms | 866.8 GB/s |
| IQ3_S gate/up exception | 0.231 GB | 0.299 ms | 772.1 GB/s |
| IQ3_S down | 4.498 GB | 4.544 ms | 989.9 GB/s |
| IQ4_XS down exceptions | 0.428 GB | 0.579 ms | 739.1 GB/s |

The main IQ2_S/IQ3_S rates (867 and 990 GB/s) match the outside 870–990 GB/s decode-batch estimate. The 1 IQ3_S gate/up and 3 IQ4_XS down exceptions use their measured native B=4 rates (772 and 738 GB/s). The floor is the sum of each active expert bank byte count divided by its matching rate.

### Depth 48K

| Rows | Active experts / layer | Rows / expert p50 / p90 / max | Union GB | Floor ms | Current stage ms | Stage GB/s | IQ2_S gate/up GB/s | IQ3_S down GB/s |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 66.2 | 1 / 4 / 15 | 25.29 | 28.1 | 145.0 | 174 | 187 | 154 |
| 20 | 78.0 | 1 / 4 / 18 | 29.79 | 33.1 | 84.0 | 355 | 683 | 397 |
| 24 | 90.0 | 1 / 4 / 22 | 34.34 | 38.1 | 94.5 | 363 | 704 | 408 |
| 32 | 103.4 | 2 / 5 / 30 | 39.47 | 43.8 | 106.9 | 369 | 718 | 412 |
| 48 | 124.4 | 2 / 6 / 45 | 47.48 | 52.7 | 126.9 | 374 | 735 | 415 |
| 64 | 145.0 | 2 / 7 / 60 | 55.35 | 61.5 | 146.2 | 379 | 748 | 417 |
| 96 | 168.1 | 3 / 10 / 82 | 64.19 | 71.3 | 170.2 | 377 | 750 | 414 |
| 112 | 179.2 | 3 / 10 / 92 | 68.42 | 76.0 | 182.0 | 376 | 751 | 411 |
| 128 | 185.3 | 3 / 11 / 106 | 70.76 | 78.6 | 191.6 | 369 | 707 | 434 |
| 161 | 211.7 | 4 / 12 / 119 | 80.85 | 89.9 | 218.5 | 370 | 708 | 432 |
| 192 | 222.5 | 4 / 14 / 136 | 84.95 | 94.4 | 233.3 | 364 | 699 | 424 |
| 256 | 244.9 | 6 / 17 / 158 | 93.50 | 103.9 | 263.3 | 355 | 686 | 414 |
| 384 | 256.9 | 8 / 25 / 237 | 98.09 | 109.0 | 301.8 | 325 | 637 | 375 |
| 512 | 265.4 | 11 / 32 / 344 | 101.30 | 112.6 | 342.5 | 296 | 580 | 346 |
| 768 | 271.6 | 15 / 47 / 543 | 103.65 | 115.1 | 387.5 | 267 | 558 | 297 |
| 1023 | 274.8 | 20 / 62 / 730 | 104.88 | 116.5 | 461.9 | 227 | 478 | 252 |

### Depth 80K

| Rows | Active experts / layer | Rows / expert p50 / p90 / max | Union GB | Floor ms | Current stage ms | Stage GB/s | IQ2_S gate/up GB/s | IQ3_S down GB/s |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 76.2 | 1 / 3 / 14 | 29.07 | 32.3 | 162.0 | 179 | 193 | 158 |
| 20 | 87.7 | 1 / 3 / 16 | 33.48 | 37.2 | 92.6 | 361 | 701 | 407 |
| 24 | 99.6 | 1 / 4 / 19 | 38.01 | 42.2 | 103.3 | 368 | 718 | 414 |
| 32 | 113.6 | 1 / 4 / 27 | 43.37 | 48.2 | 116.0 | 374 | 730 | 418 |
| 48 | 137.0 | 2 / 6 / 43 | 52.28 | 58.1 | 137.6 | 380 | 743 | 421 |
| 64 | 156.5 | 2 / 7 / 56 | 59.74 | 66.4 | 155.8 | 383 | 757 | 423 |
| 96 | 178.2 | 3 / 9 / 85 | 68.02 | 75.6 | 178.6 | 381 | 758 | 418 |
| 112 | 185.1 | 3 / 11 / 101 | 70.65 | 78.5 | 186.5 | 379 | 747 | 413 |
| 128 | 197.0 | 3 / 11 / 115 | 75.21 | 83.6 | 203.9 | 369 | 692 | 435 |
| 161 | 219.8 | 4 / 12 / 126 | 83.91 | 93.2 | 228.0 | 368 | 708 | 432 |
| 192 | 231.3 | 4 / 14 / 147 | 88.33 | 98.2 | 241.0 | 367 | 711 | 431 |
| 256 | 240.1 | 5 / 18 / 202 | 91.67 | 101.9 | 260.8 | 351 | 682 | 409 |
| 384 | 251.8 | 7 / 27 / 315 | 96.13 | 106.8 | 302.1 | 318 | 625 | 371 |
| 512 | 260.5 | 10 / 34 / 402 | 99.45 | 110.5 | 343.3 | 290 | 568 | 340 |
| 768 | 270.7 | 15 / 48 / 604 | 103.33 | 114.8 | 388.7 | 266 | 554 | 294 |
| 1023 | 274.9 | 20 / 64 / 751 | 104.89 | 116.5 | 466.0 | 225 | 476 | 252 |

At 80K, the byte floor is 48.2 / 66.4 / 93.2 / 101.9 / 110.5 ms at 32 / 64 / 161 / 256 / 512 rows, against routed stages of 116.0 / 155.8 / 228.0 / 260.8 / 343.3 ms. The stage is 2.35× or more above the type-weighted floor at every required width, so the 1.5× build gate passes. At 161 rows the measured routed stage streams the union at 368 GB/s; the type-specific expert kernel rates are 708 GB/s IQ2_S gate/up and 432 GB/s IQ3_S down. The remaining stage time includes route bucketing and weighted reduction.

## 2. Streaming kernel A/B

The first prototype used one threadgroup grid entry per expert, output tile, and possible bucket chunk. Its sparse chunk axis launched many empty groups (1.54 s at 32 rows through 6.22 s at 512 rows). The compact prototype switched to one expert/channel workgroup looping over the expert’s actual bucket in 1–16-row pieces. The M4 synthetic witness was bitwise equal to the existing decode-batch IQ2_S/IQ3_S kernels for bucket sizes 1, 2, 4, 8 and 16, with 32 rows forcing a multi-piece bucket.

The corrected M5 A/B ran at 80K on the code corpus, with three interleaved reps per route and stage. Route profiling confirms the stream path was selected on 39 MoE blocks: the stream arm dispatched `kernel_moe_route_bucket_slots_parallel_f32` 39 times at both profiled widths (32 and 512); the three IQ4_XS down blocks used their existing fallback. The route probe’s kernel-name filter omitted the `kernel_glm_stream_*` names, so the raw record proves selection through the stream-only route-bucket dispatch count rather than listing the stream gate/up and down kernel names.

| Rows | Tensor-IQ stage ms | Stream stage ms | Stage speedup | Tensor-IQ wall ms | Stream wall ms | Wall speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 115.85 | 112.36 | 1.03× | 230.60 | 226.72 | 1.02× |
| 64 | 157.45 | 189.53 | 0.83× | 311.10 | 341.55 | 0.91× |
| 161 | 217.08 | 411.62 | 0.53× | 382.03 | 576.30 | 0.66× |
| 256 | 250.21 | 638.56 | 0.39× | 457.64 | 845.53 | 0.54× |
| 512 | 322.03 | 1,367.81 | 0.24× | 691.64 | 1,738.29 | 0.40× |

The required 1.25× isolated-stage gate failed at every tested width: even 32 rows reached only 1.03×, while the 161-row median-width target was 0.53× and 512 rows was 0.24×. Thermal pressure was 1 across the measured reps (0 before the model load, 1 after); both arms were interleaved under the same M5 load. This does not erase the floor headroom in Step 1: at 80K the type-weighted floor is 2.35× or more below the incumbent stage at all five widths. It shows this streaming implementation did not convert that headroom into a faster kernel.

The ordinary prefill wall is included for context; no separate follow-up wall campaign was needed after the stage gate failed. One-shot route logits had top-1 agreement at every width, but max absolute logit deltas ranged from 0.555 to 2.189 and KL from 0.000026 to 0.007206. These are screening outputs, not the paired non-inferiority protocol. No quality gate or three-agent serve A/B was run. The streaming kernel, route selector, route arm, synthetic kernel test, and rollback knob were removed after the performance failure.

Evidence: `metagross:~/qwen-runs/2026-11-12-glm-append-experts/route-modefix-80k.jsonl`, plus `occupancy.jsonl` for Step 1. No M5 bundle, worktree, or build output is retained.
