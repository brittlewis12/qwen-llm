# V4.1 Decode Attribution On Current Main (Warm, GSQ-RCO IQ2_XS)

Main `64e314a1` plus the decode stage hook (`perf/v41-decode-stages`), M5
Ultra 80-core, 256 GiB, 240 GiB wired limit, production lease. Artifact:
GSQ-RCO IQ2_XS (~144.6 GiB resident), the candidate default from the
2026-10-25 GSQ baseline. Raw: `metagross:~/qwen-runs/2026-10-25-decode-stages/`.

## Protocol

Each model runs four `qwen-bench suite --pp 512 --tg 32 -d 0,2048` processes
back to back: a warm-up (cold page cache), a plain warm run, a run with
`QWEN_V41_DECODE_STAGES=1` (per-token sampled stage envelopes) and a run with
`QWEN_V41_CENSUS=1` (per-token dispatch census). Steps in the first quarter
of each depth group are dropped from the stage medians.

The stage hook splits a decode step into one sampled encoder per stage
(Engram projections; per layer HC pre, attention, HC post, FFN; head). For
GSQ the sampled stage sum matches the unsampled command time (18.5 vs 18.4
ms at depth 0), so the shares are trustworthy. For IQ3_XXS (~217 of 240 GiB)
the same splits cost 2.5 s (depth 0) to 7-30 s (depth 2048) per token with
~36 ms of sampled stage time and no page-ins or disk reads: near the memory
ceiling every encoder boundary appears to pay a residency cost. That arm was
stopped (SIGINT, clean cancel); its stage shares are not used here. The same
effect likely inflated the 2026-10-21 IQ3 prefill-stage wall times.

## Throughput (warm vs cold)

| depth | cold-load tg32 | warm tg32 | warm ms/token |
|---|---:|---:|---:|
| 0 | 42.5 | 55.2 | 18.1 |
| 2048 | 46.7 | 46.9 | 21.3 |

## Budget per token, depth 0 (GPU 18.4 ms, wall 20.1 ms)

| stage | GPU ms | share | weight bytes/token | effective BW |
|---|---:|---:|---:|---:|
| FFN (router, shared expert, top-6 routed, weight/sum) | 8.67 | 46.8% | 3.30 GiB | 408 GB/s |
| attention (projections, compressor/index, core, output) | 7.34 | 39.6% | 3.49 GiB | 510 GB/s |
| HC pre (projection, Sinkhorn, collapse, norm) | 1.18 | 6.4% | 0.07 GiB | 66 GB/s |
| Engram projection | 0.63 | 3.4% | 0.20 GiB | 344 GB/s |
| head | 0.53 | 2.8% | 0.42 GiB | 867 GB/s |
| HC post | 0.17 | 0.9% | - | - |
| **total** | **18.5** | | **7.28 GiB** | **425 GB/s** |

At depth 2048 attention grows to 9.57 ms (46.6%) and everything else is
unchanged (20.5 ms total). The bytes floor at the ~1.09 TB/s the GEMV probe
reaches is ~7.2 ms, so decode runs at 2.5x its floor; IQ3_XXS (10.89 GiB per
token, ~10.7 ms floor, 19.6 ms warm) at 1.8x. GSQ's attention weights are
Q5_K (3.0 GiB) where Unsloth's are Q8_0 (5.0 GiB).

## Dispatch census and dispatch cost

1,317 dispatches per token (was ~1,600 on 2026-10-21). Top kernels: 243
native Q5_K GEMVs (attention projections), 128 RoPE, 84 RMS norm, 80 each of
HC pre / HC projection / HC post / native Q8_0 GEMV, 76 native IQ2_XXS, 67
BF16 mat-vec, 40 each of route / weight / sum / clamped SwiGLU / attention
copy / score / softmax.

`dispatch_chain_latency_timing` and `dispatch_chain_tiny_marginal_timing`
(`QWEN_DISPATCH_LATENCY=1`, leased; 1,300-dispatch chains in one command
buffer):

| per dispatch | serial chain | concurrent, independent | concurrent + barrier |
|---|---:|---:|---:|
| 256 floats | 1.66 us | 0.17 us | 3.45 us |
| 5,120 floats | 1.21 us | 0.16 us | 1.81 us |
| tiny dispatch added between 4 MiB copies (marginal) | 1.65 us | | |

Serial encoders serialize independent work too (1.36 us vs 1.21 us), and a
memory barrier per dispatch in a concurrent encoder is slower than a serial
chain. Independent dispatches in a concurrent encoder cost ~0.16 us.

## Reading

Of the ~11 ms between warm decode (18.4 ms GPU + 1.7 ms host) and the bytes
floor (7.2 ms):

- ~2.2 ms is serial dispatch overhead (1,317 x ~1.65 us);
- ~1.7 ms is host time per token (encoding ~1,300 dispatches before the
  command starts, commit/wait, logits copy), during which the GPU is idle;
- ~7 ms is FFN and attention GEMVs reaching only 37-47% of bandwidth: small
  per-layer matrices (a routed IQ2 expert matrix is ~3.4 MB, ~3 us at full
  bandwidth) cannot ramp up in a strictly serial chain.

Levers, largest first:

1. **Concurrent encoding of independent decode work.** Within a layer, the
   shared expert, router and routed experts are independent until the
   weight/sum, and the query and KV projection chains are independent until
   the attention core. Overlapping them both removes their serial dispatch
   costs (0.16 vs 1.65 us) and lets small GEMVs share the GPU. The repo
   already ships this pattern for Qwen MoE decode
   (`QWEN_DECODE_MOE_CONCURRENT_SHARED`); V4.1 decode is one serial encoder.
2. **Host overlap.** Commit the step in two or more command buffers so the
   GPU starts while the host encodes the rest, or pre-encode an indirect
   command buffer per step shape; worth up to ~1.7 ms.
3. **Finer FFN/attention attribution** (router vs shared vs routed vs
   weight/sum; projections vs core) to rank kernel-level work; the stage hook
   takes more boundaries cheaply on GSQ.
4. HC pre (1.2 ms for 0.07 GiB) is pure latency: two dispatches per layer
   plus norm; a candidate for fusing into the previous layer's FFN epilogue.

## Limits

- One artifact (GSQ). IQ3's split-encoder pathology prevents stage shares
  there; its unsampled warm decode is 50.9 tok/s at depth 0.
- Stage envelopes are inclusive per encoder; with ~160 encoders the per-
  encoder overhead is inside the shares (GSQ: within 1% of unsampled).
- The dispatch probes measure copies, not GEMVs; they bound the launch cost,
  not the GEMV ramp/tail behaviour.
