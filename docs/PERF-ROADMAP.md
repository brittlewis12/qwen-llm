# Performance Roadmap

Current-state and evidence index. This page records what has shipped and which
questions the evidence has closed. Dated measurements remain in
[PERF-LOG](PERF-LOG.md) and the linked benchmark packets.

## Shipped state

- Family serving, request timing, and benchmark contracts are documented in
  [SERVE](SERVE.md), [REQUEST-TIMING](REQUEST-TIMING.md), and [BENCH](BENCH.md).
- Current profiling coverage includes the Metal trace parser and
  `qwen-bench metal-counters`; see [PERF-TOOLS](PERF-TOOLS.md).
- Automatic dense refill, ragged selection, and file-root reuse have dated
  evidence in [automatic dense ragged refill](bench/2026-08-12-automatic-dense-refill/README.md),
  [automatic dense ragged selection](bench/2026-08-12-qwen-ragged-fixed-cohorts/README.md),
  and [fixed-cohort file roots](bench/2026-08-12-qwen-fixed-file-root/README.md).
- DeepSeek V4.1 current requirements and operation live in
  [DEEPSEEK-V41-STRATEGY](DEEPSEEK-V41-STRATEGY.md) and
  [DEEPSEEK-V41-BRINGUP](DEEPSEEK-V41-BRINGUP.md).

## Open leverage map (living; read before starting a bet, update when you start or finish one)

Numbers are M5 Ultra unless noted. A bet is one line: what, measured or
expected effect, status. Status is one of `open`, `in flight (branch)`,
`landed (entry)`, or `negative (packet)`. Claim a bet by setting it to
`in flight` with your branch; a second lane finding it claimed reads the branch
before building anything adjacent. This replaces relaying between sessions.

### DeepSeek V4.1-Flash (main after the MMA sink tile: pp2048 Q2 381 / IQ3 383 tok/s, pp4096 Q2 356 / IQ3 358; decode tg32 44 / 41.5, tg32@d2048 41.6)

Decode budget at depth 2048, 22.3 ms/token: ~10.2 ms bytes (11.1 GiB at the
~1.09 TB/s the GEMV probe reaches), ~8 ms dispatch floor (~1,600 dispatches),
~4 ms spread thin. Prefill budget per 2048-token chunk before the MMA tile,
sampled spans (`QWEN_V41_PREFILL_STAGES=1`, packet 2026-10-21): FFN 39%,
attention tiles (QK^T/softmax/PV) 32%, owner/producer per-row loops 13%,
output projections 8%, prepare 6%, HC 2%. After it, FFN is roughly half of
prefill and the tiles about a fifth; re-sample before claiming.

- Attention tiles via simdgroup MMA (half-staged Q/K/V, F32 sums),
  replacing the scalar score kernel and the gather+GEMM PV for the released
  geometry: landed (2026-10-21 attention entry). Supersedes the scalar
  score tuning (`b992694a`) on that geometry; other geometries keep it.
  Next there: a fused score+softmax+PV kernel (saves the slab round trip
  and three dispatches per tile): open.
- IQ2_XXS/IQ2_XS expert layers off the native-rows path: landed twice, as
  F32-operand grouped tiles (`b3b71bb3`, the NativeF32 witness path) and as
  half generic layouts (attention entry; production takes the half route
  like every other routed layer). One ABBA on the MMA path: Q2 tie
  (378.6 vs 378.45), IQ3 half +1.0% (380.25 vs 376.55); half stays
  production and the F32x tiles stay the NativeF32 witness.
  (bench/2026-10-21-v41-prefill-mma-final.md).
- MXFP4 generic half layout: landed with the IQ2 layouts (needed for layers
  whose down is MXFP4); +1.6% on its own. Widens DFlash/GLM grouped
  eligibility on MXFP4 artifacts (unmeasured there).
- Expert tile efficiency (~8.7 TFLOPS on half-staged tiles; per-layer FFN
  time is 52-64 ms regardless of storage type, so not decode-bound): in
  flight (`perf/v41-expert-tiles`, isolated tile microbench first). Tried:
  2048-row chunks +5.6% (landed); 32-slot gate/up tile +0.25-0.9% at
  pp2048 but -1.3%/-1.6% at pp1024 (negative, packet 2026-10-21);
  4096-row chunks untested.
- Owner/producer per-row loops (compressor, index key, selection; 8 layers,
  grows with prompt length): compressor prep landed (`a7de76ab`): key norm +
  key/latent RoPE + both cache copies fused 5 -> 1, +0.5-1.0% on top of the
  MMA series, census 50,710 -> 39,563 with the MMA tile. Selection batching
  is open: `selection_prepare` 6,141,
  `sort_init`/`sort_tile` 2,560 each, `chronological` 2,560, `index_heads`
  2,560 per Q2 PP1024.
- Scalar score kernel tuning: superseded on the released geometry by the
  MMA tile; the 8-lane scalar kernel remains the fallback for other
  geometry. The 4-key/8-lane variants measured +3.0%/+1.2-1.4% on the old
  scalar path; 8 heads/threadgroup and fatter key loops were negative.
- Decode: fused attention core, routed gate/up fusion, rope/copy fusion are
  each <= 1 ms/token: open, low priority. Fused decode expert chain: +1%/+0%
  (negative, packet 2026-10-20). Q8_0 attention projections (5.1 GB/token)
  requant to Q6_K: ~-1 ms, quality-sensitive, open.
- Prefill dispatch census tool: landed (`c4125c6b`, `QWEN_V41_CENSUS=1`).
- Six-span prefill profiler (`QWEN_V41_PREFILL_STAGES=1`, per-layer HC /
  prepare / serial / tiles / finish / FFN): landed; use it before claiming
  a prefill bet.
- Final reconciliation: `bench/2026-10-21-v41-prefill-mma-final.md`.

### Qwen3.8 Flash-Next

- Native MTP heads (Unsloth): acceptance probe is the gate before building;
  open.
- GSQ-RCO IQ3_S artifact: M5 quality option, -7 to -11% decode on M4; not an
  M4 default (packet 2026-10-20).

### GLM-5.3-Flash

- Near the M4 bandwidth wall; M5 expert tensor-ops tier is owned by the
  maintainability lane.

## Closed questions

- The owned-arena loader did not pass loaded-decode and whole-request gates;
  the descriptor-derived population plan is the shipped path. See the
  [2026-10-19 entry](PERF-LOG.md#2026-10-19-descriptor-derived-gguf-population-plan-owned-arena-archived).
- The GLM Q4 pilot showed no quality advantage over the incumbent IQ3 artifact
  in its registered cases. See the
  [2026-10-17 entry](PERF-LOG.md#2026-10-17---three-way-glm-pilot-finds-no-q4-quality-advantage).
- ReplaySSM precision separation did not justify changing production FP32
  recurrence. See the
  [2026-10-13 entry](PERF-LOG.md#2026-10-13---replayssm-precision-separation-works-keep-production-fp32).
- Whole-model DeepSeek residency remains closed because forced termination left
  system-wide wired memory. See the
  [2026-08-13 entry](PERF-LOG.md#2026-08-13-whole-model-residency-optimization-lane-closed).

## Historical planning maps

The dated leverage maps, ranked queues, and their evidence pointers are retained
verbatim in [PERF-ROADMAP-2026-10-19](archive/PERF-ROADMAP-2026-10-19.md).
