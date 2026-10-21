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

### DeepSeek V4.1-Flash (`main e78f1d60`, code `c4125c6b`: pp2048 Q2 339 / IQ3 337 tok/s, pp4096 Q2 319 / IQ3 316; decode unchanged at tg32 40.8, tg32@d2048 41.6)

Decode budget at depth 2048, 22.3 ms/token: ~10.2 ms bytes (11.1 GiB at the
~1.09 TB/s the GEMV probe reaches), ~8 ms dispatch floor (~1,600 dispatches),
~4 ms spread thin. Prefill budget per 2048-token chunk, sampled spans
(`QWEN_V41_PREFILL_STAGES=1`, packet 2026-10-21): FFN 39%, attention tiles
(QK^T/softmax/PV) 32%, owner/producer per-row loops 13%, output projections
8%, prepare 6%, HC 2%.

- Attention tiles via simdgroup MMA (half-staged Q/K/V, F32 sums), replacing
  the scalar `score_rows` and the gather+GEMM PV: in flight
  (`perf/v41-prefill-attn`, scalar-equivalence tests pass; pp2048 pending).
  Supersedes scalar score-kernel tuning if it wins.
- IQ2_XXS/IQ2_XS expert layers off the native-rows path: landed with the
  F32-operand grouped tiles (`perf/v41-prefill-land`, `b3b71bb3`); the merged
  base moved Q2 +59.8% and IQ3 +25.4% at pp1024. The half generic layout on
  `perf/v41-prefill-attn` was not isolated for a one-ABBA comparison; reconcile
  before both land.
- MXFP4 generic half layout: +1.6% alone (negative-ish, packet 2026-10-21);
  needed for the IQ2 half route on layers whose down is MXFP4.
- Expert tile efficiency (~15 TFLOPS effective on grouped F32x at pp1024,
  ~8.7 TFLOPS on half-staged tiles at 32 rows/expert): open. Tried:
  2048-row chunks +5.6% (landed); 32-slot gate/up tile +0.25-0.9% at pp2048
  but -1.3%/-1.6% at pp1024 (negative, packet 2026-10-21); 4096-row chunks
  untested.
- Owner/producer per-row loops (compressor, index key, selection; 8 layers,
  grows with prompt length): in flight (`perf/v41-compressor-prep`). Census:
  `attention_copy` 8,192, `selection_prepare` 6,141, `attention_rope` 5,120,
  `rope_rows` 4,038 per Q2 PP1024. First bet: fuse per-group norm+RoPE+copies;
  selection batching follows.
- Scalar score kernel tuning: landed (`b992694a`). 4 keys/simdgroup +3.0%,
  8-lane +1.2-1.4% on the merged base; 8 heads/threadgroup and fatter key
  loops were negative. The MMA tile still supersedes this if it wins.
- Decode: fused attention core, routed gate/up fusion, rope/copy fusion are
  each <= 1 ms/token: open, low priority. Fused decode expert chain: +1%/+0%
  (negative, packet 2026-10-20). Q8_0 attention projections (5.1 GB/token)
  requant to Q6_K: ~-1 ms, quality-sensitive, open.
- Prefill dispatch census tool: landed (`c4125c6b`, `QWEN_V41_CENSUS=1`).

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
