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

### DeepSeek V4.1-Flash (main after the tensor dense projections: pp2048 Q2 545 / IQ3 542 tok/s, pp4096 Q2 496 / IQ3 493; decode tg32 43 / 41, tg32@d2048 41.6)

Decode budget at depth 2048, 22.3 ms/token: ~10.2 ms bytes (11.1 GiB at the
~1.09 TB/s the GEMV probe reaches), ~8 ms dispatch floor (~1,600 dispatches),
~4 ms spread thin. Prefill budget per 2048-token chunk after the tensor
expert tile, sampled spans (`QWEN_V41_PREFILL_STAGES=1`, packet
2026-10-22 tensor expert tile): FFN 33% (33 ms/layer: expert tiles ~20,
shared expert ~7, router/bucket/weight/sum), attention tiles
(QK^T/softmax/PV) 25% (26.5 ms/layer), owner/producer per-row loops 18%
(98 ms x 8 layers), output projections 12.5%, prepare 8.6%, HC 2.6%. The
dense projections then dropped from ~24 to ~11 ms/layer (dense entry);
re-sample before claiming.
- Post-dense attribution: the 33/25/18/12.5/8.6/2.6 split predates tensor
  dense; dense removed ~13 ms/layer from its projection set. Re-sample
  `QWEN_V41_PREFILL_STAGES=1` and `QWEN_V41_CENSUS=1` at pp2048/4096 on both
  artifacts; unchanged attention/serial shares estimate ~29%/~21%, not
  measured: open.

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
- Simdgroup-MMA expert tile efficiency: negative (packet 2026-10-21):
  20-22 TFLOPS against a ~30 TFLOPS ceiling; staging layout and 32-slot
  gate/up tiles were not levers. Superseded on M5 by the tensor tile.
- Metal 4 tensor-ops expert tile: landed (2026-10-22 entry,
  bench/2026-10-22-v41-tensor-expert-tile.md). matmul2d on threadgroup
  half operands reaches ~130 TFLOPS (4.3x simdgroup MMA); the tile runs
  the expert GEMMs at 37-46 TFLOPS, 1.86-2.19x per dispatch, pp2048
  +24.1-24.9% / pp4096 +21.9-23.1% on both artifacts. M5-class only;
  `QWEN_EXPERT_TENSOR_TILE=0` keeps the generic tile. Negative there:
  double-buffered slabs, 64-k steps (the tile wants small resident
  threadgroups). What remains is dequant + staging + the F32 activation
  path.
- Half activation copies (`_f16act` + HalfScratch): branch ABBA reports
  pp2048 +5.0-5.1% / pp4096 +4.4-4.5% against main 545318d3, including the
  conversion scratch, with byte-identical held-text logits: in flight
  (`perf/v41-tensor-f16act`). Per dispatch, routed tiles +9-24% (IQ3_XXS
  gate+up 13.1 -> 10.7 ms, down 6.7 -> 5.9) and dense tiles +14-41%
  (q_a 494 -> 350 us, output_b 2.97 -> 2.24 ms).
- Native MXFP4 right operands (`metal_fp4_e2m1_format` + `ue8m0` block-32
  scales, 27.0 entry points) would replace the software dequant for the 30
  MXFP4 down layers if packed values/scales map exactly; unmeasured: open.
- Dense prefill projections on the tensor tile (q_a/q_b/kv, output_a/b,
  shared expert; the Q8_0 `mat_mat` path at 24-29 TFLOPS): landed
  (2026-10-22 dense entry, bench/2026-10-22-v41-tensor-dense-projections.md).
  54-68 TFLOPS, 2.05-2.42x per dispatch, pp2048 +14.3% / pp4096 +13.2%
  on both artifacts. Larger dense tiles (64 x 64, more operand reuse):
  open; the activation re-read per column tile is owned by
  `perf/v41-tensor-f16act`, not an independent bet. Other families' dense
  prefill GEMMs still take `mat_mat`; unmeasured there and outside the
  V4.1 gain budget: open.
- Attention sink fusion (`score_mma` + `softmax_rows` + `pv_mma` +
  `normalize_rows`): target the slab traffic, repeated gathers and up to
  three dispatches per tile; gain unmeasured: open.
- Attention sink tensor ops: independently time `score_mma`/`pv_mma`
  substitution before combining with fusion; gather-vs-MMA attribution is
  not measured and shares the fusion budget: open.
- Owner/producer per-row loops (compressor, index key, selection; 8 layers,
  grows with prompt length): compressor prep landed (`a7de76ab`): key norm +
  key/latent RoPE + both cache copies fused 5 -> 1, +0.5-1.0% on top of the
  MMA series. Selection batching
  (one dispatch per layer for index scores, one threadgroup per row for the
  top-k select, batched chronological/prepare; the eight owner/producer
  layers cost 75-121 ms each; the serial span includes compressor pooling,
  norm, key projection, publication, indexing and sorting, so not all of it
  is removable). Compressor standalone census 50,710 -> 40,470; final
  MMA-series census 39,563. Selection batching: in flight
  (`perf/v41-select-batch`). Keep position-ordered compressor/key
  publication; batch `index_heads` in 32-row tiles, then fuse selection for
  padded visible length <=2048; incumbent fallback above 2048. Historical
  PP1024 selection families total 16,381 dispatches (`selection_prepare`
  6,141 + four families x 2,560), not ~19k. First validate a three-arm
  producer replay (incumbent; batched index/prepare; + fused
  sort/chronological).
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

- Ordinary packed HC injection, strict four-output projection: negative
  (2026-10-22, M4). Bitwise kernel and native-state witnesses pass; isolated
  projection time -5 to -7% does not improve full prefill at 128/512/2048.
  Production unchanged; `bench/2026-10-22-flash-strict-hc/README.md`.
- Native DFlash2 sampled gameplay: negative (2026-10-22, M4, API validation
  enabled). Decode 30.51 -> 8.81 tok/s; request latency +58.3%. Acceptance
  21.4%; native verification/replay dominate, not drafter compute. Keep opt-in.
  Packet: `bench/2026-10-20-dflash-native-targets/README.md`.
- Native MTP heads (Unsloth): acceptance probe is the gate before building;
  open, behind short-packet target economics.
- Native short-packet verification: measured (2026-10-22). Test-only HC changes
  cut two-row cost from 214 to 122 ms, still above 79 ms serial. Router cutoff
  crossings explain stable numerical amplification; scalar KL improves on the
  bounded fixture but held-token NLL is mixed. HC stays experimental. Further
  packet weight/dispatch amortization is open, ahead of drafter integration.
  Packet: `bench/2026-10-22-native-short-packets/README.md`.
- GSQ-RCO IQ3_S artifact: M5 quality option, -7 to -11% decode on M4; not an
  M4 default (packet 2026-10-20).

### GLM-5.3-Flash

- Ordinary run/benchmark workspace selection: landed (2026-10-22 ordinary
  workspace entry). Existing serving range replaces the fixed 512-row cap;
  saved-rollout prefill 62.51 -> 49.82 s (-20.3%), observed request mean -19.6%
  with a baseline decode outlier. Context unchanged, typed-pressure fallback,
  frozen width; all four 256-token outputs match. Three held-text cases show
  unchanged NLL/hits versus 512. Costs: larger scratch/cancellation intervals;
  serving and DFlash policy unchanged.
  Packet: `bench/2026-10-22-glm-ordinary-workspace/README.md`.
- Native DFlash2 sampled gameplay: negative (2026-10-22, M4, API validation
  enabled). Decode 26.10 -> 5.36 tok/s; request latency +55.8%, acceptance
  14.1%. Same native prefill geometry, no reported cold prefetch. Keep
  opt-in; packet shared with Flash-Next above. Full GLM-5.3 remains separate.
- Near the M4 bandwidth wall; the M5 expert tensor-ops tier is the V4.1
  tensor tile bet above (the generic grouped tile serves GLM's grouped
  layers too).
- Fast/Half short packets: landed (`ce075a09`, 2026-10-22). Half-operand native
  experts/F32 routers at 1-8 rows and Q8 fewrow at 2-4 cut two-row verification
  from 145-146 to 71-80 ms, versus 78-84 ms serial. Worst measured KL 3.07e-7,
  no top-token flips. Other precision modes and spans above eight stay unchanged.
- Native NextN block 45: measured, integration deferred. Native on-policy
  one-step overlap 71.59%, greedy agreement 75% (128 rows); reference head
  2.36 ms warmed, not a native cost. Even free drafting loses at current two-row
  verification cost before rejection replay. Post-output-norm conditioning and
  full head history are verified; raw conditioning is not equivalent.
  Packet: `bench/2026-10-22-native-short-packets/README.md`.
- Two-row weight/dispatch amortization and retaining the accepted prefix without
  full recurrent replay: open, highest speculation leverage. At observed overlap,
  verification needs to fall below roughly 65 ms even before checkpoint/replay
  overhead with the reference head cost; do not build a native head executor first.

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
