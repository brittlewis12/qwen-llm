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

### DeepSeek V4.1-Flash (main after the tensor-ops attention sink: pp2048 Q2 853 / IQ3 852 tok/s, pp4096 Q2 816 / IQ3 815; decode rechecked tg32 Q2 43.2 / IQ3 41.0, depth tg32@d2048 41.6 from the prior landing)

CED encoder-only bulk prefill is the leading architectural opportunity.
Design/dependency recon: landed (2026-10-23 CED entry;
`DEEPSEEK-V41-CED-DESIGN.md`). The first recommended executor is cold shrinking
dependency-preserving reconstruction: at most 2541 boundary rows, but only
24150 complete decoder layer-output rows for long prompts, plus KV-only seeds.
Trained bounded-128 replay is a separate policy using the same primitives.
Cold shrinking native implementation: correctness-qualified on Maya-S
(`perf/v41-ced-scratch`, `eba1f591`, 2026-10-24). Current Full/CED at 128/2048/8192
matches all 257 complete 129,280-logit vectors and all 56 causal regions at every
position. All four arms also match frozen source `01983b7b`, with exact input,
artifact, publication/completion and state metadata; no fallback. This includes
request-local scratch reuse and hierarchical selection. Long-prompt native A/B
remains unqualified. The historical validation-on pp128 pair is a CED loss:
305.17 -> 464.15 ms (+52.1% latency); its 2K sequence rejects a 7.272x
within-Full spread. Explicit production timing (`5e6683a9`, clean source
`a92e67d2`) keeps correctness validation and all safety/stability rules intact.
One supervised pp128 series accepts all ten cases: Full 297.559 ms versus CED
450.183 ms (+51.3% latency), with endpoint heads and recorded artifact/input
metadata matching the validation-on reference. This is not validation-causality
or cross-regime speedup evidence. The next pp2048 launch refuses before import
on the subsequently removed percentage-padded host check. These are historical
results, not a current admission blocker. See
`bench/2026-10-23-v41-ced-native-pair.md` for protocol,
verification and measurement limits. Bounded128 remains a separate policy arm;
whole-model explicit residency remains closed.
Attention and activation-staging kernel lanes remain independent.

CED decoder scratch reuse: in flight (`perf/v41-ced-scratch`). Instrumented
2K attribution measures about 1.18 s in block-row allocation/initialization;
source census identifies 874 removable buffer constructions with one checked
maximum-row decoder backing. Preserve all 38 submissions, original arithmetic
lineage, status initialization and the existing admitted peak. This is a reuse
bet, not a native speedup claim; completion-wall variability remains separate.
Candidate `eec23a0f` passes bounded fresh/reused bitwise, allocation, arithmetic
lineage and cleanup fixtures on M4/M5, plus broader forward/FFN/CED regressions.
Full-model qualification now passes at 128/2K/8K (`eba1f591`) with all heads and
causal state identical to frozen Full/CED. Native gains remain unmeasured.

Mapping/admission repair: landed (2026-10-23 memory-safety entry and
`bench/2026-10-23-v41-memory-safety.md`). Shared EOF removes the old 156.059 GiB
private exposure; combined host admission and bounded import/lifetime proofs
pass on M4/M5. The extra 80%-physical ceiling and 20% availability reserve were
unmeasured application policy, not Metal limits; removed in `d47e264c`. Keep
Metal's working-set limit for Metal demand, CPU availability for additional CPU/
private-copy demand, and actual combined physical capacity. Shared GPU mappings
must not be refused against a cache-blind CPU-availability estimate. The former
204.8 GiB IQ3 cutoff is no longer a valid fit
argument; IQ3 has not been rerun here. Shorter prompts do not shrink weights.
Metadata-only plans report Maya-S 165.950 GiB and Q2 185.907 GiB, both with
zero private shadow. Completed-chunk/scalar lifetimes now return to a fixed
device baseline on M4/M5; Full also prices 202 MiB of logical half staging at
2048 rows. Current Full/compacted attention lineage and fitting-model causal
equivalence pass. The controller credits only conservative inactive non-anonymous
cache availability, not raw free pages alone. No waiver or speedup claim follows
from prior successful runs or a passing Metal-only budget.

Decode budget at depth 2048, 22.3 ms/token: ~10.2 ms bytes (11.1 GiB at the
~1.09 TB/s the GEMV probe reaches), ~8 ms dispatch floor (~1,600 dispatches),
~4 ms spread thin. Prefill split per 2048-token chunk, fresh post-tile six-span sample
(`QWEN_V41_PREFILL_STAGES=1`, PP2048, shares only; the profiler's per-span
waits inflate wall): FFN 50.4%, attention tiles 23.2, finish/post 11.0,
prepare 6.7, owner/producer serial 5.3, HC 3.5. The 32-row sink tile had
attention tiles at 34.7%. Re-sample after the next landing.

- Attention tiles via simdgroup MMA (half-staged Q/K/V, F32 sums),
  replacing the scalar score kernel and the gather+GEMM PV for the released
  geometry: landed (2026-10-21 attention entry). Supersedes the scalar
  score tuning (`b992694a`) on that geometry; other geometries keep it.
  The tile width followed in the 2026-10-22 sink-tile entry; fusion is the
  next attention bet below.
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
- Half activation copies (`_f16act` + `prefill::HalfScratch`): landed
  (2026-10-22 half-activation entry,
  bench/2026-10-22-v41-tensor-half-activations.md; rebased and integrated
  with the 64-slot dense tile 2026-10-23). The rebased build-vs-build
  ABCCBA against the dense landing: IQ3 pp2048 790.2 -> 836.9 (+5.91%),
  pp4096 757.7 -> 799.3 (+5.49%); Q2 788.3 -> 838.9 (+6.42%), 754.5 ->
  805.4 (+6.75%). Held-text logits bitwise identical; 173 tests pass.
  The wide dense Q8_0 form has an `_f16act` twin, so the two landings
  stack. Fused routed gate+up (one activation staging, two weight slabs,
  no gate intermediate) is bitwise correct but flat (IQ3_XXS 9.87 ->
  9.85 ms, MXFP4 10.43 -> 10.44, Q3_K 12.74 -> 12.75, Q8_0 14.35 ->
  14.57): negative (2026-10-23 probe, branch `perf/v41-gateup-fuse`,
  local). The pair is weight-staging/MMA bound; keep two dispatches.
- Cheaper IQ3_XXS/Q3_K expert decoders: negative (2026-10-23 attribution
  packet, bench/2026-10-23-v41-expert-decoder-attribution.md). At the
  same 32-slot shape with f16act operands, simplest-decoder Q8_0 is the
  slowest tile (39.1/39.7 TFLOPS at 649/659 GB/s) while IQ3_XXS reaches
  54.1/49.5 at 324/296 and IQ2_XS 56.9/52.4 at 257/237; the tile is
  weight-byte/staging-bound, not decoder-ALU bound. Remaining routed
  lever: fewer staged weight bytes (native MXFP4 or a lower-bit
  artifact), not a faster decoder.
- Native MXFP4 blockwise right operands: negative (2026-10-23 probe,
  bench/2026-10-23-v41-mxfp4-blockwise-negative.md; branch
  `perf/v41-mxfp4-probe`). The Metal 4.1 `metal_fp4_e2m1_format` +
  E8M0 block-32 scale-plane path compiles and matches a CPU reference
  (K tile must be 256 FP4 elements), but runs at 9.7 TFLOPS (K=2304) /
  9.2 (K=5120) versus 47.3 for the existing half-staged MXFP4 down tile
  at the production shape. Do not integrate; the weight-byte bound needs
  a different route (lower-bit artifact or staging), not this op.
- Dense prefill projections on the tensor tile (q_a/q_b/kv, output_a/b,
  shared expert; the Q8_0 `mat_mat` path at 24-29 TFLOPS): landed
  (2026-10-22 dense entry, bench/2026-10-22-v41-tensor-dense-projections.md).
  54-68 TFLOPS, 2.05-2.42x per dispatch, pp2048 +14.3% / pp4096 +13.2%
  on both artifacts. The 64-slot form (one 64-row weight slab reused for
  two 32-slot activation groups) landed 2026-10-23
  (bench/2026-10-23-v41-dense-wide.md): -16 to -29% per dense dispatch,
  pp2048 +3.64% / pp4096 +3.43% combined (IQ3 +4.15/+3.98, Q2
  +3.14/+2.88), bitwise identical to the 32-slot tile. Enabled at >=128
  query rows; 64/96 rows are 12-14% slower from padding.
  `QWEN_V41_DENSE_NARROW=1` forces the 32-slot form for A/B. A
  64-slot x 128-row combined probe is bitwise correct but 22-57% slower
  (32 KiB output staging, 64-float accumulator); 64x64 is the knee:
  negative (`perf/v41-dense-combined`). The half activation copies
  (below) now feed this tile. Other families' dense prefill GEMMs still
  take `mat_mat`; unmeasured there and outside the V4.1 gain budget: open.
- Attention sink tile occupancy: landed (2026-10-22 sink-tile entry,
  bench/2026-10-22-v41-sink-tile.md). A 32-row tile dispatched
  `heads/16 * 32` threadgroups (128 at 64 heads) against 80 cores and paid
  the dependent `ids -> latent row` gather in each; 512 rows dispatches
  2,048 and costs 6.7 vs 17.2 us per row in the fixture. Same-binary
  ABCCBA (32/256/512): 512 vs 32 pp2048 +17.4-17.6%, pp4096 +18.1% on both
  artifacts; 512 beats 256 by 1.8-2.0%. Census 45,982 -> 27,592 (-40.0%);
  IQ3 held-text witness identical. The gathered value matrix (about 755 MB
  at 512 rows) leaves admission on the MMA geometry, where it is
  unreachable. 1024 rows measured flat (+0.3% pp2048, +0.1% pp4096) and is
  not worth the extra scratch: negative (sink-tile packet).
- Attention sink online fusion: negative (2026-10-23 probe,
  bench/2026-10-23-v41-sink-fusion-probes.md). One threadgroup per
  (16-head block, row) gathering an 8-key block once and running online
  softmax plus PV is 4.6-10.8x slower (36.9 ms vs 3.43 ms at 512 rows):
  the 25 KB threadgroup allows one resident threadgroup per core, the
  per-block barriers serialize gather/score/softmax/PV, and only two of
  eight simdgroups are active in score. Do not rebuild the online route
  without a smaller footprint or more residency.
- Normalize fold into `pv_mma`: negative (same packet). Moving the
  denominator division and finite check into the PV accumulator store
  makes `pv_mma` 1,049 -> 19,187 us (divide) or 19,496 us (reciprocal)
  at 512 rows; dropping the finite guard does not recover it, while
  removing the scaling returns to 1,294 us. The separate normalize pass
  (134 us, 134 MB read/write) is already the cheap form; keep it.
- Attention sink tensor ops: landed (2026-10-23 tensor-sink entry,
  bench/2026-10-23-v41-attn-tensor.md). The Metal 4 tensor-ops
  `score_tensor`/`pv_tensor` tile (matmul2d per 16-key block, same slabs,
  softmax and normalize) runs at 512 rows: IQ3 pp2048 837.9 -> 852.3
  (+1.72%), pp4096 796.9 -> 814.9 (+2.26%); Q2 838.2 -> 852.6 (+1.72%),
  797.6 -> 816.0 (+2.31%). Fixture: 512 rows -3.4%, 256 rows -6.1%, 32
  rows +18% slower, so selected only at >=128 rows on the tensor tier.
  Held-text mean NLL +0.0079 (+0.37%), top-1 unchanged;
  `QWEN_V41_SINK_TENSOR=0` restores the MMA kernels.
- Maya-S artifact (`peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF`,
  six shards, 339.5 GB): evaluated (2026-10-22 Maya entry,
  bench/2026-10-22-maya-s-eval.md). All 40 routed layers on the
  half-staged grouped route, no fallback; IQ2_XXS/IQ3_XXS in the middle
  layers and IQ2_S/IQ3_S in the first/last four. Same-binary A/B/B/A vs
  UD-IQ3_XXS: pp2048 788.5 vs 755.6 (+4.4%), pp4096 755.3 vs 723.8
  (+4.3%) with a 14.0% smaller file; held-text mean NLL 2.2260 vs 2.1464
  (+0.0797) and top-1 0.5521 vs 0.5677. A memory/speed trade, not a
  quality substitute. Maya-XS (IQ1_S) remains unsupported.
- Owner/producer per-row loops (compressor, index key, selection; 8 layers,
  grows with prompt length): compressor prep landed (`a7de76ab`): key norm +
  key/latent RoPE + both cache copies fused 5 -> 1, +0.5-1.0% on top of the
  MMA series. Compressor standalone census 50,710 -> 40,470; final
  MMA-series census 39,563. Selection batching landed (2026-10-22 entry,
  bench/2026-10-22-v41-select-batch.md): compressor/index publication stays
  ordered; prepare and index scores batch over 32-row tiles; pruning above
  16,385 keys keeps the old selector. Same-binary ABBA: Q2 pp2048 +3.05%,
  pp4096 +3.57%; IQ3 +3.46%/+3.80%; census 105,202 -> 88,133; witness
  identical. Fused selection landed (2026-10-22 entry,
  bench/2026-10-22-v41-select-fuse.md): one threadgroup per top-k row runs
  the bitonic network in threadgroup memory and emits chronological IDs
  directly, padded visible length <=2048. Three-arm ABBA: fused vs split
  +15.5%/+11.3% Q2 pp2048/4096, +15.3%/+10.8% IQ3; vs incumbent
  +19.7%/+15.2%, +19.1%/+14.8%; census 88,133 -> 45,982; serial span
  18.7% -> 5.2%; witness identical across all three arms.
  Ratio-2 compressor pair pool: negative/mixed (2026-10-23 probe,
  bench/2026-10-23-v41-pair-pool-negative.md). A 1024-thread pair
  pool+norm replaces the per-pair copies/pool/norm and is correct, but
  Q2 gains +1.4% while IQ3 pp2048 regresses 5.0%; the pair kernel's
  cache/occupancy footprint costs more than the dispatch savings. Use a
  lower-footprint fusion next (copies only or publish rows), not a
  1024-thread pair kernel.
- Selection padded-4096 extension: landed (2026-10-22 entry,
  bench/2026-10-22-v41-select-fuse4k.md). The common path keeps the 16 KiB
  2048-entry selector; a 32 KiB 4096-entry kernel handles late tiles with
  visible length 2049-4096. Q2 pp4096 570.2 -> 622.2 (+9.1%), IQ3 567.2 ->
  616.8 (+8.7%); census 173,913 -> 92,313; same-binary witness identical.
   Above padded 4096 the per-row fallback remains.
- Exact hierarchical top-512 selection above 4096 visible keys: in flight
  (`perf/v41-select-hier`). Source audit finds 6,990 CED producer queries at
  8192 expand into 83,880 legacy sort/chronological dispatches. Partition-local
  top-512 plus exact merge is a bounded experiment, not measured wall-time
  attribution. Preserve comparator/NaN/causal semantics, price tile scratch,
  retain <=4K and pruning fallbacks. Bounded exact IDs/status/canaries and
  513-row/borrowed-control reuse pass on M4/M5. Candidate 8K Full/CED and frozen
  incumbent Full/CED match all 257 vocabulary vectors and 56 causal regions
  at every position. Corrected broad regression passes; production-regime
  wall-time evidence remains open. See `bench/2026-10-23-v41-select-hier.md`.
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

- Ordinary selected-QSA value accumulation: chronological twelve-head reuse
  landed (2026-10-22, M4). Two saved-rollout brackets give prefill -4 to -6%,
  request -3 to -4%, same 256 tokens; final means 26.87 -> 25.36 s prefill and
  35.51 -> 33.97 s request. Actual N32 bands only, unchanged precision/order,
  one admitted 3 KiB logical denominator buffer; tails and decode unchanged.
  F32 MMA remains unqualified: larger leaf gain but routing
  amplification and inconclusive native timing.
  Packet: `bench/2026-10-22-flash-selected-value/README.md`.
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

- M5 dense tensor-ops transfer for ordinary Fast/Half prefill: landed
  (2026-10-22 dense tensor entry). Existing Q6_K/Q8_0 tiles at >=1024 active
  rows cut ordinary 2048/4096 prefill wall 15.1%/14.3%; saved 11K gameplay
  prefill 22.915 -> 20.002 s (-12.7%), request 29.067 -> 26.167 s (-10.0%).
  No buffers, new precision mode, or context change; Exact/F32, short packets
  and M4 keep their paths. All 22 frozen documents give +0.0040 mean NLL
  (central 90% [-0.0003,+0.0081]), one fewer hit/1408; eight tool pairs are
  correct and token-identical. Not a model-wide noninferiority claim.
  Packet: `bench/2026-10-22-glm-tensor-prefill/README.md`.
- M5 routed IQ3_S down-only tensor transfer: landed (2026-10-23 down entry).
  Actual Fast/Half widths 1024/2048 replace 39 stock down calls per chunk;
  fused gate/up, post-down weighting and three IQ4_XS layers stay unchanged.
  Against dense-only, plain 2K/4K wall falls 9.1%/8.7%; saved 11K prefill
  20.261 -> 19.033 s (-6.1%), request 26.483 -> 25.283 s (-4.5%). Cumulative
  22-document NLL delta versus original is -0.0006, central90
  [-0.0034,+0.0021], with mixed strata and two fewer hits/1408; tool outputs
  remain correct and identical. Not exact logits or model-wide noninferiority.
  Packet: `bench/2026-10-23-glm-tensor-down/README.md`.
  Activation staging is a separate follow-on to the shared V4.1 API, not
  duplicate family-local staging infrastructure.
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
