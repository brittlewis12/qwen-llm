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

### DeepSeek V4.1-Flash (main after the 4096-row chunk and 64-slot gate/up: pp2048 Q2 877.3 / IQ3 852 (not re-measured), pp4096 Q2 890.2 / IQ3 896.8; decode rechecked tg32 Q2 43.2 / IQ3 41.0, depth tg32@d2048 41.6 from the prior landing)

Current implementation campaign (2026-10-25): close the trust-boundary defects
before promoting the experimental CED composition. Independent owned lanes:

- Tensor staging ownership: landed (`e97bd5c0`, qualified source `7d80ac08`).
  Unique physical staging and zero-filled invalid rows pass eight focused CPU
  and fourteen actual M5 GPU tests, including all 11,951 strict buffer-release
  assertions. Independent structural ownership checks reject the historical map.
- Executable family contracts: landed (`de082865`, qualified source `0d6bafaa`).
  Shared backend eligibility and activation-alias checks preserve descriptive
  plans and distinguish memory fit. General output-group fallback passes an
  independent stored-precision M5 oracle at 1/8/16/17/32 groups.
- Request planning/lifecycle: in flight (`fix/v41-request-planning-v33`). Release
  obsolete live state before avoidable admission refusal, report actual selected
  geometry, and choose admitted chunk candidates before mutation. Keep Metal/CPU
  accounting domains and pending debt intact; do not shrink context or retry a
  poisoned state. Resolve live/snapshot preference explicitly rather than assuming
  longest prefix always minimizes request latency.
- Current acceptance machinery: landed (`16c70885`, `45a3ce68`, `0a4c676f`).
  Exact CPU/exclusion inventory, current suffix census, generated environment
  reference, identity probes and checked comparison reporters replace stale
  expectations. Actual main `de082865` ring capture passes dispatch and
  restore/growth self-parity but fails independent attention output tolerances
  in all 120 rows; all selection/owner and applicable candidate IDs agree.
  Numerical localization is in flight; do not weaken the acceptance gate.

The parent integration lane will compose reviewed changes onto current main,
then qualify retained-window mapping and explicit cold exact CED without losing
GSQ/IQ1 or newer chunk support. P4 remains frozen historical evidence; its old
Full/CED equality is not a race-freedom or independent model-numerics proof.
Serving/CED-origin reuse has a separate acceptance boundary from cold run/bench.
Long-context selector work removal and complete short-packet expert dataflow
remain independent structural bets, not another adaptive-tail width sweep.
The first exact selector leaf is in flight (`perf/v41-mask-valid-ids-v35`).
Source `3ab20576` passes 144 actual M5 correctness cases and retains all 864 native
samples, including the extra clear/scatter dispatch. Large-domain isolated times
are promising, but mixed timed IDs contain invalid/duplicate entries. Confirm
valid unique ascending/ranked IDs before inclusive producer/consumer and model
gates; this leaf neither removes dense mask allocation nor qualifies model speedup.
Current proof identities, failure metrics and limits are in
[runtime-contract qualification](bench/2026-10-25-v41-runtime-contracts.md).
Before automatic policy selection, measure homogeneous request streams as well
as mixed transitions: the retained P4 8K same-policy Full calls are 9.566/9.605 s
and same-policy CED calls 8.681/8.513 s. Two observations per transition do not
establish a replacement speedup estimate, but make predecessor effects material.

GSQ-RCO release bring-up: loader bridge and IQ1 expert wiring landed
(`0079569a`, `d1ad718c`), pinned HF revision
`cb02422c900df489cea5833e8a92171224059b3c`.
Reuse established Flash-Next standard-GGUF/GSQ-RCO machinery and IQ1 codecs;
this is DeepSeek-specific source/control and expert-bank wiring, not a new
quantization stack. The recommended 2.00-bpw build has 60 IQ1_M expert banks;
2.25 bpw uses already-supported expert types. Actual prefixes declare a separate
IQ4_NL Engram companion and four integer control tensors. Preserve separate
source groups/stamps, disk-backed tables and current admission. Qualify the new
artifact independently; do not substitute it into frozen Maya/P0/P4 controls or
treat the author's benchmark scores as local quality/performance evidence.
The bridge passes 109 focused CPU checks and authentic sparse header/control
replay; independent review finds no concrete P1/P2. The five requested IQ2_XS
and auxiliary files now pass pinned full LFS hashes (265821115072 bytes total).
Actual M5 validation-on loader fixtures pass 71 tests, including eight GPU
fixtures. Separate Full/CED artifact/source-group qualification now passes
128/2048/8192 at `05edb66f` (`test/v41-gsq-ced-qualification`): all 257 complete
heads and 56 causal regions per position match within the new artifact, with
one import, one KV state and no fallback. Resident shared views are 144.560 GiB,
private shadow zero, with the companion disk-only. These ordered validation-on
pairs are not quality or performance evidence and do not ship the CED executor;
see
[the serialization and ownership contract](DEEPSEEK-V41-GSQ-LOADER.md).
IQ1_S/M adapters reuse the canonical decoder and pass M4/M5 witnesses, including
65 distinct M5 checks across overlapping filters. Joint CPU plans accept both
XS/XXS variants; smaller full IQ1 artifacts are not execution-qualified. The
smallest build's audited weight views are about 112.425 GiB, leaving only
5.575 GiB below a 118 GiB MWSS before runtime/CPU costs. This is capacity
reconnaissance, not a fit guarantee or permission to change admission.

CED encoder-only bulk prefill is the leading architectural opportunity.
Design/dependency recon: landed (2026-10-23 CED entry;
`DEEPSEEK-V41-CED-DESIGN.md`). The first recommended executor is cold shrinking
dependency-preserving reconstruction: at most 2541 boundary rows, but only
24150 complete decoder layer-output rows for long prompts, plus KV-only seeds.
Trained bounded-128 replay is a separate policy using the same primitives.
Cold shrinking native implementation: correctness-qualified on Maya-S
(`perf/v41-ced-scratch`, current 4K/wide control `be874b3f`, 2026-10-24).
Current Full/CED at 128/2048/2049/4097/8192 matches all 257 complete
129,280-logit vectors and all 56 causal regions at every position. The 128/2K/8K
heads/state also preserve the measured bridge through `efea4908` to frozen
`01983b7b`; the two new transition sizes qualify current Full, not an unmeasured
pre-4K oracle. Exact input, artifact, publication/completion and state metadata;
no fallback. This includes
request-local scratch reuse, hierarchical selection and seed/first-decoder fusion.
Fresh bounded and full-model qualification includes repaired main's 32-key tensor
score, with default/rollback lineage checks and exact pre-rebase heads/state.
Stable long-prompt native performance remains unqualified. The historical
validation-on pp128 pair is a CED loss:
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
maximum-row decoder backing. The initial reuse candidate preserves all 38
submissions, original arithmetic lineage, status initialization and the existing
admitted peak. This is a reuse
bet, not a native speedup claim; completion-wall variability remains separate.
Candidate `eec23a0f` passes bounded fresh/reused bitwise, allocation, arithmetic
lineage and cleanup fixtures on M4/M5, plus broader forward/FFN/CED regressions.
Full-model qualification now passes at 128/2K/8K (`eba1f591`) with all heads and
causal state identical to frozen Full/CED. Native gains remain unestablished.

Native v2 measurement (`999a2383`) now completes the full 2K cohort rather than
vetoing timing dispersion: retained Full 2.586 / 2.305 s, CED 2.672 / 12.479 s,
with pilot-plus-main variation above 5x for both. No usable win is established.
Seed/first-decoder fusion on the same branch passes bounded M4/M5 and full-model
128/2K/8K qualification at `e50dba57`, completing 21/21/27 commands without
changing work rows or scratch capacity. Pre-score32 native cohorts at the operator's
240 GiB MWSS retain Full 5.649 / 2.279 s versus CED 3.056 / 3.213 s at 2K,
and Full 18.966 / 10.178 s versus CED 7.713 / 7.498 s at 8K. The 8K pilot also
includes CED 19.840 s: a promising long-prompt signal, not stable performance or
a fusion gain. Preserve all samples and compare within the same configuration.
Rebase and score32 requalification now pass at `efea4908`, preserving the inherited
Full logical arity and all recorded pre-rebase heads/state. Next: native attribution
on that default; separate GPU work from host/completion cost without reopening
the explicit-residency lane or inventing cache-based admission restrictions.

Fresh outside leverage map: landed reconnaissance (2026-10-24;
`bench/2026-10-24-v41-outside-leverage-map.md`), not new performance evidence.
Two independent outside reviews plus primary-source/source audits prioritize:

- CED transactional cross-layer completion batches: in flight
  (`perf/v41-ced-batches`, `perf/v41-ced-batch4`). Two-layer source `9874c68b`
  passes 18 M5 GPU fixtures and the same five full-model sizes with exact
  control heads/state and unchanged work rows: 21 -> 11 commands at 2K,
  25 -> 12 at 8K. Its native cohort establishes no win: retained Full/CED
  medians 2.913/5.970 s at 2K and 11.924/16.145 s at 8K, with large dispersion.
  Preserve pending/completed separation, sticky evidence and bounded incumbent
  selection. Four-layer source `4672dc81` changes only the grouping default;
  the same 18 M5 fixtures and five full-model sizes now pass, completing
  6/6/6/7/7 commands. Native Full/CED medians are 7.624/12.858 s at 2K and
  16.474/11.427 s at 8K, still with large dispersion and no stable win.
- Same-format cooperative/register-fed quantized matmul: negative
  (`perf/v41-register-operands`; final
  `bench/2026-10-24-v41-register-operands-32x32-f16.md`). Correct bounded
  coordinate/fragment/producer and one-SIMD alternatives all lose. The final
  matched-F16ACT 32x32 one-accumulator path takes 10.676 ms versus 5.235 ms
  incumbent on the full streaming IQ3_XXS count32 bank. Do not promote or expand
  this losing leaf. Reproducible standalone probes preserve the newer API gate;
  the product tensor library still targets macOS 26.0.
- Expert-major/count-adaptive execution and head/query/layer-shared attention
  operands: open structural follow-ons. Measure useful versus padded work and
  actual data movement; do not recast the existing 5.4% return-grid or ~2% half-KV
  probes as unmeasured multipliers. Full chunk-4096 work landed separately in
  `28d23ac9` (+3.46% Q2 pp4096); its resident-size heuristic is not a universal
  fit boundary, and its packet still owes long-chunk quality coverage. CED
  runtime integration is separate, not a documentation-only rebase.
- Actual expert-population attribution: landed diagnostic evidence
  (`perf/v41-expert-populations`, source `89b06639`; executor remains a branch).
  Actual frozen-text 8K Full/CED preserves all heads/state and work, with counts
  copied before scratch reuse and no extra submissions. Independent reduction
  finds 34.13% CED routed-projection padding (32.16% encoder, 44.97% decoder).
  Tail-only 16-slot replacement models 24.29% less routed arithmetic, not latency
  or an implemented/numerically qualified kernel. This is one real prompt, not a
  workload distribution; retain packed-singleton lineage and separate gate/up
  wide64 from decoder/down narrow32. The bounded uniform-branch adaptive-tail
  leaf is negative (`perf/v41-adaptive-tails`, `e65eb300`): 40 M5 comparisons are
  bitwise exact, but 20/28 GPU and 19/28 wall fixture medians are slower, with
  all 336 samples retained. The 22.94% arithmetic model does not establish a
  latency gain. Default dispatch stays unchanged; broader structural occupancy
  remains open. See [the measured leaf and limits](bench/2026-10-25-v41-adaptive-tail-negative.md).
- Shared retained-window partitioning: landed qualification/native evidence
  (`perf/v41-window-p0-v30`, `perf/v41-window-p4-v30`). Disabled/4-GiB sources
  `b7038406`/`7398cb76` pass 74 CPU checks each, four actual M5 mapping/lifetime
  witnesses and 2K/8K Full/CED with all 257 complete heads/state matching the
  qualified batch-two reference. Actual retained objects change 44 -> 85,
  largest 44.594 -> 3.945 GiB, exposed span +656 KiB, private shadow zero.
  Instrumented and uninstrumented P0-P4-P4-P0 cohorts reproduce a 2K tail
  improvement: native retained P4 CED 3.469-4.564 s versus P0 5.528-30.209 s.
  A P0 pilot is still faster than P4; CED does not yet beat Full at 2K.
  Same-source 8K native confirmation repeats CED/Full separation in both P4
  processes: combined retained medians 7.645/11.094 s, CED latency 31.1% lower;
  all 12 CED pilot/main calls beat all 12 Full calls. P0 still has faster best
  calls but CED tails to 34.121 s versus P4 8.681 s. This is experimental
  evidence, not a main/default landing or universal win. Preserve oversized
  eligibility, EOF/lifetimes and actual admission; this is a soft coalescing
  target, not a fit cap. It also changes mmap provenance, so the comparison
  cannot isolate buffer length as the cause. No pinning/residency/purge is
  added. Main integration and other-artifact/context coverage remain open. See
  [the frozen sources, full cohorts and limits](bench/2026-10-25-v41-retained-windows.md).
- Batched Engram preparation, CED-origin prefix reuse and candidate-only
  long-context reindexing: open, with distinct cold/warm/context economics.
  Full prefix caching already exists; CED-origin reuse needs qualification.

Fresh native controls on frozen score32 source retain an 8K CED cohort median of
7.924 s versus Full 11.016 s, with every pilot/main CED call faster than every
Full call in that cohort. The 2K cohort establishes no win. Instrumented 2K
attribution localizes 0.812-1.571 s to summed driver scheduling intervals in warm
CED, with zero warm process reads/page-ins and ordinary/tensor PSO misses. This
is not shader-active time or a residency-causality claim. These controls precede
the 4K/wide integration and do not measure batching gains; see
`bench/2026-10-24-v41-execution-leverage.md`.

The subsequent current 4K/wide control cohort (`be874b3f`) does not reproduce
that clean 8K separation: Full/CED medians are 2.854/4.184 s at 2K and
14.723/14.489 s at 8K, with 2.8-6.6x pilot/main spreads. All endpoints still
match and all sampled host pressure is normal. Neither the new control nor
two-layer batching establishes a stable native gain; retain the stalls and
diagnose them rather than inferring speedup from command counts.

The immediate execution experiment pairs native driver/host attribution and
bounded CED retirement. The matched streaming cooperative-input control is now
closed negative. Do not stop at another peak-FLOPS comparison or profiler census.
The implementation phase also integrates the concurrent Full chunk-4096 change
on `perf/v41-ced-scratch`, including actual Full/CED arithmetic lineage and cold
memory pricing. The frozen qualified source remains the comparison; existing
proofs are not extended to the new geometry or batching before fresh checks.

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
  weight-byte/staging-bound, not evidence for a pure decoder-ALU rewrite.
  The control changes byte volume too; it does not isolate DRAM from on-chip
  staging. Same-format cooperative/register operand delivery is a distinct open
  dataflow bet in the fresh outside map, without a new quant artifact.
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
- Wider-N attention score: landed (2026-10-24 score32 entry,
  bench/2026-10-24-v41-score32.md). The 32-key score tile splits the
  latent into two 256-wide chunks, keeps Q resident in the same 32 KiB,
  and leaves softmax/PV/normalize unchanged. Fixture: score -18 to -20%,
  full sink -12 to -13% at 128/256/512 rows. Same-binary Q2 ABBA on the
  landing build: pp2048 855.219 -> 877.289 (+2.58%), pp4096 817.600 ->
  839.959 (+2.74%); the pre-landing ABCCBA measured +2.84/+2.83%. IQ3
  held-text witness identical (NLL 2.1542914542, top-1
  0.5677083333). Default from 128 rows on the tensor tier;
  `QWEN_V41_SINK_TENSOR32=0` restores the 16-key form. Score is still
  ~11.9 TFLOPS vs PV ~20.4; register-cooperative or materialized
  attention remains open.
- Full 4096-row chunk selector: landed (2026-10-24 chunk4096 entry,
  bench/2026-10-24-v41-chunk4096.md). All current residents (Q2 ~186 GiB,
  Maya-S ~166 GiB, IQ3 ~217 GiB) use one 4096-row chunk at the 240 GiB
  MWSS; only >240 GiB keeps 2048. Q2 pp4096 build-vs-build ABBA:
  842.588 -> 871.755 (+3.46%); pp2048 unchanged. IQ3's first 4096 pair
  had one transient (827.866); four later samples are 879.406/879.871/
  878.797/874.775 (mean 878.212, sigma 2.33) against 838.487/836.996
  control. The current 4096 + wide gate/up build measures 897.1/896.5
  (896.823) against 839.5/838.3 (838.906), +6.9%. IQ3 now uses 4096.
  The 576-token witness does not exercise the
  4096 path; per-row arithmetic is unchanged in construction.
- 64-slot gate/up panel at 4096: landed (2026-10-24 wide64 entry,
  bench/2026-10-24-v41-expert-wide64.md). Down stays on the 32-slot tile
  (wide down is +43% slower); gate/up wide is -22.7% in the fixture and
  +1.67% at pp4096 in the same-binary Q2 ABBA (875.506 -> 890.172).
  IQ3_XXS nonzero correctness is bitwise identical; the wide tile is
  selected only at >=4096 rows with the clamped epilogue, and
  `QWEN_V41_EXPERT_WIDE=0` restores narrow.
- Bounded attention/expert probes, not landed:
  return-only expert grid width 64 -> 2 is -5.4% of the combined expert
  tile (~1.3% model arithmetic) on the 31/32/33 route fixture; a prebuilt
  half-KV score source adds ~2% full sink on top of score32 and needs
  persistent-view lifecycle work. The 33rd routed slot costs +58.2% down
  time for +3.1% useful work, so a 64-slot panel at 2K is only a ~0.5-1%
  model tail opportunity; it is worth it at 4096 where the panel is full.
  Evidence branches `perf/v41-expert-sched`, `perf/v41-expert-tail`,
  `bench/2026-10-24-v41-expert-panel-tail.md`.
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
