# V4.1 Outside Leverage Map: Change Dataflow And Execution, Not Just Tiles

Date: 2026-10-24. Research checkpoint, not a measured optimization.
Reviewed maintained main `0fdbf9c0` and CED candidate `cada5864`, whose executable
source matches qualified `efea4908`. Two fresh, independent outside model reviews
and three focused source/primary-literature reviews informed this map. Reviews
were read-only; no new GPU experiment, model run or implementation is claimed.
Private review responses, retrieved sources and a source-bound arithmetic witness
are retained outside Git.

## Decision

Pursue two structural bets first:

1. **Retire bounded groups of CED decoder layers together.** Remove CPU
   authorization round trips while preserving ordered GPU operations and checked
   publication. This targets native latency and variability, not useful FLOPs.
2. **Feed quantized matmul through cooperative/register operands.** Keep the
   artifact and decoded half values, but remove the threadgroup operand round
   trip. This is a newly verified alternative to the failed FP4/decode-ALU bets.

The first attacks execution waste; the second attacks operand delivery across
much of the remaining computation. Attention, expert occupancy and Engram are
additional structural lanes, not substitutes for establishing their actual
critical-path shares. The aim is a substantial combined native multiplier, not
a collection of independently advertised microkernel percentages.

## The Roofline Gap Is Real, But Has Several Denominators

The measured 130 TFLOP/s reference holds half operands in threadgroup memory with
no device traffic in its inner loop. It is not a streaming quantized-GEMM or
sparse-attention roofline. Some landed wide dense projections already reach
67-90 TFLOP/s; the older 54-68 figures are not the whole current dense baseline.
Expert controls reach roughly 40-57. Attention score/PV reach about 11.9/20.4,
but include dependent gathers, conversion, different matrix shapes and barriers.

Three ledgers must accompany any future roofline claim:

- Useful model arithmetic versus executed padded tensor arithmetic.
- Requested weight/latent bytes versus actual DRAM, cache and staging traffic.
- Native critical-path wall time versus command GPU envelopes, driver scheduling,
  CPU preparation and dispatch-active intervals.

For hidden 5120, FFN 2304 and top-6 of 384 experts, routed gate/up/down costs
424.673 MFLOP per complete layer-output row. Shared FFN plus the main attention
projections add 324.010 MFLOP. This deliberately omits attention products,
indexing/router/mHC/Engram work, publication and seeds:

| Cold KV workload | Complete output rows | Projection + FFN useful TFLOP | Reference seconds at 130 TFLOP/s |
| --- | ---: | ---: | ---: |
| Full 2K | 81,920 | 61.33 | 0.472 |
| Exact CED 2K | 64,393 | 48.21 | 0.371 |
| Full 8K | 327,680 | 245.33 | 1.887 |
| Exact CED 8K | 187,990 | 140.74 | 1.083 |

These are optimistic reference budgets, not rigorous physical lower bounds or
attainable runtime predictions. They leave a large gap worth attacking. They do
not support reducing the retained 7.50-second CED call to 0.75 seconds by making
the same retained matrix work run at the demonstrated reference rate.

The native 8K cohort remains 7.50/7.71 s CED versus 10.18/18.97 s Full, with a
19.84-second excluded CED pilot. It predates score32. Keep the whole chronology;
neither the fast samples nor intrusive stage shares define today's native floor.

### Work Removal And Padding

Exact CED's complete-output row reduction is 1.272x at 2K and 1.743x at 8K.
At 8K, even deleting every remaining decoder output improves that proxy by only
1.147x; bounded128 improves it by 1.130x. The encoder now dominates that work
proxy. Bounded replay is a separate approximation, not the next large exact win.

Conversely, the MoE tensor tile always computes 32 slots for a nonempty panel.
The following uniform-Poisson illustration is **not measured routing**:

| Rows in one hypothetical expert dispatch | Mean assignments/expert | N32 padded/useful slot work |
| ---: | ---: | ---: |
| 128 | 2 | 13.83x |
| 2048 | 32 | 1.453x |
| 4096 | 64 | 1.233x |
| 8192 | 128 | 1.119x |

The 4096/8192 populations are hypothetical single batches. Current default
prefill chunks at 2048, so an 8K request does not obtain the 8192-row padding
ratio; CED additionally executes shrinking decoder batches.

This can make a useful-FLOP rate look terrible while hardware computes padding.
It does not imply a 13.83x latency win: the selected experts' weights still have
to arrive. Capture actual per-layer counts, active panels and bytes before
extrapolating a balanced synthetic tile to shrinking CED.

Decode is different again. The older 22.3 ms/token budget assigns about 8 ms to
dispatch and about 10-11 ms to bytes. Removing all of the former gives only
`22.3 / 14.3 = 1.56x`. Literal 11.1 GiB at decimal 1.09 TB/s is 10.93 ms, not
10.2; retain the original estimate as approximate rather than silently mixing
GB/GiB or TB/TiB. Prefill's matrix reference cannot establish a decode multiplier.

## Ranked Leverage Map

Targets below are hypotheses, not results. For an affected fraction `f` improved
by `s`, whole-scope speedup is `1 / (1-f+f/s)`. Overlapping bets cannot be
multiplied independently; historical Full stage shares are not current CED wall
attribution. For example, 2x over half the critical path gives 1.33x overall;
4x over a 23.2% attention share gives only 1.21x.

| Priority | Structural bet | Opportunity and first discriminator | Category |
| --- | --- | --- | --- |
| 1 | CED transactional completion batches | Remove per-layer host retirement; compare one/two/four-layer batches with unchanged operators and priced lifetimes | Exact execution target |
| 2 | Register-fed quantized matmul | Investigate 1.5-2.5x affected kernels by eliminating operand staging; compare matched streaming/cooperative controls, not just resident peak | Same artifact; exact operands, numerical witness if reductions differ |
| 3 | Occupancy-aware expert-major execution | Investigate 2-4x low-count expert phases; measure real padded slots and complete pack/GEMM/scatter cost | Exact scheduling target; changed tiles need requalification |
| 4 | Head/query/layer-shared attention operands | Investigate 2-4x affected attention phases through reuse and better partitioning, not another monolithic fusion | Exact layout arm, separate numerical arm |
| 5 | Batched Engram preparation | 393,216 small row reads at 8K; deduplicate/coalesce/overlap only where inclusive native preparation pays | Exact data-access target |
| 6 | Preserve exact prefix reuse through CED serving integration | Potential multi-fold repeated-request savings; existing Full cache benefits must not be counted again | New CED-origin reuse qualification, not cold-prefill acceleration |
| 7 | Candidate-only long-context reindexing | Score the trained candidate pool before sorting; no 8K benefit, substantial work reduction beyond 16K | Exact valid-input decisions; status/NaN semantics must also be proved |
| 8 | Decode dispatch islands and economical target packets | Remove real serial launch boundaries first; qualify packet verification economics before a drafter | Exact dispatch arm; separate generation/numerical path |

## Track 1: CED Without A CPU Round Trip Per Decoder Layer

Source seams: `forward/gpu_step/ced.rs::ced_command`,
`forward/attention_gpu/ced.rs` pending/completion proofs, and
`forward/ffn_gpu.rs::Rows::checked_prefix` in the CED branch.

The numerical controls already stay on the GPU. The CPU waits, checks status,
advances completed frontiers and authorizes the next unit. Seed fusion already
shows how an encoded dependency proof can permit downstream encoding without
pretending the producer completed. Extend that locally, not into a generic graph
scheduler.

The dangerous detail is status ownership: the checked scratch prefix currently
clears a compact FFN status slab on the CPU, with indexing dependent on current
row count. Merely deleting waits races in-flight work and can erase errors.
Use disjoint checked status ranges or an ordered sticky reduction before reuse;
preserve deterministic failure precedence. Keep completed/public frontiers
unchanged until the entire batch passes completion and all status checks.

Retain serial GPU ordering for residual/mix, local rings, selected controls and
shared half scratch. Reuse seed backing through that ordering rather than
retaining one seed allocation per queued layer. Price command memory and status
storage. Cancellation drains submitted work before release and poisons the
provisional state; failure never restarts Full on that mutated state.

Two-layer batches target 21 -> 11 commands at 2K. A single decoder batch could
target 21 -> 2 at 2K and 27 -> 5 at 8K while retaining encoder-chunk boundaries.
Those are count targets, not corresponding latency multipliers. Asynchronous
submission of unchanged commands removes waits but not submission count;
combining layers in one encoder removes both. Keep those experiments distinct.

### Decisive Experiment

First capture one unchanged score32 diagnostic cohort, then compare bounded
one/two/four-layer retirement with exactly the same operators:

- Native prefill scope, actual CPU preparation/Engram/allocation, encoding,
  commit entry/return, wait return, status checking and publication.
- Driver `kernelStartTime/kernelEndTime`, existing command GPU start/end, and
  available dispatch-active/limiter evidence from a short Metal trace.
- Both ordinary **and tensor** PSO creation, process reads/page-ins, allocation
  count/bytes, completed commands and endpoint/state equivalence.

Two diagnostic gaps are real: `tensor_pipeline()` lacks the ordinary pipeline's
miss accounting, and the outer case-profile residual includes allocation,
validation, fingerprints, reporting and teardown outside the native timer.
Zero recorded PSO misses does not establish zero tensor PSO creation; an outer
residual is not automatically CPU preparation.

Apple's `kernelStartTime` is **CPU driver scheduling**, not shader-active time.
A command GPU interval can include internal stalls. `Instant` has no exported
epoch here; do not subtract it from Metal timestamps. Correlate a documented
Mach-time clock or retain separate durations. Instrumented attribution locates
the opportunity; a subsequent uninstrumented native comparison establishes gain.

If gaps shrink with retirement count, finish the transactional executor. If
they persist inside driver/GPU execution, redirect effort to the identified
resource/dataflow cost. This is a useful falsifier, not a new variance or cache
admission veto. No explicit residency or forced cache manipulation follows.

## Track 2: Packed Weights Directly Into Cooperative Operands

Source seams: `kernels/tensor/moe_grouped_slots_tensor.metal` and
`crates/qwen-llm/src/metal/moe_grouped_generic/tensor.rs`.

Today each K-step decodes packed weights into half values, writes a threadgroup
panel, stages activations, synchronizes, then feeds TensorOps. Apple explicitly
describes custom dequantization into cooperative input tensors to avoid that
round trip. Pinned MLX NAX source constructs register left/right operands and
passes them to SIMD-scoped `matmul2d`: this is an actual implementation mechanism,
not a CUDA analogy.

Keep the same compressed weights, decoded half values, routing and epilogue.
Start with operand delivery, not fused gate/up or changed quantization. An expert
K32 step currently stages about 6 KiB of half operands for 131,072 FLOPs before
consumption. Removing those transfers/barriers is materially different from
making IQ decode ALU cheaper. Output staging can remain initially to isolate the
comparison. Register growth, spills, duplicate loads and lost cross-SIMD reuse
can still defeat it.

Apple's M5 Tech Talk dates cooperative matmul inputs to macOS 26.3; WWDC26
describes additional output-to-input reuse/reduction functionality. The shipped
tensor library targets 26.0. Verify the exact methods' compiler/deployment
requirements and provide a separately gated entry point. Do not silently raise
the whole library's minimum or equate existing Tensor availability with all new
APIs. This does not require migrating host execution to Metal 4 or residency sets.

### Decisive Experiment

Use the existing production down-projection geometry, first IQ3_XXS and then a
Maya-S low-bit type. Compare the incumbent, resident cooperative arithmetic and
streaming custom-dequant/cooperative inputs. Preserve K partitioning where the
API permits. Include actual count bands around 1/2/8/16/31/32/33 and a dense band.
Run a small parity fixture before a fully priced streaming bank; a tiny resident
fixture cannot establish production DRAM behavior.

Measure complete kernel time, useful versus padded operations, packed/staging
bytes, register/spill/occupancy evidence and decoded-half/output/status parity.
Charge any packing or predecode cost. If a reduction changes, classify and test
it as a numerical change instead of relaxing the frozen exact reference.

The Q8 control changes byte volume as well as decode complexity. It argues
against a naive decoder-ALU rewrite, but does not isolate DRAM, staging, padding
and synchronization. The fresh test changes the **operand path at the same
dtype**. Existing negative FP4, bigger-K and double-buffered-slab results remain
valid; they do not close this different premise.

## Additional Structural Lanes

### Expert Layout And Occupancy

Capture actual 384-expert counts at encoder chunks and every shrinking decoder
width; compute `sum(32*ceil(count/32)) / sum(count)`. Investigate active-panel
descriptors, expert-major activation layout retained through gate/up/down, and
count-appropriate N8/N16/N32 kernels. Preserve weighting before down and final
scatter/sum order; moving weighting changes the rounded operands.

The adjacent return-only grid probe already saved about 5.4% of the expert
operator, not a model multiplier. A new design must remove padding/repeated
operand gathers, not rename that probe. Larger chunks are already being explored
on `perf/v41-chunk4096`; do not duplicate that lane. Cross-chunk weight reuse
requires actual panel reuse or demonstrated cache locality, not just a loop swap.

### Attention Reuse And A Different Work Partition

Four 16-head groups independently gather latents for score and PV. Eight logical
reads are not eight DRAM reads. Test three distinct levels: derived half owner
views; selected-global materialization shared across heads/passes/follower layers;
and query-cohort/KV-first work ownership. Local SWA stays layer-specific.

At 8K the owner-20 half view is 8 MiB; all four owners total about 20 MiB.
A per-query gathered half slab instead costs 80 MiB for 128 queries or 320 MiB
for 512. A 32-query selected-ID union/inverse map is another bounded option, but
dispersed IDs can eliminate its reuse and add map traffic. Time gather, both
consumers and invalidation, not just a prebuilt slab's matmuls. Keep the existing
score/softmax/PV normalization arithmetic for the first layout experiment.

Register/SIMD-owned attention with cooperative row reductions is a separate
numerical arm. It changes the communication/occupancy premise of the failed
25-KiB online fusion; simply retrying online fusion or folding normalization
back into PV is not justified. Attention variants share one Amdahl budget.

### Engram Preparation

The source performs `2 modules * 24 rows * 8192 tokens = 393,216` row reads,
then CPU dequantization and F32 staging. The expanded payload is 384 MiB. This
Maya-S artifact's Engram inventory is Q6_K: requested packed rows total about
78.75 MiB before deduplication, not the Q8 example's 102 MiB and not actual
filesystem/DRAM traffic. The 167.6-GiB cold process-read count is not Engram
attribution. An older warmed 2K diagnostic puts Engram near 78 ms, so call count
alone does not establish a dominant warmed bottleneck.

Measure unique rows/tuples, page locality and inclusive critical-path time.
Deduplicate/coalesce bounded requests, decode into reusable staging, and overlap
future chunk preparation only with distinct safe storage. Existing retained CPU
shard ranges are worth comparing with descriptor reads. No whole-table Metal
mapping, prewiring, cache purge or forced eviction is needed. Four equal CPU/GPU
chunk stages yield at most 1.6x from overlap alone; a single 2K chunk has no such
inter-chunk overlap opportunity.

### Reuse And Long Context

FullForward already has live and snapshot prefix reuse. The actual missing
contract is CED-origin reuse: `serve/backend_v41/reuse.rs::reuse_eligible` rejects
it, and HTTP execution remains Full. Qualify restore/growth/short suffixes and
continuation before changing that guard. At 8K the recorded F32 causal payload
is about 60.02 MiB, not the paper's packed-cache size. A 1024-token incremental
Full suffix has about 4.59x fewer complete output rows than cold 8K CED, before
restore/fixed costs; that is a warm-request opportunity, not a cold benchmark
gain or an unimplemented generic-cache feature.

For contexts beyond 16K, decoder reindexers still score the full visible domain
before applying the trained 2048-block candidate mask. Candidate-only scoring
can bound those queries at 16,384 keys while preserving absolute IDs and decision
semantics; prove invalid-input/error and tie/NaN behavior as well. This saves
nothing at 8K. Across all scalar indexers, the asymptotic work reduction is about
2.6x, not the per-reindexer's apparent 64x at 1M. DoSA-style conservative
screening is a further distinct lane with full-domain recovery, not sublinear
retrieval or established Metal performance.

### Decode And Speculation

Decode already submits one whole-stack command per token. ICB replay can reduce
CPU encoding, but does not remove its constituent dispatches. Audit independent
projection/router/shared-expert islands and real launch fusion; present shared
half buffers introduce hazards even when equations are independent. No unsafe
whole-model persistent kernel or cross-threadgroup spin-barrier is proposed.

Beyond the same-byte serial floor, measure short target verification and accepted
prefix retention before building a DSpark executor. The model's native draft
features, actual artifact support and on-policy acceptance need proof. Under
uniform independent routing, six tokens touch about 34.6 distinct experts rather
than 36: routed weights amortize far less than dense weights. Use complete cycle
cost `Tscalar * E[emitted] / (Tdraft + Tverify + Trecovery)`, not oracle acceptance
or drafter speed alone. This is separate from cold CED and exact scalar RNG traces.

## Ambitious Programme, With The Multipliers Named

The immediate programme is **execution-path removal plus operand-path removal**.
Promote occupancy, attention or Engram if their measured critical-path shares
justify a different order. Do not wait for an all-encompassing profiler or build
a generic runtime before these small experiments.

A substantial 2x over current cold CED requires removing or accelerating at least
half its native critical path. A 4x result requires three quarters; it is a stretch
across multiple components, not something bounded128 or attention alone supplies.
It does not inherently require changing the model: better operand delivery and
removing large host gaps could also supply it, if measured. The current evidence
does not establish that combination. Gains quoted over Full must not count the
already-implemented CED reduction again when discussing gains over current CED.

Tenfold acceleration needs a named additional mechanism: eliminating a dominant
pathology, reusing substantial prior computation, amortizing several accepted
tokens, or changing useful work/policy. It cannot be promised from the ratio of
a resident matrix probe to a gathered sparse kernel. The existing exact CED
implementation remains the reference, not an obstacle to ambitious experiments.

## Primary Sources And Transfer Limits

- [Apple M5/A19 ML Tech Talk](https://developer.apple.com/videos/play/tech-talks/111432/):
  cooperative inputs from 26.3, cache/occupancy-aware tensor work and tracing;
  its dense examples are not our quantized MoE measurements.
- [Apple WWDC26 TensorOps](https://developer.apple.com/videos/play/wwdc2026/330/):
  custom dequantization into cooperative inputs, row reductions and compatible
  output-to-input reuse. Verify each API's deployment requirement.
- [Apple MPP Programming Guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf):
  direct device operands, cache locality, walk order, occupancy and static extents.
- [Pinned MLX NAX implementation](https://github.com/ml-explore/mlx/blob/56c2bf9236d929097f61e2c6e55be4e4bb6da249/mlx/backend/metal/kernels/steel/gemm/nax.h#L387):
  actual cooperative left/right register operands and SIMD-scoped matmul; not
  a measured GGUF expert speedup. Retrieved directly through GitHub API.
- [Apple compute scaling](https://developer.apple.com/videos/play/wwdc2022/10159/):
  CPU waits versus serialized GPU work; these are different optimization targets.
- [Apple driver scheduling timestamps](https://developer.apple.com/documentation/metal/mtlcommandbuffer/kernelstarttime):
  `kernelStartTime` is not GPU shader-active time.
- [Apple compute ICB semantics](https://developer.apple.com/documentation/metal/mtlindirectcomputecommand):
  replay and explicit barriers, not automatic collapse of GPU launch cost.
- [V4.1 report](https://arxiv.org/html/2609.19969v1): shared-global ownership,
  decoder dependencies, persistent reuse and explicitly approximate bounded replay.
- [Engram paper](https://arxiv.org/html/2601.07372v2): lookahead/offload dataflow;
  asynchronous prior art does not characterize our serial row reader.
- [MegaBlocks](https://arxiv.org/abs/2211.15841): active-block representation;
  transfer the work-list idea, not its GPU block sizes or reported speedups.
- [FlashAttention-2](https://arxiv.org/abs/2307.08691) and
  [FSA](https://arxiv.org/html/2508.18224v3): partitioning, communication and
  KV-first reuse; NVIDIA-specific asynchronous mechanisms are not Metal APIs.
- [DoSA](https://arxiv.org/html/2610.13990): certified score screening with
  recovery; published long-context regime does not establish an 8K gain here.
- [DSpark](https://arxiv.org/abs/2607.05147): native draft/verification economics;
  published acceptance and throughput do not qualify this artifact or backend.

Whole-model explicit residency/pinning remains closed. No smaller-model
workaround, cache-based GPU veto, default percentage reserve or hidden context/
quality reduction is part of this map.
