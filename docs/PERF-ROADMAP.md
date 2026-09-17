# Performance Roadmap

Living cross-session performance plan for qwen-llm. Keep this file concise and
current: update the ranking when new measurements change expected value, risk,
or dependencies. Treat `docs/PLAN.md` as the architecture/history plan; this
file is the active optimization queue.

Working rule: optimize from causal performance hypotheses, not from measurement
novelty. Every measurement task in this file should exist only to kill or
confirm a concrete engine hypothesis.

For append-only checkpoint history and exact current handoff state, see
`docs/PERF-LOG.md`.

## Current North Star

Make one fresh process and one loaded model answer one fresh prompt as quickly,
efficiently, lightly, and accurately as possible on the local M4 Max. Repeated
process-cold CLI invocation is the primary deployment shape today. The inference
contract remains serial batch size one, not aggregate serving throughput.

The three co-primary latency objectives are:

1. Process-cold first byte: process launch through first token delivery with an
   explicit filesystem-cache contract.
2. Model-ready fresh TTFT: prompt arrival through first token delivery.
3. Warm serial inter-token latency: one live sequence with no request batching.

Track peak session memory and quality beside all three objectives. Report
process-cold first byte separately. For loaded-model balance, retain
`sqrt(TTFT speedup * decode speedup)`, but never hide either phase in that
scalar. Also report total request wall for named prompt/output lengths.

Named request archetypes carry decision authority:

- **Interactive short**: up to 512 prompt tokens and 128 output tokens.
- **Agentic long-prompt**: 8K-32K prompt tokens and 256 output tokens.
- **Generation-heavy**: up to 2K prompt tokens and 1024 output tokens.

Use canonical real fixtures inside those shapes. Report phase gains, balanced
gain, and total wall for every applicable archetype.

Scope rules:

- Internal prompt-token matmul width is part of BS=1 prefill and remains in scope.
- Speculative future-token verification is intra-request width and remains in
  scope. Greedy equivalence is not distribution exactness; stochastic exactness
  requires correct target/drafter rejection sampling.
- Multi-request batching, independent-stream concurrency, shared-prefix
  multi-query execution, and asynchronous serving form a secondary roadmap lane.
  They remain important, but do not rank against fresh serial BS=1 work today.
- Prefix caching is relevant only when a prefix is reused or reconstructed. It is
  not a fresh-prompt optimization.
- Process-cold model loading and ordinary first-use placement are a first-class
  product lane. Keep their boundary separate from model-ready TTFT, but let broad
  exact cold wins outrank narrower loaded-model work under the current deployment
  mix. This does not include explicit whole-model wiring.
- Whole-model `MTLResidencySet` optimization is closed, not merely waiting for a
  faster implementation. A hard-killed 104.2 GB FRESH process stranded
  essentially the complete set as reboot-only wired memory and destabilized the
  machine. Existing opt-ins are historical/diagnostic only. Do not run, extend,
  or transfer them under current supervision; reopening requires an explicit user
  decision, deterministic unload, a sufficient supervisor teardown window, and
  observed host-memory recovery.
- Metal initialization is process-exclusive across qwen binaries. Direct
  contention fails with owner metadata, queue-managed work may set
  `QWEN_METAL_LEASE_WAIT=1`, and a host that remains at least half wired after a
  15-second stabilization window is treated as poisoned. This heuristic does
  not coordinate non-qwen Metal frameworks or prove Metal teardown completion;
  `QWEN_METAL_LEASE_SKIP_WIRED_GATE=1` is the explicit telemetry-gate override,
  not a process-exclusion bypass.
- SIGINT and SIGTERM request cooperative qwen/qwen-bench cancellation at safe
  token, chunk, request, and benchmark boundaries. This only enables destructor
  teardown; it cannot survive a supervisor's SIGKILL before the boundary is
  reached. OpenCode's 200 ms escalation must be lengthened or disabled before
  explicit whole-model residency is safe under its process supervision.

Exactness labels:

- **Bitwise**: identical represented values and terminal model state.
- **Numerical**: the same model with an explicit floating-point tolerance.
- **Greedy semantic**: the target-authoritative greedy token stream is unchanged.
- **Distributional**: stochastic outputs follow the exact target distribution.
- **Approximate**: input, model, or target arithmetic changes and needs quality
  validation.

Primary sentinels:

- Lightweight dense: 0.8B or 4B, plus one low-bit stress format.
- Dense regression anchor: `Qwen3.6-27B-Q4_K_M.gguf`.
- Dense current capability candidates: pinned `Qwen3.8-27B-Q4_K_M.gguf` from
  `unsloth/Qwen3.8-27B-GGUF` as the higher-bit anchor, plus pinned
  `Qwen3.8-27B-Ridge-3.7bpw.gguf` from
  `empero-ai/Qwen3.8-27B-Ridge-GGUF` as the smaller interactive Pareto
  candidate. Both surfaces are text-only; compare measured capability rather
  than assuming version or bitrate equivalence.
- MoE responsiveness anchor: `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`.
- MoE heavy anchor: `Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf`.
- True-long anchor: A3B Q4 at 32K and 131K.
- Treat A17B as a separate capability/residency objective until a local model,
  memory contract, and measured phase profile exist.

Decision rules:

- Eligibility follows architecture, admitted capacity and kernel invariants, not
  the last benchmark's endpoint. A tested length is evidence, not an automatic cliff.
- Separate correctness from performance experiments. Use independent primitive
  oracles and local identical-state replays for unchanged math mechanisms; do not
  repeat a complete known-loser model path merely to show that it remains slow.
- Reuse an optimized traversal for horizon checks, live-state oracles and current
  phase attribution. Spend full-request timing on decisions it can actually change.
- Noisy controls limit performance claims; they do not veto independent correctness
  evidence. Keep prior attempts/verdicts, but do not turn their experiment gates
  into universal requirements for subsequent work.
- Prefer removing prompt tokens, target evaluations, arithmetic, or bytes over
  improving the same work unit. Prefer a new work unit over another local retune.
- Same-work kernel changes must name the measured phase and whole-phase ceiling.
- Work taking more than one week needs at least 5% expected whole-phase gain and
  a credible 10% ceiling in a primary cell. Lossy work needs at least 15%
  expected gain or a major memory benefit to pay for quality validation.
- Cheap exact work under two days may proceed with a credible 2-3% whole-phase
  gain, but it does not displace a larger strategic branch.
- Use one highest-ceiling sentinel and one dissimilar guardrail before widening.
- Estimate the minimum detectable effect for each packet. Reject a gate below its
  protocol tier's MDE, and reject any packet whose zero-cost ceiling is below its
  promotion gate.
- Never run performance benchmarks in parallel.
- Treat llama.cpp parity as a floor, not the endpoint. Carry qwen/lcpp and
  qwen/roofline comparisons where each is meaningful.
- Current M4 Max anchors are `474 GB/s` stream, `~12.5-13.3 nominal TFLOP/s`
  production Q4_K mat-mat across measured shapes, and `3.03 TFLOP/s` scalar FMA
  sanity. v0.644's `15.70-15.89` synthetic E controls are not a silicon ceiling.
- Use the repo-pinned llama.cpp lock at `scripts/bench/llama-cpp.lock.json`.
- Treat battery or thermal/performance warnings as benchmark confounds.
- Do not treat process-wide major faults as a generic cache-warm invalidity
  predicate. Require a causal I/O contract such as block-input operations,
  explicit physical-read/residency evidence, or arm-symmetric warm-up.
- Do not infer simultaneous multi-shard residency from an ordered integrity
  scan. Hashing can perturb the file-cache condition it intends to certify;
  measure phase-local I/O symmetrically or separate identity from conditioning.
- Keep dense 27B in analysis while optimizing MoE.
- Use limiter captures for kernel-shape claims and untraced runs for throughput.
- Treat `prefill_chunk=1024` as a safe cap, not a universal long-prompt optimum.
- Require fast-path-clean validation for promotion-grade family rows.
- Keep process-global environment variants in separate processes.
- Until an enforceable GPU lease exists, only the coordinating session may run
  timed GPU work; research subagents must remain read-only.
- Stop at the cheapest evidence level that can decide the current question.
  Arithmetic, model-free differentials, isolated phases, force-path pilots, and
  default-policy changes carry different authority and need not share one packet.
- Put reusable execution, correctness, timing, and JSON instruments in code;
  keep experiment-specific gates, authority, and interpretation in concise
  preregistration and result prose. Do not compile a bespoke disposition engine
  for each candidate.
- Promotion-grade timing should use a clean committed source identity. Exploratory
  dirty-tree probes are allowed when clearly non-authoritative. Reference cached
  model/fixture identities rather than rescanning weights or self-hashing source,
  binaries, logs, and result files inside candidate code.
- Sole-attempt acquisition is reserved for genuinely expensive, noisy, unsafe,
  or policy-changing packets. Otherwise retain every attempt and disclose rerun
  reasons instead of forbidding learning from a repaired observer.
- Track raw acquisition under `target/profiles`; commit packet directories only
  for decision-changing results. A focused regression and ordinary endpoint
  check are preferable to replaying unrelated historical gates.

## Muse Fresh / Decode Priority - 2026-09-09

The active Muse priority is **fresh prefill and decode**, not further cache work.
Generation now selects Q8/unified M4 Max matrix/tiled-online and split math by
default; existing run/serve variables become independent0 rollbacks (unset/1 allows
qualified execution). Unsupported lanes and explicitly named lens/fit/reference
benchmarks retain original math. These remain numerical, not bitwise or sampled-exact
paths. Default-delivery status: `docs/bench/2026-09-17-muse-defaults/RESULT.md`.
Benchmark-derived upper cliffs are removed:
model context131072, actual capacity and kernel invariants govern admission. The
1024-visible-position split minimum remains performance policy. Independent131K
attention checks and a512-transition32K live horizon pass, without another full
slow-prefix reference. This is not full-model131K numerical equivalence.

1. Decode FFN removable-work census, especially for warm/long-output flows:
   measured32K FFN64.96% (44.200ms), attention12.43%. Capture raw gate/up/product
   before in-place overwrites, exact zeros, gate-only threshold versus product
   energy, and block-aligned occupancy. Small gates do not bound products; knowing
   a product after computing up is not avoided work. Price selection, gather and
   weight-layout costs before a kernel proposal. Logical22.029GB/44.2ms equals
   498GB/s payload rate, NOT measured DRAM or a roofline. No sparsity claim yet.
2. Fresh-prefill structural work remains co-primary. Refreshed late32K packed
   GPU1113.485ms attribution is attention49.91%, FFN38.43%;8K is FFN57.0%.
   These are stage shares, not controlled full-request or cold-load authority.
   Retain earlier954.503ms attribution and ordinary1110.958ms observer outlier
   separately from controlled913.536ms calls.
   Larger batches are a credible fallback: same512 FFN rows4x128/2x256/1x512 are
   bitwise identical, but N512gate4.464%/down10.045%savings miss frozen5%bothshape
   budget. GeometryliftHOLD, not a slow-kernel kill or proof128optimal. Weighted
   ~2.9%modelchunk projection assumes gate/up and block0/alllayer transfer. Down
   asymmetry may guide scheduling, but down-only N512 needs cross-chunk activations.
3. Do not keep polishing PV without a new mechanism. F32 MMA PV saves~10.0/14.4%
   full/sliding, half-P PV~10.85/15.54%; both miss frozen15%full+sliding budget.
   Numerical screens pass, but neither earns model replay or production precision
   changes. Four-row/8SG reuse is another ownership tradeoff, not an automatic win:
   more shared/register residency and a32KiB output spill need explicit handling.
4. Product reachability no longer requires serving math opt-ins; both paths are
   qualified defaults, with existing optimized warm/reset/generated-history and actual
   dispatch proof. Native1158/17 HTTP first-request observations33.903 ->7.166s,
   optimized SSE retry1.096s, firstmodel(reasoning)delta277ms. These are NOT balanced
   speedup or OS-cold measurements. ATEM/framing, warm/reset outputs, busy503 and
   detected-abort cold recovery pass. Do not generalize sampled equivalence or
   final-user-text TTFT; backend publication still precedes final HTTP framing.
5. Preserve qualified authority and tight feedback loops: controlled32K tiled
   chunk17.953%wall saved, controlleddecode14.722forwards/s/73.134%saved. CLI6229
   193.592prefilltok/s and fresh32K231.586s are diagnostics, not new ABBA evidence.
   Use saved32640/8064 states, not another long prefix construction or slow reference.
   Keep prior per-row FAIL, online KILL and split primitive HOLD intact; separate
   numerical diagnosis and whole-forward PASS retain their own scope.

Evidence: `docs/bench/2026-09-09-muse-serve-math/RESULT.md`,
`docs/bench/2026-09-08-muse-long-context/RESULT.md`,
`docs/bench/2026-09-08-muse-math/RESULT.md` and
`docs/bench/2026-09-08-muse-live-prefix/RESULT.md`. Native ATEM, sampling and reasoning
contracts stay unchanged; temperature1/top-k64/top-p0.95 is sampled, not greedy.

## Qwen Restored-Request Leverage Map - 2026-09-07

The ranking changes materially for restored requests. This is a next-experiment
map, not a claim that warm serving is the largest opportunity in every lane.

| Priority | Lane / work to remove | Current authority | Next decision |
| --- | --- | --- | --- |
| 1 | Lossless incremental CPU snapshot publication | Existing-root publication + full restore saves 39.026/52.268% at 8840/32752 plus16; both frozen cells pass. 32K shares 2.146GB KV, all recurrent state recopied, logical accounting unchanged. Test-only ownership/byte proofs pass. | Real Sequence restore -> owned append -> publication anchor; prevent escaped raw buffer aliases from rearming reuse, keep canonical durable export explicit. Prove cache lifetime/accounting and complete request before promotion. |
| 2 | Live continuation / retained state ownership | Owned runtime tracks model provenance, capacity and consumed position; backend still allocates request-local sequences. | Measure exact-history eligibility; preserve poison, capacity, cancellation, cache clear/eviction and branching/regeneration behavior. Position alone is not token-history identity. |
| 3 | GPU-owned snapshot publication + restore | Model-free32K cycle244.042 ->85.585ms,64.930% saved;8840 control fails, overall two-cell HOLD. All-byte/immutability proofs pass. | No GPU cache rollout or threshold retrofit. Any follow-up must price both directions, driver-sized storage, durable materialization and complete request cost. |
| Policy hold | Transient prompt-capture elision | A4GiB serve cache cannot retain both32K prompt/completed snapshots, but CPU witnesses prove abort-retry and larger-cache counterexamples. | Lossless prefix sharing takes precedence. No blanket skip or inferred traffic rate; cache keys/checkpoints/retry semantics stay unchanged. |
| Hold | Tiled VT / N16 attention | Strong copy-bank/full-verifier results respectively, but no endpoint promotion authority. N8 persistent-online endpoint remains closed. | Do not rescue noisy tiled controls or manufacture N16 DFlash2 reachability. |

Cold-load remains an independent co-primary track: native Q8 embedding removes
3.735GB exactly but its latency gate held. Keep named load-stage attribution ahead
of generic PSO or whole-model residency work. Fresh long prefill and steady decode
also remain open at the structural level; closed local tile/fusion candidates do
not establish a global roofline. Reopen those cells with a new arithmetic/byte/
quality mechanism on an actually selected model-quant-architecture path.

Do not spend effort on fictitious zeroing: `zeros_f32`/`zeros_f16` already allocate
uninitialized buffers, as do CPU snapshot arenas before capture. Keep weight
quantization separate from KV dtype; the snapshot experiments use F16 KV even
when the target model's weights are Q8. Evidence and scope:
`docs/bench/2026-09-07-snapshot-lifecycle/RESULT.md` and
`docs/bench/2026-09-07-snapshot-segments/RESULT.md`. Incremental publication does
not remove any restore bytes, reduce first-root cost or qualify fresh-prompt/decode
latency. Its unique payload ledger excludes allocator/metadata overhead; anchors
outliving cache eviction require explicit admission/lifetime accounting.

Source audit: even safe `Sequence::metal_session(&self)` exposes retainable writable
KV buffers. A caller can retain an alias before restore, then write through a safe
blit afterward. Actual 0.8B F32 runtime witness `6621673a` passes both escape orders:
current full recapture sees the four-byte mutation, original checkpoint unchanged.
Invalidating only `metal_session_mut`, or clearing at accessor
time then rearming on restore, is insufficient. Resolve raw-escape lifetime or
permanent reuse taint before adding production anchors; the CPU prototype does
not exercise these low-level GPU aliases. This is not a bug in current full-copy
snapshots and does not itself implement taint tracking.

The next ownership prerequisites have concrete coverage: packed hidden capture
rejects session/scratch aliases before encoding, and ordinary serial serving now
uses owned prompt-only/decode APIs. Real F32 and Q4 tests preserve logits and all
persistent bytes, including fresh/exact-hit backend emission and completed pending
checkpoints. Packed/capture/spec paths remain raw fallbacks. No new latency or
anchor eligibility is claimed; wire a real restore-issued anchor only after raw
session escape cannot rearm reuse and shared storage is accounted across eviction.

## Attention Checkpoint - 2026-09-07

The adjacent exact tiled-VT copy experiment improves16-bank GPU time85.26/81.62%
at8840/32752, but its admitted Q8 restored16 phase is HOLD: controls spread85.309/
25.673%, far above5%. Warmup payload hashes precede only first measured A, an
asymmetric conditioning boundary. Full-state/logit bitwise witnesses pass and
scratch remains35.095/103.547 MB; no flag, promotion or rescue timing packet.
N8 online matrix remains closed: the older persistent-VT endpoint already lost
with transpose cost amortized away. Evidence:
`docs/bench/2026-09-07-tiled-vt-rebuild/RESULT.md`.

The larger source-backed boundary is snapshot publication/restore:32K restore
alone is113.654-142.007 ms in the retained rows, copying CPU arenas into Metal.
Prefer a costed ownership/copy-elision falsifier over further transpose tuning.
A GPU RAM snapshot must charge publication+restore and replace, not silently
mirror, CPU storage; a retained live continuation must enforce exact consumed
token identity, capacity, poison and cancellation rules. Both need full cache/
admission/lifetime accounting and endpoint evidence. No broad no-copy Vec wrapper,
whole-model residency, immutable-prefix aliasing or inference from noisy means.

The charged one-layer online-matrix attention screen passes: 6.250479 ->3.607279
GPU ms at32768, N16, G6 24/4/256, including a full-prefix one-layer VT rebuild.
Saving42.288%, control spread0.409%, incremental actual workspace93,847,552 B;
all row numerical checks and the32769 edge pass. This is test-only, not a kernel
retune, a selector promotion or full-verifier evidence. It differs from closed
one-pass flash and N2-8 three-pass direct-V paths.

The full-verifier follow-up also passes on an actual32752-token Q8 prefix:
257.819688 ->216.347709 ms full verification saves16.086%, partial8 saves16.865%,
and forced restore+8serial replay saves5.696%. All16 argmaxes and numerical state
gates pass; prefixKV and replay-final state are bitwise. Existing verifier/layer
scratch is2,695,266,304 B plus93,847,552 B online workspace. Setup-only shape failure,
first warmup outliers and global host compression remain disclosed. Evidence:
`docs/bench/2026-09-07-full-verifier-online-n16/RESULT.md`.

Product HOLD is the immediate boundary: default cutoff16384 excludes32K, and the
calibrated installed DFlash2 artifact has physical N8, not N16. First establish
actual artifact/block-size reachability and durable fully charged acceptance
economics. Do not manufacture N16 DFlash2 selection, broaden the controller or
infer an N8 win from these rows. This does not reopen generic skinny-GEMM staging
or weaken single-chunk-VT rejection. Keep numerical/bitwise, teacher-forced/
generated, and greedy/sampled authority separate. No production flag is added.

## Qwen3.8 Flash-Next Optimization Lane — 2026-08-26

### September 16 Current Defaults: Guarded Routing And HC On, Split QSA Retired

Qualified improvements ship enabled, not behind indefinite opt-ins. Candidates
must have a bounded qualification/disposition; remove production surface when the
timebox ends without qualification. Keep failed evidence in history/research.

Current Flash-Next CLI/library defaults on exact Apple M4 Max, capability checked:
guarded N512/K10 routing and singleton Q8 HC up-plus-mix. Independent strict0
rollback controls remain;1 permits qualified execution, never unsupported forcing.
Production split QSA and its CLI/API/scratch/route are removed after the original
dissimilar failure and one failed composed repair closure. Do not revive old flags
or broaden frozen tolerances. See `docs/bench/2026-09-16-flash-defaults/RESULT.md`.

Final default144-forward closure passes bitwise top-k/default identity and numerical
HC on natural32+dissimilar8,121terminalstates and route/scratch gates. Final HC
increment over guarded+incumbent QSA saves6.5551% GPU/5.7323% executor wall, both
mean/pair floors and controls pass. Actual unflagged known-answer CLI passes.
Earlier HC HOLD and QSA speed evidence remain historical, not current defaults.

The settled-default parent ledger is now complete and qualifies:
`docs/bench/2026-09-16-flash-defaults/PARENT-RESULT.md`. Inclusive QSA-containing
blocks25.71ms exceed GDN-containing20.14ms; bootstrap3.80ms/tail1.07ms. Strong
front-loading affects both families, so use the common late cohort rather than
multiply first-QSA6.52ms by12. These are complete blocks, not leaf attention costs.

The bounded child line is now closed: `docs/bench/2026-09-16-flash-defaults/CHILD-RESULT.md`.
Native dispatch sampling is unsupported; source/host-only five-forward packet is
bitwise/census exact but GPU observer-7.024% exceeds5%, so timing is INCONCLUSIVE
with no retry/segmentation/replay. Eleven QSA3..43 paths match structurally; layer47
is a distinct Q8-down cohort. No child GPU bottleneck ranking is established.

1. **N1024 policy coverage is now delivered.** The existing strict E8P32 packed
   router defaults on at512/527/1024/2048 only. Native1024 prefill GPU14.7703% and
   wall14.6757% savings pass frozen gates, with bitwise endpoint/every continuation
   state and real CLI default/rollback delivery. No new shader, memory or switch.
   Evidence: `docs/bench/2026-09-16-flash-defaults/ROUTER-1024-RESULT.md`.
2. **Rechart coverage by source and real planner reachability, not a width sweep.**
   N1024 proves one useful policy gap, not an arbitrary interval. Other widths stay
   generic unless independently qualified;3072 plans2048+3+1021, not2048+1024.
   The earlier CPU validation0.1719ms remains below its1ms floor; weights bind once.
   Donor CUDA/C16 changes and parallel uncached BF16 PLE preads do not establish
   local BS=1 or IQ4 mmap gains. No automatic port or storage-policy reopening.
3. **Preserve closed lanes and context.** No retired attention, stage-observer
   retry, tiny-copy fusion, validation bypass, cold-storage/range warming or
   speculative expert retile without a genuinely new source of leverage. Harness
   checkpoint/readback/disk idle is not production continuous-decode startup.
   These local negative results do not establish global optimization exhaustion.

All GPU work requires production lease, real wired-memory gate and API validation.
The server remains stopped; preserve user V4.1 work and do not push remotely.

### September 16 Earlier Delivery (Superseded): Guarded Routing PASS

Follow-up complete: `docs/bench/2026-09-15-qwen4exp-hc-up/MOE-GUARDED-BUDGET.md`.
Three-dtype capture/replay is bitwise with correct guarded routing. Rare layer2
timing is INCONCLUSIVE; common layers4/5 have stable controls but distributed
router/gate-up/down costs. **Next is one small whole-forward parent ledger on
this baseline**, then inspect the largest parent's source before another kernel.
Use complete GDN-containing/QSA-containing blocks, bootstrap and tail, not false
recurrence-only/attention-only labels. The delivery evidence and prior queue follow.

Default-off `QWEN4EXP_GUARDED_TOPK=1` is now actual-product/CLI qualified. N512/K10
singleton only, zero extra GPU memory, packed routing unchanged.53-case nonfinite
compatibility packet passes after a preserved failed variant. Empty/QSA-on/off/HC
composition rows and terminal121 states are bitwise. Fixed four-forward product
ABBA saves41.7428% GPU/37.0533% executor wall with QSA on/HC off; controls0.0369%/
0.0550%. Actual CLI known answer/status/EOS pass. No default promotion, additive
QSA percentage, prefill or request-speedup claim.

1. **Use guarded routing as the next measurement baseline**, QSA on and HC off.
   The serial-router budget is obsolete; do not keep optimizing its old bottleneck.
2. **Observe remaining complete-MoE costs before another expert body.** Reuse saved
   inputs and interval-aware accounting, then add IQ3/IQ4 dtype coverage where it
   can select between routed gate/up, down/sum and shared experts. Compare with
   the remaining native parent budget rather than extrapolating a leaf win.
3. **Keep HC/attention/RMS sweeps parked.** HC's own promotion HOLD is unchanged.
   No new quant download or broad fusion/geometry expansion prerequisite.

Evidence and coverage limits: `docs/bench/2026-09-15-qwen4exp-hc-up/TOPK-PRODUCT.md`.
Server remains stopped; production lease/real wired gate/API validation remain
mandatory for GPU work. Prior compatibility and timing failures remain recorded.

### September 15 Routing Rechart: Native Bitwise Gain, Guarded Delivery Next

The parent-budget investigation found a source-level one-thread N512/K10 selector.
Interval V2 absolute timing remains INCONCLUSIVE; a separate finite parallel-selector
screen and native packet now establish the mechanism. With QSA on/HC off,32 full
logits/hyper rows, terminal121 states and1536 router rows are bitwise. Fixed native
four-forward ABBA saves40.1901% GPU/36.1689% executor wall, controls0.1606%/0.3224%,
all frozen mean/pair floors pass. No prefill/request throughput or additive-QSA claim.

1. **Deliver a guarded singleton selector next.** Existing research parallel path
   is finite-only; preserve incumbent NaN/infinity behavior with bit-pattern
   classification and uniform serial fallback. Qualify compatibility, capability,
   custody and actual product route; added guard requires its own qualification.
2. **Do not gate this behind more expert-dtype profiling.** Native proof covers
   common routing across all48 layers. Layers4/5 complete-cost replay matters only
   when selecting a subsequent expert-body mechanism after this route saving.
3. **Keep HC/attention/RMS sweeps parked.** HC performance HOLD is independent and
   unchanged. No broad router fusion, geometry expansion or new quant prerequisite.

Evidence: `docs/bench/2026-09-15-qwen4exp-hc-up/TOPK-RESULT.md`. Production defaults
and existing parallel host n<=256 remain unchanged; this is native research PASS,
not product delivery. Every GPU experiment retains production lease/real wired gate.

### September 15 Product Decision: HC HOLD, Complete MoE Next

Default-off experimental HC option is numerically qualified independently with
QSA on/off, empty scalar prefix and actual CLI. No new GPU memory, packed math
unchanged. Actual-product GPU mean5.1092% but first pair4.814% misses frozen5%:
performance promotion HOLD; no speed recommendation or timing retry. CLI's
M128 IQ4_NL527 boundary is isolated/repaired with original-output bitwise proof.

1. **Complete-MoE accounting first**, QSA on/HC off. Native capture is bitwise
   through full logits/hyper/121states, but initial stage packet fails global
   timestamp ordering. Separate saved-layer2 diagnostic proves valid individual
   spans can appear in reversed encoded order across independent routed/shared
   branches. They are reordered disjoint spans, not demonstrated overlap.
2. **Use saved layer2 for a versioned interval-aware observation**, no new prefix
   prerequisite. Exact dispatch census, positive individual spans, sorted interval
   union/envelope, inclusive stage durations; no exclusive-cost inference from
   summed spans. Layers4/5 replay/three-dtype budget still pending. Select the next
   expert mechanism only after that parent budget, not a leaf-timing sweep.
3. **Keep HC/attention/RMS tuning parked.** Preserve product performance HOLD and
   failed instrumentation evidence. No special quants needed for this work.

Evidence: `docs/bench/2026-09-15-qwen4exp-hc-up/PRODUCT.md` and `MOE-RESULT.md` in
the same directory. Server remains stopped; production lease and real wired gate
remain mandatory for every GPU experiment.

### September 15 Completion: Native HC Useful, Delivery Next

The remaining packed validation blockers are repaired with isolated boundary and
populated bitwise evidence, not speculative broad edits: `82a21f9d` IQ4_NL down,
`09f54356` artifact-reachable IQ4_XS gate/up and Q8 down. Unreachable variants stay
unchanged. The unchanged native HC packet now passes under production GPU custody
and API validation:32 full-logit/hyper/121-state checks, then incremental GPU
5.2751% and executor-wall5.0132% savings with split-QSA enabled in both arms.
Controls0.5032%/0.4615%; frozen mean AND pair floors pass. This is not request
throughput, default-configuration performance, or broad quality authority.

1. **Deliver narrow default-off HC opt-in**, initially qualified together with
   split-QSA. Keep the same Q8/four-branch/hidden2560/rank320 shader, incumbent
   fallback and existing scratch. Exercise actual product bindings with the
   existing32-token numerical packet and four-forward bracket. HC independently
   enabled without split needs a separate bounded composition check.
2. **Measure complete-MoE costs as the next research step**, covering routing,
   expert work and combination on representative native decode state before
   choosing another topology. Do not make this a prerequisite for HC delivery.
3. **Park further HC/attention/RMS body sweeps.** Context-growing index scoring
   remains separate;2179-token results do not price far-context selection.

Evidence: `docs/bench/2026-09-15-qwen4exp-hc-up/NATIVE-RESULT.md`.
Earlier validation failures and component protocol1 failure remain recorded;
the HOLD map below is historical, not the current first action.

### September 15 Rechart: HC Useful, Native Held At Validation Boundary

The source-led rank320 HC up-plus-mix screen is useful: protocol2 complete-HC
GPU saves38.2602%, controls2.0899%, frozen mean/pair floors pass. This is a
component result, not native attribution. Protocol1 failed an incumbent hostile
raw-dot criterion; preserve that failure and the explicitly versioned
conditioning correction. All mixed-output and RMS gates stayed unchanged.
HC remains research-only; no new production option or default change.

Native qualification is HOLD before candidate execution: the unchanged packed
prefix hits a Metal API-validation boundary. `95a71437` fixes the independently
isolated grouped IQ3_XXS builtin-width case at512 experts, with historical
narrow/wide bitwise agreement and retained product regressions. Full packed
validation still aborts, and the remaining offending kernel is not identified.
Do not interpret either aborted native run as HC quality or performance evidence.

1. **Localize remaining packed-prefix validation boundary**, then qualify only
   the smallest isolated repair. Keep API validation on; no speculative broad
   builtin edits or repeated full-model attempts before an isolated packet.
2. **Resume the unchanged native HC packet**, one existing2179 prefix/32-token
   continuation, strict full-logit/state gates and frozen four-forward ABBA.
   Split-QSA stays enabled in both arms to measure incremental HC leverage.
3. **Observe complete-MoE costs**, then choose an expert scheduling mechanism.
   The prior GDN-containing block aggregate includes HC/FFN, not recurrence cost.
4. **Keep further HC, attention-body and RMS tuning parked.** Long-context
   index scoring remains a separate context-growing lane, unpriced by2179.

Implementation/reproducer `60e644ae`; partial validation repair `95a71437`.
Current evidence: `docs/bench/2026-09-15-qwen4exp-hc-up/RESULT.md`.
The September14 map below is retained as history, not the current first action.

### September 14 Reopening: Portable Donor Mechanisms

**Lease correction and exclusive completion.** The first real CLI found an
existing Qwen server holding the production lease. Test contexts had used
per-process test locks, so all September14 Flash-Next timing/profile authority
below is provisional; numeric/state checks remain observed passes. The new
benchmark guard acquires the real production lease and wired-memory check
before Metal setup and correctly refused the occupied lease. The user then
approved SIGINT to the verified server; it exited and was not restarted. The
guarded actual-product32-token packet and real CLI known-answer check now pass.
New exclusive GPU/executor-wall savings are16.0366%/15.7158%; older unguarded
timings and coarse profiles remain provisional, not retroactively promoted.
See `docs/bench/2026-09-14-qwen4exp-donor-split/PRODUCT.md`.

Flash-Next is the active optimization priority, ahead of further Muse microkernel
work. Inspect mechanisms before demanding a matched competitor benchmark or
different quant downloads. DwarfStar `9139e2a` has real portable M4 decode paths;
its special quant readers and M5 NAX are not prerequisites to transferring them.
The community M4 report (~366-379 prefill, ~39-40 ordinary /50-52 MTP decode on
older fork builds) shows a reported decode-throughput gap motivating investigation,
not an isolated engine speedup or proof of a threefold M4 prefill deficit.

1. **Delivered split opt-in; preserve the qualification boundary.**
   `0557d812` passes strict full-logit/hyper/121-state gates over four native
   teacher-forced forwards from one current packed2179 prefix. Warm shared-state
   ABBA saves16.566%GPU/16.143%executorwall; controls agree within0.04%. This is
   not request throughput, broad quality authority, or production-exclusive
   performance evidence. Default-off session-scoped product bindings and one
   priced shared scratch are implemented;24/2/256 F16 and2048-2051IDs only.
   Actual-product32-token numeric/state/census checks pass under unchanged gates.
   New production-lease-protected ABBA saves16.0366%GPU/15.7158%executorwall;
   real CLI2578/23 known-answer request succeeds. Use explicit
   `QWEN4EXP_QSA_SPLIT_DECODE=1`; default remains off. No default-promotion claim,
   new weights, second reference prefix, or repeated coarse profiles.
2. **Price remaining HC/FFN projection bodies.** Diagnostic native profiles put
   complete QSA blocks at15.304ms aftersplit and34GDN-containing blocks at31.054ms;
   both include HC/FFN, so do not call the latter recurrence cost. HC up K320 has
   ten Q8 blocks against the generic GEMV's32 initial block slots: two SIMDgroups
   do no weight-loop work. Investigate a rank-specialized up-plus-gated-mean body,
   and obtain a bounded complete-MoE observation before another IQ3 topology.
   Generic large-Q8 GEMV already matches donor topology; keep launch-only fusion,
   small bucket work and GDN-middle retuning parked.
3. **Stop polishing the attention body; RMS broadcast demoted.** The exact-order
   once-per-head candidate `30f65ca5` passes144 component comparisons but shows
   no timing gain; unstable controls mean INCONCLUSIVE, not a quantified
   regression. Source is preserved in history and removed from the active tree;
   no sweep/model replay. Indexer work still grows withcontext/4 beyond the
   selected-ID plateau: price it at a naturally available long checkpoint,
   not from this2179-token packet. Other projection donors outrank another RMS
   implementation unless a new concrete execution mechanism changes the case.

Native outcomes and updated leverage map:
`docs/bench/2026-09-14-qwen4exp-donor-split/NATIVE.md`.

Primitive evidence and donor source map:
`docs/bench/2026-09-14-qwen4exp-donor-split/RESULT.md`. `09553fb9` separately repairs
the incumbent's36-byte dynamic scratch API violation by rounding to48 bytes,
without changing arithmetic. Earlier lane closures below retain their original
scope; they do not rule out the newly identified split-softmax algorithm.

The first MoE checkpoint deliberately uses the scalar, stable 512-way top-10
selector. After end-to-end decode exists, attribute routing separately and test
a cooperative selector only if the scalar scan is material. Any replacement
must preserve lower-expert tie ordering and selected-logit softmax exactly; do
not optimize routing from isolated kernel novelty alone.

Layer-zero bring-up initializes the four HC streams with four validated copy
dispatches. A direct repeat kernel is a profile-gated launch reduction after the
complete token path exists, not a reason to delay the correctness checkpoint.

PLE bring-up keeps the 28.8 GB IQ4_NL embedding table CPU-addressed and uploads
only the 16 selected rows (1,440 bytes) for each token. Profile the host gather
and upload after end-to-end decode exists; batch or overlap staging only if it is
material, and do not make the full random-access table GPU-resident by default.

The first transactional two-layer prefix copies layer-zero output into one HC
bridge and writes PLE output back to that same buffer before layer 1. Preserve
the owning-command boundaries until full decode is correct; direct child output
handoff is a profile-gated launch/bandwidth cleanup, not a bring-up dependency.

The first complete 3-GDN/1-QSA cycle makes QSA runtime memory explicit instead
of treating weight admission as sufficient. One 262K-capacity released QSA
workspace is about 528.6 MiB; all 12 QSA layers are about 6.19 GiB before
allocator overhead. Sum these estimates into session admission before allocating
the 48-layer runtime, and keep short-capacity correctness fixtures independent
from the native-context product policy.

The complete 48-layer text session now provides that admitted baseline in one
command through final HC and Q6_K logits. Its exact logical session inventory is
`143,207,764 + 25,356 * capacity` bytes, or 6.324 GiB at 262K, before the
page-rounded per-buffer upper bound and 512 MiB dynamic reserve. Profile this
baseline before changing command topology. In particular, the parent currently
pre-stages PLE rows for transaction safety and passes the prepared history into
layers zero-one without a second gather. Profile the remaining single host
gather and 1,440-byte upload before introducing batching or overlap.

The first warm layer profile measured `54.415` ms GPU at position 27: 34
post-PLE GDN blocks consumed `35.677` ms, 12 QSA blocks `15.452` ms, layers
zero-one `2.180` ms, final HC plus logits `1.084` ms, and encoder boundaries only
`0.022` ms. The existing four-row IQ3_XXS routed SwiGLU kernel is now default for
this family after reducing the command to `49.513` ms and a 23-transition decode
from `18.65` to `20.64` token/s with identical generated bytes. Roll back with
`QWEN4EXP_MOE_IQ3_FAST=0`. The subsequent representative GDN/QSA split confirms
that the remaining lane is projection/mixer work, not command-boundary cleanup.

Architecture reconciliation against the report, HF model code, and released
weight map closes the Kimi-residual concern: Attention Residual was ablated, the
shipped Gated Residual is implemented, and optional `mtp.*` speculative weights
are absent from the pinned text GGUF by construction. The local HELLO sequence is
an internal determinism proof only. Upstream BF16 full-logit rows remain useful
future calibration, but the roughly 360 GB checkpoint is not a near-term
promotion dependency. Immediate end-to-end authority is quant-native: held-out
teacher-forced NLL first, known-answer long-context safety second, observed-token
top-1 and greedy sentinels third, implementation-diverse same-quant llama.cpp
triangulation fourth, and component oracles plus selector margins for
localization. Scalar continuity is a diagnostic reference, not numerical truth.
Fast IQ3 also has direct baseline/candidate full-logit rows at three local
boundaries; all argmax IDs match with at most `4.921e-5` relative RMS and
`7.573e-4` maximum delta.

The exact beta/alpha/decay tri-fusion at checkpoint `56bc662` removed two
dispatches from each of 36 GDN layers but saved only `0.0419 ms` at the median;
its bootstrap upper bound was `0.3090 ms`, below the predeclared `0.4 ms` kill
gate. The experiment is removed. Do not infer a large decode win from launch
count alone when the fusion leaves all material weight and activation traffic
intact.

Selected-range composition is structurally complete. The former local
scalar-distance promotion gate failed; default promotion remains `HOLD` pending
semantic evidence rather than treating singleton arithmetic as truth. Released
selected commands with total N=2-4 pass local full-logit and continuation gates
after an exact Q8 output residue route. On the earlier N=4,099 promotion workload,
one packed 2,048-row selected suffix is `6.547e-2` from default-safe execution at
the endpoint and `1.489e-1` after one continuation. Exact Q8 output worsens it;
64 commands of 32 rows reproduce the endpoint, ruling out full-chunk width and
band seams for that workload. Ordinary dense N=2,048 packed composition is
already `8.563e-2` from singleton execution, so the scalar envelope is not a
selected-specific discriminator at long N. Merged llama.cpp matches all tested
argmax IDs but its same-quant full rows are also materially separated from both
local routes.

A distinct pinned natural-roadmap N=4,099 trace reports
`8.361495e-2/1.021529e-1` endpoint/teacher-forced-continuation relative RMS
against default-safe. Capture-off/on logits and all enumerated persistent-state
tensors are bit-identical in every arm. The row-2,051 layer input is bit-exact,
and the first sampled-stage difference is the composite layer-0 attention HC
output (`8.331e-5` relative RMS), before QSA receives a different activation.
Generic packed selection first changes one cutoff pair at position 2,056/layer
31 with a `4.344e-4` default-safe margin. This is ordinary E1 packed arithmetic
propagation, not evidence of a block-boundary defect.

The local implementation resurvey, bounded screens, natural N=512 profile, and
closed cold-storage packet reshape the queue.
Rapid/OMLX-style blocked packed recurrence is closed after an exact candidate
regressed `5.967417 -> 6.117709 ms` at N=2,048. The exact two-dispatch
MTPLX-style singleton middle is also closed: its 36-layer release leaf moved
`1.118938 -> 0.962312 ms/token`, saving `0.156625 ms` against a `0.75 ms` gate.
In contrast, selected-expert IQ4_NL row reuse is promoted after reducing clean
decode command GPU `46.716405 -> 45.402361 ms/transition`. Natural N=512 now
measures the generic F32 router at `3.960750 ms/layer`, or 14.81% of command GPU;
strict E8P32 is now promoted there after reducing the leaf to
`0.690375 ms/layer` and warm command GPU `1149.058875 -> 990.813875 ms`.
GPU-equivalent N=512 throughput rises `445.58 -> 516.75 tok/s`. The same
bit-exact strict kernel is now qualified at exact N=527: an isolated B-C-C-B on
the selected `2048+3+527` plan moves aggregate prefill GPU
`5,212.211417 -> 5,065.299813 ms`, saving 2.82%. The exact admitted set is now
`{512, 527, 2048}` and every other width remains generic. The force-ranked lane
is now:

Internal placement first reduced cold first-pass wall
`43.869198 -> 23.224459 s`. The subsequent internal demand/prefetch/demand
packet charges the whole 89,986,353,824-byte read, including CPU PLE, and reduces
`22.731643 -> 14.656168 s` while warm GPU remains unchanged. Retain only the
explicit default-off three-shard option. Default-on policy, range warming, and
further storage source work are closed.

The block-local IQ3_XXS source screen is now closed. Its compiled candidate is
bit-exact over 3,072 hostile/released half values, but natural N=512 moves the
representative leaf only `5.425000 -> 5.306417 ms/layer` (2.19%) and command GPU
about 0.45%. B1 alone exceeds the fixed `4.857600 ms` leaf ceiling, so the
worst-case gate cannot recover and the candidate is removed. Compact active
panels, indirect IQ3, donor ports, and further decode reshaping remain closed.

Natural N=512 clears the retile gate with `S16=21,294`, `S32=15,377`, and
`R=0.692398`. The promoted M128xN16xK32 kernel remains bit-exact and moves the
representative IQ4_NL down leaf `3.512855 -> 2.800188 ms/layer`; warm command GPU
moves `992.155458 -> 967.134625 ms`, saving 2.52%. It is default only for exact
N in `{512, 527}` on released geometry and Apple M4 Max, with
`QWEN4EXP_MOE_IQ4_DOWN_M128_N16=0` as rollback. The selected N=527 suffix
measures `R=0.653701` across its 43 IQ4_NL layers; a conservative model-free
43-dispatch bracket moves `152.858188 -> 127.086938 ms`, saving 16.86%, while
the `R=0.75` boundary still saves 12.27%. Existing natural N=2,048 counts give
`R=0.857718`, and every individual layer fails the same `0.75` entry gate; do
not widen the scope there.

The retile-private paired IQ4_NL decoder is closed. Optimized AIR proved the
incumbent still traversed each packed payload twice, and the candidate reduced
that to one IR-visible traversal. Natural N=512 B1 moved the
representative leaf `2.800708 -> 2.686791 ms/layer`, only 4.07%, and exceeded
the fixed `2.520169 ms` futility ceiling. Warm/profiled command savings were
0.77%/0.71%, also below gate. The helper was removed without running B2/A2.

Packed bridge ownership is closed before implementation. A temporary
profile-only split measured each representative 5 MiB copy at
`0.018250-0.018625 ms`. Weighting two copies across 34 GDN and 12 QSA layers and
debiting positive split-topology inflation projects only
`1.693810/1.697572 ms` in two accepted captures, 17.5% of the preregistered
`9.671346 ms` gate. The split is removed; do not build destination-aware motors
or bridge aliases without a new source of leverage.

The preregistered GDN/QSA named-stage screen is invalid and makes no performance
decision. Both split arms completed with accepted whole-command observers and
identical generated output, but their 156-sample timestamp streams were not
globally monotonic. No child timing was emitted or persisted. The temporary
profiler is removed. Do not call this a mechanism KILL or GO, and do not rerun
adaptively. Reopen only under a new preregistered observer that preserves raw
timestamp availability and first-inversion evidence.

Selected packed QSA now defaults two GQA4 kernels inside its existing
default-off envelope. The gathered-QK leaf removes per-slot barriers and moves
`24.485458 -> 6.931125 ms`; four-head softmax/value reuse moves
`32.844667 -> 10.429312 ms`, with bytewise incumbent equality. On the frozen
2,578-token known-answer prompt, release B-C-C-B moves aggregate prefill GPU
`6,303.917229 -> 5,397.047021 ms` (14.39%) while every arm emits the expected
JSON and EOS. The independent rollback flags are
`QWEN4EXP_QSA_GQA4_LOGITS=0` and `QWEN4EXP_QSA_GQA4_VALUE=0`. This is a
performance promotion inside selected execution, not semantic authority to
enable selected packed QSA globally.

1. Selected packed QSA is default-on as of 2026-09-03: the frozen
   `2026-08-29-qwen4exp-selected-semantic` packet returned `SEMANTIC_GO`
   (pooled held-out NLL delta `0.0055` against the `ln(1.01)` gate; both
   known-answer predicates pass). `QWEN4EXP_PACKED_SELECTED_QSA=0` is the
   rollback. Do not rebuild a broad quality harness or wait on BF16 authority.
2. Revisit native K=1 MTP only with a converted or side-loaded artifact carrying
   all 31 omitted speculative tensors, full admission metadata, and atomic
   QSA/GDN/PLE state. Do not pursue K=2/K=3.

The simple IQ4_NL down-plus-sum follow-up is parked despite the down-kernel win:
it removes only about 8.8 MB/token of materialized output traffic and 43
dispatches while retaining all selected weight traffic, and the analogous
IQ4_XS fusion regressed. Reopen only with new evidence above the existing
`0.5 ms/token` gate. QSA gather, dense masking, launch-only fusion, HC private
repacking, complete singleton GDN-middle fusion, and additional blocked
recurrence tiles remain closed.

Packed-prefill S1 is closed at released `16/48/128` GDN geometry. For one and
two token rows, the existing packed convolution/SiLU prep, paired L2 norm,
batched decay chain, and packed DeltaNet recurrence match serial execution
bit-for-bit across every output and final convolution/recurrent state. The
optional parallel prep kernel remains outside this proof and must stay disabled
for the Flash-Next lane until separately qualified. This checkpoint led into
the now-complete packed PLE, HC, MoE, and QSA motors.

The private packed QSA motor is closed at its component and transaction
boundaries across dense and selected ranges. Dense N=1/2/8/16/33/64, the
three-token shoulder through sequence length 2,051, and all modulo-four residues
retain scalar state. Selected index and gathered-attention packets preserve
lower-ID ties, cache order, failure status, released BF16-weight/F32-activation
projection routes, and scalar arithmetic at B=1/32. Mixed one-band and reusable
`32+1`/`32+32` motors match chronological scalar outputs and persistent state.
Ordered audits prevent stale, failed, duplicate, or missing rows from publishing
committed length. This does not claim 48-layer full-logit equivalence after a
2,048-row packed horizon; the N=4,099 falsifier above explicitly rejects that
stronger interpretation.

The complete packed session now composes selected chunks through all 48 layers
and final logits while retaining the scalar GDN, PLE, and QSA state owners.
Because all 12 QSA layers reuse one scratch allocation inside the same command,
every selected band now starts with a GPU-ordered control reset; host band-zero
sentinels alone were insufficient. Ordinary and stage-sampled two-QSA gates lock
the reset-to-packet-to-audit order. Dense-only plans retain 55 allocations;
selected-capable plans add nine 32-query-band buffers for a 64-allocation,
1,917,948,672-logical-byte maximum sidecar.

The synchronous runner preplans consecutive commands against the reusable
2,048-row cap before causal mutation. Selected-capable opt-in execution packs
every eligible multirow range; dense-only execution stops at width 2,051 and
scalarizes the suffix, and a final one-token residue remains scalar. It splits
exactly at width 2,051 so the first selected command is never accidentally
hidden inside the dense shoulder. Every command must publish its planned
endpoint, and timing aggregates packed tokens rather than assuming one packed
command.

Released N=2,053/2,054/2,055 runs execute `2,048+3+(2/3/4)`, publish all 12 QSA
owners, lock reset/packet/audit order, and pass complete-vocabulary endpoint plus
scalar-continuation gates. Their exact Q8 output residue path uses one
token-axis GEMV dispatch per QSA layer and is scoped to total command N=2-4.
N=4,099 proves the same scheduler and 64-band topology but failed the former
local-distance promotion gate: packed-selected is `6.547e-2` from default-safe
at the endpoint on that workload. Retain its `393.5 tok/s` versus `32.7 tok/s`
result as experimental leverage, not a qualified performance row or a semantic
quality failure.

`QWEN4EXP_PACKED_SELECTED_QSA` defaults on; `=0` packs repeated dense chunks
and scalarizes selected rows as before. The preregistered quant-native NLL and
known-answer packet passed on 2026-09-03 (`SEMANTIC_GO`, see PERF-LOG). An
upstream BF16 anchor may strengthen that decision later but is not a release
prerequisite.
Released 18-token HELLO runs retain output and scalar decode handoff.

Accepted N=18 and N=2,048 first/warm/profile packets close the broad attribution
step. Warm outside-GPU time is only `2.026/6.397 ms`; child publication is
`0.008/0.016 ms`. The cold first command instead carries `4,207/1,622 ms` of
additional wall time with almost unchanged GPU intervals, making mmap/Metal
first-touch a separate cold-TTFT lane. At N=2,048, bootstrap is 4.47%, post-PLE
blocks 95.50%, and tail 0.03%.

The follow-up six-stage packet passed bitwise replay and every observer at both
N=18 and N=2,048. At full chunk, representative GDN/QSA blocks are
`76.327/81.331 ms`; common MoE work is `39.426/39.472 ms`, while the mixers are
`28.088/33.054 ms`. Extrapolation attributes 48.3% of the command to MoE, 25.4%
to GDN, 10.6% to QSA, 7.2% to HC, and 3.6% to bridge/combine. At N=18, raw
representative shares put MoE at 37.0% and bridge/combine at 28.9%, with 1.44%
whole-command over-assignment from small-N layer variance. The latter bucket
still conflates tiny copies with HC injection. The next packet therefore splits
MoE internals; bridge output, HC, bootstrap, tail, and dispatch-only rewrites
are parked.

The five-stage MoE packet at `0c7ec44` also passes bitwise motor scratch and
released first/warm/profile gates. Crediting only the 43 standard IQ3_XXS plus
IQ4_NL layers, N=18 assigns 14.56% to gate/up, 11.08% to down, and only 4.40%
to routing. At N=2,048, routing grows to 20.28%, gate/up remains 15.55%, and
down falls to 6.78%. Ordered reduction, shared-tail, and boundary work are all
below 5% at both shapes. The scaling is consistent with full chunks amortizing
expert weights across roughly 40 slots per expert while router/top-k/bucket work
still covers every token; this packet does not directly measure those traffic
or occupancy mechanisms. Split the three routing kernels before choosing
projection, selector, or fused-publication work.

The seven-stage routing packet at `e86f9b1` identifies the mechanism. At
N=2,048, F32 router projection is `15.863542 ms/layer`, or 18.167% of command
GPU across 43 standard layers. Top-k plus bucket total `1.898291 ms/layer`,
below the `4.367 ms/layer` component KILL floor, so their fusion is closed. At
N=18, the complete routing parent is only 4.215%; projection remains parked
until the shared exact-order candidate supplies a measured ceiling. The
surviving full-chunk candidate is strict E8xP32 F32 projection, which reuses
activation loads across eight output rows while preserving independent scalar
K order. Reassociated float4/matrix paths and router dtype changes remain
disqualified by discrete route sensitivity.

The strict E8xP32 packet at `b799a95`/`9aca035` closes routing by shape, and
`6df62fb` promotes the surviving full-chunk arm. A-B-A and pinned release
replay keep all route state, endpoint logits, persistent handoff state, and
non-router dispatch topology exact. At N=2,048, router time falls
`15.891167 -> 2.608167 ms/layer`; warm command GPU falls
`3764.453062 -> 3128.968083 ms`, with 99.67% of the 48-layer leaf prediction
reaching the command. Exact N=2,048 is default on Apple M4 Max with
`QWEN4EXP_PACKED_ROUTER_E8P32_STRICT=0` rollback. N=18 saves only
`0.005791 ms/layer` and regresses warm command GPU, so it remains generic.
Do not reopen selector/bucket work; the active-panel gate/up screen below is
closed. Selected-range packed QSA is structurally implemented; its quant-native
semantic battery, not more topology work, is now the correctness priority.

The clean route census at `69e51c9` now separates prompt shape from kernel
geometry. N=18 puts 80.10% of credited route mass in count 1-8 and reaches only
16.57% active-panel lane occupancy at width 16. Repeated-token N=2,048 instead
puts 97.62% of route mass in count 65+ and is retained only as a concentration
stress control. The natural technical-text N=2,048 sample activates 15,560 of
22,016 expert instances; count 65+ owns 77.49% of routes, and its 63,924 active
width-16 panels are 86.10% occupied. The current direct grid still launches
2,818,048 panels before the ten-output-panel multiplier, so 97.73% return before
matrix work.

The counter-free IQ3 gate/up screen is closed as
`INVALID_SCREEN / NO_CANDIDATE`. Three of four Full-control brackets exceeded
the frozen 2% drift limit, so validation stopped before report creation and no
KILL or triage result exists. Full/interpolated-control agreement, positive
Full-minus-NoWork, nonnegative band increments, and additivity all passed. For
rerun-value assessment only, every unscored optimistic leaf estimate was
5.34-5.97% against the conjunctive 10% screen; command estimates were
0.830-0.928% against 1%. Do not retrofit cooldown or warm-state selection,
rerun this condition, or implement the compact N16 active-panel descriptor plus
indirect-dispatch falsifier. Reopen gate/up only for a materially different
mechanism or an independently justified prospective measurement-policy change.

Runtime admission now separates exact-release qualification from behavioral
contracts. The full tokenizer fingerprint and released stop vector remain
oracles; compatible Flash-Next finetunes use structurally valid `qwen35`
tokenizer data, the supported prompt protocol for reasoning controls, and their
producer-declared in-range stop set. PLE's boundary token remains independent
model state rather than generation policy.

## Serve Follow-ups — 2026-08-20

2026-09-07 fresh-serving Q8 now has a retained explicit opt-in:
`QWEN_SERVE_FRESH_PACKED=1`, greedy fresh cache misses of 19-48 tokens, no drafter.
Balanced endpoint gates pass: fresh19 wall saves 66.646%, code32/128 saves 16.940%,
prose48/128 saves 23.115%; guards pass and all 64 Q4/Q8 paired responses agree. Q4 fails its
512-token guard (+8.482%) and remains unsupported. Automatic Q8 rollout is held
after a post-narrowing first-request outlier coinciding with large global VM
compression; neither an intrinsic first-use bug nor a PSO cause is established.
Keep the measured opt-in, preserve the PASS, and do not repeat qualification or
rewrite gates. Future attribution requires phase-local host/PSO observations under
a declared host contract, not generic kernel/PSO/residency work. Evidence:
`docs/bench/2026-09-07-fresh-serving-http/RESULT.md`.

2026-09-06 cost re-ranking finds fresh 19/3 prefill still occupies 85-88% of Q4/Q8
request wall, while landed Q8 compact-128 prefill is only 3.0-3.25%. A test-only
fresh packed screen at 19/32/48 passes all six width/model cells: allocation+
prefill saves 71-89%, actual scratch 11.7-29.1 MB, numerical KV/GDN agreement,
greedy 3/EOS,64,64 per model. The next bounded serving gate is balanced fresh
HTTP evidence with no alias machinery and unchanged fallback. No production
change yet; do not extrapolate fixed-order phase savings. CLI already uses packed
fresh prefill, so this is not a process-cold CLI optimization or a global ranking
without deployment frequencies. Evidence:
`docs/bench/2026-09-06-fresh-short-packed/RESULT.md` and
`docs/bench/2026-09-06-global-cost-ledger/RESULT.md`.

2026-09-06 single-chunk VT removes the restored-tail workspace blocker: actual
query-32/key-8840 scratch is 51.7 MB instead of 323.3 MB, bitwise versus full-VT on
Q4/Q8 including poisoned reuse. Only Qwen3.8/Q8 greedy no-drafter restored
suffix 7-32 is enabled: balanced blue-2 wall saves 70.179%; compact-128 code/prose
save 14.746%/13.978%, with all Q8 unchanged wall guards passing. Q4's reverse
code pair misses 10%; its serving selection stays off. There is no distributional
claim for sampled followups inheriting numerical checkpoints, no cold-load gain,
and no global threshold change. Evidence:
`docs/bench/2026-09-06-single-chunk-vt/RESULT.md`.

The Q4 153-row replay is now localized through a fresh engine checkpoint:
completed consumed 8987/pending newline 198 loses only its pending token when
rendered as history; logical lookup falls back to 8860. Test-only consumed-26
serial is slower than old 153-row packed (allocation+restore+prefill 1106.831 vs
952.054 ms). Consumed-26 single-VT is 329.942 ms with 46.1 MB scratch, matching 128 greedy tokens
and numerical persistent state. The follow-on atomic consumed-alias + packed
HTTP packet is now INCONCLUSIVE: wall saves an observed 12.409% (paired
15.623%/9.077%), but primary TTFT control spread 6.919% misses the frozen 5% gate.
All 40 responses agree with the intended cached-token difference; fresh guards
pass. Candidate and unused API are removed, with no same-cell repeat or gate
change. Do not enable alias-only serial or reopen the broad Q4 selector. Return
to global fresh/cold/spec phase accounting before selecting more warm work.
Evidence: `docs/bench/2026-09-06-consumed-tail-http/RESULT.md` and
`docs/bench/2026-09-06-consumed-boundary-reuse/RESULT.md`.

2026-09-06 restored suffix32 screen exposes a packed-workspace blocker:
1262.563 ms serial versus 252.341 ms construction+packed execution, but
323,256,320 bytes fails the 128 MiB gate before numerical/greedy validation.
All-layer transposed V accounts for 289,669,120 logical bytes (89.61% of total).
This prioritizes auditing single-block VT lifetimes before changing the 48-row
serve threshold; one-slot arithmetic estimates 51,691,520 bytes but is not a
priced implementation. Preserve multi-chunk VT reuse and all existing default
policies. Evidence: `docs/bench/2026-09-06-restored-tail-crossover/RESULT.md`.

2026-09-06 phase-dead scratch work keeps only ordinary serial CLI release:
the 8,840-input/64-output Qwen3.6 Q4 packet removes 1.364 GB of Metal allocation
at first delivery, with +1.194% request wall and unchanged sampled peak. This
is a resource result, not a cold/TTFT/admission win. DFlash request wall +3.091%
misses the resource guard; served release has only a correctness pair, and
prompt lookup's historical architecture gate rejects the available witnesses.
Those lifetimes remain unchanged. Independent review supports the narrow scope
and prioritizes localizing warm small-tail TTFT over widening these held paths.
Evidence: `docs/bench/2026-09-06-cli-prefill-lifetime/RESULT.md`.

2026-09-06 replay-debit follow-up is INCONCLUSIVE, prototype removed. The tested
base's template admission rejects the installed Qwen3.8 dense artifact's Flash-Next
digest; the alternative Qwen3.6/legacy-drafter pilot restores serially and never
exercises fallback accounting. Its small timing differences carry no causal
authority. Require admitted artifacts and active-path witnesses before another
controller packet; do not weaken the template gate to resume historical Q8
timing. Full attempt/revert details are in PERF-LOG.

Integration update: `a5385e0c` now admits the Unsloth-patched template for dense
Qwen3.8 with pinned oracle coverage. The artifact blocker above describes the
tested base, not current main. This reopens an admitted witness for later
experiments; it does not clear the existing endpoint regressions, unexercised
controller screen, or restored-tail memory gate.

2026-09-06 restored pending-checkpoint DFlash admission is HOLD. Comparing
capture length with consumed rather than matched position enables the previously
serial continuations, but Q8 A-B-B-A exposes a policy tradeoff: code 128-output
wall saves 43.151%/36.919%, while blue two-output loses 11.755%/19.548% and prose
loses 16.909%/20.659% (prose control spread 5.225%, no stable effect authority).
Prose pays 21 exact fallbacks over 39 packets without backoff; below-16K control
ignores that replay cost. Prototype reverted, Q4 promotion replay deferred.
This elevates charged fallback/probe economics as the prerequisite to restoring
this capability, not another verifier tile or a content-specific selector.
Evidence: `docs/bench/2026-09-06-serve-restored-dflash/RESULT.md`.

2026-09-06 serial-tail allocation is now prefix-aware for dense 1-48-forward
suffixes, including pending-token consumption. It removes 1.31/3.40 GB of
actual Metal allocations at the measured 8K/32K short-tail shapes without
retile or KV-capacity changes. Multi-turn Q8+drafter/Q4 output and cache gates
pass; Q4 fresh-long wall has a disclosed 2.934% regression. This is primarily
a resource/admission win, not a universal latency win. The Q8 restored turns
in this packet run serial even with the drafter loaded; do not transfer its
4.297% code-request wall result to active speculation. Evidence:
`docs/bench/2026-09-06-serve-serial-tail-scratch/RESULT.md`.

2026-09-06 banked request-boundary wins: dense serial tails now omit non-final
norm/head/readback while retaining concurrent-GDN topology and drafter captures.
The measured 48-token TTFT gain is 4.334% Q4 no-spec / 4.278% Q8-DFlash, not a
4% full-request claim. HTTP admission now waits on socket readiness instead of
sleeping 50 ms and explicitly clears inherited nonblocking mode before request
reads. Final cached Q8/DFlash TTFT/wall moves `56.448/232.862 -> 15.237/174.151 ms`;
the small dense guard also passes. Long-output wall guards permit up to 2.15%
observed regression, so retain the endpoint-specific scope. See
`docs/bench/2026-09-06-serve-request-elision/RESULT.md`. Reprice warm small-tail
TTFT against this transport baseline before another allocation/kernel design.

2026-09-06 request-boundary screen: terminal-Off DFlash setup elision is HOLD,
prototype removed. Natural 24,194-token Q8 release A-B-B-A preserves all output
and cache counts, but warm exact-hit 64-output wall saves only 1.06%. A 28.636 ms
followup TTFT screen positive does not reproduce in the reverse pair; noisy
fresh/exact-TTFT rows confer no broader authority. Do not widen from eliminated
allocations/commands alone. See `2026-09-06-serve-terminal-off/RESULT.md` under
`docs/bench/`; the hard context policy and production path are unchanged.

Live-production measurement (/tmp/serve_38-dflash.log, 9h, 67 requests) and a
k3 adversarial review produced a new force-ranked serve queue. The measured
fact: 33 serial requests at 66K-133K ctx run 8.35-12.4 tps because speculation
is gated off for restored requests and above 16K ctx, while the served drafter
is all-SWA-2048 and F1 (2026-08-20) proved bit-identical drafts from a
2,048-column windowed cache. Status 2026-08-21:

- 1a (windowed cold capture) and 1b (checkpoint capture tail + restored-request
  speculation) are LANDED with F1/F2-split/F5/E2E gates passing (byte-identical
  two-turn serve outputs vs serial control). Remaining serve-repo work: port
  the admission/fallback from the main repo (the deployed checkout still lacks
  it). Status 2026-08-22:

- **Rank 1 CLOSED**: the spec-vs-serial divergence class is root-caused and
  fixed — an argmax tie inversion (three CPU sites vs the GPU kernel's
  lowest-index contract) plus a batched-verify near-tie flip, now guarded by
  the margin-based exact fallback (a490444). The divergent prompt is
  byte-identical with the guard on. **Rank 2 CLOSED** (1ab937f): Off-mode is
  non-terminal — capture-fed Off keeps the drafter cross-context and the
  ring alive, and a 1-in-8 re-probe policy re-enters speculation when the
  trailing-alpha window clears the break-even. **Second audit fixes landed**
  (fc5edc6): A1-A7. Updated ranked queue (long-context goal per the review):

1. **P4 coverage gates (byte-identity, cheap):** DONE except the
   recovering-content re-entry fixture — 2,600-token wrapped generation,
   restored 2,276-token wstart>0 seed, and fallback-heavy boundary
   publication all byte-identical. The re-entry fixture stays unpinned.
2. **Serial/verify attention bandwidth audit (NEW, partially DONE):**
   the group 4|6 split-K under-partitioning is root-caused and retuned
   (3cd4751: 128/512 tiers, 1.6-2.1x on the kernel). The residual
   ~4x gap to stream (111 vs 474 GB/s at 130K) is a limiter-capture
   follow-up. The packed-N8 verify reader (8x reuse) remains the
   larger lever; pre-pricing per the P2 census.
3. **P1 = F3-amended (physical economics DONE; attribution OPEN):** served
   Q8_0 verify(8)/single minima are 1.86x/2.27x/2.71x at 8K/32K/64K, below
   the 3.4 ceiling. The attention/GDN slope split remains attribution work,
   not a blocker for the explicit canary.
4. **P3 = F4-amended (OPEN):** two 64K canaries prove content bifurcation:
   code is 1.72x exact off while fallback-heavy prose is bounded at 0.952x
   projected serial throughput. Run the natural 60-133K alpha/fallback census,
   including sampled and phase-changing outputs; keep the 16K default guard
   until every losing cluster reaches parity and probe tax stays <=5%.
5. **Guard tuning/robustness follow-up.** Attribute the 9.5e-2 outlier
   delta to its batched kernel; extend fallback coverage to MoE verify
   paths when MoE speculation reopens; port the guard to
   generate_prompt_lookup (currently annotated, not guarded).
6. **GDN wavefront verify: REPRICED.** At <=16K the 48-layer recurrence
   latency (9.44 ms/token marginal) stays the top verify term; at 130K
   the ~103 ms attention slope overtakes it — order by which band the
   product goal is.
7. **Warm small-tail TTFT: KEEP.** 186 vs 150 ms gate, orthogonal to
   speculation; phase-localize before designing the 49-256-token path.
8. **Q4_K_M alpha/beta sidecar: demoted to cleanup.** ~2%, Q4-only; live
   server is Q8_0.
7. **Residue: mma8v N=8 Q4/Q6 KEEP-small; drafter KV F32->F16 merge into 1a
   follow-on; MoE B16 ragged KILL (1.0998x < 1.10x reopen); DSpark N2 HOLD.**

Also recorded: the Metal process lease is an exclusive process-lifetime flock
with no idle yield — a long-lived serve daemon blocks every other qwen binary
for its whole life (WAIT=1 blocks until exit). In-lib #[cfg(test)] tests use a
per-PID lease dir and are the designed lane for daemon-adjacent correctness
runs; integration-test binaries take the production lease path.

Evidence: `docs/bench/2026-08-20-windowed-dflash-pre-gates/`.

Note: the old "small-span warm-tail TTFT" follow-up is retained as item 5 of
the re-ranked queue above (186 vs 150 ms gate; design a path for longer small
uncached tails without changing output, tails up to 48 tokens already bypass
matrix setup; require a named warm-8K TTFT packet below 150 ms before
promotion).

## Qwen3.8 27B Launch Lane — 2026-08-14

The pinned Qwen3.8 Q4_K_M asset is a near-drop-in dense text backbone with one
attached MTP head. Ordinary no-thinking and default-xhigh generation work, and
the maintained retention guardrail is 48/48 retained and strict. That battery is
ceilinged against historical Qwen3.6 and does not establish broader capability.
Synthetic throughput is order-sensitive and supports parity-ish operation, not
a Qwen3.8 speed claim.

The first exact cross-pollination is promoted. Native Q4_K token lookup already
supports the model, and MTP consumes that same dispatcher. Removing MTP presence
from the otherwise exact backbone fingerprint avoids a 5.09 GB F32 materialized
embedding and removes 4.07 GiB of private memory. This is ordinary pageable
loading, not whole-model residency.

Ridge adds a second operator-bit-exact WIP default and changes the local
optimization map. Its Q6_K embedding now reuses the shared row kernel, removing
a 4,042,649,600-byte F32 expansion. Its IQ2_S/IQ3_S N2 FFNs also exposed a
32-column physical-tile cliff; shape-gated NC2 projection kernels are bit-exact
against singleton rows and cut the dirty-tree measured verifier 2.324x. A clean
committed confirmation remains mandatory for release promotion. Even after the
correction Ridge D1/N2 is 0.968x, while the final source-stamped Q4 row is only
1.023x. This moves broad MTP work below capability and effort policy.

The first optimization-only screen now bounds the obvious low-bit retunes.
At Ridge `pp1024`, FFN owns about 67% of GPU time, but the three projections are
already at the matrix-compute shelf. A bit-identical IQ2_S N64 tile is only
1.001-1.002x on both real FFN orientations. Singleton decode has a larger
20.237 ms / 36.77 ms low-bit projection share, but a 1.294-1.298x-byte direct
level repack reaches only 1.062-1.063x and loses bit identity by 1-2 ULP. Both
prototypes were removed. Tile-width reuse and byte-expanding decode repacks are
therefore closed unless a materially new representation clears their explicit
ceilings.

The first representation-level census also closes the obvious sparse/factor
branches before kernel work. Across two Ridge decode traces, roughly one quarter
of FFN-inner scalars are below `1e-2`, but at most 0.046% of complete physical
256-wide blocks are; a 0.1% activation-energy budget removes zero blocks at the
median. Mature GDN states are low stable-rank but not uniformly low numerical
rank: at position 512, rank 24 leaves 1.42% median / 20.28% p95 relative
Frobenius residual, while a 1% residual requires median rank 31 / p95 rank 100.
Observed alpha reaches `2.31e-20`, making inverse rollback numerically
ill-conditioned even without exact alpha-zero resets. Do not build sparse-down,
universal rank-24 state, or exact inverse-log rollback from scalar CDFs or
real-arithmetic identities alone.

Force-ranked queue:

1. **Demand high-ceiling changed work before another Ridge kernel.** Packed IQ2
   is matrix-compute-bound, native decode is instruction-limited, and the first
   29%-larger direct representation gains only 6%. Reopen low-bit storage only
   for an exact representation near 98 bytes/block or smaller that measures at
   least 1.12x on both FFN orientations and projects at least 5% whole-phase.
   Prefer actual weight-term, target-call, prompt-token, or MMA deletion. Do not
   retry N64, rows-per-simdgroup widening, dual gate/up accumulators, a larger
   direct-level sidecar, block-sparse down on the measured layout, or fixed-rank
   GDN state.
2. **Use capability only as a bounded regression guardrail.** The retained
   direct/no-thinking packet and effort stress rows are sufficient to catch an
   obvious optimization regression. Do not expand task characterization while
   no high-ceiling engine candidate is waiting on it; run the frozen packet only
   after a candidate clears primitive and phase gates.
3. **Validate long-context semantics before a 262K product claim.** Reuse the proven
   ledger/four-key/multilingual retrieval pattern at a shared short control, 16K,
   and 64K for Qwen3.8 Q4_K_M and Ridge. Metadata capacity and synthetic pp
   throughput are not retrieval evidence. This remains a product-correctness
   lane below changed-work optimization. Treat Q8 KV as a separate memory/quality
   decision.
4. **Compose existing exact prefix reuse instead of inventing a universal
   cache.** For cohorts with a real repeated prefix, use the already-proven
   snapshot/fanout machinery to prefill once and restore private suffix lanes.
   Keep fresh BS=1 requests on the direct path; do not pay indexing, eviction,
   or disk-state coupling where reuse is absent.
5. **Keep packed D1/N2 bounded to a decision packet.** Q4_K_M remains a small
   interactive positive (`1.023x` in the final source-stamped row); Ridge remains
   negative (`0.968x`) after exact low-bit N2 repair. If revisited, interleave
   named request archetypes, allocate long-context partial scratch lazily, and
   promote only a stable regime selector. The shared-KV attention primitive is
   about 2x only at 20K-32K and stays 16K-gated/opt-in. Keep recursive D3/D7 and
   rolling N1 closed.
6. **Keep measured closed lanes closed.** The bounded payload scanner found zero
   whole-tensor duplicates across 17.10 GB and zero exact stored-row duplicates
   across 3.30 million rows in selected front projections. Generic GDN recurrence
   removal has only about a 2.5% whole-prefill oracle. Do not build payload
   interning, generic/fixed-rank factorization, inverse-log rollback, WY
   recurrence, or another local recurrence scheduler without new evidence.
7. **Make the product boundary explicit.** Qwen3.8 support is ordinary text chat.
   Vision/projector execution, developer roles, structured tool calls/results,
   response formats, and reasoning-history objects remain separate capability
   gates and must not block the text launch.
8. **Keep whole-model residency closed.** Native embeddings, capability work,
   long-context validation, and any future draft mechanism use ordinary loading.
   None authorizes `MTLResidencySet`, pre-wiring, `mlock`, uncached reads, or the
   residency-coupled A10B selector.

Evidence: `docs/bench/2026-08-14-qwen38-27b-launch/`,
`docs/bench/2026-08-15-qwen38-ridge/`, and
`docs/bench/2026-08-17-structural-thinness-falsifiers/`.

## DeepSeek V4 Optimization Lane — 2026-08-04

DS4 Flash-0731 has crossed from architecture bring-up into optimization. Full
structural execution reaches 1,048,576 context positions; ordinary short
decode uses one Metal command/encoder and runs at about 37-38 ms command-GPU /
38-39 ms wall on the legacy 95.93 GiB IQ3_XXS asset, near the pinned llama.cpp
b10254 36.1 ms wall floor. Practical long context is the leading measured
optimization opportunity, not yet a demonstrated product-speed differentiator.

The latest capture-first Lightning screen closes two proposed representation
shortcuts. On one real-text 8,412-token request, a positive-weight
real-arithmetic query/key norm envelope has zero observed violations over
397,635 deployed scores but retains 100% of rows at block sizes 16-256;
126/189 rank-512 cutoffs are nonpositive. This is not a formal fast-math error
certificate, but any required safety allowance only loosens the failed bound.
Even a perfect fixed-block oracle must retain 62.31% of rows at block 8 and
78.28% at block 16, far above the 20-25% charged-work target. Cache-order top-512
IDs have median 205 exact runs; one-gap merging still leaves 153 spans, while
eight-row pages amplify selected payload 2.55x. Keep the current cooperative F32
scorer and direct cache-order ID attention. Reopen certification only for a
materially tighter row-specific bound with complete charged economics; do not
build norm-cone blocks, page loading, or run descriptors from overlap alone.

That baseline remains historical rather than being silently transferred across
weights. The 2026-08-04 97.05 GiB refresh is now the sole resident product asset;
an initial fixed boundary probe measured 43.08 decode tokens/s versus 30.90 on
the legacy recipe, but it is not a promotion-grade throughput packet. Capture a
paired current-asset baseline before attributing further whole-token gains or
comparing against llama.cpp.

The refreshed-asset bracket also exposed and closed one bounded policy cliff.
Sparse CSA begins at row 513 / token position 2,051, but production retained the
scalar remove-one-worst-row selector through row 1,024. Dispatching the existing
radix4 selector at the first pruned row raises warmed decode on the identical
2,385-token request from a 6.685 token/s scalar midpoint to 22.62 token/s, cuts
generation time 70.4%, and preserves every generated ID and logged first-token
logit bit. Candidate prefill remains between both controls. This resolves token
positions 2,051-4,095 without changing the deep selector or attention contract.

Cooperative Lightning scoring is promoted. It preserves every production score
bit and cuts the operation from 0.701/2.0-2.2/8.3-8.4 ms to
0.141/0.533/2.102 ms per CSA layer at 16,384/65,536/262,144 rows. A real-weight
position-65,663 bracket saves 10.8-11.6 ms in both command-GPU and wall time
across two fresh processes with exact decisions, logits, and causal state.

The first exact query-reuse follow-up is closed. A reviewed R2 candidate keeps
the F32-query/F16-key arithmetic and reduction order but serializes two rows per
simdgroup. All exactness and safety checks pass. Its stable 16,384-row cell
regresses from 0.658250/0.669750 to 1.127167/1.134417 ms median/p95, and every
candidate sample is slower. The packet formally HOLDs when both 65,536-row arms
exceed 13.8% half drift, so that deeper cell is directional only and 262,144 is
not run. The mandatory stable shallow miss still rejects this exact design. It
is removed with no rerun or automatic R4 rescue. Reopen only for a materially
new schedule or specific compiler/occupancy attribution that explains how the
0.468917 ms shallow penalty is avoided.

Four-bit radix selection is also promoted. It replaces 32 full-history bit
scans with eight nibble scans while preserving the exact threshold and stable
tie contract. Mixed terminal selection falls from about 4.58 to 1.88 ms/layer;
four complete position-65,663 brackets save 1.06-1.29 ms command-GPU and
1.06-1.70 ms wall. The conservative isolated 21-layer CSA projection is now
about 12.6/29.2/93.4 ms at 64K/262K/1M-token-equivalent histories.

Online singleton HCA is promoted. One simdgroup scans shared KV once while
maintaining F32 online-softmax state; packed HCA remains on the exact tiled
kernel. At 8,192 rows it reduces 5.26 to 3.410 ms/layer, moving the isolated
20-layer subtotal from about 105.2 to 68.2 ms. Two complete warm terminal
legacy/online/legacy campaigns (four packets total) save 22.5-24.5 ms
command-GPU and wall and put the online endpoint at 210-213 ms, about
4.70-4.73 token/s. Terminal transcripts repeat
bit-for-bit and preserve every consumed CSA/MoE ID and status. First-touch wait
remains an independent unattributed cold-start observation.

The packed-indexer lane's optimistic matrix ceiling is also established. With
Q rounded to F16 once and current F16 K already decoded, an eight-simdgroup
64-head x 8-row scorer takes 0.166/0.197/0.518-0.519 ms at
16,384/65,536/262,144 rows versus current brackets around
0.658/0.797/2.102 ms. Terminal saving is 1.584-1.586 ms/layer and per-layer Q
conversion is about 0.008 ms. This is feasibility evidence only, not production: it does
not implement official Q/K FP4 QAT or change the v1 F16 cache contract.

Product-depth attribution now prevents overvaluing that scorer ceiling. The
clean 8K split assigns 25.921 seconds to CSA attention body, versus 6.401 to HCA
and 0.597 to local attention. A diagnostics-only F16-query matrix caller reduces
CSA by 1.101 seconds, but ordinary wall moves only 1.31% and final-logit bits
change. Remove it without a quality battery: scoring is not the dominant
2K-row CSA term. The incumbent packed selected-attention kernel instead launches
2,048 x 64 width-640 threadgroups per sparse layer, scans 512 dimensions in each
row lane, then scans up to 640 rows again for output. Port the already-proven
32-lane online-softmax dataflow to selected row IDs before revisiting scorer
precision.

That port now clears both mechanism and product gates. One 32-lane simdgroup
stages each selected F16 row once and updates online-softmax state in cache
order. CSA attention falls from 25.921 to 9.749 seconds on the clean 8K profile;
ordinary wall falls 12.04%, from 122.704 to 107.927 seconds, and prefill rises
from 66.76 to 75.90 token/s. Primitive error stays below `9.32e-10` maximum
absolute and `9.04e-7` relative RMS. Although final logits change, all generated
IDs and core values repeat across the 9,960-token ledger, 6,092-token structured
retrieval, and 7,263-token multilingual retrieval. Structured and multilingual
formats remain exact; the ledger's already-invalid literal-total format receives
no credit. Default packed sparse CSA to online attention; rollback is
`QWEN_DSV4_PACKED_SELECTED_ONLINE=0`.

The scalar packed contract is now frozen. Revision-addressed official, vLLM,
and DwarfStar sources pin BF16-before-amax semantics, four 32-value blocks,
adjacent low/even and high/odd E2M1 nibbles, four UE8M0 scales, the
`6 * 2^-126` floor, round-to-nearest-even, and signed zero. The strict generated
fixture is SHA-256
`0e5e2b251a960d417e7977608a363b83e072e2d90bc286cc52820b0ea7dc2b1f`.
Its 68-byte row is an oracle envelope, not an upstream cache-layout claim; the
score vectors are explicitly deterministic scalar transcriptions over decoded
packed operands with already-normalized head weights. Scoreable rows reject any
physical encoding that would overflow finite F32 dequantization.

The official packed-semantic Metal shadow now clears its frozen schedule gate.
The first implementation decoded all Q heads in every eight-row K tile and was
KILLed at 1.634 ms terminal, only 0.492 ms/layer faster than current. The
admitted schedule expands authoritative packed Q once into a transient 16 KiB
unit-value slab while K remains packed in the matrix scan. Two valid campaigns
measure terminal current/FP4/current `2.127/0.841/2.126` and
`2.127/0.841/2.124` ms, saving 1.285/1.284 ms/layer with 0.842/0.842 p95. Q
pack and unpack are included; one-row K pack is 0.0134-0.0137 ms. Every frozen
pack byte, score vector, decision, tail, offset, and status gate passes. This is
test-only schedule evidence: synthetic K is prepacked, status-0 planes are
trusted, and no cache or snapshot ABI changes.

The real-weight sidecar now clears its first whole-token decision and quality
gate. Post-Hadamard, pre-F16 K rows publish into separate packed value/scale
planes under transactional status; one preflight validates exact visibility,
all 64 Q statuses, and every visible K status. The counterfactual borrows only
FP4 selector IDs while selected attention retains F16 history. On the refreshed
asset, packed/singleton masks are exact in 8/21 and 11/21 CSA layers; every
difference is one reciprocal rank-512/rank-513 exchange. Packed and singleton
logits preserve argmax at cosine 0.999999967/0.999999996 and relative RMS
0.000260574/0.000091580. Controls are bit-identical and all snapshot, restore,
trace, memory, and fail-closed gates pass.

This promotes diagnostics infrastructure, not production FP4 selection. The
first-sparse-boundary packet deliberately computes both scorers; its positions
2,053-2,060 timing bracket spans only 513-515 visible rows and measures candidate
47.721 ms versus a 46.593 ms control midpoint. It cannot authorize terminal
savings. Snapshot v1 remains F16-authoritative, and exact selector identity is a
failed falsifier rather than a hidden gate change.

The bounded no-double-score falsifier now clears. Exhaustive session modes derive
F16-only, FP4-only, or paired execution, and common query preparation is split
from F16 score/selection in singleton and packed paths. Across paired A,
FP4-only, and paired B, positions 2,051/2,052/2,060 are bit-identical in logits,
causal state, consumed-ID trace, and committed-token transcript. Decision
transcripts including routes and FP4 reports are exact across all arms at the
singleton audit. Packed reports compare paired controls only; timed positions
capture none. Encode-time ledgers prove zero F16 score/selector pipeline
invocations on FP4-only positions.

With no report work in any timed arm, repeated GPU medians are
46.738/45.291/46.802 ms. FP4-only saves 1.447 ms against the faster control with
0.136% control drift, clearing the frozen 1.0 ms gate. This authorizes collapsed
engineering only: the campaign uses one command per layer at 513-515 visible
rows and cannot establish ordinary-token or terminal savings.

The authorized collapsed experiment is now complete and KILLs the direct FP4
selector premise. Layer-addressed outputs, inactive poison records, ledgers, and
189 consumed-layer traces all validate, but the first useful deeper packet at
position 3,070 changes every audit mask by 24-102 IDs. Candidate logits fall to
0.983313 minimum cosine, 0.182029 maximum relative RMS, and 3.19737 maximum
absolute error. GPU/wall medians regress by 1.456/1.772 ms against the faster
F16 control. The shallow result was not contradictory: selecting 512 of only
513-515 rows structurally bounded the visible set difference and could not
establish deeper ranking stability.

Retain the diagnostics implementation and packet as negative evidence. Direct
FP4-Q/K replacement, paged FP4 K, snapshot v2, and more same-design tuning are
closed. Reopen only for a materially new guarded or mixed scorer that first
clears the frozen deep quality gates and has a positive all-in timing ceiling.

The exact multi-group selector's threshold-only ceiling now clears. Eight
producer/reducer pairs preserve the deployed F16 finite-key and tie contract.
At 262,144 visible rows, the frozen 32-group geometry measures 0.351/0.351 ms
median/p95 on mixed scores and 0.600/0.600 ms on all ties, saving 1.507/1.410
ms against the faster current controls. The 64- and 80-group alternatives fail
the tied-case gate and are closed. This is Phase-A feasibility only: it emits
validated threshold state and per-partition counts, not masks or IDs.

The frozen 18-dispatch full selector now clears as well. The final reducer owns
partition tie quotas and cache-order offsets; 32 compactors write private byte
masks/IDs; one publisher validates generation, plans, completion, mask
population, and strict ID membership before exposing output or first-K
fallback. Terminal mixed GPU/wall medians are 0.674/0.818 ms versus faster
controls at 1.874/2.040 ms. All-tied medians are 0.907/1.075 versus
2.025/2.203 ms. Candidate p95 remains below 1.14 ms and every full output is
exact. This qualifies the topology, not a production dispatch switch.

The model-free crossover is now frozen. Equal-capacity mixed/tied output first
wins at 98,304 rows, but tied GPU/wall savings there are only 0.059/0.059 ms and
remain 0.264/0.268 ms at 131,072. At 196,608 visible rows, every measured target
capacity clears the 0.50 ms bound: tied savings are 0.691/0.677 ms at capacity
196,608, 0.641/0.635 at decimal-million capacity 250,112, and 0.627/0.638 at
capacity 262,144. A 262,144-row session at only 131,072 visible rows has just
0.122/0.103 ms tied headroom, so visibility alone is insufficient.

The off-by-default production seam now clears. Five session-owned buffers add
267,432 logical bytes at terminal capacity and are included in admission; a
nonzero invocation owner fails before generation reuse. Integrated model-free
current/candidate/current gates preserve every output and save at least
0.624/0.627 ms GPU/wall per layer in the frozen band. On the current asset at
position 786,431, all 21 candidate layers execute from a real-weight synthetic
zero-causal-state fixture restored through snapshot-v1, with bit-identical
logits and final normalized hidden values, matching causal, prefix, and
compatibility digests plus committed tokens, and a separate untimed exact
decision transcript. Whole-token GPU/wall saving is 13.867/17.573 ms against
the faster control. This is not real-prompt continuation evidence and qualifies
only the bounded hidden opt-in, not default-on or wider-device routing.

That hidden opt-in now has a bounded product contract. The CLI exposes only
`off|qualified-experimental`, defaults to radix4, and accepts the experimental
value only for native DeepSeek V4 on exact Apple M4 Max. It validates both
physical capacity and `forward_limit / 4` reachable visibility before snapshot
loading, admission, or residency realization; the rounded-but-unreachable
786,431-forward case rejects and 786,432 is the first accepted budget. Every
single-turn or JSONL request session seals before restore/prefill, emits
requested/sealed/actual-invocation telemetry, and retains radix4 for packed or
dynamically ineligible positions. Snapshots and memory planning are unchanged.
This promotes an operator opt-in, not default-on or broader evidence, and does
not repeat the 5.43 GB restore campaign.

The packed GPU route/schedule integration does not clear. Across all 86
layer/chunk records, same-input GPU and Rust route IDs are exact, but small route
weight differences reach 1.7762e-5 and amplify to 0.043026507 packed-logit and
0.030833979 restored-continuation relative RMS. Replacing only GPU weights with
same-input Rust weights restores output bits and all recorded state digests,
isolating the arithmetic lineage. The uninstrumented candidate also regresses
one warm comparison by 91.910 ms, or 2.15%; the final instrumented bracket is
not an isolated seam timing. Remove its ordinary switch and seven allocations;
retain the exact topology and fault harness under diagnostics.

The independent grouped-compute lane now clears. The exact Rust schedule feeds
compact 32-assignment expert-major tiles; 25 current-asset
IQ2_XS/IQ2_XS/IQ3_XXS layers replace their per-bucket chain with grouped
gate/up + SwiGLU and grouped down/scatter dispatches. Model-free edge cases and
current-asset N=12/32/128 integration preserve output bits and recorded state
identity. N=128 R5 wall medians improve from the faster 3,617.414 ms control to
2,517.379 ms, or 30.409%, with 3.040% control drift. One R3 bracket misses the
wall gate at 12.2%, so this is a strong median benefit, not a per-run guarantee.
Traced R3 medians save 35.91% post-route GPU and 21.64% summed packed-command
GPU. Enable only on the qualified Apple M4 Max by default, retain an isolated
rollback and capability fallback, and admit the 6,291,456-byte scratch even
when fallback executes.

The one authorized widening does not promote. A separate mapped IQ3_XXS kernel
and exact arena proof let all 16 IQ3_XXS/IQ3_XXS/IQ3_XXS layers execute as four
diagnostic dispatches without another allocation. Model-free stagewise and
current-asset N=12/32/128 output/state comparisons are bit-exact. Two R5
campaigns are rejected for control drift. The sealed balanced R8 stabilizes
controls at 0.721% drift and observes 10.224%/16.103% half savings, but candidate
drift is 6.050%, above the frozen 5% gate. A separate traced bracket saves 9.929%
total packed GPU and 56.534% in the 16 affected layers while unchanged-layer GPU
time is flat. Retain the exact test-only implementation and negative evidence;
do not retry the same condition or build fused IQ3 gate/up from that checkpoint.

The first pre-expert subphase now closes. M4 stage-boundary pass intervals can
overlap, so an eight-stage all-layer profiler fails its frozen ambiguity gate
and is removed as a live lane. A narrower three-pass instrument isolates the
complete 128-row chronological publication loop while preserving dispatch work
and exact outputs plus matching causal identities. Ordinary controls total
1,082.499/1,051.893/1,058.030 ms GPU; sampled runs total
1,058.170/1,074.016 ms. Chronological work normalizes to
74.620/71.925 ms, with a 54.163 ms/5.099% uncertainty-adjusted lower bound.
That misses the 150 ms/15% authorization floor. Keep it as a CSA-weighted
piggyback opportunity, not a standalone optimization target.

The same supported instrument now closes the next branch. Attention body plus
output occupies 46.897%/47.238% of pre-expert GPU and has a 500.011 ms/46.420%
uncertainty-adjusted lower estimate. A direct four-pass split reproduces that
combined envelope within 1.449/0.019 points. Attention body alone misses at
113.488 ms/10.641%; `encode_output` clears decisively at
377.528 ms/35.397%. Controls drift 2.019% and sampled runs drift 2.447%.
Packed logits, normalized hidden, and restored continuation bits remain exact;
recorded state digests and all 47,990 dispatch geometries match. This authorizes
output work, not a generic attention rewrite.

Both tested output families are closed for the current asset on Apple M4 Max.
Exact mapped T1 preserves bits but measures 8.997 versus 8.912 ms/layer at
N=128; exact T4/T8 regress to 11.408/11.855 ms. The existing F16-staged Q8
matrix schedule exposes a 75.525% model-free stage ceiling, but N=32 fails
quality and the bounded first-chunk 128+12 packet fails the frozen continuation
and routing gates. Despite passing performance at 27.851% first-chunk
pre-expert GPU, 23.291% aggregate GPU, and 15.847% aggregate wall saving,
position-140 hidden state falls to 0.997495086 cosine / 0.070735848 relative RMS
and consumed packed plus continuation expert IDs change. Remove all candidate
source; do not resweep the same condition under this device/asset contract or
hide it behind an opt-in.

The post-route decomposition now closes the remaining broad attribution gap.
Three accepted sampled runs preserve exact outputs/state and all 47,990 dispatch
geometries with 0.275% GPU drift, no more than 0.199% absolute perturbation,
1.000000 six-decimal raw coverage, and at most 0.0066% aggregate transition
ambiguity. Routed execution owns 1,015.042 ms/96.052% of the 1.055-second warm
N=128 post-route span. The shared expert, combine, and hyper/head account for
only 33.733/1.998/5.923 ms. Routed cohort means are 597.107 ms for the 25
grouped IQ2 layers, 355.062 ms for the 16 per-bucket all-IQ3 layers, and
62.874 ms for the two MXFP4-down outliers. The old 688 ms unchanged-cohort
trace is not a transferable warm budget.

This observer is a materially changed condition for the held all-IQ3 candidate:
it isolates the exact affected stage without reordering 1,036 bucket chains.
The single authorized promotion packet defines A as the current grouped-IQ2
policy and B as grouped IQ2 plus the existing all-IQ3 candidate. Both use the
same four-stage sampled encoder topology. Run one untimed `ABBA BAAB` warm block,
then time `(ABBA BAAB) x 2`: eight samples per arm, four per arm in each half.
For each arm/half/endpoint, the conventional even median is the mean of the two
middle sorted samples. For each arm and endpoint, half-to-half stationarity is
`2 * abs(median_1 - median_2) / (median_1 + median_2)` and must be at most 5%.

Endpoints are complete packed wall, total post-route command GPU, and the
16-layer affected-cohort routed GPU subtotal. In each matched half, saving is
`1 - candidate_median / control_median` and must reach 5%, 10%, and 30%,
respectively. Overall p95 uses nearest rank over all eight samples, so p95 is the
maximum; candidate p95 must not exceed control p95 at any endpoint. Every timed
sample must pass timestamp coverage/ambiguity validity; no sample may be deleted
or replaced. Exact output/state evidence and 25/0 versus 25/16 grouped invocation
counts are adjudicated before timing. Dispatch geometry must repeat within each
arm but need not match across deliberately different policies. Any failed gate
ends the one authorized attempt; no same-condition retry is permitted.

That attempt is now complete and closes production all-IQ3 promotion under this
contract. All eight warm and 16 timed executions complete, and every warm arm
passes exact output/state, topology, and invocation checks. Timed adjudication
stops on the first sample, a control arm, after it passes exactness and topology
but reports 18.27410241% aggregate transition ambiguity against the frozen 2.5%
maximum. The remaining timed executions are not adjudicated, so the packet
authorizes no wall, GPU, saving, stationarity, p95, promotion, or regression
claim. Retain the exact diagnostics-only candidate and failing ignored harness;
do not rerun, relax, or split the condition as a rescue.

The materially new two-dispatch all-IQ3 falsifier is now complete and `KILL`.
Its fused gate/up/SwiGLU kernel preserves both mapped IQ3 accumulation lineages
and exact reduced-shape gate/up/inner/final bits, while production-shape final
output, inner, guards, and the four-to-two dispatch census also pass. The sole
24-sample A/B/A packet is unusually stationary: every control/candidate drift
is at most 0.1760%. Nevertheless hot GPU/wall medians regress
0.5713%/0.7625%, and sparse medians regress 1.8763%/1.8684%; p95 regresses as
well. Do not run the current asset, retune the same 16-accumulator geometry, or
repeat this condition. Retain the test-only path as executable negative
evidence.

The subsequent current-route bank-axis packet closes the broader mapped all-IQ3
question under the existing economic contract. It replays all 16 captured
current-asset schedules and compares the actual 6,216-dispatch bucket path with
48 dispatches: depth-two mapped gate/up, exact SwiGLU, and mapped down. Every
output/read-only bit and guard passes. GPU upper median falls from 348.206875 to
144.260875 ms and p95 from 353.384000 to 144.285083 ms, with 2.7158%/0.0624%
drift. The 47.986875 ms five-range charge leaves a 155.959125 ms median lower
saving, 2.340875 ms below the frozen 158.3 ms floor, so the formal result is
`KILL_BANK_AXIS_MAPPED_ALL_IQ3`. Remove the candidate and do not rerun the same
launch-axis geometry. The 144 ms residual and 58.57% raw cohort reduction make
a materially new low-bit matrix representation or larger effective prefill
batch the next high-ceiling routed-expert hypotheses.

The precision-recovering Q8 branch clears model-free and fails full-model
promotion. A single fixed F32 `R2C4K64` geometry removes both half operand
conversions and improves synthetic A+B relative RMS 220.7x to 0.000005573 while
saving 72.2179% GPU median. In the sole canonical current-asset packet, however,
position-140 and restored-continuation hidden relative RMS remain
0.066547844/0.052519175; packed and continuation expert IDs change with a
0.352247149 maximum packed route-weight delta. First-chunk/aggregate GPU saving
passes at 26.554%/22.064%, but aggregate wall saving is only 12.725% and
candidate wall drift is 8.773%. Remove the live seam and close this output
family under the current asset/device contract. Retain the model-free kernel and
tests only as bounded arithmetic evidence.

The exact joint chronological/attention candidate stops before implementation.
A deliberately impossible zero-work screen times all 21,887 current N=128
query/KV/inverse-RoPE and raw-publication dispatches in one command over 43
disjoint production-sized banks and one independent warm bank. It grants the
candidate removal of every operation and byte, then requires stable controls
before comparing p95 plus five times the observed range with the existing
158.3 ms floor. The first and one conditioned campaign produce suggestive upper
estimates of 46.299 and 65.659 ms, but warm drift first reaches 6.675% and the
conditioned run later shifts both regimes to 12.843%/15.213%. Both are
`INCONCLUSIVE`. Enforce the one-repair stop: archive both, remove the profiler,
and implement no kernels. Reopen only for materially better instrumentation, an
independently corrected timing cause, or relevant device/compiler drift under a
new protocol.

The grouped-MXFP4 follow-up is closed analytically before another profiler. At
N=128, layers 26 and 42 execute 1,536 row-wise down matvecs and 117 scatters,
but their accepted complete routed-stage samples are only
63.101/62.246/63.273 ms. Even impossible deletion of the largest complete stage
reaches 39.97% of the existing 158.3 ms floor; a grouped down/scatter kernel
changes a strict subset and has nonzero replacement cost. The GPU packet does
not measure host encoding, but closing the 95.027 ms gap would require an
average of more than 57.49 us of removable host work per affected dispatch
before replacement encoding. Reopen only with stable direct evidence for that
premise, composition with a larger measured family, material asset census
drift, or relevant device/compiler drift.

The grouped-IQ2 count census now closes the schedule premise. One exact
current-asset N=128 execution binds accepted output/state identity and captures
25x256 expert populations with canonical payload SHA-256
`0ab99253...58a6`. The 25 layers contain 1,542 active experts and 1,748 width-32
tiles: 19,200 useful assignments occupy only 34.325% of 55,936 down columns,
leaving 36,736 padded columns. Exactly 1,395/100/35/12 experts require one/two/
three/four panels. The 206 continuation panels are only 11.78% of tile work;
uniformly scaling the entire 597.107 ms envelope by that ratio yields 70.368 ms,
but phase attribution has not established uniform per-tile latency. Keep it as
an operation-count ranking signal, not a timing ceiling or KILL. Close only
dispatch aggregation, which is already one dispatch per phase and layer.

The count census does not assign time between the two production dispatches.
Width-16/32 and width-8/16/32 down plans have 36.10% and 50.92% column-work
ceilings, while gate/up masks inactive accumulation but retains shared tile
dequantization. Preserve these as operation counts, not latency projections.

The exact-route census closes the last schedule-control ambiguity before phase
attribution. One additional bound current-asset run captures 25x768 slot-major
expert IDs with canonical SHA-256 `505cb93f...afdd`. Every token has six
distinct IDs, and independent reconstruction exactly reproduces production
source rows, destination slots, ascending expert buckets, and the prior count
fixture. The 6.76-second runtime and zero swaps are capture provenance, not
phase timing. Remove the one-shot harness and use only this exact schedule in
the primary campaign; synthetic permutations no longer constrain the KILL
claim.

The sole exact-route phase campaign is `INCONCLUSIVE - timestamp-invalid`.
Gate validation and one untimed D/W/W/D conditioning block finish, but empty
gate pre-bracket command 1 receives equal finite GPU start/end timestamps. The
frozen contract rejects that censored duration before any retained gate sample;
the down campaign never allocates or executes. Preserve no phase timing or
economic result. Remove the profiler and keep grouped-IQ2 on
`HOLD - closed inconclusive`.

A separately reviewed V2 resolves only the censored empty-bracket premise with
full per-command wall upper bounds. Gate disjoint/warm cells pass stationarity
at 0.344817%/0.059705%; `P=527.963833 ms`, `R=9.086871 ms`, and
`C=U=537.050704 ms`, while the 0.492790 ms empty bound is non-dominant. This
authorizes gate/up/SwiGLU candidate design only. Down warm drift reaches
5.483643%, so down remains `HOLD - INCONCLUSIVE` with no timing decision.

The first gate-only candidate, exact multi-bin FFN-row packing, is
`HOLD - INCONCLUSIVE` and removed. It is bit-exact and cuts nominal lane-MAC
capacity by 59.039%, but every stable GPU candidate sample regresses by roughly
24-25%. Wall cells miss stationarity, so preserve this as raw negative evidence,
not a contractual KILL. Do not rerun, tune widths, or spend an asset gate.

The dispatch-neutral follow-up is a decisive `KILL`. It preserves all 25
dispatches, routes, scalar IQ2 dequantization, accumulation order, and output
bits while replacing each 64-float TGM publication and two barriers with scalar
exact-bit SIMD shuffles. All four GPU cells pass stationarity, but every
candidate sample regresses: disjoint median/p95 moves from 527.743/540.779 to
2,406.168/2,429.985 ms and warm moves from 527.151/546.902 to
2,406.267/2,456.642 ms. Remove it and close grouped-IQ2 scalar row, panel, width,
TGM, barrier, and shuffle retuning under the current asset/device contract.

The retained four-pass pre-expert packet now closes its two previously
unreported residual intervals without another GPU run. Apply the packet's exact
normalization and uncertainty method, with the current 158.3 ms floor.
`BeforeAttentionBody` has 481.678/499.786 ms normalized samples and a
477.570 ms/44.7712% lower bound after charging 1.233408% uncertainty; mean
CSA/HCA shares are 47.6961%/43.4194%. It is `GO` for decomposition only.
`AfterAttentionOutput` has a 56.583 ms/5.3057% lower bound and only
5.9380%/6.5423% CSA/HCA shares, so it is `KILL` as a standalone current-asset
N=128 target.

The proposed one-encoder before-attention observer is also closed on the
current device/API contract. Apple M4 Max reports no dispatch-boundary counter
sampling support, and stage-boundary sampling cannot expose legal mid-pass
samples. Rotating encoder boundaries retains the same unowned transition and
is not a replacement. The rejected observer is removed; reopen only for a new
legal intra-pass primitive or relevant device/Metal capability drift.

The N=2,048 default is exact through 512 CSA and sixteen HCA publications and
the retained 64-token continuation. Same-binary AB/BA moves the 2,385-token
prompt from 53.87 to 45.08 seconds versus N=512, with an identical complete F32
prompt-logit digest. Stop cap growth here; reopen a larger work unit only if
longer-prompt attribution prices the remaining chunk boundary above the matrix
work below.

The clean 8,192-token product-depth profile now adjudicates that reopen. Its
ordinary reference is 122.328 seconds or 66.97 token/s. Across four N=2,048
chunks, attention body grows from 5.010 to 9.706 seconds while non-GPU residual
only grows from 0.854 to 1.297 seconds. Even granting each adjacent pair free
deletion of its larger residual prices the N=4,096 boundary at only 2.495
seconds or 2.04% of request wall. Real-prompt route occupancy is stable at
86.02% for BM16 and 73.44% for the width-32 cohort, with 201-256 active experts
per layer. N=4,096 is therefore HOLD as a composition opportunity rather than
a standalone cap build; reopen only with a separately measured GPU mechanism
that lets the combined credible net benefit clear the existing 2-3% gate after
replacement and memory costs.

Exact grouped IQ2/IQ3 experts now cover all packed chunks through N=2,048.
Buffer-backed 32-assignment descriptors remove the old 4 KiB inline ceiling
without changing the stable expert/token/slot schedule or arithmetic. A/B/B/A
moves the 2,385-token exact request from 46.08 to 39.50 seconds, raises prefill
from 51.8 to 60.4 token/s, and preserves the complete prompt-logit digest plus
all 64 generated IDs. The full-chunk trace still assigns 7.825 of 10.530
post-route seconds to the 25 IQ2 layers, making wider BM16 execution the next
largest exact post-route hypothesis.

BM16 now consumes the same stable schedule at the two measured full work units,
N=128 and N=2,048. A dedicated buffer-backed 16-route plan preserves every
gate/up/SwiGLU bit and keeps the simultaneous 32-route down plan disjoint.
A/B/B/A moves N=2,048 prefill from 40.43 to 36.31 seconds, or 59.0 to 65.7
token/s, with identical prompt logits and 64-token output. The full-chunk trace
moves post-route from 10.530 to 7.218 seconds and the IQ2 cohort from 7.825 to
4.542 seconds. Pre-expert work now leads at 20.368 seconds.

The reusable N=2,048 trace assigns 3.663 of those 4.542 BM16 seconds to
gate/up. A real prompt needs 22,387 width-16 descriptors but only 13,097
width-32 descriptors, so an exact two-SIMD-group BM32 candidate tested whether
two route panels could share each decoded weight tile. The mechanism is real
and bit-exact, but not economic: all 25 layers improve by 10.18-12.72%, while
aggregate gate/up moves only `3,662.943 -> 3,244.204 ms`. Ordinary warm wall
moves `28,273.485 -> 27,969.800 ms`, or 1.07%, missing the frozen 20% stage /
739 ms survival gate. Remove BM32 and close same-work panel widening at N=2,048.

The matrix-shaped Q8 compressor premise clears where exact panel sharing does
not. A diagnostics-only half-staged oracle substitutes exactly 124 compressor
projections on a full N=2,048 chunk. A/B/B/A ordinary wall moves
`28,947.093 -> 27,779.258 ms`, saving 1,167.835 ms or 4.03%; sampled wall saves
1,051.409 ms. Post-route GPU regresses by 44.633 ms, so the adjusted pre-expert
saving remains the full 1,167.835 ms. This proves a material matrix opportunity,
not product quality by itself. The bounded quality follow-up now clears: the
9,960-token ledger and 6,092-token structured retrieval reproduce every paired
exact-arm generated ID, including all five ledger values and total 2,578, while
the multilingual guard returns all three frozen values. The ledger's literal
schema clause is invalid rather than passed: its retained exact baseline also
emits the arithmetic expression despite the `<integer>` instruction. Promote
half-staged matrices only on Apple M4 Max for the structurally qualified current
1,328-tensor / 104,202,502,492-byte payload profile and its 124 eligible Q8
projections in complete N=2,048 chunks. Other assets/devices, non-Q8 weights,
tails, decode, raw KV, q_b, outputs, and experts remain unchanged; rollback is
`QWEN_DSV4_PACKED_Q8_COMPRESSOR_MATRIX=0`.

Prior exact T2 already regresses the same 4,096x512 raw-KV geometry, so do not
authorize another exact compressor kernel without a new mechanism and an
equally large ceiling. The q_b quality failure likewise remains closed; this
compressor-specific promotion does not imply a global Q8 matrix crossover.

Packed sparse-indexer query RoPE now uses the existing consecutive-position
batch kernel by default. A full N=2,048 sparse chunk replaces 43,008 scalar
dispatches with 21 batched dispatches. On the clean 8K profile, CSA attention
moves `9,748.681 -> 9,407.718 ms`, pre-expert GPU saves 1.35%, and ordinary
request wall moves `107,926.903 -> 106,576.900 ms`, or 1.25%. Prefill rises
from 75.90 to 76.87 token/s. The established primitive differential remains
within `6.41e-7` maximum absolute error and `1.47e-7` relative RMS, while all
generated IDs and requested values remain unchanged across the three retained
long-prompt discriminators. Default the batched path with
`QWEN_DSV4_PACKED_INDEXER_BATCHED_ROPE=0` as the scalar rollback. This removes
linear dispatch growth but does not independently reopen N=4,096.

The retained F32 Q8 Q-B matrix now defaults on for complete N=2,048 chunks on
the qualified Apple M4 Max/current-asset profile. Before-attention GPU falls
`21,571.387 -> 5,945.840 ms`, pre-expert GPU falls 20.77%, and ordinary 8K wall
moves `106,576.900 -> 91,647.479 ms`, or 14.01%. Prefill rises from 76.87 to
89.39 token/s. Three clean current-lineage long-prompt tasks reproduce every
incumbent generated ID and requested value, including the ledger's correct
2,578 total. This supersedes the earlier HOLD: its pre-compressor and
pre-online-attention candidate missed that total, while the promoted numerical
bundle restores it exactly. Automatic scope remains tied to the measured
device, 1,328 tensors, 104,202,502,492 source bytes, and a complete N=2,048
chunk. `QWEN_DSV4_PACKED_Q8_QB=exact` is the rollback; partial or unqualified
chunks retain exact GEMV under `auto`, while explicit `f32_matrix` remains a
force mode for full N=2,048 chunks.

Same-GGUF llama.cpp b10297 now supplies the missing external calibration on
this M4 Max. Warm pp512/2048/4096 rows are 255.64/244.47/224.67 token/s with a
2,048-token batch and 512-token ubatches. Its deterministic random tokens are
not identical to the retained native prompt, so treat this as a standard
schedule floor rather than a paired product ratio. The source and native trace
agree on the next two gaps: llama.cpp routes all prompt-width Q8 projections
through dequantizing matrix kernels and keeps MoE compaction plus indirect
IQ2/IQ3 matrices on GPU. Before the output promotion, native output
projections and routed work cost 35.11 and 29.20 seconds respectively on the
8K Q-B trace.

F32 Q8 output A/B matrices now default on under the same qualified profile as
Q-B. Output-projection GPU falls `35,110.500 -> 5,322.007 ms`, pre-expert GPU
falls 51.38%, and ordinary 8K wall moves
`91,647.479 -> 61,979.581 ms`, or 32.37%. Prefill rises from 89.39 to 132.17
token/s, and the sampled request independently saves 33.35%. Three clean
current-lineage long-prompt tasks reproduce every incumbent generated ID and
requested value. Under `auto`, partial and unqualified chunks retain exact
GEMV; `QWEN_DSV4_PACKED_Q8_OUTPUT=exact` is the rollback and `f32_matrix` is an
explicit full-chunk force mode. Post-route GPU is now the largest measured
span at 28.60 seconds, followed by attention at 16.21 seconds; output A/B has
fallen to 5.32 seconds.

The first GPU-resident MoE slice is now bounded. Generation-stamped GPU
routing and deterministic expert/token/original-slot compaction cover the 25
IQ2_XS/IQ2_XS/IQ3_XXS layers. The split pilot removes 1.152 seconds of sampled
host residual but adds 0.403 seconds of route/compaction and padded-dispatch GPU
work. Same-command router-to-expert execution preserves the complete candidate
logit vector, but a same-binary 8K comparison moves ordinary wall only
`61,405.218 -> 61,101.490 ms`, or 0.49%; its two-run brackets overlap. Keep
`QWEN_DSV4_PACKED_GPU_ROUTE_COMPACT` opt-in rather than paying a quality campaign
for near-tied learned-route rank changes. The mechanism is infrastructure, not
yet a product promotion.

Stable-route grouped all-IQ3 execution now defaults on independently. The
16-layer cohort falls `7,965.459 -> 4,748.927 ms`, sampled post-route GPU falls
`28,609.244 -> 25,593.623 ms`, and ordinary 8K wall falls
`61,448.693 -> 59,900.783 ms`, raising prefill from 133.31 to 136.76 token/s.
Complete logits and all generated IDs across the ledger, structured, and
multilingual discriminators remain exact. Rollback is
`QWEN_DSV4_PACKED_GROUPED_IQ3=0`.

Broad GPU ownership is not promotable yet. Routing all 41 grouped layers reaches
58,452.545 ms and 140.15 token/s, but changes the ledger total from 2,578 to
2,812. The 25-layer GPU-route predecessor and stable-route all-IQ3 schedule each
retain the exact answer, so the failure belongs to cumulative route ownership,
not grouped IQ3 arithmetic. `QWEN_DSV4_PACKED_GPU_ROUTE_COMPACT` keeps its
25-layer scope; `QWEN_DSV4_PACKED_GPU_ROUTE_IQ3=1` is an additional diagnostic
override only.

Grouped-head online dense attention now defaults on. One 256-thread group gives
each of eight SIMDgroups one head while staging sixteen shared F16 KV rows once.
It preserves packed raw-ring causality, raw-before-compressed order,
denominator-only sinks, and the promoted online HCA recurrence. Against the
same-binary control, attention core falls `12,434.453 -> 3,306.676 ms` and
ordinary 8K wall falls `58,608.999 -> 50,829.061 ms`, raising prefill from
139.77 to 161.17 token/s. HCA, SWA, and CSA core move
`6,286.562 -> 652.852`, `595.045 -> 52.263`, and
`5,552.847 -> 2,601.561 ms`. The kernel is bit-identical to the existing online
HCA lineage and stays below `6.87e-7` relative RMS versus cooperative attention.
The maintained 6,642-token product prompt keeps its top token and first 28
greedy IDs before legal schedule divergence; candidate and independent
llama_core continuations remain coherent three-point answers. This is numerical
and greedy-semantic authority, not incumbent bit identity. Rollback is
`QWEN_DSV4_PACKED_GROUP8_DENSE=0`.

Schema-v5 attribution splits the remaining 3.739-second sparse interval into
1.886 seconds of indexer preparation, 1.806 seconds of Lightning scoring, and
only 47.3 ms of exact radix selection. Selection is closed. The first indexer-Q
matrix candidate cuts preparation to 493.1 ms, but changed downstream routes
raise post-route GPU by about 0.70 seconds. Sampled wall improves only 311.5 ms
or 0.63%, while its ordinary wall lies inside the 1.31-second control drift
span. Remove the candidate and do not pay a quality campaign for this N=2,048
work unit. Reopen the matrix premise for larger chunks or an exact-order
shared-weight schedule that avoids route displacement.

The selected-CSA grouped-head transfer is also closed. Although its
sixteen-row shared staging is bit-identical to the promoted online recurrence,
the same-binary 8K comparison raises compressed-sparse attention core
`2,617.621 -> 3,035.176 ms`, or 15.95%. Ordinary wall rises
`49,461.968 -> 50,845.939 ms` and throughput falls
`165.62 -> 161.11 token/s`. Unlike contiguous dense history, the result is
consistent with repeated random-row loads already being cache-served: this
topology's synchronization and 16 KiB threadgroup allocation cost more than
shared staging saves. Remove the pilot. Reopen only for a barrier-free sharing
topology or evidence that selected-row device traffic is the limiter.

Packed Lightning score dispatch is now bounded by the chunk's final published
row rather than the session's physical capacity. The exact change preserves all
visible score bits, selected masks, IDs, counts, statuses, and complete 8K
logits. On the same binary, score GPU falls only
`1,816.993 -> 1,787.957 ms`, or 1.60%; ordinary request wall is
order-confounded and receives no speed claim. Default the deletion anyway: it
cannot remove a selector-visible row and prevents early chunks from launching
through unused 32K-to-1M request capacity. Rollback is
`QWEN_DSV4_PACKED_INDEXER_VISIBLE_DISPATCH=0`. The remaining 1.788-second score
producer stays open for a new arithmetic work unit.

That new work unit now defaults on. An 8-query by 32-row F32 matrix tile stages
each F16 K row once for eight neighboring queries while preserving F32 query,
weight, and accumulation values. A nine-query, 48-row differential crosses both
tile axes and preserves every visible score bit and selector result. Against the
same clean `ee5996f` binary, score GPU falls
`1,794.656 -> 1,354.564 ms`, or 24.52%; sampled and ordinary 8K wall save
593.568 and 474.723 ms respectively, and complete logits remain bit-identical.
Rollback is `QWEN_DSV4_PACKED_INDEXER_TILED_F32=0`. This closes the immediate
F32 scorer work unit; its remaining 1.355-second subtotal no longer outranks the
25.62-second routed stage.

The leading routed-expert premise now clears decisively. A four-SIMDgroup
64-output by 32-route IQ2_XS matrix work unit reuses the stable width-32 plan
while retaining F32 operands, accumulation, and exact destination ownership.
Against the same clean `2083d28` binary, gate/up falls
`14,834.930 -> 4,578.848 ms`, the complete IQ2 cohort falls
`18,093.042 -> 7,985.966 ms`, and post-route GPU falls
`25,630.046 -> 15,689.288 ms`. Ordinary 8K wall falls 16.00%, from
48,842.379 to 41,028.896 ms, raising prefill from 167.72 to 199.66 token/s;
sampled wall falls 19.28%. Complete logits remain bit-identical. The N=128
guardrail also wins despite only about 10% width-32 occupancy: sampled wall
falls 3.88% and gate/up falls 16.96%. Default the exact work-unit change with
`QWEN_DSV4_PACKED_IQ2_MM64X32=0` as rollback.

The changed expert work unit reopens larger chunks as composition rather than
padding reduction. On the same clean `76eebc2` binary, N=4,096 reduces canonical
8K ordinary wall `40,914.920 -> 39,933.627 ms`, raising prefill
`200.22 -> 205.14 token/s`; profiled wall falls 1,098.489 ms. Complete logits
remain bit-identical. Default N=4,096 with
`QWEN_DSV4_PREFILL_CHUNK_TOKENS=2048` as the prior-policy override.

The first N=4,096 pre-expert work-unit change also clears. Four SIMDgroups now
share each accepted F32 16x64 Q8 weight tile across 128 tokens for Q-B and
output A/B while preserving every result bit. Against a same-binary control,
before-attention GPU falls `6,992.796 -> 6,852.850 ms`, output projections fall
`5,526.999 -> 5,320.012 ms`, and pre-expert GPU falls
`21,295.579 -> 20,920.218 ms`. The repeat ordinary reading is
`39,004.478 -> 38,687.479 ms`, or `210.03 -> 211.75 token/s`; the first
candidate ordinary pass was a disclosed outlier, so endpoint wall is not
promotion authority. Both sampled candidate passes reproduce the GPU-stage
reduction. A cross-binary N=2,048 check saves 299.302 ms pre-expert with no
other active path change. Default the R2C4K64-bit-identical work unit with the
prior `f32_matrix` policies retained as independent Q-B and output rollback
controls.

The 4K work unit then changes the sparse-indexer matrix decision. On clean
`798a50c`, wide F32 Q8 projection reduces preparation
`1,897.744/1,924.206 -> 518.858 ms` across a control/candidate/control bracket.
Against the faster control, ordinary wall falls
`38,752.391 -> 37,040.317 ms`, or `211.39 -> 221.16 token/s`; sampled wall
falls `38,359.552 -> 36,840.900 ms`. Post-route GPU is noninferior. The
synthetic final vector changes, but a real 6,642-token prompt plus 80-token
continuation preserves every generated ID while exercising one matrix chunk and
one partial GEMV tail. Default only the pinned M4 Max/current-asset complete
N=4,096 profile, with `QWEN_DSV4_PACKED_INDEXER_Q_MATRIX=0` as rollback.
N=2,048 retains GEMV because its prior endpoint result remained unresolved.

The default cap must not greedily absorb an unqualified remainder. That policy
regressed the maintained 2,385-token prompt from about 110 to 70.98 token/s by
turning `2,048 + 337` into one 2,385-token fallback. Commit `eb3c22d` restores
hierarchical `4,096 -> 2,048 -> remainder` scheduling across ordinary,
resident, and snapshot paths. The 2,385-token prompt now reaches 160.33 token/s
and the 6,642-token prompt reaches 179.52 token/s, up from 119.81 after the 4K
indexer promotion. Future product claims must report actual chunk geometry;
exact 8K full-chunk cells cannot authorize arbitrary-tail throughput alone.

The grouped output-A dataflow then deletes all sixteen pack/scatter passes.
Against clean `3051b21`, output-projection GPU falls
`5,306.365 -> 4,455.770 ms`; sampled 8K wall falls
`36,598.057 -> 35,688.851 ms`, and ordinary prefill rises
`221.50 -> 227.80 token/s`. N=2,048 output GPU also falls
`1,301.486 -> 1,080.798 ms`, with ordinary prefill rising
`233.71 -> 238.85 token/s`. Both final-logit vectors remain bit-identical.
Default the exact-to-R2C16 strided group-axis work unit with
`QWEN_DSV4_PACKED_Q8_OUTPUT_GROUPED=0` as rollback.

The 4K result still closes capacity growth as the mechanism. The current
35.69-second sampled profile is led by 18.73 seconds pre-expert, split into 6.93
seconds before attention, 5.59 seconds in the attention body, 4.46 seconds in
output projections, and 1.75 seconds after attention. The attention body is
3.42 seconds core and 2.17 seconds sparse indexer/selection: 0.52 seconds
preparation, 1.61 seconds scoring, and 0.05 seconds selection. Inverse RoPE is
0.05 seconds. Post-route is 13.82 seconds and host residual is 3.14 seconds.

The one-row DwarfStar selected-CSA transfer is also closed. Sharing one F16 KV
row across 16 heads reduces scratch from the prior pilot's 16 KiB to 1 KiB and
preserves every incumbent result bit, but compressed-sparse attention core
regresses `2,688.779/2,674.337 -> 3,366.240 ms` in a clean
control/candidate/control bracket. The second failure with a different staging
granularity isolates cross-head synchronization rather than scratch capacity:
selected rows appear sufficiently cache-served that full-threadgroup barriers
and dual-head register pressure dominate. Remove the pilot and require a
barrier-free topology or changed traffic evidence before reopening.

Exact paired IQ2 gate/up is closed as well. Sharing one F32 activation tile,
removing both projection materializations, and folding SwiGLU into the existing
64x32 work unit preserves every bit but raises the combined target from
`4,076.981/3,994.295` to `4,207.414 ms`. Post-route GPU also regresses. The
20 KiB threadgroup allocation and sixteen simultaneous F32 accumulator lineages
erase the dataflow saving, matching the prior all-IQ3 fusion result under a
different quant format. Remove the pilot; same-tile dual-projection fusion now
requires a materially lower-pressure premise before reopening.

Half-staging the existing 64x32 IQ2 operands is the lower-pressure premise that
clears. It reduces threadgroup storage from 12 to 8 KiB while retaining F32
accumulators and output. Production-K direct error is bounded at `3.41e-4`
relative RMS for gate/up and `1.71e-3` after SwiGLU. In a clean 8K
control/candidate/control bracket, gate/up falls 6.89-8.00%, the complete BM16
cohort falls 5.02-6.43%, and ordinary wall falls 3.12-5.18%. A real 6,642-token
continuation remains coherent and reaches EOS. Default the faster arithmetic
with `QWEN_DSV4_PACKED_IQ2_F16_MATRIX=0` as the F32-staging rollback.

The K160 REAP lane supplies same-GGUF prefill and decode anchors. Mapped grouped
Q3_K/Q4_K matrices and shared/route overlap first moved native N=2,048 from
227.9 to 276.58 token/s. Current llama.cpp b10326 reaches 278.73 token/s at
pp512 on the exact GGUF, while qwen's old partial N=512 chunk collapsed to
18.428 token/s. Explicit residency plus the promoted matrix stack recovers
254.909 token/s at N=512; arbitrary edge-safe tails reach 215.34 at N=337 and
249.80 at N=498. On real 2,385/6,642-token prompts, current prefill is
297.27/274.78 token/s. Residency remains profitable after its load charge,
saving 0.382/1.819 seconds from load plus prefill, while one decode pair is flat
at 25.81 versus 26.01 token/s. Treat the result as a composed placement and
coverage repair, not a new arithmetic kernel ceiling.

The remaining N=128 K160 cliff is now repaired. A model-backed synthetic
median-class layer floor moves `32.925 -> 13.095 ms GPU` conservatively,
preserving output bits and projecting an illustrative 852.7 ms saving across 43
layers despite only 24.24% tile occupancy. The isolated production request
moves `2,394.366 -> 1,627.548 ms`, or `53.46 -> 78.65 token/s`, and both policies
emit the same 32-token greedy digest. Qualification adds exactly N=128 to the
existing authenticated M4 Max/K160/Q3_K-Q3_K-Q4_K scope; shared/route overlap
and N=129..255 remain on their old policies. Short-tail matrix coverage is no
longer a leading K160 opportunity.

The CSA/HCA leverage map is now regime-specific. At terminal singleton shape,
the production scorer costs 2.102 ms across each of 21 CSA layers, radix4 costs
1.875 ms on mixed scores, selected attention costs 0.323 ms, and online HCA
costs 3.410 ms across each of 20 HCA layers. Their subtotals are about
44.1/39.4/6.8/68.2 ms respectively. The exact multigroup selector lowers the
selection subtotal to about 14.2 ms where qualified, so "selection is closed"
remains true for 47 ms of packed 8K prefill but not for far singleton decode.

Four corrections constrain the new queue:

- Cache-order selected IDs are already strictly ascending. Sorting them again
  cannot improve locality; run/page descriptors remain a separate measured
  possibility.
- Packed terminal-capacity score and selected-mask scratch each occupy 4 GiB
  at N=4,096. Attention consumes compact IDs, counts, and status rather than the
  dense mask. Deleting materialization is a memory and dataflow opportunity.
- A simple previous-score delta certificate is not recursively exact. Skipped
  rows need conservative upper envelopes, newly published rows need exact
  scores, and signed weight changes need an outward-rounded bound containing
  both `|w| * ||delta q||` and `|delta w| * ||q||`, scaled by each published
  row norm. Lazy cumulative envelopes remain correct as slack grows, but
  periodic exact refresh may be needed for useful pruning.
- HCA is not two free GEMMs under the current contract. The 64-head logical
  reads, F32 queries, sink-aware softmax, output state, and merge lineage remain
  charged. Grouped split-K online attention is open, but a 0.1 ms matrix
  arithmetic floor is not a product projection.

The first exact deletion already clears. Direct per-lane row loads remove four
threadgroup stores, four reloads, and one barrier without changing the online
recurrence. HCA at 8,192 rows moves `3.406 -> 2.993 ms/layer`, saving about
8.3 ms across 20 layers; selected CSA moves `0.233 -> 0.200 ms/layer` and stays
flat from 16,384 through 262,144 history rows. Staged and direct outputs are
bit-identical across selected-CSA boundaries and HCA 512/513 through 8,192.
One same-binary warm K160 product pair moves the 2,385-token prompt from
13,249 to 13,111 ms (`180.01 -> 181.90 token/s`) with the same generated-ID
digest. Default the deletion with `QWEN_DSV4_ONLINE_DIRECT_LOAD=0` as rollback.

That direct-load result does not authorize the same online body for singleton
selected CSA. A default-off FRESH transfer alternates both paths from one
restored position-65,663 state. Paired command-GPU savings are
`1.288/-0.694/-1.757/0.210 ms`, median `-0.242 ms`, with only two of four wins
against a `2.0 ms` gate. Model-free online/direct numerical checks pass, but the
integrated packed-to-singleton projection does not. The cited
`0.233 -> 0.200 ms/layer` is staged-online to direct-online, not legacy
singleton to online-direct. Remove the arm and keep singleton CSA on the
cooperative body until a changed work unit establishes a new whole-token
ceiling.

The existing eight-head grouped online kernel also transfers exactly to long
singleton HCA. Direct/grouped timings are `0.326/0.324 ms` at 513 rows,
`0.790/0.668 ms` at 2,048 rows, and `2.994/2.542 ms` at 8,192 rows. Outputs are
bit-identical at 512, 513, 527, 528, 895, 896, 897, 2,048, 8,191, and 8,192
compressed rows. This removes another 0.452 ms/layer, or about 9.0 ms/token at
terminal depth. Default it with `QWEN_DSV4_GROUPED_LONG_HCA=0` as rollback.

Partitioning that grouped recurrence is the larger HCA move. Four/eight/sixteen
partitions measure `0.639/0.339/0.330 ms/layer` at 8,192 compressed rows versus
the 2.542 ms grouped incumbent. P8 is chosen because P16 buys only 0.009 ms
while doubling partial storage; P8 charges 1,052,672 scratch bytes and includes
the deterministic sink-aware reducer in every timing. It is repeat-bit stable
and remains inside the legacy envelope across 512 through 8,192 structural
rows. On the current asset at position 65,663 it saves 9.71 ms GPU and 9.24 ms
wall while preserving argmax and every consumed route/CSA decision. At restored
terminal state it saves 50.32-51.76 ms GPU and 50.19-52.26 ms wall per token,
moving warm terminal GPU to 136.78-137.23 ms. Default P8 with
`QWEN_DSV4_SPLITK_HCA=0` as rollback. HCA now contributes only about 6.8 ms at
terminal depth; close further local HCA topology work.
Automatic grouped and split-K HCA ownership is qualified to Apple M4 Max; other
Metal devices retain the direct online path until measured.

The exact 32-group selector also clears on the current asset with split-K HCA
active in both arms. At position 786,431 and 196,608 visible CSA rows, radix
controls are `125.048/123.507 ms GPU` and multigroup is `110.776 ms`, a
conservative 12.731 ms/token saving. Final logits, hidden state, causal state,
routes, and all CSA decisions are bit-identical; control drift is 1.24% GPU.
Default it on Apple M4 Max only when capacity is 196,608..262,144, visibility is
at least 196,608, and visibility occupies at least three quarters of capacity.
Packed and ineligible singleton paths remain radix4. Roll back with
`QWEN_DSV4_MULTIGROUP_SELECTOR=0` or the CLI `off` policy.

The decode-side F16-staged Lightning matrix scorer establishes a changed-work-unit
ceiling but is now product-KILLed. It retains the F16 K cache, stages only the 16
KiB query slab to F16, uses F32 head-weight reduction, and leaves selection
authoritative. At
16,384 visible rows it saves 2.966 ms GPU and 2.391 ms wall on the current
position-65,663 endpoint while preserving every consumed route/CSA ID and
bit-identical logits and causal state. At 262,144 rows, two current-asset
terminal packets save 33.355-34.087 ms GPU and 33.276-34.095 ms wall, moving
terminal GPU to 104.495-105.264 ms with bit-identical final logits. Those
integrated fixtures use synthetic zeroed long-history caches, so they establish
geometry timing but not representative rank-boundary quality. A real 97,040-token
snapshot replay supplies that missing evidence and fails decisively: the first
same-input selected-set change appears at position 97,042, layer 38, then routes,
tokens, logits, and causal state diverge. Across 65 transitions, 935/1,365 CSA
sets differ while both F32 controls remain repeat-bit exact. Keep the F32
singleton default and retain `QWEN_DSV4_LIGHTNING_F16_MATRIX=1` only as a
diagnostic ceiling. Direct authoritative F16 scoring is closed.

Packed selection no longer materializes its redundant public mask in ordinary
builds. Selected attention has always consumed only cache-order IDs, count, and
visibility; a dedicated maskless radix4 entry point now preserves exact ties,
fallback IDs, count, and status without binding a mask buffer. This removes
4,294,967,296 bytes from a full-context session and 12,582,912 bytes from the
maintained 2,385-token request-sized session. Diagnostics retain the mask for
FP4 comparisons. The packed score matrix remains the other 4 GiB half of this
interface.

Process-cold DeepSeek loading now uses the established Qwen parallel-pread
mechanism rather than charging demand faults to the first prefill. On the
current 97.05 GiB asset, cold load-plus-prefill falls 54.1% for the 11-token
endpoint and 36.4% for the 2,385-token endpoint as a phase subtotal; `load_ms`
already contains `prefetch_ms`. A full residency scan itself
cost 1.384 seconds at this scale, so DS4 uses a reusable bounded distributed
sample: a fully warm decision costs 3.9 ms and performs no reread. Default
`auto` uses a conservative 98% threshold; structured cold and prefix-warmed
fixtures keep the estimate within three percentage points of full `mincore`,
make the same policy decision, and bias ambiguous states toward warming.
`QWEN_DSV4_PREFETCH=off|always`
provides demand-page rollback and forced warmup. Loader first touch is no longer
an open DS4 optimization item unless direct destination population can beat the
6.2-6.8 GB/s pread path without changing retained topology.

K160 decode exposed a separate routed-expert work-unit defect. The first
correctness path issued gate, up, SwiGLU, and down serially for each of six
slots: 24 dispatches per layer, including eighteen quantized projections.
Exact all-slot Q3_K gate/up/SwiGLU plus Q4_K down reduces the production-shape
routed stage from 1.184 to 0.676
ms/layer. Reusing the engine's packed fast K-block arithmetic then lowers it to
0.236 ms/layer. On the same 2,385-token continuation, serial indexed, exact
all-slot, and fast all-slot decode are 11.26, 14.62, and 19.62-19.82 token/s.
The fast and exact paths emit the same 144-token digest. On the 632-token math
continuation, fast and exact all-slot reach 23.84 and 16.79 token/s and retain
the same complete digest and correct `001` result. Default the faster schedule
only for M4 Max, hidden 4,096, FFN 2,048, 160 experts, top-6, and the measured
Q3_K/Q3_K/Q4_K storage triple. Use `QWEN_DSV4_ALL_SLOTS_Q3Q4_FAST=0` as arithmetic rollback and
`QWEN_DSV4_ALL_SLOTS_Q3Q4=0` as serial rollback.

The reusable K160 prefill profile now splits routed gate/up, SwiGLU, and down at
`1,850.847/33.523/866.190 ms`. It also closes several attractive but incorrect
explanations for the llama.cpp lead. Parallel Q8 dequant is flat, a 64-row F32
Q8 tile regresses to 258.0 token/s, mapped Q3/Q4 control/scatter cleanup has no
routed-stage movement, literal llama.cpp Q3 dequant changes only 9 ms, and half
Q8 Q-B/output saves only 90 ms while changing logits. Four fully optimized
512-token chunks regress to 192.4 token/s as sparse expert occupancy and command
boundaries dominate. Do not reopen those local shapes without new hardware or
a different arithmetic/ownership premise.

The corrected Metal timeline narrows that premise. `trace-metal.py` previously
confused xctrace's duplicate duration and command-buffer-ID element types; it
now keys fields by schema mnemonic and coalesces overlapping interval unions
before computing idle gaps. Raw GPU work, union-busy time, and overlap remain
separate, so future concurrent queues cannot manufacture idle time. One native ordinary pass spans 128 command
buffers, 6,921.271 ms of GPU compute, and 267.638 ms of inter-command gap. One
llama.cpp full-ubatch pass spans two compute commands plus a blit, 6,202.558 ms
of compute, and effectively zero gap. An exact 32-row x 64-token F32 Q8 tile
then leaves before-attention/output flat at 1,790.863/1,186.187 ms and lowers
ordinary throughput to 267.39 token/s, so it is removed. Exact row-16, row-32,
and row-64 Q8 geometries are bounded; command ownership is worth hundreds of
milliseconds, not the whole external delta.

Shader-level attribution finds one concrete compute defect hidden inside that
aggregate. The 43 K160 F32 router projections consume `423.535 ms` across two
N=2,048 passes. Q3_K and Q4_K dispatches are already near llama.cpp's per-kernel
times; the router is not. A reassociated float4 E8 schedule cuts the router to
`53.024 ms` but changes route-weight bits and later IDs. An exact replacement
instead keeps each output's scalar K order while reusing activation loads across
eight experts; both reference and replacement use source-scoped Metal safe math.
Its 86 sampled intervals
total `70.000 ms`, an approximately 176.8 ms/pass sampled-family delta, and it is
bit-identical to the incumbent at N=1/32/128/2,048/4,096 and in the complete
K160 prefill digest. Default that exact path only for the qualified K160/M4
N=256..4,096 scope; roll back with
`QWEN_DSV4_PACKED_ROUTER_E8P32_STRICT=0`.

Late post-route-L to pre-expert-L+1 enqueue does not harvest command ownership.
It preserves all 128 commands on one serial queue, misses an A/B/A wall bracket,
and introduced a fail-stop lifetime hazard before removal. This KILL is narrow:
same-buffer graph ownership and designs that remove commands plus GPU work remain
open.

The direct skinny-matrix mHC premise is also closed. Replacing only the two
24x16,384 token-axis Q8 projections with the existing half-staged matrix kernel
saves about 51.9 ms across interpolated sampled stages, roughly 0.7% of the
N=2,048 request, while changing final logits. A future mHC candidate must delete
the 16,384-wide normalized representation and adjacent dispatches rather than
buying another matrix schedule.

That representation-deletion ceiling is measured but not certified. A clean
`700522e` C/A/P/Z campaign deletes only the 86 normalization/projection pairs
and preserves every endpoint bit. Its conservative wall CLEAR lower95 is
3.0033%, union-GPU lower95 is 2.3232%, and all six wall sextets exceed 2%, but
severe campaign-wide VM pressure makes the formal result HOLD.

The contributive ceiling funded one bounded exact producer falsifier rather than
another certification campaign. Scale-only RMS plus scale-aware Q8 is bitwise,
but grouping all 24 rows regresses N=2,048 wall from 7.106 to 17.596 seconds;
the sole eight-row refinement is flat to slightly slower against a following
incumbent bracket. Logical slab-traffic deletion is not the physical mechanism:
row grouping trades cache-friendly reads for register pressure and serial Q8
work. Close mHC until a producer shares or deletes dequant/dot arithmetic or an
adjacent consumer. Do not rerun the same ceiling, sweep row widths, or move the
RMS scale after the dot.

The larger token-axis defect does clear. Q-A and raw-KV were still rereading
their Q8 matrices independently for every prompt token while the neighboring
Q-B, output, shared, and compressor projections owned full-chunk matrices.
Reusing the accepted F32 R2C16 work unit for those 86 calls lowers sampled
before-attention `1,830.257 -> 1,264.883 ms` and ordinary N=2,048 wall
`7,825.465 -> 7,353.321 ms`. Real 2,385/6,642-token prefill improves 5.8/7.4%,
and an 8,470-token structured probe preserves every generated ID and exact JSON
while improving `224.68 -> 240.33 token/s`. The arithmetic schedule changes
route decisions at depth, so this is numerical authority backed by real prompt
behavior, not a bitwise claim. Default both projections only for the qualified
K160/M4 N=256..4,096 scope. Multiples of 128 retain R2C16; arbitrary tails use
the guarded R2C4 body, and N<256 stays exact.
`QWEN_DSV4_PACKED_Q8_QA_KV_MATRIX=0` is the rollback.

Changed-graph command ownership now closes as well. A force-only K160 arm gives
GPU route/compact and true active-tile Q3/Q4 indirect dispatch every advantage:
one command per layer, no CPU route, no exact-weight publication, and approximate
weights. A hot bracket is flat, while the most favorable separate trace saves
53.876 ms. Even crediting all 36.388 ms/pass of candidate GPU excess, all
18.710 ms/pass of remaining short gaps, and the complete 23.991 ms hot-control
spread reaches only 132.965 ms against the 135.190 ms 2% gate. Remove the force
path. Reopen ownership only with an independently bounded 30-50 ms deletion of
existing control GPU work; exact route weights, replay, pre-enqueue, indirect
counts, or fewer commands alone are not changed premises.

The count-aware GPU-route audit does not clear the next implementation bar.
The retained 25-layer/four-chunk pilot charges 180.875 ms to route/compact and
222.519 ms to fixed overlaunch. Metal compute encoders do support a GPU-written
indirect threadgroup count, but scaling route/compact to 40 K160 layers leaves
only about 195-222 ms against the trace's 267.638 ms idle ceiling even if every
empty launch disappears. That is at most roughly 3% request movement, while
partial expert tiles remain and learned-route rank order still changes. Hold
this lane until a different design removes more than command idle.

The temporal-certification oracle also closes before implementation. On one
65-state continuation after a 31,834-token real prompt, median selected-set
Jaccard is 0.5375 and the tight hindsight one-step bound still retains 57.25% of
rows. Charging periodic full refreshes yields 78.29/74.57/76.29/80.44/83.44/89.28%
work for cadences 2/4/8/16/32/64. Runtime head weights are 30.29% negative. This
cannot compete with the 24.6% F16 matrix work unit even before a runnable bound,
seed, compaction, and merge. Close the single uniform total-score delta premise;
row- or block-specific certificates require their own materially lower ceiling.

Exact score-to-ID streaming is also topology-KILLed. A correctness-green
128-group producer preserves the cooperative F32 score order and exact local
and global top-512 sets, but mixed terminal data takes 7.726 ms/layer versus
3.714/3.684 ms controls for the complete scorer plus multi-group selector. Even
all-tied data is slower before the mandatory merge. Mixed scores repeatedly
drive a serial shared-heap repair while 255 threads wait; ties expose the
remaining persistent-group, barrier, sort, and publication tax. Remove the
prototype and require a new nonuniform bound or representation premise before
reopening exact fusion.

That performance result does not make K160 quality-equivalent to the 256-expert
asset. The current small battery shows a correct but 23% longer math trace and a
coherent prose answer that misses a 180-word limit by about 95 words. Before
calling K160 a general product replacement, record one maintained
K160-versus-FRESH quality row and one full-defaults-versus-arithmetic-rollbacks
logit/top-1 audit. Memory, prefill, decode, and quality are separate axes.

The local K216 REAP checkpoint is a different compact asset, not a widening of
the K160 Q3_K/Q4_K lane. Its 89,060,075,612 tensor bytes retain the FRESH-style
25-layer IQ2 cohort, 16 all-IQ3 layers, and two MXFP4-down outliers. Runtime-count
grouped IQ2/IQ3 execution removes the compact per-bucket fallback: warm 2,385-token
prefill moves `89.62 -> 168.46-169.33 token/s`, while 64-token decode reaches
21.70 token/s; the 6,642-token row is 192.27/21.60 prefill/decode token/s. This
makes K216 performance-viable and leaves its asset-quality comparison open.
Model-wide residency then separates placement from arithmetic: a fully warm
2,385-token control reaches 168.37 token/s, while the same graph with five
resident allocations reaches 203.63 token/s. After the 566.9 ms load charge,
load plus prefill still saves 1.886 seconds. A short decode guard is flat at
27.80 versus 28.08 token/s with identical IDs, and one 6,642-token residency
run reaches 208.57 token/s. The measured placement benefit is exact to the M4
Max/E=216/source-byte identity, but whole-model residency now requires
`QWEN_DSV4_RESIDENCY_SET=1`. Eager residency cost remains file-state sensitive:
one partially cooled run charges 3.96 seconds to load and loses to its
post-treatment control, so do not claim a universal cache-warm subtotal win
from this promotion.

FRESH now completes the same placement policy for all three maintained 0731
assets without transferring their arithmetic policies. Its eleven allocations
cover three retained windows and eight fallbacks. On a fully warm 2,385-token
comparison, residency moves prefill `15,535.2 -> 12,489.7 ms`; after its
`1,145.5 ms` additional load wall, load plus prefill still saves 1.900 seconds.
One 6,642-token confirmation reaches 209.05 token/s after a 1.146-second load,
and a short decode guard is flat at 27.58 versus 27.80 token/s with identical
generated IDs. A partially cooled default instead moves a roughly 46-second
on-demand prefill stall into a roughly 53-second eager load. Treat that as
conditioning evidence, not a cold ratio: residency controls placement, while
prefetch and physical reads remain separate. The path remains exact to the M4
Max/E=256/1,328-tensor/104,202,502,492-byte identity, but now requires
`QWEN_DSV4_RESIDENCY_SET=1` and a lifecycle that cannot escalate directly to
SIGKILL.

K160's raw-KV matrix policy does not transfer directly. A default-off K216 arm
selects its 43 Q8_0 raw-KV projections while leaving all Q-A projections and the
337-token tail on the incumbent path. Fresh-child control/candidate/candidate/
control walls are `13,493.525/13,596.330/13,627.882/13,353.784 ms`; candidate
median regresses by `188.452 ms` or `1.40%`, and the changed N=2,048 chunk
regresses by `164.559 ms` or `2.11%`. Remove the arm and do not run the
preregistered FRESH follow-on. K160's isolated-arm materiality is not portable
authority; require a new asset-specific profile and changed work unit before
reopening raw-KV matrix ownership.

Shared-expert/CPU-route overlap is also asset-specific. A current N=2,048
profile prices `371.3 ms` of CPU route work, `365.756 ms` of shared-expert GPU,
and a `348.120 ms` per-layer sum-of-minima ceiling. The exact K160 schedule still
regresses K216's 2,385-token median by `433.235 ms` or `3.25%`; the changed full
chunk itself regresses `249.831 ms` or `3.23%`, with every logit bit preserved.
Remove the arm and do not retune enqueue timing. The same profile instead finds
`402.924 ms` of command-GPU excess in MXFP4 layers 26 and 42 relative to the
other-layer median. That is the next K216 work-unit target, not another overlap.

The cheapest causal MXFP4 candidate then fails. An exact 2D batched-GEMV body
reduces 24,576 row dispatches to 326 bucket dispatches while preserving all
25,165,824 threadgroups, dot products, and final-logit bits. K216 request median
regresses by `739.232 ms` or `5.50%`; the changed full chunk regresses by
`443.283 ms` or `5.69%`. The outlier is therefore not removable launch count.
Close batching and grid sweeps.

The changed-work premise clears. A four-SIMDgroup F32 64x32 tile decodes each
MXFP4 weight panel once for up to 32 route columns. Its real-bank two-layer GPU
floor falls `412.742 -> 50.152 ms`, or 87.85%. A stable K216 endpoint retry
moves the 2,385-token request median `14,036.665 -> 13,651.656 ms`, saving
`385.009 ms` or 2.74%; the N=2,048 chunk saves `303.150 ms` or 3.77%. Sampled
post-route GPU independently saves `341.322 ms`. On the official-chat prompt,
all 32 greedy IDs and the top-eight first-token order remain identical despite
expected downstream route and two cutoff-sensitive selected-set changes. Default
only for the authenticated M4 Max/K216/E=216/N=2,048 profile; retain scalar
buckets below 16 and `QWEN_DSV4_PACKED_MXFP4_MATRIX=0` rollback.

The same physical work unit now clears K216's complete N=4,096 cell. A
maximally fragmented all-216-expert floor over 24,576 routes per MXFP4 layer
moves GPU median `818.645 -> 93.280 ms`, conservatively deleting 723.470 ms.
On the real packed schedule, routed-expert and post-route GPU fall
`2,801.109 -> 2,141.642 ms` and `6,561.063 -> 5,915.977 ms`; sampled wall falls
637.605 ms. One ordinary reference reaches 268.79 token/s versus 257.15 for its
adjacent rollback, while other unsampled runs remain order/host-state
confounded. A 6,642-token official-chat rollback/default audit preserves all 64
greedy IDs. Extend only the exact K216 N=4,096 qualifier.

The final crossed MXFP4 width now clears independently on FRESH. At N=2,048,
the two changed layers move routed-expert GPU from
`545.777/547.867 ms` in bracketing controls to `217.289 ms`, conservatively
deleting 328.488 ms and clearing the frozen 300 ms gate. The real 2,385-token
official-chat control/candidate/control prefill is
`13,504.5/11,542.8/11,848.4 ms`; candidate beats the faster control by
305.6 ms while preserving the first-token top-eight order and all 32 greedy
IDs. Extend only the exact M4 Max/FRESH/E=256/N=2,048 qualifier. Full-width
MXFP4 coverage is now closed for FRESH and K216 without another tile search.

K216 partial-tail Q8 coverage now clears separately. Extending only the shared
compressor/shared/Q-B/output qualifier from complete chunks to N=256..4,096
moves isolated N=337 `4,373.684 -> 2,878.119 ms`, or
`77.01 -> 117.09 token/s`. Sampled pre-expert GPU falls from 2,230.149 to
670.104 ms. The real 2,385-token request reaches 233.77 token/s versus the
maintained 203.63 pre-change artifact, while the 6,642-token row stays flat at
212.32 versus 213.26 token/s and preserves all 64 generated IDs. Keep
K160-only Q-A/raw-KV excluded, N<256 exact, and every component rollback
independent. Routed experts now own 1,958.163 ms of the changed-tail profile.

The corresponding K216 IQ2 coverage repair clears the new leading pocket. The
authenticated asset now extends its existing half-staged 64x32 work unit to
N=256..4,096 without changing a kernel. At N=337, ordinary wall across the
control/candidate/control bracket is `2,853.157/1,909.523/2,902.541 ms`;
routed-expert GPU moves
`1,960.673/490.587/1,955.293 ms`, while pre-expert GPU remains flat. Direct
production-K error stays inside the promoted half-staged contract. Current
no-yearcore ring0 guards at 3,246/6,224 tokens reach 250.10/240.57 prefill
token/s over complete 32/64-token requests, with 22.19/21.38 decode token/s and
recorded generated-ID digests. No matched F32 rollback was acquired, so those
rows make no cross-policy output-equivalence claim.
That first widening remains exact to M4 Max, 1,328 tensors, 89,060,075,612
source bytes, E=216, and N=256..4,096; K160, crossed identities, and N<256 retain
their prior policies.

FRESH reproduces the same partial-IQ2 mechanism under its own exact identity.
At N=337, ordinary control/candidate/control wall moves
`4,574.917/3,537.230/4,623.495 ms`; routed-expert GPU moves
`2,149.805/549.527/2,249.189 ms`, while pre-expert GPU remains within the control
span. This removes 74.4-75.6% of the targeted stage and 22.7-23.5% of ordinary
wall without changing the kernel. A current no-yearcore 6,224-token ring0
request reaches 223.95 prefill and 21.85 decode token/s, stops at EOS after 54
tokens, and records a complete generated-ID digest. Keep this second widening
exact to M4 Max, 1,328 tensors, 104,202,502,492 source bytes, E=256, and
N=256..4,096. Partial IQ2 coverage is now closed for both supported IQ2 assets.

FRESH partial-Q8 coverage closes the remaining N=337 pre-expert cliff. Extending
only its already-promoted compressor/shared/Q-B/output qualifier to
N=256..4,096 moves ordinary control/candidate/control wall
`3,528.151/2,059.391/3,550.126 ms` and pre-expert GPU
`2,143.824/670.647/2,146.556 ms`; post-route GPU remains effectively flat. The
124 compressor matrix invocations are now present, while K160-only Q-A/raw-KV
and N=4,096-only indexer-Q remain unchanged. The no-yearcore 6,224-token ring0
guard retains the same EOS, 54 generated tokens, and generated-ID digest, but
its 4,096 + 2,048 + 80 decomposition does not exercise a newly qualified width
and carries no speed authority. Partial Q8 coverage is now closed for all three
authenticated assets; do not widen by architecture alone.

Fresh attribution then authorizes one exact transfer rather than a quant-wide
widening. FRESH layers 26 and 42 retain `525.089/511.884 ms` of routed-expert
GPU at N=4,096. The unchanged tile reduces them to `167.483/160.357 ms`, deleting
`709.132 ms` directly; post-route GPU independently saves `699.296 ms`. Ordinary
A/B/B/A medians move `19,318.509 -> 17,780.797 ms`, but broad overlapping spans
make that ratio corroborating only. An 8,429-token official-chat audit exercises
two qualified chunks plus a scalar tail and preserves all 64 greedy IDs and the
top-eight first-token order. Default only for the exact M4 Max/FRESH/E=256/N=4,096
tuple with the existing `=0` rollback.

The local DSpark census and external acceptance calibration are complete. On
the exact 2,385-token official-chat prompt, llama.cpp N1 at confidence 0.3 moves
`26.785 -> 30.367 token/s`, accepts 107 of 142 attempted drafts, preserves the
complete output digest, and realizes 1.718 useful tokens per target packet. A
5% qwen win therefore requires a fully charged packet at or below 61.09 ms.
Current qwen packed-prefill reuse is not that work unit: seven alternating
restored pairs at position 2,384 measure N2 at 272.155 ms wall / 254.399 ms GPU,
with 86 encoders and 2,994 dispatches. GPU work alone misses the packet budget
by 4.16x before target decisions, hidden capture, the three-stage drafter,
acceptance, or rollback. This KILLs packed-prefill reuse and command-only
wrapping, not DSpark or speculative decoding generally.

Singleton routed/shared dependency waves also close before a product packet.
One loaded K160 residency alternated the serial schedule and an exact wave
schedule from the same restored state. Overlapping routed Q3_K gate/up with
shared Q8 gate/up, then routed Q4_K down with shared Q8 down, saves only
`0.267 ms/token` paired-median command-GPU and `0.006 ms/token` wall against a
`1.0 ms` screening gate. All output and causal-state bits match. The 2.98 ms
shared stage is therefore not free overlap headroom: unchanged routed and shared
projections contend for the same execution resources, while added encoder
boundaries consume the residual. Reopen this pocket only with dispatch, byte, or
arithmetic deletion.

The exact A10B process-cold path has a bounded force-only rescue. The incumbent
needs 165.95 seconds to load and 111.76 seconds for a 13-token first prefill.
Authenticated W4 destination pread populates 77.02 GB in 9.82 seconds and cuts
that one-token process wall `280.58 -> 30.38 s`, with the same generated token.
Sustained inference still collapses without explicit placement. The historical
879-allocation residency result charged 59.45 seconds during load, then restored
222 ms prefill and 44.77 decode token/s while reproducing the known exact v0.538
128-token stream. Keep `QWEN_GGUF_PARALLEL_COPY=pread` unused: on A10B that force
selector implicitly creates the whole-model set. Ordinary loading remains the
safe path; residency wiring is closed rather than an open loader problem.

Force-ranked queue:

Compressor pairing also closes at accounting. The 124 projections already use
the promoted matrix path and one shared normalized input. Sixty-two KV/gate
pairs retain distinct weights, Q8 decode, dots, accumulations, and outputs; only
cached activation staging and dispatches are plainly shareable. No existing
profile attributes the roughly 147 ms required for 2% to removable pair work,
and exact shared-panel precedents are negative or about 1% whole-wall. Reopen
only with 30-50 ms of independently isolated GPU-work deletion beyond unchanged
projection arithmetic and a credible path to the complete wall bar.

1. **Demand a genuinely new fresh-decode work unit.** REAP quality governance is
   now recorded: the composed defaults-versus-rollbacks audit, prior K160
   constraint failure, and maintained v4.1 FRESH/K216/K160 rows are complete.
   Neutral retention is `24/24` for all three; misleading retention is `16/24`,
   `12/24`, and `12/24`, with only `14/24` FRESH/K216 and `10/24` FRESH/K160
   item-level agreement. This is a guardrail, not asset equivalence. Do not fund
   another launch/publication retune. The next implementation proposal must first
   show deletion of target evaluations, route assignments, weight traversal, or
   another material physical work unit with a credible `>=2 ms/token` ceiling.
2. **Keep current-asset routed gate/up closed.** After the MXFP4 repair, FRESH
   and K160 still spend about `1,833` and `1,877 ms` in routed gate/up, but neither
   profile exposes a broken layer or shared quant work unit. K160's six selected
   gate/up experts consume 41.25 MiB/layer of independent Q3_K bytes; matching
   formats share no masks, codes, scales, decode, or dots. The exact shared-input
   Q3 body is already slower than separate low-pressure packed arithmetic
   (`0.676` versus `0.236 ms/layer` for the routed stage), while exact paired IQ2
   and all-IQ3 analogues regress under accumulator pressure. Do not implement
   sequential fast-Q3 fusion, depth pairing, interleaved banks, paired staging,
   or tile retunes. Reopen only for fewer routed assignments or a changed
   model/asset representation that first demonstrates roughly 150-200 ms of
   deleted weight decode, dot work, or cohort-wide publication.
3. **Keep command ownership closed without a larger GPU-work deletion.** The
   removed merged-route arm already deletes CPU routing, publication, and 86 of
   129 commands. Its maximally favorable planning ceiling is 132.965 ms; adding
   perfect deletion of the current strict router projection raises that only to
   167.965 ms, below the 220 ms ambitious-oracle gate before charging an exact
   producer. Do not build sealed A/O/Z route replay. Reopen only when a
   producer/consumer fusion independently identifies at least another 53 ms of
   existing GPU work and preserves fail-stop mutation ownership.
4. **Hold qwen DSpark behind a materially new N2 work unit.** The Metal
   operating point is n=1, not the B200 n=3: the counterbalanced b10326 sweep
   measures 1.162x/1.103x/1.053x at n=1/2/3 with byte-identical greedy output
   at n=1 and n=3 and a reproducibly divergent stream at n=2. Reopen only when
   correct two-column causal execution plus measured resident drafter, hidden
   capture, acceptance, and transaction costs fit inside the re-priced 58.34 ms
   external complete-packet budget (73.4-89.0 ms against the qwen singleton
   anchors) on the maintained snapshot. Do not build for n=3, adapt the current
   packed path, or fund command collapse around its unchanged GPU work. The
   singleton front end (item 5) is upstream: closing the 54.4-versus-35.7
   ms/token decode gap is what moves the verifier floor inside the external
   budget. Evidence: 2026-08-12 calibration entry;
   `target/profiles/dspark-metal-nsweep-2026-08-12/`.
5. **Hold decode front-end fusion until physical work can be deleted.** The
   original census reconciles 2,146 dispatches/token. Grouped output-A removed
   301 and measured 1.50 ms/token; exact Q6/Q8 shared-expert fusion removed
   another 86 and measured 0.240 ms/token wall. Exact Q8 compressor-frontier
   fusion removed another 124 and measured 0.133 ms/transition in its
   128-transition product bracket. Paired Q-A/raw-KV projection, Q-LoRA/KV
   RMSNorm, and query/KV RoPE now remove another 129 nonzero-position dispatches,
   leaving 1,506 ordinary dispatches/token. Production-geometry intermediates and
   F16 publication are bit-exact. Two noisy short K160 brackets have pooled arm
   medians favoring the packet by 0.086 ms/token wall and 0.100 ms/token GPU. Two
   128-transition product brackets are independently positive; the conservative
   lower result saves 0.198 ms/transition (0.52%).
   The narrow KV RoPE+F16 publication leaf remains KILL: it deletes 43
   dispatches but regresses the 128-token product bracket by about 0.051 ms/token
   and broadens transient F32 KV bits under fast-math. Launch-only mHC cleanup is
   now closed. Narrow combine-through-post was killed on a calibrated
   `0.09-0.24 ms/token` ceiling. A changed-premise controls+collapse+consumer-norm
   kernel preserved every authoritative bit and could remove 172 dispatches, but
   its production-width isolated bracket averaged 0.035193 versus 0.032028
   ms/site, a projected `0.272 ms/token` regression; no code remains. The changed-
   premise routed-down screen is now closed too. At production K160 Q4_K
   `2048 -> 4096`, an exact final-only producer changed down/weighted-sum/add from
   three dispatches to one and removed the `6H` expert plus routed `H` F32
   publications. Two 256-repeat A/B/B/A screens saved only `0.001733` and
   `0.000113 ms/site`, projecting to `0.074533` and `0.004847 ms/token` against
   the `0.4 ms/token` gate. No code remains. Do not transfer final-only ownership
   to narrower dtype cohorts or reopen mHC launch grouping. The current-asset
   gate/up audit in item 2 finds no shared dequant/dot work and closes that branch
   before code. Return program priority to item 1's structural work-unit search;
   do not resume isolated local retunes until a structural deletion has a credible
   2 ms/token ceiling. Every retained fusion also lowers the DSpark N=2 verifier
   denominator toward its 58.34 ms external budget. Design:
   `docs/bench/2026-08-12-dsv4-decode-dispatch-census/README.md`.
6. **Keep whole-model residency work closed.** Do not call `requestResidency`,
   pre-wire or `mlock` model destinations, experiment with uncached source reads
   around a residency set, or use the A10B force selector under current OpenCode
   supervision. The lease, poison gate, and cooperative teardown contain repeat
   damage but cannot make SIGKILL safe or prove host unwiring.
7. **Keep direct low-precision scoring closed.** Unguarded FP4 changed every
   deep mask and regressed wall; F16 query staging now fails on real nonzero
   history. Reopen low precision only as an F32-authoritative conservative
   filter whose exact refinement, fallback, and merge beat the F32 streaming
   path all-in. Candidate margin or recall alone is not a certificate.

Short-context local tuning is bounded-KILL under the current 0.75 ms/token
two-depth gate: all-slot barrier removal, larger IQ2 row groups, fixed-geometry
metadata specialization, and four-lane IQ2 decode all miss or regress. The
row-513 repair is a separate structural complexity guard outside the original
contexts 128/512; it does not reopen those sub-threshold kernel sweeps. Reopen
local tuning only for a new structural work reduction with a credible
>=2 ms/token ceiling, or a measured candidate that clears 0.75 ms at both
contexts. The retained nonzero expert profiler reports gate/up and down
separately so a future reopen starts from attribution rather than another shape
sweep.

`llama-bench --n-depth` is not a free decode-only long-context comparator: its
first repetition executes the complete depth prompt and serializes the state.
Do not pay a 32K cold prefix merely to populate a scoreboard; first establish a
reusable prepared state or make the cross-engine depth curve the actual gating
uncertainty.

## Latest Baseline Snapshot

M4 Max, release `qwen-bench`, clean narrow family spot after `v0.344` against
fresh llama.cpp b9833 (`c818263f2`, `MTL,BLAS`). No recorded thermal or
performance warnings. Artifact:
`docs/bench/2026-06-28-1551-v0345-fresh-lcpp-c818-family/README.md`.

| Model | `pp512` qwen/lcpp | `pp4096` qwen/lcpp | `tg128` qwen/lcpp | Notes |
| --- | ---: | ---: | ---: | --- |
| 0.8B dense | `1.06x` | `1.07x` | `1.41x` | runs=1 spot |
| 2B dense | `1.01x` | `1.05x` | `1.20x` | runs=1 spot |
| 4B dense | `1.04x` | `1.05x` | `1.20x` | runs=1 spot |
| 9B dense | `1.01x` | `1.01x` | `1.07x` | runs=1 spot |
| 27B dense | `1.05x` | `1.15x` | `1.26x` | runs=1 spot |
| 35B A3B | `1.07x` | `1.19x` | `1.42x` | v0.344 Q5 K512 R2 included |
| 122B A10B | `1.02x` | `1.15x` | `1.25x` | runs=1 spot |

Read: this board is a warm prompt-throughput and decode regression guard, not a
fresh-TTFT scoreboard. Default `qwen-bench pp` warms the model, excludes session
and scratch allocation, normally skips the final tail, and does not measure first
token delivery. Do not infer product responsiveness from `pp<N>` alone.

## BS=1 Objective Reset — 2026-07-10

The provenance/correctness audit found and fixed real defects, but it is no longer
the optimization strategy. v0.546 completes the first demand-side correction:
both CLI generation loops now record or flush each selected token before its
target transition, and they skip EOS and output-limit terminal transitions.

The output token stream remains greedy semantic exact, but terminal model state
differs: the last emitted token remains unconsumed because no successor logits are
requested. Current prefix snapshots are taken during prompt prefill and are not
affected. Any future resumable or end-of-request snapshot must record the pending
terminal token explicitly. Stop-set expansion beyond the current single EOS and
CPU/GPU argmax tie semantics remain separate work.

v0.547 adds the first bounded contract row. The production single-turn path now
records one first-post-model-load request from prompt acquisition through first
stdout flush and final newline flush. It separates tokenizer construction,
tokenization, validation, request allocations, prefill, first-token selection,
callback delivery, transitions, and total wall. Device allocation is sampled at
named milestones; it is not a peak-memory measurement.

v0.548 adds one opt-in identical warm follow-up in the same process. It recreates
prompt acquisition, tokenization, scratch, sequence, prefill, and generation while
retaining the loaded model, tokenizer, and unavoidable process/runtime warm state.
The pair is an aggregate first-position discriminator, not a tokenizer-only A/B.
The first 0.8B one-token-prompt sentinel shows a `93.28 ms` TTFT gap: explicitly
bracketed tokenizer construction accounts for `70.52 ms`, while prefill moves
`35.44 -> 12.64 ms`. This is a causal search-space split, not a promotion.

v0.549 directly prices PSO misses. On the same 0.8B sentinel, request-0 prefill
has 22 misses costing `1.763 ms`, of which `1.686 ms` is the Metal compiler API.
Generation has 11 misses costing `0.376 ms`; the warm request has none. PSO work
explains only `6.62%` of the `26.65 ms` prefill gap and at most `1.60%` of first
TTFT. Close PSO prewarming or compiler optimization as a material lever in this
cell. Do not relabel the unexplained `24.89 ms` first-prefill gap as PSO.

v0.550 removes eager decoded-byte materialization for every tokenizer vocabulary
entry. Exact `OnceLock` memoization decodes only token IDs actually emitted. In 10
alternating fresh-process eager/lazy pairs, 0.8B tokenizer construction improves
`69.017 -> 36.847 ms`, first TTFT `105.135 -> 72.705 ms`, and total request wall
`114.866 -> 82.273 ms`. Process-load-plus-TTFT improves `33.191 ms`; this is work
removal, not readiness-boundary movement. Warm TTFT is neutral within `0.038 ms`,
and first-callback p95 regresses only `0.0072 ms`.

v0.551 removes transient copies of GGUF token and merge strings during tokenizer
construction. Ten alternating fresh-process pairs improve construction
`36.998 -> 25.492 ms`, first TTFT `72.212 -> 62.142 ms`, and total request wall
`81.748 -> 71.897 ms`. Warm TTFT moves only `0.009 ms`. The tokenizer remains
fully owned; borrowed metadata exists only while constructing validated maps.

v0.552 reuses one merged-token lookup `String` instead of allocating once per BPE
merge. Ten alternating pairs improve construction `25.580 -> 17.078 ms`, first
TTFT `62.636 -> 52.478 ms`, and total request `72.203 -> 61.756 ms`; warm TTFT
moves only `0.028 ms`. Across the three exact tokenizer packets, approximate
median movement is `69.017 -> 17.078 ms` construction, `105.135 -> 52.478 ms`
first TTFT, and `114.866 -> 61.756 ms` total wall. Close cheap tokenizer
micro-construction. A packed owned vocabulary arena remains a representation
redesign, not authorization for another local allocation chain.

v0.553 closes the tokenizer lane on realistic prompt guardrails. Against the
v0.549 eager baseline, the final stack removes `52.283 ms` TTFT at 437 tokens
(`147.370 -> 95.087 ms`) and `55.542 ms` at 7,986 tokens
(`1130.679 -> 1075.137 ms`). Warm TTFT is neutral within `0.26 ms`, and all
numbered eager/final outputs match. The absolute fixed-cost win generalizes; its
percentage correctly falls from `35.48%` interactive to `4.91%` long-prompt.

v0.554 corrects the physical-N8 oracle's terminal state. A terminal-limited
packet now executes only the target transitions required to validate the final
pending token, using effective N inside physical-N8 scratch. The harness records
target transitions and final N, reconstructs an outside-timing serial state,
compares committed KV numerically, and validates one-token continuation logits.
A 27B eight-token smoke passes `7/7` transitions, effective N7, equal positions,
KV cosine `0.9999999124`, exact continuation argmax, and continuation cosine
`0.9999999994`. This validates the harness, not verifier economics.

v0.556 prices that denominator and splits the family by architecture. The frozen
437-token tokenizer prefix is not a decode fixture: both models emit EOS first,
so its zero-transition rows carry no verifier evidence. A frozen, complete
418-token Qwen chat replacement emits 128 tokens. Dense 27B passes every stream,
transition, terminal, and numerical resume gate in seven fresh processes. Its
decode-only oracle is `2.602307x` median (`2.462549-2.613965x`) with a bootstrap
one-sided 95% lower bound of `2.591826x`. A separate seven-row 12-token code
guardrail is `2.630660x` median. The 418-token median serial transition is
`38.619 ms`; the physical-N8 verifier packet is `117.633 ms`, so a charged packet
costs about `3.047` serial transitions. The zero-overhead, zero-abstention
necessary floor for `1.10x` is about `3.35` emitted tokens per charged packet;
real policies need more after fallback, proposer, correction, and restore costs.

A3B does not clear the corrected state contract. The complete chat fixture
diverges from the serial greedy stream. A 12-token code prompt keeps the observed
128-token stream and next argmax but fails resume numerics by token 16 and again
at token 128; the latter reaches KV cosine `0.999698864`, continuation-logit
cosine `0.999197009`, and maximum logit delta `0.437913`. Kill the current
packed-MoE/GDN physical-N8 implementation in the numerical/exact lane. This does
not prove that every state-preserving A3B verifier organization is impossible.
Do not invent a weaker contract merely to retain this implementation.

v0.559 clears the next dense-27B semantic gate. A causal offline PLD simulator
freezes one development-selected policy: prompt plus committed-output sources,
most-recent `L=8` match, D7/physical N8. On held-out mechanism rows, optimistic
decode is `2.4185x` exact quotation, `2.1595x` periodic output, and `2.0884x`
ambiguous repeated prefix; generic prose abstains on every transition at
`1.0000x` before production lookup cost. Favorable rows have `2.10-2.42 s` of
extra wall budget before missing `1.10x`. This promotes one target-only charged
implementation, not a prevalence claim or broad product default.

v0.562 clears charged target-only execution. Clean dense-27B decode improves
`2.5503x` quotation, `2.0470x` periodic output, and `2.2891x` ambiguous repeated
prefix. The adversarial row executes a real partial restore. Generic prose is
`0.99944x` with zero attempts, verifies, or restores. Every row preserves the
exact stream and numerical terminal resume gate. The fixed reference-first order
means prefill and total ratios are not promotion evidence; use this as decisive
mechanism proof, then counterbalance the product packet.

Next contract work:

1. Harden the live path with optional event traces, every output-limit/stop
   terminal width, exact offline event parity, and a longer continuation witness.
2. Integrate the same target-only loop behind a default-off product rollback flag.
   Preserve callback, sequence, prefix-cache, stop, and output-limit semantics.
3. Run counterbalanced fresh product packets. Require favorable median decode
   `>=1.50x`, generic regression `<=1%`, and TTFT regression no worse than 1% or
   1 ms. Keep index construction separately visible.
4. Keep the current A3B physical-N8 path closed unless a materially different
   verifier reproduces serial recurrent state or an explicitly approximate lane
   first demonstrates enough upside to justify quality validation.
5. Separate work removal from boundary movement: constructing the tokenizer or
   warming pipelines before declaring model-ready improves TTFT but not process-
   cold first flush unless the underlying work also becomes cheaper.
6. Keep process-cold load/residency and warm `pp<N>` throughput as separate rows.

This is a product correction plus a bounded objective measurement, not a return to
provenance-first work. Build identity and correctness gates remain guardrails.

## Hardware-Saturation Recalibration (2026-07-08)

External audit + cx review after v0.526 initially moved attention/DFlash dataflow
to the top. v0.527/v0.528 then found and fixed the stale DFlash drafter buckets:
phase 1/2 were still tuple-at-a-time, and phase 3 attention still recomputed QK
three times. v0.529 then split phase 3 and falsified O/FFN as the long-context
target: at ctx7561, phase3 is `622.6 ms`, of which SWA attention is `285.0 ms`,
full attention is `234.1 ms`, and all Q8 O/FFN projections together are only
`101.1 ms` (`2.7%` of static decode). Exact SWA scan pruning is default-on and
improves that row `3787.4 -> 3727.3 ms` decode-only, but DFlash remains governed
by acceptance and verify cost.

v0.531 closes the direct skinny-MMA `attn_v4` Phase-A shape: the exact G8/tile4/
C64 sidecar passed its oracles but regressed A3B ctx16384 main
`0.1224 -> 0.1393 ms/layer`. Do not extend it to G16/C128 without a materially
different reuse plan. v0.532 also closes MTP N8 R2 row-wave scheduling: exact
output held, but verifier time improved only `2.6 ms`, far below its `10 ms`
short-row gate. v0.533 closes duplicated L2 normalization inside each GDN state
row: it improves verifier-shaped N8 replay `7.1%`, below both integration gates.
v0.535 closes full-cost DFlash policy probing for 64-token product windows: a
zero-accept Spec16 probe at ctx3440 costs `414.0 ms` (`10.05` Off steps), and
immediate terminal fallback still reaches only `0.894x` the no-spec comparator.
The corrected comparator makes the loss larger, not smaller. Keep cheap
non-speculative predictors conceptually open, but do not spend another corpus on
an online Spec16 probe.
v0.537 then cuts the exact A3B MTP routed-bank residency `3.0 GiB -> 464 MiB` and
isolated MTP body time by about `37%` using existing native kernels. The path is
supported behind `QWEN_MTP_MOE_NATIVE_BANKS=all`, but stays opt-in: tok128 paired
draft savings are only `16.7-17.7%`, and paired speculative-decode wins
`2.19%/3.33%/2.99%` have a strict `2.99%` median. Both default gates miss.
v0.538 adds bit-exact Q4_K/Q8_0 token-embedding row lookup behind
`QWEN_NATIVE_QUANT_EMBED=1`. It saves exactly `4,370,432,000` bytes on 27B,
`1,493,893,120` on A3B, and `2,240,839,680` on A10B versus F32 token-embedding
residency. Exact 128-token streams, Q6_K fallback, and serial MTP/DFlash checks
pass. This remains memory leverage, not a throughput promotion: the post-hardening
suite is near parity, and clean 27B `tg128=0.98895x` misses the `0.99` floor.
v0.539 then proves the true Q4_K A3B draft-head primitive but kills integration.
Seven N1 heads improve to `0.72187x` Q6_K time, yet fixed body/orchestration cost
limits paired tok16/tok128 draft wins to `10.05%/15.20%`, decode wins to
`0.89%/2.28%`, and total wins to `0.85%/2.07%`. All semantic gates pass, but all
wall gates miss and streamed setup takes `13.079 s`. No experiment code remains.
v0.541 executes and closes the one authorized DFlash full-attention attempt
positively. The fixed N16 Q32/KV8 GQA-4 split-4 path clears the primitive gate
at all three preregistered contexts and improves the valid canonical Reva
static-decode wall `10,675.7/10,671.9 -> 9,975.8 ms`. The conservative result is
`6.522%`, with exact 256-token target-greedy equivalence and every acceptance
and call count unchanged. Default the path only for the measured full-layer
shape and `ctx_len=7986..8241`, with
`QWEN_DFLASH_ATTN_FULL_GQA_SPLIT4=0` rollback. This is a bounded static-decode
promotion, not a DFlash policy, total-request, SWA, verifier, or general
attention claim. Candidate 2 remains unrun; no split, context, prompt, or model
widening is authorized.
v0.542 then executes and kills the allowed different-layout Q8_0 KV branch. The
payload/scale split-plane reader is exact versus current Q8 and passes both F16
correctness gates. At the protocol-valid 32K row it regresses `9.95%` main and
`6.75%` main+reduce versus the slower F16 anchor. The 8K row is excluded because
anchor spread was `1.0635x`, although its direction was uniformly negative. All
experiment source is removed. Exact Q8_0 at the unchanged 272-byte head-row is
now closed in the two tested reader-layout families, not for every possible
compressed-KV format or attention body.
v0.543 executes and kills the next authorized lower-byte branch before timing.
Direct interleaved canonical GGML Q4_0 at 144 bytes per head-row reconstructs
bit-exactly in the standalone GPU format/indexing oracle and exercises both
scale signs. The real A3B block-3 `ctx8192` attention output nevertheless reaches
only `0.996111664` cosine with `0.1653642654` maximum absolute error, failing both
fidelity gates. The required 32K correctness row and every performance sample
remain unrun after the mandatory stop. All experiment source is removed.
v0.544 executes and kills the measurement-gated BF16 branch without source
changes. Current-HEAD warmed A3B BF16 `pp512` is only about 2% behind pinned
llama.cpp (`0.9785/0.9827/0.9816x`). The combined causal no-op ceiling is only
`0.7643-0.7658`, failing the `>=90%` arm. No-FFN and no-routed clear the `>=40%`
trace-entry threshold, but pinned upstream llama.cpp emits zero Metal operation
profile records. The required same-operation/shape mapping and `>=1.5x` ratio
are unavailable; preregistered missing-data handling makes this KILL, not a
rescue rerun. Preserve the small warmed deficit as a guardrail only.
v0.545 executes and closes the Mei-medium DFlash preflight at its first valid A1.
The canonical 23,122-token row emits 256 tokens with exact target-greedy
equivalence, but acceptance is only `159/96`, or `alpha_chain=1.65625`, and mean
emitted per step is `2.65625`. This fails the preregistered `alpha_chain >= 3.0`
candidate-authorization gate. Static16 decode is `39,635.3 ms` versus the
embedded `11,927.8 ms` DFlash-off comparator (`0.301x`); that comparison is
corroborating economics, not a formal non-regression gate. P/A2 remain unrun
because paid-cost attribution cannot repair the prerequisite acceptance miss.
Close this sequence with no policy, prompt, floor, split, profile, or anchor
rescue. The narrow v0.541 Reva kernel result remains intact.
v0.575 then lands the immutable true-long capture seam and clean 32K/131K
artifacts. Independent Float64 replay clears every captured 131K query/head with
minimum cosine `0.9999993066`. v0.576 uses the same artifact to separate mechanism
classes. Joint K/V with symmetric 16-value Q8 groups is locally green at `0.5625x`
F16 bytes, but this is fidelity evidence only; conventional same-body Q8 remains
performance-closed. Sparse retention is killed for implementation on this fixture:
the exact-score group-shared 50% oracle is already yellow, chunk64 is red, and the
per-head ideal needs about 89-93% physical GQA union. Promote one materially
different dual-format register-light body primitive, not another reader/layout.
v0.577-v0.580 then execute and kill the first such body at 32K. The scalar
cross-simdgroup design is exact with F16 and fidelity-green with group16-scale Q8,
but loses `2.866x/3.537x` main versus the slower V4 anchor. Q8 is `1.234x` slower
than the F16 control despite `0.5625x` bytes. This closes the concrete scalar QK
reduction and loader pairing, not distributed output ownership. A matrix reopen
must first clear a decode-and-stage floor below `0.15587 ms`; no 131K row or
production integration is authorized.
v0.581 clears that fixed floor at `0.09958353 ms` with exact checksums and immutable
capture hashes. This is `36.11%` below the historical ceiling, but the current 15%
target against v0.580 anchors is tighter at `0.15172507 ms`. The floor omits QK,
softmax, V accumulation, residency effects, and production partial writes. It
therefore authorizes exactly one low-confidence integrated body, not a matrix win.
v0.583 executes and kills that body on the exact captured 32K tail query. Main is
`0.313416 ms` versus stable `0.163084/0.162792 ms` V4 anchors and a same-packet
`0.138374 ms` gate. Decoded-Q8 partial and final correctness are effectively exact,
and captured-F16 fidelity is green. v0.584's one diagnostic-only limiter capture
shows low external-memory pressure, under-target occupancy, and strong
integer/conditional demand. Close this fixed matrix point and all protocol-barred
local rescues; skip the survival-conditional true-query/cache-append step and
remove experiment source in v0.585. This is not a universal claim against matrix
attention.
v0.586 then closes the generic two-class decode-storage oracle at synthesis
preflight. Routed Q5 and attention KV share no ABI or mechanism, and their
`282-296 GB/s` logical-byte proxies do not identify one physical-storage limiter.
The old no-weight result is not a formal current-R2 bound because v0.344 changed
K512 ownership, but no named current Q5 representation or attention body clears
its charged breadth gate. Do not build decomposition diagnostics without a
candidate whose implementation decision they can change.

v0.587-v0.588 close fixed committed-tail MTP history on the first conjunctive
fixture. Clean full-history controls repeat at 59 physical-N8 packets and
`(G-1)/packets=4.322034`; K256/K128/K64/K32 need 74/77/76/79 packets and fall to
`3.445946/3.311688/3.355263/3.227848`. The accepted-prefix histogram shows old
committed context supports recursive-chain persistence rather than adding dead
attention work. Stop before narrative, K interpolation, or packed tail-history
construction. Preserve the terminal-state correction and acceptance diagnostics;
remove the dead read-window source. Full-history decode is `1.24-1.28x`, but its
serial MTP prompt construction makes the measured request only `0.310-0.315x`.
Any renewed dense-MTP program therefore needs both a step-change asset and packed
full-history prompt construction, priced together rather than as separate wins.

## Force-Ranked BS=1 Opportunity Frontier

2026-09-07 dense Qwen3.8/Q8 native-embedding force-path qualification separates
bytes from latency. Test-only `6e36f048` proves exact 3,734,732,800-byte Metal allocation
removal plus bitwise gathered rows, prompt/decode logits and KV/GDN state through
64 greedy tokens. The independent ordinary-CLI output1/128 A-B-B-A reproduces
the bytes in every child but misses latency gates: output1 first stdout saves
6.543% against 15%; output128 process wall saves 1.017% against 5%, with first-output
control spread 6.118%. Loaded generation guard passes. No default expansion or
same-cell repeat; native `1` remains an existing explicit resource option, not a
qualified dense-Q8 latency default. Bound MTP is not executed by these witnesses.
The remaining 3.2-3.6 s runtime/load interval needs source/phase decomposition before
another copy-scheduling proposal; the ledger does not yet price its serial copy
fraction. Existing PSO miss cost is about 1-1.7 ms, not a generic optimization lever.
Evidence: `docs/bench/2026-09-07-native-q8-embedding/RESULT.md` and
`docs/bench/2026-09-07-native-q8-cli/RESULT.md`.

The active queue is not empty. The exhausted neighborhood is narrower: serial
N=1, same-model, same-graph, current-layout local retuning. The active frontier is
work removal, intra-request target-step amortization, and structurally different
prefill or long-attention work units.

Gain bands below are whole-phase estimates, not isolated-kernel ratios. Unknown
bands remain unknown until a costed oracle establishes them.

Banked exact process-cold cleanup: converted-F32 Qwen tensors now dequantize
directly into their final Shared Metal allocation. A 256 MiB floor moves
`37.855 -> 25.714 ms`, saves 12.261 ms paired with 6/6 wins, and deletes one
complete 256 MiB host representation. Scale the byte/RSS benefit to actual
fallback tensors, but do not project decode or native-quant gains: hot native
weights bypass this path. Host zero deletion alone is flat, and anonymous
no-copy output backing saves only 0.238 ms at 512 MiB / 1.367 ms at 2 GiB with
4/6 wins. Keep direct destination population; close a broad scratch allocator
without a changed physical premise.

Completed v0.546 removes the token-0 transition from TTFT and the unused terminal
transition from total request wall. For base TTFT `B` and removed transition `D`,
speedup is `(B + D) / B`.

Active narrow product completion: v0.567 measures fresh 11,287-token TTFT gains
of `1.0641x` A3B and `1.2075x` A10B behind opt-in `--prefill-chunk auto`.
v0.568 caps matrix query scratch at 1024 and overlaps phase-disjoint packs,
reducing incremental allocation to about 90 MB/877 MB with measured prefill
overhead `0.391%/0.109%`. v0.600 completes the exact product path: complete
eager/deferred Metal pricing, query cap and overlay, sequence-plus-transient
reserve, cache-safe/numeric fallback, schema-5 telemetry, and no post-admission
retry. It remains opt-in because both canonical attempts stop before their first
child on the same zero-global-Pageouts validity predicate. P2 observes only 354
pages (`5.53 MiB`) while swap occupancy, Swapouts, and Compressions are unchanged
and memory remains 96% available. This is a protocol closure, not candidate
evidence. No v0.600 P3 follows. v0.572 remains decisive against 32K expansion:
A3B chunk 2048 fails its pair floor at `1.04654x`; A10B narrowly reaches
`1.05065x`, 4096 adds only `1.01338x`, and the real 25,610-token guard is only
directional warmed `1.03314x`. Do not reopen 32K, new widths, sibling assets, or
same-cell retuning.

v0.601 is the terminal confirmation. Its complete A3B subset clears every speed
gate at median `1.052773x`, AB/BA `1.053020/1.052547x`, and 4/4 wins, but this is
informative non-authority evidence because A10B stops before its first child. The
77.03 GB conditioning interval records 76 system-wide Compressions with zero
Swapouts/swap-occupancy growth and 96% memory availability. Per the frozen rule,
the packet is inconclusive and no successor follows. Both exact profiles remain
opt-in. Stop confirmation work and advance the cold-first queue.

Blocked model-choice lane, not active queue: v0.573 screens seven local Q4 assets
through a frozen six-task direct-mode contract. Scores range from `1/6` to `3/6`;
none clears every category, so no performance or lower-quant row runs. The
model-choice `1.5x+` prior remains potentially large but unpriced for this contract.
Do not build a larger harness or rescue prompts now. Reopen only for a new declared
capability contract or asset with a concrete reason to change all-category passage.

Closed routed-tail lane, not active queue: v0.574 shows that eliminating grouped
`inner` requires unavailable cross-threadgroup producer/consumer synchronization or
collapsed output parallelism, while eliminating grouped `out` requires token-owned
down that loses expert reuse or the already-negative atomic path. Ordered phases and
bounded partials interpolate between those failures. Reopen only for cooperative
grid synchronization, certified large exact sparsity, or an ABI that removes a
complete bank pass without rereads or serialization.

Closed matrix-attention lane, not active queue: v0.583 is correct but reaches
`0.313416 ms`, `1.92525x` the faster V4 anchor and `2.26500x` its current 15%
gate. v0.584 classifies a mixed under-target-occupancy and instruction-heavy
failure at only `140.236 GB/s` external bandwidth. Do not sweep or rescue shape,
split, barriers, staging ownership, launch, or format. Reopen only for a mechanism
that changes the ownership/utilization premise and earns a new zero-cost ceiling.

Closed decode-storage synthesis, not active queue: routed Q5 and attention KV are
unrelated representation/kernel projects. v0.311 bounds only its pre-R2 same-work
weight/dequant family; it does not formally bound current R2 or ownership-changing
work. Existing evidence nevertheless supplies no named candidate that clears the
charged breadth gates. Do not implement a generic two-class oracle or matched-grid
touch/no-dequant kernels.

Q5 reopen: require a named representation/work unit against current production R2
and A10B shapes that predicts `>=22%` down-wave improvement on both models and
`>=5%` charged full-token movement. The full-token projection must use measured
phase shares; 22% local movement alone is insufficient. Do not reopen NSG/R2
reshaping, packed-ulong extraction, inner staging, or another current-layout
dequant sidecar.

Attention reopen: require a named body that changes ownership/utilization,
preserves or improves effective residency, and avoids head-major-only layout,
same-body compression, TGM K/V sharing, partition packing, tile1, and the killed
matrix point. Require `>=10%` actual-shape primitive gain across medium/breadth
guards before the existing true-long gate.

Closed native-MTP history lane, not active queue: the v0.587 code fixture makes
the decision before narrative. Full history needs 59 verifier packets; every
fixed K<=256 tail needs 74-79 and shifts mass from full-depth accepts to zero/one
accepts. Full itself remains below the acceptance continuation gate. Do not build
tail-only prompt construction, interpolate K, or reinterpret cheaper per-packet
attention as a product win.

Banked process-cold win: v0.590 defaults native token embeddings on only the
measured untied 27B Q4_K and A3B Q8_0 architecture fingerprints. Ten balanced
27B pairs improve process start to first byte `1.20283x` and process exit
`1.14866x`, saving `720.61/732.48 ms` and `4,370,432,000` Metal bytes. Six A3B
pairs improve the same endpoints `1.06057x/1.05267x`, saving `149.00/154.65 ms`
and `1,493,893,120` bytes. Exact outputs and warm transition parity hold. A
separate six-pair A3B TTFT confirmation improves median TTFT `2.50 ms`. A10B,
MTP-bearing variants, tied embeddings, and lookalike architectures remain
default-off; explicit `=1` retains the structurally supported opt-in path and
`=0` is the rollback.

v0.591 closes sparse CPU-prefaulted whole-shard no-copy as a universal latency and
warm-parity candidate. Its prefault costs `1.80-1.94 s`; candidate first byte loses
`368-435 ms`, exit loses `356-415 ms`, and warm transition throughput is only
`0.96787-0.96861x`. Exact state is bitwise and RSS/footprint each fall about
16.8 GB. It does not measure demand-paged no-prefault execution and therefore does
not close a bounded fresh-process one-shot policy.

v0.592 closes exposed hazard tracking as the missing warm mechanism. Its written
tracked-versus-untracked parity gate mechanically clears but is malformed for
recovery and non-authorizing. The valid direct contrast is only `0.99944-0.99989x`
on warm decode, versus roughly `1.032-1.033x` needed to recover copied storage.
Remove the dead flag. This closes only exposed hazard mode, not file-backed page
provenance, giant-resource topology, or no-prefault cold execution.

v0.593 promotes no-prefault retained views for one narrow fresh-process envelope.
At 128 outputs, first byte improves `1.7185x`/`1.477 s` and exit improves
`1.1972x`/`1.440 s`; private footprint falls about `15.65 GiB`. The known warm
transition penalty reproduces at about `3.3%`. Output 256 clears only `1.1016x`,
so the measured cap remains 128. The claim is exact-asset, cache-warm, 419-token
fixed-chunk prefill, 1024 context, greedy, fresh disposable process, and explicit
caller ownership. Structural matching, automatic selection, arbitrary prompts,
storage-cold use, persistent/server execution, and extrapolation from the fitted
1,500-transition crossover remain unauthorized.

v0.594 first removes the model-layout special case geometrically across 54 local
dense/MoE, precision/quant, tied, MTP, and split assets at `99.8935%`
byte-weighted direct coverage. v0.594c/d then clear the live force-only gates:
overlapping-window lifetime, physical aliasing, tied Q8 and A3B bit-exact full
state, converted-embedding coexistence, split A10B resource creation and sampled
GPU reads, plus the isolated exact-27B regression guard. This is broad live
correctness and resource authority, not latency, storage-cold, persistent,
automatic-policy, MTP, or universal-lifecycle authority.

v0.595 prices the first generic production cell and exposes a real objective
split. A3B retained storage saves `1.928 s` to first byte and `1.884 s` through
128 outputs while removing about `22.12 GB` of private footprint. But loaded
prefill regresses about 19%, transition throughput is only `0.86773x`, and the
complete loaded output-128 request loses `248 ms`. The frozen result is
`needs_review`: strong cache-warm disposable-process evidence, not policy
promotion. Copied teardown contributes another `141 ms` of apparent exit gain,
so exit wall must not stand in for persistent inference.

v0.596 resolves that warning against broad warm use. Across the prospective
`ABC,BCA,CAB` blocks, repetition-1 retained/copied decode is `0.87158x`, late
decode is `0.87058x`, and normalized retained recovery is `0.99885x`. Late
spread is below 0.3%. CPU-prefault C costs `2.441 s`, touches every page in the
`22,123,544,576`-byte planned window, raises maximum RSS to `22.360 GB`, and still
lands at `0.87015x`. Repeated packed prefill recovers to `0.99519x`, but p2
inserts an untimed prefill plus transition and uses a different harness from
v0.595. This narrows the prefill delta to before p2's timed regime without
localizing it; the transition tax persists. This closes
same-request post-warmup convergence and the current CPU-prefault operation on
the frozen A3B request, not GPU translation, one-resource topology, file-backed
provenance, expert-bank phase attribution, or arbitrary routes.

The leverage inversion is now sharper: current A3B retained views are a narrow
disposable-process/private-footprint specialization and a diagnostic control, not
the broad destination. Copied A3B spends about `1.99 s` materializing `22.12 GB`
through 733 independent shared Metal buffers, only about `11 GB/s` effective.
v0.597 now proves that one four-worker anonymous arena materializes the same bytes
in median `702.167 ms` versus `2062.791 ms` for the production copied primitive.
It saves a paired median `1365.044 ms`, wins 6/6 blocks, and preserves the same
RSS/footprint envelope. Serial arena copy loses at `3584.763 ms`; resource-count
reduction alone is not the mechanism.

v0.598 proves the missing product term. One-window owned storage is bit exact and
improves fresh first byte `1.86694x/1.85162x` at outputs 1/128, but loaded decode
falls from `1204.60` to `1390.20 ms` over 127 calls and complete loaded request
wall regresses about 11-12%. File-backed retained is effectively identical to
owned when loaded. The tax therefore follows the shared giant-resource/nonzero-
offset topology class rather than file provenance or unpopulated destination
pages. One-window owned is closed; do not proceed to async or broad arenas.

v0.599 clears the topology-preserving floor decisively. It retains 733 exact
offset-zero resources yet cuts materialization from median `2066.701` to
`742.676 ms`, saves `1324.148 ms`, and wins 6/6. This retains about 97% of
v0.597's one-window saving without adopting v0.598's loaded-tax topology. The
arithmetic first-byte projection is `2.06601x`; product transfer remains unproved.

v0.602 proves product transfer almost exactly. The force-only candidate preserves
bit-exact full state and every loaded 1% gate, then improves fresh output-128 first
byte `2477.32 -> 1199.75 ms` at paired median `2.06292x` and process exit at
`1.47929x`, both 6/6 wins. Runtime plus model load falls `2094.00 -> 811.78 ms`,
saving `1283.493 ms` paired median. Candidate ready wall is `747.527 ms`, within
0.7% of the v0.599 floor. RSS/footprint are unchanged. One tradeoff remains
separate: first fresh prefill is slower in all six pairs by paired median
`4.768 ms`, making TTFT B/A `1.01225x`; model-ready complete request is slower
in 5/6 pairs by paired median `5.481 ms`, and final-output-to-exit wall is slower
6/6 by `22.159 ms`. The systematic first-request regression is not observed in
late post-warmup loaded measurements. The packet does not localize its cause, and
the `2.063x` cold endpoint is not permission to hide the separate model-ready
tradeoff.

v0.608 admits that result only for disposable single-turn CLI loads. Auto
requires the exact frozen A3B profile, unified-memory `Apple M4 Max`, and at
least 128 GiB; host or profile mismatch falls back before allocation. Reusable
runtime loads, JSONL, dense 27B, and every explicit storage or representation
override remain on ordinary or force-only policy. Explicit
`QWEN_GGUF_PARALLEL_COPY=0` is the rollback; explicit true preserves the strict
broad force path. The admission consciously accepts v0.602's first-request
tradeoff rather than claiming a strict Pareto win.

v0.609 removes another fixed cold term without changing inference.
Compatibility identity costs `17.432-20.269 ms` on 0.8B and `18.882 ms` on A3B.
Six same-release eager/lazy pairs save `18.55 ms` paired median load wall, with
all six above the 10 ms gate. File-stat inputs remain captured at load;
expensive descriptor and metadata hashing moves behind `OnceLock`. Disposable
single-turn requests avoid it, cache/snapshot paths defer it, and request-timing
telemetry computes it after the recorded request endpoints. The fixed arm order
limits protocol breadth, but direct phase timing supports the causal scoped
promotion.

v0.610 closes JSON numeric allocation removal as a material latency lever.
Removing `serde_json/arbitrary_precision` from both manifests improves eight-row
release `GgufFile::open` median only `25.094 -> 22.940 ms`, a `2.154 ms`
observed saving. That misses the 10 ms fixed-wall gate and projects to only
0.80% of v0.609's measured 0.8B process-load-plus-TTFT boundary. The fixed
baseline-then-candidate order limits causal strength but cannot rescue the gate.
Do not escalate to identity-version work from this result. Typed metadata and
tokenizer arenas require independent memory or latency cases. Both manifests and
the temporary CPU test are restored.

v0.603 transfers the same-topology materialization floor to the exact dense-27B
Q4_K_M inventory. Across six frozen pairs, parallel copied materialization falls
from arm median `1519.5535` to `557.7565 ms`; authoritative paired saving is
`965.0405 ms`, paired B/A is `0.365706589x`, and B wins 6/6 with AB/BA medians
`966.018/964.063 ms`. All `16,806,250,496` bytes, 851 independent exact-sized
offset-zero resources, memory, and pressure gates pass. The win uses CPU
parallelism: median endpoint CPU rises about 45.5% and core-equivalents rise from
about one to four. The complete frozen implementation bundle clears the floor;
worker count, allocation behavior, manual copy, and page/cache effects remain
coupled. Authority is one separately preregistered force-only dense loader pilot,
not product or default admission.

The post-v0.601 portfolio review preserves the cold-first inversion while
correcting several stale external frames. PLD is already a product path behind
`--prompt-lookup`; v0.565 clears its copy-heavy packet and v0.566 demotes only
broad expansion on one reconnaissance panel, not every routed copy workload.
Packed verification is not generic wiring debt: direct prefill-body transplants
have already lost, and recurrent-state correctness binds A3B before economics.
The `12.5-12.8` nominal-TFLOP/s Q4_K anchor remains a current-dispatcher ceiling,
not an independent silicon peak. One causal actual-shape decomposition can test
that premise, but a derived peak is not itself an optimization result.

Dense-27B mmap-copy topology validation is closed without product authority.
v0.603 proves a `965.0405 ms` materialization-floor saving; v0.605 transfers that
endpoint at `559.370 ms` median but stops on loaded instability after 12 valid
children. Do not extend that harness chain. v0.626-v0.627 enter through a changed
population premise—direct destination pread—and independently establish exact
full state plus cache-warm fresh-process transfer. v0.628-v0.629 now clear the
separate default-ColdOnly target-file-cold composition guard in both orders.
v0.630-v0.634 now close exact dense ForceOnly direct-pread admission. The
terminal packet is formally `inconclusive-instability`, not KILL: prefill is
KILL-consistent at 8/8 misses and `1.015568/1.018036/1.014478x`
all/ABBA/BAAB, but A/B prefill ranges of `13.261%/14.090%` fail the frozen
stability gate. Treat this as strong directional evidence of a `~1.45-1.80%`
loaded-prefill cost, not a stable effect estimate. The profile is closed as
unable to certify `<=1%`; no force, retry, repair, or selector authority remains.

v0.607 closes the admitted true-long direct-F16 matrix point at its 32K
prerequisite. The valid A1/P/A2 main medians are
`0.164125/0.309084/0.164209 ms`; pair medians are
`0.208626/0.356583/0.208458 ms`. Correctness is near V4, but the candidate is
`2.09247x/1.90064x` its main/pair gates. Removing compressed decode,
split-plane addressing, and 16 KiB K/V staging improves the prior v0.583 matrix
body only `1.01402x`. The remaining loss is organization-level in the
eight-simdgroup cooperative QK/softmax/PV shape, while V4's sibling rereads are
predominantly cache-served. Close direct-F16 C/NWG/thread-count, barrier,
staging, format, and score-ledger rescues; no 131K row follows.

The active Trellis3 lane is the conditional model-changing leader, not an exact
engine result. Its T=256 V2G floor is near Q4_K bandwidth but carries a
source-coding quality tax and cross-session drift caveat. V1 is the stronger
quality point but misses the current kernel hold gate. Real-weight PTQ must
clear before either synthetic quality or kernel floors become a model, phase,
or request speedup claim. If that gate passes, rerank it above the exact cold
queue because it changes resident and streamed bytes across load, prefill, and
decode.

The July systems-audit pass promotes only candidates that attack measured work.
It does not reopen generic Metal or compiler folklore. Dense 27B already
measures about `98.7%` GPU-busy with `~0.20 ms` host encode against `~42.14 ms`
GPU time; double-buffered submission measured only `~0.3%`, and dispatch
boundaries are about `2-3 us`. That is existing falsification for ICB,
argument-buffer churn, completed-handler pipelining, unretained references,
deep CPU queueing, and broad hazard-mode surgery as primary work in the current
hot decode graph. Prompt scratch already has phase-disjoint overlays, the Metal
source is compiled ahead of runtime, and untracked retained storage was flat.
The narrow 0.8B PSO cell costs only about `1.8-2.2 ms`; do not generalize that
number to every model or first-use path. v0.609 prices runtime metallib
temp-file plus library load at `1.1-1.5 ms` and the two GGUF safety walks at
`3.8-4.3 ms`; both are below the 10 ms fixed-wall gate. Residency sets only
match explicit touch; that closes the API choice, not broader cold residency.

Likewise, `target-cpu`, PGO, allocator swaps, QoS, CPU SIMD, and Accelerate are
deferred, not falsified: first require a host profile that puts at least 2-3% of
first byte in their scope. BOLT does not target Mach-O. On current copied-storage
product paths, steady loaded decode and prefill issue no model-file reads; their
measured limits are GPU/DRAM, not current syscall wall. Rank syscall work from
attributed lifecycle kernel/VM time, not arithmetic intensity.

Current A3B Auto already direct-preads into final independent Shared resources
and suppresses its redundant `ColdOnly` pass. The serial retained-view
`prefault_read` loop is force-only diagnostic work, not current product wall.
v0.642 closes W5/W7, QoS, CPU-guard relaxation, cache/storage-cold transfer,
Auto change, and production promotion only for the direct-pread worker-count
treatment. It neither tests nor closes `MADV_WILLNEED` or parallel retained-view
touching; keep those separately deferred and do not present them as v0.642
successors. `MTLIO` remains a distinct conditional file-to-resource primitive;
v0.643's non-authoritative bounded row supplies no scored MTLIO mechanism
evidence. Measure it only beside a live A10B population baseline with exact
topology, CPU, cache, and full-state gates. `F_RDADVISE` has no measured
advantage over current destination pread, while `F_NOCACHE` changes
cache/pollution semantics; defer both pending a named storage-cold or pressure
profile. Heaps, superpages, right-sized attention partials, and dead-scratch
deletion remain memory work until pressure or wall attribution says otherwise.

Paged KV, COW state, and a resident daemon remain reuse/serving work rather
than fresh-prompt acceleration. Durable snapshots now have a measured
process-cold continuation case in v0.612; keep that result labeled reuse rather
than pretending it accelerates a first unseen prompt.

Banked process-cold win: v0.611 defaults a parallel-pread cache warmer to
`ColdOnly` at `0.9` full-file residency across every convenience loader
(`load_model`, disposable single-turn CLI, four `qwen-cli` bench paths).
Four workers share each shard's retained descriptor and pread `16 MiB`
stripes to populate the macOS page cache before mmap. Dense 27B `Q4_K_M`
first byte improves `23.69 s -> 4.87 s` (`4.86x`) with first-token identity
across 12 runs and prefill unchanged. Endpoint throughput is `~6.7 GB/s`
prefetch versus `0.5-0.7 GB/s` mmap demand paging; the `0.9` gate is the
analytical break-even of that ratio rather than a swept residency curve.
Always overhead on a fully resident dense file is `~+170 ms`. A3B disposable
is a single-round `12%` first-byte win at `~20x` physical I/O. v0.612 resolves
the source-union question: dense 27B's 851 requests and A3B's 733 requests each
form one contiguous interval covering exactly 100% of the tensor-data region.
Range-selective warming cannot omit model bytes on either asset. The remaining
changed premise is avoiding separate warm-then-copy passes. A valid
metadata-keyed durable-identity entry returns `bytes_hashed=0`; full BLAKE3 is
a missing/corrupt/unreadable-entry bootstrap or repair path, and an empty blob
store defers it until post-response publication. `ModelLoadIntent::ForceOnly`
does not disable prefetch; any `LoadedModelConfig` inheriting the default
receives `ColdOnly` unless it explicitly selects `PrefetchPolicy::Off`. The
authenticated disposable A3B Auto direct-pread path is now the one measured
exception: v0.624-v0.625 suppress its redundant `ColdOnly` pass only after exact
plan authentication.

v0.612 also measures the active repeated-conversation reuse shape. Across nine
adjacent Qwen3.6 27B ring0 transitions, full-prompt checkpoints are exact token
prefixes but leave 17,014 suffix tokens to replay. Retokenized completed-turn
proxies are exact prefixes and leave only 878, a `19.378x` suffix-work reduction.
This is not a fresh-prompt gain and the historical saves lack authoritative
generated IDs, but it moves completed-turn publication ahead of another loader
primitive for process-cold continuation after one live-token proof.

v0.613 closes that proof and product seam. Automatic token-preserving messages
histories publish completed `p0` boundaries; serial output-limit, serial EOS,
and target-only prompt lookup all restore at `matched=L/restored=L-1` from an
independently rendered next request. Explicit and transform-capable surfaces
retain prompt capture. The first 27B clean-store publication reveals a new
`6,734.5 ms` subprocess-exit tax in strong content hashing; an identity hit
publishes in `159.1 ms` with zero model bytes hashed.

v0.614 removes most of that first-use tax without weakening identity. Parallel
BLAKE3 over already-mapped shards reduces clean-store publication to a
`734.8 ms` median (`9.17x`) and full process wall to `3.92 s` (`2.50x`). Cache
entries and checkpoint blobs are byte-identical. Aggregate CPU time rises about
22%; claim latency, not energy or CPU-efficiency improvement.

v0.615 transfers the mechanism to one real 6.5K-token 27B history. Against the
identical uncached turn-2 prompt, completed restore improves prefill `17.72x`,
TTFT `14.43x`, and process wall `6.15x`, saving about 26 seconds of TTFT. The
582 MB blob restores in `335.6 ms`; pending plus 16 suffix tokens still need
`1,575.7 ms` of small-N prefill. One sampled agent/messages guardrail also hits.
This is one fixed-order row, not a population ratio.

v0.616 closes the client seam in external `llm` commit `eccccb7`. A disposable
Qwen3.6 27B ring0 pair publishes completed `12536/12535`, restores that boundary
from the next fresh process in `554.5 ms`, and extends it to `13130/13129`; both
turns end by EOS and append atomically. The client keeps exact assistant bytes,
fails closed on token-limit output, uses a 32 GiB aggregate/4 GiB per-entry disk
policy, and makes assistant-boundary forks ancestor-reusable. There is no
year-core injection. Prefix productization is complete for guarded serial use;
do not keep it in the active optimization queue by inventing more cache policy.

v0.621 completes A3B direct-pread Auto admission. The actual absent-environment
selector saves median `203.771 ms` of load, `209.427 ms` to first byte, and
`252.488 ms` through exit over forced mmap, all 6/6. A two-pair target-file-cold
default-ColdOnly guard also passes after full invalidation and physical reread.
Authority remains limited to authenticated disposable A3B; explicit controls,
fallback, and reusable intent are unchanged. The flat physical footprint and
small per-pair prefill/TTFT noise forbid broader memory or Pareto claims. This
admission leaves the active queue.

v0.624 removes the remaining duplicate A3B cold pass. Across six target-file-
cold pairs, suppressing full `ColdOnly` warming before the same direct pread saves
paired medians of `815.0 ms` load and `810.5 ms` to first byte, with 6/6 wins and
both order strata clear. AB/BA medians are `517/851 ms` load and `525/846 ms`
first byte; the overall values are balanced-protocol medians, not stable per-load
constants. Every child reports one shard-equivalent physical read within
0.01 GiB resolution, total CPU falls to `0.578-0.736x`, and footprint is flat:
the mechanism removes a second logical pass, not physical I/O or final copied
residency. v0.625 then seals the edited product path: both selector tables and
exact `ColdOnly`/`Always`/`Off` marker, residency, physical-read, and first-token
contracts pass. This completes the authenticated disposable A3B Auto pread
stack; do not spend another product packet on its current four-worker policy
without a changed worker or destination premise.

v0.626-v0.627 clear dense-27B direct-pread correctness and cache-warm fresh
transfer without granting product authority. v0.626's Rust full-state test
passes, but its packet stops before fresh work because interleaved `--nocapture`
load lines break a contiguous result recognizer. v0.627 repairs only that grammar
and reruns everything. Across six counterbalanced output-128 pairs, load saves
`1172.990 ms` median, runtime B/A is `0.361548x`, first byte is `1.453035x`, and
exit is `1.154306x`; both external endpoints win 6/6. First-prefill, TTFT,
generation, and model-ready request B/A are
`1.002296/1.002437/0.998844/1.000148x`. Full state, all 38 gates, pressure, output
identity, and both order strata pass. RSS halves only as process accounting;
footprint is `0.999577x`. Keep the claim cache-warm and explicit-force-only.

v0.628-v0.629 then clear the target-file-cold composition premise. v0.628 stops
after A1 because its warm-derived zero-major-fault rule rejects the intended cold
page-ins; no B observation exists. v0.629 repairs only that validity premise and
reruns four new `AB/BA` children. Direct pread saves `1143.253/1137.355 ms` of
load and `1141.071/1132.456 ms` to first byte. Physical reads are
`0.999929/1.000000x`, CPU `1.002469/1.007444x`, and footprint
`0.999242/0.999869x`; every pair gate passes and the first token agrees. These
are two descriptive cold pairs, not an effect estimate. This authorized only the
loaded-stability sequence; selector admission remained separate.

v0.630-v0.633 execute that successor without importing earlier timing. After
three harness-only validity repairs, v0.633 completes all 32 short-period
children with exact token identity, full residency, zero block input/swaps, zero
pread-marker major faults, and valid pressure and timing evidence. Frozen
all/ABBA/BAAB B/A medians are prefill
`1.013090/1.013479/1.012442`, decode
`1.000551/1.000551/1.001001`, and request
`1.008423/1.008423/1.009059`. Prefill misses in 5/8 quartets, below the 7/8 KILL
rule, while per-arm prefill ranges of `9.419%/10.815%` fail the 5% stability
gate. The sealed result is `inconclusive-instability`, not force authority.
Quartets 3-8 have a `1.01538x` prefill median, so a prospective ramp-control
block does not favor the candidate. Permit one terminal discriminator only;
any non-GO closes this exact profile under the frozen loaded-prefill contract.

v0.634 executes that terminal discriminator with one fixed, fully validated but
unscored ABBA/BAAB ramp reversal before 32 fresh scored children. Ramp evidence
enters no score or stability series. All/ABBA/BAAB B/A is
`1.015568/1.018036/1.014478` prefill,
`1.001632/1.001446/1.002299` decode, and
`1.010219/1.011377/1.009829` request. All eight prefill quartets exceed 1.01,
but A/B prefill ranges of `13.261%/14.090%` invoke frozen instability precedence.
The formal result is `inconclusive-instability`, while the prospectively terminal
portfolio disposition closes the exact profile as unable to certify `<=1%`.
v0.629's separate `1.132-1.141 s` cold-first-byte observation remains
descriptive and does not override loaded admission.

v0.622-v0.623 attribute and close publication's duplicate staged decode. On an
exact 582.9 MB checkpoint, fixed-buffer digest validation removes one complete
snapshot population but saves only `17.4/18.1 ms`; publication and whole-process
footprint stay flat. Extra baseline page reclaims match the removed bytes, so
the negative transfer is explained: both arms still reread/hash the full file,
and the late temporary snapshot never owns the process high-water mark. Do not
reopen allocation-free readback under the same publication contract.

v0.635 closes the page-rounded independent-resource image route. B requests a
16-KiB multiple for every exposed length; 232 of 733 lengths increase, raising
the exposed-length sum by `2,758,144` bytes while preserving logical bytes,
offset-zero topology, population, modes, and full state. Prefill, within-child
stability, memory, pressure, and identity all pass, but decode and request miss
the frozen every-pair 1% gates in pairs 1, 3, and 4, mechanically requiring KILL.
The post-hoc latency clustering is not treatment-directed: the four-child fast
cluster is 2A/2B, the eight-child slow cluster is 4A/4B, and marginal A/B
decode/request means are nearly identical. Record both lessons: this packet
authorizes no successor for the frozen route, while the evidence does not
establish a page-rounding, VM/TLB, no-copy, or Metal compatibility tax. Do not
rerun or rescue it by filtering those post-hoc clusters.

v0.636 implements the changed checkpoint integrity premise, but its first packet
correctly seals `implementation_or_contract_defect`: Cargo libtest prefixes the
canonical marker on its physical line while the runner requires a whole-line
match. One A child launched, no B observation exists, and no gate, row, timing, or
authority transfers. v0.637 independently preregisters only a strict parser repair
over the unchanged implementation, authenticates the complete predecessor seal,
and reruns four fresh `AB BA` children. It seals `GO`: staged integrity falls from
`258201/262030 us` to `21/20 us`, while complete publication falls
`650608 -> 384879 us` and `613572 -> 340732 us`. Savings of
`265729/272840 us` clear the 250 ms gate in both orders. Every final record has the
frozen SHA-256/BLAKE3/namespace/topology and passes post-publication full decode and
snapshot equality. This is CPU-floor authority for one real product packet only;
verification is deferred to first use, not removed.

v0.638-v0.639 then expose two pre-model-child packet-contract defects without
producing a model execution or product-performance observation. v0.639 manifest
construction nevertheless hashes the frozen model file. v0.638 preflight
incorrectly requires the full commit as a contiguous raw substring in
`qwen-bench`; semantic build identity is exact, but LLVM does not preserve that
representation, so no packet is reserved. v0.639 repairs executable-specific
identity and all four fresh gates pass, then its exact-`0755` policy conflicts
with its own `umask(077)` build output (`0700`). It seals after gates with no
model child, row, restore, scored-timing, or performance artifact. Its sealed
gate rows and process records contain gate wall/timestamp evidence, none of which
transfers to v0.640. The complete seal is authenticated in the log. Because
v0.639 reserved and terminally closed packet ordinal 1, it authorizes no informal
retry. This post-v0.639 PERF certification independently authorizes one ordinal-2
v0.640 correction changing only the executable-mode set to `{0700,0755}`; every
gate and product observation must be fresh, and all experiment and product-
authority limits remain frozen. Any v0.640 result exhausts ordinal 2 and grants
no successor authorization.

v0.640 executes that fresh ordinal-2 packet and seals `GO`. All 12 scored children
and the unscored restore are valid. Per-pair publication savings are
`275997-294254 us`; external final-stdout-to-clean-exit savings are
`277629-299776 us`. Every pair clears both 250 ms gates, with descriptive medians
`284821.0/290482.5 us`. The restore hits the unchanged checkpoint at `6500/6499`,
publishes nothing, and reports no corruption. This banks only force-only use of
the existing explicit `QWEN_CHECKPOINT_STAGED_INTEGRITY=deferred-restore` control
for the exact tested Qwen3.6-27B-Q4_K_M surface. `decode` remains default; no
automatic, other-model, other-shape, retry, or successor authority exists. The
external endpoint is a post-response publication/exit tail, not TTFT or complete
request latency.

v0.642 completes the repaired cache-warm A3B direct-pread worker screen as a valid
KILL. W6 and W8 save paired-median `130.501/128.508 ms` against W4 and win every
wall pair, but their timer-local CPU ratios are `1.12376x/1.50363x` against the
frozen `<=1.10x` guard. W12 reverses to `92.072 ms` slower at `3.47393x` CPU.
The unchanged work and topology scale through about six workers, plateau at eight,
and reverse by twelve; the packet does not localize that knee's cause. It closes
cache-warm worker-count tuning without W5/W7, QoS, relaxed-CPU, storage-cold, Auto,
or production authority. The next independently justified candidate changes the
population primitive rather than buying wall time with more CPU.

v0.643 tests that next population primitive and seals inconclusive at P1 B after
one timer-local major fault. No scoring or successor is authorized. The bounded
prefix is directionally poor: A/B ready walls are `530.984/542.067 ms`, while B's
`90.627 ms` GPU interval sits inside `538.971 ms` of copy wall. The clean source-
lifetime proof and complete residency do not attribute the fault or the remaining
`448.344 ms`. Close this transient-source Shared-destination floor. Any reopen
requires a changed, independently justified diagnostic premise that attributes
major faults and separates command encoding, queueing, and GPU execution.

v0.644 completes the production-Q4 attribution ladder and seals
`CLOSE_TESTED_BC_ABLATION_LANE`. Fresh A is healthy at `13.220385` nominal
TFLOP/s. Sixty paired blocks give A/B `1.039287` (`1.039099-1.039475`), B/C
`1.029773` (`1.029608-1.029938`), and A/C `1.070230`
(`1.070054-1.070405`). Both charged upper bounds remain below `1.132304`; A/C
projects only `1.02748x` ideal whole-prefill movement. B's dead volatile loads are
transaction proxies, not exact source timing, while C/E jointly change activation
traffic, staging, barriers, and simdgroup-load organization. Close the tested
same-work-unit source/dequant lane. Do not chase E0/E8 direction or another local
ablation.

v0.645 seals valid `INCONCLUSIVE_RESOLUTION` with no default authority. Its
preregistered transition timer includes the candidate GPU reducer but excludes
the baseline's following CPU vocabulary scan, so it prices the replacement
subpath rather than complete product work removal. The A3B secondary product wall
is nevertheless exceptionally stable: decode/token `1.085807x`, one-sided 95%
interval `1.083085-1.088536`, all eight pairs positive, exact output/state, and
neutral prefill/TTFT. Dense is contaminated by a nonlinear opening whole-session
hump; its final six decode ratios are informatively positive but non-authoritative.

v0.646 is a sealed `INCONCLUSIVE` hybrid-GDN equivalence sweep with no default
authority. It separates bounded decode cleanup from the larger roadmap bets:

- F32 beta projection+sigmoid fusion is a repeatable but small dense-0.8B signal
  (`+0.255%` wall, `+0.276%` GPU over balanced 20-round `tg128` packets). The
  corrected A3B census removes 30 beta dispatches/token, but the post-fix GDN
  replay does not close the local timing attribution. Keep it as cheap default-on
  hygiene, not a strategic GDN branch.
- Paired Q/K RoPE removes 10 corrected A3B dispatches/token and has a small 0.8B
  wall signal (`+0.217%`), but the GPU interval crosses zero and the 27B dense
  scale check is negative/noisy. Keep it below the warm promotion gate.
- Grouped MoE finalization removes 37 corrected dispatches/token and gives a
  balanced A3B decode signal of `+0.674%` wall / `+0.636%` GPU. The cooled repeat
  remains positive but noisy; this is a bounded decode cleanup, not evidence that
  the prompt grouped-MoE lane has reopened.
- Raw-Q/qscale cancellation has no promotion signal on 0.8B, A3B, or dense 27B;
  the non-winning middle-state implementation and `QWEN_DECODE_GDN_SKIP_Q_L2`
  flag are deleted. Reopen only with the K-fold that removes the L2 preparation
  dispatch and materialization. Output-scale folding, legacy attention
  normalization folding, and cached softmax exponentials remain correctness-only
  records until rollback-capable performance variants exist.
- The direct GDN recurrence gate remains unmet: no one-layer all-in `>=20%`
  result, no projected prefill `>=5%` result, and no production arm-to-arm
  hidden/logit/token trace. Do not promote these algebraic leaves above the
  existing prompt, residency, or structural execution branches.
- The follow-up review re-ranks the next measurements away from isolated
  micro-ops: accepted tokens per MTP/DFlash verify pass, command-buffer and
  encoder counts for the wall-minus-GPU gap, and speculative-block GDN state
  amortization. At scale, prefer the already-green Q8 KV body; treat dense
  0.8B-F32 quantization as an oracle/benchmark fixture, not a micro-kernel win.

v0.648 seals `INVALID` after a process-classifier false positive and grants no
authority or reusable timing. Its changed-predicate successor v0.649 seals
valid `INCONCLUSIVE_CONTAMINATION`, also with no authority. Complete A3B decode
again strongly favors exact GPU greedy (`1.090151x`, lower bound `1.061442`,
8/8 pairs), as does first-use decode (`1.097558x`, lower bound `1.074309`), but
steady prefill fails the frozen control at `0.957634x` and TTFT mirrors it.
The middle-window disturbance affects both arms and only post-generation
requests; all 16 request-zero prefills remain stable. Do not rescue the result.
Absence is default-off at `15a7092`, while explicit `=1` retains the exact
mechanism. That sharpened the remaining question to prior-generation carryover,
not reducer efficacy or same-request prefill causality.

v0.650 closes the immediate one-generation carryover hypothesis in its exact
A3B cell. Sixteen fresh-process pairs give balanced treatment prefill
`Q=0.997324`, normalized carryover `C=0.996885`, and a one-sided 95% upper bound
of `1.000368`, strictly below the frozen `1.03` harm threshold. Exact state,
fresh conformance, token digests, terminal semantics, cache isolation, host
validity, and all 254 inventoried artifacts pass. This is diagnostic-only: it
does not rescue v0.649, prove the source of its bilateral disturbance, authorize
automatic selection, or cover cumulative/concurrent workloads. Descriptively,
the paired treatment decode-ms/token A/B geometric ratio is `1.081378`, with
16/16 ratios above one, but decode was outside the decision. Stop this exact-cell
immediate one-generation branch unless a materially different cumulative or
concurrent workload supplies a new premise.

v0.651 attempts the bounded fresh one-shot admission decision and seals `INVALID`
before scoring. Pure policy/profile and exact-state gates pass. Artifact
inspection shows identical N128 A/B output and correct policy/profile telemetry;
the formal pair comparison does not execute. The runner rejects B's legitimate
`capacity_validation_ms=0.0` because it mistakenly assigns
`capacity_validation_ms` a strict-positive domain. There are zero scored
attempts and no admission evidence. The packet is consumed and cannot be
repaired or retried; it does not rewrite v0.648-v0.650. Mandatory rollback
`0927937` restores absence to default-off and removes the unadmitted selector.
Any independent successor must use a new preregistration and artifact root,
test zero-valued timing fields before acquisition, and pool no v0.651
observation.

v0.652 consumes that one independent successor and seals `INVALID` before formal
N128 conformance. Its first A process exits zero with coherent output, policy,
profile, timing, and request telemetry, but exact rusage reports 24 major faults
with zero block-input operations. There are zero scored attempts and no
admission evidence. The process-wide fault counter is unphased and cannot
attribute I/O or policy causality. The same first-A count appears descriptively in
v0.651, while v0.652 exact-state records 27; none may be pooled or used to rescue
the packet. This repeats the validity-design mismatch exposed by v0.628: a
blanket process-wide zero-fault gate substitutes for cell-specific cache/I/O
evidence. The frozen gate correctly seals `INVALID`; it does not prove
target-file page-in or localize an engine phase. Terminal rollback `88ec2dd`
restores default-off. The preregistered stopping rule closes
automatic A3B GPU-greedy admission permanently under the current mechanism;
retain explicit `=1` and do not create a v0.653 A3B automatic-admission
successor.

The syscall/VM audit adds a secondary lifecycle queue without changing the
fresh-prompt ranking. On installed `rustc 1.96.1` for
`aarch64-apple-darwin`, both `File::sync_all` and `File::sync_data` map to
`F_FULLFSYNC`. Checkpoint blobs and the 128-byte identity cache each full-sync
their file and parent directory. Blob synchronization is post-response
publication/exit work. Identity-cache synchronization is a separate small case
and can enter pre-restore TTFT when managed blobs exist but the identity entry
misses or is invalid. Measure the exact `582,854,188`-byte blob floor and
identity-entry policy separately before changing either contract.

The restore path also has one changed premise. It decodes and hashes the full
record into CPU arenas, then copies those arenas into independent Shared session
buffers. A quarantined positional-vector-read path could populate a fresh
session directly and hash the canonical bytes before any GPU use. A bounded
serial `read_at` spike now measures the missing causal terms on a
first-completed `581,740,008`-byte 27B checkpoint. Incumbent warm lookup was
`285.623/278.115 ms`; applying the decoded arenas cost `160.876/201.160 ms`.
Direct placement nevertheless reached only `448.2/582.4 ms` total versus the
two preceding incumbent observations at `472.2/504.4 ms`. The first direct run
was `24.0-56.2 ms` lower; the second was `78.0-110.2 ms` higher. Direct lookup
added `126.850/270.882 ms`, consistent with destination page wiring and scattered
population consuming the removed memcpy. The seeded 16-token continuation
remained byte-identical. Close serial direct placement below the `60 ms` gate and
retain the current restore.

Two narrow follow-ons survive. Replace the misleading
`has_managed_blobs` accounting scan only with a separately documented may-exist
hint that may short-circuit on the first recognized nofollow regular blob; full
scans remain mandatory for eviction, publication, budgeting, and integrity.

Positive-temperature product decode now has one realized structural result.
v0.659 closes workspace-only, borrowed-only, and their nonstructural combination,
then authorizes one combined packet. v0.660 consumes that packet and seals `GO`
on the exact frozen A3B fixture: median generation saving `5.002899%` and request
saving `61.094834 ms`, both with 6/6 positive pairs; median TTFT delta is
`+1.962104 ms` and spawn saving is `46.533751 ms`. Policy retains the hidden,
default-off exact-profile switch. The generation gate clears by only `0.002899`
percentage points, so do not claim robust per-run `>=5%`, uniform TTFT
nonregression, default admission, broader transfer, or another timing packet.

Generic certified lm-head screening is now closed. The consumed clean v0.664
oracle prunes zero of 248,319 competitors at all six captures and charges about
431.1 MB per capture against a 417.2 MB complete head and 125.2 MB gate. Exact
validation confirms every unique winner; the bound is correct but wholly
nonselective. Do not build a production screener or retune survivor machinery.
Reopen only for a materially tighter certificate with charged traffic below the
head bytes it avoids, or changed model/head geometry.

Serial direct-to-session checkpoint restore is closed. Do not retune syscall
count or schedule a `preadv`-only follow-on; this spike did not isolate syscall
overhead or show that aggregation could recover the gate. Reopen only for a
materially different population/hash organization or changed product importance.

1. **A10B cold residency and split-copy floor is parked**: v0.653 consumed its
   sole packet unsealed before the first durable child launch. The no-payload
   headroom probe passed, but ordered hashing of all three shards followed by a
   global full-residency check found shard 1 nonresident. A post-stop diagnostic
   showed `0/668`, `2,944,390/3,029,833`, and
   `1,671,006/1,671,038` resident pages in read order. This invalidates the exact
   conditioning method, not either population arm; no timing or effect estimate
   exists. Reopen only with current deployment relevance and a changed-premise,
   phase-local physical-I/O oracle. It does not outrank the active items above.

   v0.653 accepts v0.538's exact native-embedding stream/warm parity and v0.594's
   split-resource proof only as a benchmark-local force-native premise. Its sole
   clean metadata describe at `41a12b0` is now frozen under raw-file SHA-256
   `ce1b3ccfd67a1a5b8cdaf71050dfd9547ec4ca06f29559da1b4a19473a0cdef9`.
   The exact `a10b-q4xl-v1` profile binds inventory digest `b331c475...a4f8`,
   879 requests, `77,018,996,736` bytes, and literal W4 cuts
   `[214,435,658]`. The implementation admits only exact-profile force-native
   copied and W4 parallel-pread, with in-child headroom, retained-descriptor,
   timer-local I/O/swap, allocation-drop, and no-GPU-command seals. The consumed
   packet supplies no native default, runtime loader, product, or mechanism
   authority.

2. **High-ceiling structural options**: true-long attention needs a source-free
   body that changes ownership, scheduling, residency, or physical bytes after
   v0.607; speculative decode needs matched MTPLX AR/D3/D7 acceptance evidence
   before asset or affine work; A3B verification needs a materially different
   serial-state-preserving organization. Keep all three behind their existing
   source-free oracle and whole-token gates. Prize high, belief low-medium.
Dense GPU-greedy deconfounding remains a separate diagnostic-only question that
v0.652 neither answers nor closes; it is intentionally deprioritized below the
active queue on leverage.

Checkpoint cleanup and the reduced-cost may-exist probe are lifecycle
infrastructure, not a new inference claim. New-format staging inodes are
kernel-leased and receive bounded cleanup after process death; legacy unlocked
names are never guessed dead. Keep the catalog-free store until measured metadata
wall justifies one; provisional instrumentation triggers are roughly 512 family
entries, 256 in one compatibility directory, warm inventory p95 above 10 ms, or
cold p95 above 50 ms. If a trigger fires, add rebuildable SQLite metadata only;
immutable blob files and descriptor leases remain authoritative. Root-wide
family budgeting, physical free-space admission, and value-per-byte eviction
require independent telemetry before implementation. v0.660 consumes the sole
sampled-product timing packet and completes its no-new-performance policy review.
Serial direct restore
is closed. `MTLIO` remains conditional on a live A10B baseline.

Below the line: v0.609 closes standalone GGUF safety-walk consolidation and
temp-metallib I/O under the 10 ms gate. v0.610 closes manifest-only JSON numeric
allocation removal under the same latency gate; typed metadata retains only an
independent memory or changed-representation case. Repack-on-load needs a named
current-kernel instruction attribution; global allocators and tokenizer automata
need new independent cases. Adaptive MoE top-k also falls below the active queue:
router mass is diffuse and ideal k8-to-k6 removal is only about `4.55%` before
overhead or quality loss. v0.655 clears only the exact response-shape direct-row
primitive and one A3B integrated packet; generic norm-certified lm-head
screening is closed by v0.664.

Blocked cold follow-ons remain conditional. v0.602 satisfies the first
prerequisite for async retained-to-copied promotion, but command-buffer-safe
cutover, copy contention, and secondary serving scope keep it below serial cold
breadth. Broad resource-count/offset/MMU attribution is not needed for the
current A3B decision: same-topology copied storage already transfers the floor.
v0.635's completed size control failed its frozen every-pair loaded gate and
closes that page-rounded independent-resource image route. Its arm-balanced
post-hoc latency clustering does not support treating the KILL as an open-ended
MMU verdict or a conclusion about every no-copy image. Metal does not expose
physical GPU page placement or TLB policy. Completed-turn durable snapshots are
banked reuse evidence above. A resident daemon remains useful deployment work,
not fresh-prompt model-process-cold acceleration.

Memory follow-up: v0.590 promotes native embeddings for 27B and A3B. v0.591 proves
that CPU-prefaulted 27B views remove about 16.8 GB but lose latency; v0.593 proves
that no-prefault views retain the same private-footprint prize and win the narrow
one-shot envelope above. v0.594 prices 0.384-77.016 GB of removable base-weight
private allocation across 54 local assets and realizes the force-only generic
resources. Converted embeddings, derived buffers, CPU mirrors, MTP weights, and
live file-backed residency remain separate. The v0.594 A10B retained-storage result
is resource and sample-read authority, not full-forward, latency, or footprint
authority. v0.596 closes current retained A3B as a broad warm representation and
shows CPU prefault raises maximum RSS by about 22 GB without restoring speed. Owned
arenas deliberately retain anonymous private allocation. v0.602 now proves their
same-topology prize is cold materialization wall plus copied warm speed, not the
retained footprint reduction; the force-only A3B result leaves RSS/footprint flat.

Dense fresh-TTFT truth: no current exact branch has a credible material gain band
for large dense prefill. Current packed compute is near the measured mat-mat
anchor. v0.644 bounds complete source/dequant removal represented by the current
N64 organization at only `1.070405x` primitive / about `1.02754x` ideal whole
prefill. Material movement requires a new GDN work unit, fewer prompt/model bytes,
a representation or ownership-changing projection work unit, or a model, input,
or precision tradeoff; another ordinary projection retune is not active.

The ranking reflects the current process-cold deployment mix while retaining
model-ready TTFT, warm decode, breadth, probability, and engineering cost. It does
not merge objective lanes; every result must retain its boundary label.

### Active attack sequence

This is the dependency-aware execution order for the force-ranked entries
above, not a second ranking. CPU-only work may advance while a foreign GPU lease
is live, but timed Metal work remains serial.

Treat v0.593 as a measured narrow 27B disposable-process optimization, v0.594 as
broad force-only retained correctness, and v0.596 as the closure of current
file-backed A3B for broad warm use. Persistent, server, MTP, storage-cold, and
automatic retained use stay copied without separate evidence.

v0.640 banks exact-27B deferred restore behind its existing explicit environment
control. It is no longer an active experiment; keep `decode` as the default.

1. v0.652 consumes the final independent GPU-greedy admission packet before
   scoring and closes the branch under its terminal stopping rule. Keep absence
   default-off at `88ec2dd`, retain explicit `=1`, and schedule no v0.653 A3B
   automatic-admission successor.
2. v0.644 closes the tested Q4 ladder. Do not schedule B/C widening, E-arm
   diagnosis, P4096 replication, or a local source/dequant implementation.
3. v0.653 is consumed unsealed before its first child launch. Record no arm or
   timing inference, do not retry its global full-residency method, and park A10B
   loading until a changed-premise phase-local I/O oracle becomes deployment-
   relevant.
4. v0.654 kills forced-run fast-forward in its exact response-shape cell;
   v0.655 clears the charged direct-row floor; v0.656 then kills the integrated
   request-local bank organization. It saves `14.229 ms` of generation but loses
   `7.662 ms` total request and regresses TTFT `15.100 ms`. Schedule no repair or
   rerun. A grammar successor needs a changed bank-lifetime, lazy-organization,
   or materially longer constrained-workload premise.
5. v0.659 clears only the structural sampling bound; v0.660 consumes its sole
   packet and clears the exact frozen fixture at `5.002899%` median generation
   saving and `61.094834 ms` request saving, both 6/6 positive. Retain only the
   hidden, default-off exact-A3B force path. Do not rerun, widen, automatically
   admit, or build workspace-only, borrowed-only, GPU-sampling, or
   prompt-borrowing variants. v0.664 subsequently kills generic norm-certified
   lm-head screening with zero rows pruned and bytes above the complete head.
6. Keep prompt reduction explicitly input-changing. Keep true-long attention,
   MTPLX/asset work, A3B verifier redesign, lm-head screening, and top-k behind
   their named source-free, state, quality, and whole-token gates.
7. Do not resume the closed page-rounded image route, transient mmap-source
   Shared-destination blits, broad topology attribution,
   generic command-graph or compiler work, retained-view retunes, broad external
   drafting, matrix/compressed attention, sparse retrieval, routed-tail work,
   generic packed GDN, mixed quant, broad prompt lookup, or local retuning without
   an explicit reopen gate.

### Decisive gates

- **Completed-turn durable state**: use authoritative generated token IDs, not
  retokenized assistant text. Record stop reason and the emitted-but-unconsumed
  pending token. The next request must decompose into the exact consumed prefix,
  pending token, and suffix. Uninterrupted, RAM-restored, and disk-restored
  continuations must preserve the declared exactness contract. Keep the current
  prompt-boundary checkpoint whenever the completed boundary is unavailable.
- **Checkpoint durability**: hold encoding, BLAKE3, no-clobber publication, and
  later decode validation fixed. Compare raw `F_FULLFSYNC`, capability-checked
  `F_BARRIERFSYNC`, raw `fsync`, and no-sync for file data and parent directory
  separately; unsupported operations fail the arm rather than silently falling
  back. No-sync is admissible only under the explicit disposable-cache contract:
  hard-link publication provides no-clobber live final-name publication within
  the documented private, cooperating namespace, while crash or power loss may
  erase, truncate, or corrupt the entry and must fall back to replay. It does not
  strengthen the store against malicious concurrent parent-component
  replacement.
- **Direct durable restore**: parse and bound the complete canonical record before
  constructing destination spans. Retain the validated descriptor, handle
  positional-vector short reads, `EINTR`, `IOV_MAX`, EOF, and section boundaries,
  then hash in canonical wire order before GPU use. Poison and discard the whole
  candidate session on failure. The serial direct-placement spike preserves this
  contract and exact continuation but misses the `60 ms` wall gate as destination
  population offsets the removed copy. Keep the current arena decoder. Reopen
  only for a materially different population/hash organization; this spike gives
  no evidence that `preadv` alone could recover the gate.
- **Checkpoint existence hint**: a short-circuit may-exist probe is not a store
  audit. It may skip later foreign-entry or I/O discovery after a valid hit. Keep
  complete scans for byte accounting, eviction, publication, cleanup, and any
  explicit integrity operation.
- **Retained storage**: planner windows are read-only weight resources, never
  unqualified scratch tensors. Synthetic overlapping resources must preserve
  bytes and lifetime under either destruction order. Every live model must match
  the ordered dry request inventory and copied full state. Physical accounting
  separates views, aliases, tail copies, conversions, derived buffers, CPU
  mirrors, and MTP. v0.596 does not authorize broad warm A3B promotion under
  current file-backed storage; A10B resource creation is not a latency promotion.
- **Owned weight arenas**: use anonymous shared Metal storage with a logical
  weight-only provenance, not retained read-only provenance or generic scratch.
  Report arena, live direct, gap, alias, fallback, and crossed-conversion bytes.
  A materialization floor authorizes only loader implementation. v0.598 proves
  that one-window owned shares retained's decode tax despite exact destination
  population; the premise is closed. Any reopen must preserve per-tensor
  offset-zero topology through the loaded gates before fresh product timing.
  v0.599 clears the A3B materialization floor, v0.602 clears exact A3B full
  state, loaded parity, and cold product endpoints, and v0.608 admits only the
  authenticated disposable single-turn use. v0.621/v0.624/v0.625 complete its
  direct-pread Auto population and redundant-prefetch suppression. v0.642 measures
  the cache-warm worker-count knee: W6/W8 clear the 60 ms wall gate but fail the
  frozen CPU guard, so that tuning is closed with no storage-cold successor or
  production authority. v0.643 then stops its changed transient mmap-source blit
  floor inconclusive on one timer-local major fault at P1 B. Its bounded row is
  slower and exposes `448.344 ms` outside reported GPU execution, but authorizes
  neither a mechanism miss nor causal fault attribution. Reopen only with a
  changed diagnostic premise that independently measures those terms. v0.603
  clears the dense-27B same-topology floor;
  v0.626-v0.627 add exact full state and cache-warm fresh transfer under direct
  pread; v0.628-v0.629 clear default-ColdOnly target-file-cold composition.
  v0.630-v0.634 close dense explicit force admission. The terminal packet is
  formally `inconclusive-instability`, with 8/8 prefill misses and strong
  directional evidence of a `~1.45-1.80%` loaded-prefill cost. Its prospectively
  terminal non-GO leaves no force, retry, repair, or selector authority.
  A10B breadth requires its own inventory, floor, correctness, and product
  evidence.
- **N8 verifier**: every speculative ratio names a current denominator artifact.
  Require a verifier-only whole-decode oracle `>=1.10x` in a regime before
  authorizing a new proposal source there. Dense 27B short/interactive clears;
  the current A3B packed-MoE/GDN implementation fails its state contract.
- **Auto-prefill admission**: candidate scratch uses explicit query cap 1024 and
  the proven overlay. Charge complete scratch plus sequence/transient reserve;
  require valid Metal working-set headroom and honor a finite process limit when
  available. On the target desktop, zero means the process limit is explicitly
  omitted, not a second valid headroom signal. Preserve the A3B/A10B file-type-15
  allowlist and `8192..=16384` range. Missing usable signals, MTP, prefill
  environment overrides, cache interaction, disabled online attention/overlay,
  or insufficient memory fall back to 1024. Require fresh TTFT confirmation on
  each admitted profile before default-on.
- **Production-Q4 attribution**: no-op arms must preserve the exact production
  grid, loop counts, initialized inputs, half-to-float MMA family,
  accumulation/store observability, and every instruction not intentionally
  removed by that named arm. Reject DCE, races, or undefined-value shortcuts.
  v0.644 measures A/B/C at `13.807170/13.285232/12.901128 ms`. The simultaneous
  A/B and A/C upper bounds are only `1.039475/1.070405`, both below the
  `1.132304x` charged gate. Close semantic dequant and combined source/dequant
  designs represented by this N64 organization. B's volatile reads are dead-result
  transaction proxies; C/E is a non-isomorphic residual, and E8>E0 is not causal
  evidence for TGM or barriers. Reopen only for a named race-free real-Q4 design
  that changes representation or work-unit/topology, includes complete source,
  activation, staging, and store costs, and conservatively predicts `>=1.133x`.
- **Grammar fast-forward**: count uniquely admissible tokenizer-token runs, not
  characters, grammar transitions, or isolated singleton positions. Charge grammar
  scanning and use measured terminal-head-only packed state cost. Require
  `sum(H_r * (r*C1-Cpack(r))) - overhead >= T0/11` independently on named request
  archetypes before engine work. Grammar-row lm_head restriction may contribute to
  the same savings but must preserve the declared constrained-output contract.
  v0.654 finds maximum grammar-global run one and no canonical forced positions
  across its 36-string response-shape language. This kills fast-forward only for
  that fingerprinted language/token-piece-policy cell. A multi-ID state or
  terminal, not trace frequency, ends every singleton chain; the oracle does not
  attribute each multi-ID state solely to same-target token segmentation.
- **External drafting**: use per-depth survival and `Q=1+sum(P(A>=j))`; never infer
  economics from mean alpha alone or from llama.cpp wall time. Require exact
  tokenizer/special-ID compatibility and charge draft prefill, D7 work, full-accept
  catch-up, rollback/replay, correction consumption, and memory. Current no-prefill
  Q floors for 1.10x are `3.88` 0.8B and `4.24` 2B; promotion still requires the
  normal two-archetype request gate and target-greedy equivalence.
- **Certified lm_head screening**: first use the production winner as an optimistic
  bound and outward-rounded per-block norms. Kill unless every primary capture
  safely prunes `>=80%` of rows with `<=30%` baseline bytes touched after metadata.
  Then add production dequant/accumulation error and require about `70%` charged
  head-wall removal plus `>=5%` whole-token projection. The claim is exact selected
  argmax, not bitwise full-logit equivalence. Exact grammar-row restriction is a
  separate direct-row contract and does not require norm screening. v0.655 clears
  its state-major Q6_K floor at `84.85%` A3B and `92.85%` dense head-wall removal.
  v0.656 then kills the A3B integrated request-local bank: generation saves
  `14.229 ms`, but setup plus post-terminal cost produces `-7.662 ms` request
  saving and `+15.100 ms` TTFT. Reopen grammar rows only on authenticated
  cross-request bank reuse, a lazy/indexed organization, or a materially longer
  constrained workload. Generic screening retains its independent optimistic
  oracle; it cannot inherit grammar's distribution-exact contract.
- **Positive-temperature sampling**: attribute full-head GPU wall, completion and
  readback, full-logit copy, candidate construction, retained-candidate weight
  allocation, and selection separately. Kill reusable workspace work unless
  its adjusted candidate-specific bound is `>=5 ms` and `>=5%` of complete
  `generation_ms` in the median and five of six children. Preserve sampler-v1
  candidate order, RNG draws, error behavior, and seeded output exactly.
  Mandatory full-logit or probability work receives no avoidable-work credit.
  On the exact fixture and current mechanism, v0.659 closes W/B/C and clears
  only S at `81.400598 ms` / `6.545897%`, 6/6. v0.660 consumes the sole
  successor and clears all frozen gates: median generation saving `5.002899%`,
  request saving `61.094834 ms`, TTFT delta `+1.962104 ms`, and spawn saving
  `46.533751 ms`, with 6/6 generation and request wins. The narrow headline
  margin and one `+15.610708 ms` TTFT pair permit only the hidden default-off
  exact-profile path, not default admission, broader transfer, or another
  timing packet.
- **Prompt lookup**: use actual charged replay, not a mean-acceptance surrogate.
  Require median decode `>=1.10x` over the prompt fixture triad, no important row
  below `0.98x`, TTFT `<=1.03x`, and proposal CPU cost below 1% of decode wall.
  The current dense-27B denominator sets a zero-overhead, zero-abstention
  necessary floor of about `3.35` emitted tokens per charged N8 packet. Apply the
  promotion gate to a frozen policy on held-out rows, not its selection corpus.
  The v0.562 charged bench clears the mechanism gate. The v0.565 product packet
  clears the favorable, generic, exact-output, TTFT, and total-wall gates for the
  validated dense-27B Q4_K_M layout. It does not estimate natural trigger rate,
  qualify other layouts, or establish distribution/bit exactness.
  A failed proposer demotes prompt lookup; a failed verifier-only oracle demotes
  all proposal sources only in the measured model/context regime.
- **Native MTP**: total request `>=1.10x` on at least two named archetypes at 128+
  output tokens, TTFT regression `<=3%`, and greedy-equivalence green.
  Price acceptance, verifier packets, and packed-history cost in one equation.
  For the v0.587 code request, 59 packets cannot clear 1.10x even with zero extra
  history cost; a one-second packed-history budget permits only about 48-50.
  MTPLX D3 evidence cannot authorize qwen D7. Require D7 plus a cross-trunk
  sidecar acceptance bridge before import. Distribution exactness remains a
  separate rejection-sampling implementation.
- **Routed-tail reopen**: first change an ownership premise; then name the removed
  intermediate work, improve complete tail `>=15%` on both models, and improve
  whole prefill `>=5%` A3B and `>=3%` A10B.
- **Q5 representation/work-unit reopen**: name the current-R2 mechanism, improve
  down wave `>=22%` on both A3B/A10B, and project `>=5%` charged full-token gain.
  Local movement alone is insufficient.
- **Attention format/body reopen**: first show `>=10%` actual-shape primitive gain
  across medium/breadth guards, then retain the true-long gate below.
- **GDN recurrence**: complete one-layer all-in gain `>=20%`, projected prefill
  gain `>=5%`, and exact state or an explicit numerical contract.
- **True-long attention**: main-body gain `>=15%`, whole-token gain `>=6%` at
  131K, and no material 32K regression before model/context widening.
- **Approximate/model-changing**: quality-harness v0 must pass the declared task
  tolerance, and expected gain must clear the lossy admission threshold.

### Work-reduction and quality frontier

These deployment/model choices can dominate engine work but alter input or target
semantics. Keep them explicit rather than mixing them with exact kernel claims.
The bands are unvalidated priors until quality-harness v0 prices them.

1. **Smallest quality-passing model**: `1.5-10x` TTFT and `1.5-15x` decode;
   evaluate as a different model on a fixed hard-task Pareto set.
2. **Prompt/context reduction**: `1.1-2x` TTFT and `5-30%` true-long decode;
   ablate prompt classes before approximate compression.
3. **Sensitivity-aware mixed quant**: `0-10%` TTFT and `5-20%` decode;
   search tensor classes rather than a per-tensor combinatorial grid. Calibrated
   class-level error injection may rank sensitivity cheaply, but does not replace
   validation with a real quantized asset.
4. **Adaptive MoE top-k**: current phase arithmetic gives about `4.55%` ideal
   short-decode throughput for k6 and roughly `7%` for k5 before overhead or
   quality loss. There is no current end-to-end quality replay; build only a named
   offline oracle before changing target execution.
5. **Sparse/retrieval attention**: the canonical source fixture fails the captured
   physical-retention frontier before implementation. Reopen only if a dissimilar
   or true-decode artifact makes group-shared 50% locally green, sharply reduces
   physical GQA union, or avoids full scoring without replacing byte savings with
   random reads.

Lossy branches need a quality corpus, hard-task guardrails, and a materially larger
gain than exact local work. Apply approximate MoE or precision changes inside a
target-verified drafter first when that preserves target authority.

### Search discipline

- Use a sparse decision surface: one highest-ceiling sentinel plus one dissimilar
  guardrail. Do not build every model x quant x context cell.
- Use a prompt fixture triad for workload-sensitive claims: favorable short,
  canonical real-long, and an adversarial witness. One prompt cannot promote a
  proposal, policy, or approximate model change. Use separate development rows
  to select proposal policy, then gate on a frozen held-out triad plus a generic-
  narrative guardrail.
- Require a costed oracle before a kernel for grammar fast-forward, external
  drafting, lm_head screening, speculation, top-k, sparse attention, mixed
  precision, and compressed KV.
- Constrain configuration dimensions from prior evidence: physical N8 for MTP,
  tensor classes for mixed quant, static per-layer top-k before adaptive policy,
  and retained-KV fraction before sparse-attention implementation.
- Measure model-ready TTFT, p50 inter-token latency, total request wall, peak
  memory, and quality/exactness. Do not promote on a microkernel ratio alone.

Every candidate should record: objective lane, model/context/format scope,
current work unit, specific work removed or reused, zero-cost oracle ceiling,
added work/state cost, exactness class, serving-state consequence, cheapest
decisive experiment, result, closure boundary, and explicit reopen condition.

Maintain a lightweight mechanism ledger from these rows: attempts, kills,
promotions, measured net gain, and reopen condition per lane. Use a one-time
history digest to calibrate admission thresholds; do not build a prose-mining
system that displaces engine work.

Advance through an oracle ladder: Amdahl ceiling, offline semantic oracle,
actual-shape primitive including packing/reduction/state writes, one body/layer,
whole-phase sentinel plus guardrail, then product request. Stop when the remaining
whole-phase gain no longer pays for the next level.

### Secondary batch-serving lane

Preserve a deliberate path to paged KV/state, continuous batching, independent
request overlap, prefix sharing, and layer-synchronous execution. Primary BS=1
changes should avoid hard-coding state ownership or layouts that make those
facilities unnecessarily difficult later.

The lane is now implementation-backed rather than hypothetical. Dense Qwen B=8
ships for compatible JSONL cohorts, preserves input order, falls back leftovers
to serial execution, and fans out shared prompt prefixes. Dense 27B whole-model
decode reaches `2.57x` aggregate throughput at short context and `2.14x` at 16K.
Independent-queue overlap remains the cross-family fallback (`1.44x` 0.8B,
`1.47x` A3B, `1.37x` DeepSeek K160 at B=2).

Qualified Qwen 35B-A3B now also ships an exact fixed B=16 executor. The product
path preserves serial JSON byte-for-byte, reaches `155.614` aggregate token/s on
16 x 32-token generation, and improves complete process wall
`9.437 -> 7.663 s` (`1.231x`) despite serial per-lane prefill. Common-prefix
fanout composes directly: a 443-token identical-prefix fixture moves
`7.898 -> 3.399 s` (`2.324x`). Keep width 16, the frozen architecture/quant
envelope, F16 KV, full-cohort requirement, model-derived admission, and serial
remainders. The exact executor is a serving capability, not a replacement for
single-stream decode.

The fallback now clears a generated-continuation gate rather than only a
teacher-forced transition. Across two distinct 32-step greedy streams, A3B
retains exact IDs and final logits at `1.378x`; DeepSeek K160 retains exact IDs
and final logits at `1.402x`, including full-logit readback. The first narrow
product slice now ships for dense and MoE Qwen as regular-file JSONL with
explicit concurrency two, serial prefill, pairwise independent-queue decode,
input-order emission, odd-tail serial fallback, and model-derived admission for
two sessions plus maximum prefill scratch. Dense 0.8B and A3B product smokes
match ordinary serial output exactly across heterogeneous prompts and token
limits, including paired-to-serial handoff. Keep dense B=8, stdin, prompt lookup,
prefix-cache mutation, and request sidecars fail-closed. Call the capability
resident concurrency, not batching.

Qwen resident concurrency now composes with pair-local prepared-checkpoint
fanout for qualifying fixed-chunk pairs without mutating the prefix cache.
Identical 6,469-token A3B prompts move
pair preparation `8,251.823 -> 4,199.627 ms`, or 1.965x, while a pair sharing
6,475 tokens at a stable 6,144-token boundary moves
`8,244.651 -> 5,293.907 ms`, or 1.557x. Both preserve complete per-request JSON
output exactly. Keep fixed-chunk alignment, the 256-token floor, snapshot-aware
memory fallback, and `QWEN_CONCURRENCY_PREFIX_FANOUT=0` rollback. This is the
highest-leverage current batch-serving composition for repeated system prompts;
it applies equally to dense and MoE Qwen because route scratch is not causal
snapshot state.

DeepSeek V4 now exposes the same surface through worker-local sessions over one
shared immutable residency. Its memory plan admits two sessions up front; K160
prices `99.15 GB` including reserve and observes `8.683 GB` of two-session state
against an `8.692 GB` session inventory. Keep packed prefill serial: overlapping
two prefills is exact but expands them enough to lose whole-request wall. Release
both prepared workers only for generation. Greedy and heterogeneous seeded
sampling fixtures match serial output exactly; equal-length sampled generation
retains `1.280x` aggregate movement. Fail closed when the command-queue-scoped
DeepSeek residency-set opt-in is active.

DeepSeek concurrency now also composes with pair-local causal-snapshot fanout.
Identical 6,219-token K216 prompts move pair wall
`58,106.922 -> 29,248.723 ms` (`1.987x`); a pair with a 6,225-token LCP selects
the shared 6,144-token packed boundary and moves
`56,376.329 -> 31,511.527 ms` (`1.789x`). Both preserve complete output rows
byte-for-byte. Keep the external prefix-logit bridge for exact restores because
DeepSeek causal snapshots intentionally omit observations, use transient
same-residency identity rather than full durable model hashing, include the
restore image in admission, and release the host snapshot before decode.

Packed-scratch aliasing is closed as a concurrency-memory lever. Reusing the
512 MiB raw-query allocation for phase-disjoint MoE outputs produced
byte-identical output in the tested fixtures and saves exactly 512 MiB/session,
but regresses a full-chunk single session by about `0.9%` and a long B=2 prefill
by `3.06%`. The implementation was deleted. Reopen only for chunk-sized
allocation or a representation that reduces capacity without imposing
shared-resource alias cost on the hot path.

Qwen A3B static B=8 remains closed after a complete whole-model spike. Aggressive
packed execution reaches `1.43x` but changes continuation; its bitwise-incremental
repair reaches only `123.7` aggregate token/s, below the `125.81` independent-
queue control. Width alone did not fix that organization.

A materially new exact B=16 organization now reopens the product lane. The first
wider complete-model probe was confounded by a non-production serial MoE FFN tail
and non-exact GDN mat-mat projections. Mirroring production route/concurrent-
shared FFN waves, retaining sequence-private attention and recurrent state, and
using token-axis exact Q8 GEMV for all GDN qkv/z/out projections produces exact
residual state. A token-axis Q6_K head with singleton-identical lane mapping and
accumulation then preserves every logit bit. The 66-transition packet has exact
generated-ID hashes and byte-identical final active KV, GDN state, convolution
state, tokens, and logits. Corrected timing charges host argmax readback across
five source-identical processes. All exactness checks pass and all five runs
clear the old `1.10x` queue line, but median candidate throughput is `140.880`
token/s (`1.119784x` the queue control), missing the frozen `1.12x` gate by
`0.0272` token/s. Keep productization HOLD: retain the exact primitives and
complete-model probe, but do not spend the remaining scheduler/JSONL work without
an exact measured gain that leaves integration margin or materially cheaper
shared executor machinery.

The first exact reopen attempt is closed. The Q6_K head has an isolated
`13.2133 -> 1.7833 ms` non-exact matrix ceiling, but a two-token exact work unit
lands between whole-model controls at `137.662` versus `138.133/136.575` token/s
(`1.0022x` interpolated). It is bit-exact, so correctness is not the issue; the
existing token-axis grid already realizes cache/fabric reuse, and added live state
consumes the proposed gain. Do not widen to T=4 without a different measured
mechanism.

Exact packed Q4 routed gate/up changed the Qwen MoE B=16 product decision.
Keep route/top-k, Q5 down, shared experts, attention, GDN, and the Q6 head on their
exact production-compatible organizations; pack only hidden rows plus ordered
top-k IDs, compute bitwise routed inners once across the cohort, then return each
inner to its owning session before production down/final waves. Three final
66-transition processes reach `147.562/150.820/150.300` token/s; median
`150.300` is `1.194659x` the frozen queue control, and every logit, residual,
generated hash, and final causal snapshot is exact. Authorize a sibling fixed-
B=16 Qwen MoE JSONL executor with prefix fanout, admission, ordered output,
cancellation/poisoning, and incomplete-cohort serial fallback. The integrated
path now clears that gate at `155.614` aggregate token/s and `1.231x` complete
process wall over ordinary serial JSONL, with byte-identical outputs.

The product executor is now capability-planned rather than tied to one model
filename. Each block can independently use exact Q8 GDN, packed Q4 gate/up, or
production per-lane fallback, and the head can use exact token-axis Q6 or Q8.
The unchanged Qwen3.6 A3B Q4 composition remains exact at `158.269` aggregate
token/s. Qwen3.5 A3B IQ4_XS also remains exact and improves serial wall
`8.77 -> 7.28 s`, but independent B=2 is slightly better at `7.17 s`; keep B=2
as that composition's preferred width. A short MTP-tagged Q4 smoke is exact.
This establishes the durable boundary: tensor/kernel contracts decide whether a
stage can batch, while measured whole-plan economics decide which width a future
scheduler selects. Do not replace that distinction with another filename
allowlist. A10B remains `HOLD` pending a memory-safe exact probe against B=8/B=2.

An opt-in automatic selector now operationalizes that boundary for regular-file
JSONL. It prepares requests once, inspects Qwen MoE plans without allocating
executor scratch, prices complete candidate state before selection, and chooses
serial, independent B=2, dense B=8, or qualified MoE B=16. The measured A3B Q4
composition prefers B=16, the exact IQ3-routed composition prefers B=2, and MTP,
A10B, or unmeasured MoE plans remain serial. Dense and DeepSeek use their
family-supported defaults with memory fallback. Keep explicit width flags as
operator overrides and keep auto opt-in until continuous arrival scheduling has
its own latency/throughput policy.

Fixed B=8/B=16 membership now permits heterogeneous requested generation limits
when prompt token counts match. Sort each prompt-length bucket by limit, allocate
the cohort maximum capacity to every lane, and batch only candidates whose
requested productive transitions fill at least three quarters of physical slots.
Moderate A3B limits 24-39 remain exact and move process wall about `1.10x`; dense
0.8B moves `1.607x`. The exact `0.75` boundary moves `1.042x`, and homogeneous
two-token work moves `1.020x`; reject the zero-transition case. A skewed
1-40-token B=16 cohort is exact but `0.744x`, so the gate sends explicit mode to
serial and lets auto choose B=2 at `1.078x`. Keep this as bounded
logical-termination flexibility, not evidence for lane refill or variable prompt
frontiers.

Equal-length cross-sequence packed prefill is closed before implementation.
Dense 27B charged projections take `15.697 s` as eight `N=512` traversals and
`15.772 s` as one `N=4096` traversal (`0.995x`); required layout moves
`16.107 -> 16.314 s`. This misses the `1.10x` ceiling before private causal
mixers and lifecycle cost. Do not build the B=8/B=16 prefill executor around the
current mat-mat kernels. Reopen only with explicit cross-sequence weight-tile
reuse or a different mechanism-level proof.

Qwen fanout now promotes the exact token LCP when the aligned boundary already
qualifies and every exact private suffix fits the six-token singleton pocket.
Dense 27B B=8 prefill moves `23.623 -> 8.307 s`, A3B B=16 moves
`7.237 -> 1.827 s`, A3B B=2 moves `1.480 -> 1.078 s`, and dense 0.8B B=8
moves `1.144 -> 0.360 s`; all outputs remain byte-exact. Preserve
`QWEN_PREFIX_FANOUT_EXACT_LCP=0` as aligned rollback. Broad exact matching stays
diagnostic because a one-token alignment gain with a long suffix regresses
`1.757 -> 1.800 s`. Do not broaden fanout admission or transfer this boundary
policy to DeepSeek.

Restored Qwen private suffixes now use singleton teacher forcing through six
tokens in B=2/B=8/B=16. Dense 27B suffix wall moves `3.325 -> 1.899 s`, A3B
B=16 moves `1.968 -> 0.741 s`, and A3B B=2 moves `0.246 -> 0.093 s`; all
product outputs remain byte-exact. Dense crossover screens keep packed execution
at seven or more tokens: eight is slightly slower and sixteen materially slower.
Keep `QWEN_PRIVATE_SUFFIX_SINGLETON=0` as rollback and retain packed scratch
admission. Exact-LCP fanout remains a separate segmentation-equivalence question.

Fixed B=8/B=16 planning now tries complete lexically adjacent prefix groups
before generation-depth packing. Adopt a prefix candidate per prompt-length
bucket only when it increases full cohorts, or preserves cohort count without
increasing estimated physical transitions; otherwise restore the baseline plan.
Two interleaved prefix families move dense 0.8B wall `4.79 -> 1.34 s` (`3.575x`)
and Qwen A3B Q4 wall `25.45 -> 7.62 s` (`3.340x`), both byte-exact. Keep
`QWEN_FIXED_COHORT_PREFIX_PACKING=0` as rollback. Overlapping prefix windows may
recover additional groups later, but missed affinity now degrades to the prior
depth plan rather than serial work.

Seekable Qwen B=2 files now plan pairs inside independent 16-request windows.
Prefer the largest actually restorable prefix edges, then pair remaining requests
by adjacent generation depth; preserve stdout input order with a hard window-sized
buffer bound. Interleaved long-prefix A3B work moves warm wall
`18.48 -> 12.81 s` (`1.443x`), while a no-prefix skewed-depth fixture moves
decode `1.218 -> 0.982 s` (`1.241x`); both remain byte-exact. Keep
`QWEN_CONCURRENCY_PAIR_PLANNER=0` as rollback. DeepSeek uses the shared planner
only when explicitly enabled until a safe K160-class validation can run. This
cheap scheduling win precedes dynamic B=2 refill; it does not remove the serial
private-prefill wall; ragged fixed-wide execution was authorized later only
inside the charged dense envelope described below.

DeepSeek K160 common-route B=8 is also closed at its cheapest model-backed
floor. Giving all eight rows the same six experts, the production all-slot
control and six N=8 Q3_K/Q4_K mat-mat chains move `1.9760 -> 1.8630 ms/layer`
GPU (`1.0607x`). Routed experts account for about `10.51` of `46.40 ms/token`,
so perfect propagation through all 43 layers predicts only about `1.013x`
whole-token movement, before realistic route diversity or scheduler cost. Do
not build a K160 static scheduler around current common-expert mat-mat. Reopen
for a materially different kernel that clears a 30% routed-stage floor, another
stage with an independently large whole-token ceiling, or a shared executor
whose marginal product cost changes the crossover.

Low-cost architectural seams to preserve now:

- Keep mutable target, draft, KV, GDN, and rollback state sequence-owned.
- Put speculative mutation behind explicit checkpoint, commit, and rollback.
- Pass logical positions through state APIs rather than adding new baked linear
  offsets that would block future slot/page mappings.
- Version new weight ABIs and support views so prefill and decode do not require
  duplicate banks.
- Represent an emitted but unconsumed terminal token explicitly if a resumable
  API later returns live sequence state.

When this lane becomes active, rank work by aggregate throughput, per-request
p50/p95 latency, memory per live sequence, scheduler occupancy, and exactness.
Existing S2/S8 evidence is prior information, not a current implementation order.
Do not require serving arrival traces to justify BS=1 work, and do not use BS=1
latency gates to reject a serving mode whose contract explicitly trades latency
for aggregate throughput.

### Closed and deferred neighborhoods

- Current-layout routed-Q5 work remains NO-GO. The pre-R2 no-weight oracle improves
  total decode only `2.2-2.4%`; it is not a formal current-R2 bound. Reopen only
  for a named representation/work unit predicting `>=22%` down-wave improvement
  on both A3B/A10B and `>=5%` charged full-token movement.
- Same-body exact Q8 KV and direct canonical Q4_0 remain closed. Reopen compressed
  KV only with a materially different format/body and a real-model fidelity gate.
- Gamma-fold hybrid K plus row-Q8 V is not such a reopen as proposed: its aligned
  K+V row is larger than the green group16 format, rotated gamma does not commute,
  and rowwise V is fidelity-yellow. Planar Q4/Q6 repack also remains closed because
  it removes no bytes and the cited 270 GB/s cap does not describe production GEMV
  generally. Reopen either only with a new captured-fidelity or counter-isolated
  premise that clears the existing whole-token gate.
- Keep BF16 parity archaeology, F16 partial/reducer work, decode glue, generated
  MTP heads, local GDN reshuffles, generic persistence, and standalone argmax out
  of the active queue.
- ANE/AMX drafting remains behind a replay-current request win and a standalone
  latency test that charges state transfer and unified-memory contention. GPU token
  graphs/deep queues remain closed under the current per-token callback and terminal
  semantics; existing GPU argmax, single-CB, boundary, and pipeline evidence does
  not support a 3-6% central estimate.
- Rejection sampling is live for DFlash2's causal sparse selector distribution
  against packed target logits. It is not a universal speed win and does not make
  alternative branches a linear N8 proposer union; further work belongs in sampled
  policy/backend numerics rather than the greedy speed queue.
- RoPE-table/GDN-exp hygiene and the v0.646 algebraic decode leaves remain below
  the warm gate. Paired RoPE and beta fusion are bounded cleanup signals; the
  raw-Q middle state is closed and deleted, and the exact epsilon/attention
  rewrites still lack rollback-capable perf A/Bs. Keep only a causal process-cold
  residency control
  for the unexplained first-prefill gap; do not infer a residency program from
  the aggregate gap alone.
- Dense fixed-cohort B=8, qualified Qwen MoE B=16, and shared-prefix fanout are
  active secondary serving capabilities. Qwen MoE static B=8 remains closed as
  described above; independent streams and future paged attention remain in
  this lane. Their serving evidence does not rank against serial BS=1.
- Defer ICB/MTL4, binary archives, no-copy loading, and residency sets as warm
  throughput priorities. Revisit them for process-cold load or memory objectives.

Dense decode update: v0.340 production-wires dense GDN front-projection overlap
and defaults it with `QWEN_DECODE_DENSE_CONCURRENT_GDN=0` as rollback. Sequential
`tg128` A/B improves the dense family by about `+2.4-3.8%`:

| Model | Old default | Concurrent GDN | Ratio |
| --- | ---: | ---: | ---: |
| 0.8B | `360.36` | `372.23` | `1.033x` |
| 2B | `218.35` | `226.60` | `1.038x` |
| 4B | `112.57` | `116.81` | `1.038x` |
| 9B | `72.09` | `73.93` | `1.026x` |
| 27B | `24.28` | `24.86` | `1.024x` |

Measured roofline anchors from v0.285 on M4 Max, AC power, high-power mode, no
recorded warnings:

| Anchor | Shape | Measured | Interpretation |
| --- | ---: | ---: | --- |
| Stream | `512 MiB` per buffer | `474.0 GB/s` | decode bandwidth denominator |
| Q4_K mat-mat | `4096x4096x1024` | `12.80 nominal TFLOP/s` | regular prompt projection ceiling |
| Q4_K mat-mat | `8192x8192x512` | `12.53 nominal TFLOP/s` | size-sensitivity check |
| Scalar FMA | `4M elems x4096` | `3.03 TFLOP/s` | scalar/latency sanity, not matmul peak |

Current caveats:

- The full v0.203 family `27B pp512` row showed `0.913x`, but immediate paired
  repeats showed `1.000x` and `1.008x`; treat that cell as parity/noise until a
  longer repeat packet says otherwise.
- Small dense short prefill is still the main primary-family residual. v0.262
  re-anchors 2B Q4 `pp512` as a real paired loss (`0.960x/0.953x`) while 0.8B
  remains too lcpp-variable to drive work alone (`0.936x/0.989x`). v0.263 widens
  dense Q4 fused-SwiGLU default coverage to `hidden <= 2048`, improving 2B
  `pp512` by `+0.5-1.3%` qwen-only and narrowing paired rows to
  `0.968x/0.978x`; rollback is `QWEN_PREFILL_DENSE_FFN_FUSED_SWIGLU_Q4=0`.
  v0.264 also disables Q4 N64 mat-mat tiles for small `n_query <= 512` shapes
  when either projection side is `<= 2048`, giving another `~0-1%` at 0.8B/2B
  `pp512` and narrowing 2B paired rows to `0.971x/0.982x`. The live residual is
  no longer a clean single-kernel story. v0.265 wall reconciliation shows 0.8B
  `pp512` CPU setup/encode/commit is only `~0.17-0.28 ms`; wall is dominated by
  GPU wait, with GPU timestamps at `~65.9-66.0 ms`. Patched local llama.cpp op
  profiles put FFN and GDN-ish aggregates in the same broad range as qwen. v0.267
  counts `205` compute encoders and `433` dispatches for both 0.8B and 2B Q4
  `pp512`, so this is not an 0.8B-only dispatch-count explosion. v0.269 count
  clusters show GDN front+alpha/beta, dense FFN, attention front+rope/scatter,
  and GDN prep dominate dispatch count; v0.270 says a simple token-channel
  parallel GDN prep is correctness-safe but flat/noisy through `pp16384`, while
  v0.274 finds a real head_dim=128 paired-L2 shape win by replacing the oversized
  per-row threadgroup with one simdgroup per row and four rows per threadgroup.
  Keep attention, N64 tile policy, residual-add epilogues, shared-memory policy,
  GDN prep token-channel parallelization, and CPU orchestration off the primary
  branch unless fresh accounting contradicts this.
- v0.272 shows small dense has a benchmark-methodology trap: default llama.cpp
  `n_ubatch=512` creates wins for qwen just above `pp512`, but llama.cpp
  `-ub 1024` reopens 0.8B at `pp512/576/1024` (`0.948x/0.937x/0.960x`). 2B is
  much closer against the same tuned control (`0.978x/0.983x/1.000x`). Treat
  family-default wins as scoreboard wins, not as evidence the tuned small-dense
  gap is closed. v0.274 defaults the head_dim=128 paired-L2 R4 specialization
  with `QWEN_L2_PAIR_HD128_R4=0` as rollback; clean current-commit tuned rows are
  now 0.8B `pp512/1024` at `0.977x/0.996x` and 2B `pp512/1024` at
  `1.019x/1.022x`. The residual is now mostly 0.8B `pp512` fixed-shape overhead,
  not a broad small-dense failure. v0.279 then defaults the same R4 row shape for
  GDN rmsnorm-gated with `QWEN_RMSNORM_GATED_HD128_R4=0` as rollback. Clean tuned
  rows are now 0.8B `pp512/1024` at `1.023x/1.023x` and 2B `pp512/1024` at
  `0.999x/1.034x`; treat 2B `pp512` as parity/noise, not a new kernel red cell.
- A10B very-short prefill is not a kernel-roadmap item unless a warmed paired row
  regresses. The v0.261 default `pp128` paired repeat still loses from cold
  first-touch variance (`0.789x/0.628x`), but `QWEN_PP_WARM_MOE_BANKS=1` gives a
  same-session steady-state win (`284.20` qwen versus `256.47` llama.cpp,
  `1.108x`). Keep outer wall visible because the warm touch is not free.
- A10B decode is now decisively green after the v0.291 Q8 mat-vec default and
  v0.292 clean re-anchor: `tg64/tg128/tg256` are `1.20x/1.20x/1.21x` versus
  pinned llama.cpp. Keep MoE decode active because hardware utilization remains
  the objective, not parity alone.
- A3B long-context decode attention had a stale group8 execution-shape miss.
  v0.293 defaults `GROUP=8` decode to tile2 for `n_pos >= 4096` with
  `QWEN_ATTN_V4_G8_TILE=8` as rollback. The A3B `ctx128/1024/4096/8192/16384`
  sweep is now `100.0/95.0/94.0/89.1/82.2 t/s` versus rollback
  `99.5/94.8/89.4/83.7/72.8`; `ctx16384` attention drops
  `4.80 -> 3.60 ms`. Attention is still the long-context slope term, but the
  first split fix is now defaulted. v0.367 lowers the group-8/group-16
  subgroup/NWG/C64 threshold to `ctx256` with
  `QWEN_ATTN_V4_SUBGROUP_MIN_POS=4096` as rollback. This closes the medium-context
  MoE valley: A3B `ctx3072` moves `94.1 -> 102.8 t/s`, and A10B `ctx3072` moves
  `37.9 -> 43.7 t/s`; phase attribution says A10B attention drops
  `7.29 -> 3.47 ms`. Stop threshold fiddling here unless a fresh context-slope
  sweep finds a new valley; the next attention branch should be true-long KV or
  layout pressure. v0.368 then defaults A3B group-8 true-long decode to
  tile4/NWG256 at `ctx >= 16384`; A3B `ctx32768` moves `75.7 -> 85.4 t/s`, and
  attention drops `5.14 -> 3.68 ms`. This validates KV byte-reduction as the
  right true-long lens, but tile8/NWG256 remains slower than tile4/NWG256. The
  next A3B attention branch needs a better read-once/group-fused occupancy plan;
  otherwise switch to the broader MoE-down branch. v0.369 banks the cheap reduce
  follow-up for that selector: the two-threadgroup reduce moves A3B `ctx16384`
  `91.2 -> 95.0 t/s` and `ctx32768` `85.4 -> 88.2 t/s`. Further attention work
  now needs a structural read-once or partial-traffic reduction, not another local
  reduce split. v0.370 kills the concrete normalized-half partial-traffic proof:
  it is correctness-safe, but A3B `ctx32768` `attn-intra` only moves
  `0.3457 -> 0.3408 ms/layer` (`1.014x`). Do not keep drilling partial storage;
  switch back to MoE/GDN dataflow unless a genuinely new read-once attention
  execution shape appears. v0.403/v0.404 refreshes the A3B long-context board:
  attention is still phase-faithful and the largest slope term, but another local
  NWG/reduce selector is small. NWG192 saves only `0.13-0.15 ms` phase time at
  `ctx16384/32768`; the next attention branch must be a main-body KV/partial
  traffic oracle that clears `>=0.30-0.40 ms` full-decode-equivalent savings.
  v0.461/v0.462 timestamp splits sharpen the diagnosis: at A3B ctx16384/window16,
  `attn_body_out` is `1.917 ms` and attn-intra main+reduce growth explains the
  context slope, while split GDN-after totals are flat from ctx4096 to ctx16384
  (`~2.65 -> ~2.60 ms`). Promote attention work only through normal-path A/B
  (`>=3-5%` at ctx16384, no ctx4096 regression), but rank attention main/reduce
  ahead of fixed GDN-after cleanup for long-context decode. v0.463 then kills the
  most concrete read-once follow-up: a two-simdgroup group8/tile4 V-staged
  sidecar preserves the four-head register shape and shares only V, but A3B
  ctx16384 `attn-intra` regresses main from `0.1047 ms/layer` to `0.2056` at C16
  and `0.3344` at C32. Do not reopen TGM-staged K/V or V-only sharing without a
  counter signal proving lower traffic and preserved occupancy. v0.488 then kills
  the opposite occupancy-only split against the current tile4/NWG256 true-long
  path: group8 tile1 is exact, but A3B ctx16384 `attn-intra` main regresses
  `0.1035 -> 0.1993 ms/layer` and one-layer total regresses
  `0.3490 -> 0.4574 ms` because estimated main bytes jump
  `0.0714 -> 0.2727 GB`. Future attention work needs byte reduction with
  preserved reuse/occupancy, or a real online-attention rewrite, not smaller
  subgroup splits. v0.490 rejects promoting the existing G8 score-broadcast
  attention sidecar to an automatic true-long default: opt-in still has a main-body
  signal, but rollback-first A3B `ctx32768 --window 8` measured rollback
  `90.4 t/s` versus auto `90.1 t/s`. Keep `QWEN_ATTN_V4_G8_BCAST=1` opt-in until
  it clears a repeated full-decode gate. v0.464 splits the tail and kills
  standalone argmax work: A3B ctx16384 has `lm head 0.87 ms / 7.2%` but
  `lm argmax 0.05 ms / 0.4%`; treat lm-head as a batched/dataflow economics
  component, not a local argmax-fusion branch.
- Shared Q8_0 SwiGLU fusion is a small default MoE decode cleanup. v0.294 fuses
  shared gate/up Q8_0 mat-vec plus `silu_mul`; rollback is
  `QWEN_DECODE_SHARED_SWIGLU_Q8=0`. A3B `tg128` moves about `+0.7%`, A10B
  `tg128` about `+0.2-0.3%`, and the shared-FFN phase drops `0.99 -> 0.85 ms`
  on A3B / `2.08 -> 1.82 ms` on A10B. This confirms the shelf, but it is not a
  strategic discontinuity by itself.
- F32 row-pair mat-vec is a small default MoE route cleanup. v0.295 defaults a
  cooperative two-row / four-simdgroup F32 mat-vec with
  `QWEN_MATVEC_F32_LCPP_R2=0` as rollback. A3B `tg128` moves about `+0.9-1.0%`,
  A10B about `+0.2-0.4%`, and A3B `ctx16384` stayed neutral-positive. Treat this
  as a route-execution shelf win, not proof that F32 mat-vec retuning alone can
  close the remaining decode headroom.
- Fused Q5_K routed down+weighted-sum is a larger default MoE decode bucket win.
  v0.297 routes single-token Q5_K down through the existing packed-slot fused
  kernel, removing `[topk, hidden]` routed-output traffic plus the separate
  weighted-sum dispatch. Rollback is `QWEN_DECODE_MOE_Q5_DOWN_FUSED=0`. A3B
  `tg128` moves `101.86 -> 103.8-105.1 t/s`, A10B moves `43.96 -> 44.65 t/s`,
  and A3B `ctx16384` moves `85.4 -> 86.1 t/s` in the guard.
- Fused MoE decode finalization is another small default cleanup. v0.298 fuses
  shared accumulation, `mixer_out` update, and residual add after the shared FFN
  core. Rollback is `QWEN_DECODE_MOE_FUSED_FINALIZER=0`. A3B `tg128` moves
  `103.78 -> 104.2-104.3 t/s`; A10B moves `44.64 -> 44.7-44.9 t/s`. Treat this
  as opportunistic tail-pass cleanup, not a new strategic lane.
- MoE decode phase attribution now times the production FFN apply path instead of
  stale split routed/shared/residual phases. v0.301 `ctx128` rows put MoE FFN
  apply at `2.83 ms` / `28.6%` on A3B and `6.92 ms` / `30.7%` on A10B. This keeps
  MoE FFN execution shape as the live decode branch, while GDN-front retreads stay
  demoted unless fresh roofline evidence contradicts v0.293.
- v0.302 kills a row-group-4 F32 mat-vec widening probe for the route/GDN F32
  projection shelf. The correctness-safe sidecar regressed the mixed MoE route
  bucket (`0.96 -> 1.00 ms` A3B, `1.34 -> 1.43 ms` A10B at `ctx128`). Do not infer
  from the low route active-byte percentage that another F32 mat-vec retile is high
  EV; split route into named subphases first, and keep the production
  `moe ffn apply` split as the next decode attribution gate.
- v0.303 adds the production-wave FFN split gate. At `ctx128`, A3B FFN apply is
  `2.70 ms` aggregate / `1.31 ms` gate-up wave / `1.24 ms` down wave / `0.12 ms`
  finalizer; A10B is `6.89 ms` / `3.75 ms` / `2.62 ms` / `0.14 ms`. With
  concurrent shared disabled for diagnosis, routed FFN is much larger than shared
  core (`2.31` versus `0.76 ms` A3B, `5.86` versus `1.53 ms` A10B). Treat gate/up
  and down mechanics as live; finalizer, route widening, and naive monolith fusion
  are not next.
- v0.304 deep-splits FFN apply into routed/shared subphases. A10B `ctx128` is
  routed gate/up `3.14 ms`, routed down `2.57 ms`, shared gate/up `0.83 ms`, shared
  down `0.70 ms`; A3B is `1.10/1.22/0.46/0.40 ms`. Shared FFN is not the next
  target. The next cheap falsifier is a routed gate/up no-op versus routed down
  no-op inside the production wave schedule; only write a new routed Q4 kernel if
  the corresponding no-op recovers the production wave and end-to-end guard.
- v0.305 runs those no-op oracles. Same-build `tg128` goes A3B `103.71 -> 115.00`
  with routed gate/up no-op and `113.48` with routed down no-op; A10B goes
  `44.86 -> 51.79` and `49.05`. Production-wave phase confirms the mechanics:
  A10B gate/up wave `3.80 -> 0.83 ms`, down wave `2.61 -> 0.68 ms`; A3B gate/up
  `1.23 -> 0.35 ms`, down `1.24 -> 0.42 ms`. Attack routed Q4 gate/up first, down
  second. Do not optimize shared, route, finalizer, or the prior monolith.
- v0.306 kills a narrow routed Q4 gate/up row-shape sidecar (`NR0=1, NSG=4`). It
  was directionally positive (`44.86 -> 45.38 t/s` A10B, `103.71 -> 104.63 t/s`
  A3B) but far below the `+3%` keep gate and below the `15-20%` routed gate/up
  subphase gate. Do not spend the next pass on simple `NR0/NSG` reshaping alone.
- v0.307 adds subphase roofline estimates and re-ranks exact MoE decode work.
  A10B routed gate/up is already `~435 GB/s` (`~92%` of the measured stream
  anchor), while routed down is `~324 GB/s` (`~68%`) and A3B routed down is only
  `~192 GB/s` (`~40%`). Gate/up is still the largest no-op ceiling, but exact
  work should now target routed down unless gate/up has a credible byte-reduction
  or reuse mechanism.
- v0.308 kills Q5 routed-down `NSG=4` widening. A3B down wave regressed to
  `1.29 ms` and A10B to `2.75 ms`, so fewer threadgroups/more simdgroups is not
  the mechanism. The weight-only roofline likely undercounts repeated `moe_inner`
  activation traffic; the next routed-down gate is an inner-load no-op, not a
  full staging kernel.
- v0.309 runs the Q5 down inner-load no-op. It drops down wave to `1.07 ms` A3B /
  `2.21 ms` A10B and `tg128` to `+2.0%` on both targets, so activation replay is
  real but too small for broad staging after overhead. Prefer Q5 down weight/dequant
  slimming; only try inner staging as a tiny S4-stage microproof with a strict
  `>=5%` down-wave and `>=0.8%` A10B `tg128` keep gate.
- v0.310 kills naive Q5 down packed-byte `ulong` loads. Replacing scalar `qh/q1/q2`
  reads with `ulong` loads plus shifts regressed A3B down wave to `1.32 ms` and
  A10B to `3.28 ms`. Do not pursue dequant slimming through wide integer load plus
  shift extraction without a smaller instruction-level microproof.
- v0.311 runs the Q5 down no-weight oracle. It preserves inner loads, top-k,
  weighted accumulation, stores, and production scheduling while replacing
  weight/dequant math with constants. Down wave improves to `1.00 ms` A3B /
  `2.15 ms` A10B and `tg128` moves only `+2.2%/+2.4%`. Q5 down remains real, but
  the removable cost is smeared across several mechanisms; stop drilling local Q5
  down sidecars unless a new mechanism or counter trace isolates a larger term.
  v0.487 also kills the prompt-side all-SG scatter epilogue rewrite for Q5/Q6
  grouped down: correctness is green, but A3B `pp512/pp1024` moves
  `1487.45/1678.78 -> 1484.06/1675.05 t/s`. Do not reopen scalar scatter
  epilogue rewrites without a trace showing the epilogue itself is material.
- v0.312 adds `QWEN_DECODE_TRACE_COUNTS=1` tg JSON accounting and reruns the
  current-default versus `QWEN_DECODE_MOE_CONCURRENT_GDN=0` packet. The default
  remains a real win (`+8.8%` A3B, `+6.0%` A10B in the clean no-trace repeat), but
  warmed trace rows show `~95.8-97.8%` GPU/wall and unchanged dispatch counts
  between default and rollback. The GDN-concurrent win is banked GPU overlap, not
  a fresh CPU/encoder-bubble mandate. Demote short MoE decode scheduling work
  unless a new trace isolates a larger GPU-overlap mechanism.
- v0.313 kills the first structural long-attention sidecar. A four-simdgroup
  cooperative subgroup TG passed correctness and gave an A3B synthetic `ctx4096`
  main-only win, but it was flat at synthetic `ctx16384/32768`, flat/regressive on
  A10B, and regressed full-model A3B `ctx16384` attention (`3.24 -> 3.57 ms`). Do
  not pursue subgroup packing without a new cache/counter signal; the next
  long-attention proof should be KV layout/reader structure, not more subgroup
  shape work.
- Quant breadth is now an active MoE guardrail, not a documentation afterthought.
  v0.219-v0.221 add A3B Q3_K_M, Q6_K, and Q8_0 native grouped routed coverage;
  v0.233 adds `IQ3_S/IQ3_S/IQ4_XS` and moves the local `UD-IQ4_XS` A3B file to
  `40/40` grouped MoE coverage. The v0.237 paired `pp512` re-anchor is now won
  across measured non-sharded local A3B quants: Qwen3.6 `UD-Q4_K_M` `1.05x`,
  Qwen3.5 `Q3_K_M` `1.04x`, `Q6_K` `1.02x`, `Q8_0` `1.03x`, plus the v0.233
  `UD-IQ4_XS` repeat at `1.01x`. `UD-IQ4_XS` also wins at `pp1024` (`1.11x`),
  `pp4096` (`1.12x`), Marcus real rollout (`1.09x`), and one-run `pp16384`
  (`1.19x`).
  Remaining quant risk is no longer this known A3B MoE file; it is BF16 prompt
  performance after v0.249 restored `40/40` grouped MoE coverage, unmeasured
  expert-bank combinations outside the local target set, plus dense UD files with
  `IQ2/IQ3` tensors. The v0.254 BF16 probes falsified two tempting local exits:
  per-layer command-buffer splitting did not improve wall time, and a more
  llama-like separate `NR1=32` BF16 gate/up sidecar was only `~3%` at `pp512`.
  The v0.255 wall no-op budget then re-centered the BF16 cliff on routed MoE:
  bfloat-act `pp512` is `77.00 t/s`, no routed MoE is `825.63 t/s`, no grouped
  SwiGLU is `205.76 t/s`, and no grouped down is `142.44 t/s`. The v0.256
  reduce split says weighted sum is not the missing budget: no grouped reduce is
  flat/noise while no grouped down+reduce still moves to `~92-94 t/s`. The
  v0.257 bucket-bin trace says the expert-projection loss is not uniform:
  `<8` buckets are `6-17x` slower per slot than `>=64` buckets, while hot buckets
  still dominate aggregate SwiGLU at `pp1024`. The remaining BF16 work should be
  a deeper `mul_mm_id`/layout parity probe for routed expert projections, not
  route/reduce/finalizer or another local routed-SwiGLU tile tweak. v0.258
  kills a Q5-style BF16 tiny-down `MR16/NR8` clone; v0.259 kills a hot-only
  separate gate/up `NR1=32` SwiGLU graft. Any remaining BF16 branch must change
  the routed projection graph more deeply, or BF16 should stay demoted behind
  primary-family guardrails.

Prompt-only anchors, release `qwen-bench pp`, synthetic prompts:

- Dense group-4 matrix attention is now default for `head_dim=256` shapes with
  `QWEN_PREFILL_ATTN_MATRIX_G4=0` as rollback. Current 9B rows are `813.36/820.51/
  775.82/690.68 t/s` at `pp512/1024/4096/16384` versus llama.cpp
  `814.10/804.64/693.95/678.06`; 0.8B/2B/4B `pp1024` smokes land at
  `0.98x/1.00x/1.01x` versus llama.cpp.
- 27B dense prefill now uses paired cross-engine rows for scoreboard claims. The
  v0.187 paired comparator puts qwen/lcpp at `236.53/232.41` for `pp512`,
  `222.28/214.08` for `pp1024`, `220.81/212.50` for `pp4096`, and
  `206.27/193.89` for `pp16384`. Treat older isolated `pp1024` and `pp4096`
  rows as stale methodology artifacts unless reproduced by the paired harness.
- `qwen-llm` 35B A3B MoE prompt default now includes prompt-native packed
  attention for the proven `group=8`, `head_dim=256` shape with family-specific
  `NWG=64`, packed activation at `n_pos >= 128`, A3B-sized router `E8xP32`, fused
  route+bucket from `pp128`, and grouped routed down for both `Q5_K` and `Q6_K`
  expert-down layers. Latest post-Q6 rows: `783.1 t/s` at `pp128`,
  `1000.6 t/s` at `pp256`, `1061.5 t/s` at `pp320`, `1172.1 t/s` at `pp512`,
  `1287.4 t/s` at `pp1024`, `1265.5 t/s` at `pp2048`, noisy `1198.1 t/s` at
  `pp4096`, and `674.3 t/s` for the `34,502`-token `v02_reva` full rollout.
- `qwen-llm` 122B A10B MoE prompt default now includes prompt-native group-16
  matrix attention plus the grouped Q5 gate/up layer-46 coverage fix. Warmed Q5
  gate/up rows move rollback/default from `377.53 -> 448.31 t/s` at `pp512`,
  `400.11 -> 513.66 t/s` at `pp1024`, and `440.08 -> 484.54 t/s` at `pp4096`
  (single directional long row). The default now has `48/48` grouped routed MoE
  phase coverage at `pp512`; rollback is `QWEN_PREFILL_MOE_GROUPED_Q5_GATEUP=0`.
- A10B/G16 matrix attention is now the default for the proven group-16 prompt
  shape, with `QWEN_PREFILL_ATTN_MATRIX_G16=0` as rollback. Clean `v0.141` rows
  with `build_dirty=0`, AC power, no thermal/perf warnings, and `96%` free memory
  show `pp1024` `411.86 -> 430.67 t/s` (`+4.6%`) and `pp16384`
  `289.97 -> 338.07 t/s` (`+16.6%`). Warmed dirty rows are also positive at
  `pp512/1024/4096/16384`, and
  a dirty phase trace shows `12/12` G16 matrix attention layers with attention body
  reduced from the old `~73 ms` packed bucket to `~14 ms` total matrix phases at
  `pp512`. A follow-up test-scratch fix makes bare
  `QWEN_PREFILL_ATTN_MATRIX_G16=1` A10B smoke pass without manual
  `QWEN_PREFILL_ATTN_MATRIX_MAX_POS`. A clean current-commit `pp512` trace shows
  `12/12` G16 matrix attention layers, expected routed MoE/GDN counts, and matrix
  body phases totaling `14.53 ms`. The clean repeat packet is positive at
  `pp512/1024/4096/16384`; paired same-session llama.cpp default anchors put G16
  at `0.90x/1.02x/1.08x/1.03x`. Active G16 matrix correctness is green under the
  default auto policy when the packed threshold is lowered for the smoke. Routed
  MoE is next because `pp512` remains a lcpp gap.
- Post-default A10B `pp512` no-op budget confirms that pivot: base/default G16 is
  `383.26/382.87 t/s`, no-attention-body is only `389.39/390.08`, and no-routed-
  MoE is `764.19/764.46`. Stop opening attention branches for A10B until routed
  `SwiGLU/down` has had a fresh structural pass.
- The first routed-MoE structural pass found the same class of issue as A3B Q6
  down: A10B layer `46` escaped the grouped path because gate/up are `Q5_K`.
  Grouped Q5 gate/up SwiGLU fixes coverage and is now the A10B default candidate;
  the remaining exact A10B work should attack grouped down/dequant locality, not
  route/reduce/finalizer.
- Static fast-path audit is now part of the workflow. Current target coverage is
  `A3B 40/40` grouped MoE, `A10B 48/48` grouped MoE, and dense `27B 64/64` FFN +
  `48/48` GDN + `16/16` attention. The dense/GDN/attention/lm-tail predicates
  now use every dtype with primitive mat-mat support (`F32`, `F16`, `BF16`,
  `Q2_K`, `Q3_K`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `Q4_0`, `Q4_1`, `Q4_K`, `Q5_K`,
  `Q6_K`, `Q8_0`, `IQ4_NL`, `IQ4_XS`), so the local 0.8B quant family is clean
  across dense FFN, GDN, attention, and lm-tail coverage. MoE grouped coverage
  includes the target A3B
  Q3/Q4/Q6/Q8 and `UD-IQ4_XS` expert-bank combinations. Q2_K/Q3_K/IQ4_NL/IQ4_XS now have
  simdgroup_matrix prompt mat-mat tiles: clean local 0.8B low-bit rows move from
  `0.15-0.26x` llama.cpp at `pp1024` to `0.96-0.98x` across `pp1024/4096`.
  The v0176 dense validation generalizes this to 2B/9B Q2/Q3/IQ4_XS and 27B Q3:
  2B is `0.97-1.00x`, 9B is `0.99-1.06x`, and 27B Q3 `pp1024` is `1.08x`
  llama.cpp. Remaining explicit coverage gaps are MoE grouped expert-bank
  variants outside target quants and unmeasured UD low-bit dense tensors. The
  v0.240 dense `IQ2_S`/`IQ3_*` matrix-kernel pass closes the measured 4B UD
  low-bit prompt cliff: local `UD-Q2_K_XL` and `UD-IQ2_M` are now parity/wins at
  `pp512/1024/4096`. v0.241 closes the measured decode side too, with clean
  `tg128` rows at `1.13-1.15x` pinned llama.cpp. v0.242 adjacent 4B breadth is
  clean: `Q3_K_M`, `IQ4_XS`, and `Q4_K_M` are parity/win at `pp512/4096`, and
  all three win at `tg128`. Older
  A3B low-bit context: clean v0.189 A3B Q3 `pp1024` paired
  evidence is `90.28` qwen versus `1392.02` llama.cpp (`0.065x`), while no-FFN
  jumps to `2858.64 t/s`. The v0.192 env-gated grouped F32/IQ4_XS candidate
  (`QWEN_PREFILL_MOE_GROUPED_F32_GATEUP=1`) moves A3B Q3 to dirty paired
  `1191.03 t/s` at `pp1024` and `1225.51 t/s` at `pp4096`, but still trails
  llama.cpp by `0.856x/0.894x`. The v0.193 native `IQ3_XXS` candidate
  (`QWEN_PREFILL_MOE_GROUPED_IQ3_GATEUP=1`) adds direct matvec and grouped-SwiGLU
  oracles, then reaches clean paired `1486.94-1490.63/1383.51-1384.73`,
  `1532.84-1533.85/1367.76-1369.89`, and `1311.75/1163.09 t/s` at
  `pp1024/4096/16384`. It is still not defaulted. The v0.195 defaultability pass
  proved the T40 blocker is GDN/Q8 sensitivity rather than native IQ3 MoE. The
  v0.196 pass then found and fixed a separate G8 matrix-attention VT-coverage bug
  that only appeared when small chunks crossed the matrix threshold mid-call. With
  blk0 `qkv+alpha` GDN matvec as a repair oracle, Q3 T128/P32 now passes and dirty
  paired rows remain above llama.cpp at `pp1024/4096/16384`
  (`1.023x/1.034x/1.108x`). The v0.197 continuation-gate pass reclassified strict
  long internal GDN/KV cosine as diagnostic by default for multi-token probes:
  Q3 and Q4 awkward-chunk controls show no argmax divergence over 64 oracle-greedy
  continuation tokens despite sub-`0.999` internal or continuation cosine. The
  v0.199 real-prompt packet keeps Q3 native-IQ3 + blk0 `qkv+alpha` above
  same-length llama.cpp anchors on Reva short, Mei medium, and Marcus long
  (`1.018x/1.014x/1.022x`), but also proves exact long greedy parity is the wrong
  release gate: Q4 default mismatches over 64 continuation tokens at T1024 too.
  The v0.201 promotion packet keeps the Q3 native-IQ3 + blk0 `qkv+alpha` branch
  positive on repeated real-prompt script rows: Reva short retained pairs are
  `1.055x/1.025x`, and Marcus long is `1.047x` after discarding block 0.
  The next low-bit MoE branch is a top-k/rank-envelope defaultability policy, then
  either default native IQ3 or build a high-accuracy blk0 GDN projection kernel if
  rank escapes are materially worse than the incumbent envelope.
  Decode sentinels also matter: Q2_K and IQ4_XS `tg128` were `0.72x/0.70x`
  before v0.177 and are now `1.09x/1.22x`. Q3_K_M now has a native row-reuse
  fast mat-vec kernel and moves from `0.94x` to `1.31x` on 0.8B, with 2B/9B/27B
  one-run sentinels at `1.10x/1.12x/1.13x`. IQ4_NL also has a native row-reuse
  fast mat-vec kernel and moves from the prior `~0.97x` residual to `1.25x` on
  the only local IQ4_NL file. Clean post-commit 0.8B decode anchors now put
  Q2/Q3/IQ4_NL/IQ4_XS/Q4_K_M at `1.36x/1.30x/1.28x/1.26x/1.26x` llama.cpp.
  The measured dense low-bit local decode family is now a win; stop harvesting
  this lane until a primary re-anchor or larger quant file exposes a real miss.
- A10B routed MoE is no longer the active top bet after the warmed re-anchor. A
  default-warmup `pp512` phase trace has qwen timed-pass
  `routed_swiglu+routed_down = 600.52 ms` versus llama.cpp profile
  `ffn_moe_gate+up+GLU+down = 654.98 ms`, and current end-to-end anchors are
  qwen/lcpp `446/440` at `pp512`, `488/436` at `pp1024`, `481/401` at `pp4096`,
  and `394/354` at `pp16384`. Keep A10B MoE in monitoring unless a fresh matched
  trace shows a real warmed routed-tail deficit or coverage drops below `48/48`.
- Dense 27B prefill is now paired-won across the measured synthetic shapes after
  the Q4_K `N=64` prompt mat-mat tile and the v0.187 comparator pass. Rollback is
  `QWEN_MATMAT_Q4_K_N64=0`. The key lesson is methodological: stale cold llama
  anchors and unpaired qwen drift made `pp1024/4096` look worse than they are.
  Do not reopen dense 27B short/medium work without a paired residual.
- The first dense FFN/GDN pass is phase-positive but not scoreboard-complete by
  itself.
  `QWEN_PREFILL_TRACE_FFN_SUBPHASES=1` plus timed-only summaries showed the FFN
  residual was mat-mat throughput, not SwiGLU epilogue or chunk policy. Adding
  llama.cpp-style full-unroll pragmas to Q4/Q5/Q6/Q8 mat-mat kernels moves 27B
  timed `pp4096` gate/up/down from `4063/4068/4199 ms` to `3959/3964/4090 ms`,
  near the lcpp `3878/3932/4099 ms` buckets; GDN QKV/Z/back also drop `2-4%`.
  Post-commit clean rows on AC power are 27B `pp512=234.38`, `pp1024=226.63`,
  `pp4096=207.34`, `pp16384=193.41`, plus A10B `pp1024=499.83` and A3B
  `pp1024=1584.48` smokes.
  A same-principle GDN recurrence-loop unroll was falsified and removed
  (`gdn_step` `593.40 -> 597.17 ms` at `pp4096`).
  A timed-pass v0.157 differential makes attention body the next active branch:
  at `pp4096`, FFN and GDN projections are parity-or-faster while attention body
  is `~115 ms` slower than llama.cpp and GDN step is `~82 ms` slower; at
  `pp16384`, attention body is `~0.85 s` slower, GDN step `~0.36 s`, and FFN
  `~1.0 s`. No-op ceilings are large (`pp16384` attention body `194.35 ->
  217.23 t/s`, GDN body `194.35 -> 215.10`), but the first generic attention
  loop-unroll probe was negative and removed.
- A first exact attention-body cleanup is now default for the 27B G6 matrix path:
  causal-tail KQ/KQV tile skip with `QWEN_PREFILL_ATTN_MATRIX_CAUSAL_SKIP=0`
  rollback. It improves 27B `pp4096` by about `0.5%` and `pp16384` by about
  `0.55%` in randomized A/B, with clean post-commit anchors at `pp4096=208.06`
  and `pp16384=194.80`; it is neutral/noisy at `pp512/1024` and passes the default
  plus long-prefix 27B correctness gates. This does not close the llama.cpp gap;
  it just removes avoidable future-tile work from the current matrix body.
- Current same-shape A3B rows against recent `llama.cpp` anchors changed sharply
  after the Q6-down grouped fix and fresh same-session lcpp anchors: `pp320` is
  now `1061.45 / 1174.57 t/s` (`0.90x`), `pp512` is `1172.07 / 1347.79 t/s`
  (`0.87x`), `pp1024` is `1287.43 / 1345.07 t/s` (`0.96x`), `pp4096` is
  `1198.11 / 1259.21 t/s` (`0.95x`), and `pp34502` is `674.28 / 865.50 t/s`
  (`0.78x`). `pp16384` needs a post-Q6 rerun. `llama.cpp` also declines at true
  long context; medium A3B is now close, while true-long remains the largest A3B
  prefill gap.
- The existing env-only A3B/group-8 matrix-attention sidecar composes with the Q6
  fix and reaches/surpasses those lcpp anchors in spot rows: `1190.45 t/s` at
  `pp320`, `1340.82 t/s` at `pp512`, `1484.98 t/s` at `pp1024`, `1434.39 t/s` at
  `pp4096`, `878.72 t/s` at synthetic `pp34502`, and `870.94 t/s` on the real
  `v02_reva` `34,502`-token rollout. It remains env-only because max-pos scratch
  policy and matrix correctness tolerance still need productionization.
- First max-pos productionization step is done: `qwen-bench pp`, `pp-wait`, and
  packed `decode` prefill now size matrix scratch from the actual prompt length
  when `QWEN_PREFILL_ATTN_MATRIX_G8=1`, so user-facing bench paths no longer need
  `QWEN_PREFILL_ATTN_MATRIX_MAX_POS`. Auto-scratch spot rows are stronger:
  `1237.87/1390.93/1561.06/1492.39/928.62 t/s` at
  `pp320/512/1024/4096/34502`.
- The expanded clean family sweep at
  `docs/bench/2026-05-24-1331-matrix-pp4k16k-family/` ran
  `QWEN_PREFILL_ATTN_MATRIX_G8=1`, `runs=1`, and
  `pp128/512/1024/4096/16384` plus `tg32/tg128`. A3B now matches/beats lcpp at
  every swept prompt size in that candidate branch: `879/769` (`1.14x`) at
  `pp128`, `1387/1392` (`1.00x`) at `pp512`, `1558/1388` (`1.12x`) at `pp1024`,
  `1492/1352` (`1.10x`) at `pp4096`, and `1212/1107` (`1.09x`) at `pp16384`.
  Do not generalize this to the whole family: dense 27B is still only `0.67x` at
  `pp16384`, and A10B is `0.83x/0.92x/1.00x/0.86x` at
  `pp512/1024/4096/16384`.
- The same family sweep makes prompt chunk policy a first-class hypothesis: every
  qwen row drops from `pp4096` to `pp16384` under the current default
  `prefill_chunk=1024`. Some decline is not itself a bug because llama.cpp also
  falls at 16K in this run, but qwen's dense/A10B slope is now a likely whole-
  family blocker.
- A targeted clean chunk sweep falsifies chunk size as the dense 16K cure but
  keeps it live for MoE. Dense 27B `pp16384` is flat/slightly worse at chunk
  `2048` (`125.09 -> 124.52 t/s` repeated), while A3B `pp16384` improves at
  chunk `2048` (`1114.61 -> 1148.52 t/s`) and A10B improves at chunk `4096` both
  at `pp4096` (`357.87 -> 378.40 t/s`) and `pp16384` (`288.99 -> 306.46 t/s`).
  Treat `2048` as the conservative cross-MoE default-cap candidate; treat `4096`
  as an A10B-specific candidate until A3B/real-rollout gates say otherwise.
- The clean repeated A3B matrix promotion gate at
  `docs/bench/2026-05-24-1934-35B-A3B-matrix-promotion-repeat-family/` keeps the
  branch above lcpp at `pp128` (`1.10x`), `pp1024` (`1.11x`), `pp4096` (`1.10x`),
  and `pp16384` (`1.07x`), with `pp512` at parity (`0.99x`) and decode `tg32/tg128`
  both `1.13x`. A follow-up chunk-2048 interaction gate shows why the chunk cap
  must be prompt-length gated: `2048` helps A3B at `pp4096` (`1.06x` over default)
  and barely at `pp16384` (`1.01x`), but regresses `pp128` and `pp512`.
- The group-8 matrix sidecar has now been generalized to runtime group-shape args
  and an env-only 27B dense group-6 gate (`QWEN_PREFILL_ATTN_MATRIX_G6=1`). The
  clean `v0.120` repeat gate in
  `docs/bench/2026-05-24-dense-g6-clean-repeat-v0120/` keeps the win across every
  swept prompt size: `pp128` `188.33 -> 198.42 t/s` (`1.05x`), `pp512`
  `195.85 -> 210.99` (`1.08x`), `pp1024` `175.66 -> 188.60` (`1.07x`), `pp4096`
  `151.87 -> 185.85` (`1.22x`), and `pp16384` `125.21 -> 173.84` (`1.39x`). This
  is now a dense promotion candidate, not just a dirty spike, but long-prefix G6
  correctness coverage is still thin.
- The `v0.122` dense family gate in
  `docs/bench/2026-05-25-0246-27B-matrix-g6-v0122-family/` adds that long-prefix
  G6 correctness coverage and compares against fresh lcpp anchors. Dense 27B is
  now near parity but not beaten: `pp128` `198/213` (`0.93x`), `pp512` `211/222`
  (`0.95x`), `pp1024` `202/205` (`0.99x`), `pp4096` `186/198` (`0.94x`), and
  `pp16384` `176/188` (`0.93x`). Decode remains won at `tg32/tg128` (`1.13x` and
  `1.12x`). Matrix-G6 is therefore a real default candidate but not the final
  dense answer.
- With matrix-G6 enabled, fresh no-op rows move the dense residual priority to
  FFN: at `pp4096`, no-FFN is `180.50 -> 541.59 t/s` while no-GDN/no-attn are only
  `193.30/193.93`; at `pp16384`, no-FFN is `173.12 -> 421.55` while no-GDN/no-attn
  are `192.37/192.25`. A llama-like threadgroup pointer-store spelling for
  Q4/Q5/Q6/Q8 mat-mat is correctness-green and gives only small dirty-spike rows
  (`188.77 t/s` at `pp4096`, `176.13 t/s` at `pp16384`), so the high-EV dense work
  remains deeper FFN mat-mat/layout/fusion evidence, not more local attention.
- A high-N dense fused-Q4 SwiGLU spike (`QWEN_PREFILL_DENSE_FFN_FUSED_SWIGLU_Q4=1`)
  validates the mechanism but not default readiness: with matrix-G6 on, dirty
  rows are flat/slightly down at `pp512` (`214.86 -> 214.45`) and noisy down at
  `pp1024` (`196.97 -> 193.45`), while `pp4096` improves in one noisy pair
  (`181.08 -> 191.67`) and `pp16384` improves slightly (`177.57 -> 179.25`). Keep
  it env-only; the remaining dense gap still needs a bigger FFN mat-mat/layout win.
- Clean `v0.125` repeats confirm the long-row fused-Q4 SwiGLU branch but also its
  limited ceiling: `pp4096` `182.05 -> 191.62 t/s` (`1.05x`) and `pp16384`
  `177.22 -> 179.24` (`1.01x`). It is a useful long-prompt candidate, not enough
  to crack lcpp, and the `pp512/pp1024` dirty rows block broad defaulting.
- A llama-parity smem cleanup for Q4/Q5/Q6/Q8 mat-mat is worth keeping but not
  over-reading. Dirty microbench rows are flat at `N=512`, clearly positive at
  `N=1024` (Q4 gate/up `19.812/17.617 -> 14.567/14.853 ms`, Q6 down
  `19.733 -> 16.119 ms`), and modest at `N=4096`, but a clean v0.127 matrix-G6
  end-to-end A/B does not prove a default win (`pp4096` new/legacy/new =
  `189.00/204.65/204.51`, `pp16384` `190.83/191.68/188.44`). Keep legacy 8192B
  requests as default; use `QWEN_MATMAT_QK_LLAMA_SMEM=1` only as an opt-in
  diagnostic while hunting larger FFN/GDN execution gaps.
- A follow-up F16-inner/Q6-F16-source dense FFN spike is falsified. Correctness was
  clean, but cooled rows show no stable end-to-end win (`pp4096` F16-inner
  `202.33` vs fused-Q4 `201.66`, `pp16384` `180.59` vs fused-Q4 `182.00`) and a
  direct Q6_K mat-mat microbench shows F16 source is slower (`ffn_down` `61.604 ms`
  F32-source vs `64.000 ms` F16-source at N=4096). Do not carry or retune this
  branch unless future profiling proves source-read bandwidth has become the wall.
- The first drift-controlled `v0.129` dense FFN rows keep both env candidates out
  of the default path. With matrix-G6/G8 on, warmed `pp4096` rows are only
  `g6=208.96`, `g6-smem=209.73`, `g6-fused=210.98`, and
  `g6-fused-smem=211.09`; `pp16384` is directionally inconsistent
  (`g6=193.46/181.14`, `g6-fused=189.70/193.60`). Treat the stable signal as
  `~1%` and below the default gate. The next dense move is phase-local FFN/GDN
  evidence, preferably paired in-process, not another total-throughput default
  decision on a sub-noise row.
- The same-process fused-SwiGLU A/B harness now exists, and it falsifies broad
  defaulting despite one exciting long-context outlier. Fused loses paired
  `pp4096` and `pp8192`; `pp16384-a` looks large-positive but the immediate
  `pp16384-b` repeat is flat. Keep the branch as an env-only long-context clue,
  not a default candidate, until a repeated same-process gate and phase-local
  FFN mechanism agree.
- A serialized qwen-vs-llama.cpp dense `pp4096` differential moves the next dense
  target away from standalone SwiGLU fusion. Qwen's split FFN buckets are only a
  few percent slower (`gate/up/SwiGLU` `8408` vs `8106 ms`, `down/resid` `4310`
  vs `4200 ms`), while GDN front projections are the larger local delta (`3613`
  vs `2998 ms`) and matrix attention body still has a smaller residual delta.
  Next dense work should inspect GDN front lowering and lcpp op shapes before
  another FFN fusion branch.
- That GDN-front inspection found one real dispatch-class mismatch: dense GDN
  `beta_proj` / `alpha_proj` are F32 skinny projections that benefit from the
  router-style E8xP32 kernel. Default-on `QWEN_PREFILL_GDN_SKINNY_E8P32` drops
  traced `gdn_beta_alpha` from `672.45 -> 105.96 ms` at 27B `pp4096` and improves
  repeated 27B matrix-G6/G8 rows by about `+2-4%` at `pp512/1024/8192/16384`
  (`pp4096` warmed is only `~+0.8%`). Correctness is green on 0.8B, active 27B
  prefill, 27B matrix-prefix, and A3B prefill gates. The promotion gate also found
  positive A3B `pp128/1024` and matrix `pp4096` canaries. Roll back with
  `QWEN_PREFILL_GDN_SKINNY_E8P32=0` if a future F32 GDN shape regresses.
- A 27B `pp16384` combined no-op budget confirms the dense gap is not only
  attention: baseline `127.57 t/s`, no-FFN `208.71`, no-attn `193.48`,
  no-FFN+no-attn `580.91`, and no-FFN+no-attn+no-GDN `845.88`. Keep dense FFN/GDN
  mat-mat work queued after the G6 matrix clean gate, not instead of it.
- `llama-bench -fa 0` is not auto for the bench tool: it disables flash attention.
  A3B `llama.cpp` `-fa 0` and `-fa 1` are flat at `pp1024` and `-fa 1` is slightly
  slower at `pp16384`, so the current A3B long target is the non-flash
  `KQ -> softmax -> KQV` Metal path and its cache/layout implementation details.
- An env-only A3B/group-8 matrix-attention sidecar
  (`QWEN_PREFILL_ATTN_MATRIX_G8=1`) became a real long-context branch after V_T
  writes moved to fused cache-fill time and the KQ/KQV B tile started using
  lcpp-like vector loads: spot rows are about `1042 t/s` at `pp4096`, `997 t/s`
  at `pp8192`, noisy `~864-904 t/s` at `pp16384`, and `725.7 t/s` at `pp34502`
  with chunk `1024`. A true-long chunk probe puts `pp34502` chunk `2048` at
  `736.9 t/s` and chunk `4096` at `718.2 t/s`. It is still not defaulted because
  allocation is manual via `QWEN_PREFILL_ATTN_MATRIX_MAX_POS` and KQV uses the
  looser half-probability correctness tolerance.
- Fresh `llama-bench` A3B anchors on the current local build are `1174.6 t/s` at
  `pp320`, `1347.8 t/s` at `pp512`, `1345.1 t/s` at `pp1024`, `1259.2 t/s` at
  `pp4096`, and `865.5 t/s` at `pp34502` (`-fa 0`, `has tensor = false`). A10B
  `pp320` still needs a fresh same-session lcpp rerun.
- prior repeated-prompt `qwen-llm` 27B dense packed prefill: `~205.4-205.9 t/s`
- current `llama.cpp` bounded `llama-cli -st` baseline: `~206.7 t/s` prompt,
  `~22.5 t/s` generation

Recent confirmed wins:

- Dense GDN front-projection overlap is now production-wired for logits and GPU
  argmax decode. The old bench-only path becomes the dense default in v0.340 and
  lifts `tg128` across measured dense Q4 models by `+2.4-3.8%`; rollback is
  `QWEN_DECODE_DENSE_CONCURRENT_GDN=0`. This banks scheduling overlap, not a new
  local Q8 mat-vec lane.
- A3B had a silent optimized-path escape: three late routed down-expert layers
  (`blk.34`, `blk.38`, `blk.39`) are `Q6_K`, while grouped routed prefill only
  accepted `Q5_K` down. Adding grouped Q6_K down and dtype dispatch moves trace
  coverage from `37/40` to `40/40` grouped routed layers and lifts A3B `pp320`
  `~830 -> 1061.45 t/s`, `pp512` `824.46 -> 1172.07 t/s`, `pp1024`
  `910.43 -> 1287.43 t/s`, and same-fixture `34,502`-token real rollout
  `587.60 -> 674.28 t/s`. This is now the canonical
  example for why fast-path coverage must be asserted, not inferred from logits.
- Prompt-native packed MoE attention is now a production default for the proven
  long/medium prompt shapes, not an experiment. The important details are now
  known and banked:
  - family-specific packed `NWG` (`g8=64`, `g16=32`)
  - family-specific packed activation thresholds (`g8 >= 128`, `g16 >= 320`)
  - per-layer packed-vs-old oracle green at the first newly activated prompt
    sizes (`pp128/pp320` for A3B, `pp320/pp512` for A10B) and at active
    long-context chunk shapes.
- A3B-sized route-logits `E8xP32` and fused route+bucket now activate from
  `pp128`; the old `512` floors left cheap medium/short-prompt wins on the table.
  A10B stays conservative at `512` for route work until cooled promotion sweeps
  resolve the mixed signal there.
- The medium-prompt board moved substantially once the packed-attention threshold
  dropped from `4096` into the medium-prompt regime and the A3B route threshold
  followed: A3B `pp128/pp320/pp512/pp1024` now lands around
  `~655 / ~830 / ~899 / ~952-966 t/s`, and A10B
  `pp320/pp512/pp1024` around `~276 / ~355 / ~418 t/s`.
- The true-long A3B board is now separated from medium-prompt work: qwen synthetic
  and real `34.5k` prompts agree (`596.88` vs `587.60 t/s`), and no-op attribution
  says attention-body cost is the long-context lever (`pp16384` `767.64 ->
  1101.72 t/s` with attention body skipped; routed-MoE skip only reaches
  `913.73 t/s`).
- The A3B matrix-attention sidecar found a real lcpp-like mechanism: V must be
  written in KQV-ready transposed layout at cache-fill time. Fused V_T scatter
  turns the prior long regression into a `~10-18%` spot win from `pp1024` through
  `pp34502`, but the remaining gap is now KQ/KQV/score traffic and mature
  `mul_mm_f16_f32` behavior.
- The next lcpp-like `mul_mm_f16_f32` detail, vector-loading the F32 B tile into
  `half2x4`, is now in the matrix sidecar. It improves the current true-long
  `pp34502` row to `725.7 t/s` at chunk `1024` and `736.9 t/s` at chunk `2048`,
  but `pp16384` is still noisy and chunk `4096` loses. Treat chunk `2048` as the
  current matrix-long candidate, not a default.
- After the Q6 grouped-down fix, that same matrix sidecar becomes the first A3B
  prefill branch to crack lcpp spot rows from medium through true-long. The next
  work is productionization and repeated promotion gates, not more proof that the
  mechanism exists.
- The matrix sidecar no longer needs manual max-pos env sizing in `qwen-bench`
  prompt paths; remaining promotion blockers are cooled repeatability, default
  activation policy, scratch budget accounting, and the correctness tolerance
  decision.
- Post-matrix `pp4096` no-op budgeting changes the next-priority read: with
  matrix attention held fixed, attention-body skip is only a `~3-7%` lever
  (`929/972 -> 997 t/s`), while routed-MoE skip reaches `1270 t/s` and broad
  FFN skip reaches `2498 t/s`. Direct KQV stores, vectorized KQV final copies,
  and F16 probability scratch were all falsified as local attention keepers.
  The next exact sprint should target routed FFN/MoE structure again.

- Grouped MoE routed prefill had a real correctness bug: grouped `Q4_K` SwiGLU
  used `u32::MAX` as an open-ended expert-count sentinel while the Metal kernel
  cast it to signed `int`, turning the bound into `-1` and early-returning active
  experts. After fixing the sentinel, A10B smoke / boundary and A3B
  prefill-vs-single gates recovered, the fused route+bucket oracle became clean,
  and the routed MoE prompt story had to be re-based.
- That corrected re-baseline changes the active MoE path ranking:
  - `route_bucket` itself is small,
  - routed `grouped_swiglu` is still the largest MoE prompt bucket,
  - router logits are the next meaningful routed cost.
- The first post-fix routed-compute attack that converts end-to-end is a GPU-owned
  hot-expert grouped-Q4 split over the existing per-expert `counts/ids` ledger.
  Combined with fused route+bucket and allowlisted at `chunk_p >= 512`, it moves
  A10B from about `297.0 -> 302.9 t/s` at `pp512` and `299.6 -> 309.3 t/s` at
  `pp1024`, and A3B from about `736.3 -> 744.9 t/s` at `pp512` and
  `754.7 -> 776.7 t/s` at `pp1024`.
- A router-only `F32` `E8xP32` kernel now removes most of the remaining router
  logits cost on the proven MoE prompt shapes. Corrected A10B `pp512`
  `route_logits` falls from `5.95 ms` to `0.47 ms`, and A10B `pp1024`
  `route_logits` is `1.03 ms`. Allowlisted with the existing composed MoE prompt
  path at `chunk_p >= 512`, it moves A10B from about `302.9 -> 315.5 t/s` at
  `pp512` and `309.3 -> 324.7 t/s` at `pp1024`, and A3B from about
  `744.9 -> 797.6 t/s` at `pp512` and `776.7 -> 815.1 t/s` at `pp1024`.
- The clean post-`v0.100` family sweep materially updates the MoE prompt picture:
  A10B is now about `0.72x` at `pp512` and `0.76x` at `pp1024`, while A3B is
  still much farther behind at about `0.55x` / `0.58x`. That keeps MoE prefill as
  the main scoreboard gap even though the current exact grouped path is much
  better than the old baseline.
- Post-`v0.100` exact local grouped-MoE search-space elimination is now large:
  `F16` grouped inner default, cold `n8`, hot `th32` rollout, hot grouped-down
  atomic accumulate, persistent/locality queues, hot `32x32`, active hot tile
  lists, and a resident paired gate/up mirror all failed to produce a default-
  worthy win on this hardware / repo shape. Treat the exact local
  `grouped_swiglu` variant family as plateaued until a new mechanism is found.
- Grouped routed `inner/out` zero-fill is now correctness-covered as a real but
  small cleanup lever (`~0.5-0.8%` on the current prompt guardrails), not a main
  roadmap item.
- A bounded exact MoE prompt concurrency branch is now real: overlapping the live
  grouped routed tail with the live shared FFN at `chunk_p >= 512` preserves the
  current correctness matrix and converts to about `1-3%` end-to-end on `pp512`,
  but fades by `pp1024`. This is the strongest near-term exact production branch,
  not the main structural MoE answer.
- Real-rollout and matched-token synthetic ladders now agree on a stronger story:
  A3B long-prompt collapse is primarily an attention/context-growth problem, not a
  prompt-template artifact and not mostly routed MoE compute. Attention no-op on
  long A3B prompts lifts throughput by roughly `4-5x`, while routed MoE no-op is
  much smaller and mostly constant per token.

- MoE decode now has a real GDN-side concurrency win. Reusing the dense
  concurrent-GDN front-projection split inside MoE decode and making it the repo
  default with `QWEN_DECODE_MOE_CONCURRENT_GDN=0` as a rollback path moves A3B
  from `~74.0/73.6 t/s` to `~77.8/78.0 t/s` at `tg32/tg128`, and A10B from
  `~32.4/32.5 t/s` to `~35.1/34.9 t/s`. At `ctx=4096`, decode-window also moves
  A3B `66.6 -> 71.5 t/s` and A10B `31.4 -> 34.0 t/s`, with GPU time improving on
  every measured row.
- A10B `pp128` cold variance is now explained as expert-bank first-touch, not as
  a steady-state runtime miss. Bench-only `QWEN_PP_WARM_MOE_BANKS=1` and
  `QWEN_PP_RESIDENCY_SET=1` collapse the cold outliers while leaving steady-state
  `pp320/pp512` flat, so treat them as benchmark methodology knobs rather than a
  top runtime roadmap item. v0.261 re-confirms this on the current stack:
  default paired `pp128` loses from cold outliers, while warm-bank paired `pp128`
  wins pinned llama.cpp at `1.108x`.

- `qwen-bench pp` now exposes a phase-matched prompt-only harness and lowering
  summary for dense/MoE prompt work. It confirmed the dense `pp320` miss is real
  (`~710/824 t/s` on 9B, `~211.9/240.9 t/s` on 27B) and uncovered the largest
  MoE issue: Q8 mixer projections were falling through to decode-shaped
  per-token paths.
- Enabling `Q8_0` packed mat-mat eligibility for MoE GDN/attention projections
  moves A3B pp320 from `~96 t/s` to `~194-198 t/s` and A10B pp320 from
  `~37.6 t/s` to `~85.1 t/s`, while dense 9B/27B guardrails stay flat.
- Activating the packed routed expert tail after fixing its production dtype gate
  and Q4_K byte layout moves A3B pp320 to `~261.5 t/s` and A10B pp320 to
  `~106.8 t/s`. The win holds across prompt lengths: A3B `pp64..pp1024` stays
  `~249-261 t/s` vs fallback `~188-197 t/s`; A10B `pp128..pp512` stays
  `~105-107 t/s` vs fallback `~84-85 t/s`.
- Batching the shared-expert branch plus rowwise shared gate/residual is another
  major MoE prompt win: A3B pp320 moves to `~399.8 t/s` and A10B pp320 to
  `~148.9 t/s`; default chunk-128 pp320 is `~375.9 t/s` on A3B and
  `~150.4 t/s` on A10B. Dense 9B/27B guardrails remain flat.
- A fully GPU-owned grouped expert-major routed backend now lands as the current
  production-resolution MoE prompt path for the proven `Q4_K/Q4_K/Q5_K`
  envelope. Default chunk-128 pp320 rises to `~556.4 t/s` on A3B and
  `~218.1 t/s` on A10B; chunk320 reaches `~710.4 t/s` / `~278.7 t/s`. Dense
  9B/27B guardrails stay flat.
- Packed attention-body cleanup now batches consecutive-position RoPE for Q/K and
  scatters the whole chunk's K/V rows into the cache in one dispatch before the
  per-token attention loop. On the repeated 320-token 27B prompt, packed prefill
  moves from ~201.5-202.3 t/s to `~205.4-205.9 t/s` and trims total wall from
  ~1582-1588 ms to ~1554-1558 ms.
- Fresh local `llama-cli -st` checks now show dense user-facing parity is real:
  `llama.cpp` is about `206.7 t/s` prompt / `22.5 t/s` generation on the same
  repeated prompt, while `qwen-llm` is already `~205.6 t/s` on prompt-only runs.
- Fresh local `llama-bench` still says the harder pure-prompt target is much
  higher: `pp320 ~240.9 t/s`. That keeps prompt-only parity, not CLI parity, as
  the main scoreboard target.
- Packed dense GDN prep is now over the whole prompt chunk. The old packed GDN
  prep loop used `P` launches of `ssm_conv_silu`, two L2 norms, and three
  scatters before the packed recurrence; the new path replaces that with one
  packed prep kernel plus two in-place batched L2 norms. On the repeated
  320-token 27B prompt, packed prefill moves from ~186.3 t/s to
  `~201.5-202.3 t/s` and trims total wall from ~1718 ms to ~1582-1588 ms.
- The new `QWEN_PREFILL_GDN_SPLIT` diagnostic confirmed the old packed GDN prep
  loop was the dominant GDN sub-bucket before the rewrite: about `~146 ms` wall
  / `~131 ms` GPU on the repeated prompt, versus only `~44 ms` for the packed
  recurrence itself.
- Production-shape packed-prefill profiling is now live in `qwen-bench decode`:
  prompt-only runs report total prefill GPU ms, and profiling no-op flags can
  remove dense FFN, GDN body, or attention body inside the real packed prefill
  graph.
- Packed dense GDN `rmsnorm_gated` is now batched over the whole prompt chunk
  instead of one dispatch per token. On the repeated 320-token 27B prompt, this
  moves packed prefill from ~183.1 t/s to ~186.2-186.5 t/s and trims GPU total
  from ~1717 ms to ~1690 ms.
- Latest same-prompt dense read is now `~205.6 t/s` on 27B, which cuts the fresh
  `llama-cli -st` prompt gap down to roughly three percent even though the harder
  `llama-bench` pure-prompt gap still remains.
- Production-shape dense prompt no-op profiling says the remaining packed-prefill
  cost is real GPU work. After the packed attention-body cleanup, the old
  attention-body no-op delta falls from about `~162 ms` wall / `~156 ms` GPU to
  roughly `~120 ms` wall / `~118 ms` GPU, leaving the remaining GDN tail /
  out-proj path as the clearest non-FFN prompt target. That keeps isolated FFN
  mat-mat kernels exonerated as the hidden prompt mystery.
- Prompt-prefill scratch now skips the unused `[P, V]` logits pack on no-spec
  prompt paths, removing a large dead allocation from timed prefill and nudging
  dense 27B prompt throughput to ~173.3 t/s.
- Packed dense `gdn_step_decay` over prompt tokens is now live and materially
  improves dense prompt processing: the same-prompt 27B plateau rises from
  ~165.0 t/s to ~172.9 t/s while decode stays unchanged.
- Dense packed GDN `alpha/beta` batching was a major prompt win: same-prompt 27B
  prefill rose from ~141.9 t/s plateau to ~165.0 t/s plateau, with decode
  unchanged.
- Dense packed-prefill GDN-tail attribution is now in place and points at the
  true `step_decay` recurrence as the sharpest next dense tail target.
- Dense packed prefill chunk tuning was a major win: on the same 321-token 27B
  prompt, moving from inherited `P=16` to dense default `P=256` improved prompt
  throughput from ~77.9 t/s to ~140.7 t/s with decode unchanged; one-chunk
  saturation is ~141.9 t/s once `P >= 321`.
- Dense packed prefill is now the default no-spec path in `qwen-bench decode`
  for dense models; on a 321-token 27B prompt it improved prefill from 24.3 t/s
  to 78.1 t/s (~3.22x) with decode unchanged.
- MoE packed prefill stage 1 is now live in `qwen-bench decode`; on a 321-token
  prompt it improves prefill from 74.8 -> 91.3 t/s on 35B A3B and 32.1 -> 36.6
  t/s on 122B A10B, with decode essentially unchanged.
- MoE packed prefill chunk tuning also matters: tuned default `P=128` lifts the
  same prompt to ~95.3 t/s on 35B A3B and ~37.6 t/s on 122B A10B.
- No-spec GPU argmax decode path landed. Dense decode is neutral within noise;
  MoE decode improves modestly by avoiding full logits readback (~1.1-1.5% on
  A3B / 122B in current 64-token runs).
- KV-Q8 remains a negative result on M4 for the existing v4 main-kernel
  structure. v0.437 extends the oracle to MoE/group8 and changes the reader to
  mirror the F16 vector shape, but attention is still slower.
- Attention v4 now supports `group=4`, unlocking the small dense family
  (0.8B / 2B / 4B / 9B) as real long-context canaries instead of failing back to
  the old threadgroup-memory-limited attention path. The local 9B sweep now runs
  cleanly through 32K: `64.0 t/s` at 4K, `59.5 t/s` at 16K, `53.8 t/s` at 32K.
- Attach-mode decode tracing is now practical via `qwen-bench decode-window`, and
  `scripts/profile/trace-metal.py` gives a compact Metal timeline summary
  without hand-written one-off parsers.
- Llama-style Q8_0 decode mat-vec is a large default MoE win with rollback
  `QWEN_MATVEC_Q8_0_LCPP=0`. A10B moves from `35.77 -> 42.90 t/s` at `tg64`,
  default `tg128` is `42.80/42.81` versus rollback `35.88`, and `tg256` moves
  `34.74 -> 42.45`; A3B `tg128` moves `84.22 -> 98.04`. Dense 27B `tg128` was
  slightly negative/noisy under forced env (`23.40 -> 23.13`), but the local 27B
  Q4 file has no Q8_0 tensors, so treat that as a repeat guardrail rather than a
  blocker.
- Group8 attention tile2 default for A3B long-context decode. At `ctx16384`, the
  default-vs-rollback phase packet moves total phase time `15.12 -> 13.66 ms` and
  attention `4.80 -> 3.60 ms`; context sweep throughput moves
  `72.8 -> 82.2 t/s` at 16K with neutral short-context behavior.
- Shared Q8_0 SwiGLU default for MoE decode, with `QWEN_DECODE_SHARED_SWIGLU_Q8=0`
  as rollback. This fuses shared expert gate/up plus `silu_mul`, moving A3B
  `tg128` roughly `98.84 -> 99.5-99.6 t/s` and A10B `42.84 -> 42.94-42.97 t/s`.
- F32 row-pair decode mat-vec default, with `QWEN_MATVEC_F32_LCPP_R2=0` as
  rollback. The cooperative two-row / four-simdgroup shape moves A3B `tg128`
  roughly `99.64 -> 100.5-100.7 t/s` and A10B `42.87 -> 42.97-43.05 t/s`.
- Fused Q5_K down+weighted-sum default for MoE decode, with
  `QWEN_DECODE_MOE_Q5_DOWN_FUSED=0` as rollback. It moves A3B `tg128` roughly
  `101.86 -> 103.8-105.1 t/s` and A10B `43.96 -> 44.65 t/s`.
- Group16 attention tile4 default for 122B long context.
- Group6 dense attention `NWG=64` at `n_pos >= 4096`.
- `QWEN_ATTN_V4_NWG`, `QWEN_ATTN_V4_TILE_C`, `QWEN_ATTN_V4_G8_TILE`, and
  `QWEN_ATTN_V4_G16_TILE` A/B knobs.
- Production-style `NWG=64` correctness coverage for attention v4.
- MoE intra-block profiler: 122B block ~0.594 ms, with mixer prep largest.

Recent measured negatives:

- v0.437 falsifies same-layout Q8_0 KV for MoE/group8 `attn_v4`. The Q8x4 reader
  is correctness-clean across group6/group8 main and group8 tile2/tile4, but A3B
  `attn-intra` regresses at both `ctx8192` (main `0.1113 -> 0.1221 ms`) and
  `ctx32768` (main `0.1767 -> 0.2213 ms`). Do not reopen scalar Q8, forced-tile
  Q8, group-tile/NWG sweeps, or same-layout Q8x4 variants without a new
  capture/counter signal and clean 8K+32K `attn-intra` wins.
- v0.542 falsifies the direct payload/scale split-plane Q8_0 layout on the same
  production group8 grids. It is exact versus the current Q8 oracle and passes
  F16 correctness, but the valid 32K candidate regresses `9.95%` main and
  `6.75%` main+reduce versus the slower F16 anchor. The 8K row is excluded for
  anchor instability, though its direction is uniformly negative. Do not claim
  a comparison with current interleaved Q8, which was not timed. Layout
  rearrangement alone is no longer a sufficient Q8_0 reopen condition.
- v0.543 falsifies direct interleaved canonical GGML Q4_0 KV for the
  preregistered A3B group8 reader. Standalone GPU reconstruction is bit-exact
  against canonical GGML with both scale signs, but real block-3 `ctx8192`
  attention fails fidelity at cosine `0.996111664` and maximum absolute error
  `0.1653642654`. The 32K correctness row and every performance sample are
  unrun after the mandatory stop. This is not a performance result or closure
  of all compressed-KV formats.
- v0.341 kills the naive MoE FFN expert-pipeline proof. Splitting top-k routed
  experts into two groups and overlapping group-A down with group-B gate/up was
  exact on the A3B serial-vs-pipeline smoke, but regressed `tg128`: A3B
  `103.90 -> 100.64 t/s`, A10B `45.01 -> 44.09 t/s`. Do not revive this split
  without a counter signal proving real overlap and a design that avoids the
  extra group-B accumulation pass.
- A3B long-context attention-v4 exposed knobs are exhausted after the v0.293 tile2
  default. At `ctx16384`, `QWEN_ATTN_V4_NWG=32` regressed attention sharply
  (`3.39 -> 4.89 ms`), `QWEN_ATTN_V4_TILE_C=32` was worse/flat, and
  `QWEN_ATTN_V4_TILE_C=128` was phase-interesting but end-to-end flat/noise
  (`83.2/84.0/83.9 t/s` default/C128/default). Reopen A3B long attention only
  with a deeper body/KV-traffic mechanism, not another knob flip.
- MoE Q8-KV subgroup attention is falsified in the straightforward reader form.
  The sidecar was numerically close to F16 KV (`cos=0.999992`) but regressed
  long-context phase profiles: A3B `ctx16384` attention `3.32 -> 4.16 ms` and
  A10B `ctx16384` attention `5.78 -> 6.61 ms`. Do not reopen KV quantization for
  MoE decode unless the reader/dequant path changes materially; the next
  attention body probe should favor layout or a structural body rewrite.
- A3B group8 tile1 decode attention is falsified. The sidecar passed the group8
  correctness gate, but `ctx16384` phase was flat/worse versus tile2: phase sum
  `12.18 -> 12.53 ms`, attention `3.25 -> 3.27 ms`. A group16 tile2 probe was
  also stopped at correctness. Do not reopen smaller subgroup splits without a
  new dataflow mechanism.
- Fused Q8_0 GDN-front decode is falsified in the tested forms. Dispatch-only
  fusion was flat/noisy on battery (`35.75/35.78/36.56 t/s` base/fused/base), and
  the x-cached four-simdgroup version was correctness-safe but catastrophic
  (`36.73/25.04/35.90 t/s`, battery-confounded but too large to ignore). Do not
  reopen this as launch fusion or hidden-vector threadgroup staging; a future Q8
  front branch must change packing/tiling materially.
- Q8_0 mat-vec row-pairing is falsified for A10B decode. An `NR0=2` variant passed
  primitive and A10B decode correctness, but regressed `tg128` from
  `36.73/36.90 t/s` base to `33.37 t/s`. Do not copy Q4/Q6 row-pair geometry to
  Q8_0 without a new memory-access mechanism.
- Split-concurrent GDN front projections are falsified for MoE decode. The probe
  was correctness-safe on A10B, but regressed `tg128` from `36.72/36.90 t/s` base
  to `35.09 t/s`; future GDN front work needs a different work shape or input
  reuse, not four concurrent encoders around the existing mat-vecs.
- Generic grouped expert-major MoE routed FFN via CPU ledger + gather/scatter +
  generic per-expert mat-mat is strongly negative on both A3B and 122B.
- Re-based “split routed FFN sidecar” experiments now show that beating the old
  packed-slot path was the wrong denominator: against the live grouped backend,
  separate gate/up grouped matmats plus the existing `silu_mul` + grouped down
  only reach parity to slight loss on `pp512`, so broad split-sidecar work is
  demoted until a smaller `MUL_MAT_ID`-style proof beats the current grouped
  projection / routed tail directly.
- Shared-expert batched stage-2 rewrite is semantically correct but slower
  end-to-end on A3B packed prefill.
- F16 routed-inner traffic reduction on the live Q5-down MoE path is a wash to
  slight loser end-to-end.
- Dense paired `gate+up` prompt fusion is exact-correct but only `~1.04x` in the
  exact-shape 27B microbench at `N=321`, below the go gate.
- Forcing single Q4 prompt mat-mat to `NR1=16` is worse than the current `NR1=32`
  path at `N=321`, so easy tile narrowing is not the answer.
- A3B Q4 short-MoE threshold/dispatch retreads are demoted: hot-threshold sweeps
  are tiny/noisy, all-`n32` regresses, all-`n16` only helps the losing side of the
  knee, old packed-routed fallback is about half-speed at `pp512`, and a simple
  bounded range-width cap failed A/B.
- A `<8`-only grouped n8 tile is also demoted: it preserves too much grouped
  underfill/control overhead and regressed Q4 `pp512` GPU time (`~0.706/0.709`
  base to `~0.732/0.732`).
- A masked cold-packed SwiGLU proof is demoted too: reusing the old packed-slot
  direct kernel only for `<8` regressed Q4 `pp512` GPU time (`0.7086/0.7027`
  base versus `0.7274/0.7380`). Packed fallback variants need a new mechanism
  before reopening.
- A first Q5 down multi-expert tiny4 microtile showed positive GPU time but failed
  correctness (`logits cos=0.981292` on the A3B prefill-vs-single gate). A
  corrected microtile remains a possible branch, but correctness must run before
  any perf row is counted.
- The corrected Q5 tiny8 R16 down proof is exact and now defaults for
  `chunk_p <= 768`, with `QWEN_PREFILL_MOE_TINY8_DOWN_R16=0` as rollback. Q4
  `pp512` default/rollback rows are `0.6864/0.6871` versus `0.6889/0.6969`
  GPU ms/token; `pp1024` is left on the old path in auto mode.
- The naive Q4 SwiGLU port of that geometry is falsified. It passed correctness
  but regressed Q4 `pp512` GPU ms/token (`0.6881/0.6854` base versus
  `0.7078/0.7086`) and failed to move the `<8` target bin (`36.28 -> 36.91 ms`).
  Do not revive one-simdgroup R16 SwiGLU without a new mechanism.
- Split gate/up with existing grouped Q4 matmuls is also falsified. It passed the
  A3B prefill-vs-single correctness gate, but Q4 `pp512` GPU ms/token regressed
  from `0.6851/0.6820` to `0.7017/0.7009`; the `<8` SwiGLU phase moved only
  `37.98 -> 36.92 ms`, far below the go gate. Do not reopen split sidecars unless
  the second projection fuses the epilogue and materially changes dispatch shape.
- The MR32 Q4 `<8` microtile is falsified too. It packed two tiny experts per
  threadgroup with two simdgroups per expert and no separate F32 epilogue, passed
  correctness, but regressed Q4 `pp512` GPU ms/token (`0.6851/0.6858` default
  versus `0.7032/0.7006`) and left `<8` SwiGLU flat (`35.88 -> 36.02 ms`).

## Force-Ranked Next Bets

Current rank after v0.356, plus the post-v0.356 hardware-headroom audit digest:

External audit read: do not turn generic code-smell cleanup into the new top of
queue without measured primary-row gates. The useful additions are narrower:
RoPE precompute is a cheap correctness-bounded sidecar because the current
kernels still compute `pow(theta, exponent)` per pair; PSO-cache locks,
`view_subrange` allocation churn, and tensor clones need warmed CPU/alloc
attribution before implementation because current hot decode rows are mostly GPU
active; `[[max_total_threads_per_threadgroup]]` and simdgroup-barrier pruning are
microbench candidates on exact active kernels, not blanket edits. Keep the active
implementation spine on concrete quant/long-context rows, while adding these as
gated hardware-headroom probes.

v0.389 counter-pivot read: autonomous Metal hardware counters are unavailable on
this M4 Max beyond timestamps, so do not wait for counter tables that cannot be
captured. Use existing software gates instead. v0.455 SUPERSEDES this pivot:
a user-saved Instruments template (`metal-counters`: Performance Limiters
counter set, Performance State Maximum) unlocks the full 64-counter Apple
limiter stream headlessly via `xctrace record --template 'metal-counters'
--attach PID`. The measurement loop lives in
`scripts/profile/gpu_limiter_capture.py` (uv script; `hold`/`capture --reuse-pid`
amortizes the ramp; per-experiment CSV under `target/profiles/gpu-limiters/`,
~30 s warm capture, several minutes cold export, ~1 s cached re-analyze). A3B
ctx16384 verdict is
recorded: low effective residency (Kernel Occupancy ~28% vs manager ~72%),
bandwidth 57-68% of stream, ALU pipes <=31%; NWG192 discriminator moves
nothing so the cap is per-kernel residency shape. Newly promoted top item:
per-kernel residency audit + one occupancy-shape retune gated on the
counter loop (inflight must move before any e2e claim). Byte reduction stays
demoted. Interpretation guide: `docs/bench/2026-07-03-xcode-decode-capture/`.
v0.457 adds the phase-noop limiter packet. Full A3B ctx16384 is still `3` kicks
at `2.94/3.70/3.49 ms` with occupancy `~28%` and Read BW `265/270/310 GB/s`.
No-GDN remains `3` kicks but lower occupancy (`17.6/18.0/23.3`); no-MoE and
no-GDN+no-MoE collapse to `1` kick/token (`8.86 ms` and `6.36 ms`) without
approaching stream bandwidth or improving occupancy. Treat noops as budget
oracles, not family-counter truth, because topology changes. The rank is now:
(1) PSO/resource audit for hot decode kernels, (2) batch-2/two-stream concurrency
discriminator, (3) one surgical occupancy-shape retune only after the resource
table predicts the cap and the counter capture moves occupancy/SIMD inflight.
v0.458 adds the cheap PSO resource audit. `qwen-bench metal-pipelines` confirms
`thread_width=32`, no static threadgroup memory, and no ICB support across the
hot decode set. Attention main/reduce and GDN-step are one-simdgroup contracts
(`max_threads_per_tg=32`, packed NSG4 `128`), while mat-vec/MoE/elementwise rows
report `1024`; metal-objdump does not expose register/private-memory counts. This
kills static-TGM and ICB as visible caps but is too coarse to pick a concrete
retune. Current rank: (1) batch-2/two-stream counter capture, gated on occupancy
rising from `~28%` toward `>=38-45%` and aggregate throughput `>=1.15x`; (2)
per-dispatch labels that cover `>=95%` of GPU interval time; (3) one focused
attention/GDN occupancy retune only after labels/counters identify the live
dispatch.
v0.459 runs that batch-2/two-stream discriminator in-process with shared model
weights, separate sessions, and separate command queues. It overlaps well
(`gpu_sum/gpu_span=1.90x`) but fails the pivot gate: A3B ctx16384 aggregate t/s
is only `88.1 -> 109.6` (`1.24x`), per-stream falls to `54.8 t/s`, and device
occupancy rises only `23.7 -> 30.6` under the limiter capture. Do not promote
scheduler/multi-slot/replay as the primary branch from this evidence. Current
rank: (1) single-stream per-dispatch labels/counter attribution, gated on
covering `>=95%` of GPU interval time and matching phase/noop totals; (2) one
focused attention/GDN occupancy retune if the top 3-5 dispatch families dominate
GPU time and retain the low-residency signature; (3) revisit multi-slot only if a
future shared-submit experiment reaches aggregate `>=1.5x`, per-stream
`>=70-75%` of baseline, and occupancy `>=40%` without read BW saturation.
v0.460 replaces the unavailable per-dispatch xctrace labels with app-side Metal
timestamp sampling at existing compute-encoder stage boundaries. The probe passes
the coverage gate (`raw_coverage_assuming_ns=1.000`) and perturbs A3B ctx4096 GPU
time by about `+13%`, so use it for attribution, not throughput. A3B ctx16384
family map: `attn_mixer_route 24.46%`, `gdn_after_route 20.13%`, `gdn_front
17.55%`, GDN-block MoE gate/up+down `14.15%`, and `tail_lm_head_argmax 8.00%`.
Current rank: (1) split the dominant mixed families only enough to decide the
retune target, after a longer/repeated stage window confirms the top-three
concentration (`attn_mixer_route` into attention vs route/post-norm;
`gdn_after_route` into GDN step/output vs residual/post-route glue); (2) run a
focused limiter capture on the winning split family; (3) edit the kernel only if
one isolated subfamily is `>=25%` alone or the top three families stay `>=60%`,
and the counter signature still shows low occupancy/inflight rather than byte
saturation. Do not read the `window=4` percentages as fine-grained speedup
ceilings.
v0.461 splits the top attention bucket. Route prep is only `~2.1%`, front
projections are flat at `~0.49 ms`, and the ctx-scaling term is `attn_body_out`
(`1.2516 ms` at ctx4096 -> `1.9174 ms` at ctx16384). Existing `attn-intra`
proportions point at v4 main+reduce for that slope, but the absolute ctx16384
share is still `16.75%`, below the combined GDN front/after-route pool (`~38.7%`).
Current rank: (1) split `gdn_after_route` and `gdn_front` with the same stage
timestamp discipline; (2) run an attention-body limiter/counter probe tied to
main+reduce, not another blind NWG/tile sweep; (3) promote attention work only if
it shows `>=0.35-0.50 ms` recoverable ctx16384 upside without ctx4096 regression,
otherwise prefer the largest isolated GDN subfamily.
v0.491-v0.493 close the occupancy-attribution thread and KILL the persistence
branch (cx-gated topology program, session `019f347b-c...`). v0.491 (A0): the
v0.455 "28% occupancy" is a duty-cycle artifact - kicks start at 56-58%, burst
to 60-70%, and spend most interior 25 us bins in recurring valleys (H3-shaped,
no tail decay). v0.492: the only sub-kick trace structure is WindowServer
compositing (~3% of the shortfall; excluded), and within-encoder dispatch
timing is triply unobservable on M4 Max - do not re-attempt trace-based
alignment. v0.493 (B0 probe, `qwen-bench topology-probe`): 32-wide
co-residency PASSES (~18/core conservative, 30-58/core max_alive; fill
ceilings ~1764 threads/core compute-bound vs ~2950 memory-stalled; ~96
simdgroups/core hard cap at W128/256); bounded self-validating cross-TG
signaling is RELIABLE (100% delivery to 720 consumers, excess p99 <= 0.1 us)
but relaxed cross-object data-then-flag is UNSAFE under traffic (2e-3..6e-3
stale rate; payload-in-flag is mandatory); dispatch boundaries cost only
~2-3 us at decode-realistic shapes (~11 us extreme mixed), so persistence
recovers <= 18% of a 110-stage glue ladder vs a >= 30% kill line - Program B
is DEAD, B1a cancelled pre-build, and the v0.443 DFlash persistent-verify
reopen condition is CLOSED. Working attribution (cx: "roadmap working
hypothesis requiring an attribution-first discriminator"): the valley mass is
INTRA-dispatch under-parallelism - narrow one-simdgroup dispatches cannot
fill 40 cores regardless of boundaries. Next program (cx-signed direction):
WIDTH/CONCURRENCY restructuring, attribution-first - (1) per-family
dispatch-width census over the v0.462 stage splits; (2) a concurrent-encoder
discriminator on 2-3 provably independent narrow stage pairs (machinery
exists: `begin_concurrent` + hazard notes); (3) only then any production
widening. Program C (ctx-65536/131072 scoreboard vs pinned llama.cpp +
limiter capture) remains GO and may re-rank attention-side work first.
v0.494 executes Program C: ours vs pinned b9833 at ctx16384/32768/65536/
131072 = `1.34x/1.30x/1.1-1.2x/1.80x` (llama.cpp halves per doubling past
65k; the moat WIDENS at true-long). Attention @131k = `46.2%` of the token
(pre-registered 45-55% band CONFIRMED); the 131k limiter capture shows the
SAME latency/occupancy-bound signature as 16k (occupancy ~26%, read BW ~66%
of stream) - true-long is NOT bandwidth-walled, so the width/concurrency
attribution applies there too. `--prefill-warm` (validated: gpu_ms within
~0.5-2% of decode-ramp) makes deep-context measurement routine (131k warm
~4 min). Parked v0.490 condition is now MET if true-long becomes primary:
revisit `QWEN_ATTN_V4_G8_BCAST=1` promotion and re-rank attention byte/
occupancy work against the W-program (Britt's call).
v0.495-v0.497 then close the board (cx-signed program-level statement,
session `019f347b-c...`): the dispatch census revises narrow-glue to a
~1.2-1.5 ms ceiling; W1b partition packing is falsified at its control row
(+61.5%); the Q8-KV reopen is blocked at scope review (v0.437 covers it);
and Program T (tree speculation) is killed at T0 by DIRECT tree-decode
simulation - tree-sim lifts emitted/step +20-26% (code 4.27->5.12,
narrative-start 2.91->3.51) but even the v0.444 all-heroics cost ceiling
only touches 1.25x on code and fails narrative-start at 0.85x, so the
recorded DFlash reopen condition stays unmet, which in turn keeps the
v0.439 >=32-row matrix-decode precondition unmet. "Under current model
assets, current drafter, current N=16 verify budget, and current M4 Max
measurements, all recorded reopen conditions in this arc are closed by
measurement. Further large decode gains require new assets or a materially
new execution primitive, not another local kernel retune." Reopen recipes
are recorded per branch: DFlash/tree needs a larger-block or
stronger-shallow drafter (the `--tree-sim` harness prices any candidate in
minutes, no engine work); matrix decode needs a real multi-token source;
attention main needs a materially different body; persistence needs
nothing (its mechanism was measured and does not exist at useful
magnitude on this GPU).
v0.498 then runs the closure itself through a falsification audit (was the
v0.443/v0.444 skinny-verify cap a single-design artifact?) and REINFORCES
it: a clean-room multi-column GEMV family (mat-vec geometry, occupancy-
corrected, E0 bit-exact per column, Q4_K + Q6_K) lands at `~270
GB/s-equiv` under hot-repeat microbench conditions — the same wall as the
v0.444 F32-dot family, now two design families deep (the GEMV family was
measured in both ALU-heavy and occupancy-split bodies). `c(2) =
1.57-1.71`, `c(4) = 2.9-3.3`: deep-N scalar multi-column decode stays
closed, and the reopen arm (c) "MTP-side economics" resolves as MARGINAL:
composed MTP-1 ceiling `~1.15-1.2x` code-only at measured on-box alpha
(code `0.984`, prose `0.693`, equivalence PASS), gated on
single-command-buffer GPU-fed drafting (the measured `28-44 ms/call`
prototype drafting is the binding loss today). Two recorded follow-ups, both with kill lines: (1) MMA-unit
small-N kernels at mat-vec-grade occupancy (the scalar-ALU cap does not
bind the MMA pool; the MM tile's 56-128 GB/s was 64-row-tile
under-occupancy) — kill line `c(2) <= 1.25` on ffn shapes, else the
shallow lane closes too; (2) whole-step packed MTP-1 measurement only if
(1) or the drafting restructure moves first. See the v0.498 PERF-LOG entry
for the full gate-session numbers (fixed-layer anchors, DFlash drafter
health at HEAD).
v0.499 executes follow-up (1) and the kill line fires: an 8-row x
8-padded-column simdgroup_matrix kernel at `n_out/8` TGs (design jam
corrected the mechanism first — M4 has no separate MMA pool, so the
candidate win was dequant amortization + dense FMA encoding + occupancy)
measures `c = 1.92-2.41` on the kill shapes at `4.5-5.2 TF` (correct at
cos 1.000000; a staged-B/wider-K v2 regressed; two variants measured,
wider micro-variants explicitly not covered). Per pre-registration THE
SHALLOW SMALL-N LANE CLOSES: across both measured kernel classes the
best-known costs are `c(2) 1.57` (scalar), `c(4)/c(8) ~1.9-2.4` (mma8),
leaving composed MTP-1 at `~1.15-1.2x` code-only — below practical
value. The kernel-level speculative-verify arc on M4 is CLOSED at the
pre-registered scope. Reopens: an untried micro-variant beating
`c(2) <= 1.25` (the harness prices candidates in minutes); M5-class
tensor silicon; a step-change-alpha drafter/MTP asset (price in
`--tree-sim`, no engine work); or a real multi-token source arriving via
batching, which inherits mma8 as the best-known N in {4,8} primitive
(`~2.0-2.8x` the incumbent MM tile at N=8).
v0.500 then answers Britt's stale-assumption challenge with a staleness
audit + cx-vetted systematic sweep (see the PERF-LOG entry for the
table). The audit finds the verify path frozen at v0.44x selections:
GENERIC 32-wide tiles at N<16 (n16 needs n_query==16 exactly), per-token
GDN/rope/scatter/attention where prefill has packed bodies, one serial
encoder, and MTP drafting paying full-graph + full-V lm_head + sync per
call. The sweep (N {2,3,4,8,16} x 7 shapes x dtype/N-dependent configs,
up to 10 per cell, correctness asserted) fires R2 hard — the current
selection is dominated at EVERY N<16 cell (5.0-8.3x vs best 1.4-2.5x on
Q4_K/Q6_K/lm_head cells; Q5_K floors at zero-code pad16 3.9) and at
N=16 on 5/7 shapes — while
R1 stands closed (best c(2) 1.53, bar 1.25) and the interaction variants
+ n16_v2 are freshly falsified rather than assumed. Material revision:
best-kernel verify(4) projections land at ~1.6-2.3x per shape (was
generic ~5x), shifting composed MTP-3 to `~1.5-1.7x` code-only
(composition estimate; gates unchanged: drafting restructure +
whole-step measurement). Top-ranked follow-up is now the VERIFY-PATH
INTEGRATION PROJECT: wire the best-kernel table (pad16 as the zero-code
floor) + packed GDN/rope/scatter into packed verify behind
greedy-equivalence and whole-step re-measurement gates; then whole-step
MTP-3 re-pricing; then the Q5_K port and drafter phase-2/sync cleanup.
v0.501 lands stage one: the small-N table is production default
(`QWEN_MATMAT_SMALLN_TABLE`, rollback documented as re-exposing the
witness below) and moves whole-step DFlash static-16 verify
`0.862/0.871x -> 0.952/0.951x` (+10.4%, two samples each side,
equivalence PASS) and the MTP-N prototype to `0.836x` (3.32
emitted/step; drafting machinery still binding). The rollback A/B also
SURFACED A LATENT CORRECTNESS FINDING, confirmed on the pristine v0.500
binary: the pre-table path deterministically violates greedy
token-identity on a short-prompt static-16 witness (first divergence at
index 84, single near-tie flip; best-fit mechanism: E1 half-staged
verify tiles vs the mat-vec decode chain). The witness is now a
required gate row; the residual E1-accept-path risk class is recorded
with tie-guarded verify (E0 recompute under a logit-margin guard, MoE
3e-4 fallback as the house pattern) as the top correctness follow-up.
v0.502 reopens native MTP from the MTPLX M4 Max 64 GB signal (`~65-80
tok/s` on Qwen3.6 27B) but the new qwen-side replay/oracle probes sharply
change attribution: on 27B MTP, `--spec-tokens 3 --tokens 64 --no-warmup`,
`replay-current` records source `21.0 t/s` and reruns the same draft token
vectors with `mtp_calls=0` at only `21.8 t/s` (`0.967x` total), while the
perfect greedy oracle reaches only `27.0 t/s` (`1.201x` total). Therefore
single-CB drafting is necessary but not sufficient; the active MTP branch is
packed-verify structural debt first/alongside drafting, not drafting sync
alone. Force-rank inside MTP: (1) packed GDN or GDN tape/capture equivalent,
(2) packed q_len 2..4 verify attention with KV reuse, (3) packed rope/scatter
and encoder concurrency where phase evidence supports it, (4) draft-only
LM-head/top-k, then (5) single-CB GPU-resident D3 drafting once verify ceiling
is no longer the blocker. Tie-guarded exact verify remains the correctness gate
for any accept-path speed row.
v0.503 refines that again: the packed verify ceiling has a strong physical-N
ladder. 27B perfect-oracle D7/N8 reaches `52.0 t/s` (`2.306x`) and D15/N16
reaches `57.1 t/s` (`2.532x`), while off-ladder D4/N5 is a loss and D8/N9 is
only `32.1 t/s`. Bucketed verify (logical D over physical N8 with padded slots
rolled back) rescues the cliff: D4/N8 `36.1 t/s`, D5/N8 `41.6`, D6/N8 `47.6`.
The actual native D3 path over physical N8 is a small real 27B win: code prompt
128 tokens moves D3/N4 `0.912x` -> D3/N8 `1.031x` at the same alpha, equivalence
PASS. Therefore the active MTP branch is now: (1) acceptance tracing for D7/N8
and D15/N16, (2) bucketed physical N8/N16 verify as the only packet shapes to
optimize, (3) single-CB recursive drafting + draft-only LM head if emitted/step
clears the bucket cost model. Do not optimize arbitrary ragged N before padding
or bucket policy fails.
v0.504 extends the real native recursive MTP bench path to D15 and selects D7/N8
as the practical branch. On the 27B code prompt at 128 tokens, D7/N8 reaches
`27.7 t/s` (`1.153x`) with alpha `0.429` / `~4.0 emitted/step` despite current
per-draft sync and shared LM-head costs; D7/N8 replay-current reaches `32.6 t/s`
(`1.362x`) with `mtp_calls=0`. D15/N16 is rejected for the current one-head
recursive drafter: `17.1 t/s` (`0.716x`) at only `~4.1 emitted/step`, far below
the N16 cost model. Active branch: D7/N8-specific GPU-resident recursive drafting
and draft LM-head reduction; keep D15/N16 as an oracle/asset-watch row only.
v0.505 closes the pure orchestration part of that branch: chaining all D7 draft
slots into one command buffer is correct but moves the 27B code prompt only
`1.153x -> 1.162x` while replay-current remains `1.362x`. Therefore per-depth
wait/readback/command-buffer overhead is not the main remaining D7 draft tax;
force the next branch through a body-vs-LM-head ablation using recorded draft ids
before building a draft-only head.
v0.506 runs that ablation. On the 27B code prompt, D7/N8 body-no-lm-head reaches
`31.1 t/s` and bridge-only reaches `32.1 t/s` versus normal single-CB `27.7 t/s`
in the same probe family; all rows preserve greedy equivalence. This says draft
`lm_head+argmax` dominates the remaining draft-side tax, recursive body is much
smaller, and bridges/KV repair are not worth optimizing. But deleting draft head
entirely still only reaches the low-30s t/s, far below the `52.0 t/s` D7/N8
oracle and the external MTPLX `~65-80 t/s` signal. Therefore exact fused
`lm_head+argmax` is not the next strategic branch unless a microbench proves a
large whole-run gain; prioritize acceptance/rank tracing and N8/N16 verify
phase debt, with draft-only low-bit/top-k head gated on acceptance.
v0.507 adds that rank trace. On the 27B D7/N8 code prompt, terminal mismatches
still contain the target token in the draft top-2 for `14/28` rows, top-4 for
`19/28`, top-8 for `24/28`, and top-16 for `26/28`. This reopens reranking and
tree-rescue as plausible acceptance levers, but only behind an online-policy
gate: top-k containment is hindsight unless draft-side features can choose the
non-top1 candidate before target verification. Next branch is top-k logit/id
capture plus offline policy simulation; do not build tree verify or low-bit
draft head before that simulator clears.
v0.508 adds top-16 ids/logits and runs that simulator. Simple margin-swap
policies are effectively flat (`4.031` emitted/step best), and even an oracle
one-token terminal rescue only reaches `4.812` emitted/step at top-16 versus the
`>=5.2` investigation gate. Close simple rerank. The live acceptance branch is
now either real alternate continuation/tree economics, a stronger learned or
engineered corrector, a better MTP policy/asset, or an MTPLX measurement/reporting
difference; inspect MTPLX before building tree machinery.
v0.509 tests the cheapest draft-head clue from MTPLX and kills it: using resident
Q4_K `token_embd.weight` as the draft LM head preserves target equivalence but
collapses 27B D7/N8 acceptance to `0/127` (`5.3 t/s`). This only kills the
embedding-alias shortcut; a proper low-bit copy of `output.weight` remains a
separate, bounded cleanup idea.
v0.510 adds MTPLX-style hidden-semantics controls and decode-only accounting.
This materially changes the MTPLX-gap attribution: D7/N8 oracle reaches
`74.7 t/s` decode-only (`52.7 t/s` total), inside the external `65-80 t/s`
screenshot band once prompt/MTP prefill is excluded. Actual D7/N8 with legacy
pre/pre hidden feeds is only `30.2 t/s` decode-only and `3.879` emitted/step on
the code prompt; defaulting both base and recursive MTP hidden feeds to post-norm
raises that to `33.0 t/s` and `4.267` emitted/step. A narrative prompt moves
`26.8 -> 30.7 t/s` decode-only and `3.459 -> 4.000` emitted/step. Therefore
short-prompt MTP verify is not the current MTPLX-class limiter; acceptance and
MTP semantic/asset parity are.
v0.390 then demotes exact route from the main branch: A3B/A10B route replay still
repeats (`1.01/1.34 ms`), but
production already fuses the high-value topk/shared half and the only remaining
exact boundary is router logits into global exact top-k. The v0.390 local route
rank was: (1) captured MoE gate/up/down projection throughput
and active-token/expert batching; (2) a bounded attention read-once prototype only
if it changes the main-body memory shape; (3) exact route only if a prototype can
save `>=0.4 ms` A3B or `>=0.5 ms` A10B route total without consumer movement.
Local route barrier/candidate variants, GDN row-shape work, attention tile/NWG
knobs, and host-only cleanup stay closed unless fresh phase/no-op/microbench
evidence reopens them.
v0.391 adds the missing captured down gate and aligns it with production R2 for
`f_exp=512`: A3B/A10B captured gate/up and down micro rows now match phase within
noise (`1.079/0.814 ms` versus `1.08/0.82`, and `3.107/2.595 ms` versus
`3.21/2.56`). Use captured MoE micro first for compute-branch proposals, with a
required `5-10%` micro win before full phase promotion.
v0.393 then kills the existing one-token Q4/Q5 routed FFN monolith under captured
routes: A3B fused is `80.513 ms` versus split captured `1.894 ms`, and A10B fused
is `210.142 ms` versus split captured `5.701 ms`. This rules out fusion that
serializes the active weight stream into one threadgroup per token/layer. It does
not rule out projection-kernel changes, active token/expert batching, or tiled
fusion that preserves split-path parallelism; require `>=10%` full captured MoE
improvement before reopening another monolith-shaped decode branch.
v0.394 proves active-token batching is a real captured-route MoE lever: A3B split
gate/up+down improves from `1.894 ms/token` at one token to `1.198 ms/token` at
sixteen tokens, and A10B improves from `5.701` to `4.450 ms/token`. This justifies
a batching suite and multi-slot architecture branch, but not an end-to-end claim:
same-context route correlation, scheduler/KV overhead, layer variance, and expert
occupancy histograms remain unmeasured. Keep projection-kernel work active because
it benefits both single-token and batched decode.
v0.395 adds route occupancy stats and a ramp token-id control. The repeated-zero
captures overstated expert reuse: A10B t16 avg unique experts jumps `8.34 ->
54.40` under ramp, but gate/up only slows `33.24 -> 34.54 ms` and down improves
`37.96 -> 37.40 ms`. Therefore the batching win is primarily packed-slot
shape/occupancy, not repeated-token expert reuse. Use ramp or real prompt traces
for future timing; high-reuse synthetic wins alone are not promotion evidence.
v0.396 adds a loaded-once `moe-batch-sweep` to map that knee without repeated
loads. Ramp sweeps show A3B combined MoE projection time improves `1.889 ->
1.211 ms/token` from t1 to t16, while A10B improves `5.659 -> 4.481 ms/token`.
The useful knee is around t8, and t16 adds little. Batching is now a serious
architecture branch, but the next gate is real-prompt/multi-context capture plus
end-to-end decode overhead; do not promote production multi-slot decode from MoE
micro evidence alone.
v0.397 closes the first realism gap with a contiguous real prompt from
`the_current.md`: A3B is `1.885 -> 1.215 ms/token` from t1 to t16 and A10B is
`5.710 -> 4.508 ms/token`, essentially matching ramp. This justifies the next
gate, not production yet: run independent prompt/disjoint-context captures to
test cross-context packing before building a scheduler architecture branch.
v0.398 runs that stronger disjoint-position gate with stride128. A3B still passes
(`1.870 -> 1.219 ms/token` at t16), but A10B collapses after t4 and regresses at
t16 (`5.751 -> 5.510 ms/token`, versus `4.508` contiguous). Generic production
batching is demoted; keep A3B batching alive, and require A10B route-aware
packing/kernel evidence before any broad scheduler work.
v0.399 adds independent-file capture with one fresh session per prompt file. A3B
passes again at ctx512 (`1.885 -> 1.264 ms/token` at b8), while A10B only has a
useful b2 row (`5.727 -> 4.756`) and weakens at b4/b8. Production batching should
be model-specific: A3B can proceed to a targeted prototype gate; A10B should cap
at b2 or stay disabled above b2 unless an expert-sorted microproof beats that cap.
v0.400 runs that A10B expert-sorted kill-test as a perf-only locality upper
bound. It sorts captured slots by expert id without preserving exact token/slot
semantics, so an exact implementation would have to pay additional overhead.
A10B stays flat at b2 (`4.725 -> 4.723 ms/token`) and regresses at b4/b8
(`4.840 -> 4.945`, `5.135 -> 5.223`), which kills expert sorting as the A10B
rescue for this workload. A3B sees only sub-1% sorted changes; keep the next
batching branch focused on exact A3B end-to-end overhead, not route ordering.
v0.401 adds a routed-FFN batching upper-bound calculator and applies it to A3B
`ctx512/2048`. Even the optimistic b8 estimate, including Q6 down fallback, is
only `~0.77 ms/token` saved (`8.2-8.3%` of phase sum, ideal `~1.09x`) before any
real scheduler, pack/scatter, command-buffer, KV, attention, shared-FFN, or
sampling overhead. This demotes the immediate A3B multi-slot scheduler prototype:
batching remains a later system feature, but the top decode branch should return
to larger single-token byte-reduction/fusion unless a future upper-bound row
clears `>=12-15%` credible end-to-end savings.
v0.402 kills the cheap MoE `max_total_threads_per_threadgroup(64)` annotation
probe on the exact hot Q4 SwiGLU and Q5 down packed kernels. The patch builds,
but A3B independent-file captured rows are flat (`b1 1.824 -> 1.828 ms/token`,
`b8 1.221 -> 1.220`) and A3B phase is flat (`9.32 -> 9.34 ms`). A10B shows a
small batch-only `b8` improvement, but `b8` remains slower than `b2` and
single-token is flat/slower. Do not blanket-annotate kernels without manual
shader-profiler evidence or a named-kernel micro win.
v0.403 refreshes A3B long-context deep phase/roofline at `ctx8192/16384` after
the batching and annotation demotions. Attention is now the largest structural
slope term (`2.30 -> 2.53 ms`, `21-23%`), with KV subgroup+partial estimates only
`~282-296 GB/s`. GDN QKV/Z and LM head are near stream shelves (`424-515 GB/s`),
route topk/shared is real but locally falsified (`0.67-0.68 ms`), and routed
gate/down remains moderate but constrained by recent monolith/batching gates. The
next A3B long-context branch should be an attention main-body KV-read shape
change with an `attn-intra`/oracle gate; more tile/NWG/reduce retunes stay closed.
v0.404 then shows the local NWG shelf is small: NWG192 saves only
`0.13-0.15 ms` in phase at `ctx16384/32768`. v0.405 kills the concrete read-once
S2 oracle: correctness is exact, but staging K/V once for two simdgroups regresses
the main body by `3.7-5.6x`. Do not reopen attention read-once work by adding
large threadgroup-memory staging or reducing Q-head grid parallelism; the next
hardware-saturation branch should move to A3B multi-slot batching or a broad
decode byte/fusion audit unless a new attention idea avoids this failure mode.
v0.406 supplies the missing broader batching signal: A3B GDN `qkv+z+out` over all
30 GDN layers improves from about `2.44-2.48 ms/token` as repeated matvecs to
`1.26 ms/token` at batch 8 and `0.55 ms/token` at batch 16 using existing matmat
kernels. That is an always-on `~1.17-1.93 ms/token` primitive saving at
`ctx32768`, larger than the routed-MoE-only bound. Make the next architecture
gate a production-shaped decode-phase batch replay over `S={1,2,4,8,16}`; promote
multi-slot/layer-batched decode only if `S=8` clears `>=10%` or `>=1.0 ms/token`
phase-equivalent savings after attention, LM head, MoE routing, and layout costs.
v0.407 broadens that primitive gate with `decode-proj-batch`: A3B projection-only
aggregate remains negative through `S=4`, then crosses over hard at `S=8`
(`4.7255 -> 2.8866 ms/token`, `+1.8389 ms`) and `S=16`
(`4.6600 -> 1.2346`, `+3.4254 ms`). A10B and dense 27B confirmations are larger
(`+5.94/+9.55 ms` and `+11.23/+26.29 ms` at `S=8/16`). This passes the
projection-only kill gate and re-promotes a fuller decode-phase batch replay as
the highest-leverage architecture gate, but not a scheduler implementation yet.
The next replay must charge attention body/KV, routed-MoE capture/replay,
routing/topk, slot pack/scatter, layout copies, and ragged occupancy; require net
`S=8` savings to stay above `>=10%` or `>=1.0 ms/token` before scheduler work.
v0.408 digests the kernel-bypass audit against that result. The frame is useful,
but it does not outrank the measured batching gate: warm single-token decode has
historically been `~95.8-97.8%` GPU-active, so command-stack bypass is not yet the
dominant proven limiter. Add these profiling targets, in order: (1) full
`decode-phase-batch` replay with command/dispatch counts and net `ms/token`;
(2) ragged continuous-batching occupancy mixes (`50/75/90%` active, joins/exits,
mixed context lengths); (3) MoE route/pack/scatter/expert-utilization accounting;
(4) fused `lm_head+argmax/top-k` as an exact "do not materialize logits" probe;
(5) no-allocation/resource-binding audit for steady decode; (6) memory-format
experiments such as GDN/KV/residual BF16 only with correctness and quality gates;
(7) one-layer megakernel or persistent-work-queue proof only after the charged
replay shows command/encoder overhead is the remaining ceiling. Deprioritize
AMX/ANE coprocessor paths, GPU-side graph traversal, hierarchical `lm_head`,
layer skipping, KV clustering, sparse FFN, gate-based attention skipping, and
near-zero drafters until they have explicit quality gates and a measured phase
ceiling. `decode_phase_roofline.py` now reports stream lower-bound columns so the
"cycles per token" question can be asked from existing phase artifacts.
v0.409 adds a charged `decode_batch_upper_bound.py` estimator and applies it to
A3B `ctx32768` using fresh deep phase plus same-context routed-MoE replay. The
charged `S=8` row saves `2.4618 ms/token` (`19.55%`, ideal `1.243x`) after leaving
attention body/KV, route/topk/shared gate, GDN tail, residual/norm/layout, and all
other unmodeled work at baseline. `S=4` is still negative and `S=16` is strong
(`4.0858 ms`, `32.45%`). This clears the continuation gate for a real
`decode-phase-batch` replay, but still does not justify scheduler architecture.
The replay must report net `ms/token`, wall/GPU split, command/encoder/dispatch
counts, attention body/KV, route/topk, routed MoE, GDN tail, pack/scatter, layout,
logits/sampling, and ragged occupancy. Promotion gate remains `S=8 >=10%` and
`>=1.0 ms/token` under realistic active-slot availability.
v0.410 charges the obvious projection layout hole directly. `decode-proj-batch`
now times GPU blit pack/scatter plus matmat, and A3B `S=8/16` only loses
`0.0747/0.0638 ms/token` versus pure batched matmat. The layout-charged upper
bound still saves `2.4151 ms/token` at `S=8` (`19.18%`, ideal `1.237x`). Copy-only
layout is not the branch killer. Do not spend another isolated-layout cycle; the
next artifact must be actual `decode-phase-batch` replay with attention body/KV,
route/topk, GDN tail, routed MoE, logits/sampling, and ragged occupancy visible.
v0.411 lands the first real replay slice: one GDN layer with per-slot pre/post
norms, real GDN tail/state mutation, packed qkv/z/out projections, and all-slot
correctness checks. A3B block-0 remains negative at `S=4` but wins at `S=8/16`
(`0.1363/0.1007 ms/token` saved in the one-layer harness, `43-54%`). This is a
continuation signal, not a scheduler gate: absolute one-layer timings do not map
directly to full-model phase rows, block-0 may be favorable, and full occupancy is
idealized. Next: measure representative GDN layers, then MoE/FFN replay with route
divergence, before full integrated phase replay or scheduler architecture.
v0.412 removes the block-0 concern: first/middle/last A3B GDN layers all pass
all-slot correctness and save about `0.033-0.035 ms/token/layer` at `S=8`, and
`0.068-0.079 ms/token/layer` at `S=16`. This makes the GDN economics plausible,
but the next highest-leverage question is integration, not more isolated layers.
Build an integrated multi-block decode slice with ragged masks; promote only if
it preserves at least `0.5 ms/token` at `S=8` or `1.0 ms/token` at `S=16` on A3B,
and kill/deprioritize if integrated savings fall below roughly `25%` of the naive
sample extrapolation or correctness drifts across chained layers/tokens.
v0.413 adds that chained-state bridge for GDN-only replay: four consecutive GDN
layers preserve all-slot correctness (`min_cos_h=0.999999509`) and save
`0.1594 ms/token` at `S=8` and `0.2536 ms/token` at `S=16`. This raises confidence
that the GDN subpath is real, but it still skips attention/MoE blocks. Next exact
step: a full block-slice for 2-4 consecutive real blocks that executes normal
attention/MoE and swaps only GDN for replay. Ragged masks and scheduler work wait
until that block-slice shows net wall-clock savings.
v0.414 clears that first integration gate: a four-block A3B MoE slice with three
GDN blocks and one attention block, normal MoE route/FFN, and only GDN mixer replay
saves `0.1409 ms/token` at `S=8` and `0.1899 ms/token` at `S=16`, with
`min_cos_x=0.999999329`. Continue, but do not promote to scheduler yet. The next
decisive tests are nonzero/long attention positions, mid/late block windows, and
ragged active-slot masks; position-0 full occupancy is still a friendly case.
v0.415 keeps the early/mid path alive and quarantines late GDN-before-attention:
early `block0..4` still saves `0.1522 ms/token` at synthetic `pos4096` and
`0.1539` at `pos16384`, while mid `block20..24` saves `0.1375` at `pos4096`.
Late `block37..40` fails correctness at both `pos0` and `pos4096`, while pure
attention block 39 is exact. Treat late windows as unsafe until route/topk ids,
route margins, and per-block boundary deltas identify whether replay drift flips
discrete MoE routing or gets amplified by attention. Scheduler gates must be
window-specific; do not generalize early/mid wins to late layers.
v0.416 external-audit + cx digest: do not let the new hardware-saturation audit
displace the cheapest live uncertainty. Finish route/topk fingerprints and
per-block boundary deltas first; v0.415 localized a correctness cliff, not a
global kill. Keep multi-slot decode replay as the top active hardware-headroom
branch for A3B/serving, but treat it as a product-shaped multi-slot feature, not
a single-stream speed claim. Add these gates before major new rewrites: manual
Xcode GPU captures for large roofline claims; a narrow `attn-intra` win before
reopening KV-Q8; an exact one-layer row/block micro-oracle before chunked GDN
prefill; profile proof that verify attention dominates before packed-N verify
attention; and a same-shape long-prefill win before FA2-style fused prefill
attention. Current ordering: (1) route/topk and boundary diagnostics for replay,
(2) ragged/realistic replay gates if that passes, (3) KV-Q8 reader micro-oracle,
(4) chunked GDN prefill micro-oracle, (5) packed verify attention behind KV-Q8,
(6) FA2 prefill attention behind a fresh long-prefill phase gate.
v0.417 localizes the late-window failure mechanism: the visible cliff is
mediated by route-set flips at near-tie topk boundaries. Passing controls
(`block20..24` and `block36..39`) have zero route-set mismatches, while failing
windows first flip a single expert set at margins around `0.0001-0.0003`
(`block37 slot1` or `block39 slot3`) and then amplify through MoE/attention.
Do not treat route-order changes with the same expert set as semantic failures.
Next scheduler gate should be a conservative two-layer policy: static early/mid
block eligibility first, late blocks exact-only, and then a replay-side route
margin guard with rollback to exact pre-window state before ragged occupancy
work. Calibrate the margin threshold from exact-vs-replay router-logit deltas and
margin histograms, not from the observed failing margins alone.
v0.418 adds that first calibration hook. The observed failing route-set flips all
have replay margins below `1e-3`, but a passing `block20..24` control also has two
safe low-margin rows below `1e-3`. Therefore a dynamic guard is plausible but not
ready: the next gate is a broader margin/fallback-rate histogram over early/mid
blocks, prompts, positions, and slot counts. Do not promote a replay-side guard
from the five-window sample; static early/mid allowlist plus late exact-only stays
the current safe policy.
v0.419 adds the loaded-once margin sweep and complicates a naive static allowlist:
`pos4096` non-overlap windows ending at `block31`, `block35`, and `block39` can
flip route sets, while `pos0` non-overlap windows do not; targeted `block37..40`
at `pos0` still fails. Treat absolute block index as a coarse risk feature, not a
complete policy. The next replay work should measure fallback frequency under a
candidate replay-side margin threshold across real prompts/positions before any
ragged scheduler implementation.
v0.420 shows smaller `blocks=2` windows remove the `pos4096` flips from the
synthetic sweep, but still fail at `block38..40 pos0` (`min_x_cos=0.985815558`).
Therefore shorter windows are only a mitigation. The live replay branch now needs
product-shaped economics: real prompt slot sets, candidate margin thresholds,
fallback frequency, and net savings after exact fallback. Do not keep tuning
synthetic late windows without that fallback-rate accounting.
v0.426 adds the first real-prompt margin harness and broadens the initial sample:
32 non-smoke A3B rows across markdown rollouts, contexts `128/512/2048/3072`, and
windows `block0..4`, `20..24`, `28..32`, `32..36`, `36..40` show zero route-set
mismatches. Route order differs in 2 rows, worst `min_replay_margin` is
`0.000047`, and min `x` cosine is `0.999999965`. A window-level margin guard
would fallback on `3.12%/12.50%/25.00%/53.12%` of rows at
`1e-4/3e-4/1e-3/5e-3`. This keeps replay alive and suggests synthetic
constant-hidden fixtures are adversarial stressors, but cx session
`019f1ffe-1f3a-7120-a872-f068a90fd92d` correctly warns that this is still not a
scheduler gate. Next: widen prompt classes beyond markdown rollouts and pair the
margin table with net replay savings after exact fallback.
v0.427 broadens that packet beyond narrative markdown to mixed narrative/code/
docs/JSON prompts. The combined 47-row A3B sample still has zero route-set
mismatches, 3 route-order-only mismatches, worst replay margin `0.000047`, and
min `x` cosine `0.999999889`. Window fallback rates are
`2.13%/8.51%/23.40%/44.68%/61.70%` at `1e-4/3e-4/1e-3/3e-3/5e-3`. Per cx
session `019f201d-9777-7810-9b23-dbdbeb6dbeae`, treat this as a keep-alive
signal, not safety proof: `0/47` remains weak, synthetic flips remain real, and
the next live gate is net replay economics. Model validation overhead, exact
fallback cost, and ragged slot occupancy before building any production
scheduler. A first plausible threshold to model is `3e-4`; `1e-4` is too close to
known failure margins, while `1e-3` probably burns the win unless fallback can
abort before most replay work.
v0.428 adds that first occupancy-economics gate. Replay is a high-occupancy tool,
not a general slot filler: `blocks=4` loses at S=1/2/3, barely wins at S=4, and
only becomes material at S=6/8 (`~11-17%` gross). `blocks=2` GDN-only pairs show
the same low-occupancy loss but better S=6/8 gross savings (`~19-23%`) and were
already safer in the v0.420 synthetic margin sweep. Treat S>=6 plus a `3e-4`
margin guard as the first policy point to model; `1e-3` likely erases the win and
S<=4 is not scheduler-worthy without a fixed-overhead reduction. Next replay work
should quantify validation overhead and exact fallback cost, then model a realistic
ragged occupancy distribution. If that does not clear a durable `>5-8%` net gate,
park replay and move to the next hardware-headroom branch.
v0.429 codifies the conservative net model. At `3e-4`, S=8 `blocks=2` models at
`~14.6-14.8%` net, S=6 `blocks=2` at `~10.4-11.1%`, and S=8 `blocks=4` at
`~7.7-8.3%`; S=6 `blocks=4` is thin, S<=4 is negative, and `1e-3` erases the
win. Replay therefore has one remaining high-EV gate: measure the actual
validation/fallback path and a realistic ragged occupancy trace. If that mechanism
gate does not preserve S>=6 `blocks=2` above `>5-8%` end-to-end, stop the replay
scheduler branch and return to broader hardware-headroom items.
v0.438 runs that first real-window mechanism gate. The validated path splits
replay by block, reads replay route margins after each block, and charges exact
fallback slots at threshold `3e-4`. On real prompts, S8 blocks=2 still nets
`~11.5-20.3%` before broader fallback and roughly `~4.8-13.7%` after applying the
new `1/15 = 6.67%` S8 blocks=2 fallback packet. S6 is only `~2.7-5.5%` before
broader fallback, and S4 is negative. Update the live policy: replay is S8-only
unless future work lowers validation overhead or proves much lower fallback.
v0.430-v0.432 digest a do-less implementation audit (fixed roofline, fewer
bytes/dispatches) across decode, prefill, and kernels. Banked defaults:
matrix-attention causal tile skip generalized from G6 to all matrix groups
(A3B `pp4096` KQ/KQV `-14%` phase, A10B `pp1024` `-47%`; e2e `+0.9%` A3B),
packed-slot MoE fallback packs stubbed on production prefill scratch
(`~84-160 MB` resident saved; lazy-grown on fallback), split_q_gate layout
copies deleted via strided q-norm + strided fused gate epilogue (decode,
prefill, packed verify; rollback branch keeps the split), and DFlash hidden
capture batched to one strided-row copy per capture layer per chunk. Audit
verdicts recorded so they are not re-derived: production decode has NO
remaining zero-fills; mat-vec activation re-read is logical-only
(cache-resident, staging falsified v0.323); RoPE precompute fails its own
`>=0.5-1%` gate (~0.2-0.3 ms per 4096-chunk); remaining decode glue fusions
(residual+norm, K-chain, conv+L2, sigmoid/decay fold) total ~350 dispatches
+ ~8-10 MB per dense token with GPU already `96-99%` busy — bundle as one
gated experiment or skip; attn_v4 Phase A reduction shape is the largest
in-kernel ALU waste but stays fenced behind the byte-reduction rule. The two
promoted do-less bets are (1) a fused online-softmax matrix-attention body
holding score tiles in registers (same simdgroup tiles, no `[kvh,N,M]` f32
round-trip: `~16 B/elem`, A3B pp16384 `~365 GB`, 27B `~876 GB` per full
prefill; measured phase budget `~4-6%` A3B / `2-3%` 27B e2e at true-long)
and (2) an F16/BF16 repack of the F32 MoE router bank at load (`~84 MB/token
A3B decode, ~1.7%`), gated on an exact-top-k route-equivalence check across
real prompts. Also banked: v0.433 confirms the GPU test suite corrupts
itself under parallel execution (rotating victims; attn_v4 `cos=0.9662`,
mat_mat q8_0 `cos=0.48` under shader validation with no shader OOB logged
— uninitialized-partials hypothesis falsified by the NaN-prime gate). The
suite now runs serially via `.cargo/config.toml` `RUST_TEST_THREADS=1`
(green 2/2, no slower); the corruptor hunt (CPU-side test-helper memcpys /
blits suspected) is an open follow-up.
v0.430-v0.432 bank the near-free tier of the do-less implementation audit (same
roofline, fewer bytes/dispatches): (1) the KQ/KQV causal tile skip was G6-gated
at the host despite group-generic kernels — enabling it for G8/G4/G16 cuts the
A3B `pp4096` matrix body `~14%` (KQ `413->339-360 ms`, KQV `417->343-361 ms`)
and A10B `pp1024` KQ/KQV `~47%` each; (2) packed-slot MoE fallback packs are
now lazy stubs on production prefill scratch (`~84 MB` A3B / `~160 MB` A10B
resident saved; grouped default never reads them); (3) `split_q_gate` is
deleted from all production attention paths via strided-source q-norm and a
strided fused gate epilogue (one layout-copy dispatch + `2*q_dim` round-trip
per attn layer, decode and prefill), and the DFlash hidden-capture tap is one
strided-row copy per capture layer instead of `chunk_p` scatters. The audit's
remaining ranked items: (a) fused online-softmax matrix attention body — the
score tensor round-trips `~16 B/elem` (`365 GB` per A3B pp16384 full prefill,
`876 GB` dense 27B; est. `4-6%`/`2-3%` e2e at true-long) and is the top
prefill do-less item, roadmap-nominated but never built; (b) MoE router
`gate_inp` F32->F16 load-time repack (`~84 MB/token` A3B decode, `~1.7%`)
gated on an exact-top-k equivalence check across real prompts because routing
is discrete; (c) decode glue-dispatch fusions (residual+norm, K-chain,
conv+L2, sigmoid/decay fold) are bounded at `<=1-2%` total with decode
`96-99%` GPU-busy — one bundled A/B at most, not five branches. Measured
non-items: RoPE precompute fails its own `>=0.5-1%` gate (`~0.2-0.3 ms` per
4096-chunk); mat-vec activation re-read is logical-only (`3.5x` amplification,
`~0` DRAM — cache-resident); attn_v4 Phase A reduction shape is the largest
raw ALU waste (`~55%` of Phase A instructions) but the body is
bandwidth-bound and same-byte rewrites stay triple-falsified.
Test-infra triage carried with v0.432: `attn_v4_matches_naive_f16kv` is
load-flaky with a real numerical divergence at `group=4 n_pos=1024 nwg=64
C=16` (`cos=0.9662`; passes isolated) — suspected uninitialized-partials
sensitivity (`zeros_f32` is uninit); needs a scratch-init proof or zero-init
before the next attn-v4 branch trusts suite-load results.
v0.433 resolves that test-infra uncertainty by serializing the suite and
falsifying the uninitialized-partials hypothesis with a permanent NaN-prime gate;
rotating parallel GPU-test victims point at cross-test buffer corruption, not a
specific attn_v4 kernel bug. v0.434 then tests the router-repack bet. The F16
router is top-k safe in the measured packet (A3B 480 checks and A10B 384 checks,
zero order/set mismatches) and prefill-safe after adding F16 to the packed-route
eligibility, but it does not pay as a default: A3B/A10B decode is flat/noise,
A3B prefill is only `~+1%`, and A10B `pp1024` regresses `129.23 -> 114.53`
because F16 loses the F32 E8xP32 route-logits specialization. Keep
`QWEN_MOE_ROUTER_F16=1` plus `decode-moe-router-repack-check` as an opt-in
diagnostic harness, but demote router repack and do not widen to BF16. The
highest-EV do-less item is now the fused online-softmax matrix attention body;
decode glue remains a single bundled A/B only. v0.452 exercises the audit's
pre-authorized SKIP on that glue bundle instead: the ~350 glue dispatches move
~8-10 MB/token = ~20 us at stream against a ~9.3 ms A3B token budget (~0.2%),
and the dispatch-count savings land CPU-side where decode-window measures
med_cpu_enc ~0.64 ms of a GPU-bound ~9.9 ms token (med_wait ~= med_gpu).
Neither side reaches the audit's own 1% floor. Decode glue is CLOSED without
the A/B; do not re-derive residual+norm/K-chain/conv+L2/sigmoid-decay fusion
proposals unless decode stops being GPU-bound. v0.474 nevertheless executes the
residual+post-RMSNorm member of that glue bundle as a concrete exact sidecar; it
is correctness-safe but flat/noise (`~1.001x` A3B, `~1.011x` 0.8B, `~0.999x`
27B `tg128`) and remains default-off. Treat that as confirmation, not a branch to
repeat.
v0.435 executes the cheap `max_total_threads_per_threadgroup` audit on fixed hot
kernels (attn_v4 decode/packed/matrix entries plus GDN recurrence). It is
correctness-clean and keeps warmed A3B `tg128`/`pp512` and 27B G6-matrix `pp512`
inside the expected band, but it does not show a visible win. Keep the hints as
fixed-dispatch contracts; do not rank this as a broad hidden lever. cx also
pushes back on a softmax+KQV matrix bridge: with current KQV `head_dim/64`
y-tiling, a bridge would either recompute probabilities per y-tile or sacrifice
parallelism, so it is likely to teach the wrong lesson. Treat FA2-style matrix
attention as a serious design/capture branch, not a quick intermediate.
v0.436 dirty-tests cx's narrow GDN row/block oracle (`rb4_tgm16`: keep four
state rows resident like NSG4, stage `T=16` Q/K tiles in threadgroup memory) and
kills it. Correctness is green, but 0.8B `pp512` regresses `~11%`, 27B `pp512`
G6 matrix regresses `~2.6%`, and a 0.8B trace shows `gdn_step` itself worsens
`21.93 -> 32.59 ms`. Do not keep the flag/kernel. The active packed NSG4 kernel
already captures the easy row-residency win; Q/K load duplication is not worth
TGM barriers. If GDN recurrence is reopened, start with a floor/no-op ladder or
a genuinely chunked delta-rule formulation, not local Q/K staging. v0.479 tests
and kills a different sequential-recurrence rewrite: lazy state-decay scaling is
correctness-safe on the 27B prefill-vs-single gate, but regresses 27B `pp1024`
(`237.13/223.27 -> 233.89/218.11 t/s`). Do not reopen lazy-scale packed GDN; it
does not falsify chunked delta-rule, which must parallelize or reduce the
recurrence work rather than trade vector multiplies for scalar scale/divide
bookkeeping. v0.483 then tests the smallest exact chunked delta-rule work
reduction: a stable carry-product algebra, a tiny `K*K^T` / `K*Q^T` Gram
precompute, and an NSG4 `chunk16` recurrence kernel. Synthetic CPU/GPU and 27B
model correctness all clear, but the all-in 27B `pp1024` gate collapses
`238.75 -> 124.34 t/s` (`4.1500 -> 8.0000 gpu ms/token`). This kills host-loop
`chunk16` GDN. Future chunked GDN must be one-dispatch-per-layer or genuinely
matmul-shaped enough to beat the current packed kernel all-in before model
integration. v0.484 also kills the remaining cheap local row-loop cleanup: merging
the recurrence update loop with the post-update output-dot loop is exact and
correctness-clean, but 27B suite rows are flat/slightly negative (`pp1024`
`234.568 -> 234.579 t/s`, `tg128` `23.777 -> 23.629 t/s`). Do not reopen local
`gdn_step` loop reshuffles; only a materially different recurrence algorithm
belongs on the live queue.
v0.437 executes the reopened KV-Q8 attention-reader micro-oracle and kills it for
the current `attn_v4` execution model. The branch preserves the F16 grid, adds
MoE/group8 Q8 main and tile2/tile4 subgroup kernels, fixes `attn-intra` Q8
scatter, and changes the reader to Q8x4 float4 accumulation. Correctness is green
against F16 KV (`cos > 0.9999`, `max_abs < 0.01`), but clean A3B phase probes
are negative: `ctx8192` main `0.1113 -> 0.1221 ms`, `ctx32768` main
`0.1767 -> 0.2213 ms`. Keep `QWEN_KV_Q8=1` default-off as an oracle only. Future
compressed-KV work needs a materially different body, format, or capture signal;
do not spend more blind time on Q8_0 reader variants. v0.542 later tests and
kills the allowed payload/scale split-plane Q8_0 layout: its valid 32K row is
`1.0995x` F16 main and `1.0675x` F16 main+reduce. Exact Q8_0 at 272 bytes per
head-row is closed for these two reader-layout families.
v0.438 then returns to the top replay uncertainty and adds real-window economics
timing to `decode-block-slice-real-margin`. S8 blocks=2 survives the conservative
validated path, S6 is marginal, and S4 is dead. The next replay branch must be an
S8/ragged-occupancy scheduler sketch or nothing; do not spend implementation time
on low-occupancy replay.
v0.439 lands the score-round-trip bet as a TWO-pass design that resolves the
v0.435-parked softmax+KQV bridge objection (no per-y-tile probability
recompute: KQ owns the softmax where its pos-tile is resident, KQV applies
only a scalar `c_t` per (query, 64-pos tile) during staging, y-parallelism
untouched): KQ folds scale+mask+per-tile online softmax into its epilogue
(F16 `P~` + (m,l) sidecar, spill stride padded to 68 floats to kill a 32-wide
threadgroup-memory bank conflict), KQV folds `exp2(m_t - m_glob)` into its F16
staging and `1/l` into its epilogue. Score traffic 16 -> 4 B/elem, softmax
dispatch deleted, F32 score scratch halved to F16 (`~1.07 -> ~0.55 GB` A3B
pp16384). Microbench `1.22-1.37x` on the summed matrix body; phase A3B pp4096
`282 -> 222 ms`, 27B pp4096 `701 -> 541 ms`; e2e A3B pp16384 `+5.2%`, 27B
pp16384 `+1.7%`, pp512 guardrail neutral. Rollback
`QWEN_PREFILL_ATTN_MATRIX_ONLINE=0`. The matrix path also gained its first
isolated micro-oracle (CPU f64 reference, all groups/edges) and a kill-gate
microbench harness. FALSIFIED en route: a true 1-pass flash-attention body at
head_dim=256 (llama.cpp `kernel_flash_attn_ext` shape: Q=8/C=64/NSG=4, O in
threadgroup memory, direct-device K/V loads) is `0.80x` the sidecar at
production shapes; Q=16 occupancy-cliffs to `0.39x`, register-resident O
spills to `0.24x`. Cause: 8..16-row query tiles re-stream K/V 4..8x more than
32-column GEMM tiles and go L2-bound. Do not reopen 1-pass matrix attention
without a >=32-row-tile design that fits registers/threadgroup memory; the
remaining matrix-body headroom (P~ 4 B/elem, F16 Vᵀ sidecar) is bounded and
closed without a fresh phase budget. The v0.439 gate runs also produced the
strongest corruptor-hunt datum yet: the v0.433 corruption class reproduces in
a fully SERIAL process while a concurrent qwen-bench (separate process) loads
the GPU — rotating victims on untouched kernels, no shader-validation OOB,
3/3 green plus a full serial suite immediately after the bench exits. The
corruptor is NOT intra-process test parallelism; suspect a driver/multi-client
issue or a latent timing-sensitive race exposed by contention. Methodology
update: correctness gates require a quiet box (no concurrent GPU processes);
the hunt's next probes should include a cross-process load generator as the
reproducer instead of parallel tests.
v0.440 corrects the v0.438 replay timing confound: timed real-window repetitions
now reset `x`, KV, and GDN state from an immutable prepared seed before every
warmup/timed rep. Corrected S8 blocks=2 is still alive but much narrower:
validated net `~10.8-12.8%` before broader fallback and `~4.1-6.1%` after the
v0.438 `3e-4` fallback packet (`1/15 = 6.67%`). S6 averages only `~4.0%` before
fallback and becomes negative after it; S4 stays negative. `replay_economics.py`
now parses real-margin timing rows and can apply simple active-slot occupancy
mixes or FIFO request traces for p95 modeling. Update the live gate again: replay
is shadow-model only, S8-only, and must clear `>=5-8%` blended net on a real
occupancy/p95 trace before any production scheduler work.
v0.441 makes that gate executable: `replay_economics.py` now accepts active-slot
occupancy traces and FIFO request traces, reporting blended wall, throughput, p95,
and observed occupancy under an S8-only policy. The smoke trace is deliberately
best-case full occupancy and only reproduces the expected `4.85%` fallback-adjusted
win. The live replay branch is now blocked on a real request/occupancy trace, not
more replay microbench rows.
v0.449 rechecks that S8 policy at longer real-prompt contexts (`8192/16384`) with
eight independent prompt files and `blocks=2`, windows `0..2` and `20..22`. All
four rows have zero fallback slots at `3e-4` and validated net saves
`11.64-14.84%`; full-S8 occupancy economics blend to `12.85%`. This keeps S8
alive for long-context serving-style work, but does not remove the production
request-trace/p95 gate. True S8 `ctx32768` is currently corpus/harness-blocked
because only three local independent files exceed `32k` tokens. v0.465 removes
that harness blocker for GDN-only windows by letting one long prompt file supply
nearby slots at `context + slot * stride` with incremental prefix prep. A3B
`chaos.json` ctx32768/S8/stride1, `blocks=2`, windows `0..2` and `20..22`, has
zero fallback and validated net saves `14.41%`/`17.78%`; full-S8 occupancy blends
to `16.09%`. This promotes replay to a minimal S8-only shadow-policy branch, not
a scheduler/default branch: still require real or captured ragged occupancy,
fallback rate, replayed-token share, blended `>=5-8%` net wall save, and p95
non-regression before production scheduler work. Do not generalize this row to
independent-prompt diversity or attention-containing slices. v0.466 makes that
shadow gate executable by adding `--slot-counts` to the real-margin harness and
replayed step/token shares to request simulation. A3B `chaos.json` ctx8192 with
S1/S2/S4/S8 shows the expected policy shape: S1/S2 are strongly negative, S4 is
flat/negative, and S8 nets `7.51%`/`10.51%` on windows `0..2`/`20..22`; synthetic
request sims save `9.01%` saturated and `6.07%` on a ragged burst. The ctx16384
packet is weaker (`-3.13%`/`14.27%` S8, with the negative window repeating at
`7.25%` in an S8-only rerun), and the ragged synthetic sim saves only `3.73%`.
So replay remains live but not promoted: the next artifact must use real or
captured request traces plus robust repeats, not more full-occupancy-only rows.
Keep S8-only/GDN-only; do not build attention-slice support or runtime scheduler
plumbing until the ragged `>=5-8%` net and p95 gate clears. v0.467 adds p50 wall
columns to the real-margin timing rows so near-threshold packets can distinguish
average-wall noise from a real validation overhead loss. Keep the gate on average
wall/request p95 unless a later artifact explicitly justifies a p50 policy metric.
v0.468 adds `request_trace_from_game.py`, a provenance-labeled scenario builder
from game transcripts. It improves stress coverage with real transcript
completion lengths, but modeled `burst`/`fixed-gap` arrivals remain synthetic:
fixed-gap 500 ms scenarios save `8.13%` at ctx8192 and `5.02%` at ctx16384, while
burst scenarios save `8.76%`/`5.42%`. This does not satisfy the empirical
arrival/request gate. Replay is now explicitly parked as blocked on real/captured
arrival traces; do not add scheduler, attention-slice, or more replay kernel work
until that evidence exists. Scenario traces may be used only to keep the economics
tooling honest.
v0.450 adds a narrow current-HEAD qwen-only guardrail after the v0.444-v0.449
churn: A3B Q4 long decode remains `101.7/99.3/93.5 t/s` at
ctx `8192/16384/32768`, A10B Q4_XL `tg128` is `45.62 t/s`, and dense 27B Q4 is
`244.31` pp512, `224.93` pp4096, `23.51` tg128. No fresh guardrail regression;
do not burn a full paired family sweep until a branch changes a primary row.
v0.451 adds trace-only request plumbing: `qwen -p ... --trace-request PATH`
appends FIFO-compatible request rows, and `replay_economics.py --request-trace`
normalizes first-arrival time so real epoch timestamps simulate correctly. The
single-request smoke necessarily shows occupancy `1:2` and no replay save; this
does not promote replay, it only makes the real occupancy/request gate collectable.
v0.442 gives the cross-turn prefix/session cache its product-shaped probe. The
old H2 harness used a per-token prefill loop and overstated the benefit; the bench
now defaults to packed cold/prefix prefill and auto-selects per-token suffix
prefill for short cache-hit suffixes. A3B results: prefix 64 `1.27x` (fails 2x),
256 `1.92x` (fails 2x), 1024 `5.28x` (passes 5x), 4096 `18.47x` (passes 5x),
restore `3.0-7.3 ms`, exact greedy agreement. Product conclusion: cache repeated
1K+ prefixes aggressively, but do not sell prefix cache as a small-prefix win.
Next cache work is runtime/CLI integration plus bounded memory policy, not kernel
work.
v0.445 lands that runtime boundary: `LoadedModel` owns a bounded cache with
in-process model/tokenizer compatibility fingerprints, stats, resize/clear
controls, and exact restore helpers; `Sequence` can snapshot/restore through the
runtime wrapper with explicit identity/capacity checks; `qwen-bench prefix-cache`
now exercises this path. A3B runtime probes reproduce the v0.442 economics:
prefix 1024 `5.16x`, prefix 4096 `18.51x`, restore `4.1/7.4 ms`, exact greedy
agreement. Remaining cache work is request/CLI product wiring, observability, and
cross-process/persistent identity policy; the kernel gate is closed.
v0.448 adds the smallest CLI product seam: `qwen -m MODEL -p PROMPT -n TOKENS`
now runs a runtime-backed greedy single-turn path with packed prefill and
state-coherent decode. v0.469 adds the next narrow product seam:
`qwen --requests-jsonl FILE` keeps one `LoadedModel` resident, accepts explicit
per-request cache-prefix lengths, emits JSONL completions plus optional cache
stats, and demonstrates a 0.8B Q4 `1024`-token exact-prefix hit (`1629`-token
prompts) moving model-internal TTFT `299.8 -> 150.2 ms` with `1.75 ms` restore
and a `32.2 MiB` snapshot. Scope this narrowly: exact in-process token-prefix
reuse for repeated 1K+ prefixes. It is not streaming, concurrent serving,
cross-process persistence, automatic prompt-prefix discovery, or a user-visible
first-byte claim yet. Next cache work should collect real request traces with
hit-rate, prefix-length distribution, p50/p95, memory, and eviction stats before
building admission/discovery policy. v0.470 adds
`scripts/profile/prefix_cache_stats.py` as the reducer for those stats; use it on
real request captures before widening cache policy. v0.472 enriches the stats
schema with arrival/finish timestamps plus prompt/cache/matched-prefix hashes and
adds stdin JSONL (`--requests-jsonl -`), so the next cache artifact should be an
actual resident-process trace packet rather than another synthetic two-request
smoke. v0.473 adds paired `--compare BASELINE CANDIDATE` stats; use no-cache and
explicit-cache runs over the same request ids as the promotion unit, because a
single faster hit can still lose after cold insert and memory costs. v0.482
removes the manual-prefix tax for JSONL request packets by defaulting an automatic
repeated-prefix admission policy at `>=1024` tokens
(`--cache-prefix-auto-min-tokens 0` disables). The policy scans exact token
prefixes across the packet, picks one prefix per request by
`prefix_len * future_hit_count`, preserves explicit per-request/CLI overrides, and
records `cache_prefix_source`, `auto_cache_prefix_tokens`, and
`auto_cache_future_hits` in stats; stdin JSONL remains streaming and does not use
lookahead auto admission. A 0.8B release file-JSONL smoke with three requests and
`1260` shared tokens improves model-TTFT sum `511.4 -> 390.9 ms`, while the tiny
first-insert packet still worsens p95; this is the expected product shape, not a
full promotion. Next cache work is real request traces with no-cache versus
auto-cache paired stats, p50/p95, memory, and eviction behavior. If real traces
lack repeated `1K+` prefixes or p95 worsens after insert/eviction, park cache
policy and move back to S8 replay or chunked GDN.

0. Dense all-quant prompt guardrail: v0.347 found a blind spot in the old
   scoreboard. Static fast-path coverage was clean across 52 local Qwen GGUFs,
   but 0.8B Q4_0/Q4_1 and F16/BF16 `pp512` were still catastrophic because their
   dense prefill mat-mat lowering used scalar row/query kernels instead of the
   simdgroup-matrix shape used by the K-quants. The new Q4_0/Q4_1 legacy MM and
   F16/BF16 half/bfloat-activation defaults close the `5.8-14x` qwen-side cliffs
   with model-level drift gates (`logits_cos >= 0.999993` on BF16, exact-looking
   cosines on F16/Q4). The clean post-fix all-quant spot is green/parity across
   all 12 local 0.8B quants at `pp512` and `tg128`. Keep an all-quant 0.8B paired
   spot as a recurring guardrail before claiming broad quant wins; do not trust
   dtype coverage alone. A follow-up BF16 dense spot shows 4B/9B pp512 still
   `0.93-0.95x` llama.cpp, so BF16 is demoted from catastrophic to a small
   dense-prefill tile-shape gap. A dirty F16/BF16 direct-store epilogue proof
   regressed 4B/9B BF16 pp512, so do not chase that copyout without counter
   evidence. v0.351 broadens the dense guardrail to local 2B/4B/9B quants:
   non-BF16 prompt rows are all `0.98-1.04x`, decode is all green/parity, and
   BF16 4B/9B pp512 remains the only small dense quant softness. v0.352 runs the
   MoE quant guardrail. Supported quantized prefill is green/parity except BF16,
   but MoE decode now has three concrete quant gaps: A3B Q6_K/Q8_0 decode cannot
   enter the routed gate/up path, and A3B Q3_K_M/IQ4_XS decode is behind
   llama.cpp (`0.82x`/`0.75x`) despite prompt wins. Treat Q6/Q8 support and
   Q3/IQ4 decode attribution as the live quant-coverage branch; do not return to
   dense BF16 tile-shape work unless it repeats worse or gains a phase mechanism.
   v0.353 attributes Q3/IQ4 decode to routed gate/up and defaults a decode-only
   fused IQ3_XXS/IQ3_S SwiGLU kernel. Rollback is
   `QWEN_DECODE_MOE_IQ3_FUSED_SWIGLU=0`. Q3 `tg128` moves
   `66.54 -> 72.62 t/s` and IQ4 moves `60.50 -> 67.46 t/s`; routed gate/up drops
   `4.93 -> 3.59 ms` on Q3 and `6.39 -> 4.60 ms` on IQ4. Reusing grouped prefill
   IQ3 SwiGLU for one-token decode regressed, so do not reopen that shape. The
   remaining low-bit MoE decode work should be either Q6/Q8 gate/up coverage or
   a deeper decode-native IQ3/IQ4 dataflow proof, not another all-expert grouped
   prefill transplant. v0.354 clean low-bit spot confirms the external row moved:
   Q3_K_M `tg128` is now `0.90x` llama.cpp and IQ4_XS is `0.84x`, while their
   `pp512` rows remain green and Q4_K_M decode remains `1.41x`. These rows are
   still red but no longer first-order catastrophic; continuing low-bit work now
   requires a second concentrated bucket, not generic quant polishing. v0.355
   kills the simple IQ4_XS routed-down fused weighted-sum sidecar: Q3 `tg128`
   regressed `72.62 -> 70.78 t/s`, IQ4 regressed `67.46 -> 65.99 t/s`, and the
   routed-down phase worsened `2.13 -> 2.54 ms`. Do not reopen IQ4 down final-pass
   fusion without a deeper dequant/work-unit change or counter signal. v0.358
   closes the Q6_K MoE decode coverage hole with a decode-native fused routed
   SwiGLU kernel: clean A3B Q6_K `tg128` is `99.79 t/s` versus pinned llama.cpp
   `79.53 t/s` (`1.25x`), and the Q4 guard remains `107.00 t/s`. Q8_0 is now the
   remaining unsupported MoE decode quant because it still needs both routed
   gate/up and routed down coverage. v0.360 closes that Q8_0 coverage hole with
   native routed SwiGLU and routed-down weighted-sum kernels: clean A3B Q8_0
   `tg128` is `90.28 t/s` versus pinned llama.cpp `73.02 t/s` (`1.24x`), and the
   Q4 guard remains `106.96 t/s`. The quant branch now returns to Q3/IQ4 low-bit
   performance and recurring all-quant guardrails rather than unsupported Q6/Q8
   coverage. v0.362 cracks the Q3/IQ4 routed gate/up bucket by defaulting fast
   IQ3_XXS/IQ3_S decode SwiGLU kernels based on the dense IQ3 row-reuse dataflow.
   Clean A3B Q3_K_M `tg128` is now `89.34 t/s` versus pinned llama.cpp
   `81.50 t/s` (`1.10x`), and UD-IQ4_XS is `89.00 t/s` versus `80.71 t/s`
   (`1.10x`); Q4 guard remains `107.49 t/s`. The mechanism is concentrated:
   Q3 routed gate/up drops `3.58 -> 1.07 ms`, and IQ4 drops `4.59 -> 1.12 ms`.
   Low-bit MoE decode is now externally green; remaining low-bit work should
   target routed-down dataflow only if it changes the work unit or appears as a
   hardware-headroom row, not as a llama-parity panic. v0.364 then proves that
   routed-down dataflow branch by defaulting a dense-IQ4-style fast IQ4_XS down
   kernel. Clean A3B Q3_K_M `tg128` is now `103.27 t/s` versus pinned llama.cpp
   `81.54 t/s` (`1.27x`), and UD-IQ4_XS is `102.36 t/s` versus `80.46 t/s`
   (`1.27x`); Q4 guard remains `107.26 t/s`. Routed down drops `2.13 ->
   0.64 ms` on Q3 and `2.12 -> 0.65 ms` on IQ4. Low-bit MoE expert-bank decode
   is no longer the local bottleneck; next quant work should be a broad guard or
   hardware-headroom row, not more A3B low-bit coverage. v0.365's clean A3B
   all-quant spot guard keeps every measured local quant green at both `pp512`
   and `tg128`: Q3/Q4/Q6/Q8/IQ4 are `1.05-1.08x` for prompt and `1.25-1.39x`
   for decode. Retire A3B low-bit coverage from the active queue; keep it as a
   recurring guardrail only.

1. Decode long-context MoE FFN down/execution shape: v0.321 makes A3B Q4
   `ctx8192` a hardware-headroom row, not just a llama comparison row: MoE FFN
   apply is the largest named phase (`2.73 ms`, `24.5%`) and active-weight BW is
   only `243.8 GB/s` overall. v0.322 splits that FFN bucket: production-wave
   gate/up is `1.29 ms`, down wave is `1.40 ms`, finalizer is `0.17 ms`; the deep
   split puts routed Q5_K down at `1.27 ms` / `184 GB/s` and shared Q8_0 down at
   `0.48 ms` / `93 GB/s`. Simple exits are killed: Q5 fused-down rollback is
   slower (`93.0 -> 92.3 t/s`), dirty Q5_K `NSG={4,1}` fails the phase gate, and
   Q8_0 lcpp mat-vec rollback regresses badly (`80.2 t/s`). v0.323 also kills
   naive threadgroup staging of the Q5_K down inner vector (`93.0 -> 88.1 t/s`),
   so repeated inner reads are not removable by a simple copy/barrier wrapper.
   v0.324 confirms the pattern on A10B Q4_XL `ctx8192`: MoE FFN apply is
   `9.31 ms` / `34.8%`, down wave is `4.90 ms`, and routed Q5_K down is
   `4.58 ms` / `182 GB/s`. v0.327 re-anchors the clean same-build harness row:
   default is `42.4 t/s`, routed-down no-op is `46.0 t/s`, and Q5 fused-off is
   `41.8 t/s`. The next
   credible MoE FFN branch must change the Q5 down work unit more deeply or bring
   counters proving the down wave is compute/dequant-bound. v0.325 kills a Q5_K
   R2 two-output-row work-unit probe: same-build A10B `ctx8192` default/rollback
   rows were `41.9/42.0 t/s`, and rollback split down was slightly faster
   (`2.75 ms` versus `2.86 ms`). v0.333 rechecks both MoE targets after the prefill
   refresh: A3B `ctx8192` default/down-noop/fused-off is `93.1/100.4/91.8 t/s`,
   and A10B is `42.4/46.1/41.3 t/s`. Phase split still puts MoE FFN apply first
   (`25.3%` A3B, `29.3%` A10B), but the local Q5-down shelf remains exhausted.
   v0.341 kills the simplest execution-granularity/overlap proof: a two-stage
   expert pipeline was exact but regressed A3B/A10B `tg128`. v0.342 true-long
   rerank shows routed down is not the whole `ctx32768` story: A3B attention is
   the largest phase (`5.49 ms` / `34.7%`) while routed down is `1.65 ms`, and
   A10B splits between attention (`8.22 ms`), GDN qkv+z+out (`8.18 ms`), and MoE
   FFN. Keep Q5/MoE as a tooling/counter branch, not a local-kernel branch. The
   next branch should bring counters or change byte movement materially rather
   than retuning the same Q5 down kernel or adding split waves. v0.343 adds the
   fast primitive gate: `qwen-bench moe-down-micro` times all Q5_K routed-down
   expert banks without a long ramp. It matches the attribution scale and shows
   A3B Q5 down at only `183 GB/s` for one token, improving to `236 GB/s` at
   synthetic `tokens=16`, while A10B is already `325-354 GB/s`. This keeps A3B
   Q5 down real but points away from command overhead and toward the small-shape
   work unit/dequant dataflow. v0.344 cracks that specific small-K issue: the
   default Q5 down kernel used only two of four K-lane groups at `f_exp=512`, and
   the new R2 kernel computes two output rows per simdgroup. Rollback is
   `QWEN_DECODE_MOE_Q5_DOWN_K512_R2=0`. A3B default-vs-rollback improves
   `ctx128` (`106.6/104.7 -> 108.2/109.8 t/s`), `ctx1024`
   (`100.7/100.6 -> 103.7/104.0`), and `ctx8192`
   (`93.8/94.0 -> 98.0/97.7`, `+4.0-4.5%`). This branch is now banked for
   `f_exp=512`; future Q5 work should target new shapes or broader dataflow, not
   reopen NSG/R2 variants for the same kernel. v0.345's fresh llama.cpp b9833
   guard keeps MoE decode green externally (`A3B tg128 1.42x`, `A10B tg128
   1.25x`), so continue this lane only for hardware-headroom/dataflow wins, not
   parity panic. v0.346 kills the analogous Q8_0 K512 row-widening idea for MoE
   shared down: the sidecar improved the named phase but regressed A3B `tg128`,
   so do not promote shared-down-only micro/phase wins without an end-to-end gate.
   v0.371 kills the closest A10B analog, a dirty Q5 `f_exp=1024` row-pair down
   sidecar. It is exact and helps batched `tokens=16` micro (`36.04 -> 31.95 ms`),
   but single-token decode only moves `2.483 -> 2.424 ms`, far below the phase
   gate. Future Q5 down work needs a different dataflow/counter signal, not
   another row-packing variant. v0.373 splits the route bucket: A10B `ctx8192`
   route logits are `0.48 ms`, while top-k/shared preparation is `0.86 ms`; A3B
   `ctx32768` splits `0.30/0.68 ms`. Do not reopen router-logits mat-vec work,
   but keep post-logits route/slot-prep layout as a measured secondary branch if
   a deeper split or cached-route lower bound shows a recoverable `>=0.15 ms`
   on A10B or `>=0.10 ms` on A3B. v0.374 adds the unsafe route-noop lower-bound
   diagnostic and it clears that gate: A10B `ctx8192` GPU moves `22.71 ->
   20.37 ms`, while A3B `ctx32768` moves `10.82 -> 9.32 ms`. The phase delta is
   route removal plus consumer effects (`routed gate/up` improves `3.18 ->
   2.41 ms` on A10B), but a dirty sorted-topk sidecar regresses, so slot order
   alone is falsified. The next route branch must be exact route-cache replay or a
   top-k/shared-gate split; do not jump straight to a production route kernel.
   v0.375 adds that top-k/shared-gate split: separate shared-gate is far slower
   than the fused route kernel (`1.50 ms` A10B and `1.04 ms` A3B versus fused
   topk/shared `0.86/0.68 ms`), so production should keep shared gate fused. The
   remaining route kernel hypothesis is a fused simdgroup-local top-k or exact
   route-cache replay; do not build a split shared-gate route path. v0.376 kills
   the simple fused SG top-k version: it is correctness-safe in an A3B smoke but
   regresses fused route topk/shared to `1.25 ms` on A10B and `1.00 ms` on A3B.
   Route now needs exact route-cache replay before more route kernels.
   v0.377 adds a CPU-route phase diagnostic that removes GPU route while writing
   CPU-computed route buffers. It saves roughly the route bucket (`23.85 ->
   22.63 ms` A10B, `11.82 -> 10.90 ms` A3B), but routed gate/up worsens, so it is
   not exact replay. Keep it as infrastructure; pivot route implementation work
   to exact GPU replay later and move active optimization back to larger buckets.
   v0.385 adds exact GPU route replay for phase mode: the real route kernels still
   populate top-k/weight/shared buffers, but their time is excluded from the phase
   sum. This keeps consumers stable while showing exact recoverable budget: A10B
   ctx8192 phase sum moves `23.98 -> 22.62 ms` with routed gate/up/down unchanged,
   and A3B ctx32768 moves `11.76 -> 10.75 ms` with consumers unchanged. Production
   route work is now justified, but only for a post-logits/top-k/shared route
   design that saves `>=0.25 ms` A10B or `>=0.15 ms` A3B and moves end-to-end
   decode without perturbing MoE consumers. v0.386 kills the lowest-risk
   production attempt: a dirty exact fused top-k/shared barrier-diet sidecar only
   moved A10B ctx8192 `moe route topk/shared` `0.86 -> 0.82 ms` and phase sum
   `23.98 -> 23.92 ms`, far below gate. Do not reopen barrier-only route kernels;
   the next route implementation needs a materially different top-k work shape or
   a new counter signal. v0.387 kills the obvious candidate-compression variant:
   exact simdgroup-local candidate lists plus a single-thread merge regressed
   A10B `moe route topk/shared` `0.86 -> 3.87 ms`. Together with v0.376 and
   v0.386, this closes local top-k rewrites as the next route bet unless new
   counters identify a different mechanism. v0.390 repeats the exact route budget
   on current code but demotes exact route implementation: A3B route split is
   logits `0.30 ms` plus fused topk/shared `0.68 ms`, A10B is `0.48 + 0.86 ms`,
   and replay deltas are `1.01/1.34 ms`. A3B deep split shows the existing
   topk/shared fusion is already the important structural overlap (`1.43 ms`
   separate versus `0.68 ms` fused). Reopen exact route only for a prototype that
   saves `>=0.4 ms` A3B or `>=0.5 ms` A10B route total without moving time into
   consumers; otherwise pivot to captured MoE compute work.
   v0.378 adds `moe-gateup-micro` as the next active harness. A10B Q4 gate/up
   micro is close to phase (`3.0399 ms` versus `3.18 ms`), so bounded A10B Q4
   gate/up shape probes can use it. A3B synthetic top-k is not representative yet
   (`1.5179 ms` micro versus `1.07 ms` phase), so A3B promotion still requires
   captured route-pattern replay or full phase/ctx confirmation. v0.379 proves
   that guardrail: dirty `NR0_Q4K=4` improves A3B synthetic micro (`1.5179 ->
   1.3430 ms`) but fails full phase (`1.07 -> 1.09 ms`) and regresses/noises A10B.
   Do not continue row-widening without captured route patterns.
   v0.380 bounds the lm-head greedy-fusion shelf: `lm argmax` is only
   `0.06 ms` on A10B and `0.05 ms` on A3B, so exact `lm_head+argmax` fusion is
   below the implementation gate. Treat lm-head as projection-weight-read work,
   not sampler overhead.
   v0.381 recalibrates current decode slopes: A3B drops `107.8 -> 88.7 t/s` from
   `ctx128` to `ctx32768`, A10B drops `45.2 -> 40.7 t/s` from `ctx128` to
   `ctx16384`, and dense 27B drops `25.6 -> 23.2 t/s` from `ctx128` to `ctx8192`.
   No new cliff appears; use phase-budgeted structural branches, not more local
   route/gateup row tweaks. v0.382 adds captured hidden+route replay to
   `moe-gateup-micro`, which makes the harness phase-faithful on A3B ctx8192
   (`1.0836 ms` replay versus `1.08 ms` phase) and A10B ctx8192 (`3.1147 ms`
   replay versus `3.19 ms` phase). It also rejects the known dirty NR4 false
   positive (`1.1014 ms` captured versus `1.0836 ms` default). Use captured
   replay as a gate for future Q4 gate/up variants, but do not keep mining
   row-width tweaks without a structural byte/dataflow rationale. v0.475 kills
   the adjacent slot-packing work-unit proof: packing two routed Q4 gate/up slots
   into one 4-SG threadgroup is not better on captured A3B ctx8192 replay
   (`1.0207 -> 1.0317 ms` GPU) and was much worse on synthetic replay. Do not
   reopen Q4 gate/up slot-pair/row-width shapes unless the proposal changes expert
   dataflow or bytes, not just threadgroup packing.
   v0.391 extends the same captured-route discipline to routed down and makes the
   down microbench production-faithful: A3B Q5 down captured R2 is `0.814 ms`
   versus phase `0.82 ms`, and A10B captured Q5 down is `2.595 ms` versus phase
   `2.56 ms`. Future MoE compute variants should clear captured gate/up or down
   before phase promotion; synthetic-only wins remain non-promotional.
2. Decode long-context attention/KV second pass: v0.293 proves A3B group8 decode
   attention still had high-EV execution-shape headroom (`ctx16384` attention
   `4.80 -> 3.60 ms`, throughput `72.8 -> 82.2 t/s`). Attention remains large in
   v0.321 (`2.28 ms`, `20.5%` at A3B `ctx8192`), but prior KV-byte-only variants
   are falsified: v0.299 kills the straightforward MoE Q8-KV subgroup reader,
   v0.300 kills smaller subgroup splits, v0.313 kills cooperative four-simdgroup
   subgroup packing, and v0.315 kills simple group-tile/reread reduction. The
   v0.333 keeps attention/KV as a measured secondary lane at `21.3%` A3B and
   `18.3%` A10B `ctx8192`, behind MoE FFN on both. The next credible proof must
   change execution shape without losing occupancy, or
   show a counter signal beyond byte count/address order. Do not reopen subgroup
   shape, `NWG`, `TILE_C`, ggml-Q8 KV, or broad KV-head-major layout without new
   data. v0.334 banks a small epilogue cleanup by fusing gated-attention
   `sigmoid+mul`: A3B `ctx8192` moves `92.1/92.2 -> 93.6/93.3 t/s`, A10B is
   neutral/noise, and a dense 27B `ctx8192` guard moves `21.9 -> 22.6 t/s`. Treat
   this as a shelf win; the remaining attention branch is still the v4 body/KV
   path, not more scalar epilogue passes. v0.335 then falsifies the most concrete
   layout-only KV proof: synthetic head-major F16 K/V is exact but flat/slower at
   A3B `ctx16384/32768` and A10B `ctx8192/16384/32768`. Do not build production
   head-major KV sidecars without a new counter signal or a body rewrite that
   changes more than address order. v0.342 promotes attention as the true-long
   scaling limiter but demotes a same-byte rewrite: `qwen-bench attn-intra` puts
   the v4 main body at `623-649 GB/s` estimated KV bandwidth on A3B and
   `533-583 GB/s` on A10B, with reduce only `0.03-0.05 ms/layer`. Reopen the
   attention body only for byte reduction, hidden-traffic counters, or a
   full-model `ctx32768` prototype that moves throughput despite those body
   numbers.
    v0.356 re-anchors current A3B true-long decode after the low-bit fixes:
    Q4_K_M drops `107.7 -> 96.5 -> 76.2 t/s` at `ctx128/8192/32768`, and
    `ctx32768` phase split puts attention first (`5.09 ms`, `38.0%`). The
    `attn-intra` body matches that scale and estimates `~692 GB/s` on the main
    pass, while reduce is only `0.34 ms` extrapolated. A dirty group8 Q8-KV
    sidecar regressed badly (`ctx8192/32768` `96.5/76.2 -> 85.5/55.8 t/s`) and
    forced F16 full-group tile8 also regressed (`89.8/62.0`), so do not reopen
    Q8-KV by giving up the default group8 tile2 execution shape. The live
    long-attention branch must reduce bytes while preserving occupancy, bring
    counters for hidden traffic, or deliver an end-to-end `ctx32768` prototype;
    same-byte retunes and reduce work stay deprioritized. v0.367-v0.369 bank the
    concrete medium/true-long attention wins, but v0.370 kills the next partial
    byte-reduction proof: normalized-half partials are exact enough, yet only move
    A3B `ctx32768` `attn-intra` `1.014x`. v0.388 refreshes current `ctx32768`
    `attn-intra` knobs and keeps the default tile4/NWG256 path best: default is
    `0.3469 ms/layer` (`3.47 ms` extrapolated), while tile2/NWG64 is
    `0.5265 ms`, tile8/NWG256 is `0.3971 ms`, and tile4/NWG128 is `0.3810 ms`.
   `NWG=128` halves reduce but slows the main body; tile8 reads fewer logical
   bytes but loses occupancy. Existing group-tile/NWG retunes remain closed.
   v0.476 reopens the lane with a real body/dataflow change instead of another
   selector retune: the G8 C64 score-broadcast sidecar keeps score/weight state
   lane-local and uses `simd_shuffle` during PV, deleting the main-pass
   score/weight TGM round-trip while preserving F16 KV and the reduce contract.
   It is exact and moves A3B main `0.1076 -> 0.1019 ms` at ctx8192 and
   `0.1948 -> 0.1755 ms` at ctx32768; full ctx-sweep is neutral at 4096 and
   positive at ctx32768 (`86.5 -> 88.9 t/s`) but below the `>=5%` default gate.
   Keep `QWEN_ATTN_V4_G8_BCAST=1` as an opt-in true-long sidecar and recheck on
   real long rollouts before promotion. v0.477 kills a dirty attempt to broaden
   the same score-broadcast body to G16/tile4/C64 because correctness failed in
   non-target group6 rows. v0.478 fixes that implementation issue and proves the
   G16 body exact, but the full A10B decode gate still fails: attention-intra
   improves directionally while `ctx8192 --window 4 --fresh-per-checkpoint`
   regresses `42.5 -> 41.4 t/s`. Do not keep a G16 bcast env knob or reopen this
   exact shape. v0.480 then tests a more literal read-once-ish G16 shape: one
   4-simdgroup TG stages half of V at a time to cut tile traffic from roughly
   `K4 + V4` to `K4 + V1`. It is exact, but A10B `attn-intra ctx8192` regresses
   badly (`0.4340 -> 0.5614 ms`, main `0.1274 -> 0.2231`). Together with the
   v0.463 A3B V-stage kill, close TGM-staged V/KV variants unless counters prove
   read savings beat synchronization/TGM/occupancy loss. Keep attention below
   MoE/GDN unless the next proposal brings hidden-traffic counters, a non-TGM
   read-once execution shape, or an end-to-end `ctx32768` prototype that moves
   throughput rather than partial storage or reduce rows. v0.481 also kills a
   dirty dense group6 bcast broadening attempt at correctness: the added
   specialization disturbed non-target group4 rows (`cos=0.931786`). Do not keep
   broadening the G8 bcast template across groups without a narrow harness and a
   clean broad-suite pass.
3. GDN decode projection mechanics, with local Q8 retunes closed: v0.336 adds
   correctness-breaking no-op attribution for the GDN projection lane. The
   recoverable lower-bound budget is
   large at `ctx8192`: A3B front/out no-op moves GPU `10.19 -> 8.41/9.58 ms`, and
   A10B moves `23.06 -> 18.38/20.88 ms`. The subprojection ladder convicts QKV,
   Z, and OUT, while beta/alpha are flat/noise. Do not spend the next branch on
   beta/alpha fusion. The next credible implementation is an exact-shape Q8
   projection microbench for `h -> conv_dim`, `h -> v_dim`, and `v_dim -> h`, with
   a required primitive win before production decode changes. v0.337 adds an
   aggregate A10B `ctx8192` phase split: QKV is `2.98 ms` / `12.2%`, Z is
   `2.04 ms` / `8.3%`, beta+alpha are only `0.45 ms` combined, and OUT is
   `2.29 ms` / `9.4%`. A correctness-safe Q8_0 R4 row-widening sidecar failed the
   gate (`92.6/94.0 -> 92.5/91.8 t/s` on A3B and `42.6 -> 42.0 t/s` on A10B), so
   do not retread Q8 row-count tweaks without counter evidence. The live Q8 branch
   must change projection dataflow, fusion, or packing; otherwise return to MoE
   FFN execution shape or attention/KV body work. v0.338 adds the exact-shape
   `qwen-bench gdn-proj-micro` harness and shows the current primitive is already
   near the measured stream roofline on weight bytes alone: A10B QKV/Z/QKV+Z/OUT
   are `485/473/476/461 GB/s` versus the `474 GB/s` stream anchor, while A3B is
   `435/418/445/395 GB/s`. This demotes local GDN Q8 mat-vec retunes; reopen only
   for structural byte reduction or a primitive proof that beats this harness.
   v0.340 banks the known dense scheduling overlap instead: default dense
   concurrent-GDN moves `tg128` by `+2.4-3.8%` across 0.8B/2B/4B/9B/27B. v0.342
   keeps structural GDN byte reduction alive for A10B long decode because
   qkv+z+out totals `8.18 ms` / `26.7%` at `ctx32768`, but the branch must reduce
   bytes, fuse a larger dataflow, or prove a primitive win; do not spend another
   pass on row-count retunes. v0.372 splits the GDN tail and demotes local tail
   work: A10B `ctx8192` tail step is only `0.69 ms`, and A3B `ctx32768` tail step
   is only `0.34 ms`; conv, L2, and norm are smaller. GDN remains a projection or
   structural byte/dataflow branch, not a tail microkernel branch. v0.383 refreshes
   that projection gate on current HEAD: A10B `qkv+z` micro runs `2.4065 GB` in
   `4.8675 ms` (`494.4 GB/s`) and A3B runs `0.8022 GB` in `1.7949 ms`
   (`446.9 GB/s`), while A10B beta+alpha projections are only `0.39 ms` in the
   ctx8192 split. Do not build local GDN projection row-shape or `qkv+z` fusion
   branches without byte elimination, an algorithmic dataflow change, or counter
   evidence beyond weight streaming.
4. A10B memory-capacity/tooling hygiene: v0.315 shows a `ctx-sweep` that allocates
   for `32768` up front can poison even A10B `ctx570/2464` rows (`~0.6 t/s`), while
   capped sweeps are normal (`43.8/38.4/43.1 t/s` through `4096`, `41.6/39.9 t/s`
   at `8192/16384`). v0.316 adds `ctx-sweep --fresh-per-checkpoint`, which
   validates A10B `ctx570/2464` at `43.5/38.5 t/s` with right-sized sessions.
   v0.317 adds `decode --kv-capacity` and shows large unused capacity is mainly a
   cold first-touch/residency hazard: A10B `ctx570 cap32768 --no-warmup` has a
   `3089 ms` first decode token, but warmed `cap32768` returns to `43.3 t/s`.
   Use fresh/right-sized modes for measurement, keep capacity explicit in product
   benches, and handle cold-start residency separately from hot kernel tuning.
5. Utilization scoreboard plumbing, bundled with long-context decode: v0.285 adds
   the one-time roofline calibration packet (`474 GB/s` stream,
   `~12.5-12.8 nominal TFLOP/s` Q4_K mat-mat, and `3.03 TFLOP/s` scalar-FMA
   sanity), v0.293 adds `scripts/profile/decode_phase_roofline.py` for phase-level
   active-byte reads, v0.312 adds optional tg JSON command/encoder/dispatch
   accounting, v0.315 adds decode attention KV byte estimates, and v0.321 adds
   active decode weight bandwidth from bench JSON or manual context-sweep rows.
   v0.326 adds `scripts/profile/decode_ctx_sweep.py` for order-aware env-variant
   `ctx-sweep` packets, closing the measurement gap that let v0.325's false Q5
   R2 win survive too long. v0.342 adds visible `qwen-bench attn-intra` so
   attention main/reduce body evidence is not trapped in ignored tests. v0.343
   adds visible `qwen-bench moe-down-micro` for fast Q5 routed-down primitive
   gates before full long-ramp decode sweeps. v0.345 moves the pinned
   llama.cpp lock to b9833 (`c818263f2`) and fixes family digests to use the
   measured `474 GB/s` stream anchor, so future scoreboards should not silently
   drift back to the stale May benchmark or the old `546 GB/s` spec-sheet peak.
   The first A3B Q4 `ctx8192` sample is `93.0 t/s`, `2.6215 GB/token`, and
   `243.8 GB/s` (`51.4%` of measured stream roofline); attention KV subgroup
   traffic is `0.6711 GB/token` at `294.3 GB/s`. Next, keep using this output to
   explain the active branch with decode t/s, wall/GPU time, phase split,
   dispatch/encoder counts where available, and defensible bytes/token or
   bandwidth proxies. This is not dashboard work; it is the decision spine for
   capacity, execution-shape, and roofline calls.
6. Packed-verify/spec decode, structural rewrite only: v0.314 fixes production
   `qwen-bench dflash` scratch allocation and shows real narrative `static-16`
   decode is catastrophically slower than no-spec despite good acceptance
   (`0.349x` at `570` prompt tokens and `0.261x` at `2464`). Treat acceptance as
   interesting but not sufficient. Do not spend time on policy knobs; promote spec
   only if packed verify/KV restore/logits accounting identifies one removable
   structural villain and a proof can plausibly clear `>=1.25x` decode on real
   prompts after full verify cost. Fresh llama.cpp b9833 has a real
   `llama-cli --spec-type draft-mtp` path, but the local smoke does not reset
   priorities: no-spec 27B-MTP generation was `23.0 t/s`, while `draft-mtp`,
   `n_max=3`, `p_min=0.75` was `23.8 t/s` and lower prompt throughput. Keep MTP
   as a separate algorithmic target; do not mix it into no-spec parity boards.
   v0.443 EXECUTES this item's precondition and finds the villain: the skinny-N
   mat-mat kernel family runs `3-6x` off weight-stream on every verify/drafter
   projection shape (`44-136 GB/s` vs mat-vec's `288-382` on identical
   tensors); verify(N=16) is `5.2-5.5x` a decode step and `82%` of the DFlash
   step, drafter `13%`, glue/restore `<6%` (the v0.314-era restore/logits
   overhead is already engineered away). Durable alpha at 256 tokens:
   code `3.34`, narrative-start `2.11`, narrative-tail `0.386` — and the
   adaptive policy fails to protect low-alpha text (`0.524x`). M2 is
   sanctioned in two tracks: (a) skinny-N mat-mat retune (`>=250-300 GB/s`
   at N<=16 on the `packed_verify_skinny_gemm_micro_27b` harness; start from
   the existing N16 simdgroup kernels — scalar row-parallel is ALU-bound at
   `~38 ops/weight` and caps below target), (b) alpha-aware policy cutoff.
   Priced: `~1.4x` mid-alpha / `~2.0x` code decode-only if (a) lands; gate is
   the M1c packet re-run clearing `>=1.25x` on real prompts at 570/2464 ctx.
   Track (a) also serves replay S8 projections and MoE small-batch decode.
   v0.444 then EXECUTES track (a) and FALSIFIES the branch: the deficit is a
   per-dispatch machinery floor (~0.27 ms per skinny GEMM: sa-swizzle stores +
   fragment loads + loop), not bytes — a raw-block-staged v2 kernel wins +24%
   on the hot-L2 micro but ~0% in production verify (occupancy trade), and an
   A-path-FREE probe inside the verify still costs 168.6 ms = 4.2x a decode
   step. All-heroics ceiling (machinery -25%, A-path halved, v0.73b GDN tail
   kernel, packed-N attn) is ~1.0-1.1x at the best measured alpha — under the
   bar. F32 dot-product alternatives are FMA-roofline-capped (~250 GB/s-equiv)
   and half-acc designs die on activation-reuse register/L2 tension. DFlash
   verify-cost work and the M2b policy co-fix are DEMOTED; reopen only with
   (a) durable alpha_chain >= 4.5, (b) a persistent-kernel/fused-layer verify
   structure that amortizes per-dispatch machinery, or (c) MTP-side economics.
   Do not propose N-tile/staging/barrier variants against the N16 family.
7. Promotion-grade paired residual search for prompt prefill: v0.279 cracks the
   tuned small-dense control except for parity/noise 2B `pp512`; current sentinel
   rows keep 27B/A3B/A10B prefill won after discarding A10B cold noise. Reopen
   prefill only if a paired repeat exposes a real current-default red cell. v0.318
   clears the stale A3B true-long row: synthetic `pp34502` is `1.196x` and a real
   `56774`-token chaos rollout is `1.168x` versus pinned llama.cpp. Do not reopen
   fused online-softmax, chunk policy, or GDN-scan work from the old true-long row
   alone.
8. BF16 accounting/differential audit, explicitly scheduled only: BF16 remains a
   catastrophic demoted red cell, but v0.319 falsifies the most concrete
   grouped-MoE scalar A-load hypothesis. Dirty `bfloat4` loads regressed BF16
   grouped `routed_swiglu` (`130.80 -> 217.73 ms`) and `routed_down`
   (`64.78 -> 98.30 ms`). v0.320 then rechecks the existing
   `QWEN_MATMAT_BF16_BFLOAT_ACT=1` sidecar: traced GDN/attention phases collapse
   (`gdn_qkv 1141.65 -> 99.97 ms`, `attn 611.94 -> 69.44 ms`), but no-trace
   production moves only modestly/noisily and the paired row remains `0.059x`
   llama.cpp (`73.63` versus `1239.10 t/s`). Do not default bfloat-act or reopen
   BF16 kernel work until named production accounting explains `>=90%` of BF16
   `pp512` wall, or one identified category is `>=40%` of wall and has a plausible
   `>=1.5x` no-trace production fix. v0.489 independently rechecks grouped BF16
   `bfloat4x4` A loads and again loses on warmed A3B BF16 `pp512` samples
   (`657.3/1373.2` scalar versus `645.0/1371.3 t/s` vector). Do not reapply
   grouped-MoE vector A-loads. v0.544 executes the final measurement gate at
   current HEAD. Warmed paired `pp512` is only `0.9785-0.9827x` llama.cpp;
   combined causal no-ops explain only `0.7643-0.7658`, below the `>=90%` arm.
   No-FFN and no-routed clear `>=40%`, but pinned upstream llama.cpp supplies
   zero operation-profile records, so the required same-shape `>=1.5x`
   replacement proof fails under preregistered missing-data handling. Close
   structural BF16 work; no local-fork profiler substitution, `pp1024` rescue,
   extra no-op row, or old kernel-family retread is authorized.
9. Short MoE decode execution-shape, only with fresh evidence: v0.292-v0.311 make
   decode materially greener, but v0.312 says the current default is already
   GPU-active at `~95.8-97.8%` wall on warmed A3B/A10B `tg128`. The GDN-concurrent
   rollback win is real (`+8.8%/+6.0%`) but already banked, and disabling it mainly
   increases GPU active time with the same dispatch count. Reopen only for a new
   GPU-overlap or byte-reduction mechanism with a measured `>=2%` tg ceiling on
   both MoE targets, or `>=3%` A10B with A3B neutral. Do not optimize encoder count
   for its own sake: serial one-encoder rollback is slower.
10. Remaining dense prompt attribution, only if a paired red cell survives: the
   latest phase trace shows `gdn_gated` and `gdn_prep_l2` are now tiny. If short
   dense still regresses in a clean repeat, target projection/FFN, GDN step, or
   attention body with a shape-matched differential. Do not retread attention
   defaults, fast-path coverage, residual-add fusion, N64 policy, shared-memory
   policy, GDN token-channel parallelization, command-buffer streaming, GDN
   matvec fallback, or HD128 normalization row count.
11. Hardware-headroom audit watchlist, gated by primary-row evidence: ICB/MTL4
    encode-once decode, fused online-softmax prefill attention, chunked
    delta-rule GDN, production residency/warm expert banks, RoPE precompute,
    targeted kernel resource annotations, host encode allocation/PSO cleanup, and
    structural decode fusions are all plausible paths to dominate beyond
    llama.cpp. They need a trace-proven idle/bandwidth/phase gate before
    implementation, not merely a green or red llama row. GDN prep falsifiers do
    not falsify chunked `gdn_step`; they are different kernels and mechanisms.
    The specific post-audit gates are:

    - RoPE precompute: replace per-pair `pow` with inv-freq or sin/cos reuse only
      if an isolated kernel win converts into `>=0.5-1%` end-to-end on at least
      one primary decode or prefill row with no drift.
    - PSO cache, `view_subrange`, and clone cleanup: first capture warmed CPU
      encode/alloc evidence showing `>=0.3 ms/token` or `>=1%` wall in a primary
      decode path; current GPU/wall rows do not justify an implementation branch
      by themselves.
    - `max_total_threads_per_threadgroup` and simdgroup-barrier pruning: test only
      on exact hot kernels with primitive microbench wins, then run clean family
      guards. Do not blanket-annotate kernels; a wrong cap can reduce occupancy.
      v0.485 tests the concrete Q4/Q6 mat-mat SG-barrier deletion and kills it:
      correctness is green, but 27B `pp1024` regresses (`235.384 -> 231.967 t/s`)
      while `pp4096` is flat/noise. Do not repeat this without shader-counter
      evidence for a specific kernel.
    - Hot-weight K-quant repack: v0.486 executes the smallest exact compressed
      prepack proof, reordering one 27B Q4_K FFN gate tensor into N16 mat-mat
      consumption order without predequantizing. Correctness is bit-exact, but
      the primitive only moves `0.4078 -> 0.4030 ms/dispatch` (`1.012x`), far
      below the `>=1.12-1.15x` gate for model wiring. Demote broad K-quant repack
      until counters identify a larger layout stall or a different tensor/layout
      has a named primitive win.
    - Decode fusions: prefer byte-reuse/dataflow fusions such as GDN `qkv+z` over
      residual/elementwise shelves, but require a primitive proof that beats the
      current near-roofline projection harness or a phase bucket of at least `5%`
      with an end-to-end `>=1%` win.
    - Mmap no-copy, heaps/residency sets, binary archives, and ICB/MTL4 remain
      product/cold-start/encode-bubble work until traces show hot throughput is
      host- or residency-limited.
    - Kernel-bypass / persistent-megakernel framing is useful as a probe source,
      not a roadmap reset while hot decode remains mostly GPU-active. Measure
      abstraction ceilings directly: command/encoder/dispatch counts, warmed CPU
      encode time, allocation count, and theoretical cycles/token. Only escalate
      ICB, persistent work queues, argument buffers, or GPU-side graph traversal
      if those probes show command starvation, host bubbles, or cross-op data reuse
      that a normal kernel path cannot access. v0.389 proves autonomous hardware
      counters are not currently available here beyond timestamps, so hidden-stall
      or bandwidth-counter claims require manual Xcode GPU capture rather than
      routine `xctrace` or in-process counter sampling.
    - AMX/ANE coprocessor, hierarchical `lm_head`, layer skipping, sparse FFN,
      KV clustering, attention-head skipping, and residual/GDN-state precision
      changes are quality or fabric research lanes. They need explicit quality
      gates and a measured phase ceiling before speed claims.

12. Quant breadth guardrails: v0.240 proves prompt prefill for local 4B
   `UD-Q2_K_XL` and `UD-IQ2_M` at `pp512/1024/4096`, v0.241 proves their
   `tg128` decode sentinels, and v0.242 keeps adjacent 4B `Q3_K_M`, `IQ4_XS`,
   and `Q4_K_M` clean at `pp512/4096/tg128`. Reopen low-bit dense kernel work
   only when a paired file or static audit exposes a fresh miss.
13. `IQ4_XS` grouped-down precision/perf audit: strict internal-state cosine is
   below the usual `0.999` floor on `UD-IQ4_XS`, and F32 gate/up reproduces the
   same envelope. The v0.234 primitive grouped-down oracle passes (`cos=1.0`,
   `max_abs=1.386e-5`), so the next accuracy check needs real captured
   `moe_inner` activations rather than another synthetic row-stride oracle.
14. Bounded llama.cpp / counter attribution: verify whether llama.cpp is actually
   faster inside comparable routed gate/up/down arithmetic, or whether remaining
   differences are orchestration, fused GDN, graph fusion, warm/cold accounting,
   or profile scope. The pinned upstream b9481 build does not expose
   `GGML_METAL_PROFILE_OPS`; the local fork has a profiling patch, but any
   attribution claim needs either a canonical profiling patch or a clearly marked
   non-scoreboard capture before more kernel code.
15. True multi-expert work-unit reset: if attribution proves the gap is inside Q4
   `<8` SwiGLU arithmetic, design a kernel that changes the work unit more deeply
   than R16, MR32, or split gate/up. It must improve `<8` by at least `25-30%`
   before any end-to-end tuning.

The sections below preserve the rationale and reopen criteria from earlier
sprints; the current rank above overrides stale branch ordering when they
conflict.

Current MoE short-branch rule:

- Target active underfilled buckets, not route work or threshold policy. Q4
  `pp512` fine-bin traces show `<8` is only `6.3%` of routed slots but costs
  `3.485 ms/k-slot` in SwiGLU and `3.278 ms/k-slot` in down, versus `>=64` at
  `0.498` and `0.252` respectively. Q3/Q6/Q8 SwiGLU traces show the same shape.
- The next branch should be attribution-first, not another Q4 tiny-SwiGLU retread.
  If attribution proves the residual is inside comparable `<8` SwiGLU arithmetic,
  require a genuinely new multi-expert work unit and gate on bin-time movement
  first, then A3B `pp512` GPU time, then no regression at `pp768/1024+` and broad
  quant generalization.
- Down now has a correctness-safe force-only proof. It reduces the target bin but
  is not enough by itself to justify defaulting. The first analogous SwiGLU port
  is falsified; future SwiGLU work must explain the dual-dequant/epilogue cost
  before adding another tiny kernel. The split gate/up sidecar also failed, so
  the remaining SwiGLU path is a real execution-shape change, not merely
  unfusing the current grouped kernel. The MR32 attempt failed that bar too;
  pause Q4 tiny-SwiGLU code until attribution identifies a new mechanism.

### 1. Hypothesis: Dense 27B residuals have pivoted from attention to GDN/FFN

Optimizes: Qwen3.6 27B dense prompt prefill after the G6 matrix-attention body
received pointer-hoist and full-tile specializations.

Why it is at the top:

- The v0.165 full-tile KQ/KQV kernels change the dense read: at `pp4096`, KQ/KQV
  are now `136/135 ms`; at `pp16384`, they are `2170/2377 ms`, with qwen softmax
  still much faster than llama.cpp. Dense attention body is no longer the obvious
  lcpp-scale residual.
- The remaining large named buckets are GDN/FFN projections and GDN step/state
  work. Earlier projection unrolls helped but did not finish the scoreboard; the
  earlier GDN recurrence-loop unroll was negative, so the next GDN branch must be a
  state/read-write/layout audit rather than another loop pragma.
- A10B routed MoE no longer explains a scoreboard gap under warmed methodology;
  qwen's warmed routed tail is faster/equal to llama.cpp and end-to-end is
  parity-or-better from `pp512` through `pp16384`.
- A3B matrix default plus Q6 coverage already wins most synthetic anchors; true
  long/real-rollout work stays live but is less clean than dense 27B.

Current design rule:

- Keep `QWEN_PREFILL_ATTN_MATRIX_G6=0` as the rollback path and preserve the
  ignored G6 prefix correctness gate plus `16/16` matrix phase coverage.
- Keep the branch scoped to KQ/KQV/softmax body. Projections are not the target
  unless a fresh phase trace says they regressed.
- Preserve `QWEN_PREFILL_ATTN_MATRIX_CAUSAL_SKIP=0` as rollback for the current
  default cleanup.
- Runtime compact-Q staging and q-head-major score/body layout are falsified:
  compact Q reduces KQ/KQV locally but its copy cost overwhelms the win, while
  q-head-major regresses KQ and leaves KQV flat. Revive only if Q is produced in a
  compact layout for free or a true llama.cpp-kernel clone needs the layout.
- Producer-side compact-Q was also tested as the cheapest "free Q layout" upper
  bound by making Q RoPE write the compact KQ view directly. It is correctness-safe
  but only moves KQ/KQV by a few milliseconds at `pp4096` while adding a small
  RoPE/scatter copy cost, far below the phase gate. Demote Q-layout-only work.
- Loop-pragma attention tweaks are falsified. A selective inner-loop-only unroll
  probe improved dirty phase rows but failed the clean gate: tiny/noisy long wins
  and clear `pp512/1024` regressions. Do not carry a duplicate env-gated kernel for
  sub-1% long-context movement.
- The first positive llama.cpp-mechanics cleanup is pointer/base-address hoisting in
  KQ/KQV. It moves 27B phase rows from `179/180 -> 160/158 ms` at `pp4096` and
  `2823/2972 -> 2516/2660 ms` at `pp16384`, with clean rows positive at
  `pp512/1024/16384` and noisy-positive at `pp4096`. Keep this style of isolated
  mechanical diff alive.
- Full-tile KQ/KQV specialization is the second positive mechanics cleanup. It
  moves phase rows to `136/135 ms` at `pp4096` and `2170/2377 ms` at `pp16384`,
  and post-commit clean rows reach `213.80 t/s` at `pp4096` and `200.16 t/s` at
  `pp16384`. Attention is now monitoring/tail-work, not the default top branch.
- GDN step NSG4 row grouping plus token-pointer increments are the first
  post-attention dense cleanups. Together they move `gdn_step` from
  `595.56 -> 482.15 ms` at `pp4096` and `2379.00 -> 1939.31 ms` at `pp16384`;
  clean rows now sit at `237.20/235.23/222.28/203.37 t/s` for
  `pp512/1024/4096/16384`.
- The matched qwen-vs-llama dense differential has been rebased after the GDN
  pointer increment cleanup. GDN step and attention body are now faster than
  llama.cpp in the serialized comparison; the only notable dense residual is a
  long-context FFN projection delta. Follow-up falsifiers did not make that
  residual actionable: chunk `512` loses to default `1024`, reduced mat-mat smem
  is phase-positive but not total-robust, and dense fused-Q4 FFN loses again in
  same-process A/B.
- Fresh v0187 paired comparison changes the dense rule again: 27B dense prefill
  is paired-won at `pp512/1024/4096/16384`, and the apparent short/medium gap was
  a stale-anchor/unpaired-drift artifact. Larger chunks, reduced QK smem, dense
  fused-Q4 FFN, and FFN up-before-gate have all failed promotion gates. Keep Q4
  N64 default-on and move dense 27B to guardrail mode.

Next branch order:

- First, use `scripts/profile/prefill_compare.py` for any future cross-engine
  prompt claim. Stale isolated llama.cpp anchors are no longer enough, especially
  at `pp512/1024/4096` where thermal/session drift can change the conclusion.
  The harness now accepts real `--file`/`--messages` qwen prompts; because
  `llama-bench` cannot consume prompt text, those llama.cpp rows are explicitly
  recorded as same-length synthetic anchors via `lcpp_prompt_mode`.
- Second, return to breadth/generalization: primary family paired guardrails,
  real-rollout prompts, and quant coverage gaps should rank above another dense
  27B microkernel unless a paired residual appears. `scripts/bench/family.py`
  remains the synthetic breadth scoreboard and now records cooldown plus
  thermal/memory context per command; use `prefill_compare.py` repeat blocks for
  promotion-grade narrow cells.
- Third, keep low-bit MoE defaultability tied to continuation/rank behavior rather
  than strict internal-state cosine alone. v0.233 shows `UD-IQ4_XS` final logits
  and continuation pass while internal GDN/KV cosines sit below `0.999`, and F32
  gate/up points at grouped `IQ4_XS` down as the envelope source.
- Fourth, keep MoE quant breadth active only where audit finds a real uncovered
  target file. A3B Q3/Q4/Q6/Q8 and `UD-IQ4_XS` now have `40/40` grouped coverage;
  prefer paired family guardrails and routed-tail attribution over adding kernels
  for quants not present in the target/product set.
- Fifth, small dense remains a paired mismatch but is no longer above MoE quant
  breadth. Start from 0.8B `pp512`
  and require 2B plus 4B/9B/27B canaries before promotion. v0.215 landed the
  low-risk GDN prep dispatch cleanup; recent falsifiers say the next branch
  should target FFN/GDN projection mechanics or dataflow, not attention, encoder
  coalescing/streaming, GDN matvec fallback, NSG8 GDN-step grouping, fused Q4
  SwiGLU N64, or broad low-threshold N64 policy.
- Defer reduced-smem promotion, fused FFN, and fused online-softmax/PV until fresh
  same-process or phase evidence crosses a total-throughput gate.

Acceptance gates:

- 27B `pp512/4096/16384` must improve without decode regression.
- A3B and A10B prompt defaults must remain neutral, with MoE coverage still
  `40/40` and `48/48` respectively.
- Promote only from AC-power rows with no thermal/performance warnings and a
  paired llama.cpp comparison on the same GGUF.

### Monitoring: A10B grouped routed down/dequant locality is not currently active

Optimizes: Qwen3.5 122B A10B prompt prefill after the G16 matrix-attention default
and the layer-46 Q5 gate/up coverage fix.

Current status: superseded by the `v0.155` warmed re-anchor. Keep this section as
the reopen criteria and historical rationale, not as the active top queue.

Why it was at the top / reopen criteria:

- A10B/G16 matrix attention is already default-on for the proven group-16 shape;
  post-default no-op rows made attention body a low-single-digit `pp512` lever.
- The first routed-MoE structural pass found and fixed the layer-46 fast-path escape:
  `Q5_K/Q5_K/Q6_K` gate/up/down now uses grouped Q5 gate/up SwiGLU and grouped Q6
  down, restoring `48/48` grouped routed MoE coverage.
- Warmed rows for that fix are large enough to be real: `pp512` `377.53 -> 448.31
  t/s`, `pp1024` `400.11 -> 513.66 t/s`, and `pp4096` `440.08 -> 484.54 t/s`
  directionally.
- After coverage is fixed, traces still point at routed projection/dataflow:
  `routed_swiglu` and `routed_down` dominate, while route/reduce/finalizer remain
  much smaller.
- cx adversarial review agrees the next exact branch should target grouped down
  locality/dequant or a locality-preserving `SwiGLU+down` sidecar, not another
  attention or route-side branch.

Current design rule:

- Keep `QWEN_PREFILL_MOE_GROUPED_Q5_GATEUP=0` as the rollback path and keep the
  dedicated layer-46 Q5 SwiGLU oracle in the gate.
- Use warmed/interleaved methodology for A10B. Cold no-warmup A10B rows can be
  dominated by first-touch/model-residency effects and should not drive decisions.
- Require fast-path coverage assertions for MoE dtype variants. A3B `37/40` and
  A10B `47/48` are now canonical failure modes.
- Do not open new A10B attention branches until routed `SwiGLU/down` gets a fresh
  structural pass on the new 48/48 baseline.

Acceptance gates:

- Dedicated Q5 grouped-SwiGLU oracle on layer 46 must remain green, including
  poison/partial/zero-count bucket coverage.
- Default and rollback A10B prefill-vs-single smokes must stay green.
- A10B `pp512` phase coverage must show `48/48` route/grouped-routed/shared labels.
- Promote broader claims only from warmed AC-power rows with no thermal/performance
  warnings; prefer `pp512`, `pp1024`, `pp4096`, and at least one real rollout.
- For the next branch, require a combined routed-tail win, not just a standalone
  down-kernel or SwiGLU microbench win.

### 2. Hypothesis: A3B matrix attention plus grouped Q6 is the lcpp-cracking prefill candidate

Optimizes: A3B MoE prompt prefill from `pp320` through true-long after the Q6-down
grouped-path escape fix moved the board.

Why it is back at the top:

- Matrix attention is now default-on for the proven A3B group-8 shape, with
  `QWEN_PREFILL_ATTN_MATRIX_G8=0` as rollback. Fresh current-HEAD rows show the
  auto/default promotion materially beats the previous packed-attention default:
  `pp128` `~+7%`, `pp512` `~+13%`, `pp1024` `~+15%`, `pp4096` `~+18%`, `pp16384`
  `~+25-33%`, and real `v02_reva` `34.5k` `655.34 -> 837.18 t/s`.
- Matrix attention passes the full ignored A3B prefill-vs-single gate with auto
  default, including prefix `4096` / `8191` active shapes. The local matrix oracle
  keeps its documented numeric envelope (`cos >= 0.9999`, `max_abs <= 2e-2`), and
  the production gate is the stronger final logits / GDN / KV model-state check.
- The biggest A3B prompt win in this sprint came from a silent fast-path miss, not
  from another grouped-SwiGLU tile: A3B had `40` MoE layers but only `37` grouped
  routed trace labels because three late expert-down tensors are `Q6_K`. Closing
  that escape moved `pp320` `~830 -> 1061.45 t/s`, `pp512`
  `824.46 -> 1172.07 t/s`, and `pp1024` `910.43 -> 1287.43 t/s` while leaving
  warmed A10B neutral.
- Therefore, the next exact sprint should make the lcpp-cracking matrix path safe
  to promote while keeping the fast-path coverage matrix in the gate. Do not infer
  coverage from final-logit correctness alone.
- Post-Q6 no-op ceilings still leave routed MoE as a major lever, but not the only
  plausible lcpp delta: A3B `pp1024` baseline `1287.43 t/s` rises to
  `1578.34 t/s` with attention body skipped and `1979.46 t/s` with routed MoE
  skipped; A3B `pp4096` rises from `1198.11 t/s` to `1583.70 t/s` no-op
  attention and `1816.71 t/s` no-op routed. Shared MoE is small (`1332.52 t/s` at
  `pp1024`).
- Earlier no-op ceilings also showed routed FFN dominating shared FFN on the
  then-current default path:
  A3B `pp320` moves from about `765 -> 1136 t/s` with routed off, while shared
  off only reaches about `790 t/s`; A10B `pp320` moves from about `305 -> 648 t/s`
  with routed off, while shared off only reaches about `309 t/s`.
- The matrix-attention branch shifts the clean `pp4096` budget back toward MoE:
  attention-body skip is only `~3-7%`, while routed-MoE skip reaches `1270 t/s`
  and broad FFN skip reaches `2498 t/s` against `~929-972 t/s` matrix baselines.
- Post-route-threshold A3B `chunk_p=320` live grouped-tail profile still puts
  `grouped_swiglu` first (`4.06 ms`, `57.2%`) and `grouped_down` second
  (`1.60 ms`, `22.5%`); route logits are now `0.27 ms` (`3.8%`).
- Fresh routed-tail microprofiles at `chunk_p=512` keep the same shape: A3B tail
  is `3.84 ms` (`2.31 ms` grouped gate/up/SwiGLU, `1.55 ms` down+reduce), and
  A10B tail is `10.30 ms` (`6.81 ms` grouped gate/up/SwiGLU, `4.25 ms`
  down+reduce).
- A cheap hybrid falsifier, grouped `SwiGLU` into packed `down+weighted_sum`,
  correctness-passed but failed hard end-to-end on rebuilt sequential A3B `pp320`
  (`775.67 -> 478.84 t/s`). Removing grouped `out` / weighted-sum passes is not
  worth giving up grouped-down locality.
- Interleaved gate/up fused-bank grouped-`SwiGLU` is exact and can look strong in
  isolated SwiGLU microprofiles, but a full-tail diagnostic now kills it as a
  general expert-bank ABI: A3B `chunk320/512/1024` is `1.264x/1.074x/1.008x`,
  and A10B `chunk320/512/1024` is only `1.022x/1.006x/1.008x` once grouped down
  and weighted sum are included.
- A runtime duplicate fused-bank proof is already killed as a production path: A3B
  correctness passed, but it cost `11.25 GiB` extra resident memory and converted
  only `775.25 -> 790.38 t/s` at `pp320` (`~1.02x`).
- A down-only lcpp-shaped final-store probe that spread grouped Q5 down stores
  across all simdgroups was exact but slower on A3B `chunk512` (`1.38 -> 1.57 ms`
  for `split_down_reduce`), so final-store vectorization alone is not the crack.
- An all-`n16` grouped Q5 down probe was also exact but slower on A3B `chunk512`
  (`1.37 -> 1.70 ms` for `split_down_reduce`), so the current `n32` down tile is
  not obviously oversized despite diffuse bucket counts.
- Splitting `down+reduce` shows weighted sum is tiny at the key `chunk512` gate:
  A3B `split_down=1.36 ms`, `split_reduce=0.07 ms`; A10B `split_down=3.38 ms`,
  `split_reduce=0.11 ms`. The actionable bucket is grouped Q5 down, not reduce.
  Fresh bucket histograms are not sharply cold-biased enough to explain the gap by
  tile waste alone: both A3B and A10B have `p50_count=28`, `ge32=44` experts at
  `chunk512`, and the all-`n16` down probe still lost.
- A scan-ledger versus atomic-ledger falsifier now kills bucket order as the
  obvious Q5-down locality crack. The atomic ledger introduced many expert-ID
  back edges (A3B `1451/3987`, A10B `1568/4002`) while preserving exact output,
  but grouped routed-tail time was flat to slightly faster (`0.994x` A3B,
  `1.043x` A10B). Do not spend a branch on route-ledger ordering unless counters
  show a new locality mechanism.
- llama.cpp's remaining A3B MoE advantage on this M4 Max is not a hidden Metal
  tensor-API win: `llama-bench` reports `has tensor = false`, and forcing
  `GGML_METAL_TENSOR_ENABLE=1` does not satisfy the device-family gate. The same
  A3B/A10B GGUFs also have separate `ffn_gate_exps` / `ffn_up_exps`, not a fused
  `ffn_gate_up_exps` tensor, so the relevant external target is non-tensor
  simdgroup `mul_mm_id`, not a tensor-API or fused-bank path.
- A fresh all-`n32` rerun keeps the subtle point straight: all-`n32` is much
  faster than all-`n16` in the isolated grouped-tail proof (`1.792x` A3B,
  `1.384x` A10B at `chunk512`), but forcing all-`n32` does not beat the current
  default hot-`n32` path end-to-end (A3B `pp512` flat, A3B `pp1024` slightly
  negative, warmed A10B `pp512` negative). The default hot gate already captures
  the high-count tile win.

Current design rule:

- The A3B matrix-attention default gate has passed. Resume routed-MoE work only
  after preserving the new matrix default in coverage gates, because it changes
  the denominator for future A3B routed-tail claims.
- Do not expect prompt chunk policy to close dense long-context prefill. It is now
  a narrow MoE production knob: promote only if a prompt-length-gated `2048` or
  arch-specific cap keeps A3B wins and A10B gains without dense changes or
  memory-pressure warnings. Do not blanket-default A3B to chunk `2048`; the repeat
  gate regressed `pp128/512`.
- Matrix scratch policy is now explicit: prompt-aware callers should allocate with
  the actual last position; auto mode falls back to packed attention if scratch is
  undersized; force-on keeps the hard error. Keep the looser matrix oracle envelope
  documented and rely on prefill-vs-single as the production correctness gate.
- Before adding another local grouped kernel variant, prove the remaining lcpp gap
  with matched per-layer/per-op attribution on the same GGUF, prompt length,
  chunk/batch shape, warmup policy, power state, and flash-attention setting.
- Add/keep a coverage gate that asserts expected routed-layer counts and flags any
  fallback by dtype/role. The A3B `37/40` miss is the failure mode to prevent.
- Do not pursue dataflow branches that sacrifice grouped-down locality unless they
  first show parity on rebuilt sequential `pp320`/`pp512` gates.
- Do not repeat local `grouped_swiglu` knob sweeps unless a new phase ladder shows
  a new mechanism. The next exact proof must reduce the combined
  `grouped_swiglu + grouped_down` routed-tail bucket.
- Do not chase bucket-order rewrites as the next grouped-down lever; scan versus
  atomic ordering is exact and essentially flat despite very different ID order.
- Do not defer the next branch waiting for llama.cpp Metal tensor-path parity on
  this hardware. The live llama.cpp path is the regular simdgroup `mul_mm_id`
  path for these files.
- Do not retread all-`n32` as a default path; use it only as a diagnostic for
  tile-count sensitivity unless a new distribution changes the cold-bucket trade.
- Demote offline/interleaved gate+up ABI to a narrow A3B `chunk320` branch. The
  next serious exact FFN branch should attack grouped Q5 down locality/dequant or
  a true `SwiGLU+down` fusion that preserves grouped-down locality and beats the
  full-tail fused-bank diagnostic, not the isolated SwiGLU microprofile.
- Keep dense GDN skinny E8xP32 scoped to prompt-prefill `beta_proj` / `alpha_proj`.
  The safety case depends on that narrow callsite, the sampled Qwen3.5/Qwen3.6
  shape audit, and the rollback env `QWEN_PREFILL_GDN_SKINNY_E8P32=0`. The
  discovery raises the EV of searching for other F32 skinny projection dispatcher
  mismatches, but do not generalize this helper without a new gate.
- Keep grouped routed zero-fill default-off as small cleanup, not a main roadmap
  lever.

Acceptance gates:

- Matrix-attention promotion is complete for A3B/group-8. Any future change to this
  path requires preserving default/rollback rows, a full A3B prefill-vs-single
  gate, and trace-label coverage showing `10/10` matrix attention layers plus
  `40/40` MoE fast-path labels.
- A MoE chunk-cap default change requires clean-build rows showing `pp512/1024`
  unchanged by construction, repeated A3B and A10B long-prompt wins at
  `pp4096/16384`, one real-rollout A3B row, and no `pmset` or memory-pressure
  confounds. Dense rows are guardrails, not expected beneficiaries.
- Dense G6, A10B/G16, non-F16 KV, or different attention shapes are not covered by
  the A3B default promotion; they need separate gates before any auto-on policy.
- For the next commit/default promotion in this lane, require trace-label or
  equivalent in-process coverage showing all expected MoE layers use the intended
  route, grouped routed, and shared packed paths for A3B and A10B.
- Before claiming llama parity, require paired qwen/llama rows plus attribution on
  identical fixtures. Throughput alone is not enough after the Q6 escape lesson.
- Any expert-bank ABI / interleaved layout branch must be exact on A3B and A10B
  small gates, improve end-to-end A3B `pp320` by at least `~1.12x` and A10B
  `pp320` by at least `~1.15x`, and be no worse than `-2%` at `pp512` with no
  material `pp1024` regression. After the full-tail diagnostic, it also must beat
  the live grouped routed tail by `>=1.15x` at A3B/A10B `chunk512`, not merely the
  isolated grouped-SwiGLU kernel.
- Any true fused `SwiGLU+down` branch must preserve grouped-down locality and beat
  the offline-bank path, not just the old baseline.
- Any structural routed-tail branch must show at least `>=1.15x` routed-tail
  speedup on both A3B and A10B at `pp512`, stay positive/neutral at `pp1024`, and
  improve A3B `pp4096` with matrix attention fixed by `>=1.10x` before default
  consideration.
- Dense GDN skinny follow-up is monitoring, not a pre-merge blocker: add one
  long-context A3B/state sanity row and a real prompt/top-k check when convenient,
  and audit any future GGUF with new F32 GDN alpha/beta shapes before assuming the
  default is risk-free.
- A3B matrix `pp4096` phase traces make `pre_norm` look large, but a llama-style
  float4 spelling of batched RMSNorm is correctness-green and end-to-end flat
  (`1408.70/1409.10` baseline vs `1408.64/1407.07` vec4). Do not pursue local
  RMSNorm vector spelling; revisit norm only as fused norm+projection or after a
  direct llama norm-node differential proves a real gap.

### 2. Historical: true-long A3B matrix-attention branch is now folded into #1

Optimizes: A3B `16k/32k+` prefill where qwen now falls from the medium-prompt
plateau faster than `llama.cpp`.

Why this section is retained:

- Same-shape sparse rows show `llama.cpp` also declines after the medium-prompt
  peak, but remains faster: post-Q6 qwen/lcpp is `0.96x` at `pp1024`, `0.95x` at
  `pp4096`, and `0.78x` at `pp34502`; `pp16384` needs a post-Q6 rerun.
- Real rollout shape is not the first-order cause: the post-Q6 real `v02_reva`
  `34.5k` row is `674.28 t/s`, still below the latest lcpp synthetic long anchor
  but far above the old `587.60 t/s` same-fixture row.
- No-op attribution at true-long shapes points at attention body: A3B `pp16384`
  goes `767.64 -> 1101.72 t/s` with attention body skipped, nearly matching
  lcpp full prefill (`1112.03 t/s`), while routed-MoE skip reaches only
  `913.73 t/s`.
- `llama-bench -fa 0` disables flash attention rather than selecting auto, and
  `-fa 1` is flat/slightly slower for A3B at the checked prompt lengths. The
  relevant lcpp target is therefore the non-flash Metal path, not
  `GGML_OP_FLASH_ATTN_EXT`.
- A deliberately matrix-shaped A3B sidecar (`V^T`, `KQ`, softmax, `KQV`) first
  regressed long prompts because it re-transposed the full V prefix in the body.
  After moving V_T writes into fused cache fill and vector-loading the F32 B tile,
  it wins real long rows: about `1042 t/s` at `pp4096`, `997 t/s` at `pp8192`,
  noisy `~864-904 t/s` at `pp16384`, and `725.7-736.9 t/s` at `pp34502`
  depending on chunk size.
- After the grouped Q6-down fix, the same sidecar now reaches `1434.39 t/s` at
  `pp4096`, `878.72 t/s` at synthetic `pp34502`, and `870.94 t/s` on real
  `v02_reva`, enough to move matrix productionization to the #1 active bet.
- After those matrix wins, local attention follow-ons did not convert: direct KQV
  final stores regressed, vectorized KQV temp copies regressed, and F16
  probability scratch was flat/slower while failing the current matrix oracle
  max-abs limit.

Current design rule:

- Do not use `pp320/512/1024` routed-FFN wins as evidence that true-long behavior
  is fixed; measure same-shape `16k/32k+` rows.
- Treat packed-attention body/main-pass work as the long-context branch. The
  explicit reduce pass was already measured tiny; the remaining attention wall is
  main-pass execution shape, KV reads, partial writes, and online-softmax work.
- Compare against `llama.cpp` at the same prompt length before claiming long
  scaling progress.
- Treat fused V_T scatter as the first lcpp-derived mechanism worth
  productionizing, but do not promote the current sidecar as-is. It still needs a
  non-manual max-pos allocation policy and a tighter KQV correctness story.
- Treat chunk `2048` as the current matrix-long candidate for `34.5k` rows;
  chunk `4096` loses despite the larger query batch.
- Next inspect/copy deeper lcpp `mul_mm_f16_f32` KQ/KQV tiling and score layout
  only after matrix default gates fail or plateau; do not spend the next sprint on
  more local KQV store/probability variants before the productionization gates.

Acceptance gates:

- A long-attention branch must improve A3B same-shape `pp16384` and `pp34502`, not
  just medium prompt rows.
- Target at least `>=1.20x` at `pp16384` or a clear path to preserving the new
  `>=1.0x` matrix-sidecar spot rows after production gating.
- Any matrix/non-flash branch must be positive at `pp4096` and `pp8192` before it
  gets long-run time at `16k+`; `pp512/1024` wins alone are not a promotion signal.
- Defaulting the fused V_T matrix path requires repeated cooled wins at
  `pp1024/2048/4096/8192/16384`, a clear max-pos scratch policy, and no dense or
  decode regression from extra V_T memory/writes.
- Any matrix chunk-size change must include memory accounting for score scratch and
  at least one true-long row; `pp1024` is no longer a sufficient chunk oracle.
- Keep A3B packed-attention oracle/correctness green at the first activated
  long-context chunk shape.

### 3. Hypothesis: remaining packed-attention variants need end-to-end systems evidence, not local kernel knobs

Optimizes: the remaining A3B/A10B prompt gap after prompt-native packed
attention, family-specific `NWG`, and `min_pos=512` are already defaulted.

Why it moves to the top:

- Prompt-native packed attention is no longer hypothetical; it is the default for
  the proven MoE prompt shapes and it materially moves `pp512`, `pp1024`, and the
  long-rollout lanes.
- The next obvious packed-kernel knobs already produced hard negative lessons:
  `QT=4` is exact but slower, and body-only `NWG=32` microbench wins were a false
  promotion signal until cooled end-to-end sweeps re-ranked the family defaults.
- The A3B matrix-attention sidecar is now a systems-level lesson rather than a
  warning only: high-level graph copying was insufficient, but moving V_T writes
  to cache-fill time converted the long rows. Further attention work needs this
  kind of dataflow evidence, not row-kernel knob sweeps.
- Even a more faithful one-layer attention-stack microbench is still not a safe
  promotion oracle for A10B. That points at multi-layer interactions, scratch /
  residency behavior, or queue/scheduling effects rather than another easy kernel
  retune.

Current design rule:

- Promotion decisions for A10B packed variants must come from cooled end-to-end
  sweeps with repeated baseline anchors, not from body-only or one-layer micros.
- Use the attach-mode tracing helper and the packed per-layer oracle only to
  explain or falsify a candidate, not to outrank the end-to-end board.
- Treat further packed-kernel knobs as hypothesis generators until a systems-level
  capture says what the next real bottleneck is.

Acceptance gates:

- Show a systems-level explanation for any new packed-attention variant that beats
  the current default on repeated cooled sweeps for A3B and/or A10B.
- Keep per-layer packed-vs-old oracle green on the first newly activated prompt
  regime and at the active long-context chunk shape.

### 4. Historical: A3B long-prompt prefill was decode-shaped attention in disguise

Status: superseded by the packed-attention default and the calibrated true-long
same-shape rows above. Keep this section as historical context for why the
prompt-native packed-attention branch became the center of gravity.

Optimized: the clearest structural prompt gap after the A3B group-8 long-context
subgroup fix.

Why it moves to the top:

- Same-fixture real-rollout ladders and matched-token synthetic ladders now agree:
  A3B collapses with long prompts while A10B is much flatter.
- Attention no-op dominates the slope; GDN and routed MoE are much smaller by
  comparison.
- The new `g8_t2` subgroup path removes a major local A3B attention bottleneck,
  but the remaining slope is still strongly attention-shaped and weakly sensitive
  to larger `prefill_chunk`, which points at the decode-shaped prompt attention
  algorithm itself.
- `llama.cpp` was prompt-native on Metal for this stage while `qwen-llm` still ran
  per-token decode attention inside prefill chunks.

Current design rule:

- Keep the new A3B group-8 subgroup path as an experimental / guarded prefill
  win, not a universal decode selector.
- The prompt-native packed-attention microproof landed and is now defaulted for
  the proven MoE shapes.
- The remaining true-long attention work is no longer “write packed attention”;
  it is main-pass/context-growth optimization inside the packed path.

Acceptance gates:

- Show a prompt-native packed-attention microproof at A3B shape (`group=8`,
  `head_dim=256`, F16 KV) that beats repeated decode-shaped attention by at least
  `~1.25x` at `16K` and `~1.35x` at `32K`, or projects to a meaningful end-to-end
  long-prompt win.
- Keep correctness against the existing path / naive reference on active long-
  context shapes.

### 2. Hypothesis: part of the remaining `pp320` gap is harness/phase mismatch

Optimizes: decision quality and scoreboard fidelity.

Why it moves to the top:

- We are near parity on the user-facing CLI harness but still far behind on
  `llama-bench pp320`.
- Before another major rewrite, we need to know how much of that miss is real
  prompt work and how much is benchmark semantics.
- This is the one measurement-heavy item that remains justified because it tests
  a direct causal hypothesis about the gap.
- The `MTL,BLAS` backend string `llama-bench` prints is a registration artifact,
  not a hot-path signal: at `pp320` on 27B Q4_K_M every mat-mat / mat-vec node
  runs on Metal and the BLAS backend executes zero ops. See the BLAS hot-path
  audit in `docs/PERF-LOG.md` for the per-node sched-debug evidence. The gap is
  Metal vs Metal, not "we are missing a CPU sgemm lane".

Current design rule:

- `qwen-bench pp` now covers the core harness need: synthetic `pp<N>`, no decode
  loop, optional tail skip, wall/GPU reporting, and lowering summaries.
- Keep using it as the scoreboard harness for dense and MoE prompt work; do not
  overfit to the older repeated-prompt decode harness.

Acceptance gates:

- Maintain `qwen-bench pp` while adding any future prompt phase buckets.
- Recompute the remaining prompt gap against `llama-bench pp320` after each major
  prompt-path change.

### 2. Hypothesis: if the `pp320` gap is real, decode-shaped prompt attention is the dominant remaining engine problem

Optimizes: the largest likely remaining structural prompt-only gap once harness
semantics are aligned.

Why it moves up:

- The cheap packed attention-body cleanup already landed, but if the pure prompt
  gap survives harness matching, GDN-tail cleanup alone cannot close it.
- Our prompt attention still retains decode-shaped structure in the inner loop.
- That makes prompt-native packed attention the strongest causal explanation for
  a large remaining `pp320` miss.

Current design rule:

- Keep the new packed RoPE + chunk-scatter shape as the base path.
- Only escalate after item 1 confirms the remaining miss is real and not mostly
  harness semantics.
- Prefer prompt-native packed attention / verify primitives over more small glue
  cleanups once the hypothesis survives.

Acceptance gates:

- End-to-end pure prompt throughput must move materially against the new harness.
- Correctness must stay green on `prefill_tokens_matches_single_token_loop_27b`.

### 3. Hypothesis: if prompt attention is not enough, the next real dense prompt miss is GDN out-proj / recurrence tail

Optimizes: the residual dense prompt GDN work after the packed prep rewrite and
attention-body cleanup.

Why it moves back to the top:

- The split ladder still says the remaining GDN tail is mostly out-proj plus a
  smaller recurrence cost.
- This is now a bounded fallback hypothesis, not the assumed main story.

Current design rule:

- Keep using the real-graph `QWEN_PREFILL_GDN_SPLIT` ladder for bounded probes.
- Prefer narrow out-proj / recurrence cleanups over another broad rewrite.

Acceptance gates:

- End-to-end dense packed prefill must move on 27B against the repeated prompt.
- Correctness must stay green on `prefill_tokens_matches_single_token_loop_27b`.

### 4. Dense Prompt: Attention Body Cleanup

Optimizes: any additional dense prompt throughput still available in the
full-attention layers.

Why it stays near the top:

- The packed consecutive RoPE + chunk-scatter cleanup was a real win, but the
  attention body still costs on the order of `~120 ms` wall on the repeated
  prompt.
- If the remaining GDN-tail work stalls, this remains the best smaller-bore
  alternate lane before a true packed prompt attention rewrite.

Current design rule:

- Keep the new packed RoPE + chunk-scatter shape as the base path.
- Only escalate to more invasive packed causal prefill attention if smaller
  cleanups stop moving the prompt.

Acceptance gates:

- End-to-end dense packed prefill must move on 27B against the repeated prompt.
- Correctness must stay green on `prefill_tokens_matches_single_token_loop_27b`.

### 5. Decode Command-Model Overlap

Optimizes: apples-to-apples dense decode latency, especially 27B at 4K and up.

Why it moves up:

- Real 27B 4K attach-mode Metal trace now exists.
- It shows `128` command buffers for `128` decode tokens, `128` compute encoders,
  encoder duration median `~0.699 ms`, and previous completion -> next submit
  median `~0.538 ms`.
- The direct decode-window profiler at 4K is the decisive result:
  `med_total ~42.69 ms`, `med_gpu ~42.14 ms`, `med_cpu_enc ~0.20 ms`, so decode
  is about `98.7%` GPU-busy on the 27B dense guardrail at 4K.
- The token loop is fully serialized today. The trace does NOT show a giant
  hidden bubble, but it does show a real low-single-digit command-model gap.
- Process-scoped compute intervals split into a small short-gap population and a
  large token-cadence population; the short intra-CB gaps total only about
  `~1.2 ms/token` at 4K, which keeps this as a real but bounded lever.

Immediate focus:

- Double-buffered / pipelined decode submission first.
- Use the new attach-mode trace helper and parser to validate any overlap claim.
- Only escalate to heavier encoder restructuring if post-overlap traces still
  show meaningful serialized slack.

Acceptance gates:

- Reduce completion -> next-submit gap and total 27B decode ms/token at 4K.
- Keep exact-token behavior and current correctness gates intact.

Status:

- A dense-only bench path now exists at `qwen-bench decode-window --pipelined`.
- Measured at 27B dense:
  - `ctx=4096`: about `~0.3%` over alternating repeats
  - `ctx=32768`: about `~0.3%`
- Keep it as an experimental harness, not a production checkpoint, unless a
  future shape/context shows a materially larger win.
- A second bench-only dense decode branch now exists at
  `qwen-bench ctx-sweep --concurrent-gdn-proj`.
- At 27B dense `ctx=4096`, `window=64`, it improves decode from
  `43.87 -> 42.14 ms/token` (`22.8 -> 23.7 t/s`), with the gain showing up in
  GPU time rather than CPU encode.
- At 27B dense `ctx=16384`, `window=64`, it also improves decode from
  `47.67 -> 46.51 ms/token` (`21.0 -> 21.5 t/s`).
- This is the first command-model branch that has cleared the “real enough to
  checkpoint” bar; the gain survives 4K and 16K, though it narrows somewhat as
  attention grows.
- Attention-only overlap is smaller:
  - `ctx=4096`: `43.89 -> 43.33 ms/token`
  - `ctx=16384`: effectively flat (`47.26 -> 47.22 ms/token`)
- Running both projection-overlap branches together is still positive and
  checkpoint-worthy:
  - `ctx=4096`: `43.64 -> 42.31 ms/token`
  - `ctx=16384`: `47.23 -> 46.14 ms/token`
- The combined branch is not additive with GDN-only overlap, but it remains the
  strongest decode-focused command-model variant measured so far.
- Treat encoder-boundary removal in these concurrent paths as an anti-bet until
  a fresh A/B proves otherwise: the split is buying GPU-side overlap between
  independent front projections, not merely adding host encode overhead. Host
  encode is already only ~0.20 ms of ~42.14 ms GPU time at 27B dense 4K, so the
  upper bound on collapsing encoders is well under 0.5%; the load-bearing piece
  is the middle `begin_concurrent` encoder for the front projections, and any
  merge must preserve that concurrent-dispatch property.
- MoE no longer shares the same decode-overlap uncertainty: concurrent GDN front
  projections are now production-wired for MoE decode on this repo, and the A3B /
  A10B `tg32/tg128` sweeps are already materially positive.

### 6. Read-Only Weight Residency And Scratch Storage Cleanup

Optimizes: decode and prompt wall via cheaper Metal bookkeeping and cleaner GPU
memory behavior.

Why it belongs near the top now:

- The command-model trace says there is not a giant host bubble, so the cheap
  structural wins become more attractive than speculative scheduler work.
- The external review's `hazardTrackingMode: untracked` + residency-set idea is
  orthogonal to no-copy GGUF views and should help regardless of mmap strategy.
- Scratch is still `StorageModeShared` everywhere today, which is convenient but
  not obviously ideal for GPU-only hot tensors.
- The new A10B `pp128` cold-run study keeps this bounded: touching or pinning MoE
  expert-bank buffers fixes a benchmark-hotness artifact, but steady-state
  `pp320/pp512` stay flat, so this is still structural cleanup rather than the
  highest-EV runtime lever.

Immediate target order:

1. Mark read-only weights as untracked and managed by a residency set.
2. Audit CPU readback/debug use, then move only proven GPU-only scratch arenas
   toward `StorageModePrivate`; avoid a blanket allocator swap.
3. Measure decode/prefill again before bundling this with larger graph changes.

Acceptance gates:

- Any change must preserve correctness and avoid regressing steady-state decode.
- Keep these as cheap structural cleanup unless traces show a larger-than-expected
  wall effect.
- Specifically: `--full-logits-decode` (exact-token oracle) and any tap-based
  hidden-state captures must stay green across a scratch storage-mode change,
  since a blanket `StorageModePrivate` swap silently breaks CPU readback paths.
  `MetalSession::fresh` currently allocates ~39 scratch tensors via
  `MetalTensor::zeros_f32` (`StorageModeShared`); the audit must classify each
  as GPU-only vs CPU-readable before any allocator change lands.

### 7. MoE Next: Execution-Model Reset After The Exact Grouped Plateau

Optimizes: the remaining MoE prompt scoreboard gap after the current exact grouped
backend has locally plateaued on this hardware / repo shape.

Current read:

- The grouped expert-major MoE prompt backend is now the stable base path again.
- Fused route+bucket and hot-th48 grouped Q4 are both kept because they compose
  positively at `chunk_p >= 512`.
- `n32-all` is still not a ship candidate: it improves grouped compute locally but
  does not clear the end-to-end pp gate.
- Router logits no longer look like the next missing kernel. The `E8xP32` kernel
  is already good enough to ship on the proven regime.
- The obvious exact local `grouped_swiglu` variant family is now well sampled and
  mostly exhausted here: tile, threshold, queue/locality, atomic-down, and
  resident-mirror branches all went flat or negative.
- The obvious route-ledger ordering hypothesis is also falsified: an exact atomic
  bucket ledger creates many expert-ID back edges but leaves A3B/A10B `chunk512`
  grouped routed-tail time essentially flat.
- Two more exact reads now narrow the field further:
  - grouped `inner/out` zero-fill is safe to skip and slightly positive, but too
    small to change the scoreboard by itself;
  - a re-based split routed FFN proof only reaches parity to slight loss versus
    the live grouped backend, so “just make it llama-like by separating gate/up”
    is not enough.
- The strongest near-term exact branch is now the guarded concurrent-tail path:
  overlap the live grouped routed tail with the live shared FFN at `chunk_p >= 512`.
  It converts to roughly `1-3%` end-to-end on `pp512`, but collapses quickly by
  `pp1024`, so it should be treated as a bounded production win, not the final
  structural answer.
- That leaves the next serious structural MoE-first branch as a **narrow** kernel
  proof, not a broad sidecar rewrite: a true `MUL_MAT_ID`-style / id-aware
  projection microproof that beats the current grouped projection or routed tail
  directly on both A3B and A10B.

Acceptance gates:

- Keep `QWEN_PREFILL_MOE_GROUPED=0` as the kill switch while rollout evidence is
  still expanding.
- Keep the route-only and hot-only flags forceable, but do not default either one
  globally outside the allowlisted combo regime.
- Default-on only for the proven `Q4_K/Q4_K/Q5_K` MoE packed-prefill envelope;
  fallback stays live for unsupported dtypes/shapes and for prompt chunks below
  the allowlist.
- Keep the current correctness matrix green: A3B single-token-loop + hidden
  capture, A10B smoke, awkward chunk-boundary A10B (`T=129`, `P=128`), and dense
  9B/27B guardrails.
- Treat the concurrent-tail branch as shape-gated until a full active-shape
  prefill oracle exists for the covered `pp512+` regime; do not assume the large
  block-local overlap effect composes into a large end-to-end win.
- Do not spend another major engineering branch on a new exact local
  `grouped_swiglu` variant unless a new diagnostic shows a new mechanism beyond
  the already-falsified tile / threshold / queue / mirror family.
- Before writing a large new id-aware kernel, require a smaller microproof to beat
  the current grouped projection / routed tail directly on both A3B and A10B, not
  just the obsolete packed denominator.
- Any next MoE-first branch must beat the current allowlisted combo on both A3B
  and A10B at `pp512` / `pp1024`, not just improve a routed microprofile.

### 8. Use 9B As The Fast Dense Long-Context Canary

Optimizes: experiment throughput and long-context turnaround while preserving the
27B guardrail.

Why it is now active:

- `group=4` attention v4 is now enabled, which unlocks the whole small dense line
  (0.8B / 2B / 4B / 9B) for realistic long-context decode.
- Local 9B now reaches 32K cleanly and is dramatically faster to iterate on than
  27B: `64.0 t/s` at 4K, `59.5 t/s` at 16K, `53.8 t/s` at 32K.

Usage rule:

- Use 9B for fast falsification of long-context attention / dense prompt ideas.
- Keep 27B in the analysis loop before claiming a real win.

Acceptance gates:

- Long-context experiments should be reproducible first on 9B, then confirmed on
  27B before the roadmap moves.

### 9. No-Copy GGUF Views And Residency Warmup

Optimizes: TTFT, cold-start variance, load-time memory pressure, and possible VM
object overhead.

Why it enters the roadmap now:

- The `ds4` close read makes this the strongest non-kernel structural crib.
- Current `qwen-llm` still copies weights tensor-by-tensor into fresh shared
  buffers; `ds4` instead wraps a few large GGUF-backed no-copy Metal views and
  warms residency up front.

Why it is not above the current prompt work:

- This is more likely a load / first-token / memory-cleanliness lever than the
  next steady-state prompt-throughput unlock.
- It still looks high-EV enough to prototype once the current MoE and dense
  prompt branches have a stable checkpoint.

Acceptance gates:

- Prototype shows materially better load time, first measured token stability, or
  memory / VM-object behavior without regressing steady-state throughput.

### 10. Frontier Benchmark Harness With Snapshot / Restore

Optimizes: benchmark quality and long-context decision speed.

Why it matters:

- `ds4-bench`'s frontier measurement style is a better mental model for prompt vs
  decode frontiers than one blended tokens/sec number.
- This would sharpen long-context dense/MoE comparisons and future speculative
  work without changing model semantics.

Acceptance gates:

- Add exact frontier prompt/decode probes that can restore from snapshots and
  measure a fixed local window.
- Use it to compare qwen vs llama phase-for-phase, not on blended totals.

### 11. Speculative Path: Attack Repeated Long-Context Attention Cost

Optimizes: DFlash / MTP viability at realistic context lengths.

Priority rule:

- Keep this behind the current dense/MoE prompt push.
- When returning to speculative work, do not lead with policy/schedule tuning;
  lead with kernel work that removes repeated long-context attention cost.
- v0.416 audit digest reopened KV compression ahead of packed-N verify attention;
  v0.437 closes the same-layout Q8_0 reader variant as negative. Future
  compressed-KV work must start from a materially lower-byte format or materially
  different attention/dequant body and a fresh `attn-intra` win. v0.542 shows
  that layout rearrangement alone is insufficient; do not keep packed verify
  blocked on Q8_0 specifically.
- v0.545 closes the bounded Mei-medium DFlash witness sequence before paid-cost
  attribution. Exact target-greedy output holds, but `alpha_chain=1.65625` and
  mean emitted per step is `2.65625`, below both preregistered authorization
  floors. P/A2 and every split-4 widening remain unrun; do not reopen the
  sequence through policy, prompt, floor, profile, or anchor rescue.

What the latest analysis says:

- Current MTP shape is structurally weak at long context because lazy verify does
  not amortize enough base work and MTP has its own growing KV attention cost.
- Current DFlash is made safe by adaptive verify `N`, but not fast, because the
  drafter and target still pay too much long-context attention work.
- v0.502 adds draft-free MTP probes: replaying current D3 draft tokens with zero
  MTP calls still loses on 27B (`0.967x` total), and a perfect greedy D3 oracle is
  only `1.201x`. This makes packed target-verify structure the current binding
  MTP loss; a single-CB drafter alone cannot explain or close the MTPLX gap.
- v0.503 finds the sharper constraint: physical verify N, not verify as a whole.
  Perfect-oracle D7/N8 and D15/N16 already reach `52.0` and `57.1 t/s`, while
  off-ladder N5/N9 are much weaker. Bucketed logical D over physical N8 turns the
  actual 27B D3 code prompt from `0.912x` to `1.031x` at the same alpha.
- v0.504 runs real recursive acceptance: D7/N8 wins on the 27B code prompt
  (`1.153x`, `~4.0 emitted/step`) and replay-current reaches `1.362x` with zero
  draft calls; D15/N16 loses (`0.716x`) because emitted/step is only `~4.1`.
- v0.505 shows single-command-buffer D7 drafting is only a thin win (`1.162x`),
  so command submission/readback is not the major remaining D7 draft tax.
- v0.506 ablates draft-side work: body-no-lm-head reaches `31.1 t/s` and
  bridge-only reaches `32.1 t/s` against normal single-CB `27.7 t/s`. Draft
  `lm_head+argmax` is the largest draft-side cost, recursive body is smaller,
  and bridges are closed; the global gap is now acceptance plus verify ceiling.
- v0.507 rank tracing shows D7/N8 terminal mismatches are often near misses:
  target is in draft top-2 for `50%`, top-4 for `67.9%`, top-8 for `85.7%`, and
  top-16 for `92.9%` of terminal mismatches on the 27B code prompt.
- v0.508 closes simple rerank: best margin-swap policy is only `4.031`
  emitted/step and oracle one-token top-16 terminal rescue is only `4.812`, short
  of the `>=5.2` gate.
- v0.509 kills `token_embd.weight` as a no-asset Q4 draft-head substitute on 27B:
  alpha drops to `0.000` and the row falls to `5.3 t/s`.
- v0.510 splits MTP prompt prefill from decode-loop timing and defaults the bench
  to MTPLX-style post-norm base/recursive hidden feeds. On the 27B code prompt,
  D7/N8 legacy pre/pre is `30.2 t/s` decode-only at `3.879` emitted/step, while
  post/post is `33.0 t/s` at `4.267` emitted/step; on a narrative prompt it moves
  `26.8 -> 30.7 t/s` and `3.459 -> 4.000` emitted/step. D7/N8 oracle is
  `74.7 t/s` decode-only, so target verify can already explain the MTPLX
  `65-80 t/s` screenshot band when acceptance is ideal and prefill is excluded.
- v0.511 tests the largest MTPLX history-policy semantic delta. `--mtp-history
  cycle` resets MTP KV every speculative step and keeps only within-chain draft KV,
  but it is a hard kill: code prompt emitted/step falls `4.267 -> 2.783` and
  narrative falls `4.000 -> 2.415`, both equivalence PASS. Committed MTP history
  remains the default.
- v0.511 also fixes the rank simulator for fixed-token benches. Post/post rank
  traces now show oracle one-terminal-rescue top16 would finish 128 tokens in
  26 steps on both prompts (`4.923` tokens/step), still below the `>=5.2` gate;
  margin-swap policies remain flat.
- v0.512 adds A3B MoE MTP asset coverage. Unsloth's
  `Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf` has a real MoE MTP head at `blk.40`, but qwen
  only recognizes it and reports unsupported execution. llama.cpp `b9833` runs
  the same file at `tg128`: `81.71 t/s` with `n_depth=0` and `82.12 t/s` with
  `n_depth=7`. This is now an explicit unsupported-model gap; the thin lcpp
  synthetic uplift argues for correctness/acceptance probing before Q2/Q3 MoE
  MTP kernel work.
- v0.513 replaces the MoE-MTP unsupported bail with a correctness-first executable
  path. The single MTP MoE expert bank is dequantized to F32, A3B UD-Q4_K_S lazy
  MTP-1 passes equivalence, and base MoE decode gains native `Q4_K` routed-down
  coverage. This is not a performance conclusion: D3/N4 still fails because the
  packed-N verifier does not implement qwen35moe base forward, and IQ2/Q2/Q3 base
  MoE still needs native expert-bank kernels rather than F32 residency expansion.
- v0.514 adds a correctness-first qwen35moe packed-N verifier branch. D3/N4 now
  executes and passes equivalence on A3B UD-Q4_K_S, but the verifier is
  row-sequential for MoE blocks and is therefore an acceptance/coverage tool, not
  the final throughput shape.
- v0.515 measures the intended A3B Q4_K_M MTP artifact. Same-file qwen no-spec
  `tg128` is green (`94.97 t/s` vs llama.cpp b9833 D0 `77.78` / D7 `79.02`).
  Native D7/N8 has the acceptance signal the Q4_K_S smoke lacked: `5.333`
  emitted/step, alpha `0.619`, equivalence PASS. But total remains a loss
  (`0.815x`) because cost dominates. Replay-current is now unblocked and reaches
  `0.962x` with `mtp_calls=0`, while a perfect oracle reaches `1.213x`.
  Therefore the A3B branch is live, but the current cost split is verifier
  row-sequential work plus full draft `lm_head+argmax`, not acceptance alone.
- v0.515 also kills the naive first batched-MoE-verifier idea. Reusing prefill
  grouped routed/shared FFN at physical N8 is correctness-safe but slower:
  `QWEN_MTP_MOE_VERIFY_GROUPED_FFN=1` replay-current regresses `337.2 ->
  377.3 ms`. The issue is execution shape, not merely missing a grouped FFN call;
  prefill grouped kernels were built for much larger prompt buckets.
- v0.516 kills the cheap low-bit draft-head copy. Setup-time GGML Q4_1 and Q4_0
  copies of `output.weight` preserve equivalence and A3B D7/N8 acceptance, but
  remain flat/slower than default: Q4_1 totals `399.7 ms`, Q4_0 totals
  `397.0 ms`, versus the v0.515 default `397.8 ms` on the same 16-token row.
  This does not kill MTPLX's actual 4-bit affine/group-size-64 draft-head layout
  or a top-k/head policy, but it closes legacy GGML output clones as the cheap
  missing lever.
- v0.517 adds explicit MTP decode phase buckets. A3B Q4_K_M D7/N8 default spends
  draft `51.7 ms`, verifier `242.8 ms`, restore `1.0 ms`, and bridge `4.9 ms`;
  replay-current spends verifier `239.2 ms`, restore `0.9 ms`, and bridge
  `0.5 ms`. Therefore the replay-current loss is verifier structure, not hidden
  MTP bridge/rollback overhead, and the normal-path draft tax is secondary until
  verifier cost moves.
- v0.518 lands the first N8-native verifier win and defaults it with rollback
  `QWEN_MTP_MOE_VERIFY_BATCHED_MIXER=0`: MoE packed verify now batches the mixer
  side across physical N8 and leaves only the MoE FFN tail row-sequential. A3B
  D7/N8 16-token total improves to `0.910x` from `0.805x`, replay-current becomes
  a win (`1.112x`), and the 64-token row improves from `0.864x` to `0.953x` while
  preserving equivalence. The remaining verifier work is now mostly the MoE FFN
  tail, not GDN/attention mixer projections.
- v0.519 defaults `QWEN_MTP_MOE_VERIFY_CONCURRENT_FFN=1` for Q4_K/Q4_K MoE
  verifier gate/up banks, with `=0` rollback. This reuses normal decode's
  routed/shared FFN wave split inside the N8 verifier. A3B D7/N8 64-token normal
  MTP now wins (`1.035x`) and replay-current reaches `1.262x`; the 16-token row
  is near parity (`0.980x`) but remains fixed-cost/draft-cost limited.
- v0.520 validates that the win survives a 128-token row: A3B D7/N8 normal MTP
  reaches `1.052x`, replay-current `1.294x`, and perfect oracle `1.364x`.
  It also kills accepted-draft KV history as a bridge shortcut: bridge drops, but
  alpha falls `0.873 -> 0.688` and the 64-token row regresses to `0.890x`.
  Canonical accepted-KV repair is worth its cost for this asset.
- v0.521 narrows draft-head work. An MTPLX-style affine Q4 group-size-64 draft
  head preserves A3B D7/N8 alpha but is slower than the default Q6_K head
  (`draft 62.4 ms` vs `51.1 ms` at 16 tokens). A worktree-only exact fused Q6_K
  `lm_head+argmax` probe is also slower (`87.2 ms`) and was not kept.
  Body-no-lm-head replay confirms the head is real cost (`9.2 ms` draft body),
  but cheaper-looking kernels are not enough; future draft-head work needs
  row-throughput evidence before integration.
- v0.522 adds count-only packed-verifier attribution and defaults a small exact
  MoE row-view cleanup. The old A3B D7/N8 row-staging verifier spends, per verify
  step, MoE FFN row loop `3200` dispatches, GDN tail+checkpoint `2160`, attention
  body `400`, and tail lm_head+argmax only `3`. Row views remove the per-row
  `x/h` staging copies plus final scatter while keeping mature per-token FFN
  kernels; MoE row-loop dispatches drop `3200 -> 2240`/step and 64-token verifier
  time drops `520.7 -> 508.6 ms`. This is worth defaulting but too small to be the
  branch answer. The prefill grouped-FFN transplant remains killed at N8: forced
  grouped FFN cuts dispatches to `517`/step but regresses verifier `186.2 ->
  248.6 ms` on the traced 16-token row.
- v0.523 defaults GDN pair-L2 in decode and the MTP verifier with
  `QWEN_DECODE_GDN_PAIR_L2=0` as rollback. It is small but positive on repeated
  primary rows: A3B `tg128` runs=3 moves `107.21 -> 107.64 t/s`, dense 27B moves
  `25.37 -> 25.53 t/s`, and A3B D7/N8 MTP remains equivalent. A larger-looking
  MTP verifier batched-alpha/beta probe is killed: it cuts dispatches but slows
  verifier wall (`179.1/186.2 ms` versus `174.8 ms` row-view default). The lesson
  is to prioritize GDN body/checkpoint dataflow over skinny N8 projection batching.
- v0.524 defaults N8 MoE verifier batched routing with
  `QWEN_MTP_MOE_VERIFY_BATCHED_ROUTE=0` as rollback. It batches route logits,
  top-k, and shared gate across physical N8, then feeds row views into the existing
  per-token FFN waves. This keeps the prefill grouped-FFN transplant killed while
  removing regular route work from the row loop: dispatches move from `2240` FFN
  row-loop/step to `1600 + 80` route-pack/step, and A3B D7/N8 64-token verifier
  improves `505.6 -> 495.1 ms` with equivalence PASS.
- v0.525 kills the two most direct GDN checkpoint dataflow exits. Inline
  checkpoint writes from the conv/step kernels pass equivalence but regress the
  A3B D7/N8 16-token verifier `166.0 -> 200.2 ms`. Sparse N8 checkpoints at
  slots `0/3/5/7` plus exact suffix replay save only `4.7 ms` of verifier work
  and lose it back in restore/replay (`1.0 -> 11.4 ms`, total `1.004x ->
  0.994x`). Do not continue checkpoint-write fusion or suffix-replay variants
  without a cheap no-tail replay design; checkpoint blits are no longer the top
  active MTP verifier lever.
- v0.526 kills the batched shared-expert verifier probe. Keeping
  routed FFN per-token but computing the shared branch as N8 mat-mats preserves
  equivalence yet regresses the A3B D7/N8 16-token verifier `166.0 -> 196.9 ms`
  and total to `0.927x`. This closes batched shared as the cheap MoE FFN exit;
  any remaining row-wave branch must keep both routed and shared per-token
  kernels and change only scratch independence / scheduling.

Highest-EV speculative kernel targets:

1. Continue N8-specific MoE verifier work only when it preserves the fast
   per-token decode kernels. Do not reuse prompt-prefill grouped MoE kernels
   blindly: v0.515 and v0.522 both kill that direct transplant at N8. v0.518
   proves the branch by batching the mixer side, v0.519 reuses decode's FFN wave
   split, v0.522 removes staging copies/scatter, v0.524 batches route work, and
   v0.526 kills N8 shared mat-mat. Further MoE work
   needs to change only the FFN wave schedule/scratch shape with a `>=15 ms`
   verifier gate. The remaining concrete shape is row-pair FFN waves with
   disjoint per-row scratch while keeping both routed and shared per-token
   kernels; do not retry grouped prefill or batched shared kernels.
2. Revisit GDN verifier tail only for body fusion, not checkpoint writes. v0.523
   kills skinny alpha/beta batching and v0.525 kills both inline checkpoint
   writes and sparse checkpoint suffix replay. A future GDN branch should start
   with a one-layer `L2-in-step` or `step+rmsnorm_gated` body micro-oracle and
   require `>=20%` tail-body win before full verifier integration.
3. Reduce D7/N8 MTP draft-head cost only through a proven execution shape. v0.515
   A3B and v0.506 27B agree that `lm_head+argmax` is the largest draft-side tax,
   but v0.516 kills legacy GGML Q4_1/Q4_0 output copies, and v0.521 kills the
   first affine Q4 gs64 and exact fused-Q6 top-1 probes. Keep only candidates
   that first beat default Q6_K row throughput in isolation, such as a mature
   Q4_K-style draft layout or an exact MTPLX kernel/layout reproduction. Gate on
   `>=20%` draft-phase reduction at both 16-token and 128-token rows with alpha
   loss `<0.02`.
4. Keep A3B Q4_K_M D7/N8 as the MoE MTP acceptance/cost gate. It now clears the
   emitted/step investigation threshold, so use it alongside 27B code/narrative
   rows for MTP changes rather than relying on dense-only evidence.
5. Add native IQ2_S MoE expert-bank gate/up support, then evaluate Q2_K/Q3_K. Do
   not dequantize base MoE expert banks to F32. The one-block MTP F32 bridge is
   acceptable; base low-bit MoE needs native routed bank kernels for residency.
6. Continue dense-27B MTP semantic parity before tree work, but with cycle history closed.
   Remaining concrete variants are position-offset semantics, history-window
   variants rather than full reset, and MTPLX contract/draft-asset details. Gate
   continuation on emitted/step `>=5.2` or `>=25%` over the post-norm committed
   default, equivalence PASS.
7. Inspect MTPLX acceptance/reporting enough to separate decode-only vs total,
   D3 vs D7, and proper draft-head asset effects. The current qwen oracle already
   reaches the screenshot band decode-only; the open question is how MTPLX gets
   much closer to oracle in actual chain mode.
8. Build an alternate-continuation/tree simulator before any tree implementation,
   but only after post-norm rank traces still show unreachable chain acceptance.
   Gates: N8 `>=5.2` emitted/step to investigate, `>=5.8` to implement; N16 `>=9`
   interesting, `>=11` viable. Require stability across prompts.
9. Keep physical N8 as the first native MTP packet shape. Use padding/rollback for
   shallower adaptive depths; do not pursue D15/N16 until a better asset/policy
   proves much higher emitted/step.
10. Keep target-verify work tied to replay-current gates. A3B now clears the
    emitted/step threshold, but the direct prefill-grouped FFN transplant regressed;
    reopen packed GDN/tape, packed multi-query attention, or an N8-specific MoE
    kernel only with a measured replay-current win. The next useful attribution is
    inside the `verify_ms` bucket, not bridge/restore/draft bookkeeping.
11. Build bucket policy around supported packet shapes; prefer padding/rollback to
    arbitrary ragged N unless bucketed verification fails a correctness or waste
    gate.
12. Do not skip accepted-KV repair by keeping approximate draft-chain KV. v0.520
    kills that shortcut: it saves bridge time but harms acceptance enough to lose.
13. DFlash two-range attention reading ctx-cache and noise directly, without
    `k_full` / `v_full` materialization.
14. Compressed-KV only if a non-Q8_0 layout/body first beats tuned F16 in
    `attn-intra` at both 8K and 32K.
15. Retile the `N=16` mat-mat specializations only if verify/draft phase profiles
    show N16 mat-mat-heavy surfaces remain material after the attention fixes.

### 12. Mid-Graph Flush / Overlap Before ICB / MTL4

Optimizes: decode and prompt wall only if later traces show more cadence slack at
other contexts or shapes.

Why it stays behind the others:

- The new 27B 4K trace shows a real but modest command-model gap, not a giant one.
- Cheaper overlap and residency work comes before heavier command-graph surgery.

Acceptance gates:

- Only pursue after double-buffered decode and structural cleanup are measured.
- Require trace evidence of additional idle gap before escalating further.

### 13. Compressed KV Cache - Exact Q8_0 And Direct Q4_0 Closed

Optimizes: long-context decode, DFlash usefulness at long context, memory.

Current read:

- The first dense KV-Q8 prototype is a negative result on M4: despite exact
  append quantization and good output similarity, the current Q8 v4 main path
  makes attention slower than the tuned F16 path.
- Codex-wrap review says the likely cause is structural: scalar Q8 dequant/load
  overhead is overpowering stored-byte savings against an already-strong F16
  vectorized path.
- v0.416 audit + cx review reopens KV-Q8 only as a narrow reader micro-oracle,
  not as a broad retry of the falsified implementation. The reason to keep it
  alive is cross-cutting: it can reduce no-spec long-context attention bytes,
  lower KV memory pressure, and raise the context length where DFlash/MTP verify
  remains viable.
- v0.437 closes that micro-oracle for same-layout Q8_0. A vector-shaped Q8x4
  reader is correctness-clean but regresses A3B `attn-intra` at both 8K and 32K.
- v0.542 closes the allowed different-layout Q8_0 branch. A direct payload/scale
  split-plane reader is exact versus current Q8 and correctness-clean versus F16,
  but its valid 32K row regresses `9.95%` main and `6.75%` main+reduce against
  the slower F16 anchor. The 8K row is protocol-invalid for anchor instability
  and only supplies a uniformly negative directional signal. No experiment code
  remains.
- v0.543 tests the next allowed byte point, direct canonical Q4_0 at 144 bytes
  per head-row. Canonical standalone reconstruction and both scale signs pass,
  but real A3B block-3 attention fails the first 8K fidelity gate at cosine
  `0.996111664` and maximum absolute error `0.1653642654`. The 32K correctness
  row and all performance samples are unrun. No experiment code remains.

Expected payoff: still potentially large for compressed KV in theory, but exact
Q8_0 at 272 bytes per head-row is closed in the two tested reader-layout
families, and direct canonical Q4_0 fails real-model fidelity before timing.
Q6/FP8-like and other formats are untested, not disproved. Do not spend more
blind sweep time on Q8_0 rearrangements or alternate four-bit readers.

Risks and constraints:

- Easy time sink.
- Needs a materially lower-byte compression format, hardware-native conversion,
  or a different attention/dequant body to be worth revisiting. Another Q8_0
  layout rearrangement is not sufficient.

Acceptance gates:

- Revisit only with a concrete lower-byte or different-body structure and a fast
  `attn-intra` feedback plan that preserves Q-head grid parallelism and avoids
  large staged-KV TGM.
- Require a preregistered real-model fidelity oracle before any performance
  packet; v0.543 shows that byte savings alone can fail before timing.
- Cut quickly if the reader does not beat tuned F16 at both 8K and 32K before
  end-to-end wiring.
- Promote only after a no-spec long-context row improves and DFlash/verify phase
  attribution shows the new reader moves the adaptive context threshold.

### 14. Dense Decode Surgery And Small Decode Hygiene

Optimizes: dense decode throughput and measurement integrity.

Why it stays late:

- v0.340 already banks the obvious dense GDN front-projection scheduling overlap
  as a default, with `QWEN_DECODE_DENSE_CONCURRENT_GDN=0` rollback.
- Dense decode is already competitive enough that prompt work dominates the
  scoreboard.
- Prior FFN mega-fusion had weak payoff and GDN recurrence semantics are
  correctness-sensitive.
- GPU argmax is already landed; dense gain is neutral within noise and MoE gain
  is modest but real.
- v0.646 now prices the small exact decode leaves directly. F32 beta fusion is
  about `+0.255%` wall / `+0.276%` GPU on dense 0.8B, while paired RoPE is only
  a small wall signal with a zero-crossing GPU interval. The 27B dense guardrail
  shows no material beta, RoPE, or raw-Q promotion signal. These are cleanup
  candidates, not replacements for a new GDN work unit.
- The raw-Q/RMSNorm cancellation is closed and deleted after review across 0.8B,
  A3B, and 27B. The exact output-scale epsilon fold remains correctness-aligned
  but has no rollback-capable performance record. Reopen raw-Q only with the
  K-fold that removes the preparation work; keep both below the strategic queue.
- The grouped MoE finalizer is a separate decode-only bounded cleanup: corrected
  census topology is `851 -> 814` dispatches/token and balanced A3B timing is
  `+0.674%` wall / `+0.636%` GPU. It does not reopen the prompt grouped-MoE
  branch, and it still needs production equivalence traces before promotion.

Acceptance gates:

- Any decode surgery must be driven by a fresh dense phase profile identifying a
  specific waste pocket.
- First candidate if the profile supports it: GDN decode recurrence tiling across
  multiple `dv` rows to reduce launch/TG overhead and duplicated Q/K loads, but
  only if `gdn_step_decay` is materially above its unavoidable state R/W floor.
  The reducible component is the redundant `q_h`/`k_h` device-pointer reload
  across the `head_dim × n_v_heads` TG grid (currently ~384× per head per token
  for `head_dim=128, n_v/n_k=3`); the state R/W itself (~6 MiB/layer/token,
  ~288 MiB/token across 48 GDN layers on 27B) is irreducible without restructuring
  the recurrence. Budget the win at ~2-4% decode if k/q-bound; 0% if state-R/W-bound.
- Exact-token A/B paths (`--full-logits-decode`) and argmax regression tests stay
  green while decode work proceeds.

## Recent Confirmed Win — Qwen B2 File Root

- One chunk-aligned checkpoint now spans all pair-participating requests in a
  regular Qwen B2 JSONL file. A realistic four-request fixture moves
  `2.59 -> 2.09/2.10 s` with byte-identical output by reducing pair-local root
  evaluation from 12,705 to 417 tokens.
- Keep this as a bounded execution optimization, not a second cache index. The
  next cache/scheduler synthesis should share checkpoint discovery across B2,
  fixed cohorts, and serial execution rather than growing B2-local policy.
- Evidence: `docs/bench/2026-08-11-qwen-b2-file-root-fanout/README.md`.

## Deprioritized For Now

- FFN mega-fusion as a first move: prior layer-major fusion produced too little
  gain for the complexity.
- Giant GDN recurrence rewrite: correctness risk is too high without a sharper
  measured target.
- Synthetic-only attention tuning: attention is much improved; further tuning
  should be driven by full phase/ctx sweeps.
- Vocab pruning or approximate lm_head shortcuts without exact-token gates.

## How To Update This File

When a session changes performance direction, update only the smallest relevant
section:

1. Add new measured baseline rows or replace stale ones.
2. Move ranked bets only when measurements change expected value or risk.
3. Record accepted wins in "Recent confirmed wins".
4. Record failed experiments in "Deprioritized" or in the relevant bet's risks.
5. Keep benchmark notes sequential and reproducible; do not mix parallel runs.
6. At each improved checkpoint, make the diff tell one optimization story:
   short `v0.xx:` subject, detailed wrapped body with measurement + validation.
7. When a win changes the broader lowering or bottleneck picture, update
   `docs/INFERENCE-GRAPH.md` alongside this file so the semantic map stays in
   sync with the engine and the current performance story.

Useful pattern for future entries:

```text
Decision: <what changed>
Evidence: <bench command + key numbers>
Impact: <models / contexts affected>
Risk: <remaining validation gap>
Next: <one concrete follow-up>
```


## Recent Promotion — Charged Dense Ragged Cohorts

- Dense B=8 and Qwen MoE B=16 can execute independent prompt frontiers exactly,
  but only dense clears the automatic product gate against the measured B2
  incumbent inside the promoted envelope. At 64/32 output tokens, 0.8B execution
  improves `1.403x/1.367x` and 27B
  improves `1.991x/1.784x`.
- Automatic dense admission is intentionally narrow: prompt cap 256, equal
  generation limits of at least 32 tokens, one-chunk prefill, and at most two
  prompt tokens charged per productive transition in every
  proposed cohort, more full cohorts than incumbent planning, no serial file
  remainder, and unchanged utilization/memory gates.
  `QWEN_FIXED_COHORT_RAGGED_PROMPTS=0` is rollback.
- Automatic MoE ragged selection is KILLED at the current executor. At 128
  requested tokens, counterbalanced A3B B16 reaches only `1.090x`; decode alone
  measures `1.0998x` before private prefill is charged. Explicit `=1` retains
  the exact mechanism. Reopen only after B16 decode improves relative to B2,
  replacement prefill overlaps active decode, or a changed executor has an
  independently measured whole-request ceiling above `1.10x`. Evidence:
  `docs/bench/2026-08-12-qwen-moe-ragged-128-screen/README.md`.


## Recent Confirmed Capability — Bounded Dense Refill

- Dense B8 can replace finished lanes across one bounded second wave while
  preserving exact output. On a 32-request skew trace, wall moves `2.29 ->
  1.83 s` versus B2 and `2.27 -> 1.83 s` versus static ragged B8.
- Explicit dense B8 now defaults only to short serial-tail rescue. Independent
  11-token cells move 0.8B B2 `2.070 -> 1.685 s` (`1.228x`) and 27B B2
  `23.79 -> 16.04 s` (`1.483x`). A 64-token boundary remains `1.232x/1.174x`
  versus static B8 on the same anchors.
- Do not generalize synchronous refill to MoE: the charged A3B ceiling loses to
  measured B2. Reopen MoE only with overlapping/chunked replacement prefill or
  materially different transition economics. Evidence:
  `docs/bench/2026-08-12-qwen-refill-charged-screen/README.md`.


## Recent Confirmed Capability — Fixed-Cohort File Roots

- Dense B8 can retain one Qwen root checkpoint across all realized static
  cohorts. Four realistic cohorts move `16.01 -> 13.81 s` (`1.159x`) with
  byte-identical output; the same mechanism at only two cohorts is `1.093x`.
- Root selection excludes serial fallback and refill work, admission prices one
  retained root plus the largest deeper cohort checkpoint, and memory denial
  retains cohort-local behavior.
- Keep dense and Qwen MoE default-on with rollback: four A3B Q4 B16 cohorts move
  `14.58 -> 12.20 s` (`1.195x`) with byte-identical output. Treat prefix-aware
  refill as a separate next experiment rather than silently broadening this
  capability. Evidence:
  `docs/bench/2026-08-12-qwen-fixed-file-root/README.md`.


## Held — Root-Aware Dense Refill

- A realistic distinct-task trace gives B2+root `7.27 s` and static B8+root
  `7.42 s`. Perfect refill scheduling can remove only 14 measured B8 steps, for
  an optimistic `7.165 s` endpoint (`1.015x`) before replacement overhead.
- This is a transition-only projection, not a formal ceiling; refill could also
  remove some setup work. Keep the composition held until profiles expose enough
  non-transition duplication or B8/B2 economics materially change. Evidence:
  `docs/bench/2026-08-12-qwen-root-aware-refill-screen/README.md`.

## Recent Confirmed Capability — DeepSeek Pair-Affinity Scheduling

- The bounded B2 planner now defaults on for DeepSeek V4. An interleaved K160
  trace moves `114.71 -> 71.57 s` (`1.603x`) by pairing two 6,268-token affinity
  groups and selecting a 6,144-token causal checkpoint for each pair.
- Evaluated prompt tokens fall from 25,080 to 12,792 while complete input-ordered
  JSONL remains byte-identical. Decode wall is unchanged; the gain is exact
  prefix organization rather than concurrent-kernel movement.
- Keep the 16-request window, process-memory admission, input-order publication,
  and `QWEN_CONCURRENCY_PAIR_PLANNER=0` rollback. Evidence:
  `docs/bench/2026-08-12-deepseek-pair-affinity/README.md`.

## Recent Promotion — Short Dense B8 Serial-Tail Rescue

- A counterbalanced 0.8B cell isolates refill from ragged prompts and prefix
  reuse. B2 takes `2.08/2.06 s`; refill takes `1.69/1.68 s`, a median `1.228x`
  gain. The same trace on 27B moves B2 `23.79 -> 16.04 s` (`1.483x`).
- The exact 10% fully batched threshold reaches only `1.034x`; 1,029-token
  serial-tail rescue reaches only `1.075x`. Default only when static planning
  would leave serial tails and every prompt is at most 64 tokens; that boundary
  clears `1.232x/1.174x` on 0.8B/27B.
- `QWEN_DENSE_BATCH8_REFILL=1` keeps broader refill explicit; `=0` is rollback.
  Qwen MoE and implicit ragged refill composition remain closed; automatic dense
  ragged cohort selection is governed by the separate charged policy. Evidence:
  `docs/bench/2026-08-12-dense-refill-default/README.md`.
## Recent Confirmed Capability — DeepSeek File-Scoped Roots

- One immutable DeepSeek causal snapshot can now span multiple independently
  scheduled B2 pairs when every pair selects the same stable boundary. A
  four-pair K160 trace moves `125.99 -> 67.17 s` (`1.876x`) with byte-identical
  output and flat decode organization.
- Total model prompt evaluation falls from 25,544 to 7,112 tokens. Admission
  separates 8.78 GB of two-session Metal state from 66.3 MB of CPU root state;
  denial and `QWEN_CONCURRENCY_FILE_ROOT_FANOUT=0` retain pair-local behavior.
- Keep V1 exact and shallow. Reopen a longest-common-root plus pair-bridge
  hierarchy only when real files lose meaningful reuse to differing pair-local
  maxima; do not add it for symmetry with Qwen. Evidence:
  `docs/bench/2026-08-12-deepseek-file-root/README.md`.
## Recent Promotion — Automatic Dense Serial-Tail Rescue

- Automatic dense selection now uses the same bounded B8 refill policy during
  lookahead and execution. Same-binary wall moves `2.16 -> 1.76 s` (`1.227x`)
  on 0.8B and `19.76 -> 15.89 s` (`1.244x`) on 27B with byte-identical output.
- Keep the promotion narrow: short equal-frontier work with static serial tails,
  two waves, existing utilization/step gates, and prompt cap 64. Fully batched,
  long-prompt, broad forced, and MoE refill remain outside this refill policy;
  charged dense ragged admission is a separate automatic decision.
- Selector admission prices refill shared capacity before choosing B8; `=0`
  remains strict rollback. Evidence:
  `docs/bench/2026-08-12-automatic-dense-refill/README.md`.

## Recent Promotion — Automatic Dense Ragged Selection

- Automatic execution now composes the existing ragged-frontier executor with a
  per-cohort prompt/decode charge. Final same-binary 0.8B wall moves
  `2.06 -> 1.53 s` (`1.346x`); execution-only cells clear `1.367x-1.991x`
  across 0.8B and 27B.
- Keep this a dense capability, not a model-name fork: the same B8 executor and
  selector gates serve both measured dense scales. Equal-frontier work retains
  its incumbent plan, and envelope, charge, utilization, cohort-gain, or memory
  failure preserves incumbent planning before mutable execution; selector memory
  denial narrows to B2 and later cohort-memory denial remains serial fallback.
- A3B B16 remains explicit. A 128-token counterbalanced screen reaches only
  `1.090x`, and decode alone is `1.0998x`; do not spend another length sweep on
  the current executor. Reopen only for faster B16 decode, overlapped replacement
  prefill, or a changed executor with a measured whole-request ceiling above
  `1.10x`. Evidence:
  `docs/bench/2026-08-12-qwen-moe-ragged-128-screen/README.md`.

## Recent Promotion — Automatic Dense Ragged Refill

- Automatic dense B8 now combines heterogeneous prompt frontiers with bounded
  two-wave refill for mixed generation limits. Final-source median wall improves
  `1.205x` on 0.8B and `1.445x` on 27B, with byte-identical output and zero
  swaps.
- Keep admission measured and file-scoped: at least two complete refill arenas,
  prompt cap 256, generation cap 40, one-chunk prefill, no static/serial
  remainder, local 90% utilization and 15% idealized-step savings, local 32x
  charge, and whole-file 9x charge.
- Runtime memory denial restores the complete planner baseline atomically.
  Selector v3 and planner v9 expose policy, envelope, planned/realized work, the
  physically denied arena count, and transaction outcome. Both ragged and refill
  rollback variables independently close the path.
- This productizes an existing exact executor before a generic scheduler. Keep
  synchronous MoE refill killed and root-aware refill held; the next scheduler
  investment should target more than two waves or overlapped replacement prefill,
  not reproduce this bounded slice. Evidence:
  `docs/bench/2026-08-12-automatic-dense-ragged-refill/README.md`.
