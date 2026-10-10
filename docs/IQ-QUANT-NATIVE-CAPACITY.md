# Native IQ quant capacity queue

Started 2026-10-08 after the GLM MLA tail screen (its experimental selection
remains off). Native capacity coverage is complete for the qualified Saluki
artifact; measured dense IQ2_XS prefill acceleration is promoted, and the DeepSeek
family-scope regression is fixed and revalidated. This note
records the implementation decisions and retains the original investigation below.

## 2026-10-08 completed capacity fix

All four projection cohorts plus the qualified IQ1_M embedding now execute
natively. Saluki loads **851 direct bindings, zero conversions**, realizing
7.345690 GiB logical weights and eliminating58.726997 GiB of expansion from
the original plan. Loaded physical footprint7.513699 GiB and RSS14.900940 GiB
are distinct point samples in the copied-plus-mapped topology, not peaks.
Independent codec/gather checks and whole prefill/continuation qualification
pass; embedding auto-promotion is scoped to the untied Dense27B fingerprint.
See the packet's complete-result section for exact scope and remaining limits.

## 2026-10-08 XS promotion, pricing fix and resolved family scope

Production IQ2_XS matrix selection now sends `None` through guarded Auto for
N>1: F32 MMA when device/pipeline support and alignment permit, otherwise native
scalar fallback. A test-only forced-Scalar comparator is retained; there is no
runtime force switch. N1 and ordinary GEMV decode remain unchanged.
This changes execution speed, not the completed role-aware native storage policy.

The prepromotion forced-Scalar/MMA whole packet measures 36.4–37.9% lower GPU
time at 128 and 40.1–40.2% at 4096 across two natural corpora, with every paired
GPU/wall comparison improving. A separate postpromotion actual-`None` confirmation
at 129/chunk128 saves 35.8–36.7% GPU and 35.7–36.5% wall. Warm witnesses show
39 MMA projections for the 128-row chunk and 39 unchanged GEMVs for the singleton
tail. All ordinary continuations retain zero MMA substitutions. The owner reports
all 12 promoted dense kernel tests passing with Metal API validation, followed
by all 12 passing debug hazard tracking and a successful non-test cargo check.

The broader `iq2_xs` suite subsequently exposed three DeepSeek scalar-lineage
failures from a real selection-scope leak through shared `encode_batch_projection`.
The owner fixed this with a checked explicit scalar entry and an IQ2_XS pin in
the family helper, retaining the existing exact tests/tolerances and adding a
scope regression. Final broad `iq2_xs` runs with Metal API validation report
**22 passed, 0 failed, 3 ignored in release**, and **22 passed, 0 failed, 3 ignored
in debug**. Both runs execute the three formerly failing DeepSeek tests and the
new scope case. The ignored physical-bindings test separately passes in debug;
the other two ignored tests are the retained leaf/whole packets executed earlier.
The final non-test cargo check passes.

Independent review approves the fix and confirms that the dense body still uses
`scalar_only=false`, with shader/metallib unchanged. The reviewer found no need
to repeat the whole packet for this family-scope correction. Test counts overlap
the earlier dense qualification and should not be added as independent coverage.

All whole-packet outputs are finite and endpoint/four-continuation top1 choices
agree. Maximum bidirectional KL is 2.612e-6 in the 128/4096 packet and 1.830e-6 in
the production 129 confirmation. These are two corpora on Apple M4 Max, short
teacher-forced continuation checks and resident-prefill timings, not broad quality,
free-running, placement-cold TTFT or postpromotion 4K remeasurement. Raw source
bindings distinguish prepromotion leaf/main packets from the historical
postpromotion default-path confirmation. All three packets precede the later
family-scope correction; production129 no longer binds the current source tree.
Raw evidence is unchanged: successful final tests and reviewer approval do not
turn historical packet hashes into current-source bindings. No whole repeat was
required or claimed. Exact timings, numerical differences, limits and commands:
[IQ2_XS MMA evidence](bench/2026-10-08-iq2-xs-mma/README.md).

The shared session pricer now uses `price_shared_buffer_upper` per allocation
before multiplying by count, with typed pricing-error propagation. All 165
allocations were already inventoried; the fix addresses standalone-buffer page
rounding, not missing buffers. **Four CPU regressions pass in debug and four in
release.** Capacity135's modeled
193,593,344-byte upper bound covers the historical 193,314,816-byte allocation
delta that exceeded the old 192,719,648-byte quote by 595,168 bytes. The dynamic
2 GiB reserve remains intact. Later packets record admission bounds but no isolated
live-session allocation delta; they are not new measurements of that rounding gap.

## Flash residual frontier: narrow default confirmed and reviewed

The current planner default groups 2048+2048 / suffix 2048 only for
selected-enabled, packed capacity 2048, dense_end=2051 requests with (start,count)
(0,4096) or (2048,2048). Actual production None versus explicit incumbent
Some(false) completed on UD and GSQ, both corpora and both stages. Production
router and arithmetic policies are unchanged; command grouping still changes
width-dependent arithmetic and GSQ's numerical trajectory relative to the old
schedule. The HC oracle localized the early seed to intended activation
rounding; neither diagnostic HC F32 intervention is promoted.

Independent review supports the narrow promotion using v1 NLL PASS on both
artifacts plus separately frozen v2 retrieval PASS on both. GSQ's v1 retrieval
remains INCONCLUSIVE (correct facts, missing marker), with no rescore. Final
review approves with no fixes and verifies the historical pins. Owner-reported
`qwen4exp_runtime::tests` passes in release and debug: 28 passed, 0 failed,
32 ignored in each. HC capture CPU tests: 3 passed in each; GDN capture CPU
tests: 3 passed in debug. The `qwen-cli` production check passes. Ignored
benchmark acquisitions were executed separately per retained release packets;
these scoped results do not claim a full-suite pass. Commit/integration remains
outstanding, with no landed hash assigned.
The earlier repeatable UD whole 4K saving remains about 2.4% GPU/wall; GSQ is
positive with uncertain magnitude, and the latest one-round means do not
replace that bounded interpretation. Next Flash performance work is
selected-QSA leaf attribution. These are residual scheduling measurements,
not another router or native-IQ kernel gain: these Flash artifacts have no
IQ2_XS tensors. Historical router/map harnesses pin the incumbent schedule
without changing production; retained packet hashes are not rebound to later
helper edits. Evidence: `docs/bench/2026-10-09-flash-frontier/README.md`.

## Earlier IQ2-only checkpoint

Dense IQ2_XS and IQ2_XXS projections now have role-aware native storage and
qualified ordinary execution. Saluki's actual load reconciles 784 direct
bindings and 67 remaining IQ1 conversions: **26.539 GiB logical weights**, a
39.533 GiB reduction from the old plan. Loaded RSS is a separate point sample
of about34.09 GiB, not26.539 GiB. Both native codecs pass real-weight primitive
and short whole-model prefill/continuation checks. Exact evidence, limits and
reproduction: `docs/bench/2026-10-08-native-iq-capacity/README.md`.

IQ1_M (including embedding) and IQ1_S were the next capacity tasks and are
now complete above. The original investigation/staged plan remain provenance;
their "current" storage totals describe the pre-change baseline.

## Historical investigation: verified scope versus reported measurements

Artifact:
`/Volumes/wdblack/weights-archive/underdog-saluki-27b/Underdog-Saluki-27B-1.0-IQ2-mix.gguf`.
Metadata-only inspection finds architecture `qwen35`, 851 tensors and
7,898,369,152 file bytes (7.356 GiB). No weight payload was decoded or model
loaded for this investigation. Header length: 10,994,812 bytes; header-only
SHA-256: `631c2d64c4b90efbe1b6cbae3c11f0103088db8ec2c4b9758e4f0582430a6d1c`.
This is a descriptor binding, not a full weight-content integrity hash.

Reported 75 GiB RSS and 7 tokens/s are measurements, not reproduced here. The
reported expansion is credible, but it is not all IQ2_XXS:

| Unsupported storage cohort | Tensors | Source GiB | F32 GiB | Avoidable GiB |
| --- | ---: | ---: | ---: | ---: |
| IQ2_XXS | 119 | 2.209 | 34.277 | 32.068 |
| IQ2_XS | 39 | 0.582 | 8.047 | 7.465 |
| IQ1_S | 36 | 0.471 | 9.648 | 9.177 |
| IQ1_M, including embedding | 31 | 0.579 | 10.596 | 10.016 |
| Total | 225 | 3.841 | 62.568 | 58.727 |

Thus roughly 4.1 decimal GB compressed to 62.6 GiB F32 describes four formats
together. The reported 226 converted tensors does not match the current 225
descriptor count. Source-derived logical weight storage is about 66.073 GiB
currently, versus 7.346 GiB if all four cohorts remain compressed. These are
not RSS predictions: session state, scratch, allocator/pages and file residency
are additional. IQ2_XXS alone would still leave about 34.00 GiB of weights.

IQ2_XXS occurs only in ordinary rank-two matrices: 53 FFN down, 17 FFN gate,
16 FFN up and 33 attention/GDN projections. Embedding is IQ1_M
`[5120,248320]`; its conversion alone expands 0.259 to 4.736 GiB. The separate
output head is already-native IQ3_S. No MoE bank is required by this artifact.
Vision projectors beside the GGUF are out of scope.

## Historical source diagnosis

`metal_forward/residency.rs::weight_dtype_kept_native` excludes all four
cohorts. `MetalWeightLoader::load_weight` therefore calls `load_f32`, allocating
the complete F32 destination and filling it through the pinned GGML CPU codec.
The bounded decoder does not bound that persistent destination. Loading writes
directly into it; do not invent a second full-size F32 temporary in estimates.
The storage planner accounts for `ConvertedF32`; mmap does not avoid a planned
conversion. CPU IQ2_XXS decoding already exists; native Metal execution is the gap.

## Original execution queue (capacity and XS acceleration now complete)

1. Inventory role/dtype/shape, source bytes, planned resident bytes and execution
   coverage before allocation. Use one role-aware decision in planning/loading;
   do not globally whitelist a dense format and accidentally admit unsupported
   MoE or embedding execution. Preserve existing coverage invariants.
2. Qualify the existing IQ2_XS dense GEMV/GEMM paths and dense prefill eligibility:
   potential 7.465 GiB saving without a new decoder. Its scalar dense GEMM still
   needs a performance check. Keep unqualified roles explicitly excluded.
3. Add canonical IQ2_XXS native GEMV and general-N compressed GEMM, followed by
   tiled prefill where useful. This is the largest new-kernel payoff: 32.068 GiB.
   Together with IQ2_XS, estimated logical weights become 26.539 GiB, not 8 GiB.
4. Complete IQ1_M matrices plus embedding row gather, and IQ1_S matrices. Only
   after these are covered can this artifact remain near compressed weight size.
5. Add few-row acceleration where measured; PR30065 is useful prior art but its
   N2-16 dense speedups neither solve N1/N512 nor establish this capacity result.

Qualification: canonical codec comparisons on small real tensor slices and
synthetic sign/grid/scale edges; offsets, tails, malformed geometry, overflow,
role admission and planner/loader agreement. GPU work uses the normal lease and
memory gate. Record persistent converted bytes/counts, Metal allocation and
process footprint separately, then cold load, warm decode, ordinary prefill and
numerical/continuation behavior. Do not allocate an unsafe inflated baseline
just to prove inflation. Partial milestones must disclose remaining conversions.

Primary success is eliminating persistent expansion while retaining usable
execution, not an arbitrary latency percentage. No changed memory limits,
admission bypass, or promise of total 8 GiB RSS.

Inspection/jam: `cx` session `01a11893-3352-75b1-9ec7-2ed940154a10`, local source
baseline `b8191f82`. Canonical IQ2_XXS is 256 weights in 66 bytes; IQ2_XS is a
different 74-byte format and its decoder is not interchangeable.
