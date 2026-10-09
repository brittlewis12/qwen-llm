# Native IQ quant capacity queue

Queued 2026-10-08 at the user's request, next after the completed GLM MLA tail
screen (experimental selection remains off). This is a capacity/coverage task,
not a small-throughput optimization.
No production admission or kernel selection changes are made by this note.

## 2026-10-08 completed capacity fix

All four projection cohorts plus the qualified IQ1_M embedding now execute
natively. Saluki loads **851 direct bindings, zero conversions**, realizing
7.345690 GiB logical weights and eliminating58.726997 GiB of expansion from
the original plan. Loaded physical footprint7.513699 GiB and RSS14.900940 GiB
are distinct point samples in the copied-plus-mapped topology, not peaks.
Independent codec/gather checks and whole prefill/continuation qualification
pass; embedding auto-promotion is scoped to the untied Dense27B fingerprint.
See the packet's complete-result section for exact scope and remaining limits.

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

## Verified scope versus reported measurements

Artifact:
`/Volumes/wdblack/weights-archive/underdog-saluki-27b/Underdog-Saluki-27B-1.0-IQ2-mix.gguf`.
Metadata-only inspection finds architecture `qwen35`, 851 tensors and
7,898,369,152 file bytes (7.356 GiB). No weight payload was decoded or model
loaded for this investigation. Header length: 10,994,812 bytes; header-only
SHA-256: `631c2d64c4b90efbe1b6cbae3c11f0103088db8ec2c4b9758e4f0582430a6d1c`.
This is a descriptor binding, not a full weight-content integrity hash.

The user's 75 GiB RSS and 7 tokens/s are reported measurements, not reproduced
here. The reported expansion is credible, but it is not all IQ2_XXS:

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

## Source diagnosis

`metal_forward/residency.rs::weight_dtype_kept_native` excludes all four
cohorts. `MetalWeightLoader::load_weight` therefore calls `load_f32`, allocating
the complete F32 destination and filling it through the pinned GGML CPU codec.
The bounded decoder does not bound that persistent destination. Loading writes
directly into it; do not invent a second full-size F32 temporary in estimates.
The storage planner accounts for `ConvertedF32`; mmap does not avoid a planned
conversion. CPU IQ2_XXS decoding already exists; native Metal execution is the gap.

## Execution queue

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
