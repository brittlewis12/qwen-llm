# Absorbed GLM MLA: Mixed-Operand Hardware Gate

## Decision

The bounded N64 QK/PV source/API gate passes actual M5 collection, complete
offline checking and independent physical/arithmetic replay. This establishes
the frozen cooperative operand contract and captured ownership, not fused
attention, model-derived operands, N2048 performance or model quality.

Proceed to the separate CURRENT attention-segment budget discriminator before
funding a complete fused operator. The accepted post-C 2K profile places the
combined MLA parent at 376.432251 ms across 11 blocks; it includes projections,
grouped transforms and cache/indexer maintenance, not just attention. It supplies
neither an attention-only denominator nor an exclusive affected fraction.

The later whole-MLA continuation bar remains at least 30% AND 8 ms saved per
2K block, with all conversion, layout, unchanged projection and publication
costs included. A primitive witness or one-block result cannot be multiplied
by 11 and presented as an application gain.

## Fixed Contract

- 32 query rows by 32 key rows, latent width 512, two 16-row cohorts and four
  dimension partitions, eight SIMD groups / 256 threads.
- Q and independently supplied P remain F32; one shared latent C cache remains
  F16. Accumulation/output are F32, with non-relaxed descriptors and
  `-fno-fast-math`; no implicit Half/TF32 permission is borrowed from Fast/Half.
- Position-zero causal visibility includes the current row. QK scale is 1/16,
  not the conventional 1/sqrt(512). There is no sink or softmax in this API probe.
- QK accumulates eight K16 operations per 128-dimension partition, then explicitly
  merges four F32 partials in order. PV separately exercises two K16 operations
  for each output tile; P is not the result of a fused softmax.
- All audit writes use physical stage/thread/slot indices, not coordinate-derived
  global scatter. Status retains error, physical tid, actual SIMD group and lane.
  `tid == sg*32 + lane` is a checked runtime ABI eligibility condition, not an
  assumed general Metal guarantee.

The producer, numerical rules and shader remain frozen at source
`6fb48573e1e3568683b2c75e714355822994153b` and the preserved `8993f917` probe.
The source experiment is retained on `probe/glm-mla-absorbed-512`; maintained
main carries this report, not an unqualified production kernel or default.

## Execution And Replay

Three fixed input families cross heads 0/1/63 and tiles (0,0), (32,0), (32,32):
27 commands, each with one QK and one PV dispatch. Every dispatch is one
256-thread group. N64 inputs include all 64 Q heads and one C cache, with
separate guarded Case, Q, C, P, output-record and provenance buffers.

The collector completes under DEBUG1, a fail-fast self-lease and an actual 120 s
timeout. It reports collection only, never self-certifies arithmetic. Process
exit and lease release are separately observed. Authored offline validation and
independent replay run afterward on an unleased CPU, without another GPU run.
The original and subsequently hardened metadata parser produce identical actual
offline results; the hardening does not change shader, data or numerical rules.

Independent replay verifies:

- All 4,202,496 records: 3,317,760 valid and 884,736 invalid, including invalid-slot
  sentinels, addresses, operand value bits, capacities, flags and ownership.
- All 13,824 status records: zero errors, complete unique actual SG/lane membership
  and the checked scratch-index eligibility relationship.
- All 110,592 stage17 replicas: 74,880 visible and 35,712 correctly masked negative
  infinities, with exact ordered F32 merges of the captured partials.
- All 1,769,472 guard bytes and 228,372,912 immutable input bytes, plus all frame,
  asset, source, collector, manifest and cached-link identities.

The trusted supervisor receipt binds the report hash, reviewed collector binary,
exit zero and lease release. These are explicit parent observations, not remote
or loaded-image attestation. The authored checker's result intentionally leaves
hardware qualification to the separate independent replay.

## Arithmetic Evidence

Exact sparse Q/P impulses retain the off-TF32-grid `1 + 2^-12` mantissa:
scaled QK bits are `0x3d800800`, PV bits `0x3f800800`. The separate `2^-25`
Half control retains `0x33000000`. All 368,640 sparse partial/PV values pass
their exact-bit rule. Cancellation follows the declared ordered F32 merge,
producing `0x3d800000`, not the real-sum 1/8 result.

General-grid references use exact integer-dyadic products and rational error
comparisons under the predeclared gamma bounds, without choosing an opaque
within-dot accumulation tree or widening bounds after the result:

| Check | Values | Maximum Absolute Error | Maximum Bound Utilization |
| --- | ---: | ---: | ---: |
| QK128 partial | 36,864 | 8.58306884765625e-6 | 11.5070% |
| PV32 output | 147,456 | 0 | 0% |
| Scaled QK512 visible logical scores | 6,240 | 4.991888999938965e-7 | 4.70948% |

Maximum-error and maximum-utilization locations may differ. This is acceptance
under frozen bounds, not F64-dot bit equality or a general IEEE implementation
claim. The exact impulses remain separate from the broader error allowance.

## Layout And Resource Limits

Q/P input capacity is eight: slots 0-7 valid, 8-15 invalid. K/V inputs and
score/output capacity are 16, with all slots valid. Each role has one observed
normalized map across its captured cases/steps. In particular, a capacity16
score fragment is NOT interchangeable with the capacity8 P-input fragment.
Use the API and role-specific proof rather than copying a donor fragment map.
No register count, spill, occupancy or instruction-selection conclusion follows.

Fresh actual PSOs report SIMD32/max256, QK static32768/PV static0 and dynamic0,
against the 32768-byte device LDS limit. The QK audit has zero LDS headroom;
this does not establish capacity for a different fused shader.

Eight guarded payloads total 13,512,720 bytes, below the 128 MiB raw-payload cap.
The priced-buffer ceiling is 13,565,952 bytes; separate Metal framework and
command policies yield a 114,229,248-byte Metal upper. Counted CPU plus its
separate framework policy is 98,394,144 bytes, with no incumbent credit.
Observed per-case allocation-counter growth is 14,155,776 bytes. These counters
and policy envelopes are not formal or measured peak bounds.

## Retained Identities

Shader compilation uses configured SDK26.2 / Metal32023.864 / Metal4.0. The
standalone collector uses configured Rust1.99.1, opt1/no-LTO and unchanged cached
dependencies; it is not a canonical application timing artifact. No model
weights, server, installed binary, production default or PATH changes occur.

| Evidence | SHA256 |
| --- | --- |
| Probe source | `dc82e173fcb2e162be8d39af47ed802fd7d40a1ad695a145a755c8fd25e405e7` |
| Probe metallib | `9149b9846f227ec6d1e9f5f2fafe09dafa21c8ccdecc2c7939f554944afc4af6` |
| Collector | `07a7f0ba89e3ca714d02c74b0bad9942df884014c4d5914965cdf38f2bb8e2aa` |
| Fixed manifest | `b86bdf09717fd3bbdc1ab9369172d724f4ce8c9780a956938c491d761cfe8b00` |
| Actual report | `89f529af4ff522c324cffe6013ffbc9b82f7e92414154af2422e00d708e19392` |
| Authored offline result | `a6213aa925d667d7ac13d50a7abe196210d36ebe59d56f4fc684ca4b6deee9ed` |
| Independent replay | `bd279eba3c46e8158ca4b4ce030d25fec4f78c00cdb5ff22108476b28e38b791` |
| Observed-map record | `a63a5f97cb7e16938d05c3e85dd46f02fcdd09cb85d97f8c9a05de581c621e4f` |

Actual-weight behavior, fused max/exp/denominator/PV coupling, N2048 throughput,
native whole-MLA savings and full-model continuation/task quality remain future
gates. None is implied by this successful API witness.
