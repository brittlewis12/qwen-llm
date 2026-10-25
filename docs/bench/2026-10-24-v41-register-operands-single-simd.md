# V4.1 Register Operands: Admission Repair And One-SIMD Continuation

Final check: [32x32x32 with matched F16ACT](2026-10-24-v41-register-operands-32x32-f16.md).
The streaming numbers in this continuation use F32 activation storage, not the
Maya F16ACT hot path. The final check compares actual F16ACT incumbent/candidate
storage and the supported one-accumulator shape; it also loses. These older
results remain unchanged and are not relabeled as hot-F16 evidence.

## Decision

The requested bounded continuation is implemented and measured. The harness now
uses separate Metal, CPU/private and combined physical-capacity domains. Routing
is bijective for all supported counts, and fixture/command autorelease pools
drain independently. A whole-macro one-SIMD API attempt, a supported 512-point
fallback, direct output epilogues and a two-stage register transpose were tested.

There is still **no winning leaf**. The best new arm, the two-stage K16 transpose
with direct output, takes 11.543889 ms versus 6.219125 ms for the incumbent at
IQ3_XXS count32: **1.856x slower**. Do not expand dtypes, production count bands,
full models, loader architecture or defaults on these losing variants.

Final measured implementation: `9575bd2c649059f1842da7039e88a0d1b40f7ca6`, frozen
under `frozen12-butterfly`. The preceding qualified one-SIMD K32 cohort is frozen
under `frozen11-single-simd` at `c64935da`. Original build1-6 result files remain
unchanged; their numerical observations are historical, not results regenerated
with the corrected admission/lifetime model.

## Admission Domains

`contracts.h` contains the same pure decision logic used by the harness and CPU
tests. Before fixture buffer allocation, `split-domains-v2` records each requested
Metal length and its independent host-page ceiling, raw device/host domains and
the explicit CPU-storage scope:

1. Incremental Metal is the sum of page-ceiled lengths of all seven Shared
   buffers. It must fit actual `recommendedMaxWorkingSetSize - currentAllocatedSize`.
2. Only additional CPU vectors, touched bytes and bounded C++ control/name
   storage face conservative CPU availability: `free + max(inactive-anonymous,0)`.
3. Existing observed owned Metal plus incremental Metal plus incremental CPU
   must fit actual `hw.memsize` physical capacity.

Shared mapped storage is counted once as Metal, even when initialized/read by the
CPU. In particular, the multi-GiB Shared bank does **not** face the conservative
CPU cache/availability gate. Unknown/inconsistent host responses, arithmetic
overflow and maxBufferLength violations fail closed. Mach host response lengths,
page size, physical capacity and send-right release are checked. No percentage
reserve, cache-based GPU veto, sysctl write or explicit residency policy is used.

At production count32, explicit prices are:

| Domain | Bytes |
| --- | ---: |
| Incremental Metal, seven independently page-ceiled buffers | 2,125,070,336 |
| Additional CPU controls; no timing seed/reference/touched arrays | 16,384 |
| Combined fixture incremental price | 2,125,086,720 |
| Read-only MWSS | 257,698,037,760 |
| Actual physical capacity | 274,877,906,944 |

The bank itself is still 1,734,082,560 native IQ3_XXS bytes for 384 x 5120 x 2304.
This is not a cache-sized parity bank. The maximum new tiny fixture price is
6,930,432 bytes, beneath the enforced 128 MiB explicit fixture cap.

Prices are scoped, not a claim about complete process RSS or an OS reservation.
Foundation/Metal runtime objects, PSO/driver heaps and allocator metadata have no
claimed byte bound. The known framework scope is two libraries, one queue, a
selected-arm PSO cache and one outstanding command; raw currentAllocatedSize and
post-scope observations remain in the records. Large known CPU arrays are priced
separately; observed capacities are also logged. C++ name/control collections
have explicit count/capacity bounds checked by the harness, not a hardware
headroom reserve.

## Routing And Lifetime Repair

The old `(i*17)%count` aliases at counts 17/34/51. The generator now chooses a
step coprime to count. A compiled CPU property test covers every count 1..64,
multiple experts, dense identity routing, both invalid-ID controls, bounds and
uniqueness of all remaining valid destinations. It includes negative checks
showing that the old generator aliases. The previously measured
1/2/8/16/31/32/33 mappings do not change.

Only parity allocates the byte-per-output-word touched array; streaming no longer
allocates an unused CPU mask. Seed, CPU reference and incumbent-copy arrays are
three separately page-priced CPU allocations. Each parity fixture and the timing
fixture has its own autorelease pool; each run/decode command also has a pool.
This avoids retaining all completed command buffers and their resources through
the outer main pool.

`fixture_scope_end` reports actual device allocation before the fixture and after
C++ destruction/pool drainage, plus retained PSO count. There is **no arbitrary
return-to-baseline tolerance or release assertion**. For example, the last build12
parity scope reports 1,769,472 bytes before and 6,225,920 after with 33 cached
pipelines. That retained allocation is reported, not silently forgiven or called
a buffer leak/spill measurement. The next fixture observes live current usage.

The final production timing fixture reports 458,752 API bytes before allocation,
2,125,479,936 with the buffers allocated, and 720,896 after fixture/pool drainage
with nine PSOs retained. Its seed/touched CPU capacities are both zero. These
are observed scopes, not an invented release allowance. Conservative CPU
availability at admission is 134,157,074,432 bytes, applied only to the 16,384-byte
CPU increment, not to the Shared Metal bank.

## Actual API Limit And Supported Fallback

The first whole-macro attempts use a single SIMD and both cooperative inputs with
descriptors 64x32x32 and 32x64x32. SDK 27 rejects them in
`MPPTensorOpsMatMul2dImpl.h:6365` and `MPPTensorOpsMatMul2dImpl.h:6366`:

```text
M must be 16 or 32 if both inputs are cooperative tensors
N must be 16 or 32 if both inputs are cooperative tensors
```

The adjacent assertions also require K in {16,32} and at least one dimension 32.
The full failed build8 source/log is preserved; this is a real API instantiation,
not the old empty language probe.

The supported fallback retains the full 64-weight-row x 32-slot macro and grid,
but one 32-thread SIMD owns four 512-point accumulators. Each operation is
16x32x32. Weight-left reuses one activation cooperative tensor across four weight
strips; activation-left reuses two activation and two weight tensors across a 2x2
output partition. The accumulators keep K32 order. Each orientation has:

- A retained 8 KiB threadgroup output control, followed by a per-slot epilogue.
- A direct cooperative-accumulator epilogue to the same valid global addresses,
  with zero dynamic threadgroup bytes.

These are separately labeled ownership/output variants, not a claim that only
input delivery changes. Count guards, ID validation, b_div, weight-row and slot
tails and the three mathematical epilogues remain. Full N2048 grid dimensions
remain `(64,80,384)` in production timing; no compaction or N8/N16 slot tile is
introduced.

## Two-Stage Transpose

The final alternative uses the verified K16 input layout. Producer rows are
`FM + 8*quarter`, with FM and quarter derived from lane bits. A 4x4 register
transpose exchanges two selected uint2 vectors in each of two XOR stages, masks
1 and 8. It does not broadcast all four fragments and then pick one. A CPU
lane/register property test verifies the complete permutation; independent
API-coordinate dumps verify the actual cooperative inputs on the GPU.

The original quant decoder still produces the same half16 values. Two right-weight
blocks cover all 64 rows without duplicate per-quarter dequant calls. K16 doubles
the number of K steps/operation calls relative to K32 but does not increase the
descriptor-padded FLOP count. It is a separately qualified numerical arm, not an
unverified K32 mapping or a new decoder/quant artifact.

## Parity And Audit Investigation

The corrected legacy IQ3 suite passes at `domain-v2-parity7-iq3` before the new
experiment: 203 output comparisons and 12,091,392 half checks, zero bit mismatch.

Build9/10 initially expose an audit discrepancy in the activation-left retained
output arm: output comparisons match, but decoder-temporary audit stores report
weight mismatches (activation mismatches zero in the localized build10 record).
Those failed runs stop before timing and remain in the archive. No tolerances or
assertions are relaxed. The audit is strengthened to read the actual populated
cooperative elements before consumption, through volatile diagnostic stores,
rather than copying decoder temporaries. The revised dumps pass. The underlying
lowering cause of the earlier temporary-store discrepancy is not established;
it is not advertised as a diagnosed compiler bug.

Qualified build11 then passes 62 output comparisons and 2,359,296 half checks.
Final build12 passes **86 output comparisons and 3,538,944 half checks**, all with
zero observed bit mismatch against the production-target incumbent and canonical
half operands. This includes exact timed PSOs and separate auditing twins,
diagnostic count1/32/33 guards, M70 tails, multiple experts, dense32 and grouped
K2304, invalid IDs, empty/negative/above-max counts, b_div=6 and all epilogues.
Diagnostic count1/33 fixtures are not production routing claims or timing sweeps.

All untouched output words/guards retain their seed bits; command statuses
complete. The existing half-reference numerical thresholds remain unchanged;
minimum cosine is 0.9999999999998951. This does not rerun the independent native
F32-dequant oracle or establish universal frozen exactness/full-model quality.

## Final IQ3 Count32 Timings

Build12 whole-kernel GPU-envelope medians, nine samples per arm. Each sample is
three serial dispatches divided by three, with warmed PSOs and rotated order.

| Arm | Median ms |
| --- | ---: |
| Incumbent metal4.0@26.0 | 6.219125 |
| Same-source incumbent metal4.1@27.0 | 6.204569 |
| Earlier best streaming wx16 | 11.820750 |
| One-SIMD K32 weight-left, direct | 17.809458 |
| One-SIMD K32 weight-left, retained output | 26.434986 |
| One-SIMD K32 activation-left, direct | 21.686917 |
| One-SIMD K32 activation-left, retained output | 27.738611 |
| Two-stage K16 transpose, direct | 11.543889 |
| Two-stage K16 transpose, retained output | 11.690528 |

Selected raw samples, milliseconds (full precision remains in JSONL):

```text
incumbent26: 6.218708 6.224819 6.219292 6.219611 6.215625 6.221444 6.219069 6.219125 6.216625
bt16 direct: 11.537861 11.539472 11.561542 11.543889 11.561750 11.546528 11.537069 11.545042 11.526125
bt16 staged: 11.681667 11.684097 11.657500 11.703583 11.690528 11.694861 11.679028 11.701375 11.698750
```

The direct transpose ranges 11.526125-11.561750 ms; the incumbent ranges
6.215625-6.224819 ms. Largest final arm max/min ratio is 1.009586, descriptive
only. All 81 final samples remain, plus all 63 build11 timed samples. Neither
cohort pools or filters away the other. The earlier build11 one-SIMD negative
agrees: 17.839250 / 26.396708 ms weight-left direct/staged against 6.217319 ms.

Useful and descriptor-padded work at count32 remain 289,910,292,480 FLOPs. The
single-SIMD arms eliminate cross-SIMD operand duplication and explicit operand
staging. Retained output writes 251,658,240 logical threadgroup bytes; direct
output removes that staging but still writes the same valid global output.
These are logical ledgers, not DRAM counter measurements. No occupancy/spill
counters were obtained; public max-thread limits and retained allocation changes
are not substitutes. Removing staging is demonstrably insufficient here.

## Reproduction And Scope

Use `parity-single iq3_xxs` before `timing-single iq3_xxs 32 9`, with both absolute
private library paths, as in `scripts/bench/register-operands/README.md`.
Build12 pins source `9575bd2c649059f1842da7039e88a0d1b40f7ca6` on M5 Ultra,
macOS 27.0.1 / SDK27 / metal 32023.921. Product compiler/deployment controls remain
independent; the product 26.0 minimum, loader and defaults are unchanged.

Private evidence archive label: `register-operands-private`; all artifact names
are relative to that archive or its local mirror. The
[source and evidence inventory](2026-10-24-v41-register-operands-inventory.md)
records the handoff without machine-specific locations.

New evidence uses `domain-v2-*`, `single-simd-*`, `butterfly-*` and
`continuation-v2-*` names. `frozen12-butterfly/identity.json` binds exact binary,
both libraries, AIR, CPU contract test and complete source archive to the HEAD,
compiler and SDK. Successful frozen identities, including original builds, are
rechecked after measurement. Original result files are not overwritten.

Final frozen SHA256 identities:

```text
probe:              abcc0c2128c7913dbca8332873c520925c397ac8f4609894d873196208abe152
private.metallib:    d26b0e19b5284aecf4a381723acfac2825afc8c1c0de64c7cf6c1f9328e9ec88
baseline26.metallib: e7a37d6471b2cfd978decffeb8443c9ca218c3b54349e46089ded1db74948e83
source.tar:         6fb9e1a8634895c49ea6f9d3dffe5f4ee4e974cf7477f6ae5685d65ddd9c1f87
```

All remote builds and GPU jobs use the production lease and established
240/360-second observer, with exclusive output files and no automatic retries.
Correctness uses MTL_DEBUG_LAYER=1; timing sanitizes validation/injection/capture
only in the child environment. No SIGKILL, purge, pinning, private coordination,
full-model run or public push occurs. The native C++ contract/property tests,
30 unchanged observer tests, reporting test, Ruff and diff checks pass.

This is the supported negative handoff, not a proposal for an endless sweep.
The tested transpose is the best new candidate but still loses decisively; the
monolithic both-cooperative whole-macro descriptor is explicitly unsupported.
Leave production unchanged and retain the fixed harness and bounded evidence.
