# Register-Fed Quantized Matmul Probe

Measured results: [original bounded negative](../../../docs/bench/2026-10-24-v41-register-operands.md)
and [corrected-admission/one-SIMD continuation](../../../docs/bench/2026-10-24-v41-register-operands-single-simd.md).
The continuation remains negative; no production integration is enabled.
The [final 32x32 matched-F16ACT check](../../../docs/bench/2026-10-24-v41-register-operands-32x32-f16.md)
also loses; historical F32 cohorts are not hot-F16 production evidence.
The [source/evidence inventory](../../../docs/bench/2026-10-24-v41-register-operands-inventory.md)
uses the generic private archive label `register-operands-private` and relative
file names. Historical source archives and raw observer logs remain private;
do not publish them wholesale as sanitized documentation.

Test/benchmark-only private library. Nothing in `build.rs`, product context
loading, dtype/storage, defaults, or deployment targets changes. Requires M5
(Apple10), SDK 27, macOS 27 and the real cooperative-input API; this is not the
product's empty MSL-language compiler probe. Both inputs are half, accumulation
is float, output staging and the three production epilogues are retained.
The frozen build also compiles the unmodified incumbent separately with the
product's `-O3 -ffast-math metal4.0@26.0` flags. Both library paths are explicit. A
`baseline27:` control uses the incumbent from the newer private library, so
deployment/language code generation is not silently conflated with dataflow.

`probe.metal` includes the maintained tensor kernel for the incumbent and calls
the unmodified `quant_tiles.h` decoders. Two mappings retain the 128-thread,
64-weight-row x 32-slot macro tile, K32 steps, and full incumbent grid:

- `ax`: four SIMD groups own 16 slots x 32 weight rows (2 x 2 output partitions).
- `wx`: four SIMD groups own 16 weight rows x 32 slots (4 x 1 partitions).

The `ax16` / `wx16` alternatives use the verified MLX per-lane K16 fragment
mapping, checked independently against API-coordinate input dumps. They change
the K partition to 16 (separate numerical variants), but not the macro tile or
grid. They avoid input-coordinate queries in the hot loop rather than making a
larger tile. The `*_resident` arithmetic controls load/decode one K32 panel once,
then reuse it for all K steps, either in threadgroup memory (`staged_resident`)
or in cooperative inputs (`ax_resident`, `wx_resident`). Their initial loads,
output staging and epilogue are still timed; they are not streaming controls.
K16 alternatives are down-only; the K32 prototype retains all three epilogues.

`parity-single` / `timing-single` are a separately labeled IQ3-only experiment:
one 32-thread SIMD group owns the whole unchanged 64-weight-row x 32-slot macro
through four 16x32 operations/accumulators, reusing inputs across those operations.
SDK 27 rejected both original 64x32/32x64 descriptors: with both inputs
cooperative, M and N must each be 16 or 32 (and K must be 16 or 32).
Both weight-left and activation-left fallback mappings have a retained
8 KiB output-staging control and a direct cooperative-accumulator epilogue with
zero dynamic threadgroup bytes. These change ownership and output delivery, not
just input delivery. K32, full macro grid, count/ID guards and all three
mathematical epilogues are preserved. Small diagnostic parity covers partial
slots, the second panel, weight-row tails, dense32 and full K2304 before any
production-bank IQ3 count32 timings; do not infer native route counts from these
fixtures. No new dtype or default is introduced.

`sg1axbt16` and its retained-output twin are the final bounded alternative:
producer rows `FM + 8*quarter` decode full half16 fragments, then transpose row
and K-quarter ownership in two XOR stages (lane masks 1 and 8). Each stage
exchanges two selected uint2 vectors, rather than broadcasting four then picking
one. A CPU lane/register property test and independent API-coordinate dumps check
the mapping. This uses the verified K16 layout, not an unverified K32 layout;
descriptor FLOPs remain unchanged but the K partition and operation calls differ.

`parity-32x32` / `timing-32x32` are the last supported-shape check, initially IQ3
count32 only. Unlike the historical F32-input cohorts, this uses actual F16ACT
storage for both incumbent (`_f16act`) and candidate: identical pre-rounded half
buffers, no F32 shadow, and no per-dispatch conversion. Fixture generation time
includes the initial rounding into that shared input. One SIMD
owns one 32x32x32 operation/accumulator, with M/32 grid height, direct output or a
4 KiB retained-output control. The incumbent retains its M/64 grid and 8 KiB
staging. Full N2048 slot stride, quant layout and valid destination addresses do
not change. Audits record actual input/accumulator capacities and SIMD width,
and inspect populated half operands. Tiny M70 tails and dense32/K2304 references
precede full-bank timing. No production benefit may be inferred from F32-only
timings; expand count1/33 or IQ2 only if this count32 case is competitive or a
specific diagnostic requires it.

The `axsh` / `wxsh` K16 variants retain that fragment mapping but decode a full
16-value vector per producer lane and distribute unchanged half values via SIMD
shuffles. This removes per-slice duplicate decoder work without changing the
decoder or using threadgroup operand staging. Both the coordinate-based K32 and
explicit-fragment K16 implementations remain as separate controls.

The new API permits input cooperative tensors only at single-SIMD scope, so
output ownership changes, not outer K partition. Reduction bit parity is
reported, never assumed. Four-element dequant slices are checked by dumping all
cooperative half inputs for the first tile against a canonical dequant16 bank.
The tiny suite checks both the exact non-auditing PSOs used for timing and their
separate operand-auditing twins; audit stores are never present in timing PSOs.

The tiny suite prices every allocation before allocation, enforces <=128 MiB,
tests counts 1/2/8/16/31/32/33, both low-bit types, all three epilogues, invalid
IDs, empty/negative/above-max-count experts, output guards, untouched slots, M70 tails,
and actual dense32/dense64 reference kernels at K2304. CPU dot products use
canonical decoded half weights and half-rounded activations. It retains the
existing grouped tests' cosine > .999, max-absolute <= .01*reference-max+.0001,
and tensor comparison <= .002*baseline-max+.00001 contracts, while separately
reporting bit mismatches. This is synthetic packed-data coverage, not full-model
quality evidence or an independent decoder reimplementation.

Resident controls have their own repeated-K32 reference test. A clamped b_div=6
fixture checks the per-token activation mapping as well as b_div=1 down.

Timing uses ordinary shared allocations of the full 384 x 5120 x 2304 native
packed bank, not the tiny parity fixture. Admission schema `split-domains-v2`
page-ceils each Metal buffer separately and admits that incremental sum against
actual `recommendedMaxWorkingSetSize - currentAllocatedSize`. Only additional
CPU arrays, touched bytes and explicit controls face conservative CPU
availability. Existing owned Metal plus both incremental domains must fit actual
physical capacity. CPU writes to Shared mapped storage do not double-charge it.
Unknown/inconsistent telemetry and maxBufferLength violations fail closed.
Raw domains and per-buffer prices are logged; no percentage reserve, CPU-cache
GPU veto, residency sets, pinning or forced eviction is used. Framework/allocator
objects have no claimed byte bound: two libraries, one queue, selected-arm PSO
cache and at most one outstanding command. Per-command and per-fixture pools
drain; observed post-scope allocation is reported without a release tolerance.
Historical frozen1-6 prices used the incorrect combined CPU-availability gate;
their result files remain immutable and new measurements use distinct labels.
`test_contracts.cpp` tests the real domain logic and all supported slot counts
1..64, including invalid-ID controls and the former 17/34/51 permutation aliases.
Native packed blocks
are initialized once; no predecode/packing is hidden from kernel timings.
The full N2048 grid is retained even for low-count bands. PSOs are warmed before
rotated matched samples. Each sample is a command GPU envelope divided by three
serial whole-kernel dispatches; raw host envelopes are also retained. Logical
bank/staging bytes are not DRAM traffic. Public PSO max-thread limits are not
occupancy/spill evidence. Every output remains externally consumed.

Use a clean owned checkout and a **new** build directory under a private archive:

```sh
uv run scripts/bench/register-operands/observe.py --output /archive/build1 --deadline 360 -- \
  uv run scripts/bench/with-metal-lease uv run scripts/bench/register-operands/build.py /archive/frozen1
uv run scripts/bench/register-operands/observe.py --output /archive/parity-iq3 --deadline 240 -- \
  uv run scripts/bench/with-metal-lease /archive/frozen1/probe /archive/frozen1/private.metallib /archive/frozen1/baseline26.metallib parity iq3_xxs
uv run scripts/bench/register-operands/observe.py --timing --output /archive/time-iq3-c32 --deadline 360 -- \
  uv run scripts/bench/with-metal-lease /archive/frozen1/probe /archive/frozen1/private.metallib /archive/frozen1/baseline26.metallib timing iq3_xxs 32 9
```

The legacy wrapper acquires the production lease exactly once; the standalone
executable does not acquire another lease. The observer uses the established
pilot's owned-child deadline, host critical-pressure stops and INT->TERM policy.
Each invocation owns exclusive output files. Timing sanitizes only the child's
debug/validation/wrapper-injection/capture environment. Builds freeze executable,
metallib, complete source archive, compiler/SDK identity and SHA256s under the
same build lease. Do not rebuild into a frozen directory or measure dirty source.
