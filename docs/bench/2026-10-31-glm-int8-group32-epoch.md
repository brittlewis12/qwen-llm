# GLM INT8 GROUP32: Epoch Correctness Gate

## Decision

The fixed M5 Ultra hardware audit passes, including independent replay of the
physical output dump. This accepts the opaque INT32 destination layout and
per-epoch reset/rescale arithmetic for two discriminating synthetic patterns.
It is not a speed result, native operator, GPU activation quantizer, model-quality
qualification or production precision-mode change.

The next useful falsifier is a resident cost screen that includes a fresh INT32
reset, conversion and F32 affine accumulation at every K32 epoch. Raw INT8 peak
does not include those costs. Do not reopen the closed Half K64/RowOwned family
or V4's per-16-scale INT8 design on the strength of this result.

## Arithmetic And Scope

GLM's Q4/Q5 weights have independent scale/min pairs per 32 coefficients. Unlike
V4's per-16-scale obstruction, a K32 integer dot can respect that group boundary.
Q6 remains outside this proposal. Normalize native logical Q4/Q5 codes to
0..15/0..31 before signed INT8 conversion; raw shifted high-plane intermediates
are not valid INT8 operands.

For already-quantized activation codes `z`, their scale `a`, sum `u`, and native
weight metadata `D = float(dHalf)*scale6`, `M = float(dminHalf)*min6`:

```text
p = sum(q[k]*z[k]) over exactly K32, fresh INT32
correction = RN_F32(-M*float(u))
affine = FMA_F32(D,float(p),correction)
running = FMA_F32(a,affine,running)
```

The audit uses `matmul2d(32,64,32)` with threadgroup INT8 inputs, an INT32
destination, 128 threads and four SIMD groups. Persistent F32 values are keyed
by coordinates queried from that same INT32 object. It assumes neither a
cross-type cooperative layout nor a hardcoded SIMD-to-matrix mapping.

ABI 2 writes only physical `(epoch, thread, slot)` audit records. The host rejects
duplicate or missing logical coordinates before reconstructing logical outputs.
An earlier source-only dense scatter was removed after independent review and
was never executed. Equal values would not have made duplicate writes safe.

This arithmetic is not bit-exact to maintained C: it removes per-coefficient
Half rounding, proposes INT8 activation quantization and changes reduction order.
Any model implementation needs new quality qualification. The audit's inputs
are synthetic prequantized values, not GPU native unpacking or activation packing.

## Hardware Evidence

- Frozen source: `0cade729e11848fc4b5ab5130e5ff72dbc7734c1`.
- Configured SDK 26.2, Metal 32023.864, Metal 4.0; no SDK override.
- Actual PSO: SIMD32, maximum 128 threads, 4864 static LDS bytes, zero dynamic LDS.
- All 128 threads report capacity 16; every epoch has exactly 2048 unique valid
  coordinates. The 12,288 records above capacity retain their sentinels.
- Both epochs' integer partials, metadata and persistent F32 values agree exactly
  with the independent rational/F64 reference for the bounded dyadic patterns.
- All 1536 guard bytes remain intact; checked terminal completion and process
  exit succeed. Canonical fail-fast lease, actual external timeout, no model load.

The independent replay separately implements the input recipes, integer dots
and affine arithmetic. Its stale-INT32 negative control fails all 2048 outputs;
deferred rescaling fails 2032. This makes the patterns discriminate the two
invalid shortcuts rather than merely accepting zero or insensitive data.

The audit deliberately retains up to 64 slots plus coordinates and records.
Actual capacity 16 is not a compiler register count, spill measurement or
occupancy proof. A slim cost implementation must be revalidated after changing
the shader; audit timing would not be a performance implementation.

## Resource And Performance Limits

Device-priced buffers total 1,097,728 bytes. The explicit fixture, CPU transients
and separate 8 MiB Metal-driver policy total 13,704,288 bytes, below the 64 MiB
explicit cap. A separate 128 MiB framework CPU policy is retained. These policies
are not formal bounds or measured peak residency. Readback uses borrowed guarded
buffers, not an extra full audit clone.

For balanced H4096/F2048/E288/top8/R2048 gate/up, the proposed K32 design issues
618,475,290,624 integer operations and 4,718,592 collective epoch calls. Updating
all padded outputs adds 9,663,676,416 affine updates, about 48.3 GF32 operations,
before reset/conversion, metadata, barriers and control. Inactive production-grid
dispatches must also be represented in a cost screen.

A useful complete-pair screen remains approximately 5-6 ms, with room needed for
native unpacking, activation packing, bucketing, epilogue and output traffic.
Resident reused synthetic tiles exclude native-bank traffic and are only an
optimistic screen, not a formal monotonic lower bound. Current exclusive
application fraction is unknown; neither old routed-parent spans nor effective
TOPS establish application gain.

## Retained Evidence

Private source/build, bounded host validation, physical dump and independent
replay receipts are retained. Public SHA256 identifiers:

| Evidence | SHA256 |
| --- | --- |
| Validation host binary | `d531b1199c0275af7b281a20363bae3363be5227e10f64f63255a2fc8b781dd7` |
| Audited metallib | `94c728d1688581342a2d2239b8ec5885a9d9e2989543c0e8c9e54502e88716f6` |
| Hardware report | `a8a04f07847e28355550fec332f54a25cadf39d8cb7fd4c05832d6aa0d438d20` |
| Physical dump | `73630dae784ba92f7454a28c09bac8ccbf08dcbcf8e0b5a5101d66ad4100f944` |
| Independent replay result | `a5720aac27701d7d142091391931958532e59a1010fa0390c202edf4dc39aef5` |

The standalone validation host is not a canonical application timing binary.
No timing, model logits, model quality or application speedup is claimed.
