# V4.1 Final Supported-Shape Check: 32x32x32 With Matched F16ACT

## Decision

The last requested shape and hot activation dtype are implemented and measured.
The one-SIMD, one-accumulator 32x32x32 direct path takes **10.675500 ms** versus
**5.235042 ms** for the matched F16ACT incumbent at IQ3_XXS count32: **2.039x
slower**. Its retained-output control takes 11.400333 ms. Close the register-fed
family as a well-tested bounded negative for now; do not expand counts, IQ2,
full models or product integration on this loss.

Measured source HEAD: `eb89e4d87391e0aec7a9f38169b79e2ec3f74562`.
Frozen build: `frozen13-tile32-f16` in the existing private archive. All older
proofs and raw files remain unchanged; 253 prior files are byte-verified against
the preexisting local mirror.

## Correct Hot-Dtype Comparison

The earlier streaming cohorts used `_f32` activation entry points. Compiling an
incumbent at the product deployment target did not make those F16ACT hot-path
measurements. This final cohort explicitly uses:

- `kernel_moe_down_iq3_xxs_f16act_grouped_slots_tensor`, unmodified production
  source compiled with `-O3 -ffast-math -std=metal4.0` at the 26.0 target.
- The same F16ACT incumbent source in the private metal4.1@27.0 library, as the
  compiler/deployment control.
- `sg1t32_iq3_xxs` and `sg1t32tg_iq3_xxs`, both taking `device const half *`.

All four arms read the **same already-rounded F16 input buffer**. There is no F32
shadow allocation, candidate packing/predecode, or per-dispatch conversion.
Synthetic fixture generation includes the initial rounding and reports its
elapsed time separately. Pipeline/storage-signature checks reject a dtype
mismatch. No production win is inferred from the historical F32 cohorts.

## Shape And Contracts

The new operation is exactly `matmul2d(32,32,32)` with `execution_simdgroup`.
One 32-thread group owns 32 slots x 32 output rows and one cooperative F32
accumulator. Audit records confirm, rather than assume:

```text
left input half elements per lane:  32
right input half elements per lane: 32
accumulator float elements per lane: 32
SIMD width:                         32
```

The candidate uses grid `(ceil(N/32),ceil(M/32),experts)` instead of the
incumbent's M/64 height. At production shape the grids are `(64,160,384)` versus
`(64,80,384)`. Both retain N2048 ID/slot stride, K2304 activation element stride,
native compressed IQ3_XXS weights and the same valid global destination
addresses. This is a different macro geometry, not a renamed four-accumulator
whole-macro path or a compaction variant.

The direct arm has zero dynamic output threadgroup bytes. The retained-output
control stages a 32x32 float tile in 4096 bytes; the incumbent uses 8192 bytes for
its 32x64 tile. Count filters, invalid IDs, slot limits, b_div, row tails and the
three original mathematical epilogues remain.

Coordinate-based cooperative population is used as requested. Actual populated
half inputs are dumped independently using API coordinates and volatile audit
stores; timed PSOs contain neither trait nor operand dumps. No producer rewrite,
larger macro, K16 variant or extra shape sweep is added after this loss.

## Bounded Parity

`tile32-f16-parity13-iq3-c32` completes under MTL_DEBUG_LAYER=1:

- 26 output comparisons, including the exact timed PSOs and separate audit twins.
- 655,360 half-operand checks and six actual-capacity/SIMD-width trait checks.
- Zero observed output or half-operand bit mismatches versus the matched F16ACT
  incumbent/canonical decoded half inputs.
- M70 row tails, grouped K512 and K2304, actual dense32/K2304 references, multiple
  experts, invalid IDs, empty/negative/above-max counts, all epilogues and b_div=6.
- Every untouched output word and guard preserves its seed bits; commands complete.
- Maximum explicit tiny fixture price 5,095,424 bytes, below the enforced 128 MiB
  fixture cap. No count1/33 or IQ2 expansion is required to establish this loss.

The CPU reference accumulates canonical half weight values and the stored half
activations. Existing numerical threshold values remain unchanged; minimum
cosine is 0.9999999999998951. This is bounded fixture evidence, not universal
frozen exactness, an independent native F32-dequant oracle rerun, or full-model
quality qualification.

## Fully Priced Streaming Result

The actual 384 x 5120 x 2304 compressed bank holds 1,734,082,560 bytes. F16 input
storage holds 60,171,264 bytes. Counts are a uniform static count32 fixture with
permuted unique IDs, not captured model routes or counts inferred from request
length. The repaired admission model remains in force:

| Explicit Incremental Domain | Bytes |
| --- | ---: |
| Seven independently page-ceiled Metal buffers | 2,064,891,904 |
| CPU controls, no timing seed/reference/touched arrays | 16,384 |
| Combined explicit fixture increment | 2,064,908,288 |

Metal is admitted against actual MWSS minus currentAllocatedSize; only CPU/private
growth faces conservative CPU availability. Existing owned Metal plus both
increments face actual physical capacity. Shared mapped storage is counted once.
Unknown telemetry/maxBufferLength failures remain closed, without a percentage
reserve or CPU-cache GPU veto. Framework/allocator objects remain explicitly
unpriced; this is not a complete process-RSS guarantee or OS reservation.

The final fixture records 458,752 API bytes before allocation, 2,065,301,504 during
allocation, and 720,896 after fixture/pool drainage with four PSOs retained. There
is no release-to-baseline tolerance or assertion. Per-command and per-fixture
pools remain active.

Nine-sample whole-kernel median milliseconds:

| Matched F16ACT Arm | Median | Min | Max |
| --- | ---: | ---: | ---: |
| Incumbent metal4.0@26.0 | 5.235042 | 5.228194 | 5.237208 |
| Incumbent metal4.1@27.0 | 5.234958 | 5.227208 | 5.247417 |
| 32x32 direct | 10.675500 | 10.659708 | 10.692097 |
| 32x32 retained output | 11.400333 | 11.387403 | 11.412472 |

Selected raw samples, milliseconds; all full-precision values remain in JSONL:

```text
incumbent26: 5.237208 5.230708 5.234681 5.228194 5.235042 5.235625 5.236208 5.237000 5.230875
direct32:   10.675500 10.673069 10.692097 10.668833 10.659708 10.678583 10.677986 10.680417 10.664458
retained32: 11.400042 11.412472 11.408597 11.387403 11.389444 11.411750 11.400333 11.411292 11.389458
```

Each sample is three serial whole-kernel dispatches divided by three. PSOs are
warmed, arm order rotates, and all 36 timed samples remain. Largest arm max/min
is 1.003866, descriptive only; no spread gate or sample exclusion is applied.
The compiler-tier controls are effectively equal, not an explanation for the
2.039x candidate regression. The input is a fully priced streaming bank, not the
tiny parity fixture or a resident arithmetic roofline.

At count32 useful and descriptor-padded work remain 289,910,292,480 FLOPs, since
M5120 is divisible by both row tile widths. The new geometry removes weight
duplication across slot partitions but doubles the logical activation reads per
64 output rows. Logical activation source reads are 4,529,848,320 incumbent bytes
versus 9,059,696,640 candidate bytes. Retained output stages 251,658,240 write
bytes per full dispatch; direct output removes that staging. These are logical
ledgers, not DRAM measurements. Occupancy/spill counters are unavailable; public
max-thread limits do not establish occupancy or the cause of the regression.

## Reproduction And Closure

Use `parity-32x32 iq3_xxs`, then `timing-32x32 iq3_xxs 32 9`, with explicit
absolute paths to both libraries. Commands and source are under
`scripts/bench/register-operands/README.md` and `tile32_f16.metalh`.

Private evidence archive label: `register-operands-private`; artifact names are
relative to that archive or its local mirror. See the
[source and evidence inventory](2026-10-24-v41-register-operands-inventory.md)
for the handoff without machine-specific locations.

`tile32-f16-report13.json`, raw stdout/stderr/observer events, immutable-history
verification and `frozen13-tile32-f16/identity.json` preserve the evidence. All
successful frozen identities are rechecked after measurement. Final SHA256s:

```text
probe:              ec969f51d667210480e90ab8fc25ca94bcafead47bad6f63eb0bf3c33e9023e2
private.metallib:    1bd26d4141abe4f30d8a8cf6b133980acd2440e5b1e2bc39c3c1e28960346fa0
baseline26.metallib: e7a37d6471b2cfd978decffeb8443c9ca218c3b54349e46089ded1db74948e83
source.tar:         a3d312c9940aeb3ebef1315445b3bcc24a17d1db03f08404618c8bad01b137d6
```

M5 Ultra / macOS27.0.1 / SDK27 / metal32023.921, private metal4.1@27 probe and
independent production-target 26.0 incumbent remain unchanged. Both library
paths and source identity are explicit. All builds/GPU jobs hold the production
lease; observers use exclusive files and 240/360-second deadlines. Correctness
uses validation; timing strips validation/injection/capture only in its child
environment. CPU contract/property tests, 31 Python tests, Ruff and diff checks
pass. No full model, product gating, main/CED edit, public push, pinning, purge,
sysctl mutation or SIGKILL occurs. This is the final supported-shape negative,
not an invitation to expand the losing register family further.
