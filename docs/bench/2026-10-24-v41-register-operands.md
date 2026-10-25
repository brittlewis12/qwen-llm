# V4.1 Register-Fed Quantized Matmul: Bounded Negative Result

Final hot-dtype check: [32x32x32 with matched F16ACT](2026-10-24-v41-register-operands-32x32-f16.md).
The original streaming controls here are F32-input entry points; product-target
compilation alone does not make them Maya F16ACT production-path measurements.

Continuation: [admission repair and one-SIMD experiments](2026-10-24-v41-register-operands-single-simd.md).
Historical build1-6 used a combined CPU-availability admission gate; the
continuation corrects that domain error, routing aliases outside these measured
bands, and fixture/command lifetimes. It preserves these original result files
and reports new API-limit, fallback and two-stage-transpose measurements
separately. Neither continuation candidate wins; production remains unchanged.

## Decision

Implemented and measured Track 2 of the outside-leverage map. Cooperative inputs
successfully remove threadgroup operand staging at the same quantized storage
and decoded half values, but **none of the six streaming variants wins**. Keep
the incumbent. No production entry point, loader, build target, default, artifact
or quant decoder changes; no native/full-model speedup or quality claim.

At count32, the best experimental variant (`wx16`) takes 11.836569 ms versus
6.218097 ms for IQ3_XXS, and 10.909958 ms versus 5.345667 ms for Maya IQ2_XXS.
That is 1.904x / 2.041x slower, respectively. The register-fed arithmetic controls
also do not beat the matched staged arithmetic control. This falsifies these
specific operand mappings, not every possible register-fed quantized GEMM.

## Source And Platform

- Measured implementation HEAD: `bd16b3e3099b754c775bae59128c03343005f6a4`.
- Owned branch: `perf/v41-register-operands`, starting at main `6c4e0cd7`.
- Hardware class: Apple M5 Ultra, 256 GiB physical RAM, macOS 27.0.1 (26A434).
- SDK 27.0, Apple metal 32023.921 (metalfe-32023.921.6).
- Read-only device recommended working set: 257,698,037,760 bytes (240 GiB).
- Private probe: `-O3 -ffast-math -std=metal4.1 -mmacosx-version-min=27.0`.
- Independent incumbent: unmodified production source, `-O3 -ffast-math
  -std=metal4.0 -mmacosx-version-min=26.0`, matching `build.rs` settings.

The actual SDK declarations and implementation, not an empty language probe,
were inspected. `get_left_input_cooperative_tensor` and its right-input twin
require `execution_simdgroup`; the SDK implementation has an explicit static
assertion prohibiting multi-SIMD input scope. The verified MLX NAX source packet
supplies the explicit K16 fragment mapping. Copies of the actual SDK headers and
that packet are retained with the experiment.

The experiment lives under `scripts/bench/register-operands`, outside the
production kernel directories. Both library paths are explicit arguments to the
standalone Objective-C++ executable. It rejects non-Apple10 devices and pre-27
runtime systems. The current product remains at 26.0, and M4/SDK26.2 builds
do not acquire unavailable calls. This conservative probe floor is not a claim
about the exact oldest OS supporting every method. Explicit per-lane mappings
remain compiler/device-specific and require requalification elsewhere.

## Implemented Controls

Every arm retains the native packed dtype/layout, F16 weight and activation
operand values, F32 accumulation, output staging, 128-thread group, 64-weight-row
x 32-slot macro tile, and the incumbent full grid. Output compaction, N8/N16 slot
tiles and larger tiles are not mixed into this comparison.

| Arm | Operand delivery and partition |
| --- | --- |
| Incumbent26 | Actual production body, threadgroup operands, K32, four-SIMD operation |
| Incumbent27 | Same body in the newer private library; compiler-tier control |
| ax | Cooperative inputs, K32; four SIMD groups each own 16 slots x 32 weight rows |
| wx | Cooperative inputs, K32; four SIMD groups each own 16 weight rows x 32 slots |
| ax16 / wx16 | Same output partitions, explicit verified per-lane K16 fragments, no hot-loop input-coordinate queries |
| axsh / wxsh | K16 producer lanes decode whole half16 fragments, then redistribute half values through SIMD shuffles |
| staged_resident | Decode/load one K32 panel, stage once, repeat matmul over those operands for all K steps |
| ax_resident / wx_resident | Load the same repeated-K32 arithmetic operands into cooperative registers once |

The initial K32 variants retain all three existing epilogues, including clamped
SwiGLU. K16 alternatives target down projection only. Their smaller K partition
is explicitly a separate numerical arm, not an exactness assumption. Every
decoder call is to the existing unmodified `quant_tiles.h` function; the shuffle
arm changes producer/consumer ownership, not decoder arithmetic. It was tried
after the direct and explicit-fragment mappings lost, rather than stopping at
the first negative result or increasing the tile.

## Bounded Parity

Final debug-layer runs: `parity6-iq3_xxs` and `parity6-iq2_xxs`.

- 203 output comparisons per dtype, including the exact non-auditing timed PSOs
  and their separate operand-auditing twins; zero observed output bit mismatches
  against Incumbent26, including the Incumbent27 control.
- 12,091,392 cooperative half values checked per dtype against the canonical
  dequant16 bank and half-rounded activations; zero bit mismatches.
- Counts 1/2/8/16/31/32/33, eight experts with distinct packed bytes, M70 tails,
  K512 grouped tests, grouped count33/K2304, actual dense32/dense64 K2304
  incumbent references, all original epilogues, and a clamped `b_div=6` fixture.
- Empty, negative-count and above-max-count experts, invalid negative and
  out-of-range IDs, untouched slots, leading/trailing guards and every untouched
  output byte retain the original seed bits. All command statuses complete.
- Resident controls have a separate repeated-first-K32 CPU reference.
- Maximum explicit buffer/CPU-copy price is 7,044,542 bytes for IQ3 and 6,883,262
  for IQ2, checked before allocation, under the enforced 128 MiB fixture cap.

The independent CPU accumulation uses canonical decoded **half** weights and
half-rounded activation values. It applies the existing grouped tests' numerical
thresholds: cosine > .999 and max-absolute error <= .01*reference-max + .0001;
the tensor comparison bound remains .002*baseline-max + .00001. Minimum observed
cosines are 0.999999999999889 / 0.9999999999998925. This is not an independent
decoder reimplementation or a rerun of the full native F32-dequant CPU oracle.
Observed fixture bit equality is not promoted to a universal frozen-exact proof.
No maintained tests or tolerances are changed.

## Streaming Method And Price

Each final timing process normally allocates the full native **384 x 5120 x
2304** compressed expert bank. IQ3 holds 1,734,082,560 packed weight bytes; IQ2
holds 1,167,851,520. Neither is represented by the tiny parity bank. The maximum
explicit full-bank fixture price is 2,170,815,128 / 1,604,584,088 bytes. These are
payload and explicit CPU-storage prices, not claimed total driver/process RSS.

Before allocation the harness logs the price, lower-bound host availability and
read-only device recommendation, then uses ordinary shared-buffer allocation and
admission. No residency sets, pinning, mlock, prewire, purge, forced cache policy,
sysctl mutation, invented reserves or memory-pressure percentage floors occur.

Counts are uniform synthetic fixtures with static permuted unique IDs, not
captured model routes. The full `(64, 80, 384)` N2048 grid remains identical
between arms even for low counts; these are controlled underpopulation bands,
not inferred native routing distributions. Real route capture remains a parent
follow-up only if a useful leaf emerges.

Each process warms all eleven PSOs, then rotates their order through nine
samples each. One sample is the command GPU envelope divided by three serial
whole-kernel dispatches, not an inner-matmul timer or a shader-active counter.
Raw host envelopes and all warmups remain in JSONL. Output is stored to external
device buffers and consumed; native packed bytes/activations are runtime data,
not constants that a compiler can erase.

There is no packing or predecode step on the streaming candidate path. Synthetic
fixture initialization is separately timed and logged for every process, as is
the parity-only dequant bank. Resident arithmetic repeats the first K32 panel
and includes initial loads/decode and output costs; it is not a streaming or
zero-DRAM roofline claim.

## Final Matched Timings

Whole-kernel median milliseconds, nine samples per cell, final frozen build 6.

| Arm, Count32 | IQ3_XXS | IQ2_XXS |
| --- | ---: | ---: |
| Incumbent26 | 6.218097 | 5.345667 |
| Incumbent27 | 6.205653 | 5.340208 |
| ax K32 | 20.920833 | 20.196750 |
| wx K32 | 19.081708 | 18.641819 |
| ax16 | 15.295083 | 13.972625 |
| wx16 | 11.836569 | 10.909958 |
| axsh | 18.301889 | 17.865250 |
| wxsh | 14.679708 | 14.096556 |
| staged_resident | 3.328056 | 3.315389 |
| ax_resident | 3.428986 | 3.417958 |
| wx_resident | 3.436736 | 3.422444 |

The compiler-tier control differs by about 0.20% / 0.10% at count32, not enough
to explain the cooperative regressions. The best experimental streaming arm is
`wx16` in every count band:

| Count | Padded/Useful Slots | IQ3 Incumbent26 ms | IQ3 wx16 ms | IQ2 Incumbent26 ms | IQ2 wx16 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 4.399597 | 11.041819 | 4.181125 | 10.662681 |
| 2 | 16 | 4.461833 | 11.080000 | 4.200208 | 10.702944 |
| 8 | 4 | 4.754375 | 11.132667 | 4.265125 | 10.738431 |
| 16 | 2 | 5.150750 | 11.193194 | 4.391083 | 10.791000 |
| 31 | 32/31 | 6.141542 | 11.771486 | 5.266458 | 10.900569 |
| 32 | 1 | 6.218097 | 11.836569 | 5.345667 | 10.909958 |
| 33 | 64/33 | 9.087750 | 21.414222 | 8.063361 | 20.612792 |

Selected raw count32 samples, milliseconds (full precision and every other arm
are retained in the raw JSONL and `final-report.json`):

```text
IQ3 incumbent26: 6.214875 6.216486 6.222528 6.217833 6.224778 6.227014 6.218097 6.215764 6.221889
IQ3 wx16:      11.836569 11.832750 11.837375 11.846917 11.848792 11.852708 11.835931 11.825653 11.832500
IQ2 incumbent26: 5.349708 5.337028 5.354153 5.346875 5.338236 5.347611 5.341125 5.341292 5.345667
IQ2 wx16:      10.908167 10.913653 10.895417 10.914361 10.954361 10.889903 10.924500 10.905292 10.909958
```

All 1,386 final timed samples remain, with no spread veto or exclusion. The
largest max/min ratio of any final nine-sample arm/band is 1.009446. At count32,
Incumbent26 ranges are 6.214875-6.227014 ms / 5.337028-5.354153 ms; `wx16` ranges
are 11.825653-11.852708 ms / 10.889903-10.954361 ms. Earlier build-2/3/4 raw
samples, including the initial negatives, remain beside the final cohort.

## Work And Traffic Ledger

At count32, useful and descriptor-padded work both equal 289,910,292,480 FLOPs.
At other bands padded work follows the table, regardless of the useful-rate
denominator. These are descriptor operation counts, not hardware performance
counter measurements. The count32 incumbents reach about 46.6 / 54.2 useful
TFLOP/s in this kernel envelope, not a whole-model rate.

For each K32 macro step the incumbent stages 4,096 weight + 2,048 activation
bytes. Over count32 this is 13,589,544,960 logical threadgroup operand-write
bytes, removed by the cooperative arms. Output staging remains 251,658,240
write bytes per complete count32 dispatch, plus its reads. The minimum logical
F32 activation source reads grow from 9,059,696,640 baseline bytes to twice that
for ax ownership and four times that for wx ownership. Quantized weight reuse
and per-slice decode duplication also differ by mapping.

These are logical source/staging ledgers, **not inferred DRAM traffic**. No DRAM,
occupancy or spill counters were obtained. The public PSO API reports execution
width 32, maximum thread limit 1024 and zero static threadgroup bytes; the host
sets 8192 dynamic bytes for every arm. Those limits are not measured occupancy.
`metal-objdump --reflection` exposes no resource statistics for the private
library; its retained output is empty apart from the section header. Do not
attribute the regression to spills or any particular hardware limiter without
additional evidence.

## Reproduction And Handoff

Source and commands: `scripts/bench/register-operands/README.md`. CPU checks:
`uv run scripts/bench/register-operands/test_report.py` and the unchanged
`uv run scripts/bench/test_pilot.py` (1 + 30 tests pass), plus Ruff and
`git diff --check`.

Private evidence archive label: `register-operands-private`. File names below
are relative to that archive or its local mirror. See the
[source and evidence inventory](2026-10-24-v41-register-operands-inventory.md)
for the complete handoff without machine-specific locations.

`frozen6/identity.json` binds the executable, both libraries, AIR and complete
source archive to the measured HEAD, compiler and SDK. All frozen successful
build hashes were rechecked after measurement. Final SHA256 identities:

```text
probe:              dada89307a13451949ff3cc5ab3a691f3dd61ad4878cbbae0b19b594177cff0d
private.metallib:    be4985a8ad6e5d1ef73cb3459886a6f1c9542724ba4b2a86dae1675089e54dbe
baseline26.metallib: e7a37d6471b2cfd978decffeb8443c9ca218c3b54349e46089ded1db74948e83
source.tar:         fcf4cf34f81005edb8225ed235e019ddabada0cdb7264637815baf885c15c04d
```

All builds and GPU jobs use the production lease, with no double acquisition or
shared Cargo output. The established pilot observer retains exclusive output
files, explicit 240/360-second deadlines, ordinary read-only host-health samples
and owned-child INT->TERM cancellation. Correctness uses `MTL_DEBUG_LAYER=1`;
timing strips validation, wrapper injection and capture only from the launched
child environment. No cancellation, critical-pressure stop, retry loop, public
push or other worktree edit occurs. A first compiler declaration error was
investigated and fixed in a new committed build; its original log remains.

There is no winning leaf to integrate or promote, so full-model native timing,
quality and real-route capture are not claimed. IQ2_S/IQ3_S expansion is not
warranted by this negative result. The next specific operand-path hypothesis is
a producer-to-fragment register transpose that avoids the full-vector broadcasts
used by `*sh`, ideally retaining K32. That is unimplemented and unmeasured, not
an advertised improvement; a larger tile or compaction would be a different
experiment. Preserve the incumbent and keep this probe as bounded evidence.
