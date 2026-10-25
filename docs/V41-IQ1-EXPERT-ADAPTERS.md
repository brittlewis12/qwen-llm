# Existing IQ1 Expert-Bank Adapters

Base: `2647dac9af57741ed791033123144205cd2a99b3`.
Implementation: `463e4e07`.

Qualification correction: independent review found that the original
malformed-count fixture was not diagnostic and that unassigned sentinels
inflated the pending M5 comparison threshold. The original `freeze-v7`
evidence is retained unchanged, but is superseded for qualification by
`docs/V41-IQ1-QUALIFICATION-REPAIR.md`. No M5 execution occurred with that
original bundle.

The subsequent parent M5 execution at `0e263131` passes all eight tensor
witnesses and the legacy ignored embedding witness. The independent native
row-capability expectation repair and corrected 8-versus-9 ignored census
are recorded in `docs/V41-IQ1-CAPABILITY-CENSUS.md`.

This adds missing expert-bank adapters for the existing IQ1_S and IQ1_M
codecs. It does not introduce a GSQ/RCO format stack, a new quantization
type, a model whitelist, or a runtime precision/memory policy. The loader
and binder retain their existing actual-native-capability queries; neither
is weakened. Auxiliary/control serialization is outside this change.

## Executable Coverage

| Family | IQ1_S and IQ1_M additions |
| --- | --- |
| Native F32 | GPU-indexed gate, up plus clamped SwiGLU, and down; scalar and row-batched host forms share one shader instantiation per dtype |
| Generic grouped half | Down/gate, UpSiluMul, fused unclamped SwiGLU, and UpClamped instantiations and storage layouts |
| Metal 4 tensor half | Existing F32/F16-activation down, UpSiluMul, UpClamped, and dense instantiations; existing F16-only wide expert probe |

The V4.1 clamped IQ1 grouped recipe is the existing serial Down/gate then
UpClamped recipe, not a newly invented fused architecture. Matching gate/up
types do not constrain the independently typed down bank. Tests cover the
released storage combinations IQ1_M/IQ1_M/IQ2_XXS (layer 7) and
IQ2_XS/IQ2_XS/IQ1_M (layer 17), using synthetic nonzero payloads, not model
weights. Existing V4.1 selection tests include native singleton execution,
grouped prefill, forced native rows, and rowwise fallback. No singleton
native result is relabeled as a COMPACT_PACKED result.

There is no new grouped F32-operand IQ1 family. Native F32 remains available;
generic/tensor paths retain their existing half-operand/F32-sum contract.
No Flash-Next dispatch code is changed.

## Decoder And Safety

`kernels/iq1_decode.metalh` contains the byte-for-byte original
`iq1_grid8`, `iq1_value`, and `iq1_dequant16` functions from `iq1.metal`.
Native adapters call decode16 directly. Generic/tensor adapters call the same
F32 decode16 and narrow only its final coefficients to half. The existing
dense entrypoint bodies, names, dispatch choices, and arithmetic remain
unchanged. The private freeze manifest verifies both the extracted decoder
and the complete dense-entrypoint suffix against the base revision.

IQ1_S uses 50 bytes per 256 values; IQ1_M uses 56 bytes with its half scale
distributed across four scale-word high nibbles. Loads remain byte/ushort
based: no 4-, 8-, or 16-byte-aligned weight-row assumption is introduced.
Grouped fixtures use nonzero bank offsets 18 and 34; K=256 IQ1_S rows step
through 50-byte misalignment. Native banks retain their existing 16-byte
binding-alignment requirement while compressed rows/blocks need only the
canonical 2-byte loads.

The existing native template retains status-before-ID-before-bank access,
invalid-ID/failed-route zeroing, compressed strides, F32 accumulation, and
upper-only gate/two-sided up clamping. All existing partition, reduction,
epilogue, and route-weight ordering remains intact. No CPU ID readback,
full-bank dequantization, gathered weight materialization, private weight
shadow, new scratch, or runtime allocation is added. Native and clamped
grouped refusal tests check atomic preflight, aliases, offsets/extents,
overflow, read-only outputs, and malformed bindings without dispatch.

The new header has an explicit build-script watch in addition to the kernel
directory watch. Compiled-source evidence includes its hash. The tensor
deployment tier remains `metal4.0@26.0`; no 27-only API is added.

## Original Local Evidence

Local hardware: Apple M4 Max. Compiler: Apple metal 32023.864
(`metalfe-32023.864`), SDK 26.2. The tensor metallib compiles, but this device
reports `GpuFamilyAbsent { metal4: false, apple10: false }`. That is not an
M5 pass.

All GPU fixtures run the frozen test executable directly with
`MTL_DEBUG_LAYER=1`, `QWEN_METAL_LEASE_WAIT=1`, and one test thread. They
self-acquire the production Metal lease; they are not double-wrapped.
Compilation and freezing occur under `scripts/bench/with-metal-lease`.
Bounded new fixtures admit a conservative 64 MiB shared-GPU plus 64 MiB CPU
envelope, with the normal 16 MiB reserve, before payload allocation. Shared
GPU demand uses MWSS headroom; only CPU growth uses CPU availability.

The focused IQ1 filter passes 30 tests with 8 explicitly ignored M5 tests.
It covers existing dense GEMV/scalar-GEMM/MMA/gather regressions, CPU-codec
differentials, native F32 and separate canonical-half references,
K=256/2304/5120, output-row tails, multiple/high expert IDs, offsets and
canaries, invalid IDs/statuses, negative/oversized grouped counts, untouched
invalid grouped slots, and actual dispatch census/PSO checks. Comparison
thresholds are unchanged; tensor comparisons now explicitly reject
nonfinite inputs rather than hiding NaNs in a max fold.

The separate generic-grouped all-format regression filter passes 51 tests.
Native all-format gate/up and down differentials, native mapping/refusal
checks, grouped/tensor source mapping, storage geometry, clamped recipes,
resident-type coverage, and the original V4.1 strategy regression also pass.
`cargo clippy -p qwen-llm --lib --tests` completes with warnings, including
the existing large `MetalError` API and deprecated atomic operations; this
is not a warning-free repository claim. Task-file rustfmt and ASCII checks
are separate from the baseline workspace formatting in `forward.rs`.

An explicit M4 invocation of the ignored tensor witness fails at its
required-M5 assertion, before tensor dispatch. Its nonzero result is retained
as unavailable-hardware evidence, not counted as tensor correctness.

Foreign-device tests retain their conditional second-device checks. A
single-device M4 run does not prove an actual cross-device binding failure.
Synthetic fixture success is neither full-model held-text quality evidence
nor a GPU performance win. Downloads, model runs, and M5 qualification belong
to the parent workflow. Frozen Maya/P0P4 experiments and oracles are untouched.

## Frozen Identity

Original private evidence bundle: `freeze-v7` (binary, three metallibs, source
manifest, and per-filter logs). SHA-256:

| Artifact | SHA-256 |
| --- | --- |
| Test executable | `584a5e1e81df389c896dca96b6d471901f114c3a6aa04f58e48e75fa97b67b40` |
| Product metallib | `d1d92b6dfa9a1ae9d9015ff02064ad9d1242217eb15c7fd832bd496721d5a162` |
| Tensor metallib | `16c47101ae5432bca60f3cd31aafdce9a87b78478095516d9fd0d45593858f8b` |
| Research metallib | `00506cb04df3d957d932d029abd35ea1ca8280be2862e82402a33cfb5caf4189` |
| Shared IQ1 header | `5178031dd7c6656c71695348741d68e22f0d9d33d7d141faa3cff8ad1bb0dcea` |
| Original extracted decoder text | `cd630e34e91770de928b1720da9ff325dc3ac3083ad418a4dba903d9a7782e1c` |

The private manifest contains hashes for every changed/new source used to
build the binary. `iq1_frozen_compiled_identity` independently prints the
embedded shader/header, host-adapter, compiler/SDK, and metallib hashes.
The parent must add the new header/test-fixture paths to its separate frozen
CDFwd source inventories when joining; those harnesses are not edited here.

## M5 Handoff

Build the lib-test executable and freeze it with its metallibs while holding
the production build lease. Use the cargo-reported executable path; do not
reuse an unrelated worktree's binary. Run the frozen binary directly:

```sh
MTL_DEBUG_LAYER=1 QWEN_METAL_LEASE_WAIT=1 "$FROZEN" \
  iq1 --skip metal_forward:: --nocapture --test-threads=1

MTL_DEBUG_LAYER=1 QWEN_METAL_LEASE_WAIT=1 "$FROZEN" \
  iq1 --ignored --nocapture --test-threads=1
```

The unskipped second filter runs nine tests: eight M5 tensor witnesses and
one legacy embedding loader/gather witness. Add `--skip metal_forward::` to
select only the eight tensor witnesses. Every IQ1 tensor execution test
requires an available tensor tier instead of silently skipping on M4. The
two `iq1_m5_` tests exercise all new tensor PSOs, F32/F16 activation storage,
canonical-half CPU references, generic/tensor agreement, the F16-only wide
form, production K values, tails, malformed routes, and mixed down types.
The other six preserve the existing all-format grouped/dense comparison
fixtures. The existing `wide_activation_type` M5 filters remain applicable
and unchanged. These local results remain distinct from the later parent
M5 execution recorded in `docs/V41-IQ1-CAPABILITY-CENSUS.md`.
