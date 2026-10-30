# Test-Only Per-SIMDgroup QK

Prototype against `9a22dd11`; no production dispatch, default, softmax, PV,
normalization, or incumbent Q-register kernel changes. No GPU execution or
speed measurement is performed for this source handoff.

## Kernel

`score_simdgroup_qk` uses the sibling V4.1 attention kernel's organization,
not its unscaled-score convention. The existing row helpers preserve ring
wrapping, causal visibility, per-query picks, invalid-ID zero keys, and score
strides. Q uses the existing post-RoPE F16 mirror; raw/owner K stays F32 and
rounds to F16 while staging. No persistent K cache is introduced.

- Grid: `(ceil(K / 64), queries, 1)`, 128 threads, four independent
  `execution_simdgroup` MPP operations, each covering 16 heads by 64 keys.
- Depth: DK64, eight ascending-depth MMA calls for D512 (multiply then seven
  MACs), versus the incumbent's four D128 calls. F32 accumulation and the final
  `rsqrt(float(D))` score multiplication remain in QK.
- Shared memory: two K64/D64 F16 buffers, 8 KiB each. After the final barrier,
  the same 16 KiB holds the H64/K64 F32 scores for masked scalar stores. There
  is no shared Q buffer or extra score allocation.
- Explicit private state: eight half4 K-prefetch vectors per thread (64 bytes)
  plus 32 logical F32 accumulator elements per lane (128 bytes), before
  compiler-generated operand/address state. No 512-byte full-Q private cache.
  These are logical sizes, not measured physical register or spill counts.

DK64 preserves the 16 KiB pipelined premise but can change reduction rounding.
DK128 would retain four depth calls at the cost of 32 KiB of ping-pong K.
This prototype does not spend that extra memory or serialize the simdgroups
merely to force bits. Exactness remains a GPU question, not an offline claim.

## Probe

Full selector:

```text
deepseek_v41::forward::attention_gpu::tests::f32_pv_batch::simdgroup_qk::attention_simdgroup_qk_probe
```

The ignored sibling module compares only against the current
`score_tensor64_qregister`. FULLSINK uses the unchanged `softmax_rows`,
`pv_tensor64`, and `normalize_rows`; it does not replay parked PV variants.

Untimed checks poison scores, output, and unused value/probability slabs before
each arm, independently reset status, check every active score/output store,
and verify inactive scores and guards. After the complete chain, denominators,
all scratch slabs, output, and status are compared too. Every finite element
contributes to changed-bit counts, maximum absolute/relative/ULP error, mean
absolute error, and RMS error. Finite differences are reported, not hidden by
a tolerance or promoted automatically. Nonfinite classification, payload/sign,
and status must match; the distinctive poison payload must never survive an
active store. Relative error uses the baseline magnitude, floored at the F64
minimum positive normal, so near-zero relative errors can be large.

The production fixtures are dense and ragged K640/N512/H64/D512 (128 raw plus
up to 512 selected keys). Counts are derived from each fixture: 37,715,968
active scores and 33,554,432 outputs across the pair. Small K256/K512 and K640
tail cases, nonzero first-query offsets, invalid and populated-but-not-visible
IDs, and a query-1/head-0 NaN are checked without timing.

For each production fixture and SCORE/FULLSINK scope, two warm ABBA blocks
immediately precede three retained ABBA blocks. Every command contains 32
launches; operand buffers stay resident and unchanged between arms. No tensor
readback, poisoning, tensor allocation, or programmed cooldown occurs within
this sequence.
All warm and retained samples print absolute GPU timestamps and per-launch
GPU/wall time. All six retained samples per arm enter the median; there is no
best-case selection. Compilation and untimed numerical checks are excluded.

## Local Checks

Offline compile and link pass with the installed Metal compiler 32023.864 and
macOS SDK 26.2, using the product tensor-tier flags:

```sh
xcrun -sdk macosx metal -c -O3 -ffast-math -std=metal4.0 \
  -mmacosx-version-min=26.0 \
  kernels/tensor/deepseek_v41_attention_simdgroup_qk.metal \
  -o /tmp/v41-simdgroup-qk.air
xcrun -sdk macosx metallib /tmp/v41-simdgroup-qk.air \
  -o /tmp/v41-simdgroup-qk.metallib
```

The incumbent also compiles with those flags and links with the candidate into
one metallib successfully.

`rustfmt --edition 2024 --check` on the changed Rust files and `git diff
--check` pass. Minimal offline `uv run --offline --no-project --python 3.13
python` CPU checks pass for the existing inventory schema/classification and
the staging/store/visibility geometry: 15 small cases, both production cases,
586 tail query rows, and the derived totals above. A temporary `rustc
--edition=2024 --test` harness using the actual error accumulator passes five
CPU checks (exact/one-ULP values, signed zero/cross-zero values, missing store,
new nonfinite, and changed nonfinite). These temporary CPU checks do not add
repository infrastructure or run Metal. The full Rust test target is not built
here; optimized compilation and GPU numerical/timing results remain with the
parent.

## Parent Build Handoff

The existing build script discovers the new tensor `.metal` file automatically
and compiles it as Metal 4.0 for macOS 26.0. The new ignored test is classified
as timing in the existing acceptance inventory. No feature or environment flag
selects the candidate in production.

Build the optimized library test with the existing private Cargo cache, offline
and unleased:

```sh
cargo test --offline --locked --release -p qwen-llm --lib --no-run
```

Use the exact selector with `--ignored --exact --nocapture
--test-threads=1` only through the parent's ordinary queued GPU wrapper. Unset
`MTL_DEBUG_LAYER`, `MTL_SHADER_VALIDATION`, and `QWEN_REQUIRE_METAL_TESTS` for
this timing probe. The existing benchmark lease and cancellation guard are
reused, not replaced. Preserve the full output, including nonzero finite-error
reports; this candidate remains unpromoted regardless of microprobe speed.
