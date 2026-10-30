# Per-SIMDgroup Prefill QK

Prototype `3a0d4a11`, based on `9a22dd11`, was initially handed off with only
source/offline checks and no production dispatch change. The parent then built
and measured that prototype on M5. The integration now enables it by default
on this candidate branch for a model trial; model verification and timing are
pending. Softmax, PV, normalization, and the incumbent Q-register kernel remain
unchanged. This is not a main-branch promotion or a model-speed claim.

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
merely to force bits. The measured microprobe is bitwise exact; this does not
establish equality for every model input or the integrated model chain.

## Measured Microprobe

Parent-reported M5 execution of source `3a0d4a11`, built locally with SDK 26.2,
Metal 4.0, opt-level 3 and thin LTO using the standard test profile. Frozen test
executable SHA256:
`0898fe028d9b211dcd6fbfc531f461b17ad1f6e5dec8e1f5d575bbf61ad32046`.

All 37,715,968 active scores and 33,554,432 outputs compare bitwise, as do
scratch, denominators, and status. Small/tail cases, bad IDs, and the head-0
NaN case are exact too. Review found no concrete correctness bug.

GPU microseconds per launch, 32 launches per command, continuous warmup,
three retained ABBA blocks and six retained samples per arm:

| Visibility | Scope | Qregister us | Per-SIMDgroup us | Speedup |
| --- | --- | ---: | ---: | ---: |
| Dense | SCORE | 977.584636 | 301.439453 | 3.243055x |
| Ragged | SCORE | 797.767580 | 251.672527 | 3.169864x |
| Dense | FULLSINK | 1932.088542 | 1239.611329 | 1.558624x |
| Ragged | FULLSINK | 1662.533203 | 1113.132163 | 1.493563x |

These are resident synthetic attention results, not native model prefill
results. Integrated model correctness and Full/CED performance remain pending.

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
repository infrastructure or run Metal. Those were the original source-only
checks; the subsequent parent-built prototype results are recorded above.
The integrated host change passes offline MSL compile/link, Rust formatting,
and the standard environment-reference generator's check and CPU self-tests.
The integrated Rust build and GPU/model verification remain with the parent.

## Trial Integration

`QWEN_V41_ATTN_SIMDGROUP_QK` is a standard default-on flag on this candidate
branch. It applies only inside the existing tensor64 route: H64/D512, the
selected tensor tier and existing logical query arity of at least 128, with an
available F16 Q mirror and both QHALF and QREGISTER enabled. CED descendants
retain their existing logical-arithmetic lineage, including physical short
tiles. Dispatch width is `ceil(maximum active row count / 64)`; height is the
current physical query span. Bindings, score stride, F32 K source types, scratch
allocations, and the memory ledger do not change.

- `QWEN_V41_ATTN_SIMDGROUP_QK=0` restores current Qregister for same-binary A/B.
- `QWEN_V41_ATTN_QREGISTER=0` bypasses the candidate and restores Q-half.
- `QWEN_V41_ATTN_QHALF=0` bypasses the candidate and restores F32-query tensor64.

Existing tensor-tier/geometry overrides retain their earlier fallback routes.
The raw kernel comparison remains an ignored probe, independent of the runtime
selector. Existing dispatch-census assertions now recognize the new score name.

## Parent Build Handoff

The existing build script discovers the new tensor `.metal` file automatically
and compiles it as Metal 4.0 for macOS 26.0. The new ignored test is classified
as timing in the existing acceptance inventory. The runtime selector and its
fallbacks are documented in the generated environment reference.

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
reports. The parent owns integrated builds and all GPU/model execution; the
candidate-branch default is for that trial, not a completed model promotion.
