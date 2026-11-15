# GLM bitwise micro64 indexer: gate PASS, default-on

The query-reuse scorer passes the frozen bitwise and >=1.25x chain gate at
both depths. Native 2K chunks improve 1.022x / 1.060x, with identical last-row
logits and persistent state in every pair. Default-on on M5 Fast/Half packed
prefill; `QWEN_GLM53_PREFILL_INDEX_REUSE=0` restores the incumbent. Decode,
Exact/DenseF32/F32, unsupported hardware and short microbatches stay incumbent.
This branch contains no NI tensor scorer and requires no NI campaign.

## Arithmetic and dataflow

The incumbent is `kernels/deepseek_v4.metal:3117-3222`: one query, eight keys,
32 heads, four SIMD groups. Each dot uses F16 operands and a zeroed F32 8x8
accumulator, with 16 ordered K=8 SIMD matrix multiply-accumulates. The scalar
epilogue visits heads 0 through 31, applying ReLU and F32 head weights in
that order.

`kernels/glm_index_reuse.metal` puts eight queries in the matrix rows and
eight keys in the columns. Each SIMD group visits eight heads. It performs
the same K=8 steps and ordered head epilogue, with unchanged operand and
accumulator precision. The transposed key tile is loaded once for eight
queries. A 264-float stride separates shared head-dot rows to avoid bank
conflicts. Threadgroup memory is 2,048 + 8,448 = 10,496 bytes; session scratch
is unchanged. Union visibility can load a key for another query, but each
visible dot sees the same bits and each masked score is still -Inf.
Full query groups use the new kernel; a remainder uses the incumbent.
Production dispatch is limited to Fast/Half H32/D128 microbatches of 32-64
rows on Apple10/M5. Shape, offset, disjointness and hardware checks precede
encoding, with an incumbent fallback for unsupported geometry/hardware.

Mechanism reference, checked Apache-2.0:
[oMLX indexer_nax.py:54-195](https://github.com/jundot/omlx/blob/624cff0e23519ba46a4d5ed2802c2c236ad1a9ac/omlx/patches/glm_moe_dsa/indexer_nax.py#L54),
revision `624cff0e23519ba46a4d5ed2802c2c236ad1a9ac`. This implementation uses
our classic SIMD arithmetic; no upstream shader code is copied. Attribution
is in the shader and THIRD-PARTY-NOTICES.

## Frozen rule and bitwise witness

Frozen before measurement: all 2,048 real captured queries at each depth
must have bitwise-identical scores and selections; then symmetric warmup
and three AB/BA/AB pairs, ten chains per arm, requiring GPU geomean A/B
>=1.25 at BOTH depths. A=incumbent, B=query reuse. Nice 0, one `iter` binary,
one physical production micro64 score/ID/count scratch reused at offset zero
for every microbatch and both arms. Status remains indexed by input row.
Functional GPU copies are outside timing. A functional-only packet verifies
both depths before timing starts.

Real baseline captures: MLA index 0, stock GLM-5.3 Flash UD-IQ3_XXS,
uncycled Metal-source corpus, 49,152/131,072 prefix tokens followed by 2,048
rows. Q `[128,32,2048]` F16; scaled weights `[32,2048]` F32; keys
`[128,33792]` F16; visibility `[2048]` I32. Each depth compares 69,206,016
score words, including identical initialized/unwritten capacity tails:
**zero mismatches**. Visible score counts are 25,689,600 / 67,632,640.
Ordered top-512 IDs, counts/status, expanded IDs/counts and padding are all
identical. The previous near-tie rows 555/1689 now select incumbent IDs.
This capture witness covers one layer/corpus; the native witness below
exercises every model layer and compares full persistent state.

## Score -> select -> expand

| Prefix + rows | A GPU ms | B GPU ms | A/B pairs | Geomean | Saved ms/layer |
| --- | ---: | ---: | --- | ---: | ---: |
| 48K + 2K | 15.88365 | 11.63483 | 1.365099 / 1.365170 / 1.365272 | **1.365181x** | 4.24882 |
| 128K + 2K | 41.77291 | 30.46288 | 1.371344 / 1.370935 / 1.371539 | **1.371273x** | 11.31003 |

Both cells pass without a threshold change. Across 11 MLA blocks the
isolated savings forecast 46.74 / 124.41 ms per 2K chunk. For context only,
the earlier non-bitwise tensor micro64 packet saved 5.08 / 17.41 ms per layer;
this variant retains about 84% / 65% of those savings. That is a
non-contemporaneous comparison, not a causal separation of tensor shape,
intrinsic implementation and changed K reduction. The bitwise variant
clears the requested gate, so no additional tensor-control lease was used.

## Boundaries and native cells

All **96 operator boundary cases** pass with bitwise visible scores:
4/16-byte nonzero offsets, guards, every prefix residue mod 4, Q31/32/33,
first>0, consecutive chunks reusing scratch, real operands and exact-zero
ties. IDs are ordered; counts/status agree; incomplete/future pools are
excluded; stable ties select lowest IDs; expansion includes only complete
selected pools and visible pending rows, with -1 padding. Q31 falls back;
Q33 uses 32 new-kernel rows and one incumbent row.

All **24 actual mixed dense/sparse frontier calls** pass. Prefixes
2032-2035, first sparse offsets 19-16, Q31/32/33, both arms. Dispatch traces
cover all 11 MLA blocks, with expected fallback/tail behavior. Each candidate
call has bitwise last-row logits and persistent state versus its incumbent
call.

Native: one stock model load, capacity 135168, same captured tokens. Each
arm restores the same incumbent-built CPU prefix snapshot outside timing;
one warmup per arm, then AB/BA/AB on one binary at nice 0. These are 2K
chunks at depth, not full cold-root or served measurements. In all six pairs,
last-row logits and the entire persistent snapshot are bitwise: KDA conv/state,
MLA latent rows, completed pooled keys and pending key/gate rings.

| Prefix + rows | A wall ms | B wall ms | A/B pairs | Geomean | Saved ms |
| --- | ---: | ---: | --- | ---: | ---: |
| 48K + 2K | 1963.434 | 1920.822 | 1.017631 / 1.024961 / 1.023977 | **1.022185x** | 42.613 |
| 128K + 2K | 2245.941 | 2119.650 | 1.056056 / 1.059464 / 1.063235 | **1.059581x** | 126.291 |

All native pairs are positive. No serve campaign was requested or run.

## Provenance, conditions and verification

Base main `6c5776db4`; isolation source `0ffe606e16914ddf01b0c035197794e46476fe16`;
native source `06a3dd34354d802c84f05c112912a6de4a8ddcf5`. Plain Cargo `iter`
builds in worktrees, nice 10; measurements nice 0, explicitly checked with
`ps` during native execution. The two binaries differ only in probe-side
frontier/per-pair assertions and formatting. All three metallibs are identical.
The final kernel source is identical to both measured packets; product changes
after measurement are the default flip and documentation only. The probe's
functional-only control was later made an explicit CLI mode, with no scoring
or timing changes.

- Isolation executable SHA256: `7b463115517a24fffd46574afcdafe0cc8c0c9c1a6ca7a1df03857005cef2bc5`.
- Native executable SHA256: `b50cbea4c1463d09af8db0a167038a0b3c0c6e154a9349c927338641361617b7`.
- Shared main metallib: `9974659c68f4c82334e1807eb8942077dae0b8d30be728ee562445972fb5b9ed`.
- Shared tensor metallib: `43ae770545fa60f1f51d2bb3220a93c0aaf3bfc13d382a99ea2345920eaf184c`.
- Kernel source SHA256: `a0370bbc62a71243285bfb4f90df1eac7957bf11a3395167b4c2aea59fa192f5`.

Accepted lease holds: 122.63 s (build, settle, both-depth functional/timing)
and 261.99 s (probe rebuild, settle, boundaries, native). Each settled <=1
for 60 s, nice 0 measurements. Thermal 0 throughout isolation, 0-2 during
native, never 3. One-minute load during replay 7.76-7.78 and native
11.05-13.21; expected background work was untouched. After acquiring the
second FIFO hold, the guard found no other model resident and 249.78 GB of
free+inactive+speculative memory before loading. An initial dispatch failed
before any build because the transfer had not created its worktree; it was
corrected, and that short hold released. No measurement was accepted from it.

CPU shape/overflow/control and policy tests cover decode/tails, precision
fallback, nested and panic-safe rollback scopes. No new ignored tests.
ENV regenerated. Final contracts/fmt and cleanup are recorded in the external
verification manifest. Both gates are closed: stage pins/bundles and M5
worktrees were deleted; exact started-PID audit found none resident. The
unused detached tensor-control worktree was also removed without building it.
No pushes, merges or roadmap edits.

Raw packets, frozen rule, hashes, thermal/load/admission/process records,
measured source archive and verification:
`{runs}/recon/2026-11-15-glm-indexer-bitwise/` (M5 and local evidence).

Reproduce with `cargo build --profile iter -p qwen-llm --features
perf-ablations --example glm_index_reuse_probe`, then `functional INPUT_DIR`,
`replay INPUT_DIR`, `boundaries INPUT_DIR`, or `native INPUT_DIR`. M5 builds
and model work require the usual <=15m FIFO lease and inherited descriptor.
