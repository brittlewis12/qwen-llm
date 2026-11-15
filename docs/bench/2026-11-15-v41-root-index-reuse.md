# V4.1 bitwise root index query reuse: gate PASS, default-on

Four-query key reuse passes the frozen bitwise, isolated and native gates.
`QWEN_V41_ROOT_INDEX_REUSE=0` restores the incumbent. The production change
only dispatches from the existing Half tensor branch, on Apple10/M5, H32/D128,
top-512, with sufficient existing selection-pair scratch. F32 MID, short-index
and compact/radix paths are unchanged. No NI or serve campaign was required.

## Arithmetic, dataflow and memory

The incumbent is `kernels/tensor/deepseek_v41_attention_tensor.metal:540`:
one threadgroup per query, 32 heads x 64 keys x K128 Half tensor multiply,
F32 destination, then the complete ascending 32-head F32 ReLU/weighted sum.
The candidate in `kernels/tensor/deepseek_v41_index_reuse.metal` preserves
that same operation, operand casts, K grouping, scale and entire epilogue.
It stages each 64-key panel once for four queries, running the unchanged
32-head multiply separately for each query. Four independent key ranges
restore threadgroup parallelism; each (query,key) still executes once with
the incumbent arithmetic. Shared memory remains 32 KiB per threadgroup.

A serial dispatch casts the microbatch queries to Half once. It temporarily
uses the existing I32 selection-pair slab as a Half view, up to 4 MiB for
512 queries. Scoring finishes before production selection overwrites pairs.
The following microbatch reuses the same physical score/pair scratch. No
allocation, retained cache or session-memory increase. Missing/short scratch,
unsupported hardware/geometry and disabled flags fall back before encoding.
Offset, extent, alias and scalar-overflow checks precede dispatch.

The key-reuse mechanism follows this tree's landed GLM implementation,
`kernels/glm_index_reuse.metal` (main `01d80aac2`), whose upstream mechanism
and Apache-2.0 attribution are documented in
[the GLM record](2026-11-15-glm-indexer-bitwise.md). This shader copies only
our incumbent V4.1 arithmetic, with attribution in its comment; no external
shader code is copied. No upstream license-bearing source was integrated.

## Frozen rule and real-capture witness

Frozen before GPU data in `{runs}/recon/2026-11-15-v41-root-index-reuse/FROZEN.md`:
bitwise scores and production selections first; score AND score+selection
GPU geomean >=1.25x; native geomean >=0.98x for every cell with bitwise
logits/causal payload. Symmetric warmups then three AB/BA/AB pairs, one
`iter` binary per packet, nice 0. A=rollback/incumbent, B=query reuse.
No samples or failed controls were dropped, and the rule was not changed.

Real source-2 inputs are from the final 4,096-row encoder chunk of a
32,768-token cold CED root (start=28,672, ratio=2), stock GSQ-RCO IQ2_XS
with IQ4_NL Engram and the uncycled Metal-source corpus. All **4,096 captured
query rows / 67,108,864 score words** compare bitwise, including initialized
unwritten tails. All ordered top-512 IDs and status agree: **zero score or
selection mismatches** for groups 2, 4, 8 and 16. The native witness below
covers the whole root and all model layers, rather than just this capture.
Functional score copies are outside timing; timed chains include the query
conversion, all eight 512-row microbatches, scratch reuse and selection.

| Final 4K chunk at 32K | A GPU ms | B GPU ms | A/B pairs | Geomean |
| --- | ---: | ---: | --- | ---: |
| Score (including query conversion) | 39.01862 | 23.57419 | 1.663820 / 1.657321 / 1.644344 | **1.655142x** |
| Score + production selection | 48.67017 | 32.95422 | 1.482018 / 1.481769 / 1.466970 | **1.476902x** |

Scoring saves 15.444 ms per captured layer; the three encoder-index layers
would save about 46.33 ms in this final 4K chunk. This is a projection, not
a replacement for the whole-root measurement. Group 4 was fastest. Other
split-range groups 2/8/16 give score+selection 1.455/1.329/1.396x.

The earlier non-bitwise oMLX mechanism control was 6.073x (38.67 -> 6.30 ms).
The new exact scorer retains about 47% of that control's absolute scoring
savings. This is non-contemporaneous and does not isolate changed arithmetic:
the upstream comparison also changed tensor shape, query/key conversion,
dispatch granularity and scratch layout. No claim that the remaining gap is
entirely arithmetic follows from this result.

Rejected controls remain in the raw record and Git history. Packing several
queries' head subsets into a 32-row tensor changed 39,171,291 score words
and 70 selected IDs for each tested group; rejected before timing. Full-query
reuse without split key ranges was bitwise but its best score+selection was
1.224x (scoring 1.288x), below the frozen chain gate. The retained shader
contains only the winning full-query/split-range implementation.

## Boundaries and native cells

**704 boundary executions pass** (including repeated Q3 controls): 4/16-byte
nonzero offsets with guards, both ratios 1/2, every prefix residue mod 4,
all-selected/top-k and 2K/4K/hierarchical selection frontiers, Q1/3/4/5/31/32/33,
real inputs and exact-zero ties. Score bits, ordered IDs and status match;
IDs remain causal and deterministic under ties, and guards are unchanged.
The real 4K replay exercises first>0 and consecutive scratch reuse.

Native: one stock model load, capacity 98,304; cold CED with 4K slices and
empty session state. Both arms warm the same token range before timings;
these are not OS-page-cache-purged roots. Snapshot capture/hash and the
follow-on teacher-forced decode are outside the measured root interval.
The 80K append uses the same incumbent-built prefix snapshot in both arms;
restores and snapshots are outside timing. Each cell has its own symmetric
warmup and three interleaved pairs on the same binary.

| Native cell | A wall ms | B wall ms | A/B pairs | Geomean |
| --- | ---: | ---: | --- | ---: |
| 32K cold CED | 12264.524 | 12115.933 | 1.008316 / 1.017575 / 1.010924 | **1.012264x** |
| 4K cold CED | 1925.920 | 1925.999 | 1.001122 / 1.005468 / 0.993324 | **0.999959x** |
| 8K cold CED | 3332.562 | 3332.295 | 1.005460 / 0.989669 / 1.005194 | **1.000080x** |
| 1K append @80K (inactive) | 955.728 | 960.414 | 0.984137 / 1.002373 / 0.998946 | **0.995121x** |

Last-row logits and complete causal tensor snapshot digests are bitwise in
all 12 pairs. All cold-root follow-on teacher-forced decode logits are also
bitwise. Candidate Half dispatch counts are 186/33/57 for 32K/4K/8K, and
zero with rollback. **Both append arms have zero Half-reuse dispatches**:
its F32 path is unchanged, so its 0.995x control is not a kernel speed claim.
The root's 148.59 ms saving is measured; 4K/8K are parity controls.

## Provenance, conditions and verification

Base main `01d80aac2`. Isolation source `a438d36ae`; native source `f37a65d65`.
The winning kernel function is unchanged between these packets; the native
build removes the rejected shader, enables the default and regenerates ENV.
The final shader is source-identical to the native packet, SHA256
`b07d6d0a626a0e038d849573c64b12797e541657e15757f82ab604fb600439b7`.
A later CPU-tested guard rejects ratios that cannot fit the shader uint;
it does not change any admitted model call or shader arithmetic.

Native binary SHA256:
`a5cc81bbd9c599702510ad743cbd2e7292b14f89c57b765c2eaad348f6c34f1a`.
Native tensor metallib:
`1b9c71d28dcd9c198a4e1158eb6cffe490ba7d21b47119e2cd13541a09858b04`.
Classic metallib is unchanged from the incumbent:
`9974659c68f4c82334e1807eb8942077dae0b8d30be728ee562445972fb5b9ed`.
Both complete hash manifests are in raw `split-isolation/pins.json` and
`pins.json`; the differing tensor libraries retain the same winning function.

Plain Cargo in worktrees, nice 10 builds, no target-directory overrides.
Python drivers inherit the lease FD with `pass_fds`; GPU probes run at nice 0
and the driver explicitly checks that priority. Every build/model load is
inside a <=15-minute FIFO hold. Isolation hold 98.51 s; final boundary/native
hold 341.92 s. Earlier rejected/control holds 152.52/128.54 s; an executable
pinning failure used 28.14 s and ran no model or GPU packet. The driver was
changed to pin Cargo's reported artifact before mbx collection. All records
are retained; no invalid native result was substituted.

Thermal <=1 for 60 s before packets, allow 2/abort 3. Isolation and boundaries
stay at 0; native ranges 0-2, with no abort. Measured one-minute host load:
isolation 5.39-5.60, boundaries 3.92-4.00, native 3.29-6.54. Preload available
memory 251.45 GB; no other model process was resident. Ghost was untouched.

Full library test profile: **2,509 passed, 0 failed**, 678 pre-existing ignores;
no new ignored tests. CPU control test passes. Contracts pass at native source;
final clean-tree contracts/fmt/ENV verification and cleanup are retained in
`verification.json`. No push or merge. The gate is closed; pins and bundles
are deleted and both worktrees removed, leaving the committed branch.

Raw records: `{runs}/recon/2026-11-15-v41-root-index-reuse/`, including all
pairs, failed controls, boundaries, thermal/load traces, hashes, driver,
`summary.json`, verification and cleanup manifests. Original real operands
remain in the prior discriminator's `capture3/inputs/root-28672/` directory.
