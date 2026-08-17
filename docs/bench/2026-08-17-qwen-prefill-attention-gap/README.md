# Qwen Long-Prefix Prefill Attention Gap

Date: 2026-08-17

Status: **CLOSED AS A STALE CATEGORY CLAIM** for the production CLI's eligible,
fully provisioned Qwen3.8 G6 prompt path. No product code changed.

Audited implementation commit:
`38714385c6b804be8e6d671dda70165e10889362`.

## Claim Under Review

The proposed experiment assumed that prompt prefill still executes attention as
one decode-shaped `encode_attn_decode_v4_f32` call per query token, repeatedly
streaming the prefix K/V cache. It therefore proposed the repository's first
Q-tiled prefill kernel: 32--128 query rows by K/V tiles with online softmax and
GQA reuse.

That premise is stale for the production Qwen3.8 G6 path. The repository already
selects a retained matrix implementation that shares K/V across both GQA
siblings and multiple token rows. This does not mean the proposed 32--128
token-row resident tile already exists: the retained kernel's inner width is 32
query/head columns, which is only about five token rows at G6. It does mean that
the work is a replacement/retile of an existing matrix path, not the first
batching of a sequential decode loop.

## Production Selection

`crates/qwen-llm/src/metal_dflash.rs` permits matrix attention by default for
head dimension 256 and the supported GQA topologies. Qwen3.8 G6 satisfies the
explicit `n_q == 24`, `n_kv == 4` branch:

- `prefill_attn_matrix_g6_may_use` admits auto/default mode unless explicitly
  disabled (`metal_dflash.rs:1520`).
- scratch planning enables G6 matrix storage for `24 / 4` heads
  (`metal_dflash.rs:3755`).
- runtime selection requires F16 K/V, head dimension 256, that same topology,
  and scratch coverage through the current prefix (`metal_dflash.rs:8704`).
- eligible execution enters the `use_matrix` branch, not the per-token fallback
  (`metal_dflash.rs:9217`).

Scratch coverage is a real selection condition, not an incidental check. A bare
library allocation defaults matrix capacity to its block size and can fall back
after that range. Absent an explicit max-position environment override, the
production CLI avoids that limitation: its legacy
allocator passes `max(prompt_tokens, chunk)` as `matrix_max_pos`, and its
admitted auto-prefill plan does the same (`crates/qwen-cli/src/main.rs:5554`,
`crates/qwen-cli/src/main.rs:5598`). The status above is intentionally scoped to
that fully provisioned path; it does not claim every library caller selects
matrix attention.

Ordinary prefill first computes packed Q/K/V, applies consecutive partial RoPE,
and scatters the chunk's K/V once. The matrix branch simultaneously appends V to
a scratch-owned transposed-V arena whose validity is invocation-scoped
(`metal_dflash.rs:8758`). It then invokes:

1. `encode_attn_matrix_kq_online_f32` for tiled KQ plus causal online softmax;
2. `encode_attn_matrix_kqv_norm_f32` for tiled probability-times-V and final
   normalization.

The query row's absolute position is carried as `chunk_start + row_base`, and
the online epilogue always computes each row's visible prefix from that base
(`metal_dflash.rs:9277`, `kernels/attn_matrix_online.metal:114`). The separate
`causal_skip` switch only avoids fully masked K/KQV tiles; it does not control
correctness masking.

## Existing Query And K/V Tiling

`kernels/attn_matrix_online.metal` defines 64-position by 32-local-column
simdgroup-matrix tiles. The global column extent is `n_rows * group`, and
`row = local_q / group`; it is not 32 token rows
(`kernels/attn_matrix_online.metal:104`,
`kernels/attn_matrix_online.metal:112`). At G6, one full inner tile covers 32
query/head columns: 5 1/3 row-equivalents touching six distinct token rows. K is
therefore shared beyond the six sibling heads of one token, but far short of a
32--128 token-row resident design. The transposed V representation supplies the
matching KQV matrix body.

The KQ epilogue stores F16 unnormalized probabilities plus a small F32 `(m, l)`
sidecar; KQV applies tile rescaling and normalization. This cuts score-tensor
traffic from 16 to 4 bytes per element relative to the older three-dispatch
sidecar.

The outer query dimension is partitionable for bounded scratch:

- `QWEN_PREFILL_ATTN_MATRIX_QUERY_CAP` and
  `PrefillScratchConfig.matrix_query_cap` bound live query rows;
- `attn_matrix_query_tiles` partitions an arbitrary prompt chunk; and
- each tile gets bounded score/softmax scratch and the correct absolute
  `base_pos` (`metal_dflash.rs:1546`, `metal_dflash.rs:1623`,
  `metal_dflash.rs:9243`).

The cap is not an inner resident tile and does not increase K/V reuse. It merely
sequences the same 32-local-column matrix kernel over bounded row ranges.
Production legacy allocation leaves it unset by default, although the
environment can supply a cap; production auto-prefill currently uses 1,024 rows
(`crates/qwen-cli/src/main.rs:5141`). Values 32--128 are legal explicit
configurations only when online matrix mode is enabled, not current defaults.

## Why The Per-Token Loop Still Exists

The ordinary attention body falls back per token only when no packed or matrix
implementation is eligible (`metal_dflash.rs:9595`). Eligible head-256 G4/G6/G8/
G16 shapes use V4; other shapes use the generic F16-KV kernel. A second nearby
loop constructs a V4 oracle only when the packed G8/G16 diagnostic flags request
it (`metal_dflash.rs:9489`). Other loops have distinct generic or packed
speculative-verification ownership.

These remain meaningful coverage and rollback paths. They do not establish that
the production G6 prompt path rereads K/V once per query.

## Prior One-Pass Falsification

The retained implementation is deliberately two-pass. A historical kernel
comment records a correct one-pass flash body at `0.80x` an earlier sidecar,
falling to `0.39x` at its Q=16 shape from a threadgroup-memory occupancy cliff
and `0.24x` with register-resident output due to spills
(`kernels/attn_matrix_online.metal:29`). It attributes the result to head-256
K/V re-streaming and records a do-not-reopen criterion.

Those comment-only numbers lack the hardware, command, exact shape packet, and a
direct comparison to today's online path. They are prior evidence, not a current
falsifier and not proof that a larger token-row tile cannot win. A future
one-pass replacement remains technically open, but it must compare against the
enabled online matrix path, preserve GQA and causal semantics, and beat both the
matrix body's K/V reuse and its score-sidecar cost at the target long-prefix
shape.

## Remaining Leverage

The close read exposes narrower, genuinely open questions:

- V_T validity starts at zero on every prefill invocation. A nonzero resumed
  prefix therefore rebuilds `0..n_pos` from canonical V on its first matrix
  chunk; because scatter already appended the current chunk to V_T, that chunk
  is written twice (`metal_dflash.rs:7568`, `metal_dflash.rs:8732`,
  `metal_dflash.rs:9225`);
- the default two-pass path still writes and rereads an F16 probability sidecar;
- unsupported topologies and memory-constrained paths retain sequential V4;
- packed MTP verification has separate query-sharing limits and must not be
  conflated with ordinary prompt prefill.

Any long-prefix campaign should therefore compare a new candidate against the
enabled matrix path, report attention-subphase time, and isolate restored-prefix
V_T rebuild where applicable. Comparing only against forced V4 would measure a
rollback path and overstate product leverage.

## Disposition

No duplicate implementation of the already-selected two-pass matrix category is
authorized by the reviewed claim. Architectural selection and kernel ownership
falsify only its novelty premise before GPU work. This closure does not claim
matrix attention is globally optimal, that every caller provisions it, or that
a genuinely larger token-row-resident replacement cannot win. Such a candidate
requires a new preregistration against the current matrix control rather than a
comparison with the sequential fallback.

No model or GPU workload ran. No GGUF, residency set, `requestResidency`,
pre-wire, `mlock`, cache-bypass read, uncached read, or residency-coupled pread
path ran. PID 8770 remained user-owned and untouched.
