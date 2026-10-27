# V4.1 Full/CED Pair: Publication Replay Guard And Q5_K Parity

2026-10-27. Source: `73e684a6` (CED publication batching and replay guard),
`3a8d9fe2` (Q5_K matmat default correction), on `abe6e6e7`. Artifact: GSQ-RCO
IQ2_XS (~144.6 GiB resident), 40 layers, 4,096-row Full chunks, M5 Ultra.
Raw: `metagross:~/qwen-runs/2026-10-27-v41-ced-guard/`.

## Decision

The 8K Full/CED pair passes with bitwise-identical endpoints. CED is
**1,437.15 tok/s** against Full **1,020.74 tok/s** (+40.8%) on the same binary,
lease, prompt and capacity. This qualifies the pair arithmetic; CED remains
opt-in, and the dedicated Q5_K matmat remains opt-in. It does not qualify CED
as the product default or a sustained campaign result.

## Two Independent Mismatches

The bisect from the prior session isolated `ddcec6b6` (batched owner
publication) as the first failing commit. It contained two separable defects.

### 1. Replay republished the authoritative cache

`encode_rows_serial` selected the batched publisher from
`batched && owns_kv(id)` alone. During CED's layer-20 decoder replay the mode
is `ReusePublished`: the cache is complete and authoritative, and the per-row
publication was already suppressed by the `GlobalKvMode::Ordinary` check in
`encode_inner_with`. The batch bypassed that check and overwrote published
rows with decoder-chunk rows.

Fix (`73e684a6`): gate `owner_batch` on `mode == GlobalKvMode::Ordinary`, and
use the same batched publisher in CED's dedicated publication command so both
paths share the published arithmetic. The gate tensor is unused at ratio 1.

### 2. The Q5_K matmat is not bitwise-equal to the generic tile

The original `mat_mat_q5_K` path was gated by per-row cosine >= 0.999, not
bitwise equality. `e331ee5b` routed Q5_K away from the generic dense tensor
tile to the dedicated 64x64 tensor form. Full's multi-row chunks take the
dedicated kernel; CED's shrinking replay chunks and its one-row tail take the
generic simdgroup route (or the n16/n64 variants). The 2K greedy endpoint is
token-robust and did not see the difference; the pair's full-logit gate does.

Fix (`3a8d9fe2`): make `QWEN_V41_Q5K_MATMAT` opt-in (`default_off`). Q5_K
returns to the generic dense tensor tile for every row count, which is the
base routing and is bitwise row-count independent. The +2.54% dedicated-tile
gain is not part of the default path until a bitwise witness exists.

## Evidence

2K discriminator (`--pp 2048 --runs 2`, production timing, interleaved):

| build | result |
| --- | --- |
| `17af573b` base | pass |
| `ddcec6b6` + replay guard | fail, case 1 |
| current + index tensor off | fail, case 1 |
| current + Q5_K matmat off | pass |

8K gate (`--pp 8192 --runs 4`, P0, production timing, `full-blocks-first`):

| policy | samples (tok/s) | mean | stddev |
| --- | --- | ---: | ---: |
| Full | 1,021.29, 1,021.09, 1,020.60, 1,019.99 | 1,020.74 | 0.58 |
| CED | 1,441.39, 1,431.83, 1,435.87, 1,439.52 | 1,437.15 | 4.23 |

All 14 recorded Full/CED calls carry the same endpoint hash,
`04267cbdf9fbfdfed674f10a3a125c970c734aeed1a2426ff1018f489d93ee2d`.
Build identity is clean (`build_dirty=0`). The 8K pair completes 12 CED
commands, 163,840 encoder output rows, 8,192 publication rows, 24,150 decoder
output rows and 2,540 seed rows with no fallback.

## Limits

The pair protocol runs Full before CED with fresh state per call. Its order
puts any thermal bias against CED, but it is not the 180 s-cooldown sustained
campaign and is not a new sustained-throughput claim. The endpoint gate is one
prompt and one artifact, not a quality sweep or a teacher-forced continuation.
CED is still opt-in; Full remains the default. The Q5_K dedicated tile needs a
raw bitwise witness against the generic Q5_K tensor path before it can be
defaulted again.
