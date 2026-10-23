# V4.1 prefill: two sink-fusion probes are negative

M5 Ultra 80-core, in-tree sink timing fixture (`QWEN_V41_SINK_TIMING=1`,
640 keys, full sink). Branches `perf/v41-sink-fuse` (online fusion) and
`perf/v41-pv-normalize` (normalize fold) are retained as evidence; neither
is landed. Production stays on `score_mma` + `softmax_rows` + `pv_mma` +
`normalize_rows`.

## Online-softmax tile

One threadgroup per (16-head block, query row), eight-key blocks gathered
once and used for QK^T and then PV, with a running max/sum seeded by the
sink. It is correct (the scalar-equivalence and invalid/finite guard tests
pass) but much slower:

| rows per dispatch | MMA path | online fused | ratio |
|---:|---:|---:|---:|
| 32 | 647 us | 2,964 us | 4.6x |
| 128 | 1,190 us | 11,252 us | 9.5x |
| 256 | 2,050 us | 20,657 us | 10.1x |
| 512 | 3,426 us | 36,985 us | 10.8x |

The fused threadgroup needs about 25 KB of static threadgroup memory
(16 KB Q + 8 KB key block + 0.5 KB scores + 0.25 KB probabilities + stats),
so only one resident threadgroup fits per core. Every eight-key block then
serializes gather -> QK^T -> online softmax -> PV behind five barriers,
and only two of the eight simdgroups are active during the score MMA.
The three-kernel path keeps four score simdgroups and overlaps many
threadgroups across the score and PV dispatches. Do not rebuild the online
route without a smaller threadgroup footprint or a way to keep more than
one threadgroup resident.

## Normalize fold into `pv_mma`

The second probe moved the softmax denominator division and finite check
into the PV accumulator store, removing the separate normalize dispatch
and its 134 MB read/write pass (134 us at 512 rows):

| variant | pv at 512 rows | full sink |
|---|---:|---:|
| main (separate normalize) | 1,049 us | 3,426 us |
| divide at the PV store | 19,187 us | 21,571 us |
| reciprocal + multiply | 19,496 us | 21,818 us |
| divide, no finite guard | 19,245 us | 21,579 us |
| no scaling (status/row helpers only) | 1,294 us | 5,282 us |

The trigger is the per-thread denominator load/scaling in the
register-heavy PV kernel, not the finite guard: removing the guard or
using a reciprocal does not recover the time, while removing the scaling
returns the kernel to about 1.3 ms. The standalone pv timing uses
uninitialized scores (NaN), and the finite-guard variants add atomics there;
the full-sink timing runs softmax first (finite probabilities) and is
still about 21.5 ms, so the conclusion holds. The separate normalize pass
is already the cheap form; keep it.

## Limits

- Fixture attribution on one M5 Ultra; no full-model run for either probe
  because the fixture gap is an order of magnitude.
- The online kernel was tested with the 8-key form only. A 16-key form
  would exceed the 32 KiB threadgroup budget without dropping Q staging or
  splitting the latent dimension, and the barrier/occupancy problem
  remains.
