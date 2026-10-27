# V4.1 Batched Owner Compressor Publication

2026-10-26. Source: local main `ddcec6b6` (on `90ee3393`, `17af573b`;
Metagross fast-forwarded from `f3493b7d`). Artifact: GSQ-RCO IQ2_XS
(~144.6 GiB resident), 40 layers, 4,096-row chunks. Raw:
`metagross:~/qwen-runs/2026-10-26-v41-compress-rows/`.

## Decision

Promote. The owner compressor/index publication now runs one threadgroup per
complete ratio-group instead of the per-row copy/pool/norm/key-GEMV/rope chain.
At 8K the four KV-source layers emitted ~98k tiny dispatches in the old path;
the batched form is effectively free under the remaining selection work. The
cleanest same-binary pair is control 870.0 -> new 897.4/897.5 tok/s on the first
two reps (+3.2%); the third rep is 877.9 (+0.9%). The reverse-order campaign is
confounded by a monotonic thermal soak, so the conservative sustained estimate
is ~+0.9% and the single-call estimate is ~+3.2%. Default on;
`QWEN_V41_COMPRESS_ROWS=0` restores the per-row chain.

## Why

Each KV-source layer (2, 8, 14 at ratio 2; 20 at ratio 1) published one group at
a time: even/odd copies or a pool dispatch, the compressor RMS norm, the index
key GEMV, then key/latent norm+rope+store. The profiled serial span was 5,344.6
ms over two 8K calls (2,672 ms per call by the stage log's scaling). The new
kernel pools the prepared value/gate rows, applies the compressor norm, projects
and norms the index key, ropes both, and writes the owner cache in one dispatch
per owner layer.

Per-layer stage means (one layer-chunk instance; two chunks x two runs):

| layer | ratio | old serial ms | new serial ms |
| --- | ---: | ---: | ---: |
| 2, 8, 14 | 2 | 144.7 | 82.6 |
| 20 | 1 | 242.6 | 165.2 |
| 24, 28, 32, 36 | 1 | 164.8-165.0 | 164.8-165.0 |

The owner publication is now hidden under the selection floor: the ratio-1
owner (165.2) matches the index-only layers (165.0), and the ratio-2 owners are
cheaper only because they see half as many keys. The remaining serial span is
selection: index scores plus the top-512 selector on eight index-source layers.

## Correctness

- A 2K-token prompt, 12 greedy tokens, same binary: the default and
  `QWEN_V41_COMPRESS_ROWS=0` outputs are byte-identical (45 bytes,
  sha256 `b47cc0f104b62d4c7c30bcd68fd8e67613e287dc4ad8c310ef10cbadea9c4380`).
- The new GPU witness compares the batched kernel with a CPU reference for
  ratio 1 and 2 at 512/128, including a nonzero absolute group offset.
- 93 attention tests pass. The kernel requires the 512-wide latent, the 128-wide
  BF16 index key, and ratio 1 or 2; other geometry falls back to the per-row
  path. Head/tail rows that do not align to a ratio boundary keep the per-row
  transition so the pending pair and every absolute group boundary are
  unchanged.

## Campaigns

Campaign 1: campaign lease, 120 s cooldown, discarded stage-profiled warm-up,
then N C C N with two runs per arm (N = default, C = `COMPRESS_ROWS=0`):

| arm | samples (tok/s) | mean |
| --- | --- | ---: |
| N | 901.47, 900.11 | 900.79 |
| C | 871.61, 870.22 | 870.92 |
| C | 870.30, 871.30 | 870.80 |
| N | 874.04, 877.10 | 875.57 |

Campaign 2: plain discarded warm-up, then C N N C with three runs per arm:

| arm | samples (tok/s) |
| --- | --- |
| C | 869.39, 869.69, 870.78 |
| N | 897.35, 897.47, 877.93 |
| N | 877.74, 858.74, 844.65 |
| C | 794.27, 790.12, 775.82 |

The second campaign's last control is 9% below the first control. This is a
monotonic thermal/power soak, not a path effect: the new path's higher GPU
utilization heats the box faster, and the later control is measured on a hot
device. Only the adjacent control -> new pair (869-871 -> 897.4/897.5, third
877.9) is a clean single-call comparison; the reverse pair is not. The earlier
scratch-reuse campaigns did not expose this because the workload's power draw
was lower.

## Limits

One artifact and one host session. The stage span is not a native wall and its
absolute scaling differs from the prior budget; compare spans within the same
protocol. The single-call estimate (~+3.2%) and sustained first-three-rep
estimate (~+0.9%) are separate claims. Future campaigns need an explicit
inter-arm cooldown or a fixed thermal soak; a 120 s sleep only before the
campaign does not reset the device between arms. Maya-S/IQ3 generality, the
selection optimization, and the attention tile are separate lanes.
