# v0.428 Replay Occupancy Economics

Goal: convert the replay-margin keep-alive signal into a first economic gate by
measuring how much occupancy is needed before block-slice GDN replay pays for
itself.

Commands:

```sh
target/release/qwen-bench decode-block-slice-replay \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --start-block 0 --blocks 4 --position 4096 \
  --tokens 1,2,3,4,6,8 --warmup 1 --iters 3 \
  > target/profiles/v0428-a3b-block-slice-occupancy-b0-pos4096.out

target/release/qwen-bench decode-block-slice-replay \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --start-block 20 --blocks 4 --position 4096 \
  --tokens 1,2,3,4,6,8 --warmup 1 --iters 3 \
  > target/profiles/v0428-a3b-block-slice-occupancy-b20-pos4096.out

target/release/qwen-bench decode-block-slice-replay \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --start-block 0 --blocks 2 --position 4096 \
  --tokens 1,2,3,4,6,8 --warmup 1 --iters 3 \
  > target/profiles/v0428-a3b-block-slice-occupancy-b0-blocks2-pos4096.out

target/release/qwen-bench decode-block-slice-replay \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --start-block 20 --blocks 2 --position 4096 \
  --tokens 1,2,3,4,6,8 --warmup 1 --iters 3 \
  > target/profiles/v0428-a3b-block-slice-occupancy-b20-blocks2-pos4096.out
```

Validation:

- A3B synthetic block-slice replay occupancy sweeps at pos4096
- Correctness checks from `decode-block-slice-replay` for S=8 rows

## Results

`blocks=4` windows contain 3 GDN blocks and 1 attention block. Replay loses at
S=1/2/3, barely wins at S=4, and becomes material at S=6/8:

| Window | S=1 | S=2 | S=3 | S=4 | S=6 | S=8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `block0..4` save % | `-80.4` | `-26.8` | `-7.8` | `3.8` | `11.0` | `16.2` |
| `block20..24` save % | `-78.0` | `-26.6` | `-8.3` | `2.5` | `11.6` | `16.8` |

`blocks=2` windows here are GDN-only pairs. They also lose at S=1/2/3 and only
become attractive at S=6/8, but their high-occupancy gross savings are larger:

| Window | S=1 | S=2 | S=3 | S=4 | S=6 | S=8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `block0..2` save % | `-143.1` | `-44.9` | `-5.2` | `1.4` | `19.6` | `23.3` |
| `block20..22` save % | `-145.3` | `-42.8` | `-4.7` | `0.4` | `18.9` | `23.1` |

Using the v0.427 combined real-margin fallback rates and cx's conservative
post-replay fallback model (`net ~= gross_save - fallback_rate`):

| Policy point | Fallback rate | `blocks=4` S=8 net | `blocks=2` S=8 net |
| --- | ---: | ---: | ---: |
| `1e-4` | `2.13%` | `~14%` | `~21%` |
| `3e-4` | `8.51%` | `~8%` | `~15%` |
| `1e-3` | `23.40%` | negative | around parity |

## Decision

Replay economics are real but occupancy-sensitive. Do not schedule replay below
roughly 6 active slots unless a future path removes fixed overhead. For the first
production-shaped model, use S>=6 as the occupancy gate and `3e-4` as the first
margin-policy point to test; `1e-3` is too expensive under post-replay fallback.

`blocks=2` is now worth reconsidering: despite shorter windows, the measured
GDN-only pairs have better high-occupancy gross savings and v0.420 already showed
that shorter windows reduce synthetic route-set flips. The next gate should
combine blocks=2/4, occupancy distribution, validation overhead, and exact
fallback cost into one scheduler-economics model.
