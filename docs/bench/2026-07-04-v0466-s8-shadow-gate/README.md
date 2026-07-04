# v0.466 S8 Shadow Gate

Goal: make the replay promotion gate request-shaped instead of full-occupancy
only, without building a runtime scheduler.

## What Changed

- `decode-block-slice-real-margin` now accepts `--slot-counts`, so one prepared
  prefix set can emit S1/S2/S4/S8 rows for the same context and windows.
- `replay_economics.py --request-trace` reports replayed step and token shares,
  making p95 request simulations harder to overread.

## Commands

```sh
cargo build --release -p qwen-cli --bin qwen-bench

target/release/qwen-bench decode-block-slice-real-margin \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --file /Users/tito/code/llm/game/chaos.json \
  --tokens 8 --slot-counts 1,2,4,8 --context 8192 --stride 1 \
  --blocks 2 --start-block 0,20 --timing-iters 3 --timing-warmup 1 \
  --margin-threshold 0.0003 \
  > target/profiles/v0466-a3b-real-economics-slotcounts-c8192.tsv

target/release/qwen-bench decode-block-slice-real-margin \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --file /Users/tito/code/llm/game/chaos.json \
  --tokens 8 --slot-counts 1,2,4,8 --context 16384 --stride 1 \
  --blocks 2 --start-block 0,20 --timing-iters 3 --timing-warmup 1 \
  --margin-threshold 0.0003 \
  > target/profiles/v0466-a3b-real-economics-slotcounts-c16384.tsv
```

The request simulations use synthetic traces only; they are policy-shape probes,
not workload evidence.

## Results

| Context | Window | S8 Net Save | Notes |
| ---: | --- | ---: | --- |
| `8192` | `0..2` | `7.51%` | S1/S2/S4 negative |
| `8192` | `20..22` | `10.51%` | S1/S2 negative, S4 negative |
| `16384` | `0..2` | `-3.13%` | repeated S8-only run: `7.25%` |
| `16384` | `20..22` | `14.27%` | one route-order-only mismatch |

Synthetic request simulation, S8-only policy:

| Context | Trace | Save | p95 Delta | Replayed Tokens |
| ---: | --- | ---: | ---: | ---: |
| `8192` | saturated | `9.01%` | `-9.01%` | `100.00%` |
| `8192` | ragged burst | `6.07%` | `-7.68%` | `68.75%` |
| `16384` | saturated | `5.57%` | `-5.57%` | `100.00%` |
| `16384` | ragged burst | `3.73%` | `-4.77%` | `68.75%` |

## Decision

The S8-only replay branch remains live but is not promoted. The new evidence is
more conservative than the full-S8 v0.465 row: ctx8192 clears the synthetic ragged
`>=5%` gate, while ctx16384 is marginal unless the noisy window-0 repeat is used.
The next gate should use a real or captured request trace and robust repeats; do
not build attention-slice support or a production scheduler from this packet.
