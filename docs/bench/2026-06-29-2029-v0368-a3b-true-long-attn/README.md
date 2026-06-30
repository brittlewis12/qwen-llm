# A3B True-Long Attention NWG256 — v0.368

Clean AC-power A3B Q4 decode packet for the true-long group-8 attention branch.
The candidate keeps the v0.367 medium-context subgroup threshold, then switches
group-8 decode to tile4/NWG256 at `ctx >= 16384`. Rollback:
`QWEN_ATTN_V4_G8_TILE=2 QWEN_ATTN_V4_NWG=64`.

## Identity

- `qwen-llm`: v0.368 candidate after `d72b42e`
- GPU: `Apple M4 Max`
- Power: AC attached
- QWEN env, default rows: none

## Results

Sequential `qwen-bench ctx-sweep --window 16`, single max-capacity session.
The selector changes only `ctx >= 16384`; rows below that threshold are included
as guard context and stayed in the prior flat/noise envelope.

| ctx | v0.367 default | v0.368 candidate | Ratio |
| ---: | ---: | ---: | ---: |
| `4096` | `100.9 t/s` | `101.1 t/s` | flat |
| `8192` | `96.3 t/s` | `96.1 t/s` | flat |
| `16384` | `88.4 t/s` | `91.2 t/s` | `1.03x` |
| `32768` | `75.7 t/s` | `85.4 t/s` | `1.13x` |

## Attribution

At `ctx32768`, phase attribution moves attention from the dominant cliff bucket
toward the rest of the decode graph:

| Phase | v0.367 | v0.368 candidate | Read |
| --- | ---: | ---: | --- |
| phase sum | `13.59 ms` | `12.05 ms` | `-1.54 ms` |
| attention mixer | `5.14 ms` | `3.68 ms` | `-28%` |
| GDN qkv/z | `1.14 / 0.63 ms` | `1.13 / 0.63 ms` | unchanged |
| MoE routed gate/up/down | `1.08 / 0.82 ms` | `1.06 / 0.81 ms` | unchanged |

`attn-intra --ctx 32768` showed why this works: tile2/NWG64 reads about 4x
logical KV and is fastest at medium contexts, while tile4/NWG256 halves repeated
KV reads and restores enough occupancy at true-long. Tile8/NWG256 reads once but
still loses to tile4 because the reduce overhead and occupancy tradeoff are not
good enough yet.

## Validation

- `cargo fmt`
- `cargo build --release --bin qwen-bench`
- `cargo test -p qwen-llm attn_v4_matches_naive_f16kv --release -- --nocapture`
- A3B `attn-intra --ctx 32768` for tile2/tile4/tile8 and NWG64/128/256 probes
- A3B `ctx-sweep` through `32768`, sequential, no concurrent GPU workloads

## Read

The single-pass/read-once direction is real but under-occupied at tile8. The
highest-leverage exact continuation is a better read-once/group-fused attention
shape that keeps tile4-or-better occupancy while cutting repeated KV further. If
that does not clear another true-long gate, switch to the broader MoE-down branch.
