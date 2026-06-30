# A3B True-Long Attention H2 Reduce — v0.369

Clean AC-power follow-up to v0.368. The candidate keeps tile4/NWG256 for A3B
true-long attention, then splits the decode v4 reduce over two threadgroups per
Q head so each reduce group handles half of `head_dim=256`.

## Identity

- `qwen-llm`: v0.369 candidate after `bf17846`
- GPU: `Apple M4 Max`
- Power: AC attached
- QWEN env: none

## Results

Sequential `qwen-bench ctx-sweep --window 16`, single max-capacity session.

| ctx | v0.368 | v0.369 candidate | Ratio |
| ---: | ---: | ---: | ---: |
| `4096` | `101.1 t/s` | `102.5 t/s` | flat/noise |
| `8192` | `96.1 t/s` | `96.5 t/s` | flat |
| `16384` | `91.2 t/s` | `95.0 t/s` | `1.04x` |
| `32768` | `85.4 t/s` | `88.2 t/s` | `1.03x` |

## Attribution

`attn-intra --ctx 32768` shows the direct mechanism:

| Metric | v0.368 | v0.369 candidate | Read |
| --- | ---: | ---: | --- |
| one attention layer | `0.4069 ms` | `0.3457 ms` | `1.18x` |
| main pass | `0.1743 ms` | `0.1743 ms` | unchanged |
| reduce pass | `0.1284 ms` | `0.0677 ms` | `1.90x` |

Whole-model phase at `ctx32768` is noisier but directionally matches: attention
drops `3.68 -> 3.33 ms`, while the context sweep moves `85.4 -> 88.2 t/s`.

## Validation

- `cargo fmt`
- `cargo build --release --bin qwen-bench`
- `cargo test -p qwen-llm attn_v4_matches_naive_f16kv --release -- --nocapture`
- A3B `attn-intra --ctx 32768 --runs 3`
- A3B `ctx-sweep --checkpoints 4096,8192,16384,32768 --window 16`

## Read

The true-long A3B attention branch still has headroom, but the cheap reduce split
is now banked. The next exact attention attempt should either improve the read-once
tile8 occupancy path or reduce partial traffic more structurally. If that does not
clear another true-long gate, switch to MoE down.
