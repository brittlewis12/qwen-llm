# v0.413 GDN Chain Replay

Goal: test whether GDN replay survives chaining multiple GDN layers on the same
per-slot sessions, preserving residual stream, conv state, and GDN state across
the chain.

The new `decode-gdn-chain-replay` command measures consecutive GDN-layer indexes.
It still skips attention/MoE blocks between selected GDN layers, so this is not a
full block-slice scheduler gate. It is a correctness and command-structure bridge
between isolated GDN layers and an integrated decode slice.

Commands:

```sh
target/release/qwen-bench decode-gdn-chain-replay \
  -m /Users/tito/models/Qwen3.5-0.8B-Q4_K_M.gguf \
  --layers 2 --tokens 1 --warmup 0 --iters 1 \
  > target/profiles/v0413-gdn-chain-replay-0p8b-smoke.out

target/release/qwen-bench decode-gdn-chain-replay \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --layers 4 --tokens 8,16 --warmup 1 --iters 3 \
  > target/profiles/v0413-a3b-gdn-chain-replay-l4-s8-s16.out
```

Validation:

- `cargo fmt`
- `cargo check -p qwen-cli --bin qwen-bench`
- `cargo build --release -p qwen-cli --bin qwen-bench`
- 0.8B two-layer smoke
- A3B four-layer GDN chain replay at `S=8/16`
- `cx ask` review, session `019f1ed4-ad69-73d3-98ac-2735704bf54c`

Artifacts:

- `target/profiles/v0413-gdn-chain-replay-0p8b-smoke.out`
- `target/profiles/v0413-a3b-gdn-chain-replay-l4-s8-s16.out`

## Results

A3B four-layer chain: `start_gdn=0`, `layers=4`, blocks `0,1,2,4`.

Correctness after all four replayed layers across `S=16` slots:

| Check | Value |
| --- | ---: |
| `min_cos_h` | `0.999999509` |
| `max_abs_h` | `0.006552` |

Replay timing, in `ms/token` for the four-layer chain:

| S | Baseline seq | Replay | Save | Save % |
| ---: | ---: | ---: | ---: | ---: |
| `8` | `0.4811` | `0.3217` | `0.1594` | `33.1` |
| `16` | `0.4783` | `0.2247` | `0.2536` | `53.0` |

Per-layer save is roughly `0.0399 ms/token` at `S=8` and `0.0634` at `S=16`,
consistent with the v0.412 layer sample.

## Decision

GDN replay is now plausible across both representative layers and chained state.
The next useful test is not more GDN-only replay: build an integrated full
block-slice for 2-4 consecutive real blocks, keeping normal attention/MoE in
place and replacing only the GDN subpath with replay.

Scheduler work remains gated on integrated net wall-clock savings and ragged-slot
behavior. The GDN-only chain does not prove those yet.
