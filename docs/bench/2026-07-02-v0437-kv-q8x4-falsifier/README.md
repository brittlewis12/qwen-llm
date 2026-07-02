# v0.437 KV-Q8x4 Attention Falsifier

Goal: close the reopened KV-Q8 question with a reader micro-oracle that preserves
the tuned F16 `attn_v4` launch grid instead of repeating the old forced-tile Q8
sidecar.

Implementation shape:

- `QWEN_KV_Q8=1` remains default-off.
- Dense group6 and MoE group8 sessions can allocate `Q8_0` KV caches when
  `head_dim=256`.
- Group8 Q8 main kernels cover `C=16/32/64/128`.
- Group8 Q8 subgroup kernels cover tile2/tile4 and `C=16/32/64/128`.
- The reader mirrors the F16 vector shape: each lane reads four int8 values,
  converts to `float4`, dots against `half4` Q, and accumulates/writes `float4`
  partials.
- `attn-intra` now dispatches the Q8 KV scatter path when the session KV dtype is
  `Q8_0`, so phase probes measure the real cache layout.

Commands:

```sh
cargo check -p qwen-llm -p qwen-cli --bin qwen-bench
cargo build --release -p qwen-cli --bin qwen-bench
cargo test --release -p qwen-llm attn_v4_q8 -- --nocapture

target/release/qwen-bench attn-intra \
  -m /Users/tito/models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --ctx 8192 --runs 3 \
  > target/profiles/v0437-a3b-attn-intra-ctx8192-f16-q8x4.out

QWEN_KV_Q8=1 target/release/qwen-bench attn-intra \
  -m /Users/tito/models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --ctx 8192 --runs 3 \
  > target/profiles/v0437-a3b-attn-intra-ctx8192-q8x4.out

target/release/qwen-bench attn-intra \
  -m /Users/tito/models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --ctx 32768 --runs 2 \
  > target/profiles/v0437-a3b-attn-intra-ctx32768-f16-q8x4.out

QWEN_KV_Q8=1 target/release/qwen-bench attn-intra \
  -m /Users/tito/models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --ctx 32768 --runs 2 \
  > target/profiles/v0437-a3b-attn-intra-ctx32768-q8x4.out
```

Validation:

- `cargo check -p qwen-llm -p qwen-cli --bin qwen-bench`
- `cargo build --release -p qwen-cli --bin qwen-bench`
- `attn_v4_q8` release tests:
  - group6 main
  - group8 main
  - group8 tile2
  - group8 tile4
- The Q8-vs-F16 KV test gate is now `cos > 0.9999` and `max_abs < 0.01`.
- cx reviews: `019f248a-ac09-7e13-8e30-179ed19ed7b8` and
  `019f24a4-d467-7d52-ba9b-b30c08147ed1`.

## Results

A3B Q4_K_M `attn-intra`, `ctx8192`, group_tile=2, `nwg=64`, `C=64`:

| KV | One-layer ms | Main ms | Reduce ms | Read |
| --- | ---: | ---: | ---: | --- |
| F16 | `0.2478` | `0.1113` | `0.0341` | baseline |
| Q8x4 | `0.2567` | `0.1221` | `0.0342` | main `~9.7%` slower |

A3B Q4_K_M `attn-intra`, `ctx32768`, group_tile=4, `nwg=256`, `C=64`:

| KV | One-layer ms | Main ms | Reduce ms | Read |
| --- | ---: | ---: | ---: | --- |
| F16 | `0.3490` | `0.1767` | `0.0680` | baseline |
| Q8x4 | `0.4185` | `0.2213` | `0.0708` | main `~25%` slower |

Development note: the initial scalar group8 Q8 reader was also correctness-clean
but slower at `ctx8192` (`main 0.1076 -> 0.1408 ms`). Q8x4 narrowed that loss
but did not reverse it, and true-long got worse.

## Decision

Keep the default-off Q8 oracle and tests, but kill KV-Q8 as an active long-context
attention path under the current `attn_v4` execution model. The byte reduction does
not overcome Q8 dequant/dataflow overhead against the tuned F16 reader.

Do not reopen scalar Q8, forced-tile Q8, group-tile/NWG retunes, or same-layout
Q8x4 variants without a new capture/counter signal and a clean `attn-intra` win
over F16 at both `ctx8192` and `ctx32768`. Future compressed-KV work needs a
materially different layout or attention body, not another Q8_0 reader retune.
