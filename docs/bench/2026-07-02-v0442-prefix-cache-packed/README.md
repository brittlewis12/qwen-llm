# v0.442 Product-Shaped Prefix Cache Probe

Goal: re-test the cross-turn prefix cache under the current packed prefill path.
The old `prefix-cache` harness used a per-token prefill loop, which made cache
hits look unrealistically strong now that product prompt prefill is much faster.

## What changed

- `qwen-bench prefix-cache` now defaults to packed prefill for cold and cached
  prefix construction.
- `--prefill-mode single` remains as a legacy diagnostic control.
- `--suffix-prefill-mode auto|packed|single` controls the cache-hit suffix path.
  `auto` uses per-token decode for suffixes up to 64 tokens because the measured
  11-token suffix was faster as single-token decode than as tiny packed prefill.

Commands:

```sh
cargo check -p qwen-llm -p qwen-cli --bin qwen-bench
cargo build --release -p qwen-cli --bin qwen-bench

target/release/qwen-bench prefix-cache \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --target-prefix-len 64 --tokens 4 \
  > target/profiles/v0442-a3b-prefix-cache-auto-64.out 2>&1

target/release/qwen-bench prefix-cache \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --target-prefix-len 256 --tokens 4 \
  > target/profiles/v0442-a3b-prefix-cache-auto-256.out 2>&1

target/release/qwen-bench prefix-cache \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --target-prefix-len 1024 --tokens 4 \
  > target/profiles/v0442-a3b-prefix-cache-auto-1024.out 2>&1

target/release/qwen-bench prefix-cache \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --target-prefix-len 4096 --tokens 4 \
  > target/profiles/v0442-a3b-prefix-cache-auto-4096.out 2>&1
```

Validation:

- `cargo check -p qwen-llm -p qwen-cli --bin qwen-bench`
- `cargo build --release -p qwen-cli --bin qwen-bench`
- A3B prefix-cache rows at prefix lengths 64, 256, 1024, and 4096
- Greedy first-token and 4-token continuation agreement on every row

## Results

| Prefix | Cold TTFT | Warm TTFT | Speedup | Restore | Gate |
| ---: | ---: | ---: | ---: | ---: | --- |
| `64` | `145.0 ms` | `114.5 ms` | `1.27x` | `3.0 ms` | fail `2x` |
| `256` | `222.6 ms` | `115.9 ms` | `1.92x` | `3.3 ms` | fail `2x` |
| `1024` | `609.8 ms` | `115.5 ms` | `5.28x` | `4.3 ms` | pass `5x` |
| `4096` | `2341.7 ms` | `126.8 ms` | `18.47x` | `7.3 ms` | pass `5x` |

For comparison, forcing the 11-token suffix through packed prefill made cache-hit
TTFT worse at the measured long prefixes: prefix 1024 was `153.4 ms` warm TTFT
and prefix 4096 was `450.0 ms`. The auto short-suffix path is therefore the
right product default for this regime.

## Decision

Prefix caching is a major product TTFT lever for repeated 1K+ prefixes and has a
cheap restore cost. It is not a small-prefix optimization. The next cache branch
should wire this into runtime/CLI request handling with bounded memory policy and
cache identity semantics; further kernel work is not needed for the current gate.
