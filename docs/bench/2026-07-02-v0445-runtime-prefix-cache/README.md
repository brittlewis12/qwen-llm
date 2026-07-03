# v0.445 Runtime Prefix-Cache API Integration

Goal: move the repeated-prefix TTFT win from a bench-local proof into the runtime
boundary without touching the parallel DFlash/kernel branch.

## What changed

- `LoadedModel` now owns a bounded `PrefixCache` initialized from
  `LoadedModelConfig::prefix_cache_max_bytes`.
- Runtime cache keys carry in-process model/tokenizer compatibility fingerprints
  plus the existing session layout identity.
- Runtime APIs now expose cache stats, memory-cap resize, clear, snapshot insert,
  and exact longest-prefix restore.
- `Sequence` now has runtime-level snapshot/restore helpers that keep the tracked
  token position coherent with restored KV/GDN state and reject explicit identity
  or capacity mismatches before writing Metal state.
- `qwen-bench prefix-cache` now exercises this runtime cache path and prints cache
  entry/byte stats.

Commands:

```sh
cargo check -p qwen-llm -p qwen-cli --bin qwen-bench
cargo build --release -p qwen-cli --bin qwen-bench
cargo test -p qwen-llm prefix_cache -- --nocapture

target/release/qwen-bench prefix-cache \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --target-prefix-len 1024 --tokens 4 \
  > target/profiles/v0445-a3b-prefix-cache-runtime-stats-1024.out 2>&1

target/release/qwen-bench prefix-cache \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --target-prefix-len 4096 --tokens 4 \
  > target/profiles/v0445-a3b-prefix-cache-runtime-stats-4096.out 2>&1
```

Validation:

- `cargo check -p qwen-llm -p qwen-cli --bin qwen-bench`
- `cargo build --release -p qwen-cli --bin qwen-bench`
- `cargo test -p qwen-llm prefix_cache -- --nocapture` (`8 passed`, one ignored
  GPU correctness spike remains ignored)
- A3B runtime-integrated prefix-cache probes at prefix lengths 1024 and 4096

## Results

| Prefix | Cold TTFT | Warm TTFT | Speedup | Restore | Agreement |
| ---: | ---: | ---: | ---: | ---: | --- |
| `1024` | `609.3 ms` | `118.2 ms` | `5.16x` | `4.1 ms` | `4/4` |
| `4096` | `2347.4 ms` | `126.8 ms` | `18.51x` | `7.4 ms` | `4/4` |

Both rows preserve the v0.442 product shape after moving through runtime-owned
cache insertion and restore. Cache stats are now visible in the probe output; the
1024 row stores an `87.8 MB` snapshot, and the 4096 row stores `150.8 MB`.

## Decision

Prefix caching is now a runtime feature boundary, not just a bench artifact. The
remaining work is product wiring: request handling, user-visible cache controls,
cache hit/miss observability, and real prefix identity policy. Further kernel work
is not required for the current prefix-cache gate.

Not solved here: cross-process persistence, shared cache reuse across model loads,
true content-addressed snapshot identity, OS memory-pressure integration,
admission/min-prefix policy, concurrent restore/cache QoS, exact-hit logits
validation beyond vocab-length checks, stochastic sampling equivalence, or
DFlash/packed-verify coverage.
