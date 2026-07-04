# v0.469 Prefix-Cache JSONL CLI

Purpose: product-shaped smoke for explicit in-process prefix-cache reuse through
`qwen --requests-jsonl`. This is not a kernel benchmark and not a streaming
first-byte measurement.

## Change

- `qwen --requests-jsonl FILE` keeps one `LoadedModel` resident across requests.
- JSONL requests accept `prompt` or `prompt_file`, optional `tokens`, and optional
  `cache_prefix_tokens`.
- `--prefix-cache-max-mib` configures the runtime cache budget.
- `--request-stats PATH` appends per-request JSON stats with hit/miss, matched
  prefix length, restore/insert/prefill/decode timings, and cache bytes.

## 1024-token smoke

Model:

```sh
/Users/tito/models/Qwen3.5-0.8B-Q4_K_M.gguf
```

Command:

```sh
target/release/qwen \
  -m /Users/tito/models/Qwen3.5-0.8B-Q4_K_M.gguf \
  --requests-jsonl target/profiles/v0469-prefix-cache-requests-1k.jsonl \
  --request-stats target/profiles/v0469-prefix-cache-stats-1k-v2.jsonl \
  --cache-prefix-tokens 1024 \
  -n 1 \
  --prefill-chunk 1024 \
  > target/profiles/v0469-prefix-cache-1k-v2.out \
  2> target/profiles/v0469-prefix-cache-1k-v2.err
```

Result:

| Request | Prompt tokens | Cache hit | Matched prefix | Restore ms | Model TTFT ms | Cache bytes |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `long-a` | `1629` | no | `0` | `0.00` | `299.78` | `33,781,808` |
| `long-b` | `1629` | yes | `1024` | `1.75` | `150.20` | `33,781,808` |

Interpretation: the second request restores the exact configured `1024`-token
prefix and avoids prefill for that prefix. The output path is JSONL after full
decode, so `model_ttft_ms` is internal model timing rather than client-visible
streaming first-byte latency.

## 16-token smoke

Command:

```sh
target/release/qwen \
  -m /Users/tito/models/Qwen3.5-0.8B-Q4_K_M.gguf \
  --requests-jsonl target/profiles/v0469-prefix-cache-requests.jsonl \
  --request-stats target/profiles/v0469-prefix-cache-stats-v2.jsonl \
  --trace-request target/profiles/v0469-prefix-cache-trace-v2.tsv \
  --cache-prefix-tokens 16 \
  -n 1 \
  --prefill-chunk 64 \
  > target/profiles/v0469-prefix-cache-v2.out \
  2> target/profiles/v0469-prefix-cache-v2.err
```

Result: request 2 restored the explicit `16`-token prefix. This is only a wiring
smoke; v0.442/v0.445 already showed small prefixes should not be sold as the
product win.

## Caveats

- Exact token-prefix identity is required.
- Cold requests pay synchronous snapshot insertion.
- The cache budget is bounded, but an oversized newest snapshot is retained alone.
- No automatic common-prefix discovery, cross-process persistence, concurrent
  serving, or streaming output is claimed by this checkpoint.

## Validation

- `cargo fmt --check`
- `cargo check -p qwen-cli --bin qwen`
- `cargo build --release -p qwen-cli --bin qwen`
- cx review `019f2ddc-c`
