# v0.343 MoE Routed-Down Microbench

Status: added `qwen-bench moe-down-micro` to isolate the exact Q5_K routed-down
weighted-sum primitive outside the long decode ramp. The harness runs all Q5_K
routed-down expert banks in one command buffer and reports active weight-byte
bandwidth.

## Validation

- `cargo fmt`
- `cargo build --release --bin qwen-bench`
- sequential A3B/A10B `moe-down-micro` token-count sweeps, no parallel GPU workloads

## Results

Command shape:

```sh
target/release/qwen-bench moe-down-micro -m <model> \
  --warmup 5 --iters 20 --tokens <1|2|4|8|16>
```

| Model | Q5 layers | Tokens | Active weight GB | GPU ms | Weight GB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| A3B Q4_K_M | 37 | 1 | `0.2134` | `1.1668` | `182.9` |
| A3B Q4_K_M | 37 | 2 | `0.4268` | `2.0106` | `212.3` |
| A3B Q4_K_M | 37 | 4 | `0.8535` | `3.7692` | `226.5` |
| A3B Q4_K_M | 37 | 8 | `1.7071` | `7.2908` | `234.1` |
| A3B Q4_K_M | 37 | 16 | `3.4142` | `14.4364` | `236.5` |
| A10B Q4_XL | 47 | 1 | `0.8132` | `2.5013` | `325.1` |
| A10B Q4_XL | 47 | 2 | `1.6263` | `4.7139` | `345.0` |
| A10B Q4_XL | 47 | 4 | `3.2527` | `9.2057` | `353.3` |
| A10B Q4_XL | 47 | 8 | `6.5054` | `18.3989` | `353.6` |
| A10B Q4_XL | 47 | 16 | `13.0107` | `37.4929` | `347.0` |

Interpretation: the microbench matches the long-decode routed-down attribution
closely enough to use as a faster primitive gate. A3B's smaller Q5 shape remains
substantially weaker than A10B, and batching synthetic tokens improves A3B only
from about `183 -> 236 GB/s`, so the single-token deficit is not just command
overhead. Future Q5 work needs a different work unit or dequant/dataflow change,
not another full-model long-ramp probe for every attempt.

## Artifacts

- `target/profiles/v0343-a3b-moe-down-micro-tokens.tsv`
- `target/profiles/v0343-a10b-moe-down-micro-tokens.tsv`
