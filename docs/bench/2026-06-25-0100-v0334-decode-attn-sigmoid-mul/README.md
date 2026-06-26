# Decode Attention Sigmoid-Mul - v0.334

Focused decode A/B for fusing the gated-attention `sigmoid(gate)` pass and the
following multiply into one elementwise kernel. Same physical box, sequential
runs only, no concurrent GPU work, AC power, and no recorded thermal or
performance warnings. This is a bounded attention-shelf cleanup, not a KV/body
execution-shape rewrite.

Rollback: `QWEN_DECODE_ATTN_SIGMOID_MUL=0`.

## Validation

```sh
cargo fmt
cargo test -p qwen-llm elementwise_add_mul_silu_mul_sigmoid_mul --release -- --nocapture
cargo build --release --bin qwen-bench
```

## Long Decode A/B

| Model | Context | Variant | t/s | GPU ms/token | Read |
| --- | ---: | --- | ---: | ---: | --- |
| A3B Q4_K_M | `8192` | default | `93.6`, `93.3` | `10.24`, `10.24` | positive |
| A3B Q4_K_M | `8192` | rollback | `92.1`, `92.2` | `10.40`, `10.35` | baseline |
| A10B Q4_K_XL | `8192` | default | `42.4`, `42.6` | `23.07`, `22.97` | neutral/noise |
| A10B Q4_K_XL | `8192` | rollback | `42.6`, `42.2` | `23.00`, `23.16` | baseline |
| 27B Q4_K_M | `8192` | default | `22.6` | `43.71` | one-run dense guard positive |
| 27B Q4_K_M | `8192` | rollback | `21.9` | `45.17` | baseline |

## Short Decode Guards

| Model | Context | Variant | t/s | GPU ms/token | Read |
| --- | ---: | --- | ---: | ---: | --- |
| A3B Q4_K_M | `1024` | default | `99.4`, `99.4`, `99.6` | `9.60`, `9.64`, `9.59` | neutral/positive |
| A3B Q4_K_M | `1024` | rollback | `99.4`, `99.2`, `99.4` | `9.62`, `9.68`, `9.64` | baseline |
| A10B Q4_K_XL | `1024` | default | `42.5` | `23.02` | one-run guard positive |
| A10B Q4_K_XL | `1024` | rollback | `42.3` | `23.16` | baseline |

## Interpretation

- Fusing the attention gate removes one pass/dispatch per attention layer and is
  default-worthy as a small, broad decode cleanup.
- A3B long-context decode shows the clearest MoE-family benefit; A10B is neutral
  within noise, and dense 27B long decode is directionally positive.
- This does not change the deeper attention/KV conclusion: the remaining large
  attention budget is the v4 body/KV path, not another scalar epilogue pass.
