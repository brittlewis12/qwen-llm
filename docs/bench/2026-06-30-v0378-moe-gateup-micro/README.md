# v0.378 MoE Gate/Up Microbench

Purpose: add a visible primitive harness for routed Q4_K gate/up SwiGLU decode,
analogous to `moe-down-micro`. This lets routed gate/up shape variants be tested
without full context ramps.

New command:

```text
target/release/qwen-bench moe-gateup-micro -m <MODEL>
```

## Measurements

```text
target/release/qwen-bench moe-gateup-micro \
  -m /Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf \
  --warmup 3 \
  --iters 10 \
  > target/profiles/v0378-a10b-q4xl-moe-gateup-micro.out

target/release/qwen-bench moe-gateup-micro \
  -m /Users/tito/models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --warmup 3 \
  --iters 10 \
  > target/profiles/v0378-a3b-q4-moe-gateup-micro.out
```

| Model row | Q4 layers | Active weight | GPU | Weight BW | Phase comparison |
| --- | ---: | ---: | ---: | ---: | --- |
| A10B `ctx8192` shape | `47` | `1.3306 GB` | `3.0399 ms` | `437.7 GB/s` | close to `3.18 ms` phase |
| A3B `ctx32768` shape | `40` | `0.3775 GB` | `1.5179 ms` | `248.7 GB/s` | above `1.07 ms` phase |

## Decision

Keep the harness. It validates well enough for A10B, where routed gate/up is the
larger live bucket, and it makes future Q4 gate/up kernel probes cheap.

Do not trust synthetic top-k patterns as promotion evidence on A3B yet: the micro
overstates the current phase by too much. Before defaulting a gate/up variant,
either add captured route-pattern replay to the harness or require full phase/ctx
confirmation on A3B.
