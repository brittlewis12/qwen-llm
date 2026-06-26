# Decode Branch Gate - v0.332

Focused long-context MoE decode re-anchor for choosing the next hardware-headroom
branch. Same physical box, sequential runs only, no concurrent GPU work, AC power,
and no recorded thermal or performance warnings. This is not a llama.cpp
scoreboard row; it is an attribution gate for qwen-side kernel work.

## Commands

```sh
uv run scripts/profile/decode_ctx_sweep.py \
  --model /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --checkpoints 8192 --window 4 --fresh-per-checkpoint \
  --cooldown-seconds 15 --repeat-blocks 1 --shuffle-seed 331 \
  --variant default \
  --variant down-noop:QWEN_DECODE_MOE_NOOP_ROUTED_DOWN=1 \
  --variant fused-off:QWEN_DECODE_MOE_Q5_DOWN_FUSED=0 \
  --output target/profiles/v0332-a3b-ctx8192-decode-branch-gate.json

uv run scripts/profile/decode_ctx_sweep.py \
  --model /Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL-00001-of-00003.gguf \
  --checkpoints 8192 --window 4 --fresh-per-checkpoint \
  --cooldown-seconds 15 --repeat-blocks 1 --shuffle-seed 332 \
  --variant default \
  --variant down-noop:QWEN_DECODE_MOE_NOOP_ROUTED_DOWN=1 \
  --variant fused-off:QWEN_DECODE_MOE_Q5_DOWN_FUSED=0 \
  --output target/profiles/v0332-a10b-ctx8192-decode-branch-gate.json
```

Phase traces and roofline summaries were captured with `qwen-bench phase`, the
local GGUF tensor dumper, and `scripts/profile/decode_phase_roofline.py` into
`target/profiles/v0332-*`.

## Variant Gate

| Model | Variant | t/s | total ms/token | GPU ms/token | Read |
| --- | --- | ---: | ---: | ---: | --- |
| A3B Q4_K_M | default | `93.1` | `10.74` | `10.29` | baseline |
| A3B Q4_K_M | routed-down no-op | `100.4` | `9.96` | `9.47` | `+7.8%` total budget |
| A3B Q4_K_M | Q5 down fused off | `91.8` | `10.90` | `10.42` | fused path still wins |
| A10B Q4_K_XL | default | `42.4` | `23.59` | `23.11` | baseline |
| A10B Q4_K_XL | routed-down no-op | `46.1` | `21.69` | `21.20` | `+8.7%` total budget |
| A10B Q4_K_XL | Q5 down fused off | `41.3` | `24.20` | `23.70` | fused path still wins |

## Phase Split

| Model | Phase | ms/token | phase share | Read |
| --- | --- | ---: | ---: | --- |
| A3B Q4_K_M | GDN front projections | `2.22` | `20.5%` | near stream-heavy |
| A3B Q4_K_M | attention mixer | `2.30` | `21.3%` | real secondary lane |
| A3B Q4_K_M | MoE route | `0.96` | `8.9%` | not primary |
| A3B Q4_K_M | MoE FFN apply | `2.74` | `25.3%` | largest named phase |
| A3B Q4_K_M | LM head | `0.83` | `7.6%` | already high bandwidth |
| A10B Q4_K_XL | GDN front projections | `5.44` | `23.1%` | large broad bucket |
| A10B Q4_K_XL | attention mixer | `4.31` | `18.3%` | secondary long-context lane |
| A10B Q4_K_XL | MoE route | `1.35` | `5.7%` | not primary |
| A10B Q4_K_XL | MoE FFN apply | `6.91` | `29.3%` | largest named phase |
| A10B Q4_K_XL | GDN out projection | `2.32` | `9.8%` | non-trivial broad bucket |
| A10B Q4_K_XL | LM head | `1.59` | `6.8%` | tail budget |

## Interpretation

- The decode branch gate remains MoE FFN execution shape, not another generic
  prefill branch.
- Routed down has a real long-context budget on both MoE targets, but the current
  fused Q5 down path remains better than split rollback.
- Prior local Q5 down retunes are still falsified; the next credible branch must
  change work granularity, byte movement, or overlap more deeply than another
  one-kernel shape tweak.
- Attention/KV remains a measured secondary lane at `18-21%`, but it is not the
  current largest named bucket.
