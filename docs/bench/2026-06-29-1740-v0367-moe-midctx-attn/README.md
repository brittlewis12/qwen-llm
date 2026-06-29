# MoE Mid-Context Attention Threshold — v0.367

Clean AC-power decode context-slope packet for the group-8 A3B and group-16 A10B
attention shapes. The candidate lowers the v4 subgroup/NWG/C64 threshold for
group-8/group-16 decode from `4096` to `256` positions. Rollback:
`QWEN_ATTN_V4_SUBGROUP_MIN_POS=4096`.

## Identity

- `qwen-llm`: v0.367 candidate after `3af7c92`
- GPU: `Apple M4 Max`
- Power: AC attached
- QWEN env, default rows: none
- QWEN env, rollback rows: `QWEN_ATTN_V4_SUBGROUP_MIN_POS=4096`

## Context Slope

Rows are sequential `qwen-bench ctx-sweep --window 32` measurements, not paired
llama.cpp comparisons.

| Model | ctx | rollback t/s | default t/s | Ratio |
| --- | ---: | ---: | ---: | ---: |
| A3B Q4_K_M | `128` | `107.0` | `107.7` | `1.01x` |
| A3B Q4_K_M | `256` | `105.9` | `107.3` | `1.01x` |
| A3B Q4_K_M | `512` | `104.9` | `105.9` | `1.01x` |
| A3B Q4_K_M | `1024` | `102.5` | `105.3` | `1.03x` |
| A3B Q4_K_M | `1536` | `100.6` | `104.6` | `1.04x` |
| A3B Q4_K_M | `2048` | `98.0` | `104.1` | `1.06x` |
| A3B Q4_K_M | `3072` | `94.1` | `102.8` | `1.09x` |
| A3B Q4_K_M | `4096` | `100.7` | `101.1` | `1.00x` |
| A10B Q4_K_XL | `128` | `45.1` | `45.2` | `1.00x` |
| A10B Q4_K_XL | `256` | `45.1` | `45.7` | `1.01x` |
| A10B Q4_K_XL | `512` | `44.4` | `45.6` | `1.03x` |
| A10B Q4_K_XL | `1024` | `42.8` | `45.4` | `1.06x` |
| A10B Q4_K_XL | `1536` | `41.7` | `45.1` | `1.08x` |
| A10B Q4_K_XL | `2048` | `40.2` | `44.1` | `1.10x` |
| A10B Q4_K_XL | `3072` | `37.9` | `43.7` | `1.15x` |
| A10B Q4_K_XL | `4096` | `44.0` | `43.1` | same-path noise |

## Phase Attribution

`QWEN_PHASE_GDN_PROJ_SPLIT=1 QWEN_PHASE_MOE_FFN_SPLIT=deep qwen-bench phase
--ctx 3072` shows the win is the attention bucket, not GDN or MoE FFN:

| Model | rollback attn mixer | default attn mixer | Phase read |
| --- | ---: | ---: | --- |
| A3B Q4_K_M | `2.53 ms` | `1.67 ms` | `-34%` attention |
| A10B Q4_K_XL | `7.29 ms` | `3.47 ms` | `-52%` attention |

The GDN qkv/z split roofline also says naive GDN-front fusion is not the next
branch: A3B qkv/z are `469/424 GB/s`, and A10B qkv/z are `486/472 GB/s` in the
ctx3072 default packet. Keep GDN-front projection retreads demoted unless a new
byte-reduction mechanism appears.

## Spillover Guards

The selector shares `NWG`/`C64` helpers with some prefill paths, so `pp512` and
short `tg128` were smoke-guarded after the ctx-sweep win.

| Model | Guard | rollback | default | Read |
| --- | --- | ---: | ---: | --- |
| A3B Q4_K_M | `tg128` | `107.39 t/s` | `107.14 t/s` | flat/noise |
| A10B Q4_K_XL | `tg128` | `45.44 t/s` | `45.40 t/s` | flat/noise |
| A3B Q4_K_M | `pp512` | `1519.8 t/s` | `1513.9 t/s` | flat/noise |
| A10B Q4_K_XL | `pp512` | `336.5 t/s` | `414.5 t/s` | no regression |

## Read

The old long-only threshold left a real medium-context MoE decode valley between
`~1k` and `4k`. Defaulting the subgroup path from `ctx256` smooths that valley
without touching the short `tg128` range. The next attention work should move to
true-long KV/layout pressure rather than further threshold fiddling.
