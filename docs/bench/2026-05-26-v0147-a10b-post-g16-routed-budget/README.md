# v0.147 A10B Post-G16 Routed-MoE Budget

Status: post-default attribution after A10B/G16 matrix attention became default.
Rows used rebuilt `qwen-bench` from `f826303`, AC power, no thermal/performance
warnings, and stable memory pressure.

## Tight Gate

The broad six-variant no-op sweep had a bad late baseline row (`328.50 t/s`), so
use this tighter repeated gate for ratios:

| Variant | Rows | Read |
| --- | ---: | --- |
| base/default G16 | `383.26`, `382.87` | baseline |
| no attention body | `389.39`, `390.08` | only `~+1.7%` |
| no routed MoE | `764.19`, `764.46` | about `2.0x` |

## Broad Sweep Clue

The broader one-block sweep is useful directionally, but not as a ratio gate
because the last baseline sagged:

| Variant | Row | Read |
| --- | ---: | --- |
| no attention body | `390.26` | matches tight gate: small |
| no routed MoE | `763.40` | matches tight gate: huge |
| no shared MoE | `402.04` | small |
| no GDN body | `446.97` | meaningful but secondary |
| no FFN | `1051.66` | broad upper bound |
| late base | `328.50` | confounded; do not ratio against it |

## Current Read

- G16 matrix default makes A10B attention-body headroom tiny at `pp512`.
- Routed MoE is now the obvious next A10B optimization frontier.
- Shared MoE is not the crack; broad GDN remains worth tracking but is secondary to
  routed `SwiGLU/down`.
- Next work should start from the clean subphase trace: `routed_swiglu` then
  `routed_down`, not another attention branch.

## Artifacts

- `v0147-a10b-pp512-post-g16-noop-tight.json`
- `v0147-a10b-pp512-post-g16-noop-budget.json`
