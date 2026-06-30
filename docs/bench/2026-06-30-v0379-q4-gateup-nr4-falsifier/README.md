# v0.379 Q4 Gate/Up NR4 Falsifier

Purpose: test a dirty routed Q4_K gate/up row-widening sidecar by changing
`NR0_Q4K` from 2 to 4. The goal was to reduce threadgroup count and improve the
routed gate/up bucket exposed by v0.378.

Dirty change, reverted after measurement:

```text
#define NR0_Q4K 4
```

## Measurements

Microbench:

| Model row | Default micro GPU | Dirty NR4 micro GPU | Read |
| --- | ---: | ---: | --- |
| A10B | `3.0399 ms` | `3.0744 ms` | regressed/noise |
| A3B | `1.5179 ms` | `1.3430 ms` | synthetic micro win |

Full phase:

| Model row | Default routed gate/up | Dirty NR4 routed gate/up | Phase read |
| --- | ---: | ---: | --- |
| A10B `ctx8192` | `3.18 ms` | `3.22 ms` | no win |
| A3B `ctx32768` | `1.07 ms` | `1.09 ms` | micro win did not transfer |

## Decision

Do not keep the sidecar. The A3B synthetic microbench improved, but the full phase
did not move and A10B was flat/regressive. This confirms the v0.378 caveat:
synthetic top-k patterns are not promotion evidence for routed gate/up.

Future gate/up work needs captured route-pattern replay or a different dataflow
signal before more row-widening variants.
