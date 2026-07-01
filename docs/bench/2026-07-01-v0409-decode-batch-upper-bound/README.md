# v0.409 Decode Batch Upper Bound

Goal: charge the v0.407 projection-batching win against a fresher A3B
`ctx32768` phase profile and a same-context routed-MoE batch sweep before building
a real scheduler or replay path.

The new `scripts/profile/decode_batch_upper_bound.py` combines measured inputs:

- deep `qwen-bench phase` rows;
- `decode-proj-batch` projection saves;
- optional `moe-batch-sweep` routed-MoE saves.

It leaves unbatched attention body/KV, route/topk/shared gate, GDN tail,
residual/norm/layout, and other unmodeled work charged at the phase baseline.

Commands:

```sh
QWEN_PHASE_GDN_PROJ_SPLIT=1 \
QWEN_PHASE_GDN_TAIL_SPLIT=1 \
QWEN_PHASE_MOE_ROUTE_SPLIT=deep \
QWEN_PHASE_MOE_FFN_SPLIT=deep \
target/release/qwen-bench phase \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --ctx 32768 \
  > target/profiles/v0409-a3b-ctx32768-deep-phase.out

target/release/qwen-bench moe-batch-sweep \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --tokens 1,2,4,8,16 \
  --route-capture-ctx 32768 \
  --route-capture-stride 1 \
  --route-capture-token-pattern ramp \
  --slot-order exact \
  --warmup 2 \
  --iters 5 \
  > target/profiles/v0409-a3b-ctx32768-moe-batch-sweep.out

uv run scripts/profile/decode_batch_upper_bound.py \
  --phase target/profiles/v0409-a3b-ctx32768-deep-phase.out \
  --proj target/profiles/v0407-a3b-decode-proj-batch.out \
  --moe-sweep target/profiles/v0409-a3b-ctx32768-moe-batch-sweep.out \
  > target/profiles/v0409-a3b-ctx32768-decode-batch-upper-bound.tsv
```

Validation:

- `uv run python -m py_compile scripts/profile/decode_batch_upper_bound.py`
- `uv run scripts/profile/decode_batch_upper_bound.py --help`
- A3B `ctx32768` deep phase profile
- A3B `ctx32768` routed-MoE batch sweep
- A3B charged upper-bound estimate
- `cx ask` review, session `019f1e7b-fd47-7aa2-a7be-db3cff7d32c4`

Artifacts:

- `target/profiles/v0409-a3b-ctx32768-deep-phase.out`
- `target/profiles/v0409-a3b-ctx32768-moe-batch-sweep.out`
- `target/profiles/v0409-a3b-ctx32768-decode-batch-upper-bound.tsv`

## Results

Fresh A3B `ctx32768` deep split phase:

| Phase | ms |
| --- | ---: |
| Phase sum | `12.59` |
| GDN `qkv/z/out` | `2.50` |
| Attention mixer | `3.30` |
| Route logits/topk/shared gate | `1.74` |
| Routed gate/up/down | `1.89` |
| Shared gate/up/down | `0.74` |
| `lm_head` | `0.81` |

A3B `ctx32768` routed-MoE exact/ramp replay:

| S | Combined ms/token | Read |
| ---: | ---: | --- |
| `1` | `1.7971` | baseline |
| `2` | `1.5082` | useful |
| `4` | `1.2975` | useful |
| `8` | `1.2056` | useful |
| `16` | `1.1681` | small extra |

Charged estimate:

| S | Projection save | Routed save | Charged ms | Saved ms | Saved % | Ideal speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1` | `-17.7658` | `0.0314` | `30.3244` | `-17.7344` | `-140.86` | `0.4152x` |
| `2` | `-6.6147` | `0.3203` | `18.8844` | `-6.2944` | `-50.00` | `0.6667x` |
| `4` | `-0.9898` | `0.5310` | `13.0488` | `-0.4588` | `-3.64` | `0.9648x` |
| `8` | `1.8389` | `0.6229` | `10.1282` | `2.4618` | `19.55` | `1.2431x` |
| `16` | `3.4254` | `0.6604` | `8.5042` | `4.0858` | `32.45` | `1.4804x` |

## Decision

The charged upper bound strengthens the case for a real `decode-phase-batch`
replay. At `S=8`, the estimate still saves `2.46 ms/token` after charging all
unmodeled phase work at baseline and replacing only measured projection and
routed-MoE components. This is well above the `>=1.0 ms/token` continuation gate.

Do not build the scheduler yet. The next artifact should be an actual
production-shaped replay over `S={1,2,4,8,16}` with attention body/KV, route/topk,
routed MoE, GDN tail, packing/scatter, layout, logits/sampling, and ragged
occupancy visible. Promote scheduler work only if `S=8` preserves at least
`>=10%` and `>=1.0 ms/token` under realistic occupancy.
