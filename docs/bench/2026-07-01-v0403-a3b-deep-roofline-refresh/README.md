# v0.403 A3B Deep Roofline Refresh

Goal: refresh the A3B long-context decode phase board after v0.400-v0.402
deep MoE/GDN/route split to avoid choosing the next structural branch from stale
or overly coarse buckets.

Command shape:

```sh
QWEN_PHASE_MOE_FFN_SPLIT=deep \
QWEN_PHASE_MOE_ROUTE_SPLIT=1 \
QWEN_PHASE_GDN_PROJ_SPLIT=1 \
QWEN_PHASE_GDN_TAIL_SPLIT=1 \
target/release/qwen-bench phase \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --ctx 8192

uv run scripts/profile/decode_phase_roofline.py \
  --phase target/profiles/v0403-a3b-ctx8192-deep-phase.out \
  --tensors target/profiles/v0403-a3b-gguf-tensors.txt \
  --metadata target/profiles/v0403-a3b-gguf-metadata.txt \
  --ctx 8192
```

Validation:

- A3B deep phase at `ctx8192`
- A3B deep phase at `ctx16384`
- `decode_phase_roofline.py` summaries for both contexts
- `cx ask` interpretation, session `019f1be9-2e89-7d13-9157-ba847a9916bd`

Artifacts:

- `target/profiles/v0403-a3b-ctx8192-deep-phase.out`
- `target/profiles/v0403-a3b-ctx16384-deep-phase.out`
- `target/profiles/v0403-a3b-ctx8192-deep-roofline.tsv`
- `target/profiles/v0403-a3b-ctx16384-deep-roofline.tsv`
- `target/profiles/v0403-a3b-gguf-tensors.txt`
- `target/profiles/v0403-a3b-gguf-metadata.txt`

## Results

| Phase | ctx8192 ms | ctx8192 pct | ctx16384 ms | ctx16384 pct | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| Attention mixer | `2.30` | `21.3%` | `2.53` | `22.8%` | largest slope term |
| GDN QKV proj | `1.14` | `10.5%` | `1.14` | `10.3%` | `469 GB/s` weight |
| GDN Z proj | `0.63` | `5.8%` | `0.63` | `5.7%` | `424 GB/s` weight |
| GDN OUT proj | `0.75` | `6.9%` | `0.76` | `6.8%` | `352-357 GB/s` |
| Route topk/shared | `0.67` | `6.2%` | `0.68` | `6.1%` | real but falsified locally |
| Routed gate/up | `1.07` | `9.9%` | `1.07` | `9.6%` | `353 GB/s` weight |
| Routed down | `0.81` | `7.5%` | `0.83` | `7.5%` | `282-289 GB/s` weight |
| LM head | `0.82` | `7.6%` | `0.81` | `7.3%` | `509-515 GB/s` weight |

Attention KV estimate:

| Ctx | Logical KV GB | Subgroup+partial GB/s | Read |
| ---: | ---: | ---: | --- |
| 8192 | `0.1678` | `296.4` | below stream shelves |
| 16384 | `0.3355` | `282.0` | below stream shelves |

## Decision

The active long-context A3B structural branch is attention body/KV traffic, not
GDN projection, LM head, route local rewrites, or MoE batching. GDN QKV/Z and LM
head are already near or above the measured stream anchor on weight bytes. Route
topk/shared is real at `~0.68 ms`, but the obvious local route variants are
already falsified. MoE routed gate/down still has moderate headroom, but the
recent batching and monolith probes constrain the easy execution-shape exits.

Next attention work must change the main body memory shape. More tile/NWG/reduce
retunes remain closed; the next bounded experiment should be an `attn-intra`
oracle or sidecar that reduces or reorganizes KV reads and clears roughly
`>=1.15x` attention-layer speedup or `>=0.30-0.40 ms` full-decode-equivalent
savings at `ctx16384` before any production rewrite.
