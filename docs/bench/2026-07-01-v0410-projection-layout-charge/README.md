# v0.410 Projection Layout Charge

Goal: charge the v0.407/v0.409 projection-batching estimate for explicit GPU
pack/scatter copies before moving to a full `decode-phase-batch` replay.

`decode-proj-batch` now reports three additional modes for every projection
group:

- `layout_pack`: blit-copy per-slot input rows into the batch buffer;
- `layout_scatter`: blit-copy batched projection outputs to a sink;
- `matmat_with_layout`: pack, batched matmat, then scatter in one command buffer.

The layout model is intentionally only a charge, not a production design. It can
overcharge outputs that a real batched path would keep batch-native, and it still
does not charge attention body/KV, route/topk, GDN tail, scheduler policy, or
ragged occupancy.

Commands:

```sh
target/release/qwen-bench decode-proj-batch \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --tokens 8,16 --warmup 1 --iters 3 \
  > target/profiles/v0410-a3b-decode-proj-batch-layout-s8-s16.out

uv run scripts/profile/decode_batch_upper_bound.py \
  --phase target/profiles/v0409-a3b-ctx32768-deep-phase.out \
  --proj target/profiles/v0410-a3b-decode-proj-batch-layout-s8-s16.out \
  --moe-sweep target/profiles/v0409-a3b-ctx32768-moe-batch-sweep.out \
  --projection-mode matmat_with_layout \
  > target/profiles/v0410-a3b-ctx32768-decode-batch-upper-bound-layout.tsv
```

Validation:

- `cargo fmt`
- `cargo check -p qwen-cli --bin qwen-bench`
- `cargo build --release -p qwen-cli --bin qwen-bench`
- 0.8B `decode-proj-batch` layout smoke at `S=1`
- A3B `decode-proj-batch` layout rows at `S=8/16`
- `uv run python -m py_compile scripts/profile/decode_batch_upper_bound.py`
- layout-charged upper-bound estimate
- `cx ask` review, session `019f1e94-6a65-7840-8fc1-ea3d8bd317ad`

Artifacts:

- `target/profiles/v0410-decode-proj-layout-smoke.out`
- `target/profiles/v0410-a3b-decode-proj-batch-layout-s8-s16.out`
- `target/profiles/v0410-a3b-ctx32768-decode-batch-upper-bound-layout.tsv`

## Results

A3B aggregate projection rows, in `ms/token`:

| S | Matvec seq | Matmat batch | With layout | Layout haircut | With-layout save |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `8` | `4.7551` | `2.8882` | `2.9629` | `0.0747` | `1.7922` |
| `16` | `4.7131` | `1.2456` | `1.3094` | `0.0638` | `3.4037` |

Layout-charged A3B `ctx32768` upper bound:

| S | Projection save | Routed save | Charged ms | Saved ms | Saved % | Ideal speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `8` | `1.7922` | `0.6229` | `10.1749` | `2.4151` | `19.18` | `1.2374x` |
| `16` | `3.4037` | `0.6604` | `8.5259` | `4.0641` | `32.28` | `1.4767x` |

## Decision

Explicit GPU copy layout is not the uncertainty that kills batching. The layout
haircut is only `0.06-0.08 ms/token`, or a few percent of the projection win, and
the `S=8` charged estimate remains well above the continuation gate.

The next artifact should be the actual `decode-phase-batch` replay. It must
include attention body/KV, route/topk, GDN tail, routed MoE, layout, `lm_head`,
sampling/readback, and ragged occupancy. Do not spend another cycle on isolated
layout charges unless the replay exposes a new concrete layout bottleneck.
