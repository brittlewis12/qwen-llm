# v0.407 Decode Projection Batch Gate

Goal: test whether the v0.406 GDN batching signal survives a broader decode
projection surface before spending engineering time on a real multi-slot
scheduler. The new `decode-proj-batch` harness compares repeated decode-shaped
matvecs against existing prompt-shaped matmat kernels for GDN projections,
attention projections, dense/shared FFN projections, and `lm_head`.

Command shape:

```sh
target/release/qwen-bench decode-proj-batch \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --tokens 1,2,4,8,16 --warmup 2 --iters 5
```

Validation:

- `cargo fmt`
- `cargo check -p qwen-cli --bin qwen-bench`
- `cargo build --release -p qwen-cli --bin qwen-bench`
- 0.8B smoke with `--tokens 1,2 --warmup 0 --iters 1`
- A3B `decode-proj-batch` at `S=1/2/4/8/16`
- A10B and 27B confirmation rows at `S=8/16`
- `cx ask` design review, session `019f1e3d-d249-7201-b59a-e3fa0c3d2716`
- `cx ask` result review, session `019f1e4b-e27f-7542-a2ad-976af24958fa`

Artifacts:

- `target/profiles/v0407-decode-proj-batch-smoke.out`
- `target/profiles/v0407-a3b-decode-proj-batch.out`
- `target/profiles/v0407-a10b-decode-proj-batch-s8-s16.out`
- `target/profiles/v0407-27b-decode-proj-batch-s8-s16.out`

## Results

Aggregate one-encoder projection GPU time, in `ms/token`:

| Model | S | Matvec seq | Matmat batch | Save | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| A3B | `1` | `4.7038` | `22.4696` | negative | matmat underfill |
| A3B | `2` | `4.7233` | `11.3381` | negative | matmat underfill |
| A3B | `4` | `4.7274` | `5.7171` | negative | below crossover |
| A3B | `8` | `4.7255` | `2.8866` | `1.8389` | passes kill gate |
| A3B | `16` | `4.6600` | `1.2346` | `3.4254` | strong upside |
| A10B | `8` | `12.2232` | `6.2833` | `5.9399` | strong confirm |
| A10B | `16` | `12.1893` | `2.6360` | `9.5533` | strong confirm |
| 27B | `8` | `35.9055` | `24.6742` | `11.2313` | dense confirm |
| 27B | `16` | `35.6295` | `9.3377` | `26.2918` | dense confirm |

A3B component saves at `S=8`, in `ms/token`:

| Component | Save | Read |
| --- | ---: | --- |
| GDN `qkv+z` | `0.9877` | main win |
| GDN `out` | `0.2092` | useful |
| Attention `q/k/v` | `0.1479` | useful but smaller |
| Attention `o` | `0.0707` | small positive |
| Shared FFN `gate/up` | `-0.2882` | bad shape at S=8 |
| Shared FFN `down` | `0.2291` | useful |
| `lm_head` | `0.4841` | large always-on win |

## Decision

The branch passes the immediate projection-only kill gate. A3B needs about eight
compatible active slots before the existing matmat kernels beat decode-shaped
matvecs, but at `S=8` the projection-only gain is `1.84 ms/token`, or about
`15.6%` of the current `ctx32768` A3B phase baseline (`11.78 ms/token`). A10B
and 27B confirmation rows show the same crossover has even larger projection
headroom on larger models.

This is not yet a scheduler green light. The harness intentionally excludes
attention body/KV cost, routing/topk, routed-MoE packed replay, slot packing and
scatter, ragged occupancy, layout copies, and real traffic availability. The next
gate should be a fuller `decode-phase-batch` replay that charges those costs and
reports net phase `ms/token` at sustained `S=8/16` before architecture work.
