# v0.406 GDN Projection Batch Scaling

Goal: test whether multi-slot/layer-batched decode has a larger always-on
surface than the routed-MoE-only upper bound from v0.401. The probe extends
`gdn-proj-micro` with `--tokens` and compares repeated decode-shaped matvecs
against existing prompt-shaped matmat kernels for GDN `qkv`, `z`, and `out`
projections across all A3B GDN layers.

Command shape:

```sh
target/release/qwen-bench gdn-proj-micro \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --tokens 16 --iters 10 --warmup 3
```

Validation:

- `cargo fmt`
- `cargo check -p qwen-cli --bin qwen-bench`
- `cargo build --release -p qwen-cli --bin qwen-bench`
- A3B `gdn-proj-micro` at token counts `1`, `2`, `4`, `8`, and `16`
- `cx ask` review, session `019f1e1c-46e4-7f60-afd9-459c0526cb03`

Artifacts:

- `target/profiles/v0406-a3b-gdn-proj-micro-t1.out`
- `target/profiles/v0406-a3b-gdn-proj-micro-t2.out`
- `target/profiles/v0406-a3b-gdn-proj-micro-t4.out`
- `target/profiles/v0406-a3b-gdn-proj-micro-t8.out`
- `target/profiles/v0406-a3b-gdn-proj-micro-t16.out`

## Results

Per-token GPU time for all 30 A3B GDN layers:

| Tokens | QKV+Z matvec | QKV+Z matmat | Out matvec | Out matmat | Combined save |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `1` | `1.7865` | `6.3663` | `0.6937` | `3.6140` | negative |
| `2` | `1.7859` | `3.1884` | `0.6933` | `1.8603` | negative |
| `4` | `1.7858` | `1.5935` | `0.6845` | `0.9196` | negative/flat |
| `8` | `1.7868` | `0.7982` | `0.6522` | `0.4660` | `~1.17 ms/token` |
| `16` | `1.7872` | `0.3003` | `0.6958` | `0.2479` | `~1.93 ms/token` |

Effective GDN `qkv+z` weight throughput:

| Tokens | Matvec seq | Matmat batch | Read |
| ---: | ---: | ---: | --- |
| `4` | `449 GB/s` | `503 GB/s` | only slightly useful |
| `8` | `449 GB/s` | `1005 GB/s` | clear reuse |
| `16` | `449 GB/s` | `2671 GB/s` | strong reuse |

## Decision

This reopens multi-slot/layer-batched decode as the leading architecture branch.
The v0.401 routed-MoE-only upper bound was too narrow: A3B has about `2.49 ms`
of always-on GDN projection time at `ctx32768`, and existing matmat kernels can
reduce that to about `1.26 ms/token` at `tokens=8` or `0.55 ms/token` at
`tokens=16` in the primitive gate.

Do not jump straight to a scheduler. The next gate should be a production-shaped
decode-phase batch replay over `S={1,2,4,8,16}` that includes GDN projections,
attention front/body enough to expose layout costs, LM head, and routed MoE
projections. Promote architecture work only if `tokens=8` clears about `10%` or
`>=1.0 ms/token` phase-equivalent savings, with a strong `tokens=16` row around
`15%` or `>=1.7 ms/token`.
