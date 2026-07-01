# v0.417 Block Slice Route Trace

Goal: identify whether the v0.415 late block-slice replay cliff is caused by
continuous drift or by discrete MoE route/topk divergence.

`decode-block-slice-trace` runs baseline and replay one block at a time, then
prints per-slot route order, route-set equality, route-margin, shared-gate delta,
post-mixer `h` cosine/max-abs, and post-FFN residual `x` cosine/max-abs.

Commands:

```sh
target/release/qwen-bench decode-block-slice-trace \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --start-block 20 --blocks 4 --position 4096 --tokens 8 \
  > target/profiles/v0417b-a3b-block-slice-trace-b20-pos4096-s8.out

target/release/qwen-bench decode-block-slice-trace \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --start-block 36 --blocks 3 --position 4096 --tokens 8 \
  > target/profiles/v0417b-a3b-block-slice-trace-b36-3-pos4096-s8.out

target/release/qwen-bench decode-block-slice-trace \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --start-block 36 --blocks 4 --position 4096 --tokens 8 \
  > target/profiles/v0417b-a3b-block-slice-trace-b36-4-pos4096-s8.out

target/release/qwen-bench decode-block-slice-trace \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --start-block 37 --blocks 3 --position 0 --tokens 8 \
  > target/profiles/v0417b-a3b-block-slice-trace-b37-pos0-s8.out

target/release/qwen-bench decode-block-slice-trace \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --start-block 37 --blocks 3 --position 4096 --tokens 8 \
  > target/profiles/v0417b-a3b-block-slice-trace-b37-pos4096-s8.out
```

Validation:

- `cargo fmt`
- `cargo check -p qwen-cli --bin qwen-bench`
- `cargo build --release -p qwen-cli --bin qwen-bench`
- A3B passing and failing route-trace probes listed above
- `cx ask` review, session `019f1f66-6a6e-7f42-ae97-134062c1aea5`

## Results

| Window | Position | Route-set mismatches | First set mismatch | Min `x` cos | Max `x` abs |
| --- | ---: | ---: | --- | ---: | ---: |
| `block20..24` | `4096` | `0` | none | `0.999999667` | `0.001847` |
| `block36..39` | `4096` | `0` | none | `0.999997721` | `0.015522` |
| `block36..40` | `4096` | `1` | `block39 slot3` | `0.997359190` | `0.124566` |
| `block37..40` | `0` | `5` | `block37 slot1` | `0.958538922` | `3.800118` |
| `block37..40` | `4096` | `3` | `block37 slot1` | `0.942962219` | `0.974661` |

The first true route-set flips happen only at tiny kth-vs-next route margins:

| Window | First flip | Baseline margin | Replay margin | Route change |
| --- | --- | ---: | ---: | --- |
| `block37..40`, `pos0/4096` | `block37 slot1` | `0.000255` | `0.000194` | expert `55 -> 34` |
| `block36..40`, `pos4096` | `block39 slot3` | `0.000340` | `0.000139` | expert `229 -> 132` |

## Decision

The visible late correctness cliff is mediated by route-set instability at
near-tie topk boundaries. Small GDN replay differences stay benign in early/mid
windows and in `block36..39`, but late blocks can have near-tie topk boundaries.
When replay crosses those boundaries, the changed expert set is amplified by the
MoE FFN and subsequent attention. This does not prove attention sensitivity is
irrelevant; route-set mismatch is the tripwire and amplifier we can gate on.

Scheduler promotion should therefore stay window-specific. Late windows should
remain exact for now. Early/mid windows need more prompt/position/slot coverage
and, if promoted, a replay-side route-margin guard with rollback to exact
pre-window state. Route-order-only changes with the same expert set should not be
treated as semantic failures by themselves.
