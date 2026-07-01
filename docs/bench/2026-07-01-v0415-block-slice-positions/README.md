# v0.415 Block Slice Positions

Goal: test whether the v0.414 block-slice replay survives nonzero attention
positions and different block windows.

`decode-block-slice-replay` now sizes benchmark sessions to `position + 32` KV
capacity so synthetic nonzero-position attention slices can run without KV-cache
OOB limits. The benchmark still uses zero-filled past KV, so this is a position
and bandwidth stressor, not a realistic prefilled-context oracle.

Commands:

```sh
target/release/qwen-bench decode-block-slice-replay \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --blocks 4 --position 4096 --tokens 8 --warmup 1 --iters 3 \
  > target/profiles/v0415-a3b-block-slice-replay-b4-pos4096-s8.out

target/release/qwen-bench decode-block-slice-replay \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --blocks 4 --position 16384 --tokens 8 --warmup 1 --iters 2 \
  > target/profiles/v0415-a3b-block-slice-replay-b4-pos16384-s8.out

target/release/qwen-bench decode-block-slice-replay \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --start-block 20 --blocks 4 --position 4096 --tokens 8 --warmup 1 --iters 3 \
  > target/profiles/v0415-a3b-block-slice-replay-b20-pos4096-s8.out
```

Additional late-window probes:

- `target/profiles/v0415-a3b-block-slice-replay-b36-one-pos4096-s8.out`
- `target/profiles/v0415-a3b-block-slice-replay-b36-two-pos4096-s8.out`
- `target/profiles/v0415-a3b-block-slice-replay-b36-three-pos4096-s8.out`
- `target/profiles/v0415-a3b-block-slice-replay-b36-pos4096-s8.out`
- `target/profiles/v0415-a3b-block-slice-replay-b37-three-pos0-s8.out`
- `target/profiles/v0415-a3b-block-slice-replay-b37-three-pos4096-s8.out`
- `target/profiles/v0415-a3b-block-slice-replay-b39-one-pos0-s8.out`

Validation:

- `cargo fmt`
- `cargo check -p qwen-cli --bin qwen-bench`
- `cargo build --release -p qwen-cli --bin qwen-bench`
- Early and mid A3B block-slice position probes
- Late-window correctness probes, including negative rows
- `cx ask` review, session `019f1eea-0af8-70e0-877e-d77e5efae969`

## Results

Early and mid windows remain positive:

| Window | Position | S | Correctness | Baseline | Replay | Save | Save % |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `block0..4` | `4096` | `8` | `min_cos=0.999999427` | `0.9434` | `0.7912` | `0.1522` | `16.1` |
| `block0..4` | `16384` | `8` | `min_cos=0.999999463` | `1.0431` | `0.8892` | `0.1539` | `14.8` |
| `block20..24` | `4096` | `8` | `min_cos=0.999999667` | `0.8853` | `0.7478` | `0.1375` | `15.5` |

Late windows expose a correctness cliff:

| Window | Position | Result |
| --- | ---: | --- |
| `block36..37` | `4096` | passes, `min_cos=0.999999806`, save `0.0183` |
| `block36..38` | `4096` | passes, `min_cos=0.999999200`, save `0.0650` |
| `block36..39` | `4096` | drift grows, `min_cos=0.999997721`, save `0.2430` |
| `block36..40` | `4096` | fails, `min_cos=0.997359190`, `max_abs=0.124566` |
| `block37..40` | `0` | fails, `min_cos=0.962250736`, `max_abs=3.800118` |
| `block37..40` | `4096` | fails, `min_cos=0.942962219`, `max_abs=0.974661` |
| `block39` only | `0` | exact, `min_cos=1.000000000` |

## Decision

This is not a global kill for GDN replay. It is a hard quarantine for late
GDN-before-attention windows until route/topk and layer-boundary drift are
understood.

Early and mid windows remain promising at nonzero and long synthetic positions.
The late failure is not explained by KV capacity or attention replay alone:
position `0` also fails, and pure attention block 39 is exact. The likely trigger
is small GDN replay numerical drift crossing a late nonlinear boundary, probably
MoE route/topk or attention sensitivity.

Next diagnostic: fingerprint route/topk ids, route margins, and per-block
boundary deltas for failing late windows against passing early/mid controls.
