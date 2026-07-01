# v0.419 Block Slice Margin Sweep

Goal: broaden the route-margin evidence without paying repeated model-load cost.

`decode-block-slice-margin-sweep` loads the model once, then runs summary-only
baseline-vs-replay traces across many block windows and positions. It reports
route-set mismatches, first mismatch, router-logit drift, low-margin counts, and
post-window residual drift.

Commands:

```sh
target/release/qwen-bench decode-block-slice-margin-sweep \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --tokens 8 --blocks 4 --position 0,4096 \
  > target/profiles/v0419-a3b-block-slice-margin-sweep-b4-pos0-4096-s8.out

target/release/qwen-bench decode-block-slice-margin-sweep \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --tokens 8 --blocks 4 --position 4096 \
  --start-block 24,25,26,27,28,29,30,31,32,33,34,35,36 \
  > target/profiles/v0419-a3b-block-slice-margin-sweep-late-pos4096-s8.out

target/release/qwen-bench decode-block-slice-margin-sweep \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --tokens 8 --blocks 4 --position 0 \
  --start-block 24,25,26,27,28,29,30,31,32,33,34,35,36 \
  > target/profiles/v0419-a3b-block-slice-margin-sweep-late-pos0-s8.out
```

Additional targeted row:

- `target/profiles/v0419-a3b-block-slice-margin-sweep-b37-3-pos0-s8.out`

Validation:

- `cargo fmt`
- `cargo check -p qwen-cli --bin qwen-bench`
- `cargo build --release -p qwen-cli --bin qwen-bench`
- A3B loaded-once non-overlap and late-overlap margin sweeps

## Results

Non-overlap `blocks=4` windows:

| Position | Windows | Route-set mismatch windows | First mismatches |
| ---: | ---: | ---: | --- |
| `0` | `10` | `0` | none |
| `4096` | `10` | `3` | `block31`, `block35`, `block39` |

Overlapping late `pos4096` windows show the brittle attention-ending windows:

| Window | Route-set mismatches | First set mismatch | Min replay margin | Min `x` cos |
| --- | ---: | --- | ---: | ---: |
| `block28..32` | `1` | `block31 slot1` | `0.000387` | `0.999059349` |
| `block32..36` | `1` | `block35 slot3` | `0.000312` | `0.996651330` |
| `block36..40` | `1` | `block39 slot3` | `0.000139` | `0.997359190` |

The same late-overlap shape at `pos0` has no route-set mismatches for `blocks=4`,
but the targeted `block37..40 pos0 blocks=3` row still fails hard with five
route-set mismatches. The failure is therefore not a simple absolute-block cutoff;
it depends on the exact window input, position, and route margins.

## Decision

The loaded-once sweep strengthens the dynamic-guard case and weakens a naive
static allowlist. Static policy should still keep late windows exact by default,
but any production-shaped scheduler needs replay-side margin histograms and
fallback-rate estimates. `tau=1e-3` catches the observed failing `pos4096` windows,
but also catches safe low-margin rows, so the guard is a conservative fallback
trigger rather than a correctness proof.
