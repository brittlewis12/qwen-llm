# v0.420 Blocks=2 Margin Sweep

Goal: test whether smaller replay windows reduce route-set instability enough to
be a viable late-window policy.

Command:

```sh
target/release/qwen-bench decode-block-slice-margin-sweep \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --tokens 8 --blocks 2 --position 0,4096 \
  > target/profiles/v0420-a3b-block-slice-margin-sweep-b2-pos0-4096-s8.out
```

Validation:

- A3B loaded-once `blocks=2` margin sweep at `pos0` and `pos4096`

## Results

| Position | Windows | Route-set mismatch windows | Failing window | Min `x` cos |
| ---: | ---: | ---: | --- | ---: |
| `0` | `20` | `1` | `block38..40` | `0.985815558` |
| `4096` | `20` | `0` | none | `0.999997144` |

The `blocks=2` shape removes the `pos4096` route-set flips seen with `blocks=4`,
but it does not make late replay universally safe: `block38..40` at `pos0` still
flips one route set and drops residual cosine to `0.985815558`.

## Decision

Shorter windows are a promising mitigation, not a correctness policy. If replay
is promoted, late-region windows should use small windows only with the same
route-margin guard and exact fallback. The next useful experiment is not another
synthetic sweep; it is fallback-rate and savings accounting on real prompt slot
sets, because smaller windows trade stability for less batching amortization.
