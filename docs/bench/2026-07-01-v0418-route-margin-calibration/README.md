# v0.418 Route Margin Calibration

Goal: quantify the router-logit perturbation and replay-side route-margin signal
needed for a possible block-slice replay acceptance guard.

`decode-block-slice-trace` now also reports exact-vs-replay router-logit max/rms
delta, minimum baseline/replay kth-vs-next route margin, and replay-margin counts
below `1e-3` and `5e-3`.

Commands are the v0.417 trace probes rerun with the extended summary fields:

- `target/profiles/v0418b-a3b-block-slice-trace-b20-pos4096-s8.out`
- `target/profiles/v0418b-a3b-block-slice-trace-b36-3-pos4096-s8.out`
- `target/profiles/v0418b-a3b-block-slice-trace-b36-4-pos4096-s8.out`
- `target/profiles/v0418b-a3b-block-slice-trace-b37-pos0-s8.out`
- `target/profiles/v0418b-a3b-block-slice-trace-b37-pos4096-s8.out`

Validation:

- `cargo fmt`
- `cargo check -p qwen-cli --bin qwen-bench`
- `cargo build --release -p qwen-cli --bin qwen-bench`
- A3B route-logit delta and margin summary probes listed above

## Results

| Window | Position | Set mismatches | Max logit abs | Min replay margin | Replay margin <1e-3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `block20..24` | `4096` | `0` | `0.003879` | `0.000854` | `2` |
| `block36..39` | `4096` | `0` | `0.004166` | `0.002066` | `0` |
| `block36..40` | `4096` | `1` | `0.005780` | `0.000139` | `3` |
| `block37..40` | `0` | `5` | `0.694589` | `0.000194` | `3` |
| `block37..40` | `4096` | `3` | `0.935274` | `0.000116` | `3` |

The first route-set flips are caught by replay margins below `1e-3`, but a
passing early/mid control also has two low-margin rows below `1e-3`. That makes a
margin guard plausible but conservative: it can catch the observed cliffs, but it
will also fallback some safe work unless calibrated on more prompts/windows.

## Decision

Do not promote a dynamic margin guard yet. The next evidence should be a broader
margin histogram over more early/mid blocks, prompts, positions, and slot counts,
with fallback-rate estimates for candidate thresholds such as `1e-3` and `5e-3`.
Until then, the safe policy is static early/mid allowlist plus late exact-only.
