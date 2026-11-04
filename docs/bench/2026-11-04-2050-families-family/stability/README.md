# Stability reruns

The main grid's 27B `tg128@d8192` lcpp cell had a 2.4x disagreement between
ABBA blocks, and its ratio (1.48x) therefore overstated the qwen lead. A clean
rerun of the same model, shape, depth, `--runs 2`, `--blocks 2` protocol gave:

| shape | engine | samples (t/s) | block means | mean |
|---|---|---|---:|---:|
| tg128 | lcpp ub2048 | 24.53, 25.02, 24.51, 24.26 | 24.77, 24.38 | 24.58 |
| tg128 | qwen | 25.22, 24.60, 21.89, 22.02 | 24.91, 21.95 | 23.43 |
| tg128@d8192 | lcpp ub2048 | 18.07, 17.79, 21.06, 22.45 | 17.93, 21.76 | 19.84 |
| tg128@d8192 | qwen | 19.74, 20.30, 22.45, 22.09 | 20.02, 22.27 | 21.14 |

The rerun ratio at 8K is **1.07x** (21.14 / 19.84), matching the 09-25 value
(1.07x), not the 1.48x in the main grid. The main grid cell is retained in the
JSON and `SAMPLES.md` but should not be used.

The A10B pp4096 stability rerun did not produce data: qwen-bench stopped with
`production prefill allocation changed chunk between reps`. That is a harness
guard on allocation behavior, not a benchmark result. The main-grid A10B
pp4096 number remains provisional.
