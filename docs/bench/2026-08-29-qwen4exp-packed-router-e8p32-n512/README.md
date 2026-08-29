# Flash-Next Strict E8P32 Router At N=512

Decision: **GO** for exact N=512. The qualified strict-router token set is now
exactly `{512, 2048}`; every other packed width remains on generic F32 routing.

## Source And Protocol

- Measured source: `0b0c468`, with the unchanged strict kernel admitted at
  N=512 only when `QWEN4EXP_PACKED_ROUTER_E8P32_STRICT_N512=1`.
- Promotion source: `0f576b0`, which removes that temporary opt-in and retains
  `QWEN4EXP_PACKED_ROUTER_E8P32_STRICT=0` as the global rollback.
- Device: Apple M4 Max with unified memory.
- Model: UD-Q3_K_XL from `unsloth/Qwen3.8-Flash-Next-GGUF`, revision
  `8bdc666649440e9bdc97e16f3f75782c98478ff5`, on the external PCIe SSD.
- Workload: the tracked 512-token natural prompt from
  `docs/bench/2026-08-29-qwen4exp-packed-natural-n512/prompt.txt`, no special
  tokens, one generated token, and exact 512-forward capacity.
- Order: separate-process A-B-A with five seconds between processes. A disabled
  the global strict router; B enabled both the global path and temporary N=512
  opt-in. Every process ran first, warm, and 128-sample profiled commands.
- Acceptance: router leaf `<=3.564675 ms/layer`, warm command GPU
  `<=1137.542876 ms`, one packed command, observer GPU ratio `0.985-1.015`, wall
  ratio `0.98-1.02`, raw coverage `0.995-1.005`, and one generated-output digest.

Cold first-pass wall is not a promotion endpoint. It includes process-local
first touch and is retained only in `results.json`.

## Correctness

Focused component gates covered strict projection at N=512, full router logits,
top-k IDs and weights, shared scale, route counts and slots, and exact generic to
strict dispatch substitution. All values were bit-exact.

One released-model replay exercised N=18 generic fallback, natural N=512, and
the existing N=2,048 qualification from one loaded session. Baseline/candidate
N=512 and N=2,048 rows retained full-vocabulary endpoint and scalar-continuation
logits, every raw persistent state tensor, QSA lengths, PLE history, and exactly
48 router substitutions with unchanged non-router topology. N=18 retained zero
substitutions.

## A-B-A Results

| Arm | Warm command GPU (ms) | Profiled command GPU (ms) | Router (ms/layer) | GPU ratio | Wall ratio |
|:---|---:|---:|---:|---:|---:|
| A1 generic | 1,148.184875 | 1,147.917208 | 3.959917 | 0.999767 | 0.999722 |
| B strict | 990.813875 | 992.874542 | 0.690375 | 1.002080 | 1.002612 |
| A2 generic | 1,149.932875 | 1,150.861792 | 3.951000 | 1.000808 | 1.001316 |

All three rows had raw timestamp coverage `1.000000`, passed observer gates, and
produced stdout SHA-256
`e530152ea80c3012dbfdb19a69e554de54aed4511184ae225fcec380233a46e9`.

## Gate Evaluation

| Metric | A/A mean | Candidate | Required | Decision |
|:---|---:|---:|---:|:---|
| router ms/layer | 3.955459 | 0.690375 | <=3.564675 | GO |
| warm command GPU ms | 1,149.058875 | 990.813875 | <=1,137.542876 | GO |

The leaf saves `3.265084 ms/layer`, or 82.55%. Across 48 routers that predicts
`156.724008 ms`; the warm command saves `158.245000 ms`, for 100.97% conversion.
GPU-equivalent throughput rises `445.58 -> 516.75 tok/s` (15.97%). The profiled
command independently saves `156.514958 ms`.

## Promotion

- Enable the existing strict-order kernel on Apple M4 Max with F32 router
  weights, `H=2560`, `E=512`, and exact N in `{512, 2048}`.
- Keep every other token width and unsupported device/dtype/geometry on generic
  F32 routing.
- Roll back both qualified widths with
  `QWEN4EXP_PACKED_ROUTER_E8P32_STRICT=0`.
- Remove the temporary N=512 opt-in; no additional kernel or scratch allocation
  remains.

Raw logs remain under
`target/profiles/qwen4exp-router-e8p32-n512-20260829/`. Their hashes are recorded
in `results.json`.

Adversarial review: `01a04e8e-057f-76b0-9fb6-3997494018dd`.
