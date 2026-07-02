# v0.435 Hot Kernel Max Threadgroup Hints

Goal: test the cheap compiler-hygiene hypothesis from the hardware-saturation
audits: fixed-threadgroup hot kernels should declare their true maximum thread
count so Metal does not have to assume a 1024-thread entry point.

Scope:

- `attn_v4` decode main/reduce kernels, packed-prefill kernels, and matrix
  prefill KQ/softmax/KQV kernels.
- GDN single-token and packed recurrence kernels.
- No algorithmic or host-path change; this is a compiler lowering hint and a
  documentation of the fixed dispatch invariant.

Commands:

```sh
cargo check -p qwen-llm -p qwen-cli --bin qwen-bench
cargo build --release -p qwen-cli --bin qwen-bench

cargo test --release -p qwen-llm attn_v4_matches_naive_f16kv -- --nocapture
cargo test --release -p qwen-llm gdn_step_matches_cpu -- --nocapture
cargo test --release -p qwen-llm \
  prefill_tokens_matches_single_token_loop_35b_a3b_moe -- --ignored --nocapture

target/release/qwen-bench tg \
  -m /Users/tito/models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  -n 128 --runs 5 -o json \
  > target/profiles/v0435-a3b-tg128-maxthreads-broad.json

target/release/qwen-bench pp \
  -m /Users/tito/models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  -p 512 --runs 5 -o json \
  > target/profiles/v0435-a3b-pp512-maxthreads-broad.json

QWEN_PREFILL_ATTN_MATRIX_G6=1 QWEN_PREFILL_ATTN_MATRIX_G8=1 \
  target/release/qwen-bench pp \
  -m /Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf \
  -p 512 --runs 3 -o json \
  > target/profiles/v0435-27b-pp512-g6-maxthreads-broad.json
```

Validation:

- `attn_v4_matches_naive_f16kv`: passed, all checked groups/contexts at
  `cos=1.000000`.
- `gdn_step_matches_cpu`: passed for `n_v=16,n_k=16` and `n_v=48,n_k=16`;
  max state delta `3.73e-9`.
- A3B ignored MoE prefill-vs-single gate: passed in default grouped path,
  including packed-attn threshold/full-tile/multi-tile cases.
- cx design review session `019f23fb-c5e2-7eb2-a45b-b5d2d732f8a4`.

## Results

Warmed spot checks are stable and show no default regression:

| Shape | Avg t/s | Samples | Read |
| --- | ---: | --- | --- |
| A3B `tg128` | `109.10` | `109.09,109.07,109.17,109.26,108.90` | flat / no regression |
| A3B `pp512` | `1527.64` | `1524.77,1530.16,1528.64,1522.83,1531.80` | flat / no regression |
| 27B `pp512` G6 matrix | `239.77` | `244.31,243.95,231.04` | noisy / no clear win |

Earlier same-session narrow-hint spots were consistent: A3B `tg128` `109.02`,
A3B `pp512` `1529.12`, and 27B `pp512` G6 `244.21` before broadening the decode
entry points.

## Decision

Keep the annotations: they are correctness-neutral, compile cleanly, and encode
the actual fixed threadgroup contract for these specialized kernels. Do not
claim a performance win. This falsifies the idea that `max_total_threads` alone
is a hidden broad step-function on the tested hot paths.

The fused softmax+KQV bridge over materialized matrix scores is parked after cx
review: preserving current KQV tiling would recompute softmax weights per
`head_dim/64` tile, while avoiding that would reduce y-parallelism or increase
register pressure. The serious FA2-style matrix body remains a design branch, not
a quick bridge.
